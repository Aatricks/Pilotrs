//! Fixed-wing terrain avoidance: a real-world-style safety layer that rewrites
//! the autopilot's altitude/course setpoint to keep clear of the ground. It is
//! the simulator's analogue of a **TAWS** (Terrain Awareness and Warning System)
//! with terrain-following and terrain-avoidance logic:
//!
//! 1. **Forward-looking climb-to-clear.** Each control gate it scans the terrain
//!    along the projected great-circle ground track and computes a *terrain
//!    clearance floor*. Crucially the floor is **sized by climb capability**, not
//!    a fixed time: to clear a point `d` metres ahead at elevation `t`, the
//!    aircraft must *already* be at `t + clearance − gradient·d`. The floor is
//!    the worst (highest) such requirement over the scan, so it rises at exactly
//!    the max climb rate as a hill approaches — a rate-limited altitude command
//!    tracks it and clears the ridge.
//! 2. **Lateral turn-away.** When flying the route course would run into terrain
//!    that cannot be out-climbed, it commits to a side and steers to the nearest
//!    clear gap, commanding that gap as an **absolute** heading (not a bias on
//!    the route course — in route mode the guidance vector field would otherwise
//!    grow its cross-track correction to cancel a bias and drag the nose back
//!    into the peak). This is what makes it *avoidance* rather than mere
//!    *following*: it steers around peaks too tall to out-climb (the world has
//!    ~820 m peaks the slow Aerosonde cannot clear).
//! 3. **GPWS reactive backstop.** If the height above ground drops below a hard
//!    minimum, it forces a max-climb pull-up regardless of the route — a
//!    last-resort net for anything the look-ahead missed.
//!
//! The layer is **opt-in**: it runs only when a [`TerrainHeight`] oracle is
//! injected into the sim (the viewer supplies the real procedural terrain; tests
//! supply an analytic one). With no terrain it is inert, so the headless flight
//! path is unchanged. All maths is pure `f64` + the terrain query — no RNG, no
//! wall clock — so runs stay bit-for-bit reproducible.

use core::f64::consts::PI;
use fsim_control::FixedWingSetpoint;
use fsim_core::planet::{altitude_of, gc_advance, gc_normal, ned_basis};
use fsim_core::{Real, State13, Vec3};

/// Terrain elevation oracle the autopilot avoids. `dir` is a planet-centered
/// (PCI) direction; the return is the surface elevation \[m\] above the sea-level
/// datum at that direction — the **same datum** as
/// [`planet::altitude_of`](fsim_core::planet::altitude_of), so the height above
/// ground is `altitude_of(p) − elevation(p̂)`.
///
/// `Send + Sync` so the threaded [`FwEngine`](crate::FwEngine) can hold one;
/// `Debug` so it composes with the `Debug`-derived config it lives in.
pub trait TerrainHeight: Send + Sync + core::fmt::Debug {
    /// Surface elevation \[m\] above the datum at PCI direction `dir`.
    fn elevation(&self, dir: Vec3) -> Real;
}

/// Tuning for [`TerrainAvoid`]. Defaults are sized for the Aerosonde (25 m/s,
/// slow climb): it climbs over modest hills and steers around anything bigger.
#[derive(Debug, Clone, Copy)]
pub struct TerrainAvoidConfig {
    /// Target height-above-ground margin to hold over terrain ahead \[m\].
    pub clearance: Real,
    /// Floor on the look-ahead scan distance \[m\] (used at low speed).
    pub lookahead_min: Real,
    /// Look-ahead horizon \[s\]: scan distance `D = max(lookahead_min, v·this)`.
    pub lookahead_time: Real,
    /// Sample spacing along the scanned track \[m\] (fine enough not to step over
    /// a thin ridge).
    pub scan_step: Real,
    /// Achievable climb gradient (metres climbed per metre of ground). Sizes the
    /// clearance floor; set **conservatively** (below the airframe's real best
    /// climb) so the floor demands the climb early enough to make it.
    pub max_climb_gradient: Real,
    /// Hard cap on the commanded climb rate \[m/s\] (rarely binds for a slow UAV).
    pub climb_rate_max: Real,
    /// Rate at which the command eases back down toward the route once terrain
    /// falls away \[m/s\] (gentle — a descent is never urgent).
    pub descent_rate: Real,
    /// Height-above-ground \[m\] below which the GPWS backstop forces a pull-up.
    pub hard_min_agl: Real,
    /// Enable lateral turn-away from un-climbable terrain.
    pub lateral: bool,
    /// Half-angle of the left/right probe corridors \[rad\] (also the heading
    /// step the gap search sweeps in).
    pub fan_angle: Real,
    /// Fallback course offset \[rad\] used if no clear gap is found within the
    /// committed side (a firm turn while the GPWS climb backstop takes over).
    pub turn_max: Real,
}

impl Default for TerrainAvoidConfig {
    fn default() -> Self {
        Self {
            clearance: 80.0,
            lookahead_min: 500.0,
            // Look well ahead so the climb starts early: at 25 m/s this is a
            // ~1.5 km horizon, enough to climb over a peak ~hundreds of m above
            // cruise before reaching it (rather than discovering it too late and
            // having to turn away from terrain that was actually out-climbable).
            lookahead_time: 60.0,
            scan_step: 30.0,
            // The Aerosonde sustains ~0.3 climb gradient at low altitude. 0.25 is
            // a working value close to that (so the rate-limited command keeps
            // pace with the gradient-sized floor and climbable hills are *climbed*
            // rather than needlessly steered around) with margin for thinner air.
            max_climb_gradient: 0.25,
            climb_rate_max: 8.0,
            descent_rate: 3.0,
            hard_min_agl: 25.0,
            lateral: true,
            fan_angle: 0.6,
            turn_max: 0.8,
        }
    }
}

/// Only engage/disengage lateral avoidance once a corridor floor is this far
/// above the current altitude (hysteresis against chatter) \[m\].
const LATERAL_HYST: Real = 5.0;

/// The stateful terrain-avoidance layer. Holds the rate-limited altitude command
/// and the latest warning state between gates.
#[derive(Debug, Clone)]
pub struct TerrainAvoid {
    cfg: TerrainAvoidConfig,
    /// Rate-limited commanded altitude \[m\]; `None` until the first gate, then
    /// seeded from the incoming route altitude.
    cmd_alt: Option<Real>,
    /// Latched lateral escape direction (`+1` = steer right, `−1` = left) while
    /// turning around un-climbable terrain; `None` when not avoiding laterally.
    /// Latching gives the turn *commitment*: once committed it holds the turn
    /// until both ahead and the inside (obstacle) side are clear, instead of
    /// releasing the moment the obstacle leaves the centre corridor and snapping
    /// back into it.
    turn_dir: Option<Real>,
    /// True when the layer is actively raising the altitude or steering (HUD).
    warn: bool,
}

impl TerrainAvoid {
    pub fn new(cfg: TerrainAvoidConfig) -> Self {
        Self {
            cfg,
            cmd_alt: None,
            turn_dir: None,
            warn: false,
        }
    }

    /// True when the layer is actively avoiding terrain (climbing, steering, or
    /// in a GPWS pull-up) — drives the HUD caution.
    pub fn warn(&self) -> bool {
        self.warn
    }

    /// Forget the rate-limited command and any latched turn — call on a mode
    /// change (new route / setpoint / manual) so avoidance restarts cleanly.
    pub fn reset(&mut self) {
        self.cmd_alt = None;
        self.turn_dir = None;
        self.warn = false;
    }

    /// Rewrite `sp` to keep the aircraft clear of `terrain`. `truth` is the PCI
    /// truth state; `dt` is the control-gate period. Returns the adjusted
    /// setpoint (the airspeed is never touched).
    pub fn adjust(
        &mut self,
        truth: &State13,
        mut sp: FixedWingSetpoint,
        terrain: &dyn TerrainHeight,
        dt: Real,
    ) -> FixedWingSetpoint {
        let p = truth.position;
        let rho = p.norm();
        if rho < 1.0 {
            // Degenerate (near the core): nothing sensible to do.
            return sp;
        }
        let phat = p / rho;
        let alt = altitude_of(p);
        let ground = terrain.elevation(phat);
        let agl = alt - ground;

        let route_alt = sp.altitude;
        let mut cmd = self.cmd_alt.unwrap_or(route_alt);

        // Forward tangent from the velocity, projected into the local tangent
        // plane. Below ~1 m/s of ground track there is no heading to look down,
        // so the look-ahead/lateral logic is skipped (the GPWS net still runs).
        let v = truth.velocity;
        let v_tan_vec = v - phat * v.dot(&phat);
        let v_tan = v_tan_vec.norm();

        let (north, east, _down) = ned_basis(p);
        // Default floor: just the local clearance at the current position.
        let mut effective_floor = ground + self.cfg.clearance;
        let mut warn = false;

        if v_tan > 1.0 {
            let vt = v_tan_vec / v_tan;
            let base_course = vt.dot(&east).atan2(vt.dot(&north));
            let d_max = self.cfg.lookahead_min.max(self.cfg.lookahead_time * v_tan);

            let floor_c = self.corridor_floor(terrain, phat, &north, &east, base_course, d_max);
            effective_floor = floor_c;

            // Lateral turn-away. The trigger and the *commitment* key off the
            // **route-course floor**: would flying the route course from here run
            // into terrain we cannot climb over? While it would, steer to the
            // nearest clear gap on a committed side — and crucially command that
            // gap as an *absolute* heading. Merely biasing the route course fails
            // in route mode: the guidance vector field grows its cross-track
            // correction to cancel the bias and drags the nose back into the peak.
            // Overriding with an absolute heading takes guidance out of the vote
            // while escaping; release (resume the route) only once the route ahead
            // is genuinely clear — i.e. the aircraft has worked far enough to the
            // side to pass the obstacle.
            if self.cfg.lateral {
                let route_course = sp.course;
                let route_floor =
                    self.corridor_floor(terrain, phat, &north, &east, route_course, d_max);
                if route_floor > alt + LATERAL_HYST {
                    // Commit to a turn side once (toward the more open one; ties go
                    // right), then hold it so the escape doesn't flip-flop.
                    if self.turn_dir.is_none() {
                        let floor_l = self.corridor_floor(
                            terrain,
                            phat,
                            &north,
                            &east,
                            base_course - self.cfg.fan_angle,
                            d_max,
                        );
                        let floor_r = self.corridor_floor(
                            terrain,
                            phat,
                            &north,
                            &east,
                            base_course + self.cfg.fan_angle,
                            d_max,
                        );
                        self.turn_dir = Some(if floor_r <= floor_l { 1.0 } else { -1.0 });
                    }
                    let dir = self.turn_dir.unwrap_or(1.0);
                    sp.course = self.escape_course(terrain, phat, &north, &east, route_course, alt, d_max, dir);
                    warn = true;
                } else {
                    self.turn_dir = None; // route ahead is clear — resume it
                }
            }

            if effective_floor > alt + 1.0 {
                warn = true;
            }
        }

        // Never command below what the route asked for.
        let mut target = route_alt.max(effective_floor);

        // GPWS reactive backstop: too close to the ground → force a climb that
        // regains clearance, regardless of the route.
        if agl < self.cfg.hard_min_agl {
            target = target.max(alt + self.cfg.clearance);
            warn = true;
        }

        // Rate-limit the command: climb fast (at the assumed gradient·speed, so it
        // keeps pace with the rising floor), descend gently.
        let climb_rate = (self.cfg.max_climb_gradient * v_tan)
            .min(self.cfg.climb_rate_max)
            .max(0.5);
        cmd = rate_limit(cmd, target, climb_rate * dt, self.cfg.descent_rate * dt);

        self.cmd_alt = Some(cmd);
        self.warn = warn;
        sp.altitude = cmd;
        sp
    }

    /// The clearance floor \[m\] along the great-circle corridor leaving `phat`
    /// on NED `course`: the highest "altitude you must already hold" over the
    /// scan, climbing at `max_climb_gradient`. Includes the immediate clearance
    /// at the current position.
    fn corridor_floor(
        &self,
        terrain: &dyn TerrainHeight,
        phat: Vec3,
        north: &Vec3,
        east: &Vec3,
        course: Real,
        d_max: Real,
    ) -> Real {
        let t_hat = north * course.cos() + east * course.sin();
        let n = gc_normal(phat, phat + t_hat);
        let mut floor = terrain.elevation(phat) + self.cfg.clearance;
        let mut d = self.cfg.scan_step;
        while d <= d_max + 1e-6 {
            let dir = gc_advance(phat, n, d);
            let need = terrain.elevation(dir) + self.cfg.clearance - self.cfg.max_climb_gradient * d;
            if need > floor {
                floor = need;
            }
            d += self.cfg.scan_step;
        }
        floor
    }

    /// The absolute escape heading: sweep outward from `route_course` on the
    /// committed side (`dir`) for the nearest corridor whose floor is clear
    /// (`≤ alt`), and steer there. If none is clear within a half-turn, steer
    /// toward the least-blocked heading found (a firm turn while the climb / GPWS
    /// backstop works). Returns a heading in `(−π, π]`.
    #[allow(clippy::too_many_arguments)]
    fn escape_course(
        &self,
        terrain: &dyn TerrainHeight,
        phat: Vec3,
        north: &Vec3,
        east: &Vec3,
        route_course: Real,
        alt: Real,
        d_max: Real,
        dir: Real,
    ) -> Real {
        let step = (self.cfg.fan_angle * 0.5).max(0.1);
        let mut best_course = wrap_pi(route_course + dir * self.cfg.turn_max);
        let mut best_floor = Real::INFINITY;
        let mut a = step;
        while a <= PI {
            let h = route_course + dir * a;
            let f = self.corridor_floor(terrain, phat, north, east, h, d_max);
            if f <= alt + LATERAL_HYST {
                return wrap_pi(h); // first clear gap nearest the route
            }
            if f < best_floor {
                best_floor = f;
                best_course = wrap_pi(h);
            }
            a += step;
        }
        best_course
    }
}

/// Move `cur` toward `target`, rising at most `up` and falling at most `down`.
fn rate_limit(cur: Real, target: Real, up: Real, down: Real) -> Real {
    if target > cur {
        (cur + up).min(target)
    } else {
        (cur - down).max(target)
    }
}

/// Wrap an angle to `(−π, π]`.
fn wrap_pi(x: Real) -> Real {
    x.sin().atan2(x.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::planet::{geodetic_to_pci, PLANET_RADIUS};
    use nalgebra::UnitQuaternion;

    /// Flat terrain at a fixed elevation — the trivial oracle.
    #[derive(Debug)]
    struct Flat(Real);
    impl TerrainHeight for Flat {
        fn elevation(&self, _dir: Vec3) -> Real {
            self.0
        }
    }

    /// A single Gaussian hill centred on PCI direction `center`, peaking `peak`
    /// metres above a `base` plain with angular (arc) width `sigma` metres.
    #[derive(Debug)]
    struct Hill {
        center: Vec3,
        peak: Real,
        sigma: Real,
        base: Real,
    }
    impl TerrainHeight for Hill {
        fn elevation(&self, dir: Vec3) -> Real {
            let c = dir.normalize().dot(&self.center.normalize()).clamp(-1.0, 1.0);
            let arc = c.acos() * PLANET_RADIUS;
            self.base + self.peak * (-(arc * arc) / (2.0 * self.sigma * self.sigma)).exp()
        }
    }

    /// Truth state at a geographic point, flying `course` at `speed`.
    fn state_at(lat: Real, lon: Real, alt: Real, course: Real, speed: Real) -> State13 {
        let p = geodetic_to_pci(lat, lon, alt);
        let (n, e, _d) = ned_basis(p);
        let t = n * course.cos() + e * course.sin();
        State13 {
            position: p,
            velocity: t * speed,
            attitude: UnitQuaternion::identity(),
            angular_rate: Vec3::zeros(),
        }
    }

    fn sp(alt: Real, course: Real) -> FixedWingSetpoint {
        FixedWingSetpoint {
            airspeed: 25.0,
            altitude: alt,
            course,
        }
    }

    fn cfg() -> TerrainAvoidConfig {
        TerrainAvoidConfig::default()
    }

    // Over flat low ground with a high route altitude, the setpoint passes
    // through unchanged (no spurious climb, course untouched).
    #[test]
    fn flat_terrain_passes_through() {
        let mut ta = TerrainAvoid::new(cfg());
        let s = state_at(0.0, 0.0, 300.0, 0.0, 25.0);
        let out = ta.adjust(&s, sp(300.0, 0.0), &Flat(-5.0), 0.01);
        assert!((out.altitude - 300.0).abs() < 1e-6, "alt drifted: {}", out.altitude);
        assert!((out.course - 0.0).abs() < 1e-9, "course drifted: {}", out.course);
        assert!(!ta.warn());
    }

    // A climbable hill ahead raises the commanded altitude (climb-to-clear), with
    // lateral steering off so we isolate the floor logic.
    #[test]
    fn floor_raises_setpoint_over_hill() {
        let mut c = cfg();
        c.lateral = false;
        let mut ta = TerrainAvoid::new(c);
        // Spawn at home, heading North at 100 m. Hill ~400 m North, peaking to
        // +150 m elevation (climbable within the look-ahead).
        let center = geodetic_to_pci(400.0 / PLANET_RADIUS, 0.0, 0.0);
        let hill = Hill {
            center,
            peak: 150.0,
            sigma: 150.0,
            base: -5.0,
        };
        let s = state_at(0.0, 0.0, 100.0, 0.0, 25.0);
        // The command rate-limits upward toward the (raised) terrain floor; over
        // ~20 s of gates it climbs well clear of the route altitude.
        let mut out = sp(100.0, 0.0);
        for _ in 0..2000 {
            out = ta.adjust(&s, sp(100.0, 0.0), &hill, 0.01);
        }
        assert!(out.altitude > 120.0, "should climb for the hill: {}", out.altitude);
        assert!(ta.warn(), "should flag a terrain warning");
    }

    // The altitude command is rate-limited: it never jumps more than the climb
    // rate per gate even facing a wall.
    #[test]
    fn command_is_rate_limited() {
        let mut ta = TerrainAvoid::new(cfg());
        let center = geodetic_to_pci(300.0 / PLANET_RADIUS, 0.0, 0.0);
        let wall = Hill {
            center,
            peak: 600.0,
            sigma: 120.0,
            base: -5.0,
        };
        let s = state_at(0.0, 0.0, 100.0, 0.0, 25.0);
        let dt = 0.01;
        let max_step = cfg().max_climb_gradient * 25.0 * dt + 1e-9; // gradient·v·dt
        let mut prev = ta.adjust(&s, sp(100.0, 0.0), &wall, dt).altitude;
        for _ in 0..100 {
            let now = ta.adjust(&s, sp(100.0, 0.0), &wall, dt).altitude;
            assert!(
                now - prev <= max_step,
                "altitude jumped {} > {} per gate",
                now - prev,
                max_step
            );
            prev = now;
        }
    }

    // The GPWS backstop forces a climb when the height above ground is below the
    // hard minimum, even over flat ground with a low route.
    #[test]
    fn gpws_forces_climb_when_low() {
        let mut ta = TerrainAvoid::new(cfg());
        // 10 m above flat ground (below the 25 m hard floor), route wants to stay.
        let s = state_at(0.0, 0.0, 10.0, 0.0, 25.0);
        let out = ta.adjust(&s, sp(10.0, 0.0), &Flat(0.0), 0.01);
        assert!(out.altitude > 10.0, "GPWS should command a climb: {}", out.altitude);
        assert!(ta.warn());
    }

    // A tall un-climbable hill offset to the LEFT of the track makes the layer
    // steer RIGHT (commanded course increases) toward the lower ground.
    #[test]
    fn lateral_steers_toward_lower_side() {
        let mut ta = TerrainAvoid::new(cfg());
        // Heading North from home; hill centre ~300 m ahead but offset West (left).
        let center = geodetic_to_pci(300.0 / PLANET_RADIUS, -250.0 / PLANET_RADIUS, 0.0);
        let hill = Hill {
            center,
            peak: 800.0,
            sigma: 180.0,
            base: -5.0,
        };
        let s = state_at(0.0, 0.0, 100.0, 0.0, 25.0);
        let out = ta.adjust(&s, sp(100.0, 0.0), &hill, 0.01);
        assert!(
            out.course > 0.05,
            "should steer right (course>0) away from the left hill: {}",
            out.course
        );
        assert!(ta.warn());
    }

    // Determinism: identical inputs give identical output, gate for gate.
    #[test]
    fn is_deterministic() {
        let run = || {
            let mut ta = TerrainAvoid::new(cfg());
            let center = geodetic_to_pci(350.0 / PLANET_RADIUS, 0.0, 0.0);
            let hill = Hill {
                center,
                peak: 300.0,
                sigma: 150.0,
                base: -5.0,
            };
            let s = state_at(0.0, 0.0, 120.0, 0.0, 25.0);
            let mut last = sp(120.0, 0.0);
            for _ in 0..200 {
                last = ta.adjust(&s, sp(120.0, 0.0), &hill, 0.01);
            }
            (last.altitude, last.course)
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
    }
}
