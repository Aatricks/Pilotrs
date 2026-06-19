//! A thin fixed-wing simulation loop (M6). It reuses only the airframe-agnostic
//! parts of the stack — `State13`, the `Rk4` integrator, the shared
//! `rigid_body_deriv`, and the fixed-wing aero `Wrench` — wired to the fixed-wing
//! autopilot. The quad scheduler is welded to `CtrlCmd` + a mixer + motors, so a
//! fixed-wing (four surfaces, body-x thrust) gets its own ~40-line loop.
//!
//! For M6 the autopilot flies on **truth** (perfect feedback), exactly as the
//! quad's M1 did before the M2/M3 estimators were added; swapping in sensors +
//! the INS is the one-line `est` change deferred to a future milestone.

use crate::fw_guidance::{FwGuidance, FwGuidanceConfig};
use crate::guidance::Waypoint;
use fsim_control::{FixedWingAutopilot, FixedWingConfig, FixedWingController, FixedWingSetpoint};
use fsim_core::{planet, EstState, FixedWingControls, Real, State13, Tick, Vec3, DEFAULT_DT};
use fsim_dynamics::{fixedwing_wrench, rigid_body_deriv, trim, FixedWingParams, Integrator, Rk4};

/// One logged fixed-wing sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwSample {
    pub t: Real,
    pub truth: State13,
    pub controls: FixedWingControls,
    pub setpoint: FixedWingSetpoint,
}

/// Everything needed to build a [`FwSim`].
#[derive(Debug, Clone)]
pub struct FwSimConfig {
    pub params: FixedWingParams,
    pub autopilot: FixedWingConfig,
    pub dt: Real,
    pub control_rate: Real,
    pub initial: State13,
    pub setpoint: FixedWingSetpoint,
}

impl FwSimConfig {
    /// Aerosonde trimmed for 25 m/s level cruise at 100 m over the **home**
    /// point, heading North — spawned on the sphere (M7).
    pub fn aerosonde_cruise() -> Self {
        Self::aerosonde_at(100.0)
    }

    /// Aerosonde 25 m/s level cruise spawned at altitude `alt` \[m\] over the
    /// home point. The trim is solved in a flat local frame, then *placed on the
    /// sphere*: its position becomes the home PCI point at `alt`, and its
    /// velocity/attitude are rotated from local NED into PCI, so the aircraft
    /// begins in level cruise tangent to the planet, flying North.
    pub fn aerosonde_at(alt: Real) -> Self {
        let params = FixedWingParams::aerosonde();
        let tr = trim(&params, 25.0, 0.0).expect("Aerosonde 25 m/s level trim converges");
        let mut autopilot = FixedWingConfig::aerosonde();
        autopilot.trim_throttle = tr.controls.throttle;
        let initial = place_on_sphere(&tr.state, planet::home_pci(alt));
        Self {
            params,
            autopilot,
            dt: DEFAULT_DT,
            control_rate: 100.0,
            initial,
            setpoint: FixedWingSetpoint {
                airspeed: 25.0,
                altitude: alt,
                course: 0.0,
            },
        }
    }
}

/// Place a flat local-NED state (e.g. a trim solution at the origin) onto the
/// sphere at PCI `anchor`: the position becomes the anchor, and velocity /
/// attitude are rotated from the local NED frame at the anchor into PCI. The
/// body angular rate is frame-invariant. This is how the fixed-wing is spawned
/// (and re-spawned at a new altitude) without re-solving trim on the sphere.
fn place_on_sphere(local: &State13, anchor: Vec3) -> State13 {
    let q_pci_from_ned = planet::pci_from_ned(anchor);
    State13 {
        position: anchor + q_pci_from_ned * local.position,
        velocity: q_pci_from_ned * local.velocity,
        attitude: q_pci_from_ned * local.attitude,
        angular_rate: local.angular_rate,
    }
}

/// Build the synthetic **local-NED** [`EstState`] the autopilot consumes, from
/// the PCI truth. The autopilot is curvature-agnostic by construction: it always
/// flies relative to the *local horizon* derived here from the current position
/// (attitude composed with `q_ned_from_pci`, velocity rotated into NED, altitude
/// from the radius). No autopilot code changes — only its inputs rotate.
fn local_est_from_pci(s: &State13) -> EstState {
    let q_ned_from_pci = planet::ned_from_pci(s.position);
    EstState {
        position: Vec3::new(0.0, 0.0, -planet::altitude_of(s.position)),
        velocity: q_ned_from_pci * s.velocity,
        attitude: q_ned_from_pci * s.attitude,
        angular_rate: s.angular_rate,
    }
}

/// How [`FwSim::step`] derives the setpoint each control gate.
enum FwMode {
    /// Hold the externally-set [`FixedWingSetpoint`] (the original behaviour).
    Setpoint,
    /// Follow a waypoint route, recomputing the setpoint from truth each gate.
    Route(FwGuidance),
}

/// A deterministic fixed-wing simulator: autopilot → aero wrench → RK4.
///
/// Flies in still air for M6: the plant's `fixedwing_wrench` supports a wind
/// field, but the truth-feedback autopilot has no way to separate airspeed from
/// ground speed without an airspeed sensor, so wind (and Dryden turbulence) are
/// deferred to the same future milestone that adds sensors to this loop.
pub struct FwSim {
    truth: State13,
    params: FixedWingParams,
    autopilot: Box<dyn FixedWingController>,
    /// Last setpoint actually applied to the autopilot (logged each sample).
    setpoint: FixedWingSetpoint,
    /// Setpoint-hold vs route-follow (selects how `step` builds the setpoint).
    mode: FwMode,
    controls: FixedWingControls,
    dt: Real,
    control_period: u64,
    tick: u64,
    log: Vec<FwSample>,
    log_every: u64,
    log_cap: Option<usize>,
}

impl FwSim {
    pub fn new(cfg: FwSimConfig) -> Self {
        let control_period = ((1.0 / (cfg.control_rate * cfg.dt)).round() as u64).max(1);
        Self {
            truth: cfg.initial,
            params: cfg.params,
            autopilot: Box::new(FixedWingAutopilot::new(cfg.autopilot)),
            setpoint: cfg.setpoint,
            mode: FwMode::Setpoint,
            controls: FixedWingControls::zero(),
            dt: cfg.dt,
            control_period,
            tick: 0,
            log: Vec::new(),
            log_every: 5,
            log_cap: None,
        }
    }

    /// Update the commanded airspeed/altitude/course. Also switches the sim back
    /// to single-setpoint mode, cancelling any active route.
    pub fn set_setpoint(&mut self, sp: FixedWingSetpoint) {
        self.setpoint = sp;
        self.mode = FwMode::Setpoint;
    }

    /// Switch to route-following mode: walk `waypoints` (NED), recomputing the
    /// setpoint from truth each control gate. The first leg runs from the
    /// aircraft's *current* position to `waypoints[0]`. Cancels any prior route
    /// or held setpoint. An empty route degrades to holding the start altitude.
    pub fn set_route(&mut self, waypoints: Vec<Waypoint>, cfg: FwGuidanceConfig) {
        let start = self.truth.position;
        let mut g = FwGuidance::new(waypoints, start, cfg);
        // Prime the setpoint so a log/read before the first gate is sensible.
        self.setpoint = g.update(self.truth.position);
        self.mode = FwMode::Route(g);
    }

    /// Active waypoint index when route-following, else `None`.
    pub fn waypoint_index(&self) -> Option<usize> {
        match &self.mode {
            FwMode::Route(g) => g.current_index(),
            FwMode::Setpoint => None,
        }
    }

    /// True once a route has captured its final waypoint (always `false` in
    /// setpoint mode).
    pub fn route_complete(&self) -> bool {
        match &self.mode {
            FwMode::Route(g) => g.is_complete(),
            FwMode::Setpoint => false,
        }
    }

    /// The active commanded setpoint (route-derived when following a route).
    pub fn setpoint(&self) -> FixedWingSetpoint {
        self.setpoint
    }

    pub fn truth(&self) -> &State13 {
        &self.truth
    }
    pub fn controls(&self) -> FixedWingControls {
        self.controls
    }
    pub fn time(&self) -> Real {
        self.tick as Real * self.dt
    }
    /// Physics step counter (0 at construction).
    pub fn tick(&self) -> Tick {
        self.tick
    }
    /// True airspeed (still air, non-rotating planet → equal to PCI ground
    /// speed, which is also the air-relative speed).
    pub fn airspeed(&self) -> Real {
        self.truth.velocity.norm()
    }
    /// Altitude above the planet surface \[m\] = `|position| − R`.
    pub fn altitude(&self) -> Real {
        planet::altitude_of(self.truth.position)
    }
    /// Course over ground χ \[rad\] in the **local** NED frame at the current
    /// position (`atan2(v_east, v_north)`).
    pub fn course(&self) -> Real {
        let v = planet::ned_from_pci(self.truth.position) * self.truth.velocity;
        v.y.atan2(v.x)
    }
    pub fn samples(&self) -> &[FwSample] {
        &self.log
    }

    /// Log every `every` base steps, keeping at most `cap` samples (a rolling
    /// window when `Some`).
    pub fn set_logging(&mut self, every: u64, cap: Option<usize>) {
        self.log_every = every.max(1);
        self.log_cap = cap;
    }

    /// Advance one base step (control runs on its own slower gate).
    pub fn step(&mut self) {
        if self.tick.is_multiple_of(self.control_period) {
            // Route mode: derive the setpoint from truth (M6 perfect feedback).
            if let FwMode::Route(g) = &mut self.mode {
                self.setpoint = g.update(self.truth.position);
            }
            // Truth feedback (M6), rotated into the local horizon (M7 sphere).
            let est = local_est_from_pci(&self.truth);
            let control_dt = self.control_period as Real * self.dt;
            self.controls = self
                .autopilot
                .step(&est, &self.setpoint, control_dt)
                .clamp(&self.params.limits);
        }

        let p = &self.params;
        let c = self.controls;
        self.truth = Rk4.step(
            &self.truth,
            |x| {
                rigid_body_deriv(
                    x,
                    &fixedwing_wrench(x, p, &c, Vec3::zeros(), planet::gravity_at(x.position)),
                    p.mass,
                    &p.inertia,
                    &p.inertia_inv,
                )
            },
            self.dt,
        );

        if self.tick.is_multiple_of(self.log_every) {
            if let Some(cap) = self.log_cap {
                if self.log.len() >= cap {
                    self.log.remove(0);
                }
            }
            self.log.push(FwSample {
                t: self.time(),
                truth: self.truth,
                controls: self.controls,
                setpoint: self.setpoint,
            });
        }
        self.tick += 1;
    }

    /// Run a fixed number of base steps headlessly.
    pub fn run_headless(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Signed horizontal cross-track error \[m\] of `pos` from the line through
/// `origin` with course `path_course` (positive = right of the path direction).
pub fn cross_track(pos: Vec3, origin: Vec3, path_course: Real) -> Real {
    let (cs, sn) = (path_course.cos(), path_course.sin());
    let (dx, dy) = (pos.x - origin.x, pos.y - origin.y);
    -sn * dx + cs * dy
}

/// Straight-line vector-field guidance: the course to fly to converge onto and
/// track the path. `chi_inf` is the approach angle far from the path; `k_path`
/// sets how aggressively cross-track is nulled.
pub fn line_course(
    pos: Vec3,
    origin: Vec3,
    path_course: Real,
    chi_inf: Real,
    k_path: Real,
) -> Real {
    let e_py = cross_track(pos, origin, path_course);
    path_course - chi_inf * (2.0 / core::f64::consts::PI) * (k_path * e_py).atan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};
    use fsim_core::planet;

    fn cruise_sim() -> FwSim {
        FwSim::new(FwSimConfig::aerosonde_cruise())
    }

    /// A waypoint at a local North/East offset (metres) from home, at altitude
    /// `alt`. Near the home point (equator/prime meridian) the small-angle map is
    /// lat ≈ north/R, lon ≈ east/R, so these mirror the old flat NE legs.
    fn wp_ne(north_m: Real, east_m: Real, alt: Real) -> Waypoint {
        Waypoint::geodetic(
            north_m / planet::PLANET_RADIUS,
            east_m / planet::PLANET_RADIUS,
            alt,
        )
    }

    // T-Hold: holding the trim setpoint keeps airspeed / altitude / course.
    // (Tolerances are a touch looser than flat-earth: the trim is solved at a
    // constant g but spawned where radial gravity is ~3 % weaker, so the closed
    // loop settles with a small transient.)
    #[test]
    fn holds_trimmed_cruise() {
        let mut sim = cruise_sim();
        sim.run_headless(30_000); // 30 s
        assert!(
            (sim.airspeed() - 25.0).abs() < 1.5,
            "airspeed {}",
            sim.airspeed()
        );
        assert!(
            (sim.altitude() - 100.0).abs() < 8.0,
            "altitude {}",
            sim.altitude()
        );
        assert!(sim.course().abs() < 0.05, "course {}", sim.course());
    }

    // T-Climb: a +50 m altitude step is tracked, airspeed stays bounded.
    #[test]
    fn climbs_to_new_altitude() {
        let mut sim = cruise_sim();
        sim.run_headless(5_000); // settle
        sim.set_setpoint(FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 150.0,
            course: 0.0,
        });
        sim.set_logging(50, None);
        sim.run_headless(40_000); // 40 s to climb + settle
        assert!(
            (sim.altitude() - 150.0).abs() < 6.0,
            "altitude {}",
            sim.altitude()
        );
        let worst_va = sim
            .samples()
            .iter()
            .map(|s| (s.truth.velocity.norm() - 25.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst_va < 4.0, "airspeed excursion {worst_va} during climb");
    }

    // T-Speed: a +5 m/s airspeed step is tracked; altitude recovers.
    #[test]
    fn tracks_new_airspeed() {
        let mut sim = cruise_sim();
        sim.run_headless(5_000);
        sim.set_setpoint(FixedWingSetpoint {
            airspeed: 30.0,
            altitude: 100.0,
            course: 0.0,
        });
        sim.run_headless(40_000);
        assert!(
            (sim.airspeed() - 30.0).abs() < 1.0,
            "airspeed {}",
            sim.airspeed()
        );
        assert!(
            (sim.altitude() - 100.0).abs() < 6.0,
            "altitude {}",
            sim.altitude()
        );
    }

    // T-Turn: a +90 deg course change is tracked; altitude held.
    #[test]
    fn turns_to_new_course() {
        let mut sim = cruise_sim();
        sim.run_headless(5_000);
        sim.set_setpoint(FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: FRAC_PI_2,
        });
        sim.run_headless(45_000);
        let course_err = (sim.course() - FRAC_PI_2).abs();
        assert!(course_err < 0.05, "course err {course_err}");
        assert!(
            (sim.altitude() - 100.0).abs() < 8.0,
            "altitude {}",
            sim.altitude()
        );
    }

    // T-Curvature: level cruise on the sphere *follows the curve* — the inertial
    // (PCI) velocity DIRECTION rotates as the aircraft flies, while altitude is
    // held. On a flat earth the velocity direction would be constant; here it
    // must turn by ~(arc/R) over the leg. This test fails on a flat model.
    #[test]
    fn level_cruise_follows_the_curve() {
        let mut sim = cruise_sim();
        sim.run_headless(10_000); // settle the spawn transient → level tangent flight
        let v0 = sim.truth().velocity.normalize();
        let a0 = sim.altitude();
        sim.run_headless(40_000); // 40 s North ≈ 1 km ≈ 0.157 rad of arc
        let v1 = sim.truth().velocity.normalize();
        let turned = v0.dot(&v1).clamp(-1.0, 1.0).acos();
        // The PCI velocity must rotate by ≈ arc/R (~0.15 rad) — on a FLAT earth it
        // would not rotate at all (turned ≡ 0), so this is a decisive curvature
        // check; the band also catches a wildly-wrong (too-large) rotation.
        assert!(
            (0.08..0.25).contains(&turned),
            "PCI velocity should rotate ≈ arc/R with the curve: {turned} rad"
        );
        assert!(
            (sim.altitude() - a0).abs() < 8.0,
            "altitude drifted: {}",
            sim.altitude()
        );
        assert!(
            sim.course().abs() < 0.05,
            "still flying local North: {}",
            sim.course()
        );
    }

    // T-Guidance: great-circle vector field nulls cross-track and tracks the leg.
    #[test]
    fn great_circle_guidance_converges() {
        let mut sim = cruise_sim();
        // A North-going great circle (meridian) offset 40 m East; the craft starts
        // ~40 m West of it and must converge onto it.
        let a = wp_ne(0.0, 40.0, 100.0).position;
        let b = wp_ne(4000.0, 40.0, 100.0).position;
        let n = planet::gc_normal(a, b);
        assert!(planet::gc_cross_track(sim.truth().position, n).abs() > 35.0);
        for _ in 0..700 {
            let path = planet::gc_course(sim.truth().position, n);
            let xt = planet::gc_cross_track(sim.truth().position, n);
            let course = path - 0.9 * (2.0 / PI) * (0.05 * xt).atan();
            sim.set_setpoint(FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course,
            });
            sim.run_headless(100);
        }
        let e = planet::gc_cross_track(sim.truth().position, n);
        assert!(e.abs() < 5.0, "cross-track not nulled: {e}");
        assert!(
            sim.course().abs() < 0.2,
            "not tracking North: {}",
            sim.course()
        );
    }

    // T-NoSnake: the shipped guidance + autopilot tuning converges onto a leg and
    // HOLDS it without left/right S-turns (measured on the sphere via great-circle
    // cross-track).
    #[test]
    fn line_following_does_not_snake() {
        let cfg = FwGuidanceConfig::default();
        let mut sim = cruise_sim();
        let a = wp_ne(0.0, 80.0, 100.0).position;
        let b = wp_ne(4000.0, 80.0, 100.0).position;
        let n = planet::gc_normal(a, b);
        let mut xt = Vec::new();
        for _ in 0..1400 {
            let path = planet::gc_course(sim.truth().position, n);
            let e = planet::gc_cross_track(sim.truth().position, n);
            let course = path - cfg.chi_inf * (2.0 / PI) * (cfg.k_path * e).atan();
            sim.set_setpoint(FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course,
            });
            sim.run_headless(100);
            xt.push(planet::gc_cross_track(sim.truth().position, n));
        }
        let tail = &xt[700..];
        let max_amp = tail.iter().fold(0.0_f64, |m, &e| m.max(e.abs()));
        let sign_changes = tail
            .windows(2)
            .filter(|w| w[0] != 0.0 && w[1] != 0.0 && w[0].signum() != w[1].signum())
            .count();
        assert!(
            max_amp < 4.0,
            "steady-state cross-track too large (snaking): {max_amp} m"
        );
        assert!(
            sign_changes <= 4,
            "too many sign changes (snaking): {sign_changes}"
        );
    }

    // T-Determinism: the truth path has no RNG, so two runs are bit-identical.
    #[test]
    fn is_deterministic() {
        let run = || {
            let mut sim = cruise_sim();
            sim.set_setpoint(FixedWingSetpoint {
                airspeed: 27.0,
                altitude: 120.0,
                course: 0.3,
            });
            sim.run_headless(8_000);
            *sim.truth()
        };
        let a = run();
        let b = run();
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.attitude, b.attitude);
        assert_eq!(a.angular_rate, b.angular_rate);
    }

    use crate::fw_guidance::TerminalAction;

    fn l_route_cfg() -> FwGuidanceConfig {
        FwGuidanceConfig {
            airspeed: 25.0,
            accept_radius: 120.0,
            chi_inf: 0.9,
            k_path: 0.05,
            terminal: TerminalAction::HoldCourse,
        }
    }

    // T-RouteFlown: an L-route (~400 m legs) on the sphere is flown to completion;
    // the index advances, leg-2 great-circle cross-track stays bounded once rolled
    // out of the corner, and airspeed/altitude are held.
    #[test]
    fn route_l_is_flown_to_completion() {
        let mut sim = cruise_sim(); // home, 25 m/s North
        let wp0 = wp_ne(400.0, 0.0, 120.0); // North leg, climb to 120 m
        let wp1 = wp_ne(400.0, 400.0, 120.0); // turn East
        sim.set_route(vec![wp0, wp1], l_route_cfg());
        assert_eq!(sim.waypoint_index(), Some(0));
        let n2 = planet::gc_normal(wp0.position, wp1.position); // leg-2 great circle

        let mut captured_final = false;
        let mut worst_xt_leg2 = 0.0_f64;
        for _ in 0..90_000 {
            sim.step();
            // On leg 2, once rolled out (>150 m past the corner along the arc).
            if sim.waypoint_index() == Some(1)
                && planet::gc_distance(wp0.position, sim.truth().position) > 150.0
            {
                worst_xt_leg2 =
                    worst_xt_leg2.max(planet::gc_cross_track(sim.truth().position, n2).abs());
            }
            if sim.route_complete() {
                captured_final = true;
                break;
            }
        }
        assert!(captured_final, "route never reached the final waypoint");
        assert_eq!(sim.waypoint_index(), Some(1), "index did not reach last");
        assert!(
            worst_xt_leg2 < 25.0,
            "leg-2 cross-track unbounded: {worst_xt_leg2} m"
        );
        assert!(
            (sim.airspeed() - 25.0).abs() < 1.5,
            "airspeed not held: {}",
            sim.airspeed()
        );
        assert!(
            (sim.altitude() - 120.0).abs() < 6.0,
            "altitude not held: {}",
            sim.altitude()
        );
    }

    // T-RouteDeterminism: the route truth path has no RNG → bit-identical runs.
    #[test]
    fn route_is_deterministic() {
        let run = || {
            let mut sim = cruise_sim();
            sim.set_route(
                vec![
                    wp_ne(350.0, 0.0, 130.0),
                    wp_ne(350.0, 350.0, 130.0),
                    wp_ne(0.0, 350.0, 130.0),
                ],
                l_route_cfg(),
            );
            sim.run_headless(60_000);
            (*sim.truth(), sim.waypoint_index())
        };
        let a = run();
        let b = run();
        assert_eq!(a.0.position, b.0.position);
        assert_eq!(a.0.velocity, b.0.velocity);
        assert_eq!(a.0.attitude, b.0.attitude);
        assert_eq!(a.0.angular_rate, b.0.angular_rate);
        assert_eq!(a.1, b.1, "waypoint index diverged");
    }

    // T-RouteDoesNotBreakSetpoint: set_setpoint after a route cancels it.
    #[test]
    fn set_setpoint_cancels_route() {
        let mut sim = cruise_sim();
        sim.set_route(vec![wp_ne(400.0, 0.0, 120.0)], l_route_cfg());
        assert_eq!(sim.waypoint_index(), Some(0));
        sim.step();
        sim.set_setpoint(FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: 0.0,
        });
        assert_eq!(sim.waypoint_index(), None, "route not cancelled");
        sim.run_headless(20_000);
        assert!((sim.altitude() - 100.0).abs() < 5.0);
    }
}
