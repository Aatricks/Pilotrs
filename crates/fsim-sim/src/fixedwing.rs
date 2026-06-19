//! A thin fixed-wing simulation loop. It reuses only the airframe-agnostic
//! parts of the stack — `State13`, the `Rk4` integrator, the shared
//! `rigid_body_deriv`, and the fixed-wing aero `Wrench` — wired to the fixed-wing
//! autopilot. The quad scheduler is welded to `CtrlCmd` + a mixer + motors, so a
//! fixed-wing (four surfaces, body-x thrust) gets its own ~40-line loop.
//!
//! The autopilot here flies on **truth** (perfect feedback); swapping in sensors
//! and the INS is a one-line `est` change left for future work.

use crate::fw_guidance::{FwGuidance, FwGuidanceConfig};
use crate::guidance::Waypoint;
use fsim_control::{
    FbwConfig, FixedWingAutopilot, FixedWingConfig, FixedWingController, FixedWingSetpoint,
    FlyByWire,
};
use fsim_core::{
    planet, EstState, FixedWingControls, Real, State13, StickInput, Tick, Vec3, DEFAULT_DT, GRAVITY,
};
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
    /// Fly-by-wire tuning + trim references for manual (piloted) flight.
    pub fbw: FbwConfig,
    pub dt: Real,
    pub control_rate: Real,
    pub initial: State13,
    pub setpoint: FixedWingSetpoint,
    /// Start the sim already under manual (pilot-stick) control rather than the
    /// autopilot. The relaxed-stability fighter uses this.
    pub start_manual: bool,
    /// Initial state of the fly-by-wire toggle when `start_manual`.
    pub fbw_enabled: bool,
}

impl FwSimConfig {
    /// Aerosonde trimmed for 25 m/s level cruise at 100 m over the **home**
    /// point, heading North — spawned on the sphere.
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
        Self::from_trim(params, alt, false)
    }

    /// The **relaxed-stability fighter** under manual fly-by-wire control,
    /// spawned in level trimmed flight at altitude `alt` \[m\]. It starts with the
    /// FCS **on** (flyable); toggling it off lets the unstable airframe diverge.
    /// Plenty of altitude is sensible so the tumble has room when the FCS is off.
    pub fn fighter_manual(alt: Real) -> Self {
        let params = FixedWingParams::fighter_relaxed();
        Self::from_trim(params, alt, true)
    }

    /// The fighter at a default 300 m.
    pub fn fighter() -> Self {
        Self::fighter_manual(300.0)
    }

    /// Build a config by solving 25 m/s level trim for `params`, placing it on
    /// the sphere at `alt`, and deriving the fly-by-wire trim references from the
    /// same trim. `manual` selects whether the sim starts piloted (FCS on).
    fn from_trim(params: FixedWingParams, alt: Real, manual: bool) -> Self {
        let tr = trim(&params, 25.0, 0.0).expect("25 m/s level trim converges");
        let mut autopilot = FixedWingConfig::aerosonde();
        autopilot.trim_throttle = tr.controls.throttle;
        // Trim angle of attack (flat local trim, before placement on the sphere).
        let vb = tr.state.attitude.inverse() * tr.state.velocity;
        let alpha_trim = vb.z.atan2(vb.x);
        let fbw = FbwConfig::fighter(
            alpha_trim,
            tr.controls.elevator,
            tr.controls.throttle,
            25.0,
            params.rho,
            params.limits,
        );
        let initial = place_on_sphere(&tr.state, planet::home_pci(alt));
        Self {
            params,
            autopilot,
            fbw,
            dt: DEFAULT_DT,
            control_rate: 100.0,
            initial,
            setpoint: FixedWingSetpoint {
                airspeed: 25.0,
                altitude: alt,
                course: 0.0,
            },
            start_manual: manual,
            fbw_enabled: true,
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

/// How [`FwSim::step`] produces the control surfaces each control gate.
enum FwMode {
    /// Hold the externally-set [`FixedWingSetpoint`] via the autopilot.
    Setpoint,
    /// Follow a waypoint route, recomputing the setpoint from truth each gate.
    Route(FwGuidance),
    /// Human pilot: a fly-by-wire law turns the [`StickInput`] into surfaces,
    /// with a runtime on/off toggle (`fbw_enabled`). Off ⇒ direct passthrough.
    Manual {
        fbw: FlyByWire,
        stick: StickInput,
        fbw_enabled: bool,
    },
}

/// A deterministic fixed-wing simulator: autopilot → aero wrench → RK4.
///
/// Flies in still air: the plant's `fixedwing_wrench` supports a wind field, but
/// the truth-feedback autopilot has no way to separate airspeed from ground speed
/// without an airspeed sensor, so wind (and Dryden turbulence) are deferred to
/// the same future work that adds sensors to this loop.
pub struct FwSim {
    truth: State13,
    params: FixedWingParams,
    autopilot: Box<dyn FixedWingController>,
    /// Fly-by-wire tuning (used to build the manual mode on demand).
    fbw_cfg: FbwConfig,
    /// Last setpoint actually applied to the autopilot (logged each sample).
    setpoint: FixedWingSetpoint,
    /// Setpoint-hold / route-follow / manual (selects how `step` builds surfaces).
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
        let mode = if cfg.start_manual {
            FwMode::Manual {
                fbw: FlyByWire::new(cfg.fbw),
                stick: StickInput::neutral(),
                fbw_enabled: cfg.fbw_enabled,
            }
        } else {
            FwMode::Setpoint
        };
        Self {
            truth: cfg.initial,
            params: cfg.params,
            autopilot: Box::new(FixedWingAutopilot::new(cfg.autopilot)),
            fbw_cfg: cfg.fbw,
            setpoint: cfg.setpoint,
            mode,
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

    /// Switch to manual (piloted) control: a fly-by-wire law maps the pilot's
    /// stick to surfaces. `fbw_enabled` is the initial state of the FCS toggle.
    /// Cancels any route or held setpoint.
    pub fn enter_manual(&mut self, fbw_enabled: bool) {
        self.mode = FwMode::Manual {
            fbw: FlyByWire::new(self.fbw_cfg),
            stick: StickInput::neutral(),
            fbw_enabled,
        };
    }

    /// Update the pilot's stick demand. No-op unless in manual mode.
    pub fn set_stick(&mut self, s: StickInput) {
        if let FwMode::Manual { stick, .. } = &mut self.mode {
            *stick = s;
        }
    }

    /// Set the fly-by-wire toggle (on = stabilized/flyable, off = passthrough).
    /// No-op unless in manual mode.
    pub fn set_fbw(&mut self, on: bool) {
        if let FwMode::Manual { fbw_enabled, .. } = &mut self.mode {
            *fbw_enabled = on;
        }
    }

    /// True when flying under manual control.
    pub fn is_manual(&self) -> bool {
        matches!(self.mode, FwMode::Manual { .. })
    }

    /// True when manual *and* the fly-by-wire FCS is engaged.
    pub fn fbw_on(&self) -> bool {
        matches!(&self.mode, FwMode::Manual { fbw_enabled, .. } if *fbw_enabled)
    }

    /// Active waypoint index when route-following, else `None`.
    pub fn waypoint_index(&self) -> Option<usize> {
        match &self.mode {
            FwMode::Route(g) => g.current_index(),
            FwMode::Setpoint | FwMode::Manual { .. } => None,
        }
    }

    /// True once a route has captured its final waypoint (always `false` in
    /// setpoint / manual mode).
    pub fn route_complete(&self) -> bool {
        match &self.mode {
            FwMode::Route(g) => g.is_complete(),
            FwMode::Setpoint | FwMode::Manual { .. } => false,
        }
    }

    /// Angle of attack \[rad\] from the current truth (still air ⇒ airspeed =
    /// ground speed): the angle of the body-frame velocity below the body x-axis.
    pub fn alpha(&self) -> Real {
        let vb = self.truth.attitude.inverse() * self.truth.velocity;
        vb.z.atan2(vb.x)
    }

    /// Aerodynamic load factor `n` (in g): the body-normal specific force from
    /// aero + thrust, divided by standard gravity. ~1 in level cruise, higher in
    /// a hard pull. Excludes gravity (the wrench is evaluated with `g = 0`).
    pub fn load_factor(&self) -> Real {
        let w = fixedwing_wrench(
            &self.truth,
            &self.params,
            &self.controls,
            Vec3::zeros(),
            Vec3::zeros(),
        );
        let f_body = self.truth.attitude.inverse() * w.force_world;
        -f_body.z / (self.params.mass * GRAVITY)
    }

    /// Inject a body-frame angular-rate perturbation into the truth state — an
    /// external upset (gust, disturbance). Used to excite the airframe
    /// independently of the pilot's stick, e.g. to test that the FCS recovers
    /// from an upset while the open-loop airframe diverges from the same one.
    pub fn nudge_angular_rate(&mut self, delta_body: Vec3) {
        self.truth.angular_rate += delta_body;
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
            let control_dt = self.control_period as Real * self.dt;
            // Truth feedback, rotated into the local horizon (sphere).
            let est = local_est_from_pci(&self.truth);
            // Route mode: derive the setpoint from truth (perfect feedback) first.
            if let FwMode::Route(g) = &mut self.mode {
                self.setpoint = g.update(self.truth.position);
            }
            self.controls = match &mut self.mode {
                // Pilot-in-the-loop: fly-by-wire (or passthrough) from the stick.
                FwMode::Manual {
                    fbw,
                    stick,
                    fbw_enabled,
                } => fbw.step(&est, stick, control_dt, *fbw_enabled),
                // Autopilot (setpoint or route-derived).
                FwMode::Setpoint | FwMode::Route(_) => {
                    self.autopilot.step(&est, &self.setpoint, control_dt)
                }
            }
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

    // --- Manual / fly-by-wire ---

    /// Hit the manually-flown fighter with an identical **external** pitch upset
    /// (independent of the stick) and report `(worst body-rate, final body-rate)`
    /// over `secs` seconds. The external kick is what makes the FCS *do work* — an
    /// FBW that ignored the airframe would fail to recover.
    fn fighter_upset_response(fbw_on: bool, secs: f64) -> (Real, Real) {
        let mut sim = FwSim::new(FwSimConfig::fighter_manual(300.0));
        sim.set_fbw(fbw_on);
        sim.set_stick(StickInput {
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            throttle: 0.55, // near the trim throttle so it holds airspeed
        });
        sim.step(); // one gate so the throttle is applied
        sim.nudge_angular_rate(Vec3::new(0.0, 0.3, 0.0)); // 0.3 rad/s pitch upset
        let steps = (secs / DEFAULT_DT) as usize;
        let mut worst = 0.0_f64;
        for _ in 0..steps {
            sim.step();
            worst = worst.max(sim.truth().angular_rate.norm());
        }
        (worst, sim.truth().angular_rate.norm())
    }

    // The headline toggle, headless: an external 0.3 rad/s pitch upset is ARRESTED
    // and damped back to trim with the FCS ON, but RUNS AWAY with it OFF. The
    // external kick (not a stick input) is essential — it forces the FCS to
    // actually stabilize rather than sit at an undisturbed equilibrium.
    #[test]
    fn fighter_fbw_recovers_from_upset_off_diverges() {
        let (on_worst, on_final) = fighter_upset_response(true, 8.0);
        let (off_worst, _) = fighter_upset_response(false, 4.0);
        // FBW ON: the 0.3 rad/s upset is arrested (never amplified) and decays
        // well below its initial size — the short period is damped. (A small
        // residual remains: the lightly-damped phugoid on the curved planet.)
        assert!(
            on_worst < 0.6,
            "FBW should bound the upset, not amplify it: worst {on_worst} rad/s"
        );
        assert!(
            on_final < 0.2 && on_final < on_worst,
            "FBW should damp the upset down from its peak: final {on_final}, worst {on_worst}"
        );
        // FBW OFF: the identical upset runs away.
        assert!(
            off_worst > 1.5,
            "no FBW ⇒ the same upset diverges: worst {off_worst} rad/s"
        );
        assert!(
            off_worst > 5.0 * on_worst,
            "off ({off_worst}) must run away far past on ({on_worst})"
        );
    }

    // The fighter starts in manual with the FCS engaged.
    #[test]
    fn fighter_starts_manual_fbw_on() {
        let sim = FwSim::new(FwSimConfig::fighter_manual(300.0));
        assert!(sim.is_manual(), "fighter should start piloted");
        assert!(sim.fbw_on(), "FCS should start engaged");
    }

    // The HUD readouts: at spawn the fighter sits at its trim angle of attack and
    // ~1 g (level cruise); both getters must reflect that.
    #[test]
    fn fighter_alpha_and_load_factor_readouts() {
        let sim = FwSim::new(FwSimConfig::fighter_manual(300.0));
        // Trim AoA for the fighter at 25 m/s level is ≈ 0.1 rad.
        assert!(
            (0.05..0.15).contains(&sim.alpha()),
            "AoA at spawn should be near trim: {}",
            sim.alpha()
        );
        // Level cruise ⇒ lift ≈ weight ⇒ load factor ≈ 1 g.
        assert!(
            (sim.load_factor() - 1.0).abs() < 0.25,
            "load factor in level cruise should be ~1 g: {}",
            sim.load_factor()
        );
    }

    // A pitch command under FBW produces a body-frame pitch-rate response (the
    // aircraft tracks the stick) without diverging.
    #[test]
    fn fbw_tracks_a_pitch_command() {
        let mut sim = FwSim::new(FwSimConfig::fighter_manual(300.0));
        sim.set_stick(StickInput {
            pitch: 0.6, // pull back
            roll: 0.0,
            yaw: 0.0,
            throttle: 0.55,
        });
        let mut peak_q = 0.0_f64;
        for _ in 0..1500 {
            sim.step();
            // angular_rate is already body-frame (FRD) — read q directly.
            peak_q = peak_q.max(sim.truth().angular_rate.y);
        }
        assert!(
            peak_q > 0.05,
            "pull-back should command a nose-up rate: {peak_q}"
        );
        assert!(
            sim.truth().angular_rate.norm() < 1.0,
            "but stay bounded (not tumbling): {}",
            sim.truth().angular_rate.norm()
        );
    }

    // T-ManualDeterminism: a scripted stick + toggle sequence is reproducible
    // (manual flight adds no RNG to the truth path).
    #[test]
    fn manual_is_deterministic() {
        let run = || {
            let mut sim = FwSim::new(FwSimConfig::fighter_manual(300.0));
            for k in 0..6000u64 {
                sim.set_stick(StickInput {
                    pitch: if k < 1000 { 0.5 } else { 0.0 },
                    roll: if (2000..3000).contains(&k) { 0.7 } else { 0.0 },
                    yaw: 0.0,
                    throttle: 0.55,
                });
                if k == 4000 {
                    sim.set_fbw(false); // flip the toggle mid-run
                }
                sim.step();
            }
            *sim.truth()
        };
        let a = run();
        let b = run();
        // Finite, so the bit-equality below compares real numbers (two diverging
        // runs that both overflowed to ±inf would compare equal and hide a bug).
        assert!(
            a.position
                .iter()
                .chain(a.angular_rate.iter())
                .all(|x| x.is_finite()),
            "state should stay finite over the run"
        );
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.attitude, b.attitude);
        assert_eq!(a.angular_rate, b.angular_rate);
    }
}
