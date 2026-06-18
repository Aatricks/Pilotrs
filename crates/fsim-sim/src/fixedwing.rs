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
use fsim_core::{EstState, FixedWingControls, Real, State13, Tick, Vec3, DEFAULT_DT};
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
    /// Aerosonde trimmed for 25 m/s level cruise at 100 m, heading North. The
    /// autopilot's throttle feed-forward is taken from the trim solution.
    pub fn aerosonde_cruise() -> Self {
        let params = FixedWingParams::aerosonde();
        let tr = trim(&params, 25.0, 0.0).expect("Aerosonde 25 m/s level trim converges");
        let mut autopilot = FixedWingConfig::aerosonde();
        autopilot.trim_throttle = tr.controls.throttle;
        let mut initial = tr.state;
        initial.position = Vec3::new(0.0, 0.0, -100.0);
        Self {
            params,
            autopilot,
            dt: DEFAULT_DT,
            control_rate: 100.0,
            initial,
            setpoint: FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course: 0.0,
            },
        }
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
    /// True airspeed (still air for M6, so equal to ground speed).
    pub fn airspeed(&self) -> Real {
        self.truth.velocity.norm()
    }
    /// Altitude (`-z`, NED).
    pub fn altitude(&self) -> Real {
        -self.truth.position.z
    }
    /// Course over ground χ \[rad\].
    pub fn course(&self) -> Real {
        self.truth.velocity.y.atan2(self.truth.velocity.x)
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
            // Truth feedback for M6 (sensors/estimator deferred).
            let est = EstState {
                position: self.truth.position,
                velocity: self.truth.velocity,
                attitude: self.truth.attitude,
                angular_rate: self.truth.angular_rate,
            };
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
                    &fixedwing_wrench(x, p, &c, Vec3::zeros()),
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

    fn cruise_sim() -> FwSim {
        FwSim::new(FwSimConfig::aerosonde_cruise())
    }

    // T-Hold: holding the trim setpoint keeps airspeed / altitude / course.
    #[test]
    fn holds_trimmed_cruise() {
        let mut sim = cruise_sim();
        sim.run_headless(30_000); // 30 s
        assert!(
            (sim.airspeed() - 25.0).abs() < 1.0,
            "airspeed {}",
            sim.airspeed()
        );
        assert!(
            (sim.altitude() - 100.0).abs() < 5.0,
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
            (sim.altitude() - 150.0).abs() < 5.0,
            "altitude {}",
            sim.altitude()
        );
        // Airspeed stayed sane throughout the climb (coupling guard).
        let worst_va = sim
            .samples()
            .iter()
            .map(|s| ((s.truth.velocity).norm() - 25.0).abs())
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
            (sim.altitude() - 100.0).abs() < 5.0,
            "altitude {}",
            sim.altitude()
        );
    }

    // T-Turn: a +90° course change is tracked; altitude held (lateral/long. decouple).
    #[test]
    fn turns_to_new_course() {
        let mut sim = cruise_sim();
        sim.run_headless(5_000);
        sim.set_setpoint(FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: core::f64::consts::FRAC_PI_2, // 90° (East)
        });
        sim.run_headless(45_000);
        let course_err = (sim.course() - core::f64::consts::FRAC_PI_2).abs();
        assert!(course_err < 0.05, "course err {course_err}");
        assert!(
            (sim.altitude() - 100.0).abs() < 8.0,
            "altitude {}",
            sim.altitude()
        );
    }

    // T-Guidance: straight-line guidance nulls cross-track and tracks the path.
    #[test]
    fn straight_line_guidance_converges() {
        let mut sim = cruise_sim();
        // Due-North path offset 40 m East; the craft starts at the origin, i.e.
        // 40 m left (West) of the path, and must converge onto it.
        let origin = Vec3::new(0.0, 40.0, -100.0);
        let path_course = 0.0; // North
        assert!(cross_track(sim.truth().position, origin, path_course).abs() > 35.0);
        for _ in 0..700 {
            let course = line_course(sim.truth().position, origin, path_course, 0.9, 0.05);
            sim.set_setpoint(FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course,
            });
            sim.run_headless(100); // re-aim every 0.1 s (70 s total)
        }
        let e = cross_track(sim.truth().position, origin, path_course);
        assert!(e.abs() < 5.0, "cross-track not nulled: {e}");
        assert!(
            sim.course().abs() < 0.15,
            "not tracking North: {}",
            sim.course()
        );
    }

    // T-NoSnake: with the SHIPPED defaults (autopilot gains + guidance k_path),
    // the closed-loop ground track converges onto a straight line and HOLDS it
    // without the left/right S-turns the old tuning produced. We measure the
    // steady-state cross-track amplitude and the number of sign changes
    // (oscillations) over the back half of a long run.
    #[test]
    fn line_following_does_not_snake() {
        use crate::fw_guidance::FwGuidanceConfig;
        let cfg = FwGuidanceConfig::default(); // shipped k_path / chi_inf
        let mut sim = cruise_sim(); // starts (0,0,-100), 25 m/s North
                                    // Due-North path 80 m East: the craft starts 80 m West of it.
        let origin = Vec3::new(0.0, 80.0, -100.0);
        let path = 0.0_f64;
        let mut xt = Vec::new();
        for _ in 0..1400 {
            let course = line_course(sim.truth().position, origin, path, cfg.chi_inf, cfg.k_path);
            sim.set_setpoint(FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course,
            });
            sim.run_headless(100); // re-aim every 0.1 s (140 s total)
            xt.push(cross_track(sim.truth().position, origin, path));
        }
        // Back half = settled flight. Amplitude must be small and the track must
        // not keep crossing the line back and forth (the snaking signature).
        let tail = &xt[700..];
        let max_amp = tail.iter().fold(0.0_f64, |m, &e| m.max(e.abs()));
        let sign_changes = tail
            .windows(2)
            .filter(|w| w[0] != 0.0 && w[1] != 0.0 && w[0].signum() != w[1].signum())
            .count();
        assert!(
            max_amp < 4.0,
            "steady-state cross-track amplitude too large (snaking): {max_amp} m"
        );
        assert!(
            sign_changes <= 4,
            "too many cross-track sign changes (snaking): {sign_changes}"
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

    use crate::fw_guidance::{FwGuidanceConfig, TerminalAction};

    /// L-route guidance tuned for the Aerosonde (accept radius > turn R ≈ 110 m).
    fn l_route_cfg() -> FwGuidanceConfig {
        FwGuidanceConfig {
            airspeed: 25.0,
            accept_radius: 120.0,
            chi_inf: 0.9,
            k_path: 0.05,
            terminal: TerminalAction::HoldCourse,
        }
    }

    // T-RouteFlown: an L-route (~400 m legs at 120 m) is flown to completion —
    // the index advances to the last waypoint, cross-track stays bounded on the
    // straight legs, and airspeed/altitude are held.
    #[test]
    fn route_l_is_flown_to_completion() {
        let mut sim = cruise_sim(); // starts at (0,0,-100), 25 m/s North
        let route = vec![
            Waypoint::ne_alt(400.0, 0.0, 120.0), // North leg, climb to 120 m
            Waypoint::ne_alt(400.0, 400.0, 120.0), // turn East
        ];
        sim.set_route(route, l_route_cfg());
        assert_eq!(sim.waypoint_index(), Some(0));

        let mut captured_final = false;
        let mut worst_xt_leg2 = 0.0_f64;
        for _ in 0..90_000 {
            sim.step();
            // Cross-track on the second (East) leg, once active and rolled out.
            if sim.waypoint_index() == Some(1) && sim.truth().position.y > 150.0 {
                let e = cross_track(
                    sim.truth().position,
                    Vec3::new(400.0, 0.0, -120.0), // leg-2 origin = wp0
                    core::f64::consts::FRAC_PI_2,  // due East
                );
                worst_xt_leg2 = worst_xt_leg2.max(e.abs());
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

    // T-RouteDeterminism: the truth route path has no RNG, so two runs are
    // bit-identical (state + the guidance index latch).
    #[test]
    fn route_is_deterministic() {
        let run = || {
            let mut sim = cruise_sim();
            sim.set_route(
                vec![
                    Waypoint::ne_alt(350.0, 0.0, 130.0),
                    Waypoint::ne_alt(350.0, 350.0, 130.0),
                    Waypoint::ne_alt(0.0, 350.0, 130.0),
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

    // T-RouteDoesNotBreakSetpoint: set_setpoint after a route cancels the route,
    // waypoint_index goes back to None, and the held setpoint is tracked.
    #[test]
    fn set_setpoint_cancels_route() {
        let mut sim = cruise_sim();
        sim.set_route(vec![Waypoint::ne_alt(400.0, 0.0, 120.0)], l_route_cfg());
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
