//! The fixed-step scheduler: one struct that owns every subsystem and advances
//! them deterministically.

use crate::atmosphere::{Atmosphere, StormCell};
use crate::config::{ControllerKind, EstimatorKind, SimConfig};
use crate::faults::QuadFaults;
use crate::guidance::{Guidance, GuidanceConfig, Waypoint};
use crate::telemetry::{Telemetry, TelemetrySample};
use fsim_actuators::{Mixer, MotorModel, XQuadMixer};
use fsim_control::{CascadedPid, Controller, LqrController, PositionConfig, PositionController};
use fsim_core::{CtrlCmd, EstState, Real, Setpoint, State13, Tick, Vec3};
use fsim_dynamics::{aerodynamic_wrench, Integrator, MultirotorParams, Plant, RigidBody, Rk4};
use fsim_estimator::{ComplementaryFilter, Estimator, Ins, Mekf};
use fsim_sensors::{Baro, Gps, Imu, Mag, Sensor, Truth};

/// Waypoint guidance + position/velocity controller (position mode).
struct PositionMode {
    guidance: Guidance,
    ctrl: PositionController,
}

/// How the autopilot is commanded.
enum ControlMode {
    /// Attitude/thrust setpoint set externally (attitude-tracking modes).
    Attitude,
    /// Waypoint guidance → position/velocity control (needs the INS, which
    /// is the only estimator returning real position/velocity). Boxed because
    /// it is much larger than the `Attitude` variant.
    Position(Box<PositionMode>),
}

/// The complete simulator. The estimator is boxed so it can be swapped (CF /
/// MEKF / INS) per [`SimConfig`]; the control mode switches between attitude
/// setpoints and waypoint position control.
pub struct Sim {
    dt: Real,
    imu_period: Tick,
    control_period: Tick,
    mag_period: Tick,
    gps_period: Tick,
    baro_period: Tick,
    imu_dt: Real,
    control_dt: Real,

    params: MultirotorParams,
    plant: RigidBody,
    mixer: XQuadMixer,
    motors: MotorModel,
    imu: Imu,
    gps: Gps,
    baro: Baro,
    mag: Mag,
    estimator: Box<dyn Estimator>,
    estimator_kind: EstimatorKind,
    controller: Box<dyn Controller>,
    control_mode: ControlMode,
    position_cfg: PositionConfig,
    rk4: Rk4,
    /// The air the quad flies through (wind + turbulence).
    atmosphere: Atmosphere,
    /// Injected effector faults (a dead rotor).
    faults: QuadFaults,

    truth: State13,
    setpoint: Setpoint,
    cmd: CtrlCmd,
    last_motors: [Real; 4],
    tick: Tick,

    telemetry: Telemetry,
    log_every: Tick,
    history_cap: Option<usize>,
}

impl Sim {
    /// Build a simulator from config, starting at rest and level.
    pub fn new(cfg: SimConfig) -> Self {
        let base_rate = 1.0 / cfg.dt;
        let period = |rate: Real| (base_rate / rate).round().max(1.0) as Tick;
        let imu_period = period(cfg.imu_rate);
        let control_period = period(cfg.control_rate);
        let mag_period = period(cfg.mag.rate_hz);
        let gps_period = period(cfg.gps.rate_hz);
        let baro_period = period(cfg.baro.rate_hz);
        let hover = cfg.hover_thrust();

        // Snap each sensor's effective rate to the gated interval `period·dt`, so
        // its internal bias-random-walk step (`dt = 1/rate_hz`) exactly matches
        // how often the scheduler actually samples it — even when the requested
        // rate doesn't evenly divide the base rate.
        let snap = |period: Tick| base_rate / period as Real;
        let mut imu_cfg = cfg.imu;
        imu_cfg.rate_hz = snap(imu_period);
        let mut gps_cfg = cfg.gps;
        gps_cfg.rate_hz = snap(gps_period);
        let mut baro_cfg = cfg.baro;
        baro_cfg.rate_hz = snap(baro_period);
        let mut mag_cfg = cfg.mag;
        mag_cfg.rate_hz = snap(mag_period);

        // Independent RNG streams per sensor, derived from the master seed.
        let estimator: Box<dyn Estimator> = match cfg.estimator_kind {
            EstimatorKind::Complementary => Box::new(ComplementaryFilter::new(cfg.complementary)),
            EstimatorKind::Mekf => Box::new(Mekf::new(cfg.mekf)),
            EstimatorKind::Ins => Box::new(Ins::new(cfg.ins)),
        };

        // Inner attitude/rate controller (PID or LQR).
        let controller: Box<dyn Controller> = match cfg.controller_kind {
            ControllerKind::Pid => Box::new(CascadedPid::new(cfg.control)),
            ControllerKind::Lqr => {
                let i = &cfg.params.inertia;
                let diag = Vec3::new(i[(0, 0)], i[(1, 1)], i[(2, 2)]);
                Box::new(LqrController::new(diag, cfg.lqr))
            }
        };

        Self {
            dt: cfg.dt,
            imu_period,
            control_period,
            mag_period,
            gps_period,
            baro_period,
            imu_dt: imu_period as Real * cfg.dt,
            control_dt: control_period as Real * cfg.dt,

            params: cfg.params,
            plant: RigidBody::new(cfg.params),
            mixer: XQuadMixer::new(cfg.arm_length, cfg.yaw_coeff, cfg.max_thrust),
            motors: MotorModel::new(cfg.motor_tau, cfg.max_thrust),
            imu: Imu::new(imu_cfg, cfg.seed),
            gps: Gps::new(gps_cfg, cfg.seed ^ 0x1111_1111),
            baro: Baro::new(baro_cfg, cfg.seed ^ 0x2222_2222),
            mag: Mag::new(mag_cfg, cfg.seed ^ 0x3333_3333),
            estimator,
            estimator_kind: cfg.estimator_kind,
            controller,
            control_mode: ControlMode::Attitude,
            position_cfg: cfg.position,
            rk4: Rk4,
            atmosphere: Atmosphere::new(cfg.atmosphere),
            faults: QuadFaults::none(),

            truth: State13::at_rest(),
            setpoint: Setpoint::level(hover),
            cmd: CtrlCmd {
                thrust: hover,
                torque: Vec3::zeros(),
            },
            last_motors: [0.0; 4],
            tick: 0,

            telemetry: Telemetry::new(),
            log_every: 1,
            history_cap: None,
        }
    }

    /// Record one sample every `log_every` ticks, keeping at most `cap` samples
    /// (a rolling window for interactive use; `None` = unbounded for tests).
    pub fn set_logging(&mut self, log_every: Tick, cap: Option<usize>) {
        self.log_every = log_every.max(1);
        self.history_cap = cap;
    }

    /// Override the truth state (e.g. start mid-air or with an initial tilt).
    pub fn set_truth(&mut self, truth: State13) {
        self.truth = truth;
    }

    /// Set the steady wind \[m/s, local NED\].
    pub fn set_wind(&mut self, wind_ned: Vec3) {
        self.atmosphere.set_wind(wind_ned);
    }

    /// Set the turbulence intensity (RMS gust \[m/s\]; 0 = calm).
    pub fn set_turbulence(&mut self, rms: Real) {
        self.atmosphere.set_turbulence(rms);
    }

    /// Place (or clear) the storm / microburst cell.
    pub fn set_storm(&mut self, storm: Option<StormCell>) {
        self.atmosphere.set_storm(storm);
    }

    /// Inject (or clear) effector faults (a dead rotor).
    pub fn set_faults(&mut self, faults: QuadFaults) {
        self.faults = faults;
    }

    /// True if any fault is active (for the HUD).
    pub fn faulted(&self) -> bool {
        self.faults.any()
    }

    /// Steady wind speed \[m/s\] (for the HUD).
    pub fn wind_speed(&self) -> Real {
        self.atmosphere.wind_speed()
    }

    /// Instantaneous turbulence gust magnitude \[m/s\] (for the HUD).
    pub fn gust(&self) -> Real {
        self.atmosphere.gust_magnitude()
    }

    /// How deep in the storm the quad is (0 = clear, 1 = core).
    pub fn storm_intensity(&self) -> Real {
        self.atmosphere.storm_intensity()
    }

    /// Set the active attitude/thrust setpoint (the viewer calls this live).
    /// Also returns the simulator to attitude-control mode.
    pub fn set_setpoint(&mut self, sp: Setpoint) {
        self.setpoint = sp;
        self.control_mode = ControlMode::Attitude;
    }

    /// Switch to **position mode**: fly the given waypoint mission with the
    /// outer position/velocity controller. Requires the INS estimator (the CF
    /// and MEKF return zero position, so position control would chase the
    /// origin) — debug-asserted.
    pub fn set_mission(&mut self, waypoints: Vec<Waypoint>, gcfg: GuidanceConfig) {
        debug_assert!(
            matches!(self.estimator_kind, EstimatorKind::Ins),
            "position mode requires the INS estimator (CF/MEKF return zero position)"
        );
        self.control_mode = ControlMode::Position(Box::new(PositionMode {
            guidance: Guidance::new(waypoints, gcfg),
            ctrl: PositionController::new(self.position_cfg),
        }));
    }

    /// Index of the active waypoint, if flying a mission.
    pub fn waypoint_index(&self) -> Option<usize> {
        match &self.control_mode {
            ControlMode::Position(pm) => Some(pm.guidance.current_index()),
            ControlMode::Attitude => None,
        }
    }

    /// Simulated time \[s\].
    pub fn time(&self) -> Real {
        self.tick as Real * self.dt
    }

    /// Physics step counter (0 at construction).
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// Snapshot the current telemetry as a [`Recording`](crate::Recording).
    pub fn recording(&self) -> crate::Recording {
        crate::Recording::from_samples(self.telemetry.samples.clone())
    }

    /// Run `steps` headless and save the resulting telemetry to a recording.
    pub fn record_headless<P: std::convert::AsRef<std::path::Path>>(
        &mut self,
        steps: usize,
        path: P,
    ) -> std::io::Result<()> {
        self.run_headless(steps);
        self.recording().save(path)
    }

    /// Current ground truth.
    pub fn truth(&self) -> &State13 {
        &self.truth
    }

    /// Current estimate (what the autopilot acts on).
    pub fn estimate(&self) -> EstState {
        self.estimator.state()
    }

    /// Active setpoint.
    pub fn setpoint(&self) -> Setpoint {
        self.setpoint
    }

    /// Actual per-motor thrust last step \[N\].
    pub fn motors(&self) -> [Real; 4] {
        self.last_motors
    }

    /// Estimator's gyro-bias estimate, if it has one (the MEKF does).
    pub fn est_gyro_bias(&self) -> Option<Vec3> {
        self.estimator.gyro_bias_estimate()
    }

    /// The true (hidden) gyro bias inside the IMU \[rad/s\].
    pub fn true_gyro_bias(&self) -> Vec3 {
        self.imu.gyro_bias()
    }

    /// The recorded telemetry.
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Advance the whole simulator one base step (`dt`).
    pub fn step(&mut self) {
        let t = self.time();

        // Wind (flat local NED is the quad's world) — advance the atmosphere once
        // per step and use the same value for the accelerometer's specific force
        // and the integration, so drag is air-relative and the IMU sees it. The
        // quad's NED position is already (north, east) metres from home.
        let wind = self.atmosphere.wind_ned(
            self.truth.position.x,
            self.truth.position.y,
            self.truth.velocity.norm(),
            self.dt,
        );

        // 1. True acceleration acting right now, from the motor thrust that was
        //    in effect over the just-completed interval. The accelerometer
        //    needs this (it isn't part of State13).
        let held = self
            .mixer
            .collect(&self.faults.apply_motors(self.motors.thrust()));
        let accel_world =
            aerodynamic_wrench(&self.truth, &self.params, held.thrust, held.torque, wind)
                .force_world
                / self.params.mass;

        // 2. Sensors + estimator (each gated to its own rate). The order is
        //    predict (IMU + its internal gravity update) → mag → baro → gps;
        //    sequential EKF updates are order-insensitive but a fixed order
        //    keeps runs deterministic. `truth_now` is a copy so the sensor
        //    bundle doesn't borrow `self` across the mutable estimator calls.
        let truth_now = self.truth;
        let bundle = Truth {
            state: &truth_now,
            accel_world,
            t,
        };
        if self.tick.is_multiple_of(self.imu_period) {
            let imu_meas = self.imu.sample(&bundle);
            self.estimator.predict(&imu_meas, self.imu_dt);
        }
        if self.tick.is_multiple_of(self.mag_period) {
            let mag_meas = self.mag.sample(&bundle);
            self.estimator.update_mag(&mag_meas);
        }
        if self.tick.is_multiple_of(self.baro_period) {
            let baro_meas = self.baro.sample(&bundle);
            self.estimator.update_baro(&baro_meas);
        }
        if self.tick.is_multiple_of(self.gps_period) {
            let gps_meas = self.gps.sample(&bundle);
            self.estimator.update_gps(&gps_meas);
        }

        // 3. Controller (at the control rate), acting on the estimate only.
        //    In position mode the outer loop produces the attitude/thrust
        //    setpoint from waypoint guidance; the inner cascade is the same.
        if self.tick.is_multiple_of(self.control_period) {
            let est = self.estimator.state();
            let setpoint = match &mut self.control_mode {
                ControlMode::Attitude => self.setpoint,
                ControlMode::Position(pm) => {
                    let tgt = pm.guidance.update(est.position);
                    pm.ctrl.step(&est, &tgt, self.control_dt)
                }
            };
            // Record the *active* inner setpoint (in position mode this is the
            // guidance-produced commanded attitude/thrust) so telemetry and the
            // attitude plots reflect what the inner controller is actually
            // tracking — not the stale construction-time hover.
            self.setpoint = setpoint;
            self.cmd = self.controller.step(&est, &setpoint, self.control_dt);
        }

        // 4. Allocate to motors and apply the motor model.
        let motor_cmd = self.mixer.mix(&self.cmd);
        // A dead rotor produces no thrust no matter what the mixer commands.
        let actual = self
            .faults
            .apply_motors(self.motors.update(&motor_cmd, self.dt));
        let achieved = self.mixer.collect(&actual);

        // 5. Integrate truth one step (copies avoid borrowing `self` in the
        //    RK4 derivative closure).
        let plant = self.plant;
        let params = self.params;
        let (thrust, torque) = (achieved.thrust, achieved.torque);
        self.truth = self.rk4.step(
            &self.truth,
            |x| plant.deriv(x, &aerodynamic_wrench(x, &params, thrust, torque, wind)),
            self.dt,
        );

        // 6. Log + advance.
        self.last_motors = actual;
        if self.tick.is_multiple_of(self.log_every) {
            self.telemetry.push(TelemetrySample {
                t,
                truth: self.truth,
                estimate: self.estimator.state(),
                setpoint: self.setpoint,
                motors: actual,
                true_gyro_bias: self.imu.gyro_bias(),
                est_gyro_bias: self
                    .estimator
                    .gyro_bias_estimate()
                    .unwrap_or_else(Vec3::zeros),
            });
            if let Some(cap) = self.history_cap {
                let len = self.telemetry.samples.len();
                if len > cap {
                    self.telemetry.samples.drain(0..len - cap);
                }
            }
        }
        self.tick += 1;
    }

    /// Run `steps` base steps holding the current setpoint, returning the log.
    pub fn run_headless(&mut self, steps: usize) -> &Telemetry {
        for _ in 0..steps {
            self.step();
        }
        &self.telemetry
    }

    /// Run `steps` base steps, refreshing the setpoint from `guidance(t)` each
    /// step — for autonomous missions and time-varying references.
    pub fn run<F: FnMut(Real) -> Setpoint>(&mut self, steps: usize, mut guidance: F) -> &Telemetry {
        for _ in 0..steps {
            self.setpoint = guidance(self.time());
            self.step();
        }
        &self.telemetry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::GRAVITY;
    use nalgebra::UnitQuaternion;

    #[test]
    fn hover_holds_attitude_and_altitude() {
        // Level hover setpoint: the quad should stay near level and near its
        // starting altitude over 5 s, despite sensor noise.
        let mut sim = Sim::new(SimConfig::quad_250_mvp());
        sim.run_headless(5000);
        let s = sim.truth();
        assert!(
            s.attitude.angle() < 0.05,
            "attitude drifted: {}",
            s.attitude.angle()
        );
        assert!(s.position.norm() < 0.5, "drifted {} m", s.position.norm());
    }

    #[test]
    fn tracks_a_gentle_roll_setpoint() {
        // A small (≈3°) roll command in gentle flight: lateral acceleration stays
        // small, so the complementary filter's gravity assumption holds and the
        // craft tracks the command closely through the full estimator-in-the-loop
        // path. (Large sustained tilts would diverge — see the module note; that's
        // the MEKF's job.)
        let cfg = SimConfig::quad_250_mvp();
        let hover = cfg.hover_thrust();
        let mut sim = Sim::new(cfg);
        sim.set_setpoint(Setpoint {
            attitude: UnitQuaternion::from_euler_angles(0.05, 0.0, 0.0),
            thrust: hover,
        });
        sim.run_headless(1500);
        let (roll, _, _) = sim.truth().attitude.euler_angles();
        assert!((roll - 0.05).abs() < 0.03, "roll={roll}");
    }

    #[test]
    fn run_is_bit_for_bit_deterministic() {
        // Fingerprint the WHOLE recorded stream — truth, the estimate, and the
        // estimator's gyro-bias — so estimator-internal nondeterminism is caught
        // too, not just truth. Check both estimators (CF and MEKF).
        let fingerprint = |cfg: SimConfig| -> Vec<f64> {
            let mut sim = Sim::new(cfg);
            sim.set_setpoint(Setpoint {
                attitude: UnitQuaternion::from_euler_angles(0.1, -0.05, 0.2),
                thrust: cfg.hover_thrust(),
            });
            sim.run_headless(3000);
            let mut v = Vec::new();
            for s in &sim.telemetry().samples {
                v.extend_from_slice(s.truth.to_vector().as_slice());
                let q = s.estimate.attitude;
                v.extend_from_slice(&[q.w, q.i, q.j, q.k]);
                v.extend_from_slice(s.estimate.angular_rate.as_slice());
                v.extend_from_slice(s.est_gyro_bias.as_slice());
            }
            v
        };
        for cfg in [SimConfig::quad_250_mvp(), SimConfig::quad_250_m2()] {
            assert_eq!(
                fingerprint(cfg),
                fingerprint(cfg),
                "simulation is not deterministic"
            );
        }
    }

    #[test]
    fn estimator_tracks_truth_attitude_in_gentle_flight() {
        // In gentle flight (small tilt), the complementary filter tracks truth
        // attitude to within a couple of degrees the whole time.
        let cfg = SimConfig::quad_250_mvp();
        let hover = cfg.hover_thrust();
        let mut sim = Sim::new(cfg);
        sim.set_setpoint(Setpoint {
            attitude: UnitQuaternion::from_euler_angles(0.05, -0.03, 0.0),
            thrust: hover,
        });
        let mut max_err = 0.0_f64;
        for _ in 0..1500 {
            sim.step();
            max_err = max_err.max(sim.truth().attitude.angle_to(&sim.estimate().attitude));
        }
        assert!(max_err < 0.04, "estimator error grew to {max_err} rad");
        let _ = GRAVITY;
    }

    #[test]
    fn mekf_estimates_bias_and_beats_complementary_filter() {
        // Fair comparison: feed the SAME realistic biased-IMU + mag stream to both
        // estimators and compare attitude error vs a level/static truth. The CF
        // has no heading reference, so it integrates the gyro's yaw bias and
        // drifts; the MEKF estimates the bias (using the magnetometer) and stays
        // put. This is the headline MEKF result.
        use fsim_estimator::{ComplementaryFilter, Estimator, Mekf};
        use fsim_sensors::{Imu, Mag, Sensor, Truth};

        let cfg = SimConfig::quad_250_m2(); // realistic IMU carries a gyro bias
        let truth = State13::at_rest();
        let bundle = Truth {
            state: &truth,
            accel_world: Vec3::zeros(),
            t: 0.0,
        };

        let mut imu = Imu::new(cfg.imu, cfg.seed);
        let mut mag = Mag::new(cfg.mag, cfg.seed ^ 0x3333_3333);
        let mut cf = ComplementaryFilter::new(cfg.complementary);
        let mut mekf = Mekf::new(cfg.mekf);

        for k in 0..30_000 {
            let imu_meas = imu.sample(&bundle);
            cf.predict(&imu_meas, 1e-3);
            mekf.predict(&imu_meas, 1e-3);
            if k % 10 == 0 {
                let mag_meas = mag.sample(&bundle);
                cf.update_mag(&mag_meas); // no-op for the CF
                mekf.update_mag(&mag_meas);
            }
        }

        let cf_err = truth.attitude.angle_to(&cf.state().attitude);
        let mekf_err = truth.attitude.angle_to(&mekf.state().attitude);
        let true_bias = imu.gyro_bias();
        let bias_err = (mekf.gyro_bias_estimate().unwrap() - true_bias).norm();

        // The realistic IMU's bias also random-walks, so the MEKF tracks a
        // moving target; a few mrad/s residual is good tracking.
        assert!(bias_err < 7e-3, "MEKF bias estimate off by {bias_err}");
        assert!(
            mekf_err < cf_err * 0.5,
            "MEKF ({mekf_err:.4}) did not clearly beat CF ({cf_err:.4})"
        );
    }

    fn square_mission() -> Vec<Waypoint> {
        vec![
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 5.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(0.0, 5.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
        ]
    }

    #[test]
    fn ins_position_holds_at_altitude() {
        // Single waypoint 2 m up: climb to it and hold near it (INS + position
        // control, realistic GPS/baro noise, motor lag).
        let mut sim = Sim::new(SimConfig::quad_250_m3());
        sim.set_mission(
            vec![Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0)],
            GuidanceConfig::default(),
        );
        sim.run_headless(15000);
        let p = sim.truth().position;
        assert!(
            (p - Vec3::new(0.0, 0.0, -2.0)).norm() < 0.8,
            "did not hold position: {p:?}"
        );
    }

    #[test]
    fn ins_flies_square_mission_and_returns() {
        let mut sim = Sim::new(SimConfig::quad_250_m3());
        sim.set_mission(square_mission(), GuidanceConfig::default());
        // Track the worst INS position error after the initial GPS-fix transient.
        let mut max_track = 0.0_f64;
        for k in 0..40000 {
            sim.step();
            if k > 2000 {
                max_track = max_track.max((sim.truth().position - sim.estimate().position).norm());
            }
        }
        // Visited every waypoint (advanced to the last).
        assert_eq!(sim.waypoint_index(), Some(4), "mission not completed");
        // Returned to and settled near the final waypoint.
        let p = sim.truth().position;
        assert!(
            (p - Vec3::new(0.0, 0.0, -2.0)).norm() < 0.8,
            "did not settle at final wp: {p:?}"
        );
        // The INS tracked truth throughout (filters the 2.5 m GPS noise).
        assert!(max_track < 1.2, "INS position tracking error {max_track} m");
    }

    // The viewer flies map-scale routes (~100 m legs) at a brisker cruise than
    // the 5 m unit mission. Confirm the INS + position controller completes a
    // 100 m square at 6 m/s and settles at the final waypoint — so the viewer's
    // default "fly square mission" doesn't look broken at terrain scale.
    #[test]
    fn ins_flies_large_square_at_map_scale() {
        let alt = -50.0;
        let square = vec![
            Waypoint::new(Vec3::new(0.0, 0.0, alt), 0.0),
            Waypoint::new(Vec3::new(100.0, 0.0, alt), 0.0),
            Waypoint::new(Vec3::new(100.0, 100.0, alt), 0.0),
            Waypoint::new(Vec3::new(0.0, 100.0, alt), 0.0),
            Waypoint::new(Vec3::new(0.0, 0.0, alt), 0.0),
        ];
        let gcfg = GuidanceConfig {
            accept_radius: 5.0,
            cruise_speed: 6.0,
        };
        let mut sim = Sim::new(SimConfig::quad_250_m3());
        sim.set_mission(square, gcfg);
        // ~100 m per leg at 6 m/s ≈ 17 s/leg × 5 legs ≈ 90 s; give 150 s of margin.
        sim.run_headless(150_000);
        assert_eq!(
            sim.waypoint_index(),
            Some(4),
            "map-scale mission not completed"
        );
        let p = sim.truth().position;
        assert!(
            (p - Vec3::new(0.0, 0.0, alt)).norm() < 3.0,
            "did not settle at final wp at map scale: {p:?}"
        );
    }

    #[test]
    fn m3_mission_is_deterministic() {
        let fingerprint = || -> Vec<f64> {
            let mut sim = Sim::new(SimConfig::quad_250_m3());
            sim.set_mission(square_mission(), GuidanceConfig::default());
            sim.set_logging(50, None);
            sim.run_headless(20000);
            let mut v = Vec::new();
            for s in &sim.telemetry().samples {
                v.extend_from_slice(s.truth.to_vector().as_slice());
                v.extend_from_slice(s.estimate.position.as_slice());
                v.extend_from_slice(s.estimate.velocity.as_slice());
            }
            v
        };
        assert_eq!(
            fingerprint(),
            fingerprint(),
            "INS mission not deterministic"
        );
    }

    #[test]
    fn lqr_inner_controller_flies_mission() {
        // The LQR is a drop-in for the PID: with the INS + LQR inner loop, the
        // quad completes the square mission and settles at the final waypoint.
        let mut cfg = SimConfig::quad_250_m3();
        cfg.controller_kind = ControllerKind::Lqr;
        let mut sim = Sim::new(cfg);
        sim.set_mission(square_mission(), GuidanceConfig::default());
        sim.run_headless(40000);
        assert_eq!(sim.waypoint_index(), Some(4), "LQR mission not completed");
        let p = sim.truth().position;
        assert!(
            (p - Vec3::new(0.0, 0.0, -2.0)).norm() < 1.0,
            "LQR did not settle: {p:?}"
        );
    }

    #[test]
    fn mekf_in_the_loop_holds_attitude() {
        // The MEKF closes the loop end-to-end on the realistic-sensor config:
        // truth stays near level AND the estimate stays close to truth throughout
        // (the latter is what the autopilot actually flies on).
        let mut sim = Sim::new(SimConfig::quad_250_m2());
        let mut max_est_err = 0.0_f64;
        for _ in 0..5000 {
            sim.step();
            max_est_err = max_est_err.max(sim.truth().attitude.angle_to(&sim.estimate().attitude));
        }
        assert!(
            sim.truth().attitude.angle() < 0.08,
            "attitude drifted: {}",
            sim.truth().attitude.angle()
        );
        assert!(
            max_est_err < 0.08,
            "estimate error grew to {max_est_err} rad"
        );
    }

    // --- Weather (wind + turbulence) ---

    /// Hover the default quad through the given air for 5 s, returning the truth.
    fn quad_hover_in_weather(wind_ned: Vec3, turbulence: Real) -> State13 {
        let mut cfg = SimConfig::quad_250_mvp();
        cfg.atmosphere = crate::AtmosphereConfig::wind(wind_ned, turbulence);
        let mut sim = Sim::new(cfg);
        sim.run_headless(5000);
        *sim.truth()
    }

    // A steady wind drags the hovering quad downwind (attitude mode has no
    // position hold), carrying it toward the wind speed as the drag falls to zero.
    #[test]
    fn quad_drifts_downwind() {
        let calm = quad_hover_in_weather(Vec3::zeros(), 0.0);
        let windy = quad_hover_in_weather(Vec3::new(0.0, 6.0, 0.0), 0.0); // wind toward +East
        assert!(
            calm.position.norm() < 0.5,
            "calm hover should stay put: {}",
            calm.position.norm()
        );
        assert!(
            windy.position.y > 5.0,
            "a 6 m/s east wind should drift the quad east: {}",
            windy.position.y
        );
        assert!(
            (windy.velocity.y - 6.0).abs() < 2.0,
            "the quad should be carried toward the wind speed: {}",
            windy.velocity.y
        );
    }

    // Turbulence buffets the quad, but it rides it out — finite, upright, and not
    // blown away. (Linear drag applies no moment, so gusts perturb position more
    // than attitude — which the controller holds.)
    #[test]
    fn quad_survives_turbulence() {
        let s = quad_hover_in_weather(Vec3::zeros(), 3.0);
        assert!(
            s.position.iter().all(|x| x.is_finite()) && s.attitude.angle() < 0.3,
            "quad should ride out turbulence upright: att {} rad",
            s.attitude.angle()
        );
        assert!(
            s.position.norm() < 30.0,
            "should not be blown away: {} m",
            s.position.norm()
        );
    }

    // T-QuadWeatherDeterminism: wind + seeded turbulence is reproducible.
    #[test]
    fn quad_weather_is_deterministic() {
        let run = || quad_hover_in_weather(Vec3::new(-3.0, 2.0, 0.0), 3.0);
        let a = run();
        let b = run();
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.attitude, b.attitude);
        assert_eq!(a.angular_rate, b.angular_rate);
    }

    // --- Faults ---

    // A standard X-quad cannot hold attitude on three motors: kill a rotor and the
    // craft departs (the mixer can no longer produce the commanded torque).
    #[test]
    fn dead_rotor_upsets_the_quad() {
        let mut sim = Sim::new(SimConfig::quad_250_mvp());
        sim.run_headless(1000); // hover a moment
        assert!(
            sim.truth().attitude.angle() < 0.05,
            "level before the failure"
        );
        sim.set_faults(crate::QuadFaults {
            dead_rotor: Some(0),
        });
        assert!(sim.faulted());
        sim.run_headless(3000); // 3 s on three motors
        assert!(
            sim.truth().attitude.angle() > 0.5,
            "a dead rotor should upset the quad: {} rad",
            sim.truth().attitude.angle()
        );
    }
}
