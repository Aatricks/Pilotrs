//! The fixed-step scheduler: one struct that owns every subsystem and advances
//! them deterministically.

use crate::config::{EstimatorKind, SimConfig};
use crate::telemetry::{Telemetry, TelemetrySample};
use fsim_actuators::{Mixer, MotorModel, XQuadMixer};
use fsim_control::{CascadedPid, Controller};
use fsim_core::{CtrlCmd, EstState, Real, Setpoint, State13, Tick, Vec3};
use fsim_dynamics::{aerodynamic_wrench, Integrator, MultirotorParams, Plant, RigidBody, Rk4};
use fsim_estimator::{ComplementaryFilter, Estimator, Mekf};
use fsim_sensors::{Baro, Gps, Imu, Mag, Sensor, Truth};

/// The complete simulator. The estimator is boxed so it can be swapped (CF or
/// MEKF) per [`SimConfig`]; everything else is concrete.
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
    controller: CascadedPid,
    rk4: Rk4,

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
            controller: CascadedPid::new(cfg.control),
            rk4: Rk4,

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

    /// Set the active attitude/thrust setpoint (the viewer calls this live).
    pub fn set_setpoint(&mut self, sp: Setpoint) {
        self.setpoint = sp;
    }

    /// Simulated time \[s\].
    pub fn time(&self) -> Real {
        self.tick as Real * self.dt
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

        // 1. True acceleration acting right now, from the motor thrust that was
        //    in effect over the just-completed interval. The accelerometer
        //    needs this (it isn't part of State13).
        let held = self.mixer.collect(&self.motors.thrust());
        let accel_world = aerodynamic_wrench(&self.truth, &self.params, held.thrust, held.torque)
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
        if self.tick.is_multiple_of(self.control_period) {
            self.cmd =
                self.controller
                    .step(&self.estimator.state(), &self.setpoint, self.control_dt);
        }

        // 4. Allocate to motors and apply the motor model.
        let motor_cmd = self.mixer.mix(&self.cmd);
        let actual = self.motors.update(&motor_cmd, self.dt);
        let achieved = self.mixer.collect(&actual);

        // 5. Integrate truth one step (copies avoid borrowing `self` in the
        //    RK4 derivative closure).
        let plant = self.plant;
        let params = self.params;
        let (thrust, torque) = (achieved.thrust, achieved.torque);
        self.truth = self.rk4.step(
            &self.truth,
            |x| plant.deriv(x, &aerodynamic_wrench(x, &params, thrust, torque)),
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
        // the M2 MEKF's job.)
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
        // put. This is the headline M2 result.
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
}
