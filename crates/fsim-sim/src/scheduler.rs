//! The fixed-step scheduler: one struct that owns every subsystem and advances
//! them deterministically.

use crate::config::SimConfig;
use crate::telemetry::{Telemetry, TelemetrySample};
use fsim_actuators::{Mixer, MotorModel, XQuadMixer};
use fsim_control::{CascadedPid, Controller};
use fsim_core::{CtrlCmd, EstState, Real, Setpoint, State13, Tick};
use fsim_dynamics::{aerodynamic_wrench, Integrator, MultirotorParams, Plant, RigidBody, Rk4};
use fsim_estimator::{ComplementaryFilter, Estimator};
use fsim_sensors::{Imu, Sensor, Truth};

/// The complete simulator. Concrete for M1 (swapping the complementary filter
/// for the MEKF, or PID for LQR, is a localized change here).
pub struct Sim {
    dt: Real,
    imu_period: Tick,
    control_period: Tick,
    imu_dt: Real,
    control_dt: Real,

    params: MultirotorParams,
    plant: RigidBody,
    mixer: XQuadMixer,
    motors: MotorModel,
    imu: Imu,
    estimator: ComplementaryFilter,
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
        let imu_period = (base_rate / cfg.imu_rate).round().max(1.0) as Tick;
        let control_period = (base_rate / cfg.control_rate).round().max(1.0) as Tick;
        let hover = cfg.hover_thrust();

        Self {
            dt: cfg.dt,
            imu_period,
            control_period,
            imu_dt: imu_period as Real * cfg.dt,
            control_dt: control_period as Real * cfg.dt,

            params: cfg.params,
            plant: RigidBody::new(cfg.params),
            mixer: XQuadMixer::new(cfg.arm_length, cfg.yaw_coeff, cfg.max_thrust),
            motors: MotorModel::new(cfg.motor_tau, cfg.max_thrust),
            imu: Imu::new(cfg.imu, cfg.seed),
            estimator: ComplementaryFilter::new(cfg.estimator),
            controller: CascadedPid::new(cfg.control),
            rk4: Rk4,

            truth: State13::at_rest(),
            setpoint: Setpoint::level(hover),
            cmd: CtrlCmd {
                thrust: hover,
                torque: fsim_core::Vec3::zeros(),
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

        // 2. IMU sample + estimator predict (at the IMU rate).
        if self.tick.is_multiple_of(self.imu_period) {
            let bundle = Truth {
                state: &self.truth,
                accel_world,
                t,
            };
            let imu_meas = self.imu.sample(&bundle);
            self.estimator.predict(&imu_meas, self.imu_dt);
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
        // Same config + same (no) guidance -> identical telemetry, twice.
        let run = || {
            let mut sim = Sim::new(SimConfig::quad_250_mvp());
            let sp = Setpoint {
                attitude: UnitQuaternion::from_euler_angles(0.1, -0.05, 0.2),
                thrust: SimConfig::quad_250_mvp().hover_thrust(),
            };
            sim.set_setpoint(sp);
            sim.run_headless(3000);
            sim.truth().to_vector()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "simulation is not deterministic");
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
}
