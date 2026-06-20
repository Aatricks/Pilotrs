//! Cascaded attitude + rate controller for a quadrotor.

use crate::{Controller, Pid};
use fsim_core::{CtrlCmd, EstState, Real, Setpoint, Vec3};

/// Gains and limits for the [`CascadedPid`] controller.
#[derive(Debug, Clone, Copy)]
pub struct CascadedConfig {
    /// Attitude-loop proportional gain per axis \[1/s\]: desired body rate =
    /// `att_p · attitude_error`.
    pub att_p: Vec3,
    /// Maximum commanded body rate per axis \[rad/s\] (attitude-loop output clamp).
    pub max_rate: Vec3,
    /// Rate-loop PID gains per axis (roll, pitch, yaw).
    pub rate_kp: Vec3,
    pub rate_ki: Vec3,
    pub rate_kd: Vec3,
    /// Anti-windup limit on each rate integral term \[N·m\].
    pub max_integral: Vec3,
    /// Output torque clamp per axis \[N·m\].
    pub max_torque: Vec3,
}

impl CascadedConfig {
    /// Tuned defaults for [`fsim_dynamics::MultirotorParams::quad_250`].
    pub fn quad_250() -> Self {
        Self {
            att_p: Vec3::new(8.0, 8.0, 4.0),
            max_rate: Vec3::new(10.0, 10.0, 5.0),
            rate_kp: Vec3::new(0.06, 0.06, 0.10),
            rate_ki: Vec3::new(0.04, 0.04, 0.06),
            rate_kd: Vec3::new(0.0015, 0.0015, 0.0),
            max_integral: Vec3::new(0.1, 0.1, 0.1),
            max_torque: Vec3::new(1.0, 1.0, 0.5),
        }
    }
}

/// Inner rate loop wrapped by an attitude loop. Collective thrust passes
/// through from the setpoint.
#[derive(Debug, Clone)]
pub struct CascadedPid {
    cfg: CascadedConfig,
    rate_pid: [Pid; 3],
}

impl CascadedPid {
    pub fn new(cfg: CascadedConfig) -> Self {
        let rate_pid = [
            Pid::new(
                cfg.rate_kp.x,
                cfg.rate_ki.x,
                cfg.rate_kd.x,
                cfg.max_integral.x,
                cfg.max_torque.x,
            ),
            Pid::new(
                cfg.rate_kp.y,
                cfg.rate_ki.y,
                cfg.rate_kd.y,
                cfg.max_integral.y,
                cfg.max_torque.y,
            ),
            Pid::new(
                cfg.rate_kp.z,
                cfg.rate_ki.z,
                cfg.rate_kd.z,
                cfg.max_integral.z,
                cfg.max_torque.z,
            ),
        ];
        Self { cfg, rate_pid }
    }

    /// Controller tuned to the default 250-class quad.
    pub fn quad_250() -> Self {
        Self::new(CascadedConfig::quad_250())
    }

    /// Attitude loop: body-frame attitude error -> desired body rate.
    fn desired_rate(&self, est: &EstState, sp: &Setpoint) -> Vec3 {
        // Error rotation expressed in the body frame: q_err = q_est⁻¹ · q_sp.
        let q_err = est.attitude.inverse() * sp.attitude;
        // Take the short way around the double cover (w >= 0).
        let q_err = if q_err.as_ref().w < 0.0 {
            nalgebra::UnitQuaternion::new_unchecked(-q_err.into_inner())
        } else {
            q_err
        };
        let e = q_err.scaled_axis(); // rotation vector (body frame)
        let rate = Vec3::new(
            self.cfg.att_p.x * e.x,
            self.cfg.att_p.y * e.y,
            self.cfg.att_p.z * e.z,
        );
        Vec3::new(
            rate.x.clamp(-self.cfg.max_rate.x, self.cfg.max_rate.x),
            rate.y.clamp(-self.cfg.max_rate.y, self.cfg.max_rate.y),
            rate.z.clamp(-self.cfg.max_rate.z, self.cfg.max_rate.z),
        )
    }
}

impl Controller for CascadedPid {
    fn step(&mut self, est: &EstState, sp: &Setpoint, dt: Real) -> CtrlCmd {
        let rate_sp = self.desired_rate(est, sp);
        let rate_err = rate_sp - est.angular_rate;
        let torque = Vec3::new(
            self.rate_pid[0].step(rate_err.x, dt),
            self.rate_pid[1].step(rate_err.y, dt),
            self.rate_pid[2].step(rate_err.z, dt),
        );
        CtrlCmd {
            thrust: sp.thrust,
            torque,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_actuators::{Mixer, MotorModel, XQuadMixer};
    use fsim_core::{EstState, Setpoint, State13, GRAVITY};
    use fsim_dynamics::{aerodynamic_wrench, Integrator, MultirotorParams, Plant, RigidBody, Rk4};
    use nalgebra::UnitQuaternion;

    fn est_from_truth(s: &State13) -> EstState {
        EstState {
            position: s.position,
            velocity: s.velocity,
            attitude: s.attitude,
            angular_rate: s.angular_rate,
        }
    }

    /// Fly the controller against the true plant with *perfect* feedback, and
    /// return the final state. This isolates the control law from estimator
    /// error.
    fn fly_to(sp: &Setpoint, seconds: Real) -> State13 {
        let params = MultirotorParams::quad_250();
        let body = RigidBody::new(params);
        let mixer = XQuadMixer::quad_250();
        let mut motors = MotorModel::ideal(4.0);
        let mut ctrl = CascadedPid::quad_250();
        let mut s = State13::at_rest();
        let dt = 1e-3;
        let n = (seconds / dt) as usize;
        for _ in 0..n {
            let est = est_from_truth(&s);
            let cmd = ctrl.step(&est, sp, dt);
            let motor_cmd = mixer.mix(&cmd);
            let actual = motors.update(&motor_cmd, dt);
            let achieved = mixer.collect(&actual);
            s = rk4_step(&body, &s, &achieved, dt);
        }
        s
    }

    fn rk4_step(body: &RigidBody, s: &State13, cmd: &CtrlCmd, dt: Real) -> State13 {
        let rk4 = Rk4;
        rk4.step(
            s,
            |x| {
                body.deriv(
                    x,
                    &aerodynamic_wrench(x, &body.params, cmd.thrust, cmd.torque, Vec3::zeros()),
                )
            },
            dt,
        )
    }

    #[test]
    fn converges_to_commanded_roll_with_zero_steady_state_error() {
        let thrust = MultirotorParams::quad_250().mass * GRAVITY;
        let sp = Setpoint {
            attitude: UnitQuaternion::from_euler_angles(0.175, 0.0, 0.0), // 10° roll
            thrust,
        };
        let s = fly_to(&sp, 4.0);
        let (roll, pitch, _yaw) = s.attitude.euler_angles();
        assert!((roll - 0.175).abs() < 1e-3, "roll={roll}");
        assert!(pitch.abs() < 1e-3, "pitch leaked: {pitch}");
        assert!(
            s.angular_rate.norm() < 1e-2,
            "not settled: {}",
            s.angular_rate.norm()
        );
    }

    #[test]
    fn holds_level_attitude() {
        let thrust = MultirotorParams::quad_250().mass * GRAVITY;
        let sp = Setpoint::level(thrust);
        let s = fly_to(&sp, 2.0);
        assert!(s.attitude.angle() < 1e-3, "drifted off level");
    }

    #[test]
    fn converges_to_combined_attitude_setpoint() {
        let thrust = MultirotorParams::quad_250().mass * GRAVITY;
        let sp = Setpoint {
            attitude: UnitQuaternion::from_euler_angles(-0.12, 0.20, 0.30),
            thrust,
        };
        let s = fly_to(&sp, 5.0);
        let (roll, pitch, yaw) = s.attitude.euler_angles();
        assert!((roll + 0.12).abs() < 2e-3, "roll={roll}");
        assert!((pitch - 0.20).abs() < 2e-3, "pitch={pitch}");
        assert!((yaw - 0.30).abs() < 2e-3, "yaw={yaw}");
    }
}
