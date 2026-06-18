//! LQR attitude/rate controller — an optimal state-feedback drop-in for the
//! cascaded PID, behind the same [`Controller`] trait.
//!
//! ## Model and per-axis decoupling
//!
//! Linearizing the rigid-body rotational dynamics about hover (dropping the
//! second-order gyroscopic `ω×Iω`), and with the **diagonal** quad inertia, the
//! attitude/rate dynamics decouple into three independent per-axis double
//! integrators. Per axis, with state `x = [e_att; ω]`, input `τ`, and
//! `ω̇ = τ/I`:
//!
//! ```text
//!   ė_att = −ω          (the error-to-setpoint rotation vector shrinks as the
//!   ω̇    = (1/I)·τ       body rotates toward it)
//! ```
//!
//! ## Closed-form LQR
//!
//! Minimizing `J = ∫ (q_att·e² + q_rate·ω² + r·τ²) dt`, the per-axis 2×2
//! continuous algebraic Riccati equation has the closed-form solution
//!
//! ```text
//!   K_att  = √(q_att / r)
//!   K_rate = √((2·I·√(q_att·r) + q_rate) / r)
//!   τ      = K_att·e_att − K_rate·ω      (both gains > 0)
//! ```
//!
//! The control law's sign matches the PID's: a positive attitude error drives a
//! positive torque (toward the setpoint), a positive rate drives a negative
//! (damping) torque. The closed loop is `s² + (K_rate/I)·s + K_att/I`, i.e.
//! `ωₙ = √(K_att/I)`, `ζ = K_rate / (2√(I·K_att))` — Hurwitz for positive gains.

use crate::Controller;
use fsim_core::{CtrlCmd, EstState, Real, Setpoint, Vec3};
use nalgebra::UnitQuaternion;
use num_traits::Float;

/// Cost weights + limits for the [`LqrController`] (per body axis).
#[derive(Debug, Clone, Copy)]
pub struct LqrConfig {
    /// Attitude-error weight per axis.
    pub q_att: Vec3,
    /// Body-rate weight per axis.
    pub q_rate: Vec3,
    /// Control (torque) weight per axis.
    pub r_torque: Vec3,
    /// Output torque clamp per axis \[N·m\].
    pub max_torque: Vec3,
}

impl LqrConfig {
    /// Defaults for the 250-class quad: ~12.5 rad/s roll/pitch and ~7.6 rad/s
    /// yaw bandwidth at ζ≈0.9 — comparable to (slightly crisper than) the PID.
    pub fn quad_250() -> Self {
        Self {
            q_att: Vec3::new(0.25, 0.25, 0.1),
            q_rate: Vec3::new(0.002, 0.002, 0.003),
            r_torque: Vec3::new(1.0, 1.0, 1.0),
            max_torque: Vec3::new(1.0, 1.0, 0.5),
        }
    }
}

/// LQR attitude/rate controller. Precomputes the per-axis gains; the hot path is
/// a single state-feedback evaluation.
#[derive(Debug, Clone)]
pub struct LqrController {
    k_att: Vec3,
    k_rate: Vec3,
    max_torque: Vec3,
}

impl LqrController {
    /// Build from the body inertia diagonal `(Ixx, Iyy, Izz)` and cost weights.
    pub fn new(inertia_diag: Vec3, cfg: LqrConfig) -> Self {
        let mut k_att = Vec3::zeros();
        let mut k_rate = Vec3::zeros();
        for i in 0..3 {
            let (q1, q2, r, inertia) = (
                cfg.q_att[i],
                cfg.q_rate[i],
                cfg.r_torque[i],
                inertia_diag[i],
            );
            // Closed-form 2×2 Riccati (see module docs).
            k_att[i] = Float::sqrt(q1 / r);
            let p2 = inertia * Float::sqrt(q1 * r);
            k_rate[i] = Float::sqrt((2.0 * p2 + q2) / r);
        }
        Self {
            k_att,
            k_rate,
            max_torque: cfg.max_torque,
        }
    }

    /// The precomputed gains (for inspection/tests): `(K_att, K_rate)`.
    pub fn gains(&self) -> (Vec3, Vec3) {
        (self.k_att, self.k_rate)
    }
}

impl Controller for LqrController {
    fn step(&mut self, est: &EstState, sp: &Setpoint, _dt: Real) -> CtrlCmd {
        // Body-frame attitude error rotation vector (short way), exactly as the
        // cascaded controller computes it.
        let q_err = est.attitude.inverse() * sp.attitude;
        let q_err = if q_err.as_ref().w < 0.0 {
            UnitQuaternion::new_unchecked(-q_err.into_inner())
        } else {
            q_err
        };
        let e = q_err.scaled_axis();
        let w = est.angular_rate;

        let torque = Vec3::new(
            (self.k_att.x * e.x - self.k_rate.x * w.x).clamp(-self.max_torque.x, self.max_torque.x),
            (self.k_att.y * e.y - self.k_rate.y * w.y).clamp(-self.max_torque.y, self.max_torque.y),
            (self.k_att.z * e.z - self.k_rate.z * w.z).clamp(-self.max_torque.z, self.max_torque.z),
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

    fn quad_lqr() -> LqrController {
        let p = MultirotorParams::quad_250();
        let diag = Vec3::new(p.inertia[(0, 0)], p.inertia[(1, 1)], p.inertia[(2, 2)]);
        LqrController::new(diag, LqrConfig::quad_250())
    }

    fn est(s: &State13) -> EstState {
        EstState {
            position: s.position,
            velocity: s.velocity,
            attitude: s.attitude,
            angular_rate: s.angular_rate,
        }
    }

    #[test]
    fn gains_are_positive_and_stable() {
        // Read the inertia from the same params used to build the controller, and
        // verify the closed-loop ωn/ζ on EVERY axis (yaw differs from roll/pitch).
        let p = MultirotorParams::quad_250();
        let inertia = [p.inertia[(0, 0)], p.inertia[(1, 1)], p.inertia[(2, 2)]];
        let (ka, kr) = quad_lqr().gains();
        let wn_band = [(10.0, 15.0), (10.0, 15.0), (6.0, 10.0)]; // roll, pitch, yaw
        for i in 0..3 {
            assert!(ka[i] > 0.0 && kr[i] > 0.0, "axis {i}: non-positive gain");
            let wn = (ka[i] / inertia[i]).sqrt(); // √(K_att/I)
            let zeta = kr[i] / (2.0 * (inertia[i] * ka[i]).sqrt());
            let (lo, hi) = wn_band[i];
            assert!(
                (lo..hi).contains(&wn),
                "axis {i} ωn {wn} out of [{lo},{hi}]"
            );
            assert!(
                (0.7..1.1).contains(&zeta),
                "axis {i} ζ {zeta} out of [0.7,1.1]"
            );
        }
    }

    #[test]
    fn positive_roll_error_drives_positive_torque() {
        let mut c = quad_lqr();
        let e = est(&State13::at_rest()); // identity attitude, zero rate
        let sp = Setpoint {
            attitude: UnitQuaternion::from_euler_angles(0.1, 0.0, 0.0), // want +roll
            thrust: 4.9,
        };
        assert!(
            c.step(&e, &sp, 1e-3).torque.x > 0.0,
            "should rotate toward setpoint"
        );
    }

    #[test]
    fn positive_rate_damps() {
        let mut c = quad_lqr();
        let mut s = State13::at_rest();
        s.angular_rate = Vec3::new(1.0, 0.0, 0.0); // spinning +roll, no error
        let sp = Setpoint::level(4.9);
        assert!(
            c.step(&est(&s), &sp, 1e-3).torque.x < 0.0,
            "should damp the rate"
        );
    }

    #[test]
    fn converges_to_attitude_setpoint_against_plant() {
        // Closed loop against the true plant (perfect feedback): the LQR drives
        // a combined roll/pitch/yaw setpoint to zero steady-state error.
        let params = MultirotorParams::quad_250();
        let body = RigidBody::new(params);
        let mixer = XQuadMixer::quad_250();
        let mut motors = MotorModel::ideal(4.0);
        let mut ctrl = quad_lqr();
        let sp = Setpoint {
            attitude: UnitQuaternion::from_euler_angles(0.15, -0.1, 0.25),
            thrust: params.mass * GRAVITY,
        };
        let mut s = State13::at_rest();
        let rk4 = Rk4;
        for _ in 0..5000 {
            let cmd = ctrl.step(&est(&s), &sp, 1e-3);
            let motor_cmd = mixer.mix(&cmd);
            let actual = motors.update(&motor_cmd, 1e-3);
            let achieved = mixer.collect(&actual);
            s = rk4.step(
                &s,
                |x| {
                    body.deriv(
                        x,
                        &aerodynamic_wrench(x, &params, achieved.thrust, achieved.torque),
                    )
                },
                1e-3,
            );
        }
        let (roll, pitch, yaw) = s.attitude.euler_angles();
        assert!((roll - 0.15).abs() < 2e-3, "roll={roll}");
        assert!((pitch + 0.1).abs() < 2e-3, "pitch={pitch}");
        assert!((yaw - 0.25).abs() < 2e-3, "yaw={yaw}");
        assert!(s.angular_rate.norm() < 1e-2, "not settled");
    }
}
