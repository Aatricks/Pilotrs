//! Force & moment aggregation: turn the actuator output `(collective thrust,
//! body torque)` plus the current state into the net [`Wrench`] on the body.

use crate::plant::MultirotorParams;
use fsim_core::{gravity_world, Real, State13, Vec3, Wrench};

/// Net external wrench from gravity, rotor thrust, and translational drag.
///
/// - **Gravity** acts along world `+z` (NED down): `m·g`.
/// - **Thrust** is collective, directed along body `-z` (FRD up) and rotated
///   into the world frame by the attitude.
/// - **Drag** is a simple linear model opposing world velocity.
/// - **Moment** is the mixer's commanded body torque (rotor differential).
pub fn aerodynamic_wrench(
    state: &State13,
    params: &MultirotorParams,
    collective_thrust: Real,
    body_torque: Vec3,
) -> Wrench {
    let weight = gravity_world() * params.mass;
    // Thrust pushes "up" = body -z; rotate body -> world.
    let thrust_world = state.attitude * Vec3::new(0.0, 0.0, -collective_thrust);
    let drag = state.velocity * (-params.drag_coeff);

    Wrench {
        force_world: weight + thrust_world + drag,
        moment_body: body_torque,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Integrator, Rk4};
    use crate::plant::{MultirotorParams, Plant, RigidBody};
    use fsim_core::{State13, Vec3, GRAVITY};

    fn step_n(
        body: &RigidBody,
        mut s: State13,
        thrust: Real,
        torque: Vec3,
        dt: Real,
        n: usize,
    ) -> State13 {
        let rk4 = Rk4;
        for _ in 0..n {
            s = rk4.step(
                &s,
                |x| body.deriv(x, &aerodynamic_wrench(x, &body.params, thrust, torque)),
                dt,
            );
        }
        s
    }

    #[test]
    fn free_fall_matches_closed_form() {
        // No thrust, no drag: z(t) = ½ g t² along NED +z (down).
        let params = MultirotorParams::diagonal(1.0, 1e-2, 1e-2, 2e-2, 0.0);
        let body = RigidBody::new(params);
        let dt = 1e-3;
        let n = 1000; // t = 1.0 s
        let s = step_n(&body, State13::at_rest(), 0.0, Vec3::zeros(), dt, n);
        let t = dt * n as Real;
        let expected = 0.5 * GRAVITY * t * t;
        assert!(
            (s.position.z - expected).abs() < 1e-6,
            "z={} expected={}",
            s.position.z,
            expected
        );
        assert!((s.velocity.z - GRAVITY * t).abs() < 1e-6);
    }

    #[test]
    fn hover_equilibrium_is_stationary() {
        // Thrust = m g, level, no rate -> the craft does not move.
        let params = MultirotorParams::quad_250();
        let body = RigidBody::new(params);
        let thrust = params.mass * GRAVITY;
        let s = step_n(&body, State13::at_rest(), thrust, Vec3::zeros(), 1e-3, 5000);
        assert!(s.position.norm() < 1e-6, "drifted: {}", s.position.norm());
        assert!(s.velocity.norm() < 1e-6);
    }

    #[test]
    fn torque_free_symmetric_spin_conserves_rate() {
        // Spherical inertia, spin about x, no moment -> ω constant, |L| constant.
        let params = MultirotorParams::diagonal(1.0, 5e-3, 5e-3, 5e-3, 0.0);
        let body = RigidBody::new(params);
        let mut s = State13::at_rest();
        s.angular_rate = Vec3::new(2.0, 0.0, 0.0);
        let out = step_n(&body, s, 0.0, Vec3::zeros(), 1e-3, 2000);
        assert!((out.angular_rate - s.angular_rate).norm() < 1e-9);
        assert!((out.attitude.as_ref().norm() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn asymmetric_body_conserves_angular_momentum() {
        // Distinct principal inertias, initial tumble, no moment: the body-frame
        // rate precesses (ω×Iω ≠ 0) but world-frame angular momentum |L| is
        // conserved.
        let params = MultirotorParams::diagonal(1.0, 2e-3, 5e-3, 9e-3, 0.0);
        let body = RigidBody::new(params);
        let mut s = State13::at_rest();
        s.angular_rate = Vec3::new(1.0, 2.0, 0.5);
        let l0 = (s.attitude * (params.inertia * s.angular_rate)).norm();
        let out = step_n(&body, s, 0.0, Vec3::zeros(), 5e-4, 4000);
        let l1 = (out.attitude * (params.inertia * out.angular_rate)).norm();
        assert!((l1 - l0).abs() / l0 < 1e-6, "L0={} L1={}", l0, l1);
        // And it actually precessed (sanity: the rate vector changed).
        assert!((out.angular_rate - s.angular_rate).norm() > 1e-3);
    }
}
