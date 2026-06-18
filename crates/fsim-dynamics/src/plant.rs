//! The quadrotor rigid body and its Newton-Euler equations of motion.

use fsim_core::{attitude_kinematics, Real, State13, StateDeriv, Vec3, Wrench};
use nalgebra::Matrix3;

/// Mass and inertia of a multirotor. Inertia is expressed in the body frame;
/// for a symmetric quad it is diagonal.
#[derive(Debug, Clone, Copy)]
pub struct MultirotorParams {
    /// Total mass \[kg\].
    pub mass: Real,
    /// Body-frame inertia tensor \[kg·m^2\].
    pub inertia: Matrix3<Real>,
    /// Precomputed inverse inertia.
    pub inertia_inv: Matrix3<Real>,
    /// Simple linear translational drag coefficient \[N·s/m\]
    /// (drag force = `-drag_coeff · velocity_world`).
    pub drag_coeff: Real,
}

impl MultirotorParams {
    /// Build from a diagonal inertia (the usual symmetric-quad case).
    pub fn diagonal(mass: Real, ixx: Real, iyy: Real, izz: Real, drag_coeff: Real) -> Self {
        let inertia = Matrix3::from_diagonal(&Vec3::new(ixx, iyy, izz));
        let inertia_inv = Matrix3::from_diagonal(&Vec3::new(1.0 / ixx, 1.0 / iyy, 1.0 / izz));
        Self {
            mass,
            inertia,
            inertia_inv,
            drag_coeff,
        }
    }

    /// A small ~0.5 kg research quad (250-class), sensible defaults.
    pub fn quad_250() -> Self {
        // Roll/pitch inertia ~ equal, yaw inertia ~ twice (typical for a quad).
        Self::diagonal(0.5, 3.2e-3, 3.2e-3, 5.5e-3, 0.10)
    }
}

/// The rigid-body plant. Stateless: it maps `(state, wrench)` to a derivative.
#[derive(Debug, Clone, Copy)]
pub struct RigidBody {
    pub params: MultirotorParams,
}

impl RigidBody {
    pub fn new(params: MultirotorParams) -> Self {
        Self { params }
    }
}

/// A 6DOF rigid body: F = m·a (translation) and I·ω̇ + ω×Iω = M (rotation).
pub trait Plant {
    /// Evaluate the state derivative given the net external wrench.
    fn deriv(&self, state: &State13, wrench: &Wrench) -> StateDeriv;
}

impl Plant for RigidBody {
    fn deriv(&self, state: &State13, wrench: &Wrench) -> StateDeriv {
        let p = &self.params;

        // Translation (world frame): a = F / m.
        let d_velocity = wrench.force_world / p.mass;

        // Attitude kinematics: q̇ = ½ q ⊗ [0, ω_body].
        let d_attitude = attitude_kinematics(&state.attitude, &state.angular_rate);

        // Rotation (body frame): ω̇ = I⁻¹ (M − ω × Iω).  The gyroscopic term
        // ω×Iω is what couples the axes for an asymmetric body.
        let omega = state.angular_rate;
        let gyroscopic = omega.cross(&(p.inertia * omega));
        let d_angular_rate = p.inertia_inv * (wrench.moment_body - gyroscopic);

        StateDeriv {
            d_position: state.velocity,
            d_velocity,
            d_attitude,
            d_angular_rate,
        }
    }
}
