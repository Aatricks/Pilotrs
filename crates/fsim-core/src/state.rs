//! The 13-element rigid-body state and its time derivative, plus the
//! pack/unpack used by the RK4 integrator.

use crate::{Quat, Real, Vec3};
use nalgebra::{Quaternion, SVector, UnitQuaternion};

/// Dimension of the packed state vector: position(3) + velocity(3) +
/// quaternion(4) + body rate(3).
pub const STATE_DIM: usize = 13;

/// The full rigid-body state — the simulator's "truth".
///
/// Layout when packed (see [`State13::to_vector`]):
/// `[ px py pz | vx vy vz | qw qx qy qz | wx wy wz ]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct State13 {
    /// Position in the NED world frame \[m\].
    pub position: Vec3,
    /// Velocity in the NED world frame \[m/s\].
    pub velocity: Vec3,
    /// Attitude `q_{world<-body}` (Hamilton), rotates body -> world.
    pub attitude: Quat,
    /// Angular rate in the **body** frame \[rad/s\].
    pub angular_rate: Vec3,
}

impl State13 {
    /// A craft at rest at the origin, level (identity attitude).
    pub fn at_rest() -> Self {
        Self {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
            attitude: UnitQuaternion::identity(),
            angular_rate: Vec3::zeros(),
        }
    }

    /// Pack into a flat 13-vector for the integrator. Quaternion is stored
    /// scalar-first: `[w, x, y, z]`.
    pub fn to_vector(&self) -> SVector<Real, STATE_DIM> {
        let q = self.attitude.as_ref(); // underlying Quaternion
        SVector::<Real, STATE_DIM>::from_column_slice(&[
            self.position.x,
            self.position.y,
            self.position.z,
            self.velocity.x,
            self.velocity.y,
            self.velocity.z,
            q.w,
            q.i,
            q.j,
            q.k,
            self.angular_rate.x,
            self.angular_rate.y,
            self.angular_rate.z,
        ])
    }

    /// Reconstruct from a packed 13-vector, **renormalizing** the quaternion
    /// (RK4 combines push it slightly off the unit sphere every step).
    pub fn from_vector(v: &SVector<Real, STATE_DIM>) -> Self {
        Self {
            position: Vec3::new(v[0], v[1], v[2]),
            velocity: Vec3::new(v[3], v[4], v[5]),
            // Quaternion::new takes (w, i, j, k); from_quaternion normalizes.
            attitude: UnitQuaternion::from_quaternion(Quaternion::new(v[6], v[7], v[8], v[9])),
            angular_rate: Vec3::new(v[10], v[11], v[12]),
        }
    }

    /// Renormalize the attitude quaternion in place.
    pub fn renormalize(&mut self) {
        self.attitude.renormalize();
    }
}

impl Default for State13 {
    fn default() -> Self {
        Self::at_rest()
    }
}

/// Time derivative of [`State13`].
///
/// Note the attitude derivative is a *raw* [`Quaternion`] (not unit) — the
/// kinematic relation `q̇ = ½ q ⊗ [0, ω_body]` does not preserve unit norm,
/// which is exactly why [`State13::from_vector`] renormalizes after each step.
#[derive(Debug, Clone, Copy)]
pub struct StateDeriv {
    /// d(position)/dt = world velocity.
    pub d_position: Vec3,
    /// d(velocity)/dt = world acceleration.
    pub d_velocity: Vec3,
    /// d(attitude)/dt as a raw quaternion.
    pub d_attitude: Quaternion<Real>,
    /// d(angular_rate)/dt = body angular acceleration.
    pub d_angular_rate: Vec3,
}

impl StateDeriv {
    /// Pack into a flat 13-vector, matching [`State13::to_vector`]'s layout.
    pub fn to_vector(&self) -> SVector<Real, STATE_DIM> {
        SVector::<Real, STATE_DIM>::from_column_slice(&[
            self.d_position.x,
            self.d_position.y,
            self.d_position.z,
            self.d_velocity.x,
            self.d_velocity.y,
            self.d_velocity.z,
            self.d_attitude.w,
            self.d_attitude.i,
            self.d_attitude.j,
            self.d_attitude.k,
            self.d_angular_rate.x,
            self.d_angular_rate.y,
            self.d_angular_rate.z,
        ])
    }
}

/// Attitude kinematics: `q̇ = ½ · q ⊗ [0, ω_body]`.
///
/// `q` is `q_{world<-body}`, `omega_body` is the body-frame angular rate.
/// Returns the raw (non-unit) quaternion derivative.
#[inline]
pub fn attitude_kinematics(q: &Quat, omega_body: &Vec3) -> Quaternion<Real> {
    let omega_quat = Quaternion::new(0.0, omega_body.x, omega_body.y, omega_body.z);
    (q.as_ref() * omega_quat) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;
    use core::f64::consts::FRAC_PI_2;
    use nalgebra::UnitQuaternion;

    #[test]
    fn pack_unpack_round_trips() {
        let s = State13 {
            position: Vec3::new(1.0, -2.0, 3.0),
            velocity: Vec3::new(-0.5, 0.25, 10.0),
            attitude: UnitQuaternion::from_euler_angles(0.1, -0.2, 0.3),
            angular_rate: Vec3::new(0.01, -0.02, 0.03),
        };
        let back = State13::from_vector(&s.to_vector());
        assert!((back.position - s.position).norm() < 1e-12);
        assert!((back.velocity - s.velocity).norm() < 1e-12);
        assert!((back.angular_rate - s.angular_rate).norm() < 1e-12);
        // angle_to handles the quaternion double-cover.
        assert!(back.attitude.angle_to(&s.attitude) < 1e-12);
    }

    #[test]
    fn from_vector_renormalizes() {
        // A deliberately non-unit quaternion in the packed vector.
        let mut v = State13::at_rest().to_vector();
        v[6] = 2.0; // qw = 2
        let s = State13::from_vector(&v);
        assert!((s.attitude.as_ref().norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kinematics_yaw_rate_matches_finite_difference() {
        // Pure yaw rate about body z; integrate q̇ for a small dt and compare
        // to the closed-form rotation.
        let q0 = UnitQuaternion::identity();
        let omega = Vec3::new(0.0, 0.0, 1.0); // 1 rad/s yaw
        let dt = 1e-4;
        let qdot = attitude_kinematics(&q0, &omega);
        let q1 = UnitQuaternion::from_quaternion(*q0.as_ref() + qdot * dt);
        let expected = UnitQuaternion::from_euler_angles(0.0, 0.0, 1.0 * dt);
        assert!(q1.angle_to(&expected) < 1e-7);
    }

    #[test]
    fn attitude_rotates_body_to_world() {
        // 90° yaw: body +x (forward) should map to world +y (east) in NED.
        let q = UnitQuaternion::from_euler_angles(0.0, 0.0, FRAC_PI_2);
        let world = q * Vec3::new(1.0, 0.0, 0.0);
        assert!((world - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
    }
}
