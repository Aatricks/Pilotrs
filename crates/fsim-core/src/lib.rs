//! # fsim-core
//!
//! The shared contract for every Pilotrs crate: the state vector, the
//! reference-frame and quaternion conventions, physical constants, and the
//! message types that flow between subsystems. Define them **once, here** —
//! half of all attitude bugs come from convention drift between modules.
//!
//! ## Reference frames
//!
//! - **World frame: NED** (North, East, Down). Gravity points along world
//!   `+z`; altitude is `-z`. This matches the autopilot literature
//!   (PX4 / ArduPilot) you'll cross-check controllers against.
//! - **Body frame: FRD** (Forward, Right, Down), origin at the center of
//!   gravity. Roll is about body `x`, pitch about body `y`, yaw about body `z`.
//!
//! ## Attitude convention
//!
//! The attitude quaternion is `q_{world<-body}` (Hamilton convention, stored
//! as [`nalgebra::UnitQuaternion`]). It rotates a vector expressed in the body
//! frame into the world frame: `v_world = q * v_body`. It is renormalized every
//! integrator step.
//!
//! The body angular rate `omega` is expressed in the **body frame** — that is
//! what a gyro measures and what Euler's rotational equation uses.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod messages;
pub mod planet;
mod state;

pub use messages::{
    BaroMeas, ControlLimits, CtrlCmd, EstState, FixedWingControls, GpsMeas, ImuMeas, MagMeas,
    Setpoint, StickInput, Wrench,
};
pub use state::{attitude_kinematics, State13, StateDeriv, STATE_DIM};

use nalgebra::{UnitQuaternion, Vector3};

/// Scalar type used throughout the simulator. `f64` for trajectory accuracy.
pub type Real = f64;

/// A 3-vector of [`Real`] (position, velocity, rate, force, ...).
pub type Vec3 = Vector3<Real>;

/// A unit quaternion attitude, `q_{world<-body}`.
pub type Quat = UnitQuaternion<Real>;

/// Standard gravity magnitude \[m/s^2\].
pub const GRAVITY: Real = 9.80665;

/// Default physics step: 1 kHz. The integrator and scheduler are built around
/// a *fixed* `dt`; wall-clock time never enters the math (determinism).
pub const DEFAULT_DT: Real = 1.0 / 1000.0;

/// Monotonic physics step counter. Simulated time is `tick as Real * dt`,
/// never `Instant::now()`.
pub type Tick = u64;

/// Gravity as a world-frame (NED) acceleration vector: points along `+z` (down).
#[inline]
pub fn gravity_world() -> Vec3 {
    Vec3::new(0.0, 0.0, GRAVITY)
}

/// Reference geomagnetic field direction in the NED world frame (unit vector).
///
/// Mid-latitude-ish: zero declination, ~60° inclination (field dips below
/// horizontal, pointing North and Down). The **same** reference is used by the
/// magnetometer sensor model and the estimator's mag update, so they cannot
/// disagree on the field — a convention bug we deliberately design out.
#[inline]
pub fn magnetic_field_ned() -> Vec3 {
    // (cos 60°, 0, sin 60°) is already unit length.
    Vec3::new(0.5, 0.0, 0.866_025_403_784_438_6)
}
