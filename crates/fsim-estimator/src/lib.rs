//! # fsim-estimator
//!
//! Fuses noisy sensors back into a best estimate of the state — the module the
//! autopilot actually acts on. M1 ships a [`ComplementaryFilter`] (a
//! Mahony-style explicit complementary filter on attitude); M2 replaces it with
//! a quaternion Multiplicative EKF behind the same [`Estimator`] trait.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod complementary;

pub use complementary::{ComplementaryConfig, ComplementaryFilter};

use fsim_core::{EstState, ImuMeas, Real};

/// A state estimator. The predict step propagates with the IMU; measurement
/// updates (GPS/baro/mag) arrive in M2.
pub trait Estimator {
    /// Propagate the estimate forward one IMU step.
    fn predict(&mut self, imu: &ImuMeas, dt: Real);
    /// The current best estimate (what the controller consumes).
    fn state(&self) -> EstState;
}
