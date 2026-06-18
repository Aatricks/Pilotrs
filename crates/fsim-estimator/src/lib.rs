//! # fsim-estimator
//!
//! Fuses noisy sensors back into a best estimate of the state — the module the
//! autopilot actually acts on. M1 ships a [`ComplementaryFilter`] (a
//! Mahony-style explicit complementary filter on attitude); M2 replaces it with
//! a quaternion Multiplicative EKF behind the same [`Estimator`] trait.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod complementary;
mod mekf;

pub use complementary::{ComplementaryConfig, ComplementaryFilter};
pub use mekf::{Mekf, MekfConfig};

use fsim_core::{BaroMeas, EstState, GpsMeas, ImuMeas, MagMeas, Real, Vec3};

/// A state estimator.
///
/// `predict` propagates with the IMU (gyro, plus the accelerometer's gravity
/// reference for an AHRS). The measurement updates default to no-ops so an
/// attitude-only estimator (the complementary filter) need not implement them;
/// the MEKF overrides the ones it uses.
pub trait Estimator {
    /// Propagate the estimate forward one IMU step.
    fn predict(&mut self, imu: &ImuMeas, dt: Real);

    /// Correct heading/attitude with a magnetometer sample.
    fn update_mag(&mut self, _mag: &MagMeas) {}

    /// Correct position/velocity with a GPS fix (used by the M3 INS).
    fn update_gps(&mut self, _gps: &GpsMeas) {}

    /// Correct altitude with a barometer sample (used by the M3 INS).
    fn update_baro(&mut self, _baro: &BaroMeas) {}

    /// The current best estimate (what the controller consumes).
    fn state(&self) -> EstState;

    /// The estimator's gyro-bias estimate, if it has one (the MEKF does; the
    /// complementary filter does not). For diagnostics/plots only.
    fn gyro_bias_estimate(&self) -> Option<Vec3> {
        None
    }
}
