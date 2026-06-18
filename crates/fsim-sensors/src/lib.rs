//! # fsim-sensors
//!
//! The whole point of the project: the autopilot never sees truth, only
//! deliberately-degraded measurements. Each [`Sensor`] reads the [`Truth`]
//! bundle and emits a noisy sample using its **own seeded** `ChaCha8Rng`, so
//! every run is bit-for-bit reproducible (never `thread_rng`).
//!
//! M1 ships the [`Imu`]; GPS, baro, and magnetometer slot in for M2 behind the
//! same [`Sensor`] trait without changing the scheduler.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod baro;
mod gps;
mod imu;
mod mag;

pub use baro::{Baro, BaroConfig};
pub use gps::{Gps, GpsConfig};
pub use imu::{Imu, ImuConfig};
pub use mag::{Mag, MagConfig};

use fsim_core::{Real, State13, Vec3};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Draw a zero-mean Gaussian scalar of std `std` (0 if `std <= 0`). Shared by
/// every sensor so they all consume RNG the same, reproducible way.
pub(crate) fn gaussian(rng: &mut ChaCha8Rng, std: Real) -> Real {
    if std <= 0.0 {
        return 0.0;
    }
    Normal::new(0.0, std).expect("std >= 0").sample(rng)
}

/// Draw a zero-mean Gaussian 3-vector with per-axis std `std`.
pub(crate) fn gaussian_vec3(rng: &mut ChaCha8Rng, std: Real) -> Vec3 {
    Vec3::new(gaussian(rng, std), gaussian(rng, std), gaussian(rng, std))
}

/// Everything a sensor may need to read from the simulator's truth at one
/// instant. The accelerometer needs world-frame acceleration (not part of
/// [`State13`]), so it travels alongside the state here.
#[derive(Debug, Clone, Copy)]
pub struct Truth<'a> {
    /// The true rigid-body state.
    pub state: &'a State13,
    /// True world-frame (NED) acceleration `d(velocity)/dt`, **including**
    /// gravity — equals the plant's `d_velocity`.
    pub accel_world: Vec3,
    /// Simulated time \[s\].
    pub t: Real,
}

/// A sensor that degrades truth into a measurement at a fixed rate.
pub trait Sensor {
    /// The measurement type this sensor produces.
    type Measurement;
    /// Nominal sampling rate \[Hz\] (the scheduler gates calls to this).
    fn rate_hz(&self) -> Real;
    /// Produce one measurement. `&mut self` because it advances internal RNG
    /// and bias-random-walk state.
    fn sample(&mut self, truth: &Truth<'_>) -> Self::Measurement;
}
