//! GPS receiver model: low-rate NED position + velocity with larger noise than
//! the IMU and an optional slowly-drifting position bias.

use crate::{gaussian_vec3, Sensor, Truth};
use fsim_core::{GpsMeas, Real, Vec3};
use num_traits::Float;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Noise/bias parameters for a [`Gps`].
#[derive(Debug, Clone, Copy)]
pub struct GpsConfig {
    /// Fix rate \[Hz\].
    pub rate_hz: Real,
    /// Position white-noise std per axis \[m\].
    pub pos_noise_std: Real,
    /// Velocity white-noise std per axis \[m/s\].
    pub vel_noise_std: Real,
    /// Position bias random-walk intensity \[m / √s\] (slow drift).
    pub pos_bias_walk: Real,
}

impl GpsConfig {
    /// Light, well-behaved GPS for the MVP.
    pub fn mvp(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            pos_noise_std: 1.0,
            vel_noise_std: 0.1,
            pos_bias_walk: 0.0,
        }
    }

    /// Realistic consumer GPS with a wandering position bias.
    pub fn realistic(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            pos_noise_std: 2.5,
            vel_noise_std: 0.25,
            pos_bias_walk: 0.05,
        }
    }
}

/// A simulated GPS receiver with its own deterministic RNG stream.
#[derive(Debug, Clone)]
pub struct Gps {
    cfg: GpsConfig,
    rng: ChaCha8Rng,
    pos_bias: Vec3,
}

impl Gps {
    pub fn new(cfg: GpsConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
            pos_bias: Vec3::zeros(),
        }
    }
}

impl Sensor for Gps {
    type Measurement = GpsMeas;

    fn rate_hz(&self) -> Real {
        self.cfg.rate_hz
    }

    fn sample(&mut self, truth: &Truth<'_>) -> GpsMeas {
        let dt = 1.0 / self.cfg.rate_hz;
        self.pos_bias += gaussian_vec3(&mut self.rng, self.cfg.pos_bias_walk * Float::sqrt(dt));
        GpsMeas {
            position: truth.state.position
                + self.pos_bias
                + gaussian_vec3(&mut self.rng, self.cfg.pos_noise_std),
            velocity: truth.state.velocity + gaussian_vec3(&mut self.rng, self.cfg.vel_noise_std),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::{State13, Vec3};

    fn truth(state: &State13) -> Truth<'_> {
        Truth {
            state,
            accel_world: Vec3::zeros(),
            t: 0.0,
        }
    }

    #[test]
    fn same_seed_reproducible() {
        let mut a = Gps::new(GpsConfig::realistic(5.0), 11);
        let mut b = Gps::new(GpsConfig::realistic(5.0), 11);
        let s = State13::at_rest();
        for _ in 0..50 {
            assert_eq!(a.sample(&truth(&s)).position, b.sample(&truth(&s)).position);
        }
    }

    #[test]
    fn unbiased_mean_tracks_truth() {
        let cfg = GpsConfig {
            pos_bias_walk: 0.0,
            ..GpsConfig::mvp(5.0)
        };
        let mut g = Gps::new(cfg, 5);
        let mut s = State13::at_rest();
        s.position = Vec3::new(10.0, -5.0, -2.0);
        let n = 5000;
        let mut mean = Vec3::zeros();
        for _ in 0..n {
            mean += g.sample(&truth(&s)).position;
        }
        mean /= n as Real;
        assert!((mean - s.position).norm() < 0.05, "mean={mean:?}");
    }
}
