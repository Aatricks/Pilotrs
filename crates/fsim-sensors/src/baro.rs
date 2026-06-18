//! Barometric altimeter model: altitude (`-z`) with a slowly-drifting pressure
//! bias and white noise.

use crate::{gaussian, Sensor, Truth};
use fsim_core::{BaroMeas, Real};
use num_traits::Float;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Noise/bias parameters for a [`Baro`].
#[derive(Debug, Clone, Copy)]
pub struct BaroConfig {
    /// Sample rate \[Hz\].
    pub rate_hz: Real,
    /// White-noise std \[m\].
    pub noise_std: Real,
    /// Bias random-walk intensity \[m / √s\] (pressure drift).
    pub bias_walk: Real,
}

impl BaroConfig {
    /// Light barometer for the MVP.
    pub fn mvp(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            noise_std: 0.3,
            bias_walk: 0.0,
        }
    }

    /// Realistic barometer with a drifting bias.
    pub fn realistic(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            noise_std: 0.6,
            bias_walk: 0.05,
        }
    }
}

/// A simulated barometer with its own deterministic RNG stream.
#[derive(Debug, Clone)]
pub struct Baro {
    cfg: BaroConfig,
    rng: ChaCha8Rng,
    bias: Real,
}

impl Baro {
    pub fn new(cfg: BaroConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
            bias: 0.0,
        }
    }
}

impl Sensor for Baro {
    type Measurement = BaroMeas;

    fn rate_hz(&self) -> Real {
        self.cfg.rate_hz
    }

    fn sample(&mut self, truth: &Truth<'_>) -> BaroMeas {
        let dt = 1.0 / self.cfg.rate_hz;
        self.bias += gaussian(&mut self.rng, self.cfg.bias_walk * Float::sqrt(dt));
        // Altitude is +up = -z in NED.
        BaroMeas {
            altitude: -truth.state.position.z
                + self.bias
                + gaussian(&mut self.rng, self.cfg.noise_std),
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
    fn altitude_is_negative_z() {
        let cfg = BaroConfig {
            noise_std: 0.0,
            bias_walk: 0.0,
            ..BaroConfig::mvp(25.0)
        };
        let mut b = Baro::new(cfg, 1);
        let mut s = State13::at_rest();
        s.position.z = -3.0; // 3 m up
        assert!((b.sample(&truth(&s)).altitude - 3.0).abs() < 1e-12);
    }

    #[test]
    fn same_seed_reproducible() {
        let mut a = Baro::new(BaroConfig::realistic(25.0), 9);
        let mut b = Baro::new(BaroConfig::realistic(25.0), 9);
        let s = State13::at_rest();
        for _ in 0..50 {
            assert_eq!(a.sample(&truth(&s)).altitude, b.sample(&truth(&s)).altitude);
        }
    }
}
