//! Magnetometer model: the world geomagnetic reference field rotated into the
//! body frame, plus white noise and an optional constant hard-iron bias.
//!
//! `m_body = R(q)^T · m_world_ref` — i.e. `q.inverse() * magnetic_field_ned()`.
//! This is the heading reference the accelerometer (gravity only) can't give.

use crate::{gaussian_vec3, Sensor, Truth};
use fsim_core::{magnetic_field_ned, MagMeas, Real, Vec3};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Noise/bias parameters for a [`Mag`].
#[derive(Debug, Clone, Copy)]
pub struct MagConfig {
    /// Sample rate \[Hz\].
    pub rate_hz: Real,
    /// White-noise std per axis (units of the unit reference field).
    pub noise_std: Real,
    /// Constant hard-iron bias in the body frame.
    pub hard_iron: Vec3,
}

impl MagConfig {
    /// Light magnetometer for the MVP.
    pub fn mvp(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            noise_std: 0.01,
            hard_iron: Vec3::zeros(),
        }
    }

    /// Realistic magnetometer with a small hard-iron offset.
    pub fn realistic(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            noise_std: 0.02,
            hard_iron: Vec3::new(0.02, -0.015, 0.01),
        }
    }
}

/// A simulated magnetometer with its own deterministic RNG stream.
#[derive(Debug, Clone)]
pub struct Mag {
    cfg: MagConfig,
    rng: ChaCha8Rng,
}

impl Mag {
    pub fn new(cfg: MagConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Sensor for Mag {
    type Measurement = MagMeas;

    fn rate_hz(&self) -> Real {
        self.cfg.rate_hz
    }

    fn sample(&mut self, truth: &Truth<'_>) -> MagMeas {
        let body = truth.state.attitude.inverse() * magnetic_field_ned();
        MagMeas {
            field: body + self.cfg.hard_iron + gaussian_vec3(&mut self.rng, self.cfg.noise_std),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::{State13, Vec3};
    use nalgebra::UnitQuaternion;

    fn truth(state: &State13) -> Truth<'_> {
        Truth {
            state,
            accel_world: Vec3::zeros(),
            t: 0.0,
        }
    }

    #[test]
    fn level_reads_reference_field() {
        let cfg = MagConfig {
            noise_std: 0.0,
            hard_iron: Vec3::zeros(),
            ..MagConfig::mvp(50.0)
        };
        let mut m = Mag::new(cfg, 1);
        let s = State13::at_rest(); // identity attitude
        assert!((m.sample(&truth(&s)).field - magnetic_field_ned()).norm() < 1e-12);
    }

    #[test]
    fn yaw_rotates_horizontal_component() {
        // 90° yaw: the North-pointing horizontal component should rotate into
        // the body -y axis (a heading signal the accelerometer can't provide).
        let cfg = MagConfig {
            noise_std: 0.0,
            hard_iron: Vec3::zeros(),
            ..MagConfig::mvp(50.0)
        };
        let mut m = Mag::new(cfg, 2);
        let mut s = State13::at_rest();
        s.attitude = UnitQuaternion::from_euler_angles(0.0, 0.0, core::f64::consts::FRAC_PI_2);
        let field = m.sample(&truth(&s)).field;
        let r = magnetic_field_ned(); // (0.5, 0, 0.866)
                                      // R(yaw90)^T maps North(+x) -> +... let's just check the horizontal part rotated.
        assert!(
            (field.x).abs() < 1e-9,
            "x should vanish after 90° yaw: {}",
            field.x
        );
        assert!(
            (field.y + r.x).abs() < 1e-9,
            "y should be -north_component: {}",
            field.y
        );
        assert!(
            (field.z - r.z).abs() < 1e-12,
            "vertical component unchanged"
        );
    }
}
