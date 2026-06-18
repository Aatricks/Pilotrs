//! Strapdown IMU: a 3-axis accelerometer (specific force) and 3-axis gyro,
//! both in the body frame, corrupted by bias random-walk + white Gaussian
//! noise.

use crate::{Sensor, Truth};
use fsim_core::{gravity_world, ImuMeas, Real, Vec3};
use num_traits::Float;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};

/// Noise/bias parameters for an [`Imu`].
#[derive(Debug, Clone, Copy)]
pub struct ImuConfig {
    /// Sample rate \[Hz\].
    pub rate_hz: Real,
    /// Accelerometer white-noise std \[m/s^2\].
    pub accel_noise_std: Real,
    /// Gyro white-noise std \[rad/s\].
    pub gyro_noise_std: Real,
    /// Accelerometer bias random-walk intensity \[m/s^2 / √s\].
    pub accel_bias_walk: Real,
    /// Gyro bias random-walk intensity \[rad/s / √s\].
    pub gyro_bias_walk: Real,
    /// Initial constant gyro bias \[rad/s\] (observable by the estimator).
    pub gyro_bias0: Vec3,
}

impl ImuConfig {
    /// Light-noise config for the M1 MVP (consumer-MEMS-ish, mild).
    pub fn mvp(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            accel_noise_std: 0.05,
            gyro_noise_std: 0.002,
            accel_bias_walk: 0.0,
            gyro_bias_walk: 0.0,
            gyro_bias0: Vec3::zeros(),
        }
    }

    /// Realistic noisy MEMS with bias random-walk (M2).
    pub fn realistic(rate_hz: Real) -> Self {
        Self {
            rate_hz,
            accel_noise_std: 0.30,
            gyro_noise_std: 0.01,
            accel_bias_walk: 0.02,
            gyro_bias_walk: 0.001,
            gyro_bias0: Vec3::new(0.01, -0.008, 0.005),
        }
    }
}

/// A simulated IMU with its own deterministic RNG stream.
#[derive(Debug, Clone)]
pub struct Imu {
    cfg: ImuConfig,
    rng: ChaCha8Rng,
    accel_bias: Vec3,
    gyro_bias: Vec3,
}

impl Imu {
    /// Create an IMU seeded with `seed` (independent stream per sensor).
    pub fn new(cfg: ImuConfig, seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            accel_bias: Vec3::zeros(),
            gyro_bias: cfg.gyro_bias0,
            cfg,
        }
    }

    /// The current (hidden) gyro bias — for tests/plots, not for the autopilot.
    pub fn gyro_bias(&self) -> Vec3 {
        self.gyro_bias
    }

    fn sample_vec3(&mut self, std: Real) -> Vec3 {
        if std <= 0.0 {
            return Vec3::zeros();
        }
        let n = Normal::new(0.0, std).expect("std >= 0");
        Vec3::new(
            n.sample(&mut self.rng),
            n.sample(&mut self.rng),
            n.sample(&mut self.rng),
        )
    }
}

impl Sensor for Imu {
    type Measurement = ImuMeas;

    fn rate_hz(&self) -> Real {
        self.cfg.rate_hz
    }

    fn sample(&mut self, truth: &Truth<'_>) -> ImuMeas {
        let dt = 1.0 / self.cfg.rate_hz;
        let sqrt_dt = Float::sqrt(dt);

        // Bias random walk: b_{k+1} = b_k + w·√dt. (Locals avoid borrowing
        // `self` mutably twice in one expression.)
        let accel_walk = self.sample_vec3(self.cfg.accel_bias_walk * sqrt_dt);
        let gyro_walk = self.sample_vec3(self.cfg.gyro_bias_walk * sqrt_dt);
        self.accel_bias += accel_walk;
        self.gyro_bias += gyro_walk;

        // Accelerometer measures specific force f = a_world − g, rotated into
        // the body frame. At rest & level this reads (0, 0, −g): "1g up".
        let specific_force_world = truth.accel_world - gravity_world();
        let accel_body = truth.state.attitude.inverse() * specific_force_world;
        let accel = accel_body + self.accel_bias + self.sample_vec3(self.cfg.accel_noise_std);

        // Gyro measures body rate + bias + noise.
        let gyro =
            truth.state.angular_rate + self.gyro_bias + self.sample_vec3(self.cfg.gyro_noise_std);

        ImuMeas { accel, gyro }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::{State13, Vec3, GRAVITY};

    fn rest_truth(state: &State13) -> Truth<'_> {
        Truth {
            state,
            accel_world: Vec3::zeros(), // truly at rest -> a_world = 0
            t: 0.0,
        }
    }

    #[test]
    fn level_at_rest_reads_one_g_up() {
        // Zero-noise IMU, level & at rest: accel == (0, 0, -g), gyro == 0.
        let cfg = ImuConfig {
            accel_noise_std: 0.0,
            gyro_noise_std: 0.0,
            ..ImuConfig::mvp(1000.0)
        };
        let mut imu = Imu::new(cfg, 1);
        let s = State13::at_rest();
        let m = imu.sample(&rest_truth(&s));
        assert!((m.accel - Vec3::new(0.0, 0.0, -GRAVITY)).norm() < 1e-12);
        assert!(m.gyro.norm() < 1e-12);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let cfg = ImuConfig::realistic(1000.0);
        let mut a = Imu::new(cfg, 42);
        let mut b = Imu::new(cfg, 42);
        let s = State13::at_rest();
        for _ in 0..100 {
            let ma = a.sample(&rest_truth(&s));
            let mb = b.sample(&rest_truth(&s));
            assert_eq!(ma.accel, mb.accel);
            assert_eq!(ma.gyro, mb.gyro);
        }
    }

    #[test]
    fn noise_statistics_are_sane() {
        // Mean ≈ truth, sample std ≈ configured, over many samples (no bias walk).
        let cfg = ImuConfig {
            accel_bias_walk: 0.0,
            gyro_bias_walk: 0.0,
            gyro_bias0: Vec3::zeros(),
            ..ImuConfig::realistic(1000.0)
        };
        let mut imu = Imu::new(cfg, 7);
        let s = State13::at_rest();
        let n = 20_000;
        let (mut mean, mut m2) = (0.0_f64, 0.0_f64);
        for k in 1..=n {
            let g = imu.sample(&rest_truth(&s)).gyro.x;
            // Welford online mean/variance.
            let delta = g - mean;
            mean += delta / k as f64;
            m2 += delta * (g - mean);
        }
        let std = (m2 / (n as f64 - 1.0)).sqrt();
        assert!(mean.abs() < 5e-4, "gyro mean drifted: {mean}");
        assert!(
            (std - cfg.gyro_noise_std).abs() < 1e-3,
            "gyro std off: {std}"
        );
    }

    #[test]
    fn bias_random_walk_grows() {
        let cfg = ImuConfig::realistic(200.0);
        let mut imu = Imu::new(cfg, 3);
        let s = State13::at_rest();
        let b0 = imu.gyro_bias();
        for _ in 0..10_000 {
            imu.sample(&rest_truth(&s));
        }
        // The walk should have moved the bias well away from its start.
        assert!((imu.gyro_bias() - b0).norm() > 1e-3);
    }
}
