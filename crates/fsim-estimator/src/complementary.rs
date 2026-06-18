//! Mahony-style explicit complementary filter for attitude.
//!
//! It blends the two sensors by their reliable frequency band: the **gyro**
//! drives attitude at high frequency (integrate body rate), while the
//! **accelerometer** anchors roll/pitch at low frequency by measuring the
//! gravity direction. Yaw is unobservable from gravity alone, so it rides on
//! the gyro until the magnetometer arrives (M2).
//!
//! Each step:
//! 1. Estimate the "up" direction in the body frame from the current attitude,
//!    `v_est = q⁻¹ · (0,0,−1)`.
//! 2. The accelerometer's normalized specific force `v_meas` also points "up"
//!    in the body frame at low acceleration.
//! 3. The correction `e = v_est × v_meas` is the small rotation that aligns the
//!    estimate to the measurement; add `Kp·e` to the gyro before integrating.

use crate::Estimator;
use fsim_core::{EstState, ImuMeas, Quat, Real, Vec3, GRAVITY};
use nalgebra::UnitQuaternion;
use num_traits::Float;

/// Tuning for the [`ComplementaryFilter`].
#[derive(Debug, Clone, Copy)]
pub struct ComplementaryConfig {
    /// Proportional accel-correction gain \[1/s\]. Higher = trust accel more
    /// (faster tilt correction, more noise sensitivity).
    pub kp: Real,
    /// Reject accel correction when |accel| deviates from g by more than this
    /// fraction (linear acceleration corrupts the gravity estimate).
    pub accel_gate: Real,
}

impl Default for ComplementaryConfig {
    fn default() -> Self {
        // M1's gyro is clean (no bias), so gyro integration is already an
        // excellent attitude reference and we keep `kp` low — the accel term is
        // there for slow leveling / initial alignment, not heavy correction. A
        // high `kp` actively *hurts* during accelerated flight: a quad holding a
        // tilt accelerates laterally, and with thrust ≈ mg the specific force
        // points along body −z ("apparent level") regardless of true tilt, so a
        // strong accel correction drags the estimate toward level. Removing that
        // acceleration from the specific force needs velocity aiding — exactly
        // what the M2 MEKF/INS adds.
        Self {
            kp: 0.3,
            accel_gate: 0.15,
        }
    }
}

/// Explicit complementary filter estimating attitude (and passing gyro through
/// as the body-rate estimate).
#[derive(Debug, Clone)]
pub struct ComplementaryFilter {
    cfg: ComplementaryConfig,
    q: Quat,
    rate: Vec3,
}

impl ComplementaryFilter {
    /// Start level (identity attitude) with the given config.
    pub fn new(cfg: ComplementaryConfig) -> Self {
        Self {
            cfg,
            q: UnitQuaternion::identity(),
            rate: Vec3::zeros(),
        }
    }

    /// Start from a known initial attitude (e.g. a coarse alignment).
    pub fn with_attitude(cfg: ComplementaryConfig, q0: Quat) -> Self {
        Self {
            cfg,
            q: q0,
            rate: Vec3::zeros(),
        }
    }

    /// Current attitude estimate.
    pub fn attitude(&self) -> Quat {
        self.q
    }
}

impl Default for ComplementaryFilter {
    fn default() -> Self {
        Self::new(ComplementaryConfig::default())
    }
}

impl Estimator for ComplementaryFilter {
    fn predict(&mut self, imu: &ImuMeas, dt: Real) {
        let mut omega = imu.gyro;

        // Accel correction, gated to near-1g samples.
        let acc_norm = imu.accel.norm();
        if acc_norm > 1e-6 {
            let deviation = Float::abs(acc_norm - GRAVITY) / GRAVITY;
            if deviation < self.cfg.accel_gate {
                let v_meas = imu.accel / acc_norm; // measured "up" in body
                let up_world = Vec3::new(0.0, 0.0, -1.0); // NED up
                let v_est = self.q.inverse() * up_world; // estimated "up" in body
                                                         // For q_{world<-body} with a body-frame rate, the body increment
                                                         // updates v_est by ≈ -ω·dt × v_est, so the correction that drives
                                                         // v_est -> v_meas is ω += Kp·(v_meas × v_est).
                let error = v_meas.cross(&v_est);
                omega += error * self.cfg.kp;
            }
        }

        // Integrate the (corrected) body rate via the exponential map; the
        // body-frame increment composes on the right of q_{world<-body}.
        let dq = UnitQuaternion::from_scaled_axis(omega * dt);
        self.q *= dq;
        self.q.renormalize();

        // Best body-rate estimate is the raw gyro (bias estimation comes with
        // the MEKF in M2).
        self.rate = imu.gyro;
    }

    fn state(&self) -> EstState {
        EstState {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
            attitude: self.q,
            angular_rate: self.rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::Vec3;

    /// Specific force a level/ tilted IMU reads at rest for a given true
    /// attitude: f_body = q⁻¹ · (0,0,−g).
    fn accel_for(q_true: &Quat) -> Vec3 {
        q_true.inverse() * Vec3::new(0.0, 0.0, -GRAVITY)
    }

    #[test]
    fn stays_level_when_level() {
        let mut f = ComplementaryFilter::default();
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        for _ in 0..1000 {
            f.predict(&imu, 1e-3);
        }
        assert!(f.attitude().angle() < 1e-6);
    }

    #[test]
    fn converges_to_tilt_from_accel() {
        // True attitude has 20° roll, 10° pitch. Filter starts level, gyro
        // reads zero, accel is consistent with the tilt -> roll/pitch converge.
        let q_true = UnitQuaternion::from_euler_angles(0.349, 0.175, 0.0);
        let imu = ImuMeas {
            accel: accel_for(&q_true),
            gyro: Vec3::zeros(),
        };
        let mut f = ComplementaryFilter::default();
        for _ in 0..20_000 {
            f.predict(&imu, 1e-3);
        }
        // Compare the "up" directions (yaw is unobservable here).
        let up = Vec3::new(0.0, 0.0, -1.0);
        let true_up_body = q_true.inverse() * up;
        let est_up_body = f.attitude().inverse() * up;
        assert!(
            (true_up_body - est_up_body).norm() < 1e-3,
            "tilt not recovered"
        );
    }

    #[test]
    fn tracks_pure_yaw_from_gyro() {
        // Yaw at 1 rad/s for 1 s with level accel -> ~1 rad of yaw, roll/pitch 0.
        let mut f = ComplementaryFilter::default();
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::new(0.0, 0.0, 1.0),
        };
        for _ in 0..1000 {
            f.predict(&imu, 1e-3);
        }
        let (roll, pitch, yaw) = f.attitude().euler_angles();
        assert!(
            roll.abs() < 1e-3 && pitch.abs() < 1e-3,
            "tilt leaked into yaw test"
        );
        assert!((yaw - 1.0).abs() < 1e-2, "yaw={yaw}");
    }
}
