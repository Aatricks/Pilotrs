//! Quaternion Multiplicative EKF (MEKF) — a 6-state AHRS.
//!
//! Nominal state: attitude `q_{world←body}` and gyro bias `b_g` (body frame).
//! Error state `δx = [δθ, δb_g] ∈ ℝ⁶` with the **local / body-frame**
//! multiplicative convention `q_true = q_nom ⊗ δq(δθ)`. Choosing the body-frame
//! (right-multiplicative) error means *every* quaternion product in the filter
//! — predict, mag reset, accel reset — is on the same side, which is the single
//! best guard against the kind of mixed-side sign bug that bit the
//! complementary filter.
//!
//! Why MEKF over the complementary filter: it estimates the gyro bias (so a
//! biased gyro no longer makes attitude drift), carries a principled
//! covariance, and χ²-gates its measurement updates — so a corrupted
//! accelerometer sample during a maneuver is *rejected* rather than dragging
//! the estimate toward false-level.
//!
//! ## Derived conventions (proven, not guessed — see the sign tests)
//!
//! Gyro `ω̃ = ω_true + b_g + n`, so bias-corrected rate `ω̂ = ω̃ − b̂_g`. The
//! error dynamics are `δθ̇ = −[ω̂]ₓ δθ − δb_g − n_g`, `δḃ_g = n_b`, giving the
//! discrete transition `F = [[I − [ω̂]ₓΔt, −IΔt],[0, I]]` (the top-right block
//! is `−IΔt`, *not* `−R̂Δt` — a consequence of the body-frame error choice).
//!
//! For a world reference unit vector `v` (gravity-up or the geomagnetic field),
//! the predicted body measurement is `ŷ = R(q)ᵀ v = q⁻¹·v`, and to first order
//! `h(q_nom ⊗ δq) ≈ ŷ + [ŷ]ₓ δθ`, so `H = [[ŷ]ₓ, 0]` (sign verified in tests).

use crate::Estimator;
use fsim_core::{
    magnetic_field_ned, BaroMeas, EstState, GpsMeas, ImuMeas, MagMeas, Quat, Real, Vec3, GRAVITY,
};
use nalgebra::{Matrix3, SMatrix, UnitQuaternion};

type Mat6 = SMatrix<Real, 6, 6>;
type Mat36 = SMatrix<Real, 3, 6>;

/// Tuning for the [`Mekf`].
#[derive(Debug, Clone, Copy)]
pub struct MekfConfig {
    /// Gyro white-noise density \[rad/s\] (process noise on δθ).
    pub gyro_noise: Real,
    /// Gyro bias random-walk density \[rad/s/√s\] (process noise on δb_g).
    pub gyro_bias_walk: Real,
    /// Accelerometer noise \[m/s²\]; the gravity update runs in *unit-vector*
    /// space, so R = `(accel_noise/g)² + accel_dir_floor²`.
    pub accel_noise: Real,
    /// Direction-error floor \[rad\] folded into R_acc to absorb vibration /
    /// mild unmodeled acceleration.
    pub accel_dir_floor: Real,
    /// Magnetometer measurement std (units of the unit reference field).
    pub mag_noise: Real,
    /// Reject the accel update when ‖accel‖ deviates from g by more than this
    /// fraction (a magnitude pre-gate; the χ² gate handles direction).
    pub accel_gate: Real,
    /// χ² threshold (df 3, 99%) for the accel innovation gate.
    pub chi2_accel: Real,
    /// χ² threshold (df 3, 99%) for the mag innovation gate.
    pub chi2_mag: Real,
    /// Initial attitude-error std \[rad\] (P₀, isotropic). Sized for a roughly
    /// level start (~10°): small enough that the early high-gain transient
    /// doesn't dump motion-corrupted accel into the bias state, large enough to
    /// converge a modest initial error. Converging from a *large* unknown
    /// attitude needs a bigger value (see the convergence test).
    pub init_att: Real,
    /// Initial gyro-bias std \[rad/s\] (P₀).
    pub init_bias: Real,
}

impl Default for MekfConfig {
    fn default() -> Self {
        Self {
            gyro_noise: 0.01,
            gyro_bias_walk: 0.001,
            accel_noise: 0.30,
            accel_dir_floor: 0.05,
            mag_noise: 0.05,
            accel_gate: 0.15,
            chi2_accel: 11.34,
            chi2_mag: 11.34,
            init_att: 0.18,
            init_bias: 0.05,
        }
    }
}

/// 6-state quaternion MEKF (attitude + gyro bias).
#[derive(Debug, Clone)]
pub struct Mekf {
    cfg: MekfConfig,
    q: Quat,
    bias: Vec3,
    p: Mat6,
    rate: Vec3,
}

/// Skew-symmetric (cross-product) matrix `[v]ₓ` such that `[v]ₓ a = v × a`.
fn skew(v: Vec3) -> Matrix3<Real> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

impl Mekf {
    /// Create an MEKF, level, with the given config.
    pub fn new(cfg: MekfConfig) -> Self {
        let mut p = Mat6::zeros();
        let att = cfg.init_att * cfg.init_att;
        let b = cfg.init_bias * cfg.init_bias;
        for i in 0..3 {
            p[(i, i)] = att;
            p[(i + 3, i + 3)] = b;
        }
        Self {
            cfg,
            q: UnitQuaternion::identity(),
            bias: Vec3::zeros(),
            p,
            rate: Vec3::zeros(),
        }
    }

    /// Current attitude estimate.
    pub fn attitude(&self) -> Quat {
        self.q
    }

    /// Current gyro-bias estimate \[rad/s\].
    pub fn gyro_bias(&self) -> Vec3 {
        self.bias
    }

    /// Trace of the attitude block of P (a scalar "attitude uncertainty").
    pub fn attitude_variance(&self) -> Real {
        self.p[(0, 0)] + self.p[(1, 1)] + self.p[(2, 2)]
    }

    /// The full 6×6 error covariance (diagnostics / tests).
    pub fn covariance(&self) -> Mat6 {
        self.p
    }

    /// Vector measurement update against a **unit** world reference `v_ref`,
    /// with `z` already normalized to a unit direction. Forms `H = [[ŷ]ₓ, 0]`
    /// (ŷ = q⁻¹·v_ref), χ²-gates the innovation, then does a Joseph-form EKF
    /// update and the multiplicative error injection + reset. Returns whether
    /// the update was applied.
    fn update_reference(&mut self, z: Vec3, v_ref: Vec3, var: Real, chi2: Real) -> bool {
        let y_hat = self.q.inverse() * v_ref;

        let mut h = Mat36::zeros();
        h.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(y_hat));

        let r = Matrix3::identity() * var;
        let s = h * self.p * h.transpose() + r;
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return false,
        };

        let innov = z - y_hat;
        // χ² innovation gate (also rejects the gravity update when the
        // accelerometer direction is corrupted by vehicle acceleration).
        let nis = (innov.transpose() * s_inv * innov)[(0, 0)];
        if nis > chi2 {
            return false;
        }

        let k = self.p * h.transpose() * s_inv; // 6x3 Kalman gain
        let dx = k * innov;

        // Inject: q ← q ⊗ δq(δθ), b_g ← b_g + δb_g (then error reset to 0).
        let dtheta = Vec3::new(dx[0], dx[1], dx[2]);
        let dbias = Vec3::new(dx[3], dx[4], dx[5]);
        self.q *= UnitQuaternion::from_scaled_axis(dtheta);
        self.q.renormalize();
        self.bias += dbias;

        // Joseph-form covariance update (numerically stable, stays PSD).
        let ikh = Mat6::identity() - k * h;
        self.p = ikh * self.p * ikh.transpose() + k * r * k.transpose();
        self.symmetrize();
        true
    }

    fn symmetrize(&mut self) {
        self.p = (self.p + self.p.transpose()) * 0.5;
    }
}

impl Default for Mekf {
    fn default() -> Self {
        Self::new(MekfConfig::default())
    }
}

impl Estimator for Mekf {
    fn predict(&mut self, imu: &ImuMeas, dt: Real) {
        // Bias-corrected body rate for attitude propagation.
        let omega = imu.gyro - self.bias;
        // Report the RAW gyro as the rate estimate for the controller: the fast
        // rate loop is insensitive to a ~0.01 rad/s bias, and decoupling it from
        // the bias estimate prevents a transiently-wrong bias (e.g. while
        // accelerating) from destabilizing the loop. Debiasing still happens
        // inside the attitude integration above.
        self.rate = imu.gyro;

        // Nominal attitude propagation (body-frame increment, right-multiply).
        self.q *= UnitQuaternion::from_scaled_axis(omega * dt);
        self.q.renormalize();

        // Covariance propagation: P ← F P Fᵀ + Q.
        let mut f = Mat6::identity();
        f.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(Matrix3::identity() - skew(omega) * dt));
        f.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(-Matrix3::identity() * dt));

        // Discrete process noise (config stds are continuous densities
        // integrated over dt — one convention, never also divide by rate).
        let sg2 = self.cfg.gyro_noise * self.cfg.gyro_noise;
        let sb2 = self.cfg.gyro_bias_walk * self.cfg.gyro_bias_walk;
        let q_tt = sg2 * dt + sb2 * dt * dt * dt / 3.0;
        let q_tb = -sb2 * dt * dt / 2.0;
        let q_bb = sb2 * dt;
        let mut qn = Mat6::zeros();
        for i in 0..3 {
            qn[(i, i)] = q_tt;
            qn[(i, i + 3)] = q_tb;
            qn[(i + 3, i)] = q_tb;
            qn[(i + 3, i + 3)] = q_bb;
        }
        self.p = f * self.p * f.transpose() + qn;
        self.symmetrize();

        // Accelerometer gravity update (folded in — the data is already here).
        // Direction-only: reference is world "up" = (0,0,−1) in NED.
        let acc_norm = imu.accel.norm();
        if acc_norm > 1e-6 {
            let deviation = (acc_norm - GRAVITY).abs() / GRAVITY;
            if deviation < self.cfg.accel_gate {
                let var = {
                    let s = self.cfg.accel_noise / GRAVITY;
                    s * s + self.cfg.accel_dir_floor * self.cfg.accel_dir_floor
                };
                self.update_reference(
                    imu.accel / acc_norm,
                    Vec3::new(0.0, 0.0, -1.0),
                    var,
                    self.cfg.chi2_accel,
                );
            }
        }
    }

    fn update_mag(&mut self, mag: &MagMeas) {
        // Full-vector update against the (exact, shared) field model. With an
        // accurate field this is consistent and gives full attitude aiding; a
        // real system with inclination/declination error would instead use a
        // tilt-compensated *heading-only* update to keep model error out of
        // roll/pitch — deferred until that error source exists.
        let n = mag.field.norm();
        if n > 1e-9 {
            self.update_reference(
                mag.field / n,
                magnetic_field_ned(),
                self.cfg.mag_noise * self.cfg.mag_noise,
                self.cfg.chi2_mag,
            );
        }
    }

    // GPS/baro position fusion is the INS; the 6-state AHRS ignores them.
    fn update_gps(&mut self, _gps: &GpsMeas) {}
    fn update_baro(&mut self, _baro: &BaroMeas) {}

    fn state(&self) -> EstState {
        EstState {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
            attitude: self.q,
            angular_rate: self.rate,
        }
    }

    fn gyro_bias_estimate(&self) -> Option<Vec3> {
        Some(self.bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::ImuMeas;

    fn accel_for(q: &Quat) -> Vec3 {
        q.inverse() * Vec3::new(0.0, 0.0, -GRAVITY)
    }
    fn mag_for(q: &Quat) -> Vec3 {
        q.inverse() * magnetic_field_ned()
    }

    #[test]
    fn measurement_jacobian_matches_finite_difference() {
        // The most important guard: numerically differentiate h(q⊗δq(εeᵢ)) and
        // compare to the analytic H = [ŷ]ₓ, per axis. Catches a flipped sign.
        let q = UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3);
        let v_ref = Vec3::new(0.0, 0.0, -1.0);
        let y_hat = q.inverse() * v_ref;
        let h_analytic = skew(y_hat);
        let eps = 1e-6;
        for axis in 0..3 {
            let mut e = Vec3::zeros();
            e[axis] = eps;
            let q_pert = q * UnitQuaternion::from_scaled_axis(e);
            let h_pert = q_pert.inverse() * v_ref;
            let fd = (h_pert - y_hat) / eps; // ∂h/∂δθ_axis
            let col = h_analytic.column(axis);
            assert!(
                (fd - col).norm() < 1e-4,
                "axis {axis}: fd={fd:?} analytic={col:?}"
            );
        }
    }

    #[test]
    fn sign_check_roll_drives_positive_dtheta_x() {
        // Nominal level, true attitude +10° roll, zero gyro: one accel update
        // must rotate the estimate toward +roll (not away).
        let mut f = Mekf::default();
        let q_true = UnitQuaternion::from_euler_angles(0.1745, 0.0, 0.0);
        f.predict(
            &ImuMeas {
                accel: accel_for(&q_true),
                gyro: Vec3::zeros(),
            },
            1e-3,
        );
        let (roll, _, _) = f.attitude().euler_angles();
        assert!(roll > 0.0, "estimate rolled the wrong way: {roll}");
    }

    #[test]
    fn converges_to_tilt_with_accel_and_mag() {
        // Large unknown initial attitude -> needs a generous P₀ so the χ² gate
        // accepts the (valid, large) early innovations.
        let q_true = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
        let imu = ImuMeas {
            accel: accel_for(&q_true),
            gyro: Vec3::zeros(),
        };
        let mag = MagMeas {
            field: mag_for(&q_true),
        };
        let mut f = Mekf::new(MekfConfig {
            init_att: 1.0,
            ..MekfConfig::default()
        });
        for k in 0..4000 {
            f.predict(&imu, 1e-3);
            if k % 20 == 0 {
                f.update_mag(&mag);
            }
        }
        assert!(
            f.attitude().angle_to(&q_true) < 1e-2,
            "did not converge: {} rad",
            f.attitude().angle_to(&q_true)
        );
    }

    #[test]
    fn estimates_constant_gyro_bias() {
        // Level & static truth, but the gyro carries a constant bias. The MEKF
        // recovers the bias and keeps attitude near level — the thing the
        // complementary filter cannot do.
        let true_bias = Vec3::new(0.02, -0.015, 0.01);
        let q_true = UnitQuaternion::identity();
        let mut f = Mekf::default();
        for k in 0..60_000 {
            f.predict(
                &ImuMeas {
                    accel: accel_for(&q_true),
                    gyro: true_bias, // ω_true = 0, so gyro reads the bias
                },
                1e-3,
            );
            if k % 20 == 0 {
                f.update_mag(&MagMeas {
                    field: mag_for(&q_true),
                });
            }
        }
        assert!(
            (f.gyro_bias() - true_bias).norm() < 2e-3,
            "bias est {:?}",
            f.gyro_bias()
        );
        assert!(
            f.attitude().angle() < 1e-2,
            "attitude drifted: {}",
            f.attitude().angle()
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // test-only: seed a wrong bias
    fn bias_sign_corrects_toward_truth() {
        // Spec sign test 8: filter initialized with a wrong +x bias, truth level
        // & static (gyro reads 0). The bias estimate must move *down* toward 0.
        let q_true = UnitQuaternion::identity();
        let mut f = Mekf::default();
        f.bias = Vec3::new(0.02, 0.0, 0.0);
        for k in 0..40_000 {
            f.predict(
                &ImuMeas {
                    accel: accel_for(&q_true),
                    gyro: Vec3::zeros(),
                },
                1e-3,
            );
            if k % 20 == 0 {
                f.update_mag(&MagMeas {
                    field: mag_for(&q_true),
                });
            }
        }
        assert!(f.gyro_bias().x < 0.02, "bias x did not decrease");
        assert!(
            f.gyro_bias().norm() < 2e-3,
            "bias did not converge to 0: {:?}",
            f.gyro_bias()
        );
    }

    #[test]
    fn mag_makes_yaw_observable() {
        let q_true = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.4);
        let imu = ImuMeas {
            accel: accel_for(&q_true),
            gyro: Vec3::zeros(),
        };
        let mut f = Mekf::default();
        for _ in 0..2000 {
            f.predict(&imu, 1e-3); // accel only — yaw stays unobserved
        }
        let (_, _, yaw_no_mag) = f.attitude().euler_angles();
        for k in 0..4000 {
            f.predict(&imu, 1e-3);
            if k % 20 == 0 {
                f.update_mag(&MagMeas {
                    field: mag_for(&q_true),
                });
            }
        }
        let (roll, pitch, yaw_mag) = f.attitude().euler_angles();
        assert!(
            yaw_no_mag.abs() < 0.05,
            "accel alone changed yaw: {yaw_no_mag}"
        );
        assert!(
            (yaw_mag - 0.4).abs() < 1e-2,
            "mag didn't fix yaw: {yaw_mag}"
        );
        assert!(
            roll.abs() < 1e-2 && pitch.abs() < 1e-2,
            "mag corrupted roll/pitch"
        );
    }

    #[test]
    fn covariance_shrinks_and_stays_psd() {
        // Truth is tilted while the filter starts level, so the updates see real
        // (nonzero) innovations — not a trivial zero-innovation pass. Assert the
        // covariance shrinks, stays symmetric and PSD, and the estimate actually
        // converges (proving the updates moved the state, not just P).
        let q_true = UnitQuaternion::from_euler_angles(0.2, -0.15, 0.3);
        let mut f = Mekf::default();
        let p0 = f.attitude_variance();
        for k in 0..3000 {
            f.predict(
                &ImuMeas {
                    accel: accel_for(&q_true),
                    gyro: Vec3::zeros(),
                },
                1e-3,
            );
            if k % 20 == 0 {
                f.update_mag(&MagMeas {
                    field: mag_for(&q_true),
                });
            }
        }
        assert!(f.attitude_variance() < p0, "covariance did not shrink");

        let p = f.covariance();
        assert!((p - p.transpose()).norm() < 1e-9, "P is not symmetric");
        // Positive semidefinite: Cholesky of P + tiny jitter must succeed.
        let jitter = SMatrix::<Real, 6, 6>::identity() * 1e-12;
        assert!((p + jitter).cholesky().is_some(), "P is not PSD");

        assert!(
            f.attitude().angle_to(&q_true) < 1e-2,
            "estimate did not converge"
        );
    }
}
