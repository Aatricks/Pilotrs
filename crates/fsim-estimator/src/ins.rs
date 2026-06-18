//! 15-state Inertial Navigation System (loosely-coupled error-state Kalman
//! filter). Unlike the [`Mekf`](crate::Mekf) AHRS, the INS uses the
//! accelerometer as the **strapdown propagation input** (not a gravity
//! reference), so a sustained translating maneuver no longer corrupts attitude
//! — that's the whole point of M3. GPS (position + velocity), baro (altitude),
//! and the magnetometer (heading) aid the propagation; attitude becomes
//! observable through the velocity↔attitude coupling whenever thrust is on.
//!
//! State (nominal): position p (NED), velocity v (NED), attitude
//! `q_{world←body}`, accel bias b_a (body), gyro bias b_g (body).
//! Error state `δx = [δp, δv, δθ, δb_a, δb_g] ∈ ℝ¹⁵` with the **same
//! body-frame right-multiplicative** attitude convention as the MEKF, so all
//! the M2 attitude signs carry over unchanged.
//!
//! ## Load-bearing derivation
//!
//! Strapdown: `f̂ = accel − b_a`, `ω̂ = gyro − b_g`, `a_world = R̂ f̂ + g_w`
//! (g_w points down, +z). The velocity-error couples to attitude error as
//! `δv̇ = −R̂[f̂]ₓ δθ − R̂ δb_a` (two independent minus signs: the
//! right-multiplicative error gives `R̂(I+[δθ]ₓ)`, and `[δθ]ₓf̂ = −[f̂]ₓδθ`).
//! A finite-difference test pins this sign.

use crate::Estimator;
use fsim_core::{
    gravity_world, magnetic_field_ned, BaroMeas, EstState, GpsMeas, ImuMeas, MagMeas, Quat, Real,
    Vec3,
};
use nalgebra::{Matrix3, SMatrix, SVector, UnitQuaternion};

type Mat15 = SMatrix<Real, 15, 15>;
type Mat3_15 = SMatrix<Real, 3, 15>;
type Mat1_15 = SMatrix<Real, 1, 15>;

/// Skew-symmetric (cross-product) matrix `[v]ₓ` (`[v]ₓ a = v × a`).
fn skew(v: Vec3) -> Matrix3<Real> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Tuning for the [`Ins`]. Process terms are continuous densities integrated
/// over `dt` (the same one-convention rule as the MEKF). Measurement stds are
/// deliberately inflated above the raw sensor noise because the GPS and baro
/// models carry unmodeled slowly-wandering biases the INS does not estimate.
#[derive(Debug, Clone, Copy)]
pub struct InsConfig {
    pub gyro_noise: Real,
    pub gyro_bias_walk: Real,
    pub accel_noise: Real,
    pub accel_bias_walk: Real,
    pub gps_pos_noise: Real,
    pub gps_vel_noise: Real,
    pub baro_noise: Real,
    pub mag_noise: Real,
    pub chi2_gps_pos: Real,
    pub chi2_gps_vel: Real,
    pub chi2_baro: Real,
    pub chi2_mag: Real,
    pub init_pos: Real,
    pub init_vel: Real,
    pub init_att: Real,
    pub init_accel_bias: Real,
    pub init_gyro_bias: Real,
}

impl Default for InsConfig {
    fn default() -> Self {
        Self {
            gyro_noise: 0.01,
            gyro_bias_walk: 0.001,
            accel_noise: 0.30,
            accel_bias_walk: 0.02,
            gps_pos_noise: 3.0,
            gps_vel_noise: 0.25,
            baro_noise: 0.8,
            mag_noise: 0.05,
            chi2_gps_pos: 11.34,
            chi2_gps_vel: 11.34,
            chi2_baro: 6.63,
            chi2_mag: 11.34,
            init_pos: 5.0,
            init_vel: 0.5,
            init_att: 0.18,
            init_accel_bias: 0.1,
            init_gyro_bias: 0.05,
        }
    }
}

/// 15-state INS / error-state Kalman filter.
#[derive(Debug, Clone)]
pub struct Ins {
    cfg: InsConfig,
    p: Vec3,
    v: Vec3,
    q: Quat,
    b_a: Vec3,
    b_g: Vec3,
    cov: Mat15,
    rate: Vec3,
    have_fix: bool,
}

impl Ins {
    /// Create an INS, level, at the origin (position is hard-set on the first
    /// GPS fix to avoid a huge P₀ transient).
    pub fn new(cfg: InsConfig) -> Self {
        let mut cov = Mat15::zeros();
        let diag = [
            (0, cfg.init_pos),
            (3, cfg.init_vel),
            (6, cfg.init_att),
            (9, cfg.init_accel_bias),
            (12, cfg.init_gyro_bias),
        ];
        for (base, std) in diag {
            for i in 0..3 {
                cov[(base + i, base + i)] = std * std;
            }
        }
        Self {
            cfg,
            p: Vec3::zeros(),
            v: Vec3::zeros(),
            q: UnitQuaternion::identity(),
            b_a: Vec3::zeros(),
            b_g: Vec3::zeros(),
            cov,
            rate: Vec3::zeros(),
            have_fix: false,
        }
    }

    pub fn position(&self) -> Vec3 {
        self.p
    }
    pub fn velocity(&self) -> Vec3 {
        self.v
    }
    pub fn attitude(&self) -> Quat {
        self.q
    }
    pub fn accel_bias(&self) -> Vec3 {
        self.b_a
    }
    pub fn gyro_bias(&self) -> Vec3 {
        self.b_g
    }
    pub fn covariance(&self) -> Mat15 {
        self.cov
    }

    fn symmetrize(&mut self) {
        self.cov = (self.cov + self.cov.transpose()) * 0.5;
    }

    /// Inject an error-state correction into the nominal state (additive for
    /// p/v/biases, multiplicative right-side for attitude), then reset (the
    /// error mean is consumed, never stored).
    fn inject(&mut self, dx: &SVector<Real, 15>) {
        self.p += Vec3::new(dx[0], dx[1], dx[2]);
        self.v += Vec3::new(dx[3], dx[4], dx[5]);
        self.q *= UnitQuaternion::from_scaled_axis(Vec3::new(dx[6], dx[7], dx[8]));
        self.q.renormalize();
        self.b_a += Vec3::new(dx[9], dx[10], dx[11]);
        self.b_g += Vec3::new(dx[12], dx[13], dx[14]);
    }

    /// Joseph-form EKF update for a 3-vector measurement with isotropic noise.
    fn update3(&mut self, h: &Mat3_15, innov: Vec3, var: Real, chi2: Real) -> bool {
        let r = Matrix3::identity() * var;
        let s = h * self.cov * h.transpose() + r;
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return false,
        };
        let nis = (innov.transpose() * s_inv * innov)[(0, 0)];
        if nis > chi2 {
            return false;
        }
        let k = self.cov * h.transpose() * s_inv; // 15x3
        let dx = k * innov;
        self.inject(&dx);
        let ikh = Mat15::identity() - k * h;
        self.cov = ikh * self.cov * ikh.transpose() + k * r * k.transpose();
        self.symmetrize();
        true
    }

    /// Joseph-form EKF update for a scalar measurement.
    fn update1(&mut self, h: &Mat1_15, innov: Real, var: Real, chi2: Real) -> bool {
        let s = (h * self.cov * h.transpose())[(0, 0)] + var;
        if s <= 0.0 {
            return false;
        }
        if innov * innov / s > chi2 {
            return false;
        }
        let k = self.cov * h.transpose() / s; // 15x1
        let dx = k * innov;
        self.inject(&dx);
        let ikh = Mat15::identity() - k * h;
        self.cov = ikh * self.cov * ikh.transpose() + (k * k.transpose()) * var;
        self.symmetrize();
        true
    }
}

impl Default for Ins {
    fn default() -> Self {
        Self::new(InsConfig::default())
    }
}

impl Estimator for Ins {
    fn predict(&mut self, imu: &ImuMeas, dt: Real) {
        // Raw gyro to the controller (M2 policy); debias internally.
        self.rate = imu.gyro;
        let omega = imu.gyro - self.b_g; // ω̂
        let f = imu.accel - self.b_a; // f̂

        let r = self.q.to_rotation_matrix().into_inner(); // R̂ (pre-update)
        let a_w = r * f + gravity_world(); // a_world = R̂ f̂ + g_w

        // Nominal strapdown integration (½ a dt² in position).
        self.p += self.v * dt + a_w * (0.5 * dt * dt);
        self.v += a_w * dt;
        self.q *= UnitQuaternion::from_scaled_axis(omega * dt);
        self.q.renormalize();

        // F = I + F_c dt (only the nonzero off-diagonal blocks set explicitly).
        let i3 = Matrix3::identity();
        let mut fm = Mat15::identity();
        fm.fixed_view_mut::<3, 3>(0, 3).copy_from(&(i3 * dt)); // δp ← δv
        fm.fixed_view_mut::<3, 3>(3, 6)
            .copy_from(&(-r * skew(f) * dt)); // δv ← δθ
        fm.fixed_view_mut::<3, 3>(3, 9).copy_from(&(-r * dt)); // δv ← δb_a
        fm.fixed_view_mut::<3, 3>(6, 6)
            .copy_from(&(i3 - skew(omega) * dt)); // δθ ← δθ
        fm.fixed_view_mut::<3, 3>(6, 12).copy_from(&(-i3 * dt)); // δθ ← δb_g

        // Discrete process noise Q.
        let sa2 = self.cfg.accel_noise * self.cfg.accel_noise;
        let sba2 = self.cfg.accel_bias_walk * self.cfg.accel_bias_walk;
        let sg2 = self.cfg.gyro_noise * self.cfg.gyro_noise;
        let sbg2 = self.cfg.gyro_bias_walk * self.cfg.gyro_bias_walk;
        let q_tt = sg2 * dt + sbg2 * dt * dt * dt / 3.0;
        let q_tb = -sbg2 * dt * dt / 2.0;
        let q_bb = sbg2 * dt;
        let mut qm = Mat15::zeros();
        qm.fixed_view_mut::<3, 3>(6, 6).copy_from(&(i3 * q_tt));
        qm.fixed_view_mut::<3, 3>(6, 12).copy_from(&(i3 * q_tb));
        qm.fixed_view_mut::<3, 3>(12, 6).copy_from(&(i3 * q_tb));
        qm.fixed_view_mut::<3, 3>(12, 12).copy_from(&(i3 * q_bb));
        qm.fixed_view_mut::<3, 3>(3, 3)
            .copy_from(&(i3 * (sa2 * dt))); // δv from accel noise
        qm.fixed_view_mut::<3, 3>(9, 9)
            .copy_from(&(i3 * (sba2 * dt))); // accel-bias walk
        qm.fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&(i3 * (sa2 * dt * dt * dt / 3.0)));
        qm.fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&(i3 * (sa2 * dt * dt / 2.0)));
        qm.fixed_view_mut::<3, 3>(3, 0)
            .copy_from(&(i3 * (sa2 * dt * dt / 2.0)));

        self.cov = fm * self.cov * fm.transpose() + qm;
        self.symmetrize();
    }

    fn update_gps(&mut self, gps: &GpsMeas) {
        // First fix: hard-set position/velocity and collapse their covariance,
        // rather than fight a 5 m P₀ vs a (0,0,0) boot from a single update.
        if !self.have_fix {
            self.p = gps.position;
            self.v = gps.velocity;
            self.have_fix = true;
            for i in 0..6 {
                for j in 0..15 {
                    self.cov[(i, j)] = 0.0;
                    self.cov[(j, i)] = 0.0;
                }
            }
            let pv = self.cfg.gps_pos_noise * self.cfg.gps_pos_noise;
            let vv = self.cfg.gps_vel_noise * self.cfg.gps_vel_noise;
            for i in 0..3 {
                self.cov[(i, i)] = pv;
                self.cov[(i + 3, i + 3)] = vv;
            }
            return;
        }
        // Position update (H_δp = I), then velocity update (H_δv = I).
        let mut h_pos = Mat3_15::zeros();
        h_pos
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(&Matrix3::identity());
        self.update3(
            &h_pos,
            gps.position - self.p,
            self.cfg.gps_pos_noise * self.cfg.gps_pos_noise,
            self.cfg.chi2_gps_pos,
        );

        let mut h_vel = Mat3_15::zeros();
        h_vel
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());
        self.update3(
            &h_vel,
            gps.velocity - self.v,
            self.cfg.gps_vel_noise * self.cfg.gps_vel_noise,
            self.cfg.chi2_gps_vel,
        );
    }

    fn update_baro(&mut self, baro: &BaroMeas) {
        // Altitude = -p_z, so H[0,2] = -1; innov = altitude - (-p_z).
        let mut h = Mat1_15::zeros();
        h[(0, 2)] = -1.0;
        let innov = baro.altitude - (-self.p.z);
        self.update1(
            &h,
            innov,
            self.cfg.baro_noise * self.cfg.baro_noise,
            self.cfg.chi2_baro,
        );
    }

    fn update_mag(&mut self, mag: &MagMeas) {
        let n = mag.field.norm();
        if n < 1e-9 {
            return;
        }
        let y_hat = self.q.inverse() * magnetic_field_ned();
        let mut h = Mat3_15::zeros();
        h.fixed_view_mut::<3, 3>(0, 6).copy_from(&skew(y_hat)); // H_δθ = +[ŷ]ₓ
        self.update3(
            &h,
            mag.field / n - y_hat,
            self.cfg.mag_noise * self.cfg.mag_noise,
            self.cfg.chi2_mag,
        );
    }

    fn state(&self) -> EstState {
        EstState {
            position: self.p,
            velocity: self.v,
            attitude: self.q,
            angular_rate: self.rate,
        }
    }

    fn gyro_bias_estimate(&self) -> Option<Vec3> {
        Some(self.b_g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::GRAVITY;

    fn accel_for(q: &Quat, a_world: Vec3) -> Vec3 {
        // What the (bias-free, noise-free) IMU reads: R(q)^T (a_world - g_w).
        q.inverse() * (a_world - gravity_world())
    }
    fn mag_for(q: &Quat) -> Vec3 {
        q.inverse() * magnetic_field_ned()
    }

    #[test]
    fn stationary_level_has_zero_world_accel() {
        // accel reads (0,0,-g); a_world = R f + g_w must be ~0 (a -g_w sign here
        // would make a level craft "fall up").
        let mut ins = Ins::default();
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        let v0 = ins.velocity();
        ins.predict(&imu, 1e-3);
        assert!(
            (ins.velocity() - v0).norm() < 1e-9,
            "velocity moved: {:?}",
            ins.velocity()
        );
    }

    #[test]
    fn free_fall_accelerates_downward() {
        // Zero specific force -> a_world = g_w (down, +z) -> v_z grows positive.
        let mut ins = Ins::default();
        let imu = ImuMeas {
            accel: Vec3::zeros(),
            gyro: Vec3::zeros(),
        };
        for _ in 0..100 {
            ins.predict(&imu, 1e-3);
        }
        assert!(
            ins.velocity().z > 0.0,
            "did not fall down (+z): {}",
            ins.velocity().z
        );
    }

    #[test]
    fn velocity_attitude_jacobian_matches_finite_difference() {
        // The single most important INS sign check: d(a_world)/dδθ = -R̂[f̂]ₓ.
        let q = UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3);
        let f_hat = Vec3::new(0.3, -0.5, -9.6); // a tilted specific force
        let r = q.to_rotation_matrix().into_inner();
        let analytic = -r * skew(f_hat); // 3x3, ∂a_w/∂δθ
        let a0 = r * f_hat + gravity_world();
        let eps = 1e-6;
        for axis in 0..3 {
            let mut e = Vec3::zeros();
            e[axis] = eps;
            let q_pert = q * UnitQuaternion::from_scaled_axis(e);
            let a_pert = q_pert.to_rotation_matrix().into_inner() * f_hat + gravity_world();
            let fd = (a_pert - a0) / eps;
            let col = analytic.column(axis);
            assert!(
                (fd - col).norm() < 1e-4,
                "axis {axis}: fd={fd:?} analytic={col:?}"
            );
        }
    }

    #[test]
    fn gps_position_update_pulls_toward_measurement() {
        let mut ins = Ins::default();
        // First fix sets position; second fix should nudge it.
        ins.update_gps(&GpsMeas {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
        });
        // Build some covariance so the update has gain.
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        for _ in 0..100 {
            ins.predict(&imu, 1e-3);
        }
        let before = ins.position().x;
        ins.update_gps(&GpsMeas {
            position: Vec3::new(1.0, 0.0, 0.0),
            velocity: Vec3::zeros(),
        });
        assert!(ins.position().x > before, "GPS pos update did not pull +x");
    }

    #[test]
    fn baro_update_sign_raises_altitude() {
        let mut ins = Ins::default();
        ins.update_gps(&GpsMeas {
            position: Vec3::new(0.0, 0.0, -1.0),
            velocity: Vec3::zeros(),
        });
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        for _ in 0..100 {
            ins.predict(&imu, 1e-3);
        }
        let alt_before = -ins.position().z;
        // Baro says we are 1 m higher than the estimate.
        ins.update_baro(&BaroMeas {
            altitude: alt_before + 1.0,
        });
        assert!(
            -ins.position().z > alt_before,
            "baro update lowered altitude"
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // seed an initial attitude error
    fn ins_fixes_sustained_acceleration_attitude_where_ahrs_drifts() {
        // Truth: level, constant world acceleration (3,0,0) for 8 s. The
        // accelerometer reads the tilted specific force. Both filters START with
        // the same 6° attitude error. The INS treats accel as an input and
        // *recovers* to level (GPS/mag aiding); the AHRS treats accel as gravity
        // and *diverges*. This is the M3 thesis (recovery, not trivial hold).
        use crate::Mekf;
        let q_true = UnitQuaternion::identity();
        let a_world = Vec3::new(3.0, 0.0, 0.0);
        let imu = ImuMeas {
            accel: accel_for(&q_true, a_world),
            gyro: Vec3::zeros(),
        };
        let mag = MagMeas {
            field: mag_for(&q_true),
        };

        let mut ins = Ins::default();
        ins.update_gps(&GpsMeas {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
        });
        let q_err = UnitQuaternion::from_euler_angles(0.07, -0.05, 0.04);
        ins.q = q_err;
        let mut ahrs = Mekf::default();

        let mut v = Vec3::zeros();
        let mut p = Vec3::zeros();
        for k in 0..8000 {
            ins.predict(&imu, 1e-3);
            ahrs.predict(&imu, 1e-3);
            p += v * 1e-3 + a_world * (0.5e-6);
            v += a_world * 1e-3;
            if k % 100 == 0 {
                ins.update_gps(&GpsMeas {
                    position: p,
                    velocity: v,
                });
            }
            if k % 10 == 0 {
                ins.update_mag(&mag);
                ahrs.update_mag(&mag);
            }
        }
        let ins_err = ins.attitude().angle_to(&q_true);
        let ahrs_err = ahrs.attitude().angle_to(&q_true);
        assert!(
            ins_err < 0.02,
            "INS attitude did not recover: {ins_err} rad"
        );
        assert!(
            ahrs_err > 0.05,
            "AHRS unexpectedly stayed level: {ahrs_err} rad"
        );
    }

    #[test]
    #[allow(clippy::field_reassign_with_default)] // seed an initial roll error
    fn ins_recovers_attitude_error_via_velocity_aiding_only() {
        // Isolate the load-bearing velocity↔attitude coupling: level hover, an
        // initial roll error, GPS position+velocity aiding, and DELIBERATELY NO
        // magnetometer. The only path to fix roll is the coupling — the wrong
        // attitude tilts the (gravity-cancelling) specific force, producing a
        // velocity error GPS observes and back-corrects into δθ. This exercises
        // F[3:6,6:9] = -R̂[f̂]ₓ dt inside the real filter, not just the analytic
        // Jacobian.
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        let mut ins = Ins::default();
        ins.update_gps(&GpsMeas {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
        });
        ins.q = UnitQuaternion::from_euler_angles(0.17, 0.0, 0.0); // ~10° roll error
        let err0 = ins.attitude().angle_to(&UnitQuaternion::identity());
        for k in 0..30000 {
            ins.predict(&imu, 1e-3);
            if k % 100 == 0 {
                ins.update_gps(&GpsMeas {
                    position: Vec3::zeros(),
                    velocity: Vec3::zeros(),
                });
            }
            // No mag update — the coupling is the only observability path.
        }
        let err = ins.attitude().angle_to(&UnitQuaternion::identity());
        assert!(
            err < err0 * 0.3,
            "roll not recovered via velocity aiding alone: {err0} -> {err}"
        );
    }

    #[test]
    fn covariance_stays_symmetric_and_psd() {
        let mut ins = Ins::default();
        ins.update_gps(&GpsMeas {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
        });
        let imu = ImuMeas {
            accel: Vec3::new(0.0, 0.0, -GRAVITY),
            gyro: Vec3::zeros(),
        };
        for k in 0..2000 {
            ins.predict(&imu, 1e-3);
            if k % 100 == 0 {
                ins.update_gps(&GpsMeas {
                    position: Vec3::zeros(),
                    velocity: Vec3::zeros(),
                });
            }
            if k % 20 == 0 {
                ins.update_baro(&BaroMeas { altitude: 0.0 });
                ins.update_mag(&MagMeas {
                    field: mag_for(&UnitQuaternion::identity()),
                });
            }
        }
        let c = ins.covariance();
        assert!(
            (c - c.transpose()).norm() < 1e-9,
            "covariance not symmetric"
        );
        let jitter = Mat15::identity() * 1e-12;
        assert!((c + jitter).cholesky().is_some(), "covariance not PSD");
    }
}
