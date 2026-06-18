//! Everything needed to build a [`Sim`](crate::Sim), with presets.

use fsim_control::CascadedConfig;
use fsim_core::{Real, DEFAULT_DT};
use fsim_dynamics::MultirotorParams;
use fsim_estimator::{ComplementaryConfig, MekfConfig};
use fsim_sensors::{BaroConfig, GpsConfig, ImuConfig, MagConfig};

/// Which estimator the scheduler runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorKind {
    /// Mahony complementary filter (M1).
    Complementary,
    /// Quaternion MEKF / AHRS (M2).
    Mekf,
}

/// Full simulator configuration. All rates are gated against the base `dt`.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Base physics step \[s\] (the integrator + motors run at `1/dt`).
    pub dt: Real,
    /// IMU + estimator-predict rate \[Hz\].
    pub imu_rate: Real,
    /// Controller (cascaded loops) rate \[Hz\].
    pub control_rate: Real,

    /// Airframe mass/inertia/drag.
    pub params: MultirotorParams,
    /// IMU noise model.
    pub imu: ImuConfig,
    /// GPS model (sampled + wired now; position fusion is M3).
    pub gps: GpsConfig,
    /// Barometer model (sampled + wired now; altitude fusion is M3).
    pub baro: BaroConfig,
    /// Magnetometer model (used by the MEKF for yaw).
    pub mag: MagConfig,

    /// Which estimator to run.
    pub estimator_kind: EstimatorKind,
    /// Complementary-filter tuning.
    pub complementary: ComplementaryConfig,
    /// MEKF tuning.
    pub mekf: MekfConfig,
    /// Cascaded-PID gains/limits.
    pub control: CascadedConfig,

    /// Mixer arm length \[m\].
    pub arm_length: Real,
    /// Mixer yaw reaction coefficient \[m\].
    pub yaw_coeff: Real,
    /// Per-motor thrust limit \[N\].
    pub max_thrust: Real,
    /// Motor first-order lag \[s\] (0 = ideal, the M1 default).
    pub motor_tau: Real,

    /// Master RNG seed (each sensor derives an independent stream from it).
    pub seed: u64,
}

impl SimConfig {
    /// The M1 MVP: 250-class quad, 1 kHz physics/IMU, 500 Hz control, light IMU
    /// noise, complementary filter, ideal motors.
    pub fn quad_250_mvp() -> Self {
        Self {
            dt: DEFAULT_DT,
            imu_rate: 1000.0,
            control_rate: 500.0,
            params: MultirotorParams::quad_250(),
            imu: ImuConfig::mvp(1000.0),
            gps: GpsConfig::mvp(5.0),
            baro: BaroConfig::mvp(25.0),
            mag: MagConfig::mvp(50.0),
            estimator_kind: EstimatorKind::Complementary,
            complementary: ComplementaryConfig::default(),
            mekf: MekfConfig::default(),
            control: CascadedConfig::quad_250(),
            arm_length: 0.12,
            yaw_coeff: 0.016,
            max_thrust: 4.0,
            motor_tau: 0.0,
            seed: 0xC0FFEE,
        }
    }

    /// M2: realistic noisy sensors (with gyro bias) + the quaternion MEKF, which
    /// estimates that bias. The same config with [`EstimatorKind::Complementary`]
    /// shows the filter the MEKF improves on (the CF drifts on the biased gyro).
    pub fn quad_250_m2() -> Self {
        Self {
            imu: ImuConfig::realistic(1000.0),
            gps: GpsConfig::realistic(10.0),
            baro: BaroConfig::realistic(50.0),
            mag: MagConfig::realistic(100.0),
            estimator_kind: EstimatorKind::Mekf,
            ..Self::quad_250_mvp()
        }
    }

    /// Collective thrust that exactly cancels gravity \[N\].
    pub fn hover_thrust(&self) -> Real {
        self.params.mass * fsim_core::GRAVITY
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self::quad_250_mvp()
    }
}
