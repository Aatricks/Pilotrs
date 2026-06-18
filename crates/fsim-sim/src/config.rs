//! Everything needed to build a [`Sim`](crate::Sim), with MVP defaults.

use fsim_control::CascadedConfig;
use fsim_core::{Real, DEFAULT_DT};
use fsim_dynamics::MultirotorParams;
use fsim_estimator::ComplementaryConfig;
use fsim_sensors::ImuConfig;

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
    /// Complementary-filter tuning.
    pub estimator: ComplementaryConfig,
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
    /// The M1 MVP: 250-class quad, 1 kHz physics/IMU, 500 Hz control, light
    /// IMU noise, ideal motors.
    pub fn quad_250_mvp() -> Self {
        Self {
            dt: DEFAULT_DT,
            imu_rate: 1000.0,
            control_rate: 500.0,
            params: MultirotorParams::quad_250(),
            imu: ImuConfig::mvp(1000.0),
            estimator: ComplementaryConfig::default(),
            control: CascadedConfig::quad_250(),
            arm_length: 0.12,
            yaw_coeff: 0.016,
            max_thrust: 4.0,
            motor_tau: 0.0,
            seed: 0xC0FFEE,
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
