//! Everything needed to build a [`Sim`](crate::Sim), with presets.

use crate::atmosphere::AtmosphereConfig;
use fsim_control::{CascadedConfig, LqrConfig, PositionConfig};
use fsim_core::{Real, DEFAULT_DT};
use fsim_dynamics::MultirotorParams;
use fsim_estimator::{ComplementaryConfig, InsConfig, MekfConfig};
use fsim_sensors::{BaroConfig, GpsConfig, ImuConfig, MagConfig};

/// Which estimator the scheduler runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorKind {
    /// Mahony complementary filter.
    Complementary,
    /// Quaternion MEKF / AHRS.
    Mekf,
    /// 15-state INS — fuses GPS/baro/velocity; the only estimator that
    /// returns real position/velocity (required for position control).
    Ins,
}

/// Which inner attitude/rate controller the scheduler runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerKind {
    /// Cascaded attitude→rate PID.
    Pid,
    /// LQR optimal state feedback.
    Lqr,
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
    /// GPS model (sampled + wired now; position fusion is done by the INS).
    pub gps: GpsConfig,
    /// Barometer model (sampled + wired now; altitude fusion is done by the INS).
    pub baro: BaroConfig,
    /// Magnetometer model (used by the MEKF for yaw).
    pub mag: MagConfig,

    /// Which estimator to run.
    pub estimator_kind: EstimatorKind,
    /// Complementary-filter tuning.
    pub complementary: ComplementaryConfig,
    /// MEKF tuning.
    pub mekf: MekfConfig,
    /// 15-state INS tuning.
    pub ins: InsConfig,
    /// Which inner attitude/rate controller to run.
    pub controller_kind: ControllerKind,
    /// Inner cascaded-PID (attitude→rate) gains/limits.
    pub control: CascadedConfig,
    /// LQR controller weights (used when `controller_kind == Lqr`).
    pub lqr: LqrConfig,
    /// Outer position/velocity controller gains/limits (position mode).
    pub position: PositionConfig,

    /// Mixer arm length \[m\].
    pub arm_length: Real,
    /// Mixer yaw reaction coefficient \[m\].
    pub yaw_coeff: Real,
    /// Per-motor thrust limit \[N\].
    pub max_thrust: Real,
    /// Motor first-order lag \[s\] (0 = ideal, the default).
    pub motor_tau: Real,

    /// Master RNG seed (each sensor derives an independent stream from it).
    pub seed: u64,

    /// The air the quad flies through (wind + turbulence). Defaults to calm.
    pub atmosphere: AtmosphereConfig,
}

impl SimConfig {
    /// A 250-class quad: 1 kHz physics/IMU, 500 Hz control, light IMU
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
            ins: InsConfig::default(),
            controller_kind: ControllerKind::Pid,
            control: CascadedConfig::quad_250(),
            lqr: LqrConfig::quad_250(),
            position: PositionConfig::quad_250(),
            arm_length: 0.12,
            yaw_coeff: 0.016,
            max_thrust: 4.0,
            motor_tau: 0.0,
            seed: 0xC0FFEE,
            atmosphere: AtmosphereConfig::calm(),
        }
    }

    /// Realistic noisy sensors (with gyro bias) + the quaternion MEKF, which
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

    /// Realistic sensors + the 15-state INS + realistic motor lag. The INS
    /// returns real position/velocity, enabling position control + waypoint
    /// guidance (see [`crate::Sim::set_mission`]).
    pub fn quad_250_m3() -> Self {
        Self {
            estimator_kind: EstimatorKind::Ins,
            motor_tau: 0.025, // 25 ms first-order motor lag
            ..Self::quad_250_m2()
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
