//! The vocabulary that flows between subsystems. Kept in `fsim-core` so every
//! crate shares one definition and nothing depends on anything but core.

use crate::{Quat, Real, Vec3};

/// Net external load applied to the rigid body for one integration step.
#[derive(Debug, Clone, Copy)]
pub struct Wrench {
    /// Resultant force in the NED world frame \[N\].
    pub force_world: Vec3,
    /// Resultant moment in the **body** frame \[N·m\].
    pub moment_body: Vec3,
}

impl Wrench {
    /// Zero force and moment.
    pub fn zero() -> Self {
        Self {
            force_world: Vec3::zeros(),
            moment_body: Vec3::zeros(),
        }
    }
}

/// A raw IMU sample (what the estimator's predict step consumes).
///
/// The accelerometer measures **specific force** in the body frame (gravity
/// reaction included): at rest it reads `+g` "up" along body `-z`, not zero.
#[derive(Debug, Clone, Copy)]
pub struct ImuMeas {
    /// Specific force in the body frame \[m/s^2\].
    pub accel: Vec3,
    /// Angular rate in the body frame \[rad/s\].
    pub gyro: Vec3,
}

/// A GPS fix: NED position and velocity (low rate, larger noise than the IMU).
#[derive(Debug, Clone, Copy)]
pub struct GpsMeas {
    /// Position in the NED world frame \[m\].
    pub position: Vec3,
    /// Velocity in the NED world frame \[m/s\].
    pub velocity: Vec3,
}

/// A barometric altitude measurement (height above the launch plane).
///
/// `altitude = -z` (NED down is `+z`), plus a slowly-varying pressure bias and
/// white noise.
#[derive(Debug, Clone, Copy)]
pub struct BaroMeas {
    /// Measured altitude \[m\] (`+up`).
    pub altitude: Real,
}

/// A 3-axis magnetometer sample: the world geomagnetic reference field rotated
/// into the body frame, `m_body = R(q)^T · m_world` (+ noise / hard-iron bias).
/// Provides the heading (yaw) reference the gravity vector cannot.
#[derive(Debug, Clone, Copy)]
pub struct MagMeas {
    /// Field direction in the body frame (units of the reference field).
    pub field: Vec3,
}

/// The estimator's best estimate of the state (what the controller acts on —
/// it never sees truth). Mirrors [`crate::State13`]; fields the current
/// estimator can't observe are filled best-effort (e.g. zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstState {
    /// Estimated position, NED world \[m\].
    pub position: Vec3,
    /// Estimated velocity, NED world \[m/s\].
    pub velocity: Vec3,
    /// Estimated attitude `q_{world<-body}`.
    pub attitude: Quat,
    /// Estimated body angular rate \[rad/s\].
    pub angular_rate: Vec3,
}

impl EstState {
    /// Level, at the origin, at rest.
    pub fn at_rest() -> Self {
        Self {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
            attitude: Quat::identity(),
            angular_rate: Vec3::zeros(),
        }
    }
}

/// Setpoint fed to the autopilot. For the attitude controller this is a
/// desired attitude plus a collective thrust; outer loops will populate it
/// from position/velocity/waypoint guidance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Setpoint {
    /// Desired attitude `q_{world<-body}`.
    pub attitude: Quat,
    /// Desired collective thrust \[N\] (sum across motors).
    pub thrust: Real,
}

impl Setpoint {
    /// Hold level attitude at the given collective thrust.
    pub fn level(thrust: Real) -> Self {
        Self {
            attitude: Quat::identity(),
            thrust,
        }
    }
}

/// The controller's output: collective thrust + desired body torque. The mixer
/// turns this into individual motor commands.
#[derive(Debug, Clone, Copy)]
pub struct CtrlCmd {
    /// Collective thrust \[N\].
    pub thrust: Real,
    /// Desired moment in the body frame \[N·m\] (roll, pitch, yaw).
    pub torque: Vec3,
}

impl CtrlCmd {
    /// Zero thrust and torque.
    pub fn zero() -> Self {
        Self {
            thrust: 0.0,
            torque: Vec3::zeros(),
        }
    }
}

/// A fixed-wing actuator command: control-surface angles \[rad\] and
/// throttle \[0,1\]. The fixed-wing analogue of [`CtrlCmd`].
///
/// Sign conventions (FRD, trailing-edge-down-positive surfaces): `+elevator`
/// pitches **nose-down** (`Cmde < 0`), `+aileron` rolls **right** (`Clda > 0`),
/// `+rudder` yaws **left** (`Cndr < 0`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedWingControls {
    /// Aileron δa \[rad\].
    pub aileron: Real,
    /// Elevator δe \[rad\].
    pub elevator: Real,
    /// Rudder δr \[rad\].
    pub rudder: Real,
    /// Throttle δt \[0,1\].
    pub throttle: Real,
}

impl FixedWingControls {
    /// All surfaces neutral, throttle off.
    pub fn zero() -> Self {
        Self {
            aileron: 0.0,
            elevator: 0.0,
            rudder: 0.0,
            throttle: 0.0,
        }
    }

    /// Clamp surfaces to `±surface_max` and throttle to its range.
    pub fn clamp(self, lim: &ControlLimits) -> Self {
        Self {
            aileron: self.aileron.clamp(-lim.surface_max, lim.surface_max),
            elevator: self.elevator.clamp(-lim.surface_max, lim.surface_max),
            rudder: self.rudder.clamp(-lim.surface_max, lim.surface_max),
            throttle: self.throttle.clamp(lim.throttle.0, lim.throttle.1),
        }
    }
}

/// Actuator limits for a [`FixedWingControls`].
#[derive(Debug, Clone, Copy)]
pub struct ControlLimits {
    /// Max control-surface deflection magnitude \[rad\].
    pub surface_max: Real,
    /// Throttle `(min, max)`.
    pub throttle: (Real, Real),
}

/// A human pilot's normalized stick demand — what a joystick/keyboard produces,
/// before any control law turns it into surfaces. Frame-agnostic *intent*: the
/// fly-by-wire law (or a direct passthrough) maps it onto [`FixedWingControls`].
///
/// Conventions (matching the FRD body frame, pilot's view):
/// - `pitch`: +1 = full **nose-up** demand (pull back), −1 = nose-down.
/// - `roll`:  +1 = roll **right**, −1 = roll left.
/// - `yaw`:   +1 = nose **right** (yaw right), −1 = yaw left.
/// - `throttle`: 0 = idle, 1 = full.
///
/// `pitch/roll/yaw ∈ [−1, 1]`, `throttle ∈ [0, 1]` once [`clamped`](Self::clamped).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StickInput {
    /// Pitch demand ∈ [−1, 1] (+ = nose up).
    pub pitch: Real,
    /// Roll demand ∈ [−1, 1] (+ = roll right).
    pub roll: Real,
    /// Yaw demand ∈ [−1, 1] (+ = yaw right).
    pub yaw: Real,
    /// Throttle ∈ [0, 1].
    pub throttle: Real,
}

impl StickInput {
    /// Centred stick, idle throttle.
    pub fn neutral() -> Self {
        Self {
            pitch: 0.0,
            roll: 0.0,
            yaw: 0.0,
            throttle: 0.0,
        }
    }

    /// Clamp every axis into its valid range (defensive: input devices and
    /// rate-limiters can briefly overshoot).
    pub fn clamped(self) -> Self {
        Self {
            pitch: self.pitch.clamp(-1.0, 1.0),
            roll: self.roll.clamp(-1.0, 1.0),
            yaw: self.yaw.clamp(-1.0, 1.0),
            throttle: self.throttle.clamp(0.0, 1.0),
        }
    }
}
