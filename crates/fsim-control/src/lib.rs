//! # fsim-control
//!
//! The autopilot. M1 implements the two innermost loops of the cascade:
//!
//! ```text
//! attitude setpoint --[attitude P]--> desired body rate --[rate PID]--> torque
//! ```
//!
//! The attitude loop turns an attitude error into a desired body rate; the rate
//! loop (the fast inner loop, gyro feedback) turns the rate error into a body
//! torque. Collective thrust passes through from the setpoint. Velocity/
//! position/guidance loops wrap around these in M3; LQR/MPC can replace the
//! inner loops behind the same [`Controller`] trait.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod cascaded;
mod pid;
mod position;

pub use cascaded::{CascadedConfig, CascadedPid};
pub use pid::Pid;
pub use position::{accel_to_setpoint, GuidanceTarget, PositionConfig, PositionController};

use fsim_core::{CtrlCmd, EstState, Real, Setpoint};

/// An autopilot: maps the current estimate + setpoint to an actuator command.
pub trait Controller {
    fn step(&mut self, est: &EstState, sp: &Setpoint, dt: Real) -> CtrlCmd;
}
