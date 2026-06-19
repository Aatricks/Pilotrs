//! # fsim-actuators
//!
//! Closes the loop from the controller back into the dynamics:
//!
//! 1. [`Mixer::mix`] solves the control-allocation problem — turning the
//!    desired `(collective thrust, body torque)` into four individual motor
//!    thrusts via the airframe mixing matrix.
//! 2. [`MotorModel`] applies a first-order lag so motors don't respond
//!    instantly (by default the motors are ideal via [`MotorModel::ideal`];
//!    a nonzero lag models real motors).
//! 3. [`Mixer::collect`] recombines the *actual* motor thrusts back into the
//!    achieved wrench — so motor saturation and lag actually affect the plant.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod mixer;
mod motor;

pub use mixer::{Mixer, XQuadMixer};
pub use motor::MotorModel;
