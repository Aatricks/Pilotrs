//! # fsim-dynamics
//!
//! The "plant": the simulated rigid body. Given the net [`Wrench`] acting on
//! the craft, [`Plant::deriv`] evaluates the Newton-Euler equations of motion,
//! and an [`Integrator`] (RK4) advances the [`State13`] one fixed step,
//! renormalizing the attitude quaternion.
//!
//! [`Wrench`]: fsim_core::Wrench

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

mod fixedwing;
mod forces;
mod integrator;
mod plant;

pub use fixedwing::{
    fixedwing_wrench, short_period_modes, trim, FixedWingParams, ShortPeriodModes, Trim,
};
pub use forces::aerodynamic_wrench;
pub use integrator::{Euler, Integrator, Rk4};
pub use plant::{rigid_body_deriv, MultirotorParams, Plant, RigidBody};
