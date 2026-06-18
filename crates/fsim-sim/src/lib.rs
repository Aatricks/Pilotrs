//! # fsim-sim
//!
//! The deterministic fixed-step engine. Each base step (`dt`, 1 kHz) it runs
//! the full loop — truth → IMU → estimator → controller → mixer → motors →
//! dynamics → RK4 → truth' — gating each subsystem to its own rate. Nothing
//! reads the wall clock and all RNG is seeded, so a run is bit-for-bit
//! reproducible (see the determinism test).
//!
//! The same engine drives both the headless test/batch path
//! ([`Sim::run_headless`]) and the interactive viewer (which calls
//! [`Sim::step`] from its render loop).

mod config;
mod scheduler;
mod telemetry;

pub use config::{EstimatorKind, SimConfig};
pub use scheduler::Sim;
pub use telemetry::{Telemetry, TelemetrySample};

// Re-export the pieces a front-end (viz) commonly needs.
pub use fsim_core::{CtrlCmd, EstState, Quat, Real, Setpoint, State13, Vec3, GRAVITY};
