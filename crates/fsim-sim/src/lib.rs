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

mod atmosphere;
mod batch;
mod config;
mod engine;
mod fixedwing;
mod fw_guidance;
mod fwengine;
mod guidance;
mod recording;
mod scheduler;
mod telemetry;

pub use atmosphere::{Atmosphere, AtmosphereConfig};
pub use batch::{
    aggregate, run_batch, run_batch_seq, run_one, seed_sweep, square_mission, summarize_default,
    McSummary, RunMetrics, RunSpec, RunTask,
};
pub use config::{ControllerKind, EstimatorKind, SimConfig};
pub use engine::{Command, EngineClosed, LoggingCfg, RunMode, RunReport, SimEngine, Snapshot};
pub use fixedwing::{cross_track, line_course, FwSample, FwSim, FwSimConfig};
pub use fw_guidance::{FwGuidance, FwGuidanceConfig, TerminalAction};
pub use fwengine::{FwCommand, FwEngine, FwLoggingCfg, FwRunMode, FwRunReport, FwSnapshot};
// Fixed-wing types a front-end commonly needs (re-exported from their crates).
pub use fsim_control::{FixedWingConfig, FixedWingController, FixedWingSetpoint};
pub use fsim_dynamics::{trim, FixedWingParams, Trim};
pub use guidance::{Guidance, GuidanceConfig, Waypoint};
pub use recording::{Recording, ReplayPlayer, RECORDING_VERSION};
pub use scheduler::Sim;
pub use telemetry::{Telemetry, TelemetrySample};

// Re-export the pieces a front-end (viz) commonly needs.
pub use fsim_core::{
    planet, CtrlCmd, EstState, FixedWingControls, Quat, Real, Setpoint, State13, StickInput, Vec3,
    GRAVITY,
};
