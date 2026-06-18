//! Recorded time series for plotting estimate-vs-truth-vs-setpoint and for
//! golden-trajectory regression tests.

use fsim_core::{EstState, Real, Setpoint, State13, Vec3};

/// One logged instant.
#[derive(Debug, Clone, Copy)]
pub struct TelemetrySample {
    /// Simulated time \[s\].
    pub t: Real,
    /// Ground truth.
    pub truth: State13,
    /// Estimator output (what the autopilot saw).
    pub estimate: EstState,
    /// Active setpoint.
    pub setpoint: Setpoint,
    /// Actual per-motor thrust \[N\].
    pub motors: [Real; 4],
    /// True gyro bias hidden inside the IMU \[rad/s\].
    pub true_gyro_bias: Vec3,
    /// Estimator's gyro-bias estimate \[rad/s\] (zero if it has none, e.g. CF).
    pub est_gyro_bias: Vec3,
}

/// A growing log of [`TelemetrySample`]s.
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub samples: Vec<TelemetrySample>,
}

impl Telemetry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, s: TelemetrySample) {
        self.samples.push(s);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn last(&self) -> Option<&TelemetrySample> {
        self.samples.last()
    }
}
