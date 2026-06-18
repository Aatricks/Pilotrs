//! Faster-than-real-time batch / Monte-Carlo harness.
//!
//! Each run is a plain [`Sim`] driven by [`Sim::run_headless`] as fast as the
//! CPU allows — no wall-clock pacing, no threads-that-publish. Every run owns
//! its own seeded RNG streams with zero shared state, so the aggregate is
//! **reproducible regardless of parallelism**: `run_batch(specs, W, f)` equals
//! `run_batch_seq(specs, f)` for any worker count `W` (a tested invariant).

use crate::config::{EstimatorKind, SimConfig};
use crate::guidance::{GuidanceConfig, Waypoint};
use crate::{Setpoint, Sim};
use fsim_core::Tick;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

/// What a run does after construction.
#[derive(Debug, Clone)]
pub enum RunTask {
    /// Hold a fixed attitude/thrust setpoint.
    Attitude(Setpoint),
    /// Fly a waypoint mission (requires an INS config).
    Mission {
        waypoints: Vec<Waypoint>,
        guidance: GuidanceConfig,
    },
}

/// A single batch job.
#[derive(Debug, Clone)]
pub struct RunSpec {
    pub config: SimConfig,
    pub task: RunTask,
    pub steps: usize,
    pub log_every: Tick,
    pub log_cap: Option<usize>,
}

impl RunSpec {
    fn build(&self) -> Sim {
        let mut s = Sim::new(self.config);
        s.set_logging(self.log_every, self.log_cap);
        match &self.task {
            RunTask::Attitude(sp) => s.set_setpoint(*sp),
            RunTask::Mission {
                waypoints,
                guidance,
            } => s.set_mission(waypoints.clone(), *guidance),
        }
        s
    }
}

/// Per-run metrics from [`summarize_default`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunMetrics {
    pub seed: u64,
    pub estimator: EstimatorKind,
    /// `Some` for missions: did it reach the final waypoint?
    pub mission_completed: Option<bool>,
    /// `Some` for missions: distance from the final waypoint at the end \[m\].
    pub final_position_error: Option<f64>,
    pub rms_ins_position_error: f64,
    pub rms_attitude_error: f64,
    pub max_tilt: f64,
    pub peak_ins_position_error: f64,
    /// False if any truth/estimate field went non-finite (a blown-up run).
    pub finite: bool,
}

/// Walk a finished run's telemetry once and summarize it.
pub fn summarize_default(spec: &RunSpec, sim: &Sim) -> RunMetrics {
    let samples = &sim.telemetry().samples;
    let (mut sum_pos2, mut sum_att2, mut max_tilt, mut peak_pos) = (0.0, 0.0, 0.0_f64, 0.0_f64);
    let mut finite = true;
    for s in samples {
        let pe = (s.truth.position - s.estimate.position).norm();
        sum_pos2 += pe * pe;
        peak_pos = peak_pos.max(pe);
        let ae = s.truth.attitude.angle_to(&s.estimate.attitude);
        sum_att2 += ae * ae;
        max_tilt = max_tilt.max(s.truth.attitude.angle());
        if !s.truth.position.iter().all(|x| x.is_finite())
            || !s.estimate.position.iter().all(|x| x.is_finite())
        {
            finite = false;
        }
    }
    let n = samples.len().max(1) as f64;
    let (mission_completed, final_position_error) = match &spec.task {
        RunTask::Mission { waypoints, .. } => {
            let last = waypoints.len().saturating_sub(1);
            let completed = sim.waypoint_index() == Some(last);
            let fpe = waypoints
                .last()
                .map(|w| (sim.truth().position - w.position).norm());
            (Some(completed), fpe)
        }
        RunTask::Attitude(_) => (None, None),
    };
    RunMetrics {
        seed: spec.config.seed,
        estimator: spec.config.estimator_kind,
        mission_completed,
        final_position_error,
        rms_ins_position_error: (sum_pos2 / n).sqrt(),
        rms_attitude_error: (sum_att2 / n).sqrt(),
        max_tilt,
        peak_ins_position_error: peak_pos,
        finite,
    }
}

/// Build, run, and summarize one job.
pub fn run_one<S>(spec: &RunSpec, f: &impl Fn(&RunSpec, &Sim) -> S) -> S {
    let mut sim = spec.build();
    sim.run_headless(spec.steps);
    f(spec, &sim)
}

/// Run all jobs sequentially.
pub fn run_batch_seq<S>(
    specs: impl IntoIterator<Item = RunSpec>,
    f: impl Fn(&RunSpec, &Sim) -> S,
) -> Vec<S> {
    specs.into_iter().map(|s| run_one(&s, &f)).collect()
}

/// Run all jobs across `workers` threads (0 = available parallelism), returning
/// results in **input order** regardless of completion order. Determinism is
/// independent of `workers`.
pub fn run_batch<S: Send>(
    specs: Vec<RunSpec>,
    workers: usize,
    f: impl Fn(&RunSpec, &Sim) -> S + Sync,
) -> Vec<S> {
    let n = specs.len();
    if n == 0 {
        return Vec::new();
    }
    let workers = if workers == 0 {
        thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    } else {
        workers
    }
    .clamp(1, n);

    let results: Vec<Mutex<Option<S>>> = (0..n).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let (specs, results, next, f) = (&specs, &results, &next, &f);

    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(move || loop {
                // Atomic work-stealing index: dynamic load balance across
                // uneven run costs.
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n {
                    break;
                }
                let s = run_one(&specs[i], f);
                *results[i].lock().unwrap() = Some(s); // each slot written once
            });
        }
    });

    results
        .iter()
        .map(|m| m.lock().unwrap().take().expect("every slot filled"))
        .collect()
}

/// Build a seed sweep: `n` copies of `base` with `seed = base.seed + i`.
pub fn seed_sweep(base: SimConfig, task: RunTask, steps: usize, n: u64) -> Vec<RunSpec> {
    (0..n)
        .map(|i| {
            let mut config = base;
            config.seed = base.seed.wrapping_add(i);
            RunSpec {
                config,
                task: task.clone(),
                steps,
                log_every: 10,
                log_cap: None,
            }
        })
        .collect()
}

/// Aggregate Monte-Carlo statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McSummary {
    pub n: usize,
    pub success_rate: f64,
    pub mean_rms_attitude_error: f64,
    pub worst_rms_attitude_error: f64,
    pub mean_final_position_error: f64,
    pub worst_final_position_error: f64,
    pub worst_peak_ins_position_error: f64,
    pub diverged: usize,
}

pub fn aggregate(metrics: &[RunMetrics]) -> McSummary {
    let n = metrics.len();
    let nf = n.max(1) as f64;
    let success = metrics
        .iter()
        .filter(|m| m.mission_completed == Some(true))
        .count();
    let diverged = metrics.iter().filter(|m| !m.finite).count();
    let mean = |f: &dyn Fn(&RunMetrics) -> f64| metrics.iter().map(f).sum::<f64>() / nf;
    let worst = |f: &dyn Fn(&RunMetrics) -> f64| metrics.iter().map(f).fold(0.0_f64, f64::max);
    McSummary {
        n,
        success_rate: if n > 0 { success as f64 / nf } else { 0.0 },
        mean_rms_attitude_error: mean(&|m| m.rms_attitude_error),
        worst_rms_attitude_error: worst(&|m| m.rms_attitude_error),
        mean_final_position_error: mean(&|m| m.final_position_error.unwrap_or(0.0)),
        worst_final_position_error: worst(&|m| m.final_position_error.unwrap_or(0.0)),
        worst_peak_ins_position_error: worst(&|m| m.peak_ins_position_error),
        diverged,
    }
}

/// A standard 5 m square mission at 2 m altitude (NED), returning to start.
pub fn square_mission() -> Vec<Waypoint> {
    use fsim_core::Vec3;
    vec![
        Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
        Waypoint::new(Vec3::new(5.0, 0.0, -2.0), 0.0),
        Waypoint::new(Vec3::new(5.0, 5.0, -2.0), 0.0),
        Waypoint::new(Vec3::new(0.0, 5.0, -2.0), 0.0),
        Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(n: u64, steps: usize) -> Vec<RunSpec> {
        seed_sweep(
            SimConfig::quad_250_m3(),
            RunTask::Mission {
                waypoints: square_mission(),
                guidance: GuidanceConfig::default(),
            },
            steps,
            n,
        )
    }

    #[test]
    fn parallel_matches_sequential() {
        let s = specs(16, 4000);
        let seq = run_batch_seq(s.clone(), summarize_default);
        for w in [1usize, 2, 8] {
            let par = run_batch(s.clone(), w, summarize_default);
            assert_eq!(seq, par, "parallelism W={w} perturbed determinism");
        }
    }

    #[test]
    fn output_order_matches_input_order() {
        let s = specs(20, 2000);
        let m = run_batch(s.clone(), 8, summarize_default);
        for (spec, metric) in s.iter().zip(&m) {
            assert_eq!(spec.config.seed, metric.seed, "results out of order");
        }
    }

    #[test]
    fn empty_and_single_batches() {
        assert!(run_batch(Vec::new(), 4, summarize_default).is_empty());
        let one = run_batch(specs(1, 1000), 4, summarize_default);
        assert_eq!(one.len(), 1);
    }

    #[test]
    fn run_one_matches_inline() {
        let spec = &specs(1, 3000)[0];
        let m = run_one(spec, &summarize_default);
        let mut sim = spec.build();
        sim.run_headless(spec.steps);
        assert_eq!(m, summarize_default(spec, &sim));
    }

    #[test]
    fn aggregate_reports_success() {
        let m = run_batch(specs(8, 30_000), 0, summarize_default);
        let agg = aggregate(&m);
        assert_eq!(agg.n, 8);
        assert!(
            agg.success_rate > 0.99,
            "missions should complete: {}",
            agg.success_rate
        );
        assert_eq!(agg.diverged, 0);
    }
}
