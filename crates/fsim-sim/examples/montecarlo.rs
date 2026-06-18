//! Faster-than-real-time Monte-Carlo: fly the square mission across many RNG
//! seeds, in parallel, and aggregate mission success + INS tracking error. Each
//! run is independent and deterministic, so the aggregate is reproducible
//! regardless of how many worker threads run it (`run_batch == run_batch_seq`).
//!
//! There is deliberately **no wall-clock / sleep** here — the batch runs as
//! fast as the CPU allows (millions of fixed steps in a second or two).
//!
//! Run with `cargo run -p fsim-sim --release --example montecarlo`.

use fsim_sim::{
    aggregate, run_batch, run_batch_seq, seed_sweep, square_mission, summarize_default,
    GuidanceConfig, RunTask, SimConfig,
};

fn main() {
    let n = 64u64;
    let steps = 40_000; // 40 s per run
    let specs = seed_sweep(
        SimConfig::quad_250_m3(),
        RunTask::Mission {
            waypoints: square_mission(),
            guidance: GuidanceConfig::default(),
        },
        steps,
        n,
    );

    println!("Monte-Carlo: {n} square-mission runs × {steps} steps (INS + position control)");
    println!(
        "({} total fixed steps, parallel, no wall-clock pacing)\n",
        n as usize * steps
    );

    let metrics = run_batch(specs.clone(), 0, summarize_default); // 0 = all cores
    let agg = aggregate(&metrics);

    println!(
        "  missions completed:     {:.0}/{}",
        agg.success_rate * n as f64,
        n
    );
    println!(
        "  final position error:   mean {:.2} m   worst {:.2} m",
        agg.mean_final_position_error, agg.worst_final_position_error
    );
    println!(
        "  RMS INS attitude error: mean {:.2}°  worst {:.2}°",
        agg.mean_rms_attitude_error.to_degrees(),
        agg.worst_rms_attitude_error.to_degrees()
    );
    println!(
        "  worst INS pos tracking: {:.2} m",
        agg.worst_peak_ins_position_error
    );
    println!("  diverged runs:          {}", agg.diverged);

    // Parallelism never changes the answer.
    let seq = run_batch_seq(specs, summarize_default);
    println!("\n  parallel == sequential: {}", metrics == seq);
}
