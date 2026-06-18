//! Record a mission to a `.fsimrec` file, load it back, and replay it — all
//! without re-running the simulator. Demonstrates the bit-exact round-trip and
//! the time-indexed [`ReplayPlayer`].
//!
//! Run with `cargo run -p fsim-sim --example record_replay`.

use fsim_sim::{GuidanceConfig, Recording, ReplayPlayer, Sim, SimConfig, Vec3, Waypoint};

fn main() -> std::io::Result<()> {
    // Fly a short mission and record it to a file.
    let mut sim = Sim::new(SimConfig::quad_250_m3());
    sim.set_logging(10, None); // 100 Hz, keep everything
    sim.set_mission(
        vec![
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(4.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
        ],
        GuidanceConfig::default(),
    );
    let path = std::env::temp_dir().join("pilotrs_demo.fsimrec");
    sim.record_headless(20_000, &path)?;
    let bytes = std::fs::metadata(&path)?.len();
    println!(
        "recorded 20000 steps -> {} ({} KB)",
        path.display(),
        bytes / 1024
    );

    // Load it back (no Sim needed) and confirm it round-tripped exactly.
    let rec = Recording::load(&path)?;
    println!(
        "loaded {} samples, {:.1} s of flight",
        rec.len(),
        rec.duration()
    );

    // Replay: sample the recorded trajectory at a few wall-clock instants.
    let player = ReplayPlayer::new(&rec);
    println!("\n  t[s]   replayed truth pos (N,E,Up)   waypoint progress (via est)");
    for &t in &[0.0, 5.0, 10.0, 15.0, 19.9] {
        if let Some(s) = player.sample_at(t) {
            println!(
                "  {:4.1}   ({:6.2},{:6.2},{:6.2})",
                t, s.truth.position.x, s.truth.position.y, -s.truth.position.z
            );
        }
    }

    // The recording replays the exact stored samples (bit-for-bit).
    let exact = rec
        .samples
        .iter()
        .zip(ReplayPlayer::new(&rec).iter_all())
        .all(|(a, b)| a == b);
    println!("\n  replay reproduces recording exactly: {exact}");

    let _ = std::fs::remove_file(&path);
    Ok(())
}
