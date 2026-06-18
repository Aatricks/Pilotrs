//! Headless demo: fly a short attitude maneuver and print truth vs estimate.
//!
//! Run with `cargo run -p fsim-sim --example headless`. Same engine as the
//! viewer, no graphics — useful as a smoke test and for batch/regression runs.

use fsim_sim::{Setpoint, Sim, SimConfig};
use nalgebra::UnitQuaternion;

fn deg(r: f64) -> f64 {
    r.to_degrees()
}

fn main() {
    let cfg = SimConfig::quad_250_mvp();
    let hover = cfg.hover_thrust();
    let mut sim = Sim::new(cfg);

    // A gentle 4° roll / 3° pitch hold — within the complementary filter's
    // honest envelope.
    let sp = Setpoint {
        attitude: UnitQuaternion::from_euler_angles(0.07, -0.05, 0.0),
        thrust: hover,
    };
    sim.set_setpoint(sp);

    println!("  t[s]   truth(roll,pitch)   est(roll,pitch)    est_err   alt[m]");
    println!("  ---------------------------------------------------------------");
    for k in 0..=2000 {
        sim.step();
        if k % 250 == 0 {
            let truth = sim.truth();
            let est = sim.estimate();
            let (tr, tp, _) = truth.attitude.euler_angles();
            let (er, ep, _) = est.attitude.euler_angles();
            let err = deg(truth.attitude.angle_to(&est.attitude));
            println!(
                "  {:4.2}   ({:6.2},{:6.2})     ({:6.2},{:6.2})     {:5.2}°   {:6.3}",
                sim.time(),
                deg(tr),
                deg(tp),
                deg(er),
                deg(ep),
                err,
                -truth.position.z,
            );
        }
    }

    // Determinism: a second identical run lands on the exact same state.
    let rerun = {
        let mut s2 = Sim::new(SimConfig::quad_250_mvp());
        s2.set_setpoint(sp);
        s2.run_headless(2001);
        *s2.truth()
    };
    let identical = rerun.to_vector() == sim.truth().to_vector();
    println!("\n  deterministic re-run identical: {identical}");
}
