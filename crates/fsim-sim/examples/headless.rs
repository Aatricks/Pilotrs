//! Headless demo (no graphics): fly the M2 stack and show the MEKF estimating
//! the IMU's hidden gyro bias and tracking truth, then contrast it with the
//! complementary filter on the *same* sensor stream.
//!
//! Run with `cargo run -p fsim-sim --example headless`.

use fsim_sim::{EstimatorKind, Setpoint, Sim, SimConfig};

fn deg(r: f64) -> f64 {
    r.to_degrees()
}

fn main() {
    // ---- MEKF in the loop, realistic biased sensors ----
    let cfg = SimConfig::quad_250_m2();
    let hover = cfg.hover_thrust();
    let mut sim = Sim::new(cfg);
    sim.set_setpoint(Setpoint::level(hover)); // level hover: bias is observable
                                              // without sustained translation

    println!("MEKF in the loop (realistic IMU with a hidden, wandering gyro bias)\n");
    println!("  t[s]   att_err   | true gyro bias        | MEKF bias estimate");
    println!("  ------------------------------------------------------------------");
    for k in 0..=8000 {
        sim.step();
        if k % 1000 == 0 {
            let err = deg(sim.truth().attitude.angle_to(&sim.estimate().attitude));
            let tb = sim.true_gyro_bias();
            let eb = sim.est_gyro_bias().unwrap();
            println!(
                "  {:4.1}   {:5.2}°   | ({:+.4},{:+.4},{:+.4}) | ({:+.4},{:+.4},{:+.4})",
                sim.time(),
                err,
                tb.x,
                tb.y,
                tb.z,
                eb.x,
                eb.y,
                eb.z,
            );
        }
    }

    // ---- Fair head-to-head: same stream, CF vs MEKF, level/static truth ----
    use fsim_estimator::{ComplementaryFilter, Estimator, Mekf};
    use fsim_sensors::{Imu, Mag, Sensor, Truth};
    let c = SimConfig::quad_250_m2();
    let truth = fsim_sim::State13::at_rest();
    let bundle = Truth {
        state: &truth,
        accel_world: fsim_sim::Vec3::zeros(),
        t: 0.0,
    };
    let mut imu = Imu::new(c.imu, c.seed);
    let mut mag = Mag::new(c.mag, c.seed ^ 0x3333_3333);
    let mut cf = ComplementaryFilter::new(c.complementary);
    let mut mekf = Mekf::new(c.mekf);
    for k in 0..30_000 {
        let m = imu.sample(&bundle);
        cf.predict(&m, 1e-3);
        mekf.predict(&m, 1e-3);
        if k % 10 == 0 {
            let mm = mag.sample(&bundle);
            cf.update_mag(&mm);
            mekf.update_mag(&mm);
        }
    }
    let cf_err = deg(truth.attitude.angle_to(&cf.state().attitude));
    let mekf_err = deg(truth.attitude.angle_to(&mekf.state().attitude));
    println!("\nHead-to-head after 30 s on the same biased-IMU stream (static truth):");
    println!("  complementary filter attitude error: {cf_err:6.2}°  (yaw drifts on the gyro bias)");
    println!("  MEKF                attitude error: {mekf_err:6.2}°  (bias estimated, mag-aided)");

    // Determinism still holds with the full M2 sensor suite.
    let rerun = {
        let mut s2 = Sim::new(SimConfig {
            estimator_kind: EstimatorKind::Mekf,
            ..SimConfig::quad_250_m2()
        });
        s2.set_setpoint(Setpoint::level(hover));
        s2.run_headless(8001);
        *s2.truth()
    };
    let identical = rerun.to_vector() == sim.truth().to_vector();
    println!("\n  deterministic re-run identical: {identical}");
}
