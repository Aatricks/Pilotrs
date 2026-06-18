//! Headless demo (no graphics) of the M3 stack: the 15-state INS flies a square
//! waypoint mission with position/velocity control, realistic GPS/baro/mag
//! noise, and motor lag. Prints waypoint progress + INS-vs-truth tracking, then
//! the M2 contrast (MEKF estimates gyro bias; CF drifts).
//!
//! Run with `cargo run -p fsim-sim --example headless`.

use fsim_sim::{GuidanceConfig, Sim, SimConfig, Vec3, Waypoint};

fn deg(r: f64) -> f64 {
    r.to_degrees()
}

fn main() {
    // ---- M3: INS + position control flying a square mission ----
    let mut sim = Sim::new(SimConfig::quad_250_m3());
    sim.set_mission(
        vec![
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 5.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(0.0, 5.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
        ],
        GuidanceConfig::default(),
    );

    println!("M3: 15-state INS flying a 5 m square mission (position control)\n");
    println!("  t[s]  wp   truth N,E,Up [m]        INS est N,E,Up [m]      |est-truth|");
    println!("  ------------------------------------------------------------------------");
    let mut max_track = 0.0_f64;
    for k in 0..30000 {
        sim.step();
        if k > 2000 {
            max_track = max_track.max((sim.truth().position - sim.estimate().position).norm());
        }
        if k % 3000 == 0 {
            let t = *sim.truth();
            let e = sim.estimate();
            println!(
                "  {:4.1}   {}   ({:6.2},{:6.2},{:6.2})   ({:6.2},{:6.2},{:6.2})   {:.2}",
                sim.time(),
                sim.waypoint_index().unwrap(),
                t.position.x,
                t.position.y,
                -t.position.z,
                e.position.x,
                e.position.y,
                -e.position.z,
                (t.position - e.position).norm(),
            );
        }
    }
    let p = sim.truth().position;
    println!(
        "\n  mission waypoint reached: {}/4   final pos ({:.2},{:.2},{:.2})   max |est-truth| (after settle): {:.2} m",
        sim.waypoint_index().unwrap(),
        p.x,
        p.y,
        -p.z,
        max_track,
    );

    // ---- M2 contrast: MEKF estimates the gyro bias; the CF drifts ----
    use fsim_estimator::{ComplementaryFilter, Estimator, Mekf};
    use fsim_sensors::{Imu, Mag, Sensor, Truth};
    let c = SimConfig::quad_250_m2();
    let truth = fsim_sim::State13::at_rest();
    let bundle = Truth {
        state: &truth,
        accel_world: Vec3::zeros(),
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
    println!("\nM2 estimator contrast (same biased-IMU stream, static truth, 30 s):");
    println!(
        "  complementary filter attitude error: {:6.2} deg  (yaw drifts on the gyro bias)",
        deg(truth.attitude.angle_to(&cf.state().attitude))
    );
    println!(
        "  MEKF                attitude error: {:6.2} deg  (bias estimated, mag-aided)",
        deg(truth.attitude.angle_to(&mekf.state().attitude))
    );

    // Determinism with the full M3 stack.
    let rerun = {
        let mut s2 = Sim::new(SimConfig::quad_250_m3());
        s2.set_mission(
            vec![Waypoint::new(Vec3::new(2.0, 0.0, -2.0), 0.5)],
            GuidanceConfig::default(),
        );
        s2.run_headless(5000);
        *s2.truth()
    };
    let mut s3 = Sim::new(SimConfig::quad_250_m3());
    s3.set_mission(
        vec![Waypoint::new(Vec3::new(2.0, 0.0, -2.0), 0.5)],
        GuidanceConfig::default(),
    );
    s3.run_headless(5000);
    println!(
        "\n  deterministic M3 re-run identical: {}",
        rerun.to_vector() == s3.truth().to_vector()
    );
}
