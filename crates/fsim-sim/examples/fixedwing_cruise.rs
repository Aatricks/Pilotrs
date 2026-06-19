//! Fixed-wing demo: a trimmed Aerosonde flies a sequence of autopilot
//! setpoints — hold cruise, climb, change heading, change airspeed — proving the
//! same `State13` + RK4 + rigid-body EOM that flies the quad also flies a totally
//! different airframe with its own aero model and autopilot.
//!
//! Run with `cargo run -p fsim-sim --release --example fixedwing_cruise`.

use fsim_sim::{trim, FixedWingParams, FixedWingSetpoint, FwSim, FwSimConfig};

fn main() {
    let params = FixedWingParams::aerosonde();
    let tr = trim(&params, 25.0, 0.0).expect("Aerosonde 25 m/s level trim converges");
    let (_, theta, _) = tr.state.attitude.euler_angles();
    println!("Aerosonde fixed-wing — reuses the quad's RK4 + rigid-body EOM\n");
    println!(
        "trim @ 25 m/s level: alpha/theta {:.2}°  elevator {:.3} rad  throttle {:.2}\n",
        theta.to_degrees(),
        tr.controls.elevator,
        tr.controls.throttle
    );

    let mut sim = FwSim::new(FwSimConfig::aerosonde_cruise());
    let report = |sim: &FwSim, label: &str| {
        println!(
            "  {label:<22}  t={:5.1}s   Va={:5.2} m/s   alt={:6.2} m   course={:6.1}°",
            sim.time(),
            sim.airspeed(),
            sim.altitude(),
            sim.course().to_degrees(),
        );
    };

    // Each phase: set a target, fly to it, report.
    let phases: &[(&str, f64, FixedWingSetpoint)] = &[
        (
            "hold cruise",
            10.0,
            FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 100.0,
                course: 0.0,
            },
        ),
        (
            "climb to 150 m",
            40.0,
            FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 150.0,
                course: 0.0,
            },
        ),
        (
            "turn East (+90°)",
            45.0,
            FixedWingSetpoint {
                airspeed: 25.0,
                altitude: 150.0,
                course: std::f64::consts::FRAC_PI_2,
            },
        ),
        (
            "accelerate to 30 m/s",
            40.0,
            FixedWingSetpoint {
                airspeed: 30.0,
                altitude: 150.0,
                course: std::f64::consts::FRAC_PI_2,
            },
        ),
    ];

    println!("Autopilot setpoint sequence (flying on truth feedback):");
    for (label, secs, sp) in phases {
        sim.set_setpoint(*sp);
        sim.run_headless((*secs / 0.001) as usize);
        report(&sim, label);
    }

    println!("\nThe airframe-agnostic core (State13, Rk4, rigid_body_deriv) is shared");
    println!("verbatim with the quadrotor — only the wrench + autopilot differ.");
}
