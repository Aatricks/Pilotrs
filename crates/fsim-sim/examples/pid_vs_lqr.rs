//! A/B comparison of the two inner attitude controllers: cascaded PID (M1) vs
//! LQR (M5). Both fly the SAME plant, sensors, and seed — only the controller
//! changes — so the step-response and mission-tracking differences are
//! attributable to the controller alone.
//!
//! Run with `cargo run -p fsim-sim --release --example pid_vs_lqr`.

use fsim_actuators::{Mixer, MotorModel, XQuadMixer};
use fsim_control::{CascadedPid, Controller, LqrConfig, LqrController};
use fsim_core::{EstState, State13, GRAVITY};
use fsim_dynamics::{aerodynamic_wrench, Integrator, MultirotorParams, Plant, RigidBody, Rk4};
use fsim_sim::{
    run_batch, seed_sweep, square_mission, ControllerKind, GuidanceConfig, RunSpec, RunTask,
    Setpoint, Sim, SimConfig, Vec3,
};
use nalgebra::UnitQuaternion;

/// Fly a 15° roll step against the true plant with PERFECT feedback (est =
/// truth), isolating the controller from estimator effects, and measure it.
fn step_response(kind: ControllerKind) -> (f64, f64, f64, f64) {
    let target = 0.262_f64; // 15°
    let params = MultirotorParams::quad_250();
    let mut ctrl: Box<dyn Controller> = match kind {
        ControllerKind::Pid => Box::new(CascadedPid::quad_250()),
        ControllerKind::Lqr => {
            let i = &params.inertia;
            let diag = Vec3::new(i[(0, 0)], i[(1, 1)], i[(2, 2)]);
            Box::new(LqrController::new(diag, LqrConfig::quad_250()))
        }
    };
    let body = RigidBody::new(params);
    let mixer = XQuadMixer::quad_250();
    let mut motors = MotorModel::ideal(4.0);
    let rk4 = Rk4;
    let sp = Setpoint {
        attitude: UnitQuaternion::from_euler_angles(target, 0.0, 0.0),
        thrust: params.mass * GRAVITY,
    };
    let mut s = State13::at_rest();
    let mut rolls: Vec<(f64, f64)> = Vec::new();
    for k in 0..3000 {
        let est = EstState {
            position: s.position,
            velocity: s.velocity,
            attitude: s.attitude,
            angular_rate: s.angular_rate,
        };
        let cmd = ctrl.step(&est, &sp, 1e-3);
        let actual = motors.update(&mixer.mix(&cmd), 1e-3);
        let achieved = mixer.collect(&actual);
        s = rk4.step(
            &s,
            |x| {
                body.deriv(
                    x,
                    &aerodynamic_wrench(x, &params, achieved.thrust, achieved.torque),
                )
            },
            1e-3,
        );
        rolls.push((k as f64 * 1e-3, s.attitude.euler_angles().0));
    }

    let rise = rolls
        .iter()
        .find(|(_, r)| *r >= 0.9 * target)
        .map(|(t, _)| *t)
        .unwrap_or(f64::NAN);
    let peak = rolls.iter().map(|(_, r)| *r).fold(0.0_f64, f64::max);
    let overshoot = ((peak - target) / target * 100.0).max(0.0);
    // Settle = last time it left the ±2% band.
    let band = 0.02 * target;
    let settle = rolls
        .iter()
        .rev()
        .find(|(_, r)| (*r - target).abs() > band)
        .map(|(t, _)| *t)
        .unwrap_or(0.0);
    let sse = (rolls.last().unwrap().1 - target).abs();
    (rise, overshoot, settle, sse)
}

fn main() {
    println!("Inner controller A/B: cascaded PID vs LQR (same plant + seed)\n");

    println!("15° roll step response:");
    println!("  controller   rise(90%)   overshoot   settle(2%)   steady-state err");
    for (name, kind) in [("PID", ControllerKind::Pid), ("LQR", ControllerKind::Lqr)] {
        let (rise, os, settle, sse) = step_response(kind);
        println!(
            "  {name:<10}   {:6.3} s    {:6.1} %    {:6.3} s     {:.4} rad",
            rise, os, settle, sse
        );
    }

    // Mission tracking: how well each inner controller tracks the *commanded*
    // attitude (the guidance/position controller's output, now logged), over 32
    // seeds of the square mission with the INS. This is a genuine controller
    // metric: RMS(setpoint − truth), unlike the estimator's truth-vs-estimate
    // error which is controller-independent.
    println!("\nSquare-mission inner-loop attitude tracking, RMS(setpoint−truth), 32 seeds:");
    // Custom summarizer: (completed, rms_tracking_deg, final_pos_err).
    let track = |_spec: &RunSpec, sim: &Sim| -> (bool, f64, f64) {
        let (mut sum, mut n) = (0.0_f64, 0.0_f64);
        for s in sim.telemetry().samples.iter().filter(|s| s.t > 3.0) {
            let e = s.setpoint.attitude.angle_to(&s.truth.attitude);
            sum += e * e;
            n += 1.0;
        }
        (
            sim.waypoint_index() == Some(4),
            (sum / n.max(1.0)).sqrt().to_degrees(),
            (sim.truth().position - Vec3::new(0.0, 0.0, -2.0)).norm(),
        )
    };
    for (name, kind) in [("PID", ControllerKind::Pid), ("LQR", ControllerKind::Lqr)] {
        let mut base = SimConfig::quad_250_m3();
        base.controller_kind = kind;
        let specs = seed_sweep(
            base,
            RunTask::Mission {
                waypoints: square_mission(),
                guidance: GuidanceConfig::default(),
            },
            30_000,
            32,
        );
        let results = run_batch(specs, 0, track);
        let n = results.len() as f64;
        let completed = results.iter().filter(|(c, ..)| *c).count();
        let mean_rms = results.iter().map(|(_, r, _)| r).sum::<f64>() / n;
        let worst_pos = results.iter().map(|(.., p)| *p).fold(0.0_f64, f64::max);
        println!(
            "  {name:<10}  completed {completed}/32   mean RMS track {mean_rms:.2}°   worst final pos {worst_pos:.2} m",
        );
    }
}
