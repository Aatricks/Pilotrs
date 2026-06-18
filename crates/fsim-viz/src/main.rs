//! # fsim-viz
//!
//! Interactive 3D viewer + live plots for the Pilotrs simulator. This is the
//! std-only leaf of the workspace: all the GPU/windowing/egui baggage lives
//! here and nothing in the flight-control core depends on it.
//!
//! The renderer is a *consumer* of the deterministic [`Sim`]: each display
//! frame it steps the fixed-`dt` physics enough times to catch up to wall-clock
//! time (the standard "fix your timestep" accumulator), then draws the latest
//! state. The physics stays bit-for-bit deterministic; only how many sub-steps
//! run per frame varies.
//!
//! ## Frame convention
//!
//! The scene is drawn directly in the simulator's NED world frame (x = North,
//! y = East, z = Down). The camera's up vector is world `-z`, so altitude
//! (`-z`) points up on screen.

use fsim_sim::{GuidanceConfig, Quat, Setpoint, Sim, SimConfig, Waypoint};
use three_d::egui;
use three_d::*;

const DT: f64 = 0.001;

/// Estimator selection: 0 = complementary filter (M1), 1 = MEKF (M2),
/// 2 = INS (M3). Switching rebuilds the sim.
fn make_cfg(est_kind: u8) -> SimConfig {
    match est_kind {
        0 => SimConfig::quad_250_mvp(),
        1 => SimConfig::quad_250_m2(),
        _ => SimConfig::quad_250_m3(),
    }
}

/// A 5 m square mission at 2 m altitude (NED), returning to the start.
fn square_mission() -> Vec<Waypoint> {
    use fsim_sim::Vec3 as V;
    vec![
        Waypoint::new(V::new(0.0, 0.0, -2.0), 0.0),
        Waypoint::new(V::new(5.0, 0.0, -2.0), 0.0),
        Waypoint::new(V::new(5.0, 5.0, -2.0), 0.0),
        Waypoint::new(V::new(0.0, 5.0, -2.0), 0.0),
        Waypoint::new(V::new(0.0, 0.0, -2.0), 0.0),
    ]
}

/// Convert a simulator world position (nalgebra `f64`) to a three-d `Vec3`.
fn to_v(p: &fsim_sim::Vec3) -> Vec3 {
    vec3(p.x as f32, p.y as f32, p.z as f32)
}

/// Convert a simulator attitude quaternion to a three-d rotation matrix.
fn to_rot(q: &Quat) -> Mat4 {
    let cg = three_d::Quat::new(q.w as f32, q.i as f32, q.j as f32, q.k as f32);
    Mat4::from(cg)
}

fn opaque(context: &Context, r: u8, g: u8, b: u8) -> PhysicalMaterial {
    PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba { r, g, b, a: 255 },
            roughness: 0.6,
            metallic: 0.1,
            ..Default::default()
        },
    )
}

fn main() {
    let window = Window::new(WindowSettings {
        title: "Pilotrs — 6DOF Quadrotor".to_string(),
        max_size: Some((1500, 950)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Camera in NED with up = world -z (altitude points up on screen).
    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(2.6, 2.6, -1.8),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 0.0, -1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(vec3(0.0, 0.0, 0.0), 0.8, 60.0);

    let ambient = AmbientLight::new(&context, 0.5, Srgba::WHITE);
    // Light travelling +z (downward in NED) so it lands on the upward (−z) faces.
    let directional = DirectionalLight::new(&context, 2.5, Srgba::WHITE, vec3(0.4, 0.3, 1.0));

    // Ground: a large thin slab at z = 0.
    let mut ground = Gm::new(
        Mesh::new(&context, &CpuMesh::cube()),
        opaque(&context, 38, 42, 52),
    );
    ground.set_transformation(
        Mat4::from_translation(vec3(0.0, 0.0, 0.02)) * Mat4::from_nonuniform_scale(8.0, 8.0, 0.01),
    );

    // Quad body: a flat box.
    let mut body = Gm::new(
        Mesh::new(&context, &CpuMesh::cube()),
        opaque(&context, 70, 130, 215),
    );

    // Four rotors (front = +x orange, rear = blue) — radius tracks motor thrust.
    let d = 0.12 / std::f32::consts::SQRT_2;
    let rotor_local = [
        vec3(d, d, -0.025),   // M0 front-right
        vec3(-d, d, -0.025),  // M1 rear-right
        vec3(-d, -d, -0.025), // M2 rear-left
        vec3(d, -d, -0.025),  // M3 front-left
    ];
    let rotor_color = [
        (240u8, 140, 40),
        (60, 110, 200),
        (60, 110, 200),
        (240, 140, 40),
    ];
    let mut rotors: Vec<Gm<Mesh, PhysicalMaterial>> = rotor_color
        .iter()
        .map(|&(r, g, b)| {
            Gm::new(
                Mesh::new(&context, &CpuMesh::sphere(16)),
                opaque(&context, r, g, b),
            )
        })
        .collect();

    // Trajectory trail: instanced small spheres.
    let mut trail = Gm::new(
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::sphere(6)),
        opaque(&context, 90, 200, 210),
    );
    let mut trail_pts: Vec<Vec3> = Vec::new();

    // --- simulation + UI state ---
    let mut est_kind: u8 = 2; // default to the M3 INS stack
    let mut mission_on = true; // INS only: fly the square mission
    let cfg = make_cfg(est_kind);
    let hover = cfg.hover_thrust();
    let mut sim = Sim::new(cfg);
    sim.set_logging(5, Some(4000)); // log at 200 Hz, keep a ~20 s window
    if mission_on && est_kind == 2 {
        sim.set_mission(square_mission(), GuidanceConfig::default());
    }

    let mut sp_roll_deg = 0.0_f32;
    let mut sp_pitch_deg = 0.0_f32;
    let mut sp_yaw_deg = 0.0_f32;
    let mut thrust = hover as f32;
    let mut paused = false;
    let mut speed = 1.0_f32;
    let mut accumulator = 0.0_f64;

    let mut gui = GUI::new(&context);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);
        control.handle_events(&mut camera, &mut frame_input.events);

        // In attitude mode, drive from the sliders. In mission mode the INS
        // position controller flies the waypoints, so don't override it.
        let in_mission = mission_on && est_kind == 2;
        if !in_mission {
            sim.set_setpoint(Setpoint {
                attitude: Quat::from_euler_angles(
                    sp_roll_deg.to_radians() as f64,
                    sp_pitch_deg.to_radians() as f64,
                    sp_yaw_deg.to_radians() as f64,
                ),
                thrust: thrust as f64,
            });
        }
        if !paused {
            accumulator += frame_input.elapsed_time * 0.001 * speed as f64;
            let mut n = (accumulator / DT) as i32;
            if n > 60 {
                n = 60; // clamp to avoid the spiral of death after a stall
                accumulator = 0.0;
            }
            for _ in 0..n {
                sim.step();
            }
            accumulator -= n as f64 * DT;
        }

        // --- update 3D transforms from the latest truth ---
        let truth = *sim.truth();
        let motors = sim.motors();
        let pos = to_v(&truth.position);
        let pose = Mat4::from_translation(pos) * to_rot(&truth.attitude);
        body.set_transformation(pose * Mat4::from_nonuniform_scale(0.10, 0.10, 0.02));
        for (i, rotor) in rotors.iter_mut().enumerate() {
            let r = 0.03 + 0.035 * (motors[i] / 4.0) as f32; // size ∝ thrust
            rotor.set_transformation(
                pose * Mat4::from_translation(rotor_local[i]) * Mat4::from_scale(r),
            );
        }

        // --- trail ---
        let moved = trail_pts
            .last()
            .map(|p| (*p - pos).magnitude() > 0.01)
            .unwrap_or(true);
        if !paused && moved {
            trail_pts.push(pos);
            if trail_pts.len() > 1500 {
                trail_pts.remove(0);
            }
        }
        trail.set_instances(&Instances {
            transformations: trail_pts
                .iter()
                .map(|p| Mat4::from_translation(*p) * Mat4::from_scale(0.012))
                .collect(),
            ..Default::default()
        });

        // --- GUI (reads telemetry; edits UI state for next frame) ---
        let mut do_reset = false;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |ui| {
                controls_window(
                    ui,
                    &sim,
                    hover,
                    &mut sp_roll_deg,
                    &mut sp_pitch_deg,
                    &mut sp_yaw_deg,
                    &mut thrust,
                    &mut paused,
                    &mut speed,
                    &mut est_kind,
                    &mut mission_on,
                    &mut do_reset,
                );
                telemetry_window(ui, &sim);
            },
        );
        if do_reset {
            sim = Sim::new(make_cfg(est_kind));
            sim.set_logging(5, Some(4000));
            if mission_on && est_kind == 2 {
                sim.set_mission(square_mission(), GuidanceConfig::default());
            }
            trail_pts.clear();
            accumulator = 0.0;
            sp_roll_deg = 0.0;
            sp_pitch_deg = 0.0;
            sp_yaw_deg = 0.0;
            thrust = hover as f32;
        }

        // --- render scene then GUI on top ---
        let mut objects: Vec<&dyn Object> = vec![&ground, &body, &trail];
        for r in &rotors {
            objects.push(r);
        }
        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.06, 0.07, 0.09, 1.0, 1.0))
            .render(&camera, objects, &[&ambient, &directional])
            .write(|| gui.render())
            .unwrap();

        FrameOutput::default()
    });
}

#[allow(clippy::too_many_arguments)]
fn controls_window(
    ui: &mut egui::Ui,
    sim: &Sim,
    hover: f64,
    roll: &mut f32,
    pitch: &mut f32,
    yaw: &mut f32,
    thrust: &mut f32,
    paused: &mut bool,
    speed: &mut f32,
    est_kind: &mut u8,
    mission_on: &mut bool,
    do_reset: &mut bool,
) {
    egui::Window::new("Flight controls")
        .default_pos([12.0, 12.0])
        .default_width(300.0)
        .show(ui.ctx(), |ui| {
            // Estimator selector — switching rebuilds the sim.
            ui.label("estimator");
            ui.horizontal(|ui| {
                for (k, name) in [(0u8, "CF (M1)"), (1, "MEKF (M2)"), (2, "INS (M3)")] {
                    if ui.radio(*est_kind == k, name).clicked() && *est_kind != k {
                        *est_kind = k;
                        *do_reset = true;
                    }
                }
            });
            // The INS is the only estimator with real position — gate mission mode.
            if *est_kind == 2 {
                if ui.checkbox(mission_on, "fly square mission").changed() {
                    *do_reset = true;
                }
            } else if *mission_on {
                ui.label("(mission needs the INS)");
            }
            ui.separator();

            let mission = *est_kind == 2 && *mission_on;
            ui.add_enabled_ui(!mission, |ui| {
                ui.label("Attitude setpoint (deg)");
                ui.add(egui::Slider::new(roll, -35.0..=35.0).text("roll"));
                ui.add(egui::Slider::new(pitch, -35.0..=35.0).text("pitch"));
                ui.add(egui::Slider::new(yaw, -180.0..=180.0).text("yaw"));
                ui.add(egui::Slider::new(thrust, 0.0..=(4.0 * hover as f32)).text("thrust (N)"));
                if ui.button("hover thrust").clicked() {
                    *thrust = hover as f32;
                }
            });
            if mission {
                if let Some(idx) = sim.waypoint_index() {
                    ui.monospace(format!("mission waypoint: {idx}"));
                }
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.checkbox(paused, "pause");
                if ui.button("reset").clicked() {
                    *do_reset = true;
                }
            });
            ui.add(egui::Slider::new(speed, 0.1..=4.0).text("sim speed ×"));
            ui.separator();

            // Truth vs estimate readout (the autopilot only sees the estimate).
            let truth = sim.truth();
            let est = sim.estimate();
            let (tr, tp, ty) = truth.attitude.euler_angles();
            let (er, ep, ey) = est.attitude.euler_angles();
            let att_err = truth.attitude.angle_to(&est.attitude).to_degrees();
            ui.monospace(format!("t        {:7.2} s", sim.time()));
            ui.monospace(format!("alt (-z) {:7.2} m", -truth.position.z));
            ui.monospace(format!(
                "truth RPY {:6.1} {:6.1} {:6.1}",
                tr.to_degrees(),
                tp.to_degrees(),
                ty.to_degrees()
            ));
            ui.monospace(format!(
                "est   RPY {:6.1} {:6.1} {:6.1}",
                er.to_degrees(),
                ep.to_degrees(),
                ey.to_degrees()
            ));
            ui.monospace(format!("est err   {:6.2} deg", att_err));

            // Position estimate (real only under the INS; zero for CF/MEKF).
            ui.monospace(format!(
                "truth pos {:6.2} {:6.2} {:6.2}",
                truth.position.x, truth.position.y, -truth.position.z
            ));
            ui.monospace(format!(
                "est   pos {:6.2} {:6.2} {:6.2}",
                est.position.x, est.position.y, -est.position.z
            ));

            // Gyro-bias estimation is the MEKF's M2 win — show true vs estimate.
            if let Some(eb) = sim.est_gyro_bias() {
                let tb = sim.true_gyro_bias();
                ui.separator();
                ui.monospace(format!(
                    "gyro bias true {:+.4} {:+.4} {:+.4}",
                    tb.x, tb.y, tb.z
                ));
                ui.monospace(format!(
                    "gyro bias est  {:+.4} {:+.4} {:+.4}",
                    eb.x, eb.y, eb.z
                ));
            }
        });
}

fn telemetry_window(ui: &mut egui::Ui, sim: &Sim) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};

    let samples = &sim.telemetry().samples;
    // Build [t, deg] series for an euler component selected by `axis` (0/1/2)
    // from a quaternion extracted by `pick`.
    let series = |pick: &dyn Fn(&fsim_sim::TelemetrySample) -> (f64, f64, f64),
                  axis: usize|
     -> Vec<[f64; 2]> {
        samples
            .iter()
            .map(|s| {
                let e = pick(s);
                let v = [e.0, e.1, e.2][axis].to_degrees();
                [s.t, v]
            })
            .collect()
    };

    egui::Window::new("Estimate vs truth vs setpoint")
        .default_pos([12.0, 360.0])
        .default_width(440.0)
        .show(ui.ctx(), |ui| {
            for (axis, name) in [(0usize, "roll"), (1, "pitch"), (2, "yaw")] {
                ui.label(format!("{name} (deg)"));
                Plot::new(format!("plot_{name}"))
                    .height(110.0)
                    .legend(Legend::default())
                    .show(ui, |p| {
                        p.line(
                            Line::new(
                                "setpoint",
                                PlotPoints::from(series(
                                    &|s| s.setpoint.attitude.euler_angles(),
                                    axis,
                                )),
                            )
                            .color(egui::Color32::GRAY),
                        );
                        p.line(
                            Line::new(
                                "truth",
                                PlotPoints::from(series(
                                    &|s| s.truth.attitude.euler_angles(),
                                    axis,
                                )),
                            )
                            .color(egui::Color32::from_rgb(90, 200, 210)),
                        );
                        p.line(
                            Line::new(
                                "estimate",
                                PlotPoints::from(series(
                                    &|s| s.estimate.attitude.euler_angles(),
                                    axis,
                                )),
                            )
                            .color(egui::Color32::from_rgb(240, 140, 40)),
                        );
                    });
            }
            // Position estimate vs truth (M3 INS only; CF/MEKF leave it at zero).
            if samples.iter().any(|s| s.estimate.position.norm() > 1e-6) {
                ui.label("position truth vs est (m)");
                let pos = |sel: &dyn Fn(&fsim_sim::TelemetrySample) -> f64| -> Vec<[f64; 2]> {
                    samples.iter().map(|s| [s.t, sel(s)]).collect()
                };
                let truth_c = egui::Color32::from_rgb(90, 200, 210);
                let est_c = egui::Color32::from_rgb(240, 140, 40);
                Plot::new("plot_pos")
                    .height(130.0)
                    .legend(Legend::default())
                    .show(ui, |p| {
                        p.line(
                            Line::new("N truth", PlotPoints::from(pos(&|s| s.truth.position.x)))
                                .color(truth_c),
                        );
                        p.line(
                            Line::new("N est", PlotPoints::from(pos(&|s| s.estimate.position.x)))
                                .color(est_c),
                        );
                        p.line(
                            Line::new("E truth", PlotPoints::from(pos(&|s| s.truth.position.y)))
                                .color(truth_c),
                        );
                        p.line(
                            Line::new("E est", PlotPoints::from(pos(&|s| s.estimate.position.y)))
                                .color(est_c),
                        );
                        p.line(
                            Line::new("Up truth", PlotPoints::from(pos(&|s| -s.truth.position.z)))
                                .color(truth_c),
                        );
                        p.line(
                            Line::new("Up est", PlotPoints::from(pos(&|s| -s.estimate.position.z)))
                                .color(est_c),
                        );
                    });
            }

            ui.label("motor thrust (N)");
            Plot::new("plot_motors")
                .height(110.0)
                .legend(Legend::default())
                .show(ui, |p| {
                    for m in 0..4 {
                        let pts: Vec<[f64; 2]> =
                            samples.iter().map(|s| [s.t, s.motors[m]]).collect();
                        p.line(Line::new(format!("m{m}"), PlotPoints::from(pts)));
                    }
                });

            // Gyro-bias estimate vs the hidden truth (the MEKF's M2 win). Only
            // meaningful when the MEKF runs; the CF leaves its estimate at zero.
            if sim.est_gyro_bias().is_some() {
                ui.label("gyro bias est vs true (rad/s)");
                let axis_pts = |sel: &dyn Fn(&fsim_sim::TelemetrySample) -> f64| -> Vec<[f64; 2]> {
                    samples.iter().map(|s| [s.t, sel(s)]).collect()
                };
                Plot::new("plot_bias")
                    .height(120.0)
                    .legend(Legend::default())
                    .show(ui, |p| {
                        for (i, name) in ["x", "y", "z"].iter().enumerate() {
                            p.line(
                                Line::new(
                                    format!("true {name}"),
                                    PlotPoints::from(axis_pts(&|s| s.true_gyro_bias[i])),
                                )
                                .color(egui::Color32::from_rgb(90, 200, 210)),
                            );
                            p.line(
                                Line::new(
                                    format!("est {name}"),
                                    PlotPoints::from(axis_pts(&|s| s.est_gyro_bias[i])),
                                )
                                .color(egui::Color32::from_rgb(240, 140, 40)),
                            );
                        }
                    });
            }
        });
}
