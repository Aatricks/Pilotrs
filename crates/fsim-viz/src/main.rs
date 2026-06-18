//! # fsim-viz
//!
//! Interactive 3D viewer + live plots for the Pilotrs simulator. This is the
//! std-only leaf of the workspace: all the GPU/windowing/egui baggage lives
//! here and nothing in the flight-control core depends on it.
//!
//! The renderer is a pure *consumer*: the deterministic physics runs on its own
//! thread inside a `SimEngine` (M4), and each display frame the viewer just
//! reads the latest published `Snapshot` and sends UI changes as commands —
//! physics and rendering are fully decoupled. The same viewer can instead play
//! back a recorded run (`Source::Replay`).
//!
//! ## Frame convention
//!
//! The scene is drawn directly in the simulator's NED world frame (x = North,
//! y = East, z = Down). The camera's up vector is world `-z`, so altitude
//! (`-z`) points up on screen.

mod replay;

use fsim_sim::{
    Command, GuidanceConfig, Quat, Recording, Setpoint, SimConfig, Snapshot, TelemetrySample,
    Waypoint,
};
use replay::{ReplayState, Source};
use three_d::egui;
use three_d::*;

/// Mutable viewer state, edited by the controls window each frame.
struct Ui {
    est_kind: u8,
    mission_on: bool,
    roll: f32,
    pitch: f32,
    yaw: f32,
    thrust: f32,
    paused: bool,
    speed: f32,
    recording: bool,
    // One-shot actions, cleared each frame after they are applied.
    do_reset: bool,
    do_save: bool,
    do_replay: bool,
    do_live: bool,
    replay_toggle_play: bool,
    replay_seek: Option<f64>,
}

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
    let hover = make_cfg(2).hover_thrust();
    let mut ui = Ui {
        est_kind: 2,      // default to the M3 INS stack
        mission_on: true, // INS only: fly the square mission
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        thrust: hover as f32,
        paused: false,
        speed: 1.0,
        recording: false,
        do_reset: false,
        do_save: false,
        do_replay: false,
        do_live: false,
        replay_toggle_play: false,
        replay_seek: None,
    };

    // The sim runs on its own thread; the viewer reads snapshots and sends
    // commands (this is the M4 physics/render decoupling).
    let mut source = Source::live(make_cfg(ui.est_kind));
    if ui.mission_on && ui.est_kind == 2 {
        source.command(Command::SetMission {
            waypoints: square_mission(),
            guidance: GuidanceConfig::default(),
        });
    }
    let record_path = std::env::temp_dir().join("pilotrs_live.fsimrec");

    let mut gui = GUI::new(&context);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);
        control.handle_events(&mut camera, &mut frame_input.events);

        let dt_frame = frame_input.elapsed_time * 0.001;
        source.tick(dt_frame); // advances replay; the live engine paces itself

        // Drive the live engine from the UI (no-op in replay mode). The physics
        // runs on its own thread; these are just commands.
        let in_mission = ui.mission_on && ui.est_kind == 2;
        if !source.is_replay() {
            source.command(Command::Pause(ui.paused));
            source.command(Command::SetSpeed(ui.speed as f64));
            if !in_mission {
                source.command(Command::SetSetpoint(Setpoint {
                    attitude: Quat::from_euler_angles(
                        ui.roll.to_radians() as f64,
                        ui.pitch.to_radians() as f64,
                        ui.yaw.to_radians() as f64,
                    ),
                    thrust: ui.thrust as f64,
                }));
            }
        }

        let snap = source.snapshot();
        let telem = source.telemetry();

        // --- update 3D transforms from the snapshot ---
        let truth = snap.truth;
        let motors = snap.motors;
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
        if moved {
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

        // --- GUI (reads the snapshot; edits UI state for next frame) ---
        let replay_info = if let Source::Replay(r) = &source {
            let (t0, t1) = r.range();
            Some((r.time(), t0, t1))
        } else {
            None
        };
        ui.do_reset = false;
        ui.do_save = false;
        ui.do_replay = false;
        ui.do_live = false;
        ui.replay_toggle_play = false;
        ui.replay_seek = None;
        let prev_recording = ui.recording;
        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_ui| {
                controls_window(egui_ui, &snap, hover, &mut ui, replay_info);
                telemetry_window(egui_ui, &telem);
            },
        );

        // --- apply UI actions ---
        if ui.recording != prev_recording {
            source.command(Command::Record(ui.recording));
        }
        if ui.do_save {
            source.command(Command::SaveRecording(record_path.clone()));
        }
        if ui.do_reset || ui.do_live {
            source = Source::live(make_cfg(ui.est_kind));
            if ui.mission_on && ui.est_kind == 2 {
                source.command(Command::SetMission {
                    waypoints: square_mission(),
                    guidance: GuidanceConfig::default(),
                });
            }
            trail_pts.clear();
            if ui.do_reset {
                ui.roll = 0.0;
                ui.pitch = 0.0;
                ui.yaw = 0.0;
                ui.thrust = hover as f32;
                ui.recording = false;
            }
            // The fresh engine starts un-recording; re-sync if the checkbox is on
            // (e.g. "back to live" from replay while still recording).
            if ui.recording {
                source.command(Command::Record(true));
            }
        }
        if ui.do_replay {
            if let Ok(rec) = Recording::load(&record_path) {
                if !rec.is_empty() {
                    source = Source::Replay(ReplayState::new(rec));
                    trail_pts.clear();
                }
            }
        }
        if let Source::Replay(r) = &mut source {
            if ui.replay_toggle_play {
                r.playing = !r.playing;
            }
            if let Some(t) = ui.replay_seek {
                r.seek(t);
            }
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

fn controls_window(
    ui: &mut egui::Ui,
    snap: &Snapshot,
    hover: f64,
    st: &mut Ui,
    replay: Option<(f64, f64, f64)>,
) {
    egui::Window::new("Flight controls")
        .default_pos([12.0, 12.0])
        .default_width(300.0)
        .show(ui.ctx(), |ui| {
            if let Some((t, t0, t1)) = replay {
                // --- replay transport ---
                ui.label("REPLAY");
                ui.horizontal(|ui| {
                    if ui.button("play / pause").clicked() {
                        st.replay_toggle_play = true;
                    }
                    if ui.button("back to live").clicked() {
                        st.do_live = true;
                    }
                });
                if t1 > t0 {
                    let mut tt = t;
                    if ui
                        .add(egui::Slider::new(&mut tt, t0..=t1).text("time (s)"))
                        .changed()
                    {
                        st.replay_seek = Some(tt);
                    }
                }
                ui.separator();
            } else {
                // --- estimator selector (switching rebuilds the sim) ---
                ui.label("estimator");
                ui.horizontal(|ui| {
                    for (k, name) in [(0u8, "CF (M1)"), (1, "MEKF (M2)"), (2, "INS (M3)")] {
                        if ui.radio(st.est_kind == k, name).clicked() && st.est_kind != k {
                            st.est_kind = k;
                            st.do_reset = true;
                        }
                    }
                });
                if st.est_kind == 2 {
                    if ui
                        .checkbox(&mut st.mission_on, "fly square mission")
                        .changed()
                    {
                        st.do_reset = true;
                    }
                } else if st.mission_on {
                    ui.label("(mission needs the INS)");
                }
                ui.separator();

                let mission = st.est_kind == 2 && st.mission_on;
                ui.add_enabled_ui(!mission, |ui| {
                    ui.label("Attitude setpoint (deg)");
                    ui.add(egui::Slider::new(&mut st.roll, -35.0..=35.0).text("roll"));
                    ui.add(egui::Slider::new(&mut st.pitch, -35.0..=35.0).text("pitch"));
                    ui.add(egui::Slider::new(&mut st.yaw, -180.0..=180.0).text("yaw"));
                    ui.add(
                        egui::Slider::new(&mut st.thrust, 0.0..=(4.0 * hover as f32))
                            .text("thrust (N)"),
                    );
                    if ui.button("hover thrust").clicked() {
                        st.thrust = hover as f32;
                    }
                });
                if mission {
                    if let Some(idx) = snap.waypoint_index {
                        ui.monospace(format!("mission waypoint: {idx}"));
                    }
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.checkbox(&mut st.paused, "pause");
                    if ui.button("reset").clicked() {
                        st.do_reset = true;
                    }
                });
                ui.add(egui::Slider::new(&mut st.speed, 0.1..=8.0).text("sim speed ×"));
                ui.separator();
            }

            // --- record / replay ---
            ui.horizontal(|ui| {
                ui.checkbox(&mut st.recording, "record");
                if ui.button("save").clicked() {
                    st.do_save = true;
                }
                if replay.is_none() && ui.button("replay file").clicked() {
                    st.do_replay = true;
                }
            });
            ui.separator();

            // --- truth vs estimate readout (the autopilot only sees estimate) ---
            let (tr, tp, ty) = snap.truth.attitude.euler_angles();
            let (er, ep, ey) = snap.estimate.attitude.euler_angles();
            let att_err = snap
                .truth
                .attitude
                .angle_to(&snap.estimate.attitude)
                .to_degrees();
            ui.monospace(format!("t        {:7.2} s", snap.t));
            ui.monospace(format!("alt (-z) {:7.2} m", -snap.truth.position.z));
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
            ui.monospace(format!(
                "truth pos {:6.2} {:6.2} {:6.2}",
                snap.truth.position.x, snap.truth.position.y, -snap.truth.position.z
            ));
            ui.monospace(format!(
                "est   pos {:6.2} {:6.2} {:6.2}",
                snap.estimate.position.x, snap.estimate.position.y, -snap.estimate.position.z
            ));
            if snap.has_bias {
                let (tb, eb) = (snap.true_gyro_bias, snap.est_gyro_bias);
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

fn telemetry_window(ui: &mut egui::Ui, samples: &[TelemetrySample]) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};
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
            if samples.iter().any(|s| s.est_gyro_bias.norm() > 1e-12) {
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
