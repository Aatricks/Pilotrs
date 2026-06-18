//! # fsim-viz
//!
//! Interactive 3D viewer + live plots for the Pilotrs simulator. This is the
//! std-only leaf of the workspace: all the GPU/windowing/egui baggage lives
//! here and nothing in the flight-control core depends on it.
//!
//! The renderer is a pure *consumer*: the deterministic physics runs on its own
//! thread inside a `SimEngine`/`FwEngine` (M4), and each display frame the
//! viewer reads the latest published [`ViewSnapshot`](replay::ViewSnapshot) and
//! sends UI changes as commands — physics and rendering are fully decoupled. The
//! same viewer flies either airframe (quad or fixed-wing) and can play back a
//! recorded run (`Source::Replay`).
//!
//! ## Map + route planner
//!
//! The ground is a procedural elevation terrain ([`terrain`]); a top-down
//! shaded-relief minimap ([`minimap`]) lets you click a route and fly it —
//! dispatched as a quad `SetMission` (INS waypoint guidance) or a fixed-wing
//! `SetRoute` (vector-field line guidance) depending on the selected airframe.
//!
//! ## Frame convention
//!
//! The scene is drawn directly in the simulator's NED world frame (x = North,
//! y = East, z = Down). The camera's up vector is world `-z`, so altitude
//! (`-z`) points up on screen.

mod minimap;
mod replay;
mod terrain;

use fsim_sim::{
    Command, ControllerKind, FixedWingSetpoint, FwCommand, FwGuidanceConfig, FwSample, FwSimConfig,
    GuidanceConfig, Quat, Recording, Setpoint, SimConfig, TelemetrySample, Waypoint,
};
use minimap::{Minimap, MinimapActions, MinimapView, Route, TerrainLike};
use replay::{AircraftKind, ReplayState, Source, ViewSnapshot, ViewTelemetry};
use terrain::Terrain;
use three_d::egui;
use three_d::*;

/// Procedural-terrain seed (fixed so the map is the same every run).
const TERRAIN_SEED: u32 = 0x5EED_1234;

/// Mutable viewer state, edited by the controls window each frame.
struct Ui {
    // Airframe selection.
    fixed_wing: bool,
    do_airframe_switch: bool,
    // Quad-specific.
    est_kind: u8,
    controller_lqr: bool,
    mission_on: bool,
    roll: f32,
    pitch: f32,
    yaw: f32, // doubles as the fixed-wing manual course (deg)
    thrust: f32,
    // Fixed-wing manual cruise.
    fw_airspeed: f32,
    fw_altitude: f32,
    /// True while a drawn route is installed on the fixed-wing engine — gates
    /// off the per-frame manual `SetCruise` so it can't cancel the route
    /// (synchronous; avoids racing the lagging snapshot).
    fw_route_on: bool,
    // Shared transport.
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

/// Build the config for the selected estimator + inner controller (PID or LQR).
fn build_cfg(est_kind: u8, lqr: bool) -> SimConfig {
    let mut c = make_cfg(est_kind);
    if lqr {
        c.controller_kind = ControllerKind::Lqr;
    }
    c
}

/// A ~100 m square mission at 50 m altitude (scaled for the terrain map).
fn square_mission() -> Vec<Waypoint> {
    vec![
        Waypoint::ne_alt(0.0, 0.0, 50.0),
        Waypoint::ne_alt(100.0, 0.0, 50.0),
        Waypoint::ne_alt(100.0, 100.0, 50.0),
        Waypoint::ne_alt(0.0, 100.0, 50.0),
        Waypoint::ne_alt(0.0, 0.0, 50.0),
    ]
}

/// Quad waypoint-guidance tuning for the map scale (gentle cruise, a few-metre
/// acceptance radius).
fn quad_guidance() -> GuidanceConfig {
    GuidanceConfig {
        accept_radius: 5.0,
        cruise_speed: 6.0,
    }
}

/// Build the data source for the currently-selected airframe, dispatching the
/// default demo route where applicable. Single rebuild point for airframe /
/// estimator / controller switches.
fn make_source(ui: &Ui) -> Source {
    if ui.fixed_wing {
        // Fixed-wing flies on truth feedback; routes are drawn on the minimap.
        Source::live_fixedwing(FwSimConfig::aerosonde_cruise())
    } else {
        // Quad missions need the INS estimator (M3) — `build_cfg(2, _)` — or the
        // worker silently counts SetMission as rejected.
        let s = Source::live_quad(build_cfg(ui.est_kind, ui.controller_lqr));
        if ui.mission_on && ui.est_kind == 2 {
            s.quad_command(Command::SetMission {
                waypoints: square_mission(),
                guidance: quad_guidance(),
            });
        }
        s
    }
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

/// Adapter so the minimap can sample the real [`Terrain`] without depending on
/// its concrete type.
impl TerrainLike for Terrain {
    fn height(&self, north: f32, east: f32) -> f32 {
        Terrain::height(self, north, east)
    }
    fn normal(&self, north: f32, east: f32) -> Vec3 {
        Terrain::normal(self, north, east)
    }
    fn color(&self, north: f32, east: f32) -> (u8, u8, u8) {
        let c = Terrain::color(self, north, east);
        (c.r, c.g, c.b)
    }
}

/// A simple low-poly fixed-wing body (fuselage + wing + tail) in FRD body axes
/// (nose +x, right wing +y, down +z). Returns the parts and their fixed local
/// base transforms; each frame the live pose is applied as `pose * base`.
fn build_fw_body(context: &Context) -> (Vec<Gm<Mesh, PhysicalMaterial>>, Vec<Mat4>) {
    let mut parts = Vec::new();
    let mut bases = Vec::new();
    let add = |parts: &mut Vec<Gm<Mesh, PhysicalMaterial>>,
               bases: &mut Vec<Mat4>,
               mat: PhysicalMaterial,
               base: Mat4| {
        parts.push(Gm::new(Mesh::new(context, &CpuMesh::cube()), mat));
        bases.push(base);
    };
    // Fuselage: long along +x (≈16 m).
    add(
        &mut parts,
        &mut bases,
        opaque(context, 205, 205, 215),
        Mat4::from_nonuniform_scale(8.0, 1.0, 1.0),
    );
    // Main wing: wide along ±y (≈18 m span), thin in z, slightly aft of nose.
    add(
        &mut parts,
        &mut bases,
        opaque(context, 70, 130, 215),
        Mat4::from_translation(vec3(-1.0, 0.0, 0.0)) * Mat4::from_nonuniform_scale(1.6, 9.0, 0.25),
    );
    // Horizontal tail at the rear (-x).
    add(
        &mut parts,
        &mut bases,
        opaque(context, 70, 130, 215),
        Mat4::from_translation(vec3(-7.0, 0.0, 0.0)) * Mat4::from_nonuniform_scale(1.0, 3.5, 0.2),
    );
    // Vertical fin: "up" is body -z, so it extends toward -z.
    add(
        &mut parts,
        &mut bases,
        opaque(context, 240, 140, 40),
        Mat4::from_translation(vec3(-7.0, 0.0, -1.6)) * Mat4::from_nonuniform_scale(1.0, 0.25, 1.8),
    );
    (parts, bases)
}

fn main() {
    let window = Window::new(WindowSettings {
        title: "Pilotrs — quad / fixed-wing over terrain".to_string(),
        max_size: Some((1500, 950)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Camera in NED with up = world -z (altitude points up on screen). Framed
    // for the ~1 km terrain; the orbit target follows the aircraft each frame.
    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(170.0, 170.0, -150.0),
        vec3(0.0, 0.0, -40.0),
        vec3(0.0, 0.0, -1.0),
        degrees(45.0),
        0.5,
        6000.0,
    );
    let mut control = OrbitControl::new(vec3(0.0, 0.0, -40.0), 2.0, 2500.0);

    let ambient = AmbientLight::new(&context, 0.55, Srgba::WHITE);
    // Light travelling +z (downward in NED) so it lands on the upward (−z) faces.
    let directional = DirectionalLight::new(&context, 2.2, Srgba::WHITE, vec3(0.4, 0.3, 1.0));

    // Ground: the procedural elevation terrain (built once; lit per-vertex colours).
    let terrain = Terrain::new(TERRAIN_SEED);
    let ground = Gm::new(
        Mesh::new(&context, &terrain.build_mesh(200)),
        terrain.material(&context),
    );

    // Quad body: a flat box, scaled up for visibility against the 1 km map.
    let mut body = Gm::new(
        Mesh::new(&context, &CpuMesh::cube()),
        opaque(&context, 70, 130, 215),
    );

    // Four rotors (front = +x orange, rear = blue) — radius tracks motor thrust.
    let arm = 4.0 / std::f32::consts::SQRT_2;
    let rotor_local = [
        vec3(arm, arm, -0.9),   // M0 front-right
        vec3(-arm, arm, -0.9),  // M1 rear-right
        vec3(-arm, -arm, -0.9), // M2 rear-left
        vec3(arm, -arm, -0.9),  // M3 front-left
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

    // Fixed-wing body parts + their fixed local base transforms.
    let (mut fw_body, fw_base) = build_fw_body(&context);

    // Trajectory trail: instanced small spheres.
    let mut trail = Gm::new(
        InstancedMesh::new(&context, &Instances::default(), &CpuMesh::sphere(6)),
        opaque(&context, 90, 200, 210),
    );
    let mut trail_pts: Vec<Vec3> = Vec::new();

    // Minimap route planner (top-down) + the editable route + a downsampled
    // NED trail it paints.
    let mut minimap = Minimap::new(terrain.half_extent);
    let mut route = Route::default();
    let mut map_trail: Vec<(f32, f32)> = Vec::new();

    // --- simulation + UI state ---
    let hover = make_cfg(2).hover_thrust();
    let mut ui = Ui {
        fixed_wing: false,
        do_airframe_switch: false,
        est_kind: 2,           // default to the M3 INS stack
        controller_lqr: false, // default to the cascaded PID
        mission_on: true,      // INS only: fly the square mission
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        thrust: hover as f32,
        fw_airspeed: 25.0,
        fw_altitude: 120.0,
        fw_route_on: false,
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

    let mut source = make_source(&ui);
    let record_path = std::env::temp_dir().join("pilotrs_live.fsimrec");

    let mut gui = GUI::new(&context);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);

        let dt_frame = frame_input.elapsed_time * 0.001;
        source.tick(dt_frame); // advances replay; the live engines pace themselves

        let view = source.view();

        // Drive the live engine from the UI (no-op in replay mode). Physics runs
        // on its own thread; these are just commands.
        if !source.is_replay() {
            match view.kind {
                AircraftKind::Quad => {
                    source.quad_command(Command::Pause(ui.paused));
                    source.quad_command(Command::SetSpeed(ui.speed as f64));
                    let in_mission = ui.mission_on && ui.est_kind == 2;
                    if !in_mission {
                        source.quad_command(Command::SetSetpoint(Setpoint {
                            attitude: Quat::from_euler_angles(
                                ui.roll.to_radians() as f64,
                                ui.pitch.to_radians() as f64,
                                ui.yaw.to_radians() as f64,
                            ),
                            thrust: ui.thrust as f64,
                        }));
                    }
                }
                AircraftKind::FixedWing => {
                    source.fw_command(FwCommand::Pause(ui.paused));
                    source.fw_command(FwCommand::SetSpeed(ui.speed as f64));
                    // Manual cruise only when not flying a route. Gate on the
                    // *synchronous* flag, not the lagging snapshot's
                    // `waypoint_index`: dispatching SetRoute later this frame
                    // would otherwise race a stale SetCruise that cancels it.
                    if !ui.fw_route_on {
                        source.fw_command(FwCommand::SetCruise(FixedWingSetpoint {
                            airspeed: ui.fw_airspeed as f64,
                            altitude: ui.fw_altitude as f64,
                            course: (ui.yaw as f64).to_radians(),
                        }));
                    }
                }
            }
        }

        // --- update 3D transforms from the view ---
        let pos = to_v(&view.position);
        let pose = Mat4::from_translation(pos) * to_rot(&view.attitude);
        match view.kind {
            AircraftKind::Quad => {
                body.set_transformation(pose * Mat4::from_nonuniform_scale(3.5, 3.5, 0.7));
                for (i, rotor) in rotors.iter_mut().enumerate() {
                    let r = 1.0 + 1.2 * (view.motors[i] / 4.0) as f32; // size ∝ thrust
                    rotor.set_transformation(
                        pose * Mat4::from_translation(rotor_local[i]) * Mat4::from_scale(r),
                    );
                }
            }
            AircraftKind::FixedWing => {
                for (part, base) in fw_body.iter_mut().zip(&fw_base) {
                    part.set_transformation(pose * *base);
                }
            }
        }

        // --- trails (3D instanced + the flat minimap trail) ---
        let moved = trail_pts
            .last()
            .map(|p| (*p - pos).magnitude() > 1.0)
            .unwrap_or(true);
        if moved {
            trail_pts.push(pos);
            if trail_pts.len() > 1500 {
                trail_pts.remove(0);
            }
            map_trail.push((view.position.x as f32, view.position.y as f32));
            if map_trail.len() > 2000 {
                map_trail.remove(0);
            }
        }
        trail.set_instances(&Instances {
            transformations: trail_pts
                .iter()
                .map(|p| Mat4::from_translation(*p) * Mat4::from_scale(1.5))
                .collect(),
            ..Default::default()
        });

        // --- GUI (reads the view; edits UI state + the route for next frame) ---
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
        ui.do_airframe_switch = false;
        ui.replay_toggle_play = false;
        ui.replay_seek = None;
        let prev_recording = ui.recording;
        let is_fixed_wing = source.kind() == AircraftKind::FixedWing;
        let is_replay = source.is_replay();
        let telemetry = source.telemetry();
        let mut map_actions = MinimapActions::default();

        // GUI first so it can mark pointer events handled before the camera
        // control reads them (the event-consumption fix); `gui.update` returns
        // whether egui wants the pointer this frame.
        let egui_using = gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |egui_ui| {
                controls_window(egui_ui, &view, hover, &mut ui, replay_info);
                match &telemetry {
                    ViewTelemetry::Quad(s) => telemetry_window(egui_ui, s),
                    ViewTelemetry::FixedWing(s) => fw_telemetry_window(egui_ui, s),
                }
                let mview = MinimapView {
                    pos_north: view.position.x as f32,
                    pos_east: view.position.y as f32,
                    course: view.course() as f32,
                    active_wp: view.waypoint_index,
                    trail: &map_trail,
                };
                map_actions = minimap.show(
                    egui_ui.ctx(),
                    &terrain,
                    &mut route,
                    &mview,
                    is_fixed_wing,
                    is_replay,
                );
            },
        );

        // Follow-cam: re-aim the camera so its look-at target tracks the
        // aircraft as it ranges over the 1 km map, preserving the current
        // camera→target offset (so the user's orbit/zoom still apply). Setting
        // only `control.target` would move the orbit pivot but never the camera.
        // up = world -z (altitude up on screen), matching construction.
        let cam_offset = camera.position() - camera.target();
        camera.set_view(pos + cam_offset, pos, vec3(0.0, 0.0, -1.0));
        control.target = pos;
        // Camera control second, only when egui isn't claiming the pointer, so
        // dragging on the minimap never rotates the camera under it.
        if !egui_using {
            control.handle_events(&mut camera, &mut frame_input.events);
        }

        // --- apply UI actions ---
        if ui.recording != prev_recording {
            source.quad_command(Command::Record(ui.recording));
        }
        if ui.do_save {
            source.quad_command(Command::SaveRecording(record_path.clone()));
        }
        if ui.do_reset || ui.do_live || ui.do_airframe_switch {
            source = make_source(&ui);
            trail_pts.clear();
            map_trail.clear();
            ui.fw_route_on = false; // a fresh engine has no route installed
            if ui.do_reset {
                ui.roll = 0.0;
                ui.pitch = 0.0;
                ui.yaw = 0.0;
                ui.thrust = hover as f32;
                ui.recording = false;
            }
            // A fresh engine starts un-recording; re-sync if the checkbox is on.
            if ui.recording {
                source.quad_command(Command::Record(true));
            }
        }
        if ui.do_replay {
            if let Ok(rec) = Recording::load(&record_path) {
                if !rec.is_empty() {
                    source = Source::Replay(ReplayState::new(rec));
                    trail_pts.clear();
                    map_trail.clear();
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

        // --- route dispatch (after the egui pass, so no live egui borrow) ---
        if map_actions.fly && route.wps.len() >= 2 {
            let alt = route.alt_up as f64;
            let wps: Vec<Waypoint> = route
                .wps
                .iter()
                .map(|w| Waypoint::ne_alt(w.north as f64, w.east as f64, alt))
                .collect();
            if source.kind() == AircraftKind::FixedWing {
                let cfg = FwGuidanceConfig {
                    airspeed: route.cruise as f64,
                    ..FwGuidanceConfig::default()
                };
                source.fw_command(FwCommand::SetRoute {
                    waypoints: wps,
                    cfg,
                });
                ui.fw_route_on = true; // suppress manual cruise from now on
            } else {
                // Quad missions need the INS (M3); switch + rebuild if necessary.
                if ui.est_kind != 2 {
                    ui.est_kind = 2;
                    source = make_source(&ui);
                    trail_pts.clear();
                    map_trail.clear();
                }
                source.quad_command(Command::SetMission {
                    waypoints: wps,
                    guidance: quad_guidance(),
                });
                ui.mission_on = true;
            }
        }
        if map_actions.clear {
            match source.kind() {
                AircraftKind::Quad => {
                    source.quad_command(Command::SetAttitudeMode);
                    ui.mission_on = false;
                }
                AircraftKind::FixedWing => {
                    ui.fw_route_on = false; // back to manual cruise
                    source.fw_command(FwCommand::SetCruise(FixedWingSetpoint {
                        airspeed: ui.fw_airspeed as f64,
                        altitude: ui.fw_altitude as f64,
                        course: (ui.yaw as f64).to_radians(),
                    }));
                }
            }
        }

        // --- render scene then GUI on top ---
        let mut objects: Vec<&dyn Object> = vec![&ground, &trail];
        match view.kind {
            AircraftKind::Quad => {
                objects.push(&body);
                for r in &rotors {
                    objects.push(r);
                }
            }
            AircraftKind::FixedWing => {
                for p in &fw_body {
                    objects.push(p);
                }
            }
        }
        frame_input
            .screen()
            .clear(ClearState::color_and_depth(0.55, 0.70, 0.85, 1.0, 1.0))
            .render(&camera, objects, &[&ambient, &directional])
            .write(|| gui.render())
            .unwrap();

        FrameOutput::default()
    });
}

fn controls_window(
    ui: &mut egui::Ui,
    view: &ViewSnapshot,
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
                // --- airframe selector (switching rebuilds the source) ---
                ui.label("airframe");
                ui.horizontal(|ui| {
                    if ui.radio(!st.fixed_wing, "Quad").clicked() && st.fixed_wing {
                        st.fixed_wing = false;
                        st.do_airframe_switch = true;
                    }
                    if ui.radio(st.fixed_wing, "Fixed-wing").clicked() && !st.fixed_wing {
                        st.fixed_wing = true;
                        st.do_airframe_switch = true;
                    }
                });
                ui.separator();

                if !st.fixed_wing {
                    quad_controls(ui, view, hover, st);
                } else {
                    ui.label("Fixed-wing cruise (or draw a route on the minimap)");
                    ui.add(egui::Slider::new(&mut st.fw_airspeed, 12.0..=35.0).text("airspeed"));
                    ui.add(
                        egui::Slider::new(&mut st.fw_altitude, 20.0..=200.0).text("altitude (m)"),
                    );
                    ui.add(egui::Slider::new(&mut st.yaw, -180.0..=180.0).text("course (deg)"));
                    if let Some(idx) = view.waypoint_index {
                        ui.monospace(format!("flying route — leg {idx}"));
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

            // --- record / replay (quad only) ---
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

            // --- state readout (airframe-agnostic) ---
            let (r, p, y) = view.attitude.euler_angles();
            ui.monospace(format!("t        {:7.2} s", view.t));
            ui.monospace(format!("alt (-z) {:7.2} m", -view.position.z));
            ui.monospace(format!(
                "pos N/E  {:8.1} {:8.1} m",
                view.position.x, view.position.y
            ));
            ui.monospace(format!(
                "att RPY  {:6.1} {:6.1} {:6.1}",
                r.to_degrees(),
                p.to_degrees(),
                y.to_degrees()
            ));
            ui.monospace(format!("course   {:6.1} deg", view.course().to_degrees()));
            if let Some(s) = view.surfaces {
                ui.monospace(format!(
                    "surf a/e/r {:+.2} {:+.2} {:+.2}  thr {:.2}",
                    s.aileron, s.elevator, s.rudder, s.throttle
                ));
            }
        });
}

/// The quad-only selectors + attitude sliders (estimator, controller, mission).
fn quad_controls(ui: &mut egui::Ui, _view: &ViewSnapshot, hover: f64, st: &mut Ui) {
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
    // Inner attitude controller: PID (M1) vs LQR (M5).
    ui.horizontal(|ui| {
        ui.label("controller");
        if ui.radio(!st.controller_lqr, "PID").clicked() && st.controller_lqr {
            st.controller_lqr = false;
            st.do_reset = true;
        }
        if ui.radio(st.controller_lqr, "LQR").clicked() && !st.controller_lqr {
            st.controller_lqr = true;
            st.do_reset = true;
        }
    });
    ui.separator();

    let mission = st.est_kind == 2 && st.mission_on;
    ui.add_enabled_ui(!mission, |ui| {
        ui.label("Attitude setpoint (deg)");
        ui.add(egui::Slider::new(&mut st.roll, -35.0..=35.0).text("roll"));
        ui.add(egui::Slider::new(&mut st.pitch, -35.0..=35.0).text("pitch"));
        ui.add(egui::Slider::new(&mut st.yaw, -180.0..=180.0).text("yaw"));
        ui.add(egui::Slider::new(&mut st.thrust, 0.0..=(4.0 * hover as f32)).text("thrust (N)"));
        if ui.button("hover thrust").clicked() {
            st.thrust = hover as f32;
        }
    });
}

fn telemetry_window(ui: &mut egui::Ui, samples: &[TelemetrySample]) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};
    // Build [t, deg] series for an euler component selected by `axis` (0/1/2)
    // from a quaternion extracted by `pick`.
    let series =
        |pick: &dyn Fn(&TelemetrySample) -> (f64, f64, f64), axis: usize| -> Vec<[f64; 2]> {
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
                let pos = |sel: &dyn Fn(&TelemetrySample) -> f64| -> Vec<[f64; 2]> {
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

            // Gyro-bias estimate vs the hidden truth (the MEKF's M2 win).
            if samples.iter().any(|s| s.est_gyro_bias.norm() > 1e-12) {
                ui.label("gyro bias est vs true (rad/s)");
                let axis_pts = |sel: &dyn Fn(&TelemetrySample) -> f64| -> Vec<[f64; 2]> {
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

/// Fixed-wing telemetry: airspeed / altitude / course, commanded vs actual.
fn fw_telemetry_window(ui: &mut egui::Ui, samples: &[FwSample]) {
    use egui_plot::{Legend, Line, Plot, PlotPoints};
    let truth_c = egui::Color32::from_rgb(90, 200, 210);
    let cmd_c = egui::Color32::GRAY;
    egui::Window::new("Fixed-wing telemetry")
        .default_pos([12.0, 360.0])
        .default_width(440.0)
        .show(ui.ctx(), |ui| {
            ui.label("airspeed (m/s)");
            Plot::new("fw_va")
                .height(120.0)
                .legend(Legend::default())
                .show(ui, |p| {
                    p.line(
                        Line::new(
                            "Va",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| [s.t, s.truth.velocity.norm()])
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(truth_c),
                    );
                    p.line(
                        Line::new(
                            "cmd",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| [s.t, s.setpoint.airspeed])
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(cmd_c),
                    );
                });
            ui.label("altitude (m)");
            Plot::new("fw_alt")
                .height(120.0)
                .legend(Legend::default())
                .show(ui, |p| {
                    p.line(
                        Line::new(
                            "alt",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| [s.t, -s.truth.position.z])
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(truth_c),
                    );
                    p.line(
                        Line::new(
                            "cmd",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| [s.t, s.setpoint.altitude])
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(cmd_c),
                    );
                });
            ui.label("course χ (deg)");
            Plot::new("fw_chi")
                .height(120.0)
                .legend(Legend::default())
                .show(ui, |p| {
                    p.line(
                        Line::new(
                            "χ",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| {
                                        [
                                            s.t,
                                            s.truth
                                                .velocity
                                                .y
                                                .atan2(s.truth.velocity.x)
                                                .to_degrees(),
                                        ]
                                    })
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(truth_c),
                    );
                    p.line(
                        Line::new(
                            "cmd",
                            PlotPoints::from(
                                samples
                                    .iter()
                                    .map(|s| [s.t, s.setpoint.course.to_degrees()])
                                    .collect::<Vec<_>>(),
                            ),
                        )
                        .color(cmd_c),
                    );
                });
        });
}
