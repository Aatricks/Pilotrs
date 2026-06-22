//! # fsim-viz
//!
//! Interactive 3D viewer + live plots for the Pilotrs simulator. This is the
//! std-only leaf of the workspace: all the GPU/windowing/egui baggage lives
//! here and nothing in the flight-control core depends on it.
//!
//! The renderer is a pure *consumer*: the deterministic physics runs on its own
//! thread inside a `SimEngine`/`FwEngine`, and each display frame the
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

mod input;
mod minimap;
mod replay;
mod terrain;

use fsim_sim::planet;
use fsim_sim::{
    Command, ControllerKind, FixedWingSetpoint, FwCommand, FwFaults, FwGuidanceConfig, FwSample,
    FwSimConfig, GuidanceConfig, QuadFaults, Quat, Recording, SensorFault, SensorFaults, Setpoint,
    SimConfig, SurfaceFault, TelemetrySample, Waypoint,
};
use input::StickSource;
use minimap::{Minimap, MinimapActions, MinimapView, Route, TerrainLike};
use replay::{AircraftKind, ReplayState, Source, ViewSnapshot, ViewTelemetry};
use terrain::Terrain;
use three_d::egui;
use three_d::*;

/// Procedural-terrain seed (fixed so the map is the same every run).
const TERRAIN_SEED: u32 = 0x5EED_1234;

/// Latitude bands of the globe mesh (× 2 longitudes). 400 → ~50 m cells near
/// home, ~640 k triangles — built once.
const GLOBE_BANDS: usize = 400;

/// Clearance \[m\] below which the aircraft is considered to have hit the terrain.
const CRASH_MARGIN: f32 = 3.0;

/// Mutable viewer state, edited by the controls window each frame.
struct Ui {
    // Airframe selection.
    fixed_wing: bool,
    /// The fixed-wing is the relaxed-stability fighter under manual (pilot)
    /// control, not the autopilot UAV. Implies `fixed_wing`.
    fighter: bool,
    /// Pilot's intent for the fly-by-wire toggle (authoritative; the snapshot's
    /// `fbw_on` just confirms it). Flipped by the F key / gamepad button.
    fbw_on: bool,
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
    // Weather (fixed-wing): steady wind speed + turbulence RMS [m/s].
    wind_speed: f32,
    turbulence: f32,
    // Injected faults (sent to the live engine each frame).
    fw_faults: FwFaults,
    quad_dead_rotor: Option<usize>,
    quad_sensors: SensorFaults,
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
    drop_storm: bool,
    clear_storm: bool,
    replay_toggle_play: bool,
    replay_seek: Option<f64>,
}

/// Estimator selection: 0 = complementary filter, 1 = MEKF,
/// 2 = INS. Switching rebuilds the sim.
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

/// A ~100 m square mission at 60 m altitude, flown inside the terrain's flat
/// home clearing (radius `home_inner` ≈ 170 m, so the ±100 m corners stay on
/// the level field). The clearing floor sits at `home_level` ≈ −12 m, so a
/// 60 m mission clears it by ~72 m and a quad spawned at altitude 0 clears it
/// by ~12 m.
fn square_mission() -> Vec<Waypoint> {
    vec![
        Waypoint::ne_alt(0.0, 0.0, 60.0),
        Waypoint::ne_alt(100.0, 0.0, 60.0),
        Waypoint::ne_alt(100.0, 100.0, 60.0),
        Waypoint::ne_alt(0.0, 100.0, 60.0),
        Waypoint::ne_alt(0.0, 0.0, 60.0),
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
        // Fixed-wing flies on truth feedback over the spherical planet;
        // routes are drawn on the minimap. Spawn above the mountain peaks
        // (≈320 m) so level cruise and routes fly over the terrain, not through
        // it (the sim has no terrain collision).
        let cfg = if ui.fighter {
            // The relaxed-stability fighter under manual fly-by-wire control.
            FwSimConfig::fighter_manual(ui.fw_altitude as f64)
        } else {
            FwSimConfig::aerosonde_at(ui.fw_altitude as f64)
        };
        Source::live_fixedwing(cfg)
    } else {
        // Quad missions need the INS estimator — `build_cfg(2, _)` — or the
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

/// The aircraft's **PCI** render position and body→PCI rotation for a snapshot.
/// The fixed-wing truth is already planet-centered; the quad runs in a flat
/// local-NED frame anchored at the home surface point, so we lift it onto the
/// globe (`pci = anchor + q_pci_from_ned · local`).
fn render_pose(view: &ViewSnapshot) -> (Vec3, Mat4) {
    match view.kind {
        AircraftKind::FixedWing => (to_v(&view.position), to_rot(&view.attitude)),
        AircraftKind::Quad => {
            let anchor = planet::home_pci(0.0);
            let q = planet::pci_from_ned(anchor);
            (
                to_v(&(anchor + q * view.position)),
                to_rot(&(q * view.attitude)),
            )
        }
    }
}

/// The aircraft's geographic `(lat, lon)` \[rad\] and local course for the
/// planisphere minimap. The quad flies in a flat local-NED patch at home
/// (lat≈N/R, lon≈E/R near the prime-meridian equator); the fixed-wing is PCI.
fn map_pose(view: &ViewSnapshot) -> (f32, f32, f32) {
    let course = view.course() as f32;
    match view.kind {
        AircraftKind::Quad => {
            let r = planet::PLANET_RADIUS;
            (
                (view.position.x / r) as f32,
                (view.position.y / r) as f32,
                course,
            )
        }
        AircraftKind::FixedWing => {
            let (lat, lon, _) = planet::pci_to_geodetic(view.position);
            (lat as f32, lon as f32, course)
        }
    }
}

/// Geographic latitude / longitude (rad) → unit direction on the planet
/// (PCI: `+z` = North pole, lon 0 along `+x`).
fn geo_dir3(lat: f32, lon: f32) -> Vec3 {
    let cl = lat.cos();
    vec3(cl * lon.cos(), cl * lon.sin(), lat.sin())
}

/// A globe follow-camera in the aircraft's **local tangent frame** (up = radial),
/// so it never gimbal-flips as the fixed-wing flies around the planet. Drag
/// orbits (azimuth/elevation); scroll zooms.
struct OrbitCam {
    az: f32,
    el: f32,
    dist: f32,
}

impl OrbitCam {
    /// Local (north, east) tangent basis at the outward radial `up`.
    fn tangent(up: Vec3) -> (Vec3, Vec3) {
        let axis = vec3(0.0, 0.0, 1.0);
        let mut north = axis - up * axis.dot(up);
        if north.magnitude() < 1e-4 {
            let pm = vec3(1.0, 0.0, 0.0);
            north = pm - up * pm.dot(up);
        }
        let north = north.normalize();
        (north, up.cross(north))
    }

    /// Re-aim the camera to chase the aircraft at `target` (PCI), with up = local
    /// radial. The camera sits at azimuth/elevation/distance in the local frame.
    fn aim(&self, camera: &mut Camera, target: Vec3) {
        let up = target.normalize();
        let (north, east) = Self::tangent(up);
        let (ce, se) = (self.el.cos(), self.el.sin());
        let dir = north * (ce * self.az.cos()) + east * (ce * self.az.sin()) + up * se;
        camera.set_view(target + dir * self.dist, target, up);
    }

    /// Apply drag (orbit) and wheel (zoom) from the frame's events.
    fn handle(&mut self, events: &[Event]) {
        for ev in events {
            match ev {
                Event::MouseMotion {
                    button: Some(MouseButton::Left),
                    delta,
                    handled: false,
                    ..
                } => {
                    self.az -= delta.0 * 0.006;
                    self.el = (self.el + delta.1 * 0.006).clamp(-1.3, 1.45);
                }
                Event::MouseWheel {
                    delta,
                    handled: false,
                    ..
                } => {
                    self.dist = (self.dist * (1.0 - delta.1 * 0.0015)).clamp(40.0, 14000.0);
                }
                _ => {}
            }
        }
    }
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

/// An opaque physically-based material with explicit roughness/metallic — for
/// the satin-metal aircraft skin and the glossy glass canopy.
fn mat_pbr(
    context: &Context,
    r: u8,
    g: u8,
    b: u8,
    roughness: f32,
    metallic: f32,
) -> PhysicalMaterial {
    PhysicalMaterial::new_opaque(
        context,
        &CpuMaterial {
            albedo: Srgba { r, g, b, a: 255 },
            roughness,
            metallic,
            ..Default::default()
        },
    )
}

/// Adapter so the minimap can sample the real [`Terrain`] without depending on
/// its concrete type. The minimap is a *local tangent map* at home, so a North/
/// East offset maps to a direction on the globe ([`ne_to_dir`]) and we sample the
/// spherical terrain there.
impl TerrainLike for Terrain {
    /// Shaded colour at a geographic point for the planisphere: the Earth-like
    /// `color_dir` tinted by a hillshade from a fixed NW-and-above light in the
    /// local tangent frame (so relief reads consistently across the whole map).
    fn map_sample(&self, lat: f32, lon: f32) -> ((u8, u8, u8), f32) {
        let dir = geo_dir3(lat, lon);
        let c = self.color_dir(dir);
        let (north, east) = OrbitCam::tangent(dir);
        let light = (north * 0.45 - east * 0.35 + dir * 0.82).normalize();
        let shade = (0.45 + 0.6 * self.normal_dir(dir).dot(light).clamp(0.0, 1.0)).clamp(0.0, 1.0);
        ((c.r, c.g, c.b), shade)
    }
}

/// A low-poly but jet-like fixed-wing model in FRD body axes (nose +x, right
/// wing +y, down +z): a rounded fuselage + tapered nose cone, a dark glass
/// canopy, swept wings and stabilisers, and a swept fin. Returns the parts and
/// their fixed local base transforms; each frame the live pose is `pose * base`.
fn build_fw_body(context: &Context) -> (Vec<Gm<Mesh, PhysicalMaterial>>, Vec<Mat4>) {
    let mut parts = Vec::new();
    let mut bases = Vec::new();
    let mut add = |mesh: CpuMesh, mat: PhysicalMaterial, base: Mat4| {
        parts.push(Gm::new(Mesh::new(context, &mesh), mat));
        bases.push(base);
    };
    // Satin-metal airframe grey; swept surfaces.
    let skin = |r, g, b| mat_pbr(context, r, g, b, 0.45, 0.3);
    let sweep = degrees(17.0);
    let fin_sweep = degrees(30.0);

    // Fuselage: a rounded body along +x (≈13 m).
    add(
        CpuMesh::cylinder(20),
        skin(150, 156, 168),
        Mat4::from_translation(vec3(-6.5, 0.0, 0.0)) * Mat4::from_nonuniform_scale(13.0, 0.8, 0.92),
    );
    // Tapered nose cone off the front.
    add(
        CpuMesh::cone(20),
        skin(150, 156, 168),
        Mat4::from_translation(vec3(6.5, 0.0, 0.0)) * Mat4::from_nonuniform_scale(3.3, 0.8, 0.92),
    );
    // Dark exhaust nozzle at the tail.
    add(
        CpuMesh::cylinder(16),
        mat_pbr(context, 58, 60, 68, 0.5, 0.4),
        Mat4::from_translation(vec3(-7.6, 0.0, 0.0)) * Mat4::from_nonuniform_scale(1.2, 0.62, 0.72),
    );
    // Canopy: a dark glass bubble on top-front (up is body -z).
    add(
        CpuMesh::sphere(16),
        mat_pbr(context, 46, 58, 80, 0.15, 0.2),
        Mat4::from_translation(vec3(2.0, 0.0, -0.62))
            * Mat4::from_nonuniform_scale(2.4, 0.74, 0.64),
    );
    // Swept main wings (right, then left), thin in z, mid-mounted. Each is built
    // root-at-origin, swept about z, then attached to the fuselage.
    add(
        CpuMesh::cube(),
        skin(126, 132, 144),
        Mat4::from_translation(vec3(-0.5, 0.7, 0.12))
            * Mat4::from_angle_z(sweep)
            * Mat4::from_translation(vec3(0.0, 3.0, 0.0))
            * Mat4::from_nonuniform_scale(2.6, 6.0, 0.16),
    );
    add(
        CpuMesh::cube(),
        skin(126, 132, 144),
        Mat4::from_translation(vec3(-0.5, -0.7, 0.12))
            * Mat4::from_angle_z(-sweep)
            * Mat4::from_translation(vec3(0.0, -3.0, 0.0))
            * Mat4::from_nonuniform_scale(2.6, 6.0, 0.16),
    );
    // Swept horizontal stabilisers at the tail.
    add(
        CpuMesh::cube(),
        skin(126, 132, 144),
        Mat4::from_translation(vec3(-5.6, 0.5, 0.0))
            * Mat4::from_angle_z(sweep)
            * Mat4::from_translation(vec3(0.0, 1.5, 0.0))
            * Mat4::from_nonuniform_scale(1.7, 3.0, 0.14),
    );
    add(
        CpuMesh::cube(),
        skin(126, 132, 144),
        Mat4::from_translation(vec3(-5.6, -0.5, 0.0))
            * Mat4::from_angle_z(-sweep)
            * Mat4::from_translation(vec3(0.0, -1.5, 0.0))
            * Mat4::from_nonuniform_scale(1.7, 3.0, 0.14),
    );
    // Swept vertical fin (up is body -z), in a steel-blue accent.
    add(
        CpuMesh::cube(),
        skin(96, 116, 150),
        Mat4::from_translation(vec3(-5.9, 0.0, -0.35))
            * Mat4::from_angle_y(fin_sweep)
            * Mat4::from_translation(vec3(0.0, 0.0, -1.4))
            * Mat4::from_nonuniform_scale(2.4, 0.18, 2.8),
    );
    (parts, bases)
}

/// A scattered field of fluffy cumulus spread across the **whole globe** (so
/// zooming out shows a cloud-flecked planet, not a cap over home), at a few
/// altitude decks. Each cloud is a *cluster* of overlapping spheres (a lumpy
/// billow, not a flat disc), laid flat to the local horizon via `pci_from_ned`.
/// Opaque white with an emissive lift so the undersides stay bright rather than
/// going muddy grey under the sun. Built once; tuned from screenshots.
fn build_clouds(context: &Context) -> Gm<InstancedMesh, PhysicalMaterial> {
    let r = planet::PLANET_RADIUS;
    // Lobe offsets (local NED, in puff-radius units) + relative size — overlapping
    // blobs that read as one billowing cloud rather than a single ball.
    let lobes = [
        (0.0, 0.0, 0.0, 1.0),
        (0.9, 0.3, 0.10, 0.72),
        (-0.8, 0.4, 0.05, 0.66),
        (0.3, -0.9, 0.12, 0.62),
        (-0.4, -0.7, 0.0, 0.58),
        (0.6, 0.9, 0.15, 0.50),
    ];
    let mut xforms = Vec::new();
    // Distribute cluster centres evenly over the sphere (a Fibonacci spiral); an
    // integer hash opens ~40% clear sky and jitters each cloud's size/altitude.
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let count = 440usize;
    for k in 0..count {
        let mut h = (k as u32).wrapping_mul(2_654_435_761);
        h ^= h >> 15;
        h = h.wrapping_mul(2_246_822_519);
        h ^= h >> 13;
        if h % 100 < 40 {
            continue; // clear patches of sky
        }
        // Fibonacci-sphere unit direction.
        let yv = 1.0 - 2.0 * (k as f64 + 0.5) / count as f64;
        let ring = (1.0 - yv * yv).max(0.0).sqrt();
        let theta = golden * k as f64;
        let dir = fsim_sim::Vec3::new(ring * theta.cos(), yv, ring * theta.sin());
        let alt = 780.0 + (h % 7) as f64 * 55.0; // a few decks, ~780–1110 m
        let p = dir * (r + alt);
        let q = planet::pci_from_ned(p);
        let rot = to_rot(&q);
        let base = 60.0 + (h % 5) as f64 * 12.0; // puff radius [m]
        for (dx, dy, dz, sc) in lobes {
            let off = q * fsim_sim::Vec3::new(dx * base, dy * base, dz * base);
            let s = (base * sc) as f32;
            xforms.push(
                Mat4::from_translation(to_v(&(p + off)))
                    * rot
                    * Mat4::from_nonuniform_scale(s, s, s * 0.78),
            );
        }
    }
    Gm::new(
        InstancedMesh::new(
            context,
            &Instances {
                transformations: xforms,
                ..Default::default()
            },
            &CpuMesh::sphere(10),
        ),
        PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba {
                    r: 250,
                    g: 252,
                    b: 255,
                    a: 255,
                },
                // Lifts the shadowed undersides toward white (cumulus, not slate).
                emissive: Srgba {
                    r: 72,
                    g: 74,
                    b: 82,
                    a: 255,
                },
                roughness: 1.0,
                metallic: 0.0,
                ..Default::default()
            },
        ),
    )
}

fn main() {
    let window = Window::new(WindowSettings {
        title: "Pilotrs — quad / fixed-wing over terrain".to_string(),
        max_size: Some((1500, 950)),
        ..Default::default()
    })
    .unwrap();
    let context = window.gl();

    // Globe follow-camera: up = the local radial at the aircraft, so it never
    // gimbal-flips as the fixed-wing flies around the planet (see OrbitCam). The
    // far plane clears the whole ~12.7 km planet so the curved horizon shows.
    let r_planet = planet::PLANET_RADIUS as f32;
    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(r_planet + 1200.0, 0.0, 700.0),
        vec3(r_planet, 0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        degrees(45.0),
        // near 10 m (the orbit min distance is 40 m); far 20 km clears the whole
        // planet (the zoom-out is bounded to 6 km so the camera core-distance stays
        // under ~13 km and the back of the globe — ~R beyond it — is never clipped).
        10.0,
        30000.0,
    );
    let mut orbit = OrbitCam {
        az: std::f32::consts::PI, // behind the aircraft, looking North
        el: 0.5,                  // ~29° above the local horizon
        dist: 1400.0,
    };

    let ambient = AmbientLight::new(&context, 0.4, Srgba::WHITE);
    // Sun: lights the home hemisphere (+x) from the upper side. On the globe a
    // surface's outward normal is its radial; this direction (L = −dir) has a
    // positive component along the home radial so the airfield is lit.
    let directional = DirectionalLight::new(&context, 2.8, Srgba::WHITE, vec3(-0.7, -0.35, -0.55));

    // The planet: the procedural terrain baked onto a globe (built once; lit
    // per-vertex colours), centred at the PCI origin.
    let terrain = Terrain::new(TERRAIN_SEED);
    let ground = Gm::new(
        Mesh::new(&context, &terrain.sphere_mesh(GLOBE_BANDS)),
        terrain.material(&context),
    );

    // Sea: a smooth sphere just below sea level. Land pokes through where the
    // terrain rises above it; deep valleys sit below it and read as water/ocean.
    // The −1 m offset keeps it from coinciding exactly with the terrain surface
    // at the shoreline (height == sea_level), which would z-fight.
    let sea_radius = r_planet + terrain.sea_level - 1.0;
    let mut sea = Gm::new(
        Mesh::new(&context, &CpuMesh::sphere(48)),
        opaque(&context, 30, 78, 130),
    );
    sea.set_transformation(Mat4::from_scale(sea_radius));

    // Quad body: a flat box, scaled up for visibility against the 4.8 km map.
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

    // Cloud layer: scattered translucent cumulus over the home region at a couple
    // of altitude decks (a first visual cut — refine from screenshots). Each puff
    // is a squashed sphere oriented flat to the local horizon via `pci_from_ned`.
    let clouds = build_clouds(&context);

    // Minimap route planner (top-down) + the editable route + a downsampled
    // NED trail it paints.
    let mut minimap = Minimap::new(terrain.half_extent);
    let mut route = Route::default();
    let mut map_trail: Vec<(f32, f32)> = Vec::new();

    // --- simulation + UI state ---
    let hover = make_cfg(2).hover_thrust();
    let mut ui = Ui {
        fixed_wing: false,
        fighter: false,
        fbw_on: true,
        do_airframe_switch: false,
        est_kind: 2,           // default to the INS stack
        controller_lqr: false, // default to the cascaded PID
        mission_on: true,      // INS only: fly the square mission
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        thrust: hover as f32,
        fw_airspeed: 25.0,
        fw_altitude: 400.0, // above the ~320 m mountain peaks
        wind_speed: 0.0,
        turbulence: 0.0,
        fw_faults: FwFaults::none(),
        quad_dead_rotor: None,
        quad_sensors: SensorFaults::none(),
        fw_route_on: false,
        paused: false,
        speed: 1.0,
        recording: false,
        do_reset: false,
        do_save: false,
        do_replay: false,
        do_live: false,
        drop_storm: false,
        clear_storm: false,
        replay_toggle_play: false,
        replay_seek: None,
    };

    let mut source = make_source(&ui);
    let record_path = std::env::temp_dir().join("pilotrs_live.fsimrec");

    // The fighter's level-cruise trim throttle (from the solver, not a guess), so
    // the manual throttle holds speed before the pilot touches it.
    let fighter_trim_throttle = FwSimConfig::fighter().fbw.throttle_trim as f32;

    // Pilot input (keyboard + gamepad) for the manual fighter, seeded at trim.
    let mut stick_src = StickSource::new(fighter_trim_throttle);

    let mut gui = GUI::new(&context);

    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);

        let dt_frame = frame_input.elapsed_time * 0.001;
        source.tick(dt_frame); // advances replay; the live engines pace themselves

        let view = source.view();

        // Read the pilot's stick + toggle/reset edges for this frame (used by the
        // manual fighter; harmless otherwise). Events are read before egui's pass.
        stick_src.handle_window_events(&frame_input.events);
        let pilot = stick_src.poll(dt_frame as f32);

        // Drive the live engine from the UI (no-op in replay mode). Physics runs
        // on its own thread; these are just commands.
        if !source.is_replay() {
            match view.kind {
                AircraftKind::Quad => {
                    source.quad_command(Command::Pause(ui.paused));
                    source.quad_command(Command::SetSpeed(ui.speed as f64));
                    // Weather: a crosswind toward the east + turbulence.
                    source.quad_command(Command::SetWind(fsim_sim::Vec3::new(
                        0.0,
                        ui.wind_speed as f64,
                        0.0,
                    )));
                    source.quad_command(Command::SetTurbulence(ui.turbulence as f64));
                    source.quad_command(Command::SetFaults(QuadFaults {
                        dead_rotor: ui.quad_dead_rotor,
                        sensors: ui.quad_sensors,
                    }));
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
                    // Weather: a crosswind toward the east + turbulence.
                    source.fw_command(FwCommand::SetWind(fsim_sim::Vec3::new(
                        0.0,
                        ui.wind_speed as f64,
                        0.0,
                    )));
                    source.fw_command(FwCommand::SetTurbulence(ui.turbulence as f64));
                    source.fw_command(FwCommand::SetFaults(ui.fw_faults));
                    if ui.fighter {
                        // Pilot in the loop: drive the fly-by-wire from the stick.
                        // The F-key/gamepad edge flips our authoritative intent.
                        if pilot.toggle_fbw {
                            ui.fbw_on = !ui.fbw_on;
                        }
                        source.fw_command(FwCommand::SetFbw(ui.fbw_on));
                        source.fw_command(FwCommand::SetStick(pilot.stick));
                    } else if !ui.fw_route_on {
                        // Autopilot cruise only when not flying a route. Gate on
                        // the *synchronous* flag, not the lagging snapshot's
                        // `waypoint_index`: dispatching SetRoute later this frame
                        // would otherwise race a stale SetCruise that cancels it.
                        source.fw_command(FwCommand::SetCruise(FixedWingSetpoint {
                            airspeed: ui.fw_airspeed as f64,
                            altitude: ui.fw_altitude as f64,
                            course: (ui.yaw as f64).to_radians(),
                        }));
                    }
                }
            }
        }

        // --- update 3D transforms from the view (PCI render pose) ---
        let (pos, rot) = render_pose(&view);
        let pose = Mat4::from_translation(pos) * rot;

        // Terrain collision: the sim has no terrain, so the viewer (which does)
        // flags impact. If the aircraft is at or below the ground at its position,
        // tell the engine to crash. `height_dir` returns the elevation above the
        // datum, `view.altitude()` the aircraft's; cross them with a small margin.
        if !source.is_replay() && !view.crashed {
            let ground = terrain.height_dir(pos);
            if (view.altitude() as f32) <= ground + CRASH_MARGIN {
                match view.kind {
                    AircraftKind::FixedWing => source.fw_command(FwCommand::Crash),
                    AircraftKind::Quad => source.quad_command(Command::Crash),
                }
            }
        }
        // Aircraft geographic pose (lat, lon, course) for the planisphere.
        let (map_lat, map_lon, map_course) = map_pose(&view);
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
            map_trail.push((map_lat, map_lon));
            if map_trail.len() > 2000 {
                map_trail.remove(0);
            }
        }
        // A slim contrail that fades out along its older tail (oldest third
        // tapers to nothing), rather than a fat uniform tube.
        let n_trail = trail_pts.len().max(1) as f32;
        trail.set_instances(&Instances {
            transformations: trail_pts
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let frac = i as f32 / n_trail; // 0 = oldest tail, 1 = nearest
                    let s = 0.65 * (frac * 3.0).min(1.0);
                    Mat4::from_translation(*p) * Mat4::from_scale(s.max(0.08))
                })
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
        ui.drop_storm = false;
        ui.clear_storm = false;
        ui.replay_toggle_play = false;
        ui.replay_seek = None;
        // Fighter reset-to-trim (R key / gamepad Start) re-spawns at trim. Set it
        // here, after the one-shot clears above, so the rebuild block acts on it.
        if ui.fighter && pilot.reset {
            ui.do_reset = true;
        }
        let prev_recording = ui.recording;
        let is_fixed_wing = source.kind() == AircraftKind::FixedWing;
        let is_replay = source.is_replay();
        let telemetry = source.telemetry();
        let has_gamepad = stick_src.has_gamepad();
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
                if view.manual {
                    hud_overlay(egui_ui.ctx(), &view, has_gamepad);
                }
                match &telemetry {
                    ViewTelemetry::Quad(s) => telemetry_window(egui_ui, s),
                    ViewTelemetry::FixedWing(s) => fw_telemetry_window(egui_ui, s),
                }
                let mview = MinimapView {
                    lat: map_lat,
                    lon: map_lon,
                    course: map_course,
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

        // Globe follow-cam: orbit the aircraft in its local tangent frame (up =
        // radial) so the camera never gimbal-flips as the fixed-wing circles the
        // planet. Drag orbits / scroll zooms — but only when egui isn't using the
        // pointer, so dragging on the minimap never moves the camera.
        if !egui_using {
            orbit.handle(&frame_input.events);
        }
        orbit.aim(&mut camera, pos);

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
            if ui.fighter {
                // Fresh fighter spawns at trim with the FCS engaged; recenter the
                // stick to a trim-throttle hold.
                ui.fbw_on = true;
                stick_src.recenter(fighter_trim_throttle);
            }
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

        // --- storm dispatch: drop a microburst at the aircraft's map position ---
        if ui.drop_storm {
            let r = planet::PLANET_RADIUS;
            let storm = Some(fsim_sim::StormCell::microburst((
                map_lat as f64 * r,
                map_lon as f64 * r,
            )));
            match source.kind() {
                AircraftKind::FixedWing => source.fw_command(FwCommand::SetStorm(storm)),
                AircraftKind::Quad => source.quad_command(Command::SetStorm(storm)),
            }
        }
        if ui.clear_storm {
            match source.kind() {
                AircraftKind::FixedWing => source.fw_command(FwCommand::SetStorm(None)),
                AircraftKind::Quad => source.quad_command(Command::SetStorm(None)),
            }
        }

        // --- route dispatch (after the egui pass, so no live egui borrow) ---
        if map_actions.fly && route.wps.len() >= 2 {
            let alt = route.alt_up as f64;
            if source.kind() == AircraftKind::FixedWing {
                // Fixed-wing routes are PCI great circles from the planisphere's
                // geographic (lat, lon) waypoints.
                let wps: Vec<Waypoint> = route
                    .wps
                    .iter()
                    .map(|w| Waypoint::geodetic(w.lat as f64, w.lon as f64, alt))
                    .collect();
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
                // The quad flies in its flat local-NED frame at home: map the
                // geographic waypoints back to local N/E near the home point.
                let r = planet::PLANET_RADIUS;
                let wps: Vec<Waypoint> = route
                    .wps
                    .iter()
                    .map(|w| Waypoint::ne_alt(w.lat as f64 * r, w.lon as f64 * r, alt))
                    .collect();
                // Quad missions need the INS; switch + rebuild if necessary.
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
        let mut objects: Vec<&dyn Object> = vec![&sea, &ground, &trail];
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
        objects.push(&clouds);
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
                    let quad = !st.fixed_wing;
                    let fw_ap = st.fixed_wing && !st.fighter;
                    let fighter = st.fixed_wing && st.fighter;
                    if ui.radio(quad, "Quad").clicked() && !quad {
                        st.fixed_wing = false;
                        st.fighter = false;
                        st.do_airframe_switch = true;
                    }
                    if ui.radio(fw_ap, "Fixed-wing").clicked() && !fw_ap {
                        st.fixed_wing = true;
                        st.fighter = false;
                        st.do_airframe_switch = true;
                    }
                    if ui.radio(fighter, "Fighter (FBW)").clicked() && !fighter {
                        st.fixed_wing = true;
                        st.fighter = true;
                        st.fbw_on = true;
                        st.do_airframe_switch = true;
                    }
                });
                ui.separator();

                if !st.fixed_wing {
                    quad_controls(ui, view, hover, st);
                } else if st.fighter {
                    fighter_controls(ui, st);
                } else {
                    ui.label("Fixed-wing cruise (or draw a route on the minimap)");
                    ui.add(egui::Slider::new(&mut st.fw_airspeed, 12.0..=35.0).text("airspeed"));
                    ui.add(
                        egui::Slider::new(&mut st.fw_altitude, 50.0..=800.0).text("altitude (m)"),
                    );
                    ui.add(egui::Slider::new(&mut st.yaw, -180.0..=180.0).text("course (deg)"));
                    if let Some(idx) = view.waypoint_index {
                        ui.monospace(format!("flying route — leg {idx}"));
                    }
                }
                ui.separator();
                ui.label("weather");
                ui.add(egui::Slider::new(&mut st.wind_speed, 0.0..=15.0).text("wind (m/s)"));
                ui.add(egui::Slider::new(&mut st.turbulence, 0.0..=8.0).text("turbulence (m/s)"));
                ui.horizontal(|ui| {
                    if ui.button("drop microburst").clicked() {
                        st.drop_storm = true;
                    }
                    if ui.button("clear storm").clicked() {
                        st.clear_storm = true;
                    }
                });
                ui.separator();
                ui.label("faults");
                if st.fixed_wing {
                    fault_controls(ui, st);
                } else {
                    ui.horizontal(|ui| {
                        for i in 0..4 {
                            if ui.button(format!("kill rotor {i}")).clicked() {
                                st.quad_dead_rotor = Some(i);
                            }
                        }
                    });
                    // Sensor faults — most visible on the INS estimate-vs-truth plots.
                    sensor_checkbox(
                        ui,
                        "GPS dropout",
                        &mut st.quad_sensors.gps,
                        SensorFault::Dropout,
                    );
                    sensor_checkbox(
                        ui,
                        "baro dropout",
                        &mut st.quad_sensors.baro,
                        SensorFault::Dropout,
                    );
                    sensor_checkbox(
                        ui,
                        "mag dropout",
                        &mut st.quad_sensors.mag,
                        SensorFault::Dropout,
                    );
                    sensor_checkbox(
                        ui,
                        "gyro bias",
                        &mut st.quad_sensors.imu,
                        SensorFault::Bias(0.05),
                    );
                    if ui.button("repair").clicked() {
                        st.quad_dead_rotor = None;
                        st.quad_sensors = SensorFaults::none();
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
            let (r, p, y) = view.local_attitude().euler_angles();
            if view.crashed {
                ui.colored_label(egui::Color32::from_rgb(235, 70, 60), "CRASHED — reset");
            }
            ui.monospace(format!("t        {:7.2} s", view.t));
            ui.monospace(format!("altitude {:7.1} m", view.altitude()));
            match view.kind {
                AircraftKind::Quad => ui.monospace(format!(
                    "pos N/E  {:8.1} {:8.1} m",
                    view.position.x, view.position.y
                )),
                AircraftKind::FixedWing => {
                    let (lat, lon, _) = planet::pci_to_geodetic(view.position);
                    ui.monospace(format!(
                        "lat/lon  {:7.2} {:7.2} deg",
                        lat.to_degrees(),
                        lon.to_degrees()
                    ))
                }
            };
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

/// The relaxed-stability fighter's manual panel: the FCS toggle + the keymap +
/// the spawn altitude. The toggle is the whole demo — ON the airframe is docile,
/// OFF it diverges within a second.
fn fighter_controls(ui: &mut egui::Ui, st: &mut Ui) {
    ui.label("RELAXED-STABILITY FIGHTER — you have the stick.");
    let (txt, col) = if st.fbw_on {
        ("FCS: ON — flyable", egui::Color32::from_rgb(60, 200, 90))
    } else {
        (
            "FCS: OFF — diverging!",
            egui::Color32::from_rgb(235, 70, 60),
        )
    };
    if ui
        .add(
            egui::Button::new(
                egui::RichText::new(txt)
                    .color(egui::Color32::BLACK)
                    .strong(),
            )
            .fill(col)
            .min_size(egui::vec2(220.0, 26.0)),
        )
        .clicked()
    {
        st.fbw_on = !st.fbw_on;
    }
    ui.monospace("F toggle FCS    R reset to trim");
    ui.separator();
    ui.label("controls (keyboard or gamepad)");
    ui.monospace("pitch  W/↑   S/↓");
    ui.monospace("roll   A/←   D/→");
    ui.monospace("yaw    Q     E");
    ui.monospace("thrust Shift / Ctrl");
    ui.separator();
    ui.add(egui::Slider::new(&mut st.fw_altitude, 100.0..=1500.0).text("spawn alt (m)"));
}

/// The fixed-wing fault toggles: engine, control surfaces, tail.
fn fault_controls(ui: &mut egui::Ui, st: &mut Ui) {
    ui.checkbox(&mut st.fw_faults.engine_out, "engine out");
    ui.checkbox(&mut st.fw_faults.tail_loss, "tail loss");
    jam_checkbox(ui, "jam elevator", &mut st.fw_faults.elevator);
    jam_checkbox(ui, "jam aileron", &mut st.fw_faults.aileron);
    jam_checkbox(ui, "jam rudder", &mut st.fw_faults.rudder);
    if ui.button("repair").clicked() {
        st.fw_faults = FwFaults::none();
    }
}

/// A checkbox that jams a surface at neutral (on) or clears the fault (off).
fn jam_checkbox(ui: &mut egui::Ui, label: &str, surf: &mut SurfaceFault) {
    let mut on = matches!(surf, SurfaceFault::Jammed(_));
    if ui.checkbox(&mut on, label).changed() {
        *surf = if on {
            SurfaceFault::Jammed(0.0)
        } else {
            SurfaceFault::Normal
        };
    }
}

/// A checkbox that applies `on_fault` to a sensor (on) or clears it (off).
fn sensor_checkbox(ui: &mut egui::Ui, label: &str, sf: &mut SensorFault, on_fault: SensorFault) {
    let mut on = *sf != SensorFault::Normal;
    if ui.checkbox(&mut on, label).changed() {
        *sf = if on { on_fault } else { SensorFault::Normal };
    }
}

/// The pilot HUD overlay for the fighter: a prominent FCS ON/OFF banner (with a
/// DIVERGING warning when off) plus an airspeed/altitude/AoA/g readout. Drawn as
/// borderless anchored egui areas over the 3D scene.
fn hud_overlay(ctx: &egui::Context, view: &ViewSnapshot, has_gamepad: bool) {
    let (roll, pitch, _) = view.local_attitude().euler_angles();
    let speed = view.velocity.norm();
    let on = view.fbw_on;

    egui::Area::new(egui::Id::new("hud_fcs"))
        .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 12.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                if view.crashed {
                    ui.label(
                        egui::RichText::new("CRASHED — press R")
                            .color(egui::Color32::from_rgb(235, 70, 60))
                            .strong()
                            .size(26.0),
                    );
                    return;
                }
                let (txt, col) = if on {
                    ("FCS: ON", egui::Color32::from_rgb(60, 200, 90))
                } else {
                    ("FCS: OFF", egui::Color32::from_rgb(235, 70, 60))
                };
                ui.label(egui::RichText::new(txt).color(col).strong().size(24.0));
                if !on {
                    ui.label(
                        egui::RichText::new("⚠ DIVERGING — press F")
                            .color(egui::Color32::from_rgb(235, 70, 60))
                            .strong(),
                    );
                }
                if view.storm > 0.15 {
                    ui.label(
                        egui::RichText::new("⛈ STORM")
                            .color(egui::Color32::from_rgb(255, 200, 40))
                            .strong(),
                    );
                }
                if view.faulted {
                    ui.label(
                        egui::RichText::new("⚠ FAULT")
                            .color(egui::Color32::from_rgb(235, 70, 60))
                            .strong(),
                    );
                }
            });
        });

    egui::Area::new(egui::Id::new("hud_readout"))
        .anchor(egui::Align2::LEFT_BOTTOM, egui::vec2(12.0, -12.0))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.monospace(format!("airspeed {speed:6.1} m/s"));
                ui.monospace(format!("altitude {:6.1} m", view.altitude()));
                ui.monospace(format!("density  {:6.3} kg/m3", view.density));
                ui.monospace(format!("AoA      {:6.1} deg", view.alpha.to_degrees()));
                ui.monospace(format!("load     {:6.2} g", view.load_factor));
                ui.monospace(format!("pitch    {:6.1} deg", pitch.to_degrees()));
                ui.monospace(format!("roll     {:6.1} deg", roll.to_degrees()));
                if view.wind_speed > 0.1 || view.gust > 0.1 {
                    ui.monospace(format!(
                        "wind {:4.1}  gust {:4.1} m/s",
                        view.wind_speed, view.gust
                    ));
                }
                ui.monospace(format!(
                    "input    {}",
                    if has_gamepad { "gamepad" } else { "keyboard" }
                ));
            });
        });
}

/// The quad-only selectors + attitude sliders (estimator, controller, mission).
fn quad_controls(ui: &mut egui::Ui, _view: &ViewSnapshot, hover: f64, st: &mut Ui) {
    ui.label("estimator");
    ui.horizontal(|ui| {
        for (k, name) in [(0u8, "CF"), (1, "MEKF"), (2, "INS")] {
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
    // Inner attitude controller: PID vs LQR.
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
            // Position estimate vs truth (INS only; CF/MEKF leave it at zero).
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

            // Gyro-bias estimate vs the hidden truth (the MEKF's key win).
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
                                    .map(|s| [s.t, planet::altitude_of(s.truth.position)])
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
                                        // Local-frame course (PCI velocity → local NED).
                                        let v = planet::ned_from_pci(s.truth.position)
                                            * s.truth.velocity;
                                        [s.t, v.y.atan2(v.x).to_degrees()]
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
