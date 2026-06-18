//! Top-down route-planner minimap.
//!
//! A precomputed shaded-relief image of the terrain (uploaded to egui **once**),
//! over which we paint the planned route, the recent trail, and the aircraft.
//! Clicking the map edits the route in world (North, East) coordinates.
//!
//! ## Coordinate convention (do not drift)
//! World is NED: x = North, y = East, z = Down. The map is **North up, East
//! right**. So in *screen pixels* (see [`world_to_screen`]):
//! ```text
//!   screen.x = center.x + (E / half_extent) * (w/2)   // +East  -> +x (right)
//!   screen.y = center.y - (N / half_extent) * (h/2)   // +North -> -y (UP)
//! ```
//! The y flip is the whole ballgame: North up means +N maps to a *smaller*
//! pixel y. Every transform here carries that minus sign, and the aircraft
//! marker ([`course_screen_dir`]) uses the same convention.

use three_d::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Vec2,
};

/// Side of the (square) relief texture in texels. 320² Color32 ≈ 410 KB — built
/// once, well under egui's `max_texture_side`.
const MAP_TEXELS: usize = 320;

/// Minimal trait so the minimap doesn't hard-depend on the concrete `Terrain`
/// type (and stays unit-testable with a stub). Heights/normals are in the sim's
/// NED frame (north, east in metres); an **up-facing** surface normal has a
/// *negative* z (NED `+z` is Down), which the NW-and-above hillshade light
/// below relies on.
pub trait TerrainLike {
    fn height(&self, north: f32, east: f32) -> f32;
    fn normal(&self, north: f32, east: f32) -> three_d::Vec3;
    fn color(&self, north: f32, east: f32) -> (u8, u8, u8);
}

/// One planned waypoint, in NED world metres (horizontal only; altitude is the
/// route-wide [`Route::alt_up`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteWp {
    pub north: f32,
    pub east: f32,
}

/// The editable route plus its shared parameters.
#[derive(Clone, Debug)]
pub struct Route {
    pub wps: Vec<RouteWp>,
    /// Altitude applied to every leg, metres **+up** (converted to NED `-z` only
    /// at dispatch).
    pub alt_up: f32,
    /// Cruise airspeed \[m/s\] (fixed-wing only).
    pub cruise: f32,
}

impl Default for Route {
    fn default() -> Self {
        Self {
            wps: Vec::new(),
            alt_up: 60.0,
            cruise: 25.0,
        }
    }
}

impl Route {
    fn leg_len(a: RouteWp, b: RouteWp) -> f32 {
        ((b.north - a.north).powi(2) + (b.east - a.east).powi(2)).sqrt()
    }
    pub fn total_len(&self) -> f32 {
        self.wps.windows(2).map(|w| Self::leg_len(w[0], w[1])).sum()
    }
    pub fn leg_lens(&self) -> Vec<f32> {
        self.wps
            .windows(2)
            .map(|w| Self::leg_len(w[0], w[1]))
            .collect()
    }
}

/// Plain per-frame view data the minimap paints (decouples it from `Source`).
pub struct MinimapView<'a> {
    /// Aircraft NED position (metres).
    pub pos_north: f32,
    pub pos_east: f32,
    /// Course χ (rad) — `atan2(vel.y, vel.x)`, heading fallback near hover.
    pub course: f32,
    /// Active waypoint = the one we're flying toward (highlights `wp-1 -> wp`).
    pub active_wp: Option<usize>,
    /// Recent (north, east) trail samples; oldest first.
    pub trail: &'a [(f32, f32)],
}

/// What the user did to the route this frame (consumed by `main.rs`).
#[derive(Default)]
pub struct MinimapActions {
    /// "Fly route" pressed.
    pub fly: bool,
    /// Route was cleared (so the caller can cancel the mission too).
    pub clear: bool,
}

/// Owns the cached relief texture + a little drag state.
pub struct Minimap {
    tex: Option<TextureHandle>,
    half_extent: f32,
    dragging: Option<usize>,
}

impl Minimap {
    pub fn new(half_extent: f32) -> Self {
        Self {
            tex: None,
            half_extent,
            dragging: None,
        }
    }

    /// Build the shaded-relief texture **once** and cache the handle. Hillshade
    /// = clamp(n · L) with the light to the NW-and-above, plus a gentle
    /// elevation ramp for legibility.
    fn ensure_texture(&mut self, ctx: &egui::Context, terrain: &impl TerrainLike) {
        if self.tex.is_some() {
            return;
        }
        let (w, h) = (MAP_TEXELS, MAP_TEXELS);
        let he = self.half_extent;
        let l = hillshade_light();
        // Height span for the ramp (cheap two-pass; runs once at startup).
        let mut hmin = f32::INFINITY;
        let mut hmax = f32::NEG_INFINITY;
        for j in 0..h {
            for i in 0..w {
                let (n, e) = texel_to_world(i, j, w, h, he);
                let z = terrain.height(n, e);
                hmin = hmin.min(z);
                hmax = hmax.max(z);
            }
        }
        let span = (hmax - hmin).max(1.0);
        let mut pixels = Vec::with_capacity(w * h);
        for j in 0..h {
            for i in 0..w {
                let (n, e) = texel_to_world(i, j, w, h, he);
                let z = terrain.height(n, e);
                let nrm = terrain.normal(n, e); // NED up-normal (z < 0)
                let lit = hillshade(nrm, l);
                let t = ((z - hmin) / span).clamp(0.0, 1.0);
                let (br, bg, bb) = terrain.color(n, e);
                // Tint the base albedo by the elevation ramp, then by hillshade.
                let r = (lerp_u8(br, 235, t) as f32 * lit) as u8;
                let g = (lerp_u8(bg, 225, t) as f32 * lit) as u8;
                let b = (lerp_u8(bb, 200, t) as f32 * lit) as u8;
                pixels.push(Color32::from_rgb(r, g, b));
            }
        }
        let img = ColorImage::new([w, h], pixels);
        self.tex = Some(ctx.load_texture("minimap_relief", img, TextureOptions::LINEAR));
    }

    /// Show the window. Returns actions the caller must apply (fly/clear). The
    /// route is edited in place. `view` is the live aircraft projection.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
        terrain: &impl TerrainLike,
        route: &mut Route,
        view: &MinimapView,
        is_fixed_wing: bool,
        is_replay: bool,
    ) -> MinimapActions {
        self.ensure_texture(ctx, terrain);
        let mut act = MinimapActions::default();
        let tex = self.tex.as_ref().expect("texture built above").clone();
        let he = self.half_extent;

        egui::Window::new("Route planner")
            .default_pos([1180.0, 12.0])
            .default_width(360.0)
            .resizable(false)
            .show(ctx, |ui| {
                let side = 320.0_f32;
                // CRITICAL: click_and_drag so egui "uses the pointer" while the
                // user interacts (so OrbitControl skips the events — see the
                // module-level event-consumption note in main.rs).
                let (resp, painter) =
                    ui.allocate_painter(Vec2::new(side, side), Sense::click_and_drag());
                let rect = resp.rect;

                painter.image(
                    tex.id(),
                    rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                let painter = painter.with_clip_rect(rect);

                let to_screen = |n: f32, e: f32| world_to_screen(rect, he, n, e);
                let to_world = |p: Pos2| screen_to_world(rect, he, p);

                if !is_replay {
                    handle_interaction(&resp, route, &mut self.dragging, &to_world, &to_screen);
                }

                // Route polyline (active leg highlighted).
                if route.wps.len() >= 2 {
                    let active = view.active_wp.unwrap_or(usize::MAX);
                    for (i, w) in route.wps.windows(2).enumerate() {
                        let a = to_screen(w[0].north, w[0].east);
                        let b = to_screen(w[1].north, w[1].east);
                        let stroke = if i + 1 == active {
                            Stroke::new(3.0, Color32::from_rgb(255, 210, 60))
                        } else {
                            Stroke::new(2.0, Color32::from_rgb(90, 200, 210))
                        };
                        painter.line_segment([a, b], stroke);
                    }
                }
                // Waypoint dots + index labels.
                for (i, wp) in route.wps.iter().enumerate() {
                    let p = to_screen(wp.north, wp.east);
                    let c = if Some(i) == view.active_wp {
                        Color32::from_rgb(255, 210, 60)
                    } else {
                        Color32::from_rgb(240, 140, 40)
                    };
                    painter.circle_filled(p, 5.0, c);
                    painter.circle_stroke(p, 5.0, Stroke::new(1.0, Color32::BLACK));
                    painter.text(
                        p + Vec2::new(7.0, -7.0),
                        Align2::LEFT_BOTTOM,
                        format!("{i}"),
                        FontId::monospace(11.0),
                        Color32::WHITE,
                    );
                }
                // Recent trail.
                if view.trail.len() >= 2 {
                    let pts: Vec<Pos2> = view.trail.iter().map(|&(n, e)| to_screen(n, e)).collect();
                    painter.line(pts, Stroke::new(1.5, Color32::from_rgb(120, 120, 160)));
                }
                // Aircraft triangle, nose along course.
                draw_aircraft(
                    &painter,
                    to_screen(view.pos_north, view.pos_east),
                    view.course,
                );
                // North hint.
                painter.text(
                    rect.left_top() + Vec2::new(6.0, 4.0),
                    Align2::LEFT_TOP,
                    "N↑",
                    FontId::monospace(12.0),
                    Color32::WHITE,
                );

                // Controls below the map.
                ui.add_space(4.0);
                ui.add(egui::Slider::new(&mut route.alt_up, 0.0..=200.0).text("route alt (m, up)"));
                if is_fixed_wing {
                    ui.add(egui::Slider::new(&mut route.cruise, 12.0..=35.0).text("cruise (m/s)"));
                }
                ui.horizontal(|ui| {
                    if ui.button("Remove last").clicked() {
                        route.wps.pop();
                        self.dragging = None;
                    }
                    if ui.button("Clear").clicked() {
                        route.wps.clear();
                        self.dragging = None;
                        act.clear = true;
                    }
                    let can_fly = route.wps.len() >= 2 && !is_replay;
                    if ui
                        .add_enabled(can_fly, egui::Button::new("Fly route"))
                        .clicked()
                    {
                        act.fly = true;
                    }
                });
                let legs = route.leg_lens();
                ui.monospace(format!(
                    "{} wp   total {:8.1} m",
                    route.wps.len(),
                    route.total_len()
                ));
                if let Some(active) = view.active_wp {
                    if active >= 1 && active - 1 < legs.len() {
                        ui.monospace(format!("leg {active}: {:8.1} m", legs[active - 1]));
                    }
                }
                ui.label("click: add  ·  right-click: remove  ·  drag: move");
                if is_replay {
                    ui.colored_label(Color32::GRAY, "(replay: route editing disabled)");
                }
            });

        act
    }
}

/// World (north, east) → screen pixel (North up, East right; `+N → −y`).
fn world_to_screen(rect: Rect, he: f32, n: f32, e: f32) -> Pos2 {
    Pos2::new(
        rect.center().x + (e / he) * (rect.width() * 0.5),
        rect.center().y - (n / he) * (rect.height() * 0.5),
    )
}

/// Screen pixel → world (north, east). Inverse of [`world_to_screen`].
fn screen_to_world(rect: Rect, he: f32, p: Pos2) -> (f32, f32) {
    let e = (p.x - rect.center().x) / (rect.width() * 0.5) * he;
    let n = -(p.y - rect.center().y) / (rect.height() * 0.5) * he;
    (n, e)
}

/// Forward direction of the aircraft marker in *screen* space for a course χ:
/// `dN → −y` (up), `dE → +x` (right). Course 0 (North) → (0, −1); +π/2 (East) →
/// (+1, 0).
fn course_screen_dir(course: f32) -> Vec2 {
    Vec2::new(course.sin(), -course.cos())
}

/// Add / move / remove waypoints from the canvas `Response`. Reads the pointer
/// via the egui `Response` only (never raw three-d events).
fn handle_interaction(
    resp: &egui::Response,
    route: &mut Route,
    dragging: &mut Option<usize>,
    to_world: &dyn Fn(Pos2) -> (f32, f32),
    to_screen: &dyn Fn(f32, f32) -> Pos2,
) {
    let hit_radius = 9.0_f32;
    if resp.drag_started() {
        if let Some(p) = resp.interact_pointer_pos() {
            *dragging = nearest_wp(route, p, to_screen, hit_radius);
        }
    }
    if resp.dragged() {
        if let (Some(idx), Some(p)) = (*dragging, resp.interact_pointer_pos()) {
            let (n, e) = to_world(p);
            route.wps[idx] = RouteWp { north: n, east: e };
        }
    }
    if resp.drag_stopped() {
        *dragging = None;
    }
    // Primary click that wasn't a drag-on-a-wp: append a waypoint.
    if resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            if nearest_wp(route, p, to_screen, hit_radius).is_none() {
                let (n, e) = to_world(p);
                route.wps.push(RouteWp { north: n, east: e });
            }
        }
    }
    // Secondary click: remove nearest waypoint.
    if resp.secondary_clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            if let Some(idx) = nearest_wp(route, p, to_screen, hit_radius * 1.6) {
                route.wps.remove(idx);
            }
        }
    }
}

fn nearest_wp(
    route: &Route,
    p: Pos2,
    to_screen: &dyn Fn(f32, f32) -> Pos2,
    radius: f32,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, wp) in route.wps.iter().enumerate() {
        let s = to_screen(wp.north, wp.east);
        let d2 = (s.x - p.x).powi(2) + (s.y - p.y).powi(2);
        if d2 <= radius * radius && best.is_none_or(|(_, bd)| d2 < bd) {
            best = Some((i, d2));
        }
    }
    best.map(|(i, _)| i)
}

/// An isosceles triangle whose apex points along `course` (rad, χ).
fn draw_aircraft(painter: &egui::Painter, c: Pos2, course: f32) {
    let fwd = course_screen_dir(course);
    let right = Vec2::new(-fwd.y, fwd.x); // 90° CW of fwd in screen space
    let len = 9.0;
    let half = 5.0;
    let nose = c + fwd * len;
    let l = c - fwd * (len * 0.5) + right * half;
    let r = c - fwd * (len * 0.5) - right * half;
    painter.add(egui::Shape::convex_polygon(
        vec![nose, l, r],
        Color32::from_rgb(70, 130, 215),
        Stroke::new(1.5, Color32::WHITE),
    ));
}

// --- helpers ---

/// Texel `(i,j)` center → world (north, east). Row `j=0` is the TOP = max North.
fn texel_to_world(i: usize, j: usize, w: usize, h: usize, he: f32) -> (f32, f32) {
    let u = (i as f32 + 0.5) / w as f32; // 0..1 left→right = West→East
    let v = (j as f32 + 0.5) / h as f32; // 0..1 top→bottom = North→South
    let east = (u * 2.0 - 1.0) * he;
    let north = (1.0 - v * 2.0) * he; // v=0 → +he (North up)
    (north, east)
}

fn normalize3(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let m = (a * a + b * b + c * c).sqrt().max(1e-6);
    (a / m, b / m, c / m)
}

/// Hillshade light direction (TOWARD the source): NW and above. Up-facing NED
/// normals have `z < 0`, so "above" is the −z component — the light's z is
/// negative so flat ground reads bright.
fn hillshade_light() -> (f32, f32, f32) {
    normalize3(0.4, -0.5, -0.75)
}

/// Diffuse hillshade (ambient floor + clamped `n · L`) for an up-facing NED
/// normal `nrm` (z < 0) and light direction `l` (toward the source).
fn hillshade(nrm: three_d::Vec3, l: (f32, f32, f32)) -> f32 {
    let shade = (nrm.x * l.0 + nrm.y * l.1 + nrm.z * l.2).clamp(0.0, 1.0);
    0.35 + 0.65 * shade // ambient floor + diffuse
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        // A 320×320 canvas at the origin.
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(320.0, 320.0))
    }

    #[test]
    fn screen_world_round_trip() {
        let r = rect();
        let he = 500.0;
        for &(n, e) in &[(0.0, 0.0), (250.0, -100.0), (-400.0, 480.0), (123.0, 45.0)] {
            let p = world_to_screen(r, he, n, e);
            let (n2, e2) = screen_to_world(r, he, p);
            assert!((n - n2).abs() < 1e-3, "north {n} != {n2}");
            assert!((e - e2).abs() < 1e-3, "east {e} != {e2}");
        }
    }

    #[test]
    fn north_up_east_right_known_points() {
        let r = rect();
        let he = 500.0;
        let center = r.center();
        // +North (max) → top-center: same x, smaller y.
        let top = world_to_screen(r, he, he, 0.0);
        assert!((top.x - center.x).abs() < 1e-3);
        assert!(top.y < center.y, "max North should map ABOVE center");
        assert!((top.y - r.top()).abs() < 1e-3, "max North → top edge");
        // +East (max) → right-center: same y, larger x.
        let right = world_to_screen(r, he, 0.0, he);
        assert!((right.y - center.y).abs() < 1e-3);
        assert!(right.x > center.x, "max East should map RIGHT of center");
        assert!((right.x - r.right()).abs() < 1e-3, "max East → right edge");
    }

    #[test]
    fn aircraft_heading_directions() {
        // Course 0 (North) → up = (0, -1).
        let up = course_screen_dir(0.0);
        assert!(up.x.abs() < 1e-6 && (up.y + 1.0).abs() < 1e-6, "{up:?}");
        // Course +π/2 (East) → right = (+1, 0).
        let east = course_screen_dir(std::f32::consts::FRAC_PI_2);
        assert!(
            (east.x - 1.0).abs() < 1e-6 && east.y.abs() < 1e-6,
            "{east:?}"
        );
    }

    #[test]
    fn hillshade_lights_up_facing_ground() {
        let l = hillshade_light();
        // Flat ground: up-facing normal in NED has z = -1 → near full diffuse.
        let lit_flat = hillshade(three_d::Vec3::new(0.0, 0.0, -1.0), l);
        assert!(
            lit_flat > 0.8,
            "flat up-facing ground should be bright: {lit_flat}"
        );
        // A down-facing normal (z = +1, into the ground) must sit at the ambient
        // floor — guards against the light/normal sign convention flipping.
        let lit_down = hillshade(three_d::Vec3::new(0.0, 0.0, 1.0), l);
        assert!(
            lit_down < 0.4,
            "down-facing should be at the ambient floor: {lit_down}"
        );
        assert!(
            lit_flat > lit_down,
            "up-facing must be brighter than down-facing"
        );
    }

    #[test]
    fn route_lengths() {
        let route = Route {
            wps: vec![
                RouteWp {
                    north: 0.0,
                    east: 0.0,
                },
                RouteWp {
                    north: 300.0,
                    east: 0.0,
                },
                RouteWp {
                    north: 300.0,
                    east: 400.0,
                },
            ],
            ..Default::default()
        };
        assert!((route.total_len() - 700.0).abs() < 1e-3);
        assert_eq!(route.leg_lens().len(), 2);
        assert!((route.leg_lens()[1] - 400.0).abs() < 1e-3);
    }
}
