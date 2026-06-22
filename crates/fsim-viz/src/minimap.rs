//! Planisphere route planner — a flat **equirectangular world map** of the whole
//! planet. The terrain is baked into one shaded-relief texture once (longitude →
//! x, latitude → y); the visible window is a sub-rectangle of that texture, so
//! **zooming** (scroll) and panning cost nothing. Clicking the map edits a route
//! in geographic (latitude, longitude) coordinates; route legs are great circles
//! drawn as projected polylines.
//!
//! ## Projection
//! Longitude `lon ∈ (−π, π]` maps left→right, latitude `lat ∈ (−π/2, π/2)` maps
//! bottom→top (North up). The view shows a window of half-spans
//! `hlon = π/zoom`, `hlat = (π/2)/zoom` centred on `center = (lon, lat)`; the
//! 2:1 longitude:latitude ratio keeps continents un-stretched at `zoom = 1`
//! (the whole world).

use three_d::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, Sense, Stroke, TextureHandle,
    TextureOptions, Vec2,
};

/// Equirectangular relief texture size (2:1). Built once at startup.
const MAP_W: usize = 512;
const MAP_H: usize = 256;
const PI: f32 = std::f32::consts::PI;
const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;

/// Minimal terrain interface for the map: a shaded colour at a geographic point.
/// Returns `(rgb, hillshade ∈ [0,1])` so the relief reads as topography.
pub trait TerrainLike {
    fn map_sample(&self, lat: f32, lon: f32) -> ((u8, u8, u8), f32);
}

/// One planned waypoint in **geographic** coordinates (radians).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RouteWp {
    pub lat: f32,
    pub lon: f32,
}

/// The editable route plus its shared parameters.
#[derive(Clone, Debug)]
pub struct Route {
    pub wps: Vec<RouteWp>,
    /// Altitude applied to every leg, metres **+up**.
    pub alt_up: f32,
    /// Cruise airspeed \[m/s\] (fixed-wing only).
    pub cruise: f32,
}

impl Default for Route {
    fn default() -> Self {
        Self {
            wps: Vec::new(),
            alt_up: 400.0,
            cruise: 25.0,
        }
    }
}

impl Route {
    /// Great-circle distance \[m\] between two waypoints.
    fn leg_len(a: RouteWp, b: RouteWp) -> f32 {
        let da = geo_dir(a.lat, a.lon);
        let db = geo_dir(b.lat, b.lon);
        let dot = (da[0] * db[0] + da[1] * db[1] + da[2] * db[2]).clamp(-1.0, 1.0);
        R_PLANET * dot.acos()
    }
    pub fn total_len(&self) -> f32 {
        self.wps.windows(2).map(|w| Self::leg_len(w[0], w[1])).sum()
    }
}

/// Per-frame aircraft projection the minimap paints.
pub struct MinimapView<'a> {
    /// Aircraft geographic position (radians).
    pub lat: f32,
    pub lon: f32,
    /// Course χ (rad) in the local NED frame.
    pub course: f32,
    /// Active waypoint = the one we're flying toward (highlights `wp-1 → wp`).
    pub active_wp: Option<usize>,
    /// Recent `(lat, lon)` trail; oldest first.
    pub trail: &'a [(f32, f32)],
}

/// What the user did to the route this frame (consumed by `main.rs`).
#[derive(Default)]
pub struct MinimapActions {
    pub fly: bool,
    pub clear: bool,
}

const R_PLANET: f32 = 6371.0;

/// Owns the cached relief texture + zoom/pan/drag state.
pub struct Minimap {
    tex: Option<TextureHandle>,
    zoom: f32,
    /// View centre `(lon, lat)` in radians.
    center: (f32, f32),
    dragging: Option<usize>,
}

impl Minimap {
    pub fn new(_half_extent: f32) -> Self {
        Self {
            tex: None,
            zoom: 1.0,
            center: (0.0, 0.0),
            dragging: None,
        }
    }

    /// Build the whole-planet relief texture once.
    fn ensure_texture(&mut self, ctx: &egui::Context, terrain: &impl TerrainLike) {
        if self.tex.is_some() {
            return;
        }
        let mut pixels = Vec::with_capacity(MAP_W * MAP_H);
        for j in 0..MAP_H {
            let lat = (1.0 - 2.0 * (j as f32 + 0.5) / MAP_H as f32) * FRAC_PI_2;
            for i in 0..MAP_W {
                let lon = (2.0 * (i as f32 + 0.5) / MAP_W as f32 - 1.0) * PI;
                let ((r, g, b), shade) = terrain.map_sample(lat, lon);
                let s = shade.clamp(0.0, 1.0);
                pixels.push(Color32::from_rgb(
                    (r as f32 * s) as u8,
                    (g as f32 * s) as u8,
                    (b as f32 * s) as u8,
                ));
            }
        }
        let img = ColorImage::new([MAP_W, MAP_H], pixels);
        self.tex = Some(ctx.load_texture("planisphere", img, TextureOptions::LINEAR));
    }

    /// Clamp the view centre so the visible window never runs off the world.
    fn clamp_center(&mut self) {
        self.zoom = self.zoom.clamp(1.0, 32.0);
        let hlon = PI / self.zoom;
        let hlat = FRAC_PI_2 / self.zoom;
        self.center.0 = self.center.0.clamp(-PI + hlon, PI - hlon);
        self.center.1 = self.center.1.clamp(-FRAC_PI_2 + hlat, FRAC_PI_2 - hlat);
    }

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

        egui::Window::new("Route planner")
            .default_pos([1180.0, 12.0])
            .default_width(360.0)
            .resizable(false)
            .show(ctx, |ui| {
                let (w, h) = (340.0_f32, 200.0_f32); // 1.7:1 canvas
                let (resp, painter) = ui.allocate_painter(Vec2::new(w, h), Sense::click_and_drag());
                let rect = resp.rect;

                // --- zoom (scroll, toward the cursor) ---
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll.abs() > 0.01 {
                        if let Some(cur) = resp.hover_pos() {
                            let (lat_c, lon_c) = self.to_world(rect, cur);
                            self.zoom = (self.zoom * (1.0 + scroll * 0.0015)).clamp(1.0, 32.0);
                            // Keep the cursor's world point fixed under the cursor.
                            let hlon = PI / self.zoom;
                            let hlat = FRAC_PI_2 / self.zoom;
                            let fx = (cur.x - rect.center().x) / (rect.width() * 0.5);
                            let fy = (cur.y - rect.center().y) / (rect.height() * 0.5);
                            self.center = (lon_c - fx * hlon, lat_c + fy * hlat);
                            self.clamp_center();
                        }
                    }
                }

                // --- terrain (visible UV sub-rect of the world texture) ---
                let hlon = PI / self.zoom;
                let hlat = FRAC_PI_2 / self.zoom;
                let uc = (self.center.0 / PI + 1.0) * 0.5;
                let vc = (1.0 - self.center.1 / FRAC_PI_2) * 0.5;
                let uh = (hlon / PI) * 0.5;
                let vh = (hlat / FRAC_PI_2) * 0.5;
                painter.image(
                    tex.id(),
                    rect,
                    Rect::from_min_max(Pos2::new(uc - uh, vc - vh), Pos2::new(uc + uh, vc + vh)),
                    Color32::WHITE,
                );
                let painter = painter.with_clip_rect(rect);

                if !is_replay {
                    self.handle_interaction(&resp, route, rect);
                }

                // --- route great-circle legs ---
                let active = view.active_wp.unwrap_or(usize::MAX);
                for (i, w2) in route.wps.windows(2).enumerate() {
                    let stroke = if i + 1 == active {
                        Stroke::new(3.0, Color32::from_rgb(255, 210, 60))
                    } else {
                        Stroke::new(2.0, Color32::from_rgb(90, 200, 210))
                    };
                    let pts = self.great_circle_screen(rect, w2[0], w2[1]);
                    if pts.len() >= 2 {
                        painter.line(pts, stroke);
                    }
                }
                // --- waypoint dots + labels ---
                for (i, wp) in route.wps.iter().enumerate() {
                    let p = self.to_screen(rect, wp.lat, wp.lon);
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
                // --- recent trail ---
                if view.trail.len() >= 2 {
                    let pts: Vec<Pos2> = view
                        .trail
                        .iter()
                        .map(|&(la, lo)| self.to_screen(rect, la, lo))
                        .collect();
                    painter.line(pts, Stroke::new(1.5, Color32::from_rgb(120, 120, 160)));
                }
                // --- aircraft ---
                draw_aircraft(
                    &painter,
                    self.to_screen(rect, view.lat, view.lon),
                    view.course,
                );
                painter.text(
                    rect.left_top() + Vec2::new(6.0, 4.0),
                    Align2::LEFT_TOP,
                    "N↑",
                    FontId::monospace(12.0),
                    Color32::WHITE,
                );

                // --- controls ---
                ui.add_space(4.0);
                ui.add(
                    egui::Slider::new(&mut route.alt_up, 0.0..=2000.0).text("route alt (m, up)"),
                );
                if is_fixed_wing {
                    ui.add(egui::Slider::new(&mut route.cruise, 12.0..=80.0).text("cruise (m/s)"));
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
                    if ui.button("Reset view").clicked() {
                        self.zoom = 1.0;
                        self.center = (0.0, 0.0);
                    }
                });
                ui.monospace(format!(
                    "{} wp   total {:8.0} m   zoom {:.1}×",
                    route.wps.len(),
                    route.total_len(),
                    self.zoom
                ));
                ui.label("click: add · right-click: remove · drag: move · scroll: zoom");
                if is_replay {
                    ui.colored_label(Color32::GRAY, "(replay: route editing disabled)");
                }
            });

        act
    }

    /// Screen pixel for a geographic point under the current view.
    fn to_screen(&self, rect: Rect, lat: f32, lon: f32) -> Pos2 {
        let hlon = PI / self.zoom;
        let hlat = FRAC_PI_2 / self.zoom;
        let dlon = wrap_pi(lon - self.center.0);
        Pos2::new(
            rect.center().x + (dlon / hlon) * (rect.width() * 0.5),
            rect.center().y - ((lat - self.center.1) / hlat) * (rect.height() * 0.5),
        )
    }

    /// Geographic point (lat, lon) under a screen pixel.
    fn to_world(&self, rect: Rect, p: Pos2) -> (f32, f32) {
        let hlon = PI / self.zoom;
        let hlat = FRAC_PI_2 / self.zoom;
        let lon = self.center.0 + (p.x - rect.center().x) / (rect.width() * 0.5) * hlon;
        let lat = (self.center.1 - (p.y - rect.center().y) / (rect.height() * 0.5) * hlat)
            .clamp(-FRAC_PI_2 + 1e-4, FRAC_PI_2 - 1e-4);
        (lat, wrap_pi(lon))
    }

    /// A great-circle leg, sampled and projected to screen-space points.
    fn great_circle_screen(&self, rect: Rect, a: RouteWp, b: RouteWp) -> Vec<Pos2> {
        let da = geo_dir(a.lat, a.lon);
        let db = geo_dir(b.lat, b.lon);
        let dot = (da[0] * db[0] + da[1] * db[1] + da[2] * db[2]).clamp(-1.0, 1.0);
        let ang = dot.acos();
        let steps = ((ang / 0.02).ceil() as usize).clamp(2, 256);
        (0..=steps)
            .map(|k| {
                let t = k as f32 / steps as f32;
                // SLERP on the sphere.
                let (s0, s1) = if ang.abs() < 1e-5 {
                    (1.0 - t, t)
                } else {
                    (
                        ((1.0 - t) * ang).sin() / ang.sin(),
                        (t * ang).sin() / ang.sin(),
                    )
                };
                let d = [
                    da[0] * s0 + db[0] * s1,
                    da[1] * s0 + db[1] * s1,
                    da[2] * s0 + db[2] * s1,
                ];
                let (lat, lon) = dir_geo(d);
                self.to_screen(rect, lat, lon)
            })
            .collect()
    }

    fn handle_interaction(&mut self, resp: &egui::Response, route: &mut Route, rect: Rect) {
        let hit = 9.0_f32;
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                self.dragging = self.nearest_wp(route, rect, p, hit);
            }
        }
        if resp.dragged() {
            if let (Some(idx), Some(p)) = (self.dragging, resp.interact_pointer_pos()) {
                let (lat, lon) = self.to_world(rect, p);
                route.wps[idx] = RouteWp { lat, lon };
            }
        }
        if resp.drag_stopped() {
            self.dragging = None;
        }
        if resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if self.nearest_wp(route, rect, p, hit).is_none() {
                    let (lat, lon) = self.to_world(rect, p);
                    route.wps.push(RouteWp { lat, lon });
                }
            }
        }
        if resp.secondary_clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                if let Some(idx) = self.nearest_wp(route, rect, p, hit * 1.6) {
                    route.wps.remove(idx);
                }
            }
        }
    }

    fn nearest_wp(&self, route: &Route, rect: Rect, p: Pos2, radius: f32) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, wp) in route.wps.iter().enumerate() {
            let s = self.to_screen(rect, wp.lat, wp.lon);
            let d2 = (s.x - p.x).powi(2) + (s.y - p.y).powi(2);
            if d2 <= radius * radius && best.is_none_or(|(_, bd)| d2 < bd) {
                best = Some((i, d2));
            }
        }
        best.map(|(i, _)| i)
    }
}

/// Geographic (lat, lon) → unit direction (PCI convention: +z = North pole,
/// lon 0 along +x).
fn geo_dir(lat: f32, lon: f32) -> [f32; 3] {
    let cl = lat.cos();
    [cl * lon.cos(), cl * lon.sin(), lat.sin()]
}

/// Unit direction → geographic (lat, lon).
fn dir_geo(d: [f32; 3]) -> (f32, f32) {
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
    ((d[2] / r).clamp(-1.0, 1.0).asin(), d[1].atan2(d[0]))
}

/// Wrap an angle to `(−π, π]`.
fn wrap_pi(x: f32) -> f32 {
    x.sin().atan2(x.cos())
}

/// An isosceles triangle whose apex points along `course` (rad, χ): North up.
fn draw_aircraft(painter: &egui::Painter, c: Pos2, course: f32) {
    let fwd = Vec2::new(course.sin(), -course.cos());
    let right = Vec2::new(-fwd.y, fwd.x);
    let (len, half) = (9.0, 5.0);
    let nose = c + fwd * len;
    let l = c - fwd * (len * 0.5) + right * half;
    let r = c - fwd * (len * 0.5) - right * half;
    painter.add(egui::Shape::convex_polygon(
        vec![nose, l, r],
        Color32::from_rgb(70, 130, 215),
        Stroke::new(1.5, Color32::WHITE),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map() -> Minimap {
        Minimap::new(0.0)
    }
    fn rect() -> Rect {
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(340.0, 200.0))
    }

    #[test]
    fn screen_world_round_trip() {
        let m = map(); // zoom 1, centre (0,0)
        let r = rect();
        for &(lat, lon) in &[(0.0, 0.0), (0.5, -1.0), (-0.9, 2.0), (0.2, 0.7)] {
            let p = m.to_screen(r, lat, lon);
            let (la, lo) = m.to_world(r, p);
            assert!((la - lat).abs() < 1e-3, "lat {lat} != {la}");
            assert!((lo - lon).abs() < 1e-3, "lon {lon} != {lo}");
        }
    }

    #[test]
    fn equator_prime_meridian_is_centre() {
        let m = map();
        let r = rect();
        let c = m.to_screen(r, 0.0, 0.0);
        assert!((c.x - r.center().x).abs() < 1e-3);
        assert!((c.y - r.center().y).abs() < 1e-3);
        // +lat is up (smaller y); +lon is right (larger x).
        assert!(m.to_screen(r, 1.0, 0.0).y < c.y);
        assert!(m.to_screen(r, 0.0, 1.0).x > c.x);
    }

    #[test]
    fn geo_dir_round_trips() {
        for &(lat, lon) in &[(0.0, 0.0), (0.6, 1.5), (-0.9, -2.7)] {
            let (la, lo) = dir_geo(geo_dir(lat, lon));
            assert!((la - lat).abs() < 1e-5 && (lo - lon).abs() < 1e-5);
        }
    }

    #[test]
    fn aircraft_heading_north_points_up() {
        // Course 0 (North) → marker forward (0, -1).
        let fwd = Vec2::new(0.0_f32.sin(), -0.0_f32.cos());
        assert!(fwd.x.abs() < 1e-6 && (fwd.y + 1.0).abs() < 1e-6);
    }

    #[test]
    fn route_leg_length_is_arc() {
        let r = Route {
            wps: vec![
                RouteWp { lat: 0.0, lon: 0.0 },
                RouteWp {
                    lat: 0.0,
                    lon: FRAC_PI_2,
                },
            ],
            ..Default::default()
        };
        // Quarter way around the planet.
        assert!((r.total_len() - R_PLANET * FRAC_PI_2).abs() < 1.0);
    }
}
