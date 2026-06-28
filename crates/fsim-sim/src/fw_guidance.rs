//! Fixed-wing route-following guidance on the **sphere**: walks a list of
//! planet-centered (PCI) [`Waypoint`]s, emitting a [`FixedWingSetpoint`]
//! (airspeed / altitude / course) each control gate and advancing to the next
//! waypoint once inside a *great-circle* acceptance radius. Like the quad
//! [`Guidance`](crate::Guidance), the waypoint *list* lives here (std/heap) so
//! the `fsim-control` math stays no_std-clean.
//!
//! ## Guidance law (great-circle vector field)
//!
//! Each leg from the previous waypoint (the leg origin) to the active waypoint
//! defines a **great circle** with unit normal `n = â × b̂`
//! ([`gc_normal`](fsim_core::planet::gc_normal)). The course to fly is the
//! great-circle *tangent at the aircraft's current position*
//! ([`gc_course`](fsim_core::planet::gc_course)) corrected by the signed
//! cross-track distance ([`gc_cross_track`](fsim_core::planet::gc_cross_track)):
//!
//! ```text
//! χ_cmd = gc_course(pos, n) − χ∞ · (2/π) · atan(k_path · cross_track)
//! ```
//!
//! This is the spherical twin of the flat-earth `line_course` field — same
//! sign convention (positive cross-track = right of the path → steer left),
//! same `χ∞` / `k_path` tuning — but the path is a geodesic, so a long leg
//! follows the curve of the planet instead of cutting the chord. Capture uses
//! the **great-circle (arc) distance** to the active waypoint; the commanded
//! altitude is the active waypoint's altitude above the surface.

use crate::guidance::Waypoint;
use core::f64::consts::{FRAC_PI_2, PI};
use fsim_control::FixedWingSetpoint;
use fsim_core::planet::{altitude_of, gc_course, gc_cross_track, gc_distance, gc_normal};
use fsim_core::{Real, Vec3};

/// What to do once the *last* waypoint is captured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TerminalAction {
    /// Hold the course/altitude flown into the final waypoint, i.e. fly straight
    /// off the end of the route. Simplest and most robust: no new geometry, no
    /// risk of an un-capturable turn, the plant just keeps cruising. (The
    /// default — see the module RISKS for why orbit was *not* the default.)
    HoldCourse,
    /// Loiter: circle the final waypoint at `radius` \[m\] holding its altitude,
    /// `dir = +1` clockwise / `−1` counter-clockwise (viewed from above). A
    /// great-circle orbit vector field steers tangent to the circle with a radial
    /// correction, so the aircraft captures and holds the loiter from any
    /// approach — the realistic "hold here" behaviour for an autopilot that has
    /// reached the end of its route.
    Orbit { radius: Real, dir: Real },
}

/// Route-guidance tuning.
#[derive(Debug, Clone, Copy)]
pub struct FwGuidanceConfig {
    /// Constant commanded airspeed \[m/s\].
    pub airspeed: Real,
    /// **Great-circle** acceptance radius \[m\]: advance to the next waypoint once
    /// the surface (arc) distance to the active one drops below this. Must
    /// comfortably exceed the turn radius `R = Va²/(g·tan φ_max)` (≈110 m for a
    /// 25 m/s Aerosonde at 30°) or tight corners become un-capturable.
    pub accept_radius: Real,
    /// Approach angle far from the path `χ∞ ∈ (0, π/2]` for the vector field.
    pub chi_inf: Real,
    /// Cross-track convergence gain `k_path > 0`.
    pub k_path: Real,
    /// What to do after the final waypoint.
    pub terminal: TerminalAction,
}

impl Default for FwGuidanceConfig {
    /// Sized for the Aerosonde (25 m/s, φ_max = 30° → R ≈ 110 m). The 120 m
    /// accept radius is just above one turn radius so corners are capturable.
    /// `k_path` is the gentle 0.01 /m tuned with the course loop so the ground
    /// track converges smoothly instead of snaking (see the fixed-wing tuning note).
    fn default() -> Self {
        Self {
            airspeed: 25.0,
            accept_radius: 120.0,
            chi_inf: 0.9,
            k_path: 0.01,
            terminal: TerminalAction::HoldCourse,
        }
    }
}

/// A fixed-wing route follower on the sphere. Stateful only in the active-leg
/// index (and the latched terminal setpoint), so two runs from the same inputs
/// produce identical setpoints — no RNG, deterministic.
#[derive(Debug, Clone)]
pub struct FwGuidance {
    waypoints: Vec<Waypoint>,
    /// Index of the active *target* waypoint (the end of the current leg).
    idx: usize,
    /// Origin of the very first leg: the PCI position where the route began. For
    /// leg `i>0` the origin is `waypoints[i-1]`; for `i==0` it is this start.
    start: Vec3,
    cfg: FwGuidanceConfig,
    /// True once the final waypoint has been captured (the route is complete).
    completed: bool,
    /// Latched `HoldCourse` setpoint, frozen the gate the final waypoint is
    /// captured so it flies a fixed course (no re-aiming jitter). Unused by
    /// `Orbit`, which recomputes the loiter setpoint every gate.
    terminal_sp: Option<FixedWingSetpoint>,
}

impl FwGuidance {
    /// Build a follower. `start` is the aircraft's current PCI position, used as
    /// the origin of the first leg (so leg 0 is start → `waypoints[0]`). An empty
    /// list degrades to holding the start altitude on a North course.
    pub fn new(waypoints: Vec<Waypoint>, start: Vec3, cfg: FwGuidanceConfig) -> Self {
        Self {
            waypoints,
            idx: 0,
            start,
            cfg,
            completed: false,
            terminal_sp: None,
        }
    }

    /// Index of the active waypoint, or `None` if the route is empty.
    pub fn current_index(&self) -> Option<usize> {
        if self.waypoints.is_empty() {
            None
        } else {
            Some(self.idx)
        }
    }

    /// True once the active waypoint is the last one.
    pub fn on_final(&self) -> bool {
        !self.waypoints.is_empty() && self.idx + 1 >= self.waypoints.len()
    }

    /// True once the final waypoint has been captured and we are holding/loitering.
    pub fn is_complete(&self) -> bool {
        self.completed
    }

    /// Origin (start of the current leg) for the active index.
    fn leg_origin(&self) -> Vec3 {
        if self.idx == 0 {
            self.start
        } else {
            self.waypoints[self.idx - 1].position
        }
    }

    /// Great-circle course to fly along the leg `origin → target`, evaluated at
    /// the aircraft's current position `pos`, corrected by cross-track. Falls
    /// back to a direct bearing for a degenerate (zero-length / antipodal) leg.
    fn leg_course(&self, pos: Vec3, origin: Vec3, target: Vec3) -> Real {
        let n = gc_normal(origin, target);
        if n == Vec3::zeros() {
            // Degenerate leg: aim straight at the target along its own great circle.
            let n2 = gc_normal(pos, target);
            if n2 == Vec3::zeros() {
                0.0
            } else {
                gc_course(pos, n2)
            }
        } else {
            let path = gc_course(pos, n);
            let xt = gc_cross_track(pos, n);
            path - self.cfg.chi_inf * (2.0 / PI) * (self.cfg.k_path * xt).atan()
        }
    }

    /// Advance if inside the great-circle acceptance radius (one-way latch), then
    /// emit the setpoint for the (possibly newly-selected) active waypoint.
    ///
    /// `pos` is the truth/estimated PCI position. Capture uses the surface (arc)
    /// distance only; the active waypoint's altitude is commanded regardless of
    /// the current radius.
    pub fn update(&mut self, pos: Vec3) -> FixedWingSetpoint {
        // Empty route: hold start altitude, fly North.
        if self.waypoints.is_empty() {
            return FixedWingSetpoint {
                airspeed: self.cfg.airspeed,
                altitude: altitude_of(self.start),
                course: 0.0,
            };
        }

        // Great-circle capture: advance the latch if within accept_radius (arc).
        let active = self.waypoints[self.idx].position;
        if gc_distance(pos, active) < self.cfg.accept_radius {
            if self.idx + 1 < self.waypoints.len() {
                self.idx += 1;
            } else if !self.completed {
                self.completed = true;
                // HoldCourse freezes a fixed setpoint at the capture point; Orbit
                // recomputes its loiter every gate, so nothing is latched for it.
                if let TerminalAction::HoldCourse = self.cfg.terminal {
                    self.terminal_sp = Some(self.hold_setpoint(pos));
                }
            }
        }

        // Route complete: hold (latched) or loiter (recomputed each gate).
        if self.completed {
            return match self.cfg.terminal {
                TerminalAction::HoldCourse => {
                    self.terminal_sp.expect("HoldCourse latched on completion")
                }
                TerminalAction::Orbit { radius, dir } => self.orbit_setpoint(pos, radius, dir),
            };
        }

        let origin = self.leg_origin();
        let target = self.waypoints[self.idx].position;
        FixedWingSetpoint {
            airspeed: self.cfg.airspeed,
            altitude: self.leg_altitude(pos, origin, target),
            course: wrap_pi(self.leg_course(pos, origin, target)),
        }
    }

    /// Commanded altitude along the leg `origin → target`: a linear glide-slope
    /// from the origin's altitude to the target's by along-track fraction (so a
    /// climb/descent is flown gradually across the leg instead of stepping the
    /// moment the leg begins). Fraction is `1 − remaining/leg_length`, clamped.
    fn leg_altitude(&self, pos: Vec3, origin: Vec3, target: Vec3) -> Real {
        let origin_alt = altitude_of(origin);
        let target_alt = altitude_of(target);
        let leg_len = gc_distance(origin, target);
        if leg_len < 1.0 {
            return target_alt;
        }
        let f = (1.0 - gc_distance(pos, target) / leg_len).clamp(0.0, 1.0);
        origin_alt + (target_alt - origin_alt) * f
    }

    /// `HoldCourse` setpoint, latched at final capture: hold the final-leg
    /// great-circle course (frozen at the capture point) and the last altitude.
    fn hold_setpoint(&self, pos: Vec3) -> FixedWingSetpoint {
        let last = self.waypoints[self.idx].position;
        FixedWingSetpoint {
            airspeed: self.cfg.airspeed,
            altitude: altitude_of(last),
            course: wrap_pi(self.leg_course(pos, self.leg_origin(), last)),
        }
    }

    /// `Orbit` loiter setpoint: a great-circle orbit vector field around the final
    /// waypoint. With `λ` the bearing **to** the centre, the commanded course is
    /// the tangent `λ + dir·π/2` corrected by `−dir·atan(k·(d−radius))`: outside
    /// the circle (`d>radius`) it bends toward the centre, inside it bends away,
    /// so the aircraft captures and holds the loiter from any approach. Altitude
    /// is the centre's.
    fn orbit_setpoint(&self, pos: Vec3, radius: Real, dir: Real) -> FixedWingSetpoint {
        let center = self.waypoints[self.idx].position;
        let n = gc_normal(pos, center);
        let lambda = if n == Vec3::zeros() {
            0.0
        } else {
            gc_course(pos, n) // bearing toward the centre
        };
        let d = gc_distance(pos, center);
        let k = 1.0 / radius.max(1.0);
        let chi = lambda + dir.signum() * (FRAC_PI_2 - (k * (d - radius)).atan());
        FixedWingSetpoint {
            airspeed: self.cfg.airspeed,
            altitude: altitude_of(center),
            course: wrap_pi(chi),
        }
    }
}

/// Wrap an angle to `(−π, π]` — same definition as the autopilot's `wrap_pi`.
fn wrap_pi(x: Real) -> Real {
    x.sin().atan2(x.cos())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_2;

    fn cfg() -> FwGuidanceConfig {
        FwGuidanceConfig::default()
    }

    // A craft North of a due-East equatorial leg (left of the East heading) is
    // commanded to steer right (course > π/2), back onto the path. Pure law.
    #[test]
    fn course_sign_on_sphere() {
        let start = Waypoint::geodetic(0.0, 0.0, 400.0).position;
        let wps = vec![Waypoint::geodetic(0.0, 0.06, 400.0)]; // due-East leg
        let mut g = FwGuidance::new(wps, start, cfg());
        // ~30 km... no — small angles: lat 0.005 rad ≈ 32 m North, lon 0.03 mid-leg.
        let craft = Waypoint::geodetic(0.005, 0.03, 400.0).position;
        let sp = g.update(craft);
        assert!(
            sp.course > FRAC_PI_2,
            "north of an east leg should steer right (course>π/2): {}",
            sp.course
        );
        assert!((sp.altitude - 400.0).abs() < 1e-6);
        assert!((sp.airspeed - 25.0).abs() < 1e-12);
    }

    // Reaching the active waypoint advances the index; the final capture latches.
    #[test]
    fn advances_and_latches_terminal() {
        let start = Waypoint::geodetic(0.0, 0.0, 400.0).position;
        let wp0 = Waypoint::geodetic(0.02, 0.0, 400.0);
        let wp1 = Waypoint::geodetic(0.02, 0.02, 400.0);
        let mut g = FwGuidance::new(vec![wp0, wp1], start, cfg());
        assert_eq!(g.current_index(), Some(0));
        // Sit at wp0 → advance to wp1.
        g.update(wp0.position);
        assert_eq!(g.current_index(), Some(1));
        assert!(!g.is_complete());
        // Sit at wp1 (final) → latch HoldCourse.
        g.update(wp1.position);
        assert!(g.is_complete());
        assert!(g.on_final());
    }

    // HoldCourse keeps a fixed setpoint after capture (no jitter).
    #[test]
    fn terminal_holds_a_fixed_setpoint() {
        let start = Waypoint::geodetic(0.0, 0.0, 300.0).position;
        let wps = vec![Waypoint::geodetic(0.0, 0.05, 300.0)]; // East leg
        let mut g = FwGuidance::new(wps, start, cfg());
        g.update(Waypoint::geodetic(0.0, 0.05, 300.0).position); // capture final
        let a = g.update(Waypoint::geodetic(0.001, 0.06, 300.0).position);
        let b = g.update(Waypoint::geodetic(-0.002, 0.09, 300.0).position);
        assert_eq!(a, b, "terminal setpoint should be latched/constant");
        assert!(
            (a.course - FRAC_PI_2).abs() < 0.05,
            "should hold ~East: {}",
            a.course
        );
    }

    // Along-leg altitude interpolation: a climbing leg ramps the commanded
    // altitude from the origin's to the target's across the leg, not as a step.
    #[test]
    fn leg_altitude_ramps_across_the_leg() {
        use fsim_core::planet::PLANET_RADIUS;
        let start = Waypoint::geodetic(0.0, 0.0, 100.0).position;
        let wp = Waypoint::geodetic(2000.0 / PLANET_RADIUS, 0.0, 200.0); // 2 km N, climb 100→200
        let mut g = FwGuidance::new(vec![wp], start, cfg());
        // At the start of the leg: near the origin altitude (not the target's).
        let s0 = g.update(start);
        assert!(
            (s0.altitude - 100.0).abs() < 5.0,
            "start ~100: {}",
            s0.altitude
        );
        // Halfway along: about midway in altitude.
        let mid = Waypoint::geodetic(1000.0 / PLANET_RADIUS, 0.0, 0.0).position;
        let sm = g.update(mid);
        assert!(
            (sm.altitude - 150.0).abs() < 12.0,
            "mid ~150: {}",
            sm.altitude
        );
    }

    // Orbit terminal: after capturing the final waypoint it loiters — holding the
    // fix's altitude and commanding a course tangent to the circle.
    #[test]
    fn orbit_terminal_loiters_tangent() {
        use fsim_core::planet::PLANET_RADIUS;
        let cfg = FwGuidanceConfig {
            terminal: TerminalAction::Orbit {
                radius: 150.0,
                dir: 1.0,
            },
            ..FwGuidanceConfig::default()
        };
        let center = Waypoint::geodetic(500.0 / PLANET_RADIUS, 0.0, 200.0);
        let start = Waypoint::geodetic(0.0, 0.0, 200.0).position;
        let mut g = FwGuidance::new(vec![center], start, cfg);
        g.update(center.position); // capture the final waypoint
        assert!(g.is_complete());
        // On the circle, East of the centre: tangent course, centre's altitude.
        let on = Waypoint::geodetic(500.0 / PLANET_RADIUS, 150.0 / PLANET_RADIUS, 200.0).position;
        let sp = g.update(on);
        assert!(
            (sp.altitude - 200.0).abs() < 1.0,
            "orbit holds the centre altitude: {}",
            sp.altitude
        );
        let n = gc_normal(on, center.position);
        let off = wrap_pi(sp.course - gc_course(on, n)).abs();
        assert!(
            (off - FRAC_PI_2).abs() < 0.25,
            "course should be ~tangent (±π/2 off the bearing to centre): {off}"
        );
    }

    // An empty route degrades gracefully: hold start altitude, fly North.
    #[test]
    fn empty_route_holds_start() {
        let start = Waypoint::geodetic(0.1, -0.2, 150.0).position;
        let mut g = FwGuidance::new(Vec::new(), start, cfg());
        assert_eq!(g.current_index(), None);
        let sp = g.update(Waypoint::geodetic(0.1, -0.2, 160.0).position);
        assert!((sp.altitude - 150.0).abs() < 1e-6);
        assert!(sp.course.abs() < 1e-12);
    }
}
