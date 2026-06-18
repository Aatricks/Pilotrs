//! Fixed-wing route-following guidance (M6): walks a list of NED [`Waypoint`]s,
//! emitting a [`FixedWingSetpoint`] (airspeed / altitude / course) each control
//! gate and advancing to the next waypoint once inside a *horizontal*
//! acceptance radius. Like the quad [`Guidance`](crate::Guidance), the waypoint
//! *list* lives here (std/heap) so the `fsim-control` math stays no_std-clean.
//!
//! ## Guidance law (all signs derived in our NED frame: x North, y East, z Down)
//!
//! Course comes from straight-line vector-field guidance
//! ([`line_course`](crate::line_course)) along the active *leg* from the
//! previous waypoint (the leg origin) to the active waypoint, with
//! `path_course = atan2(dE, dN) = atan2(dy, dx)` — the same `atan2(y, x)`
//! convention as the course `χ` itself. Altitude is the active waypoint's
//! altitude (`-z`); airspeed is constant from config. See [`cross_track`] /
//! [`line_course`] for the sign of the cross-track term (positive = right of the
//! path direction → command a course to the left, back onto the path).
//!
//! The fixed-wing reuses the quad's [`Waypoint`] type; only the horizontal
//! `(x, y)` and the altitude `-z` are used (the `yaw` field is ignored).

use crate::fixedwing::line_course;
use crate::guidance::Waypoint;
use fsim_control::FixedWingSetpoint;
use fsim_core::{Real, Vec3};

/// What to do once the *last* waypoint is captured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalAction {
    /// Hold the course/altitude flown into the final waypoint, i.e. fly straight
    /// off the end of the route. Simplest and most robust: no new geometry, no
    /// risk of an un-capturable turn, the plant just keeps cruising. (Chosen
    /// default — see the module RISKS for why orbit was *not* the default.) The
    /// enum exists so an `Orbit { center, radius, dir }` variant can be added
    /// later without touching the leg logic.
    HoldCourse,
}

/// Route-guidance tuning.
#[derive(Debug, Clone, Copy)]
pub struct FwGuidanceConfig {
    /// Constant commanded airspeed \[m/s\].
    pub airspeed: Real,
    /// **Horizontal** acceptance radius \[m\]: advance to the next waypoint once
    /// the horizontal distance to the active one drops below this. Must
    /// comfortably exceed the turn radius `R = Va²/(g·tan φ_max)` (≈110 m for a
    /// 25 m/s Aerosonde at 30°) or tight corners become un-capturable.
    pub accept_radius: Real,
    /// Approach angle far from the path `χ∞ ∈ (0, π/2]` for [`line_course`].
    pub chi_inf: Real,
    /// Cross-track convergence gain `k_path > 0` for [`line_course`].
    pub k_path: Real,
    /// What to do after the final waypoint.
    pub terminal: TerminalAction,
}

impl Default for FwGuidanceConfig {
    /// Sized for the Aerosonde (25 m/s, φ_max = 30° → R ≈ 110 m). The 120 m
    /// accept radius is just above one turn radius so corners are capturable;
    /// `chi_inf` / `k_path` match the values exercised by the existing
    /// `straight_line_guidance_converges` test (0.9 rad, 0.05 /m).
    fn default() -> Self {
        Self {
            airspeed: 25.0,
            accept_radius: 120.0,
            chi_inf: 0.9,
            k_path: 0.05,
            terminal: TerminalAction::HoldCourse,
        }
    }
}

/// A fixed-wing route follower. Stateful only in the active-leg index (and the
/// latched terminal setpoint), so two runs from the same inputs produce
/// identical setpoints — no RNG, deterministic.
#[derive(Debug, Clone)]
pub struct FwGuidance {
    waypoints: Vec<Waypoint>,
    /// Index of the active *target* waypoint (the end of the current leg).
    idx: usize,
    /// Origin of the very first leg: the position where the route began. For
    /// leg `i>0` the origin is `waypoints[i-1]`; for `i==0` it is this start.
    start: Vec3,
    cfg: FwGuidanceConfig,
    /// Latched terminal setpoint, frozen the first gate after the final capture
    /// so `HoldCourse` flies a fixed straight line (no further re-aiming jitter).
    terminal_sp: Option<FixedWingSetpoint>,
}

impl FwGuidance {
    /// Build a follower. `start` is the aircraft's current NED position, used as
    /// the origin of the first leg (so leg 0 is start → `waypoints[0]`). An empty
    /// list degrades to holding the start altitude on a North course.
    pub fn new(waypoints: Vec<Waypoint>, start: Vec3, cfg: FwGuidanceConfig) -> Self {
        Self {
            waypoints,
            idx: 0,
            start,
            cfg,
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

    /// True once the final waypoint has been captured and we are holding.
    pub fn is_complete(&self) -> bool {
        self.terminal_sp.is_some()
    }

    /// Origin (start of the current leg) for the active index.
    fn leg_origin(&self) -> Vec3 {
        if self.idx == 0 {
            self.start
        } else {
            self.waypoints[self.idx - 1].position
        }
    }

    /// Advance if inside the *horizontal* acceptance radius (one-way latch), then
    /// emit the setpoint for the (possibly newly-selected) active waypoint.
    ///
    /// `pos` is the truth/estimated NED position. Only `(x, y)` is used for
    /// capture; the active waypoint's altitude is commanded regardless of `z`.
    pub fn update(&mut self, pos: Vec3) -> FixedWingSetpoint {
        // Empty route: hold start altitude, fly North.
        if self.waypoints.is_empty() {
            return FixedWingSetpoint {
                airspeed: self.cfg.airspeed,
                altitude: -self.start.z,
                course: 0.0,
            };
        }

        // Horizontal capture: advance the latch if within accept_radius (x,y only).
        let active = self.waypoints[self.idx].position;
        let horiz = (pos.x - active.x).hypot(pos.y - active.y);
        if horiz < self.cfg.accept_radius {
            if self.idx + 1 < self.waypoints.len() {
                self.idx += 1;
            } else if self.terminal_sp.is_none() {
                // Final waypoint captured: latch the terminal action.
                self.terminal_sp = Some(self.terminal_setpoint(pos));
            }
        }

        // Once a terminal setpoint is latched, keep flying it unchanged.
        if let Some(sp) = self.terminal_sp {
            return sp;
        }

        // Active leg: origin → waypoints[idx]. Vector-field course law.
        let origin = self.leg_origin();
        let wp = self.waypoints[self.idx].position;
        let (dn, de) = (wp.x - origin.x, wp.y - origin.y); // North, East deltas

        // Degenerate (zero-length) leg: aim straight at the waypoint instead.
        let course = if dn.hypot(de) < 1e-3 {
            let (gn, ge) = (wp.x - pos.x, wp.y - pos.y);
            if gn.hypot(ge) < 1e-6 {
                0.0 // on top of it: hold North (will capture next gate anyway)
            } else {
                ge.atan2(gn) // atan2(East, North) = χ
            }
        } else {
            let path_course = de.atan2(dn); // atan2(dy, dx) — leg's NED course
            line_course(pos, origin, path_course, self.cfg.chi_inf, self.cfg.k_path)
        };

        FixedWingSetpoint {
            airspeed: self.cfg.airspeed,
            altitude: -wp.z, // active waypoint altitude
            course: wrap_pi(course),
        }
    }

    /// Setpoint to latch when the final waypoint is captured.
    fn terminal_setpoint(&self, pos: Vec3) -> FixedWingSetpoint {
        let last = self.waypoints[self.idx].position;
        match self.cfg.terminal {
            TerminalAction::HoldCourse => {
                // Hold the final-leg course (origin → last) and fly straight off
                // the end. Fall back to current bearing for a 1-point route.
                let origin = self.leg_origin();
                let (dn, de) = (last.x - origin.x, last.y - origin.y);
                let course = if dn.hypot(de) < 1e-3 {
                    (last.y - pos.y).atan2(last.x - pos.x)
                } else {
                    de.atan2(dn)
                };
                FixedWingSetpoint {
                    airspeed: self.cfg.airspeed,
                    altitude: -last.z,
                    course: wrap_pi(course),
                }
            }
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

    // A craft North of a due-East leg (left of the East heading, e_py < 0) is
    // commanded to steer right (course > π/2), back onto the path. Pure law.
    #[test]
    fn course_sign_in_ned() {
        let wps = vec![Waypoint::ne_alt(0.0, 400.0, 120.0)]; // due-East leg from origin
        let mut g = FwGuidance::new(wps, Vec3::zeros(), cfg());
        let sp = g.update(Vec3::new(30.0, 50.0, -120.0)); // 30 m North of the East leg
        assert!(
            sp.course > FRAC_PI_2,
            "north of an east leg should steer right (course>π/2): {}",
            sp.course
        );
        assert!((sp.altitude - 120.0).abs() < 1e-12);
        assert!((sp.airspeed - 25.0).abs() < 1e-12);
    }

    // Reaching the active waypoint advances the index; the final capture latches.
    #[test]
    fn advances_and_latches_terminal() {
        let wps = vec![
            Waypoint::ne_alt(400.0, 0.0, 120.0),
            Waypoint::ne_alt(400.0, 400.0, 120.0),
        ];
        let mut g = FwGuidance::new(wps, Vec3::new(0.0, 0.0, -120.0), cfg());
        assert_eq!(g.current_index(), Some(0));
        // Sit at wp0 → advance to wp1.
        g.update(Vec3::new(400.0, 0.0, -120.0));
        assert_eq!(g.current_index(), Some(1));
        assert!(!g.is_complete());
        // Sit at wp1 (final) → latch HoldCourse.
        g.update(Vec3::new(400.0, 400.0, -120.0));
        assert!(g.is_complete());
        assert!(g.on_final());
    }

    // HoldCourse keeps a fixed setpoint after capture (no jitter).
    #[test]
    fn terminal_holds_a_fixed_setpoint() {
        let wps = vec![Waypoint::ne_alt(0.0, 400.0, 100.0)]; // East leg
        let mut g = FwGuidance::new(wps, Vec3::zeros(), cfg());
        g.update(Vec3::new(0.0, 400.0, -100.0)); // capture final
        let a = g.update(Vec3::new(10.0, 500.0, -100.0));
        let b = g.update(Vec3::new(-20.0, 900.0, -100.0));
        assert_eq!(a, b, "terminal setpoint should be latched/constant");
        assert!(
            (a.course - FRAC_PI_2).abs() < 1e-9,
            "should hold East: {}",
            a.course
        );
    }

    // An empty route degrades gracefully: hold start altitude, fly North.
    #[test]
    fn empty_route_holds_start() {
        let mut g = FwGuidance::new(Vec::new(), Vec3::new(0.0, 0.0, -150.0), cfg());
        assert_eq!(g.current_index(), None);
        let sp = g.update(Vec3::new(10.0, 10.0, -140.0));
        assert!((sp.altitude - 150.0).abs() < 1e-12);
        assert!(sp.course.abs() < 1e-12);
    }
}
