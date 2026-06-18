//! Waypoint guidance: walks a list of NED waypoints, emitting a
//! [`GuidanceTarget`] for the position controller and advancing to the next
//! waypoint once inside an acceptance radius. The waypoint *list* lives here
//! (std/heap) so the `fsim-control` math stays no_std-clean.

use fsim_control::GuidanceTarget;
use fsim_core::{Real, Vec3};

/// A mission waypoint: a NED position to reach and the heading to hold there.
#[derive(Debug, Clone, Copy)]
pub struct Waypoint {
    pub position: Vec3,
    pub yaw: Real,
}

impl Waypoint {
    pub fn new(position: Vec3, yaw: Real) -> Self {
        Self { position, yaw }
    }

    /// Build from North / East / altitude-up (stored as NED `z = -altitude`),
    /// with zero yaw. Convenient for routes drawn on a map (the fixed-wing
    /// guidance ignores `yaw`; the quad holds it).
    pub fn ne_alt(north: Real, east: Real, altitude: Real) -> Self {
        Self {
            position: Vec3::new(north, east, -altitude),
            yaw: 0.0,
        }
    }
}

/// Guidance tuning.
#[derive(Debug, Clone, Copy)]
pub struct GuidanceConfig {
    /// Switch to the next waypoint once within this distance \[m\].
    pub accept_radius: Real,
    /// Speed to command toward the active waypoint \[m/s\].
    pub cruise_speed: Real,
}

impl Default for GuidanceConfig {
    fn default() -> Self {
        Self {
            accept_radius: 0.5,
            cruise_speed: 3.0,
        }
    }
}

/// A waypoint follower.
#[derive(Debug, Clone)]
pub struct Guidance {
    waypoints: Vec<Waypoint>,
    idx: usize,
    cfg: GuidanceConfig,
}

impl Guidance {
    /// Build a guidance from a non-empty waypoint list (an empty list degrades
    /// to holding the origin).
    pub fn new(waypoints: Vec<Waypoint>, cfg: GuidanceConfig) -> Self {
        Self {
            waypoints,
            idx: 0,
            cfg,
        }
    }

    /// Index of the active waypoint.
    pub fn current_index(&self) -> usize {
        self.idx
    }

    /// True once the last waypoint is the active one.
    pub fn on_final(&self) -> bool {
        self.idx + 1 >= self.waypoints.len()
    }

    /// Advance if within the acceptance radius (one-way latch), then emit the
    /// target for the (possibly newly-selected) active waypoint this same tick.
    pub fn update(&mut self, p_est: Vec3) -> GuidanceTarget {
        if self.waypoints.is_empty() {
            return GuidanceTarget::hold(Vec3::zeros(), 0.0);
        }
        let reached = (self.waypoints[self.idx].position - p_est).norm() < self.cfg.accept_radius;
        if reached && self.idx + 1 < self.waypoints.len() {
            self.idx += 1;
        }
        let wp = self.waypoints[self.idx];
        let to_wp = wp.position - p_est;
        let dist = to_wp.norm();
        let final_hold = self.on_final() && dist < self.cfg.accept_radius;
        let velocity_ff = if final_hold || dist < 1e-6 {
            Vec3::zeros()
        } else {
            to_wp / dist * self.cfg.cruise_speed
        };
        GuidanceTarget {
            position: wp.position,
            velocity_ff,
            yaw: wp.yaw,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mission() -> Guidance {
        let wps = vec![
            Waypoint::new(Vec3::new(0.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 0.0, -2.0), 0.0),
            Waypoint::new(Vec3::new(5.0, 5.0, -2.0), 0.0),
        ];
        Guidance::new(wps, GuidanceConfig::default())
    }

    #[test]
    fn advances_within_radius_but_not_past_last() {
        let mut g = mission();
        assert_eq!(g.current_index(), 0);
        // At wp0 -> advance to 1.
        g.update(Vec3::new(0.0, 0.0, -2.0));
        assert_eq!(g.current_index(), 1);
        // At wp1 -> advance to 2 (last).
        g.update(Vec3::new(5.0, 0.0, -2.0));
        assert_eq!(g.current_index(), 2);
        // At wp2 (last) -> stays.
        g.update(Vec3::new(5.0, 5.0, -2.0));
        assert_eq!(g.current_index(), 2);
        assert!(g.on_final());
    }

    #[test]
    fn feedforward_points_at_active_waypoint() {
        let mut g = mission();
        let tgt = g.update(Vec3::new(0.0, 0.0, -2.0)); // reaches wp0, targets wp1 (+x)
        assert_eq!(g.current_index(), 1);
        assert!(tgt.velocity_ff.x > 0.0, "ff not toward wp1");
        assert!(
            (tgt.velocity_ff.norm() - 3.0).abs() < 1e-9,
            "ff not cruise speed"
        );
    }

    #[test]
    fn final_hold_zeroes_feedforward() {
        let mut g = mission();
        g.update(Vec3::new(0.0, 0.0, -2.0));
        g.update(Vec3::new(5.0, 0.0, -2.0));
        let tgt = g.update(Vec3::new(5.0, 5.0, -2.0)); // at final wp
        assert!(
            tgt.velocity_ff.norm() < 1e-9,
            "ff should be zero at final hold"
        );
    }
}
