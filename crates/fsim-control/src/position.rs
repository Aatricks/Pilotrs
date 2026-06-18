//! Outer position/velocity control for a quadrotor.
//!
//! The cascade is: guidance target → **position loop** (P) → desired velocity →
//! **velocity loop** (PID) → desired world acceleration → **accel→attitude**
//! inversion → an attitude/thrust [`Setpoint`] for the existing inner
//! attitude→rate [`CascadedPid`](crate::CascadedPid).
//!
//! ## The accel→attitude inversion (load-bearing, NED/FRD)
//!
//! The rotor force in world is `−T·R·e_z` (thrust along body −z). To achieve
//! `m·a_des` we need `−T·R·e_z = m(a_des − g_w)`, so the required rotor force is
//! `f = m(a_des − g_w)`, the thrust magnitude `T = ‖f‖`, and the desired body-z
//! axis in world is `zb = −f/‖f‖ = normalize(g_w − a_des)`. At hover
//! (`a_des = 0`): `zb = (0,0,1)` (level) and `T = m·g`. Gravity enters **once**,
//! here — never add it to `a_des` (the classic double-gravity bug).

use crate::Pid;
use fsim_core::{gravity_world, EstState, Real, Setpoint, Vec3, GRAVITY};
use nalgebra::{Matrix3, Rotation3, Unit, UnitQuaternion};
use num_traits::Float;

/// What guidance hands the position controller each tick.
#[derive(Debug, Clone, Copy)]
pub struct GuidanceTarget {
    /// Target position, NED world \[m\].
    pub position: Vec3,
    /// Velocity feedforward, NED world \[m/s\] (0 when holding).
    pub velocity_ff: Vec3,
    /// Desired heading \[rad\].
    pub yaw: Real,
}

impl GuidanceTarget {
    /// Hold a position with zero feedforward and the given heading.
    pub fn hold(position: Vec3, yaw: Real) -> Self {
        Self {
            position,
            velocity_ff: Vec3::zeros(),
            yaw,
        }
    }
}

/// Gains and limits for the [`PositionController`].
#[derive(Debug, Clone, Copy)]
pub struct PositionConfig {
    /// Vehicle mass \[kg\].
    pub mass: Real,
    /// Position-loop P gain per axis \[1/s\].
    pub kp_pos: Vec3,
    /// Max horizontal speed \[m/s\].
    pub v_max_xy: Real,
    /// Max climb (−z) / descent (+z) speed \[m/s\].
    pub v_max_up: Real,
    pub v_max_down: Real,
    /// Velocity-loop PID gains per axis.
    pub kp_vel: Vec3,
    pub ki_vel: Vec3,
    pub kd_vel: Vec3,
    /// Velocity integral term clamp per axis \[m/s²\].
    pub vel_i_max: Vec3,
    /// Max commanded horizontal / vertical acceleration \[m/s²\].
    pub a_max_xy: Real,
    pub a_max_z: Real,
    /// Max tilt angle \[rad\].
    pub tilt_max: Real,
    /// Thrust limits \[N\].
    pub t_min: Real,
    pub t_max: Real,
}

impl PositionConfig {
    /// Tuned defaults for the 250-class quad (mass 0.5 kg).
    pub fn quad_250() -> Self {
        Self {
            mass: 0.5,
            kp_pos: Vec3::new(1.0, 1.0, 1.5),
            v_max_xy: 5.0,
            v_max_up: 3.0,
            v_max_down: 2.0,
            kp_vel: Vec3::new(3.0, 3.0, 4.0),
            ki_vel: Vec3::new(0.5, 0.5, 1.0),
            kd_vel: Vec3::zeros(),
            vel_i_max: Vec3::new(2.0, 2.0, 3.0),
            a_max_xy: 6.0,
            a_max_z: 8.0,
            tilt_max: 0.611, // 35°
            t_min: 0.49,
            t_max: 16.0,
        }
    }
}

/// Scale the horizontal `(x,y)` part of `v` so its norm is at most `max`.
fn clamp_horizontal(v: &mut Vec3, max: Real) {
    let h = Float::sqrt(v.x * v.x + v.y * v.y);
    if h > max && h > 1e-9 {
        let s = max / h;
        v.x *= s;
        v.y *= s;
    }
}

/// Limit the tilt of a desired body-z axis (world) to `tilt_max` from vertical.
fn clamp_tilt(zb: &mut Vec3, tilt_max: Real) {
    let up = Vec3::new(0.0, 0.0, 1.0);
    let theta = Float::acos(zb.z.clamp(-1.0, 1.0));
    if theta > tilt_max {
        let axis = up.cross(zb);
        if axis.norm() > 1e-6 {
            let axis = Unit::new_normalize(axis);
            *zb = UnitQuaternion::from_axis_angle(&axis, tilt_max) * up;
        }
    }
}

/// Map a desired world acceleration + heading to an attitude/thrust setpoint.
/// Exposed for unit testing the inversion in isolation.
pub fn accel_to_setpoint(a_des: Vec3, yaw: Real, cfg: &PositionConfig) -> Setpoint {
    let f = (a_des - gravity_world()) * cfg.mass; // required rotor force, world
    let t_mag = f.norm();
    let mut zb = if t_mag > 0.1 * cfg.mass * GRAVITY {
        -f / t_mag
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    clamp_tilt(&mut zb, cfg.tilt_max);

    // Desired attitude from zb + yaw (TRIAD); yaw convention matches fsim-core
    // (90° yaw maps body +x to world +y).
    let xc = Vec3::new(Float::cos(yaw), Float::sin(yaw), 0.0);
    let yb_raw = zb.cross(&xc);
    let yb = if yb_raw.norm() > 1e-4 {
        yb_raw.normalize()
    } else {
        // Degenerate (only reachable past the tilt clamp): pick any horizontal.
        Vec3::new(-Float::sin(yaw), Float::cos(yaw), 0.0)
    };
    let xb = yb.cross(&zb);
    let r_des = Matrix3::from_columns(&[xb, yb, zb]);
    let attitude = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r_des));

    Setpoint {
        attitude,
        thrust: t_mag.clamp(cfg.t_min, cfg.t_max),
    }
}

/// Position + velocity controller. Emits an attitude/thrust [`Setpoint`] for the
/// inner attitude→rate cascade (it never touches `est.angular_rate`).
#[derive(Debug, Clone)]
pub struct PositionController {
    cfg: PositionConfig,
    vel_pid: [Pid; 3],
}

impl PositionController {
    pub fn new(cfg: PositionConfig) -> Self {
        let a_max = [cfg.a_max_xy, cfg.a_max_xy, cfg.a_max_z];
        let vel_pid = [
            Pid::new(
                cfg.kp_vel.x,
                cfg.ki_vel.x,
                cfg.kd_vel.x,
                cfg.vel_i_max.x,
                a_max[0],
            ),
            Pid::new(
                cfg.kp_vel.y,
                cfg.ki_vel.y,
                cfg.kd_vel.y,
                cfg.vel_i_max.y,
                a_max[1],
            ),
            Pid::new(
                cfg.kp_vel.z,
                cfg.ki_vel.z,
                cfg.kd_vel.z,
                cfg.vel_i_max.z,
                a_max[2],
            ),
        ];
        Self { cfg, vel_pid }
    }

    /// Controller tuned to the default 250-class quad.
    pub fn quad_250() -> Self {
        Self::new(PositionConfig::quad_250())
    }

    /// Forget velocity-loop integral state (call on arm / mode switch).
    pub fn reset(&mut self) {
        for p in &mut self.vel_pid {
            p.reset();
        }
    }

    /// Compute the attitude/thrust setpoint from the estimate + guidance target.
    pub fn step(&mut self, est: &EstState, tgt: &GuidanceTarget, dt: Real) -> Setpoint {
        // Position loop (P) → desired velocity.
        let e_p = tgt.position - est.position;
        let mut v_des = e_p.component_mul(&self.cfg.kp_pos) + tgt.velocity_ff;
        clamp_horizontal(&mut v_des, self.cfg.v_max_xy);
        v_des.z = v_des.z.clamp(-self.cfg.v_max_up, self.cfg.v_max_down);

        // Velocity loop (PID) → desired world acceleration.
        let e_v = v_des - est.velocity;
        let mut a_des = Vec3::new(
            self.vel_pid[0].step(e_v.x, dt),
            self.vel_pid[1].step(e_v.y, dt),
            self.vel_pid[2].step(e_v.z, dt),
        );
        clamp_horizontal(&mut a_des, self.cfg.a_max_xy);
        a_des.z = a_des.z.clamp(-self.cfg.a_max_z, self.cfg.a_max_z);

        accel_to_setpoint(a_des, tgt.yaw, &self.cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::Quat;

    fn cfg() -> PositionConfig {
        PositionConfig::quad_250()
    }

    #[test]
    fn hover_command_is_level_at_mg() {
        let sp = accel_to_setpoint(Vec3::zeros(), 0.4, &cfg());
        assert!(
            (sp.thrust - 0.5 * GRAVITY).abs() < 1e-9,
            "thrust={}",
            sp.thrust
        );
        let (roll, pitch, yaw) = sp.attitude.euler_angles();
        assert!(roll.abs() < 1e-9 && pitch.abs() < 1e-9, "not level");
        assert!((yaw - 0.4).abs() < 1e-9, "yaw={yaw}");
    }

    #[test]
    fn forward_accel_tilts_thrust_north() {
        // a_des = +x (North). The world rotor force -T*R*e_z must have a +x
        // component (convention-free check).
        let a_des = Vec3::new(2.0, 0.0, 0.0);
        let sp = accel_to_setpoint(a_des, 0.0, &cfg());
        let world_thrust = sp.attitude * Vec3::new(0.0, 0.0, -sp.thrust);
        assert!(
            world_thrust.x > 0.0,
            "thrust not tilted North: {}",
            world_thrust.x
        );
    }

    #[test]
    fn climb_command_stays_level_higher_thrust() {
        // a_des = (0,0,-1) is upward (NED). Stays level, T = m(g+1).
        let sp = accel_to_setpoint(Vec3::new(0.0, 0.0, -1.0), 0.0, &cfg());
        assert!(sp.attitude.angle() < 1e-9, "not level");
        assert!((sp.thrust - 0.5 * (GRAVITY + 1.0)).abs() < 1e-9);
    }

    #[test]
    fn excessive_accel_is_tilt_clamped() {
        let sp = accel_to_setpoint(Vec3::new(50.0, 0.0, 0.0), 0.0, &cfg());
        let zb = sp.attitude * Vec3::new(0.0, 0.0, 1.0); // body-z in world
        let tilt = zb.z.clamp(-1.0, 1.0).acos();
        assert!(
            (tilt - cfg().tilt_max).abs() < 1e-6,
            "tilt={tilt} != {}",
            cfg().tilt_max
        );
        assert!(sp.thrust.is_finite() && sp.thrust > 0.0);
    }

    #[test]
    fn round_trip_recovers_acceleration() {
        // Within clamps, the achieved accel (-T R e_z)/m + g_w == a_des.
        let a_des = Vec3::new(1.0, -1.5, -0.5);
        let sp = accel_to_setpoint(a_des, 0.3, &cfg());
        let world_thrust = sp.attitude * Vec3::new(0.0, 0.0, -sp.thrust);
        let achieved = world_thrust / cfg().mass + gravity_world();
        assert!(
            (achieved - a_des).norm() < 1e-9,
            "achieved={achieved:?} a_des={a_des:?}"
        );
    }

    #[test]
    fn degenerate_freefall_command_is_safe() {
        // a_des = g_w → required force 0 → fall back to level at t_min.
        let sp = accel_to_setpoint(gravity_world(), 0.0, &cfg());
        assert!(sp.thrust.is_finite());
        assert!(
            (sp.attitude * Vec3::new(0.0, 0.0, 1.0)).z > 0.99,
            "not level"
        );
    }

    #[test]
    fn position_hold_at_target_commands_hover() {
        // Estimate exactly at target, zero velocity → ~hover (level, ~mg).
        let mut pc = PositionController::quad_250();
        let est = EstState {
            position: Vec3::new(1.0, 2.0, -3.0),
            velocity: Vec3::zeros(),
            attitude: Quat::identity(),
            angular_rate: Vec3::zeros(),
        };
        let tgt = GuidanceTarget::hold(Vec3::new(1.0, 2.0, -3.0), 0.0);
        let sp = pc.step(&est, &tgt, 1e-3);
        assert!(sp.attitude.angle() < 1e-6, "not level at target");
        assert!((sp.thrust - 0.5 * GRAVITY).abs() < 1e-6);
    }

    #[test]
    fn position_error_commands_motion_toward_target() {
        // Target 5 m North of the estimate → tilt the thrust North.
        let mut pc = PositionController::quad_250();
        let est = EstState {
            position: Vec3::zeros(),
            velocity: Vec3::zeros(),
            attitude: Quat::identity(),
            angular_rate: Vec3::zeros(),
        };
        let tgt = GuidanceTarget::hold(Vec3::new(5.0, 0.0, 0.0), 0.0);
        let sp = pc.step(&est, &tgt, 1e-3);
        let world_thrust = sp.attitude * Vec3::new(0.0, 0.0, -sp.thrust);
        assert!(world_thrust.x > 0.0, "did not command North");
    }
}
