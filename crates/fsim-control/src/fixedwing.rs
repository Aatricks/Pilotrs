//! Fixed-wing autopilot (M6): a classic decoupled successive-loop-closure
//! controller producing the four [`FixedWingControls`]. It reuses the same
//! scalar [`Pid`](crate::Pid) as the quad cascade; inner-loop rate damping is
//! applied to the measured rate directly (no derivative kick on setpoint steps).
//!
//! ## Loop structure (each feedback sign derived for our FRD aero conventions)
//!
//! - **Lateral:** course χ → commanded bank φ_cmd (clamped); roll attitude →
//!   aileron (`+aileron` rolls right, so `δa = kp·(φ_cmd−φ) − kd·p`); a yaw
//!   damper drives the yaw rate toward the coordinated-turn rate.
//! - **Longitudinal:** altitude → commanded pitch θ_cmd (clamped); pitch
//!   attitude → elevator (`+elevator` pitches *down*, so `δe = kp·(θ−θ_cmd) +
//!   kd·q`); airspeed → throttle (around a trim feed-forward).
//!
//! The course/altitude/airspeed/pitch integrators absorb the steady trim
//! offsets, so cruise needs no exact feed-forward beyond the throttle.

use crate::Pid;
use fsim_core::{ControlLimits, EstState, FixedWingControls, Real, GRAVITY};
use num_traits::Float;

/// What the fixed-wing autopilot is asked to hold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedWingSetpoint {
    /// Commanded airspeed \[m/s\].
    pub airspeed: Real,
    /// Commanded altitude (`-z`) \[m\].
    pub altitude: Real,
    /// Commanded course χ \[rad\] (direction of travel over ground, NED).
    pub course: Real,
}

/// A fixed-wing autopilot: maps the estimate + setpoint to control surfaces.
/// (Separate from the quad [`Controller`](crate::Controller), whose command is
/// thrust + torque rather than four surfaces.)
pub trait FixedWingController: Send {
    fn step(&mut self, est: &EstState, sp: &FixedWingSetpoint, dt: Real) -> FixedWingControls;
}

/// Gains + limits for the [`FixedWingAutopilot`].
#[derive(Debug, Clone, Copy)]
pub struct FixedWingConfig {
    // Lateral.
    pub kp_phi: Real,
    pub kd_phi: Real,
    pub kr: Real,
    pub kp_chi: Real,
    pub ki_chi: Real,
    pub phi_max: Real,
    // Longitudinal.
    pub kp_theta: Real,
    pub ki_theta: Real,
    pub kd_theta: Real,
    pub kp_h: Real,
    pub ki_h: Real,
    pub theta_max: Real,
    pub kp_va: Real,
    pub ki_va: Real,
    /// Throttle feed-forward (set from the trim solution).
    pub trim_throttle: Real,
    /// Floor on airspeed used in the `1/Va` coordinated-turn term.
    pub va_min: Real,
    pub limits: ControlLimits,
}

impl FixedWingConfig {
    /// Starting gains for the Aerosonde at ~25 m/s. `trim_throttle` defaults to
    /// a placeholder; set it from the dynamics trim solver for best speed
    /// tracking (the airspeed integrator absorbs the rest).
    pub fn aerosonde() -> Self {
        Self {
            kp_phi: 1.2,
            kd_phi: 0.15,
            kr: 0.2,
            // Course loop: raised bandwidth (kp 0.8→1.4, ki 0.1→0.3) so the bank
            // tracks the commanded course crisply. A slow course loop under the
            // line-following vector field is what makes the ground track snake;
            // a faster, still-well-damped loop (ζ≈0.8) converges cleanly. See the
            // k_path note in fw_guidance.rs — the two are tuned together.
            kp_chi: 1.4,
            ki_chi: 0.3,
            phi_max: 0.52, // 30°
            kp_theta: 1.0,
            ki_theta: 0.3,
            kd_theta: 0.2,
            kp_h: 0.04,
            ki_h: 0.01,
            theta_max: 0.44, // 25°
            kp_va: 0.05,
            ki_va: 0.02,
            trim_throttle: 0.5,
            va_min: 1.0,
            limits: ControlLimits {
                surface_max: 0.4363,
                throttle: (0.0, 1.0),
            },
        }
    }
}

/// The fixed-wing autopilot. Holds the four outer/inner PID channels.
#[derive(Debug, Clone)]
pub struct FixedWingAutopilot {
    cfg: FixedWingConfig,
    course_pid: Pid, // χ error → φ_cmd
    alt_pid: Pid,    // altitude error → θ_cmd
    speed_pid: Pid,  // airspeed error → throttle delta
    pitch_pid: Pid,  // pitch error → elevator (P + I)
}

impl FixedWingAutopilot {
    pub fn new(cfg: FixedWingConfig) -> Self {
        let course_pid = Pid::new(cfg.kp_chi, cfg.ki_chi, 0.0, cfg.phi_max * 0.5, cfg.phi_max);
        let alt_pid = Pid::new(cfg.kp_h, cfg.ki_h, 0.0, cfg.theta_max * 0.5, cfg.theta_max);
        let speed_pid = Pid::new(cfg.kp_va, cfg.ki_va, 0.0, 0.5, 1.0);
        let sm = cfg.limits.surface_max;
        let pitch_pid = Pid::new(cfg.kp_theta, cfg.ki_theta, 0.0, sm, sm);
        Self {
            cfg,
            course_pid,
            alt_pid,
            speed_pid,
            pitch_pid,
        }
    }
}

/// Wrap an angle to `(−π, π]`.
fn wrap_pi(x: Real) -> Real {
    Float::atan2(Float::sin(x), Float::cos(x))
}

impl FixedWingController for FixedWingAutopilot {
    fn step(&mut self, est: &EstState, sp: &FixedWingSetpoint, dt: Real) -> FixedWingControls {
        let (phi, theta, _psi) = est.attitude.euler_angles();
        let (p, q, r) = (est.angular_rate.x, est.angular_rate.y, est.angular_rate.z);
        let va = est.velocity.norm();
        let va_s = Float::max(va, self.cfg.va_min);
        let chi = Float::atan2(est.velocity.y, est.velocity.x); // NED: x North, y East
        let h = -est.position.z;

        // --- LATERAL: course → bank → aileron, plus a coordinated yaw damper ---
        let phi_cmd = self.course_pid.step(wrap_pi(sp.course - chi), dt);
        let aileron = self.cfg.kp_phi * (phi_cmd - phi) - self.cfg.kd_phi * p;
        let r_coord = (GRAVITY / va_s) * Float::tan(phi) * Float::cos(theta);
        let rudder = self.cfg.kr * (r - r_coord);

        // --- LONGITUDINAL: altitude → pitch → elevator; airspeed → throttle ---
        let theta_cmd = self.alt_pid.step(sp.altitude - h, dt);
        let elevator = self.pitch_pid.step(theta - theta_cmd, dt) + self.cfg.kd_theta * q;
        let throttle = self.cfg.trim_throttle + self.speed_pid.step(sp.airspeed - va, dt);

        FixedWingControls {
            aileron,
            elevator,
            rudder,
            throttle,
        }
        .clamp(&self.cfg.limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::Vec3;
    use nalgebra::UnitQuaternion;

    fn ap() -> FixedWingAutopilot {
        FixedWingAutopilot::new(FixedWingConfig::aerosonde())
    }

    /// Level cruise estimate at 25 m/s North, at the given altitude.
    fn cruise(alt: Real) -> EstState {
        EstState {
            position: Vec3::new(0.0, 0.0, -alt),
            velocity: Vec3::new(25.0, 0.0, 0.0),
            attitude: UnitQuaternion::identity(),
            angular_rate: Vec3::zeros(),
        }
    }

    // T-RollLaw: a right-turn course command banks right; roll rate is damped.
    #[test]
    fn roll_law_banks_into_the_turn() {
        let mut c = ap();
        let mut e = cruise(100.0);
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: 0.5, // turn right
        };
        assert!(
            c.step(&e, &sp, 1e-2).aileron > 0.0,
            "+course → roll right (+aileron)"
        );

        // Pure roll-rate damping: rolling right with no bank command → −aileron.
        let mut c2 = ap();
        e.angular_rate = Vec3::new(1.0, 0.0, 0.0);
        let level = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: 0.0,
        };
        assert!(
            c2.step(&e, &level, 1e-2).aileron < 0.0,
            "roll rate should be damped"
        );
    }

    // T-CourseLaw: wrap-around takes the short way across ±π.
    #[test]
    fn course_law_wraps_short_way() {
        let mut c = ap();
        let mut e = cruise(100.0);
        e.velocity = Vec3::new(25.0 * (3.0_f64).cos(), 25.0 * (3.0_f64).sin(), 0.0); // χ≈3.0
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: -3.0, // short way is +0.28 rad (through π), i.e. turn right
        };
        assert!(
            c.step(&e, &sp, 1e-2).aileron > 0.0,
            "should turn the short way (right)"
        );
    }

    // T-PitchLaw: nose above the pitch target → elevator down; pitch rate damped.
    #[test]
    fn pitch_law_corrects_and_damps() {
        let mut c = ap();
        let mut e = cruise(0.0); // h=0 so altitude error is 0 → θ_cmd≈0
        e.attitude = UnitQuaternion::from_euler_angles(0.0, 0.1, 0.0); // θ=+0.1
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 0.0,
            course: 0.0,
        };
        assert!(
            c.step(&e, &sp, 1e-2).elevator > 0.0,
            "θ>θ_cmd → nose-down (+elevator)"
        );

        let mut c2 = ap();
        let mut e2 = cruise(0.0);
        e2.angular_rate = Vec3::new(0.0, 1.0, 0.0); // pitching up
        assert!(
            c2.step(&e2, &sp, 1e-2).elevator > 0.0,
            "pitch rate should be damped"
        );
    }

    // T-AltLaw: a climb command pitches the nose up (NED: h = -z).
    #[test]
    fn altitude_law_climbs_nose_up() {
        let mut c = ap();
        let e = cruise(100.0); // θ=0
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 150.0, // climb 50 m
            course: 0.0,
        };
        // θ_cmd > 0 (nose up) → pitch error (0 − θ_cmd) < 0 → elevator < 0 (nose up).
        assert!(
            c.step(&e, &sp, 1e-2).elevator < 0.0,
            "climb command → nose up"
        );
    }

    // T-SpeedLaw: below target airspeed adds throttle; above removes it.
    #[test]
    fn speed_law_throttles() {
        let mut c = ap();
        let mut e = cruise(100.0);
        e.velocity = Vec3::new(20.0, 0.0, 0.0); // slow
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: 0.0,
        };
        assert!(
            c.step(&e, &sp, 1e-2).throttle > c.cfg.trim_throttle,
            "slow → more throttle"
        );

        let mut c2 = ap();
        e.velocity = Vec3::new(30.0, 0.0, 0.0); // fast
        assert!(
            c2.step(&e, &sp, 1e-2).throttle < c2.cfg.trim_throttle,
            "fast → less throttle"
        );
    }

    // T-RudLaw: yawing faster than the coordinated rate → +rudder, which (Cndr<0)
    // is a left-yaw moment that opposes the excess.
    #[test]
    fn rudder_damps_toward_coordination() {
        let mut c = ap();
        let mut e = cruise(100.0); // φ=0 → r_coord=0
        e.angular_rate = Vec3::new(0.0, 0.0, 0.5); // yawing right
        let sp = FixedWingSetpoint {
            airspeed: 25.0,
            altitude: 100.0,
            course: 0.0,
        };
        assert!(
            c.step(&e, &sp, 1e-2).rudder > 0.0,
            "r>r_coord → +rudder (left-yaw moment)"
        );
    }

    #[test]
    fn wrap_pi_is_short_way() {
        // −6.0 rad wraps up to +0.283 (the short way through ±π).
        assert!((wrap_pi(-6.0) - 0.2831853).abs() < 1e-6);
        assert!((wrap_pi(6.0) + 0.2831853).abs() < 1e-6);
        assert!((wrap_pi(0.5) - 0.5).abs() < 1e-9);
    }
}
