//! Fault injection — making the aircraft break, on purpose.
//!
//! Effector and powerplant failures: a control surface that **jams**, **floats**,
//! or goes **mushy**; an **engine** that quits; a **tail** that's torn off; a
//! **dead rotor**. They are applied in the sim loop between the controller's
//! command and the plant, so the autopilot / fly-by-wire keeps trying — and you
//! watch it cope (a dead-stick glide) or lose it (a jammed elevator on a
//! relaxed-stability fighter is as good as switching the FCS off).
//!
//! Sensor faults live with the quad's estimator (see the scheduler); the
//! fixed-wing flies on truth, so it has none.

use fsim_core::{FixedWingControls, Real};
use fsim_dynamics::FixedWingParams;

/// What's wrong with one control surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceFault {
    /// Working normally.
    Normal,
    /// Jammed at a fixed deflection \[rad\], ignoring the command.
    Jammed(Real),
    /// Floating free — no authority (deflection forced to 0).
    Floating,
    /// Mushy: the commanded deflection is scaled by this factor in `[0, 1]`.
    Degraded(Real),
}

impl SurfaceFault {
    /// Map a commanded deflection through the fault.
    fn apply(self, cmd: Real) -> Real {
        match self {
            Self::Normal => cmd,
            Self::Jammed(angle) => angle,
            Self::Floating => 0.0,
            Self::Degraded(k) => cmd * k.clamp(0.0, 1.0),
        }
    }

    fn is_faulted(self) -> bool {
        !matches!(self, Self::Normal)
    }
}

/// Fixed-wing effector + powerplant faults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwFaults {
    /// The engine has quit — throttle forced to 0 (a dead-stick glide).
    pub engine_out: bool,
    pub aileron: SurfaceFault,
    pub elevator: SurfaceFault,
    pub rudder: SurfaceFault,
    /// The tail is gone: it carried the elevator + rudder *and* most of the
    /// pitch/yaw damping, so all four are lost. On the relaxed fighter, fatal.
    pub tail_loss: bool,
}

impl FwFaults {
    /// A healthy aircraft.
    pub fn none() -> Self {
        Self {
            engine_out: false,
            aileron: SurfaceFault::Normal,
            elevator: SurfaceFault::Normal,
            rudder: SurfaceFault::Normal,
            tail_loss: false,
        }
    }

    /// True if anything is broken.
    pub fn any(&self) -> bool {
        self.engine_out
            || self.tail_loss
            || self.aileron.is_faulted()
            || self.elevator.is_faulted()
            || self.rudder.is_faulted()
    }

    /// Apply the control-path faults to the commanded surfaces (after the
    /// controller, before the aero).
    pub fn apply_controls(&self, mut c: FixedWingControls) -> FixedWingControls {
        c.aileron = self.aileron.apply(c.aileron);
        c.elevator = self.elevator.apply(c.elevator);
        c.rudder = self.rudder.apply(c.rudder);
        if self.engine_out {
            c.throttle = 0.0;
        }
        if self.tail_loss {
            c.elevator = 0.0;
            c.rudder = 0.0;
        }
        c
    }

    /// Apply the airframe-damage faults to the aero coefficients: losing the tail
    /// guts the pitch/yaw damping and removes the elevator/rudder authority.
    pub fn apply_params(&self, mut p: FixedWingParams) -> FixedWingParams {
        if self.tail_loss {
            p.cm_q *= 0.2; // most pitch damping gone
            p.cnr *= 0.2; // most yaw damping gone
            p.cm_de = 0.0; // no elevator authority
            p.cn_dr = 0.0; // no rudder authority
        }
        p
    }
}

/// Quadrotor effector faults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadFaults {
    /// A dead rotor: this motor index (0..3) produces no thrust.
    pub dead_rotor: Option<usize>,
}

impl QuadFaults {
    /// A healthy aircraft.
    pub fn none() -> Self {
        Self { dead_rotor: None }
    }

    /// True if anything is broken.
    pub fn any(&self) -> bool {
        self.dead_rotor.is_some()
    }

    /// Zero out a dead rotor's thrust.
    pub fn apply_motors(&self, mut thrust: [Real; 4]) -> [Real; 4] {
        if let Some(i) = self.dead_rotor {
            if i < 4 {
                thrust[i] = 0.0;
            }
        }
        thrust
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::FixedWingControls;

    #[test]
    fn surface_faults_map_the_command() {
        assert_eq!(SurfaceFault::Normal.apply(0.3), 0.3);
        assert_eq!(SurfaceFault::Jammed(0.1).apply(0.3), 0.1); // ignores command
        assert_eq!(SurfaceFault::Floating.apply(0.3), 0.0);
        assert_eq!(SurfaceFault::Degraded(0.5).apply(0.4), 0.2);
    }

    #[test]
    fn engine_out_kills_throttle() {
        let mut f = FwFaults::none();
        f.engine_out = true;
        let c = f.apply_controls(FixedWingControls {
            aileron: 0.1,
            elevator: 0.2,
            rudder: 0.0,
            throttle: 0.6,
        });
        assert_eq!(c.throttle, 0.0);
        assert_eq!(c.elevator, 0.2, "surfaces still work on a glide");
    }

    #[test]
    fn tail_loss_zeros_pitch_yaw_control_and_damping() {
        let mut f = FwFaults::none();
        f.tail_loss = true;
        let c = f.apply_controls(FixedWingControls {
            aileron: 0.1,
            elevator: 0.2,
            rudder: 0.3,
            throttle: 0.6,
        });
        assert_eq!(c.elevator, 0.0, "no elevator without a tail");
        assert_eq!(c.rudder, 0.0, "no rudder without a tail");
        assert_eq!(c.aileron, 0.1, "wings still have ailerons");

        let healthy = FixedWingParams::fighter_relaxed();
        let damaged = f.apply_params(healthy);
        assert!(
            damaged.cm_q.abs() < healthy.cm_q.abs(),
            "lost pitch damping"
        );
        assert_eq!(damaged.cm_de, 0.0, "lost elevator authority");
        assert_eq!(damaged.cn_dr, 0.0, "lost rudder authority");
    }

    #[test]
    fn dead_rotor_zeros_one_motor() {
        let f = QuadFaults {
            dead_rotor: Some(2),
        };
        assert_eq!(f.apply_motors([1.0, 1.0, 1.0, 1.0]), [1.0, 1.0, 0.0, 1.0]);
        assert!(!QuadFaults::none().any());
    }
}
