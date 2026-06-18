//! Control allocation for an X-configuration quadrotor.
//!
//! Motor layout (body FRD, looking down the `+z`/down axis), arm length `L`,
//! each rotor offset `d = L/√2` in `x` and `y`:
//!
//! ```text
//!        x (forward)
//!          ^
//!   M3 •   |   • M0          M0 front-right  (CW)
//!       \  |  /              M1 rear-right   (CCW)
//!        \ | /               M2 rear-left    (CW)
//!  -------+-+-------> y       M3 front-left   (CCW)
//!        / | \  (right)
//!       /  |  \
//!   M2 •   |   • M1
//! ```
//!
//! Each rotor `i` produces thrust `f_i ≥ 0` along body `-z` (up). The
//! allocation matrix `A` maps `f → (T, Mx, My, Mz)`:
//!
//! ```text
//! T  =  f0 + f1 + f2 + f3
//! Mx = -d·f0 - d·f1 + d·f2 + d·f3      (roll,  about body x)
//! My =  d·f0 - d·f1 - d·f2 + d·f3      (pitch, about body y)
//! Mz =  k·f0 - k·f1 + k·f2 - k·f3      (yaw,   rotor reaction torque)
//! ```

use fsim_core::{CtrlCmd, Real, Vec3};
use nalgebra::{Matrix4, Vector4};
use num_traits::Float;

/// Maps between `(collective thrust, body torque)` and individual motor
/// thrusts for a quad.
pub trait Mixer {
    /// Allocate a desired wrench to four motor thrusts \[N\], clamped to the
    /// physically realizable `[0, max_thrust]` per motor.
    fn mix(&self, cmd: &CtrlCmd) -> [Real; 4];

    /// Recombine actual motor thrusts into the achieved wrench (after clamping
    /// and motor lag, the realized thrust/torque differ from the command).
    fn collect(&self, motors: &[Real; 4]) -> CtrlCmd;
}

/// X-quad mixer built from arm length, yaw reaction coefficient, and per-motor
/// thrust limit.
#[derive(Debug, Clone)]
pub struct XQuadMixer {
    /// Forward allocation `f -> (T, Mx, My, Mz)`.
    alloc: Matrix4<Real>,
    /// Inverse allocation `(T, Mx, My, Mz) -> f`.
    alloc_inv: Matrix4<Real>,
    /// Maximum thrust per motor \[N\].
    max_thrust: Real,
}

impl XQuadMixer {
    /// `arm_length` \[m\] (motor-to-center), `yaw_coeff` = reaction-torque /
    /// thrust ratio \[m\], `max_thrust` \[N\] per motor.
    pub fn new(arm_length: Real, yaw_coeff: Real, max_thrust: Real) -> Self {
        let d = arm_length / Float::sqrt(2.0 as Real);
        let k = yaw_coeff;
        #[rustfmt::skip]
        let alloc = Matrix4::new(
            1.0,  1.0,  1.0,  1.0,
            -d,   -d,    d,    d,
             d,   -d,   -d,    d,
             k,   -k,    k,   -k,
        );
        let alloc_inv = alloc
            .try_inverse()
            .expect("X-quad allocation matrix is always invertible");
        Self {
            alloc,
            alloc_inv,
            max_thrust,
        }
    }

    /// Defaults matching [`fsim_dynamics::MultirotorParams::quad_250`]:
    /// 12 cm arm, yaw coeff 0.016 m, 4 N max per motor (~1.6:1 thrust:weight).
    pub fn quad_250() -> Self {
        Self::new(0.12, 0.016, 4.0)
    }

    fn wrench_vector(cmd: &CtrlCmd) -> Vector4<Real> {
        Vector4::new(cmd.thrust, cmd.torque.x, cmd.torque.y, cmd.torque.z)
    }
}

impl Mixer for XQuadMixer {
    fn mix(&self, cmd: &CtrlCmd) -> [Real; 4] {
        let f = self.alloc_inv * Self::wrench_vector(cmd);
        [
            f[0].clamp(0.0, self.max_thrust),
            f[1].clamp(0.0, self.max_thrust),
            f[2].clamp(0.0, self.max_thrust),
            f[3].clamp(0.0, self.max_thrust),
        ]
    }

    fn collect(&self, motors: &[Real; 4]) -> CtrlCmd {
        let u = self.alloc * Vector4::new(motors[0], motors[1], motors[2], motors[3]);
        CtrlCmd {
            thrust: u[0],
            torque: Vec3::new(u[1], u[2], u[3]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::{CtrlCmd, Vec3, GRAVITY};

    #[test]
    fn mix_then_collect_round_trips_within_limits() {
        let m = XQuadMixer::quad_250();
        // A modest, realizable command (well below saturation).
        let cmd = CtrlCmd {
            thrust: 0.5 * GRAVITY, // ~4.9 N total -> ~1.2 N/motor
            torque: Vec3::new(0.02, -0.015, 0.01),
        };
        let motors = m.mix(&cmd);
        for f in motors {
            assert!((0.0..=4.0).contains(&f));
        }
        let back = m.collect(&motors);
        assert!((back.thrust - cmd.thrust).abs() < 1e-9);
        assert!((back.torque - cmd.torque).norm() < 1e-9);
    }

    #[test]
    fn pure_hover_thrust_splits_evenly() {
        let m = XQuadMixer::quad_250();
        let cmd = CtrlCmd {
            thrust: 4.0,
            torque: Vec3::zeros(),
        };
        let motors = m.mix(&cmd);
        for f in motors {
            assert!((f - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn positive_roll_command_loads_correct_side() {
        // +Mx (roll-right) should raise the left rotors (M2,M3) vs right (M0,M1).
        let m = XQuadMixer::quad_250();
        let cmd = CtrlCmd {
            thrust: 4.0,
            torque: Vec3::new(0.05, 0.0, 0.0),
        };
        let f = m.mix(&cmd);
        assert!(f[2] > f[0] && f[3] > f[1]);
    }
}
