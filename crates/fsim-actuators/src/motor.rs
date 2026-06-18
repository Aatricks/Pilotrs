//! First-order motor model: thrust lags its command with time constant `tau`.
//!
//! M1 uses [`MotorModel::ideal`] (`tau = 0`, instantaneous). M3 sets a real
//! `tau` (a few tens of ms) and can later add a nonlinear thrust curve.

use fsim_core::Real;
use num_traits::Float;

/// Four motors, each a first-order lag on commanded thrust \[N\].
#[derive(Debug, Clone)]
pub struct MotorModel {
    /// Current actual thrust per motor \[N\].
    thrust: [Real; 4],
    /// Lag time constant \[s\]. `0` = instantaneous (ideal).
    tau: Real,
    /// Per-motor thrust limit \[N\].
    max_thrust: Real,
}

impl MotorModel {
    /// A motor model with first-order lag `tau` \[s\] and thrust limit \[N\].
    pub fn new(tau: Real, max_thrust: Real) -> Self {
        Self {
            thrust: [0.0; 4],
            tau,
            max_thrust,
        }
    }

    /// Ideal motors (no lag) — used by the M1 MVP.
    pub fn ideal(max_thrust: Real) -> Self {
        Self::new(0.0, max_thrust)
    }

    /// Current actual thrust per motor \[N\].
    pub fn thrust(&self) -> [Real; 4] {
        self.thrust
    }

    /// Advance the motors one step toward the commanded thrusts and return the
    /// new actual thrusts. Exact first-order discretization:
    /// `f += (cmd − f)·(1 − e^{−dt/τ})`.
    pub fn update(&mut self, command: &[Real; 4], dt: Real) -> [Real; 4] {
        let alpha = if self.tau <= 0.0 {
            1.0
        } else {
            1.0 - Float::exp(-dt / self.tau)
        };
        for (thrust, &raw_cmd) in self.thrust.iter_mut().zip(command.iter()) {
            let cmd = raw_cmd.clamp(0.0, self.max_thrust);
            *thrust += (cmd - *thrust) * alpha;
        }
        self.thrust
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ideal_motors_track_command_instantly() {
        let mut m = MotorModel::ideal(4.0);
        let out = m.update(&[1.0, 2.0, 3.0, 4.0], 1e-3);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn lagged_motor_reaches_63pct_after_one_tau() {
        // Classic first-order step response: ~63.2% of the step after t = tau.
        let tau = 0.05;
        let mut m = MotorModel::new(tau, 4.0);
        let dt = 1e-4;
        let steps = (tau / dt) as usize;
        for _ in 0..steps {
            m.update(&[2.0; 4], dt);
        }
        assert!(
            (m.thrust()[0] - 2.0 * 0.632).abs() < 0.02,
            "got {}",
            m.thrust()[0]
        );
    }

    #[test]
    fn commands_saturate_at_max() {
        let mut m = MotorModel::ideal(4.0);
        let out = m.update(&[10.0, -5.0, 4.0, 0.0], 1e-3);
        assert_eq!(out, [4.0, 0.0, 4.0, 0.0]);
    }
}
