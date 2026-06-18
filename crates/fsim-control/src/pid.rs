//! A reusable scalar PID with derivative-on-error, integral anti-windup, and
//! output clamping.

use fsim_core::Real;

/// One PID channel.
#[derive(Debug, Clone)]
pub struct Pid {
    kp: Real,
    ki: Real,
    kd: Real,
    /// Clamp on the accumulated integral *term* `ki·integral` (anti-windup).
    integral_limit: Real,
    /// Symmetric clamp on the output.
    output_limit: Real,
    integral: Real,
    prev_error: Real,
    primed: bool,
}

impl Pid {
    pub fn new(kp: Real, ki: Real, kd: Real, integral_limit: Real, output_limit: Real) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral_limit,
            output_limit,
            integral: 0.0,
            prev_error: 0.0,
            primed: false,
        }
    }

    /// Forget integral and derivative history (e.g. when re-arming).
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.primed = false;
    }

    /// Advance one step on the given error and return the (clamped) output.
    pub fn step(&mut self, error: Real, dt: Real) -> Real {
        // Integrate, then clamp the integral so its *contribution* stays bounded.
        self.integral += error * dt;
        if self.ki > 0.0 {
            let i_max = self.integral_limit / self.ki;
            self.integral = self.integral.clamp(-i_max, i_max);
        }

        // Derivative on error (skip the first call to avoid a spurious kick).
        let derivative = if self.primed {
            (error - self.prev_error) / dt
        } else {
            0.0
        };
        self.prev_error = error;
        self.primed = true;

        let out = self.kp * error + self.ki * self.integral + self.kd * derivative;
        out.clamp(-self.output_limit, self.output_limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proportional_sign_and_magnitude() {
        let mut pid = Pid::new(2.0, 0.0, 0.0, 0.0, 100.0);
        assert!((pid.step(3.0, 1e-3) - 6.0).abs() < 1e-12);
        assert!((pid.step(-1.5, 1e-3) + 3.0).abs() < 1e-12);
    }

    #[test]
    fn integral_accumulates_and_winds_up_bounded() {
        // ki=1, integral term clamped to ±0.5.
        let mut pid = Pid::new(0.0, 1.0, 0.0, 0.5, 100.0);
        for _ in 0..10_000 {
            pid.step(1.0, 1e-3); // constant positive error
        }
        // ki*integral must not exceed the anti-windup limit.
        assert!((pid.step(1.0, 1e-3) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn output_is_clamped() {
        let mut pid = Pid::new(1000.0, 0.0, 0.0, 0.0, 1.0);
        assert!((pid.step(5.0, 1e-3) - 1.0).abs() < 1e-12);
    }
}
