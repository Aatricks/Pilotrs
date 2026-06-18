//! Fixed-step integrators. RK4 is the deterministic heart of the plant; Euler
//! is kept for A/B drift comparisons in tests.

use fsim_core::{Real, State13, StateDeriv};

/// A fixed-step ODE integrator over the rigid-body state.
///
/// `deriv` evaluates the state derivative at a given state; the integrator is
/// free to call it at intermediate stages. The returned state has its attitude
/// quaternion renormalized.
pub trait Integrator {
    fn step<F>(&self, state: &State13, deriv: F, dt: Real) -> State13
    where
        F: Fn(&State13) -> StateDeriv;
}

/// Classic 4th-order Runge-Kutta. Intermediate stages are rebuilt through
/// [`State13::from_vector`], which renormalizes the quaternion each stage —
/// standard, stable practice for attitude integration.
#[derive(Debug, Clone, Copy, Default)]
pub struct Rk4;

impl Integrator for Rk4 {
    fn step<F>(&self, state: &State13, deriv: F, dt: Real) -> State13
    where
        F: Fn(&State13) -> StateDeriv,
    {
        let y0 = state.to_vector();
        let half = dt * 0.5;

        let k1 = deriv(state).to_vector();
        let k2 = deriv(&State13::from_vector(&(y0 + k1 * half))).to_vector();
        let k3 = deriv(&State13::from_vector(&(y0 + k2 * half))).to_vector();
        let k4 = deriv(&State13::from_vector(&(y0 + k3 * dt))).to_vector();

        let y1 = y0 + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
        State13::from_vector(&y1)
    }
}

/// Forward Euler. First-order, for comparison only — never use it for the real
/// loop (it bleeds energy and drifts attitude).
#[derive(Debug, Clone, Copy, Default)]
pub struct Euler;

impl Integrator for Euler {
    fn step<F>(&self, state: &State13, deriv: F, dt: Real) -> State13
    where
        F: Fn(&State13) -> StateDeriv,
    {
        let y1 = state.to_vector() + deriv(state).to_vector() * dt;
        State13::from_vector(&y1)
    }
}
