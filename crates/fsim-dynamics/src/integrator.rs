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

        // k1, k2 : les deux premières pentes RK4 (k1 au départ, k2 au milieu prédit par
        // k1). Chaque étape intermédiaire passe par `State13::from_vector`, qui renormalise
        // le quaternion d'attitude.
        let k1 = deriv(state).to_vector();
        let k2 = deriv(&State13::from_vector(&(y0 + half * k1))).to_vector();

        // k3 : pente au milieu corrigée par k2 ; k4 : pente en fin d'intervalle (pas complet dt).
        let k3 = deriv(&State13::from_vector(&(y0 + half * k2))).to_vector();
        let k4 = deriv(&State13::from_vector(&(y0 + dt * k3))).to_vector();

        // Combinaison pondérée (poids 1,2,2,1)/6 = quadrature de Simpson → précision
        // d'ordre 4 ; `from_vector` renormalise une dernière fois.
        let y1 = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
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
