//! Fixed-wing aircraft: aerodynamic force/moment model, parameters, and a trim
//! solver. The 6DOF rigid-body EOM and the RK4 integrator are **shared**
//! with the multirotor — only the [`Wrench`] computation and the mass/inertia
//! differ. The aero model is the Beard & McLain linear-coefficient model
//! expressed in our NED/FRD conventions.
//!
//! ## Frames and air data
//!
//! Air-relative body velocity `v_body = q⁻¹·(velocity_world − wind_world) =
//! (u,v,w)`, airspeed `Va = ‖v_body‖`, angle of attack `α = atan2(w,u)`,
//! sideslip `β = asin(v/Va)`. Aero forces are formed in stability axes (lift ⟂
//! airflow, drag ‖ airflow) then rotated into the body by `α`; gravity is added
//! in the world frame (never rotated), exactly as the quad does. Propeller
//! thrust acts along body **+x** (forward), not body −z (the quad's lift axis).

use crate::plant::rigid_body_deriv;
use fsim_core::{gravity_world, ControlLimits, FixedWingControls, Real, State13, Vec3, Wrench};
use nalgebra::{Matrix3, UnitQuaternion};
use num_traits::Float;

/// Mass, geometry, air, and aerodynamic coefficients of a fixed-wing airframe.
#[derive(Debug, Clone, Copy)]
pub struct FixedWingParams {
    /// Mass \[kg\].
    pub mass: Real,
    /// Body-frame inertia tensor (may be non-diagonal: `Jxz` cross term) \[kg·m²\].
    pub inertia: Matrix3<Real>,
    /// Precomputed inverse inertia.
    pub inertia_inv: Matrix3<Real>,

    /// Air density \[kg/m³\].
    pub rho: Real,
    /// Wing reference area \[m²\].
    pub s: Real,
    /// Wing span \[m\].
    pub b: Real,
    /// Mean aerodynamic chord \[m\].
    pub c: Real,
    /// Aspect ratio `b²/S`.
    pub ar: Real,
    /// Oswald efficiency.
    pub e_osw: Real,

    /// Minimum airspeed used only for `1/Va` damping factors \[m/s\].
    pub va_min: Real,
    /// Stall angle \[rad\] (blend centre).
    pub alpha_stall: Real,
    /// Stall blend sharpness.
    pub m_blend: Real,

    /// Propeller disc area \[m²\].
    pub sprop: Real,
    /// Propeller thrust coefficient.
    pub cprop: Real,
    /// Motor constant (max induced velocity) \[m/s\].
    pub kmotor: Real,

    // Longitudinal coefficients.
    pub cl0: Real,
    pub cl_alpha: Real,
    pub cl_q: Real,
    pub cl_de: Real,
    pub cd_p: Real,
    pub cd_q: Real,
    pub cd_de: Real,
    pub cm0: Real,
    pub cm_alpha: Real,
    pub cm_q: Real,
    pub cm_de: Real,
    /// Pitch-break magnitude \[—\]: a stall-gated nose-down increment to `Cm`,
    /// blended in by `σ(α)`. Negative ⇒ nose-down (the textbook break a
    /// straight-wing airframe shows as the flow separates). Dormant below stall.
    pub cm_stall: Real,

    // Lateral coefficients.
    pub cy0: Real,
    pub cy_beta: Real,
    pub cy_p: Real,
    pub cy_r: Real,
    pub cy_da: Real,
    pub cy_dr: Real,
    pub cl0_roll: Real,
    pub cl_beta: Real,
    pub clp: Real,
    /// Post-stall roll-damping coefficient \[—\]: the value `Clp` is blended
    /// *toward* past the stall. Positive ⇒ the roll damping reverses into
    /// **anti-damping** (autorotation): a dropped wing keeps dropping, the
    /// mechanism of an incipient spin. Below stall `Clp` is used unchanged.
    pub clp_spin: Real,
    pub clr: Real,
    pub cl_da: Real,
    pub cl_dr: Real,
    pub cn0: Real,
    pub cn_beta: Real,
    pub cnp: Real,
    pub cnr: Real,
    pub cn_da: Real,
    pub cn_dr: Real,

    /// Actuator limits.
    pub limits: ControlLimits,
}

impl FixedWingParams {
    /// The Aerosonde UAV (Beard & McLain, Appendix E) — a ~13.5 kg fixed-wing
    /// that cruises near 25 m/s. Inertia carries the `Jxz` cross term.
    pub fn aerosonde() -> Self {
        let (jx, jy, jz, jxz) = (0.8244, 1.135, 1.759, 0.1204);
        // B&M sign: the off-diagonal entries are −Jxz.
        let inertia = Matrix3::new(jx, 0.0, -jxz, 0.0, jy, 0.0, -jxz, 0.0, jz);
        let inertia_inv = inertia.try_inverse().expect("inertia singular");
        let b = 2.8956;
        let s = 0.55;
        Self {
            mass: 13.5,
            inertia,
            inertia_inv,
            rho: 1.2682,
            s,
            b,
            c: 0.18994,
            ar: b * b / s,
            e_osw: 0.9,
            va_min: 0.1,
            alpha_stall: 0.4712,
            m_blend: 50.0,
            sprop: 0.2027,
            cprop: 1.0,
            kmotor: 80.0,
            cl0: 0.28,
            cl_alpha: 3.45,
            cl_q: 0.0,
            cl_de: 0.36,
            cd_p: 0.03,
            cd_q: 0.0,
            cd_de: 0.0,
            cm0: -0.02338,
            cm_alpha: -0.38,
            cm_q: -3.6,
            cm_de: -0.5,
            cm_stall: -0.20,
            cy0: 0.0,
            cy_beta: -0.98,
            cy_p: 0.0,
            cy_r: 0.0,
            cy_da: 0.0,
            cy_dr: 0.17,
            cl0_roll: 0.0,
            cl_beta: -0.12,
            clp: -0.26,
            clp_spin: 0.10,
            clr: 0.14,
            cl_da: 0.08,
            cl_dr: 0.105,
            cn0: 0.0,
            cn_beta: 0.25,
            cnp: 0.022,
            cnr: -0.35,
            cn_da: 0.06,
            cn_dr: -0.032,
            limits: ControlLimits {
                surface_max: 0.4363, // ±25°
                throttle: (0.0, 1.0),
            },
        }
    }

    /// A **relaxed-stability** airframe: the Aerosonde with its centre of gravity
    /// moved aft of the aerodynamic centre, i.e. a *negative static margin*. The
    /// only change from [`aerosonde`](Self::aerosonde) is the sign of `cm_alpha`
    /// (−0.38 → **+0.30**) — every other coefficient, mass, and inertia is
    /// identical. That single flip is exactly what relaxed stability *is*, and it
    /// turns the docile, self-correcting UAV into a divergent airframe: the
    /// short-period mode picks up a positive real eigenvalue (time-to-double a
    /// few tenths of a second), so it pitches away from trim and tumbles within
    /// about a second unless a control law holds it. It still *trims* — an
    /// unstable equilibrium is still an equilibrium — so the Newton [`trim`]
    /// solver converges; it just can't be flown open-loop.
    pub fn fighter_relaxed() -> Self {
        Self {
            // Aft CG → the pitching moment now *grows* with angle of attack.
            cm_alpha: 0.30,
            ..Self::aerosonde()
        }
    }

    /// Stall blend σ(α) ∈ [0,1]: ~0 in the linear regime, 0.5 at the stall
    /// angle, →1 past it (Beard & McLain sigmoid).
    fn sigma(&self, alpha: Real) -> Real {
        let (a0, m) = (self.alpha_stall, self.m_blend);
        let e_neg = Float::exp(-m * (alpha - a0));
        let e_pos = Float::exp(m * (alpha + a0));
        (1.0 + e_neg + e_pos) / ((1.0 + e_neg) * (1.0 + e_pos))
    }
}

/// Net wrench on a fixed-wing from aerodynamics, propeller thrust, and gravity.
///
/// `gravity_world` is the gravity acceleration in the *same* world frame the
/// state lives in — the caller supplies it so the same plant serves both the
/// flat-earth trim (constant `(0,0,+g)`) and the spherical sim (radial
/// `fsim_core::planet::gravity_at(position)`); the aero forces are body-relative
/// and frame-agnostic, so only this term differs between the two worlds.
pub fn fixedwing_wrench(
    state: &State13,
    p: &FixedWingParams,
    c: &FixedWingControls,
    wind_world: Vec3,
    gravity_world: Vec3,
) -> Wrench {
    // --- air data (body frame) ---
    let v_body = state.attitude.inverse() * (state.velocity - wind_world);
    let (u, v, w) = (v_body.x, v_body.y, v_body.z);
    let va = v_body.norm();
    let va_s = Float::max(va, p.va_min);
    let alpha = Float::atan2(w, u);
    let beta = if va > p.va_min {
        Float::asin((v / va).clamp(-1.0, 1.0))
    } else {
        0.0
    };
    let qbar = 0.5 * p.rho * va * va; // raw Va: vanishes at Va→0
    let cf = p.c / (2.0 * va_s); // longitudinal damping factor
    let bf = p.b / (2.0 * va_s); // lateral damping factor
    let (pp, qq, rr) = (
        state.angular_rate.x,
        state.angular_rate.y,
        state.angular_rate.z,
    );

    // --- longitudinal coefficients (with stall blend) ---
    let sigma = p.sigma(alpha);
    let cl_lin = p.cl0 + p.cl_alpha * alpha;
    let sa = Float::sin(alpha);
    let ca = Float::cos(alpha);
    // Portance avec mélange de décrochage : sous le décrochage (σ≈0) cl suit la droite Cl_α ;
    // au-delà (σ→1) il suit la courbe de plaque plane 2·signum(α)·sin²α·cosα (max vers 45°
    // puis chute). Le mélange σ rend la transition continue (pas de saut qui casserait RK4).
    let cl_plate = 2.0 * Float::signum(alpha) * sa * sa * ca;
    let cl = (1.0 - sigma) * cl_lin + sigma * cl_plate + p.cl_q * cf * qq + p.cl_de * c.elevator;
    let cd = p.cd_p
        + cl_lin * cl_lin / (core::f64::consts::PI * p.e_osw * p.ar)
        + p.cd_q * cf * qq
        + p.cd_de * c.elevator;
    // Pitch break: past the stall angle the moment breaks nose-down (toward
    // reducing |α|). `σ` gates it on at stall; `signum(α)` makes it restoring on
    // both the upright and inverted stall — and since `σ ≈ 0` below stall, the
    // discontinuity at α = 0 is multiplied away, so this is dormant in cruise.
    let cm = p.cm0
        + p.cm_alpha * alpha
        + p.cm_q * cf * qq
        + p.cm_de * c.elevator
        + sigma * p.cm_stall * Float::signum(alpha);

    // --- lateral coefficients ---
    let cy = p.cy0
        + p.cy_beta * beta
        + p.cy_p * bf * pp
        + p.cy_r * bf * rr
        + p.cy_da * c.aileron
        + p.cy_dr * c.rudder;
    // Lateral stall behaviour, both gated by the same `σ`:
    //  * roll damping reverses into anti-damping past stall (autorotation) — a
    //    dropped wing keeps dropping, the seed of an incipient spin;
    //  * aileron authority fades to zero — you cannot pick a stalled wing back up
    //    with aileron, only by unstalling.
    // Below stall σ ≈ 0, so both are bit-for-bit the conventional model.
    // Décrochage latéral (gated par σ) : l'amortissement en roulis s'inverse en
    // anti-amortissement (autorotation — germe de vrille) et l'aileron perd son autorité.
    // Sous le décrochage (σ≈0) on retrouve exactement le modèle conventionnel.
    let clp_eff = p.clp * (1.0 - sigma) + p.clp_spin * sigma;
    let cl_da_eff = p.cl_da * (1.0 - sigma);
    let cl_roll = p.cl0_roll
        + p.cl_beta * beta
        + clp_eff * bf * pp
        + p.clr * bf * rr
        + cl_da_eff * c.aileron
        + p.cl_dr * c.rudder;
    let cn = p.cn0
        + p.cn_beta * beta
        + p.cnp * bf * pp
        + p.cnr * bf * rr
        + p.cn_da * c.aileron
        + p.cn_dr * c.rudder;

    // --- forces: stability axes → body (rotate lift/drag by α about body y) ---
    let f_lift = qbar * p.s * cl;
    let f_drag = qbar * p.s * cd;
    let f_y = qbar * p.s * cy;
    // Rotation des axes de stabilité (portance ⟂ au vent, traînée ∥) vers le repère corps,
    // tourné de l'incidence α autour de l'axe y : fx = −D·cosα + L·sinα, fz = −D·sinα − L·cosα.
    // La portance « tire » vers le haut = −z corps.
    let mut fx = -f_drag * ca + f_lift * sa;
    let fy = f_y;
    let fz = -f_drag * sa - f_lift * ca;

    // Propeller thrust along body +x.
    let kt = p.kmotor * c.throttle;
    // Poussée hélice (disque actuateur) : T = max(½·ρ·Sprop·Cprop·(kt²−Va²), 0). L'hélice
    // accélère l'air de Va à kt ; bornée à ≥ 0, et → 0 quand Va → kt (fixe la vitesse max).
    let thrust = Float::max(
        0.5 * p.rho * p.sprop * p.cprop * ((kt * kt) - (va * va)),
        0.0,
    );
    fx += thrust;

    let f_body = Vec3::new(fx, fy, fz);
    // Force nette en repère MONDE : les forces aéro/poussée (en corps) sont ramenées au monde
    // par l'attitude, et on y ajoute la gravité (déjà fournie en repère monde par l'appelant).
    let force_world = state.attitude * f_body + gravity_world * p.mass;

    // --- moments (body frame) ---
    let l = qbar * p.s * p.b * cl_roll;
    let m = qbar * p.s * p.c * cm;
    let n = qbar * p.s * p.b * cn;

    Wrench {
        force_world,
        moment_body: Vec3::new(l, m, n),
    }
}

/// A steady-flight trim point: the state and controls that hold it.
#[derive(Debug, Clone, Copy)]
pub struct Trim {
    pub state: State13,
    pub controls: FixedWingControls,
}

/// Build the (purely kinematic) trimmed `State13` for a candidate `α`.
fn trim_state(va: Real, gamma: Real, alpha: Real) -> State13 {
    let theta = alpha + gamma;
    let v_body = Vec3::new(va * Float::cos(alpha), 0.0, va * Float::sin(alpha));
    let attitude = UnitQuaternion::from_euler_angles(0.0, theta, 0.0);
    State13 {
        position: Vec3::zeros(),
        velocity: attitude * v_body,
        attitude,
        angular_rate: Vec3::zeros(),
    }
}

/// Solve for wings-level trim at airspeed `va` and flight-path angle `gamma`
/// (0 = level cruise). Newton iteration on `[α, elevator, throttle]` driving the
/// body-axis forward/normal accelerations and pitch acceleration to zero.
///
/// Returns `None` when the request is infeasible — e.g. an airspeed below the
/// stall speed leaves a control surface pinned at its limit with a large
/// residual, which is reported as failure rather than a spurious "equilibrium".
pub fn trim(p: &FixedWingParams, va: Real, gamma: Real) -> Option<Trim> {
    // Residual r(x) for x = [alpha, elevator, throttle].
    let residual = |x: &Vec3| -> Vec3 {
        let state = trim_state(va, gamma, x[0]);
        let controls = FixedWingControls {
            aileron: 0.0,
            elevator: x[1],
            rudder: 0.0,
            throttle: x[2],
        };
        let wrench = fixedwing_wrench(&state, p, &controls, Vec3::zeros(), gravity_world());
        let d = rigid_body_deriv(&state, &wrench, p.mass, &p.inertia, &p.inertia_inv);
        // Résidu de trim (à annuler à l'équilibre) : accélération AVANT (x corps, réglée par
        // les gaz), accélération NORMALE (z corps, réglée par α), accélération de TANGAGE q̇
        // (réglée par la profondeur). Roulis/lacet/dérapage sont nuls par symétrie.
        Vec3::new(
            (state.attitude.inverse() * d.d_velocity).x,
            (state.attitude.inverse() * d.d_velocity).z,
            d.d_angular_rate.y,
        )
    };

    // Start the throttle *above* the propeller's thrust dead-zone for this speed:
    // thrust is zero until `kmotor·δt > Va`, and in that flat region `∂T/∂δt = 0`,
    // so the throttle column of the forward-difference Jacobian vanishes and Newton
    // stalls. Beginning above the knee lets fast cruises (where the dead-zone is
    // wide) converge; for slow flight it clamps back to a sane mid-throttle.
    let dt0 = (va / p.kmotor + 0.15).clamp(0.3, 0.9);
    let mut x = Vec3::new(gamma, 0.0, dt0);
    for _ in 0..50 {
        let r = residual(&x);
        if r.norm() < 1e-12 {
            break;
        }
        // Forward-difference 3×3 Jacobian.
        let h = 1e-6;
        let mut j = Matrix3::zeros();
        for k in 0..3 {
            let mut xp = x;
            xp[k] += h;
            // Colonne k du Jacobien par différence finie avant : ∂r/∂x_k ≈ (r(x+h·eₖ) − r(x))/h.
            // Dérivée numérique (pas à la main) ⇒ le Jacobien reste fidèle au modèle aéro réel.
            let col = Vec3::new(
                (residual(&xp).x - r.x) / h,
                (residual(&xp).y - r.y) / h,
                (residual(&xp).z - r.z) / h,
            );
            j.set_column(k, &col);
        }
        match j.try_inverse() {
            Some(jinv) => {
                // Pas de Newton : x ← x − J⁻¹·r (pousse le résidu vers 0). J singulier ⇒ abandon.
                x -= jinv * r;
            }
            None => break,
        }
        x[1] = x[1].clamp(-p.limits.surface_max, p.limits.surface_max);
        x[2] = x[2].clamp(p.limits.throttle.0, p.limits.throttle.1);
    }

    // Convergence/feasibility gate: the surfaces are clamped each iteration but
    // the residual is not, so a saturated solver freezes at a non-stationary
    // point with a large residual. Recompute it once more and only return an
    // actual equilibrium.
    if residual(&x).norm() >= 1e-6 {
        return None;
    }
    Some(Trim {
        state: trim_state(va, gamma, x[0]),
        controls: FixedWingControls {
            aileron: 0.0,
            elevator: x[1],
            rudder: 0.0,
            throttle: x[2],
        },
    })
}

/// The two eigenvalues of the short-period mode, each as a `(real, imag)` pair.
///
/// The sign of the largest real part is the whole story: `< 0` ⇒ perturbations
/// decay (flyable); `> 0` ⇒ they grow (the relaxed-stability airframe diverging
/// open-loop). This is the *formal* counterpart to watching the sim tumble.
#[derive(Debug, Clone, Copy)]
pub struct ShortPeriodModes {
    /// Eigenvalues `[(re, im); 2]` of the 2×2 `[α, q]` state matrix.
    pub eig: [(Real, Real); 2],
}

impl ShortPeriodModes {
    /// The largest real part across both eigenvalues (the dominant growth rate).
    pub fn max_real(&self) -> Real {
        Float::max(self.eig[0].0, self.eig[1].0)
    }

    /// Stable iff every eigenvalue sits strictly in the left-half plane.
    pub fn is_stable(&self) -> bool {
        self.max_real() < 0.0
    }

    /// Time for the most unstable mode to double in amplitude \[s\]; only
    /// meaningful when [`max_real`](Self::max_real) `> 0`.
    pub fn time_to_double(&self) -> Real {
        core::f64::consts::LN_2 / self.max_real()
    }
}

/// Linearize the **short-period** pitch dynamics about a trim point and return
/// the two eigenvalues. The reduced state is `[α, q]` (angle of attack, pitch
/// rate) at fixed airspeed — the textbook approximation that isolates the fast
/// pitch mode where the static-margin instability lives.
///
/// The `2×2` state matrix `A = ∂[α̇, q̇]/∂[α, q]` is *finite-differenced from the
/// real aero + rigid-body model* (not re-derived by hand), so it can never drift
/// out of sync with [`fixedwing_wrench`]. Gravity is excluded (it drives the slow
/// phugoid, not this mode), so only the aerodynamic/thrust forces enter `α̇`:
///
/// ```text
/// α̇ = q + (u·Fz_body − w·Fx_body) / (m·Va²)      (flight-path rotation in pitch)
/// q̇ = M_body.y / Iyy  (via the shared EOM, incl. the Jxz cross term)
/// ```
///
/// `elevator_of(α, q)` supplies the elevator deflection as a function of the
/// state: pass a constant (`|_, _| δe_trim`) for the **open-loop** matrix, or the
/// fly-by-wire feedback law for the **closed-loop** matrix. Folding the feedback
/// into `A` this way makes the eigenvalues a direct, faithful check of the FCS.
/// It is `FnMut` so the closure can drive a stateful controller (e.g. call the
/// real `FlyByWire::step`) rather than only a hand-coded gain expression.
pub fn short_period_modes(
    p: &FixedWingParams,
    trim: &Trim,
    mut elevator_of: impl FnMut(Real, Real) -> Real,
) -> ShortPeriodModes {
    let va = trim.state.velocity.norm();
    let attitude = trim.state.attitude;
    let vb_trim = attitude.inverse() * trim.state.velocity;
    let alpha_trim = Float::atan2(vb_trim.z, vb_trim.x);
    let throttle = trim.controls.throttle;

    let mut f = |alpha: Real, q: Real| -> (Real, Real) {
        let vb = Vec3::new(va * Float::cos(alpha), 0.0, va * Float::sin(alpha));
        let state = State13 {
            position: Vec3::zeros(),
            velocity: attitude * vb,
            attitude,
            angular_rate: Vec3::new(0.0, q, 0.0),
        };
        let controls = FixedWingControls {
            aileron: 0.0,
            elevator: elevator_of(alpha, q),
            rudder: 0.0,
            throttle,
        };
        let wrench = fixedwing_wrench(&state, p, &controls, Vec3::zeros(), Vec3::zeros());
        let f_body = attitude.inverse() * wrench.force_world;
        let q_dot = rigid_body_deriv(&state, &wrench, p.mass, &p.inertia, &p.inertia_inv)
            .d_angular_rate
            .y;
        // Rotation de la trajectoire en tangage : α̇ = q + (u·Fz − w·Fx)/(m·Va²). À Va fixe,
        // l'incidence tourne sous l'effet de la force normale. Gravité exclue (elle pilote le
        // phugoïde lent, pas le mode court période linéarisé ici).
        let alpha_dot = q + (vb.x * f_body.z - vb.z * f_body.x) / (p.mass * va * va);
        (alpha_dot, q_dot)
    };

    let h = 1e-6;
    let (base_a, base_q) = f(alpha_trim, 0.0);
    let (da_a, da_q) = f(alpha_trim + h, 0.0);
    let (dq_a, dq_q) = f(alpha_trim, h);
    // Matrice d'état 2×2 A = ∂[α̇,q̇]/∂[α,q] par différences finies avant (base = valeur au
    // trim, da_* = α perturbé de +h, dq_* = q perturbé de +h).
    let a00 = (da_a - base_a) / h;
    let a01 = (dq_a - base_a) / h;
    let a10 = (da_q - base_q) / h;
    let a11 = (dq_q - base_q) / h;
    // Invariants de A : les valeurs propres d'une 2×2 ne dépendent que de sa trace et de son
    // déterminant (polynôme caractéristique λ² − tr·λ + det = 0). Le signe du discriminant
    // tr²−4·det décide : ≥ 0 ⇒ deux racines réelles (une > 0 = divergence pure, l'instabilité
    // statique) ; < 0 ⇒ paire complexe (oscillation, divergente si tr > 0). Le bloc ci-dessous
    // applique λ = (tr ± √disc)/2.
    let tr = a00 + a11;
    let det = a00 * a11 - a01 * a10;
    let discriminant = tr * tr - 4.0 * det;
    let eig = if discriminant >= 0.0 {
        let sqrt_disc = Float::sqrt(discriminant);
        [((tr + sqrt_disc) / 2.0, 0.0), ((tr - sqrt_disc) / 2.0, 0.0)]
    } else {
        let sqrt_disc = Float::sqrt(-discriminant);
        [(tr / 2.0, sqrt_disc / 2.0), (tr / 2.0, -sqrt_disc / 2.0)]
    };
    ShortPeriodModes { eig }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::{Integrator, Rk4};
    use crate::plant::rigid_body_deriv;

    fn aero() -> FixedWingParams {
        FixedWingParams::aerosonde()
    }

    fn deriv(s: &State13, p: &FixedWingParams, c: &FixedWingControls) -> fsim_core::StateDeriv {
        rigid_body_deriv(
            s,
            &fixedwing_wrench(s, p, c, Vec3::zeros(), gravity_world()),
            p.mass,
            &p.inertia,
            &p.inertia_inv,
        )
    }

    fn step_open(p: &FixedWingParams, mut s: State13, c: &FixedWingControls, n: usize) -> State13 {
        let rk4 = Rk4;
        for _ in 0..n {
            s = rk4.step(&s, |x| deriv(x, p, c), 1e-3);
        }
        s
    }

    // --- T-Stab: the typo-catcher (static + damping stability signs) ---
    #[test]
    fn stability_derivative_signs() {
        let p = aero();
        assert!(p.cm_alpha < 0.0, "pitch stiffness");
        assert!(p.cl_beta < 0.0, "dihedral");
        assert!(p.cn_beta > 0.0, "weathercock");
        assert!(p.cm_q < 0.0 && p.clp < 0.0 && p.cnr < 0.0, "damping");
        assert!(p.cl_alpha > 0.0 && p.cy_beta < 0.0);
        assert!(p.cm_de < 0.0, "elevator authority");
    }

    // --- T-AirData ---
    #[test]
    fn air_data_is_correct() {
        let p = aero();
        // Body x along North, small climb component: build from a known v_body.
        let s = trim_state(25.0, 0.0, 0.1); // alpha=0.1 by construction
        let v_body = s.attitude.inverse() * s.velocity;
        assert!((v_body.norm() - 25.0).abs() < 1e-9, "Va");
        assert!(
            (Float::atan2(v_body.z, v_body.x) - 0.1).abs() < 1e-9,
            "alpha"
        );
        // Va=0 guard.
        let z = State13::at_rest();
        let wr = fixedwing_wrench(
            &z,
            &p,
            &FixedWingControls::zero(),
            Vec3::zeros(),
            gravity_world(),
        );
        assert!(wr.force_world.iter().all(|f| f.is_finite()));
    }

    // --- T-LiftSign / T-DragSign ---
    #[test]
    fn lift_and_drag_signs() {
        let p = aero();
        // Level body, airflow along +x with a small +alpha (w>0).
        let mut s = State13::at_rest();
        s.velocity = Vec3::new(25.0, 0.0, 1.25); // alpha ~ +2.86°, body==world here
        let c = FixedWingControls {
            throttle: 0.0,
            ..FixedWingControls::zero()
        };
        let wr = fixedwing_wrench(&s, &p, &c, Vec3::zeros(), gravity_world());
        let f_body = s.attitude.inverse() * (wr.force_world - gravity_world() * p.mass);
        assert!(
            f_body.z < 0.0,
            "lift should push body -z (up): {}",
            f_body.z
        );
        assert!(
            f_body.x < 0.0,
            "drag should push body -x (aft): {}",
            f_body.x
        );
    }

    // --- T-Stall ---
    #[test]
    fn lift_curve_stalls() {
        let p = aero();
        let cl = |alpha: Real| {
            let mut s = State13::at_rest();
            s.velocity = Vec3::new(25.0 * Float::cos(alpha), 0.0, 25.0 * Float::sin(alpha));
            s.attitude = UnitQuaternion::identity();
            let wr = fixedwing_wrench(
                &s,
                &p,
                &FixedWingControls::zero(),
                Vec3::zeros(),
                gravity_world(),
            );
            // Recover CL from the body-frame normal force.
            let f_body = s.attitude.inverse() * (wr.force_world - gravity_world() * p.mass);
            let qbar = 0.5 * p.rho * 25.0 * 25.0;
            // f_z = -D sin - L cos ; with throttle 0, fx has no thrust.
            let (sa, ca) = (Float::sin(alpha), Float::cos(alpha));
            (-f_body.x * sa - f_body.z * ca) / (qbar * p.s)
        };
        let cl_peak = cl(p.alpha_stall);
        let cl_post = cl(p.alpha_stall + 0.15);
        assert!(
            cl_post < cl_peak,
            "CL should drop past stall: {cl_peak} -> {cl_post}"
        );
        // σ monotone in the relevant range.
        assert!(p.sigma(0.0) < 0.1 && p.sigma(p.alpha_stall) > 0.4 && p.sigma(1.0) > 0.9);
    }

    // --- T-PitchBreak: Cm tracks the linear law below stall, then breaks
    // nose-down past it (the stall pitch break). ---
    #[test]
    fn pitch_moment_breaks_nose_down() {
        let p = aero();
        // Recover Cm from the body pitching moment at airspeed 25, zero rates and
        // surfaces, so only the static + break terms remain.
        let cm = |alpha: Real| {
            let mut s = State13::at_rest();
            s.velocity = Vec3::new(25.0 * Float::cos(alpha), 0.0, 25.0 * Float::sin(alpha));
            s.attitude = UnitQuaternion::identity();
            let wr = fixedwing_wrench(
                &s,
                &p,
                &FixedWingControls::zero(),
                Vec3::zeros(),
                gravity_world(),
            );
            let qbar = 0.5 * p.rho * 25.0 * 25.0;
            wr.moment_body.y / (qbar * p.s * p.c)
        };
        // Below stall the break is dormant: Cm matches the linear cm0 + cm_α·α.
        let a_lin = 0.1;
        let cm_linear = p.cm0 + p.cm_alpha * a_lin;
        assert!(
            (cm(a_lin) - cm_linear).abs() < 1e-3,
            "Cm should be linear below stall: {} vs {cm_linear}",
            cm(a_lin)
        );
        // Past stall it breaks *below* the linear extrapolation (nose-down).
        let a_post = p.alpha_stall + 0.2;
        let cm_linear_post = p.cm0 + p.cm_alpha * a_post;
        assert!(
            cm(a_post) < cm_linear_post - 0.05,
            "Cm should break nose-down past stall: {} vs linear {cm_linear_post}",
            cm(a_post)
        );
    }

    // --- T-Thrust ---
    #[test]
    fn thrust_model() {
        let p = aero();
        let t = |va: Real, dt: Real| {
            let kt = p.kmotor * dt;
            (0.5 * p.rho * p.sprop * p.cprop * (kt * kt - va * va)).max(0.0)
        };
        assert!(t(0.0, 1.0) > 0.0, "static thrust positive");
        assert!(t(25.0, 1.0) < t(0.0, 1.0), "thrust drops with airspeed");
        assert_eq!(t(200.0, 0.1), 0.0, "clamped at zero");
    }

    // --- T-Trim ---
    #[test]
    fn trim_is_an_equilibrium() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let d = deriv(&tr.state, &p, &tr.controls);
        let a_body = tr.state.attitude.inverse() * d.d_velocity;
        assert!(a_body.x.abs() < 1e-6, "forward accel {}", a_body.x);
        assert!(a_body.z.abs() < 1e-6, "normal accel {}", a_body.z);
        assert!(
            d.d_angular_rate.y.abs() < 1e-6,
            "pitch accel {}",
            d.d_angular_rate.y
        );
        let (_, theta, _) = tr.state.attitude.euler_angles();
        assert!(
            (0.02..0.18).contains(&theta),
            "alpha/theta out of range: {theta}"
        );
        assert!(tr.controls.throttle > 0.0 && tr.controls.throttle < 1.0);
    }

    // An infeasible request (below the ~14 m/s stall speed) must report failure,
    // not a spurious equilibrium with a pinned surface.
    #[test]
    fn infeasible_trim_returns_none() {
        let p = aero();
        assert!(
            trim(&p, 12.0, 0.0).is_none(),
            "below stall speed should not trim"
        );
        assert!(trim(&p, 25.0, 0.0).is_some(), "cruise should trim");
    }

    // --- T-Cruise: the master sign-error catcher ---
    #[test]
    fn trimmed_cruise_holds() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let s = step_open(&p, tr.state, &tr.controls, 30_000); // 30 s open loop
        let va = s.velocity.norm();
        assert!((va - 25.0).abs() < 0.5, "airspeed drifted: {va}");
        assert!(
            s.position.z.abs() < 5.0,
            "altitude drifted: {} m",
            -s.position.z
        );
        let (roll, _, _) = s.attitude.euler_angles();
        assert!(roll.abs() < 0.01, "rolled: {roll}");
    }

    // --- T-Glide: throttle 0 descends (catches thrust-axis flip) ---
    #[test]
    fn power_off_glides_down() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let c = FixedWingControls {
            throttle: 0.0,
            ..tr.controls
        };
        let s = step_open(&p, tr.state, &c, 5_000); // 5 s
        assert!(
            s.position.z > 0.1,
            "should sink (NED +z down): {}",
            s.position.z
        );
    }

    // --- Control-derivative sign tests (one step from trim) ---
    #[test]
    fn elevator_pitches_nose_down() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let c = FixedWingControls {
            elevator: tr.controls.elevator + 0.1,
            ..tr.controls
        };
        assert!(
            deriv(&tr.state, &p, &c).d_angular_rate.y < -1e-4,
            "+elevator -> nose down"
        );
    }

    #[test]
    fn aileron_rolls_right() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let c = FixedWingControls {
            aileron: 0.1,
            ..tr.controls
        };
        assert!(
            deriv(&tr.state, &p, &c).d_angular_rate.x > 1e-4,
            "+aileron -> roll right"
        );
    }

    #[test]
    fn rudder_yaws_left() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).expect("nominal trim converges");
        let c = FixedWingControls {
            rudder: 0.1,
            ..tr.controls
        };
        assert!(
            deriv(&tr.state, &p, &c).d_angular_rate.z < -1e-4,
            "+rudder -> yaw left"
        );
    }

    // --- T-LateralFade: past stall the aileron loses authority and the roll
    // damping reverses sign (autorotation). Recovered straight from the roll
    // moment with one lateral input active at a time. ---
    #[test]
    fn lateral_authority_fades_and_roll_damping_reverses_past_stall() {
        let p = aero();
        // cl_roll recovered from the body roll moment at airspeed 25, with only the
        // requested roll rate / aileron active (everything else zero).
        let cl_roll = |alpha: Real, pp: Real, aileron: Real| {
            let mut s = State13::at_rest();
            s.velocity = Vec3::new(25.0 * Float::cos(alpha), 0.0, 25.0 * Float::sin(alpha));
            s.attitude = UnitQuaternion::identity();
            s.angular_rate = Vec3::new(pp, 0.0, 0.0);
            let c = FixedWingControls {
                aileron,
                ..FixedWingControls::zero()
            };
            let wr = fixedwing_wrench(&s, &p, &c, Vec3::zeros(), gravity_world());
            wr.moment_body.x / (0.5 * p.rho * 25.0 * 25.0 * p.s * p.b)
        };
        let bf = p.b / (2.0 * 25.0);
        // Roll damping: stabilizing (negative) below stall, reversed (positive,
        // anti-damping) past it.
        let clp_below = cl_roll(0.05, 1.0, 0.0) / bf;
        let clp_above = cl_roll(0.8, 1.0, 0.0) / bf;
        assert!(
            clp_below < 0.0,
            "roll damping stable below stall: {clp_below}"
        );
        assert!(
            clp_above > 0.0,
            "roll damping must reverse (autorotation) past stall: {clp_above}"
        );
        // Aileron authority: present below stall, faded ~to nothing past it.
        let da_below = cl_roll(0.05, 0.0, 0.1) / 0.1;
        let da_above = cl_roll(0.8, 0.0, 0.1) / 0.1;
        assert!(
            da_below > 0.0,
            "aileron rolls right below stall: {da_below}"
        );
        assert!(
            da_above.abs() < 0.1 * da_below.abs(),
            "aileron authority must fade past stall: {da_below} -> {da_above}"
        );
    }

    // --- T-Spin (mechanism): the roll acceleration shares the sign of the roll
    // rate past stall (a dropped wing keeps dropping) but opposes it in cruise
    // (damping). This is the autorotation feedback that seeds an incipient spin,
    // tested at the derivative so it's free of trajectory confounds. ---
    #[test]
    fn roll_self_amplifies_past_stall_but_damps_in_cruise() {
        let p = aero();
        let roll_accel = |alpha: Real, pp: Real| {
            let mut s = State13::at_rest();
            s.velocity = Vec3::new(25.0 * Float::cos(alpha), 0.0, 25.0 * Float::sin(alpha));
            s.attitude = UnitQuaternion::identity();
            s.angular_rate = Vec3::new(pp, 0.0, 0.0);
            deriv(&s, &p, &FixedWingControls::zero()).d_angular_rate.x
        };
        let pp = 0.2;
        // Cruise: roll damping is restoring (opposes the rate).
        assert!(roll_accel(0.05, pp) < 0.0, "roll should damp in cruise");
        // Past stall: the acceleration shares the sign of the rate, either wing.
        assert!(
            roll_accel(0.8, pp) > 0.0,
            "roll should self-amplify past stall"
        );
        assert!(
            roll_accel(0.8, -pp) < 0.0,
            "...and symmetrically for the other wing"
        );
    }

    // --- T-Spin (boundedness): the case where the anti-damped roll could actually
    // run away is the relaxed fighter — FCS off it pitch-diverges *into* the stall
    // and departs, holding high α with the reversed roll damping live. That
    // departure must stay finite and physically bounded at the 1 ms step, never a
    // numerical blow-up. (The stable Aerosonde instead pitches back out of the
    // stall almost at once — it is spin-resistant — so it can't exercise this.) ---
    #[test]
    fn fighter_departure_past_stall_stays_bounded() {
        let p = fighter();
        let tr = trim(&p, 25.0, 0.0).unwrap();
        let mut s = tr.state;
        s.angular_rate += Vec3::new(0.1, 0.1, 0.0); // roll + pitch upset
        let c = tr.controls; // frozen at trim — pure airframe, no control law
        let rk4 = Rk4;
        let mut peak = 0.0;
        let mut reached_stall = false;
        for _ in 0..4_000 {
            // 4 s
            s = rk4.step(&s, |x| deriv(x, &p, &c), 1e-3);
            assert!(
                s.angular_rate.iter().all(|v| v.is_finite()),
                "departure diverged to a non-finite rate"
            );
            peak = Float::max(peak, s.angular_rate.norm());
            let vb = s.attitude.inverse() * s.velocity;
            if Float::atan2(vb.z, vb.x).abs() > p.alpha_stall {
                reached_stall = true;
            }
        }
        assert!(
            reached_stall,
            "the relaxed fighter should depart past the stall angle"
        );
        assert!(
            peak < 50.0,
            "the post-stall departure must stay physically bounded: peak {peak}"
        );
    }

    // --- T-Inertia: non-diagonal Jxz couples roll/yaw, conserves |L| ---
    // Exercise the shared EOM with the fixed-wing's full (Jxz≠0) inertia under a
    // genuinely torque-free wrench (zero force + moment), so gravity/aero can't
    // confound it — the fixed-wing analogue of the quad's asymmetric-body test.
    #[test]
    fn jxz_couples_and_conserves_momentum() {
        let p = aero();
        let mut s = State13::at_rest();
        s.angular_rate = Vec3::new(2.0, 0.0, 0.5); // initial tumble
        let free = Wrench {
            force_world: Vec3::zeros(),
            moment_body: Vec3::zeros(),
        };
        let l0 = (s.attitude * (p.inertia * s.angular_rate)).norm();
        let rk4 = Rk4;
        let mut out = s;
        for _ in 0..4_000 {
            out = rk4.step(
                &out,
                |x| rigid_body_deriv(x, &free, p.mass, &p.inertia, &p.inertia_inv),
                5e-4,
            );
        }
        let l1 = (out.attitude * (p.inertia * out.angular_rate)).norm();
        assert!(
            (l1 - l0).abs() / l0 < 1e-6,
            "|L| not conserved: {l0} -> {l1}"
        );
        // Jxz cross term precesses the body rate (it changed).
        assert!(
            (out.angular_rate - s.angular_rate).norm() > 1e-3,
            "Jxz should couple axes"
        );
    }

    fn fighter() -> FixedWingParams {
        FixedWingParams::fighter_relaxed()
    }

    // The relaxed-stability fighter is the Aerosonde with ONE coefficient flipped
    // (the static margin): aft CG, `cm_alpha` positive, everything else identical.
    #[test]
    fn fighter_is_aerosonde_with_aft_cg() {
        let (a, f) = (aero(), fighter());
        assert!(f.cm_alpha > 0.0, "relaxed stability: cm_alpha must be ≥ 0");
        assert_eq!(a.mass, f.mass, "same mass");
        assert_eq!(a.cl_alpha, f.cl_alpha, "same lift slope");
        assert_eq!(a.cm_de, f.cm_de, "same elevator authority");
        assert_eq!(a.cm_q, f.cm_q, "same pitch damping");
        assert_eq!(a.inertia, f.inertia, "same inertia");
    }

    // An unstable airframe is still an *equilibrium*: the Newton trim converges.
    #[test]
    fn fighter_still_trims() {
        let p = fighter();
        let tr = trim(&p, 25.0, 0.0).expect("relaxed-stability airframe still trims");
        let d = deriv(&tr.state, &p, &tr.controls);
        let a_body = tr.state.attitude.inverse() * d.d_velocity;
        assert!(a_body.x.abs() < 1e-6 && a_body.z.abs() < 1e-6, "force trim");
        assert!(d.d_angular_rate.y.abs() < 1e-6, "pitch-accel trim");
    }

    // --- Eigenvalue proof (the "not faked" check) ---
    // OPEN loop: the fighter's short period has a right-half-plane pole, and its
    // time-to-double is sub-second (faster than any human pitch reflex).
    #[test]
    fn fighter_open_loop_short_period_diverges() {
        let p = fighter();
        let tr = trim(&p, 25.0, 0.0).unwrap();
        let modes = short_period_modes(&p, &tr, |_, _| tr.controls.elevator);
        assert!(
            modes.max_real() > 0.0,
            "open-loop short period must be unstable (RHP): {:?}",
            modes.eig
        );
        let t2 = modes.time_to_double();
        assert!(
            t2 > 0.0 && t2 < 1.0,
            "time-to-double {t2} s should be sub-second"
        );
    }

    // The control case: the conventional Aerosonde's short period is in the LHP.
    #[test]
    fn aerosonde_open_loop_short_period_is_stable() {
        let p = aero();
        let tr = trim(&p, 25.0, 0.0).unwrap();
        let modes = short_period_modes(&p, &tr, |_, _| tr.controls.elevator);
        assert!(
            modes.is_stable(),
            "stable UAV's short period must be in the LHP: {:?}",
            modes.eig
        );
    }

    // CLOSED loop: folding the fly-by-wire SAS feedback
    //   δe = δe_trim + Kα·(α − α_trim) + Kq·q
    // (the same gains `FbwConfig::fighter` uses) into the state matrix pulls both
    // eigenvalues into the LHP. This proves the *physics* — that this feedback
    // shape stabilizes — with explicit gains; `fsim-control` additionally drives
    // `short_period_modes` with the **real** `FlyByWire::step` so the proof tracks
    // the shipped law and its gain schedule, not hand-coded constants.
    #[test]
    fn fbw_feedback_stabilizes_the_short_period() {
        let p = fighter();
        let tr = trim(&p, 25.0, 0.0).unwrap();
        let (k_alpha, k_q) = (1.0, 0.2);
        let de_trim = tr.controls.elevator;
        let vb = tr.state.attitude.inverse() * tr.state.velocity;
        let alpha_trim = Float::atan2(vb.z, vb.x);
        let modes = short_period_modes(&p, &tr, |a, q| {
            de_trim + k_alpha * (a - alpha_trim) + k_q * q
        });
        assert!(
            modes.is_stable(),
            "FBW SAS must pull the short period into the LHP: {:?}",
            modes.eig
        );
    }

    // --- Behavioural divergence: the same pitch nudge that the stable UAV damps
    // makes the relaxed-stability airframe run away. An A/B comparison (rather
    // than an absolute threshold) keeps it robust to the nonlinear stall that
    // eventually caps the runaway rate. ---
    #[test]
    fn fighter_diverges_where_aerosonde_recovers() {
        let kick = 0.05; // a gentle 0.05 rad/s pitch kick
        let departure = |p: &FixedWingParams| {
            let tr = trim(p, 25.0, 0.0).unwrap();
            let mut s = tr.state;
            s.angular_rate.y += kick;
            // Controls frozen at trim — no control law, just the airframe.
            step_open(p, s, &tr.controls, 1200).angular_rate.y.abs() // 1.2 s
        };
        let stable = departure(&aero());
        let relaxed = departure(&fighter());
        assert!(
            stable < kick,
            "stable UAV should damp the kick: q = {stable}"
        );
        assert!(
            relaxed > 4.0 * kick,
            "relaxed airframe should diverge: q = {relaxed}"
        );
        assert!(
            relaxed > 8.0 * stable,
            "relaxed ({relaxed}) must run away far past the stable case ({stable})"
        );
    }
}
