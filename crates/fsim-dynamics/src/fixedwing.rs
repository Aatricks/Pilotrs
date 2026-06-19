//! Fixed-wing aircraft: aerodynamic force/moment model, parameters, and a trim
//! solver (M6). The 6DOF rigid-body EOM and the RK4 integrator are **shared**
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
            cy0: 0.0,
            cy_beta: -0.98,
            cy_p: 0.0,
            cy_r: 0.0,
            cy_da: 0.0,
            cy_dr: 0.17,
            cl0_roll: 0.0,
            cl_beta: -0.12,
            clp: -0.26,
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
    let cl_plate = 2.0 * Float::signum(alpha) * sa * sa * ca;
    let cl = (1.0 - sigma) * cl_lin + sigma * cl_plate + p.cl_q * cf * qq + p.cl_de * c.elevator;
    let cd = p.cd_p
        + cl_lin * cl_lin / (core::f64::consts::PI * p.e_osw * p.ar)
        + p.cd_q * cf * qq
        + p.cd_de * c.elevator;
    let cm = p.cm0 + p.cm_alpha * alpha + p.cm_q * cf * qq + p.cm_de * c.elevator;

    // --- lateral coefficients ---
    let cy = p.cy0
        + p.cy_beta * beta
        + p.cy_p * bf * pp
        + p.cy_r * bf * rr
        + p.cy_da * c.aileron
        + p.cy_dr * c.rudder;
    let cl_roll = p.cl0_roll
        + p.cl_beta * beta
        + p.clp * bf * pp
        + p.clr * bf * rr
        + p.cl_da * c.aileron
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
    let mut fx = -f_drag * ca + f_lift * sa;
    let fz = -f_drag * sa - f_lift * ca;
    let fy = f_y;

    // Propeller thrust along body +x.
    let kt = p.kmotor * c.throttle;
    let thrust = Float::max(0.5 * p.rho * p.sprop * p.cprop * (kt * kt - va * va), 0.0);
    fx += thrust;

    let f_body = Vec3::new(fx, fy, fz);
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
        let a_body = state.attitude.inverse() * d.d_velocity;
        Vec3::new(a_body.x, a_body.z, d.d_angular_rate.y)
    };

    let mut x = Vec3::new(gamma, 0.0, 0.5);
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
            j.set_column(k, &((residual(&xp) - r) / h));
        }
        match j.try_inverse() {
            Some(jinv) => x -= jinv * r,
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
}
