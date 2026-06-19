//! Fly-by-wire flight control system for a relaxed-stability airframe.
//!
//! The relaxed-stability fighter (see `FixedWingParams::fighter_relaxed` in
//! `fsim-dynamics`) is *unstable open-loop*: its short-period pitch mode has a
//! right-half-plane pole, so it diverges within about a second and no human can
//! fly it directly. This module is what makes it flyable — and the toggle that
//! turns it off is the demonstration of *why* a modern fighter is its control
//! laws.
//!
//! ## Two jobs, one law
//!
//! - **Stability augmentation (SAS):** feed angle of attack and pitch rate back
//!   into the elevator with the sign that synthesizes a *negative* effective
//!   `Cm_alpha` and extra pitch damping — relocating the divergent pole into the
//!   left-half plane. (The closed-loop eigenvalues are checked directly in
//!   `fsim_dynamics::short_period_modes`.)
//! - **Command augmentation (CAS):** the pilot's stick commands a *response*
//!   (a pitch rate, a roll rate), not a raw surface deflection. The loop tracks
//!   the commanded rate while the SAS holds the airframe together.
//!
//! ## Sign conventions (FRD body frame; see [`fsim_core::StickInput`])
//!
//! Elevator authority is *negative* (`Cm_de < 0`: `+elevator` ⇒ nose-down),
//! while aileron authority is *positive* (`Cl_da > 0`: `+aileron` ⇒ roll-right).
//! That asymmetry is why the pitch and roll laws below look different:
//! - `δe = δe_trim + Kα·(α − α_trim) + Kq·(q − q_cmd)` — `+α` (nose-high, the
//!   unstable tendency) adds `+elevator` (nose-down) to correct it; a pitch-rate
//!   below the command adds `−elevator` (nose-up) to chase it.
//! - `δa = Kp·(p_cmd − p)` — a roll rate below the command adds `+aileron`
//!   (roll-right) to chase it.
//!
//! ## Gain scheduling (a relaxed-stability subtlety)
//!
//! Control power scales with dynamic pressure `q̄ = ½ρV²` — but so does the
//! *instability*: `M_α ∝ q̄` exactly as `M_δe ∝ q̄`. The closed-loop stiffness is
//! `Cm_α + Cm_δe·k_α`, a **q̄-independent** condition. So the conventional
//! `∝ 1/q̄` schedule (which keeps bandwidth constant for a *stable* plant) would
//! be fatal here: as `q̄` rises it would shrink `k_α` below the value that makes
//! `Cm_α + Cm_δe·k_α < 0`, and the airframe would diverge at high speed. Instead
//! the schedule is **boost-only**, `clamp(q̄_ref/q̄, 1, …)`: nominal (proven-stable)
//! gains at and above the reference speed, with extra authority when slow.
//!
//! ## The toggle
//!
//! [`FlyByWire::step`] takes an `enabled` flag. `true` runs the SAS+CAS law
//! above (docile, flyable). `false` is a **direct passthrough** — the stick maps
//! straight to surface deflections with no stabilization, so the unstable
//! airframe diverges. That contrast is the whole point.

use crate::Pid;
use fsim_core::{ControlLimits, EstState, FixedWingControls, Real, StickInput};
use num_traits::Float;

/// Gains, command authorities, trim references, and limits for a [`FlyByWire`].
///
/// The SAS gains and the gain-schedule reference are airframe properties; the
/// trim references (`alpha_trim`, `elevator_trim`, `throttle_trim`) come from the
/// dynamics trim solver and are filled in by the sim layer (which sees both
/// crates — `fsim-control` itself stays decoupled from `fsim-dynamics`).
#[derive(Debug, Clone, Copy)]
pub struct FbwConfig {
    /// AoA → elevator feedback (synthesizes static stability). `[1/rad → rad]`.
    pub k_alpha: Real,
    /// Pitch-rate-error → elevator feedback (damping + rate tracking). `[s]`.
    pub k_q: Real,
    /// Roll-rate-error → aileron feedback. `[s]`.
    pub k_roll: Real,
    /// Yaw-rate → rudder yaw damper. `[s]`.
    pub k_yaw_damp: Real,

    /// Max commanded pitch rate at full pitch stick \[rad/s\].
    pub q_max: Real,
    /// Max commanded roll rate at full roll stick \[rad/s\].
    pub p_max: Real,
    /// Max pilot rudder authority at full yaw stick \[rad\].
    pub rudder_max: Real,

    /// Trim angle of attack \[rad\] (the SAS holds AoA toward this).
    pub alpha_trim: Real,
    /// Trim elevator deflection \[rad\] (the SAS feed-forward).
    pub elevator_trim: Real,
    /// Trim throttle \[0,1\] — the neutral reference for the throttle axis.
    pub throttle_trim: Real,

    /// Reference airspeed for gain scheduling \[m/s\].
    pub va_ref: Real,
    /// Air density used to form dynamic pressure \[kg/m³\].
    pub rho: Real,
    /// Airspeed floor in the schedule (avoids divide-by-zero at `Va→0`) \[m/s\].
    pub va_min: Real,
    /// Clamp `(min, max)` on the schedule factor `q̄_ref/q̄`. The minimum is `1.0`
    /// (boost-only): the SAS gain never falls below its proven-stable nominal —
    /// see the module note on the relaxed-stability scheduling subtlety.
    pub sched: (Real, Real),

    /// Surface/throttle limits.
    pub limits: ControlLimits,
}

impl FbwConfig {
    /// Default fly-by-wire tuning for the relaxed-stability fighter. The SAS
    /// gains (`k_alpha`, `k_q`) are the ones proven to pull the short-period mode
    /// into the left-half plane in `fsim_dynamics::short_period_modes`.
    ///
    /// The trim references come from the dynamics trim solver, e.g.
    /// `let tr = trim(&params, va, 0.0).unwrap();` then pass
    /// `tr.controls.elevator` / `tr.controls.throttle` and the trim AoA.
    pub fn fighter(
        alpha_trim: Real,
        elevator_trim: Real,
        throttle_trim: Real,
        va_ref: Real,
        rho: Real,
        limits: ControlLimits,
    ) -> Self {
        Self {
            k_alpha: 1.0,
            k_q: 0.2,
            k_roll: 0.15,
            k_yaw_damp: 0.2,
            q_max: 0.6,
            p_max: 1.5,
            rudder_max: 0.3,
            alpha_trim,
            elevator_trim,
            throttle_trim,
            va_ref,
            rho,
            va_min: 1.0,
            sched: (1.0, 4.0), // boost-only: never drop below nominal (see module note)
            limits,
        }
    }
}

/// The fly-by-wire FCS. Memoryless SAS+CAS today (the `Pid` is reserved for an
/// optional integral trim term); holds only its configuration and the toggle's
/// last state for telemetry.
#[derive(Debug, Clone)]
pub struct FlyByWire {
    cfg: FbwConfig,
    /// Reserved for an optional pitch-rate integrator (not yet wired in); kept so
    /// the law can grow an integral term without an API change.
    _pitch_trim: Pid,
}

impl FlyByWire {
    pub fn new(cfg: FbwConfig) -> Self {
        let sm = cfg.limits.surface_max;
        Self {
            cfg,
            _pitch_trim: Pid::new(0.0, 0.0, 0.0, sm, sm),
        }
    }

    /// The configuration (gains, limits, trim references).
    pub fn config(&self) -> &FbwConfig {
        &self.cfg
    }

    /// Map the pilot's stick to control surfaces.
    ///
    /// `enabled = true` runs the stabilizing SAS+CAS law; `enabled = false` is a
    /// direct passthrough (no stabilization → the unstable airframe diverges).
    /// `dt` is currently unused (the law is memoryless) but kept in the signature
    /// for a future integral term.
    pub fn step(
        &mut self,
        est: &EstState,
        stick: &StickInput,
        _dt: Real,
        enabled: bool,
    ) -> FixedWingControls {
        let stick = stick.clamped();
        let controls = if enabled {
            self.augmented(est, &stick)
        } else {
            self.passthrough(&stick)
        };
        controls.clamp(&self.cfg.limits)
    }

    /// SAS + CAS: stabilize the airframe while tracking the pilot's rate commands.
    ///
    /// ┌─ BLANK-AND-FILL · theory 2 of 2 · the fly-by-wire control law ──────────┐
    /// Reference answer: `reference/fbw.rs.reference`  (gitignored).
    /// Predict in DEVLOG.md BEFORE you read the reference, then implement.
    ///
    /// All the physics you need is in this module's header (signs, the law, the
    /// gain-schedule subtlety). Returns the four [`FixedWingControls`]; the outer
    /// [`step`](Self::step) clamps to the surface/throttle limits.
    ///
    /// BUILD:
    ///  1. Air data from `est`: body velocity `v_body = attitude⁻¹ · velocity`;
    ///     `alpha = atan2(w, u)`; `va = |velocity|`; body rates `p, q, r` straight
    ///     off `est.angular_rate` (x, y, z).
    ///  2. Gain schedule (BOOST-ONLY — read the module note on why the lower clamp
    ///     is 1.0, not <1): `q̄ = ½·ρ·va²` (floor `va` at `va_min`),
    ///     `q̄_ref = ½·ρ·va_ref²`, `sched = clamp(q̄_ref/q̄, sched.0, sched.1)`.
    ///  3. Command augmentation: `q_cmd = q_max·stick.pitch` (+pitch = nose up
    ///     = +q); `p_cmd = p_max·stick.roll` (+roll = roll right = +p).
    ///  4. Surfaces (mind the authority signs — elevator authority is NEGATIVE,
    ///     aileron POSITIVE):
    ///       • `elevator = elevator_trim + sched·(k_alpha·(alpha − alpha_trim)
    ///                      + k_q·(q − q_cmd))`   — SAS stabilizes AoA + tracks rate
    ///       • `aileron  = sched·k_roll·(p_cmd − p)`            — roll-rate tracking
    ///       • `rudder   = −stick.yaw·rudder_max + sched·k_yaw_damp·r`  — pedal +
    ///                                                            yaw damper
    ///       • `throttle = stick.throttle`
    ///
    /// WHY: jobs (1) stabilize the relaxed airframe (synthesize a negative
    /// effective `Cm_alpha` via AoA feedback) and (2) let the pilot command a
    /// *response* (a rate), not a raw deflection. Get a sign wrong and the
    /// closed-loop eigenvalue test in `fsim-control` (`real_fbw_law_*`) goes red.
    /// └─────────────────────────────────────────────────────────────────────────┘
    fn augmented(&self, est: &EstState, stick: &StickInput) -> FixedWingControls {
        let _ = (&self.cfg, est, stick);
        todo!("implement the fly-by-wire SAS+CAS law — see reference/fbw.rs.reference")
    }

    /// FCS OFF: stick straight to surfaces, no stabilization. On the relaxed
    /// airframe this diverges — that is exactly what the toggle demonstrates.
    fn passthrough(&self, stick: &StickInput) -> FixedWingControls {
        let sm = self.cfg.limits.surface_max;
        FixedWingControls {
            elevator: -stick.pitch * sm, // +pitch (nose up) → −elevator (nose up)
            aileron: stick.roll * sm,    // +roll → +aileron (right)
            rudder: -stick.yaw * sm,     // +yaw (right) → −rudder (right)
            throttle: stick.throttle,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::Vec3;
    use nalgebra::UnitQuaternion;

    // A fighter trimmed near 25 m/s level: α_trim ≈ 0.095, δe_trim ≈ 0.01.
    fn cfg() -> FbwConfig {
        FbwConfig::fighter(
            0.095,
            0.01,
            0.5,
            25.0,
            1.2682,
            ControlLimits {
                surface_max: 0.4363,
                throttle: (0.0, 1.0),
            },
        )
    }

    /// An estimate flying at airspeed `va`, angle of attack `alpha`, pitch rate
    /// `q`. Body == world (identity attitude) so `α = atan2(w,u)` reads back the
    /// chosen `alpha` directly and the law is exercised in isolation.
    fn flight(va: Real, alpha: Real, q: Real) -> EstState {
        let v_body = Vec3::new(va * alpha.cos(), 0.0, va * alpha.sin());
        EstState {
            position: Vec3::new(0.0, 0.0, -100.0),
            velocity: v_body, // identity attitude ⇒ world velocity == body velocity
            attitude: UnitQuaternion::identity(),
            angular_rate: Vec3::new(0.0, q, 0.0),
        }
    }

    // SAS: angle of attack above trim adds nose-DOWN elevator (+) to correct the
    // relaxed airframe's nose-up divergence; below trim, nose-up (−).
    #[test]
    fn sas_stabilizes_angle_of_attack() {
        let mut fbw = FlyByWire::new(cfg());
        let neutral = StickInput::neutral();
        let e_high = fbw
            .step(&flight(25.0, 0.3, 0.0), &neutral, 1e-2, true)
            .elevator;
        let e_low = fbw
            .step(&flight(25.0, -0.1, 0.0), &neutral, 1e-2, true)
            .elevator;
        assert!(
            e_high > e_low,
            "more AoA ⇒ more nose-down elevator: {e_high} vs {e_low}"
        );
    }

    // SAS: pitch rate is damped (q>0 nose-up ⇒ +elevator nose-down opposes it).
    #[test]
    fn sas_damps_pitch_rate() {
        let mut fbw = FlyByWire::new(cfg());
        let neutral = StickInput::neutral();
        // At trim AoA so only the rate term differs.
        let base = fbw
            .step(&flight(25.0, 0.095, 0.0), &neutral, 1e-2, true)
            .elevator;
        let damped = fbw
            .step(&flight(25.0, 0.095, 0.5), &neutral, 1e-2, true)
            .elevator;
        assert!(damped > base, "pitch-up rate should add nose-down elevator");
    }

    // CAS: a nose-up pitch command (pull back) produces nose-up elevator (−).
    #[test]
    fn cas_pitch_command_pitches_up() {
        let mut fbw = FlyByWire::new(cfg());
        let e = flight(25.0, 0.095, 0.0); // at trim, q=0
        let base = fbw.step(&e, &StickInput::neutral(), 1e-2, true).elevator;
        let pull = StickInput {
            pitch: 1.0,
            ..StickInput::neutral()
        };
        let pulled = fbw.step(&e, &pull, 1e-2, true).elevator;
        assert!(
            pulled < base,
            "pull back ⇒ nose-up ⇒ less (more negative) elevator: {pulled} vs {base}"
        );
    }

    // CAS: a right-roll command produces right (+) aileron.
    #[test]
    fn cas_roll_command_rolls_right() {
        let mut fbw = FlyByWire::new(cfg());
        let right = StickInput {
            roll: 1.0,
            ..StickInput::neutral()
        };
        assert!(
            fbw.step(&flight(25.0, 0.0, 0.0), &right, 1e-2, true)
                .aileron
                > 0.0,
            "+roll stick ⇒ +aileron (roll right)"
        );
    }

    // The toggle: FCS OFF is a direct passthrough — pull back ⇒ nose-up (−)
    // elevator with NO stabilizing AoA/rate terms (independent of airframe state).
    #[test]
    fn passthrough_is_direct_and_unstabilized() {
        let mut fbw = FlyByWire::new(cfg());
        let pull = StickInput {
            pitch: 1.0,
            ..StickInput::neutral()
        };
        let sm = cfg().limits.surface_max;
        // Same stick, two very different flight states ⇒ identical passthrough.
        let a = fbw.step(&flight(25.0, 0.0, 0.0), &pull, 1e-2, false);
        let b = fbw.step(&flight(40.0, 0.4, 0.6), &pull, 1e-2, false);
        assert_eq!(a, b, "passthrough must ignore the airframe state");
        assert!((a.elevator + sm).abs() < 1e-9, "full pull ⇒ −surface_max");
    }

    // Gain scheduling: boosted below the reference speed, floored at nominal at
    // and above it (the boost-only clamp). A modest AoA keeps the elevator off its
    // limit so surface clamping can't mask the comparison.
    #[test]
    fn gains_scale_up_when_slow_and_floor_at_reference() {
        let mut fbw = FlyByWire::new(cfg());
        let neutral = StickInput::neutral();
        let trim_e = cfg().elevator_trim;
        let mut de = |va: Real| {
            fbw.step(&flight(va, 0.15, 0.0), &neutral, 1e-2, true)
                .elevator
                - trim_e
        };
        let de_slow = de(15.0); // below va_ref ⇒ schedule boosts the gain
        let de_ref = de(25.0); // at va_ref ⇒ schedule factor exactly 1.0
        let de_fast = de(35.0); // above va_ref ⇒ boost-only floor pins it at 1.0
        assert!(
            de_slow.abs() < cfg().limits.surface_max - trim_e.abs(),
            "slow deflection should be unclamped (else the test is hollow): {de_slow}"
        );
        assert!(
            de_slow.abs() > de_ref.abs() * 1.5,
            "slow flight ⇒ boosted gain: {de_slow} vs {de_ref}"
        );
        assert!(
            (de_fast - de_ref).abs() < 1e-12,
            "boost-only floor: at/above va_ref the gain stays nominal: {de_fast} vs {de_ref}"
        );
    }

    // --- Authoritative eigenvalue proof: drive `short_period_modes` with the
    // REAL `FlyByWire::step`, so the closed-loop stability check tracks the
    // shipped law (gains, signs, gain schedule) rather than hand-coded constants.
    // (`fsim-dynamics` is a dev-dependency, so this lives here, not there.) ---

    /// Closed-loop short-period modes of the fighter trimmed at `va`, with the
    /// real fly-by-wire law folded into the linearization.
    fn closed_loop_modes(va: Real) -> fsim_dynamics::ShortPeriodModes {
        let p = fsim_dynamics::FixedWingParams::fighter_relaxed();
        let tr = fsim_dynamics::trim(&p, va, 0.0).expect("fighter trims");
        let vb = tr.state.attitude.inverse() * tr.state.velocity;
        let alpha_trim = vb.z.atan2(vb.x);
        let cfg = FbwConfig::fighter(
            alpha_trim,
            tr.controls.elevator,
            tr.controls.throttle,
            25.0,
            p.rho,
            p.limits,
        );
        let mut fbw = FlyByWire::new(cfg);
        let vtrim = tr.state.velocity.norm();
        // Build the EstState the real law reads (α from body velocity, q from the
        // gyro) at the requested (α, q); identity attitude keeps it isolated.
        let est_at = |a: Real, q: Real| EstState {
            position: Vec3::new(0.0, 0.0, -300.0),
            velocity: Vec3::new(vtrim * a.cos(), 0.0, vtrim * a.sin()),
            attitude: UnitQuaternion::identity(),
            angular_rate: Vec3::new(0.0, q, 0.0),
        };
        fsim_dynamics::short_period_modes(&p, &tr, |a, q| {
            fbw.step(&est_at(a, q), &StickInput::neutral(), 1e-2, true)
                .elevator
        })
    }

    #[test]
    fn real_fbw_law_stabilizes_short_period_at_trim() {
        let modes = closed_loop_modes(25.0);
        assert!(
            modes.is_stable(),
            "the shipped FBW law must pull the short period into the LHP: {:?}",
            modes.eig
        );
    }

    // The boost-only schedule's reason for being: above va_ref the gain floors at
    // nominal and the relaxed airframe stays stable — a conventional ∝1/q̄ schedule
    // would shrink the gain here and let it diverge.
    #[test]
    fn real_fbw_law_stable_above_reference_speed() {
        let modes = closed_loop_modes(40.0);
        assert!(
            modes.is_stable(),
            "FBW must stay stable above the reference speed: {:?}",
            modes.eig
        );
    }
}
