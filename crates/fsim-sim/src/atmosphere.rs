//! Atmosphere: steady wind + turbulence, the air the aircraft actually flies
//! through.
//!
//! The aerodynamic plant consumes a single **wind vector in the aircraft's world
//! frame** — `fixedwing_wrench(.., wind_world, ..)` forms the air-relative body
//! velocity `q⁻¹·(velocity − wind_world)`, so a wind that matches the aircraft's
//! motion produces zero airspeed. This module supplies that vector.
//!
//! It works entirely in the **local North-East-Down** frame (so "wind from the
//! north at 10 m/s" is `wind_ned = (−10, 0, 0)` — wind *blows toward* −N); the sim
//! rotates the result into the world frame (PCI on the sphere, flat NED for the
//! quad). Turbulence is a **first-order Dryden approximation**: each NED axis is a
//! seeded **Gauss–Markov** process whose stationary RMS is the configured gust
//! intensity and whose correlation time is `scale_length / airspeed`. Like the
//! sensors, it is driven by a seeded `ChaCha8` stream, so a run is reproducible
//! bit-for-bit.

use fsim_core::{Real, Vec3};
use num_traits::Float;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// A storm / microburst cell — a localized hazard centred on a point in the
/// local map frame `(north, east)` \[m\] from home, optionally drifting.
///
/// Near the centre it stacks three effects, each falling off as a Gaussian in
/// the horizontal distance `r` from the core: a **downdraft** (sinking air), a
/// radial **outflow** (the diverging surface wind a microburst is infamous for,
/// peaking near the cell radius), and a **turbulence boost** (the air gets rough
/// inside the cell). Fly through one and the FBW fighter earns its keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StormCell {
    /// Cell centre at `t = 0`: `(north, east)` \[m\] from home.
    pub center: (Real, Real),
    /// Drift velocity of the centre `(north, east)` \[m/s\].
    pub velocity: (Real, Real),
    /// Horizontal scale (Gaussian 1/e radius) \[m\].
    pub radius: Real,
    /// Peak downdraft speed at the core \[m/s\] (NED `+z` = down).
    pub downdraft: Real,
    /// Peak radial outflow speed \[m/s\].
    pub outflow: Real,
    /// Extra turbulence RMS \[m/s\] added at the core.
    pub turbulence_boost: Real,
}

impl StormCell {
    /// A punchy microburst at map point `(north, east)` \[m\], stationary.
    pub fn microburst(center: (Real, Real)) -> Self {
        Self {
            center,
            velocity: (0.0, 0.0),
            radius: 400.0,
            downdraft: 10.0,
            outflow: 8.0,
            turbulence_boost: 4.0,
        }
    }
}

/// Steady wind, turbulence intensity, and the turbulence length scales.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphereConfig {
    /// Steady wind in the local NED frame \[m/s\] — the velocity the air moves at
    /// (`(−10,0,0)` is a 10 m/s wind *from* the north).
    pub wind_ned: Vec3,
    /// Turbulence intensity: the stationary RMS gust speed \[m/s\] (0 = calm).
    /// Rough guide: ~1 light, ~3 moderate, ~6 severe.
    pub turbulence: Real,
    /// Dryden length scales \[m\] for the (horizontal, horizontal, vertical) axes.
    /// Vertical gusts are shorter-scale (choppier) than horizontal.
    pub scale_lengths: Vec3,
    /// Airspeed floor for the gust time constant (avoids divide-by-zero at rest).
    pub va_min: Real,
    /// An optional storm / microburst cell layered on top of the steady field.
    pub storm: Option<StormCell>,
    /// RNG seed for the turbulence stream (independent of the sensor streams).
    pub seed: u64,
}

impl AtmosphereConfig {
    /// Dead calm: no wind, no turbulence. The default — every existing run stays
    /// bit-for-bit unchanged (the gust process is not even advanced when calm).
    pub fn calm() -> Self {
        Self {
            wind_ned: Vec3::zeros(),
            turbulence: 0.0,
            scale_lengths: Vec3::new(200.0, 200.0, 50.0),
            va_min: 1.0,
            storm: None,
            seed: 0xA1_4036_0000_0001,
        }
    }

    /// A steady wind \[m/s, NED\] plus a turbulence RMS \[m/s\], default scales.
    pub fn wind(wind_ned: Vec3, turbulence: Real) -> Self {
        Self {
            wind_ned,
            turbulence,
            ..Self::calm()
        }
    }
}

/// The flying atmosphere: holds the turbulence filter state + its RNG.
#[derive(Debug, Clone)]
pub struct Atmosphere {
    cfg: AtmosphereConfig,
    rng: ChaCha8Rng,
    /// Current gust velocity \[m/s, NED\] (the Gauss–Markov state).
    gust: Vec3,
    /// Elapsed sim time \[s\] (drives a moving storm cell).
    time: Real,
    /// Storm proximity at the last sample (Gaussian factor 0..1; 0 = no storm).
    last_g: Real,
}

impl Atmosphere {
    pub fn new(cfg: AtmosphereConfig) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(cfg.seed),
            gust: Vec3::zeros(),
            time: 0.0,
            last_g: 0.0,
            cfg,
        }
    }

    pub fn config(&self) -> &AtmosphereConfig {
        &self.cfg
    }

    /// Set the steady wind \[m/s, NED\] (leaves turbulence + RNG untouched).
    pub fn set_wind(&mut self, wind_ned: Vec3) {
        self.cfg.wind_ned = wind_ned;
    }

    /// Set the turbulence RMS \[m/s\]. Setting it to 0 also clears the gust so the
    /// air goes still immediately.
    pub fn set_turbulence(&mut self, rms: Real) {
        self.cfg.turbulence = rms.max(0.0);
        if self.cfg.turbulence == 0.0 {
            self.gust = Vec3::zeros();
        }
    }

    /// Place (or clear) the storm cell.
    pub fn set_storm(&mut self, storm: Option<StormCell>) {
        self.cfg.storm = storm;
    }

    /// The current total wind \[m/s, NED\] for an aircraft at map position
    /// `(pos_n, pos_e)` \[m from home\], airspeed `va`: steady wind + storm cell
    /// (downdraft + outflow) + one advanced step of the Gauss–Markov gust (whose
    /// intensity is boosted inside the storm). Call **once per physics step**
    /// (hold it constant across the RK4 sub-steps).
    pub fn wind_ned(&mut self, pos_n: Real, pos_e: Real, va: Real, dt: Real) -> Vec3 {
        self.time += dt;
        let mut wind = self.cfg.wind_ned;
        let mut turb = self.cfg.turbulence;
        self.last_g = 0.0;

        if let Some(s) = self.cfg.storm {
            // Cell centre at this time (it may drift); Gaussian falloff in r.
            let cn = s.center.0 + s.velocity.0 * self.time;
            let ce = s.center.1 + s.velocity.1 * self.time;
            let (dn, de) = (pos_n - cn, pos_e - ce);
            let r2 = dn * dn + de * de;
            let rad = Float::max(s.radius, 1.0);
            let g = Float::exp(-r2 / (rad * rad));
            self.last_g = g;
            // Downdraft (NED +z = down) sinks the air at the core.
            wind.z += s.downdraft * g;
            // Radial outflow: diverging horizontal wind, peaking near the radius.
            let r = Float::sqrt(r2);
            if r > 1e-3 {
                let out = s.outflow * (r / rad) * g;
                wind.x += out * dn / r;
                wind.y += out * de / r;
            }
            turb += s.turbulence_boost * g;
        }

        if turb > 0.0 {
            let va_s = Float::max(va, self.cfg.va_min);
            for i in 0..3 {
                // Gauss–Markov: g ← a·g + σ·√(1−a²)·N(0,1), a = exp(−dt·V/L).
                // Stationary variance is σ², correlation time L/V — a first-order
                // Dryden gust. `turb` is boosted inside the storm.
                let l = Float::max(self.cfg.scale_lengths[i], 1.0);
                let a = Float::exp(-dt * va_s / l);
                let n: Real = StandardNormal.sample(&mut self.rng);
                self.gust[i] = a * self.gust[i] + turb * Float::sqrt(1.0 - a * a) * n;
            }
        }
        wind + self.gust
    }

    /// How deep in the storm the aircraft was at the last sample (0 = clear air,
    /// 1 = dead centre) — for a HUD warning.
    pub fn storm_intensity(&self) -> Real {
        self.last_g
    }

    /// The storm cell's current centre `(north, east)` \[m\], if active (for the
    /// HUD / map marker).
    pub fn storm_center(&self) -> Option<(Real, Real)> {
        self.cfg.storm.map(|s| {
            (
                s.center.0 + s.velocity.0 * self.time,
                s.center.1 + s.velocity.1 * self.time,
            )
        })
    }

    /// The steady wind speed \[m/s\] (for a HUD readout).
    pub fn wind_speed(&self) -> Real {
        self.cfg.wind_ned.norm()
    }

    /// The instantaneous gust magnitude \[m/s\] (for a HUD readout).
    pub fn gust_magnitude(&self) -> Real {
        self.gust.norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Calm air is exactly still and never touches the RNG, so existing runs stay
    // bit-for-bit identical.
    #[test]
    fn calm_is_still() {
        let mut atm = Atmosphere::new(AtmosphereConfig::calm());
        for _ in 0..1000 {
            assert_eq!(atm.wind_ned(0.0, 0.0, 25.0, 1e-3), Vec3::zeros());
        }
        assert_eq!(atm.gust_magnitude(), 0.0);
    }

    // The Gauss–Markov gust's stationary RMS matches the configured intensity
    // (the σ·√(1−a²) scaling is what makes this true regardless of dt/scale).
    #[test]
    fn turbulence_rms_matches_intensity() {
        let sigma = 3.0;
        let mut atm = Atmosphere::new(AtmosphereConfig::wind(Vec3::zeros(), sigma));
        let (va, dt) = (25.0, 1e-2);
        // Burn in past the correlation time, then accumulate variance.
        for _ in 0..2000 {
            atm.wind_ned(0.0, 0.0, va, dt);
        }
        let mut sum_sq = [0.0_f64; 3];
        let n = 200_000;
        for _ in 0..n {
            let g = atm.wind_ned(0.0, 0.0, va, dt);
            for i in 0..3 {
                sum_sq[i] += g[i] * g[i];
            }
        }
        for (i, s) in sum_sq.iter().enumerate() {
            let rms = (s / n as f64).sqrt();
            assert!(
                (rms - sigma).abs() < 0.3,
                "axis {i} RMS {rms} should be ≈ σ {sigma}"
            );
        }
    }

    // Gusts are correlated in time (not white noise): the lag-1 autocorrelation is
    // high for the chosen scale/airspeed. Catches a missing filter (pure noise).
    #[test]
    fn gusts_are_time_correlated() {
        let mut atm = Atmosphere::new(AtmosphereConfig::wind(Vec3::zeros(), 3.0));
        let (va, dt) = (25.0, 1e-2);
        for _ in 0..2000 {
            atm.wind_ned(0.0, 0.0, va, dt);
        }
        let mut prev = atm.wind_ned(0.0, 0.0, va, dt).x;
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for _ in 0..50_000 {
            let cur = atm.wind_ned(0.0, 0.0, va, dt).x;
            num += prev * cur;
            den += prev * prev;
            prev = cur;
        }
        let autocorr = num / den;
        // a = exp(−dt·V/L) = exp(−0.01·25/200) ≈ 0.9988 — strongly correlated.
        assert!(
            autocorr > 0.9,
            "successive gusts should be correlated, got {autocorr}"
        );
    }

    // Same seed + same calls ⇒ identical gust sequence (determinism).
    #[test]
    fn turbulence_is_deterministic() {
        let run = || {
            let mut atm = Atmosphere::new(AtmosphereConfig::wind(Vec3::new(-5.0, 2.0, 0.0), 4.0));
            let mut last = Vec3::zeros();
            for _ in 0..5000 {
                last = atm.wind_ned(0.0, 0.0, 30.0, 1e-3);
            }
            last
        };
        assert_eq!(run(), run());
    }

    // Setting turbulence to zero stills the air immediately (gust cleared).
    #[test]
    fn set_turbulence_zero_clears_gust() {
        let mut atm = Atmosphere::new(AtmosphereConfig::wind(Vec3::zeros(), 5.0));
        for _ in 0..500 {
            atm.wind_ned(0.0, 0.0, 25.0, 1e-2);
        }
        assert!(atm.gust_magnitude() > 0.0);
        atm.set_turbulence(0.0);
        assert_eq!(atm.gust_magnitude(), 0.0);
        assert_eq!(atm.wind_ned(0.0, 0.0, 25.0, 1e-2), Vec3::zeros());
    }

    // --- Storm cell ---

    fn storm_atm(center: (Real, Real)) -> Atmosphere {
        Atmosphere::new(AtmosphereConfig {
            storm: Some(StormCell::microburst(center)),
            ..AtmosphereConfig::calm()
        })
    }

    // At the core the air sinks hard (downdraft, NED +z) and the storm reads full
    // intensity; far outside the cell it's calm.
    #[test]
    fn storm_sinks_air_at_the_core() {
        let mut atm = storm_atm((0.0, 0.0));
        let w = atm.wind_ned(0.0, 0.0, 25.0, 1e-2);
        assert!(w.z > 5.0, "downdraft at the core: {}", w.z);
        assert!(
            atm.storm_intensity() > 0.9,
            "core intensity: {}",
            atm.storm_intensity()
        );

        let mut far = storm_atm((0.0, 0.0));
        let wf = far.wind_ned(3000.0, 0.0, 25.0, 1e-2);
        assert!(wf.norm() < 0.1, "far from the storm is calm: {wf:?}");
        assert!(far.storm_intensity() < 0.01);
    }

    // Off-centre, the horizontal wind blows radially outward (a microburst's
    // diverging outflow) — north of the core it pushes further north.
    #[test]
    fn storm_outflow_is_radial() {
        let mut atm = storm_atm((0.0, 0.0));
        let w = atm.wind_ned(200.0, 0.0, 25.0, 1e-2); // 200 m north, inside r=400
        assert!(
            w.x > 1.0,
            "outflow should blow radially out (north): {}",
            w.x
        );
        assert!(w.z > 0.0, "still sinking off-centre: {}", w.z);
    }

    // The storm roughens the air even with no base turbulence.
    #[test]
    fn storm_roughens_the_air() {
        let mut atm = storm_atm((0.0, 0.0));
        for _ in 0..500 {
            atm.wind_ned(0.0, 0.0, 25.0, 1e-2);
        }
        assert!(
            atm.gust_magnitude() > 0.5,
            "storm should add turbulence: {}",
            atm.gust_magnitude()
        );
    }

    // A storm (with base turbulence too) is reproducible bit-for-bit.
    #[test]
    fn storm_is_deterministic() {
        let cfg = AtmosphereConfig {
            storm: Some(StormCell::microburst((100.0, 50.0))),
            turbulence: 1.0,
            ..AtmosphereConfig::calm()
        };
        let run = || {
            let mut atm = Atmosphere::new(cfg);
            let mut w = Vec3::zeros();
            for k in 0..3000 {
                w = atm.wind_ned(k as Real * 0.01, 0.0, 30.0, 1e-2);
            }
            w
        };
        assert_eq!(run(), run());
    }
}
