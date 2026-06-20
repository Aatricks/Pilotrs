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
}

impl Atmosphere {
    pub fn new(cfg: AtmosphereConfig) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(cfg.seed),
            gust: Vec3::zeros(),
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

    /// The current total wind \[m/s, NED\] at airspeed `va`: the steady wind plus
    /// one advanced step of the Gauss–Markov gust. Call **once per physics step**
    /// (hold it constant across the RK4 sub-steps).
    pub fn wind_ned(&mut self, va: Real, dt: Real) -> Vec3 {
        if self.cfg.turbulence > 0.0 {
            let va_s = Float::max(va, self.cfg.va_min);
            let sigma = self.cfg.turbulence;
            for i in 0..3 {
                // Gauss–Markov: g ← a·g + σ·√(1−a²)·N(0,1), a = exp(−dt·V/L).
                // Stationary variance is σ², correlation time L/V — a first-order
                // Dryden gust.
                let l = Float::max(self.cfg.scale_lengths[i], 1.0);
                let a = Float::exp(-dt * va_s / l);
                let n: Real = StandardNormal.sample(&mut self.rng);
                self.gust[i] = a * self.gust[i] + sigma * Float::sqrt(1.0 - a * a) * n;
            }
        }
        self.cfg.wind_ned + self.gust
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
            assert_eq!(atm.wind_ned(25.0, 1e-3), Vec3::zeros());
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
            atm.wind_ned(va, dt);
        }
        let mut sum_sq = [0.0_f64; 3];
        let n = 200_000;
        for _ in 0..n {
            let g = atm.wind_ned(va, dt);
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
            atm.wind_ned(va, dt);
        }
        let mut prev = atm.wind_ned(va, dt).x;
        let (mut num, mut den) = (0.0_f64, 0.0_f64);
        for _ in 0..50_000 {
            let cur = atm.wind_ned(va, dt).x;
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
                last = atm.wind_ned(30.0, 1e-3);
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
            atm.wind_ned(25.0, 1e-2);
        }
        assert!(atm.gust_magnitude() > 0.0);
        atm.set_turbulence(0.0);
        assert_eq!(atm.gust_magnitude(), 0.0);
        assert_eq!(atm.wind_ned(25.0, 1e-2), Vec3::zeros());
    }
}
