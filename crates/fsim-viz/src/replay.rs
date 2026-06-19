//! Viz data source: a live quad/fixed-wing engine or a recorded replay. The
//! non-GUI logic lives here and is unit-tested; `main.rs` only calls
//! `view()` / `telemetry()` / `quad_command()` / `fw_command()` / `tick()`.
//!
//! The render loop reads **one** [`ViewSnapshot`] per frame regardless of which
//! airframe is flying or whether the data is live or replayed. Replay remains
//! quad-only (recordings store the quad `TelemetrySample`).

use fsim_sim::{
    Command, FixedWingControls, FwCommand, FwEngine, FwSample, FwSimConfig, FwSnapshot, Quat, Real,
    Recording, SimConfig, SimEngine, Snapshot, TelemetrySample, Vec3 as NaVec3,
};
use std::sync::Arc;

/// Which airframe a [`ViewSnapshot`] describes — lets the render loop pick
/// geometry (quad body + 4 rotors vs. fixed-wing fuselage + surfaces) and the
/// telemetry window pick its plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AircraftKind {
    Quad,
    FixedWing,
}

/// The single per-frame projection the renderer consumes, regardless of
/// airframe or whether the data is live or replayed. Pure `Copy` data in the
/// sim's NED world frame.
#[derive(Debug, Clone, Copy)]
pub struct ViewSnapshot {
    pub t: Real,
    pub kind: AircraftKind,
    /// World position (NED). nalgebra `f64` — converted to three-d in `main.rs`.
    pub position: NaVec3,
    pub attitude: Quat,
    /// World velocity (NED).
    pub velocity: NaVec3,
    pub waypoint_index: Option<usize>,
    /// Quad motor thrusts \[N\]; zeros for fixed-wing.
    pub motors: [Real; 4],
    /// Fixed-wing control surfaces; `None` for quad.
    pub surfaces: Option<FixedWingControls>,
    /// Flying under manual (pilot-stick) control (fixed-wing fighter).
    pub manual: bool,
    /// Fly-by-wire FCS engaged (only meaningful when `manual`).
    pub fbw_on: bool,
    /// Angle of attack \[rad\] (fixed-wing; 0 for quad).
    pub alpha: Real,
    /// Aerodynamic load factor \[g\] (fixed-wing; 0 for quad).
    pub load_factor: Real,
}

impl ViewSnapshot {
    fn from_quad(s: &Snapshot) -> Self {
        Self {
            t: s.t,
            kind: AircraftKind::Quad,
            position: s.truth.position,
            attitude: s.truth.attitude,
            velocity: s.truth.velocity,
            waypoint_index: s.waypoint_index,
            motors: s.motors,
            surfaces: None,
            manual: false,
            fbw_on: false,
            alpha: 0.0,
            load_factor: 0.0,
        }
    }

    fn from_fixedwing(s: &FwSnapshot) -> Self {
        Self {
            t: s.t,
            kind: AircraftKind::FixedWing,
            position: s.truth.position,
            attitude: s.truth.attitude,
            velocity: s.truth.velocity,
            waypoint_index: s.waypoint_index,
            motors: [0.0; 4],
            surfaces: Some(s.controls),
            manual: s.manual,
            fbw_on: s.fbw_on,
            alpha: s.alpha,
            load_factor: s.load_factor,
        }
    }

    /// Altitude above the planet surface \[m\]: `−z` for the quad (flat local
    /// NED), `|p| − R` for the planet-centered fixed-wing.
    pub fn altitude(&self) -> Real {
        match self.kind {
            AircraftKind::Quad => -self.position.z,
            AircraftKind::FixedWing => fsim_sim::planet::altitude_of(self.position),
        }
    }

    /// Attitude relative to the **local horizon** (`q_localNED_from_body`). The
    /// quad's stored attitude already is local; the fixed-wing's PCI attitude is
    /// composed with the local-NED frame at its current position.
    pub fn local_attitude(&self) -> Quat {
        match self.kind {
            AircraftKind::Quad => self.attitude,
            AircraftKind::FixedWing => {
                fsim_sim::planet::ned_from_pci(self.position) * self.attitude
            }
        }
    }

    /// Local-frame velocity (NED) — the fixed-wing's PCI velocity rotated into
    /// the local horizon; the quad's is already local.
    fn local_velocity(&self) -> NaVec3 {
        match self.kind {
            AircraftKind::Quad => self.velocity,
            AircraftKind::FixedWing => {
                fsim_sim::planet::ned_from_pci(self.position) * self.velocity
            }
        }
    }

    /// Course over ground χ \[rad\] in the local NED frame, derived from the
    /// local velocity. Falls back to the local heading when nearly stationary
    /// (quad hover) so the minimap aircraft marker still points somewhere sane.
    pub fn course(&self) -> Real {
        let v = self.local_velocity();
        if v.x.hypot(v.y) < 0.1 {
            let (_, _, yaw) = self.local_attitude().euler_angles();
            yaw
        } else {
            v.y.atan2(v.x)
        }
    }
}

/// Telemetry for the plotting window, tagged by airframe. The quad path keeps
/// the existing `TelemetrySample` plots; the fixed-wing path carries `FwSample`
/// for an airspeed/altitude/course window.
pub enum ViewTelemetry {
    Quad(Arc<Vec<TelemetrySample>>),
    FixedWing(Arc<Vec<FwSample>>),
}

/// Playback state over a loaded (quad) recording (pure index/time math).
/// Replay remains quad-only for now (recordings store `TelemetrySample`).
pub struct ReplayState {
    samples: Arc<Vec<TelemetrySample>>,
    t: f64,
    t0: f64,
    t1: f64,
    pub playing: bool,
    pub speed: f64,
}

impl ReplayState {
    pub fn new(rec: Recording) -> Self {
        let t0 = rec.samples.first().map(|s| s.t).unwrap_or(0.0);
        let t1 = rec.samples.last().map(|s| s.t).unwrap_or(0.0);
        Self {
            samples: Arc::new(rec.samples),
            t: t0,
            t0,
            t1,
            playing: true,
            speed: 1.0,
        }
    }

    /// Advance the playhead by a wall-clock frame interval, stopping at the end.
    pub fn advance(&mut self, dt_frame: f64) {
        if self.playing {
            self.t = (self.t + dt_frame * self.speed).min(self.t1);
            if self.t >= self.t1 {
                self.playing = false;
            }
        }
    }

    pub fn seek(&mut self, t: f64) {
        self.t = t.clamp(self.t0, self.t1);
        self.playing = false;
    }

    pub fn time(&self) -> f64 {
        self.t
    }

    pub fn range(&self) -> (f64, f64) {
        (self.t0, self.t1)
    }

    /// The recorded quad sample at the playhead (latest with `s.t <= t`).
    fn current_quad(&self) -> Option<Snapshot> {
        let count = self.samples.partition_point(|s| s.t <= self.t);
        self.samples
            .get(count.saturating_sub(1))
            .map(Snapshot::from_telemetry_sample)
    }

    fn view(&self) -> ViewSnapshot {
        self.current_quad()
            .map(|s| ViewSnapshot::from_quad(&s))
            .unwrap_or_else(|| {
                ViewSnapshot::from_quad(&Snapshot::initial(&SimConfig::quad_250_mvp()))
            })
    }

    pub fn samples(&self) -> Arc<Vec<TelemetrySample>> {
        Arc::clone(&self.samples)
    }
}

/// Where the viewer's data comes from. Three live/replay variants; the render
/// loop only ever sees a [`ViewSnapshot`].
pub enum Source {
    LiveQuad(SimEngine),
    LiveFixedWing(FwEngine),
    Replay(ReplayState),
}

impl Source {
    /// Spawn a live quad engine.
    pub fn live_quad(cfg: SimConfig) -> Self {
        Source::LiveQuad(SimEngine::spawn_realtime(cfg))
    }

    /// Spawn a live fixed-wing engine.
    pub fn live_fixedwing(cfg: FwSimConfig) -> Self {
        Source::LiveFixedWing(FwEngine::spawn_realtime(cfg))
    }

    pub fn is_replay(&self) -> bool {
        matches!(self, Source::Replay(_))
    }

    pub fn kind(&self) -> AircraftKind {
        match self {
            Source::LiveQuad(_) | Source::Replay(_) => AircraftKind::Quad,
            Source::LiveFixedWing(_) => AircraftKind::FixedWing,
        }
    }

    /// Advance time (a no-op for the live engines, which pace themselves).
    pub fn tick(&mut self, dt_frame: f64) {
        if let Source::Replay(r) = self {
            r.advance(dt_frame);
        }
    }

    /// The single per-frame projection the renderer consumes.
    pub fn view(&self) -> ViewSnapshot {
        match self {
            Source::LiveQuad(e) => ViewSnapshot::from_quad(&e.latest()),
            Source::LiveFixedWing(e) => ViewSnapshot::from_fixedwing(&e.latest()),
            Source::Replay(r) => r.view(),
        }
    }

    /// Telemetry tagged by airframe for the plotting window.
    pub fn telemetry(&self) -> ViewTelemetry {
        match self {
            Source::LiveQuad(e) => ViewTelemetry::Quad(e.telemetry()),
            Source::LiveFixedWing(e) => ViewTelemetry::FixedWing(e.telemetry()),
            Source::Replay(r) => ViewTelemetry::Quad(r.samples()),
        }
    }

    /// Forward a quad command (ignored unless this is a live quad).
    pub fn quad_command(&self, cmd: Command) {
        if let Source::LiveQuad(e) = self {
            let _ = e.send(cmd);
        }
    }

    /// Forward a fixed-wing command (ignored unless this is a live fixed-wing).
    pub fn fw_command(&self, cmd: FwCommand) {
        if let Source::LiveFixedWing(e) = self {
            let _ = e.send(cmd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_sim::{Sim, SimConfig};

    fn recording() -> Recording {
        let mut sim = Sim::new(SimConfig::quad_250_mvp());
        sim.set_logging(10, None);
        sim.run_headless(1000); // ~1 s
        sim.recording()
    }

    #[test]
    fn replay_advances_and_stops_at_end() {
        let rec = recording();
        let t1 = rec.duration();
        let mut r = ReplayState::new(rec);
        assert!(r.playing);
        for _ in 0..100000 {
            r.advance(0.001);
        }
        assert!(!r.playing, "should stop at end");
        assert!((r.time() - t1).abs() < 1e-9, "playhead at end");
    }

    #[test]
    fn replay_seek_clamps_and_pauses() {
        let mut r = ReplayState::new(recording());
        let (t0, t1) = r.range();
        r.seek(t1 + 100.0);
        assert!((r.time() - t1).abs() < 1e-9);
        r.seek(t0 - 100.0);
        assert!((r.time() - t0).abs() < 1e-9);
        assert!(!r.playing, "seek pauses");
    }

    #[test]
    fn replay_view_is_quad_and_tracks_playhead() {
        let mut r = ReplayState::new(recording());
        let (t0, t1) = r.range();
        r.seek(t0);
        let v0 = r.view();
        assert_eq!(v0.kind, AircraftKind::Quad);
        r.seek(t1);
        let v1 = r.view();
        assert!(v1.t >= v0.t, "later playhead → later sample");
    }
}
