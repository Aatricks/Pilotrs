//! Viz data source: a live [`SimEngine`] or a recorded replay. The non-GUI
//! logic lives here and is unit-tested; `main.rs` only calls `snapshot()` /
//! `telemetry()` / `command()` / `tick()`.

use fsim_sim::{Command, Recording, SimConfig, SimEngine, Snapshot, TelemetrySample};
use std::sync::Arc;

/// Playback state over a loaded recording (pure index/time math).
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

    /// The recorded sample at the playhead (latest with `s.t <= t`).
    pub fn current(&self) -> Option<Snapshot> {
        let count = self.samples.partition_point(|s| s.t <= self.t);
        self.samples
            .get(count.saturating_sub(1))
            .map(Snapshot::from_telemetry_sample)
    }

    pub fn samples(&self) -> Arc<Vec<TelemetrySample>> {
        Arc::clone(&self.samples)
    }
}

/// Where the viewer's data comes from.
pub enum Source {
    Live(SimEngine),
    Replay(ReplayState),
}

impl Source {
    pub fn live(cfg: SimConfig) -> Self {
        Source::Live(SimEngine::spawn_realtime(cfg))
    }

    pub fn is_replay(&self) -> bool {
        matches!(self, Source::Replay(_))
    }

    /// Advance time (a no-op for the live engine, which paces itself).
    pub fn tick(&mut self, dt_frame: f64) {
        if let Source::Replay(r) = self {
            r.advance(dt_frame);
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        match self {
            Source::Live(e) => e.latest(),
            Source::Replay(r) => r
                .current()
                .unwrap_or_else(|| Snapshot::initial(&SimConfig::quad_250_mvp())),
        }
    }

    pub fn telemetry(&self) -> Arc<Vec<TelemetrySample>> {
        match self {
            Source::Live(e) => e.telemetry(),
            Source::Replay(r) => r.samples(),
        }
    }

    /// Forward a command to the live engine (ignored in replay).
    pub fn command(&self, cmd: Command) {
        if let Source::Live(e) = self {
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
        let (_, t1) = (0.0, rec.duration());
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
    fn replay_current_tracks_playhead() {
        let mut r = ReplayState::new(recording());
        let (t0, t1) = r.range();
        r.seek(t0);
        let s0 = r.current().unwrap();
        r.seek(t1);
        let s1 = r.current().unwrap();
        assert!(s1.t >= s0.t, "later playhead → later sample");
    }
}
