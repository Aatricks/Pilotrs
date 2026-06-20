//! Threaded simulation engine: runs the deterministic [`Sim`] on its own
//! worker thread, publishing the latest [`Snapshot`] for a consumer (the
//! viewer) to read without blocking the physics, and accepting [`Command`]s
//! over a channel. This is where Rust earns its place — a fixed-step,
//! GC-pause-free physics loop decoupled from rendering.
//!
//! Two run modes: `Realtime` paces to the wall clock (the step *count* per
//! second varies, but each step's math has no clock, so individual steps stay
//! deterministic); `FixedSteps` runs an exact number of steps on the thread and
//! is **bit-for-bit identical** to [`Sim::run_headless`].

use crate::atmosphere::StormCell;
use crate::config::{EstimatorKind, SimConfig};
use crate::guidance::Waypoint;
use crate::telemetry::{Telemetry, TelemetrySample};
use crate::{GuidanceConfig, Sim};
use fsim_core::{EstState, Real, Setpoint, State13, Tick, Vec3};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// An immutable, `Copy` projection of the simulator state for the consumer.
/// Small (~370 B) so publishing it every worker iteration is cheap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snapshot {
    pub t: Real,
    pub tick: Tick,
    pub truth: State13,
    pub estimate: EstState,
    pub setpoint: Setpoint,
    pub motors: [Real; 4],
    pub true_gyro_bias: Vec3,
    pub est_gyro_bias: Vec3,
    /// Whether the estimator reports a gyro bias (false for the CF).
    pub has_bias: bool,
    pub waypoint_index: Option<usize>,
    /// Steady wind speed \[m/s\] (for the HUD).
    pub wind_speed: Real,
    /// Instantaneous turbulence gust magnitude \[m/s\] (for the HUD).
    pub gust: Real,
    /// Storm proximity (0 = clear air, 1 = microburst core).
    pub storm: Real,
    pub paused: bool,
    pub recording: bool,
    /// Publish counter — advances every iteration, even when paused (distinct
    /// from `tick`, which only advances when the physics steps).
    pub seq: u64,
}

impl Snapshot {
    fn from_sim(sim: &Sim, seq: u64, paused: bool, recording: bool) -> Self {
        let eb = sim.est_gyro_bias();
        Self {
            t: sim.time(),
            tick: sim.tick(),
            truth: *sim.truth(),
            estimate: sim.estimate(),
            setpoint: sim.setpoint(),
            motors: sim.motors(),
            true_gyro_bias: sim.true_gyro_bias(),
            est_gyro_bias: eb.unwrap_or_else(Vec3::zeros),
            has_bias: eb.is_some(),
            waypoint_index: sim.waypoint_index(),
            wind_speed: sim.wind_speed(),
            gust: sim.gust(),
            storm: sim.storm_intensity(),
            paused,
            recording,
            seq,
        }
    }

    /// A snapshot of a freshly-built sim (seeds the cell before step 1, so
    /// `latest()` never has to wait for the first publish).
    pub fn initial(cfg: &SimConfig) -> Self {
        Self::from_sim(&Sim::new(*cfg), 0, false, false)
    }

    /// Build a snapshot from a recorded [`TelemetrySample`] (for replay). The
    /// engine-only fields (`tick`/`seq`/`waypoint_index`) are synthesized.
    pub fn from_telemetry_sample(s: &TelemetrySample) -> Self {
        Self {
            t: s.t,
            tick: 0,
            truth: s.truth,
            estimate: s.estimate,
            setpoint: s.setpoint,
            motors: s.motors,
            true_gyro_bias: s.true_gyro_bias,
            est_gyro_bias: s.est_gyro_bias,
            has_bias: s.est_gyro_bias != Vec3::zeros(),
            waypoint_index: None,
            wind_speed: 0.0, // recordings predate the weather model
            gust: 0.0,
            storm: 0.0,
            paused: false,
            recording: false,
            seq: 0,
        }
    }
}

/// A latest-value cell: single in-place memcpy under an uncontended,
/// poison-recovering lock. Encapsulated so the primitive can be swapped (e.g.
/// for `triple_buffer`) without touching callers.
struct LatestCell<T: Copy>(Mutex<T>);

impl<T: Copy> LatestCell<T> {
    fn new(init: T) -> Self {
        Self(Mutex::new(init))
    }
    fn publish(&self, v: T) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = v;
    }
    fn load(&self) -> T {
        *self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Commands sent to the engine's worker thread.
#[derive(Debug, Clone)]
pub enum Command {
    SetSetpoint(Setpoint),
    SetMission {
        waypoints: Vec<Waypoint>,
        guidance: GuidanceConfig,
    },
    /// Return to attitude mode, holding the current setpoint.
    SetAttitudeMode,
    /// Set the steady wind \[m/s, local NED\].
    SetWind(Vec3),
    /// Set the turbulence intensity (RMS gust \[m/s\]; 0 = calm).
    SetTurbulence(Real),
    /// Place (or clear) the storm / microburst cell.
    SetStorm(Option<StormCell>),
    Pause(bool),
    /// Real-time speed multiplier (clamped to [0, 16]); ignored in fixed-step.
    SetSpeed(f64),
    /// Rebuild the sim from a new config (tick/time reset to 0). Boxed because
    /// `SimConfig` is large relative to the other variants.
    Reset(Box<SimConfig>),
    /// Toggle recording (retain the full, uncapped telemetry from now).
    Record(bool),
    /// Flush the current recording to a `.fsimrec` file.
    SaveRecording(PathBuf),
    Shutdown,
}

/// Logging configuration for the engine's rolling telemetry window.
#[derive(Debug, Clone, Copy)]
pub struct LoggingCfg {
    pub log_every: Tick,
    pub cap: Option<usize>,
}

impl Default for LoggingCfg {
    fn default() -> Self {
        Self {
            log_every: 5,
            cap: Some(4000),
        }
    }
}

/// How the worker drives the sim.
#[derive(Debug, Clone, Copy)]
pub enum RunMode {
    /// Wall-clock paced; the step *count* per second is not reproducible.
    Realtime { speed: f64 },
    /// Exactly `total` steps, then idle — bit-for-bit `Sim::run_headless`.
    FixedSteps { total: u64 },
}

/// Returned when the engine has already exited (the command was not delivered).
#[derive(Debug, Clone, Copy)]
pub struct EngineClosed;

/// Summary returned when the worker joins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunReport {
    pub steps_taken: Tick,
    pub final_tick: Tick,
    pub rejected_missions: u64,
}

struct Shared {
    latest: LatestCell<Snapshot>,
    telem: Mutex<Arc<Vec<TelemetrySample>>>,
}

/// Handle to a sim running on a worker thread. `Send`, single-consumer (not
/// `Clone`/`Sync`).
pub struct SimEngine {
    shared: Arc<Shared>,
    tx: Sender<Command>,
    join: Option<JoinHandle<(RunReport, Telemetry)>>,
}

impl SimEngine {
    /// Spawn the worker with the given config, run mode, and logging.
    pub fn spawn(cfg: SimConfig, mode: RunMode, logging: LoggingCfg) -> SimEngine {
        let shared = Arc::new(Shared {
            latest: LatestCell::new(Snapshot::initial(&cfg)),
            telem: Mutex::new(Arc::new(Vec::new())),
        });
        let (tx, rx) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let join = thread::spawn(move || run_worker(cfg, mode, logging, worker_shared, rx));
        SimEngine {
            shared,
            tx,
            join: Some(join),
        }
    }

    /// Spawn a real-time engine (speed 1×, default rolling window).
    pub fn spawn_realtime(cfg: SimConfig) -> SimEngine {
        Self::spawn(cfg, RunMode::Realtime { speed: 1.0 }, LoggingCfg::default())
    }

    /// The latest published snapshot (never blocks the physics thread).
    pub fn latest(&self) -> Snapshot {
        self.shared.latest.load()
    }

    /// The current rolling telemetry window (O(1) `Arc` clone).
    pub fn telemetry(&self) -> Arc<Vec<TelemetrySample>> {
        Arc::clone(&self.shared.telem.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Send a command; `Err` if the engine has already exited.
    pub fn send(&self, cmd: Command) -> Result<(), EngineClosed> {
        self.tx.send(cmd).map_err(|_| EngineClosed)
    }

    /// Stop the worker and join, returning its report.
    pub fn shutdown(mut self) -> RunReport {
        self.shutdown_inner().0
    }

    /// Stop the worker and join, returning its report **and** final telemetry
    /// (used by the determinism test).
    pub fn shutdown_with_telemetry(mut self) -> (RunReport, Telemetry) {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> (RunReport, Telemetry) {
        let _ = self.tx.send(Command::Shutdown);
        match self.join.take() {
            Some(h) => h.join().unwrap_or_else(|_| {
                (
                    RunReport {
                        steps_taken: 0,
                        final_tick: 0,
                        rejected_missions: 0,
                    },
                    Telemetry::new(),
                )
            }),
            None => (
                RunReport {
                    steps_taken: 0,
                    final_tick: 0,
                    rejected_missions: 0,
                },
                Telemetry::new(),
            ),
        }
    }
}

impl Drop for SimEngine {
    fn drop(&mut self) {
        if self.join.is_some() {
            // Reap the worker: no detached threads, no panic on disconnect.
            let _ = self.tx.send(Command::Shutdown);
            if let Some(h) = self.join.take() {
                let _ = h.join();
            }
        }
    }
}

struct Worker {
    sim: Sim,
    kind: EstimatorKind,
    dt: Real,
    logging: LoggingCfg,
    paused: bool,
    speed: f64,
    recording: bool,
    seq: u64,
    rejected: u64,
    shared: Arc<Shared>,
}

impl Worker {
    fn publish(&mut self) {
        self.seq += 1;
        self.shared.latest.publish(Snapshot::from_sim(
            &self.sim,
            self.seq,
            self.paused,
            self.recording,
        ));
    }

    fn publish_telemetry(&self) {
        let arc = Arc::new(self.sim.telemetry().samples.clone());
        *self.shared.telem.lock().unwrap_or_else(|e| e.into_inner()) = arc;
    }

    /// Apply a command; returns true if the worker should shut down.
    fn handle(&mut self, cmd: Command) -> bool {
        match cmd {
            Command::SetSetpoint(sp) => self.sim.set_setpoint(sp),
            Command::SetMission {
                waypoints,
                guidance,
            } => {
                // Position mode needs the INS; reject (count) otherwise so the
                // worker never trips Sim::set_mission's debug-assert.
                if self.kind == EstimatorKind::Ins {
                    self.sim.set_mission(waypoints, guidance);
                } else {
                    self.rejected += 1;
                }
            }
            Command::SetAttitudeMode => {
                let sp = self.sim.setpoint();
                self.sim.set_setpoint(sp);
            }
            Command::SetWind(w) => self.sim.set_wind(w),
            Command::SetTurbulence(rms) => self.sim.set_turbulence(rms),
            Command::SetStorm(s) => self.sim.set_storm(s),
            Command::Pause(p) => self.paused = p,
            Command::SetSpeed(s) => self.speed = s.clamp(0.0, 16.0),
            Command::Reset(cfg) => {
                let cfg = *cfg;
                self.sim = Sim::new(cfg);
                self.sim
                    .set_logging(self.logging.log_every, self.logging.cap);
                self.kind = cfg.estimator_kind;
                self.dt = cfg.dt;
                self.paused = false;
                self.recording = false;
            }
            Command::Record(r) => {
                self.recording = r;
                if r {
                    // Retain everything from now (uncapped) for a faithful record.
                    self.sim.set_logging(self.logging.log_every, None);
                }
            }
            Command::SaveRecording(path) => {
                let _ = self.sim.recording().save(path);
            }
            Command::Shutdown => return true,
        }
        false
    }

    /// Drain all currently-available commands (non-blocking). Returns true on
    /// shutdown.
    fn drain(&mut self, rx: &Receiver<Command>) -> bool {
        while let Ok(cmd) = rx.try_recv() {
            if self.handle(cmd) {
                return true;
            }
        }
        false
    }

    fn run_realtime(&mut self, rx: &Receiver<Command>) {
        let mut last = Instant::now();
        let mut accum = 0.0_f64;
        let mut last_telem = Instant::now();
        loop {
            // Block up to 1 ms (paces idle wake-ups, gives instant command
            // latency), then drain the rest.
            match rx.recv_timeout(Duration::from_millis(1)) {
                Ok(cmd) => {
                    if self.handle(cmd) {
                        return;
                    }
                    if self.drain(rx) {
                        return;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return,
            }

            let now = Instant::now();
            let elapsed = (now - last).as_secs_f64().min(0.25); // spiral-of-death clamp
            last = now;
            if !self.paused {
                accum += elapsed * self.speed;
                // Cap the catch-up to discard excess backlog; subtract only what
                // was actually stepped and clamp the residual to >= 0 so a cap
                // never injects negative time-debt (which would freeze the sim).
                let n = ((accum / self.dt) as u64).min(250);
                for _ in 0..n {
                    self.sim.step();
                }
                accum = (accum - n as f64 * self.dt).max(0.0);
            }
            self.publish();
            if last_telem.elapsed() >= Duration::from_millis(50) {
                self.publish_telemetry();
                last_telem = now;
            }
        }
    }

    fn run_fixed(&mut self, rx: &Receiver<Command>, total: u64) {
        let mut done: u64 = 0;
        while done < total {
            if self.drain(rx) {
                return;
            }
            if self.paused {
                self.publish();
                thread::sleep(Duration::from_millis(1));
                continue;
            }
            let batch = (total - done).min(50);
            for _ in 0..batch {
                self.sim.step();
            }
            done += batch;
            self.publish();
        }
        // Reached `total`: publish the final telemetry, then idle until told to
        // stop (keeping the final state visible).
        self.publish_telemetry();
        loop {
            match rx.recv() {
                Ok(cmd) => {
                    if self.handle(cmd) {
                        return;
                    }
                    self.publish();
                }
                Err(_) => return, // disconnected → shut down
            }
        }
    }
}

fn run_worker(
    cfg: SimConfig,
    mode: RunMode,
    logging: LoggingCfg,
    shared: Arc<Shared>,
    rx: Receiver<Command>,
) -> (RunReport, Telemetry) {
    let mut sim = Sim::new(cfg);
    sim.set_logging(logging.log_every, logging.cap);
    let speed = match mode {
        RunMode::Realtime { speed } => speed.clamp(0.0, 16.0),
        RunMode::FixedSteps { .. } => 1.0,
    };
    let mut w = Worker {
        sim,
        kind: cfg.estimator_kind,
        dt: cfg.dt,
        logging,
        paused: false,
        speed,
        recording: false,
        seq: 0,
        rejected: 0,
        shared,
    };
    w.publish(); // seed seq=1 snapshot before the loop
    match mode {
        RunMode::Realtime { .. } => w.run_realtime(&rx),
        RunMode::FixedSteps { total } => w.run_fixed(&rx, total),
    }
    let report = RunReport {
        steps_taken: w.sim.tick(),
        final_tick: w.sim.tick(),
        rejected_missions: w.rejected,
    };
    let telem = w.sim.telemetry().clone();
    (report, telem)
}

// --- static guarantees (no data races) ---
fn _assert_send_sync() {
    fn send<T: Send>() {}
    fn sync<T: Sync>() {}
    fn copy<T: Copy>() {}
    send::<Snapshot>();
    sync::<Snapshot>();
    copy::<Snapshot>();
    send::<Command>();
    send::<Sim>();
    send::<SimEngine>();
}
// Keep the linter from flagging the assertion helper as dead.
const _: fn() = _assert_send_sync;

#[cfg(test)]
mod tests {
    use super::*;
    use fsim_core::Quat;

    fn deadline_poll<F: FnMut() -> bool>(mut cond: F, secs: f64) -> bool {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < secs {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(2));
        }
        false
    }

    #[test]
    fn command_round_trip_setpoint() {
        let eng = SimEngine::spawn_realtime(SimConfig::quad_250_mvp());
        let sp = Setpoint {
            attitude: Quat::from_euler_angles(0.1, -0.05, 0.2),
            thrust: 5.0,
        };
        eng.send(Command::SetSetpoint(sp)).unwrap();
        let got = deadline_poll(|| eng.latest().setpoint == sp, 2.0);
        assert!(got, "setpoint command not reflected in snapshot");
        eng.shutdown();
    }

    #[test]
    fn snapshot_advances_then_pause_freezes_tick() {
        let eng = SimEngine::spawn_realtime(SimConfig::quad_250_mvp());
        let s0 = eng.latest();
        assert!(
            deadline_poll(|| eng.latest().tick > s0.tick, 2.0),
            "tick did not advance"
        );
        eng.send(Command::Pause(true)).unwrap();
        // Let the pause take effect, then confirm tick is frozen while seq advances.
        thread::sleep(Duration::from_millis(50));
        let a = eng.latest();
        thread::sleep(Duration::from_millis(50));
        let b = eng.latest();
        assert_eq!(a.tick, b.tick, "tick advanced while paused");
        assert!(b.seq > a.seq, "seq should advance even when paused");
        eng.shutdown();
    }

    #[test]
    fn reset_restarts_tick() {
        let eng = SimEngine::spawn_realtime(SimConfig::quad_250_mvp());
        assert!(deadline_poll(|| eng.latest().tick > 100, 2.0));
        eng.send(Command::Reset(Box::default())).unwrap(); // == quad_250_mvp
        assert!(
            deadline_poll(|| eng.latest().tick < 100, 2.0),
            "tick did not reset"
        );
        eng.shutdown();
    }

    #[test]
    fn mission_rejected_without_ins() {
        // MEKF (not INS) → SetMission must be rejected (counted), not panic.
        let eng = SimEngine::spawn_realtime(SimConfig::quad_250_m2());
        eng.send(Command::SetMission {
            waypoints: vec![Waypoint::new(Vec3::new(1.0, 0.0, -2.0), 0.0)],
            guidance: GuidanceConfig::default(),
        })
        .unwrap();
        thread::sleep(Duration::from_millis(100));
        assert_eq!(
            eng.latest().waypoint_index,
            None,
            "mission accepted without INS"
        );
        let report = eng.shutdown();
        assert_eq!(report.rejected_missions, 1);
    }

    #[test]
    fn speed_zero_does_not_advance() {
        let eng = SimEngine::spawn_realtime(SimConfig::quad_250_mvp());
        eng.send(Command::SetSpeed(0.0)).unwrap();
        thread::sleep(Duration::from_millis(50));
        let a = eng.latest();
        thread::sleep(Duration::from_millis(80));
        let b = eng.latest();
        assert_eq!(a.tick, b.tick, "advanced at speed 0");
        assert!(
            b.seq > a.seq,
            "seq should still advance (no spin starvation)"
        );
        eng.shutdown();
    }

    #[test]
    fn clean_join_on_drop_and_disconnect() {
        // Dropping the handle must reap the worker (no hang, no panic).
        {
            let eng = SimEngine::spawn_realtime(SimConfig::quad_250_mvp());
            assert!(deadline_poll(|| eng.latest().tick > 10, 2.0));
        } // drop here joins
          // If we got here without hanging, the worker was reaped.
    }

    #[test]
    fn fixedsteps_matches_run_headless() {
        // The threaded fixed-step path is bit-for-bit identical to inline.
        let cfg = SimConfig::quad_250_m2();
        let total = 4000u64;

        let mut inline = Sim::new(cfg);
        inline.set_logging(10, None);
        inline.run_headless(total as usize);
        let want = inline.telemetry().samples.clone();

        let eng = SimEngine::spawn(
            cfg,
            RunMode::FixedSteps { total },
            LoggingCfg {
                log_every: 10,
                cap: None,
            },
        );
        assert!(
            deadline_poll(|| eng.latest().tick >= total, 5.0),
            "fixed run did not finish"
        );
        let (_report, telem) = eng.shutdown_with_telemetry();

        assert_eq!(telem.samples.len(), want.len(), "sample count differs");
        for (a, b) in telem.samples.iter().zip(&want) {
            assert_eq!(a, b, "threaded fixed-step diverged from run_headless");
        }
    }
}
