//! Threaded fixed-wing simulation engine: runs the deterministic [`FwSim`] on
//! its own worker thread, publishing the latest [`FwSnapshot`] and accepting
//! [`FwCommand`]s. The fixed-wing analogue of [`SimEngine`](crate::SimEngine):
//! the realtime pacing (accumulator, spiral-of-death clamp, command drain,
//! periodic telemetry publish) is identical so the two engines behave the same
//! from the viewer's point of view.
//!
//! Like the quad engine it has two run modes: `Realtime` paces to the wall
//! clock (the step *count* per second varies, but each step's math has no
//! clock, so individual steps stay deterministic); `FixedSteps` runs an exact
//! number of steps and is **bit-for-bit identical** to [`FwSim::run_headless`].

use crate::engine::EngineClosed;
use crate::fixedwing::{FwSample, FwSim, FwSimConfig};
use crate::fw_guidance::FwGuidanceConfig;
use crate::guidance::Waypoint;
use fsim_control::FixedWingSetpoint;
use fsim_core::{FixedWingControls, Real, State13, StickInput, Tick, Vec3};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// An immutable, `Copy` projection of the fixed-wing sim state for the consumer.
/// Mirrors [`Snapshot`](crate::Snapshot); small enough to publish every
/// iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwSnapshot {
    pub t: Real,
    pub tick: Tick,
    pub truth: State13,
    pub controls: FixedWingControls,
    pub setpoint: FixedWingSetpoint,
    pub waypoint_index: Option<usize>,
    pub airspeed: Real,
    pub altitude: Real,
    pub course: Real,
    /// Angle of attack \[rad\] (for the HUD).
    pub alpha: Real,
    /// Aerodynamic load factor \[g\] (for the HUD).
    pub load_factor: Real,
    /// Flying under manual (pilot-stick) control.
    pub manual: bool,
    /// Fly-by-wire FCS engaged (only meaningful when `manual`).
    pub fbw_on: bool,
    /// Steady wind speed \[m/s\] (for the HUD).
    pub wind_speed: Real,
    /// Instantaneous turbulence gust magnitude \[m/s\] (for the HUD).
    pub gust: Real,
    pub paused: bool,
    /// Publish counter — advances every iteration, even when paused (distinct
    /// from `tick`, which only advances when the physics steps).
    pub seq: u64,
}

impl FwSnapshot {
    fn from_sim(sim: &FwSim, seq: u64, paused: bool) -> Self {
        Self {
            t: sim.time(),
            tick: sim.tick(),
            truth: *sim.truth(),
            controls: sim.controls(),
            setpoint: sim.setpoint(),
            waypoint_index: sim.waypoint_index(),
            airspeed: sim.airspeed(),
            altitude: sim.altitude(),
            course: sim.course(),
            alpha: sim.alpha(),
            load_factor: sim.load_factor(),
            manual: sim.is_manual(),
            fbw_on: sim.fbw_on(),
            wind_speed: sim.wind_speed(),
            gust: sim.gust(),
            paused,
            seq,
        }
    }

    /// A snapshot of a freshly-built sim (seeds the cell before step 1, so
    /// `latest()` never has to wait for the first publish).
    pub fn initial(cfg: &FwSimConfig) -> Self {
        Self::from_sim(&FwSim::new(cfg.clone()), 0, false)
    }
}

/// A latest-value cell: single in-place memcpy under an uncontended,
/// poison-recovering lock. (Mirror of the one in `engine.rs`, kept private.)
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

/// Commands sent to the fixed-wing engine's worker thread.
#[derive(Debug, Clone)]
pub enum FwCommand {
    /// Install a waypoint route follower (NED waypoints + guidance tuning).
    SetRoute {
        waypoints: Vec<Waypoint>,
        cfg: FwGuidanceConfig,
    },
    /// Clear any route and hold this raw airspeed/altitude/course.
    SetCruise(FixedWingSetpoint),
    /// Switch to manual (pilot-stick) control; `bool` is the initial FCS state.
    EnterManual(bool),
    /// Update the pilot's stick demand (manual mode only).
    SetStick(StickInput),
    /// Toggle the fly-by-wire FCS on/off (manual mode only).
    SetFbw(bool),
    /// Set the steady wind \[m/s, local NED\].
    SetWind(Vec3),
    /// Set the turbulence intensity (RMS gust \[m/s\]; 0 = calm).
    SetTurbulence(Real),
    Pause(bool),
    /// Real-time speed multiplier (clamped to [0, 16]); ignored in fixed-step.
    SetSpeed(f64),
    /// Rebuild the sim from a new config (tick/time reset to 0). Boxed because
    /// `FwSimConfig` is large relative to the other variants.
    Reset(Box<FwSimConfig>),
    Shutdown,
}

/// Logging configuration for the engine's rolling telemetry window.
#[derive(Debug, Clone, Copy)]
pub struct FwLoggingCfg {
    pub log_every: Tick,
    pub cap: Option<usize>,
}

impl Default for FwLoggingCfg {
    fn default() -> Self {
        Self {
            log_every: 5,
            cap: Some(4000),
        }
    }
}

/// How the worker drives the sim.
#[derive(Debug, Clone, Copy)]
pub enum FwRunMode {
    /// Wall-clock paced; the step *count* per second is not reproducible.
    Realtime { speed: f64 },
    /// Exactly `total` steps, then idle — bit-for-bit `FwSim::run_headless`.
    FixedSteps { total: u64 },
}

/// Summary returned when the worker joins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FwRunReport {
    pub steps_taken: Tick,
    pub final_tick: Tick,
}

struct Shared {
    latest: LatestCell<FwSnapshot>,
    telem: Mutex<Arc<Vec<FwSample>>>,
}

/// Handle to a fixed-wing sim running on a worker thread. `Send`,
/// single-consumer (not `Clone`/`Sync`).
pub struct FwEngine {
    shared: Arc<Shared>,
    tx: Sender<FwCommand>,
    join: Option<JoinHandle<FwRunReport>>,
}

impl FwEngine {
    /// Spawn the worker with the given config, run mode, and logging.
    pub fn spawn(cfg: FwSimConfig, mode: FwRunMode, logging: FwLoggingCfg) -> FwEngine {
        let shared = Arc::new(Shared {
            latest: LatestCell::new(FwSnapshot::initial(&cfg)),
            telem: Mutex::new(Arc::new(Vec::new())),
        });
        let (tx, rx) = mpsc::channel();
        let worker_shared = Arc::clone(&shared);
        let join = thread::spawn(move || run_worker(cfg, mode, logging, worker_shared, rx));
        FwEngine {
            shared,
            tx,
            join: Some(join),
        }
    }

    /// Spawn a real-time engine (speed 1×, default rolling window).
    pub fn spawn_realtime(cfg: FwSimConfig) -> FwEngine {
        Self::spawn(
            cfg,
            FwRunMode::Realtime { speed: 1.0 },
            FwLoggingCfg::default(),
        )
    }

    /// The latest published snapshot (never blocks the physics thread).
    pub fn latest(&self) -> FwSnapshot {
        self.shared.latest.load()
    }

    /// The current rolling telemetry window (O(1) `Arc` clone).
    pub fn telemetry(&self) -> Arc<Vec<FwSample>> {
        Arc::clone(&self.shared.telem.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Send a command; `Err` if the engine has already exited.
    pub fn send(&self, cmd: FwCommand) -> Result<(), EngineClosed> {
        self.tx.send(cmd).map_err(|_| EngineClosed)
    }

    /// Stop the worker and join, returning its report.
    pub fn shutdown(mut self) -> FwRunReport {
        self.shutdown_inner()
    }

    fn shutdown_inner(&mut self) -> FwRunReport {
        let _ = self.tx.send(FwCommand::Shutdown);
        match self.join.take() {
            Some(h) => h.join().unwrap_or(FwRunReport {
                steps_taken: 0,
                final_tick: 0,
            }),
            None => FwRunReport {
                steps_taken: 0,
                final_tick: 0,
            },
        }
    }
}

impl Drop for FwEngine {
    fn drop(&mut self) {
        if self.join.is_some() {
            // Reap the worker: no detached threads, no panic on disconnect.
            let _ = self.tx.send(FwCommand::Shutdown);
            if let Some(h) = self.join.take() {
                let _ = h.join();
            }
        }
    }
}

struct Worker {
    sim: FwSim,
    dt: Real,
    logging: FwLoggingCfg,
    paused: bool,
    speed: f64,
    seq: u64,
    shared: Arc<Shared>,
}

impl Worker {
    fn publish(&mut self) {
        self.seq += 1;
        self.shared
            .latest
            .publish(FwSnapshot::from_sim(&self.sim, self.seq, self.paused));
    }

    fn publish_telemetry(&self) {
        let arc = Arc::new(self.sim.samples().to_vec());
        *self.shared.telem.lock().unwrap_or_else(|e| e.into_inner()) = arc;
    }

    /// Apply a command; returns true if the worker should shut down.
    fn handle(&mut self, cmd: FwCommand) -> bool {
        match cmd {
            FwCommand::SetRoute { waypoints, cfg } => self.sim.set_route(waypoints, cfg),
            // `set_setpoint` itself cancels any active route (back to setpoint mode).
            FwCommand::SetCruise(sp) => self.sim.set_setpoint(sp),
            FwCommand::EnterManual(fbw_on) => self.sim.enter_manual(fbw_on),
            FwCommand::SetStick(s) => self.sim.set_stick(s),
            FwCommand::SetFbw(on) => self.sim.set_fbw(on),
            FwCommand::SetWind(w) => self.sim.set_wind(w),
            FwCommand::SetTurbulence(rms) => self.sim.set_turbulence(rms),
            FwCommand::Pause(p) => self.paused = p,
            FwCommand::SetSpeed(s) => self.speed = s.clamp(0.0, 16.0),
            FwCommand::Reset(cfg) => {
                let cfg = *cfg;
                self.dt = cfg.dt;
                self.sim = FwSim::new(cfg);
                self.sim
                    .set_logging(self.logging.log_every, self.logging.cap);
                self.paused = false;
            }
            FwCommand::Shutdown => return true,
        }
        false
    }

    /// Drain all currently-available commands (non-blocking). Returns true on
    /// shutdown.
    fn drain(&mut self, rx: &Receiver<FwCommand>) -> bool {
        while let Ok(cmd) = rx.try_recv() {
            if self.handle(cmd) {
                return true;
            }
        }
        false
    }

    fn run_realtime(&mut self, rx: &Receiver<FwCommand>) {
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

    fn run_fixed(&mut self, rx: &Receiver<FwCommand>, total: u64) {
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
    cfg: FwSimConfig,
    mode: FwRunMode,
    logging: FwLoggingCfg,
    shared: Arc<Shared>,
    rx: Receiver<FwCommand>,
) -> FwRunReport {
    let dt = cfg.dt;
    let mut sim = FwSim::new(cfg);
    sim.set_logging(logging.log_every, logging.cap);
    let speed = match mode {
        FwRunMode::Realtime { speed } => speed.clamp(0.0, 16.0),
        FwRunMode::FixedSteps { .. } => 1.0,
    };
    let mut w = Worker {
        sim,
        dt,
        logging,
        paused: false,
        speed,
        seq: 0,
        shared,
    };
    w.publish(); // seed seq=1 snapshot before the loop
    match mode {
        FwRunMode::Realtime { .. } => w.run_realtime(&rx),
        FwRunMode::FixedSteps { total } => w.run_fixed(&rx, total),
    }
    FwRunReport {
        steps_taken: w.sim.tick(),
        final_tick: w.sim.tick(),
    }
}

// --- static guarantees (no data races) ---
fn _assert_send_sync() {
    fn send<T: Send>() {}
    fn copy<T: Copy>() {}
    send::<FwSnapshot>();
    copy::<FwSnapshot>();
    send::<FwCommand>();
    send::<FwSim>();
    send::<FwEngine>();
}
// Keep the linter from flagging the assertion helper as dead.
const _: fn() = _assert_send_sync;

#[cfg(test)]
mod tests {
    use super::*;

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

    fn cfg() -> FwSimConfig {
        FwSimConfig::aerosonde_cruise()
    }

    #[test]
    fn command_round_trip_set_cruise() {
        let eng = FwEngine::spawn_realtime(cfg());
        let sp = FixedWingSetpoint {
            airspeed: 30.0,
            altitude: 150.0,
            course: 0.3,
        };
        eng.send(FwCommand::SetCruise(sp)).unwrap();
        let got = deadline_poll(|| eng.latest().setpoint == sp, 2.0);
        assert!(got, "SetCruise not reflected in snapshot");
        eng.shutdown();
    }

    #[test]
    fn snapshot_tick_advances() {
        let eng = FwEngine::spawn_realtime(cfg());
        let s0 = eng.latest();
        assert!(
            deadline_poll(|| eng.latest().tick > s0.tick, 2.0),
            "tick did not advance"
        );
        eng.shutdown();
    }

    #[test]
    fn pause_freezes_tick_while_seq_advances() {
        let eng = FwEngine::spawn_realtime(cfg());
        assert!(deadline_poll(|| eng.latest().tick > 10, 2.0));
        eng.send(FwCommand::Pause(true)).unwrap();
        thread::sleep(Duration::from_millis(50));
        let a = eng.latest();
        thread::sleep(Duration::from_millis(50));
        let b = eng.latest();
        assert_eq!(a.tick, b.tick, "tick advanced while paused");
        assert!(b.seq > a.seq, "seq should advance even when paused");
        eng.shutdown();
    }

    #[test]
    fn speed_zero_does_not_advance() {
        let eng = FwEngine::spawn_realtime(cfg());
        eng.send(FwCommand::SetSpeed(0.0)).unwrap();
        thread::sleep(Duration::from_millis(50));
        let a = eng.latest();
        thread::sleep(Duration::from_millis(80));
        let b = eng.latest();
        assert_eq!(a.tick, b.tick, "advanced at speed 0");
        assert!(b.seq > a.seq, "seq should still advance");
        eng.shutdown();
    }

    #[test]
    fn reset_restarts_tick() {
        let eng = FwEngine::spawn_realtime(cfg());
        assert!(deadline_poll(|| eng.latest().tick > 100, 2.0));
        eng.send(FwCommand::Reset(Box::new(cfg()))).unwrap();
        assert!(
            deadline_poll(|| eng.latest().tick < 100, 2.0),
            "tick did not reset"
        );
        eng.shutdown();
    }

    #[test]
    fn route_round_trip_sets_waypoint_index() {
        let eng = FwEngine::spawn_realtime(cfg());
        eng.send(FwCommand::SetRoute {
            waypoints: vec![
                Waypoint::ne_alt(0.0, 0.0, 120.0),
                Waypoint::ne_alt(400.0, 0.0, 120.0),
            ],
            cfg: FwGuidanceConfig::default(),
        })
        .unwrap();
        assert!(
            deadline_poll(|| eng.latest().waypoint_index.is_some(), 2.0),
            "route not installed (waypoint_index stayed None)"
        );
        eng.shutdown();
    }

    #[test]
    fn clean_join_on_drop() {
        {
            let eng = FwEngine::spawn_realtime(cfg());
            assert!(deadline_poll(|| eng.latest().tick > 10, 2.0));
        } // drop joins; reaching here without hanging means the worker was reaped
    }

    #[test]
    fn fixedsteps_matches_run_headless() {
        // The threaded fixed-step path is bit-for-bit identical to inline.
        let total = 4000u64;
        let mut inline = FwSim::new(cfg());
        inline.set_logging(10, None);
        inline.run_headless(total as usize);
        let want = inline.samples().to_vec();

        let eng = FwEngine::spawn(
            cfg(),
            FwRunMode::FixedSteps { total },
            FwLoggingCfg {
                log_every: 10,
                cap: None,
            },
        );
        assert!(
            deadline_poll(|| eng.latest().tick >= total, 5.0),
            "fixed run did not finish"
        );
        let telem = eng.telemetry();
        eng.shutdown();
        assert_eq!(telem.len(), want.len(), "sample count differs");
        for (a, b) in telem.iter().zip(&want) {
            assert_eq!(a, b, "threaded fixed-step diverged from run_headless");
        }
    }
}
