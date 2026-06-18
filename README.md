# Pilotrs — a Rust 6DOF flight simulator + autopilot

A from-scratch **6 degrees-of-freedom rigid-body simulator** for a quadrotor,
with a full **estimation + control stack** wrapped around it. The defining
constraint: the autopilot never sees truth — only noisy, degraded sensor
measurements — which is what makes this an estimation/control problem rather
than a math toy.

Built in Rust with [`nalgebra`](https://nalgebra.org), kept
**Ferrocene-compatible** (MSRV 1.91, `no_std`-clean flight-control core), and
visualized with [`three-d`](https://github.com/asny/three-d) + `egui`.

```
truth ─▶ SENSORS ──────▶ ESTIMATOR ─────────▶ CONTROLLER ─────▶ MIXER+MOTORS ─▶ DYNAMICS ─▶ RK4 ─▶ truth'
 ▲    (IMU/GPS/baro/mag)  (compl. filter /     (cascaded PID,    (X-mix +        (Newton-     (1 kHz) │
 └──────────────────────  quat MEKF / 15-INS)   pos + waypoints)  motor lag)      Euler)──────────────┘
```

The controller and estimator **only** consume sensor-derived estimates — that's
what makes the estimate-vs-truth plots meaningful.

## Status: M5 (LQR controller) complete

The estimator is selectable across three filters, the **inner controller** is
selectable (cascaded PID or LQR), the autopilot flies attitude setpoints or full
**waypoint missions**, and the whole thing runs on a **dedicated physics thread**
with record/replay and batch tooling:

- **M1 — complementary filter** (attitude only).
- **M2 — 6-state quaternion MEKF**: estimates the IMU's hidden gyro bias + fuses
  a magnetometer for heading (≈0.95° vs the CF's ≈3.1°; the CF's yaw drifts).
- **M3 — 15-state INS** (GPS/baro/vel/mag fusion): the accelerometer is the
  *strapdown input*, so sustained acceleration no longer corrupts attitude — the
  AHRS limitation, **fixed**. Real position/velocity enables **position control
  + waypoint guidance**; the quad flies a 5 m square mission, returning home with
  the INS tracking truth to under ~1 m through 2.5 m GPS noise.
- **M4 — threaded `SimEngine`**: the deterministic physics runs on its own
  thread, publishing latest-state `Snapshot`s the viewer reads without blocking
  it (physics/render decoupled). Plus **telemetry record/replay** (bit-exact CSV)
  and a **faster-than-real-time parallel Monte-Carlo** harness (2.5 M fixed steps
  in ~1 s; `run_batch == run_batch_seq` for any worker count).
- **M5 — LQR controller**: an optimal state-feedback drop-in for the cascaded
  PID, selectable per run. The diagonal inertia decouples the attitude/rate
  dynamics into three per-axis double integrators with a closed-form Riccati
  solution — no runtime solver. On a 15° step the LQR shows ~0.1% overshoot and
  ~0.38 s settling vs the PID's ~4.7% / ~0.65 s, matching it on mission tracking.

97 tests pass across the workspace, the core ring builds `no_std`, and runs are
bit-for-bit deterministic.

## Workspace layout

The `core → … → control` crates form a **`no_std`-clean flight-control ring**
(they build with `--no-default-features`); everything OS/GPU lives only in
`fsim-viz`.

| Crate | Role |
|-------|------|
| `fsim-core` | `State13`, frame/quaternion conventions, shared message types. The contract everyone imports. |
| `fsim-dynamics` | Newton-Euler plant (`I·ω̇ + ω×Iω = M`) + RK4 integrator with per-step quaternion renormalization. |
| `fsim-actuators` | X-quad control-allocation mixer + first-order motor model. |
| `fsim-sensors` | `Sensor` trait + IMU / GPS / baro / magnetometer models, each with its own seeded `ChaCha8` noise + bias random-walk. |
| `fsim-estimator` | `Estimator` trait + complementary filter, 6-state quaternion MEKF, **and a 15-state INS** (GPS/baro/mag fusion). |
| `fsim-control` | `Controller` trait + cascaded attitude→rate PID, **LQR**, and position/velocity control (accel→attitude inversion). |
| `fsim-sim` | Deterministic scheduler, control-mode switch, waypoint `Guidance`, **threaded `SimEngine`**, **record/replay**, **batch/Monte-Carlo**, telemetry. |
| `fsim-viz` | three-d + egui_plot interactive viewer (std-only leaf). |

## Conventions (defined once in `fsim-core`)

- **World frame: NED** (North-East-Down) — gravity is world `+z`, altitude is `-z`.
- **Body frame: FRD** (Forward-Right-Down) at the CoG.
- **Attitude:** `q_{world←body}`, Hamilton convention, renormalized every step.
- **Angular rate** is in the body frame (what the gyro reads).

## Running

```bash
cargo test  --workspace                          # the full test suite (92 tests)
cargo run   -p fsim-sim --example headless        # INS flies a square mission + M2 contrast
cargo run   -p fsim-sim --release --example montecarlo  # parallel Monte-Carlo (faster-than-real-time)
cargo run   -p fsim-sim --release --example pid_vs_lqr   # PID vs LQR step-response + mission A/B
cargo run   -p fsim-sim --example record_replay   # record a mission, reload, replay it
cargo run   -p fsim-viz --release                 # the interactive 3D viewer (sim on its own thread)
```

In the viewer: drag to orbit; the **Flight controls** window switches the
estimator (CF / MEKF / INS), toggles the **square mission** (INS only), sets the
attitude setpoint / thrust, and shows true-vs-estimated attitude, gyro bias, and
position. The **Estimate vs truth vs setpoint** window plots roll/pitch/yaw,
motor thrusts, the gyro-bias estimate, and (under the INS) position tracking.

> The MEKF is an *AHRS*: it assumes the accelerometer sees gravity, so a
> sustained translating maneuver degrades its attitude. The **INS removes this**
> by using the accelerometer as the strapdown input with GPS/baro velocity
> aiding — try a large tilt under the MEKF, then switch to the INS mission.

## Toolchain & Ferrocene

Develop on stable Rust. The code is kept Ferrocene-compatible by construction
(MSRV 1.91, `no_std`-clean core). `aarch64-apple-darwin` is a supported
full-`std` Ferrocene host, but the prebuilt toolchain requires a subscription;
`criticalup.toml` is in place (dormant) so the Ferrocene compiler swaps in with
zero refactoring once a token is configured:

```bash
criticalup auth set && criticalup install && criticalup run cargo build
```

## Roadmap

- **M1 ✅** quad dynamics + RK4 + PID + complementary filter + 3D/plots.
- **M2 ✅** realistic sensors (GPS/baro/mag) + 6-state quaternion MEKF/AHRS.
- **M3 ✅** 15-state INS (GPS/baro/vel fusion) + position/velocity control + waypoint guidance + motor lag.
- **M4 ✅** threaded `SimEngine` (sim on its own thread) + record/replay + parallel faster-than-real-time Monte-Carlo.
- **M5 ✅** LQR inner loop behind the `Controller` trait, A/B-comparable with the PID (MPC deferred).
- **M6** fixed-wing plant (lift/drag/stall) reusing the same infrastructure.

## License

MIT OR Apache-2.0.
