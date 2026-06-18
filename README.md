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
truth ─▶ SENSORS ─▶ ESTIMATOR ─▶ CONTROLLER ─▶ MIXER+MOTORS ─▶ DYNAMICS ─▶ RK4 ─▶ truth'
 ▲       (noisy IMU)  (compl.       (cascaded     (X-mix +        (Newton-     (1 kHz)  │
 └────────────────────  filter)      PID)          motor model)    Euler)──────────────┘
```

The controller and estimator **only** consume sensor-derived estimates — that's
what makes the estimate-vs-truth plots meaningful.

## Status: M1 (MVP) complete

A quadrotor holds hover and tracks attitude setpoints through the full
estimator-in-the-loop pipeline, with a live 3D view and plots. 31 tests pass
across the workspace, the core ring builds `no_std`, and runs are bit-for-bit
deterministic.

## Workspace layout

The `core → … → control` crates form a **`no_std`-clean flight-control ring**
(they build with `--no-default-features`); everything OS/GPU lives only in
`fsim-viz`.

| Crate | Role |
|-------|------|
| `fsim-core` | `State13`, frame/quaternion conventions, shared message types. The contract everyone imports. |
| `fsim-dynamics` | Newton-Euler plant (`I·ω̇ + ω×Iω = M`) + RK4 integrator with per-step quaternion renormalization. |
| `fsim-actuators` | X-quad control-allocation mixer + first-order motor model. |
| `fsim-sensors` | `Sensor` trait + IMU model with seeded `ChaCha8` noise/bias random-walk. |
| `fsim-estimator` | `Estimator` trait + Mahony complementary filter (MEKF in M2). |
| `fsim-control` | `Controller` trait + cascaded attitude→rate PID. |
| `fsim-sim` | Deterministic fixed-step scheduler, telemetry, headless runner. |
| `fsim-viz` | three-d + egui_plot interactive viewer (std-only leaf). |

## Conventions (defined once in `fsim-core`)

- **World frame: NED** (North-East-Down) — gravity is world `+z`, altitude is `-z`.
- **Body frame: FRD** (Forward-Right-Down) at the CoG.
- **Attitude:** `q_{world←body}`, Hamilton convention, renormalized every step.
- **Angular rate** is in the body frame (what the gyro reads).

## Running

```bash
cargo test --workspace --exclude fsim-viz   # the headless test suite (31 tests)
cargo run  -p fsim-viz --release            # the interactive 3D viewer + plots
```

In the viewer: drag to orbit; use the **Flight controls** window to set the
attitude setpoint / thrust, pause, change sim speed, or reset. The
**Estimate vs truth vs setpoint** window plots roll/pitch/yaw and motor thrusts
live. Try a large sustained tilt and watch the complementary-filter estimate
diverge from truth — the exact reason M2 brings a quaternion MEKF.

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
- **M2** realistic sensors (GPS/baro/mag) + quaternion MEKF/AHRS.
- **M3** velocity/position loops + waypoint guidance + motor lag + 15-state INS.
- **M4** sim-on-its-own-thread, headless faster-than-real-time, record/replay.
- **M5** LQR / MPC inner loops behind the `Controller` trait.
- **M6** fixed-wing plant (lift/drag/stall) reusing the same infrastructure.

## License

MIT OR Apache-2.0.
