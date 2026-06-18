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
truth ─▶ SENSORS ──────▶ ESTIMATOR ────▶ CONTROLLER ─▶ MIXER+MOTORS ─▶ DYNAMICS ─▶ RK4 ─▶ truth'
 ▲    (IMU/GPS/baro/mag)  (compl. filter   (cascaded     (X-mix +        (Newton-     (1 kHz) │
 └──────────────────────  or quat MEKF)     PID)          motor model)    Euler)──────────────┘
```

The controller and estimator **only** consume sensor-derived estimates — that's
what makes the estimate-vs-truth plots meaningful.

## Status: M2 (realistic sensors + MEKF) complete

A quadrotor holds hover and tracks attitude through the full
estimator-in-the-loop pipeline. The estimator is selectable: a Mahony
complementary filter (M1) or a **6-state quaternion MEKF** (M2) that estimates
the IMU's hidden gyro bias and fuses a magnetometer for heading. On a realistic
biased-IMU stream the MEKF clearly beats the CF (≈0.95° vs ≈3.1° attitude error,
and the CF's yaw drifts on the gyro bias while the MEKF's does not). GPS and
barometer models are wired (position fusion / 15-state INS lands in M3).

46 tests pass across the workspace, the core ring builds `no_std`, and runs are
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
| `fsim-estimator` | `Estimator` trait + Mahony complementary filter **and** a 6-state quaternion MEKF (attitude + gyro bias, accel + mag updates). |
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
cargo test --workspace --exclude fsim-viz   # the headless test suite (46 tests)
cargo run  -p fsim-sim --example headless    # headless: MEKF bias-estimation + CF-vs-MEKF
cargo run  -p fsim-viz --release            # the interactive 3D viewer + plots
```

In the viewer: drag to orbit; the **Flight controls** window switches the
estimator (MEKF ↔ complementary), sets the attitude setpoint / thrust, pauses,
changes sim speed, or resets, and shows the true-vs-estimated gyro bias. The
**Estimate vs truth vs setpoint** window plots roll/pitch/yaw, motor thrusts,
and (under the MEKF) the gyro-bias estimate converging on the hidden truth.

> Note: the MEKF is an *AHRS* — it assumes the accelerometer sees gravity, so a
> sustained translating maneuver (which keeps the craft accelerating) degrades
> the attitude estimate. Removing vehicle acceleration from the specific force
> needs velocity aiding — the M3 INS (GPS/baro fusion).

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
- **M3** velocity/position loops + waypoint guidance + motor lag + 15-state INS (fuse GPS/baro).
- **M4** sim-on-its-own-thread, headless faster-than-real-time, record/replay.
- **M5** LQR / MPC inner loops behind the `Controller` trait.
- **M6** fixed-wing plant (lift/drag/stall) reusing the same infrastructure.

## License

MIT OR Apache-2.0.
