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

## Status: M3 (INS + waypoint guidance) complete

The estimator is selectable across three increasingly capable filters, and the
autopilot flies either attitude setpoints or full **waypoint missions**:

- **M1 — complementary filter** (attitude only).
- **M2 — 6-state quaternion MEKF**: estimates the IMU's hidden gyro bias + fuses
  a magnetometer for heading. Beats the CF on a biased-IMU stream (≈0.95° vs
  ≈3.1°; the CF's yaw drifts on the bias).
- **M3 — 15-state INS** (error-state KF fusing GPS position+velocity, baro, mag):
  uses the accelerometer as the *strapdown input*, so a sustained translating
  maneuver no longer corrupts attitude — the AHRS limitation, **fixed**. It
  returns real position/velocity, enabling **position/velocity control +
  waypoint guidance**: the quad flies a 5 m square mission and returns home,
  with the INS tracking truth to under ~1 m through 2.5 m GPS noise.

67 tests pass across the workspace, the core ring builds `no_std`, and runs are
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
| `fsim-control` | Cascaded attitude→rate PID **+ position/velocity control** with an accel→attitude inversion. |
| `fsim-sim` | Deterministic fixed-step scheduler, control-mode switch, waypoint `Guidance`, telemetry, headless runner. |
| `fsim-viz` | three-d + egui_plot interactive viewer (std-only leaf). |

## Conventions (defined once in `fsim-core`)

- **World frame: NED** (North-East-Down) — gravity is world `+z`, altitude is `-z`.
- **Body frame: FRD** (Forward-Right-Down) at the CoG.
- **Attitude:** `q_{world←body}`, Hamilton convention, renormalized every step.
- **Angular rate** is in the body frame (what the gyro reads).

## Running

```bash
cargo test --workspace --exclude fsim-viz   # the headless test suite (67 tests)
cargo run  -p fsim-sim --example headless    # headless: INS flies a square mission + M2 contrast
cargo run  -p fsim-viz --release            # the interactive 3D viewer + plots
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
- **M4** sim-on-its-own-thread, headless faster-than-real-time, record/replay.
- **M5** LQR / MPC inner loops behind the `Controller` trait.
- **M6** fixed-wing plant (lift/drag/stall) reusing the same infrastructure.

## License

MIT OR Apache-2.0.
