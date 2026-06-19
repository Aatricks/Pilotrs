# Pilotrs

A from-scratch **6-degrees-of-freedom flight simulator and autopilot**, written
in Rust. It simulates the full rigid-body dynamics of two aircraft — a
**quadrotor** and a **fixed-wing UAV** — and wraps each in a complete
**sensing → estimation → control** stack.

The defining constraint: **the autopilot never sees ground truth.** It flies on
noisy, degraded sensor measurements fused by an onboard estimator — which is
what makes this a real estimation-and-control problem rather than a kinematics
demo.

Built on [`nalgebra`](https://nalgebra.org), kept **`no_std`-clean** in the
flight-control core (so it can target embedded / Ferrocene toolchains), and
visualized with [`three-d`](https://github.com/asny/three-d) +
[`egui`](https://github.com/emilk/egui).

```
truth ─▶ SENSORS ─────▶ ESTIMATOR ───────▶ CONTROLLER ─────▶ MIXER + MOTORS ─▶ DYNAMICS ─▶ RK4 ─▶ truth'
 ▲    (IMU/GPS/baro/mag) (complementary /    (cascaded PID /   (X-mixer +        (Newton-     (1 kHz)  │
 └──────────────────────  quaternion MEKF /   LQR, position +   motor lag)        Euler EOM) ───────────┘
                          15-state INS)        waypoints)
```

The controller and estimator consume **only** the estimator's output, never
truth, so the estimate-vs-truth plots in the viewer actually mean something.

## Features

**Dynamics & integration**
- 6DOF rigid-body equations of motion with the full non-diagonal inertia tensor,
  shared verbatim by both airframes.
- Fixed-step **RK4** integrator at 1 kHz with per-step quaternion
  renormalization.
- A quadrotor plant (thrust + drag) and a Beard & McLain **fixed-wing
  aerodynamic model** (lift/drag/moment coefficients, stall blend, control
  surfaces, propeller) with a Newton **trim** solver.

**Sensing & estimation**
- Sensor models — IMU, GPS, barometer, magnetometer — each degrading truth with
  seeded, reproducible noise and bias random-walk.
- Three selectable estimators: a **complementary filter**, a 6-state
  **quaternion MEKF** (attitude + gyro bias), and a 15-state **INS**
  (GPS/baro/velocity/mag fusion) that treats the accelerometer as a strapdown
  input, so sustained acceleration no longer corrupts attitude.

**Control & guidance**
- A cascaded **PID** and an **LQR** inner loop, swappable per run behind a common
  `Controller` trait.
- Position/velocity control and **waypoint missions** for the quadrotor.
- A successive-loop **fixed-wing autopilot** holding airspeed, altitude, and
  course, with coordinated turns.

**Spherical world**
- The planet is a **1/1000-scale Earth** (6371 m radius). The fixed-wing flies in
  a planet-centered inertial frame with **radial, inverse-square gravity** and
  **great-circle** routes — "straight and level" follows the curve of the
  planet. The quadrotor flies in a flat local-tangent frame near home, where the
  curvature is negligible.

**Tooling**
- The deterministic simulation runs on its own thread, decoupled from rendering.
- Bit-exact telemetry **record/replay** and a **parallel Monte-Carlo** harness
  that runs faster than real time.
- An interactive **3D viewer**: switch airframes, watch the aircraft fly over a
  displaced globe with a follow-camera, plan routes on a zoomable **planisphere**
  world map, and read live estimate-vs-truth telemetry.

Everything is **deterministic** — fixed timestep, no wall-clock in the math, one
seeded RNG per sensor — so a run reproduces bit-for-bit.

## Workspace layout

The `fsim-core → … → fsim-control` crates form a **`no_std`-clean flight-control
ring** (they build with `--no-default-features`); everything OS/GPU-bound lives
only in `fsim-viz`.

| Crate | Role |
|-------|------|
| `fsim-core` | `State13`, frame/quaternion conventions, shared message types, and the spherical-world math. The contract every other crate imports. |
| `fsim-dynamics` | Shared Newton-Euler EOM + RK4 with per-step renormalization; the quadrotor plant and the fixed-wing aero model + trim. |
| `fsim-actuators` | Quadrotor control-allocation mixer + first-order motor model. |
| `fsim-sensors` | `Sensor` trait + IMU / GPS / barometer / magnetometer models, each with its own seeded noise and bias random-walk. |
| `fsim-estimator` | `Estimator` trait + complementary filter, quaternion MEKF, and 15-state INS. |
| `fsim-control` | `Controller` trait + cascaded PID, LQR, position/velocity control, and the fixed-wing autopilot. |
| `fsim-sim` | Deterministic scheduler, waypoint guidance, threaded engine, record/replay, Monte-Carlo, and the fixed-wing simulation. |
| `fsim-viz` | Interactive `three-d` + `egui` viewer (std-only leaf crate). |

## Conventions

Defined once in `fsim-core`:

- **World frame:** North-East-Down — gravity is +z, altitude is −z.
- **Body frame:** Forward-Right-Down, at the center of gravity.
- **Attitude:** `q_{world←body}`, Hamilton convention, renormalized every step.
- **Angular rate** is expressed in the body frame (what the gyro measures).

The fixed-wing's "world" is instead a planet-centered inertial frame; its local
North-East-Down is a tangent frame derived from the current position. The
equations of motion are frame-agnostic, so only gravity (now radial) and the
navigation math (great circles) differ.

## Building and running

```bash
cargo test --workspace                                     # run the test suite
cargo run -p fsim-viz  --release                           # the interactive 3D viewer
cargo run -p fsim-sim  --example headless                  # quad flies a waypoint mission, headless
cargo run -p fsim-sim  --release --example montecarlo      # parallel Monte-Carlo
cargo run -p fsim-sim  --release --example pid_vs_lqr       # PID vs LQR step-response comparison
cargo run -p fsim-sim  --release --example fixedwing_cruise # the fixed-wing climbs, turns, changes speed
cargo run -p fsim-sim  --example record_replay             # record a run, reload it, replay it
```

In the viewer, an airframe toggle flies either the quad or the fixed-wing. The
**Flight controls** panel switches the estimator (complementary / MEKF / INS) and
the inner controller (PID / LQR) and sets the attitude or cruise target; the
telemetry panel plots estimate vs. truth vs. setpoint. The **Route planner** is a
zoomable world map — click to drop waypoints, then dispatch them to the active
aircraft.

> The MEKF is an AHRS: it assumes the accelerometer sees gravity, so a sustained
> translating maneuver degrades its attitude estimate. The INS removes this
> limitation — try a large tilt under the MEKF, then switch to the INS.

## Toolchain & Ferrocene

Develop on stable Rust. The code is kept compatible with
[Ferrocene](https://ferrous-systems.com/ferrocene/) (the qualified Rust
toolchain) by construction — MSRV 1.91 and a `no_std`-clean core. A dormant
`criticalup.toml` is in place so the Ferrocene compiler can be swapped in without
refactoring once a subscription token is configured:

```bash
criticalup auth set && criticalup install && criticalup run cargo build
```

## Possible extensions

Sensors and the INS in the fixed-wing loop; wind and turbulence modeling
(Dryden); orbit / Dubins guidance; and model-predictive control behind the
`Controller` trait.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE-2.0) or
[MIT license](LICENSE-MIT), at your option.
