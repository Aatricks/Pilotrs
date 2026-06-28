# Pilotrs

[![CI](https://github.com/Aatricks/Pilotrs/actions/workflows/ci.yml/badge.svg)](https://github.com/Aatricks/Pilotrs/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/Rust-1.91%2B-orange.svg?logo=rust)](https://www.rust-lang.org)
[![Core](https://img.shields.io/badge/core-no__std-success.svg)](#workspace-layout)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A 6-DOF flight simulator and autopilot written from scratch in Rust. It runs full rigid-body physics for a quadrotor and a fixed-wing over a 1/1000-scale spherical Earth, using a custom sensing -> estimation -> control stack.

<p align="center">
  <img src="docs/screenshot.png" alt="Pilotrs — hand-flying the fly-by-wire fighter over the 1/1000-scale Earth" width="640">
</p>
<p align="center"></p>

The catch: **the autopilot never gets ground truth.** It only knows what its noisy, drifted sensors tell it, which goes through an onboard estimator before reaching the flight controller.

Everything is built on top of [`nalgebra`](https://nalgebra.org). The flight core is strictly `no_std` so it could run on real microcontroller hardware or Ferrocene. The frontend is visualized with [`three-d`](https://github.com/asny/three-d) and [`egui`](https://github.com/emilk/egui).

<p align="center">
  <img src="docs/diagram.svg" alt="Diagram of the simulation modules interactions" width="640">
</p>



## What's inside

- **Physics & Models**: 6-DOF rigid body dynamics shared by both the quadrotor and the fixed-wing. Integrates using RK4 at 1 kHz. Includes aerodynamic lift/drag/moment curves and a trim solver for the plane.
- **Sensors**: Simulated IMU, GPS, barometer, and magnetometer. They all have configurable, reproducible noise and bias walk so they behave like real hardware.
- **State Estimation**: Three estimators you can swap: a simple complementary filter, a 6-state MEKF (attitude + gyro bias), and a 15-state INS (GPS/baro/velocity/mag fusion). The INS is robust enough that pulling Gs won't mess up your attitude.
- **Flight Controllers**: Swappable cascaded PID and LQR inner loops sharing a `Controller` trait. The quad has waypoint tracking, and the plane has a successive-loop autopilot to hold heading, altitude, and airspeed.
- **Fly-by-Wire Fighter**: The fighter jet is aerodynamically unstable (negative static margin), so it crashes instantly without computer help. The onboard FCS uses angle-of-attack and rate feedback to stabilize it. You can press `F` to turn off the FCS and feel how bad the raw airframe actually is.
- **Interactive 3D Viewer**: Built with `three-d` and `egui`. Runs the simulator on a separate thread, lets you record/replay flight logs, run parallel Monte-Carlo simulations, drop waypoints on a map, and see live telemetry.


## Coordinate Frames & Math

To keep sanity, everything in `fsim-core` uses:
- **World frame:** North-East-Down (NED), so +z is down and altitude is -z.
- **Body frame:** Forward-Right-Down (FRD), centered at the CG.
- **Attitude:** Hamilton quaternions (`q_{world<-body}`), normalized every step.
- **Rates:** Angular velocity is in the body frame (matches what the gyros output).

For the fixed-wing, the "world" is actually a planet-centered frame, and its local NED frame shifts dynamically as it flies. The physics code is frame-agnostic, so we just swap in radial gravity and great-circle navigation.

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

Once the viewer is running, you can use the UI panels to:
- Swap between the quadrotor, the fixed-wing autopilot, and the manual **Fighter (FBW)**.
- Hot-swap the estimator (complementary / MEKF / INS) and the inner controller (PID / LQR).
- Set targets (attitude or cruise) and see live plots comparing the estimate vs. ground truth.
- Click on the zoomable world map in the **Route planner** to set waypoints for the aircraft to follow.

Pick **Fighter (FBW)** and you fly by hand:

| | pitch | roll | yaw | throttle | toggle FCS | reset to trim |
|---|---|---|---|---|---|---|
| **keyboard** | `W` / `S` | `A` / `D` | `Q` / `E` | `Shift` / `Ctrl` | `F` | `R` |
| **gamepad** | left stick | left stick | right stick | right stick | `A` | `Start` |

### Some things to try:
1. **Manual Flying**: Try flying the fighter with the computer assist (FCS) turned on, then hit `F` to disable it. It will immediately tumble out of control because the raw airframe is aerodynamically unstable.
2. **Estimator Drift**: The MEKF assumes gravity is the only acceleration it feels. If you pull a long, sustained turn, the attitude estimate will drift because it gets confused by the centrifugal acceleration. If you swap to the INS, it uses GPS/velocity fusion to handle sustained turns without drifting.

## Toolchain & Safety-Critical Rust

The project runs on stable Rust, but it's built to support Ferrocene (the certified safety-critical Rust compiler) with an MSRV of 1.91 and a clean `no_std` core.

If you have a Ferrocene license, you can swap the compiler in using the included `criticalup.toml`:

```bash
criticalup auth set && criticalup install && criticalup run cargo build
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
