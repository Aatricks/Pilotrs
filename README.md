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

## Status: M7 — a spherical 1/1000-Earth world (roadmap + beyond)

A second airframe — a **fixed-wing aircraft** — now flies on the *same*
integrator, `State13`, conventions, and rigid-body equations of motion as the
quad. The estimator is selectable across three filters, the quad's inner
controller is selectable (cascaded PID or LQR), the autopilot flies attitude
setpoints or full **waypoint missions**, the whole thing runs on a **dedicated
physics thread** with record/replay and batch tooling, and the fixed-wing adds
an aerodynamic plant + trim + its own autopilot:

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
- **M6 — fixed-wing**: a Beard & McLain aerodynamic plant (lift/drag/moment
  coefficients, stall blend, control surfaces, propeller) for the Aerosonde UAV,
  a Newton **trim** solver, and a decoupled successive-loop **autopilot** holding
  airspeed / altitude / course. It reuses the quad's RK4 and the *shared*
  `rigid_body_deriv` verbatim (the full non-diagonal `Jxz` inertia and all) —
  only the wrench and the autopilot differ. The Aerosonde climbs, turns, and
  changes speed under autopilot, all deterministic and headless.
- **M7 — a spherical world**: the planet becomes a **1/1000-scale Earth**
  (radius 6371 m). The fixed-wing flies in a **planet-centered inertial frame**
  with **radial inverse-square gravity**; its autopilot is unchanged — it just
  reads a *local horizon* derived from the current position each step (attitude
  composed with `q_ned_from_pci`, altitude = `|p| − R`, local course), so "level"
  follows the curve and routes are **great circles**. The quad stays in a flat
  local-tangent frame at home (curvature there is ~1e-8 rad). The viewer draws
  the whole planet as a displaced globe with a local-radial follow-camera.

169 tests pass across the workspace, the core ring builds `no_std`, and runs are
bit-for-bit deterministic.

## Workspace layout

The `core → … → control` crates form a **`no_std`-clean flight-control ring**
(they build with `--no-default-features`); everything OS/GPU lives only in
`fsim-viz`.

| Crate | Role |
|-------|------|
| `fsim-core` | `State13`, frame/quaternion conventions, shared message types (`CtrlCmd`, `FixedWingControls`, …). The contract everyone imports. |
| `fsim-dynamics` | Shared Newton-Euler EOM (`rigid_body_deriv`) + RK4 with per-step quaternion renormalization; the quad plant **and the M6 fixed-wing aero plant + trim**. |
| `fsim-actuators` | X-quad control-allocation mixer + first-order motor model. |
| `fsim-sensors` | `Sensor` trait + IMU / GPS / baro / magnetometer models, each with its own seeded `ChaCha8` noise + bias random-walk. |
| `fsim-estimator` | `Estimator` trait + complementary filter, 6-state quaternion MEKF, **and a 15-state INS** (GPS/baro/mag fusion). |
| `fsim-control` | `Controller` trait + cascaded attitude→rate PID, **LQR**, position/velocity control, **and the fixed-wing autopilot**. |
| `fsim-sim` | Deterministic scheduler, waypoint `Guidance`, **threaded `SimEngine`**, **record/replay**, **batch/Monte-Carlo**, **the fixed-wing `FwSim`**, telemetry. |
| `fsim-viz` | three-d + egui_plot interactive viewer (std-only leaf). |

## Conventions (defined once in `fsim-core`)

- **World frame: NED** (North-East-Down) — gravity is world `+z`, altitude is `-z`.
- **Body frame: FRD** (Forward-Right-Down) at the CoG.
- **Attitude:** `q_{world←body}`, Hamilton convention, renormalized every step.
- **Angular rate** is in the body frame (what the gyro reads).

> M7 note: the quad and all the flat-earth subsystems keep these conventions
> verbatim. The fixed-wing's "world" is instead a **planet-centered inertial
> (PCI)** frame (`fsim_core::planet`); the *local* NED above is then a tangent
> frame derived from the current position — the attitude/EOM are frame-agnostic,
> so only gravity (now radial) and the navigation maths (great circles) differ.

## Running

```bash
cargo test  --workspace                          # the full test suite (169 tests)
cargo run   -p fsim-sim --example headless        # INS flies a square mission + M2 contrast
cargo run   -p fsim-sim --release --example montecarlo     # parallel Monte-Carlo (faster-than-real-time)
cargo run   -p fsim-sim --release --example pid_vs_lqr      # PID vs LQR step-response + mission A/B
cargo run   -p fsim-sim --release --example fixedwing_cruise # the fixed-wing climbs, turns, changes speed
cargo run   -p fsim-sim --example record_replay   # record a mission, reload, replay it
cargo run   -p fsim-viz --release                 # the interactive 3D viewer (sim on its own thread)
```

In the viewer: an **airframe toggle** flies either the quad or the fixed-wing
over a **1/1000-scale spherical Earth** (M7) — a ~6.4 km-radius "tiny planet"
(~40 km around) with procedural **mountain ranges** rising ~320 m
(water → shore → grass → rock → snow) and a flat **home airfield clearing**. The
fixed-wing flies the planet with **full spherical physics** (gravity points to
the core; "level" follows the curve) and **great-circle routes**; the quad flies
its mission in a flat local-tangent patch at home (it never strays far enough for
curvature to matter). The follow-camera orbits the globe with a local-radial up
vector. The **Route planner** minimap is a top-down tangent map at home — click
to drop waypoints, drag to move, right-click to remove — and **Fly route**
dispatches it to the active aircraft (the quad as an INS waypoint mission, the
fixed-wing as a great-circle route, tuned so the ground track converges onto each
leg smoothly instead of snaking). The **Flight controls** window switches the estimator (CF / MEKF / INS),
the inner controller (PID / LQR), and sets the attitude / cruise setpoint; the
telemetry window plots estimate-vs-truth-vs-setpoint (quad) or airspeed /
altitude / course (fixed-wing).

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
- **M6 ✅** fixed-wing aero plant (lift/drag/stall/surfaces) + trim + autopilot, reusing the shared EOM/RK4.

The original roadmap is complete. Beyond it, the viewer now flies **both
airframes over a procedural terrain map** with an **interactive route-planner
minimap** (the fixed-wing runs on a threaded `FwEngine` + waypoint line
guidance). Natural extensions: sensors + the INS in the fixed-wing loop,
wind/turbulence (Dryden), orbit/Dubins guidance, and MPC behind the
`Controller` trait.

## License

MIT OR Apache-2.0.
