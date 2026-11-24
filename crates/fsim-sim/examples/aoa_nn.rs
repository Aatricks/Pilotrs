//! Learned angle-of-attack (AoA) estimator: dataset + closed-loop test.
//!
//! The fixed-wing flies through random wind and turbulence while the simulated
//! IMU/GPS/baro/mag and the 15-state INS (Kalman) run beside it.
//!
//! - `dataset <aerosonde|fighter> out.csv [runs]` writes one CSV row at 50 Hz:
//!   raw sensors, INS estimate, controls, the INS AoA (`atan2(w, u)` of the
//!   *ground* velocity) and the true *air-relative* AoA. `scripts/aoa_nn.py`
//!   trains a small MLP on it and exports `<csv>.weights.txt`.
//! - `loop weights.txt [runs]` flies the relaxed-stability fighter on fly-by-wire
//!   with four AoA sources in its SAS — today's default, the INS, the network
//!   and the true air AoA — over the same flights, and compares how it flies.
//!
//! Run with `cargo run -p fsim-sim --release --example aoa_nn -- <command> ...`.
//!
//! Sensors and INS assume a flat world, but the fixed-wing flies on the small
//! sphere. So each step we build a consistent flat "virtual truth": local-NED
//! velocity and attitude, body rate *relative to local NED* and dv/dt by finite
//! difference, position = integrated NED velocity with altitude as −z.

use fsim_core::{planet, ImuMeas};
use fsim_estimator::{Estimator, Ins, InsConfig};
use fsim_sensors::{
    Baro, BaroConfig, Gps, GpsConfig, Imu, ImuConfig, Mag, MagConfig, Sensor, Truth,
};
use fsim_sim::{
    FixedWingSetpoint, FwSim, FwSimConfig, Quat, Real, State13, StickInput, Vec3, GRAVITY,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Uniform};
use std::io::Write;

const RUN_SECS: Real = 60.0;
/// Skip the INS convergence transient in datasets and metrics.
const SETTLE_SECS: Real = 10.0;
const LOG_HZ: Real = 50.0;

/// Network inputs, in order: sensors, INS estimate and controls only.
const FEATURES: [&str; 16] = [
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "baro",
    "ins_ub",
    "ins_vb",
    "ins_wb",
    "ins_roll",
    "ins_pitch",
    "elevator",
    "aileron",
    "throttle",
    "alpha_ins",
];
const ALPHA_INS: usize = 15;

fn u(rng: &mut ChaCha8Rng, a: Real, b: Real) -> Real {
    Uniform::new(a, b).unwrap().sample(rng)
}

/// Local-NED (velocity, attitude) of a PCI truth state.
fn local(s: &State13) -> (Vec3, Quat) {
    let q = planet::ned_from_pci(s.position);
    (q * s.velocity, q * s.attitude)
}

fn alpha_of(v_body: Vec3) -> Real {
    v_body.z.atan2(v_body.x)
}

#[derive(Clone, Copy, PartialEq)]
enum Airframe {
    Aerosonde,
    Fighter,
}

/// One flight: the sim, its sensors + INS, and a scripted pilot.
struct Rig {
    sim: FwSim,
    airframe: Airframe,
    rng: ChaCha8Rng,
    wind: Real,
    turb: Real,
    imu: Imu,
    gps: Gps,
    baro: Baro,
    mag: Mag,
    ins: Ins,
    dt: Real,
    k: u64,
    /// Sensor / control / pilot periods in base steps.
    p_imu: u64,
    p_gps: u64,
    p_baro: u64,
    p_mag: u64,
    p_ctrl: u64,
    p_pilot: u64,
    flat_ne: (Real, Real),
    last_imu: Option<ImuMeas>,
    last_baro: Real,
    /// Fighter pilot target (pitch, roll) \[rad\] and throttle.
    target: (Real, Real, Real),
    /// True load factor of the last step \[g\].
    nz: Real,
}

impl Rig {
    fn new(airframe: Airframe, run: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(0xA0A_0000 + run);
        let mut cfg = match airframe {
            Airframe::Aerosonde => FwSimConfig::aerosonde_cruise(),
            Airframe::Fighter => FwSimConfig::fighter_manual(1500.0),
        };
        cfg.atmosphere.seed = 0xB0B_0000 + run;
        let dt = cfg.dt;
        let ctrl_hz = cfg.control_rate;
        let mut sim = FwSim::new(cfg);
        // Every 10th run is calm air: the baseline where INS AoA should be right.
        let (wind_max, turb_max) = match airframe {
            Airframe::Aerosonde => (12.0, 3.0),
            Airframe::Fighter => (15.0, 5.0),
        };
        let calm = run.is_multiple_of(10);
        let wind = if calm {
            0.0
        } else {
            u(&mut rng, 0.0, wind_max)
        };
        let turb = if calm {
            0.0
        } else {
            u(&mut rng, 0.0, turb_max)
        };
        let dir = u(&mut rng, -3.2, 3.2);
        sim.set_wind(Vec3::new(wind * dir.cos(), wind * dir.sin(), 0.0));
        sim.set_turbulence(turb);

        let seed = 0xC0C_0000 + run * 16;
        let period = |hz: Real| ((1.0 / (hz * dt)).round() as u64).max(1);
        Self {
            sim,
            airframe,
            rng,
            wind,
            turb,
            imu: Imu::new(ImuConfig::realistic(1000.0), seed),
            gps: Gps::new(GpsConfig::realistic(10.0), seed + 1),
            baro: Baro::new(BaroConfig::realistic(50.0), seed + 2),
            mag: Mag::new(MagConfig::realistic(100.0), seed + 3),
            ins: Ins::new(InsConfig::default()),
            dt,
            k: 0,
            p_imu: period(1000.0),
            p_gps: period(10.0),
            p_baro: period(50.0),
            p_mag: period(100.0),
            p_ctrl: period(ctrl_hz),
            p_pilot: period(match airframe {
                Airframe::Aerosonde => 1.0 / 6.0,
                Airframe::Fighter => 1.0 / 2.0,
            }),
            flat_ne: (0.0, 0.0),
            last_imu: None,
            last_baro: 0.0,
            target: (0.0, 0.0, 0.7),
            nz: 1.0,
        }
    }

    fn time(&self) -> Real {
        self.k as Real * self.dt
    }

    /// True local-NED (roll, pitch) \[rad\].
    fn attitude(&self) -> (Real, Real) {
        let (r, p, _) = local(self.sim.truth()).1.euler_angles();
        (r, p)
    }

    /// The scripted pilot. Aerosonde: random autopilot setpoints. Fighter: hold
    /// a random pitch/bank target with the stick (it sees the true horizon).
    fn pilot(&mut self) {
        let new_target = self.k.is_multiple_of(self.p_pilot) && self.k > 0;
        match self.airframe {
            Airframe::Aerosonde => {
                if new_target {
                    self.sim.set_setpoint(FixedWingSetpoint {
                        airspeed: u(&mut self.rng, 18.0, 32.0),
                        altitude: u(&mut self.rng, 70.0, 160.0),
                        course: u(&mut self.rng, -3.2, 3.2),
                    });
                }
            }
            Airframe::Fighter => {
                if new_target {
                    let low = self.sim.altitude() < 800.0;
                    self.target = (
                        if low {
                            0.3
                        } else {
                            u(&mut self.rng, -0.2, 0.35)
                        },
                        u(&mut self.rng, -0.8, 0.8),
                        u(&mut self.rng, 0.4, 1.0),
                    );
                }
                if self.k.is_multiple_of(self.p_ctrl) {
                    let (roll, pitch) = self.attitude();
                    self.sim.set_stick(StickInput {
                        pitch: (2.0 * (self.target.0 - pitch)).clamp(-1.0, 1.0),
                        roll: (2.0 * (self.target.1 - roll)).clamp(-1.0, 1.0),
                        yaw: 0.0,
                        throttle: self.target.2,
                    });
                }
            }
        }
    }

    /// Advance one base step. `false` once the aircraft hits the ground.
    fn step(&mut self) -> bool {
        self.pilot();
        let before = *self.sim.truth();
        self.sim.step();
        if self.sim.crashed() || self.sim.altitude() < 5.0 {
            return false;
        }
        let (v0, q0) = local(&before);
        let (v1, q1) = local(self.sim.truth());
        let dt = self.dt;
        self.flat_ne.0 += v0.x * dt;
        self.flat_ne.1 += v0.y * dt;
        let flat = State13 {
            position: Vec3::new(
                self.flat_ne.0,
                self.flat_ne.1,
                -planet::altitude_of(before.position),
            ),
            velocity: v0,
            attitude: q0,
            angular_rate: (q0.inverse() * q1).scaled_axis() / dt,
        };
        let accel_world = (v1 - v0) / dt;
        self.nz = -(q0.inverse() * (accel_world - Vec3::new(0.0, 0.0, GRAVITY))).z / GRAVITY;
        let bundle = Truth {
            state: &flat,
            accel_world,
            t: self.time(),
        };
        let k = self.k;
        if k.is_multiple_of(self.p_imu) {
            let m = self.imu.sample(&bundle);
            self.ins.predict(&m, self.p_imu as Real * dt);
            self.last_imu = Some(m);
        }
        if k.is_multiple_of(self.p_mag) {
            self.ins.update_mag(&self.mag.sample(&bundle));
        }
        if k.is_multiple_of(self.p_baro) {
            let b = self.baro.sample(&bundle);
            self.last_baro = b.altitude;
            self.ins.update_baro(&b);
        }
        if k.is_multiple_of(self.p_gps) {
            self.ins.update_gps(&self.gps.sample(&bundle));
        }
        self.k += 1;
        true
    }

    /// The network inputs ([`FEATURES`]), once the IMU has sampled.
    fn features(&self) -> Option<[Real; 16]> {
        let m = self.last_imu?;
        let est = self.ins.state();
        let vb = est.attitude.inverse() * est.velocity;
        let (roll, pitch, _) = est.attitude.euler_angles();
        let c = self.sim.controls();
        Some([
            m.accel.x,
            m.accel.y,
            m.accel.z,
            m.gyro.x,
            m.gyro.y,
            m.gyro.z,
            self.last_baro,
            vb.x,
            vb.y,
            vb.z,
            roll,
            pitch,
            c.elevator,
            c.aileron,
            c.throttle,
            alpha_of(vb),
        ])
    }

    /// True air-relative AoA \[rad\].
    fn alpha_true(&self) -> Real {
        let s = self.sim.truth();
        alpha_of(s.attitude.inverse() * (s.velocity - self.sim.wind_world()))
    }

    /// Today's FBW AoA: `atan2(w, u)` of the true *ground* velocity.
    fn alpha_ground(&self) -> Real {
        let s = self.sim.truth();
        alpha_of(s.attitude.inverse() * s.velocity)
    }
}

/// A ReLU MLP exported by `scripts/aoa_nn.py` (input scaler included).
struct Mlp {
    mean: Vec<Real>,
    scale: Vec<Real>,
    /// (weights in×out row-major, bias, in, out) per layer.
    layers: Vec<(Vec<Real>, Vec<Real>, usize, usize)>,
}

impl Mlp {
    /// Text format: line 1 = layer sizes (`16 32 32 1`), then all numbers:
    /// mean, scale, then per layer W (in×out, row-major) and b.
    fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("read weights");
        let (head, body) = text.split_once('\n').expect("weights header");
        let sizes: Vec<usize> = head
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let mut nums = body.split_whitespace().map(|s| s.parse::<Real>().unwrap());
        let mut take = |n: usize| nums.by_ref().take(n).collect::<Vec<_>>();
        assert_eq!(
            sizes[0],
            FEATURES.len(),
            "weights trained on other features"
        );
        let mean = take(sizes[0]);
        let scale = take(sizes[0]);
        let layers = sizes
            .windows(2)
            .map(|w| (take(w[0] * w[1]), take(w[1]), w[0], w[1]))
            .collect();
        Self {
            mean,
            scale,
            layers,
        }
    }

    fn predict(&self, f: &[Real; 16]) -> Real {
        let mut x: Vec<Real> = (0..f.len())
            .map(|i| (f[i] - self.mean[i]) / self.scale[i])
            .collect();
        for (i, (w, b, n_in, n_out)) in self.layers.iter().enumerate() {
            let last = i + 1 == self.layers.len();
            x = (0..*n_out)
                .map(|o| {
                    let z = b[o] + (0..*n_in).map(|j| x[j] * w[j * n_out + o]).sum::<Real>();
                    if last {
                        z
                    } else {
                        z.max(0.0)
                    }
                })
                .collect();
        }
        x[0]
    }
}

fn dataset(airframe: Airframe, path: &str, runs: u64) -> std::io::Result<()> {
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "run,t,wind,turb,{},alpha_true", FEATURES.join(","))?;
    for run in 0..runs {
        let mut rig = Rig::new(airframe, run);
        let p_log = ((1.0 / (LOG_HZ * rig.dt)).round() as u64).max(1);
        while rig.time() < RUN_SECS {
            if !rig.step() {
                eprintln!("run {run}: ground contact, run truncated");
                break;
            }
            if !rig.k.is_multiple_of(p_log) || rig.time() < SETTLE_SECS {
                continue;
            }
            let Some(f) = rig.features() else { continue };
            let cols: Vec<String> = f.iter().map(|v| format!("{v:.6}")).collect();
            writeln!(
                out,
                "{run},{:.3},{:.3},{:.3},{},{:.7}",
                rig.time(),
                rig.wind,
                rig.turb,
                cols.join(","),
                rig.alpha_true()
            )?;
        }
    }
    out.flush()?;
    eprintln!("wrote {path}");
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Source {
    Default,
    Ins,
    Network,
    TrueAir,
}

fn closed_loop(weights: &str, runs: u64) {
    let nn = Mlp::load(weights);
    println!("Fighter on fly-by-wire, {runs} unseen flights of {RUN_SECS} s, wind + turbulence.");
    println!("Same flights for every AoA source; only the SAS's AoA input changes.\n");
    println!(
        "{:<10}{:>14}{:>16}{:>14}{:>12}{:>9}",
        "AoA source", "AoA err (deg)", "pitch err (deg)", "load std (g)", "max AoA", "crashes"
    );
    for src in [
        Source::Default,
        Source::Ins,
        Source::Network,
        Source::TrueAir,
    ] {
        let (mut n, mut e_a, mut e_p, mut nz1, mut nz2, mut amax, mut crashes) =
            (0.0, 0.0, 0.0, 0.0, 0.0, Real::MIN, 0);
        // Test seeds start far from the training seeds (0..runs).
        for run in 1000..1000 + runs {
            let mut rig = Rig::new(Airframe::Fighter, run);
            let mut used = None;
            while rig.time() < RUN_SECS {
                if rig.k.is_multiple_of(rig.p_ctrl) {
                    let f = rig.features();
                    used = match src {
                        Source::Default => None,
                        Source::Ins => f.map(|f| f[ALPHA_INS]),
                        Source::Network => f.map(|f| nn.predict(&f)),
                        Source::TrueAir => Some(rig.alpha_true()),
                    };
                    rig.sim.set_fbw_alpha(used);
                }
                if !rig.step() {
                    crashes += 1;
                    break;
                }
                if rig.time() < SETTLE_SECS {
                    continue;
                }
                let a = rig.alpha_true();
                let err = used.unwrap_or_else(|| rig.alpha_ground()) - a;
                let (_, pitch) = rig.attitude();
                n += 1.0;
                e_a += err * err;
                e_p += (rig.target.0 - pitch).powi(2);
                nz1 += rig.nz;
                nz2 += rig.nz * rig.nz;
                amax = amax.max(a);
            }
        }
        let rms = |s: Real| (s / n).sqrt().to_degrees();
        println!(
            "{:<10}{:>14.2}{:>16.2}{:>14.3}{:>11.1}°{:>9}",
            format!("{src:?}"),
            rms(e_a),
            rms(e_p),
            (nz2 / n - (nz1 / n).powi(2)).sqrt(),
            amax.to_degrees(),
            crashes
        );
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |i: usize| args.get(i).map(String::as_str);
    let usage =
        "usage: aoa_nn dataset <aerosonde|fighter> out.csv [runs] | loop weights.txt [runs]";
    match (arg(0), arg(1)) {
        (Some("dataset"), Some(frame)) => {
            let airframe = match frame {
                "aerosonde" => Airframe::Aerosonde,
                "fighter" => Airframe::Fighter,
                _ => panic!("{usage}"),
            };
            let runs = arg(3).map_or(60, |s| s.parse().expect("runs"));
            dataset(airframe, arg(2).expect(usage), runs)
        }
        (Some("loop"), Some(weights)) => {
            closed_loop(weights, arg(2).map_or(20, |s| s.parse().expect("runs")));
            Ok(())
        }
        _ => panic!("{usage}"),
    }
}
