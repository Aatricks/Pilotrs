//! Dataset for a learned angle-of-attack (AoA) estimator.
//!
//! Flies the Aerosonde through random setpoints, wind and turbulence, runs the
//! simulated IMU/GPS/baro/mag + the 15-state INS (Kalman) beside it, and writes
//! one CSV row at 50 Hz with: raw sensors, the INS estimate, the controls, the
//! INS-derived AoA (`atan2(w, u)` of the *ground* velocity — what the FBW uses)
//! and the true *air-relative* AoA. `scripts/aoa_nn.py` trains on it.
//!
//! Run with `cargo run -p fsim-sim --release --example aoa_dataset -- out.csv [runs]`.
//!
//! Sensors and INS assume a flat world, but the fixed-wing flies on the small
//! sphere. So each step we build a consistent flat "virtual truth": local-NED
//! velocity and attitude, body rate *relative to local NED* and dv/dt by finite
//! difference, position = integrated NED velocity with altitude as −z.

use fsim_estimator::{Estimator, Ins, InsConfig};
use fsim_sensors::{
    Baro, BaroConfig, Gps, GpsConfig, Imu, ImuConfig, Mag, MagConfig, Sensor, Truth,
};
use fsim_sim::{FixedWingSetpoint, FwSim, FwSimConfig, Quat, Real, State13, Vec3};
use fsim_core::planet;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Uniform};
use std::io::Write;

const RUN_SECS: Real = 60.0;
const SETPOINT_EVERY: Real = 6.0;
const LOG_HZ: Real = 50.0;

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

fn main() -> std::io::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| "aoa_dataset.csv".into());
    let runs: u64 = args.next().map_or(60, |s| s.parse().expect("runs"));
    let mut out = std::io::BufWriter::new(std::fs::File::create(&path)?);
    writeln!(
        out,
        "run,t,wind,turb,ax,ay,az,gx,gy,gz,baro,ins_ub,ins_vb,ins_wb,ins_roll,ins_pitch,\
         elevator,aileron,throttle,alpha_ins,alpha_true"
    )?;

    for run in 0..runs {
        let mut rng = ChaCha8Rng::seed_from_u64(0xA0A_0000 + run);
        let mut cfg = FwSimConfig::aerosonde_cruise();
        cfg.atmosphere.seed = 0xB0B_0000 + run;
        let dt = cfg.dt;
        let mut sim = FwSim::new(cfg);
        // Every 10th run is calm air: the baseline where INS AoA should be right.
        let calm = run % 10 == 0;
        let wind = if calm { 0.0 } else { u(&mut rng, 0.0, 12.0) };
        let turb = if calm { 0.0 } else { u(&mut rng, 0.0, 3.0) };
        let dir = u(&mut rng, -3.2, 3.2);
        sim.set_wind(Vec3::new(wind * dir.cos(), wind * dir.sin(), 0.0));
        sim.set_turbulence(turb);

        let seed = 0xC0C_0000 + run * 16;
        let mut imu = Imu::new(ImuConfig::realistic(1000.0), seed);
        let mut gps = Gps::new(GpsConfig::realistic(10.0), seed + 1);
        let mut baro = Baro::new(BaroConfig::realistic(50.0), seed + 2);
        let mut mag = Mag::new(MagConfig::realistic(100.0), seed + 3);
        let mut ins = Ins::new(InsConfig::default());
        let period = |hz: Real| ((1.0 / (hz * dt)).round() as u64).max(1);
        let (p_imu, p_gps, p_baro, p_mag, p_log) = (
            period(1000.0),
            period(10.0),
            period(50.0),
            period(100.0),
            period(LOG_HZ),
        );
        let p_sp = period(1.0 / SETPOINT_EVERY);

        let mut flat_ne = (0.0, 0.0);
        let mut last_imu = None;
        let mut last_baro = 0.0;
        let steps = (RUN_SECS / dt) as u64;
        for k in 0..steps {
            if k % p_sp == 0 && k > 0 {
                sim.set_setpoint(FixedWingSetpoint {
                    airspeed: u(&mut rng, 18.0, 32.0),
                    altitude: u(&mut rng, 70.0, 160.0),
                    course: u(&mut rng, -3.2, 3.2),
                });
            }
            let before = *sim.truth();
            sim.step();
            if sim.crashed() || sim.altitude() < 5.0 {
                eprintln!("run {run}: ground contact, run truncated");
                break;
            }
            let after = *sim.truth();
            let (v0, q0) = local(&before);
            let (v1, q1) = local(&after);
            flat_ne.0 += v0.x * dt;
            flat_ne.1 += v0.y * dt;
            let flat = State13 {
                position: Vec3::new(flat_ne.0, flat_ne.1, -planet::altitude_of(before.position)),
                velocity: v0,
                attitude: q0,
                angular_rate: (q0.inverse() * q1).scaled_axis() / dt,
            };
            let bundle = Truth {
                state: &flat,
                accel_world: (v1 - v0) / dt,
                t: k as Real * dt,
            };
            if k % p_imu == 0 {
                let m = imu.sample(&bundle);
                ins.predict(&m, p_imu as Real * dt);
                last_imu = Some(m);
            }
            if k % p_mag == 0 {
                ins.update_mag(&mag.sample(&bundle));
            }
            if k % p_baro == 0 {
                let b = baro.sample(&bundle);
                last_baro = b.altitude;
                ins.update_baro(&b);
            }
            if k % p_gps == 0 {
                ins.update_gps(&gps.sample(&bundle));
            }

            if k % p_log == 0 {
                let Some(m) = last_imu else { continue };
                let est = ins.state();
                let vb_ins = est.attitude.inverse() * est.velocity;
                let (roll, pitch, _) = est.attitude.euler_angles();
                let v_air_body = after.attitude.inverse() * (after.velocity - sim.wind_world());
                let c = sim.controls();
                writeln!(
                    out,
                    "{run},{:.3},{wind:.3},{turb:.3},{:.5},{:.5},{:.5},{:.6},{:.6},{:.6},{:.3},\
                     {:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6},{:.5},{:.7},{:.7}",
                    sim.time(),
                    m.accel.x,
                    m.accel.y,
                    m.accel.z,
                    m.gyro.x,
                    m.gyro.y,
                    m.gyro.z,
                    last_baro,
                    vb_ins.x,
                    vb_ins.y,
                    vb_ins.z,
                    roll,
                    pitch,
                    c.elevator,
                    c.aileron,
                    c.throttle,
                    alpha_of(vb_ins),
                    alpha_of(v_air_body),
                )?;
            }
        }
    }
    out.flush()?;
    eprintln!("wrote {path}");
    Ok(())
}
