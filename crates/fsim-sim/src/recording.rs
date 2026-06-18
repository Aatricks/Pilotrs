//! Telemetry recording + replay.
//!
//! A [`Recording`] is just the full [`TelemetrySample`] stream, so it replays
//! **without** a running `Sim` (it depends only on `fsim-core` types) and
//! survives future estimator changes. The on-disk format is a dependency-light,
//! human-inspectable CSV: a 3-line header (magic + version, metadata, column
//! names) followed by one row per sample. Every `f64` is written as `{:.17e}`,
//! which round-trips IEEE-754 **bit-for-bit** (asserted in tests), so two
//! identical headless runs produce byte-identical files.

use crate::telemetry::{Telemetry, TelemetrySample};
use fsim_core::{EstState, Quat, Real, State13, Vec3};
use nalgebra::{Quaternion, UnitQuaternion};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::Path;

/// On-disk format version.
pub const RECORDING_VERSION: u32 = 1;

const MAGIC: &str = "#pilotrs-recording";

/// The 42 data columns, in order. A single source of truth for writer, reader,
/// and the header-validation check — they cannot drift.
const COLUMNS: [&str; 42] = [
    "t",
    "tru_px",
    "tru_py",
    "tru_pz",
    "tru_vx",
    "tru_vy",
    "tru_vz",
    "tru_qw",
    "tru_qx",
    "tru_qy",
    "tru_qz",
    "tru_wx",
    "tru_wy",
    "tru_wz",
    "est_px",
    "est_py",
    "est_pz",
    "est_vx",
    "est_vy",
    "est_vz",
    "est_qw",
    "est_qx",
    "est_qy",
    "est_qz",
    "est_wx",
    "est_wy",
    "est_wz",
    "sp_qw",
    "sp_qx",
    "sp_qy",
    "sp_qz",
    "sp_thrust",
    "m0",
    "m1",
    "m2",
    "m3",
    "tgb_x",
    "tgb_y",
    "tgb_z",
    "egb_x",
    "egb_y",
    "egb_z",
];

/// A recorded run: the full telemetry stream.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recording {
    pub samples: Vec<TelemetrySample>,
}

fn inval(line: usize, msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("line {line}: {msg}"))
}

/// Project a sample to its 42 column values in order.
fn sample_values(s: &TelemetrySample) -> [f64; 42] {
    let (t, e, sp) = (&s.truth, &s.estimate, &s.setpoint);
    let tq = t.attitude.as_ref();
    let eq = e.attitude.as_ref();
    let pq = sp.attitude.as_ref();
    [
        s.t,
        t.position.x,
        t.position.y,
        t.position.z,
        t.velocity.x,
        t.velocity.y,
        t.velocity.z,
        tq.w,
        tq.i,
        tq.j,
        tq.k,
        t.angular_rate.x,
        t.angular_rate.y,
        t.angular_rate.z,
        e.position.x,
        e.position.y,
        e.position.z,
        e.velocity.x,
        e.velocity.y,
        e.velocity.z,
        eq.w,
        eq.i,
        eq.j,
        eq.k,
        e.angular_rate.x,
        e.angular_rate.y,
        e.angular_rate.z,
        pq.w,
        pq.i,
        pq.j,
        pq.k,
        sp.thrust,
        s.motors[0],
        s.motors[1],
        s.motors[2],
        s.motors[3],
        s.true_gyro_bias.x,
        s.true_gyro_bias.y,
        s.true_gyro_bias.z,
        s.est_gyro_bias.x,
        s.est_gyro_bias.y,
        s.est_gyro_bias.z,
    ]
}

/// Rebuild a sample from its 42 parsed values. Attitudes use `new_unchecked`
/// (the stored quaternions are already unit, from `UnitQuaternion`) so the
/// round-trip is bit-exact rather than perturbed by a re-normalization.
fn sample_from_values(v: &[f64; 42]) -> TelemetrySample {
    let vec = |a: usize| Vec3::new(v[a], v[a + 1], v[a + 2]);
    let quat = |a: usize| -> Quat {
        UnitQuaternion::new_unchecked(Quaternion::new(v[a], v[a + 1], v[a + 2], v[a + 3]))
    };
    TelemetrySample {
        t: v[0],
        truth: State13 {
            position: vec(1),
            velocity: vec(4),
            attitude: quat(7),
            angular_rate: vec(11),
        },
        estimate: EstState {
            position: vec(14),
            velocity: vec(17),
            attitude: quat(20),
            angular_rate: vec(24),
        },
        setpoint: crate::Setpoint {
            attitude: quat(27),
            thrust: v[31],
        },
        motors: [v[32], v[33], v[34], v[35]],
        true_gyro_bias: vec(36),
        est_gyro_bias: vec(39),
    }
}

fn write_row(out: &mut String, vals: &[f64; 42]) {
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Writing to a String is infallible.
        let _ = write!(out, "{v:.17e}");
    }
    out.push('\n');
}

fn row_to_sample(line: &str, line_no: usize) -> io::Result<TelemetrySample> {
    let mut vals = [0.0_f64; 42];
    let mut it = line.split(',');
    for slot in vals.iter_mut() {
        let tok = it.next().ok_or_else(|| inval(line_no, "too few columns"))?;
        *slot = tok
            .trim()
            .parse::<f64>()
            .map_err(|_| inval(line_no, "non-numeric field"))?;
    }
    if it.next().is_some() {
        return Err(inval(line_no, "too many columns"));
    }
    Ok(sample_from_values(&vals))
}

impl Recording {
    pub fn from_telemetry(t: Telemetry) -> Self {
        Self { samples: t.samples }
    }

    pub fn from_samples(samples: Vec<TelemetrySample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Span of the recording \[s\] (0 if fewer than two samples).
    pub fn duration(&self) -> Real {
        match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => b.t - a.t,
            _ => 0.0,
        }
    }

    /// Write the recording (testable without a filesystem).
    pub fn write<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "{MAGIC} v{RECORDING_VERSION}")?;
        writeln!(w, "#format=csv kind=telemetry columns={}", COLUMNS.len())?;
        writeln!(w, "{}", COLUMNS.join(","))?;
        let mut buf = String::with_capacity(42 * 24);
        for s in &self.samples {
            buf.clear();
            write_row(&mut buf, &sample_values(s));
            w.write_all(buf.as_bytes())?;
        }
        Ok(())
    }

    /// Read a recording, validating the header. Malformed input returns an
    /// `InvalidData` error naming the offending line — never panics.
    pub fn read<R: Read>(r: R) -> io::Result<Self> {
        let mut lines = BufReader::new(r).lines();

        let l1 = lines.next().ok_or_else(|| inval(1, "empty file"))??;
        let mut it = l1.split_whitespace();
        if it.next() != Some(MAGIC) {
            return Err(inval(1, "not a pilotrs recording (bad magic)"));
        }
        let ver = it
            .next()
            .and_then(|s| s.strip_prefix('v'))
            .and_then(|s| s.parse::<u32>().ok());
        if ver != Some(RECORDING_VERSION) {
            return Err(inval(1, "unsupported recording version"));
        }

        let l2 = lines.next().ok_or_else(|| inval(2, "missing metadata"))??;
        if !l2.starts_with('#') {
            return Err(inval(2, "missing metadata line"));
        }

        let l3 = lines
            .next()
            .ok_or_else(|| inval(3, "missing column header"))??;
        if l3.trim() != COLUMNS.join(",") {
            return Err(inval(3, "column header does not match schema"));
        }

        let mut samples = Vec::new();
        let mut line_no = 3;
        for line in lines {
            line_no += 1;
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            samples.push(row_to_sample(&line, line_no)?);
        }
        Ok(Self { samples })
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        self.write(File::create(path)?)
    }

    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Self::read(File::open(path)?)
    }
}

/// Plays a [`Recording`] back by simulated time — pure index/time math, no
/// `Sim`, sensors, or RNG.
pub struct ReplayPlayer<'a> {
    samples: &'a [TelemetrySample],
    t: Real,
    speed: Real,
    looping: bool,
}

impl<'a> ReplayPlayer<'a> {
    pub fn new(rec: &'a Recording) -> Self {
        Self {
            samples: &rec.samples,
            t: rec.samples.first().map(|s| s.t).unwrap_or(0.0),
            speed: 1.0,
            looping: false,
        }
    }

    pub fn with_speed(mut self, s: Real) -> Self {
        self.speed = s;
        self
    }

    pub fn looping(mut self, on: bool) -> Self {
        self.looping = on;
        self
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn duration(&self) -> Real {
        match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => b.t - a.t,
            _ => 0.0,
        }
    }

    /// Nominal sample spacing \[s\] (0 if fewer than two samples).
    pub fn frame_dt(&self) -> Real {
        if self.samples.len() >= 2 {
            self.samples[1].t - self.samples[0].t
        } else {
            0.0
        }
    }

    pub fn time(&self) -> Real {
        self.t
    }

    /// The latest sample at or before simulated time `t` (binary search on the
    /// monotonic timestamps). `None` if `t` precedes the first sample.
    pub fn sample_at(&self, t: Real) -> Option<&'a TelemetrySample> {
        let count = self.samples.partition_point(|s| s.t <= t);
        if count == 0 {
            None
        } else {
            Some(&self.samples[count - 1])
        }
    }

    /// The sample at the current playhead.
    pub fn current(&self) -> Option<&'a TelemetrySample> {
        self.sample_at(self.t)
    }

    /// Advance the playhead by a wall-clock frame interval (scaled by speed),
    /// clamping at the end (or wrapping if looping).
    pub fn advance(&mut self, dt_frame: Real) {
        let (first, last) = match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => (a.t, b.t),
            _ => return,
        };
        self.t += dt_frame * self.speed;
        if self.t > last {
            if self.looping {
                self.t = first + (self.t - first) % (last - first).max(1e-9);
            } else {
                self.t = last;
            }
        }
    }

    pub fn seek(&mut self, t: Real) {
        let (first, last) = match (self.samples.first(), self.samples.last()) {
            (Some(a), Some(b)) => (a.t, b.t),
            _ => return,
        };
        self.t = t.clamp(first, last);
    }

    pub fn reset(&mut self) {
        self.t = self.samples.first().map(|s| s.t).unwrap_or(0.0);
    }

    /// Every sample once, in order (for headless verify / re-record).
    pub fn iter_all(&self) -> std::slice::Iter<'a, TelemetrySample> {
        self.samples.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Setpoint;
    use nalgebra::UnitQuaternion;

    fn sample(t: f64, seed: f64) -> TelemetrySample {
        let q = UnitQuaternion::from_euler_angles(0.1 + seed, -0.2, 0.3 + seed);
        TelemetrySample {
            t,
            truth: State13 {
                position: Vec3::new(seed, seed * 2.0, -seed),
                velocity: Vec3::new(0.5, -0.25, seed),
                attitude: q,
                angular_rate: Vec3::new(0.01, -0.02, 0.03),
            },
            estimate: EstState {
                position: Vec3::new(seed + 0.1, seed * 2.0, -seed),
                velocity: Vec3::new(0.5, -0.25, seed),
                attitude: q,
                angular_rate: Vec3::new(0.01, -0.02, 0.03),
            },
            setpoint: Setpoint {
                attitude: UnitQuaternion::from_euler_angles(0.05, 0.0, seed),
                thrust: 4.9 + seed,
            },
            motors: [1.0, 2.0, 3.0, 4.0 + seed],
            true_gyro_bias: Vec3::new(0.01, -0.008, 0.005),
            est_gyro_bias: Vec3::new(0.009, -0.007, 0.004),
        }
    }

    fn rec(n: usize) -> Recording {
        Recording::from_samples(
            (0..n)
                .map(|i| sample(i as f64 * 0.01, i as f64 * 0.013))
                .collect(),
        )
    }

    #[test]
    fn round_trip_is_bit_exact() {
        let r = rec(50);
        let mut buf = Vec::new();
        r.write(&mut buf).unwrap();
        let back = Recording::read(&buf[..]).unwrap();
        assert_eq!(back.len(), r.len());
        for (a, b) in r.samples.iter().zip(&back.samples) {
            for (x, y) in sample_values(a).iter().zip(sample_values(b).iter()) {
                assert_eq!(x.to_bits(), y.to_bits(), "field not bit-exact");
            }
        }
    }

    #[test]
    fn two_identical_recordings_write_identical_bytes() {
        let mut a = Vec::new();
        let mut b = Vec::new();
        rec(20).write(&mut a).unwrap();
        rec(20).write(&mut b).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn quaternion_round_trips() {
        let r = rec(1);
        let mut buf = Vec::new();
        r.write(&mut buf).unwrap();
        let back = Recording::read(&buf[..]).unwrap();
        assert!(
            back.samples[0]
                .truth
                .attitude
                .angle_to(&r.samples[0].truth.attitude)
                < 1e-15
        );
    }

    #[test]
    fn empty_and_single_recordings() {
        for n in [0, 1] {
            let r = rec(n);
            let mut buf = Vec::new();
            r.write(&mut buf).unwrap();
            let back = Recording::read(&buf[..]).unwrap();
            assert_eq!(back.len(), n);
            assert_eq!(back.duration(), 0.0);
        }
    }

    #[test]
    fn special_values_round_trip() {
        let mut s = sample(0.0, 0.0);
        s.truth.position.x = f64::NAN;
        s.truth.position.y = f64::INFINITY;
        s.truth.position.z = f64::NEG_INFINITY;
        let r = Recording::from_samples(vec![s]);
        let mut buf = Vec::new();
        r.write(&mut buf).unwrap();
        let back = Recording::read(&buf[..]).unwrap();
        assert!(back.samples[0].truth.position.x.is_nan());
        assert!(
            back.samples[0].truth.position.y.is_infinite()
                && back.samples[0].truth.position.y > 0.0
        );
        assert!(
            back.samples[0].truth.position.z.is_infinite()
                && back.samples[0].truth.position.z < 0.0
        );
    }

    #[test]
    fn malformed_input_errors_without_panic() {
        assert!(Recording::read(&b""[..]).is_err());
        assert!(Recording::read(&b"garbage\n"[..]).is_err());
        assert!(Recording::read(&b"#pilotrs-recording v2\n#x\nhdr\n"[..]).is_err());
        let mut good = Vec::new();
        rec(2).write(&mut good).unwrap();
        // Corrupt a data row (drop a column).
        let text = String::from_utf8(good).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines[3] = "1.0,2.0,3.0"; // too few columns
        let corrupt = lines.join("\n");
        assert!(Recording::read(corrupt.as_bytes()).is_err());
    }

    #[test]
    fn replay_sample_at_boundaries() {
        let r = rec(10); // t = 0.00, 0.01, ..., 0.09
        let p = ReplayPlayer::new(&r);
        assert!(p.sample_at(-1.0).is_none(), "before first → None");
        assert_eq!(p.sample_at(0.0).unwrap().t, 0.0, "exact");
        assert_eq!(
            p.sample_at(0.025).unwrap().t.to_bits(),
            0.02_f64.to_bits(),
            "latest <= t"
        );
        assert_eq!(
            p.sample_at(100.0).unwrap().t.to_bits(),
            0.09_f64.to_bits(),
            "after last → last"
        );
    }

    #[test]
    fn replay_advance_and_stop() {
        let r = rec(10);
        let mut p = ReplayPlayer::new(&r);
        for _ in 0..1000 {
            p.advance(0.01);
        }
        assert_eq!(
            p.time().to_bits(),
            0.09_f64.to_bits(),
            "playhead stops at end"
        );
    }

    #[test]
    fn replay_iter_all_reproduces_samples() {
        let r = rec(30);
        let p = ReplayPlayer::new(&r);
        let back: Vec<_> = p.iter_all().cloned().collect();
        assert_eq!(back.len(), 30);
        for (a, b) in r.samples.iter().zip(&back) {
            assert_eq!(a.t.to_bits(), b.t.to_bits());
        }
    }
}
