//! # Spherical planet model
//!
//! A 1/1000-scale Earth: a sphere of radius [`PLANET_RADIUS`] ≈ 6371 m
//! (~40 km circumference — a "tiny planet"). This module holds the pure,
//! deterministic, `no_std` maths the fixed-wing flies by:
//!
//! * **PCI (Planet-Centered Inertial):** the world frame. Cartesian, origin at
//!   the planet core, axes fixed in inertial space. `State13.position` /
//!   `velocity` live here for the fixed-wing; the attitude is `q_pci_from_body`.
//! * **Local NED:** the *tangent* frame at a position — North, East, Down — with
//!   Down = −r̂ (toward the core). It is **not stored**; it is derived from the
//!   current position whenever the autopilot or guidance needs a local horizon.
//!
//! The quad never uses this module: it stays in a flat local-NED frame near
//! home where the sphere is flat to ~1e-8 rad (see `gravity_world`).
//!
//! ## Why a fixed PCI frame (and not a rotating "world")
//!
//! Keeping the attitude in one *fixed* inertial frame for the whole flight means
//! the quaternion is never re-referenced — there is no state machine, no frame
//! hand-off. Curvature enters purely through the **change of basis**: as the
//! aircraft moves, the local NED basis [`ned_basis`] rotates, so "level" and
//! "course" continuously update while the integrator math stays frame-agnostic.
//!
//! All functions are pure (`no_std`, `libm`/`num_traits` trig, no RNG, no
//! wall-clock) so two runs reproduce bit-for-bit.

use crate::{Quat, Real, Vec3, GRAVITY};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion};
use num_traits::Float;

/// Planet radius \[m\]: Earth's mean radius (6371 km) at 1/1000 scale.
pub const PLANET_RADIUS: Real = 6371.0;

/// The shared "home" location both airframes and the terrain are anchored to:
/// latitude / longitude \[rad\]. The equator at the prime meridian — the home
/// surface point is the PCI `+x` axis `(R, 0, 0)`, which keeps the maths simple
/// and the terrain's flat clearing centred on a clean direction.
pub const HOME_LAT: Real = 0.0;
/// See [`HOME_LAT`].
pub const HOME_LON: Real = 0.0;

/// PCI position of the home point at altitude `alt` \[m\].
#[inline]
pub fn home_pci(alt: Real) -> Vec3 {
    geodetic_to_pci(HOME_LAT, HOME_LON, alt)
}

/// Standard gravitational parameter `GM = g0 · R²` so that surface gravity is
/// exactly [`GRAVITY`] and gravity falls off as the inverse square with radius.
/// At this tiny scale altitude is a large fraction of `R` (400 m ≈ 6 %), so the
/// inverse-square drop (~12 % at 400 m) is *not* negligible — we model it.
pub const PLANET_MU: Real = GRAVITY * PLANET_RADIUS * PLANET_RADIUS;

/// Radial, inverse-square gravity at a PCI position `p`: points toward the core
/// with magnitude `GM/|p|²` (= [`GRAVITY`] at the surface). Zero near the core
/// (guards the singularity; never reached in flight).
#[inline]
pub fn gravity_at(p: Vec3) -> Vec3 {
    let r2 = p.norm_squared();
    if r2 < 1.0 {
        Vec3::zeros()
    } else {
        let r = r2.sqrt();
        p * (-PLANET_MU / (r2 * r)) // -GM · p / |p|³
    }
}

/// Altitude above the (sea-level) planet surface \[m\]: `|p| − R`.
#[inline]
pub fn altitude_of(p: Vec3) -> Real {
    p.norm() - PLANET_RADIUS
}

/// Magnitude of gravity at altitude `alt` above the surface \[m/s²\] — the
/// inverse-square field (`= GRAVITY` at the surface, ~12 % weaker at 400 m on
/// this tiny planet). For controllers that need the local `|g|` (e.g. the
/// fixed-wing's coordinated-turn rate) without forming the full vector.
#[inline]
pub fn gravity_magnitude(alt: Real) -> Real {
    let r = PLANET_RADIUS + alt;
    PLANET_MU / (r * r)
}

/// Sea-level air density \[kg/m³\] — the reference `ρ₀` for [`density_at`]
/// (matches the Aerosonde's historical fixed value).
pub const RHO0: Real = 1.2682;

/// Atmosphere scale height `H` \[m\]: air density falls as `exp(−alt/H)`, halving
/// every `H·ln2 ≈ 485 m`. Deliberately *compressed* for this 1/1000-scale planet
/// so the flight envelope bites within the ~100–400 m band the aircraft actually
/// fly in — real Earth's ~8.5 km scale height would change density only a percent
/// or two over that range and be invisible. The model is a fictional thin shell,
/// not Earth's troposphere; it is the air-density analogue of the radial
/// inverse-square gravity above.
pub const ATMOSPHERE_SCALE_HEIGHT: Real = 700.0;

/// Air density at altitude `alt` \[m\]: an exponential atmosphere
/// `ρ = ρ₀·exp(−alt/H)`. Monotone-decreasing, always positive, `= ρ₀` at the
/// surface. Both lift and propeller thrust scale with this, so it gives a stall
/// speed that climbs with altitude *and* a service ceiling for free.
#[inline]
pub fn density_at(alt: Real) -> Real {
    RHO0 * Float::exp(-alt / ATMOSPHERE_SCALE_HEIGHT)
}

/// The local NED basis `(north, east, down)` at PCI position `p`, as unit PCI
/// vectors. `down = −p̂`; `north` is the polar axis (PCI `+z`) projected into the
/// local tangent plane (pointing toward the North pole); `east = down × north`.
/// At a pole `north` falls back to the prime-meridian direction.
pub fn ned_basis(p: Vec3) -> (Vec3, Vec3, Vec3) {
    let rho = p.norm();
    let down = if rho > 1.0 {
        -p / rho
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };
    let axis = Vec3::new(0.0, 0.0, 1.0); // PCI +z = North-pole axis
    let mut north = axis - down * axis.dot(&down);
    if north.norm() < 1e-9 {
        // On the pole axis: pick the prime-meridian direction in the tangent plane.
        let pm = Vec3::new(1.0, 0.0, 0.0);
        north = pm - down * pm.dot(&down);
    }
    let north = north.normalize();
    let east = down.cross(&north);
    (north, east, down)
}

/// Quaternion `q_ned_from_pci` at `p`: rotates a PCI vector into the local NED
/// frame (`v_ned = q · v_pci`). Composed with `q_pci_from_body` it yields the
/// body attitude *relative to the local horizon*.
#[inline]
pub fn ned_from_pci(p: Vec3) -> Quat {
    let (n, e, d) = ned_basis(p);
    // Columns = the NED basis expressed in PCI ⇒ this rotation maps NED → PCI.
    let r_pci_from_ned = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[n, e, d]));
    UnitQuaternion::from_rotation_matrix(&r_pci_from_ned).inverse()
}

/// Quaternion `q_pci_from_ned` at `p`: rotates a local-NED vector into PCI
/// (`v_pci = q · v_ned`). Used to place a locally-trimmed state onto the sphere.
#[inline]
pub fn pci_from_ned(p: Vec3) -> Quat {
    ned_from_pci(p).inverse()
}

/// PCI position of a geographic point: latitude / longitude \[rad\] and altitude
/// above the surface \[m\]. Latitude 0 = equator, `+z` = North pole;
/// longitude 0 along PCI `+x`.
#[inline]
pub fn geodetic_to_pci(lat: Real, lon: Real, alt: Real) -> Vec3 {
    let r = PLANET_RADIUS + alt;
    let cl = Float::cos(lat);
    Vec3::new(
        r * cl * Float::cos(lon),
        r * cl * Float::sin(lon),
        r * Float::sin(lat),
    )
}

/// Geographic `(lat, lon, alt)` \[rad, rad, m\] of a PCI position.
#[inline]
pub fn pci_to_geodetic(p: Vec3) -> (Real, Real, Real) {
    let rho = p.norm();
    let lat = Float::asin((p.z / rho).clamp(-1.0, 1.0));
    let lon = Float::atan2(p.y, p.x);
    (lat, lon, rho - PLANET_RADIUS)
}

// --- great-circle navigation -------------------------------------------------

/// Unit normal of the great circle through surface points `a` and `b`, oriented
/// for the forward direction `a → b`. Returns zero for a degenerate pair
/// (identical or antipodal points), which callers treat as "aim straight at it".
#[inline]
pub fn gc_normal(a: Vec3, b: Vec3) -> Vec3 {
    let c = a.cross(&b);
    let m = c.norm();
    if m > 1e-9 {
        c / m
    } else {
        Vec3::zeros()
    }
}

/// Local-NED course \[rad\] of the great-circle tangent *at `p`*, flowing forward
/// along the circle whose normal is `n`. Evaluating the tangent at the current
/// position (not a fixed origin bearing) is what lets a long leg track the arc
/// rather than cut the chord.
#[inline]
pub fn gc_course(p: Vec3, n: Vec3) -> Real {
    let pu = p.normalize();
    let t = n.cross(&pu); // tangent to the great circle, forward direction
    let (north, east, _down) = ned_basis(p);
    Float::atan2(t.dot(&east), t.dot(&north))
}

/// Signed cross-track distance \[m\] of `p` from the great circle with normal
/// `n`. Positive = **right** of the forward path direction (same sign as the
/// flat-earth `cross_track`, so the line-following law is unchanged).
#[inline]
pub fn gc_cross_track(p: Vec3, n: Vec3) -> Real {
    let pu = p.normalize();
    -PLANET_RADIUS * Float::asin(pu.dot(&n).clamp(-1.0, 1.0))
}

/// Great-circle (surface) distance \[m\] between the surface projections of `a`
/// and `b`.
#[inline]
pub fn gc_distance(a: Vec3, b: Vec3) -> Real {
    let c = a.normalize().dot(&b.normalize()).clamp(-1.0, 1.0);
    PLANET_RADIUS * Float::acos(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_PI_2, PI};

    fn surface(lat: Real, lon: Real) -> Vec3 {
        geodetic_to_pci(lat, lon, 0.0)
    }

    #[test]
    fn gravity_is_radial_and_inverse_square() {
        let p = surface(0.5, 1.2);
        let g = gravity_at(p);
        // Points toward the core (opposite the radial).
        let radial = p.normalize();
        assert!((g.normalize() + radial).norm() < 1e-9, "gravity not radial");
        // Surface magnitude is g0.
        assert!(
            (g.norm() - GRAVITY).abs() < 1e-6,
            "surface g = {}",
            g.norm()
        );
        // Inverse square: double the radius → quarter the magnitude.
        let g2 = gravity_at(p * 2.0);
        assert!((g.norm() / g2.norm() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn density_falls_exponentially_with_altitude() {
        // Surface density is exactly the reference.
        assert!((density_at(0.0) - RHO0).abs() < 1e-12);
        // Strictly monotone-decreasing with altitude, always positive.
        let (mut prev, mut a) = (density_at(0.0), 0.0);
        while a < 2000.0 {
            a += 50.0;
            let d = density_at(a);
            assert!(d > 0.0 && d < prev, "density not decreasing at {a} m: {d}");
            prev = d;
        }
        // Halves every H·ln2 ≈ 485 m.
        let half_height = ATMOSPHERE_SCALE_HEIGHT * core::f64::consts::LN_2;
        assert!((density_at(half_height) - RHO0 / 2.0).abs() < 1e-9);
    }

    #[test]
    fn ned_basis_is_orthonormal_right_handed() {
        for &(lat, lon) in &[(0.0, 0.0), (0.7, -2.0), (-1.1, 0.3), (1.2, 3.0)] {
            let p = surface(lat, lon);
            let (n, e, d) = ned_basis(p);
            for v in [n, e, d] {
                assert!((v.norm() - 1.0).abs() < 1e-12);
            }
            assert!(n.dot(&e).abs() < 1e-12);
            assert!(n.dot(&d).abs() < 1e-12);
            assert!(e.dot(&d).abs() < 1e-12);
            // Right-handed NED: N × E = D.
            assert!((n.cross(&e) - d).norm() < 1e-12);
            // Down points at the core.
            assert!((d + p.normalize()).norm() < 1e-12);
            // North has a non-negative polar (+z) component away from the poles.
            assert!(n.z >= -1e-12, "north should lean toward the pole: {:?}", n);
        }
    }

    #[test]
    fn ned_from_pci_rotates_velocity_and_round_trips() {
        let p = surface(0.4, -0.8);
        let q = ned_from_pci(p);
        let (north, east, down) = ned_basis(p);
        // A PCI vector along `north` must map to local +N = (1,0,0).
        assert!((q * north - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((q * east - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-9);
        assert!((q * down - Vec3::new(0.0, 0.0, 1.0)).norm() < 1e-9);
        // pci_from_ned is the inverse.
        let v = Vec3::new(3.0, -1.0, 2.0);
        assert!((pci_from_ned(p) * (q * v) - v).norm() < 1e-9);
    }

    #[test]
    fn geodetic_round_trips() {
        for &(lat, lon, alt) in &[(0.0, 0.0, 0.0), (0.6, 1.5, 400.0), (-0.9, -2.7, 120.0)] {
            let p = geodetic_to_pci(lat, lon, alt);
            let (la, lo, al) = pci_to_geodetic(p);
            assert!((la - lat).abs() < 1e-9, "lat {la} != {lat}");
            assert!((lo - lon).abs() < 1e-9, "lon {lo} != {lon}");
            assert!((al - alt).abs() < 1e-6, "alt {al} != {alt}");
        }
    }

    #[test]
    fn great_circle_distance_and_course() {
        let a = surface(0.0, 0.0); // equator, prime meridian
        let b = surface(0.0, FRAC_PI_2); // equator, 90° East
                                         // A quarter of the way around the planet.
        assert!((gc_distance(a, b) - PLANET_RADIUS * FRAC_PI_2).abs() < 1e-3);
        // Heading East along the equator: tangent course at `a` ≈ +π/2.
        let n = gc_normal(a, b);
        assert!(
            (gc_course(a, n) - FRAC_PI_2).abs() < 1e-6,
            "{}",
            gc_course(a, n)
        );
        // Toward the pole: course ≈ 0 (North).
        let pole = surface(FRAC_PI_2 - 1e-4, 0.0);
        let nn = gc_normal(a, pole);
        assert!(gc_course(a, nn).abs() < 1e-3, "{}", gc_course(a, nn));
    }

    #[test]
    fn cross_track_sign_matches_flat_convention() {
        // Forward leg due East along the equator (a → b).
        let a = surface(0.0, 0.0);
        let b = surface(0.0, 1.0);
        let n = gc_normal(a, b);
        // A point NORTH of the eastbound leg is to its LEFT ⇒ negative cross-track
        // (exactly as the flat `cross_track`/`course_sign_in_ned` test expects).
        let north_of = surface(0.05, 0.5);
        assert!(
            gc_cross_track(north_of, n) < 0.0,
            "north of an east leg should be negative (left): {}",
            gc_cross_track(north_of, n)
        );
        let south_of = surface(-0.05, 0.5);
        assert!(gc_cross_track(south_of, n) > 0.0);
        // On the leg: ~zero.
        let on = surface(0.0, 0.5);
        assert!(gc_cross_track(on, n).abs() < 1e-3);
    }

    #[test]
    fn antipodal_and_identical_normals_are_degenerate() {
        let a = surface(0.3, 0.4);
        assert_eq!(gc_normal(a, a), Vec3::zeros());
        assert_eq!(gc_normal(a, -a), Vec3::zeros());
        let _ = PI;
    }
}
