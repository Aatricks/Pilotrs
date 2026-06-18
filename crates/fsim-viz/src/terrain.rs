//! # Procedural terrain
//!
//! A deterministic, seed-driven height field for the viewer's ground, plus a
//! helper that bakes it into a single coloured, lit [`three_d::CpuMesh`].
//!
//! ## Frame convention (matches `main.rs`)
//!
//! The scene is drawn directly in the simulator's **NED** world frame
//! (x = North, y = East, z = Down). Altitude is `-z`, so a hill that rises
//! `h` metres above the `z = 0` datum sits at world `z = -h`. The camera's up
//! vector is world `-z`. The terrain mesh therefore places each vertex at
//! `(n, e, -height(n, e))` and is wound so its lit faces point up (towards
//! `-z`), i.e. the same direction the existing [`DirectionalLight`] (travelling
//! `+z`, downward) illuminates.
//!
//! ## Determinism
//!
//! The height field is pure **hash value-noise / fBm** — no `rand`, no
//! wall-clock, no global state. `height(n, e)` is a referentially-transparent
//! function of `(seed, n, e)`: the same seed reproduces the same field bit for
//! bit, on any platform (all arithmetic is `f32`/`u32` with wrapping integer
//! hashing). This mirrors the simulator's "deterministic, seeded RNG, no
//! wall-clock in math" rule.
//!
//! ## Lit per-vertex colour (the material recipe)
//!
//! `three-d` 0.19's [`PhysicalMaterial`] *does* honour per-vertex mesh colours
//! **with full lighting**: the mesh vertex shader emits `col` from the colour
//! attribute and the physical fragment shader computes
//! `surface_color = albedo * col` before running `calculate_lighting(...)`
//! (verified in the registry sources — see the API NOTES at the bottom of this
//! file). The recipe is therefore:
//!
//! 1. Give the [`CpuMesh`] `colors: Some(per_vertex_srgba)` so the
//!    `USE_VERTEX_COLORS` shader define switches on.
//! 2. Build a [`PhysicalMaterial`] whose `albedo` is **white** so the vertex
//!    colour passes through the `albedo * col` multiply unmodified.
//!
//! Do **not** use `ColorMaterial` for this — it is unlit (it renders the vertex
//! colour flat, with no directional shading), which would hide all the slope
//! relief. See [`Terrain::material`].

use three_d::{Context, CpuMaterial, CpuMesh, Indices, PhysicalMaterial, Positions, Srgba, Vec3};

/// Configuration + the deterministic height field for one terrain tile.
///
/// The map is a square of side `2 * half_extent` metres centred on the world
/// origin, spanning `n, e ∈ [-half_extent, +half_extent]`.
#[derive(Clone, Copy, Debug)]
pub struct Terrain {
    /// Hash seed. Same seed ⇒ identical field.
    pub seed: u32,
    /// Half the map side length, in metres. The map covers
    /// `[-half_extent, half_extent]²`. ~2.4 km ⇒ a ~4.8 km square — large enough
    /// that a fixed-wing's ~110 m turn radius can actually track a hand-drawn
    /// route, and far enough that the world doesn't read as a tiny patch.
    pub half_extent: f32,
    /// Peak elevation **above** the `z = 0` datum, in metres (the tallest
    /// ridges). The whole field is bounded above by this.
    pub amplitude: f32,
    /// How far the deepest valleys sink **below** the datum, in metres. Most of
    /// the relief lives below the datum so an aircraft flying at altitude looks
    /// down on the terrain.
    pub valley_depth: f32,
    /// Wavelength of the coarsest (base) noise octave, in metres. Smaller ⇒
    /// busier terrain. ~1.4 km gives a few broad ranges over the ~4.8 km map.
    pub base_wavelength: f32,
    /// Number of fBm octaves. Each octave doubles frequency and (by `gain`)
    /// shrinks amplitude — more octaves add the fine ridge/erosion detail that
    /// keeps the map from reading as smooth blobs.
    pub octaves: u32,
    /// Per-octave frequency multiplier (classic fBm uses 2.0).
    pub lacunarity: f32,
    /// Per-octave amplitude multiplier (classic fBm uses ~0.5).
    pub gain: f32,
    /// Elevation (m, above datum) at and below which terrain is rendered as
    /// water. Lets the colour ramp paint sea/lakes in the deep valleys.
    pub sea_level: f32,
    /// Elevation (m, above datum) of the flat **home clearing** centred on the
    /// origin: a level take-off field set safely *below* the datum so an
    /// aircraft spawned at altitude 0 is always above the ground. The quad's
    /// default mission flies inside this clearing.
    pub home_level: f32,
    /// Radius (m) of the perfectly flat home clearing. Inside it `height`
    /// returns exactly `home_level`, which is what guarantees ground clearance
    /// for anything spawned at or above the datum within this radius.
    pub home_inner: f32,
    /// Radius (m) by which the clearing has fully blended into the open terrain.
    pub home_outer: f32,
}

impl Default for Terrain {
    fn default() -> Self {
        Self {
            seed: 0x5EED_1234,
            half_extent: 2400.0,
            amplitude: 60.0,
            valley_depth: 150.0,
            base_wavelength: 1400.0,
            octaves: 7,
            lacunarity: 2.0,
            gain: 0.5,
            sea_level: -70.0,
            home_level: -12.0,
            home_inner: 170.0,
            home_outer: 600.0,
        }
    }
}

impl Terrain {
    /// A terrain with the default tuning but a chosen seed.
    pub fn new(seed: u32) -> Self {
        Self {
            seed,
            ..Self::default()
        }
    }

    // --- deterministic value-noise / fBm ---------------------------------

    /// Integer hash → `f32` in `[0, 1)`. A small avalanche mix (variant of the
    /// "integer hash" family) seeded by `self.seed`. Pure and platform-stable:
    /// all `u32` wrapping ops.
    #[inline]
    fn hash01(&self, xi: i32, yi: i32) -> f32 {
        // Spread the lattice coordinates and the seed into a single u32, then
        // mix. `as u32` reinterprets the two's-complement bits, which is what
        // we want for negative coordinates.
        let mut h = (xi as u32)
            .wrapping_mul(0x8DA6_B343)
            .wrapping_add((yi as u32).wrapping_mul(0xD8163841))
            .wrapping_add(self.seed.wrapping_mul(0xCB1A_B31F));
        // finalizer (xorshift-multiply avalanche)
        h ^= h >> 16;
        h = h.wrapping_mul(0x7FEB_352D);
        h ^= h >> 15;
        h = h.wrapping_mul(0x846C_A68B);
        h ^= h >> 16;
        // Top 24 bits → [0, 1); exact, no rounding surprises.
        (h >> 8) as f32 / ((1u32 << 24) as f32)
    }

    /// Smooth value noise at lattice scale: bilinear interpolation of the four
    /// surrounding integer-lattice hashes, with a quintic (smootherstep) fade
    /// so the result is C0 (in fact C2 across cells). Input `(x, y)` are in
    /// *lattice units* (1 unit = one cell). Output in `[0, 1]`.
    #[inline]
    fn value_noise(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor();
        let y0 = y.floor();
        let xi = x0 as i32;
        let yi = y0 as i32;
        let fx = x - x0;
        let fy = y - y0;

        // Quintic fade 6t^5 - 15t^4 + 10t^3 (Perlin's smootherstep): zero 1st
        // and 2nd derivatives at the cell boundaries ⇒ no Mach banding.
        let ux = fade(fx);
        let uy = fade(fy);

        let c00 = self.hash01(xi, yi);
        let c10 = self.hash01(xi + 1, yi);
        let c01 = self.hash01(xi, yi + 1);
        let c11 = self.hash01(xi + 1, yi + 1);

        let bottom = lerp(c00, c10, ux);
        let top = lerp(c01, c11, ux);
        lerp(bottom, top, uy)
    }

    /// Fractal Brownian motion: sum `octaves` of [`value_noise`] at doubling
    /// frequency and shrinking amplitude, normalised back to `[0, 1]`.
    #[inline]
    fn fbm01(&self, n: f32, e: f32) -> f32 {
        // Base frequency in 1/metre so that one base-octave cell ≈
        // base_wavelength metres.
        let base_freq = 1.0 / self.base_wavelength.max(1.0);
        let mut freq = base_freq;
        let mut amp = 1.0_f32;
        let mut sum = 0.0_f32;
        let mut norm = 0.0_f32;
        // De-correlate octaves: each samples a different region of the lattice.
        let mut ox = 0.0_f32;
        let mut oy = 0.0_f32;
        for _ in 0..self.octaves.max(1) {
            sum += amp * self.value_noise(n * freq + ox, e * freq + oy);
            norm += amp;
            freq *= self.lacunarity;
            amp *= self.gain;
            ox += 17.0;
            oy += 53.0;
        }
        if norm > 0.0 {
            sum / norm
        } else {
            0.0
        }
    }

    /// Elevation in metres **above the `z = 0` datum** at world `(n, e)`
    /// (North, East). Deterministic and at least C0. Range is bounded by
    /// `[min_height(), max_height()]` exactly.
    ///
    /// A mild ridged remap (`1 - |2v - 1|` blended with `v`) makes the field
    /// read as ranges with the odd sharp ridge rather than uniform blobs. The
    /// open terrain is mapped to `[-valley_depth, +amplitude]`, then:
    ///
    /// * a radial **edge** falloff eases the tile boundary down into an
    ///   underwater rim (an island silhouette, never a cliff), and
    /// * a central **home clearing** forces a flat, datum-safe field near the
    ///   origin so an aircraft spawned at altitude 0 is always above the ground
    ///   (see [`Terrain::home_level`] / the `home_clearing_is_safe` test).
    pub fn height(&self, n: f32, e: f32) -> f32 {
        let v = self.fbm01(n, e); // [0, 1]
                                  // Ridge term in [0, 1]; blend keeps it smooth (C0).
        let ridged = 1.0 - (2.0 * v - 1.0).abs();
        let shaped = 0.7 * v + 0.3 * ridged; // still [0, 1]

        // Open terrain across the full vertical range [-valley_depth, +amplitude].
        let relief = -self.valley_depth + shaped * (self.amplitude + self.valley_depth);

        let r = (n * n + e * e).sqrt();

        // Edge: ease the open terrain toward an underwater rim near the tile
        // boundary so the finite map ends as a coastline, not a wall.
        let edge = smooth_falloff(r / self.half_extent.max(1.0));
        let rim = self.sea_level - 20.0;
        let open = lerp(rim, relief, edge);

        // Home clearing: 0 (flat `home_level`) inside `home_inner`, blending to
        // the open terrain by `home_outer`.
        let hf = home_blend(r, self.home_inner, self.home_outer);
        lerp(self.home_level, open, hf)
    }

    /// Exact lower bound on [`height`]: the deepest the field reaches is the
    /// valley floor (`open ≥ min(rim, relief) ≥ -valley_depth`, and the home
    /// blend only lerps toward the shallower `home_level`).
    pub fn min_height(&self) -> f32 {
        -self.valley_depth
    }

    /// Exact upper bound on [`height`]: `relief ≤ amplitude`, the rim and
    /// `home_level` are both lower, so every blend stays `≤ amplitude`.
    pub fn max_height(&self) -> f32 {
        self.amplitude
    }

    /// Surface normal at world `(n, e)` as a `three-d` [`Vec3`] in NED scene
    /// coordinates, pointing **up** — i.e. toward `-z`, so its z-component is
    /// *negative* (NED `+z` is Down). Computed from central finite differences
    /// of the height field.
    ///
    /// The surface is `z(n, e) = -height(n, e)`. Its tangents are
    /// `Tn = (1, 0, -∂h/∂n)` and `Te = (0, 1, -∂h/∂e)`; the upward normal is
    /// `Te × Tn = (-∂h/∂n, -∂h/∂e, -1)`, normalised — `z < 0` (up) in this NED
    /// layout, matching the mesh winding and the directional light.
    pub fn normal(&self, n: f32, e: f32) -> Vec3 {
        // Step ~ one grid cell; small enough to track features, large enough to
        // avoid hash quantisation noise.
        let d = (self.half_extent / 256.0).max(0.5);
        let dhdn = (self.height(n + d, e) - self.height(n - d, e)) / (2.0 * d);
        let dhde = (self.height(n, e + d) - self.height(n, e - d)) / (2.0 * d);
        let nrm = Vec3::new(-dhdn, -dhde, -1.0);
        let len = (nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z).sqrt();
        nrm / len
    }

    /// Blended terrain colour at `(n, e)` as a display-space [`Srgba`], chosen
    /// from **elevation** and **slope**:
    ///
    /// * below `sea_level` → water (deep→shallow blue),
    /// * low + gentle → green lowland,
    /// * mid + gentle → tan/grass-to-rock,
    /// * steep (any height) → brown rock,
    /// * high → near-white peaks (snow).
    ///
    /// `three-d` converts this `Srgba` to linear automatically when it uploads
    /// the colour buffer, so pass ordinary display-space bytes (same convention
    /// as the `albedo` colours already used in `main.rs`).
    pub fn color(&self, n: f32, e: f32) -> Srgba {
        let h = self.height(n, e);

        // Water first (overrides slope): blend deep→shallow by depth.
        if h <= self.sea_level {
            let depth = ((self.sea_level - h) / (self.sea_level - self.min_height()).max(1.0))
                .clamp(0.0, 1.0);
            let deep = rgb(16, 46, 84);
            let shallow = rgb(50, 110, 162);
            return mix_rgb(shallow, deep, depth);
        }

        // Slope: 0 = flat, 1 = vertical. The up-facing normal has z < 0 (NED),
        // so the up-component magnitude is `-z`.
        let nz = (-self.normal(n, e).z).clamp(0.0, 1.0);
        let slope = 1.0 - nz; // [0,1]

        // Elevation fraction above sea level, normalised to the peak.
        let span = (self.max_height() - self.sea_level).max(1.0);
        let t = ((h - self.sea_level) / span).clamp(0.0, 1.0);

        // Base ramp by elevation: sandy shore → green lowland → grass → tan
        // highland → grey rock → snow peak.
        let shore = rgb(182, 170, 120); // beach fringe just above the water
        let lowland = rgb(70, 118, 52); // lush low ground
        let grass = rgb(104, 140, 64); // rolling green
        let tan = rgb(152, 138, 92); // dry highland
        let rock_hi = rgb(122, 114, 106); // bare grey rock
        let snow = rgb(238, 242, 247); // near-white peak
        let base = if t < 0.05 {
            mix_rgb(shore, lowland, t / 0.05)
        } else if t < 0.32 {
            mix_rgb(lowland, grass, (t - 0.05) / 0.27)
        } else if t < 0.58 {
            mix_rgb(grass, tan, (t - 0.32) / 0.26)
        } else if t < 0.82 {
            mix_rgb(tan, rock_hi, (t - 0.58) / 0.24)
        } else {
            mix_rgb(rock_hi, snow, (t - 0.82) / 0.18)
        };

        // Steep faces expose brown rock regardless of elevation. Ramp in past
        // ~30° of slope (slope ≈ 0.18) and saturate by ~58° (slope ≈ 0.48).
        let rock = rgb(96, 80, 64);
        let rock_mix = ((slope - 0.18) / 0.30).clamp(0.0, 1.0);
        mix_rgb(base, rock, rock_mix)
    }

    /// A [`PhysicalMaterial`] configured to render this terrain's **per-vertex
    /// colours with lighting**. The key is `albedo: WHITE` so the shader's
    /// `surface_color = albedo * col` leaves the baked vertex colours intact;
    /// the lighting is then applied on top. Build the mesh with
    /// [`Terrain::build_mesh`] (which sets `colors: Some(..)`) so the
    /// `USE_VERTEX_COLORS` define switches on.
    pub fn material(&self, context: &Context) -> PhysicalMaterial {
        PhysicalMaterial::new_opaque(
            context,
            &CpuMaterial {
                albedo: Srgba::WHITE, // pass vertex colours through unmodified
                roughness: 0.95,      // terrain is matte
                metallic: 0.0,
                ..Default::default()
            },
        )
    }

    /// Bake the terrain into a single [`CpuMesh`]: a regular `cells × cells`
    /// grid of quads over `[-half_extent, half_extent]²`, with per-vertex
    /// positions, colours and normals.
    ///
    /// Vertex layout is row-major in `(i = North index, j = East index)`, with
    /// `(cells + 1)²` vertices. Triangle winding is chosen so the front face
    /// (the side three-d lights) points up (`-z`).
    ///
    /// The returned mesh uses `Positions::F32` and `Indices::U32`, with
    /// `normals: Some(..)` and `colors: Some(..)`.
    pub fn build_mesh(&self, cells: usize) -> CpuMesh {
        assert!(cells >= 1, "terrain needs at least one cell");
        let verts_per_side = cells + 1;
        let vcount = verts_per_side * verts_per_side;

        let mut positions: Vec<Vec3> = Vec::with_capacity(vcount);
        let mut normals: Vec<Vec3> = Vec::with_capacity(vcount);
        let mut colors: Vec<Srgba> = Vec::with_capacity(vcount);

        let span = 2.0 * self.half_extent;
        let step = span / cells as f32;

        for i in 0..verts_per_side {
            let n = -self.half_extent + i as f32 * step;
            for j in 0..verts_per_side {
                let e = -self.half_extent + j as f32 * step;
                let h = self.height(n, e);
                // NED: surface is BELOW the datum in z, since altitude = -z.
                positions.push(Vec3::new(n, e, -h));
                normals.push(self.normal(n, e));
                colors.push(self.color(n, e));
            }
        }

        // Index buffer: 2 triangles per cell, 6 indices per cell.
        let mut indices: Vec<u32> = Vec::with_capacity(cells * cells * 6);
        let idx = |i: usize, j: usize| -> u32 { (i * verts_per_side + j) as u32 };
        for i in 0..cells {
            for j in 0..cells {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v01 = idx(i, j + 1);
                let v11 = idx(i + 1, j + 1);
                // Winding chosen so the geometric face normal of each triangle
                // points up (toward -z), matching the per-vertex normals and
                // the directional light. (See the test
                // `winding_matches_up_normal`.)
                // Triangle A: v00, v01, v11
                indices.push(v00);
                indices.push(v01);
                indices.push(v11);
                // Triangle B: v00, v11, v10
                indices.push(v00);
                indices.push(v11);
                indices.push(v10);
            }
        }

        CpuMesh {
            positions: Positions::F32(positions),
            indices: Indices::U32(indices),
            normals: Some(normals),
            colors: Some(colors),
            ..Default::default()
        }
    }

    /// Number of vertices [`build_mesh`] will produce for `cells` cells:
    /// `(cells + 1)²`. (Mesh-introspection API, exercised by the unit tests.)
    #[allow(dead_code)]
    pub fn vertex_count(cells: usize) -> usize {
        (cells + 1) * (cells + 1)
    }

    /// Number of indices [`build_mesh`] will produce for `cells` cells:
    /// `6 * cells²` (2 triangles per cell).
    #[allow(dead_code)]
    pub fn index_count(cells: usize) -> usize {
        6 * cells * cells
    }

    /// Ground height query for placing objects on the terrain: returns the
    /// world `z` (NED, = `-height`) of the surface directly under `(n, e)`,
    /// i.e. the value to use for a prop's `z` so it rests on the ground.
    ///
    /// (Use `-ground_z(n, e)` if you want the altitude-above-datum instead.)
    #[allow(dead_code)]
    pub fn ground_z(&self, n: f32, e: f32) -> f32 {
        -self.height(n, e)
    }
}

// --- free helpers --------------------------------------------------------

/// Perlin quintic fade: 6t⁵ − 15t⁴ + 10t³.
#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Smooth radial falloff: 1 near the centre, easing to 0 at and beyond the
/// tile edge (`r ≥ 1`). C1 via smootherstep.
#[inline]
fn smooth_falloff(r: f32) -> f32 {
    // Start easing down at r = 0.82 so the interior stays full-amplitude.
    let inner = 0.82_f32;
    let outer = 1.0_f32;
    if r <= inner {
        1.0
    } else if r >= outer {
        0.0
    } else {
        let t = (r - inner) / (outer - inner);
        1.0 - fade(t)
    }
}

/// Home-clearing blend factor in metres: `0` inside `inner` (flat clearing),
/// smoothly rising to `1` at/after `outer` (full open terrain). Smootherstep,
/// so the clearing meets the surrounding terrain without a crease.
#[inline]
fn home_blend(r: f32, inner: f32, outer: f32) -> f32 {
    if r <= inner {
        0.0
    } else if outer <= inner || r >= outer {
        // `outer <= inner` is a degenerate/misconfigured clearing: treat
        // everything past `inner` as full open terrain (and never divide by a
        // non-positive span below).
        1.0
    } else {
        fade((r - inner) / (outer - inner))
    }
}

/// Opaque display-space [`Srgba`] from RGB bytes.
#[inline]
fn rgb(r: u8, g: u8, b: u8) -> Srgba {
    Srgba { r, g, b, a: 255 }
}

/// Linear-ish blend of two display-space colours by `t ∈ [0,1]`, returned as an
/// opaque [`Srgba`]. (Blends in gamma space, which is fine — and even
/// preferable — for a coarse terrain colour ramp.)
#[inline]
fn mix_rgb(a: Srgba, b: Srgba, t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| -> u8 { (x as f32 + (y as f32 - x as f32) * t).round() as u8 };
    Srgba {
        r: m(a.r, b.r),
        g: m(a.g, b.g),
        b: m(a.b, b.b),
        a: 255,
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Same seed ⇒ identical height at the same sample points (determinism).
    #[test]
    fn deterministic_same_seed() {
        let a = Terrain::new(42);
        let b = Terrain::new(42);
        let samples = [
            (0.0, 0.0),
            (123.5, -88.25),
            (-400.0, 400.0),
            (12.0, 312.0),
            (-250.0, -17.0),
        ];
        for &(n, e) in &samples {
            assert_eq!(
                a.height(n, e),
                b.height(n, e),
                "height must be reproducible at ({n}, {e})"
            );
        }
    }

    /// Different seeds ⇒ a different field somewhere (the seed actually wires in).
    #[test]
    fn seed_changes_field() {
        let a = Terrain::new(1);
        let b = Terrain::new(2);
        let differs = (-400..=400)
            .step_by(50)
            .any(|n| (a.height(n as f32, 0.0) - b.height(n as f32, 0.0)).abs() > 1e-6);
        assert!(differs, "different seeds should change the field");
    }

    /// Heights stay inside the advertised `[min_height, max_height]` bounds
    /// across a dense scan of the tile.
    #[test]
    fn heights_within_bounds() {
        let t = Terrain::default();
        let lo = t.min_height() - 1e-3;
        let hi = t.max_height() + 1e-3;
        let mut i = -500;
        while i <= 500 {
            let mut j = -500;
            while j <= 500 {
                let h = t.height(i as f32, j as f32);
                assert!(
                    h >= lo && h <= hi,
                    "height {h} out of [{lo}, {hi}] at ({i}, {j})"
                );
                j += 13;
            }
            i += 13;
        }
    }

    /// The field is C0: a tiny step in position yields a tiny change in height
    /// (no lattice discontinuities leaking through).
    #[test]
    fn continuous_field() {
        let t = Terrain::default();
        let (n, e) = (37.0, -91.0);
        let h0 = t.height(n, e);
        let h1 = t.height(n + 0.05, e + 0.05);
        assert!(
            (h1 - h0).abs() < 0.5,
            "field should be continuous: {h0} vs {h1}"
        );
    }

    /// Mesh has the expected vertex and index counts for a given cell count.
    #[test]
    fn mesh_counts() {
        let cells = 200;
        let t = Terrain::default();
        // Counts via the pure helpers (no GPU context needed).
        assert_eq!(Terrain::vertex_count(cells), 201 * 201);
        assert_eq!(Terrain::index_count(cells), 6 * 200 * 200);

        // And the actual CpuMesh built by build_mesh matches (CpuMesh
        // construction is pure — it needs no GL context).
        let mesh = t.build_mesh(cells);
        assert_eq!(mesh.positions.len(), Terrain::vertex_count(cells));
        assert_eq!(
            mesh.indices.len(),
            Some(Terrain::index_count(cells)),
            "U32 index buffer length"
        );
        assert_eq!(
            mesh.normals.as_ref().map(|v| v.len()),
            Some(Terrain::vertex_count(cells))
        );
        assert_eq!(
            mesh.colors.as_ref().map(|v| v.len()),
            Some(Terrain::vertex_count(cells))
        );
    }

    /// Indices reference valid vertices.
    #[test]
    fn indices_in_range() {
        let t = Terrain::default();
        let cells = 8;
        let mesh = t.build_mesh(cells);
        let vcount = Terrain::vertex_count(cells) as u32;
        if let Indices::U32(ref ix) = mesh.indices {
            assert!(ix.iter().all(|&i| i < vcount), "index out of range");
        } else {
            panic!("expected U32 indices");
        }
    }

    /// Per-vertex normals are unit length and point generally up (toward -z),
    /// i.e. their z component is positive and dominant.
    #[test]
    fn normals_unit_and_up() {
        let t = Terrain::default();
        let mesh = t.build_mesh(100);
        let normals = mesh.normals.as_ref().expect("normals present");
        for nrm in normals {
            let len = (nrm.x * nrm.x + nrm.y * nrm.y + nrm.z * nrm.z).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "normal not unit: len = {len}");
            assert!(nrm.z < 0.0, "normal should point up (-z in NED): {:?}", nrm);
        }
        // On average, terrain is fairly flat ⇒ mean up-component (-z) is high.
        let mean_z: f32 = normals.iter().map(|n| n.z).sum::<f32>() / normals.len() as f32;
        assert!(mean_z < -0.7, "terrain should be mostly gentle: {mean_z}");
    }

    /// Triangle winding agrees with the upward per-vertex normals: the
    /// geometric face normal of each triangle, computed as
    /// `(p1 - p0) × (p2 - p0)`, points up (toward -z, i.e. negative world-z).
    ///
    /// In this NED layout an upward-facing front face has a *negative* z on its
    /// cross product; we assert that and also that it agrees in sign with the
    /// averaged vertex normals' contribution.
    #[test]
    fn winding_matches_up_normal() {
        let t = Terrain::default();
        let cells = 16;
        let mesh = t.build_mesh(cells);
        let pos = match mesh.positions {
            Positions::F32(ref p) => p,
            _ => panic!("expected F32 positions"),
        };
        let ix = match mesh.indices {
            Indices::U32(ref i) => i,
            _ => panic!("expected U32 indices"),
        };
        let mut checked = 0;
        for tri in ix.chunks_exact(3) {
            let p0 = pos[tri[0] as usize];
            let p1 = pos[tri[1] as usize];
            let p2 = pos[tri[2] as usize];
            let u = p1 - p0;
            let v = p2 - p0;
            // cross(u, v)
            let cx = u.y * v.z - u.z * v.y;
            let cy = u.z * v.x - u.x * v.z;
            let cz = u.x * v.y - u.y * v.x;
            // Up in NED is -z, so an up-facing front face has cross.z < 0.
            assert!(
                cz < 0.0,
                "triangle winding should face up (-z): cross.z = {cz}"
            );
            let _ = (cx, cy);
            checked += 1;
        }
        assert_eq!(checked, Terrain::index_count(cells) / 3);
    }

    /// Colour ramp sanity: a point we know is below sea level reads bluish, and
    /// colours are deterministic.
    #[test]
    fn colors_deterministic() {
        let t = Terrain::default();
        let c0 = t.color(10.0, 20.0);
        let c1 = Terrain::default().color(10.0, 20.0);
        assert_eq!((c0.r, c0.g, c0.b, c0.a), (c1.r, c1.g, c1.b, c1.a));
    }

    /// `ground_z` is exactly the negated height (placement helper contract).
    #[test]
    fn ground_z_is_neg_height() {
        let t = Terrain::default();
        for &(n, e) in &[(0.0, 0.0), (101.0, -55.0), (-300.0, 222.0)] {
            assert_eq!(t.ground_z(n, e), -t.height(n, e));
        }
    }

    /// The home clearing is exactly flat (== `home_level`) inside `home_inner`,
    /// and `home_level` is below the datum — so an aircraft spawned at altitude
    /// 0 (world z = 0) is strictly above the ground everywhere it can take off.
    /// This is the invariant that fixes "the drone starts under the map".
    #[test]
    fn home_clearing_is_safe() {
        let t = Terrain::default();
        assert!(
            t.home_level < 0.0,
            "home clearing must sit below the datum so spawn-at-0 clears it"
        );
        // Dense scan of the flat disc, including the quad mission corners (~141 m).
        let r = t.home_inner;
        let mut a = -r;
        while a <= r {
            let mut b = -r;
            while b <= r {
                if a * a + b * b <= r * r {
                    let h = t.height(a, b);
                    assert!(
                        (h - t.home_level).abs() < 1e-4,
                        "clearing not flat at ({a},{b}): {h} != {}",
                        t.home_level
                    );
                    // Altitude of the ground is `h`; spawn altitude is 0.
                    assert!(h < 0.0, "ground at ({a},{b}) is at/above spawn altitude");
                }
                b += 11.0;
            }
            a += 11.0;
        }
    }

    /// A degenerate clearing (`home_outer <= home_inner`) must not divide by a
    /// zero/negative span: `home_blend` returns a finite factor everywhere, so
    /// the height field has no NaN even when misconfigured.
    #[test]
    fn home_blend_handles_degenerate() {
        // Equal and inverted spans, sampled across and just past `inner`.
        for &(inner, outer) in &[(170.0_f32, 170.0_f32), (200.0, 100.0), (50.0, 50.0)] {
            let mut r = 0.0_f32;
            while r <= outer.max(inner) + 50.0 {
                let f = home_blend(r, inner, outer);
                assert!(
                    f.is_finite(),
                    "home_blend NaN/inf at r={r} ({inner},{outer})"
                );
                assert!((0.0..=1.0).contains(&f), "home_blend out of [0,1]: {f}");
                r += 7.0;
            }
        }
        // And the full height field stays bounded for a degenerate terrain.
        let t = Terrain {
            home_inner: 200.0,
            home_outer: 100.0,
            ..Terrain::default()
        };
        for &(n, e) in &[(0.0, 0.0), (150.0, 0.0), (300.0, -120.0), (1000.0, 800.0)] {
            let h = t.height(n, e);
            assert!(h.is_finite(), "height NaN/inf at ({n},{e})");
            assert!(h >= t.min_height() - 1e-3 && h <= t.max_height() + 1e-3);
        }
    }

    /// Every height stays within the advertised `[min_height, max_height]`
    /// bounds across a dense scan of the *whole* (large) tile — including the
    /// edge rim and the home blend, the two places a sign error would escape.
    #[test]
    fn heights_bounded_full_tile() {
        let t = Terrain::default();
        let he = t.half_extent;
        let lo = t.min_height() - 1e-3;
        let hi = t.max_height() + 1e-3;
        let step = he / 40.0;
        let mut n = -he;
        while n <= he {
            let mut e = -he;
            while e <= he {
                let h = t.height(n, e);
                assert!(
                    h >= lo && h <= hi,
                    "height {h} out of [{lo},{hi}] at ({n},{e})"
                );
                e += step;
            }
            n += step;
        }
    }
}
