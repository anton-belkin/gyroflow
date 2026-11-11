// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2026 Adrian <adrian.eddy at gmail>

//! The motion data correction measured from the video itself ("Analyze image optically", see
//! `synchronization::optical_motion`).
//!
//! Some cameras record motion data that is right at low frequencies and wrong above a few Hz: a hard-mounted DJI O4
//! picks up motor vibration as rotation in two of its axes and misses part of the real one, and nothing in the
//! metadata says so. The image does. The correction is the small rotation that makes the quaternions agree with it,
//! `q' = q * exp(δ(t))`, with δ in the quaternions' own frame on the motion data timeline, stored as a uniform cubic
//! B-spline. It's measured against one particular set of quaternions, so it only applies to those: anything that
//! changes them (another integration method, a filter, the IMU orientation) makes it stale, see `quats_checksum`. And
//! against one particular picture of them - which pixel was read out when, and which ray it came from: another sync,
//! lens profile or rolling shutter time makes it stale too, see `context_checksum`

use nalgebra::{ UnitQuaternion, Vector3 };
use super::TimeQuat;

/// What only the user can judge: nothing in the metadata says how far the motion data can be trusted
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OpticalCorrectionSettings {
    /// How far the correction may take the motion data from what it says, 0 to 1. Low keeps it to the small, fast
    /// errors of a vibrating gyro; high lets the image override the motion data down to slow motion and by degrees - a
    /// gyro whose gain collapses during a roll, glitches - at the cost of following the image's own mistakes too
    pub strength: f64,
    /// One correction per frame instead of one per band of rows: it can't follow what changes within a frame
    /// (vibration), but doesn't need every band of rows measured either - for a global shutter, too little texture
    /// to track in parts of the frame, or a rolling shutter time that's wrong or unknown
    pub per_frame: bool,
}
impl Default for OpticalCorrectionSettings {
    fn default() -> Self { Self { strength: 0.5, per_frame: false } }
}

/// How the analysis follows the points from one frame to the next. Unlike the settings above it takes another
/// analysis: what it changes is what's measured
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrackingMethod {
    /// Pyramidal Lucas-Kanade at corners: the most accurate where there's texture to track
    #[default]
    Klt,
    /// Dense optical flow (DIS), sampled at the same points: a flow everywhere, also where corners are few and weak
    /// (sky, water, fog), at a few times the time
    Dis,
}
impl TrackingMethod {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() { "klt" => Some(Self::Klt), "dis" => Some(Self::Dis), _ => None }
    }
    pub fn name(&self) -> &'static str {
        match self { Self::Klt => "klt", Self::Dis => "dis" }
    }
}

#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct OpticalCorrection {
    /// Applied at all. Switched off rather than dropped, so the analysis survives toggling it
    pub enabled: bool,
    /// The settings it was fitted with
    pub settings: OpticalCorrectionSettings,
    /// Motion data timestamp of the first control point, microseconds
    pub start_us: f64,
    /// Distance between the control points, microseconds
    pub spacing_us: f64,
    /// Control points of δ: rotation vectors in the quaternions' frame, radians. Stored quantized: six of them per
    /// frame make a long clip's worth heavy for a project file
    #[serde(with = "quantized")]
    pub coeffs: Vec<[f32; 3]>,
    /// `checksum` of the quaternions the correction was measured against
    pub quats_checksum: u64,
    /// And of everything else it was measured with, see `synchronization::optical_motion::context_checksum`
    pub context_checksum: u64,
    /// For a file without motion data: the rotation the analysis measured between the frames, chained - the orientation
    /// the correction sits on. One per frame, (timestamp µs, w x y z); `integrate` uses it densified, see `base_quats`
    pub video_base: Vec<(i64, [f32; 4])>,
    /// How the analysis followed the points. Only to tell the user it changed since: the correction still applies, it's
    /// just measured the other way
    pub tracking: TrackingMethod,

    /// Frames the analysis tracked, and how many of them yielded a measurement
    pub frames: usize,
    pub measured_frames: usize,
    /// Rms of the correction where it was measured, degrees
    pub rms_deg: f64,
}

impl OpticalCorrection {
    /// δ at a motion data timestamp. Zero outside of the analyzed range: there the motion data is used as it is
    pub fn at(&self, timestamp_us: f64) -> Vector3<f64> {
        if self.coeffs.is_empty() || self.spacing_us <= 0.0 { return Vector3::zeros(); }
        let u = (timestamp_us - self.start_us) / self.spacing_us;
        let (s, w) = bspline_weights(u);
        let mut v = Vector3::zeros();
        for (i, w) in w.iter().enumerate() {
            let k = s + i as i64 - 1;
            if k >= 0 && (k as usize) < self.coeffs.len() {
                let c = self.coeffs[k as usize];
                v += Vector3::new(c[0] as f64, c[1] as f64, c[2] as f64) * *w;
            }
        }
        v
    }

    /// The orientation of a file without motion data, as the renderer gets it: `video_base` at 1 kHz
    pub fn base_quats(&self) -> TimeQuat { base_quats(&self.video_base) }

    /// Whether it was measured on (uncorrected) quaternions of this fingerprint, in this context (`context_checksum`).
    /// Up to the last bits another machine computes them with, see `Fingerprint::matches`
    pub fn measured_on(&self, quats: &Fingerprint, context: &Fingerprint) -> bool {
        !self.coeffs.is_empty() && quats.matches(self.quats_checksum) && context.matches(self.context_checksum)
    }
    /// What of `measured_on` doesn't match, for the log
    pub fn mismatch(&self, quats: &Fingerprint, context: &Fingerprint) -> String {
        let mut what = Vec::new();
        if !quats.matches(self.quats_checksum) { what.push(format!("the motion data ({:016x} here, {:016x} measured)", quats.checksum(), self.quats_checksum)); }
        if !context.matches(self.context_checksum) { what.push(format!("the sync, lens or frame timing ({:016x} here, {:016x} measured)", context.checksum(), self.context_checksum)); }
        what.join(" and ")
    }

    pub fn apply(&self, quats: &mut TimeQuat) {
        for (ts, q) in quats.iter_mut() {
            let d = self.at(*ts as f64);
            if d != Vector3::zeros() {
                *q *= UnitQuaternion::from_scaled_axis(d);
            }
        }
    }

    pub fn hash_into(&self, hasher: &mut impl std::hash::Hasher) {
        hasher.write_u8(self.enabled as u8);
        hasher.write_u64(self.settings.strength.to_bits());
        hasher.write_u8(self.settings.per_frame as u8);
        hasher.write_u64(self.quats_checksum);
        hasher.write_u64(self.context_checksum);
        hasher.write_u64(self.start_us.to_bits());
        hasher.write_u64(self.spacing_us.to_bits());
        hasher.write_usize(self.video_base.len());
        for (t, q) in self.video_base.iter().step_by((self.video_base.len() / 256).max(1)) {
            hasher.write_i64(*t); hasher.write_u32(q[0].to_bits()); hasher.write_u32(q[1].to_bits()); hasher.write_u32(q[2].to_bits());
        }
        hasher.write_usize(self.coeffs.len());
        for c in self.coeffs.iter().step_by((self.coeffs.len() / 512).max(1)) {
            hasher.write_u32(c[0].to_bits()); hasher.write_u32(c[1].to_bits()); hasher.write_u32(c[2].to_bits());
        }
    }
}

/// Control points as 16-bit integers of a common scale: the full range of the largest one, which leaves the rest a
/// resolution far below a pixel
mod quantized {
    use serde::{ Deserialize, Deserializer, Serialize, Serializer };
    #[derive(Serialize, Deserialize)]
    struct Q { scale: f32, v: Vec<i16> }

    pub fn serialize<S: Serializer>(c: &Vec<[f32; 3]>, s: S) -> Result<S::Ok, S::Error> {
        let max = c.iter().flatten().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if max > 0.0 { max / i16::MAX as f32 } else { 1.0 };
        Q { scale, v: c.iter().flatten().map(|v| (v / scale).round() as i16).collect() }.serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[f32; 3]>, D::Error> {
        let q = Q::deserialize(d)?;
        Ok(q.v.chunks_exact(3).map(|c| [c[0] as f32 * q.scale, c[1] as f32 * q.scale, c[2] as f32 * q.scale]).collect())
    }
}

/// Uniform cubic B-spline at the knot coordinate `u`: the segment `s` and the weights of control points `s-1 ..= s+2`
pub fn bspline_weights(u: f64) -> (i64, [f64; 4]) {
    let s = u.floor();
    let f = u - s;
    let g = 1.0 - f;
    (s as i64, [
        g * g * g / 6.0,
        2.0 / 3.0 - f * f + f * f * f / 2.0,
        2.0 / 3.0 - g * g + g * g * g / 2.0,
        f * f * f / 6.0,
    ])
}

/// A `video_base` as the renderer gets it. The analysis fingerprints the base this way too, from the stored single
/// precision, or a rounding would make it look like other motion data
pub fn base_quats(video_base: &[(i64, [f32; 4])]) -> TimeQuat {
    densify(&video_base.iter().map(|(t, q)| (*t, UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64)))).collect())
}

/// Orientations sampled at the frames, at 1 kHz: the geodesic between each two, so a lookup between the samples gives
/// what it would have between the frames
pub fn densify(sparse: &TimeQuat) -> TimeQuat {
    let mut out = TimeQuat::new();
    let mut it = sparse.iter().peekable();
    while let Some((&t0, q0)) = it.next() {
        out.insert(t0, *q0);
        if let Some(&(&t1, q1)) = it.peek() {
            let mut t = t0 - t0.rem_euclid(1000) + 1000;
            while t < t1 {
                out.insert(t, q0.slerp(q1, (t - t0) as f64 / (t1 - t0) as f64));
                t += 1000;
            }
        }
    }
    out
}

/// `quats` held at their first and last orientation over the rest of the clip, `0..=duration_ms`, on the grid of
/// `densify`. The smoothing takes the sample rate from how many samples there are per the clip's duration: the motion
/// of an analyzed trim range alone smooths as if sampled at a fraction of its rate - next to not at all
pub fn hold_over_clip(quats: &mut TimeQuat, duration_ms: f64) {
    let (Some((&first, &q0)), Some((&last, &q1))) = (quats.first_key_value(), quats.last_key_value()) else { return };
    let end = (duration_ms * 1000.0).round() as i64;
    let mut t = first - first.rem_euclid(1000);
    if t == first { t -= 1000; }
    while t >= 0 { quats.insert(t, q0); t -= 1000; }
    let mut t = last - last.rem_euclid(1000) + 1000;
    while t <= end { quats.insert(t, q1); t += 1000; }
}

/// Fingerprint of a set of quaternions. Stored in project files, so it's FNV-1a rather than std's hasher, whose
/// output may change between Rust releases, and it rounds the components, so the last bit of a filter computed on
/// another machine doesn't make a correction stale (`Fingerprint::matches`)
pub fn checksum(quats: &TimeQuat) -> u64 {
    fingerprint(quats).checksum()
}
/// What `checksum` is taken of
pub fn fingerprint(quats: &TimeQuat) -> Fingerprint {
    let mut h = Fingerprint::default();
    h.eat(quats.len() as u64);
    for (ts, q) in quats.iter().step_by((quats.len() / 1024).max(1)) {
        h.eat(*ts as u64);
        for v in q.coords.iter() {
            h.eat_rounded(*v, 1e6);
        }
    }
    h
}

/// What a fingerprint stored in a project (`checksum`, `optical_motion::context_checksum`) is taken of: its words in
/// order, the rounded ones with the value they were rounded from. Rounding alone doesn't make it portable: another
/// machine (another CPU, libm or GPU backend) computes the last bits of a value differently, and where that value is
/// near the middle between two units it rounds the other way and the checksum is another one: `matches` takes the
/// values nearest to the middle either way
#[derive(Clone, Debug, Default)]
pub struct Fingerprint {
    words: Vec<Word>,
    /// The last `matches`: the search it may take is done once per checksum
    matched: std::sync::Arc<parking_lot::Mutex<Option<(u64, bool)>>>,
}
#[derive(Clone, Copy, Debug)]
enum Word {
    Exact(u64),
    /// In units of the resolution, before the rounding
    Rounded(f64),
}
impl Word {
    fn word(&self) -> u64 {
        match *self { Word::Exact(v) => v, Word::Rounded(v) => v.round() as i64 as u64 }
    }
    /// Rounded the other way: the unit on the other side of the middle it's nearest to
    fn other(&self) -> u64 {
        match *self {
            Word::Exact(v) => v,
            Word::Rounded(v) => { let r = v.round(); (if v >= r { r + 1.0 } else { r - 1.0 }) as i64 as u64 },
        }
    }
    /// How near the middle between two units it is, for a value of its size: the distance in units, relative to the
    /// value, as its last bits are (a value up to a million units counts as one of a million). None for an exact one
    fn nearness(&self) -> Option<f64> {
        match *self { Word::Exact(_) => None, Word::Rounded(v) => Some(((v - v.floor()) - 0.5).abs() / v.abs().max(1e6)) }
    }
}
impl Fingerprint {
    /// How many words the search may hash on each side, all tries together: some milliseconds, also for a correction
    /// that is stale (then it's searched once per context, `matched`)
    const BUDGET: usize = 1 << 20;

    pub fn eat(&mut self, v: u64) { self.words.push(Word::Exact(v)); }
    /// `v` in whole units of `1 / scale`
    pub fn eat_rounded(&mut self, v: f64, scale: f64) { self.words.push(Word::Rounded(v * scale)); }
    /// Nothing taken yet: not known
    pub fn is_empty(&self) -> bool { self.words.is_empty() }

    pub fn checksum(&self) -> u64 {
        self.words.iter().fold(Fnv::default().0, |h, w| fnv_eat(h, w.word()))
    }

    /// Whether `checksum` is this fingerprint's as another machine may have computed it: with the values nearest to
    /// the middle between two units (`Word::nearness`) rounded either way, as many of them as `BUDGET` affords - 26
    /// of the context's 88, 14 of the quaternions'. Every combination of them is tried, meeting in the middle: those
    /// of the first half of them hashed forward from the start, those of the second half unwound from `checksum`
    /// (FNV-1a is invertible), a match where the two meet
    pub fn matches(&self, checksum: u64) -> bool {
        if self.words.is_empty() { return false; }
        if let Some((c, m)) = *self.matched.lock() { if c == checksum { return m; } }
        let m = self.checksum() == checksum || self.search(checksum);
        *self.matched.lock() = Some((checksum, m));
        m
    }
    fn search(&self, checksum: u64) -> bool {
        let mut near: Vec<(f64, usize)> = self.words.iter().enumerate().filter_map(|(i, w)| Some((w.nearness()?, i))).collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        // The larger half's 2^((k + 1) / 2) tries, over up to all of the words
        let mut k = 0;
        while k < near.len() && (self.words.len() << ((k + 2) / 2)) <= Self::BUDGET { k += 1; }
        let mut at: Vec<usize> = near[..k].iter().map(|(_, i)| *i).collect();
        if at.is_empty() { return false; }
        at.sort_unstable();
        let step = |states: Vec<u64>, i: usize, f: fn(u64, u64) -> u64| -> Vec<u64> {
            let w = self.words[i];
            if at.binary_search(&i).is_ok() { states.iter().flat_map(|&h| [f(h, w.word()), f(h, w.other())]).collect() }
            else { states.into_iter().map(|h| f(h, w.word())).collect() }
        };
        let split = at[at.len() / 2];
        let forward: std::collections::HashSet<u64> = (0..split).fold(vec![Fnv::default().0], |h, i| step(h, i, fnv_eat)).into_iter().collect();
        let backward = (split..self.words.len()).rev().fold(vec![checksum], |h, i| step(h, i, fnv_uneat));
        backward.iter().any(|h| forward.contains(h))
    }
}

const FNV_PRIME: u64 = 0x100000001b3;
/// Its inverse modulo 2^64, by Newton's iteration (each one doubles the bits that are right)
const FNV_PRIME_INVERSE: u64 = {
    let mut x = FNV_PRIME;
    let mut i = 0;
    while i < 6 { x = x.wrapping_mul(2u64.wrapping_sub(FNV_PRIME.wrapping_mul(x))); i += 1; }
    x
};
/// `h` with the word `v` hashed in
fn fnv_eat(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}
/// The `h` that `fnv_eat(h, v)` was
fn fnv_uneat(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes().into_iter().rev() {
        h = h.wrapping_mul(FNV_PRIME_INVERSE) ^ b as u64;
    }
    h
}

/// FNV-1a over 64-bit words, for the fingerprints kept in project files
pub struct Fnv(pub u64);
impl Default for Fnv {
    fn default() -> Self { Self(0xcbf29ce484222325) }
}
impl Fnv {
    pub fn eat(&mut self, v: u64) {
        self.0 = fnv_eat(self.0, v);
    }
    /// `v` in whole units of `1 / scale`
    pub fn eat_rounded(&mut self, v: f64, scale: f64) {
        self.eat((v * scale).round() as i64 as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(values: &[f64]) -> Fingerprint {
        let mut f = Fingerprint::default();
        f.eat(values.len() as u64);
        for v in values { f.eat_rounded(*v, 1e6); }
        f
    }

    #[test]
    fn fingerprint_checksum_is_the_old_one() {
        let values: [f64; 4] = [0.1234564999, -0.5, 0.25, 1e-7];
        let mut h = Fnv::default();
        h.eat(values.len() as u64);
        for v in values { h.eat((v * 1e6).round() as i64 as u64); }
        assert_eq!(of(&values).checksum(), h.0);
    }

    #[test]
    fn last_bits_elsewhere_still_match() {
        // Measured where these came out a hair below the middle, recomputed here a hair above it
        let there: Vec<f64> = (0..300).map(|i| i as f64 * 1e-5).chain([0.1234564999, -0.0000025001, 0.9876545]).collect();
        let here: Vec<f64> = (0..300).map(|i| i as f64 * 1e-5).chain([0.1234565001, -0.0000024999, 0.98765451]).collect();
        let stored = of(&there).checksum();
        assert_ne!(of(&here).checksum(), stored);
        assert!(of(&here).matches(stored));
        assert!(of(&there).matches(stored));
        // Memoized either way
        let f = of(&here);
        assert!(f.matches(stored) && f.matches(stored));
    }

    #[test]
    fn fnv_unwinds() {
        assert_eq!(FNV_PRIME.wrapping_mul(FNV_PRIME_INVERSE), 1);
        let h = fnv_eat(fnv_eat(Fnv::default().0, 7), u64::MAX - 3);
        assert_eq!(fnv_uneat(fnv_uneat(h, u64::MAX - 3), 7), Fnv::default().0);
    }

    #[test]
    fn many_last_bits_elsewhere_still_match() {
        // Twenty values a hair from the middle, on the other side of it here, among values on a unit
        let base = |i: usize| i as f64 * 1e-5;
        let there: Vec<f64> = (0..300).map(|i| if i % 15 == 7 { base(i) + 0.5e-6 - 1e-10 * (i as f64 + 1.0) } else { base(i) }).collect();
        let here: Vec<f64> = (0..300).map(|i| if i % 15 == 7 { base(i) + 0.5e-6 + 1e-10 * (i as f64 + 1.0) } else { base(i) }).collect();
        let stored = of(&there).checksum();
        assert_ne!(of(&here).checksum(), stored);
        assert!(of(&here).matches(stored));
    }

    #[test]
    fn a_real_change_does_not_match() {
        let there: Vec<f64> = (0..300).map(|i| i as f64 * 1e-5).collect();
        let mut here = there.clone();
        here[17] += 3e-6;
        let f = of(&here);
        assert!(!f.matches(of(&there).checksum()));
        assert!(!f.matches(of(&there).checksum()));
        assert!(!Fingerprint::default().matches(Fingerprint::default().checksum()));
    }
}
