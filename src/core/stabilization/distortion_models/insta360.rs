// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright © 2022 Adrian <adrian.eddy at gmail>

use crate::stabilization::KernelParams;

#[derive(Default, Clone)]
pub struct Insta360 { }

impl Insta360 {
    // The coefficients, in `k`: `k1 k2 k3 p1 p2 ξ` (Insta360's `offset_v3`, OpenCV's order), then the terms of
    // `offset_v6` (RadtanPro) on top, zero for a v3 profile: `k4 k5` (r⁸, r¹⁰), `q1 q2` (the tangential pair's r²
    // parts: p1 + q1·r², p2 + q2·r²), and the thin prism `s1 s2` (x: r², r⁴), `s3 s4` (y: r², r⁴). A RadtanPro
    // profile's 13 terms `d0..d12` are `k1 k2 k3 k4 k5 = d0..d4`, `p2 = d5, p1 = d6, q2 = d7, q1 = d8`,
    // `s1 = d9, s3 = d10, s2 = d11, s4 = d12` (video-types' `MeiDistortion::RadtanPro`)

    /// Radial part of the projection: image radius of a unified-sphere plane radius `m`
    #[inline] fn radial(m: f32, kr: &[f32; 5]) -> f32 {
        // The v3 terms as they always were, so a v3 profile solves as before
        let m2 = m * m;
        m * (1.0 + kr[0]*m2 + kr[1]*m2*m2 + kr[2]*m2*m2*m2 + kr[3]*m2*m2*m2*m2 + kr[4]*m2*m2*m2*m2*m2)
    }
    /// `d(radial)/dm`
    #[inline] fn radial_derivative(m: f32, kr: &[f32; 5]) -> f32 {
        let m2 = m * m;
        1.0 + 3.0*kr[0]*m2 + 5.0*kr[1]*m2*m2 + 7.0*kr[2]*m2*m2*m2 + 9.0*kr[3]*m2*m2*m2*m2 + 11.0*kr[4]*m2*m2*m2*m2*m2
    }
    /// The tangential and thin prism terms at `(x, y)`: what `distort_point` adds to the radial part
    #[inline] fn tangential(x: f32, y: f32, k: &[f32]) -> (f32, f32) {
        let r2 = x*x + y*y;
        let p1 = k[3] + k[8] * r2;
        let p2 = k[4] + k[9] * r2;
        (2.0*p1*x*y + p2*(r2 + 2.0*x*x) + r2*(k[10] + k[11]*r2),
         2.0*p2*x*y + p1*(r2 + 2.0*y*y) + r2*(k[12] + k[13]*r2))
    }
    /// How far the *sphere* reaches: it folds at `θ = acos(-1/ξ)` (120° for the X5's ξ = 2), and for ξ ≤ 1
    /// it never folds but runs to infinity at `acos(-ξ)`, which [`Self::field_limit`] puts an epsilon short
    /// of and which is what the placeholder below stands for.
    ///
    /// Only half the story, and never a bound on the *image* radius on its own: the radial polynomial has a
    /// fold of its own wherever `radial'` first reaches zero, and it can come first. `radial(m_max)` is the
    /// largest radius the lens images only where the curve is still rising there - `radial_derivative` at
    /// `m_max` says whether it is - and past the fold the polynomial says whatever a degree-7 polynomial
    /// says: with the X5's own `k₃ = -3` and this placeholder it says -3·10²⁸, which read as a maximum
    /// image radius rejects every pixel in the frame
    #[inline] fn m_max(xi: f32) -> f32 {
        if xi > 1.0 { 1.0 / (xi * xi - 1.0).sqrt() } else { 1e4 }
    }
    /// Newton on `radial(m) = r`, kept on the branch the camera projects with.
    ///
    /// The bracket is what keeps it there. `radial` leaves 0 with slope 1 and the camera images the stretch
    /// up to its first fold; a Newton step that lands where the curve has stopped rising has stepped over
    /// that fold, and pulling the bracket's far end back to there - rather than letting the step stand -
    /// walks up to the fold instead of across it. So the fold never has to be solved for up front (it is
    /// the first positive root of a cubic in `m²`) and costs nothing at all where there is none, which is
    /// every calibration that folds later than the sphere does - the X5's own among them.
    ///
    /// The answer is only an answer if it reaches `r`: past the end of the rising branch there is no `m`
    /// with `radial(m) = r` and this returns the end of the branch instead, which the caller checks for
    #[inline] fn solve_radial(r: f32, guess: f32, m_max: f32, kr: &[f32; 5]) -> f32 {
        let (mut lo, mut hi) = (0.0f32, m_max);
        let mut m = guess.max(0.0).min(m_max);
        // Newton is done in three or four; the twenty are for the last ring of the image circle, where the
        // curve is nearly flat and a step overshoots the fold, and the way back is a bisection
        for _ in 0..20 {
            let d = Self::radial_derivative(m, kr);
            if d <= 1e-9 {
                // The curve has folded at or before `m`; the root, if there is one, is below it
                hi = m;
                m = 0.5 * (lo + hi);
            } else {
                let f = Self::radial(m, kr) - r;
                // Solved, judged on the residual: `radial(m)` is only good to a few ulp of itself, so this
                // is the floor the solve can reach, and the final check below asks for 1e-4. Not on the
                // size of the Newton step: below an ulp of `m` that step is one a GPU computes exactly as
                // `-f/d` and then rounds away when it adds it to `m`, and a test on the difference of those
                // two never came out true on an RTX 5090 (the CPU only passes it because its `next - m`
                // happens to be exactly zero)
                if f.abs() <= 2e-7 * (1.0 + r) { break; }
                if f < 0.0 { lo = m; } else { hi = m; }
                let next = m - f / d;
                // A step that lands on an end of the bracket is a step of nothing, and the end it lands on
                // is `m` itself - kept, so the next iteration is this one again. Read as *outside* the
                // bracket it would be replaced by the bracket's midpoint, and the far end of the bracket
                // is still the sphere's own `m_max` whenever Newton has only ever approached the root from
                // one side: that threw a converged answer 0.18 away and left the remaining iterations
                // bisecting their way back, to about 1e-4 - a coin flip against the check below, and on
                // the GO 3 at 0% correction a fifth of the frame was background, pixel by pixel
                m = if next >= lo && next <= hi { next } else { 0.5 * (lo + hi) }; // outside (or NaN): bisect
            }
        }
        m
    }

    pub fn undistort_point(&self, point: (f32, f32), params: &KernelParams) -> Option<(f32, f32)> {
        let kr = &[params.k[0], params.k[1], params.k[2], params.k[6], params.k[7]];
        let xi = params.k[5];

        let r_d = (point.0 * point.0 + point.1 * point.1).sqrt();
        if r_d < 1e-9 { return Some(point); }

        // Past the end of the projection's rising branch there is no ray at all - but only where the
        // sphere's own end *is* that end. Where the polynomial folds first, `radial(m_max)` is a value off
        // the far side of the fold and says nothing about the largest radius the lens images; the solve's
        // own answer, checked below, is what decides there
        let m_max = Self::m_max(xi);
        if Self::radial_derivative(m_max, kr) > 0.0 && Self::radial(m_max, kr) < r_d { return None; }

        // Strip the tangential terms, invert the radial polynomial, re-evaluate them: three passes are
        // already exact to ~1e-9 of the ray radius, and unlike a fixed-point iteration on the whole
        // 2D map it stays stable where the projection compresses hardest (the old 200-step loop
        // silently gave up well inside the valid field, from ~74° on the X5).
        let (mut ax, mut ay) = point;
        let (mut u, mut v, mut m, mut r_s) = (0.0f32, 0.0f32, r_d, r_d);
        for _ in 0..3 {
            r_s = (ax * ax + ay * ay).sqrt();
            m = Self::solve_radial(r_s, m, m_max, kr);
            let s = if r_s > 1e-12 { m / r_s } else { 0.0 };
            u = ax * s;
            v = ay * s;
            let (tx, ty) = Self::tangential(u, v, &params.k);
            ax = point.0 - tx;
            ay = point.1 - ty;
        }
        // `solve_radial` returns the end of the rising branch when `r` is past everything the lens images,
        // so the radius it actually reached is the test of whether there was a ray here at all. This is the
        // whole of the rejection wherever the polynomial folds inside the sphere - a ξ ≤ 1 body has no
        // sphere fold to be bounded by, and asking a folded polynomial for its maximum gives nonsense
        if (Self::radial(m, kr) - r_s).abs() > 1e-4 * (1.0 + r_s) { return None; }

        // Unified sphere -> ray angle. The sphere point is `(u·α, v·α, α - ξ)`, already a unit vector, so
        // the angle falls out of `atan2` - where the old `x/z` went to infinity at 90° and cost an X4/X5
        // the outer 10% of its own image circle
        let rho2 = u*u + v*v;
        let disc = 1.0 + (1.0 - xi*xi) * rho2;
        if disc < 0.0 { return None; }
        let alpha = (xi + disc.sqrt()) / (rho2 + 1.0);
        let rho = rho2.sqrt();
        if rho < 1e-12 { return Some((0.0, 0.0)); }
        let theta = (rho * alpha).atan2(alpha - xi);
        let s = theta / rho;

        Some((u * s, v * s))
    }

    pub fn distort_point(&self, mut x: f32, mut y: f32, z: f32, params: &KernelParams) -> (f32, f32) {
        let kr = &[params.k[0], params.k[1], params.k[2], params.k[6], params.k[7]];
        let xi = params.k[5];

        let len = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();

        x = (x / len) / ((z / len) + xi);
        y = (y / len) / ((z / len) + xi);

        // The v3 terms in the order they always were, so a v3 profile renders bit for bit as before
        let k = &params.k;
        let r2 = x*x + y*y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let r8 = r4 * r4;
        let radial = 1.0 + kr[0]*r2 + kr[1]*r4 + kr[2]*r6 + kr[3]*r8 + kr[4]*r8*r2;
        let p1 = k[3] + k[8] * r2;
        let p2 = k[4] + k[9] * r2;
        (
            x * radial + 2.0*p1*x*y + p2*(r2 + 2.0*x*x) + r2*(k[10] + k[11]*r2),
            y * radial + 2.0*p2*x*y + p1*(r2 + 2.0*y*y) + r2*(k[12] + k[13]*r2)
        )
    }
    pub fn adjust_lens_profile(&self, _profile: &mut crate::LensProfile) { }

    /// The lens's own field end, in radians: the unified sphere folds at `acos(-1/ξ)` - 120° for the X5's
    /// ξ = 2, well past the 100° its image circle actually holds - and for ξ ≤ 1 it never folds but runs to
    /// infinity at `acos(-ξ)` instead. There is no ray past either
    pub fn field_limit(k: &[f64]) -> Option<f64> {
        let xi = *k.get(5)?;
        if xi > 1.0 { Some((-1.0 / xi).acos()) }
        else if xi > 1e-6 { Some((-xi).acos() - 1e-4) }
        else { Some(std::f64::consts::FRAC_PI_2 - 1e-4) } // ξ = 0 is a plain pinhole
    }

    /// `d(image radius)/dθ`, `<= 0` where the radial curve folds back. `dm/dθ` carries the unified
    /// sphere's own fold, `dr/dm` the polynomial's
    pub fn distortion_derivative(&self, theta: f64, k: &[f64]) -> Option<f64> {
        if k.len() < 6 { return None; }
        let (k1, k2, k3, xi) = (k[0], k[1], k[2], k[5]);
        let (k4, k5) = (k.get(6).copied().unwrap_or(0.0), k.get(7).copied().unwrap_or(0.0));
        let c = theta.cos();
        let d = c + xi;
        if d.abs() < 1e-9 { return None; }
        let m = theta.sin() / d;
        let m2 = m * m;
        let dm_dtheta = (1.0 + xi * c) / (d * d);
        let m4 = m2 * m2;
        let dr_dm = 1.0 + 3.0*k1*m2 + 5.0*k2*m4 + 7.0*k3*m4*m2 + 9.0*k4*m4*m4 + 11.0*k5*m4*m4*m2;
        Some(dr_dm * dm_dtheta)
    }

    pub fn id() -> &'static str { "insta360" }
    pub fn name() -> &'static str { "Insta360" }

    pub fn opencl_functions(&self) -> &'static str { include_str!("insta360.cl") }
    pub fn wgsl_functions(&self)   -> &'static str { include_str!("insta360.wgsl") }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The X5's lens 0 of the sample clip: Mei ξ = 2, RadtanPro d0..d12 (telemetry-parser's CameraRig)
    const X5: [f64; 14] = [0.24259843, 1.33183777, 0.03402105, -4.60448074, 0.0, -0.00078776, -0.00109209, 0.01341257, 0.00341814, -0.00505066, 0.00070317, -0.00831295, -0.00288465, 2.0];

    /// RadtanPro's 13 terms and ξ into the model's coefficients
    fn radtan_pro(d: &[f64; 14]) -> KernelParams {
        let mut p = KernelParams::default();
        let k = [d[0], d[1], d[2], d[6], d[5], d[13], d[3], d[4], d[8], d[7], d[9], d[11], d[10], d[12]];
        for (i, v) in k.iter().enumerate() { p.k[i] = *v as f32; }
        p
    }

    /// video-types' `MeiDistortion::RadtanPro`, in f64, as written there
    fn reference(d: &[f64; 14], ray: [f64; 3]) -> (f64, f64) {
        let n = (ray[0] * ray[0] + ray[1] * ray[1] + ray[2] * ray[2]).sqrt();
        let (x, y, z) = (ray[0] / n, ray[1] / n, ray[2] / n);
        let (mx, my) = (x / (d[13] + z), y / (d[13] + z));
        let r2 = mx * mx + my * my;
        let r = 1.0 + d[0] * r2 + d[1] * r2.powi(2) + d[2] * r2.powi(3) + d[3] * r2.powi(4) + d[4] * r2.powi(5);
        (mx * r + (r2 + 2.0 * mx * mx) * (d[5] + d[7] * r2) + 2.0 * mx * my * (d[6] + d[8] * r2) + d[9] * r2 + d[11] * r2 * r2,
         my * r + (r2 + 2.0 * my * my) * (d[6] + d[8] * r2) + 2.0 * mx * my * (d[5] + d[7] * r2) + d[10] * r2 + d[12] * r2 * r2)
    }

    fn rays() -> impl Iterator<Item = [f64; 3]> {
        (0..=50).flat_map(|i| (0..16).map(move |j| {
            let theta = (i as f64 * 2.0).to_radians();
            let phi = (j as f64 * 22.5 + 3.0).to_radians();
            [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
        }))
    }

    #[test]
    fn radtan_pro_as_video_types_has_it() {
        let p = radtan_pro(&X5);
        for ray in rays() {
            let (u, v) = Insta360::default().distort_point(ray[0] as f32, ray[1] as f32, ray[2] as f32, &p);
            let (eu, ev) = reference(&X5, ray);
            // f32 of the image radius, about 1.2 at 100°: a thousandth of a pixel at the X5's focal (~3000 px)
            assert!((u as f64 - eu).abs() < 2e-6 && (v as f64 - ev).abs() < 2e-6, "{ray:?}: {u} {v} against {eu} {ev}");
        }
    }

    #[test]
    fn radtan_pro_undistorts_back() {
        let p = radtan_pro(&X5);
        let m = Insta360::default();
        for ray in rays() {
            let theta = ray[0].hypot(ray[1]).atan2(ray[2]);
            let (u, v) = m.distort_point(ray[0] as f32, ray[1] as f32, ray[2] as f32, &p);
            let (a, b) = m.undistort_point((u, v), &p).expect("a ray inside the field");
            // The point path's rays are angle vectors: θ·(x, y)/ρ
            let rho = ray[0].hypot(ray[1]);
            let (ea, eb) = if rho > 1e-12 { (theta * ray[0] / rho, theta * ray[1] / rho) } else { (0.0, 0.0) };
            assert!((a as f64 - ea).abs() < 2e-5 && (b as f64 - eb).abs() < 2e-5, "θ {:.1}°: {a} {b} against {ea} {eb}", theta.to_degrees());
        }
    }

    #[test]
    fn a_v3_profile_is_untouched_by_the_new_terms() {
        // k1 k2 k3 p1 p2 ξ, the rest zero: the formula as it was before the RadtanPro terms
        let k = [0.1f32, -0.05, 0.01, 0.002, -0.001, 1.8];
        let mut p = KernelParams::default();
        p.k[..6].copy_from_slice(&k);
        for ray in rays() {
            let (x, y, z) = (ray[0] as f32, ray[1] as f32, ray[2] as f32);
            let len = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
            let (x0, y0) = ((x / len) / ((z / len) + k[5]), (y / len) / ((z / len) + k[5]));
            let r2 = x0*x0 + y0*y0;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let old = (x0 * (1.0 + k[0]*r2 + k[1]*r4 + k[2]*r6) + 2.0*k[3]*x0*y0 + k[4]*(r2 + 2.0*x0*x0),
                       y0 * (1.0 + k[0]*r2 + k[1]*r4 + k[2]*r6) + 2.0*k[4]*x0*y0 + k[3]*(r2 + 2.0*y0*y0));
            assert_eq!(Insta360::default().distort_point(x, y, z, &p), old);
        }
    }
}
