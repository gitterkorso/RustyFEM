//! Area gaps1: the sections CHANNEL and HAT (only *BEAM GENERAL SECTION;
//! included from `beams2_sections.rs`).
//!
//! Both are treated as open thin-walled sections with a midline model, like
//! ARBITRARY: the segment chain below runs through the wall midlines and
//! the properties (area, area moments about the origin, J = Σ l t³/3, shear center from
//! sectorial quantities) are computed by `arbitrary_props`.
//!
//! - CHANNEL (l, h, b₁, b₂, t₁, t₂, t₃, o): web on the left, flanges towards
//!   +1. l is the distance from the bottom edge of the lower flange to the origin along 2, o
//!   the distance from the outer edge of the web to the origin along 1.
//! - HAT (l, h, b, b₁, b₂, t₁, t₂, t₃): symmetric about the 2-axis, top flange
//!   b₁ × t₁, bottom flanges b₂ × t₂, inclined webs t₃, total width b; l from
//!   the bottom edge of the bottom flanges to the origin.

use super::*;

/// Data for ARBITRARY (count, x, y, x, y, t, x, y, t, ...) from the points and thicknesses.
fn chain(points: &[[f64; 2]], t: &[f64]) -> Vec<f64> {
    let mut v = vec![t.len() as f64, points[0][0], points[0][1], points[1][0], points[1][1], t[0]];
    for k in 1..t.len() {
        v.extend_from_slice(&[points[k + 1][0], points[k + 1][1], t[k]]);
    }
    v
}

pub fn channel_props(v: &[f64]) -> Result<SectionProps> {
    if v.len() < 8 {
        return fail("SECTION=CHANNEL requires l, h, b1, b2, t1, t2, t3, o on the first data line");
    }
    let (l, h, b1, b2, t1, t2, t3, o) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    if !(h > 0.0 && b1 > 0.0 && b2 > 0.0 && t1 > 0.0 && t2 > 0.0 && t3 > 0.0) {
        return fail("SECTION=CHANNEL: the dimensions h, b1, b2, t1, t2, t3 must be positive");
    }
    let xw = -o + t3 / 2.0;
    let yb = -l + t1 / 2.0;
    let yt = -l + h - t2 / 2.0;
    let pts = [[-o + b1, yb], [xw, yb], [xw, yt], [-o + b2, yt]];
    let mut p = arbitrary_props(&chain(&pts, &[t1, t3, t2]))?;
    // area and area moments exact for the rectangles
    // (flanges full width, web between the flanges); J and shear center
    // from the midline model
    let rects = [(-o, -o + b1, -l, -l + t1), (-o, -o + b2, -l + h - t2, -l + h), (-o, -o + t3, -l + t1, -l + h - t2)];
    exact_rects(&mut p, &rects);
    Ok(p)
}

/// Area and area moments about the origin for axis-parallel rectangles (x0, x1, y0, y1).
fn exact_rects(p: &mut SectionProps, rects: &[(f64, f64, f64, f64)]) {
    let (mut a, mut s1, mut s2, mut i11, mut i22, mut i12) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x0, x1, y0, y1) in rects {
        let (w, hh) = (x1 - x0, y1 - y0);
        a += w * hh;
        s1 += w * (y1 * y1 - y0 * y0) / 2.0;
        s2 += hh * (x1 * x1 - x0 * x0) / 2.0;
        i11 += w * (y1.powi(3) - y0.powi(3)) / 3.0;
        i22 += hh * (x1.powi(3) - x0.powi(3)) / 3.0;
        i12 += (x1 * x1 - x0 * x0) / 2.0 * (y1 * y1 - y0 * y0) / 2.0;
    }
    p.a = a;
    p.s1 = s1;
    p.s2 = s2;
    p.i11 = i11;
    p.i22 = i22;
    p.i12 = i12;
}

pub fn hat_props(v: &[f64]) -> Result<SectionProps> {
    if v.len() < 8 {
        return fail("SECTION=HAT requires l, h, b, b1, b2, t1, t2, t3 on the first data line");
    }
    let (l, h, b, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    if !(h > 0.0 && b > 0.0 && b1 > 0.0 && b2 > 0.0 && t1 > 0.0 && t2 > 0.0 && t3 > 0.0) {
        return fail("SECTION=HAT: the dimensions must be positive");
    }
    let yb = -l + t2 / 2.0;
    let yt = -l + h - t1 / 2.0;
    let xo = b / 2.0;
    let xi = b / 2.0 - b2;
    let xt = b1 / 2.0;
    let pts = [[-xo, yb], [-xi, yb], [-xt, yt], [xt, yt], [xi, yb], [xo, yb]];
    arbitrary_props(&chain(&pts, &[t2, t3, t1, t3, t2]))
}
