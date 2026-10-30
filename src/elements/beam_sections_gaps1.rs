//! Område gaps1: tvärsnitten CHANNEL och HAT (bara *BEAM GENERAL SECTION;
//! inkluderas från `beams2_sections.rs`).
//!
//! Båda räknas som öppna tunnväggiga tvärsnitt med mittlinjemodell, som
//! ARBITRARY: segmentkedjan nedan går genom väggarnas mittlinjer och
//! egenskaperna (area, ytmoment kring origo, J = Σ l t³/3, skjuvcentrum ur
//! sektoriella storheter) räknas av `arbitrary_props`.
//!
//! - CHANNEL (l, h, b₁, b₂, t₁, t₂, t₃, o): livet till vänster, flänsarna åt
//!   +1. l är avståndet från underflänsens underkant till origo längs 2, o
//!   avståndet från livets yttre kant till origo längs 1.
//! - HAT (l, h, b, b₁, b₂, t₁, t₂, t₃): symmetrisk kring 2-axeln, överfläns
//!   b₁ × t₁, bottenflänsar b₂ × t₂, sneda liv t₃, total bredd b; l från
//!   bottenflänsarnas underkant till origo.

use super::*;

/// Data för ARBITRARY (antal, x, y, x, y, t, x, y, t, ...) ur punkterna och tjocklekarna.
fn chain(points: &[[f64; 2]], t: &[f64]) -> Vec<f64> {
    let mut v = vec![t.len() as f64, points[0][0], points[0][1], points[1][0], points[1][1], t[0]];
    for k in 1..t.len() {
        v.extend_from_slice(&[points[k + 1][0], points[k + 1][1], t[k]]);
    }
    v
}

pub fn channel_props(v: &[f64]) -> Result<SectionProps> {
    if v.len() < 8 {
        return fail("SECTION=CHANNEL kräver l, h, b1, b2, t1, t2, t3, o på första dataraden");
    }
    let (l, h, b1, b2, t1, t2, t3, o) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    if !(h > 0.0 && b1 > 0.0 && b2 > 0.0 && t1 > 0.0 && t2 > 0.0 && t3 > 0.0) {
        return fail("SECTION=CHANNEL: måtten h, b1, b2, t1, t2, t3 ska vara positiva");
    }
    let xw = -o + t3 / 2.0;
    let yb = -l + t1 / 2.0;
    let yt = -l + h - t2 / 2.0;
    let pts = [[-o + b1, yb], [xw, yb], [xw, yt], [-o + b2, yt]];
    let mut p = arbitrary_props(&chain(&pts, &[t1, t3, t2]))?;
    // area och ytmoment exakt för rektanglarna
    // (flänsarna hela bredden, livet mellan flänsarna); J och skjuvcentrum
    // från mittlinjemodellen
    let rects = [(-o, -o + b1, -l, -l + t1), (-o, -o + b2, -l + h - t2, -l + h), (-o, -o + t3, -l + t1, -l + h - t2)];
    exact_rects(&mut p, &rects);
    Ok(p)
}

/// Area och ytmoment kring origo för axelparallella rektanglar (x0, x1, y0, y1).
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
        return fail("SECTION=HAT kräver l, h, b, b1, b2, t1, t2, t3 på första dataraden");
    }
    let (l, h, b, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
    if !(h > 0.0 && b > 0.0 && b1 > 0.0 && b2 > 0.0 && t1 > 0.0 && t2 > 0.0 && t3 > 0.0) {
        return fail("SECTION=HAT: måtten ska vara positiva");
    }
    let yb = -l + t2 / 2.0;
    let yt = -l + h - t1 / 2.0;
    let xo = b / 2.0;
    let xi = b / 2.0 - b2;
    let xt = b1 / 2.0;
    let pts = [[-xo, yb], [-xi, yb], [-xt, yt], [xt, yt], [xi, yb], [xo, yb]];
    arbitrary_props(&chain(&pts, &[t2, t3, t1, t3, t2]))
}
