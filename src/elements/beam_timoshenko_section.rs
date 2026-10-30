//! Tvärsnittsegenskaper för Timoshenkobalkarna (B21, B22, B31, B32).
//!
//! Egenskaperna räknas från *BEAM SECTION / *BEAM GENERAL SECTION (formen och
//! måtten i `SectionDef::raw`) och från materialet. Koordinater i tvärsnittet:
//! x1 längs n1, x2 längs n2, origo i balkens referensaxel (nodernas linje).
//!
//! Tunnväggiga former (PIPE, BOX, I, L) räknas med mittlinjemodell (väggarnas
//! egen böjstyvhet kring mittlinjen försummas). BOX, I och L
//! kan ha tyngdpunkt och skjuvcentrum utanför origo; det ger koppling mellan
//! normalkraft och böjning respektive mellan vridning och skjuvning.
//!
//! Skjuvformfaktorer k (skjuvstyvhet k·G·A): RECT 0,85, CIRC 0,89, PIPE 0,53, BOX 0,44, I 0,44, L 1,0, GENERAL 1,0.
//! *TRANSVERSE SHEAR STIFFNESS (K23, K13) ersätter k·G·A direkt.

use crate::elements::api::ElemCtx;
use crate::fem::model::{parse_f64, SectionDef, SectionKind};
use crate::model::{fail, Result};
use crate::section::BeamSection;

/// Parameter som *TRANSVERSE SHEAR STIFFNESS lägger på tvärsnittets `raw`.
pub const TSS_PARAM: &str = "_TRANSVERSE SHEAR STIFFNESS";

#[derive(Debug, Clone, Default)]
pub struct SectionProps {
    /// Area och ytmoment kring origo: S1 = ∫x2 dA, S2 = ∫x1 dA,
    /// I11 = ∫x2² dA, I22 = ∫x1² dA, I12 = ∫x1·x2 dA.
    pub a: f64,
    pub s1: f64,
    pub s2: f64,
    pub i11: f64,
    pub i22: f64,
    pub i12: f64,
    /// Tröghetsmoment kring tyngdpunkten (för slankhetskompensationen).
    pub i11c: f64,
    pub i22c: f64,
    /// Tyngdpunkt (x1, x2) och skjuvcentrum (x1, x2).
    pub c1: f64,
    pub c2: f64,
    pub x1s: f64,
    pub x2s: f64,
    pub j: f64,
    pub e: f64,
    pub g: f64,
    /// Skjuvstyvheter: k23 för skjuvning längs n2 (γ2), k13 längs n1 (γ1).
    pub k23: f64,
    pub k13: f64,
    pub rho: f64,
    /// Område beams2: tröghetsmoment som används i
    /// slankhetskompensationen (för γ2 respektive γ1) när det skiljer från
    /// I11c/I22c (I-profil, HEX); 0 = använd I11c/I22c.
    pub i11s: f64,
    pub i22s: f64,
}

/// Mittlinjesegment för tunnväggiga tvärsnitt: ändpunkter och tjocklek.
struct Wall {
    a: [f64; 2],
    b: [f64; 2],
    t: f64,
}

/// Ytmoment för en uppsättning väggar (mittlinjemodell, exakt integrering).
fn wall_moments(walls: &[Wall]) -> (f64, f64, f64, f64, f64, f64) {
    let (mut a, mut s1, mut s2, mut i11, mut i22, mut i12) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for w in walls {
        let l = ((w.b[0] - w.a[0]).powi(2) + (w.b[1] - w.a[1]).powi(2)).sqrt();
        let da = l * w.t;
        let (x0, y0, x1, y1) = (w.a[0], w.a[1], w.b[0], w.b[1]);
        a += da;
        s2 += da * (x0 + x1) / 2.0;
        s1 += da * (y0 + y1) / 2.0;
        // ∫ x² längs linjärt segment = L·(x0² + x0·x1 + x1²)/3
        i22 += da * (x0 * x0 + x0 * x1 + x1 * x1) / 3.0;
        i11 += da * (y0 * y0 + y0 * y1 + y1 * y1) / 3.0;
        i12 += da * (2.0 * x0 * y0 + x0 * y1 + x1 * y0 + 2.0 * x1 * y1) / 6.0;
    }
    (a, s1, s2, i11, i22, i12)
}

/// Skjuvcentrum (x1, x2) för ett slutet tunnväggigt tvärsnitt med en cell.
///
/// Väggarna ska bilda en sluten slinga i ordning (b för vägg k = a för vägg
/// k+1, eller omvänt – riktningen kontrolleras per vägg). Skjuvflödet för en
/// enhetstvärkraft räknas öppet längs slingan (exakt för raka väggar, q är
/// andragradspolynom i s), sluts med ett konstant flöde så att ∮ q/t ds = 0
/// (ingen vridning) och skjuvcentrum fås ur momentjämvikt kring origo.
fn closed_shear_centre(walls: &[Wall]) -> (f64, f64) {
    let (a, s1, s2, i11, i22, i12) = wall_moments(walls);
    let (c1, c2) = (s2 / a, s1 / a);
    // tröghetsmoment kring tyngdpunkten: ixx = ∫y², iyy = ∫x², ixy = ∫xy
    let ixx = i11 - a * c2 * c2;
    let iyy = i22 - a * c1 * c1;
    let ixy = i12 - a * c1 * c2;
    let d = ixx * iyy - ixy * ixy;
    // ordna väggarna till en slinga med ändpunkter p0 → p1
    let mut segs: Vec<([f64; 2], [f64; 2], f64)> = Vec::new();
    let mut cur = walls[0].a;
    for _ in 0..walls.len() {
        let close = |p: [f64; 2], q: [f64; 2]| (p[0] - q[0]).abs() + (p[1] - q[1]).abs() < 1e-9 * (1.0 + p[0].abs() + p[1].abs());
        let w = walls.iter().find(|w| (close(w.a, cur) || close(w.b, cur)) && !segs.iter().any(|s| close(s.0, w.a) && close(s.1, w.b) || close(s.0, w.b) && close(s.1, w.a))).unwrap();
        let (p0, p1) = if close(w.a, cur) { (w.a, w.b) } else { (w.b, w.a) };
        segs.push((p0, p1, w.t));
        cur = p1;
    }
    // moment kring origo av skjuvflödet för last (vx, vy)
    let moment = |vx: f64, vy: f64| -> f64 {
        let (mut q, mut sum_qt, mut sum_1t, mut m_open, mut m_unit) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for &(p0, p1, t) in &segs {
            let l = ((p1[0] - p0[0]).powi(2) + (p1[1] - p0[1]).powi(2)).sqrt();
            let (tx, ty) = ((p1[0] - p0[0]) / l, (p1[1] - p0[1]) / l);
            // f(s) = vy·(iyy·y − ixy·x) + vx·(ixx·x − ixy·y), dq/ds = −t·f/d
            let f = |x: f64, y: f64| (vy * (iyy * (y - c2) - ixy * (x - c1)) + vx * (ixx * (x - c1) - ixy * (y - c2))) / d;
            let f0 = f(p0[0], p0[1]);
            let f1 = (f(p1[0], p1[1]) - f0) / l;
            let int_q = q * l - t * (f0 * l * l / 2.0 + f1 * l * l * l / 6.0);
            let h = p0[0] * ty - p0[1] * tx;
            sum_qt += int_q / t;
            sum_1t += l / t;
            m_open += h * int_q;
            m_unit += h * l;
            q -= t * (f0 * l + f1 * l * l / 2.0);
        }
        let q0 = -sum_qt / sum_1t;
        m_open + q0 * m_unit
    };
    // Vy i x1 = xs ger momentet xs; Vx i x2 = ys ger −ys
    (moment(0.0, 1.0), -moment(1.0, 0.0))
}

/// Egenskaper för en standardform (utan material). Används även av
/// `inp::keywords::section` för former som kärnan inte känner till.
pub fn shape_profile(shape: &str, v: &[f64]) -> Option<BeamSection> {
    let p = shape_props(shape, v).ok()?;
    Some(BeamSection { a: p.a, i11: p.i11c, i22: p.i22c, j: p.j })
}

fn need(shape: &str, v: &[f64], n: usize) -> Result<()> {
    if v.len() < n {
        return fail(format!("SECTION={shape} kräver {n} mått på första dataraden"));
    }
    Ok(())
}

/// Geometriska egenskaper och skjuvformfaktor för en tvärsnittsform.
pub fn shape_props(shape: &str, v: &[f64]) -> Result<SectionProps> {
    let mut p = SectionProps::default();
    let kfac;
    match shape {
        "RECT" => {
            need(shape, v, 2)?;
            let s = BeamSection::rect(v[0], v[1]);
            p.a = s.a;
            p.i11 = s.i11;
            p.i22 = s.i22;
            // område beams2: J ur en elementlösning (tabell), inte Roarks formel
            p.j = crate::elements::beams2::sections::rect_torsion(v[0], v[1]);
            kfac = 0.85;
        }
        "CIRC" => {
            need(shape, v, 1)?;
            let s = BeamSection::circle(v[0]);
            p.a = s.a;
            p.i11 = s.i11;
            p.i22 = s.i22;
            p.j = s.j;
            kfac = 0.89;
        }
        "PIPE" => {
            need(shape, v, 2)?;
            let s = BeamSection::pipe_thin(v[0], v[1]);
            p.a = s.a;
            p.i11 = s.i11;
            p.i22 = s.i22;
            p.j = s.j;
            kfac = 0.53;
        }
        "BOX" => {
            // a (längs 1), b (längs 2), t1 (höger vägg, +x1), t2 (överkant), t3 (vänster), t4 (underkant)
            need(shape, v, 6)?;
            let (a, b, t1, t2, t3, t4) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let xr = a / 2.0 - t1 / 2.0;
            let xl = -a / 2.0 + t3 / 2.0;
            let yt = b / 2.0 - t2 / 2.0;
            let yb = -b / 2.0 + t4 / 2.0;
            let walls = [
                Wall { a: [xr, yb], b: [xr, yt], t: t1 },
                Wall { a: [xl, yt], b: [xr, yt], t: t2 },
                Wall { a: [xl, yb], b: [xl, yt], t: t3 },
                Wall { a: [xl, yb], b: [xr, yb], t: t4 },
            ];
            let (aa, s1, s2, i11, i22, i12) = wall_moments(&walls);
            p.a = aa;
            p.s1 = s1;
            p.s2 = s2;
            p.i11 = i11;
            p.i22 = i22;
            p.i12 = i12;
            // sluten tunnväggig: J = 4·Am² / ∮ds/t
            let am = (xr - xl) * (yt - yb);
            let circ = (yt - yb) / t1 + (xr - xl) / t2 + (yt - yb) / t3 + (xr - xl) / t4;
            p.j = 4.0 * am * am / circ;
            let (x1s, x2s) = closed_shear_centre(&walls);
            p.x1s = x1s;
            p.x2s = x2s;
            kfac = 0.44;
        }
        "I" => {
            // l (origo till underkant), h (höjd), b1 (underfläns), b2 (överfläns), t1, t2 (flänsar), t3 (liv)
            need(shape, v, 7)?;
            let (l, h, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            let yb = -l + t1 / 2.0;
            let yt = -l + h - t2 / 2.0;
            let walls = [
                Wall { a: [-b1 / 2.0, yb], b: [b1 / 2.0, yb], t: t1 },
                Wall { a: [-b2 / 2.0, yt], b: [b2 / 2.0, yt], t: t2 },
                Wall { a: [0.0, yb], b: [0.0, yt], t: t3 },
            ];
            let (aa, s1, s2, i11, i22, i12) = wall_moments(&walls);
            // livet går mellan flänsarnas mittlinjer; den del som ligger inne i
            // flänsarna (t3·t1/2 resp. t3·t2/2) dras av i flänsens mittlinje,
            // så att arean blir b1·t1 + b2·t2 + (h − t1 − t2)·t3
            let (db, dt) = (t3 * t1 / 2.0, t3 * t2 / 2.0);
            p.a = aa - db - dt;
            p.s1 = s1 - db * yb - dt * yt;
            p.s2 = s2;
            p.i11 = i11 - db * yb * yb - dt * yt * yt;
            p.i22 = i22;
            p.i12 = i12;
            p.j = (b1 * t1.powi(3) + b2 * t2.powi(3) + (yt - yb) * t3.powi(3)) / 3.0;
            // skjuvcentrum på livets linje, där flänsarnas böjmoment balanserar
            let if1 = t1 * b1.powi(3);
            let if2 = t2 * b2.powi(3);
            p.x2s = (if1 * yb + if2 * yt) / (if1 + if2);
            // område beams2: slankhetskompensation för γ2
            p.i11s = crate::elements::beams2::sections::i_slender_i11(v, p.a, p.i11);
            kfac = 0.44;
        }
        "L" => {
            // a (ben längs 1), b (ben längs 2), t1 (benet längs 1), t2 (benet längs 2); origo i hörnet
            need(shape, v, 4)?;
            let (a, b, t1, t2) = (v[0], v[1], v[2], v[3]);
            // origo i skärningen mellan benens mittlinjer: benen
            // går från origo till a − t2/2 längs 1 och b − t1/2 längs 2
            let walls = [
                Wall { a: [0.0, 0.0], b: [a - t2 / 2.0, 0.0], t: t1 },
                Wall { a: [0.0, 0.0], b: [0.0, b - t1 / 2.0], t: t2 },
            ];
            let (aa, s1, s2, i11, i22, i12) = wall_moments(&walls);
            p.a = aa;
            p.s1 = s1;
            p.s2 = s2;
            p.i11 = i11;
            p.i22 = i22;
            p.i12 = i12;
            p.j = ((a - t2 / 2.0) * t1.powi(3) + (b - t1 / 2.0) * t2.powi(3)) / 3.0;
            // skjuvcentrum i benens skärningspunkt = origo
            kfac = 1.0;
        }
        // område beams2: HEX, TRAPEZOID, THICK PIPE, ARBITRARY
        _ => match crate::elements::beams2::sections::shape_props(shape, v) {
            Some(r) => {
                p = r?;
                kfac = p.k13;
            }
            None => return fail(format!("SECTION={shape} stöds inte ännu för Timoshenkobalkar (RECT, CIRC, PIPE, BOX, I, L, HEX, TRAPEZOID, THICK PIPE, ARBITRARY, GENERAL)")),
        },
    }
    p.c1 = if p.a > 0.0 { p.s2 / p.a } else { 0.0 };
    p.c2 = if p.a > 0.0 { p.s1 / p.a } else { 0.0 };
    p.i11c = p.i11 - p.a * p.c2 * p.c2;
    p.i22c = p.i22 - p.a * p.c1 * p.c1;
    // skjuvformfaktor lagras tillfälligt i k13/k23 (multipliceras med G·A nedan)
    p.k13 = kfac;
    p.k23 = kfac;
    Ok(p)
}

/// Fullständiga tvärsnittsegenskaper för ett element.
pub fn section_props(sec: &SectionDef, ctx: &ElemCtx, elem: &str) -> Result<SectionProps> {
    let SectionKind::Beam(b) = &sec.kind else {
        return fail(format!("element {} ({elem}) kräver *BEAM SECTION eller *BEAM GENERAL SECTION", ctx.label));
    };
    let general = sec.raw.name == "BEAM GENERAL SECTION";
    // område gaps1: NONLINEAR GENERAL har samma data (A, I11, I12, I22, J)
    let mut p = if b.shape == "GENERAL" || b.shape == "NONLINEAR GENERAL" {
        let mut p = SectionProps { a: b.profile.a, i11: b.profile.i11, i22: b.profile.i22, i12: b.i12, j: b.profile.j, ..Default::default() };
        p.i11c = p.i11;
        p.i22c = p.i22;
        p.k13 = 1.0;
        p.k23 = 1.0;
        p
    } else {
        let v = crate::elements::beams2::sections::data_values(&sec.raw, &b.shape)?;
        let mut p = shape_props(&b.shape, &v).map_err(|e| crate::model::FemError(format!("{}: {}", sec.raw.location(), e.0)))?;
        if general && b.shape == "RECT" {
            p.j = crate::elements::beams2::sections::rect_torsion(v[0], v[1]);
        }
        p
    };
    let (e, g) = match b.moduli {
        Some(m) => m,
        None => {
            let mat = ctx.material(elem)?;
            // område struct_nonlin: olinjärt material med *BEAM SECTION integreras i
            // sektionspunkterna (beam_timoshenko_plastic.rs); de elastiska
            // egenskaperna används för vridning, skjuvning och linjära steg
            if !mat.is_linear() && sec.raw.name != "BEAM SECTION" {
                return crate::elements::api::not_implemented(&format!("olinjärt material ({})", mat.name()), elem);
            }
            let Some((e, nu)) = mat.elastic_moduli() else {
                return fail(format!("{elem} kräver isotropt linjärelastiskt material"));
            };
            (e, e / (2.0 * (1.0 + nu)))
        }
    };
    p.e = e;
    p.g = g;
    p.rho = if general { b.density } else if b.density > 0.0 { b.density } else { ctx.density };
    // skjuvstyvhet
    let tss = sec.raw.param(TSS_PARAM).map(|s| s.split(',').map(|x| parse_f64(x).unwrap_or(0.0)).collect::<Vec<f64>>());
    match tss {
        Some(v) if !v.is_empty() => {
            p.k23 = v[0];
            p.k13 = v.get(1).copied().unwrap_or(v[0]);
        }
        _ => {
            p.k23 *= g * p.a;
            p.k13 *= g * p.a;
        }
    }
    Ok(p)
}
