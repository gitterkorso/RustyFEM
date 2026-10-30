//! Tvärsnittsbiblioteket, del 2 (område beams2): HEX, TRAPEZOID, THICK PIPE
//! och ARBITRARY, vridstyvhet för RECT, välvningsstorheter för öppna
//! tunnväggiga tvärsnitt (B31OS/B32OS) och sektionspunkter för L och
//! ARBITRARY med olinjärt material.
//!
//! Testerna finns i `tests/beams2.rs`:
//!
//! - HEX (d = omskrivna radien, t): mittlinjemodell med omskrivna radien
//!   R = d − t/√3, Bredts formel för J, skjuvformfaktor 0,53.
//!   Slankhetskompensationen använder den massiva sexkantens tröghetsmoment
//!   (5√3/16)(d⁴ − dᵢ⁴), inte mittlinjemodellens.
//! - TRAPEZOID (a nedre bredd, b höjd, c övre bredd, d origo ovanför
//!   underkanten): massivt tvärsnitt, skjuvformfaktor 0,822, skjuvcentrum i
//!   tyngdpunkten. J räknas numeriskt ur Prandtls spänningsfunktion med
//!   ett fint nät av kvadratiska element (≈ exakt). För a = c används RECT-tabellen nedan.
//! - THICK PIPE (r, t): massiv ring, J = 2I, skjuvformfaktor enligt Cowper
//!   med ν = 0,3 (oberoende av materialet).
//! - ARBITRARY: tunnväggiga segment (mittlinjemodell, tjocklek 0 för
//!   hjälpsegment), öppna (J = Σ l t³/3, skjuvcentrum med sektoriella
//!   koordinater) eller slutna med en cell (Bredts formel). Skjuvformfaktor 1.
//!   Slutna tvärsnitt med olika väggtjocklek använder Bredts formel, som
//!   skiljer i andra ordningen i tjockleksskillnaden från en numerisk
//!   lösning (BOX likaså).
//! - RECT (*BEAM SECTION och *BEAM GENERAL SECTION): J är Saint-Venants
//!   exakta värde ur serielösningen för rektangeln (`rect_torsion`).

use crate::elements::beam_timoshenko::section::SectionProps;
use crate::fem::model::RawKeyword;
use crate::model::{fail, Result};
use std::f64::consts::PI;

// område gaps1: CHANNEL och HAT
#[path = "beam_sections_gaps1.rs"]
mod gaps1;

/// Tunnväggigt segment (mittlinje a → b, tjocklek t).
#[derive(Debug, Clone, Copy)]
pub struct Seg {
    pub a: [f64; 2],
    pub b: [f64; 2],
    pub t: f64,
}

impl Seg {
    fn len(&self) -> f64 {
        ((self.b[0] - self.a[0]).powi(2) + (self.b[1] - self.a[1]).powi(2)).sqrt()
    }
}

/// Raden för n1-riktningen i *BEAM SECTION/*BEAM GENERAL SECTION (0-baserad).
/// ARBITRARY har en rad per segment före n1-raden.
pub fn n1_row(shape: &str, first: &[f64]) -> usize {
    if shape == "ARBITRARY" {
        first.first().map(|&n| (n.max(1.0)) as usize).unwrap_or(1)
    } else {
        1
    }
}

/// Måtten för en form: första dataraden, för ARBITRARY alla segmentrader
/// efter varandra (antal, x1, x2, x1, x2, t, x1, x2, t, ...).
pub fn data_values(kw: &RawKeyword, shape: &str) -> Result<Vec<f64>> {
    let first = kw.first_floats()?;
    if shape != "ARBITRARY" {
        return Ok(first);
    }
    let n = n1_row(shape, &first);
    let mut v = Vec::new();
    for (i, d) in kw.data.iter().take(n).enumerate() {
        let f = kw.floats(d)?;
        let need = if i == 0 { 6 } else { 3 };
        if f.len() < need {
            return kw.err_at(d.line, format!("SECTION=ARBITRARY: rad {} ska ha {need} tal", i + 1));
        }
        v.extend_from_slice(&f[..need]);
    }
    if v.len() != 6 + 3 * (n - 1) {
        return kw.err(format!("SECTION=ARBITRARY: {n} segment angivna men färre datarader"));
    }
    Ok(v)
}

/// Segmenten i ett ARBITRARY-tvärsnitt.
pub fn arbitrary_segments(v: &[f64]) -> Result<Vec<Seg>> {
    if v.len() < 6 {
        return fail("SECTION=ARBITRARY kräver antal segment, två punkter och en tjocklek på första dataraden");
    }
    let n = v[0] as usize;
    if n < 2 {
        return fail("SECTION=ARBITRARY kräver minst två segment");
    }
    if v.len() < 6 + 3 * (n - 1) {
        return fail(format!("SECTION=ARBITRARY: {n} segment angivna men data för färre"));
    }
    let mut segs = vec![Seg { a: [v[1], v[2]], b: [v[3], v[4]], t: v[5] }];
    for k in 1..n {
        let o = 6 + 3 * (k - 1);
        let a = segs[k - 1].b;
        segs.push(Seg { a, b: [v[o], v[o + 1]], t: v[o + 2] });
    }
    Ok(segs)
}

/// Area och ytmoment kring origo (mittlinjemodell, exakt integrering).
pub fn seg_moments(segs: &[Seg]) -> (f64, f64, f64, f64, f64, f64) {
    let (mut a, mut s1, mut s2, mut i11, mut i22, mut i12) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for w in segs {
        let da = w.len() * w.t;
        let (x0, y0, x1, y1) = (w.a[0], w.a[1], w.b[0], w.b[1]);
        a += da;
        s2 += da * (x0 + x1) / 2.0;
        s1 += da * (y0 + y1) / 2.0;
        i22 += da * (x0 * x0 + x0 * x1 + x1 * x1) / 3.0;
        i11 += da * (y0 * y0 + y0 * y1 + y1 * y1) / 3.0;
        i12 += da * (2.0 * x0 * y0 + x0 * y1 + x1 * y0 + 2.0 * x1 * y1) / 6.0;
    }
    (a, s1, s2, i11, i22, i12)
}

fn close(p: [f64; 2], q: [f64; 2], scale: f64) -> bool {
    (p[0] - q[0]).abs() + (p[1] - q[1]).abs() <= 1e-9 * scale
}

fn scale_of(segs: &[Seg]) -> f64 {
    segs.iter().fold(1e-300f64, |m, s| m.max(s.a[0].abs()).max(s.a[1].abs()).max(s.b[0].abs()).max(s.b[1].abs()))
}

/// Är segmentkedjan sluten (sista punkten = första)?
pub fn is_closed(segs: &[Seg]) -> bool {
    close(segs[0].a, segs[segs.len() - 1].b, scale_of(segs))
}

/// Sektoriella storheter för en öppen kedja: skjuvcentrum (x1, x2) och
/// välvningskonstanten Γ = ∫ω² dA med ω kring skjuvcentrum och ∫ω dA = 0.
/// Hjälpsegment (t = 0) som går tillbaka längs en gren ger ω tillbaka till
/// förgreningspunktens värde, så kedjan kan beskriva grenade tvärsnitt.
pub fn open_sectorial(segs: &[Seg]) -> (f64, f64, f64) {
    let (a, s1, s2, i11, i22, i12) = seg_moments(segs);
    let (c1, c2) = (s2 / a, s1 / a);
    let ixx = i11 - a * c2 * c2; // ∫ȳ²
    let iyy = i22 - a * c1 * c1; // ∫x̄²
    let ixy = i12 - a * c1 * c2;
    // ω kring origo längs kedjan: dω = x dy − y dx
    let mut w0 = 0.0;
    let mut omega: Vec<(f64, f64)> = Vec::new();
    for s in segs {
        let dw = s.a[0] * s.b[1] - s.a[1] * s.b[0];
        omega.push((w0, w0 + dw));
        w0 += dw;
    }
    // ∫ f g dA för linjära f, g längs segmentet: L t (2 f0 g0 + f0 g1 + f1 g0 + 2 f1 g1)/6
    let prod = |f: &dyn Fn(&Seg, usize) -> (f64, f64), g: &dyn Fn(&Seg, usize) -> (f64, f64)| -> f64 {
        segs.iter()
            .enumerate()
            .map(|(k, s)| {
                let (f0, f1) = f(s, k);
                let (g0, g1) = g(s, k);
                s.len() * s.t * (2.0 * f0 * g0 + f0 * g1 + f1 * g0 + 2.0 * f1 * g1) / 6.0
            })
            .sum()
    };
    let om = |_: &Seg, k: usize| omega[k];
    let xb = |s: &Seg, _: usize| (s.a[0] - c1, s.b[0] - c1);
    let yb = |s: &Seg, _: usize| (s.a[1] - c2, s.b[1] - c2);
    let iwx = prod(&om, &yb);
    let iwy = prod(&om, &xb);
    // ω_P = ω_O − xP (y − y0) + yP (x − x0): villkoren ∫ω_P ȳ = ∫ω_P x̄ = 0
    // ger [ixx −ixy; ixy −iyy] [xP; yP] = [iwx; iwy]
    let det = -ixx * iyy + ixy * ixy;
    let xp = (iwx * (-iyy) - (-ixy) * iwy) / det;
    let yp = (ixx * iwy - ixy * iwx) / det;
    let (x0, y0) = (segs[0].a[0], segs[0].a[1]);
    let wp = |s: &Seg, k: usize| {
        let (w0, w1) = omega[k];
        (w0 - xp * (s.a[1] - y0) + yp * (s.a[0] - x0), w1 - xp * (s.b[1] - y0) + yp * (s.b[0] - x0))
    };
    let one = |_: &Seg, _: usize| (1.0, 1.0);
    let mean = prod(&wp, &one) / a;
    let wn = |s: &Seg, k: usize| {
        let (p0, p1) = wp(s, k);
        (p0 - mean, p1 - mean)
    };
    let gamma = prod(&wn, &wn);
    (xp, yp, gamma)
}

/// Normerade sektoriella koordinaten (som i `open_sectorial`) i origo om
/// origo ligger på ett segment med tjocklek > 0, annars 0 (Ω₀: ger
/// koppling mellan bikrökning och normalkraft/böjning).
pub fn origin_omega(segs: &[Seg]) -> f64 {
    let (xp, yp, _) = open_sectorial(segs);
    let (a, ..) = seg_moments(segs);
    let (x0, y0) = (segs[0].a[0], segs[0].a[1]);
    let mut w0 = 0.0;
    let mut om = Vec::new();
    for sg in segs {
        let dw = sg.a[0] * sg.b[1] - sg.a[1] * sg.b[0];
        let f = |p: [f64; 2], w: f64| w - xp * (p[1] - y0) + yp * (p[0] - x0);
        om.push((f(sg.a, w0), f(sg.b, w0 + dw)));
        w0 += dw;
    }
    let mean: f64 = segs.iter().zip(&om).map(|(sg, (p0, p1))| sg.len() * sg.t * (p0 + p1) / 2.0).sum::<f64>() / a;
    let tol = 1e-9 * scale_of(segs);
    for (sg, (p0, p1)) in segs.iter().zip(&om) {
        if sg.t <= 0.0 {
            continue;
        }
        let (dx, dy) = (sg.b[0] - sg.a[0], sg.b[1] - sg.a[1]);
        let l2 = dx * dx + dy * dy;
        let u = -(sg.a[0] * dx + sg.a[1] * dy) / l2;
        let d = ((sg.a[0] + u * dx).powi(2) + (sg.a[1] + u * dy).powi(2)).sqrt();
        if d <= tol && u >= -1e-12 && u <= 1.0 + 1e-12 {
            return p0 + u * (p1 - p0) - mean;
        }
    }
    0.0
}

/// Skjuvcentrum för ett slutet tunnväggigt tvärsnitt med en cell (segmenten
/// i ordning runt cellen): öppet skjuvflöde för enhetstvärkrafter, slutet med
/// ett konstant flöde så att ∮ q/t ds = 0, och momentjämvikt kring origo.
pub fn closed_shear_centre(segs: &[Seg]) -> (f64, f64) {
    let (a, s1, s2, i11, i22, i12) = seg_moments(segs);
    let (c1, c2) = (s2 / a, s1 / a);
    let ixx = i11 - a * c2 * c2;
    let iyy = i22 - a * c1 * c1;
    let ixy = i12 - a * c1 * c2;
    let d = ixx * iyy - ixy * ixy;
    let moment = |vx: f64, vy: f64| -> f64 {
        let (mut q, mut sum_qt, mut sum_1t, mut m_open, mut m_unit) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for s in segs.iter().filter(|s| s.t > 0.0) {
            let (p0, p1, t) = (s.a, s.b, s.t);
            let l = s.len();
            let (tx, ty) = ((p1[0] - p0[0]) / l, (p1[1] - p0[1]) / l);
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
    (moment(0.0, 1.0), -moment(1.0, 0.0))
}

/// Egenskaper för ett ARBITRARY-tvärsnitt.
fn arbitrary_props(v: &[f64]) -> Result<SectionProps> {
    let segs = arbitrary_segments(v)?;
    if segs.iter().any(|s| s.t < 0.0) {
        return fail("SECTION=ARBITRARY: negativ tjocklek");
    }
    let (a, s1, s2, i11, i22, i12) = seg_moments(&segs);
    if !(a > 0.0) {
        return fail("SECTION=ARBITRARY: tvärsnittet saknar area");
    }
    let mut p = SectionProps { a, s1, s2, i11, i22, i12, ..Default::default() };
    if is_closed(&segs) {
        // en cell: Bredts formel med mittlinjens area
        let am = 0.5 * segs.iter().map(|s| s.a[0] * s.b[1] - s.b[0] * s.a[1]).sum::<f64>().abs();
        if segs.iter().any(|s| s.t == 0.0) {
            return fail("SECTION=ARBITRARY: slutna tvärsnitt med hjälpsegment (tjocklek 0) eller grenar stöds inte");
        }
        let circ: f64 = segs.iter().map(|s| s.len() / s.t).sum();
        p.j = 4.0 * am * am / circ;
        let (x1s, x2s) = closed_shear_centre(&segs);
        p.x1s = x1s;
        p.x2s = x2s;
    } else {
        p.j = segs.iter().map(|s| s.len() * s.t.powi(3) / 3.0).sum();
        let (x1s, x2s, _) = open_sectorial(&segs);
        p.x1s = x1s;
        p.x2s = x2s;
    }
    p.k13 = 1.0;
    p.k23 = 1.0;
    Ok(p)
}

/// Egenskaper för de former som `beam_timoshenko_section.rs` inte har.
/// Skjuvformfaktorn ligger i `k13`/`k23` (multipliceras med G·A av anroparen).
pub fn shape_props(shape: &str, v: &[f64]) -> Option<Result<SectionProps>> {
    let need = |n: usize| -> Result<()> {
        if v.len() < n {
            return fail(format!("SECTION={shape} kräver {n} mått på första dataraden"));
        }
        Ok(())
    };
    let r = match shape {
        "HEX" => need(2).and_then(|_| hex_props(v[0], v[1])),
        "TRAPEZOID" => need(4).and_then(|_| trapezoid_props(v[0], v[1], v[2], v[3])),
        "THICK PIPE" => need(2).and_then(|_| thick_pipe_props(v[0], v[1])),
        "ARBITRARY" => arbitrary_props(v),
        "CHANNEL" => gaps1::channel_props(v),
        "HAT" => gaps1::hat_props(v),
        _ => return None,
    };
    Some(r)
}

fn hex_props(d: f64, t: f64) -> Result<SectionProps> {
    if !(d > 0.0 && t > 0.0 && t < d * 3f64.sqrt()) {
        return fail("SECTION=HEX: omskrivna radien och väggtjockleken ska vara positiva");
    }
    let r = d - t / 3f64.sqrt();
    // hörn i (r cos(30° + 60°k), r sin(30° + 60°k)); egenskaperna är isotropa
    let pts: Vec<[f64; 2]> = (0..=6).map(|k| {
        let f = (30.0 + 60.0 * k as f64).to_radians();
        [r * f.cos(), r * f.sin()]
    }).collect();
    let segs: Vec<Seg> = (0..6).map(|k| Seg { a: pts[k], b: pts[k + 1], t }).collect();
    let (a, s1, s2, i11, i22, i12) = seg_moments(&segs);
    let am = 1.5 * 3f64.sqrt() * r * r;
    let circ = 6.0 * r / t;
    let di = d - 2.0 * t / 3f64.sqrt();
    let is = 5.0 * 3f64.sqrt() / 16.0 * (d.powi(4) - di.powi(4));
    Ok(SectionProps { a, s1, s2, i11, i22, i12, j: 4.0 * am * am / circ, k13: 0.53, k23: 0.53, i11s: is, i22s: is, ..Default::default() })
}

fn trapezoid_props(a: f64, b: f64, c: f64, d: f64) -> Result<SectionProps> {
    if !(a >= 0.0 && c >= 0.0 && b > 0.0 && a + c > 0.0) {
        return fail("SECTION=TRAPEZOID: måtten a, b, c ska vara positiva");
    }
    let area = b * (a + c) / 2.0;
    let yc = b * (a + 2.0 * c) / (3.0 * (a + c)) - d;
    let i11c = b.powi(3) * (a * a + 4.0 * a * c + c * c) / (36.0 * (a + c));
    let i22 = b * (a + c) * (a * a + c * c) / 48.0;
    let j = if (a - c).abs() <= 1e-12 * a.max(c) { rect_torsion(a, b) } else { trapezoid_torsion(a, b, c) };
    Ok(SectionProps {
        a: area,
        s1: area * yc,
        s2: 0.0,
        i11: i11c + area * yc * yc,
        i22,
        i12: 0.0,
        j,
        x2s: yc,
        k13: 0.822,
        k23: 0.822,
        ..Default::default()
    })
}

fn thick_pipe_props(r: f64, t: f64) -> Result<SectionProps> {
    if !(r > 0.0 && t > 0.0 && t <= r) {
        return fail("SECTION=THICK PIPE: radien och väggtjockleken ska vara positiva (t ≤ r)");
    }
    let ri = r - t;
    let a = PI * (r * r - ri * ri);
    let i = PI / 4.0 * (r.powi(4) - ri.powi(4));
    let m2 = (ri / r).powi(2);
    let nu = 0.3;
    let k = 6.0 * (1.0 + nu) * (1.0 + m2).powi(2) / ((7.0 + 6.0 * nu) * (1.0 + m2).powi(2) + (20.0 + 12.0 * nu) * m2);
    Ok(SectionProps { a, i11: i, i22: i, j: 2.0 * i, k13: k, k23: k, ..Default::default() })
}

/// Saint-Venants vridstyvhet för en rektangel med sidorna a ≥ b:
/// J = a·b³·[1/3 − (64/π⁵)(b/a) Σ tanh(nπa/(2b))/n⁵] över udda n
/// (Timoshenko & Goodier, *Theory of Elasticity*, 1970, §109).
pub fn rect_torsion(a: f64, b: f64) -> f64 {
    let (l, s) = if a >= b { (a, b) } else { (b, a) };
    if !(s > 0.0) {
        return 0.0;
    }
    let r = s / l;
    let mut sum = 0.0;
    let mut n = 1.0_f64;
    loop {
        let term = (n * PI / (2.0 * r)).tanh() / n.powi(5);
        sum += term;
        if term < 1e-17 * sum {
            break;
        }
        n += 2.0;
    }
    (1.0 / 3.0 - 64.0 / PI.powi(5) * r * sum) * l * s.powi(3)
}

/// Niondsnodselement: formfunktioner och derivator (nodnummer j·3 + i, i längs ξ).
fn q9(xi: f64, eta: f64) -> ([f64; 9], [f64; 9], [f64; 9]) {
    let l = |t: f64| ([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], [t - 0.5, -2.0 * t, t + 0.5]);
    let (a, da) = l(xi);
    let (b, db) = l(eta);
    let mut n = [0.0; 9];
    let mut nx = [0.0; 9];
    let mut ny = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            n[j * 3 + i] = b[j] * a[i];
            nx[j * 3 + i] = b[j] * da[i];
            ny[j * 3 + i] = db[j] * a[i];
        }
    }
    (n, nx, ny)
}

/// Gausselimination med partiell pivotering (täta, små system).
fn solve_dense(mut k: Vec<f64>, mut f: Vec<f64>, n: usize) -> Vec<f64> {
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| k[i * n + c].abs().partial_cmp(&k[j * n + c].abs()).unwrap()).unwrap();
        if p != c {
            for j in 0..n {
                k.swap(c * n + j, p * n + j);
            }
            f.swap(c, p);
        }
        let d = k[c * n + c];
        for r in c + 1..n {
            let m = k[r * n + c] / d;
            if m != 0.0 {
                for j in c..n {
                    k[r * n + j] -= m * k[c * n + j];
                }
                f[r] -= m * f[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|j| k[r * n + j] * x[j]).sum();
        x[r] = (f[r] - s) / k[r * n + r];
    }
    x
}

/// Saint-Venants vridning för ett fyrsidigt område med raka sidor (hörnen
/// moturs) och ett n × n nät av niondsnodselement. `warping` = välvningsfunktion
/// (J = Ip − ωᵀKω), annars Prandtls spänningsfunktion (J = 2∫φ dA, φ = 0 på
/// randen). `ng` gausspunkter per riktning.
fn torsion_fe(corners: [[f64; 2]; 4], n: usize, ng: usize, warping: bool) -> f64 {
    // skalas till enhetsstorlek (J ∝ längd⁴) för en välkonditionerad lösning
    let sc = corners.iter().fold(0.0f64, |m, c| m.max(c[0].abs()).max(c[1].abs()));
    if !(sc > 0.0) {
        return 0.0;
    }
    let unit = corners.map(|c| [c[0] / sc, c[1] / sc]);
    torsion_fe_unit(unit, n, ng, warping) * sc.powi(4)
}

fn torsion_fe_unit(corners: [[f64; 2]; 4], n: usize, ng: usize, warping: bool) -> f64 {
    let g2 = [(-1.0 / 3f64.sqrt(), 1.0), (1.0 / 3f64.sqrt(), 1.0)];
    let s = (3.0f64 / 5.0).sqrt();
    let g3 = [(-s, 5.0 / 9.0), (0.0, 8.0 / 9.0), (s, 5.0 / 9.0)];
    let gp: &[(f64, f64)] = if ng == 2 { &g2 } else { &g3 };
    let m = 2 * n + 1;
    let nn = m * m;
    // nodkoordinater genom bilinjär avbildning av hörnen
    let map = |u: f64, v: f64| -> [f64; 2] {
        let w = [(1.0 - u) * (1.0 - v) / 4.0, (1.0 + u) * (1.0 - v) / 4.0, (1.0 + u) * (1.0 + v) / 4.0, (1.0 - u) * (1.0 + v) / 4.0];
        [(0..4).map(|k| w[k] * corners[k][0]).sum(), (0..4).map(|k| w[k] * corners[k][1]).sum()]
    };
    let xy: Vec<[f64; 2]> = (0..nn).map(|i| map(-1.0 + 2.0 * (i % m) as f64 / (m - 1) as f64, -1.0 + 2.0 * (i / m) as f64 / (m - 1) as f64)).collect();
    let mut k = vec![0.0; nn * nn];
    let mut f = vec![0.0; nn];
    let mut ip = 0.0;
    for ey in 0..n {
        for ex in 0..n {
            let ids: Vec<usize> = (0..9).map(|q| (2 * ey + q / 3) * m + 2 * ex + q % 3).collect();
            for &(gx, wx) in gp {
                for &(gy, wy) in gp {
                    let (nf, nxi, neta) = q9(gx, gy);
                    let mut jm = [[0.0; 2]; 2];
                    for q in 0..9 {
                        for d in 0..2 {
                            jm[0][d] += nxi[q] * xy[ids[q]][d];
                            jm[1][d] += neta[q] * xy[ids[q]][d];
                        }
                    }
                    let det = jm[0][0] * jm[1][1] - jm[0][1] * jm[1][0];
                    let w = wx * wy * det;
                    let mut dx = [0.0; 9];
                    let mut dy = [0.0; 9];
                    let (mut x, mut y) = (0.0, 0.0);
                    for q in 0..9 {
                        dx[q] = (jm[1][1] * nxi[q] - jm[0][1] * neta[q]) / det;
                        dy[q] = (-jm[1][0] * nxi[q] + jm[0][0] * neta[q]) / det;
                        x += nf[q] * xy[ids[q]][0];
                        y += nf[q] * xy[ids[q]][1];
                    }
                    for a in 0..9 {
                        for b in 0..9 {
                            k[ids[a] * nn + ids[b]] += w * (dx[a] * dx[b] + dy[a] * dy[b]);
                        }
                        f[ids[a]] += if warping { -w * (x * dy[a] - y * dx[a]) } else { 2.0 * w * nf[a] };
                    }
                    ip += w * (x * x + y * y);
                }
            }
        }
    }
    let keep: Vec<usize> = if warping {
        // ω = 0 i första noden (tar bort den konstanta moden)
        (1..nn).collect()
    } else {
        (0..nn).filter(|&i| i % m != 0 && i % m != m - 1 && i / m != 0 && i / m != m - 1).collect()
    };
    let nk = keep.len();
    let kr: Vec<f64> = keep.iter().flat_map(|&i| keep.iter().map(move |&j| (i, j))).map(|(i, j)| k[i * nn + j]).collect();
    let fr: Vec<f64> = keep.iter().map(|&i| f[i]).collect();
    let x = solve_dense(kr, fr.clone(), nk);
    let fx: f64 = fr.iter().zip(&x).map(|(a, b)| a * b).sum();
    if warping { ip - fx } else { fx }
}

/// Vridstyvhet för ett trapets (nedre bredd a, höjd b, övre bredd c) med
/// Prandtls spänningsfunktion på ett fint nät (8 × 8 niondsnodselement).
pub fn trapezoid_torsion(a: f64, b: f64, c: f64) -> f64 {
    use std::sync::Mutex;
    static CACHE: Mutex<Vec<([u64; 3], f64)>> = Mutex::new(Vec::new());
    let key = [a.to_bits(), b.to_bits(), c.to_bits()];
    if let Some(&(_, j)) = CACHE.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return j;
    }
    let j = torsion_fe([[-a / 2.0, 0.0], [a / 2.0, 0.0], [c / 2.0, b], [-c / 2.0, b]], 8, 3, false);
    let mut cache = CACHE.lock().unwrap();
    if cache.len() > 256 {
        cache.clear();
    }
    cache.push((key, j));
    j
}

/// Tröghetsmoment för slankhetskompensationen av I-profilens
/// skjuvstyvhet för γ2: I11 kring origo (mittlinjemodellen) minus S1'²/A, där
/// S1' räknas med livet mellan flänsarnas insidor (kontrolleras mot
/// k_I-matrisen i tests/cases/shells_beams/matriser).
pub fn i_slender_i11(v: &[f64], a: f64, i11: f64) -> f64 {
    let (l, h, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    let hw = h - t1 - t2;
    let s1 = b1 * t1 * (-l + t1 / 2.0) + b2 * t2 * (-l + h - t2 / 2.0) + t3 * hw * (-l + t1 + hw / 2.0);
    i11 - s1 * s1 / a
}

/// Välvningsstorheter för öppna tvärsnitt (B31OS/B32OS): (Γ, Ip) med Ip =
/// polärt tröghetsmoment kring tyngdpunkten (mittlinjemodellen; för I-profilen
/// med samma I11 som i slankhetskompensationen).
pub fn open_section(shape: &str, v: &[f64], p: &SectionProps) -> Result<(f64, f64)> {
    open_section_full(shape, v, p).map(|(g, ip, _)| (g, ip))
}

/// Som `open_section` men även Ω₀ (`origin_omega`).
pub fn open_section_full(shape: &str, v: &[f64], p: &SectionProps) -> Result<(f64, f64, f64)> {
    let mut om0 = 0.0;
    let ip = if p.i11s > 0.0 { p.i11s } else { p.i11c } + p.i22c;
    let gamma = match shape {
        "I" => {
            if v.len() < 7 {
                return fail("SECTION=I kräver 7 mått");
            }
            let (l, h, b1, b2, t1, t2) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let hm = (-l + h - t2 / 2.0) - (-l + t1 / 2.0);
            let (f1, f2) = (t1 * b1.powi(3) / 12.0, t2 * b2.powi(3) / 12.0);
            if f1 + f2 > 0.0 { hm * hm * f1 * f2 / (f1 + f2) } else { 0.0 }
        }
        // två ben som möts i skjuvcentrum: ingen välvning i mittlinjemodellen
        "L" => 0.0,
        "ARBITRARY" => {
            let segs = arbitrary_segments(v)?;
            if is_closed(&segs) {
                return fail("öppna balkar (B31OS, B32OS) kräver ett öppet ARBITRARY-tvärsnitt");
            }
            om0 = origin_omega(&segs);
            open_sectorial(&segs).2
        }
        _ => return fail(format!("SECTION={shape} kan inte användas med öppna balkar (B31OS, B32OS: I, L, ARBITRARY)")),
    };
    Ok((gamma, ip, om0))
}

/// Sektionspunkter (x1, x2, vikt) för former som `beam_timoshenko_plastic.rs`
/// inte har:
/// - L: 5 punkter i varje ben (9 totalt, Simpson), från spetsen av benet längs
///   1-axeln till hörnet och vidare till spetsen av benet längs 2-axeln.
/// - ARBITRARY: Simpson med 3 punkter per segment, punkterna i ordning
///   (A, mitt AB, B, mitt BC, C, ...); hjälpsegment ger vikt 0.
pub fn section_points(shape: &str, v: &[f64]) -> Option<Result<Vec<(f64, f64, f64)>>> {
    let simp = [1.0, 4.0, 2.0, 4.0, 1.0];
    match shape {
        "L" => {
            if v.len() < 4 {
                return Some(fail("SECTION=L kräver 4 mått på första dataraden"));
            }
            let (a, b, t1, t2) = (v[0], v[1], v[2], v[3]);
            let (l1, l2) = (a - t2 / 2.0, b - t1 / 2.0);
            let mut p = Vec::new();
            for k in 0..5 {
                p.push((l1 * (1.0 - k as f64 / 4.0), 0.0, l1 / 12.0 * simp[k] * t1));
            }
            p[4].2 += l2 / 12.0 * t2;
            for k in 1..5 {
                p.push((0.0, l2 * k as f64 / 4.0, l2 / 12.0 * simp[k] * t2));
            }
            Some(Ok(p))
        }
        "ARBITRARY" => Some(arbitrary_segments(v).map(|segs| {
            let mut p: Vec<(f64, f64, f64)> = vec![(segs[0].a[0], segs[0].a[1], 0.0)];
            for s in &segs {
                let w = s.len() * s.t / 6.0;
                p.last_mut().unwrap().2 += w;
                p.push(((s.a[0] + s.b[0]) / 2.0, (s.a[1] + s.b[1]) / 2.0, 4.0 * w));
                p.push((s.b[0], s.b[1], w));
            }
            p
        })),
        _ => None,
    }
}
