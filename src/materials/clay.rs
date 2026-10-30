//! Modifierad Cam-clay (*CLAY PLASTICITY, *CLAY HARDENING) och porös
//! elasticitet (*POROUS ELASTIC), en kritisk-tillståndsmodell (Roscoe &
//! Burland 1968).
//!
//! Porös elasticitet (logaritmisk, små töjningar, ε_v = tr ε):
//!
//!   p = −p_t^el + (p0 + p_t^el) exp[(1 + e0)/κ (1 − exp ε_v^el)],
//!
//! där p0 är begynnelsetrycket (*INITIAL CONDITIONS, TYPE=STRESS) och e0
//! portalet (*INITIAL CONDITIONS, TYPE=RATIO). Deviatoriskt: SHEAR=G ger
//! S = S0 + 2G e^el; SHEAR=POISSON (standard) ger ΔS = 2Ĝ Δe^el med
//! Ĝ = 3(1 − 2ν)/(2(1 + ν)) · K̄, där K̄ = −Δp/Δε_v^el är inkrementets
//! sekantkompressionsmodul (oberoende av inkrementens storlek för
//! proportionella vägar).
//! Med *ELASTIC (bara HARDENING=TABULAR) är K och G konstanta.
//!
//! Flytyta (associativ flytning):
//!
//!   F = ((p − p_t)/a − 1)²/β² + (t/(M a))² − 1 = 0,
//!   t = q/2 [1 + 1/K − (1 − 1/K)(r/q)³],
//!
//! med β = 1 på den torra sidan (p − p_t < a). Hårdnande:
//! - HARDENING=EXPONENTIAL: a = a0 exp[(1 + e0)(1 − J^pl)/(λ − κ J^pl)],
//!   J^pl = exp(tr ε^pl); a0 direkt eller ur INTERCEPT=e1:
//!   a0 = exp[(e1 − e0 − κ ln p0)/(λ − κ)]/(1 + β).
//! - HARDENING=TABULAR: p_c(x) och p_t(x) (*CLAY HARDENING, TYPE=COMPRESSION/
//!   TENSION) med x = ε_vol^pl|0 − tr ε^pl, a = (p_c − p_t)/(1 + β).
//!
//! Ligger begynnelsespänningen utanför flytytan flyttas ytan så att punkten
//! hamnar på den (a0 respektive ε_vol^pl|0 justeras).
//!
//! Implicit Euler: med K = 1 behåller den deviatoriska spänningen
//! riktningen hos S_n + 2Ĝ Δe, och de två okända Δε_v^pl och Δε_q^pl löses
//! med Newton ur F = 0 och normalitetsvillkoret Δε_v^pl ∂F/∂q + Δε_q^pl ∂F/∂p
//! = 0. Med K ≠ 1 löses hela systemet (S, Δε_v^pl, Δλ) med Newton. Den
//! konsistenta tangenten är numerisk (centraldifferens av hela
//! returalgoritmen), osymmetrisk i allmänhet.
//!
//! Utdata: PEEQ = a (flytytans centrum) och PE.

use super::plasticity::{M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Cam-clay-plasticitet / porös elasticitet",
    consumes: &["CLAY HARDENING", "CLAY PLASTICITY", "ELASTIC", "POROUS ELASTIC"],
    priority: 10,
    build: Some(build),
}];

/// Tillståndslayout per integrationspunkt.
const XEL: usize = 0; // elastisk volymtöjning sedan start
const EPV: usize = 1; // tr ε^pl
const A: usize = 2; // flytytans storlek a
const EP: usize = 3; // ε^pl (6, ingenjörsskjuvning)
const P0: usize = 9; // begynnelsetryck
const E0: usize = 10; // begynnelseportal (NaN = inte givet)
const A0: usize = 11; // a0 (exponentiellt) eller ε_vol^pl|0 (tabell)
const NST: usize = 12;

#[derive(Debug, Clone)]
enum Elastic {
    /// κ, G (SHEAR=G) eller ν (SHEAR=POISSON), p_t^el.
    Porous { kappa: f64, g: Option<f64>, nu: f64, pt: f64 },
    Linear { k: f64, g: f64, e: f64, nu: f64 },
}

#[derive(Debug, Clone)]
enum Hard {
    Exponential { lam: f64, a0: f64, e1: Option<f64> },
    /// (x, p_c), (x, p_t) med x = ε_vol^pl|0 − tr ε^pl; x0 = ε_vol^pl|0.
    Tabular { pc: Vec<(f64, f64)>, pt: Vec<(f64, f64)>, x0: f64 },
}

#[derive(Debug, Clone)]
struct Yield {
    m: f64,
    beta: f64,
    kk: f64,
    hard: Hard,
}

#[derive(Debug, Clone)]
pub struct Clay {
    el: Elastic,
    yld: Option<Yield>,
}

fn num(kw: &RawKeyword, key: &str) -> Result<Option<f64>> {
    match kw.param(key) {
        Some(s) => crate::fem::model::parse_f64(s.trim())
            .map(Some)
            .ok_or_else(|| crate::model::FemError(format!("{}: {key}={s} är inget tal", kw.location()))),
        None => Ok(None),
    }
}

fn single_row(kw: &RawKeyword) -> Result<Vec<f64>> {
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperatur- och fältberoende parametrar stöds inte ännu");
    }
    kw.first_floats()
}

fn hardening_table(kw: &RawKeyword, tension: bool) -> Result<Vec<(f64, f64)>> {
    if kw.has("DEPENDENCIES") {
        return kw.err("fältberoende hårdnande stöds inte ännu");
    }
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "temperaturberoende hårdnande stöds inte ännu");
        }
        let (p, x) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !tension && !(p > 0.0) {
            return kw.err_at(d.line, "p_c ska vara positivt");
        }
        if tension && p > 0.0 {
            return kw.err_at(d.line, "p_t ska vara noll eller negativt");
        }
        match t.last() {
            Some(&(xl, _)) if !(x > xl) => return kw.err_at(d.line, "den volymetriska plastiska töjningen ska öka"),
            None if x != 0.0 => return kw.err_at(d.line, "första punkten ska ha volymetrisk plastisk töjning 0"),
            _ => {}
        }
        t.push((x, p));
    }
    if t.is_empty() {
        return kw.err("hårdnandetabell saknas");
    }
    Ok(t)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let porous = def.behavior("POROUS ELASTIC");
    let clay = def.behavior("CLAY PLASTICITY");
    if porous.is_none() && clay.is_none() {
        return Ok(None);
    }
    if def.behavior("CLAY HARDENING").is_some() && clay.is_none() {
        let kw = def.behavior("CLAY HARDENING").unwrap();
        return kw.err("*CLAY HARDENING kräver *CLAY PLASTICITY, HARDENING=TABULAR");
    }
    let el = match (porous, def.behavior("ELASTIC")) {
        (Some(_), Some(e)) => return e.err("*POROUS ELASTIC och *ELASTIC kan inte användas i samma material"),
        (Some(pk), None) => {
            let ty = pk.param_upper("TYPE").unwrap_or_else(|| "LOGARITHMIC".into());
            if ty != "LOGARITHMIC" {
                return pk.err(format!("TYPE={ty} stöds inte ännu (bara LOGARITHMIC)"));
            }
            let shear = pk.param_upper("SHEAR").unwrap_or_else(|| "POISSON".into());
            let v = single_row(pk)?;
            let kappa = v.first().copied().unwrap_or(0.0);
            let s = v.get(1).copied().unwrap_or(0.0);
            let pt = v.get(2).copied().unwrap_or(0.0);
            if v.get(3).is_some_and(|&t| t != 0.0) {
                return pk.err("temperaturberoende parametrar stöds inte ännu");
            }
            if !(kappa > 0.0) {
                return pk.err("κ ska vara positivt");
            }
            if pt < 0.0 {
                return pk.err("p_t^el får inte vara negativt");
            }
            match shear.as_str() {
                "G" => {
                    if !(s > 0.0) {
                        return pk.err("G ska vara positivt");
                    }
                    Elastic::Porous { kappa, g: Some(s), nu: 0.0, pt }
                }
                "POISSON" => {
                    if !(s > -1.0 && s < 0.5) {
                        return pk.err("ν ska ligga mellan −1 och 0,5");
                    }
                    Elastic::Porous { kappa, g: None, nu: s, pt }
                }
                o => return pk.err(format!("SHEAR={o} stöds inte (G, POISSON)")),
            }
        }
        (None, Some(_)) => {
            let ck = clay.unwrap();
            let (e, nu) = super::mohr_coulomb::iso_elastic(def, ck, "Cam-clay med *ELASTIC")?;
            Elastic::Linear { k: e / (3.0 * (1.0 - 2.0 * nu)), g: e / (2.0 * (1.0 + nu)), e, nu }
        }
        (None, None) => {
            let ck = clay.unwrap();
            return ck.err("*CLAY PLASTICITY kräver *POROUS ELASTIC eller *ELASTIC");
        }
    };
    let yld = match clay {
        None => None,
        Some(ck) => {
            let v = single_row(ck)?;
            let lam = v.first().copied().unwrap_or(0.0);
            let m = v.get(1).copied().unwrap_or(0.0);
            let a0 = v.get(2).copied().unwrap_or(0.0);
            let beta = match v.get(3).copied().unwrap_or(0.0) {
                b if b == 0.0 => 1.0,
                b => b,
            };
            let kk = match v.get(4).copied().unwrap_or(0.0) {
                k if k == 0.0 => 1.0,
                k => k,
            };
            if v.get(5).is_some_and(|&t| t != 0.0) {
                return ck.err("temperaturberoende parametrar stöds inte ännu");
            }
            if !(m > 0.0) {
                return ck.err("M ska vara positivt");
            }
            if !(beta > 0.0) {
                return ck.err("β ska vara positivt");
            }
            if !(0.778..=1.0).contains(&kk) {
                return ck.err("K ska ligga mellan 0,778 och 1,0");
            }
            let hardening = ck.param_upper("HARDENING").unwrap_or_else(|| "EXPONENTIAL".into());
            let hard = match hardening.as_str() {
                "EXPONENTIAL" => {
                    let Elastic::Porous { kappa, .. } = el else {
                        return ck.err("HARDENING=EXPONENTIAL kräver *POROUS ELASTIC");
                    };
                    if !(lam > kappa) {
                        return ck.err("λ ska vara större än κ");
                    }
                    let e1 = num(ck, "INTERCEPT")?;
                    if e1.is_none() && !(a0 > 0.0) {
                        return ck.err("a0 ska vara positivt (eller ange INTERCEPT=e1)");
                    }
                    if def.behavior("CLAY HARDENING").is_some() {
                        return ck.err("*CLAY HARDENING kräver HARDENING=TABULAR");
                    }
                    Hard::Exponential { lam, a0, e1 }
                }
                "TABULAR" => {
                    if ck.has("INTERCEPT") {
                        return ck.err("INTERCEPT kan inte användas med HARDENING=TABULAR");
                    }
                    let mut pc = None;
                    let mut pt = vec![(0.0, 0.0)];
                    for hk in def.behaviors.iter().filter(|b| b.name == "CLAY HARDENING") {
                        if hk.has("SR") {
                            return hk.err("SR= (*SOFTENING REGULARIZATION) stöds inte ännu");
                        }
                        match hk.param_upper("TYPE").unwrap_or_else(|| "COMPRESSION".into()).as_str() {
                            "COMPRESSION" => pc = Some(hardening_table(hk, false)?),
                            "TENSION" => pt = hardening_table(hk, true)?,
                            o => return hk.err(format!("TYPE={o} stöds inte (COMPRESSION, TENSION)")),
                        }
                    }
                    let Some(pc) = pc else {
                        return ck.err("HARDENING=TABULAR kräver *CLAY HARDENING (TYPE=COMPRESSION)");
                    };
                    Hard::Tabular { pc, pt, x0: a0 }
                }
                o => return ck.err(format!("HARDENING={o} stöds inte (EXPONENTIAL, TABULAR)")),
            };
            Some(Yield { m, beta, kk, hard })
        }
    };
    Ok(Some(Arc::new(Clay { el, yld })))
}

fn interp(t: &[(f64, f64)], x: f64) -> f64 {
    super::principal::interp(t, x).0
}

/// Inverterar en stigande (eller fallande) styckvis linjär tabell: x där y(x) = y.
fn inverse(t: &[(f64, f64)], y: f64) -> Option<f64> {
    if t.len() == 1 {
        return None;
    }
    for w in t.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if (y - y0) * (y - y1) <= 0.0 && y1 != y0 {
            return Some(x0 + (x1 - x0) * (y - y0) / (y1 - y0));
        }
    }
    None
}

/// Deviatoriska invarianter för en spänningslik vektor (tensorskjuvning):
/// q, t (t̃ med K) och ∂t/∂S (deviatorisk tensor, Voigt med tensorskjuvning).
fn t_measure(s: &V6, kk: f64) -> (f64, f64, V6) {
    let m = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
    let mut ss = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            ss += m[i][j] * m[i][j];
        }
    }
    let q = (1.5 * ss).sqrt();
    if q <= 0.0 {
        return (0.0, 0.0, [0.0; 6]);
    }
    let dq: V6 = std::array::from_fn(|i| 1.5 * s[i] / q);
    if kk == 1.0 {
        return (q, q, dq);
    }
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let r3 = 13.5 * det;
    let t = q / 2.0 * (1.0 + 1.0 / kk) - (1.0 - 1.0 / kk) / 2.0 * r3 / (q * q);
    let mut s2 = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                s2[i][j] += m[i][k] * m[k][j];
            }
        }
    }
    let tr2 = s2[0][0] + s2[1][1] + s2[2][2];
    let s2v = [s2[0][0] - tr2 / 3.0, s2[1][1] - tr2 / 3.0, s2[2][2] - tr2 / 3.0, s2[0][1], s2[0][2], s2[1][2]];
    let n: V6 = std::array::from_fn(|i| {
        (1.0 + 1.0 / kk) / 2.0 * dq[i] - (1.0 - 1.0 / kk) / 2.0 * (13.5 * s2v[i] / (q * q) - 2.0 * r3 / (q * q * q) * dq[i])
    });
    (q, t, n)
}

/// Spänningar och storheter vid inkrementets slut för givna plastiska delar.
struct EndState {
    p: f64,
    g: f64,
}

impl Clay {
    fn kappa(&self) -> f64 {
        match self.el {
            Elastic::Porous { kappa, .. } => kappa,
            Elastic::Linear { .. } => 0.0,
        }
    }

    /// p som funktion av total elastisk volymtöjning x.
    fn p_of(&self, st: &[f64], x: f64) -> f64 {
        match self.el {
            Elastic::Porous { kappa, pt, .. } => {
                let b = (1.0 + st[E0]) / kappa;
                -pt + (st[P0] + pt) * (b * (-x.exp_m1())).exp()
            }
            Elastic::Linear { k, .. } => st[P0] - k * x,
        }
    }

    /// Skjuvmodul för inkrementet: Δx = elastisk volymtöjningsändring.
    fn g_of(&self, st: &[f64], pn: f64, dx: f64) -> f64 {
        match self.el {
            Elastic::Porous { g: Some(g), .. } => g,
            Elastic::Porous { kappa, g: None, nu, pt } => {
                let b = (1.0 + st[E0]) / kappa;
                let xn = st[XEL];
                // sekant (p(xn+dx) − p(xn))/dx utan kancellation
                let kbar = if dx == 0.0 {
                    (pn + pt) * b * xn.exp()
                } else {
                    let y = -b * xn.exp() * dx.exp_m1();
                    -(pn + pt) * y.exp_m1() / dx
                };
                3.0 * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu)) * kbar
            }
            Elastic::Linear { g, .. } => g,
        }
    }

    /// Flytytans storlek a och p_t vid tr ε^pl = epv.
    fn size(&self, y: &Yield, st: &[f64], epv: f64) -> (f64, f64) {
        match &y.hard {
            Hard::Exponential { lam, .. } => {
                let j = epv.exp();
                let k = self.kappa();
                (st[A0] * ((1.0 + st[E0]) * (-epv.exp_m1()) / (lam - k * j)).exp(), 0.0)
            }
            Hard::Tabular { pc, pt, .. } => {
                let x = st[A0] - epv;
                let (c, t) = (interp(pc, x), interp(pt, x));
                ((c - t) / (1.0 + y.beta), t)
            }
        }
    }

    fn yield_f(y: &Yield, p: f64, t: f64, a: f64, pt: f64) -> f64 {
        let b = if p - pt < a { 1.0 } else { y.beta };
        ((p - pt) / a - 1.0).powi(2) / (b * b) + (t / (y.m * a)).powi(2) - 1.0
    }

    /// ∂F/∂p och ∂F/∂t.
    fn yield_grad(y: &Yield, p: f64, t: f64, a: f64, pt: f64) -> (f64, f64) {
        let b = if p - pt < a { 1.0 } else { y.beta };
        (2.0 / (b * b * a) * ((p - pt) / a - 1.0), 2.0 * t / (y.m * y.m * a * a))
    }

    /// Spänningsuppdatering i 3D. `de`: töjningsinkrement (ingenjörsskjuvning),
    /// `sn`: spänning vid inkrementets början. Returnerar ny spänning; `stn`
    /// får nytt tillstånd.
    fn update3d(&self, de: &V6, sn: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let needs_e0 = matches!(self.el, Elastic::Porous { .. }) || self.yld.as_ref().is_some_and(|y| matches!(y.hard, Hard::Exponential { .. }));
        if needs_e0 && st[E0].is_nan() {
            return fail("*POROUS ELASTIC och HARDENING=EXPONENTIAL kräver portalet e0 (*INITIAL CONDITIONS, TYPE=RATIO)");
        }
        stn[..NST].copy_from_slice(&st[..NST]);
        let pn = -(sn[0] + sn[1] + sn[2]) / 3.0;
        let s_n: V6 = [sn[0] + pn, sn[1] + pn, sn[2] + pn, sn[3], sn[4], sn[5]];
        let dv = de[0] + de[1] + de[2];
        let dd: V6 = [de[0] - dv / 3.0, de[1] - dv / 3.0, de[2] - dv / 3.0, de[3] / 2.0, de[4] / 2.0, de[5] / 2.0];
        // elastiskt prov
        let end = |v: f64| -> EndState {
            let p = self.p_of(st, st[XEL] + dv - v);
            let g = self.g_of(st, pn, dv - v);
            EndState { p, g }
        };
        let e0s = end(0.0);
        let str_: V6 = std::array::from_fn(|i| s_n[i] + 2.0 * e0s.g * dd[i]);
        let sig = |p: f64, s: &V6| -> V6 { [s[0] - p, s[1] - p, s[2] - p, s[3], s[4], s[5]] };
        let Some(y) = &self.yld else {
            stn[XEL] = st[XEL] + dv;
            return Ok(sig(e0s.p, &str_));
        };
        let (an, ptn) = self.size(y, st, st[EPV]);
        let (_, t_tr, _) = t_measure(&str_, y.kk);
        if Self::yield_f(y, e0s.p, t_tr, an, ptn) <= 1e-12 {
            stn[XEL] = st[XEL] + dv;
            return Ok(sig(e0s.p, &str_));
        }
        let (v, s_end, dep_dev) = if y.kk == 1.0 { self.return_k1(y, st, &s_n, &dd, dv, &end)? } else { self.return_general(y, st, &s_n, &dd, dv, &end, &str_)? };
        let es = end(v);
        let (a, _) = self.size(y, st, st[EPV] + v);
        stn[XEL] = st[XEL] + dv - v;
        stn[EPV] = st[EPV] + v;
        stn[A] = a;
        for i in 0..6 {
            let f = if i < 3 { 1.0 } else { 2.0 };
            stn[EP + i] = st[EP + i] + if i < 3 { v / 3.0 } else { 0.0 } + f * dep_dev[i];
        }
        Ok(sig(es.p, &s_end))
    }

    /// Retur med K = 1: okända v = Δε_v^pl och w = Δε_q^pl.
    #[allow(clippy::type_complexity)]
    fn return_k1(&self, y: &Yield, st: &[f64], s_n: &V6, dd: &V6, _dv: f64, end: &dyn Fn(f64) -> EndState) -> Result<(f64, V6, V6)> {
        let eval = |v: f64, w: f64| -> (f64, f64, V6, f64, f64) {
            let es = end(v);
            let ss: V6 = std::array::from_fn(|i| s_n[i] + 2.0 * es.g * dd[i]);
            let (qs, _, _) = t_measure(&ss, 1.0);
            let q = qs - 3.0 * es.g * w;
            let s: V6 = if qs > 0.0 { std::array::from_fn(|i| ss[i] * q / qs) } else { ss };
            let (a, pt) = self.size(y, st, st[EPV] + v);
            (es.p, q, s, a, pt)
        };
        let res = |v: f64, w: f64| -> [f64; 2] {
            let (p, q, _, a, pt) = eval(v, w);
            let (fp, fq) = Self::yield_grad(y, p, q, a, pt);
            [Self::yield_f(y, p, q, a, pt), (v * fq + w * fp) * a]
        };
        let (mut v, mut w) = (0.0f64, 0.0f64);
        let mut r = res(v, w);
        // normerad residual: F är dimensionslös, normalitetsvillkoret i töjning
        let norm = |r: &[f64; 2], v: f64, w: f64| r[0].abs().max(r[1].abs() / v.abs().max(w.abs()).max(1e-12));
        for _ in 0..100 {
            let n0 = norm(&r, v, w);
            if n0 <= 1e-13 {
                break;
            }
            let h = 1e-8 * (v.abs() + w.abs()).max(1e-6);
            let (r1, r2) = (res(v + h, w), res(v, w + h));
            let j = [[(r1[0] - r[0]) / h, (r2[0] - r[0]) / h], [(r1[1] - r[1]) / h, (r2[1] - r[1]) / h]];
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let dvv = (r[0] * j[1][1] - r[1] * j[0][1]) / det;
            let dww = (j[0][0] * r[1] - j[1][0] * r[0]) / det;
            // dämpat steg om residualen inte minskar
            let mut t = 1.0;
            let (vn, wn, rn) = loop {
                let (vn, wn) = (v - t * dvv, (w - t * dww).max(0.0));
                let rn = res(vn, wn);
                if (rn[0].is_finite() && rn[1].is_finite() && norm(&rn, vn, wn) < n0) || t < 1e-4 {
                    break (vn, wn, rn);
                }
                t *= 0.5;
            };
            let small = (vn - v).abs() <= 1e-15 * v.abs().max(1e-12) && (wn - w).abs() <= 1e-15 * w.abs().max(1e-12);
            v = vn;
            w = wn;
            r = rn;
            if small {
                break;
            }
        }
        if !(norm(&r, v, w) <= 1e-9) {
            return fail("Cam-clay: returalgoritmen konvergerade inte");
        }
        let (_, q, s, _, _) = eval(v, w);
        let dep: V6 = if q > 0.0 { std::array::from_fn(|i| 1.5 * w * s[i] / q) } else { [0.0; 6] };
        Ok((v, s, dep))
    }

    /// Retur med K ≠ 1: okända S (6), v = Δε_v^pl och Δλ.
    #[allow(clippy::too_many_arguments)]
    fn return_general(&self, y: &Yield, st: &[f64], s_n: &V6, dd: &V6, _dv: f64, end: &dyn Fn(f64) -> EndState, str_: &V6) -> Result<(f64, V6, V6)> {
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(end(0.0).p.abs()).max(1e-30);
        let res = |x: &[f64; 8]| -> [f64; 8] {
            let s: V6 = std::array::from_fn(|i| x[i]);
            let (v, dl) = (x[6], x[7]);
            let es = end(v);
            let (a, pt) = self.size(y, st, st[EPV] + v);
            let (_, t, n) = t_measure(&s, y.kk);
            let (fp, ft) = Self::yield_grad(y, es.p, t, a, pt);
            let mut r = [0.0; 8];
            for i in 0..6 {
                r[i] = (s[i] + 2.0 * es.g * dl * ft * n[i] - s_n[i] - 2.0 * es.g * dd[i]) / scale;
            }
            r[6] = Self::yield_f(y, es.p, t, a, pt);
            r[7] = (v + dl * fp) * 1e2;
            r
        };
        let mut x = [str_[0], str_[1], str_[2], str_[3], str_[4], str_[5], 0.0, 0.0];
        let norm = |r: &[f64; 8]| r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut r = res(&x);
        let mut ok = false;
        for _ in 0..100 {
            if norm(&r) <= 1e-14 {
                ok = true;
                break;
            }
            let mut jac = vec![0.0; 64];
            for c in 0..8 {
                let h = if c < 6 { 1e-7 * scale } else { 1e-7 * x[c].abs().max(1e-6) };
                let mut x2 = x;
                x2[c] += h;
                let r2 = res(&x2);
                for i in 0..8 {
                    jac[i * 8 + c] = (r2[i] - r[i]) / h;
                }
            }
            let Some(d) = super::anisotropic::solve_dense(&jac, &r) else { break };
            let n0 = norm(&r);
            let mut t = 1.0;
            loop {
                let xn: [f64; 8] = std::array::from_fn(|i| x[i] - t * d[i]);
                let rn = res(&xn);
                if (rn.iter().all(|v| v.is_finite()) && norm(&rn) < n0) || t < 1e-6 {
                    x = xn;
                    r = rn;
                    break;
                }
                t *= 0.5;
            }
        }
        if !ok && norm(&r) > 1e-10 {
            return fail("Cam-clay: returalgoritmen konvergerade inte");
        }
        let s: V6 = std::array::from_fn(|i| x[i]);
        let (v, dl) = (x[6], x[7]);
        let es = end(v);
        let (a, pt) = self.size(y, st, st[EPV] + v);
        let (_, t, n) = t_measure(&s, y.kk);
        let (_, ft) = Self::yield_grad(y, es.p, t, a, pt);
        let dep: V6 = std::array::from_fn(|i| dl * ft * n[i]);
        Ok((v, s, dep))
    }

    /// Uppdatering i elementets spänningstillstånd (komponenterna i `mode`).
    fn update_mode(&self, mode: StressMode, de: &[f64], sn: &[f64], st: &[f64], stn: &mut [f64]) -> Result<Vec<f64>> {
        let n = mode.ncomp();
        let mut d6 = [0.0; 6];
        let mut s6 = [0.0; 6];
        d6[..n].copy_from_slice(de);
        s6[..n].copy_from_slice(sn);
        let s = self.update3d(&d6, &s6, st, stn)?;
        Ok(s[..n].to_vec())
    }
}

impl MaterialModel for Clay {
    fn name(&self) -> &str {
        if self.yld.is_some() {
            "CLAY PLASTICITY"
        } else {
            "POROUS ELASTIC"
        }
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn init_state(&self, state: &mut [f64]) {
        state[..NST].fill(0.0);
        state[E0] = f64::NAN;
        if let Some(y) = &self.yld {
            match &y.hard {
                Hard::Exponential { a0, .. } => {
                    state[A0] = *a0;
                    state[A] = *a0;
                }
                Hard::Tabular { pc, pt, x0 } => {
                    state[A0] = *x0;
                    state[A] = (interp(pc, *x0) - interp(pt, *x0)) / (1.0 + y.beta);
                }
            }
        }
    }

    fn init_state_at(&self, mode: StressMode, stress: &[f64], void_ratio: Option<f64>, state: &mut [f64]) -> Result<()> {
        if matches!(mode, StressMode::PlaneStress | StressMode::Uniaxial) {
            return fail(format!("{} stöds inte i plan spänning eller enaxligt spänningstillstånd", self.name()));
        }
        let p0 = -(stress.iter().take(3).sum::<f64>()) / 3.0;
        state[P0] = p0;
        if let Some(e) = void_ratio {
            state[E0] = e;
        }
        if let Elastic::Porous { pt, .. } = self.el {
            if !(p0 + pt > 0.0) {
                return fail("porös elasticitet kräver p0 + p_t^el > 0 (begynnelsetryck, *INITIAL CONDITIONS, TYPE=STRESS)");
            }
        }
        let Some(y) = &self.yld else { return Ok(()) };
        let mut s6 = [0.0; 6];
        s6[..stress.len().min(6)].copy_from_slice(&stress[..stress.len().min(6)]);
        let s: V6 = [s6[0] + p0, s6[1] + p0, s6[2] + p0, s6[3], s6[4], s6[5]];
        let (_, t, _) = t_measure(&s, y.kk);
        if let Hard::Exponential { lam, e1: Some(e1), .. } = &y.hard {
            if state[E0].is_nan() {
                return fail("INTERCEPT= kräver portalet e0 (*INITIAL CONDITIONS, TYPE=RATIO)");
            }
            if !(p0 > 0.0) {
                return fail("INTERCEPT= kräver ett positivt begynnelsetryck p0");
            }
            let k = self.kappa();
            state[A0] = ((e1 - state[E0] - k * p0.ln()) / (lam - k)).exp() / (1.0 + y.beta);
        }
        let (a, pt) = self.size(y, state, 0.0);
        state[A] = a;
        if Self::yield_f(y, p0, t, a, pt) > 1e-12 {
            // begynnelsepunkten utanför flytytan: flytta ytan så att punkten ligger på den
            let pp = p0 - pt;
            let tm = t / y.m;
            // torra sidan (β = 1): a = (p² + (t/M)²)/(2p), gäller om p < a
            let dry = if pp > 0.0 { (pp * pp + tm * tm) / (2.0 * pp) } else { f64::INFINITY };
            let anew = if pp < dry {
                dry
            } else {
                // våta sidan: (p u − 1)²/β² + (t u/M)² = 1 med u = 1/a
                let b2 = y.beta * y.beta;
                let qa = pp * pp / b2 + tm * tm;
                let qb = -2.0 * pp / b2;
                let qc = 1.0 / b2 - 1.0;
                // största roten u (minsta a) ger p − p_t ≥ a, dvs. punkten på våta sidan
                let disc = (qb * qb - 4.0 * qa * qc).max(0.0).sqrt();
                2.0 * qa / (-qb + disc)
            };
            match &y.hard {
                Hard::Exponential { .. } => state[A0] = anew,
                Hard::Tabular { pc, pt: ptab, .. } => {
                    // p_c(x) − p_t(x) = (1 + β) a: sök x i p_c-tabellen (p_t antas konstant här)
                    let target = (1.0 + y.beta) * anew + interp(ptab, state[A0]);
                    match inverse(pc, target) {
                        Some(x) => state[A0] = x,
                        None => return fail("Cam-clay: begynnelsespänningen ligger utanför flytytan och hårdnandetabellen räcker inte för att flytta ytan dit"),
                    }
                }
            }
            state[A] = self.size(y, state, 0.0).0;
        }
        Ok(())
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mode = input.mode;
        if matches!(mode, StressMode::PlaneStress | StressMode::Uniaxial) {
            return fail(format!("{} stöds inte i plan spänning eller enaxligt spänningstillstånd", self.name()));
        }
        let n = mode.ncomp();
        let stress = self.update_mode(mode, input.dstrain, input.stress_old, state_old, state_new)?;
        // numerisk konsistent tangent: centraldifferens av hela uppdateringen
        let scale = input.dstrain.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-4);
        let h = 1e-7 * scale;
        let mut tmp = vec![0.0; NST];
        let mut c = vec![0.0; n * n];
        for j in 0..n {
            let mut ep = input.dstrain.to_vec();
            let mut em = input.dstrain.to_vec();
            ep[j] += h;
            em[j] -= h;
            let sp = self.update_mode(mode, &ep, input.stress_old, state_old, &mut tmp)?;
            let sm = self.update_mode(mode, &em, input.stress_old, state_old, &mut tmp)?;
            for i in 0..n {
                c[i * n + j] = (sp[i] - sm[i]) / (2.0 * h);
            }
        }
        Ok(MatOutput { stress, tangent: c })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        // elastisk tangent vid p = 1 (bara för linjära hjälpberäkningar)
        let (k, g) = match self.el {
            Elastic::Porous { kappa, g, nu, pt } => {
                let k = (1.0 + pt) / kappa;
                (k, g.unwrap_or(3.0 * (1.0 - 2.0 * nu) / (2.0 * (1.0 + nu)) * k))
            }
            Elastic::Linear { k, g, .. } => (k, g),
        };
        let e = 9.0 * k * g / (3.0 * k + g);
        let nu = (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g));
        let d: M6 = super::plasticity::iso_d6(e, nu);
        Ok(super::anisotropic::reduce_6x6(&d, mode))
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        match self.el {
            Elastic::Linear { e, nu, .. } => Some((e, nu)),
            _ => None,
        }
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        super::plasticity::rotate_v6(r, &mut state[EP..EP + 6], true);
    }

    fn output_names(&self) -> &[&'static str] {
        if self.yld.is_some() {
            &["PEEQ"]
        } else {
            &[]
        }
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        if self.yld.is_some() {
            vec![state[A]]
        } else {
            Vec::new()
        }
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        if self.yld.is_none() {
            return Vec::new();
        }
        vec![("PEEQ", vec!["PEEQ"]), ("PE", super::plasticity::pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        if self.yld.is_none() {
            return Vec::new();
        }
        let mut v = vec![state[A]];
        v.extend(super::plasticity::pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        v
    }
}
