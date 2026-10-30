//! Utökade Drucker–Prager-modeller (*DRUCKER PRAGER) utöver den linjära med
//! K = 1 i `plasticity_dp.rs`: SHEAR CRITERION=HYPERBOLIC, SHEAR
//! CRITERION=EXPONENT FORM och SHEAR CRITERION=LINEAR med K ≠ 1, med
//! hårdnande från *DRUCKER PRAGER HARDENING (TYPE=COMPRESSION, TENSION, SHEAR).
//!
//! Utökade Drucker–Prager-modeller:
//!
//! - Hyperbolisk: F = √(l0² + q²) − p tan β − d′ = 0 med l0 = d′|0 − pt|0 tan β
//!   (pt|0 = hydrostatisk draghållfasthet i indata) och
//!   d′ = √(l0² + σc²) − σc tan β/3 (COMPRESSION), √(l0² + σt²) + σt tan β/3
//!   (TENSION), √(l0² + d²) (SHEAR). l0 är fast; med d′|0 insatt blir
//!   √(l0² + σ0²) = l0 + A, dvs. l0 = (σ0² − A²)/(2A) med A = (pt|0 + σ0/3) tan β,
//!   (pt|0 − σ0/3) tan β respektive pt|0 tan β.
//! - Exponentform: F = a qᵇ − p − pt = 0 med pt = a σcᵇ − σc/3, a σtᵇ + σt/3
//!   respektive a dᵇ.
//! - Båda har den hyperboliska flödespotentialen
//!   G = √((ε σ̄|0 tan ψ)² + q²) − p tan ψ, ε från ECCENTRICITY= (standard 0,1;
//!   för den hyperboliska modellen med ψ = β ε = l0/(σ̄|0 tan β), associativt
//!   flöde), och ekvivalent plastisk töjning σ̄ ε̄̇ᵖ = σ:ε̇ᵖ.
//! - Linjär med K ≠ 1: F = t − p tan β − d, G = t − p tan ψ; d = (1 − tan β/3) σc,
//!   (1/K + tan β/3) σt respektive d (TYPE=SHEAR: kohesionen direkt); ε̄̇ᵖ = λ̇ (1 − tan ψ/3),
//!   λ̇ (1/K + tan ψ/3) respektive λ̇ (1 + 1/K)/2 (axiell plastisk töjning i
//!   enaxligt tryck/drag, γ̇ᵖ/√3 i skjuvning; oberoende av K).
//!
//! Integrationen är bakåt Euler med den gemensamma returalgoritmen i
//! `invariant_return.rs`; tangenten är numerisk (centraldifferens av hela
//! uppdateringen) och osymmetrisk vid icke-associativt flöde (helt med
//! *STEP, UNSYMM=YES, annars den symmetriska delen). Utdata PEEQ och PE.

use super::invariant_return::{return_map, Inv, Surface};
use super::plasticity::{ddot, iso_d6, mv6, pe_labels, rotate_v6, update_reduced, Hardening, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "utökad Drucker–Prager (hyperbolisk, exponentform, linjär med K ≠ 1)",
    consumes: &["ELASTIC", "DRUCKER PRAGER", "DRUCKER PRAGER HARDENING"],
    priority: 11,
    build: Some(build),
}];

/// Om materialet ska byggas här i stället för i `plasticity_dp.rs`.
pub fn claims(def: &MaterialDef) -> bool {
    let Some(kw) = def.behavior("DRUCKER PRAGER") else { return false };
    match kw.param_upper("SHEAR CRITERION").as_deref() {
        Some("HYPERBOLIC") | Some("EXPONENT FORM") => true,
        None | Some("LINEAR") => {
            let k = kw.data.first().and_then(|d| kw.floats(d).ok()).and_then(|v| v.get(1).copied()).unwrap_or(0.0);
            k != 0.0 && (k - 1.0).abs() > 1e-12
        }
        _ => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum HType {
    Compression,
    Tension,
    Shear,
}

#[derive(Debug, Clone, Copy)]
enum Criterion {
    /// l0, ε σ̄0 tan ψ
    Hyperbolic { l0: f64 },
    /// a, b
    Exponent { a: f64, b: f64 },
    /// K
    Linear { k: f64 },
}

#[derive(Debug, Clone)]
pub struct DruckerPrager2 {
    e: f64,
    nu: f64,
    g: f64,
    kb: f64,
    ce: M6,
    crit: Criterion,
    tan_beta: f64,
    tan_psi: f64,
    /// ε σ̄|0 tan ψ i den hyperboliska flödespotentialen.
    ecc: f64,
    htype: HType,
    hard: Hardening,
}

const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const NST: usize = 13;

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "hastighets- eller temperaturberoende hårdnande stöds inte ännu");
        }
        let (s, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(s > 0.0) {
            return kw.err_at(d.line, "flytspänningen ska vara positiv");
        }
        match t.last() {
            Some(&(pl, _)) if !(p > pl) => return kw.err_at(d.line, "den plastiska töjningen ska öka"),
            None if p != 0.0 => return kw.err_at(d.line, "första punkten ska ha plastisk töjning 0"),
            _ => {}
        }
        t.push((p, s));
    }
    if t.is_empty() {
        return kw.err("flytspänning saknas");
    }
    Ok(t)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    let kw = def.behavior("DRUCKER PRAGER").unwrap();
    let Some(el) = def.behavior("ELASTIC") else { return kw.err("*DRUCKER PRAGER kräver *ELASTIC") };
    if el.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || el.data.len() > 1 {
        return el.err("Drucker–Prager kräver isotrop, temperaturoberoende elasticitet");
    }
    let ev = el.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0 && nu > -1.0 && nu < 0.5) {
        return el.err("E ska vara positivt och ν mellan −1 och 0,5");
    }
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperaturberoende parametrar stöds inte ännu");
    }
    if kw.has("TEST DATA") {
        return kw.err("TEST DATA (anpassning till triaxialförsök) stöds inte ännu");
    }
    let Some(hk) = def.behavior("DRUCKER PRAGER HARDENING") else {
        return kw.err("*DRUCKER PRAGER kräver *DRUCKER PRAGER HARDENING");
    };
    if hk.has("RATE") || hk.has("DEPENDENCIES") || def.behavior("RATE DEPENDENT").is_some() {
        return hk.err("hastighets- eller temperaturberoende hårdnande stöds inte ännu");
    }
    let htype = match hk.param_upper("TYPE").as_deref() {
        None | Some("COMPRESSION") => HType::Compression,
        Some("TENSION") => HType::Tension,
        Some("SHEAR") => HType::Shear,
        Some(o) => return hk.err(format!("TYPE={o} ska vara COMPRESSION, TENSION eller SHEAR")),
    };
    let hard = Hardening::Table(table(hk)?);
    let s0 = hard.initial();
    let v = kw.first_floats()?;
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let deg = |x: f64| x.to_radians().tan();
    let ecc_param = match kw.param("ECCENTRICITY") {
        Some(s) => Some(crate::fem::model::parse_f64(s).filter(|&x| x > 0.0).ok_or_else(|| crate::model::FemError(format!("{}: ECCENTRICITY ska vara ett positivt tal", kw.location())))?),
        None => None,
    };
    let crit_name = kw.param_upper("SHEAR CRITERION").unwrap_or_else(|| "LINEAR".into());
    let (crit, tb, tp, ecc) = match crit_name.as_str() {
        "HYPERBOLIC" => {
            let (beta, pt0, psi) = (g(0), g(1), g(3));
            if !(beta > 0.0 && beta < 90.0) || !(0.0..90.0).contains(&psi) {
                return kw.err("β ska ligga mellan 0 och 90 grader (ej 0) och ψ mellan 0 och 90 grader");
            }
            if !(pt0 > 0.0) {
                return kw.err("den hydrostatiska draghållfastheten pt|0 ska vara positiv");
            }
            let (tb, tp) = (deg(beta), deg(psi));
            let a = match htype {
                HType::Compression => (pt0 + s0 / 3.0) * tb,
                HType::Tension => (pt0 - s0 / 3.0) * tb,
                HType::Shear => pt0 * tb,
            };
            if !(a > 0.0) {
                return kw.err("pt|0 är för liten jämfört med flytspänningen (l0 kan inte bestämmas)");
            }
            let l0 = (s0 * s0 - a * a) / (2.0 * a);
            if !(l0 > 0.0) {
                return kw.err("l0 = d′|0 − pt|0 tan β blir inte positiv – kontrollera pt|0 och flytspänningen");
            }
            let eps = ecc_param.unwrap_or(l0 / (s0 * tb));
            (Criterion::Hyperbolic { l0 }, tb, tp, eps * s0 * tp)
        }
        "EXPONENT FORM" => {
            let (a, b, psi) = (g(0), g(1), g(3));
            if !(a > 0.0) || !(b >= 1.0) {
                return kw.err("exponentformen kräver a > 0 och b ≥ 1");
            }
            if !(0.0..90.0).contains(&psi) {
                return kw.err("ψ ska ligga mellan 0 och 90 grader");
            }
            let tp = deg(psi);
            let eps = ecc_param.unwrap_or(0.1);
            (Criterion::Exponent { a, b }, 0.0, tp, eps * s0 * tp)
        }
        _ => {
            let (beta, k, psi) = (g(0), g(1), v.get(2).copied().unwrap_or(g(0)));
            if !(0.778 - 1e-9..=1.0).contains(&k) {
                return kw.err("K ska ligga mellan 0,778 och 1 (konvex flytyta)");
            }
            if !(0.0..90.0).contains(&beta) || !(0.0..90.0).contains(&psi) {
                return kw.err("β och ψ ska ligga mellan 0 och 90 grader");
            }
            (Criterion::Linear { k }, deg(beta), deg(psi), 0.0)
        }
    };
    let gm = e / (2.0 * (1.0 + nu));
    let kb = e / (3.0 * (1.0 - 2.0 * nu));
    let m = DruckerPrager2 { e, nu, g: gm, kb, ce: iso_d6(e, nu), crit, tan_beta: tb, tan_psi: tp, ecc, htype, hard };
    if let Criterion::Linear { k } = crit {
        if m.cohesion_factor(k) <= 0.0 {
            return kw.err("tan β ≥ 3 ger ingen kohesion för hårdnande i tryck");
        }
    }
    if let Criterion::Exponent { .. } = crit {
        if m.pt(s0) <= 0.0 {
            return kw.err("exponentformen ger ingen positiv hydrostatisk draghållfasthet pt = a σᵇ ∓ σ/3 vid den första flytspänningen");
        }
    }
    Ok(Some(Arc::new(m)))
}

impl DruckerPrager2 {
    fn cohesion_factor(&self, k: f64) -> f64 {
        match self.htype {
            HType::Compression => 1.0 - self.tan_beta / 3.0,
            HType::Tension => 1.0 / k + self.tan_beta / 3.0,
            // tabellen ger kohesionen d direkt
            HType::Shear => 1.0,
        }
    }

    /// d′ (hyperbolisk) som funktion av hårdnandespänningen σ̄ och dd′/dσ̄.
    fn dprime(&self, l0: f64, s: f64) -> (f64, f64) {
        let r = (l0 * l0 + s * s).sqrt();
        match self.htype {
            HType::Compression => (r - s * self.tan_beta / 3.0, s / r - self.tan_beta / 3.0),
            HType::Tension => (r + s * self.tan_beta / 3.0, s / r + self.tan_beta / 3.0),
            HType::Shear => (r, s / r),
        }
    }

    /// pt (exponentform) som funktion av σ̄.
    fn pt(&self, s: f64) -> f64 {
        let Criterion::Exponent { a, b } = self.crit else { return 0.0 };
        match self.htype {
            HType::Compression => a * s.powf(b) - s / 3.0,
            HType::Tension => a * s.powf(b) + s / 3.0,
            HType::Shear => a * s.powf(b),
        }
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        stn[..NST].copy_from_slice(&st[..NST]);
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        match return_map(self, self.g, self.kb, &sig_tr, st[P]) {
            Ok(None) => Ok(sig_tr),
            Ok(Some(r)) => {
                stn[P] = r.h;
                for i in 0..6 {
                    stn[EP + i] = st[EP + i] + r.dep[i];
                }
                Ok(r.sig)
            }
            Err(e) => Err(crate::model::FemError(format!("Drucker–Prager ({}): {e}", self.name_crit()))),
        }
    }

    fn name_crit(&self) -> &'static str {
        match self.crit {
            Criterion::Hyperbolic { .. } => "HYPERBOLIC",
            Criterion::Exponent { .. } => "EXPONENT FORM",
            Criterion::Linear { .. } => "LINEAR",
        }
    }

    fn update_tangent(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let sig = self.update3d(eps, st, stn)?;
        if stn[P] == st[P] && (0..6).all(|i| stn[EP + i] == st[EP + i]) {
            return Ok((sig, self.ce));
        }
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.hard.initial() / self.e);
        let h = 1e-6 * scale;
        let mut c = [[0.0; 6]; 6];
        let mut tmp = vec![0.0; NST];
        for j in 0..6 {
            let mut ep = *eps;
            let mut em = *eps;
            ep[j] += h;
            em[j] -= h;
            let sp = self.update3d(&ep, st, &mut tmp)?;
            let sm = self.update3d(&em, st, &mut tmp)?;
            for i in 0..6 {
                c[i][j] = (sp[i] - sm[i]) / (2.0 * h);
            }
        }
        Ok((sig, c))
    }
}

impl Surface for DruckerPrager2 {
    fn k(&self) -> f64 {
        match self.crit {
            Criterion::Linear { k } => k,
            _ => 1.0,
        }
    }

    fn yield_fn(&self, p: f64, t: f64, h: f64) -> (f64, f64, f64) {
        let s = self.hard.eval(h).0;
        match self.crit {
            Criterion::Hyperbolic { l0 } => {
                let r = (l0 * l0 + t * t).sqrt();
                (r - p * self.tan_beta - self.dprime(l0, s).0, -self.tan_beta, t / r)
            }
            Criterion::Exponent { a, b } => (a * t.powf(b) - p - self.pt(s), -1.0, if t > 0.0 { a * b * t.powf(b - 1.0) } else { 0.0 }),
            Criterion::Linear { k } => (t - p * self.tan_beta - self.cohesion_factor(k) * s, -self.tan_beta, 1.0),
        }
    }

    fn potential(&self, _p: f64, t: f64, _h: f64) -> (f64, f64) {
        match self.crit {
            Criterion::Linear { .. } => (-self.tan_psi, 1.0),
            _ => {
                let r = (self.ecc * self.ecc + t * t).sqrt();
                (-self.tan_psi, if r > 0.0 { t / r } else { 0.0 })
            }
        }
    }

    fn hardening_rate(&self, sig: &V6, _inv: &Inv, n: &V6, h: f64, _h_n: f64) -> f64 {
        match self.crit {
            Criterion::Linear { k } => match self.htype {
                HType::Compression => 1.0 - self.tan_psi / 3.0,
                HType::Tension => 1.0 / k + self.tan_psi / 3.0,
                HType::Shear => 0.5 * (1.0 + 1.0 / k),
            },
            _ => ddot(sig, n) / self.hard.eval(h).0,
        }
    }

    fn stress_scale(&self, h: f64) -> f64 {
        self.hard.eval(h).0
    }
}

impl MaterialModel for DruckerPrager2 {
    fn name(&self) -> &str {
        "DRUCKER PRAGER"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[EP..EP + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update_tangent(e, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[P]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[P]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v
    }
}
