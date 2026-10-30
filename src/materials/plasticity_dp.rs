//! Linjär Drucker–Prager-plasticitet (*DRUCKER PRAGER, SHEAR CRITERION=LINEAR,
//! K = 1) med hårdnande från *DRUCKER PRAGER HARDENING.
//!
//! Flytvillkor och flödespotential (p = −tr σ/3, q = von Mises-spänning):
//!
//!   F = q − p tan β − d(ε̄p) = 0,   G = q − p tan ψ
//!
//! där β är friktionsvinkeln och ψ dilatationsvinkeln (icke-associativt flöde
//! när ψ ≠ β). Kohesionen d bestäms av hårdnandekurvan:
//! - TYPE=COMPRESSION (standard): d = (1 − tan β/3) σc(ε̄p),
//!   ε̄̇p = λ̇ (1 − tan ψ/3) (axiell plastisk töjning vid enaxligt tryck),
//! - TYPE=TENSION: d = (1 + tan β/3) σt(ε̄p), ε̄̇p = λ̇ (1 + tan ψ/3),
//! - TYPE=SHEAR: d = √3 τ(ε̄p)... stöds inte ännu.
//!
//! Implicit Euler: med prövotillståndet (q_tr, p_tr) blir
//! q = q_tr − 3G Δλ och p = p_tr + K Δλ tan ψ, och den skalära ekvationen
//! q − p tan β − d(ε̄p_n + c_ψ Δλ) = 0 löses med Newton. Om q skulle bli
//! negativt görs retur till konens spets (q = 0). Tangenten är numerisk
//! (framåtdifferens) och osymmetrisk vid icke-associativt flöde. Assembleringen
//! använder dess symmetriska del med symmetriska lösare (standard utan
//! UNSYMM) och hela tangenten med *STEP, UNSYMM=YES.

use super::plasticity::{dev, iso_d6, mises, mv6, update_reduced, Hardening, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Drucker–Prager-plasticitet",
    consumes: &["ELASTIC", "DRUCKER PRAGER", "DRUCKER PRAGER HARDENING"],
    priority: 10,
    build: Some(build),
}];

#[derive(Debug, Clone)]
pub struct DruckerPrager {
    pub e: f64,
    pub nu: f64,
    pub g: f64,
    pub k: f64,
    pub ce: M6,
    pub tan_beta: f64,
    pub tan_psi: f64,
    /// d = cd · σ_kurva(ε̄p), ε̄̇p = ce_rate · λ̇
    pub cd: f64,
    pub c_rate: f64,
    pub hard: Hardening,
}

const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const NST: usize = 13;

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 3 && v[3..].iter().any(|&x| x != 0.0) || v.get(2).is_some_and(|&x| x != 0.0) {
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
    let Some(kw) = def.behavior("DRUCKER PRAGER") else { return Ok(None) };
    if super::plasticity_dp2::claims(def) {
        return Ok(None);
    }
    let Some(el) = def.behavior("ELASTIC") else { return kw.err("*DRUCKER PRAGER kräver *ELASTIC") };
    if el.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || el.data.len() > 1 {
        return el.err("Drucker–Prager kräver isotrop, temperaturoberoende elasticitet");
    }
    let ev = el.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0 && nu > -1.0 && nu < 0.5) {
        return el.err("E ska vara positivt och ν mellan −1 och 0,5");
    }
    match kw.param_upper("SHEAR CRITERION").as_deref() {
        None | Some("LINEAR") => {}
        Some(o) => return kw.err(format!("SHEAR CRITERION={o} stöds inte ännu (bara LINEAR)")),
    }
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperaturberoende parametrar stöds inte ännu");
    }
    let v = kw.first_floats()?;
    let beta = v.first().copied().unwrap_or(0.0);
    let kk = v.get(1).copied().filter(|&x| x != 0.0).unwrap_or(1.0);
    let psi = v.get(2).copied().unwrap_or(beta);
    if (kk - 1.0).abs() > 1e-12 {
        return kw.err("K ≠ 1 (tredje invariantens inverkan) stöds inte ännu");
    }
    if !(0.0..90.0).contains(&beta) || !(0.0..90.0).contains(&psi) {
        return kw.err("β och ψ ska ligga mellan 0 och 90 grader");
    }
    let Some(hk) = def.behavior("DRUCKER PRAGER HARDENING") else {
        return kw.err("*DRUCKER PRAGER kräver *DRUCKER PRAGER HARDENING");
    };
    if hk.has("RATE") || hk.has("DEPENDENCIES") {
        return hk.err("hastighets- eller temperaturberoende hårdnande stöds inte ännu");
    }
    let (tb, tp) = (beta.to_radians().tan(), psi.to_radians().tan());
    let (cd, c_rate) = match hk.param_upper("TYPE").as_deref() {
        None | Some("COMPRESSION") => (1.0 - tb / 3.0, 1.0 - tp / 3.0),
        Some("TENSION") => (1.0 + tb / 3.0, 1.0 + tp / 3.0),
        Some(o) => return hk.err(format!("TYPE={o} stöds inte ännu (COMPRESSION, TENSION)")),
    };
    if !(cd > 0.0) {
        return kw.err("tan β ≥ 3 ger ingen kohesion för hårdnande i tryck");
    }
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    Ok(Some(Arc::new(DruckerPrager { e, nu, g, k, ce: iso_d6(e, nu), tan_beta: tb, tan_psi: tp, cd, c_rate, hard: Hardening::Table(table(hk)?) })))
}

impl DruckerPrager {
    fn ret(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, bool)> {
        let (g, k) = (self.g, self.k);
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        let s_tr = dev(&sig_tr);
        let q_tr = mises(&s_tr);
        let p_tr = -(sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0;
        let pn = st[P];
        stn[..NST].copy_from_slice(&st[..NST]);
        let d = |ep: f64| -> (f64, f64) {
            let (s, h) = self.hard.eval(ep);
            (self.cd * s, self.cd * h)
        };
        let f_tr = q_tr - p_tr * self.tan_beta - d(pn).0;
        if f_tr <= 1e-12 * d(pn).0 {
            return Ok((sig_tr, false));
        }
        // Newton på Δλ
        let gfun = |dl: f64| -> (f64, f64) {
            let (dv, h) = d(pn + self.c_rate * dl);
            let q = q_tr - 3.0 * g * dl;
            let p = p_tr + k * dl * self.tan_psi;
            (q - p * self.tan_beta - dv, -3.0 * g - k * self.tan_psi * self.tan_beta - h * self.c_rate)
        };
        let mut dl = 0.0;
        let mut conv = false;
        for _ in 0..100 {
            let (gv, dg) = gfun(dl);
            if gv.abs() <= 1e-13 * q_tr.max(d(pn).0) {
                conv = true;
                break;
            }
            dl -= gv / dg;
            if dl < 0.0 {
                dl = 0.0;
            }
        }
        if !conv {
            return fail("returalgoritmen för Drucker–Prager konvergerade inte");
        }
        let sig;
        let mut dep = [0.0; 6];
        if q_tr - 3.0 * g * dl >= 0.0 {
            let q = q_tr - 3.0 * g * dl;
            let p = p_tr + k * dl * self.tan_psi;
            let f = if q_tr > 0.0 { q / q_tr } else { 0.0 };
            let mut s = [0.0; 6];
            for i in 0..6 {
                s[i] = s_tr[i] * f;
            }
            for i in 0..3 {
                s[i] -= p;
            }
            sig = s;
            // Δεp = Δλ (3/2 s_tr/q_tr + tan ψ/3 I), ingenjörsskjuvning
            for i in 0..6 {
                let n = if q_tr > 0.0 { 1.5 * s_tr[i] / q_tr } else { 0.0 };
                dep[i] = dl * n * if i < 3 { 1.0 } else { 2.0 };
            }
            for i in 0..3 {
                dep[i] += dl * self.tan_psi / 3.0;
            }
            stn[P] = pn + self.c_rate * dl;
        } else {
            // retur till spetsen: q = 0, p så att F = 0 (kohesion vid ε̄p_n)
            if self.tan_beta <= 0.0 {
                return fail("Drucker–Prager: spetsretur kräver β > 0");
            }
            let p = -d(pn).0 / self.tan_beta;
            sig = [-p, -p, -p, 0.0, 0.0, 0.0];
            for i in 0..6 {
                dep[i] = s_tr[i] / (2.0 * g) * if i < 3 { 1.0 } else { 2.0 };
            }
            let dvol = (p - p_tr) / k;
            for i in 0..3 {
                dep[i] += dvol / 3.0;
            }
            // deviatorisk del som vid q = 0: Δλ = q_tr/(3G)
            stn[P] = pn + self.c_rate * q_tr / (3.0 * g);
        }
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dep[i];
        }
        Ok((sig, true))
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let (sig, plastic) = self.ret(eps, st, stn)?;
        if !plastic {
            return Ok((sig, self.ce));
        }
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.cd * self.hard.initial() / self.e);
        let h = 1e-7 * scale;
        let mut c = [[0.0; 6]; 6];
        let mut tmp = vec![0.0; NST];
        for j in 0..6 {
            let mut e2 = *eps;
            e2[j] += h;
            let (s2, _) = self.ret(&e2, st, &mut tmp)?;
            for i in 0..6 {
                c[i][j] = (s2[i] - sig[i]) / h;
            }
        }
        Ok((sig, c))
    }
}

impl MaterialModel for DruckerPrager {
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
        super::plasticity::rotate_v6(r, &mut state[EP..EP + 6], true);
        super::plasticity::rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, state_old, state_new) };
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
        vec![("PEEQ", vec!["PEEQ"]), ("PE", super::plasticity::pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[P]];
        v.extend(super::plasticity::pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v
    }
}
