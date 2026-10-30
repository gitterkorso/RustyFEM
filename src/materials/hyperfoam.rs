//! Elastomerskum (*HYPERFOAM, N=n): kompressibel Ogden-form enligt Storåkers
//! (1986).
//!
//!   U = Σ_i 2μ_i/α_i² [ λ1^α_i + λ2^α_i + λ3^α_i − 3 + (1/β_i)(J^(−α_i β_i) − 1) ]
//!
//! med huvudsträckningarna λ_a (inte deviatoriska), J = λ1 λ2 λ3 och
//! β_i = ν_i/(1 − 2ν_i). Data: μ1, α1, ..., μn, αn, ν1, ..., νn (n ≤ 6);
//! med `POISSON=ν` gäller ν för alla termer och datat är bara μ_i, α_i.
//! β_i = 0 (ν_i = 0) ger gränsvärdet −α_i ln J i hakparentesen.
//!
//! Kirchhoffspänningen har huvudvärdena τ_a = Σ_i (2μ_i/α_i)(λ_a^α_i − J^(−α_i β_i)),
//! dvs τ = Σ_i (2μ_i/α_i)(b^(α_i/2) − J^(−α_i β_i) I) med b = F Fᵀ. Initial
//! skjuvmodul μ0 = Σ μ_i och kompressionsmodul K0 = Σ 2μ_i (1/3 + β_i).
//!
//! Modellen delas inte i deviatorisk och volymetrisk del, så den ger hela
//! spänningen genom `HyperModel::kirchhoff_full`; kontinuumelementen (F-bar i
//! C3D8/CPE4, plan spänning i CPS4, motorn i `solid_higher.rs` för CAX4 m.fl.)
//! använder den direkt. Hybridelement stöds inte med *HYPERFOAM.

use super::hyperelastic_split::{self as hs, HyperModel};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory { name: "elastomerskum (*HYPERFOAM)", consumes: &["HYPERFOAM"], priority: 10, build: Some(build) }];

/// Hyperfoam: termer (μ_i, α_i, β_i).
#[derive(Debug, Clone)]
pub struct HyperFoam {
    pub terms: Vec<(f64, f64, f64)>,
}

/// N= på ett hyperelastiskt nyckelord (1–6, standard 1).
pub fn n_param(kw: &crate::fem::model::RawKeyword) -> Result<usize> {
    match kw.param("N") {
        Some(s) => s.trim().parse::<usize>().ok().filter(|&n| (1..=6).contains(&n)).ok_or_else(|| FemError(format!("{}: N={s} ska vara 1–6", kw.location()))),
        None => Ok(1),
    }
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERFOAM") else { return Ok(None) };
    if def.behavior("ELASTIC").is_some() || def.behavior("HYPERELASTIC").is_some() {
        return kw.err("*HYPERFOAM tillsammans med *ELASTIC eller *HYPERELASTIC i samma material stöds inte");
    }
    if def.behavior("MULLINS EFFECT").is_some() {
        // byggs av mullins.rs
        return Ok(None);
    }
    Ok(Some(Arc::new(parse(def)?)))
}

/// Läser *HYPERFOAM ur materialdefinitionen.
pub fn parse(def: &MaterialDef) -> Result<HyperFoam> {
    let Some(kw) = def.behavior("HYPERFOAM") else { return fail(format!("material {}: *HYPERFOAM saknas", def.name)) };
    for p in ["TEST DATA INPUT", "USER", "PROPERTIES"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu för *HYPERFOAM"));
        }
    }
    if kw.has("MODULI") && kw.param_upper("MODULI").as_deref() != Some("LONG TERM") {
        return kw.err("MODULI stöds bara som LONG TERM");
    }
    let n = n_param(kw)?;
    let poisson = match kw.param("POISSON") {
        Some(s) => Some(s.trim().parse::<f64>().map_err(|_| FemError(format!("{}: POISSON={s} är inget tal", kw.location())))?),
        None => None,
    };
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    let need = if poisson.is_some() { 2 * n } else { 3 * n };
    if v.len() < need {
        return kw.err(format!("*HYPERFOAM, N={n} kräver {need} konstanter, {} angivna", v.len()));
    }
    if v[need..].iter().any(|&x| x != 0.0) {
        return kw.err("temperaturberoende konstanter (flera datarader) stöds inte ännu");
    }
    let mut terms = Vec::with_capacity(n);
    for i in 0..n {
        let (mu, al) = (v[2 * i], v[2 * i + 1]);
        let nu = poisson.unwrap_or_else(|| v[2 * n + i]);
        if al == 0.0 {
            return kw.err("*HYPERFOAM: α_i = 0 är inte tillåtet");
        }
        if !(nu < 0.5 && nu > -1.0) {
            return kw.err(format!("*HYPERFOAM: ν_{} = {nu} ska ligga i (−1, 0,5)", i + 1));
        }
        terms.push((mu, al, nu / (1.0 - 2.0 * nu)));
    }
    let m = HyperFoam { terms };
    if !(m.mu0() > 0.0 && m.k0() > 0.0) {
        return kw.err("*HYPERFOAM: den initiala skjuvmodulen Σμ_i och kompressionsmodulen ska vara positiva");
    }
    Ok(m)
}

impl HyperFoam {
    fn tau(&self, f: &M3) -> Result<M3> {
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("*HYPERFOAM: J ≤ 0 (elementet har vänts ut och in)");
        }
        let b = t::mul_bt(f, f);
        let (ev, vecs) = t::eig_sym(&b);
        let lam2: [f64; 3] = ev.map(|e| e.max(1e-300));
        let mut ta = [0.0; 3];
        let mut iso = 0.0;
        for &(mu, al, be) in &self.terms {
            for a in 0..3 {
                ta[a] += 2.0 * mu / al * lam2[a].powf(0.5 * al);
            }
            iso -= 2.0 * mu / al * j.powf(-al * be);
        }
        let mut tau: M3 = std::array::from_fn(|i| std::array::from_fn(|k| (0..3).map(|a| ta[a] * vecs[i][a] * vecs[k][a]).sum()));
        for i in 0..3 {
            tau[i][i] += iso;
        }
        Ok(tau)
    }

    /// Töjningsenergin per referensvolym.
    pub fn energy_total(&self, f: &M3) -> f64 {
        let j = t::det(f);
        let (ev, _) = t::eig_sym(&t::mul_bt(f, f));
        let mut u = 0.0;
        for &(mu, al, be) in &self.terms {
            let s: f64 = ev.iter().map(|e| e.max(1e-300).powf(0.5 * al)).sum();
            let v = if be == 0.0 { -al * j.ln() } else { (j.powf(-al * be) - 1.0) / be };
            u += 2.0 * mu / (al * al) * (s - 3.0 + v);
        }
        u
    }
}

/// Isotrop elastisk tangent (μ, K) i 6×6 reducerad till spänningstillståndet.
pub fn isotropic_tangent(mu: f64, k: f64, mode: StressMode) -> Vec<f64> {
    let lam = k - 2.0 * mu / 3.0;
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for jj in 0..3 {
            d[i][jj] = lam;
        }
        d[i][i] = lam + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    super::anisotropic::reduce_6x6(&d, mode)
}

/// (E, ν) ur (μ, K).
pub fn moduli(mu: f64, k: f64) -> (f64, f64) {
    if !k.is_finite() {
        return (3.0 * mu, 0.5);
    }
    (9.0 * k * mu / (3.0 * k + mu), (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu)))
}

impl HyperModel for HyperFoam {
    fn mu0(&self) -> f64 {
        self.terms.iter().map(|&(m, _, _)| m).sum()
    }

    fn k0(&self) -> f64 {
        self.terms.iter().map(|&(m, _, b)| 2.0 * m * (1.0 / 3.0 + b)).sum()
    }

    fn tau_dev(&self, f: &M3) -> Result<M3> {
        let mut tau = self.tau(f)?;
        let tr = (tau[0][0] + tau[1][1] + tau[2][2]) / 3.0;
        for i in 0..3 {
            tau[i][i] -= tr;
        }
        Ok(tau)
    }

    fn vol(&self, _j: f64) -> (f64, f64) {
        (0.0, 0.0)
    }

    fn kirchhoff_full(&self, f: &M3) -> Option<Result<M3>> {
        Some(self.tau(f))
    }

    fn energy(&self, f: &M3) -> Option<(f64, f64)> {
        Some((self.energy_total(f), 0.0))
    }
}

impl MaterialModel for HyperFoam {
    fn name(&self) -> &str {
        "HYPERFOAM"
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some(moduli(self.mu0(), self.k0()))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        if input.defgrad.is_none() {
            return fail("*HYPERFOAM kräver NLGEOM=YES på *STEP");
        }
        hs::update_hyper_hist(self, input, &[], &mut [])
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(isotropic_tangent(self.mu0(), self.k0(), mode))
    }

    fn hyperelastic(&self) -> Option<&dyn HyperModel> {
        Some(self)
    }
}
