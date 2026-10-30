//! Hyperelasticitet: Van der Waals (*HYPERELASTIC, VAN DER WAALS), Kilian
//! (1981):
//!
//!   U = μ { −(λm² − 3)[ln(1 − η) + η] − ⅔ a ((Ĩ − 3)/2)^(3/2) } + (1/D)((J² − 1)/2 − ln J)
//!
//! med Ĩ = (1 − β) Ī1 + β Ī2 och η = √((Ĩ − 3)/(λm² − 3)). Data: μ, λm, a, β, D.
//! Initial skjuvmodul μ0 = μ, kompressionsmodul K0 = 2/D (D = 0: helt
//! inkompressibelt, hybridelement).
//!
//! Med U_Ĩ = ∂U/∂Ĩ = μ [1/(2(1 − η)) − (a/2)√((Ĩ − 3)/2)] blir U1 = (1 − β) U_Ĩ och
//! U2 = β U_Ĩ, och den deviatoriska Kirchhoffspänningen är som för
//! polynomformerna τ_dev = 2 dev[(U1 + Ī1 U2) b̄ − U2 b̄²].

use super::hyperelastic_ogden::update_hyper;
use super::hyperelastic_split::HyperModel;
use super::hyperfoam::{isotropic_tangent, moduli};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "hyperelasticitet (Van der Waals)", consumes: &["HYPERELASTIC"], priority: 10, build: Some(build) }];

#[derive(Debug, Clone)]
pub struct VanDerWaals {
    pub mu: f64,
    pub lm: f64,
    pub a: f64,
    pub beta: f64,
    pub d: f64,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    if !kw.has("VAN DER WAALS") {
        return Ok(None);
    }
    if def.behavior("MULLINS EFFECT").is_some() {
        return Ok(None);
    }
    Ok(Some(Arc::new(parse(def)?)))
}

/// Läser *HYPERELASTIC, VAN DER WAALS.
pub fn parse(def: &MaterialDef) -> Result<VanDerWaals> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return fail(format!("material {}: *HYPERELASTIC saknas", def.name)) };
    if def.behavior("ELASTIC").is_some() {
        return kw.err("*HYPERELASTIC och *ELASTIC i samma material stöds inte");
    }
    for p in ["TEST DATA INPUT", "USER", "TYPE", "PROPERTIES"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu för VAN DER WAALS"));
        }
    }
    if kw.has("MODULI") && kw.param_upper("MODULI").as_deref() != Some("LONG TERM") {
        return kw.err("MODULI stöds bara som LONG TERM");
    }
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    if v.len() < 5 {
        return kw.err(format!("VAN DER WAALS kräver 5 konstanter (μ, λm, a, β, D), {} angivna", v.len()));
    }
    if v[5..].iter().any(|&x| x != 0.0) {
        return kw.err("temperaturberoende konstanter (flera datarader) stöds inte ännu");
    }
    let m = VanDerWaals { mu: v[0], lm: v[1], a: v[2], beta: v[3], d: v[4] };
    if !(m.mu > 0.0) {
        return kw.err("VAN DER WAALS: μ ska vara positiv");
    }
    if !(m.lm * m.lm > 3.0) {
        return kw.err("VAN DER WAALS: låsningssträckningen λm ska vara större än √3");
    }
    if !(0.0..=1.0).contains(&m.beta) || m.d < 0.0 {
        return kw.err("VAN DER WAALS: β ska ligga i [0, 1] och D ≥ 0");
    }
    Ok(m)
}

impl VanDerWaals {
    /// (Ī1, Ī2, b̄) för F.
    fn inv(f: &M3) -> Result<(f64, f64, M3)> {
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let bb = t::scale(&t::mul_bt(f, f), j.powf(-2.0 / 3.0));
        let b2 = t::mul(&bb, &bb);
        let i1 = bb[0][0] + bb[1][1] + bb[2][2];
        let i2 = 0.5 * (i1 * i1 - (b2[0][0] + b2[1][1] + b2[2][2]));
        Ok((i1, i2, bb))
    }

    fn eta(&self, it: f64) -> Result<f64> {
        let e = ((it - 3.0).max(0.0) / (self.lm * self.lm - 3.0)).sqrt();
        if !(e < 1.0) {
            return fail("VAN DER WAALS: låsningssträckningen λm är överskriden (η ≥ 1)");
        }
        Ok(e)
    }

    /// Deviatorisk energi.
    pub fn energy_dev(&self, f: &M3) -> Result<f64> {
        let (i1, i2, _) = Self::inv(f)?;
        let it = (1.0 - self.beta) * i1 + self.beta * i2;
        let e = self.eta(it)?;
        let x = (it - 3.0).max(0.0) / 2.0;
        Ok(self.mu * (-(self.lm * self.lm - 3.0) * ((1.0 - e).ln() + e) - 2.0 / 3.0 * self.a * x.powf(1.5)))
    }
}

impl HyperModel for VanDerWaals {
    fn mu0(&self) -> f64 {
        self.mu
    }

    fn k0(&self) -> f64 {
        if self.d > 0.0 {
            2.0 / self.d
        } else {
            f64::INFINITY
        }
    }

    fn tau_dev(&self, f: &M3) -> Result<M3> {
        let (i1, i2, bb) = Self::inv(f)?;
        let it = (1.0 - self.beta) * i1 + self.beta * i2;
        let e = self.eta(it)?;
        let ui = self.mu * (0.5 / (1.0 - e) - 0.5 * self.a * ((it - 3.0).max(0.0) / 2.0).sqrt());
        let (u1, u2) = ((1.0 - self.beta) * ui, self.beta * ui);
        let b2 = t::mul(&bb, &bb);
        let mut a: M3 = std::array::from_fn(|i| std::array::from_fn(|k| (u1 + i1 * u2) * bb[i][k] - u2 * b2[i][k]));
        let tr = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
        for i in 0..3 {
            a[i][i] -= tr;
        }
        Ok(t::scale(&a, 2.0))
    }

    fn vol(&self, j: f64) -> (f64, f64) {
        if self.d > 0.0 {
            ((j - 1.0 / j) / self.d, (1.0 + 1.0 / (j * j)) / self.d)
        } else {
            (0.0, 0.0)
        }
    }

    fn energy(&self, f: &M3) -> Option<(f64, f64)> {
        let j = t::det(f);
        let uv = if self.d > 0.0 { ((j * j - 1.0) / 2.0 - j.ln()) / self.d } else { 0.0 };
        self.energy_dev(f).ok().map(|ud| (ud, uv))
    }
}

impl MaterialModel for VanDerWaals {
    fn name(&self) -> &str {
        "HYPERELASTIC"
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some(moduli(self.mu0(), self.k0()))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        update_hyper(self, input)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        if self.incompressible() {
            return fail("*HYPERELASTIC med D = 0 (helt inkompressibelt) kräver hybridelement (C3D8H, CPE4H, ...)");
        }
        Ok(isotropic_tangent(self.mu0(), self.k0(), mode))
    }

    fn hyperelastic(&self) -> Option<&dyn HyperModel> {
        Some(self)
    }
}
