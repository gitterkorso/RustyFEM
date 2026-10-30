//! Mullins-effekt (*MULLINS EFFECT) enligt Ogden & Roxburgh (1999),
//! ovanpå *HYPERELASTIC (alla former) och *HYPERFOAM.
//!
//! Energin är U = η Ũ_dev(F) + φ(η) + Ũ_vol(J) med skadevariabeln
//!
//!   η = 1 − (1/r) erf[(U_m − Ũ_dev)/(m + β U_m)],
//!
//! där Ũ_dev är grundmaterialets deviatoriska energi och U_m dess största värde
//! hittills (uppdateras vid varje inkrement). Spänningen blir
//! τ = η τ̃_dev + J Ũ_vol′ I. För *HYPERFOAM (ingen uppdelning) gäller η hela
//! energin: Ũ_dev = Ũ och τ = η τ̃. Data: r, m, β.
//!
//! Skadefunktionen (φ′(η) = −Ũ_dev, φ(1) = 0):
//!
//!   φ(η) = (1 − η) U_m − (m + β U_m)/(r√π) [1 − exp(−z²)],  z = (U_m − Ũ_dev)/(m + β U_m).
//!
//! Utdata: SENER = η Ũ_dev + Ũ_vol (återvinningsbar energi per referensvolym) och
//! DMENER = φ(η) (energi som skadan förbrukat), så att SENER + DMENER = det
//! arbete som tillförts.
//!
//! Tillstånd per integrationspunkt: [U_m, SENER, DMENER]. Tangenten är
//! derivatan av spänningen som funktion av F vid givet U_m vid inkrementets
//! början (pålastning på grundkurvan ger η = 1 och grundmaterialets tangent).

use super::hyperelastic_split::{self as hs, HyperModel};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Mullins-effekt (*MULLINS EFFECT)",
    consumes: &["HYPERELASTIC", "HYPERFOAM", "MULLINS EFFECT", "UNIAXIAL TEST DATA", "BIAXIAL TEST DATA", "PLANAR TEST DATA", "VOLUMETRIC TEST DATA"],
    priority: 20,
    build: Some(build),
}];

pub struct Mullins {
    pub inner: Arc<dyn MaterialModel>,
    pub r: f64,
    pub m: f64,
    pub beta: f64,
    /// grundmaterialet är *HYPERFOAM (η gäller hela energin)
    pub full: bool,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("MULLINS EFFECT") else { return Ok(None) };
    if def.behavior("HYPERELASTIC").is_none() && def.behavior("HYPERFOAM").is_none() {
        return kw.err("*MULLINS EFFECT kräver *HYPERELASTIC eller *HYPERFOAM i samma material");
    }
    for p in ["TEST DATA INPUT", "USER", "DEPENDENCIES", "PROPERTIES"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu för *MULLINS EFFECT"));
        }
    }
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    if v.len() < 3 {
        return kw.err(format!("*MULLINS EFFECT kräver 3 konstanter (r, m, β), {} angivna", v.len()));
    }
    if v[3..].iter().any(|&x| x != 0.0) {
        return kw.err("temperaturberoende konstanter (flera datarader) stöds inte ännu");
    }
    let (r, m, beta) = (v[0], v[1], v[2]);
    if !(r > 0.0) || m < 0.0 || beta < 0.0 || (m == 0.0 && beta == 0.0) {
        return kw.err("*MULLINS EFFECT: r > 0, m ≥ 0 och β ≥ 0 med m och β inte båda noll");
    }
    let mut d2 = def.clone();
    d2.behaviors.retain(|b| b.name != "MULLINS EFFECT");
    let inner = super::build(&d2)?;
    let Some(h) = inner.hyperelastic() else {
        return kw.err("*MULLINS EFFECT stöds bara ovanpå *HYPERELASTIC och *HYPERFOAM (inte med *VISCOELASTIC)");
    };
    if h.energy(&t::I3).is_none() || h.n_hist() != 0 {
        return kw.err("*MULLINS EFFECT stöds inte för den här hyperelastiska formen");
    }
    let full = h.kirchhoff_full(&t::I3).is_some();
    Ok(Some(Arc::new(Mullins { inner, r, m, beta, full })))
}

/// Felfunktionen (serie för |x| < 2,5, kedjebråk för erfc annars; ~1e-14).
pub fn erf(x: f64) -> f64 {
    if x < 0.0 {
        return -erf(-x);
    }
    if x < 2.5 {
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.0;
        while n < 200.0 {
            n += 1.0;
            term *= -x2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        return 2.0 / std::f64::consts::PI.sqrt() * sum;
    }
    // erfc(x) = exp(−x²)/√π · 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...)))) (Lentz)
    let tiny = 1e-300;
    let mut f = x;
    let mut c = x;
    let mut d = 0.0;
    for k in 1..300 {
        let a = k as f64 / 2.0;
        d = x + a * d;
        d = if d.abs() < tiny { tiny } else { d };
        c = x + a / c;
        c = if c.abs() < tiny { tiny } else { c };
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    1.0 - (-x * x).exp() / (std::f64::consts::PI.sqrt() * f)
}

impl Mullins {
    fn h(&self) -> &dyn HyperModel {
        self.inner.hyperelastic().expect("grundmaterialet är hyperelastiskt")
    }

    /// (η, U_m, SENER, DMENER) för (Ũ_dev, Ũ_vol) och tidigare U_m.
    fn damage(&self, ud: f64, uv: f64, um_old: f64) -> (f64, f64, f64, f64) {
        let um = um_old.max(ud);
        let den = self.m + self.beta * um;
        let z = if den > 0.0 { ((um - ud) / den).max(0.0) } else { 0.0 };
        let eta = 1.0 - erf(z) / self.r;
        let phi = (1.0 - eta) * um - den / (self.r * std::f64::consts::PI.sqrt()) * (1.0 - (-z * z).exp());
        (eta, um, eta * ud + uv, phi)
    }

    fn energies(&self, f: &M3) -> Result<(f64, f64)> {
        match self.h().energy(f) {
            Some(e) => Ok(e),
            None => fail("*MULLINS EFFECT: grundmaterialets energi kunde inte beräknas"),
        }
    }
}

impl HyperModel for Mullins {
    fn mu0(&self) -> f64 {
        self.h().mu0()
    }

    fn k0(&self) -> f64 {
        self.h().k0()
    }

    fn tau_dev(&self, f: &M3) -> Result<M3> {
        self.h().tau_dev(f)
    }

    fn vol(&self, j: f64) -> (f64, f64) {
        self.h().vol(j)
    }

    fn kirchhoff_full(&self, f: &M3) -> Option<Result<M3>> {
        self.h().kirchhoff_full(f)
    }

    fn energy(&self, f: &M3) -> Option<(f64, f64)> {
        self.h().energy(f)
    }

    fn n_hist(&self) -> usize {
        3
    }

    fn kirchhoff_hist(&self, f: &M3, old: &[f64], new: &mut [f64]) -> Result<M3> {
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let (ud, uv) = self.energies(f)?;
        let (eta, um, sener, phi) = self.damage(ud, uv, old[0]);
        new[0] = um;
        new[1] = sener;
        new[2] = phi;
        let h = self.h();
        if self.full {
            return Ok(t::scale(&hs::kirchhoff(h, f)?, eta));
        }
        let mut tau = t::scale(&h.tau_dev(f)?, eta);
        let (up, _) = h.vol(j);
        for i in 0..3 {
            tau[i][i] += j * up;
        }
        Ok(tau)
    }

    fn tau_dev_hist(&self, f: &M3, old: &[f64], new: &mut [f64]) -> Result<M3> {
        if self.full {
            return fail("*HYPERFOAM stöds inte i hybridelement");
        }
        let (ud, uv) = self.energies(f)?;
        let (eta, um, sener, phi) = self.damage(ud, uv, old[0]);
        new[0] = um;
        new[1] = sener;
        new[2] = phi;
        Ok(t::scale(&self.h().tau_dev(f)?, eta))
    }
}

impl MaterialModel for Mullins {
    fn name(&self) -> &str {
        "MULLINS EFFECT"
    }

    fn n_state(&self) -> usize {
        3
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        hs::update_hyper_hist(self, input, state_old, state_new)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn output_names(&self) -> &[&'static str] {
        &["SENER", "DMENER"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state.get(1).copied().unwrap_or(0.0), state.get(2).copied().unwrap_or(0.0)]
    }

    fn hyperelastic(&self) -> Option<&dyn HyperModel> {
        Some(self)
    }
}
