//! Deformationsplasticitet (*DEFORMATION PLASTICITY): Ramberg–Osgood som
//! ickelinjärt elastiskt material (totalformulering), för implicit analys.
//!
//! Data: E, ν, σ0, n, α. Sambandet mellan total töjning och spänning är
//!
//!   E ε = (1 + ν) S − (1 − 2ν) p I + (3/2) α (q/σ0)^(n−1) S,
//!
//! med deviatorn S, trycket p = −tr σ/3 och von Mises-spänningen q. Volymdelen
//! är linjärt elastisk, p = −K tr ε. Deviatordelen ger en skalär ekvation för q
//! ur den ekvivalenta deviatoriska töjningen ē = √(⅔ e:e):
//!
//!   ē = q/(3G) + (α/E) q (q/σ0)^(n−1),
//!
//! som löses med Newton (konvex och växande, startvärdet 3Gē ligger över
//! roten), varefter S = (2q/(3ē)) e. Avlastning följer samma kurva
//! (ingen plastisk historia). Tangenten är analytisk.
//!
//! Utdata: PE = (3/2)(α/E)(q/σ0)^(n−1) S (ingenjörsskjuvning) och
//! PEEQ = (α/E) q (q/σ0)^(n−1). Plan spänning och stänger via
//! `plasticity::update_reduced`.

use super::plasticity::{iso_d6, pe_labels, update_reduced, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "deformationsplasticitet (Ramberg–Osgood)", consumes: &["DEFORMATION PLASTICITY"], priority: 10, build: Some(build) }];

/// Tillstånd: [3D-töjning (6), PEEQ, PE (6)].
const EPS3: usize = 0;
const PEEQ: usize = 6;
const PE: usize = 7;
const NSTATE: usize = 13;

pub struct RambergOsgood {
    e: f64,
    nu: f64,
    s0: f64,
    n: f64,
    alpha: f64,
    g: f64,
    k: f64,
    ce: M6,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("DEFORMATION PLASTICITY") else { return Ok(None) };
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperatur- eller fältberoende deformationsplasticitet stöds inte ännu");
    }
    if def.behavior("ELASTIC").is_some() || def.behavior("PLASTIC").is_some() {
        return kw.err("*DEFORMATION PLASTICITY ska inte kombineras med *ELASTIC eller *PLASTIC (elasticiteten ingår)");
    }
    let v = kw.first_floats()?;
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let (e, nu, s0, n, alpha) = (g(0), g(1), g(2), g(3), g(4));
    if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) || !(s0 > 0.0) || !(n >= 1.0) || !(alpha >= 0.0) {
        return kw.err("data ska vara E > 0, −1 < ν < 0,5, σ0 > 0, n ≥ 1 och α ≥ 0");
    }
    let gm = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    Ok(Some(Arc::new(RambergOsgood { e, nu, s0, n, alpha, g: gm, k, ce: iso_d6(e, nu) })))
}

impl RambergOsgood {
    /// q ur ē och dq/dē.
    fn solve_q(&self, eb: f64) -> Result<(f64, f64)> {
        let f = |q: f64| q / (3.0 * self.g) + self.alpha / self.e * q * (q / self.s0).powf(self.n - 1.0) - eb;
        let df = |q: f64| 1.0 / (3.0 * self.g) + self.alpha / self.e * self.n * (q / self.s0).powf(self.n - 1.0);
        if eb <= 0.0 {
            return Ok((0.0, 1.0 / df(0.0)));
        }
        let mut q = 3.0 * self.g * eb;
        for _ in 0..200 {
            let r = f(q);
            let dq = r / df(q);
            q -= dq;
            if dq.abs() <= 1e-15 * q.abs().max(self.s0 * 1e-12) {
                return Ok((q, 1.0 / df(q)));
            }
        }
        fail(format!("*DEFORMATION PLASTICITY: ekvationen för q konvergerade inte (ē = {eb:e})"))
    }

    /// 3D: spänning, tangent och plastisk töjning (PEEQ, PE) ur total töjning.
    fn update3d(&self, eps: &V6) -> Result<(V6, M6, f64, V6)> {
        let ev = eps[0] + eps[1] + eps[2];
        // deviatorisk töjning med tensorskjuvning
        let e: V6 = [eps[0] - ev / 3.0, eps[1] - ev / 3.0, eps[2] - ev / 3.0, 0.5 * eps[3], 0.5 * eps[4], 0.5 * eps[5]];
        let ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + 2.0 * (e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
        let eb = (2.0 / 3.0 * ee).max(0.0).sqrt();
        let (q, dq) = self.solve_q(eb)?;
        // sekantmodul μs = q/(3ē) och dess derivata
        let (mu, dmu) = if eb > 1e-300 { (q / (3.0 * eb), (dq * eb - q) / (3.0 * eb * eb)) } else { (self.g, 0.0) };
        let p = -self.k * ev;
        let mut sig = [0.0; 6];
        for i in 0..3 {
            sig[i] = 2.0 * mu * e[i] - p;
        }
        for i in 3..6 {
            sig[i] = 2.0 * mu * e[i];
        }
        let mut c = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = self.k + 2.0 * mu * (if i == j { 1.0 } else { 0.0 } - 1.0 / 3.0);
            }
        }
        for i in 3..6 {
            c[i][i] = mu;
        }
        if eb > 1e-300 {
            // dσ += 2 e dμs, dμs = μs' dē, dē = (2/3)(e:dε)/ē
            let b = 2.0 * dmu * 2.0 / (3.0 * eb);
            for i in 0..6 {
                for j in 0..6 {
                    c[i][j] += b * e[i] * e[j];
                }
            }
        }
        let fac = self.alpha / self.e * (q / self.s0).powf(self.n - 1.0);
        let peeq = fac * q;
        let mut pe = [0.0; 6];
        for i in 0..6 {
            let s = sig[i] + if i < 3 { p } else { 0.0 };
            pe[i] = 1.5 * fac * s * if i < 3 { 1.0 } else { 2.0 };
        }
        Ok((sig, c, peeq, pe))
    }
}

impl MaterialModel for RambergOsgood {
    fn name(&self) -> &str {
        "DEFORMATION PLASTICITY"
    }

    fn n_state(&self) -> usize {
        NSTATE
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut last = (0.0, [0.0; 6]);
        let mut f = |e: &V6| -> Result<(V6, M6)> {
            let (s, c, peeq, pe) = self.update3d(e)?;
            last = (peeq, pe);
            Ok((s, c))
        };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        state_new[PEEQ] = last.0;
        state_new[PE..PE + 6].copy_from_slice(&last.1);
        Ok(out)
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        super::plasticity::rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
        super::plasticity::rotate_v6(r, &mut state[PE..PE + 6], true);
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[PEEQ]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[PEEQ]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[PE + i]));
        if mode == StressMode::PlaneStress {
            // i plan spänning redovisas PE33 = 0 och PEEQ = √(⅔ PE:PE)
            // av komponenterna i planet (√(5/6) ε̄ vid enaxligt drag)
            v[3] = 0.0;
            let (a, b, c) = (state[PE], state[PE + 1], state[PE + 3]);
            v[0] = (2.0 / 3.0 * (a * a + b * b + 0.5 * c * c)).sqrt();
        }
        v
    }
}
