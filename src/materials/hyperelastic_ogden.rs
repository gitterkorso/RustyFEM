//! Hyperelasticitet: Ogden (*HYPERELASTIC, OGDEN, N=n) och Arruda–Boyce
//! (*HYPERELASTIC, ARRUDA-BOYCE).
//!
//! Ogden (Ogden 1972) i de deviatoriska huvudsträckningarna λ̄ = J^(−1/3) λ:
//!
//!   U = Σ_i 2μ_i/α_i² (λ̄1^α_i + λ̄2^α_i + λ̄3^α_i − 3) + Σ_k (1/D_k)(J − 1)^(2k)
//!
//! med data μ1, α1, ..., μn, αn, D1, ..., Dn (n ≤ 6) och μ0 = Σ μ_i. Den
//! deviatoriska Kirchhoffspänningen har huvudvärdena
//! τ_a = Σ_i (2μ_i/α_i)(λ̄_a^α_i − ⅓ Σ_b λ̄_b^α_i) i b̄:s egenriktningar.
//!
//! Arruda–Boyce (Arruda & Boyce 1993, åttakedjemodellen med fem termer i
//! Langevin-utvecklingen):
//!
//!   U = μ Σ_{i=1}^{5} C_i/λ_m^(2i−2) (Ī1^i − 3^i) + (1/D)((J² − 1)/2 − ln J)
//!
//! med C = 1/2, 1/20, 11/1050, 19/7000, 519/673750, data μ, λ_m, D och
//! μ0 = μ (1 + 3/(5λ_m²) + 99/(175λ_m⁴) + 513/(875λ_m⁶) + 42039/(67375λ_m⁸)).
//!
//! Båda implementerar `HyperModel`; materialpunktens `update` (CPS/CPE/C3D
//! utan egen volymbehandling) använder de gemensamma funktionerna i
//! `hyperelastic_split.rs`. D = 0 betyder helt inkompressibelt (hybridelement).

use super::anisotropic::reduce_6x6;
use super::hyperelastic_split::{self as hs, HyperModel};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "hyperelasticitet (Ogden, Arruda–Boyce)", consumes: &["HYPERELASTIC"], priority: 10, build: Some(build) }];

#[derive(Debug, Clone)]
pub enum Form {
    /// (μ_i, α_i)
    Ogden(Vec<(f64, f64)>),
    /// (μ, λ_m)
    ArrudaBoyce(f64, f64),
}

#[derive(Debug, Clone)]
pub struct HyperOgden {
    pub form: Form,
    /// D_k (Ogden: polynom i J − 1; Arruda–Boyce: bara D)
    pub d: Vec<f64>,
}

const AB_C: [f64; 5] = [0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0];

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    let ogden = kw.has("OGDEN");
    if !ogden && !kw.has("ARRUDA-BOYCE") {
        return Ok(None);
    }
    // område materials4: TEST DATA INPUT anpassas i hyperelastic_fit.rs
    if super::hyperelastic_fit::claims(kw) {
        return Ok(None);
    }
    if def.behavior("ELASTIC").is_some() {
        return kw.err("*HYPERELASTIC och *ELASTIC i samma material stöds inte");
    }
    for p in ["TEST DATA INPUT", "USER", "TYPE", "PROPERTIES"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    if kw.has("MODULI") && kw.param_upper("MODULI").as_deref() != Some("LONG TERM") {
        return kw.err("MODULI stöds bara som LONG TERM");
    }
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    let m = if ogden {
        let n = match kw.param("N") {
            Some(s) => s.trim().parse::<usize>().ok().filter(|&n| (1..=6).contains(&n)).ok_or_else(|| FemError(format!("{}: N={s} ska vara 1–6", kw.location())))?,
            None => 1,
        };
        let need = 3 * n;
        if v.len() < need {
            return kw.err(format!("OGDEN, N={n} kräver {need} konstanter, {} angivna", v.len()));
        }
        let terms: Vec<(f64, f64)> = (0..n).map(|i| (v[2 * i], v[2 * i + 1])).collect();
        if terms.iter().any(|&(_, a)| a == 0.0) {
            return kw.err("OGDEN: α_i = 0 är inte tillåtet");
        }
        HyperOgden { form: Form::Ogden(terms), d: v[2 * n..3 * n].to_vec() }
    } else {
        if v.len() < 3 {
            return kw.err(format!("ARRUDA-BOYCE kräver 3 konstanter (μ, λm, D), {} angivna", v.len()));
        }
        if !(v[1] > 1.0) {
            return kw.err("ARRUDA-BOYCE: λm ska vara större än 1");
        }
        HyperOgden { form: Form::ArrudaBoyce(v[0], v[1]), d: vec![v[2]] }
    };
    if m.d[0] < 0.0 || (m.d[0] == 0.0 && m.d.iter().any(|&x| x != 0.0)) {
        return kw.err("D1 < 0, eller D1 = 0 med D2/D3 ≠ 0, stöds inte (D1 = 0 betyder helt inkompressibelt)");
    }
    if !(m.mu0() > 0.0) {
        return kw.err("den initiala skjuvmodulen ska vara positiv");
    }
    Ok(Some(Arc::new(m)))
}

impl HyperModel for HyperOgden {
    fn mu0(&self) -> f64 {
        match &self.form {
            Form::Ogden(t) => t.iter().map(|&(m, _)| m).sum(),
            Form::ArrudaBoyce(mu, lm) => {
                let l2 = 1.0 / (lm * lm);
                mu * (1.0 + 3.0 / 5.0 * l2 + 99.0 / 175.0 * l2 * l2 + 513.0 / 875.0 * l2.powi(3) + 42039.0 / 67375.0 * l2.powi(4))
            }
        }
    }

    fn k0(&self) -> f64 {
        if self.d[0] > 0.0 {
            2.0 / self.d[0]
        } else {
            f64::INFINITY
        }
    }

    fn tau_dev(&self, f: &M3) -> Result<M3> {
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let bb = t::scale(&t::mul_bt(f, f), j.powf(-2.0 / 3.0));
        match &self.form {
            Form::ArrudaBoyce(mu, lm) => {
                let i1 = bb[0][0] + bb[1][1] + bb[2][2];
                let mut u1 = 0.0;
                for (k, c) in AB_C.iter().enumerate() {
                    let i = k + 1;
                    u1 += mu * c * i as f64 * i1.powi(i as i32 - 1) / lm.powi(2 * i as i32 - 2);
                }
                let mut a = t::scale(&bb, 2.0 * u1);
                let tr = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
                for i in 0..3 {
                    a[i][i] -= tr;
                }
                Ok(a)
            }
            Form::Ogden(terms) => {
                let (ev, vecs) = t::eig_sym(&bb);
                // huvudsträckningar λ̄_a = √(egenvärden av b̄)
                let lam: [f64; 3] = ev.map(|e| e.max(1e-300).sqrt());
                let mut ta = [0.0; 3];
                for &(m, al) in terms {
                    let p: [f64; 3] = lam.map(|l| l.powf(al));
                    let s = (p[0] + p[1] + p[2]) / 3.0;
                    for a in 0..3 {
                        ta[a] += 2.0 * m / al * (p[a] - s);
                    }
                }
                // τ = Σ τ_a n_a ⊗ n_a (kolumn a i `vecs` är egenvektorn)
                Ok(std::array::from_fn(|i| std::array::from_fn(|k| (0..3).map(|a| ta[a] * vecs[i][a] * vecs[k][a]).sum())))
            }
        }
    }

    fn energy(&self, f: &M3) -> Option<(f64, f64)> {
        let j = t::det(f);
        let bb = t::scale(&t::mul_bt(f, f), j.powf(-2.0 / 3.0));
        match &self.form {
            Form::ArrudaBoyce(mu, lm) => {
                let i1 = bb[0][0] + bb[1][1] + bb[2][2];
                let ud = AB_C.iter().enumerate().map(|(k, c)| mu * c / lm.powi(2 * k as i32) * (i1.powi(k as i32 + 1) - 3f64.powi(k as i32 + 1))).sum();
                let uv = if self.d[0] > 0.0 { ((j * j - 1.0) / 2.0 - j.ln()) / self.d[0] } else { 0.0 };
                Some((ud, uv))
            }
            Form::Ogden(terms) => {
                let (ev, _) = t::eig_sym(&bb);
                let ud = terms.iter().map(|&(m, al)| 2.0 * m / (al * al) * (ev.iter().map(|e| e.max(1e-300).powf(0.5 * al)).sum::<f64>() - 3.0)).sum();
                let uv = self.d.iter().enumerate().filter(|(_, &dk)| dk > 0.0).map(|(k, &dk)| (j - 1.0).powi(2 * (k as i32 + 1)) / dk).sum();
                Some((ud, uv))
            }
        }
    }

    fn vol(&self, j: f64) -> (f64, f64) {
        match &self.form {
            Form::ArrudaBoyce(..) => {
                if self.d[0] > 0.0 {
                    ((j - 1.0 / j) / self.d[0], (1.0 + 1.0 / (j * j)) / self.d[0])
                } else {
                    (0.0, 0.0)
                }
            }
            Form::Ogden(_) => {
                let (mut u1, mut u2) = (0.0, 0.0);
                for (k, &dk) in self.d.iter().enumerate() {
                    if dk > 0.0 {
                        let n = 2 * (k + 1);
                        u1 += n as f64 / dk * (j - 1.0).powi(n as i32 - 1);
                        u2 += (n * (n - 1)) as f64 / dk * (j - 1.0).powi(n as i32 - 2);
                    }
                }
                (u1, u2)
            }
        }
    }
}

/// Materialpunktens spänningsuppdatering för en `HyperModel` (Cauchyspänning och
/// C^(τJ)/J, samma konvention som `hyperelastic.rs`).
pub fn update_hyper(h: &dyn HyperModel, input: &MatInput) -> Result<MatOutput> {
    let Some(f0) = input.defgrad else {
        return fail("*HYPERELASTIC kräver NLGEOM=YES på *STEP");
    };
    if h.incompressible() {
        return fail("*HYPERELASTIC med D1 = 0 (helt inkompressibelt) kräver hybridelement (C3D8H, CPE4H, ...)");
    }
    let mode = input.mode;
    let mut f = f0;
    match mode {
        StressMode::PlaneStress => {
            f[0][2] = 0.0;
            f[1][2] = 0.0;
            f[2][0] = 0.0;
            f[2][1] = 0.0;
            f[2][2] = 1.0;
            f[2][2] = hs::plane_stress_stretch(h, &f)?;
        }
        StressMode::Uniaxial => return fail("*HYPERELASTIC OGDEN/ARRUDA-BOYCE stöds inte i stänger"),
        _ => {}
    }
    let j = t::det(&f);
    let tau = hs::kirchhoff(h, &f)?;
    let s = hs::voigt(&tau).map(|x| x / j);
    let mut c = hs::jaumann_tangent(&|g: &M3| hs::kirchhoff(h, g), &f)?;
    for r in c.iter_mut() {
        for v in r.iter_mut() {
            *v /= j;
        }
    }
    let idx: &[usize] = match mode {
        StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
        StressMode::PlaneStrain | StressMode::Axisymmetric => &[0, 1, 2, 3],
        _ => &[0, 1, 3],
    };
    let stress: Vec<f64> = idx.iter().map(|&i| s[i]).collect();
    let tangent = match mode {
        StressMode::PlaneStress => reduce_6x6(&c, mode),
        _ => idx.iter().flat_map(|&i| idx.iter().map(move |&k| c[i][k])).collect(),
    };
    Ok(MatOutput { stress, tangent })
}

impl MaterialModel for HyperOgden {
    fn name(&self) -> &str {
        "HYPERELASTIC"
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        let (mu, k) = (self.mu0(), self.k0());
        if !k.is_finite() {
            return Some((3.0 * mu, 0.5));
        }
        Some((9.0 * k * mu / (3.0 * k + mu), (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        update_hyper(self, input)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        if self.incompressible() {
            return fail("*HYPERELASTIC med D1 = 0 (helt inkompressibelt) kräver hybridelement (C3D8H, CPE4H, ...)");
        }
        let (mu, k) = (self.mu0(), self.k0());
        let lam = k - 2.0 * mu / 3.0;
        let mut d = [[0.0; 6]; 6];
        for i in 0..3 {
            for jj in 0..3 {
                d[i][jj] = lam;
            }
            d[i][i] = lam + 2.0 * mu;
            d[i + 3][i + 3] = mu;
        }
        Ok(reduce_6x6(&d, mode))
    }

    fn hyperelastic(&self) -> Option<&dyn HyperModel> {
        Some(self)
    }
}
