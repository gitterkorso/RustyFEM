//! Hyperelasticity: Ogden (*HYPERELASTIC, OGDEN, N=n) and Arruda–Boyce
//! (*HYPERELASTIC, ARRUDA-BOYCE).
//!
//! Ogden (Ogden 1972) in the deviatoric principal stretches λ̄ = J^(−1/3) λ:
//!
//!   U = Σ_i 2μ_i/α_i² (λ̄1^α_i + λ̄2^α_i + λ̄3^α_i − 3) + Σ_k (1/D_k)(J − 1)^(2k)
//!
//! with data μ1, α1, ..., μn, αn, D1, ..., Dn (n ≤ 6) and μ0 = Σ μ_i. The
//! deviatoric Kirchhoff stress has the principal values
//! τ_a = Σ_i (2μ_i/α_i)(λ̄_a^α_i − ⅓ Σ_b λ̄_b^α_i) in the eigendirections of b̄.
//!
//! Arruda–Boyce (Arruda & Boyce 1993, the eight-chain model with five terms of
//! the Langevin expansion):
//!
//!   U = μ Σ_{i=1}^{5} C_i/λ_m^(2i−2) (Ī1^i − 3^i) + (1/D)((J² − 1)/2 − ln J)
//!
//! with C = 1/2, 1/20, 11/1050, 19/7000, 519/673750, data μ, λ_m, D and
//! μ0 = μ (1 + 3/(5λ_m²) + 99/(175λ_m⁴) + 513/(875λ_m⁶) + 42039/(67375λ_m⁸)).
//!
//! Both implement `HyperModel`; the material point's `update` (CPS/CPE/C3D
//! without their own volumetric treatment) uses the shared functions in
//! `hyperelastic_split.rs`. D = 0 means fully incompressible (hybrid elements).

use super::anisotropic::reduce_6x6;
use super::hyperelastic_split::{self as hs, HyperModel};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "hyperelasticity (Ogden, Arruda–Boyce)", consumes: &["HYPERELASTIC"], priority: 10, build: Some(build) }];

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
    /// D_k (Ogden: polynomial in J − 1; Arruda–Boyce: only D)
    pub d: Vec<f64>,
}

const AB_C: [f64; 5] = [0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0];

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    let ogden = kw.has("OGDEN");
    if !ogden && !kw.has("ARRUDA-BOYCE") {
        return Ok(None);
    }
    // feature area materials4: TEST DATA INPUT is fitted in hyperelastic_fit.rs
    if super::hyperelastic_fit::claims(kw) {
        return Ok(None);
    }
    if def.behavior("ELASTIC").is_some() {
        return kw.err("*HYPERELASTIC and *ELASTIC in the same material are not supported");
    }
    for p in ["TEST DATA INPUT", "USER", "TYPE", "PROPERTIES"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    if kw.has("MODULI") && kw.param_upper("MODULI").as_deref() != Some("LONG TERM") {
        return kw.err("MODULI is only supported as LONG TERM");
    }
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    let m = if ogden {
        let n = match kw.param("N") {
            Some(s) => s.trim().parse::<usize>().ok().filter(|&n| (1..=6).contains(&n)).ok_or_else(|| FemError(format!("{}: N={s} must be 1–6", kw.location())))?,
            None => 1,
        };
        let need = 3 * n;
        if v.len() < need {
            return kw.err(format!("OGDEN, N={n} requires {need} constants, {} given", v.len()));
        }
        let terms: Vec<(f64, f64)> = (0..n).map(|i| (v[2 * i], v[2 * i + 1])).collect();
        if terms.iter().any(|&(_, a)| a == 0.0) {
            return kw.err("OGDEN: α_i = 0 is not allowed");
        }
        HyperOgden { form: Form::Ogden(terms), d: v[2 * n..3 * n].to_vec() }
    } else {
        if v.len() < 3 {
            return kw.err(format!("ARRUDA-BOYCE requires 3 constants (μ, λm, D), {} given", v.len()));
        }
        if !(v[1] > 1.0) {
            return kw.err("ARRUDA-BOYCE: λm must be greater than 1");
        }
        HyperOgden { form: Form::ArrudaBoyce(v[0], v[1]), d: vec![v[2]] }
    };
    if m.d[0] < 0.0 || (m.d[0] == 0.0 && m.d.iter().any(|&x| x != 0.0)) {
        return kw.err("D1 < 0, or D1 = 0 with D2/D3 ≠ 0, is not supported (D1 = 0 means fully incompressible)");
    }
    if !(m.mu0() > 0.0) {
        return kw.err("the initial shear modulus must be positive");
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
            return fail("hyperelastic material: J ≤ 0 (the element has been turned inside out)");
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
                // principal stretches λ̄_a = √(eigenvalues of b̄)
                let lam: [f64; 3] = ev.map(|e| e.max(1e-300).sqrt());
                let mut ta = [0.0; 3];
                for &(m, al) in terms {
                    let p: [f64; 3] = lam.map(|l| l.powf(al));
                    let s = (p[0] + p[1] + p[2]) / 3.0;
                    for a in 0..3 {
                        ta[a] += 2.0 * m / al * (p[a] - s);
                    }
                }
                // τ = Σ τ_a n_a ⊗ n_a (column a in `vecs` is the eigenvector)
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

/// The material point's stress update for a `HyperModel` (Cauchy stress and
/// C^(τJ)/J, same convention as `hyperelastic.rs`).
pub fn update_hyper(h: &dyn HyperModel, input: &MatInput) -> Result<MatOutput> {
    let Some(f0) = input.defgrad else {
        return fail("*HYPERELASTIC requires NLGEOM=YES on *STEP");
    };
    if h.incompressible() {
        return fail("*HYPERELASTIC with D1 = 0 (fully incompressible) requires hybrid elements (C3D8H, CPE4H, ...)");
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
        StressMode::Uniaxial => return fail("*HYPERELASTIC OGDEN/ARRUDA-BOYCE is not supported in trusses"),
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
            return fail("*HYPERELASTIC with D1 = 0 (fully incompressible) requires hybrid elements (C3D8H, CPE4H, ...)");
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
