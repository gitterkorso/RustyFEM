//! Hyperelasticitet (*HYPERELASTIC) för gummiliknande material.
//!
//! Isotropa, kompressibla polynomformer i de deviatoriska invarianterna
//! Ī1 = tr b̄, Ī2 = ½(Ī1² − tr b̄²) med b̄ = J^(−2/3) F Fᵀ och volymdelen i J:
//!
//!   U = Σ C_ij (Ī1 − 3)^i (Ī2 − 3)^j + Σ_k (1/D_k)(J − 1)^(2k)
//!
//! - NEO HOOKE: C10, D1
//! - MOONEY-RIVLIN: C10, C01, D1
//! - YEOH: C10, C20, C30, D1, D2, D3
//! - REDUCED POLYNOMIAL, N=n: C10, C20, ..., Cn0, D1, ..., Dn
//! - POLYNOMIAL, N=n (n ≤ 6): C10, C01, C20, C11, C02, ..., D1, ..., Dn
//!   (för varje grad k: C(k,0), C(k−1,1), ..., C(0,k)).
//!
//! Kirchhoffspänningen är τ = 2 dev[(U1 + Ī1 U2) b̄ − U2 b̄²] + J U_J I
//! (U1 = ∂U/∂Ī1, U2 = ∂U/∂Ī2, U_J = ∂U/∂J) och Cauchyspänningen σ = τ/J.
//!
//! Gränssnittet mot elementen: vid NLGEOM ger elementet deformations-
//! gradienten (`MatInput::defgrad`), materialet returnerar Cauchyspänningen i
//! globala axlar och tangenten C^(τJ)/J (Jaumannderivatan av Kirchhoff-
//! spänningen delad med J – samma konvention som för UMAT), här beräknad
//! numeriskt med perturbation av F (Miehe 1996). Utan NLGEOM ger materialet
//! fel (hyperelastiska material kräver NLGEOM=YES).
//!
//! Element med selektivt reducerad integrering (CPE4, C3D8 m.fl.) ska skicka
//! F̄ = (J̄/J)^(1/n) F (n = 2 i planet för plan töjning, 3 i 3D) med J̄ =
//! elementets volymmedelvärde av J i referensläget (kontrollerat med ett CPE4-
//! och ett C3D8-element med föreskriven inhomogen deformation); elementets
//! inre kraft måste också formuleras med F̄, se rapporten för
//! NLGEOM-området.
//!
//! Plan spänning: sträckningen λ3 = F33 löses lokalt så att σ33 = 0.
//! Inkompressibla material (D1 = 0) kräver hybridelement (`elements/hybrid.rs`);
//! elementen med egen volymbehandling använder `HyperModel`
//! (`hyperelastic_split.rs`) i stället för `update`.

use super::anisotropic::reduce_6x6;
use super::hyperelastic_split::HyperModel;
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "hyperelasticitet (polynom)", consumes: &["HYPERELASTIC"], priority: 10, build: Some(build) }];

pub type M3 = [[f64; 3]; 3];

/// Polynommodell: C[i][j] för (Ī1 − 3)^i (Ī2 − 3)^j och D_k.
#[derive(Debug, Clone)]
pub struct Hyperelastic {
    pub kind: String,
    pub c: Vec<(usize, usize, f64)>,
    pub d: Vec<f64>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    if def.behavior("ELASTIC").is_some() {
        return kw.err("*HYPERELASTIC och *ELASTIC i samma material stöds inte");
    }
    if kw.has("TEST DATA INPUT") && !kw.has("OGDEN") && !kw.has("ARRUDA-BOYCE") && !kw.has("VAN DER WAALS") && !kw.has("MARLOW") {
        // byggs av hyperelastic_testdata.rs
        return Ok(None);
    }
    // område materials4: OGDEN/ARRUDA-BOYCE/VAN DER WAALS med TEST DATA INPUT (hyperelastic_fit.rs)
    if super::hyperelastic_fit::claims(kw) {
        return Ok(None);
    }
    for p in ["TEST DATA INPUT", "USER", "TYPE", "MODULI", "PROPERTIES", "BETA"] {
        if kw.has(p) {
            if p == "MODULI" && kw.param_upper("MODULI").as_deref() == Some("LONG TERM") {
                continue;
            }
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let n_param = || -> Result<usize> {
        match kw.param("N") {
            Some(s) => s.trim().parse::<usize>().ok().filter(|&n| (1..=6).contains(&n)).ok_or_else(|| crate::model::FemError(format!("{}: N={s} ska vara 1–6", kw.location()))),
            None => Ok(1),
        }
    };
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    let (kind, cij, nd): (&str, Vec<(usize, usize)>, usize) = if kw.has("NEO HOOKE") {
        ("NEO HOOKE", vec![(1, 0)], 1)
    } else if kw.has("MOONEY-RIVLIN") {
        ("MOONEY-RIVLIN", vec![(1, 0), (0, 1)], 1)
    } else if kw.has("YEOH") {
        ("YEOH", vec![(1, 0), (2, 0), (3, 0)], 3)
    } else if kw.has("REDUCED POLYNOMIAL") {
        let n = n_param()?;
        ("REDUCED POLYNOMIAL", (1..=n).map(|i| (i, 0)).collect(), n)
    } else if kw.has("POLYNOMIAL") {
        let n = n_param()?;
        let mut t = Vec::new();
        for k in 1..=n {
            for j in 0..=k {
                t.push((k - j, j));
            }
        }
        ("POLYNOMIAL", t, n)
    } else if kw.has("OGDEN") || kw.has("ARRUDA-BOYCE") || kw.has("VAN DER WAALS") || kw.has("MARLOW") {
        // byggs av hyperelastic_ogden.rs, hyperelastic_vdw.rs och hyperelastic_marlow.rs
        return Ok(None);
    } else {
        let what = kw.params.first().map(|p| p.0.clone()).unwrap_or_else(|| "(ingen)".into());
        return kw.err(format!("hyperelastisk form {what} stöds inte ännu (NEO HOOKE, MOONEY-RIVLIN, YEOH, POLYNOMIAL, REDUCED POLYNOMIAL)"));
    };
    let need = cij.len() + nd;
    if v.len() < need {
        return kw.err(format!("{kind} kräver {need} konstanter, {} angivna", v.len()));
    }
    if v.len() > need && v[need..].iter().any(|&x| x != 0.0) {
        return kw.err("temperaturberoende konstanter (flera datarader) stöds inte ännu");
    }
    let c: Vec<(usize, usize, f64)> = cij.iter().enumerate().map(|(k, &(i, j))| (i, j, v[k])).collect();
    let d: Vec<f64> = v[cij.len()..need].to_vec();
    if d[0] < 0.0 || (d[0] == 0.0 && d.iter().any(|&x| x != 0.0)) {
        return kw.err("D1 < 0, eller D1 = 0 med D2/D3 ≠ 0, stöds inte (D1 = 0 betyder helt inkompressibelt)");
    }
    let m = Hyperelastic { kind: kind.into(), c, d };
    let (mu, k) = m.initial_moduli();
    if !(mu > 0.0 && k > 0.0) {
        return kw.err("initial skjuvmodul 2(C10 + C01) och kompressionsmodul 2/D1 ska vara positiva");
    }
    Ok(Some(Arc::new(m)))
}

fn mm(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn det(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Symmetrisk tensor → Voigt (11, 22, 33, 12, 13, 23), tensorkomponenter.
fn voigt(a: &M3) -> [f64; 6] {
    [a[0][0], a[1][1], a[2][2], a[0][1], a[0][2], a[1][2]]
}

const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

impl Hyperelastic {
    /// (μ0, K0) vid odeformerat tillstånd.
    pub fn initial_moduli(&self) -> (f64, f64) {
        let mut mu = 0.0;
        for &(i, j, c) in &self.c {
            if i + j == 1 {
                mu += 2.0 * c;
            }
        }
        (mu, if self.d[0] > 0.0 { 2.0 / self.d[0] } else { f64::INFINITY })
    }

    /// (U1, U2, U_J)
    fn derivs(&self, i1: f64, i2: f64, j: f64) -> (f64, f64, f64) {
        let (a, b) = (i1 - 3.0, i2 - 3.0);
        let mut u1 = 0.0;
        let mut u2 = 0.0;
        for &(p, q, c) in &self.c {
            if p > 0 {
                u1 += c * p as f64 * a.powi(p as i32 - 1) * b.powi(q as i32);
            }
            if q > 0 {
                u2 += c * q as f64 * a.powi(p as i32) * b.powi(q as i32 - 1);
            }
        }
        let mut uj = 0.0;
        for (k, &dk) in self.d.iter().enumerate() {
            if dk != 0.0 {
                let n = 2 * (k + 1);
                uj += n as f64 / dk * (j - 1.0).powi(n as i32 - 1);
            }
        }
        (u1, u2, uj)
    }

    /// Töjningsenergi per referensvolym.
    pub fn energy(&self, f: &M3) -> f64 {
        let (i1, i2, j) = self.invariants(f);
        let (a, b) = (i1 - 3.0, i2 - 3.0);
        let mut u: f64 = self.c.iter().map(|&(p, q, c)| c * a.powi(p as i32) * b.powi(q as i32)).sum();
        for (k, &dk) in self.d.iter().enumerate() {
            if dk != 0.0 {
                u += (j - 1.0).powi(2 * (k as i32 + 1)) / dk;
            }
        }
        u
    }

    fn invariants(&self, f: &M3) -> (f64, f64, f64) {
        let j = det(f);
        let b = mm(f, &std::array::from_fn(|i| std::array::from_fn(|k| f[k][i])));
        let s = j.powf(-2.0 / 3.0);
        let bb: M3 = std::array::from_fn(|i| std::array::from_fn(|k| s * b[i][k]));
        let i1 = bb[0][0] + bb[1][1] + bb[2][2];
        let b2 = mm(&bb, &bb);
        let i2 = 0.5 * (i1 * i1 - (b2[0][0] + b2[1][1] + b2[2][2]));
        (i1, i2, j)
    }

    /// Kirchhoffspänningen τ(F).
    pub fn kirchhoff(&self, f: &M3) -> Result<M3> {
        let j = det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let ft: M3 = std::array::from_fn(|i| std::array::from_fn(|k| f[k][i]));
        let b = mm(f, &ft);
        let s = j.powf(-2.0 / 3.0);
        let bb: M3 = std::array::from_fn(|i| std::array::from_fn(|k| s * b[i][k]));
        let b2 = mm(&bb, &bb);
        let i1 = bb[0][0] + bb[1][1] + bb[2][2];
        let i2 = 0.5 * (i1 * i1 - (b2[0][0] + b2[1][1] + b2[2][2]));
        let (u1, u2, uj) = self.derivs(i1, i2, j);
        let mut a: M3 = std::array::from_fn(|i| std::array::from_fn(|k| (u1 + i1 * u2) * bb[i][k] - u2 * b2[i][k]));
        let tr = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
        for i in 0..3 {
            a[i][i] -= tr;
        }
        let mut tau: M3 = std::array::from_fn(|i| std::array::from_fn(|k| 2.0 * a[i][k]));
        for i in 0..3 {
            tau[i][i] += j * uj;
        }
        Ok(tau)
    }

    /// Cauchyspänning (Voigt) och tangenten C^(τJ)/J (6×6, ingenjörsskjuvning).
    pub fn cauchy_and_tangent(&self, f: &M3) -> Result<([f64; 6], [[f64; 6]; 6])> {
        let j = det(f);
        let tau = self.kirchhoff(f)?;
        let sig = voigt(&tau).map(|x| x / j);
        let eps = 1e-7;
        let mut c = [[0.0; 6]; 6];
        for (col, &(k, l)) in PAIRS.iter().enumerate() {
            // ΔF = ε/2 (e_k ⊗ e_l + e_l ⊗ e_k) F
            let mut df = [[0.0; 3]; 3];
            for m in 0..3 {
                df[k][m] += 0.5 * eps * f[l][m];
                df[l][m] += 0.5 * eps * f[k][m];
            }
            let fp: M3 = std::array::from_fn(|i| std::array::from_fn(|m| f[i][m] + df[i][m]));
            let tp = self.kirchhoff(&fp)?;
            let dv = voigt(&tp);
            let t0 = voigt(&tau);
            for r in 0..6 {
                c[r][col] = (dv[r] - t0[r]) / eps / j;
            }
        }
        for r in 0..6 {
            for s in 0..r {
                let m = 0.5 * (c[r][s] + c[s][r]);
                c[r][s] = m;
                c[s][r] = m;
            }
        }
        Ok((sig, c))
    }

    fn elastic_d6(&self) -> [[f64; 6]; 6] {
        let (mu, k) = self.initial_moduli();
        let lam = k - 2.0 * mu / 3.0;
        let mut d = [[0.0; 6]; 6];
        for i in 0..3 {
            for jj in 0..3 {
                d[i][jj] = lam;
            }
            d[i][i] = lam + 2.0 * mu;
            d[i + 3][i + 3] = mu;
        }
        d
    }
}

impl MaterialModel for Hyperelastic {
    fn name(&self) -> &str {
        "HYPERELASTIC"
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        let (mu, k) = self.initial_moduli();
        if !k.is_finite() {
            return Some((3.0 * mu, 0.5));
        }
        let e = 9.0 * k * mu / (3.0 * k + mu);
        Some((e, (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let Some(f0) = input.defgrad else {
            return fail("*HYPERELASTIC kräver NLGEOM=YES på *STEP");
        };
        if self.d[0] == 0.0 {
            return fail("*HYPERELASTIC med D1 = 0 (helt inkompressibelt) kräver hybridelement (C3D8H, CPE4H, ...)");
        }
        let mode = input.mode;
        let idx: &[usize] = match mode {
            StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
            StressMode::PlaneStrain | StressMode::Axisymmetric => &[0, 1, 2, 3],
            StressMode::PlaneStress => &[0, 1, 3],
            StressMode::Uniaxial => &[0],
        };
        let mut f = f0;
        match mode {
            StressMode::PlaneStress => {
                // λ3 så att σ33 = 0 (Newton på F33)
                f[0][2] = 0.0;
                f[1][2] = 0.0;
                f[2][0] = 0.0;
                f[2][1] = 0.0;
                let j2 = f[0][0] * f[1][1] - f[0][1] * f[1][0];
                f[2][2] = 1.0 / j2.max(1e-12);
                let mut ok = false;
                for _ in 0..50 {
                    let (s, c) = self.cauchy_and_tangent(&f)?;
                    let scale = s.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12 * self.initial_moduli().0);
                    if s[2].abs() <= 1e-12 * scale {
                        ok = true;
                        break;
                    }
                    // dσ33/d(ln λ3) ≈ c33 (ln-töjningsinkrement)
                    let dl = -s[2] / c[2][2];
                    f[2][2] *= dl.exp();
                }
                if !ok {
                    return fail("hyperelastiskt material: plan spänning konvergerade inte (σ33 ≠ 0)");
                }
            }
            StressMode::Uniaxial => {
                // λ2 = λ3 så att σ22 = σ33 = 0
                let l1 = f0[0][0];
                f = [[l1, 0.0, 0.0], [0.0, 1.0 / l1.sqrt(), 0.0], [0.0, 0.0, 1.0 / l1.sqrt()]];
                let mut ok = false;
                for _ in 0..50 {
                    let (s, c) = self.cauchy_and_tangent(&f)?;
                    if s[1].abs() <= 1e-12 * s[0].abs().max(1e-12 * self.initial_moduli().0) {
                        ok = true;
                        break;
                    }
                    let dl = -s[1] / (c[1][1] + c[1][2]);
                    f[1][1] *= dl.exp();
                    f[2][2] = f[1][1];
                }
                if !ok {
                    return fail("hyperelastiskt material: enaxlig spänning konvergerade inte");
                }
            }
            _ => {}
        }
        let (s, c) = self.cauchy_and_tangent(&f)?;
        let stress: Vec<f64> = idx.iter().map(|&i| s[i]).collect();
        let tangent = match mode {
            StressMode::PlaneStress | StressMode::Uniaxial => reduce_6x6(&c, mode),
            _ => idx.iter().flat_map(|&i| idx.iter().map(move |&k| c[i][k])).collect(),
        };
        Ok(MatOutput { stress, tangent })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        if self.d[0] == 0.0 {
            return fail("*HYPERELASTIC med D1 = 0 (helt inkompressibelt) kräver hybridelement (C3D8H, CPE4H, ...)");
        }
        Ok(reduce_6x6(&self.elastic_d6(), mode))
    }

    fn hyperelastic(&self) -> Option<&dyn HyperModel> {
        Some(self)
    }
}

impl HyperModel for Hyperelastic {
    fn mu0(&self) -> f64 {
        self.initial_moduli().0
    }

    fn k0(&self) -> f64 {
        self.initial_moduli().1
    }

    fn tau_dev(&self, f: &M3) -> Result<M3> {
        let j = det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let ft: M3 = std::array::from_fn(|i| std::array::from_fn(|k| f[k][i]));
        let b = mm(f, &ft);
        let s = j.powf(-2.0 / 3.0);
        let bb: M3 = std::array::from_fn(|i| std::array::from_fn(|k| s * b[i][k]));
        let b2 = mm(&bb, &bb);
        let i1 = bb[0][0] + bb[1][1] + bb[2][2];
        let i2 = 0.5 * (i1 * i1 - (b2[0][0] + b2[1][1] + b2[2][2]));
        let (u1, u2, _) = self.derivs(i1, i2, j);
        let mut a: M3 = std::array::from_fn(|i| std::array::from_fn(|k| (u1 + i1 * u2) * bb[i][k] - u2 * b2[i][k]));
        let tr = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
        for i in 0..3 {
            a[i][i] -= tr;
        }
        Ok(std::array::from_fn(|i| std::array::from_fn(|k| 2.0 * a[i][k])))
    }

    fn energy(&self, f: &M3) -> Option<(f64, f64)> {
        let (i1, i2, j) = self.invariants(f);
        let ud = self.c.iter().map(|&(p, q, c)| c * (i1 - 3.0).powi(p as i32) * (i2 - 3.0).powi(q as i32)).sum();
        let uv = self.d.iter().enumerate().filter(|(_, &dk)| dk != 0.0).map(|(k, &dk)| (j - 1.0).powi(2 * (k as i32 + 1)) / dk).sum();
        Some((ud, uv))
    }

    fn vol(&self, j: f64) -> (f64, f64) {
        let (mut u1, mut u2) = (0.0, 0.0);
        for (k, &dk) in self.d.iter().enumerate() {
            if dk != 0.0 {
                let n = 2 * (k + 1);
                u1 += n as f64 / dk * (j - 1.0).powi(n as i32 - 1);
                u2 += (n * (n - 1)) as f64 / dk * (j - 1.0).powi(n as i32 - 2);
            }
        }
        (u1, u2)
    }
}
