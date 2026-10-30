//! Uppdelad hyperelasticitet: gränssnittet som element med egen behandling av
//! volymdelen använder (F-bar-elementen CPE4/C3D8 vid NLGEOM, plan spänning
//! CPS4/CPS3 och hybridelementen C3D8H, CPE4H, ...).
//!
//! En isotrop hyperelastisk modell skrivs som
//!
//!   U(F) = U_dev(b̄) + U_vol(J),   b̄ = J^(−2/3) F Fᵀ,
//!
//! så att Kirchhoffspänningen delas i τ = τ_dev(b̄) + J U_vol′(J) I
//! (Simo & Taylor 1991; Holzapfel 2000, kap. 6). `HyperModel` ger de två
//! delarna var för sig; hybridelementen ersätter J U_vol′ med det oberoende
//! trycket, F-bar-elementen beräknar τ med F̄.
//!
//! Tangenter räknas med perturbation av F som i `hyperelastic.rs` (Miehe 1996):
//! ΔF = ε/2 (e_k ⊗ e_l + e_l ⊗ e_k) F ger Jaumann-tangenten C^(τJ) kolumn för kolumn.

use super::MaterialFactory;
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, Result};

/// Inga egna fabriker här (Ogden och Arruda–Boyce ligger i `hyperelastic_ogden.rs`).
pub static FACTORIES: &[MaterialFactory] = &[];

/// Voigt-ordning (11, 22, 33, 12, 13, 23).
pub const PAIRS: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];

/// Isotrop hyperelastisk modell uppdelad i deviatorisk och volymetrisk del.
pub trait HyperModel: Send + Sync {
    /// Initial skjuvmodul μ0 (t.ex. 2(C10 + C01)).
    fn mu0(&self) -> f64;
    /// Initial kompressionsmodul K0 = U_vol″(1); `f64::INFINITY` om inkompressibelt (D1 = 0).
    fn k0(&self) -> f64;
    /// Deviatorisk Kirchhoffspänning τ_dev (spårfri, beror bara på b̄ = J^(−2/3) F Fᵀ).
    fn tau_dev(&self, f: &M3) -> Result<M3>;
    /// Volymdelen: (U_vol′(J), U_vol″(J)); (0, 0) om inkompressibelt.
    fn vol(&self, j: f64) -> (f64, f64);
    /// Helt inkompressibelt (D1 = 0): kräver hybridelement.
    fn incompressible(&self) -> bool {
        !self.k0().is_finite()
    }
    /// Hela Kirchhoffspänningen för modeller som inte delas i deviatorisk och
    /// volymetrisk del (*HYPERFOAM, `materials/hyperfoam.rs`). Standard: `None`,
    /// dvs τ = τ_dev + J U_vol′ I.
    fn kirchhoff_full(&self, _f: &M3) -> Option<Result<M3>> {
        None
    }
    /// Töjningsenergin per referensvolym uppdelad i (U_dev, U_vol); modeller
    /// utan uppdelning ger (U, 0). Används av *MULLINS EFFECT och SENER.
    /// Standard: `None` (okänd).
    fn energy(&self, _f: &M3) -> Option<(f64, f64)> {
        None
    }
    /// Antal historievariabler per integrationspunkt (*MULLINS EFFECT:
    /// största energin, SENER, DMENER). De ligger direkt efter spänningen i
    /// elementtillståndet, där `MaterialModel::n_state` säger. Standard: 0.
    fn n_hist(&self) -> usize {
        0
    }
    /// Kirchhoffspänningen med historia: `old` vid inkrementets början, `new`
    /// (lika lång) skrivs med historien vid F. Standard: `kirchhoff`.
    fn kirchhoff_hist(&self, f: &M3, _old: &[f64], _new: &mut [f64]) -> Result<M3> {
        if let Some(t) = self.kirchhoff_full(f) {
            return t;
        }
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let mut tau = self.tau_dev(f)?;
        let (up, _) = self.vol(j);
        for i in 0..3 {
            tau[i][i] += j * up;
        }
        Ok(tau)
    }
    /// Deviatorisk Kirchhoffspänning med historia (hybridelement). Standard: `tau_dev`.
    fn tau_dev_hist(&self, f: &M3, _old: &[f64], _new: &mut [f64]) -> Result<M3> {
        self.tau_dev(f)
    }
}

/// Kirchhoffspänningen med historia (se `HyperModel::kirchhoff_hist`); utan
/// historia samma som `kirchhoff`. `old` får vara kortare än `n_hist` (start: nollor).
pub fn kirchhoff_hist(h: &dyn HyperModel, f: &M3, old: &[f64], new: &mut [f64]) -> Result<M3> {
    let n = h.n_hist();
    if n == 0 {
        return kirchhoff(h, f);
    }
    let z;
    let old = if old.len() >= n {
        &old[..n]
    } else {
        z = vec![0.0; n];
        &z[..]
    };
    h.kirchhoff_hist(f, old, &mut new[..n])
}

/// Hela Kirchhoffspänningen τ = τ_dev + J U_vol′(J) I.
pub fn kirchhoff(h: &dyn HyperModel, f: &M3) -> Result<M3> {
    if let Some(t) = h.kirchhoff_full(f) {
        return t;
    }
    let j = t::det(f);
    if !(j > 0.0) {
        return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
    }
    let mut tau = h.tau_dev(f)?;
    let (up, _) = h.vol(j);
    for i in 0..3 {
        tau[i][i] += j * up;
    }
    Ok(tau)
}

/// Symmetrisk tensor → Voigt (tensorkomponenter).
pub fn voigt(a: &M3) -> [f64; 6] {
    [a[0][0], a[1][1], a[2][2], a[0][1], a[0][2], a[1][2]]
}

/// Jaumann-tangenten C^(τJ) (6×6, ingenjörsskjuvning i kolumnerna) för
/// spänningsfunktionen `tau`: Δτ = C^(τJ) : ΔD + ΔW τ − τ ΔW.
pub fn jaumann_tangent(tau_fn: &dyn Fn(&M3) -> Result<M3>, f: &M3) -> Result<[[f64; 6]; 6]> {
    let scale = f.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())).max(1.0);
    let eps = 1e-7 * scale.min(10.0);
    let mut c = [[0.0; 6]; 6];
    for (col, &(k, l)) in PAIRS.iter().enumerate() {
        let mut fp = *f;
        let mut fm = *f;
        for m in 0..3 {
            fp[k][m] += 0.5 * eps * f[l][m];
            fp[l][m] += 0.5 * eps * f[k][m];
            fm[k][m] -= 0.5 * eps * f[l][m];
            fm[l][m] -= 0.5 * eps * f[k][m];
        }
        let tp = voigt(&tau_fn(&fp)?);
        let tm = voigt(&tau_fn(&fm)?);
        for r in 0..6 {
            c[r][col] = (tp[r] - tm[r]) / (2.0 * eps);
        }
    }
    for r in 0..6 {
        for s in 0..r {
            let m = 0.5 * (c[r][s] + c[s][r]);
            c[r][s] = m;
            c[s][r] = m;
        }
    }
    Ok(c)
}

/// Sträckningen λ3 = F33 för plan spänning: τ33(F med F33 = λ3) = 0.
/// `f` har F13 = F23 = F31 = F32 = 0.
pub fn plane_stress_stretch(h: &dyn HyperModel, f: &M3) -> Result<f64> {
    let mut g = *f;
    let j2 = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    if !(j2 > 0.0) {
        return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
    }
    let mut l3 = if g[2][2] > 0.0 { g[2][2] } else { 1.0 / j2 };
    let scale = h.mu0().max(1e-300);
    for _ in 0..60 {
        g[2][2] = l3;
        let tau = kirchhoff(h, &g)?;
        if tau[2][2].abs() <= 1e-13 * scale * (1.0 + tau.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())) / scale) {
            return Ok(l3);
        }
        // dτ33/d(ln λ3) med central differens
        let e = 1e-6;
        g[2][2] = l3 * (1.0 + e);
        let tp = kirchhoff(h, &g)?[2][2];
        g[2][2] = l3 * (1.0 - e);
        let tm = kirchhoff(h, &g)?[2][2];
        let d = (tp - tm) / (2.0 * e);
        if !(d.abs() > 0.0) {
            break;
        }
        let mut dl = -tau[2][2] / d;
        dl = dl.clamp(-0.5, 0.5);
        l3 *= dl.exp();
    }
    fail("hyperelastiskt material: plan spänning konvergerade inte (σ33 ≠ 0)")
}

/// Som `plane_stress_stretch`, men med historia (`old` vid inkrementets början).
pub fn plane_stress_stretch_hist(h: &dyn HyperModel, f: &M3, old: &[f64]) -> Result<f64> {
    if h.n_hist() == 0 {
        return plane_stress_stretch(h, f);
    }
    let mut scratch = vec![0.0; h.n_hist()];
    let mut g = *f;
    let j2 = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    if !(j2 > 0.0) {
        return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
    }
    let mut l3 = if g[2][2] > 0.0 { g[2][2] } else { 1.0 / j2 };
    let scale = h.mu0().max(1e-300);
    for _ in 0..60 {
        g[2][2] = l3;
        let tau = kirchhoff_hist(h, &g, old, &mut scratch)?;
        if tau[2][2].abs() <= 1e-13 * scale * (1.0 + tau.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs())) / scale) {
            return Ok(l3);
        }
        let e = 1e-6;
        g[2][2] = l3 * (1.0 + e);
        let tp = kirchhoff_hist(h, &g, old, &mut scratch)?[2][2];
        g[2][2] = l3 * (1.0 - e);
        let tm = kirchhoff_hist(h, &g, old, &mut scratch)?[2][2];
        let d = (tp - tm) / (2.0 * e);
        if !(d.abs() > 0.0) {
            break;
        }
        let dl = (-tau[2][2] / d).clamp(-0.5, 0.5);
        l3 *= dl.exp();
    }
    fail("hyperelastiskt material: plan spänning konvergerade inte (σ33 ≠ 0)")
}

/// Materialpunktens spänningsuppdatering för en `HyperModel` med historia
/// (Cauchyspänning och C^(τJ)/J som `hyperelastic_ogden::update_hyper`).
/// Tillståndet `state_old`/`state_new` är historievariablerna (`n_hist`).
pub fn update_hyper_hist(h: &dyn HyperModel, input: &super::MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<super::MatOutput> {
    use super::StressMode;
    let Some(f0) = input.defgrad else {
        return fail("hyperelastiska material kräver NLGEOM=YES på *STEP");
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
            f[2][2] = plane_stress_stretch_hist(h, &f, state_old)?;
        }
        StressMode::Uniaxial => return fail("hyperelastiska material med historia eller *HYPERFOAM stöds inte i stänger"),
        _ => {}
    }
    let j = t::det(&f);
    let tau = kirchhoff_hist(h, &f, state_old, state_new)?;
    let s = voigt(&tau).map(|x| x / j);
    let nh = h.n_hist();
    let mut c = jaumann_tangent(&|g: &M3| kirchhoff_hist(h, g, state_old, &mut vec![0.0; nh]), &f)?;
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
        StressMode::PlaneStress => super::anisotropic::reduce_6x6(&c, mode),
        _ => idx.iter().flat_map(|&i| idx.iter().map(move |&k| c[i][k])).collect(),
    };
    Ok(super::MatOutput { stress, tangent })
}
