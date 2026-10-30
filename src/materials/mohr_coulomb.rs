//! Mohr–Coulomb plasticity (*MOHR COULOMB, *MOHR COULOMB HARDENING) with
//! isotropic linear elasticity.
//!
//! Yield condition (p = −tr σ/3, q = von Mises stress, Θ the deviatoric angle with
//! cos 3Θ = (r/q)³, Θ = 0 in uniaxial tension and π/3 in uniaxial compression):
//!
//!   F = R_mc(Θ, φ) q − p tan φ − c(ε̄p) = 0,
//!   R_mc(Θ, φ) = sin(Θ + π/3)/(√3 cos φ) + cos(Θ + π/3) tan φ/3,
//!
//! i.e. (σ1 − σ3) + (σ1 + σ3) sin φ = 2c cos φ. The angle is limited to
//! δ ≤ Θ ≤ π/3 − δ with 1 − cos 3δ = 1.4·10⁻⁷ (δ ≈ 0.0101°), which shows up as a
//! relative difference of 1.8·10⁻⁴ in the yield surface right at the corners (triaxial compression and
//! tension).
//!
//! The flow potential is the smooth hyperbolic surface of Menétrey–Willam
//!
//!   G = √((ε c|₀ tan ψ)² + (R_mw(Θ, e) q)²) − p tan ψ,
//!   R_mw = [4(1−e²)cos²Θ + (2e−1)²] / [2(1−e²)cos Θ + (2e−1)√(4(1−e²)cos²Θ + 5e² − 4e)] · R_mc(π/3, φ),
//!
//! with meridional eccentricity ε = 0.1 (fixed, no input parameter) and
//! deviatoric eccentricity e (DEVIATORIC ECCENTRICITY, default
//! (3 − sin φ)/(3 + sin φ)). G is smooth everywhere (also at the apex and the corners), so
//! the flow direction ∂G/∂σ is unique even where the yield surface has corners.
//! The hardening variable is ε̄̇p = σ:ε̇p / c (plastic work per cohesion,
//! reported as PEEQ).
//!
//! Implicit Euler in principal stress space (isotropy: the trial stress's
//! principal directions are kept): for a given Δλ, σ = σ_tr − Δλ D ∂G/∂σ(σ) is solved with
//! Newton (smooth, since G is smooth) and ε̄p from the hardening law; the scalar
//! function F(Δλ) is then zeroed with a bracketed secant method. Corners and apex
//! of the yield surface therefore need no special treatment. The consistent tangent is
//! numerical (forward difference of the whole return algorithm) and unsymmetric when
//! ψ ≠ φ; with *STEP, UNSYMM=YES the full matrix is assembled, otherwise the symmetric part.
//!
//! *TENSION CUTOFF (Rankine surface σ_max = σt) gives the model in
//! `mohr_coulomb_cutoff.rs` (feature area geomaterials).
//!
//! Output: PEEQ (ε̄p) and PE (plastic strain).

use super::plasticity::{iso_d6, mv6, update_reduced, Hardening, M6, V6};
use super::principal::{fd_tangent, fold_theta, from_principal, grad_pq_theta, pq_theta, principal, root_from_zero};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::f64::consts::FRAC_PI_3;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Mohr–Coulomb plasticity",
    consumes: &["ELASTIC", "MOHR COULOMB", "MOHR COULOMB HARDENING", "TENSION CUTOFF"],
    priority: 10,
    build: Some(build),
}];

// feature area geomaterials: tension cutoff (*TENSION CUTOFF)
#[path = "mohr_coulomb_cutoff.rs"]
mod cutoff;

/// Limit on the deviatoric angle at the corners: 1 − cos 3δ = 1.4e−7.
fn corner_delta() -> f64 {
    (1.0 - 1.4e-7f64).acos() / 3.0
}

#[derive(Debug, Clone)]
pub struct MohrCoulomb {
    pub e: f64,
    pub nu: f64,
    pub g: f64,
    pub k: f64,
    pub ce: M6,
    pub tan_phi: f64,
    pub cos_phi: f64,
    pub tan_psi: f64,
    /// Deviatoric eccentricity e (1/2 < e ≤ 1).
    pub ecc: f64,
    /// ε c|₀ tan ψ (the hyperbola's asymptote distance in the meridional plane).
    pub a: f64,
    /// Cohesion c(ε̄p).
    pub hard: Hardening,
    delta: f64,
}

/// State layout: [ε̄p, εp (6), ε 3D (6)].
const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const NST: usize = 13;

fn cohesion_table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "rate- or temperature-dependent cohesion is not supported yet");
        }
        let (c, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(c > 0.0) {
            return kw.err_at(d.line, "the cohesion must be positive");
        }
        match t.last() {
            Some(&(pl, _)) if !(p > pl) => return kw.err_at(d.line, "the plastic strain must increase"),
            None if p != 0.0 => return kw.err_at(d.line, "the first point must have plastic strain 0"),
            _ => {}
        }
        t.push((p, c));
    }
    if t.is_empty() {
        return kw.err("cohesion is missing");
    }
    Ok(t)
}

/// (E, ν) from *ELASTIC (isotropic, temperature-independent).
pub fn iso_elastic(def: &MaterialDef, who: &RawKeyword, what: &str) -> Result<(f64, f64)> {
    let Some(el) = def.behavior("ELASTIC") else { return who.err(format!("*{} requires *ELASTIC", who.name)) };
    if el.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || el.data.len() > 1 || el.has("DEPENDENCIES") {
        return el.err(format!("{what} requires isotropic, temperature-independent elasticity"));
    }
    let ev = el.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0 && nu > -1.0 && nu < 0.5) {
        return el.err("E must be positive and ν between −1 and 0.5");
    }
    Ok((e, nu))
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("MOHR COULOMB") else { return Ok(None) };
    let (e, nu) = iso_elastic(def, kw, "Mohr–Coulomb")?;
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperature-dependent parameters are not supported yet");
    }
    let v = kw.first_floats()?;
    let phi = v.first().copied().unwrap_or(0.0);
    let psi = v.get(1).copied().unwrap_or(0.0);
    if v.get(2).is_some_and(|&t| t != 0.0) {
        return kw.err("temperature-dependent parameters are not supported yet");
    }
    if !(0.0..90.0).contains(&phi) || !(0.0..90.0).contains(&psi) || psi > phi {
        return kw.err("φ and ψ must lie between 0 and 90 degrees with ψ ≤ φ");
    }
    let (sp, cp) = (phi.to_radians().sin(), phi.to_radians().cos());
    let ecc = match kw.param("DEVIATORIC ECCENTRICITY") {
        Some(s) => s.trim().parse::<f64>().map_err(|_| crate::model::FemError(format!("{}: invalid DEVIATORIC ECCENTRICITY", kw.location())))?,
        None => (3.0 - sp) / (3.0 + sp),
    };
    if !(ecc > 0.5 && ecc <= 1.0) {
        return kw.err("DEVIATORIC ECCENTRICITY must lie in (0.5, 1]");
    }
    // meridional eccentricity: fixed at 0.1 (no input parameter)
    let mer = 0.1;
    let Some(hk) = def.behavior("MOHR COULOMB HARDENING") else {
        return kw.err("*MOHR COULOMB requires *MOHR COULOMB HARDENING");
    };
    if hk.has("RATE") || hk.has("DEPENDENCIES") {
        return hk.err("rate- or temperature-dependent hardening is not supported yet");
    }
    let tab = cohesion_table(hk)?;
    let c0 = tab[0].1;
    let tan_psi = psi.to_radians().tan();
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    let mc = MohrCoulomb {
        e,
        nu,
        g,
        k,
        ce: iso_d6(e, nu),
        tan_phi: sp / cp,
        cos_phi: cp,
        tan_psi,
        ecc,
        a: mer * c0 * tan_psi,
        hard: Hardening::Table(tab),
        delta: corner_delta(),
    };
    if let Some(tk) = def.behavior("TENSION CUTOFF") {
        return Ok(Some(Arc::new(cutoff::build(mc, tk)?)));
    }
    Ok(Some(Arc::new(mc)))
}

impl MohrCoulomb {
    /// R_mc(Θ, φ) with the limit on Θ at the corners.
    fn r_mc(&self, th: f64) -> f64 {
        let t = th.clamp(self.delta, FRAC_PI_3 - self.delta);
        (t + FRAC_PI_3).sin() / (3f64.sqrt() * self.cos_phi) + (t + FRAC_PI_3).cos() * self.tan_phi / 3.0
    }

    /// R_mw(Θ, e) and dR_mw/dΘ.
    fn r_mw(&self, th: f64) -> (f64, f64) {
        let e = self.ecc;
        let r60 = (FRAC_PI_3 + FRAC_PI_3).sin() / (3f64.sqrt() * self.cos_phi) + (FRAC_PI_3 + FRAC_PI_3).cos() * self.tan_phi / 3.0;
        let (c, s) = (th.cos(), th.sin());
        let w = 1.0 - e * e;
        let a = 4.0 * w * c * c + (2.0 * e - 1.0).powi(2);
        let d = (4.0 * w * c * c + 5.0 * e * e - 4.0 * e).max(0.0);
        let sd = d.sqrt();
        let b = 2.0 * w * c + (2.0 * e - 1.0) * sd;
        let da = -8.0 * w * c * s;
        let db = -2.0 * w * s + if sd > 0.0 { (2.0 * e - 1.0) * da / (2.0 * sd) } else { 0.0 };
        (a / b * r60, (da * b - a * db) / (b * b) * r60)
    }

    /// ∂G/∂σ in principal stress space.
    fn flow(&self, s: &[f64; 3]) -> [f64; 3] {
        let (_, q, th) = pq_theta(s);
        let (tf, sgn) = fold_theta(th);
        let (r, dr) = self.r_mw(tf);
        let gh = (self.a * self.a + r * r * q * q).sqrt();
        let (gq, gt) = if gh > 0.0 { (r * r * q / gh, r * q * dr * sgn / gh) } else { (r, 0.0) };
        grad_pq_theta(th, -self.tan_psi, gq, gt)
    }

    fn d_times(&self, n: &[f64; 3]) -> [f64; 3] {
        let tr = n[0] + n[1] + n[2];
        std::array::from_fn(|i| self.k * tr + 2.0 * self.g * (n[i] - tr / 3.0))
    }

    /// σ(Δλ): solves σ = σ_tr − Δλ D ∂G/∂σ(σ) with Newton (initial value in `s`).
    fn stress_at(&self, str_: &[f64; 3], dl: f64, s: &mut [f64; 3]) -> Result<()> {
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.hard.initial());
        let res = |s: &[f64; 3]| -> [f64; 3] {
            let dn = self.d_times(&self.flow(s));
            std::array::from_fn(|i| s[i] - str_[i] + dl * dn[i])
        };
        let mut r = res(s);
        for _ in 0..100 {
            let nr = r.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if nr <= 1e-14 * scale {
                return Ok(());
            }
            // Jacobian I + Δλ D H with H = ∂²G/∂σ² numerically (central difference)
            let h = 1e-6 * scale;
            let mut j = [[0.0; 3]; 3];
            for c in 0..3 {
                let (mut sp, mut sm) = (*s, *s);
                sp[c] += h;
                sm[c] -= h;
                let (np, nm) = (self.flow(&sp), self.flow(&sm));
                let dn: [f64; 3] = std::array::from_fn(|i| (np[i] - nm[i]) / (2.0 * h));
                let ddn = self.d_times(&dn);
                for i in 0..3 {
                    j[i][c] = if i == c { 1.0 } else { 0.0 } + dl * ddn[i];
                }
            }
            let jv: Vec<f64> = j.iter().flatten().copied().collect();
            let Some(d) = super::anisotropic::solve_dense(&jv, &r) else {
                return fail("Mohr–Coulomb: singular Jacobian in the return algorithm");
            };
            // damped step if the residual does not decrease
            let mut t = 1.0;
            loop {
                let sn: [f64; 3] = std::array::from_fn(|i| s[i] - t * d[i]);
                let rn = res(&sn);
                let nn = rn.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                if nn < nr || t < 1e-4 {
                    *s = sn;
                    r = rn;
                    break;
                }
                t *= 0.5;
            }
        }
        let nr = r.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        if nr <= 1e-10 * scale {
            return Ok(());
        }
        fail("Mohr–Coulomb: the return algorithm (stress for given Δλ) did not converge")
    }

    /// ε̄p at the end of the increment: (ε̄p − ε̄p_n) c(ε̄p) = Δλ σ:∂G/∂σ.
    fn peeq_at(&self, pn: f64, dl: f64, w: f64) -> f64 {
        let mut p = pn + dl * w / self.hard.eval(pn).0;
        for _ in 0..60 {
            let (c, h) = self.hard.eval(p);
            let r = (p - pn) * c - dl * w;
            let dr = c + (p - pn) * h;
            let dp = r / dr;
            p -= dp;
            if dp.abs() <= 1e-15 * p.abs().max(1e-12) {
                break;
            }
        }
        p
    }

    fn ret(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        stn[..NST].copy_from_slice(&st[..NST]);
        let pn = st[P];
        let (ltr, v) = principal(&sig_tr);
        let fyield = |s: &[f64; 3], c: f64| -> f64 {
            let (p, q, th) = pq_theta(s);
            self.r_mc(fold_theta(th).0) * q - p * self.tan_phi - c
        };
        let c0 = self.hard.eval(pn).0;
        let f_tr = fyield(&ltr, c0);
        let scale = ltr.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(c0);
        if f_tr <= 1e-12 * scale {
            return Ok(sig_tr);
        }
        let mut s = ltr;
        let mut f = |dl: f64| -> Result<f64> {
            if dl == 0.0 {
                return Ok(f_tr);
            }
            self.stress_at(&ltr, dl, &mut s)?;
            let n = self.flow(&s);
            let w = s[0] * n[0] + s[1] * n[1] + s[2] * n[2];
            Ok(fyield(&s, self.hard.eval(self.peeq_at(pn, dl, w)).0))
        };
        let x0 = f_tr / (3.0 * self.g + self.k * self.tan_phi * self.tan_psi);
        let dl = root_from_zero(&mut f, x0, 1e-13 * scale, "Mohr–Coulomb")?;
        let mut s = ltr;
        self.stress_at(&ltr, dl, &mut s)?;
        let n = self.flow(&s);
        let w = s[0] * n[0] + s[1] * n[1] + s[2] * n[2];
        stn[P] = self.peeq_at(pn, dl, w);
        let dep = from_principal(&[dl * n[0], dl * n[1], dl * n[2]], &v, true);
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dep[i];
        }
        Ok(from_principal(&s, &v, false))
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let sig = self.ret(eps, st, stn)?;
        if stn[P] == st[P] && (0..6).all(|i| stn[EP + i] == st[EP + i]) {
            return Ok((sig, self.ce));
        }
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.hard.initial() / self.e);
        let mut tmp = vec![0.0; NST];
        let c = fd_tangent(eps, &sig, 1e-7 * scale, |e2| self.ret(e2, st, &mut tmp))?;
        Ok((sig, c))
    }
}

impl MaterialModel for MohrCoulomb {
    fn name(&self) -> &str {
        "MOHR COULOMB"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn init_state(&self, state: &mut [f64]) {
        // PEEQ for Mohr–Coulomb starts at 2.22e−16 (visible in the results even
        // before plastic yielding)
        state[P] = 2.22e-16;
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
