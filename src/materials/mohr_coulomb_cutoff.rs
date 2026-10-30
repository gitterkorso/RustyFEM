//! Tension cutoff for Mohr–Coulomb (*TENSION CUTOFF): the Rankine surface.
//! The module is included from `mohr_coulomb.rs` (`#[path]`) and
//! uses private parts of the Mohr–Coulomb model.
//!
//!   F_t = R_r(Θ) q − p − σ_t(ε̄_t^pl) = σ_max − σ_t = 0,  R_r = (2/3) cos Θ,
//!
//! with nearly associated flow
//!
//!   G_t = √((ε_t σ_t|₀)² + (R_t(Θ, e_t) q)²) − p,
//!   R_t = (2e_t/3) [4(1−e_t²)cos²Θ + (2e_t−1)²] / [2(1−e_t²)cos Θ + (2e_t−1)√(4(1−e_t²)cos²Θ + 5e_t² − 4e_t)],
//!
//! ε_t = 0.1 and e_t = 0.6 (fixed). The factor 2e_t/3 (= 0.4, so that
//! R_t(0) = R_r(0) = 2/3 in uniaxial tension) is verified with plastic
//! strains in tension and in tension with shear. The hardening variable of the tension part
//! (PEEQT) is ε̄̇_t^pl = σ:ε̇_t^pl/σ_t (the same form as
//! the Mohr–Coulomb PEEQ). σ_t(ε̄_t^pl) is given as a table on *TENSION CUTOFF.
//!
//! Return in principal stress space with implicit Euler: first the surface that
//! the trial stress exceeds most, then the other, finally both simultaneously
//! (two plastic multipliers, σ = σ_tr − D(Δλ₁ ∂G/∂σ + Δλ₂ ∂G_t/∂σ),
//! Newton in (Δλ₁, Δλ₂)). The first combination that satisfies both
//! conditions with non-negative multipliers is chosen. The tangent is numerical
//! (forward difference of the return algorithm) as for Mohr–Coulomb.

use super::*;
use crate::materials::principal::{fd_tangent, fold_theta, from_principal, grad_pq_theta, pq_theta, principal, root_from_zero};

/// Fixed parameters for the flow potential of the Rankine surface.
const ECC_T: f64 = 0.6;
const MER_T: f64 = 0.1;

/// State: Mohr–Coulomb's [ε̄p, εp (6), ε (6)] followed by ε̄_t^pl.
const PT: usize = NST;
const NST_T: usize = NST + 1;

#[derive(Debug, Clone)]
pub struct MohrCoulombCutoff {
    mc: MohrCoulomb,
    /// σ_t(ε̄_t^pl).
    tc: Hardening,
    /// ε_t σ_t|₀.
    at: f64,
}

/// σ_t table from *TENSION CUTOFF (σ_t, ε̄_t^pl).
fn cutoff_table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    if kw.has("DEPENDENCIES") {
        return kw.err("field-dependent tension cutoff is not yet supported");
    }
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "temperature-dependent tension cutoff is not yet supported");
        }
        let (s, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if s < 0.0 {
            return kw.err_at(d.line, "the tension cutoff σ_t must not be negative");
        }
        match t.last() {
            Some(&(pl, _)) if !(p > pl) => return kw.err_at(d.line, "the plastic strain must increase"),
            None if p != 0.0 => return kw.err_at(d.line, "the first point must have plastic strain 0"),
            _ => {}
        }
        t.push((p, s));
    }
    if t.is_empty() {
        return kw.err("tension cutoff σ_t is missing");
    }
    if !(t[0].1 > 0.0) {
        return kw.err("the first tension cutoff σ_t must be positive");
    }
    Ok(t)
}

pub fn build(mc: MohrCoulomb, kw: &RawKeyword) -> Result<MohrCoulombCutoff> {
    let tab = cutoff_table(kw)?;
    let s0 = tab[0].1;
    Ok(MohrCoulombCutoff { mc, tc: Hardening::Table(tab), at: MER_T * s0 })
}

/// Menétrey–Willam R(Θ, e) without scale factor, and dR/dΘ.
fn mw(th: f64, e: f64) -> (f64, f64) {
    let (c, s) = (th.cos(), th.sin());
    let w = 1.0 - e * e;
    let a = 4.0 * w * c * c + (2.0 * e - 1.0).powi(2);
    let d = (4.0 * w * c * c + 5.0 * e * e - 4.0 * e).max(0.0);
    let sd = d.sqrt();
    let b = 2.0 * w * c + (2.0 * e - 1.0) * sd;
    let da = -8.0 * w * c * s;
    let db = -2.0 * w * s + if sd > 0.0 { (2.0 * e - 1.0) * da / (2.0 * sd) } else { 0.0 };
    (a / b, (da * b - a * db) / (b * b))
}

/// ε̄ at the end of the increment: (ε̄ − ε̄_n) h(ε̄) = Δλ σ:∂G/∂σ.
fn eq_at(hard: &Hardening, pn: f64, dl: f64, w: f64) -> f64 {
    let mut p = pn + dl * w / hard.eval(pn).0.max(1e-300);
    for _ in 0..60 {
        let (c, h) = hard.eval(p);
        let r = (p - pn) * c - dl * w;
        let dr = c + (p - pn) * h;
        if dr == 0.0 {
            break;
        }
        let dp = r / dr;
        p -= dp;
        if dp.abs() <= 1e-15 * p.abs().max(1e-12) {
            break;
        }
    }
    p
}

/// Result of a return with given multipliers.
struct Ret {
    s: [f64; 3],
    p: f64,
    pt: f64,
    fm: f64,
    ft: f64,
    n1: [f64; 3],
    n2: [f64; 3],
}

impl MohrCoulombCutoff {
    /// ∂G_t/∂σ in principal stress space.
    fn flow_t(&self, s: &[f64; 3]) -> [f64; 3] {
        let (_, q, th) = pq_theta(s);
        let (tf, sgn) = fold_theta(th);
        let (r0, dr0) = mw(tf, ECC_T);
        let f = 2.0 / 3.0 * ECC_T;
        let (r, dr) = (r0 * f, dr0 * f);
        let gh = (self.at * self.at + r * r * q * q).sqrt();
        let (gq, gt) = if gh > 0.0 { (r * r * q / gh, r * q * dr * sgn / gh) } else { (r, 0.0) };
        grad_pq_theta(th, -1.0, gq, gt)
    }

    fn f_mc(&self, s: &[f64; 3], c: f64) -> f64 {
        let (p, q, th) = pq_theta(s);
        self.mc.r_mc(fold_theta(th).0) * q - p * self.mc.tan_phi - c
    }

    fn f_t(s: &[f64; 3], st: f64) -> f64 {
        s[0].max(s[1]).max(s[2]) - st
    }

    /// σ(Δλ₁, Δλ₂): σ = σ_tr − D(Δλ₁ n_mc(σ) + Δλ₂ n_t(σ)) with Newton.
    fn stress_at(&self, str_: &[f64; 3], d1: f64, d2: f64, s: &mut [f64; 3]) -> Result<()> {
        let mc = &self.mc;
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(mc.hard.initial()).max(self.tc.initial());
        let res = |s: &[f64; 3]| -> [f64; 3] {
            let (a, b) = (mc.flow(s), self.flow_t(s));
            let n: [f64; 3] = std::array::from_fn(|i| d1 * a[i] + d2 * b[i]);
            let dn = mc.d_times(&n);
            std::array::from_fn(|i| s[i] - str_[i] + dn[i])
        };
        let mut r = res(s);
        for _ in 0..100 {
            let nr = r.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if nr <= 1e-14 * scale {
                return Ok(());
            }
            let h = 1e-6 * scale;
            let mut j = [0.0; 9];
            for c in 0..3 {
                let (mut sp, mut sm) = (*s, *s);
                sp[c] += h;
                sm[c] -= h;
                let (rp, rm) = (res(&sp), res(&sm));
                for i in 0..3 {
                    j[i * 3 + c] = (rp[i] - rm[i]) / (2.0 * h);
                }
            }
            let Some(d) = crate::materials::anisotropic::solve_dense(&j, &r) else {
                return fail("Mohr–Coulomb with tension cutoff: singular Jacobian in the return algorithm");
            };
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
        fail("Mohr–Coulomb with tension cutoff: the return algorithm (stress for given multipliers) did not converge")
    }

    fn eval(&self, ltr: &[f64; 3], pn: f64, ptn: f64, d1: f64, d2: f64, s0: &[f64; 3]) -> Result<Ret> {
        let mut s = *s0;
        self.stress_at(ltr, d1, d2, &mut s)?;
        let (n1, n2) = (self.mc.flow(&s), self.flow_t(&s));
        let w1 = s[0] * n1[0] + s[1] * n1[1] + s[2] * n1[2];
        let w2 = s[0] * n2[0] + s[1] * n2[1] + s[2] * n2[2];
        let p = if d1 > 0.0 { eq_at(&self.mc.hard, pn, d1, w1) } else { pn };
        let pt = if d2 > 0.0 { eq_at(&self.tc, ptn, d2, w2) } else { ptn };
        let fm = self.f_mc(&s, self.mc.hard.eval(p).0);
        let ft = Self::f_t(&s, self.tc.eval(pt).0);
        Ok(Ret { s, p, pt, fm, ft, n1, n2 })
    }

    fn ret(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let mc = &self.mc;
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&mc.ce, &ee);
        stn[..NST_T].copy_from_slice(&st[..NST_T]);
        let (pn, ptn) = (st[P], st[PT]);
        let (ltr, v) = principal(&sig_tr);
        let (c0, t0) = (mc.hard.eval(pn).0, self.tc.eval(ptn).0);
        let scale = ltr.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(c0);
        let tol = 1e-12 * scale;
        let (fm0, ft0) = (self.f_mc(&ltr, c0), Self::f_t(&ltr, t0));
        if fm0 <= tol && ft0 <= tol {
            return Ok(sig_tr);
        }
        let x0m = fm0.max(0.0) / (3.0 * mc.g + mc.k * mc.tan_phi * mc.tan_psi);
        let x0t = ft0.max(0.0) / (4.0 / 3.0 * mc.g + mc.k);
        let ok = |r: &Ret, d1: f64, d2: f64| -> bool {
            let ftol = 1e-9 * scale;
            d1 >= 0.0 && d2 >= 0.0 && r.fm <= ftol && r.ft <= ftol
        };
        // one surface at a time
        let single_mc = || -> Result<Option<(f64, f64, Ret)>> {
            if fm0 <= tol {
                return Ok(None);
            }
            let mut s = ltr;
            let d1 = root_from_zero(
                |dl| {
                    if dl == 0.0 {
                        return Ok(fm0);
                    }
                    let r = self.eval(&ltr, pn, ptn, dl, 0.0, &s)?;
                    s = r.s;
                    Ok(r.fm)
                },
                x0m.max(1e-30),
                1e-13 * scale,
                "Mohr–Coulomb",
            )?;
            Ok(Some((d1, 0.0, self.eval(&ltr, pn, ptn, d1, 0.0, &ltr)?)))
        };
        let single_t = || -> Result<Option<(f64, f64, Ret)>> {
            if ft0 <= tol {
                return Ok(None);
            }
            let mut s = ltr;
            let d2 = root_from_zero(
                |dl| {
                    if dl == 0.0 {
                        return Ok(ft0);
                    }
                    let r = self.eval(&ltr, pn, ptn, 0.0, dl, &s)?;
                    s = r.s;
                    Ok(r.ft)
                },
                x0t.max(1e-30),
                1e-13 * scale,
                "Mohr–Coulomb (tension cutoff)",
            )?;
            Ok(Some((0.0, d2, self.eval(&ltr, pn, ptn, 0.0, d2, &ltr)?)))
        };
        let first_mc = fm0 / c0.max(1e-300) >= ft0 / t0.max(1e-300);
        let mut found: Option<(f64, f64, Ret)> = None;
        let order: [bool; 2] = if first_mc { [true, false] } else { [false, true] };
        let mut guess = (x0m, x0t);
        for use_mc in order {
            let cand = if use_mc { single_mc()? } else { single_t()? };
            if let Some((d1, d2, r)) = cand {
                if ok(&r, d1, d2) {
                    found = Some((d1, d2, r));
                    break;
                }
                if use_mc {
                    guess.0 = d1;
                } else {
                    guess.1 = d2;
                }
            }
        }
        if found.is_none() {
            found = Some(self.corner(&ltr, pn, ptn, guess, scale)?);
        }
        let (d1, d2, r) = found.unwrap();
        stn[P] = r.p;
        stn[PT] = r.pt;
        let dep: [f64; 3] = std::array::from_fn(|i| d1 * r.n1[i] + d2 * r.n2[i]);
        let dep6 = from_principal(&dep, &v, true);
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dep6[i];
        }
        Ok(from_principal(&r.s, &v, false))
    }

    /// Both surfaces active: Newton in (Δλ₁, Δλ₂).
    fn corner(&self, ltr: &[f64; 3], pn: f64, ptn: f64, guess: (f64, f64), scale: f64) -> Result<(f64, f64, Ret)> {
        let (c0, t0) = (self.mc.hard.eval(pn).0.max(1e-300), self.tc.eval(ptn).0.max(1e-300));
        let mut x = [guess.0.max(0.0), guess.1.max(0.0)];
        let mut s = *ltr;
        let resid = |x: &[f64; 2], s: &mut [f64; 3]| -> Result<([f64; 2], Ret)> {
            let r = self.eval(ltr, pn, ptn, x[0], x[1], s)?;
            *s = r.s;
            Ok(([r.fm / c0, r.ft / t0], r))
        };
        let (mut f, mut r) = resid(&x, &mut s)?;
        for _ in 0..60 {
            let nf = f[0].abs().max(f[1].abs());
            if nf <= 1e-13 * scale / c0.min(t0) {
                break;
            }
            let mut j = [0.0; 4];
            for c in 0..2 {
                let h = 1e-7 * x[c].abs().max(1e-10);
                let mut x2 = x;
                x2[c] += h;
                let mut s2 = s;
                let (f2, _) = resid(&x2, &mut s2)?;
                j[c] = (f2[0] - f[0]) / h;
                j[2 + c] = (f2[1] - f[1]) / h;
            }
            let det = j[0] * j[3] - j[1] * j[2];
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let dx = [(f[0] * j[3] - f[1] * j[1]) / det, (j[0] * f[1] - j[2] * f[0]) / det];
            let mut t = 1.0;
            loop {
                let xn = [(x[0] - t * dx[0]).max(0.0), (x[1] - t * dx[1]).max(0.0)];
                let mut sn = s;
                let (fnew, rn) = resid(&xn, &mut sn)?;
                if fnew[0].abs().max(fnew[1].abs()) < nf || t < 1e-4 {
                    x = xn;
                    s = sn;
                    f = fnew;
                    r = rn;
                    break;
                }
                t *= 0.5;
            }
        }
        if f[0].abs().max(f[1].abs()) > 1e-8 {
            return fail("Mohr–Coulomb with tension cutoff: the return algorithm at the corner did not converge");
        }
        Ok((x[0], x[1], r))
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let sig = self.ret(eps, st, stn)?;
        if stn[P] == st[P] && stn[PT] == st[PT] && (0..6).all(|i| stn[EP + i] == st[EP + i]) {
            return Ok((sig, self.mc.ce));
        }
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.tc.initial() / self.mc.e);
        let mut tmp = vec![0.0; NST_T];
        let c = fd_tangent(eps, &sig, 1e-7 * scale, |e2| self.ret(e2, st, &mut tmp))?;
        Ok((sig, c))
    }
}

impl MaterialModel for MohrCoulombCutoff {
    fn name(&self) -> &str {
        "MOHR COULOMB"
    }

    fn n_state(&self) -> usize {
        NST_T
    }

    fn init_state(&self, state: &mut [f64]) {
        self.mc.init_state(state);
        state[PT] = 2.22e-16;
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.mc.e, self.mc.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        self.mc.rotate_state(r, state);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.mc.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.mc.tangent_linear(mode)
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ", "PEEQT"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[P], state[PT]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.mc.output_fields(mode);
        v.push(("PEEQT", vec!["PEEQT"]));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = self.mc.output_field_values(mode, state);
        v.push(state[PT]);
        v
    }
}
