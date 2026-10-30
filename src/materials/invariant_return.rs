//! Shared implicit return algorithm (backward Euler) for plasticity written
//! in terms of the invariants p, q and t (extended Drucker–Prager models and
//! Drucker–Prager Cap, `plasticity_dp2.rs` and `cap.rs`).
//!
//! Invariants:
//!
//!   p = −tr σ/3,  q = √(3/2 S:S),  r³ = 9/2 S:(S·S),
//!   t = q/2 [1 + 1/K − (1 − 1/K)(r/q)³]   (t = q when K = 1).
//!
//! The gradients with respect to σ (tensor components) are ∂p/∂σ = −I/3,
//! ∂q/∂σ = 3/2 S/q and ∂(r³)/∂σ = 27/2 dev(S·S).
//!
//! With isotropic elasticity (G, K) the unknowns (σ, Δλ, h) are solved from:
//!
//!   σ − σ_tr + Δλ C:N(σ, h) = 0,  F(σ, h) = 0,  h − hₙ − Δλ ḣ(σ, N, h) = 0,
//!
//! where N = ∂G/∂σ is the flow direction and h the model's hardening variable
//! (equivalent plastic strain or volumetric plastic strain). The system
//! (8 unknowns) is solved with Newton and a Jacobian from central differences of the
//! residual; the residual itself is analytical, so the converged solution is
//! the exact backward Euler solution. The plastic strain becomes Δεᵖ = Δλ N
//! (engineering shear strain in the output).

use super::plasticity::{ddot, dev, V6};
use super::MaterialFactory;
use crate::model::{fail, Result};

/// No factories: the module is a building block for other material models.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Invariants and their gradients for the stress σ.
#[derive(Debug, Clone, Copy)]
pub struct Inv {
    pub p: f64,
    pub q: f64,
    pub t: f64,
    /// ∂t/∂σ (tensor components).
    pub dt: V6,
}

/// ∂p/∂σ.
pub const DP: V6 = [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0];

/// S·S for a deviator in Voigt form (tensor shear components).
fn square(s: &V6) -> V6 {
    let a = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
    let m = |i: usize, j: usize| (0..3).map(|k| a[i][k] * a[k][j]).sum::<f64>();
    [m(0, 0), m(1, 1), m(2, 2), m(0, 1), m(0, 2), m(1, 2)]
}

/// The invariants with K (the ratio of the yield stresses in triaxial tension and compression).
pub fn invariants(sig: &V6, k: f64) -> Inv {
    let s = dev(sig);
    let p = -(sig[0] + sig[1] + sig[2]) / 3.0;
    let q = (1.5 * ddot(&s, &s)).max(0.0).sqrt();
    let scale = sig.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if !(q > 1e-14 * scale) {
        return Inv { p, q: q.max(0.0), t: q.max(0.0), dt: [0.0; 6] };
    }
    let dq: V6 = std::array::from_fn(|i| 1.5 * s[i] / q);
    if (k - 1.0).abs() < 1e-15 {
        return Inv { p, q, t: q, dt: dq };
    }
    let ss = square(&s);
    let r3 = 4.5 * (ss[0] * s[0] + ss[1] * s[1] + ss[2] * s[2] + 2.0 * (ss[3] * s[3] + ss[4] * s[4] + ss[5] * s[5]));
    let ik = 1.0 / k;
    let t = 0.5 * q * (1.0 + ik) - (1.0 - ik) * r3 / (2.0 * q * q);
    let tq = 0.5 * (1.0 + ik) + (1.0 - ik) * r3 / (q * q * q);
    let tr3 = -(1.0 - ik) / (2.0 * q * q);
    let dss = dev(&ss);
    let dt: V6 = std::array::from_fn(|i| tq * dq[i] + tr3 * 13.5 * dss[i]);
    Inv { p, q, t, dt }
}

/// Yield surface, flow potential and hardening for the return algorithm.
pub trait Surface {
    /// K in t (1 = von Mises circle in the deviatoric plane).
    fn k(&self) -> f64;
    /// (F, ∂F/∂p, ∂F/∂t) at hardening h.
    fn yield_fn(&self, p: f64, t: f64, h: f64) -> (f64, f64, f64);
    /// (∂G/∂p, ∂G/∂t) at hardening h.
    fn potential(&self, p: f64, t: f64, h: f64) -> (f64, f64);
    /// ḣ/λ̇ for the stress σ with flow direction N.
    fn hardening_rate(&self, sig: &V6, inv: &Inv, n: &V6, h: f64, h_n: f64) -> f64;
    /// Stress scale for tolerances (e.g. the yield stress).
    fn stress_scale(&self, h: f64) -> f64;
}

/// The flow direction N = ∂G/∂σ (tensor components).
pub fn flow(s: &dyn Surface, inv: &Inv, h: f64) -> V6 {
    let (gp, gt) = s.potential(inv.p, inv.t, h);
    std::array::from_fn(|i| gp * DP[i] + gt * inv.dt[i])
}

/// Result of a plastic return.
#[derive(Debug, Clone, Copy)]
pub struct Return {
    pub sig: V6,
    /// Plastic strain increment (engineering shear strain).
    pub dep: V6,
    pub h: f64,
    pub dl: f64,
}

fn solve8(a: &mut [[f64; 8]; 8], b: &mut [f64; 8]) -> bool {
    for c in 0..8 {
        let piv = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        if a[piv][c].abs() < 1e-300 {
            return false;
        }
        a.swap(c, piv);
        b.swap(c, piv);
        for r in c + 1..8 {
            let f = a[r][c] / a[c][c];
            if f != 0.0 {
                for k in c..8 {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    for c in (0..8).rev() {
        let s: f64 = (c + 1..8).map(|k| a[c][k] * b[k]).sum();
        b[c] = (b[c] - s) / a[c][c];
    }
    true
}

/// Damped Newton with a Jacobian from central differences for a small system
/// r(x) = 0 (scaled unknowns of order 1). `typ` is the typical size
/// per unknown (step length for the differences). Returns the max norm of the final residual.
pub fn newton_fd(x: &mut [f64], typ: &[f64], r: &dyn Fn(&[f64]) -> Vec<f64>, tol: f64) -> f64 {
    let n = x.len();
    let norm = |v: &[f64]| v.iter().fold(0.0f64, |m, a| m.max(a.abs()));
    let mut rv = r(x);
    for _ in 0..100 {
        if norm(&rv) <= tol {
            break;
        }
        let mut jac = vec![0.0; n * n];
        for j in 0..n {
            let h = 1e-6 * x[j].abs().max(typ[j] * 1e-3).max(1e-12);
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[j] += h;
            xm[j] -= h;
            let (rp, rm) = (r(&xp), r(&xm));
            for i in 0..n {
                jac[i * n + j] = (rp[i] - rm[i]) / (2.0 * h);
            }
        }
        let rhs: Vec<f64> = rv.iter().map(|v| -v).collect();
        let Some(d) = super::anisotropic::solve_dense(&jac, &rhs) else { break };
        let r0 = norm(&rv);
        let mut lam = 1.0;
        let mut ok = false;
        for _ in 0..30 {
            let xn: Vec<f64> = (0..n).map(|i| x[i] + lam * d[i]).collect();
            let rn = r(&xn);
            if norm(&rn) < r0 {
                x.copy_from_slice(&xn);
                rv = rn;
                ok = true;
                break;
            }
            lam *= 0.5;
        }
        if !ok {
            break;
        }
    }
    norm(&rv)
}

/// Backward Euler return from the trial stress `sig_tr` with hardening `h_n` and
/// elasticity (G, K). `None` if the trial state is elastic.
pub fn return_map(s: &dyn Surface, g: f64, kb: f64, sig_tr: &V6, h_n: f64) -> Result<Option<Return>> {
    let inv_tr = invariants(sig_tr, s.k());
    let sc0 = s.stress_scale(h_n);
    let (f_tr, _, _) = s.yield_fn(inv_tr.p, inv_tr.t, h_n);
    if f_tr <= 1e-12 * sc0 {
        return Ok(None);
    }
    let sc = sc0.max(sig_tr.iter().fold(0.0f64, |m, x| m.max(x.abs())));
    // typical size of Δλ and h (strains)
    let e_typ = sc / (2.0 * g);
    // unknowns: x = [σ/sc (6), Δλ/e_typ, h]
    let resid = |x: &[f64; 8]| -> [f64; 8] {
        let sig: V6 = std::array::from_fn(|i| x[i] * sc);
        let dl = x[6] * e_typ;
        let h = x[7];
        let inv = invariants(&sig, s.k());
        let n = flow(s, &inv, h);
        let tr = n[0] + n[1] + n[2];
        let dn = dev(&n);
        let mut r = [0.0; 8];
        for i in 0..6 {
            let c = 2.0 * g * dn[i] + if i < 3 { kb * tr } else { 0.0 };
            r[i] = (sig[i] - sig_tr[i] + dl * c) / sc;
        }
        r[6] = s.yield_fn(inv.p, inv.t, h).0 / sc;
        r[7] = (h - h_n - dl * s.hardening_rate(&sig, &inv, &n, h, h_n)) / e_typ;
        r
    };
    let mut x = [0.0; 8];
    for i in 0..6 {
        x[i] = sig_tr[i] / sc;
    }
    x[7] = h_n;
    let norm = |r: &[f64; 8]| r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut r = resid(&x);
    let mut conv = false;
    for _ in 0..80 {
        if norm(&r) <= 1e-13 {
            conv = true;
            break;
        }
        let mut jac = [[0.0; 8]; 8];
        for j in 0..8 {
            let typ = if j == 7 { e_typ } else { 1.0 };
            let hstep = 1e-6 * x[j].abs().max(typ * 1e-3).max(1e-12);
            let mut xp = x;
            let mut xm = x;
            xp[j] += hstep;
            xm[j] -= hstep;
            let (rp, rm) = (resid(&xp), resid(&xm));
            for i in 0..8 {
                jac[i][j] = (rp[i] - rm[i]) / (2.0 * hstep);
            }
        }
        let mut d = r.map(|v| -v);
        if !solve8(&mut jac, &mut d) {
            break;
        }
        // damped Newton: halve the step until the residual decreases
        let r0 = norm(&r);
        let mut lam = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let mut xn = x;
            for i in 0..8 {
                xn[i] += lam * d[i];
            }
            if xn[6] < 0.0 {
                xn[6] = 0.0;
            }
            let rn = resid(&xn);
            if norm(&rn) < r0 || lam < 1e-6 {
                x = xn;
                r = rn;
                accepted = true;
                break;
            }
            lam *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    if !conv && norm(&r) > 1e-9 {
        return fail(format!("the return algorithm did not converge (residual {:e})", norm(&r)));
    }
    let sig: V6 = std::array::from_fn(|i| x[i] * sc);
    let dl = x[6] * e_typ;
    let h = x[7];
    let inv = invariants(&sig, s.k());
    let n = flow(s, &inv, h);
    let dep: V6 = std::array::from_fn(|i| dl * n[i] * if i < 3 { 1.0 } else { 2.0 });
    Ok(Some(Return { sig, dep, h, dl }))
}
