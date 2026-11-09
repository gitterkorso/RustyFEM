//! S4R in explicit dynamics (NLGEOM=NO).
//!
//! The explicit element is the same S4R as the implicit one (membrane and bending
//! at one point, assumed transverse shear, see `elements/shell.rs`) except in three
//! respects (checked with reaction forces for prescribed
//! displacement ramps in each degree of freedom, square, parallelogram,
//! t = 0.01 and 0.001):
//!
//! - The membrane hourglass stiffness c_m = 0.0125·G·t·A·Σᵢ|∇Nᵢ|² (implicit:
//!   0.0003125) and the bending c_b = 0.0125·G·t³·A·Σᵢ|∇Nᵢ|² (implicit:
//!   0.01125·G·t³/12·A·Σᵢ|∇Nᵢ|²), with the Flanagan–Belytschko γ in both.
//! - The hourglass force is by default ("relax stiffness") integral-viscoelastic
//!   with a kernel per increment: Q(n) = c·(0.1·q(n) + 0.9·Σₖ e^{−(n−k−½)/500} Δq(k)),
//!   where q = γ·u is the mode amplitude. The kernel was read from the reaction for a
//!   prescribed ramp in the membrane hourglass mode (identical for Δt = 10⁻⁸ and
//!   10⁻⁷, i.e. per increment and not per time; 5 digits over 20–20 000
//!   increments). `*SECTION CONTROLS, HOURGLASS=STIFFNESS` gives Q = c·q.
//! - Drill: the moment at node i is kᵢθᵢ (kᵢ = 0.2·k_ts/(1 + 0.1·A/t²)·det Jᵢ as
//!   in the implicit element) and the membrane gets the force −(Σⱼ kⱼθⱼ)·∂ω/∂u, ω = ½(v,x − u,y)
//!   at the element center. The coupling is thus unsymmetric (no kᵢω² contribution
//!   to the membrane).
//!
//! The implementation takes the implicit element's stiffness (`ElementType::stiffness`)
//! and replaces the three parts in the element's local system. The element's
//! explicit state (in the driver, not the element's `state`) is
//! [q (4), Q_v (4), Q (4), drill force (24)] for the mode amplitudes
//! (membrane x, y, bending θx, θy), the viscous part and the previous forces
//! (for the hourglass and drill energy ALLAE).

use crate::elements::shell::{local_frame, q4pt, vec_to_global, ShellSec};
use crate::elements::ElemCtx;
use crate::model::Result;

/// Number of values in the explicit state per S4R.
pub const STATE_LEN: usize = 36;

/// Hourglass control (*SECTION CONTROLS, HOURGLASS=).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hourglass {
    Relax,
    Stiffness,
}

/// Parameters of the relaxation kernel (non-relaxing fraction, time constant in increments).
pub const RELAX_ALPHA: f64 = 0.1;
pub const RELAX_TAU: f64 = 500.0;

/// New hourglass force for the mode amplitude `q` with the previous value `q0` and
/// the viscous part `qv0`: (Q, Q_v).
pub fn relax(mode: Hourglass, c: f64, q: f64, q0: f64, qv0: f64) -> (f64, f64) {
    match mode {
        Hourglass::Stiffness => (c * q, 0.0),
        Hourglass::Relax => {
            let qv = qv0 * (-1.0 / RELAX_TAU).exp() + (1.0 - RELAX_ALPHA) * (q - q0) * (-0.5 / RELAX_TAU).exp();
            (c * (RELAX_ALPHA * q + qv), qv)
        }
    }
}

/// Internal force (global, 24) for S4R with explicit hourglass and drill.
/// Returns (force, new explicit state, increment of ALLAE).
pub fn s4r_force(ctx: &ElemCtx, k_std: &[f64], u: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let loc = local_frame(&ctx.x, ctx.label, "S4R")?;
    let sec = ShellSec::get(ctx, "S4R")?;
    if sec.zoff != 0.0 {
        return crate::model::fail(format!("element {} (S4R): *SHELL SECTION, OFFSET= is not supported in explicit dynamics", ctx.label));
    }
    let n = 24;
    // local displacements
    let mut ul = vec![0.0; n];
    for b in 0..8 {
        for i in 0..3 {
            ul[3 * b + i] = (0..3).map(|j| loc.r[i][j] * u[3 * b + j]).sum();
        }
    }
    let x = &loc.xl;
    let p0 = q4pt(x, 0.0, 0.0);
    let a = 4.0 * p0.det;
    let bb: f64 = (0..4).map(|i| p0.bx[i] * p0.bx[i] + p0.by[i] * p0.by[i]).sum();
    let h = [1.0, -1.0, 1.0, -1.0];
    let hx: f64 = (0..4).map(|i| h[i] * x[i][0]).sum();
    let hy: f64 = (0..4).map(|i| h[i] * x[i][1]).sum();
    let gam: [f64; 4] = std::array::from_fn(|i| h[i] - hx * p0.bx[i] - hy * p0.by[i]);
    let t = sec.t;
    let g = sec.g;
    // implicit: cm = 0.0003125 G t A Σb², cb = 0.01125 G t³/12 A Σb²
    let std_c = [0.0003125 * g * t * a * bb, 0.0003125 * g * t * a * bb, 0.01125 * g * t.powi(3) / 12.0 * a * bb, 0.01125 * g * t.powi(3) / 12.0 * a * bb];
    let exp_c = [0.0125 * g * t * a * bb, 0.0125 * g * t * a * bb, 0.0125 * g * t.powi(3) * a * bb, 0.0125 * g * t.powi(3) * a * bb];
    let dofs = [0usize, 1, 3, 4];
    let mut fl = vec![0.0; n];
    let mut st = vec![0.0; STATE_LEN];
    let mut dae = 0.0;
    for m in 0..4 {
        let d = dofs[m];
        let q: f64 = (0..4).map(|i| gam[i] * ul[i * 6 + d]).sum();
        let (qn, qv) = relax(mode, exp_c[m], q, old.get(m).copied().unwrap_or(0.0), old.get(4 + m).copied().unwrap_or(0.0));
        st[m] = q;
        st[4 + m] = qv;
        st[8 + m] = qn;
        dae += 0.5 * (qn + old.get(8 + m).copied().unwrap_or(0.0)) * (q - old.get(m).copied().unwrap_or(0.0));
        for i in 0..4 {
            fl[i * 6 + d] += gam[i] * (qn - std_c[m] * q);
        }
    }
    // drill: implicit Σ kᵢ (θᵢ − ωᵢ)² out, the explicit form in
    let kts = sec.tss.map(|k| k[0]).unwrap_or(5.0 / 6.0 * g * t);
    let kfac = 0.2 * kts / (1.0 + 0.1 * a / (t * t));
    let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut fdr = vec![0.0; n];
    let mut ksum = 0.0;
    for i in 0..4 {
        let p = q4pt(x, nat[i][0], nat[i][1]);
        let ki = kfac * p.det;
        let mut gv = vec![0.0; n];
        gv[i * 6 + 5] = 1.0;
        for j in 0..4 {
            gv[j * 6] += 0.5 * p.by[j];
            gv[j * 6 + 1] -= 0.5 * p.bx[j];
        }
        let s: f64 = gv.iter().zip(&ul).map(|(a, b)| a * b).sum();
        for (f, gq) in fl.iter_mut().zip(&gv) {
            *f -= ki * s * gq;
        }
        fdr[i * 6 + 5] += ki * ul[i * 6 + 5];
        ksum += ki * ul[i * 6 + 5];
    }
    for j in 0..4 {
        // ∂ω/∂u = −½ ∂N/∂y, ∂ω/∂v = ½ ∂N/∂x at the center
        fdr[j * 6] += ksum * 0.5 * p0.by[j];
        fdr[j * 6 + 1] -= ksum * 0.5 * p0.bx[j];
    }
    for q in 0..n {
        fl[q] += fdr[q];
        st[12 + q] = fdr[q];
    }
    // fint = K_std u + correction
    let fstd: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k_std[i * n + j] * u[j]).sum()).collect();
    let corr = vec_to_global(&fl, &loc.r, 8);
    let f: Vec<f64> = fstd.iter().zip(&corr).map(|(a, b)| a + b).collect();
    Ok((f, st, dae))
}

/// The drill work in the increment: ½ (f_old + f_new)·Δu (local).
pub fn drill_work(ctx: &ElemCtx, old: &[f64], new: &[f64], du: &[f64]) -> Result<f64> {
    if old.len() < 12 || old.len() != new.len() {
        return Ok(0.0);
    }
    let loc = local_frame(&ctx.x, ctx.label, "S4R")?;
    let mut w = 0.0;
    for b in 0..(old.len() - 12) / 3 {
        for i in 0..3 {
            let dl: f64 = (0..3).map(|j| loc.r[i][j] * du[3 * b + j]).sum();
            w += 0.5 * (old[12 + 3 * b + i] + new[12 + 3 * b + i]) * dl;
        }
    }
    Ok(w)
}

/// Internal force (global, 18) for S3R with explicit drill: the moment at node
/// i is kᵢθᵢ with kᵢ = 0.2·k_ts/(1 + 0.1·A/t²)·(A/2, A/4, A/4) and the membrane
/// gets −(Σⱼ kⱼθⱼ)·∂ω/∂u (checked in the same way as for S4R;
/// the implicit element's drill penalty, see `elements/shell_tri.rs`, is removed).
/// The explicit state has the same layout as for S4R (drill force from slot 12).
pub fn s3r_force(ctx: &ElemCtx, k_std: &[f64], u: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let loc = local_frame(&ctx.x, ctx.label, "S3R")?;
    let sec = ShellSec::get(ctx, "S3R")?;
    if sec.zoff != 0.0 {
        return crate::model::fail(format!("element {} (S3R): *SHELL SECTION, OFFSET= is not supported in explicit dynamics", ctx.label));
    }
    let n = 18;
    let mut ul = vec![0.0; n];
    for b in 0..6 {
        for i in 0..3 {
            ul[3 * b + i] = (0..3).map(|j| loc.r[i][j] * u[3 * b + j]).sum();
        }
    }
    let (bx, by, a) = crate::elements::shell::tri::cst(&loc.xl);
    let t = sec.t;
    let kts = sec.tss.map(|k| k[0]).unwrap_or(5.0 / 6.0 * sec.g * t);
    let kfac = 0.2 * kts / (1.0 + 0.1 * a / (t * t));
    let g: Vec<Vec<f64>> = (0..3)
        .map(|i| {
            let mut g = vec![0.0; n];
            g[i * 6 + 5] = 1.0;
            for j in 0..3 {
                g[j * 6] += 0.5 * by[j];
                g[j * 6 + 1] -= 0.5 * bx[j];
            }
            g
        })
        .collect();
    let dotu = |v: &[f64]| -> f64 { v.iter().zip(&ul).map(|(a, b)| a * b).sum() };
    let mut fl = vec![0.0; n];
    // implicit: ½kA (g₂² + g₃²) + kA/320 (2g₁ + g₂ + g₃)²
    for i in 1..3 {
        let s = dotu(&g[i]);
        for q in 0..n {
            fl[q] -= 0.5 * kfac * a * s * g[i][q];
        }
    }
    let gm: Vec<f64> = (0..n).map(|q| 2.0 * g[0][q] + g[1][q] + g[2][q]).collect();
    let s = dotu(&gm);
    for q in 0..n {
        fl[q] -= kfac * a / 320.0 * s * gm[q];
    }
    let w = [0.5 * a, 0.25 * a, 0.25 * a];
    let mut fdr = vec![0.0; n];
    let mut ksum = 0.0;
    for i in 0..3 {
        let ki = kfac * w[i];
        fdr[i * 6 + 5] += ki * ul[i * 6 + 5];
        ksum += ki * ul[i * 6 + 5];
    }
    for j in 0..3 {
        fdr[j * 6] += ksum * 0.5 * by[j];
        fdr[j * 6 + 1] -= ksum * 0.5 * bx[j];
    }
    let mut st = vec![0.0; 12 + n];
    for q in 0..n {
        fl[q] += fdr[q];
        st[12 + q] = fdr[q];
    }
    let fstd: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k_std[i * n + j] * u[j]).sum()).collect();
    let corr = vec_to_global(&fl, &loc.r, 6);
    Ok((fstd.iter().zip(&corr).map(|(a, b)| a + b).collect(), st))
}
