//! S4R i explicit dynamik (NLGEOM=NO).
//!
//! Det explicita elementet är samma S4R som det implicita (membran och böjning
//! i en punkt, antagen tvärskjuvning, se `elements/shell.rs`) utom på tre
//! punkter (kontrollerade med reaktionskrafter för föreskrivna
//! förskjutningsramper i varje frihetsgrad, kvadrat, parallellogram,
//! t = 0,01 och 0,001):
//!
//! - Membranets hourglass-styvhet c_m = 0,0125·G·t·A·Σᵢ|∇Nᵢ|² (Standard
//!   0,0003125) och böjningens c_b = 0,0125·G·t³·A·Σᵢ|∇Nᵢ|² (Standard
//!   0,01125·G·t³/12·A·Σᵢ|∇Nᵢ|²), med Flanagan–Belytschkos γ i båda.
//! - Hourglass-kraften är som standard ("relax stiffness") integralviskoelastisk
//!   med en kärna per inkrement: Q(n) = c·(0,1·q(n) + 0,9·Σₖ e^{−(n−k−½)/500} Δq(k)),
//!   där q = γ·u är modamplituden. Kärnan är avläst ur reaktionen för en
//!   föreskriven ramp i membranets hourglass-mod (identisk för Δt = 10⁻⁸ och
//!   10⁻⁷, alltså per inkrement och inte per tid; 5 siffror över 20–20 000
//!   inkrement). `*SECTION CONTROLS, HOURGLASS=STIFFNESS` ger Q = c·q.
//! - Drill: momentet i nod i är kᵢθᵢ (kᵢ = 0,2·k_ts/(1 + 0,1·A/t²)·det Jᵢ som
//!   i Standard) och membranet får kraften −(Σⱼ kⱼθⱼ)·∂ω/∂u, ω = ½(v,x − u,y)
//!   i elementets mitt. Kopplingen är alltså osymmetrisk (inget kᵢω²-bidrag
//!   till membranet).
//!
//! Implementationen tar Standard-elementets styvhet (`ElementType::stiffness`)
//! och byter ut de tre delarna i elementets lokala system. Elementets
//! explicit-tillstånd (i drivern, inte elementets `state`) är
//! [q (4), Q_v (4), Q (4), drillkraft (24)] för modernas amplituder
//! (membran x, y, böjning θx, θy), den viskösa delen och föregående krafter
//! (för hourglass- och drillenergin ALLAE).

use crate::elements::shell::{local_frame, q4pt, vec_to_global, ShellSec};
use crate::elements::ElemCtx;
use crate::model::Result;

/// Antal värden i explicit-tillståndet per S4R.
pub const STATE_LEN: usize = 36;

/// Hourglass-kontroll (*SECTION CONTROLS, HOURGLASS=).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hourglass {
    Relax,
    Stiffness,
}

/// Relaxationskärnans parametrar (andel som inte relaxerar, tidskonstant i inkrement).
pub const RELAX_ALPHA: f64 = 0.1;
pub const RELAX_TAU: f64 = 500.0;

/// Ny hourglass-kraft för modamplituden `q` med förra värdet `q0` och
/// den viskösa delen `qv0`: (Q, Q_v).
pub fn relax(mode: Hourglass, c: f64, q: f64, q0: f64, qv0: f64) -> (f64, f64) {
    match mode {
        Hourglass::Stiffness => (c * q, 0.0),
        Hourglass::Relax => {
            let qv = qv0 * (-1.0 / RELAX_TAU).exp() + (1.0 - RELAX_ALPHA) * (q - q0) * (-0.5 / RELAX_TAU).exp();
            (c * (RELAX_ALPHA * q + qv), qv)
        }
    }
}

/// Inre kraft (globalt, 24) för S4R med explicit hourglass och drill.
/// Returnerar (kraft, nytt explicit-tillstånd, ökning av ALLAE).
pub fn s4r_force(ctx: &ElemCtx, k_std: &[f64], u: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, Vec<f64>, f64)> {
    let loc = local_frame(&ctx.x, ctx.label, "S4R")?;
    let sec = ShellSec::get(ctx, "S4R")?;
    let n = 24;
    // lokala förskjutningar
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
    // Standard: cm = 0,0003125 G t A Σb², cb = 0,01125 G t³/12 A Σb²
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
    // drill: Standard Σ kᵢ (θᵢ − ωᵢ)² bort, den explicita formen in
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
        // ∂ω/∂u = −½ ∂N/∂y, ∂ω/∂v = ½ ∂N/∂x i mitten
        fdr[j * 6] += ksum * 0.5 * p0.by[j];
        fdr[j * 6 + 1] -= ksum * 0.5 * p0.bx[j];
    }
    for q in 0..n {
        fl[q] += fdr[q];
        st[12 + q] = fdr[q];
    }
    // fint = K_std u + korrektion
    let fstd: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k_std[i * n + j] * u[j]).sum()).collect();
    let corr = vec_to_global(&fl, &loc.r, 8);
    let f: Vec<f64> = fstd.iter().zip(&corr).map(|(a, b)| a + b).collect();
    Ok((f, st, dae))
}

/// Drillens arbete i inkrementet: ½ (f_gammal + f_ny)·Δu (lokalt).
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

/// Inre kraft (globalt, 18) för S3R med explicit drill: momentet i nod
/// i är kᵢθᵢ med kᵢ = 0,2·k_ts/(1 + 0,1·A/t²)·(A/2, A/4, A/4) och membranet
/// får −(Σⱼ kⱼθⱼ)·∂ω/∂u (kontrollerat på samma sätt som för S4R;
/// Standard-elementets drillstraff, se `elements/shell_tri.rs`, tas bort).
/// Explicit-tillståndet har samma layout som för S4R (drillkraften från plats 12).
pub fn s3r_force(ctx: &ElemCtx, k_std: &[f64], u: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    let loc = local_frame(&ctx.x, ctx.label, "S3R")?;
    let sec = ShellSec::get(ctx, "S3R")?;
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
    // Standard: ½kA (g₂² + g₃²) + kA/320 (2g₁ + g₂ + g₃)²
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
