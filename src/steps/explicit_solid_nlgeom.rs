//! Område explicit_adv: explicit hourglass-kontroll för reducerat
//! integrerade kontinuumelement (C3D8R, CPE4R, CPS4R) med NLGEOM.
//!
//! Elementets spänningsdel räknas som tidigare (`kinematics::continuum`,
//! töjningsinkrementet i slutkonfigurationen och Jaumann-termen), men
//! Standard-elementets hourglass-styvhet (totalform i referensläget,
//! `elements/solid_higher_nlgeom.rs`) stängs av (`skip_standard`) och ersätts
//! av en inkrementell form: hourglass-förskjutningens ökning
//! Δqₐ = Σᵢ γₐᵢ Δuᵢ med Flanagan–Belytschkos γ i den aktuella geometrin,
//! styvheten c = 0,0125·(2/n)·G·V·Σᵢ|∇Nᵢ|² (som utan NLGEOM, aktuell geometri)
//! och relaxationsformen (RELAX STIFFNESS, standard) eller ren styvhet:
//!
//! ```text
//! Qₑ ← Qₑ + c Δq,   Qᵥ ← Qᵥ e^{−1/τ} + (1 − α) c Δq e^{−1/(2τ)},   Q = α Qₑ + Qᵥ
//! ```
//!
//! (hourglass-delen av det inkrementella förskjutningsfältet; α och τ som i
//! `explicit_solid.rs`).
//! Hourglass-krafterna fᵢ = γₐᵢ Q och ALLAE = Σ ½(Qₙ + Qₙ₊₁) Δq. Tillståndet
//! (Qₑ, Qᵥ, Q) vrids med elementets spinn ΔW före ökningen (Q ← Q + ΔW Q).
//! Kontrollerat med en C3D8R (fast botten, föreskriven hourglass-mod,
//! skjuvning med fria ovannoder, 10 % hoptryckning med och utan plasticitet).
//!
//! Explicit-tillståndet per element är [Qₑ, Qᵥ, Q] för alla (mod, riktning).

use super::dt::geometry;
use super::shell::Hourglass;
use crate::elements::{ElemCtx, Family};
use crate::model::{fail, Result};
use std::cell::Cell;

thread_local! {
    static SKIP: Cell<bool> = const { Cell::new(false) };
}

/// Standard-elementets hourglass-styvhet vid NLGEOM stängs av i den aktuella
/// tråden (explicit-hourglass läggs till här i stället).
pub fn set_skip_standard(on: bool) {
    SKIP.with(|c| c.set(on));
}

pub fn skip_standard() -> bool {
    SKIP.with(|c| c.get())
}

fn base_vectors(nn: usize) -> Vec<Vec<f64>> {
    if nn == 8 {
        let nat: [[f64; 3]; 8] = [[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]];
        vec![
            nat.iter().map(|p| p[0] * p[1]).collect(),
            nat.iter().map(|p| p[1] * p[2]).collect(),
            nat.iter().map(|p| p[2] * p[0]).collect(),
            nat.iter().map(|p| p[0] * p[1] * p[2]).collect(),
        ]
    } else {
        vec![vec![1.0, -1.0, 1.0, -1.0]]
    }
}

/// Lägger till explicit-hourglass i `fint` (elementets frihetsgrader, nd per
/// nod) för slutläget `u` och inkrementet `du`. Returnerar (nytt tillstånd, ΔALLAE).
#[allow(clippy::too_many_arguments)]
pub fn hourglass_force(ctx: &ElemCtx, el: &crate::elements::solid_higher::Solid, family: Family, fint: &mut [f64], u: &[f64], du: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, f64)> {
    let mat = ctx.material(el.name)?;
    let nn = ctx.x.len();
    let nd = if nn == 8 { 3 } else { 2 };
    if u.len() != nn * nd {
        return fail(format!("element {} ({}): oväntat antal frihetsgrader för hourglass", ctx.label, el.name));
    }
    // aktuell geometri vid inkrementets slut
    let x: Vec<[f64; 3]> = (0..nn).map(|i| std::array::from_fn(|k| ctx.x[i][k] + if k < nd { u[i * nd + k] } else { 0.0 })).collect();
    let Some(g) = geometry(family, el.name, &x) else {
        return fail(format!("element {} ({}): kan inte beräkna hourglass-vektorerna", ctx.label, el.name));
    };
    let thick = if nd == 2 { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 };
    let grad: Vec<[f64; 3]> = g.b.iter().map(|b| [b[0] / g.vol, b[1] / g.vol, b[2] / g.vol]).collect();
    let sbb: f64 = grad.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum();
    let gmod = crate::elements::solid_higher::tech::shear_modulus_of(el, mat)?;
    let c = 0.0125 * (2.0 / nd as f64) * gmod * g.vol * thick * sbb;
    // elementets spinn ΔW = skew(Σ Δuᵢ ⊗ ∇Nᵢ): hourglass-krafterna vrids med
    // elementet (avläst: skjuvning med fria ovannoder 4e-2 → 3e-2, stor
    // hoptryckning 5e-3)
    let mut w = [[0.0f64; 3]; 3];
    for i in 0..nn {
        for a in 0..nd {
            for b in 0..nd {
                w[a][b] += 0.5 * (du[i * nd + a] * grad[i][b] - du[i * nd + b] * grad[i][a]);
            }
        }
    }
    let mut oldr = old.to_vec();
    if old.len() >= 3 * (base_vectors(nn).len() * nd) {
        let nmm = base_vectors(nn).len() * nd;
        for blk in 0..3 {
            for a in 0..base_vectors(nn).len() {
                let v: Vec<f64> = (0..nd).map(|d| old[blk * nmm + a * nd + d]).collect();
                for d in 0..nd {
                    oldr[blk * nmm + a * nd + d] = v[d] + (0..nd).map(|k| w[d][k] * v[k]).sum::<f64>();
                }
            }
        }
    }
    let old = &oldr[..];
    let hs = base_vectors(nn);
    let nm = hs.len() * nd;
    let mut st = vec![0.0; 3 * nm];
    let mut dae = 0.0;
    let alpha = super::shell::RELAX_ALPHA;
    let tau = super::solid::RELAX_TAU_SOLID;
    for (a, h) in hs.iter().enumerate() {
        // γ = h − (h·x) ∇N (Flanagan–Belytschko, aktuell geometri)
        let hx: [f64; 3] = std::array::from_fn(|k| (0..nn).map(|i| h[i] * x[i][k]).sum());
        let gam: Vec<f64> = (0..nn).map(|i| h[i] - (0..nd).map(|k| hx[k] * grad[i][k]).sum::<f64>()).collect();
        for d in 0..nd {
            let m = a * nd + d;
            let dq: f64 = (0..nn).map(|i| gam[i] * du[i * nd + d]).sum();
            let qe0 = old.get(m).copied().unwrap_or(0.0);
            let qv0 = old.get(nm + m).copied().unwrap_or(0.0);
            let q0 = old.get(2 * nm + m).copied().unwrap_or(0.0);
            let qe = qe0 + c * dq;
            let (qn, qv) = match mode {
                Hourglass::Stiffness => (qe, 0.0),
                Hourglass::Relax => {
                    let qv = qv0 * (-1.0 / tau).exp() + (1.0 - alpha) * c * dq * (-0.5 / tau).exp();
                    (alpha * qe + qv, qv)
                }
            };
            st[m] = qe;
            st[nm + m] = qv;
            st[2 * nm + m] = qn;
            dae += 0.5 * (qn + q0) * dq;
            for i in 0..nn {
                fint[i * nd + d] += gam[i] * qn;
            }
        }
    }
    Ok((st, dae))
}
