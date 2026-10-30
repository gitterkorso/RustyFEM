//! Area explicit_adv: explicit hourglass control for reduced-integration
//! continuum elements (C3D8R, CPE4R, CPS4R) with NLGEOM.
//!
//! The element's stress part is computed as before (`kinematics::continuum`,
//! the strain increment in the final configuration and the Jaumann term), but
//! the implicit element's hourglass stiffness (total form in the reference configuration,
//! `elements/solid_higher_nlgeom.rs`) is switched off (`skip_standard`) and replaced
//! by an incremental form: the increment of the hourglass displacement
//! Δqₐ = Σᵢ γₐᵢ Δuᵢ with the Flanagan–Belytschko γ in the current geometry,
//! the stiffness c = 0.0125·(2/n)·G·V·Σᵢ|∇Nᵢ|² (as without NLGEOM, current geometry)
//! and the relaxation form (RELAX STIFFNESS, default) or pure stiffness:
//!
//! ```text
//! Qₑ ← Qₑ + c Δq,   Qᵥ ← Qᵥ e^{−1/τ} + (1 − α) c Δq e^{−1/(2τ)},   Q = α Qₑ + Qᵥ
//! ```
//!
//! (the hourglass part of the incremental displacement field; α and τ as in
//! `explicit_solid.rs`).
//! The hourglass forces fᵢ = γₐᵢ Q and ALLAE = Σ ½(Qₙ + Qₙ₊₁) Δq. The state
//! (Qₑ, Qᵥ, Q) is rotated with the element spin ΔW before the increment (Q ← Q + ΔW Q).
//! Checked with a C3D8R (fixed bottom, prescribed hourglass mode,
//! shear with free top nodes, 10 % compression with and without plasticity).
//!
//! The explicit state per element is [Qₑ, Qᵥ, Q] for all (mode, direction).

use super::dt::geometry;
use super::shell::Hourglass;
use crate::elements::{ElemCtx, Family};
use crate::model::{fail, Result};
use std::cell::Cell;

thread_local! {
    static SKIP: Cell<bool> = const { Cell::new(false) };
}

/// The implicit element's hourglass stiffness with NLGEOM is switched off in the current
/// thread (explicit hourglass is added here instead).
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

/// Adds explicit hourglass to `fint` (the element's degrees of freedom, nd per
/// node) for the final state `u` and the increment `du`. Returns (new state, ΔALLAE).
#[allow(clippy::too_many_arguments)]
pub fn hourglass_force(ctx: &ElemCtx, el: &crate::elements::solid_higher::Solid, family: Family, fint: &mut [f64], u: &[f64], du: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, f64)> {
    let mat = ctx.material(el.name)?;
    let nn = ctx.x.len();
    let nd = if nn == 8 { 3 } else { 2 };
    if u.len() != nn * nd {
        return fail(format!("element {} ({}): unexpected number of degrees of freedom for hourglass", ctx.label, el.name));
    }
    // current geometry at the end of the increment
    let x: Vec<[f64; 3]> = (0..nn).map(|i| std::array::from_fn(|k| ctx.x[i][k] + if k < nd { u[i * nd + k] } else { 0.0 })).collect();
    let Some(g) = geometry(family, el.name, &x) else {
        return fail(format!("element {} ({}): cannot compute the hourglass vectors", ctx.label, el.name));
    };
    let thick = if nd == 2 { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 };
    let grad: Vec<[f64; 3]> = g.b.iter().map(|b| [b[0] / g.vol, b[1] / g.vol, b[2] / g.vol]).collect();
    let sbb: f64 = grad.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum();
    let gmod = crate::elements::solid_higher::tech::shear_modulus_of(el, mat)?;
    let c = 0.0125 * (2.0 / nd as f64) * gmod * g.vol * thick * sbb;
    // the element spin ΔW = skew(Σ Δuᵢ ⊗ ∇Nᵢ): the hourglass forces rotate with
    // the element (observed: shear with free top nodes 4e-2 → 3e-2, large
    // compression 5e-3)
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
        // γ = h − (h·x) ∇N (Flanagan–Belytschko, current geometry)
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
