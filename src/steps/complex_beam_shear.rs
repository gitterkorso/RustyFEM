//! The transverse shear stiffness of Timoshenko beams is left out of the
//! stiffness-proportional (*DAMPING, BETA) and the structural
//! (*DAMPING, STRUCTURAL) damping in the projected steps
//! (*COMPLEX FREQUENCY, *STEADY STATE DYNAMICS, SUBSPACE PROJECTION,
//! *DYNAMIC, SUBSPACE). `subtract` subtracts β·Φᵀ K_γ Φ from Cᵣ and
//! s·Φᵀ K_γ Φ from Sᵣ, where K_γ is the shear part of the element stiffness
//! (`elements::beam_timoshenko::damping`). Included with `#[path]` from
//! `complex_frequency.rs`.

use crate::steps::modal::basis::ModalBasis;
use crate::steps::StepCtx;
use crate::elements::beam_timoshenko::damping::shear_stiffness_matrix;
use crate::fem::assembly::NONE;
use crate::fem::model::parse_f64;
use crate::model::{FemError, Result};

/// (β, s) from the material's *DAMPING.
fn beta_structural(ctx: &StepCtx, mi: Option<usize>) -> Result<(f64, f64)> {
    let m = ctx.prep.model;
    let Some(kw) = mi.and_then(|i| m.materials[i].behavior("DAMPING")) else { return Ok((0.0, 0.0)) };
    let num = |k: &str| -> Result<f64> {
        match kw.param(k) {
            Some(v) => parse_f64(v).ok_or_else(|| FemError(format!("{}: {k}={v} is not a number", kw.location()))),
            None => Ok(0.0),
        }
    };
    Ok((num("BETA")?, num("STRUCTURAL")?))
}

/// Subtracts the shear part of the beam stiffness from Cᵣ and Sᵣ (n × n, row-wise).
pub fn subtract(ctx: &StepCtx, b: &ModalBasis, cr: &mut [f64], sr: &mut [f64]) -> Result<()> {
    let p = ctx.prep;
    let m = p.model;
    let n = b.len();
    let ns = b.all.n_solve();
    for (e, el) in m.elements.iter().enumerate() {
        let mi = el.section.and_then(|s| m.sections[s].material_index);
        let (br, s) = beta_structural(ctx, mi)?;
        if br == 0.0 && s == 0.0 {
            continue;
        }
        let ectx = p.ctx(e, false, 0.0, 0.0);
        let Some(ks) = shear_stiffness_matrix(el.etype.name(), &ectx) else { continue };
        let ks = ks.map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        let map: Vec<usize> = p.edofs[e].iter().map(|&(nd, sl)| b.all.idx[nd][sl]).collect();
        let nd = map.len();
        // projection: Φᵀ K_γ Φ over the element's degrees of freedom
        let phi: Vec<Vec<f64>> = (0..n).map(|i| map.iter().map(|&g| if g == NONE || g >= ns { 0.0 } else { b.x[i][g] }).collect()).collect();
        let kp: Vec<Vec<f64>> = phi.iter().map(|v| (0..nd).map(|a| (0..nd).map(|c| ks[a * nd + c] * v[c]).sum()).collect()).collect();
        for i in 0..n {
            for j in 0..n {
                let v: f64 = phi[i].iter().zip(&kp[j]).map(|(x, y)| x * y).sum();
                cr[i * n + j] -= br * v;
                sr[i * n + j] -= s * v;
            }
        }
    }
    Ok(())
}
