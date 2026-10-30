//! Timoshenkobalkarnas tvärskjuvstyvhet utelämnas ur den
//! styvhetsproportionella (*DAMPING, BETA) och den strukturella
//! (*DAMPING, STRUCTURAL) dämpningen i de projicerade stegen
//! (*COMPLEX FREQUENCY, *STEADY STATE DYNAMICS, SUBSPACE PROJECTION,
//! *DYNAMIC, SUBSPACE). `subtract` drar av β·Φᵀ K_γ Φ från Cᵣ och
//! s·Φᵀ K_γ Φ från Sᵣ, där K_γ är skjuvdelen av elementstyvheten
//! (`elements::beam_timoshenko::damping`). Inkluderas med `#[path]` från
//! `complex_frequency.rs`.

use crate::steps::modal::basis::ModalBasis;
use crate::steps::StepCtx;
use crate::elements::beam_timoshenko::damping::shear_stiffness_matrix;
use crate::fem::assembly::NONE;
use crate::fem::model::parse_f64;
use crate::model::{FemError, Result};

/// (β, s) ur materialets *DAMPING.
fn beta_structural(ctx: &StepCtx, mi: Option<usize>) -> Result<(f64, f64)> {
    let m = ctx.prep.model;
    let Some(kw) = mi.and_then(|i| m.materials[i].behavior("DAMPING")) else { return Ok((0.0, 0.0)) };
    let num = |k: &str| -> Result<f64> {
        match kw.param(k) {
            Some(v) => parse_f64(v).ok_or_else(|| FemError(format!("{}: {k}={v} är inget tal", kw.location()))),
            None => Ok(0.0),
        }
    };
    Ok((num("BETA")?, num("STRUCTURAL")?))
}

/// Drar av skjuvdelen av balkarnas styvhet ur Cᵣ och Sᵣ (n × n, radvis).
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
        // projektion: Φᵀ K_γ Φ över elementets frihetsgrader
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
