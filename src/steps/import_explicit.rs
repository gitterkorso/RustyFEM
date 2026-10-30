//! Area gaps1: *IMPORT from explicit to implicit analysis
//! (included from `restart.rs`).
//!
//! The state from an explicit analysis is in dynamic, not static, equilibrium.
//! The out-of-balance forces F0 = ∫Bᵀσ dV (with the
//! imported configuration and state) are therefore applied as an external load and
//! ramped down linearly during the first step; at the end of the step the residual stresses that
//! belong to static equilibrium remain. The same mechanism as for initial stresses
//! (`materials::initial_state::UNBALANCE`, `fem/system.rs`, `steps/nlgeom.rs`).
//!
//! The velocities are not imported into a static step.

use crate::fem::assembly::Prepared;
use crate::fem::dof::SLOTS;
use crate::fem::state::SolutionState;
use crate::model::{FemError, Result};

/// *IMPORT, UPDATE=YES: the imported configuration becomes the new
/// reference configuration. The node coordinates in the old input text
/// (the *NODE blocks in definition order, one node per data line) are moved by
/// the displacements `u` (node index in the same order); the displacements
/// are then measured from the new configuration. The element states (stress,
/// material state and the integrated strain that the materials work
/// with) are kept unchanged.
pub fn move_nodes(blocks: &mut [crate::fem::model::RawKeyword], u: &[crate::fem::dof::NodeVec]) -> Result<()> {
    let mut k = 0usize;
    for b in blocks.iter_mut().filter(|b| b.name == "NODE") {
        if b.has("INPUT") || b.has("SYSTEM") {
            return b.err("*IMPORT, UPDATE=YES: *NODE with INPUT= or SYSTEM= is not supported");
        }
        for d in b.data.iter_mut() {
            if d.fields.first().is_none_or(|f| f.trim().is_empty()) {
                continue;
            }
            let Some(du) = u.get(k) else { return crate::model::fail("*IMPORT, UPDATE=YES: more nodes in the input text than in the restart file") };
            for c in 0..3 {
                let x = d.fields.get(c + 1).and_then(|s| crate::fem::model::parse_f64(s)).unwrap_or(0.0);
                let v = format!("{:.17e}", x + du[c]);
                if d.fields.len() > c + 1 {
                    d.fields[c + 1] = v;
                } else {
                    d.fields.push(v);
                }
            }
            k += 1;
        }
    }
    if k != u.len() {
        return crate::model::fail(format!("*IMPORT, UPDATE=YES: {k} nodes in the input text but {} in the restart file", u.len()));
    }
    Ok(())
}

/// The old analysis was explicit (*DYNAMIC, EXPLICIT).
pub fn deck_is_explicit(deck: &str) -> bool {
    deck.lines().any(|l| {
        let u = l.trim_start().to_uppercase();
        u.starts_with("*DYNAMIC") && u.contains("EXPLICIT")
    })
}

/// The out-of-balance forces F0 for the imported state; stored in
/// `state.vectors` under `initial_state::UNBALANCE`.
pub fn set_unbalance(prep: &Prepared, state: &mut SolutionState) -> Result<()> {
    let model = prep.model;
    let mut f0 = vec![[0.0; SLOTS]; model.nodes.len()];
    let nlgeom = state.nlgeom;
    for e in 0..model.elements.len() {
        let ctx = prep.ctx(e, nlgeom, 0.0, 0.0);
        if ctx.material.is_none() || state.elem_state[e].is_empty() {
            continue;
        }
        let et = model.elements[e].etype;
        let ue: Vec<f64> = prep.edofs[e].iter().map(|&(n, s)| state.u[n][s]).collect();
        let z = vec![0.0; ue.len()];
        let mut sn = Vec::new();
        let r = et.internal_force(&ctx, &ue, &z, &state.elem_state[e], &mut sn, false).map_err(|err| FemError(format!("{}: {}", model.elem_name(e), err.0)))?;
        for (k, &(n, s)) in prep.edofs[e].iter().enumerate() {
            f0[n][s] += r.fint[k];
        }
    }
    state.vectors.insert(crate::materials::initial_state::UNBALANCE.to_string(), f0);
    Ok(())
}
