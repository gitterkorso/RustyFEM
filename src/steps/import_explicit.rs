//! Område gaps1: *IMPORT från explicit till implicit analys
//! (inkluderas från `restart.rs`).
//!
//! Tillståndet från en explicit analys är i dynamisk, inte statisk, jämvikt.
//! Obalanskrafterna F0 = ∫Bᵀσ dV (med det
//! importerade läget och tillståndet) läggs därför på som yttre last och
//! avvecklas linjärt under första steget; i stegets slut återstår de residualspänningar som
//! hör till statisk jämvikt. Samma mekanism som för begynnelsespänningar
//! (`materials::initial_state::UNBALANCE`, `fem/system.rs`, `steps/nlgeom.rs`).
//!
//! Hastigheterna importeras inte till ett statiskt steg.

use crate::fem::assembly::Prepared;
use crate::fem::dof::SLOTS;
use crate::fem::state::SolutionState;
use crate::model::{FemError, Result};

/// *IMPORT, UPDATE=YES: den importerade konfigurationen blir ny
/// referenskonfiguration. Nodernas koordinater i den gamla indatatexten
/// (*NODE-blocken i definitionsordning, en nod per datarad) flyttas med
/// förskjutningarna `u` (nodindex i samma ordning); förskjutningarna
/// räknas sedan från den nya konfigurationen. Elementtillstånden (spänning,
/// materialtillstånd och den integrerade töjningen som materialen räknar
/// med) behålls oförändrade.
pub fn move_nodes(blocks: &mut [crate::fem::model::RawKeyword], u: &[crate::fem::dof::NodeVec]) -> Result<()> {
    let mut k = 0usize;
    for b in blocks.iter_mut().filter(|b| b.name == "NODE") {
        if b.has("INPUT") || b.has("SYSTEM") {
            return b.err("*IMPORT, UPDATE=YES: *NODE med INPUT= eller SYSTEM= stöds inte");
        }
        for d in b.data.iter_mut() {
            if d.fields.first().is_none_or(|f| f.trim().is_empty()) {
                continue;
            }
            let Some(du) = u.get(k) else { return crate::model::fail("*IMPORT, UPDATE=YES: fler noder i indatatexten än i omstartfilen") };
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
        return crate::model::fail(format!("*IMPORT, UPDATE=YES: {k} noder i indatatexten men {} i omstartfilen", u.len()));
    }
    Ok(())
}

/// Den gamla analysen var explicit (*DYNAMIC, EXPLICIT).
pub fn deck_is_explicit(deck: &str) -> bool {
    deck.lines().any(|l| {
        let u = l.trim_start().to_uppercase();
        u.starts_with("*DYNAMIC") && u.contains("EXPLICIT")
    })
}

/// Obalanskrafterna F0 för det importerade tillståndet; sparas i
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
