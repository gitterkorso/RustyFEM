//! Material for XFEM cracks: the base material (e.g. isotropic elasticity)
//! with *DAMAGE INITIATION, CRITERION=MAXPS/MAXPE/MAXS/QUADS/MAXE/QUADE,
//! *DAMAGE EVOLUTION and *DAMAGE STABILIZATION.
//!
//! The damage behaviors apply to the crack surfaces in the enriched elements and are read by
//! `elements/xfem.rs`; the continuum itself uses the material without them.

use super::{MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "XFEM crack (base material with damage initiation on the crack surface)",
    consumes: &[
        "CREEP",
        "DAMAGE EVOLUTION",
        "DAMAGE INITIATION",
        "DAMAGE STABILIZATION",
        "DRUCKER PRAGER",
        "DRUCKER PRAGER HARDENING",
        "ELASTIC",
        "EXPANSION",
        "PLASTIC",
        "POTENTIAL",
        "RATE DEPENDENT",
        "VISCOELASTIC",
    ],
    priority: 120,
    build: Some(build),
}];

/// Criteria used by the XFEM cracks.
pub const CRITERIA: &[&str] = &["MAXPS", "MAXPE", "MAXS", "QUADS", "MAXE", "QUADE"];

/// True if the material is an XFEM material: continuum elasticity (not
/// TYPE=TRACTION) with an XFEM criterion.
pub fn claims(def: &MaterialDef) -> bool {
    let traction = def.behavior("ELASTIC").is_some_and(|k| k.param_upper("TYPE").as_deref() == Some("TRACTION"));
    // MAXPS/MAXPE exist only for XFEM; the traction criteria (which are also used by
    // cohesive elements) count as XFEM only with a *DAMAGE EVOLUTION, otherwise
    // materials/cohesive.rs reports the TYPE=TRACTION error
    let evolution = def.behavior("DAMAGE EVOLUTION").is_some();
    !traction
        && def.behaviors.iter().any(|b| {
            b.name == "DAMAGE INITIATION"
                && b.param_upper("CRITERION").is_some_and(|c| matches!(c.as_str(), "MAXPS" | "MAXPE") || (evolution && CRITERIA.contains(&c.as_str())))
        })
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        if let Some(b) = def.behavior("DAMAGE STABILIZATION") {
            if def.behavior("DAMAGE INITIATION").is_none() {
                return b.err("*DAMAGE STABILIZATION requires *DAMAGE INITIATION and *DAMAGE EVOLUTION");
            }
        }
        return Ok(None);
    }
    let mut base = def.clone();
    base.behaviors.retain(|b| !matches!(b.name.as_str(), "DAMAGE INITIATION" | "DAMAGE EVOLUTION" | "DAMAGE STABILIZATION"));
    Ok(Some(super::build(&base)?))
}
