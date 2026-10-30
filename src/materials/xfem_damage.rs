//! Material för XFEM-sprickor: grundmaterialet (t.ex. isotrop elasticitet)
//! med *DAMAGE INITIATION, CRITERION=MAXPS/MAXPE/MAXS/QUADS/MAXE/QUADE,
//! *DAMAGE EVOLUTION och *DAMAGE STABILIZATION.
//!
//! Skadebeteendena gäller sprickytorna i de berikade elementen och läses av
//! `elements/xfem.rs`; själva kontinuumet använder materialet utan dem.

use super::{MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "XFEM-spricka (grundmaterial med skadeinitiering på sprickytan)",
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

/// Kriterier som XFEM-sprickorna använder.
pub const CRITERIA: &[&str] = &["MAXPS", "MAXPE", "MAXS", "QUADS", "MAXE", "QUADE"];

/// Sant om materialet är ett XFEM-material: kontinuumelasticitet (inte
/// TYPE=TRACTION) med ett XFEM-kriterium.
pub fn claims(def: &MaterialDef) -> bool {
    let traction = def.behavior("ELASTIC").is_some_and(|k| k.param_upper("TYPE").as_deref() == Some("TRACTION"));
    // MAXPS/MAXPE finns bara för XFEM; tractionkriterierna (som också används av
    // kohesiva element) räknas som XFEM bara med en *DAMAGE EVOLUTION, annars ger
    // materials/cohesive.rs felet om TYPE=TRACTION
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
                return b.err("*DAMAGE STABILIZATION kräver *DAMAGE INITIATION och *DAMAGE EVOLUTION");
            }
        }
        return Ok(None);
    }
    let mut base = def.clone();
    base.behaviors.retain(|b| !matches!(b.name.as_str(), "DAMAGE INITIATION" | "DAMAGE EVOLUTION" | "DAMAGE STABILIZATION"));
    Ok(Some(super::build(&base)?))
}
