//! Area cavity: cohesive contact (*COHESIVE BEHAVIOR under *SURFACE INTERACTION).
//!
//! *COHESIVE BEHAVIOR is stored in `FeModel::extra` after its *SURFACE INTERACTION.
//! *DAMAGE INITIATION, *DAMAGE EVOLUTION and *DAMAGE STABILIZATION are
//! material behaviors, but when they follow *COHESIVE BEHAVIOR (with no *MATERIAL
//! in between) they are instead stored in `FeModel::extra` (`surface_damage`, called
//! from `inp::dispatch`). The contact is computed in `interactions/cohesive_contact.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::{store_model, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "COHESIVE BEHAVIOR", kind: Kind::Model, handler: cohesive_behavior }];

/// Suboptions that belong to a cohesive contact property.
pub const DAMAGE_OPTIONS: &[&str] = &["DAMAGE INITIATION", "DAMAGE EVOLUTION", "DAMAGE STABILIZATION"];

fn cohesive_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.mesh().extra.last().is_some_and(|k| crate::interactions::cohesive_contact::SI_OPTIONS.contains(&k.name.as_str()) || k.name == "SURFACE INTERACTION") {
        return kw.err("*COHESIVE BEHAVIOR must appear under *SURFACE INTERACTION");
    }
    store_model(ctx, kw)
}

/// Stores a damage keyword that follows *COHESIVE BEHAVIOR. True if it was handled.
pub fn surface_damage(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<bool> {
    if !DAMAGE_OPTIONS.contains(&kw.name.as_str()) || ctx.step.is_some() {
        return Ok(false);
    }
    let after = ctx.mesh().extra.last().is_some_and(|k| k.name == "COHESIVE BEHAVIOR" || DAMAGE_OPTIONS.contains(&k.name.as_str()));
    if !after {
        return Ok(false);
    }
    store_model(ctx, kw)?;
    Ok(true)
}
