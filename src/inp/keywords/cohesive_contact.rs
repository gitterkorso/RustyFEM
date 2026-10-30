//! Område cavity: kohesiv kontakt (*COHESIVE BEHAVIOR under *SURFACE INTERACTION).
//!
//! *COHESIVE BEHAVIOR sparas i `FeModel::extra` efter sin *SURFACE INTERACTION.
//! *DAMAGE INITIATION, *DAMAGE EVOLUTION och *DAMAGE STABILIZATION är
//! materialbeteenden, men när de följer *COHESIVE BEHAVIOR (utan *MATERIAL
//! emellan) sparas de i stället i `FeModel::extra` (`surface_damage`, anropas
//! från `inp::dispatch`). Kontakten räknas i `interactions/cohesive_contact.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::{store_model, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "COHESIVE BEHAVIOR", kind: Kind::Model, handler: cohesive_behavior }];

/// Underalternativ som hör till en kohesiv kontaktegenskap.
pub const DAMAGE_OPTIONS: &[&str] = &["DAMAGE INITIATION", "DAMAGE EVOLUTION", "DAMAGE STABILIZATION"];

fn cohesive_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if !ctx.mesh().extra.last().is_some_and(|k| crate::interactions::cohesive_contact::SI_OPTIONS.contains(&k.name.as_str()) || k.name == "SURFACE INTERACTION") {
        return kw.err("*COHESIVE BEHAVIOR ska stå under *SURFACE INTERACTION");
    }
    store_model(ctx, kw)
}

/// Sparar ett skadenyckelord som följer *COHESIVE BEHAVIOR. Sant om det togs om hand.
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
