//! Analytiska stela ytor: *SURFACE, TYPE=SEGMENTS, CYLINDER eller REVOLUTION
//! (anropas från `surface.rs`, som äger nyckelordet *SURFACE).
//!
//! Ytan registreras med sitt namn i `FeModel::surfaces` (typen i `kind`, inga
//! elementytor) och nyckelordet sparas i `FeModel::extra`, där
//! `interactions/analytical.rs` läser geometrin. Inne i en *PART markeras
//! nyckelordet med `_PART`; *INSTANCE lägger då till instansens namn och
//! placering (`_INSTANCE`, `_PLACE`, se `assembly.rs`).

use crate::fem::model::{RawKeyword, Surface};
use crate::inp::{InpCtx, Keyword};
use crate::model::Result;

/// Inga egna nyckelord (*SURFACE registreras i `surface.rs`).
pub static KEYWORDS: &[Keyword] = &[];

/// TYPE= för analytiska ytor.
pub fn is_analytical_type(kind: &str) -> bool {
    matches!(kind, "SEGMENTS" | "CYLINDER" | "REVOLUTION")
}

pub fn surface(ctx: &mut InpCtx, kw: &RawKeyword, name: &str) -> Result<()> {
    let kind = kw.param_upper("TYPE").unwrap_or_default();
    if kw.data.iter().all(|d| d.fields.iter().all(|f| f.trim().is_empty())) {
        return kw.err("den analytiska ytan saknar datarader (START, LINE, CIRCL, PARAB)");
    }
    let mut k = kw.clone();
    if let Some((part, _)) = &ctx.part {
        k.params.push(("_PART".into(), part.clone()));
    }
    let sf = Surface { kind, source: kw.location(), ..Default::default() };
    let mesh = ctx.mesh();
    if mesh.surfaces.insert(name.to_string(), sf).is_some() {
        return kw.err(format!("ytan {name} är definierad två gånger"));
    }
    mesh.extra.push(k);
    Ok(())
}
