//! Analytical rigid surfaces: *SURFACE, TYPE=SEGMENTS, CYLINDER or REVOLUTION
//! (called from `surface.rs`, which owns the *SURFACE keyword).
//!
//! The surface is registered by name in `FeModel::surfaces` (the type in `kind`, no
//! element faces) and the keyword is stored in `FeModel::extra`, where
//! `interactions/analytical.rs` reads the geometry. Inside a *PART the
//! keyword is marked with `_PART`; *INSTANCE then adds the instance name and
//! placement (`_INSTANCE`, `_PLACE`, see `assembly.rs`).

use crate::fem::model::{RawKeyword, Surface};
use crate::inp::{InpCtx, Keyword};
use crate::model::Result;

/// No keywords of its own (*SURFACE is registered in `surface.rs`).
pub static KEYWORDS: &[Keyword] = &[];

/// TYPE= for analytical surfaces.
pub fn is_analytical_type(kind: &str) -> bool {
    matches!(kind, "SEGMENTS" | "CYLINDER" | "REVOLUTION")
}

pub fn surface(ctx: &mut InpCtx, kw: &RawKeyword, name: &str) -> Result<()> {
    let kind = kw.param_upper("TYPE").unwrap_or_default();
    if kw.data.iter().all(|d| d.fields.iter().all(|f| f.trim().is_empty())) {
        return kw.err("the analytical surface has no data lines (START, LINE, CIRCL, PARAB)");
    }
    let mut k = kw.clone();
    if let Some((part, _)) = &ctx.part {
        k.params.push(("_PART".into(), part.clone()));
    }
    let sf = Surface { kind, source: kw.location(), ..Default::default() };
    let mesh = ctx.mesh();
    if mesh.surfaces.insert(name.to_string(), sf).is_some() {
        return kw.err(format!("surface {name} is defined twice"));
    }
    mesh.extra.push(k);
    Ok(())
}
