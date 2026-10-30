//! Nyckelord för akustik.
//!
//! - *ACOUSTIC MEDIUM: materialbeteende (BULK MODULUS, VOLUMETRIC DRAG), får
//!   förekomma flera gånger i ett material (`materials/acoustic.rs`).
//! - *IMPEDANCE PROPERTY: modelldata, sparas i `FeModel::extra`.
//! - *IMPEDANCE (elementytor In) och *SIMPEDANCE (ytor): impedans eller
//!   icke-reflekterande rand (NONREFLECTING=PLANAR/CIRCULAR/SPHERICAL). De är
//!   stegdata; utanför ett steg sparas de i modellen (gäller alla
//!   steg). Läses av `steps/acoustic.rs`.
//!
//! Akustisk–strukturell koppling görs med *TIE mellan en akustisk och en
//! strukturell yta (`interactions/acoustic.rs`).

use crate::fem::model::RawKeyword;
use crate::inp::{store_model, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ACOUSTIC MEDIUM", kind: Kind::Behavior, handler: acoustic_medium },
    Keyword { name: "IMPEDANCE", kind: Kind::Both, handler: impedance },
    Keyword { name: "IMPEDANCE PROPERTY", kind: Kind::Model, handler: store_model },
    Keyword { name: "SIMPEDANCE", kind: Kind::Both, handler: impedance },
];

/// Materialbeteende som får upprepas (en post per parameter).
fn acoustic_medium(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    // område gaps1: en post per slag (BULK MODULUS, VOLUMETRIC DRAG, COMPLEX ..., POROUS MODEL, CAVITATION LIMIT)
    let kind = |b: &RawKeyword| -> &'static str {
        ["VOLUMETRIC DRAG", "COMPLEX BULK MODULUS", "COMPLEX DENSITY", "POROUS MODEL", "CAVITATION LIMIT"].into_iter().find(|p| b.has(p)).unwrap_or("BULK MODULUS")
    };
    let this = kind(kw);
    let m = ctx.material_mut(kw)?;
    if m.behaviors.iter().any(|b| b.name == kw.name && kind(b) == this) {
        return kw.err(format!("materialet {} har redan *ACOUSTIC MEDIUM, {this}", m.name));
    }
    m.behaviors.push(kw.clone());
    Ok(())
}

fn impedance(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.step.is_some() {
        ctx.step_mut(kw)?.extra.push(kw.clone());
        Ok(())
    } else {
        store_model(ctx, kw)
    }
}
