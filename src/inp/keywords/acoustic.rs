//! Keywords for acoustics.
//!
//! - *ACOUSTIC MEDIUM: material behavior (BULK MODULUS, VOLUMETRIC DRAG), may
//!   appear several times in a material (`materials/acoustic.rs`).
//! - *IMPEDANCE PROPERTY: model data, stored in `FeModel::extra`.
//! - *IMPEDANCE (element faces In) and *SIMPEDANCE (surfaces): impedance or
//!   non-reflecting boundary (NONREFLECTING=PLANAR/CIRCULAR/SPHERICAL). They are
//!   step data; outside a step they are stored in the model (apply to all
//!   steps). Read by `steps/acoustic.rs`.
//!
//! Acoustic–structural coupling is done with *TIE between an acoustic and a
//! structural surface (`interactions/acoustic.rs`).

use crate::fem::model::RawKeyword;
use crate::inp::{store_model, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ACOUSTIC MEDIUM", kind: Kind::Behavior, handler: acoustic_medium },
    Keyword { name: "IMPEDANCE", kind: Kind::Both, handler: impedance },
    Keyword { name: "IMPEDANCE PROPERTY", kind: Kind::Model, handler: store_model },
    Keyword { name: "SIMPEDANCE", kind: Kind::Both, handler: impedance },
];

/// Material behavior that may be repeated (one entry per parameter).
fn acoustic_medium(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    // Area gaps1: one entry per kind (BULK MODULUS, VOLUMETRIC DRAG, COMPLEX ..., POROUS MODEL, CAVITATION LIMIT)
    let kind = |b: &RawKeyword| -> &'static str {
        ["VOLUMETRIC DRAG", "COMPLEX BULK MODULUS", "COMPLEX DENSITY", "POROUS MODEL", "CAVITATION LIMIT"].into_iter().find(|p| b.has(p)).unwrap_or("BULK MODULUS")
    };
    let this = kind(kw);
    let m = ctx.material_mut(kw)?;
    if m.behaviors.iter().any(|b| b.name == kw.name && kind(b) == this) {
        return kw.err(format!("material {} already has *ACOUSTIC MEDIUM, {this}", m.name));
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
