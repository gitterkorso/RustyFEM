//! *MATERIAL, *ELASTIC and *DENSITY.
//!
//! New material behaviors are registered in their own files with the handler `behavior`
//! and `Kind::Behavior`, e.g.
//! `Keyword { name: "PLASTIC", kind: Kind::Behavior, handler: material::behavior }`.

use crate::fem::model::{MaterialDef, RawKeyword};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "DENSITY", kind: Kind::Behavior, handler: density },
    Keyword { name: "ELASTIC", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "MATERIAL", kind: Kind::Model, handler: material },
];

fn material(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "NAME")?.to_uppercase();
    if ctx.model.material_index(&name).is_some() {
        return kw.err(format!("material {name} is defined twice"));
    }
    ctx.model.materials.push(MaterialDef { name, behaviors: Vec::new(), density: 0.0 });
    ctx.material = Some(ctx.model.materials.len() - 1);
    Ok(())
}

/// Stores a material behavior in the current material (parsed by the material registry).
pub fn behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    // Area materials4: *PLASTIC, RATE= is repeated for each strain rate
    let rate_curve = kw.name == "PLASTIC" && kw.has("RATE");
    if m.behavior(&kw.name).is_some() && !rate_curve {
        return kw.err(format!("material {} already has *{}", m.name, kw.name));
    }
    m.behaviors.push(kw.clone());
    Ok(())
}

fn density(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperature-dependent density is not supported yet");
    }
    let v = kw.first_floats()?;
    let rho = v.first().copied().unwrap_or(0.0);
    let m = ctx.material_mut(kw)?;
    m.density = rho;
    m.behaviors.push(kw.clone());
    Ok(())
}
