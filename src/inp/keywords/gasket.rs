//! Packningar: *GASKET SECTION, *GASKET BEHAVIOR, *GASKET THICKNESS BEHAVIOR
//! och *GASKET ELASTICITY (`elements/gasket.rs`, `materials/gasket.rs`).
//!
//! *GASKET BEHAVIOR, NAME= skapar ett material (som *MATERIAL) som
//! beteendena läggs i; *GASKET SECTION, BEHAVIOR= kopplar det till elementen.

use crate::fem::model::{MaterialDef, RawKeyword, SectionDef, SectionKind};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::materials::gasket::{MARKER, MEMBRANE, SHEAR, THICKNESS, THICKNESS_UNLOADING};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "GASKET BEHAVIOR", kind: Kind::Model, handler: gasket_behavior },
    Keyword { name: "GASKET ELASTICITY", kind: Kind::Behavior, handler: elasticity },
    Keyword { name: "GASKET SECTION", kind: Kind::Model, handler: section },
    Keyword { name: "GASKET THICKNESS BEHAVIOR", kind: Kind::Behavior, handler: thickness },
];

fn gasket_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = required(kw, "NAME")?.to_uppercase();
    if ctx.model.material_index(&name).is_some() {
        return kw.err(format!("materialet eller packningsbeteendet {name} är definierat två gånger"));
    }
    let marker = RawKeyword { name: MARKER.into(), ..kw.clone() };
    ctx.model.materials.push(MaterialDef { name, behaviors: vec![marker], density: 0.0 });
    ctx.material = Some(ctx.model.materials.len() - 1);
    Ok(())
}

fn push(ctx: &mut InpCtx, kw: &RawKeyword, name: &str) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    if m.behavior(MARKER).is_none() {
        return kw.err("ska stå efter *GASKET BEHAVIOR (packningsbeteende i *MATERIAL stöds inte ännu)");
    }
    if m.behavior(name).is_some() {
        return kw.err(format!("packningsbeteendet {} har redan *{}", m.name, kw.name));
    }
    m.behaviors.push(RawKeyword { name: name.into(), ..kw.clone() });
    Ok(())
}

fn thickness(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = match kw.param_upper("DIRECTION").as_deref() {
        None | Some("LOADING") => THICKNESS,
        Some("UNLOADING") => THICKNESS_UNLOADING,
        Some(d) => return kw.err(format!("DIRECTION={d} är okänt (LOADING, UNLOADING)")),
    };
    if kw.param_upper("VARIABLE").is_some_and(|v| v != "STRESS" && v != "FORCE") {
        return kw.err("VARIABLE= ska vara STRESS eller FORCE");
    }
    push(ctx, kw, name)
}

fn elasticity(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let name = match kw.param_upper("COMPONENT").as_deref() {
        Some("MEMBRANE") => MEMBRANE,
        Some("TRANSVERSE SHEAR") => SHEAR,
        Some(c) => return kw.err(format!("COMPONENT={c} stöds inte (MEMBRANE, TRANSVERSE SHEAR)")),
        None => return kw.err("COMPONENT= saknas (MEMBRANE eller TRANSVERSE SHEAR)"),
    };
    push(ctx, kw, name)
}

fn section(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let elset = required(kw, "ELSET")?.to_uppercase();
    let Some(beh) = kw.param("BEHAVIOR") else {
        return kw.err("BEHAVIOR= saknas (packningar med MATERIAL= stöds inte ännu)");
    };
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= stöds inte ännu för packningar");
    }
    let v = kw.first_floats()?;
    for (i, x) in v.iter().enumerate().skip(1) {
        if *x != 0.0 {
            return kw.err(format!("fält {} på dataraden stöds inte ännu (bara initial tjocklek)", i + 1));
        }
    }
    let material = beh.trim().to_uppercase();
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind: SectionKind::Other, raw: kw.clone() });
    Ok(())
}
