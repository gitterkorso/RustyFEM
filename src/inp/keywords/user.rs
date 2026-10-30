//! User subroutines: *USER MATERIAL and *DEPVAR (material behaviors, parsed by
//! `materials/user_material.rs`), *USER ELEMENT (stored in the model, parsed by
//! `elements/user_element.rs` when *ELEMENT, TYPE=Un is read) and *UEL PROPERTY
//! (stored as a section for the elements in ELSET=).

use crate::fem::model::{RawKeyword, SectionDef, SectionKind};
use crate::inp::keywords::material::behavior;
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "DEPVAR", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "UEL PROPERTY", kind: Kind::Model, handler: uel_property },
    Keyword { name: "USER ELEMENT", kind: Kind::Model, handler: user_element },
    Keyword { name: "USER MATERIAL", kind: Kind::Behavior, handler: behavior },
];

fn user_element(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = required(kw, "TYPE")?.to_uppercase();
    let Some(n) = ty.strip_prefix('U').and_then(|s| s.parse::<u32>().ok()) else {
        return kw.err(format!("TYPE={ty} must be Un (n = integer)"));
    };
    if crate::elements::lookup(&ty).is_some() {
        return kw.err(format!("TYPE={ty} is already a built-in element type"));
    }
    // check the definition right here (errors with the correct line number)
    crate::elements::user_element::define(kw, &ty, n)?;
    ctx.mesh().extra.push(kw.clone());
    Ok(())
}

fn uel_property(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let elset = required(kw, "ELSET")?.to_uppercase();
    if kw.has("MATERIAL") {
        return kw.err("MATERIAL= on *UEL PROPERTY is not supported yet");
    }
    ctx.mesh().sections.push(SectionDef { elset, material: String::new(), material_index: None, kind: SectionKind::Other, raw: kw.clone() });
    Ok(())
}
