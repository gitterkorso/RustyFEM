//! Area gaps1: keywords for equations of state and beam sections.
//!
//! - *DETONATION POINT: material behavior after *EOS, TYPE=JWL (`materials/eos_forms.rs`).
//! - *EOS COMPACTION: material behavior (P-α); gives an error until the model exists.
//! - *AXIAL, *M1, *M2, *TORQUE: nonlinear section relations for
//!   *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL (`elements/beam_nonlinear_general.rs`).
//!   They are attached to the latest section (stored as parameters on its `raw`).

use crate::fem::model::{RawKeyword, SectionKind};
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "AXIAL", kind: Kind::Model, handler: section_behavior },
    Keyword { name: "DETONATION POINT", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "EOS COMPACTION", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "M1", kind: Kind::Model, handler: section_behavior },
    Keyword { name: "M2", kind: Kind::Model, handler: section_behavior },
    Keyword { name: "TORQUE", kind: Kind::Model, handler: section_behavior },
];

/// Parameter prefix for the nonlinear relations on the section's `raw`.
pub const NLG_PARAM: &str = "_NLG_";

/// *AXIAL/*M1/*M2/*TORQUE: attached to the nearest preceding
/// *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL. The relation is stored as a
/// parameter `_NLG_<NAME>` = "kind;x1,y1;x2,y2;..." (kind LINEAR, ELASTIC or
/// PLASTIC).
fn section_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("DEPENDENCIES") {
        return kw.err("DEPENDENCIES is not supported for nonlinear section relations");
    }
    let kind = if kw.has("LINEAR") {
        "LINEAR"
    } else if kw.has("ELASTIC") {
        "ELASTIC"
    } else {
        "PLASTIC"
    };
    let mut text = kind.to_string();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.is_empty() {
            continue;
        }
        if v.len() > 2 && v[2] != 0.0 {
            return kw.err_at(d.line, "temperature-dependent section relations are not supported");
        }
        text.push_str(&format!(";{:e},{:e}", v[0], v.get(1).copied().unwrap_or(0.0)));
    }
    let mesh = ctx.mesh();
    let Some(sec) = mesh.sections.iter_mut().rev().find(|s| s.raw.name == "BEAM GENERAL SECTION") else {
        return kw.err(format!("*{} must follow *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL", kw.name));
    };
    if sec.raw.param_upper("SECTION").as_deref() != Some("NONLINEAR GENERAL") || !matches!(sec.kind, SectionKind::Beam(_)) {
        return kw.err(format!("*{} requires *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL", kw.name));
    }
    let key = format!("{NLG_PARAM}{}", kw.name);
    sec.raw.params.retain(|(k, _)| k != &key);
    sec.raw.params.push((key, text));
    // nonlinear relations: the model is solved with Newton (`interactions/gaps1.rs`)
    ctx.model.extra.push(kw.clone());
    Ok(())
}
