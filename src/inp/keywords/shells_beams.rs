//! Keywords for shells, membranes and Timoshenko beams: *MEMBRANE SECTION,
//! *SHELL GENERAL SECTION and *TRANSVERSE SHEAR STIFFNESS.
//!
//! *TRANSVERSE SHEAR STIFFNESS applies to the nearest preceding section and is added
//! as an internal parameter on its `raw` (read by the beam and shell elements).

use crate::elements::beam_timoshenko::section::TSS_PARAM;
use crate::fem::model::{RawKeyword, SectionDef, SectionKind};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "MEMBRANE SECTION", kind: Kind::Model, handler: membrane },
    Keyword { name: "SHELL GENERAL SECTION", kind: Kind::Model, handler: shell_general },
    Keyword { name: "TRANSVERSE SHEAR STIFFNESS", kind: Kind::Model, handler: transverse_shear },
];

fn push(ctx: &mut InpCtx, kw: &RawKeyword, kind: SectionKind) -> Result<()> {
    let elset = required(kw, "ELSET")?.to_uppercase();
    let material = kw.param("MATERIAL").map(|m| m.trim().to_uppercase()).unwrap_or_default();
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= is not supported yet");
    }
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind, raw: kw.clone() });
    Ok(())
}

/// *MEMBRANE SECTION: thickness on the first data line (default 1).
fn membrane(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("MATERIAL").is_none() {
        return kw.err("MATERIAL= is missing");
    }
    push(ctx, kw, SectionKind::Other)
}

/// *SHELL GENERAL SECTION: homogeneous (MATERIAL=, thickness on the first data line)
/// or a given stiffness matrix (data lines with the lower triangle of the 6×6 matrix
/// [N11, N22, N12, M11, M22, M12]). Parsed by the shell elements.
fn shell_general(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("COMPOSITE") || kw.has("USER") {
        return kw.err("COMPOSITE and USER are not supported yet");
    }
    if kw.param("MATERIAL").is_some() {
        let v = kw.first_floats()?;
        let thickness = v.first().copied().unwrap_or(0.0);
        return push(ctx, kw, SectionKind::Shell { thickness, n_int: 0 });
    }
    push(ctx, kw, SectionKind::Other)
}

fn transverse_shear(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let v = kw.first_floats()?;
    if v.is_empty() {
        return kw.err("the shear stiffnesses are missing on the data line");
    }
    let text = v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",");
    let Some(sec) = ctx.mesh().sections.last_mut() else {
        return kw.err("*TRANSVERSE SHEAR STIFFNESS must follow a *BEAM SECTION or *SHELL SECTION");
    };
    sec.raw.params.push((TSS_PARAM.to_string(), text));
    Ok(())
}
