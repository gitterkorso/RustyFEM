//! Nyckelord för skal, membran och Timoshenkobalkar: *MEMBRANE SECTION,
//! *SHELL GENERAL SECTION och *TRANSVERSE SHEAR STIFFNESS.
//!
//! *TRANSVERSE SHEAR STIFFNESS gäller närmast föregående tvärsnitt och läggs
//! som en intern parameter på dess `raw` (läses av balk- och skalelementen).

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
        return kw.err("ORIENTATION= stöds inte ännu");
    }
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind, raw: kw.clone() });
    Ok(())
}

/// *MEMBRANE SECTION: tjocklek på första dataraden (standard 1).
fn membrane(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("MATERIAL").is_none() {
        return kw.err("MATERIAL= saknas");
    }
    push(ctx, kw, SectionKind::Other)
}

/// *SHELL GENERAL SECTION: homogen (MATERIAL=, tjocklek på första dataraden)
/// eller given styvhetsmatris (datarader med den undre triangeln av 6×6-matrisen
/// [N11, N22, N12, M11, M22, M12]). Tolkas av skalelementen.
fn shell_general(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("COMPOSITE") || kw.has("USER") {
        return kw.err("COMPOSITE och USER stöds inte ännu");
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
        return kw.err("skjuvstyvheterna saknas på dataraden");
    }
    let text = v.iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(",");
    let Some(sec) = ctx.mesh().sections.last_mut() else {
        return kw.err("*TRANSVERSE SHEAR STIFFNESS måste följa ett *BEAM SECTION eller *SHELL SECTION");
    };
    sec.raw.params.push((TSS_PARAM.to_string(), text));
    Ok(())
}
