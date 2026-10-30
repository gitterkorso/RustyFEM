//! *SOLID SECTION, *BEAM SECTION, *BEAM GENERAL SECTION and *SHELL SECTION.
//!
//! The sections are assigned to elements via ELSET= once the whole file has been read
//! (`FeModel::finalize`). *SHELL SECTION is stored right away, even if shell elements
//! are missing.

use crate::fem::model::{BeamSectionDef, RawKeyword, SectionDef, SectionKind};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;
use crate::section::BeamSection;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "BEAM GENERAL SECTION", kind: Kind::Model, handler: beam_general },
    Keyword { name: "BEAM SECTION", kind: Kind::Model, handler: beam },
    Keyword { name: "SHELL SECTION", kind: Kind::Model, handler: shell },
    Keyword { name: "SOLID SECTION", kind: Kind::Model, handler: solid },
];

fn push(ctx: &mut InpCtx, kw: &RawKeyword, kind: SectionKind) -> Result<()> {
    let elset = required(kw, "ELSET")?.to_uppercase();
    let material = kw.param("MATERIAL").map(|m| m.trim().to_uppercase()).unwrap_or_default();
    // ORIENTATION= on *SOLID SECTION is handled by materials/anisotropic.rs,
    // on *SHELL SECTION by elements/shell_section.rs and elements/shell2_*.rs
    if kw.has("ORIENTATION") && kw.name != "SOLID SECTION" && kw.name != "SHELL SECTION" && !shell_orientation_ok(ctx, kw, &elset)? {
        return kw.err("ORIENTATION= is not supported yet");
    }
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind, raw: kw.clone() });
    Ok(())
}

/// Area shells2: *SHELL SECTION, ORIENTATION= for the elements that read
/// the orientation (`elements::shell2::ORIENTATION_TYPES`). The elements in ELSET
/// must already be defined and all be of those types.
fn shell_orientation_ok(ctx: &mut InpCtx, kw: &RawKeyword, elset: &str) -> Result<bool> {
    if kw.name != "SHELL SECTION" {
        return Ok(false);
    }
    let types = crate::elements::shell2::ORIENTATION_TYPES;
    let mesh = ctx.mesh();
    let Some(set) = mesh.elsets.get(elset) else {
        return kw.err(format!("ORIENTATION=: element set '{elset}' must be defined before *SHELL SECTION"));
    };
    for &e in set {
        let name = mesh.elements[e].etype.name();
        if !types.contains(&name) {
            return kw.err(format!("ORIENTATION= on *SHELL SECTION is only supported for {} (not {name})", types.join(", ")));
        }
    }
    Ok(true)
}

fn solid(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("MATERIAL").is_none() {
        return kw.err("MATERIAL= is missing");
    }
    let v = kw.first_floats()?;
    let thickness = v.first().copied().filter(|&t| t != 0.0);
    push(ctx, kw, SectionKind::Solid { thickness })
}

/// Section properties for the standard shapes.
fn profile(kw: &RawKeyword, shape: &str, v: &[f64]) -> Result<BeamSection> {
    let need = |n: usize| if v.len() >= n { Ok(()) } else { kw.err(format!("SECTION={shape} requires {n} dimensions on the first data line")) };
    match shape {
        // a = dimension along the 1-axis, b = along the 2-axis
        // Area beams2: J from an internal element solution (not Roark's formula)
        "RECT" => need(2).map(|_| BeamSection { j: crate::elements::beams2::sections::rect_torsion(v[0], v[1]), ..BeamSection::rect(v[0], v[1]) }),
        "CIRC" => need(1).map(|_| BeamSection::circle(v[0])),
        // thin-walled formulas for PIPE
        "PIPE" => need(2).map(|_| BeamSection::pipe_thin(v[0], v[1])),
        // BOX, I, L etc.: thin-walled shapes are computed in Area shells_beams
        _ => match crate::elements::beam_timoshenko::section::shape_profile(shape, v) {
            Some(p) => Ok(p),
            None => kw.err(format!("SECTION={shape} is not supported yet (RECT, CIRC, PIPE, BOX, I, L, HEX, TRAPEZOID, THICK PIPE, ARBITRARY, GENERAL)")),
        },
    }
}

fn n1_line(kw: &RawKeyword, row: usize) -> Result<Option<[f64; 3]>> {
    match kw.data.get(row) {
        Some(d) => {
            let v = kw.floats(d)?;
            if v.iter().all(|&x| x == 0.0) {
                return Ok(None);
            }
            Ok(Some([v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)]))
        }
        None => Ok(None),
    }
}

fn beam(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let shape = required(kw, "SECTION")?.to_uppercase();
    if kw.param("MATERIAL").is_none() {
        return kw.err("MATERIAL= is missing");
    }
    if kw.param_upper("POISSON").is_some() {
        return kw.err("POISSON= is not supported yet");
    }
    // Area beams3: the elbows (ELBOW) read their data from the keyword (elements/elbow.rs)
    if shape == "ELBOW" {
        return push(ctx, kw, SectionKind::Other);
    }
    // Area beams2: ARBITRARY has one data line per segment before the n1 line
    let v = crate::elements::beams2::sections::data_values(kw, &shape)?;
    let profile = profile(kw, &shape, &v)?;
    let n1 = n1_line(kw, crate::elements::beams2::sections::n1_row(&shape, &v))?;
    push(ctx, kw, SectionKind::Beam(BeamSectionDef { shape, profile, i12: 0.0, n1, moduli: None, density: 0.0 }))
}

fn beam_general(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let shape = kw.param_upper("SECTION").unwrap_or_else(|| "GENERAL".into());
    let v = crate::elements::beams2::sections::data_values(kw, &shape)?;
    let row = crate::elements::beams2::sections::n1_row(&shape, &v);
    let (mut profile, i12) = if shape == "GENERAL" || shape == "NONLINEAR GENERAL" {
        if v.len() < 5 {
            return kw.err("SECTION=GENERAL requires A, I11, I12, I22, J on the first data line");
        }
        (BeamSection::general(v[0], v[1], v[3], v[4]), v[2])
    } else {
        (profile(kw, &shape, &v)?, 0.0)
    };
    if shape == "RECT" {
        profile.j = crate::elements::beams2::sections::rect_torsion(v[0], v[1]);
    }
    let n1 = n1_line(kw, row)?;
    // Area gaps1: NONLINEAR GENERAL has no line with E and G (the relations
    // *AXIAL, *M1, *M2, *TORQUE, `elements/beam_nonlinear_general.rs`)
    if shape == "NONLINEAR GENERAL" {
        let density = kw.param("DENSITY").and_then(crate::fem::model::parse_f64).unwrap_or(0.0);
        return push(ctx, kw, SectionKind::Beam(BeamSectionDef { shape, profile, i12, n1, moduli: Some((0.0, 0.0)), density }));
    }
    let em = match kw.data.get(row + 1) {
        Some(d) => kw.floats(d)?,
        None => return kw.err("the third data line (E, G) is missing"),
    };
    if em.len() < 2 || !(em[0] > 0.0) {
        return kw.err("the third data line must contain E and G");
    }
    let density = match kw.param("DENSITY") {
        Some(d) => crate::fem::model::parse_f64(d).unwrap_or(0.0),
        None => 0.0,
    };
    push(ctx, kw, SectionKind::Beam(BeamSectionDef { shape, profile, i12, n1, moduli: Some((em[0], em[1])), density }))
}

fn shell(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("COMPOSITE") {
        // Area struct_nonlin: the layers (thickness, points, material, angle)
        // are read by elements/shell_section.rs; all layers must have the same material
        let mut mats: Vec<String> = Vec::new();
        for d in kw.records() {
            if let Some(m) = d.fields.get(2).map(|f| f.trim().to_uppercase()).filter(|m| !m.is_empty()) {
                if !mats.contains(&m) {
                    mats.push(m);
                }
            }
        }
        if mats.len() != 1 {
            return kw.err("*SHELL SECTION, COMPOSITE is only supported with the same material in all layers (different angles and thicknesses are fine)");
        }
        let mut k = kw.clone();
        k.params.retain(|(p, _)| !p.eq_ignore_ascii_case("MATERIAL"));
        k.params.push(("MATERIAL".into(), mats[0].clone()));
        return push(ctx, &k, SectionKind::Other);
    }
    let v = kw.first_floats()?;
    let thickness = v.first().copied().unwrap_or(0.0);
    let n_int = v.get(1).copied().unwrap_or(5.0).max(1.0) as usize;
    push(ctx, kw, SectionKind::Shell { thickness, n_int })
}
