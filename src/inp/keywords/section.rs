//! *SOLID SECTION, *BEAM SECTION, *BEAM GENERAL SECTION och *SHELL SECTION.
//!
//! Tvärsnitten kopplas till element via ELSET= när hela filen är läst
//! (`FeModel::finalize`). *SHELL SECTION lagras redan nu, även om skalelement
//! saknas.

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
    // ORIENTATION= på *SOLID SECTION hanteras av materials/anisotropic.rs,
    // på *SHELL SECTION av elements/shell_section.rs och elements/shell2_*.rs
    if kw.has("ORIENTATION") && kw.name != "SOLID SECTION" && kw.name != "SHELL SECTION" && !shell_orientation_ok(ctx, kw, &elset)? {
        return kw.err("ORIENTATION= stöds inte ännu");
    }
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind, raw: kw.clone() });
    Ok(())
}

/// Område shells2: *SHELL SECTION, ORIENTATION= för de element som läser
/// orienteringen (`elements::shell2::ORIENTATION_TYPES`). Elementen i ELSET
/// måste redan vara definierade och alla vara av de typerna.
fn shell_orientation_ok(ctx: &mut InpCtx, kw: &RawKeyword, elset: &str) -> Result<bool> {
    if kw.name != "SHELL SECTION" {
        return Ok(false);
    }
    let types = crate::elements::shell2::ORIENTATION_TYPES;
    let mesh = ctx.mesh();
    let Some(set) = mesh.elsets.get(elset) else {
        return kw.err(format!("ORIENTATION=: elementsetet '{elset}' måste definieras före *SHELL SECTION"));
    };
    for &e in set {
        let name = mesh.elements[e].etype.name();
        if !types.contains(&name) {
            return kw.err(format!("ORIENTATION= på *SHELL SECTION stöds bara för {} (inte {name})", types.join(", ")));
        }
    }
    Ok(true)
}

fn solid(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("MATERIAL").is_none() {
        return kw.err("MATERIAL= saknas");
    }
    let v = kw.first_floats()?;
    let thickness = v.first().copied().filter(|&t| t != 0.0);
    push(ctx, kw, SectionKind::Solid { thickness })
}

/// Tvärsnittsegenskaper för standardformerna.
fn profile(kw: &RawKeyword, shape: &str, v: &[f64]) -> Result<BeamSection> {
    let need = |n: usize| if v.len() >= n { Ok(()) } else { kw.err(format!("SECTION={shape} kräver {n} mått på första dataraden")) };
    match shape {
        // a = mått längs 1-axeln, b = längs 2-axeln
        // område beams2: J ur en intern elementlösning (inte Roarks formel)
        "RECT" => need(2).map(|_| BeamSection { j: crate::elements::beams2::sections::rect_torsion(v[0], v[1]), ..BeamSection::rect(v[0], v[1]) }),
        "CIRC" => need(1).map(|_| BeamSection::circle(v[0])),
        // tunnväggiga formler för PIPE
        "PIPE" => need(2).map(|_| BeamSection::pipe_thin(v[0], v[1])),
        // BOX, I, L m.fl.: tunnväggiga former räknas i området shells_beams
        _ => match crate::elements::beam_timoshenko::section::shape_profile(shape, v) {
            Some(p) => Ok(p),
            None => kw.err(format!("SECTION={shape} stöds inte ännu (RECT, CIRC, PIPE, BOX, I, L, HEX, TRAPEZOID, THICK PIPE, ARBITRARY, GENERAL)")),
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
        return kw.err("MATERIAL= saknas");
    }
    if kw.param_upper("POISSON").is_some() {
        return kw.err("POISSON= stöds inte ännu");
    }
    // område beams3: rörkrökarna (ELBOW) läser sina data ur nyckelordet (elements/elbow.rs)
    if shape == "ELBOW" {
        return push(ctx, kw, SectionKind::Other);
    }
    // område beams2: ARBITRARY har en datarad per segment före n1-raden
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
            return kw.err("SECTION=GENERAL kräver A, I11, I12, I22, J på första dataraden");
        }
        (BeamSection::general(v[0], v[1], v[3], v[4]), v[2])
    } else {
        (profile(kw, &shape, &v)?, 0.0)
    };
    if shape == "RECT" {
        profile.j = crate::elements::beams2::sections::rect_torsion(v[0], v[1]);
    }
    let n1 = n1_line(kw, row)?;
    // område gaps1: NONLINEAR GENERAL har ingen rad med E och G (sambanden
    // *AXIAL, *M1, *M2, *TORQUE, `elements/beam_nonlinear_general.rs`)
    if shape == "NONLINEAR GENERAL" {
        let density = kw.param("DENSITY").and_then(crate::fem::model::parse_f64).unwrap_or(0.0);
        return push(ctx, kw, SectionKind::Beam(BeamSectionDef { shape, profile, i12, n1, moduli: Some((0.0, 0.0)), density }));
    }
    let em = match kw.data.get(row + 1) {
        Some(d) => kw.floats(d)?,
        None => return kw.err("tredje dataraden (E, G) saknas"),
    };
    if em.len() < 2 || !(em[0] > 0.0) {
        return kw.err("tredje dataraden ska innehålla E och G");
    }
    let density = match kw.param("DENSITY") {
        Some(d) => crate::fem::model::parse_f64(d).unwrap_or(0.0),
        None => 0.0,
    };
    push(ctx, kw, SectionKind::Beam(BeamSectionDef { shape, profile, i12, n1, moduli: Some((em[0], em[1])), density }))
}

fn shell(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("COMPOSITE") {
        // område struct_nonlin: skikten (tjocklek, punkter, material, vinkel)
        // läses av elements/shell_section.rs; alla skikt ska ha samma material
        let mut mats: Vec<String> = Vec::new();
        for d in kw.records() {
            if let Some(m) = d.fields.get(2).map(|f| f.trim().to_uppercase()).filter(|m| !m.is_empty()) {
                if !mats.contains(&m) {
                    mats.push(m);
                }
            }
        }
        if mats.len() != 1 {
            return kw.err("*SHELL SECTION, COMPOSITE stöds bara med samma material i alla skikt (olika vinklar och tjocklekar går bra)");
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
