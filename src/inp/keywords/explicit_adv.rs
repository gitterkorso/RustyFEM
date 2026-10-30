//! Area explicit_adv: keywords for advanced methods in explicit analysis
//! (SPH, DEM, coupled Euler–Lagrange and ALE), interpreted by
//! `steps/explicit_adv.rs`.
//!
//! - `*DISCRETE SECTION` and `*EULERIAN SECTION` become sections without material
//!   (`SectionKind::Other`, data in `raw`) so that the elements PD3D and EC3D8R
//!   get their sections from `FeModel::finalize`.
//! - `*ADAPTIVE MESH`, `*ADAPTIVE MESH CONTROLS` are stored in the step's `extra`.
//! - `*INITIAL CONDITIONS, TYPE=VOLUME FRACTION` (called from
//!   `temperature.rs`) is stored resolved (element index, volume fraction, material instance).

use crate::fem::model::{DataLine, RawKeyword, SectionDef, SectionKind};
use crate::inp::{store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ADAPTIVE MESH", kind: Kind::Step, handler: store_step },
    Keyword { name: "ADAPTIVE MESH CONSTRAINT", kind: Kind::Step, handler: unsupported },
    Keyword { name: "ADAPTIVE MESH CONTROLS", kind: Kind::Both, handler: adaptive_controls },
    Keyword { name: "DISCRETE ELASTICITY", kind: Kind::Model, handler: discrete_elasticity },
    Keyword { name: "DISCRETE SECTION", kind: Kind::Model, handler: section },
    Keyword { name: "EULERIAN BOUNDARY", kind: Kind::Step, handler: unsupported },
    Keyword { name: "EULERIAN MESH MOTION", kind: Kind::Step, handler: unsupported },
    Keyword { name: "EULERIAN SECTION", kind: Kind::Model, handler: section },
    Keyword { name: "PARTICLE GENERATOR", kind: Kind::Model, handler: unsupported },
    Keyword { name: "PARTICLE GENERATOR FLOW", kind: Kind::Step, handler: unsupported },
    Keyword { name: "PARTICLE GENERATOR INLET", kind: Kind::Model, handler: unsupported },
    Keyword { name: "PARTICLE OUTLET", kind: Kind::Model, handler: unsupported },
    Keyword { name: "SPH SURFACE BEHAVIOR", kind: Kind::Both, handler: unsupported },
];

/// Parameter that marks a resolved *INITIAL CONDITIONS, TYPE=VOLUME
/// FRACTION (lines: element index, volume fraction, material instance).
pub const VF_RESOLVED: &str = "RUSTYFEM_VF";

fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} is not supported yet in RustyFEM", kw.name))
}

/// *ADAPTIVE MESH CONTROLS: in model data or the step (formally history data);
/// stored in the model's `extra` so that all steps find the name.
fn adaptive_controls(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("NAME").is_none() {
        return kw.err("NAME= is missing");
    }
    ctx.model.extra.push(kw.clone());
    Ok(())
}

/// *DISCRETE SECTION and *EULERIAN SECTION: sections without material.
fn section(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let Some(elset) = kw.param("ELSET") else { return kw.err("ELSET= is missing") };
    let elset = elset.trim().to_uppercase();
    if kw.name == "DISCRETE SECTION" {
        if kw.param_upper("SHAPE").is_some_and(|s| s != "SPHERE") {
            return kw.err("only SHAPE=SPHERE is supported");
        }
        for p in ["FLUID BEHAVIOR", "PARTICLE BEHAVIOR"] {
            if kw.has(p) {
                return kw.err(format!("parameter {p} is not supported yet"));
            }
        }
        if kw.data.len() > 1 {
            return kw.err("particle clusters (several data lines) are not supported yet");
        }
    } else {
        for p in ["ORIENTATION"] {
            if kw.has(p) {
                return kw.err(format!("parameter {p} is not supported yet"));
            }
        }
        if kw.data.is_empty() {
            return kw.err("give at least one material on the data lines");
        }
    }
    ctx.mesh().sections.push(SectionDef { elset, material: String::new(), material_index: None, kind: SectionKind::Other, raw: kw.clone() });
    Ok(())
}

/// *DISCRETE ELASTICITY (E, ν for Hertz contact) after *DISCRETE SECTION:
/// stored as parameters on the section's keyword.
fn discrete_elasticity(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let v = kw.first_floats()?;
    let Some(&e) = v.first() else { return kw.err("give E and ν") };
    let nu = v.get(1).copied().unwrap_or(0.0);
    if !(e > 0.0) {
        return kw.err("E must be > 0");
    }
    let Some(sec) = ctx.mesh().sections.last_mut().filter(|s| s.raw.name == "DISCRETE SECTION") else {
        return kw.err("*DISCRETE ELASTICITY must appear directly after *DISCRETE SECTION");
    };
    sec.raw.params.push((crate::steps::explicit::adv::dem::ELASTIC_E.to_string(), format!("{e:e}")));
    sec.raw.params.push((crate::steps::explicit::adv::dem::ELASTIC_NU.to_string(), format!("{nu:e}")));
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=VOLUME FRACTION: lines (element or
/// element set, volume fraction, material instance) are stored resolved in `FeModel::extra`.
pub fn initial_volume_fraction(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "USER", "INPUT", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported for TYPE=VOLUME FRACTION"));
        }
    }
    let mut out = kw.clone();
    out.params.push((VF_RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        // element, material instance, volume fraction
        let Some(mat) = d.fields.get(1).map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()) else {
            return kw.err_at(d.line, "give an element (or element set), material instance and volume fraction");
        };
        let v = match d.fields.get(2) {
            Some(f) => match crate::fem::model::parse_f64(f) {
                Some(v) => v,
                None => return kw.err_at(d.line, "the volume fraction is not a number"),
            },
            None => 1.0,
        };
        if !(0.0..=1.0 + 1e-9).contains(&v) {
            return kw.err_at(d.line, "the volume fraction must lie between 0 and 1");
        }
        for e in ctx.elem_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![e.to_string(), format!("{v:e}"), mat.clone()], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
