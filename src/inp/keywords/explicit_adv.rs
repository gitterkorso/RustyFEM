//! Område explicit_adv: nyckelord för avancerade metoder i explicit analys
//! (SPH, DEM, kopplad Euler–Lagrange och ALE), tolkade av
//! `steps/explicit_adv.rs`.
//!
//! - `*DISCRETE SECTION` och `*EULERIAN SECTION` blir tvärsnitt utan material
//!   (`SectionKind::Other`, data i `raw`) så att elementen PD3D och EC3D8R
//!   får sina tvärsnitt av `FeModel::finalize`.
//! - `*ADAPTIVE MESH`, `*ADAPTIVE MESH CONTROLS` sparas i stegets `extra`.
//! - `*INITIAL CONDITIONS, TYPE=VOLUME FRACTION` (anropas från
//!   `temperature.rs`) sparas upplöst (elementindex, volymandel, materialinstans).

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

/// Parameter som markerar ett upplöst *INITIAL CONDITIONS, TYPE=VOLUME
/// FRACTION (rader: elementindex, volymandel, materialinstans).
pub const VF_RESOLVED: &str = "RUSTYFEM_VF";

fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} stöds inte ännu i RustyFEM", kw.name))
}

/// *ADAPTIVE MESH CONTROLS: i modelldata eller steget (formellt historiedata);
/// sparas i modellens `extra` så att alla steg hittar namnet.
fn adaptive_controls(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.param("NAME").is_none() {
        return kw.err("NAME= saknas");
    }
    ctx.model.extra.push(kw.clone());
    Ok(())
}

/// *DISCRETE SECTION och *EULERIAN SECTION: tvärsnitt utan material.
fn section(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let Some(elset) = kw.param("ELSET") else { return kw.err("ELSET= saknas") };
    let elset = elset.trim().to_uppercase();
    if kw.name == "DISCRETE SECTION" {
        if kw.param_upper("SHAPE").is_some_and(|s| s != "SPHERE") {
            return kw.err("bara SHAPE=SPHERE stöds");
        }
        for p in ["FLUID BEHAVIOR", "PARTICLE BEHAVIOR"] {
            if kw.has(p) {
                return kw.err(format!("parametern {p} stöds inte ännu"));
            }
        }
        if kw.data.len() > 1 {
            return kw.err("partikelkluster (fler datarader) stöds inte ännu");
        }
    } else {
        for p in ["ORIENTATION"] {
            if kw.has(p) {
                return kw.err(format!("parametern {p} stöds inte ännu"));
            }
        }
        if kw.data.is_empty() {
            return kw.err("ange minst ett material på dataraderna");
        }
    }
    ctx.mesh().sections.push(SectionDef { elset, material: String::new(), material_index: None, kind: SectionKind::Other, raw: kw.clone() });
    Ok(())
}

/// *DISCRETE ELASTICITY (E, ν för Hertz-kontakt) efter *DISCRETE SECTION:
/// sparas som parametrar på tvärsnittets nyckelord.
fn discrete_elasticity(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let v = kw.first_floats()?;
    let Some(&e) = v.first() else { return kw.err("ange E och ν") };
    let nu = v.get(1).copied().unwrap_or(0.0);
    if !(e > 0.0) {
        return kw.err("E ska vara > 0");
    }
    let Some(sec) = ctx.mesh().sections.last_mut().filter(|s| s.raw.name == "DISCRETE SECTION") else {
        return kw.err("*DISCRETE ELASTICITY ska stå direkt efter *DISCRETE SECTION");
    };
    sec.raw.params.push((crate::steps::explicit::adv::dem::ELASTIC_E.to_string(), format!("{e:e}")));
    sec.raw.params.push((crate::steps::explicit::adv::dem::ELASTIC_NU.to_string(), format!("{nu:e}")));
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=VOLUME FRACTION: rader (element eller
/// elementmängd, volymandel, materialinstans) sparas upplösta i `FeModel::extra`.
pub fn initial_volume_fraction(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "USER", "INPUT", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte för TYPE=VOLUME FRACTION"));
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
        // element, materialinstans, volymandel
        let Some(mat) = d.fields.get(1).map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()) else {
            return kw.err_at(d.line, "ange element (eller elementmängd), materialinstans och volymandel");
        };
        let v = match d.fields.get(2) {
            Some(f) => match crate::fem::model::parse_f64(f) {
                Some(v) => v,
                None => return kw.err_at(d.line, "volymandelen är inte ett tal"),
            },
            None => 1.0,
        };
        if !(0.0..=1.0 + 1e-9).contains(&v) {
            return kw.err_at(d.line, "volymandelen ska ligga mellan 0 och 1");
        }
        for e in ctx.elem_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![e.to_string(), format!("{v:e}"), mat.clone()], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
