//! *COHESIVE SECTION for cohesive elements (`elements/cohesive.rs`).
//!
//! Parameters: ELSET=, MATERIAL=, RESPONSE=TRACTION SEPARATION (default),
//! THICKNESS=SPECIFIED (default) or GEOMETRY, CONTROLS= (*SECTION
//! CONTROLS with VISCOSITY= for viscous regularization of the damage).
//! Data line: constitutive thickness T0 (empty or 0 = 1) and the out-of-plane
//! thickness for COH2D4 (empty or 0 = 1).
//!
//! The section is stored as `SectionKind::Other` with the original keyword;
//! *SECTION CONTROLS (stored by `keywords/explicit.rs`) is looked up here and
//! the viscosity is added as the internal parameter `_VISCOSITY`. Therefore
//! *SECTION CONTROLS must appear before *COHESIVE SECTION in RustyFEM.

use crate::fem::model::{parse_f64, RawKeyword, SectionDef, SectionKind};
use crate::inp::{required, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "COHESIVE SECTION", kind: Kind::Model, handler: cohesive_section }];

/// Internal parameter holding the viscosity from *SECTION CONTROLS.
pub const VISCOSITY: &str = "_VISCOSITY";

fn cohesive_section(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let elset = required(kw, "ELSET")?.trim().to_uppercase();
    let material = required(kw, "MATERIAL")?.trim().to_uppercase();
    let resp = kw.param_upper("RESPONSE").unwrap_or_else(|| "TRACTION SEPARATION".into());
    let resp = resp.split_whitespace().collect::<Vec<_>>().join(" ");
    if resp != "TRACTION SEPARATION" {
        return kw.err(format!("RESPONSE={resp} is not supported yet (only TRACTION SEPARATION)"));
    }
    match kw.param_upper("THICKNESS").as_deref() {
        None | Some("SPECIFIED") | Some("GEOMETRY") => {}
        Some(t) => return kw.err(format!("THICKNESS={t} is not supported (SPECIFIED, GEOMETRY)")),
    }
    for p in ["ORIENTATION", "STACK DIRECTION"] {
        if kw.has(p) {
            return kw.err(format!("{p}= is not supported yet for cohesive elements"));
        }
    }
    let v = kw.first_floats()?;
    if v.iter().any(|x| *x < 0.0) {
        return kw.err("the thicknesses may not be negative");
    }
    let mut raw = kw.clone();
    if let Some(name) = kw.param_upper("CONTROLS") {
        let find = |extra: &[RawKeyword]| -> Option<RawKeyword> {
            extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())).cloned()
        };
        let sc = find(&ctx.mesh_ref().extra).or_else(|| find(&ctx.model.extra));
        let Some(sc) = sc else {
            return kw.err(format!("*SECTION CONTROLS, NAME={name} is missing (it must appear before *COHESIVE SECTION in RustyFEM)"));
        };
        let mu = match sc.param("VISCOSITY") {
            Some(s) => parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: invalid VISCOSITY=", sc.location())))?,
            None => 0.0,
        };
        if mu < 0.0 {
            return sc.err("VISCOSITY= may not be negative");
        }
        raw.params.push((VISCOSITY.into(), format!("{mu:e}")));
    }
    ctx.mesh().sections.push(SectionDef { elset, material, material_index: None, kind: SectionKind::Other, raw });
    Ok(())
}

/// Constitutive thickness (None = from the geometry), out-of-plane thickness and viscosity.
pub fn section_values(sec: &SectionDef) -> (Option<f64>, f64, f64) {
    let v = sec.raw.first_floats().unwrap_or_default();
    let geometry = sec.raw.param_upper("THICKNESS").as_deref() == Some("GEOMETRY");
    let t0 = v.first().copied().filter(|&t| t > 0.0).unwrap_or(1.0);
    let w = v.get(1).copied().filter(|&t| t > 0.0).unwrap_or(1.0);
    let mu = sec.raw.param(VISCOSITY).and_then(parse_f64).unwrap_or(0.0);
    ((!geometry).then_some(t0), w, mu)
}
