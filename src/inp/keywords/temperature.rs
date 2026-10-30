//! Temperature fields in stress analysis: *TEMPERATURE (in steps) and
//! *INITIAL CONDITIONS, TYPE=TEMPERATURE (model data).
//!
//! The node references (label, "INSTANCE.label" or node set) are resolved when the file
//! is read; the keyword is stored with data lines [node index, temperature] and
//! the parameter `materials::thermal::RESOLVED`. The temperature history is built by
//! `materials::thermal::TemperatureField`.
//!
//! *TEMPERATURE, FILE=job: the heat transfer analysis `job.inp` (in the same directory) is run
//! with RustyFEM and its NT11 is used (`steps/heat_stress.rs`); the parameters
//! BSTEP, ESTEP and EINC select the step and increment.
//!
//! TYPE=VELOCITY is forwarded to `dynamics::initial_conditions`. Other
//! types of *INITIAL CONDITIONS (STRESS, ...) give an error until they are implemented
//! – add them to `initial_conditions` below.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::materials::thermal::RESOLVED;
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "INITIAL CONDITIONS", kind: Kind::Model, handler: initial_conditions },
    Keyword { name: "TEMPERATURE", kind: Kind::Step, handler: temperature },
];

/// Data lines "node or node set, value" with resolved node indices.
fn resolve(ctx: &InpCtx, kw: &RawKeyword) -> Result<RawKeyword> {
    let mut out = kw.clone();
    out.params.push((RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let rest: Vec<f64> = d.fields[1..]
            .iter()
            .map(|f| crate::fem::model::parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} line {}: '{f}' is not a number", kw.file, d.line))))
            .collect::<Result<_>>()?;
        if rest.len() > 2 && rest[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "a temperature gradient in a second direction (beams) is not supported yet");
        }
        let value = rest.first().copied().unwrap_or(0.0);
        // Area struct_nonlin: the gradient through the thickness (shells) as the third field
        let grad = rest.get(1).copied();
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            let mut fields = vec![n.to_string(), format!("{value:e}")];
            if let Some(g) = grad {
                fields.push(format!("{g:e}"));
            }
            out.data.push(DataLine { fields, line: d.line, continued: false });
        }
    }
    Ok(out)
}

fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    if ty == "VELOCITY" {
        return super::dynamics::initial_conditions(ctx, kw);
    }
    // STRESS and RATIO are read both by the pore pressure area (the elements in
    // elements/soils.rs) and the geomaterials (initial_state.rs); each one
    // uses only its own elements
    if ty == "STRESS" && kw.has("USER") {
        return super::user2::initial_stress_user(ctx, kw);
    }
    if matches!(ty.as_str(), "STRESS" | "RATIO") {
        super::soils::initial_conditions(ctx, kw)?;
        return super::geomaterials::initial_conditions(ctx, kw);
    }
    if matches!(ty.as_str(), "PORE PRESSURE" | "SATURATION") {
        return super::soils::initial_conditions(ctx, kw);
    }
    if ty == "ENRICHMENT" {
        return super::xfem::initial_conditions(ctx, kw);
    }
    if ty == "FIELD" {
        return super::user2::initial_field(ctx, kw);
    }
    if ty == "CONCENTRATION" {
        return super::multiphysics::initial_concentration(ctx, kw);
    }
    if ty == "SPECIFIC ENERGY" {
        return super::eos::initial_specific_energy(ctx, kw);
    }
    // Area materials5: porous metals (*POROUS METAL PLASTICITY)
    if ty == "RELATIVE DENSITY" {
        return super::materials5::initial_relative_density(ctx, kw);
    }
    // Area explicit_adv: volume fractions of the Eulerian elements
    if ty == "VOLUME FRACTION" {
        return super::explicit_adv::initial_volume_fraction(ctx, kw);
    }
    // Area acoustics2
    if ty == "ACOUSTIC STATIC PRESSURE" {
        return super::acoustics2::initial_static_pressure(ctx, kw);
    }
    // Area cavity
    if ty == "FLUID PRESSURE" {
        return super::cavity::initial_fluid_pressure(ctx, kw);
    }
    if ty != "TEMPERATURE" {
        return kw.err(format!("TYPE={ty} is not supported yet (only TYPE=TEMPERATURE and VELOCITY)"));
    }
    for p in ["FILE", "USER", "INPUT"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    let r = resolve(ctx, kw)?;
    ctx.model.extra.push(r);
    Ok(())
}

fn temperature(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["USER", "INPUT", "AMPLITUDE", "INTERPOLATE", "MIDSIDE", "DRIVING ELSETS"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    let r = match kw.param("FILE") {
        // FILE=job: the heat transfer analysis job.inp is run with RustyFEM (`steps/heat_stress.rs`)
        Some(file) => {
            let values = crate::steps::heat_stress::temperatures_from_job(ctx, kw, file.trim())?;
            let mut out = kw.clone();
            out.params.retain(|(k, _)| !k.eq_ignore_ascii_case("FILE"));
            out.params.push((RESOLVED.to_string(), String::new()));
            out.data = values.into_iter().map(|(n, t)| DataLine { fields: vec![n.to_string(), format!("{t:e}")], line: kw.line, continued: false }).collect();
            out
        }
        None => resolve(ctx, kw)?,
    };
    ctx.step_mut(kw)?.extra.push(r);
    Ok(())
}
