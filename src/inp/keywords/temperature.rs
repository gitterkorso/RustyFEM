//! Temperaturfält i spänningsanalys: *TEMPERATURE (i steg) och
//! *INITIAL CONDITIONS, TYPE=TEMPERATURE (modelldata).
//!
//! Nodreferenserna (nummer, "INSTANS.nummer" eller nodset) löses upp när filen
//! läses; nyckelordet sparas med datarader [nodindex, temperatur] och
//! parametern `materials::thermal::RESOLVED`. Temperaturhistorien byggs av
//! `materials::thermal::TemperatureField`.
//!
//! *TEMPERATURE, FILE=jobb: värmeanalysen `jobb.inp` (i samma katalog) körs
//! med RustyFEM och dess NT11 används (`steps/heat_stress.rs`); parametrarna
//! BSTEP, ESTEP och EINC väljer steg och inkrement.
//!
//! TYPE=VELOCITY skickas vidare till `dynamics::initial_conditions`. Andra
//! typer av *INITIAL CONDITIONS (STRESS, ...) ger fel tills de implementeras
//! – lägg till dem i `initial_conditions` nedan.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::{InpCtx, Keyword, Kind};
use crate::materials::thermal::RESOLVED;
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "INITIAL CONDITIONS", kind: Kind::Model, handler: initial_conditions },
    Keyword { name: "TEMPERATURE", kind: Kind::Step, handler: temperature },
];

/// Datarader "nod eller nodset, värde" med upplösta nodindex.
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
            .map(|f| crate::fem::model::parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{f}' är inget tal", kw.file, d.line))))
            .collect::<Result<_>>()?;
        if rest.len() > 2 && rest[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "temperaturgradient i en andra riktning (balkar) stöds inte ännu");
        }
        let value = rest.first().copied().unwrap_or(0.0);
        // område struct_nonlin: gradienten genom tjockleken (skal) som tredje fält
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
    // STRESS och RATIO läses av både porvattentrycksområdet (elementen i
    // elements/soils.rs) och geomaterialen (initial_state.rs); vart och ett
    // använder bara sina egna element
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
    // område materials5: porösa metaller (*POROUS METAL PLASTICITY)
    if ty == "RELATIVE DENSITY" {
        return super::materials5::initial_relative_density(ctx, kw);
    }
    // område explicit_adv: Euler-elementens volymandelar
    if ty == "VOLUME FRACTION" {
        return super::explicit_adv::initial_volume_fraction(ctx, kw);
    }
    // område acoustics2
    if ty == "ACOUSTIC STATIC PRESSURE" {
        return super::acoustics2::initial_static_pressure(ctx, kw);
    }
    // område cavity
    if ty == "FLUID PRESSURE" {
        return super::cavity::initial_fluid_pressure(ctx, kw);
    }
    if ty != "TEMPERATURE" {
        return kw.err(format!("TYPE={ty} stöds inte ännu (bara TYPE=TEMPERATURE och VELOCITY)"));
    }
    for p in ["FILE", "USER", "INPUT"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let r = resolve(ctx, kw)?;
    ctx.model.extra.push(r);
    Ok(())
}

fn temperature(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["USER", "INPUT", "AMPLITUDE", "INTERPOLATE", "MIDSIDE", "DRIVING ELSETS"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let r = match kw.param("FILE") {
        // FILE=jobb: värmeanalysen jobb.inp körs med RustyFEM (`steps/heat_stress.rs`)
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
