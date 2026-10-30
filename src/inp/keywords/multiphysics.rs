//! Keywords for multiphysics: coupled thermal-electrical analysis,
//! piezoelectricity, mass diffusion and thermal contact.
//!
//! - *COUPLED THERMAL-ELECTRICAL, *MASS DIFFUSION: procedures
//!   (`steps/thermal_electric.rs`, `steps/mass_diffusion.rs`).
//! - *ELECTRICAL CONDUCTIVITY, *JOULE HEAT FRACTION, *PIEZOELECTRIC,
//!   *DIELECTRIC, *DIFFUSIVITY, *SOLUBILITY: material behaviors
//!   (`materials/multiphysics.rs`).
//! - *CECURRENT and *CECHARGE: concentrated current/charge on degree of freedom 9,
//!   stored as point loads so that the steps' inheritance rules apply.
//! - *DECURRENT (CBF per volume, CSn per area) and *DSECURRENT (surface, CS):
//!   distributed currents with the types `ECUR:CBF` and `ECUR:CSn`.
//! - *GAP CONDUCTANCE, *GAP RADIATION: suboptions of *SURFACE
//!   INTERACTION, stored in `FeModel::extra` (`interactions/thermal_contact.rs`).
//! - *INITIAL CONDITIONS, TYPE=CONCENTRATION is forwarded here from
//!   `temperature.rs` and stored with resolved node indices [node, value].

use crate::fem::dof::EPOT;
use crate::fem::model::{parse_f64, CloadBlock, DataLine, DistLoad, DloadBlock, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{parse_label, procedure, store_model, InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CECHARGE", kind: Kind::Step, handler: cecharge },
    Keyword { name: "CECURRENT", kind: Kind::Step, handler: cecurrent },
    Keyword { name: "COUPLED THERMAL-ELECTRICAL", kind: Kind::Step, handler: procedure },
    Keyword { name: "DECURRENT", kind: Kind::Step, handler: decurrent },
    Keyword { name: "DIELECTRIC", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DIFFUSIVITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DSECURRENT", kind: Kind::Step, handler: dsecurrent },
    Keyword { name: "ELECTRICAL CONDUCTIVITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "GAP CONDUCTANCE", kind: Kind::Model, handler: store_model },
    Keyword { name: "GAP RADIATION", kind: Kind::Model, handler: store_model },
    Keyword { name: "JOULE HEAT FRACTION", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "MASS DIFFUSION", kind: Kind::Step, handler: procedure },
    Keyword { name: "PIEZOELECTRIC", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SOLUBILITY", kind: Kind::Behavior, handler: behavior },
];

fn num(kw: &RawKeyword, line: usize, s: &str) -> Result<f64> {
    parse_f64(s).ok_or_else(|| FemError(format!("{} line {line} (*{}): '{s}' is not a number", kw.file, kw.name)))
}

fn op_new(kw: &RawKeyword) -> bool {
    kw.param_upper("OP").is_some_and(|o| o == "NEW")
}

fn no_amplitude(kw: &RawKeyword) -> Result<()> {
    for p in ["AMPLITUDE", "USER", "FIELD"] {
        if kw.has(p) {
            return kw.err(format!("{p}= is not supported yet"));
        }
    }
    Ok(())
}

/// Point load on degree of freedom 9 (current or charge).
fn point9(ctx: &mut InpCtx, kw: &RawKeyword, what: &str) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = CloadBlock { op_new: op_new(kw), amplitude: None, follower: false, items: Vec::new(), source: kw.location() };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, format!("write 'node/nset, 9, {what}'"));
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        if parse_label(&d.fields[1]).unwrap_or(0) != 9 {
            return kw.err_at(d.line, format!("degree of freedom {} – *{} applies to the electric potential (9)", d.fields[1].trim(), kw.name));
        }
        let v = num(kw, d.line, &d.fields[2])?;
        for n in nodes {
            block.items.push((n, EPOT, v));
        }
    }
    ctx.step_mut(kw)?.cloads.push(block);
    Ok(())
}

fn cecurrent(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    point9(ctx, kw, "current")
}

fn cecharge(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    point9(ctx, kw, "charge")
}

fn decurrent(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = DloadBlock { op_new: op_new(kw), amplitude: None, items: Vec::new(), source: kw.location() };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'element/elset, CBF or CSn, current density'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        // Area physics3: CJ (volume current density with direction) in *ELECTROMAGNETIC
        let ok = label == "CBF" || label == "CJ" || label.strip_prefix("CS").and_then(|s| s.parse::<usize>().ok()).is_some_and(|n| n >= 1);
        if !ok {
            return kw.err_at(d.line, format!("current type {label} is not supported (CBF, CS1, CS2, ..., CJ)"));
        }
        let q = num(kw, d.line, &d.fields[2])?;
        for e in elems {
            let extra = if label == "CJ" { (3..6).map(|i| d.fields.get(i).map_or(Ok(0.0), |f| num(kw, d.line, f))).collect::<Result<Vec<f64>>>()? } else { Vec::new() };
            block.items.push(DistLoad { elem: e, kind: format!("ECUR:{label}"), magnitude: q, extra });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

fn dsecurrent(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = DloadBlock { op_new: op_new(kw), amplitude: None, items: Vec::new(), source: kw.location() };
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'surface, CS, current density'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "CS" {
            return kw.err_at(d.line, format!("current type {label} is not supported (CS)"));
        }
        let q = num(kw, d.line, &d.fields[2])?;
        let up = d.fields[0].trim().to_uppercase();
        let faces = {
            let mesh = ctx.mesh_ref();
            let Some(s) = mesh.surfaces.get(&up) else { return kw.err_at(d.line, format!("surface '{}' does not exist", d.fields[0].trim())) };
            s.faces.clone()
        };
        for (e, f) in faces {
            let Some(n) = f.strip_prefix('S').and_then(|s| s.parse::<usize>().ok()) else {
                return kw.err_at(d.line, format!("face identifier {f} is not supported (S1, S2, ...)"));
            };
            block.items.push(DistLoad { elem: e, kind: format!("ECUR:CS{n}"), magnitude: q, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=CONCENTRATION (normalized concentration per node).
pub fn initial_concentration(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "USER", "INPUT"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    let mut out = kw.clone();
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let v = match d.fields.get(1) {
            Some(s) => num(kw, d.line, s)?,
            None => 0.0,
        };
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![n.to_string(), format!("{v:e}")], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
