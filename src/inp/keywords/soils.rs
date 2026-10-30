//! Keywords for coupled pore pressure–displacement (geotechnics).
//!
//! - *SOILS (steady-state or CONSOLIDATION) and *GEOSTATIC: procedures, see
//!   `steps/soils.rs`.
//! - *PERMEABILITY, *POROUS BULK MODULI: material behaviors (`materials/porous.rs`).
//! - *CFLOW: concentrated pore fluid flow (outflow positive), stored as a
//!   point load on degree of freedom 8 with reversed sign so that the steps' inheritance rules apply.
//! - *DFLOW/*DSFLOW: prescribed outflow per unit area, the types `DFLOW:Sn`.
//! - *FLOW/*SFLOW: outflow c·(u_w − u_w∞), the types `FLOW:Qn` (magnitude c,
//!   u_w∞ as an extra value).
//! - *INITIAL CONDITIONS, TYPE=PORE PRESSURE, RATIO and STRESS (also
//!   GEOSTATIC) are forwarded here from `temperature.rs`; node and element references
//!   are resolved to indices (the marker `steps::soils::RESOLVED`).

use crate::fem::dof::POR;
use crate::fem::model::{parse_f64, CloadBlock, DataLine, DistLoad, DloadBlock, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};
use crate::steps::soils::RESOLVED;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CFLOW", kind: Kind::Step, handler: cflow },
    Keyword { name: "DFLOW", kind: Kind::Step, handler: dflow },
    Keyword { name: "DSFLOW", kind: Kind::Step, handler: dsflow },
    Keyword { name: "FLOW", kind: Kind::Step, handler: flow },
    Keyword { name: "GEOSTATIC", kind: Kind::Step, handler: geostatic },
    Keyword { name: "PERMEABILITY", kind: Kind::Behavior, handler: permeability },
    Keyword { name: "POROUS BULK MODULI", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SFLOW", kind: Kind::Step, handler: sflow },
    Keyword { name: "SOILS", kind: Kind::Step, handler: soils },
];

fn num(kw: &RawKeyword, line: usize, s: &str) -> Result<f64> {
    parse_f64(s).ok_or_else(|| FemError(format!("{} line {line} (*{}): '{s}' is not a number", kw.file, kw.name)))
}

fn op_new(kw: &RawKeyword) -> bool {
    kw.param_upper("OP").is_some_and(|o| o == "NEW")
}

fn no_amplitude(kw: &RawKeyword) -> Result<()> {
    for p in ["AMPLITUDE", "USER"] {
        if kw.has(p) {
            return kw.err(format!("{p}= is not supported yet for *{}", kw.name));
        }
    }
    Ok(())
}

fn soils(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    procedure(ctx, kw)
}

fn geostatic(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    procedure(ctx, kw)
}

/// *PERMEABILITY may appear several times (TYPE=SATURATION, VELOCITY);
/// parsed in `materials/porous.rs`.
fn permeability(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    m.behaviors.push(kw.clone());
    Ok(())
}

fn cflow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = CloadBlock { op_new: op_new(kw), amplitude: None, follower: false, items: Vec::new(), source: kw.location() };
    for d in &kw.data {
        if d.fields.len() < 2 {
            return kw.err_at(d.line, "write 'node/nset, , flow'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        // the second field is empty (pore fluid); the flow comes last
        let v = num(kw, d.line, d.fields.last().map(|s| s.as_str()).unwrap_or("0"))?;
        for n in nodes {
            // outflow positive in the .inp format → external inflow −q
            block.items.push((n, POR, -v));
        }
    }
    ctx.step_mut(kw)?.cloads.push(block);
    Ok(())
}

fn dblock(kw: &RawKeyword) -> DloadBlock {
    DloadBlock { op_new: op_new(kw), amplitude: None, items: Vec::new(), source: kw.location() }
}

fn face_no(label: &str, prefix: char) -> Option<usize> {
    label.strip_prefix(prefix).and_then(|s| s.parse::<usize>().ok()).filter(|&n| n >= 1)
}

fn dflow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'element/elset, Sn, outflow'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(f) = face_no(&label, 'S') else {
            return kw.err_at(d.line, format!("flow type {label} is not supported (S1, S2, ...)"));
        };
        let q = num(kw, d.line, &d.fields[2])?;
        for e in elems {
            block.items.push(DistLoad { elem: e, kind: format!("DFLOW:S{f}"), magnitude: q, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

fn flow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "write 'element/elset, Qn, sink pressure, seepage coefficient'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(f) = face_no(&label, 'Q') else {
            return kw.err_at(d.line, format!("flow type {label} is not supported (Q1, Q2, ...; QnD/QnNU are not supported yet)"));
        };
        let sink = num(kw, d.line, &d.fields[2])?;
        let c = num(kw, d.line, &d.fields[3])?;
        for e in elems {
            block.items.push(DistLoad { elem: e, kind: format!("FLOW:Q{f}"), magnitude: c, extra: vec![sink] });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

/// Element faces (element, face number) for a *SURFACE.
fn surface_faces(ctx: &InpCtx, kw: &RawKeyword, line: usize, name: &str) -> Result<Vec<(usize, usize)>> {
    let up = name.trim().to_uppercase();
    let mesh = ctx.mesh_ref();
    let Some(s) = mesh.surfaces.get(&up) else {
        return kw.err_at(line, format!("surface '{}' does not exist", name.trim()));
    };
    if s.kind != "ELEMENT" {
        return kw.err_at(line, format!("surface '{}' is not an element-based surface", name.trim()));
    }
    let mut out = Vec::new();
    for (e, f) in &s.faces {
        match face_no(f, 'S') {
            Some(n) => out.push((*e, n)),
            None => return kw.err_at(line, format!("face identifier {f} is not supported for pore fluid flows (S1, S2, ...)")),
        }
    }
    Ok(out)
}

fn dsflow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'surface, S, outflow'");
        }
        let q = num(kw, d.line, &d.fields[2])?;
        for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
            block.items.push(DistLoad { elem: e, kind: format!("DFLOW:S{f}"), magnitude: q, extra: Vec::new() });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

fn sflow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 4 {
            return kw.err_at(d.line, "write 'surface, Q, sink pressure, seepage coefficient'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "Q" {
            return kw.err_at(d.line, format!("flow type {label} is not supported (Q)"));
        }
        let sink = num(kw, d.line, &d.fields[2])?;
        let c = num(kw, d.line, &d.fields[3])?;
        for (e, f) in surface_faces(ctx, kw, d.line, &d.fields[0])? {
            block.items.push(DistLoad { elem: e, kind: format!("FLOW:Q{f}"), magnitude: c, extra: vec![sink] });
        }
    }
    ctx.step_mut(kw)?.dloads.push(block);
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=PORE PRESSURE / RATIO (nodal values, possibly
/// linear in height) and TYPE=STRESS (per element, GEOSTATIC: linear in height
/// with lateral pressure coefficients). Stored with resolved indices in `FeModel::extra`.
pub fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    for p in ["FILE", "USER", "INPUT", "SECTION POINTS", "REBAR", "UNBALANCED STRESS"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet for TYPE={ty}"));
        }
    }
    if ty == "SATURATION" {
        // fully saturated soil: saturation 1 is the default, other values are not supported
        for d in kw.records() {
            let v: Vec<f64> = d.fields[1..].iter().map(|f| num(kw, d.line, f)).collect::<Result<_>>()?;
            if v.first().is_some_and(|&s| (s - 1.0).abs() > 1e-12) {
                return kw.err_at(d.line, "partially saturated soil (saturation < 1) is not supported yet");
            }
        }
        return Ok(());
    }
    let mut out = kw.clone();
    out.params.push((RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let rest: Vec<f64> = d.fields[1..].iter().map(|f| num(kw, d.line, f)).collect::<Result<_>>()?;
        let idx = if ty == "STRESS" { ctx.elem_ref(kw, d.line, tok, None)? } else { ctx.node_ref(kw, d.line, tok, None)? };
        for i in idx {
            let mut fields = vec![i.to_string()];
            fields.extend(rest.iter().map(|v| format!("{v:e}")));
            out.data.push(DataLine { fields, line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
