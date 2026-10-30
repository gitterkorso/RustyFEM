//! Nyckelord för kopplad porvattentryck–förskjutning (geoteknik).
//!
//! - *SOILS (stationär eller CONSOLIDATION) och *GEOSTATIC: procedurer, se
//!   `steps/soils.rs`.
//! - *PERMEABILITY, *POROUS BULK MODULI: materialbeteenden (`materials/porous.rs`).
//! - *CFLOW: koncentrerat porvätskeflöde (utflöde positivt), lagras som
//!   punktlast på frihetsgrad 8 med omvänt tecken så att stegens arvsregler gäller.
//! - *DFLOW/*DSFLOW: föreskrivet utflöde per ytenhet, typerna `DFLOW:Sn`.
//! - *FLOW/*SFLOW: utflöde c·(u_w − u_w∞), typerna `FLOW:Qn` (storlek c,
//!   u_w∞ som extra värde).
//! - *INITIAL CONDITIONS, TYPE=PORE PRESSURE, RATIO och STRESS (även
//!   GEOSTATIC) skickas hit från `temperature.rs`; nod- och elementreferenser
//!   löses upp till index (markören `steps::soils::RESOLVED`).

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
    parse_f64(s).ok_or_else(|| FemError(format!("{} rad {line} (*{}): '{s}' är inget tal", kw.file, kw.name)))
}

fn op_new(kw: &RawKeyword) -> bool {
    kw.param_upper("OP").is_some_and(|o| o == "NEW")
}

fn no_amplitude(kw: &RawKeyword) -> Result<()> {
    for p in ["AMPLITUDE", "USER"] {
        if kw.has(p) {
            return kw.err(format!("{p}= stöds inte ännu för *{}", kw.name));
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

/// *PERMEABILITY får förekomma flera gånger (TYPE=SATURATION, VELOCITY);
/// tolkas i `materials/porous.rs`.
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
            return kw.err_at(d.line, "skriv 'nod/nset, , flöde'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        // andra fältet är tomt (porvätska); flödet står sist
        let v = num(kw, d.line, d.fields.last().map(|s| s.as_str()).unwrap_or("0"))?;
        for n in nodes {
            // utflöde positivt i .inp-formatet → yttre inflöde −q
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
            return kw.err_at(d.line, "skriv 'element/elset, Sn, utflöde'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(f) = face_no(&label, 'S') else {
            return kw.err_at(d.line, format!("flödestypen {label} stöds inte (S1, S2, ...)"));
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
            return kw.err_at(d.line, "skriv 'element/elset, Qn, sänktryck, läckagekoefficient'");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let label = d.fields[1].trim().to_uppercase();
        let Some(f) = face_no(&label, 'Q') else {
            return kw.err_at(d.line, format!("flödestypen {label} stöds inte (Q1, Q2, ...; QnD/QnNU stöds inte ännu)"));
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

/// Elementytor (element, ytnummer) för en *SURFACE.
fn surface_faces(ctx: &InpCtx, kw: &RawKeyword, line: usize, name: &str) -> Result<Vec<(usize, usize)>> {
    let up = name.trim().to_uppercase();
    let mesh = ctx.mesh_ref();
    let Some(s) = mesh.surfaces.get(&up) else {
        return kw.err_at(line, format!("ytan '{}' finns inte", name.trim()));
    };
    if s.kind != "ELEMENT" {
        return kw.err_at(line, format!("ytan '{}' är inte en elementyta", name.trim()));
    }
    let mut out = Vec::new();
    for (e, f) in &s.faces {
        match face_no(f, 'S') {
            Some(n) => out.push((*e, n)),
            None => return kw.err_at(line, format!("ytbeteckningen {f} stöds inte för porvätskeflöden (S1, S2, ...)")),
        }
    }
    Ok(out)
}

fn dsflow(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    no_amplitude(kw)?;
    let mut block = dblock(kw);
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'yta, S, utflöde'");
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
            return kw.err_at(d.line, "skriv 'yta, Q, sänktryck, läckagekoefficient'");
        }
        let label = d.fields[1].trim().to_uppercase();
        if label != "Q" {
            return kw.err_at(d.line, format!("flödestypen {label} stöds inte (Q)"));
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

/// *INITIAL CONDITIONS, TYPE=PORE PRESSURE / RATIO (nodvärden, eventuellt
/// linjära i höjden) och TYPE=STRESS (per element, GEOSTATIC: linjär i höjden
/// med sidotryckskoefficienter). Sparas med upplösta index i `FeModel::extra`.
pub fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    for p in ["FILE", "USER", "INPUT", "SECTION POINTS", "REBAR", "UNBALANCED STRESS"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu för TYPE={ty}"));
        }
    }
    if ty == "SATURATION" {
        // helt mättad jord: mättnadsgraden 1 är standard, andra värden stöds inte
        for d in kw.records() {
            let v: Vec<f64> = d.fields[1..].iter().map(|f| num(kw, d.line, f)).collect::<Result<_>>()?;
            if v.first().is_some_and(|&s| (s - 1.0).abs() > 1e-12) {
                return kw.err_at(d.line, "delvis mättad jord (mättnadsgrad < 1) stöds inte ännu");
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
