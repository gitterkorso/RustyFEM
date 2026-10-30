//! Analysis infrastructure: restart, submodeling, substructures and matrices.
//!
//! - *RESTART, WRITE/READ (`steps/restart.rs`)
//! - *SUBMODEL and *BOUNDARY, SUBMODEL, STEP= (`steps/submodel.rs`;
//!   *BOUNDARY calls `boundary_submodel`)
//! - *SUBSTRUCTURE GENERATE, *RETAINED NODAL DOFS (`steps/substructure.rs`),
//!   *SUBSTRUCTURE PROPERTY (section for TYPE=Zn, `elements/matrix_element.rs`)
//! - *MATRIX GENERATE, *MATRIX OUTPUT (`steps/matrix_generate.rs`)
//! - *IMPORT (state from a previous analysis, `steps/restart.rs`)
//! - *MATRIX INPUT, *MATRIX ASSEMBLE: the matrices become a matrix element over
//!   the matrix's nodes (`elements/matrix_element.rs`)

use crate::fem::dof::{self, Dof};
use crate::fem::model::{parse_f64, BcBlock, DataLine, RawKeyword, SectionDef, SectionKind};
use crate::inp::{parse_label, procedure, required, store_model, store_step, InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};
use std::collections::BTreeMap;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "IMPORT", kind: Kind::Model, handler: import },
    Keyword { name: "MATRIX ASSEMBLE", kind: Kind::Model, handler: matrix_assemble },
    Keyword { name: "MATRIX GENERATE", kind: Kind::Step, handler: procedure },
    Keyword { name: "MATRIX INPUT", kind: Kind::Model, handler: matrix_input },
    Keyword { name: "MATRIX OUTPUT", kind: Kind::Step, handler: store_step },
    Keyword { name: "RESTART", kind: Kind::Both, handler: restart },
    Keyword { name: "RETAINED NODAL DOFS", kind: Kind::Step, handler: retained },
    Keyword { name: "SUBMODEL", kind: Kind::Model, handler: submodel },
    Keyword { name: "SUBSTRUCTURE GENERATE", kind: Kind::Step, handler: procedure },
    Keyword { name: "SUBSTRUCTURE PROPERTY", kind: Kind::Model, handler: substructure_property },
];

/// Marker on keywords whose nodes have already been converted to model indices.
pub const RESOLVED: &str = "_RESOLVED";

fn restart(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("READ") {
        if ctx.step.is_some() {
            return kw.err("*RESTART, READ must appear in model data (before the first *STEP)");
        }
        if kw.param("STEP").is_none() {
            return kw.err("*RESTART, READ lacks STEP=");
        }
        return store_model(ctx, kw);
    }
    if !kw.has("WRITE") {
        return kw.err("give WRITE or READ");
    }
    if ctx.step.is_some() { store_step(ctx, kw) } else { store_model(ctx, kw) }
}

fn dof_range(kw: &RawKeyword, d: &DataLine, first: Option<&String>, last: Option<&String>) -> Result<Vec<Dof>> {
    let Some(first) = first.and_then(|s| parse_label(s)) else {
        return kw.err_at(d.line, "the first degree of freedom is missing");
    };
    let last = match last.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        Some(s) => parse_label(s).ok_or_else(|| FemError(format!("{} line {}: '{s}' is not a degree of freedom", kw.file, d.line)))?,
        None => first,
    };
    if first < 1 || last < first {
        return kw.err_at(d.line, "invalid degree-of-freedom range");
    }
    let mut v = Vec::new();
    for k in first..=last {
        if dof::slot(k as Dof).is_none() {
            return kw.err_at(d.line, format!("degree of freedom {k} is not supported (1–7 and 11)"));
        }
        v.push(k as Dof);
    }
    Ok(v)
}

fn submodel(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if let Some(t) = kw.param_upper("TYPE") {
        if t != "NODE" {
            return kw.err(format!("TYPE={t} is not supported (only node-based submodeling)"));
        }
    }
    for p in ["SHELL TO SOLID", "SHELL THICKNESS"] {
        if kw.has(p) {
            return kw.err(format!("{p} (shell-to-solid submodeling) is not supported"));
        }
    }
    let mut nodes = Vec::new();
    for d in &kw.data {
        for f in d.fields.iter().filter(|s| !s.trim().is_empty()) {
            nodes.extend(ctx.node_ref(kw, d.line, f, None)?);
        }
    }
    if nodes.is_empty() {
        return kw.err("*SUBMODEL has no driven nodes");
    }
    ctx.mesh().add_to_nset(crate::steps::submodel::DRIVEN_NSET, &nodes);
    store_model(ctx, kw)
}

/// *BOUNDARY, SUBMODEL, STEP=n: driven degrees of freedom (the values are set by
/// `steps::submodel::prepare` when the analysis starts).
pub fn boundary_submodel(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.step.is_none() {
        return kw.err("*BOUNDARY, SUBMODEL may only appear in a step");
    }
    let gstep = match kw.param("STEP") {
        Some(v) => parse_label(v).filter(|&n| n > 0).ok_or_else(|| FemError(format!("{}: STEP={v} must be a positive integer", kw.location())))?,
        None => 1,
    };
    if kw.has("AMPLITUDE") {
        return kw.err("AMPLITUDE= with SUBMODEL is not supported");
    }
    let mut block = BcBlock {
        op_new: kw.param_upper("OP").is_some_and(|o| o == "NEW"),
        kind: format!("{}{gstep}", crate::steps::submodel::SUBMODEL_KIND),
        amplitude: None,
        items: Vec::new(),
        source: kw.location(),
    };
    for d in &kw.data {
        if d.fields.len() < 2 {
            return kw.err_at(d.line, "write 'nset, first dof[, last dof]'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        let dofs = dof_range(kw, d, d.fields.get(1), d.fields.get(2))?;
        for &n in &nodes {
            for &dd in &dofs {
                block.items.push((n, dd, 0.0));
            }
        }
    }
    ctx.step_mut(kw)?.bcs.push(block);
    Ok(())
}

fn retained(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut out = kw.clone();
    out.params.push((RESOLVED.into(), String::new()));
    out.data.clear();
    for d in &kw.data {
        let f: Vec<&String> = d.fields.iter().collect();
        if f.is_empty() || f[0].trim().is_empty() {
            continue;
        }
        let nodes = ctx.node_ref(kw, d.line, f[0], None)?;
        let dofs = dof_range(kw, d, f.get(1).copied(), f.get(2).copied())?;
        for &n in &nodes {
            for &dd in &dofs {
                out.data.push(DataLine { fields: vec![n.to_string(), dd.to_string()], line: d.line, continued: false });
            }
        }
    }
    store_step(ctx, &out)
}

fn substructure_property(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let elset = required(kw, "ELSET")?.to_uppercase();
    if kw.data.len() > 2 {
        ctx.warn(kw, "mirroring (third data line) is not supported and is ignored");
    }
    ctx.mesh().sections.push(SectionDef { elset, material: String::new(), material_index: None, kind: SectionKind::Other, raw: kw.clone() });
    Ok(())
}

// ---------------------------------------------------------------------------
// *MATRIX INPUT / *MATRIX ASSEMBLE

/// Reads the matrix lines ("node, dof, node, dof, value") from INPUT= or
/// the data lines and stores them as data lines in the keyword.
fn matrix_input(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    required(kw, "NAME")?;
    if let Some(t) = kw.param_upper("MATRIX TYPE") {
        if t != "SYMMETRIC" && t != "UNSYMMETRIC" {
            return kw.err(format!("MATRIX TYPE={t} must be SYMMETRIC or UNSYMMETRIC"));
        }
    }
    if let Some(f) = kw.param_upper("FORMAT") {
        if f != "MATRIX INPUT" && f != "LABELS" {
            return kw.err(format!("FORMAT={f} is not supported (only FORMAT=MATRIX INPUT)"));
        }
    }
    let mut out = kw.clone();
    if let Some(file) = kw.param("INPUT") {
        let dir = ctx.dir.clone().unwrap_or_else(|| std::path::PathBuf::from("."));
        let path = dir.join(file.trim().trim_matches('"'));
        let text = std::fs::read_to_string(&path).map_err(|e| FemError(format!("{}: could not read {}: {e}", kw.location(), path.display())))?;
        out.data.clear();
        for (i, line) in text.lines().enumerate() {
            let t = line.trim();
            if t.is_empty() || t.starts_with("**") {
                continue;
            }
            let fields: Vec<String> = t.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            out.data.push(DataLine { fields, line: i + 1, continued: false });
        }
        if !ctx.included.contains(&path) {
            ctx.included.push(path);
        }
    }
    // check the lines right away
    for d in &out.data {
        if d.fields.len() != 5 {
            return kw.err_at(d.line, format!("a matrix line must be 'node, dof, node, dof, value' (the line has {} fields)", d.fields.len()));
        }
        if parse_f64(&d.fields[4]).is_none() {
            return kw.err_at(d.line, format!("'{}' is not a number", d.fields[4]));
        }
    }
    store_model(ctx, &out)
}

fn matrix_node(ctx: &mut InpCtx, kw: &RawKeyword, d: &DataLine, tok: &str) -> Result<usize> {
    let inst_name = kw.param("INSTANCE").map(|s| s.to_uppercase());
    let n = ctx.node_ref(kw, d.line, tok, inst_name.as_deref())?;
    match n.as_slice() {
        [one] => Ok(*one),
        _ => kw.err_at(d.line, format!("'{tok}' must be a node label")),
    }
}

/// Creates a matrix element from the matrices that *MATRIX INPUT has read.
fn matrix_assemble(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["VISCOUS DAMPING", "STRUCTURAL DAMPING", "LOAD"] {
        if kw.has(p) {
            return kw.err(format!("{p}= is not supported yet (only STIFFNESS= and MASS=)"));
        }
    }
    let find = |ctx: &InpCtx, name: &str| -> Option<RawKeyword> {
        let up = name.trim().to_uppercase();
        ctx.mesh_ref()
            .extra
            .iter()
            .chain(ctx.model.extra.iter())
            .rev()
            .find(|k| k.name == "MATRIX INPUT" && k.param_upper("NAME").as_deref() == Some(up.as_str()))
            .cloned()
    };
    let mut mats: Vec<(usize, RawKeyword)> = Vec::new();
    for (i, p) in ["STIFFNESS", "MASS"].iter().enumerate() {
        if let Some(name) = kw.param(p) {
            match find(ctx, name) {
                Some(m) => mats.push((i, m)),
                None => return kw.err(format!("{p}={name}: no *MATRIX INPUT with NAME={name}")),
            }
        }
    }
    if mats.is_empty() {
        return kw.err("give STIFFNESS= and/or MASS=");
    }
    // (node, dof) → values
    let mut entries: Vec<BTreeMap<((usize, Dof), (usize, Dof)), f64>> = vec![BTreeMap::new(), BTreeMap::new()];
    let mut dofs: BTreeMap<usize, Vec<Dof>> = BTreeMap::new();
    for (which, m) in &mats {
        let scale = match m.param("SCALE FACTOR") {
            Some(s) => parse_f64(s).ok_or_else(|| FemError(format!("{}: SCALE FACTOR={s} is not a number", m.location())))?,
            None => 1.0,
        };
        let sym = m.param_upper("MATRIX TYPE").is_none_or(|t| t == "SYMMETRIC");
        for d in &m.data {
            let na = matrix_node(ctx, m, d, &d.fields[0])?;
            let nb = matrix_node(ctx, m, d, &d.fields[2])?;
            let da = parse_label(&d.fields[1]).filter(|&x| dof::slot(x as Dof).is_some()).ok_or_else(|| FemError(format!("{} line {}: '{}' is not a degree of freedom", m.file, d.line, d.fields[1])))? as Dof;
            let db = parse_label(&d.fields[3]).filter(|&x| dof::slot(x as Dof).is_some()).ok_or_else(|| FemError(format!("{} line {}: '{}' is not a degree of freedom", m.file, d.line, d.fields[3])))? as Dof;
            let v = parse_f64(&d.fields[4]).unwrap_or(0.0) * scale;
            entries[*which].insert(((na, da), (nb, db)), v);
            if sym {
                entries[*which].insert(((nb, db), (na, da)), v);
            }
            for (n, dd) in [(na, da), (nb, db)] {
                let e = dofs.entry(n).or_default();
                if !e.contains(&dd) {
                    e.push(dd);
                }
            }
        }
    }
    let nodes: Vec<usize> = dofs.keys().copied().collect();
    let mut node_dofs: Vec<Vec<Dof>> = Vec::new();
    let mut index: BTreeMap<(usize, Dof), usize> = BTreeMap::new();
    for &n in &nodes {
        let mut d = dofs[&n].clone();
        d.sort_unstable();
        for &x in &d {
            let k = index.len();
            index.insert((n, x), k);
        }
        node_dofs.push(d);
    }
    let nd = index.len();
    let dense = |m: &BTreeMap<((usize, Dof), (usize, Dof)), f64>| -> Vec<f64> {
        let mut a = vec![0.0; nd * nd];
        for (&(ra, cb), &v) in m {
            a[index[&ra] * nd + index[&cb]] = v;
        }
        a
    };
    let k = dense(&entries[0]);
    let m = mats.iter().any(|(w, _)| *w == 1).then(|| dense(&entries[1]));
    let name = format!("MATRIX-{}", kw.param("STIFFNESS").or(kw.param("MASS")).unwrap_or("").to_uppercase());
    let et = crate::elements::matrix_element::MatrixElement::leak(&name, node_dofs, k, m, false);
    let inst = ctx.mesh_instance();
    // element labels that do not clash with the user's
    let label = 1_000_000_000 + ctx.mesh_ref().elements.len() as i64;
    let id = ctx.mesh().add_element(et, inst, label, nodes).or_else(|e| kw.err(e.0))?;
    if let Some(set) = kw.param("ELSET") {
        ctx.mesh().add_to_elset(set, &[id]);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// *IMPORT

/// Stores *IMPORT with the element sets converted to element indices.
fn import(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut out = kw.clone();
    out.params.push((RESOLVED.into(), String::new()));
    out.data.clear();
    for d in &kw.data {
        for f in d.fields.iter().filter(|s| !s.trim().is_empty()) {
            let elems = ctx.elem_ref(kw, d.line, f, None)?;
            out.data.push(DataLine { fields: elems.iter().map(|e| e.to_string()).collect(), line: d.line, continued: false });
        }
    }
    store_model(ctx, &out)
}
