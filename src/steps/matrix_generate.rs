//! *MATRIX GENERATE, STIFFNESS, MASS[, ELEMENT BY ELEMENT] and
//! *MATRIX OUTPUT, STIFFNESS, MASS, FORMAT=MATRIX INPUT | LABELS | COORDINATE.
//!
//! The step builds the stiffness (the tangent in the base state) and the mass (the same
//! mass as *FREQUENCY, `frequency::mass_kind`) and writes them as
//! `.mtx` files in the job directory: `<job>_STIF<step>.mtx` and
//! `<job>_MASS<step>.mtx`. Formats:
//!
//! - MATRIX INPUT / LABELS: `node,dof, node,dof,  value`, the lower triangle
//!   column by column (nodes in ascending number, degrees of freedom ascending).
//!   Constrained degrees of freedom (*BOUNDARY) get the diagonal 1e36.
//!   The file can be read back with *MATRIX INPUT.
//! - COORDINATE: `row column  value` (both triangles, 1-based numbering in
//!   the same order).
//! - ELEMENT BY ELEMENT: `element, node,dof, node,dof,  value`, the upper triangle of the element matrix
//!   column by column in the element node order, without boundary conditions.
//!
//! Zeros are not written. LOAD, VISCOUS DAMPING and STRUCTURAL DAMPING are not supported.

use super::api::{StepCtx, StepType};
use super::substructure::{assemble_pairs, element_stiffness, PairMatrix};
use crate::fem::dof;
use crate::fem::model::{FeModel, StepDef};
use crate::model::{FemError, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

pub struct MatrixGenerate;

pub static STEP_TYPES: &[&dyn StepType] = &[&MatrixGenerate];

/// Number in the `.mtx` format: `% .15e` with a two-digit exponent.
pub fn fmt_e(v: f64) -> String {
    let s = format!("{v:.15e}");
    let (mant, exp) = s.split_once('e').unwrap_or((&s, "0"));
    let e: i32 = exp.parse().unwrap_or(0);
    let sign = if e < 0 { '-' } else { '+' };
    let lead = if v.is_sign_negative() { "" } else { " " };
    format!("{lead}{mant}e{sign}{:02}", e.abs())
}

/// The node name in the matrix files: number, or INSTANCE.number in models with instances.
fn node_tag(m: &FeModel, n: usize) -> String {
    let inst = &m.instances[m.node_inst[n] as usize];
    if inst == crate::fem::model::FLAT_INSTANCE || m.instances.len() == 1 {
        m.node_label[n].to_string()
    } else {
        format!("{inst}.{}", m.node_label[n])
    }
}

/// The degree of freedom order: (node, slot) with the nodes in ascending number.
fn dof_order(m: &FeModel, active: &[u16]) -> Vec<(usize, usize)> {
    let mut nodes: Vec<usize> = (0..m.nodes.len()).filter(|&n| active[n] != 0).collect();
    nodes.sort_by_key(|&n| (m.node_inst[n], m.node_label[n]));
    let mut out = Vec::new();
    for n in nodes {
        for s in 0..dof::SLOTS {
            if active[n] & (1 << s) != 0 {
                out.push((n, s));
            }
        }
    }
    out
}

/// Writes a global matrix in the MATRIX INPUT or COORDINATE format.
pub fn global_text(m: &FeModel, active: &[u16], a: &PairMatrix, coordinate: bool) -> String {
    let order = dof_order(m, active);
    let pos: BTreeMap<(usize, usize), usize> = order.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    // column-wise: (column, row) → value, only row ≥ column for MATRIX INPUT
    let mut cols: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for (&(r, c), &v) in a {
        let (Some(&ri), Some(&ci)) = (pos.get(&r), pos.get(&c)) else { continue };
        if v == 0.0 || (!coordinate && ri < ci) {
            continue;
        }
        cols.insert(if coordinate { (ri, ci) } else { (ci, ri) }, v);
    }
    let mut s = String::new();
    for (&(x, y), &v) in &cols {
        if coordinate {
            let _ = writeln!(s, "{} {} {}", x + 1, y + 1, fmt_e(v));
        } else {
            let (c, r) = (order[x], order[y]);
            let _ = writeln!(s, "{},{}, {},{}, {}", node_tag(m, r.0), dof::SLOT_DOF[r.1], node_tag(m, c.0), dof::SLOT_DOF[c.1], fmt_e(v));
        }
    }
    s
}

fn wants(kw: &crate::fem::model::RawKeyword, what: &str) -> bool {
    kw.has(what) && kw.flag(what, true).unwrap_or(true)
}

impl StepType for MatrixGenerate {
    fn name(&self) -> &'static str {
        "MATRIX GENERATE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "MATRIX GENERATE").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        let m = p.model;
        let kw = ctx.step.procedure.as_ref().unwrap();
        for what in ["LOAD", "VISCOUS DAMPING", "STRUCTURAL DAMPING"] {
            if wants(kw, what) {
                return kw.err(format!("{what} is not supported yet (only STIFFNESS and MASS)"));
            }
        }
        let gen_k = wants(kw, "STIFFNESS");
        let gen_m = wants(kw, "MASS");
        let ebe = kw.has("ELEMENT BY ELEMENT");
        let elems: Option<BTreeSet<usize>> = match kw.param("ELSET") {
            Some(s) => Some(m.elsets.get(&s.trim().to_uppercase()).ok_or_else(|| FemError(format!("{}: the element set {s} does not exist", kw.location())))?.iter().copied().collect()),
            None => None,
        };
        let step_no = ctx.index + 1;
        let mut files: Vec<(String, String)> = Vec::new();
        let outputs: Vec<_> = ctx.step.extra.iter().filter(|k| k.name == "MATRIX OUTPUT").cloned().collect();
        if ebe {
            for o in &outputs {
                for (what, tag, on) in [("STIFFNESS", "STIF", gen_k), ("MASS", "MASS", gen_m)] {
                    if !wants(o, what) || !on {
                        continue;
                    }
                    let mut s = String::new();
                    let mut ids: Vec<usize> = (0..m.elements.len()).filter(|e| elems.as_ref().is_none_or(|s| s.contains(e))).collect();
                    ids.sort_by_key(|&e| (m.elements[e].inst, m.elements[e].label));
                    for e in ids {
                        let ed = &p.edofs[e];
                        let a = if what == "STIFFNESS" {
                            element_stiffness(p, e, ctx.state, ctx.nlgeom)?
                        } else {
                            let et = m.elements[e].etype;
                            et.mass(&p.ctx(e, ctx.nlgeom, ctx.state.time, 0.0), super::frequency::mass_kind(et)).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?
                        };
                        let n = ed.len();
                        for c in 0..n {
                            for r in 0..=c {
                                let v = a[r * n + c];
                                if v != 0.0 {
                                    let _ = writeln!(s, "{}, {},{}, {},{}, {}", m.elements[e].label, node_tag(m, ed[r].0), dof::SLOT_DOF[ed[r].1], node_tag(m, ed[c].0), dof::SLOT_DOF[ed[c].1], fmt_e(v));
                                }
                            }
                        }
                    }
                    files.push((format!("_{tag}{step_no}.mtx"), s));
                }
            }
        } else if !outputs.is_empty() {
            let (mut kp, mp) = assemble_pairs(p, ctx.state, ctx.nlgeom, gen_m, elems.as_ref())?;
            // constrained degrees of freedom: the diagonal 1e36
            for &(n, s) in ctx.loads.bc.keys() {
                if p.active[n] & (1 << s) != 0 {
                    kp.insert(((n, s), (n, s)), 1.0e36);
                }
            }
            for o in &outputs {
                let coordinate = o.param_upper("FORMAT").is_some_and(|f| f == "COORDINATE");
                if let Some(f) = o.param_upper("FORMAT") {
                    if f != "COORDINATE" && f != "MATRIX INPUT" && f != "LABELS" {
                        return o.err(format!("FORMAT={f} is not supported (MATRIX INPUT, LABELS or COORDINATE)"));
                    }
                }
                if wants(o, "STIFFNESS") && gen_k {
                    files.push((format!("_STIF{step_no}.mtx"), global_text(m, &p.active, &kp, coordinate)));
                }
                if let (true, true, Some(mp)) = (wants(o, "MASS"), gen_m, mp.as_ref()) {
                    files.push((format!("_MASS{step_no}.mtx"), global_text(m, &p.active, mp, coordinate)));
                }
            }
        }
        match super::restart::job() {
            Some(j) => {
                for (suffix, text) in files {
                    let path = j.file(&suffix);
                    std::fs::write(&path, text).map_err(|e| FemError(format!("could not write {}: {e}", path.display())))?;
                    if ctx.opts.verbose {
                        eprintln!("  matrix: {}", path.display());
                    }
                }
            }
            None if !files.is_empty() && ctx.opts.verbose => eprintln!("warning: *MATRIX OUTPUT: no job directory – the matrices are not written"),
            None => {}
        }
        ctx.out.increments = 1;
        Ok(())
    }
}
