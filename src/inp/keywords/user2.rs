//! Keywords for field variables and more user subroutines:
//!
//! - `*FIELD, VARIABLE=n[, OP=NEW|MOD][, USER]` (in steps): nodal values for
//!   field variable n at the end of the step, or UFIELD for the nodes (USER),
//! - `*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=n` (called from
//!   `temperature.rs`, which owns *INITIAL CONDITIONS),
//! - `*USER DEFINED FIELD` (material behavior, USDFLD),
//! - `*HEAT GENERATION` (material behavior, HETVAL).
//!
//! The node references are resolved here; the data lines become [node index, value] with
//! the parameter `_NODINDEX` (`materials::field_vars`). For USER the node set's
//! name is stored in `_NAME` (UFIELD is selected by it).

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::materials::field_vars::RESOLVED;
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "FIELD", kind: Kind::Step, handler: field },
    Keyword { name: "HEAT GENERATION", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "USER DEFINED FIELD", kind: Kind::Behavior, handler: behavior },
];

/// Data lines "node or node set, value" → [node index, value]. With `user` the node is enough.
fn resolve(ctx: &InpCtx, kw: &RawKeyword, user: bool) -> Result<RawKeyword> {
    let mut out = kw.clone();
    out.params.push((RESOLVED.to_string(), String::new()));
    out.data.clear();
    let mut name = String::new();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let value = match d.fields.get(1) {
            Some(f) => crate::fem::model::parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} line {}: '{f}' is not a number", kw.file, d.line)))?,
            None if user => 0.0,
            None => return kw.err_at(d.line, "write 'node or node set, value'"),
        };
        if name.is_empty() && crate::inp::parse_label(tok).is_none() {
            name = tok.trim().to_uppercase();
        }
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![n.to_string(), format!("{value:e}")], line: d.line, continued: false });
        }
    }
    if user {
        out.params.push(("_NAME".to_string(), name));
    }
    Ok(out)
}

fn check(kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "INPUT", "AMPLITUDE", "INTERPOLATE", "MIDSIDE", "DRIVING ELSETS", "BSTEP", "ESTEP"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    if let Some(v) = kw.param("VARIABLE") {
        if !crate::inp::parse_label(v).is_some_and(|n| n >= 1) {
            return kw.err(format!("VARIABLE={v} must be an integer ≥ 1"));
        }
    }
    Ok(())
}

fn field(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    check(kw)?;
    let r = resolve(ctx, kw, kw.has("USER"))?;
    ctx.step_mut(kw)?.extra.push(r);
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=n.
pub fn initial_field(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    check(kw)?;
    if kw.has("USER") {
        return kw.err("USER is not supported for TYPE=FIELD (use *FIELD, USER in the first step)");
    }
    let r = resolve(ctx, kw, false)?;
    ctx.model.extra.push(r);
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=STRESS, USER: the elements on the data lines (element set
/// or label; without data lines all elements) get the stress from SIGINI, which
/// is selected by the element set name (`materials::initial_state`).
pub fn initial_stress_user(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "INPUT", "REBAR", "SECTION POINTS", "GEOSTATIC", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported together with USER"));
        }
    }
    let records = kw.records();
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for d in &records {
        let Some(tok) = d.fields.first().filter(|t| !t.trim().is_empty()) else { continue };
        let name = if crate::inp::parse_label(tok).is_some() { String::new() } else { tok.trim().to_uppercase() };
        groups.push((name, ctx.elem_ref(kw, d.line, tok, None)?));
    }
    if groups.is_empty() {
        groups.push((String::new(), (0..ctx.model.elements.len()).collect()));
    }
    for (name, elems) in groups {
        let mut out = kw.clone();
        out.params.push((crate::materials::initial_state::RESOLVED.to_string(), String::new()));
        out.params.push(("_NAME".to_string(), name));
        out.data = elems.into_iter().map(|e| DataLine { fields: vec![e.to_string()], line: kw.line, continued: false }).collect();
        ctx.model.extra.push(out);
    }
    Ok(())
}
