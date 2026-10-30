//! Nyckelord för fältvariabler och fler användarrutiner:
//!
//! - `*FIELD, VARIABLE=n[, OP=NEW|MOD][, USER]` (i steg): nodvärden för
//!   fältvariabel n vid stegets slut, eller UFIELD för noderna (USER),
//! - `*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=n` (anropas från
//!   `temperature.rs`, som äger *INITIAL CONDITIONS),
//! - `*USER DEFINED FIELD` (materialbeteende, USDFLD),
//! - `*HEAT GENERATION` (materialbeteende, HETVAL).
//!
//! Nodreferenserna löses upp här; dataraderna blir [nodindex, värde] med
//! parametern `_NODINDEX` (`materials::field_vars`). För USER sparas nodsetets
//! namn i `_NAME` (UFIELD väljs med det).

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

/// Datarader "nod eller nodset, värde" → [nodindex, värde]. Med `user` räcker noden.
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
            Some(f) => crate::fem::model::parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{f}' är inget tal", kw.file, d.line)))?,
            None if user => 0.0,
            None => return kw.err_at(d.line, "skriv 'nod eller nodset, värde'"),
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
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    if let Some(v) = kw.param("VARIABLE") {
        if !crate::inp::parse_label(v).is_some_and(|n| n >= 1) {
            return kw.err(format!("VARIABLE={v} ska vara ett heltal ≥ 1"));
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
        return kw.err("USER stöds inte för TYPE=FIELD (använd *FIELD, USER i första steget)");
    }
    let r = resolve(ctx, kw, false)?;
    ctx.model.extra.push(r);
    Ok(())
}

/// *INITIAL CONDITIONS, TYPE=STRESS, USER: elementen på dataraderna (elementset
/// eller nummer; utan datarader alla element) får spänningen från SIGINI, som
/// väljs med elementsetets namn (`materials::initial_state`).
pub fn initial_stress_user(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "INPUT", "REBAR", "SECTION POINTS", "GEOSTATIC", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte tillsammans med USER"));
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
