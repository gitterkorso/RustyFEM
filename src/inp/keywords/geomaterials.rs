//! Geotekniska material och betong: Cam-clay (*CLAY PLASTICITY,
//! *CLAY HARDENING), porös elasticitet (*POROUS ELASTIC), Mohr–Coulombs
//! dragbegränsning (*TENSION CUTOFF), utsmetad sprickbildning (*CONCRETE,
//! *TENSION STIFFENING, *SHEAR RETENTION, *FAILURE RATIOS) och
//! Drucker–Prager Cap (*CAP PLASTICITY, *CAP HARDENING).
//!
//! Beteendena sparas i materialet och tolkas av `materials/clay.rs`,
//! `materials/mohr_coulomb.rs` (`mohr_coulomb_cutoff.rs`),
//! `materials/concrete_smeared.rs` och `materials/cap.rs`.
//!
//! *INITIAL CONDITIONS, TYPE=STRESS och TYPE=RATIO (nyckelordet ägs av
//! `temperature.rs`, som skickar dem hit) löses upp här och används av
//! `materials/initial_state.rs`.

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::materials::initial_state::RESOLVED;
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CAP HARDENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CAP PLASTICITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CLAY HARDENING", kind: Kind::Behavior, handler: repeated },
    Keyword { name: "CLAY PLASTICITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CONCRETE", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "FAILURE RATIOS", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "POROUS ELASTIC", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SHEAR RETENTION", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "TENSION CUTOFF", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "TENSION STIFFENING", kind: Kind::Behavior, handler: behavior },
];

/// *CLAY HARDENING får förekomma två gånger (TYPE=COMPRESSION och TENSION).
fn repeated(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    let ty = |k: &RawKeyword| k.param_upper("TYPE").unwrap_or_else(|| "COMPRESSION".into());
    if m.behaviors.iter().any(|b| b.name == kw.name && ty(b) == ty(kw)) {
        return kw.err(format!("materialet {} har redan *{}, TYPE={}", m.name, kw.name, ty(kw)));
    }
    m.behaviors.push(kw.clone());
    Ok(())
}

fn floats(kw: &RawKeyword, d: &DataLine, from: usize) -> Result<Vec<f64>> {
    d.fields[from..]
        .iter()
        .map(|f| {
            if f.trim().is_empty() {
                Ok(0.0)
            } else {
                crate::fem::model::parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{f}' är inget tal", kw.file, d.line)))
            }
        })
        .collect()
}

/// *INITIAL CONDITIONS, TYPE=STRESS (även GEOSTATIC) och TYPE=RATIO.
pub fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    for p in ["FILE", "USER", "INPUT", "REBAR", "SECTION POINTS", "INTERPOLATE", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu för TYPE={ty}"));
        }
    }
    let mut out = kw.clone();
    out.params.push((RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let vals = floats(kw, &d, 1)?;
        let targets = if ty == "RATIO" { ctx.node_ref(kw, d.line, tok, None)? } else { ctx.elem_ref(kw, d.line, tok, None)? };
        if ty == "RATIO" && vals.is_empty() {
            return kw.err_at(d.line, "portalet saknas");
        }
        for t in targets {
            let mut fields = vec![t.to_string()];
            fields.extend(vals.iter().map(|v| format!("{v:e}")));
            out.data.push(DataLine { fields, line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
