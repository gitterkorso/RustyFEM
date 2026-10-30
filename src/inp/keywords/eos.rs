//! Material behaviors and initial conditions for explicit dynamics: *EOS and
//! *TENSILE FAILURE (parsed by `materials/eos.rs`) plus *INITIAL CONDITIONS,
//! TYPE=SPECIFIC ENERGY (called from `temperature.rs`).

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "EOS", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "TENSILE FAILURE", kind: Kind::Behavior, handler: behavior },
];

/// *INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY: lines (element or element set,
/// Eₘ₀) are stored resolved in `FeModel::extra` (element index, Eₘ₀).
pub fn initial_specific_energy(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    for p in ["FILE", "USER", "INPUT", "STEP", "INC"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported for TYPE=SPECIFIC ENERGY"));
        }
    }
    let mut out = kw.clone();
    out.params.push((crate::materials::eos::RESOLVED.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let Some(v) = d.fields.get(1).and_then(|f| crate::fem::model::parse_f64(f)) else {
            return kw.err_at(d.line, "give an element (or element set) and the specific energy");
        };
        for e in ctx.elem_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![e.to_string(), format!("{v:e}")], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
