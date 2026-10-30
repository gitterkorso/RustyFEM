//! Material behaviors for Area materials5: crushable foam, porous metals
//! (Gurson–Tvergaard–Needleman), deformation plasticity (Ramberg–Osgood) and
//! *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY.
//!
//! The behaviors are parsed by `materials/crushable_foam.rs`, `porous_metal.rs` and
//! `deformation_plasticity.rs`. Composite damage (CRITERION=HASHIN) and
//! the forming limit criteria (FLD, FLSD, MSFLD) use the existing
//! *DAMAGE INITIATION/*DAMAGE EVOLUTION/*DAMAGE STABILIZATION
//! (`materials/hashin.rs`, `materials/forming_limit.rs`).

use crate::fem::model::{DataLine, RawKeyword};
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::{FemError, Result};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CRUSHABLE FOAM", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CRUSHABLE FOAM HARDENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DEFORMATION PLASTICITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "POROUS FAILURE CRITERIA", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "POROUS METAL PLASTICITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "VOID NUCLEATION", kind: Kind::Behavior, handler: behavior },
];

/// Parameter that marks a resolved *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY.
pub const RESOLVED_RD: &str = "RUSTYFEM_RD";

/// *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY: lines "node or node set,
/// relative density" (nodal values). Stored in `FeModel::extra` with
/// the lines [node index, r] and the parameter `RESOLVED_RD`; read by
/// `materials::porous_metal::apply_relative_density`, which takes the element's
/// nodal mean value at all integration points.
pub fn initial_relative_density(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let mut out = kw.clone();
    out.params.push((RESOLVED_RD.to_string(), String::new()));
    out.data.clear();
    for d in kw.records() {
        let Some(tok) = d.fields.first() else { continue };
        if tok.trim().is_empty() {
            continue;
        }
        let r = d
            .fields
            .get(1)
            .and_then(|f| crate::fem::model::parse_f64(f))
            .ok_or_else(|| FemError(format!("{} line {}: relative density is missing", kw.file, d.line)))?;
        if !(r > 0.0 && r <= 1.0) {
            return kw.err_at(d.line, format!("relative density {r} must lie in (0, 1]"));
        }
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![n.to_string(), format!("{r:e}")], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
