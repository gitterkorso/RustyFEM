//! Materialbeteenden för område materials5: krossbart skum, porösa metaller
//! (Gurson–Tvergaard–Needleman), deformationsplasticitet (Ramberg–Osgood) och
//! *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY.
//!
//! Beteendena tolkas av `materials/crushable_foam.rs`, `porous_metal.rs` och
//! `deformation_plasticity.rs`. Kompositskada (CRITERION=HASHIN) och
//! formbarhetskriterierna (FLD, FLSD, MSFLD) använder de befintliga
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

/// Parameter som markerar ett upplöst *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY.
pub const RESOLVED_RD: &str = "RUSTYFEM_RD";

/// *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY: rader "nod eller nodset,
/// relativ densitet" (nodvärden). Sparas i `FeModel::extra` med
/// raderna [nodindex, r] och parametern `RESOLVED_RD`; läses av
/// `materials::porous_metal::apply_relative_density`, som tar elementets
/// nodmedelvärde i alla integrationspunkter.
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
            .ok_or_else(|| FemError(format!("{} rad {}: relativ densitet saknas", kw.file, d.line)))?;
        if !(r > 0.0 && r <= 1.0) {
            return kw.err_at(d.line, format!("relativ densitet {r} ska ligga i (0, 1]"));
        }
        for n in ctx.node_ref(kw, d.line, tok, None)? {
            out.data.push(DataLine { fields: vec![n.to_string(), format!("{r:e}")], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
