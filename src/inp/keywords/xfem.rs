//! XFEM: *ENRICHMENT, *ENRICHMENT ACTIVATION, *DAMAGE STABILIZATION and
//! *INITIAL CONDITIONS, TYPE=ENRICHMENT (called from `temperature.rs`).
//! The elements are replaced by XFEM elements once the file has been read (`elements/xfem.rs`).

use crate::fem::model::{parse_f64, DataLine, RawKeyword};
use crate::inp::{store_model, store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "DAMAGE STABILIZATION", kind: Kind::Behavior, handler: crate::inp::keywords::material::behavior },
    Keyword { name: "ENRICHMENT", kind: Kind::Model, handler: enrichment },
    Keyword { name: "ENRICHMENT ACTIVATION", kind: Kind::Step, handler: store_step },
];

fn enrichment(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() {
        return kw.err("*ENRICHMENT inside *PART is not supported yet (write it in *ASSEMBLY or without parts)");
    }
    store_model(ctx, kw)
}

/// *INITIAL CONDITIONS, TYPE=ENRICHMENT: element, local node, enrichment, φ, ψ.
/// Stored as "XFEM INITIAL" with element indices in the model.
pub fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() {
        return kw.err("TYPE=ENRICHMENT inside *PART is not supported yet");
    }
    let mut data = Vec::new();
    for d in kw.records() {
        if d.fields.len() < 5 {
            return kw.err_at(d.line, "the line must have element, local node, enrichment name, φ and ψ");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let local = d.fields[1].trim().trim_end_matches('.').parse::<usize>().ok().filter(|&n| n >= 1);
        let Some(local) = local else { return kw.err_at(d.line, format!("invalid local node number '{}'", d.fields[1])) };
        let name = d.fields[2].trim().to_uppercase();
        let name = name.rsplit('.').next().unwrap_or(&name).to_string();
        let (Some(phi), Some(psi)) = (parse_f64(&d.fields[3]), parse_f64(&d.fields[4])) else {
            return kw.err_at(d.line, "φ and ψ must be numbers");
        };
        for e in elems {
            data.push(DataLine { fields: vec![e.to_string(), local.to_string(), name.clone(), phi.to_string(), psi.to_string()], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(RawKeyword { name: "XFEM INITIAL".into(), params: Vec::new(), data, file: kw.file.clone(), line: kw.line });
    Ok(())
}
