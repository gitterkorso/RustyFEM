//! XFEM: *ENRICHMENT, *ENRICHMENT ACTIVATION, *DAMAGE STABILIZATION och
//! *INITIAL CONDITIONS, TYPE=ENRICHMENT (anropas från `temperature.rs`).
//! Elementen byts mot XFEM-element när filen är läst (`elements/xfem.rs`).

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
        return kw.err("*ENRICHMENT inne i *PART stöds inte ännu (skriv den i *ASSEMBLY eller utan parter)");
    }
    store_model(ctx, kw)
}

/// *INITIAL CONDITIONS, TYPE=ENRICHMENT: element, lokal nod, berikning, φ, ψ.
/// Sparas som "XFEM INITIAL" med elementindex i modellen.
pub fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if ctx.part.is_some() {
        return kw.err("TYPE=ENRICHMENT inne i *PART stöds inte ännu");
    }
    let mut data = Vec::new();
    for d in kw.records() {
        if d.fields.len() < 5 {
            return kw.err_at(d.line, "raden ska ha element, lokal nod, berikningens namn, φ och ψ");
        }
        let elems = ctx.elem_ref(kw, d.line, &d.fields[0], None)?;
        let local = d.fields[1].trim().trim_end_matches('.').parse::<usize>().ok().filter(|&n| n >= 1);
        let Some(local) = local else { return kw.err_at(d.line, format!("ogiltigt lokalt nodnummer '{}'", d.fields[1])) };
        let name = d.fields[2].trim().to_uppercase();
        let name = name.rsplit('.').next().unwrap_or(&name).to_string();
        let (Some(phi), Some(psi)) = (parse_f64(&d.fields[3]), parse_f64(&d.fields[4])) else {
            return kw.err_at(d.line, "φ och ψ ska vara tal");
        };
        for e in elems {
            data.push(DataLine { fields: vec![e.to_string(), local.to_string(), name.clone(), phi.to_string(), psi.to_string()], line: d.line, continued: false });
        }
    }
    ctx.model.extra.push(RawKeyword { name: "XFEM INITIAL".into(), params: Vec::new(), data, file: kw.file.clone(), line: kw.line });
    Ok(())
}
