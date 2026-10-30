//! Nyckelord för dynamik: procedurerna *DYNAMIC (implicit och EXPLICIT),
//! *MODAL DYNAMIC, *STEADY STATE DYNAMICS och *RESPONSE SPECTRUM, samt
//! *INITIAL CONDITIONS (TYPE=VELOCITY), materialbeteendet *DAMPING
//! (Rayleigh), *MODAL DAMPING, *BULK VISCOSITY och *SPECTRUM.
//!
//! *MODAL DYNAMIC, *STEADY STATE DYNAMICS och *RESPONSE SPECTRUM registreras i
//! `modal.rs` (linjära perturbationssteg). *FREQUENCY registreras i `eigen.rs`. *INITIAL CONDITIONS registreras i
//! `temperature.rs`, som skickar TYPE=VELOCITY vidare till
//! `initial_conditions` här.

use crate::fem::model::RawKeyword;
use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, store_model, store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "BULK VISCOSITY", kind: Kind::Step, handler: store_step },
    Keyword { name: "DAMPING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DYNAMIC", kind: Kind::Step, handler: procedure },
    Keyword { name: "MODAL DAMPING", kind: Kind::Step, handler: store_step },
    Keyword { name: "SPECTRUM", kind: Kind::Model, handler: store_model },
];

/// *INITIAL CONDITIONS: bara TYPE=VELOCITY stöds (läses av dynamikstegen).
/// Raderna översätts till nodindex här, eftersom set kan vara definierade i parter.
pub(crate) fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    if ty != "VELOCITY" {
        return kw.err(format!("TYPE={ty} stöds inte ännu (bara TYPE=VELOCITY)"));
    }
    let mut out = kw.clone();
    out.data.clear();
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "skriv 'nod/nset, frihetsgrad, hastighet'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        for n in nodes {
            let mut dl = d.clone();
            // första fältet ersätts med internt nodindex (prefix '#')
            dl.fields[0] = format!("#{n}");
            out.data.push(dl);
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
