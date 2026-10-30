//! Keywords for dynamics: the procedures *DYNAMIC (implicit and EXPLICIT),
//! *MODAL DYNAMIC, *STEADY STATE DYNAMICS and *RESPONSE SPECTRUM, plus
//! *INITIAL CONDITIONS (TYPE=VELOCITY), the material behavior *DAMPING
//! (Rayleigh), *MODAL DAMPING, *BULK VISCOSITY and *SPECTRUM.
//!
//! *MODAL DYNAMIC, *STEADY STATE DYNAMICS and *RESPONSE SPECTRUM are registered in
//! `modal.rs` (linear perturbation steps). *FREQUENCY is registered in `eigen.rs`. *INITIAL CONDITIONS is registered in
//! `temperature.rs`, which forwards TYPE=VELOCITY to
//! `initial_conditions` here.

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

/// *INITIAL CONDITIONS: only TYPE=VELOCITY is supported (read by the dynamic steps).
/// The lines are translated to node indices here, since sets may be defined in parts.
pub(crate) fn initial_conditions(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let ty = kw.param_upper("TYPE").unwrap_or_default();
    if ty != "VELOCITY" {
        return kw.err(format!("TYPE={ty} is not supported yet (only TYPE=VELOCITY)"));
    }
    let mut out = kw.clone();
    out.data.clear();
    for d in &kw.data {
        if d.fields.len() < 3 {
            return kw.err_at(d.line, "write 'node/nset, degree of freedom, velocity'");
        }
        let nodes = ctx.node_ref(kw, d.line, &d.fields[0], None)?;
        for n in nodes {
            let mut dl = d.clone();
            // the first field is replaced with the internal node index (prefix '#')
            dl.fields[0] = format!("#{n}");
            out.data.push(dl);
        }
    }
    ctx.model.extra.push(out);
    Ok(())
}
