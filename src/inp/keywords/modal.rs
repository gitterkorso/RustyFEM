//! Nyckelord för de modala stegen: procedurerna *MODAL DYNAMIC,
//! *STEADY STATE DYNAMICS och *RESPONSE SPECTRUM samt *BASE MOTION och
//! *SELECT EIGENMODES (sparas i steget och läses av `steps/modal_basis.rs`).
//!
//! De modala procedurerna är linjära perturbationssteg (som
//! *FREQUENCY): bara stegets egna laster gäller och tillståndet återställs
//! efteråt, så laster i ett modalt steg följer inte med till nästa.

use crate::fem::model::RawKeyword;
use crate::inp::{procedure, store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "BASE MOTION", kind: Kind::Step, handler: store_step },
    Keyword { name: "MODAL DYNAMIC", kind: Kind::Step, handler: modal_procedure },
    Keyword { name: "RESPONSE SPECTRUM", kind: Kind::Step, handler: modal_procedure },
    Keyword { name: "SELECT EIGENMODES", kind: Kind::Step, handler: store_step },
    Keyword { name: "STEADY STATE DYNAMICS", kind: Kind::Step, handler: modal_procedure },
];

fn modal_procedure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    procedure(ctx, kw)?;
    ctx.step_mut(kw)?.perturbation = true;
    Ok(())
}
