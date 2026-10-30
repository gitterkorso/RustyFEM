//! Keywords for the modal steps: the procedures *MODAL DYNAMIC,
//! *STEADY STATE DYNAMICS and *RESPONSE SPECTRUM plus *BASE MOTION and
//! *SELECT EIGENMODES (stored in the step and read by `steps/modal_basis.rs`).
//!
//! The modal procedures are linear perturbation steps (like
//! *FREQUENCY): only the step's own loads apply and the state is restored
//! afterwards, so loads in a modal step do not carry over to the next one.

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
