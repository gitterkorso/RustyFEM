//! Procedure keywords for eigenvalue steps: *FREQUENCY and *BUCKLE.
//!
//! Both are linear perturbation steps: the step starts from
//! the base state (the preceding general step), only the step's own loads apply
//! and the state is restored afterwards. The handler therefore sets the step's
//! `perturbation` even if *STEP lacks PERTURBATION.

use crate::fem::model::RawKeyword;
use crate::inp::{procedure, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "BUCKLE", kind: Kind::Step, handler: eigen_procedure },
    Keyword { name: "FREQUENCY", kind: Kind::Step, handler: eigen_procedure },
];

fn eigen_procedure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    procedure(ctx, kw)?;
    ctx.step_mut(kw)?.perturbation = true;
    Ok(())
}
