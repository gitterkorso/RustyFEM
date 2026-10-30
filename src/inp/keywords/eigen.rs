//! Procedurnyckelorden för egenvärdessteg: *FREQUENCY och *BUCKLE.
//!
//! Båda är linjära perturbationssteg: steget utgår från
//! bastillståndet (föregående allmänna steg), bara stegets egna laster gäller
//! och tillståndet återställs efteråt. Hanteraren sätter därför stegets
//! `perturbation` även om *STEP saknar PERTURBATION.

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
