//! Nyckelord för stegprocedurerna i `steps/complex_frequency.rs`,
//! `steps/random_response.rs`, `steps/subspace.rs`, `steps/direct_cyclic.rs`
//! och `steps/explicit_coupled.rs`:
//!
//! - *COMPLEX FREQUENCY och *RANDOM RESPONSE: linjära perturbationssteg
//!   (som *FREQUENCY), tillståndet återställs efteråt.
//! - *PSD-DEFINITION (modellen) och *CORRELATION (steget): sparas och läses
//!   av *RANDOM RESPONSE.
//! - *DIRECT CYCLIC, *ANNEAL och *DYNAMIC TEMPERATURE-DISPLACEMENT: allmänna steg.
//!
//! *DYNAMIC, SUBSPACE och *STEADY STATE DYNAMICS, SUBSPACE PROJECTION
//! använder de befintliga procedurnyckelorden.

use crate::fem::model::RawKeyword;
use crate::inp::{procedure, store_model, store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "ANNEAL", kind: Kind::Step, handler: procedure },
    Keyword { name: "COMPLEX FREQUENCY", kind: Kind::Step, handler: perturbation_procedure },
    Keyword { name: "CORRELATION", kind: Kind::Step, handler: store_step },
    Keyword { name: "DIRECT CYCLIC", kind: Kind::Step, handler: procedure },
    Keyword { name: "DYNAMIC TEMPERATURE-DISPLACEMENT", kind: Kind::Step, handler: procedure },
    Keyword { name: "PSD-DEFINITION", kind: Kind::Model, handler: store_model },
    Keyword { name: "RANDOM RESPONSE", kind: Kind::Step, handler: perturbation_procedure },
];

fn perturbation_procedure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    procedure(ctx, kw)?;
    ctx.step_mut(kw)?.perturbation = true;
    Ok(())
}
