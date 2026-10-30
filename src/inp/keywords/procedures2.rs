//! Keywords for the step procedures in `steps/complex_frequency.rs`,
//! `steps/random_response.rs`, `steps/subspace.rs`, `steps/direct_cyclic.rs`
//! and `steps/explicit_coupled.rs`:
//!
//! - *COMPLEX FREQUENCY and *RANDOM RESPONSE: linear perturbation steps
//!   (like *FREQUENCY), the state is restored afterwards.
//! - *PSD-DEFINITION (the model) and *CORRELATION (the step): stored and read
//!   by *RANDOM RESPONSE.
//! - *DIRECT CYCLIC, *ANNEAL and *DYNAMIC TEMPERATURE-DISPLACEMENT: general steps.
//!
//! *DYNAMIC, SUBSPACE and *STEADY STATE DYNAMICS, SUBSPACE PROJECTION
//! use the existing procedure keywords.

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
