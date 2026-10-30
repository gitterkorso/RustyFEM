//! Keywords for explicit dynamics: mass scaling (*FIXED MASS SCALING,
//! *VARIABLE MASS SCALING, step data) and *SECTION CONTROLS (hourglass control,
//! read by `steps/explicit.rs` via the section's CONTROLS=).

use crate::inp::{store_model, store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "FIXED MASS SCALING", kind: Kind::Step, handler: store_step },
    Keyword { name: "SECTION CONTROLS", kind: Kind::Model, handler: store_model },
    Keyword { name: "VARIABLE MASS SCALING", kind: Kind::Step, handler: store_step },
];
