//! Nyckelord för explicit dynamik: massskalning (*FIXED MASS SCALING,
//! *VARIABLE MASS SCALING, stegdata) och *SECTION CONTROLS (hourglass-kontroll,
//! läses av `steps/explicit.rs` via tvärsnittets CONTROLS=).

use crate::inp::{store_model, store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "FIXED MASS SCALING", kind: Kind::Step, handler: store_step },
    Keyword { name: "SECTION CONTROLS", kind: Kind::Model, handler: store_model },
    Keyword { name: "VARIABLE MASS SCALING", kind: Kind::Step, handler: store_step },
];
