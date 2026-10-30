//! Time-dependent materials: *CREEP and *VISCOELASTIC (material behaviors) and
//! the procedure keyword *VISCO.
//!
//! The behaviors are parsed by `materials/creep.rs` and `materials/viscoelastic.rs`,
//! the step is run by `steps/visco.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CREEP", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "VISCO", kind: Kind::Step, handler: procedure },
    Keyword { name: "VISCOELASTIC", kind: Kind::Behavior, handler: behavior },
];
