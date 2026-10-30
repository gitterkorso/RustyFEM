//! Tidsberoende material: *CREEP och *VISCOELASTIC (materialbeteenden) och
//! procedurnyckelordet *VISCO.
//!
//! Beteendena tolkas av `materials/creep.rs` och `materials/viscoelastic.rs`,
//! steget körs av `steps/visco.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CREEP", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "VISCO", kind: Kind::Step, handler: procedure },
    Keyword { name: "VISCOELASTIC", kind: Kind::Behavior, handler: behavior },
];
