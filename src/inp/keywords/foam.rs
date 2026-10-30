//! Elastomers and foams: *HYPERFOAM, *MULLINS EFFECT and test data for
//! *HYPERELASTIC, TEST DATA INPUT (material behaviors).
//!
//! The behaviors are parsed by `materials/hyperfoam.rs`, `materials/mullins.rs` and
//! `materials/hyperelastic_testdata.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "BIAXIAL TEST DATA", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "HYPERFOAM", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "MULLINS EFFECT", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "PLANAR TEST DATA", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "UNIAXIAL TEST DATA", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "VOLUMETRIC TEST DATA", kind: Kind::Behavior, handler: behavior },
];
