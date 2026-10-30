//! Elastomerer och skum: *HYPERFOAM, *MULLINS EFFECT och provdata för
//! *HYPERELASTIC, TEST DATA INPUT (materialbeteenden).
//!
//! Beteendena tolkas av `materials/hyperfoam.rs`, `materials/mullins.rs` och
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
