//! Materialbeteenden för olinjära och anisotropa material samt *ORIENTATION.
//!
//! Beteendena sparas i materialet och tolkas av materialfabrikerna i
//! `materials/plasticity.rs`, `materials/hyperelastic.rs`,
//! `materials/thermal.rs` och `materials/plasticity_dp.rs`.
//! *ORIENTATION sparas i modellen och kopplas till element via tvärsnittets
//! ORIENTATION= (`materials/anisotropic.rs`).

use crate::inp::keywords::material::behavior;
use crate::inp::{store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CYCLIC HARDENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DRUCKER PRAGER", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DRUCKER PRAGER HARDENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "EXPANSION", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "HYPERELASTIC", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "ORIENTATION", kind: Kind::Model, handler: store_model },
    Keyword { name: "PLASTIC", kind: Kind::Behavior, handler: behavior },
];
