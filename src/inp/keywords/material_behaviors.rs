//! Material behaviors for nonlinear and anisotropic materials plus *ORIENTATION.
//!
//! The behaviors are stored in the material and parsed by the material factories in
//! `materials/plasticity.rs`, `materials/hyperelastic.rs`,
//! `materials/thermal.rs` and `materials/plasticity_dp.rs`.
//! *ORIENTATION is stored in the model and assigned to elements via the section's
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
