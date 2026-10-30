//! Keywords for constraints, springs, masses, connector elements and contact.
//! All are stored in `FeModel::extra` and built into interactions by
//! `src/interactions/` (subordinate keywords such as *KINEMATIC after *COUPLING
//! and *FRICTION after *SURFACE INTERACTION are read in order from there).

use crate::inp::{ignore, store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CLEARANCE", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR BEHAVIOR", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR ELASTICITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR SECTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONTACT", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT CONTROLS", kind: Kind::Both, handler: ignore },
    Keyword { name: "CONTACT DAMPING", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONTACT EXCLUSIONS", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT FORMULATION", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT INCLUSIONS", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT PAIR", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT PROPERTY ASSIGNMENT", kind: Kind::Both, handler: store_model },
    Keyword { name: "COUPLING", kind: Kind::Model, handler: store_model },
    Keyword { name: "DASHPOT", kind: Kind::Model, handler: store_model },
    Keyword { name: "DISTRIBUTING", kind: Kind::Model, handler: store_model },
    Keyword { name: "EQUATION", kind: Kind::Model, handler: store_model },
    Keyword { name: "FRICTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "KINEMATIC", kind: Kind::Model, handler: store_model },
    Keyword { name: "KINEMATIC COUPLING", kind: Kind::Model, handler: store_model },
    Keyword { name: "MASS", kind: Kind::Model, handler: store_model },
    Keyword { name: "MPC", kind: Kind::Model, handler: store_model },
    // *ORIENTATION is registered in material_behaviors.rs (stored in the model with
    // store_model); the connector elements' local axes are read from there.
    Keyword { name: "ROTARY INERTIA", kind: Kind::Model, handler: store_model },
    Keyword { name: "SPRING", kind: Kind::Model, handler: store_model },
    Keyword { name: "SURFACE BEHAVIOR", kind: Kind::Model, handler: store_model },
    Keyword { name: "SURFACE INTERACTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "TIE", kind: Kind::Model, handler: store_model },
];
