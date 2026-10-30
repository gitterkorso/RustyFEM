//! Keywords for connector element behaviors and step data
//! (`src/interactions/connector.rs`). *CONNECTOR SECTION, *CONNECTOR BEHAVIOR
//! and *CONNECTOR ELASTICITY are registered in `interactions.rs`. The behaviors
//! are stored in the model and read in order after *CONNECTOR BEHAVIOR;
//! *CONNECTOR MOTION and *CONNECTOR LOAD are stored in the step.

use crate::inp::{store_model, store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONNECTOR CONSTITUTIVE REFERENCE", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR DAMAGE EVOLUTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR DAMAGE INITIATION", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR DAMPING", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR DERIVED COMPONENT", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR FRICTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR HARDENING", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR LOAD", kind: Kind::Step, handler: store_step },
    Keyword { name: "CONNECTOR LOCK", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR MOTION", kind: Kind::Step, handler: store_step },
    Keyword { name: "CONNECTOR PLASTICITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR POTENTIAL", kind: Kind::Model, handler: store_model },
    Keyword { name: "CONNECTOR STOP", kind: Kind::Model, handler: store_model },
];
