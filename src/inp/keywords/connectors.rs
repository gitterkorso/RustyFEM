//! Nyckelord för kopplingselementens beteenden och stegdata
//! (`src/interactions/connector.rs`). *CONNECTOR SECTION, *CONNECTOR BEHAVIOR
//! och *CONNECTOR ELASTICITY registreras i `interactions.rs`. Beteendena
//! sparas i modellen och läses i ordning efter *CONNECTOR BEHAVIOR;
//! *CONNECTOR MOTION och *CONNECTOR LOAD sparas i steget.

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
