//! Nyckelord för villkor, fjädrar, massor, kopplingselement och kontakt.
//! Alla sparas i `FeModel::extra` och byggs till interaktioner av
//! `src/interactions/` (underordnade nyckelord som *KINEMATIC efter *COUPLING
//! och *FRICTION efter *SURFACE INTERACTION läses i ordning därifrån).

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
    // *ORIENTATION registreras i material_behaviors.rs (sparas i modellen med
    // store_model); kopplingselementens lokala axlar läses därifrån.
    Keyword { name: "ROTARY INERTIA", kind: Kind::Model, handler: store_model },
    Keyword { name: "SPRING", kind: Kind::Model, handler: store_model },
    Keyword { name: "SURFACE BEHAVIOR", kind: Kind::Model, handler: store_model },
    Keyword { name: "SURFACE INTERACTION", kind: Kind::Model, handler: store_model },
    Keyword { name: "TIE", kind: Kind::Model, handler: store_model },
];
