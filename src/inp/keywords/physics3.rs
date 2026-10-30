//! Keywords for Area physics3: cavity radiation, forced convection,
//! fracture mechanics in contact (VCCT) and low-frequency electromagnetics.
//!
//! - *SURFACE PROPERTY, *EMISSIVITY, *CAVITY DEFINITION: model data, stored in
//!   `FeModel::extra` and parsed by `steps/heat_cavity.rs`.
//! - *RADIATION VIEW FACTOR, *RADIATION SYMMETRY, *REFLECTION, *PERIODIC,
//!   *CYCLIC, *MOTION: step data (the step's `extra`), same parser. *PERIODIC,
//!   *CYCLIC and *MOTION give an error there until they are supported.
//! - *MASS FLOW RATE: step data for the DCC elements (`steps/heat_convection.rs`).
//! - *ELECTROMAGNETIC (procedure, `steps/electromagnetic.rs`) and *MAGNETIC
//!   PERMEABILITY (material behavior, `materials/electromagnetic.rs`).
//! - *RADIATION PRINT, *RADIATION OUTPUT, *RADIATION FILE, *VIEW FACTOR
//!   OUTPUT: output requests that do not affect the solution (ignored).

use crate::inp::keywords::material::behavior;
use crate::inp::{ignore, procedure, store_model, store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CAVITY DEFINITION", kind: Kind::Model, handler: store_model },
    Keyword { name: "CYCLIC", kind: Kind::Step, handler: store_step },
    Keyword { name: "ELECTROMAGNETIC", kind: Kind::Step, handler: procedure },
    Keyword { name: "EMISSIVITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "MAGNETIC PERMEABILITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "MASS FLOW RATE", kind: Kind::Step, handler: store_step },
    Keyword { name: "MOTION", kind: Kind::Step, handler: store_step },
    Keyword { name: "PERIODIC", kind: Kind::Step, handler: store_step },
    Keyword { name: "RADIATION FILE", kind: Kind::Step, handler: ignore },
    Keyword { name: "RADIATION OUTPUT", kind: Kind::Step, handler: ignore },
    Keyword { name: "RADIATION PRINT", kind: Kind::Step, handler: ignore },
    Keyword { name: "RADIATION SYMMETRY", kind: Kind::Step, handler: store_step },
    Keyword { name: "RADIATION VIEW FACTOR", kind: Kind::Step, handler: store_step },
    Keyword { name: "REFLECTION", kind: Kind::Step, handler: store_step },
    Keyword { name: "SURFACE PROPERTY", kind: Kind::Model, handler: store_model },
    Keyword { name: "VIEW FACTOR OUTPUT", kind: Kind::Step, handler: ignore },
];
