//! Nyckelord för område physics3: kavitetsstrålning, påtvingad konvektion,
//! brottmekanik i kontakt (VCCT) och lågfrekvent elektromagnetism.
//!
//! - *SURFACE PROPERTY, *EMISSIVITY, *CAVITY DEFINITION: modelldata, sparas i
//!   `FeModel::extra` och tolkas av `steps/heat_cavity.rs`.
//! - *RADIATION VIEW FACTOR, *RADIATION SYMMETRY, *REFLECTION, *PERIODIC,
//!   *CYCLIC, *MOTION: stegdata (stegets `extra`), samma tolkare. *PERIODIC,
//!   *CYCLIC och *MOTION ger fel där tills de stöds.
//! - *MASS FLOW RATE: stegdata för DCC-elementen (`steps/heat_convection.rs`).
//! - *ELECTROMAGNETIC (procedur, `steps/electromagnetic.rs`) och *MAGNETIC
//!   PERMEABILITY (materialbeteende, `materials/electromagnetic.rs`).
//! - *RADIATION PRINT, *RADIATION OUTPUT, *RADIATION FILE, *VIEW FACTOR
//!   OUTPUT: utdatabegäran som inte påverkar lösningen (ignoreras).

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
