//! Keywords for Area coupled2:
//!
//! - *INELASTIC HEAT FRACTION: material behavior (`materials/inelastic_heat.rs`).
//! - *COUPLED THERMAL-ELECTRICAL-STRUCTURAL: procedure
//!   (`steps/thermal_electric_structural.rs`).
//!
//! ADIABATIC on *STATIC and *DYNAMIC is read by `steps/adiabatic.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "COUPLED THERMAL-ELECTRICAL-STRUCTURAL", kind: Kind::Step, handler: procedure },
    Keyword { name: "INELASTIC HEAT FRACTION", kind: Kind::Behavior, handler: behavior },
];
