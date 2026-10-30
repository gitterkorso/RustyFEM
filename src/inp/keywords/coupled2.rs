//! Nyckelord för område coupled2:
//!
//! - *INELASTIC HEAT FRACTION: materialbeteende (`materials/inelastic_heat.rs`).
//! - *COUPLED THERMAL-ELECTRICAL-STRUCTURAL: procedur
//!   (`steps/thermal_electric_structural.rs`).
//!
//! ADIABATIC på *STATIC och *DYNAMIC läses av `steps/adiabatic.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{procedure, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "COUPLED THERMAL-ELECTRICAL-STRUCTURAL", kind: Kind::Step, handler: procedure },
    Keyword { name: "INELASTIC HEAT FRACTION", kind: Kind::Behavior, handler: behavior },
];
