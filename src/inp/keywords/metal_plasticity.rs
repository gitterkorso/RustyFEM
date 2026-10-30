//! Suboptions of *PLASTIC for metal plasticity: Hill's anisotropic
//! yield criterion (*POTENTIAL) and rate-dependent yield stress
//! (*RATE DEPENDENT). The behaviors are stored in the material and parsed by
//! `materials/plasticity_hill.rs` and `materials/plasticity_rate.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "POTENTIAL", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "RATE DEPENDENT", kind: Kind::Behavior, handler: behavior },
];
