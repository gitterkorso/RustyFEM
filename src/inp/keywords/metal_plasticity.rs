//! Underalternativ till *PLASTIC för metallplasticitet: Hills anisotropa
//! flytvillkor (*POTENTIAL) och hastighetsberoende flytspänning
//! (*RATE DEPENDENT). Beteendena sparas i materialet och tolkas av
//! `materials/plasticity_hill.rs` och `materials/plasticity_rate.rs`.

use crate::inp::keywords::material::behavior;
use crate::inp::{Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "POTENTIAL", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "RATE DEPENDENT", kind: Kind::Behavior, handler: behavior },
];
