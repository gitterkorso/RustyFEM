//! Material behaviors for creep, swelling and viscoelasticity beyond
//! `materials_time.rs`: *SWELLING (`materials/creep_ext.rs`), *TRS
//! (time shift for viscoelasticity, `materials/viscoelastic_trs.rs`)
//! and *CAP CREEP (`materials/cap.rs`).

use crate::inp::keywords::material::behavior;
use crate::inp::{Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CAP CREEP", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SWELLING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "TRS", kind: Kind::Behavior, handler: behavior },
];
