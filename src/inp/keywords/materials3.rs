//! Materialbeteenden för krypning, svällning och viskoelasticitet utöver
//! `materials_time.rs`: *SWELLING (`materials/creep_ext.rs`), *TRS
//! (tidsförskjutning för viskoelasticitet, `materials/viscoelastic_trs.rs`)
//! och *CAP CREEP (`materials/cap.rs`).

use crate::inp::keywords::material::behavior;
use crate::inp::{Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CAP CREEP", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "SWELLING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "TRS", kind: Kind::Behavior, handler: behavior },
];
