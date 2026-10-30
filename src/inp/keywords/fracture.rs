//! Fracture mechanics: *CONTOUR INTEGRAL (stored in the step, computed by `steps/fracture.rs`).

use crate::inp::{store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "CONTOUR INTEGRAL", kind: Kind::Step, handler: store_step }];
