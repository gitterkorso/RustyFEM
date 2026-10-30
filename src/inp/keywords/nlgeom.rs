//! Nyckelord för geometriskt olinjär analys.
//!
//! *CONTROLS sparas i steget och läses av NLGEOM-stegen (`steps::nlgeom`):
//! PARAMETERS=FIELD (R_n, C_n) och PARAMETERS=TIME INCREMENTATION (I_0, I_R,
//! I_P, I_C, I_L, I_G, ..., I_A). Andra parametrar ignoreras.

use crate::inp::{store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "CONTROLS", kind: Kind::Step, handler: store_step }];
