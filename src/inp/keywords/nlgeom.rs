//! Keywords for geometrically nonlinear analysis.
//!
//! *CONTROLS is stored in the step and read by the NLGEOM steps (`steps::nlgeom`):
//! PARAMETERS=FIELD (R_n, C_n) and PARAMETERS=TIME INCREMENTATION (I_0, I_R,
//! I_P, I_C, I_L, I_G, ..., I_A). Other parameters are ignored.

use crate::inp::{store_step, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "CONTROLS", kind: Kind::Step, handler: store_step }];
