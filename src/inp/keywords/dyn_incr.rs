//! Keywords for automatic time incrementation in implicit dynamics:
//! *TIME POINTS (output time points for *OUTPUT, TIME POINTS=name), read by
//! `steps/dynamic_incr.rs`.

use crate::inp::{store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "TIME POINTS", kind: Kind::Both, handler: store_model }];
