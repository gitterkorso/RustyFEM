//! Nyckelord för den automatiska tidsinkrementeringen i implicit dynamik:
//! *TIME POINTS (utdatatidpunkter för *OUTPUT, TIME POINTS=namn), läses av
//! `steps/dynamic_incr.rs`.

use crate::inp::{store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[Keyword { name: "TIME POINTS", kind: Kind::Both, handler: store_model }];
