//! Område materials4: nyckelord som påverkar resultatet men inte stöds ännu
//! och därför ger fel i stället för att ignoreras med en varning.
//!
//! - *CONCRETE FAILURE (elementborttagning för betong med skadeplasticitet i
//!   explicit analys),
//! - *PRESSURE PENETRATION (tryck som tränger in i en kontaktspalt).

use crate::fem::model::RawKeyword;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONCRETE FAILURE", kind: Kind::Behavior, handler: unsupported },
    Keyword { name: "PRESSURE PENETRATION", kind: Kind::Step, handler: unsupported },
];

fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} stöds inte ännu (se README, Begränsningar)", kw.name))
}
