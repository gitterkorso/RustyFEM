//! Area materials4: keywords that affect the result but are not supported yet
//! and therefore give an error instead of being ignored with a warning.
//!
//! - *CONCRETE FAILURE (element deletion for concrete with damaged plasticity in
//!   explicit analysis),
//! - *PRESSURE PENETRATION (pressure penetrating into a contact gap).

use crate::fem::model::RawKeyword;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONCRETE FAILURE", kind: Kind::Behavior, handler: unsupported },
    Keyword { name: "PRESSURE PENETRATION", kind: Kind::Step, handler: unsupported },
];

fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} is not supported yet (see docs/REFERENCE.md, Limitations)", kw.name))
}
