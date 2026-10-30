//! Område cavity: vätske- och gasfyllda håligheter (ytbaserade kaviteter).
//!
//! - *FLUID CAVITY, NAME=, BEHAVIOR=, REF NODE=, SURFACE= (ADDED VOLUME=,
//!   AMBIENT PRESSURE=; dataraden är tjockleken för plana modeller).
//! - *FLUID BEHAVIOR, NAME= följt av *FLUID DENSITY, *FLUID BULK MODULUS,
//!   *FLUID EXPANSION (hydraulisk vätska) eller *MOLECULAR WEIGHT och
//!   *CAPACITY (ideal gas). Underalternativen sparas i ordning i
//!   `FeModel::extra` och knyts till närmast föregående *FLUID BEHAVIOR av
//!   `interactions/cavity.rs`.
//! - *FLUID FLUX (i steg): massflöde in i kaviteten.
//! - *INITIAL CONDITIONS, TYPE=FLUID PRESSURE (anropas från `temperature.rs`).
//!
//! Allt räknas i `interactions/cavity.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::{store_model, store_step, InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CAPACITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "FLUID BEHAVIOR", kind: Kind::Model, handler: fluid_behavior },
    Keyword { name: "FLUID BULK MODULUS", kind: Kind::Model, handler: store_model },
    Keyword { name: "FLUID CAVITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "FLUID DENSITY", kind: Kind::Model, handler: store_model },
    Keyword { name: "FLUID EXCHANGE", kind: Kind::Model, handler: unsupported },
    Keyword { name: "FLUID EXCHANGE ACTIVATION", kind: Kind::Step, handler: unsupported },
    Keyword { name: "FLUID EXCHANGE PROPERTY", kind: Kind::Model, handler: unsupported },
    Keyword { name: "FLUID EXPANSION", kind: Kind::Model, handler: store_model },
    Keyword { name: "FLUID FLUX", kind: Kind::Step, handler: store_step },
    Keyword { name: "FLUID INFLATOR", kind: Kind::Model, handler: unsupported },
    Keyword { name: "FLUID INFLATOR ACTIVATION", kind: Kind::Step, handler: unsupported },
    Keyword { name: "FLUID INFLATOR MIXTURE", kind: Kind::Model, handler: unsupported },
    Keyword { name: "FLUID INFLATOR PROPERTY", kind: Kind::Model, handler: unsupported },
    Keyword { name: "MOLECULAR WEIGHT", kind: Kind::Model, handler: store_model },
];

/// Underalternativen till *FLUID BEHAVIOR (i den ordning de kan stå).
pub const BEHAVIOR_OPTIONS: &[&str] = &["FLUID DENSITY", "FLUID BULK MODULUS", "FLUID EXPANSION", "MOLECULAR WEIGHT", "CAPACITY"];

fn fluid_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("USER") {
        return kw.err("USER (UFLUID) stöds inte ännu");
    }
    if kw.param("NAME").is_none() {
        return kw.err("*FLUID BEHAVIOR kräver NAME=");
    }
    store_model(ctx, kw)
}

/// Nyckelord som påverkar resultatet men inte stöds ännu.
fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} stöds inte ännu (se docs/REFERENS.md, Begränsningar)", kw.name))
}

/// *INITIAL CONDITIONS, TYPE=FLUID PRESSURE: sparas och läses av `interactions/cavity.rs`.
pub fn initial_fluid_pressure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    store_model(ctx, kw)
}
