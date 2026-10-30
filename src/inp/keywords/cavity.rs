//! Area cavity: liquid- and gas-filled cavities (surface-based cavities).
//!
//! - *FLUID CAVITY, NAME=, BEHAVIOR=, REF NODE=, SURFACE= (ADDED VOLUME=,
//!   AMBIENT PRESSURE=; the data line is the thickness for planar models).
//! - *FLUID BEHAVIOR, NAME= followed by *FLUID DENSITY, *FLUID BULK MODULUS,
//!   *FLUID EXPANSION (hydraulic fluid) or *MOLECULAR WEIGHT and
//!   *CAPACITY (ideal gas). The suboptions are stored in order in
//!   `FeModel::extra` and attached to the nearest preceding *FLUID BEHAVIOR by
//!   `interactions/cavity.rs`.
//! - *FLUID FLUX (in steps): mass flow into the cavity.
//! - *INITIAL CONDITIONS, TYPE=FLUID PRESSURE (called from `temperature.rs`).
//!
//! Everything is computed in `interactions/cavity.rs`.

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

/// The suboptions of *FLUID BEHAVIOR (in the order they may appear).
pub const BEHAVIOR_OPTIONS: &[&str] = &["FLUID DENSITY", "FLUID BULK MODULUS", "FLUID EXPANSION", "MOLECULAR WEIGHT", "CAPACITY"];

fn fluid_behavior(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    if kw.has("USER") {
        return kw.err("USER (UFLUID) is not supported yet");
    }
    if kw.param("NAME").is_none() {
        return kw.err("*FLUID BEHAVIOR requires NAME=");
    }
    store_model(ctx, kw)
}

/// Keywords that affect the result but are not supported yet.
fn unsupported(_ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    kw.err(format!("*{} is not supported yet (see docs/REFERENCE.md, Limitations)", kw.name))
}

/// *INITIAL CONDITIONS, TYPE=FLUID PRESSURE: stored and read by `interactions/cavity.rs`.
pub fn initial_fluid_pressure(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    store_model(ctx, kw)
}
