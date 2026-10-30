//! Material behaviors for pressure-dependent plasticity and damage: Mohr–Coulomb,
//! concrete damaged plasticity (Lee–Fenves) and ductile damage.
//!
//! The behaviors are stored in the material and parsed by `materials/mohr_coulomb.rs`,
//! `materials/concrete_damaged.rs` and `materials/ductile_damage.rs`.

use crate::fem::model::RawKeyword;
use crate::inp::keywords::material::behavior;
use crate::inp::{InpCtx, Keyword, Kind};
use crate::model::Result;

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONCRETE COMPRESSION DAMAGE", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CONCRETE COMPRESSION HARDENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CONCRETE DAMAGED PLASTICITY", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CONCRETE TENSION DAMAGE", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "CONCRETE TENSION STIFFENING", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "DAMAGE EVOLUTION", kind: Kind::Behavior, handler: damage },
    Keyword { name: "DAMAGE INITIATION", kind: Kind::Behavior, handler: damage },
    Keyword { name: "MOHR COULOMB", kind: Kind::Behavior, handler: behavior },
    Keyword { name: "MOHR COULOMB HARDENING", kind: Kind::Behavior, handler: behavior },
];

/// *DAMAGE INITIATION and *DAMAGE EVOLUTION may appear several times in a
/// material (one criterion per *DAMAGE INITIATION, followed by its evolution);
/// the order is preserved and interpreted in `materials/ductile_damage.rs`.
fn damage(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    m.behaviors.push(kw.clone());
    Ok(())
}
