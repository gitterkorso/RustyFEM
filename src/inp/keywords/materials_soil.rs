//! Materialbeteenden för tryckberoende plasticitet och skada: Mohr–Coulomb,
//! betong med skadeplasticitet (Lee–Fenves) och duktil skada.
//!
//! Beteendena sparas i materialet och tolkas av `materials/mohr_coulomb.rs`,
//! `materials/concrete_damaged.rs` och `materials/ductile_damage.rs`.

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

/// *DAMAGE INITIATION och *DAMAGE EVOLUTION får förekomma flera gånger i ett
/// material (ett kriterium per *DAMAGE INITIATION, följt av sin utveckling);
/// ordningen bevaras och tolkas i `materials/ductile_damage.rs`.
fn damage(ctx: &mut InpCtx, kw: &RawKeyword) -> Result<()> {
    let m = ctx.material_mut(kw)?;
    m.behaviors.push(kw.clone());
    Ok(())
}
