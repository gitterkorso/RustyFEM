//! Nya element (område elements3):
//!
//! - `elements3_stri65.rs`: STRI65 (sexnodigt tunt triangelskal med fem
//!   frihetsgrader per nod),
//! - `elements3_cgax.rs`: CGAX3, CGAX4, CGAX6, CGAX8, CGAX8R (generaliserade
//!   axisymmetriska element med vridning, frihetsgrad 5),
//! - `elements3_saxa.rs`: SAXA11–SAXA14, SAXA21–SAXA24 (axisymmetriska skal
//!   med asymmetrisk deformation, Fouriermoder), linjärt elastiska.
//!
//! Formuleringarna beskrivs i respektive fil; se även
//! `tests/cases/elements3/matriser/` och `tests/elements3.rs`.
//!
//! ELBOW31/ELBOW32/ELBOW31B/ELBOW31C finns i `elbow.rs` (område beams3).

#[path = "elements3_cgax.rs"]
pub mod cgax;
#[path = "elements3_saxa.rs"]
pub mod saxa;
#[path = "elements3_stri65.rs"]
pub mod stri65;

use super::api::ElementType;

pub static TYPES: &[&dyn ElementType] = &[
    &stri65::STRI65,
    &cgax::CGAX3,
    &cgax::CGAX4,
    &cgax::CGAX6,
    &cgax::CGAX8,
    &cgax::CGAX8R,
    &saxa::SAXA11,
    &saxa::SAXA12,
    &saxa::SAXA13,
    &saxa::SAXA14,
    &saxa::SAXA21,
    &saxa::SAXA22,
    &saxa::SAXA23,
    &saxa::SAXA24,
];
