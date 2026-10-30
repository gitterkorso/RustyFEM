//! Nyckelord för kontaktens initiering och kontroller i general contact
//! (område contact3). *CONTACT INITIALIZATION DATA och
//! *CONTACT INITIALIZATION ASSIGNMENT sparas i modellen och används när filen
//! är läst (`interactions::contact::adjust_slaves`). *CONTACT CONTROLS
//! ASSIGNMENT (STABILIZE, AUTOMATIC TOLERANCES för general contact) godtas
//! som *CONTACT CONTROLS: de påverkar iterationsvägen men inte det
//! konvergerade slutläget.

use crate::inp::{ignore, store_model, Keyword, Kind};

pub static KEYWORDS: &[Keyword] = &[
    Keyword { name: "CONTACT CONTROLS ASSIGNMENT", kind: Kind::Both, handler: ignore },
    Keyword { name: "CONTACT INITIALIZATION ASSIGNMENT", kind: Kind::Both, handler: store_model },
    Keyword { name: "CONTACT INITIALIZATION DATA", kind: Kind::Model, handler: store_model },
];
