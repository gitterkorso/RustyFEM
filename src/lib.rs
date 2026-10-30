//! RustyFEM – en FEM-lösare i Rust.
//!
//! Två ingångar:
//! - `.rusty`-skriptspråket och `Model` (enkel modellbyggare, se `script`/`model`);
//!   modellen översätts till kärnans `FeModel` och löses med step-ramverket.
//! - indatafiler `.inp` (`inp::read_file`) → `FeModel` → `steps::run`.
//!
//! Struktur (se ARCHITECTURE.md): `fem` (kärna), `elements`, `materials`,
//! `steps`, `inp`, `interactions`, `solver`, `regression`. Varje område har ett
//! register där nya delar läggs till med en rad.

pub mod elements;
pub mod fem;
pub mod inp;
pub mod interactions;
pub mod kinematics;
pub mod legacy;
pub mod materials;
pub mod model;
mod par;
pub mod progress;
pub mod regression;
pub mod results;
pub mod script;
pub mod section;
pub mod settings;
pub mod solver;
pub mod sparse;
pub mod steps;
pub mod user;

pub use model::{Dim, ElemKind, FemError, Material, Model, Result, Sel, RX, RY, RZ, UX, UY, UZ};
pub use results::Results;
pub use section::BeamSection;
