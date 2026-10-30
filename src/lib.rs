//! RustyFEM – a FEM solver in Rust.
//!
//! Two entry points:
//! - the `.rusty` script language and `Model` (simple model builder, see `script`/`model`);
//!   the model is translated into the core `FeModel` and solved with the step framework.
//! - input files in .inp format (`inp::read_file`) → `FeModel` → `steps::run`.
//!
//! Structure (see ARCHITECTURE.md): `fem` (core), `elements`, `materials`,
//! `steps`, `inp`, `interactions`, `solver`, `regression`. Each area has a
//! registry where new parts are added with one line.

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
