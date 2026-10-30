//! Core: model, degrees of freedom, assembly, state and output.
//!
//! The files here are shared by all areas. Only change them when an interface
//! really needs to be extended (see ARCHITECTURE.md).

pub mod assembly;
pub mod dat;
pub mod dof;
pub mod model;
pub mod mpc;
pub mod output;
pub mod state;
pub mod system;
