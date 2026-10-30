//! Linear equation solvers and their registry.
//!
//! Each module exports `pub static SOLVERS: &[&dyn LinearSolver]`.
//! Register a new module with ONE line in the list below (alphabetical order).
//! The solver is selected by name (environment variable `RUSTYFEM_SOLVER`, default "direct").

mod api;
pub use api::*;

use crate::model::{fail, Result};

macro_rules! solver_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered solvers.
        pub fn all() -> Vec<&'static dyn LinearSolver> {
            let mut v: Vec<&'static dyn LinearSolver> = Vec::new();
            $(v.extend($m::SOLVERS.iter().copied());)*
            v
        }
    };
}

// One line per module, in alphabetical order.
solver_modules! {
    direct,
    eigen,
    indefinite,
    iterative,
}

/// Solver named `name`, or `RUSTYFEM_SOLVER`/"direct" if none is given.
pub fn select(name: Option<&str>) -> Result<&'static dyn LinearSolver> {
    let want = name.map(|s| s.to_lowercase()).or_else(|| crate::settings::with(|s| s.solver.clone())).unwrap_or_else(|| "direct".into());
    let all = all();
    match all.iter().find(|s| s.name() == want) {
        Some(s) => Ok(*s),
        None => fail(format!(
            "solver '{want}' does not exist (registered: {})",
            all.iter().map(|s| s.name()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Linear (Model) solution for .rusty models; kept for backward compatibility.
pub use crate::legacy::solve;
