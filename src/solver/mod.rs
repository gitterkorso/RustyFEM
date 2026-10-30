//! Linjära ekvationslösare och deras register.
//!
//! Varje modul exporterar `pub static SOLVERS: &[&dyn LinearSolver]`.
//! Registrera en ny modul med EN rad i listan nedan (alfabetisk ordning).
//! Lösaren väljs med namn (miljövariabeln `RUSTYFEM_SOLVER`, standard "direct").

mod api;
pub use api::*;

use crate::model::{fail, Result};

macro_rules! solver_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// Alla registrerade lösare.
        pub fn all() -> Vec<&'static dyn LinearSolver> {
            let mut v: Vec<&'static dyn LinearSolver> = Vec::new();
            $(v.extend($m::SOLVERS.iter().copied());)*
            v
        }
    };
}

// En rad per modul, i alfabetisk ordning.
solver_modules! {
    direct,
    eigen,
    indefinite,
    iterative,
}

/// Lösare med namnet `name`, eller `RUSTYFEM_SOLVER`/"direct" om inget anges.
pub fn select(name: Option<&str>) -> Result<&'static dyn LinearSolver> {
    let want = name.map(|s| s.to_lowercase()).or_else(|| crate::settings::with(|s| s.solver.clone())).unwrap_or_else(|| "direct".into());
    let all = all();
    match all.iter().find(|s| s.name() == want) {
        Some(s) => Ok(*s),
        None => fail(format!(
            "lösaren '{want}' finns inte (registrerade: {})",
            all.iter().map(|s| s.name()).collect::<Vec<_>>().join(", ")
        )),
    }
}

/// Linjär (Model-)lösning för .rusty-modeller; finns kvar för bakåtkompatibilitet.
pub use crate::legacy::solve;
