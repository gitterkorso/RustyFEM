//! Interaktioner och villkor med register.
//!
//! Varje modul exporterar `pub static FACTORIES: &[InteractionFactory]`.
//! Registrera en ny modul med EN rad i listan nedan (alfabetisk ordning).

mod api;
pub mod geom;
pub use api::*;

use crate::fem::model::FeModel;
use crate::model::Result;

macro_rules! interaction_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// Alla registrerade interaktionsfabriker.
        pub fn factories() -> Vec<&'static InteractionFactory> {
            let mut v = Vec::new();
            $(v.extend($m::FACTORIES.iter());)*
            v
        }
    };
}

// En rad per modul, i alfabetisk ordning.
interaction_modules! {
    acoustic,
    analytical,
    cavity,
    cohesive_contact,
    connector,
    constraints,
    contact,
    gaps1,
    radiation_cavity,
    rigid_body,
    springs,
    thermal_contact,
    user_element,
    xfem,
}

/// Bygger alla interaktioner i modellen från `FeModel::extra`.
pub fn build_all(model: &FeModel) -> Result<Vec<Box<dyn Interaction>>> {
    let facs = factories();
    let mut out = Vec::new();
    for kw in &model.extra {
        for f in &facs {
            if f.keywords.contains(&kw.name.as_str()) {
                out.extend((f.build)(model, kw)?);
            }
        }
    }
    Ok(out)
}
