//! Interactions and constraints with a registry.
//!
//! Each module exports `pub static FACTORIES: &[InteractionFactory]`.
//! Register a new module with ONE line in the list below (alphabetical order).

mod api;
pub mod geom;
pub use api::*;

use crate::fem::model::FeModel;
use crate::model::Result;

macro_rules! interaction_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered interaction factories.
        pub fn factories() -> Vec<&'static InteractionFactory> {
            let mut v = Vec::new();
            $(v.extend($m::FACTORIES.iter());)*
            v
        }
    };
}

// One line per module, in alphabetical order.
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

/// Builds all interactions in the model from `FeModel::extra`.
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
