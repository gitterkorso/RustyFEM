//! Area gaps1: *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL has no
//! material, so the core does not see that the model is nonlinear. As for
//! user elements (`user_element.rs`), an interaction without contribution is registered
//! that says the model is not linear, so that *STATIC uses Newton.
//! Only nonlinear relations (ELASTIC or elastic-plastic) do this.

use super::{Interaction, InteractionFactory};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["AXIAL", "M1", "M2", "TORQUE"], build }];

struct NewtonForSection(String);

impl Interaction for NewtonForSection {
    fn name(&self) -> String {
        self.0.clone()
    }
    fn is_linear(&self) -> bool {
        false
    }
}

fn build(_m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    if kw.has("LINEAR") {
        return Ok(Vec::new());
    }
    Ok(vec![Box::new(NewtonForSection(format!("*{} ({})", kw.name, kw.location())))])
}
