//! Område gaps1: *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL har inget
//! material, så kärnan ser inte att modellen är olinjär. Som för
//! användarelement (`user_element.rs`) registreras en interaktion utan bidrag
//! som säger att modellen inte är linjär, så att *STATIC använder Newton.
//! Bara olinjära samband (ELASTIC eller elastiskt-plastiskt) gör det.

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
