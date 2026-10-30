//! User elements (*USER ELEMENT) can be nonlinear without the core knowing
//! it (the element interface has no `is_linear`). Implicit analysis always
//! iterates with Newton; an interaction without contribution is therefore registered here that
//! says the model is not linear, so that *STATIC uses Newton–Raphson
//! (a linear UEL then converges in the first iteration). As a consequence,
//! perturbation steps with user elements are not supported.

use super::{Interaction, InteractionFactory};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["USER ELEMENT"], build }];

struct NewtonForUel(String);

impl Interaction for NewtonForUel {
    fn name(&self) -> String {
        self.0.clone()
    }
    fn is_linear(&self) -> bool {
        false
    }
}

fn build(_m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    Ok(vec![Box::new(NewtonForUel(format!("*USER ELEMENT ({})", kw.location())))])
}
