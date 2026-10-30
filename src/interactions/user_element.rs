//! Användarelement (*USER ELEMENT) kan vara olinjära utan att kärnan vet om
//! det (elementgränssnittet saknar `is_linear`). Implicit analys itererar
//! alltid med Newton; här registreras därför en interaktion utan bidrag som
//! säger att modellen inte är linjär, så att *STATIC använder Newton–Raphson
//! (en linjär UEL konvergerar då i första iterationen). Följden är att
//! perturbationssteg med användarelement inte stöds.

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
