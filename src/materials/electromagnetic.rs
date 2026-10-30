//! Electromagnetic material properties: *MAGNETIC PERMEABILITY.
//!
//! The values are read directly from the material definition by `steps/electromagnetic.rs`
//! (together with *ELECTRICAL CONDUCTIVITY). A passive entry consumes
//! the behavior when another material is selected, and a lowest-priority factory
//! builds a non-mechanical model for materials that only have permeability.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory { name: "electromagnetic properties (read by the step type)", consumes: &["MAGNETIC PERMEABILITY"], priority: 0, build: None },
    MaterialFactory { name: "electromagnetic properties only", consumes: &["MAGNETIC PERMEABILITY", "ELECTRICAL CONDUCTIVITY"], priority: -102, build: Some(build) },
];

#[derive(Debug)]
struct Magnetic {
    name: String,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if def.behavior("MAGNETIC PERMEABILITY").is_none() {
        return Ok(None);
    }
    Ok(Some(Arc::new(Magnetic { name: def.name.clone() })))
}

impl MaterialModel for Magnetic {
    fn name(&self) -> &str {
        "electromagnetic properties only"
    }
    fn is_linear(&self) -> bool {
        true
    }
    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: has no mechanical properties but is used in a mechanical element", self.name))
    }
}
