//! Elektromagnetiska materialegenskaper: *MAGNETIC PERMEABILITY.
//!
//! Värdena läses direkt ur materialdefinitionen av `steps/electromagnetic.rs`
//! (tillsammans med *ELECTRICAL CONDUCTIVITY). En passiv post förbrukar
//! beteendet när ett annat material väljs, och en fabrik med lägst prioritet
//! bygger en icke-mekanisk modell för material som bara har permeabilitet.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory { name: "elektromagnetiska egenskaper (läses av steg-typen)", consumes: &["MAGNETIC PERMEABILITY"], priority: 0, build: None },
    MaterialFactory { name: "endast elektromagnetiska egenskaper", consumes: &["MAGNETIC PERMEABILITY", "ELECTRICAL CONDUCTIVITY"], priority: -102, build: Some(build) },
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
        "endast elektromagnetiska egenskaper"
    }
    fn is_linear(&self) -> bool {
        true
    }
    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: saknar mekaniska egenskaper men används i ett mekaniskt element", self.name))
    }
}
