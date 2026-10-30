//! Helt inkompressibel isotrop elasticitet (*ELASTIC med ν = 0,5).
//!
//! ν = 0,5 godtas i hybridelement. Elementmatriserna från
//! *MATRIX GENERATE (C3D4H, C3D8H, ...) räknas då med
//! ν = 0,4999999, dvs. kompressionsmodulen K = E / (3 · 2·10⁻⁷) ≈ 5·10⁶ μ
//! (βK-termen i förskjutningsblocket och tryckblocket −(1 − β)/K), och att
//! elementet dessutom vet att materialet är inkompressibelt (C3D4H:s
//! trycksstabilisering skalas då med 1/μ i stället för 1/K, se
//! `elements/hybrid_tet.rs`).
//!
//! `claim` anropas från `materials::build` före registret: har materialet
//! *ELASTIC (isotrop, en datarad) med ν = 0,5 byggs materialet om med
//! ν = `NU_INCOMPRESSIBLE` på samma sätt som vanligt (elasticitet,
//! *PLASTIC, ...). Andra element än hybridelementen får då ett nästan
//! inkompressibelt material; hybridelementen känner igen
//! ν ≥ `NU_FLAG` som helt inkompressibelt.

use super::{MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[];

/// Poissontalet som används för ν = 0,5.
pub const NU_INCOMPRESSIBLE: f64 = 0.4999999;

/// Gräns över vilken ett elastiskt material räknas som helt inkompressibelt.
pub const NU_FLAG: f64 = 0.4999999 - 1e-12;

/// Är (E, ν) ett helt inkompressibelt material (ν = 0,5 i indata)?
pub fn is_incompressible(nu: f64) -> bool {
    nu >= NU_FLAG
}

/// Materialet byggt med ν = 0,4999999 om *ELASTIC har ν = 0,5, annars `None`.
pub fn claim(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(i) = def.behaviors.iter().position(|b| b.name == "ELASTIC") else {
        return Ok(None);
    };
    let kw = &def.behaviors[i];
    if kw.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || kw.data.len() != 1 || kw.has("DEPENDENCIES") {
        return Ok(None);
    }
    let fields = &kw.data[0].fields;
    let Some(nu) = fields.get(1).and_then(|s| crate::fem::model::parse_f64(s)) else {
        return Ok(None);
    };
    if (nu - 0.5).abs() > 1e-12 {
        return Ok(None);
    }
    let mut d = def.clone();
    d.behaviors[i].data[0].fields[1] = format!("{NU_INCOMPRESSIBLE}");
    super::build(&d).map(Some)
}
