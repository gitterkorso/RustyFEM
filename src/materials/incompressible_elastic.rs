//! Fully incompressible isotropic elasticity (*ELASTIC with ν = 0.5).
//!
//! ν = 0.5 is accepted in hybrid elements. The element matrices from
//! *MATRIX GENERATE (C3D4H, C3D8H, ...) are then computed with
//! ν = 0.4999999, i.e. bulk modulus K = E / (3 · 2·10⁻⁷) ≈ 5·10⁶ μ
//! (the βK term in the displacement block and the pressure block −(1 − β)/K), and
//! the element additionally knows that the material is incompressible (C3D4H's
//! pressure stabilization is then scaled by 1/μ instead of 1/K, see
//! `elements/hybrid_tet.rs`).
//!
//! `claim` is called from `materials::build` before the registry: if the material
//! has *ELASTIC (isotropic, one data line) with ν = 0.5, the material is rebuilt with
//! ν = `NU_INCOMPRESSIBLE` in the usual way (elasticity,
//! *PLASTIC, ...). Elements other than the hybrid elements then get a nearly
//! incompressible material; the hybrid elements recognize
//! ν ≥ `NU_FLAG` as fully incompressible.

use super::{MaterialFactory, MaterialModel};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[];

/// Poisson's ratio used for ν = 0.5.
pub const NU_INCOMPRESSIBLE: f64 = 0.4999999;

/// Threshold above which an elastic material counts as fully incompressible.
pub const NU_FLAG: f64 = 0.4999999 - 1e-12;

/// Is (E, ν) a fully incompressible material (ν = 0.5 in the input)?
pub fn is_incompressible(nu: f64) -> bool {
    nu >= NU_FLAG
}

/// The material built with ν = 0.4999999 if *ELASTIC has ν = 0.5, otherwise `None`.
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
