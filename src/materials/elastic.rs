//! Linjär isotrop elasticitet (*ELASTIC, TYPE=ISOTROPIC).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "ELASTIC (isotrop)", consumes: &["ELASTIC"], priority: 0, build: Some(build) }];

#[derive(Debug, Clone)]
pub struct IsotropicElastic {
    pub e: f64,
    pub nu: f64,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("ELASTIC") else { return Ok(None) };
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" {
        return Ok(None);
    }
    if kw.data.len() > 1 && !kw.has("DEPENDENCIES") {
        // temperaturberoende (E, ν, T): hanteras av thermal_elastic.rs
        return Ok(None);
    }
    if kw.has("DEPENDENCIES") || kw.has("TEMPERATURE") || kw.data.len() > 1 {
        return kw.err("temperaturberoende elasticitet stöds inte ännu");
    }
    let v = kw.first_floats()?;
    if v.is_empty() || !(v[0] > 0.0) {
        return kw.err("E saknas eller är inte positivt");
    }
    let nu = v.get(1).copied().unwrap_or(0.0);
    if !(nu > -1.0 && nu < 0.5) {
        return kw.err(format!("ν = {nu} ska ligga mellan −1 och 0,5"));
    }
    Ok(Some(Arc::new(IsotropicElastic { e: v[0], nu })))
}

impl IsotropicElastic {
    pub fn shear_modulus(&self) -> f64 {
        self.e / (2.0 * (1.0 + self.nu))
    }

    /// Elasticitetsmatrisen i komponentordningen för `mode`.
    pub fn dmat(&self, mode: StressMode) -> Vec<f64> {
        let (e, nu) = (self.e, self.nu);
        match mode {
            StressMode::Uniaxial => vec![e],
            StressMode::PlaneStress => {
                let c = e / (1.0 - nu * nu);
                vec![c, c * nu, 0.0, c * nu, c, 0.0, 0.0, 0.0, c * (1.0 - nu) / 2.0]
            }
            _ => {
                let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
                let mu = e / (2.0 * (1.0 + nu));
                let n = mode.ncomp();
                let mut d = vec![0.0; n * n];
                for i in 0..3 {
                    for j in 0..3 {
                        d[i * n + j] = lam;
                    }
                    d[i * n + i] = lam + 2.0 * mu;
                }
                for i in 3..n {
                    d[i * n + i] = mu;
                }
                d
            }
        }
    }
}

impl MaterialModel for IsotropicElastic {
    fn name(&self) -> &str {
        "ELASTIC"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let d = self.dmat(input.mode);
        let stress = super::matvec(&d, input.strain);
        Ok(MatOutput { stress, tangent: d })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(self.dmat(mode))
    }
}
