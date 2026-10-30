//! Material properties for multiphysics: electrical conduction (*ELECTRICAL
//! CONDUCTIVITY, *JOULE HEAT FRACTION), piezoelectricity (*PIEZOELECTRIC,
//! *DIELECTRIC) and mass transport (*DIFFUSIVITY, *SOLUBILITY).
//!
//! The properties are read directly from the material definition by the step types
//! (`steps/thermal_electric.rs`, `steps/piezo.rs`, `steps/mass_diffusion.rs`).
//! A passive entry consumes the behaviors when another material is selected (e.g.
//! *ELASTIC for piezoelectric materials, *CONDUCTIVITY for conductors), and a
//! lowest-priority factory builds a model for materials that only have
//! diffusion or electrical properties.

use super::heat::{read_table, CondKind, Table};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

const BEHAVIORS: &[&str] = &["ELECTRICAL CONDUCTIVITY", "JOULE HEAT FRACTION", "PIEZOELECTRIC", "DIELECTRIC", "DIFFUSIVITY", "SOLUBILITY"];

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory { name: "multiphysics properties (read by the step types)", consumes: BEHAVIORS, priority: 0, build: None },
    MaterialFactory {
        name: "only diffusion or electrical properties",
        consumes: &["ELECTRICAL CONDUCTIVITY", "JOULE HEAT FRACTION", "DIELECTRIC", "DIFFUSIVITY", "SOLUBILITY", "CONDUCTIVITY", "SPECIFIC HEAT", "LATENT HEAT"],
        priority: -101,
        build: Some(build_only),
    },
];

#[derive(Debug)]
struct NonMechanical {
    name: String,
}

fn build_only(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if ["ELECTRICAL CONDUCTIVITY", "DIFFUSIVITY", "SOLUBILITY", "DIELECTRIC"].iter().all(|b| def.behavior(b).is_none()) {
        return Ok(None);
    }
    Ok(Some(Arc::new(NonMechanical { name: def.name.clone() })))
}

impl MaterialModel for NonMechanical {
    fn name(&self) -> &str {
        "only diffusion or electrical properties"
    }
    fn is_linear(&self) -> bool {
        true
    }
    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: lacks mechanical properties but is used in a mechanical element", self.name))
    }
}

fn kind_of(kw: &RawKeyword) -> Result<(CondKind, usize)> {
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
    Ok(match ty.as_str() {
        "ISO" | "ISOTROPIC" => (CondKind::Iso, 1),
        "ORTHO" | "ORTHOTROPIC" => (CondKind::Ortho, 3),
        "ANISO" | "ANISOTROPIC" => (CondKind::Aniso, 6),
        other => return kw.err(format!("TYPE={other} is not supported (ISO, ORTHO, ANISO)")),
    })
}

/// 3×3 tensor from table values.
pub fn tensor(kind: CondKind, v: &[f64]) -> [[f64; 3]; 3] {
    match kind {
        CondKind::Iso => [[v[0], 0.0, 0.0], [0.0, v[0], 0.0], [0.0, 0.0, v[0]]],
        CondKind::Ortho => [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]],
        CondKind::Aniso => [[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]],
    }
}

/// Electrical properties: σ(T) and the fraction of the Joule heat that becomes heat.
#[derive(Debug, Clone)]
pub struct ElecProps {
    pub sigma: (CondKind, Table),
    pub joule_fraction: f64,
}

impl ElecProps {
    pub fn from_def(def: &MaterialDef) -> Result<Option<ElecProps>> {
        let Some(kw) = def.behavior("ELECTRICAL CONDUCTIVITY") else { return Ok(None) };
        if kw.has("FREQUENCY") {
            return kw.err("FREQUENCY is not supported (direct current only)");
        }
        let (kind, n) = kind_of(kw)?;
        let sigma = (kind, read_table(kw, n)?);
        let joule_fraction = match def.behavior("JOULE HEAT FRACTION") {
            Some(j) => j.first_floats()?.first().copied().unwrap_or(1.0),
            None => 1.0,
        };
        Ok(Some(ElecProps { sigma, joule_fraction }))
    }

    /// σ and dσ/dT at the temperature `t`.
    pub fn conductivity(&self, t: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let (v, d) = self.sigma.1.eval(t);
        (tensor(self.sigma.0, &v), tensor(self.sigma.0, &d))
    }
}

/// Piezoelectric properties: the stress coefficients e (3×6, row m,
/// columns 11, 22, 33, 12, 13, 23) and the dielectric tensor κ.
#[derive(Debug, Clone)]
pub struct PiezoProps {
    /// e if TYPE=S; the strain coefficients d if TYPE=E (e = d C is computed by the element).
    pub coef: [[f64; 6]; 3],
    pub strain_type: bool,
    pub kappa: [[f64; 3]; 3],
}

impl PiezoProps {
    pub fn from_def(def: &MaterialDef) -> Result<Option<PiezoProps>> {
        let Some(kd) = def.behavior("DIELECTRIC") else {
            if def.behavior("PIEZOELECTRIC").is_some() {
                return fail(format!("material {}: *PIEZOELECTRIC requires *DIELECTRIC", def.name));
            }
            return Ok(None);
        };
        let (kind, n) = kind_of(kd)?;
        let kt = read_table(kd, n)?;
        if kt.varies() {
            return kd.err("temperature-dependent dielectric properties are not supported yet");
        }
        let kappa = tensor(kind, &kt.pts[0].1);
        let mut coef = [[0.0; 6]; 3];
        let mut strain_type = false;
        if let Some(kp) = def.behavior("PIEZOELECTRIC") {
            let ty = kp.param_upper("TYPE").unwrap_or_else(|| "S".into());
            strain_type = match ty.as_str() {
                "S" => false,
                "E" => true,
                other => return kp.err(format!("TYPE={other} is not supported (S or E)")),
            };
            let mut v = Vec::new();
            for d in kp.records() {
                v.extend(kp.floats(&d)?);
            }
            if v.len() < 18 {
                return kp.err("18 coefficients were expected");
            }
            if v.len() > 18 && v[18..].iter().any(|x| *x != 0.0) {
                return kp.err("temperature-dependent piezoelectric coefficients are not supported yet");
            }
            for m in 0..3 {
                for k in 0..6 {
                    coef[m][k] = v[6 * m + k];
                }
            }
        }
        Ok(Some(PiezoProps { coef, strain_type, kappa }))
    }
}

/// Diffusion properties: the diffusivity D (tensor) and the solubility s.
#[derive(Debug, Clone)]
pub struct DiffProps {
    pub d: [[f64; 3]; 3],
    pub s: f64,
}

impl DiffProps {
    pub fn from_def(def: &MaterialDef) -> Result<Option<DiffProps>> {
        let Some(kw) = def.behavior("DIFFUSIVITY") else { return Ok(None) };
        if let Some(l) = kw.param_upper("LAW") {
            if l != "GENERAL" {
                return kw.err(format!("LAW={l} is not supported yet (only GENERAL)"));
            }
        }
        let (kind, n) = kind_of(kw)?;
        let rows: Vec<Vec<f64>> = kw.records().iter().map(|d| kw.floats(d)).collect::<Result<_>>()?;
        let rows: Vec<Vec<f64>> = rows.into_iter().filter(|r| r.iter().any(|x| *x != 0.0)).collect();
        if rows.len() != 1 {
            return kw.err("diffusivity depending on concentration or temperature is not supported yet (one data line)");
        }
        let mut v = rows[0].clone();
        v.resize(n, 0.0);
        let s = match def.behavior("SOLUBILITY") {
            Some(k) => {
                let t = read_table(k, 1)?;
                if t.varies() {
                    return k.err("temperature-dependent solubility is not supported yet");
                }
                t.pts[0].1[0]
            }
            None => return fail(format!("material {}: *DIFFUSIVITY requires *SOLUBILITY", def.name)),
        };
        Ok(Some(DiffProps { d: tensor(kind, &v), s }))
    }
}
