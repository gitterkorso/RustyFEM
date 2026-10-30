//! Materialegenskaper för multifysik: elektrisk ledning (*ELECTRICAL
//! CONDUCTIVITY, *JOULE HEAT FRACTION), piezoelektricitet (*PIEZOELECTRIC,
//! *DIELECTRIC) och masstransport (*DIFFUSIVITY, *SOLUBILITY).
//!
//! Egenskaperna läses direkt ur materialdefinitionen av steg-typerna
//! (`steps/thermal_electric.rs`, `steps/piezo.rs`, `steps/mass_diffusion.rs`).
//! En passiv post förbrukar beteendena när ett annat material väljs (t.ex.
//! *ELASTIC för piezoelektriska material, *CONDUCTIVITY för ledare), och en
//! fabrik med lägst prioritet bygger en modell för material som bara har
//! diffusions- eller elektriska egenskaper.

use super::heat::{read_table, CondKind, Table};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

const BEHAVIORS: &[&str] = &["ELECTRICAL CONDUCTIVITY", "JOULE HEAT FRACTION", "PIEZOELECTRIC", "DIELECTRIC", "DIFFUSIVITY", "SOLUBILITY"];

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory { name: "multifysikegenskaper (läses av steg-typerna)", consumes: BEHAVIORS, priority: 0, build: None },
    MaterialFactory {
        name: "endast diffusions- eller elektriska egenskaper",
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
        "endast diffusions- eller elektriska egenskaper"
    }
    fn is_linear(&self) -> bool {
        true
    }
    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: saknar mekaniska egenskaper men används i ett mekaniskt element", self.name))
    }
}

fn kind_of(kw: &RawKeyword) -> Result<(CondKind, usize)> {
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
    Ok(match ty.as_str() {
        "ISO" | "ISOTROPIC" => (CondKind::Iso, 1),
        "ORTHO" | "ORTHOTROPIC" => (CondKind::Ortho, 3),
        "ANISO" | "ANISOTROPIC" => (CondKind::Aniso, 6),
        other => return kw.err(format!("TYPE={other} stöds inte (ISO, ORTHO, ANISO)")),
    })
}

/// 3×3-tensor ur tabellvärden.
pub fn tensor(kind: CondKind, v: &[f64]) -> [[f64; 3]; 3] {
    match kind {
        CondKind::Iso => [[v[0], 0.0, 0.0], [0.0, v[0], 0.0], [0.0, 0.0, v[0]]],
        CondKind::Ortho => [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]],
        CondKind::Aniso => [[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]],
    }
}

/// Elektriska egenskaper: σ(T) och andelen av Joulevärmen som blir värme.
#[derive(Debug, Clone)]
pub struct ElecProps {
    pub sigma: (CondKind, Table),
    pub joule_fraction: f64,
}

impl ElecProps {
    pub fn from_def(def: &MaterialDef) -> Result<Option<ElecProps>> {
        let Some(kw) = def.behavior("ELECTRICAL CONDUCTIVITY") else { return Ok(None) };
        if kw.has("FREQUENCY") {
            return kw.err("FREQUENCY stöds inte (bara likström)");
        }
        let (kind, n) = kind_of(kw)?;
        let sigma = (kind, read_table(kw, n)?);
        let joule_fraction = match def.behavior("JOULE HEAT FRACTION") {
            Some(j) => j.first_floats()?.first().copied().unwrap_or(1.0),
            None => 1.0,
        };
        Ok(Some(ElecProps { sigma, joule_fraction }))
    }

    /// σ och dσ/dT vid temperaturen `t`.
    pub fn conductivity(&self, t: f64) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        let (v, d) = self.sigma.1.eval(t);
        (tensor(self.sigma.0, &v), tensor(self.sigma.0, &d))
    }
}

/// Piezoelektriska egenskaper: spänningskoefficienterna e (3×6, rad m,
/// kolumner 11, 22, 33, 12, 13, 23) och dielektricitetstensorn κ.
#[derive(Debug, Clone)]
pub struct PiezoProps {
    /// e om TYPE=S; töjningskoefficienterna d om TYPE=E (e = d C räknas av elementet).
    pub coef: [[f64; 6]; 3],
    pub strain_type: bool,
    pub kappa: [[f64; 3]; 3],
}

impl PiezoProps {
    pub fn from_def(def: &MaterialDef) -> Result<Option<PiezoProps>> {
        let Some(kd) = def.behavior("DIELECTRIC") else {
            if def.behavior("PIEZOELECTRIC").is_some() {
                return fail(format!("material {}: *PIEZOELECTRIC kräver *DIELECTRIC", def.name));
            }
            return Ok(None);
        };
        let (kind, n) = kind_of(kd)?;
        let kt = read_table(kd, n)?;
        if kt.varies() {
            return kd.err("temperaturberoende dielektricitet stöds inte ännu");
        }
        let kappa = tensor(kind, &kt.pts[0].1);
        let mut coef = [[0.0; 6]; 3];
        let mut strain_type = false;
        if let Some(kp) = def.behavior("PIEZOELECTRIC") {
            let ty = kp.param_upper("TYPE").unwrap_or_else(|| "S".into());
            strain_type = match ty.as_str() {
                "S" => false,
                "E" => true,
                other => return kp.err(format!("TYPE={other} stöds inte (S eller E)")),
            };
            let mut v = Vec::new();
            for d in kp.records() {
                v.extend(kp.floats(&d)?);
            }
            if v.len() < 18 {
                return kp.err("18 koefficienter väntades");
            }
            if v.len() > 18 && v[18..].iter().any(|x| *x != 0.0) {
                return kp.err("temperaturberoende piezoelektriska koefficienter stöds inte ännu");
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

/// Diffusionsegenskaper: diffusiviteten D (tensor) och lösligheten s.
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
                return kw.err(format!("LAW={l} stöds inte ännu (bara GENERAL)"));
            }
        }
        let (kind, n) = kind_of(kw)?;
        let rows: Vec<Vec<f64>> = kw.records().iter().map(|d| kw.floats(d)).collect::<Result<_>>()?;
        let rows: Vec<Vec<f64>> = rows.into_iter().filter(|r| r.iter().any(|x| *x != 0.0)).collect();
        if rows.len() != 1 {
            return kw.err("diffusivitet som beror av koncentration eller temperatur stöds inte ännu (en datarad)");
        }
        let mut v = rows[0].clone();
        v.resize(n, 0.0);
        let s = match def.behavior("SOLUBILITY") {
            Some(k) => {
                let t = read_table(k, 1)?;
                if t.varies() {
                    return k.err("temperaturberoende löslighet stöds inte ännu");
                }
                t.pts[0].1[0]
            }
            None => return fail(format!("material {}: *DIFFUSIVITY kräver *SOLUBILITY", def.name)),
        };
        Ok(Some(DiffProps { d: tensor(kind, &v), s }))
    }
}
