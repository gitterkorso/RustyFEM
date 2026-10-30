//! Thermal material properties: *CONDUCTIVITY, *SPECIFIC HEAT and *EXPANSION.
//!
//! The heat conduction (`steps/heat.rs`) and the thermal stress analysis
//! (`steps/heat_stress.rs`) read the properties directly from the material definition
//! with `HeatProps::from_def`. The properties may depend on temperature (table
//! with a temperature last on each line); between the points the interpolation is linear
//! and outside the table the value is held constant.
//!
//! The registry:
//! - a passive entry consumes the thermal behaviors when a mechanical
//!   material (e.g. *ELASTIC) is selected;
//! - a low-priority factory builds a "thermal only" model for
//!   materials without mechanical properties (pure heat conduction models).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory {
        name: "thermal properties (read by the heat conduction)",
        consumes: &["CONDUCTIVITY", "SPECIFIC HEAT", "EXPANSION", "LATENT HEAT"],
        priority: 0,
        build: None,
    },
    MaterialFactory {
        name: "thermal properties only",
        consumes: &["CONDUCTIVITY", "SPECIFIC HEAT", "EXPANSION"],
        priority: -100,
        build: Some(build_thermal_only),
    },
];

/// Material model for materials that only have thermal properties. It cannot
/// be used in mechanical elements.
#[derive(Debug)]
pub struct ThermalOnly {
    name: String,
}

fn build_thermal_only(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if def.behavior("CONDUCTIVITY").is_none() && def.behavior("SPECIFIC HEAT").is_none() {
        return Ok(None);
    }
    Ok(Some(Arc::new(ThermalOnly { name: def.name.clone() })))
}

impl MaterialModel for ThermalOnly {
    fn name(&self) -> &str {
        "thermal properties only"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: has only thermal properties (*CONDUCTIVITY/*SPECIFIC HEAT) but is used in a mechanical element", self.name))
    }
}

/// Table over temperature: (T, values) sorted by T. Linear interpolation
/// between the points, constant outside.
#[derive(Debug, Clone, Default)]
pub struct Table {
    pub pts: Vec<(f64, Vec<f64>)>,
}

impl Table {
    pub fn constant(v: Vec<f64>) -> Self {
        Table { pts: vec![(0.0, v)] }
    }

    pub fn ncomp(&self) -> usize {
        self.pts.first().map_or(0, |p| p.1.len())
    }

    /// Temperature dependent (more than one point).
    pub fn varies(&self) -> bool {
        self.pts.len() > 1
    }

    /// Values and derivatives d/dT at temperature `t`.
    pub fn eval(&self, t: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.ncomp();
        let p = &self.pts;
        if p.len() == 1 || t <= p[0].0 {
            return (p[0].1.clone(), vec![0.0; n]);
        }
        let last = p.len() - 1;
        if t >= p[last].0 {
            return (p[last].1.clone(), vec![0.0; n]);
        }
        let i = p.windows(2).position(|w| t >= w[0].0 && t <= w[1].0).unwrap_or(0);
        let (t0, v0) = (&p[i].0, &p[i].1);
        let (t1, v1) = (&p[i + 1].0, &p[i + 1].1);
        let s = (t - t0) / (t1 - t0);
        let val = (0..n).map(|k| v0[k] + s * (v1[k] - v0[k])).collect();
        let der = (0..n).map(|k| (v1[k] - v0[k]) / (t1 - t0)).collect();
        (val, der)
    }

    /// First component at `t`.
    pub fn value(&self, t: f64) -> f64 {
        self.eval(t).0[0]
    }

    /// ∫ₐᵇ f(T) dT for the first component (exact for the piecewise linear table).
    pub fn integral(&self, a: f64, b: f64) -> f64 {
        if a == b {
            return 0.0;
        }
        let (lo, hi, sign) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
        let mut xs = vec![lo];
        xs.extend(self.pts.iter().map(|p| p.0).filter(|&t| t > lo && t < hi));
        xs.push(hi);
        let mut s = 0.0;
        for w in xs.windows(2) {
            s += 0.5 * (self.value(w[0]) + self.value(w[1])) * (w[1] - w[0]);
        }
        sign * s
    }
}

/// Reads a table with `ncomp` values followed by a temperature per data line.
pub fn read_table(kw: &RawKeyword, ncomp: usize) -> Result<Table> {
    if kw.has("DEPENDENCIES") {
        let d = kw.param("DEPENDENCIES").unwrap_or("").trim();
        if !(d.is_empty() || d == "0") {
            return kw.err("dependence on field variables (DEPENDENCIES=) is not supported yet");
        }
    }
    let mut pts: Vec<(f64, Vec<f64>)> = Vec::new();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.iter().all(|x| *x == 0.0) && d.fields.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        let mut vals: Vec<f64> = v.iter().take(ncomp).copied().collect();
        vals.resize(ncomp, 0.0);
        let t = v.get(ncomp).copied().unwrap_or(0.0);
        pts.push((t, vals));
    }
    if pts.is_empty() {
        return kw.err("data line missing");
    }
    if pts.len() > 1 {
        for w in pts.windows(2) {
            if w[1].0 <= w[0].0 {
                return kw.err("the temperatures in the table must be strictly increasing");
            }
        }
    }
    Ok(Table { pts })
}

/// Form of the conductivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CondKind {
    /// k
    Iso,
    /// k11, k22, k33
    Ortho,
    /// k11, k12, k22, k13, k23, k33
    Aniso,
}

/// Thermal properties of a material.
#[derive(Debug, Clone)]
pub struct HeatProps {
    pub name: String,
    pub cond: Option<(CondKind, Table)>,
    pub specific_heat: Option<Table>,
    pub density: f64,
    /// Linear expansion α(T) (isotropic) and the reference temperature ZERO=.
    pub expansion: Option<(Table, f64)>,
    /// Latent heat (*LATENT HEAT): (L, solidus temperature, liquidus temperature)
    /// per phase change.
    pub latent: Vec<(f64, f64, f64)>,
}

impl HeatProps {
    pub fn from_def(def: &MaterialDef) -> Result<HeatProps> {
        let cond = match def.behavior("CONDUCTIVITY") {
            Some(kw) => {
                let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
                let kind = match ty.as_str() {
                    "ISO" | "ISOTROPIC" => CondKind::Iso,
                    "ORTHO" | "ORTHOTROPIC" => CondKind::Ortho,
                    "ANISO" | "ANISOTROPIC" => CondKind::Aniso,
                    other => return kw.err(format!("TYPE={other} is not supported (ISO, ORTHO, ANISO)")),
                };
                let n = match kind {
                    CondKind::Iso => 1,
                    CondKind::Ortho => 3,
                    CondKind::Aniso => 6,
                };
                Some((kind, read_table(kw, n)?))
            }
            None => None,
        };
        let specific_heat = match def.behavior("SPECIFIC HEAT") {
            Some(kw) => {
                if kw.has("PER VOLUME") {
                    return kw.err("PER VOLUME is not supported yet");
                }
                Some(read_table(kw, 1)?)
            }
            None => None,
        };
        let expansion = match def.behavior("EXPANSION") {
            Some(kw) => {
                let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
                if ty != "ISO" {
                    return kw.err(format!("TYPE={ty} is not supported yet (only ISO)"));
                }
                if kw.has("USER") {
                    return kw.err("USER is not supported");
                }
                let zero = match kw.param("ZERO") {
                    Some(z) => crate::fem::model::parse_f64(z).ok_or_else(|| crate::model::FemError(format!("{}: ZERO={z} is not a number", kw.location())))?,
                    None => 0.0,
                };
                Some((read_table(kw, 1)?, zero))
            }
            None => None,
        };
        let mut latent = Vec::new();
        if let Some(kw) = def.behavior("LATENT HEAT") {
            for d in kw.records() {
                let v = kw.floats(&d)?;
                if v.len() < 3 || v.iter().all(|x| *x == 0.0) {
                    continue;
                }
                if v[2] < v[1] {
                    return kw.err_at(d.line, "the liquidus temperature must be at least the solidus temperature");
                }
                latent.push((v[0], v[1], v[2]));
            }
            if latent.is_empty() {
                return kw.err("write 'latent heat, solidus temperature, liquidus temperature'");
            }
        }
        Ok(HeatProps { name: def.name.clone(), cond, specific_heat, density: def.density, expansion, latent })
    }

    /// The conductivity as a 3×3 tensor and its derivative d/dT at temperature `t`.
    pub fn conductivity(&self, t: f64) -> Result<([[f64; 3]; 3], [[f64; 3]; 3])> {
        let Some((kind, tab)) = &self.cond else {
            return fail(format!("material {}: *CONDUCTIVITY missing", self.name));
        };
        let (v, d) = tab.eval(t);
        let m = |v: &[f64]| -> [[f64; 3]; 3] {
            match kind {
                CondKind::Iso => [[v[0], 0.0, 0.0], [0.0, v[0], 0.0], [0.0, 0.0, v[0]]],
                CondKind::Ortho => [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]],
                CondKind::Aniso => [[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]],
            }
        };
        Ok((m(&v), m(&d)))
    }

    /// ρ c(T) and ρ ∫c dT between two temperatures (internal energy per volume).
    pub fn capacity(&self) -> Option<&Table> {
        self.specific_heat.as_ref()
    }

    /// Latent heat per unit mass released at temperature `t` and its
    /// derivative: Σ L·f(T), where the fraction f goes linearly from 0 at solidus to 1
    /// at liquidus (step function if they coincide).
    pub fn latent_energy(&self, t: f64) -> (f64, f64) {
        let mut u = 0.0;
        let mut du = 0.0;
        for &(l, ts, tl) in &self.latent {
            if t >= tl {
                u += l;
            } else if t > ts {
                u += l * (t - ts) / (tl - ts);
                du += l / (tl - ts);
            }
        }
        (u, du)
    }

    /// Thermal strain α(T)(T − T0) − α(TI)(TI − T0) (definition with
    /// the reference temperature T0 = ZERO and the initial temperature TI).
    pub fn thermal_strain(&self, t: f64, ti: f64) -> f64 {
        match &self.expansion {
            Some((tab, zero)) => tab.value(t) * (t - zero) - tab.value(ti) * (ti - zero),
            None => 0.0,
        }
    }
}
