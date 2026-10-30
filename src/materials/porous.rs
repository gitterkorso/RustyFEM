//! Pore fluid properties for coupled pore pressure–displacement (*SOILS):
//! *PERMEABILITY and *POROUS BULK MODULI.
//!
//! The behaviors do not affect the stress update (the effective stress
//! is computed by the material's own model, e.g. *ELASTIC or *MOHR COULOMB) and
//! are therefore consumed by a passive entry. The elements in `elements/soils.rs` and
//! the step type in `steps/soils.rs` read them with `PoroProps::from_def`.
//!
//! Darcy's law: the volume flux per unit area is
//! f = −(k/γw)·(∂u_w/∂x − ρw g) where k is the permeability (velocity units) and
//! γw = SPECIFIC= the specific weight of the fluid. The permeability may depend on the void ratio e
//! (several data lines k, e; piecewise linear interpolation, constant outside
//! the table). TYPE=ISOTROPIC (k), ORTHOTROPIC (k11, k22, k33) and
//! ANISOTROPIC (k11, k12, k22, k13, k23, k33).

use super::MaterialFactory;
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "pore fluid (permeability, bulk moduli)",
    consumes: &["PERMEABILITY", "POROUS BULK MODULI"],
    priority: 0,
    build: None,
}];

/// Permeability as a function of the void ratio: (k tensor 3×3 row-major, e).
#[derive(Debug, Clone, Default)]
pub struct Permeability {
    pub gamma_w: f64,
    pub table: Vec<([f64; 9], f64)>,
}

impl Permeability {
    /// The permeability varies with the void ratio.
    pub fn varies(&self) -> bool {
        self.table.len() > 1
    }

    /// k and dk/de at the void ratio `e`.
    pub fn at(&self, e: f64) -> ([f64; 9], [f64; 9]) {
        let t = &self.table;
        if t.len() == 1 || e <= t[0].1 {
            return (t[0].0, [0.0; 9]);
        }
        if e >= t[t.len() - 1].1 {
            return (t[t.len() - 1].0, [0.0; 9]);
        }
        for w in t.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if e <= b.1 {
                let d = b.1 - a.1;
                let s = (e - a.1) / d;
                let k = std::array::from_fn(|i| a.0[i] + s * (b.0[i] - a.0[i]));
                let dk = std::array::from_fn(|i| (b.0[i] - a.0[i]) / d);
                return (k, dk);
            }
        }
        (t[t.len() - 1].0, [0.0; 9])
    }
}

/// Pore fluid properties of a material.
#[derive(Debug, Clone, Default)]
pub struct PoroProps {
    pub perm: Permeability,
    /// Bulk moduli of the grains and the fluid (0 = incompressible).
    pub k_grain: f64,
    pub k_fluid: f64,
}

fn parse_perm(kw: &RawKeyword) -> Result<Permeability> {
    for p in ["DEPENDENCIES", "USER"] {
        if kw.has(p) {
            return kw.err(format!("{p} is not supported yet for *PERMEABILITY"));
        }
    }
    let gamma_w = match kw.param("SPECIFIC") {
        Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: SPECIFIC={s} is not a number", kw.location())))?,
        None => return kw.err("SPECIFIC= (specific weight of the fluid γw) missing"),
    };
    if !(gamma_w > 0.0) {
        return kw.err("SPECIFIC= (γw) must be positive");
    }
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    let mut table: Vec<([f64; 9], f64)> = Vec::new();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.is_empty() {
            continue;
        }
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let (k, e) = match ty.as_str() {
            "ISO" | "ISOTROPIC" => {
                if v.len() > 2 && g(2) != 0.0 {
                    return kw.err_at(d.line, "temperature-dependent permeability is not supported yet");
                }
                ([g(0), 0.0, 0.0, 0.0, g(0), 0.0, 0.0, 0.0, g(0)], g(1))
            }
            "ORTHO" | "ORTHOTROPIC" => ([g(0), 0.0, 0.0, 0.0, g(1), 0.0, 0.0, 0.0, g(2)], g(3)),
            "ANISO" | "ANISOTROPIC" => {
                // k11, k12, k22, k13, k23, k33, e
                let (k11, k12, k22, k13, k23, k33) = (g(0), g(1), g(2), g(3), g(4), g(5));
                ([k11, k12, k13, k12, k22, k23, k13, k23, k33], g(6))
            }
            other => return kw.err(format!("TYPE={other} is not supported yet for *PERMEABILITY (only ISO, ORTHO, ANISO)")),
        };
        table.push((k, e));
    }
    if table.is_empty() {
        return kw.err("the permeability is missing");
    }
    table.sort_by(|a, b| a.1.total_cmp(&b.1));
    Ok(Permeability { gamma_w, table })
}

impl PoroProps {
    /// Pore fluid properties from the material, or `None` if *PERMEABILITY is missing.
    pub fn from_def(def: &MaterialDef) -> Result<Option<PoroProps>> {
        let mut perm = None;
        for kw in def.behaviors.iter().filter(|b| b.name == "PERMEABILITY") {
            let ty = kw.param_upper("TYPE").unwrap_or_default();
            match ty.as_str() {
                "SATURATION" | "VELOCITY" => {
                    return kw.err(format!("*PERMEABILITY, TYPE={ty} is not supported yet (only fully saturated soil without a Forchheimer term)"));
                }
                _ => {
                    if perm.is_some() {
                        return kw.err(format!("the material {} already has *PERMEABILITY", def.name));
                    }
                    perm = Some(parse_perm(kw)?);
                }
            }
        }
        let Some(perm) = perm else { return Ok(None) };
        let mut p = PoroProps { perm, ..Default::default() };
        if let Some(kw) = def.behavior("POROUS BULK MODULI") {
            if kw.has("DEPENDENCIES") || kw.data.len() > 1 {
                return kw.err("temperature-dependent *POROUS BULK MODULI is not supported yet");
            }
            let v = kw.first_floats()?;
            p.k_grain = v.first().copied().unwrap_or(0.0);
            p.k_fluid = v.get(1).copied().unwrap_or(0.0);
            if p.k_grain < 0.0 || p.k_fluid < 0.0 {
                return kw.err("the bulk moduli must be positive (or empty for incompressible)");
            }
        }
        Ok(Some(p))
    }

    /// Like `from_def` but an error if *PERMEABILITY is missing.
    pub fn require(def: &MaterialDef) -> Result<PoroProps> {
        match Self::from_def(def)? {
            Some(p) => Ok(p),
            None => fail(format!("material {}: *PERMEABILITY missing (required for pore pressure elements)", def.name)),
        }
    }
}
