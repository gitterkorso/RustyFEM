//! Porvätskans egenskaper för kopplad porvattentryck–förskjutning (*SOILS):
//! *PERMEABILITY och *POROUS BULK MODULI.
//!
//! Beteendena påverkar inte spänningsuppdateringen (den effektiva spänningen
//! räknas av materialets egen modell, t.ex. *ELASTIC eller *MOHR COULOMB) och
//! förbrukas därför av en passiv post. Elementen i `elements/soils.rs` och
//! steg-typen i `steps/soils.rs` läser dem med `PoroProps::from_def`.
//!
//! Darcys lag: volymflödet per ytenhet är
//! f = −(k/γw)·(∂u_w/∂x − ρw g) där k är permeabiliteten (hastighetsenhet) och
//! γw = SPECIFIC= vätskans tunghet. Permeabiliteten får bero på portalet e
//! (flera datarader k, e; styckvis linjär interpolation, konstant utanför
//! tabellen). TYPE=ISOTROPIC (k), ORTHOTROPIC (k11, k22, k33) och
//! ANISOTROPIC (k11, k12, k22, k13, k23, k33).

use super::MaterialFactory;
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "porvätska (permeabilitet, kompressionsmoduler)",
    consumes: &["PERMEABILITY", "POROUS BULK MODULI"],
    priority: 0,
    build: None,
}];

/// Permeabilitet som funktion av portalet: (k-tensor 3×3 radvis, e).
#[derive(Debug, Clone, Default)]
pub struct Permeability {
    pub gamma_w: f64,
    pub table: Vec<([f64; 9], f64)>,
}

impl Permeability {
    /// Permeabiliteten varierar med portalet.
    pub fn varies(&self) -> bool {
        self.table.len() > 1
    }

    /// k och dk/de vid portalet `e`.
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

/// Porvätskans egenskaper för ett material.
#[derive(Debug, Clone, Default)]
pub struct PoroProps {
    pub perm: Permeability,
    /// Kornens och vätskans kompressionsmoduler (0 = inkompressibel).
    pub k_grain: f64,
    pub k_fluid: f64,
}

fn parse_perm(kw: &RawKeyword) -> Result<Permeability> {
    for p in ["DEPENDENCIES", "USER"] {
        if kw.has(p) {
            return kw.err(format!("{p} stöds inte ännu för *PERMEABILITY"));
        }
    }
    let gamma_w = match kw.param("SPECIFIC") {
        Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: SPECIFIC={s} är inget tal", kw.location())))?,
        None => return kw.err("SPECIFIC= (vätskans tunghet γw) saknas"),
    };
    if !(gamma_w > 0.0) {
        return kw.err("SPECIFIC= (γw) ska vara positivt");
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
                    return kw.err_at(d.line, "temperaturberoende permeabilitet stöds inte ännu");
                }
                ([g(0), 0.0, 0.0, 0.0, g(0), 0.0, 0.0, 0.0, g(0)], g(1))
            }
            "ORTHO" | "ORTHOTROPIC" => ([g(0), 0.0, 0.0, 0.0, g(1), 0.0, 0.0, 0.0, g(2)], g(3)),
            "ANISO" | "ANISOTROPIC" => {
                // k11, k12, k22, k13, k23, k33, e
                let (k11, k12, k22, k13, k23, k33) = (g(0), g(1), g(2), g(3), g(4), g(5));
                ([k11, k12, k13, k12, k22, k23, k13, k23, k33], g(6))
            }
            other => return kw.err(format!("TYPE={other} stöds inte ännu för *PERMEABILITY (bara ISO, ORTHO, ANISO)")),
        };
        table.push((k, e));
    }
    if table.is_empty() {
        return kw.err("permeabiliteten saknas");
    }
    table.sort_by(|a, b| a.1.total_cmp(&b.1));
    Ok(Permeability { gamma_w, table })
}

impl PoroProps {
    /// Porvätskans egenskaper ur materialet, eller `None` om *PERMEABILITY saknas.
    pub fn from_def(def: &MaterialDef) -> Result<Option<PoroProps>> {
        let mut perm = None;
        for kw in def.behaviors.iter().filter(|b| b.name == "PERMEABILITY") {
            let ty = kw.param_upper("TYPE").unwrap_or_default();
            match ty.as_str() {
                "SATURATION" | "VELOCITY" => {
                    return kw.err(format!("*PERMEABILITY, TYPE={ty} stöds inte ännu (bara helt mättad jord utan Forchheimer-term)"));
                }
                _ => {
                    if perm.is_some() {
                        return kw.err(format!("materialet {} har redan *PERMEABILITY", def.name));
                    }
                    perm = Some(parse_perm(kw)?);
                }
            }
        }
        let Some(perm) = perm else { return Ok(None) };
        let mut p = PoroProps { perm, ..Default::default() };
        if let Some(kw) = def.behavior("POROUS BULK MODULI") {
            if kw.has("DEPENDENCIES") || kw.data.len() > 1 {
                return kw.err("temperaturberoende *POROUS BULK MODULI stöds inte ännu");
            }
            let v = kw.first_floats()?;
            p.k_grain = v.first().copied().unwrap_or(0.0);
            p.k_fluid = v.get(1).copied().unwrap_or(0.0);
            if p.k_grain < 0.0 || p.k_fluid < 0.0 {
                return kw.err("kompressionsmodulerna ska vara positiva (eller tomma för inkompressibel)");
            }
        }
        Ok(Some(p))
    }

    /// Som `from_def` men fel om *PERMEABILITY saknas.
    pub fn require(def: &MaterialDef) -> Result<PoroProps> {
        match Self::from_def(def)? {
            Some(p) => Ok(p),
            None => fail(format!("material {}: *PERMEABILITY saknas (krävs för porvattentryckselement)", def.name)),
        }
    }
}
