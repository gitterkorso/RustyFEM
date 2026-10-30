//! Gasket behavior (*GASKET BEHAVIOR) for the gasket elements in
//! `elements/gasket.rs`.
//!
//! The material is used with the gasket's own component order (as in
//! the output for gaskets): the strain is
//! [closure, ε₂₂, ε₃₃, γ₁₂, γ₁₃, ε₂₃] and the stress
//! [pressure, S22, S33, S12, S13, S23]:
//!
//! - Thickness direction (*GASKET THICKNESS BEHAVIOR): pressure versus closure from
//!   the loading curve (piecewise linear, the last slope is extrapolated).
//!   TYPE=ELASTIC-PLASTIC: elastic with the first slope k₀ up to
//!   the yield point (the first point where the slope drops more than YIELD ONSET,
//!   default 10 %, below k₀), then the loading curve; unloading and
//!   reloading with k₀ from the plastic closure; after the last point of the curve
//!   it continues with k₀ (DAMAGE: the last slope). TYPE=DAMAGE without
//!   unloading curves: unloading along the loading curve (this is done when
//!   no unloading curve is given). Tension: TENSILE STIFFNESS FACTOR (default
//!   10⁻³) times k₀.
//! - Membrane (*GASKET ELASTICITY, COMPONENT=MEMBRANE): plane stress with E, ν.
//! - Transverse shear (*GASKET ELASTICITY, COMPONENT=TRANSVERSE SHEAR):
//!   S12 = G·γ₁₂ with γ as relative displacement.

use super::api::{MatInput, MatOutput, MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub const THICKNESS: &str = "GASKET THICKNESS BEHAVIOR";
pub const THICKNESS_UNLOADING: &str = "GASKET THICKNESS BEHAVIOR UNLOADING";
pub const MEMBRANE: &str = "GASKET ELASTICITY MEMBRANE";
pub const SHEAR: &str = "GASKET ELASTICITY TRANSVERSE SHEAR";
/// Marker that *GASKET BEHAVIOR adds to the material.
pub const MARKER: &str = "GASKET BEHAVIOR";

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "gasket",
    consumes: &[MARKER, THICKNESS, THICKNESS_UNLOADING, MEMBRANE, SHEAR],
    priority: 50,
    build: Some(build),
}];

/// The model's name (the elements check that the material is a gasket).
pub const NAME: &str = "GASKET";

#[derive(Debug, Clone)]
pub struct Gasket {
    /// The loading curve (closure, pressure), ascending closure, starts at (0, 0).
    curve: Vec<(f64, f64)>,
    plastic: bool,
    k0: f64,
    /// The closure at which yielding starts (ELASTIC-PLASTIC).
    c_yield: f64,
    tension: f64,
    membrane: Option<(f64, f64)>,
    shear: Option<f64>,
}

fn floats(kw: &RawKeyword) -> Result<Vec<Vec<f64>>> {
    kw.records().iter().map(|d| kw.floats(d)).collect()
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if def.behavior(MARKER).is_none() {
        return Ok(None);
    }
    let Some(tb) = def.behavior(THICKNESS) else {
        return fail(format!("the gasket behavior {} lacks *GASKET THICKNESS BEHAVIOR", def.name));
    };
    if def.behavior(THICKNESS_UNLOADING).is_some() {
        return tb.err("unloading curves (DIRECTION=UNLOADING) are not supported yet");
    }
    for p in ["DEPENDENCIES", "TEMPERATURE"] {
        if tb.has(p) {
            return tb.err(format!("{p}= is not supported yet for gaskets"));
        }
    }
    let plastic = match tb.param_upper("TYPE").as_deref() {
        None | Some("DAMAGE") => false,
        Some("ELASTIC-PLASTIC") => true,
        Some(t) => return tb.err(format!("TYPE={t} is not supported (DAMAGE, ELASTIC-PLASTIC)")),
    };
    if let Some(m) = tb.param_upper("YIELD ONSET METHOD") {
        if m != "RELATIVE SLOPE DROP" {
            return tb.err(format!("YIELD ONSET METHOD={m} is not supported yet"));
        }
    }
    let onset = tb.param("YIELD ONSET").and_then(crate::fem::model::parse_f64).unwrap_or(0.1);
    let factor = tb.param("TENSILE STIFFNESS FACTOR").and_then(crate::fem::model::parse_f64).unwrap_or(1e-3);
    let mut curve: Vec<(f64, f64)> = Vec::new();
    for v in floats(tb)? {
        if v.len() < 2 {
            return tb.err("each data line must have pressure and closure");
        }
        curve.push((v[1], v[0]));
    }
    if curve.first().is_none_or(|p| p.0 != 0.0 || p.1 != 0.0) {
        curve.insert(0, (0.0, 0.0));
    }
    for w in curve.windows(2) {
        if w[1].0 <= w[0].0 {
            return tb.err("the closure must increase in the loading curve");
        }
    }
    if curve.len() < 2 {
        return tb.err("the loading curve needs at least one point besides (0, 0)");
    }
    let k0 = curve[1].1 / curve[1].0;
    if k0 <= 0.0 {
        return tb.err("the first slope of the loading curve must be positive");
    }
    let mut c_yield = f64::INFINITY;
    for i in 1..curve.len() - 1 {
        let s = (curve[i + 1].1 - curve[i].1) / (curve[i + 1].0 - curve[i].0);
        if s < (1.0 - onset) * k0 {
            c_yield = curve[i].0;
            break;
        }
    }
    let membrane = match def.behavior(MEMBRANE) {
        Some(kw) => {
            let v = kw.first_floats()?;
            Some((v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0)))
        }
        None => None,
    };
    let shear = match def.behavior(SHEAR) {
        Some(kw) => Some(kw.first_floats()?.first().copied().unwrap_or(0.0)),
        None => None,
    };
    Ok(Some(Arc::new(Gasket { curve, plastic, k0, c_yield, tension: factor * k0, membrane, shear })))
}

impl Gasket {
    /// The loading curve and its slope at the closure c ≥ 0.
    fn loading(&self, c: f64) -> (f64, f64) {
        let n = self.curve.len();
        // ELASTIC-PLASTIC: after the last point of the curve it continues with k₀
        // DAMAGE: with the last slope
        let last = self.curve[n - 1];
        if self.plastic && c > last.0 {
            return (last.1 + self.k0 * (c - last.0), self.k0);
        }
        let mut i = 0;
        while i + 2 < n && c > self.curve[i + 1].0 {
            i += 1;
        }
        let (a, b) = (self.curve[i], self.curve[i + 1]);
        let s = (b.1 - a.1) / (b.0 - a.0);
        (a.1 + s * (c - a.0), s)
    }

    /// Pressure, tangent and new plastic closure.
    pub fn thickness(&self, c: f64, cp_old: f64) -> (f64, f64, f64) {
        if !self.plastic {
            if c < 0.0 {
                return (self.tension * c, self.tension, 0.0);
            }
            let (p, s) = self.loading(c);
            return (p, s, 0.0);
        }
        if c < cp_old {
            return (self.tension * (c - cp_old), self.tension, cp_old);
        }
        let p = self.k0 * (c - cp_old);
        if c > self.c_yield {
            let (f, s) = self.loading(c);
            if p > f {
                return (f, s, c - f / self.k0);
            }
        }
        (p, self.k0, cp_old)
    }
}

impl MaterialModel for Gasket {
    fn name(&self) -> &str {
        NAME
    }

    fn is_linear(&self) -> bool {
        false
    }

    fn n_state(&self) -> usize {
        1
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let e = input.strain;
        if e.len() != 6 {
            return fail("the gasket behavior can only be used by gasket elements (GK*)");
        }
        let (p, kc, cp) = self.thickness(e[0], state_old.first().copied().unwrap_or(0.0));
        if let Some(s) = state_new.first_mut() {
            *s = cp;
        }
        let mut stress = vec![0.0; 6];
        let mut d = vec![0.0; 36];
        stress[0] = p;
        d[0] = kc;
        if let Some((em, nu)) = self.membrane {
            let c = em / (1.0 - nu * nu);
            stress[1] = c * (e[1] + nu * e[2]);
            stress[2] = c * (e[2] + nu * e[1]);
            stress[5] = em / (2.0 * (1.0 + nu)) * e[5];
            d[7] = c;
            d[8] = c * nu;
            d[13] = c * nu;
            d[14] = c;
            d[35] = em / (2.0 * (1.0 + nu));
        }
        if let Some(g) = self.shear {
            stress[3] = g * e[3];
            stress[4] = g * e[4];
            d[21] = g;
            d[28] = g;
        }
        Ok(MatOutput { stress, tangent: d })
    }
}
