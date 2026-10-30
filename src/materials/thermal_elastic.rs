//! Temperature-dependent isotropic elasticity (*ELASTIC, TYPE=ISOTROPIC with
//! data lines E, ν, T).
//!
//! The stress is computed in total form with the moduli at the current temperature:
//!
//!   σ = D(E(T), ν(T)) ε_mech
//!
//! where ε_mech is the mechanical strain (thermal strain is subtracted in
//! `thermal.rs` if *EXPANSION is present). E and ν are interpolated linearly between
//! the table's temperatures and held constant outside the table.
//! The model is nonlinear in the sense that D depends on the temperature, so
//! static steps are solved with Newton (the tangent is D(T)).
//!
//! The table helpers (`iso_table`, `interp_iso`, `yield_tables`,
//! `yield_at`) are also used by von Mises plasticity.

use super::anisotropic::reduce_6x6;
use super::plasticity::iso_d6;
use super::{matvec, MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "ELASTIC (isotropic, temperature-dependent)",
    consumes: &["ELASTIC"],
    priority: 2,
    build: Some(build),
}];

/// Is *ELASTIC (isotropic) temperature-dependent, i.e. does it have more than one data line?
pub fn is_temperature_dependent(kw: &RawKeyword) -> bool {
    kw.data.len() > 1
}

/// Table (E, ν, T) in ascending temperature from *ELASTIC, TYPE=ISOTROPIC.
/// A single data line gives a constant entry (the temperature is ignored).
pub fn iso_table(kw: &RawKeyword) -> Result<Vec<(f64, f64, f64)>> {
    if kw.has("DEPENDENCIES") {
        return kw.err("parameter DEPENDENCIES (field-dependent elasticity) is not supported yet");
    }
    if kw.has("MODULI") {
        return kw.err("parameter MODULI is not supported yet");
    }
    let mut t: Vec<(f64, f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        let e = v.first().copied().unwrap_or(0.0);
        let nu = v.get(1).copied().unwrap_or(0.0);
        let temp = v.get(2).copied().unwrap_or(0.0);
        if v.len() > 3 && v[3..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "field-dependent elasticity is not supported yet");
        }
        if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) {
            return kw.err_at(d.line, "E must be positive and ν between −1 and 0.5");
        }
        if let Some(&(_, _, tl)) = t.last() {
            if !(temp > tl) {
                return kw.err_at(d.line, "the temperatures in *ELASTIC must increase");
            }
        }
        t.push((e, nu, temp));
    }
    if t.is_empty() {
        return kw.err("E is missing");
    }
    Ok(t)
}

/// Linear interpolation (E, ν) at temperature `t`, constant outside the table.
pub fn interp_iso(tab: &[(f64, f64, f64)], t: f64) -> (f64, f64) {
    if tab.len() == 1 || t <= tab[0].2 {
        return (tab[0].0, tab[0].1);
    }
    for w in tab.windows(2) {
        let ((e0, n0, t0), (e1, n1, t1)) = (w[0], w[1]);
        if t < t1 {
            let s = (t - t0) / (t1 - t0);
            return (e0 + s * (e1 - e0), n0 + s * (n1 - n0));
        }
    }
    let l = tab[tab.len() - 1];
    (l.0, l.1)
}

/// Yield stress tables per temperature from *PLASTIC (lines σ0, ε̄p, T):
/// [(T, [(ε̄p, σ0), ...]), ...] in ascending temperature. Without a temperature column
/// (or with the same temperature on all lines) there is a single table.
pub fn yield_tables(kw: &RawKeyword) -> Result<Vec<(f64, Vec<(f64, f64)>)>> {
    let mut out: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 3 && v[3..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "field-dependent hardening is not supported yet");
        }
        let s = v.first().copied().unwrap_or(0.0);
        let p = v.get(1).copied().unwrap_or(0.0);
        let temp = v.get(2).copied().unwrap_or(0.0);
        if !(s > 0.0) {
            return kw.err_at(d.line, "the yield stress must be positive");
        }
        let new_group = match out.last() {
            None => true,
            Some((tl, _)) => temp != *tl,
        };
        if new_group {
            if let Some((tl, _)) = out.last() {
                if !(temp > *tl) {
                    return kw.err_at(d.line, "the temperatures in *PLASTIC must increase");
                }
            }
            if p != 0.0 {
                return kw.err_at(d.line, "the first point for each temperature must have plastic strain 0");
            }
            out.push((temp, vec![(p, s)]));
        } else {
            let tab = &mut out.last_mut().unwrap().1;
            if !(p > tab[tab.len() - 1].0) {
                return kw.err_at(d.line, "the plastic strain must increase in the table");
            }
            tab.push((p, s));
        }
    }
    if out.is_empty() {
        return kw.err("yield stress is missing");
    }
    Ok(out)
}

/// σ0(ε̄p) in a table: linear interpolation, constant after the last point.
fn table_eval(t: &[(f64, f64)], p: f64) -> f64 {
    if t.len() == 1 || p <= t[0].0 {
        return t[0].1;
    }
    for w in t.windows(2) {
        if p < w[1].0 {
            return w[0].1 + (w[1].1 - w[0].1) / (w[1].0 - w[0].0) * (p - w[0].0);
        }
    }
    t[t.len() - 1].1
}

/// The yield stress table at temperature `t`: linear interpolation between the
/// curves of the two nearest temperatures at the same ε̄p (constant outside the
/// temperature range). The result is piecewise linear with breakpoints at
/// the union of the two curves' ε̄p values, so it becomes an ordinary table.
pub fn yield_at(tabs: &[(f64, Vec<(f64, f64)>)], t: f64) -> Vec<(f64, f64)> {
    if tabs.len() == 1 || t <= tabs[0].0 {
        return tabs[0].1.clone();
    }
    let last = &tabs[tabs.len() - 1];
    if t >= last.0 {
        return last.1.clone();
    }
    let k = tabs.windows(2).position(|w| t < w[1].0).unwrap_or(tabs.len() - 2);
    let (t0, a) = (&tabs[k].0, &tabs[k].1);
    let (t1, b) = (&tabs[k + 1].0, &tabs[k + 1].1);
    let s = (t - t0) / (t1 - t0);
    let mut ps: Vec<f64> = a.iter().chain(b.iter()).map(|x| x.0).collect();
    ps.sort_by(f64::total_cmp);
    ps.dedup();
    ps.into_iter().map(|p| (p, (1.0 - s) * table_eval(a, p) + s * table_eval(b, p))).collect()
}

/// Isotropic elasticity with temperature-dependent E and ν.
#[derive(Debug, Clone)]
pub struct ThermoElastic {
    pub table: Vec<(f64, f64, f64)>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("ELASTIC") else { return Ok(None) };
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" || !is_temperature_dependent(kw) {
        return Ok(None);
    }
    Ok(Some(Arc::new(ThermoElastic { table: iso_table(kw)? })))
}

impl MaterialModel for ThermoElastic {
    fn name(&self) -> &str {
        "ELASTIC (temperature-dependent)"
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.table[0].0, self.table[0].1))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let (e, nu) = interp_iso(&self.table, input.temperature);
        let d = reduce_6x6(&iso_d6(e, nu), input.mode);
        Ok(MatOutput { stress: matvec(&d, input.strain), tangent: d })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let (e, nu) = (self.table[0].0, self.table[0].1);
        Ok(reduce_6x6(&iso_d6(e, nu), mode))
    }
}
