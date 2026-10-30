//! Area materials5: strain-rate- and temperature-dependent tables for
//! *DAMAGE INITIATION, CRITERION=DUCTILE and SHEAR (columns ε̄D, x, ε̄̇pl, θ
//! with x = η or θs), used by `ductile_damage.rs`.
//!
//! The lines are grouped by (θ, ε̄̇pl); within a group the interpolation is
//! linear in x, between the groups linear in ε̄̇pl and then in θ (ε̄D linear in
//! ε̄̇pl), with constant values outside the table.

use super::MaterialFactory;
use crate::fem::model::RawKeyword;
use crate::model::Result;

/// The module has no material factories of its own.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Table ε̄D(x, ε̄̇pl, θ).
#[derive(Debug, Clone)]
pub struct DepTable {
    /// Per temperature (ascending): per rate (ascending): the curve (x, ε̄D).
    temps: Vec<(f64, Vec<(f64, Vec<(f64, f64)>)>)>,
}

impl DepTable {
    /// Parses the table; `None` if it does not depend on rate or temperature.
    pub fn parse(kw: &RawKeyword) -> Result<Option<DepTable>> {
        let mut rows = Vec::new();
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() > 4 && v[4..].iter().any(|&x| x != 0.0) {
                return kw.err_at(d.line, "field-dependent initiation is not supported yet");
            }
            let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
            if !(g(0) > 0.0) {
                return kw.err_at(d.line, "the initiation strain must be positive");
            }
            rows.push((d.line, g(0), g(1), g(2), g(3)));
        }
        if !rows.iter().any(|r| r.3 != 0.0 || r.4 != 0.0) {
            return Ok(None);
        }
        let mut temps: Vec<(f64, Vec<(f64, Vec<(f64, f64)>)>)> = Vec::new();
        for &(line, e, x, rate, th) in &rows {
            if temps.last().map_or(true, |t| t.0 != th) {
                if temps.iter().any(|t| t.0 >= th) {
                    return kw.err_at(line, "the temperature must increase between the groups");
                }
                temps.push((th, Vec::new()));
            }
            let rg = &mut temps.last_mut().unwrap().1;
            if rg.last().map_or(true, |r| r.0 != rate) {
                if rg.iter().any(|r| r.0 >= rate) {
                    return kw.err_at(line, "the strain rate must increase within a temperature");
                }
                rg.push((rate, Vec::new()));
            }
            let curve = &mut rg.last_mut().unwrap().1;
            if curve.last().is_some_and(|&(xl, _)| !(x > xl)) {
                return kw.err_at(line, "the triaxiality (or θs) must increase within a group");
            }
            curve.push((x, e));
        }
        Ok(Some(DepTable { temps }))
    }

    /// ε̄D at (x, ε̄̇pl, θ).
    pub fn eval(&self, x: f64, rate: f64, theta: f64) -> f64 {
        let in_rate = |rg: &Vec<(f64, Vec<(f64, f64)>)>| -> f64 { lin(rg.iter().map(|(r, c)| (*r, super::principal::interp(c, x).0)), rate) };
        lin(self.temps.iter().map(|(t, rg)| (*t, in_rate(rg))), theta)
    }
}

/// Linear interpolation at points (ascending x), constant outside.
fn lin(pts: impl Iterator<Item = (f64, f64)>, x: f64) -> f64 {
    let p: Vec<(f64, f64)> = pts.collect();
    super::principal::interp(&p, x).0
}
