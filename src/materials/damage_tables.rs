//! Område materials5: töjningshastighets- och temperaturberoende tabeller för
//! *DAMAGE INITIATION, CRITERION=DUCTILE och SHEAR (kolumnerna ε̄D, x, ε̄̇pl, θ
//! med x = η eller θs), som `ductile_damage.rs` använder.
//!
//! Raderna grupperas efter (θ, ε̄̇pl); inom en grupp interpoleras
//! linjärt i x, mellan grupperna linjärt i ε̄̇pl och sedan i θ (ε̄D linjär i
//! ε̄̇pl), med konstanta värden utanför tabellen.

use super::MaterialFactory;
use crate::fem::model::RawKeyword;
use crate::model::Result;

/// Modulen har inga egna materialfabriker.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Tabell ε̄D(x, ε̄̇pl, θ).
#[derive(Debug, Clone)]
pub struct DepTable {
    /// Per temperatur (stigande): per hastighet (stigande): kurvan (x, ε̄D).
    temps: Vec<(f64, Vec<(f64, Vec<(f64, f64)>)>)>,
}

impl DepTable {
    /// Tolkar tabellen; `None` om den inte beror på hastighet eller temperatur.
    pub fn parse(kw: &RawKeyword) -> Result<Option<DepTable>> {
        let mut rows = Vec::new();
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() > 4 && v[4..].iter().any(|&x| x != 0.0) {
                return kw.err_at(d.line, "fältberoende initiering stöds inte ännu");
            }
            let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
            if !(g(0) > 0.0) {
                return kw.err_at(d.line, "initieringstöjningen ska vara positiv");
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
                    return kw.err_at(line, "temperaturen ska öka mellan grupperna");
                }
                temps.push((th, Vec::new()));
            }
            let rg = &mut temps.last_mut().unwrap().1;
            if rg.last().map_or(true, |r| r.0 != rate) {
                if rg.iter().any(|r| r.0 >= rate) {
                    return kw.err_at(line, "töjningshastigheten ska öka inom en temperatur");
                }
                rg.push((rate, Vec::new()));
            }
            let curve = &mut rg.last_mut().unwrap().1;
            if curve.last().is_some_and(|&(xl, _)| !(x > xl)) {
                return kw.err_at(line, "triaxialiteten (eller θs) ska öka inom en grupp");
            }
            curve.push((x, e));
        }
        Ok(Some(DepTable { temps }))
    }

    /// ε̄D vid (x, ε̄̇pl, θ).
    pub fn eval(&self, x: f64, rate: f64, theta: f64) -> f64 {
        let in_rate = |rg: &Vec<(f64, Vec<(f64, f64)>)>| -> f64 { lin(rg.iter().map(|(r, c)| (*r, super::principal::interp(c, x).0)), rate) };
        lin(self.temps.iter().map(|(t, rg)| (*t, in_rate(rg))), theta)
    }
}

/// Linjär interpolation i punkter (stigande x), konstant utanför.
fn lin(pts: impl Iterator<Item = (f64, f64)>, x: f64) -> f64 {
    let p: Vec<(f64, f64)> = pts.collect();
    super::principal::interp(&p, x).0
}
