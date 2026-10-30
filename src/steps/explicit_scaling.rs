//! Mass scaling in explicit dynamics: *FIXED MASS SCALING and
//! *VARIABLE MASS SCALING (step data).
//!
//! The scaling multiplies the element's density by fₑ ≥ 0, which changes
//! the element's stability limit to Δtₑ √fₑ:
//!
//! - `FACTOR=f`: fₑ = f.
//! - `DT=d, TYPE=BELOW MIN` (default when DT is given): fₑ = (d/Δtₑ)² for elements
//!   with Δtₑ < d, otherwise 1.
//! - `DT=d, TYPE=UNIFORM`: the same factor (d/min Δtₑ)² for all elements in
//!   ELSET if min Δtₑ < d.
//! - `DT=d, TYPE=SET EQUAL DT`: fₑ = (d/Δtₑ)² for all elements (also < 1).
//! - FACTOR and DT together: the factor first, then DT.
//!
//! *FIXED MASS SCALING applies from the start of the step. *VARIABLE MASS SCALING
//! is recomputed at the start of the step and every n-th increment (FREQUENCY=n) or at
//! NUMBER INTERVAL=n evenly spaced times, with the current geometry. Without
//! ELSET the scaling applies to all elements. Δtₑ is the damped element-by-element
//! limit (with bulk viscosity), before SCALE FACTOR.

use crate::fem::assembly::Prepared;
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::model::Result;

/// The scaling gives Δtₑ ≥ d even after rounding (so that Δt becomes exactly d
/// and output at d·n ends up in the right increment).
const ROUND: f64 = 1.0 + 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    BelowMin,
    Uniform,
    SetEqual,
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub elems: Option<Vec<usize>>,
    pub factor: Option<f64>,
    pub dt: Option<(f64, Kind)>,
    pub variable: bool,
    /// Every n-th increment (FREQUENCY) or number of intervals (NUMBER INTERVAL).
    pub frequency: Option<usize>,
    pub intervals: Option<usize>,
}

fn rule(p: &Prepared, kw: &RawKeyword, variable: bool) -> Result<Rule> {
    let elems = match kw.param("ELSET") {
        Some(s) => Some(crate::interactions::geom::elem_ref(p.model, kw, s.trim())?),
        None => None,
    };
    let factor = kw.param("FACTOR").and_then(parse_f64);
    let kind = match kw.param_upper("TYPE").as_deref() {
        None | Some("BELOW MIN") => Kind::BelowMin,
        Some("UNIFORM") => Kind::Uniform,
        Some("SET EQUAL DT") => Kind::SetEqual,
        Some(o) => return kw.err(format!("TYPE={o} is not supported (BELOW MIN, UNIFORM, SET EQUAL DT)")),
    };
    let dt = kw.param("DT").and_then(parse_f64).map(|d| (d, kind));
    if factor.is_none() && dt.is_none() {
        return kw.err("give FACTOR= or DT=");
    }
    if kw.has("TARGET") || kw.has("CROSS SECTION") || kw.has("DIRECTION") {
        return kw.err("mass scaling for rolling (TYPE=ROLLING) is not supported");
    }
    let frequency = kw.param("FREQUENCY").and_then(parse_f64).map(|n| (n.round() as usize).max(1));
    let intervals = kw.param("NUMBER INTERVAL").and_then(parse_f64).map(|n| (n.round() as usize).max(1));
    if variable && frequency.is_none() && intervals.is_none() {
        return kw.err("*VARIABLE MASS SCALING requires FREQUENCY= or NUMBER INTERVAL=");
    }
    Ok(Rule { elems, factor, dt, variable, frequency, intervals })
}

/// The step's mass scaling rules in order.
pub fn rules(p: &Prepared, step: &StepDef) -> Result<Vec<Rule>> {
    let mut out = Vec::new();
    for kw in &step.extra {
        match kw.name.as_str() {
            "FIXED MASS SCALING" => out.push(rule(p, kw, false)?),
            "VARIABLE MASS SCALING" => out.push(rule(p, kw, true)?),
            _ => {}
        }
    }
    Ok(out)
}

/// Applies the rules (only the variable ones if `only_variable`) to the factors
/// `f`, given the unscaled element-by-element limit `dt0[e]` (infinite for elements
/// without mass).
pub fn apply(rules: &[Rule], only_variable: bool, dt0: &[f64], f: &mut [f64]) {
    for r in rules.iter().filter(|r| r.variable || !only_variable) {
        let all: Vec<usize>;
        let set: &[usize] = match &r.elems {
            Some(v) => v,
            None => {
                all = (0..dt0.len()).collect();
                &all
            }
        };
        if let Some(c) = r.factor {
            for &e in set {
                f[e] = c;
            }
        }
        if let Some((d, kind)) = r.dt {
            let cur = |e: usize, f: &[f64]| dt0[e] * f[e].sqrt();
            match kind {
                Kind::BelowMin => {
                    for &e in set {
                        let t = cur(e, f);
                        if t.is_finite() && t < d {
                            f[e] *= (d / t).powi(2) * ROUND;
                        }
                    }
                }
                Kind::SetEqual => {
                    for &e in set {
                        let t = cur(e, f);
                        if t.is_finite() && t > 0.0 {
                            f[e] *= (d / t).powi(2) * ROUND;
                        }
                    }
                }
                Kind::Uniform => {
                    let tmin = set.iter().map(|&e| cur(e, f)).fold(f64::INFINITY, f64::min);
                    if tmin.is_finite() && tmin < d {
                        let s = (d / tmin).powi(2) * ROUND;
                        for &e in set {
                            f[e] *= s;
                        }
                    }
                }
            }
        }
    }
}
