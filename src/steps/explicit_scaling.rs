//! Massskalning i explicit dynamik: *FIXED MASS SCALING och
//! *VARIABLE MASS SCALING (stegdata).
//!
//! Skalningen multiplicerar elementets densitet med fₑ ≥ 0, vilket ändrar
//! elementets stabilitetsgräns till Δtₑ √fₑ:
//!
//! - `FACTOR=f`: fₑ = f.
//! - `DT=d, TYPE=BELOW MIN` (standard när DT ges): fₑ = (d/Δtₑ)² för element
//!   med Δtₑ < d, annars 1.
//! - `DT=d, TYPE=UNIFORM`: samma faktor (d/min Δtₑ)² för alla element i
//!   ELSET om min Δtₑ < d.
//! - `DT=d, TYPE=SET EQUAL DT`: fₑ = (d/Δtₑ)² för alla element (även < 1).
//! - FACTOR och DT tillsammans: faktorn först, sedan DT.
//!
//! *FIXED MASS SCALING gäller från stegets början. *VARIABLE MASS SCALING
//! räknas om i stegets början och var n:te inkrement (FREQUENCY=n) eller vid
//! NUMBER INTERVAL=n jämnt fördelade tider, med aktuell geometri. Utan
//! ELSET gäller skalningen alla element. Δtₑ är den dämpade elementvisa
//! gränsen (med volymviskositet), före SCALE FACTOR.

use crate::fem::assembly::Prepared;
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::model::Result;

/// Skalningen ger Δtₑ ≥ d även efter avrundning (så att Δt blir exakt d
/// och utdata vid d·n hamnar i rätt inkrement).
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
    /// Var n:te inkrement (FREQUENCY) eller antal intervall (NUMBER INTERVAL).
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
        Some(o) => return kw.err(format!("TYPE={o} stöds inte (BELOW MIN, UNIFORM, SET EQUAL DT)")),
    };
    let dt = kw.param("DT").and_then(parse_f64).map(|d| (d, kind));
    if factor.is_none() && dt.is_none() {
        return kw.err("ange FACTOR= eller DT=");
    }
    if kw.has("TARGET") || kw.has("CROSS SECTION") || kw.has("DIRECTION") {
        return kw.err("massskalning för valsning (TYPE=ROLLING) stöds inte");
    }
    let frequency = kw.param("FREQUENCY").and_then(parse_f64).map(|n| (n.round() as usize).max(1));
    let intervals = kw.param("NUMBER INTERVAL").and_then(parse_f64).map(|n| (n.round() as usize).max(1));
    if variable && frequency.is_none() && intervals.is_none() {
        return kw.err("*VARIABLE MASS SCALING kräver FREQUENCY= eller NUMBER INTERVAL=");
    }
    Ok(Rule { elems, factor, dt, variable, frequency, intervals })
}

/// Stegets massskalningsregler i ordning.
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

/// Tillämpar reglerna (bara de variabla om `only_variable`) på faktorerna
/// `f`, givet den oskalade elementvisa gränsen `dt0[e]` (oändlig för element
/// utan massa).
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
