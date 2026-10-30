//! Feature area materials4: yield curves per strain rate (*PLASTIC, RATE=) and
//! temperature-dependent Hill ratios (*POTENTIAL with temperature), for
//! `plasticity_hill.rs`.
//!
//! *PLASTIC, RATE=ε̄̇p is repeated for each rate starting with the static
//! curve (RATE= missing or 0). The yield stress at (ε̄p, ε̄̇p) is interpolated
//! linearly in the rate between the two nearest curves and is constant
//! outside (the lowest/highest rate's curve). In the return algorithm
//! ε̄̇p = Δε̄p/Δt; without time (Δt = 0) the highest rate's curve is used.
//!
//! *POTENTIAL with several lines (R11, R22, R33, R12, R13, R23, T): Rij
//! are interpolated linearly in the temperature and are constant outside the table.

use super::plasticity::Hardening;
use super::MaterialFactory;
use crate::fem::model::{parse_f64, MaterialDef, RawKeyword};
use crate::model::{FemError, Result};

/// No factories of its own: used by `plasticity_hill.rs`.
pub static FACTORIES: &[MaterialFactory] = &[];

/// The curves (ε̄̇p, [(ε̄p, σ0)]) in increasing rate, or an empty list without RATE=.
pub fn rate_tables(def: &MaterialDef) -> Result<Vec<(f64, Vec<(f64, f64)>)>> {
    let ks: Vec<&RawKeyword> = def.behaviors.iter().filter(|b| b.name == "PLASTIC").collect();
    if !ks.iter().any(|k| k.has("RATE")) {
        return Ok(Vec::new());
    }
    let mut out: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
    for k in ks {
        if k.param_upper("HARDENING").is_some_and(|h| h != "ISOTROPIC") {
            return k.err("*PLASTIC, RATE= is only supported with HARDENING=ISOTROPIC");
        }
        let rate = match k.param("RATE") {
            Some(v) => parse_f64(v).filter(|&r| r >= 0.0).ok_or_else(|| FemError(format!("{}: RATE={v} must be a non-negative number", k.location())))?,
            None => 0.0,
        };
        let tabs = super::thermal_elastic::yield_tables(k)?;
        if tabs.len() > 1 {
            return k.err("temperature-dependent yield curves together with RATE= are not supported yet");
        }
        if out.last().is_some_and(|(r, _)| rate <= *r) {
            return k.err("the curves for *PLASTIC, RATE= must come in increasing rate (the static one first)");
        }
        out.push((rate, tabs[0].1.clone()));
    }
    Ok(out)
}

/// (σy, dσy/dΔλ) at ε̄p = `p` with Δε̄p = `dl` over the time `dt`.
pub fn yield_stress(tabs: &[(f64, Vec<(f64, f64)>)], p: f64, dl: f64, dt: f64) -> (f64, f64) {
    let eval = |k: usize| Hardening::Table(tabs[k].1.clone()).eval(p);
    let rate = if dt > 0.0 { dl / dt } else { f64::INFINITY };
    let n = tabs.len();
    if n == 1 || rate <= tabs[0].0 {
        return eval(0);
    }
    if rate >= tabs[n - 1].0 {
        return eval(n - 1);
    }
    let k = tabs.windows(2).position(|w| rate < w[1].0).unwrap_or(n - 2);
    let (r0, r1) = (tabs[k].0, tabs[k + 1].0);
    let w = (rate - r0) / (r1 - r0);
    let (s0, h0) = eval(k);
    let (s1, h1) = eval(k + 1);
    let s = (1.0 - w) * s0 + w * s1;
    // dσ/dΔλ = ∂σ/∂ε̄p + ∂σ/∂ε̄̇p / Δt
    let h = (1.0 - w) * h0 + w * h1 + (s1 - s0) / (r1 - r0) / dt;
    (s, h)
}

/// The table (T, Rij) from *POTENTIAL with several lines.
pub fn potential_table(pot: &RawKeyword) -> Result<Vec<(f64, [f64; 6])>> {
    let mut out: Vec<(f64, [f64; 6])> = Vec::new();
    for d in pot.records() {
        let v = pot.floats(&d)?;
        if v.len() < 6 {
            return pot.err_at(d.line, "*POTENTIAL requires R11, R22, R33, R12, R13 and R23");
        }
        let r: [f64; 6] = std::array::from_fn(|i| v[i]);
        if r.iter().any(|&x| !(x > 0.0)) {
            return pot.err_at(d.line, "the ratios Rij must be positive");
        }
        let t = v.get(6).copied().unwrap_or(0.0);
        if out.last().is_some_and(|(tl, _)| t <= *tl) {
            return pot.err_at(d.line, "the temperatures in *POTENTIAL must increase");
        }
        out.push((t, r));
    }
    Ok(out)
}

/// Rij at temperature `t`.
pub fn potential_at(tab: &[(f64, [f64; 6])], t: f64) -> [f64; 6] {
    let n = tab.len();
    if n == 1 || t <= tab[0].0 {
        return tab[0].1;
    }
    if t >= tab[n - 1].0 {
        return tab[n - 1].1;
    }
    let k = tab.windows(2).position(|w| t < w[1].0).unwrap_or(n - 2);
    let w = (t - tab[k].0) / (tab[k + 1].0 - tab[k].0);
    std::array::from_fn(|i| (1.0 - w) * tab[k].1[i] + w * tab[k + 1].1[i])
}
