//! Område materials4: flytkurvor per töjningshastighet (*PLASTIC, RATE=) och
//! temperaturberoende Hill-förhållanden (*POTENTIAL med temperatur), för
//! `plasticity_hill.rs`.
//!
//! *PLASTIC, RATE=ε̄̇p upprepas för varje hastighet med början i den statiska
//! kurvan (RATE= saknas eller 0). Flytspänningen vid (ε̄p, ε̄̇p) interpoleras
//! linjärt i hastigheten mellan de två närmaste kurvorna och är konstant
//! utanför (lägsta/högsta hastighetens kurva). I returalgoritmen
//! är ε̄̇p = Δε̄p/Δt; utan tid (Δt = 0) används den högsta hastighetens kurva.
//!
//! *POTENTIAL med flera rader (R11, R22, R33, R12, R13, R23, T): Rij
//! interpoleras linjärt i temperaturen och är konstanta utanför tabellen.

use super::plasticity::Hardening;
use super::MaterialFactory;
use crate::fem::model::{parse_f64, MaterialDef, RawKeyword};
use crate::model::{FemError, Result};

/// Inga egna fabriker: används av `plasticity_hill.rs`.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Kurvorna (ε̄̇p, [(ε̄p, σ0)]) i stigande hastighet, eller tom lista utan RATE=.
pub fn rate_tables(def: &MaterialDef) -> Result<Vec<(f64, Vec<(f64, f64)>)>> {
    let ks: Vec<&RawKeyword> = def.behaviors.iter().filter(|b| b.name == "PLASTIC").collect();
    if !ks.iter().any(|k| k.has("RATE")) {
        return Ok(Vec::new());
    }
    let mut out: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
    for k in ks {
        if k.param_upper("HARDENING").is_some_and(|h| h != "ISOTROPIC") {
            return k.err("*PLASTIC, RATE= stöds bara med HARDENING=ISOTROPIC");
        }
        let rate = match k.param("RATE") {
            Some(v) => parse_f64(v).filter(|&r| r >= 0.0).ok_or_else(|| FemError(format!("{}: RATE={v} ska vara ett icke-negativt tal", k.location())))?,
            None => 0.0,
        };
        let tabs = super::thermal_elastic::yield_tables(k)?;
        if tabs.len() > 1 {
            return k.err("temperaturberoende flytkurvor tillsammans med RATE= stöds inte ännu");
        }
        if out.last().is_some_and(|(r, _)| rate <= *r) {
            return k.err("kurvorna för *PLASTIC, RATE= ska komma i stigande hastighet (den statiska först)");
        }
        out.push((rate, tabs[0].1.clone()));
    }
    Ok(out)
}

/// (σy, dσy/dΔλ) vid ε̄p = `p` med Δε̄p = `dl` under tiden `dt`.
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

/// Tabellen (T, Rij) ur *POTENTIAL med flera rader.
pub fn potential_table(pot: &RawKeyword) -> Result<Vec<(f64, [f64; 6])>> {
    let mut out: Vec<(f64, [f64; 6])> = Vec::new();
    for d in pot.records() {
        let v = pot.floats(&d)?;
        if v.len() < 6 {
            return pot.err_at(d.line, "*POTENTIAL kräver R11, R22, R33, R12, R13 och R23");
        }
        let r: [f64; 6] = std::array::from_fn(|i| v[i]);
        if r.iter().any(|&x| !(x > 0.0)) {
            return pot.err_at(d.line, "förhållandena Rij ska vara positiva");
        }
        let t = v.get(6).copied().unwrap_or(0.0);
        if out.last().is_some_and(|(tl, _)| t <= *tl) {
            return pot.err_at(d.line, "temperaturerna i *POTENTIAL ska öka");
        }
        out.push((t, r));
    }
    Ok(out)
}

/// Rij vid temperaturen `t`.
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
