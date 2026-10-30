//! Temperaturberoende isotrop elasticitet (*ELASTIC, TYPE=ISOTROPIC med
//! datarader E, ν, T).
//!
//! Spänningen räknas i totalform med modulerna vid aktuell temperatur:
//!
//!   σ = D(E(T), ν(T)) ε_mek
//!
//! där ε_mek är den mekaniska töjningen (termisk töjning dras av i
//! `thermal.rs` om *EXPANSION finns). E och ν interpoleras linjärt mellan
//! tabellens temperaturer och hålls konstanta utanför tabellen.
//! Modellen är olinjär i den meningen att D beror på temperaturen, så
//! statiska steg löses med Newton (tangenten är D(T)).
//!
//! Tabellhjälparna (`iso_table`, `interp_iso`, `yield_tables`,
//! `yield_at`) används också av von Mises-plasticiteten.

use super::anisotropic::reduce_6x6;
use super::plasticity::iso_d6;
use super::{matvec, MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "ELASTIC (isotrop, temperaturberoende)",
    consumes: &["ELASTIC"],
    priority: 2,
    build: Some(build),
}];

/// Är *ELASTIC (isotrop) temperaturberoende, dvs. har den mer än en datarad?
pub fn is_temperature_dependent(kw: &RawKeyword) -> bool {
    kw.data.len() > 1
}

/// Tabell (E, ν, T) i stigande temperatur från *ELASTIC, TYPE=ISOTROPIC.
/// En enda datarad ger en konstant post (temperaturen ignoreras).
pub fn iso_table(kw: &RawKeyword) -> Result<Vec<(f64, f64, f64)>> {
    if kw.has("DEPENDENCIES") {
        return kw.err("parametern DEPENDENCIES (fältberoende elasticitet) stöds inte ännu");
    }
    if kw.has("MODULI") {
        return kw.err("parametern MODULI stöds inte ännu");
    }
    let mut t: Vec<(f64, f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        let e = v.first().copied().unwrap_or(0.0);
        let nu = v.get(1).copied().unwrap_or(0.0);
        let temp = v.get(2).copied().unwrap_or(0.0);
        if v.len() > 3 && v[3..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "fältberoende elasticitet stöds inte ännu");
        }
        if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) {
            return kw.err_at(d.line, "E ska vara positivt och ν mellan −1 och 0,5");
        }
        if let Some(&(_, _, tl)) = t.last() {
            if !(temp > tl) {
                return kw.err_at(d.line, "temperaturerna i *ELASTIC ska öka");
            }
        }
        t.push((e, nu, temp));
    }
    if t.is_empty() {
        return kw.err("E saknas");
    }
    Ok(t)
}

/// Linjär interpolation (E, ν) vid temperaturen `t`, konstant utanför tabellen.
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

/// Flytspänningstabeller per temperatur från *PLASTIC (rader σ0, ε̄p, T):
/// [(T, [(ε̄p, σ0), ...]), ...] i stigande temperatur. Utan temperaturkolumn
/// (eller med samma temperatur på alla rader) blir det en enda tabell.
pub fn yield_tables(kw: &RawKeyword) -> Result<Vec<(f64, Vec<(f64, f64)>)>> {
    let mut out: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 3 && v[3..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "fältberoende hårdnande stöds inte ännu");
        }
        let s = v.first().copied().unwrap_or(0.0);
        let p = v.get(1).copied().unwrap_or(0.0);
        let temp = v.get(2).copied().unwrap_or(0.0);
        if !(s > 0.0) {
            return kw.err_at(d.line, "flytspänningen ska vara positiv");
        }
        let new_group = match out.last() {
            None => true,
            Some((tl, _)) => temp != *tl,
        };
        if new_group {
            if let Some((tl, _)) = out.last() {
                if !(temp > *tl) {
                    return kw.err_at(d.line, "temperaturerna i *PLASTIC ska öka");
                }
            }
            if p != 0.0 {
                return kw.err_at(d.line, "första punkten för varje temperatur ska ha plastisk töjning 0");
            }
            out.push((temp, vec![(p, s)]));
        } else {
            let tab = &mut out.last_mut().unwrap().1;
            if !(p > tab[tab.len() - 1].0) {
                return kw.err_at(d.line, "den plastiska töjningen ska öka i tabellen");
            }
            tab.push((p, s));
        }
    }
    if out.is_empty() {
        return kw.err("flytspänning saknas");
    }
    Ok(out)
}

/// σ0(ε̄p) i en tabell: linjär interpolation, konstant efter sista punkten.
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

/// Flytspänningstabellen vid temperaturen `t`: linjär interpolation mellan de
/// två närmaste temperaturernas kurvor vid samma ε̄p (konstant utanför
/// temperaturintervallet). Resultatet är styckvis linjärt med brytpunkter i
/// unionen av de två kurvornas ε̄p-värden, så det blir en vanlig tabell.
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

/// Isotrop elasticitet med temperaturberoende E och ν.
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
        "ELASTIC (temperaturberoende)"
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
