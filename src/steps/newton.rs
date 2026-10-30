//! Generell Newton–Raphson-driver med automatisk lastinkrementering.
//!
//! Används av alla steg som löser jämvikt inkrementellt (statik med olinjära
//! material, NLGEOM, kontakt, implicit dynamik). Problemet beskrivs av
//! `NonlinearSystem`: residual R = F_ext − F_int, tangent och uppdatering.
//!
//! Konvergenskontrollen (standardvärden motsvarar *CONTROLS utan ändringar), men kontrolleras för varje fält
//! (förskjutning, rotation, välvning, temperatur) för sig:
//! - residual: max |r| ≤ R_tol · q̃, där q̃ är en tidsmedelvärdesbildad typisk
//!   kraft (medelbelopp av inre och yttre krafter) över stegets inkrement;
//! - korrektion: max |c| ≤ C_tol · max |Δu| i inkrementet;
//! - linjärt inkrement: max |r| ≤ R_lin · q̃ godtas utan korrektionskontroll.
//!
//! Inkrementering: start med `initial` (andel av steget), halvering till en
//! fjärdedel vid divergens eller för många iterationer (högst `max_cutbacks`
//! gånger per inkrement), ökning med faktorn 1,5 efter två inkrement i följd
//! som konvergerat på högst `ig` iterationer. DIRECT ger fasta inkrement.

use crate::fem::dof::Field;
use crate::fem::model::RawKeyword;
use crate::model::{fail, Result};
use std::collections::BTreeMap;

/// Residual och typiska krafter per fält.
#[derive(Debug, Clone, Default)]
pub struct Residual {
    /// R = F_ext − F_int för varje ekvation.
    pub r: Vec<f64>,
    /// Typisk kraftstorlek per fält (för q̃).
    pub flux: BTreeMap<Field, f64>,
}

/// Ett olinjärt jämviktsproblem för ett step.
pub trait NonlinearSystem {
    fn n_eq(&self) -> usize;
    /// Fält för varje ekvation.
    fn eq_field(&self, eq: usize) -> Field;
    /// Problemet är linjärt (ett lösningssteg räcker).
    fn is_linear(&self) -> bool;
    /// Startar ett inkrement från stegtid `t0` till `t1` (andelar av steget, 0–1).
    /// Sätter föreskrivna värden vid `t1` och återställer obekanta till senast godkända.
    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()>;
    /// Residual vid aktuell lösning; tangenten assembleras om `tangent`.
    fn evaluate(&mut self, tangent: bool) -> Result<Residual>;
    /// Löser K c = r med senast assemblerade tangent.
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>>;
    /// u += c för de fria ekvationerna.
    fn apply(&mut self, c: &[f64]);
    /// Ändringen sedan inkrementets början för varje ekvation.
    fn increment(&self) -> Vec<f64>;
    /// Godkänner inkrementet.
    fn commit(&mut self) -> Result<()>;
}

/// Inkrement- och konvergensparametrar.
#[derive(Debug, Clone)]
pub struct Controls {
    /// Inkrementlängder som andel av stegets tid (0–1).
    pub initial: f64,
    pub min: f64,
    pub max: f64,
    /// Stegets tidslängd (för utskrift av tider).
    pub period: f64,
    pub max_increments: usize,
    /// Fasta inkrement (DIRECT).
    pub fixed: bool,
    pub r_tol: f64,
    pub c_tol: f64,
    pub r_linear: f64,
    pub max_iter: usize,
    /// Iteration från vilken divergens kontrolleras.
    pub i0: usize,
    pub max_cutbacks: usize,
    pub cutback: f64,
    pub growth: f64,
    pub ig: usize,
}

impl Default for Controls {
    fn default() -> Self {
        Controls {
            initial: 1.0,
            min: 1e-5,
            max: 1.0,
            period: 1.0,
            max_increments: 100,
            fixed: false,
            r_tol: 5e-3,
            c_tol: 1e-2,
            r_linear: 1e-8,
            max_iter: 16,
            i0: 4,
            max_cutbacks: 5,
            cutback: 0.25,
            growth: 1.5,
            ig: 4,
        }
    }
}

impl Controls {
    /// Läser *STATIC-datarad: initialt inkrement, stegtid, minsta, största.
    pub fn from_procedure(proc_kw: Option<&RawKeyword>, max_increments: usize) -> Result<Controls> {
        let mut c = Controls { max_increments: max_increments.max(1), ..Default::default() };
        let Some(kw) = proc_kw else { return Ok(c) };
        let v = kw.first_floats()?;
        let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        c.period = period;
        let frac = |x: f64| x / period;
        if let Some(&dt0) = v.first().filter(|&&x| x > 0.0) {
            c.initial = frac(dt0).min(1.0);
        }
        if let Some(&dmin) = v.get(2).filter(|&&x| x > 0.0) {
            c.min = frac(dmin);
        } else {
            c.min = (c.initial * 1e-5).min(1e-5);
        }
        if let Some(&dmax) = v.get(3).filter(|&&x| x > 0.0) {
            c.max = frac(dmax).min(1.0);
        }
        c.fixed = kw.has("DIRECT");
        Ok(c)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Report {
    pub increments: usize,
    pub iterations: usize,
    pub cutbacks: usize,
}

/// Löser ett linjärt problem i ett enda steg (t = 1) och godkänner det.
pub fn solve_linear<S: NonlinearSystem + ?Sized>(sys: &mut S) -> Result<Report> {
    sys.begin_increment(0.0, 1.0)?;
    let r = sys.evaluate(true)?;
    if sys.n_eq() > 0 {
        let c = sys.solve(&r.r)?;
        sys.apply(&c);
    }
    sys.evaluate(false)?;
    sys.commit()?;
    Ok(Report { increments: 1, iterations: 1, cutbacks: 0 })
}

fn max_abs_by_field<S: NonlinearSystem + ?Sized>(sys: &S, v: &[f64]) -> BTreeMap<Field, f64> {
    let mut m: BTreeMap<Field, f64> = BTreeMap::new();
    for (i, x) in v.iter().enumerate() {
        let e = m.entry(sys.eq_field(i)).or_default();
        *e = e.max(x.abs());
    }
    m
}

/// Kör steget med Newton–Raphson. `on_increment(sys, t)` anropas efter varje
/// godkänt inkrement (t = stegtidsandel), t.ex. för att spara en frame.
pub fn run<S: NonlinearSystem + ?Sized>(
    sys: &mut S,
    ctrl: &Controls,
    on_increment: &mut dyn FnMut(&mut S, f64) -> Result<()>,
) -> Result<Report> {
    if sys.is_linear() && ctrl.initial >= 1.0 {
        let rep = solve_linear(sys)?;
        crate::progress::emit(crate::progress::Event::Increment { fraction: 1.0, iterations: 1 })?;
        on_increment(sys, 1.0)?;
        return Ok(rep);
    }
    let mut rep = Report::default();
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
    let mut n_flux = 0usize;
    let mut quick = 0usize;
    let mut had_cutback = false;
    while t < 1.0 - 1e-12 {
        if rep.increments >= ctrl.max_increments {
            return fail(format!(
                "steget nådde största antal inkrement ({}) vid stegtid {:.4} – öka INC= på *STEP",
                ctrl.max_increments,
                t * ctrl.period
            ));
        }
        dt = dt.min(1.0 - t);
        let mut cutbacks = 0;
        let iters = loop {
            // användarrutiner: KINC och PNEWDT (user/context.rs)
            crate::user::context::begin_increment(rep.increments + 1);
            sys.begin_increment(t, t + dt)?;
            match iterate(sys, ctrl, &flux_sum, n_flux)? {
                Some((it, flux)) => {
                    for (f, q) in flux {
                        *flux_sum.entry(f).or_default() += q;
                    }
                    n_flux += 1;
                    break it;
                }
                None => {
                    if ctrl.fixed {
                        return fail(format!("inkrementet vid stegtid {:.4} konvergerade inte (DIRECT, fasta inkrement)", (t + dt) * ctrl.period));
                    }
                    cutbacks += 1;
                    rep.cutbacks += 1;
                    had_cutback = true;
                    // PNEWDT < 1 från en användarrutin: nytt inkrement PNEWDT · Δt
                    dt *= match crate::user::context::pnewdt().filter(|&p| p < 1.0) {
                        Some(p) => p.max(1e-3),
                        None => ctrl.cutback,
                    };
                    if cutbacks > ctrl.max_cutbacks || dt < ctrl.min {
                        return fail(format!(
                            "ingen konvergens vid stegtid {:.4}: inkrementet blev mindre än minimum ({:.3e}) eller för många halveringar",
                            t * ctrl.period,
                            ctrl.min * ctrl.period
                        ));
                    }
                }
            }
        };
        sys.commit()?;
        t += dt;
        rep.increments += 1;
        rep.iterations += iters;
        crate::progress::emit(crate::progress::Event::Increment { fraction: t, iterations: iters })?;
        on_increment(sys, t)?;
        if !ctrl.fixed {
            if iters <= ctrl.ig && cutbacks == 0 {
                quick += 1;
                if quick >= 2 {
                    dt = (dt * ctrl.growth).min(ctrl.max);
                    // inkrementet ökas så länge de två senaste
                    // inkrementen konvergerade på högst I_G iterationer
                    // (räknaren nollställs inte vid ökningen;
                    // tests/cases/rubber/c3d8h_block_plast). Har steget redan
                    // halverats behålls den försiktigare regeln (två nya
                    // inkrement före nästa ökning), eftersom svåra
                    // diskontinuitetsiterationer (kontakt) inte räknas separat.
                    if had_cutback {
                        quick = 0;
                    }
                }
            } else {
                quick = 0;
            }
            dt = dt.min(ctrl.max);
        }
    }
    store_time_average(&flux_sum, n_flux);
    Ok(rep)
}

// ------------------------------------------------------------ nollflöde
//
// Fältet α har nollflöde när q̄ ≤ ε·q̃ (ε = 1e-5), där
// q̃ är tidsmedelvärdet av q̄ över stegets inkrement med flöde, i stegets början
// värdet från föregående steg. Lösningen godtas då om r_max ≤ ε·q̃, annars om
// c_max ≤ C_ε·Δu_max (C_ε = 1e-3). Utan detta konvergerar inte ett steg där alla
// krafter går mot noll (t.ex. ett tomt steg efter avlastning till ett
// spänningsfritt läge): det relativa kriteriet jämför då avrundningsfel med
// avrundningsfel.

/// Kriteriet för nollflöde relativt tidsmedelvärdet (ε^α).
const ZERO_FLUX: f64 = 1e-5;
/// Korrektionskriteriet vid nollflöde (C_ε^α).
const ZERO_FLUX_CORRECTION: f64 = 1e-3;

thread_local! {
    /// q̃ per fält vid slutet av föregående steg (startvärde för nästa steg).
    static PREV_TIME_AVERAGE: std::cell::RefCell<BTreeMap<Field, f64>> = const { std::cell::RefCell::new(BTreeMap::new()) };
}

/// Glömmer q̃ från tidigare steg (anropas när en analys startar, `steps::run`).
pub fn reset_time_average() {
    PREV_TIME_AVERAGE.with(|p| p.borrow_mut().clear());
}

fn store_time_average(flux_sum: &BTreeMap<Field, f64>, n_flux: usize) {
    if n_flux == 0 {
        return;
    }
    PREV_TIME_AVERAGE.with(|p| {
        let mut p = p.borrow_mut();
        for (f, s) in flux_sum {
            if *s > 0.0 {
                p.insert(*f, s / n_flux as f64);
            }
        }
    });
}

/// `Some(konvergerat)` om fältet `f` har nollflöde i aktuell iteration, annars `None`.
fn zero_flux_check(
    f: &Field,
    r: f64,
    flux: &BTreeMap<Field, f64>,
    flux_sum: &BTreeMap<Field, f64>,
    n_flux: usize,
    last_c: &Option<BTreeMap<Field, f64>>,
    inc: &BTreeMap<Field, f64>,
) -> Option<bool> {
    // q̃ utan aktuellt inkrement: stegets tidigare inkrement, annars föregående steg
    let sum = flux_sum.get(f).copied().unwrap_or(0.0);
    let qt = if sum > 0.0 && n_flux > 0 { sum / n_flux as f64 } else { PREV_TIME_AVERAGE.with(|p| p.borrow().get(f).copied().unwrap_or(0.0)) };
    let qc = flux.get(f).copied().unwrap_or(0.0);
    if !(qt > 0.0 && qc <= ZERO_FLUX * qt) {
        return None;
    }
    if r <= ZERO_FLUX * qt {
        return Some(true);
    }
    Some(match last_c {
        Some(c) => {
            let cm = c.get(f).copied().unwrap_or(0.0);
            cm == 0.0 || cm <= ZERO_FLUX_CORRECTION * inc.get(f).copied().unwrap_or(0.0)
        }
        None => false,
    })
}

/// Itererar ett inkrement som redan startats med `begin_increment`, med
/// tidsmedelvärdet av flux från `n_flux` tidigare godkända inkrement
/// (`flux_sum` är summan). För steg som sköter inkrementeringen själva
/// (implicit dynamik, `dynamic_incr.rs`). Returvärde som `iterate`.
pub fn iterate_increment<S: NonlinearSystem + ?Sized>(
    sys: &mut S,
    ctrl: &Controls,
    flux_sum: &BTreeMap<Field, f64>,
    n_flux: usize,
) -> Result<Option<(usize, BTreeMap<Field, f64>)>> {
    crate::user::context::reset_pnewdt();
    iterate(sys, ctrl, flux_sum, n_flux)
}

/// Itererar ett inkrement. `Some((iterationer, flux))` vid konvergens, `None` om
/// inkrementet ska halveras.
fn iterate<S: NonlinearSystem + ?Sized>(sys: &mut S, ctrl: &Controls, flux_sum: &BTreeMap<Field, f64>, n_flux: usize) -> Result<Option<(usize, BTreeMap<Field, f64>)>> {
    let mut history: Vec<f64> = Vec::new();
    let mut last_c: Option<BTreeMap<Field, f64>> = None;
    for it in 0..=ctrl.max_iter {
        let res = match sys.evaluate(true) {
            Ok(r) => r,
            // t.ex. förvrängda element eller misslyckad materialuppdatering: halvera
            Err(_) if it > 0 => return Ok(None),
            Err(e) => return Err(e),
        };
        // en användarrutin har begärt ett mindre inkrement (PNEWDT < 1): gör om inkrementet
        if crate::user::context::pnewdt().is_some_and(|p| p < 1.0) {
            return Ok(None);
        }
        let rmax = max_abs_by_field(sys, &res.r);
        let qref: BTreeMap<Field, f64> = res
            .flux
            .iter()
            .map(|(f, q)| (*f, (flux_sum.get(f).copied().unwrap_or(0.0) + q) / (n_flux + 1) as f64))
            .collect();
        let qall = qref.values().copied().fold(0.0, f64::max).max(1e-300);
        let inc = max_abs_by_field(sys, &sys.increment());
        let mut converged = true;
        // fält med nollflöde ingår inte i tidsmedelvärdet
        let mut zero_fields: Vec<Field> = Vec::new();
        for (f, &r) in &rmax {
            let q = qref.get(f).copied().unwrap_or(0.0).max(1e-12 * qall);
            if let Some(conv) = zero_flux_check(f, r, &res.flux, flux_sum, n_flux, &last_c, &inc) {
                converged &= conv;
                zero_fields.push(*f);
                continue;
            }
            if r <= ctrl.r_linear * q {
                continue;
            }
            let c_ok = match &last_c {
                Some(c) => {
                    let cm = c.get(f).copied().unwrap_or(0.0);
                    let du = inc.get(f).copied().unwrap_or(0.0);
                    cm <= ctrl.c_tol * du || cm == 0.0
                }
                None => false,
            };
            if !(r <= ctrl.r_tol * q && c_ok) {
                converged = false;
            }
        }
        if converged {
            let mut flux = res.flux;
            for f in &zero_fields {
                flux.insert(*f, 0.0);
            }
            return Ok(Some((it, flux)));
        }
        let rn = rmax.values().copied().fold(0.0, f64::max);
        crate::progress::emit(crate::progress::Event::Iteration { iteration: it, residual: rn })?;
        history.push(rn);
        let k = history.len();
        if it >= ctrl.i0 && k >= 3 && history[k - 1] > history[k - 2] && history[k - 2] > history[k - 3] {
            return Ok(None);
        }
        if it == ctrl.max_iter {
            return Ok(None);
        }
        let c = match sys.solve(&res.r) {
            Ok(c) => c,
            Err(_) if it > 0 => return Ok(None),
            Err(e) => return Err(e),
        };
        last_c = Some(max_abs_by_field(sys, &c));
        sys.apply(&c);
    }
    Ok(None)
}
