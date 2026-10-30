//! General Newton–Raphson driver with automatic load incrementation.
//!
//! Used by all steps that solve equilibrium incrementally (statics with nonlinear
//! materials, NLGEOM, contact, implicit dynamics). The problem is described by
//! `NonlinearSystem`: residual R = F_ext − F_int, tangent and update.
//!
//! The convergence check (default values correspond to *CONTROLS without changes), but checked for each field
//! (displacement, rotation, warping, temperature) separately:
//! - residual: max |r| ≤ R_tol · q̃, where q̃ is a time-averaged typical
//!   force (mean magnitude of internal and external forces) over the increments of the step;
//! - correction: max |c| ≤ C_tol · max |Δu| in the increment;
//! - linear increment: max |r| ≤ R_lin · q̃ is accepted without a correction check.
//!
//! Incrementation: start with `initial` (fraction of the step), cutback to a
//! quarter on divergence or too many iterations (at most `max_cutbacks`
//! times per increment), increase by the factor 1.5 after two consecutive increments
//! that converged in at most `ig` iterations. DIRECT gives fixed increments.

use crate::fem::dof::Field;
use crate::fem::model::RawKeyword;
use crate::model::{fail, Result};
use std::collections::BTreeMap;

/// Residual and typical forces per field.
#[derive(Debug, Clone, Default)]
pub struct Residual {
    /// R = F_ext − F_int for each equation.
    pub r: Vec<f64>,
    /// Typical force magnitude per field (for q̃).
    pub flux: BTreeMap<Field, f64>,
}

/// A nonlinear equilibrium problem for a step.
pub trait NonlinearSystem {
    fn n_eq(&self) -> usize;
    /// Field for each equation.
    fn eq_field(&self, eq: usize) -> Field;
    /// The problem is linear (one solution step is enough).
    fn is_linear(&self) -> bool;
    /// Starts an increment from step time `t0` to `t1` (fractions of the step, 0–1).
    /// Sets prescribed values at `t1` and resets the unknowns to the last accepted ones.
    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()>;
    /// Residual at the current solution; the tangent is assembled if `tangent`.
    fn evaluate(&mut self, tangent: bool) -> Result<Residual>;
    /// Solves K c = r with the most recently assembled tangent.
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>>;
    /// u += c for the free equations.
    fn apply(&mut self, c: &[f64]);
    /// The change since the start of the increment for each equation.
    fn increment(&self) -> Vec<f64>;
    /// Accepts the increment.
    fn commit(&mut self) -> Result<()>;
}

/// Increment and convergence parameters.
#[derive(Debug, Clone)]
pub struct Controls {
    /// Increment lengths as a fraction of the step time (0–1).
    pub initial: f64,
    pub min: f64,
    pub max: f64,
    /// Time length of the step (for printing times).
    pub period: f64,
    pub max_increments: usize,
    /// Fixed increments (DIRECT).
    pub fixed: bool,
    pub r_tol: f64,
    pub c_tol: f64,
    pub r_linear: f64,
    pub max_iter: usize,
    /// Iteration from which divergence is checked.
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
    /// Reads the *STATIC data line: initial increment, step time, minimum, maximum.
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

/// Solves a linear problem in a single step (t = 1) and accepts it.
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

/// Runs the step with Newton–Raphson. `on_increment(sys, t)` is called after each
/// accepted increment (t = step time fraction), e.g. to save a frame.
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
                "the step reached the maximum number of increments ({}) at step time {:.4} – increase INC= on *STEP",
                ctrl.max_increments,
                t * ctrl.period
            ));
        }
        dt = dt.min(1.0 - t);
        let mut cutbacks = 0;
        let iters = loop {
            // user subroutines: KINC and PNEWDT (user/context.rs)
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
                        return fail(format!("the increment at step time {:.4} did not converge (DIRECT, fixed increments)", (t + dt) * ctrl.period));
                    }
                    cutbacks += 1;
                    rep.cutbacks += 1;
                    had_cutback = true;
                    // PNEWDT < 1 from a user subroutine: new increment PNEWDT · Δt
                    dt *= match crate::user::context::pnewdt().filter(|&p| p < 1.0) {
                        Some(p) => p.max(1e-3),
                        None => ctrl.cutback,
                    };
                    if cutbacks > ctrl.max_cutbacks || dt < ctrl.min {
                        return fail(format!(
                            "no convergence at step time {:.4}: the increment became smaller than the minimum ({:.3e}) or too many cutbacks",
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
                    // the increment is increased as long as the two latest
                    // increments converged in at most I_G iterations
                    // (the counter is not reset at the increase;
                    // tests/cases/rubber/c3d8h_block_plast). If the step has already
                    // been cut back, the more cautious rule is kept (two new
                    // increments before the next increase), since difficult
                    // discontinuity iterations (contact) are not counted separately.
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

// ------------------------------------------------------------ zero flux
//
// The field α has zero flux when q̄ ≤ ε·q̃ (ε = 1e-5), where
// q̃ is the time average of q̄ over the increments of the step with flux, at the start of the step
// the value from the previous step. The solution is then accepted if r_max ≤ ε·q̃, otherwise if
// c_max ≤ C_ε·Δu_max (C_ε = 1e-3). Without this a step where all
// forces go to zero does not converge (e.g. an empty step after unloading to a
// stress-free state): the relative criterion then compares round-off errors with
// round-off errors.

/// The criterion for zero flux relative to the time average (ε^α).
const ZERO_FLUX: f64 = 1e-5;
/// The correction criterion at zero flux (C_ε^α).
const ZERO_FLUX_CORRECTION: f64 = 1e-3;

thread_local! {
    /// q̃ per field at the end of the previous step (starting value for the next step).
    static PREV_TIME_AVERAGE: std::cell::RefCell<BTreeMap<Field, f64>> = const { std::cell::RefCell::new(BTreeMap::new()) };
}

/// Forgets q̃ from earlier steps (called when an analysis starts, `steps::run`).
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

/// `Some(converged)` if the field `f` has zero flux in the current iteration, otherwise `None`.
fn zero_flux_check(
    f: &Field,
    r: f64,
    flux: &BTreeMap<Field, f64>,
    flux_sum: &BTreeMap<Field, f64>,
    n_flux: usize,
    last_c: &Option<BTreeMap<Field, f64>>,
    inc: &BTreeMap<Field, f64>,
) -> Option<bool> {
    // q̃ without the current increment: the earlier increments of the step, otherwise the previous step
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

/// Iterates an increment that has already been started with `begin_increment`, with
/// the time average of the flux from `n_flux` earlier accepted increments
/// (`flux_sum` is the sum). For steps that handle the incrementation themselves
/// (implicit dynamics, `dynamic_incr.rs`). Return value as for `iterate`.
pub fn iterate_increment<S: NonlinearSystem + ?Sized>(
    sys: &mut S,
    ctrl: &Controls,
    flux_sum: &BTreeMap<Field, f64>,
    n_flux: usize,
) -> Result<Option<(usize, BTreeMap<Field, f64>)>> {
    crate::user::context::reset_pnewdt();
    iterate(sys, ctrl, flux_sum, n_flux)
}

/// Iterates an increment. `Some((iterations, flux))` on convergence, `None` if
/// the increment is to be cut back.
fn iterate<S: NonlinearSystem + ?Sized>(sys: &mut S, ctrl: &Controls, flux_sum: &BTreeMap<Field, f64>, n_flux: usize) -> Result<Option<(usize, BTreeMap<Field, f64>)>> {
    let mut history: Vec<f64> = Vec::new();
    let mut last_c: Option<BTreeMap<Field, f64>> = None;
    for it in 0..=ctrl.max_iter {
        let res = match sys.evaluate(true) {
            Ok(r) => r,
            // e.g. distorted elements or a failed material update: cut back
            Err(_) if it > 0 => return Ok(None),
            Err(e) => return Err(e),
        };
        // a user subroutine has requested a smaller increment (PNEWDT < 1): redo the increment
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
        // fields with zero flux are not included in the time average
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
