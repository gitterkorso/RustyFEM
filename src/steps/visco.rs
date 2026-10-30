//! *VISCO: quasi-static analysis with time-dependent materials (creep,
//! viscoelasticity), with or without NLGEOM.
//!
//! The data line is (initial time increment, step time, minimum, maximum). Without
//! CETOL, fixed increments equal to the initial one are used (DIRECT
//! does not exist on *VISCO). With CETOL=tol the increments are controlled by the accuracy of the
//! time integration according to the following rules (default values for *CONTROLS,
//! PARAMETERS=TIME INCREMENTATION):
//!
//! - the error measure S = the largest difference in creep/viscous strain increment computed
//!   from the conditions at the start and at the end of the increment (the materials
//!   report it via `time_dependence.rs`);
//! - S > tol: the increment is redone with Δt · D_A · tol/S, D_A = 0.85;
//! - increase: if Δt (S/Δt)ᵢ < W_G tol (W_G = 0.75) in each of the
//!   I_T = 3 most recent increments (without retries), the proposed increment is
//!   Δt_p = I_T tol / Σ (S/Δt)ᵢ; the next increment becomes min(D_G Δt_p, D_M Δt)
//!   (D_G = 0.8, D_M = 2 for creep) but only if D_G Δt_p ≥ 1.875 Δt, otherwise
//!   Δt is kept (small changes of Δt are avoided). No increase is made if the increased
//!   increment would reach the end of the step (increase factor 1 before a time point).
//!
//! If the increment required iterations (nonlinear response, e.g. implicit creep),
//! the largest increase is D_D = 1.5 instead of D_M.
//!
//! Explicit creep integration is also limited by the stability limit Δt_s
//! (see `materials/creep.rs`): if Δt > Δt_s the increment is redone with 0.95 Δt_s.
//! If Δt_s is smaller than the accuracy limit Δt_c = tol Δt/S in nine consecutive
//! increments (and at least 50 Δt remain of the step), the integration switches to implicit for
//! the rest of the step (not with CREEP=EXPLICIT).
//!
//! Loads with default amplitude: in *VISCO the default is STEP (the load acts
//! immediately) except for prescribed displacements, which are ramped. *STEP, AMPLITUDE=RAMP/STEP applies to everything.
//! Predefined temperatures follow the same rule (`materials/thermal.rs`).
//!
//! Convergence: Newton–Raphson with the standard criteria, and then further to
//! a negligible residual (`nlgeom::iterate`). Cutback to a quarter on
//! divergence (error with fixed increments).

use super::api::{StepCtx, StepType};
use super::newton::{Controls, NonlinearSystem, Residual};
use super::nlgeom::{self, NlSystem};
use crate::fem::assembly;
use crate::fem::dof::Field;
use crate::fem::model::{parse_f64, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::fem::system::StaticSystem;
use crate::materials::time_dependence::{self as td, TimeDependence, CREEP_EXPLICIT, CREEP_IMPLICIT};
use crate::model::{fail, FemError, Result};
use std::collections::BTreeMap;

pub struct Visco;

pub static STEP_TYPES: &[&dyn StepType] = &[&Visco];

/// Default values for the time incrementation (*CONTROLS).
const D_A: f64 = 0.85;
const W_G: f64 = 0.75;
const D_G: f64 = 0.8;
const D_M: f64 = 2.0;
const D_D: f64 = 1.5;
/// Minimum increase factor D_G Δt_p/Δt for Δt to be increased.
const F_MIN: f64 = 1.875;
const I_T: usize = 3;

/// Shared by the systems with and without NLGEOM.
trait ViscoSystem: NonlinearSystem {
    fn frame_at(&self, value: f64, inc: usize) -> Result<FrameData>;
    fn store_state(&self, state: &mut SolutionState);
}

impl ViscoSystem for StaticSystem<'_, '_> {
    fn frame_at(&self, value: f64, inc: usize) -> Result<FrameData> {
        assembly::frame(
            self.p,
            &self.dm,
            &self.u_committed,
            &self.fint,
            &self.fext,
            &self.state_old,
            false,
            self.time0 + value,
            value,
            format!("Increment {inc}: Step Time = {value:.4}"),
        )
    }

    fn store_state(&self, state: &mut SolutionState) {
        self.store(state);
    }
}

impl ViscoSystem for NlSystem<'_, '_> {
    fn frame_at(&self, value: f64, inc: usize) -> Result<FrameData> {
        self.frame(value, format!("Increment {inc}: Step Time = {value:.4}"))
    }

    fn store_state(&self, state: &mut SolutionState) {
        self.store(state);
    }
}

/// Resets the materials' monitoring before each evaluation, so that the error measure
/// after convergence belongs to the accepted solution.
struct Monitored<'a, S: ?Sized> {
    sys: &'a mut S,
    tds: &'a [&'a TimeDependence],
}

impl<S: NonlinearSystem + ?Sized> NonlinearSystem for Monitored<'_, S> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        self.sys.is_linear()
    }
    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        self.sys.begin_increment(t0, t1)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        td::reset_all(self.tds);
        self.sys.evaluate(tangent)
    }
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        self.sys.solve(r)
    }
    fn apply(&mut self, c: &[f64]) {
        self.sys.apply(c)
    }
    fn increment(&self) -> Vec<f64> {
        self.sys.increment()
    }
    fn commit(&mut self) -> Result<()> {
        self.sys.commit()
    }
}

/// The step settings.
struct Setup {
    ctrl: Controls,
    /// CETOL (None = fixed increments).
    cetol: Option<f64>,
    /// Maximum increment (fraction of step time).
    max: f64,
    explicit_forced: bool,
}

fn setup(ctx: &StepCtx) -> Result<Setup> {
    let kw = ctx.step.procedure.as_ref().ok_or_else(|| FemError("*VISCO missing".into()))?;
    for p in ["STABILIZE", "ALLSDTOL", "FACTOR", "CONTINUE"] {
        if kw.has(p) {
            return kw.err(format!("the parameter {p} is not yet supported"));
        }
    }
    let cetol = match kw.param("CETOL") {
        None => None,
        Some(s) => Some(parse_f64(s).filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: CETOL={s} must be a positive number", kw.location())))?),
    };
    let explicit_forced = match kw.param_upper("CREEP").as_deref() {
        None => false,
        Some("EXPLICIT") => true,
        Some(o) => return kw.err(format!("CREEP={o} is not supported (EXPLICIT)")),
    };
    let v = kw.first_floats()?;
    let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
    let dt0 = v.first().copied().filter(|&x| x > 0.0).unwrap_or(period).min(period);
    let mut ctrl = Controls { max_increments: ctx.step.max_increments.max(1), period, initial: dt0 / period, ..Default::default() };
    ctrl.min = match v.get(2).copied().filter(|&x| x > 0.0) {
        Some(m) => m / period,
        None => (dt0 / period).min(1e-5),
    };
    ctrl.fixed = cetol.is_none();
    let max = v.get(3).copied().filter(|&x| x > 0.0).map_or(1.0, |m| (m / period).min(1.0));
    // *CONTROLS apply from the step where they are specified until RESET
    for s in ctx.prep.model.steps.iter().take(ctx.index + 1) {
        nlgeom::apply_controls(&mut ctrl, &s.extra)?;
    }
    Ok(Setup { ctrl, cetol, max, explicit_forced })
}

/// The loads with the *VISCO default amplitude: concentrated and distributed loads
/// act immediately (STEP), prescribed displacements are ramped.
fn visco_loads(step: &StepDef, loads: &StepLoads) -> StepLoads {
    let mut l = loads.clone();
    if td::default_step_amplitude(step) {
        for v in l.cload.values_mut() {
            v.0 = v.1;
        }
        for (d, start) in l.dload.iter_mut() {
            *start = d.magnitude;
        }
        l.released.clear();
        l.ramp = true;
    }
    l
}

impl StepType for Visco {
    fn name(&self) -> &'static str {
        "VISCO"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "VISCO").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        if ctx.step.perturbation {
            return fail("*VISCO cannot be a perturbation step");
        }
        let su = setup(ctx)?;
        let loads = visco_loads(ctx.step, ctx.loads);
        let prep = ctx.prep;
        let tds = td::all(&prep.materials);
        let (incs, iters) = if ctx.nlgeom {
            let mut sys = NlSystem::new(prep, &loads, ctx.state, ctx.solver, su.ctrl.period)?;
            let r = drive(&mut sys, &su, &tds, ctx)?;
            sys.store_state(ctx.state);
            r
        } else {
            let mut sys = StaticSystem::new(prep, &loads, ctx.state, false, ctx.solver, su.ctrl.period)?;
            let r = drive(&mut sys, &su, &tds, ctx)?;
            sys.store_state(ctx.state);
            r
        };
        ctx.out.increments = incs;
        ctx.out.iterations = iters;
        ctx.state.time += su.ctrl.period;
        Ok(())
    }
}

/// Incrementation with accuracy control (CETOL) and stability limit.
fn drive<S: ViscoSystem>(sys: &mut S, su: &Setup, tds: &[&TimeDependence], ctx: &mut StepCtx) -> Result<(usize, usize)> {
    let ctrl = &su.ctrl;
    let period = ctrl.period;
    let all = ctx.opts.all_frames;
    let verbose = ctx.opts.verbose;
    let mut frames = Vec::new();
    if all {
        // frame 0: the state at the start of the step
        let mut m = Monitored { sys: &mut *sys, tds };
        m.begin_increment(0.0, 0.0)?;
        m.evaluate(false)?;
        frames.push(sys.frame_at(0.0, 0)?);
    }
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
    let mut n_flux = 0usize;
    let (mut incs, mut iters) = (0usize, 0usize);
    // (S/Δt) for the most recent accepted increments without retries
    let mut hist: Vec<f64> = Vec::new();
    let mut stab_limited = 0usize;
    while t < 1.0 - 1e-12 {
        if incs >= ctrl.max_increments {
            return fail(format!("the step reached the maximum number of increments ({}) at step time {:.4} – increase INC= on *STEP", ctrl.max_increments, t * period));
        }
        if t + dt > 1.0 - 1e-12 {
            dt = 1.0 - t;
        }
        let mut cutbacks = 0;
        let (out, err, dts) = loop {
            let mut m = Monitored { sys: &mut *sys, tds };
            m.begin_increment(t, t + dt)?;
            let mut solve = |s: &mut Monitored<S>, r: &[f64]| s.solve(r);
            let res = nlgeom::iterate(&mut m, ctrl, &flux_sum, n_flux, &mut solve)?;
            let retry = |dt: &mut f64, f: f64, why: &str| -> Result<()> {
                *dt *= f;
                if *dt < ctrl.min {
                    return fail(format!("{why} at step time {:.4}: the increment became smaller than the minimum ({:.3e})", t * period, ctrl.min * period));
                }
                Ok(())
            };
            let Some(o) = res else {
                if ctrl.fixed {
                    return fail(format!("the increment at step time {:.4} did not converge (fixed increments)", (t + dt) * period));
                }
                cutbacks += 1;
                hist.clear();
                if cutbacks > ctrl.max_cutbacks {
                    return fail(format!("no convergence at step time {:.4}: too many cutbacks", t * period));
                }
                retry(&mut dt, ctrl.cutback, "no convergence")?;
                continue;
            };
            let (err, dts) = td::monitor(tds);
            let explicit = tds.iter().any(|x| x.creep() == CREEP_EXPLICIT);
            if explicit && dts.is_finite() && dt * period > dts * (1.0 + 1e-9) {
                // stability limit for explicit creep integration: new attempt with 0.95 Δt_s
                let f = 0.95 * dts / (dt * period);
                retry(&mut dt, f, "the stability limit for explicit creep")?;
                continue;
            }
            if let Some(tol) = su.cetol {
                if err > tol {
                    cutbacks += 1;
                    hist.clear();
                    retry(&mut dt, D_A * tol / err, "creep accuracy (CETOL)")?;
                    continue;
                }
            }
            break (o, err, dts);
        };
        for (f, q) in &out.flux {
            *flux_sum.entry(*f).or_default() += q;
        }
        n_flux += 1;
        sys.commit()?;
        t += dt;
        incs += 1;
        iters += out.iterations;
        crate::progress::emit(crate::progress::Event::Increment { fraction: t, iterations: out.iterations })?;
        if verbose {
            eprintln!("  increment {incs}: step time {:.6} (Δt = {:.4e}, error {:.3e}, Δt_s {:.4e})", t * period, dt * period, err, dts);
        }
        if all || t >= 1.0 - 1e-12 {
            frames.push(sys.frame_at(t * period, incs)?);
        }
        // switch to implicit integration when the stability limit Δt_s is smaller than
        // the accuracy limit Δt_c = tol Δt/S in nine consecutive increments
        let dtp = dt * period;
        let explicit = tds.iter().any(|x| x.creep() == CREEP_EXPLICIT);
        let dtc = match su.cetol {
            Some(tol) if err > 0.0 => tol * dtp / err,
            Some(_) => f64::INFINITY,
            None => dtp,
        };
        stab_limited = if explicit && dts < dtc { stab_limited + 1 } else { 0 };
        if stab_limited >= 9 && !su.explicit_forced && (1.0 - t) >= 50.0 * dt {
            tds.iter().filter(|x| x.creep() == CREEP_EXPLICIT).for_each(|x| x.set_creep(CREEP_IMPLICIT));
            if verbose {
                eprintln!("  switching to implicit creep integration at step time {:.6}", t * period);
            }
        }
        if let Some(tol) = su.cetol {
            // largest increase: D_M = 2 for creep, 1.5 (D_D) if the increment required iterations
            let d_m = if out.iterations <= 1 { D_M } else { D_D };
            if cutbacks == 0 {
                hist.push(err / dtp);
                if hist.len() > I_T {
                    hist.remove(0);
                }
            }
            if hist.len() == I_T && hist.iter().all(|h| h * dtp < W_G * tol) {
                let s: f64 = hist.iter().sum();
                let prop = if s > 0.0 { I_T as f64 * tol / s } else { f64::INFINITY };
                // increase only by at least the factor F_MIN (avoids small changes of Δt)
                let ok = D_G * prop >= F_MIN * dtp;
                let new = (D_G * prop).min(d_m * dtp) / period;
                // increase factor 1 before a time point (end of the step)
                if ok && t + new < 1.0 - 1e-12 {
                    dt = new;
                }
            }
            dt = dt.min(su.max);
        } else {
            dt = ctrl.initial;
        }
    }
    ctx.out.frames.extend(frames);
    Ok((incs, iters))
}
