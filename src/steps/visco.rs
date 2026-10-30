//! *VISCO: kvasistatisk analys med tidsberoende material (krypning,
//! viskoelasticitet), med eller utan NLGEOM.
//!
//! Dataraden är (initialt tidsinkrement, stegtid, minsta, största). Utan
//! CETOL används fasta inkrement lika med det initiala (DIRECT
//! finns inte på *VISCO). Med CETOL=tol styrs inkrementen av tidsintegrationens
//! noggrannhet enligt följande regler (standardvärden för *CONTROLS,
//! PARAMETERS=TIME INCREMENTATION):
//!
//! - felmåttet S = största skillnaden i kryp-/viskös töjningsökning beräknad
//!   från villkoren vid inkrementets början respektive slut (materialen
//!   rapporterar det via `time_dependence.rs`);
//! - S > tol: inkrementet görs om med Δt · D_A · tol/S, D_A = 0,85;
//! - ökning: om Δt (S/Δt)ᵢ < W_G tol (W_G = 0,75) i vart och ett av de
//!   I_T = 3 senaste inkrementen (utan omtag) är det föreslagna inkrementet
//!   Δt_p = I_T tol / Σ (S/Δt)ᵢ; nästa inkrement blir min(D_G Δt_p, D_M Δt)
//!   (D_G = 0,8, D_M = 2 för krypning) men bara om D_G Δt_p ≥ 1,875 Δt, annars
//!   behålls Δt (små ändringar av Δt undviks). Ingen ökning görs om det ökade
//!   inkrementet skulle nå stegets slut (ökningsfaktor 1 före en tidpunkt).
//!
//! Krävde inkrementet iterationer (olinjärt svar, t.ex. implicit krypning) är
//! den största ökningen D_D = 1,5 i stället för D_M.
//!
//! Explicit krypintegration begränsas dessutom av stabilitetsgränsen Δt_s
//! (se `materials/creep.rs`): är Δt > Δt_s görs inkrementet om med 0,95 Δt_s.
//! Är Δt_s mindre än noggrannhetsgränsen Δt_c = tol Δt/S i nio inkrement i
//! följd (och minst 50 Δt kvar av steget) byts till implicit integration för
//! resten av steget (inte med CREEP=EXPLICIT).
//!
//! Laster med standardamplitud: i *VISCO är standarden STEP (lasten verkar
//! direkt) utom för föreskrivna förskjutningar som rampas. *STEP, AMPLITUDE=RAMP/STEP gäller för allt.
//! Fördefinierade temperaturer följer samma regel (`materials/thermal.rs`).
//!
//! Konvergens: Newton–Raphson med standardkriterierna, och därefter vidare till
//! försumbar residual (`nlgeom::iterate`). Halvering till en fjärdedel vid
//! divergens (fel med fasta inkrement).

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

/// Standardvärden för tidsinkrementeringen (*CONTROLS).
const D_A: f64 = 0.85;
const W_G: f64 = 0.75;
const D_G: f64 = 0.8;
const D_M: f64 = 2.0;
const D_D: f64 = 1.5;
/// Minsta ökningsfaktor D_G Δt_p/Δt för att Δt ska ökas.
const F_MIN: f64 = 1.875;
const I_T: usize = 3;

/// Gemensamt för systemen med och utan NLGEOM.
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

/// Nollställer materialens övervakning före varje utvärdering, så att felmåttet
/// efter konvergens hör till den godkända lösningen.
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

/// Stegets inställningar.
struct Setup {
    ctrl: Controls,
    /// CETOL (None = fasta inkrement).
    cetol: Option<f64>,
    /// Största inkrement (stegtidsandel).
    max: f64,
    explicit_forced: bool,
}

fn setup(ctx: &StepCtx) -> Result<Setup> {
    let kw = ctx.step.procedure.as_ref().ok_or_else(|| FemError("*VISCO saknas".into()))?;
    for p in ["STABILIZE", "ALLSDTOL", "FACTOR", "CONTINUE"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let cetol = match kw.param("CETOL") {
        None => None,
        Some(s) => Some(parse_f64(s).filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: CETOL={s} ska vara ett positivt tal", kw.location())))?),
    };
    let explicit_forced = match kw.param_upper("CREEP").as_deref() {
        None => false,
        Some("EXPLICIT") => true,
        Some(o) => return kw.err(format!("CREEP={o} stöds inte (EXPLICIT)")),
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
    // *CONTROLS gäller från steget där de anges tills RESET
    for s in ctx.prep.model.steps.iter().take(ctx.index + 1) {
        nlgeom::apply_controls(&mut ctrl, &s.extra)?;
    }
    Ok(Setup { ctrl, cetol, max, explicit_forced })
}

/// Lasterna med *VISCO:s standardamplitud: punktlaster och fördelade laster
/// verkar direkt (STEP), föreskrivna förskjutningar rampas.
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
            return fail("*VISCO kan inte vara ett perturbationssteg");
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

/// Inkrementering med noggrannhetskontroll (CETOL) och stabilitetsgräns.
fn drive<S: ViscoSystem>(sys: &mut S, su: &Setup, tds: &[&TimeDependence], ctx: &mut StepCtx) -> Result<(usize, usize)> {
    let ctrl = &su.ctrl;
    let period = ctrl.period;
    let all = ctx.opts.all_frames;
    let verbose = ctx.opts.verbose;
    let mut frames = Vec::new();
    if all {
        // frame 0: tillståndet vid stegets början
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
    // (S/Δt) för de senaste godkända inkrementen utan omtag
    let mut hist: Vec<f64> = Vec::new();
    let mut stab_limited = 0usize;
    while t < 1.0 - 1e-12 {
        if incs >= ctrl.max_increments {
            return fail(format!("steget nådde största antal inkrement ({}) vid stegtid {:.4} – öka INC= på *STEP", ctrl.max_increments, t * period));
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
                    return fail(format!("{why} vid stegtid {:.4}: inkrementet blev mindre än minimum ({:.3e})", t * period, ctrl.min * period));
                }
                Ok(())
            };
            let Some(o) = res else {
                if ctrl.fixed {
                    return fail(format!("inkrementet vid stegtid {:.4} konvergerade inte (fasta inkrement)", (t + dt) * period));
                }
                cutbacks += 1;
                hist.clear();
                if cutbacks > ctrl.max_cutbacks {
                    return fail(format!("ingen konvergens vid stegtid {:.4}: för många halveringar", t * period));
                }
                retry(&mut dt, ctrl.cutback, "ingen konvergens")?;
                continue;
            };
            let (err, dts) = td::monitor(tds);
            let explicit = tds.iter().any(|x| x.creep() == CREEP_EXPLICIT);
            if explicit && dts.is_finite() && dt * period > dts * (1.0 + 1e-9) {
                // stabilitetsgränsen för explicit krypintegration: nytt försök med 0,95 Δt_s
                let f = 0.95 * dts / (dt * period);
                retry(&mut dt, f, "stabilitetsgränsen för explicit krypning")?;
                continue;
            }
            if let Some(tol) = su.cetol {
                if err > tol {
                    cutbacks += 1;
                    hist.clear();
                    retry(&mut dt, D_A * tol / err, "krypningens noggrannhet (CETOL)")?;
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
            eprintln!("  inkrement {incs}: stegtid {:.6} (Δt = {:.4e}, fel {:.3e}, Δt_s {:.4e})", t * period, dt * period, err, dts);
        }
        if all || t >= 1.0 - 1e-12 {
            frames.push(sys.frame_at(t * period, incs)?);
        }
        // byte till implicit integration när stabilitetsgränsen Δt_s är mindre än
        // noggrannhetens Δt_c = tol Δt/S i nio inkrement i följd
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
                eprintln!("  byter till implicit krypintegration vid stegtid {:.6}", t * period);
            }
        }
        if let Some(tol) = su.cetol {
            // största ökning: D_M = 2 för krypning, 1,5 (D_D) om inkrementet krävde iterationer
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
                // ökning bara med minst faktorn F_MIN (undviker små ändringar av Δt)
                let ok = D_G * prop >= F_MIN * dtp;
                let new = (D_G * prop).min(d_m * dtp) / period;
                // ökningsfaktor 1 före en tidpunkt (stegets slut)
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
