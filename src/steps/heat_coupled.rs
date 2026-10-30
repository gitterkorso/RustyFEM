//! *COUPLED TEMPERATURE-DISPLACEMENT (stationär eller transient) med de
//! kopplade elementen i `elements/heat_coupled.rs` (C3D8T, CPE4T, CPS4T, CAX4T).
//!
//! Vid små töjningar utan värmealstring från deformationen (ingen plastisk
//! dissipation, ingen spaltkonduktans) beror värmeledningen inte av
//! förskjutningarna: tangentmatrisen är blocktriangulär,
//!
//! ```text
//! [ K_uu  K_uT ] [Δu]   [R_u]
//! [  0    K_TT ] [ΔT] = [R_T]
//! ```
//!
//! och varje inkrement kan lösas exakt i två led: först temperaturen (samma
//! system som *HEAT TRANSFER, `steps/heat.rs`), sedan jämvikten med den
//! termiska töjningen ur temperaturen vid inkrementets början och slut
//! (`steps/heat_stress.rs`; töjningen dras av i materialet, `materials/thermal.rs`). Lösningen är
//! densamma som med den fullt kopplade Newtonmetoden (samma diskreta
//! lösning som när hela systemet löses samtidigt).
//!
//! Inkrementering: transient med fasta inkrement Δt0 (utan DELTMX) eller
//! automatiskt efter DELTMX (samma regel som *HEAT TRANSFER); stationärt med
//! inkrementet Δt0 (fasta inkrement). Laster: alla mekaniska laster och
//! randvillkor samt värmeflödena (*CFLUX, *DFLUX, *FILM, *RADIATE).
//! Utdata: U, NT11, RF, RFL11, S, E, HFL.

use super::api::{StepCtx, StepType};
use super::heat::{ensure_initial, heat_controls, heat_loads, HeatSystem, INIT_KEY, TS};
use super::heat_stress::ThermoSystem;
use super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::fem::dof::{Field, NodeVec, SLOTS};
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::fem::state::StepLoads;
use crate::model::{fail, Result};

#[path = "heat_coupled_contact.rs"]
mod contact;

pub struct CoupledTempDisp;

pub static STEP_TYPES: &[&dyn StepType] = &[&CoupledTempDisp];

const PROC: &str = "COUPLED TEMPERATURE-DISPLACEMENT";

/// Ett inkrement [t0, t1] av ett delsystem med fast Δt.
struct Sub<'a, S: NonlinearSystem> {
    sys: &'a mut S,
    t0: f64,
    t1: f64,
}

impl<S: NonlinearSystem> NonlinearSystem for Sub<'_, S> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        // linjära delsystem löses med en enda lösning (ingen konvergenskontroll,
        // som annars misslyckas när alla flöden är noll, t.ex. fri utvidgning)
        self.sys.is_linear()
    }
    fn begin_increment(&mut self, _a: f64, _b: f64) -> Result<()> {
        self.sys.begin_increment(self.t0, self.t1)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
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
        Ok(())
    }
}

/// Löser ett inkrement av delsystemet och godkänner det. Linjära delsystem
/// löses med en enda lösning (`newton::solve_linear`).
fn solve_increment<S: NonlinearSystem>(sys: &mut S, ctrl: &Controls, t0: f64, t1: f64) -> Result<usize> {
    let sub = Controls { initial: 1.0, fixed: true, ..ctrl.clone() };
    let mut one = Sub { sys: &mut *sys, t0, t1 };
    let rep = newton::run(&mut one, &sub, &mut |_, _| Ok(()))?;
    sys.commit()?;
    Ok(rep.iterations)
}

/// Delar upp stegets laster: värmeledningens (temperaturer och värmeflöden)
/// och mekanikens (resten).
fn split_loads(l: &StepLoads) -> (StepLoads, StepLoads) {
    let thermal_kind = |k: &str| k.starts_with("DFLUX:") || k.starts_with("FILM") || k.starts_with("RAD:");
    let mut th = l.clone();
    let mut me = l.clone();
    th.bc.retain(|&(_, s), _| s == TS);
    me.bc.retain(|&(_, s), _| s != TS);
    th.cload.retain(|&(_, s), _| s == TS);
    me.cload.retain(|&(_, s), _| s != TS);
    th.dload.retain(|(d, _)| thermal_kind(&d.kind));
    me.dload.retain(|(d, _)| !thermal_kind(&d.kind));
    th.released.retain(|&(_, s, _)| s == TS);
    me.released.retain(|&(_, s, _)| s != TS);
    (th, me)
}

fn temps(u: &[NodeVec]) -> Vec<f64> {
    u.iter().map(|v| v[TS]).collect()
}

/// Sammanslagen frame: förskjutningar, spänningar och reaktioner från
/// mekaniken, temperaturer, HFL och reaktionsflöden från värmeledningen.
fn frame(heat: &HeatSystem, mech: &ThermoSystem, value: f64, description: String) -> Result<FrameData> {
    let h = heat.frame(value, description.clone())?;
    let mut m = mech.frame(value, description)?;
    for n in 0..m.u.len() {
        m.u[n][TS] = h.u[n][TS];
        m.rf[n][TS] = h.rf[n][TS];
        let bit = 1u16 << TS;
        m.active[n] = (m.active[n] & !bit) | (h.active[n] & bit);
        m.constrained[n] = (m.constrained[n] & !bit) | (h.constrained[n] & bit);
    }
    for (a, b) in m.elem.iter_mut().zip(h.elem) {
        a.extend(b);
    }
    // värmeflödet över termisk kontakt (HFL) från värmeledningsdelen
    m.node_extra.extend(h.node_extra);
    Ok(m)
}

impl StepType for CoupledTempDisp {
    fn name(&self) -> &'static str {
        PROC
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == PROC).then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.nlgeom {
            return fail("*COUPLED TEMPERATURE-DISPLACEMENT med NLGEOM=YES stöds inte ännu");
        }
        if ctx.step.perturbation {
            return fail("*COUPLED TEMPERATURE-DISPLACEMENT kan inte vara ett perturbationssteg");
        }
        for (e, el) in model.elements.iter().enumerate() {
            if crate::elements::heat_coupled::lookup(el.etype.name()).is_none() {
                return fail(format!(
                    "{}: *COUPLED TEMPERATURE-DISPLACEMENT kräver kopplade element (C3D8T, CPE4T, CPS4T, CAX4T), inte {}",
                    model.elem_name(e),
                    el.etype.name()
                ));
            }
        }
        let hc = heat_controls(ctx.step)?;
        let had_init = ctx.state.vectors.contains_key(INIT_KEY);
        ensure_initial(model, ctx.state)?;
        let mut loads = heat_loads(ctx.step, ctx.loads, hc.steady);
        if !had_init {
            // randvillkor som rampas i det första steget utgår från begynnelsetemperaturen
            for ((n, s), v) in loads.bc.iter_mut() {
                if *s == TS && !ctx.state.bcs.contains_key(&(*n, *s)) {
                    v.0 = ctx.state.u[*n][TS];
                }
            }
        }
        let (th_loads, me_loads) = split_loads(&loads);
        let period = hc.ctrl.period;
        let mut heat = HeatSystem::new(prep, th_loads, ctx.state, !hc.steady, ctx.solver, period)?;
        let mut mech = ThermoSystem::new(prep, &me_loads, ctx.state, ctx.solver, period, true)?;
        mech.coupled = (temps(&ctx.state.u), temps(&ctx.state.u));
        // kontakt och spaltkonduktans: delsystemen beror av varandra och
        // itereras i varje inkrement (heat_coupled_contact.rs)
        // område coupled2: värme från plastisk dissipation (*INELASTIC HEAT FRACTION)
        let eta = crate::materials::inelastic_heat::fractions(model)?;
        let plastic = eta.iter().any(|&x| x > 0.0) && !hc.steady;
        let interacting = !prep.interactions.is_empty() || !heat.gaps.is_empty() || plastic;
        heat.gap_disp = !heat.gaps.is_empty();
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            heat.begin_increment(0.0, 0.0)?;
            heat.u.clone_from(&heat.u_committed);
            let tr = heat.transient;
            heat.transient = false;
            heat.evaluate(false)?;
            heat.transient = tr;
            mech.begin_increment(0.0, 0.0)?;
            mech.evaluate(false)?;
            let mut f0 = frame(&heat, &mech, 0.0, "Increment 0: Step Time = 0.0".into())?;
            // frame 0 är begynnelsetillståndet: inga reaktioner
            for r in f0.rf.iter_mut() {
                *r = [0.0; SLOTS];
            }
            frames.push(f0);
        }
        if hc.deltmx > 0.0 && !hc.steady {
            return fail("DELTMX i *COUPLED TEMPERATURE-DISPLACEMENT stöds inte ännu (ange fasta inkrement)");
        }
        let ctrl = &hc.ctrl;
        let dt0 = ctrl.initial.clamp(1e-12, 1.0);
        let mut t = 0.0f64;
        let mut inc = 0usize;
        let mut iterations = 0usize;
        while t < 1.0 - 1e-12 {
            if inc >= ctrl.max_increments {
                return fail(format!("steget nådde största antal inkrement ({}) – öka INC= på *STEP", ctrl.max_increments));
            }
            let t1 = (t + dt0).min(1.0);
            let t1 = if 1.0 - t1 < 1e-9 { 1.0 } else { t1 };
            let before = temps(&heat.u_committed);
            if interacting {
                let pl = if plastic { Some((&eta[..], (t1 - t) * period)) } else { None };
                iterations += contact::solve_coupled_with(&mut heat, &mut mech, ctrl, t, t1, ctx.opts.verbose, pl)?;
            } else {
                iterations += solve_increment(&mut heat, ctrl, t, t1)?;
                mech.coupled = (before, temps(&heat.u_committed));
                iterations += solve_increment(&mut mech, ctrl, t, t1)?;
            }
            t = t1;
            inc += 1;
            if ctx.opts.verbose {
                eprintln!("  inkrement {inc}: stegtid {:.6}", t * period);
            }
            if all || t >= 1.0 - 1e-12 {
                frames.push(frame(&heat, &mech, t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = inc;
        ctx.out.iterations = iterations;
        let mut u = mech.u_committed.clone();
        let mut rf = mech.reactions();
        let hr = heat.reactions();
        for n in 0..u.len() {
            u[n][TS] = heat.u_committed[n][TS];
            rf[n][TS] = hr[n][TS];
        }
        ctx.state.u = u;
        ctx.state.reaction = rf;
        ctx.state.elem_state = mech.state_old.clone();
        ctx.state.time += period;
        Ok(())
    }
}
