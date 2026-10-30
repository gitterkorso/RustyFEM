//! *STATIC utan NLGEOM.
//!
//! - Linjär modell (alla material linjära, inga interaktioner): en direkt
//!   lösning med lasterna vid stegets slut.
//! - Materiellt olinjär modell (t.ex. plasticitet): Newton–Raphson med
//!   automatisk inkrementering (`newton.rs`), små töjningar.
//! - PERTURBATION: linjärt svar på stegets egna laster; utdata är
//!   perturbationen och tillståndet återställs efteråt.
//!
//! NLGEOM=YES hanteras av `nlgeom.rs`.

use super::api::{StepCtx, StepType};
use super::newton::{self, Controls, NonlinearSystem};
use crate::fem::assembly;
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::fem::state::SolutionState;
use crate::fem::system::StaticSystem;
use crate::model::{fail, Result};

pub struct Static;

pub static STEP_TYPES: &[&dyn StepType] = &[&Static];

fn frame_of(sys: &StaticSystem, value: f64, inc: usize) -> Result<FrameData> {
    assembly::frame(
        sys.p,
        &sys.dm,
        &sys.u_committed,
        &sys.fint,
        &sys.fext,
        &sys.state_old,
        sys.nlgeom,
        sys.time0 + value,
        value,
        format!("Increment {inc}: Step Time = {value:.4}"),
    )
}

impl StepType for Static {
    fn name(&self) -> &'static str {
        "STATIC"
    }

    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "STATIC" && !nlgeom && !step.procedure.as_ref().is_some_and(|p| p.has("RIKS"))).then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let mut ctrl = Controls::from_procedure(ctx.step.procedure.as_ref(), ctx.step.max_increments)?;
        // *CONTROLS gäller från steget där de anges tills RESET
        for s in prep.model.steps.iter().take(ctx.index + 1) {
            super::nlgeom::apply_controls(&mut ctrl, &s.extra)?;
        }
        if ctx.step.perturbation {
            if !prep.is_linear() {
                return fail("perturbationssteg med olinjära material eller interaktioner stöds inte ännu");
            }
            // linjärt svar: oberoende av bastillståndet för linjära material
            let mut zero = SolutionState::new(prep.model);
            zero.elem_state = prep.init_states()?;
            let mut sys = StaticSystem::new(prep, ctx.loads, &zero, false, ctx.solver, 1.0)?;
            newton::solve_linear(&mut sys)?;
            ctx.out.frames.push(frame_of(&sys, 0.0, 1)?);
            ctx.out.increments = 1;
            ctx.out.iterations = 1;
            return Ok(());
        }
        let mut sys = StaticSystem::new(prep, ctx.loads, ctx.state, false, ctx.solver, ctrl.period)?;
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            // frame 0: tillståndet vid stegets början
            sys.begin_increment(0.0, 0.0)?;
            sys.u.clone_from(&sys.u_committed);
            sys.evaluate(false)?;
            frames.push(frame_of(&sys, 0.0, 0)?);
        }
        let mut inc = 0;
        let period = ctrl.period;
        let verbose = ctx.opts.verbose;
        let rep = newton::run(&mut sys, &ctrl, &mut |s: &mut StaticSystem, t| {
            inc += 1;
            if verbose {
                eprintln!("  inkrement {inc}: stegtid {:.4}", t * period);
            }
            if all || t >= 1.0 - 1e-12 {
                frames.push(frame_of(s, t * period, inc)?);
            }
            Ok(())
        })?;
        ctx.out.frames.extend(frames);
        ctx.out.increments = rep.increments;
        ctx.out.iterations = rep.iterations;
        sys.store(ctx.state);
        ctx.state.time += period;
        Ok(())
    }
}
