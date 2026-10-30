//! The step framework: a sequence of steps (*STEP) with inherited state.
//!
//! Each step type is its own module that exports
//! `pub static STEP_TYPES: &[&dyn StepType]`. Register a new module with ONE
//! line in the list below (alphabetical order). `newton.rs` is the shared
//! Newton–Raphson driver.

mod api;
pub mod newton;
pub use api::*;

use crate::fem::assembly::Prepared;
use crate::fem::dof;
use crate::fem::model::FeModel;
use crate::fem::mpc;
use crate::fem::output::{AnalysisOutput, StepOutput};
use crate::fem::state::{SolutionState, StepLoads};
use crate::model::{fail, FemError, Result};

macro_rules! step_modules {
    ($($m:ident),* $(,)?) => {
        $(pub mod $m;)*
        /// All registered step types.
        pub fn all() -> Vec<&'static dyn StepType> {
            let mut v: Vec<&'static dyn StepType> = Vec::new();
            $(v.extend($m::STEP_TYPES.iter().copied());)*
            v
        }
    };
}

// One line per step type, in alphabetical order.
step_modules! {
    acoustic,
    adiabatic,
    anneal,
    buckle,
    complex_frequency,
    direct_cyclic,
    dynamic,
    electromagnetic,
    explicit,
    explicit_coupled,
    fracture,
    frequency,
    heat,
    heat_cavity,
    heat_convection,
    heat_coupled,
    heat_stress,
    mass_diffusion,
    matrix_generate,
    modal,
    model_change,
    nlgeom,
    nlgeom_riks,
    piezo,
    random_response,
    restart,
    soils,
    static_linear,
    static_stabilize,
    submodel,
    subspace,
    substructure,
    thermal_electric,
    visco,
    xfem_contour,
}

/// Runs all steps in the model in sequence. Returns the output and the final state.
pub fn run(model: &FeModel, opts: &RunOptions) -> Result<(AnalysisOutput, SolutionState)> {
    if model.steps.is_empty() {
        return fail("the model has no steps (*STEP ... *END STEP)");
    }
    if let Some(s) = &opts.settings {
        crate::settings::set(s.clone());
    }
    // submodeling: *BOUNDARY, SUBMODEL gets values from the global model (steps/submodel.rs)
    let submodel = crate::steps::submodel::prepare(model)?;
    let model = submodel.as_ref().unwrap_or(model);
    newton::reset_time_average();
    // the user routines' execution context and the field variables (user/context.rs)
    let _user_ctx = crate::user::context::begin_analysis(model)?;
    let prep = Prepared::new(model)?;
    mpc::check(&prep)?;
    let solver = crate::solver::select(opts.solver.as_deref())?;
    crate::steps::model_change::begin_analysis(&prep)?;
    let mut state = SolutionState::new(model);
    state.elem_state = prep.init_states()?;
    crate::materials::initial_state::apply(&prep, &mut state)?;
    crate::materials::porous_metal::apply_relative_density(&prep, &mut state)?; // area materials5
    for b in &model.initial_bcs {
        if b.op_new {
            state.bcs.clear();
        }
        for &(n, d, v) in &b.items {
            let s = dof::slot_of(d);
            if prep.active[n] & (1 << s) != 0 {
                state.bcs.insert((n, s), v);
                state.u[n][s] = v;
            }
        }
    }
    // *RESTART, READ: the state from the restart file (steps/restart.rs)
    let mut restart = crate::steps::restart::before_run(model, &prep, &mut state)?;
    let types = all();
    let mut out = AnalysisOutput::default();
    for (i, step) in model.steps.iter().enumerate() {
        if i < restart.skip {
            continue;
        }
        let nlgeom = if step.perturbation { state.nlgeom } else { step.nlgeom.unwrap_or(state.nlgeom) };
        let wrap = |e: FemError| FemError(format!("step {} ({}): {}", i + 1, step.name, e.0));
        let chosen = types.iter().filter_map(|t| t.claim_model(step, nlgeom, model).map(|p| (p, *t))).max_by_key(|(p, _)| *p);
        let Some((_, ty)) = chosen else {
            let what = if step.procedure_name().is_empty() { "without procedure".to_string() } else { format!("*{}", step.procedure_name()) };
            let nl = if nlgeom { " with NLGEOM=YES" } else { "" };
            return Err(wrap(FemError(format!("the step type {what}{nl} is not supported yet (see src/steps/)"))));
        };
        let (loads, prop) = StepLoads::build_with(model, step, &state, &prep.active, ty.uses_amplitudes()).map_err(wrap)?;
        if opts.verbose {
            for w in &prop.warnings {
                eprintln!("warning: {w}");
            }
        }
        if opts.verbose {
            eprintln!("step {} ({}): {}", i + 1, step.name, ty.name());
        }
        // *STATIC, SOLVER=ITERATIVE and *STEP, UNSYMM=YES select the solver per step
        let solver = crate::solver::iterative::for_step(step, solver, opts.solver.is_some() || crate::settings::with(|s| s.solver.is_some())).map_err(wrap)?;
        let solver = crate::materials::principal::auto_unsymmetric(step, &prep, solver, opts.solver.is_some() || crate::settings::with(|s| s.solver.is_some()));
        // C3D4H (nodal pressure degrees of freedom): saddle point system, indefinite solver (elements/hybrid_tet.rs)
        let solver = crate::elements::hybrid_tet::auto_solver(model, solver, opts.solver.is_some() || crate::settings::with(|s| s.solver.is_some()) || step.header.has("SOLVER") || step.header.has("UNSYMM"));
        // area cavity: fluid-filled cavities give a saddle point system (interactions/cavity.rs)
        let solver = crate::interactions::cavity::auto_solver(model, solver, opts.solver.is_some() || crate::settings::with(|s| s.solver.is_some()) || step.header.has("SOLVER") || step.header.has("UNSYMM"));
        let base = step.perturbation.then(|| state.clone());
        // *MODEL CHANGE (steps/model_change.rs)
        crate::steps::model_change::begin_step(&prep, step, &mut state).map_err(wrap)?;
        if !step.perturbation {
            state.nlgeom = nlgeom;
        }
        // time-dependent materials (creep, viscoelasticity) are activated per procedure
        crate::materials::time_dependence::begin_step(model, &prep.materials, step, nlgeom);
        crate::progress::emit(crate::progress::Event::Step { index: i, name: &step.name, procedure: step.procedure_name() }).map_err(wrap)?;
        crate::user::context::begin_step(i + 1, state.time, crate::user::context::step_period(step), crate::user::context::procedure_code(step, nlgeom), nlgeom, step.perturbation);
        let mut so = StepOutput { name: step.name.clone(), procedure: format!("*{}", step.procedure_name()), ..Default::default() };
        {
            let mut ctx = StepCtx { prep: &prep, step, index: i, loads: &loads, state: &mut state, nlgeom, out: &mut so, opts, solver };
            ty.run(&mut ctx).map_err(wrap)?;
        }
        state.vectors.remove(crate::materials::initial_state::UNBALANCE);
        let constrained = loads.bc.keys().filter(|&&(n, s)| prep.active[n] & (1 << s) != 0).count();
        out.n_eq = prep.active.iter().map(|a| a.count_ones() as usize).sum::<usize>() - constrained;
        crate::interactions::cavity::step_history(model, &mut so);
        out.steps.push(so);
        match base {
            Some(b) => state = b,
            None => {
                state.bcs = prop.bcs;
                // *BOUNDARY, USER: the value DISP gave at the end of the step is inherited (user/loads.rs)
                for b in &loads.user_bc {
                    state.bcs.insert((b.node, b.slot), state.u[b.node][b.slot]);
                }
                state.cloads = prop.cloads;
                state.dloads = prop.dloads;
            }
        }
        restart.after_step(model, i, &state, out.steps.last().unwrap()).map_err(wrap)?;
    }
    Ok((out, state))
}
