//! Interface for step types (procedures).
//!
//! A step type claims a `StepDef` (via `claim`) and runs it with
//! `run`. Among the types making a claim, the one with the highest priority is chosen, so that a
//! specialized type (e.g. NLGEOM) can take over from a general one.

use crate::fem::assembly::Prepared;
use crate::fem::model::StepDef;
use crate::fem::output::StepOutput;
use crate::fem::state::{SolutionState, StepLoads};
use crate::model::Result;
use crate::solver::LinearSolver;

/// Settings for a whole analysis.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Save a frame for each increment (otherwise only the step's last).
    pub all_frames: bool,
    /// Write progress to stderr.
    pub verbose: bool,
    /// Linear solver (default: `Settings::solver` or "direct").
    pub solver: Option<String>,
    /// Settings for the process (otherwise the environment variables in the CLI, see `crate::settings`).
    pub settings: Option<crate::settings::Settings>,
}

/// Everything a step type needs to run a step.
pub struct StepCtx<'a, 'm> {
    pub prep: &'a Prepared<'m>,
    pub step: &'a StepDef,
    /// The step number (0-based).
    pub index: usize,
    /// Boundary conditions and loads during the step (inherited + the step's own).
    pub loads: &'a StepLoads,
    /// The state before the step; the step type writes the state after the step.
    /// For perturbation steps it is restored automatically afterwards.
    pub state: &'a mut SolutionState,
    /// NLGEOM applies in the step (inherited or specified).
    pub nlgeom: bool,
    pub out: &'a mut StepOutput,
    pub opts: &'a RunOptions,
    pub solver: &'static dyn LinearSolver,
}

pub trait StepType: Send + Sync {
    /// Name for messages, e.g. "STATIC".
    fn name(&self) -> &'static str;
    /// Priority if the type can run the step, otherwise `None`.
    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32>;
    /// Like `claim` but with the model (e.g. for types that only apply to models
    /// with certain elements). Default: `claim`.
    fn claim_model(&self, step: &StepDef, nlgeom: bool, _model: &crate::fem::model::FeModel) -> Option<i32> {
        self.claim(step, nlgeom)
    }
    fn run(&self, ctx: &mut StepCtx) -> Result<()>;
    /// The type itself evaluates AMPLITUDE= on boundary conditions and loads
    /// (`StepLoads::amp_bc` etc.). Otherwise AMPLITUDE= gives an error.
    fn uses_amplitudes(&self) -> bool {
        false
    }
}
