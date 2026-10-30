//! Gränssnitt för steg-typer (procedurer).
//!
//! En steg-typ gör anspråk på ett `StepDef` (via `claim`) och kör det med
//! `run`. Bland typer som gör anspråk väljs den med högst prioritet, så att en
//! specialiserad typ (t.ex. NLGEOM) kan ta över från en allmän.

use crate::fem::assembly::Prepared;
use crate::fem::model::StepDef;
use crate::fem::output::StepOutput;
use crate::fem::state::{SolutionState, StepLoads};
use crate::model::Result;
use crate::solver::LinearSolver;

/// Inställningar för en hel analys.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Spara en frame för varje inkrement (annars bara stegets sista).
    pub all_frames: bool,
    /// Skriv förlopp till stderr.
    pub verbose: bool,
    /// Linjär lösare (standard: `Settings::solver` eller "direct").
    pub solver: Option<String>,
    /// Inställningar för processen (annars miljövariablerna i CLI:t, se `crate::settings`).
    pub settings: Option<crate::settings::Settings>,
}

/// Allt en steg-typ behöver för att köra ett step.
pub struct StepCtx<'a, 'm> {
    pub prep: &'a Prepared<'m>,
    pub step: &'a StepDef,
    /// Stegets nummer (0-baserat).
    pub index: usize,
    /// Randvillkor och laster under steget (ärvda + stegets egna).
    pub loads: &'a StepLoads,
    /// Tillståndet före steget; steg-typen skriver tillståndet efter steget.
    /// Vid perturbationssteg återställs det automatiskt efteråt.
    pub state: &'a mut SolutionState,
    /// NLGEOM gäller i steget (ärvt eller angivet).
    pub nlgeom: bool,
    pub out: &'a mut StepOutput,
    pub opts: &'a RunOptions,
    pub solver: &'static dyn LinearSolver,
}

pub trait StepType: Send + Sync {
    /// Namn för meddelanden, t.ex. "STATIC".
    fn name(&self) -> &'static str;
    /// Prioritet om typen kan köra steget, annars `None`.
    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32>;
    /// Som `claim` men med modellen (t.ex. för typer som bara gäller modeller
    /// med vissa element). Standard: `claim`.
    fn claim_model(&self, step: &StepDef, nlgeom: bool, _model: &crate::fem::model::FeModel) -> Option<i32> {
        self.claim(step, nlgeom)
    }
    fn run(&self, ctx: &mut StepCtx) -> Result<()>;
    /// Typen utvärderar själv AMPLITUDE= på randvillkor och laster
    /// (`StepLoads::amp_bc` m.fl.). Annars ger AMPLITUDE= ett fel.
    fn uses_amplitudes(&self) -> bool {
        false
    }
}
