//! Gränssnitt för interaktioner och villkor: kontakt, tie, coupling, MPC,
//! *EQUATION, kopplingselement, fjädrar och punktmassor.
//!
//! En interaktion byggs från ett eller flera sparade modellnyckelord
//! (`FeModel::extra`) av en registrerad `InteractionFactory`. Den kan bidra på
//! tre sätt:
//! - `linear_constraints`: linjära villkor Σ cᵢ uᵢ = r (tie, MPC, *EQUATION).
//!   Kärnan eliminerar dem i `fem/mpc.rs` (inte implementerat ännu).
//! - `contribute`: inre kraft och tangent (straff, kontakt, fjädrar).
//! - `mass`: massbidrag (punktmassor) för dynamik och egenfrekvenser.

use crate::fem::dof::{Dof, NodeVec};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

/// Σ cᵢ u(nodᵢ, dofᵢ) = rhs
#[derive(Debug, Clone, Default)]
pub struct LinearConstraint {
    pub terms: Vec<(usize, Dof, f64)>,
    pub rhs: f64,
    /// Första termen elimineras även om den har ett randvillkor; randvillkoret
    /// ignoreras då (som för slavnoder i *TIE).
    pub force_slave: bool,
}

/// Mottagare för tangentbidrag mellan (nod, frihetsgrad)-par.
pub trait TangentSink {
    fn add(&mut self, node_a: usize, dof_a: Dof, node_b: usize, dof_b: Dof, v: f64);

    /// Matrisen lagras osymmetriskt (osymmetrisk lösare, t.ex. *STEP, UNSYMM=YES).
    fn is_unsymmetric(&self) -> bool {
        false
    }

    /// Osymmetriskt bidrag till K[(a, dof_a), (b, dof_b)] (bara den raden/kolumnen).
    /// Standard: ignoreras (symmetriska lösare använder den symmetriska tangenten).
    fn add_unsym(&mut self, _node_a: usize, _dof_a: Dof, _node_b: usize, _dof_b: Dof, _v: f64) {}
}

pub struct InteractionCtx<'a> {
    pub model: &'a FeModel,
    pub time: f64,
    pub dt: f64,
    pub nlgeom: bool,
}

pub trait Interaction: Send + Sync {
    fn name(&self) -> String;

    /// Frihetsgrader som interaktionen behöver (läggs till nodernas aktiva).
    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        Vec::new()
    }

    /// Nodgrupper som kopplas ihop i styvhetsmatrisen.
    fn connectivity(&self) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn linear_constraints(&self) -> Result<Vec<LinearConstraint>> {
        Ok(Vec::new())
    }

    /// Villkoren beror på läget (t.ex. stela kroppar med ändliga rotationer).
    /// NLGEOM-steget linjäriserar dem då på nytt i varje iteration med
    /// `linear_constraints_at`. Standard: nej.
    fn constraints_depend_on_u(&self) -> bool {
        false
    }

    /// Villkoren linjäriserade i läget `u`: samma termer (och slavar) som
    /// `linear_constraints`, koefficienterna är derivator mot frihetsgraderna
    /// (rumsliga spinn för rotationer) och rhs är vald så att slavens uttryck
    /// u_s = rhs' + Σ c' u_m ger dess exakta läge i `u`. Standard: de linjära
    /// villkoren.
    fn linear_constraints_at(&self, _u: &[NodeVec], _nlgeom: bool) -> Result<Vec<LinearConstraint>> {
        self.linear_constraints()
    }

    /// Lägger till inre kraft i `fint` och (om `k` ges) tangenten.
    fn contribute(&self, _ctx: &InteractionCtx, _u: &[NodeVec], _fint: &mut [NodeVec], _k: Option<&mut dyn TangentSink>) -> Result<()> {
        Ok(())
    }

    /// Massbidrag (nod, frihetsgrad, nod, frihetsgrad, värde).
    fn mass(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        Vec::new()
    }

    /// Dämpningsbidrag (*DASHPOT): (nod, frihetsgrad, nod, frihetsgrad, värde).
    fn damping(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        Vec::new()
    }

    /// Interaktionen är linjär (bidraget beror linjärt på u och har inget
    /// tillstånd), så att linjära steg och perturbationssteg kan använda den.
    fn is_linear(&self) -> bool {
        false
    }

    /// Godkänner tillståndet från senaste `contribute` (friktion, kontaktstatus).
    /// Anropas när ett inkrement har konvergerat.
    fn commit(&self) {}

    /// Nodutdata efter senaste godkända inkrement (t.ex. CPRESS, COPEN).
    fn node_output(&self) -> Vec<NodeOutput> {
        Vec::new()
    }

    /// Elementutdata efter senaste godkända inkrement (t.ex. kopplingselementens
    /// CTF, CU): (elementindex, fält). Läggs till elementets egna utdata.
    fn elem_output(&self) -> Vec<(usize, crate::elements::IpField)> {
        Vec::new()
    }

    /// Den okondenserade inre och yttre kraften (före elimineringen av de
    /// linjära villkoren) och slavarna efter en utvärdering, så att en
    /// interaktion kan räkna ut sina villkorskrafter (kopplingselementens
    /// CTF/CTM i låsta komponenter). Anropas av `fem::system::StaticSystem`.
    fn observe_residual(&self, _fint: &[NodeVec], _fext: &[NodeVec], _slaves: &[crate::fem::mpc::Slave]) {}

    /// Kinematiska villkor för explicit dynamik (som inte eliminerar linjära
    /// villkor): per villkor raden c (mot nodfrihetsgraderna, rotationer som
    /// rumsliga spinn) och värdet g(u), som ska vara 0. Villkorskraften λ ger
    /// den inre kraften cᵀλ (`steps/explicit_constraints.rs`). Standard: inga.
    fn kinematic_constraints(&self, _u: &[NodeVec], _nlgeom: bool) -> Vec<(Vec<(usize, Dof, f64)>, f64)> {
        Vec::new()
    }

    /// Villkorskrafterna λ för `kinematic_constraints` (samma ordning), för utdata.
    fn set_constraint_forces(&self, _lambda: &[f64]) {}
}

/// Ett nodfält från en interaktion: värden för vissa noder.
#[derive(Debug, Clone, Default)]
pub struct NodeOutput {
    pub name: String,
    pub labels: Vec<String>,
    pub values: Vec<(usize, Vec<f64>)>,
}

/// Bygger interaktioner från sparade nyckelord.
pub struct InteractionFactory {
    /// Nyckelorden (utan '*', versaler) som fabriken hanterar.
    pub keywords: &'static [&'static str],
    pub build: fn(&FeModel, &RawKeyword) -> Result<Vec<Box<dyn Interaction>>>,
}
