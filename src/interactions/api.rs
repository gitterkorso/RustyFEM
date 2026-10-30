//! Interface for interactions and constraints: contact, tie, coupling, MPC,
//! *EQUATION, connector elements, springs and point masses.
//!
//! An interaction is built from one or more stored model keywords
//! (`FeModel::extra`) by a registered `InteractionFactory`. It can contribute
//! in three ways:
//! - `linear_constraints`: linear constraints Σ cᵢ uᵢ = r (tie, MPC, *EQUATION).
//!   The core eliminates them in `fem/mpc.rs` (not implemented yet).
//! - `contribute`: internal force and tangent (penalty, contact, springs).
//! - `mass`: mass contributions (point masses) for dynamics and eigenfrequencies.

use crate::fem::dof::{Dof, NodeVec};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

/// Σ cᵢ u(nodeᵢ, dofᵢ) = rhs
#[derive(Debug, Clone, Default)]
pub struct LinearConstraint {
    pub terms: Vec<(usize, Dof, f64)>,
    pub rhs: f64,
    /// The first term is eliminated even if it has a boundary condition; the
    /// boundary condition is then ignored (as for slave nodes in *TIE).
    pub force_slave: bool,
}

/// Receiver of tangent contributions between (node, DOF) pairs.
pub trait TangentSink {
    fn add(&mut self, node_a: usize, dof_a: Dof, node_b: usize, dof_b: Dof, v: f64);

    /// The matrix is stored unsymmetric (unsymmetric solver, e.g. *STEP, UNSYMM=YES).
    fn is_unsymmetric(&self) -> bool {
        false
    }

    /// Unsymmetric contribution to K[(a, dof_a), (b, dof_b)] (only that row/column).
    /// Default: ignored (symmetric solvers use the symmetric tangent).
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

    /// DOFs the interaction needs (added to the nodes' active DOFs).
    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        Vec::new()
    }

    /// Node groups that are coupled in the stiffness matrix.
    fn connectivity(&self) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn linear_constraints(&self) -> Result<Vec<LinearConstraint>> {
        Ok(Vec::new())
    }

    /// The constraints depend on the configuration (e.g. rigid bodies with finite
    /// rotations). The NLGEOM step then relinearizes them in every iteration with
    /// `linear_constraints_at`. Default: no.
    fn constraints_depend_on_u(&self) -> bool {
        false
    }

    /// The constraints linearized in configuration `u`: the same terms (and slaves)
    /// as `linear_constraints`, the coefficients are derivatives with respect to the
    /// DOFs (spatial spins for rotations) and rhs is chosen so that the slave's
    /// expression u_s = rhs' + Σ c' u_m gives its exact position in `u`. Default:
    /// the linear constraints.
    fn linear_constraints_at(&self, _u: &[NodeVec], _nlgeom: bool) -> Result<Vec<LinearConstraint>> {
        self.linear_constraints()
    }

    /// Adds internal force to `fint` and (if `k` is given) the tangent.
    fn contribute(&self, _ctx: &InteractionCtx, _u: &[NodeVec], _fint: &mut [NodeVec], _k: Option<&mut dyn TangentSink>) -> Result<()> {
        Ok(())
    }

    /// Mass contributions (node, DOF, node, DOF, value).
    fn mass(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        Vec::new()
    }

    /// Damping contributions (*DASHPOT): (node, DOF, node, DOF, value).
    fn damping(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        Vec::new()
    }

    /// The interaction is linear (the contribution depends linearly on u and has
    /// no state), so that linear steps and perturbation steps can use it.
    fn is_linear(&self) -> bool {
        false
    }

    /// Commits the state from the latest `contribute` (friction, contact status).
    /// Called when an increment has converged.
    fn commit(&self) {}

    /// Nodal output after the latest converged increment (e.g. CPRESS, COPEN).
    fn node_output(&self) -> Vec<NodeOutput> {
        Vec::new()
    }

    /// Element output after the latest converged increment (e.g. the connector
    /// elements' CTF, CU): (element index, field). Added to the element's own output.
    fn elem_output(&self) -> Vec<(usize, crate::elements::IpField)> {
        Vec::new()
    }

    /// The uncondensed internal and external force (before elimination of the
    /// linear constraints) and the slaves after an evaluation, so that an
    /// interaction can compute its constraint forces (the connector elements'
    /// CTF/CTM in constrained components). Called by `fem::system::StaticSystem`.
    fn observe_residual(&self, _fint: &[NodeVec], _fext: &[NodeVec], _slaves: &[crate::fem::mpc::Slave]) {}

    /// Kinematic constraints for explicit dynamics (which does not eliminate linear
    /// constraints): per constraint the row c (against the nodal DOFs, rotations as
    /// spatial spins) and the value g(u), which must be 0. The constraint force λ
    /// gives the internal force cᵀλ (`steps/explicit_constraints.rs`). Default: none.
    fn kinematic_constraints(&self, _u: &[NodeVec], _nlgeom: bool) -> Vec<(Vec<(usize, Dof, f64)>, f64)> {
        Vec::new()
    }

    /// The constraint forces λ for `kinematic_constraints` (same order), for output.
    fn set_constraint_forces(&self, _lambda: &[f64]) {}
}

/// A nodal field from an interaction: values for some nodes.
#[derive(Debug, Clone, Default)]
pub struct NodeOutput {
    pub name: String,
    pub labels: Vec<String>,
    pub values: Vec<(usize, Vec<f64>)>,
}

/// Builds interactions from stored keywords.
pub struct InteractionFactory {
    /// The keywords (without '*', upper case) that the factory handles.
    pub keywords: &'static [&'static str],
    pub build: fn(&FeModel, &RawKeyword) -> Result<Vec<Box<dyn Interaction>>>,
}
