//! Interface for linear equation solvers.
//!
//! The core describes the system structure as a graph of "blocks" (nodes with
//! free degrees of freedom). The solver chooses the equation ordering (`first_dof`),
//! receives the lower triangle of a symmetric matrix and solves K x = b.
//! The same matrix object is reused between Newton iterations: `zero`,
//! assembly with `add`, `factor`, `solve`.

use crate::model::Result;

/// System structure: one block per node with free degrees of freedom.
pub struct Graph {
    /// Block coordinates (for geometric ordering).
    pub coords: Vec<[f64; 3]>,
    /// Neighbour blocks (excluding the block itself), sorted.
    pub adj: Vec<Vec<usize>>,
    /// Number of equations in each block.
    pub ndof: Vec<usize>,
}

pub trait LinearSolver: Send + Sync {
    /// Name for selection and printing, e.g. "direct".
    fn name(&self) -> &'static str;
    /// Symbolic analysis: returns a zeroed matrix to assemble into.
    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>>;
}

pub trait SystemMatrix: Send {
    /// Number of equations.
    fn n(&self) -> usize;
    /// Equation number of the first degree of freedom in each block; the block's other
    /// degrees of freedom follow directly after.
    fn first_dof(&self) -> &[usize];
    fn zero(&mut self);
    /// Adds `v` to (r, c) in the lower triangle (r ≥ c).
    fn add(&mut self, r: usize, c: usize, v: f64);
    /// Factorizes. `Err(equation)` if the matrix is singular or not
    /// positive definite at that equation.
    fn factor(&mut self) -> std::result::Result<(), usize>;
    /// Solves K x = b in place (after `factor`).
    fn solve(&self, b: &mut [f64]);

    /// Adds `v` to (r, c) in an unsymmetric matrix (r and c arbitrary).
    /// `add(r, c, v)` always means a symmetric contribution (also (c, r));
    /// a symmetric solver keeps only the lower triangle.
    fn add_unsym(&mut self, r: usize, c: usize, v: f64) {
        if r >= c {
            self.add(r, c, v);
        }
    }
    /// `false` if the solver stores and solves an unsymmetric matrix.
    fn is_symmetric(&self) -> bool {
        true
    }
    /// Inertia (number of negative, zero and positive pivots) after `factor`,
    /// if the solver computes it. Gives the warning about negative eigenvalues.
    fn inertia(&self) -> Option<Inertia> {
        None
    }
}

/// Inertia of a factorized symmetric matrix (Sylvester's law of inertia).
pub use crate::sparse::indef::Inertia;
