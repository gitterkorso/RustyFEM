//! Direct solvers for indefinite and unsymmetric systems.
//!
//! - `indefinite`: multifrontal LDLᵀ with 1×1/2×2 pivoting and delayed
//!   pivots (`sparse::indef`). Handles symmetric matrices that are not
//!   positive definite (Lagrange multipliers, contact, instability,
//!   shifted eigenvalue problems) and computes the inertia. If the matrix has negative
//!   pivots, a warning about negative eigenvalues is printed.
//! - `unsymmetric`: multifrontal LU with threshold pivoting (`sparse::lu`) for
//!   unsymmetric tangents (friction, follower loads). Selected automatically for
//!   steps with `*STEP, UNSYMM=YES`.
//!
//! Both use the same ordering (nested dissection) as `direct` and perform up
//! to two steps of iterative refinement if the residual is large.

use super::api::{Graph, Inertia, LinearSolver, SystemMatrix};
use crate::model::Result;
use crate::sparse::indef::{self, IndefFactor};
use crate::sparse::lu::{self, Full, LuFactor};
use crate::sparse::{self, Lower, Symbolic};

pub struct Indefinite;
pub struct Unsymmetric;

pub static INDEFINITE: Indefinite = Indefinite;
pub static UNSYMMETRIC: Unsymmetric = Unsymmetric;

pub static SOLVERS: &[&dyn LinearSolver] = &[&INDEFINITE, &UNSYMMETRIC];

fn timing() -> bool {
    crate::settings::with(|s| s.timing)
}

fn analyse(g: &Graph) -> Symbolic {
    let t0 = std::time::Instant::now();
    let sym = sparse::analyse(&g.coords, &g.adj, &g.ndof);
    if timing() {
        eprintln!("analysis: {:.3} s  flops: {:.2e}  largest front: {:?}", t0.elapsed().as_secs_f64(), sym.work(), sym.largest_front());
    }
    sym
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Up to two steps of iterative refinement x ← x + A⁻¹(b − A x).
fn refine(b: &[f64], x: &mut [f64], matvec: impl Fn(&[f64], &mut [f64]), solve: impl Fn(&mut [f64])) {
    let bn = norm(b);
    if bn == 0.0 {
        return;
    }
    let mut ax = vec![0.0; b.len()];
    for _ in 0..2 {
        matvec(x, &mut ax);
        let mut r: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        if norm(&r) <= 1e-13 * bn {
            return;
        }
        solve(&mut r);
        for (xi, ri) in x.iter_mut().zip(&r) {
            *xi += ri;
        }
    }
}

/// Warning that the system matrix has N negative eigenvalues.
pub fn warn_negative(inertia: &Inertia) {
    if inertia.neg > 0 && !crate::settings::with(|s| s.quiet) {
        eprintln!("warning: the system matrix has {} negative eigenvalues (indefinite)", inertia.neg);
    }
}

struct IndefMatrix {
    sym: Symbolic,
    k: Lower,
    fac: Option<IndefFactor>,
    /// No warning about negative eigenvalues (the caller reports the inertia itself).
    quiet: bool,
}

/// Indefinite matrix without the negative-eigenvalue warning at every
/// factorization. Used by NLGEOM/Riks (`steps/nlgeom_solver.rs`), which
/// reports the number of negative eigenvalues per increment instead.
pub fn prepare_quiet(g: &Graph) -> Result<Box<dyn SystemMatrix>> {
    let sym = analyse(g);
    let k = sym.pattern(&g.adj, &g.ndof);
    Ok(Box::new(IndefMatrix { sym, k, fac: None, quiet: true }))
}

impl LinearSolver for Indefinite {
    fn name(&self) -> &'static str {
        "indefinite"
    }

    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>> {
        let sym = analyse(g);
        let k = sym.pattern(&g.adj, &g.ndof);
        Ok(Box::new(IndefMatrix { sym, k, fac: None, quiet: false }))
    }
}

impl SystemMatrix for IndefMatrix {
    fn n(&self) -> usize {
        self.sym.n
    }
    fn first_dof(&self) -> &[usize] {
        &self.sym.first_dof
    }
    fn zero(&mut self) {
        self.k.val.iter_mut().for_each(|v| *v = 0.0);
        self.fac = None;
    }
    #[inline]
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.k.add(r, c, v);
    }
    fn factor(&mut self) -> std::result::Result<(), usize> {
        let t0 = std::time::Instant::now();
        let f = indef::factor(&self.sym, &self.k)?;
        if timing() {
            eprintln!(
                "indefinite factorization: {:.3} s  inertia {:?}  delayed pivots {}  2×2 blocks {}",
                t0.elapsed().as_secs_f64(),
                f.inertia,
                f.delayed,
                f.two_by_two
            );
        }
        if !self.quiet {
            warn_negative(&f.inertia);
        }
        self.fac = Some(f);
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        let f = self.fac.as_ref().expect("internal: solve before factor");
        let rhs = b.to_vec();
        f.solve(b);
        refine(&rhs, b, |x, y| indef::sym_matvec(&self.k, x, y), |r| f.solve(r));
    }
    fn inertia(&self) -> Option<Inertia> {
        self.fac.as_ref().map(|f| f.inertia)
    }
}

struct LuMatrix {
    sym: Symbolic,
    k: Full,
    fac: Option<LuFactor>,
}

impl LinearSolver for Unsymmetric {
    fn name(&self) -> &'static str {
        "unsymmetric"
    }

    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>> {
        let sym = analyse(g);
        let k = Full::from_pattern(sym.pattern(&g.adj, &g.ndof));
        Ok(Box::new(LuMatrix { sym, k, fac: None }))
    }
}

impl SystemMatrix for LuMatrix {
    fn n(&self) -> usize {
        self.sym.n
    }
    fn first_dof(&self) -> &[usize] {
        &self.sym.first_dof
    }
    fn zero(&mut self) {
        self.k.zero();
        self.fac = None;
    }
    /// Symmetric contribution: (r, c) and (c, r).
    #[inline]
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.k.lower.add(r, c, v);
        if r != c {
            self.k.upper_t.add(r, c, v);
        }
    }
    #[inline]
    fn add_unsym(&mut self, r: usize, c: usize, v: f64) {
        self.k.add(r, c, v);
    }
    fn is_symmetric(&self) -> bool {
        false
    }
    fn factor(&mut self) -> std::result::Result<(), usize> {
        let t0 = std::time::Instant::now();
        let f = lu::factor(&self.sym, &self.k)?;
        if timing() {
            eprintln!("LU factorization: {:.3} s  delayed pivots {}", t0.elapsed().as_secs_f64(), f.delayed);
        }
        self.fac = Some(f);
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        let f = self.fac.as_ref().expect("internal: solve before factor");
        let rhs = b.to_vec();
        f.solve(b);
        refine(&rhs, b, |x, y| self.k.matvec(x, y), |r| f.solve(r));
    }
}
