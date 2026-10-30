//! Iterative solver for very large, positive definite models:
//! preconditioned conjugate gradient (PCG).
//!
//! Solvers (select with `--solver`, `RUSTYFEM_SOLVER` or `*STATIC, SOLVER=ITERATIVE`):
//! - `iterative`: PCG with algebraic multigrid, smoothed aggregation (`amg.rs`),
//! - `iterative-ic0`: PCG with incomplete Cholesky IC(0),
//! - `iterative-jacobi`: PCG with diagonal preconditioning.
//!
//! Equations are numbered in node order (no fill-in is needed). The tolerance is
//! ‖b − K x‖ ≤ tol·‖b‖ with tol = 10⁻¹⁰ (change with `RUSTYFEM_ITER_TOL`).
//! If PCG does not converge (e.g. indefinite matrix), the system is solved
//! directly instead, with a warning; a mechanism is then reported as for `direct`.

mod amg;
pub mod csr;

use super::api::{Graph, LinearSolver, SystemMatrix};
use crate::fem::model::StepDef;
use crate::model::Result;
use crate::sparse::{self, Factor, Lower, Symbolic};
use csr::{pcg, Csr, Ic0, Jacobi, PcgReport, Precond};
use std::sync::Mutex;

pub use amg::rigid_body_modes;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecondKind {
    Amg,
    Ic0,
    Jacobi,
}

pub struct Iterative(pub PrecondKind);

pub static ITERATIVE_AMG: Iterative = Iterative(PrecondKind::Amg);
pub static ITERATIVE_IC0: Iterative = Iterative(PrecondKind::Ic0);
pub static ITERATIVE_JACOBI: Iterative = Iterative(PrecondKind::Jacobi);

pub static SOLVERS: &[&dyn LinearSolver] = &[&ITERATIVE_AMG, &ITERATIVE_IC0, &ITERATIVE_JACOBI];

/// Default tolerance for the relative residual.
pub const TOL: f64 = 1e-10;

fn timing() -> bool {
    crate::settings::with(|s| s.timing)
}

/// The solver for a step: `*STATIC, SOLVER=ITERATIVE` (or `*STEP, SOLVER=ITERATIVE`)
/// gives `iterative`, `*STEP, UNSYMM=YES` gives `unsymmetric`. A solver chosen
/// explicitly (`explicit`: `--solver` or `RUSTYFEM_SOLVER`) takes precedence.
pub fn for_step(step: &StepDef, default: &'static dyn LinearSolver, explicit: bool) -> Result<&'static dyn LinearSolver> {
    if explicit {
        return Ok(default);
    }
    let solver_param = step
        .procedure
        .as_ref()
        .and_then(|p| p.param_upper("SOLVER"))
        .or_else(|| step.header.param_upper("SOLVER"));
    if let Some(s) = solver_param {
        match s.as_str() {
            "ITERATIVE" => return Ok(&ITERATIVE_AMG),
            "DIRECT" => {}
            other => return crate::model::fail(format!("SOLVER={other} must be DIRECT or ITERATIVE")),
        }
    }
    if step.header.has("UNSYMM") && step.header.flag("UNSYMM", false)? {
        return Ok(&super::indefinite::UNSYMMETRIC);
    }
    Ok(default)
}

/// Lower triangle in node order (equations are numbered block by block).
pub fn natural_pattern(adj: &[Vec<usize>], ndof: &[usize]) -> (Vec<usize>, Lower) {
    let nb = ndof.len();
    let mut first = vec![0usize; nb];
    let mut n = 0;
    for v in 0..nb {
        first[v] = n;
        n += ndof[v];
    }
    let mut colptr = vec![0usize; n + 1];
    let mut rowidx = Vec::new();
    for v in 0..nb {
        let mut nbrs: Vec<usize> = adj[v].iter().copied().filter(|&w| w > v).collect();
        nbrs.sort_unstable();
        for c in first[v]..first[v] + ndof[v] {
            rowidx.extend(c..first[v] + ndof[v]);
            for &w in &nbrs {
                rowidx.extend(first[w]..first[w] + ndof[w]);
            }
            colptr[c + 1] = rowidx.len();
        }
    }
    let val = vec![0.0; rowidx.len()];
    (first, Lower { colptr, rowidx, val })
}

struct IterMatrix {
    kind: PrecondKind,
    first: Vec<usize>,
    k: Lower,
    coords: Vec<[f64; 3]>,
    adj: Vec<Vec<usize>>,
    ndof: Vec<usize>,
    a: Option<Csr>,
    pre: Option<Box<dyn Precond>>,
    /// Direct factor (fallback) and equation mapping natural → direct solver's.
    direct: Mutex<Option<(Symbolic, Factor, Vec<usize>)>>,
    tol: f64,
}

impl LinearSolver for Iterative {
    fn name(&self) -> &'static str {
        match self.0 {
            PrecondKind::Amg => "iterative",
            PrecondKind::Ic0 => "iterative-ic0",
            PrecondKind::Jacobi => "iterative-jacobi",
        }
    }

    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>> {
        let (first, k) = natural_pattern(&g.adj, &g.ndof);
        let tol = crate::settings::with(|s| s.iter_tol).unwrap_or(TOL);
        Ok(Box::new(IterMatrix {
            kind: self.0,
            first,
            k,
            coords: g.coords.clone(),
            adj: g.adj.clone(),
            ndof: g.ndof.clone(),
            a: None,
            pre: None,
            direct: Mutex::new(None),
            tol,
        }))
    }
}

impl IterMatrix {
    /// Direct factorization with the same values (fallback and mechanism diagnosis).
    /// `Err(equation)` in natural numbering.
    fn direct_factor(&self) -> std::result::Result<(Symbolic, Factor, Vec<usize>), usize> {
        let sym = sparse::analyse(&self.coords, &self.adj, &self.ndof);
        let mut map = vec![0usize; self.k.colptr.len() - 1];
        for v in 0..self.ndof.len() {
            for l in 0..self.ndof[v] {
                map[self.first[v] + l] = sym.first_dof[v] + l;
            }
        }
        let mut inv = vec![0usize; map.len()];
        for (q, &d) in map.iter().enumerate() {
            inv[d] = q;
        }
        let mut kd = sym.pattern(&self.adj, &self.ndof);
        for c in 0..map.len() {
            for p in self.k.colptr[c]..self.k.colptr[c + 1] {
                let (r, v) = (map[self.k.rowidx[p]], map[c]);
                let (r, cc) = if r >= v { (r, v) } else { (v, r) };
                kd.add(r, cc, self.k.val[p]);
            }
        }
        let f = sparse::factor(&sym, &kd).map_err(|q| inv[q])?;
        Ok((sym, f, map))
    }

    fn solve_direct(&self, b: &mut [f64]) {
        let mut guard = self.direct.lock().unwrap();
        if guard.is_none() {
            match self.direct_factor() {
                Ok(d) => *guard = Some(d),
                Err(_) => {
                    eprintln!("warning: the direct fallback solution failed (singular matrix)");
                    b.iter_mut().for_each(|v| *v = f64::NAN);
                    return;
                }
            }
        }
        let (_, f, map) = guard.as_ref().unwrap();
        let mut x = vec![0.0; b.len()];
        for (q, &d) in map.iter().enumerate() {
            x[d] = b[q];
        }
        f.solve(&mut x);
        for (q, &d) in map.iter().enumerate() {
            b[q] = x[d];
        }
    }
}

impl SystemMatrix for IterMatrix {
    fn n(&self) -> usize {
        self.k.colptr.len() - 1
    }
    fn first_dof(&self) -> &[usize] {
        &self.first
    }
    fn zero(&mut self) {
        self.k.val.iter_mut().for_each(|v| *v = 0.0);
        self.a = None;
        self.pre = None;
        *self.direct.lock().unwrap() = None;
    }
    #[inline]
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.k.add(r, c, v);
    }
    fn factor(&mut self) -> std::result::Result<(), usize> {
        let t0 = std::time::Instant::now();
        // non-positive diagonal: diagnose with the direct solver
        if (0..self.n()).any(|c| !(self.k.val[self.k.colptr[c]] > 0.0)) {
            let d = self.direct_factor()?;
            *self.direct.lock().unwrap() = Some(d);
            return Ok(());
        }
        let a = Csr::from_lower(&self.k);
        let pre: Option<Box<dyn Precond>> = match self.kind {
            PrecondKind::Jacobi => Some(Box::new(Jacobi::new(&a))),
            PrecondKind::Ic0 => Some(Box::new(Ic0::new(&a))),
            PrecondKind::Amg => {
                let bptr: Vec<usize> = self.first.iter().copied().chain(std::iter::once(self.n())).collect();
                let (b, nb) = rigid_body_modes(&self.coords, &self.ndof);
                match amg::Amg::new(a.clone(), bptr, b, nb) {
                    Ok(m) => Some(Box::new(m)),
                    Err(()) => None,
                }
            }
        };
        match pre {
            Some(p) => {
                if timing() {
                    eprintln!("preconditioning ({}): {:.3} s", p.name(), t0.elapsed().as_secs_f64());
                }
                self.pre = Some(p);
                self.a = Some(a);
            }
            None => {
                // AMG could not be built (coarsest level singular): direct, with diagnosis
                let d = self.direct_factor()?;
                *self.direct.lock().unwrap() = Some(d);
            }
        }
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        let (Some(a), Some(pre)) = (&self.a, &self.pre) else {
            return self.solve_direct(b);
        };
        let t0 = std::time::Instant::now();
        let mut x = vec![0.0; b.len()];
        let maxit = 20_000.max(b.len() / 10);
        let rep: PcgReport = pcg(a, b, &mut x, pre.as_ref(), self.tol, maxit);
        if timing() {
            eprintln!(
                "PCG ({}): {} iterations, relative residual {:.2e}, {:.3} s",
                pre.name(),
                rep.iterations,
                rep.rel_residual,
                t0.elapsed().as_secs_f64()
            );
        }
        if !rep.converged {
            eprintln!(
                "warning: PCG did not converge ({} iterations, relative residual {:.2e}); solving directly",
                rep.iterations, rep.rel_residual
            );
            return self.solve_direct(b);
        }
        b.copy_from_slice(&x);
    }
}

/// PCG with the chosen preconditioner on a lower-triangle matrix (for tests
/// and benchmarks): returns the solution and the report.
pub fn solve_lower(k: &Lower, b: &[f64], kind: PrecondKind, coords: &[[f64; 3]], ndof: &[usize], tol: f64) -> (Vec<f64>, PcgReport) {
    let a = Csr::from_lower(k);
    let pre: Box<dyn Precond> = match kind {
        PrecondKind::Jacobi => Box::new(Jacobi::new(&a)),
        PrecondKind::Ic0 => Box::new(Ic0::new(&a)),
        PrecondKind::Amg => {
            let mut bptr = vec![0usize];
            for &d in ndof {
                bptr.push(bptr.last().unwrap() + d);
            }
            let (bm, nb) = rigid_body_modes(coords, ndof);
            Box::new(amg::Amg::new(a.clone(), bptr, bm, nb).expect("AMG could not be built"))
        }
    };
    let mut x = vec![0.0; b.len()];
    let rep = pcg(&a, b, &mut x, pre.as_ref(), tol, 20_000.max(b.len()));
    (x, rep)
}
