//! Sparse direct solver: nested dissection + multifrontal LDLᵀ (`crate::sparse`).
//! Requires a positive definite matrix.

use super::api::{Graph, LinearSolver, SystemMatrix};
use crate::model::Result;
use crate::sparse::{self, Factor, Lower, Symbolic};

pub struct Direct;

pub static SOLVERS: &[&dyn LinearSolver] = &[&Direct];

struct DirectMatrix {
    sym: Symbolic,
    k: Lower,
    fac: Option<Factor>,
}

impl LinearSolver for Direct {
    fn name(&self) -> &'static str {
        "direct"
    }

    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>> {
        let t0 = std::time::Instant::now();
        let sym = sparse::analyse(&g.coords, &g.adj, &g.ndof);
        let k = sym.pattern(&g.adj, &g.ndof);
        if crate::settings::with(|s| s.timing) {
            eprintln!(
                "analysis: {:.3} s  flops: {:.2e}  factor: {} MB  largest front: {:?}",
                t0.elapsed().as_secs_f64(),
                sym.work(),
                sym.factor_size() * 8 >> 20,
                sym.largest_front()
            );
        }
        Ok(Box::new(DirectMatrix { sym, k, fac: None }))
    }
}

impl SystemMatrix for DirectMatrix {
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
        self.fac = Some(sparse::factor(&self.sym, &self.k)?);
        if crate::settings::with(|s| s.timing) {
            eprintln!("factorization: {:.3} s", t0.elapsed().as_secs_f64());
        }
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        self.fac.as_ref().expect("internal: solve before factor").solve(b);
    }
}
