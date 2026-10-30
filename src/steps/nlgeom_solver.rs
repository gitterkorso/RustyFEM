//! Equation solving for NLGEOM and Riks when the tangent is not positive definite.
//!
//! After a limit point (buckling, snap-through) or at a poor iteration point
//! the tangent stiffness can have negative eigenvalues. The core's direct solver requires
//! a positive definite matrix, so the NLGEOM steps use `DualSolver`: the matrix
//! is assembled both in the selected solver and in the indefinite multifrontal
//! LDLᵀ solver (`solver/indefinite.rs`, 1×1/2×2 pivoting). The selected solver
//! is used as long as it manages the factorization, otherwise the indefinite one. The number of
//! negative pivots (= number of negative eigenvalues according to Sylvester's
//! law of inertia) is reported, which the Riks step uses.
//!
//! The two solvers may choose different equation orderings (`first_dof`), so indices
//! are translated block by block between the numberings.

use crate::model::Result;
use crate::solver::{Graph, LinearSolver, SystemMatrix};
use std::sync::{Arc, Mutex};

/// Information from the latest factorization.
#[derive(Debug, Clone, Copy, Default)]
pub struct FactorInfo {
    /// Number of negative pivots (0 if the selected solver handled the matrix).
    pub negative: usize,
    /// The indefinite solver was used.
    pub fallback: bool,
}

pub struct DualSolver {
    pub main: &'static dyn LinearSolver,
    pub info: Arc<Mutex<FactorInfo>>,
}

impl DualSolver {
    pub fn new(main: &'static dyn LinearSolver) -> Self {
        DualSolver { main, info: Arc::new(Mutex::new(FactorInfo::default())) }
    }
}

impl LinearSolver for DualSolver {
    fn name(&self) -> &'static str {
        "nlgeom-dual"
    }

    fn prepare(&self, g: &Graph) -> Result<Box<dyn SystemMatrix>> {
        let main = self.main.prepare(g)?;
        // no warning per factorization: Riks reports the number of negative eigenvalues
        let indef = crate::solver::indefinite::prepare_quiet(g)?;
        // equation in the main solver's numbering -> equation in the indefinite one's
        let n = main.n();
        let mut map = vec![0; n];
        for (b, &nd) in g.ndof.iter().enumerate() {
            let (fm, fi) = (main.first_dof()[b], indef.first_dof()[b]);
            for k in 0..nd {
                map[fm + k] = fi + k;
            }
        }
        let identity = map.iter().enumerate().all(|(i, &j)| i == j);
        let mut inv = vec![0; n];
        for (i, &j) in map.iter().enumerate() {
            inv[j] = i;
        }
        Ok(Box::new(DualMatrix { main, indef, map, inv, identity, prefer_indef: false, use_indef: false, info: self.info.clone() }))
    }
}

struct DualMatrix {
    main: Box<dyn SystemMatrix>,
    indef: Box<dyn SystemMatrix>,
    map: Vec<usize>,
    inv: Vec<usize>,
    identity: bool,
    prefer_indef: bool,
    use_indef: bool,
    info: Arc<Mutex<FactorInfo>>,
}

impl SystemMatrix for DualMatrix {
    fn n(&self) -> usize {
        self.main.n()
    }
    fn first_dof(&self) -> &[usize] {
        self.main.first_dof()
    }
    fn zero(&mut self) {
        self.main.zero();
        self.indef.zero();
    }
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.main.add(r, c, v);
        let (a, b) = (self.map[r], self.map[c]);
        if a >= b {
            self.indef.add(a, b, v)
        } else {
            self.indef.add(b, a, v)
        }
    }
    fn factor(&mut self) -> std::result::Result<(), usize> {
        if !self.prefer_indef && self.main.factor().is_ok() {
            self.use_indef = false;
            *self.info.lock().unwrap() = FactorInfo { negative: 0, fallback: false };
            return Ok(());
        }
        self.indef.factor().map_err(|e| self.inv.get(e).copied().unwrap_or(e))?;
        let neg = self.indef.inertia().map(|i| i.neg).unwrap_or(0);
        self.use_indef = true;
        // back to the selected solver when the matrix is positive definite again
        self.prefer_indef = neg > 0;
        *self.info.lock().unwrap() = FactorInfo { negative: neg, fallback: true };
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        if !self.use_indef {
            return self.main.solve(b);
        }
        if self.identity {
            return self.indef.solve(b);
        }
        let mut y = vec![0.0; b.len()];
        for (i, &j) in self.map.iter().enumerate() {
            y[j] = b[i];
        }
        self.indef.solve(&mut y);
        for (j, &i) in self.inv.iter().enumerate() {
            b[i] = y[j];
        }
    }
    fn inertia(&self) -> Option<crate::solver::Inertia> {
        if self.use_indef {
            self.indef.inertia()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Indefinite tridiagonal matrix: the main solver (direct) fails, the
    /// indefinite one takes over and gives the correct solution and two negative eigenvalues
    /// (computed by hand: leading minors 2, −7, −9, 19 change
    /// sign twice).
    #[test]
    fn dual_indefinite() {
        let g = Graph { coords: (0..4).map(|i| [i as f64, 0.0, 0.0]).collect(), adj: vec![vec![1], vec![0, 2], vec![1, 3], vec![2]], ndof: vec![1; 4] };
        let main = crate::solver::select(Some("direct")).unwrap();
        let dual = DualSolver::new(main);
        let mut s = dual.prepare(&g).unwrap();
        let a = [[2.0, 1.0, 0.0, 0.0], [1.0, -3.0, 1.0, 0.0], [0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0]];
        // a in block numbering; the equation for block b is first_dof[b]
        let fd = s.first_dof().to_vec();
        for i in 0..4 {
            for j in 0..=i {
                if a[i][j] != 0.0 {
                    let (r, c) = (fd[i], fd[j]);
                    if r >= c {
                        s.add(r, c, a[i][j])
                    } else {
                        s.add(c, r, a[i][j])
                    }
                }
            }
        }
        s.factor().unwrap();
        let info = *dual.info.lock().unwrap();
        assert!(info.fallback);
        assert_eq!(info.negative, 2);
        let x = [1.0, -2.0, 0.5, 3.0];
        let mut b = vec![0.0; 4];
        for i in 0..4 {
            b[fd[i]] = (0..4).map(|j| a[i][j] * x[j]).sum();
        }
        s.solve(&mut b);
        for i in 0..4 {
            assert!((b[fd[i]] - x[i]).abs() < 1e-12);
        }
    }
}
