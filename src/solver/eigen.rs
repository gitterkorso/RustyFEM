//! Eigenvalue solver for *FREQUENCY and *BUCKLE.
//!
//! The generalized symmetric problem is written in the form
//!
//! ```text
//!   Op x = (1/θ) B x,   Op = K − σM (frequency) or K₀ (buckling)
//! ```
//!
//! where `Op` is positive definite (factorized with RustyFEM's sparse LDLᵀ) and `B`
//! is symmetric: the mass matrix M (positive semidefinite) for natural frequencies,
//! −K_σ (indefinite) for buckling. The operator A = Op⁻¹B is self-adjoint in
//! the inner product ⟨x, y⟩ = xᵀ Op y, and we seek the θ of largest
//! magnitude (the eigenvalues closest to the shift: λ = σ + 1/θ).
//!
//! The method is block subspace iteration with shift-invert and Rayleigh–Ritz
//! (Bathe, *Finite Element Procedures*, 11.6; Saad, *Numerical Methods for
//! Large Eigenvalue Problems*, ch. 5):
//!
//! 1. Y = B X, X̄ = Op⁻¹ Y (one forward/backward substitution per vector).
//! 2. Orthonormalize X̄ in the Op norm: G = X̄ᵀY = X̄ᵀ Op X̄ = U D Uᵀ, T = U D^{−½}
//!    (directions with D ≈ 0 – e.g. vectors in the null space of B – are removed).
//! 3. Project: C = Tᵀ (X̄ᵀ B X̄) T, solve C v = θ v (Jacobi), X = X̄ T V.
//! 4. Converged when ‖Op⁻¹B xᵢ − θᵢ xᵢ‖∞ ≤ tol · |θᵢ| ‖xᵢ‖∞ for all requested modes.
//!
//! Subspace iteration handles multiple eigenvalues (symmetric cross-sections)
//! without special treatment, which a single-vector Lanczos does not.
//! Rigid body modes (singular K) are handled by the caller with a negative shift.
//!
//! This module also contains a simple sparse symmetric matrix (`SymCsr`) for M and K_σ,
//! and dense helper routines (Cholesky, Jacobi) for the projected problems.

use super::{LinearSolver, SystemMatrix};
use crate::model::{fail, Result};

/// The solver module registers no linear solvers.
pub static SOLVERS: &[&dyn LinearSolver] = &[];

// ---------------------------------------------------------------------------
// Sparse symmetric matrix
// ---------------------------------------------------------------------------

/// Collects contributions (r, c, v) to a symmetric matrix (both triangles are stored
/// after `build`). Contributions with r < c are mirrored automatically.
#[derive(Debug, Clone, Default)]
pub struct Triplets {
    pub n: usize,
    items: Vec<(u32, u32, f64)>,
}

impl Triplets {
    pub fn new(n: usize) -> Self {
        Triplets { n, items: Vec::new() }
    }

    /// Adds `v` to (r, c) and, if r ≠ c, to (c, r). Call only for one
    /// triangle (e.g. r ≥ c) so that nothing is counted twice.
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        if v != 0.0 {
            self.items.push((r as u32, c as u32, v));
        }
    }

    pub fn build(mut self) -> SymCsr {
        let mut all = Vec::with_capacity(self.items.len() * 2);
        for &(r, c, v) in &self.items {
            all.push((r, c, v));
            if r != c {
                all.push((c, r, v));
            }
        }
        self.items.clear();
        all.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut rowptr = vec![0usize; self.n + 1];
        let mut col = Vec::with_capacity(all.len());
        let mut val: Vec<f64> = Vec::with_capacity(all.len());
        let mut last: Option<(u32, u32)> = None;
        for (r, c, v) in all {
            if last == Some((r, c)) {
                *val.last_mut().unwrap() += v;
            } else {
                col.push(c);
                val.push(v);
                rowptr[r as usize + 1] += 1;
                last = Some((r, c));
            }
        }
        for i in 0..self.n {
            rowptr[i + 1] += rowptr[i];
        }
        SymCsr { n: self.n, rowptr, col, val }
    }
}

/// Sparse symmetric matrix in CSR format with both triangles.
#[derive(Debug, Clone, Default)]
pub struct SymCsr {
    pub n: usize,
    pub rowptr: Vec<usize>,
    pub col: Vec<u32>,
    pub val: Vec<f64>,
}

impl SymCsr {
    /// y = A x
    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|r| (self.rowptr[r]..self.rowptr[r + 1]).map(|k| self.val[k] * x[self.col[k] as usize]).sum())
            .collect()
    }

    /// The diagonal.
    pub fn diag(&self) -> Vec<f64> {
        (0..self.n)
            .map(|r| (self.rowptr[r]..self.rowptr[r + 1]).find(|&k| self.col[k] as usize == r).map_or(0.0, |k| self.val[k]))
            .collect()
    }

    /// xᵀ A y
    pub fn bilinear(&self, x: &[f64], y: &[f64]) -> f64 {
        dot(x, &self.mul(y))
    }

    /// Adds `s · A` (rows and columns < `m`) to the lower triangle of `k`.
    pub fn add_to(&self, k: &mut dyn SystemMatrix, s: f64, m: usize) {
        for r in 0..self.n.min(m) {
            for idx in self.rowptr[r]..self.rowptr[r + 1] {
                let c = self.col[idx] as usize;
                if c <= r {
                    k.add(r, c, s * self.val[idx]);
                }
            }
        }
    }

    /// The submatrix with rows and columns < m.
    pub fn leading(&self, m: usize) -> SymCsr {
        let mut t = Triplets::new(m);
        for r in 0..m.min(self.n) {
            for idx in self.rowptr[r]..self.rowptr[r + 1] {
                let c = self.col[idx] as usize;
                if c <= r {
                    t.add(r, c, self.val[idx]);
                }
            }
        }
        t.build()
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// ---------------------------------------------------------------------------
// Dense helper routines
// ---------------------------------------------------------------------------

/// Eigenvalues and eigenvectors (column-wise, `v[i*n + j]` = component i of
/// vector j) for a dense symmetric n×n matrix using cyclic Jacobi.
pub fn jacobi(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let scale: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| a[i * n + j].powi(2)).sum();
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

// ---------------------------------------------------------------------------
// Subspace-iteration
// ---------------------------------------------------------------------------

/// Settings for `subspace`.
#[derive(Debug, Clone)]
pub struct EigenOptions {
    /// Number of requested eigenpairs.
    pub nev: usize,
    /// Number of iteration vectors (0 = min(2·nev, nev + 8)).
    pub vectors: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Relative residual tolerance for the eigenvectors.
    pub tol: f64,
    /// Return the latest Ritz pairs even if the tolerance was not reached within
    /// `max_iter` (for rough estimates, e.g. choosing a shift).
    pub allow_unconverged: bool,
}

impl Default for EigenOptions {
    fn default() -> Self {
        EigenOptions { nev: 1, vectors: 0, max_iter: 2000, tol: 1e-10, allow_unconverged: false }
    }
}

/// Result: θ (largest magnitude first) and the corresponding vectors,
/// normalized so that xᵀ Op x = 1.
#[derive(Debug, Clone, Default)]
pub struct EigenResult {
    pub theta: Vec<f64>,
    pub vectors: Vec<Vec<f64>>,
    pub iterations: usize,
}

/// Simple deterministic random generator (xorshift) for start vectors.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Solves Op x = (1/θ) B x for the `nev` θ of largest magnitude.
///
/// `op` must be factorized. `bmul(x)` gives B x. Returns fewer pairs than
/// `nev` if B has lower rank (e.g. buckling load on few elements).
pub fn subspace(n: usize, op: &dyn SystemMatrix, bmul: &dyn Fn(&[f64]) -> Vec<f64>, opts: &EigenOptions) -> Result<EigenResult> {
    if n == 0 || opts.nev == 0 {
        return Ok(EigenResult::default());
    }
    let nev = opts.nev.min(n);
    let mut q = if opts.vectors > 0 { opts.vectors.max(nev) } else { (2 * nev).min(nev + 8) };
    q = q.min(n);
    let solve = |y: &[f64]| {
        let mut x = y.to_vec();
        op.solve(&mut x);
        x
    };
    // start vectors: B · random (removes the null space of B)
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut y: Vec<Vec<f64>> = (0..q).map(|_| bmul(&(0..n).map(|_| rng.next()).collect::<Vec<_>>())).collect();
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut opx: Vec<Vec<f64>> = Vec::new();
    let mut theta: Vec<f64> = Vec::new();
    for it in 1..=opts.max_iter {
        // X̄ = Op⁻¹ Y
        let xb: Vec<Vec<f64>> = y.iter().map(|yi| solve(yi)).collect();
        // convergence check for the previous Ritz vectors: Op⁻¹ B x = X̄
        if !x.is_empty() {
            let k = nev.min(x.len());
            // residual r = Op⁻¹Bx − θx measured in the Op norm: ‖r‖² = rᵀ(Bx − θ·Op x).
            // The Op norm damps round-off noise in nearly singular directions
            // (rigid body modes with a small shift).
            let res: Vec<f64> = (0..k)
                .map(|i| {
                    let rr: f64 = (0..n).map(|j| (xb[i][j] - theta[i] * x[i][j]) * (y[i][j] - theta[i] * opx[i][j])).sum();
                    rr.abs().sqrt() / theta[i].abs()
                })
                .collect();
            let ok = res.iter().all(|r| *r <= opts.tol);
            // fewer vectors than requested means B has lower rank: return what exists
            if ok && (x.len() >= nev || it > 5) {
                let mut res = EigenResult { theta: theta[..k].to_vec(), vectors: x[..k].to_vec(), iterations: it - 1 };
                normalize(&mut res, op, &solve, bmul);
                return Ok(res);
            }
        }
        let m = xb.len();
        // G = X̄ᵀ Y (Op inner products), A = X̄ᵀ B X̄
        let z: Vec<Vec<f64>> = xb.iter().map(|v| bmul(v)).collect();
        let mut g = vec![0.0; m * m];
        let mut a = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..=i {
                let gij = 0.5 * (dot(&xb[i], &y[j]) + dot(&xb[j], &y[i]));
                let aij = 0.5 * (dot(&xb[i], &z[j]) + dot(&xb[j], &z[i]));
                g[i * m + j] = gij;
                g[j * m + i] = gij;
                a[i * m + j] = aij;
                a[j * m + i] = aij;
            }
        }
        // column scaling so that G has unit diagonal (the vectors can differ by many
        // orders of magnitude, e.g. rigid body modes vs elastic modes)
        let gmax = (0..m).map(|i| g[i * m + i]).fold(0.0f64, f64::max);
        let sc: Vec<f64> = (0..m).map(|i| if g[i * m + i] > 1e-300 * gmax.max(1e-300) { 1.0 / g[i * m + i].sqrt() } else { 0.0 }).collect();
        for i in 0..m {
            for j in 0..m {
                g[i * m + j] *= sc[i] * sc[j];
                a[i * m + j] *= sc[i] * sc[j];
            }
        }
        // T = S U D^{-1/2} for D > eps
        let (d, u) = jacobi(&g, m);
        let dmax = d.iter().fold(0.0f64, |s, v| s.max(*v));
        if dmax <= 0.0 || gmax <= 0.0 {
            return fail("the eigenvalue problem has no solutions (the B matrix is zero – missing mass or buckling load?)");
        }
        let keep: Vec<usize> = (0..m).filter(|&j| d[j] > 1e-13 * dmax).collect();
        let r = keep.len();
        let mut t = vec![0.0; m * r];
        for (jj, &j) in keep.iter().enumerate() {
            let s = 1.0 / d[j].sqrt();
            for i in 0..m {
                t[i * r + jj] = u[i * m + j] * s;
            }
        }
        // C = Tᵀ A T
        let mut at = vec![0.0; m * r];
        for i in 0..m {
            for jj in 0..r {
                at[i * r + jj] = (0..m).map(|k| a[i * m + k] * t[k * r + jj]).sum();
            }
        }
        let mut c = vec![0.0; r * r];
        for ii in 0..r {
            for jj in 0..r {
                c[ii * r + jj] = (0..m).map(|k| t[k * r + ii] * at[k * r + jj]).sum();
            }
        }
        for ii in 0..r {
            for jj in 0..ii {
                let s = 0.5 * (c[ii * r + jj] + c[jj * r + ii]);
                c[ii * r + jj] = s;
                c[jj * r + ii] = s;
            }
        }
        let (th, v) = jacobi(&c, r);
        let mut order: Vec<usize> = (0..r).collect();
        order.sort_by(|&i, &j| th[j].abs().partial_cmp(&th[i].abs()).unwrap());
        // W = T V (m × r), X = X̄ W, BX = Z W
        let mut w = vec![0.0; m * r];
        for i in 0..m {
            for (jj, &o) in order.iter().enumerate() {
                w[i * r + jj] = sc[i] * (0..r).map(|k| t[i * r + k] * v[k * r + o]).sum::<f64>();
            }
        }
        x = (0..r).map(|jj| lincomb(&xb, &w, r, jj, n)).collect();
        opx = (0..r).map(|jj| lincomb(&y, &w, r, jj, n)).collect();
        let bx: Vec<Vec<f64>> = (0..r).map(|jj| lincomb(&z, &w, r, jj, n)).collect();
        theta = order.iter().map(|&o| th[o]).collect();
        // next block: Y = B X, top up with new random vectors if the rank has decreased
        // (vectors in the null space of B vanish again in the next orthonormalization)
        y = bx;
        while y.len() < q {
            y.push(bmul(&(0..n).map(|_| rng.next()).collect::<Vec<_>>()));
        }
    }
    if opts.allow_unconverged && !x.is_empty() {
        let k = nev.min(x.len());
        let mut res = EigenResult { theta: theta[..k].to_vec(), vectors: x[..k].to_vec(), iterations: opts.max_iter };
        normalize(&mut res, op, &|v: &[f64]| v.to_vec(), bmul);
        return Ok(res);
    }
    fail(format!("the eigenvalue solver did not converge in {} iterations", opts.max_iter))
}

fn lincomb(vs: &[Vec<f64>], w: &[f64], r: usize, jj: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (i, v) in vs.iter().enumerate() {
        let c = w[i * r + jj];
        if c != 0.0 {
            for (o, x) in out.iter_mut().zip(v) {
                *o += c * x;
            }
        }
    }
    out
}

/// Normalizes the vectors so that xᵀ Op x = 1 (Op x = B x / θ).
fn normalize(res: &mut EigenResult, _op: &dyn SystemMatrix, _solve: &dyn Fn(&[f64]) -> Vec<f64>, bmul: &dyn Fn(&[f64]) -> Vec<f64>) {
    for (x, th) in res.vectors.iter_mut().zip(&res.theta) {
        let s = dot(x, &bmul(x)) / th;
        if s > 0.0 {
            let f = 1.0 / s.sqrt();
            x.iter_mut().for_each(|v| *v *= f);
        }
    }
}

// ---------------------------------------------------------------------------
// Factorization of a SymCsr with any linear solver
// ---------------------------------------------------------------------------

/// The sum Σ sᵢ Aᵢ of symmetric matrices (the submatrices < `n`).
pub fn combine(terms: &[(&SymCsr, f64)], n: usize) -> SymCsr {
    let mut t = Triplets::new(n);
    for (a, s) in terms {
        if *s == 0.0 {
            continue;
        }
        for r in 0..a.n.min(n) {
            for idx in a.rowptr[r]..a.rowptr[r + 1] {
                let c = a.col[idx] as usize;
                if c <= r {
                    t.add(r, c, s * a.val[idx]);
                }
            }
        }
    }
    t.build()
}

/// Records assembly (`zero`/`add`) instead of factorizing, so that
/// the core assembly (elements + interactions) can produce a `SymCsr`.
pub struct Recorder {
    n: usize,
    t: Triplets,
}

impl Recorder {
    pub fn new(n: usize) -> Self {
        Recorder { n, t: Triplets::new(n) }
    }

    pub fn build(self) -> SymCsr {
        self.t.build()
    }
}

impl SystemMatrix for Recorder {
    fn n(&self) -> usize {
        self.n
    }
    fn first_dof(&self) -> &[usize] {
        &[]
    }
    fn zero(&mut self) {
        self.t = Triplets::new(self.n);
    }
    fn add(&mut self, r: usize, c: usize, v: f64) {
        self.t.add(r, c, v);
    }
    fn factor(&mut self) -> std::result::Result<(), usize> {
        Ok(())
    }
    fn solve(&self, _b: &mut [f64]) {}
}

/// Factorized matrix in its own numbering: `solve` permutes to the solver's
/// equation ordering and back.
pub struct Factored {
    mat: Box<dyn SystemMatrix>,
    /// The solver's equation number for each equation in our numbering.
    perm: Vec<usize>,
}

impl SystemMatrix for Factored {
    fn n(&self) -> usize {
        self.perm.len()
    }
    fn first_dof(&self) -> &[usize] {
        &[]
    }
    fn zero(&mut self) {}
    fn add(&mut self, _r: usize, _c: usize, _v: f64) {}
    fn factor(&mut self) -> std::result::Result<(), usize> {
        Ok(())
    }
    fn solve(&self, b: &mut [f64]) {
        let mut y = vec![0.0; b.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            y[p] = b[i];
        }
        self.mat.solve(&mut y);
        for (i, &p) in self.perm.iter().enumerate() {
            b[i] = y[p];
        }
    }
}

/// Factorizes `a` with the solver `solver`. `blocks` divides the equations into
/// blocks (a node, an element's internal degrees of freedom) with a coordinate for
/// the solver's ordering; each equation must belong to exactly one block.
/// Returns `Ok(Err(eq))` if the matrix is not positive definite.
pub fn factor_sym(
    a: &SymCsr,
    blocks: &[([f64; 3], Vec<usize>)],
    solver: &dyn super::LinearSolver,
) -> Result<std::result::Result<Factored, usize>> {
    let n = a.n;
    let mut block_of = vec![usize::MAX; n];
    for (b, (_, eqs)) in blocks.iter().enumerate() {
        for &q in eqs {
            block_of[q] = b;
        }
    }
    if block_of.contains(&usize::MAX) {
        return fail("factor_sym: an equation has no block");
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for r in 0..n {
        for idx in a.rowptr[r]..a.rowptr[r + 1] {
            let (br, bc) = (block_of[r], block_of[a.col[idx] as usize]);
            if br != bc {
                adj[br].push(bc);
            }
        }
    }
    for v in adj.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let g = super::Graph { coords: blocks.iter().map(|b| b.0).collect(), adj, ndof: blocks.iter().map(|b| b.1.len()).collect() };
    let mut mat = solver.prepare(&g)?;
    let mut perm = vec![0usize; n];
    for (b, (_, eqs)) in blocks.iter().enumerate() {
        let f = mat.first_dof()[b];
        for (k, &q) in eqs.iter().enumerate() {
            perm[q] = f + k;
        }
    }
    mat.zero();
    for r in 0..n {
        for idx in a.rowptr[r]..a.rowptr[r + 1] {
            let c = a.col[idx] as usize;
            if c <= r {
                let (pr, pc) = (perm[r], perm[c]);
                mat.add(pr.max(pc), pr.min(pc), a.val[idx]);
            }
        }
    }
    Ok(match mat.factor() {
        Ok(()) => Ok(Factored { mat, perm }),
        Err(q) => Err(perm.iter().position(|&p| p == q).unwrap_or(q)),
    })
}
