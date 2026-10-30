//! Linear algebra for the acoustic steps: sparse general (unsymmetric) matrix,
//! factorization with an optional linear solver (normally `unsymmetric`), dense
//! real Schur form and subspace iteration with shift-invert for the
//! unsymmetric eigenvalue problem K x = λ M x.
//!
//! The coupled acoustic–structural problem is unsymmetric
//! (Zienkiewicz–Newton) but has real eigenvalues. The subspace iteration
//! X ← (K − σM)⁻¹ M X with orthonormalization and Rayleigh–Ritz projection
//! (K_r = Yᵀ K Y, M_r = Yᵀ M Y, eigenvalues of M_r⁻¹ K_r from the real
//! Schur form) converges to the eigenpairs closest to the shift; for symmetric
//! problems it is the same method as `solver::eigen::subspace`.

use crate::model::{fail, Result};
use crate::solver::{Graph, LinearSolver, SystemMatrix};

/// Sparse general matrix: rows of (column, value), sorted and merged
/// after `finish`.
#[derive(Debug, Clone, Default)]
pub struct Gen {
    pub n: usize,
    pub rows: Vec<Vec<(u32, f64)>>,
}

impl Gen {
    pub fn new(n: usize) -> Self {
        Gen { n, rows: vec![Vec::new(); n] }
    }

    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        if v != 0.0 {
            self.rows[r].push((c as u32, v));
        }
    }

    /// Sorts and merges duplicates.
    pub fn finish(&mut self) {
        for row in self.rows.iter_mut() {
            row.sort_unstable_by_key(|p| p.0);
            let mut out: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for &(c, v) in row.iter() {
                match out.last_mut() {
                    Some(l) if l.0 == c => l.1 += v,
                    _ => out.push((c, v)),
                }
            }
            *row = out;
        }
    }

    /// y = A x (x may be longer than the number of columns used).
    pub fn mul(&self, x: &[f64]) -> Vec<f64> {
        self.rows.iter().map(|r| r.iter().map(|&(c, v)| v * x[c as usize]).sum()).collect()
    }

    /// The submatrix with rows and columns < m.
    pub fn leading(&self, m: usize) -> Gen {
        let mut g = Gen::new(m);
        for (r, row) in self.rows.iter().enumerate().take(m) {
            g.rows[r] = row.iter().copied().filter(|p| (p.0 as usize) < m).collect();
        }
        g
    }

    /// Σ sᵢ Aᵢ (the submatrices < n).
    pub fn combine(terms: &[(&Gen, f64)], n: usize) -> Gen {
        let mut g = Gen::new(n);
        for (a, s) in terms {
            if *s == 0.0 {
                continue;
            }
            for (r, row) in a.rows.iter().enumerate().take(n) {
                for &(c, v) in row {
                    if (c as usize) < n {
                        g.rows[r].push((c, s * v));
                    }
                }
            }
        }
        g.finish();
        g
    }

    pub fn diag(&self) -> Vec<f64> {
        self.rows.iter().enumerate().map(|(r, row)| row.iter().find(|p| p.0 as usize == r).map_or(0.0, |p| p.1)).collect()
    }
}

/// A factorized matrix in its own numbering.
pub struct GenFactor {
    mat: Box<dyn SystemMatrix>,
    perm: Vec<usize>,
}

impl GenFactor {
    pub fn solve(&self, b: &mut [f64]) {
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

/// Factorizes the general matrix `a` (all entries, including unsymmetric ones)
/// with the solver `solver`. `blocks` (coordinate, equations) partitions
/// the equations for the solver ordering. `Ok(Err(eq))` if the matrix is singular.
pub fn factor_gen(a: &Gen, blocks: &[([f64; 3], Vec<usize>)], solver: &dyn LinearSolver) -> Result<std::result::Result<GenFactor, usize>> {
    let n = a.n;
    let mut block_of = vec![usize::MAX; n];
    for (b, (_, eqs)) in blocks.iter().enumerate() {
        for &q in eqs {
            block_of[q] = b;
        }
    }
    if block_of.contains(&usize::MAX) {
        return fail("factor_gen: an equation has no block");
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for (r, row) in a.rows.iter().enumerate() {
        for &(c, _) in row {
            let (br, bc) = (block_of[r], block_of[c as usize]);
            if br != bc {
                adj[br].push(bc);
                adj[bc].push(br);
            }
        }
    }
    for v in adj.iter_mut() {
        v.sort_unstable();
        v.dedup();
    }
    let g = Graph { coords: blocks.iter().map(|b| b.0).collect(), adj, ndof: blocks.iter().map(|b| b.1.len()).collect() };
    let mut mat = solver.prepare(&g)?;
    let mut perm = vec![0usize; n];
    for (b, (_, eqs)) in blocks.iter().enumerate() {
        let f = mat.first_dof()[b];
        for (k, &q) in eqs.iter().enumerate() {
            perm[q] = f + k;
        }
    }
    mat.zero();
    let sym = mat.is_symmetric();
    for (r, row) in a.rows.iter().enumerate() {
        for &(c, v) in row {
            let (pr, pc) = (perm[r], perm[c as usize]);
            if sym {
                if pr >= pc {
                    mat.add(pr, pc, v);
                }
            } else {
                mat.add_unsym(pr, pc, v);
            }
        }
    }
    Ok(match mat.factor() {
        Ok(()) => Ok(GenFactor { mat, perm }),
        Err(q) => Err(perm.iter().position(|&p| p == q).unwrap_or(q)),
    })
}

// ---------------------------------------------------------------------------
// Dense routines
// ---------------------------------------------------------------------------

/// Solves A X = B (A n×n row-wise, B n×m row-wise) with Gaussian elimination and
/// partial pivoting. `None` if A is singular.
pub fn dense_solve(a: &[f64], b: &[f64], n: usize, m: usize) -> Option<Vec<f64>> {
    let mut a = a.to_vec();
    let mut x = b.to_vec();
    let scale = a.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + k].abs().partial_cmp(&a[j * n + k].abs()).unwrap())?;
        if a[p * n + k].abs() <= 1e-300 * scale.max(1.0) || a[p * n + k] == 0.0 {
            return None;
        }
        if p != k {
            for c in 0..n {
                a.swap(k * n + c, p * n + c);
            }
            for c in 0..m {
                x.swap(k * m + c, p * m + c);
            }
        }
        let d = a[k * n + k];
        for i in k + 1..n {
            let f = a[i * n + k] / d;
            if f == 0.0 {
                continue;
            }
            for c in k..n {
                a[i * n + c] -= f * a[k * n + c];
            }
            for c in 0..m {
                x[i * m + c] -= f * x[k * m + c];
            }
        }
    }
    for k in (0..n).rev() {
        for c in 0..m {
            let mut s = x[k * m + c];
            for j in k + 1..n {
                s -= a[k * n + j] * x[j * m + c];
            }
            x[k * m + c] = s / a[k * n + k];
        }
    }
    Some(x)
}

/// Real Schur form A = Q T Qᵀ (T upper triangular; 2×2 blocks that do not
/// converge, i.e. complex pairs, are forced to triangular form). Returns (T, Q)
/// row-wise n×n.
pub fn real_schur(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut h = a.to_vec();
    let mut q = vec![0.0; n * n];
    for i in 0..n {
        q[i * n + i] = 1.0;
    }
    // Hessenberg form with Householder reflections
    for k in 0..n.saturating_sub(2) {
        let alpha: f64 = (k + 1..n).map(|i| h[i * n + k] * h[i * n + k]).sum::<f64>().sqrt();
        if alpha == 0.0 {
            continue;
        }
        let mut v = vec![0.0; n];
        let s = if h[(k + 1) * n + k] > 0.0 { -alpha } else { alpha };
        for i in k + 1..n {
            v[i] = h[i * n + k];
        }
        v[k + 1] -= s;
        let vn: f64 = v.iter().map(|x| x * x).sum();
        if vn == 0.0 {
            continue;
        }
        // H ← (I − 2vvᵀ/vᵀv) H (I − 2vvᵀ/vᵀv), Q ← Q (I − 2vvᵀ/vᵀv)
        for c in 0..n {
            let d: f64 = (k + 1..n).map(|i| v[i] * h[i * n + c]).sum::<f64>() * 2.0 / vn;
            for i in k + 1..n {
                h[i * n + c] -= d * v[i];
            }
        }
        for r in 0..n {
            let d: f64 = (k + 1..n).map(|i| h[r * n + i] * v[i]).sum::<f64>() * 2.0 / vn;
            for i in k + 1..n {
                h[r * n + i] -= d * v[i];
            }
            let d: f64 = (k + 1..n).map(|i| q[r * n + i] * v[i]).sum::<f64>() * 2.0 / vn;
            for i in k + 1..n {
                q[r * n + i] -= d * v[i];
            }
        }
        for i in k + 2..n {
            h[i * n + k] = 0.0;
        }
    }
    // shifted QR iteration with Givens rotations
    let norm = h.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
    let eps = f64::EPSILON;
    let mut hi = n;
    let mut iter = 0usize;
    while hi > 1 {
        let k = hi - 1;
        // search for deflation from below
        let mut lo = k;
        while lo > 0 {
            let s = h[lo * n + lo].abs() + h[(lo - 1) * n + lo - 1].abs();
            if h[lo * n + lo - 1].abs() <= eps * (if s == 0.0 { norm } else { s }) {
                h[lo * n + lo - 1] = 0.0;
                break;
            }
            lo -= 1;
        }
        if lo == k {
            hi -= 1;
            iter = 0;
            continue;
        }
        iter += 1;
        if iter > 60 {
            // complex pair or no convergence: force deflation
            h[k * n + k - 1] = 0.0;
            hi -= 1;
            iter = 0;
            continue;
        }
        // Wilkinson shift (real part for complex eigenvalues), occasionally an exceptional shift
        let (a11, a12, a21, a22) = (h[(k - 1) * n + k - 1], h[(k - 1) * n + k], h[k * n + k - 1], h[k * n + k]);
        let tr = 0.5 * (a11 + a22);
        let det = a11 * a22 - a12 * a21;
        let disc = tr * tr - det;
        let mu = if iter % 11 == 10 {
            a22 + h[k * n + k - 1].abs()
        } else if disc >= 0.0 {
            let r = disc.sqrt();
            let (l1, l2) = (tr + r, tr - r);
            if (l1 - a22).abs() < (l2 - a22).abs() { l1 } else { l2 }
        } else {
            tr
        };
        for i in lo..=k {
            h[i * n + i] -= mu;
        }
        let mut rots = Vec::with_capacity(k - lo);
        for j in lo..k {
            let (x, y) = (h[j * n + j], h[(j + 1) * n + j]);
            let r = x.hypot(y);
            let (c, s) = if r == 0.0 { (1.0, 0.0) } else { (x / r, y / r) };
            for col in j..n {
                let (p, t) = (h[j * n + col], h[(j + 1) * n + col]);
                h[j * n + col] = c * p + s * t;
                h[(j + 1) * n + col] = -s * p + c * t;
            }
            rots.push((c, s));
        }
        for (idx, j) in (lo..k).enumerate() {
            let (c, s) = rots[idx];
            for row in 0..=(j + 1).min(k) {
                let (p, t) = (h[row * n + j], h[row * n + j + 1]);
                h[row * n + j] = c * p + s * t;
                h[row * n + j + 1] = -s * p + c * t;
            }
            for row in 0..n {
                let (p, t) = (q[row * n + j], q[row * n + j + 1]);
                q[row * n + j] = c * p + s * t;
                q[row * n + j + 1] = -s * p + c * t;
            }
        }
        for i in lo..=k {
            h[i * n + i] += mu;
        }
    }
    // rows above the window have been rotated along; zero below the diagonal
    for r in 1..n {
        for c in 0..r {
            h[r * n + c] = 0.0;
        }
    }
    (h, q)
}

/// Eigenvalues and eigenvectors (column-wise in `v[i*n + j]`) of a dense
/// unsymmetric matrix with real eigenvalues.
pub fn dense_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (l, v, _) = dense_eig_schur(a, n);
    (l, v)
}

/// Like `dense_eig` but also returns the Schur vectors Q.
pub fn dense_eig_schur(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (t, q) = real_schur(a, n);
    let norm = t.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
    let small = 1e-14 * norm;
    let lam: Vec<f64> = (0..n).map(|i| t[i * n + i]).collect();
    let mut v = vec![0.0; n * n];
    for j in 0..n {
        let mut w = vec![0.0; n];
        w[j] = 1.0;
        for i in (0..j).rev() {
            let s: f64 = (i + 1..=j).map(|k| t[i * n + k] * w[k]).sum();
            let mut d = t[i * n + i] - lam[j];
            if d.abs() < small {
                d = if d < 0.0 { -small } else { small };
            }
            w[i] = -s / d;
        }
        // x = Q w, normalized
        let mut x: Vec<f64> = (0..n).map(|r| (0..=j).map(|k| q[r * n + k] * w[k]).sum()).collect();
        let nx = x.iter().map(|y| y * y).sum::<f64>().sqrt().max(1e-300);
        x.iter_mut().for_each(|y| *y /= nx);
        for r in 0..n {
            v[r * n + j] = x[r];
        }
    }
    (lam, v, q)
}

// ---------------------------------------------------------------------------
// Subspace iteration for K x = λ M x
// ---------------------------------------------------------------------------

pub struct UnsymEigen {
    pub lambda: Vec<f64>,
    pub vectors: Vec<Vec<f64>>,
    pub iterations: usize,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Orthonormalizes the columns (Gram–Schmidt twice); linearly dependent
/// vectors are replaced with random vectors.
fn orthonormalize(y: &mut [Vec<f64>], seed: &mut u64) {
    let rnd = |seed: &mut u64| {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    for j in 0..y.len() {
        for attempt in 0..3 {
            let n0 = dot(&y[j], &y[j]).sqrt();
            for _ in 0..2 {
                for i in 0..j {
                    let c = dot(&y[i], &y[j]);
                    let (a, b) = y.split_at_mut(j);
                    for (bj, ai) in b[0].iter_mut().zip(&a[i]) {
                        *bj -= c * ai;
                    }
                }
            }
            let n1 = dot(&y[j], &y[j]).sqrt();
            if n1 > 1e-10 * n0 && n1 > 0.0 {
                y[j].iter_mut().for_each(|v| *v /= n1);
                break;
            }
            let len = y[j].len();
            y[j] = (0..len).map(|_| rnd(seed)).collect();
            if attempt == 2 {
                let n2 = dot(&y[j], &y[j]).sqrt().max(1e-300);
                y[j].iter_mut().for_each(|v| *v /= n2);
            }
        }
    }
}

/// The `nev` eigenpairs closest to the shift σ for K x = λ M x, where `op` solves
/// (K − σM) x = b. `start` gives starting vectors (optional).
pub fn subspace_unsym(k: &Gen, m: &Gen, sigma: f64, op: &dyn Fn(&mut [f64]), nev: usize, tol: f64, max_iter: usize) -> Result<UnsymEigen> {
    let n = k.n;
    if n == 0 || nev == 0 {
        return Ok(UnsymEigen { lambda: Vec::new(), vectors: Vec::new(), iterations: 0 });
    }
    let nev = nev.min(n);
    let q = (2 * nev).max(nev + 16).min(n);
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut x: Vec<Vec<f64>> = (0..q)
        .map(|_| {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
                })
                .collect()
        })
        .collect();
    // M-weighted starting vectors: the diagonal of M (degrees of freedom without mass do not contribute)
    let md = m.diag();
    for v in x.iter_mut() {
        for (a, d) in v.iter_mut().zip(&md) {
            if *d == 0.0 {
                *a *= 1e-3;
            }
        }
    }
    let mut prev: Vec<f64> = Vec::new();
    for it in 1..=max_iter {
        // Y = (K − σM)⁻¹ M X
        let mut y: Vec<Vec<f64>> = x
            .iter()
            .map(|xi| {
                let mut b = m.mul(xi);
                op(&mut b);
                b
            })
            .collect();
        orthonormalize(&mut y, &mut seed);
        // Rayleigh–Ritz
        let ky: Vec<Vec<f64>> = y.iter().map(|v| k.mul(v)).collect();
        let my: Vec<Vec<f64>> = y.iter().map(|v| m.mul(v)).collect();
        let mut kr = vec![0.0; q * q];
        let mut mr = vec![0.0; q * q];
        for i in 0..q {
            for j in 0..q {
                kr[i * q + j] = dot(&y[i], &ky[j]);
                mr[i * q + j] = dot(&y[i], &my[j]);
            }
        }
        let Some(c) = dense_solve(&mr, &kr, q, q) else {
            return fail("acoustic eigenvalue extraction: the projected mass matrix is singular");
        };
        let (lam, v, qs) = dense_eig_schur(&c, q);
        let mut order: Vec<usize> = (0..q).collect();
        order.sort_by(|&a, &b| (lam[a] - sigma).abs().partial_cmp(&(lam[b] - sigma).abs()).unwrap_or(std::cmp::Ordering::Equal));
        let lam_s: Vec<f64> = order.iter().map(|&i| lam[i]).collect();
        // eigenvectors (for checking and results) and Schur vectors (next basis;
        // they stay independent even for multiple eigenvalues)
        let ritz: Vec<Vec<f64>> = order.iter().take(nev).map(|&j| (0..n).map(|r| (0..q).map(|i| y[i][r] * v[i * q + j]).sum()).collect()).collect();
        x = (0..q).map(|j| (0..n).map(|r| (0..q).map(|i| y[i][r] * qs[i * q + j]).sum()).collect()).collect();
        // convergence: the eigenvalues and residuals of the requested pairs
        let scale = lam_s.iter().take(nev).fold(0.0f64, |s, l| s.max(l.abs())).max(1e-300);
        let conv_val = prev.len() == lam_s.len() && (0..nev).all(|i| (lam_s[i] - prev[i]).abs() <= tol * lam_s[i].abs() + 1e-12 * scale);
        let mut conv = conv_val;
        if conv {
            for i in 0..nev {
                let kx = k.mul(&ritz[i]);
                let mx = m.mul(&ritz[i]);
                let r: f64 = kx.iter().zip(&mx).map(|(a, b)| (a - lam_s[i] * b).powi(2)).sum::<f64>().sqrt();
                let d: f64 = kx.iter().map(|a| a * a).sum::<f64>().sqrt() + lam_s[i].abs().max(1e-6 * scale) * mx.iter().map(|a| a * a).sum::<f64>().sqrt();
                if r > 1e3 * tol * d.max(1e-300) {
                    conv = false;
                    break;
                }
            }
        }
        prev = lam_s.clone();
        if conv || it == max_iter {
            if !conv {
                return fail(format!("acoustic eigenvalue extraction did not converge in {max_iter} iterations"));
            }
            let mut pairs: Vec<(f64, Vec<f64>)> = lam_s.into_iter().zip(ritz).take(nev).collect();
            pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            let (lambda, vectors) = pairs.into_iter().unzip();
            return Ok(UnsymEigen { lambda, vectors, iterations: it });
        }
    }
    fail("acoustic eigenvalue extraction: no iterations")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schur_eig_nonsymmetric() {
        // A = V D V⁻¹ with real eigenvalues 1, 2, 2, 5 (double eigenvalue)
        let n = 4;
        let vm = [1.0, 0.2, 0.0, 0.3, 0.1, 1.0, 0.4, 0.0, 0.0, 0.3, 1.0, 0.2, 0.5, 0.0, 0.1, 1.0];
        let d = [1.0, 2.0, 2.0, 5.0];
        let mut vd = vm;
        for r in 0..n {
            for c in 0..n {
                vd[r * n + c] *= d[c];
            }
        }
        // A Vᵀ... A = (V D) V⁻¹: solve Vᵀ Aᵀ = (V D)ᵀ
        let vt: Vec<f64> = (0..n * n).map(|i| vm[(i % n) * n + i / n]).collect();
        let vdt: Vec<f64> = (0..n * n).map(|i| vd[(i % n) * n + i / n]).collect();
        let at = dense_solve(&vt, &vdt, n, n).unwrap();
        let a: Vec<f64> = (0..n * n).map(|i| at[(i % n) * n + i / n]).collect();
        let (lam, v) = dense_eig(&a, n);
        let mut ls = lam.clone();
        ls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (x, y) in ls.iter().zip(d) {
            assert!((x - y).abs() < 1e-9, "{ls:?}");
        }
        for j in 0..n {
            for r in 0..n {
                let av: f64 = (0..n).map(|c| a[r * n + c] * v[c * n + j]).sum();
                assert!((av - lam[j] * v[r * n + j]).abs() < 1e-8, "eigenpair {j}: {av} vs {}", lam[j] * v[r * n + j]);
            }
        }
    }
}
