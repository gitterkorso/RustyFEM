//! C interface to the RustyFEM sparse solvers (`rustyfem::sparse`).
//!
//! Input is a symmetric matrix in CSC format (both triangles or only
//! the lower one) in the caller's numbering, plus, for each degree of freedom, the node
//! it belongs to and the node coordinates. The ordering (nested dissection at
//! node level) and the factorization are done here; the caller only sees its own
//! numbering. No external dependencies, so the library can be loaded with ctypes.

use rustyfem::sparse::indef::{self, IndefFactor, Inertia};
use rustyfem::sparse::lu::{self, Full, LuFactor};
use rustyfem::sparse::{self, Factor};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

/// Leaf size for the FFI path: measured on openFEM's shell and solid models,
/// 16 nodes per leaf gives about half the fill-in compared with the default 48.
const FFI_LEAF: usize = 16;
const NONE: usize = usize::MAX;

/// Factor type: positive definite LDLᵀ, indefinite LDLᵀ (1×1/2×2) or unsymmetric LU.
enum Fac {
    Pd(Factor),
    Indef(IndefFactor),
    Lu(LuFactor),
}

impl Fac {
    fn solve(&self, x: &mut [f64]) {
        match self {
            Fac::Pd(f) => f.solve(x),
            Fac::Indef(f) => f.solve(x),
            Fac::Lu(f) => f.solve(x),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Pd,
    Indef,
    Lu,
}

pub struct Handle {
    fac: Fac,
    /// Internal equation for each degree of freedom in the caller's numbering (NONE = uncoupled).
    eq: Vec<usize>,
    /// Uncoupled degrees of freedom and their diagonal.
    iso: Vec<(usize, f64)>,
    n_sym: usize,
    work: f64,
    factor_size: usize,
    /// Inertia [negative, zero, positive]; None for LU.
    inertia: Option<[i64; 3]>,
}

/// Factorizes the matrix. Returns null on error: `*err_dof` is then the
/// degree of freedom (caller's numbering) where a non-positive pivot occurred, or
/// −1 for invalid input. `d_out` (length n, may be null) is filled with the pivots D.
///
/// # Safety
/// The pointers must point to arrays of the given length: `loc` and `d_out` n,
/// `coords` 3·nn, `indptr` n+1, `indices` and `data` indptr[n].
#[no_mangle]
pub unsafe extern "C" fn rustyfem_factor(
    n: usize,
    nn: usize,
    loc: *const i64,
    coords: *const f64,
    indptr: *const i64,
    indices: *const i64,
    data: *const f64,
    d_out: *mut f64,
    err_dof: *mut i64,
) -> *mut Handle {
    *err_dof = -1;
    let loc = slice::from_raw_parts(loc, n);
    let coords = slice::from_raw_parts(coords, 3 * nn);
    let indptr = slice::from_raw_parts(indptr, n + 1);
    let nnz = indptr[n] as usize;
    let indices = slice::from_raw_parts(indices, nnz);
    let data = slice::from_raw_parts(data, nnz);
    let res = catch_unwind(AssertUnwindSafe(|| build(n, nn, loc, coords, indptr, indices, data, Mode::Pd)));
    match res {
        Ok(Ok((h, d))) => {
            if !d_out.is_null() {
                slice::from_raw_parts_mut(d_out, n).copy_from_slice(&d);
            }
            Box::into_raw(Box::new(h))
        }
        Ok(Err(q)) => {
            *err_dof = q;
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

#[allow(clippy::too_many_arguments)]
fn build(n: usize, nn: usize, loc: &[i64], coords: &[f64], indptr: &[i64], indices: &[i64], data: &[f64], mode: Mode) -> Result<(Handle, Vec<f64>), i64> {
    let loc: Vec<usize> = loc.iter().map(|&l| l as usize).collect();
    if loc.iter().any(|&l| l >= nn) || indices.iter().any(|&r| r < 0 || r as usize >= n) {
        return Err(-1);
    }
    // Degrees of freedom without couplings (diagonal term only, e.g. stabilized
    // rotations in solid nodes) are solved directly. Otherwise they end up in the node's dense
    // block and the fill-in becomes (6/3)² = 4 times too large.
    let mut coupled = vec![false; n];
    let mut diag = vec![0.0; n];
    for c in 0..n {
        for p in indptr[c] as usize..indptr[c + 1] as usize {
            let r = indices[p] as usize;
            if r == c {
                diag[c] += data[p];
            } else if data[p] != 0.0 {
                coupled[c] = true;
                coupled[r] = true;
            }
        }
    }
    let mut iso = Vec::new();
    for i in (0..n).filter(|&i| !coupled[i]) {
        let ok = if mode == Mode::Pd { diag[i] > 0.0 } else { diag[i] != 0.0 && diag[i].is_finite() };
        if !ok {
            return Err(i as i64);
        }
        iso.push((i, diag[i]));
    }
    // nodes with at least one coupled degree of freedom, in compact numbering
    let mut nid = vec![NONE; nn];
    let mut xyz: Vec<[f64; 3]> = Vec::new();
    for i in (0..n).filter(|&i| coupled[i]) {
        let v = loc[i];
        if nid[v] == NONE {
            nid[v] = xyz.len();
            xyz.push([coords[3 * v], coords[3 * v + 1], coords[3 * v + 2]]);
        }
    }
    let nc = xyz.len();
    // the position of the degrees of freedom within the node, in ascending order
    let mut ndof = vec![0usize; nc];
    let mut local = vec![0usize; n];
    for i in (0..n).filter(|&i| coupled[i]) {
        let v = nid[loc[i]];
        local[i] = ndof[v];
        ndof[v] += 1;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nc];
    for c in (0..n).filter(|&c| coupled[c]) {
        for p in indptr[c] as usize..indptr[c + 1] as usize {
            let r = indices[p] as usize;
            if coupled[r] && nid[loc[r]] != nid[loc[c]] {
                adj[nid[loc[c]]].push(nid[loc[r]]);
                adj[nid[loc[r]]].push(nid[loc[c]]);
            }
        }
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let sym = sparse::analyse_with_leaf(&xyz, &adj, &ndof, FFI_LEAF);
    if std::env::var("RUSTYFEM_FRONTS").is_ok() {
        for (s, k, u, c) in sym.fronts() {
            eprintln!("front {s} {k} {u} {c}");
        }
    }
    let eq: Vec<usize> = (0..n).map(|i| if coupled[i] { sym.first_dof[nid[loc[i]]] + local[i] } else { NONE }).collect();
    let mut owner = vec![0usize; sym.n];
    for (i, &q) in eq.iter().enumerate().filter(|(_, &q)| q != NONE) {
        owner[q] = i;
    }
    let mut k = sym.pattern(&adj, &ndof);
    drop(adj);
    if mode == Mode::Lu {
        // unsymmetric: all entries (both triangles required)
        let mut full = Full::from_pattern(k);
        for c in (0..n).filter(|&c| coupled[c]) {
            let qc = eq[c];
            for p in indptr[c] as usize..indptr[c + 1] as usize {
                let qr = eq[indices[p] as usize];
                if qr != NONE {
                    full.add(qr, qc, data[p]);
                }
            }
        }
        let fac = lu::factor(&sym, &full).map_err(|q| owner[q] as i64)?;
        let size = fac.factor_size();
        return Ok((Handle { fac: Fac::Lu(fac), eq, iso, n_sym: sym.n, work: sym.work(), factor_size: size, inertia: None }, Vec::new()));
    }
    if mode == Mode::Indef {
        // symmetric: the caller's lower triangle is enough (both or only the lower one may be given)
        for c in (0..n).filter(|&c| coupled[c]) {
            let qc = eq[c];
            for p in indptr[c] as usize..indptr[c + 1] as usize {
                let r = indices[p] as usize;
                let qr = eq[r];
                if qr != NONE && r >= c {
                    k.add(qr.max(qc), qr.min(qc), data[p]);
                }
            }
        }
        let fac = indef::factor(&sym, &k).map_err(|q| owner[q] as i64)?;
        let mut ine: Inertia = fac.inertia;
        for &(_, d) in &iso {
            if d < 0.0 {
                ine.neg += 1;
            } else {
                ine.pos += 1;
            }
        }
        let size = fac.factor_size();
        let inertia = Some([ine.neg as i64, ine.zero as i64, ine.pos as i64]);
        return Ok((Handle { fac: Fac::Indef(fac), eq, iso, n_sym: sym.n, work: sym.work(), factor_size: size, inertia }, Vec::new()));
    }
    for c in (0..n).filter(|&c| coupled[c]) {
        let qc = eq[c];
        for p in indptr[c] as usize..indptr[c + 1] as usize {
            let qr = eq[indices[p] as usize];
            if qr != NONE && qr >= qc {
                k.add(qr, qc, data[p]);
            }
        }
    }
    let fac = sparse::factor(&sym, &k).map_err(|q| owner[q] as i64)?;
    let dq = fac.pivots(sym.n);
    let d = (0..n).map(|i| if coupled[i] { dq[eq[i]] } else { diag[i] }).collect();
    Ok((Handle { fac: Fac::Pd(fac), eq, iso, n_sym: sym.n, work: sym.work(), factor_size: sym.factor_size(), inertia: Some([0, 0, n as i64]) }, d))
}

/// Common entry point for the new factorizations (without d_out).
#[allow(clippy::too_many_arguments)]
unsafe fn factor_mode(
    n: usize,
    nn: usize,
    loc: *const i64,
    coords: *const f64,
    indptr: *const i64,
    indices: *const i64,
    data: *const f64,
    err_dof: *mut i64,
    mode: Mode,
) -> *mut Handle {
    *err_dof = -1;
    let loc = slice::from_raw_parts(loc, n);
    let coords = slice::from_raw_parts(coords, 3 * nn);
    let indptr = slice::from_raw_parts(indptr, n + 1);
    let nnz = indptr[n] as usize;
    let indices = slice::from_raw_parts(indices, nnz);
    let data = slice::from_raw_parts(data, nnz);
    match catch_unwind(AssertUnwindSafe(|| build(n, nn, loc, coords, indptr, indices, data, mode))) {
        Ok(Ok((h, _))) => Box::into_raw(Box::new(h)),
        Ok(Err(q)) => {
            *err_dof = q;
            std::ptr::null_mut()
        }
        Err(_) => std::ptr::null_mut(),
    }
}

/// Factorizes a symmetric, **indefinite** matrix (P·A·Pᵀ = L·D·Lᵀ with 1×1
/// and 2×2 blocks in D, Bunch–Kaufman pivoting within the fronts and delayed
/// pivots). Same input as `rustyfem_factor` (both triangles or only the
/// lower one). `inertia_out` (3 elements, may be null) receives [number of negative, zero,
/// positive] pivots, i.e. eigenvalues by Sylvester's law of inertia. Null on
/// error: `*err_dof` is the degree of freedom where the matrix is singular (caller's
/// numbering), −1 for invalid input. The handle is used with `rustyfem_solve`,
/// `rustyfem_info`, `rustyfem_order`, `rustyfem_inertia` and `rustyfem_free`.
///
/// # Safety
/// As `rustyfem_factor`; `inertia_out` null or 3 elements.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_factor_indef(
    n: usize,
    nn: usize,
    loc: *const i64,
    coords: *const f64,
    indptr: *const i64,
    indices: *const i64,
    data: *const f64,
    inertia_out: *mut i64,
    err_dof: *mut i64,
) -> *mut Handle {
    let h = factor_mode(n, nn, loc, coords, indptr, indices, data, err_dof, Mode::Indef);
    if !h.is_null() && !inertia_out.is_null() {
        let i = (*h).inertia.unwrap_or([-1; 3]);
        slice::from_raw_parts_mut(inertia_out, 3).copy_from_slice(&i);
    }
    h
}

/// Factorizes an **unsymmetric** matrix with multifrontal LU (threshold pivoting,
/// delayed pivots, partial pivoting at the root). CSC with all entries (both
/// triangles); the pattern is symmetrized internally. Null on error as
/// `rustyfem_factor_indef`. Solve A x = b with `rustyfem_solve`.
///
/// # Safety
/// As `rustyfem_factor`.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_factor_unsym(
    n: usize,
    nn: usize,
    loc: *const i64,
    coords: *const f64,
    indptr: *const i64,
    indices: *const i64,
    data: *const f64,
    err_dof: *mut i64,
) -> *mut Handle {
    factor_mode(n, nn, loc, coords, indptr, indices, data, err_dof, Mode::Lu)
}

/// The inertia [negative, zero, positive] for a handle from `rustyfem_factor`
/// (always [0, 0, n]) or `rustyfem_factor_indef`. Returns 0, or −1 if
/// the inertia is not available (LU handle).
///
/// # Safety
/// `h` from a factorization, `out` with 3 elements.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_inertia(h: *const Handle, out: *mut i64) -> i32 {
    let h = &*h;
    match h.inertia {
        Some(i) => {
            slice::from_raw_parts_mut(out, 3).copy_from_slice(&i);
            0
        }
        None => -1,
    }
}

/// Solves K X = B in place for `nrhs` right-hand sides, each stored
/// contiguously (column-wise, length n). The right-hand sides are solved in parallel.
///
/// # Safety
/// `h` from `rustyfem_factor`, `b` with n·nrhs elements.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_solve(h: *const Handle, nrhs: usize, b: *mut f64) -> i32 {
    let h = &*h;
    let n = h.eq.len();
    if n == 0 || nrhs == 0 {
        return 0;
    }
    let b = slice::from_raw_parts_mut(b, n * nrhs);
    let one = |col: &mut [f64]| {
        let mut x = vec![0.0; h.n_sym];
        for (i, &q) in h.eq.iter().enumerate().filter(|(_, &q)| q != NONE) {
            x[q] = col[i];
        }
        h.fac.solve(&mut x);
        for (i, &q) in h.eq.iter().enumerate().filter(|(_, &q)| q != NONE) {
            col[i] = x[q];
        }
        for &(i, d) in &h.iso {
            col[i] /= d;
        }
    };
    let res = catch_unwind(AssertUnwindSafe(|| {
        let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(1).min(nrhs);
        if threads <= 1 {
            b.chunks_mut(n).for_each(one);
        } else {
            let per = nrhs.div_ceil(threads);
            std::thread::scope(|s| {
                for chunk in b.chunks_mut(n * per) {
                    s.spawn(|| chunk.chunks_mut(n).for_each(one));
                }
            });
        }
    }));
    if res.is_ok() { 0 } else { -1 }
}

/// Estimated number of flops in the factorization and number of stored values in L.
///
/// # Safety
/// `h` from `rustyfem_factor`, `out` with 2 elements.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_info(h: *const Handle, out: *mut f64) {
    let h = &*h;
    let out = slice::from_raw_parts_mut(out, 2);
    out[0] = h.work;
    out[1] = h.factor_size as f64;
}

/// Internal equation for each degree of freedom (caller's numbering), for diagnostics.
/// Uncoupled degrees of freedom are numbered last.
///
/// # Safety
/// `h` from `rustyfem_factor`, `out` with n elements.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_order(h: *const Handle, out: *mut i64) {
    let h = &*h;
    let out = slice::from_raw_parts_mut(out, h.eq.len());
    let mut next = h.n_sym;
    for (o, &q) in out.iter_mut().zip(&h.eq) {
        *o = if q == NONE { next += 1; (next - 1) as i64 } else { q as i64 };
    }
}

/// # Safety
/// `h` from `rustyfem_factor` (or null), at most once.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[cfg(test)]
mod tests {
    //! The new FFI functions against dense solutions (hand-computed 1D chains).
    use super::*;

    /// Chain with n degrees of freedom (one per node), CSC with both triangles.
    fn chain(n: usize, diag: impl Fn(usize) -> f64, off: impl Fn(usize, usize) -> f64) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for c in 0..n {
            for r in c.saturating_sub(1)..(c + 2).min(n) {
                indices.push(r as i64);
                data.push(if r == c { diag(c) } else { off(r, c) });
            }
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    }

    #[test]
    fn indefinite_and_unsymmetric_handles_solve_and_report_inertia() {
        let n = 40;
        let loc: Vec<i64> = (0..n as i64).collect();
        let coords: Vec<f64> = (0..n).flat_map(|i| [i as f64, 0.0, 0.0]).collect();
        // K − σI for K = tridiag(−1, 2, −1): eigenvalues 2 − 2cos(kπ/(n+1)) − σ
        let sigma = 1.3;
        let (ip, ix, d) = chain(n, |_| 2.0 - sigma, |_, _| -1.0);
        let mut err = 0i64;
        let mut ine = [0i64; 3];
        unsafe {
            // the positive definite path must refuse
            assert!(rustyfem_factor(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), std::ptr::null_mut(), &mut err).is_null());
            let h = rustyfem_factor_indef(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), ine.as_mut_ptr(), &mut err);
            assert!(!h.is_null());
            let below = (1..=n).filter(|&k| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos() < sigma).count() as i64;
            assert_eq!(ine, [below, 0, n as i64 - below]);
            let mut b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
            let b0 = b.clone();
            assert_eq!(rustyfem_solve(h, 1, b.as_mut_ptr()), 0);
            for i in 0..n {
                let ax = (2.0 - sigma) * b[i] - if i > 0 { b[i - 1] } else { 0.0 } - if i + 1 < n { b[i + 1] } else { 0.0 };
                assert!((ax - b0[i]).abs() < 1e-10, "row {i}");
            }
            let mut out = [0i64; 3];
            assert_eq!(rustyfem_inertia(h, out.as_mut_ptr()), 0);
            assert_eq!(out, ine);
            rustyfem_free(h);

            // unsymmetric convection–diffusion
            let (ip, ix, d) = chain(n, |_| 2.0, |r, c| if r > c { -1.3 } else { -0.7 });
            let h = rustyfem_factor_unsym(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), &mut err);
            assert!(!h.is_null());
            let mut b = b0.clone();
            assert_eq!(rustyfem_solve(h, 1, b.as_mut_ptr()), 0);
            for i in 0..n {
                // A(i, i−1) = −1.3, A(i, i+1) = −0.7
                let ax = 2.0 * b[i] - if i > 0 { 1.3 * b[i - 1] } else { 0.0 } - if i + 1 < n { 0.7 * b[i + 1] } else { 0.0 };
                assert!((ax - b0[i]).abs() < 1e-10, "row {i}");
            }
            assert_eq!(rustyfem_inertia(h, out.as_mut_ptr()), -1);
            rustyfem_free(h);

            // singular: zero matrix in one degree of freedom
            let (ip, ix, d) = chain(n, |c| if c == 7 { 0.0 } else { 1.0 }, |_, _| 0.0);
            let h = rustyfem_factor_indef(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), std::ptr::null_mut(), &mut err);
            assert!(h.is_null());
            assert_eq!(err, 7);
        }
    }
}
