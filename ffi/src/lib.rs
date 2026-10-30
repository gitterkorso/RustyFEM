//! C-gränssnitt till RustyFEMs glesa lösare (`rustyfem::sparse`).
//!
//! Indata är en symmetrisk matris i CSC-format (båda trianglarna eller bara
//! den undre) i anroparens numrering, samt för varje frihetsgrad vilken nod
//! den hör till och nodernas koordinater. Ordningen (nested dissection på
//! nodnivå) och faktoriseringen görs här; anroparen ser bara sin egen
//! numrering. Inga externa beroenden, så biblioteket kan laddas med ctypes.

use rustyfem::sparse::indef::{self, IndefFactor, Inertia};
use rustyfem::sparse::lu::{self, Full, LuFactor};
use rustyfem::sparse::{self, Factor};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

/// Lövstorlek för FFI-vägen: mätt på openFEM:s skal- och solidmodeller ger
/// 16 noder per löv ungefär halva fyllnaden jämfört med standardvärdet 48.
const FFI_LEAF: usize = 16;
const NONE: usize = usize::MAX;

/// Faktortyp: positivt definit LDLᵀ, indefinit LDLᵀ (1×1/2×2) eller osymmetrisk LU.
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
    /// Intern ekvation för varje frihetsgrad i anroparens numrering (NONE = okopplad).
    eq: Vec<usize>,
    /// Okopplade frihetsgrader och deras diagonal.
    iso: Vec<(usize, f64)>,
    n_sym: usize,
    work: f64,
    factor_size: usize,
    /// Tröghet [negativa, noll, positiva]; None för LU.
    inertia: Option<[i64; 3]>,
}

/// Faktoriserar matrisen. Returnerar null vid fel: `*err_dof` blir då den
/// frihetsgrad (anroparens numrering) där en icke-positiv pivot uppstod, eller
/// −1 vid ogiltig indata. `d_out` (längd n, får vara null) fylls med pivoterna D.
///
/// # Safety
/// Pekarna ska peka på arrayer av angiven längd: `loc` och `d_out` n,
/// `coords` 3·nn, `indptr` n+1, `indices` och `data` indptr[n].
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
    // Frihetsgrader utan kopplingar (bara diagonalterm, t.ex. stabiliserade
    // rotationer i solidnoder) löses direkt. Annars hamnar de i nodens täta
    // block och fyllnaden blir (6/3)² = 4 gånger för stor.
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
    // noder med minst en kopplad frihetsgrad, i kompakt numrering
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
    // frihetsgradernas plats inom noden i stigande ordning
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
        // osymmetrisk: alla poster (båda trianglarna krävs)
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
        // symmetrisk: anroparens undre triangel räcker (båda eller bara den undre får anges)
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

/// Gemensam ingång för de nya faktoriseringarna (utan d_out).
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

/// Faktoriserar en symmetrisk, **indefinit** matris (P·A·Pᵀ = L·D·Lᵀ med 1×1-
/// och 2×2-block i D, Bunch–Kaufman-pivotering inom fronterna och fördröjda
/// pivoter). Samma indata som `rustyfem_factor` (båda trianglarna eller bara den
/// undre). `inertia_out` (3 element, får vara null) får [antal negativa, noll,
/// positiva] pivoter, dvs. egenvärden enligt Sylvesters tröghetslag. Null vid
/// fel: `*err_dof` blir frihetsgraden där matrisen är singulär (anroparens
/// numrering), −1 vid ogiltig indata. Handtaget används med `rustyfem_solve`,
/// `rustyfem_info`, `rustyfem_order`, `rustyfem_inertia` och `rustyfem_free`.
///
/// # Safety
/// Som `rustyfem_factor`; `inertia_out` null eller 3 element.
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

/// Faktoriserar en **osymmetrisk** matris med multifrontal LU (tröskelpivotering,
/// fördröjda pivoter, partiell pivotering i roten). CSC med alla poster (båda
/// trianglarna); mönstret görs symmetriskt internt. Null vid fel som
/// `rustyfem_factor_indef`. Lös A x = b med `rustyfem_solve`.
///
/// # Safety
/// Som `rustyfem_factor`.
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

/// Trögheten [negativa, noll, positiva] för ett handtag från `rustyfem_factor`
/// (alltid [0, 0, n]) eller `rustyfem_factor_indef`. Returnerar 0, eller −1 om
/// trögheten inte finns (LU-handtag).
///
/// # Safety
/// `h` från en faktorisering, `out` med 3 element.
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

/// Löser K X = B på plats för `nrhs` högerled, vart och ett lagrat
/// sammanhängande (kolumnvis, längd n). Högerleden löses parallellt.
///
/// # Safety
/// `h` från `rustyfem_factor`, `b` med n·nrhs element.
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

/// Uppskattat antal flyttal i faktoriseringen och antal lagrade tal i L.
///
/// # Safety
/// `h` från `rustyfem_factor`, `out` med 2 element.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_info(h: *const Handle, out: *mut f64) {
    let h = &*h;
    let out = slice::from_raw_parts_mut(out, 2);
    out[0] = h.work;
    out[1] = h.factor_size as f64;
}

/// Intern ekvation för varje frihetsgrad (anroparens numrering), för diagnostik.
/// Okopplade frihetsgrader numreras sist.
///
/// # Safety
/// `h` från `rustyfem_factor`, `out` med n element.
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
/// `h` från `rustyfem_factor` (eller null), högst en gång.
#[no_mangle]
pub unsafe extern "C" fn rustyfem_free(h: *mut Handle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

#[cfg(test)]
mod tests {
    //! De nya FFI-funktionerna mot täta lösningar (handräknade 1D-kedjor).
    use super::*;

    /// Kedja med n frihetsgrader (en per nod), CSC med båda trianglarna.
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
        // K − σI för K = tridiag(−1, 2, −1): egenvärden 2 − 2cos(kπ/(n+1)) − σ
        let sigma = 1.3;
        let (ip, ix, d) = chain(n, |_| 2.0 - sigma, |_, _| -1.0);
        let mut err = 0i64;
        let mut ine = [0i64; 3];
        unsafe {
            // den positivt definita vägen ska vägra
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
                assert!((ax - b0[i]).abs() < 1e-10, "rad {i}");
            }
            let mut out = [0i64; 3];
            assert_eq!(rustyfem_inertia(h, out.as_mut_ptr()), 0);
            assert_eq!(out, ine);
            rustyfem_free(h);

            // osymmetrisk konvektion–diffusion
            let (ip, ix, d) = chain(n, |_| 2.0, |r, c| if r > c { -1.3 } else { -0.7 });
            let h = rustyfem_factor_unsym(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), &mut err);
            assert!(!h.is_null());
            let mut b = b0.clone();
            assert_eq!(rustyfem_solve(h, 1, b.as_mut_ptr()), 0);
            for i in 0..n {
                // A(i, i−1) = −1.3, A(i, i+1) = −0.7
                let ax = 2.0 * b[i] - if i > 0 { 1.3 * b[i - 1] } else { 0.0 } - if i + 1 < n { 0.7 * b[i + 1] } else { 0.0 };
                assert!((ax - b0[i]).abs() < 1e-10, "rad {i}");
            }
            assert_eq!(rustyfem_inertia(h, out.as_mut_ptr()), -1);
            rustyfem_free(h);

            // singulär: nollmatris i en frihetsgrad
            let (ip, ix, d) = chain(n, |c| if c == 7 { 0.0 } else { 1.0 }, |_, _| 0.0);
            let h = rustyfem_factor_indef(n, n, loc.as_ptr(), coords.as_ptr(), ip.as_ptr(), ix.as_ptr(), d.as_ptr(), std::ptr::null_mut(), &mut err);
            assert!(h.is_null());
            assert_eq!(err, 7);
        }
    }
}
