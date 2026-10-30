//! Gles radlagrad matris (CSR) och de enkla förkonditionerarna Jacobi och IC(0).

use crate::sparse::Lower;

/// Gles matris i radformat (båda trianglarna för symmetriska matriser).
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub nrows: usize,
    pub ncols: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<usize>,
    pub val: Vec<f64>,
}

/// Under så här många rader görs matris–vektorprodukten på en tråd.
const PAR_ROWS: usize = 20_000;

impl Csr {
    /// Full symmetrisk matris från undre triangeln i kolumnformat.
    pub fn from_lower(a: &Lower) -> Csr {
        let n = a.colptr.len() - 1;
        let mut cnt = vec![0usize; n + 1];
        for c in 0..n {
            for p in a.colptr[c]..a.colptr[c + 1] {
                let r = a.rowidx[p];
                cnt[r + 1] += 1;
                if r != c {
                    cnt[c + 1] += 1;
                }
            }
        }
        for i in 0..n {
            cnt[i + 1] += cnt[i];
        }
        let nnz = cnt[n];
        let mut pos = cnt.clone();
        let mut idx = vec![0usize; nnz];
        let mut val = vec![0.0; nnz];
        // kolumnvis genomgång ger stigande kolumnindex i varje rad
        for c in 0..n {
            for p in a.colptr[c]..a.colptr[c + 1] {
                let r = a.rowidx[p];
                idx[pos[r]] = c;
                val[pos[r]] = a.val[p];
                pos[r] += 1;
            }
        }
        // den övre triangeln: rad c får kolumnerna r > c, men de måste sorteras in
        let mut m = Csr { nrows: n, ncols: n, ptr: cnt, idx, val };
        for c in 0..n {
            for p in a.colptr[c]..a.colptr[c + 1] {
                let r = a.rowidx[p];
                if r != c {
                    m.idx[pos[c]] = r;
                    m.val[pos[c]] = a.val[p];
                    pos[c] += 1;
                }
            }
        }
        // varje rad: först kolumner ≤ rad (stigande), sedan > rad (stigande) – redan sorterat
        m
    }

    pub fn nnz(&self) -> usize {
        self.idx.len()
    }

    #[inline]
    fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for p in self.ptr[i]..self.ptr[i + 1] {
            s += self.val[p] * x[self.idx[p]];
        }
        s
    }

    /// y = A x (parallellt för stora matriser).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let n = self.nrows;
        let t = crate::par::threads().min(n / PAR_ROWS).max(1);
        if t == 1 {
            for (i, yi) in y.iter_mut().enumerate().take(n) {
                *yi = self.row_dot(i, x);
            }
            return;
        }
        let chunk = n.div_ceil(t);
        std::thread::scope(|s| {
            for (ci, yc) in y[..n].chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (k, yi) in yc.iter_mut().enumerate() {
                        *yi = self.row_dot(ci * chunk + k, x);
                    }
                });
            }
        });
    }

    pub fn diag(&self) -> Vec<f64> {
        (0..self.nrows)
            .map(|i| {
                let (lo, hi) = (self.ptr[i], self.ptr[i + 1]);
                match self.idx[lo..hi].binary_search(&i) {
                    Ok(p) => self.val[lo + p],
                    Err(_) => self.idx[lo..hi].iter().position(|&c| c == i).map(|p| self.val[lo + p]).unwrap_or(0.0),
                }
            })
            .collect()
    }

    pub fn transpose(&self) -> Csr {
        let mut cnt = vec![0usize; self.ncols + 1];
        for &c in &self.idx {
            cnt[c + 1] += 1;
        }
        for i in 0..self.ncols {
            cnt[i + 1] += cnt[i];
        }
        let mut pos = cnt.clone();
        let mut idx = vec![0usize; self.nnz()];
        let mut val = vec![0.0; self.nnz()];
        for r in 0..self.nrows {
            for p in self.ptr[r]..self.ptr[r + 1] {
                let c = self.idx[p];
                idx[pos[c]] = r;
                val[pos[c]] = self.val[p];
                pos[c] += 1;
            }
        }
        Csr { nrows: self.ncols, ncols: self.nrows, ptr: cnt, idx, val }
    }

    /// C = A·B (Gustavson), med sorterade kolumnindex.
    pub fn mul(&self, b: &Csr) -> Csr {
        assert_eq!(self.ncols, b.nrows);
        let n = self.nrows;
        let t = crate::par::threads().min(n / 2000).max(1);
        let chunk = n.div_ceil(t).max(1);
        let work = |r0: usize, r1: usize| -> (Vec<usize>, Vec<usize>, Vec<f64>) {
            let mut mark = vec![usize::MAX; b.ncols];
            let mut acc = vec![0.0; b.ncols];
            let mut cnt = Vec::with_capacity(r1 - r0);
            let mut idx = Vec::new();
            let mut val = Vec::new();
            let mut cols: Vec<usize> = Vec::new();
            for i in r0..r1 {
                cols.clear();
                for p in self.ptr[i]..self.ptr[i + 1] {
                    let (k, a) = (self.idx[p], self.val[p]);
                    for q in b.ptr[k]..b.ptr[k + 1] {
                        let c = b.idx[q];
                        if mark[c] != i {
                            mark[c] = i;
                            acc[c] = 0.0;
                            cols.push(c);
                        }
                        acc[c] += a * b.val[q];
                    }
                }
                cols.sort_unstable();
                for &c in &cols {
                    idx.push(c);
                    val.push(acc[c]);
                }
                cnt.push(cols.len());
            }
            (cnt, idx, val)
        };
        let parts: Vec<(Vec<usize>, Vec<usize>, Vec<f64>)> = if t == 1 {
            vec![work(0, n)]
        } else {
            std::thread::scope(|s| {
                let hs: Vec<_> = (0..t).map(|c| (c * chunk).min(n)).map(|r0| s.spawn(move || work(r0, (r0 + chunk).min(n)))).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        let mut ptr = Vec::with_capacity(n + 1);
        ptr.push(0);
        let mut idx = Vec::new();
        let mut val = Vec::new();
        for (cnt, i, v) in parts {
            for c in cnt {
                ptr.push(ptr.last().unwrap() + c);
            }
            idx.extend(i);
            val.extend(v);
        }
        Csr { nrows: n, ncols: b.ncols, ptr, idx, val }
    }
}

/// Förkonditionerare z = M⁻¹ r.
pub trait Precond: Send + Sync {
    fn apply(&self, r: &[f64], z: &mut [f64]);
    fn name(&self) -> String;
}

pub struct Jacobi {
    inv: Vec<f64>,
}

impl Jacobi {
    pub fn new(a: &Csr) -> Jacobi {
        Jacobi { inv: a.diag().iter().map(|&d| if d != 0.0 { 1.0 / d } else { 1.0 }).collect() }
    }
}

impl Precond for Jacobi {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for ((zi, ri), di) in z.iter_mut().zip(r).zip(&self.inv) {
            *zi = ri * di;
        }
    }
    fn name(&self) -> String {
        "Jacobi".into()
    }
}

/// Ofullständig Cholesky utan utfyllnad, A ≈ L·D·Lᵀ på A:s mönster.
/// Vid sammanbrott (icke-positiv pivot) görs om med diagonalskift
/// A + α·diag(A) (Manteuffel 1980), α = 10⁻³, 4·10⁻³, ...
pub struct Ic0 {
    /// Strikt undre L radvis.
    ptr: Vec<usize>,
    idx: Vec<usize>,
    val: Vec<f64>,
    d: Vec<f64>,
    pub shift: f64,
}

impl Ic0 {
    pub fn new(a: &Csr) -> Ic0 {
        let n = a.nrows;
        let mut ptr = vec![0usize; n + 1];
        let mut idx = Vec::new();
        let mut a_val = Vec::new();
        let mut diag = vec![0.0; n];
        for i in 0..n {
            for p in a.ptr[i]..a.ptr[i + 1] {
                let j = a.idx[p];
                if j < i {
                    idx.push(j);
                    a_val.push(a.val[p]);
                } else if j == i {
                    diag[i] += a.val[p];
                }
            }
            ptr[i + 1] = idx.len();
        }
        let mut shift = 0.0;
        loop {
            match Self::try_factor(n, &ptr, &idx, &a_val, &diag, shift) {
                Some((val, d)) => return Ic0 { ptr, idx, val, d, shift },
                None => shift = if shift == 0.0 { 1e-3 } else { shift * 4.0 },
            }
        }
    }

    fn try_factor(n: usize, ptr: &[usize], idx: &[usize], a: &[f64], diag: &[f64], shift: f64) -> Option<(Vec<f64>, Vec<f64>)> {
        let mut val = a.to_vec();
        let mut d = vec![0.0; n];
        let mut mark = vec![usize::MAX; n];
        for i in 0..n {
            let (lo, hi) = (ptr[i], ptr[i + 1]);
            for p in lo..hi {
                mark[idx[p]] = p;
            }
            for p in lo..hi {
                let j = idx[p];
                let mut s = val[p];
                for q in ptr[j]..ptr[j + 1] {
                    let k = idx[q];
                    let pk = mark[k];
                    if pk != usize::MAX && k < j {
                        s -= val[pk] * d[k] * val[q];
                    }
                }
                val[p] = s / d[j];
            }
            let mut di = diag[i] * (1.0 + shift);
            for p in lo..hi {
                di -= val[p] * val[p] * d[idx[p]];
            }
            for p in lo..hi {
                mark[idx[p]] = usize::MAX;
            }
            if !(di > 1e-12 * diag[i].abs()) {
                return None;
            }
            d[i] = di;
        }
        Some((val, d))
    }
}

impl Precond for Ic0 {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.d.len();
        z[..n].copy_from_slice(&r[..n]);
        for i in 0..n {
            let mut s = z[i];
            for p in self.ptr[i]..self.ptr[i + 1] {
                s -= self.val[p] * z[self.idx[p]];
            }
            z[i] = s;
        }
        for i in 0..n {
            z[i] /= self.d[i];
        }
        for i in (0..n).rev() {
            let zi = z[i];
            for p in self.ptr[i]..self.ptr[i + 1] {
                z[self.idx[p]] -= self.val[p] * zi;
            }
        }
    }
    fn name(&self) -> String {
        if self.shift > 0.0 {
            format!("IC(0), skift {:.1e}", self.shift)
        } else {
            "IC(0)".into()
        }
    }
}

/// Resultat av en PCG-lösning.
#[derive(Debug, Clone, Copy)]
pub struct PcgReport {
    pub iterations: usize,
    pub rel_residual: f64,
    pub converged: bool,
}

#[inline]
fn dotv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Förkonditionerad konjugerad gradient för A x = b, start x = 0.
/// Avbryter när ‖b − A x‖ ≤ tol·‖b‖.
pub fn pcg(a: &Csr, b: &[f64], x: &mut [f64], m: &dyn Precond, tol: f64, maxit: usize) -> PcgReport {
    let n = b.len();
    x.iter_mut().for_each(|v| *v = 0.0);
    let bn = dotv(b, b).sqrt();
    if bn == 0.0 {
        return PcgReport { iterations: 0, rel_residual: 0.0, converged: true };
    }
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    m.apply(&r, &mut z);
    let mut p = z.clone();
    let mut q = vec![0.0; n];
    let mut rz = dotv(&r, &z);
    let mut rel = 1.0;
    for it in 1..=maxit {
        a.matvec(&p, &mut q);
        let pq = dotv(&p, &q);
        if !(pq > 0.0) {
            // inte positivt definit (eller sammanbrott)
            return PcgReport { iterations: it, rel_residual: rel, converged: false };
        }
        let alpha = rz / pq;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        rel = dotv(&r, &r).sqrt() / bn;
        if rel <= tol {
            return PcgReport { iterations: it, rel_residual: rel, converged: true };
        }
        m.apply(&r, &mut z);
        let rz_new = dotv(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }
    PcgReport { iterations: maxit, rel_residual: rel, converged: false }
}
