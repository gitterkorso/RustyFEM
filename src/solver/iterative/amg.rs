//! Algebraisk multigrid med utjämnad aggregering (smoothed aggregation,
//! Vaněk–Mandel–Brezina 1996) som förkonditionerare till CG.
//!
//! - Aggregering på nodnivå (alla frihetsgrader i en nod i samma aggregat),
//!   med starka kopplingar ‖A_IJ‖ > θ·√(‖A_II‖·‖A_JJ‖) och Vaněks tre pass.
//! - Tentativ prolongering från nollrummet (stelkroppsrörelser beräknade ur
//!   nodkoordinaterna: 3 i 2D, 6 i 3D, med rotationsfrihetsgrader för balkar
//!   och skal), ortonormerad per aggregat med Gram–Schmidt.
//! - Utjämning P = (I − ω D⁻¹ A) P̃, ω = 4/(3 ρ(D⁻¹A)), Galerkin A_c = Pᵀ A P.
//! - V-cykel med symmetrisk Gauss–Seidel (framåt före, bakåt efter), tät
//!   Cholesky på den grövsta nivån.

use super::csr::{Csr, Precond};

/// Grövsta nivåns största storlek.
const COARSE: usize = 1500;
const MAX_LEVELS: usize = 12;

struct Level {
    a: Csr,
    diag: Vec<f64>,
    /// Prolongering (fin × grov) och restriktion (grov × fin).
    p: Csr,
    r: Csr,
}

pub struct Amg {
    levels: Vec<Level>,
    coarse_a: usize,
    /// Tät LDLᵀ (Cholesky utan rot) av den grövsta matrisen, radvis.
    coarse_l: Vec<f64>,
    coarse_d: Vec<f64>,
    pub sizes: Vec<usize>,
    pub complexity: f64,
}

/// Nollrum (stelkroppsrörelser) för frihetsgraderna. `bptr`: blockens
/// frihetsgradsintervall, `coords`: blockens koordinater.
pub fn rigid_body_modes(coords: &[[f64; 3]], ndof: &[usize]) -> (Vec<f64>, usize) {
    let planar = coords.iter().all(|c| c[2] == 0.0);
    let nb = if planar { 3 } else { 6 };
    let n: usize = ndof.iter().sum();
    let mut c0 = [0.0; 3];
    for c in coords {
        for a in 0..3 {
            c0[a] += c[a] / coords.len().max(1) as f64;
        }
    }
    let mut b = vec![0.0; n * nb];
    let mut row = 0;
    for (v, &d) in ndof.iter().enumerate() {
        let x = [coords[v][0] - c0[0], coords[v][1] - c0[1], coords[v][2] - c0[2]];
        for l in 0..d {
            let br = &mut b[(row + l) * nb..(row + l + 1) * nb];
            if planar {
                // (U1, U2) eller (U1, U2, UR3)
                match (d, l) {
                    (_, 0) => {
                        br[0] = 1.0;
                        br[2] = -x[1];
                    }
                    (_, 1) => {
                        br[1] = 1.0;
                        br[2] = x[0];
                    }
                    (3, 2) => br[2] = 1.0,
                    _ => br[l % nb] = 1.0,
                }
            } else {
                // translationer t_a och rotationer θ_a × x
                match l {
                    0 => {
                        br[0] = 1.0;
                        br[4] = x[2];
                        br[5] = -x[1];
                    }
                    1 => {
                        br[1] = 1.0;
                        br[3] = -x[2];
                        br[5] = x[0];
                    }
                    2 => {
                        br[2] = 1.0;
                        br[3] = x[1];
                        br[4] = -x[0];
                    }
                    3..=5 if d >= 6 => br[l] = 1.0,
                    _ => br[l % nb] = 1.0,
                }
            }
        }
        row += d;
    }
    (b, nb)
}

fn block_of(bptr: &[usize]) -> Vec<usize> {
    let mut o = vec![0usize; *bptr.last().unwrap()];
    for b in 0..bptr.len() - 1 {
        for x in o.iter_mut().take(bptr[b + 1]).skip(bptr[b]) {
            *x = b;
        }
    }
    o
}

/// Aggregat för varje block (Vaněks tre pass).
fn aggregate(a: &Csr, bptr: &[usize], theta: f64) -> (Vec<usize>, usize) {
    let nb = bptr.len() - 1;
    let blk = block_of(bptr);
    // blocknormer i kvadrat per blockrad
    let mut dnorm = vec![0.0f64; nb];
    for i in 0..a.nrows {
        for p in a.ptr[i]..a.ptr[i + 1] {
            if blk[a.idx[p]] == blk[i] {
                dnorm[blk[i]] += a.val[p] * a.val[p];
            }
        }
    }
    let mut strong: Vec<Vec<usize>> = vec![Vec::new(); nb];
    let mut acc = vec![0.0f64; nb];
    let mut mark = vec![usize::MAX; nb];
    let mut list = Vec::new();
    for b in 0..nb {
        list.clear();
        for i in bptr[b]..bptr[b + 1] {
            for p in a.ptr[i]..a.ptr[i + 1] {
                let c = blk[a.idx[p]];
                if c == b {
                    continue;
                }
                if mark[c] != b {
                    mark[c] = b;
                    acc[c] = 0.0;
                    list.push(c);
                }
                acc[c] += a.val[p] * a.val[p];
            }
        }
        for &c in &list {
            if acc[c] > theta * theta * (dnorm[b] * dnorm[c]).sqrt() {
                strong[b].push(c);
            }
        }
    }
    const NONE: usize = usize::MAX;
    let mut agg = vec![NONE; nb];
    let mut n = 0;
    // pass 1: block vars alla starka grannar är fria bildar ett nytt aggregat
    for b in 0..nb {
        if agg[b] == NONE && strong[b].iter().all(|&c| agg[c] == NONE) {
            agg[b] = n;
            for &c in &strong[b] {
                agg[c] = n;
            }
            n += 1;
        }
    }
    // pass 2: anslut till ett angränsande aggregat
    let snapshot = agg.clone();
    for b in 0..nb {
        if agg[b] == NONE {
            if let Some(&c) = strong[b].iter().find(|&&c| snapshot[c] != NONE) {
                agg[b] = snapshot[c];
            }
        }
    }
    // pass 3: resten
    for b in 0..nb {
        if agg[b] == NONE {
            agg[b] = n;
            for &c in &strong[b] {
                if agg[c] == NONE {
                    agg[c] = n;
                }
            }
            n += 1;
        }
    }
    (agg, n)
}

/// Tentativ prolongering och grovt nollrum. Returnerar (P̃, B_c, grova blockgränser).
fn tentative(bptr: &[usize], agg: &[usize], nagg: usize, b: &[f64], nb: usize) -> (Csr, Vec<f64>, Vec<usize>) {
    let n = *bptr.last().unwrap();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); nagg];
    for (blk, &g) in agg.iter().enumerate() {
        members[g].push(blk);
    }
    // rader i P̃: (rad, grov kolumn, värde)
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    let mut bc: Vec<f64> = Vec::new();
    let mut cptr = vec![0usize];
    let mut nc = 0;
    for mem in &members {
        let dofs: Vec<usize> = mem.iter().flat_map(|&blk| bptr[blk]..bptr[blk + 1]).collect();
        let nd = dofs.len();
        // kolumnvis kopia av B på aggregatet
        let mut q: Vec<Vec<f64>> = (0..nb).map(|c| dofs.iter().map(|&i| b[i * nb + c]).collect()).collect();
        let mut kept: Vec<Vec<f64>> = Vec::new();
        let mut r = vec![vec![0.0; nb]; nb.min(nd)];
        let scale = q.iter().map(|v| v.iter().map(|x| x * x).sum::<f64>()).fold(0.0, f64::max).sqrt();
        for c in 0..nb {
            let v = &mut q[c];
            for (k, qk) in kept.iter().enumerate() {
                let s: f64 = qk.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                r[k][c] = s;
                for (x, y) in v.iter_mut().zip(qk) {
                    *x -= s * y;
                }
            }
            // omortogonalisering för noggrannhet
            for (k, qk) in kept.iter().enumerate() {
                let s: f64 = qk.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                r[k][c] += s;
                for (x, y) in v.iter_mut().zip(qk) {
                    *x -= s * y;
                }
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if kept.len() < nd && norm > 1e-10 * scale.max(f64::MIN_POSITIVE) {
                let k = kept.len();
                r[k][c] = norm;
                kept.push(v.iter().map(|x| x / norm).collect());
            }
        }
        if kept.is_empty() {
            // inget nollrum på aggregatet: konstant vektor
            let s = 1.0 / (nd as f64).sqrt();
            kept.push(vec![s; nd]);
        }
        for (k, qk) in kept.iter().enumerate() {
            for (li, &i) in dofs.iter().enumerate() {
                if qk[li] != 0.0 {
                    entries.push((i, nc + k, qk[li]));
                }
            }
        }
        for k in 0..kept.len() {
            for c in 0..nb {
                bc.push(if k < r.len() { r[k][c] } else { 0.0 });
            }
        }
        nc += kept.len();
        cptr.push(nc);
    }
    entries.sort_unstable_by_key(|e| (e.0, e.1));
    let mut ptr = vec![0usize; n + 1];
    for e in &entries {
        ptr[e.0 + 1] += 1;
    }
    for i in 0..n {
        ptr[i + 1] += ptr[i];
    }
    let p = Csr { nrows: n, ncols: nc, ptr, idx: entries.iter().map(|e| e.1).collect(), val: entries.iter().map(|e| e.2).collect() };
    (p, bc, cptr)
}

/// Uppskattar ρ(D⁻¹A) med potensmetoden.
fn spectral_radius(a: &Csr, diag: &[f64]) -> f64 {
    let n = a.nrows;
    let mut x: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7919) % 13) as f64 / 13.0).collect();
    let mut y = vec![0.0; n];
    let mut rho = 1.0;
    for _ in 0..15 {
        let nx = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        x.iter_mut().for_each(|v| *v /= nx);
        a.matvec(&x, &mut y);
        for i in 0..n {
            y[i] /= diag[i];
        }
        rho = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        std::mem::swap(&mut x, &mut y);
    }
    rho * 1.05
}

impl Amg {
    /// Bygger hierarkin. `bptr`: blockens frihetsgradsintervall på fin nivå,
    /// `b`: nollrummet (n × nb, radvis). `Err` om den grövsta matrisen inte är
    /// positivt definit.
    pub fn new(a: Csr, mut bptr: Vec<usize>, mut b: Vec<f64>, nb: usize) -> Result<Amg, ()> {
        let mut levels = Vec::new();
        let mut a = a;
        let mut sizes = vec![a.nrows];
        let fine_nnz = a.nnz() as f64;
        let mut total_nnz = fine_nnz;
        let mut theta: f64 = 0.08;
        while a.nrows > COARSE && levels.len() < MAX_LEVELS {
            let diag = a.diag();
            if diag.iter().any(|&d| !(d > 0.0)) {
                return Err(());
            }
            let (agg, nagg) = aggregate(&a, &bptr, theta);
            let (pt, bc, cptr) = tentative(&bptr, &agg, nagg, &b, nb);
            if pt.ncols as f64 > 0.85 * a.nrows as f64 {
                break;
            }
            // P = (I − ω D⁻¹ A) P̃
            let omega = 4.0 / 3.0 / spectral_radius(&a, &diag);
            let ap = a.mul(&pt);
            let p = sub_scaled(&pt, &ap, &diag, omega);
            let r = p.transpose();
            let ac = r.mul(&a.mul(&p));
            total_nnz += ac.nnz() as f64;
            sizes.push(ac.nrows);
            levels.push(Level { a, diag, p, r });
            a = ac;
            bptr = cptr;
            b = bc;
            theta *= 0.5;
        }
        // tät LDLᵀ på den grövsta nivån
        let n = a.nrows;
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for p in a.ptr[i]..a.ptr[i + 1] {
                l[i * n + a.idx[p]] = a.val[p];
            }
        }
        let mut d = vec![0.0; n];
        let dmax = (0..n).map(|i| l[i * n + i].abs()).fold(0.0, f64::max);
        for j in 0..n {
            let mut dj = l[j * n + j];
            for k in 0..j {
                dj -= l[j * n + k] * l[j * n + k] * d[k];
            }
            if !(dj > 1e-13 * dmax) {
                return Err(());
            }
            d[j] = dj;
            for i in j + 1..n {
                let mut s = l[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k] * d[k];
                }
                l[i * n + j] = s / dj;
            }
        }
        Ok(Amg { levels, coarse_a: n, coarse_l: l, coarse_d: d, sizes, complexity: total_nnz / fine_nnz })
    }

    fn coarse_solve(&self, b: &[f64], x: &mut [f64]) {
        let n = self.coarse_a;
        let l = &self.coarse_l;
        x[..n].copy_from_slice(&b[..n]);
        for i in 0..n {
            let mut s = x[i];
            for k in 0..i {
                s -= l[i * n + k] * x[k];
            }
            x[i] = s;
        }
        for i in 0..n {
            x[i] /= self.coarse_d[i];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s -= l[k * n + i] * x[k];
            }
            x[i] = s;
        }
    }

    fn vcycle(&self, lev: usize, b: &[f64], x: &mut [f64]) {
        if lev == self.levels.len() {
            self.coarse_solve(b, x);
            return;
        }
        let l = &self.levels[lev];
        let a = &l.a;
        let n = a.nrows;
        x.iter_mut().for_each(|v| *v = 0.0);
        // framåt Gauss–Seidel
        for i in 0..n {
            let mut s = b[i];
            for p in a.ptr[i]..a.ptr[i + 1] {
                let j = a.idx[p];
                if j != i {
                    s -= a.val[p] * x[j];
                }
            }
            x[i] = s / l.diag[i];
        }
        let mut ax = vec![0.0; n];
        a.matvec(x, &mut ax);
        let res: Vec<f64> = b.iter().zip(&ax).map(|(bi, ai)| bi - ai).collect();
        let mut rc = vec![0.0; l.r.nrows];
        l.r.matvec(&res, &mut rc);
        let mut xc = vec![0.0; l.r.nrows];
        self.vcycle(lev + 1, &rc, &mut xc);
        let mut corr = vec![0.0; n];
        l.p.matvec(&xc, &mut corr);
        for i in 0..n {
            x[i] += corr[i];
        }
        // bakåt Gauss–Seidel
        for i in (0..n).rev() {
            let mut s = b[i];
            for p in a.ptr[i]..a.ptr[i + 1] {
                let j = a.idx[p];
                if j != i {
                    s -= a.val[p] * x[j];
                }
            }
            x[i] = s / l.diag[i];
        }
    }
}

/// P̃ − ω D⁻¹ (A P̃), radvis sammanslagning av två matriser med sorterade index.
fn sub_scaled(pt: &Csr, ap: &Csr, diag: &[f64], omega: f64) -> Csr {
    let n = pt.nrows;
    let mut ptr = vec![0usize];
    let mut idx = Vec::with_capacity(ap.nnz());
    let mut val = Vec::with_capacity(ap.nnz());
    for i in 0..n {
        let s = omega / diag[i];
        let (mut p, pe) = (pt.ptr[i], pt.ptr[i + 1]);
        let (mut q, qe) = (ap.ptr[i], ap.ptr[i + 1]);
        while p < pe || q < qe {
            let cp = if p < pe { pt.idx[p] } else { usize::MAX };
            let cq = if q < qe { ap.idx[q] } else { usize::MAX };
            let (c, v) = if cp == cq {
                let v = pt.val[p] - s * ap.val[q];
                p += 1;
                q += 1;
                (cp, v)
            } else if cp < cq {
                p += 1;
                (cp, pt.val[p - 1])
            } else {
                q += 1;
                (cq, -s * ap.val[q - 1])
            };
            if v != 0.0 {
                idx.push(c);
                val.push(v);
            }
        }
        ptr.push(idx.len());
    }
    Csr { nrows: n, ncols: pt.ncols, ptr, idx, val }
}

impl Precond for Amg {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        self.vcycle(0, r, z);
    }
    fn name(&self) -> String {
        format!("AMG (utjämnad aggregering, {} nivåer {:?}, operatorkomplexitet {:.2})", self.sizes.len(), self.sizes, self.complexity)
    }
}
