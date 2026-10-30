//! Multifrontal LDLᵀ for symmetric indefinite matrices.
//!
//! Same ordering and tree structure as the positive definite factorization
//! (`sparse::analyse`), but D gets 1×1 and 2×2 blocks:
//!
//! - Within a front, pivots are chosen among the fully summed variables with
//!   Bunch–Kaufman-type threshold pivoting (Duff–Reid, MA27/MA57): a 1×1 pivot
//!   is accepted if |a_jj| ≥ u·max_i |a_ij| and a 2×2 pivot (j, r) if
//!   |D⁻¹|·[max|a_ij|, max|a_ir|]ᵀ ≤ 1/u, where the maxima are taken over the whole front
//!   (including rows eliminated later). Default u = 0.01.
//! - Variables that fail the test are delayed (delayed pivots): they are passed
//!   on to the parent as extra fully summed variables together with
//!   the child's update matrix. At the root (no later rows) ordinary
//!   Bunch–Kaufman with α = (1+√17)/8 is used on the rest, which always gives a stable
//!   pivot if the matrix is regular.
//! - Inertia (number of negative, zero and positive eigenvalues) is computed from D using
//!   Sylvester's law of inertia; the number of negative ones gives the warning about
//!   negative eigenvalues.
//!
//! References: Bunch & Kaufman (1977), Duff & Reid (1983) "The multifrontal
//! solution of indefinite sparse symmetric linear equations", Liu (1992).

use super::{dot, kern, Lower, Symbolic, PAR_FLOPS};
use crate::par;

/// Default threshold u for pivoting within the fronts.
pub const THRESHOLD: f64 = 0.01;
/// A pivot counts as zero if |d| ≤ ZERO_TOL·(largest entry in the column of the original matrix).
pub const ZERO_TOL: f64 = 1e-12;
/// Bunch–Kaufman's α = (1 + √17)/8.
const ALPHA: f64 = 0.640_388_203_202_207_9;
/// Width of the pivot block.
const KB: usize = 64;

/// Inertia: number of negative, zero and positive eigenvalues.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Inertia {
    pub neg: usize,
    pub zero: usize,
    pub pos: usize,
}

/// A factorized part of a front: L = [L11; L21], D block diagonal.
struct IFront {
    /// The pivots' equations in elimination order.
    piv: Vec<usize>,
    /// Strictly lower L11, packed row-wise (row i starts at i(i−1)/2).
    l11: Vec<f64>,
    /// Diagonal of D.
    d: Vec<f64>,
    /// e[i] = D(i+1, i) if a 2×2 pivot starts at i, otherwise 0.
    e: Vec<f64>,
    /// 1 = 1×1, 2 = first of a 2×2 block, 0 = second of a 2×2 block.
    kind: Vec<u8>,
    /// Equations for the rows of L21.
    rows: Vec<usize>,
    /// L21, rows.len() × piv.len(), row-wise.
    l21: Vec<f64>,
}

/// Indefinite factor P·A·Pᵀ = L·D·Lᵀ.
pub struct IndefFactor {
    fronts: Vec<IFront>,
    pub inertia: Inertia,
    /// Number of pivots delayed at least one step and number of 2×2 blocks.
    pub delayed: usize,
    pub two_by_two: usize,
}

/// Settings.
#[derive(Debug, Clone, Copy)]
pub struct IndefOptions {
    /// Threshold u ∈ (0, 0.5]; larger is more stable but gives more delayed pivots.
    pub threshold: f64,
}

impl Default for IndefOptions {
    fn default() -> Self {
        IndefOptions { threshold: THRESHOLD }
    }
}

struct Ctx<'a> {
    sym: &'a Symbolic,
    a: &'a Lower,
    scale: Vec<f64>,
    thr: f64,
}

struct NodeOut {
    upd: Vec<f64>,
    idx: Vec<usize>,
    fronts: Vec<IFront>,
    delayed: usize,
}

/// Largest entry in each column (both triangles) of the lower triangle `a`.
pub(crate) fn column_scale(a: &Lower, n: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n];
    for c in 0..n {
        for p in a.colptr[c]..a.colptr[c + 1] {
            let (r, v) = (a.rowidx[p], a.val[p].abs());
            s[c] = s[c].max(v);
            s[r] = s[r].max(v);
        }
    }
    s
}

/// Numerical indefinite factorization. `Err(equation)` if the matrix is singular
/// (a pivot that is zero relative to the column size remains at the root).
pub fn factor(sym: &Symbolic, a: &Lower) -> Result<IndefFactor, usize> {
    factor_with(sym, a, IndefOptions::default())
}

pub fn factor_with(sym: &Symbolic, a: &Lower, opt: IndefOptions) -> Result<IndefFactor, usize> {
    let ctx = Ctx { sym, a, scale: column_scale(a, sym.n), thr: opt.threshold.clamp(1e-8, 0.5) };
    let mut fronts = Vec::new();
    let mut delayed = 0;
    for &r in &sym.roots {
        let o = factor_node(&ctx, r, par::threads(), true)?;
        fronts.extend(o.fronts);
        delayed += o.delayed;
    }
    let mut inertia = Inertia::default();
    let mut two = 0;
    for fr in &fronts {
        let mut i = 0;
        while i < fr.d.len() {
            if fr.kind[i] == 2 {
                two += 1;
                let (a, b, c) = (fr.d[i], fr.e[i], fr.d[i + 1]);
                let det = a * c - b * b;
                if det < 0.0 {
                    inertia.neg += 1;
                    inertia.pos += 1;
                } else if a + c < 0.0 {
                    inertia.neg += 2;
                } else {
                    inertia.pos += 2;
                }
                i += 2;
            } else {
                if fr.d[i] < 0.0 {
                    inertia.neg += 1;
                } else if fr.d[i] > 0.0 {
                    inertia.pos += 1;
                } else {
                    inertia.zero += 1;
                }
                i += 1;
            }
        }
    }
    Ok(IndefFactor { fronts, inertia, delayed, two_by_two: two })
}

fn factor_node(ctx: &Ctx, t: usize, threads: usize, root: bool) -> Result<NodeOut, usize> {
    let sym = ctx.sym;
    let node = &sym.tree[t];
    let ch = &node.children;
    let child_res: Vec<Result<NodeOut, usize>> = if ch.len() == 2 && threads > 1 && sym.tree[ch[0]].work.min(sym.tree[ch[1]].work) > 2e7 {
        let half = threads / 2;
        std::thread::scope(|s| {
            let h = s.spawn(|| factor_node(ctx, ch[0], half, false));
            let r1 = factor_node(ctx, ch[1], threads - half, false);
            vec![h.join().unwrap(), r1]
        })
    } else {
        ch.iter().map(|&c| factor_node(ctx, c, threads, false)).collect()
    };

    let (start, k, u) = (node.start, node.k, node.rows.len());
    let mut fs: Vec<usize> = (start..start + k).collect();
    let mut out = Vec::new();
    let mut parts = Vec::with_capacity(child_res.len());
    let mut delayed = 0;
    for res in child_res {
        let o = res?;
        out.extend(o.fronts);
        delayed += o.delayed;
        // delayed variables from the child come before this front's equations
        fs.extend(o.idx.iter().copied().filter(|&g| g < start));
        parts.push((o.upd, o.idx));
    }
    let kf = fs.len();
    delayed += kf - k;
    let m = kf + u;
    let mut dmap: Vec<(usize, usize)> = fs[k..].iter().enumerate().map(|(i, &g)| (g, k + i)).collect();
    dmap.sort_unstable();
    let local = |g: usize| -> usize {
        if g >= start && g < start + k {
            g - start
        } else if g < start {
            dmap[dmap.binary_search_by_key(&g, |x| x.0).expect("internal: delayed variable missing")].1
        } else {
            kf + node.rows.binary_search(&g).expect("internal: row missing in the front")
        }
    };

    // the front matrix: lower triangle, and the whole fully summed block
    let mut f = vec![0.0; m * m];
    let a = ctx.a;
    for c in 0..k {
        let col = start + c;
        for p in a.colptr[col]..a.colptr[col + 1] {
            f[local(a.rowidx[p]) * m + c] += a.val[p];
        }
    }
    for (upd, idx) in &parts {
        let s = idx.len();
        let map: Vec<usize> = idx.iter().map(|&g| local(g)).collect();
        for x in 0..s {
            for y in 0..=x {
                let (i, j) = (map[x], map[y]);
                let (i, j) = if i >= j { (i, j) } else { (j, i) };
                f[i * m + j] += upd[x * s + y];
            }
        }
    }
    drop(parts);
    for i in 0..kf {
        for j in 0..i {
            f[j * m + i] = f[i * m + j];
        }
    }

    let mut d = Vec::new();
    let mut e = Vec::new();
    let mut kind = Vec::new();
    let npiv = eliminate(&mut f, m, kf, &mut fs, ctx, root, threads, &mut d, &mut e, &mut kind)?;

    // pack the factor
    let mut l11 = Vec::with_capacity(npiv * npiv.saturating_sub(1) / 2);
    for i in 0..npiv {
        let start_len = l11.len();
        l11.extend_from_slice(&f[i * m..i * m + i]);
        if kind[i] == 0 {
            // the coupling in a 2×2 block belongs to D
            l11[start_len + i - 1] = 0.0;
        }
    }
    let rows: Vec<usize> = fs[npiv..kf].iter().copied().chain(node.rows.iter().copied()).collect();
    let nr = rows.len();
    let mut l21 = vec![0.0; nr * npiv];
    let mut upd = vec![0.0; nr * nr];
    for r in 0..nr {
        let fr = &f[(npiv + r) * m..(npiv + r + 1) * m];
        l21[r * npiv..(r + 1) * npiv].copy_from_slice(&fr[..npiv]);
        upd[r * nr..r * nr + r + 1].copy_from_slice(&fr[npiv..npiv + r + 1]);
    }
    out.push(IFront { piv: fs[..npiv].to_vec(), l11, d, e, kind, rows: rows.clone(), l21 });
    Ok(NodeOut { upd, idx: rows, fronts: out, delayed })
}

enum Pivot {
    One(usize),
    Two(usize, usize),
}

/// Swaps p and q (< kf) symmetrically.
fn swap(f: &mut [f64], m: usize, kf: usize, fs: &mut [usize], p: usize, q: usize) {
    if p == q {
        return;
    }
    for c in 0..kf {
        f.swap(p * m + c, q * m + c);
    }
    for i in 0..m {
        f.swap(i * m + p, i * m + q);
    }
    fs.swap(p, q);
}

/// max |A(i, j)| for i ∈ [p, m), i ∉ {j, x}.
#[inline]
fn colmax(f: &[f64], m: usize, p: usize, j: usize, x: usize) -> f64 {
    let mut g = 0.0f64;
    for i in p..m {
        if i != j && i != x {
            g = g.max(f[i * m + j].abs());
        }
    }
    g
}

/// Threshold pivot among the candidates [p, end).
fn choose(f: &[f64], m: usize, p: usize, end: usize, fs: &[usize], ctx: &Ctx) -> Option<Pivot> {
    let thr = ctx.thr;
    for j in p..end {
        let ajj = f[j * m + j];
        let sj = ctx.scale[fs[j]];
        let gj = colmax(f, m, p, j, usize::MAX);
        if ajj.abs() > ZERO_TOL * sj && ajj.abs() >= thr * gj {
            return Some(Pivot::One(j));
        }
        // 2×2 with the largest coupling among the candidates
        let mut r = usize::MAX;
        let mut best = 0.0;
        for i in p..end {
            if i != j && f[i * m + j].abs() > best {
                best = f[i * m + j].abs();
                r = i;
            }
        }
        if r == usize::MAX {
            continue;
        }
        let (a, b, c) = (ajj, f[r * m + j], f[r * m + r]);
        let det = a * c - b * b;
        let sr = ctx.scale[fs[r]];
        if !(det.abs() > ZERO_TOL * sj * sr) {
            continue;
        }
        let gj2 = colmax(f, m, p, j, r);
        let gr2 = colmax(f, m, p, r, j);
        let ad = det.abs();
        if (c.abs() * gj2 + b.abs() * gr2) / ad <= 1.0 / thr && (b.abs() * gj2 + a.abs() * gr2) / ad <= 1.0 / thr {
            return Some(Pivot::Two(j, r));
        }
    }
    None
}

/// Bunch–Kaufman on the rest [p, kf) when there are no later rows (the root).
fn choose_forced(f: &[f64], m: usize, p: usize, kf: usize, fs: &[usize], ctx: &Ctx) -> Result<Pivot, usize> {
    // largest entry in the remaining square
    let (mut bi, mut bj, mut big) = (p, p, -1.0f64);
    for j in p..kf {
        for i in j..kf {
            let v = f[i * m + j].abs();
            if v > big {
                big = v;
                bi = i;
                bj = j;
            }
        }
    }
    let one = |j: usize| -> Result<Pivot, usize> {
        if f[j * m + j].abs() > ZERO_TOL * ctx.scale[fs[j]] {
            Ok(Pivot::One(j))
        } else {
            Err(fs[j])
        }
    };
    if !(big > ZERO_TOL * ctx.scale[fs[bj]].max(ctx.scale[fs[bi]])) {
        return Err(fs[bj]);
    }
    if bi == bj {
        return one(bj);
    }
    let j = bj;
    let r = bi;
    let lam = big;
    let ajj = f[j * m + j].abs();
    if ajj >= ALPHA * lam {
        return one(j);
    }
    let sig = (p..kf).filter(|&i| i != r).map(|i| if i >= r { f[i * m + r] } else { f[r * m + i] }.abs()).fold(0.0, f64::max);
    if ajj * sig >= ALPHA * lam * lam {
        return one(j);
    }
    if f[r * m + r].abs() >= ALPHA * sig {
        return one(r);
    }
    let (a, b, c) = (f[j * m + j], f[r * m + j], f[r * m + r]);
    let det = a * c - b * b;
    if !(det.abs() > ZERO_TOL * ctx.scale[fs[j]] * ctx.scale[fs[r]]) {
        return Err(fs[j]);
    }
    Ok(Pivot::Two(j, r))
}

/// Eliminates a 1×1 pivot at p and updates the columns (p, bend) for all rows > p.
fn elim1(f: &mut [f64], m: usize, p: usize, bend: usize) -> f64 {
    let dp = f[p * m + p];
    let w: Vec<f64> = (p + 1..bend).map(|c| f[c * m + p]).collect();
    for i in p + 1..m {
        let li = f[i * m + p] / dp;
        if li != 0.0 {
            let row = &mut f[i * m + p + 1..i * m + bend];
            for (x, &wc) in row.iter_mut().zip(&w) {
                *x -= li * wc;
            }
        }
        f[i * m + p] = li;
    }
    dp
}

/// Eliminates a 2×2 pivot at (p, p+1).
fn elim2(f: &mut [f64], m: usize, p: usize, bend: usize) -> (f64, f64, f64) {
    let (a, b, c) = (f[p * m + p], f[(p + 1) * m + p], f[(p + 1) * m + p + 1]);
    let det = a * c - b * b;
    let w0: Vec<f64> = (p + 2..bend).map(|q| f[q * m + p]).collect();
    let w1: Vec<f64> = (p + 2..bend).map(|q| f[q * m + p + 1]).collect();
    for i in p + 2..m {
        let (x0, x1) = (f[i * m + p], f[i * m + p + 1]);
        let l0 = (x0 * c - x1 * b) / det;
        let l1 = (x1 * a - x0 * b) / det;
        if l0 != 0.0 || l1 != 0.0 {
            let row = &mut f[i * m + p + 2..i * m + bend];
            for ((x, &u0), &u1) in row.iter_mut().zip(&w0).zip(&w1) {
                *x -= l0 * u0 + l1 * u1;
            }
        }
        f[i * m + p] = l0;
        f[i * m + p + 1] = l1;
    }
    (a, b, c)
}

/// Blocked elimination of the fully summed variables [0, kf). Returns
/// the number of eliminated pivots; the rest are delayed (except at the root).
#[allow(clippy::too_many_arguments)]
fn eliminate(
    f: &mut [f64],
    m: usize,
    kf: usize,
    fs: &mut [usize],
    ctx: &Ctx,
    root: bool,
    threads: usize,
    d: &mut Vec<f64>,
    e: &mut Vec<f64>,
    kind: &mut Vec<u8>,
) -> Result<usize, usize> {
    let mut p = 0;
    let mut width = KB;
    while p < kf {
        let p0 = p;
        let b1 = (p + width).min(kf);
        while p < b1 {
            match choose(f, m, p, b1, fs, ctx) {
                Some(Pivot::One(j)) => {
                    swap(f, m, kf, fs, p, j);
                    d.push(elim1(f, m, p, b1));
                    e.push(0.0);
                    kind.push(1);
                    p += 1;
                }
                Some(Pivot::Two(j, r)) => {
                    let r = if r == p { j } else { r };
                    swap(f, m, kf, fs, p, j);
                    swap(f, m, kf, fs, p + 1, r);
                    let (a, b, c) = elim2(f, m, p, b1);
                    d.extend([a, c]);
                    e.extend([b, 0.0]);
                    kind.extend([2, 0]);
                    p += 2;
                }
                None => break,
            }
        }
        let npb = p - p0;
        if npb > 0 && b1 < m {
            trailing_update(f, m, p0, npb, b1, &d[p0..p], &e[p0..p], &kind[p0..p], threads);
            // mirror the remaining fully summed block (lower → upper)
            for i in b1..kf {
                for j in p..i {
                    f[j * m + i] = f[i * m + j];
                }
            }
        }
        if npb == 0 {
            if b1 == kf {
                break;
            }
            width *= 2;
        } else {
            width = KB;
        }
    }
    if p < kf && root {
        // root: ordinary Bunch–Kaufman on the rest (no later rows)
        while p < kf {
            match choose_forced(f, m, p, kf, fs, ctx)? {
                Pivot::One(j) => {
                    swap(f, m, kf, fs, p, j);
                    d.push(elim1(f, m, p, kf));
                    e.push(0.0);
                    kind.push(1);
                    p += 1;
                }
                Pivot::Two(j, r) => {
                    let r = if r == p { j } else { r };
                    swap(f, m, kf, fs, p, j);
                    swap(f, m, kf, fs, p + 1, r);
                    let (a, b, c) = elim2(f, m, p, kf);
                    d.extend([a, c]);
                    e.extend([b, 0.0]);
                    kind.extend([2, 0]);
                    p += 2;
                }
            }
        }
        // the rest of the root has no later rows; no update needed
    }
    Ok(p)
}

/// F[x][y] −= (L·D)[x]·L[y]ᵀ for b1 ≤ y ≤ x < m, with the pivots [p0, p0+npb).
#[allow(clippy::too_many_arguments)]
fn trailing_update(f: &mut [f64], m: usize, p0: usize, npb: usize, b1: usize, d: &[f64], e: &[f64], kind: &[u8], threads: usize) {
    let rows = m - b1;
    let mut w = vec![0.0; rows * npb];
    let mut lt = vec![0.0; npb * rows];
    for r in 0..rows {
        let fr = &f[(b1 + r) * m + p0..(b1 + r) * m + p0 + npb];
        let wr = &mut w[r * npb..(r + 1) * npb];
        let mut q = 0;
        while q < npb {
            if kind[q] == 2 {
                let (l0, l1) = (fr[q], fr[q + 1]);
                wr[q] = l0 * d[q] + l1 * e[q];
                wr[q + 1] = l0 * e[q] + l1 * d[q + 1];
                q += 2;
            } else {
                wr[q] = fr[q] * d[q];
                q += 1;
            }
        }
        for q in 0..npb {
            lt[q * rows + r] = fr[q];
        }
    }
    let tp = if (rows * rows * npb) as f64 / 2.0 > PAR_FLOPS { threads } else { 1 };
    let (w, lt) = (&w, &lt);
    par::blocks_mut(&mut f[b1 * m..], m, rows, tp, |r| (r + 1) as f64, |r0, chunk| {
        kern::syrk(chunk, m, r0, chunk.len() / m, b1, rows, npb, w, lt);
    });
}

impl IndefFactor {
    /// Solves A x = b in place (internal numbering).
    pub fn solve(&self, x: &mut [f64]) {
        for fr in &self.fronts {
            let k = fr.piv.len();
            let mut xp: Vec<f64> = fr.piv.iter().map(|&g| x[g]).collect();
            for i in 0..k {
                let r = i * i.saturating_sub(1) / 2;
                let s = dot(&fr.l11[r..r + i], &xp[..i]);
                xp[i] -= s;
            }
            for (r, &g) in fr.rows.iter().enumerate() {
                x[g] -= dot(&fr.l21[r * k..r * k + k], &xp);
            }
            for (i, &g) in fr.piv.iter().enumerate() {
                x[g] = xp[i];
            }
        }
        for fr in &self.fronts {
            let mut i = 0;
            while i < fr.piv.len() {
                if fr.kind[i] == 2 {
                    let (a, b, c) = (fr.d[i], fr.e[i], fr.d[i + 1]);
                    let det = a * c - b * b;
                    let (g0, g1) = (fr.piv[i], fr.piv[i + 1]);
                    let (y0, y1) = (x[g0], x[g1]);
                    x[g0] = (c * y0 - b * y1) / det;
                    x[g1] = (a * y1 - b * y0) / det;
                    i += 2;
                } else {
                    x[fr.piv[i]] /= fr.d[i];
                    i += 1;
                }
            }
        }
        for fr in self.fronts.iter().rev() {
            let k = fr.piv.len();
            let mut xp: Vec<f64> = fr.piv.iter().map(|&g| x[g]).collect();
            for (r, &g) in fr.rows.iter().enumerate() {
                let xr = x[g];
                if xr != 0.0 {
                    for (p, l) in fr.l21[r * k..r * k + k].iter().enumerate() {
                        xp[p] -= l * xr;
                    }
                }
            }
            for i in (0..k).rev() {
                let xi = xp[i];
                let r = i * i.saturating_sub(1) / 2;
                for q in 0..i {
                    xp[q] -= fr.l11[r + q] * xi;
                }
            }
            for (i, &g) in fr.piv.iter().enumerate() {
                x[g] = xp[i];
            }
        }
    }

    /// Number of stored entries in L.
    pub fn factor_size(&self) -> usize {
        self.fronts.iter().map(|f| f.l11.len() + f.l21.len()).sum()
    }
}

/// y = A x for the symmetric matrix in the lower triangle `a`.
pub fn sym_matvec(a: &Lower, x: &[f64], y: &mut [f64]) {
    y.iter_mut().for_each(|v| *v = 0.0);
    for c in 0..a.colptr.len() - 1 {
        for p in a.colptr[c]..a.colptr[c + 1] {
            let (r, v) = (a.rowidx[p], a.val[p]);
            y[r] += v * x[c];
            if r != c {
                y[c] += v * x[r];
            }
        }
    }
}
