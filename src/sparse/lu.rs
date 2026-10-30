//! Multifrontal LU for unsymmetric matrices with a symmetric pattern
//! (e.g. friction or follower loads, where the tangent becomes unsymmetric but
//! the couplings are the same as for the stiffness).
//!
//! - Same ordering and tree as LDLᵀ (`sparse::analyse`). The fronts are stored
//!   in full (both triangles).
//! - Pivoting: diagonal pivot with a threshold test on the column,
//!   |a_jj| ≥ u·max_{i≠j} |a_ij| over the whole front, u = 0.01. Variables that
//!   fail the test are delayed to the parent (symmetric permutation, so that
//!   row and column move together and the pattern stays symmetric).
//! - At the root, whatever remains is factorized with ordinary partial pivoting
//!   (row swaps), which handles all regular matrices.
//!
//! References: Duff & Reid (1984) "The multifrontal solution of unsymmetric
//! sets of linear equations", Amestoy & Puglisi (2002).

use super::{dot, kern, Lower, Symbolic, PAR_FLOPS};
use super::indef::ZERO_TOL;
use crate::par;

/// Default threshold.
pub const THRESHOLD: f64 = 0.01;
const KB: usize = 64;

/// Full matrix with a symmetric pattern: lower triangle (incl. diagonal) in
/// `lower`, the strictly upper triangle transposed in `upper_t` (same pattern,
/// A(c, r) for r > c is stored in column c, row r).
pub struct Full {
    pub lower: Lower,
    pub upper_t: Lower,
}

impl Full {
    /// Empty pattern from a lower-triangle pattern.
    pub fn from_pattern(l: Lower) -> Full {
        let u = Lower { colptr: l.colptr.clone(), rowidx: l.rowidx.clone(), val: vec![0.0; l.val.len()] };
        Full { lower: l, upper_t: u }
    }

    /// A(r, c) += v.
    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        if r >= c {
            self.lower.add(r, c, v);
        } else {
            self.upper_t.add(c, r, v);
        }
    }

    pub fn zero(&mut self) {
        self.lower.val.iter_mut().for_each(|v| *v = 0.0);
        self.upper_t.val.iter_mut().for_each(|v| *v = 0.0);
    }

    /// y = A x.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        y.iter_mut().for_each(|v| *v = 0.0);
        let (l, u) = (&self.lower, &self.upper_t);
        for c in 0..l.colptr.len() - 1 {
            for p in l.colptr[c]..l.colptr[c + 1] {
                let r = l.rowidx[p];
                y[r] += l.val[p] * x[c];
                if r != c {
                    // A(c, r) = upper_t(r, c)
                    y[c] += u.val[p] * x[r];
                }
            }
        }
    }

    fn scale(&self, n: usize) -> Vec<f64> {
        let mut s = vec![0.0f64; n];
        for (m, t) in [(&self.lower, false), (&self.upper_t, true)] {
            for c in 0..n {
                for p in m.colptr[c]..m.colptr[c + 1] {
                    let r = m.rowidx[p];
                    if t && r == c {
                        continue;
                    }
                    let v = m.val[p].abs();
                    s[c] = s[c].max(v);
                    s[r] = s[r].max(v);
                }
            }
        }
        s
    }
}

struct LuFront {
    /// Pivot variables (columns) in elimination order.
    piv: Vec<usize>,
    /// Row permutation within the block (root only; otherwise empty = identity).
    rperm: Vec<usize>,
    /// L11 (strictly lower, unit diagonal) and U11 (upper incl. diagonal), k×k row-wise.
    lu11: Vec<f64>,
    rows: Vec<usize>,
    /// L21 rows×k and U12 k×rows, row-wise.
    l21: Vec<f64>,
    u12: Vec<f64>,
}

pub struct LuFactor {
    fronts: Vec<LuFront>,
    pub delayed: usize,
}

struct Ctx<'a> {
    sym: &'a Symbolic,
    a: &'a Full,
    scale: Vec<f64>,
    thr: f64,
}

struct NodeOut {
    upd: Vec<f64>,
    idx: Vec<usize>,
    fronts: Vec<LuFront>,
    delayed: usize,
}

/// Numerical LU factorization. `Err(equation)` if the matrix is singular.
pub fn factor(sym: &Symbolic, a: &Full) -> Result<LuFactor, usize> {
    factor_with(sym, a, THRESHOLD)
}

pub fn factor_with(sym: &Symbolic, a: &Full, threshold: f64) -> Result<LuFactor, usize> {
    let ctx = Ctx { sym, a, scale: a.scale(sym.n), thr: threshold.clamp(1e-8, 1.0) };
    let mut fronts = Vec::new();
    let mut delayed = 0;
    for &r in &sym.roots {
        let o = factor_node(&ctx, r, par::threads(), true)?;
        fronts.extend(o.fronts);
        delayed += o.delayed;
    }
    Ok(LuFactor { fronts, delayed })
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

    let mut f = vec![0.0; m * m];
    let (lo, ut) = (&ctx.a.lower, &ctx.a.upper_t);
    for c in 0..k {
        let col = start + c;
        for p in lo.colptr[col]..lo.colptr[col + 1] {
            let lr = local(lo.rowidx[p]);
            f[lr * m + c] += lo.val[p];
            if lr != c {
                f[c * m + lr] += ut.val[p];
            }
        }
    }
    for (upd, idx) in &parts {
        let s = idx.len();
        let map: Vec<usize> = idx.iter().map(|&g| local(g)).collect();
        for x in 0..s {
            let fx = map[x] * m;
            for y in 0..s {
                f[fx + map[y]] += upd[x * s + y];
            }
        }
    }
    drop(parts);

    let npiv = eliminate(&mut f, m, kf, &mut fs, ctx, threads);
    let rows: Vec<usize> = fs[npiv..kf].iter().copied().chain(node.rows.iter().copied()).collect();
    let nr = rows.len();
    let piece = |f: &[f64], piv: Vec<usize>| -> LuFront {
        let k = piv.len();
        let mut lu11 = vec![0.0; k * k];
        for i in 0..k {
            lu11[i * k..(i + 1) * k].copy_from_slice(&f[i * m..i * m + k]);
        }
        let mut l21 = vec![0.0; nr * k];
        let mut u12 = vec![0.0; k * nr];
        for r in 0..nr {
            l21[r * k..(r + 1) * k].copy_from_slice(&f[(k + r) * m..(k + r) * m + k]);
        }
        for i in 0..k {
            u12[i * nr..(i + 1) * nr].copy_from_slice(&f[i * m + k..i * m + k + nr]);
        }
        LuFront { piv, rperm: Vec::new(), lu11, rows: rows.clone(), l21, u12 }
    };
    out.push(piece(&f, fs[..npiv].to_vec()));
    if root && npiv < kf {
        // root: partial pivoting with row swaps on the rest (no later rows)
        let r = kf - npiv;
        let mut s = vec![0.0; r * r];
        for i in 0..r {
            s[i * r..(i + 1) * r].copy_from_slice(&f[(npiv + i) * m + npiv..(npiv + i) * m + npiv + r]);
        }
        let vars = fs[npiv..kf].to_vec();
        let rperm = dense_lu(&mut s, r, &vars, &ctx.scale)?;
        out.push(LuFront { piv: vars, rperm, lu11: s, rows: Vec::new(), l21: Vec::new(), u12: Vec::new() });
        return Ok(NodeOut { upd: Vec::new(), idx: Vec::new(), fronts: out, delayed });
    }
    let mut upd = vec![0.0; nr * nr];
    for x in 0..nr {
        upd[x * nr..(x + 1) * nr].copy_from_slice(&f[(npiv + x) * m + npiv..(npiv + x) * m + npiv + nr]);
    }
    Ok(NodeOut { upd, idx: rows, fronts: out, delayed })
}

/// Dense LU with partial pivoting in place. Returns the row permutation
/// (row i of the factor = original row perm[i]).
fn dense_lu(s: &mut [f64], r: usize, vars: &[usize], scale: &[f64]) -> Result<Vec<usize>, usize> {
    let mut perm: Vec<usize> = (0..r).collect();
    for p in 0..r {
        let (mut bi, mut big) = (p, 0.0f64);
        for i in p..r {
            if s[i * r + p].abs() > big {
                big = s[i * r + p].abs();
                bi = i;
            }
        }
        if !(big > ZERO_TOL * scale[vars[p]]) {
            return Err(vars[p]);
        }
        if bi != p {
            for c in 0..r {
                s.swap(p * r + c, bi * r + c);
            }
            perm.swap(p, bi);
        }
        let dp = s[p * r + p];
        for i in p + 1..r {
            let l = s[i * r + p] / dp;
            s[i * r + p] = l;
            if l != 0.0 {
                for c in p + 1..r {
                    s[i * r + c] -= l * s[p * r + c];
                }
            }
        }
    }
    Ok(perm)
}

fn swap(f: &mut [f64], m: usize, fs: &mut [usize], p: usize, q: usize) {
    if p == q {
        return;
    }
    for c in 0..m {
        f.swap(p * m + c, q * m + c);
    }
    for i in 0..m {
        f.swap(i * m + p, i * m + q);
    }
    fs.swap(p, q);
}

fn eliminate(f: &mut [f64], m: usize, kf: usize, fs: &mut [usize], ctx: &Ctx, threads: usize) -> usize {
    let mut p = 0;
    let mut width = KB;
    while p < kf {
        let p0 = p;
        let b1 = (p + width).min(kf);
        while p < b1 {
            // first candidate that passes the threshold test
            let mut pick = None;
            for j in p..b1 {
                let ajj = f[j * m + j].abs();
                if !(ajj > ZERO_TOL * ctx.scale[fs[j]]) {
                    continue;
                }
                let mut g = 0.0f64;
                for i in p..m {
                    if i != j {
                        g = g.max(f[i * m + j].abs());
                    }
                }
                if ajj >= ctx.thr * g {
                    pick = Some(j);
                    break;
                }
            }
            let Some(j) = pick else { break };
            swap(f, m, fs, p, j);
            let dp = f[p * m + p];
            let urow: Vec<f64> = f[p * m + p + 1..p * m + m].to_vec();
            for i in p + 1..m {
                let li = f[i * m + p] / dp;
                f[i * m + p] = li;
                if li == 0.0 {
                    continue;
                }
                // the block's columns for all rows, the whole row for rows in the block
                let end = if i < b1 { m } else { b1 };
                let row = &mut f[i * m + p + 1..i * m + end];
                for (x, &uc) in row.iter_mut().zip(&urow) {
                    *x -= li * uc;
                }
            }
            p += 1;
        }
        let npb = p - p0;
        if npb > 0 && b1 < m {
            // F[b1.., b1..] −= L[b1.., p0..p]·U[p0..p, b1..]
            let rows = m - b1;
            let mut a = vec![0.0; rows * npb];
            for r in 0..rows {
                a[r * npb..(r + 1) * npb].copy_from_slice(&f[(b1 + r) * m + p0..(b1 + r) * m + p]);
            }
            let mut b = vec![0.0; npb * rows];
            for q in 0..npb {
                b[q * rows..(q + 1) * rows].copy_from_slice(&f[(p0 + q) * m + b1..(p0 + q) * m + m]);
            }
            let tp = if (rows * rows * npb) as f64 > PAR_FLOPS { threads } else { 1 };
            let (a, b) = (&a, &b);
            par::blocks_mut(&mut f[b1 * m..], m, rows, tp, |_| 1.0, |r0, chunk| {
                let nr = chunk.len() / m;
                kern::gemm(&mut chunk[b1..], m, nr, rows, &a[r0 * npb..], npb, b, rows, npb);
            });
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
    p
}

impl LuFactor {
    /// Solves A x = b in place (internal numbering).
    pub fn solve(&self, x: &mut [f64]) {
        for fr in &self.fronts {
            let k = fr.piv.len();
            let mut xp: Vec<f64> = if fr.rperm.is_empty() {
                fr.piv.iter().map(|&g| x[g]).collect()
            } else {
                fr.rperm.iter().map(|&i| x[fr.piv[i]]).collect()
            };
            for i in 0..k {
                let s = dot(&fr.lu11[i * k..i * k + i], &xp[..i]);
                xp[i] -= s;
            }
            for (r, &g) in fr.rows.iter().enumerate() {
                x[g] -= dot(&fr.l21[r * k..r * k + k], &xp);
            }
            for (i, &g) in fr.piv.iter().enumerate() {
                x[g] = xp[i];
            }
        }
        for fr in self.fronts.iter().rev() {
            let k = fr.piv.len();
            let nr = fr.rows.len();
            let xr: Vec<f64> = fr.rows.iter().map(|&g| x[g]).collect();
            let mut xp: Vec<f64> = fr.piv.iter().map(|&g| x[g]).collect();
            for i in 0..k {
                xp[i] -= dot(&fr.u12[i * nr..(i + 1) * nr], &xr);
            }
            for i in (0..k).rev() {
                let s = dot(&fr.lu11[i * k + i + 1..(i + 1) * k], &xp[i + 1..k]);
                xp[i] = (xp[i] - s) / fr.lu11[i * k + i];
            }
            for (i, &g) in fr.piv.iter().enumerate() {
                x[g] = xp[i];
            }
        }
    }

    pub fn factor_size(&self) -> usize {
        self.fronts.iter().map(|f| f.lu11.len() + f.l21.len() + f.u12.len()).sum()
    }
}
