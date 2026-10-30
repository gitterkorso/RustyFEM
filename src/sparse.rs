//! Sparse direct solver for symmetric positive definite systems.
//!
//! - Ordering: geometric nested dissection at node level. The nodes are split by the
//!   axis-perpendicular plane near the median that gives the smallest separator; the separator
//!   is numbered last and the halves recursively.
//!   All degrees of freedom of a node are kept together.
//! - Factorization: multifrontal LDLᵀ. Each separator is a dense block
//!   (supernode) whose front receives updates from its children. Independent
//!   subtrees and large fronts are computed in parallel.

use crate::par;

/// Maximum number of nodes in a leaf of the dissection tree (default).
pub const LEAF: usize = 48;
/// Below this number of flops a front is computed on one thread.
const PAR_FLOPS: f64 = 4e6;

struct TNode {
    nodes: Vec<usize>,
    children: Vec<usize>,
    /// First degree of freedom and number of pivots in the block.
    start: usize,
    k: usize,
    /// The front's other rows (later degrees of freedom), sorted.
    rows: Vec<usize>,
    /// Estimated number of flops in the whole subtree.
    work: f64,
}

pub struct Symbolic {
    tree: Vec<TNode>,
    roots: Vec<usize>,
    /// First degree of freedom for each node.
    pub first_dof: Vec<usize>,
    pub n: usize,
}

/// Lower triangle (including the diagonal) in column format.
pub struct Lower {
    pub colptr: Vec<usize>,
    pub rowidx: Vec<usize>,
    pub val: Vec<f64>,
}

impl Lower {
    #[inline]
    pub fn add(&mut self, r: usize, c: usize, v: f64) {
        let (lo, hi) = (self.colptr[c], self.colptr[c + 1]);
        let p = lo + self.rowidx[lo..hi].binary_search(&r).expect("internal: entry missing in the pattern");
        self.val[p] += v;
    }

    fn diag(&self, c: usize) -> f64 {
        self.val[self.colptr[c]]
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut s = [0.0f64; 8];
    let ca = a.chunks_exact(8);
    let cb = b.chunks_exact(8);
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            s[l] += x[l] * y[l];
        }
    }
    let mut t = (s[0] + s[4]) + (s[1] + s[5]) + ((s[2] + s[6]) + (s[3] + s[7]));
    for (x, y) in ra.iter().zip(rb) {
        t += x * y;
    }
    t
}

fn dissect(mut set: Vec<usize>, coords: &[[f64; 3]], adj: &[Vec<usize>], side: &mut [u8], tree: &mut Vec<TNode>, leaf_max: usize) -> Option<usize> {
    if set.is_empty() {
        return None;
    }
    let leaf = |nodes, children| TNode { nodes, children, start: 0, k: 0, rows: Vec::new(), work: 0.0 };
    if set.len() <= leaf_max {
        tree.push(leaf(set, Vec::new()));
        return Some(tree.len() - 1);
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &v in &set {
        for a in 0..3 {
            lo[a] = lo[a].min(coords[v][a]);
            hi[a] = hi[a].max(coords[v][a]);
        }
    }
    // Try planes perpendicular to all three axes near the median and choose the
    // smallest separator relative to the balance. Always splitting along the
    // longest axis gives whole cross-sections as separators in slender bodies (e.g.
    // an I-beam of solid elements), where a cut through the web is much smaller.
    // The longest axis is tried first and wins on ties.
    let mut axes = [0, 1, 2];
    axes.sort_by(|&a, &b| (hi[b] - lo[b]).partial_cmp(&(hi[a] - lo[a])).unwrap());
    let n = set.len();
    let mut best: Option<(f64, usize, f64, bool)> = None; // (score, axis, cut value, separator in left half)
    for &ax in &axes {
        if !(hi[ax] > lo[ax]) {
            continue;
        }
        set.sort_unstable_by(|&a, &b| coords[a][ax].partial_cmp(&coords[b][ax]).unwrap());
        for q in [n / 2, 2 * n / 5, 3 * n / 5] {
            // split on coordinate value so that planes in a structured mesh stay together
            let t = coords[set[q]][ax];
            let nl = set.partition_point(|&v| coords[v][ax] < t);
            if nl == 0 || nl == n {
                continue;
            }
            for (i, &v) in set.iter().enumerate() {
                side[v] = (i >= nl) as u8;
            }
            let sep_l = set[..nl].iter().filter(|&&v| adj[v].iter().any(|&w| side[w] == 1)).count();
            let sep_r = set[nl..].iter().filter(|&&v| adj[v].iter().any(|&w| side[w] == 0)).count();
            let (s, in_left) = if sep_r < sep_l { (sep_r, false) } else { (sep_l, true) };
            let score = s as f64 / (nl.min(n - nl) as f64 / n as f64);
            if best.is_none_or(|b| score < b.0) {
                best = Some((score, ax, t, in_left));
            }
        }
    }
    let (left, right): (Vec<usize>, Vec<usize>) = match best {
        Some((_, ax, t, _)) => set.iter().partition(|&&v| coords[v][ax] < t),
        None => {
            // all nodes at the same point: split on index
            let right = set.split_off(n / 2);
            (set, right)
        }
    };
    let in_left = best.is_none_or(|b| b.3);
    for &v in &left {
        side[v] = 0;
    }
    for &v in &right {
        side[v] = 1;
    }
    let (sep, a, b) = if in_left {
        let (sep, rest): (Vec<usize>, Vec<usize>) = left.iter().partition(|&&v| adj[v].iter().any(|&w| side[w] == 1));
        (sep, rest, right.clone())
    } else {
        let (sep, rest): (Vec<usize>, Vec<usize>) = right.iter().partition(|&&v| adj[v].iter().any(|&w| side[w] == 0));
        (sep, left.clone(), rest)
    };
    for &v in left.iter().chain(&right) {
        side[v] = 2;
    }
    drop((left, right));
    let children: Vec<usize> = [dissect(a, coords, adj, side, tree, leaf_max), dissect(b, coords, adj, side, tree, leaf_max)].into_iter().flatten().collect();
    tree.push(leaf(sep, children));
    Some(tree.len() - 1)
}

/// Symbolic analysis: ordering, tree structure and front pattern.
/// `adj` are the node neighbours and `ndof` the number of free degrees of freedom per node.
pub fn analyse(coords: &[[f64; 3]], adj: &[Vec<usize>], ndof: &[usize]) -> Symbolic {
    analyse_with_leaf(coords, adj, ndof, LEAF)
}

/// Like [`analyse`], with at most `leaf_max` nodes per leaf. Smaller leaves give less
/// fill-in for sparse meshes with many degrees of freedom per node (e.g. shells).
pub fn analyse_with_leaf(coords: &[[f64; 3]], adj: &[Vec<usize>], ndof: &[usize], leaf_max: usize) -> Symbolic {
    let leaf_max = leaf_max.max(1);
    let nn = coords.len();
    let mut tree = Vec::new();
    let mut side = vec![2u8; nn];
    // one dissection per connected component is unnecessary; the whole set suffices
    let roots: Vec<usize> = dissect((0..nn).collect(), coords, adj, &mut side, &mut tree, leaf_max).into_iter().collect();

    // postorder: children before parent
    let mut post = Vec::with_capacity(tree.len());
    let mut stack: Vec<(usize, bool)> = roots.iter().map(|&r| (r, false)).collect();
    while let Some((t, done)) = stack.pop() {
        if done {
            post.push(t);
        } else {
            stack.push((t, true));
            for &c in tree[t].children.iter().rev() {
                stack.push((c, false));
            }
        }
    }

    let mut first_dof = vec![0usize; nn];
    let mut next = 0;
    for &t in &post {
        tree[t].start = next;
        for i in 0..tree[t].nodes.len() {
            let v = tree[t].nodes[i];
            first_dof[v] = next;
            next += ndof[v];
        }
        tree[t].k = next - tree[t].start;
    }

    // front pattern (node level): neighbours of the pivots and the children's rows that
    // are eliminated later
    let mut unodes: Vec<Vec<usize>> = vec![Vec::new(); tree.len()];
    let mut stamp = vec![usize::MAX; nn];
    for &t in &post {
        let end = tree[t].start + tree[t].k;
        let mut list = Vec::new();
        let mut take = |w: usize, list: &mut Vec<usize>| {
            if first_dof[w] >= end && stamp[w] != t {
                stamp[w] = t;
                list.push(w);
            }
        };
        for &v in &tree[t].nodes {
            for &w in &adj[v] {
                take(w, &mut list);
            }
        }
        for &c in &tree[t].children {
            for &w in &unodes[c] {
                take(w, &mut list);
            }
        }
        list.sort_unstable_by_key(|&w| first_dof[w]);
        tree[t].rows = list.iter().flat_map(|&w| first_dof[w]..first_dof[w] + ndof[w]).collect();
        let (k, u) = (tree[t].k as f64, tree[t].rows.len() as f64);
        let own = k * k * k / 3.0 + u * k * k + u * u * k;
        tree[t].work = own + tree[t].children.iter().map(|&c| tree[c].work).sum::<f64>();
        unodes[t] = list;
    }
    Symbolic { tree, roots, first_dof, n: next }
}

impl Symbolic {
    /// Estimated number of floating-point operations in the factorization.
    pub fn work(&self) -> f64 {
        self.roots.iter().map(|&r| self.tree[r].work).sum()
    }

    /// Number of stored entries in the factor L.
    pub fn factor_size(&self) -> usize {
        self.tree.iter().map(|t| t.k * t.k.saturating_sub(1) / 2 + t.k * t.rows.len()).sum()
    }

    /// (first degree of freedom, pivots k, other rows u, number of children) for each front.
    pub fn fronts(&self) -> Vec<(usize, usize, usize, usize)> {
        self.tree.iter().map(|t| (t.start, t.k, t.rows.len(), t.children.len())).collect()
    }

    /// Largest front (k pivots, u rows).
    pub fn largest_front(&self) -> (usize, usize) {
        self.tree.iter().map(|t| (t.k, t.rows.len())).max_by_key(|&(k, u)| k + u).unwrap_or((0, 0))
    }

    /// Empty lower-triangle pattern, given the node neighbours.
    pub fn pattern(&self, adj: &[Vec<usize>], ndof: &[usize]) -> Lower {
        let mut colptr = vec![0usize; self.n + 1];
        let mut rowidx = Vec::new();
        let mut order: Vec<usize> = (0..adj.len()).collect();
        order.sort_unstable_by_key(|&v| self.first_dof[v]);
        for &v in &order {
            let mut nb: Vec<usize> = adj[v].iter().copied().chain(std::iter::once(v)).collect();
            nb.sort_unstable_by_key(|&w| self.first_dof[w]);
            let f = self.first_dof[v];
            for c in f..f + ndof[v] {
                for &w in &nb {
                    let fw = self.first_dof[w];
                    rowidx.extend((fw..fw + ndof[w]).filter(|&r| r >= c));
                }
                colptr[c + 1] = rowidx.len();
            }
        }
        let val = vec![0.0; rowidx.len()];
        Lower { colptr, rowidx, val }
    }
}

/// Dense kernels. The AVX2/FMA variant is selected at run time if the processor supports it.
mod kern {
    #[inline(always)]
    fn fma<const FMA: bool>(a: f64, b: f64, c: f64) -> f64 {
        if FMA {
            a.mul_add(b, c)
        } else {
            c + a * b
        }
    }

    /// Forward substitution w ← w·L⁻ᵀ for one row, with transposed unit triangle lt (n×n).
    #[inline(always)]
    fn forward_impl<const FMA: bool>(w: &mut [f64], lt: &[f64], n: usize) {
        for p in 0..n {
            let wp = -w[p];
            if wp == 0.0 {
                continue;
            }
            let rest = &mut w[p + 1..n];
            let col = &lt[p * n + p + 1..p * n + n];
            for (x, &l) in rest.iter_mut().zip(col) {
                *x = fma::<FMA>(wp, l, *x);
            }
        }
    }

    /// C[i][j] −= Σₚ A[i][p]·B[p][j] for i < nrows, j < ncols, p < kk.
    /// The micro-kernel keeps a 4×8 block of C in registers over the whole p loop.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn gemm_impl<const FMA: bool>(
        c: &mut [f64],
        ldc: usize,
        nrows: usize,
        ncols: usize,
        a: &[f64],
        lda: usize,
        b: &[f64],
        ldb: usize,
        kk: usize,
    ) {
        const YB: usize = 512;
        let mut j0 = 0;
        while j0 < ncols {
            let j1 = (j0 + YB).min(ncols);
            let mut i = 0;
            while i + 4 <= nrows {
                let mut j = j0;
                while j + 8 <= j1 {
                    let mut acc = [[0.0f64; 8]; 4];
                    for p in 0..kk {
                        let bp = &b[p * ldb + j..p * ldb + j + 8];
                        for r in 0..4 {
                            let ar = a[(i + r) * lda + p];
                            for l in 0..8 {
                                acc[r][l] = fma::<FMA>(ar, bp[l], acc[r][l]);
                            }
                        }
                    }
                    for r in 0..4 {
                        let cr = &mut c[(i + r) * ldc + j..(i + r) * ldc + j + 8];
                        for l in 0..8 {
                            cr[l] -= acc[r][l];
                        }
                    }
                    j += 8;
                }
                for r in 0..4 {
                    for jj in j..j1 {
                        let mut s = 0.0;
                        for p in 0..kk {
                            s = fma::<FMA>(a[(i + r) * lda + p], b[p * ldb + jj], s);
                        }
                        c[(i + r) * ldc + jj] -= s;
                    }
                }
                i += 4;
            }
            while i < nrows {
                for jj in j0..j1 {
                    let mut s = 0.0;
                    for p in 0..kk {
                        s = fma::<FMA>(a[i * lda + p], b[p * ldb + jj], s);
                    }
                    c[i * ldc + jj] -= s;
                }
                i += 1;
            }
            j0 = j1;
        }
    }

    /// Update of the lower triangle for rows r0..r0+nr (relative) in a
    /// row block: C[x][y] −= W[x]·Lᵀ[y] for y ≤ x. The columns start at `coloff`.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn syrk_impl<const FMA: bool>(chunk: &mut [f64], ldc: usize, r0: usize, nr: usize, coloff: usize, rows: usize, kk: usize, w: &[f64], lt: &[f64]) {
        const YB: usize = 512;
        let xend = r0 + nr;
        let mut yb0 = 0;
        while yb0 < xend {
            let yb1 = (yb0 + YB).min(xend);
            let rs = r0.max(yb0);
            gemm_impl::<FMA>(&mut chunk[(rs - r0) * ldc + coloff + yb0..], ldc, xend - rs, yb1 - yb0, &w[rs * kk..], kk, &lt[yb0..], rows, kk);
            yb0 = yb1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn avx() -> bool {
        static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *F.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn forward_avx(w: &mut [f64], lt: &[f64], n: usize) {
        forward_impl::<true>(w, lt, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn syrk_avx(chunk: &mut [f64], ldc: usize, r0: usize, nr: usize, coloff: usize, rows: usize, kk: usize, w: &[f64], lt: &[f64]) {
        syrk_impl::<true>(chunk, ldc, r0, nr, coloff, rows, kk, w, lt)
    }

    pub fn forward(w: &mut [f64], lt: &[f64], n: usize) {
        #[cfg(target_arch = "x86_64")]
        if avx() {
            // SAFETY: the processor supports AVX2 and FMA (checked above)
            return unsafe { forward_avx(w, lt, n) };
        }
        forward_impl::<false>(w, lt, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_avx(c: &mut [f64], ldc: usize, nrows: usize, ncols: usize, a: &[f64], lda: usize, b: &[f64], ldb: usize, kk: usize) {
        gemm_impl::<true>(c, ldc, nrows, ncols, a, lda, b, ldb, kk)
    }

    /// General update C −= A·B (C nrows×ncols with row stride ldc, A nrows×kk, B kk×ncols).
    /// Used by the unsymmetric LU factorization (`sparse::lu`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(c: &mut [f64], ldc: usize, nrows: usize, ncols: usize, a: &[f64], lda: usize, b: &[f64], ldb: usize, kk: usize) {
        #[cfg(target_arch = "x86_64")]
        if avx() {
            // SAFETY: the processor supports AVX2 and FMA (checked above)
            return unsafe { gemm_avx(c, ldc, nrows, ncols, a, lda, b, ldb, kk) };
        }
        gemm_impl::<false>(c, ldc, nrows, ncols, a, lda, b, ldb, kk)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn syrk(chunk: &mut [f64], ldc: usize, r0: usize, nr: usize, coloff: usize, rows: usize, kk: usize, w: &[f64], lt: &[f64]) {
        #[cfg(target_arch = "x86_64")]
        if avx() {
            // SAFETY: the processor supports AVX2 and FMA (checked above)
            return unsafe { syrk_avx(chunk, ldc, r0, nr, coloff, rows, kk, w, lt) };
        }
        syrk_impl::<false>(chunk, ldc, r0, nr, coloff, rows, kk, w, lt)
    }
}

struct Front {
    start: usize,
    k: usize,
    rows: Vec<usize>,
    l11: Vec<f64>,
    d: Vec<f64>,
    l21: Vec<f64>,
}

pub struct Factor {
    fronts: Vec<Front>,
}

type NodeResult = Result<(Vec<f64>, Vec<Front>), usize>;

/// Size of the pivot block in the blocked factorization.
const KB: usize = 96;

fn factor_node(sym: &Symbolic, a: &Lower, t: usize, threads: usize) -> NodeResult {
    let node = &sym.tree[t];
    let ch = &node.children;
    let child_res: Vec<NodeResult> = if ch.len() == 2 && threads > 1 && sym.tree[ch[0]].work.min(sym.tree[ch[1]].work) > 2e7 {
        let half = threads / 2;
        std::thread::scope(|s| {
            let h = s.spawn(|| factor_node(sym, a, ch[0], half));
            let r1 = factor_node(sym, a, ch[1], threads - half);
            vec![h.join().unwrap(), r1]
        })
    } else {
        ch.iter().map(|&c| factor_node(sym, a, c, threads)).collect()
    };

    let (start, k, u) = (node.start, node.k, node.rows.len());
    let m = k + u;
    let local = |dof: usize| -> usize {
        if dof >= start && dof < start + k {
            dof - start
        } else {
            k + node.rows.binary_search(&dof).expect("internal: row missing in the front")
        }
    };
    // the front matrix (lower triangle is used): original entries + the children's updates
    let mut f = vec![0.0; m * m];
    for c in 0..k {
        let col = start + c;
        for p in a.colptr[col]..a.colptr[col + 1] {
            f[local(a.rowidx[p]) * m + c] += a.val[p];
        }
    }
    let mut out = Vec::new();
    for (&c, res) in ch.iter().zip(child_res) {
        let (upd, fronts) = res?;
        out.extend(fronts);
        let crows = &sym.tree[c].rows;
        let map: Vec<usize> = crows.iter().map(|&d| local(d)).collect();
        let uc = crows.len();
        for x in 0..uc {
            let fx = map[x] * m;
            for (y, &v) in upd[x * uc..x * uc + x + 1].iter().enumerate() {
                f[fx + map[y]] += v;
            }
        }
    }

    // Blocked right-looking LDLᵀ of the first k pivots. Afterwards
    // f contains: L (first k columns) and the Schur complement (the rest).
    let mut d = vec![0.0; k];
    let mut j0 = 0;
    while j0 < k {
        let j1 = (j0 + KB).min(k);
        let jb = j1 - j0;
        // 1. the diagonal block, row-wise
        let mut lb = vec![0.0; jb * jb];
        let mut w = vec![0.0; jb];
        for i in 0..jb {
            let row = (j0 + i) * m + j0;
            for j in 0..i {
                w[j] = f[row + j] - dot(&w[..j], &lb[j * jb..j * jb + j]);
            }
            let mut di = f[row + i];
            for p in 0..i {
                di -= w[p] * w[p] / d[j0 + p];
            }
            let orig = a.diag(start + j0 + i);
            if !(orig > 0.0) || !(di > 1e-12 * orig) {
                return Err(start + j0 + i);
            }
            d[j0 + i] = di;
            for p in 0..i {
                let l = w[p] / d[j0 + p];
                lb[i * jb + p] = l;
                f[row + p] = l;
            }
        }
        if j1 < m {
            let rows = m - j1;
            let mut lbt = vec![0.0; jb * jb];
            for i in 0..jb {
                for p in 0..i {
                    lbt[p * jb + i] = lb[i * jb + p];
                }
            }
            // 2. the panel below the block: L = A·L⁻ᵀ·D⁻¹
            let tp = if (rows * jb * jb) as f64 > PAR_FLOPS { threads } else { 1 };
            {
                let (lbt, dj) = (&lbt, &d[j0..j1]);
                par::rows_mut(&mut f[j1 * m..], m, rows, tp, |_| 1.0, |_, row| {
                    let wr = &mut row[j0..j1];
                    kern::forward(wr, lbt, jb);
                    for p in 0..jb {
                        wr[p] /= dj[p];
                    }
                });
            }
            // 3. update the rest: F[x][y] −= (L·D)[x]·L[y]ᵀ for j1 ≤ y ≤ x
            let mut wt = vec![0.0; rows * jb];
            let mut lt = vec![0.0; jb * rows];
            for r in 0..rows {
                let fr = &f[(j1 + r) * m + j0..(j1 + r) * m + j1];
                for p in 0..jb {
                    wt[r * jb + p] = fr[p] * d[j0 + p];
                    lt[p * rows + r] = fr[p];
                }
            }
            let tp = if (rows * rows * jb) as f64 / 2.0 > PAR_FLOPS { threads } else { 1 };
            let (wt, lt) = (&wt, &lt);
            par::blocks_mut(&mut f[j1 * m..], m, rows, tp, |r| (r + 1) as f64, |r0, chunk| {
                kern::syrk(chunk, m, r0, chunk.len() / m, j1, rows, jb, wt, lt);
            });
        }
        j0 = j1;
    }

    // L11 packed row-wise: row i starts at i(i−1)/2
    let mut l11 = Vec::with_capacity(k * k.saturating_sub(1) / 2);
    for i in 0..k {
        l11.extend_from_slice(&f[i * m..i * m + i]);
    }
    let mut l21 = vec![0.0; u * k];
    let mut upd = vec![0.0; u * u];
    for r in 0..u {
        let fr = &f[(k + r) * m..(k + r + 1) * m];
        l21[r * k..r * k + k].copy_from_slice(&fr[..k]);
        upd[r * u..r * u + r + 1].copy_from_slice(&fr[k..k + r + 1]);
    }
    out.push(Front { start, k, rows: node.rows.clone(), l11, d, l21 });
    Ok((upd, out))
}

/// Numerical factorization. Err(degree of freedom) on a non-positive pivot.
pub fn factor(sym: &Symbolic, a: &Lower) -> Result<Factor, usize> {
    let mut fronts = Vec::new();
    for &r in &sym.roots {
        let (_, f) = factor_node(sym, a, r, par::threads())?;
        fronts.extend(f);
    }
    Ok(Factor { fronts })
}

impl Factor {
    /// The pivots D in the internal numbering (length n).
    pub fn pivots(&self, n: usize) -> Vec<f64> {
        let mut d = vec![0.0; n];
        for fr in &self.fronts {
            d[fr.start..fr.start + fr.k].copy_from_slice(&fr.d);
        }
        d
    }

    /// Solves K x = b in place.
    pub fn solve(&self, x: &mut [f64]) {
        for fr in &self.fronts {
            let k = fr.k;
            let (xp, _) = x[fr.start..].split_at_mut(k);
            for i in 0..k {
                let r = i * i.saturating_sub(1) / 2;
                let s = dot(&fr.l11[r..r + i], &xp[..i]);
                xp[i] -= s;
            }
            let xp = xp.to_vec();
            for (r, &row) in fr.rows.iter().enumerate() {
                x[row] -= dot(&fr.l21[r * k..r * k + k], &xp);
            }
        }
        for fr in &self.fronts {
            for i in 0..fr.k {
                x[fr.start + i] /= fr.d[i];
            }
        }
        for fr in self.fronts.iter().rev() {
            let k = fr.k;
            let mut xp = x[fr.start..fr.start + k].to_vec();
            for (r, &row) in fr.rows.iter().enumerate() {
                let xr = x[row];
                for (p, l) in fr.l21[r * k..r * k + k].iter().enumerate() {
                    xp[p] -= l * xr;
                }
            }
            for i in (0..k).rev() {
                let xi = xp[i];
                let r = i * i.saturating_sub(1) / 2;
                for q in 0..i {
                    xp[q] -= fr.l11[r + q] * xi;
                }
            }
            x[fr.start..fr.start + k].copy_from_slice(&xp);
        }
    }
}

// Extension (solver area): indefinite LDLᵀ with 2×2 pivoting and delayed
// pivots, unsymmetric LU and natural ordering for iterative solvers.
pub mod indef;
pub mod lu;
