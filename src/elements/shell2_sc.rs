//! Continuum shells SC8R (8 nodes, hexahedron) and SC6R (6 nodes, wedge) with
//! reduced integration and displacement degrees of freedom only.
//!
//! Nodes 1–4 (1–3) lie on the bottom face and 5–8 (4–6) on the top face;
//! the thickness direction is given by the node order. The element is computed as a shell on
//! the midsurface xᵢ = ½(x_bottom + x_top) with the directors dᵢ = x_t − x_b
//! (thickness hᵢ = |dᵢ|, t = the mean):
//!
//! - The midsurface displacement uᵢ = ½(u_b + u_t), the "rotations" from
//!   Δᵢ = u_t − u_b perpendicular to the director: θy = e1′·Δ/h, θx = −e2′·Δ/h
//!   (e1′, e2′ = the local axes projected perpendicular to dᵢ), and
//!   the thickness strain ε33,ᵢ = dᵢ·Δᵢ/|dᵢ|².
//! - Membrane, bending, hourglass control and transverse shear exactly as S4R
//!   (SC8R) and S3R (SC6R), respectively, in `shell.rs`/`shell_tri.rs`, but without drill.
//! - Thickness direction: penalty with the thickness modulus E_t (default 2G) on
//!   the effective thickness strain, ε33 − ε33,P where the Poisson strain
//!   ε33,P = −(ε11 + ε22) (the section default POISSON = 0.5):
//!   Φ = ½ E_t t [A (ε̄33 + ε11 + ε22)² + Σ Aᵢ (ε33,ᵢ − ε̄33)²], with
//!   Aᵢ = ∫Nᵢ dA, ε̄33 = Σ Aᵢ ε33,ᵢ / A and the membrane strain at the center.
//!   SF6 = E_t t (ε̄33 + ε11 + ε22).
//! - Mass: in the normal direction diagonal ρV/8
//!   per node; in the plane for the midsurface displacements ρV(1/16 + C_m/144000) and
//!   for Δ ρVt(C_Δ + c·C_h)/2304, where C_m, C_Δ and C_h are circulant 4×4
//!   matrices with the first rows (7, −1, −5, −1), (5, 3, 1, 3) and (1, −1, 1, −1)
//!   and c = 1/3000 (SC8R) and 1/3 (SC6R), respectively. V = the midsurface area times
//!   the mean projection of the directors onto the normal. SC6R is computed as a collapsed
//!   SC8R (node 4 = node 1). The matrix for Δ is not dimensionally consistent (∝ ρVt);
//!   the rotary inertia is small for thin shells.
//!
//! For generally distorted quadrilaterals the same limitations apply as for
//! S4R (docs/REFERENCE.md).

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, sub};
use super::{add_btcb, add_outer, axes, output, unit, GenStrain, Sec};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

/// Local degrees of freedom per node in the underlying shell: [u, v, w, θx, θy, θz].
const ND: usize = 6;
const NAT4: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

pub const HEX8_FACES: &[&[usize]] = &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]];
pub const WEDGE6_FACES: &[&[usize]] = &[&[0, 1, 2], &[3, 5, 4], &[0, 3, 4, 1], &[1, 4, 5, 2], &[2, 5, 3, 0]];

fn inv2(j: [[f64; 2]; 2]) -> ([[f64; 2]; 2], f64) {
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    ([[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]], det)
}

/// Bilinear quadrilateral: N, dN/dξ, dN/dη, J, det J, dN/dx, dN/dy.
struct Q4 {
    n: [f64; 4],
    ds: [f64; 4],
    dt: [f64; 4],
    j: [[f64; 2]; 2],
    det: f64,
    bx: [f64; 4],
    by: [f64; 4],
}

fn q4(x: &[[f64; 2]], s: f64, t: f64) -> Q4 {
    let mut n = [0.0; 4];
    let mut ds = [0.0; 4];
    let mut dt = [0.0; 4];
    for i in 0..4 {
        let (a, b) = (NAT4[i][0], NAT4[i][1]);
        n[i] = 0.25 * (1.0 + a * s) * (1.0 + b * t);
        ds[i] = 0.25 * a * (1.0 + b * t);
        dt[i] = 0.25 * b * (1.0 + a * s);
    }
    let mut j = [[0.0; 2]; 2];
    for i in 0..4 {
        j[0][0] += ds[i] * x[i][0];
        j[0][1] += ds[i] * x[i][1];
        j[1][0] += dt[i] * x[i][0];
        j[1][1] += dt[i] * x[i][1];
    }
    let (ji, det) = inv2(j);
    let bx = [0, 1, 2, 3].map(|i| ji[0][0] * ds[i] + ji[0][1] * dt[i]);
    let by = [0, 1, 2, 3].map(|i| ji[1][0] * ds[i] + ji[1][1] * dt[i]);
    Q4 { n, ds, dt, j, det, bx, by }
}

/// Linear triangle: dN/dx, dN/dy and the area.
fn cst(x: &[[f64; 2]]) -> ([f64; 3], [f64; 3], f64) {
    let (x1, y1, x2, y2, x3, y3) = (x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1]);
    let det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    ([(y2 - y3) / det, (y3 - y1) / det, (y1 - y2) / det], [(x3 - x2) / det, (x1 - x3) / det, (x2 - x1) / det], 0.5 * det)
}

/// Membrane B (3 × 6nc).
fn bmem(bx: &[f64], by: &[f64]) -> Vec<f64> {
    let nc = bx.len();
    let n = ND * nc;
    let mut b = vec![0.0; 3 * n];
    for i in 0..nc {
        b[i * ND] = bx[i];
        b[n + i * ND + 1] = by[i];
        b[2 * n + i * ND] = by[i];
        b[2 * n + i * ND + 1] = bx[i];
    }
    b
}

/// Bending B (3 × 6nc): κ = [θy,x, −θx,y, θy,y − θx,x].
fn bbend(bx: &[f64], by: &[f64]) -> Vec<f64> {
    let nc = bx.len();
    let n = ND * nc;
    let mut b = vec![0.0; 3 * n];
    for i in 0..nc {
        b[i * ND + 4] = bx[i];
        b[n + i * ND + 3] = -by[i];
        b[2 * n + i * ND + 4] = by[i];
        b[2 * n + i * ND + 3] = -bx[i];
    }
    b
}

/// Covariant transverse shear strain (S4R) e_ξ (dir = 0) or e_η (dir = 1) at (s, t).
fn cov_shear(x: &[[f64; 2]], s: f64, t: f64, dir: usize) -> Vec<f64> {
    let p = q4(x, s, t);
    let d = if dir == 0 { p.ds } else { p.dt };
    let xv = p.j[dir];
    let mut g = vec![0.0; ND * 4];
    for i in 0..4 {
        g[i * ND + 2] = d[i];
        g[i * ND + 4] = p.n[i] * xv[0];
        g[i * ND + 3] = -p.n[i] * xv[1];
    }
    g
}

/// The transverse shear as in S4R: edge strains s (4 × 24), coefficient matrix c
/// (4 × 4) and M (2 × 4) with the constant field γ0 = M s.
fn shear_q4(x: &[[f64; 2]], ks: [f64; 4]) -> (Vec<f64>, [f64; 16], [[f64; 4]; 2]) {
    let rows = [cov_shear(x, 0.0, -1.0, 0), cov_shear(x, -1.0, 0.0, 1), cov_shear(x, 0.0, 1.0, 0), cov_shear(x, 1.0, 0.0, 1)];
    let mut s = vec![0.0; 4 * 24];
    for (r, v) in rows.iter().enumerate() {
        s[r * 24..(r + 1) * 24].copy_from_slice(v);
    }
    let p0 = q4(x, 0.0, 0.0);
    let a = 4.0 * p0.det;
    let (ji, _) = inv2(p0.j);
    let m = [
        [0.5 * ji[0][0], 0.5 * ji[0][1], 0.5 * ji[0][0], 0.5 * ji[0][1]],
        [0.5 * ji[1][0], 0.5 * ji[1][1], 0.5 * ji[1][0], 0.5 * ji[1][1]],
    ];
    let mut c = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            // γ0ᵀ C γ0 with C = [[K11, K12], [K12, K22]]
            let mut v = 0.0;
            for r in 0..2 {
                for q in 0..2 {
                    v += m[r][i] * ks[r * 2 + q] * m[q][j];
                }
            }
            c[i * 4 + j] = a * v;
        }
    }
    // the hourglass part: [T Q2]⁻¹ with T = [x,ξ; x,η; x,ξ; x,η], Q2 = [(−1,−1,1,1), (−1,1,1,−1)]
    let xs = p0.j[0];
    let xt = p0.j[1];
    let tq = [xs[0], xs[1], -1.0, -1.0, xt[0], xt[1], -1.0, 1.0, xs[0], xs[1], 1.0, 1.0, xt[0], xt[1], 1.0, -1.0];
    if let Some(mi) = inv4(&tq) {
        let alpha = ks[0] * 8.0 / 3.0 * (xs[0] * xs[0] + xs[1] * xs[1] + xt[0] * xt[0] + xt[1] * xt[1]) / a;
        for (h, cf) in [alpha, 0.004 * alpha].into_iter().enumerate() {
            let row = &mi[(2 + h) * 4..(3 + h) * 4];
            for i in 0..4 {
                for j in 0..4 {
                    c[i * 4 + j] += cf * row[i] * row[j];
                }
            }
        }
    }
    (s, c, m)
}

/// Inverse of a 4×4 matrix (Gauss–Jordan with pivoting).
fn inv4(a: &[f64; 16]) -> Option<[f64; 16]> {
    let n = 4;
    let mut m = *a;
    let mut inv = [0.0; 16];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().partial_cmp(&m[j * n + c].abs()).unwrap())?;
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        for k in 0..n {
            m.swap(c * n + k, p * n + k);
            inv.swap(c * n + k, p * n + k);
        }
        let d = m[c * n + c];
        for k in 0..n {
            m[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * n + c];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[c * n + k];
                        inv[r * n + k] -= f * inv[c * n + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

/// The transverse shear as in S3R: constant field B0 (2 × 18), the circulation ρ (18) and the edge lengths.
fn shear_tri(x: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>, [f64; 3]) {
    let n = 3 * ND;
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    let mut s = vec![vec![0.0; n]; 3];
    let mut t = [[0.0; 2]; 3];
    let mut l = [0.0; 3];
    for (k, &(i, j)) in edges.iter().enumerate() {
        let d = [x[j][0] - x[i][0], x[j][1] - x[i][1]];
        l[k] = (d[0] * d[0] + d[1] * d[1]).sqrt();
        t[k] = [d[0] / l[k], d[1] / l[k]];
        s[k][i * ND + 2] -= 1.0 / l[k];
        s[k][j * ND + 2] += 1.0 / l[k];
        for m in [i, j] {
            s[k][m * ND + 4] += 0.5 * t[k][0];
            s[k][m * ND + 3] -= 0.5 * t[k][1];
        }
    }
    let pairs = [(2usize, 0usize), (0, 1), (1, 2)];
    let mut w = [0.0; 3];
    for (v, &(a, b)) in pairs.iter().enumerate() {
        w[v] = (l[a] * l[a] - l[b] * l[b]).powi(2);
    }
    let sw: f64 = w.iter().sum();
    if sw > 1e-24 * (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).powi(2) {
        for wv in w.iter_mut() {
            *wv /= sw;
        }
    } else {
        w = [1.0 / 3.0; 3];
    }
    let mut b0 = vec![0.0; 2 * n];
    for (v, &(a, b)) in pairs.iter().enumerate() {
        let det = t[a][0] * t[b][1] - t[a][1] * t[b][0];
        let inv = [[t[b][1] / det, -t[a][1] / det], [-t[b][0] / det, t[a][0] / det]];
        for r in 0..2 {
            for q in 0..n {
                b0[r * n + q] += w[v] * (inv[r][0] * s[a][q] + inv[r][1] * s[b][q]);
            }
        }
    }
    let rho: Vec<f64> = (0..n).map(|q| (0..3).map(|k| l[k] * s[k][q]).sum()).collect();
    (b0, rho, l)
}

pub struct ContShell {
    name: &'static str,
    /// Corners per face (4 = SC8R, 3 = SC6R).
    nc: usize,
}

pub static SC8R: ContShell = ContShell { name: "SC8R", nc: 4 };
pub static SC6R: ContShell = ContShell { name: "SC6R", nc: 3 };

/// Geometry: local system, planar midsurface coordinates, directors.
pub struct Geo {
    /// Local axes (rows e1, e2, n) at the midsurface center.
    pub r: [[f64; 3]; 3],
    pub xl: Vec<[f64; 2]>,
    /// Directors dᵢ = x_t − x_b.
    pub d: Vec<[f64; 3]>,
    pub h: Vec<f64>,
    /// Mean thickness.
    pub t: f64,
    /// The midsurface area and the nodal areas ∫Nᵢ dA.
    pub area: f64,
    pub ai: Vec<f64>,
}

/// Section and thickness modulus.
pub struct Setup {
    pub geo: Geo,
    pub sec: Sec,
    pub et: f64,
}

impl ContShell {
    fn geo(&self, ctx: &ElemCtx) -> Result<Geo> {
        let nc = self.nc;
        let x = &ctx.x;
        let xm: Vec<[f64; 3]> = (0..nc).map(|i| scale([0, 1, 2].map(|c| x[i][c] + x[i + nc][c]), 0.5)).collect();
        let d: Vec<[f64; 3]> = (0..nc).map(|i| sub(x[i + nc], x[i])).collect();
        let h: Vec<f64> = d.iter().map(|v| norm(*v)).collect();
        let n = if nc == 4 { cross(sub(xm[2], xm[0]), sub(xm[3], xm[1])) } else { cross(sub(xm[1], xm[0]), sub(xm[2], xm[0])) };
        if norm(n) == 0.0 || h.iter().any(|&v| !(v > 0.0)) {
            return fail(format!("element {} ({}) has zero area or thickness", ctx.label, self.name));
        }
        let c = common::centroid(&xm);
        let r = axes(ctx, n, c);
        let xl: Vec<[f64; 2]> = xm.iter().map(|p| {
            let q = sub(*p, c);
            [dot(r[0], q), dot(r[1], q)]
        }).collect();
        // the directors must point in the same direction as the midsurface normal (bottom → top)
        if d.iter().any(|v| dot(*v, r[2]) <= 0.0) {
            return fail(format!(
                "element {} ({}): nodes 1–{} must lie on the bottom face and {}–{} on the top face with the normal pointing up",
                ctx.label,
                self.name,
                nc,
                nc + 1,
                2 * nc
            ));
        }
        let (area, ai) = if nc == 4 {
            let mut ai = vec![0.0; 4];
            let mut area = 0.0;
            for &(s, t) in &super::s8r::GP4 {
                let p = q4(&xl, s, t);
                if p.det <= 0.0 {
                    return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
                }
                area += p.det;
                for i in 0..4 {
                    ai[i] += p.n[i] * p.det;
                }
            }
            (area, ai)
        } else {
            let (_, _, a) = cst(&xl);
            if a <= 0.0 {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
            (a, vec![a / 3.0; 3])
        };
        let t = h.iter().sum::<f64>() / nc as f64;
        Ok(Geo { r, xl, d, h, t, area, ai })
    }

    pub fn setup(&self, ctx: &ElemCtx) -> Result<Setup> {
        let geo = self.geo(ctx)?;
        let mut sec = Sec::get(ctx, self.name)?;
        // the thickness is given by the nodes; the section thickness is not used
        let t0 = sec.t;
        sec.t = geo.t;
        if sec.ks[0] > 0.0 && ctx.section.map(|s| s.raw.param(super::super::beam_timoshenko::section::TSS_PARAM).is_none()).unwrap_or(true) {
            for k in sec.ks.iter_mut() {
                *k *= geo.t / t0;
            }
        }
        let raw = &ctx.section(self.name)?.raw;
        if let Some(v) = raw.param("POISSON") {
            if crate::fem::model::parse_f64(v) != Some(0.5) {
                return fail(format!("element {} ({}): *SHELL SECTION, POISSON={v} is not supported (only the default value 0.5)", ctx.label, self.name));
            }
        }
        let et = match raw.param("THICKNESS MODULUS") {
            Some(v) => match crate::fem::model::parse_f64(v) {
                Some(e) if e > 0.0 => e,
                _ => return fail(format!("element {} ({}): THICKNESS MODULUS={v} is not supported (give a positive number)", ctx.label, self.name)),
            },
            None => 2.0 * sec.g,
        };
        Ok(Setup { geo, sec, et })
    }

    /// Transformation from the element's degrees of freedom (3 per node) to the
    /// underlying shell's local ones (6 per corner), plus the thickness strains
    /// (one row per corner).
    fn transform(&self, g: &Geo) -> (Vec<f64>, Vec<f64>) {
        let nc = self.nc;
        let n = 3 * 2 * nc;
        let ns = ND * nc;
        let mut t = vec![0.0; ns * n];
        let mut te = vec![0.0; nc * n];
        for i in 0..nc {
            let (b, tp) = (3 * i, 3 * (i + nc));
            let dn = scale(g.d[i], 1.0 / g.h[i]);
            let e1 = unit(sub(g.r[0], scale(dn, dot(g.r[0], dn))));
            let e2 = cross(dn, e1);
            for c in 0..3 {
                for k in 0..3 {
                    t[(ND * i + k) * n + b + c] += 0.5 * g.r[k][c];
                    t[(ND * i + k) * n + tp + c] += 0.5 * g.r[k][c];
                }
                // θy = e1′·Δ/h, θx = −e2′·Δ/h
                t[(ND * i + 4) * n + tp + c] += e1[c] / g.h[i];
                t[(ND * i + 4) * n + b + c] -= e1[c] / g.h[i];
                t[(ND * i + 3) * n + tp + c] -= e2[c] / g.h[i];
                t[(ND * i + 3) * n + b + c] += e2[c] / g.h[i];
                // ε33 = d·Δ/|d|²
                let f = g.d[i][c] / (g.h[i] * g.h[i]);
                te[i * n + tp + c] += f;
                te[i * n + b + c] -= f;
            }
        }
        (t, te)
    }

    /// The midsurface membrane B (constant/center).
    fn bm0(&self, g: &Geo) -> Vec<f64> {
        if self.nc == 4 {
            let p = q4(&g.xl, 0.0, 0.0);
            bmem(&p.bx, &p.by)
        } else {
            let (bx, by, _) = cst(&g.xl);
            bmem(&bx, &by)
        }
    }

    /// The local stiffness of the underlying shell (6nc × 6nc) without drill.
    fn shell_k(&self, g: &Geo, sec: &Sec) -> Vec<f64> {
        let nc = self.nc;
        let n = ND * nc;
        let mut k = vec![0.0; n * n];
        let tb = sec.t.powi(3) / 12.0;
        let mut c6 = [0.0; 36];
        for i in 0..3 {
            for j in 0..3 {
                c6[i * 6 + j] = sec.d[i * 3 + j] * sec.t;
                c6[(i + 3) * 6 + j + 3] = sec.d[i * 3 + j] * tb;
            }
        }
        let stack = |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().chain(b).copied().collect() };
        if nc == 4 {
            let p0 = q4(&g.xl, 0.0, 0.0);
            let a = 4.0 * p0.det;
            let b = stack(&bmem(&p0.bx, &p0.by), &bbend(&p0.bx, &p0.by));
            add_btcb(&mut k, n, &b, &c6, &b, 6, a);
            // Flanagan–Belytschko stabilization like S4R
            let hv = [1.0, -1.0, 1.0, -1.0];
            let hx: f64 = (0..4).map(|i| hv[i] * g.xl[i][0]).sum();
            let hy: f64 = (0..4).map(|i| hv[i] * g.xl[i][1]).sum();
            let gam: [f64; 4] = [0, 1, 2, 3].map(|i| hv[i] - hx * p0.bx[i] - hy * p0.by[i]);
            let bb: f64 = (0..4).map(|i| p0.bx[i] * p0.bx[i] + p0.by[i] * p0.by[i]).sum();
            let cm = 0.0003125 * sec.d[8] * sec.t * a * bb;
            let cb = 0.01125 * sec.d[8] * tb * a * bb;
            for (dd, cf) in [(0usize, cm), (1, cm), (3, cb), (4, cb)] {
                let mut gv = vec![0.0; n];
                for i in 0..4 {
                    gv[i * ND + dd] = gam[i];
                }
                add_outer(&mut k, n, &gv, cf);
            }
            let (s, c, _) = shear_q4(&g.xl, sec.shear_c(a));
            add_btcb(&mut k, n, &s, &c, &s, 4, 1.0);
        } else {
            let (bx, by, a) = cst(&g.xl);
            let b = stack(&bmem(&bx, &by), &bbend(&bx, &by));
            add_btcb(&mut k, n, &b, &c6, &b, 6, a);
            let sc = sec.shear_c(a);
            let (b0, rho, l) = shear_tri(&g.xl);
            add_btcb(&mut k, n, &b0, &sc, &b0, 2, a);
            add_outer(&mut k, n, &rho, sc[0] * (l[0] * l[0] + l[2] * l[2]) / (48000.0 * a));
        }
        k
    }

    /// The thickness penalty vectors: (Aᵢ, ε33,ᵢ) and ε̄33 + ε11 + ε22 (rows in the element's degrees of freedom).
    fn thickness_rows(&self, g: &Geo, t: &[f64], te: &[f64]) -> Vec<f64> {
        let n = 6 * self.nc;
        let ns = ND * self.nc;
        let bm = self.bm0(g);
        let mut ev = vec![0.0; n];
        for q in 0..n {
            let mut v = 0.0;
            for r in 0..2 {
                for p in 0..ns {
                    v += bm[r * ns + p] * t[p * n + q];
                }
            }
            ev[q] = v;
        }
        let mut em = vec![0.0; n];
        for i in 0..self.nc {
            for q in 0..n {
                em[q] += g.ai[i] * te[i * n + q] / g.area;
            }
        }
        let mut rows: Vec<f64> = (0..n).map(|q| em[q] + ev[q]).collect();
        rows.extend_from_slice(&em);
        rows
    }

    pub fn stiffness_of(&self, su: &Setup) -> Vec<f64> {
        let g = &su.geo;
        let nc = self.nc;
        let n = 6 * nc;
        let (t, te) = self.transform(g);
        let ks = self.shell_k(g, &su.sec);
        let mut k = common::congruence(&ks, &t, n);
        let rows = self.thickness_rows(g, &t, &te);
        let (ev, em) = rows.split_at(n);
        let f = su.et * g.t;
        for i in 0..nc {
            add_outer(&mut k, n, &te[i * n..(i + 1) * n], f * g.ai[i]);
        }
        add_outer(&mut k, n, em, -f * g.area);
        add_outer(&mut k, n, ev, f * g.area);
        // to global coordinates: the element's degrees of freedom are already global
        k
    }

    /// Generalized strains (one integration point) from the displacements.
    fn gen_strain(&self, su: &Setup, u: &[f64]) -> GenStrain {
        let g = &su.geo;
        let nc = self.nc;
        let n = 6 * nc;
        let ns = ND * nc;
        let (t, te) = self.transform(g);
        let ul: Vec<f64> = (0..ns).map(|p| (0..n).map(|q| t[p * n + q] * u[q]).sum()).collect();
        let mv = |b: &[f64], r: usize| -> f64 { (0..ns).map(|q| b[r * ns + q] * ul[q]).sum() };
        let (eps, kap, gam, shear_c) = if nc == 4 {
            let p0 = q4(&g.xl, 0.0, 0.0);
            let a = 4.0 * p0.det;
            let bm = bmem(&p0.bx, &p0.by);
            let bb = bbend(&p0.bx, &p0.by);
            let sc = su.sec.shear_c(a);
            let (s, _, m) = shear_q4(&g.xl, sc);
            let sv: Vec<f64> = (0..4).map(|r| mv(&s, r)).collect();
            let g0 = [0, 1].map(|r| (0..4).map(|q| m[r][q] * sv[q]).sum::<f64>());
            ([mv(&bm, 0), mv(&bm, 1), mv(&bm, 2)], [mv(&bb, 0), mv(&bb, 1), mv(&bb, 2)], g0, sc)
        } else {
            let (bx, by, a) = cst(&g.xl);
            let bm = bmem(&bx, &by);
            let bb = bbend(&bx, &by);
            let (b0, _, _) = shear_tri(&g.xl);
            ([mv(&bm, 0), mv(&bm, 1), mv(&bm, 2)], [mv(&bb, 0), mv(&bb, 1), mv(&bb, 2)], [mv(&b0, 0), mv(&b0, 1)], su.sec.shear_c(a))
        };
        let rows = self.thickness_rows(g, &t, &te);
        let evol: f64 = (0..n).map(|q| rows[q] * u[q]).sum();
        GenStrain { eps, kap, gam, shear_c, sf6: su.et * g.t * evol }
    }

    /// dN/dx and dN/dy for the midsurface at the center.
    fn center_grad(&self, g: &Geo) -> (Vec<f64>, Vec<f64>) {
        if self.nc == 4 {
            let p = q4(&g.xl, 0.0, 0.0);
            (p.bx.to_vec(), p.by.to_vec())
        } else {
            let (bx, by, _) = cst(&g.xl);
            (bx.to_vec(), by.to_vec())
        }
    }

    /// The membrane integration point (the center) for NLGEOM: u,α for the midsurface
    /// displacement ½(u_b + u_t) in global components.
    pub fn membrane_ops(&self, su: &Setup) -> Vec<super::nlgeom::QuadGroup> {
        let g = &su.geo;
        let nc = self.nc;
        let n = 6 * nc;
        let ns = ND * nc;
        let (t, _) = self.transform(g);
        let bm0 = self.bm0(g);
        let mut bm = vec![0.0; 3 * n];
        for r in 0..3 {
            for q in 0..n {
                bm[r * n + q] = (0..ns).map(|p| bm0[r * ns + p] * t[p * n + q]).sum();
            }
        }
        let (bx, by) = self.center_grad(g);
        let a = [bx, by].map(|b| {
            let mut m = vec![0.0; 3 * n];
            for i in 0..nc {
                for c in 0..3 {
                    m[c * n + 3 * i + c] = 0.5 * b[i];
                    m[c * n + 3 * (i + nc) + c] = 0.5 * b[i];
                }
            }
            m
        });
        let mut c = [0.0; 9];
        for i in 0..9 {
            c[i] = su.sec.d[i] * su.sec.t;
        }
        let mem = super::nlgeom::QuadGroup::membrane(bm, a, c, g.area);
        // the thickness direction: e = [ε33,1 … ε33,nc, ε11 + ε22] with the Green terms
        // ½|Δᵢ|²/hᵢ² and the membrane's, respectively, C according to the module comment
        let (_, te) = self.transform(g);
        let m = nc + 1;
        let mut l = vec![0.0; m * n];
        l[..nc * n].copy_from_slice(&te);
        for q in 0..n {
            l[nc * n + q] = mem.l[q] + mem.l[n + q];
        }
        let mut q = Vec::with_capacity(m);
        for i in 0..nc {
            let mut qi = vec![0.0; n * n];
            let h2 = g.h[i] * g.h[i];
            let (b, tp) = (3 * i, 3 * (i + nc));
            for c in 0..3 {
                // Δ = u_t − u_b
                for (p1, s1) in [(tp + c, 1.0), (b + c, -1.0)] {
                    for (p2, s2) in [(tp + c, 1.0), (b + c, -1.0)] {
                        qi[p1 * n + p2] += s1 * s2 / h2;
                    }
                }
            }
            q.push(qi);
        }
        q.push((0..n * n).map(|k| mem.q[0][k] + mem.q[1][k]).collect());
        let f = su.et * g.t;
        let wbar: Vec<f64> = (0..m).map(|k| if k < nc { g.ai[k] / g.area } else { 0.0 }).collect();
        let ev = |k: usize| wbar[k] + if k == nc { 1.0 } else { 0.0 };
        let mut cm = vec![0.0; m * m];
        for a in 0..m {
            for bb in 0..m {
                let mut v = g.area * ev(a) * ev(bb);
                for i in 0..nc {
                    let di = |k: usize| (if k == i { 1.0 } else { 0.0 }) - wbar[k];
                    v += g.ai[i] * di(a) * di(bb);
                }
                cm[a * m + bb] = f * v;
            }
        }
        vec![mem, super::nlgeom::QuadGroup { l, q, c: cm, w: 1.0 }]
    }

    /// The mass matrix (see the module comment).
    fn mass_of(&self, su: &Setup, kind: MassKind) -> Vec<f64> {
        let g = &su.geo;
        let nc = self.nc;
        let n = 6 * nc;
        // the element volume: the midsurface area times the projection of the directors onto the normal
        let hn = g.d.iter().map(|d| dot(*d, g.r[2])).sum::<f64>() / nc as f64;
        let m = su.sec.rho * g.area * hn;
        let r = &g.r;
        let mut out = vec![0.0; n * n];
        // the matrices of the square (b/t nodes), collapsed for SC6R: node 4 = node 1
        let map: Vec<usize> = if nc == 4 { vec![0, 1, 2, 3] } else { vec![0, 1, 2, 0] };
        let circ = |v: [f64; 4], i: usize, j: usize| v[(j + 4 - i) % 4];
        let lumped = kind == MassKind::Lumped;
        // the normal direction (and with diagonal mass all directions): ρV/8 per node in the square
        for qa in 0..8 {
            let a = map[qa % 4] + if qa < 4 { 0 } else { nc };
            for c in 0..3 {
                for d in 0..3 {
                    let w = if lumped { if c == d { 1.0 } else { 0.0 } } else { r[2][c] * r[2][d] };
                    out[(3 * a + c) * n + 3 * a + d] += m / 8.0 * w;
                }
            }
        }
        if lumped {
            return out;
        }
        // in the plane: midsurface and Δ, expressed in the bottom/top face nodes; the hourglass
        // term of the Δ part has the factor 1/3000 for SC8R and 1/3 for SC6R
        let t = g.t;
        let hg = if nc == 4 { 1.0 / 3000.0 } else { 1.0 / 3.0 };
        for i in 0..4 {
            for j in 0..4 {
                let mm = m * (1.0 / 16.0 + circ([7.0, -1.0, -5.0, -1.0], i, j) / 144000.0);
                let md = m * t * (circ([5.0, 3.0, 1.0, 3.0], i, j) + circ([1.0, -1.0, 1.0, -1.0], i, j) * hg) / 2304.0;
                // u_b = u_m − Δ/2, u_t = u_m + Δ/2 ⇒ M_bb = M_m/4 + M_Δ, M_bt = M_m/4 − M_Δ
                for (si, sj, v) in [(0, 0, mm / 4.0 + md), (1, 1, mm / 4.0 + md), (0, 1, mm / 4.0 - md), (1, 0, mm / 4.0 - md)] {
                    let a = map[i] + si * nc;
                    let b = map[j] + sj * nc;
                    for k in 0..2 {
                        for c in 0..3 {
                            for d in 0..3 {
                                out[(3 * a + c) * n + 3 * b + d] += v * r[k][c] * r[k][d];
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

impl ElementType for ContShell {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Shell
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        2 * self.nc
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3]
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        if self.nc == 4 { 12 } else { 13 }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.nc == 4 { HEX8_FACES } else { WEDGE6_FACES }
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        Ok(self.stiffness_of(&su))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let su = self.setup(ctx)?;
        let k = self.stiffness_of(&su);
        if ctx.nlgeom {
            // corotational formulation (shell2_nlgeom.rs)
            return super::nlgeom::response_sc(ctx, self.name, &k, &self.membrane_ops(&su), u, tangent);
        }
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        // the membrane forces at the center: ∫ N_αβ u,α·u,β dA for the midsurface displacement
        // (all three components, see
        // tests/cases/shells2/sc8r_buckle_panel)
        let su = self.setup(ctx)?;
        let mut groups = self.membrane_ops(&su);
        groups.truncate(1);
        Ok(super::nlgeom::geometric(&groups, u))
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        Ok(self.mass_of(&su, kind))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        if let Some(f) = common::pressure_load(&ctx.x, self.faces(), &load.kind, load.magnitude, 1.0, 3, self.name)? {
            return Ok(f);
        }
        let su = self.setup(ctx)?;
        let Some(b) = common::body_load_vector(&load.kind, load.magnitude, &load.extra, su.sec.rho, self.name)? else {
            return not_implemented(&format!("load type {}", load.kind), self.name);
        };
        let w = self.volume_weights(ctx)?;
        let mut f = vec![0.0; 3 * w.len()];
        for (i, wi) in w.iter().enumerate() {
            for c in 0..3 {
                f[3 * i + c] = b[c] * wi;
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let g = self.geo(ctx)?;
        let nc = self.nc;
        let mut v = vec![0.0; 2 * nc];
        for i in 0..nc {
            v[i] = 0.5 * g.ai[i] * g.t;
            v[i + nc] = 0.5 * g.ai[i] * g.t;
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let su = self.setup(ctx)?;
        let ud;
        let u = if ctx.nlgeom {
            ud = super::nlgeom::deformational_sc(ctx, u);
            &ud[..]
        } else {
            u
        };
        let mut g = self.gen_strain(&su, u);
        if ctx.nlgeom {
            let eq = super::nlgeom::quad_strains(&self.membrane_ops(&su), u);
            for r in 0..3 {
                g.eps[r] += eq[0][r];
            }
            let nc = self.nc;
            let ebar: f64 = (0..nc).map(|i| su.geo.ai[i] * eq[1][i]).sum::<f64>() / su.geo.area;
            g.sf6 += su.et * su.geo.t * (ebar + eq[1][nc]);
        }
        Ok(output(&su.sec, &[g], true))
    }
}
