//! S8R: eight-node doubly curved thick shell with reduced integration (2×2),
//! six degrees of freedom per node (U1–U3, UR1–UR3).
//!
//! Geometry and kinematics (Mindlin–Reissner, isoparametric serendipity
//! interpolation, cf. Bathe ch. 5.4 and Hughes ch. 6):
//!
//! - The nodal normals nᵢ: the mean of the elements' own normals (x,ξ × x,η at
//!   the node) where they differ by < 20°, or `*NORMAL` (`elements/normals.rs`).
//!   The drill penalty at the corners uses the element's own normal, at the midside nodes
//!   the nodal normal.
//! - At each integration point: normal n = x,ξ × x,η/|…|, local axes (e1, e2)
//!   according to `shell2::axes`, dN/dX in the tangent plane and the rotation vector
//!   θ = Σ Nᵢ θᵢ. The change of the director φ = θ × n, with the derivative
//!   φ,α = θ,α × n + θ × n,α where n,α = Σ Nᵢ,α nᵢ.
//! - Membrane: εαβ = sym(eα·u,β).
//! - Bending: καβ = sym(eα·φ,β) plus three terms in the displacement gradient
//!   H_γβ = e_γ·u,β (Cαβ = n,α·eβ, ω = ½(H21 − H12), a = ½(C12 − C21)):
//!   −½ω·[−(C12 + C21), C12 + C21, 2(C11 − C22)] (Sanders term),
//!   [(n·n,1) w,1, (n·n,2) w,2, (n·n,1) w,2 + (n·n,2) w,1] (w,β = n·u,β) and
//!   a·[H21, −H12, H22 − H11] (Voigt: κ11, κ22, 2κ12).
//! - Transverse shear: γα = n·u,α + eα·φ with the stiffness K·f(det J) at each point
//!   (K = 5/6·G·t, f = the slenderness compensation with the point's area det J).
//! - Drill: penalty kᵢ(θᵢ·nᵢ − ωᵢ)² at each node. Corners: ωᵢ = ½(e2·u,1 − e1·u,2)
//!   at the node, kᵢ = K·det Jᵢ·f(det Jᵢ). Midside nodes: ωᵢ = the chord rotation
//!   (u_q − u_p)·(nᵢ × τ)/L for the edge p→q, kᵢ = K·Ā·f(L²) with Ā = the mean
//!   of the corners' det J.
//! - Mass: consistent, 3×3 Gauss points, ρt for the displacements and ρt³/12
//!   for all three rotations.
//!
//! Tests in `tests/shells2.rs`.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, sub, G2};
use super::{add_btcb, add_outer, axes, fp, output, unit, GenStrain, Sec};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

/// Natural coordinates of the nodes (order: corners, then midside nodes 1-2, 2-3, 3-4, 4-1).
pub const NAT8: [[f64; 2]; 8] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
/// Edges (corners p → q) for the midside nodes 5–8.
const EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
/// 2×2 Gauss points (ξ fastest).
pub const GP4: [(f64, f64); 4] = [(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)];
const ND: usize = 6;
const NDOF: usize = 48;

/// Shape functions and derivatives (dN/dξ, dN/dη) for the serendipity element.
pub fn shape8(s: f64, t: f64) -> ([f64; 8], [[f64; 8]; 2]) {
    let mut n = [0.0; 8];
    let mut d = [[0.0; 8]; 2];
    for (i, &[a, b]) in NAT8.iter().enumerate() {
        if i < 4 {
            n[i] = 0.25 * (1.0 + a * s) * (1.0 + b * t) * (a * s + b * t - 1.0);
            d[0][i] = 0.25 * a * (1.0 + b * t) * (2.0 * a * s + b * t);
            d[1][i] = 0.25 * b * (1.0 + a * s) * (a * s + 2.0 * b * t);
        } else if a == 0.0 {
            n[i] = 0.5 * (1.0 - s * s) * (1.0 + b * t);
            d[0][i] = -s * (1.0 + b * t);
            d[1][i] = 0.5 * b * (1.0 - s * s);
        } else {
            n[i] = 0.5 * (1.0 + a * s) * (1.0 - t * t);
            d[0][i] = 0.5 * a * (1.0 - t * t);
            d[1][i] = -t * (1.0 + a * s);
        }
    }
    (n, d)
}

/// The surface tangent vectors x,ξ and x,η.
fn tangents(x: &[[f64; 3]], d: &[[f64; 8]; 2]) -> [[f64; 3]; 2] {
    [0, 1].map(|a| [0, 1, 2].map(|c| (0..8).map(|i| d[a][i] * x[i][c]).sum()))
}

/// Nodal normals from the element geometry.
pub fn nodal_normals(x: &[[f64; 3]]) -> [[f64; 3]; 8] {
    NAT8.map(|[s, t]| {
        let (_, d) = shape8(s, t);
        let a = tangents(x, &d);
        unit(cross(a[0], a[1]))
    })
}

/// Data at a point (ξ, η) of the surface.
pub struct Pt {
    pub n: [f64; 8],
    /// dN/dX in the local axes (2 × 8).
    pub b: [[f64; 8]; 2],
    /// Local axes (rows e1, e2, n).
    pub r: [[f64; 3]; 3],
    pub det: f64,
    /// n,X_α (global vectors) of the interpolated nodal normals.
    pub nx: [[f64; 3]; 2],
    pub x: [f64; 3],
}

pub fn point(ctx: &ElemCtx, x: &[[f64; 3]], nrm: &[[f64; 3]; 8], s: f64, t: f64) -> Pt {
    let (n, d) = shape8(s, t);
    let a = tangents(x, &d);
    let nv = cross(a[0], a[1]);
    let xp = [0, 1, 2].map(|c| (0..8).map(|i| n[i] * x[i][c]).sum());
    let r = axes(ctx, nv, xp);
    let j = [[dot(a[0], r[0]), dot(a[0], r[1])], [dot(a[1], r[0]), dot(a[1], r[1])]];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let ji = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
    let mut b = [[0.0; 8]; 2];
    for i in 0..8 {
        for al in 0..2 {
            b[al][i] = ji[al][0] * d[0][i] + ji[al][1] * d[1][i];
        }
    }
    let dn: [[f64; 3]; 2] = [0, 1].map(|al| [0, 1, 2].map(|c| (0..8).map(|i| d[al][i] * nrm[i][c]).sum()));
    let nx = [0, 1].map(|al| [0, 1, 2].map(|c| ji[al][0] * dn[0][c] + ji[al][1] * dn[1][c]));
    Pt { n, b, r, det, nx, x: xp }
}

/// Adds the 3-vector `v` (coefficient) to node i's displacement or rotation.
fn put(row: &mut [f64], i: usize, off: usize, v: [f64; 3], f: f64) {
    for c in 0..3 {
        row[i * ND + off + c] += f * v[c];
    }
}

/// B operators at an integration point: membrane (3 × 48), bending (3 × 48)
/// and transverse shear (2 × 48), in the point's local axes.
pub fn strain_ops(p: &Pt) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = NDOF;
    let [e1, e2, nv] = p.r;
    let e = [e1, e2];
    let mut bm = vec![0.0; 3 * n];
    let mut bb = vec![0.0; 3 * n];
    let mut bs = vec![0.0; 2 * n];
    let c = [0, 1].map(|a| [0, 1].map(|b| dot(p.nx[a], e[b])));
    let nn = [dot(p.nx[0], nv), dot(p.nx[1], nv)];
    let om_c = [-(c[0][1] + c[1][0]), c[0][1] + c[1][0], 2.0 * (c[0][0] - c[1][1])];
    let a_skew = 0.5 * (c[0][1] - c[1][0]);
    for i in 0..8 {
        let (b0, b1, ni) = (p.b[0][i], p.b[1][i], p.n[i]);
        // membrane
        put(&mut bm[0..n], i, 0, e1, b0);
        put(&mut bm[n..2 * n], i, 0, e2, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e1, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e2, b0);
        // bending from φ,α: eβ·φ,α = θ·(Bα (n × eβ) + N (n,α × eβ))
        let phi = |al: usize, eb: [f64; 3]| -> [f64; 3] {
            let bal = p.b[al][i];
            let u = scale(cross(nv, eb), bal);
            let w = scale(cross(p.nx[al], eb), ni);
            [u[0] + w[0], u[1] + w[1], u[2] + w[2]]
        };
        put(&mut bb[0..n], i, 3, phi(0, e1), 1.0);
        put(&mut bb[n..2 * n], i, 3, phi(1, e2), 1.0);
        put(&mut bb[2 * n..3 * n], i, 3, phi(1, e1), 1.0);
        put(&mut bb[2 * n..3 * n], i, 3, phi(0, e2), 1.0);
        // displacement terms in the curvature: H[g][b] = e_g·u,b has the coefficient B_b e_g
        // −½ω·om_c with ω = ½(H10 − H01)
        for (r, &oc) in om_c.iter().enumerate() {
            let f = -0.5 * oc * 0.5;
            put(&mut bb[r * n..(r + 1) * n], i, 0, e2, f * b0);
            put(&mut bb[r * n..(r + 1) * n], i, 0, e1, -f * b1);
        }
        // (n·n,α) w,β
        put(&mut bb[0..n], i, 0, nv, nn[0] * b0);
        put(&mut bb[n..2 * n], i, 0, nv, nn[1] * b1);
        put(&mut bb[2 * n..3 * n], i, 0, nv, nn[0] * b1 + nn[1] * b0);
        // a·[H10, −H01, H11 − H00]
        put(&mut bb[0..n], i, 0, e2, a_skew * b0);
        put(&mut bb[n..2 * n], i, 0, e1, -a_skew * b1);
        put(&mut bb[2 * n..3 * n], i, 0, e2, a_skew * b1);
        put(&mut bb[2 * n..3 * n], i, 0, e1, -a_skew * b0);
        // transverse shear γα = n·u,α + eα·φ, eα·(θ × n) = θ·(n × eα)
        put(&mut bs[0..n], i, 0, nv, b0);
        put(&mut bs[0..n], i, 3, cross(nv, e1), ni);
        put(&mut bs[n..2 * n], i, 0, nv, b1);
        put(&mut bs[n..2 * n], i, 3, cross(nv, e2), ni);
    }
    (bm, bb, bs)
}

pub struct ShellS8 {
    name: &'static str,
}

pub static S8R: ShellS8 = ShellS8 { name: "S8R" };

pub struct Setup {
    pub nrm: [[f64; 3]; 8],
    pub sec: Sec,
    pub ips: Vec<Pt>,
}

impl ShellS8 {
    pub fn setup(&self, ctx: &ElemCtx) -> Result<Setup> {
        let x = &ctx.x;
        for &[s, t] in NAT8.iter() {
            let (_, d) = shape8(s, t);
            let a = tangents(x, &d);
            if norm(cross(a[0], a[1])) <= 0.0 {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
        }
        // area normals: averaged normals / *NORMAL if present
        let nrm = match super::super::normals::of(ctx) {
            Some(v) if v.len() == 8 => [0, 1, 2, 3, 4, 5, 6, 7].map(|i| v[i]),
            _ => nodal_normals(x),
        };
        let ips: Vec<Pt> = GP4.iter().map(|&(s, t)| point(ctx, x, &nrm, s, t)).collect();
        if ips.iter().any(|p| !(p.det > 0.0)) {
            return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
        }
        Ok(Setup { nrm, sec: Sec::get(ctx, self.name)?, ips })
    }

    /// The drill penalty: vectors gᵢ (48) and stiffnesses kᵢ.
    fn drill(ctx: &ElemCtx, su: &Setup) -> Vec<(Vec<f64>, f64)> {
        let x = &ctx.x;
        let own = nodal_normals(x);
        let sec = &su.sec;
        let k0 = sec.ks[0];
        let mut dc = [0.0; 4];
        let mut out = Vec::with_capacity(8);
        for i in 0..8 {
            let [s, t] = NAT8[i];
            // area normals: the corners use the element's own normal (even when
            // the nodal normals are averaged), the midside nodes the nodal normal
            let ni = if i < 4 { own[i] } else { su.nrm[i] };
            let mut g = vec![0.0; NDOF];
            put(&mut g, i, 3, ni, 1.0);
            if i < 4 {
                // default axes suffice: ω is independent of the in-plane rotation
                let (_, d) = shape8(s, t);
                let a = tangents(x, &d);
                let r = super::default_axes(ni);
                let j = [[dot(a[0], r[0]), dot(a[0], r[1])], [dot(a[1], r[0]), dot(a[1], r[1])]];
                let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                let ji = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
                for jn in 0..8 {
                    let bx = ji[0][0] * d[0][jn] + ji[0][1] * d[1][jn];
                    let by = ji[1][0] * d[0][jn] + ji[1][1] * d[1][jn];
                    // −½(e2·u,1 − e1·u,2)
                    put(&mut g, jn, 0, r[1], -0.5 * bx);
                    put(&mut g, jn, 0, r[0], 0.5 * by);
                }
                dc[i] = det;
                out.push((g, k0 * det * fp(det, sec.t)));
            } else {
                let (p, q) = EDGES[i - 4];
                let dv = sub(x[q], x[p]);
                let l = norm(dv);
                let m = cross(ni, scale(dv, 1.0 / l));
                put(&mut g, q, 0, m, -1.0 / l);
                put(&mut g, p, 0, m, 1.0 / l);
                out.push((g, l * l));
            }
        }
        let am = dc.iter().sum::<f64>() / 4.0;
        for (g, k) in out.iter_mut().skip(4) {
            let l2 = *k;
            *k = k0 * am * fp(l2, sec.t);
            let _ = g;
        }
        out
    }

    pub fn stiffness_of(ctx: &ElemCtx, su: &Setup) -> Vec<f64> {
        let n = NDOF;
        let sec = &su.sec;
        let mut k = vec![0.0; n * n];
        let mut cm = [0.0; 9];
        let mut cb = [0.0; 9];
        let tb = sec.t.powi(3) / 12.0;
        for i in 0..9 {
            cm[i] = sec.d[i] * sec.t;
            cb[i] = sec.d[i] * tb;
        }
        for p in &su.ips {
            let (bm, bb, bs) = strain_ops(p);
            add_btcb(&mut k, n, &bm, &cm, &bm, 3, p.det);
            add_btcb(&mut k, n, &bb, &cb, &bb, 3, p.det);
            add_btcb(&mut k, n, &bs, &sec.shear_c(p.det), &bs, 2, p.det);
        }
        for (g, kd) in Self::drill(ctx, su) {
            add_outer(&mut k, n, &g, kd);
        }
        k
    }

    /// The membrane integration points for NLGEOM (quadratic terms).
    pub fn membrane_ops(su: &Setup) -> Vec<super::nlgeom::QuadGroup> {
        let n = NDOF;
        let mut c = [0.0; 9];
        for i in 0..9 {
            c[i] = su.sec.d[i] * su.sec.t;
        }
        su.ips
            .iter()
            .map(|p| {
                let (bm, _, _) = strain_ops(p);
                let a = [0, 1].map(|al| {
                    let mut m = vec![0.0; 3 * n];
                    for i in 0..8 {
                        for cc in 0..3 {
                            m[cc * n + i * ND + cc] = p.b[al][i];
                        }
                    }
                    m
                });
                super::nlgeom::QuadGroup::membrane(bm, a, c, p.det)
            })
            .collect()
    }

    /// Generalized strains at the integration points.
    pub fn gen_strains(su: &Setup, u: &[f64]) -> Vec<GenStrain> {
        let n = NDOF;
        su.ips
            .iter()
            .map(|p| {
                let (bm, bb, bs) = strain_ops(p);
                let mv = |b: &[f64], r: usize| -> f64 { (0..n).map(|q| b[r * n + q] * u[q]).sum() };
                GenStrain {
                    eps: [mv(&bm, 0), mv(&bm, 1), mv(&bm, 2)],
                    kap: [mv(&bb, 0), mv(&bb, 1), mv(&bb, 2)],
                    gam: [mv(&bs, 0), mv(&bs, 1)],
                    shear_c: su.sec.shear_c(p.det),
                    sf6: 0.0,
                }
            })
            .collect()
    }
}

/// 3×3 Gauss points (ξ, η, weight).
pub fn gp9() -> Vec<(f64, f64, f64)> {
    let g = 0.6f64.sqrt();
    let pts = [(-g, 5.0 / 9.0), (0.0, 8.0 / 9.0), (g, 5.0 / 9.0)];
    let mut v = Vec::with_capacity(9);
    for &(t, wt) in &pts {
        for &(s, ws) in &pts {
            v.push((s, t, ws * wt));
        }
    }
    v
}

/// The surface area per point and normal: (N, normal (unit), dA·weight) for 3×3 points.
fn surface_points(x: &[[f64; 3]]) -> Vec<([f64; 8], [f64; 3], f64)> {
    gp9()
        .into_iter()
        .map(|(s, t, w)| {
            let (n, d) = shape8(s, t);
            let a = tangents(x, &d);
            let c = cross(a[0], a[1]);
            let da = norm(c);
            (n, scale(c, 1.0 / da), da * w)
        })
        .collect()
}

impl ElementType for ShellS8 {
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
        8
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        4
    }
    fn vtk_type(&self) -> u8 {
        23
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2, 3, 4, 5, 6, 7]]
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        Ok(Self::stiffness_of(ctx, &su))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let su = self.setup(ctx)?;
        let k = Self::stiffness_of(ctx, &su);
        if ctx.nlgeom {
            // corotational formulation (shell2_nlgeom.rs)
            return super::nlgeom::response_s8r(ctx, self.name, &k, &Self::membrane_ops(&su), u, tangent);
        }
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        // the membrane forces at 2×2 points: ∫ N_αβ u,α·u,β dA
        let su = self.setup(ctx)?;
        Ok(super::nlgeom::geometric(&Self::membrane_ops(&su), u))
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let n = NDOF;
        let mut m = vec![0.0; n * n];
        let mt = su.sec.rho * su.sec.t;
        let mr = su.sec.rho * su.sec.t.powi(3) / 12.0;
        for (nv, _, da) in surface_points(&ctx.x) {
            for i in 0..8 {
                for j in 0..8 {
                    let f = nv[i] * nv[j] * da;
                    for d in 0..3 {
                        m[(i * ND + d) * n + j * ND + d] += f * mt;
                        m[(i * ND + 3 + d) * n + j * ND + 3 + d] += f * mr;
                    }
                }
            }
        }
        if kind == MassKind::Lumped {
            // diagonal mass with preserved total mass (Hinton–Rock–Zienkiewicz)
            let mut out = vec![0.0; n * n];
            for d in 0..ND {
                let tot: f64 = (0..8).map(|i| (0..8).map(|j| m[(i * ND + d) * n + j * ND + d]).sum::<f64>()).sum();
                let diag: f64 = (0..8).map(|i| m[(i * ND + d) * n + i * ND + d]).sum();
                for i in 0..8 {
                    let q = i * ND + d;
                    out[q * n + q] = if diag != 0.0 { m[q * n + q] * tot / diag } else { 0.0 };
                }
            }
            return Ok(out);
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let sec = Sec::get(ctx, self.name)?;
        let mut f = vec![0.0; NDOF];
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, self.name)?;
        if load.kind != "P" && body.is_none() {
            return not_implemented(&format!("load type {}", load.kind), self.name);
        }
        for (nv, nrm, da) in surface_points(&ctx.x) {
            // positive pressure P acts in the direction of the normal (as for S4R)
            let q = if load.kind == "P" { scale(nrm, load.magnitude) } else { scale(body.unwrap(), sec.t) };
            for i in 0..8 {
                for c in 0..3 {
                    f[i * ND + c] += nv[i] * da * q[c];
                }
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let sec = Sec::get(ctx, self.name)?;
        let mut v = vec![0.0; 8];
        for (nv, _, da) in surface_points(&ctx.x) {
            for i in 0..8 {
                v[i] += nv[i] * da * sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let su = self.setup(ctx)?;
        let ud;
        let u = if ctx.nlgeom {
            ud = super::nlgeom::deformational_s8r(ctx, self.name, u)?;
            &ud[..]
        } else {
            u
        };
        let mut g = Self::gen_strains(&su, u);
        if ctx.nlgeom {
            // the quadratic terms of the membrane strain in the co-rotating frame
            for (gi, eq) in g.iter_mut().zip(super::nlgeom::quad_strains(&Self::membrane_ops(&su), u)) {
                for r in 0..3 {
                    gi.eps[r] += eq[r];
                }
            }
        }
        Ok(output(&su.sec, &g, false))
    }
}
