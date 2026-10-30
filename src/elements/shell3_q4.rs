//! S4R5: thin four-node shell with reduced integration and five degrees of freedom
//! per node (area shells3).
//!
//! Flat formulation in a local system (normal (x3 − x1) × (x4 − x2), 1-axis
//! according to the default axes), for flat elements of arbitrary shape:
//!
//! - Membrane and bending: one-point integration (A = 4·det J at the center) with
//!   Flanagan–Belytschko stabilization γ = (h − (h·x)bx − (h·y)by)/4,
//!   h = (1, −1, 1, −1), on u, v with 0.08·G·t·A·Σb·b/16 and on θx, θy with
//!   0.72·G·t³/12·A·Σb·b/16.
//! - Transverse shear (the Kirchhoff condition as a penalty): the shear strain along
//!   the edge, γ = (w,ξ + β·x,ξ)/|x,ξ|, at the midpoint of each edge with the weight
//!   det J(0, 0)·2 and the stiffness 5/6·G·t·f(A/2) (βx = θy, βy = −θx).
//! - The rotation about the normal has no stiffness (five degrees of freedom per node),
//!   see `shell3.rs`.
//! - Mass: out of plane ρtA/36·[4 2 1 2] (cyclic), in plane ρtA/16 plus
//!   ρtA·(7, −1, −5, −1)/144000, rotary inertia
//!   ρt³/12·A·(17/48, 1/16, 1/48, 1/16) (values for rectangles).
//!
//! Output at one integration point: S, SF (SF4 = SF5 = 0) and SM.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, sub};
use super::{add_drill_eps, default_axes, fp, k_shear, output, project_tangent, section, shear_modulus, Gen};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct S4r5;
pub static S4R5: S4r5 = S4r5;

const ND: usize = 6;
const NDOF: usize = 24;
const NAT4: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

fn shape4(s: f64, t: f64) -> ([f64; 4], [[f64; 4]; 2]) {
    let mut n = [0.0; 4];
    let mut d = [[0.0; 4]; 2];
    for i in 0..4 {
        let [a, b] = NAT4[i];
        n[i] = 0.25 * (1.0 + a * s) * (1.0 + b * t);
        d[0][i] = 0.25 * a * (1.0 + b * t);
        d[1][i] = 0.25 * b * (1.0 + a * s);
    }
    (n, d)
}

fn jac(x: &[[f64; 2]; 4], d: &[[f64; 4]; 2]) -> [[f64; 2]; 2] {
    let mut j = [[0.0; 2]; 2];
    for a in 0..2 {
        for i in 0..4 {
            j[a][0] += d[a][i] * x[i][0];
            j[a][1] += d[a][i] * x[i][1];
        }
    }
    j
}

/// Local system and in-plane coordinates relative to the center.
fn frame(ctx: &ElemCtx) -> Result<([[f64; 3]; 3], [[f64; 2]; 4])> {
    let xg = &ctx.x;
    let nv = cross(sub(xg[2], xg[0]), sub(xg[3], xg[1]));
    if !(norm(nv) > 0.0) {
        return fail(format!("element {} (S4R5) has zero area", ctx.label));
    }
    let r = default_axes(nv);
    let c = common::centroid(xg);
    let mut x = [[0.0; 2]; 4];
    for i in 0..4 {
        let d = sub(xg[i], c);
        x[i] = [dot(r[0], d), dot(r[1], d)];
    }
    Ok((r, x))
}

struct Ops {
    a: f64,
    bm: Vec<[f64; NDOF]>,
    bb: Vec<[f64; NDOF]>,
    /// The shear vectors and weights of the edges (force per strain).
    sh: Vec<([f64; NDOF], f64)>,
    /// Hourglass vectors with stiffnesses.
    hg: Vec<([f64; NDOF], f64)>,
}

fn ops(x: &[[f64; 2]; 4], sec: &super::Sec, label: i64) -> Result<Ops> {
    let (_, d0) = shape4(0.0, 0.0);
    let j0 = jac(x, &d0);
    let det0 = j0[0][0] * j0[1][1] - j0[0][1] * j0[1][0];
    if !(det0 > 0.0) {
        return fail(format!("element {label} (S4R5): {}", common::DISTORTED));
    }
    let ji = [[j0[1][1] / det0, -j0[0][1] / det0], [-j0[1][0] / det0, j0[0][0] / det0]];
    let mut b = [[0.0; 4]; 2];
    for i in 0..4 {
        for a in 0..2 {
            b[a][i] = ji[a][0] * d0[0][i] + ji[a][1] * d0[1][i];
        }
    }
    let area = 4.0 * det0;
    let mut bm = vec![[0.0; NDOF]; 3];
    let mut bb = vec![[0.0; NDOF]; 3];
    for i in 0..4 {
        bm[0][ND * i] = b[0][i];
        bm[1][ND * i + 1] = b[1][i];
        bm[2][ND * i] = b[1][i];
        bm[2][ND * i + 1] = b[0][i];
        bb[0][ND * i + 4] = b[0][i];
        bb[1][ND * i + 3] = -b[1][i];
        bb[2][ND * i + 4] = b[1][i];
        bb[2][ND * i + 3] = -b[0][i];
    }
    let h = [1.0, -1.0, 1.0, -1.0];
    let hx: f64 = (0..4).map(|i| h[i] * x[i][0]).sum();
    let hy: f64 = (0..4).map(|i| h[i] * x[i][1]).sum();
    let gam: Vec<f64> = (0..4).map(|i| (h[i] - hx * b[0][i] - hy * b[1][i]) / 4.0).collect();
    let sb: f64 = (0..4).map(|i| b[0][i] * b[0][i] + b[1][i] * b[1][i]).sum();
    let g = shear_modulus(sec);
    let cm = 0.08 * g * sec.t * area * sb / 16.0;
    let cb = 0.72 * g * sec.t.powi(3) / 12.0 * area * sb / 16.0;
    let mut hg = Vec::new();
    for (dof, c) in [(0, cm), (1, cm), (3, cb), (4, cb)] {
        let mut v = [0.0; NDOF];
        for i in 0..4 {
            v[ND * i + dof] = gam[i];
        }
        hg.push((v, c));
    }
    let ks = k_shear(sec) * fp(area / 2.0, sec.t);
    let mut sh = Vec::new();
    for comp in 0..2 {
        for e in [-1.0, 1.0] {
            let (s, t) = if comp == 0 { (0.0, e) } else { (e, 0.0) };
            let (n, d) = shape4(s, t);
            let j = jac(x, &d);
            let l = (j[comp][0] * j[comp][0] + j[comp][1] * j[comp][1]).sqrt();
            let mut v = [0.0; NDOF];
            for i in 0..4 {
                v[ND * i + 2] = d[comp][i] / l;
                v[ND * i + 4] += n[i] * j[comp][0] / l;
                v[ND * i + 3] -= n[i] * j[comp][1] / l;
            }
            sh.push((v, ks * det0 * 2.0));
        }
    }
    Ok(Ops { a: area, bm, bb, sh, hg })
}

impl S4r5 {
    fn local(&self, ctx: &ElemCtx) -> Result<([[f64; 3]; 3], [[f64; 2]; 4], super::Sec)> {
        let (r, x) = frame(ctx)?;
        Ok((r, x, section(ctx, "S4R5", false)?))
    }

    fn local_k(o: &Ops, sec: &super::Sec) -> Vec<f64> {
        let n = NDOF;
        let mut k = vec![0.0; n * n];
        let tb = sec.t.powi(3) / 12.0;
        for p in 0..n {
            for q in 0..n {
                let mut v = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        let dij = sec.d[i * 3 + j];
                        v += dij * (sec.t * o.bm[i][p] * o.bm[j][q] + tb * o.bb[i][p] * o.bb[j][q]);
                    }
                }
                k[p * n + q] = v * o.a;
            }
        }
        for (g, c) in o.hg.iter().map(|(g, c)| (&g[..], *c)).chain(o.sh.iter().map(|(g, c)| (&g[..], *c))) {
            super::super::shell2::add_outer(&mut k, n, g, c);
        }
        k
    }
}

impl ElementType for S4r5 {
    fn name(&self) -> &'static str {
        "S4R5"
    }
    fn family(&self) -> Family {
        Family::Shell
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        4
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        9
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2, 3]]
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (r, x, sec) = self.local(ctx)?;
        let o = ops(&x, &sec, ctx.label)?;
        let kg = common::congruence(&Self::local_k(&o, &sec), &common::block_diag(&r, 8), NDOF);
        let nrm = [r[2]; 4];
        let mut k = project_tangent(&kg, &nrm);
        add_drill_eps(&mut k, &nrm);
        Ok(k)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometric nonlinearity (NLGEOM)", "S4R5");
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (r, x, sec) = self.local(ctx)?;
        let o = ops(&x, &sec, ctx.label)?;
        let m = sec.rho * sec.t * o.a;
        let mr = sec.rho * sec.t.powi(3) / 12.0 * o.a;
        let n = NDOF;
        let mut ml = vec![0.0; n * n];
        let w = [4.0 / 36.0, 2.0 / 36.0, 1.0 / 36.0, 2.0 / 36.0];
        let ip = [7.0, -1.0, -5.0, -1.0];
        let rot = [17.0 / 48.0, 1.0 / 16.0, 1.0 / 48.0, 1.0 / 16.0];
        for i in 0..4 {
            for j in 0..4 {
                let k = (j + 4 - i) % 4;
                let (lu, lw, lr) = if kind == MassKind::Lumped {
                    if i != j {
                        continue;
                    }
                    (m / 4.0, m / 4.0, mr / 2.0)
                } else {
                    (m / 16.0 + m * ip[k] / 144000.0, m * w[k], mr * rot[k])
                };
                for d in 0..2 {
                    ml[(ND * i + d) * n + ND * j + d] = lu;
                    ml[(ND * i + 3 + d) * n + ND * j + 3 + d] = lr;
                }
                ml[(ND * i + 2) * n + ND * j + 2] = lw;
            }
        }
        Ok(common::congruence(&ml, &common::block_diag(&r, 8), n))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (r, x, sec) = self.local(ctx)?;
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, "S4R5")?;
        let q = match (load.kind.as_str(), body) {
            ("P", _) => scale(r[2], load.magnitude),
            (_, Some(b)) => scale(b, sec.t),
            _ => return not_implemented(&format!("load type {}", load.kind), "S4R5"),
        };
        let mut f = vec![0.0; NDOF];
        for &(s, t) in &[(-common::G2, -common::G2), (common::G2, -common::G2), (-common::G2, common::G2), (common::G2, common::G2)] {
            let (nn, d) = shape4(s, t);
            let j = jac(&x, &d);
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            for i in 0..4 {
                for c in 0..3 {
                    f[ND * i + c] += nn[i] * det * q[c];
                }
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (_, x, sec) = self.local(ctx)?;
        let mut v = vec![0.0; 4];
        for &(s, t) in &[(-common::G2, -common::G2), (common::G2, -common::G2), (-common::G2, common::G2), (common::G2, common::G2)] {
            let (nn, d) = shape4(s, t);
            let j = jac(&x, &d);
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            for i in 0..4 {
                v[i] += nn[i] * det * sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let (r, x, sec) = self.local(ctx)?;
        let o = ops(&x, &sec, ctx.label)?;
        let t = common::block_diag(&r, 8);
        let ul: Vec<f64> = (0..NDOF).map(|p| (0..NDOF).map(|q| t[p * NDOF + q] * u[q]).sum()).collect();
        let eps = [0, 1, 2].map(|i| super::row_dot(&o.bm[i], &ul));
        let kap = [0, 1, 2].map(|i| super::row_dot(&o.bb[i], &ul));
        // no transverse shear force is reported (SF4 = SF5 = 0) for thin shells
        Ok(output(&sec, &[Gen { eps, kap, q: [0.0; 2] }]))
    }
}
