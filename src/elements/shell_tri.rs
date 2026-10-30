//! Triangular shells: helper functions (constant strain) and S3/S3R.
//!
//! S3 and S3R are the same element. Formulation in the element's
//! local plane (see `shell.rs` for the axes):
//!
//! - Membrane: constant strain (CST).
//! - Bending: constant curvature from linearly interpolated rotations.
//! - Transverse shear: the tangential shear strains of the edges
//!   sₖ = (w_j − w_i)/Lₖ + tₖ·β̄ₖ (β = (θy, −θx), mean value on the edge).
//!   Each corner v gives a constant shear field γᵥ that satisfies the two edges
//!   meeting at v exactly. The element's constant field is γ0 = Σ wᵥ γᵥ with the weights
//!   wᵥ ∝ (L_a² − L_b²)² (L_a, L_b = the edges at v), normalized to sum 1;
//!   for an equilateral triangle (all weights zero) 1/3 is used. The part of
//!   the edge strains that a constant field cannot describe, ρ = Σ Lₖ sₖ
//!   (the circulation around the element), is stabilized with the stiffness
//!   ks·(L₁₂² + L₃₁²)/(48000·A).
//! - Drill: penalty on gᵢ = θzᵢ − ω (ω = ½(v,x − u,y), constant) with the weight
//!   ½·A at nodes 2 and 3 and a mean-value term A/320·(2g₁ + g₂ + g₃)²,
//!   multiplied by the same kfac as for S4R.
//!
//! The mass is consistent (ρt and ρt³/12
//! for θx, θy; no mass for θz).

use crate::elements::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::elements::common;
use super::{add_btcb, add_outer, area_load_vector, bbend, bmem, local_frame, section, section_c, shell_output, stack, to_global, vec_to_global, Local, ShellSec, ND};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

/// Linear triangle in local coordinates: dN/dx, dN/dy and the area.
pub fn cst(x: &[[f64; 2]]) -> ([f64; 3], [f64; 3], f64) {
    let (x1, y1, x2, y2, x3, y3) = (x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1]);
    let det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
    let bx = [(y2 - y3) / det, (y3 - y1) / det, (y1 - y2) / det];
    let by = [(x3 - x2) / det, (x1 - x3) / det, (x2 - x1) / det];
    (bx, by, 0.5 * det)
}

pub struct ShellTri {
    name: &'static str,
}

pub static S3: ShellTri = ShellTri { name: "S3" };
pub static S3R: ShellTri = ShellTri { name: "S3R" };

/// The constant transverse shear field B0 (2 × 18), the circulation ρ (18) and the edge lengths.
fn tri_shear(x: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>, [f64; 3]) {
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
    // corner v: the edges (a, b) that meet at v
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
        // [t_a; t_b] γ = [s_a; s_b]
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

impl ShellTri {
    fn setup(&self, ctx: &ElemCtx) -> Result<(Local, ShellSec)> {
        let loc = local_frame(&ctx.x, ctx.label, self.name)?;
        let (_, _, a) = cst(&loc.xl);
        if a <= 0.0 {
            return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
        }
        Ok((loc, ShellSec::get(ctx, self.name)?))
    }

    /// Local stiffness (18 × 18) in [u, v, w, θx, θy, θz] per node.
    fn local_k(&self, x: &[[f64; 2]], sec: &ShellSec) -> Vec<f64> {
        let n = 3 * ND;
        let (bx, by, a) = cst(x);
        let b = stack(&bmem(&bx, &by), &bbend(&bx, &by));
        let ks = sec.ks(a);
        let mut k = Self::rest_k(x, sec, [ks, ks, 0.0]);
        add_btcb(&mut k, n, &b, &section_c(sec), &b, 6, a);
        k
    }

    /// Transverse shear (the stiffness [K11, K22, K12]) and drill (elastic parts of the stiffness).
    fn rest_k(x: &[[f64; 2]], sec: &ShellSec, kk: [f64; 3]) -> Vec<f64> {
        let n = 3 * ND;
        let mut k = vec![0.0; n * n];
        let (bx, by, a) = cst(x);
        // transverse shear: constant field plus stabilization of the circulation
        let ks = 0.5 * (kk[0] + kk[1]);
        let (b0, rho, l) = tri_shear(x);
        add_btcb(&mut k, n, &b0, &[kk[0], kk[2], kk[2], kk[1]], &b0, 2, a);
        add_outer(&mut k, n, &rho, ks * (l[0] * l[0] + l[2] * l[2]) / (48000.0 * a));
        // drill
        let kts = sec.tss.map(|kk| kk[0]).unwrap_or(5.0 / 6.0 * sec.g * sec.t);
        let kfac = 0.2 * kts / (1.0 + 0.1 * a / (sec.t * sec.t));
        let g: Vec<Vec<f64>> = (0..3)
            .map(|i| {
                let mut g = vec![0.0; n];
                g[i * ND + 5] = 1.0;
                for j in 0..3 {
                    g[j * ND] += 0.5 * by[j];
                    g[j * ND + 1] -= 0.5 * bx[j];
                }
                g
            })
            .collect();
        add_outer(&mut k, n, &g[1], 0.5 * kfac * a);
        add_outer(&mut k, n, &g[2], 0.5 * kfac * a);
        let gm: Vec<f64> = (0..n).map(|q| 2.0 * g[0][q] + g[1][q] + g[2][q]).collect();
        add_outer(&mut k, n, &gm, kfac * a / 320.0);
        k
    }

    /// Generalized strains [ε(3), κ(3), γ(2)] (one integration point).
    fn strains(&self, x: &[[f64; 2]], ul: &[f64]) -> [f64; 8] {
        let n = 3 * ND;
        let (bx, by, _) = cst(x);
        let mv = |b: &[f64], r: usize| -> f64 { (0..n).map(|q| b[r * n + q] * ul[q]).sum() };
        let bm = bmem(&bx, &by);
        let bb = bbend(&bx, &by);
        let (b0, _, _) = tri_shear(x);
        [mv(&bm, 0), mv(&bm, 1), mv(&bm, 2), mv(&bb, 0), mv(&bb, 1), mv(&bb, 2), mv(&b0, 0), mv(&b0, 1)]
    }

    fn local_u(loc: &Local, u: &[f64]) -> Vec<f64> {
        let t = common::block_diag(&loc.r, 6);
        common::matvec(&t, u)
    }

    /// Kinematics for integration through the thickness (area struct_nonlin).
    fn kin(&self, ctx: &ElemCtx) -> Result<section::Kin> {
        let (loc, mut sec) = self.setup(ctx)?;
        let (bx, by, a) = cst(&loc.xl);
        let kk = section::shear_stiffness(ctx, self.name, &loc, &mut sec, a)?;
        let rest = Self::rest_k(&loc.xl, &sec, kk);
        let (b0, _, _) = tri_shear(&loc.xl);
        let ips = vec![(stack(&bmem(&bx, &by), &bbend(&bx, &by)), a)];
        Ok(section::Kin { loc, sec, ips, rest, shear: vec![b0], kshear: kk, rule: super::nlgeom::MemRule::Tri })
    }
}

impl ElementType for ShellTri {
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
        3
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        5
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2]]
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            let loc = section::frame(ctx, self.name)?;
            return Ok(section::Layup::new(ctx, self.name, &loc)?.init_state(1));
        }
        Ok(Vec::new())
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            // area struct_nonlin: the elastic section stiffness from the section points
            let kin = self.kin(ctx)?;
            return kin.stiffness(&section::Layup::new(ctx, self.name, &kin.loc)?);
        }
        let (loc, sec) = self.setup(ctx)?;
        Ok(to_global(&self.local_k(&loc.xl, &sec), &loc.r, 3))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if section::is_general(ctx) {
            // area struct_nonlin: integration through the thickness (shell_section.rs)
            let kin = self.kin(ctx)?;
            let lay = section::Layup::new(ctx, self.name, &kin.loc)?;
            if ctx.nlgeom {
                return kin.response_nlgeom(ctx, self.name, &lay, u, _so, _sn, tangent);
            }
            return kin.response(ctx, &lay, u, _so, _sn, tangent);
        }
        if ctx.nlgeom {
            // area nlgeom_struct: corotational formulation (shell_nlgeom.rs)
            let k0 = self.stiffness(ctx)?;
            return super::nlgeom::response(ctx, self.name, &k0, u, super::nlgeom::MemRule::Tri, tangent);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let (_, _, a) = cst(&loc.xl);
        let n = 3 * ND;
        let mut m = vec![0.0; n * n];
        let mt = sec.rho * sec.t;
        let mr = sec.rho * sec.t.powi(3) / 12.0;
        for i in 0..3 {
            for j in 0..3 {
                let f = if kind == MassKind::Lumped {
                    if i == j { a / 3.0 } else { 0.0 }
                } else if i == j {
                    a / 6.0
                } else {
                    a / 12.0
                };
                for d in 0..3 {
                    m[(i * ND + d) * n + j * ND + d] += f * mt;
                }
                // area struct_nonlin: diagonal mass (*DYNAMIC) has
                // rotary inertia also about the normal
                let dmax = if kind == MassKind::Lumped { 6 } else { 5 };
                for d in 3..dmax {
                    m[(i * ND + d) * n + j * ND + d] += f * mr;
                }
            }
        }
        Ok(to_global(&m, &loc.r, 3))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            return not_implemented("geometric stiffness with integration through the thickness", self.name);
        }
        let (loc, sec) = self.setup(ctx)?;
        let ul = Self::local_u(&loc, u);
        let e = self.strains(&loc.xl, &ul);
        let nm: Vec<f64> = (0..3).map(|i| (0..3).map(|j| sec.d[i * 3 + j] * e[j]).sum::<f64>() * sec.t).collect();
        let (bx, by, a) = cst(&loc.xl);
        let n = 3 * ND;
        let mut k = vec![0.0; n * n];
        for i in 0..3 {
            for j in 0..3 {
                let v = (bx[i] * (nm[0] * bx[j] + nm[2] * by[j]) + by[i] * (nm[2] * bx[j] + nm[1] * by[j])) * a;
                for d in 0..3 {
                    k[(i * ND + d) * n + j * ND + d] += v;
                }
            }
        }
        Ok(to_global(&k, &loc.r, 3))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let Some(q) = area_load_vector(load, loc.r[2], &sec, self.name)? else {
            return not_implemented(&format!("load type {}", load.kind), self.name);
        };
        let (_, _, a) = cst(&loc.xl);
        let ql = [0usize, 1, 2].map(|r| common::dot(loc.r[r], q));
        let mut f = vec![0.0; 3 * ND];
        for i in 0..3 {
            for d in 0..3 {
                f[i * ND + d] = ql[d] * a / 3.0;
            }
        }
        Ok(vec_to_global(&f, &loc.r, 6))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let (_, _, a) = cst(&loc.xl);
        Ok(vec![a * sec.t / 3.0; 3])
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        if section::is_general(ctx) {
            let kin = self.kin(ctx)?;
            let lay = section::Layup::new(ctx, self.name, &kin.loc)?;
            return kin.output(ctx, self.name, &lay, u, _state);
        }
        let (loc, sec) = self.setup(ctx)?;
        let u_tot = u;
        let d;
        let u = if ctx.nlgeom {
            d = super::nlgeom::deformational_u(ctx, self.name, u)?;
            &d[..]
        } else {
            u
        };
        let ul = Self::local_u(&loc, u);
        let mut e = self.strains(&loc.xl, &ul);
        let (_, _, a) = cst(&loc.xl);
        if !ctx.nlgeom {
            return Ok(shell_output(&sec, &[e], sec.ks(a)));
        }
        // area nlgeom_struct: logarithmic membrane strain and current thickness
        let el = super::nlgeom::membrane_strains(ctx, self.name, u_tot, super::nlgeom::MemRule::Tri)?[0];
        e[..3].copy_from_slice(&el);
        let mut out = shell_output(&sec, &[e], sec.ks(a));
        super::nlgeom::scale_membrane_forces(&mut out, el);
        Ok(out)
    }
}

pub static TYPES: &[&dyn ElementType] = &[&S3, &S3R];
