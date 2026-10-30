//! SAXA1N and SAXA2N (N = 1..4): axisymmetric shells with asymmetric
//! deformation (area elements3).
//!
//! Nodes and degrees of freedom (U1, U2, UR3 at each node):
//! - planes θ_k = kπ/N, k = 0..N: the element's nodes (k = 0) and generated nodes
//!   with the number node + 100000·k (created by `setup` when the file has been read, shared
//!   between elements) carry the even components u_r, u_z and β (the rotation in
//!   the r–z plane, like SAX1) as values in the plane; between the planes they are interpolated with
//!   cos(mθ), m = 0..N.
//! - odd nodes (one per meridian node and mode m = 1..N;
//!   node + 10⁷·m, shared between the elements) carry the odd components u_θ,
//!   ω_r, ω_z as coefficients of sin(mθ). Boundary conditions on the element's node
//!   apply only to the plane θ = 0.
//!
//! The plane nodes are already created by *ELEMENT (`create_plane_nodes`) so that
//! *BOUNDARY and *CLOAD can reference them; `setup` adds them to the elements when
//! the file has been read. Only linear elasticity without NLGEOM; no element output.
//!
//! Formulation: three-dimensional shell kinematics in (s, θ) with local axes
//! e1 = the meridian tangent, e2 = e_θ, n = e1 × e2:
//! - membrane ε = (e1·u,s, e2·u,θ/r, e1·u,θ/r + e2·u,s) and bending from
//!   the change of the normal δn = ω × n (like SAX1), 2κ12 additionally gets −t_z ω_n/r;
//!   integrated exactly around the revolution and at one (SAXA1) or two (SAXA2)
//!   Gauss points along the meridian, respectively,
//! - transverse shear γ1 = n·u,s + δn·e1 at the meridian Gauss points and the planes
//!   θ = kπ/N with the weight 2π/(2N+1), γ2 = n·u,θ/r + δn·e2 at the meridian nodes
//!   (equal weight per node) and θ = jπ/(N+1), j = 0..2N+1, with the weight π/(2N+1)
//!   (the radius in the weight is the element's mid radius); the stiffness 5/6·G·t·f(a) with
//!   a = r·L·(2π/3) and r·L·π/(2N+1), respectively,
//! - the rotation about the normal: the penalty G·t/40·(ω·n + e1·u,θ/r)² at
//!   the meridian nodes and the same θ points as γ2 with the weight π/3,
//! - mass: diagonal along the meridian (half the element per node, SAXA1) or
//!   consistent (SAXA2), exact around the revolution, no rotary inertia.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, MassKind};
use super::super::common::{self, cross, dot, scale};
use super::super::shell2::Sec;
use super::super::shell3::{fp, k_shear, section, shear_modulus};
use crate::fem::dof::{Dof, U1, U2, UR3};
use crate::fem::model::{DistLoad, FeModel};
use crate::model::{fail, Result};
use std::f64::consts::PI;

pub struct Saxa {
    pub name: &'static str,
    /// Nodes along the meridian (2 or 3).
    pub nn: usize,
    /// Number of Fourier modes N.
    pub modes: usize,
}

pub static SAXA11: Saxa = Saxa { name: "SAXA11", nn: 2, modes: 1 };
pub static SAXA12: Saxa = Saxa { name: "SAXA12", nn: 2, modes: 2 };
pub static SAXA13: Saxa = Saxa { name: "SAXA13", nn: 2, modes: 3 };
pub static SAXA14: Saxa = Saxa { name: "SAXA14", nn: 2, modes: 4 };
pub static SAXA21: Saxa = Saxa { name: "SAXA21", nn: 3, modes: 1 };
pub static SAXA22: Saxa = Saxa { name: "SAXA22", nn: 3, modes: 2 };
pub static SAXA23: Saxa = Saxa { name: "SAXA23", nn: 3, modes: 3 };
pub static SAXA24: Saxa = Saxa { name: "SAXA24", nn: 3, modes: 4 };

pub static ALL: [&Saxa; 8] = [&SAXA11, &SAXA12, &SAXA13, &SAXA14, &SAXA21, &SAXA22, &SAXA23, &SAXA24];

/// The node number step for the generated planes.
pub const PLANE_OFFSET: i64 = 100000;
/// The node number step for the nodes of the odd components (one per meridian node and
/// mode; node + 10⁷·m).
pub const ODD_OFFSET: i64 = 10_000_000;

type V3 = [f64; 3];

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The meridian shape functions (node order 1, 2 (middle), 3).
fn shape(nn: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    super::super::shell3::axi::shape(nn, xi)
}

/// Inverted 3×3/n×n matrix (Gauss–Jordan) for the plane interpolation.
fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs())).unwrap();
        for k in 0..n {
            m.swap(c * n + k, p * n + k);
            inv.swap(c * n + k, p * n + k);
        }
        let d = m[c * n + c];
        for k in 0..n {
            m[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for i in 0..n {
            if i != c {
                let f = m[i * n + c];
                if f != 0.0 {
                    for k in 0..n {
                        m[i * n + k] -= f * m[c * n + k];
                        inv[i * n + k] -= f * inv[c * n + k];
                    }
                }
            }
        }
    }
    inv
}

/// Point data along the meridian.
struct Mer {
    n: Vec<f64>,
    ns: Vec<f64>,
    r: f64,
    /// ds/dξ
    j: f64,
    t: [f64; 2],
    /// dn/ds in (r, z)
    dn: [f64; 2],
}

fn mer(x: &[[f64; 3]], nn: usize, xi: f64) -> Mer {
    let tang = |xi: f64| -> ([f64; 2], f64) {
        let (_, d) = shape(nn, xi);
        let a = [(0..nn).map(|i| d[i] * x[i][0]).sum::<f64>(), (0..nn).map(|i| d[i] * x[i][1]).sum::<f64>()];
        let j = (a[0] * a[0] + a[1] * a[1]).sqrt();
        ([a[0] / j, a[1] / j], j)
    };
    let (n, d) = shape(nn, xi);
    let (t, j) = tang(xi);
    let r = (0..nn).map(|i| n[i] * x[i][0]).sum();
    let dn = if nn == 2 {
        [0.0, 0.0]
    } else {
        let h = 1e-6;
        let (tp, _) = tang(xi + h);
        let (tm, _) = tang(xi - h);
        // n = (−t_z, t_r)
        [-(tp[1] - tm[1]) / (2.0 * h * j), (tp[0] - tm[0]) / (2.0 * h * j)]
    };
    Mer { ns: d.iter().map(|v| v / j).collect(), n, r, j, t, dn }
}

/// Fields (vector per degree of freedom) and their derivatives.
struct Fields {
    u: Vec<V3>,
    us: Vec<V3>,
    ut: Vec<V3>,
    w: Vec<V3>,
    ws: Vec<V3>,
    wt: Vec<V3>,
}

/// Strains at a point (rows over the element's degrees of freedom).
struct Ops {
    m: [Vec<f64>; 3],
    b: [Vec<f64>; 3],
    g1: Vec<f64>,
    g2: Vec<f64>,
    drill: Vec<f64>,
}

impl Saxa {
    fn nodes_total(&self) -> usize {
        self.nn * (2 * self.modes + 1)
    }
    fn nd(&self) -> usize {
        3 * self.nodes_total()
    }
    /// Local node for meridian node a in plane k (even components).
    fn even_node(&self, a: usize, k: usize) -> usize {
        if k == 0 {
            a
        } else {
            self.nn + self.nn * self.modes + (k - 1) * self.nn + a
        }
    }
    /// Local node for meridian node a and mode m = 1..N (odd components).
    fn odd_node(&self, a: usize, m: usize) -> usize {
        self.nn + (m - 1) * self.nn + a
    }

    /// Weights for the plane values: f(θ) = Σ_k c_k(θ) f_k and the derivative.
    fn even_weights(&self, th: f64) -> (Vec<f64>, Vec<f64>) {
        let n = self.modes;
        let np = n + 1;
        let mut a = vec![0.0; np * np];
        for k in 0..np {
            let thk = k as f64 * PI / n as f64;
            for m in 0..np {
                a[k * np + m] = (m as f64 * thk).cos();
            }
        }
        let ai = invert(&a, np);
        let c: Vec<f64> = (0..np).map(|m| (m as f64 * th).cos()).collect();
        let dc: Vec<f64> = (0..np).map(|m| -(m as f64) * (m as f64 * th).sin()).collect();
        let w = (0..np).map(|k| (0..np).map(|m| c[m] * ai[m * np + k]).sum()).collect();
        let dw = (0..np).map(|k| (0..np).map(|m| dc[m] * ai[m * np + k]).sum()).collect();
        (w, dw)
    }

    fn fields(&self, me: &Mer, nsv: &[f64], th: f64) -> Fields {
        let nd = self.nd();
        let z = [0.0; 3];
        let mut f = Fields { u: vec![z; nd], us: vec![z; nd], ut: vec![z; nd], w: vec![z; nd], ws: vec![z; nd], wt: vec![z; nd] };
        let er = [th.cos(), th.sin(), 0.0];
        let et = [-th.sin(), th.cos(), 0.0];
        let ez = [0.0, 0.0, 1.0];
        let met = scale(et, -1.0);
        let mer_ = scale(er, -1.0);
        let (we, wed) = self.even_weights(th);
        for a in 0..self.nn {
            let (na, nsa) = (nsv[a], me.ns[a]);
            for k in 0..=self.modes {
                let node = self.even_node(a, k);
                for (c, base, dbase, rot) in [(0, er, et, false), (1, ez, z, false), (2, met, er, true)] {
                    let j = 3 * node + c;
                    let v = scale(base, na * we[k]);
                    let vs = scale(base, nsa * we[k]);
                    let vt = add3(scale(base, na * wed[k]), scale(dbase, na * we[k]));
                    if rot {
                        f.w[j] = v;
                        f.ws[j] = vs;
                        f.wt[j] = vt;
                    } else {
                        f.u[j] = v;
                        f.us[j] = vs;
                        f.ut[j] = vt;
                    }
                }
            }
            for m in 1..=self.modes {
                let node = self.odd_node(a, m);
                let (s, ds) = ((m as f64 * th).sin(), m as f64 * (m as f64 * th).cos());
                for (c, base, dbase, rot) in [(0, et, mer_, false), (1, er, et, true), (2, ez, z, true)] {
                    let j = 3 * node + c;
                    let v = scale(base, na * s);
                    let vs = scale(base, nsa * s);
                    let vt = add3(scale(base, na * ds), scale(dbase, na * s));
                    if rot {
                        f.w[j] = v;
                        f.ws[j] = vs;
                        f.wt[j] = vt;
                    } else {
                        f.u[j] = v;
                        f.us[j] = vs;
                        f.ut[j] = vt;
                    }
                }
            }
        }
        f
    }

    /// Operators at the meridian point `me` (shape function values `nsv`) and θ.
    fn ops(&self, me: &Mer, nsv: &[f64], th: f64) -> Ops {
        let nd = self.nd();
        let f = self.fields(me, nsv, th);
        let (tr, tz) = (me.t[0], me.t[1]);
        let er = [th.cos(), th.sin(), 0.0];
        let et = [-th.sin(), th.cos(), 0.0];
        let ez = [0.0, 0.0, 1.0];
        let e1 = add3(scale(er, tr), scale(ez, tz));
        let e2 = et;
        let nv = add3(scale(er, -tz), scale(ez, tr));
        let ns = add3(scale(er, me.dn[0]), scale(ez, me.dn[1]));
        let nt = scale(et, -tz);
        let r = me.r;
        let mut o = Ops { m: [vec![0.0; nd], vec![0.0; nd], vec![0.0; nd]], b: [vec![0.0; nd], vec![0.0; nd], vec![0.0; nd]], g1: vec![0.0; nd], g2: vec![0.0; nd], drill: vec![0.0; nd] };
        for j in 0..nd {
            let dn = cross(f.w[j], nv);
            let dns = add3(cross(f.ws[j], nv), cross(f.w[j], ns));
            let dnt = add3(cross(f.wt[j], nv), cross(f.w[j], nt));
            let wn = dot(nv, f.w[j]);
            o.m[0][j] = dot(e1, f.us[j]);
            o.m[1][j] = dot(e2, f.ut[j]) / r;
            o.m[2][j] = dot(e1, f.ut[j]) / r + dot(e2, f.us[j]);
            o.b[0][j] = dot(e1, dns);
            o.b[1][j] = dot(e2, dnt) / r;
            o.b[2][j] = dot(e1, dnt) / r + dot(e2, dns) - tz * wn / r;
            o.g1[j] = dot(nv, f.us[j]) + dot(e1, dn);
            o.g2[j] = dot(nv, f.ut[j]) / r + dot(e2, dn);
            o.drill[j] = wn + dot(e1, f.ut[j]) / r;
        }
        o
    }

    fn gauss_s(&self) -> Vec<(f64, f64)> {
        super::super::shell3::axi::gauss(self.nn - 1)
    }

    fn check(&self, ctx: &ElemCtx) -> Result<()> {
        if ctx.x.len() != self.nodes_total() {
            return fail(format!("element {} ({}) lacks the generated nodes (has {} nodes)", ctx.label, self.name, ctx.x.len()));
        }
        for (xi, _) in super::super::shell3::axi::gauss(3) {
            let p = mer(&ctx.x, self.nn, xi);
            if !(p.j > 0.0) || !(p.r > 0.0) {
                return fail(format!("element {} ({}): the radius must be positive and the element must have a length", ctx.label, self.name));
            }
        }
        Ok(())
    }

    pub fn stiffness_of(&self, ctx: &ElemCtx, sec: &Sec) -> Vec<f64> {
        let nd = self.nd();
        let nm = self.modes;
        let mut k = vec![0.0; nd * nd];
        let x = &ctx.x;
        let tb = sec.t.powi(3) / 12.0;
        let (ks, g, t) = (k_shear(sec), shear_modulus(sec), sec.t);
        let add = |k: &mut Vec<f64>, a: &[f64], b: &[f64], c: f64| {
            for p in 0..nd {
                if a[p] != 0.0 {
                    let ap = a[p] * c;
                    for q in 0..nd {
                        k[p * nd + q] += ap * b[q];
                    }
                }
            }
        };
        let nth = 4 * nm + 4;
        let mut len = 0.0;
        for (xi, ws) in self.gauss_s() {
            let me = mer(x, self.nn, xi);
            len += me.j * ws;
            let nsv = me.n.clone();
            for q in 0..nth {
                let th = 2.0 * PI * q as f64 / nth as f64;
                let o = self.ops(&me, &nsv, th);
                let w = me.r * me.j * ws * 2.0 * PI / nth as f64;
                for i in 0..3 {
                    for jx in 0..3 {
                        let dm = sec.d[i * 3 + jx];
                        if dm != 0.0 {
                            add(&mut k, &o.m[i], &o.m[jx], dm * t * w);
                            add(&mut k, &o.b[i], &o.b[jx], dm * tb * w);
                        }
                    }
                }
            }
            // γ1 in the planes θ = kπ/N
            let w1 = 2.0 * PI / (2 * nm + 1) as f64;
            for kk in 0..2 * nm {
                let o = self.ops(&me, &nsv, kk as f64 * PI / nm as f64);
                let a = me.r * me.j * ws;
                add(&mut k, &o.g1, &o.g1, ks * fp(a * 2.0 * PI / 3.0, t) * w1 * a);
            }
        }
        // γ2 and the rotation about the normal at the meridian nodes
        let rmid = mer(x, self.nn, 0.0).r;
        let w2 = PI / (2 * nm + 1) as f64;
        let nodes_xi: Vec<f64> = if self.nn == 2 { vec![-1.0, 1.0] } else { vec![-1.0, 0.0, 1.0] };
        let node_w: Vec<f64> = if self.nn == 2 { vec![0.5, 0.5] } else { vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0] };
        for (a, &xi) in nodes_xi.iter().enumerate() {
            let me = mer(x, self.nn, xi);
            let mut nsv = vec![0.0; self.nn];
            nsv[a] = 1.0;
            let wl = rmid * len * node_w[a];
            for jj in 0..2 * nm + 2 {
                let o = self.ops(&me, &nsv, jj as f64 * PI / (nm + 1) as f64);
                add(&mut k, &o.g2, &o.g2, ks * fp(rmid * len * w2, t) * w2 * wl);
                add(&mut k, &o.drill, &o.drill, g * t / 40.0 * (PI / 3.0) * wl);
            }
        }
        k
    }

    /// Mass per meridian node (lumped, SAXA1) or consistent (SAXA2) times
    /// the exact θ integral.
    fn mass_of(&self, ctx: &ElemCtx, sec: &Sec) -> Vec<f64> {
        let nd = self.nd();
        let nm = self.modes;
        let mut m = vec![0.0; nd * nd];
        let rho_t = sec.rho * sec.t;
        // meridian matrix S[a][b] = ∫ N_a N_b r ds
        let mut sm = vec![0.0; self.nn * self.nn];
        if self.nn == 2 {
            let me = mer(&ctx.x, 2, 0.0);
            for a in 0..2 {
                sm[a * 2 + a] = me.r * me.j * 2.0 / 2.0;
            }
        } else {
            for (xi, w) in super::super::shell3::axi::gauss(3) {
                let me = mer(&ctx.x, 3, xi);
                for a in 0..3 {
                    for b in 0..3 {
                        sm[a * 3 + b] += me.n[a] * me.n[b] * me.r * me.j * w;
                    }
                }
            }
        }
        // θ integrals for the plane weights
        let nth = 8 * nm + 8;
        let np = nm + 1;
        let mut ce = vec![0.0; np * np];
        for q in 0..nth {
            let th = 2.0 * PI * q as f64 / nth as f64;
            let (w, _) = self.even_weights(th);
            for i in 0..np {
                for j in 0..np {
                    ce[i * np + j] += w[i] * w[j] * 2.0 * PI / nth as f64;
                }
            }
        }
        for a in 0..self.nn {
            for b in 0..self.nn {
                let s = rho_t * sm[a * self.nn + b];
                if s == 0.0 {
                    continue;
                }
                for i in 0..np {
                    for j in 0..np {
                        let (p, q) = (self.even_node(a, i), self.even_node(b, j));
                        for c in 0..2 {
                            m[(3 * p + c) * nd + 3 * q + c] += s * ce[i * np + j];
                        }
                    }
                }
                for mm in 1..=nm {
                    let (p, q) = (self.odd_node(a, mm), self.odd_node(b, mm));
                    m[(3 * p) * nd + 3 * q] += s * PI;
                }
            }
        }
        m
    }
}

impl ElementType for Saxa {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Shell
    }
    fn dim(&self) -> usize {
        2
    }
    fn n_nodes(&self) -> usize {
        self.nodes_total()
    }
    fn n_connectivity(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, UR3]
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, false)?;
        Ok(self.stiffness_of(ctx, &sec))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometric nonlinearity (NLGEOM)", self.name);
        }
        if let Some(m) = ctx.material {
            if !m.is_linear() {
                return not_implemented(&format!("nonlinear material ({})", m.name()), self.name);
            }
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, false)?;
        Ok(self.mass_of(ctx, &sec))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, false)?;
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, self.name)?;
        if load.kind != "P" && body.is_none() {
            return not_implemented(&format!("load type {}", load.kind), self.name);
        }
        // axisymmetric load: the plane weights ∫ c_k(θ) dθ
        let nm = self.modes;
        let nth = 8 * nm + 8;
        let mut ck = vec![0.0; nm + 1];
        for q in 0..nth {
            let (w, _) = self.even_weights(2.0 * PI * q as f64 / nth as f64);
            for k in 0..=nm {
                ck[k] += w[k] * 2.0 * PI / nth as f64;
            }
        }
        let mut f = vec![0.0; self.nd()];
        for (xi, w) in super::super::shell3::axi::gauss(3) {
            let p = mer(&ctx.x, self.nn, xi);
            let da = p.r * p.j * w;
            let q = match body {
                Some(b) if load.kind != "P" => [b[0] * sec.t, b[1] * sec.t],
                _ => [-load.magnitude * p.t[1], load.magnitude * p.t[0]],
            };
            for a in 0..self.nn {
                for k in 0..=nm {
                    let node = self.even_node(a, k);
                    for c in 0..2 {
                        f[3 * node + c] += p.n[a] * da * q[c] * ck[k];
                    }
                }
            }
        }
        Ok(f)
    }
}

/// Creates the plane nodes (node + 100000·k) and the odd component nodes
/// (node + 10⁷·m) with the same position as the node for the SAXA elements `ids`
/// (called by *ELEMENT, so that *BOUNDARY and *CLOAD can reference them).
pub fn create_plane_nodes(m: &mut FeModel, ids: &[usize]) {
    for &e in ids {
        let name = m.elements[e].etype.name();
        let Some(sx) = ALL.iter().find(|s| s.name == name) else { return };
        let inst = m.elements[e].inst;
        for i in 0..sx.nn {
            let n = m.elements[e].nodes[i];
            for k in 1..=sx.modes {
                for off in [PLANE_OFFSET, ODD_OFFSET] {
                    let label = m.node_label[n] + off * k as i64;
                    if m.node(inst, label).is_none() {
                        let x = m.nodes[n];
                        m.add_node(inst, label, x);
                    }
                }
            }
        }
    }
}

/// Appends the odd nodes and the plane nodes to the node lists of the SAXA elements
/// (boundary conditions on the element nodes apply only to the plane
/// θ = 0, not to the odd components). Called when the file has been read (after `FeModel::finalize`).
pub fn setup(m: &mut FeModel) -> Result<()> {
    for e in 0..m.elements.len() {
        let name = m.elements[e].etype.name();
        let Some(sx) = ALL.iter().find(|s| s.name == name) else { continue };
        if m.elements[e].nodes.len() != sx.nn {
            continue;
        }
        let inst = m.elements[e].inst;
        let base: Vec<usize> = m.elements[e].nodes[..sx.nn].to_vec();
        let mut extra = Vec::new();
        for off in [ODD_OFFSET, PLANE_OFFSET] {
            for k in 1..=sx.modes {
                for &n in &base {
                    let label = m.node_label[n] + off * k as i64;
                    let id = match m.node(inst, label) {
                        Some(id) => id,
                        None => {
                            let x = m.nodes[n];
                            m.add_node(inst, label, x)
                        }
                    };
                    extra.push(id);
                }
            }
        }
        m.elements[e].nodes.extend(extra);
    }
    Ok(())
}
