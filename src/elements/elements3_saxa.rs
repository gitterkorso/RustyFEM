//! SAXA1N och SAXA2N (N = 1..4): axisymmetriska skal med asymmetrisk
//! deformation (område elements3).
//!
//! Noder och frihetsgrader (U1, U2, UR3 i varje nod):
//! - plan θ_k = kπ/N, k = 0..N: elementets noder (k = 0) och genererade noder
//!   med numret nod + 100000·k (skapas av `setup` när filen är läst, delas
//!   mellan element) bär de jämna komponenterna u_r, u_z och β (rotationen i
//!   r–z-planet, som SAX1) som värden i planet; mellan planen interpoleras med
//!   cos(mθ), m = 0..N.
//! - udda noder (en per meridiannod och mod m = 1..N;
//!   nod + 10⁷·m, delade mellan elementen) bär de udda komponenterna u_θ,
//!   ω_r, ω_z som koefficienter för sin(mθ). Randvillkor på elementets nod
//!   gäller bara planet θ = 0.
//!
//! Planens noder skapas redan av *ELEMENT (`create_plane_nodes`) så att
//! *BOUNDARY och *CLOAD kan ange dem; `setup` lägger dem i elementen när
//! filen är läst. Bara linjär elasticitet utan NLGEOM; ingen elementutdata.
//!
//! Formulering: tredimensionell skalkinematik i (s, θ) med lokala axlar
//! e1 = meridianens tangent, e2 = e_θ, n = e1 × e2:
//! - membran ε = (e1·u,s, e2·u,θ/r, e1·u,θ/r + e2·u,s) och böjning ur
//!   normalens ändring δn = ω × n (som SAX1), 2κ12 får dessutom −t_z ω_n/r;
//!   integreras exakt runt varvet och i en (SAXA1) respektive två (SAXA2)
//!   gausspunkter längs meridianen,
//! - tvärskjuvning γ1 = n·u,s + δn·e1 i meridianens gausspunkter och planen
//!   θ = kπ/N med vikten 2π/(2N+1), γ2 = n·u,θ/r + δn·e2 i meridiannoderna
//!   (lika vikt per nod) och θ = jπ/(N+1), j = 0..2N+1, med vikten π/(2N+1)
//!   (radien i vikten är elementets mittradie); styvheten 5/6·G·t·f(a) med
//!   a = r·L·(2π/3) respektive r·L·π/(2N+1),
//! - rotationen kring normalen: straffet G·t/40·(ω·n + e1·u,θ/r)² i
//!   meridiannoderna och samma θ-punkter som γ2 med vikten π/3,
//! - massa: diagonal längs meridianen (halva elementet per nod, SAXA1) eller
//!   konsistent (SAXA2), exakt runt varvet, ingen rotationströghet.

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
    /// Noder längs meridianen (2 eller 3).
    pub nn: usize,
    /// Antal Fouriermoder N.
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

/// Nodnummersteget för de genererade planen.
pub const PLANE_OFFSET: i64 = 100000;
/// Nodnummersteget för de udda komponenternas noder (en per meridiannod och
/// mod; nod + 10⁷·m).
pub const ODD_OFFSET: i64 = 10_000_000;

type V3 = [f64; 3];

fn add3(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Meridianens formfunktioner (nodordning 1, 2 (mitt), 3).
fn shape(nn: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    super::super::shell3::axi::shape(nn, xi)
}

/// Inverterad 3×3/n×n-matris (Gauss–Jordan) för planinterpolationen.
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

/// Punktdata längs meridianen.
struct Mer {
    n: Vec<f64>,
    ns: Vec<f64>,
    r: f64,
    /// ds/dξ
    j: f64,
    t: [f64; 2],
    /// dn/ds i (r, z)
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

/// Fält (vektor per frihetsgrad) och deras derivator.
struct Fields {
    u: Vec<V3>,
    us: Vec<V3>,
    ut: Vec<V3>,
    w: Vec<V3>,
    ws: Vec<V3>,
    wt: Vec<V3>,
}

/// Töjningar i en punkt (rader över elementets frihetsgrader).
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
    /// Lokal nod för meridiannod a i planet k (jämna komponenter).
    fn even_node(&self, a: usize, k: usize) -> usize {
        if k == 0 {
            a
        } else {
            self.nn + self.nn * self.modes + (k - 1) * self.nn + a
        }
    }
    /// Lokal nod för meridiannod a och mod m = 1..N (udda komponenter).
    fn odd_node(&self, a: usize, m: usize) -> usize {
        self.nn + (m - 1) * self.nn + a
    }

    /// Vikter för planvärdena: f(θ) = Σ_k c_k(θ) f_k och derivatan.
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

    /// Operatorer i meridianpunkten `me` (formfunktionsvärden `nsv`) och θ.
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
            return fail(format!("element {} ({}) saknar de genererade noderna (har {} noder)", ctx.label, self.name, ctx.x.len()));
        }
        for (xi, _) in super::super::shell3::axi::gauss(3) {
            let p = mer(&ctx.x, self.nn, xi);
            if !(p.j > 0.0) || !(p.r > 0.0) {
                return fail(format!("element {} ({}): radien måste vara positiv och elementet ha längd", ctx.label, self.name));
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
            // γ1 i planen θ = kπ/N
            let w1 = 2.0 * PI / (2 * nm + 1) as f64;
            for kk in 0..2 * nm {
                let o = self.ops(&me, &nsv, kk as f64 * PI / nm as f64);
                let a = me.r * me.j * ws;
                add(&mut k, &o.g1, &o.g1, ks * fp(a * 2.0 * PI / 3.0, t) * w1 * a);
            }
        }
        // γ2 och rotationen kring normalen i meridiannoderna
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

    /// Massa per meridiannod (lumpad, SAXA1) eller konsistent (SAXA2) gånger
    /// den exakta θ-integralen.
    fn mass_of(&self, ctx: &ElemCtx, sec: &Sec) -> Vec<f64> {
        let nd = self.nd();
        let nm = self.modes;
        let mut m = vec![0.0; nd * nd];
        let rho_t = sec.rho * sec.t;
        // meridianmatris S[a][b] = ∫ N_a N_b r ds
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
        // θ-integraler för planvikterna
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
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        if let Some(m) = ctx.material {
            if !m.is_linear() {
                return not_implemented(&format!("olinjärt material ({})", m.name()), self.name);
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
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        }
        // axisymmetrisk last: planvikterna ∫ c_k(θ) dθ
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

/// Skapar planens noder (nod + 100000·k) och de udda komponenternas noder
/// (nod + 10⁷·m) med samma läge som noden för SAXA-elementen `ids`
/// (anropas av *ELEMENT, så att *BOUNDARY och *CLOAD kan ange dem).
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

/// Lägger de udda noderna och planens noder sist i SAXA-elementens
/// nodlistor (randvillkor på elementens noder gäller bara planet
/// θ = 0, inte de udda komponenterna). Anropas när filen är läst (efter `FeModel::finalize`).
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
