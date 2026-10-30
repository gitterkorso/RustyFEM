//! S9R5 och S8R5: tunna dubbelkrökta skal med reducerad integrering och fem
//! frihetsgrader per nod (område shells3).
//!
//! S8R5 är ett S9R5 med en intern mittnod (fem
//! frihetsgrader). Här skapas den interna noden av
//! `FeModel::finalize` (`internal_nodes`) och placeras i serendipityytans
//! mittpunkt x(0, 0) = −¼Σ hörn + ½Σ mittnoder.
//!
//! Formulering (9-nods Lagrange, kinematik som S8R i `shell2_s8r.rs`):
//!
//! - Membran och böjning i 2×2 gausspunkter (lokala standardaxlar
//!   i punkten, nodnormaler ur elementets egen geometri).
//! - Membranets nollenergimod h = (−1 i hörnen, 1 i mittnoderna, 0 i mitten)
//!   stabiliseras med 0,005·G·t·(e_ξ e_ξᵀ + e_η e_ηᵀ) för förskjutningarna
//!   (e_ξ, e_η = enhetstangenterna i mitten).
//! - Tvärskjuvning (Kirchhoffvillkoret som straff): γ_ξ = (n·u,ξ + θ·(n × x,ξ))
//!   /|x,ξ| i punkterna (±a, η) längs nodlinjerna η = −1, 0, 1 med vikterna
//!   1/3, 4/3, 1/3; a = 1/√3 på kantlinjerna och 1,01/√3 på mittlinjen.
//!   Ytelementet är det J i (±1/√3, 0) (γ_ξ) respektive (0, ±1/√3) (γ_η), där
//!   γ_ξ på linjerna η = 0 och η = 1 tar punkten med motsatt
//!   tecken. Styvheten 5/6·G·t·f(a) med a = ytelementet·linjevikten.
//! - Massa: diagonal, ρt∫Nᵢ dA (3×3 punkter) och rotationströghet
//!   ρt³/12·∫N₁ dA i alla noder.
//!
//! Utdata i 2×2 punkter: S, SF (SF4 = SF5 = 0) och SM.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, G2};
use super::{add_drill_eps, default_axes, fp, k_shear, output, project_tangent, section, shear_modulus, unit, Gen};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct Q9 {
    name: &'static str,
    internal: bool,
}

pub static S9R5: Q9 = Q9 { name: "S9R5", internal: false };
pub static S8R5: Q9 = Q9 { name: "S8R5", internal: true };

const ND: usize = 6;
const NN: usize = 9;
const NDOF: usize = 54;
const NAT9: [[f64; 2]; 9] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]];
const GP4: [(f64, f64); 4] = [(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)];

fn l1(a: f64, s: f64) -> (f64, f64) {
    if a < 0.0 {
        (0.5 * s * (s - 1.0), s - 0.5)
    } else if a > 0.0 {
        (0.5 * s * (s + 1.0), s + 0.5)
    } else {
        (1.0 - s * s, -2.0 * s)
    }
}

pub fn shape9(s: f64, t: f64) -> ([f64; 9], [[f64; 9]; 2]) {
    let mut n = [0.0; 9];
    let mut d = [[0.0; 9]; 2];
    for i in 0..9 {
        let (fa, da) = l1(NAT9[i][0], s);
        let (fb, db) = l1(NAT9[i][1], t);
        n[i] = fa * fb;
        d[0][i] = da * fb;
        d[1][i] = fa * db;
    }
    (n, d)
}

fn tangents(x: &[[f64; 3]], d: &[[f64; 9]; 2]) -> [[f64; 3]; 2] {
    [0, 1].map(|a| [0, 1, 2].map(|c| (0..9).map(|i| d[a][i] * x[i][c]).sum()))
}

struct Pt {
    n: [f64; 9],
    d: [[f64; 9]; 2],
    a: [[f64; 3]; 2],
    r: [[f64; 3]; 3],
    det: f64,
    b: [[f64; 9]; 2],
    nx: [[f64; 3]; 2],
}

fn point(x: &[[f64; 3]], nrm: &[[f64; 3]; 9], s: f64, t: f64) -> Pt {
    let (n, d) = shape9(s, t);
    let a = tangents(x, &d);
    let r = default_axes(cross(a[0], a[1]));
    let j = [[dot(a[0], r[0]), dot(a[0], r[1])], [dot(a[1], r[0]), dot(a[1], r[1])]];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let ji = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
    let mut b = [[0.0; 9]; 2];
    for i in 0..9 {
        for al in 0..2 {
            b[al][i] = ji[al][0] * d[0][i] + ji[al][1] * d[1][i];
        }
    }
    let dn: [[f64; 3]; 2] = [0, 1].map(|al| [0, 1, 2].map(|c| (0..9).map(|i| d[al][i] * nrm[i][c]).sum()));
    let nx = [0, 1].map(|al| [0, 1, 2].map(|c| ji[al][0] * dn[0][c] + ji[al][1] * dn[1][c]));
    Pt { n, d, a, r, det, b, nx }
}

fn put(row: &mut [f64], i: usize, off: usize, v: [f64; 3], f: f64) {
    for c in 0..3 {
        row[i * ND + off + c] += f * v[c];
    }
}

/// Membran- och böjoperatorer (3 × 54 vardera), som S8R.
fn strain_ops(p: &Pt) -> (Vec<f64>, Vec<f64>) {
    let n = NDOF;
    let [e1, e2, nv] = p.r;
    let e = [e1, e2];
    let mut bm = vec![0.0; 3 * n];
    let mut bb = vec![0.0; 3 * n];
    let c = [0, 1].map(|a| [0, 1].map(|b| dot(p.nx[a], e[b])));
    let nn = [dot(p.nx[0], nv), dot(p.nx[1], nv)];
    let om_c = [-(c[0][1] + c[1][0]), c[0][1] + c[1][0], 2.0 * (c[0][0] - c[1][1])];
    let a_skew = 0.5 * (c[0][1] - c[1][0]);
    for i in 0..NN {
        let (b0, b1, ni) = (p.b[0][i], p.b[1][i], p.n[i]);
        put(&mut bm[0..n], i, 0, e1, b0);
        put(&mut bm[n..2 * n], i, 0, e2, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e1, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e2, b0);
        let phi = |al: usize, eb: [f64; 3]| -> [f64; 3] {
            let u = scale(cross(nv, eb), p.b[al][i]);
            let w = scale(cross(p.nx[al], eb), ni);
            [u[0] + w[0], u[1] + w[1], u[2] + w[2]]
        };
        put(&mut bb[0..n], i, 3, phi(0, e1), 1.0);
        put(&mut bb[n..2 * n], i, 3, phi(1, e2), 1.0);
        put(&mut bb[2 * n..3 * n], i, 3, phi(1, e1), 1.0);
        put(&mut bb[2 * n..3 * n], i, 3, phi(0, e2), 1.0);
        for (r, &oc) in om_c.iter().enumerate() {
            let f = -0.25 * oc;
            put(&mut bb[r * n..(r + 1) * n], i, 0, e2, f * b0);
            put(&mut bb[r * n..(r + 1) * n], i, 0, e1, -f * b1);
        }
        put(&mut bb[0..n], i, 0, nv, nn[0] * b0);
        put(&mut bb[n..2 * n], i, 0, nv, nn[1] * b1);
        put(&mut bb[2 * n..3 * n], i, 0, nv, nn[0] * b1 + nn[1] * b0);
        put(&mut bb[0..n], i, 0, e2, a_skew * b0);
        put(&mut bb[n..2 * n], i, 0, e1, -a_skew * b1);
        put(&mut bb[2 * n..3 * n], i, 0, e2, a_skew * b1);
        put(&mut bb[2 * n..3 * n], i, 0, e1, -a_skew * b0);
    }
    (bm, bb)
}

/// Tvärskjuvningen längs ξ (comp 0) eller η (comp 1) i punkten: (n·u,α + θ·(n × x,α))/|x,α|.
fn shear_vec(p: &Pt, comp: usize) -> Vec<f64> {
    let mut g = vec![0.0; NDOF];
    let nv = p.r[2];
    let l = norm(p.a[comp]);
    let nxa = cross(nv, p.a[comp]);
    for i in 0..NN {
        put(&mut g, i, 0, nv, p.d[comp][i] / l);
        put(&mut g, i, 3, nxa, p.n[i] / l);
    }
    g
}

struct Setup {
    x: Vec<[f64; 3]>,
    nrm: [[f64; 3]; 9],
    sec: super::Sec,
}

impl Q9 {
    fn setup(&self, ctx: &ElemCtx) -> Result<Setup> {
        let mut x = ctx.x.clone();
        if x.len() != 9 {
            return fail(format!("element {} ({}) har {} noder, ska ha 9", ctx.label, self.name, x.len()));
        }
        if self.internal {
            // serendipityytans mittpunkt
            x[8] = [0, 1, 2].map(|c| -0.25 * (0..4).map(|i| x[i][c]).sum::<f64>() + 0.5 * (4..8).map(|i| x[i][c]).sum::<f64>());
        }
        let mut nrm = [[0.0; 3]; 9];
        for (i, &[s, t]) in NAT9.iter().enumerate() {
            let (_, d) = shape9(s, t);
            let a = tangents(&x, &d);
            let c = cross(a[0], a[1]);
            if !(norm(c) > 0.0) {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
            nrm[i] = unit(c);
        }
        for &(s, t) in &GP4 {
            if !(point(&x, &nrm, s, t).det > 0.0) {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
        }
        Ok(Setup { x, nrm, sec: section(ctx, self.name, false)? })
    }

    /// Skjuvprover: (vektor, styvhet·vikt, komponent, punkt (ξ, η)).
    fn shear_samples(su: &Setup) -> Vec<(Vec<f64>, f64, usize)> {
        let sec = &su.sec;
        let k0 = k_shear(sec);
        let a_mid = 1.01 * G2;
        let mut out = Vec::with_capacity(12);
        for comp in 0..2 {
            for (li, (bl, wb)) in [(-1.0, 1.0 / 3.0), (0.0, 4.0 / 3.0), (1.0, 1.0 / 3.0)].into_iter().enumerate() {
                let sa = if li == 1 { a_mid } else { G2 };
                for a in [-sa, sa] {
                    let (s, t) = if comp == 0 { (a, bl) } else { (bl, a) };
                    let p = point(&su.x, &su.nrm, s, t);
                    let g = shear_vec(&p, comp);
                    let gs = G2 * a.signum();
                    let q = if comp == 0 { (if li == 0 { gs } else { -gs }, 0.0) } else { (0.0, gs) };
                    let det = point(&su.x, &su.nrm, q.0, q.1).det;
                    let k = k0 * fp(det * wb, sec.t);
                    out.push((g, k * det * wb, comp));
                }
            }
        }
        out
    }

    fn hourglass(su: &Setup) -> (Vec<f64>, [f64; 9]) {
        let p0 = point(&su.x, &su.nrm, 0.0, 0.0);
        let (e1, e2) = (unit(p0.a[0]), unit(p0.a[1]));
        let c0 = 0.005 * shear_modulus(&su.sec) * su.sec.t;
        let mut c = [0.0; 9];
        for a in 0..3 {
            for b in 0..3 {
                c[a * 3 + b] = c0 * (e1[a] * e1[b] + e2[a] * e2[b]);
            }
        }
        let h = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let mut t = vec![0.0; 3 * NDOF];
        for i in 0..NN {
            for d in 0..3 {
                t[d * NDOF + ND * i + d] = h[i];
            }
        }
        (t, c)
    }

    fn stiffness_of(su: &Setup) -> Vec<f64> {
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
        for &(s, t) in &GP4 {
            let p = point(&su.x, &su.nrm, s, t);
            let (bm, bb) = strain_ops(&p);
            super::super::shell2::add_btcb(&mut k, n, &bm, &cm, &bm, 3, p.det);
            super::super::shell2::add_btcb(&mut k, n, &bb, &cb, &bb, 3, p.det);
        }
        for (g, c, _) in Self::shear_samples(su) {
            super::super::shell2::add_outer(&mut k, n, &g, c);
        }
        let (t, c) = Self::hourglass(su);
        super::super::shell2::add_btcb(&mut k, n, &t, &c, &t, 3, 1.0);
        let mut k = project_tangent(&k, &su.nrm);
        add_drill_eps(&mut k, &su.nrm);
        k
    }

    fn area_points(su: &Setup) -> Vec<([f64; 9], [f64; 3], f64)> {
        super::super::shell2::s8r::gp9()
            .into_iter()
            .map(|(s, t, w)| {
                let (n, d) = shape9(s, t);
                let a = tangents(&su.x, &d);
                let c = cross(a[0], a[1]);
                let da = norm(c);
                (n, scale(c, 1.0 / da), da * w)
            })
            .collect()
    }
}

impl ElementType for Q9 {
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
        9
    }
    fn n_connectivity(&self) -> usize {
        if self.internal {
            8
        } else {
            9
        }
    }
    fn internal_nodes(&self) -> usize {
        usize::from(self.internal)
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        4
    }
    fn vtk_type(&self) -> u8 {
        if self.internal {
            23
        } else {
            28
        }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.internal {
            &[&[0, 1, 2, 3, 4, 5, 6, 7]]
        } else {
            &[&[0, 1, 2, 3, 4, 5, 6, 7, 8]]
        }
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Self::stiffness_of(&self.setup(ctx)?))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let mut w = [0.0; 9];
        for (nv, _, da) in Self::area_points(&su) {
            for i in 0..9 {
                w[i] += nv[i] * da;
            }
        }
        let n = NDOF;
        let mut m = vec![0.0; n * n];
        let mr = su.sec.rho * su.sec.t.powi(3) / 12.0 * w[0];
        for i in 0..9 {
            for d in 0..3 {
                m[(ND * i + d) * n + ND * i + d] = su.sec.rho * su.sec.t * w[i];
            }
            // rotationströghet för rotationerna i nodens tangentplan
            let ni = su.nrm[i];
            for a in 0..3 {
                for b in 0..3 {
                    let pab = if a == b { 1.0 } else { 0.0 } - ni[a] * ni[b];
                    m[(ND * i + 3 + a) * n + ND * i + 3 + b] = mr * pab;
                }
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, su.sec.rho, self.name)?;
        if load.kind != "P" && body.is_none() {
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        }
        let mut f = vec![0.0; NDOF];
        for (nv, nrm, da) in Self::area_points(&su) {
            let q = if load.kind == "P" { scale(nrm, load.magnitude) } else { scale(body.unwrap(), su.sec.t) };
            for i in 0..9 {
                for c in 0..3 {
                    f[ND * i + c] += nv[i] * da * q[c];
                }
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let mut v = vec![0.0; 9];
        for (nv, _, da) in Self::area_points(&su) {
            for i in 0..9 {
                v[i] += nv[i] * da * su.sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let su = self.setup(ctx)?;
        // rotationerna verkar bara genom tangentplanets komponenter
        let mut up = u.to_vec();
        for i in 0..NN {
            let ni = su.nrm[i];
            let th = [u[ND * i + 3], u[ND * i + 4], u[ND * i + 5]];
            let tn = dot(th, ni);
            for c in 0..3 {
                up[ND * i + 3 + c] = th[c] - tn * ni[c];
            }
        }
        let pts: Vec<Gen> = GP4
            .iter()
            .map(|&(s, t)| {
                let p = point(&su.x, &su.nrm, s, t);
                let (bm, bb) = strain_ops(&p);
                let mv = |b: &[f64], r: usize| -> f64 { (0..NDOF).map(|q| b[r * NDOF + q] * up[q]).sum() };
                // ingen tvärkraft redovisas (SF4 = SF5 = 0) för tunna skal
                Gen { eps: [mv(&bm, 0), mv(&bm, 1), mv(&bm, 2)], kap: [mv(&bb, 0), mv(&bb, 1), mv(&bb, 2)], q: [0.0; 2] }
            })
            .collect();
        Ok(output(&su.sec, &pts))
    }
}
