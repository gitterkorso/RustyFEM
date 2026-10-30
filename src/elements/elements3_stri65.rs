//! STRI65: sexnodigt tunt triangelskal med fem frihetsgrader per nod
//! (område elements3).
//!
//! Formulering:
//!
//! - Geometri och förskjutningar med de kvadratiska formfunktionerna (T6),
//!   naturliga koordinater (r, s) = (L2, L3).
//! - Membran och böjning (krökning ur rotationerna, som S8R/S9R5 i
//!   `shell3_q9.rs`) i tre punkter (2/3, 1/6, 1/6) osv. med vikten A/3.
//! - Kirchhoffvillkoret som straff på tvärskjuvningen γ_t = n·u,t + θ·(n × t):
//!   - längs varje kant i kantens två gausspunkter (±1/√3), riktning längs
//!     kanten, vikt A/5 per punkt (sex punkter),
//!   - inne i elementet i mittpunkterna P_k av den inre triangelns kanter
//!     (L_k = 1/2, de andra 1/4) med riktningen längs motstående kant k:
//!     värdena h₁ (2→3), h₂ (3→1), h₃ (1→2). Med kantlängderna L_q = |x₃ − x₂|,
//!     L_s = |x₁ − x₃|, L_r = |x₂ − x₁|, S = L_q + L_s + L_r och
//!     λ = h₁/S + c·(L_s h₂ + L_r h₃), c = (L_s + L_r)/(S(L_s² + L_r²)),
//!     straffas (h₃ − L_r λ) och (h₂ − L_s λ) med vikten 0,4·A vardera (nod 1
//!     är alltså särskild).
//!   Varje straff har styvheten 5/6·G·t·f(a)·a med vikten a och
//!   slankhetskompensationen f(a) = 1/(1 + 0,25·10⁻⁴·a/t²).
//! - Femfrihetsgradsnoderna assembleras med sex globala frihetsgrader (se
//!   `shell3.rs`): rotationerna verkar genom sina komponenter i nodens
//!   tangentplan och rotationen kring nodnormalen får styvheten `DRILL_EPS`.
//! - Massa: konsistent för förskjutningarna (sexpunktsregel av grad 4,
//!   även för krökta element), ingen rotationströghet.
//!
//! Utdata i de tre integrationspunkterna: S (sektionspunkterna 1 och n), SF
//! (SF4 = SF5 = 0 som för tunna skal) och SM.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale};
use super::super::shell2::{add_btcb, add_outer, Sec};
use super::super::shell3::{add_drill_eps, default_axes, fp, k_shear, output, project_tangent, section, unit, Gen};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct Stri65;
pub static STRI65: Stri65 = Stri65;

const NAME: &str = "STRI65";
const ND: usize = 6;
const NN: usize = 6;
const NDOF: usize = 36;
/// Integrationspunkter (r, s) = (L2, L3), vikt 1/6 (referenstriangelns area 1/2).
pub const IPS: [(f64, f64); 3] = [(1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0)];
/// Naturliga koordinater för noderna.
pub const NAT6: [[f64; 2]; 6] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];

/// T6-formfunktioner och derivator med avseende på (r, s).
pub fn shape6(r: f64, s: f64) -> ([f64; 6], [[f64; 6]; 2]) {
    let l = 1.0 - r - s;
    let n = [l * (2.0 * l - 1.0), r * (2.0 * r - 1.0), s * (2.0 * s - 1.0), 4.0 * l * r, 4.0 * r * s, 4.0 * s * l];
    let dr = [-(4.0 * l - 1.0), 4.0 * r - 1.0, 0.0, 4.0 * (l - r), 4.0 * s, -4.0 * s];
    let ds = [-(4.0 * l - 1.0), 0.0, 4.0 * s - 1.0, -4.0 * r, 4.0 * r, 4.0 * (l - s)];
    (n, [dr, ds])
}

fn tangents(x: &[[f64; 3]], d: &[[f64; 6]; 2]) -> [[f64; 3]; 2] {
    [0, 1].map(|a| [0, 1, 2].map(|c| (0..NN).map(|i| d[a][i] * x[i][c]).sum()))
}

/// Nodnormaler ur elementets egen geometri.
pub fn own_normals(x: &[[f64; 3]]) -> Option<[[f64; 3]; 6]> {
    let mut nrm = [[0.0; 3]; 6];
    for (i, &[r, s]) in NAT6.iter().enumerate() {
        let (_, d) = shape6(r, s);
        let a = tangents(x, &d);
        let c = cross(a[0], a[1]);
        if !(norm(c) > 0.0) {
            return None;
        }
        nrm[i] = unit(c);
    }
    Some(nrm)
}

struct Pt {
    n: [f64; 6],
    d: [[f64; 6]; 2],
    a: [[f64; 3]; 2],
    r: [[f64; 3]; 3],
    det: f64,
    b: [[f64; 6]; 2],
    nx: [[f64; 3]; 2],
}

fn point(x: &[[f64; 3]], nrm: &[[f64; 3]; 6], r: f64, s: f64) -> Pt {
    let (n, d) = shape6(r, s);
    let a = tangents(x, &d);
    let ax = default_axes(cross(a[0], a[1]));
    let j = [[dot(a[0], ax[0]), dot(a[0], ax[1])], [dot(a[1], ax[0]), dot(a[1], ax[1])]];
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let ji = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
    let mut b = [[0.0; 6]; 2];
    for i in 0..NN {
        for al in 0..2 {
            b[al][i] = ji[al][0] * d[0][i] + ji[al][1] * d[1][i];
        }
    }
    let dn: [[f64; 3]; 2] = [0, 1].map(|al| [0, 1, 2].map(|c| (0..NN).map(|i| d[al][i] * nrm[i][c]).sum()));
    let nx = [0, 1].map(|al| [0, 1, 2].map(|c| ji[al][0] * dn[0][c] + ji[al][1] * dn[1][c]));
    Pt { n, d, a, r: ax, det, b, nx }
}

fn put(row: &mut [f64], i: usize, off: usize, v: [f64; 3], f: f64) {
    for c in 0..3 {
        row[i * ND + off + c] += f * v[c];
    }
}

/// Membran- och böjoperatorer (3 × 36 vardera), som S8R/S9R5.
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

/// Tvärskjuvningen i punkten (r, s) längs den naturliga riktningen `dir`:
/// (vektor, |x,dir|).
fn shear_vec(x: &[[f64; 3]], nrm: &[[f64; 3]; 6], r: f64, s: f64, dir: [f64; 2]) -> (Vec<f64>, f64) {
    let p = point(x, nrm, r, s);
    let nv = p.r[2];
    let xd = [0, 1, 2].map(|c| dir[0] * p.a[0][c] + dir[1] * p.a[1][c]);
    let l = norm(xd);
    let nxd = cross(nv, xd);
    let mut g = vec![0.0; NDOF];
    for i in 0..NN {
        let di = dir[0] * p.d[0][i] + dir[1] * p.d[1][i];
        put(&mut g, i, 0, nv, di / l);
        put(&mut g, i, 3, nxd, p.n[i] / l);
    }
    (g, l)
}

pub struct Setup {
    pub x: Vec<[f64; 3]>,
    pub nrm: [[f64; 3]; 6],
    pub sec: Sec,
    pub area: f64,
}

/// Kirchhoffstraffen: (vektor, styvhet).
fn shear_terms(su: &Setup) -> Vec<(Vec<f64>, f64)> {
    let sec = &su.sec;
    let (k0, t, a) = (k_shear(sec), sec.t, su.area);
    let mut out = Vec::with_capacity(8);
    let g1 = 0.5 - 0.5 / 3f64.sqrt();
    // kanter: (start, riktning)
    let edges: [([f64; 2], [f64; 2]); 3] = [([0.0, 0.0], [1.0, 0.0]), ([1.0, 0.0], [-1.0, 1.0]), ([0.0, 1.0], [0.0, -1.0])];
    let we = a / 5.0;
    for (p0, dir) in edges {
        for g in [g1, 1.0 - g1] {
            let (v, _) = shear_vec(&su.x, &su.nrm, p0[0] + g * dir[0], p0[1] + g * dir[1], dir);
            out.push((v, k0 * fp(we, t) * we));
        }
    }
    // inre punkter: P1 (2→3), P2 (3→1), P3 (1→2)
    let (h1, lq) = shear_vec(&su.x, &su.nrm, 0.25, 0.25, [-1.0, 1.0]);
    let (h2, ls) = shear_vec(&su.x, &su.nrm, 0.5, 0.25, [0.0, -1.0]);
    let (h3, lr) = shear_vec(&su.x, &su.nrm, 0.25, 0.5, [1.0, 0.0]);
    let sum = lq + ls + lr;
    let c = (ls + lr) / (sum * (ls * ls + lr * lr));
    let lam: Vec<f64> = (0..NDOF).map(|q| h1[q] / sum + c * (ls * h2[q] + lr * h3[q])).collect();
    let wi = 0.4 * a;
    let ki = k0 * fp(wi, t) * wi;
    out.push(((0..NDOF).map(|q| h3[q] - lr * lam[q]).collect(), ki));
    out.push(((0..NDOF).map(|q| h2[q] - ls * lam[q]).collect(), ki));
    out
}

impl Stri65 {
    pub fn setup(&self, ctx: &ElemCtx) -> Result<Setup> {
        let x = ctx.x.clone();
        if x.len() != 6 {
            return fail(format!("element {} ({NAME}) har {} noder, ska ha 6", ctx.label, x.len()));
        }
        let Some(own) = own_normals(&x) else {
            return fail(format!("element {} ({NAME}): {}", ctx.label, common::DISTORTED));
        };
        let nrm = match super::super::normals::of(ctx) {
            Some(v) if v.len() == 6 => {
                let mut a = [[0.0; 3]; 6];
                a.copy_from_slice(&v);
                a
            }
            _ => own,
        };
        let mut area = 0.0;
        for &(r, s) in &IPS {
            let p = point(&x, &nrm, r, s);
            if !(p.det > 0.0) {
                return fail(format!("element {} ({NAME}): {}", ctx.label, common::DISTORTED));
            }
            area += p.det / 6.0;
        }
        Ok(Setup { x, nrm, sec: section(ctx, NAME, false)?, area })
    }

    pub fn stiffness_of(su: &Setup) -> Vec<f64> {
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
        for &(r, s) in &IPS {
            let p = point(&su.x, &su.nrm, r, s);
            let (bm, bb) = strain_ops(&p);
            add_btcb(&mut k, n, &bm, &cm, &bm, 3, p.det / 6.0);
            add_btcb(&mut k, n, &bb, &cb, &bb, 3, p.det / 6.0);
        }
        for (g, c) in shear_terms(su) {
            add_outer(&mut k, n, &g, c);
        }
        let mut k = project_tangent(&k, &su.nrm);
        add_drill_eps(&mut k, &su.nrm);
        k
    }

    /// Punkter för ytintegraler (6 punkter, grad 4, som massan): (N, normal, dA).
    fn area_points(su: &Setup) -> Vec<([f64; 6], [f64; 3], f64)> {
        let (a, b) = (0.445_948_490_915_965, 0.091_576_213_509_771);
        let (wa, wb) = (0.223_381_589_678_011, 0.109_951_743_655_322);
        let pts = [(a, a, wa), (1.0 - 2.0 * a, a, wa), (a, 1.0 - 2.0 * a, wa), (b, b, wb), (1.0 - 2.0 * b, b, wb), (b, 1.0 - 2.0 * b, wb)];
        pts.iter()
            .map(|&(r, s, w)| {
                let (n, d) = shape6(r, s);
                let a = tangents(&su.x, &d);
                let c = cross(a[0], a[1]);
                let da = norm(c);
                (n, scale(c, 1.0 / da), 0.5 * w * da)
            })
            .collect()
    }

    fn up(su: &Setup, u: &[f64]) -> Vec<f64> {
        let mut up = u.to_vec();
        for i in 0..NN {
            let ni = su.nrm[i];
            let th = [u[ND * i + 3], u[ND * i + 4], u[ND * i + 5]];
            let tn = dot(th, ni);
            for c in 0..3 {
                up[ND * i + 3 + c] = th[c] - tn * ni[c];
            }
        }
        up
    }

    /// Generaliserade töjningar i integrationspunkterna.
    pub fn gen(su: &Setup, u: &[f64]) -> Vec<Gen> {
        let up = Self::up(su, u);
        IPS.iter()
            .map(|&(r, s)| {
                let p = point(&su.x, &su.nrm, r, s);
                let (bm, bb) = strain_ops(&p);
                let mv = |b: &[f64], k: usize| -> f64 { (0..NDOF).map(|q| b[k * NDOF + q] * up[q]).sum() };
                Gen { eps: [mv(&bm, 0), mv(&bm, 1), mv(&bm, 2)], kap: [mv(&bb, 0), mv(&bb, 1), mv(&bb, 2)], q: [0.0; 2] }
            })
            .collect()
    }
}

impl ElementType for Stri65 {
    fn name(&self) -> &'static str {
        NAME
    }
    fn family(&self) -> Family {
        Family::Shell
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        6
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        3
    }
    fn vtk_type(&self) -> u8 {
        22
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2, 3, 4, 5]]
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Self::stiffness_of(&self.setup(ctx)?))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", NAME);
        }
        if let Some(m) = ctx.material {
            if !m.is_linear() {
                return not_implemented(&format!("olinjärt material ({})", m.name()), NAME);
            }
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let n = NDOF;
        let mut m = vec![0.0; n * n];
        let rt = su.sec.rho * su.sec.t;
        let pts = Self::area_points(&su);
        for (nv, _, da) in &pts {
            for i in 0..NN {
                for j in 0..NN {
                    let v = rt * nv[i] * nv[j] * da;
                    for d in 0..3 {
                        m[(ND * i + d) * n + ND * j + d] += v;
                    }
                }
            }
        }
        if kind == MassKind::Lumped {
            // radsummorna i förskjutningarna
            for i in 0..NN {
                for d in 0..3 {
                    let r = ND * i + d;
                    let s: f64 = (0..NN).map(|j| m[r * n + ND * j + d]).sum();
                    for j in 0..NN {
                        m[r * n + ND * j + d] = 0.0;
                    }
                    m[r * n + r] = s;
                }
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, su.sec.rho, NAME)?;
        if load.kind != "P" && body.is_none() {
            return not_implemented(&format!("lasttypen {}", load.kind), NAME);
        }
        let mut f = vec![0.0; NDOF];
        for (nv, nrm, da) in Self::area_points(&su) {
            let q = if load.kind == "P" { scale(nrm, load.magnitude) } else { scale(body.unwrap(), su.sec.t) };
            for i in 0..NN {
                for c in 0..3 {
                    f[ND * i + c] += nv[i] * da * q[c];
                }
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let su = self.setup(ctx)?;
        let mut v = vec![0.0; NN];
        for (nv, _, da) in Self::area_points(&su) {
            for i in 0..NN {
                v[i] += nv[i] * da * su.sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let su = self.setup(ctx)?;
        let pts = Self::gen(&su, u);
        Ok(output(&su.sec, &pts))
    }
}
