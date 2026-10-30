//! STRI3: plant treknutsskal för tunna skal, sex frihetsgrader per nod
//! (område shells3).
//!
//! Formulering:
//!
//! - Lokalt system: normal n = (x2 − x1) × (x3 − x1), 1-axel enligt
//!   standardaxlarna (projektionen av global X), 2 = n × 1.
//! - Membran: konstant töjning (CST).
//! - Böjning: diskret Kirchhofftriangel (DKT, Batoz–Bathe–Ho 1980) med
//!   βx = θy, βy = −θx, integrerad med tre punkter; ingen tvärskjuvning.
//! - Drill: straff kᵢ(θzᵢ − ωᵢ)² i varje nod, där ωᵢ är medelvärdet av de två
//!   anslutande kanternas kordrotationer och kᵢ = 10⁻³·G·t·A/3.
//! - Massa: diagonal, ρtA/3 per nod och ρt³/12·A/3 för alla tre rotationerna.
//!
//! Utdata i tre integrationspunkter (arealkoordinaterna (2/3, 1/6, 1/6) osv.):
//! S i sektionspunkterna 1 och n, SF (SF4 = SF5 = 0) och SM.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, sub};
use super::{default_axes, output, section, shear_modulus, Gen};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct Stri3;
pub static STRI3: Stri3 = Stri3;

const ND: usize = 6;
const NDOF: usize = 18;
/// Integrationspunkter (L2, L3) för utdata och böjning.
const IPS: [(f64, f64); 3] = [(1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0)];

/// DKT-krökningsoperatorn (3 × 9, frihetsgrader (w, θx, θy) per nod) i
/// punkten (ξ, η) = (L2, L3).
pub fn dkt_b(x: [f64; 3], y: [f64; 3], xi: f64, eta: f64) -> [[f64; 9]; 3] {
    let (x23, x31, x12) = (x[1] - x[2], x[2] - x[0], x[0] - x[1]);
    let (y23, y31, y12) = (y[1] - y[2], y[2] - y[0], y[0] - y[1]);
    let xs = [x23, x31, x12];
    let ys = [y23, y31, y12];
    let l2: Vec<f64> = (0..3).map(|k| xs[k] * xs[k] + ys[k] * ys[k]).collect();
    let p: Vec<f64> = (0..3).map(|k| -6.0 * xs[k] / l2[k]).collect();
    let q: Vec<f64> = (0..3).map(|k| 3.0 * xs[k] * ys[k] / l2[k]).collect();
    let r: Vec<f64> = (0..3).map(|k| 3.0 * ys[k] * ys[k] / l2[k]).collect();
    let t: Vec<f64> = (0..3).map(|k| -6.0 * ys[k] / l2[k]).collect();
    let (p4, p5, p6) = (p[0], p[1], p[2]);
    let (q4, q5, q6) = (q[0], q[1], q[2]);
    let (r4, r5, r6) = (r[0], r[1], r[2]);
    let (t4, t5, t6) = (t[0], t[1], t[2]);
    let a = 1.0 - 2.0 * xi;
    let b = 1.0 - 2.0 * eta;
    let hx_xi = [
        p6 * a + (p5 - p6) * eta,
        q6 * a - (q5 + q6) * eta,
        -4.0 + 6.0 * (xi + eta) + r6 * a - eta * (r5 + r6),
        -p6 * a + eta * (p4 + p6),
        q6 * a - eta * (q6 - q4),
        -2.0 + 6.0 * xi + r6 * a + eta * (r4 - r6),
        -eta * (p5 + p4),
        eta * (q4 - q5),
        -eta * (r5 - r4),
    ];
    let hy_xi = [
        t6 * a + eta * (t5 - t6),
        1.0 + r6 * a - eta * (r5 + r6),
        -q6 * a + eta * (q5 + q6),
        -t6 * a + eta * (t4 + t6),
        -1.0 + r6 * a + eta * (r4 - r6),
        -q6 * a - eta * (q4 - q6),
        -eta * (t4 + t5),
        eta * (r4 - r5),
        -eta * (q4 - q5),
    ];
    let hx_eta = [
        -p5 * b - xi * (p6 - p5),
        q5 * b - xi * (q5 + q6),
        -4.0 + 6.0 * (xi + eta) + r5 * b - xi * (r5 + r6),
        xi * (p4 + p6),
        xi * (q4 - q6),
        -xi * (r6 - r4),
        p5 * b - xi * (p4 + p5),
        q5 * b + xi * (q4 - q5),
        -2.0 + 6.0 * eta + r5 * b + xi * (r4 - r5),
    ];
    let hy_eta = [
        -t5 * b - xi * (t6 - t5),
        1.0 + r5 * b - xi * (r5 + r6),
        -q5 * b + xi * (q5 + q6),
        xi * (t4 + t6),
        xi * (r4 - r6),
        -xi * (q4 - q6),
        t5 * b - xi * (t4 + t5),
        -1.0 + r5 * b + xi * (r4 - r5),
        -q5 * b - xi * (q4 - q5),
    ];
    let a2 = x31 * y12 - x12 * y31;
    let mut out = [[0.0; 9]; 3];
    for k in 0..9 {
        out[0][k] = (y31 * hx_xi[k] + y12 * hx_eta[k]) / a2;
        out[1][k] = (-x31 * hy_xi[k] - x12 * hy_eta[k]) / a2;
        out[2][k] = (-x31 * hx_xi[k] - x12 * hx_eta[k] + y31 * hy_xi[k] + y12 * hy_eta[k]) / a2;
    }
    out
}

/// Lokalt system: rader (e1, e2, n) och plana koordinater relativt nod 1.
pub fn frame(ctx: &ElemCtx, name: &str) -> Result<([[f64; 3]; 3], [f64; 3], [f64; 3], f64)> {
    let xg = &ctx.x;
    let nv = cross(sub(xg[1], xg[0]), sub(xg[2], xg[0]));
    let a2 = norm(nv);
    if !(a2 > 0.0) {
        return fail(format!("element {} ({name}) har ingen area", ctx.label));
    }
    let r = default_axes(nv);
    let mut x = [0.0; 3];
    let mut y = [0.0; 3];
    for i in 0..3 {
        let d = sub(xg[i], xg[0]);
        x[i] = dot(r[0], d);
        y[i] = dot(r[1], d);
    }
    Ok((r, x, y, 0.5 * a2))
}

/// Lokala operatorer: membran (3 × 18), drillvektorer (3 × 18) och DKT per punkt.
struct Ops {
    bm: [[f64; NDOF]; 3],
    drill: [[f64; NDOF]; 3],
}

fn ops(x: [f64; 3], y: [f64; 3], a: f64) -> Ops {
    let a2 = 2.0 * a;
    let bx = [(y[1] - y[2]) / a2, (y[2] - y[0]) / a2, (y[0] - y[1]) / a2];
    let by = [(x[2] - x[1]) / a2, (x[0] - x[2]) / a2, (x[1] - x[0]) / a2];
    let mut bm = [[0.0; NDOF]; 3];
    for i in 0..3 {
        bm[0][ND * i] = bx[i];
        bm[1][ND * i + 1] = by[i];
        bm[2][ND * i] = by[i];
        bm[2][ND * i + 1] = bx[i];
    }
    // kordrotationer: kant p → q, ω = (τx Δv − τy Δu)/L
    let chord = |p: usize, q: usize, g: &mut [f64; NDOF], f: f64| {
        let (dx, dy) = (x[q] - x[p], y[q] - y[p]);
        let l2 = dx * dx + dy * dy;
        g[ND * q] += -f * dy / l2;
        g[ND * q + 1] += f * dx / l2;
        g[ND * p] -= -f * dy / l2;
        g[ND * p + 1] -= f * dx / l2;
    };
    let mut drill = [[0.0; NDOF]; 3];
    for i in 0..3 {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let g = &mut drill[i];
        g[ND * i + 5] = 1.0;
        chord(i, j, g, -0.5);
        chord(i, k, g, -0.5);
    }
    Ops { bm, drill }
}

impl Stri3 {
    fn local_k(&self, ctx: &ElemCtx) -> Result<(Vec<f64>, [[f64; 3]; 3])> {
        let (r, x, y, a) = frame(ctx, "STRI3")?;
        let sec = section(ctx, "STRI3", false)?;
        let o = ops(x, y, a);
        let n = NDOF;
        let mut k = vec![0.0; n * n];
        let tb = sec.t.powi(3) / 12.0;
        for p in 0..n {
            for q in 0..n {
                let mut v = 0.0;
                for i in 0..3 {
                    for j in 0..3 {
                        v += o.bm[i][p] * sec.d[i * 3 + j] * o.bm[j][q];
                    }
                }
                k[p * n + q] += v * sec.t * a;
            }
        }
        let kd = 1e-3 * shear_modulus(&sec) * sec.t * a / 3.0;
        for g in &o.drill {
            super::super::shell2::add_outer(&mut k, n, g, kd);
        }
        let map = |i: usize| -> usize { ND * (i / 3) + 2 + i % 3 };
        for &(xi, eta) in &IPS {
            let b = dkt_b(x, y, xi, eta);
            for p in 0..9 {
                for q in 0..9 {
                    let mut v = 0.0;
                    for i in 0..3 {
                        for j in 0..3 {
                            v += b[i][p] * sec.d[i * 3 + j] * b[j][q];
                        }
                    }
                    k[map(p) * n + map(q)] += v * tb * a / 3.0;
                }
            }
        }
        Ok((k, r))
    }

    fn rot_matrix(r: &[[f64; 3]; 3]) -> Vec<f64> {
        common::block_diag(r, 6)
    }
}

impl ElementType for Stri3 {
    fn name(&self) -> &'static str {
        "STRI3"
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
        3
    }
    fn vtk_type(&self) -> u8 {
        5
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2]]
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (kl, r) = self.local_k(ctx)?;
        Ok(common::congruence(&kl, &Self::rot_matrix(&r), NDOF))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", "STRI3");
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let (_, _, _, a) = frame(ctx, "STRI3")?;
        let sec = section(ctx, "STRI3", false)?;
        let n = NDOF;
        let mut m = vec![0.0; n * n];
        let mt = sec.rho * sec.t * a / 3.0;
        let mr = sec.rho * sec.t.powi(3) / 12.0 * a / 3.0;
        for i in 0..3 {
            for d in 0..6 {
                m[(ND * i + d) * n + ND * i + d] = if d < 3 { mt } else { mr };
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (r, _, _, a) = frame(ctx, "STRI3")?;
        let sec = section(ctx, "STRI3", false)?;
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, "STRI3")?;
        let q = match (load.kind.as_str(), body) {
            ("P", _) => scale(r[2], load.magnitude),
            (_, Some(b)) => scale(b, sec.t),
            _ => return not_implemented(&format!("lasttypen {}", load.kind), "STRI3"),
        };
        let mut f = vec![0.0; NDOF];
        for i in 0..3 {
            for c in 0..3 {
                f[ND * i + c] = q[c] * a / 3.0;
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (_, _, _, a) = frame(ctx, "STRI3")?;
        let sec = section(ctx, "STRI3", false)?;
        Ok(vec![a * sec.t / 3.0; 3])
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let (r, x, y, a) = frame(ctx, "STRI3")?;
        let sec = section(ctx, "STRI3", false)?;
        let t = Self::rot_matrix(&r);
        let ul: Vec<f64> = (0..NDOF).map(|p| (0..NDOF).map(|q| t[p * NDOF + q] * u[q]).sum()).collect();
        let o = ops(x, y, a);
        let eps = [0, 1, 2].map(|i| super::row_dot(&o.bm[i], &ul));
        let pts: Vec<Gen> = IPS
            .iter()
            .map(|&(xi, eta)| {
                let b = dkt_b(x, y, xi, eta);
                let mut kap = [0.0; 3];
                for i in 0..3 {
                    for p in 0..9 {
                        kap[i] += b[i][p] * ul[ND * (p / 3) + 2 + p % 3];
                    }
                }
                Gen { eps, kap, q: [0.0; 2] }
            })
            .collect();
        Ok(output(&sec, &pts))
    }
}
