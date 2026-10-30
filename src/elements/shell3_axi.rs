//! Axisymmetriska skal SAX1, SAX2 och membran MAX1, MAX2 (område shells3).
//!
//! Koordinat 1 är radien r och 2 axeln z; frihetsgrader U1 (u_r), U2 (u_z)
//! och, för skalen, UR3 (rotationen β i r–z-planet). Noderna i SAX2/MAX2 är
//! 1, 2 (mitt), 3. Integrationen tas över hela varvet (2πr).
//!
//! Formulering:
//!
//! - Meridianens tangent t = x,s, normal n = (−t_z, t_r) (tangenten vriden
//!   90° moturs).
//! - Membran: ε_s = t·u,s, ε_θ = u_r/r.
//! - Böjning: κ_s = β,s, κ_θ = β·t_r/r.
//! - Tvärskjuvning: γ = n·u,s − β med styvheten 5/6·G·t·f, där
//!   f = 1/(1 + 0,25·10⁻⁴·a/t²) och a = 2πr·ds för integrationspunkten.
//! - Integrering: SAX1/MAX1 en punkt, SAX2/MAX2 två gausspunkter.
//! - Massa: SAX1/MAX1 diagonal (halva massan per nod), SAX2/MAX2 konsistent
//!   med 3 gausspunkter; ingen rotationströghet.
//! - Utdata: S (S11 meridional, S22 i ringled) i sektionspunkterna 1 och n,
//!   (S33 = S12 = 0), SF (SF11 = N_s, SF22 = N_θ, SF33 = 0, SF12 = tvärkraft)
//!   och SM (SM1 = M_θ, SM2 = M_s, SM3 = 0); krökningen redovisas
//!   med omvänt tecken (sektionspunkt n på sidan −n).

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common;
use super::{fp, k_shear, section};
use crate::fem::dof::{Dof, U1, U2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};
use std::f64::consts::PI;

pub struct Axi {
    name: &'static str,
    nn: usize,
    shell: bool,
}

pub static SAX1: Axi = Axi { name: "SAX1", nn: 2, shell: true };
pub static SAX2: Axi = Axi { name: "SAX2", nn: 3, shell: true };
pub static MAX1: Axi = Axi { name: "MAX1", nn: 2, shell: false };
pub static MAX2: Axi = Axi { name: "MAX2", nn: 3, shell: false };

/// Formfunktioner och dN/dξ (nodordning 1, 2 (mitt), 3 för tre noder).
pub fn shape(nn: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    if nn == 2 {
        (vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], vec![-0.5, 0.5])
    } else {
        (vec![-0.5 * xi * (1.0 - xi), 1.0 - xi * xi, 0.5 * xi * (1.0 + xi)], vec![xi - 0.5, -2.0 * xi, xi + 0.5])
    }
}

/// Gausspunkter (ξ, vikt).
pub fn gauss(n: usize) -> Vec<(f64, f64)> {
    match n {
        1 => vec![(0.0, 2.0)],
        2 => vec![(-common::G2, 1.0), (common::G2, 1.0)],
        _ => {
            let g = 0.6f64.sqrt();
            vec![(-g, 5.0 / 9.0), (0.0, 8.0 / 9.0), (g, 5.0 / 9.0)]
        }
    }
}

/// Data i en punkt: N, dN/ds, r, ds/dξ, t, n.
struct Pt {
    n: Vec<f64>,
    ns: Vec<f64>,
    r: f64,
    j: f64,
    t: [f64; 2],
    nv: [f64; 2],
}

fn point(x: &[[f64; 3]], nn: usize, xi: f64) -> Pt {
    let (n, d) = shape(nn, xi);
    let a = [(0..nn).map(|i| d[i] * x[i][0]).sum::<f64>(), (0..nn).map(|i| d[i] * x[i][1]).sum::<f64>()];
    let j = (a[0] * a[0] + a[1] * a[1]).sqrt();
    let t = [a[0] / j, a[1] / j];
    let r = (0..nn).map(|i| n[i] * x[i][0]).sum();
    let ns = d.iter().map(|v| v / j).collect();
    Pt { n, ns, r, j, t, nv: [-t[1], t[0]] }
}

impl Axi {
    fn nd(&self) -> usize {
        if self.shell {
            3
        } else {
            2
        }
    }
    fn n_ip(&self) -> usize {
        self.nn - 1
    }
    fn check(&self, ctx: &ElemCtx) -> Result<()> {
        for (xi, _) in gauss(3) {
            let p = point(&ctx.x, self.nn, xi);
            if !(p.j > 0.0) || !(p.r > 0.0) {
                return fail(format!("element {} ({}): radien måste vara positiv och elementet ha längd", ctx.label, self.name));
            }
        }
        Ok(())
    }
    /// Operatorer i integrationspunkterna: (membran 2×n, böjning 2×n, skjuvning n, vikt dA, skjuvstyvhet).
    fn ops(&self, ctx: &ElemCtx, ks: f64, t: f64) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
        let nd = self.nd();
        let n = nd * self.nn;
        gauss(self.n_ip())
            .into_iter()
            .map(|(xi, w)| {
                let p = point(&ctx.x, self.nn, xi);
                let mut bm = vec![0.0; 2 * n];
                let mut bb = vec![0.0; 2 * n];
                let mut bs = vec![0.0; n];
                for i in 0..self.nn {
                    bm[nd * i] = p.ns[i] * p.t[0];
                    bm[nd * i + 1] = p.ns[i] * p.t[1];
                    bm[n + nd * i] = p.n[i] / p.r;
                    if self.shell {
                        bb[nd * i + 2] = p.ns[i];
                        bb[n + nd * i + 2] = p.n[i] * p.t[0] / p.r;
                        bs[nd * i] = p.ns[i] * p.nv[0];
                        bs[nd * i + 1] = p.ns[i] * p.nv[1];
                        bs[nd * i + 2] = -p.n[i];
                    }
                }
                let da = 2.0 * PI * p.r * p.j * w;
                (bm, bb, bs, da, ks * fp(da, t))
            })
            .collect()
    }

    fn stiffness_of(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, !self.shell)?;
        let n = self.nd() * self.nn;
        let d2 = [sec.d[0], sec.d[1], sec.d[3], sec.d[4]];
        let tb = sec.t.powi(3) / 12.0;
        let mut k = vec![0.0; n * n];
        for (bm, bb, bs, da, kss) in self.ops(ctx, k_shear(&sec), sec.t) {
            for a in 0..2 {
                for b in 0..2 {
                    let cm = d2[a * 2 + b] * sec.t * da;
                    let cb = d2[a * 2 + b] * tb * da;
                    for p in 0..n {
                        let (ma, ba) = (bm[a * n + p], bb[a * n + p]);
                        if ma == 0.0 && ba == 0.0 {
                            continue;
                        }
                        for q in 0..n {
                            k[p * n + q] += cm * ma * bm[b * n + q] + cb * ba * bb[b * n + q];
                        }
                    }
                }
            }
            if self.shell {
                for p in 0..n {
                    for q in 0..n {
                        k[p * n + q] += kss * da * bs[p] * bs[q];
                    }
                }
            }
        }
        Ok(k)
    }
}

impl ElementType for Axi {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        if self.shell {
            Family::Shell
        } else {
            Family::Membrane
        }
    }
    fn dim(&self) -> usize {
        2
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.shell {
            &[U1, U2, UR3]
        } else {
            &[U1, U2]
        }
    }
    fn n_ips(&self) -> usize {
        self.n_ip()
    }
    fn vtk_type(&self) -> u8 {
        if self.nn == 2 {
            3
        } else {
            21
        }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.nn == 2 {
            &[&[0, 1]]
        } else {
            &[&[0, 1, 2]]
        }
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.stiffness_of(ctx)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        let k = self.stiffness_of(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, !self.shell)?;
        let nd = self.nd();
        let n = nd * self.nn;
        let mut m = vec![0.0; n * n];
        let mt = sec.rho * sec.t;
        if self.nn == 2 {
            // diagonal: halva massan (2π r_mitt L ρ t) per nod
            let p = point(&ctx.x, 2, 0.0);
            let half = 0.5 * 2.0 * PI * p.r * 2.0 * p.j * mt;
            for i in 0..2 {
                for d in 0..2 {
                    m[(nd * i + d) * n + nd * i + d] = half;
                }
            }
            return Ok(m);
        }
        for (xi, w) in gauss(3) {
            let p = point(&ctx.x, 3, xi);
            let da = 2.0 * PI * p.r * p.j * w * mt;
            for i in 0..3 {
                for j in 0..3 {
                    for d in 0..2 {
                        m[(nd * i + d) * n + nd * j + d] += p.n[i] * p.n[j] * da;
                    }
                }
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, !self.shell)?;
        let nd = self.nd();
        let mut f = vec![0.0; nd * self.nn];
        let body = common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, self.name)?;
        if load.kind != "P" && body.is_none() {
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        }
        for (xi, w) in gauss(3) {
            let p = point(&ctx.x, self.nn, xi);
            let da = 2.0 * PI * p.r * p.j * w;
            // positivt tryck P verkar i normalens riktning n = (−t_z, t_r)
            let q = match body {
                Some(b) if load.kind != "P" => [b[0] * sec.t, b[1] * sec.t],
                _ => [load.magnitude * p.nv[0], load.magnitude * p.nv[1]],
            };
            for i in 0..self.nn {
                for c in 0..2 {
                    f[nd * i + c] += p.n[i] * da * q[c];
                }
            }
        }
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let sec = section(ctx, self.name, !self.shell)?;
        let mut v = vec![0.0; self.nn];
        for (xi, w) in gauss(3) {
            let p = point(&ctx.x, self.nn, xi);
            for i in 0..self.nn {
                v[i] += p.n[i] * 2.0 * PI * p.r * p.j * w * sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        self.check(ctx)?;
        let sec = section(ctx, self.name, !self.shell)?;
        let n = self.nd() * self.nn;
        let d2 = [sec.d[0], sec.d[1], sec.d[3], sec.d[4]];
        let tb = sec.t.powi(3) / 12.0;
        let mut s = IpField::new("S", &["S11", "S22", "S33", "S12"]);
        let mut sf = IpField::new("SF", &["SF11", "SF22", "SF33", "SF12"]);
        let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
        for (q, (bm, bb, bs, _da, kss)) in self.ops(ctx, k_shear(&sec), sec.t).into_iter().enumerate() {
            let ip = q + 1;
            let dot = |b: &[f64]| -> f64 { (0..n).map(|p| b[p] * u[p]).sum() };
            let e = [dot(&bm[..n]), dot(&bm[n..])];
            // krökningen redovisas med motsatt tecken (positiv z = −n)
            let k = [-dot(&bb[..n]), -dot(&bb[n..])];
            let dv = |v: [f64; 2]| [d2[0] * v[0] + d2[1] * v[1], d2[2] * v[0] + d2[3] * v[1]];
            let sn = dv(e);
            let mb = dv(k);
            if self.shell {
                sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, 0.0, kss * dot(&bs)]));
                // SM1 = M_θ och SM2 = M_s
                sm.points.push((ip, 0, vec![mb[1] * tb, mb[0] * tb, 0.0]));
                let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
                for sp in sps {
                    let z = sec.section_z(sp);
                    let st = dv([e[0] + z * k[0], e[1] + z * k[1]]);
                    s.points.push((ip, sp, vec![st[0], st[1], 0.0, 0.0]));
                }
            } else {
                sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, 0.0, 0.0]));
                s.points.push((ip, 0, vec![sn[0], sn[1], 0.0, 0.0]));
            }
        }
        if self.shell {
            Ok(vec![s, sf, sm])
        } else {
            Ok(vec![s, sf])
        }
    }
}
