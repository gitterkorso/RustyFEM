//! S8R: åttanodigt dubbelkrökt tjockt skal med reducerad integrering (2×2),
//! sex frihetsgrader per nod (U1–U3, UR1–UR3).
//!
//! Geometri och kinematik (Mindlin–Reissner, isoparametrisk serendipity-
//! interpolation, jfr Bathe kap. 5.4 och Hughes kap. 6):
//!
//! - Nodnormalerna nᵢ: medelvärdet av elementens egna normaler (x,ξ × x,η i
//!   noden) där de skiljer < 20°, eller `*NORMAL` (`elements/normals.rs`).
//!   Drillstraffet i hörnen använder elementets egen normal, i mittnoderna
//!   nodnormalen.
//! - I varje integrationspunkt: normal n = x,ξ × x,η/|…|, lokala axler (e1, e2)
//!   enligt `shell2::axes`, dN/dX i tangentplanet och rotationsvektorn
//!   θ = Σ Nᵢ θᵢ. Direktörens ändring φ = θ × n, med derivatan
//!   φ,α = θ,α × n + θ × n,α där n,α = Σ Nᵢ,α nᵢ.
//! - Membran: εαβ = sym(eα·u,β).
//! - Böjning: καβ = sym(eα·φ,β) plus tre termer i förskjutningsgradienten
//!   H_γβ = e_γ·u,β (Cαβ = n,α·eβ, ω = ½(H21 − H12), a = ½(C12 − C21)):
//!   −½ω·[−(C12 + C21), C12 + C21, 2(C11 − C22)] (Sanders-term),
//!   [(n·n,1) w,1, (n·n,2) w,2, (n·n,1) w,2 + (n·n,2) w,1] (w,β = n·u,β) och
//!   a·[H21, −H12, H22 − H11] (Voigt: κ11, κ22, 2κ12).
//! - Tvärskjuvning: γα = n·u,α + eα·φ med styvheten K·f(det J) i varje punkt
//!   (K = 5/6·G·t, f = slankhetskompensationen med punktens area det J).
//! - Drill: straff kᵢ(θᵢ·nᵢ − ωᵢ)² i varje nod. Hörn: ωᵢ = ½(e2·u,1 − e1·u,2)
//!   i noden, kᵢ = K·det Jᵢ·f(det Jᵢ). Mittnoder: ωᵢ = kordans rotation
//!   (u_q − u_p)·(nᵢ × τ)/L för kanten p→q, kᵢ = K·Ā·f(L²) med Ā = medelvärdet
//!   av hörnens det J.
//! - Massa: konsistent, 3×3 gausspunkter, ρt för förskjutningarna och ρt³/12
//!   för alla tre rotationerna.
//!
//! Tester i `tests/shells2.rs`.

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, cross, dot, norm, scale, sub, G2};
use super::{add_btcb, add_outer, axes, fp, output, unit, GenStrain, Sec};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

/// Naturliga koordinater för noderna (ordning: hörn, sedan mittnoder 1-2, 2-3, 3-4, 4-1).
pub const NAT8: [[f64; 2]; 8] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
/// Kanter (hörn p → q) för mittnoderna 5–8.
const EDGES: [(usize, usize); 4] = [(0, 1), (1, 2), (2, 3), (3, 0)];
/// 2×2 gausspunkter (ξ snabbast).
pub const GP4: [(f64, f64); 4] = [(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)];
const ND: usize = 6;
const NDOF: usize = 48;

/// Formfunktioner och derivator (dN/dξ, dN/dη) för serendipity-elementet.
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

/// Ytans tangentvektorer x,ξ och x,η.
fn tangents(x: &[[f64; 3]], d: &[[f64; 8]; 2]) -> [[f64; 3]; 2] {
    [0, 1].map(|a| [0, 1, 2].map(|c| (0..8).map(|i| d[a][i] * x[i][c]).sum()))
}

/// Nodnormaler ur elementets geometri.
pub fn nodal_normals(x: &[[f64; 3]]) -> [[f64; 3]; 8] {
    NAT8.map(|[s, t]| {
        let (_, d) = shape8(s, t);
        let a = tangents(x, &d);
        unit(cross(a[0], a[1]))
    })
}

/// Data i en punkt (ξ, η) av ytan.
pub struct Pt {
    pub n: [f64; 8],
    /// dN/dX i de lokala axlarna (2 × 8).
    pub b: [[f64; 8]; 2],
    /// Lokala axlar (rader e1, e2, n).
    pub r: [[f64; 3]; 3],
    pub det: f64,
    /// n,X_α (globala vektorer) av de interpolerade nodnormalerna.
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

/// Lägger 3-vektorn `v` (koefficient) på nod i:s förskjutning eller rotation.
fn put(row: &mut [f64], i: usize, off: usize, v: [f64; 3], f: f64) {
    for c in 0..3 {
        row[i * ND + off + c] += f * v[c];
    }
}

/// B-operatorer i en integrationspunkt: membran (3 × 48), böjning (3 × 48)
/// och tvärskjuvning (2 × 48), i punktens lokala axlar.
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
        // membran
        put(&mut bm[0..n], i, 0, e1, b0);
        put(&mut bm[n..2 * n], i, 0, e2, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e1, b1);
        put(&mut bm[2 * n..3 * n], i, 0, e2, b0);
        // böjning från φ,α: eβ·φ,α = θ·(Bα (n × eβ) + N (n,α × eβ))
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
        // förskjutningstermer i krökningen: H[g][b] = e_g·u,b har koefficienten B_b e_g
        // −½ω·om_c med ω = ½(H10 − H01)
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
        // tvärskjuvning γα = n·u,α + eα·φ, eα·(θ × n) = θ·(n × eα)
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
        // område normals: medelvärdesbildade normaler / *NORMAL om sådana finns
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

    /// Drillstraffet: vektorer gᵢ (48) och styvheter kᵢ.
    fn drill(ctx: &ElemCtx, su: &Setup) -> Vec<(Vec<f64>, f64)> {
        let x = &ctx.x;
        let own = nodal_normals(x);
        let sec = &su.sec;
        let k0 = sec.ks[0];
        let mut dc = [0.0; 4];
        let mut out = Vec::with_capacity(8);
        for i in 0..8 {
            let [s, t] = NAT8[i];
            // område normals: hörnen använder elementets egen normal (även när
            // nodnormalerna är medelvärdesbildade), mittnoderna nodnormalen
            let ni = if i < 4 { own[i] } else { su.nrm[i] };
            let mut g = vec![0.0; NDOF];
            put(&mut g, i, 3, ni, 1.0);
            if i < 4 {
                // standardaxlar räcker: ω är oberoende av vridningen i planet
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

    /// Membranets integrationspunkter för NLGEOM (kvadratiska termer).
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

    /// Generaliserade töjningar i integrationspunkterna.
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

/// 3×3 gausspunkter (ξ, η, vikt).
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

/// Ytans area per punkt och normal: (N, normal (enhet), dA·vikt) för 3×3 punkter.
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
            // korotationell formulering (shell2_nlgeom.rs)
            return super::nlgeom::response_s8r(ctx, self.name, &k, &Self::membrane_ops(&su), u, tangent);
        }
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        // membrankrafterna i 2×2 punkter: ∫ N_αβ u,α·u,β dA
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
            // diagonal massa med bevarad total massa (Hinton–Rock–Zienkiewicz)
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
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        }
        for (nv, nrm, da) in surface_points(&ctx.x) {
            // positivt tryck P verkar i normalens riktning (som för S4R)
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
            // membrantöjningens kvadratiska termer i det medföljande systemet
            for (gi, eq) in g.iter_mut().zip(super::nlgeom::quad_strains(&Self::membrane_ops(&su), u)) {
                for r in 0..3 {
                    gi.eps[r] += eq[r];
                }
            }
        }
        Ok(output(&su.sec, &g, false))
    }
}
