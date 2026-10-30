//! Timoshenkobalkar (skjuvdeformerbara): B21, B22 (plana) och B31, B32 (rumsliga).
//!
//! Formulering (Timoshenko-balk, se t.ex. Bathe kap. 5.4 eller Zienkiewicz–Taylor
//! vol. 2 kap. 10): förskjutningar och rotationer interpoleras oberoende med
//! linjära (B21/B31) eller kvadratiska (B22/B32) formfunktioner. Generaliserade
//! töjningar i lokala axlar (t, n1, n2):
//!
//! - ε = u',  κ1 = θ1',  κ2 = θ2',  τ = θt'  (vridning)
//! - γ1 = v1' − θ2,  γ2 = v2' + θ1  (skjuvning)
//!
//! Integrering längs balken med reducerad gaussregel: 1 punkt för B21/B31,
//! 2 punkter för B22/B32 (samma punkter som för utdata).
//!
//! Tvärsnittets skjuvstyvhet är k·G·A med formfaktorn k per tvärsnittsform
//! (eller värdena från *TRANSVERSE SHEAR STIFFNESS). Den skalas
//! med en slankhetskompensation f = 1/(1 + ξ·l²·A/(12·I)), där l är
//! elementlängden och ξ = 0,25 för linjära element. För kvadratiska element är
//! korrektionen mycket liten (ξ·l² = 6,25·10⁻⁶·l²).
//!
//! Massa: B21/B31 har diagonal massa (translation ρAL/2, rotationströghet
//! ρ·I·L/2 per nod), B22/B32 konsistent massa med rotationströghet.
//! NLGEOM och geometrisk styvhet (alla snittkrafter) i
//! `beam_timoshenko_nlgeom.rs`.
//!
//! Laster (*DLOAD): PX, PY, PZ (globala, per längdenhet), P1, P2 (i tvärsnittets
//! 1- och 2-riktning), GRAV. Utdata: SF (SF1 = normalkraft, SF2/SF3 tvärkrafter)
//! och SM (SM1, SM2 böjmoment, SM3 vridmoment) per integrationspunkt.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
#[path = "beam_timoshenko_section.rs"]
pub mod section;
#[path = "beam_timoshenko_nlgeom.rs"]
pub mod nlgeom;
#[path = "beam_timoshenko_plastic.rs"]
pub mod plastic;
// område gaps1: *BEAM GENERAL SECTION, SECTION=NONLINEAR GENERAL
#[path = "beam_nonlinear_general.rs"]
mod nlg;
// område coupled2: skjuvdelen av styvheten för materialdämpning i modala steg
#[path = "beam_timoshenko_damping.rs"]
pub mod damping;

use self::section::{section_props, SectionProps};
use super::common::{self, cross, norm, scale, sub};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct TimoBeam {
    name: &'static str,
    dim: usize,
    nn: usize,
}

pub static B21: TimoBeam = TimoBeam { name: "B21", dim: 2, nn: 2 };
pub static B22: TimoBeam = TimoBeam { name: "B22", dim: 2, nn: 3 };
pub static B31: TimoBeam = TimoBeam { name: "B31", dim: 3, nn: 2 };
pub static B32: TimoBeam = TimoBeam { name: "B32", dim: 3, nn: 3 };

pub static TYPES: &[&dyn ElementType] = &[&B21, &B22, &B31, &B32];

impl TimoBeam {
    /// Område beams2: varianter med eget namn (hybridbalkarna B21H, ...).
    pub const fn named(name: &'static str, dim: usize, nn: usize) -> TimoBeam {
        TimoBeam { name, dim, nn }
    }
}

/// Gausspunkter (ξ, vikt) på [−1, 1].
pub(crate) fn gauss(n: usize) -> &'static [(f64, f64)] {
    const G1: &[(f64, f64)] = &[(0.0, 2.0)];
    const G2P: &[(f64, f64)] = &[(-common::G2, 1.0), (common::G2, 1.0)];
    const G3: &[(f64, f64)] = &[(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];
    match n {
        1 => G1,
        2 => G2P,
        _ => G3,
    }
}

/// Formfunktioner och derivator (d/dξ) för 2 eller 3 noder. Nodordning:
/// längs balken, för B22/B32 är nod 2 mittnoden.
pub(crate) fn shape(nn: usize, xi: f64) -> (Vec<f64>, Vec<f64>) {
    if nn == 2 {
        (vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], vec![-0.5, 0.5])
    } else {
        (vec![0.5 * xi * (xi - 1.0), 1.0 - xi * xi, 0.5 * xi * (xi + 1.0)], vec![xi - 0.5, -2.0 * xi, xi + 0.5])
    }
}

/// Geometri och tvärsnitt för ett element.
pub struct BeamGeom {
    /// Lokala axlar [t, n1, n2].
    pub axes: [[f64; 3]; 3],
    /// Längd mellan ändnoderna.
    pub l: f64,
    /// |dx/dξ| per integrationspunkt i en given regel beräknas i `jac`.
    pub x: Vec<[f64; 3]>,
    pub sec: SectionProps,
}

impl BeamGeom {
    fn jac(&self, nn: usize, xi: f64) -> f64 {
        let (_, dn) = shape(nn, xi);
        let mut d = [0.0; 3];
        for (i, p) in self.x.iter().enumerate() {
            for k in 0..3 {
                d[k] += dn[i] * p[k];
            }
        }
        norm(d)
    }
}

/// Lokala axlar: n1 ungefärlig (standard (0, 0, −1)), n2 = t × n1.
pub fn beam_axes(t: [f64; 3], n1a: Option<[f64; 3]>, dim: usize, label: i64, name: &str) -> Result<[[f64; 3]; 3]> {
    let n1a = if dim == 2 { [0.0, 0.0, -1.0] } else { n1a.unwrap_or([0.0, 0.0, -1.0]) };
    let n2 = cross(t, n1a);
    let ln2 = norm(n2);
    if ln2 < 1e-6 * norm(n1a).max(1e-300) {
        return fail(format!(
            "element {label} ({name}) är parallellt med tvärsnittets n1-riktning – ange n1 på tvärsnittets andra datarad"
        ));
    }
    let n2 = scale(n2, 1.0 / ln2);
    let n1 = cross(n2, t);
    Ok([t, n1, n2])
}

impl TimoBeam {
    fn quadratic(&self) -> bool {
        self.nn == 3
    }

    pub(crate) fn n_gauss(&self) -> usize {
        if self.quadratic() { 2 } else { 1 }
    }

    pub fn geom(&self, ctx: &ElemCtx) -> Result<BeamGeom> {
        let sec = ctx.section(self.name)?;
        let props = section_props(sec, ctx, self.name)?;
        let a = ctx.x[0];
        let b = ctx.x[self.nn - 1];
        let d = sub(b, a);
        let l = norm(d);
        if l == 0.0 {
            return fail(format!("element {} ({}) har ingen längd", ctx.label, self.name));
        }
        if self.dim == 2 && (d[2].abs() > 1e-12 * l) {
            return fail(format!("element {} ({}): plana balkar måste ligga i x–y-planet", ctx.label, self.name));
        }
        let t = scale(d, 1.0 / l);
        let n1 = sec.beam().and_then(|b| b.n1);
        let axes = beam_axes(t, n1, self.dim, ctx.label, self.name)?;
        // område beams3: nodnormaler vridna kring axeln ger medelvinkeln (beam_twist.rs)
        let (axes, _) = super::beam_twist::mean_axes(ctx, self.nn, self.dim, axes, l);
        Ok(BeamGeom { axes, l, x: ctx.x.clone(), sec: props })
    }

    /// Skjuvstyvheter (γ1, γ2) med slankhetskompensation.
    pub(crate) fn shear_stiffness(&self, g: &BeamGeom) -> (f64, f64) {
        let s = &g.sec;
        let xi_l2 = if self.quadratic() { 6.25e-6 * g.l * g.l } else { 0.25 * g.l * g.l };
        let fp = |i: f64| if i > 0.0 { 1.0 / (1.0 + xi_l2 * s.a / (12.0 * i)) } else { 1.0 };
        // område beams2: tröghetsmoment för kompensationen (I, HEX) och
        // hybridbalkarnas skjuvstyvhet (kondenserad matris: faktor 1 + 10⁻⁶)
        let (i22, i11) = (if s.i22s > 0.0 { s.i22s } else { s.i22c }, if s.i11s > 0.0 { s.i11s } else { s.i11c });
        let hyb = if self.name.ends_with('H') { 1.0 + 1e-6 } else { 1.0 };
        // γ1 (utböjning längs n1) hör ihop med böjning kring n2 (I22), γ2 med I11.
        (hyb * s.k13 * fp(i22), hyb * s.k23 * fp(i11))
    }

    /// Tvärsnittets styvhetsmatris (6×6) för [ε, κ1, κ2, τ, γ1, γ2].
    pub(crate) fn section_matrix(&self, g: &BeamGeom) -> [[f64; 6]; 6] {
        let s = &g.sec;
        let (k1, k2) = self.shear_stiffness(g);
        let e = s.e;
        let mut c = [[0.0; 6]; 6];
        // axialkraft och böjning kring referensaxeln (origo i tvärsnittet)
        c[0][0] = e * s.a;
        c[0][1] = e * s.s1;
        c[0][2] = -e * s.s2;
        c[1][1] = e * s.i11;
        c[1][2] = -e * s.i12;
        c[2][2] = e * s.i22;
        // vridning kring skjuvcentrum och skjuvning
        let (x1s, x2s) = (s.x1s, s.x2s);
        c[3][3] = s.g * s.j + k1 * x2s * x2s + k2 * x1s * x1s;
        c[3][4] = -k1 * x2s;
        c[3][5] = k2 * x1s;
        c[4][4] = k1;
        c[5][5] = k2;
        for i in 0..6 {
            for j in 0..i {
                c[i][j] = c[j][i];
            }
        }
        c
    }

    /// B-matris (6 × 6nn) i lokala frihetsgrader [u, v1, v2, θt, θ1, θ2] per nod.
    pub(crate) fn bmat(&self, g: &BeamGeom, xi: f64) -> (Vec<f64>, f64) {
        let nn = self.nn;
        let (n, dn) = shape(nn, xi);
        let j = g.jac(nn, xi);
        let nd = 6 * nn;
        let mut b = vec![0.0; 6 * nd];
        for i in 0..nn {
            let dx = dn[i] / j;
            let c = 6 * i;
            b[0 * nd + c] = dx; // ε = u'
            b[1 * nd + c + 4] = dx; // κ1 = θ1'
            b[2 * nd + c + 5] = dx; // κ2 = θ2'
            b[3 * nd + c + 3] = dx; // τ = θt'
            b[4 * nd + c + 1] = dx; // γ1 = v1' − θ2
            b[4 * nd + c + 5] = -n[i];
            b[5 * nd + c + 2] = dx; // γ2 = v2' + θ1
            b[5 * nd + c + 4] = n[i];
        }
        (b, j)
    }

    /// Transformation globalt → lokalt för 6 frihetsgrader per nod.
    pub(crate) fn transform(&self, g: &BeamGeom) -> Vec<f64> {
        common::block_diag(&g.axes, 2 * self.nn)
    }

    /// Lokal styvhet (6nn × 6nn).
    fn local_stiffness(&self, g: &BeamGeom) -> Vec<f64> {
        let nd = 6 * self.nn;
        let c = self.section_matrix(g);
        let mut k = vec![0.0; nd * nd];
        for &(xi, w) in gauss(self.n_gauss()) {
            let (b, j) = self.bmat(g, xi);
            let wj = w * j;
            // cb = C B
            let mut cb = vec![0.0; 6 * nd];
            for r in 0..6 {
                for s in 0..6 {
                    let crs = c[r][s];
                    if crs != 0.0 {
                        for q in 0..nd {
                            cb[r * nd + q] += crs * b[s * nd + q];
                        }
                    }
                }
            }
            for r in 0..6 {
                for p in 0..nd {
                    let brp = b[r * nd + p];
                    if brp != 0.0 {
                        for q in 0..nd {
                            k[p * nd + q] += wj * brp * cb[r * nd + q];
                        }
                    }
                }
            }
        }
        k
    }

    /// Plockar ut de plana frihetsgraderna (u1, u2, ur3) ur en 3D-matris/vektor.
    fn planar_index(&self) -> Vec<usize> {
        (0..self.nn).flat_map(|i| [6 * i, 6 * i + 1, 6 * i + 5]).collect()
    }

    fn reduce_matrix(&self, k: &[f64]) -> Vec<f64> {
        if self.dim == 3 {
            return k.to_vec();
        }
        let nd = 6 * self.nn;
        let idx = self.planar_index();
        let m = idx.len();
        let mut out = vec![0.0; m * m];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                out[a * m + b] = k[i * nd + j];
            }
        }
        out
    }

    fn expand_vector(&self, u: &[f64]) -> Vec<f64> {
        if self.dim == 3 {
            return u.to_vec();
        }
        let mut v = vec![0.0; 6 * self.nn];
        for (a, &i) in self.planar_index().iter().enumerate() {
            v[i] = u[a];
        }
        v
    }

    fn reduce_vector(&self, f: &[f64]) -> Vec<f64> {
        if self.dim == 3 {
            return f.to_vec();
        }
        self.planar_index().iter().map(|&i| f[i]).collect()
    }

    fn global_stiffness(&self, g: &BeamGeom) -> Vec<f64> {
        let t = self.transform(g);
        let kl = self.local_stiffness(g);
        common::congruence(&kl, &t, 6 * self.nn)
    }

    /// Inre kraft och tangent med sektionspunkter (område struct_nonlin,
    /// beam_timoshenko_plastic.rs), geometriskt linjärt.
    fn plastic_response(&self, ctx: &ElemCtx, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let g = self.geom(ctx)?;
        let fib = plastic::Fibers::new(ctx, self.name)?;
        let ng = self.n_gauss();
        let len = fib.ip_len();
        let init;
        let so = if so.len() == ng * len {
            so
        } else {
            init = fib.init_state(ng);
            &init[..]
        };
        sn.clear();
        sn.extend_from_slice(so);
        let t = self.transform(&g);
        let ul = common::matvec(&t, &self.expand_vector(u));
        let c = self.section_matrix(&g);
        let nd = 6 * self.nn;
        let mut fl = vec![0.0; nd];
        let mut kl = vec![0.0; nd * nd];
        for (q, &(xi, w)) in gauss(ng).iter().enumerate() {
            let (b, j) = self.bmat(&g, xi);
            let e: [f64; 6] = std::array::from_fn(|r| (0..nd).map(|k| b[r * nd + k] * ul[k]).sum());
            let (nsh, _) = shape(self.nn, xi);
            let (s, ct) = fib.response(ctx, &c, &e, &nsh, &so[q * len..(q + 1) * len], &mut sn[q * len..(q + 1) * len])?;
            let wj = w * j;
            for p in 0..nd {
                fl[p] += wj * (0..6).map(|r| b[r * nd + p] * s[r]).sum::<f64>();
            }
            if tangent {
                let mut cb = vec![0.0; 6 * nd];
                for r in 0..6 {
                    for s2 in 0..6 {
                        if ct[r][s2] != 0.0 {
                            for k in 0..nd {
                                cb[r * nd + k] += ct[r][s2] * b[s2 * nd + k];
                            }
                        }
                    }
                }
                for r in 0..6 {
                    for p in 0..nd {
                        let brp = b[r * nd + p];
                        if brp != 0.0 {
                            for k in 0..nd {
                                kl[p * nd + k] += wj * brp * cb[r * nd + k];
                            }
                        }
                    }
                }
            }
        }
        let fg: Vec<f64> = (0..nd).map(|jj| (0..nd).map(|i| t[i * nd + jj] * fl[i]).sum()).collect();
        let k = tangent.then(|| self.reduce_matrix(&common::congruence(&kl, &t, nd)));
        Ok(ElemResponse { fint: self.reduce_vector(&fg), k })
    }

    /// Snittkrafter [N, M1, M2, T, V1, V2] per integrationspunkt.
    pub fn section_forces(&self, ctx: &ElemCtx, u: &[f64]) -> Result<Vec<[f64; 6]>> {
        let g = self.geom(ctx)?;
        let t = self.transform(&g);
        let ul = common::matvec(&t, &self.expand_vector(u));
        let mut c = self.section_matrix(&g);
        // område beams3: koppling axialtöjning–vridning för vridna B31 (beam_twist.rs)
        let (_, tau) = super::beam_twist::mean_axes(ctx, self.nn, self.dim, g.axes, g.l);
        if tau != 0.0 && g.sec.a > 0.0 {
            let cp = g.sec.e * (g.sec.i11 + g.sec.i22) * tau;
            c[0][3] += cp;
            c[3][0] += cp;
            c[3][3] += cp * cp / (g.sec.e * g.sec.a);
        }
        let nd = 6 * self.nn;
        let mut out = Vec::new();
        for &(xi, _) in gauss(self.n_gauss()) {
            let (b, _) = self.bmat(&g, xi);
            let e: Vec<f64> = (0..6).map(|r| (0..nd).map(|q| b[r * nd + q] * ul[q]).sum()).collect();
            let mut f = [0.0; 6];
            for r in 0..6 {
                f[r] = (0..6).map(|s| c[r][s] * e[s]).sum();
            }
            out.push(f);
        }
        Ok(out)
    }
}

impl ElementType for TimoBeam {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.dim == 2 { &[U1, U2, UR3] } else { &[U1, U2, U3, UR1, UR2, UR3] }
    }
    fn n_ips(&self) -> usize {
        self.n_gauss()
    }
    fn vtk_type(&self) -> u8 {
        if self.quadratic() { 21 } else { 3 }
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if nlg::is_nlg(ctx) {
            return Ok(self.nlg_init_state());
        }
        if plastic::is_plastic(ctx) {
            return Ok(plastic::Fibers::new(ctx, self.name)?.init_state(self.n_gauss()));
        }
        Ok(Vec::new())
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if nlg::is_nlg(ctx) {
            return self.nlg_stiffness(ctx);
        }
        let g = self.geom(ctx)?;
        let mut k = self.global_stiffness(&g);
        // område beams3: koppling axialtöjning–vridning för vridna B31 (beam_twist.rs)
        let (_, tau) = super::beam_twist::mean_axes(ctx, self.nn, self.dim, g.axes, g.l);
        if tau != 0.0 {
            let s = &g.sec;
            let kc = super::beam_twist::coupling(&g.axes, g.l, tau, s.e * s.a, s.e * (s.i11 + s.i22));
            for (a, b) in k.iter_mut().zip(&kc) {
                *a += b;
            }
        }
        Ok(self.reduce_matrix(&k))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // område nlgeom_struct: geometriskt exakt balk (beam_timoshenko_nlgeom.rs)
            return nlgeom::response(self, ctx, u, _du, _so, _sn, tangent);
        }
        if nlg::is_nlg(ctx) {
            return self.nlg_response(ctx, u, _so, _sn, tangent);
        }
        if plastic::is_plastic(ctx) {
            return self.plastic_response(ctx, u, _so, _sn, tangent);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let g = self.geom(ctx)?;
        let s = &g.sec;
        let rho = s.rho;
        let nd = 6 * self.nn;
        // lokal massmatris per nodpar: translation ρA, rotation ρ·[Ip, I11, I22],
        // koppling translation–rotation från tyngdpunktens läge.
        let block = |w: f64| -> [[f64; 6]; 6] {
            let mut m = [[0.0; 6]; 6];
            let ma = rho * s.a * w;
            for d in 0..3 {
                m[d][d] = ma;
            }
            // r_c = c1 n1 + c2 n2; rotationens bidrag till hastighet: θ × r_c
            let (c1, c2) = (s.c1, s.c2);
            // u_t += θ1 c2 − θ2 c1 ; v1 += −θt c2 ; v2 += θt c1
            let mut cpl = [[0.0; 3]; 3]; // translation (rad) × rotation (kolumn)
            cpl[0][1] = c2;
            cpl[0][2] = -c1;
            cpl[1][0] = -c2;
            cpl[2][0] = c1;
            for i in 0..3 {
                for j in 0..3 {
                    m[i][3 + j] = ma * cpl[i][j];
                    m[3 + j][i] = ma * cpl[i][j];
                }
            }
            m[3][3] = rho * (s.i11 + s.i22) * w;
            m[4][4] = rho * s.i11 * w;
            m[5][5] = rho * s.i22 * w;
            m[4][5] = -rho * s.i12 * w;
            m[5][4] = m[4][5];
            m
        };
        let mut ml = vec![0.0; nd * nd];
        let lumped = !self.quadratic() || kind == MassKind::Lumped;
        if lumped {
            // ∫N dx per nod
            let mut wsum = vec![0.0; self.nn];
            for &(xi, w) in gauss(3) {
                let (n, _) = shape(self.nn, xi);
                let j = g.jac(self.nn, xi);
                for i in 0..self.nn {
                    wsum[i] += n[i] * w * j;
                }
            }
            for i in 0..self.nn {
                let b = block(wsum[i]);
                for r in 0..6 {
                    for c in 0..6 {
                        ml[(6 * i + r) * nd + 6 * i + c] = b[r][c];
                    }
                }
            }
        } else {
            for &(xi, w) in gauss(3) {
                let (n, _) = shape(self.nn, xi);
                let j = g.jac(self.nn, xi);
                let b = block(w * j);
                for p in 0..self.nn {
                    for q in 0..self.nn {
                        let f = n[p] * n[q];
                        for r in 0..6 {
                            for c in 0..6 {
                                ml[(6 * p + r) * nd + 6 * q + c] += f * b[r][c];
                            }
                        }
                    }
                }
            }
        }
        let t = self.transform(&g);
        let mut mg = common::congruence(&ml, &t, nd);
        // område beams3: rotationströgheten i nodernas egna riktningar (beam_twist.rs)
        if lumped {
            if let Some(jr) = super::beam_twist::rotary(ctx, self.nn, self.dim, rho, [s.i11 + s.i22, s.i11, s.i22]) {
                for (i, ji) in jr.iter().enumerate() {
                    for r in 0..3 {
                        for c in 0..3 {
                            mg[(6 * i + 3 + r) * nd + 6 * i + 3 + c] = ji[r][c];
                        }
                    }
                }
            }
        }
        Ok(self.reduce_matrix(&mg))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        // område nlgeom_struct: spänningsdelen av NLGEOM-tangenten (normalkraft,
        // tvärkrafter, moment och vridning; beam_timoshenko_nlgeom.rs)
        let forces = self.section_forces(ctx, u)?;
        nlgeom::geometric(self, ctx, &forces)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let g = self.geom(ctx)?;
        let q = match load.kind.as_str() {
            "PX" => [load.magnitude, 0.0, 0.0],
            "PY" => [0.0, load.magnitude, 0.0],
            "PZ" => [0.0, 0.0, load.magnitude],
            "P1" => scale(g.axes[1], load.magnitude),
            "P2" => scale(g.axes[2], load.magnitude),
            "GRAV" => {
                let b = common::body_load_vector("GRAV", load.magnitude, &load.extra, g.sec.rho, self.name)?.unwrap();
                scale(b, g.sec.a)
            }
            _ => return not_implemented(&format!("lasttypen {}", load.kind), self.name),
        };
        if self.dim == 2 && q[2] != 0.0 {
            return fail(format!("{}: last i z-led på en plan balk", self.name));
        }
        let mut f = vec![0.0; 6 * self.nn];
        for &(xi, w) in gauss(3) {
            let (n, _) = shape(self.nn, xi);
            let j = g.jac(self.nn, xi);
            for i in 0..self.nn {
                for d in 0..3 {
                    f[6 * i + d] += n[i] * q[d] * w * j;
                }
            }
        }
        Ok(self.reduce_vector(&f))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let g = self.geom(ctx)?;
        let mut v = vec![0.0; self.nn];
        for &(xi, w) in gauss(3) {
            let (n, _) = shape(self.nn, xi);
            let j = g.jac(self.nn, xi);
            for i in 0..self.nn {
                v[i] += n[i] * g.sec.a * w * j;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        // område struct_nonlin: med sektionspunkter ur det lagrade tillståndet
        let fib = if plastic::is_plastic(ctx) { Some(plastic::Fibers::new(ctx, self.name)?) } else { None };
        let ng = self.n_gauss();
        let pst: Option<Vec<f64>> = fib.as_ref().map(|fb| {
            let n = ng * fb.ip_len();
            if _state.len() >= n { _state[..n].to_vec() } else { fb.init_state(ng) }
        });
        let f = match (&fib, &pst) {
            (Some(fb), Some(st)) => {
                let g = self.geom(ctx)?;
                let c = if ctx.nlgeom { nlgeom::current_c_for(self, &g, u) } else { self.section_matrix(&g) };
                (0..ng).map(|q| fb.forces_from_state(&c, &st[q * fb.ip_len()..(q + 1) * fb.ip_len()])).collect()
            }
            _ if nlg::is_nlg(ctx) => self.nlg_forces(ctx, u, _state)?,
            _ if ctx.nlgeom => nlgeom::section_forces(self, ctx, u, _state)?,
            _ => self.section_forces(ctx, u)?,
        };
        // böjmomenten redovisas kring tyngdpunkten och vridmomentet
        // kring skjuvcentrum (snittkrafterna ovan gäller referensaxeln)
        let s = self.geom(ctx)?.sec;
        let (c1, c2) = (s.c1, s.c2);
        let (x1s, x2s) = (s.x1s, s.x2s);
        let (mut sf, mut sm) = if self.dim == 2 {
            // för plana balkar heter komponenterna SF11, SF22, SF12
            // (normalkraft, tvärkraft, 0) och SM1, SM2, SM3 (moment, 0, 0).
            (IpField::new("SF", &["SF11", "SF22", "SF12"]), IpField::new("SM", &["SM1", "SM2", "SM3"]))
        } else {
            (IpField::new("SF", &["SF1", "SF2", "SF3"]), IpField::new("SM", &["SM1", "SM2", "SM3"]))
        };
        for (q, v) in f.iter().enumerate() {
            let [n, m1, m2, tq, v1, v2] = *v;
            let (m1, m2, tq) = (m1 - n * c2, m2 + n * c1, tq + x2s * v1 - x1s * v2);
            if self.dim == 2 {
                sf.points.push((q + 1, 0, vec![n, v2, 0.0]));
                sm.points.push((q + 1, 0, vec![m1, 0.0, 0.0]));
            } else {
                sf.points.push((q + 1, 0, vec![n, v2, v1]));
                sm.points.push((q + 1, 0, vec![m1, m2, tq]));
            }
        }
        let mut out = vec![sf, sm];
        if let (Some(fb), Some(st)) = (&fib, &pst) {
            let states: Vec<&[f64]> = (0..ng).map(|q| &st[q * fb.ip_len()..(q + 1) * fb.ip_len()]).collect();
            out.extend(fb.point_output(&states));
        }
        Ok(out)
    }
}
