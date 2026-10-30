//! Infinita element CINPE4, CINPS4, CINAX4 och CIN3D8 (statik och frekvens).
//!
//! Formulering (Zienkiewicz m.fl. 1983):
//!
//! - Nod 1–2 (1–4 i 3D) ligger på gränsytan mot de finita elementen (yta 1),
//!   nod 3–4 (5–8) på kanterna mot oändligheten, ovanför nod 2 respektive 1
//!   (5–8 ovanför 1–4). Den yttre noden ska ligga dubbelt så långt från polen
//!   som gränsnoden; polen är därmed implicit 2·x_inre − x_yttre per kant.
//! - Geometrin längs varje kant avbildas med x(s) = −2s/(1−s)·x₁ + (1+s)/(1−s)·x₂,
//!   s ∈ [−1, 1), så att r → ∞ när s → 1 (r = 2a/(1−s) från polen).
//! - Förskjutningen interpoleras kvadratiskt i s med u = 0 i oändligheten:
//!   u = ½s(s−1)·u₁ + (1−s²)·u₂, dvs. termer av ordning 1/r och 1/r².
//! - Tvärs kanterna linjär (2D) respektive bilinjär (3D) interpolation, i
//!   produktform med ovanstående.
//! - 2×2 (2×2×2) gausspunkter; punkterna närmast gränsytan numreras först.
//!
//! Elementen är linjärt elastiska (bara linjära material godtas) och har
//! ingen massa. I *FREQUENCY ger de varken styvhet eller massa (se
//! `frequency_stiffness`); de yttre nodernas modförskjutning blir 0.
//! I direkta dynamiska steg och *STEADY STATE DYNAMICS fungerar infinita
//! element normalt som dämpare (tysta ränder) utan styvhet; det är inte implementerat
//! (elementen ger där sin statiska styvhet).
//!
//! De kvadratiska varianterna (CINPE5R, CINPS5R, CINAX5R, CIN3D12R) saknas:
//! deras interpolation för mittnoderna (som saknar yttre nod) är inte
//! återskapad.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip, DISTORTED};
use crate::fem::assembly::{Prepared, NONE};
use crate::fem::dof::{Dof, SLOTS, U1, U2, U3};
use crate::materials::StressMode;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::{combine, SymCsr, Triplets};

pub struct Infinite {
    name: &'static str,
    mode: StressMode,
}

pub static CINPE4: Infinite = Infinite { name: "CINPE4", mode: StressMode::PlaneStrain };
pub static CINPS4: Infinite = Infinite { name: "CINPS4", mode: StressMode::PlaneStress };
pub static CINAX4: Infinite = Infinite { name: "CINAX4", mode: StressMode::Axisymmetric };
pub static CIN3D8: Infinite = Infinite { name: "CIN3D8", mode: StressMode::ThreeD };

pub static TYPES: &[&dyn ElementType] = &[&CIN3D8, &CINAX4, &CINPE4, &CINPS4];

const FACES2: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
const FACES3: &[&[usize]] = super::continuum3d::HEX8_FACES;

/// Avbildningens och förskjutningens funktioner i den oändliga riktningen:
/// (M₀, M₁, dM₀, dM₁) för geometrin och (L₀, L₁, dL₀, dL₁) för förskjutningen.
fn infinite_1d(s: f64) -> ([f64; 4], [f64; 4]) {
    let q = 1.0 - s;
    (
        [-2.0 * s / q, (1.0 + s) / q, -2.0 / (q * q), 2.0 / (q * q)],
        [0.5 * s * (s - 1.0), 1.0 - s * s, s - 0.5, -2.0 * s],
    )
}

impl Infinite {
    fn three_d(&self) -> bool {
        self.mode == StressMode::ThreeD
    }

    fn nn(&self) -> usize {
        if self.three_d() { 8 } else { 4 }
    }

    fn ndim(&self) -> usize {
        common::ndim(self.mode)
    }

    /// Tvärsfunktioner för gränsnoderna och de yttre noderna i elementets
    /// nodordning: (index för inre noder, index för yttre noder) och
    /// funktioner med derivator i tvärskoordinaterna.
    fn transverse(&self, t: &[f64]) -> (Vec<f64>, Vec<[f64; 2]>) {
        if self.three_d() {
            let s = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let n = s.iter().map(|&(p, q)| 0.25 * (1.0 + p * t[0]) * (1.0 + q * t[1])).collect();
            let d = s.iter().map(|&(p, q)| [0.25 * p * (1.0 + q * t[1]), 0.25 * q * (1.0 + p * t[0])]).collect();
            (n, d)
        } else {
            (vec![0.5 * (1.0 - t[0]), 0.5 * (1.0 + t[0])], vec![[-0.5, 0.0], [0.5, 0.0]])
        }
    }

    /// Nodindex för (inre, yttre) nod i tvärsposition k.
    fn pair(&self, k: usize) -> (usize, usize) {
        if self.three_d() {
            (k, k + 4)
        } else if k == 0 {
            (0, 3)
        } else {
            (1, 2)
        }
    }

    /// Integrationspunkter (gaussvikt × det J × tjocklek eller 2πr) och
    /// B-matriser.
    fn ips(&self, ctx: &ElemCtx) -> Result<(Vec<Ip>, Vec<Vec<f64>>)> {
        let x = &ctx.x;
        let nn = self.nn();
        let g = common::G2;
        let pts: Vec<Vec<f64>> = if self.three_d() {
            let mut v = Vec::new();
            for &s in &[-g, g] {
                for &b in &[-g, g] {
                    for &a in &[-g, g] {
                        v.push(vec![a, b, s]);
                    }
                }
            }
            v
        } else {
            let mut v = Vec::new();
            for &s in &[-g, g] {
                for &a in &[-g, g] {
                    v.push(vec![a, s]);
                }
            }
            v
        };
        let t = match self.mode {
            StressMode::PlaneStress | StressMode::PlaneStrain => ctx.section(self.name)?.solid_thickness(),
            _ => 1.0,
        };
        let pd = if self.three_d() { 3 } else { 2 };
        let mut ips = Vec::new();
        let mut bs = Vec::new();
        for p in pts {
            let s = p[pd - 1];
            let (m, l) = infinite_1d(s);
            let (nt, dnt) = self.transverse(&p[..pd - 1]);
            // geometri- och förskjutningsfunktioner med derivator i (t.., s)
            let mut gn = vec![0.0; nn];
            let mut gd = vec![[0.0; 3]; nn];
            let mut un = vec![0.0; nn];
            let mut ud = vec![[0.0; 3]; nn];
            for k in 0..nt.len() {
                let (i, o) = self.pair(k);
                for (node, f, df, h, dh) in [(i, m[0], m[2], l[0], l[2]), (o, m[1], m[3], l[1], l[3])] {
                    gn[node] = nt[k] * f;
                    un[node] = nt[k] * h;
                    for c in 0..pd - 1 {
                        gd[node][c] = dnt[k][c] * f;
                        ud[node][c] = dnt[k][c] * h;
                    }
                    gd[node][pd - 1] = nt[k] * df;
                    ud[node][pd - 1] = nt[k] * dh;
                }
            }
            let mut j = [[0.0; 3]; 3];
            for r in 0..pd {
                for c in 0..pd {
                    j[r][c] = (0..nn).map(|i| gd[i][r] * x[i][c]).sum();
                }
            }
            let (ji, det) = if pd == 3 {
                common::inv3(j)
            } else {
                let d = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                ([[j[1][1] / d, -j[0][1] / d, 0.0], [-j[1][0] / d, j[0][0] / d, 0.0], [0.0; 3]], d)
            };
            if !(det.abs() > 0.0) || !det.is_finite() {
                return fail(DISTORTED);
            }
            let mut dndx = vec![[0.0; 3]; nn];
            for i in 0..nn {
                for c in 0..pd {
                    dndx[i][c] = (0..pd).map(|r| ji[c][r] * ud[i][r]).sum();
                }
            }
            let factor = if self.mode == StressMode::Axisymmetric {
                let r: f64 = (0..nn).map(|i| gn[i] * x[i][0]).sum();
                2.0 * std::f64::consts::PI * r
            } else {
                t
            };
            let w = det.abs() * factor;
            let b = if self.mode == StressMode::Axisymmetric {
                let r: f64 = (0..nn).map(|i| gn[i] * x[i][0]).sum();
                super::axisym::bmat_axi(&un, &dndx, r)
            } else {
                common::bmat(self.mode, &dndx)
            };
            ips.push(Ip { n: un, dndx, w });
            bs.push(b);
        }
        Ok((ips, bs))
    }

    fn check_material(&self, ctx: &ElemCtx) -> Result<()> {
        let m = ctx.material(self.name)?;
        if !m.is_linear() {
            return fail(format!("element {} ({}): infinita element kan bara ha linjärt elastiskt material (inte {})", ctx.label, self.name, m.name()));
        }
        if ctx.nlgeom {
            return fail(format!("element {} ({}): infinita element är linjära och stöder inte NLGEOM", ctx.label, self.name));
        }
        Ok(())
    }
}

/// Är elementtypen ett infinit element?
pub fn is_infinite(et: &dyn ElementType) -> bool {
    TYPES.iter().any(|t| t.name() == et.name())
}

/// Styvheten för *FREQUENCY: de infinita elementen ger varken
/// styvhet eller massa i egenvärdesanalysen. Elementens styvhet dras därför av från K₀, och
/// frihetsgrader som då saknar styvhet (de yttre noderna) hålls fasta med
/// elementets egen diagonal – de har ingen massa och påverkar inte moderna,
/// och deras modförskjutning blir noll.
pub fn frequency_stiffness(p: &Prepared, idx: &[[usize; SLOTS]], k0: SymCsr) -> Result<SymCsr> {
    let m = p.model;
    let ns = k0.n;
    let mut t = Triplets::new(ns);
    let mut kept = vec![0.0; ns];
    let mut any = false;
    for (e, el) in m.elements.iter().enumerate() {
        if !is_infinite(el.etype) {
            continue;
        }
        any = true;
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let ke = el.etype.stiffness(&ctx).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        let map: Vec<usize> = p.edofs[e].iter().map(|&(n, s)| idx[n][s]).collect();
        let nd = map.len();
        for a in 0..nd {
            let ia = map[a];
            if ia == NONE || ia >= ns {
                continue;
            }
            kept[ia] += ke[a * nd + a];
            for b in 0..nd {
                let ib = map[b];
                if ib != NONE && ib < ns && ia >= ib {
                    t.add(ia, ib, -ke[a * nd + b]);
                }
            }
        }
    }
    if !any {
        return Ok(k0);
    }
    let dk = t.build();
    let k = combine(&[(&k0, 1.0), (&dk, 1.0)], ns);
    let (d0, d1) = (k0.diag(), k.diag());
    let mut fix = Triplets::new(ns);
    for i in 0..ns {
        if kept[i] > 0.0 && d1[i].abs() <= 1e-10 * d0[i].abs() {
            fix.add(i, i, kept[i]);
        }
    }
    Ok(combine(&[(&k, 1.0), (&fix.build(), 1.0)], ns))
}

impl ElementType for Infinite {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        match self.mode {
            StressMode::ThreeD => Family::Continuum3D,
            StressMode::Axisymmetric => Family::Axisymmetric,
            _ => Family::Continuum2D,
        }
    }
    fn dim(&self) -> usize {
        if self.three_d() { 3 } else { 2 }
    }
    fn n_nodes(&self) -> usize {
        self.nn()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.three_d() { &[U1, U2, U3] } else { &[U1, U2] }
    }
    fn n_ips(&self) -> usize {
        if self.three_d() { 8 } else { 4 }
    }
    fn vtk_type(&self) -> u8 {
        if self.three_d() { 12 } else { 9 }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.three_d() { FACES3 } else { FACES2 }
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(self.n_ips(), self.mode, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.check_material(ctx)?;
        let (ips, bs) = self.ips(ctx)?;
        common::stiffness_ctx(ctx, self.mode, &ips, &bs, ctx.material(self.name)?)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        self.check_material(ctx)?;
        let (ips, bs) = self.ips(ctx)?;
        common::response(ctx, self.mode, &ips, &bs, ctx.material(self.name)?, u, du, so, sn, tangent)
    }

    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let n = self.nn() * self.ndim();
        Ok(vec![0.0; n * n])
    }

    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(vec![0.0; self.nn()])
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let (ips, bs) = self.ips(ctx)?;
        Ok(common::output_ctx(ctx, self.mode, &ips, &bs, ctx.material(self.name)?, u, state))
    }
}
