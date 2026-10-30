//! Högre ordningens, reducerat integrerade och övriga kontinuumelement.
//!
//! En gemensam motor (`Solid`) för alla isoparametriska kontinuumelement
//! utöver de i `continuum2d.rs`/`continuum3d.rs`:
//!
//! | Element | Form | Integrering | Teknik |
//! |---|---|---|---|
//! | C3D4 | tetraeder, 4 noder | 1 punkt | – |
//! | C3D10 | tetraeder, 10 noder | 4 punkter | – |
//! | C3D6 | kil, 6 noder | 2 punkter (1 × 2) | – |
//! | C3D15 | kil, 15 noder | 9 punkter (3 × 3) | – |
//! | C3D20 | hexaeder, 20 noder | 3×3×3 | – |
//! | C3D20R | hexaeder, 20 noder | 2×2×2 | – |
//! | C3D8R | hexaeder, 8 noder | 1 punkt | hourglass-stabilisering |
//! | C3D8I | hexaeder, 8 noder | 2×2×2 | inkompatibla moder |
//! | CPS6/CPE6 | triangel, 6 noder | 3 punkter | – |
//! | CPS8/CPE8 | fyrhörning, 8 noder | 3×3 | – |
//! | CPS8R/CPE8R | fyrhörning, 8 noder | 2×2 | – |
//! | CPS4R/CPE4R | fyrhörning, 4 noder | 1 punkt | hourglass-stabilisering |
//! | CPS4I/CPE4I | fyrhörning, 4 noder | 2×2 | inkompatibla moder |
//!
//! Axisymmetriska element (CAX*) använder samma motor och finns i `axisym.rs`.
//! Teknikerna (hourglass och inkompatibla moder) finns i
//! `solid_higher_tech.rs`, formfunktioner och regler i `solid_higher_shapes.rs`.
//!
//! Integrationspunkterna numreras med första naturliga koordinaten
//! snabbast och ytorna har .inp-formatets numrering (*DLOAD Pn).

#[path = "solid_higher_shapes.rs"]
pub mod shapes;
#[path = "solid_higher_tech.rs"]
pub mod tech;
#[path = "solid_higher_nlgeom.rs"]
pub mod nlgeom;
#[path = "solid_higher_incompat.rs"]
pub mod incompat;

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip, DISTORTED};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::materials::StressMode;
use crate::model::{fail, Result};
use shapes::{gauss_tensor, tet_rule, tri_rule, wedge_rule, Rule, Shape};

/// Integrationsregel för styvheten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleKind {
    /// n×n(×n) gausspunkter (fyrhörningar och hexaedrar).
    Gauss(usize),
    /// Triangelregel med n punkter.
    Tri(usize),
    /// Tetraederregel med n punkter.
    Tet(usize),
    /// Kil: triangelregel × gaussregel i ζ.
    Wedge(usize, usize),
}

impl RuleKind {
    pub fn points(self, pdim: usize) -> Rule {
        match self {
            RuleKind::Gauss(n) => gauss_tensor(n, pdim),
            RuleKind::Tri(n) => tri_rule(n).into_iter().map(|(p, w)| ([p[0], p[1], 0.0], w)).collect(),
            RuleKind::Tet(n) => tet_rule(n),
            RuleKind::Wedge(a, b) => wedge_rule(a, b),
        }
    }

    pub fn count(self, pdim: usize) -> usize {
        match self {
            RuleKind::Gauss(n) => n.pow(pdim as u32),
            RuleKind::Tri(n) | RuleKind::Tet(n) => n,
            RuleKind::Wedge(a, b) => a * b,
        }
    }
}

/// Elementteknik utöver ren isoparametrisk förskjutningsformulering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tech {
    Full,
    /// Selektivt reducerad integrering (B-bar) av volymtöjningen.
    Bbar,
    /// En integrationspunkt med hourglass-stabilisering.
    Hourglass,
    /// Inkompatibla moder (förstärkt töjning, kondenseras på elementnivå).
    Incompatible,
}

/// Generiskt isoparametriskt kontinuumelement.
pub struct Solid {
    pub name: &'static str,
    pub shape: Shape,
    pub mode: StressMode,
    pub rule: RuleKind,
    pub tech: Tech,
}

const fn s3(name: &'static str, shape: Shape, rule: RuleKind, tech: Tech) -> Solid {
    Solid { name, shape, mode: StressMode::ThreeD, rule, tech }
}
const fn ps(name: &'static str, shape: Shape, rule: RuleKind, tech: Tech) -> Solid {
    Solid { name, shape, mode: StressMode::PlaneStress, rule, tech }
}
const fn pe(name: &'static str, shape: Shape, rule: RuleKind, tech: Tech) -> Solid {
    Solid { name, shape, mode: StressMode::PlaneStrain, rule, tech }
}

pub static C3D4: Solid = s3("C3D4", Shape::Tet4, RuleKind::Tet(1), Tech::Full);
pub static C3D10: Solid = s3("C3D10", Shape::Tet10, RuleKind::Tet(4), Tech::Full);
pub static C3D6: Solid = s3("C3D6", Shape::Wedge6, RuleKind::Wedge(1, 2), Tech::Full);
pub static C3D15: Solid = s3("C3D15", Shape::Wedge15, RuleKind::Wedge(3, 3), Tech::Full);
pub static C3D20: Solid = s3("C3D20", Shape::Hex20, RuleKind::Gauss(3), Tech::Full);
pub static C3D20R: Solid = s3("C3D20R", Shape::Hex20, RuleKind::Gauss(2), Tech::Full);
pub static C3D8R: Solid = s3("C3D8R", Shape::Hex8, RuleKind::Gauss(1), Tech::Hourglass);
pub static C3D8I: Solid = s3("C3D8I", Shape::Hex8, RuleKind::Gauss(2), Tech::Incompatible);
pub static CPS6: Solid = ps("CPS6", Shape::Tri6, RuleKind::Tri(3), Tech::Full);
pub static CPE6: Solid = pe("CPE6", Shape::Tri6, RuleKind::Tri(3), Tech::Full);
pub static CPS8: Solid = ps("CPS8", Shape::Quad8, RuleKind::Gauss(3), Tech::Full);
pub static CPE8: Solid = pe("CPE8", Shape::Quad8, RuleKind::Gauss(3), Tech::Full);
pub static CPS8R: Solid = ps("CPS8R", Shape::Quad8, RuleKind::Gauss(2), Tech::Full);
pub static CPE8R: Solid = pe("CPE8R", Shape::Quad8, RuleKind::Gauss(2), Tech::Full);
pub static CPS4R: Solid = ps("CPS4R", Shape::Quad4, RuleKind::Gauss(1), Tech::Hourglass);
pub static CPE4R: Solid = pe("CPE4R", Shape::Quad4, RuleKind::Gauss(1), Tech::Hourglass);
pub static CPS4I: Solid = ps("CPS4I", Shape::Quad4, RuleKind::Gauss(2), Tech::Incompatible);
pub static CPE4I: Solid = pe("CPE4I", Shape::Quad4, RuleKind::Gauss(2), Tech::Incompatible);

pub static TYPES: &[&dyn ElementType] = &[
    &C3D10, &C3D15, &C3D20, &C3D20R, &C3D4, &C3D6, &C3D8I, &C3D8R, &CPE4I, &CPE4R, &CPE6, &CPE8, &CPE8R, &CPS4I, &CPS4R,
    &CPS6, &CPS8, &CPS8R,
];

/// Integrationspunkt i naturliga koordinater `p` med gaussvikten `gw`.
///
/// Vikten blir det(J)·gw gånger tjockleken `t` (plana element) eller 2πr
/// (axisymmetri, r = interpolerad radie).
pub fn ip_at(shape: Shape, mode: StressMode, x: &[[f64; 3]], p: [f64; 3], gw: f64, t: f64) -> Result<Ip> {
    let (n, dn) = shapes::shape(shape, p);
    let dim = shape.pdim();
    let mut dndx = vec![[0.0; 3]; n.len()];
    let det;
    if dim == 3 {
        let mut j = [[0.0; 3]; 3];
        for r in 0..3 {
            for c in 0..3 {
                j[r][c] = (0..n.len()).map(|i| dn[i][r] * x[i][c]).sum();
            }
        }
        let (ji, d) = common::inv3(j);
        det = d;
        for i in 0..n.len() {
            for c in 0..3 {
                dndx[i][c] = (0..3).map(|r| ji[c][r] * dn[i][r]).sum();
            }
        }
    } else {
        let mut j = [[0.0; 2]; 2];
        for r in 0..2 {
            for c in 0..2 {
                j[r][c] = (0..n.len()).map(|i| dn[i][r] * x[i][c]).sum();
            }
        }
        det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        for i in 0..n.len() {
            dndx[i][0] = (j[1][1] * dn[i][0] - j[0][1] * dn[i][1]) / det;
            dndx[i][1] = (-j[1][0] * dn[i][0] + j[0][0] * dn[i][1]) / det;
        }
    }
    if !(det > 0.0) {
        return fail(DISTORTED);
    }
    let factor = match mode {
        StressMode::ThreeD => 1.0,
        StressMode::Axisymmetric => {
            let r: f64 = n.iter().zip(x).map(|(ni, xi)| ni * xi[0]).sum();
            2.0 * std::f64::consts::PI * r
        }
        _ => t,
    };
    Ok(Ip { n, dndx, w: det * gw * factor })
}

/// Integrationspunkter för en regel.
pub fn ips_for(shape: Shape, mode: StressMode, x: &[[f64; 3]], rule: &Rule, t: f64) -> Result<Vec<Ip>> {
    rule.iter().map(|&(p, w)| ip_at(shape, mode, x, p, w, t)).collect()
}

/// Radie i en integrationspunkt (axisymmetri).
pub fn radius(ip: &Ip, x: &[[f64; 3]]) -> f64 {
    ip.n.iter().zip(x).map(|(ni, xi)| ni * xi[0]).sum()
}

/// B-matris för valfritt spänningstillstånd, inklusive axisymmetri.
pub fn bmat_any(mode: StressMode, ip: &Ip, x: &[[f64; 3]]) -> Vec<f64> {
    match mode {
        StressMode::Axisymmetric => super::axisym::bmat_axi(&ip.n, &ip.dndx, radius(ip, x)),
        _ => common::bmat(mode, &ip.dndx),
    }
}

/// Medelvärdesbildar volymtöjningen (B-bar) över integrationspunkterna.
/// Plan töjning: ε11+ε22; 3D: ε11+ε22+ε33; axisymmetri: se nedan.
pub fn apply_bbar(mode: StressMode, ips: &[Ip], bs: &mut [Vec<f64>]) {
    if bs.is_empty() {
        return;
    }
    let ns = mode.ncomp();
    let nd = bs[0].len() / ns;
    let nv = if mode == StressMode::PlaneStrain { 2 } else { 3 };
    let vol = |b: &[f64], c: usize| (0..nv).map(|r| b[r * nd + c]).sum::<f64>();
    let wsum: f64 = ips.iter().map(|ip| ip.w).sum();
    if mode == StressMode::Axisymmetric {
        // Axisymmetri (CAX4): både
        // den plana volymtöjningen εrr+εzz och ringtöjningen εθθ ersätts med sina
        // volymmedelvärden; εrr−εzz och γrz behålls i punkterna.
        let mean = |f: &dyn Fn(&[f64], usize) -> f64, c: usize| ips.iter().zip(bs.iter()).map(|(ip, b)| ip.w * f(b, c)).sum::<f64>() / wsum;
        let pl = |b: &[f64], c: usize| b[c] + b[nd + c];
        let hoop = |b: &[f64], c: usize| b[2 * nd + c];
        let pavg: Vec<f64> = (0..nd).map(|c| mean(&pl, c)).collect();
        let havg: Vec<f64> = (0..nd).map(|c| mean(&hoop, c)).collect();
        for b in bs.iter_mut() {
            for c in 0..nd {
                let corr = (pavg[c] - pl(b, c)) / 2.0;
                b[c] += corr;
                b[nd + c] += corr;
                b[2 * nd + c] = havg[c];
            }
        }
        return;
    }
    let avg: Vec<f64> = (0..nd).map(|c| ips.iter().zip(bs.iter()).map(|(ip, b)| ip.w * vol(b, c)).sum::<f64>() / wsum).collect();
    for b in bs.iter_mut() {
        for c in 0..nd {
            let corr = (avg[c] - vol(b, c)) / nv as f64;
            for r in 0..nv {
                b[r * nd + c] += corr;
            }
        }
    }
}

/// Volymkraft för *DLOAD BX/BY/BZ/GRAV (och BR/BZ för axisymmetri).
fn body_vector(mode: StressMode, load: &DistLoad, density: f64, elem: &str) -> Result<Option<[f64; 3]>> {
    if mode == StressMode::Axisymmetric {
        match load.kind.as_str() {
            "BR" => return Ok(Some([load.magnitude, 0.0, 0.0])),
            "BZ" => return Ok(Some([0.0, load.magnitude, 0.0])),
            _ => {}
        }
    }
    common::body_load_vector(&load.kind, load.magnitude, &load.extra, density, elem)
}

/// Centrifugallast *DLOAD CENTRIF: storleken är ω², densiteten tas från
/// materialet och volymkraften är ρω²·r⊥ där r⊥ är avståndet från
/// rotationsaxeln (punkt och riktning i datafälten 4–9). För axisymmetriska
/// element är axeln symmetriaxeln (r⊥ = r).
fn centrifugal_load(el: &Solid, ctx: &ElemCtx, load: &DistLoad) -> Result<Vec<f64>> {
    if ctx.density == 0.0 {
        return fail(format!("CENTRIF-lasten på {} kräver *DENSITY i materialet", el.name));
    }
    let e = &load.extra;
    let (a, n) = if el.mode == StressMode::Axisymmetric {
        ([0.0; 3], [0.0, 1.0, 0.0])
    } else {
        let g = |i: usize| e.get(i).copied().unwrap_or(0.0);
        let a = [g(0), g(1), g(2)];
        let mut n = [g(3), g(4), g(5)];
        if el.mode != StressMode::ThreeD && common::norm(n) == 0.0 {
            n = [0.0, 0.0, 1.0];
        }
        let l = common::norm(n);
        if l == 0.0 {
            return fail(format!("CENTRIF-lasten på {} saknar axelriktning", el.name));
        }
        (a, common::scale(n, 1.0 / l))
    };
    let k = ctx.density * load.magnitude;
    let nd = el.ndim();
    let ips = tech::load_ips(el, ctx)?;
    let mut f = vec![0.0; ctx.x.len() * nd];
    for ip in &ips {
        let x: [f64; 3] = [0, 1, 2].map(|c| ip.n.iter().zip(&ctx.x).map(|(ni, xi)| ni * xi[c]).sum());
        let d = common::sub(x, a);
        let rp = common::sub(d, common::scale(n, common::dot(d, n)));
        for (i, ni) in ip.n.iter().enumerate() {
            for c in 0..nd {
                f[i * nd + c] += ni * k * rp[c] * ip.w;
            }
        }
    }
    Ok(f)
}

/// Ekvivalenta nodkrafter för normaltryck `p` (positivt in i materialet) på
/// ytan `face` (index i `shape.faces()`). Ytan integreras med sin egen
/// interpolation, så kvadratiska ytor får konsistenta nodkrafter.
pub fn face_pressure(shape: Shape, mode: StressMode, x: &[[f64; 3]], face: usize, p: f64, t: f64) -> Vec<f64> {
    let fl = shape.faces()[face];
    let fs = shape.face_shape(fl.len());
    let xf: Vec<[f64; 3]> = fl.iter().map(|&i| x[i]).collect();
    let ndim = common::ndim(mode);
    // normal·dA per punkt (före tecken)
    let mut pts: Vec<(Vec<f64>, [f64; 3])> = Vec::new();
    let mut total = [0.0; 3];
    for (q, w) in shapes::face_rule(fs) {
        let (n, dn) = shapes::shape(fs, q);
        let a: [f64; 3] = [0, 1, 2].map(|c| (0..n.len()).map(|i| dn[i][0] * xf[i][c]).sum());
        let nv = if fs.pdim() == 2 {
            let b: [f64; 3] = [0, 1, 2].map(|c| (0..n.len()).map(|i| dn[i][1] * xf[i][c]).sum());
            common::scale(common::cross(a, b), w)
        } else {
            let f = match mode {
                StressMode::Axisymmetric => {
                    let r: f64 = n.iter().zip(&xf).map(|(ni, xi)| ni * xi[0]).sum();
                    2.0 * std::f64::consts::PI * r
                }
                _ => t,
            };
            [a[1] * w * f, -a[0] * w * f, 0.0]
        };
        for c in 0..3 {
            total[c] += nv[c];
        }
        pts.push((n, nv));
    }
    let out = common::sub(common::centroid(&xf), common::centroid(x));
    let sign = if common::dot(out, total) < 0.0 { -1.0 } else { 1.0 };
    let mut f = vec![0.0; x.len() * ndim];
    for (n, nv) in &pts {
        for (k, &i) in fl.iter().enumerate() {
            for d in 0..ndim {
                f[i * ndim + d] -= p * sign * n[k] * nv[d];
            }
        }
    }
    f
}

impl Solid {
    /// Temperature sampling for stress/displacement elements. Compatible
    /// linear solids use the nodal mean. Quadratic quadrilaterals/bricks sample
    /// the full nodal field on the 2-point Gauss grid and interpolate those
    /// samples to a 3-point grid for full integration. This retains midside
    /// temperatures while avoiding a quadratic thermal-strain distribution.
    pub fn temperatures(&self, ctx: &ElemCtx) -> Vec<(f64, f64)> {
        use Shape::*;
        if ctx.temp.is_empty() { return vec![(0.0, 0.0); self.rule.count(self.pdim())]; }
        // CAX4I retains a constant temperature, unlike CPS4I/CPE4I/C3D8I.
        let interpolate = (self.tech == Tech::Incompatible && self.mode != StressMode::Axisymmetric)
            || matches!(self.shape, Tri6 | Quad8 | Tet10 | Wedge15 | Hex20);
        self.rule.points(self.pdim()).iter().map(|&(p, _)| {
            if interpolate {
                let n = if self.rule == RuleKind::Gauss(3) && matches!(self.shape, Quad8 | Hex20) {
                    let mut n = vec![0.0; self.shape.n_nodes()];
                    for (sample, _) in gauss_tensor(2, self.pdim()) {
                        let weight: f64 = (0..self.pdim()).map(|k| 0.5 * (1.0 + p[k] / sample[k])).product();
                        let (ns, _) = shapes::shape(self.shape, sample);
                        for (ni, value) in n.iter_mut().zip(ns) { *ni += weight * value; }
                    }
                    n
                } else { shapes::shape(self.shape, p).0 };
                ctx.temperature_interpolated(&n)
            } else {
                ctx.temperature_at(&vec![0.0; self.shape.n_nodes()])
            }
        }).collect()
    }

    pub fn pdim(&self) -> usize {
        self.shape.pdim()
    }

    pub fn ndim(&self) -> usize {
        common::ndim(self.mode)
    }

    /// Tjocklek för plana element (1 annars).
    pub fn thickness(&self, ctx: &ElemCtx) -> Result<f64> {
        match self.mode {
            StressMode::PlaneStress | StressMode::PlaneStrain => Ok(ctx.section(self.name)?.solid_thickness()),
            _ => Ok(1.0),
        }
    }

    /// Integrationspunkter för styvhet och spänningar.
    pub fn ips(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let t = self.thickness(ctx)?;
        ips_for(self.shape, self.mode, &ctx.x, &self.rule.points(self.pdim()), t)
    }

    /// Integrationspunkter för massa och volymkrafter (noggrann regel).
    pub fn ips_exact(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let t = self.thickness(ctx)?;
        let rule = match self.shape {
            Shape::Tri3 | Shape::Tri6 => RuleKind::Tri(7),
            Shape::Tet4 | Shape::Tet10 => RuleKind::Tet(64),
            Shape::Wedge6 | Shape::Wedge15 => RuleKind::Wedge(7, 4),
            _ => RuleKind::Gauss(4),
        };
        ips_for(self.shape, self.mode, &ctx.x, &rule.points(self.pdim()), t)
    }

    /// B-matriser (med B-bar om tekniken anger det).
    pub fn bmats(&self, x: &[[f64; 3]], ips: &[Ip]) -> Vec<Vec<f64>> {
        let mut bs: Vec<Vec<f64>> = ips.iter().map(|ip| bmat_any(self.mode, ip, x)).collect();
        if self.tech == Tech::Bbar {
            apply_bbar(self.mode, ips, &mut bs);
        }
        bs
    }
}

impl ElementType for Solid {
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
        if self.mode == StressMode::ThreeD { 3 } else { 2 }
    }
    fn n_nodes(&self) -> usize {
        self.shape.n_nodes()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.mode == StressMode::ThreeD { &[U1, U2, U3] } else { &[U1, U2] }
    }
    fn n_ips(&self) -> usize {
        self.rule.count(self.pdim())
    }
    fn vtk_type(&self) -> u8 {
        self.shape.vtk()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.shape.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mat = ctx.material(self.name)?;
        let mut s = common::init_state(self.n_ips(), self.mode, mat);
        if self.tech == Tech::Incompatible {
            s.extend(std::iter::repeat_n(0.0, tech::n_incompatible(self)));
        }
        Ok(s)
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mat = ctx.material(self.name)?;
        match self.tech {
            Tech::Hourglass => tech::hourglass_stiffness(self, ctx, mat),
            Tech::Incompatible => tech::incompatible_stiffness(self, ctx, mat),
            _ => {
                let ips = self.ips(ctx)?;
                let bs = self.bmats(&ctx.x, &ips);
                common::stiffness(self.mode, &ips, &bs, mat)
            }
        }
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let mat = ctx.material(self.name)?;
        if ctx.nlgeom {
            // område skum: CAX4 med hyperelastiskt material som CAX4H (hybrid.rs)
            if let Some(r) = super::hybrid::cax4_hyper(self, ctx, mat, u, so, sn, tangent)? {
                return Ok(r);
            }
            return nlgeom::response(self, ctx, mat, u, du, so, sn, tangent);
        }
        match self.tech {
            Tech::Hourglass => tech::hourglass_response(self, ctx, mat, u, du, so, sn, tangent),
            Tech::Incompatible => tech::incompatible_response(self, ctx, mat, u, du, so, sn, tangent),
            _ => {
                let ips = self.ips(ctx)?;
                let bs = self.bmats(&ctx.x, &ips);
                common::response_with_temperatures(ctx, self.mode, &ips, &bs, mat, u, du, so, sn, tangent, Some(&self.temperatures(ctx)))
            }
        }
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        tech::mass(self, ctx, kind)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        if let Some(num) = load.kind.strip_prefix('P').and_then(|s| s.parse::<usize>().ok()) {
            let nf = self.shape.faces().len();
            if num == 0 || num > nf {
                return fail(format!("{} har bara ytorna P1–P{nf}, inte {}", self.name, load.kind));
            }
            let t = self.thickness(ctx)?;
            return Ok(face_pressure(self.shape, self.mode, &ctx.x, num - 1, load.magnitude, t));
        }
        if load.kind == "CENTRIF" {
            return centrifugal_load(self, ctx, load);
        }
        if let Some(b) = body_vector(self.mode, load, ctx.density, self.name)? {
            return Ok(common::body_force(&tech::load_ips(self, ctx)?, b, self.ndim()));
        }
        not_implemented(&format!("lasttypen {}", load.kind), self.name)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        tech::geometric_stiffness(self, ctx, ctx.material(self.name)?, state)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::volume_weights(&tech::load_ips(self, ctx)?))
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let mat = ctx.material(self.name)?;
        if ctx.nlgeom {
            return nlgeom::output(self, ctx, mat, u, state);
        }
        match self.tech {
            Tech::Incompatible => tech::incompatible_output(self, ctx, mat, u, state),
            Tech::Hourglass => tech::hourglass_output(self, ctx, mat, u, state),
            _ => {
                let ips = self.ips(ctx)?;
                let bs = self.bmats(&ctx.x, &ips);
                Ok(common::output(self.mode, &bs, mat, u, state))
            }
        }
    }
}

/// Alla element som använder motorn (för tester och verktyg).
pub fn solids() -> Vec<&'static Solid> {
    use super::axisym::*;
    vec![
        &C3D4, &C3D10, &C3D6, &C3D15, &C3D20, &C3D20R, &C3D8R, &C3D8I, &CPS6, &CPE6, &CPS8, &CPE8, &CPS8R, &CPE8R, &CPS4R, &CPE4R,
        &CPS4I, &CPE4I, &CAX3, &CAX4, &CAX4R, &CAX4I, &CAX6, &CAX8, &CAX8R,
    ]
}

/// Integrationspunkternas lägen (Σ Nᵢ xᵢ) för elementtypen `name`, eller
/// `None` om typen inte hör till motorn.
pub fn ip_positions(name: &str, ctx: &ElemCtx) -> Option<Vec<[f64; 3]>> {
    let el = solids().into_iter().find(|s| s.name == name)?;
    if el.tech == Tech::Hourglass && el.mode == StressMode::Axisymmetric {
        // CAX4R:s punkt redovisas i areans tyngdpunkt
        let ips = ips_for(el.shape, StressMode::PlaneStrain, &ctx.x, &RuleKind::Gauss(2).points(2), 1.0).ok()?;
        let a: f64 = ips.iter().map(|ip| ip.w).sum();
        let p = [0, 1, 2].map(|c| ips.iter().map(|ip| ip.w * ip.n.iter().zip(&ctx.x).map(|(ni, xi)| ni * xi[c]).sum::<f64>()).sum::<f64>() / a);
        return Some(vec![p]);
    }
    let rule = el.rule.points(el.pdim());
    Some(
        rule.iter()
            .map(|&(p, _)| {
                let (n, _) = shapes::shape(el.shape, p);
                [0, 1, 2].map(|c| n.iter().zip(&ctx.x).map(|(ni, xi)| ni * xi[c]).sum())
            })
            .collect(),
    )
}
