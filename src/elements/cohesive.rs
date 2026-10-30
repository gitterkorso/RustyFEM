//! Kohesiva element med traction–separation: COH2D4 (plant),
//! COHAX4 (axisymmetriskt), COH3D8 och COH3D6, med *COHESIVE SECTION,
//! RESPONSE=TRACTION SEPARATION och materialet i `materials/cohesive.rs`.
//!
//! Nodordning: undersidans noder först, sedan översidans (COH2D4/COHAX4:
//! 1–2 under, 4 över 1 och 3 över 2; COH3D8: 1–4 under, 5–8 över;
//! COH3D6: 1–3 under, 4–6 över). Mittytan är medelvärdet av under- och
//! översidan. Formulering:
//!
//! - Integrationspunkterna ligger i noderna (Newton–Cotes) med vikterna
//!   ∫Nᵢ dA över mittytan i referensläget (2πr ingår för COHAX4).
//! - Normalseparationen δn i punkt i är (xᵢ,över − xᵢ,under)·n med normalen n
//!   till mittytan i punkten.
//! - Skjuvseparationerna är antagna (som MITC-skal): COH2D4/COHAX4 har en
//!   konstant skjuvseparation från elementets mitt; COH3D8 tar den
//!   kovarianta komponenten längs ξ i (0, ηᵢ) och längs η i (ξᵢ, 0); COH3D6
//!   knyter de kovarianta komponenterna i kanternas mittpunkter som MITC3.
//!   Komponenten längs gᵢ är Δx·gᵢ med separationen Δx och mittytans
//!   tangent gᵢ i knytpunkten, så elementet är rotationsoberoende
//!   (skjuvningen ger ett moment med tjockleken h som hävarm).
//! - Lokala riktningar: 3 = normalen, 1 = globala X projicerad på mittytan
//!   (Z om X ligger inom 0,1° från normalen), 2 = n × 1. Vid NLGEOM roteras
//!   riktningarna med rotationen ur polärdekompositionen av mittytans
//!   deformationsgradient i punkten; för plana element är 1 = tangenten.
//! - Separationerna mäts från referensläget (δ = δ(x) − δ(X)) och
//!   töjningarna är nominella, ε = δ/T0 (S, E; LE vid NLGEOM).
//! - Utan NLGEOM är δ linjär i u (derivatan i referensläget), med NLGEOM
//!   räknas δ(x) i aktuellt läge med vikterna i referensläget; kraft och
//!   tangent (med geometriska termen Σ tₖ ∂²δₖ/∂x²) fås exakt med
//!   hyperduala tal (`kinematics/hyperdual.rs`).
//!
//! Utdata: S (S22, S12 i 2D; S33, S13, S23 i 3D), E eller LE, SDEG,
//! initieringskriteriet (MAXSCRT, QUADSCRT, MAXECRT, QUADECRT) och STATUS
//! (0 när alla punkter har D = 1 och ingen är i tryck; elementet tas då bort
//! och töjningen redovisas som vid borttagningen).

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::inp::keywords::cohesive::section_values;
use crate::kinematics::hyperdual::Hd;
use crate::materials::cohesive::TractionSeparation;
use crate::model::{fail, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    /// Plant tvånodigt mittlinjeelement (COH2D4).
    Line,
    /// Axisymmetriskt (COHAX4).
    LineAxi,
    /// Fyrhörning (COH3D8).
    Quad,
    /// Triangel (COH3D6).
    Tri,
}

pub struct Cohesive {
    name: &'static str,
    shape: Shape,
}

pub static COH2D4: Cohesive = Cohesive { name: "COH2D4", shape: Shape::Line };
pub static COHAX4: Cohesive = Cohesive { name: "COHAX4", shape: Shape::LineAxi };
pub static COH3D8: Cohesive = Cohesive { name: "COH3D8", shape: Shape::Quad };
pub static COH3D6: Cohesive = Cohesive { name: "COH3D6", shape: Shape::Tri };

pub static TYPES: &[&dyn ElementType] = &[&COH2D4, &COH3D6, &COH3D8, &COHAX4];

type V<const M: usize> = [Hd<M>; 3];

fn c3<const M: usize>(a: [f64; 3]) -> V<M> {
    [Hd::cst(a[0]), Hd::cst(a[1]), Hd::cst(a[2])]
}
fn add<const M: usize>(a: &V<M>, b: &V<M>) -> V<M> {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub<const M: usize>(a: &V<M>, b: &V<M>) -> V<M> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scl<const M: usize>(a: &V<M>, s: f64) -> V<M> {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn mulv<const M: usize>(a: &V<M>, s: Hd<M>) -> V<M> {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot<const M: usize>(a: &V<M>, b: &V<M>) -> Hd<M> {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross<const M: usize>(a: &V<M>, b: &V<M>) -> V<M> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit<const M: usize>(a: &V<M>) -> V<M> {
    let r = dot(a, a).sqrt().recip();
    mulv(a, r)
}
fn val<const M: usize>(a: &V<M>) -> [f64; 3] {
    [a[0].v, a[1].v, a[2].v]
}

const QNAT: [(f64, f64); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
const TNAT: [(f64, f64); 3] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];

fn qshape(x: f64, e: f64) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let mut n = [0.0; 4];
    let mut dx = [0.0; 4];
    let mut de = [0.0; 4];
    for (i, &(a, b)) in QNAT.iter().enumerate() {
        n[i] = 0.25 * (1.0 + x * a) * (1.0 + e * b);
        dx[i] = 0.25 * a * (1.0 + e * b);
        de[i] = 0.25 * b * (1.0 + x * a);
    }
    (n, dx, de)
}

fn tshape(r: f64, s: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
    ([1.0 - r - s, r, s], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0])
}

fn interp<const M: usize>(c: &[f64], v: &[V<M>]) -> V<M> {
    let mut r = c3([0.0; 3]);
    for (ci, vi) in c.iter().zip(v) {
        if *ci != 0.0 {
            r = add(&r, &scl(vi, *ci));
        }
    }
    r
}

/// Referensdata per integrationspunkt: lokala riktningar e1, e2 i
/// referensläget uttryckta i tangenterna (e = α G1 + β G2).
#[derive(Debug, Clone, Copy)]
struct IpRef {
    e1: [f64; 3],
    e1c: (f64, f64),
    e2c: (f64, f64),
}

/// Globala X projicerad på planet med normalen n (Z om X ligger inom 0,1° från n).
fn default_dir(n: [f64; 3]) -> [f64; 3] {
    let proj = |a: [f64; 3]| {
        let d = a[0] * n[0] + a[1] * n[1] + a[2] * n[2];
        [a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2]]
    };
    let mut v = proj([1.0, 0.0, 0.0]);
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < (0.1f64).to_radians().sin() {
        v = proj([0.0, 0.0, 1.0]);
    }
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

/// Koefficienterna (α, β) i e = α g1 + β g2 för en vektor e i tangentplanet.
fn coef(e: [f64; 3], g1: [f64; 3], g2: [f64; 3]) -> (f64, f64) {
    let d = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (a11, a12, a22) = (d(g1, g1), d(g1, g2), d(g2, g2));
    let (b1, b2) = (d(e, g1), d(e, g2));
    let det = a11 * a22 - a12 * a12;
    ((a22 * b1 - a12 * b2) / det, (a11 * b2 - a12 * b1) / det)
}

impl Cohesive {
    fn nn(&self) -> usize {
        match self.shape {
            Shape::Line | Shape::LineAxi => 4,
            Shape::Quad => 8,
            Shape::Tri => 6,
        }
    }
    fn nd(&self) -> usize {
        match self.shape {
            Shape::Line | Shape::LineAxi => 2,
            _ => 3,
        }
    }
    fn nip(&self) -> usize {
        match self.shape {
            Shape::Line | Shape::LineAxi => 2,
            Shape::Quad => 4,
            Shape::Tri => 3,
        }
    }
    /// (undersida, översida) som elementnodindex, parvis.
    fn pairs(&self) -> (&'static [usize], &'static [usize]) {
        match self.shape {
            Shape::Line | Shape::LineAxi => (&[0, 1], &[3, 2]),
            Shape::Quad => (&[0, 1, 2, 3], &[4, 5, 6, 7]),
            Shape::Tri => (&[0, 1, 2], &[3, 4, 5]),
        }
    }

    fn law<'a>(&self, ctx: &ElemCtx<'a>) -> Result<&'a TractionSeparation> {
        let m = ctx.material(self.name)?;
        match m.cohesive() {
            Some(l) => Ok(l),
            None => fail(format!(
                "element {} ({}): *COHESIVE SECTION, RESPONSE=TRACTION SEPARATION kräver *ELASTIC, TYPE=TRACTION (materialet är {})",
                ctx.label,
                self.name,
                m.name()
            )),
        }
    }

    /// Referensdata: vikter, lokala riktningar och separationerna i referensläget.
    fn reference(&self, ctx: &ElemCtx) -> Result<(Vec<f64>, Vec<IpRef>, Vec<[f64; 3]>)> {
        let sec = ctx.section(self.name)?;
        let (_, width, _) = section_values(sec);
        let (bot, top) = self.pairs();
        let x = &ctx.x;
        let mid: Vec<[f64; 3]> = bot.iter().zip(top).map(|(&b, &t)| std::array::from_fn(|d| 0.5 * (x[b][d] + x[t][d]))).collect();
        let nrm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let mut w = Vec::new();
        let mut refs = Vec::new();
        match self.shape {
            Shape::Line | Shape::LineAxi => {
                let g = [mid[1][0] - mid[0][0], mid[1][1] - mid[0][1], 0.0];
                let l = nrm(g);
                if !(l > 0.0) {
                    return fail(format!("element {} ({}): mittlinjen har längden noll", ctx.label, self.name));
                }
                for a in 0..2 {
                    if self.shape == Shape::LineAxi {
                        let (ra, rb) = (mid[a][0], mid[1 - a][0]);
                        w.push(2.0 * std::f64::consts::PI * l * (2.0 * ra + rb) / 6.0);
                    } else {
                        w.push(0.5 * l * width);
                    }
                    refs.push(IpRef { e1: [g[0] / l, g[1] / l, 0.0], e1c: (0.0, 0.0), e2c: (0.0, 0.0) });
                }
            }
            Shape::Quad => {
                // vikter ∫Nᵢ dA med 2×2 Gauss (exakt)
                let gp = 1.0 / 3f64.sqrt();
                let mut wi = [0.0; 4];
                for &(a, b) in &QNAT {
                    let (n, dx, de) = qshape(a * gp, b * gp);
                    let g1: [f64; 3] = std::array::from_fn(|d| (0..4).map(|k| dx[k] * mid[k][d]).sum());
                    let g2: [f64; 3] = std::array::from_fn(|d| (0..4).map(|k| de[k] * mid[k][d]).sum());
                    let ja = nrm(crate::elements::common::cross(g1, g2));
                    for k in 0..4 {
                        wi[k] += n[k] * ja;
                    }
                }
                w.extend_from_slice(&wi);
                for &(a, b) in &QNAT {
                    let (_, dx, de) = qshape(a, b);
                    let g1: [f64; 3] = std::array::from_fn(|d| (0..4).map(|k| dx[k] * mid[k][d]).sum());
                    let g2: [f64; 3] = std::array::from_fn(|d| (0..4).map(|k| de[k] * mid[k][d]).sum());
                    refs.push(self.ip_ref(ctx, g1, g2)?);
                }
            }
            Shape::Tri => {
                let g1: [f64; 3] = std::array::from_fn(|d| mid[1][d] - mid[0][d]);
                let g2: [f64; 3] = std::array::from_fn(|d| mid[2][d] - mid[0][d]);
                let a = 0.5 * nrm(crate::elements::common::cross(g1, g2));
                for _ in 0..3 {
                    w.push(a / 3.0);
                    refs.push(self.ip_ref(ctx, g1, g2)?);
                }
            }
        }
        let xr: Vec<V<0>> = x.iter().map(|p| c3(*p)).collect();
        let d0 = self.seps(&xr, &refs).iter().map(val).collect();
        Ok((w, refs, d0))
    }

    fn ip_ref(&self, ctx: &ElemCtx, g1: [f64; 3], g2: [f64; 3]) -> Result<IpRef> {
        let c = crate::elements::common::cross(g1, g2);
        let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        if !(l > 0.0) {
            return fail(format!("element {} ({}): mittytan är urartad", ctx.label, self.name));
        }
        let n = [c[0] / l, c[1] / l, c[2] / l];
        let e1 = default_dir(n);
        let e2 = crate::elements::common::cross(n, e1);
        Ok(IpRef { e1, e1c: coef(e1, g1, g2), e2c: coef(e2, g1, g2) })
    }

    /// Separationerna (n, s, t) i varje integrationspunkt för nodlägena `x`
    /// (utan referensläget dragit ifrån).
    fn seps<const M: usize>(&self, x: &[V<M>], refs: &[IpRef]) -> Vec<V<M>> {
        let (bot, top) = self.pairs();
        let mid: Vec<V<M>> = bot.iter().zip(top).map(|(&b, &t)| scl(&add(&x[b], &x[t]), 0.5)).collect();
        let dd: Vec<V<M>> = bot.iter().zip(top).map(|(&b, &t)| sub(&x[t], &x[b])).collect();
        let zero = Hd::cst(0.0);
        match self.shape {
            Shape::Line | Shape::LineAxi => {
                let t = unit(&sub(&mid[1], &mid[0]));
                let n = [-t[1], t[0], zero];
                let s = dot(&scl(&add(&dd[0], &dd[1]), 0.5), &t);
                (0..2).map(|a| [dot(&dd[a], &n), s, zero]).collect()
            }
            Shape::Quad => {
                let mut out = Vec::with_capacity(4);
                for (i, &(xi, eta)) in QNAT.iter().enumerate() {
                    let (_, dx, de) = qshape(xi, eta);
                    let g1 = interp(&dx, &mid);
                    let g2 = interp(&de, &mid);
                    let n = unit(&cross(&g1, &g2));
                    let dn = dot(&dd[i], &n);
                    // knytpunkter: (0, η) för komponenten längs ξ, (ξ, 0) för η
                    let (na, dxa, _) = qshape(0.0, eta);
                    let ga = interp(&dxa, &mid);
                    let ya = dot(&interp(&na, &dd), &ga);
                    let (nb, _, deb) = qshape(xi, 0.0);
                    let gb = interp(&deb, &mid);
                    let yb = dot(&interp(&nb, &dd), &gb);
                    let (e1, e2) = frame(&refs[i], &g1, &g2, &n);
                    out.push(solve_shear(dn, &ga, &gb, ya, yb, &e1, &e2));
                }
                out
            }
            Shape::Tri => {
                let (_, dr, ds) = tshape(0.0, 0.0);
                let g1 = interp(&dr, &mid);
                let g2 = interp(&ds, &mid);
                let n = unit(&cross(&g1, &g2));
                let at = |r: f64, s: f64| interp(&tshape(r, s).0, &dd);
                let er1 = dot(&at(0.5, 0.0), &g1);
                let es2 = dot(&at(0.0, 0.5), &g2);
                let d3 = at(0.5, 0.5);
                let c = (es2 - er1) - (dot(&d3, &g2) - dot(&d3, &g1));
                let mut out = Vec::with_capacity(3);
                for (i, &(r, s)) in TNAT.iter().enumerate() {
                    let dn = dot(&dd[i], &n);
                    let ya = er1 + c * s;
                    let yb = es2 - c * r;
                    let (e1, e2) = frame(&refs[i], &g1, &g2, &n);
                    out.push(solve_shear(dn, &g1, &g2, ya, yb, &e1, &e2));
                }
                out
            }
        }
    }

    /// Integrationspunkternas separationer δ och derivator: linjärt (B i
    /// referensläget, δ = B u) eller vid NLGEOM i aktuellt läge med Hessian.
    #[allow(clippy::type_complexity)]
    fn kinematics(&self, ctx: &ElemCtx, u: &[f64], refs: &[IpRef], d0: &[[f64; 3]]) -> (Vec<[f64; 3]>, Vec<[Vec<f64>; 3]>, Option<Vec<[Vec<f64>; 3]>>) {
        let nd = self.nd();
        let ndof = self.nn() * nd;
        let nip = self.nip();
        if !ctx.nlgeom {
            // B = ∂δ/∂x i referensläget, en riktning i taget
            let mut b: Vec<[Vec<f64>; 3]> = (0..nip).map(|_| [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]]).collect();
            for j in 0..ndof {
                let x: Vec<V<1>> = ctx
                    .x
                    .iter()
                    .enumerate()
                    .map(|(a, p)| {
                        let mut v = c3::<1>(*p);
                        if a == j / nd {
                            v[j % nd].g[0] = 1.0;
                        }
                        v
                    })
                    .collect();
                for (q, s) in self.seps(&x, refs).iter().enumerate() {
                    for k in 0..3 {
                        b[q][k][j] = s[k].g[0];
                    }
                }
            }
            let d = b.iter().map(|bq| std::array::from_fn(|k| bq[k].iter().zip(u).map(|(p, q)| p * q).sum())).collect();
            return (d, b, None);
        }
        let (d, b, h) = match self.shape {
            Shape::Line | Shape::LineAxi => self.full::<8>(ctx, u, refs),
            Shape::Tri => self.full::<18>(ctx, u, refs),
            Shape::Quad => self.full::<24>(ctx, u, refs),
        };
        let d = d.iter().zip(d0).map(|(a, r)| std::array::from_fn(|k| a[k] - r[k])).collect();
        (d, b, Some(h))
    }

    #[allow(clippy::type_complexity)]
    fn full<const N: usize>(&self, ctx: &ElemCtx, u: &[f64], refs: &[IpRef]) -> (Vec<[f64; 3]>, Vec<[Vec<f64>; 3]>, Vec<[Vec<f64>; 3]>) {
        let nd = self.nd();
        let x: Vec<V<N>> = ctx
            .x
            .iter()
            .enumerate()
            .map(|(a, p)| {
                std::array::from_fn(|c| if c < nd { Hd::var(p[c] + u[a * nd + c], a * nd + c) } else { Hd::cst(p[c]) })
            })
            .collect();
        let s = self.seps(&x, refs);
        let d = s.iter().map(val).collect();
        let b = s.iter().map(|v| std::array::from_fn(|k| v[k].g.to_vec())).collect();
        let h = s.iter().map(|v| std::array::from_fn(|k| v[k].h.iter().flat_map(|r| r.iter().copied()).collect())).collect();
        (d, b, h)
    }

    fn t0s(&self, ctx: &ElemCtx, d0: &[[f64; 3]]) -> Result<Vec<f64>> {
        let (t0, _, _) = section_values(ctx.section(self.name)?);
        match t0 {
            Some(t) => Ok(vec![t; self.nip()]),
            None => {
                let v: Vec<f64> = d0.iter().map(|d| d[0]).collect();
                if v.iter().any(|&h| !(h > 0.0)) {
                    return fail(format!("element {} ({}): THICKNESS=GEOMETRY men elementets tjocklek är noll eller negativ", ctx.label, self.name));
                }
                Ok(v)
            }
        }
    }
}

/// Lokala riktningar i aktuellt läge: referensriktningarna roterade med
/// rotationen ur mittytans deformationsgradient (polärdekomposition i planet).
fn frame<const M: usize>(r: &IpRef, g1: &V<M>, g2: &V<M>, n: &V<M>) -> (V<M>, V<M>) {
    let fe1 = add(&scl(g1, r.e1c.0), &scl(g2, r.e1c.1));
    let fe2 = add(&scl(g1, r.e2c.0), &scl(g2, r.e2c.1));
    let e10: V<M> = c3(r.e1);
    let a1 = unit(&sub(&e10, &mulv(n, dot(&e10, n))));
    let a2 = cross(n, &a1);
    let cc = dot(&a1, &fe1) + dot(&a2, &fe2);
    let ss = dot(&a2, &fe1) - dot(&a1, &fe2);
    let rr = (cc * cc + ss * ss).sqrt().recip();
    let e1 = add(&mulv(&a1, cc * rr), &mulv(&a2, ss * rr));
    let e2 = cross(n, &e1);
    (e1, e2)
}

/// [δn, s1, s2] ur de kovarianta skjuvkomponenterna ya = s·ga, yb = s·gb.
fn solve_shear<const M: usize>(dn: Hd<M>, ga: &V<M>, gb: &V<M>, ya: Hd<M>, yb: Hd<M>, e1: &V<M>, e2: &V<M>) -> V<M> {
    let (a00, a01, a10, a11) = (dot(ga, e1), dot(ga, e2), dot(gb, e1), dot(gb, e2));
    let det = (a00 * a11 - a01 * a10).recip();
    [dn, (a11 * ya - a01 * yb) * det, (a00 * yb - a10 * ya) * det]
}

/// Per integrationspunkt först dragspänningen (3) och töjningen (3), sedan lagens tillstånd.
const NTR: usize = 6;

impl ElementType for Cohesive {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        self.nd()
    }
    fn n_nodes(&self) -> usize {
        self.nn()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.nd() == 2 {
            &[U1, U2]
        } else {
            &[U1, U2, U3]
        }
    }
    fn n_ips(&self) -> usize {
        self.nip()
    }
    fn vtk_type(&self) -> u8 {
        match self.shape {
            Shape::Line | Shape::LineAxi => 9,
            Shape::Quad => 12,
            Shape::Tri => 13,
        }
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let law = self.law(ctx)?;
        let stride = NTR + law.n_state();
        let mut s = vec![0.0; self.nip() * stride];
        for q in 0..self.nip() {
            law.init_state(&mut s[q * stride + NTR..(q + 1) * stride]);
        }
        Ok(s)
    }

    fn mass(&self, ctx: &ElemCtx, _kind: super::api::MassKind) -> Result<Vec<f64>> {
        // masslösa skikt går bra i dynamiska steg; densitet stöds inte ännu
        if ctx.density != 0.0 {
            return fail(format!("element {} ({}): *DENSITY i kohesiva element stöds inte ännu", ctx.label, self.name));
        }
        let n = self.nn() * self.nd();
        Ok(vec![0.0; n * n])
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let law = self.law(ctx)?;
        let (w, refs, d0) = self.reference(ctx)?;
        let t0 = self.t0s(ctx, &d0)?;
        let ndof = self.nn() * self.nd();
        let lin = ElemCtx { nlgeom: false, ..ctx.clone() };
        let (_, b, _) = self.kinematics(&lin, &vec![0.0; ndof], &refs, &d0);
        let dmat = law.elastic();
        let mut k = vec![0.0; ndof * ndof];
        for q in 0..self.nip() {
            add_btdb(&mut k, &b[q], &dmat, w[q] / t0[q], ndof);
        }
        Ok(k)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let law = self.law(ctx)?;
        let (_, _, visc) = section_values(ctx.section(self.name)?);
        let (w, refs, d0) = self.reference(ctx)?;
        let t0 = self.t0s(ctx, &d0)?;
        let nd = self.nd();
        let ndof = self.nn() * nd;
        let nip = self.nip();
        let stride = NTR + law.n_state();
        let init;
        let so = if so.len() == nip * stride {
            so
        } else {
            init = self.init_state(ctx)?;
            &init[..]
        };
        sn.clear();
        sn.extend_from_slice(so);
        let (d, b, h) = self.kinematics(ctx, u, &refs, &d0);
        // borttaget element: töjningen redovisas som vid borttagningen
        let was_deleted = law.has_damage() && (0..nip).all(|q| law.is_deleted(&so[q * stride + NTR..(q + 1) * stride]));
        let mut outs = Vec::with_capacity(nip);
        for q in 0..nip {
            let e = [d[q][0] / t0[q], d[q][1] / t0[q], d[q][2] / t0[q]];
            let base = q * stride;
            if !was_deleted {
                sn[base + 3..base + 6].copy_from_slice(&e);
            }
            let mut st = so[base + NTR..base + stride].to_vec();
            let o = law.update(&e, nd, t0[q], ctx.dt, visc, &so[base + NTR..base + stride], &mut st);
            sn[base + NTR..base + stride].copy_from_slice(&st);
            outs.push(o);
        }
        // elementet tas bort när alla punkter gått sönder och ingen är i tryck
        let deleted = was_deleted
            || law.has_damage() && (0..nip).all(|q| law.outputs(&sn[q * stride + NTR..(q + 1) * stride]).2 && d[q][0] >= 0.0);
        if deleted {
            for q in 0..nip {
                law.mark_deleted(&mut sn[q * stride + NTR..(q + 1) * stride]);
            }
        }
        let mut fint = vec![0.0; ndof];
        let mut k = if tangent { vec![0.0; ndof * ndof] } else { Vec::new() };
        for (q, o) in outs.iter().enumerate() {
            let t = if deleted { [0.0; 3] } else { o.t };
            sn[q * stride..q * stride + 3].copy_from_slice(&t);
            if deleted {
                continue;
            }
            for kk in 0..3 {
                if t[kk] != 0.0 {
                    for j in 0..ndof {
                        fint[j] += w[q] * t[kk] * b[q][kk][j];
                    }
                }
            }
            if tangent {
                add_btdb(&mut k, &b[q], &o.dtde, w[q] / t0[q], ndof);
                if let Some(h) = &h {
                    for kk in 0..3 {
                        if t[kk] != 0.0 {
                            for (kv, hv) in k.iter_mut().zip(&h[q][kk]) {
                                *kv += w[q] * t[kk] * hv;
                            }
                        }
                    }
                }
            }
        }
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let law = self.law(ctx)?;
        let nip = self.nip();
        let stride = NTR + law.n_state();
        // töjningen ur tillståndet; utan tillstånd ur förskjutningen
        let eps: Vec<[f64; 3]> = if state.len() == nip * stride {
            (0..nip).map(|q| [state[q * stride + 3], state[q * stride + 4], state[q * stride + 5]]).collect()
        } else {
            let (_, refs, d0) = self.reference(ctx)?;
            let t0 = self.t0s(ctx, &d0)?;
            let (d, _, _) = self.kinematics(ctx, u, &refs, &d0);
            d.iter().zip(&t0).map(|(dq, t)| [dq[0] / t, dq[1] / t, dq[2] / t]).collect()
        };
        let three = self.nd() == 3;
        let (sl, el): (&[&str], &[&str]) = if three { (&["S33", "S13", "S23"], &["E33", "E13", "E23"]) } else { (&["S22", "S12"], &["E22", "E12"]) };
        let lel: Vec<String> = el.iter().map(|l| format!("L{l}")).collect();
        let mut s = IpField::new("S", sl);
        let mut e = if ctx.nlgeom {
            IpField { name: "LE".into(), labels: lel, points: Vec::new() }
        } else {
            IpField::new("E", el)
        };
        let mut sdeg = IpField::new("SDEG", &["SDEG"]);
        let cname = law.criterion_name();
        let mut crit = cname.map(|c| IpField::new(c, &[c]));
        let mut deleted = true;
        let nc = if three { 3 } else { 2 };
        for q in 0..nip {
            let st = state.get(q * stride..(q + 1) * stride).unwrap_or(&[]);
            let t: Vec<f64> = (0..nc).map(|k| st.get(k).copied().unwrap_or(0.0)).collect();
            s.points.push((q + 1, 0, t));
            e.points.push((q + 1, 0, eps[q][..nc].to_vec()));
            let (dg, cr, _) = law.outputs(st.get(NTR..).unwrap_or(&[]));
            deleted &= law.is_deleted(st.get(NTR..).unwrap_or(&[]));
            sdeg.points.push((q + 1, 0, vec![dg]));
            if let Some(c) = &mut crit {
                c.points.push((q + 1, 0, vec![cr]));
            }
        }
        let mut out = vec![s, e];
        if law.has_damage() {
            out.push(sdeg);
            let mut status = IpField::new("STATUS", &["STATUS"]);
            status.points.push((0, 0, vec![if deleted { 0.0 } else { 1.0 }]));
            out.push(status);
        }
        out.extend(crit);
        Ok(out)
    }
}

/// K += c · Bᵀ D B med B som tre rader (n, s, t).
fn add_btdb(k: &mut [f64], b: &[Vec<f64>; 3], d: &[[f64; 3]; 3], c: f64, ndof: usize) {
    let mut db = [vec![0.0; ndof], vec![0.0; ndof], vec![0.0; ndof]];
    for r in 0..3 {
        for m in 0..3 {
            if d[r][m] != 0.0 {
                for j in 0..ndof {
                    db[r][j] += d[r][m] * b[m][j];
                }
            }
        }
    }
    for r in 0..3 {
        for i in 0..ndof {
            let bi = b[r][i] * c;
            if bi != 0.0 {
                for j in 0..ndof {
                    k[i * ndof + j] += bi * db[r][j];
                }
            }
        }
    }
}
