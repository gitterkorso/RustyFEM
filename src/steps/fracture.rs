//! Fracture mechanics: *CONTOUR INTEGRAL in *STATIC (domain integral method).
//!
//! The step type takes *STATIC steps (without NLGEOM) that have *CONTOUR INTEGRAL, runs
//! them with the regular static step type and then computes the contour integrals
//! from the final state. Output is written as history with the standard names in the region
//! "ElementSet  ALL ELEMENTS":
//!
//! | TYPE= | Output |
//! |---|---|
//! | J (default) | `J at <CRACK>_<NODESET>_Contour_<k>` |
//! | K FACTORS | `K1`, `K2` (`K3` in 3D), `JKs` (J from K), `Cpd` (crack propagation direction, MTS, degrees) |
//! | T-STRESS | `T-stress` |
//!
//! Method: contour 1 is the elements around the crack-tip nodes, contour
//! k + 1 the elements adjacent to contour k. The virtual crack extension
//! q = 1 at all nodes inside the contour and 0 outside, interpolated with
//! the elements' shape functions, and
//!
//! J = ∫ (σᵢⱼ ∂uⱼ/∂x₁ − W δ₁ᵢ) ∂q/∂xᵢ dA
//!
//! in the crack tip's local system (x₁ = the q vector) with stress and strain at
//! the elements' integration points and W = ½ σ:ε. Axisymmetry: the integrand
//! is weighted by r, the term (σ_θθ u_r/r − W) q_r/r is added and J is divided by
//! the tip radius. 3D: one virtual extension per front node (q distributed along
//! the front with the element edges' shape functions, normalized with ∫ q ds).
//! The K factors are obtained from the interaction integral with Williams fields (K = 1),
//! the T-stress with the point-force field (`fracture_aux.rs`). SYMM doubles J,
//! K₁ and T (half model, the crack plane is a symmetry plane). In 3D and axisymmetry
//! ν E* ε₃₃ (the strain along the front at the front node) is added to T.
//!
//! Limitations: only linear isotropic elastic materials without thermal
//! strain or pressure on the crack faces; K and T at 3D front nodes on a free surface
//! are less accurate than J; TYPE=C(t), CRACK TIP NODES and
//! DIRECTION=MERR/KII0 are not supported.

#[path = "fracture_aux.rs"]
pub mod aux;

use super::api::{StepCtx, StepType};
use crate::elements::solid_higher::shapes::{self, Shape};
use crate::elements::solid_higher::RuleKind;
use crate::elements::{Family, IpField};
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::{FeModel, RawKeyword, StepDef};
use crate::model::{fail, FemError, Result};
use aux::{AuxMode, Elastic};
use std::collections::BTreeSet;

pub struct ContourStatic;

pub static STEP_TYPES: &[&dyn StepType] = &[&ContourStatic];

/// History region for contour integrals in the results file.
pub const REGION: &str = "ElementSet  ALL ELEMENTS";

fn has_contour(step: &StepDef) -> bool {
    step.extra.iter().any(|k| k.name == "CONTOUR INTEGRAL")
}

impl StepType for ContourStatic {
    fn name(&self) -> &'static str {
        "STATIC with *CONTOUR INTEGRAL"
    }

    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32> {
        let riks = step.procedure.as_ref().is_some_and(|p| p.has("RIKS"));
        (step.procedure_name() == "STATIC" && !nlgeom && !riks && !step.perturbation && has_contour(step)).then_some(5)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let t0 = ctx.state.time;
        super::static_linear::Static.run(ctx)?;
        let period = ctx.state.time - t0;
        let prep = ctx.prep;
        let reqs: Vec<Request> = ctx.step.extra.iter().filter(|k| k.name == "CONTOUR INTEGRAL").map(|k| Request::parse(prep.model, k)).collect::<Result<_>>()?;
        let out = evaluate(prep, &ctx.state.u, &ctx.state.elem_state, ctx.state.time, &reqs)?;
        for (name, v) in out {
            ctx.out.history.push((REGION.to_string(), name, vec![(period, v)]));
        }
        Ok(())
    }
}

/// Type of contour integral (TYPE=).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegralType {
    J,
    KFactors,
    TStress,
}

/// A node set along the crack front (one data line).
#[derive(Debug, Clone)]
pub struct Front {
    pub name: String,
    pub nodes: Vec<usize>,
    /// q vector (or the crack plane's normal with NORMAL).
    pub vec: [f64; 3],
}

/// A *CONTOUR INTEGRAL request.
#[derive(Debug, Clone)]
pub struct Request {
    pub crack: String,
    pub contours: usize,
    pub kind: IntegralType,
    pub symm: bool,
    pub normal: bool,
    pub fronts: Vec<Front>,
}

fn node_set(model: &FeModel, kw: &RawKeyword, tok: &str) -> Result<Vec<usize>> {
    let up = tok.trim().to_uppercase();
    if let Some(s) = model.nsets.get(&up) {
        return Ok(s.clone());
    }
    let suffix = format!(".{up}");
    if let Some((_, s)) = model.nsets.iter().find(|(k, _)| k.ends_with(&suffix)) {
        return Ok(s.clone());
    }
    if let Ok(label) = up.parse::<i64>() {
        if let Some(n) = (0..model.nodes.len()).find(|&n| model.node_label[n] == label) {
            return Ok(vec![n]);
        }
    }
    kw.err(format!("the node set or node '{tok}' does not exist"))
}

impl Request {
    pub fn parse(model: &FeModel, kw: &RawKeyword) -> Result<Request> {
        let kind = match kw.param_upper("TYPE").as_deref().map(|s| s.replace(' ', "")) {
            None => IntegralType::J,
            Some(t) if t == "J" => IntegralType::J,
            Some(t) if t == "KFACTORS" => IntegralType::KFactors,
            Some(t) if t == "T-STRESS" => IntegralType::TStress,
            Some(t) => return kw.err(format!("TYPE={t} is not yet supported (J, K FACTORS, T-STRESS)")),
        };
        if kw.has("CRACK TIP NODES") {
            return kw.err("CRACK TIP NODES is not yet supported – specify one node set per front position and the q vector");
        }
        if let Some(d) = kw.param_upper("DIRECTION") {
            if d != "MTS" {
                return kw.err(format!("DIRECTION={d} is not yet supported (MTS)"));
            }
        }
        let contours = match kw.param("CONTOURS") {
            Some(c) => c.trim().parse::<usize>().map_err(|_| FemError(format!("{}: CONTOURS={c} is not an integer", kw.location())))?,
            None => 1,
        };
        let mut fronts = Vec::new();
        let normal = kw.has("NORMAL");
        let records = kw.records();
        // NORMAL: the first data line is the crack plane's normal, then the node sets
        let mut nvec: Option<[f64; 3]> = None;
        for (ri, d) in records.iter().enumerate() {
            if d.fields.iter().all(|f| f.trim().is_empty()) {
                continue;
            }
            if normal && ri == 0 {
                let v: Vec<f64> = d.fields.iter().map(|f| crate::fem::model::parse_f64(f).unwrap_or(0.0)).collect();
                let n = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
                if n.iter().all(|x| *x == 0.0) {
                    return kw.err_at(d.line, "NORMAL: the normal is missing or zero");
                }
                nvec = Some(n);
                continue;
            }
            if normal {
                for name in d.fields.iter().filter(|f| !f.trim().is_empty()) {
                    let nodes = node_set(model, kw, name)?;
                    fronts.push(Front { name: name.trim().to_uppercase(), nodes, vec: nvec.unwrap() });
                }
                continue;
            }
            let name = &d.fields[0];
            let nodes = node_set(model, kw, name)?;
            let v: Vec<f64> = d.fields[1..].iter().map(|f| crate::fem::model::parse_f64(f).unwrap_or(0.0)).collect();
            let vec = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
            if vec.iter().all(|x| *x == 0.0) {
                return kw.err_at(d.line, "the q vector is missing or zero");
            }
            fronts.push(Front { name: name.trim().to_uppercase(), nodes, vec });
        }
        if fronts.is_empty() {
            return kw.err("data line with node set and q vector missing");
        }
        Ok(Request {
            crack: kw.param("CRACK NAME").map(|s| s.trim().to_uppercase()).unwrap_or_else(|| "H-OUTPUT-1".into()),
            contours: contours.max(1),
            kind,
            symm: kw.has("SYMM"),
            normal,
            fronts,
        })
    }
}

/// The element's geometry for the domain integral.
#[derive(Debug, Clone, Copy)]
struct Geo {
    shape: Shape,
    rule: RuleKind,
    family: Family,
    plane_stress: bool,
}

fn geometry(name: &str) -> Option<Geo> {
    use crate::materials::StressMode;
    let g = |shape, rule, family, plane_stress| Some(Geo { shape, rule, family, plane_stress });
    match name {
        "CPS4" => return g(Shape::Quad4, RuleKind::Gauss(2), Family::Continuum2D, true),
        "CPE4" => return g(Shape::Quad4, RuleKind::Gauss(2), Family::Continuum2D, false),
        "CPS3" => return g(Shape::Tri3, RuleKind::Tri(1), Family::Continuum2D, true),
        "CPE3" => return g(Shape::Tri3, RuleKind::Tri(1), Family::Continuum2D, false),
        "C3D8" => return g(Shape::Hex8, RuleKind::Gauss(2), Family::Continuum3D, false),
        _ => {}
    }
    let s = crate::elements::solid_higher::solids().into_iter().find(|s| s.name == name)?;
    if s.tech == crate::elements::solid_higher::Tech::Incompatible {
        return None;
    }
    let family = match s.mode {
        StressMode::Axisymmetric => Family::Axisymmetric,
        StressMode::ThreeD => Family::Continuum3D,
        _ => Family::Continuum2D,
    };
    g(s.shape, s.rule, family, s.mode == StressMode::PlaneStress)
}

/// Isotropic elastic constants (E, ν) for the element's material.
fn elastic_of(prep: &Prepared, e: usize) -> Result<(f64, f64)> {
    let m = prep.model;
    let sec = m.elements[e].section.map(|s| &m.sections[s]);
    let Some(mi) = sec.and_then(|s| s.material_index) else {
        return fail(format!("{}: material missing for the contour integral", m.elem_name(e)));
    };
    let mat = &m.materials[mi];
    let Some(el) = mat.behavior("ELASTIC") else {
        return fail(format!("material {}: contour integrals require *ELASTIC (isotropic)", mat.name));
    };
    if el.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") {
        return fail(format!("material {}: contour integrals are only supported for isotropic elasticity", mat.name));
    }
    let v = el.first_floats()?;
    if v.len() < 2 {
        return fail(format!("material {}: *ELASTIC has no E and ν", mat.name));
    }
    Ok((v[0], v[1]))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Local system at a front point: rows e₁ (q), e₂, e₃ (front tangent).
fn frame_of(q: [f64; 3], tangent: [f64; 3]) -> [[f64; 3]; 3] {
    let e3 = unit(tangent);
    let q = sub(q, scale(e3, dot(q, e3)));
    let e1 = unit(q);
    let e2 = cross(e3, e1);
    [e1, e2, e3]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Values at an integration point.
struct IpData {
    x: [f64; 3],
    w: f64,
    /// q and ∇q (global)
    q: f64,
    dq: [f64; 3],
    /// ∇u global: g[j][i] = ∂u_j/∂x_i
    g: [[f64; 3]; 3],
    /// u (for the axisymmetric u_r/r)
    u: [f64; 3],
    s: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
}

fn tensor(labels: &[String], v: &[f64], shear_half: bool) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    let f = if shear_half { 0.5 } else { 1.0 };
    for (l, &x) in labels.iter().zip(v) {
        let b = l.as_bytes();
        if b.len() < 3 {
            continue;
        }
        let i = (b[b.len() - 2] - b'1') as usize;
        let j = (b[b.len() - 1] - b'1') as usize;
        if i > 2 || j > 2 {
            continue;
        }
        if i == j {
            t[i][i] = x;
        } else {
            t[i][j] = x * f;
            t[j][i] = x * f;
        }
    }
    t
}

fn rot(r: &[[f64; 3]; 3], t: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // R t Rᵀ
    let mut a = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    s += r[i][k] * t[k][l] * r[j][l];
                }
            }
            a[i][j] = s;
        }
    }
    a
}
fn rotv(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
}

/// The integration points in element `e` with the q values `qn` at the element's nodes.
fn ip_data(prep: &Prepared, e: usize, geo: &Geo, u: &[NodeVec], out: &[IpField], qn: &[f64]) -> Result<Vec<IpData>> {
    let m = prep.model;
    let el = &m.elements[e];
    let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
    let pdim = geo.shape.pdim();
    let rule = geo.rule.points(pdim);
    let sfield = out.iter().find(|f| f.name == "S").ok_or_else(|| FemError(format!("{}: stress missing", m.elem_name(e))))?;
    let efield = out.iter().find(|f| f.name == "E").ok_or_else(|| FemError(format!("{}: strain missing", m.elem_name(e))))?;
    let mut res = Vec::with_capacity(rule.len());
    for (k, (p, w)) in rule.iter().enumerate() {
        let (n, dn) = shapes::shape(geo.shape, *p);
        // Jacobian (pdim × pdim)
        let mut jac = [[0.0; 3]; 3];
        for (a, xa) in x.iter().enumerate() {
            for r in 0..pdim {
                for c in 0..pdim {
                    jac[r][c] += dn[a][r] * xa[c];
                }
            }
        }
        if pdim == 2 {
            jac[2][2] = 1.0;
        }
        let (inv, det) = crate::elements::common::inv3(jac);
        if det <= 0.0 {
            return fail(format!("{}: {}", m.elem_name(e), crate::elements::common::DISTORTED));
        }
        let dndx: Vec<[f64; 3]> = dn
            .iter()
            .map(|d| {
                let mut g = [0.0; 3];
                for c in 0..pdim {
                    for r in 0..pdim {
                        g[c] += inv[c][r] * d[r];
                    }
                }
                g
            })
            .collect();
        let mut xp = [0.0; 3];
        let mut up = [0.0; 3];
        let mut g = [[0.0; 3]; 3];
        let mut q = 0.0;
        let mut dq = [0.0; 3];
        for (a, &na) in el.nodes.iter().enumerate() {
            for i in 0..3 {
                xp[i] += n[a] * x[a][i];
                up[i] += n[a] * u[na][i];
                dq[i] += dndx[a][i] * qn[a];
                for j in 0..3 {
                    g[j][i] += u[na][j] * dndx[a][i];
                }
            }
            q += n[a] * qn[a];
        }
        let ipn = k + 1;
        let sv = sfield.points.iter().find(|pt| pt.0 == ipn).map(|pt| pt.2.clone());
        let ev = efield.points.iter().find(|pt| pt.0 == ipn).map(|pt| pt.2.clone());
        let (Some(sv), Some(ev)) = (sv, ev) else {
            return fail(format!("{}: stress at integration point {ipn} missing", m.elem_name(e)));
        };
        res.push(IpData { x: xp, w: w * det, q, dq, g, u: up, s: tensor(&sfield.labels, &sv, false), eps: tensor(&efield.labels, &ev, true) });
    }
    Ok(res)
}

/// Integrals for one virtual extension: J and the interaction integrals.
#[derive(Debug, Clone, Copy, Default)]
struct Sums {
    j: f64,
    i: [f64; 4],
}

/// Contribution from one integration point. `r` = local system, `tip` = the tip position.
fn contribution(d: &IpData, r: &[[f64; 3]; 3], tip: [f64; 3], qhat: [f64; 3], el: &Elastic, axi: bool, need_aux: bool) -> Sums {
    let s = rot(r, &d.s);
    let eps = rot(r, &d.eps);
    let g = rot(r, &d.g);
    let dq = rotv(r, d.dq);
    let mut w = 0.0;
    for k in 0..3 {
        for l in 0..3 {
            w += 0.5 * s[k][l] * eps[k][l];
        }
    }
    let mut out = Sums::default();
    let mut jv = 0.0;
    for i in 0..3 {
        let mut f = -w * if i == 0 { 1.0 } else { 0.0 };
        for j in 0..3 {
            f += s[i][j] * g[j][0];
        }
        jv += f * dq[i];
    }
    // the radius and q_r (global r component of Q = q·q̂)
    let (rad, qr) = (d.x[0], d.q * qhat[0]);
    let ur_r = if axi { d.u[0] / rad } else { 0.0 };
    // σ_θθ in the local system: e₃ = θ (the 2D plane is r–z)
    if axi {
        jv += (d.s[2][2] * ur_r - w) * qr / rad;
    }
    let wt = d.w * if axi { rad } else { 1.0 };
    out.j = jv * wt;
    if need_aux {
        let xl = rotv(r, sub(d.x, tip));
        for (m, mode) in [AuxMode::I, AuxMode::II, AuxMode::III, AuxMode::T].into_iter().enumerate() {
            let a = aux::aux(mode, el, xl[0], xl[1]);
            let mut sae = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    sae += a.s[k][l] * eps[k][l];
                }
            }
            let mut iv = 0.0;
            for i in 0..3 {
                let mut f = -sae * if i == 0 { 1.0 } else { 0.0 };
                for j in 0..3 {
                    f += s[i][j] * a.g[j][0] + a.s[i][j] * g[j][0];
                }
                iv += f * dq[i];
            }
            if axi {
                let sa33 = a.s[2][2];
                // the auxiliary field's ε_θθ = 0 (plane strain), the term σ_θθ u^aux_r/r is not included
                iv += (sa33 * ur_r - sae) * qr / rad;
            }
            out.i[m] = iv * wt;
        }
    }
    out
}

/// The contour integrals for all requests: (history name, value).
pub fn evaluate(prep: &Prepared, u: &[NodeVec], states: &[Vec<f64>], time: f64, reqs: &[Request]) -> Result<Vec<(String, f64)>> {
    let m = prep.model;
    let ne = m.elements.len();
    let geos: Vec<Option<Geo>> = m.elements.iter().map(|e| geometry(e.etype.name())).collect();
    let mut node_elems: Vec<Vec<usize>> = vec![Vec::new(); m.nodes.len()];
    for e in 0..ne {
        if geos[e].is_some() {
            for &n in &m.elements[e].nodes {
                node_elems[n].push(e);
            }
        }
    }
    let mut out_cache: Vec<Option<Vec<IpField>>> = vec![None; ne];
    let mut res = Vec::new();
    for req in reqs {
        let three_d = req.fronts.iter().any(|f| f.nodes.iter().any(|&n| node_elems[n].iter().any(|&e| geos[e].unwrap().family == Family::Continuum3D)));
        // 2D: each data line is a separate crack tip; 3D: the lines are the front in order
        let groups: Vec<Vec<usize>> = if three_d { vec![(0..req.fronts.len()).collect()] } else { (0..req.fronts.len()).map(|i| vec![i]).collect() };
        for grp in groups {
            let fronts: Vec<&Front> = grp.iter().map(|&i| &req.fronts[i]).collect();
            let vals = evaluate_front(prep, u, states, time, req, &fronts, &geos, &node_elems, &mut out_cache, three_d)?;
            for (fi, f) in fronts.iter().enumerate() {
                for (k, v) in vals[fi].iter().enumerate() {
                    let tag = format!("{}_{}_Contour_{}", req.crack, f.name, k + 1);
                    match req.kind {
                        IntegralType::J => res.push((format!("J at {tag}"), v.j)),
                        IntegralType::KFactors => {
                            res.push((format!("K1 at {tag}"), v.k[0]));
                            res.push((format!("K2 at {tag}"), v.k[1]));
                            if three_d {
                                res.push((format!("K3 at {tag}"), v.k[2]));
                            }
                            res.push((format!("JKs at {tag}"), v.jks));
                            res.push((format!("Cpd at {tag}"), v.cpd));
                        }
                        IntegralType::TStress => res.push((format!("T-stress at {tag}"), v.t)),
                    }
                }
            }
        }
    }
    Ok(res)
}

/// Result per front node and contour.
#[derive(Debug, Clone, Copy, Default)]
pub struct Values {
    pub j: f64,
    pub k: [f64; 3],
    pub jks: f64,
    pub cpd: f64,
    pub t: f64,
}

/// The MTS criterion's crack propagation angle in degrees.
pub fn mts_angle(k1: f64, k2: f64) -> f64 {
    if k2 == 0.0 {
        return 0.0;
    }
    let r = k1 / k2;
    2.0 * (0.25 * (r - k2.signum() * (r * r + 8.0).sqrt())).atan() * 180.0 / std::f64::consts::PI
}

#[allow(clippy::too_many_arguments)]
fn evaluate_front(
    prep: &Prepared,
    u: &[NodeVec],
    states: &[Vec<f64>],
    time: f64,
    req: &Request,
    fronts: &[&Front],
    geos: &[Option<Geo>],
    node_elems: &[Vec<usize>],
    cache: &mut [Option<Vec<IpField>>],
    three_d: bool,
) -> Result<Vec<Vec<Values>>> {
    let m = prep.model;
    let nf = fronts.len();
    let pos: Vec<[f64; 3]> = fronts.iter().map(|f| m.nodes[f.nodes[0]]).collect();
    // front tangent at each front node
    let tangent: Vec<[f64; 3]> = (0..nf)
        .map(|k| if !three_d || nf < 2 { [0.0, 0.0, 1.0] } else { unit(sub(pos[(k + 1).min(nf - 1)], pos[k.saturating_sub(1)])) })
        .collect();
    let qvec: Vec<[f64; 3]> = (0..nf)
        .map(|k| {
            let v = fronts[k].vec;
            if req.normal {
                // q = t × n (t = the front tangent from the first to the last node set)
                cross(tangent[k], v)
            } else {
                v
            }
        })
        .collect();
    let frames: Vec<[[f64; 3]; 3]> = (0..nf).map(|k| frame_of(qvec[k], tangent[k])).collect();
    // second-order front: the front nodes alternate corner/midside
    let quadratic = three_d
        && fronts.iter().any(|f| f.nodes.iter().any(|&n| node_elems[n].iter().any(|&e| m.elements[e].etype.n_nodes() >= 20)));
    // weights along the front: ∫ q_k ds and q_k at a point
    let segs: Vec<(usize, usize, usize)> = if !three_d {
        Vec::new()
    } else if quadratic {
        (0..(nf - 1) / 2).map(|s| (2 * s, 2 * s + 1, 2 * s + 2)).collect()
    } else {
        (0..nf - 1).map(|s| (s, usize::MAX, s + 1)).collect()
    };
    let mut length = vec![0.0; nf];
    for &(a, mid, b) in &segs {
        let l = dot(sub(pos[b], pos[a]), sub(pos[b], pos[a])).sqrt();
        if mid == usize::MAX {
            length[a] += 0.5 * l;
            length[b] += 0.5 * l;
        } else {
            length[a] += l / 6.0;
            length[b] += l / 6.0;
            length[mid] += 2.0 * l / 3.0;
        }
    }
    // q value for front node k at position x (3D)
    let weight = |k: usize, x: [f64; 3]| -> f64 {
        if !three_d {
            return 1.0;
        }
        let mut best: Option<(f64, usize, f64)> = None;
        for (si, &(a, _, b)) in segs.iter().enumerate() {
            let ab = sub(pos[b], pos[a]);
            let lam = dot(sub(x, pos[a]), ab) / dot(ab, ab);
            let lc = lam.clamp(0.0, 1.0);
            let p = [pos[a][0] + lc * ab[0], pos[a][1] + lc * ab[1], pos[a][2] + lc * ab[2]];
            let dist = dot(sub(x, p), sub(x, p)).sqrt() + (lam - lc).abs() * 1e3 * dot(ab, ab).sqrt();
            if best.is_none_or(|b| dist < b.0 - 1e-12 * dist.abs()) {
                best = Some((dist, si, lc));
            }
        }
        let Some((_, si, lam)) = best else { return 0.0 };
        let (a, mid, b) = segs[si];
        if mid == usize::MAX {
            if k == a {
                1.0 - lam
            } else if k == b {
                lam
            } else {
                0.0
            }
        } else {
            let t = 2.0 * lam - 1.0;
            if k == a {
                0.5 * t * (t - 1.0)
            } else if k == b {
                0.5 * t * (t + 1.0)
            } else if k == mid {
                1.0 - t * t
            } else {
                0.0
            }
        }
    };
    let need_aux = req.kind != IntegralType::J;
    // the strain along the front (∂(u·t)/∂s) at each front node: derivative of
    // the quadratic interpolation through the node and its neighbors (one-sided at the ends)
    let eps_front: Vec<f64> = (0..nf)
        .map(|k| {
            if !three_d || nf < 3 {
                return 0.0;
            }
            let a = if k == 0 { 0 } else if k == nf - 1 { nf - 3 } else { k - 1 };
            let idx = [a, a + 1, a + 2];
            let mut sv = [0.0; 3];
            for i in 1..3 {
                sv[i] = sv[i - 1] + dot(sub(pos[idx[i]], pos[idx[i - 1]]), sub(pos[idx[i]], pos[idx[i - 1]])).sqrt();
            }
            let t = unit(sub(pos[idx[2]], pos[idx[0]]));
            let uv: Vec<f64> = idx.iter().map(|&i| {
                let n = fronts[i].nodes[0];
                dot([u[n][0], u[n][1], u[n][2]], t)
            }).collect();
            let x = sv[k - a];
            // derivative of the Lagrange polynomial through (sv, uv) at x
            let mut d = 0.0;
            for i in 0..3 {
                let mut term = 0.0;
                for j in 0..3 {
                    if j == i {
                        continue;
                    }
                    let mut p = 1.0 / (sv[i] - sv[j]);
                    for l in 0..3 {
                        if l != i && l != j {
                            p *= (x - sv[l]) / (sv[i] - sv[l]);
                        }
                    }
                    term += p;
                }
                d += uv[i] * term;
            }
            d
        })
        .collect();
    let mut inside: BTreeSet<usize> = fronts.iter().flat_map(|f| f.nodes.iter().copied()).collect();
    let mut used = vec![false; m.elements.len()];
    // all elements in the contours so far (in 3D q varies along the front even
    // inside the contour, so the whole domain is integrated)
    let mut domain: Vec<usize> = Vec::new();
    let mut result = vec![vec![Values::default(); req.contours]; nf];
    let mut elastic: Option<(f64, f64)> = None;
    let axi_any = fronts.iter().any(|f| f.nodes.iter().any(|&n| node_elems[n].iter().any(|&e| geos[e].unwrap().family == Family::Axisymmetric)));
    for c in 0..req.contours {
        let ring: Vec<usize> = {
            let mut s = BTreeSet::new();
            for &n in &inside {
                for &e in &node_elems[n] {
                    if !used[e] {
                        s.insert(e);
                    }
                }
            }
            s.into_iter().collect()
        };
        if ring.is_empty() {
            return fail(format!("*CONTOUR INTEGRAL {}: contour {} has no elements (too few elements around the crack tip)", req.crack, c + 1));
        }
        let mut sums = vec![Sums::default(); nf];
        let mut plane_stress = false;
        for &e in &ring {
            used[e] = true;
            domain.push(e);
        }
        // axisymmetry: the terms with q/r are nonzero even where q = 1
        let elems: &[usize] = if three_d || axi_any { &domain } else { &ring };
        for &e in elems {
            let geo = geos[e].unwrap();
            plane_stress = geo.plane_stress;
            if elastic.is_none() {
                elastic = Some(elastic_of(prep, e)?);
            }
            if cache[e].is_none() {
                let ctx = prep.ctx(e, false, time, 0.0);
                let ue = prep.gather(e, u);
                let f = m.elements[e].etype.output(&ctx, &ue, &states[e]).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
                cache[e] = Some(f);
            }
            let fields = cache[e].as_ref().unwrap();
            let (ee, nu) = elastic.unwrap();
            let el = Elastic { e: ee, nu, plane_stress: geo.plane_stress };
            for k in 0..nf {
                let mut qn: Vec<f64> = m.elements[e].nodes.iter().map(|&n| if inside.contains(&n) { weight(k, m.nodes[n]) } else { 0.0 }).collect();
                mid_average(geo.shape, &m.elements[e].nodes, &m.nodes, &inside, &mut qn);
                if qn.iter().all(|&v| v == 0.0) && !three_d {
                    continue;
                }
                if qn.iter().all(|&v| v == 0.0) {
                    continue;
                }
                let ips = ip_data(prep, e, &geo, u, fields, &qn)?;
                for d in &ips {
                    let s = contribution(d, &frames[k], pos[k], frames[k][0], &el, geo.family == Family::Axisymmetric, need_aux);
                    sums[k].j += s.j;
                    for i in 0..4 {
                        sums[k].i[i] += s.i[i];
                    }
                }
            }
        }
        let (ee, nu) = elastic.unwrap();
        let el = Elastic { e: ee, nu, plane_stress };
        let es = el.e_star();
        let mu = el.mu();
        for k in 0..nf {
            let mut norm = if three_d { length[k] } else { 1.0 };
            if axi_any {
                norm *= pos[k][0];
            }
            let f = if req.symm { 2.0 } else { 1.0 };
            let s = sums[k];
            let j = f * s.j / norm;
            let mut k1 = f * 0.5 * es * s.i[0] / norm;
            let mut k2 = 0.5 * es * s.i[1] / norm;
            let mut k3 = mu * s.i[2] / norm;
            if req.symm {
                k2 = 0.0;
                k3 = 0.0;
            }
            if !three_d {
                k3 = 0.0;
            }
            if k1 == 0.0 {
                k1 = 0.0;
            }
            let jks = (k1 * k1 + k2 * k2) / es + k3 * k3 / (2.0 * mu);
            // 3D and axisymmetry: T = E*·I/f + ν E* ε₃₃ with the strain along the front at the front node
            // axisymmetry: ε₃₃ = ε_θθ = u_r/r at the tip
            let e33 = if three_d {
                eps_front[k]
            } else if axi_any {
                u[fronts[k].nodes[0]][0] / pos[k][0]
            } else {
                0.0
            };
            let t = f * es * s.i[3] / norm + nu * es * e33;
            result[k][c] = Values { j, k: [k1, k2, k3], jks, cpd: mts_angle(k1, k2), t };
        }
        for &e in &ring {
            for &n in &m.elements[e].nodes {
                inside.insert(n);
            }
        }
    }
    Ok(result)
}

/// Midside nodes outside the contour get q linearly interpolated along the edge between
/// the corner nodes by distance (q = 0.75 at a quarter-point node
/// next to the crack tip, 0.5 at a regular midside node).
fn mid_average(shape: Shape, nodes: &[usize], x: &[[f64; 3]], inside: &BTreeSet<usize>, qn: &mut [f64]) {
    let edges: &[(usize, usize, usize)] = match shape {
        Shape::Tri6 => &[(3, 0, 1), (4, 1, 2), (5, 2, 0)],
        Shape::Quad8 => &[(4, 0, 1), (5, 1, 2), (6, 2, 3), (7, 3, 0)],
        Shape::Tet10 => &[(4, 0, 1), (5, 1, 2), (6, 2, 0), (7, 0, 3), (8, 1, 3), (9, 2, 3)],
        Shape::Wedge15 => &[(6, 0, 1), (7, 1, 2), (8, 2, 0), (9, 3, 4), (10, 4, 5), (11, 5, 3), (12, 0, 3), (13, 1, 4), (14, 2, 5)],
        Shape::Hex20 => &[
            (8, 0, 1),
            (9, 1, 2),
            (10, 2, 3),
            (11, 3, 0),
            (12, 4, 5),
            (13, 5, 6),
            (14, 6, 7),
            (15, 7, 4),
            (16, 0, 4),
            (17, 1, 5),
            (18, 2, 6),
            (19, 3, 7),
        ],
        _ => &[],
    };
    for &(m, a, b) in edges {
        if m < nodes.len() && !inside.contains(&nodes[m]) {
            let (xa, xb, xm) = (x[nodes[a]], x[nodes[b]], x[nodes[m]]);
            let l = dot(sub(xb, xa), sub(xb, xa)).sqrt();
            let t = if l > 0.0 { dot(sub(xm, xa), sub(xm, xa)).sqrt() / l } else { 0.5 };
            qn[m] = qn[a] + (qn[b] - qn[a]) * t;
        }
    }
}
