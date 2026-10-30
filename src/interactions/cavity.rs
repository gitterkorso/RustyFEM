//! Surface-based fluid- and gas-filled cavities (*FLUID CAVITY) with
//! hydrostatic fluid computation.
//!
//! The cavity volume is the sum of pyramids (3D), triangles (planar, times
//! the thickness) or bodies of revolution of triangles (axisymmetric) formed
//! by the surface's linear facets and the reference node x_R (elements
//! F3D3/F3D4, F2D2 and FAX2; second-order faces are split into linear facets):
//!
//!   V_e = −1/3 ∫ (x − x_R)·n dA,  n pointing into the cavity.
//!
//! Implicit analysis: the pressure p is degree of freedom 8 (slot 7, like
//! POR) at the reference node and the augmented virtual work equation is
//! δΠ* = δΠ − p δV − δp (V − V̄(p, θ, m)). The force on the surface is only
//! the variation of the "base" (the pressure load) p ∂V_b/∂x = −p ∫ Nᵢ n dA,
//! and the tangent [K − p H, −g; −gᵀ, dV̄/dp] is symmetric (H is symmetrized).
//! The system is indefinite, so the step is solved with the indefinite
//! solver (`auto_solver`).
//!
//! Fluids:
//! - hydraulic: V̄ = m/ρ_R [1 + 3α(θ)(θ − θ₀) − 3α(θ_I)(θ_I − θ₀) − p/K]
//!   (incompressible without *FLUID BULK MODULUS), m = ρ(p_I, θ_I) V_I;
//! - ideal gas: ρ = (p + p_A)/(R (θ − θ_Z)), R = R̃/M, V̄ = m/ρ.
//!
//! Prescribed pressure (*BOUNDARY, ref, 8): the fluid mass is adjusted so
//! that the cavity is filled (m = ρ(p, θ) V), and the mass is "sealed" when
//! the boundary condition is removed. *FLUID FLUX changes the mass by q Δt.
//!
//! Explicit analysis: the pressure is computed directly from the volume (only
//! compressible fluids) and gives the pressure load on the surface.
//!
//! Output: PCAV (gauge pressure) and CVOL (volume including ADDED VOLUME) as
//! nodal values at the reference node and as history output (`step_history`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use super::geom;
use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::fem::dof::{NodeVec, POR, U1, U2, U3};
use crate::fem::model::{FeModel, RawKeyword};
use crate::inp::keywords::cavity::BEHAVIOR_OPTIONS;
use crate::materials::thermal::TemperatureField;
use crate::model::{fail, Result};
use crate::steps::dynamic::amplitude::Amplitude;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["FLUID CAVITY"], build }];

/// Slot of the pressure degree of freedom (DOF 8).
const PSLOT: usize = 7;

/// A table v(θ) (one value = constant).
#[derive(Debug, Clone)]
struct Table(Vec<(f64, f64)>);

impl Table {
    fn parse(kw: &RawKeyword) -> Result<Table> {
        if kw.has("DEPENDENCIES") {
            return kw.err("DEPENDENCIES (field dependencies) is not supported yet");
        }
        let mut v = Vec::new();
        for d in kw.records() {
            let f = kw.floats(&d)?;
            if f.is_empty() {
                continue;
            }
            v.push((f[0], f.get(1).copied().unwrap_or(0.0)));
        }
        if v.is_empty() {
            return kw.err("data line missing");
        }
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(Table(v))
    }

    fn at(&self, t: f64) -> f64 {
        let v = &self.0;
        if v.len() == 1 || t <= v[0].1 {
            return v[0].0;
        }
        for w in v.windows(2) {
            if t <= w[1].1 {
                let f = (t - w[0].1) / (w[1].1 - w[0].1).max(1e-300);
                return w[0].0 + f * (w[1].0 - w[0].0);
            }
        }
        v[v.len() - 1].0
    }
}

#[derive(Debug, Clone)]
enum Fluid {
    /// ρ_R, K(θ) (None = incompressible), α(θ) with θ₀.
    Hydraulic { rho: f64, k: Option<Table>, alpha: Option<(Table, f64)> },
    /// Gas constant R = R̃/M, absolute zero θ_Z.
    Gas { r: f64, zero: f64 },
}

impl Fluid {
    /// Thermal volumetric strain 3α(θ)(θ − θ₀) − 3α(θ_I)(θ_I − θ₀).
    fn thermal(&self, th: f64, th_i: f64) -> f64 {
        match self {
            Fluid::Hydraulic { alpha: Some((a, z)), .. } => 3.0 * a.at(th) * (th - z) - 3.0 * a.at(th_i) * (th_i - z),
            _ => 0.0,
        }
    }

    /// Fluid volume V̄(p, θ, m) and dV̄/dp.
    fn volume(&self, p: f64, th: f64, th_i: f64, m: f64, pa: f64) -> (f64, f64) {
        match self {
            Fluid::Hydraulic { rho, k, .. } => {
                let kinv = k.as_ref().map_or(0.0, |k| 1.0 / k.at(th));
                (m / rho * (1.0 + self.thermal(th, th_i) - p * kinv), -m / rho * kinv)
            }
            Fluid::Gas { r, zero } => {
                let v = m * r * (th - zero) / (p + pa);
                (v, -v / (p + pa))
            }
        }
    }

    /// Mass that fills the volume V at pressure p and temperature θ.
    fn mass(&self, v: f64, p: f64, th: f64, th_i: f64, pa: f64) -> f64 {
        let (v1, _) = self.volume(p, th, th_i, 1.0, pa);
        v / v1
    }

    /// Pressure from the volume (explicit): solves V̄(p) = V.
    fn pressure(&self, v: f64, th: f64, th_i: f64, m: f64, pa: f64) -> f64 {
        match self {
            Fluid::Hydraulic { rho, k, .. } => {
                let kk = k.as_ref().map_or(0.0, |k| k.at(th));
                kk * (1.0 + self.thermal(th, th_i) - v * rho / m)
            }
            Fluid::Gas { r, zero } => m * r * (th - zero) / v - pa,
        }
    }
}

/// How the volume is computed.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Geo {
    /// Planar with thickness.
    Planar(f64),
    Axisymmetric,
    ThreeD,
}

pub struct Cavity {
    name: String,
    refn: usize,
    /// Linear facets with nodes ordered so that the normal points into the cavity.
    facets: Vec<Vec<usize>>,
    geo: Geo,
    added: f64,
    pa: f64,
    fluid: Fluid,
    explicit: bool,
    temp: Option<TemperatureField>,
    th_i: f64,
    /// Per (time-carrying) step: prescribed pressure and mass fluxes (q, amplitude).
    prescribed: Vec<bool>,
    fluxes: Vec<Vec<(f64, Option<String>)>>,
    amps: BTreeMap<String, Amplitude>,
    st: Mutex<State>,
}

#[derive(Debug, Clone, Default)]
struct State {
    /// Committed mass and the mass from the latest evaluation.
    m: f64,
    m_tr: f64,
    p: f64,
    v: f64,
    p_tr: f64,
    v_tr: f64,
}

fn trimmed_name(kw: &RawKeyword, key: &str) -> Result<String> {
    match kw.param(key) {
        Some(v) if !v.trim().is_empty() => Ok(v.trim().to_uppercase()),
        _ => kw.err(format!("parameter {key}= missing")),
    }
}

/// *FLUID BEHAVIOR named `name` and its suboptions.
fn fluid(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<Fluid> {
    let Some(start) = m.extra.iter().position(|k| k.name == "FLUID BEHAVIOR" && k.param("NAME").is_some_and(|n| n.trim().eq_ignore_ascii_case(name))) else {
        return kw.err(format!("*FLUID BEHAVIOR, NAME={name} does not exist"));
    };
    let (mut rho, mut k, mut alpha, mut mw) = (None, None, None, None);
    for o in m.extra[start + 1..].iter().take_while(|k| BEHAVIOR_OPTIONS.contains(&k.name.as_str())) {
        match o.name.as_str() {
            "FLUID DENSITY" => rho = o.first_floats()?.first().copied(),
            "FLUID BULK MODULUS" => k = Some(Table::parse(o)?),
            "FLUID EXPANSION" => {
                let z = match o.param("ZERO") {
                    Some(z) => crate::fem::model::parse_f64(z).ok_or_else(|| crate::model::FemError(format!("{}: ZERO={z} is not a number", o.location())))?,
                    None => 0.0,
                };
                alpha = Some((Table::parse(o)?, z));
            }
            "MOLECULAR WEIGHT" => mw = o.first_floats()?.first().copied(),
            // the heat capacity is only needed for adiabatic gases (Explicit)
            _ => {}
        }
    }
    match (rho, mw) {
        (Some(rho), None) => {
            if rho <= 0.0 {
                return kw.err("*FLUID DENSITY must be positive");
            }
            Ok(Fluid::Hydraulic { rho, k, alpha })
        }
        (None, Some(mw)) => {
            let pc = m.extra.iter().rev().find(|k| k.name == "PHYSICAL CONSTANTS");
            let get = |key: &str| -> Result<f64> {
                let v = pc.and_then(|k| k.param(key)).ok_or_else(|| crate::model::FemError(format!("{}: ideal gas in the cavity requires *PHYSICAL CONSTANTS, {key}=", kw.location())))?;
                crate::fem::model::parse_f64(v).ok_or_else(|| crate::model::FemError(format!("{}: {key}={v} is not a number", kw.location())))
            };
            let (rt, zero) = (get("UNIVERSAL GAS CONSTANT")?, get("ABSOLUTE ZERO")?);
            if mw <= 0.0 {
                return kw.err("*MOLECULAR WEIGHT must be positive");
            }
            Ok(Fluid::Gas { r: rt / mw, zero })
        }
        (Some(_), Some(_)) => kw.err(format!("*FLUID BEHAVIOR {name}: both *FLUID DENSITY and *MOLECULAR WEIGHT")),
        (None, None) => kw.err(format!("*FLUID BEHAVIOR {name} has neither *FLUID DENSITY (hydraulic) nor *MOLECULAR WEIGHT (gas)")),
    }
}

fn is_axisymmetric(m: &FeModel, e: usize) -> bool {
    let et = m.elements[e].etype;
    et.family() == crate::elements::Family::Axisymmetric || ["CAX", "SAX", "MAX", "RAX", "SFMAX", "CGAX", "ACAX", "DCAX"].iter().any(|p| et.name().starts_with(p))
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Splits an element face into linear facets (nodes in .inp format order).
fn split(m: &FeModel, nodes: &[usize], dim: usize) -> Result<Vec<Vec<usize>>> {
    let n = nodes;
    Ok(match (dim, n.len()) {
        (2, 2) | (3, 3) | (3, 4) => vec![n.to_vec()],
        (2, 3) => {
            // the midside node is the one closest to the midpoint of the other two
            let x = |i: usize| m.nodes[n[i]];
            let mid = (0..3)
                .min_by(|&a, &b| {
                    let d = |i: usize| {
                        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                        let c = [0, 1, 2].map(|q| 0.5 * (x(j)[q] + x(k)[q]));
                        let s = sub(x(i), c);
                        dot(s, s)
                    };
                    d(a).total_cmp(&d(b))
                })
                .unwrap();
            let (j, k) = ((mid + 1) % 3, (mid + 2) % 3);
            vec![vec![n[j], n[mid]], vec![n[mid], n[k]]]
        }
        (3, 6) => vec![vec![n[0], n[3], n[5]], vec![n[3], n[1], n[4]], vec![n[5], n[4], n[2]], vec![n[3], n[4], n[5]]],
        (3, 8) => vec![
            vec![n[0], n[4], n[7]],
            vec![n[4], n[1], n[5]],
            vec![n[5], n[2], n[6]],
            vec![n[6], n[3], n[7]],
            vec![n[4], n[5], n[7]],
            vec![n[5], n[6], n[7]],
        ],
        (3, 9) => vec![vec![n[0], n[4], n[8], n[7]], vec![n[4], n[1], n[5], n[8]], vec![n[8], n[5], n[2], n[6]], vec![n[7], n[8], n[6], n[3]]],
        _ => return fail(format!("*FLUID CAVITY: faces with {} nodes in {dim}D are not supported", n.len())),
    })
}

/// Normal (not normalized) of a linear facet in the reference configuration
/// with the node order as given: 2D (t_y, −t_x), 3D right-hand rule.
fn ref_normal(x: &[[f64; 3]]) -> [f64; 3] {
    if x.len() == 2 {
        let t = sub(x[1], x[0]);
        [t[1], -t[0], 0.0]
    } else {
        cross(sub(x[1], x[0]), sub(x[x.len() - 1], x[0]))
    }
}

/// Shape functions and derivatives of a bilinear quadrilateral.
fn quad_shape(g: f64, h: f64) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let gs = [-1.0, 1.0, 1.0, -1.0];
    let hs = [-1.0, -1.0, 1.0, 1.0];
    let mut n = [0.0; 4];
    let mut ng = [0.0; 4];
    let mut nh = [0.0; 4];
    for i in 0..4 {
        n[i] = 0.25 * (1.0 + gs[i] * g) * (1.0 + hs[i] * h);
        ng[i] = 0.25 * gs[i] * (1.0 + hs[i] * h);
        nh[i] = 0.25 * hs[i] * (1.0 + gs[i] * g);
    }
    (n, ng, nh)
}

/// Volume contribution V_e of the facet and the base variation G_i = ∂V_b/∂x_i.
fn facet_vg(geo: Geo, x: &[[f64; 3]], xr: [f64; 3]) -> (f64, Vec<[f64; 3]>) {
    let xb: Vec<[f64; 3]> = x.iter().map(|&p| sub(p, xr)).collect();
    match (geo, x.len()) {
        (Geo::Planar(th), 2) => {
            let n = ref_normal(x);
            let v = -0.5 * th * cross(xb[0], xb[1])[2];
            let g = [-0.5 * th * n[0], -0.5 * th * n[1], 0.0];
            (v, vec![g, g])
        }
        (Geo::Axisymmetric, 2) => {
            let n = ref_normal(x);
            let (ra, rb) = (xb[0][0], xb[1][0]);
            let v = -std::f64::consts::PI / 3.0 * cross(xb[0], xb[1])[2] * (ra + rb);
            let c = -std::f64::consts::PI / 3.0;
            // consistent pressure load −2π ∫ Nᵢ r n dl (tested against nodal and mean r)
            let (wa, wb) = (2.0 * ra + rb, ra + 2.0 * rb);
            let ga = [c * n[0] * wa, c * n[1] * wa, 0.0];
            let gb = [c * n[0] * wb, c * n[1] * wb, 0.0];
            (v, vec![ga, gb])
        }
        (_, 3) => {
            let nn = cross(sub(x[1], x[0]), sub(x[2], x[0]));
            let v = -dot(xb[0], cross(xb[1], xb[2])) / 6.0;
            let g = [-nn[0] / 6.0, -nn[1] / 6.0, -nn[2] / 6.0];
            (v, vec![g, g, g])
        }
        _ => {
            // bilinear quadrilateral: 2 × 2 Gauss points integrate exactly
            let a = 1.0 / 3f64.sqrt();
            let mut v = 0.0;
            let mut gr = vec![[0.0; 3]; 4];
            for (g, h) in [(-a, -a), (a, -a), (a, a), (-a, a)] {
                let (n, ng, nh) = quad_shape(g, h);
                let mut xp = [0.0; 3];
                let mut xg = [0.0; 3];
                let mut xh = [0.0; 3];
                for i in 0..4 {
                    for q in 0..3 {
                        xp[q] += n[i] * xb[i][q];
                        xg[q] += ng[i] * xb[i][q];
                        xh[q] += nh[i] * xb[i][q];
                    }
                }
                let nd = cross(xg, xh);
                v -= dot(xp, nd) / 3.0;
                for i in 0..4 {
                    for q in 0..3 {
                        gr[i][q] -= n[i] * nd[q];
                    }
                }
            }
            (v, gr)
        }
    }
}

fn build(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let name = trimmed_name(kw, "NAME")?;
    for p in ["MIXTURE", "ADIABATIC", "SWITCHTIME", "MINIMUM VOLUME", "AMBIENT TEMPERATURE"] {
        if kw.has(p) {
            return kw.err(format!("parameter {p} is not supported yet"));
        }
    }
    let behavior = trimmed_name(kw, "BEHAVIOR")?;
    let fluid = fluid(m, kw, &behavior)?;
    let refs = geom::node_ref(m, kw, kw.param("REF NODE").ok_or_else(|| crate::model::FemError(format!("{}: REF NODE= missing", kw.location())))?)?;
    let [refn] = refs[..] else {
        return kw.err("REF NODE must be exactly one node");
    };
    let num = |key: &str| -> Result<f64> {
        match kw.param(key) {
            Some(v) => crate::fem::model::parse_f64(v).ok_or_else(|| crate::model::FemError(format!("{}: {key}={v} is not a number", kw.location()))),
            None => Ok(0.0),
        }
    };
    let added = num("ADDED VOLUME")?;
    let pa = num("AMBIENT PRESSURE")?;
    let thick = kw.first_floats().ok().and_then(|v| v.first().copied()).filter(|&t| t != 0.0).unwrap_or(1.0);
    let explicit = m.steps.iter().any(|s| s.procedure.as_ref().is_some_and(|p| p.has("EXPLICIT")));
    // the surface and its linear facets
    let mut facets = Vec::new();
    let mut geo = None;
    if let Some(sname) = kw.param("SURFACE") {
        let s = geom::surface(m, kw, sname)?;
        if s.kind == "NODE" {
            return kw.err("SURFACE= must be an element-based surface");
        }
        for f in geom::facets(m, s, "*FLUID CAVITY")? {
            let g = match f.dim {
                2 if is_axisymmetric(m, f.elem) => Geo::Axisymmetric,
                2 => Geo::Planar(thick),
                _ => Geo::ThreeD,
            };
            if geo.is_some_and(|x| x != g) {
                return kw.err("the surface mixes planar, axisymmetric and three-dimensional elements");
            }
            geo = Some(g);
            for mut sf in split(m, &f.nodes, f.dim)? {
                let x: Vec<[f64; 3]> = sf.iter().map(|&n| m.nodes[n]).collect();
                let c = crate::elements::common::centroid(&x);
                // the normal must point away from the element (into the cavity)
                if dot(ref_normal(&x), sub(c, f.inside)) < 0.0 {
                    sf.reverse();
                }
                facets.push(sf);
            }
        }
    } else if !explicit {
        return kw.err("SURFACE= is required in implicit analysis");
    }
    let geo = geo.unwrap_or(Geo::ThreeD);
    if geo == Geo::Axisymmetric && m.nodes[refn][0].abs() > 1e-9 * (1.0 + m.nodes[refn][1].abs()) {
        return kw.err("the reference node of an axisymmetric cavity must lie on the symmetry axis (r = 0)");
    }
    if explicit && matches!(fluid, Fluid::Hydraulic { k: None, .. }) {
        return kw.err("explicit analysis requires a compressible fluid (*FLUID BULK MODULUS)");
    }
    // initial pressure and temperature
    let mut p_init = 0.0;
    for ic in m.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.param_upper("TYPE").as_deref() == Some("FLUID PRESSURE")) {
        for d in ic.records() {
            let Some(tok) = d.fields.first().filter(|t| !t.trim().is_empty()) else { continue };
            if geom::node_ref(m, ic, tok)?.contains(&refn) {
                p_init = d.fields.get(1).and_then(|v| crate::fem::model::parse_f64(v)).unwrap_or(0.0);
            }
        }
    }
    let temp = TemperatureField::build(m)?;
    let th_i = temp.as_ref().map_or(0.0, |t| t.at(&[refn], 0.0)[0]);
    if let Fluid::Gas { zero, .. } = fluid {
        if th_i - zero <= 0.0 {
            return kw.err("the gas temperature is below absolute zero");
        }
    }
    // step-wise boundary conditions on the pressure DOF and mass fluxes
    let mut prescribed = Vec::new();
    let mut fluxes = Vec::new();
    let mut bc = m.initial_bcs.iter().any(|b| b.items.iter().any(|&(n, d, _)| n == refn && d == POR));
    let mut fl: BTreeMap<usize, (f64, Option<String>)> = BTreeMap::new();
    for s in &m.steps {
        if s.perturbation {
            // perturbation step: the state from the previous step
            prescribed.push(bc);
            fluxes.push(Vec::new());
            continue;
        }
        for b in &s.bcs {
            if b.op_new {
                bc = m.initial_bcs.iter().any(|b| b.items.iter().any(|&(n, d, _)| n == refn && d == POR));
            }
            if b.items.iter().any(|&(n, d, _)| n == refn && d == POR) {
                bc = true;
            }
        }
        for k in s.extra.iter().filter(|k| k.name == "FLUID FLUX") {
            if k.param_upper("OP").as_deref() == Some("NEW") {
                fl.clear();
            }
            for d in k.records() {
                let Some(tok) = d.fields.first().filter(|t| !t.trim().is_empty()) else { continue };
                if geom::node_ref(m, k, tok)?.contains(&refn) {
                    let q = d.fields.get(1).and_then(|v| crate::fem::model::parse_f64(v)).unwrap_or(0.0);
                    fl.insert(0, (q, k.param("AMPLITUDE").map(|a| a.trim().to_uppercase())));
                }
            }
        }
        prescribed.push(bc);
        fluxes.push(fl.values().cloned().collect());
    }
    if explicit && fluxes.iter().any(|f| !f.is_empty()) {
        return kw.err("*FLUID FLUX is not supported in explicit analysis");
    }
    let amps = crate::steps::dynamic::amplitude::amplitudes(m)?;
    let mut cav = Cavity { name, refn, facets, geo, added, pa, fluid, explicit, temp, th_i, prescribed, fluxes, amps, st: Mutex::new(State::default()) };
    let v0 = cav.volume_grad(m, &[])?.0;
    if v0 <= 0.0 {
        return kw.err(format!("initial cavity volume {v0:e} is not positive (check the surface normals and the reference node)"));
    }
    // the mass is the sum of the facet masses: ADDED VOLUME carries no fluid
    let m0 = cav.fluid.mass(v0 - added, p_init, th_i, th_i, pa);
    cav.st = Mutex::new(State { m: m0, m_tr: m0, p: p_init, v: v0, p_tr: p_init, v_tr: v0 });
    Ok(vec![Box::new(cav)])
}

impl Cavity {
    /// Node position (reference position + displacement).
    fn pos(m: &FeModel, u: &[NodeVec], n: usize) -> [f64; 3] {
        let x = m.nodes[n];
        match u.get(n) {
            Some(d) => [x[0] + d[0], x[1] + d[1], x[2] + d[2]],
            None => x,
        }
    }

    /// The volume (including ADDED VOLUME) and the base variation per facet.
    fn volume_grad(&self, m: &FeModel, u: &[NodeVec]) -> Result<(f64, Vec<Vec<[f64; 3]>>)> {
        let xr = m.nodes[self.refn];
        let mut v = self.added;
        let mut gs = Vec::with_capacity(self.facets.len());
        for f in &self.facets {
            let x: Vec<[f64; 3]> = f.iter().map(|&n| Self::pos(m, u, n)).collect();
            let (ve, g) = facet_vg(self.geo, &x, xr);
            v += ve;
            gs.push(g);
        }
        Ok((v, gs))
    }

    /// Facet H_ij = ∂G_i/∂x_j (central differences), row-major (3·nn)².
    fn facet_hessian(&self, x: &[[f64; 3]], xr: [f64; 3]) -> Vec<f64> {
        let nn = x.len();
        let nd = 3 * nn;
        let size = x.iter().map(|p| sub(*p, x[0])).map(|d| dot(d, d).sqrt()).fold(0.0, f64::max).max(1e-12);
        let h = 1e-6 * size;
        let dims = if self.geo == Geo::ThreeD { 3 } else { 2 };
        let mut hm = vec![0.0; nd * nd];
        let mut xp = x.to_vec();
        for j in 0..nn {
            for d in 0..dims {
                xp[j][d] = x[j][d] + h;
                let (_, gp) = facet_vg(self.geo, &xp, xr);
                xp[j][d] = x[j][d] - h;
                let (_, gm) = facet_vg(self.geo, &xp, xr);
                xp[j][d] = x[j][d];
                for i in 0..nn {
                    for c in 0..dims {
                        hm[(3 * i + c) * nd + 3 * j + d] = (gp[i][c] - gm[i][c]) / (2.0 * h);
                    }
                }
            }
        }
        hm
    }

    /// Step index (among the time-carrying steps), step time and total time at the end of the increment.
    fn step_info(&self, time: f64) -> (usize, f64) {
        let (kstep, _) = crate::user::context::step_inc();
        let k = (kstep.max(1) as usize - 1).min(self.prescribed.len().saturating_sub(1));
        (k, crate::user::context::step_time(time))
    }

    fn temperature(&self, time: f64) -> f64 {
        self.temp.as_ref().map_or(self.th_i, |t| t.at(&[self.refn], time)[0])
    }

    /// Mass at the end of the increment without prescribed pressure.
    fn mass_at(&self, st: &State, k: usize, step_time: f64, time: f64, dt: f64) -> f64 {
        let mut m = st.m;
        if let Some(fl) = self.fluxes.get(k) {
            for (q, amp) in fl {
                let a = match amp {
                    Some(name) => self.amps.get(name).map_or(1.0, |a| a.value(step_time - 0.5 * dt, time - 0.5 * dt)),
                    None => 1.0,
                };
                m += q * a * dt;
            }
        }
        m
    }
}

impl Interaction for Cavity {
    fn name(&self) -> String {
        format!("FLUID CAVITY {}", self.name)
    }

    fn active_dofs(&self) -> Vec<(usize, crate::fem::dof::Dof)> {
        if self.explicit {
            return Vec::new();
        }
        vec![(self.refn, POR)]
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.facets
            .iter()
            .map(|f| {
                let mut g = f.clone();
                if !self.explicit {
                    g.push(self.refn);
                }
                g
            })
            .collect()
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], k: Option<&mut dyn TangentSink>) -> Result<()> {
        let m = ctx.model;
        let (v, gs) = self.volume_grad(m, u)?;
        let th = self.temperature(ctx.time);
        let mut st = self.st.lock().unwrap();
        let (kstep, step_time) = self.step_info(ctx.time);
        let p;
        let mut dvdp = 0.0;
        if self.explicit {
            p = self.fluid.pressure(v, th, self.th_i, st.m, self.pa);
            st.m_tr = st.m;
        } else {
            p = u.get(self.refn).map_or(0.0, |d| d[PSLOT]);
            let mm = if self.prescribed.get(kstep).copied().unwrap_or(false) {
                // prescribed pressure: fluid flows in/out so that the cavity is filled
                self.fluid.mass(v, p, th, self.th_i, self.pa)
            } else {
                self.mass_at(&st, kstep, step_time, ctx.time, ctx.dt)
            };
            let (vbar, d) = self.fluid.volume(p, th, self.th_i, mm, self.pa);
            dvdp = d;
            st.m_tr = mm;
            fint[self.refn][PSLOT] += -(v - vbar);
        }
        st.p_tr = p;
        st.v_tr = v;
        // pressure load on the surface: f_int −= p G
        let dims = if self.geo == Geo::ThreeD { 3 } else { 2 };
        for (f, g) in self.facets.iter().zip(&gs) {
            for (a, &n) in f.iter().enumerate() {
                for c in 0..dims {
                    fint[n][c] -= p * g[a][c];
                }
            }
        }
        if self.explicit {
            st.p = p;
            st.v = v;
        }
        drop(st);
        let Some(k) = k else { return Ok(()) };
        let dof = [U1, U2, U3];
        let xr = m.nodes[self.refn];
        for (f, g) in self.facets.iter().zip(&gs) {
            let x: Vec<[f64; 3]> = f.iter().map(|&n| Self::pos(m, u, n)).collect();
            let nd = 3 * f.len();
            if p != 0.0 {
                let h = self.facet_hessian(&x, xr);
                for (a, &na) in f.iter().enumerate() {
                    for c in 0..dims {
                        for (b, &nb) in f.iter().enumerate() {
                            for d in 0..dims {
                                let (i, j) = (3 * a + c, 3 * b + d);
                                let v = -p * 0.5 * (h[i * nd + j] + h[j * nd + i]);
                                if v != 0.0 {
                                    k.add(na, dof[c], nb, dof[d], v);
                                }
                            }
                        }
                    }
                }
            }
            for (a, &na) in f.iter().enumerate() {
                for c in 0..dims {
                    k.add(na, dof[c], self.refn, POR, -g[a][c]);
                    k.add(self.refn, POR, na, dof[c], -g[a][c]);
                }
            }
        }
        k.add(self.refn, POR, self.refn, POR, dvdp);
        Ok(())
    }

    fn commit(&self) {
        let mut st = self.st.lock().unwrap();
        st.m = st.m_tr;
        st.p = st.p_tr;
        st.v = st.v_tr;
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let st = self.st.lock().unwrap();
        vec![
            NodeOutput { name: "PCAV".into(), labels: vec!["PCAV".into()], values: vec![(self.refn, vec![st.p])] },
            NodeOutput { name: "CVOL".into(), labels: vec!["CVOL".into()], values: vec![(self.refn, vec![st.v])] },
        ]
    }
}

/// History output PCAV and CVOL (region "Node INSTANCE.node") from the step frames.
pub fn step_history(model: &FeModel, so: &mut crate::fem::output::StepOutput) {
    if !model.extra.iter().any(|k| k.name == "FLUID CAVITY") {
        return;
    }
    let mut hist: BTreeMap<(usize, String), Vec<(f64, f64)>> = BTreeMap::new();
    for fr in &so.frames {
        for (name, _, vals) in &fr.node_extra {
            if name != "PCAV" && name != "CVOL" {
                continue;
            }
            for (n, v) in vals.iter().enumerate() {
                if let Some(&x) = v.first() {
                    hist.entry((n, name.clone())).or_default().push((fr.value, x));
                }
            }
        }
    }
    for ((n, name), data) in hist {
        so.history.push((format!("Node {}", model.node_name(n)), name, data));
    }
}

/// Indefinite solver without a warning about negative eigenvalues (the
/// saddle-point system has one negative eigenvalue per cavity).
struct QuietIndefinite;

static QUIET: QuietIndefinite = QuietIndefinite;

impl crate::solver::LinearSolver for QuietIndefinite {
    fn name(&self) -> &'static str {
        "indefinite"
    }
    fn prepare(&self, g: &crate::solver::Graph) -> Result<Box<dyn crate::solver::SystemMatrix>> {
        crate::solver::indefinite::prepare_quiet(g)
    }
}

/// The solver for a step: the indefinite one instead of the default solver
/// when the model has a cavity in implicit analysis and the user has not
/// chosen a solver.
pub fn auto_solver(model: &FeModel, solver: &'static dyn crate::solver::LinearSolver, forced: bool) -> &'static dyn crate::solver::LinearSolver {
    let explicit = model.steps.iter().any(|s| s.procedure.as_ref().is_some_and(|p| p.has("EXPLICIT")));
    if !forced && !explicit && solver.name() == "direct" && model.extra.iter().any(|k| k.name == "FLUID CAVITY") {
        return &QUIET;
    }
    solver
}
