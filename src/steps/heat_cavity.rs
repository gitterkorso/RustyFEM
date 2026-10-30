//! Cavity radiation in *HEAT TRANSFER.
//!
//! Input:
//! - `*SURFACE PROPERTY, NAME=` + `*EMISSIVITY` (ε, T): surface properties (model data);
//! - `*CAVITY DEFINITION, NAME=, AMBIENT TEMP=, SET PROPERTY`: the cavity as a
//!   list of surfaces (`surface, surface property` per line with SET PROPERTY);
//! - `*RADIATION VIEW FACTOR, CAVITY=, BLOCKING=, REFLECTION=, VTOL=,
//!   INFINITESIMAL=, LUMPED AREA=, INTEGRATION=, SYMMETRY=, OFF` (step data,
//!   inherited by later steps);
//! - `*RADIATION SYMMETRY, NAME=` + `*REFLECTION, TYPE=LINE/PLANE` (reflection);
//! - `*PHYSICAL CONSTANTS` (σ and absolute zero) is required.
//!
//! Formulation: each facet (element face)
//! is isothermal with the radiation potential ηᵢ = Σ_N P_iN (θ_N − θ_Z)⁴, where
//! P_iN = (1/Aᵢ)∫Nᴺ dA. The fluxes per unit area into the facets solve
//!
//!   Σⱼ (δᵢⱼ + (εⱼ − 1)Fᵢⱼ) qⱼ/εⱼ = σ Σⱼ Fᵢⱼ (ηⱼ − ηᵢ) + σ Fᵢ,amb (η_amb − ηᵢ)
//!
//! (REFLECTION=NO: qᵢ = the right-hand side, black body). The solution is linear in η:
//! Qᵢ = Aᵢ qᵢ = Σⱼ Mᵢⱼ ηⱼ + m⁰ᵢ. The nodal flux is Q_N = Σᵢ P_iN Qᵢ and the
//! Jacobian 4 R̄_NM (θ_M − θ_Z)³ with R̄ = Pᵀ M P (unsymmetric; the solver
//! `unsymmetric` is selected automatically). The emissivity is taken at the
//! facet temperature FTEMP = θ_Z + (Σ P_iN (θ_N − θ_Z)⁴)^¼ at the start of the
//! increment, linearly extrapolated from the two most recent increments as a
//! predictor (explicit, without contribution to the Jacobian).
//! With automatic incrementation the increment is redone with half the length when
//! the emissivity changes by more than MXDEM= (default 0.1) during the increment.
//!
//! View factors:
//! - 2D (planar elements): Hottel's crossed strings, exact for straight edges,
//!   after the facets have been clipped against each other's planes;
//! - 3D: lumped area (eq. 3, with the atan regularization) when
//!   R²/max(Aᵢ, Aⱼ) > LUMPED AREA (5), infinitesimal–finite (Hottel, eq. 4)
//!   when the area ratio > INFINITESIMAL (64), otherwise the Mitalas–Stephenson
//!   contour integral (eq. 5) with INTEGRATION (3) Gauss points per edge and
//!   the inner edge integral analytically;
//! - blocking (BLOCKING=ALL, default): the pair is removed if the ray between
//!   the facet midpoints intersects another facet in the cavity;
//! - open cavity (AMBIENT TEMP=): Fᵢ,amb = 1 − Σⱼ Fᵢⱼ if the deviation is
//!   larger than VTOL (0.05), otherwise no radiation to the ambient;
//! - reflection (*REFLECTION): the contribution from the mirror image of each facet
//!   is added to Fᵢⱼ.
//!
//! Second-order faces: the midside nodes are ignored in the view factors but
//! included in P_iN. Axisymmetric cavities, *PERIODIC, *CYCLIC,
//! *MOTION and BLOCKING=PARTIAL give an error.

use crate::elements::heat::{self as hel};
use crate::fem::assembly::{DofMap, NONE};
use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::materials::heat::{read_table, Table};
use crate::model::{fail, FemError, Result};
use crate::solver::SystemMatrix;
use std::collections::BTreeMap;

use super::api::StepType;

/// No step types of its own; used by `steps/heat.rs`.
pub static STEP_TYPES: &[&dyn StepType] = &[];

/// Slot for the temperature NT11.
const TS: usize = 7;

/// A facet (element face) in a cavity.
#[derive(Debug, Clone)]
pub struct Facet {
    pub elem: usize,
    pub face: String,
    /// Indices into the cavity's node list and P_iN.
    pub p: Vec<(usize, f64)>,
    pub area: f64,
    /// The corners (2D: two points) in order; the normal points out of the element.
    pub corners: Vec<[f64; 3]>,
    pub normal: [f64; 3],
    pub center: [f64; 3],
    /// Index of the surface property in `Radiation::props`.
    prop: usize,
}

/// View factor settings from *RADIATION VIEW FACTOR.
#[derive(Debug, Clone)]
struct VfOpts {
    blocking: bool,
    reflection: bool,
    vtol: f64,
    infinitesimal: f64,
    lumped: f64,
    integration: usize,
    /// Reflections (point, normal) from *RADIATION SYMMETRY/*REFLECTION.
    mirrors: Vec<([f64; 3], [f64; 3])>,
}

pub struct Cavity {
    pub name: String,
    pub facets: Vec<Facet>,
    /// The cavity's nodes (global indices).
    pub nodes: Vec<usize>,
    /// View factors Fᵢⱼ (nf × nf, row-major).
    pub f: Vec<f64>,
    /// Fᵢ,amb (open cavity).
    pub famb: Vec<f64>,
    ambient: Option<f64>,
    opts: VfOpts,
    /// R̄_NM (nodes × nodes) and m⁰_N (nodal flux from the ambient).
    rbar: Vec<f64>,
    m0: Vec<f64>,
    /// Emissivities that R̄ was computed with.
    eps: Vec<f64>,
}

/// All active cavities in a step.
pub struct Radiation {
    pub cavities: Vec<Cavity>,
    /// Largest allowed emissivity change per increment (MXDEM), None = no check.
    pub mxdem: Option<f64>,
    /// (time, temperatures) at the start of the current and previous increment, for
    /// the linear extrapolation of the temperature at which the emissivity is taken.
    cur: Option<(f64, Vec<f64>)>,
    prev: Option<(f64, Vec<f64>)>,
    props: Vec<Table>,
    abs_zero: f64,
    sigma: f64,
}

fn num(kw: &RawKeyword, key: &str) -> Result<Option<f64>> {
    match kw.param(key) {
        Some(v) => parse_f64(v).map(Some).ok_or_else(|| FemError(format!("{}: {key}={v} is not a number", kw.location()))),
        None => Ok(None),
    }
}

/// Surface properties: name → emissivity table.
fn surface_properties(model: &FeModel) -> Result<BTreeMap<String, Table>> {
    let mut out = BTreeMap::new();
    let mut current: Option<String> = None;
    for kw in &model.extra {
        match kw.name.as_str() {
            "SURFACE PROPERTY" => {
                let Some(n) = kw.param_upper("NAME") else { return kw.err("NAME= is missing") };
                current = Some(n);
            }
            "EMISSIVITY" => {
                let Some(n) = current.clone() else { return kw.err("*EMISSIVITY must come directly after *SURFACE PROPERTY") };
                let t = read_table(kw, 1)?;
                for (_, v) in &t.pts {
                    if !(0.0..=1.0).contains(&v[0]) {
                        return kw.err(format!("the emissivity {} must lie in [0, 1]", v[0]));
                    }
                }
                out.insert(n, t);
            }
            _ => {}
        }
    }
    Ok(out)
}

fn physical_constants(model: &FeModel) -> Result<(Option<f64>, Option<f64>)> {
    let mut z = None;
    let mut s = None;
    for kw in model.extra.iter().filter(|k| k.name == "PHYSICAL CONSTANTS") {
        if let Some(v) = num(kw, "ABSOLUTE ZERO")? {
            z = Some(v);
        }
        if let Some(v) = num(kw, "STEFAN BOLTZMANN")? {
            s = Some(v);
        }
    }
    Ok((z, s))
}

/// Does the model have any cavity?
pub fn has_cavities(model: &FeModel) -> bool {
    model.extra.iter().any(|k| k.name == "CAVITY DEFINITION")
}

/// View factor settings per cavity that apply in step `index` (they are inherited
/// from earlier steps). `None` = no cavity radiation (OFF or never
/// requested).
fn step_options(model: &FeModel, index: usize, names: &[String]) -> Result<BTreeMap<String, Option<VfOpts>>> {
    let mut out: BTreeMap<String, Option<VfOpts>> = BTreeMap::new();
    for step in model.steps.iter().take(index + 1) {
        // reflections in the step (*RADIATION SYMMETRY followed by *REFLECTION)
        let mut syms: BTreeMap<String, Vec<([f64; 3], [f64; 3])>> = BTreeMap::new();
        let mut cur: Option<String> = None;
        for kw in &step.extra {
            match kw.name.as_str() {
                "RADIATION SYMMETRY" => {
                    let Some(n) = kw.param_upper("NAME") else { return kw.err("NAME= is missing") };
                    syms.entry(n.clone()).or_default();
                    cur = Some(n);
                }
                "REFLECTION" => {
                    let Some(n) = cur.clone() else { return kw.err("*REFLECTION must come after *RADIATION SYMMETRY") };
                    let ty = kw.param_upper("TYPE").unwrap_or_default();
                    // TYPE=PLANE has the points on two lines
                    let mut v = Vec::new();
                    for d in &kw.data {
                        v.extend(kw.floats(d)?);
                    }
                    let m = match ty.as_str() {
                        "LINE" => {
                            // two points on the line
                            if v.len() < 4 {
                                return kw.err("TYPE=LINE: write 'x1, y1, x2, y2'");
                            }
                            let t = [v[2] - v[0], v[3] - v[1], 0.0];
                            let l = (t[0] * t[0] + t[1] * t[1]).sqrt();
                            if l == 0.0 {
                                return kw.err("the points coincide");
                            }
                            ([v[0], v[1], 0.0], [-t[1] / l, t[0] / l, 0.0])
                        }
                        "PLANE" => {
                            // three points in the plane
                            if v.len() < 9 {
                                return kw.err("TYPE=PLANE: write three points (9 coordinates)");
                            }
                            let a = [v[0], v[1], v[2]];
                            let n = cross(sub([v[3], v[4], v[5]], a), sub([v[6], v[7], v[8]], a));
                            let l = norm(n);
                            if l == 0.0 {
                                return kw.err("the points lie on a line");
                            }
                            (a, scale(n, 1.0 / l))
                        }
                        _ => return kw.err(format!("TYPE={ty} is not supported (LINE, PLANE)")),
                    };
                    syms.get_mut(&n).unwrap().push(m);
                }
                "PERIODIC" | "CYCLIC" => return kw.err(format!("*{} for cavity radiation is not supported yet", kw.name)),
                "MOTION" => return kw.err("*MOTION (moving cavity surfaces) is not supported yet"),
                _ => {}
            }
        }
        for kw in step.extra.iter().filter(|k| k.name == "RADIATION VIEW FACTOR") {
            let which: Vec<String> = match kw.param_upper("CAVITY") {
                Some(c) => {
                    if !names.contains(&c) {
                        return kw.err(format!("the cavity {c} does not exist (*CAVITY DEFINITION)"));
                    }
                    vec![c]
                }
                None => names.to_vec(),
            };
            if kw.has("OFF") {
                for c in which {
                    out.insert(c, None);
                }
                continue;
            }
            let blocking = match kw.param_upper("BLOCKING").as_deref() {
                None | Some("ALL") => true,
                Some("NO") => false,
                Some(o) => return kw.err(format!("BLOCKING={o} is not supported (ALL, NO)")),
            };
            let reflection = match kw.param_upper("REFLECTION").as_deref() {
                None | Some("YES") => true,
                Some("NO") => false,
                Some(o) => return kw.err(format!("REFLECTION={o} is unknown (YES, NO)")),
            };
            if kw.has("RANGE") {
                return kw.err("RANGE= is not supported yet");
            }
            let mirrors = match kw.param_upper("SYMMETRY") {
                Some(s) => match syms.get(&s) {
                    Some(m) => m.clone(),
                    None => return kw.err(format!("the symmetry {s} is missing in the step (*RADIATION SYMMETRY, NAME={s})")),
                },
                None => Vec::new(),
            };
            let o = VfOpts {
                blocking,
                reflection,
                vtol: num(kw, "VTOL")?.unwrap_or(0.05),
                infinitesimal: num(kw, "INFINITESIMAL")?.unwrap_or(64.0),
                lumped: num(kw, "LUMPED AREA")?.unwrap_or(5.0),
                integration: num(kw, "INTEGRATION")?.unwrap_or(3.0).clamp(1.0, 5.0) as usize,
                mirrors,
            };
            for c in which {
                out.insert(c, Some(o.clone()));
            }
        }
    }
    Ok(out)
}

/// MXDEM= on the procedure line (*HEAT TRANSFER), default 0.1.
pub fn mxdem(step: &crate::fem::model::StepDef) -> Option<f64> {
    let v = step.procedure.as_ref().and_then(|k| k.param("MXDEM")).and_then(parse_f64);
    Some(v.unwrap_or(0.1))
}

/// The cavity radiation in step `index` (None if no cavity is active).
pub fn for_step(model: &FeModel, index: usize) -> Result<Option<Radiation>> {
    if !has_cavities(model) {
        return Ok(None);
    }
    let props_by_name = surface_properties(model)?;
    let mut prop_names: Vec<String> = Vec::new();
    let mut props: Vec<Table> = Vec::new();
    let defs: Vec<&RawKeyword> = model.extra.iter().filter(|k| k.name == "CAVITY DEFINITION").collect();
    let mut names = Vec::new();
    for kw in &defs {
        let Some(n) = kw.param_upper("NAME") else { return kw.err("NAME= is missing") };
        names.push(n);
    }
    let opts = step_options(model, index, &names)?;
    let (z, s) = physical_constants(model)?;
    let mut cavities = Vec::new();
    for (kw, name) in defs.iter().zip(&names) {
        let Some(Some(o)) = opts.get(name) else { continue };
        if z.is_none() || s.is_none() {
            return kw.err("cavity radiation requires *PHYSICAL CONSTANTS, ABSOLUTE ZERO=..., STEFAN BOLTZMANN=...");
        }
        if kw.param_upper("PARALLEL DECOMPOSITION").as_deref() == Some("ON") {
            // same equations, just a different solution method
        }
        let mut pairs: Vec<(String, String)> = Vec::new();
        for d in kw.records() {
            let f: Vec<String> = d.fields.iter().map(|x| x.trim().to_uppercase()).filter(|x| !x.is_empty()).collect();
            if kw.has("SET PROPERTY") {
                for c in f.chunks(2) {
                    if c.len() < 2 {
                        return kw.err_at(d.line, "SET PROPERTY: write 'surface, surface property'");
                    }
                    pairs.push((c[0].clone(), c[1].clone()));
                }
            } else {
                return kw.err_at(d.line, "the surface property must be given with SET PROPERTY ('surface, surface property'); PROPERTY= on *SURFACE is not supported yet");
            }
        }
        let mut facets = Vec::new();
        let mut node_idx: BTreeMap<usize, usize> = BTreeMap::new();
        let mut nodes = Vec::new();
        for (sname, pname) in &pairs {
            let Some(surf) = model.surfaces.get(sname) else { return kw.err(format!("the surface {sname} does not exist")) };
            let Some(t) = props_by_name.get(pname) else { return kw.err(format!("the surface property {pname} has no *EMISSIVITY")) };
            let pi = match prop_names.iter().position(|p| p == pname) {
                Some(i) => i,
                None => {
                    prop_names.push(pname.clone());
                    props.push(t.clone());
                    props.len() - 1
                }
            };
            for (e, label) in &surf.faces {
                let fct = build_facet(model, *e, label, pi, &mut node_idx, &mut nodes).map_err(|err| FemError(format!("cavity {name}, surface {sname}: {}", err.0)))?;
                facets.push(fct);
            }
        }
        if facets.is_empty() {
            return kw.err("the cavity has no facets");
        }
        let dim3 = facets[0].corners.len() > 2;
        let nf = facets.len();
        let f = view_factors(&facets, o, dim3);
        // debugging: RUSTYFEM_VF_DUMP=file writes the view factors (element, face, area, row)
        if let Ok(path) = std::env::var("RUSTYFEM_VF_DUMP") {
            let mut s = String::new();
            for (i, fc) in facets.iter().enumerate() {
                let row: Vec<String> = f[i * nf..(i + 1) * nf].iter().map(|v| format!("{v:.15e}")).collect();
                s += &format!("{} {} {:.15e} {}\n", model.elements[fc.elem].label, &fc.face[1..], fc.area, row.join(" "));
            }
            let _ = std::fs::write(path, s);
        }
        let ambient = num(kw, "AMBIENT TEMP")?;
        let mut famb = vec![0.0; nf];
        for i in 0..nf {
            let sum: f64 = f[i * nf..(i + 1) * nf].iter().sum();
            if ambient.is_some() && (1.0 - sum).abs() > o.vtol {
                famb[i] = (1.0 - sum).max(0.0);
            }
        }
        let nn = nodes.len();
        cavities.push(Cavity {
            name: name.clone(),
            facets,
            nodes,
            f,
            famb,
            ambient,
            opts: o.clone(),
            rbar: vec![0.0; nn * nn],
            m0: vec![0.0; nn],
            eps: Vec::new(),
        });
    }
    if cavities.is_empty() {
        return Ok(None);
    }
    Ok(Some(Radiation { cavities, mxdem: None, cur: None, prev: None, props, abs_zero: z.unwrap_or(0.0), sigma: s.unwrap_or(0.0) }))
}

fn build_facet(model: &FeModel, e: usize, label: &str, prop: usize, node_idx: &mut BTreeMap<usize, usize>, nodes: &mut Vec<usize>) -> Result<Facet> {
    let el = &model.elements[e];
    let name = el.etype.name();
    let Some(he) = hel::lookup(name).or_else(|| crate::elements::thermal_electric::lookup(name)) else {
        return fail(format!("{}: element type {name} cannot be part of a cavity (heat transfer elements are required)", model.elem_name(e)));
    };
    if he.axisym {
        return fail(format!("{}: axisymmetric cavities are not supported yet", model.elem_name(e)));
    }
    let f = he.face_index(label)?;
    let thickness = el.section.map(|s| model.sections[s].solid_thickness()).unwrap_or(1.0);
    let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
    let pts = he.face_points(&x, f, thickness, false)?;
    let area: f64 = pts.iter().map(|p| p.w).sum();
    if area <= 0.0 {
        return fail(format!("{}: face {label} has no area", model.elem_name(e)));
    }
    let fnodes = he.faces()[f];
    let dim = he.dim();
    let ncorner = if dim == 2 { 2 } else if fnodes.len() == 3 || fnodes.len() == 6 { 3 } else { 4 };
    let corners: Vec<[f64; 3]> = fnodes[..ncorner].iter().map(|&i| x[i]).collect();
    let mut center = [0.0; 3];
    for c in &corners {
        for k in 0..3 {
            center[k] += c[k] / ncorner as f64;
        }
    }
    let mut ecen = [0.0; 3];
    for xi in &x {
        for k in 0..3 {
            ecen[k] += xi[k] / x.len() as f64;
        }
    }
    let mut normal = if dim == 2 {
        let t = sub(corners[1], corners[0]);
        [t[1], -t[0], 0.0]
    } else {
        // Newell's method for the polygon normal
        let mut n = [0.0; 3];
        for a in 0..ncorner {
            let p = corners[a];
            let q = corners[(a + 1) % ncorner];
            n[0] += (p[1] - q[1]) * (p[2] + q[2]);
            n[1] += (p[2] - q[2]) * (p[0] + q[0]);
            n[2] += (p[0] - q[0]) * (p[1] + q[1]);
        }
        n
    };
    let l = norm(normal);
    normal = scale(normal, 1.0 / l);
    // the normal out of the element
    if dot(normal, sub(center, ecen)) < 0.0 {
        normal = scale(normal, -1.0);
    }
    let mut p = Vec::new();
    for (i, &n) in el.nodes.iter().enumerate() {
        let w: f64 = pts.iter().map(|fp| fp.n[i] * fp.w).sum::<f64>() / area;
        if w != 0.0 && fnodes.contains(&i) {
            let k = *node_idx.entry(n).or_insert_with(|| {
                nodes.push(n);
                nodes.len() - 1
            });
            p.push((k, w));
        }
    }
    Ok(Facet { elem: e, face: label.to_string(), p, area, corners, normal, center, prop })
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

/// Reflects the point `x` in the plane/line (point `o`, unit normal `n`).
fn mirror(x: [f64; 3], o: [f64; 3], n: [f64; 3]) -> [f64; 3] {
    let d = dot(sub(x, o), n);
    sub(x, scale(n, 2.0 * d))
}

/// The mirror image of the facet (the corner order is reversed so that the polygon keeps its
/// orientation relative to the normal).
fn mirrored(f: &Facet, o: [f64; 3], n: [f64; 3]) -> Facet {
    let mut g = f.clone();
    g.corners = f.corners.iter().rev().map(|&c| mirror(c, o, n)).collect();
    g.center = mirror(f.center, o, n);
    let nn = sub(f.normal, scale(n, 2.0 * dot(f.normal, n)));
    g.normal = nn;
    g
}

// ---------------------------------------------------------------------------
// View factors
// ---------------------------------------------------------------------------

/// Clips the polygon (2D: segment) against the half-space in front of the plane (point `o`,
/// normal `n`). Returns an empty vector if nothing lies in front.
fn clip(poly: &[[f64; 3]], o: [f64; 3], n: [f64; 3], segment: bool) -> Vec<[f64; 3]> {
    let tol = 1e-12 * poly.iter().map(|p| norm(sub(*p, o))).fold(0.0, f64::max).max(1e-300);
    let d: Vec<f64> = poly.iter().map(|p| dot(sub(*p, o), n)).collect();
    if d.iter().all(|&v| v <= tol) {
        return Vec::new();
    }
    if d.iter().all(|&v| v >= -tol) {
        return poly.to_vec();
    }
    if segment {
        let (a, b) = (poly[0], poly[1]);
        let t = d[0] / (d[0] - d[1]);
        let c = add(a, scale(sub(b, a), t));
        return if d[0] > 0.0 { vec![a, c] } else { vec![c, b] };
    }
    let mut out = Vec::new();
    let m = poly.len();
    for k in 0..m {
        let (p, q) = (poly[k], poly[(k + 1) % m]);
        let (dp, dq) = (d[k], d[(k + 1) % m]);
        if dp >= 0.0 {
            out.push(p);
        }
        if (dp > 0.0 && dq < 0.0) || (dp < 0.0 && dq > 0.0) {
            let t = dp / (dp - dq);
            out.push(add(p, scale(sub(q, p), t)));
        }
    }
    out
}

/// Aᵢ Fᵢⱼ in 2D (per unit thickness times the thickness `t`): crossed strings.
fn vf2d(a: &Facet, b: &Facet) -> f64 {
    let pa = clip(&a.corners, b.corners[0], b.normal, true);
    if pa.is_empty() {
        return 0.0;
    }
    let pb = clip(&b.corners, a.corners[0], a.normal, true);
    if pb.is_empty() {
        return 0.0;
    }
    let crossed = dist(pa[0], pb[0]) + dist(pa[1], pb[1]);
    let uncrossed = dist(pa[0], pb[1]) + dist(pa[1], pb[0]);
    0.5 * (crossed - uncrossed).abs()
}

/// ∫ ln|y − x| along the segment a → b (length L) for the point x: analytically.
fn line_log(a: [f64; 3], b: [f64; 3], x: [f64; 3]) -> f64 {
    let e = sub(b, a);
    let l = norm(e);
    if l == 0.0 {
        return 0.0;
    }
    let e = scale(e, 1.0 / l);
    let r = sub(a, x);
    let s0 = dot(r, e);
    let h2 = (dot(r, r) - s0 * s0).max(0.0);
    let h = h2.sqrt();
    let prim = |s: f64| -> f64 {
        let q = s * s + h2;
        let lg = if q > 0.0 { s * q.ln() } else { 0.0 };
        let at = if h > 0.0 { 2.0 * h * (s / h).atan() } else { 0.0 };
        0.5 * (lg - 2.0 * s + at)
    };
    prim(s0 + l) - prim(s0)
}

fn gauss(n: usize) -> (&'static [f64], &'static [f64]) {
    match n {
        1 => (&[0.0], &[2.0]),
        2 => (&[-0.577_350_269_189_625_8, 0.577_350_269_189_625_8], &[1.0, 1.0]),
        4 => (
            &[-0.861_136_311_594_052_6, -0.339_981_043_584_856_3, 0.339_981_043_584_856_3, 0.861_136_311_594_052_6],
            &[0.347_854_845_137_453_9, 0.652_145_154_862_546_1, 0.652_145_154_862_546_1, 0.347_854_845_137_453_9],
        ),
        5 => (
            &[-0.906_179_845_938_664, -0.538_469_310_105_683_1, 0.0, 0.538_469_310_105_683_1, 0.906_179_845_938_664],
            &[0.236_926_885_056_189_1, 0.478_628_670_499_366_5, 0.568_888_888_888_888_9, 0.478_628_670_499_366_5, 0.236_926_885_056_189_1],
        ),
        _ => (&[-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4], &[0.555_555_555_555_555_6, 0.888_888_888_888_888_9, 0.555_555_555_555_555_6]),
    }
}

/// Area of the polygon.
fn poly_area(p: &[[f64; 3]]) -> f64 {
    let mut n = [0.0; 3];
    for k in 1..p.len().saturating_sub(1) {
        n = add(n, cross(sub(p[k], p[0]), sub(p[k + 1], p[0])));
    }
    0.5 * norm(n)
}

/// Orients the polygon counterclockwise about the normal `n`.
fn orient(p: Vec<[f64; 3]>, n: [f64; 3]) -> Vec<[f64; 3]> {
    let mut s = [0.0; 3];
    for k in 1..p.len().saturating_sub(1) {
        s = add(s, cross(sub(p[k], p[0]), sub(p[k + 1], p[0])));
    }
    if dot(s, n) < 0.0 {
        p.into_iter().rev().collect()
    } else {
        p
    }
}

/// ∫∫ ln|s − t| ds dt for two edges on the same line that overlap (length
/// > 0); None otherwise.
fn colinear_overlap(a0: [f64; 3], a1: [f64; 3], b0: [f64; 3], b1: [f64; 3]) -> Option<f64> {
    let ea = sub(a1, a0);
    let la = norm(ea);
    let eb = sub(b1, b0);
    let tol = 1e-10 * la;
    if norm(cross(ea, eb)) > tol * norm(eb) || norm(cross(ea, sub(b0, a0))) > tol * la {
        return None;
    }
    let e = scale(ea, 1.0 / la);
    let (s1, s2) = (0.0, la);
    let (mut t1, mut t2) = (dot(sub(b0, a0), e), dot(sub(b1, a0), e));
    if t1 > t2 {
        std::mem::swap(&mut t1, &mut t2);
    }
    if t2.min(s2) - t1.max(s1) <= tol {
        return None;
    }
    // G(u) = u²/2 ln|u| − 3u²/4 with ∂²G(t − s)/∂s∂t = −ln|t − s|
    let g = |u: f64| if u != 0.0 { 0.5 * u * u * u.abs().ln() - 0.75 * u * u } else { 0.0 };
    Some(-(g(t2 - s2) - g(t2 - s1) - g(t1 - s2) + g(t1 - s1)))
}

/// Aᵢ Fᵢⱼ in 3D.
fn vf3d(a: &Facet, b: &Facet, o: &VfOpts) -> f64 {
    let pa = clip(&a.corners, b.center, b.normal, false);
    let pb = clip(&b.corners, a.center, a.normal, false);
    if pa.len() < 3 || pb.len() < 3 {
        return 0.0;
    }
    let (aa, ab) = (poly_area(&pa), poly_area(&pb));
    if aa <= 0.0 || ab <= 0.0 {
        return 0.0;
    }
    let ca = centroid(&pa);
    let cb = centroid(&pb);
    let rv = sub(cb, ca);
    let r2 = dot(rv, rv);
    // R/√A ≥ LUMPED AREA
    if r2 > 0.0 && r2 / aa.max(ab) >= o.lumped * o.lumped * (1.0 - 1e-9) {
        // lumped area with atan regularization (eq. 3)
        let r = r2.sqrt();
        let ci = dot(a.normal, rv) / r;
        let cj = -dot(b.normal, rv) / r;
        if ci <= 0.0 || cj <= 0.0 {
            return 0.0;
        }
        let pi = std::f64::consts::PI;
        return 4.0 * (aa * ab).sqrt() / (pi * pi) * ((pi * aa).sqrt() * ci / (2.0 * r)).atan() * ((pi * ab).sqrt() * cj / (2.0 * r)).atan();
    }
    let ratio = aa.max(ab) / aa.min(ab);
    if ratio > o.infinitesimal {
        // infinitesimal–finite (Hottel): the point at the midpoint of the smaller face
        let (small, sn, big) = if aa < ab { (ca, a.normal, &pb) } else { (cb, b.normal, &pa) };
        let mut s = 0.0;
        let m = big.len();
        for k in 0..m {
            let p = sub(big[k], small);
            let q = sub(big[(k + 1) % m], small);
            let g = cross(p, q);
            let gl = norm(g);
            if gl == 0.0 {
                continue;
            }
            let ang = (dot(p, q) / (norm(p) * norm(q))).clamp(-1.0, 1.0).acos();
            s += ang * dot(g, sn) / gl;
        }
        return aa.min(ab) * (s / (2.0 * std::f64::consts::PI)).abs();
    }
    // contour integral (Mitalas–Stephenson): (1/2π) ∮∮ ln r dsᵢ·dsⱼ
    let pa = orient(pa, a.normal);
    let pb = orient(pb, b.normal);
    let (gx, gw) = gauss(o.integration);
    let mut s = 0.0;
    let (ma, mb) = (pa.len(), pb.len());
    for p in 0..mb {
        let (b0, b1) = (pb[p], pb[(p + 1) % mb]);
        let eb = sub(b1, b0);
        let lb = norm(eb);
        if lb == 0.0 {
            continue;
        }
        for q in 0..ma {
            let (a0, a1) = (pa[q], pa[(q + 1) % ma]);
            let ea = sub(a1, a0);
            let la = norm(ea);
            if la == 0.0 {
                continue;
            }
            let phi = dot(ea, eb) / (la * lb);
            if phi.abs() < 1e-14 {
                continue;
            }
            let mut acc = 0.0;
            // overlapping edges (shared edge): the double integral
            // analytically; the others with Gauss along the outer edge
            if let Some(v) = colinear_overlap(a0, a1, b0, b1) {
                acc = v;
            } else {
                for (xg, wg) in gx.iter().zip(gw) {
                    let x = add(b0, scale(eb, 0.5 * (1.0 + xg)));
                    acc += wg * 0.5 * lb * line_log(a0, a1, x);
                }
            }
            s += phi * acc;
        }
    }
    // dsᵢ·dsⱼ with oppositely oriented normals (the faces facing each other)
    (s / (2.0 * std::f64::consts::PI)).abs()
}

fn centroid(p: &[[f64; 3]]) -> [f64; 3] {
    let mut c = [0.0; 3];
    let mut w = 0.0;
    for k in 1..p.len().saturating_sub(1) {
        let a = 0.5 * norm(cross(sub(p[k], p[0]), sub(p[k + 1], p[0])));
        let m = scale(add(add(p[0], p[k]), p[k + 1]), 1.0 / 3.0);
        c = add(c, scale(m, a));
        w += a;
    }
    if w > 0.0 {
        scale(c, 1.0 / w)
    } else {
        p[0]
    }
}

/// Does the segment p → q (2D) intersect the facet g (segment) in its interior?
fn hits2d(p: [f64; 3], q: [f64; 3], g: &Facet) -> bool {
    let (a, b) = (g.corners[0], g.corners[1]);
    let d = sub(q, p);
    let e = sub(b, a);
    let den = d[0] * e[1] - d[1] * e[0];
    if den.abs() < 1e-300 {
        return false;
    }
    let w = sub(a, p);
    let t = (w[0] * e[1] - w[1] * e[0]) / den;
    let s = (w[0] * d[1] - w[1] * d[0]) / den;
    let eps = 1e-9;
    t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps
}

/// Does the segment p → q (3D) intersect the polygon g?
fn hits3d(p: [f64; 3], q: [f64; 3], g: &Facet) -> bool {
    let d = sub(q, p);
    let den = dot(d, g.normal);
    if den.abs() < 1e-300 {
        return false;
    }
    let t = dot(sub(g.corners[0], p), g.normal) / den;
    let eps = 1e-9;
    if !(t > eps && t < 1.0 - eps) {
        return false;
    }
    let x = add(p, scale(d, t));
    // the point in the polygon: same sign on all edge products
    let m = g.corners.len();
    let mut sign = 0.0;
    for k in 0..m {
        let c = dot(cross(sub(g.corners[(k + 1) % m], g.corners[k]), sub(x, g.corners[k])), g.normal);
        if c.abs() < 1e-14 * norm(sub(g.corners[(k + 1) % m], g.corners[k])).powi(2) {
            return false;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

/// The view factor matrix Fᵢⱼ (row-major).
fn view_factors(facets: &[Facet], o: &VfOpts, dim3: bool) -> Vec<f64> {
    let nf = facets.len();
    let mut af = vec![0.0; nf * nf];
    // the mirror images (one per reflection, plus combinations if several)
    let mut images: Vec<Vec<Facet>> = vec![facets.to_vec()];
    for &(po, pn) in &o.mirrors {
        let cur = images.clone();
        for set in cur {
            images.push(set.iter().map(|f| mirrored(f, po, pn)).collect());
        }
    }
    let all: Vec<&Facet> = images.iter().flatten().collect();
    for i in 0..nf {
        for (k, set) in images.iter().enumerate() {
            for j in 0..nf {
                if k == 0 && j <= i {
                    continue;
                }
                let (a, b) = (&facets[i], &set[j]);
                let v = if dim3 { vf3d(a, b, o) } else { vf2d(a, b) };
                if v <= 0.0 {
                    continue;
                }
                if o.blocking {
                    let blocked = all.iter().any(|g| {
                        !std::ptr::eq(*g, a) && !std::ptr::eq(*g, b) && if dim3 { hits3d(a.center, b.center, g) } else { hits2d(a.center, b.center, g) }
                    });
                    if blocked {
                        continue;
                    }
                }
                // 2D: per unit thickness → times the thickness (facet area/length)
                let v = if dim3 { v } else { v * a.area / dist(a.corners[0], a.corners[1]) };
                if k == 0 {
                    af[i * nf + j] += v;
                    af[j * nf + i] += v;
                } else {
                    // the mirror image of j seen from i (all pairs, also j ≤ i)
                    af[i * nf + j] += v;
                }
            }
        }
    }
    for i in 0..nf {
        for j in 0..nf {
            af[i * nf + j] /= facets[i].area;
        }
    }
    af
}

// ---------------------------------------------------------------------------
// Radiation fluxes
// ---------------------------------------------------------------------------

/// Solves A X = B (A n × n, B n × m, row-major) with LU and partial pivoting.
fn solve_dense(mut a: Vec<f64>, n: usize, mut b: Vec<f64>, m: usize) -> Result<Vec<f64>> {
    for k in 0..n {
        let mut piv = k;
        for r in k + 1..n {
            if a[r * n + k].abs() > a[piv * n + k].abs() {
                piv = r;
            }
        }
        if a[piv * n + k].abs() < 1e-300 {
            return fail("the cavity radiation equation system is singular");
        }
        if piv != k {
            for c in 0..n {
                a.swap(k * n + c, piv * n + c);
            }
            for c in 0..m {
                b.swap(k * m + c, piv * m + c);
            }
        }
        let d = a[k * n + k];
        for r in k + 1..n {
            let f = a[r * n + k] / d;
            if f == 0.0 {
                continue;
            }
            for c in k..n {
                a[r * n + c] -= f * a[k * n + c];
            }
            for c in 0..m {
                b[r * m + c] -= f * b[k * m + c];
            }
        }
    }
    for k in (0..n).rev() {
        let d = a[k * n + k];
        for c in 0..m {
            let mut s = b[k * m + c];
            for j in k + 1..n {
                s -= a[k * n + j] * b[j * m + c];
            }
            b[k * m + c] = s / d;
        }
    }
    Ok(b)
}

impl Radiation {
    /// The facet temperature (FTEMP) θ_Z + (Σ P_iN (θ_N − θ_Z)⁴)^¼, at which
    /// the emissivity is taken.
    fn facet_temp(c: &Cavity, f: &Facet, u: &[NodeVec], z: f64) -> f64 {
        let eta: f64 = f.p.iter().map(|&(k, w)| w * (u[c.nodes[k]][TS] - z).max(0.0).powi(4)).sum();
        z + eta.max(0.0).powf(0.25)
    }

    /// Start of an increment at time `t0` with length `dt`: the emissivity
    /// is taken at the temperature extrapolated linearly from the two
    /// most recent increments (θₙ + (θₙ − θₙ₋₁) Δt/Δtₙ₋₁; in the first increment
    /// θₙ).
    /// Prescribed temperatures (`constrained`) are taken from `u_bc` (the value at
    /// the end of the increment) instead of being extrapolated.
    pub fn start_increment(&mut self, u_committed: &[NodeVec], u_bc: &[NodeVec], constrained: &[u16], t0: f64, dt: f64) -> Result<()> {
        let temps: Vec<f64> = u_committed.iter().map(|v| v[TS]).collect();
        match &self.cur {
            Some((tc, _)) if (*tc - t0).abs() <= 1e-12 * t0.abs().max(1.0) => {}
            _ => {
                self.prev = self.cur.take();
                self.cur = Some((t0, temps.clone()));
            }
        }
        let mut pred: Vec<NodeVec> = u_committed.to_vec();
        if let (Some((tp, tprev)), true) = (&self.prev, dt > 0.0) {
            let dtp = t0 - tp;
            if dtp > 0.0 {
                for (n, p) in pred.iter_mut().enumerate() {
                    p[TS] = temps[n] + (temps[n] - tprev[n]) * dt / dtp;
                }
            }
        }
        for (n, p) in pred.iter_mut().enumerate() {
            if constrained.get(n).is_some_and(|c| c & (1 << TS) != 0) {
                p[TS] = u_bc[n][TS];
            }
        }
        self.begin_increment(&pred)
    }

    /// Emissivities at the temperatures `u_start` and the matrices R̄, m⁰
    /// (recomputed only when the emissivities have changed).
    pub fn begin_increment(&mut self, u_start: &[NodeVec]) -> Result<()> {
        let sigma = self.sigma;
        let z = self.abs_zero;
        for c in self.cavities.iter_mut() {
            let eps: Vec<f64> = c.facets.iter().map(|f| self.props[f.prop].eval(Self::facet_temp(c, f, u_start, z)).0[0]).collect();
            if !c.eps.is_empty() && c.eps == eps {
                continue;
            }
            let nf = c.facets.len();
            // serial form: J = C⁻¹(ε E + (1 − ε) f_amb E_amb),
            // C = I − diag(1 − ε) F, the irradiation H = F J + f_amb E_amb =
            // Σⱼ hᵢⱼ Eⱼ + hᵢₐ E_amb and qᵢ = εᵢ (Σⱼ hᵢⱼ (Eⱼ − Eᵢ) + hᵢₐ (E_amb − Eᵢ)),
            // i.e. an isothermal enclosure gives no flux even when Σⱼ Fᵢⱼ ≠ 1
            // (verified for a closed 3D cavity)
            let m = nf + 1;
            let mut h = vec![0.0; nf * m];
            if c.opts.reflection {
                let mut a = vec![0.0; nf * nf];
                let mut rhs = vec![0.0; nf * m];
                for i in 0..nf {
                    for j in 0..nf {
                        a[i * nf + j] = if i == j { 1.0 } else { 0.0 } - (1.0 - eps[i]) * c.f[i * nf + j];
                    }
                    rhs[i * m + i] = eps[i];
                    rhs[i * m + nf] = (1.0 - eps[i]) * c.famb[i];
                }
                let y = solve_dense(a, nf, rhs, m)?;
                for i in 0..nf {
                    for k in 0..nf {
                        let fik = c.f[i * nf + k];
                        if fik == 0.0 {
                            continue;
                        }
                        for col in 0..m {
                            h[i * m + col] += fik * y[k * m + col];
                        }
                    }
                    h[i * m + nf] += c.famb[i];
                }
            } else {
                for i in 0..nf {
                    h[i * m..i * m + nf].copy_from_slice(&c.f[i * nf..(i + 1) * nf]);
                    h[i * m + nf] = c.famb[i];
                }
            }
            // x: flux per area qᵢ = Σⱼ xᵢⱼ ηⱼ + xᵢₐ (black body without ε)
            let mut x = vec![0.0; nf * m];
            let eamb = c.ambient.map_or(0.0, |ta| (ta - z).powi(4));
            for i in 0..nf {
                let e = if c.opts.reflection { eps[i] } else { 1.0 };
                let w: f64 = h[i * m..(i + 1) * m].iter().sum();
                for j in 0..nf {
                    x[i * m + j] = sigma * e * h[i * m + j];
                }
                x[i * m + i] -= sigma * e * w;
                x[i * m + nf] = sigma * e * h[i * m + nf] * eamb;
            }
            // Mᵢⱼ = Aᵢ xᵢⱼ; R̄ = Pᵀ M P
            let nn = c.nodes.len();
            let mut rbar = vec![0.0; nn * nn];
            let mut m0 = vec![0.0; nn];
            let mut mp = vec![0.0; nf * nn];
            for i in 0..nf {
                for j in 0..nf {
                    let v = c.facets[i].area * x[i * m + j];
                    if v == 0.0 {
                        continue;
                    }
                    for &(k, w) in &c.facets[j].p {
                        mp[i * nn + k] += v * w;
                    }
                }
            }
            for i in 0..nf {
                let q0 = c.facets[i].area * x[i * m + nf];
                for &(k, w) in &c.facets[i].p {
                    m0[k] += w * q0;
                    for l in 0..nn {
                        rbar[k * nn + l] += w * mp[i * nn + l];
                    }
                }
            }
            c.rbar = rbar;
            c.m0 = m0;
            c.eps = eps;
        }
        Ok(())
    }

    /// Requests a halved increment (PNEWDT = 0.5) if the emissivity at
    /// the temperatures `u` differs by more than MXDEM from the one used.
    pub fn check_mxdem(&self, u: &[NodeVec]) {
        let Some(mx) = self.mxdem else { return };
        for c in &self.cavities {
            for (f, e0) in c.facets.iter().zip(&c.eps) {
                if !self.props[f.prop].varies() {
                    continue;
                }
                let e1 = self.props[f.prop].eval(Self::facet_temp(c, f, u, self.abs_zero)).0[0];
                if (e1 - e0).abs() > mx {
                    crate::user::context::request_pnewdt(0.5);
                    return;
                }
            }
        }
    }

    /// Adds −Q_N to `fint` and the tangent −4 R̄_NM (θ_M − θ_Z)³.
    pub fn contribute(&self, u: &[NodeVec], dm: &DofMap, fint: &mut [NodeVec], mat: Option<&mut dyn SystemMatrix>) {
        let z = self.abs_zero;
        let mut mat = mat;
        for c in &self.cavities {
            let nn = c.nodes.len();
            let eta: Vec<f64> = c.nodes.iter().map(|&n| (u[n][TS] - z).powi(4)).collect();
            let deta: Vec<f64> = c.nodes.iter().map(|&n| 4.0 * (u[n][TS] - z).powi(3)).collect();
            for a in 0..nn {
                let mut q = c.m0[a];
                for b in 0..nn {
                    q += c.rbar[a * nn + b] * eta[b];
                }
                fint[c.nodes[a]][TS] -= q;
            }
            if let Some(m) = mat.as_deref_mut() {
                for a in 0..nn {
                    let r = dm.eq[c.nodes[a]][TS];
                    if r == NONE {
                        continue;
                    }
                    for b in 0..nn {
                        let s = dm.eq[c.nodes[b]][TS];
                        let v = c.rbar[a * nn + b] * deta[b];
                        if s != NONE && v != 0.0 {
                            m.add_unsym(r, s, -v);
                        }
                    }
                }
            }
        }
    }
}
