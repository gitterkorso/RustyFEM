//! XFEM: `*ENRICHMENT` with the level set functions
//! PHILSM (φ, distance to the crack surface) and PSILSM (ψ, distance to the front)
//! from `*INITIAL CONDITIONS, TYPE=ENRICHMENT`.
//!
//! `setup` (called when the .inp file has been read) replaces the elements in the enrichment's
//! ELSET with XFEM elements (one leaked `XfemElem` per element, which delegates to
//! the base element when the element is not enriched) and adds the extra
//! nodes for the enrichment:
//!
//! - Heaviside (crack through the element) with phantom nodes: one phantom node per
//!   enriched node (number −(10⁹ + 8·node)). In a cut element, points
//!   on the node's side use the node's displacement and points on the other side the phantom node,
//!   u = Σ Nᵢ ûᵢ(side), so U at the nodes is the physical displacement.
//! - TYPE=STATIONARY CRACK: additionally the four asymptotic tip functions
//!   Fα(r, θ) (shifted, Nᵢ (Fα − Fα(xᵢ)) bᵢα, nodes −(10⁹ + 8·node + α)) at
//!   the nodes of the elements within two element layers around the tip elements
//!   (independent of the element size and of ENRICHMENT RADIUS). Heaviside nodes are the nodes of fully cut elements;
//!   in the tip elements only the tip functions are used (Heaviside there gives a too
//!   soft solution).
//! - TYPE=PROPAGATION CRACK: all nodes in the domain get phantom nodes from the start
//!   (unused phantom nodes are held at the real node with a very weak
//!   spring). The crack grows one element layer per accepted increment
//!   (`Crack::propagate`, called from the interaction's `commit`) when
//!   *DAMAGE INITIATION (MAXPS, MAXPE, MAXS, QUADS, MAXE, QUADE) at the element
//!   centroid ahead of the front reaches 1; the new segment passes through
//!   the front point perpendicular to the maximum principal stress (or to the global
//!   axis NORMAL DIRECTION=). The crack surfaces follow traction–separation
//!   (`materials/cohesive.rs`) with *DAMAGE EVOLUTION and *DAMAGE STABILIZATION,
//!   initial cracks are traction-free, contact (hard, penalty) with INTERACTION=.
//!
//! Integration: cut elements are divided into subsimplices along φ = 0 (tip elements
//! also along ψ = 0) with collapsed Gauss rules; enriched uncut elements
//! use 3×3×3 Gauss points. Heaviside elements without tip functions get
//! the base element's B-bar (C3D8, CPE4) per subelement (side), like two superimposed
//! elements. Stresses are reported only for elements without
//! enrichment (not at the internally generated points).

#[path = "xfem_geom.rs"]
pub mod geom;

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::{DataLine, DistLoad, FeModel, MaterialDef, RawKeyword, SectionDef};
use crate::materials::{MaterialModel, StressMode};
use crate::model::{fail, FemError, Result};
use geom::Ref;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

pub static TYPES: &[&dyn ElementType] = &[];

/// Number offset for the extra nodes of the enrichment.
pub const LABEL_BASE: i64 = 1_000_000_000;
/// Weak spring (relative to the element stiffness) for unused enrichment degrees of freedom.
const TIE: f64 = 1e-8;
/// Header length in the element state.
const HDR: usize = 4;

/// Initiation criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criterion {
    MaxPs(f64),
    MaxPe(f64),
    MaxS([f64; 3]),
    QuadS([f64; 3]),
    MaxE([f64; 3]),
    QuadE([f64; 3]),
}

/// The crack material: initiation and cohesive law.
pub struct Law {
    pub crit: Criterion,
    /// NORMAL DIRECTION= (0-based global axis), otherwise the direction of the maximum principal stress.
    pub normal_dir: Option<usize>,
    /// Cohesive law (TractionSeparation) or None (no *DAMAGE EVOLUTION: elastic).
    pub ts: Arc<dyn MaterialModel>,
    pub visc: f64,
}

/// A cracked element: nodal values of φ and whether the segment is initial (traction-free).
#[derive(Debug, Clone)]
pub struct Seg {
    pub phi: Vec<f64>,
    pub initial: bool,
}

/// A crack front (propagation): point on the element face, growth direction
/// and the element ahead.
#[derive(Debug, Clone)]
pub struct Front {
    pub p: [f64; 3],
    pub dir: [f64; 3],
    pub ahead: usize,
}

/// A crack (*ENRICHMENT).
pub struct Crack {
    pub name: String,
    pub stationary: bool,
    pub dim: usize,
    pub mode: StressMode,
    pub domain: Vec<usize>,
    pub e_mod: f64,
    pub nu: f64,
    /// The node numbers (model indices) and coordinates of the domain's elements.
    pub conn: HashMap<usize, (Ref, Vec<usize>)>,
    pub coords: HashMap<usize, [f64; 3]>,
    /// Face neighbors within the domain: element → (local face, neighbor or None).
    pub faces: HashMap<usize, Vec<(Vec<usize>, Option<usize>)>>,
    pub law: Option<Law>,
    pub contact: bool,
    pub penalty: f64,
    /// Nodal values for the output PHILSM/PSILSM (initial).
    pub nodal: BTreeMap<usize, (f64, f64)>,
    /// Initial segments (propagation).
    pub initial: HashMap<usize, Seg>,
    /// Accepted segments (propagation).
    pub segs: RwLock<HashMap<usize, Seg>>,
    /// Initial and current crack fronts (propagation).
    pub init_fronts: RwLock<Vec<Front>>,
    pub fronts: RwLock<Vec<Front>>,
    /// The latest centroid stress per uncut element (for the initiation).
    pub stash: Mutex<HashMap<usize, Vec<f64>>>,
    /// Crack growth on/off per step (*ENRICHMENT ACTIVATION) and the end times of the steps.
    pub activation: Vec<(f64, bool)>,
    pub time: Mutex<f64>,
}

/// An extra node in the element after the base nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Pseudo {
    Phantom(usize),
    Tip(usize, usize),
}

/// Fixed geometry for stationary cracks.
#[derive(Debug, Clone)]
struct StatGeo {
    phi: Vec<f64>,
    psi: Vec<f64>,
    cut: bool,
    tip: bool,
    hnode: Vec<bool>,
    tipf: Vec<Option<[f64; 4]>>,
}

/// The element geometry in one call.
struct Geo {
    phi: Vec<f64>,
    psi: Option<Vec<f64>>,
    side: Vec<i8>,
    hact: Vec<bool>,
    cut: bool,
    tipf: Vec<Option<[f64; 4]>>,
    initial: bool,
    order: usize,
}

pub struct XfemElem {
    pub base: &'static dyn ElementType,
    pub r: Ref,
    pub crack: &'static Crack,
    pub elem: usize,
    pub nb: usize,
    pub pseudo: Vec<Pseudo>,
    stat: Option<StatGeo>,
    nodes_total: usize,
}

fn registry() -> &'static Mutex<HashMap<usize, &'static XfemElem>> {
    static R: OnceLock<Mutex<HashMap<usize, &'static XfemElem>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn key(t: &dyn ElementType) -> usize {
    t as *const dyn ElementType as *const () as usize
}

/// The XFEM element behind an element type (if the type is an XFEM element).
pub fn as_xfem(t: &'static dyn ElementType) -> Option<&'static XfemElem> {
    registry().lock().unwrap().get(&key(t)).copied()
}

/// The crack named `name` in the model.
pub fn crack_of(model: &FeModel, name: &str) -> Option<&'static Crack> {
    let up = name.to_uppercase();
    model.elements.iter().filter_map(|e| as_xfem(e.etype)).map(|x| x.crack).find(|c| c.name == up)
}

/// All cracks in the model.
pub fn cracks(model: &FeModel) -> Vec<&'static Crack> {
    let mut v: Vec<&'static Crack> = Vec::new();
    for e in &model.elements {
        if let Some(x) = as_xfem(e.etype) {
            if !v.iter().any(|c| std::ptr::eq(*c, x.crack)) {
                v.push(x.crack);
            }
        }
    }
    v
}

fn dofs(dim: usize) -> &'static [Dof] {
    if dim == 3 {
        &[U1, U2, U3]
    } else {
        &[U1, U2]
    }
}

fn mode_of(name: &str) -> StressMode {
    if name.starts_with("CPS") {
        StressMode::PlaneStress
    } else if name.starts_with("CPE") {
        StressMode::PlaneStrain
    } else {
        StressMode::ThreeD
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn kw(name: &str, params: &[(&str, &str)], data: Vec<Vec<String>>) -> RawKeyword {
    RawKeyword {
        name: name.into(),
        params: params.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        data: data.into_iter().map(|fields| DataLine { fields, line: 0, continued: false }).collect(),
        file: "(xfem)".into(),
        line: 0,
    }
}

fn elastic_of(def: &MaterialDef) -> Result<(f64, f64)> {
    let Some(e) = def.behavior("ELASTIC") else {
        return fail(format!("material {}: XFEM requires *ELASTIC", def.name));
    };
    if e.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") {
        return e.err("XFEM is supported only with isotropic elasticity");
    }
    let v = e.first_floats()?;
    let (em, nu) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
    if !(em > 0.0) {
        return e.err("the elastic modulus must be positive");
    }
    Ok((em, nu))
}

fn law_of(def: &MaterialDef, em: f64, h: f64) -> Result<Option<Law>> {
    let Some(di) = def.behavior("DAMAGE INITIATION") else { return Ok(None) };
    let v = di.first_floats()?;
    let a = v.first().copied().unwrap_or(0.0);
    let s = v.get(1).copied().filter(|&x| x != 0.0).unwrap_or(a);
    let t = v.get(2).copied().filter(|&x| x != 0.0).unwrap_or(s);
    if !(a > 0.0) {
        return di.err("the initiation value must be positive");
    }
    if let Some(p) = di.param_upper("POSITION").filter(|p| p != "CENTROID") {
        return di.err(format!("POSITION={p} is not supported yet for XFEM (only CENTROID)"));
    }
    let crit = match di.param_upper("CRITERION").as_deref() {
        Some("MAXPS") => Criterion::MaxPs(a),
        Some("MAXPE") => Criterion::MaxPe(a),
        Some("MAXS") => Criterion::MaxS([a, s, t]),
        Some("QUADS") => Criterion::QuadS([a, s, t]),
        Some("MAXE") => Criterion::MaxE([a, s, t]),
        Some("QUADE") => Criterion::QuadE([a, s, t]),
        Some(o) => return di.err(format!("CRITERION={o} is not supported for XFEM (MAXPS, MAXPE, MAXS, QUADS, MAXE, QUADE)")),
        None => return di.err("CRITERION= is missing"),
    };
    // MAXPS/MAXPE: normal along the maximum principal direction; the traction criteria:
    // NORMAL DIRECTION= (default 1)
    let traction = matches!(crit, Criterion::MaxS(_) | Criterion::QuadS(_) | Criterion::MaxE(_) | Criterion::QuadE(_));
    let normal_dir = match di.param("NORMAL DIRECTION") {
        None if traction => Some(0),
        None => None,
        Some(s) => match s.trim() {
            "1" => Some(0),
            "2" => Some(1),
            "3" => Some(2),
            o => return di.err(format!("NORMAL DIRECTION={o} must be 1, 2 or 3")),
        },
    };
    // cohesive law: traction limits in the local directions of the crack
    let lim = match crit {
        Criterion::MaxPs(x) => [x, x, x],
        Criterion::MaxPe(x) => [em * x / (1.0 - elastic_of(def)?.1.powi(2)); 3],
        Criterion::MaxS(l) | Criterion::QuadS(l) => l,
        Criterion::MaxE(l) | Criterion::QuadE(l) => [em * l[0], em * l[1], em * l[2]],
    };
    let tsc = if matches!(crit, Criterion::QuadS(_) | Criterion::QuadE(_)) { "QUADS" } else { "MAXS" };
    // penalty stiffness from the base material and the element size
    let k = em / h;
    let mut behaviors = vec![
        kw("ELASTIC", &[("TYPE", "TRACTION")], vec![vec![k.to_string(), k.to_string(), k.to_string()]]),
        kw("DAMAGE INITIATION", &[("CRITERION", tsc)], vec![lim.iter().map(|x| x.to_string()).collect()]),
    ];
    if let Some(de) = def.behavior("DAMAGE EVOLUTION") {
        behaviors.push(de.clone());
    }
    let tdef = MaterialDef { name: format!("{}-XFEM", def.name), behaviors, density: 0.0 };
    let ts = crate::materials::build(&tdef)?;
    if ts.cohesive().is_none() {
        return fail(format!("material {}: could not build the traction–separation of the crack", def.name));
    }
    let visc = match def.behavior("DAMAGE STABILIZATION") {
        Some(k) => k.first_floats()?.first().copied().unwrap_or(0.0),
        None => 0.0,
    };
    Ok(Some(Law { crit, normal_dir, ts, visc }))
}

/// Nodal values of an element's level sets from the initial conditions.
struct Ic {
    phi: Vec<f64>,
    psi: Vec<f64>,
}

fn grad_at_centroid(r: Ref, x: &[[f64; 3]], vals: &[f64]) -> ([f64; 3], [f64; 3], f64) {
    let c = r.centroid();
    let (n, dndx, _) = physical(r, x, c).unwrap_or_else(|_| (vec![0.0; r.nn()], vec![[0.0; 3]; r.nn()], 0.0));
    let mut p = [0.0; 3];
    let mut g = [0.0; 3];
    let mut v = 0.0;
    for i in 0..r.nn() {
        v += n[i] * vals[i];
        for k in 0..3 {
            p[k] += n[i] * x[i][k];
            g[k] += dndx[i][k] * vals[i];
        }
    }
    (p, g, v)
}

/// Shape functions, global derivatives and det(J) at the natural point `p`.
pub fn physical(r: Ref, x: &[[f64; 3]], p: [f64; 3]) -> Result<(Vec<f64>, Vec<[f64; 3]>, f64)> {
    let (n, dn) = r.shape(p);
    let nn = r.nn();
    if r.dim() == 3 {
        let mut j = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                j[a][b] = (0..nn).map(|i| dn[i][a] * x[i][b]).sum();
            }
        }
        let (ji, det) = common::inv3(j);
        if det <= 0.0 {
            return fail(common::DISTORTED);
        }
        let d = (0..nn).map(|i| {
            let mut g = [0.0; 3];
            for c in 0..3 {
                g[c] = (0..3).map(|a| ji[c][a] * dn[i][a]).sum();
            }
            g
        });
        Ok((n, d.collect(), det))
    } else {
        let mut j = [[0.0; 2]; 2];
        for a in 0..2 {
            for b in 0..2 {
                j[a][b] = (0..nn).map(|i| dn[i][a] * x[i][b]).sum();
            }
        }
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det <= 0.0 {
            return fail(common::DISTORTED);
        }
        let ji = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
        let d = (0..nn).map(|i| [ji[0][0] * dn[i][0] + ji[0][1] * dn[i][1], ji[1][0] * dn[i][0] + ji[1][1] * dn[i][1], 0.0]);
        Ok((n, d.collect(), det))
    }
}

fn sign_change(v: &[f64]) -> bool {
    v.iter().any(|&x| x < 0.0) && v.iter().any(|&x| x >= 0.0) && v.iter().any(|&x| x > 0.0 || x < 0.0)
}

fn snap(v: &[f64], h: f64) -> Vec<f64> {
    // values very close to zero are moved to +ε (the node then lies on the positive side)
    v.iter().map(|&x| if x.abs() < 1e-9 * h { 1e-9 * h } else { x }).collect()
}

/// Replaces the elements in the *ENRICHMENT domains with XFEM elements.
pub fn setup(model: &mut FeModel) -> Result<()> {
    let enr: Vec<RawKeyword> = model.extra.iter().filter(|k| k.name == "ENRICHMENT").cloned().collect();
    let ics: Vec<RawKeyword> = model.extra.iter().filter(|k| k.name == "XFEM INITIAL").cloned().collect();
    if enr.is_empty() {
        if let Some(k) = ics.first() {
            return fail(format!("{}: *INITIAL CONDITIONS, TYPE=ENRICHMENT without *ENRICHMENT", k.location()));
        }
        return Ok(());
    }
    for k in &enr {
        setup_crack(model, k, &ics)?;
    }
    Ok(())
}

fn setup_crack(model: &mut FeModel, kw0: &RawKeyword, ics: &[RawKeyword]) -> Result<()> {
    let name = kw0.param_upper("NAME").ok_or_else(|| FemError(format!("{}: NAME= is missing", kw0.location())))?;
    let ty = kw0.param_upper("TYPE").unwrap_or_else(|| "PROPAGATION CRACK".into());
    let stationary = match ty.as_str() {
        "PROPAGATION CRACK" => false,
        "STATIONARY CRACK" | "STATIONARY" => true,
        o => return kw0.err(format!("TYPE={o} is not supported (PROPAGATION CRACK, STATIONARY CRACK)")),
    };
    let elset = kw0.param_upper("ELSET").ok_or_else(|| FemError(format!("{}: ELSET= is missing", kw0.location())))?;
    let domain = match model.elsets.get(&elset) {
        Some(s) => s.clone(),
        None => match model.elsets.iter().find(|(k, _)| k.ends_with(&format!(".{elset}"))) {
            Some((_, s)) => s.clone(),
            None => return kw0.err(format!("the element set '{elset}' does not exist")),
        },
    };
    if domain.is_empty() {
        return kw0.err("the domain is empty");
    }
    // base element, material
    let mut refs = HashMap::new();
    let mut dim = 0;
    let mut mname = String::new();
    let mut ename = "";
    for &e in &domain {
        let el = &model.elements[e];
        if as_xfem(el.etype).is_some() {
            return kw0.err(format!("element {} already belongs to another enrichment", el.label));
        }
        let n = el.etype.name();
        let Some(r) = Ref::of(n) else {
            return kw0.err(format!("the element type {n} cannot be enriched (XFEM is supported for C3D8, C3D8R, C3D4, CPE4, CPS4, CPE4R, CPS4R, CPE3, CPS3)"));
        };
        if stationary && r.dim() == 2 {
            return kw0.err(format!(
                "TYPE=STATIONARY CRACK allows only first-order three-dimensional elements, not {n}"
            ));
        }
        if dim != 0 && dim != r.dim() {
            return kw0.err("the domain mixes plane and three-dimensional elements");
        }
        dim = r.dim();
        if ename.is_empty() {
            ename = n;
        } else if mode_of(ename) != mode_of(n) {
            return kw0.err("the domain mixes plane stress and plane strain");
        }
        let Some(si) = el.section else { return kw0.err(format!("element {} has no section", el.label)) };
        let mn = model.sections[si].material.to_uppercase();
        if mname.is_empty() {
            mname = mn;
        } else if mname != mn {
            return kw0.err("all elements in the domain must have the same material");
        }
        refs.insert(e, r);
    }
    let mode = mode_of(ename);
    let mi = model.material_index(&mname).ok_or_else(|| FemError(format!("the material {mname} does not exist")))?;
    let def = model.materials[mi].clone();
    let (em, nu) = elastic_of(&def)?;
    // initial conditions for this crack
    let mut ic: BTreeMap<usize, Ic> = BTreeMap::new();
    for k in ics {
        for d in &k.data {
            if d.fields.get(2).map(|s| s.as_str()) != Some(name.as_str()) {
                continue;
            }
            let e: usize = d.fields[0].parse().unwrap_or(usize::MAX);
            let ln: usize = d.fields[1].parse().unwrap_or(0);
            let phi: f64 = d.fields[3].parse().unwrap_or(0.0);
            let psi: f64 = d.fields[4].parse().unwrap_or(0.0);
            let Some(&r) = refs.get(&e) else {
                return fail(format!("*INITIAL CONDITIONS, TYPE=ENRICHMENT: element {} does not belong to the domain of {name}", model.elements.get(e).map_or(0, |x| x.label)));
            };
            if ln == 0 || ln > r.nn() {
                return fail(format!("*INITIAL CONDITIONS, TYPE=ENRICHMENT: invalid local node number {ln} for element {}", model.elements[e].label));
            }
            let en = ic.entry(e).or_insert_with(|| Ic { phi: vec![f64::NAN; r.nn()], psi: vec![f64::NAN; r.nn()] });
            en.phi[ln - 1] = phi;
            en.psi[ln - 1] = psi;
        }
    }
    for (&e, v) in &ic {
        if v.phi.iter().any(|x| x.is_nan()) {
            return fail(format!("*INITIAL CONDITIONS, TYPE=ENRICHMENT: all nodes in element {} must have values", model.elements[e].label));
        }
    }
    // geometry
    let mut conn = HashMap::new();
    let mut coords = HashMap::new();
    let mut hsum = 0.0;
    for &e in &domain {
        let r = refs[&e];
        let nodes: Vec<usize> = model.elements[e].nodes[..r.nn()].to_vec();
        let x: Vec<[f64; 3]> = nodes.iter().map(|&n| model.nodes[n]).collect();
        for (&n, &p) in nodes.iter().zip(&x) {
            coords.insert(n, p);
        }
        let w: f64 = r.rule(2).iter().map(|&(p, w)| physical(r, &x, p).map(|v| v.2 * w).unwrap_or(0.0)).sum();
        hsum += if dim == 3 { w.abs().cbrt() } else { w.abs().sqrt() };
        conn.insert(e, (r, nodes));
    }
    let h = hsum / domain.len() as f64;
    // face neighbors
    let mut face_map: HashMap<Vec<usize>, Vec<(usize, Vec<usize>)>> = HashMap::new();
    for &e in &domain {
        let (r, nodes) = &conn[&e];
        for f in r.faces() {
            let mut k: Vec<usize> = f.iter().map(|&i| nodes[i]).collect();
            k.sort_unstable();
            face_map.entry(k).or_default().push((e, f));
        }
    }
    let mut faces: HashMap<usize, Vec<(Vec<usize>, Option<usize>)>> = HashMap::new();
    for v in face_map.values() {
        for (e, f) in v {
            let other = v.iter().find(|(o, _)| o != e).map(|(o, _)| *o);
            faces.entry(*e).or_default().push((f.clone(), other));
        }
    }
    // elements that share nodes with the domain but do not belong to it (neighbors outside) count as uncracked
    let law = if stationary { None } else { law_of(&def, em, h)? };
    if !stationary && law.is_none() {
        return kw0.err("TYPE=PROPAGATION CRACK requires *DAMAGE INITIATION in the domain material");
    }
    let contact = kw0.param("INTERACTION").is_some();
    if let Some(i) = kw0.param_upper("INTERACTION") {
        if !model.extra.iter().any(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(i.as_str())) {
            return kw0.err(format!("*SURFACE INTERACTION {i} does not exist"));
        }
    }
    let mut nodal = BTreeMap::new();
    for (&e, v) in &ic {
        for (i, &n) in conn[&e].1.iter().enumerate() {
            nodal.entry(n).or_insert((v.phi[i], v.psi[i]));
        }
    }
    // the end times of the steps and *ENRICHMENT ACTIVATION
    let mut activation = Vec::new();
    let mut t = 0.0;
    let mut on = true;
    for s in &model.steps {
        let period = s.procedure.as_ref().and_then(|p| p.data.first()).and_then(|d| d.fields.get(1)).and_then(|f| crate::fem::model::parse_f64(f)).filter(|&p| p > 0.0).unwrap_or(1.0);
        t += period;
        for k in s.extra.iter().filter(|k| k.name == "ENRICHMENT ACTIVATION") {
            if k.param_upper("NAME").as_deref() == Some(name.as_str()) {
                on = k.param_upper("ACTIVATE").map_or(true, |v| v != "OFF");
            }
        }
        activation.push((t, on));
    }
    let mut initial = HashMap::new();
    if !stationary {
        for (&e, v) in &ic {
            let ph = snap(&v.phi, h);
            if sign_change(&ph) && v.psi.iter().any(|&x| x < 0.0) {
                initial.insert(e, Seg { phi: ph, initial: true });
            }
        }
    }
    let crack: &'static Crack = Box::leak(Box::new(Crack {
        name: name.clone(),
        stationary,
        dim,
        mode,
        domain: domain.clone(),
        e_mod: em,
        nu,
        conn,
        coords,
        faces,
        law,
        contact,
        penalty: 10.0 * em / h,
        nodal,
        initial: initial.clone(),
        segs: RwLock::new(initial),
        init_fronts: RwLock::new(Vec::new()),
        fronts: RwLock::new(Vec::new()),
        stash: Mutex::new(HashMap::new()),
        activation,
        time: Mutex::new(0.0),
    }));
    if stationary {
        setup_stationary(model, crack, &ic, h)
    } else {
        let mut fr = Vec::new();
        for (&e, v) in &ic {
            let Some(seg) = crack.initial.get(&e) else { continue };
            let (r, nodes) = &crack.conn[&e];
            let x: Vec<[f64; 3]> = nodes.iter().map(|n| crack.coords[n]).collect();
            let (c, gp, _) = grad_at_centroid(*r, &x, &seg.phi);
            let (_, gs, _) = grad_at_centroid(*r, &x, &v.psi);
            let nn = common::scale(gp, 1.0 / common::norm(gp).max(1e-300));
            let d = common::sub(gs, common::scale(nn, common::dot(gs, nn)));
            let d = common::scale(d, 1.0 / common::norm(d).max(1e-300));
            if let Some(f) = crack.exit_front(e, &seg.phi, c, d) {
                if !crack.initial.contains_key(&f.ahead) {
                    fr.push(f);
                }
            }
        }
        *crack.init_fronts.write().unwrap() = fr.clone();
        *crack.fronts.write().unwrap() = fr;
        setup_propagation(model, crack)
    }
}

fn add_pseudo(model: &mut FeModel, n: usize, k: i64) -> usize {
    let inst = model.node_inst[n];
    let l = model.node_label[n];
    let x = model.nodes[n];
    model.add_node(inst, -(LABEL_BASE + 8 * l + k), x)
}

fn install(model: &mut FeModel, crack: &'static Crack, e: usize, pseudo: Vec<Pseudo>, stat: Option<StatGeo>) {
    let (r, nodes) = &crack.conn[&e];
    let base = model.elements[e].etype;
    let mut extra = Vec::new();
    for p in &pseudo {
        let id = match *p {
            Pseudo::Phantom(i) => add_pseudo(model, nodes[i], 0),
            Pseudo::Tip(i, a) => add_pseudo(model, nodes[i], a as i64 + 1),
        };
        extra.push(id);
    }
    let x: &'static XfemElem = Box::leak(Box::new(XfemElem { base, r: *r, crack, elem: e, nb: r.nn(), nodes_total: r.nn() + pseudo.len(), pseudo, stat }));
    registry().lock().unwrap().insert(key(x), x);
    let el = &mut model.elements[e];
    el.nodes.truncate(r.nn());
    el.nodes.extend(extra);
    el.etype = x;
}

fn setup_propagation(model: &mut FeModel, crack: &'static Crack) -> Result<()> {
    for &e in &crack.domain {
        let nb = crack.conn[&e].0.nn();
        install(model, crack, e, (0..nb).map(Pseudo::Phantom).collect(), None);
    }
    Ok(())
}

fn setup_stationary(model: &mut FeModel, crack: &'static Crack, ic: &BTreeMap<usize, Ic>, h: f64) -> Result<()> {
    // classification
    let mut cut = BTreeSet::new();
    let mut full = BTreeSet::new();
    let mut tips = BTreeSet::new();
    for (&e, v) in ic {
        let ph = snap(&v.phi, h);
        if !(sign_change(&ph) && v.psi.iter().any(|&x| x < 0.0)) {
            continue;
        }
        // cut only if the crack surface (φ = 0, ψ < 0) actually passes through the element
        if geom::partition(crack.conn[&e].0, &ph, Some(&v.psi)).facets.is_empty() {
            continue;
        }
        cut.insert(e);
        if v.psi.iter().all(|&x| x <= 0.0) {
            full.insert(e);
        } else {
            tips.insert(e);
        }
    }
    if cut.is_empty() {
        return fail(format!("enrichment {}: no element is cut by the crack (check *INITIAL CONDITIONS, TYPE=ENRICHMENT)", crack.name));
    }
    let hnodes: BTreeSet<usize> = full.iter().flat_map(|e| crack.conn[e].1.iter().copied()).collect();
    // tip enrichment: the nodes of the elements within two layers around the tip elements
    let mut node_elems: HashMap<usize, Vec<usize>> = HashMap::new();
    for &e in &crack.domain {
        for &n in &crack.conn[&e].1 {
            node_elems.entry(n).or_default().push(e);
        }
    }
    let mut ring: BTreeSet<usize> = tips.clone();
    for _ in 0..2 {
        let nodes: BTreeSet<usize> = ring.iter().flat_map(|e| crack.conn[e].1.iter().copied()).collect();
        for n in nodes {
            ring.extend(node_elems[&n].iter().copied());
        }
    }
    let tnodes: BTreeSet<usize> = ring.iter().flat_map(|e| crack.conn[e].1.iter().copied()).collect();
    // nodal values: the initial conditions, otherwise linear extrapolation from the nearest cut element
    let fits: Vec<([f64; 3], [f64; 3], f64, [f64; 3], f64)> = cut
        .iter()
        .map(|e| {
            let (r, nodes) = &crack.conn[e];
            let x: Vec<[f64; 3]> = nodes.iter().map(|n| crack.coords[n]).collect();
            let (c, gp, vp) = grad_at_centroid(*r, &x, &ic[e].phi);
            let (_, gs, vs) = grad_at_centroid(*r, &x, &ic[e].psi);
            (c, gp, vp, gs, vs)
        })
        .collect();
    let level = |n: usize| -> (f64, f64) {
        if let Some(&v) = crack.nodal.get(&n) {
            return v;
        }
        let x = crack.coords[&n];
        let mut best = (f64::MAX, (0.0, 0.0));
        for (c, gp, vp, gs, vs) in &fits {
            let d = common::norm(common::sub(x, *c));
            if d < best.0 {
                let dx = common::sub(x, *c);
                best = (d, (vp + common::dot(*gp, dx), vs + common::dot(*gs, dx)));
            }
        }
        best.1
    };
    for &e in &crack.domain {
        let (r, nodes) = &crack.conn[&e];
        let (phi, psi) = match ic.get(&e) {
            Some(v) => (snap(&v.phi, h), v.psi.clone()),
            None => {
                let v: Vec<(f64, f64)> = nodes.iter().map(|&n| level(n)).collect();
                (snap(&v.iter().map(|p| p.0).collect::<Vec<_>>(), h), v.iter().map(|p| p.1).collect())
            }
        };
        let is_cut = cut.contains(&e);
        let mut pseudo = Vec::new();
        let mut hnode = vec![false; r.nn()];
        let mut tipf = vec![None; r.nn()];
        if is_cut {
            for (i, n) in nodes.iter().enumerate() {
                if hnodes.contains(n) {
                    hnode[i] = true;
                    pseudo.push(Pseudo::Phantom(i));
                }
            }
        }
        for (i, n) in nodes.iter().enumerate() {
            if tnodes.contains(n) {
                let (p, s) = level(*n);
                let p = if p.abs() < 1e-9 * h { 1e-9 * h } else { p };
                tipf[i] = Some(geom::tip_values(p, s));
                for a in 0..4 {
                    pseudo.push(Pseudo::Tip(i, a));
                }
            }
        }
        let stat = StatGeo { phi, psi, cut: is_cut, tip: tips.contains(&e), hnode, tipf };
        install(model, crack, e, pseudo, Some(stat));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Crack growth
// ---------------------------------------------------------------------------

fn sym3(mode: StressMode, s: &[f64]) -> [[f64; 3]; 3] {
    match mode {
        StressMode::ThreeD => [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]],
        StressMode::PlaneStress => [[s[0], s[2], 0.0], [s[2], s[1], 0.0], [0.0, 0.0, 0.0]],
        _ => [[s[0], s[3], 0.0], [s[3], s[1], 0.0], [0.0, 0.0, s[2]]],
    }
}

impl Crack {
    /// Crack growth active at the time t.
    fn active_at(&self, t: f64) -> bool {
        for &(end, on) in &self.activation {
            if t <= end + 1e-12 {
                return on;
            }
        }
        self.activation.last().is_none_or(|a| a.1)
    }

    /// Resets the crack to the initial state (new analysis).
    pub fn reset(&self) {
        *self.segs.write().unwrap() = self.initial.clone();
        *self.fronts.write().unwrap() = self.init_fronts.read().unwrap().clone();
        self.stash.lock().unwrap().clear();
        *self.time.lock().unwrap() = 0.0;
    }

    pub fn set_time(&self, t: f64) {
        *self.time.lock().unwrap() = t;
    }

    pub fn criterion_value(&self, law: &Law, s: &[f64]) -> f64 {
        self.criterion(law, s).0
    }

    /// The criterion value and the crack plane normal for the stress `s`.
    fn criterion(&self, law: &Law, s: &[f64]) -> (f64, [f64; 3]) {
        let m = sym3(self.mode, s);
        let (vals, vecs) = crate::kinematics::tensor::eig_sym(&m);
        // in the plane: only in-plane principal directions
        let mut best = (f64::MIN, [1.0, 0.0, 0.0]);
        for k in 0..3 {
            let v = [vecs[0][k], vecs[1][k], vecs[2][k]];
            if self.dim == 2 && v[2].abs() > 0.5 {
                continue;
            }
            if vals[k] > best.0 {
                best = (vals[k], v);
            }
        }
        let n = match law.normal_dir {
            Some(d) => {
                let mut v = [0.0; 3];
                v[d] = 1.0;
                v
            }
            None => best.1,
        };
        let tr = [m[0][0] * n[0] + m[0][1] * n[1] + m[0][2] * n[2], m[1][0] * n[0] + m[1][1] * n[1] + m[1][2] * n[2], m[2][0] * n[0] + m[2][1] * n[1] + m[2][2] * n[2]];
        let tn = common::dot(tr, n);
        let ts = common::norm(common::sub(tr, common::scale(n, tn)));
        let em = self.e_mod;
        let f = match law.crit {
            Criterion::MaxPs(s0) => best.0.max(0.0) / s0,
            Criterion::MaxPe(e0) => {
                // principal strain from the stress (isotropic elasticity)
                let nu = self.nu;
                let tr3 = vals[0] + vals[1] + vals[2];
                let mut emax = (best.0 * (1.0 + nu) - nu * tr3) / em;
                if self.mode == StressMode::PlaneStress {
                    emax = (best.0 - nu * (tr3 - best.0)) / em;
                }
                emax.max(0.0) / e0
            }
            Criterion::MaxS(l) => (tn.max(0.0) / l[0]).max(ts / l[1]),
            Criterion::QuadS(l) => (tn.max(0.0) / l[0]).powi(2) + (ts / l[1]).powi(2),
            Criterion::MaxE(l) => (tn.max(0.0) / (em * l[0])).max(ts / (em * l[1])),
            Criterion::QuadE(l) => (tn.max(0.0) / (em * l[0])).powi(2) + (ts / (em * l[1])).powi(2),
        };
        (f, n)
    }

    /// Front faces of a cracked element: faces that are cut by the crack and
    /// border an uncracked element in the domain (or outside it).
    fn front_nodes(&self, e: usize, phi: &[f64], segs: &HashMap<usize, Seg>) -> BTreeSet<usize> {
        let mut out = BTreeSet::new();
        let Some(fs) = self.faces.get(&e) else { return out };
        for (f, nb) in fs {
            let vals: Vec<f64> = f.iter().map(|&i| phi[i]).collect();
            if !sign_change(&vals) {
                continue;
            }
            let open = match nb {
                Some(o) => !segs.contains_key(o),
                None => false,
            };
            if open {
                out.extend(f.iter().copied());
            }
        }
        out
    }

    /// The intersection point (the mean of the edge intersections) between φ = 0 and a face.
    fn face_cut(&self, e: usize, phi: &[f64], f: &[usize]) -> Option<[f64; 3]> {
        let nodes = &self.conn[&e].1;
        let mut pts = Vec::new();
        for k in 0..f.len() {
            let (i, j) = (f[k], f[(k + 1) % f.len()]);
            let (pi, pj) = (phi[i], phi[j]);
            if (pi < 0.0) != (pj < 0.0) {
                let tt = pi / (pi - pj);
                let (xi, xj) = (self.coords[&nodes[i]], self.coords[&nodes[j]]);
                pts.push([xi[0] + tt * (xj[0] - xi[0]), xi[1] + tt * (xj[1] - xi[1]), xi[2] + tt * (xj[2] - xi[2])]);
            }
        }
        if pts.is_empty() {
            return None;
        }
        Some(geom::bary_point(&pts, &vec![1.0 / pts.len() as f64; pts.len()]))
    }

    /// The front where the crack (φ in element e, from the point p in the direction d)
    /// leaves the element: the face cut furthest ahead. None if it reaches a free surface.
    fn exit_front(&self, e: usize, phi: &[f64], p: [f64; 3], d: [f64; 3]) -> Option<Front> {
        let fs = self.faces.get(&e)?;
        let mut best: Option<(f64, [f64; 3], Option<usize>)> = None;
        for (f, nb) in fs {
            let Some(c) = self.face_cut(e, phi, f) else { continue };
            let v = common::dot(common::sub(c, p), d);
            if v > 1e-9 && best.as_ref().is_none_or(|b| v > b.0) {
                best = Some((v, c, *nb));
            }
        }
        let (_, c, nb) = best?;
        Some(Front { p: c, dir: d, ahead: nb? })
    }

    /// Crack growth after an accepted increment.
    pub fn propagate(&self) {
        let Some(law) = &self.law else { return };
        let t = *self.time.lock().unwrap();
        if !self.active_at(t) {
            return;
        }
        let stash = self.stash.lock().unwrap().clone();
        let mut segs = self.segs.write().unwrap();
        let mut fronts = self.fronts.write().unwrap();
        let plane = |n: [f64; 3], p: [f64; 3], e: usize| -> Vec<f64> {
            let phi: Vec<f64> = self.conn[&e].1.iter().map(|k| common::dot(n, common::sub(self.coords[k], p))).collect();
            let hb = phi.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            snap(&phi, hb.max(1e-30))
        };
        let inplane = |v: [f64; 3]| -> [f64; 3] {
            let mut v = v;
            if self.dim == 2 {
                v[2] = 0.0;
            }
            common::scale(v, 1.0 / common::norm(v).max(1e-300))
        };
        if segs.is_empty() {
            // no crack yet: the element with the largest criterion value ≥ 1, fronts in both directions
            let mut best: Option<(f64, usize, [f64; 3])> = None;
            for &e in &self.domain {
                if let Some(s) = stash.get(&e) {
                    let (f, n) = self.criterion(law, s);
                    if f >= 1.0 && best.as_ref().is_none_or(|b| f > b.0) {
                        best = Some((f, e, n));
                    }
                }
            }
            if let Some((_, e, n)) = best {
                let (r, nodes) = &self.conn[&e];
                let x: Vec<[f64; 3]> = nodes.iter().map(|k| self.coords[k]).collect();
                let c = geom::bary_point(&x, &vec![1.0 / r.nn() as f64; r.nn()]);
                let phi = plane(n, c, e);
                let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
                let d = if self.dim == 2 { [-n[1], n[0], 0.0] } else { inplane(common::cross(n, common::cross(a, n))) };
                for dd in [d, common::scale(d, -1.0)] {
                    if let Some(f) = self.exit_front(e, &phi, c, dd) {
                        fronts.push(f);
                    }
                }
                segs.insert(e, Seg { phi, initial: false });
            }
            return;
        }
        let mut next = Vec::new();
        for f in fronts.iter() {
            let b = f.ahead;
            if segs.contains_key(&b) {
                continue;
            }
            let Some(s) = stash.get(&b) else {
                next.push(f.clone());
                continue;
            };
            let (fv, n) = self.criterion(law, s);
            if fv < 1.0 {
                next.push(f.clone());
                continue;
            }
            let phi = plane(n, f.p, b);
            if !sign_change(&phi) {
                next.push(f.clone());
                continue;
            }
            let d = inplane(common::sub(f.dir, common::scale(n, common::dot(f.dir, n))));
            if let Some(nf) = self.exit_front(b, &phi, f.p, d) {
                if !segs.contains_key(&nf.ahead) {
                    next.push(nf);
                }
            }
            segs.insert(b, Seg { phi, initial: false });
        }
        *fronts = next;
    }
}

// ---------------------------------------------------------------------------
// The element
// ---------------------------------------------------------------------------

impl XfemElem {
    fn dim(&self) -> usize {
        self.r.dim()
    }

    fn mode(&self) -> StressMode {
        self.crack.mode
    }

    fn base_ctx<'a>(&self, ctx: &ElemCtx<'a>) -> ElemCtx<'a> {
        let mut c = ctx.clone();
        c.x.truncate(self.nb);
        c
    }

    fn thickness(&self, ctx: &ElemCtx) -> f64 {
        if self.dim() == 2 {
            ctx.section.map_or(1.0, |s: &SectionDef| s.solid_thickness())
        } else {
            1.0
        }
    }

    /// The geometry in this call (None = the element is not enriched).
    fn geo(&self) -> Option<Geo> {
        let nb = self.nb;
        if let Some(st) = &self.stat {
            if !st.cut && st.tipf.iter().all(|t| t.is_none()) {
                return None;
            }
            let side: Vec<i8> = st.phi.iter().map(|&p| if p >= 0.0 { 1 } else { -1 }).collect();
            let order = if st.tip { 3 } else if st.tipf.iter().any(|t| t.is_some()) { 3 } else { 2 };
            return Some(Geo {
                phi: st.phi.clone(),
                psi: if st.tip { Some(st.psi.clone()) } else { None },
                side,
                hact: if st.tip { vec![false; nb] } else { st.hnode.clone() },
                cut: st.cut,
                tipf: st.tipf.clone(),
                initial: true,
                order,
            });
        }
        let segs = self.crack.segs.read().unwrap();
        let seg = segs.get(&self.elem)?;
        let front = self.crack.front_nodes(self.elem, &seg.phi, &segs);
        let side: Vec<i8> = seg.phi.iter().map(|&p| if p >= 0.0 { 1 } else { -1 }).collect();
        let hact = (0..nb).map(|i| !front.contains(&i)).collect();
        Some(Geo { phi: seg.phi.clone(), psi: None, side, hact, cut: true, tipf: vec![None; nb], initial: seg.initial, order: 2 })
    }

    /// Integration points (natural point, weight in the reference measure, side).
    fn points(&self, g: &Geo) -> Vec<([f64; 3], f64, i8)> {
        if !g.cut {
            let n = if self.dim() == 3 { 3 } else { 4 };
            return self.r.rule(n).into_iter().map(|(p, w)| (p, w, 0)).collect();
        }
        let part = geom::partition(self.r, &g.phi, g.psi.as_deref());
        let rule = geom::simplex_rule(self.dim(), g.order);
        let mut out = Vec::new();
        for (c, s) in &part.cells {
            let m = geom::measure(&c.v);
            // ahead of the front (ψ > 0) in tip elements there is no crack: no jump
            let s = match &g.psi {
                Some(ps) => {
                    let cc = geom::bary_point(&c.v, &vec![1.0 / c.v.len() as f64; c.v.len()]);
                    if geom::interp(self.r, ps, cc) > 0.0 {
                        0
                    } else {
                        *s
                    }
                }
                _ => *s,
            };
            for (b, w) in &rule {
                out.push((geom::bary_point(&c.v, b), w * m, s));
            }
        }
        out
    }

    /// Generalized integration point (shape functions per element node including enrichment).
    fn ip(&self, ctx: &ElemCtx, g: &Geo, p: [f64; 3], wnat: f64, side: i8, t: f64) -> Result<Ip> {
        let nb = self.nb;
        let (n, dndx, det) = physical(self.r, &ctx.x[..nb], p)?;
        let tot = self.nodes_total;
        let mut nv = vec![0.0; tot];
        let mut gv = vec![[0.0; 3]; tot];
        for i in 0..nb {
            let other = g.cut && side != 0 && g.hact[i] && g.side[i] != side;
            if !other {
                nv[i] = n[i];
                gv[i] = dndx[i];
            }
        }
        let need_tip = self.pseudo.iter().any(|q| matches!(q, Pseudo::Tip(..)));
        let (f, gf) = if need_tip {
            let psi = self.stat.as_ref().map(|s| &s.psi).unwrap();
            let mut ph = 0.0;
            let mut ps = 0.0;
            let mut gp = [0.0; 3];
            let mut gs = [0.0; 3];
            for i in 0..nb {
                ph += n[i] * g.phi[i];
                ps += n[i] * psi[i];
                for k in 0..3 {
                    gp[k] += dndx[i][k] * g.phi[i];
                    gs[k] += dndx[i][k] * psi[i];
                }
            }
            if g.cut && side != 0 && ph * side as f64 <= 0.0 {
                ph = side as f64 * 1e-12 * (1.0 + ps.abs());
            }
            geom::tip_functions(ph, ps, gp, gs)
        } else {
            ([0.0; 4], [[0.0; 3]; 4])
        };
        for (k, q) in self.pseudo.iter().enumerate() {
            let c = nb + k;
            match *q {
                Pseudo::Phantom(i) => {
                    if g.cut && side != 0 && g.hact[i] && g.side[i] != side {
                        nv[c] = n[i];
                        gv[c] = dndx[i];
                    }
                }
                Pseudo::Tip(i, a) => {
                    if let Some(fi) = g.tipf[i] {
                        let d = f[a] - fi[a];
                        nv[c] = n[i] * d;
                        for m in 0..3 {
                            gv[c][m] = dndx[i][m] * d + n[i] * gf[a][m];
                        }
                    }
                }
            }
        }
        Ok(Ip { n: nv, dndx: gv, w: wnat * det * t })
    }

    /// The base element's B-bar (C3D8, CPE4) or none.
    fn bbar(&self) -> common::Bbar {
        match self.base.name() {
            "C3D8" => common::Bbar::Volumetric,
            "CPE4" => common::Bbar::Planar,
            _ => common::Bbar::None,
        }
    }

    /// Integration points and B matrices. Heaviside elements without tip functions
    /// get the base element's B-bar per subelement (side), like two superimposed elements.
    fn ips_b(&self, ctx: &ElemCtx, g: &Geo) -> Result<(Vec<Ip>, Vec<Vec<f64>>)> {
        let mode = self.mode();
        let t = self.thickness(ctx);
        let pts = self.points(g);
        let ips: Vec<Ip> = pts.iter().map(|&(p, w, s)| self.ip(ctx, g, p, w, s, t)).collect::<Result<_>>()?;
        let bb = self.bbar();
        let tip = self.pseudo.iter().any(|q| matches!(q, Pseudo::Tip(..)));
        if bb == common::Bbar::None || tip {
            let bs = common::bmats(mode, &ips, common::Bbar::None);
            return Ok((ips, bs));
        }
        let mut bs = vec![Vec::new(); ips.len()];
        for s in [-1i8, 0, 1] {
            let idx: Vec<usize> = (0..pts.len()).filter(|&k| pts[k].2 == s).collect();
            if idx.is_empty() {
                continue;
            }
            let sub: Vec<Ip> = idx.iter().map(|&k| ips[k].clone()).collect();
            for (k, b) in idx.iter().zip(common::bmats(mode, &sub, bb)) {
                bs[*k] = b;
            }
        }
        Ok((ips, bs))
    }

    /// The jump operator and local axes at a point on the crack surface.
    fn jump(&self, ctx: &ElemCtx, g: &Geo, p: [f64; 3]) -> Result<(Vec<f64>, [[f64; 3]; 3])> {
        let nb = self.nb;
        let (n, dndx, _) = physical(self.r, &ctx.x[..nb], p)?;
        let tot = self.nodes_total;
        let mut c = vec![0.0; tot];
        for i in 0..nb {
            if g.hact[i] {
                let s = g.side[i] as f64;
                c[i] = s * n[i];
            }
        }
        for (k, q) in self.pseudo.iter().enumerate() {
            if let Pseudo::Phantom(i) = *q {
                if g.hact[i] {
                    c[nb + k] = -(g.side[i] as f64) * n[i];
                }
            }
        }
        let mut gp = [0.0; 3];
        for i in 0..nb {
            for k in 0..3 {
                gp[k] += dndx[i][k] * g.phi[i];
            }
        }
        let nn = common::norm(gp).max(1e-300);
        let nrm = common::scale(gp, 1.0 / nn);
        let (s, t) = if self.dim() == 2 {
            ([-nrm[1], nrm[0], 0.0], [0.0, 0.0, 1.0])
        } else {
            let a = if nrm[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
            let s = common::cross(nrm, a);
            let s = common::scale(s, 1.0 / common::norm(s));
            (s, common::cross(nrm, s))
        };
        Ok((c, [nrm, s, t]))
    }

    /// Points on the crack surface: (natural point, physical weight).
    fn facet_points(&self, ctx: &ElemCtx, g: &Geo) -> Vec<([f64; 3], f64)> {
        let part = geom::partition(self.r, &g.phi, None);
        let t = self.thickness(ctx);
        let map = |p: [f64; 3]| -> [f64; 3] {
            let (n, _) = self.r.shape(p);
            geom::bary_point(&ctx.x[..self.nb], &n)
        };
        let mut out = Vec::new();
        for f in &part.facets {
            let xs: Vec<[f64; 3]> = f.iter().map(|&p| map(p)).collect();
            if f.len() == 2 {
                let len = common::norm(common::sub(xs[1], xs[0]));
                for (q, w) in geom::gauss_m11(2) {
                    let s = 0.5 * (q + 1.0);
                    let p = [f[0][0] + s * (f[1][0] - f[0][0]), f[0][1] + s * (f[1][1] - f[0][1]), 0.0];
                    out.push((p, 0.5 * w * len * t));
                }
            } else {
                let area = 0.5 * common::norm(common::cross(common::sub(xs[1], xs[0]), common::sub(xs[2], xs[0])));
                for (b, w) in geom::simplex_rule(2, 2) {
                    out.push((geom::bary_point(f, &b), w * area));
                }
            }
        }
        out
    }

    fn kref(&self, ctx: &ElemCtx) -> f64 {
        let x = &ctx.x[..self.nb];
        let c = geom::bary_point(x, &vec![1.0 / self.nb as f64; self.nb]);
        let h = x.iter().map(|p| common::norm(common::sub(*p, c))).fold(0.0, f64::max);
        self.crack.e_mod * if self.dim() == 2 { self.thickness(ctx) } else { h }
    }

    /// Weak springs for enrichment degrees of freedom that the element does not use.
    fn ties(&self, ctx: &ElemCtx, g: Option<&Geo>, u: &[f64], fint: &mut [f64], k: Option<&mut Vec<f64>>) {
        let d = self.dim();
        let nd = d * self.nodes_total;
        let kt = TIE * self.kref(ctx);
        let mut pairs: Vec<(usize, Option<usize>)> = Vec::new();
        for (q, p) in self.pseudo.iter().enumerate() {
            let used = match (*p, g) {
                (Pseudo::Phantom(i), Some(g)) => g.cut && g.hact[i],
                (Pseudo::Tip(i, _), Some(g)) => g.tipf[i].is_some(),
                _ => false,
            };
            if !used {
                let other = match *p {
                    Pseudo::Phantom(i) => Some(i),
                    Pseudo::Tip(..) => None,
                };
                pairs.push((self.nb + q, other));
            }
        }
        let mut kk = k;
        for (a, b) in pairs {
            for c in 0..d {
                let ia = a * d + c;
                let ib = b.map(|b| b * d + c);
                let diff = u[ia] - ib.map_or(0.0, |i| u[i]);
                fint[ia] += kt * diff;
                if let Some(i) = ib {
                    fint[i] -= kt * diff;
                }
                if let Some(k) = kk.as_deref_mut() {
                    k[ia * nd + ia] += kt;
                    if let Some(i) = ib {
                        k[i * nd + i] += kt;
                        k[ia * nd + i] -= kt;
                        k[i * nd + ia] -= kt;
                    }
                }
            }
        }
    }

    fn pad(&self, v: &[f64]) -> Vec<f64> {
        let mut out = v.to_vec();
        out.resize(self.dim() * self.nodes_total, 0.0);
        out
    }

    fn pad_matrix(&self, k: &[f64], nb: usize) -> Vec<f64> {
        let nd = self.dim() * self.nodes_total;
        let mut out = vec![0.0; nd * nd];
        for i in 0..nb {
            out[i * nd..i * nd + nb].copy_from_slice(&k[i * nb..(i + 1) * nb]);
        }
        out
    }

    /// The cohesive/contact contribution of the crack surface.
    #[allow(clippy::too_many_arguments)]
    fn faces_response(&self, ctx: &ElemCtx, g: &Geo, u: &[f64], so: &[f64], sn: &mut Vec<f64>, fint: &mut [f64], k: Option<&mut Vec<f64>>) -> Result<()> {
        let d = self.dim();
        let nd = d * self.nodes_total;
        let pts = self.facet_points(ctx, g);
        let law = self.crack.law.as_ref();
        let ts = law.and_then(|l| l.ts.cohesive());
        let ns = ts.map_or(0, |t| t.n_state());
        sn.clear();
        let mut kk = k;
        for (q, (p, w)) in pts.iter().enumerate() {
            let (c, rot) = self.jump(ctx, g, *p)?;
            let mut jmp = [0.0; 3];
            for (a, &ca) in c.iter().enumerate() {
                if ca != 0.0 {
                    for m in 0..d {
                        jmp[m] += ca * u[a * d + m];
                    }
                }
            }
            let dl = [common::dot(rot[0], jmp), common::dot(rot[1], jmp), common::dot(rot[2], jmp)];
            let mut tl = [0.0; 3];
            let mut dt = [[0.0; 3]; 3];
            let old: Vec<f64> = if so.len() >= (q + 1) * ns && ns > 0 { so[q * ns..(q + 1) * ns].to_vec() } else { vec![0.0; ns] };
            let mut new = old.clone();
            if !g.initial {
                if let Some(t) = ts {
                    let out = t.update(&dl, d, 1.0, ctx.dt, law.map_or(0.0, |l| l.visc), &old, &mut new);
                    tl = out.t;
                    dt = out.dtde;
                }
            } else if self.crack.contact && dl[0] < 0.0 {
                tl[0] = self.crack.penalty * dl[0];
                dt[0][0] = self.crack.penalty;
            }
            sn.extend_from_slice(&new);
            if tl.iter().all(|&x| x == 0.0) && dt.iter().flatten().all(|&x| x == 0.0) {
                continue;
            }
            // global traction and tangent
            let mut tg = [0.0; 3];
            for a in 0..3 {
                for m in 0..3 {
                    tg[m] += rot[a][m] * tl[a];
                }
            }
            let mut kg = [[0.0; 3]; 3];
            for a in 0..3 {
                for b in 0..3 {
                    if dt[a][b] != 0.0 {
                        for m in 0..3 {
                            for l in 0..3 {
                                kg[m][l] += rot[a][m] * dt[a][b] * rot[b][l];
                            }
                        }
                    }
                }
            }
            for (a, &ca) in c.iter().enumerate() {
                if ca == 0.0 {
                    continue;
                }
                for m in 0..d {
                    fint[a * d + m] += ca * tg[m] * w;
                }
                if let Some(k) = kk.as_deref_mut() {
                    for (b, &cb) in c.iter().enumerate() {
                        if cb == 0.0 {
                            continue;
                        }
                        for m in 0..d {
                            for l in 0..d {
                                k[(a * d + m) * nd + b * d + l] += ca * cb * kg[m][l] * w;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Centroid stress (elastic) for the initiation criterion.
    fn stash_centroid(&self, ctx: &ElemCtx, u: &[f64]) -> Result<()> {
        if self.crack.stationary {
            return Ok(());
        }
        let mode = self.mode();
        let mat = ctx.material(self.base.name())?;
        let d = mat.tangent_linear(mode)?;
        let (_, dndx, _) = physical(self.r, &ctx.x[..self.nb], self.r.centroid())?;
        let b = common::bmat(mode, &dndx);
        let eps = common::strain(&b, &u[..self.dim() * self.nb]);
        let ns = mode.ncomp();
        let s: Vec<f64> = (0..ns).map(|i| (0..ns).map(|j| d[i * ns + j] * eps[j]).sum()).collect();
        self.crack.stash.lock().unwrap().insert(self.elem, s);
        Ok(())
    }

    fn status(&self, g: Option<&Geo>, state: &[f64]) -> f64 {
        match g {
            None => 0.0,
            Some(g) if g.initial => {
                if g.cut {
                    1.0
                } else {
                    0.0
                }
            }
            Some(_) => state.get(2).copied().unwrap_or(0.0),
        }
    }
}

impl ElementType for XfemElem {
    fn name(&self) -> &'static str {
        self.base.name()
    }
    fn family(&self) -> Family {
        self.base.family()
    }
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn n_nodes(&self) -> usize {
        self.nodes_total
    }
    fn n_connectivity(&self) -> usize {
        self.nodes_total
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        dofs(self.r.dim())
    }
    fn n_ips(&self) -> usize {
        self.base.n_ips()
    }
    fn vtk_type(&self) -> u8 {
        self.base.vtk_type()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.base.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut s = vec![0.0; HDR];
        s.extend(self.base.init_state(&self.base_ctx(ctx))?);
        Ok(s)
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let nd = self.dim() * self.nodes_total;
        let st = self.init_state(ctx)?;
        let mut sn = Vec::new();
        let r = self.internal_force(ctx, &vec![0.0; nd], &vec![0.0; nd], &st, &mut sn, true)?;
        Ok(r.k.unwrap_or_default())
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return super::api::not_implemented("geometric nonlinearity (NLGEOM) with XFEM", self.base.name());
        }
        let d = self.dim();
        let nbd = d * self.nb;
        let geo = self.geo();
        let hdr: Vec<f64> = if so.len() >= HDR { so[..HDR].to_vec() } else { vec![0.0; HDR] };
        let Some(g) = geo else {
            // not enriched: the base element
            self.stash_centroid(ctx, u)?;
            let bctx = self.base_ctx(ctx);
            let bold = if hdr[3] == 0.0 && so.len() > HDR { &so[HDR..] } else { &[][..] };
            let bold_owned;
            let bold = if bold.is_empty() {
                bold_owned = self.base.init_state(&bctx)?;
                &bold_owned[..]
            } else {
                bold
            };
            let mut bnew = Vec::new();
            let r = self.base.internal_force(&bctx, &u[..nbd], &du[..nbd], bold, &mut bnew, tangent)?;
            // the initiation: the integration point with the largest criterion value
            if !self.crack.stationary {
                let mat = ctx.material(self.base.name())?;
                let ns = self.mode().ncomp();
                let stride = ns + mat.n_state();
                if stride > 0 && bnew.len() % stride == 0 && !bnew.is_empty() {
                    let law = self.crack.law.as_ref().unwrap();
                    let best = bnew.chunks(stride).map(|c| c[..ns].to_vec()).max_by(|a, b| self.crack.criterion_value(law, a).partial_cmp(&self.crack.criterion_value(law, b)).unwrap());
                    if let Some(b) = best {
                        self.crack.stash.lock().unwrap().insert(self.elem, b);
                    }
                }
            }
            let mut fint = self.pad(&r.fint);
            let mut k = r.k.map(|k| self.pad_matrix(&k, nbd));
            self.ties(ctx, None, u, &mut fint, k.as_mut());
            sn.clear();
            sn.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            sn.extend(bnew);
            return Ok(ElemResponse { fint, k });
        };
        let mode = self.mode();
        let mat = ctx.material(self.base.name())?;
        let (ips, bs) = self.ips_b(ctx, &g)?;
        let ncoh = hdr[0] as usize;
        let ts_ns = self.crack.law.as_ref().and_then(|l| l.ts.cohesive()).map_or(0, |t| t.n_state());
        let enriched_before = hdr[3] == 1.0;
        let coh_old: Vec<f64> = if enriched_before && so.len() >= HDR + ncoh * ts_ns { so[HDR..HDR + ncoh * ts_ns].to_vec() } else { Vec::new() };
        let bulk_old: Vec<f64> = if enriched_before && so.len() > HDR + ncoh * ts_ns { so[HDR + ncoh * ts_ns..].to_vec() } else { Vec::new() };
        let mut bulk_new = Vec::new();
        let bctx = self.base_ctx(ctx);
        let r = common::response(&bctx, mode, &ips, &bs, mat, u, du, &bulk_old, &mut bulk_new, tangent)?;
        let mut fint = r.fint;
        let mut k = r.k;
        let mut coh_new = Vec::new();
        if !self.crack.stationary || self.crack.contact {
            self.faces_response(ctx, &g, u, &coh_old, &mut coh_new, &mut fint, k.as_mut())?;
        }
        self.ties(ctx, Some(&g), u, &mut fint, k.as_mut());
        let nc = if ts_ns > 0 { coh_new.len() / ts_ns } else { 0 };
        let dmax = (0..nc)
            .map(|q| self.crack.law.as_ref().and_then(|l| l.ts.cohesive()).map_or(0.0, |t| t.outputs(&coh_new[q * ts_ns..(q + 1) * ts_ns]).0))
            .fold(0.0, f64::max);
        sn.clear();
        sn.extend_from_slice(&[nc as f64, ips.len() as f64, dmax, 1.0]);
        sn.extend(coh_new);
        sn.extend(bulk_new);
        Ok(ElemResponse { fint, k })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let m = self.base.mass(&self.base_ctx(ctx), kind)?;
        Ok(self.pad_matrix(&m, self.dim() * self.nb))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let nbd = self.dim() * self.nb;
        let f = self.base.distributed_load(&self.base_ctx(ctx), load, &u[..nbd.min(u.len())])?;
        Ok(self.pad(&f))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut v = self.base.volume_weights(&self.base_ctx(ctx))?;
        v.resize(self.nodes_total, 0.0);
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let geo = self.geo();
        let mut out = Vec::new();
        if geo.is_none() {
            let bs = if state.len() > HDR && state[3] == 0.0 { &state[HDR..] } else { &[][..] };
            out = self.base.output(&self.base_ctx(ctx), &u[..self.dim() * self.nb], bs)?;
        }
        let mut f = IpField::new("STATUSXFEM", &["STATUSXFEM"]);
        f.points.push((0, 0, vec![self.status(geo.as_ref(), state)]));
        out.push(f);
        Ok(out)
    }
}

/// An integration point for contour integrals.
pub struct FieldPoint {
    pub x: [f64; 3],
    pub w: f64,
    /// Stress and displacement gradient g[j][i] = ∂u_j/∂x_i (tensors).
    pub s: [[f64; 3]; 3],
    pub g: [[f64; 3]; 3],
    pub eps: [[f64; 3]; 3],
    /// The shape functions and derivatives of the base nodes (for q).
    pub n: Vec<f64>,
    pub dndx: Vec<[f64; 3]>,
}

/// Integration points (linear elastic) for the element displacement `u`.
pub fn field_points(x: &XfemElem, ctx: &ElemCtx, u: &[f64]) -> Result<Vec<FieldPoint>> {
    let mode = x.mode();
    let mat = ctx.material(x.base.name())?;
    let dmat = mat.tangent_linear(mode)?;
    let ns = mode.ncomp();
    let d = x.dim();
    let t = x.thickness(ctx);
    let (g, pts) = match x.geo() {
        Some(g) => {
            let p = x.points(&g);
            (g, p)
        }
        None => {
            let nb = x.nb;
            let g = Geo { phi: vec![1.0; nb], psi: None, side: vec![1; nb], hact: vec![false; nb], cut: false, tipf: vec![None; nb], initial: false, order: 2 };
            (g, x.r.rule(2).into_iter().map(|(p, w)| (p, w, 0)).collect())
        }
    };
    let mut out = Vec::new();
    for (p, wn, side) in pts {
        let ip = x.ip(ctx, &g, p, wn, side, t)?;
        let (nn, dndx, _) = physical(x.r, &ctx.x[..x.nb], p)?;
        let pos = geom::bary_point(&ctx.x[..x.nb], &nn);
        let mut gu = [[0.0; 3]; 3];
        for (a, gr) in ip.dndx.iter().enumerate() {
            for m in 0..d {
                for k in 0..3 {
                    gu[m][k] += u[a * d + m] * gr[k];
                }
            }
        }
        let b = common::bmat(mode, &ip.dndx);
        let eps = common::strain(&b, u);
        let s: Vec<f64> = (0..ns).map(|i| (0..ns).map(|j| dmat[i * ns + j] * eps[j]).sum()).collect();
        let mut e = sym3(mode, &eps);
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    e[i][j] *= 0.5;
                }
            }
        }
        out.push(FieldPoint { x: pos, w: ip.w, s: sym3(mode, &s), g: gu, eps: e, n: nn, dndx });
    }
    Ok(out)
}

/// The tip elements and the nodal level sets for a stationary crack:
/// (element, φ, ψ per base node).
pub fn tip_elements(model: &FeModel, crack: &'static Crack) -> Vec<(usize, Vec<f64>, Vec<f64>)> {
    let mut v = Vec::new();
    for &e in &crack.domain {
        if let Some(x) = as_xfem(model.elements[e].etype) {
            if let Some(st) = &x.stat {
                if st.tip {
                    v.push((e, st.phi.clone(), st.psi.clone()));
                }
            }
        }
    }
    v
}
