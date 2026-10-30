//! Area explicit_adv: DEM (discrete element method) with PD3D in
//! explicit analysis, and contact between spherical particles (PD3D and SPH
//! PC3D) and element surfaces.
//!
//! The particle is a rigid sphere with radius r and density ρ from
//! `*DISCRETE SECTION, DENSITY=ρ` (data line: r): m = ρ·4πr³/3 and
//! the rotary inertia I = 2/5 m r² (degrees of freedom UR1–UR3).
//! `ALPHA=α` gives mass-proportional damping −α m v and −α I ω (ALLVD).
//!
//! Contact is part of general contact (`*CONTACT`, `*CONTACT INCLUSIONS` with
//! the particle surface, `*SURFACE, TYPE=ELEMENT` with the particle elements without
//! a face label, or a node surface for PC3D) against particles or element surfaces.
//! Properties from `*CONTACT PROPERTY ASSIGNMENT` and `*SURFACE INTERACTION`:
//!
//! - Normal force: `*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR` (force =
//!   slope · penetration δ), `=HERTZ` (F = 4/3 E* √R* δ^{3/2} with E and ν
//!   from `*DISCRETE ELASTICITY`, 1/E* = Σ(1 − ν²)/E, R* = r₁r₂/(r₁ + r₂);
//!   the data line gives the largest stiffness K_max) or hard contact with
//!   mass-based penalty stiffness κ m/Δt² (like `explicit_contact.rs`).
//! - Damping: c = 2ζ √(K m) with K = the normal stiffness and m = the smaller of the
//!   two masses (a rigid/massless counter surface: the particle mass), ζ = 0.03 by
//!   default for penalty contact or `*CONTACT DAMPING, DEFINITION=
//!   CRITICAL DAMPING FRACTION` (checked for a chain of three
//!   particles with linear contact).
//!   The damping is counted as ALLVD.
//! - Friction (`*FRICTION` μ): tangential spring with the normal stiffness and
//!   the Coulomb limit μ Fₙ; the force acts at the contact point and gives moments on
//!   the particles (rolling). The work of sliding gives ALLFD.
//!
//! Particles against element surfaces: closest point on the facet, δ = r − the distance
//! along the facet normal; the force is distributed to the facet nodes with
//! the shape functions. SPH particles (PC3D) are spheres with radius L (the
//! characteristic length, as contact thickness) without rotation.

use super::{Eval, Sink};
use crate::elements::common::{cross, dot, norm, sub};
use crate::fem::assembly::Prepared;
use crate::fem::model::{parse_f64, FeModel, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::interactions::geom::{self, Facet};
use crate::model::{fail, FemError, Result};
use std::collections::{HashMap, HashSet};

/// Default damping (fraction of critical damping) for penalty contact.
pub const DEFAULT_ZETA: f64 = 0.03;

/// A spherical particle.
#[derive(Debug, Clone)]
pub struct Sphere {
    pub e: usize,
    pub node: usize,
    pub r: f64,
    pub mass: f64,
    pub inertia: f64,
    pub rot: bool,
    pub alpha: f64,
    /// Hertz: (E, ν) from *DISCRETE ELASTICITY.
    pub elastic: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy)]
enum Normal {
    Hard,
    Linear(f64),
    Hertz(Option<f64>),
}

#[derive(Debug, Clone, Copy)]
struct Props {
    normal: Normal,
    mu: f64,
    zeta: f64,
}

impl Default for Props {
    fn default() -> Self {
        Props { normal: Normal::Hard, mu: 0.0, zeta: DEFAULT_ZETA }
    }
}

enum Target {
    Spheres(Vec<usize>),
    Facets(Vec<Facet>),
}

struct Pair {
    a: Vec<usize>,
    b: Target,
    props: Props,
    /// Tangential force (on b) per contact (a, b index).
    ft: HashMap<(usize, usize), [f64; 3]>,
}

pub struct Dem {
    pub spheres: Vec<Sphere>,
    pairs: Vec<Pair>,
    /// Sphere index per node.
    of_node: HashMap<usize, usize>,
}

/// The elements in the surface if it consists only of particles (PD3D, PC3D), or
/// the nodes of the node surface if all belong to particles. `None` otherwise.
pub fn particle_surface(m: &FeModel, name: &str) -> Option<Vec<usize>> {
    let s = m.surfaces.get(&name.trim().to_uppercase())?;
    let part_of_node = || -> HashMap<usize, usize> {
        let mut h = HashMap::new();
        for (e, el) in m.elements.iter().enumerate() {
            if crate::elements::explicit_adv::is_particle(el.etype.name()) {
                h.insert(el.nodes[0], e);
            }
        }
        h
    };
    if s.kind == "NODE" {
        let h = part_of_node();
        let v: Option<Vec<usize>> = s.nodes.iter().map(|(n, _)| h.get(n).copied()).collect();
        return v.filter(|v| !v.is_empty());
    }
    if s.faces.is_empty() {
        return None;
    }
    let v: Vec<usize> = s.faces.iter().map(|(e, _)| *e).collect();
    v.iter().all(|&e| crate::elements::explicit_adv::is_particle(m.elements[e].etype.name())).then_some(v)
}

/// The *CONTACT block contains particle surfaces (read here and not in `explicit_contact.rs`).
pub fn is_particle_pair(m: &FeModel, a: &str, b: &str) -> bool {
    particle_surface(m, a).is_some() || particle_surface(m, b).is_some()
}

fn sub_options<'a>(m: &'a FeModel, kw: &RawKeyword, names: &[&str]) -> Vec<&'a RawKeyword> {
    let Some(i) = m.extra.iter().position(|k| std::ptr::eq(k, kw)) else { return Vec::new() };
    m.extra.iter().skip(i + 1).take_while(|k| names.contains(&k.name.as_str())).collect()
}

fn properties(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<Props> {
    let up = name.trim().to_uppercase();
    let Some(pos) = m.extra.iter().position(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*SURFACE INTERACTION, NAME={name} does not exist"));
    };
    let mut pr = Props::default();
    for sub in m.extra.iter().skip(pos + 1).take_while(|k| matches!(k.name.as_str(), "SURFACE BEHAVIOR" | "FRICTION" | "CONTACT DAMPING")) {
        match sub.name.as_str() {
            "FRICTION" => {
                if sub.has("ROUGH") || sub.data.len() > 1 || sub.has("ANISOTROPIC") || sub.has("DEPENDENCIES") {
                    return sub.err("particle contact: only isotropic Coulomb friction with a constant μ is supported");
                }
                pr.mu = sub.first_floats()?.first().copied().unwrap_or(0.0);
            }
            "SURFACE BEHAVIOR" => {
                let v = sub.first_floats().unwrap_or_default();
                pr.normal = match sub.param_upper("PRESSURE-OVERCLOSURE").as_deref() {
                    None | Some("HARD") => Normal::Hard,
                    Some("LINEAR") => match v.first().copied().filter(|&k| k > 0.0) {
                        Some(k) => Normal::Linear(k),
                        None => return sub.err("PRESSURE-OVERCLOSURE=LINEAR: specify the slope (> 0)"),
                    },
                    Some("HERTZ") => Normal::Hertz(v.first().copied().filter(|&k| k > 0.0)),
                    Some(o) => return sub.err(format!("particle contact: PRESSURE-OVERCLOSURE={o} is not supported yet (HARD, LINEAR, HERTZ)")),
                };
                if sub.has("NO SEPARATION") {
                    return sub.err("particle contact: NO SEPARATION is not supported");
                }
            }
            _ => {
                let d = crate::interactions::contact::damping::parse(sub, true)?;
                if !d.critical {
                    return sub.err("particle contact: only DEFINITION=CRITICAL DAMPING FRACTION is supported");
                }
                pr.zeta = d.mu0;
            }
        }
    }
    Ok(pr)
}

impl Dem {
    pub fn build(p: &Prepared, _step: &StepDef) -> Result<Option<Dem>> {
        let m = p.model;
        let mut spheres = Vec::new();
        let mut of_node = HashMap::new();
        let mut of_elem: HashMap<usize, usize> = HashMap::new();
        for (e, el) in m.elements.iter().enumerate() {
            let sph = match el.etype.name() {
                "PD3D" => {
                    let Some(si) = el.section else { return fail(format!("{} has no section (*DISCRETE SECTION)", m.elem_name(e))) };
                    let sec = &m.sections[si];
                    if sec.raw.name != "DISCRETE SECTION" {
                        return sec.raw.err("PD3D requires *DISCRETE SECTION");
                    }
                    let Some(rho) = sec.raw.param("DENSITY").and_then(parse_f64).filter(|&r| r > 0.0) else {
                        return sec.raw.err("*DISCRETE SECTION: DENSITY= (> 0) is missing (distributions are not supported yet)");
                    };
                    let r = sec.raw.first_floats()?.first().copied().unwrap_or(0.0);
                    if !(r > 0.0) {
                        return sec.raw.err("*DISCRETE SECTION: specify the particle radius (> 0) on the data line");
                    }
                    let alpha = sec.raw.param("ALPHA").and_then(parse_f64).unwrap_or(0.0);
                    let elastic = match (sec.raw.param(ELASTIC_E).and_then(parse_f64), sec.raw.param(ELASTIC_NU).and_then(parse_f64)) {
                        (Some(e), Some(nu)) => Some((e, nu)),
                        _ => None,
                    };
                    let mass = rho * 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
                    Sphere { e, node: el.nodes[0], r, mass, inertia: 0.4 * mass * r * r, rot: true, alpha, elastic }
                }
                "PC3D" => {
                    // SPH particle: the contact radius is the characteristic length
                    let Some(si) = el.section else { continue };
                    let sec = &m.sections[si];
                    let l = sec.raw.first_floats()?.first().copied().unwrap_or(0.0);
                    let rho = p.ctx(e, false, 0.0, 0.0).density;
                    Sphere { e, node: el.nodes[0], r: l, mass: rho * (2.0 * l).powi(3), inertia: 0.0, rot: false, alpha: 0.0, elastic: None }
                }
                _ => continue,
            };
            of_node.insert(sph.node, spheres.len());
            of_elem.insert(e, spheres.len());
            spheres.push(sph);
        }
        if spheres.is_empty() {
            return Ok(None);
        }
        // contact pairs from *CONTACT
        let mut pairs = Vec::new();
        for kw in m.extra.iter().filter(|k| k.name == "CONTACT") {
            let subs = sub_options(m, kw, &["CONTACT INCLUSIONS", "CONTACT PROPERTY ASSIGNMENT", "CONTACT EXCLUSIONS", "CONTACT FORMULATION", "CONTACT CONTROLS ASSIGNMENT", "CONTACT INITIALIZATION ASSIGNMENT", "SURFACE PROPERTY ASSIGNMENT"]);
            let mut glob = Props::default();
            let mut per: Vec<(String, String, Props)> = Vec::new();
            for s in subs.iter().filter(|s| s.name == "CONTACT PROPERTY ASSIGNMENT") {
                for d in &s.data {
                    let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                    if f.len() >= 3 && !f[2].is_empty() {
                        // errors in the properties are reported only for particle pairs
                        let pr = match properties(m, s, f[2]) {
                            Ok(pr) => pr,
                            Err(_) => continue,
                        };
                        if f[0].is_empty() && f[1].is_empty() {
                            glob = pr;
                        } else {
                            per.push((f[0].to_uppercase(), f[1].to_uppercase(), pr));
                        }
                    }
                }
            }
            for s in subs.iter().filter(|s| s.name == "CONTACT INCLUSIONS") {
                for d in &s.data {
                    let f: Vec<String> = d.fields.iter().map(|x| x.trim().to_uppercase()).collect();
                    let (a, b) = match (f.first().filter(|x| !x.is_empty()), f.get(1).filter(|x| !x.is_empty())) {
                        (Some(a), Some(b)) => (a.clone(), b.clone()),
                        (Some(a), None) => (a.clone(), a.clone()),
                        _ => continue,
                    };
                    let (pa, pb) = (particle_surface(m, &a), particle_surface(m, &b));
                    if pa.is_none() && pb.is_none() {
                        continue;
                    }
                    // the properties: per surface pair (in either order) or global
                    let pr = match per.iter().find(|(x, y, _)| (x == &a && y == &b) || (x == &b && y == &a)) {
                        Some(x) => x.2,
                        None => glob,
                    };
                    // errors in the property used
                    for s2 in subs.iter().filter(|s| s.name == "CONTACT PROPERTY ASSIGNMENT") {
                        for d2 in &s2.data {
                            let f2: Vec<&str> = d2.fields.iter().map(|x| x.trim()).collect();
                            if f2.len() >= 3 && !f2[2].is_empty() {
                                properties(m, s2, f2[2])?;
                            }
                        }
                    }
                    let to_spheres = |els: &[usize]| -> Vec<usize> { els.iter().filter_map(|e| of_elem.get(e).copied()).collect() };
                    match (pa, pb) {
                        (Some(ea), Some(eb)) => pairs.push(Pair { a: to_spheres(&ea), b: Target::Spheres(to_spheres(&eb)), props: pr, ft: HashMap::new() }),
                        (Some(ea), None) => {
                            let fb = geom::facets(m, geom::surface(m, kw, &b)?, &b)?;
                            pairs.push(Pair { a: to_spheres(&ea), b: Target::Facets(fb), props: pr, ft: HashMap::new() });
                        }
                        (None, Some(eb)) => {
                            let fa = geom::facets(m, geom::surface(m, kw, &a)?, &a)?;
                            pairs.push(Pair { a: to_spheres(&eb), b: Target::Facets(fa), props: pr, ft: HashMap::new() });
                        }
                        (None, None) => {}
                    }
                }
            }
        }
        if !spheres.iter().any(|s| s.rot) && pairs.is_empty() {
            // only SPH particles without contact
            return Ok(None);
        }
        Ok(Some(Dem { spheres, pairs, of_node }))
    }

    pub fn patch_mass(&self, mass: &mut super::LumpedMass) {
        for s in self.spheres.iter().filter(|s| s.rot) {
            for k in 0..3 {
                mass.diag[s.node][k] += s.mass;
                mass.diag[s.node][3 + k] += s.inertia;
            }
        }
    }

    /// Contact and damping forces. Returns the frictional dissipation in the increment.
    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink) -> Result<f64> {
        let m = p.model;
        let x: Vec<[f64; 3]> = self.spheres.iter().map(|s| std::array::from_fn(|c| m.nodes[s.node][c] + ev.u[s.node][c])).collect();
        let v: Vec<[f64; 3]> = self.spheres.iter().map(|s| std::array::from_fn(|c| ev.v[s.node][c])).collect();
        let w: Vec<[f64; 3]> = self.spheres.iter().map(|s| if s.rot { std::array::from_fn(|c| ev.v[s.node][3 + c]) } else { [0.0; 3] }).collect();
        // mass-proportional damping
        for (i, s) in self.spheres.iter().enumerate() {
            if s.alpha != 0.0 && s.rot {
                for c in 0..3 {
                    out.fvis[s.node][c] += s.alpha * s.mass * v[i][c];
                    out.fvis[s.node][3 + c] += s.alpha * s.inertia * w[i][c];
                }
            }
        }
        if ev.dt <= 0.0 {
            return Ok(0.0);
        }
        let dt = ev.dt;
        let mut fric = 0.0;
        let spheres = &self.spheres;
        let nodemass = |n: usize| out.mass.diag[n][0];
        for pair in self.pairs.iter_mut() {
            let pr = pair.props;
            let mut seen: HashSet<(usize, usize)> = HashSet::new();
            match &pair.b {
                Target::Spheres(bs) => {
                    // neighbours via grid (the spheres in b)
                    let rmax = pair.a.iter().chain(bs.iter()).map(|&i| spheres[i].r).fold(0.0f64, f64::max);
                    let cell = 2.0 * rmax.max(1e-300);
                    let key = |q: &[f64; 3]| ((q[0] / cell).floor() as i64, (q[1] / cell).floor() as i64, (q[2] / cell).floor() as i64);
                    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
                    for &j in bs {
                        grid.entry(key(&x[j])).or_default().push(j);
                    }
                    for &i in &pair.a {
                        let (ka, kb, kc) = key(&x[i]);
                        for da in -1..=1 {
                            for db in -1..=1 {
                                for dc in -1..=1 {
                                    let Some(cellv) = grid.get(&(ka + da, kb + db, kc + dc)) else { continue };
                                    for &j in cellv {
                                        if j == i {
                                            continue;
                                        }
                                        let k = (i.min(j), i.max(j));
                                        if seen.contains(&k) {
                                            continue;
                                        }
                                        let d = sub(x[j], x[i]);
                                        let dist = norm(d);
                                        let delta = spheres[i].r + spheres[j].r - dist;
                                        if delta <= 0.0 || dist <= 0.0 {
                                            pair.ft.remove(&k);
                                            continue;
                                        }
                                        seen.insert(k);
                                        let (i0, j0) = k;
                                        let d = sub(x[j0], x[i0]);
                                        let n = [d[0] / dist, d[1] / dist, d[2] / dist];
                                        let (si, sj) = (&spheres[i0], &spheres[j0]);
                                        let mmin = si.mass.min(sj.mass);
                                        let estar = match (si.elastic, sj.elastic) {
                                            (Some(a), Some(b)) => Some(1.0 / ((1.0 - a.1 * a.1) / a.0 + (1.0 - b.1 * b.1) / b.0)),
                                            _ => None,
                                        };
                                        let rstar = si.r * sj.r / (si.r + sj.r);
                                        let (fe, kn) = normal_force(pr.normal, delta, mmin, dt, estar, rstar).map_err(|e| FemError(format!("{}: {}", m.elem_name(si.e), e.0)))?;
                                        // the contact point and the relative velocity (j relative to i)
                                        let vci: [f64; 3] = std::array::from_fn(|c| v[i0][c] + cross(w[i0], [si.r * n[0], si.r * n[1], si.r * n[2]])[c]);
                                        let vcj: [f64; 3] = std::array::from_fn(|c| v[j0][c] + cross(w[j0], [-sj.r * n[0], -sj.r * n[1], -sj.r * n[2]])[c]);
                                        let vrel = sub(vcj, vci);
                                        let vn = dot(vrel, n);
                                        let c_n = 2.0 * pr.zeta * (kn * mmin).sqrt();
                                        let fd = -c_n * vn;
                                        // normal: on j along +n, on i along −n
                                        for c in 0..3 {
                                            out.fcon[sj.node][c] += fe * n[c];
                                            out.fcon[si.node][c] -= fe * n[c];
                                            // the damping as a viscous force (fvis is subtracted)
                                            out.fvis[sj.node][c] -= fd * n[c];
                                            out.fvis[si.node][c] += fd * n[c];
                                        }
                                        if pr.mu > 0.0 {
                                            // Coulomb limit with the elastic part of the normal force (read off: rolling on a plate 2e-2 → 1e-3)
                                let fnorm = fe.max(0.0);
                                            let vt: [f64; 3] = std::array::from_fn(|c| vrel[c] - vn * n[c]);
                                            let ft_old = pair.ft.get(&k).copied().unwrap_or([0.0; 3]);
                                            let inv = 1.0 / si.mass + 1.0 / sj.mass + if si.rot { si.r * si.r / si.inertia } else { 0.0 } + if sj.rot { sj.r * sj.r / sj.inertia } else { 0.0 };
                                            let (ft, diss) = tangential(ft_old, n, vt, tangent_stiffness(inv, dt), dt, pr.mu * fnorm);
                                            fric += diss;
                                            pair.ft.insert(k, ft);
                                            let mj = cross([-sj.r * n[0], -sj.r * n[1], -sj.r * n[2]], ft);
                                            let mi = cross([si.r * n[0], si.r * n[1], si.r * n[2]], [-ft[0], -ft[1], -ft[2]]);
                                            for c in 0..3 {
                                                out.fcon[sj.node][c] += ft[c];
                                                out.fcon[si.node][c] -= ft[c];
                                                if sj.rot {
                                                    out.fcon[sj.node][3 + c] += mj[c];
                                                }
                                                if si.rot {
                                                    out.fcon[si.node][3 + c] += mi[c];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Target::Facets(fs) => {
                    for &i in &pair.a {
                        let si = &spheres[i];
                        for (fi, f) in fs.iter().enumerate() {
                            let xf = geom::coords(m, f, Some(ev.u));
                            let (xi, _, inside) = geom::project(f, &xf, x[i]);
                            let k = (i, fi);
                            if !inside {
                                pair.ft.remove(&k);
                                continue;
                            }
                            let fp = geom::eval(f, &xf, xi);
                            let gap = dot(sub(x[i], fp.x), fp.normal);
                            let delta = si.r - gap;
                            if delta <= 0.0 || gap < -si.r {
                                pair.ft.remove(&k);
                                continue;
                            }
                            // the normal from the facet towards the particle
                            let n = fp.normal;
                            let mf: f64 = f.nodes.iter().zip(&fp.n).map(|(&nd, &w)| w * nodemass(nd)).sum();
                            let mmin = if mf > 0.0 { si.mass.min(mf) } else { si.mass };
                            let estar = si.elastic.map(|a| a.0 / (1.0 - a.1 * a.1));
                            let (fe, kn) = normal_force(pr.normal, delta, mmin, dt, estar, si.r).map_err(|e| FemError(format!("{}: {}", m.elem_name(si.e), e.0)))?;
                            let vf: [f64; 3] = std::array::from_fn(|c| f.nodes.iter().zip(&fp.n).map(|(&nd, &w)| w * ev.v[nd][c]).sum());
                            // the particle's contact point (towards the facet: −r n)
                            let vc: [f64; 3] = std::array::from_fn(|c| v[i][c] + cross(w[i], [-si.r * n[0], -si.r * n[1], -si.r * n[2]])[c]);
                            let vrel = sub(vc, vf);
                            let vn = dot(vrel, n);
                            let c_n = 2.0 * pr.zeta * (kn * mmin).sqrt();
                            let fd = -c_n * vn;
                            for c in 0..3 {
                                out.fcon[si.node][c] += fe * n[c];
                                out.fvis[si.node][c] -= fd * n[c];
                            }
                            for (&nd, &wk) in f.nodes.iter().zip(&fp.n) {
                                for c in 0..3 {
                                    out.fcon[nd][c] -= wk * fe * n[c];
                                    out.fvis[nd][c] += wk * fd * n[c];
                                }
                            }
                            if pr.mu > 0.0 {
                                // Coulomb limit with the elastic part of the normal force (read off: rolling on a plate 2e-2 → 1e-3)
                                let fnorm = fe.max(0.0);
                                let vt: [f64; 3] = std::array::from_fn(|c| vrel[c] - vn * n[c]);
                                let ft_old = pair.ft.get(&k).copied().unwrap_or([0.0; 3]);
                                let inv = 1.0 / si.mass + if si.rot { si.r * si.r / si.inertia } else { 0.0 };
                                let (ft, diss) = tangential(ft_old, n, vt, tangent_stiffness(inv, dt), dt, pr.mu * fnorm);
                                fric += diss;
                                pair.ft.insert(k, ft);
                                let mi = cross([-si.r * n[0], -si.r * n[1], -si.r * n[2]], ft);
                                for c in 0..3 {
                                    out.fcon[si.node][c] += ft[c];
                                    if si.rot {
                                        out.fcon[si.node][3 + c] += mi[c];
                                    }
                                }
                                for (&nd, &wk) in f.nodes.iter().zip(&fp.n) {
                                    for c in 0..3 {
                                        out.fcon[nd][c] -= wk * ft[c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let _ = &self.of_node;
        Ok(fric)
    }

    pub fn output(&self, _p: &Prepared, _fr: &mut FrameData) {}
}

/// The elastic part of the normal force and the normal stiffness at the penetration δ.
fn normal_force(normal: Normal, delta: f64, mmin: f64, dt: f64, estar: Option<f64>, rstar: f64) -> Result<(f64, f64)> {
    Ok(match normal {
        Normal::Linear(k) => (k * delta, k),
        Normal::Hard => {
            let k = crate::steps::explicit::contact::KAPPA * mmin / (dt * dt);
            (k * delta, k)
        }
        Normal::Hertz(kmax) => {
            let Some(es) = estar else { return fail("PRESSURE-OVERCLOSURE=HERTZ requires *DISCRETE ELASTICITY (E, ν)") };
            let f = 4.0 / 3.0 * es * rstar.sqrt() * delta.powf(1.5);
            let k = 2.0 * es * (rstar * delta).sqrt();
            match kmax {
                Some(km) if k > km => {
                    // F grows linearly with the slope at K_max beyond δ_max
                    let dmax = (km / (2.0 * es)).powi(2) / rstar;
                    let fmax = 4.0 / 3.0 * es * rstar.sqrt() * dmax.powf(1.5);
                    (fmax + km * (delta - dmax), km)
                }
                _ => (f, k.max(1e-300)),
            }
        }
    })
}

/// Tangential penalty stiffness kₜ = m*/Δt², m* = the effective mass for
/// relative sliding at the contact point (1/m* = Σ 1/m + r²/I): sticking contact
/// stops the sliding velocity in one increment (rolling without sliding,
/// v = 5/7 v₀ for a sphere on a plane).
fn tangent_stiffness(inv_mass: f64, dt: f64) -> f64 {
    1.0 / (inv_mass * dt * dt)
}

/// Tangential spring (stiffness kₜ) with the Coulomb limit `limit`:
/// the new tangential force and the dissipation of sliding.
fn tangential(ft_old: [f64; 3], n: [f64; 3], vt: [f64; 3], kt: f64, dt: f64, limit: f64) -> ([f64; 3], f64) {
    // the previous force rotated to the new tangent plane
    let fo = dot(ft_old, n);
    let mut ft: [f64; 3] = std::array::from_fn(|c| ft_old[c] - fo * n[c]);
    let l0 = norm(ft_old);
    let l1 = norm(ft);
    if l1 > 0.0 {
        ft = ft.map(|x| x * l0 / l1);
    }
    // the force on b opposes the sliding of b relative to a
    for c in 0..3 {
        ft[c] -= kt * vt[c] * dt;
    }
    let l = norm(ft);
    if l > limit && l > 0.0 {
        let s = limit / l;
        let slip = (l - limit) / kt;
        return (ft.map(|x| x * s), limit * slip);
    }
    (ft, 0.0)
}

/// Location of *DISCRETE ELASTICITY on the section keyword (E and ν).
pub const ELASTIC_E: &str = "RUSTYFEM_DISCRETE_E";
pub const ELASTIC_NU: &str = "RUSTYFEM_DISCRETE_NU";
