//! Acoustic–structural coupling (ASI) with *TIE between an acoustic surface
//! (faces of AC elements) and a structural surface (continuum, shell), in any
//! order.
//!
//! The coupling matrix S_{a,(b,i)} = ∫ N_a n_i N_b dS with n = the fluid's
//! outward normal. Surface-to-surface (default for solids): integration over
//! the acoustic surface with N_b the structural surface's shape functions at
//! the projection of the integration point (closest facet within the position
//! tolerance). Node-to-surface (TYPE=NODE TO SURFACE, and by default when the
//! structural surface is a shell): the slave surface nodes are coupled with
//! their nodal area ∫ N dS to the master surface shape functions at the node's
//! projection. Axisymmetric surfaces are integrated over the full revolution
//! (2π r) and planar ones with the thickness.
//!
//! With S (Zienkiewicz–Newton, see `steps/acoustic.rs`):
//!
//! ```text
//!   M_s ü + K_s u − Sᵀ p = F          (the pressure acts on the structure along n)
//!   M_f p̈ + K_f p + S ü = Q
//! ```
//!
//! The interaction here only provides the matrix structure (`connectivity`) so
//! that the solver's pattern contains the coupling; the matrix itself is built
//! by the acoustic steps with `coupling`. `constraints.rs` skips *TIE pairs
//! that are acoustic–structural (`is_acoustic_pair`).

use super::geom::{self, Facet};
use super::{Interaction, InteractionFactory};
use crate::elements::acoustic;
use crate::fem::model::{parse_f64, FeModel, RawKeyword, Surface};
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["TIE"], build: tie }];

/// The surface consists of acoustic element faces.
pub fn is_acoustic_surface(m: &FeModel, s: &Surface) -> bool {
    s.kind != "NODE" && !s.faces.is_empty() && s.faces.iter().all(|(e, _)| acoustic::is_acoustic(m.elements[*e].etype))
}

/// The *TIE pair (slave surface, master surface) couples an acoustic and a structural surface.
pub fn is_acoustic_pair(m: &FeModel, kw: &RawKeyword, a: &str, b: &str) -> Result<bool> {
    let sa = geom::surface(m, kw, a)?;
    let sb = geom::surface(m, kw, b)?;
    Ok(is_acoustic_surface(m, sa) != is_acoustic_surface(m, sb))
}

/// An ASI pair: the coupling matrix entries (pressure node, structural node, direction 0–2, value).
pub struct Coupling {
    pub name: String,
    pub entries: Vec<(usize, usize, usize, f64)>,
    /// Node groups (a fluid facet and the structural nodes it is coupled to).
    pub groups: Vec<Vec<usize>>,
    /// Node-to-surface with the fluid as slave: the pressure node's displacement
    /// interpolated from the structure (node, [(structural node, weight)]); U is output there.
    pub fluid_u: Vec<(usize, Vec<(usize, f64)>)>,
}

struct AsiTie {
    c: Coupling,
}

impl Interaction for AsiTie {
    fn name(&self) -> String {
        self.c.name.clone()
    }
    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.c.groups.clone()
    }
}

fn tie(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    Ok(couplings_of(m, kw)?.into_iter().map(|c| Box::new(AsiTie { c }) as Box<dyn Interaction>).collect())
}

/// All ASI couplings in the model (from *TIE).
pub fn couplings(m: &FeModel) -> Result<Vec<Coupling>> {
    let mut out = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "TIE") {
        out.extend(couplings_of(m, kw)?);
    }
    Ok(out)
}

fn surface_size(m: &FeModel, f: &[Facet]) -> f64 {
    let mut size = 0.0;
    for fc in f {
        let xs = geom::coords(m, fc, None);
        let mut d = 0.0f64;
        for a in &xs {
            for b in &xs {
                d = d.max(crate::elements::common::norm(crate::elements::common::sub(*a, *b)));
            }
        }
        size += d;
    }
    size / f.len().max(1) as f64
}

/// Closest facet among `sf` for the point `p`: (facet, ξ, distance).
fn closest(sf: &[Facet], sx: &[Vec<[f64; 3]>], p: [f64; 3], size: f64) -> Option<(usize, [f64; 2], f64)> {
    let mut best: Option<(usize, [f64; 2], f64, bool)> = None;
    for (i, s) in sf.iter().enumerate() {
        let (xi, dist, inside) = geom::project(s, &sx[i], p);
        let better = match &best {
            None => true,
            Some(b) => (inside && !b.3) || (inside == b.3 && dist < b.2 - 1e-12 * size),
        };
        if better {
            best = Some((i, xi, dist, inside));
        }
    }
    best.map(|b| (b.0, b.1, b.2))
}

/// The facet weight at the integration point: length · thickness (planar), area
/// (3D) or length · 2π r (axisymmetric elements).
fn weight(m: &FeModel, fc: &Facet, fp: &geom::FacetPoint, wg: f64) -> f64 {
    let et = m.elements[fc.elem].etype;
    let axisym = acoustic::lookup(et.name()).is_some_and(|a| a.geom.axisym) || et.family() == crate::elements::Family::Axisymmetric;
    let w = wg * fp.jac;
    if axisym && fc.dim == 2 {
        w / fc.thick * 2.0 * std::f64::consts::PI * fp.x[0]
    } else {
        w
    }
}

fn couplings_of(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Coupling>> {
    let tol_given = kw.param("POSITION TOLERANCE").and_then(parse_f64);
    let ty = kw.param_upper("TYPE");
    let mut out = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.len() < 2 || !is_acoustic_pair(m, kw, f[0], f[1])? {
            continue;
        }
        let (sa, sb) = (geom::surface(m, kw, f[0])?, geom::surface(m, kw, f[1])?);
        let fluid_slave = is_acoustic_surface(m, sa);
        let (fluid, solid) = if fluid_slave { (sa, sb) } else { (sb, sa) };
        if solid.kind == "NODE" {
            return kw.err("the structural surface in an acoustic–structural *TIE must be element-based");
        }
        let name = format!("*TIE {} ({}, acoustic–structural)", kw.param("NAME").unwrap_or(""), kw.location());
        let ff = geom::facets(m, fluid, &name)?;
        let sf = geom::facets(m, solid, &name)?;
        let fx: Vec<Vec<[f64; 3]>> = ff.iter().map(|f| geom::coords(m, f, None)).collect();
        let sx: Vec<Vec<[f64; 3]>> = sf.iter().map(|f| geom::coords(m, f, None)).collect();
        let size = surface_size(m, &sf).max(surface_size(m, &ff));
        let tol = tol_given.unwrap_or(0.05 * size);
        // node-to-surface if TYPE=NODE TO SURFACE, or by default when the
        // structural surface consists of shells/membranes (with a warning)
        let shells = sf.iter().all(|f| matches!(m.elements[f.elem].etype.family(), crate::elements::Family::Shell | crate::elements::Family::Membrane));
        let n2s = match ty.as_deref() {
            Some("NODE TO SURFACE") => true,
            Some("SURFACE TO SURFACE") => false,
            None => shells,
            Some(x) => return kw.err(format!("TYPE={x} is not supported for acoustic–structural *TIE")),
        };
        let mut entries = Vec::new();
        let mut groups = Vec::new();
        let mut fluid_u = Vec::new();
        if !n2s {
            // surface-to-surface: integrate over the fluid surface
            for (fi, fc) in ff.iter().enumerate() {
                let xs = &fx[fi];
                let mut touched: Vec<usize> = fc.nodes.clone();
                for (xi, wg) in geom::gauss(fc.dim, xs.len(), 4) {
                    let fp = geom::eval(fc, xs, xi);
                    let w = weight(m, fc, &fp, wg);
                    let Some((si, sxi, dist)) = closest(&sf, &sx, fp.x, size) else { continue };
                    if dist > tol {
                        continue;
                    }
                    let s = &sf[si];
                    let (ns, _, _) = geom::shape(s.dim, s.nodes.len(), sxi);
                    for (a, &pn) in fc.nodes.iter().enumerate() {
                        let na = fp.n[a];
                        if na == 0.0 {
                            continue;
                        }
                        for (b, &un) in s.nodes.iter().enumerate() {
                            for (dir, &nd) in fp.normal.iter().enumerate().take(fc.dim) {
                                let v = w * na * nd * ns[b];
                                if v != 0.0 {
                                    entries.push((pn, un, dir, v));
                                }
                            }
                        }
                    }
                    touched.extend(s.nodes.iter().copied());
                }
                touched.sort_unstable();
                touched.dedup();
                groups.push(touched);
            }
        } else if fluid_slave {
            // node-to-surface, the fluid is slave: each pressure node a gets A_a = ∫ N_a n dS
            // and is coupled to the structural surface shape functions at the node's projection
            use std::collections::BTreeMap;
            let mut area: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
            for (fi, fc) in ff.iter().enumerate() {
                for (xi, wg) in geom::gauss(fc.dim, fx[fi].len(), 4) {
                    let fp = geom::eval(fc, &fx[fi], xi);
                    let w = weight(m, fc, &fp, wg);
                    for (a, &pn) in fc.nodes.iter().enumerate() {
                        let e = area.entry(pn).or_default();
                        for dir in 0..3 {
                            e[dir] += w * fp.n[a] * fp.normal[dir];
                        }
                    }
                }
            }
            for (pn, av) in area {
                let Some((si, sxi, dist)) = closest(&sf, &sx, m.nodes[pn], size) else { continue };
                if dist > tol {
                    continue;
                }
                let s = &sf[si];
                let (ns, _, _) = geom::shape(s.dim, s.nodes.len(), sxi);
                let mut interp = Vec::new();
                for (b, &un) in s.nodes.iter().enumerate() {
                    if ns[b] == 0.0 {
                        continue;
                    }
                    interp.push((un, ns[b]));
                    for (dir, &a) in av.iter().enumerate() {
                        if a != 0.0 {
                            entries.push((pn, un, dir, a * ns[b]));
                        }
                    }
                }
                let mut g: Vec<usize> = s.nodes.clone();
                g.push(pn);
                groups.push(g);
                fluid_u.push((pn, interp));
            }
        } else {
            // node-to-surface, the structure is slave: each structural node b gets ∫ N_b dS
            // and is coupled to the fluid surface shape functions (and normal) at the projection
            use std::collections::BTreeMap;
            let mut area: BTreeMap<usize, f64> = BTreeMap::new();
            for (si, sc) in sf.iter().enumerate() {
                for (xi, wg) in geom::gauss(sc.dim, sx[si].len(), 4) {
                    let fp = geom::eval(sc, &sx[si], xi);
                    let w = weight(m, sc, &fp, wg);
                    for (b, &un) in sc.nodes.iter().enumerate() {
                        *area.entry(un).or_default() += w * fp.n[b];
                    }
                }
            }
            for (un, ab) in area {
                let Some((fi, fxi, dist)) = closest(&ff, &fx, m.nodes[un], size) else { continue };
                if dist > tol {
                    continue;
                }
                let fc = &ff[fi];
                let fp = geom::eval(fc, &fx[fi], fxi);
                for (a, &pn) in fc.nodes.iter().enumerate() {
                    if fp.n[a] == 0.0 {
                        continue;
                    }
                    for (dir, &nd) in fp.normal.iter().enumerate().take(fc.dim) {
                        let v = ab * fp.n[a] * nd;
                        if v != 0.0 {
                            entries.push((pn, un, dir, v));
                        }
                    }
                }
                let mut g: Vec<usize> = fc.nodes.clone();
                g.push(un);
                groups.push(g);
            }
        }
        if entries.is_empty() {
            return kw.err(format!("{name}: no part of the acoustic surface lies within the position tolerance of the structural surface"));
        }
        out.push(Coupling { name, entries, groups, fluid_u });
    }
    Ok(out)
}
