//! Area explicit_adv: coupled Eulerian–Lagrangian contact (CEL) in general
//! contact (`*CONTACT` with `*CONTACT INCLUSIONS, ALL EXTERIOR`), where
//! the Eulerian material's surface is included automatically.
//!
//! The outer surfaces of the Lagrangian solids are slaves against the Eulerian material with a
//! penalty method (immersed boundary). The slave points are the surface nodes and, for
//! quadrilaterals, the edge midpoints and the center point (the values are interpolated with
//! the surface shape functions), so that the material does not flow past between the nodes:
//!
//! - the point is located in the Eulerian element that contains it (isoparametric
//!   inverse), the material velocity is interpolated from the Eulerian nodes;
//! - when the element contains material (volume fraction ≥ ½) the
//!   penetration d grows with the relative normal velocity towards the surface,
//!   d ← max(0, d − Δt (v_f − v_s)·n), n = the outward surface normal;
//!   d is reset when the point leaves the material;
//! - the force F = m*/Δt² (κ d + γ Δd) (mass-based penalty stiffness,
//!   `explicit_contact.rs`), m* = the smaller of the point's mass (the surface nodal masses divided by
//!   the number of points per node) and the interpolated Eulerian mass, acts on
//!   the surface nodes along −n and on the Eulerian nodes along +n with the shape functions.
//!
//! Limitations: no shells or rigid surfaces as the Lagrangian surface, no
//! friction between Eulerian material and Lagrangian body, the Lagrangian body
//! does not displace the material in elements it covers initially (they must
//! be empty).

use super::euler::Euler;
use super::{Eval, Sink};
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::model::Result;
use std::collections::HashMap;

/// A slave point: the surface nodes with weights and the outward normal.
struct Slave {
    nodes: Vec<(usize, f64)>,
    normal: [f64; 3],
    /// The point's share of the nodal masses.
    share: f64,
}

pub struct Cel {
    slaves: Vec<Slave>,
    /// Penetration per slave point.
    pen: Vec<f64>,
    /// Bounding boxes of the Eulerian elements (fixed mesh).
    boxes: Vec<([f64; 3], [f64; 3])>,
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    (l > 0.0).then(|| a.map(|v| v / l))
}

impl Cel {
    /// The contact is built when the model has Eulerian elements, Lagrangian solids and
    /// general contact with ALL EXTERIOR.
    pub fn build(p: &Prepared, eu: &Euler) -> Result<Option<Cel>> {
        let m = p.model;
        let general = m.extra.iter().enumerate().any(|(i, k)| {
            k.name == "CONTACT" && m.extra.iter().skip(i + 1).take_while(|s| s.name.starts_with("CONTACT ") || s.name == "SURFACE PROPERTY ASSIGNMENT").any(|s| s.name == "CONTACT INCLUSIONS" && (s.has("ALL EXTERIOR") || s.data.iter().all(|d| d.fields.iter().all(|f| f.trim().is_empty()))))
        });
        if !general {
            return Ok(None);
        }
        // outer surfaces of Lagrangian solids
        let mut faces: HashMap<Vec<usize>, (usize, usize, usize)> = HashMap::new();
        for (e, el) in m.elements.iter().enumerate() {
            if el.etype.family() != Family::Continuum3D {
                continue;
            }
            for (fi, f) in el.etype.faces().iter().enumerate() {
                let mut key: Vec<usize> = f.iter().map(|&k| el.nodes[k]).collect();
                key.sort_unstable();
                faces.entry(key).and_modify(|x| x.2 += 1).or_insert((e, fi, 1));
            }
        }
        let mut node_normal: HashMap<usize, [f64; 3]> = HashMap::new();
        let mut node_count: HashMap<usize, f64> = HashMap::new();
        let mut slaves = Vec::new();
        for &(e, fi, cnt) in faces.values() {
            if cnt != 1 {
                continue;
            }
            let el = &m.elements[e];
            let f = el.etype.faces()[fi];
            let fnodes: Vec<usize> = f.iter().map(|&k| el.nodes[k]).collect();
            let x: Vec<[f64; 3]> = fnodes.iter().map(|&n| m.nodes[n]).collect();
            let mut a = [0.0; 3];
            for i in 1..x.len() - 1 {
                let c = cross([x[i][0] - x[0][0], x[i][1] - x[0][1], x[i][2] - x[0][2]], [x[i + 1][0] - x[0][0], x[i + 1][1] - x[0][1], x[i + 1][2] - x[0][2]]);
                for k in 0..3 {
                    a[k] += 0.5 * c[k];
                }
            }
            // outward: away from the element centroid
            let xe: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            let ce = crate::elements::common::centroid(&xe);
            let cf = crate::elements::common::centroid(&x);
            if (0..3).map(|k| a[k] * (cf[k] - ce[k])).sum::<f64>() < 0.0 {
                a = a.map(|v| -v);
            }
            for &n in &fnodes {
                let s = node_normal.entry(n).or_insert([0.0; 3]);
                for c in 0..3 {
                    s[c] += a[c];
                }
            }
            let Some(nrm) = unit(a) else { continue };
            if fnodes.len() == 4 {
                // edge midpoints and center point with bilinear weights
                for (r, s) in [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, 0.0)] {
                    let nat = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
                    let w: Vec<(usize, f64)> = (0..4).map(|k| (fnodes[k], 0.25 * (1.0 + nat[k].0 * r) * (1.0 + nat[k].1 * s))).filter(|x| x.1 > 0.0).collect();
                    for &(n, wk) in &w {
                        *node_count.entry(n).or_insert(0.0) += wk;
                    }
                    slaves.push(Slave { nodes: w, normal: nrm, share: 1.0 });
                }
            }
        }
        let mut corner: Vec<(usize, [f64; 3])> = node_normal.into_iter().filter_map(|(n, a)| unit(a).map(|u| (n, u))).collect();
        corner.sort_by_key(|s| s.0);
        for (n, u) in corner {
            *node_count.entry(n).or_insert(0.0) += 1.0;
            slaves.push(Slave { nodes: vec![(n, 1.0)], normal: u, share: 1.0 });
        }
        if slaves.is_empty() {
            return Ok(None);
        }
        // each node's mass is divided among its slave points
        for s in slaves.iter_mut() {
            let c: f64 = s.nodes.iter().map(|&(n, w)| w / node_count[&n].max(1e-300)).sum::<f64>();
            s.share = c / s.nodes.iter().map(|x| x.1).sum::<f64>().max(1e-300);
        }
        let boxes = eu
            .elems
            .iter()
            .map(|el| {
                let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
                let lo: [f64; 3] = std::array::from_fn(|c| x.iter().map(|p| p[c]).fold(f64::INFINITY, f64::min));
                let hi: [f64; 3] = std::array::from_fn(|c| x.iter().map(|p| p[c]).fold(f64::NEG_INFINITY, f64::max));
                (lo, hi)
            })
            .collect();
        let n = slaves.len();
        Ok(Some(Cel { slaves, pen: vec![0.0; n], boxes }))
    }

    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink, eu: &Euler) -> Result<()> {
        if ev.dt <= 0.0 {
            return Ok(());
        }
        let m = p.model;
        let dt = ev.dt;
        for (s, sl) in self.slaves.iter().enumerate() {
            let x: [f64; 3] = std::array::from_fn(|c| sl.nodes.iter().map(|&(n, w)| w * (m.nodes[n][c] + ev.u[n][c])).sum());
            let vs: [f64; 3] = std::array::from_fn(|c| sl.nodes.iter().map(|&(n, w)| w * ev.v[n][c]).sum());
            let mut found = None;
            for (k, (lo, hi)) in self.boxes.iter().enumerate() {
                if (0..3).all(|c| x[c] >= lo[c] - 1e-12 && x[c] <= hi[c] + 1e-12) {
                    let el = &eu.elems[k];
                    let xe: Vec<[f64; 3]> = el.nodes.iter().map(|&q| m.nodes[q]).collect();
                    if let Some(xi) = super::conversion::inverse(&xe, x) {
                        if xi.iter().all(|v| v.abs() <= 1.0 + 1e-9) {
                            found = Some((k, xi));
                            break;
                        }
                    }
                }
            }
            let Some((k, xi)) = found else {
                self.pen[s] = 0.0;
                continue;
            };
            let el = &eu.elems[k];
            let phi = eu.material_fraction(k, &out.state_new[el.e]);
            if phi < 0.5 {
                self.pen[s] = 0.0;
                continue;
            }
            let nrm = sl.normal;
            let sh = super::conversion::hex_shape(xi);
            let vf: [f64; 3] = std::array::from_fn(|c| (0..8).map(|a| sh[a] * ev.v[el.nodes[a]][c]).sum());
            let rel: f64 = (0..3).map(|c| (vf[c] - vs[c]) * nrm[c]).sum();
            let old = self.pen[s];
            let d = (old - dt * rel).max(0.0);
            self.pen[s] = d;
            if d <= 0.0 {
                continue;
            }
            let mf: f64 = (0..8).map(|a| sh[a] * out.mass.diag[el.nodes[a]][0]).sum();
            let ms: f64 = sl.share * sl.nodes.iter().map(|&(n, w)| w * out.mass.diag[n][0]).sum::<f64>();
            let mm = if ms > 0.0 && mf > 0.0 { ms.min(mf) } else { ms.max(mf) };
            let f = mm / (dt * dt) * (crate::steps::explicit::contact::KAPPA * d + crate::steps::explicit::contact::GAMMA * (d - old));
            for c in 0..3 {
                for &(n, w) in &sl.nodes {
                    out.fcon[n][c] -= w * f * nrm[c];
                }
                for a in 0..8 {
                    out.fcon[el.nodes[a]][c] += sh[a] * f * nrm[c];
                }
            }
        }
        Ok(())
    }
}
