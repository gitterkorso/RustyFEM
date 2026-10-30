//! Thermal contact between surfaces: *GAP CONDUCTANCE and *GAP RADIATION in the
//! *SURFACE INTERACTION referenced by a *CONTACT PAIR.
//!
//! Node-to-surface discretization (default for *CONTACT PAIR): each
//! slave node A with the slave surface nodal area A_A = ∫N_A dA is projected
//! onto the closest master facet (the shape functions M_C at the projection).
//! The gap distance d is the distance along the master surface normal
//! (positive when the surfaces are separated, 0 at penetration). The heat
//! flux from master to slave:
//!
//! - conduction: q = k(d) (T_M − T_A), T_M = Σ M_C T_C;
//! - radiation: q = F(d) σ ((T_M − T_Z)⁴ − (T_A − T_Z)⁴) with
//!   F = f(d) / (1/ε_A + 1/ε_B − 1) (f = the view factor from the table).
//!
//! The slave node gets +A_A q and the master nodes −M_C A_A q. The tables k(d)
//! and f(d) are interpolated linearly and held constant outside (one line =
//! constant). The heat transfer step type (`steps/heat.rs`) calls `build` and
//! `contribute`; in mechanics the pair contributes as usual via `contact.rs`.

use super::geom::{self, Facet};
use super::{Interaction, InteractionFactory};
use crate::elements::common::{dot, sub};
use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::{fail, FemError, Result};

/// The registry only provides the couplings (slave node–master nodes) to the
/// matrix structure; the fluxes are computed by the heat transfer via `build`.
pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CONTACT PAIR"], build: links }];

/// The matrix structure for thermal contact.
struct ThermalLink {
    groups: Vec<Vec<usize>>,
}

impl Interaction for ThermalLink {
    fn name(&self) -> String {
        "thermal contact".into()
    }
    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.groups.clone()
    }
}

fn links(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let pairs = build_pair(m, kw)?;
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let groups = pairs.iter().flat_map(|p| p.nodes.iter().map(|s| std::iter::once(s.node).chain(s.master.iter().map(|c| c.0)).collect())).collect();
    Ok(vec![Box::new(ThermalLink { groups })])
}

/// A slave node with its projection.
#[derive(Debug, Clone)]
struct SlaveNode {
    node: usize,
    area: f64,
    /// Master nodes and shape functions at the projection.
    master: Vec<(usize, f64)>,
    /// Distance in the reference configuration along the master normal.
    gap0: f64,
    /// The master surface's outward normal at the projection.
    normal: [f64; 3],
    /// Projection point in the reference configuration.
    xm: [f64; 3],
}

/// Thermal contact for a surface pair.
#[derive(Debug, Clone)]
pub struct ThermalPair {
    nodes: Vec<SlaveNode>,
    /// (d, k) sorted by d.
    conductance: Vec<(f64, f64)>,
    /// (d, f) and 1/(1/ε_A + 1/ε_B − 1).
    radiation: Option<(Vec<(f64, f64)>, f64)>,
}

fn table(v: &[(f64, f64)], d: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    if d <= v[0].0 {
        return v[0].1;
    }
    for w in v.windows(2) {
        if d <= w[1].0 {
            let s = (d - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + s * (w[1].1 - w[0].1);
        }
    }
    v[v.len() - 1].1
}

fn floats(kw: &RawKeyword, d: &crate::fem::model::DataLine) -> Result<Vec<f64>> {
    d.fields
        .iter()
        .filter(|f| !f.trim().is_empty())
        .map(|f| parse_f64(f).ok_or_else(|| FemError(format!("{} line {}: '{f}' is not a number", kw.file, d.line))))
        .collect()
}

/// Suboptions of a *SURFACE INTERACTION (the keywords directly after it).
fn interaction_options<'a>(m: &'a FeModel, name: &str) -> Vec<&'a RawKeyword> {
    const SUB: &[&str] = &["FRICTION", "SURFACE BEHAVIOR", "GAP CONDUCTANCE", "GAP RADIATION", "GAP HEAT GENERATION", "CONTACT DAMPING"];
    let Some(i) = m.extra.iter().position(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(name)) else {
        return Vec::new();
    };
    m.extra.iter().skip(i + 1).take_while(|k| SUB.contains(&k.name.as_str())).collect()
}

/// A *CONTACT PAIR whose surfaces all lie on heat transfer elements (no
/// mechanical contact; only thermal contact via `build`).
pub fn only_heat(m: &FeModel, kw: &RawKeyword) -> bool {
    let mut any = false;
    for d in kw.records() {
        for f in d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let Some(s) = m.surfaces.get(&f.to_uppercase()) else { return false };
            for (e, _) in &s.faces {
                if m.elements[*e].etype.family() != crate::elements::Family::Heat {
                    return false;
                }
                any = true;
            }
        }
    }
    any
}

/// All pairs with thermal properties in the model.
pub fn build(m: &FeModel) -> Result<Vec<ThermalPair>> {
    let mut out = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT PAIR") {
        out.extend(build_pair(m, kw)?);
    }
    Ok(out)
}

/// Thermal pairs for a *CONTACT PAIR (empty if the interaction has neither
/// *GAP CONDUCTANCE nor *GAP RADIATION).
fn build_pair(m: &FeModel, kw: &RawKeyword) -> Result<Vec<ThermalPair>> {
    let mut out = Vec::new();
    {
        let Some(iname) = kw.param_upper("INTERACTION") else { return Ok(out) };
        let opts = interaction_options(m, &iname);
        let mut conductance = Vec::new();
        let mut radiation = None;
        for o in &opts {
            match o.name.as_str() {
                "GAP CONDUCTANCE" => {
                    for p in ["PRESSURE", "USER", "DEPENDENCIES"] {
                        if o.has(p) {
                            return o.err(format!("{p} is not supported yet (only k versus gap distance)"));
                        }
                    }
                    for d in o.records() {
                        let v = floats(o, &d)?;
                        if v.is_empty() {
                            continue;
                        }
                        conductance.push((v.get(1).copied().unwrap_or(0.0), v[0]));
                    }
                }
                "GAP RADIATION" => {
                    let rec = o.records();
                    let Some(first) = rec.first() else { return o.err("write 'ε slave, ε master'") };
                    let e = floats(o, first)?;
                    if e.len() < 2 || e[0] <= 0.0 || e[1] <= 0.0 {
                        return o.err("write 'ε slave, ε master' (both > 0)");
                    }
                    let mut view = Vec::new();
                    for d in &rec[1..] {
                        let v = floats(o, d)?;
                        if !v.is_empty() {
                            view.push((v.get(1).copied().unwrap_or(0.0), v[0]));
                        }
                    }
                    if view.is_empty() {
                        view.push((0.0, 1.0));
                    }
                    view.sort_by(|a, b| a.0.total_cmp(&b.0));
                    radiation = Some((view, 1.0 / (1.0 / e[0] + 1.0 / e[1] - 1.0)));
                }
                "GAP HEAT GENERATION" => return o.err("*GAP HEAT GENERATION is not supported yet"),
                _ => {}
            }
        }
        if conductance.is_empty() && radiation.is_none() {
            return Ok(out);
        }
        conductance.sort_by(|a, b| a.0.total_cmp(&b.0));
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if f.len() < 2 {
                return kw.err_at(d.line, "write 'slave surface, master surface'");
            }
            let ss = geom::surface(m, kw, f[0])?;
            let ms = geom::surface(m, kw, f[1])?;
            let sf = geom::facets(m, ss, "slave surface")?;
            let mf = geom::facets(m, ms, "master surface")?;
            out.push(ThermalPair { nodes: slave_nodes(m, &sf, &mf)?, conductance: conductance.clone(), radiation: radiation.clone() });
        }
    }
    Ok(out)
}

fn slave_nodes(m: &FeModel, sf: &[Facet], mf: &[Facet]) -> Result<Vec<SlaveNode>> {
    let mut area: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for f in sf {
        let x = geom::coords(m, f, None);
        for (k, a) in geom::nodal_areas(f, &x).into_iter().enumerate() {
            *area.entry(f.nodes[k]).or_default() += a;
        }
    }
    let mut out = Vec::new();
    for (&n, &a) in &area {
        let p = m.nodes[n];
        // closest master facet (preferably with the projection inside the facet)
        let mut best: Option<(bool, f64, usize, [f64; 2])> = None;
        for (i, f) in mf.iter().enumerate() {
            let x = geom::coords(m, f, None);
            let (xi, dist, inside) = geom::project(f, &x, p);
            let better = match best {
                None => true,
                Some((bi, bd, _, _)) => (inside && !bi) || (inside == bi && dist < bd - 1e-14),
            };
            if better {
                best = Some((inside, dist, i, xi));
            }
        }
        let Some((_, _, i, xi)) = best else { return fail("thermal contact: the master surface is empty") };
        let f = &mf[i];
        let x = geom::coords(m, f, None);
        let fp = geom::eval(f, &x, xi);
        let gap0 = dot(sub(p, fp.x), fp.normal);
        out.push(SlaveNode {
            node: n,
            area: a,
            master: f.nodes.iter().copied().zip(fp.n.iter().copied()).filter(|(_, w)| w.abs() > 1e-14).collect(),
            gap0,
            normal: fp.normal,
            xm: fp.x,
        });
    }
    Ok(out)
}

impl ThermalPair {
    /// The flux per area into each slave node, (node, q).
    pub fn surface_flux(&self, u: &[NodeVec], ts: usize, disp: bool, abs_zero: f64, sigma: f64) -> Vec<(usize, f64)> {
        let mut f = vec![[0.0; crate::fem::dof::SLOTS]; u.len()];
        let _ = self.contribute(u, ts, disp, abs_zero, sigma, &mut f, false);
        self.nodes.iter().map(|s| (s.node, -f[s.node][ts] / s.area)).collect()
    }

    /// Contribution to the internal heat flux (slot `ts`) and tangent (node list,
    /// matrix) at the temperatures in `u`. With `disp` the gap distance is
    /// computed in the current configuration.
    pub fn contribute(
        &self,
        u: &[NodeVec],
        ts: usize,
        disp: bool,
        abs_zero: f64,
        sigma: f64,
        fint: &mut [NodeVec],
        tangent: bool,
    ) -> Result<Vec<(Vec<usize>, Vec<f64>)>> {
        if self.radiation.is_some() && sigma == 0.0 {
            return fail("*GAP RADIATION requires *PHYSICAL CONSTANTS, ABSOLUTE ZERO=..., STEFAN BOLTZMANN=...");
        }
        let mut mats = Vec::new();
        for s in &self.nodes {
            let mut d = s.gap0;
            if disp {
                let mut um = [0.0; 3];
                for &(c, w) in &s.master {
                    for k in 0..3 {
                        um[k] += w * u[c][k];
                    }
                }
                let us = u[s.node];
                let xs = [0, 1, 2].map(|k| s.xm[k] + s.gap0 * s.normal[k] + us[k] - um[k] - s.xm[k]);
                d = dot(xs, s.normal);
            }
            let d = d.max(0.0);
            let ta = u[s.node][ts];
            let tm: f64 = s.master.iter().map(|&(c, w)| w * u[c][ts]).sum();
            // q = flux into the slave per area and the derivatives d/dT_A, d/dT_M
            let k = table(&self.conductance, d);
            let mut q = k * (tm - ta);
            let mut dqa = -k;
            let mut dqm = k;
            if let Some((view, fe)) = &self.radiation {
                let c = sigma * fe * table(view, d);
                let a = ta - abs_zero;
                let b = tm - abs_zero;
                q += c * (b.powi(4) - a.powi(4));
                dqa -= 4.0 * c * a.powi(3);
                dqm += 4.0 * c * b.powi(3);
            }
            // internal flux = −applied flux
            fint[s.node][ts] -= s.area * q;
            for &(c, w) in &s.master {
                fint[c][ts] += w * s.area * q;
            }
            if tangent {
                let mut nodes = vec![s.node];
                nodes.extend(s.master.iter().map(|p| p.0));
                let coef: Vec<f64> = std::iter::once(-1.0).chain(s.master.iter().map(|p| p.1)).collect();
                let dq: Vec<f64> = std::iter::once(dqa).chain(s.master.iter().map(|p| dqm * p.1)).collect();
                let nd = nodes.len();
                let mut ke = vec![0.0; nd * nd];
                for i in 0..nd {
                    for j in 0..nd {
                        ke[i * nd + j] = coef[i] * s.area * dq[j];
                    }
                }
                mats.push((nodes, ke));
            }
        }
        Ok(mats)
    }
}
