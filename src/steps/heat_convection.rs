//! Forced convection in *HEAT TRANSFER: the DCC elements
//! (`elements/heat_convection.rs`) and *MASS FLOW RATE.
//!
//! *MASS FLOW RATE (step data, OP=MOD/NEW, inherited by later steps): mass flow
//! per area (ρv₁, ρv₂, ρv₃) per node, for DCC1D2 the whole mass flow through
//! the cross-section (positive from node 1 towards node 2). The values are ramped from
//! the value of the previous step (steady-state analysis, default) or apply directly
//! (transient analysis), according to AMPLITUDE= on *STEP as for other loads.
//! The starting value is 0 (*INITIAL CONDITIONS, TYPE=MASS FLOW RATE is not supported).
//!
//! The element contributions (unsymmetric; the solver `unsymmetric` is selected automatically)
//! are added to the heat balance in `steps/heat.rs`:
//! r = C(θ₁ − θ₀)/Δt + ½K(θ₁ + θ₀) (transient), r = Kθ (steady-state).
//! HFL = −k∇T is written at the element integration points (transient at the
//! midpoint of the increment).

use super::api::StepType;
use crate::elements::heat_convection::{self as hc, ConvElem};
use crate::elements::IpField;
use crate::fem::assembly::{DofMap, Prepared, NONE};
use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, FeModel};
use crate::fem::state::StepLoads;
use crate::materials::heat::HeatProps;
use crate::model::{fail, FemError, Result};
use crate::solver::SystemMatrix;

/// No step types of its own; used by `steps/heat.rs`.
pub static STEP_TYPES: &[&dyn StepType] = &[];

const TS: usize = 7;

struct CElem {
    e: usize,
    el: &'static ConvElem,
    x: Vec<[f64; 3]>,
    section: f64,
    props: HeatProps,
}

pub struct Convection {
    elems: Vec<CElem>,
    /// Mass flow per node at the start and end of the step (components 1–3).
    flow0: Vec<[f64; 3]>,
    flow1: Vec<[f64; 3]>,
    ramp: bool,
    /// Diagonal Galerkin part of the capacity matrix.
    lumped: bool,
    /// The temperatures at the start of the latest transient increment (HFL is taken at
    /// the midpoint of the increment, ½(θ₀ + θ₁)).
    start: std::cell::RefCell<Option<Vec<f64>>>,
}

/// Does the model have DCC elements?
pub fn has_convection(model: &FeModel) -> bool {
    model.elements.iter().any(|e| hc::lookup(e.etype.name()).is_some())
}

/// The mass flows (per node) after steps 0..=index, and the value at the start of step `index`.
fn mass_flow(model: &FeModel, index: usize) -> Result<(Vec<[f64; 3]>, Vec<[f64; 3]>)> {
    let nn = model.nodes.len();
    let mut cur = vec![[0.0; 3]; nn];
    let mut start = cur.clone();
    for (si, step) in model.steps.iter().enumerate().take(index + 1) {
        if si == index {
            start = cur.clone();
        }
        for kw in step.extra.iter().filter(|k| k.name == "MASS FLOW RATE") {
            for p in ["USER", "INPUT", "AMPLITUDE"] {
                if kw.has(p) {
                    return kw.err(format!("{p}= is not supported yet"));
                }
            }
            if kw.param_upper("OP").as_deref() == Some("NEW") {
                cur = vec![[0.0; 3]; nn];
            }
            for d in &kw.data {
                let f: Vec<&str> = d.fields.iter().map(|s| s.as_str()).collect();
                if f.len() < 2 {
                    return kw.err_at(d.line, "write 'node/node set, mass flow 1, 2, 3'");
                }
                let Some(nodes) = super::heat::resolve_nodes(model, f[0]) else {
                    return kw.err_at(d.line, format!("the node/node set '{}' does not exist", f[0].trim()));
                };
                let mut v = [0.0; 3];
                for (c, s) in f.iter().skip(1).take(3).enumerate() {
                    v[c] = parse_f64(s).ok_or_else(|| FemError(format!("{} line {}: '{}' is not a number", kw.file, d.line, s.trim())))?;
                }
                for n in nodes {
                    cur[n] = v;
                }
            }
        }
    }
    Ok((start, cur))
}

/// The convection in step `index` (None if the model has no DCC elements).
pub fn for_step(p: &Prepared, index: usize, loads: &StepLoads, transient: bool) -> Result<Option<Convection>> {
    let model = p.model;
    if !has_convection(model) {
        return Ok(None);
    }
    let mut elems = Vec::new();
    for (e, el) in model.elements.iter().enumerate() {
        let Some(ce) = hc::lookup(el.etype.name()) else { continue };
        if ce.dispersion {
            return fail(format!("{}: {} (dispersion control) is not supported yet", model.elem_name(e), ce.name));
        }
        let Some(si) = el.section else { return fail(format!("{}: missing *SOLID SECTION", model.elem_name(e))) };
        let sec = &model.sections[si];
        let Some(mi) = sec.material_index else { return fail(format!("{}: the section has no material", model.elem_name(e))) };
        let props = HeatProps::from_def(&model.materials[mi])?;
        if props.cond.is_none() {
            return fail(format!("material {}: *CONDUCTIVITY is missing", props.name));
        }
        if transient && (props.density <= 0.0 || props.capacity().is_none()) {
            return fail(format!("material {}: *DENSITY and *SPECIFIC HEAT are required for transient convection", props.name));
        }
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
        elems.push(CElem { e, el: ce, x, section: sec.solid_thickness(), props });
    }
    // distributed loads on DCC elements are not supported (they would otherwise be ignored)
    for (l, _) in &loads.dload {
        if hc::lookup(model.elements[l.elem].etype.name()).is_some() {
            return fail(format!("{}: the load {} on a convection element is not supported yet", model.elem_name(l.elem), l.kind));
        }
    }
    let (flow0, flow1) = mass_flow(model, index)?;
    let step = &model.steps[index];
    let ramp = match step.header.param_upper("AMPLITUDE").as_deref() {
        Some("RAMP") => true,
        Some("STEP") => false,
        _ => !transient,
    };
    let lumped = std::env::var("RUSTYFEM_DCC_LUMPED").is_ok();
    Ok(Some(Convection { elems, flow0, flow1, ramp, lumped, start: std::cell::RefCell::new(None) }))
}

impl Convection {
    /// The mass flow per node at the step time fraction `t`.
    fn flow_at(&self, n: usize, t: f64) -> [f64; 3] {
        if !self.ramp {
            return self.flow1[n];
        }
        [0, 1, 2].map(|c| self.flow0[n][c] + (self.flow1[n][c] - self.flow0[n][c]) * t)
    }

    /// The element's mass flow per area (ρv) per node.
    fn g(&self, ce: &CElem, nodes: &[usize], t: f64) -> Vec<[f64; 3]> {
        nodes
            .iter()
            .map(|&n| {
                let m = self.flow_at(n, t);
                if ce.el.is_link() {
                    // the whole mass flow along the link (node 1 → node 2) per cross-sectional area
                    let d = [ce.x[1][0] - ce.x[0][0], ce.x[1][1] - ce.x[0][1], ce.x[1][2] - ce.x[0][2]];
                    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    let area = if ce.el.kind == hc::Kind::LinkAx { ce.section * 2.0 * std::f64::consts::PI * 0.5 * (ce.x[0][0] + ce.x[1][0]) } else { ce.section };
                    [0, 1, 2].map(|c| m[0] / area * d[c] / l)
                } else {
                    m
                }
            })
            .collect()
    }

    fn elem_matrices(&self, model: &FeModel, ce: &CElem, u: &[NodeVec], t: f64, transient: bool) -> Result<hc::ConvMatrices> {
        let nodes = &model.elements[ce.e].nodes;
        let tm: f64 = nodes.iter().map(|&n| u[n][TS]).sum::<f64>() / nodes.len() as f64;
        let (k, _) = ce.props.conductivity(tm)?;
        let c = ce.props.capacity().map_or(0.0, |c| c.eval(tm).0[0]);
        let g = self.g(ce, nodes, t);
        ce.el.matrices(&ce.x, ce.section, k, ce.props.density, c, &g, transient, self.lumped).map_err(|err| FemError(format!("{}: {}", model.elem_name(ce.e), err.0)))
    }

    /// Adds the element internal fluxes to `fint` and the tangent to `mat`.
    /// `t1` = step time fraction at the end of the increment, `dt` = the time increment.
    #[allow(clippy::too_many_arguments)]
    pub fn contribute(&self, model: &FeModel, u: &[NodeVec], u0: &[NodeVec], t1: f64, dt: f64, transient: bool, dm: &DofMap, fint: &mut [NodeVec], mut mat: Option<&mut dyn SystemMatrix>) -> Result<()> {
        *self.start.borrow_mut() = if transient && dt > 0.0 { Some(u0.iter().map(|v| v[TS]).collect()) } else { None };
        for ce in &self.elems {
            let nodes = &model.elements[ce.e].nodes;
            let nn = nodes.len();
            let m = self.elem_matrices(model, ce, u, t1, transient)?;
            let th1: Vec<f64> = nodes.iter().map(|&n| u[n][TS]).collect();
            let th0: Vec<f64> = nodes.iter().map(|&n| u0[n][TS]).collect();
            let mut ke = vec![0.0; nn * nn];
            for a in 0..nn {
                let mut r = 0.0;
                for b in 0..nn {
                    if transient {
                        r += m.c[a * nn + b] * (th1[b] - th0[b]) / dt + 0.5 * m.k[a * nn + b] * (th1[b] + th0[b]);
                        ke[a * nn + b] = m.c[a * nn + b] / dt + 0.5 * m.k[a * nn + b];
                    } else {
                        r += m.k[a * nn + b] * th1[b];
                        ke[a * nn + b] = m.k[a * nn + b];
                    }
                }
                fint[nodes[a]][TS] += r;
            }
            if let Some(mt) = mat.as_deref_mut() {
                for a in 0..nn {
                    let ra = dm.eq[nodes[a]][TS];
                    if ra == NONE {
                        continue;
                    }
                    for b in 0..nn {
                        let cb = dm.eq[nodes[b]][TS];
                        if cb != NONE && ke[a * nn + b] != 0.0 {
                            mt.add_unsym(ra, cb, ke[a * nn + b]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// HFL at the integration points of the DCC elements (replaces the empty entries).
    pub fn output(&self, model: &FeModel, u: &[NodeVec], out: &mut [Vec<IpField>]) -> Result<()> {
        let start = self.start.borrow();
        let temp = |n: usize| match start.as_ref() {
            Some(s0) => 0.5 * (s0[n] + u[n][TS]),
            None => u[n][TS],
        };
        for ce in &self.elems {
            let nodes = &model.elements[ce.e].nodes;
            let m = self.elem_matrices(model, ce, u, 1.0, false)?;
            let dim = if ce.el.is_link() || ce.el.kind == hc::Kind::Hex { 3 } else { 2 };
            let labels: &[&str] = if dim == 2 { &["HFL1", "HFL2"] } else { &["HFL1", "HFL2", "HFL3"] };
            let mut f = IpField::new("HFL", labels);
            for (q, fl) in m.flux.iter().enumerate() {
                let v: Vec<f64> = (0..dim).map(|r| fl[r].iter().zip(nodes).map(|(a, &n)| a * temp(n)).sum()).collect();
                f.points.push((q + 1, 0, v));
            }
            out[ce.e] = vec![f];
        }
        Ok(())
    }
}
