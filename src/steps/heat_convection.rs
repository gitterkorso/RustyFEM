//! Påtvingad konvektion i *HEAT TRANSFER: DCC-elementen
//! (`elements/heat_convection.rs`) och *MASS FLOW RATE.
//!
//! *MASS FLOW RATE (stegdata, OP=MOD/NEW, ärvs till senare steg): massflöde
//! per area (ρv₁, ρv₂, ρv₃) per nod, för DCC1D2 hela massflödet genom
//! tvärsnittet (positivt från nod 1 mot nod 2). Värdena rampas från
//! föregående stegs värde (stationär analys, standard) eller gäller direkt
//! (transient analys), enligt AMPLITUDE= på *STEP som för övriga laster.
//! Startvärdet är 0 (*INITIAL CONDITIONS, TYPE=MASS FLOW RATE stöds inte).
//!
//! Elementens bidrag (osymmetriska; lösaren `unsymmetric` väljs automatiskt)
//! läggs till i värmebalansen i `steps/heat.rs`:
//! r = C(θ₁ − θ₀)/Δt + ½K(θ₁ + θ₀) (transient), r = Kθ (stationär).
//! HFL = −k∇T skrivs i elementens integrationspunkter (transient vid
//! inkrementets mitt).

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

/// Inga egna steg-typer; används av `steps/heat.rs`.
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
    /// Massflöde per nod vid stegets början och slut (komponenter 1–3).
    flow0: Vec<[f64; 3]>,
    flow1: Vec<[f64; 3]>,
    ramp: bool,
    /// Diagonal Galerkindel i kapacitetsmatrisen.
    lumped: bool,
    /// Temperaturerna i början av senaste transienta inkrementet (HFL tas vid
    /// inkrementets mitt, ½(θ₀ + θ₁)).
    start: std::cell::RefCell<Option<Vec<f64>>>,
}

/// Har modellen DCC-element?
pub fn has_convection(model: &FeModel) -> bool {
    model.elements.iter().any(|e| hc::lookup(e.etype.name()).is_some())
}

/// Massflödena (per nod) efter stegen 0..=index, och värdet vid början av steget `index`.
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
                    return kw.err(format!("{p}= stöds inte ännu"));
                }
            }
            if kw.param_upper("OP").as_deref() == Some("NEW") {
                cur = vec![[0.0; 3]; nn];
            }
            for d in &kw.data {
                let f: Vec<&str> = d.fields.iter().map(|s| s.as_str()).collect();
                if f.len() < 2 {
                    return kw.err_at(d.line, "skriv 'nod/nodset, massflöde 1, 2, 3'");
                }
                let Some(nodes) = super::heat::resolve_nodes(model, f[0]) else {
                    return kw.err_at(d.line, format!("noden/nodsetet '{}' finns inte", f[0].trim()));
                };
                let mut v = [0.0; 3];
                for (c, s) in f.iter().skip(1).take(3).enumerate() {
                    v[c] = parse_f64(s).ok_or_else(|| FemError(format!("{} rad {}: '{}' är inget tal", kw.file, d.line, s.trim())))?;
                }
                for n in nodes {
                    cur[n] = v;
                }
            }
        }
    }
    Ok((start, cur))
}

/// Konvektionen i steget `index` (None om modellen saknar DCC-element).
pub fn for_step(p: &Prepared, index: usize, loads: &StepLoads, transient: bool) -> Result<Option<Convection>> {
    let model = p.model;
    if !has_convection(model) {
        return Ok(None);
    }
    let mut elems = Vec::new();
    for (e, el) in model.elements.iter().enumerate() {
        let Some(ce) = hc::lookup(el.etype.name()) else { continue };
        if ce.dispersion {
            return fail(format!("{}: {} (dispersionskontroll) stöds inte ännu", model.elem_name(e), ce.name));
        }
        let Some(si) = el.section else { return fail(format!("{}: saknar *SOLID SECTION", model.elem_name(e))) };
        let sec = &model.sections[si];
        let Some(mi) = sec.material_index else { return fail(format!("{}: tvärsnittet saknar material", model.elem_name(e))) };
        let props = HeatProps::from_def(&model.materials[mi])?;
        if props.cond.is_none() {
            return fail(format!("material {}: *CONDUCTIVITY saknas", props.name));
        }
        if transient && (props.density <= 0.0 || props.capacity().is_none()) {
            return fail(format!("material {}: *DENSITY och *SPECIFIC HEAT krävs för transient konvektion", props.name));
        }
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
        elems.push(CElem { e, el: ce, x, section: sec.solid_thickness(), props });
    }
    // fördelade laster på DCC-element stöds inte (de skulle annars ignoreras)
    for (l, _) in &loads.dload {
        if hc::lookup(model.elements[l.elem].etype.name()).is_some() {
            return fail(format!("{}: lasten {} på ett konvektionselement stöds inte ännu", model.elem_name(l.elem), l.kind));
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
    /// Massflödet per nod vid stegtidsandelen `t`.
    fn flow_at(&self, n: usize, t: f64) -> [f64; 3] {
        if !self.ramp {
            return self.flow1[n];
        }
        [0, 1, 2].map(|c| self.flow0[n][c] + (self.flow1[n][c] - self.flow0[n][c]) * t)
    }

    /// Elementets massflöde per area (ρv) per nod.
    fn g(&self, ce: &CElem, nodes: &[usize], t: f64) -> Vec<[f64; 3]> {
        nodes
            .iter()
            .map(|&n| {
                let m = self.flow_at(n, t);
                if ce.el.is_link() {
                    // hela massflödet längs länken (nod 1 → nod 2) per tvärsnittsarea
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

    /// Lägger till elementens inre flöden i `fint` och tangenten i `mat`.
    /// `t1` = stegtidsandel vid inkrementets slut, `dt` = tidsinkrementet.
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

    /// HFL i DCC-elementens integrationspunkter (ersätter de tomma posterna).
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
