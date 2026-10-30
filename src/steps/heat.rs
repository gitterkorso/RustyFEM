//! *HEAT TRANSFER: steady-state (STEADY STATE) and transient heat conduction with NT11.
//!
//! The heat balance for the free temperatures T is solved with Newton–Raphson
//! (`newton.rs`):
//!
//! R(T) = Q_ext − Q_conduction(T) − Q_film/radiation(T) − C(T)/Δt = 0
//!
//! - conduction, capacity and surface fluxes are computed by the elements in `elements/heat.rs`;
//! - transient: backward Euler (implicit), capacity term ρ c(T) (T − Tₙ)/Δt
//!   with c at the end of the increment;
//! - steady state: no capacity term, the step time is only a load parameter.
//!
//! Incrementation: data line `Δt0, time period, Δtmin, Δtmax` and
//! the parameter DELTMX=. Transient without DELTMX gives fixed increments Δt0. With
//! DELTMX the increment is chosen automatically so that the largest temperature change per
//! increment stays below DELTMX (see `run_deltmx`).
//! Steady-state analysis uses the Newton driver's automatic incrementation.
//!
//! Loads: *CFLUX (point flux on DOF 11), *DFLUX BF/Sn, *FILM Fn,
//! *RADIATE Rn (requires *PHYSICAL CONSTANTS) and *BOUNDARY on DOF 11.
//! Initial temperatures from *INITIAL CONDITIONS, TYPE=TEMPERATURE.
//! Default amplitude: RAMP for STEADY STATE, STEP for transient.
//! AMPLITUDE= on *BOUNDARY, *CFLUX, *DFLUX, *FILM (T∞) and FILM AMPLITUDE= (h);
//! transient uses, for the *CFLUX/*DFLUX fluxes, the mean of the amplitude values
//! at the start and end of the increment. *FILM with *FILM PROPERTY
//! (h(T) at the surface temperature), *LATENT HEAT (`materials/heat.rs`), thermal
//! contact *GAP CONDUCTANCE/*GAP RADIATION (`interactions/thermal_contact.rs`)
//! and *CONTROLS, PARAMETERS=FIELD (R_n, C_n).
//!
//! Output per frame: NT11, RFL11 (reaction flux) and HFL (heat flux −k∇T)
//! at the integration points.

use super::api::{StepCtx, StepType};
use super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::elements::common::Ip;
use crate::elements::heat::{self as hel, HeatElem};
use crate::elements::{Family, IpField};
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::model::{FeModel, StepDef, FLAT_INSTANCE};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::materials::heat::HeatProps;
use crate::model::{fail, FemError, Result};
use crate::solver::{LinearSolver, SystemMatrix};
use std::collections::BTreeMap;

pub struct HeatTransfer;

pub static STEP_TYPES: &[&dyn StepType] = &[&HeatTransfer];

/// Slot for the temperature NT11.
pub const TS: usize = 7;

/// Key in `SolutionState::vectors` for the initial temperatures (slot 7).
pub const INIT_KEY: &str = "NT11 INITIAL";

/// Film and radiation on first-order surfaces are integrated diagonally (each
/// corner gets the weight of the nearest Gauss point), like body flux and capacity.
const FILM_LUMPED: bool = true;

/// Nodes for a reference stored raw in the model data: node number (in
/// the default instance), "INSTANCE.number" or node set.
pub fn resolve_nodes(model: &FeModel, token: &str) -> Option<Vec<usize>> {
    let tok = token.trim();
    let up = tok.to_uppercase();
    if let Some(l) = crate::inp::parse_label(tok) {
        for inst in [FLAT_INSTANCE, crate::fem::model::ASSEMBLY_INSTANCE] {
            if let Some(i) = model.instance_index(inst) {
                if let Some(n) = model.node(i, l) {
                    return Some(vec![n]);
                }
            }
        }
        return None;
    }
    if let Some((a, b)) = up.rsplit_once('.') {
        if let (Some(l), Some(i)) = (crate::inp::parse_label(b), model.instance_index(a)) {
            return model.node(i, l).map(|n| vec![n]);
        }
    }
    model.nsets.get(&up).cloned()
}

/// Initial temperatures from *INITIAL CONDITIONS, TYPE=TEMPERATURE (node → T).
pub fn initial_temperatures(model: &FeModel) -> Result<BTreeMap<usize, f64>> {
    let mut out = BTreeMap::new();
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS") {
        if kw.param_upper("TYPE").as_deref() != Some("TEMPERATURE") {
            continue;
        }
        for p in ["FILE", "USER", "INPUT", "INTERPOLATE"] {
            if kw.has(p) {
                return kw.err(format!("{p}= is not supported yet for TYPE=TEMPERATURE"));
            }
        }
        if kw.has(crate::materials::thermal::RESOLVED) {
            // resolved by `inp/keywords/temperature.rs`: [node index, temperature]
            for d in &kw.data {
                let v = kw.floats(d)?;
                if let (Some(&n), Some(&t)) = (v.first(), v.get(1)) {
                    out.insert(n as usize, t);
                }
            }
            continue;
        }
        for d in &kw.data {
            let f: Vec<&str> = d.fields.iter().map(|s| s.as_str()).filter(|s| !s.trim().is_empty()).collect();
            if f.len() < 2 {
                return kw.err_at(d.line, "write 'node/nset, temperature'");
            }
            let Some(nodes) = resolve_nodes(model, f[0]) else {
                return kw.err_at(d.line, format!("node/node set '{}' does not exist", f[0]));
            };
            let t = crate::fem::model::parse_f64(f[1]).ok_or_else(|| FemError(format!("{} line {}: '{}' is not a number", kw.file, d.line, f[1])))?;
            for n in nodes {
                out.insert(n, t);
            }
        }
    }
    Ok(out)
}

/// Sets the initial temperatures in the state the first time they are needed.
pub fn ensure_initial(model: &FeModel, state: &mut SolutionState) -> Result<()> {
    if state.vectors.contains_key(INIT_KEY) {
        return Ok(());
    }
    let init = initial_temperatures(model)?;
    let mut v = vec![[0.0; SLOTS]; model.nodes.len()];
    for (&n, &t) in &init {
        v[n][TS] = t;
        if !state.bcs.contains_key(&(n, TS)) {
            state.u[n][TS] = t;
        }
    }
    state.vectors.insert(INIT_KEY.into(), v);
    Ok(())
}

/// (absolute zero, Stefan–Boltzmann constant) from *PHYSICAL CONSTANTS.
fn physical_constants(model: &FeModel) -> Result<(Option<f64>, Option<f64>)> {
    let mut z = None;
    let mut s = None;
    for kw in model.extra.iter().filter(|k| k.name == "PHYSICAL CONSTANTS") {
        let num = |key: &str| -> Result<Option<f64>> {
            match kw.param(key) {
                Some(v) => crate::fem::model::parse_f64(v).map(Some).ok_or_else(|| FemError(format!("{}: {key}={v} is not a number", kw.location()))),
                None => Ok(None),
            }
        };
        if let Some(v) = num("ABSOLUTE ZERO")? {
            z = Some(v);
        }
        if let Some(v) = num("STEFAN BOLTZMANN")? {
            s = Some(v);
        }
    }
    Ok((z, s))
}

/// A heat conduction element with its precomputed integration points.
struct HElem {
    el: &'static HeatElem,
    props: usize,
    thickness: f64,
    ips: Vec<Ip>,
    /// Points for capacity and body flux (diagonal for first order).
    cips: Vec<Ip>,
    bips: Vec<Ip>,
}

/// The heat balance for a step as a `NonlinearSystem`.
pub struct HeatSystem<'p, 'm> {
    pub p: &'p Prepared<'m>,
    pub loads: StepLoads,
    elems: Vec<Option<HElem>>,
    props: Vec<HeatProps>,
    pub dm: DofMap,
    mat: Box<dyn SystemMatrix>,
    pub u: Vec<NodeVec>,
    pub u_committed: Vec<NodeVec>,
    pub fint: Vec<NodeVec>,
    pub fext: Vec<NodeVec>,
    pub time0: f64,
    pub period: f64,
    pub t0: f64,
    pub t1: f64,
    pub transient: bool,
    linear: bool,
    abs_zero: f64,
    sigma: f64,
    /// Amplitudes (*AMPLITUDE) when any load or boundary condition has AMPLITUDE=.
    amps: Option<BTreeMap<String, super::dynamic::amplitude::Amplitude>>,
    /// Film coefficients h(T) from *FILM PROPERTY.
    film_props: BTreeMap<String, crate::materials::heat::Table>,
    /// Thermal contact (*GAP CONDUCTANCE, *GAP RADIATION).
    pub gaps: Vec<crate::interactions::thermal_contact::ThermalPair>,
    /// The gap distance follows the displacements in `u` (coupled analysis).
    pub gap_disp: bool,
    /// Extra supplied heat flux per node (e.g. Joule heat from
    /// `steps/thermal_electric.rs`); empty = none.
    pub source: Vec<f64>,
    /// Cavity radiation (`steps/heat_cavity.rs`); None = none.
    pub cavities: Option<super::heat_cavity::Radiation>,
    /// Forced convection (`steps/heat_convection.rs`); None = none.
    pub convection: Option<super::heat_convection::Convection>,
}

/// Adds the element matrix (symmetrized) to the system matrix.
fn add_matrix(k: &mut dyn SystemMatrix, map: &[usize], ke: &[f64]) {
    let nd = map.len();
    for a in 0..nd {
        let ia = map[a];
        if ia == NONE {
            continue;
        }
        for b in 0..nd {
            let ib = map[b];
            if ib != NONE && ia >= ib {
                let v = 0.5 * (ke[a * nd + b] + ke[b * nd + a]);
                if v != 0.0 {
                    k.add(ia, ib, v);
                }
            }
        }
    }
}

impl<'p, 'm> HeatSystem<'p, 'm> {
    pub fn new(p: &'p Prepared<'m>, loads: StepLoads, state: &SolutionState, transient: bool, solver: &dyn LinearSolver, period: f64) -> Result<Self> {
        Self::new_with(p, loads, state, transient, solver, period, &HeatProps::from_def)
    }

    /// Like `new` but with the material properties from `props_of` (e.g.
    /// mass transport, `steps/mass_diffusion.rs`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_with(
        p: &'p Prepared<'m>,
        loads: StepLoads,
        state: &SolutionState,
        transient: bool,
        solver: &dyn LinearSolver,
        period: f64,
        props_of: &dyn Fn(&crate::fem::model::MaterialDef) -> Result<HeatProps>,
    ) -> Result<Self> {
        let model = p.model;
        let mut props: Vec<HeatProps> = Vec::new();
        let mut pidx: BTreeMap<usize, usize> = BTreeMap::new();
        let mut elems = Vec::with_capacity(model.elements.len());
        let mut linear = true;
        for (e, el) in model.elements.iter().enumerate() {
            // forced convection (DCC elements): `steps/heat_convection.rs`
            if crate::elements::heat_convection::lookup(el.etype.name()).is_some() {
                elems.push(None);
                continue;
            }
            let Some(he) = hel::lookup(el.etype.name()).or_else(|| crate::elements::thermal_electric::lookup(el.etype.name())) else {
                if el.etype.family() == Family::Heat {
                    return fail(format!("{}: the heat element is not supported by *HEAT TRANSFER", model.elem_name(e)));
                }
                return fail(format!(
                    "{}: *HEAT TRANSFER requires heat conduction elements (DC2D4, DC3D8, ...); mechanical elements belong to other steps",
                    model.elem_name(e)
                ));
            };
            let Some(si) = el.section else {
                return fail(format!("{}: missing *SOLID SECTION", model.elem_name(e)));
            };
            let sec = &model.sections[si];
            let Some(mi) = sec.material_index else {
                return fail(format!("{}: the section has no material", model.elem_name(e)));
            };
            let pi = match pidx.get(&mi) {
                Some(&i) => i,
                None => {
                    let hp = props_of(&model.materials[mi])?;
                    // UMATHT and HETVAL (user/loads.rs)
                    let (umatht, hetval) = crate::user::loads::heat_user(&model.materials[mi]);
                    if umatht || hetval {
                        linear = false;
                    }
                    if hp.cond.is_none() && !umatht {
                        return fail(format!("material {}: *CONDUCTIVITY is missing", hp.name));
                    }
                    if hp.cond.as_ref().is_some_and(|c| c.1.varies())
                        || (transient && hp.specific_heat.as_ref().is_some_and(|c| c.varies()))
                        || (transient && !hp.latent.is_empty())
                    {
                        linear = false;
                    }
                    props.push(hp);
                    pidx.insert(mi, props.len() - 1);
                    props.len() - 1
                }
            };
            let thickness = sec.solid_thickness();
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            let ips = he.ips(&x, thickness).map_err(wrap)?;
            let bips = he.cap_ips(&x, thickness).map_err(wrap)?;
            let cips = if transient { he.cap_ips(&x, thickness).map_err(wrap)? } else { Vec::new() };
            elems.push(Some(HElem { el: he, props: pi, thickness, ips, cips, bips }));
        }
        if loads.dload.iter().any(|(l, _)| l.kind.starts_with("RAD:")) {
            linear = false;
        }
        let (z, s) = physical_constants(model)?;
        let needs_rad = loads.dload.iter().any(|(l, _)| l.kind.starts_with("RAD:"));
        if needs_rad && (z.is_none() || s.is_none()) {
            return fail("*RADIATE requires *PHYSICAL CONSTANTS, ABSOLUTE ZERO=..., STEFAN BOLTZMANN=...");
        }
        // coupled elements: the displacements belong to the mechanical part and
        // are locked here (pure DC models only have temperatures)
        let mut constrained = loads.constrained();
        for (n, &a) in p.active.iter().enumerate() {
            for s in (0..SLOTS).filter(|&s| s != TS && a & (1 << s) != 0) {
                constrained.insert((n, s));
            }
        }
        let (dm, mat) = assembly::dofmap(p, &constrained, solver)?;
        let nn = model.nodes.len();
        let has_amps = !loads.amp_bc.is_empty() || !loads.amp_cload.is_empty() || loads.amp_dload.iter().any(|a| a.is_some());
        let amps = if has_amps { Some(super::dynamic::amplitude::LoadEval::new(model, &loads, loads.ramp, period, state.time)?.amps) } else { None };
        let film_props = film_properties(model)?;
        if loads.dload.iter().any(|(l, _)| l.kind.starts_with("FILMP:")) {
            linear = false;
        }
        let gaps = crate::interactions::thermal_contact::build(model)?;
        if !gaps.is_empty() {
            linear = false;
        }
        Ok(HeatSystem {
            p,
            loads,
            elems,
            props,
            dm,
            mat,
            u: state.u.clone(),
            u_committed: state.u.clone(),
            fint: vec![[0.0; SLOTS]; nn],
            fext: vec![[0.0; SLOTS]; nn],
            time0: state.time,
            period,
            t0: 0.0,
            t1: 0.0,
            transient,
            linear,
            abs_zero: z.unwrap_or(0.0),
            sigma: s.unwrap_or(0.0),
            amps,
            film_props,
            gaps,
            gap_disp: false,
            source: Vec::new(),
            cavities: None,
            convection: None,
        })
    }

    fn eval_loads(&self) -> Option<super::dynamic::amplitude::LoadEval<'_>> {
        let amps = self.amps.clone()?;
        Some(super::dynamic::amplitude::LoadEval {
            loads: &self.loads,
            amps,
            ramp: self.loads.ramp,
            period: self.period,
            time0: self.time0,
            step_full_at_start: false,
        })
    }

    /// Prescribed temperatures at the step time fraction `t` (with amplitudes).
    pub fn bc_now(&self, t: f64) -> Vec<(usize, usize, f64)> {
        match self.eval_loads() {
            Some(e) => e.bc_at(t * self.period),
            None => self.loads.bc_at(t),
        }
    }

    /// Amplitude factor for a flux in the increment ending at `t`: in
    /// transient analysis the mean of the amplitude values at the start and
    /// end of the increment (trapezoidal rule, also when a
    /// table point lies inside the increment), otherwise the value at `t`.
    fn flux_amp(&self, name: &str, b: f64, t: f64) -> Option<f64> {
        let amp = self.amps.as_ref()?.get(name)?;
        let s1 = t * self.period;
        let v = if self.transient {
            let s0 = self.t0 * self.period;
            0.5 * (amp.value(s0, self.time0 + s0) + amp.value(s1, self.time0 + s1))
        } else {
            amp.value(s1, self.time0 + s1)
        };
        Some(if amp.absolute { v } else { b * v })
    }

    fn cloads_now(&self, t: f64) -> Vec<(usize, usize, f64)> {
        match self.eval_loads() {
            Some(e) => {
                let mut v = e.cloads_at(t * self.period);
                for (i, (k, c)) in self.loads.cload.iter().enumerate() {
                    if let Some(q) = self.loads.amp_cload.get(k).and_then(|a| self.flux_amp(a, c.1, t)) {
                        v[i].2 = q;
                    }
                }
                v
            }
            None => self.loads.cloads_at(t),
        }
    }

    fn dloads_now(&self, t: f64) -> Vec<crate::fem::model::DistLoad> {
        match self.eval_loads() {
            Some(e) => {
                let mut v = e.dloads_at(t * self.period);
                for (i, (l, _)) in self.loads.dload.iter().enumerate() {
                    if !l.kind.starts_with("DFLUX:") {
                        continue;
                    }
                    if let Some(q) = self.loads.amp_dload.get(i).and_then(|a| a.as_ref()).and_then(|a| self.flux_amp(a, l.magnitude, t)) {
                        v[i].magnitude = q;
                    }
                }
                v
            }
            None => self.loads.dloads_at(t),
        }
    }

    fn temps(&self, e: usize, u: &[NodeVec]) -> Vec<f64> {
        self.p.model.elements[e].nodes.iter().map(|&n| u[n][TS]).collect()
    }

    /// Reaction fluxes (F_int − F_ext in locked DOFs).
    pub fn reactions(&self) -> Vec<NodeVec> {
        let mut r = vec![[0.0; SLOTS]; self.u.len()];
        for (n, rn) in r.iter_mut().enumerate() {
            for s in 0..SLOTS {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    rn[s] = self.fint[n][s] - self.fext[n][s];
                }
            }
        }
        r
    }

    /// Output HFL per element.
    pub fn element_output(&self, u: &[NodeVec]) -> Result<Vec<Vec<IpField>>> {
        let mut out = Vec::with_capacity(self.elems.len());
        for (e, he) in self.elems.iter().enumerate() {
            match he {
                // UMATHT: HFL is not written (the flux only exists in the routine)
                Some(h) if self.p.model.materials.iter().find(|m| m.name == self.props[h.props].name).is_some_and(|d| crate::user::loads::heat_user(d).0) => out.push(Vec::new()),
                Some(h) => {
                    let t = self.temps(e, u);
                    out.push(vec![h.el.heat_flux(&h.ips, &self.props[h.props], &t)?]);
                }
                None => out.push(Vec::new()),
            }
        }
        Ok(out)
    }

    pub fn frame(&self, value: f64, description: String) -> Result<FrameData> {
        Ok(FrameData {
            value,
            mode: -1,
            description,
            u: self.u_committed.clone(),
            rf: self.reactions(),
            active: self.dm.active.clone(),
            constrained: self.dm.constrained.clone(),
            elem: {
                let mut v = self.element_output(&self.u_committed)?;
                if let Some(cv) = &self.convection {
                    cv.output(self.p.model, &self.u_committed, &mut v)?;
                }
                v
            },
            node_extra: self.contact_output(),
        })
    }

    /// Heat flux across thermal contact per slave node (nodal field HFL, −q).
    fn contact_output(&self) -> Vec<(String, Vec<String>, Vec<Vec<f64>>)> {
        if self.gaps.is_empty() {
            return Vec::new();
        }
        let mut v = vec![vec![0.0]; self.u_committed.len()];
        for g in &self.gaps {
            for (n, q) in g.surface_flux(&self.u_committed, TS, self.gap_disp, self.abs_zero, self.sigma) {
                v[n][0] = -q;
            }
        }
        vec![("HFL".into(), vec!["HFL".into()], v)]
    }

    /// Surface fluxes, film and radiation at step time `t1`. Adds external and
    /// internal fluxes and returns the element matrices (equation numbers, matrix)
    /// for film and radiation if `tangent`.
    fn surface_terms(&self, fint: &mut [NodeVec], fext: &mut [NodeVec], tangent: bool) -> Result<Vec<(Vec<usize>, Vec<f64>)>> {
        let model = self.p.model;
        let mut mats = Vec::new();
        for l in self.dloads_now(self.t1) {
            let e = l.elem;
            let Some(h) = &self.elems[e] else { continue };
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            let x: Vec<[f64; 3]> = model.elements[e].nodes.iter().map(|&n| model.nodes[n]).collect();
            let nodes = &model.elements[e].nodes;
            let nn = nodes.len();
            let (kind, label) = l.kind.split_once(':').unwrap_or((l.kind.as_str(), ""));
            match kind {
                "DFLUX" if label == "BF" => {
                    for ip in &h.bips {
                        for i in 0..nn {
                            fext[nodes[i]][TS] += ip.n[i] * l.magnitude * ip.w;
                        }
                    }
                }
                "DFLUX" | "FILMS" => {
                    let f = h.el.face_index(label).map_err(wrap)?;
                    let lumped = kind == "FILMS" && FILM_LUMPED && h.el.first_order();
                    for fp in h.el.face_points(&x, f, h.thickness, lumped).map_err(wrap)? {
                        for i in 0..nn {
                            fext[nodes[i]][TS] += fp.n[i] * l.magnitude * fp.w;
                        }
                    }
                }
                "FILM" | "RAD" | "FILMP" | "FILMU" => {
                    let (label, prop) = label.split_once(':').unwrap_or((label, ""));
                    let f = h.el.face_index(label).map_err(wrap)?;
                    let lumped = FILM_LUMPED && h.el.first_order();
                    let t = self.temps(e, &self.u);
                    let mut fe = vec![0.0; nn];
                    let mut ke = vec![0.0; nn * nn];
                    for (g, fp) in h.el.face_points(&x, f, h.thickness, lumped).map_err(wrap)?.into_iter().enumerate() {
                        let tp: f64 = fp.n.iter().zip(&t).map(|(a, b)| a * b).sum();
                        let (q, dq) = if kind == "FILMU" {
                            // FILM (user/loads.rs): h and T∞ at the surface point
                            let mut xc = [0.0; 3];
                            for (ni, xi) in fp.n.iter().zip(&x) {
                                for c in 0..3 {
                                    xc[c] += ni * xi[c];
                                }
                            }
                            crate::user::loads::film(prop, f + 1, tp, xc, model.elements[e].label, g + 1, self.time0 + self.t1 * self.period).map_err(wrap)?
                        } else if kind == "FILM" {
                            (l.magnitude * tp, l.magnitude)
                        } else if kind == "FILMP" {
                            // h(T) from *FILM PROPERTY at the surface temperature
                            let Some(tab) = self.film_props.get(prop) else {
                                return fail(format!("film property {prop} does not exist (*FILM PROPERTY, NAME={prop})"));
                            };
                            let sink = l.extra.first().copied().unwrap_or(0.0);
                            let (hv, dh) = tab.eval(tp);
                            (l.magnitude * hv[0] * (tp - sink), l.magnitude * (hv[0] + dh[0] * (tp - sink)))
                        } else {
                            let sink = l.extra.first().copied().unwrap_or(0.0);
                            let a = tp - self.abs_zero;
                            let b = sink - self.abs_zero;
                            let c = self.sigma * l.magnitude;
                            (c * (a.powi(4) - b.powi(4)), 4.0 * c * a.powi(3))
                        };
                        for i in 0..nn {
                            if fp.n[i] == 0.0 {
                                continue;
                            }
                            fe[i] += fp.n[i] * q * fp.w;
                            for j in 0..nn {
                                ke[i * nn + j] += fp.n[i] * dq * fp.n[j] * fp.w;
                            }
                        }
                    }
                    for i in 0..nn {
                        fint[nodes[i]][TS] += fe[i];
                    }
                    if tangent {
                        mats.push((nodes.iter().map(|&n| self.dm.eq[n][TS]).collect(), ke));
                    }
                }
                _ => return fail(format!("{}: load type {} is not supported in *HEAT TRANSFER", model.elem_name(e), l.kind)),
            }
        }
        Ok(mats)
    }
}

impl NonlinearSystem for HeatSystem<'_, '_> {
    fn n_eq(&self) -> usize {
        self.dm.n_eq
    }

    fn eq_field(&self, eq: usize) -> Field {
        dof::field(SLOT_DOF[self.dm.owner[eq].1])
    }

    fn is_linear(&self) -> bool {
        self.linear && self.cavities.is_none()
    }

    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        self.t0 = t0;
        self.t1 = t1;
        self.u.clone_from(&self.u_committed);
        for (n, s, v) in self.bc_now(t1) {
            if self.dm.constrained[n] & (1 << s) != 0 {
                self.u[n][s] = v;
            }
        }
        if let Some(c) = self.cavities.as_mut() {
            c.start_increment(&self.u_committed, &self.u, &self.dm.constrained, self.time0 + t0 * self.period, (t1 - t0) * self.period)?;
        }
        Ok(())
    }

    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        let model = self.p.model;
        let nn = model.nodes.len();
        let dt = (self.t1 - self.t0) * self.period;
        if self.transient && dt <= 0.0 {
            return fail("internal: transient heat conduction with time increment 0");
        }
        let mut fint = vec![[0.0; SLOTS]; nn];
        let mut fext = vec![[0.0; SLOTS]; nn];
        if tangent {
            self.mat.zero();
        }
        for (e, he) in self.elems.iter().enumerate() {
            let Some(h) = he else { continue };
            let nodes = &model.elements[e].nodes;
            let t = self.temps(e, &self.u);
            let pr = &self.props[h.props];
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            // user routines: UMATHT replaces conduction and capacity, HETVAL gives heat generation
            let def = model.materials.iter().find(|m| m.name == pr.name);
            let (umatht, hetval) = def.map_or((false, false), crate::user::loads::heat_user);
            let time_start = self.time0 + self.t0 * self.period;
            let (mut fe, mut ke) = if let (true, Some(d)) = (umatht, def) {
                let told = self.temps(e, &self.u_committed);
                let x: Vec<[f64; 3]> = nodes.iter().map(|&n| model.nodes[n]).collect();
                crate::user::loads::umatht(d, &h.ips, &h.cips, &x, &t, &told, time_start, dt, self.transient, h.el.dim(), model.elements[e].label, tangent).map_err(wrap)?
            } else {
                h.el.conduction(&h.ips, pr, &t, tangent).map_err(wrap)?
            };
            if let (true, Some(d)) = (hetval, def) {
                let told = self.temps(e, &self.u_committed);
                let (fh, kh) = crate::user::loads::hetval(d, &h.bips, &t, &told, time_start, dt, tangent).map_err(wrap)?;
                for (a, b) in fe.iter_mut().zip(&fh) {
                    *a += b;
                }
                if let (Some(k), Some(kh)) = (ke.as_mut(), kh) {
                    for (a, b) in k.iter_mut().zip(&kh) {
                        *a += b;
                    }
                }
            }
            if self.transient && !umatht {
                let told = self.temps(e, &self.u_committed);
                let (fc, kc) = h.el.capacity(&h.cips, pr, &t, &told, dt, tangent).map_err(wrap)?;
                for (a, b) in fe.iter_mut().zip(&fc) {
                    *a += b;
                }
                if let (Some(k), Some(kc)) = (ke.as_mut(), kc) {
                    for (a, b) in k.iter_mut().zip(&kc) {
                        *a += b;
                    }
                }
            }
            for (i, &n) in nodes.iter().enumerate() {
                fint[n][TS] += fe[i];
            }
            if let Some(ke) = ke {
                let map: Vec<usize> = nodes.iter().map(|&n| self.dm.eq[n][TS]).collect();
                add_matrix(&mut *self.mat, &map, &ke);
            }
        }
        for (map, ke) in self.surface_terms(&mut fint, &mut fext, tangent)? {
            add_matrix(&mut *self.mat, &map, &ke);
        }
        for g in &self.gaps {
            for (nodes, ke) in g.contribute(&self.u, TS, self.gap_disp, self.abs_zero, self.sigma, &mut fint, tangent)? {
                let map: Vec<usize> = nodes.iter().map(|&n| self.dm.eq[n][TS]).collect();
                add_matrix(&mut *self.mat, &map, &ke);
            }
        }
        if let Some(c) = &self.cavities {
            c.contribute(&self.u, &self.dm, &mut fint, if tangent { Some(&mut *self.mat) } else { None });
            // MXDEM: too large an emissivity change in the increment gives a shorter increment
            c.check_mxdem(&self.u);
        }
        if let Some(cv) = &self.convection {
            cv.contribute(model, &self.u, &self.u_committed, self.t1, dt, self.transient, &self.dm, &mut fint, if tangent { Some(&mut *self.mat) } else { None })?;
        }
        for (n, s, v) in self.cloads_now(self.t1) {
            fext[n][s] += v;
        }
        for (n, q) in self.source.iter().enumerate() {
            fext[n][TS] += q;
        }
        self.fint = fint;
        self.fext = fext;
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            r[q] = self.fext[n][s] - self.fint[n][s];
        }
        let mut sum: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
        for n in 0..nn {
            for s in 0..SLOTS {
                if self.dm.active[n] & (1 << s) != 0 {
                    let e = sum.entry(dof::field(SLOT_DOF[s])).or_default();
                    let v = self.fint[n][s].abs().max(self.fext[n][s].abs());
                    if v > 0.0 {
                        e.0 += v;
                        e.1 += 1;
                    }
                }
            }
        }
        let flux = sum.into_iter().map(|(f, (s, c))| (f, if c > 0 { s / c as f64 } else { 0.0 })).collect();
        Ok(Residual { r, flux })
    }

    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        if let Err(q) = self.mat.factor() {
            return Err(assembly::mechanism_message(self.p, &self.dm, q));
        }
        let mut x = r.to_vec();
        self.mat.solve(&mut x);
        Ok(x)
    }

    fn apply(&mut self, c: &[f64]) {
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            self.u[n][s] += c[q];
        }
    }

    fn increment(&self) -> Vec<f64> {
        self.dm.owner.iter().map(|&(n, s)| self.u[n][s] - self.u_committed[n][s]).collect()
    }

    fn commit(&mut self) -> Result<()> {
        self.u_committed.clone_from(&self.u);
        Ok(())
    }
}

/// Film coefficients from *FILM PROPERTY, NAME= (h, temperature per line).
pub fn film_properties(model: &FeModel) -> Result<BTreeMap<String, crate::materials::heat::Table>> {
    let mut m = BTreeMap::new();
    for kw in model.extra.iter().filter(|k| k.name == "FILM PROPERTY") {
        let Some(name) = kw.param_upper("NAME") else { return kw.err("NAME= is missing") };
        m.insert(name, crate::materials::heat::read_table(kw, 1)?);
    }
    Ok(m)
}

/// Settings from the *HEAT TRANSFER line.
pub struct HeatControls {
    pub ctrl: Controls,
    pub steady: bool,
    /// Largest allowed temperature change per increment (DELTMX), 0 = fixed increments.
    pub deltmx: f64,
}

pub fn heat_controls(step: &StepDef) -> Result<HeatControls> {
    let kw = step.procedure.as_ref();
    let mut ctrl = Controls::from_procedure(kw, step.max_increments)?;
    // *CONTROLS, PARAMETERS=FIELD (R_n, C_n) as in the NLGEOM steps
    super::nlgeom::apply_controls(&mut ctrl, &step.extra)?;
    let steady = kw.is_some_and(|k| k.has("STEADY STATE"));
    let v = match kw {
        Some(k) => k.first_floats()?,
        None => Vec::new(),
    };
    let deltmx = match kw.and_then(|k| k.param("DELTMX")) {
        Some(d) => crate::fem::model::parse_f64(d).ok_or_else(|| FemError(format!("DELTMX={d} is not a number")))?.max(0.0),
        None => 0.0,
    };
    if let Some(k) = kw {
        if k.has("END") {
            return k.err("END=SS is not supported yet");
        }
    }
    if !steady {
        if v.first().is_none_or(|&x| x <= 0.0) {
            // transient analysis requires an initial increment; without it a single increment
            ctrl.initial = 1.0;
        }
        if deltmx == 0.0 {
            ctrl.fixed = true;
        }
    }
    Ok(HeatControls { ctrl, steady, deltmx })
}

/// The step's loads with the default amplitude for heat conduction.
pub fn heat_loads(step: &StepDef, loads: &StepLoads, steady: bool) -> StepLoads {
    let mut l = loads.clone();
    let explicit = step.header.param_upper("AMPLITUDE");
    l.ramp = match explicit.as_deref() {
        Some("RAMP") => true,
        Some("STEP") => false,
        _ => steady,
    };
    l
}

impl StepType for HeatTransfer {
    fn name(&self) -> &'static str {
        "HEAT TRANSFER"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "HEAT TRANSFER").then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.step.perturbation {
            return fail("*HEAT TRANSFER cannot be a perturbation step");
        }
        let hc = heat_controls(ctx.step)?;
        let had_init = ctx.state.vectors.contains_key(INIT_KEY);
        ensure_initial(model, ctx.state)?;
        let mut loads = heat_loads(ctx.step, ctx.loads, hc.steady);
        if !had_init {
            // boundary conditions ramped in the first step start from the initial temperature
            for ((n, s), v) in loads.bc.iter_mut() {
                if *s == TS && !ctx.state.bcs.contains_key(&(*n, *s)) {
                    v.0 = ctx.state.u[*n][TS];
                }
            }
        }
        let period = hc.ctrl.period;
        // cavity radiation: unsymmetric Jacobian, unsymmetric solver
        let cavity = super::heat_cavity::for_step(model, ctx.index)?;
        // forced convection (DCC elements): unsymmetric element matrices
        let convection = super::heat_convection::for_step(prep, ctx.index, &loads, !hc.steady)?;
        let solver = if (cavity.is_some() || convection.is_some()) && ctx.solver.name() != "unsymmetric" { crate::solver::select(Some("unsymmetric"))? } else { ctx.solver };
        let mut sys = HeatSystem::new(prep, loads, ctx.state, !hc.steady, solver, period)?;
        sys.convection = convection;
        sys.cavities = cavity.map(|mut c| {
            // MXDEM applies to automatic incrementation (not fixed increments or DELTMX)
            c.mxdem = if hc.ctrl.fixed || hc.deltmx > 0.0 { None } else { super::heat_cavity::mxdem(ctx.step) };
            c
        });
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            sys.begin_increment(0.0, 0.0)?;
            sys.u.clone_from(&sys.u_committed);
            if !hc.steady {
                // frame 0: no capacity contribution (Δt = 0)
                sys.transient = false;
                sys.evaluate(false)?;
                sys.transient = true;
            } else {
                sys.evaluate(false)?;
            }
            let mut f0 = sys.frame(0.0, "Increment 0: Step Time = 0.0".into())?;
            // frame 0 in the first step is the initial state: no
            // reaction fluxes; in later steps the
            // reaction fluxes from the end of the previous step are shown
            if !had_init {
                for r in f0.rf.iter_mut() {
                    *r = [0.0; SLOTS];
                }
            }
            frames.push(f0);
        }
        let verbose = ctx.opts.verbose;
        let mut inc = 0;
        let rep = if hc.deltmx > 0.0 && !hc.steady {
            run_deltmx(&mut sys, &hc, &mut |s: &mut HeatSystem, t| {
                inc += 1;
                if verbose {
                    eprintln!("  increment {inc}: step time {:.6}", t * period);
                }
                if all || t >= 1.0 - 1e-12 {
                    frames.push(s.frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
                }
                Ok(())
            })?
        } else {
            newton::run(&mut sys, &hc.ctrl, &mut |s: &mut HeatSystem, t| {
                inc += 1;
                if verbose {
                    eprintln!("  increment {inc}: step time {:.6}", t * period);
                }
                if all || t >= 1.0 - 1e-12 {
                    frames.push(s.frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
                }
                Ok(())
            })?
        };
        ctx.out.frames.extend(frames);
        ctx.out.increments = rep.increments;
        ctx.out.iterations = rep.iterations;
        ctx.state.u = sys.u_committed.clone();
        ctx.state.reaction = sys.reactions();
        ctx.state.time += period;
        Ok(())
    }
}

/// Transient with automatic incrementation by DELTMX (largest
/// temperature change in a free node per increment). The rule:
/// - factor f = 0.8·DELTMX/ΔT_max;
/// - ΔT_max > DELTMX: the increment is redone with Δt·f;
/// - linear: the next Δt is increased by min(f, 2) only if f ≥ 1.9 (otherwise Δt is kept);
/// - nonlinear: an increase is allowed only after two consecutive accepted increments
///   (the step's first increment and the increment directly after a DELTMX-limited
///   change or a halving do not count). The increase is min(f, g) with g = 2 if
///   the increment already converged after the first solve (residual below
///   R_tol·q̃) and otherwise g = 1.5. If the increase was limited by DELTMX (f < g) the
///   count restarts;
/// - at most Δt_max; the last increment is shortened to the end of the step.
pub fn run_deltmx(sys: &mut HeatSystem, hc: &HeatControls, on_increment: &mut dyn FnMut(&mut HeatSystem, f64) -> Result<()>) -> Result<newton::Report> {
    let ctrl = &hc.ctrl;
    let mut rep = newton::Report::default();
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    // consecutive accepted increments that may count towards an increase (−1 = the next does not count)
    let mut count: i64 = -1;
    // sum of typical fluxes in accepted increments (for q̃ as in newton.rs)
    let mut flux_sum = 0.0f64;
    let mut n_flux = 0usize;
    while t < 1.0 - 1e-12 {
        if rep.increments >= ctrl.max_increments {
            return fail(format!("the step reached the maximum number of increments ({}) – increase INC= on *STEP", ctrl.max_increments));
        }
        dt = dt.min(1.0 - t);
        let mut tries = 0;
        let (dtmax_change, first) = loop {
            let sub = Controls { initial: 1.0, fixed: true, ..ctrl.clone() };
            // one increment from t to t+dt with Newton (fixed increment)
            let mut one = OneIncrement { sys, t0: t, t1: t + dt, evals: 0, first: None };
            let r = newton::run(&mut one, &sub, &mut |_, _| Ok(()));
            let first = one.first;
            match r {
                Ok(r1) => {
                    rep.iterations += r1.iterations;
                    let dmax = sys
                        .dm
                        .owner
                        .iter()
                        .map(|&(n, s)| (sys.u[n][s] - sys.u_committed[n][s]).abs())
                        .fold(0.0, f64::max);
                    if dmax > hc.deltmx * (1.0 + 1e-9) {
                        tries += 1;
                        if tries > ctrl.max_cutbacks || dt * 0.8 * hc.deltmx / dmax < ctrl.min {
                            return fail(format!(
                                "DELTMX={} cannot be satisfied at step time {:.4} without the increment becoming smaller than the minimum",
                                hc.deltmx,
                                t * ctrl.period
                            ));
                        }
                        dt *= 0.8 * hc.deltmx / dmax;
                        count = -1;
                        continue;
                    }
                    break (dmax, first);
                }
                Err(e) => {
                    tries += 1;
                    dt *= 0.25;
                    count = -1;
                    if tries > ctrl.max_cutbacks || dt < ctrl.min {
                        return Err(e);
                    }
                }
            }
        };
        sys.commit()?;
        t += dt;
        rep.increments += 1;
        crate::progress::emit(crate::progress::Event::Increment { fraction: t, iterations: 0 })?;
        on_increment(sys, t)?;
        count += 1;
        let f = if dtmax_change > 0.0 { 0.8 * hc.deltmx / dtmax_change } else { f64::INFINITY };
        if sys.linear {
            if f >= 1.9 {
                dt *= f.min(2.0);
            }
        } else {
            // convergence after the first solve: the residual after it is below R_tol·q̃
            let one_iter = first.is_some_and(|(r, q)| r <= ctrl.r_tol * (flux_sum + q) / (n_flux + 1) as f64);
            if let Some((_, q)) = first {
                flux_sum += q;
                n_flux += 1;
            }
            let g = if one_iter { 2.0 } else { ctrl.growth };
            if count >= 2 && f > 1.0 {
                dt *= f.min(g);
                if f < g {
                    count = -1;
                }
            }
        }
        dt = dt.min(ctrl.max);
    }
    Ok(rep)
}

/// An increment with fixed Δt for `run_deltmx`. Stores the largest residual and
/// typical flux after the first solve (second residual evaluation).
struct OneIncrement<'a, 'p, 'm> {
    sys: &'a mut HeatSystem<'p, 'm>,
    t0: f64,
    t1: f64,
    evals: usize,
    first: Option<(f64, f64)>,
}

impl NonlinearSystem for OneIncrement<'_, '_, '_> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        false
    }
    fn begin_increment(&mut self, _a: f64, _b: f64) -> Result<()> {
        self.sys.begin_increment(self.t0, self.t1)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        let res = self.sys.evaluate(tangent)?;
        self.evals += 1;
        if self.evals == 2 {
            let r = res.r.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let q = res.flux.values().copied().fold(0.0f64, f64::max);
            self.first = Some((r, q));
        }
        Ok(res)
    }
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        self.sys.solve(r)
    }
    fn apply(&mut self, c: &[f64]) {
        self.sys.apply(c)
    }
    fn increment(&self) -> Vec<f64> {
        self.sys.increment()
    }
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }
}
