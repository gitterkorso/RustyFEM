//! Coupled pore pressure–displacement (geotechnics): *SOILS (steady-state or
//! CONSOLIDATION) and *GEOSTATIC with the elements in `elements/soils.rs`.
//!
//! The whole system (displacements and pore pressures) is solved simultaneously with
//! Newton–Raphson. The rows of the continuity equation are multiplied by −Δt
//! (transient) so that the linear consolidation becomes symmetric,
//!
//! ```text
//! [ K    −Q   ] [Δu  ]   [R_u]
//! [ −Qᵀ  −Δt H] [Δu_w] = [R_w]
//! ```
//!
//! and solved with the indefinite solver (the pressure degrees of freedom are scaled by
//! the elastic modulus so that the pivots for short increments do not become too
//! small); steady-state flow, k(e), GRAV, *FLOW and nonlinear materials
//! use the unsymmetric solver. *CONTROLS, PARAMETERS=FIELD/TIME
//! INCREMENTATION apply as in the NLGEOM step.
//!
//! Incrementation:
//! - *SOILS, CONSOLIDATION without UTOL: fixed increments Δt0 (data line `Δt0,
//!   time period, Δtmin, Δtmax, ss rate`);
//! - UTOL=: automatic so that the largest pore pressure change per increment
//!   stays below UTOL: cutback with Δt·0.85·UTOL/Δu_max, increase as DELTMX
//!   in `steps/heat.rs` (linear: ×min(f, 2) when f = 0.8·UTOL/Δu_max ≥ 1.9);
//! - END=SS: the step ends when all pore pressures change more slowly than
//!   the fifth value of the data line;
//! - steady-state: the Newton driver's incrementation;
//! - *GEOSTATIC without GRAV: one transient increment with Δt = the step time, where
//!   an imbalance between initial stress and loads is carried by the pore pressure
//!   (RVF is reported without the volume term); with GRAV on the pore pressure elements
//!   steady-state flow (drained).
//!
//! Default amplitude: STEP for CONSOLIDATION, RAMP for steady-state analysis and
//! *GEOSTATIC.
//!
//! GRAV on pore pressure elements gives total pore pressure: the weight is
//! (ρ_dry + n·ρw)·g with the current porosity and ρw = γw/g, and Darcy's law gets
//! the term −ρw g. BX/BY/BZ and P act only on the skeleton (excess pressure).
//!
//! Initial conditions (`inp/keywords/soils.rs`): *INITIAL CONDITIONS,
//! TYPE=PORE PRESSURE, RATIO and STRESS (also GEOSTATIC) are inserted into
//! the state before the first step that uses them.
//!
//! Loads: mechanical loads and boundary conditions, *BOUNDARY on degree of freedom 8,
//! *CFLOW (point flow), *DFLOW/*DSFLOW (prescribed outflow per unit area) and
//! *FLOW/*SFLOW (outflow c·(u_w − u_w∞)).
//! Output: U, RF, POR, RVF (reaction flow), RVT (summed reaction volume),
//! S (effective stress), E, FLVEL (volume flux per unit area), VOIDR.

use super::api::{StepCtx, StepType};
use super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::elements::soils::{FluidMode, SoilElem};
use crate::elements::IpField;
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::model::{DistLoad, FeModel, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::materials::porous::PoroProps;
use crate::model::{fail, FemError, Result};
use crate::solver::{LinearSolver, SystemMatrix};
use std::collections::BTreeMap;

pub struct Soils;

pub static STEP_TYPES: &[&dyn StepType] = &[&Soils];

/// Slot for the pore pressure (degree of freedom 8).
pub const PS: usize = 7;

/// Key in `SolutionState::vectors`: the initial conditions have been inserted.
const INIT_KEY: &str = "POR INITIAL";
/// Key in `SolutionState::vectors`: summed reaction volume RVT (slot 7).
const RVT_KEY: &str = "RVT";

/// Marker for resolved *INITIAL CONDITIONS (node or element index).
pub const RESOLVED: &str = "__SOILS_RESOLVED";

/// Distributed loads that belong to the pore fluid (not to the mechanics).
fn is_flow(kind: &str) -> bool {
    kind.starts_with("DFLOW:") || kind.starts_with("FLOW:")
}

/// Vertical coordinate (y in 2D, z in 3D).
fn elevation(model: &FeModel, x: [f64; 3]) -> f64 {
    if model.dim() == 3 {
        x[2]
    } else {
        x[1]
    }
}

/// Linear variation with height: value1, height1, value2, height2.
fn linear_in(v: &[f64], y: f64) -> f64 {
    let v1 = v.first().copied().unwrap_or(0.0);
    if v.len() < 4 {
        return v1;
    }
    let (y1, v2, y2) = (v[1], v[2], v[3]);
    if (y2 - y1).abs() < 1e-300 {
        return v1;
    }
    v1 + (v2 - v1) * (y - y1) / (y2 - y1)
}

/// Element in the model: pore pressure element with properties, or other.
struct Elems {
    soil: Vec<Option<(&'static SoilElem, usize)>>,
    props: Vec<PoroProps>,
}

fn classify(p: &Prepared) -> Result<Elems> {
    let model = p.model;
    let mut soil = Vec::with_capacity(model.elements.len());
    let mut props: Vec<PoroProps> = Vec::new();
    let mut idx: BTreeMap<usize, usize> = BTreeMap::new();
    for (e, el) in model.elements.iter().enumerate() {
        let Some(se) = crate::elements::soils::lookup(el.etype.name()) else {
            soil.push(None);
            continue;
        };
        let Some(si) = el.section else {
            return fail(format!("{}: *SOLID SECTION is missing", model.elem_name(e)));
        };
        let Some(mi) = model.sections[si].material_index else {
            return fail(format!("{}: the section has no material", model.elem_name(e)));
        };
        let pi = match idx.get(&mi) {
            Some(&i) => i,
            None => {
                props.push(PoroProps::require(&model.materials[mi])?);
                idx.insert(mi, props.len() - 1);
                props.len() - 1
            }
        };
        soil.push(Some((se, pi)));
    }
    if soil.iter().all(|s| s.is_none()) {
        return fail("*SOILS/*GEOSTATIC requires pore pressure elements (CPE4P, CPE8P, CAX4P, C3D8P, C3D20P, ...)");
    }
    Ok(Elems { soil, props })
}

/// Inserts *INITIAL CONDITIONS (PORE PRESSURE, RATIO, STRESS) into the state
/// the first time a *SOILS/*GEOSTATIC step runs.
fn ensure_initial(p: &Prepared, els: &Elems, state: &mut SolutionState) -> Result<()> {
    if state.vectors.contains_key(INIT_KEY) {
        return Ok(());
    }
    let model = p.model;
    let nn = model.nodes.len();
    let mut ratio: Vec<Option<f64>> = vec![None; nn];
    let mut por: Vec<Option<f64>> = vec![None; nn];
    let mut stress: Vec<Option<(bool, Vec<f64>)>> = vec![None; model.elements.len()];
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED)) {
        let ty = kw.param_upper("TYPE").unwrap_or_default();
        for d in &kw.data {
            let v = kw.floats(d)?;
            let Some(&i) = v.first() else { continue };
            let i = i as usize;
            let rest = v[1..].to_vec();
            match ty.as_str() {
                "PORE PRESSURE" => por[i] = Some(linear_in(&rest, elevation(model, model.nodes[i]))),
                "RATIO" => ratio[i] = Some(linear_in(&rest, elevation(model, model.nodes[i]))),
                "STRESS" => stress[i] = Some((kw.has("GEOSTATIC"), rest)),
                _ => {}
            }
        }
    }
    for (n, v) in por.iter().enumerate() {
        if let Some(v) = v {
            if p.active[n] & (1 << PS) != 0 && !state.bcs.contains_key(&(n, PS)) {
                state.u[n][PS] = *v;
            }
        }
    }
    for (e, s) in els.soil.iter().enumerate() {
        let Some((se, _)) = s else {
            // other elements get their initial stress in materials/initial_state.rs
            continue;
        };
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let mat = ctx.material(se.name)?;
        let lay = se.layout(mat);
        let stride = lay.stride();
        let nodes = &model.elements[e].nodes;
        let x: Vec<[f64; 3]> = nodes.iter().map(|&n| model.nodes[n]).collect();
        let gs = se.geometry(&x, se.thickness(&ctx)?)?;
        let st = &mut state.elem_state[e];
        if st.len() != gs.len() * stride {
            *st = se.fresh_state(mat, gs.len());
        }
        let ns = lay.ns;
        // the GEOSTATIC stress is evaluated at the element midpoint
        // (the same value at all integration points)
        let center = se.center(&x);
        for (q, g) in gs.iter().enumerate() {
            let base = q * stride;
            // void ratio at the integration point: the nodal values interpolated
            let e0: f64 = g.ip.n.iter().zip(nodes).map(|(ni, &n)| ni * ratio[n].unwrap_or(0.0)).sum();
            st[base + lay.e0()] = e0;
            if let Some((geo, vals)) = &stress[e] {
                let sig = initial_stress(model, se, *geo, vals, if *geo { center } else { g.x }, ns);
                let Some((ee, nu)) = mat.elastic_moduli() else {
                    return fail(format!("{}: initial stress requires a material with isotropic elasticity", model.elem_name(e)));
                };
                let eps0 = compliance(&sig, ee, nu, ns);
                st[base..base + ns].copy_from_slice(&sig);
                st[base + lay.eps0()..base + lay.eps0() + ns].copy_from_slice(&eps0);
            }
        }
    }
    state.vectors.insert(INIT_KEY.into(), Vec::new());
    Ok(())
}

/// Initial stress (component order 11, 22, 33, 12, ...) at the point `x`.
fn initial_stress(model: &FeModel, se: &SoilElem, geo: bool, v: &[f64], x: [f64; 3], ns: usize) -> Vec<f64> {
    let mut s = vec![0.0; ns];
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    if geo {
        // σv1, height1, σv2, height2, K0x, K0y
        let sv = linear_in(&v[..4.min(v.len())], elevation(model, x));
        let kx = g(4);
        let ky = if v.len() > 5 { g(5) } else { kx };
        if se.ndim() == 3 {
            s[0] = kx * sv;
            s[1] = ky * sv;
            s[2] = sv;
        } else {
            s[0] = kx * sv;
            s[1] = sv;
            s[2] = ky * sv;
        }
    } else {
        // S11, S22, S33, S12, S13, S23
        for (i, si) in s.iter_mut().enumerate() {
            *si = g(i);
        }
    }
    s
}

/// Elastic strain C⁻¹σ (engineering shear strain) for isotropic elasticity.
fn compliance(s: &[f64], e: f64, nu: f64, ns: usize) -> Vec<f64> {
    let mut eps = vec![0.0; ns];
    for i in 0..3 {
        let o: f64 = (0..3).filter(|&j| j != i).map(|j| s[j]).sum();
        eps[i] = (s[i] - nu * o) / e;
    }
    let g = e / (2.0 * (1.0 + nu));
    for i in 3..ns {
        eps[i] = s[i] / g;
    }
    eps
}

/// Settings from the procedure line.
struct SoilControls {
    ctrl: Controls,
    transient: bool,
    geostatic: bool,
    /// Largest allowed pore pressure change per increment (UTOL), 0 = fixed increments.
    utol: f64,
    /// END=SS: rate limit for steady state (0 = off).
    ss_rate: f64,
}

fn controls(step: &StepDef) -> Result<SoilControls> {
    let Some(kw) = step.procedure.as_ref() else { return fail("internal: *SOILS has no procedure") };
    let geostatic = kw.name == "GEOSTATIC";
    let mut ctrl = Controls::from_procedure(Some(kw), step.max_increments)?;
    // *CONTROLS, PARAMETERS=FIELD / TIME INCREMENTATION as in the NLGEOM step
    crate::steps::nlgeom::apply_controls(&mut ctrl, &step.extra)?;
    let transient = !geostatic && kw.has("CONSOLIDATION");
    let v = kw.first_floats()?;
    let num = |key: &str| -> Result<f64> {
        match kw.param(key) {
            Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: {key}={s} is not a number", kw.location()))),
            None => Ok(0.0),
        }
    };
    let utol = if transient { num("UTOL")?.max(0.0) } else { 0.0 };
    let mut ss_rate = 0.0;
    if let Some(end) = kw.param_upper("END") {
        if end != "SS" && end != "PERIOD" {
            return kw.err(format!("END={end} is not supported (SS or PERIOD)"));
        }
        if end == "SS" {
            if !transient {
                return kw.err("END=SS applies only to *SOILS, CONSOLIDATION");
            }
            ss_rate = v.get(4).copied().unwrap_or(0.0);
            if !(ss_rate > 0.0) {
                return kw.err("END=SS requires the rate limit for steady state (the fifth value on the data line)");
            }
        }
    }
    for p in ["CREEP", "STABILIZE", "ALLSDTOL"] {
        if kw.has(p) {
            return kw.err(format!("{p} is not supported yet for *{}", kw.name));
        }
    }
    if transient {
        if v.first().is_none_or(|&x| x <= 0.0) {
            ctrl.initial = 1.0;
        }
        if utol == 0.0 {
            ctrl.fixed = true;
        }
    }
    Ok(SoilControls { ctrl, transient, geostatic, utol, ss_rate })
}

/// The indefinite solver without a warning about negative eigenvalues (the saddle point system
/// for consolidation always has as many negative eigenvalues as pressure degrees of freedom).
struct QuietIndefinite;

static QUIET_INDEFINITE: QuietIndefinite = QuietIndefinite;

impl LinearSolver for QuietIndefinite {
    fn name(&self) -> &'static str {
        "indefinite"
    }
    fn prepare(&self, g: &crate::solver::Graph) -> Result<Box<dyn SystemMatrix>> {
        crate::solver::indefinite::prepare_quiet(g)
    }
}

/// The whole system for a step.
struct SoilSystem<'p, 'm> {
    p: &'p Prepared<'m>,
    loads: StepLoads,
    els: Elems,
    dm: DofMap,
    mat: Box<dyn SystemMatrix>,
    u: Vec<NodeVec>,
    u_committed: Vec<NodeVec>,
    state_old: Vec<Vec<f64>>,
    state_new: Vec<Vec<f64>>,
    fint: Vec<NodeVec>,
    fext: Vec<NodeVec>,
    time0: f64,
    period: f64,
    t0: f64,
    t1: f64,
    transient: bool,
    linear: bool,
    /// Weight of the pore fluid per volume ρw·g per element (from GRAV, otherwise 0).
    grav: Vec<[f64; 3]>,
    /// Scaling of the pressure degrees of freedom in the equation system (the largest
    /// elastic modulus): the coupling terms become comparable to the stiffness,
    /// so that the pivot tests do not take the small pressure pivots for short
    /// time increments (undrained load) as zero.
    pscale: f64,
}

impl<'p, 'm> SoilSystem<'p, 'm> {
    fn new(p: &'p Prepared<'m>, loads: StepLoads, els: Elems, state: &SolutionState, transient: bool, period: f64) -> Result<Self> {
        // GRAV on pore pressure elements: the fluid weight depends on the void ratio (nonlinear)
        let grav = loads.dload.iter().any(|(l, _)| l.kind == "GRAV" && els.soil[l.elem].is_some());
        let linear = p.is_linear() && els.props.iter().all(|pp| !pp.perm.varies()) && !loads.dload.iter().any(|(l, _)| l.kind.starts_with("FLOW:")) && !grav;
        let sym = transient && linear;
        let solver: &'static dyn LinearSolver = if sym { &QUIET_INDEFINITE } else { crate::solver::select(Some("unsymmetric"))? };
        let (dm, mat) = assembly::dofmap(p, &loads.constrained(), solver)?;
        let nn = p.model.nodes.len();
        Ok(SoilSystem {
            p,
            loads,
            els,
            dm,
            mat,
            u: state.u.clone(),
            u_committed: state.u.clone(),
            state_old: state.elem_state.clone(),
            state_new: state.elem_state.clone(),
            fint: vec![[0.0; SLOTS]; nn],
            fext: vec![[0.0; SLOTS]; nn],
            time0: state.time,
            period,
            t0: 0.0,
            t1: 0.0,
            transient,
            linear,
            grav: vec![[0.0; 3]; p.model.elements.len()],
            pscale: p.materials.iter().flatten().filter_map(|m| m.elastic_moduli()).map(|(e, _)| e).fold(0.0, f64::max).max(1.0),
        })
    }

    /// ρw·g per element at the step time `t1`: GRAV on a pore pressure element
    /// gives total pore pressure (Darcy with −ρw g and the fluid weight n·ρw·g),
    /// ρw = γw/g with g = the magnitude of the GRAV load at the end of the step.
    fn fluid_gravity(&self) -> Result<Vec<[f64; 3]>> {
        let mut g = vec![[0.0; 3]; self.p.model.elements.len()];
        for ((end, _), now) in self.loads.dload.iter().zip(self.loads.dloads_at(self.t1)) {
            if now.kind != "GRAV" {
                continue;
            }
            let Some((_, pi)) = self.els.soil[now.elem] else { continue };
            let d = [0, 1, 2].map(|i| now.extra.get(i).copied().unwrap_or(0.0));
            let l = crate::elements::common::norm(d);
            if l == 0.0 {
                return fail(format!("{}: the GRAV load has no direction", self.p.model.elem_name(now.elem)));
            }
            let gref = if end.magnitude != 0.0 { end.magnitude } else { now.magnitude };
            if gref == 0.0 {
                continue;
            }
            let gw = self.els.props[pi].perm.gamma_w;
            let s = gw * now.magnitude / gref / l;
            for i in 0..3 {
                g[now.elem][i] += s * d[i];
            }
        }
        Ok(g)
    }

    /// The weight (ρ_dry + n·ρw)·g for GRAV loads on pore pressure elements.
    fn gravity_load(&self, fext: &mut [NodeVec]) -> Result<()> {
        let model = self.p.model;
        for l in self.loads.dloads_at(self.t1) {
            if l.kind != "GRAV" {
                continue;
            }
            let e = l.elem;
            let Some((se, _)) = self.els.soil[e] else { continue };
            let ctx = self.p.ctx(e, false, 0.0, 0.0);
            let d = [0, 1, 2].map(|i| l.extra.get(i).copied().unwrap_or(0.0));
            let len = crate::elements::common::norm(d);
            if len == 0.0 {
                return fail(format!("{}: the GRAV load has no direction", model.elem_name(e)));
            }
            let dry = d.map(|x| x * l.magnitude * ctx.density / len);
            let v = self.p.gather(e, &self.u);
            let fe = se.gravity(&ctx, &v, &self.state_old[e], dry, self.grav[e]).map_err(|err| FemError(format!("{}: {err}", model.elem_name(e))))?;
            for (a, &(n, s)) in self.p.edofs[e].iter().enumerate() {
                fext[n][s] += fe[a];
            }
        }
        Ok(())
    }

    fn dt(&self) -> f64 {
        (self.t1 - self.t0) * self.period
    }

    /// Row scaling: the continuity equation is multiplied by −Δt (transient) or −1.
    fn row_scale(&self, s: usize) -> f64 {
        if s != PS {
            1.0
        } else if self.transient {
            -self.dt()
        } else {
            -1.0
        }
    }

    fn reactions(&self) -> Vec<NodeVec> {
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

    /// Surface loads of the pore fluid: prescribed outflow (DFLOW) and outflow
    /// c·(u_w − u_w∞) (FLOW). Returns element matrices for FLOW.
    fn flow_terms(&self, fint: &mut [NodeVec], fext: &mut [NodeVec], tangent: bool) -> Result<Vec<(Vec<(usize, usize)>, Vec<f64>)>> {
        let model = self.p.model;
        let mut mats = Vec::new();
        for l in self.loads.dloads_at(self.t1) {
            if !is_flow(&l.kind) {
                continue;
            }
            let e = l.elem;
            let Some((se, _)) = self.els.soil[e] else {
                return fail(format!("{}: {} requires a pore pressure element", model.elem_name(e), l.kind));
            };
            let (kind, label) = l.kind.split_once(':').unwrap_or((l.kind.as_str(), ""));
            let face = label[1..].parse::<usize>().ok().filter(|&f| f >= 1 && f <= se.shape.faces().len()).ok_or_else(|| {
                FemError(format!("{}: face {label} does not exist for {}", model.elem_name(e), se.name))
            })?;
            let ctx = self.p.ctx(e, false, 0.0, 0.0);
            let t = se.thickness(&ctx)?;
            let nodes = &model.elements[e].nodes;
            let pts = face_points(se, &ctx.x, face - 1, t)?;
            match kind {
                "DFLOW" => {
                    // prescribed outflow per unit area: external "inflow" −q
                    for (np, w) in &pts {
                        for (i, &k) in se.shape.faces()[face - 1].iter().enumerate().filter(|(_, &k)| k < se.n_corner()) {
                            fext[nodes[k]][PS] -= np[i] * l.magnitude * w;
                        }
                    }
                }
                "FLOW" => {
                    let sink = l.extra.first().copied().unwrap_or(0.0);
                    let fnodes: Vec<usize> = se.shape.faces()[face - 1].iter().copied().filter(|&k| k < se.n_corner()).collect();
                    let m = fnodes.len();
                    let mut ke = vec![0.0; m * m];
                    for (np, w) in &pts {
                        let pw: f64 = (0..m).map(|i| np[i] * self.u[nodes[fnodes[i]]][PS]).sum();
                        let q = l.magnitude * (pw - sink);
                        for i in 0..m {
                            fint[nodes[fnodes[i]]][PS] += np[i] * q * w;
                            for j in 0..m {
                                ke[i * m + j] += np[i] * l.magnitude * np[j] * w;
                            }
                        }
                    }
                    if tangent {
                        mats.push((fnodes.iter().map(|&k| (nodes[k], PS)).collect(), ke));
                    }
                }
                _ => {}
            }
        }
        Ok(mats)
    }

    fn add_matrix(&mut self, dl: &[(usize, usize)], ke: &[f64]) {
        let nd = dl.len();
        let sym = self.mat.is_symmetric();
        let map: Vec<usize> = dl.iter().map(|&(n, s)| self.dm.eq[n][s]).collect();
        // row scaling (−Δt for the continuity) and symmetric scaling D K D of
        // the pressure degrees of freedom (see `pscale`)
        let d: Vec<f64> = dl.iter().map(|&(_, s)| if s == PS { self.pscale } else { 1.0 }).collect();
        let sc: Vec<f64> = dl.iter().zip(&d).map(|(&(_, s), &di)| self.row_scale(s) * di).collect();
        for a in 0..nd {
            let ia = map[a];
            if ia == NONE {
                continue;
            }
            for b in 0..nd {
                let ib = map[b];
                if ib == NONE {
                    continue;
                }
                if sym {
                    if ia >= ib {
                        let v = 0.5 * (sc[a] * ke[a * nd + b] * d[b] + sc[b] * ke[b * nd + a] * d[a]);
                        if v != 0.0 {
                            self.mat.add(ia, ib, v);
                        }
                    }
                } else {
                    let v = sc[a] * ke[a * nd + b] * d[b];
                    if v != 0.0 {
                        self.mat.add_unsym(ia, ib, v);
                    }
                }
            }
        }
    }

    /// Element output (S, E, FLVEL, VOIDR, ...).
    fn element_output(&self, u: &[NodeVec], state: &[Vec<f64>]) -> Result<Vec<Vec<IpField>>> {
        let model = self.p.model;
        let time = self.time0 + self.t1 * self.period;
        let mut out = Vec::with_capacity(model.elements.len());
        for e in 0..model.elements.len() {
            let ctx = self.p.ctx(e, false, time, 0.0);
            let v = self.p.gather(e, u);
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            out.push(match self.els.soil[e] {
                Some((se, pi)) => se.soil_output(&ctx, Some(&self.els.props[pi]), &v, &state[e], self.grav[e]).map_err(wrap)?,
                None => {
                    if ctx.material.is_none() {
                        Vec::new()
                    } else {
                        model.elements[e].etype.output(&ctx, &v, &state[e]).map_err(wrap)?
                    }
                }
            });
        }
        Ok(out)
    }

    fn frame(&self, value: f64, description: String, rvt: &[f64]) -> Result<FrameData> {
        let nn = self.u.len();
        let rf = self.reactions();
        let has_p: Vec<bool> = (0..nn).map(|n| self.dm.active[n] & (1 << PS) != 0).collect();
        let mut por: Vec<Vec<f64>> = (0..nn).map(|n| if has_p[n] { vec![self.u_committed[n][PS]] } else { Vec::new() }).collect();
        // midside nodes in second-order elements: the pressure interpolated from the corners
        for (e, s) in self.els.soil.iter().enumerate() {
            let Some((se, _)) = s else { continue };
            let nodes = &self.p.model.elements[e].nodes;
            if nodes.len() == se.n_corner() {
                continue;
            }
            let pp: Vec<f64> = nodes[..se.n_corner()].iter().map(|&n| self.u_committed[n][PS]).collect();
            for (k, v) in se.midside_pressure(&pp) {
                let n = nodes[k];
                if !has_p[n] {
                    por[n] = vec![v];
                }
            }
        }
        let rvf: Vec<Vec<f64>> = (0..nn).map(|n| if has_p[n] { vec![rf[n][PS]] } else { Vec::new() }).collect();
        let rv: Vec<Vec<f64>> = (0..nn).map(|n| if has_p[n] { vec![rvt[n]] } else { Vec::new() }).collect();
        Ok(FrameData {
            nforc: Vec::new(),
            value,
            mode: -1,
            description,
            u: self.u_committed.clone(),
            rf,
            active: self.dm.active.clone(),
            constrained: self.dm.constrained.clone(),
            elem: self.element_output(&self.u_committed, &self.state_old)?,
            node_extra: vec![
                ("POR".into(), vec!["POR".into()], por),
                ("RVF".into(), vec!["RVF".into()], rvf),
                ("RVT".into(), vec!["RVT".into()], rv),
            ],
        })
    }
}

/// Integration points on the element face `face`: (the pressure shape functions for
/// the face nodes in the face node order, weight with thickness or 2πr).
fn face_points(se: &SoilElem, x: &[[f64; 3]], face: usize, t: f64) -> Result<Vec<(Vec<f64>, f64)>> {
    use crate::elements::solid_higher::shapes::{self, gauss1d, Shape};
    let fl = se.shape.faces()[face];
    let fshape = se.shape.face_shape(fl.len());
    let xf: Vec<[f64; 3]> = fl.iter().map(|&k| x[k]).collect();
    let ncorner_face = fl.iter().filter(|&&k| k < se.n_corner()).count();
    let pshape = if fshape.pdim() == 1 { Shape::Line2 } else { Shape::Quad4 };
    let g = gauss1d(2);
    let mut pts = Vec::new();
    let mut push = |p: [f64; 3], gw: f64| -> Result<()> {
        let (n, dn) = shapes::shape(fshape, p);
        let (np, _) = shapes::shape(pshape, p);
        let pos: [f64; 3] = [0, 1, 2].map(|c| n.iter().zip(&xf).map(|(a, b)| a * b[c]).sum());
        let jac = if fshape.pdim() == 1 {
            let d: [f64; 3] = [0, 1, 2].map(|c| dn.iter().zip(&xf).map(|(a, b)| a[0] * b[c]).sum());
            crate::elements::common::norm(d) * t_factor(se, pos, t)
        } else {
            let a: [f64; 3] = [0, 1, 2].map(|c| dn.iter().zip(&xf).map(|(d, b)| d[0] * b[c]).sum());
            let b: [f64; 3] = [0, 1, 2].map(|c| dn.iter().zip(&xf).map(|(d, b)| d[1] * b[c]).sum());
            crate::elements::common::norm(crate::elements::common::cross(a, b))
        };
        let mut full = np;
        full.resize(fl.len(), 0.0);
        debug_assert!(ncorner_face <= full.len());
        pts.push((full, jac * gw));
        Ok(())
    };
    if fshape.pdim() == 1 {
        for &(xi, w) in &g {
            push([xi, 0.0, 0.0], w)?;
        }
    } else {
        for &(eta, wj) in &g {
            for &(xi, wi) in &g {
                push([xi, eta, 0.0], wi * wj)?;
            }
        }
    }
    Ok(pts)
}

fn t_factor(se: &SoilElem, pos: [f64; 3], t: f64) -> f64 {
    match se.mode {
        crate::materials::StressMode::Axisymmetric => 2.0 * std::f64::consts::PI * pos[0],
        _ => t,
    }
}

impl NonlinearSystem for SoilSystem<'_, '_> {
    fn n_eq(&self) -> usize {
        self.dm.n_eq
    }

    fn eq_field(&self, eq: usize) -> Field {
        dof::field(SLOT_DOF[self.dm.owner[eq].1])
    }

    fn is_linear(&self) -> bool {
        self.linear
    }

    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        self.t0 = t0;
        self.t1 = t1;
        self.u.clone_from(&self.u_committed);
        for (n, s, v) in self.loads.bc_at(t1) {
            if self.dm.constrained[n] & (1 << s) != 0 {
                self.u[n][s] = v;
            }
        }
        Ok(())
    }

    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        let p = self.p;
        let m = p.model;
        let nn = m.nodes.len();
        let dt = self.dt();
        if self.transient && dt <= 0.0 {
            return fail("internal: consolidation with time increment 0");
        }
        let time = self.time0 + self.t1 * self.period;
        self.grav = self.fluid_gravity()?;
        let grav = &self.grav;
        if tangent {
            self.mat.zero();
        }
        let res = crate::par::map(m.elements.len(), crate::par::threads(), |e| -> Result<(Vec<f64>, Option<Vec<f64>>, Vec<f64>)> {
            let ctx = p.ctx(e, false, time, dt);
            let v = p.gather(e, &self.u);
            let v0 = p.gather(e, &self.u_committed);
            let mut sn = Vec::new();
            let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
            let r = match self.els.soil[e] {
                Some((se, pi)) => {
                    let fm = FluidMode { dt, transient: self.transient, gravity: grav[e] };
                    se.coupled(&ctx, &self.els.props[pi], &v, &v0, &self.state_old[e], &mut sn, fm, tangent).map_err(wrap)?
                }
                None => {
                    let dv: Vec<f64> = v.iter().zip(&v0).map(|(a, b)| a - b).collect();
                    m.elements[e].etype.internal_force(&ctx, &v, &dv, &self.state_old[e], &mut sn, tangent).map_err(wrap)?
                }
            };
            Ok((r.fint, r.k, sn))
        });
        let mut fint = vec![[0.0; SLOTS]; nn];
        let mut state_new = Vec::with_capacity(m.elements.len());
        for (e, r) in res.into_iter().enumerate() {
            let (fe, ke, sn) = r?;
            let dl = p.edofs[e].clone();
            for (a, &(n, s)) in dl.iter().enumerate() {
                fint[n][s] += fe[a];
            }
            if let Some(ke) = ke {
                self.add_matrix(&dl, &ke);
            }
            state_new.push(sn);
        }
        self.state_new = state_new;
        let soil_grav = |l: &DistLoad| l.kind == "GRAV" && self.els.soil[l.elem].is_some();
        let mech_dl: Vec<DistLoad> = self.loads.dloads_at(self.t1).into_iter().filter(|l| !is_flow(&l.kind) && !soil_grav(l)).collect();
        let mut fext = assembly::external(p, &self.loads.cloads_at(self.t1), &mech_dl, &self.u, false, time)?;
        self.gravity_load(&mut fext)?;
        for (dl, ke) in self.flow_terms(&mut fint, &mut fext, tangent)? {
            self.add_matrix(&dl, &ke);
        }
        self.fint = fint;
        self.fext = fext;
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            r[q] = self.row_scale(s) * (self.fext[n][s] - self.fint[n][s]);
        }
        let mut sum: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
        for n in 0..nn {
            for s in 0..SLOTS {
                if self.dm.active[n] & (1 << s) != 0 {
                    let e = sum.entry(dof::field(SLOT_DOF[s])).or_default();
                    let v = (self.fint[n][s].abs().max(self.fext[n][s].abs())) * self.row_scale(s).abs();
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
        // (D K D) x̂ = D r, x = D x̂
        let d: Vec<f64> = self.dm.owner.iter().map(|&(_, s)| if s == PS { self.pscale } else { 1.0 }).collect();
        let mut x: Vec<f64> = r.iter().zip(&d).map(|(a, b)| a * b).collect();
        self.mat.solve(&mut x);
        for (a, b) in x.iter_mut().zip(&d) {
            *a *= b;
        }
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
        std::mem::swap(&mut self.state_old, &mut self.state_new);
        self.state_new.clone_from(&self.state_old);
        Ok(())
    }
}

/// An increment [t0, t1] with fixed length (for the own time stepping).
struct One<'a, 'p, 'm> {
    sys: &'a mut SoilSystem<'p, 'm>,
    t0: f64,
    t1: f64,
    evals: usize,
    first: Option<(f64, f64)>,
}

impl NonlinearSystem for One<'_, '_, '_> {
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

/// Factor for cutback when UTOL is exceeded: Δt·0.85·UTOL/Δu_max.
const UTOL_CUT: f64 = 0.85;

/// Transient time stepping: fixed increments, UTOL and END=SS. The rule for UTOL
/// is the same as for DELTMX in `steps/heat.rs`:
/// f = 0.8·UTOL/Δu_max; the increment is redone with Δt·0.85·UTOL/Δu_max if
/// Δu_max > UTOL;
/// linear: the next Δt is increased by min(f, 2) if f ≥ 1.9; nonlinear: after two
/// consecutive accepted increments by min(f, g), g = 2 if the first solution
/// converged, otherwise 1.5. `on_increment(sys, t, dt)` is called after each
/// accepted increment. Returns (report, step time fraction at the end).
fn run_transient(
    sys: &mut SoilSystem,
    sc: &SoilControls,
    on_increment: &mut dyn FnMut(&mut SoilSystem, f64, f64) -> Result<()>,
) -> Result<(newton::Report, f64)> {
    let ctrl = &sc.ctrl;
    let mut rep = newton::Report::default();
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    let mut count: i64 = -1;
    let mut flux_sum = 0.0f64;
    let mut n_flux = 0usize;
    while t < 1.0 - 1e-12 {
        if rep.increments >= ctrl.max_increments {
            return fail(format!("the step reached the maximum number of increments ({}) – increase INC= on *STEP", ctrl.max_increments));
        }
        dt = dt.min(1.0 - t);
        if 1.0 - (t + dt) < 1e-9 {
            dt = 1.0 - t;
        }
        let mut tries = 0;
        let (dmax, first) = loop {
            let sub = Controls { initial: 1.0, fixed: true, ..ctrl.clone() };
            let mut one = One { sys: &mut *sys, t0: t, t1: t + dt, evals: 0, first: None };
            let r = newton::run(&mut one, &sub, &mut |_, _| Ok(()));
            let first = one.first;
            match r {
                Ok(r1) => {
                    rep.iterations += r1.iterations;
                    let dmax = sys.dm.owner.iter().filter(|&&(_, s)| s == PS).map(|&(n, s)| (sys.u[n][s] - sys.u_committed[n][s]).abs()).fold(0.0, f64::max);
                    if sc.utol > 0.0 && dmax > sc.utol * (1.0 + 1e-9) {
                        tries += 1;
                        if tries > ctrl.max_cutbacks || dt * UTOL_CUT * sc.utol / dmax < ctrl.min {
                            return fail(format!(
                                "UTOL={} cannot be satisfied at step time {:.4} without the increment becoming smaller than the minimum",
                                sc.utol,
                                t * ctrl.period
                            ));
                        }
                        dt *= UTOL_CUT * sc.utol / dmax;
                        count = -1;
                        continue;
                    }
                    break (dmax, first);
                }
                Err(e) => {
                    if ctrl.fixed {
                        return Err(e);
                    }
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
        on_increment(sys, t, dt)?;
        if sc.ss_rate > 0.0 && dmax / (dt * ctrl.period) < sc.ss_rate {
            return Ok((rep, t));
        }
        if sc.utol > 0.0 {
            count += 1;
            let f = if dmax > 0.0 { 0.8 * sc.utol / dmax } else { f64::INFINITY };
            if sys.linear {
                if f >= 1.9 {
                    dt *= f.min(2.0);
                }
            } else {
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
    }
    Ok((rep, 1.0))
}

impl StepType for Soils {
    fn name(&self) -> &'static str {
        "SOILS"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        matches!(step.procedure_name(), "SOILS" | "GEOSTATIC").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        let pname = ctx.step.procedure_name().to_string();
        if ctx.nlgeom {
            return fail(format!("*{pname} with NLGEOM=YES is not supported yet"));
        }
        if ctx.step.perturbation {
            return fail(format!("*{pname} cannot be a perturbation step"));
        }
        if !prep.interactions.is_empty() {
            return fail(format!("*{pname} together with contact/constraints is not supported yet"));
        }
        let mut sc = controls(ctx.step)?;
        let els = classify(prep)?;
        // *GEOSTATIC: with GRAV on the pore pressure elements (total pore pressure)
        // steady-state flow is solved; otherwise one transient increment with Δt =
        // the step time (the undrained response carries the imbalance with pore pressure).
        let soil_grav = ctx.loads.dload.iter().any(|(l, _)| l.kind == "GRAV" && els.soil[l.elem].is_some());
        if sc.geostatic && !soil_grav {
            sc.transient = true;
            sc.ctrl.fixed = true;
            sc.ctrl.initial = 1.0;
        }
        let had_init = ctx.state.vectors.contains_key(INIT_KEY);
        ensure_initial(prep, &els, ctx.state)?;
        let mut loads = ctx.loads.clone();
        loads.ramp = match ctx.step.header.param_upper("AMPLITUDE").as_deref() {
            Some("RAMP") => true,
            Some("STEP") => false,
            _ => !sc.transient,
        };
        if !had_init {
            // new boundary conditions on the pressure start from the initial pressure
            for ((n, s), v) in loads.bc.iter_mut() {
                if *s == PS && !ctx.state.bcs.contains_key(&(*n, *s)) {
                    v.0 = ctx.state.u[*n][PS];
                }
            }
        }
        let period = sc.ctrl.period;
        let mut sys = SoilSystem::new(prep, loads, els, ctx.state, sc.transient, period)?;
        let nn = model.nodes.len();
        let mut rvt: Vec<f64> = match ctx.state.vectors.get(RVT_KEY) {
            Some(v) => v.iter().map(|x| x[PS]).collect(),
            None => vec![0.0; nn],
        };
        let all = ctx.opts.all_frames;
        let verbose = ctx.opts.verbose;
        let mut frames = Vec::new();
        if all {
            sys.begin_increment(0.0, 0.0)?;
            sys.u.clone_from(&sys.u_committed);
            let tr = sys.transient;
            sys.transient = false;
            sys.evaluate(false)?;
            sys.transient = tr;
            let mut f0 = sys.frame(0.0, "Increment 0: Step Time = 0.0".into(), &rvt)?;
            // frame 0 reports the reactions from the previous step;
            // in the first step the internal forces from the initial state
            // and no reaction flow
            if had_init {
                f0.rf = ctx.state.reaction.clone();
            } else {
                for r in f0.rf.iter_mut() {
                    r[PS] = 0.0;
                }
            }
            let rf0 = f0.rf.clone();
            if let Some(x) = f0.node_extra.iter_mut().find(|x| x.0 == "RVF") {
                for (n, v) in x.2.iter_mut().enumerate() {
                    if !v.is_empty() {
                        v[0] = rf0[n][PS];
                    }
                }
            }
            frames.push(f0);
        }
        let mut inc = 0usize;
        let (rep, t_end) = if sc.transient {
            run_transient(&mut sys, &sc, &mut |s: &mut SoilSystem, t, dt| {
                inc += 1;
                // the geostatic step reports the reaction flow without the volume term
                // and sums no reaction volume
                if sc.geostatic {
                    s.transient = false;
                    s.evaluate(false)?;
                    s.transient = true;
                } else {
                    let rf = s.reactions();
                    for n in 0..nn {
                        if s.dm.constrained[n] & (1 << PS) != 0 {
                            rvt[n] += rf[n][PS] * dt * period;
                        }
                    }
                }
                if verbose {
                    eprintln!("  increment {inc}: step time {:.6}", t * period);
                }
                if all || t >= 1.0 - 1e-12 || sc.ss_rate > 0.0 {
                    frames.push(s.frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period), &rvt)?);
                }
                Ok(())
            })?
        } else {
            // steady-state: no reaction volume is summed (RVT unchanged)
            let rep = newton::run(&mut sys, &sc.ctrl, &mut |s: &mut SoilSystem, t| {
                inc += 1;
                if verbose {
                    eprintln!("  increment {inc}: step time {:.6}", t * period);
                }
                if all || t >= 1.0 - 1e-12 {
                    frames.push(s.frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period), &rvt)?);
                }
                Ok(())
            })?;
            (rep, 1.0)
        };
        if !all && sc.ss_rate > 0.0 && frames.len() > 1 {
            // END=SS: only the last frame
            let last = frames.pop();
            frames.clear();
            frames.extend(last);
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = rep.increments;
        ctx.out.iterations = rep.iterations;
        ctx.state.u = sys.u_committed.clone();
        ctx.state.reaction = sys.reactions();
        ctx.state.elem_state = sys.state_old.clone();
        ctx.state.time += t_end * period;
        let mut rv = vec![[0.0; SLOTS]; nn];
        for n in 0..nn {
            rv[n][PS] = rvt[n];
        }
        ctx.state.vectors.insert(RVT_KEY.into(), rv);
        Ok(())
    }
}
