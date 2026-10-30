//! Removed and reactivated elements (*MODEL CHANGE) and element deletion
//! on full damage (*SECTION CONTROLS, ELEMENT DELETION=, MAX DEGRADATION=).
//!
//! No step type of its own: the module gives the element loops in `fem/assembly.rs`,
//! `steps/nlgeom.rs` and `steps/explicit.rs` a state per element
//! (`Prepared::activity`) that they query via `internal_force`, `output` and
//! `interaction_off`. `steps::run` calls `begin_analysis` and `begin_step`.
//!
//! *MODEL CHANGE, REMOVE (implicit analysis):
//! - the element's nodal forces at the start of the step are stored and ramped linearly down
//!   to zero during the step (the removal step); after that the element
//!   does not contribute at all. No element computations are done for removed elements.
//! - nodes that only belong to removed elements stay still (the element gives a
//!   negligible diagonal stiffness ε·K₀ᵢᵢ, ε = 1e-12, so that the system of equations does not
//!   become singular; the residual is not affected).
//! - removed elements have no output, and distributed loads on them are
//!   inactive.
//!
//! *MODEL CHANGE, ADD=STRAIN FREE (default): the element becomes fully active
//! directly in an "annealed" state (zero stress, strain and plastic
//! strain) with the configuration at the start of the step as the new reference: the element
//! sees the displacement since the reactivation and, with NLGEOM, the current
//! nodal positions as reference coordinates.
//!
//! *MODEL CHANGE, ADD=WITH STRAIN: annealed state, but the element sees
//! uₑ = α(t)·u (u = the displacement from the reference, normally the original
//! configuration) where α is ramped linearly from 0 to 1 during the step, and
//! the element stiffness is multiplied by α. After the step the
//! strains are the total ones.
//!
//! *MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE/ADD: the contact pair (slave surface,
//! master surface) does not contribute as long as it is removed.
//!
//! Element deletion on damage (ductile damage with *DAMAGE EVOLUTION):
//! By default the element is removed when the damage D reaches Dmax = 1 at
//! any integration point (ELEMENT DELETION=YES; with ELEMENT DELETION=NO
//! Dmax = 0.99 and the element stays). MAX DEGRADATION= gives Dmax. With
//! deletion D is limited to min(Dmax, 0.99999). The check
//! is made on the accepted state, so the element is removed after the increment
//! where D reaches Dmax. A removed element gives no force (only the ε stiffness) and
//! writes STATUS = 0 (already in the increment where D reaches Dmax); active elements
//! with damage evolution write STATUS = 1. The stress in removed elements
//! is reported as zero from the increment after the removal onwards.

use super::api::StepType;
use crate::elements::{ElemCtx, ElemResponse, IpField};
use crate::fem::assembly::Prepared;
use crate::fem::model::{FeModel, RawKeyword, StepDef};
use crate::fem::state::SolutionState;
use crate::model::{FemError, Result};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;

pub static STEP_TYPES: &[&dyn StepType] = &[];

/// Relative stiffness for nodes that only belong to removed elements.
const EPS: f64 = 1e-12;

thread_local! {
    static DMAX: Cell<Option<f64>> = const { Cell::new(None) };
}

/// Upper limit for the damage D in the element currently being computed (set around
/// the element call for elements with *SECTION CONTROLS/element deletion), or
/// `None` (the material default, 0.99).
pub fn max_degradation() -> Option<f64> {
    DMAX.with(|d| d.get())
}

#[derive(Debug, Clone, Default)]
enum Mode {
    #[default]
    Active,
    /// Removed: nodal forces at removal, ramped down from `t0` over `period`.
    Removed { fr: Vec<f64>, t0: f64, period: f64 },
    /// Reactivated without strain: reference displacement and (NLGEOM) reference coordinates.
    StrainFree { uref: Vec<f64>, xref: Option<Vec<[f64; 3]>> },
    /// Reactivated with strain: uₑ = α(t)(u − uref), α from 0 (at t0) to 1.
    WithStrain { uref: Vec<f64>, xref: Option<Vec<[f64; 3]>>, t0: f64, period: f64 },
}

#[derive(Debug, Clone, Copy)]
struct Deletion {
    /// The element is removed when D reaches `dmax`.
    delete: bool,
    dmax: f64,
}

#[derive(Default)]
struct Inner {
    modes: Vec<Mode>,
    deletion: Vec<Option<Deletion>>,
    /// Diagonal scale ε·K₀ᵢᵢ per element (computed when needed).
    eps: Vec<Option<Vec<f64>>>,
    /// Removed contact pairs (slave surface, master surface) in upper case.
    contact_off: Vec<(String, String)>,
}

/// Element states during the analysis (field in `Prepared`).
#[derive(Default)]
pub struct Activity {
    any: AtomicBool,
    inner: RwLock<Inner>,
    deleted: RwLock<Vec<AtomicBool>>,
    checked: RwLock<Vec<AtomicU64>>,
    /// Explicit analysis: the element is removed only when all integration points
    /// have reached Dmax (set by `steps/explicit.rs`, `set_explicit_rule`).
    all_ips: AtomicBool,
}

impl Activity {
    /// Some element is removed, reactivated or can be removed.
    pub fn any(&self) -> bool {
        self.any.load(Ordering::Relaxed)
    }

    /// The element is removed (*MODEL CHANGE or damage).
    pub fn is_removed(&self, e: usize) -> bool {
        if !self.any() {
            return false;
        }
        if self.is_deleted(e) {
            return true;
        }
        let inner = self.inner.read().unwrap();
        matches!(inner.modes.get(e), Some(Mode::Removed { .. }))
    }

    /// The element is removed because of damage.
    pub fn is_deleted(&self, e: usize) -> bool {
        self.deleted.read().unwrap().get(e).is_some_and(|d| d.load(Ordering::Relaxed))
    }

    /// The deletion rule in explicit analysis: a continuum element is removed when
    /// the damage has reached Dmax at all integration points; implicit analysis: at any point.
    pub fn set_explicit_rule(&self, on: bool) {
        self.all_ips.store(on, Ordering::Relaxed);
    }

    fn rule_all(&self) -> bool {
        self.all_ips.load(Ordering::Relaxed)
    }

    /// Number of elements removed because of damage.
    pub fn n_deleted(&self) -> usize {
        self.deleted.read().unwrap().iter().filter(|d| d.load(Ordering::Relaxed)).count()
    }
}

fn flag_value(kw: &RawKeyword, key: &str, default: bool) -> Result<bool> {
    kw.flag(key, default)
}

/// Damage evolution and *SECTION CONTROLS per element.
fn deletion_settings(m: &FeModel) -> Result<Vec<Option<Deletion>>> {
    let mut per_section: Vec<Option<Deletion>> = Vec::with_capacity(m.sections.len());
    for s in &m.sections {
        let damage = s.material_index.is_some_and(|i| m.materials[i].behavior("DAMAGE EVOLUTION").is_some());
        // *TENSILE FAILURE, ELEMENT DELETION=YES (default): the material writes SDEG = 1 (materials/eos.rs)
        let damage = damage || s.material_index.is_some_and(|i| m.materials[i].behavior("TENSILE FAILURE").is_some_and(|t| !matches!(t.param_upper("ELEMENT DELETION").as_deref(), Some("NO"))));
        let controls = match s.raw.param("CONTROLS") {
            Some(name) => {
                let name = name.trim().to_uppercase();
                let Some(sc) = m.extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())) else {
                    return s.raw.err(format!("*SECTION CONTROLS, NAME={name} is missing"));
                };
                Some(sc)
            }
            None => None,
        };
        let mut d = None;
        // cohesive elements handle their own deletion (elements/cohesive.rs)
        let cohesive = s.raw.name == "COHESIVE SECTION";
        if damage && !cohesive {
            let delete = match controls {
                Some(sc) => flag_value(sc, "ELEMENT DELETION", true)?,
                None => true,
            };
            let dmax = match controls.and_then(|sc| sc.param("MAX DEGRADATION")) {
                Some(v) => {
                    let x = crate::fem::model::parse_f64(v).ok_or_else(|| FemError(format!("{}: invalid MAX DEGRADATION={v}", controls.unwrap().location())))?;
                    if !(x > 0.0 && x <= 1.0) {
                        return controls.unwrap().err("MAX DEGRADATION must lie in (0, 1]");
                    }
                    x
                }
                None if delete => 1.0,
                None => 0.99,
            };
            // with deletion D is limited to at most 0.99999 (the stress 1e-5·σ̄
            // in the increment where the element is removed)
            let dmax = if delete { dmax.min(0.99999) } else { dmax };
            d = Some(Deletion { delete, dmax });
        }
        per_section.push(d);
    }
    Ok(m.elements.iter().map(|e| e.section.and_then(|s| per_section[s])).collect())
}

/// Resets the state at the start of the analysis and reads the element deletion.
pub fn begin_analysis(p: &Prepared) -> Result<()> {
    let m = p.model;
    let ne = m.elements.len();
    let deletion = deletion_settings(m)?;
    let any_del = deletion.iter().any(|d| d.is_some());
    let mut inner = p.activity.inner.write().unwrap();
    *inner = Inner { modes: vec![Mode::Active; ne], deletion, eps: vec![None; ne], contact_off: Vec::new() };
    *p.activity.deleted.write().unwrap() = (0..ne).map(|_| AtomicBool::new(false)).collect();
    *p.activity.checked.write().unwrap() = (0..ne).map(|_| AtomicU64::new(u64::MAX)).collect();
    let uses_mc = m.steps.iter().any(|s| s.extra.iter().any(|k| k.name == "MODEL CHANGE"));
    p.activity.any.store(any_del || uses_mc, Ordering::Relaxed);
    Ok(())
}

/// The elements in *MODEL CHANGE (indices stored by the keyword handler as "#index").
fn kw_elements(kw: &RawKeyword) -> Vec<usize> {
    kw.data.iter().flat_map(|d| d.fields.iter()).filter_map(|f| f.trim().strip_prefix('#').and_then(|s| s.parse().ok())).collect()
}

fn ctx_with_x<'a>(ctx: &ElemCtx<'a>, x: &Option<Vec<[f64; 3]>>) -> ElemCtx<'a> {
    let mut c = ctx.clone();
    if let Some(x) = x {
        c.x = x.clone();
    }
    c
}

/// *MODEL CHANGE in the step: removes and reactivates elements and contact pairs.
pub fn begin_step(p: &Prepared, step: &StepDef, state: &mut SolutionState) -> Result<()> {
    let kws: Vec<&RawKeyword> = step.extra.iter().filter(|k| k.name == "MODEL CHANGE").collect();
    if kws.is_empty() {
        return Ok(());
    }
    if step.perturbation {
        return kws[0].err("*MODEL CHANGE cannot be used in a perturbation step");
    }
    let m = p.model;
    let period = super::newton::Controls::from_procedure(step.procedure.as_ref(), step.max_increments).map(|c| c.period).unwrap_or(1.0);
    let nlgeom = step.nlgeom.unwrap_or(state.nlgeom);
    let mut inner = p.activity.inner.write().unwrap();
    for kw in kws {
        let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ELEMENT".into());
        let remove = kw.has("REMOVE");
        let add = kw.has("ADD");
        if remove == add {
            return kw.err("specify REMOVE or ADD");
        }
        if ty == "CONTACT PAIR" {
            for d in kw.records() {
                let f: Vec<String> = d.fields.iter().map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).collect();
                if f.is_empty() {
                    continue;
                }
                if f.len() < 2 {
                    return kw.err_at(d.line, "write slave surface, master surface");
                }
                let pair = (f[0].clone(), f[1].clone());
                let known = p.interactions.iter().any(|it| contact_pair_of(&it.name()).is_some_and(|q| q == pair));
                if !known {
                    return kw.err_at(d.line, format!("contact pair {}, {} does not exist", f[0], f[1]));
                }
                inner.contact_off.retain(|q| *q != pair);
                if remove {
                    inner.contact_off.push(pair);
                }
            }
            continue;
        }
        if ty != "ELEMENT" {
            return kw.err(format!("TYPE={ty} is not supported (ELEMENT, CONTACT PAIR)"));
        }
        let with_strain = match kw.param_upper("ADD").as_deref() {
            None | Some("") | Some("STRAIN FREE") => false,
            Some("WITH STRAIN") => true,
            Some(o) => return kw.err(format!("ADD={o} is not supported (STRAIN FREE, WITH STRAIN)")),
        };
        for e in kw_elements(kw) {
            let el = &m.elements[e];
            if crate::elements::rigid::is_rigid(el.etype) {
                return kw.err(format!("rigid elements cannot be removed ({})", m.elem_name(e)));
            }
            let ue = p.gather(e, &state.u);
            let cur = std::mem::take(&mut inner.modes[e]);
            if remove {
                inner.modes[e] = match cur {
                    Mode::Removed { .. } => cur,
                    _ => {
                        // nodal forces towards the surroundings at the start of the step
                        let ctx0 = p.ctx(e, nlgeom, state.time, 0.0);
                        let (ctx, ue_e) = match &cur {
                            Mode::StrainFree { uref, xref } | Mode::WithStrain { uref, xref, .. } => (ctx_with_x(&ctx0, xref), ue.iter().zip(uref).map(|(a, b)| a - b).collect()),
                            _ => (ctx0, ue.clone()),
                        };
                        let zero = vec![0.0; ue_e.len()];
                        let mut fr = if ctx.material.is_none() && ctx.section.and_then(|s| s.beam()).is_none() {
                            zero.clone()
                        } else {
                            let mut sn = Vec::new();
                            el.etype.internal_force(&ctx, &ue_e, &zero, &state.elem_state[e], &mut sn, false).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?.fint
                        };
                        // net force towards the surroundings: the distributed loads on the element
                        // disappear with it (nodes that only belong to removed elements then get
                        // no force)
                        for l in state.dloads.values().filter(|l| l.elem == e) {
                            let fe = el.etype.distributed_load(&ctx, l, &ue).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
                            for (a, v) in fr.iter_mut().zip(fe) {
                                *a -= v;
                            }
                        }
                        Mode::Removed { fr, t0: state.time, period }
                    }
                };
            } else {
                if !matches!(cur, Mode::Removed { .. }) {
                    // already active: no change (with a warning)
                    inner.modes[e] = cur;
                    continue;
                }
                // annealed state
                let ctx = p.ctx(e, false, 0.0, 0.0);
                if ctx.material.is_some() {
                    state.elem_state[e] = el.etype.init_state(&ctx).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
                }
                // strain free: the current configuration is the new reference
                let (uref, xref) = if with_strain {
                    (vec![0.0; ue.len()], None)
                } else {
                    let xref = nlgeom.then(|| el.nodes.iter().map(|&n| std::array::from_fn(|c| m.nodes[n][c] + state.u[n][c])).collect());
                    (ue, xref)
                };
                inner.modes[e] = if with_strain { Mode::WithStrain { uref, xref, t0: state.time, period } } else { Mode::StrainFree { uref, xref } };
            }
        }
    }
    // nodes that only belong to removed elements get no force
    let mut alive = vec![false; m.nodes.len()];
    for (e, el) in m.elements.iter().enumerate() {
        if !matches!(inner.modes[e], Mode::Removed { .. }) {
            for &n in &el.nodes {
                alive[n] = true;
            }
        }
    }
    for e in 0..m.elements.len() {
        if let Mode::Removed { fr, .. } = &mut inner.modes[e] {
            for (a, &(n, _)) in p.edofs[e].iter().enumerate() {
                if !alive[n] {
                    fr[a] = 0.0;
                }
            }
        }
    }
    p.activity.any.store(true, Ordering::Relaxed);
    Ok(())
}

/// (slave surface, master surface) for a contact pair from the interaction name.
fn contact_pair_of(name: &str) -> Option<(String, String)> {
    let rest = name.strip_prefix("*CONTACT PAIR ")?;
    let pair = rest.split(" (").next()?;
    let (a, b) = pair.split_once('/')?;
    Some((a.trim().to_uppercase(), b.trim().to_uppercase()))
}

/// The interaction is a removed contact pair.
pub fn interaction_off(p: &Prepared, name: &str) -> bool {
    if !p.activity.any() {
        return false;
    }
    let inner = p.activity.inner.read().unwrap();
    if inner.contact_off.is_empty() {
        return false;
    }
    contact_pair_of(name).is_some_and(|q| inner.contact_off.contains(&q))
}

/// The element does not receive distributed loads (removed).
pub fn load_off(p: &Prepared, e: usize) -> bool {
    p.activity.is_removed(e)
}

fn hash(v: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for x in v {
        h ^= x.to_bits();
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^ (v.len() as u64)
}

/// Largest damage (SDEG) per integration point: all section points at some
/// integration point ≥ dmax.
fn fully_damaged(fields: &[IpField], dmax: f64, all: bool) -> bool {
    let Some(f) = fields.iter().find(|f| f.name == "SDEG") else { return false };
    let mut per_ip: std::collections::BTreeMap<usize, bool> = std::collections::BTreeMap::new();
    for (ip, _sp, v) in &f.points {
        let full = v.first().is_some_and(|&d| d >= dmax * (1.0 - 1e-9));
        let e = per_ip.entry(*ip).or_insert(true);
        *e &= full;
    }
    if all {
        !per_ip.is_empty() && per_ip.values().all(|&b| b)
    } else {
        per_ip.values().any(|&b| b)
    }
}

/// ε·K₀ᵢᵢ for the element (elastic stiffness in the original position).
fn eps_diag(p: &Prepared, e: usize, ctx: &ElemCtx, n: usize) -> Vec<f64> {
    if let Some(Some(d)) = p.activity.inner.read().unwrap().eps.get(e) {
        return d.clone();
    }
    let el = &p.model.elements[e];
    let mut c = ctx.clone();
    c.nlgeom = false;
    let d = (|| -> Option<Vec<f64>> {
        let st = if c.material.is_some() { el.etype.init_state(&c).ok()? } else { Vec::new() };
        let zero = vec![0.0; n];
        let mut sn = Vec::new();
        let r = el.etype.internal_force(&c, &zero, &zero, &st, &mut sn, true).ok()?;
        let k = r.k?;
        let kmax = (0..n).map(|i| k[i * n + i].abs()).fold(0.0, f64::max);
        Some((0..n).map(|i| EPS * k[i * n + i].abs().max(1e-6 * kmax).max(1e-300)).collect())
    })()
    .unwrap_or_else(|| vec![EPS; n]);
    if let Ok(mut inner) = p.activity.inner.write() {
        if let Some(slot) = inner.eps.get_mut(e) {
            *slot = Some(d.clone());
        }
    }
    d
}

fn eps_response(p: &Prepared, e: usize, ctx: &ElemCtx, n: usize, fint: Vec<f64>, tangent: bool) -> ElemResponse {
    let k = tangent.then(|| {
        let d = eps_diag(p, e, ctx, n);
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            k[i * n + i] = d[i];
        }
        k
    });
    ElemResponse { fint, k }
}

fn ramp(time: f64, t0: f64, period: f64) -> f64 {
    if period > 0.0 {
        ((time - t0) / period).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// The element's contribution when it is removed, reactivated or can be removed
/// on damage; `None` = ordinary element (the caller computes it itself).
#[allow(clippy::too_many_arguments)]
pub fn internal_force(
    p: &Prepared,
    e: usize,
    ctx: &ElemCtx,
    ue: &[f64],
    due: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
) -> Option<Result<ElemResponse>> {
    let act = &p.activity;
    if !act.any() {
        return None;
    }
    let n = ue.len();
    let et = p.model.elements[e].etype;
    if act.is_deleted(e) {
        state_new.clear();
        state_new.extend_from_slice(state_old);
        return Some(Ok(eps_response(p, e, ctx, n, vec![0.0; n], tangent)));
    }
    let (mode, del) = {
        let inner = act.inner.read().unwrap();
        (inner.modes.get(e).cloned().unwrap_or_default(), inner.deletion.get(e).copied().flatten())
    };
    if matches!(mode, Mode::Active) && del.is_none() {
        return None;
    }
    // element deletion: checked on the accepted state
    if let Some(d) = del.filter(|d| d.delete) {
        let h = hash(state_old);
        let checked = act.checked.read().unwrap();
        if let Some(c) = checked.get(e) {
            if c.load(Ordering::Relaxed) != h {
                c.store(h, Ordering::Relaxed);
                let full = et.output(ctx, ue, state_old).map(|f| fully_damaged(&f, d.dmax, act.rule_all())).unwrap_or(false);
                if full {
                    if let Some(x) = act.deleted.read().unwrap().get(e) {
                        x.store(true, Ordering::Relaxed);
                    }
                    state_new.clear();
                    state_new.extend_from_slice(state_old);
                    return Some(Ok(eps_response(p, e, ctx, n, vec![0.0; n], tangent)));
                }
            }
        }
    }
    let call = |c: &ElemCtx, u: &[f64], du: &[f64], sn: &mut Vec<f64>| -> Result<ElemResponse> {
        DMAX.with(|x| x.set(del.map(|d| d.dmax)));
        let r = et.internal_force(c, u, du, state_old, sn, tangent);
        DMAX.with(|x| x.set(None));
        r
    };
    Some(match mode {
        Mode::Active => call(ctx, ue, due, state_new),
        Mode::Removed { fr, t0, period } => {
            state_new.clear();
            state_new.extend_from_slice(state_old);
            let f = 1.0 - ramp(ctx.time, t0, period);
            Ok(eps_response(p, e, ctx, n, fr.iter().map(|x| f * x).collect(), tangent))
        }
        Mode::StrainFree { uref, xref } => {
            let c = ctx_with_x(ctx, &xref);
            let u: Vec<f64> = ue.iter().zip(&uref).map(|(a, b)| a - b).collect();
            call(&c, &u, due, state_new)
        }
        Mode::WithStrain { uref, xref, t0, period } => {
            let c = ctx_with_x(ctx, &xref);
            let a1 = ramp(ctx.time, t0, period);
            let a0 = ramp(ctx.time - ctx.dt, t0, period);
            let u: Vec<f64> = ue.iter().zip(&uref).map(|(a, b)| a1 * (a - b)).collect();
            let du: Vec<f64> = (0..n).map(|i| a1 * (ue[i] - uref[i]) - a0 * (ue[i] - due[i] - uref[i])).collect();
            call(&c, &u, &du, state_new).map(|mut r| {
                if let Some(k) = r.k.as_mut() {
                    for v in k.iter_mut() {
                        *v *= a1;
                    }
                    if a1 == 0.0 {
                        let d = eps_diag(p, e, ctx, n);
                        for i in 0..n {
                            k[i * n + i] = d[i];
                        }
                    }
                }
                r
            })
        }
    })
}

/// The element's output when it is removed, reactivated or can be removed;
/// `None` = ordinary element.
pub fn output(p: &Prepared, e: usize, ctx: &ElemCtx, ue: &[f64], state: &[f64]) -> Option<Result<Vec<IpField>>> {
    let act = &p.activity;
    if !act.any() {
        return None;
    }
    let et = p.model.elements[e].etype;
    let (mode, del) = {
        let inner = act.inner.read().unwrap();
        (inner.modes.get(e).cloned().unwrap_or_default(), inner.deletion.get(e).copied().flatten())
    };
    if matches!(mode, Mode::Active) && del.is_none() {
        return None;
    }
    if matches!(mode, Mode::Removed { .. }) {
        // removed elements have no output
        return Some(Ok(Vec::new()));
    }
    let res = match &mode {
        Mode::StrainFree { uref, xref } | Mode::WithStrain { uref, xref, .. } => {
            let c = ctx_with_x(ctx, xref);
            let u: Vec<f64> = ue.iter().zip(uref).map(|(a, b)| a - b).collect();
            et.output(&c, &u, state)
        }
        _ => et.output(ctx, ue, state),
    };
    let Some(d) = del.filter(|d| d.delete) else { return Some(res) };
    Some(res.map(|mut fields| {
        // removed on damage: STATUS = 0 from the increment where D reaches Dmax,
        // the stress zero from the next increment onwards
        let deleted = act.is_deleted(e);
        let gone = deleted || fully_damaged(&fields, d.dmax, act.rule_all());
        let status = if gone { 0.0 } else { 1.0 };
        // STATUS applies to the whole element (point 0)
        let mut st = IpField::new("STATUS", &["STATUS"]);
        if deleted {
            for f in fields.iter_mut().filter(|f| f.name == "S") {
                for pt in f.points.iter_mut() {
                    for v in pt.2.iter_mut() {
                        *v = 0.0;
                    }
                }
            }
        }
        st.points = vec![(0, 0, vec![status])];
        fields.push(st);
        fields
    }))
}
