//! Solution state inherited between steps, and loads/boundary conditions for a step.
//!
//! Rules (for general steps):
//! - Boundary conditions and loads are inherited from the previous step. OP=MOD (default)
//!   modifies or adds entries; OP=NEW first removes all earlier
//!   entries of the same kind.
//! - Within a step the values are ramped linearly from the previous step's values to
//!   the new ones (AMPLITUDE=RAMP, default for statics), or applied immediately
//!   (AMPLITUDE=STEP).
//! - A removed boundary condition is replaced by its reaction force, which is ramped down
//!   to zero during the step.
//! - A new prescribed value is ramped from the node's current displacement.
//!
//! Linear perturbation steps (PERTURBATION, FREQUENCY, BUCKLE) start from the
//! base state: only the step's own loads apply, inherited boundary conditions have
//! the value zero, and the state after the step is reset to the base state.

use crate::fem::dof::{self, NodeVec};
use crate::fem::model::{DistLoad, FeModel, StepDef};
use crate::model::{fail, Result};
use std::collections::{BTreeMap, BTreeSet};

/// State inherited from step to step.
#[derive(Debug, Clone, Default)]
pub struct SolutionState {
    /// Total nodal values (displacement, rotation, temperature, ...).
    pub u: Vec<NodeVec>,
    /// Accepted element state (stresses, material state).
    pub elem_state: Vec<Vec<f64>>,
    /// Total time at the end of the latest step.
    pub time: f64,
    /// NLGEOM is switched on (inherited by subsequent steps).
    pub nlgeom: bool,
    /// Current boundary conditions (node, slot) → value at the end of the step.
    pub bcs: BTreeMap<(usize, usize), f64>,
    pub cloads: BTreeMap<(usize, usize), f64>,
    pub dloads: BTreeMap<(usize, String), DistLoad>,
    /// Reactions at the end of the latest step.
    pub reaction: Vec<NodeVec>,
    /// Other nodal vectors that step types need to inherit (e.g. "V", "A").
    pub vectors: BTreeMap<String, Vec<NodeVec>>,
}

impl SolutionState {
    pub fn new(model: &FeModel) -> Self {
        SolutionState { u: vec![[0.0; dof::SLOTS]; model.nodes.len()], reaction: vec![[0.0; dof::SLOTS]; model.nodes.len()], ..Default::default() }
    }
}

/// Boundary conditions and loads during a step: (value at the start of the step, at the end).
#[derive(Debug, Clone, Default)]
pub struct StepLoads {
    pub bc: BTreeMap<(usize, usize), (f64, f64)>,
    pub cload: BTreeMap<(usize, usize), (f64, f64)>,
    /// Distributed loads with magnitude at the end (`magnitude`) and at the start.
    pub dload: Vec<(DistLoad, f64)>,
    /// Reactions from removed boundary conditions: (node, slot, force at the start of the step).
    pub released: Vec<(usize, usize, f64)>,
    /// Linear ramp over the step (otherwise step amplitude).
    pub ramp: bool,
    pub perturbation: bool,
    /// Amplitude references (AMPLITUDE=) for entries defined in the step.
    /// Only filled by `build_with` with `allow_amplitudes`; a step type that
    /// reads them computes the value as magnitude · A(t) (see `steps/dynamic_amplitude.rs`).
    pub amp_bc: BTreeMap<(usize, usize), String>,
    pub amp_cload: BTreeMap<(usize, usize), String>,
    /// Parallel to `dload`.
    pub amp_dload: Vec<Option<String>>,
    /// *BOUNDARY, USER: boundary conditions whose value is given by DISP (`user/loads.rs`).
    pub user_bc: Vec<crate::user::loads::UserBc>,
}

impl StepLoads {
    /// Amplitude at the step-time fraction `t` ∈ [0, 1].
    pub fn amp(&self, t: f64) -> f64 {
        if self.ramp {
            t.clamp(0.0, 1.0)
        } else if t > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn lerp(&self, (a, b): (f64, f64), t: f64) -> f64 {
        a + (b - a) * self.amp(t)
    }

    pub fn constrained(&self) -> BTreeSet<(usize, usize)> {
        self.bc.keys().copied().collect()
    }

    /// Prescribed values at `t`.
    pub fn bc_at(&self, t: f64) -> Vec<(usize, usize, f64)> {
        let mut v: Vec<(usize, usize, f64)> = self.bc.iter().map(|(&(n, s), &v)| (n, s, self.lerp(v, t))).collect();
        if !self.user_bc.is_empty() {
            let user: BTreeMap<(usize, usize), f64> = crate::user::loads::disp_values(&self.user_bc, t).into_iter().map(|(n, s, x)| ((n, s), x)).collect();
            for (n, s, x) in v.iter_mut() {
                if let Some(u) = user.get(&(*n, *s)) {
                    *x = *u;
                }
            }
        }
        v
    }

    /// Point loads at `t` (including ramped reactions from removed boundary conditions).
    pub fn cloads_at(&self, t: f64) -> Vec<(usize, usize, f64)> {
        let mut v: Vec<(usize, usize, f64)> = self.cload.iter().map(|(&(n, s), &c)| (n, s, self.lerp(c, t))).collect();
        for &(n, s, r) in &self.released {
            v.push((n, s, -r * (1.0 - self.amp(t))));
        }
        v
    }

    pub fn dloads_at(&self, t: f64) -> Vec<DistLoad> {
        self.dload
            .iter()
            .map(|(l, start)| {
                let mut l = l.clone();
                l.magnitude = self.lerp((*start, l.magnitude), t);
                l
            })
            .collect()
    }

    /// Builds the step's loads and boundary conditions from the state before the step.
    /// Also returns the new (inherited) values that apply after the step.
    pub fn build(model: &FeModel, step: &StepDef, state: &SolutionState, active: &[u16]) -> Result<(StepLoads, Propagated)> {
        Self::build_with(model, step, state, active, false)
    }

    /// Like `build`. With `allow_amplitudes`, AMPLITUDE= is accepted on boundary conditions and
    /// loads; the references are stored in `amp_bc`/`amp_cload`/`amp_dload` (the step type
    /// must then evaluate the amplitudes itself).
    pub fn build_with(model: &FeModel, step: &StepDef, state: &SolutionState, active: &[u16], allow_amplitudes: bool) -> Result<(StepLoads, Propagated)> {
        let mut amp_bc: BTreeMap<(usize, usize), String> = BTreeMap::new();
        let mut amp_cl: BTreeMap<(usize, usize), String> = BTreeMap::new();
        let mut amp_dl: BTreeMap<(usize, String), String> = BTreeMap::new();
        let mut warn = Vec::new();
        let on = |n: usize, s: usize| active.get(n).is_some_and(|a| a & (1 << s) != 0);
        let mut bcs = if step.perturbation { state.bcs.keys().map(|&k| (k, 0.0)).collect() } else { state.bcs.clone() };
        let mut cl: BTreeMap<(usize, usize), f64> = if step.perturbation { BTreeMap::new() } else { state.cloads.clone() };
        let mut dl: BTreeMap<(usize, String), DistLoad> = if step.perturbation { BTreeMap::new() } else { state.dloads.clone() };
        for b in &step.bcs {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= is not supported yet", b.source));
            }
            if !(b.kind.is_empty() || b.kind == "DISPLACEMENT") {
                return fail(format!("{}: TYPE={} is not supported in this step", b.source, b.kind));
            }
            if b.op_new {
                bcs.clear();
                amp_bc.clear();
            }
            for &(n, d, v) in &b.items {
                let s = dof::slot_of(d);
                if on(n, s) {
                    bcs.insert((n, s), v);
                    match &b.amplitude {
                        Some(a) => amp_bc.insert((n, s), a.clone()),
                        None => amp_bc.remove(&(n, s)),
                    };
                } else if v != 0.0 {
                    warn.push(format!("{}: node {} has no degree of freedom {} – the boundary condition is ignored", b.source, model.node_name(n), d));
                }
            }
        }
        for b in &step.cloads {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= is not supported yet", b.source));
            }
            if b.follower {
                return fail(format!("{}: FOLLOWER=YES is not supported yet", b.source));
            }
            if b.op_new {
                cl.clear();
                amp_cl.clear();
            }
            // entries in the same block are summed and replace the previous step's value
            let mut block: BTreeMap<(usize, usize), f64> = BTreeMap::new();
            for &(n, d, v) in &b.items {
                let s = dof::slot_of(d);
                if !on(n, s) {
                    return fail(format!("{}: node {} has no degree of freedom {d}", b.source, model.node_name(n)));
                }
                *block.entry((n, s)).or_default() += v;
            }
            for k in block.keys() {
                match &b.amplitude {
                    Some(a) => amp_cl.insert(*k, a.clone()),
                    None => amp_cl.remove(k),
                };
            }
            cl.extend(block);
        }
        for b in &step.dloads {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= is not supported yet", b.source));
            }
            if b.op_new {
                dl.clear();
                amp_dl.clear();
            }
            let mut block: BTreeMap<(usize, String), DistLoad> = BTreeMap::new();
            for l in &b.items {
                let key = (l.elem, l.kind.clone());
                match block.get_mut(&key) {
                    Some(old) if old.extra == l.extra => old.magnitude += l.magnitude,
                    Some(_) => return fail(format!("{}: two {}-loads with different directions on the same element", b.source, l.kind)),
                    None => {
                        block.insert(key, l.clone());
                    }
                }
            }
            for k in block.keys() {
                match &b.amplitude {
                    Some(a) => amp_dl.insert(k.clone(), a.clone()),
                    None => amp_dl.remove(k),
                };
            }
            dl.extend(block);
        }
        let mut loads = StepLoads { ramp: !step.amplitude_step, perturbation: step.perturbation, ..Default::default() };
        for (&k, &v) in &bcs {
            let start = if step.perturbation {
                0.0
            } else {
                state.bcs.get(&k).copied().unwrap_or_else(|| state.u.get(k.0).map_or(0.0, |u| u[k.1]))
            };
            loads.bc.insert(k, (start, v));
        }
        if !step.perturbation {
            for (&k, _) in &state.bcs {
                if !bcs.contains_key(&k) {
                    loads.released.push((k.0, k.1, state.reaction.get(k.0).map_or(0.0, |r| r[k.1])));
                }
            }
        }
        let keys: BTreeSet<(usize, usize)> = cl.keys().chain(if step.perturbation { None } else { Some(state.cloads.keys()) }.into_iter().flatten()).copied().collect();
        for k in keys {
            let start = if step.perturbation { 0.0 } else { state.cloads.get(&k).copied().unwrap_or(0.0) };
            loads.cload.insert(k, (start, cl.get(&k).copied().unwrap_or(0.0)));
        }
        let mut dkeys: BTreeSet<(usize, String)> = dl.keys().cloned().collect();
        if !step.perturbation {
            dkeys.extend(state.dloads.keys().cloned());
        }
        for k in dkeys {
            let start = if step.perturbation { 0.0 } else { state.dloads.get(&k).map_or(0.0, |l| l.magnitude) };
            let end = match (dl.get(&k), state.dloads.get(&k)) {
                (Some(l), _) => l.clone(),
                (None, Some(old)) => DistLoad { magnitude: 0.0, ..old.clone() },
                (None, None) => unreachable!(),
            };
            if let (Some(old), false) = (state.dloads.get(&k), step.perturbation) {
                if old.extra != end.extra && start != 0.0 {
                    return fail(format!("step {}: changed direction for {} load is not supported", step.name, k.1));
                }
            }
            loads.amp_dload.push(amp_dl.get(&k).cloned());
            loads.dload.push((end, start));
        }
        loads.amp_bc = amp_bc.into_iter().filter(|(k, _)| loads.bc.contains_key(k)).collect();
        loads.user_bc = crate::user::loads::user_bcs(model, step)?.into_iter().filter(|b| loads.bc.contains_key(&(b.node, b.slot))).collect();
        loads.amp_cload = amp_cl;
        Ok((loads, Propagated { bcs, cloads: cl, dloads: dl, warnings: warn }))
    }
}

/// Boundary conditions and loads that apply after a general step.
#[derive(Debug, Clone, Default)]
pub struct Propagated {
    pub bcs: BTreeMap<(usize, usize), f64>,
    pub cloads: BTreeMap<(usize, usize), f64>,
    pub dloads: BTreeMap<(usize, String), DistLoad>,
    pub warnings: Vec<String>,
}
