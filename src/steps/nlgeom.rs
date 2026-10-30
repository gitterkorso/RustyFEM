//! *STATIC with NLGEOM=YES: large displacements, rotations and strains.
//!
//! - The elements provide the kinematics (`crate::kinematics`): updated
//!   Lagrange for continua, logarithmic strain in trusses, Hermite beams
//!   (B23/B33).
//! - `NlSystem` is the equilibrium problem R = F_ext(u, t) − F_int(u) with follower
//!   pressure and load stiffness, multiplicative update of spatial rotations
//!   (B33) and linear extrapolation of the previous increment's solution as
//!   initial guess (default, EXTRAPOLATION=LINEAR).
//! - `drive` is a Newton–Raphson driver with the following increment rules:
//!   start with *STATIC's initial increment, increase by 1.5
//!   when two consecutive increments have converged in fewer than I_G = 4 iterations,
//!   cutback to a quarter on divergence, and the last increment is cut so
//!   that the step ends exactly at the step time. Convergence is judged with
//!   the standard criteria (R = 5e-3, C = 1e-2 per field); after that the accepted
//!   increment is iterated further until the residual is negligible (1e-9 · q̃) so that
//!   the result does not carry the solver's convergence error. The extra iterations
//!   are not counted in the increment logic.
//!
//! *STATIC, RIKS is handled by `nlgeom_riks.rs` with the same system.

use super::api::{StepCtx, StepType};
use super::newton::{Controls, NonlinearSystem, Residual};
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::model::{DistLoad, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::kinematics::{self, pressure, tensor};
use crate::model::{fail, Result};
use crate::solver::{LinearSolver, SystemMatrix};
use std::collections::BTreeMap;

#[path = "nlgeom_solver.rs"]
pub mod solver;

pub struct StaticNlgeom;

pub static STEP_TYPES: &[&dyn StepType] = &[&StaticNlgeom];

/// The equilibrium system for NLGEOM.
pub struct NlSystem<'p, 'm> {
    pub p: &'p Prepared<'m>,
    pub loads: &'p StepLoads,
    pub dm: DofMap,
    pub mat: Box<dyn SystemMatrix>,
    pub u: Vec<NodeVec>,
    pub u_committed: Vec<NodeVec>,
    pub state_old: Vec<Vec<f64>>,
    pub state_new: Vec<Vec<f64>>,
    pub fint: Vec<NodeVec>,
    pub fext: Vec<NodeVec>,
    pub time0: f64,
    pub period: f64,
    pub t0: f64,
    pub t1: f64,
    /// Nodes with three active rotations (spatial rotations, multiplicative update).
    rot3: Vec<bool>,
    /// Change and length of the previous accepted increment (for extrapolation).
    pub du_prev: Vec<NodeVec>,
    pub dt_prev: f64,
    pub extrapolate: bool,
    /// Load scaling (Riks): external load = `load_scale(t1)`; `None` = the step's ramp.
    pub riks: Option<RiksLoads>,
    /// Latest factorization (number of negative pivots).
    pub info: std::sync::Arc<std::sync::Mutex<solver::FactorInfo>>,
    /// Area gaps1: unbalanced forces F0 that are removed linearly during the step
    /// (`initial_state::UNBALANCE`, e.g. *IMPORT from an explicit analysis).
    unbalance: Option<Vec<NodeVec>>,
}

/// Loads for Riks: F = F₀ + λ (F_ref − F₀), where F₀ is the load at the start of the step.
#[derive(Debug, Clone)]
pub struct RiksLoads {
    pub lambda: f64,
}

impl<'p, 'm> NlSystem<'p, 'm> {
    pub fn new(p: &'p Prepared<'m>, loads: &'p StepLoads, state: &SolutionState, solver: &'static dyn LinearSolver, period: f64) -> Result<Self> {
        let dual = solver::DualSolver::new(solver);
        let info = dual.info.clone();
        let (dm, mat) = assembly::dofmap(p, &loads.constrained(), &dual)?;
        let nn = p.model.nodes.len();
        let rot3 = (0..nn).map(|n| (3..6).all(|s| p.active[n] & (1 << s) != 0)).collect();
        Ok(NlSystem {
            p,
            loads,
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
            rot3,
            du_prev: vec![[0.0; SLOTS]; nn],
            dt_prev: 0.0,
            extrapolate: true,
            unbalance: state.vectors.get(crate::materials::initial_state::UNBALANCE).cloned(),
            riks: None,
            info,
        })
    }

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

    pub fn store(&self, state: &mut SolutionState) {
        state.u = self.u_committed.clone();
        state.elem_state = self.state_old.clone();
        state.reaction = self.reactions();
    }

    /// Point and distributed loads at step time t (or load factor for Riks).
    pub fn loads_at(&self, t: f64) -> (Vec<(usize, usize, f64)>, Vec<DistLoad>) {
        match &self.riks {
            None => (self.loads.cloads_at(t), self.loads.dloads_at(t)),
            Some(r) => {
                // F₀ + λ (F_ref − F₀): ramp with t = λ, without clamping to [0, 1]
                let lam = r.lambda;
                let cl = self.loads.cload.iter().map(|(&(n, s), &(a, b))| (n, s, a + (b - a) * lam)).collect();
                let dl = self
                    .loads
                    .dload
                    .iter()
                    .map(|(l, start)| {
                        let mut l = l.clone();
                        l.magnitude = start + (l.magnitude - start) * lam;
                        l
                    })
                    .collect();
                (cl, dl)
            }
        }
    }

    /// The reference load vector F_ref − F₀ per equation (for Riks) at the current geometry.
    pub fn load_pattern(&self) -> Result<Vec<f64>> {
        let cl: Vec<(usize, usize, f64)> = self.loads.cload.iter().map(|(&(n, s), &(a, b))| (n, s, b - a)).collect();
        let dl: Vec<DistLoad> = self
            .loads
            .dload
            .iter()
            .map(|(l, start)| {
                let mut l = l.clone();
                l.magnitude -= start;
                l
            })
            .collect();
        let time = self.time0 + self.t1 * self.period;
        let (f, _) = pressure::external(self.p, &cl, &dl, &self.u, time, false)?;
        Ok(self.dm.owner.iter().map(|&(n, s)| f[n][s]).collect())
    }

    pub fn frame(&self, value: f64, description: String) -> Result<FrameData> {
        let mut fr = assembly::frame(
            self.p,
            &self.dm,
            &self.u_committed,
            &self.fint,
            &self.fext,
            &self.state_old,
            true,
            self.time0 + value,
            value,
            description,
        )?;
        kinematics::fix_frame_output(self.p, &mut fr, &self.state_old);
        Ok(fr)
    }

    /// Solves K x = a and K y = b with a single factorization (Riks).
    pub fn solve_pair(&mut self, a: &[f64], b: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if let Err(q) = self.mat.factor() {
            return Err(assembly::mechanism_message(self.p, &self.dm, q));
        }
        let mut x = a.to_vec();
        let mut y = b.to_vec();
        self.mat.solve(&mut x);
        self.mat.solve(&mut y);
        Ok((x, y))
    }

    /// Predictor for prescribed displacements: K₀ c = r₀ − (∂f_int/∂u_bc) Δu_bc
    /// with the tangent and internal forces at the start of the increment (the derivative along
    /// the boundary condition change by a forward difference, like `StaticSystem::predict`).
    /// If anything fails, the state with only the boundary conditions applied is kept.
    fn predict_bc(&mut self) -> Result<()> {
        let target = std::mem::replace(&mut self.u, self.u_committed.clone());
        let r0 = match self.evaluate(true) {
            Ok(r) => r.r,
            Err(_) => {
                self.u = target;
                return Ok(());
            }
        };
        let fint0 = self.fint.clone();
        let h = 1e-6;
        for (n, un) in self.u.iter_mut().enumerate() {
            for s in 0..SLOTS {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    un[s] = self.u_committed[n][s] + h * (target[n][s] - self.u_committed[n][s]);
                }
            }
        }
        let ok = self.evaluate(false).is_ok();
        let mut rhs = r0;
        if ok {
            for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
                rhs[q] -= (self.fint[n][s] - fint0[n][s]) / h;
            }
        }
        self.u = target;
        self.state_new.clone_from(&self.state_old);
        if !ok || self.mat.factor().is_err() {
            return Ok(());
        }
        self.mat.solve(&mut rhs);
        if rhs.iter().all(|x| x.is_finite()) {
            self.apply(&rhs);
        }
        Ok(())
    }

    fn add_load_stiffness(&mut self, ks: &pressure::LoadStiffness) {
        for (e, k) in ks {
            let dl = &self.p.edofs[*e];
            let nd = dl.len();
            if !self.dm.mpc.is_empty() {
                for a in 0..nd {
                    for b in 0..nd {
                        if k[a * nd + b] != 0.0 {
                            self.dm.add_entry(&mut *self.mat, dl[a].0, dl[a].1, dl[b].0, dl[b].1, k[a * nd + b]);
                        }
                    }
                }
                continue;
            }
            let map: Vec<usize> = dl.iter().map(|&(n, s)| self.dm.eq[n][s]).collect();
            for a in 0..nd {
                if map[a] == NONE {
                    continue;
                }
                for b in 0..nd {
                    if map[b] != NONE && map[a] >= map[b] {
                        let v = k[a * nd + b];
                        if v != 0.0 {
                            self.mat.add(map[a], map[b], v);
                        }
                    }
                }
            }
        }
    }
}

impl NonlinearSystem for NlSystem<'_, '_> {
    fn n_eq(&self) -> usize {
        self.dm.n_eq
    }

    fn eq_field(&self, eq: usize) -> Field {
        dof::field(SLOT_DOF[self.dm.owner[eq].1])
    }

    fn is_linear(&self) -> bool {
        false
    }

    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        self.t0 = t0;
        self.t1 = t1;
        self.u.clone_from(&self.u_committed);
        if self.extrapolate && self.dt_prev > 0.0 {
            let f = (t1 - t0) / self.dt_prev;
            for &(n, s) in &self.dm.owner {
                self.u[n][s] += f * self.du_prev[n][s];
            }
        }
        let extrapolated = self.extrapolate && self.dt_prev > 0.0;
        let mut moved = false;
        if self.riks.is_none() {
            for (n, s, v) in self.loads.bc_at(t1) {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    moved |= self.u[n][s] != v;
                    self.u[n][s] = v;
                }
            }
        }
        // Without extrapolation (the step's first increment): prescribed displacements
        // are introduced with the tangent from the start of the increment. Otherwise
        // the first iteration is evaluated in the state where only the boundary nodes have moved,
        // which e.g. gives reversed yielding in plastic shells on unloading and
        // divergence with fixed increments (DIRECT). Not with contact (see StaticSystem).
        let contact = self.p.interactions.iter().any(|i| !i.is_linear());
        if moved && !extrapolated && !contact && self.dm.n_eq > 0 {
            self.predict_bc()?;
        }
        Ok(())
    }

    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        // linear constraints (tie, coupling, rigid bodies): the slaves are set from their
        // masters; configuration-dependent constraints are relinearized in the current configuration
        let dm = &mut self.dm;
        dm.mpc.update_at(self.p, &dm.constrained, &mut self.u)?;
        let du: Vec<NodeVec> = self.u.iter().zip(&self.u_committed).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
        let time = self.time0 + self.t1 * self.period;
        let dt = (self.t1 - self.t0) * self.period;
        let k: Option<&mut dyn SystemMatrix> = if tangent { Some(&mut *self.mat as &mut dyn SystemMatrix) } else { None };
        let int = internal(self.p, &self.dm, &self.u, &du, &self.state_old, time, dt, k)?;
        self.fint = int.fint;
        self.state_new = int.state_new;
        let elem_flux = int.flux;
        let (cl, dl) = self.loads_at(self.t1);
        let (fext, ks) = pressure::external(self.p, &cl, &dl, &self.u, time, tangent)?;
        self.fext = fext;
        if let Some(f0) = self.unbalance.as_ref().filter(|_| self.t1 > 0.0) {
            crate::materials::initial_state::add_unbalance(&mut self.fext, f0, self.t1);
        }
        if tangent {
            self.add_load_stiffness(&ks);
        }
        // forces in the constraints' slaves are transferred to their masters
        self.dm.mpc.condense(&mut self.fint);
        self.dm.mpc.condense(&mut self.fext);
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            r[q] = self.fext[n][s] - self.fint[n][s];
        }
        // typical force: the mean magnitude of all element nodal forces
        // (not only the assembled nodal forces) and the external loads
        let mut sum: BTreeMap<Field, (f64, usize)> = elem_flux;
        for n in 0..self.u.len() {
            for s in 0..SLOTS {
                if self.dm.active[n] & (1 << s) != 0 {
                    let v = self.fext[n][s].abs();
                    if v > 0.0 {
                        let e = sum.entry(dof::field(SLOT_DOF[s])).or_default();
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
        let mut spins: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            if (3..6).contains(&s) && self.rot3[n] {
                spins.entry(n).or_default()[s - 3] = c[q];
            } else {
                self.u[n][s] += c[q];
            }
        }
        // spatial rotations: R ← exp([δw]) R, stored as total rotation vector
        for (n, w) in spins {
            let old = [self.u[n][3], self.u[n][4], self.u[n][5]];
            let r = tensor::mul(&tensor::exp_so3(w), &tensor::exp_so3(old));
            let near = [old[0] + w[0], old[1] + w[1], old[2] + w[2]];
            let t = tensor::log_so3_near(&r, near);
            self.u[n][3..6].copy_from_slice(&t);
        }
    }

    fn increment(&self) -> Vec<f64> {
        self.dm.owner.iter().map(|&(n, s)| self.u[n][s] - self.u_committed[n][s]).collect()
    }

    fn commit(&mut self) -> Result<()> {
        // contact etc.: accepted increment (friction state, contact status, output)
        for it in &self.p.interactions {
            it.commit();
        }
        for n in 0..self.u.len() {
            for s in 0..SLOTS {
                self.du_prev[n][s] = self.u[n][s] - self.u_committed[n][s];
            }
        }
        self.dt_prev = self.t1 - self.t0;
        self.u_committed.clone_from(&self.u);
        std::mem::swap(&mut self.state_old, &mut self.state_new);
        self.state_new.clone_from(&self.state_old);
        Ok(())
    }
}

/// Internal forces, new state and the elements' force magnitudes per field.
struct Internal {
    fint: Vec<NodeVec>,
    state_new: Vec<Vec<f64>>,
    flux: BTreeMap<Field, (f64, usize)>,
}

/// Like `assembly::internal` with NLGEOM, but also collects the sum of the magnitudes of
/// each element's nodal forces (for the typical force q̃).
#[allow(clippy::too_many_arguments)]
fn internal(
    p: &Prepared,
    dm: &DofMap,
    u: &[NodeVec],
    du: &[NodeVec],
    state_old: &[Vec<f64>],
    time: f64,
    dt: f64,
    mut k: Option<&mut dyn SystemMatrix>,
) -> Result<Internal> {
    use crate::interactions::InteractionCtx;
    let m = p.model;
    let ne = m.elements.len();
    let tangent = k.is_some();
    if let Some(k) = k.as_deref_mut() {
        k.zero();
    }
    let mut fint = vec![[0.0; SLOTS]; m.nodes.len()];
    let mut state_new = Vec::with_capacity(ne);
    let mut flux: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
    let res = crate::par::map(ne, crate::par::threads(), |e| -> Result<(Vec<f64>, Option<Vec<f64>>, Vec<f64>)> {
        let ctx = p.ctx(e, true, time, dt);
        let ue = p.gather(e, u);
        let due = p.gather(e, du);
        let mut sn = Vec::new();
        // *MODEL CHANGE and element deletion (steps/model_change.rs)
        if let Some(r) = crate::steps::model_change::internal_force(p, e, &ctx, &ue, &due, &state_old[e], &mut sn, tangent) {
            let r = r.map_err(|err| crate::model::FemError(format!("{}: {err}", m.elem_name(e))))?;
            return Ok((r.fint, r.k, sn));
        }
        let r = m.elements[e]
            .etype
            .internal_force(&ctx, &ue, &due, &state_old[e], &mut sn, tangent)
            .map_err(|err| crate::model::FemError(format!("{}: {err}", m.elem_name(e))))?;
        Ok((r.fint, r.k, sn))
    });
    for (e, r) in res.into_iter().enumerate() {
        let (fe, ke, sn) = r?;
        let dl = &p.edofs[e];
        for (a, &(n, s)) in dl.iter().enumerate() {
            fint[n][s] += fe[a];
            if fe[a] != 0.0 {
                let f = flux.entry(dof::field(SLOT_DOF[s])).or_default();
                f.0 += fe[a].abs();
                f.1 += 1;
            }
        }
        if let (Some(k), Some(ke)) = (k.as_deref_mut(), ke.as_ref().filter(|_| !dm.mpc.is_empty())) {
            // constraints: K_red = Tᵀ K T
            let nd = dl.len();
            for a in 0..nd {
                for b in 0..nd {
                    if ke[a * nd + b] != 0.0 {
                        dm.add_entry(k, dl[a].0, dl[a].1, dl[b].0, dl[b].1, ke[a * nd + b]);
                    }
                }
            }
        } else if let (Some(k), Some(ke)) = (k.as_deref_mut(), ke) {
            let nd = dl.len();
            let map: Vec<usize> = dl.iter().map(|&(n, s)| dm.eq[n][s]).collect();
            let sym = k.is_symmetric();
            for a in 0..nd {
                if map[a] == NONE {
                    continue;
                }
                for b in 0..nd {
                    if map[b] != NONE && !sym {
                        // unsymmetric solver (*STEP, UNSYMM=YES): the full element matrix
                        let v = ke[a * nd + b];
                        if v != 0.0 {
                            k.add_unsym(map[a], map[b], v);
                        }
                    } else if map[b] != NONE && map[a] >= map[b] {
                        // symmetric part (unsymmetric material tangents)
                        let v = 0.5 * (ke[a * nd + b] + ke[b * nd + a]);
                        if v != 0.0 {
                            k.add(map[a], map[b], v);
                        }
                    }
                }
            }
        }
        state_new.push(sn);
    }
    if !p.interactions.is_empty() {
        // interactions via the core interface (the contribution is not counted in q̃)
        let ictx = InteractionCtx { model: m, time, dt, nlgeom: true };
        for it in p.interactions.iter().filter(|it| !crate::steps::model_change::interaction_off(p, &it.name())) {
            match k.as_deref_mut() {
                Some(k) => {
                    let mut sink = Sink { eq: &dm.eq, k, dm };
                    it.contribute(&ictx, u, &mut fint, Some(&mut sink))?
                }
                None => it.contribute(&ictx, u, &mut fint, None)?,
            }
        }
    }
    Ok(Internal { fint, state_new, flux })
}

struct Sink<'a> {
    eq: &'a [[usize; SLOTS]],
    k: &'a mut dyn SystemMatrix,
    dm: &'a DofMap,
}

impl crate::interactions::TangentSink for Sink<'_> {
    fn add(&mut self, na: usize, da: dof::Dof, nb: usize, db: dof::Dof, v: f64) {
        if !self.dm.mpc.is_empty() {
            self.dm.add_entry(self.k, na, dof::slot_of(da), nb, dof::slot_of(db), v);
            return;
        }
        let (a, b) = (self.eq[na][dof::slot_of(da)], self.eq[nb][dof::slot_of(db)]);
        if a != NONE && b != NONE && a >= b {
            self.k.add(a, b, v);
        }
    }
}

/// Largest magnitude per field.
pub fn max_by_field<S: NonlinearSystem + ?Sized>(sys: &S, v: &[f64]) -> BTreeMap<Field, f64> {
    let mut m: BTreeMap<Field, f64> = BTreeMap::new();
    for (i, x) in v.iter().enumerate() {
        let e = m.entry(sys.eq_field(i)).or_default();
        *e = e.max(x.abs());
    }
    m
}

/// Result of an increment iteration.
pub struct IterOutcome {
    pub iterations: usize,
    pub flux: BTreeMap<Field, f64>,
}

/// Iterates an increment to the convergence criteria and then to a
/// negligible residual. `None` if the increment should be halved.
pub fn iterate<S: NonlinearSystem + ?Sized>(
    sys: &mut S,
    ctrl: &Controls,
    flux_sum: &BTreeMap<Field, f64>,
    n_flux: usize,
    hook: &mut dyn FnMut(&mut S, &[f64]) -> Result<Vec<f64>>,
) -> Result<Option<IterOutcome>> {
    let mut history: Vec<f64> = Vec::new();
    let mut prev_r: Option<BTreeMap<Field, f64>> = None;
    let mut last_c: Option<BTreeMap<Field, f64>> = None;
    let mut accepted: Option<(usize, BTreeMap<Field, f64>)> = None;
    let mut extra = 0;
    for it in 0..=ctrl.max_iter + 8 {
        let res = match sys.evaluate(true) {
            Ok(r) => r,
            Err(e) if it > 0 => {
                if crate::settings::with(|s| s.nl_debug) {
                    eprintln!("    it {it}: {e}");
                }
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        let rmax = max_by_field(sys, &res.r);
        crate::progress::emit(crate::progress::Event::Iteration { iteration: it, residual: rmax.values().copied().fold(0.0, f64::max) })?;
        let qref: BTreeMap<Field, f64> = res
            .flux
            .iter()
            .map(|(f, q)| (*f, (flux_sum.get(f).copied().unwrap_or(0.0) + q) / (n_flux + 1) as f64))
            .collect();
        let qall = qref.values().copied().fold(0.0, f64::max).max(1e-300);
        let inc = max_by_field(sys, &sys.increment());
        let mut converged = true;
        let mut tight = true;
        for (f, &r) in &rmax {
            let q = qref.get(f).copied().unwrap_or(0.0);
            if q < 1e-5 * qall {
                // fields with (almost) zero flux, e.g. the forces in a beam under pure
                // moment: the residual is compared with the largest field's flux (independent
                // of R_n, since it only contains round-off errors)
                if r > 1e-10 * qall {
                    tight = false;
                }
                // (or if the correction is negligible: round-off errors
                // amplified by stiff penalty coefficients, e.g. the connector elements'
                // locked components with NLGEOM)
                let c_small = last_c.as_ref().is_some_and(|c| c.get(f).copied().unwrap_or(0.0) <= 1e-12 * inc.get(f).copied().unwrap_or(0.0).max(1e-300));
                if r > 1e-8 * qall && !c_small {
                    converged = false;
                }
                continue;
            }
            if r > 1e-9 * q {
                tight = false;
            }
            if r <= ctrl.r_linear * q {
                continue;
            }
            let c_ok = match &last_c {
                Some(c) => {
                    let cm = c.get(f).copied().unwrap_or(0.0);
                    let du = inc.get(f).copied().unwrap_or(0.0);
                    // accept also if the estimated next correction
                    // (latest correction · the residual ratio) is small
                    let est = prev_r.as_ref().and_then(|p| p.get(f)).filter(|&&rp| rp > 0.0).map(|&rp| cm * r / rp);
                    cm <= ctrl.c_tol * du || cm == 0.0 || est.is_some_and(|e| e <= ctrl.c_tol * du)
                }
                None => false,
            };
            if !(r <= ctrl.r_tol * q && c_ok) {
                converged = false;
            }
        }
        if crate::settings::with(|s| s.nl_debug) {
            eprintln!("    it {it}: r {:?} q {:?} c {:?} du {:?} ok {converged}", rmax, qref, last_c, inc);
        }
        if converged && accepted.is_none() {
            accepted = Some((it, res.flux.clone()));
        }
        if let Some((n, flux)) = &accepted {
            if tight || extra >= 4 {
                return Ok(Some(IterOutcome { iterations: *n, flux: flux.clone() }));
            }
            extra += 1;
        } else {
            let rn = rmax.values().copied().fold(0.0, f64::max);
            history.push(rn);
            let k = history.len();
            if it >= ctrl.i0 && k >= 3 && history[k - 1] > history[k - 2] && history[k - 2] > history[k - 3] {
                return Ok(None);
            }
            if it >= ctrl.max_iter {
                return Ok(None);
            }
        }
        let c = match hook(sys, &res.r) {
            Ok(c) => c,
            Err(_) if it > 0 => return Ok(None),
            Err(e) => return Err(e),
        };
        if c.is_empty() {
            return Ok(None);
        }
        last_c = Some(max_by_field(sys, &c));
        prev_r = Some(rmax);
        sys.apply(&c);
    }
    Ok(accepted.map(|(n, flux)| IterOutcome { iterations: n, flux }))
}

/// Automatic incrementation. `on_increment(sys, t)` is called after
/// each accepted increment.
pub fn drive(
    sys: &mut NlSystem,
    ctrl: &Controls,
    on_increment: &mut dyn FnMut(&mut NlSystem, f64, usize) -> Result<()>,
) -> Result<(usize, usize)> {
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
    let mut n_flux = 0usize;
    let mut quick_prev = false;
    let mut incs = 0;
    let mut iters = 0;
    while t < 1.0 - 1e-12 {
        if incs >= ctrl.max_increments {
            return fail(format!(
                "the step reached the maximum number of increments ({}) at step time {:.4} – increase INC= on *STEP",
                ctrl.max_increments,
                t * ctrl.period
            ));
        }
        if !ctrl.fixed && t + dt > 1.0 - 1e-12 {
            dt = 1.0 - t;
        }
        let mut cutbacks = 0;
        let out = loop {
            sys.begin_increment(t, t + dt)?;
            let mut solve = |s: &mut NlSystem, r: &[f64]| s.solve(r);
            match iterate(sys, ctrl, &flux_sum, n_flux, &mut solve)? {
                Some(o) => break o,
                None => {
                    if ctrl.fixed {
                        return fail(format!("the increment at step time {:.4} did not converge (DIRECT, fixed increments)", (t + dt) * ctrl.period));
                    }
                    cutbacks += 1;
                    dt *= ctrl.cutback;
                    if cutbacks > ctrl.max_cutbacks || dt < ctrl.min {
                        return fail(format!(
                            "no convergence at step time {:.4}: the increment became smaller than the minimum ({:.3e}) or too many cutbacks",
                            t * ctrl.period,
                            ctrl.min * ctrl.period
                        ));
                    }
                }
            }
        };
        for (f, q) in &out.flux {
            *flux_sum.entry(*f).or_default() += q;
        }
        n_flux += 1;
        sys.commit()?;
        t += dt;
        incs += 1;
        iters += out.iterations;
        crate::progress::emit(crate::progress::Event::Increment { fraction: t, iterations: out.iterations })?;
        on_increment(sys, t, incs)?;
        if !ctrl.fixed {
            let quick = out.iterations <= ctrl.ig && cutbacks == 0;
            if quick && quick_prev {
                dt *= ctrl.growth;
            }
            quick_prev = quick;
            dt = dt.min(ctrl.max);
        }
    }
    Ok((incs, iters))
}

/// The step's control parameters for NLGEOM.
pub fn controls(step: &StepDef) -> Result<Controls> {
    let mut c = Controls::from_procedure(step.procedure.as_ref(), step.max_increments)?;
    apply_controls(&mut c, &step.extra)?;
    Ok(c)
}

/// The numbers on a data line; empty fields (default value) become 0.
fn control_values(kw: &crate::fem::model::RawKeyword, d: &crate::fem::model::DataLine) -> Result<Vec<f64>> {
    d.fields
        .iter()
        .map(|f| {
            let f = f.trim();
            if f.is_empty() {
                Ok(0.0)
            } else {
                f.replace(['d', 'D'], "e").parse::<f64>().map_err(|_| crate::model::FemError(format!("{} line {} (*CONTROLS): '{f}' is not a number", kw.file, d.line)))
            }
        })
        .collect()
}

/// *CONTROLS in the step: PARAMETERS=FIELD gives R_n and C_n (the same for all fields),
/// PARAMETERS=TIME INCREMENTATION gives I_0, I_C, I_G and I_A (first line) and
/// D_C, D_A (second line; the factors for increase/cutback are not used). RESET
/// restores the default values.
pub fn apply_controls(c: &mut Controls, extra: &[crate::fem::model::RawKeyword]) -> Result<()> {
    let def = Controls::default();
    for kw in extra.iter().filter(|k| k.name == "CONTROLS") {
        if kw.has("RESET") {
            c.r_tol = def.r_tol;
            c.c_tol = def.c_tol;
            c.i0 = def.i0;
            c.max_iter = def.max_iter;
            c.ig = def.ig;
            c.max_cutbacks = def.max_cutbacks;
        }
        let recs = kw.records();
        match kw.param_upper("PARAMETERS").as_deref() {
            Some("FIELD") => {
                if let Some(d) = recs.first() {
                    let v = control_values(kw, d)?;
                    if let Some(&r) = v.first().filter(|x| **x > 0.0) {
                        c.r_tol = r;
                    }
                    if let Some(&cc) = v.get(1).filter(|x| **x > 0.0) {
                        c.c_tol = cc;
                    }
                }
            }
            Some("TIME INCREMENTATION") => {
                if let Some(d) = recs.first() {
                    let v = control_values(kw, d)?;
                    let get = |i: usize| v.get(i).copied().filter(|x| *x > 0.0).map(|x| x as usize);
                    if let Some(x) = get(0) {
                        c.i0 = x;
                    }
                    if let Some(x) = get(3) {
                        c.max_iter = x;
                    }
                    if let Some(x) = get(5) {
                        c.ig = x;
                    }
                    if let Some(x) = get(7) {
                        c.max_cutbacks = x;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

impl StepType for StaticNlgeom {
    fn name(&self) -> &'static str {
        "STATIC, NLGEOM"
    }

    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "STATIC" && nlgeom && !step.perturbation && !step.procedure.as_ref().is_some_and(|p| p.has("RIKS")))
            .then_some(10)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let ctrl = controls(ctx.step)?;
        let mut sys = NlSystem::new(ctx.prep, ctx.loads, ctx.state, ctx.solver, ctrl.period)?;
        let all = ctx.opts.all_frames;
        let verbose = ctx.opts.verbose;
        let period = ctrl.period;
        let mut frames = Vec::new();
        if all {
            sys.begin_increment(0.0, 0.0)?;
            sys.u.clone_from(&sys.u_committed);
            sys.evaluate(false)?;
            frames.push(sys.frame(0.0, "Increment 0: Step Time = 0.0".into())?);
        }
        let (incs, iters) = drive(&mut sys, &ctrl, &mut |s, t, inc| {
            if verbose {
                eprintln!("  increment {inc}: step time {:.6}", t * period);
            }
            if all || t >= 1.0 - 1e-12 {
                frames.push(s.frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
            }
            Ok(())
        })?;
        ctx.out.frames.extend(frames);
        ctx.out.increments = incs;
        ctx.out.iterations = iters;
        sys.store(ctx.state);
        ctx.state.time += period;
        Ok(())
    }
}
