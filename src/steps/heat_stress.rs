//! Thermal strain in the coupled analysis (`steps/heat_coupled.rs`) and
//! *TEMPERATURE, FILE=job.
//!
//! The sequentially coupled thermal stress (*STATIC with *TEMPERATURE)
//! is solved by the ordinary mechanical steps: the temperature field is built by
//! `materials::thermal::TemperatureField` and the thermal strain is subtracted in
//! the material (`materials::thermal::Thermal`, *EXPANSION). This module holds
//! the equilibrium system for the mechanical part of *COUPLED
//! TEMPERATURE-DISPLACEMENT, where the nodal temperatures at the start and
//! end of the increment instead come from the heat transfer part and are set in the element context
//! (`ElemCtx::temp`, `temp_old`). The material then handles the thermal
//! strain in the same way (the mean temperature in first-order elements).
//!
//! `temperatures_from_job` runs the heat analysis for *TEMPERATURE, FILE=job
//! (the input file of the heat analysis is run with RustyFEM).

use super::newton::{NonlinearSystem, Residual};
use super::heat::TS;
use crate::elements::{ElemCtx, IpField};
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::inp::InpCtx;
use crate::fem::model::RawKeyword;
use crate::model::{FemError, Result};
use crate::solver::{LinearSolver, SystemMatrix};
use std::collections::BTreeMap;

/// No step types of its own: *STATIC with *TEMPERATURE is run by the ordinary
/// mechanical steps (the module is in the registry for `ThermoSystem` and FILE=).
pub static STEP_TYPES: &[&dyn super::api::StepType] = &[];

/// Equilibrium with thermal strain.
pub(crate) struct ThermoSystem<'p, 'm> {
    p: &'p Prepared<'m>,
    loads: &'p StepLoads,
    dm: DofMap,
    mat: Box<dyn SystemMatrix>,
    u: Vec<NodeVec>,
    pub(crate) u_committed: Vec<NodeVec>,
    pub(crate) state_old: Vec<Vec<f64>>,
    state_new: Vec<Vec<f64>>,
    fint: Vec<NodeVec>,
    fext: Vec<NodeVec>,
    time0: f64,
    period: f64,
    t0: f64,
    t1: f64,
    linear: bool,
    /// Nodal temperatures at the start and end of the increment from
    /// the heat transfer part (set by `heat_coupled.rs` before each increment).
    pub(crate) coupled: (Vec<f64>, Vec<f64>),
}

impl<'p, 'm> ThermoSystem<'p, 'm> {
    /// Element context with the nodal temperatures from the heat transfer part. With
    /// `output` the end temperature applies both at the start and the end.
    fn with_temps<'a>(&self, e: usize, base: ElemCtx<'a>, output: bool) -> ElemCtx<'a> {
        let nodes = &self.p.model.elements[e].nodes;
        let (a, b) = &self.coupled;
        if a.len() != self.u.len() || b.len() != self.u.len() {
            return base;
        }
        let temp: Vec<f64> = nodes.iter().map(|&n| b[n]).collect();
        let temp_old: Vec<f64> = if output { temp.clone() } else { nodes.iter().map(|&n| a[n]).collect() };
        ElemCtx { temp, temp_old, ..base }
    }

    fn internal(&mut self, tangent: bool) -> Result<()> {
        let m = self.p.model;
        let nn = m.nodes.len();
        let time = self.time0 + self.t1 * self.period;
        let dt = (self.t1 - self.t0) * self.period;
        let du: Vec<NodeVec> = self.u.iter().zip(&self.u_committed).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
        if tangent {
            self.mat.zero();
        }
        let mut fint = vec![[0.0; SLOTS]; nn];
        let mut state_new = Vec::with_capacity(m.elements.len());
        for e in 0..m.elements.len() {
            let base = self.p.ctx(e, false, time, dt);
            let ue = self.p.gather(e, &self.u);
            let due = self.p.gather(e, &du);
            let mut sn = Vec::new();
            let et = m.elements[e].etype;
            let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
            let ctx = self.with_temps(e, base, false);
            let r = et.internal_force(&ctx, &ue, &due, &self.state_old[e], &mut sn, tangent).map_err(wrap)?;
            let dl = &self.p.edofs[e];
            for (a, &(n, s)) in dl.iter().enumerate() {
                fint[n][s] += r.fint[a];
            }
            if let Some(ke) = r.k {
                let nd = dl.len();
                let map: Vec<usize> = dl.iter().map(|&(n, s)| self.dm.eq[n][s]).collect();
                for a in 0..nd {
                    if map[a] == NONE {
                        continue;
                    }
                    for b in 0..nd {
                        if map[b] != NONE && map[a] >= map[b] {
                            let v = ke[a * nd + b];
                            if v != 0.0 {
                                self.mat.add(map[a], map[b], v);
                            }
                        }
                    }
                }
            }
            state_new.push(sn);
        }
        // contact and other interactions in the coupled analysis (area contact3)
        if !self.p.interactions.is_empty() {
            let ictx = crate::interactions::InteractionCtx { model: m, time, dt, nlgeom: false };
            for it in &self.p.interactions {
                if tangent {
                    let mut sink = EqSink { eq: &self.dm.eq, k: &mut *self.mat };
                    it.contribute(&ictx, &self.u, &mut fint, Some(&mut sink))?;
                } else {
                    it.contribute(&ictx, &self.u, &mut fint, None)?;
                }
            }
        }
        self.fint = fint;
        self.state_new = state_new;
        Ok(())
    }

    fn element_output(&self) -> Result<Vec<Vec<IpField>>> {
        let m = self.p.model;
        let time = self.time0 + self.t1 * self.period;
        let mut out = Vec::with_capacity(m.elements.len());
        for e in 0..m.elements.len() {
            let base = self.p.ctx(e, false, time, 0.0);
            if base.material.is_none() && base.section.and_then(|s| s.beam()).is_none() {
                out.push(Vec::new());
                continue;
            }
            let ue = self.p.gather(e, &self.u_committed);
            let et = m.elements[e].etype;
            let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
            let ctx = self.with_temps(e, base, true);
            let f = et.output(&ctx, &ue, &self.state_old[e]).map_err(wrap)?;
            out.push(f);
        }
        Ok(out)
    }

    pub(crate) fn reactions(&self) -> Vec<NodeVec> {
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

    pub(crate) fn frame(&self, value: f64, description: String) -> Result<FrameData> {
        Ok(FrameData {
            nforc: Vec::new(),
            value,
            mode: -1,
            description,
            u: self.u_committed.clone(),
            rf: self.reactions(),
            active: self.dm.active.clone(),
            constrained: self.dm.constrained.clone(),
            elem: self.element_output()?,
            node_extra: assembly::interaction_output(self.p),
        })
    }
}

impl NonlinearSystem for ThermoSystem<'_, '_> {
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
        self.internal(tangent)?;
        let time = self.time0 + self.t1 * self.period;
        self.fext = assembly::external(self.p, &self.loads.cloads_at(self.t1), &self.loads.dloads_at(self.t1), &self.u, false, time)?;
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            r[q] = self.fext[n][s] - self.fint[n][s];
        }
        let mut sum: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
        for n in 0..self.u.len() {
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
        for it in &self.p.interactions {
            it.commit();
        }
        std::mem::swap(&mut self.state_old, &mut self.state_new);
        self.state_new.clone_from(&self.state_old);
        Ok(())
    }
}

impl<'p, 'm> ThermoSystem<'p, 'm> {
    /// Builds the equilibrium system for the step. `lock_temperature`: the temperature
    /// degrees of freedom (coupled elements) are locked and set by the caller.
    pub(crate) fn new(
        prep: &'p Prepared<'m>,
        loads: &'p StepLoads,
        state: &SolutionState,
        solver: &dyn LinearSolver,
        period: f64,
        lock_temperature: bool,
    ) -> Result<Self> {
        let model = prep.model;
        let nn = model.nodes.len();
        let mut constrained = loads.constrained();
        if lock_temperature {
            // coupled analysis: the temperatures are solved by the heat transfer part
            for (n, &a) in prep.active.iter().enumerate() {
                if a & (1 << TS) != 0 {
                    constrained.insert((n, TS));
                }
            }
        }
        let (dm, mat) = assembly::dofmap(prep, &constrained, solver)?;
        if !dm.mpc.is_empty() {
            return Err(FemError("linear constraints (tie, MPC, connector elements with locked components) in *COUPLED TEMPERATURE-DISPLACEMENT are not supported yet".into()));
        }
        Ok(ThermoSystem {
            p: prep,
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
            // the temperature is given in the mechanical part: thermal strain on
            // linear materials gives a linear subsystem
            linear: prep.materials.iter().flatten().all(|m| m.is_linear_at_given_temperature()) && prep.interactions.iter().all(|i| i.is_linear()),
            coupled: (Vec::new(), Vec::new()),
        })
    }
}

/// Temperatures (node index, T) from a previous heat analysis for *TEMPERATURE,
/// FILE=job. The heat analysis `job.inp`
/// (in the same directory as the input file) is run with RustyFEM. The temperatures are taken from
/// the last increment in step ESTEP (default = BSTEP = 1), or increment EINC.
/// The nodes are matched via (instance, node number).
pub fn temperatures_from_job(ctx: &InpCtx, kw: &RawKeyword, file: &str) -> Result<Vec<(usize, f64)>> {
    let name = file.trim().trim_matches('"');
    // a possible file extension in FILE= is ignored; the heat analysis is always job.inp
    let stem = name.rsplit_once('.').filter(|(_, ext)| !ext.contains(['/', '\\'])).map_or(name, |(s, _)| s);
    let dir = ctx.dir.clone().unwrap_or_else(|| std::path::PathBuf::from("."));
    let path = dir.join(format!("{stem}.inp"));
    if !path.exists() {
        return kw.err(format!(
            "FILE={name}: the heat analysis {} is run with RustyFEM – the file does not exist",
            path.display()
        ));
    }
    let int_param = |key: &str| -> Result<Option<usize>> {
        match kw.param(key) {
            Some(v) => crate::inp::parse_label(v).filter(|&x| x >= 0).map(|x| Some(x as usize)).ok_or_else(|| FemError(format!("{}: {key}={v} must be an integer", kw.location()))),
            None => Ok(None),
        }
    };
    let bstep = int_param("BSTEP")?.unwrap_or(1);
    let estep = int_param("ESTEP")?.unwrap_or(bstep);
    let einc = int_param("EINC")?;
    let m = crate::inp::read_file(&path).map_err(|e| FemError(format!("{}: the heat analysis {}: {e}", kw.location(), path.display())))?;
    let opts = crate::steps::RunOptions { all_frames: einc.is_some(), ..Default::default() };
    let (out, _) = crate::steps::run(&m.model, &opts).map_err(|e| FemError(format!("{}: the heat analysis {}: {e}", kw.location(), path.display())))?;
    let Some(st) = out.steps.get(estep.wrapping_sub(1)) else {
        return kw.err(format!("the heat analysis has {} steps, ESTEP={estep} does not exist", out.steps.len()));
    };
    let fr = match einc {
        Some(i) => st.frames.get(i).ok_or_else(|| FemError(format!("{}: increment EINC={i} does not exist in the heat analysis", kw.location())))?,
        None => st.frames.last().ok_or_else(|| FemError(format!("{}: the heat analysis has no results", kw.location())))?,
    };
    let hm = &m.model;
    let mesh = &ctx.model;
    let mut vals = Vec::new();
    for n in 0..hm.nodes.len() {
        if fr.active.get(n).is_none_or(|a| a & (1 << TS) == 0) {
            continue;
        }
        let inst = &hm.instances[hm.node_inst[n] as usize];
        let Some(i) = mesh.instance_index(inst) else { continue };
        if let Some(k) = mesh.node(i, hm.node_label[n]) {
            vals.push((k, fr.u[n][TS]));
        }
    }
    if vals.is_empty() {
        return kw.err(format!("no nodes in {} correspond to the nodes in this model", path.display()));
    }
    Ok(vals)
}

/// Tangent contributions from interactions to the equations (lower triangle).
struct EqSink<'a> {
    eq: &'a [[usize; SLOTS]],
    k: &'a mut dyn SystemMatrix,
}

impl crate::interactions::TangentSink for EqSink<'_> {
    fn add(&mut self, na: usize, da: dof::Dof, nb: usize, db: dof::Dof, v: f64) {
        let (a, b) = (self.eq[na][dof::slot_of(da)], self.eq[nb][dof::slot_of(db)]);
        if a != NONE && b != NONE && a >= b {
            self.k.add(a, b, v);
        }
    }
}
