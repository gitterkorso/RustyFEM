//! Prepared analysis, DOF numbering and assembly.
//!
//! - `Prepared`: the model with built material models and interactions.
//! - `DofMap`: active and constrained DOFs per node and their equation numbers
//!   (the ordering is chosen by the linear solver).
//! - `internal`: internal forces (and tangent) from all elements, in parallel.
//! - `external`: nodal loads from *CLOAD and *DLOAD.
//! - `frame`: output (U, RF, S, ...) for a frame.

use crate::elements::{ElemCtx, IpField};
use crate::fem::dof::{self, Dof, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::model::{DistLoad, FeModel};
use crate::fem::mpc::MpcMap;
use crate::fem::output::FrameData;
use crate::interactions::{self, Interaction, InteractionCtx, TangentSink};
use crate::materials::{self, MaterialModel};
use crate::model::{fail, FemError, Result};
use crate::par;
use crate::solver::{Graph, LinearSolver, SystemMatrix};
use std::collections::BTreeSet;
use std::sync::Arc;

pub const NONE: usize = usize::MAX;

/// The model ready for analysis.
pub struct Prepared<'m> {
    pub model: &'m FeModel,
    /// Material model per material definition (only those used by sections).
    pub materials: Vec<Option<Arc<dyn MaterialModel>>>,
    pub interactions: Vec<Box<dyn Interaction>>,
    /// Active slots per node (from elements and interactions).
    pub active: Vec<u16>,
    /// The element's DOFs as (node, slot), node by node.
    pub edofs: Vec<Vec<(usize, usize)>>,
    /// Predefined temperature field as a function of total time (if any).
    pub temperature: Option<crate::materials::thermal::TemperatureField>,
    /// Material orientation per section (ORIENTATION= on the section).
    pub orientations: Vec<Option<crate::materials::anisotropic::Orientation>>,
    /// Removed/reactivated elements and element deletion (`steps/model_change.rs`).
    pub activity: crate::steps::model_change::Activity,
}

impl<'m> Prepared<'m> {
    pub fn new(model: &'m FeModel) -> Result<Self> {
        if model.elements.is_empty() {
            return fail("the model has no elements");
        }
        let mut used = vec![false; model.materials.len()];
        for s in &model.sections {
            if let Some(i) = s.material_index {
                used[i] = true;
            }
        }
        let mut materials = Vec::with_capacity(model.materials.len());
        for (i, def) in model.materials.iter().enumerate() {
            materials.push(if used[i] { Some(materials::build(def)?) } else { None });
        }
        let interactions = interactions::build_all(model)?;
        let mut active = vec![0u16; model.nodes.len()];
        let mut edofs = Vec::with_capacity(model.elements.len());
        for e in &model.elements {
            let mut v = Vec::with_capacity(e.etype.n_dofs());
            if e.nodes.len() != e.etype.n_nodes() {
                return fail(format!("element {} has {} nodes, {} should have {}", e.label, e.nodes.len(), e.etype.name(), e.etype.n_nodes()));
            }
            for (k, &n) in e.nodes.iter().enumerate() {
                for &d in e.etype.node_dofs(k) {
                    let s = dof::slot(d).ok_or_else(|| FemError(format!("{}: degree of freedom {d} is not supported", e.etype.name())))?;
                    active[n] |= 1 << s;
                    v.push((n, s));
                }
            }
            edofs.push(v);
        }
        for it in &interactions {
            for (n, d) in it.active_dofs() {
                active[n] |= 1 << dof::slot_of(d);
            }
        }
        let temperature = crate::materials::thermal::TemperatureField::build(model)?;
        let orientations = crate::materials::anisotropic::section_orientations(model)?;
        Ok(Prepared { model, materials, interactions, active, edofs, temperature, orientations, activity: Default::default() })
    }

    /// Context for element `e`.
    pub fn ctx(&self, e: usize, nlgeom: bool, time: f64, dt: f64) -> ElemCtx<'_> {
        let el = &self.model.elements[e];
        let section = el.section.map(|s| &self.model.sections[s]);
        let mi = section.and_then(|s| s.material_index);
        let (mut temp, mut temp_old) = match &self.temperature {
            Some(tf) => (tf.at(&el.nodes, time), tf.at(&el.nodes, time - dt)),
            None => (Vec::new(), Vec::new()),
        };
        // struct_nonlin area: shells and beams get the temperature gradients through
        // the thickness from the nodal temperatures (2·nn values, see elements/shell_section.rs)
        if let Some(tf) = self.temperature.as_ref().filter(|tf| tf.has_gradients()) {
            if matches!(el.etype.family(), crate::elements::Family::Shell | crate::elements::Family::Beam) {
                temp.extend(tf.gradient_at(&el.nodes, time));
                temp_old.extend(tf.gradient_at(&el.nodes, time - dt));
            }
        }
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| self.model.nodes[n]).collect();
        // user routines and field variables: the element being evaluated (user/context.rs)
        crate::user::context::set_element(e, el.label, &el.nodes, &x, time, dt);
        ElemCtx {
            index: e,
            label: el.label,
            x,
            section,
            material: mi.and_then(|i| self.materials[i].as_deref()),
            density: mi.map_or(0.0, |i| self.model.materials[i].density),
            nlgeom,
            time,
            dt,
            temp,
            temp_old,
            orientation: el.section.and_then(|s| self.orientations.get(s)).and_then(|o| o.as_ref()),
        }
    }

    /// All materials and interactions are linear.
    pub fn is_linear(&self) -> bool {
        self.materials.iter().flatten().all(|m| m.is_linear()) && self.interactions.iter().all(|i| i.is_linear())
    }

    /// The element's values from a nodal vector.
    pub fn gather(&self, e: usize, u: &[NodeVec]) -> Vec<f64> {
        self.edofs[e].iter().map(|&(n, s)| u[n][s]).collect()
    }

    /// Initial state for all elements.
    pub fn init_states(&self) -> Result<Vec<Vec<f64>>> {
        (0..self.model.elements.len())
            .map(|e| {
                let ctx = self.ctx(e, false, 0.0, 0.0);
                let et = self.model.elements[e].etype;
                // elements without material (e.g. *BEAM GENERAL SECTION) have no state
                if ctx.material.is_none() {
                    return Ok(Vec::new());
                }
                et.init_state(&ctx).map_err(|err| FemError(format!("{}: {err}", self.model.elem_name(e))))
            })
            .collect()
    }
}

/// DOF numbering.
pub struct DofMap {
    /// Equation number per node and slot (`NONE` = constrained or inactive).
    pub eq: Vec<[usize; SLOTS]>,
    pub active: Vec<u16>,
    pub constrained: Vec<u16>,
    /// (node, slot) for each equation.
    pub owner: Vec<(usize, usize)>,
    pub n_eq: usize,
    /// Dependent DOFs from linear constraints (empty if there are no constraints).
    pub mpc: MpcMap,
}

impl DofMap {
    /// Equations and coefficients that (node, slot) contributes to: one equation
    /// with coefficient 1 for a free DOF, the master equations for a
    /// slave and nothing for a constrained one.
    #[inline]
    pub fn for_each_eq(&self, n: usize, s: usize, mut f: impl FnMut(usize, f64)) {
        match self.mpc.slave(n, s) {
            Some(sl) => {
                for &(m, t, c) in &sl.terms {
                    let q = self.eq[m][t];
                    if q != NONE {
                        f(q, c);
                    }
                }
            }
            None => {
                let q = self.eq[n][s];
                if q != NONE {
                    f(q, 1.0);
                }
            }
        }
    }

    /// Adds `v` to K[(na, sa), (nb, sb)] with the constraints included (only
    /// the lower triangle is written; call for the whole symmetric matrix).
    pub fn add_entry(&self, k: &mut dyn SystemMatrix, na: usize, sa: usize, nb: usize, sb: usize, v: f64) {
        if self.mpc.is_empty() {
            let (a, b) = (self.eq[na][sa], self.eq[nb][sb]);
            if a != NONE && b != NONE && a >= b {
                k.add(a, b, v);
            }
            return;
        }
        self.for_each_eq(na, sa, |a, ca| {
            self.for_each_eq(nb, sb, |b, cb| {
                if a >= b {
                    k.add(a, b, ca * cb * v);
                }
            })
        });
    }
}

impl DofMap {
    /// Adds `v` to K[(na, sa), (nb, sb)] in an unsymmetric matrix
    /// (`SystemMatrix::add_unsym`) with the constraints included.
    pub fn add_entry_unsym(&self, k: &mut dyn SystemMatrix, na: usize, sa: usize, nb: usize, sb: usize, v: f64) {
        self.for_each_eq(na, sa, |a, ca| self.for_each_eq(nb, sb, |b, cb| k.add_unsym(a, b, ca * cb * v)));
    }
}

struct MatrixSink<'a> {
    dm: &'a DofMap,
    k: &'a mut dyn SystemMatrix,
}

impl TangentSink for MatrixSink<'_> {
    fn add(&mut self, na: usize, da: Dof, nb: usize, db: Dof, v: f64) {
        self.dm.add_entry(self.k, na, dof::slot_of(da), nb, dof::slot_of(db), v);
    }
    fn is_unsymmetric(&self) -> bool {
        !self.k.is_symmetric()
    }
    fn add_unsym(&mut self, na: usize, da: Dof, nb: usize, db: Dof, v: f64) {
        if !self.k.is_symmetric() {
            self.dm.add_entry_unsym(self.k, na, dof::slot_of(da), nb, dof::slot_of(db), v);
        }
    }
}

/// Builds the numbering and an empty system matrix. `constrained` are the constrained (node, slot);
/// constraints on inactive DOFs are ignored.
pub fn dofmap(p: &Prepared, constrained: &BTreeSet<(usize, usize)>, solver: &dyn LinearSolver) -> Result<(DofMap, Box<dyn SystemMatrix>)> {
    let nn = p.model.nodes.len();
    let mut cmask = vec![0u16; nn];
    for &(n, s) in constrained {
        if p.active[n] & (1 << s) != 0 {
            cmask[n] |= 1 << s;
        }
    }
    let mpc = MpcMap::build(p, &cmask)?;
    for &(n, s) in &mpc.released {
        cmask[n] &= !(1 << s);
    }
    let free = |n: usize, s: usize| p.active[n] & (1 << s) != 0 && cmask[n] & (1 << s) == 0 && !mpc.is_slave(n, s);
    let mut cid = vec![NONE; nn];
    let mut blocks = Vec::new();
    for n in 0..nn {
        if (0..SLOTS).any(|s| free(n, s)) {
            cid[n] = blocks.len();
            blocks.push(n);
        }
    }
    let ndof: Vec<usize> = blocks.iter().map(|&n| (0..SLOTS).filter(|&s| free(n, s)).count()).collect();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    let mut link = |nodes: &[usize]| {
        let mut ids: Vec<usize> = nodes.iter().map(|&n| cid[n]).filter(|&c| c != NONE).collect();
        // nodes with slaves are coupled via their masters
        for n in nodes {
            if let Some(ms) = mpc.masters.get(n) {
                ids.extend(ms.iter().map(|&m| cid[m]).filter(|&c| c != NONE));
            }
        }
        if !mpc.is_empty() {
            ids.sort_unstable();
            ids.dedup();
        }
        for &a in &ids {
            adj[a].extend(ids.iter().copied().filter(|&b| b != a));
        }
    };
    for e in &p.model.elements {
        link(&e.nodes);
    }
    for it in &p.interactions {
        for g in it.connectivity() {
            link(&g);
        }
    }
    let groups: Vec<Vec<usize>> = mpc.masters.iter().map(|(&n, ms)| std::iter::once(n).chain(ms.iter().copied()).collect()).collect();
    for g in &groups {
        link(g);
    }
    for a in adj.iter_mut() {
        a.sort_unstable();
        a.dedup();
    }
    let coords: Vec<[f64; 3]> = blocks.iter().map(|&n| p.model.nodes[n]).collect();
    let mat = solver.prepare(&Graph { coords, adj, ndof })?;
    let mut eq = vec![[NONE; SLOTS]; nn];
    let mut owner = vec![(0, 0); mat.n()];
    for (c, &n) in blocks.iter().enumerate() {
        let mut q = mat.first_dof()[c];
        for s in 0..SLOTS {
            if free(n, s) {
                eq[n][s] = q;
                owner[q] = (n, s);
                q += 1;
            }
        }
    }
    let n_eq = mat.n();
    Ok((DofMap { eq, active: p.active.clone(), constrained: cmask, owner, n_eq, mpc }, mat))
}

/// Result of an evaluation of internal forces.
pub struct Internal {
    pub fint: Vec<NodeVec>,
    pub state_new: Vec<Vec<f64>>,
}

/// Internal forces for the displacement `u` (total) with `du` since the start of the
/// increment. If `k` is given the tangent is assembled (the matrix is zeroed first).
#[allow(clippy::too_many_arguments)]
pub fn internal(
    p: &Prepared,
    dm: &DofMap,
    u: &[NodeVec],
    du: &[NodeVec],
    state_old: &[Vec<f64>],
    nlgeom: bool,
    time: f64,
    dt: f64,
    mut k: Option<&mut dyn SystemMatrix>,
) -> Result<Internal> {
    let m = p.model;
    let ne = m.elements.len();
    let nn = m.nodes.len();
    let threads = par::threads();
    let tangent = k.is_some();
    if let Some(k) = k.as_deref_mut() {
        k.zero();
    }
    let mut fint = vec![[0.0; SLOTS]; nn];
    let mut state_new = Vec::with_capacity(ne);
    const BLOCK: usize = 8192;
    for b0 in (0..ne).step_by(BLOCK) {
        let b1 = (b0 + BLOCK).min(ne);
        let res = par::map(b1 - b0, threads, |i| -> Result<(Vec<f64>, Option<Vec<f64>>, Vec<f64>)> {
            let e = b0 + i;
            let ctx = p.ctx(e, nlgeom, time, dt);
            let ue = p.gather(e, u);
            let due = p.gather(e, du);
            let mut sn = Vec::new();
            // *MODEL CHANGE and element deletion (steps/model_change.rs)
            if let Some(r) = crate::steps::model_change::internal_force(p, e, &ctx, &ue, &due, &state_old[e], &mut sn, tangent) {
                let r = r.map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
                return Ok((r.fint, r.k, sn));
            }
            let r = m.elements[e]
                .etype
                .internal_force(&ctx, &ue, &due, &state_old[e], &mut sn, tangent)
                .map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
            Ok((r.fint, r.k, sn))
        });
        for (i, r) in res.into_iter().enumerate() {
            let (fe, ke, sn) = r?;
            let e = b0 + i;
            let dl = &p.edofs[e];
            for (a, &(n, s)) in dl.iter().enumerate() {
                fint[n][s] += fe[a];
            }
            if let (Some(k), Some(ke)) = (k.as_deref_mut(), ke.as_ref().filter(|_| !dm.mpc.is_empty())) {
                // constraints: K_red = Tᵀ K T
                let nd = dl.len();
                let sym = k.is_symmetric();
                for a in 0..nd {
                    for b in 0..nd {
                        // unsymmetric element matrix (non-associative plasticity):
                        // symmetric part to symmetric solvers, the whole matrix to unsymmetric ones
                        let v = if sym { 0.5 * (ke[a * nd + b] + ke[b * nd + a]) } else { ke[a * nd + b] };
                        if v != 0.0 {
                            if sym {
                                dm.add_entry(k, dl[a].0, dl[a].1, dl[b].0, dl[b].1, v);
                            } else {
                                dm.add_entry_unsym(k, dl[a].0, dl[a].1, dl[b].0, dl[b].1, v);
                            }
                        }
                    }
                }
            } else if let (Some(k), Some(ke)) = (k.as_deref_mut(), ke) {
                let nd = dl.len();
                let map: Vec<usize> = dl.iter().map(|&(n, s)| dm.eq[n][s]).collect();
                let sym = k.is_symmetric();
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
                        if !sym {
                            // unsymmetric solver (*STEP, UNSYMM=YES): the whole element matrix
                            let v = ke[a * nd + b];
                            if v != 0.0 {
                                k.add_unsym(ia, ib, v);
                            }
                        } else if ia >= ib {
                            // symmetric part (equal to ke for symmetric element matrices)
                            let v = 0.5 * (ke[a * nd + b] + ke[b * nd + a]);
                            if v != 0.0 {
                                k.add(ia, ib, v);
                            }
                        }
                    }
                }
            }
            state_new.push(sn);
        }
    }
    if !p.interactions.is_empty() {
        let ictx = InteractionCtx { model: m, time, dt, nlgeom };
        for it in p.interactions.iter().filter(|it| !crate::steps::model_change::interaction_off(p, &it.name())) {
            match k.as_deref_mut() {
                Some(k) => {
                    let mut sink = MatrixSink { dm, k };
                    it.contribute(&ictx, u, &mut fint, Some(&mut sink))?
                }
                None => it.contribute(&ictx, u, &mut fint, None)?,
            }
        }
    }
    Ok(Internal { fint, state_new })
}

/// Nodal loads from point loads (node, slot, value) and distributed loads.
pub fn external(p: &Prepared, cloads: &[(usize, usize, f64)], dloads: &[DistLoad], u: &[NodeVec], nlgeom: bool, time: f64) -> Result<Vec<NodeVec>> {
    let m = p.model;
    let mut f = vec![[0.0; SLOTS]; m.nodes.len()];
    for &(n, s, v) in cloads {
        f[n][s] += v;
    }
    for l in dloads {
        let e = l.elem;
        if crate::steps::model_change::load_off(p, e) {
            continue;
        }
        // user load PnNU (DLOAD, user/loads.rs)
        if let Some(fe) = crate::user::loads::dload(p, l, time).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))? {
            for (a, &(n, s)) in p.edofs[e].iter().enumerate() {
                f[n][s] += fe[a];
            }
            continue;
        }
        let ctx = p.ctx(e, nlgeom, time, 0.0);
        let ue = p.gather(e, u);
        // surface traction TRVECn/TRSHRn (*DSLOAD, elements/traction.rs)
        let fe = match crate::elements::traction::element_load(&ctx, m.elements[e].etype, l) {
            Ok(Some(fe)) => Ok(fe),
            Ok(None) => m.elements[e].etype.distributed_load(&ctx, l, &ue),
            Err(err) => Err(err),
        }
        .map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        for (a, &(n, s)) in p.edofs[e].iter().enumerate() {
            f[n][s] += fe[a];
        }
    }
    Ok(f)
}

/// Output per element for the displacement `u` and the state `state`.
pub fn element_output(p: &Prepared, u: &[NodeVec], state: &[Vec<f64>], nlgeom: bool, time: f64) -> Result<Vec<Vec<IpField>>> {
    let m = p.model;
    let res = par::map(m.elements.len(), par::threads(), |e| {
        let ctx = p.ctx(e, nlgeom, time, 0.0);
        if ctx.material.is_none() && ctx.section.and_then(|s| s.beam()).is_none() {
            return Ok(Vec::new());
        }
        if let Some(r) = crate::steps::model_change::output(p, e, &ctx, &p.gather(e, u), &state[e]) {
            return r.map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))));
        }
        m.elements[e].etype.output(&ctx, &p.gather(e, u), &state[e]).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))
    });
    res.into_iter().collect()
}

/// Builds a frame: U, RF (= Fint − Fext at constrained DOFs) and element output.
#[allow(clippy::too_many_arguments)]
pub fn frame(
    p: &Prepared,
    dm: &DofMap,
    u: &[NodeVec],
    fint: &[NodeVec],
    fext: &[NodeVec],
    state: &[Vec<f64>],
    nlgeom: bool,
    time: f64,
    value: f64,
    description: String,
) -> Result<FrameData> {
    let nn = p.model.nodes.len();
    let mut rf = vec![[0.0; SLOTS]; nn];
    for n in 0..nn {
        for s in 0..SLOTS {
            if dm.constrained[n] & (1 << s) != 0 {
                rf[n][s] = fint[n][s] - fext[n][s];
            }
        }
    }
    let mut elem = element_output(p, u, state, nlgeom, time)?;
    // element output from interactions (connector elements: CTF, CU, ...)
    for it in &p.interactions {
        for (e, f) in it.elem_output() {
            if let Some(v) = elem.get_mut(e) {
                v.push(f);
            }
        }
    }
    Ok(FrameData {
        value,
        mode: -1,
        description,
        u: u.to_vec(),
        rf,
        active: dm.active.clone(),
        constrained: dm.constrained.clone(),
        elem,
        node_extra: interaction_output(p),
    })
}

/// Nodal output from interactions (CPRESS, COPEN, ...), merged per field.
pub fn interaction_output(p: &Prepared) -> Vec<(String, Vec<String>, Vec<Vec<f64>>)> {
    let mut out: Vec<(String, Vec<String>, Vec<Vec<f64>>)> = Vec::new();
    for it in &p.interactions {
        for no in it.node_output() {
            let i = match out.iter().position(|o| o.0 == no.name) {
                Some(i) => i,
                None => {
                    out.push((no.name.clone(), no.labels.clone(), vec![Vec::new(); p.model.nodes.len()]));
                    out.len() - 1
                }
            };
            for (n, v) in no.values {
                out[i].2[n] = v;
            }
        }
    }
    out
}

/// Plain-text description of a mechanism at equation `q`.
pub fn mechanism_message(p: &Prepared, dm: &DofMap, q: usize) -> FemError {
    let m = p.model;
    let (n, s) = dm.owner[q];
    let x = m.nodes[n];
    let on_solid = m.elements.iter().any(|e| {
        matches!(e.etype.family(), crate::elements::Family::Continuum2D | crate::elements::Family::Continuum3D) && e.nodes.contains(&n)
    });
    let hint = if (3..6).contains(&s) && on_solid {
        " – a beam attached to a single node of a plate/block can rotate freely there; let the beam follow an edge over several nodes or fix the rotation with support"
    } else {
        " – add supports (fix/pin/support, *BOUNDARY) or check that all parts are connected"
    };
    let inst = &m.instances[m.node_inst[n] as usize];
    let name = if inst == crate::fem::model::FLAT_INSTANCE { m.node_label[n].to_string() } else { m.node_name(n) };
    let legacy = ["ux", "uy", "uz", "rx", "ry", "rz", "warp", "nt11"];
    FemError(format!(
        "the model is unstable (mechanism): node {} at ({}, {}, {}) can move freely in {} ({}){}",
        name,
        x[0],
        x[1],
        x[2],
        legacy[s],
        dof::name(SLOT_DOF[s]),
        hint
    ))
}
