//! Implicit direct time integration (*DYNAMIC) with Hilber–Hughes–Taylor (HHT-α).
//!
//! The equation of motion M ü + C u̇ + I(u) = P(t) is integrated with Newmark's
//! updates
//!
//! ```text
//! ü₁ = (u₁ − u₀ − Δt u̇₀)/(β Δt²) − (1/(2β) − 1) ü₀
//! u̇₁ = u̇₀ + Δt [(1 − γ) ü₀ + γ ü₁]
//! ```
//!
//! and HHT's weighted equilibrium (Hilber, Hughes & Taylor 1977; Hughes, "The
//! Finite Element Method", ch. 9):
//!
//! ```text
//! M ü₁ + (1 + α)(C u̇₁ + I₁ − P₁) − α (C u̇₀ + I₀ − P₀) = 0,  −1/3 ≤ α ≤ 0
//! ```
//!
//! The equation is divided by (1 + α), so that the effective tangent becomes
//! K + M/((1+α) β Δt²) + γ/(β Δt) C and can be assembled with the usual
//! tangent from the elements. The Newton–Raphson driver (`newton.rs`) solves each
//! increment; linear models are solved with a single equation solve per increment.
//!
//! Parameters:
//! - `APPLICATION=TRANSIENT FIDELITY` (default): α = −0.05, β = ¼(1−α)², γ = ½ − α,
//!   initial acceleration is computed.
//! - `APPLICATION=MODERATE DISSIPATION`: α = −0.41421, β = ¼(1−α)², γ = ½ − α.
//! - `APPLICATION=QUASI-STATIC`: backward Euler (TIME INTEGRATOR=BWE):
//!   u̇₁ = (u₁ − u₀)/Δt, ü₁ = (u̇₁ − u̇₀)/Δt, equilibrium at the end of the
//!   increment; the default amplitude is RAMP. `TIME INTEGRATOR=BWE/HHT-TF/HHT-MD`
//!   overrides the application's integrator.
//! - `ALPHA=`, `BETA=`, `GAMMA=` override (β and γ follow α unless given).
//! - `INITIAL=YES/NO`: compute ü₀ = M⁻¹(P₀ − I₀ − C u̇₀) at the start of the step.
//!
//! Data line: initial time increment, step time, minimum, maximum increment.
//! `DIRECT` gives fixed increments; otherwise automatic incrementation
//! (half-step residual, output time points, see `dynamic_incr.rs`).
//!
//! Damping: Rayleigh damping per material (*DAMPING, ALPHA=αR, BETA=βR) gives
//! the element C = αR M + βR K (linear stiffness).
//!
//! The mass matrix is diagonal (row-summed) for
//! first-order elements (CPS4, CPE4, C3D8, T2D2, T3D2, ...).
//!
//! Initial velocity: *INITIAL CONDITIONS, TYPE=VELOCITY, or the velocity
//! from a preceding dynamic step (`SolutionState::vectors["V"]`).

#[path = "dynamic_amplitude.rs"]
pub mod amplitude;
#[path = "dynamic_incr.rs"]
pub mod incr;

use super::api::{StepCtx, StepType};
use super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::elements::MassKind;
use crate::fem::assembly::{self, DofMap, Prepared};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::model::{FeModel, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::SolutionState;
use crate::model::{fail, FemError, Result};
use crate::solver::SystemMatrix;
use amplitude::LoadEval;
use std::collections::BTreeMap;

pub struct Implicit;

pub static STEP_TYPES: &[&dyn StepType] = &[&Implicit];

// ---------------------------------------------------------------------------
// Shared helpers for dynamic steps (also used by explicit.rs and modal.rs)
// ---------------------------------------------------------------------------

/// Sparse symmetric matrix over node slots (node · 8 + slot), stored by row.
#[derive(Debug, Clone, Default)]
pub struct NodeMatrix {
    pub rows: Vec<Vec<(usize, f64)>>,
}

impl NodeMatrix {
    fn from_maps(maps: Vec<BTreeMap<usize, f64>>) -> Self {
        NodeMatrix { rows: maps.into_iter().map(|m| m.into_iter().filter(|(_, v)| *v != 0.0).collect()).collect() }
    }

    pub fn is_zero(&self) -> bool {
        self.rows.iter().all(|r| r.is_empty())
    }

    /// y = A x.
    pub fn mul(&self, x: &[NodeVec]) -> Vec<NodeVec> {
        let mut y = vec![[0.0; SLOTS]; x.len()];
        for (g, row) in self.rows.iter().enumerate() {
            let mut s = 0.0;
            for &(c, v) in row {
                s += v * x[c / SLOTS][c % SLOTS];
            }
            y[g / SLOTS][g % SLOTS] = s;
        }
        y
    }

    /// The diagonal.
    pub fn diag(&self) -> Vec<NodeVec> {
        let mut d = vec![[0.0; SLOTS]; self.rows.len() / SLOTS];
        for (g, row) in self.rows.iter().enumerate() {
            d[g / SLOTS][g % SLOTS] = row.iter().find(|p| p.0 == g).map_or(0.0, |p| p.1);
        }
        d
    }

    /// The matrix is diagonal.
    pub fn is_diagonal(&self) -> bool {
        self.rows.iter().enumerate().all(|(g, r)| r.iter().all(|p| p.0 == g))
    }

    /// Adds `c · A` to the lower triangle of the system matrix (free DOFs).
    /// With linear constraints (tie, MPC) Tᵀ A T is added, i.e. the slaves' mass is
    /// transferred to their masters as for the stiffness.
    pub fn add_to(&self, dm: &DofMap, k: &mut dyn SystemMatrix, c: f64) {
        if c == 0.0 {
            return;
        }
        if !dm.mpc.is_empty() {
            for (g, row) in self.rows.iter().enumerate() {
                for &(h, v) in row {
                    dm.add_entry(k, g / SLOTS, g % SLOTS, h / SLOTS, h % SLOTS, c * v);
                }
            }
            return;
        }
        for (g, row) in self.rows.iter().enumerate() {
            let a = dm.eq[g / SLOTS][g % SLOTS];
            if a == assembly::NONE {
                continue;
            }
            for &(h, v) in row {
                let b = dm.eq[h / SLOTS][h % SLOTS];
                if b != assembly::NONE && a >= b {
                    k.add(a, b, c * v);
                }
            }
        }
    }
}

/// Sets the slaves' velocity or acceleration from their masters
/// (u̇_s = Σ c u̇_m; the constraints' right-hand sides are constant in time).
pub fn expand_rate(dm: &DofMap, x: &mut [NodeVec]) {
    for sl in &dm.mpc.slaves {
        x[sl.node][sl.slot] = sl.terms.iter().map(|&(n, s, c)| c * x[n][s]).sum();
    }
}

/// Rayleigh damping (αR, βR) for material `mi` (*DAMPING).
pub fn rayleigh(model: &FeModel, mi: Option<usize>) -> Result<(f64, f64)> {
    let Some(m) = mi.map(|i| &model.materials[i]) else { return Ok((0.0, 0.0)) };
    let Some(kw) = m.behavior("DAMPING") else { return Ok((0.0, 0.0)) };
    if kw.has("STRUCTURAL") || kw.has("COMPOSITE") {
        return kw.err("STRUCTURAL/COMPOSITE damping is not supported in time integration");
    }
    let num = |k: &str| -> Result<f64> {
        match kw.param(k) {
            Some(v) => crate::fem::model::parse_f64(v).ok_or_else(|| FemError(format!("{}: {k}={v} is not a number", kw.location()))),
            None => Ok(0.0),
        }
    };
    Ok((num("ALPHA")?, num("BETA")?))
}

/// Mass matrix (and damping matrix C = αR M + βR K) for the whole model.
pub fn global_matrices(p: &Prepared, kind: MassKind) -> Result<(NodeMatrix, NodeMatrix)> {
    global_matrices_by(p, &|_| kind)
}

/// Mass type per element in implicit dynamics (area struct_nonlin): B22/B32
/// have consistent mass as in *FREQUENCY, other elements (including shells, unlike
/// *FREQUENCY) diagonal mass.
pub fn implicit_mass_kind(et: &dyn crate::elements::ElementType) -> MassKind {
    match et.name() {
        "B22" | "B32" | "B22H" | "B32H" => MassKind::Consistent,
        _ => MassKind::Lumped,
    }
}

/// Like `global_matrices` with the mass type chosen per element type.
pub fn global_matrices_by(p: &Prepared, kind_of: &dyn Fn(&dyn crate::elements::ElementType) -> MassKind) -> Result<(NodeMatrix, NodeMatrix)> {
    let m = p.model;
    let ng = m.nodes.len() * SLOTS;
    let mut mm: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); ng];
    let mut cm: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); ng];
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let mi = el.section.and_then(|s| m.sections[s].material_index);
        let (ar, br) = rayleigh(m, mi)?;
        let dl = &p.edofs[e];
        let nd = dl.len();
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        let me = if ctx.density > 0.0 { Some(el.etype.mass(&ctx, kind_of(el.etype)).map_err(wrap)?) } else { None };
        let ke = if br != 0.0 { Some(el.etype.stiffness(&ctx).map_err(wrap)?) } else { None };
        for a in 0..nd {
            let ga = dl[a].0 * SLOTS + dl[a].1;
            for b in 0..nd {
                let gb = dl[b].0 * SLOTS + dl[b].1;
                let mv = me.as_ref().map_or(0.0, |x| x[a * nd + b]);
                if mv != 0.0 {
                    *mm[ga].entry(gb).or_default() += mv;
                }
                let cv = ar * mv + ke.as_ref().map_or(0.0, |x| br * x[a * nd + b]);
                if cv != 0.0 {
                    *cm[ga].entry(gb).or_default() += cv;
                }
            }
        }
    }
    // MASS/ROTARY INERTIA elements carry their mass through interactions,
    // not ElementType::mass. Include it in inertia and the effective tangent.
    for interaction in &p.interactions {
        for (na, da, nb, db, value) in interaction.mass() {
            let (sa, sb) = (dof::slot_of(da), dof::slot_of(db));
            if p.active[na] & (1 << sa) != 0 && p.active[nb] & (1 << sb) != 0 {
                *mm[na * SLOTS + sa].entry(nb * SLOTS + sb).or_default() += value;
            }
        }
    }
    Ok((NodeMatrix::from_maps(mm), NodeMatrix::from_maps(cm)))
}

/// Initial velocity: from a preceding dynamic step, otherwise *INITIAL CONDITIONS, TYPE=VELOCITY.
pub fn initial_velocity(p: &Prepared, state: &SolutionState) -> Result<Vec<NodeVec>> {
    let nn = p.model.nodes.len();
    if let Some(v) = state.vectors.get("V") {
        return Ok(v.clone());
    }
    let mut v = vec![[0.0; SLOTS]; nn];
    for kw in p.model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.param_upper("TYPE").as_deref() == Some("VELOCITY")) {
        for d in &kw.data {
            let n: usize = d.fields[0].trim_start_matches('#').parse().map_err(|_| FemError(format!("{}: internal node reference", kw.location())))?;
            let vals = kw.floats(&DataLineRef(d).tail())?;
            let (dofv, val) = (vals[0], vals[1]);
            let Some(s) = dof::slot(dofv.round() as dof::Dof) else {
                return kw.err_at(d.line, format!("'{dofv}' is not a degree of freedom"));
            };
            if p.active[n] & (1 << s) != 0 {
                v[n][s] = val;
            }
        }
    }
    Ok(v)
}

/// Helper: data line without the first field.
struct DataLineRef<'a>(&'a crate::fem::model::DataLine);
impl DataLineRef<'_> {
    fn tail(&self) -> crate::fem::model::DataLine {
        crate::fem::model::DataLine { fields: self.0.fields[1..].to_vec(), line: self.0.line, continued: false }
    }
}

/// When field output is written (from *OUTPUT, FIELD in the step).
#[derive(Debug, Clone, Copy)]
pub enum Schedule {
    /// Every n-th increment.
    Every(usize),
    /// At evenly spaced times (step time) with this spacing.
    Interval(f64),
}

/// Reads the output frequency from the step's *OUTPUT, FIELD (first occurrence).
pub fn output_schedule(step: &StepDef, period: f64, default: Schedule) -> Result<Schedule> {
    for kw in step.output.iter().filter(|k| k.name == "OUTPUT" && (k.has("FIELD") || !k.has("HISTORY"))) {
        if !kw.has("FIELD") {
            continue;
        }
        let num = |k: &str| kw.param(k).and_then(crate::fem::model::parse_f64);
        if let Some(n) = num("FREQUENCY") {
            return Ok(Schedule::Every((n.round() as usize).max(1)));
        }
        if let Some(n) = num("NUMBER INTERVAL") {
            return Ok(Schedule::Interval(period / n.max(1.0)));
        }
        if let Some(dt) = num("TIME INTERVAL") {
            return Ok(Schedule::Interval(dt));
        }
        if kw.has("TIME POINTS") {
            return kw.err("TIME POINTS is not supported yet");
        }
        return Ok(default);
    }
    Ok(default)
}

/// Adds velocity and acceleration (V, A) to a frame.
pub fn add_va(fr: &mut FrameData, dim: usize, v: &[NodeVec], a: &[NodeVec]) {
    let labels = |p: &str| (1..=dim).map(|i| format!("{p}{i}")).collect::<Vec<_>>();
    fr.node_extra.push(("V".into(), labels("V"), v.iter().map(|x| x[..dim].to_vec()).collect()));
    fr.node_extra.push(("A".into(), labels("A"), a.iter().map(|x| x[..dim].to_vec()).collect()));
}

/// Default amplitude for dynamic steps: AMPLITUDE= on *STEP, otherwise step (the load
/// acts fully from the start of the step – verified in
/// tests/cases/dynamics/t2d2_hht_steglast).
pub fn default_ramp(step: &StepDef) -> bool {
    match step.header.param_upper("AMPLITUDE") {
        Some(a) => a == "RAMP",
        None => false,
    }
}

fn param_f64(kw: &RawKeyword, key: &str) -> Result<Option<f64>> {
    match kw.param(key) {
        Some(v) => crate::fem::model::parse_f64(v).map(Some).ok_or_else(|| FemError(format!("{}: {key}={v} is not a number", kw.location()))),
        None => Ok(None),
    }
}

// ---------------------------------------------------------------------------
// HHT-α
// ---------------------------------------------------------------------------

/// HHT parameters.
#[derive(Debug, Clone, Copy)]
pub struct Hht {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub initial_accel: bool,
    /// Backward Euler (APPLICATION=QUASI-STATIC, TIME INTEGRATOR=BWE) instead of HHT.
    pub bwe: bool,
}

impl Hht {
    pub fn from_keyword(kw: &RawKeyword) -> Result<Hht> {
        Self::from_keyword_app(kw, "TRANSIENT FIDELITY")
    }

    /// Like `from_keyword` with `default_app` when APPLICATION= is missing.
    pub fn from_keyword_app(kw: &RawKeyword, default_app: &str) -> Result<Hht> {
        let app = kw.param_upper("APPLICATION").unwrap_or_else(|| default_app.into());
        // (α, β, γ, initial acceleration) per application. MODERATE DISSIPATION
        // does not compute ü₀ (ü = 0 in frame 0) and has α = 1 − √2
        // exactly, β = ¼(1−α)² = ½, γ = ½ − α (tests/cases/dynamics/cps4_hht_md_rayleigh).
        let (a0, b0, g0, init) = match app.as_str() {
            "TRANSIENT FIDELITY" => (-0.05, 0.275625, 0.55, true),
            "MODERATE DISSIPATION" => (1.0 - std::f64::consts::SQRT_2, 0.5, std::f64::consts::SQRT_2 - 0.5, false),
            "QUASI-STATIC" => (-1.0 / 3.0, 4.0 / 9.0, 5.0 / 6.0, false),
            other => return kw.err(format!("APPLICATION={other} is not supported")),
        };
        let integrator = kw.param_upper("TIME INTEGRATOR");
        let (a0, b0, g0) = match integrator.as_deref() {
            Some("HHT-TF") => (-0.05, 0.275625, 0.55),
            Some("HHT-MD") => (1.0 - std::f64::consts::SQRT_2, 0.5, std::f64::consts::SQRT_2 - 0.5),
            _ => (a0, b0, g0),
        };
        let bwe = match integrator.as_deref() {
            Some("BWE") => true,
            Some("HHT-TF") | Some("HHT-MD") => false,
            Some(other) => return kw.err(format!("TIME INTEGRATOR={other} is not supported")),
            None => app == "QUASI-STATIC",
        };
        if bwe {
            return Ok(Hht { alpha: 0.0, beta: 0.0, gamma: 0.0, initial_accel: kw.flag("INITIAL", false)?, bwe });
        }
        let given = param_f64(kw, "ALPHA")?;
        let alpha = given.unwrap_or(a0);
        if !(-0.5..=0.0).contains(&alpha) {
            return kw.err(format!("ALPHA={alpha} must lie in [−1/2, 0]"));
        }
        let beta = param_f64(kw, "BETA")?.unwrap_or(if given.is_some() { 0.25 * (1.0 - alpha) * (1.0 - alpha) } else { b0 });
        let gamma = param_f64(kw, "GAMMA")?.unwrap_or(if given.is_some() { 0.5 - alpha } else { g0 });
        let initial_accel = kw.flag("INITIAL", init)?;
        Ok(Hht { alpha, beta, gamma, initial_accel, bwe })
    }
}

/// One time increment with HHT-α as a `NonlinearSystem`.
pub struct HhtSystem<'p, 'm> {
    pub p: &'p Prepared<'m>,
    pub nlgeom: bool,
    pub dm: DofMap,
    pub mat: Box<dyn SystemMatrix>,
    pub lev: &'p LoadEval<'p>,
    pub m: &'p NodeMatrix,
    pub c: &'p NodeMatrix,
    pub hht: Hht,
    /// Total time at the start of the step.
    pub time0: f64,
    /// Step time at the start of the increment and the increment length.
    pub tn: f64,
    pub h: f64,
    /// End of the increment as a fraction of `h` (from the driver, normally 1).
    f1: f64,
    pub u: Vec<NodeVec>,
    pub un: Vec<NodeVec>,
    pub vn: Vec<NodeVec>,
    pub an: Vec<NodeVec>,
    /// I₀ − P₀ + C u̇₀ at the start of the increment.
    pub rn: Vec<NodeVec>,
    /// Velocity and acceleration in prescribed DOFs with amplitude at the end of the increment.
    pub bc_rates: Vec<(usize, usize, f64, f64)>,
    /// `rn` at the start of the last accepted increment (for the reactions).
    pub rn_prev: Vec<NodeVec>,
    pub state_old: Vec<Vec<f64>>,
    pub state_new: Vec<Vec<f64>>,
    // latest evaluation
    pub v1: Vec<NodeVec>,
    pub a1: Vec<NodeVec>,
    pub fint: Vec<NodeVec>,
    pub fext: Vec<NodeVec>,
    pub ma: Vec<NodeVec>,
    pub cv: Vec<NodeVec>,
    /// Flux to use when the increment's flux is zero (rigid-body motion without
    /// forces): the step's time average so far, otherwise 10⁻² (the default value
    /// for the initial time-average force in *CONTROLS).
    pub q_floor: BTreeMap<Field, f64>,
    /// Contact forces at the end of the increment (not included in `fint`).
    pub fc: Vec<NodeVec>,
    /// Indices in `p.interactions` of the contact interactions.
    pub contact: Vec<usize>,
    /// P − I − C u̇ before condensation of the constraints at the latest evaluation
    /// and at the start of the increment (for the constraint forces in connector elements).
    pub rs_trial: Vec<NodeVec>,
    pub rs_old: Vec<NodeVec>,
    /// The constraint forces (as −residual in the slaves) at the latest evaluation and the start of the increment.
    pub cf_trial: Vec<NodeVec>,
    pub cf_old: Vec<NodeVec>,
}

/// Tangent contributions scaled by `f` (for the contact tangent in HHT).
struct ScaledSink<'a> {
    dm: &'a DofMap,
    k: &'a mut dyn SystemMatrix,
    f: f64,
}

impl crate::interactions::TangentSink for ScaledSink<'_> {
    fn add(&mut self, na: usize, da: dof::Dof, nb: usize, db: dof::Dof, v: f64) {
        self.dm.add_entry(self.k, na, dof::slot_of(da), nb, dof::slot_of(db), self.f * v);
    }
}

/// The contact interactions (those that output CPRESS).
pub fn contact_interactions(p: &Prepared) -> Vec<usize> {
    (0..p.interactions.len()).filter(|&i| p.interactions[i].node_output().iter().any(|o| o.name == "CPRESS")).collect()
}

/// Snapshot used to redo an increment.
#[derive(Clone)]
pub struct Snapshot {
    un: Vec<NodeVec>,
    vn: Vec<NodeVec>,
    an: Vec<NodeVec>,
    rn: Vec<NodeVec>,
    rn_prev: Vec<NodeVec>,
    state: Vec<Vec<f64>>,
    tn: f64,
    fint: Vec<NodeVec>,
    fext: Vec<NodeVec>,
    ma: Vec<NodeVec>,
    cv: Vec<NodeVec>,
}

impl<'p, 'm> HhtSystem<'p, 'm> {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            un: self.un.clone(),
            vn: self.vn.clone(),
            an: self.an.clone(),
            rn: self.rn.clone(),
            rn_prev: self.rn_prev.clone(),
            state: self.state_old.clone(),
            tn: self.tn,
            fint: self.fint.clone(),
            fext: self.fext.clone(),
            ma: self.ma.clone(),
            cv: self.cv.clone(),
        }
    }

    pub fn restore(&mut self, s: Snapshot) {
        self.un = s.un;
        self.vn = s.vn;
        self.an = s.an;
        self.rn = s.rn;
        self.rn_prev = s.rn_prev;
        self.state_old = s.state;
        self.state_new = self.state_old.clone();
        self.tn = s.tn;
        self.u = self.un.clone();
        self.fint = s.fint;
        self.fext = s.fext;
        self.ma = s.ma;
        self.cv = s.cv;
        self.v1 = self.vn.clone();
        self.a1 = self.an.clone();
    }

    fn kinematics(&self, u: &[NodeVec], h: f64) -> (Vec<NodeVec>, Vec<NodeVec>) {
        let Hht { beta, gamma, bwe, .. } = self.hht;
        let mut a = vec![[0.0; SLOTS]; u.len()];
        let mut v = vec![[0.0; SLOTS]; u.len()];
        if bwe {
            // backward Euler: u̇₁ = Δu/Δt, ü₁ = (u̇₁ − u̇₀)/Δt
            for n in 0..u.len() {
                for s in 0..SLOTS {
                    v[n][s] = (u[n][s] - self.un[n][s]) / h;
                    a[n][s] = (v[n][s] - self.vn[n][s]) / h;
                }
            }
        } else {
            let c0 = 1.0 / (beta * h * h);
            let c1 = 1.0 / (beta * h);
            let c2 = 1.0 / (2.0 * beta) - 1.0;
            for n in 0..u.len() {
                for s in 0..SLOTS {
                    let du = u[n][s] - self.un[n][s];
                    a[n][s] = c0 * du - c1 * self.vn[n][s] - c2 * self.an[n][s];
                    v[n][s] = self.vn[n][s] + h * ((1.0 - gamma) * self.an[n][s] + gamma * a[n][s]);
                }
            }
        }
        for &(n, s, dv, da) in &self.bc_rates {
            if self.dm.constrained[n] & (1 << s) != 0 {
                v[n][s] = dv;
                a[n][s] = da;
            }
        }
        (a, v)
    }

    /// The contact forces in `fint` (from `assembly::internal`) are subtracted and
    /// returned. With `tangent = Some(f)`, f·K_contact is also added to the
    /// system matrix.
    ///
    /// The contact forces are not HHT-weighted: the equilibrium is
    /// M ü₁ + (1+α)(I₁ + C u̇₁ − P₁) − α(I₀ + C u̇₀ − P₀) + F_c,₁ = 0, as for
    /// constraint forces (Lagrange multipliers) in implicit analysis. With
    /// the weighting (1+α)F_c,₁ − αF_c,₀ the impact between two blocks was completely wrong
    /// (tests/cases/dyn_incr/cpe4_stot_*).
    fn split_contact(&mut self, u: &[NodeVec], t: f64, fint: &mut [NodeVec], tangent: Option<f64>) -> Result<Vec<NodeVec>> {
        let nn = u.len();
        let mut fc = vec![[0.0; SLOTS]; nn];
        if self.contact.is_empty() {
            return Ok(fc);
        }
        let ictx = crate::interactions::InteractionCtx { model: self.p.model, time: self.time0 + t, dt: t - self.tn, nlgeom: self.nlgeom };
        for &i in &self.contact {
            let it = &self.p.interactions[i];
            match tangent {
                Some(f) => {
                    let mut sink = ScaledSink { dm: &self.dm, k: &mut *self.mat, f };
                    it.contribute(&ictx, u, &mut fc, Some(&mut sink))?
                }
                None => it.contribute(&ictx, u, &mut fc, None)?,
            }
        }
        for n in 0..nn {
            for s in 0..SLOTS {
                fint[n][s] -= fc[n][s];
            }
        }
        Ok(fc)
    }

    /// I − P + C u̇ at step time `t` for the displacement u (without tangent),
    /// with the contact forces separate: (fint without contact, fext, C u̇, F_c).
    #[allow(clippy::type_complexity)]
    pub fn static_residual_at(&mut self, t: f64, u: &[NodeVec], v: &[NodeVec]) -> Result<(Vec<NodeVec>, Vec<NodeVec>, Vec<NodeVec>, Vec<NodeVec>)> {
        let du: Vec<NodeVec> = u.iter().zip(&self.un).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
        let int = assembly::internal(self.p, &self.dm, u, &du, &self.state_old, self.nlgeom, self.time0 + t, t - self.tn, None)?;
        let fext = assembly::external(self.p, &self.lev.cloads_at(t), &self.lev.dloads_at(t), u, self.nlgeom, self.time0 + t)?;
        let (mut fint, mut fext) = (int.fint, fext);
        let mut fc = self.split_contact(u, t, &mut fint, None)?;
        let mut cv = self.c.mul(v);
        // linear constraints: forces in slaves are transferred to their masters
        for f in [&mut fint, &mut fext, &mut cv, &mut fc] {
            self.dm.mpc.condense(f);
        }
        Ok((fint, fext, cv, fc))
    }
}

impl NonlinearSystem for HhtSystem<'_, '_> {
    fn n_eq(&self) -> usize {
        self.dm.n_eq
    }

    fn eq_field(&self, eq: usize) -> Field {
        dof::field(SLOT_DOF[self.dm.owner[eq].1])
    }

    fn is_linear(&self) -> bool {
        self.p.is_linear() && !self.nlgeom
    }

    fn begin_increment(&mut self, _t0: f64, t1: f64) -> Result<()> {
        self.f1 = t1;
        self.u.clone_from(&self.un);
        let t = self.tn + t1 * self.h;
        for (n, s, v) in self.lev.bc_at(t) {
            if self.dm.constrained[n] & (1 << s) != 0 {
                self.u[n][s] = v;
            }
        }
        self.dm.mpc.expand(&mut self.u);
        self.bc_rates = self.lev.bc_rates_at(t);
        Ok(())
    }

    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        // NLGEOM: constraints that depend on the configuration (rigid bodies with finite rotations)
        // are relinearized in every iteration, as in steps/nlgeom.rs; the slave positions
        // become exact and their accelerations follow from Newmark on the slave position
        if self.nlgeom {
            self.dm.mpc.update_at(self.p, &self.dm.constrained, &mut self.u)?;
        }
        let h = self.f1 * self.h;
        let t = self.tn + h;
        let alpha = self.hht.alpha;
        let (a, v) = self.kinematics(&self.u, h);
        let du: Vec<NodeVec> = self.u.iter().zip(&self.un).map(|(x, y)| std::array::from_fn(|s| x[s] - y[s])).collect();
        let k: Option<&mut dyn SystemMatrix> = if tangent { Some(&mut *self.mat as &mut dyn SystemMatrix) } else { None };
        let int = assembly::internal(self.p, &self.dm, &self.u, &du, &self.state_old, self.nlgeom, self.time0 + t, h, k)?;
        if tangent {
            let Hht { beta, gamma, bwe, .. } = self.hht;
            if bwe {
                self.m.add_to(&self.dm, &mut *self.mat, 1.0 / (h * h));
                self.c.add_to(&self.dm, &mut *self.mat, 1.0 / h);
            } else {
                self.m.add_to(&self.dm, &mut *self.mat, 1.0 / ((1.0 + alpha) * beta * h * h));
                self.c.add_to(&self.dm, &mut *self.mat, gamma / (beta * h));
            }
        }
        let mut fint = int.fint;
        // contact: the tangent must be K_c/(1+α) (the residual is divided by 1+α)
        let u = std::mem::take(&mut self.u);
        let fc = self.split_contact(&u, t, &mut fint, tangent.then_some(1.0 / (1.0 + alpha) - 1.0));
        self.u = u;
        self.fc = fc?;
        self.fint = fint;
        self.state_new = int.state_new;
        self.fext = assembly::external(self.p, &self.lev.cloads_at(t), &self.lev.dloads_at(t), &self.u, self.nlgeom, self.time0 + t)?;
        self.ma = self.m.mul(&a);
        self.cv = self.c.mul(&v);
        // constraint forces (connector elements) from the uncondensed residual;
        // the constraint force is HHT-weighted like the internal forces
        // (tests/cases/contact3/conn_dyn_join)
        if !self.dm.mpc.is_empty() {
            let nn = self.u.len();
            let mut neg = vec![[0.0; SLOTS]; nn];
            for n in 0..nn {
                for s in 0..SLOTS {
                    let st = self.fext[n][s] - self.fint[n][s] - self.cv[n][s];
                    self.rs_trial[n][s] = st;
                    // (1+α)λ₁ − αλ₀ = −(M ü + (1+α)(I − P + C u̇)₁ − α(I − P + C u̇)₀)
                    let r = (1.0 + alpha) * st - alpha * self.rs_old[n][s] - self.fc[n][s] - self.ma[n][s];
                    neg[n][s] = (-r + alpha * self.cf_old[n][s]) / (1.0 + alpha);
                    self.cf_trial[n][s] = neg[n][s];
                }
            }
            let zero = vec![[0.0; SLOTS]; nn];
            for it in &self.p.interactions {
                it.observe_residual(&neg, &zero, &self.dm.mpc.slaves);
            }
        }
        // linear constraints: forces in slaves are transferred to their masters
        for f in [&mut self.fint, &mut self.fext, &mut self.ma, &mut self.cv, &mut self.fc] {
            self.dm.mpc.condense(f);
        }
        self.a1 = a;
        self.v1 = v;
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            let full = (1.0 + alpha) * (self.fext[n][s] - self.fint[n][s] - self.cv[n][s]) - alpha * self.rn[n][s] - self.fc[n][s] - self.ma[n][s];
            r[q] = full / (1.0 + alpha);
        }
        let mut sum: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
        for n in 0..self.u.len() {
            for s in 0..SLOTS {
                if self.dm.active[n] & (1 << s) != 0 {
                    let e = sum.entry(dof::field(SLOT_DOF[s])).or_default();
                    let q = (self.fint[n][s] + self.fc[n][s]).abs().max(self.fext[n][s].abs()).max(self.ma[n][s].abs());
                    if q > 0.0 {
                        e.0 += q;
                        e.1 += 1;
                    }
                }
            }
        }
        let flux = sum
            .into_iter()
            .map(|(f, (s, c))| {
                let q = if c > 0 { s / c as f64 } else { 0.0 };
                let floor = self.q_floor.get(&f).copied().unwrap_or(1e-2);
                // zero flux (less than 10⁻⁵ of the time average)
                (f, if q < 1e-5 * floor { floor } else { q })
            })
            .collect();
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
        self.dm.mpc.expand(&mut self.u);
    }

    fn increment(&self) -> Vec<f64> {
        self.dm.owner.iter().map(|&(n, s)| self.u[n][s] - self.un[n][s]).collect()
    }

    fn commit(&mut self) -> Result<()> {
        // contact etc.: accepted increment (friction state, output)
        for it in &self.p.interactions {
            it.commit();
        }
        self.rn_prev.clone_from(&self.rn);
        self.rs_old.clone_from(&self.rs_trial);
        self.cf_old.clone_from(&self.cf_trial);
        self.un.clone_from(&self.u);
        self.vn.clone_from(&self.v1);
        self.an.clone_from(&self.a1);
        // I − P + C u̇ at the end of the increment becomes the next increment's "old" term
        for n in 0..self.u.len() {
            for s in 0..SLOTS {
                self.rn[n][s] = self.fext[n][s] - self.fint[n][s] - self.cv[n][s];
            }
        }
        std::mem::swap(&mut self.state_old, &mut self.state_new);
        self.state_new.clone_from(&self.state_old);
        self.tn += self.f1 * self.h;
        Ok(())
    }
}

/// Time stepping data from the *DYNAMIC line.
#[derive(Debug, Clone, Copy)]
pub struct TimeControls {
    pub dt0: f64,
    pub period: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub haftol: Option<f64>,
    pub fixed: bool,
    /// INCREMENTATION=CONSERVATIVE (the half-step residual controls the increments).
    pub conservative: bool,
    /// Half-step tolerance (None: no check, NOHAF).
    pub half_tol: Option<incr::HalfTol>,
}

impl TimeControls {
    pub fn from_keyword(kw: &RawKeyword) -> Result<TimeControls> {
        Self::from_keyword_app(kw, "TRANSIENT FIDELITY", false)
    }

    /// Like `from_keyword` with application `default_app` when APPLICATION=
    /// is missing and `contact` if the model has contact. Defaults:
    /// maximum increment max(T/100, Δt₀)
    /// for TRANSIENT FIDELITY, max(T/10, Δt₀) for MODERATE DISSIPATION and T
    /// for QUASI-STATIC; minimum min(Δt₀, 10⁻⁵ T); half-step tolerance HAFTOL
    /// or HALFINC SCALE FACTOR (default 1000, with contact 10000) times
    /// the time average of the force.
    pub fn from_keyword_app(kw: &RawKeyword, default_app: &str, contact: bool) -> Result<TimeControls> {
        let app = kw.param_upper("APPLICATION").unwrap_or_else(|| default_app.into());
        let v = kw.first_floats()?;
        let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        let dt0 = v.first().copied().filter(|&x| x > 0.0).unwrap_or(period).min(period);
        let dt_min = v.get(2).copied().filter(|&x| x > 0.0).unwrap_or(dt0.min(period * 1e-5));
        let dt_max = match v.get(3).copied().filter(|&x| x > 0.0) {
            Some(x) => x,
            None => match app.as_str() {
                "TRANSIENT FIDELITY" => (period / 100.0).max(dt0),
                "MODERATE DISSIPATION" => (period / 10.0).max(dt0),
                _ => period,
            },
        };
        let haftol = param_f64(kw, "HAFTOL")?;
        let scale = param_f64(kw, "HALFINC SCALE FACTOR")?;
        let conservative = match kw.param_upper("INCREMENTATION").as_deref() {
            Some("CONSERVATIVE") => true,
            Some("AGGRESSIVE") => false,
            Some(other) => return kw.err(format!("INCREMENTATION={other} is not supported")),
            None => app == "TRANSIENT FIDELITY",
        } && !kw.has("NOHAF");
        let half_tol = if !conservative {
            None
        } else if let Some(s) = scale {
            Some(incr::HalfTol::Scale(s))
        } else if let Some(h) = haftol {
            Some(incr::HalfTol::Absolute(h))
        } else {
            Some(incr::HalfTol::Scale(if contact { 1e4 } else { 1e3 }))
        };
        Ok(TimeControls { dt0, period, dt_min, dt_max, haftol, fixed: kw.has("DIRECT"), conservative, half_tol })
    }
}

/// Characteristic element length (for the half-step tolerance for moments):
/// the mean over the elements of the mean extent along the coordinate axes
/// in which the element has extent. For trusses and beams this becomes the
/// element length, for rectangular CPE4 (0.1 × 0.05) 0.075.
pub fn characteristic_length(model: &FeModel) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for el in &model.elements {
        if el.nodes.len() < 2 {
            continue;
        }
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &k in &el.nodes {
            for d in 0..3 {
                lo[d] = lo[d].min(model.nodes[k][d]);
                hi[d] = hi[d].max(model.nodes[k][d]);
            }
        }
        let ext: Vec<f64> = (0..3).map(|d| hi[d] - lo[d]).filter(|&x| x > 0.0).collect();
        if ext.is_empty() {
            continue;
        }
        sum += ext.iter().sum::<f64>() / ext.len() as f64;
        n += 1;
    }
    if n > 0 {
        sum / n as f64
    } else {
        1.0
    }
}

impl HhtSystem<'_, '_> {
    /// "Internal force" for the reactions: RF = M ü₁ + (1+α)(I₁ + C u̇₁ − P₁) − α(I₀ + C u̇₀ − P₀) + P₁,
    /// i.e. HHT-weighted equilibrium.
    /// `rn_prev` is I₀ − P₀ + C u̇₀ with reversed sign at the start of the increment.
    fn reaction_force(&self, rn_prev: &[NodeVec]) -> Vec<NodeVec> {
        let a = self.hht.alpha;
        (0..self.fint.len())
            .map(|n| std::array::from_fn(|s| self.ma[n][s] + (1.0 + a) * (self.fint[n][s] + self.cv[n][s] - self.fext[n][s]) + a * rn_prev[n][s] + self.fc[n][s] + self.fext[n][s]))
            .collect()
    }
}

fn frame_of(sys: &HhtSystem, inc: usize) -> Result<FrameData> {
    let fdyn = sys.reaction_force(&sys.rn_prev);
    let t = sys.tn;
    let mut fr = assembly::frame(
        sys.p,
        &sys.dm,
        &sys.un,
        &fdyn,
        &sys.fext,
        &sys.state_old,
        sys.nlgeom,
        sys.time0 + t,
        t,
        format!("Increment {inc}: Step Time = {t:.4}"),
    )?;
    add_va(&mut fr, sys.p.model.dim(), &sys.vn, &sys.an);
    Ok(fr)
}

/// Solves M x = r for the free DOFs (diagonal mass: division).
pub fn solve_mass(p: &Prepared, dm: &DofMap, mat: &mut dyn SystemMatrix, m: &NodeMatrix, r: &[NodeVec]) -> Result<Vec<NodeVec>> {
    let nn = r.len();
    let mut x = vec![[0.0; SLOTS]; nn];
    if m.is_diagonal() && dm.mpc.is_empty() {
        let d = m.diag();
        for n in 0..nn {
            for s in 0..SLOTS {
                if dm.eq[n][s] != assembly::NONE && d[n][s] != 0.0 {
                    x[n][s] = r[n][s] / d[n][s];
                }
            }
        }
        return Ok(x);
    }
    mat.zero();
    m.add_to(dm, mat, 1.0);
    // area struct_nonlin: DOFs entirely without mass (shell rotation about
    // the normal with consistent mass) get zero acceleration
    let mut d = m.diag();
    // mass that the constraints transfer from slaves (e.g. rigid-body nodes to
    // the reference node) is also counted; otherwise the reference node's initial acceleration became zero
    for sl in &dm.mpc.slaves {
        let ds = d[sl.node][sl.slot].abs();
        if ds != 0.0 || !m.rows[sl.node * SLOTS + sl.slot].is_empty() {
            for &(n, s, c) in &sl.terms {
                d[n][s] += c * c * ds.max(f64::MIN_POSITIVE);
            }
        }
    }
    let massless = |n: usize, s: usize| d[n][s] == 0.0 && m.rows[n * SLOTS + s].is_empty();
    for (q, &(n, s)) in dm.owner.iter().enumerate() {
        if massless(n, s) {
            mat.add(q, q, 1.0);
        }
    }
    if let Err(q) = mat.factor() {
        let (n, s) = dm.owner[q];
        return fail(format!("the mass matrix is singular at node {} DOF {} – missing density?", p.model.node_name(n), dof::name(SLOT_DOF[s])));
    }
    let mut b: Vec<f64> = dm.owner.iter().map(|&(n, s)| if massless(n, s) { 0.0 } else { r[n][s] }).collect();
    mat.solve(&mut b);
    for (q, &(n, s)) in dm.owner.iter().enumerate() {
        x[n][s] = b[q];
    }
    expand_rate(dm, &mut x);
    Ok(x)
}

impl StepType for Implicit {
    fn name(&self) -> &'static str {
        "DYNAMIC (implicit HHT)"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (p.name == "DYNAMIC" && !p.has("EXPLICIT") && !p.has("SUBSPACE")).then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if ctx.step.perturbation {
            return fail("*DYNAMIC cannot be a perturbation step");
        }
        // the default application is MODERATE DISSIPATION when
        // the model has contact (verified in .dat, tests/cases/contact/kontakt_dynamik_*)
        let contact = ctx.prep.model.extra.iter().any(|k| k.name == "CONTACT PAIR" || k.name == "CONTACT");
        let default_app = if contact { "MODERATE DISSIPATION" } else { "TRANSIENT FIDELITY" };
        let hht = Hht::from_keyword_app(kw, default_app)?;
        let tc = TimeControls::from_keyword_app(kw, default_app, contact)?;
        let prep = ctx.prep;
        let model = prep.model;
        // QUASI-STATIC: the default amplitude is RAMP
        let quasi = kw.param_upper("APPLICATION").as_deref() == Some("QUASI-STATIC");
        let ramp = match ctx.step.header.param_upper("AMPLITUDE") {
            Some(a) => a == "RAMP",
            None => quasi,
        };
        let lev = LoadEval::new(model, ctx.loads, ramp, tc.period, ctx.state.time)?;
        let (m, c) = global_matrices_by(prep, &implicit_mass_kind)?;
        let (dm, mat) = assembly::dofmap(prep, &ctx.loads.constrained(), ctx.solver)?;
        let nn = model.nodes.len();
        let v0 = initial_velocity(prep, ctx.state)?;
        let mut sys = HhtSystem {
            p: prep,
            nlgeom: ctx.nlgeom,
            dm,
            mat,
            lev: &lev,
            m: &m,
            c: &c,
            hht,
            time0: ctx.state.time,
            tn: 0.0,
            h: tc.dt0,
            f1: 1.0,
            u: ctx.state.u.clone(),
            un: ctx.state.u.clone(),
            vn: v0.clone(),
            an: vec![[0.0; SLOTS]; nn],
            rn: vec![[0.0; SLOTS]; nn],
            rn_prev: vec![[0.0; SLOTS]; nn],
            bc_rates: Vec::new(),
            state_old: ctx.state.elem_state.clone(),
            state_new: ctx.state.elem_state.clone(),
            v1: v0.clone(),
            a1: vec![[0.0; SLOTS]; nn],
            fint: vec![[0.0; SLOTS]; nn],
            fext: vec![[0.0; SLOTS]; nn],
            ma: vec![[0.0; SLOTS]; nn],
            cv: vec![[0.0; SLOTS]; nn],
            q_floor: BTreeMap::new(),
            fc: vec![[0.0; SLOTS]; nn],
            contact: contact_interactions(prep),
            rs_trial: vec![[0.0; SLOTS]; nn],
            rs_old: vec![[0.0; SLOTS]; nn],
            cf_trial: vec![[0.0; SLOTS]; nn],
            cf_old: vec![[0.0; SLOTS]; nn],
        };
        // prescribed values at the start of the step
        for (n, s, v) in lev.bc_at(0.0) {
            if sys.dm.constrained[n] & (1 << s) != 0 {
                sys.un[n][s] = v;
            }
        }
        for &(n, s, dv, _) in &lev.bc_rates_at(0.0) {
            if sys.dm.constrained[n] & (1 << s) != 0 {
                sys.vn[n][s] = dv;
            }
        }
        sys.dm.mpc.expand(&mut sys.un);
        expand_rate(&sys.dm, &mut sys.vn);
        let v0 = sys.vn.clone();
        let u0 = sys.un.clone();
        // the loads at the start of the step are evaluated just after the start
        // (t = 10⁻⁶ Δt₀): a step load then acts fully already in ü₀
        // and in the HHT term for t₀, while a tabular amplitude starting at zero
        // gives ü₀ ≈ 0 (tests/cases/dynamics/t2d2_hht_*).
        let te = 1e-6 * tc.dt0;
        let (fint0, fext0, cv0, fc0) = sys.static_residual_at(te, &u0, &v0)?;
        for n in 0..nn {
            for s in 0..SLOTS {
                sys.rn[n][s] = fext0[n][s] - fint0[n][s] - cv0[n][s];
            }
        }
        sys.rn_prev.clone_from(&sys.rn);
        let r0: Vec<NodeVec> = (0..nn).map(|n| std::array::from_fn(|s| sys.rn[n][s] - fc0[n][s])).collect();
        sys.an = match (hht.initial_accel, ctx.state.vectors.get("A")) {
            (true, _) => solve_mass(prep, &sys.dm, &mut *sys.mat, &m, &r0)?,
            (false, Some(a)) => a.clone(),
            (false, None) => vec![[0.0; SLOTS]; nn],
        };
        for &(n, s, _, da) in &lev.bc_rates_at(0.0) {
            if sys.dm.constrained[n] & (1 << s) != 0 {
                sys.an[n][s] = da;
            }
        }
        expand_rate(&sys.dm, &mut sys.an);
        sys.fint = fint0;
        sys.fext = fext0;
        sys.cv = cv0;
        sys.fc = fc0;
        sys.ma = m.mul(&sys.an);
        sys.u.clone_from(&sys.un);
        sys.a1.clone_from(&sys.an);

        let plan = incr::OutputPlan::from_step(model, ctx.step, tc.period)?;
        let all = ctx.opts.all_frames;
        let verbose = ctx.opts.verbose;
        let mut frames = Vec::new();
        if all {
            let mut f0 = frame_of(&sys, 0)?;
            if ctx.index > 0 {
                // the reactions in frame 0 are those of the preceding step
                for n in 0..nn {
                    for s in 0..SLOTS {
                        if f0.constrained[n] & (1 << s) != 0 {
                            f0.rf[n][s] = ctx.state.reaction[n][s];
                        }
                    }
                }
            }
            frames.push(f0);
        }
        let eps = 1e-9 * tc.period;
        let mode = if tc.fixed {
            incr::Mode::Fixed
        } else if tc.conservative {
            incr::Mode::Conservative
        } else {
            incr::Mode::Aggressive
        };
        let linear = sys.is_linear();
        let mut inc_ctl = incr::Incrementer::new(
            mode,
            incr::IncParams::from_step(ctx.step)?,
            tc.dt0,
            tc.dt_min,
            tc.dt_max,
            tc.period,
            tc.half_tol,
            characteristic_length(model),
            linear,
            plan.points.clone(),
        );
        // Newton controls (*CONTROLS, PARAMETERS=FIELD); the time average of
        // flux is carried over the step's accepted increments
        let mut nctl = Controls { fixed: true, ..Default::default() };
        super::nlgeom::apply_controls(&mut nctl, &ctx.step.extra)?;
        let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
        let mut n_flux = 0usize;
        let mut inc = 0usize;
        let mut iterations = 0usize;
        // the residual at the start of the preceding increment (the half-step residual)
        let mut prev_start: Option<Vec<NodeVec>> = None;
        let mut hh = inc_ctl.next(0.0);
        while sys.tn < tc.period - eps {
            crate::progress::emit(crate::progress::Event::Increment { fraction: sys.tn / tc.period, iterations: 0 })?;
            if inc >= ctx.step.max_increments {
                return fail(format!("the step reached the maximum number of increments ({}) at step time {:.4e} – increase INC= on *STEP", ctx.step.max_increments, sys.tn));
            }
            sys.h = hh;
            if n_flux > 0 {
                sys.q_floor = flux_sum.iter().map(|(f, q)| (*f, q / n_flux as f64)).collect();
            }
            let snap = sys.snapshot();
            let snap_rn = snap.rn.clone();
            sys.begin_increment(0.0, 1.0)?;
            let res = newton::iterate_increment(&mut sys, &nctl, &flux_sum, n_flux)?;
            let Some((iters, flux)) = res else {
                if verbose {
                    eprintln!("  increment {} attempt {}: Δt {hh:.4e} did not converge", inc + 1, inc_ctl.attempt + 1);
                }
                sys.restore(snap);
                hh = inc_ctl.diverged().map_err(|e| FemError(format!("{} at step time {:.4e}", e.0, sys.tn)))?;
                continue;
            };
            // time average of flux including this attempt (for HALFINC SCALE FACTOR)
            let tavg: BTreeMap<Field, f64> = flux.iter().map(|(f, q)| (*f, (flux_sum.get(f).copied().unwrap_or(0.0) + q) / (n_flux + 1) as f64)).collect();
            let tol = inc_ctl.tolerance(&tavg);
            let rh = if mode == incr::Mode::Conservative { half_step_residual(&mut sys, &snap, prev_start.as_deref())? } else { BTreeMap::new() };
            if verbose {
                let fx: Vec<String> = flux.iter().map(|(f, q)| format!("{f:?} {q:.4e}")).collect();
                let rs: Vec<String> = rh.iter().map(|(f, r)| format!("{f:?} {r:.4e}")).collect();
                eprintln!("  increment {} attempt {}: Δt {hh:.4e}, {iters} iterations, flux [{}], half-step residual [{}]", inc + 1, inc_ctl.attempt + 1, fx.join(", "), rs.join(", "));
            }
            if let incr::Verdict::Retry(h) = inc_ctl.check(&rh, &tol).map_err(|e| FemError(format!("{} at step time {:.4e}", e.0, sys.tn)))? {
                sys.restore(snap);
                hh = h;
                continue;
            }
            sys.commit()?;
            for (f, q) in flux {
                *flux_sum.entry(f).or_default() += q;
            }
            n_flux += 1;
            iterations += iters;
            inc += 1;
            // the residual at the start of this increment (for the next half-step residual)
            prev_start = Some(snap_rn);
            let grown = inc_ctl.accepted(sys.tn, rh, &tol, iters);
            let last = sys.tn >= tc.period - eps;
            if (all && plan.write_frame(inc, sys.tn, eps)) || last {
                frames.push(frame_of(&sys, inc)?);
            }
            if verbose {
                eprintln!("  increment {inc}: step time {:.6e}, Δt {:.4e}{}", sys.tn, hh, grown.map_or(String::new(), |g| format!(", Δt may grow to {g:.4e}")));
            }
            if !last {
                hh = inc_ctl.next(sys.tn);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = inc;
        ctx.out.iterations = iterations;
        ctx.state.u = sys.un.clone();
        ctx.state.elem_state = sys.state_old.clone();
        let fdyn = sys.reaction_force(&sys.rn_prev);
        let mut rf = vec![[0.0; SLOTS]; nn];
        for n in 0..nn {
            for s in 0..SLOTS {
                if sys.dm.constrained[n] & (1 << s) != 0 {
                    rf[n][s] = fdyn[n][s] - sys.fext[n][s];
                }
            }
        }
        ctx.state.reaction = rf;
        ctx.state.vectors.insert("V".into(), sys.vn.clone());
        ctx.state.vectors.insert("A".into(), sys.an.clone());
        ctx.state.time += tc.period;
        Ok(())
    }
}

/// Largest residual in the middle of the latest increment (equilibrium error at t + Δt/2),
/// with the displacement interpolated with Newmark's formulas from the start of the
/// increment and the acceleration as the mean of the end points:
///
/// ```text
/// R½ = M ü½ + (1+α)(I½ + C u̇½ − P₁) − α · ½[(I + C u̇ − P)ₜ + (I + C u̇ − P)ₜ₋Δt]
/// ```
///
/// The HHT term for the preceding half step t − Δt/2 is taken as the mean of
/// the residuals at the start of the increment and at the start of the preceding increment
/// (`prev_start`; at the first increment only the start). The load is taken at
/// the end of the increment (tests/cases/dynamics/t2d2_hht_auto_haftol).
fn half_step_residual(sys: &mut HhtSystem, snap: &Snapshot, prev_start: Option<&[NodeVec]>) -> Result<BTreeMap<Field, f64>> {
    let h = sys.h;
    let hh = 0.5 * h;
    let Hht { alpha, beta, gamma, .. } = sys.hht;
    let nn = sys.un.len();
    let a_half: Vec<NodeVec> = (0..nn).map(|n| std::array::from_fn(|s| 0.5 * (snap.an[n][s] + sys.a1[n][s]))).collect();
    let mut u_half = vec![[0.0; SLOTS]; nn];
    let mut v_half = vec![[0.0; SLOTS]; nn];
    for n in 0..nn {
        for s in 0..SLOTS {
            u_half[n][s] = snap.un[n][s] + hh * snap.vn[n][s] + hh * hh * ((0.5 - beta) * snap.an[n][s] + beta * a_half[n][s]);
            v_half[n][s] = snap.vn[n][s] + hh * ((1.0 - gamma) * snap.an[n][s] + gamma * a_half[n][s]);
        }
    }
    let t_half = snap.tn + hh;
    // evaluate with the state at the start of the increment
    let state_end = std::mem::replace(&mut sys.state_old, snap.state.clone());
    let un_end = std::mem::replace(&mut sys.un, snap.un.clone());
    let tn_end = std::mem::replace(&mut sys.tn, snap.tn);
    let res = sys.static_residual_at(t_half, &u_half, &v_half);
    sys.state_old = state_end;
    sys.un = un_end;
    sys.tn = tn_end;
    let (fint, _, cv, fc) = res?;
    let fext = &sys.fext;
    let mut ma = sys.m.mul(&a_half);
    sys.dm.mpc.condense(&mut ma);
    let prev = prev_start.unwrap_or(&snap.rn);
    let mut rmax: BTreeMap<Field, f64> = BTreeMap::new();
    for (n, s) in sys.dm.owner.clone() {
        let r_old = 0.5 * (snap.rn[n][s] + prev[n][s]);
        let r = (1.0 + alpha) * (fext[n][s] - fint[n][s] - cv[n][s]) - alpha * r_old - fc[n][s] - ma[n][s];
        let e = rmax.entry(dof::field(SLOT_DOF[s])).or_default();
        *e = e.max(r.abs());
    }
    Ok(rmax)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hht_defaults() {
        let kw = RawKeyword { name: "DYNAMIC".into(), params: vec![("APPLICATION".into(), "MODERATE DISSIPATION".into())], ..Default::default() };
        let h = Hht::from_keyword(&kw).unwrap();
        assert!((h.beta - 0.5).abs() < 1e-5 && (h.gamma - 0.91421).abs() < 1e-5 && !h.initial_accel);
    }
}
