//! Execution context for user subroutines and field variables.
//!
//! User subroutines receive the step number (KSTEP), increment number (KINC),
//! step time and total time (TIME(1), TIME(2)), element number (NOEL),
//! integration point (NPT), the point's position (COORDS), the element's
//! characteristic length (CELENT) and the field variables at the point (PREDEF/FIELD).
//! RustyFEM's element and material interfaces do not pass all of this, so it
//! lives here:
//!
//! - `Run` (one per analysis, shared between threads): step, increment, the step's
//!   start time and length, procedure code, the field history and requested PNEWDT.
//!   Created by `steps::run` (`begin_analysis`) and passed on to the
//!   worker threads in `par::map` (`current`/`install`).
//! - `Point` (per thread): the element and integration point currently being
//!   evaluated. Set by `fem::assembly::Prepared::ctx` (element, nodal fields) and by
//!   the continuum kernel (`elements::common::response`, `kinematics::continuum`)
//!   for each integration point (NPT, COORDS, CELENT).
//!
//! Without an execution context (e.g. when an element is called directly in a test)
//! KSTEP = KINC = 0, TIME(1) = TIME(2) and there are no fields.

use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The field history (nodal values as a function of time), see `materials::field_vars`.
pub type Fields = crate::materials::field_vars::FieldHistory;

/// The analysis context (shared between threads).
pub struct Run {
    kstep: AtomicI32,
    kinc: AtomicI32,
    /// Procedure type as in JSTEP(2)/LFLAGS(1) (1 = static, 11 = implicit dynamics, 31 = heat ...).
    procedure: AtomicI32,
    nlgeom: AtomicI32,
    perturbation: AtomicI32,
    step_start: AtomicU64,
    period: AtomicU64,
    /// Smallest requested PNEWDT in the increment (f64 bits).
    pnewdt: AtomicU64,
    pub fields: Option<Arc<Fields>>,
    /// Print output (*NODE PRINT/*EL PRINT/*ENERGY PRINT) per step, see `user::dat`.
    pub dat: Mutex<Vec<String>>,
}

fn f(bits: &AtomicU64) -> f64 {
    f64::from_bits(bits.load(Ordering::Relaxed))
}

impl Run {
    pub fn new(fields: Option<Arc<Fields>>) -> Run {
        Run {
            kstep: AtomicI32::new(0),
            kinc: AtomicI32::new(0),
            procedure: AtomicI32::new(1),
            nlgeom: AtomicI32::new(0),
            perturbation: AtomicI32::new(0),
            step_start: AtomicU64::new(0f64.to_bits()),
            period: AtomicU64::new(1f64.to_bits()),
            pnewdt: AtomicU64::new(f64::INFINITY.to_bits()),
            fields,
            dat: Mutex::new(Vec::new()),
        }
    }
}

thread_local! {
    /// Integration point positions seen by `set_ip` (while recording, `record_ips`).
    static RECORD: RefCell<Option<Vec<(usize, [f64; 3], f64)>>> = const { RefCell::new(None) };
    static RUN: RefCell<Option<Arc<Run>>> = const { RefCell::new(None) };
    static POINT: RefCell<Point> = RefCell::new(Point::default());
}

/// The material point currently being evaluated (per thread).
#[derive(Debug, Clone, Default)]
pub struct Point {
    /// Element index in the model (`usize::MAX` = unknown) and element number in the input file.
    pub elem: usize,
    pub noel: i64,
    /// Integration point (1..), 0 = not set by the element.
    pub npt: usize,
    /// Number of material calls since the element was set (fallback for NPT).
    pub calls: usize,
    pub coords: [f64; 3],
    pub celent: f64,
    /// The field variables at the element's nodes at the end and start of the increment
    /// (node by node, `nfield` values per node).
    pub node_fields: Vec<f64>,
    pub node_fields_old: Vec<f64>,
    pub nfield: usize,
    /// The field variables at the point at the end and start of the increment (the average
    /// of the nodal values, or interpolated at the integration point).
    pub fields: Vec<f64>,
    pub fields_old: Vec<f64>,
    /// Fields set by USDFLD (replace `fields` in the material).
    pub user_fields: Option<(Vec<f64>, Vec<f64>)>,
}

/// The analysis context in this thread.
pub fn current() -> Option<Arc<Run>> {
    RUN.with(|r| r.borrow().clone())
}

/// Sets the analysis context in this thread (worker threads in `par::map`).
pub fn install(run: Option<Arc<Run>>) {
    RUN.with(|r| *r.borrow_mut() = run);
}

/// Restores the previous context when the analysis is done.
pub struct Guard(Option<Arc<Run>>);

impl Drop for Guard {
    fn drop(&mut self) {
        install(self.0.take());
    }
}

/// Starts an analysis: new context with the model's field history.
pub fn begin_analysis(model: &crate::fem::model::FeModel) -> crate::model::Result<Guard> {
    let fields = Fields::build(model)?.map(Arc::new);
    let prev = current();
    install(Some(Arc::new(Run::new(fields))));
    Ok(Guard(prev))
}

/// Procedure code for a step (JSTEP(2), LFLAGS(1)).
pub fn procedure_code(step: &crate::fem::model::StepDef, nlgeom: bool) -> i32 {
    let p = step.procedure.as_ref();
    match step.procedure_name() {
        "STATIC" if p.is_some_and(|k| k.has("RIKS")) => 2,
        "STATIC" => 1,
        "VISCO" => 10,
        "DYNAMIC" if p.is_some_and(|k| k.has("EXPLICIT")) => 17,
        "DYNAMIC" => 11 + nlgeom as i32,
        "HEAT TRANSFER" if p.is_some_and(|k| k.param_upper("STEADY STATE").is_some() || k.has("STEADY STATE")) => 31,
        "HEAT TRANSFER" => 32,
        "COUPLED TEMPERATURE-DISPLACEMENT" => 72,
        "SOILS" => 61,
        "GEOSTATIC" => 61,
        "FREQUENCY" => 41,
        "BUCKLE" => 2,
        _ => 1,
    }
}

/// Step length (second number on the procedure's data line, default 1).
pub fn step_period(step: &crate::fem::model::StepDef) -> f64 {
    step.procedure.as_ref().and_then(|p| p.first_floats().ok()).and_then(|v| v.get(1).copied()).filter(|&x| x > 0.0).unwrap_or(1.0)
}

/// Starts step number `kstep` (1..) with start time `start` (total time) and length `period`.
pub fn begin_step(kstep: usize, start: f64, period: f64, procedure: i32, nlgeom: bool, perturbation: bool) {
    if let Some(r) = current() {
        r.kstep.store(kstep as i32, Ordering::Relaxed);
        r.kinc.store(0, Ordering::Relaxed);
        r.procedure.store(procedure, Ordering::Relaxed);
        r.nlgeom.store(nlgeom as i32, Ordering::Relaxed);
        r.perturbation.store(perturbation as i32, Ordering::Relaxed);
        r.step_start.store(start.to_bits(), Ordering::Relaxed);
        r.period.store(period.to_bits(), Ordering::Relaxed);
    }
}

/// Starts increment number `kinc` (1..) and forgets the requested PNEWDT.
pub fn begin_increment(kinc: usize) {
    if let Some(r) = current() {
        r.kinc.store(kinc as i32, Ordering::Relaxed);
        r.pnewdt.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }
}

/// Forgets the requested PNEWDT (before each attempt at an increment).
pub fn reset_pnewdt() {
    if let Some(r) = current() {
        r.pnewdt.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }
}

/// A user subroutine requests the increment size `pnewdt` · Δt (PNEWDT, < 1 = redo the increment).
pub fn request_pnewdt(pnewdt: f64) {
    if !(pnewdt.is_finite()) {
        return;
    }
    if let Some(r) = current() {
        let mut cur = r.pnewdt.load(Ordering::Relaxed);
        while pnewdt < f64::from_bits(cur) {
            match r.pnewdt.compare_exchange_weak(cur, pnewdt.to_bits(), Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
}

/// Smallest requested PNEWDT in the increment (`None` if no routine has requested anything).
pub fn pnewdt() -> Option<f64> {
    current().map(|r| f(&r.pnewdt)).filter(|p| p.is_finite())
}

/// (KSTEP, KINC).
pub fn step_inc() -> (i32, i32) {
    current().map_or((0, 0), |r| (r.kstep.load(Ordering::Relaxed), r.kinc.load(Ordering::Relaxed)))
}

/// JSTEP(1..4): step, procedure, NLGEOM, linear perturbation.
pub fn jstep(nlgeom: bool) -> [i32; 4] {
    match current() {
        Some(r) => [r.kstep.load(Ordering::Relaxed), r.procedure.load(Ordering::Relaxed), nlgeom as i32, r.perturbation.load(Ordering::Relaxed)],
        None => [0, 1, nlgeom as i32, 0],
    }
}

/// Step time for the total time `total`.
pub fn step_time(total: f64) -> f64 {
    match current() {
        Some(r) => total - f(&r.step_start),
        None => total,
    }
}

/// Step length (PERIOD).
pub fn period() -> f64 {
    current().map_or(1.0, |r| f(&r.period))
}

/// Total time at the start of the step.
pub fn step_start() -> f64 {
    current().map_or(0.0, |r| f(&r.step_start))
}

/// The field history (if the model has *FIELD or *INITIAL CONDITIONS, TYPE=FIELD).
pub fn fields() -> Option<Arc<Fields>> {
    current().and_then(|r| r.fields.clone())
}

/// Sets the element being evaluated (called by `Prepared::ctx`).
pub fn set_element(elem: usize, noel: i64, nodes: &[usize], x: &[[f64; 3]], time: f64, dt: f64) {
    let fl = fields();
    POINT.with(|p| {
        let mut p = p.borrow_mut();
        p.elem = elem;
        p.noel = noel;
        p.npt = 0;
        p.calls = 0;
        p.celent = 0.0;
        p.user_fields = None;
        p.coords = crate::elements::common::centroid(x);
        match &fl {
            Some(fh) => {
                p.nfield = fh.nfield();
                p.node_fields = fh.at(nodes, time);
                p.node_fields_old = fh.at(nodes, time - dt);
                p.fields = average(&p.node_fields, nodes.len(), p.nfield);
                p.fields_old = average(&p.node_fields_old, nodes.len(), p.nfield);
            }
            None => {
                p.nfield = 0;
                p.node_fields.clear();
                p.node_fields_old.clear();
                p.fields.clear();
                p.fields_old.clear();
            }
        }
    });
}

fn average(v: &[f64], nn: usize, nf: usize) -> Vec<f64> {
    if nn == 0 || nf == 0 || v.len() != nn * nf {
        return vec![0.0; nf];
    }
    (0..nf).map(|k| (0..nn).map(|i| v[i * nf + k]).sum::<f64>() / nn as f64).collect()
}

/// Sets the integration point (1..) with position and characteristic length.
/// With the shape functions `n` the fields are interpolated at the point; without (`None`)
/// the element average is used (as for temperature in first-order elements).
/// `w` is the integration point weight (volume) in the reference configuration.
pub fn set_ip(npt: usize, coords: [f64; 3], celent: f64, n: Option<&[f64]>, w: f64) {
    RECORD.with(|r| {
        if let Some(v) = r.borrow_mut().as_mut() {
            if !v.iter().any(|(k, _, _)| *k == npt) {
                v.push((npt, coords, w));
            }
        }
    });
    POINT.with(|p| {
        let mut p = p.borrow_mut();
        p.npt = npt;
        p.coords = coords;
        p.celent = celent;
        p.user_fields = None;
        if let Some(n) = n {
            let nf = p.nfield;
            if nf > 0 && p.node_fields.len() == n.len() * nf {
                p.fields = (0..nf).map(|k| n.iter().enumerate().map(|(i, w)| w * p.node_fields[i * nf + k]).sum()).collect();
                p.fields_old = (0..nf).map(|k| n.iter().enumerate().map(|(i, w)| w * p.node_fields_old[i * nf + k]).sum()).collect();
            }
        }
    });
}

/// Counts a material call and returns the point (NPT = number of calls if the element did not set NPT).
pub fn point_for_call() -> Point {
    POINT.with(|p| {
        let mut p = p.borrow_mut();
        p.calls += 1;
        let mut q = p.clone();
        if q.npt == 0 {
            q.npt = q.calls;
        }
        q
    })
}

/// The point without counting the call.
pub fn point() -> Point {
    POINT.with(|p| p.borrow().clone())
}

/// The field variables at the point at the end and start of the increment (USDFLD's values if set).
pub fn point_fields() -> (Vec<f64>, Vec<f64>) {
    POINT.with(|p| {
        let p = p.borrow();
        match &p.user_fields {
            Some((a, b)) => (a.clone(), b.clone()),
            None => (p.fields.clone(), p.fields_old.clone()),
        }
    })
}

/// USDFLD: replaces the field variables at the point (`None` = reset).
pub fn set_user_fields(f: Option<(Vec<f64>, Vec<f64>)>) {
    POINT.with(|p| p.borrow_mut().user_fields = f);
}

/// Runs `f` and returns the integration points (NPT, COORDS, weight) that
/// the element reported meanwhile (empty if the element does not report).
pub fn record_ips(f: impl FnOnce()) -> Vec<(usize, [f64; 3], f64)> {
    RECORD.with(|r| *r.borrow_mut() = Some(Vec::new()));
    f();
    RECORD.with(|r| r.borrow_mut().take().unwrap_or_default())
}
