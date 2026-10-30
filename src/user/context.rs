//! Körkontext för användarrutiner och fältvariabler.
//!
//! Användarrutinerna får stegnummer (KSTEP), inkrementnummer (KINC),
//! stegtid och total tid (TIME(1), TIME(2)), elementnummer (NOEL),
//! integrationspunkt (NPT), punktens läge (COORDS), elementets
//! karakteristiska längd (CELENT) och fältvariablerna i punkten (PREDEF/FIELD).
//! RustyFEMs element- och materialgränssnitt skickar inte allt detta, så det
//! läggs här:
//!
//! - `Run` (en per analys, delas mellan trådar): steg, inkrement, stegets
//!   starttid och längd, procedurkod, fälthistorien och begärt PNEWDT.
//!   Skapas av `steps::run` (`begin_analysis`) och förs vidare till
//!   arbetstrådarna i `par::map` (`current`/`install`).
//! - `Point` (per tråd): elementet och integrationspunkten som utvärderas just
//!   nu. Sätts av `fem::assembly::Prepared::ctx` (element, nodernas fält) och av
//!   kontinuumkärnan (`elements::common::response`, `kinematics::continuum`)
//!   för varje integrationspunkt (NPT, COORDS, CELENT).
//!
//! Utan körkontext (t.ex. när ett element anropas direkt i ett test) är
//! KSTEP = KINC = 0, TIME(1) = TIME(2) och inga fält finns.

use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Fälthistorien (nodvärden som funktion av tiden), se `materials::field_vars`.
pub type Fields = crate::materials::field_vars::FieldHistory;

/// Analysens kontext (delas mellan trådarna).
pub struct Run {
    kstep: AtomicI32,
    kinc: AtomicI32,
    /// Procedurtyp som i JSTEP(2)/LFLAGS(1) (1 = statisk, 11 = implicit dynamik, 31 = värme ...).
    procedure: AtomicI32,
    nlgeom: AtomicI32,
    perturbation: AtomicI32,
    step_start: AtomicU64,
    period: AtomicU64,
    /// Minsta begärda PNEWDT i inkrementet (f64-bitar).
    pnewdt: AtomicU64,
    pub fields: Option<Arc<Fields>>,
    /// Utskrifter (*NODE PRINT/*EL PRINT/*ENERGY PRINT) per steg, se `user::dat`.
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
    /// Integrationspunkternas lägen som `set_ip` har sett (när inspelning pågår, `record_ips`).
    static RECORD: RefCell<Option<Vec<(usize, [f64; 3], f64)>>> = const { RefCell::new(None) };
    static RUN: RefCell<Option<Arc<Run>>> = const { RefCell::new(None) };
    static POINT: RefCell<Point> = RefCell::new(Point::default());
}

/// Materialpunkten som utvärderas just nu (per tråd).
#[derive(Debug, Clone, Default)]
pub struct Point {
    /// Elementindex i modellen (`usize::MAX` = okänt) och elementnummer i indatafilen.
    pub elem: usize,
    pub noel: i64,
    /// Integrationspunkt (1..), 0 = inte satt av elementet.
    pub npt: usize,
    /// Antal materialanrop sedan elementet sattes (reserv för NPT).
    pub calls: usize,
    pub coords: [f64; 3],
    pub celent: f64,
    /// Fältvariablerna i elementets noder vid inkrementets slut och början
    /// (nodvis, `nfield` värden per nod).
    pub node_fields: Vec<f64>,
    pub node_fields_old: Vec<f64>,
    pub nfield: usize,
    /// Fältvariablerna i punkten vid inkrementets slut och början (medelvärdet
    /// av nodvärdena, eller interpolerat i integrationspunkten).
    pub fields: Vec<f64>,
    pub fields_old: Vec<f64>,
    /// Fält som USDFLD har satt (ersätter `fields` i materialet).
    pub user_fields: Option<(Vec<f64>, Vec<f64>)>,
}

/// Analysens kontext i den här tråden.
pub fn current() -> Option<Arc<Run>> {
    RUN.with(|r| r.borrow().clone())
}

/// Sätter analysens kontext i den här tråden (arbetstrådar i `par::map`).
pub fn install(run: Option<Arc<Run>>) {
    RUN.with(|r| *r.borrow_mut() = run);
}

/// Återställer föregående kontext när analysen är klar.
pub struct Guard(Option<Arc<Run>>);

impl Drop for Guard {
    fn drop(&mut self) {
        install(self.0.take());
    }
}

/// Startar en analys: ny kontext med modellens fälthistoria.
pub fn begin_analysis(model: &crate::fem::model::FeModel) -> crate::model::Result<Guard> {
    let fields = Fields::build(model)?.map(Arc::new);
    let prev = current();
    install(Some(Arc::new(Run::new(fields))));
    Ok(Guard(prev))
}

/// Procedurkod för ett steg (JSTEP(2), LFLAGS(1)).
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

/// Stegets längd (andra talet på procedurens datarad, standard 1).
pub fn step_period(step: &crate::fem::model::StepDef) -> f64 {
    step.procedure.as_ref().and_then(|p| p.first_floats().ok()).and_then(|v| v.get(1).copied()).filter(|&x| x > 0.0).unwrap_or(1.0)
}

/// Startar steg nummer `kstep` (1..) med starttiden `start` (total tid) och längden `period`.
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

/// Startar inkrement nummer `kinc` (1..) och glömmer begärt PNEWDT.
pub fn begin_increment(kinc: usize) {
    if let Some(r) = current() {
        r.kinc.store(kinc as i32, Ordering::Relaxed);
        r.pnewdt.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }
}

/// Glömmer begärt PNEWDT (före varje försök med ett inkrement).
pub fn reset_pnewdt() {
    if let Some(r) = current() {
        r.pnewdt.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }
}

/// En användarrutin begär inkrementlängden `pnewdt` · Δt (PNEWDT, < 1 = gör om inkrementet).
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

/// Minsta begärda PNEWDT i inkrementet (`None` om ingen rutin har begärt något).
pub fn pnewdt() -> Option<f64> {
    current().map(|r| f(&r.pnewdt)).filter(|p| p.is_finite())
}

/// (KSTEP, KINC).
pub fn step_inc() -> (i32, i32) {
    current().map_or((0, 0), |r| (r.kstep.load(Ordering::Relaxed), r.kinc.load(Ordering::Relaxed)))
}

/// JSTEP(1..4): steg, procedur, NLGEOM, linjär perturbation.
pub fn jstep(nlgeom: bool) -> [i32; 4] {
    match current() {
        Some(r) => [r.kstep.load(Ordering::Relaxed), r.procedure.load(Ordering::Relaxed), nlgeom as i32, r.perturbation.load(Ordering::Relaxed)],
        None => [0, 1, nlgeom as i32, 0],
    }
}

/// Stegtid för den totala tiden `total`.
pub fn step_time(total: f64) -> f64 {
    match current() {
        Some(r) => total - f(&r.step_start),
        None => total,
    }
}

/// Stegets längd (PERIOD).
pub fn period() -> f64 {
    current().map_or(1.0, |r| f(&r.period))
}

/// Total tid vid stegets början.
pub fn step_start() -> f64 {
    current().map_or(0.0, |r| f(&r.step_start))
}

/// Fälthistorien (om modellen har *FIELD eller *INITIAL CONDITIONS, TYPE=FIELD).
pub fn fields() -> Option<Arc<Fields>> {
    current().and_then(|r| r.fields.clone())
}

/// Sätter elementet som utvärderas (anropas av `Prepared::ctx`).
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

/// Sätter integrationspunkten (1..) med läge och karakteristisk längd.
/// Med formfunktionerna `n` interpoleras fälten i punkten; utan (`None`)
/// används elementets medelvärde (som för temperaturen i första ordningens element).
/// `w` är integrationspunktens vikt (volym) i referensläget.
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

/// Räknar ett materialanrop och ger punkten (NPT = antalet anrop om elementet inte satt NPT).
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

/// Punkten utan att räkna anropet.
pub fn point() -> Point {
    POINT.with(|p| p.borrow().clone())
}

/// Fältvariablerna i punkten vid inkrementets slut och början (USDFLD:s värden om de är satta).
pub fn point_fields() -> (Vec<f64>, Vec<f64>) {
    POINT.with(|p| {
        let p = p.borrow();
        match &p.user_fields {
            Some((a, b)) => (a.clone(), b.clone()),
            None => (p.fields.clone(), p.fields_old.clone()),
        }
    })
}

/// USDFLD: ersätter fältvariablerna i punkten (`None` = återställ).
pub fn set_user_fields(f: Option<(Vec<f64>, Vec<f64>)>) {
    POINT.with(|p| p.borrow_mut().user_fields = f);
}

/// Kör `f` och returnerar integrationspunkterna (NPT, COORDS, vikt) som
/// elementet rapporterade under tiden (tom om elementet inte rapporterar).
pub fn record_ips(f: impl FnOnce()) -> Vec<(usize, [f64; 3], f64)> {
    RECORD.with(|r| *r.borrow_mut() = Some(Vec::new()));
    f();
    RECORD.with(|r| r.borrow_mut().take().unwrap_or_default())
}
