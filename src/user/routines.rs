//! More user subroutines with the established Fortran interface (besides UMAT, UEL and VUMAT):
//!
//! | Routine | Input | Selected by (longest prefix) |
//! |---|---|---|
//! | DLOAD | `*DLOAD` with load type PnNU (pressure on face n) | the element set name on the data line |
//! | DISP | `*BOUNDARY, USER` | the node set name on the data line |
//! | UAMP | `*AMPLITUDE, DEFINITION=USER` | the amplitude name |
//! | USDFLD | `*USER DEFINED FIELD` in the material | the material name |
//! | UFIELD | `*FIELD, USER` | the node set name on the data line |
//! | UHARD | `*PLASTIC, HARDENING=USER` | the material name |
//! | SIGINI | `*INITIAL CONDITIONS, TYPE=STRESS, USER` | the element set name on the data line |
//! | UEXPAN | `*EXPANSION, USER` | the material name |
//! | CREEP | `*CREEP, LAW=USER` | the material name |
//! | HETVAL | `*HEAT GENERATION` in the material | the material name |
//! | FILM | `*FILM` with load type FnNU | the element set name on the data line |
//! | UMATHT | `*USER MATERIAL, TYPE=THERMAL` | the material name |
//!
//! The arguments have the established names and meanings. Indices are 0-based in Rust; TIME = [step time, total time]
//! at the start of the increment (for DLOAD, DISP, UFIELD, FILM and UAMP: at the
//! time the value applies to). A routine is written in Rust by implementing the trait
//! and registered with `register_<routine>(prefix, Arc::new(...))` before
//! the run; empty prefix = all. Examples are in `samples2.rs`.
//! Routines in a shared library (Fortran convention) are loaded with
//! `load_routines` (`dynlib2.rs`) or `RUSTYFEM_USER`.

use crate::model::Result;
use std::sync::{Arc, OnceLock, RwLock};

/// Registry of routines per name prefix (upper case).
pub struct Registry<T: ?Sized + Send + Sync + 'static> {
    cell: OnceLock<RwLock<Vec<(String, Arc<T>)>>>,
    init: fn() -> Vec<(String, Arc<T>)>,
}

impl<T: ?Sized + Send + Sync + 'static> Registry<T> {
    pub const fn new(init: fn() -> Vec<(String, Arc<T>)>) -> Self {
        Registry { cell: OnceLock::new(), init }
    }

    fn get(&self) -> &RwLock<Vec<(String, Arc<T>)>> {
        self.cell.get_or_init(|| RwLock::new((self.init)()))
    }

    /// Registers `r` for names starting with `prefix` (replaces the same prefix).
    pub fn register(&self, prefix: &str, r: Arc<T>) {
        let p = prefix.trim().to_uppercase();
        let mut v = self.get().write().unwrap_or_else(|e| e.into_inner());
        v.retain(|(q, _)| *q != p);
        v.push((p, r));
    }

    /// The routine for `name` (longest registered prefix).
    pub fn find(&self, name: &str) -> Option<Arc<T>> {
        let up = name.trim().to_uppercase();
        let v = self.get().read().unwrap_or_else(|e| e.into_inner());
        v.iter().filter(|(p, _)| up.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, r)| r.clone())
    }
}

/// Error message when no routine exists.
pub fn missing<T>(routine: &str, what: &str, name: &str) -> Result<T> {
    // older callers may still pass the Swedish noun
    let what = match what {
        "materialet" => "material",
        "nodsetet" => "node set",
        "elementsetet" => "element set",
        "amplituden" => "amplitude",
        w => w,
    };
    crate::model::fail(format!(
        "no {routine} for {what} {name}: register one in Rust with rustyfem::user::routines::register_{} (see user/samples2.rs) or specify a library with {}_ in RUSTYFEM_USER",
        routine.to_lowercase(),
        routine.to_lowercase()
    ))
}

macro_rules! routine {
    ($trait_:ident, $args:ident, $method:ident, $reg:ident, $register:ident, $find:ident, $samples:path) => {
        pub trait $trait_: Send + Sync {
            fn $method(&self, a: &mut $args) -> Result<()>;
        }
        pub static $reg: Registry<dyn $trait_> = Registry::new($samples);
        /// Registers the routine for names starting with `prefix`.
        pub fn $register(prefix: &str, r: Arc<dyn $trait_>) {
            $reg.register(prefix, r);
        }
        /// The routine for `name`: registered (longest prefix) or the library in `RUSTYFEM_USER`.
        pub fn $find(name: &str) -> Result<Option<Arc<dyn $trait_>>> {
            if let Some(r) = $reg.find(name) {
                return Ok(Some(r));
            }
            super::dynlib2::$find()
        }
    };
}

// ------------------------------------------------------------------ DLOAD

/// DLOAD(F, KSTEP, KINC, TIME, NOEL, NPT, LAYER, KSPT, COORDS, JLTYP, SNAME).
pub struct DloadArgs<'a> {
    /// The load magnitude at the point (in: the magnitude on the data line).
    pub f: f64,
    pub kstep: i32,
    pub kinc: i32,
    pub time: [f64; 2],
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub coords: [f64; 3],
    /// Load type: 20 + n for PnNU.
    pub jltyp: i32,
    /// The surface name (empty for element-based loads).
    pub sname: &'a str,
}
routine!(Dload, DloadArgs, dload, DLOADS, register_dload, find_dload, super::samples2::dloads);

// ------------------------------------------------------------------ DISP

/// DISP(U, KSTEP, KINC, TIME, NODE, NOEL, JDOF, COORDS).
pub struct DispArgs<'a> {
    /// U(1) = prescribed value at the end of the increment (in: the value on the data line),
    /// U(2), U(3) = velocity and acceleration (not used in statics).
    pub u: [f64; 3],
    pub kstep: i32,
    pub kinc: i32,
    /// [step time, total time] at the end of the increment.
    pub time: [f64; 2],
    pub node: i64,
    pub noel: i64,
    pub jdof: i32,
    pub coords: [f64; 3],
    /// The node set on the data line (empty for a node number).
    pub set: &'a str,
}
routine!(Disp, DispArgs, disp, DISPS, register_disp, find_disp, super::samples2::disps);

// ------------------------------------------------------------------ UAMP

/// UAMP (simplified): the amplitude value and derivatives at a point in time.
pub struct UampArgs<'a> {
    pub amp_name: &'a str,
    /// [step time, total time] at which the value is sought.
    pub time: [f64; 2],
    /// The constants on the *AMPLITUDE lines (PROPS).
    pub props: &'a [f64],
    pub amp_value_new: f64,
    pub amp_derivative: f64,
    pub amp_sec_derivative: f64,
    pub kstep: i32,
}
routine!(Uamp, UampArgs, uamp, UAMPS, register_uamp, find_uamp, super::samples2::uamps);

/// *AMPLITUDE, DEFINITION=USER: the routine and PROPS (the data lines).
#[derive(Clone)]
pub struct UampRef {
    pub name: String,
    pub props: Vec<f64>,
    pub routine: Arc<dyn Uamp>,
}

impl std::fmt::Debug for UampRef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "UAMP {} {:?}", self.name, self.props)
    }
}

impl UampRef {
    pub fn new(kw: &crate::fem::model::RawKeyword, name: &str) -> Result<UampRef> {
        let mut props = Vec::new();
        for d in &kw.data {
            props.extend(kw.floats(d)?.into_iter().zip(d.fields.iter()).filter(|(_, f)| !f.trim().is_empty()).map(|(v, _)| v));
        }
        if let Some(n) = kw.param("PROPERTIES").and_then(crate::inp::parse_label) {
            props.resize(n.max(0) as usize, 0.0);
        }
        let Some(routine) = find_uamp(name)? else {
            return missing("UAMP", "amplitude", name);
        };
        Ok(UampRef { name: name.to_string(), props, routine })
    }

    /// (A, A′, A″) at step time and total time. An error in the routine gives A = 0 (with a warning).
    pub fn eval(&self, step_time: f64, total_time: f64) -> (f64, f64, f64) {
        let mut a = UampArgs {
            amp_name: &self.name,
            time: [step_time, total_time],
            props: &self.props,
            amp_value_new: 0.0,
            amp_derivative: 0.0,
            amp_sec_derivative: 0.0,
            kstep: super::context::step_inc().0,
        };
        if let Err(e) = self.routine.uamp(&mut a) {
            eprintln!("warning: UAMP {}: {e}", self.name);
            return (0.0, 0.0, 0.0);
        }
        (a.amp_value_new, a.amp_derivative, a.amp_sec_derivative)
    }
}

// ------------------------------------------------------------------ USDFLD

/// USDFLD(FIELD, STATEV, PNEWDT, DIRECT, T, CELENT, TIME, DTIME, CMNAME, ...).
/// The stress and strain at the start of the increment replace GETVRM('S'/'E').
pub struct UsdfldArgs<'a> {
    pub field: &'a mut [f64],
    pub statev: &'a mut [f64],
    pub pnewdt: &'a mut f64,
    pub celent: f64,
    pub time: [f64; 2],
    pub dtime: f64,
    pub cmname: &'a str,
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub kstep: i32,
    pub kinc: i32,
    pub ndi: usize,
    pub nshr: usize,
    pub coords: [f64; 3],
    /// The temperature at the point at the end of the increment.
    pub temp: f64,
    /// Stress and total strain at the start of the increment (order 11, 22, 33, 12, 13, 23).
    pub stress: &'a [f64],
    pub strain: &'a [f64],
}
routine!(Usdfld, UsdfldArgs, usdfld, USDFLDS, register_usdfld, find_usdfld, super::samples2::usdflds);

// ------------------------------------------------------------------ UFIELD

/// UFIELD(FIELD, KFIELD, NSECPT, KSTEP, KINC, TIME, NODE, COORDS, TEMP, DTEMP, NFIELD).
pub struct UfieldArgs<'a> {
    /// All field variables at the node (in: the values without the routine).
    pub field: &'a mut [f64],
    /// The variable defined by *FIELD, USER (VARIABLE=, 1..).
    pub kfield: usize,
    pub kstep: i32,
    pub kinc: i32,
    pub time: [f64; 2],
    pub node: i64,
    pub coords: [f64; 3],
    pub temp: f64,
    pub dtemp: f64,
    pub nfield: usize,
}
routine!(Ufield, UfieldArgs, ufield, UFIELDS, register_ufield, find_ufield, super::samples2::ufields);

// ------------------------------------------------------------------ UHARD

/// UHARD(SYIELD, HARD, EQPLAS, EQPLASRT, TIME, DTIME, TEMP, DTEMP, NOEL, NPT, ...).
pub struct UhardArgs<'a> {
    pub syield: f64,
    /// HARD(1) = dσy/dε̄p, HARD(2) = dσy/d(ε̄p rate), HARD(3) = dσy/dT.
    pub hard: [f64; 3],
    pub eqplas: f64,
    pub eqplasrt: f64,
    pub time: [f64; 2],
    pub dtime: f64,
    pub temp: f64,
    pub dtemp: f64,
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub kstep: i32,
    pub kinc: i32,
    pub cmname: &'a str,
    pub statev: &'a mut [f64],
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
    pub props: &'a [f64],
}
routine!(Uhard, UhardArgs, uhard, UHARDS, register_uhard, find_uhard, super::samples2::uhards);

// ------------------------------------------------------------------ SIGINI

/// SIGINI(SIGMA, COORDS, NTENS, NCRDS, NOEL, NPT, LAYER, KSPT, LREBAR, NAMES).
pub struct SiginiArgs<'a> {
    pub sigma: &'a mut [f64],
    pub coords: [f64; 3],
    pub ntens: usize,
    pub ncrds: usize,
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub lrebar: i32,
    /// NAMES(1) = rebar name (empty), NAMES(2) = the element set.
    pub names: [&'a str; 2],
}
routine!(Sigini, SiginiArgs, sigini, SIGINIS, register_sigini, find_sigini, super::samples2::siginis);

// ------------------------------------------------------------------ UEXPAN

/// UEXPAN(EXPAN, DEXPANDT, TEMP, TIME, DTIME, PREDEF, DPRED, STATEV, CMNAME, NSTATV, NOEL).
pub struct UexpanArgs<'a> {
    /// Thermal strain increments (1 isotropic, 3 orthotropic, 6 anisotropic; tensor shear).
    pub expan: &'a mut [f64],
    pub dexpandt: &'a mut [f64],
    /// TEMP(1) = temperature at the end of the increment, TEMP(2) = the change.
    pub temp: [f64; 2],
    pub time: [f64; 2],
    pub dtime: f64,
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
    pub statev: &'a mut [f64],
    pub cmname: &'a str,
    pub noel: i64,
}
routine!(Uexpan, UexpanArgs, uexpan, UEXPANS, register_uexpan, find_uexpan, super::samples2::uexpans);

// ------------------------------------------------------------------ CREEP

/// CREEP(DECRA, DESWA, STATEV, SERD, EC, ESW, P, QTILD, TEMP, DTEMP, PREDEF, DPRED, TIME, DTIME, CMNAME, LEXIMP, LEND, COORDS, ...).
pub struct CreepArgs<'a> {
    /// DECRA(1) = Δε̄cr, DECRA(2..5) = derivatives w.r.t. ε̄cr, ε̄sw, p, q̃ (when LEXIMP = 1).
    pub decra: [f64; 5],
    pub deswa: [f64; 5],
    pub statev: &'a mut [f64],
    pub serd: f64,
    /// EC(1), EC(2) = ε̄cr at the start and end of the increment.
    pub ec: [f64; 2],
    pub esw: [f64; 2],
    pub p: f64,
    pub qtild: f64,
    pub temp: f64,
    pub dtemp: f64,
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
    pub time: [f64; 2],
    pub dtime: f64,
    pub cmname: &'a str,
    /// 0 = explicit, 1 = implicit integration (the derivatives are needed).
    pub leximp: i32,
    pub lend: i32,
    pub coords: [f64; 3],
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub kstep: i32,
    pub kinc: i32,
}
routine!(Creep, CreepArgs, creep, CREEPS, register_creep, find_creep, super::samples2::creeps);

/// *CREEP, LAW=USER: the routine for a material.
#[derive(Clone)]
pub struct CreepRef {
    pub cmname: String,
    pub routine: Arc<dyn Creep>,
}

impl std::fmt::Debug for CreepRef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "CREEP {}", self.cmname)
    }
}

impl CreepRef {
    pub fn new(cmname: &str) -> Result<CreepRef> {
        let Some(routine) = find_creep(cmname)? else {
            return missing("CREEP", "material", cmname);
        };
        Ok(CreepRef { cmname: cmname.to_string(), routine })
    }

    /// (Δε̄cr, ∂Δε̄cr/∂q̃) for q̃ = `q`, ε̄cr = `ecr` at the start of the increment and
    /// the time `t` (total time or creep time) at the start. The pressure P is passed as 0.
    pub fn increment(&self, q: f64, ecr: f64, t: f64, dt: f64, temp: f64) -> (f64, f64) {
        let pt = super::context::point();
        let (kstep, kinc) = super::context::step_inc();
        let (f, fo) = super::context::point_fields();
        let dpred: Vec<f64> = f.iter().zip(&fo).map(|(a, b)| a - b).collect();
        let mut statev: Vec<f64> = Vec::new();
        let mut a = CreepArgs {
            decra: [0.0; 5],
            deswa: [0.0; 5],
            statev: &mut statev,
            serd: 0.0,
            ec: [ecr, ecr],
            esw: [0.0; 2],
            p: 0.0,
            qtild: q,
            temp,
            dtemp: 0.0,
            predef: &fo,
            dpred: &dpred,
            time: [super::context::step_time(t), t],
            dtime: dt,
            cmname: &self.cmname,
            leximp: 1,
            lend: 0,
            coords: pt.coords,
            noel: pt.noel,
            npt: pt.npt,
            layer: 1,
            kspt: 1,
            kstep,
            kinc,
        };
        if let Err(e) = self.routine.creep(&mut a) {
            eprintln!("warning: CREEP for {}: {e}", self.cmname);
            return (0.0, 0.0);
        }
        (a.decra[0], a.decra[4])
    }
}

// ------------------------------------------------------------------ HETVAL

/// HETVAL(CMNAME, TEMP, TIME, DTIME, STATEV, FLUX, PREDEF, DPRED).
pub struct HetvalArgs<'a> {
    pub cmname: &'a str,
    /// TEMP(1) = temperature at the end of the increment, TEMP(2) = the change.
    pub temp: [f64; 2],
    pub time: [f64; 2],
    pub dtime: f64,
    pub statev: &'a mut [f64],
    /// FLUX(1) = heat generation per volume, FLUX(2) = d FLUX(1)/dT.
    pub flux: [f64; 2],
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
}
routine!(Hetval, HetvalArgs, hetval, HETVALS, register_hetval, find_hetval, super::samples2::hetvals);

// ------------------------------------------------------------------ FILM

/// FILM(H, SINK, TEMP, KSTEP, KINC, TIME, NOEL, NPT, COORDS, JLTYP, FIELD, NFIELD, SNAME, NODE, AREA).
pub struct FilmArgs<'a> {
    /// H(1) = film coefficient, H(2) = dH/dT.
    pub h: [f64; 2],
    pub sink: f64,
    pub temp: f64,
    pub kstep: i32,
    pub kinc: i32,
    pub time: [f64; 2],
    pub noel: i64,
    pub npt: usize,
    pub coords: [f64; 3],
    pub jltyp: i32,
    pub field: &'a [f64],
    pub sname: &'a str,
    pub node: i64,
    pub area: f64,
}
routine!(Film, FilmArgs, film, FILMS, register_film, find_film, super::samples2::films);

// ------------------------------------------------------------------ UMATHT

/// UMATHT(U, DUDT, DUDG, FLUX, DFDT, DFDG, STATEV, TEMP, DTEMP, DTEMDX, TIME, DTIME, PREDEF, DPRED, CMNAME, NTGRD, NSTATV, PROPS, ...).
pub struct UmathtArgs<'a> {
    /// Internal energy per unit mass (in: the value at the start of the increment) and dU/dT.
    pub u: f64,
    pub dudt: f64,
    pub dudg: &'a mut [f64],
    /// Heat flux (in: at the start of the increment), dF/dT and dF/d(∂T/∂x) (NTGRD × NTGRD column-wise).
    pub flux: &'a mut [f64],
    pub dfdt: &'a mut [f64],
    pub dfdg: &'a mut [f64],
    pub statev: &'a mut [f64],
    pub temp: f64,
    pub dtemp: f64,
    pub dtemdx: &'a [f64],
    pub time: [f64; 2],
    pub dtime: f64,
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
    pub cmname: &'a str,
    pub ntgrd: usize,
    pub props: &'a [f64],
    pub coords: [f64; 3],
    pub pnewdt: f64,
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub kstep: i32,
    pub kinc: i32,
}
routine!(Umatht, UmathtArgs, umatht, UMATHTS, register_umatht, find_umatht, super::samples2::umathts);
