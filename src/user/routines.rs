//! Fler användarrutiner med det etablerade Fortran-gränssnittet (utöver UMAT, UEL och VUMAT):
//!
//! | Rutin | Indata | Väljs med (längsta prefix) |
//! |---|---|---|
//! | DLOAD | `*DLOAD` med lasttypen PnNU (tryck på yta n) | elementsetets namn på dataraden |
//! | DISP | `*BOUNDARY, USER` | nodsetets namn på dataraden |
//! | UAMP | `*AMPLITUDE, DEFINITION=USER` | amplitudens namn |
//! | USDFLD | `*USER DEFINED FIELD` i materialet | materialnamnet |
//! | UFIELD | `*FIELD, USER` | nodsetets namn på dataraden |
//! | UHARD | `*PLASTIC, HARDENING=USER` | materialnamnet |
//! | SIGINI | `*INITIAL CONDITIONS, TYPE=STRESS, USER` | elementsetets namn på dataraden |
//! | UEXPAN | `*EXPANSION, USER` | materialnamnet |
//! | CREEP | `*CREEP, LAW=USER` | materialnamnet |
//! | HETVAL | `*HEAT GENERATION` i materialet | materialnamnet |
//! | FILM | `*FILM` med lasttypen FnNU | elementsetets namn på dataraden |
//! | UMATHT | `*USER MATERIAL, TYPE=THERMAL` | materialnamnet |
//!
//! Argumenten har de etablerade namnen och betydelserna. Index är 0-baserade i Rust; TIME = [stegtid, total tid]
//! vid inkrementets början (för DLOAD, DISP, UFIELD, FILM och UAMP: vid den
//! tid värdet gäller). En rutin skrivs i Rust genom att implementera traiten
//! och registreras med `register_<rutin>(prefix, Arc::new(...))` före
//! körningen; tomt prefix = alla. Exempel finns i `samples2.rs`.
//! Rutiner i ett delat bibliotek (Fortrans konvention) laddas med
//! `load_routines` (`dynlib2.rs`) eller `RUSTYFEM_USER`.

use crate::model::Result;
use std::sync::{Arc, OnceLock, RwLock};

/// Register av rutiner per namnprefix (versaler).
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

    /// Registrerar `r` för namn som börjar med `prefix` (ersätter samma prefix).
    pub fn register(&self, prefix: &str, r: Arc<T>) {
        let p = prefix.trim().to_uppercase();
        let mut v = self.get().write().unwrap_or_else(|e| e.into_inner());
        v.retain(|(q, _)| *q != p);
        v.push((p, r));
    }

    /// Rutinen för `name` (längsta registrerade prefix).
    pub fn find(&self, name: &str) -> Option<Arc<T>> {
        let up = name.trim().to_uppercase();
        let v = self.get().read().unwrap_or_else(|e| e.into_inner());
        v.iter().filter(|(p, _)| up.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, r)| r.clone())
    }
}

/// Felmeddelande när ingen rutin finns.
pub fn missing<T>(routine: &str, what: &str, name: &str) -> Result<T> {
    crate::model::fail(format!(
        "ingen {routine} för {what} {name}: registrera en i Rust med rustyfem::user::routines::register_{} (se user/samples2.rs) eller ange ett bibliotek med {}_ i RUSTYFEM_USER",
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
        /// Registrerar rutinen för namn som börjar med `prefix`.
        pub fn $register(prefix: &str, r: Arc<dyn $trait_>) {
            $reg.register(prefix, r);
        }
        /// Rutinen för `name`: registrerad (längsta prefix) eller biblioteket i `RUSTYFEM_USER`.
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
    /// Lastens storlek i punkten (in: storleken på dataraden).
    pub f: f64,
    pub kstep: i32,
    pub kinc: i32,
    pub time: [f64; 2],
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    pub coords: [f64; 3],
    /// Lasttyp: 20 + n för PnNU.
    pub jltyp: i32,
    /// Ytans namn (tomt för elementbaserade laster).
    pub sname: &'a str,
}
routine!(Dload, DloadArgs, dload, DLOADS, register_dload, find_dload, super::samples2::dloads);

// ------------------------------------------------------------------ DISP

/// DISP(U, KSTEP, KINC, TIME, NODE, NOEL, JDOF, COORDS).
pub struct DispArgs<'a> {
    /// U(1) = föreskrivet värde vid inkrementets slut (in: värdet på dataraden),
    /// U(2), U(3) = hastighet och acceleration (används inte i statik).
    pub u: [f64; 3],
    pub kstep: i32,
    pub kinc: i32,
    /// [stegtid, total tid] vid inkrementets slut.
    pub time: [f64; 2],
    pub node: i64,
    pub noel: i64,
    pub jdof: i32,
    pub coords: [f64; 3],
    /// Nodsetet på dataraden (tomt för ett nodnummer).
    pub set: &'a str,
}
routine!(Disp, DispArgs, disp, DISPS, register_disp, find_disp, super::samples2::disps);

// ------------------------------------------------------------------ UAMP

/// UAMP (förenklat): amplitudens värde och derivator vid en tidpunkt.
pub struct UampArgs<'a> {
    pub amp_name: &'a str,
    /// [stegtid, total tid] där värdet söks.
    pub time: [f64; 2],
    /// Konstanterna på *AMPLITUDE-raderna (PROPS).
    pub props: &'a [f64],
    pub amp_value_new: f64,
    pub amp_derivative: f64,
    pub amp_sec_derivative: f64,
    pub kstep: i32,
}
routine!(Uamp, UampArgs, uamp, UAMPS, register_uamp, find_uamp, super::samples2::uamps);

/// *AMPLITUDE, DEFINITION=USER: rutinen och PROPS (dataraderna).
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
            return missing("UAMP", "amplituden", name);
        };
        Ok(UampRef { name: name.to_string(), props, routine })
    }

    /// (A, A′, A″) vid stegtid och total tid. Fel i rutinen ger A = 0 (med varning).
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
            eprintln!("varning: UAMP {}: {e}", self.name);
            return (0.0, 0.0, 0.0);
        }
        (a.amp_value_new, a.amp_derivative, a.amp_sec_derivative)
    }
}

// ------------------------------------------------------------------ USDFLD

/// USDFLD(FIELD, STATEV, PNEWDT, DIRECT, T, CELENT, TIME, DTIME, CMNAME, ...).
/// Spänningen och töjningen vid inkrementets början ersätter GETVRM('S'/'E').
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
    /// Temperaturen i punkten vid inkrementets slut.
    pub temp: f64,
    /// Spänning och total töjning vid inkrementets början (ordning 11, 22, 33, 12, 13, 23).
    pub stress: &'a [f64],
    pub strain: &'a [f64],
}
routine!(Usdfld, UsdfldArgs, usdfld, USDFLDS, register_usdfld, find_usdfld, super::samples2::usdflds);

// ------------------------------------------------------------------ UFIELD

/// UFIELD(FIELD, KFIELD, NSECPT, KSTEP, KINC, TIME, NODE, COORDS, TEMP, DTEMP, NFIELD).
pub struct UfieldArgs<'a> {
    /// Alla fältvariabler i noden (in: värdena utan rutinen).
    pub field: &'a mut [f64],
    /// Variabeln som *FIELD, USER definierar (VARIABLE=, 1..).
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
    /// HARD(1) = dσy/dε̄p, HARD(2) = dσy/d(ε̄p-hastighet), HARD(3) = dσy/dT.
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
    /// NAMES(1) = armeringens namn (tomt), NAMES(2) = elementsetet.
    pub names: [&'a str; 2],
}
routine!(Sigini, SiginiArgs, sigini, SIGINIS, register_sigini, find_sigini, super::samples2::siginis);

// ------------------------------------------------------------------ UEXPAN

/// UEXPAN(EXPAN, DEXPANDT, TEMP, TIME, DTIME, PREDEF, DPRED, STATEV, CMNAME, NSTATV, NOEL).
pub struct UexpanArgs<'a> {
    /// Termiska töjningsinkrement (1 isotropt, 3 ortotropt, 6 anisotropt; tensorskjuvning).
    pub expan: &'a mut [f64],
    pub dexpandt: &'a mut [f64],
    /// TEMP(1) = temperatur vid inkrementets slut, TEMP(2) = ändringen.
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
    /// DECRA(1) = Δε̄cr, DECRA(2..5) = derivator m.a.p. ε̄cr, ε̄sw, p, q̃ (vid LEXIMP = 1).
    pub decra: [f64; 5],
    pub deswa: [f64; 5],
    pub statev: &'a mut [f64],
    pub serd: f64,
    /// EC(1), EC(2) = ε̄cr vid inkrementets början och slut.
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
    /// 0 = explicit, 1 = implicit integration (derivatorna behövs).
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

/// *CREEP, LAW=USER: rutinen för ett material.
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
            return missing("CREEP", "materialet", cmname);
        };
        Ok(CreepRef { cmname: cmname.to_string(), routine })
    }

    /// (Δε̄cr, ∂Δε̄cr/∂q̃) för q̃ = `q`, ε̄cr = `ecr` vid inkrementets början och
    /// tiden `t` (total tid eller kryptid) vid början. Trycket P skickas som 0.
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
            eprintln!("varning: CREEP för {}: {e}", self.cmname);
            return (0.0, 0.0);
        }
        (a.decra[0], a.decra[4])
    }
}

// ------------------------------------------------------------------ HETVAL

/// HETVAL(CMNAME, TEMP, TIME, DTIME, STATEV, FLUX, PREDEF, DPRED).
pub struct HetvalArgs<'a> {
    pub cmname: &'a str,
    /// TEMP(1) = temperatur vid inkrementets slut, TEMP(2) = ändringen.
    pub temp: [f64; 2],
    pub time: [f64; 2],
    pub dtime: f64,
    pub statev: &'a mut [f64],
    /// FLUX(1) = värmegenerering per volym, FLUX(2) = d FLUX(1)/dT.
    pub flux: [f64; 2],
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
}
routine!(Hetval, HetvalArgs, hetval, HETVALS, register_hetval, find_hetval, super::samples2::hetvals);

// ------------------------------------------------------------------ FILM

/// FILM(H, SINK, TEMP, KSTEP, KINC, TIME, NOEL, NPT, COORDS, JLTYP, FIELD, NFIELD, SNAME, NODE, AREA).
pub struct FilmArgs<'a> {
    /// H(1) = filmkoefficient, H(2) = dH/dT.
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
    /// Inre energi per massenhet (in: värdet vid inkrementets början) och dU/dT.
    pub u: f64,
    pub dudt: f64,
    pub dudg: &'a mut [f64],
    /// Värmeflöde (in: vid inkrementets början), dF/dT och dF/d(∂T/∂x) (NTGRD × NTGRD kolumnvis).
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
