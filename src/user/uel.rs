//! UEL: användarelement med det etablerade UEL-gränssnittet.
//!
//! Elementtypen definieras i .inp-filen med *USER ELEMENT, TYPE=Un, NODES=,
//! COORDINATES=, PROPERTIES=, I PROPERTIES=, VARIABLES= (och UNSYMM) följt
//! av de aktiva frihetsgraderna per nod, och egenskaperna med *UEL PROPERTY,
//! ELSET= (se `elements/user_element.rs`). Själva elementet skrivs i Rust
//! genom att implementera `Uel` och registrera det för typnumret n med
//! `register(Some(n), ...)` (eller `None` för alla typer), eller laddas från
//! ett delat bibliotek (symbolen `uel_`/`UEL`) med `super::load_library`
//! eller `RUSTYFEM_USER`.
//!
//! Argumenten har de etablerade namnen och betydelserna:
//!
//! | Argument | Innehåll |
//! |---|---|
//! | RHS(MLVARX, NRHS) | residualvektorn −F_int (NRHS = 1) |
//! | AMATRX(NDOFEL, NDOFEL) | Jacobian ∂F_int/∂u (styvhet) eller massmatris, kolumnvis |
//! | SVARS(NSVARS) | tillståndsvariabler vid början → slut |
//! | ENERGY(8) | energier (ENERGY(2) = elastisk töjningsenergi, ...) |
//! | PROPS, JPROPS | reella och heltaliga egenskaper från *UEL PROPERTY |
//! | COORDS(MCRD, NNODE) | nodernas koordinater i referensläget, kolumnvis |
//! | U, DU(MLVARX, 1) | total förskjutning och ändring under inkrementet |
//! | V, A | hastighet och acceleration (0 i statiska steg) |
//! | JTYPE | n i TYPE=Un |
//! | TIME(2), DTIME | stegtid och total tid vid början, inkrementets längd |
//! | JELEM | elementnummer |
//! | PREDEF(2, NPREDF, NNODE) | temperatur och dess ändring per nod (NPREDF = 1) |
//! | LFLAGS(5) | (1) procedur: 1 = statisk, 41 = egenfrekvens; (2) NLGEOM 0/1; (3) vad som ska beräknas: 1 = RHS och styvhet, 2 = styvhet, 4 = massa, 5 = bara RHS; (4) perturbation 0/1; (5) 0 = Newton |
//! | PNEWDT | sätts < 1 för att begära mindre inkrement |
//!
//! Frihetsgraderna i U, RHS och AMATRX är nodvis: för nod 1
//! dess aktiva frihetsgrader i stigande nummer, sedan nod 2 osv.
//!
//! RustyFEMs elementgränssnitt saknar stegnummer, stegtid och fördelade
//! laster: KSTEP = KINC = 0, TIME(1) = TIME(2) = total tid vid inkrementets
//! början, PERIOD = 0, NDLOAD = 0. Laster (*DLOAD) på användarelement stöds inte.

use super::dynlib::DynLib;
use crate::model::{fail, Result};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// Argumenten till en UEL (Fortran-namnen).
pub struct UelArgs<'a> {
    /// RHS(MLVARX, NRHS), här NRHS = 1.
    pub rhs: &'a mut [f64],
    /// AMATRX(NDOFEL, NDOFEL) kolumnvis: element (i, j) = `amatrx[i + j * ndofel]`.
    pub amatrx: &'a mut [f64],
    pub svars: &'a mut [f64],
    pub energy: &'a mut [f64; 8],
    pub ndofel: usize,
    pub nrhs: usize,
    pub nsvars: usize,
    pub props: &'a [f64],
    pub jprops: &'a [i32],
    /// COORDS(MCRD, NNODE) kolumnvis: koordinat i för nod k = `coords[i + k * mcrd]`.
    pub coords: &'a [f64],
    pub mcrd: usize,
    pub nnode: usize,
    pub u: &'a [f64],
    pub du: &'a [f64],
    pub v: &'a [f64],
    pub a: &'a [f64],
    pub jtype: u32,
    pub time: [f64; 2],
    pub dtime: f64,
    pub kstep: usize,
    pub kinc: usize,
    pub jelem: i64,
    pub params: [f64; 3],
    /// PREDEF(2, NPREDF, NNODE) kolumnvis med NPREDF = 1: (temperatur, ändring) per nod.
    pub predef: &'a [f64],
    pub npredf: usize,
    pub lflags: [i32; 5],
    pub mlvarx: usize,
    pub pnewdt: &'a mut f64,
    pub period: f64,
}

impl UelArgs<'_> {
    /// AMATRX(i+1, j+1).
    pub fn k(&self, i: usize, j: usize) -> f64 {
        self.amatrx[i + j * self.ndofel]
    }

    /// Lägger `v` till AMATRX(i+1, j+1).
    pub fn add_k(&mut self, i: usize, j: usize, v: f64) {
        let n = self.ndofel;
        self.amatrx[i + j * n] += v;
    }

    /// Koordinat i (0-baserad) för nod k (0-baserad).
    pub fn x(&self, k: usize, i: usize) -> f64 {
        self.coords[i + k * self.mcrd]
    }
}

/// Ett användarelement med UEL:s semantik.
pub trait Uel: Send + Sync {
    fn name(&self) -> String {
        "UEL".into()
    }

    /// Fyller RHS, AMATRX (beroende på LFLAGS(3)), SVARS och ENERGY.
    fn uel(&self, a: &mut UelArgs) -> Result<()>;
}

type Registry = RwLock<Vec<(Option<u32>, Arc<dyn Uel>)>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| RwLock::new(super::samples::uels()))
}

/// Registrerar en UEL för typen U`jtype` (`None` = alla typer som inte har
/// en egen registrering). En senare registrering ersätter en tidigare.
pub fn register(jtype: Option<u32>, uel: Arc<dyn Uel>) {
    let mut r = registry().write().unwrap_or_else(|e| e.into_inner());
    r.retain(|(t, _)| *t != jtype);
    r.push((jtype, uel));
}

/// UEL för typen U`jtype`: egen registrering, registrering för alla typer,
/// annars biblioteket i `RUSTYFEM_USER`.
pub fn find(jtype: u32) -> Result<Option<Arc<dyn Uel>>> {
    {
        let r = registry().read().unwrap_or_else(|e| e.into_inner());
        if let Some((_, u)) = r.iter().find(|(t, _)| *t == Some(jtype)).or_else(|| r.iter().find(|(t, _)| t.is_none())) {
            return Ok(Some(u.clone()));
        }
    }
    match super::env_library()? {
        Some(lib) => Ok(lib.uel()),
        None => Ok(None),
    }
}

/// UEL i Fortrans anropskonvention (alla argument som pekare).
pub type UelFn = unsafe extern "C" fn(
    rhs: *mut f64,
    amatrx: *mut f64,
    svars: *mut f64,
    energy: *mut f64,
    ndofel: *const i32,
    nrhs: *const i32,
    nsvars: *const i32,
    props: *const f64,
    nprops: *const i32,
    coords: *const f64,
    mcrd: *const i32,
    nnode: *const i32,
    u: *const f64,
    du: *const f64,
    v: *const f64,
    a: *const f64,
    jtype: *const i32,
    time: *const f64,
    dtime: *const f64,
    kstep: *const i32,
    kinc: *const i32,
    jelem: *const i32,
    params: *const f64,
    ndload: *const i32,
    jdltyp: *const i32,
    adlmag: *const f64,
    predef: *const f64,
    npredf: *const i32,
    lflags: *const i32,
    mlvarx: *const i32,
    ddlmag: *const f64,
    mdload: *const i32,
    pnewdt: *mut f64,
    jprops: *const i32,
    njprop: *const i32,
    period: *const f64,
);

/// UEL i ett delat bibliotek.
pub struct DllUel {
    pub f: UelFn,
    pub lib: Arc<DynLib>,
    pub symbol: String,
    lock: Mutex<()>,
}

impl DllUel {
    pub fn new(f: UelFn, lib: Arc<DynLib>, symbol: String) -> Self {
        DllUel { f, lib, symbol, lock: Mutex::new(()) }
    }
}

impl Uel for DllUel {
    fn name(&self) -> String {
        format!("{} i {}", self.symbol, self.lib.path)
    }

    fn uel(&self, a: &mut UelArgs) -> Result<()> {
        let c = |v: usize| v as i32;
        let (ndofel, nrhs, nsvars, nprops, mcrd, nnode) = (c(a.ndofel), c(a.nrhs), c(a.nsvars), a.props.len() as i32, c(a.mcrd), c(a.nnode));
        let (jtype, kstep, kinc, jelem, npredf, mlvarx, njprop) = (a.jtype as i32, c(a.kstep), c(a.kinc), a.jelem as i32, c(a.npredf), c(a.mlvarx), a.jprops.len() as i32);
        let (ndload, mdload) = (0i32, 1i32);
        let jdltyp = [0i32; 1];
        let adlmag = [0.0f64; 1];
        let ddlmag = [0.0f64; 1];
        let one = [0.0f64; 1];
        let ione = [0i32; 1];
        let mut sv_dummy = [0.0f64; 1];
        let svars: *mut f64 = if a.svars.is_empty() { sv_dummy.as_mut_ptr() } else { a.svars.as_mut_ptr() };
        let ptr = |s: &[f64]| if s.is_empty() { one.as_ptr() } else { s.as_ptr() };
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(
                a.rhs.as_mut_ptr(),
                a.amatrx.as_mut_ptr(),
                svars,
                a.energy.as_mut_ptr(),
                &ndofel,
                &nrhs,
                &nsvars,
                ptr(a.props),
                &nprops,
                a.coords.as_ptr(),
                &mcrd,
                &nnode,
                a.u.as_ptr(),
                a.du.as_ptr(),
                a.v.as_ptr(),
                a.a.as_ptr(),
                &jtype,
                a.time.as_ptr(),
                &a.dtime,
                &kstep,
                &kinc,
                &jelem,
                a.params.as_ptr(),
                &ndload,
                jdltyp.as_ptr(),
                adlmag.as_ptr(),
                ptr(a.predef),
                &npredf,
                a.lflags.as_ptr(),
                &mlvarx,
                ddlmag.as_ptr(),
                &mdload,
                &mut *a.pnewdt,
                if a.jprops.is_empty() { ione.as_ptr() } else { a.jprops.as_ptr() },
                &njprop,
                &a.period,
            );
        }
        if a.rhs.iter().chain(a.amatrx.iter()).any(|v| !v.is_finite()) {
            return fail(format!("UEL {} gav RHS eller AMATRX som inte är ändliga tal", self.symbol));
        }
        Ok(())
    }
}
