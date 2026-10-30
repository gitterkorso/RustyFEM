//! UEL: user elements with the established UEL interface.
//!
//! The element type is defined in the .inp file with *USER ELEMENT, TYPE=Un, NODES=,
//! COORDINATES=, PROPERTIES=, I PROPERTIES=, VARIABLES= (and UNSYMM) followed
//! by the active degrees of freedom per node, and the properties with *UEL PROPERTY,
//! ELSET= (see `elements/user_element.rs`). The element itself is written in Rust
//! by implementing `Uel` and registering it for type number n with
//! `register(Some(n), ...)` (or `None` for all types), or loaded from
//! a shared library (symbol `uel_`/`UEL`) with `super::load_library`
//! or `RUSTYFEM_USER`.
//!
//! The arguments have the established names and meanings:
//!
//! | Argument | Content |
//! |---|---|
//! | RHS(MLVARX, NRHS) | the residual vector −F_int (NRHS = 1) |
//! | AMATRX(NDOFEL, NDOFEL) | Jacobian ∂F_int/∂u (stiffness) or mass matrix, column-wise |
//! | SVARS(NSVARS) | state variables at start → end |
//! | ENERGY(8) | energies (ENERGY(2) = elastic strain energy, ...) |
//! | PROPS, JPROPS | real and integer properties from *UEL PROPERTY |
//! | COORDS(MCRD, NNODE) | nodal coordinates in the reference configuration, column-wise |
//! | U, DU(MLVARX, 1) | total displacement and change during the increment |
//! | V, A | velocity and acceleration (0 in static steps) |
//! | JTYPE | n in TYPE=Un |
//! | TIME(2), DTIME | step time and total time at the start, increment size |
//! | JELEM | element number |
//! | PREDEF(2, NPREDF, NNODE) | temperature and its change per node (NPREDF = 1) |
//! | LFLAGS(5) | (1) procedure: 1 = static, 41 = natural frequency; (2) NLGEOM 0/1; (3) what to compute: 1 = RHS and stiffness, 2 = stiffness, 4 = mass, 5 = RHS only; (4) perturbation 0/1; (5) 0 = Newton |
//! | PNEWDT | set < 1 to request a smaller increment |
//!
//! The degrees of freedom in U, RHS and AMATRX are node by node: for node 1
//! its active degrees of freedom in ascending order, then node 2 etc.
//!
//! RustyFEM's element interface lacks the step number, step time and distributed
//! loads: KSTEP = KINC = 0, TIME(1) = TIME(2) = total time at the start of the
//! increment, PERIOD = 0, NDLOAD = 0. Loads (*DLOAD) on user elements are not supported.

use super::dynlib::DynLib;
use crate::model::{fail, Result};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// The arguments to a UEL (the Fortran names).
pub struct UelArgs<'a> {
    /// RHS(MLVARX, NRHS), here NRHS = 1.
    pub rhs: &'a mut [f64],
    /// AMATRX(NDOFEL, NDOFEL) column-wise: entry (i, j) = `amatrx[i + j * ndofel]`.
    pub amatrx: &'a mut [f64],
    pub svars: &'a mut [f64],
    pub energy: &'a mut [f64; 8],
    pub ndofel: usize,
    pub nrhs: usize,
    pub nsvars: usize,
    pub props: &'a [f64],
    pub jprops: &'a [i32],
    /// COORDS(MCRD, NNODE) column-wise: coordinate i of node k = `coords[i + k * mcrd]`.
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
    /// PREDEF(2, NPREDF, NNODE) column-wise with NPREDF = 1: (temperature, change) per node.
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

    /// Adds `v` to AMATRX(i+1, j+1).
    pub fn add_k(&mut self, i: usize, j: usize, v: f64) {
        let n = self.ndofel;
        self.amatrx[i + j * n] += v;
    }

    /// Coordinate i (0-based) of node k (0-based).
    pub fn x(&self, k: usize, i: usize) -> f64 {
        self.coords[i + k * self.mcrd]
    }
}

/// A user element with UEL semantics.
pub trait Uel: Send + Sync {
    fn name(&self) -> String {
        "UEL".into()
    }

    /// Fills RHS, AMATRX (depending on LFLAGS(3)), SVARS and ENERGY.
    fn uel(&self, a: &mut UelArgs) -> Result<()>;
}

type Registry = RwLock<Vec<(Option<u32>, Arc<dyn Uel>)>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| RwLock::new(super::samples::uels()))
}

/// Registers a UEL for type U`jtype` (`None` = all types without
/// their own registration). A later registration replaces an earlier one.
pub fn register(jtype: Option<u32>, uel: Arc<dyn Uel>) {
    let mut r = registry().write().unwrap_or_else(|e| e.into_inner());
    r.retain(|(t, _)| *t != jtype);
    r.push((jtype, uel));
}

/// The UEL for type U`jtype`: own registration, registration for all types,
/// otherwise the library in `RUSTYFEM_USER`.
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

/// UEL in the Fortran calling convention (all arguments as pointers).
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

/// UEL in a shared library.
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
        format!("{} in {}", self.symbol, self.lib.path)
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
            return fail(format!("UEL {} returned RHS or AMATRX values that are not finite numbers", self.symbol));
        }
        Ok(())
    }
}
