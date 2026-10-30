//! User subroutines: UMAT (user material) and UEL (user element) with the
//! established Fortran interface.
//!
//! - `umat`: the `Umat` trait and `UmatArgs` (the arguments), registry per
//!   material-name prefix. Hooked up to *USER MATERIAL/*DEPVAR by
//!   `materials/user_material.rs`.
//! - `uel`: the `Uel` trait and `UelArgs`, registry per type number (TYPE=Un).
//!   Hooked up to *USER ELEMENT/*UEL PROPERTY by `elements/user_element.rs`.
//! - `dynlib`: loading of shared libraries (LoadLibraryW/dlopen via FFI).
//! - `samples`: examples in Rust: linear elastic UMAT, von Mises UMAT with
//!   isotropic hardening, truss UEL and four-node plane-stress UEL.
//!
//! Two ways to supply your own routines:
//! 1. In Rust: implement `umat::Umat`/`uel::Uel` and register with
//!    `umat::register(prefix, Arc::new(...))` or
//!    `uel::register(Some(n), Arc::new(...))` respectively, before the model runs.
//! 2. A shared library (compiled Fortran or C) exporting `umat_`
//!    and/or `uel_` (`umat`, `UMAT`, `uel`, `UEL` are also tried) with the
//!    standard argument list, all arguments by reference (Fortran convention),
//!    and the length of CMNAME as a hidden last argument. Loaded with `load_library` (and
//!    then registered like any Rust routine) or with the
//!    environment variable `RUSTYFEM_USER=path`, which is used
//!    for all materials and element types without their own registration.

pub mod context;
pub mod dynlib;
pub mod dynlib2;
pub mod loads;
pub mod routines;
pub mod samples;
pub mod samples2;
pub mod uel;
pub mod umat;
pub mod vumat;

use crate::model::Result;
use dynlib::DynLib;
use std::path::Path;
use std::sync::{Arc, OnceLock};

/// A loaded library of user subroutines.
pub struct Library {
    pub lib: Arc<DynLib>,
    umat: Option<Arc<dyn umat::Umat>>,
    uel: Option<Arc<dyn uel::Uel>>,
}

impl Library {
    /// The UMAT in the library (if the symbol exists).
    pub fn umat(&self) -> Option<Arc<dyn umat::Umat>> {
        self.umat.clone()
    }

    /// The UEL in the library (if the symbol exists).
    pub fn uel(&self) -> Option<Arc<dyn uel::Uel>> {
        self.uel.clone()
    }
}

/// Loads a shared library and looks up `umat_`/`uel_`. Error if neither
/// exists.
pub fn load_library(path: &Path) -> Result<Library> {
    let lib = Arc::new(DynLib::open(path)?);
    let umat = lib.symbol(&["umat_", "umat", "UMAT", "UMAT_"]).map(|(p, s)| {
        // SAFETY: the symbol is assumed to have the UMAT signature (documented requirement)
        let f: umat::UmatFn = unsafe { std::mem::transmute::<*mut std::ffi::c_void, umat::UmatFn>(p) };
        Arc::new(umat::DllUmat::new(f, lib.clone(), s)) as Arc<dyn umat::Umat>
    });
    let uel = lib.symbol(&["uel_", "uel", "UEL", "UEL_"]).map(|(p, s)| {
        // SAFETY: the symbol is assumed to have the UEL signature (documented requirement)
        let f: uel::UelFn = unsafe { std::mem::transmute::<*mut std::ffi::c_void, uel::UelFn>(p) };
        Arc::new(uel::DllUel::new(f, lib.clone(), s)) as Arc<dyn uel::Uel>
    });
    if umat.is_none() && uel.is_none() {
        return crate::model::fail(format!("{} contains neither umat_ nor uel_ (nor UMAT/UEL)", path.display()));
    }
    Ok(Library { lib, umat, uel })
}

/// The library named by the environment variable `RUSTYFEM_USER` (loaded the first time it is needed).
pub fn env_library() -> Result<Option<&'static Library>> {
    static L: OnceLock<std::result::Result<Option<Library>, String>> = OnceLock::new();
    // the environment is only read by the CLI (feature `cli`)
    let r = L.get_or_init(|| match std::env::var("RUSTYFEM_USER").ok().filter(|_| cfg!(feature = "cli")).ok_or(()) {
        Ok(p) if !p.trim().is_empty() => load_library(Path::new(p.trim())).map(Some).map_err(|e| format!("RUSTYFEM_USER: {e}")),
        _ => Ok(None),
    });
    match r {
        Ok(l) => Ok(l.as_ref()),
        Err(e) => crate::model::fail(e.clone()),
    }
}
