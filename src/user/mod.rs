//! Användarrutiner: UMAT (användarmaterial) och UEL (användarelement) med
//! det etablerade Fortran-gränssnittet.
//!
//! - `umat`: `Umat`-traiten och `UmatArgs` (argumenten), register per
//!   materialnamnsprefix. Kopplas till *USER MATERIAL/*DEPVAR av
//!   `materials/user_material.rs`.
//! - `uel`: `Uel`-traiten och `UelArgs`, register per typnummer (TYPE=Un).
//!   Kopplas till *USER ELEMENT/*UEL PROPERTY av `elements/user_element.rs`.
//! - `dynlib`: laddning av delade bibliotek (LoadLibraryW/dlopen via FFI).
//! - `samples`: exempel i Rust: linjärt elastisk UMAT, von Mises-UMAT med
//!   isotropt hårdnande, stång-UEL och fyrnodig plan-spänning-UEL.
//!
//! Två sätt att ge egna rutiner:
//! 1. I Rust: implementera `umat::Umat`/`uel::Uel` och registrera med
//!    `umat::register(prefix, Arc::new(...))` respektive
//!    `uel::register(Some(n), Arc::new(...))` innan modellen körs.
//! 2. Ett delat bibliotek (kompilerad Fortran eller C) som exporterar `umat_`
//!    och/eller `uel_` (även `umat`, `UMAT`, `uel`, `UEL` provas) med
//!    standardargumentlistan, alla argument som referenser (Fortrans konvention),
//!    CMNAME:s längd som dolt sista argument. Laddas med `load_library` (och
//!    registreras sedan som vilken Rust-rutin som helst) eller med
//!    miljövariabeln `RUSTYFEM_USER=sökväg`, som används
//!    för alla material och elementtyper utan egen registrering.

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

/// Ett laddat bibliotek med användarrutiner.
pub struct Library {
    pub lib: Arc<DynLib>,
    umat: Option<Arc<dyn umat::Umat>>,
    uel: Option<Arc<dyn uel::Uel>>,
}

impl Library {
    /// UMAT i biblioteket (om symbolen finns).
    pub fn umat(&self) -> Option<Arc<dyn umat::Umat>> {
        self.umat.clone()
    }

    /// UEL i biblioteket (om symbolen finns).
    pub fn uel(&self) -> Option<Arc<dyn uel::Uel>> {
        self.uel.clone()
    }
}

/// Laddar ett delat bibliotek och slår upp `umat_`/`uel_`. Fel om ingen av
/// dem finns.
pub fn load_library(path: &Path) -> Result<Library> {
    let lib = Arc::new(DynLib::open(path)?);
    let umat = lib.symbol(&["umat_", "umat", "UMAT", "UMAT_"]).map(|(p, s)| {
        // SAFETY: symbolen antas ha UMAT-signaturen (dokumenterat krav)
        let f: umat::UmatFn = unsafe { std::mem::transmute::<*mut std::ffi::c_void, umat::UmatFn>(p) };
        Arc::new(umat::DllUmat::new(f, lib.clone(), s)) as Arc<dyn umat::Umat>
    });
    let uel = lib.symbol(&["uel_", "uel", "UEL", "UEL_"]).map(|(p, s)| {
        // SAFETY: symbolen antas ha UEL-signaturen (dokumenterat krav)
        let f: uel::UelFn = unsafe { std::mem::transmute::<*mut std::ffi::c_void, uel::UelFn>(p) };
        Arc::new(uel::DllUel::new(f, lib.clone(), s)) as Arc<dyn uel::Uel>
    });
    if umat.is_none() && uel.is_none() {
        return crate::model::fail(format!("{} innehåller varken umat_ eller uel_ (eller UMAT/UEL)", path.display()));
    }
    Ok(Library { lib, umat, uel })
}

/// Biblioteket i miljövariabeln `RUSTYFEM_USER` (laddas första gången det behövs).
pub fn env_library() -> Result<Option<&'static Library>> {
    static L: OnceLock<std::result::Result<Option<Library>, String>> = OnceLock::new();
    // miljön läses bara av CLI:t (feature `cli`)
    let r = L.get_or_init(|| match std::env::var("RUSTYFEM_USER").ok().filter(|_| cfg!(feature = "cli")).ok_or(()) {
        Ok(p) if !p.trim().is_empty() => load_library(Path::new(p.trim())).map(Some).map_err(|e| format!("RUSTYFEM_USER: {e}")),
        _ => Ok(None),
    });
    match r {
        Ok(l) => Ok(l.as_ref()),
        Err(e) => crate::model::fail(e.clone()),
    }
}
