//! Inställningar som tidigare bara fanns som miljövariabler (`RUSTYFEM_*`).
//!
//! CLI:t (feature `cli`) läser miljövariablerna en gång; utan `cli` (appen,
//! iOS) läses miljön aldrig och standardvärdena gäller. Appen sätter egna
//! värden via `RunOptions::settings`. Inställningarna gäller hela processen.

use std::sync::RwLock;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Settings {
    /// Linjär lösare när ingen anges (`RUSTYFEM_SOLVER`).
    pub solver: Option<String>,
    /// Relativ tolerans för PCG (`RUSTYFEM_ITER_TOL`).
    pub iter_tol: Option<f64>,
    /// Tider för assemblering och faktorisering till stderr (`RUSTYFEM_TIMING`).
    pub timing: bool,
    /// Inga varningar om indefinita system (`RUSTYFEM_QUIET`).
    pub quiet: bool,
    /// Felsökning av Newton-iterationerna vid NLGEOM (`RUSTYFEM_NLDEBUG`).
    pub nl_debug: bool,
    /// Felsökning av kontakt (`RUSTYFEM_CONTACT_DEBUG`).
    pub contact_debug: bool,
    /// Tangent med finita differenser i kontinuumkinematiken (`RUSTYFEM_NL_FDTANGENT`).
    pub nl_fd_tangent: bool,
    /// Bara spänningsdelen i timglasstabiliseringen (`RUSTYFEM_HG_OFF`).
    pub hourglass_off: bool,
    /// Hybridelementens tryckbas (`RUSTYFEM_HYB_NBAR`).
    pub hybrid_nbar: Option<usize>,
    /// Hybridelementens timglasstabilisering (`RUSTYFEM_HYB_HG` = 0/1).
    pub hybrid_hourglass: Option<bool>,
    /// Behåll arbetskataloger för regressionsfall (`RUSTYFEM_KEEP`).
    pub keep_work_dirs: bool,
}

impl Settings {
    /// Från miljövariablerna (bara CLI:t).
    pub fn from_env() -> Settings {
        let var = |k: &str| std::env::var(k).ok();
        let on = |k: &str| std::env::var_os(k).is_some();
        Settings {
            solver: var("RUSTYFEM_SOLVER"),
            iter_tol: var("RUSTYFEM_ITER_TOL").and_then(|s| s.parse().ok()),
            timing: on("RUSTYFEM_TIMING"),
            quiet: on("RUSTYFEM_QUIET"),
            nl_debug: on("RUSTYFEM_NLDEBUG"),
            contact_debug: on("RUSTYFEM_CONTACT_DEBUG"),
            nl_fd_tangent: on("RUSTYFEM_NL_FDTANGENT"),
            hourglass_off: on("RUSTYFEM_HG_OFF"),
            hybrid_nbar: var("RUSTYFEM_HYB_NBAR").and_then(|s| s.parse().ok()),
            hybrid_hourglass: var("RUSTYFEM_HYB_HG").map(|s| s == "1"),
            keep_work_dirs: on("RUSTYFEM_KEEP"),
        }
    }
}

static CURRENT: RwLock<Option<Settings>> = RwLock::new(None);

/// Läser de aktuella inställningarna.
pub fn with<T>(f: impl FnOnce(&Settings) -> T) -> T {
    if let Some(s) = CURRENT.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return f(s);
    }
    let mut w = CURRENT.write().unwrap_or_else(|e| e.into_inner());
    let s = w.get_or_insert_with(|| if cfg!(feature = "cli") { Settings::from_env() } else { Settings::default() });
    f(s)
}

/// Sätter inställningarna för processen.
pub fn set(s: Settings) {
    *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = Some(s);
}
