//! Settings that used to exist only as environment variables (`RUSTYFEM_*`).
//!
//! The CLI (feature `cli`) reads the environment variables once; without `cli` (the app,
//! iOS) the environment is never read and the defaults apply. The app sets its own
//! values via `RunOptions::settings`. The settings apply to the whole process.

use std::sync::RwLock;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Settings {
    /// Linear solver when none is specified (`RUSTYFEM_SOLVER`).
    pub solver: Option<String>,
    /// Relative tolerance for PCG (`RUSTYFEM_ITER_TOL`).
    pub iter_tol: Option<f64>,
    /// Assembly and factorization timings to stderr (`RUSTYFEM_TIMING`).
    pub timing: bool,
    /// No warnings about indefinite systems (`RUSTYFEM_QUIET`).
    pub quiet: bool,
    /// Debugging of the Newton iterations with NLGEOM (`RUSTYFEM_NLDEBUG`).
    pub nl_debug: bool,
    /// Contact debugging (`RUSTYFEM_CONTACT_DEBUG`).
    pub contact_debug: bool,
    /// Finite-difference tangent in the continuum kinematics (`RUSTYFEM_NL_FDTANGENT`).
    pub nl_fd_tangent: bool,
    /// Only the stress part of the hourglass stabilization (`RUSTYFEM_HG_OFF`).
    pub hourglass_off: bool,
    /// Pressure basis of the hybrid elements (`RUSTYFEM_HYB_NBAR`).
    pub hybrid_nbar: Option<usize>,
    /// Hourglass stabilization of the hybrid elements (`RUSTYFEM_HYB_HG` = 0/1).
    pub hybrid_hourglass: Option<bool>,
    /// Keep work directories for regression cases (`RUSTYFEM_KEEP`).
    pub keep_work_dirs: bool,
}

impl Settings {
    /// From the environment variables (CLI only).
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

/// Reads the current settings.
pub fn with<T>(f: impl FnOnce(&Settings) -> T) -> T {
    if let Some(s) = CURRENT.read().unwrap_or_else(|e| e.into_inner()).as_ref() {
        return f(s);
    }
    let mut w = CURRENT.write().unwrap_or_else(|e| e.into_inner());
    let s = w.get_or_insert_with(|| if cfg!(feature = "cli") { Settings::from_env() } else { Settings::default() });
    f(s)
}

/// Sets the settings for the process.
pub fn set(s: Settings) {
    *CURRENT.write().unwrap_or_else(|e| e.into_inner()) = Some(s);
}
