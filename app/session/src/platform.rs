//! Fysiskt minne och minnesbudget per plattform (se PLAN.md, "Minnestak och lösarval").
//!
//! | Plattform | Budget |
//! |---|---|
//! | iOS | 50 % av `os_proc_available_memory()`, läst före varje körning; gäller alltid |
//! | macOS, Windows, Linux | 80 % av det fysiska minnet; användaren kan köra ändå |

/// Fysiskt minne i byte (0 om okänt).
pub fn total() -> f64 {
    imp::total()
}

/// Budgeten är ett hårt tak (iOS stänger appen vid för högt minne).
pub const HARD_BUDGET: bool = cfg!(target_os = "ios");

/// Plattformens namn ("windows", "macos", "linux", "ios").
pub fn name() -> &'static str {
    std::env::consts::OS
}

/// Budget för en körning (byte).
pub fn budget() -> f64 {
    #[cfg(target_os = "ios")]
    {
        // SAFETY: os_proc_available_memory är trådsäker och saknar argument.
        let avail = unsafe { ios::os_proc_available_memory() } as f64;
        if avail > 0.0 {
            return 0.5 * avail;
        }
    }
    let t = total();
    if t > 0.0 { 0.8 * t } else { f64::INFINITY }
}

#[cfg(target_os = "ios")]
mod ios {
    extern "C" {
        /// libSystem (iOS 13+): minne som processen får använda innan systemet stänger den.
        pub fn os_proc_available_memory() -> usize;
    }
}

#[cfg(windows)]
mod imp {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buf: *mut MemoryStatusEx) -> i32;
    }

    pub fn total() -> f64 {
        let mut m = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        // SAFETY: bufferten har rätt storlek och `length` är satt enligt API:t.
        if unsafe { GlobalMemoryStatusEx(&mut m) } != 0 { m.total_phys as f64 } else { 0.0 }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    pub fn total() -> f64 {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| s.lines().find(|l| l.starts_with("MemTotal:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<f64>().ok()))
            .map(|kb| kb * 1024.0)
            .unwrap_or(0.0)
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    extern "C" {
        fn sysctlbyname(name: *const i8, oldp: *mut std::ffi::c_void, oldlenp: *mut usize, newp: *mut std::ffi::c_void, newlen: usize) -> i32;
    }

    pub fn total() -> f64 {
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        // SAFETY: namnet är nollterminerat och bufferten är en u64.
        let r = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), &mut v as *mut u64 as *mut _, &mut len, std::ptr::null_mut(), 0) };
        if r == 0 { v as f64 } else { 0.0 }
    }
}

#[cfg(not(any(windows, target_os = "linux", target_os = "macos", target_os = "ios")))]
mod imp {
    pub fn total() -> f64 {
        0.0
    }
}
