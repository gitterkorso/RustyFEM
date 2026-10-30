//! Physical memory and memory budget per platform (see PLAN.md, "Memory cap and solver choice").
//!
//! | Platform | Budget |
//! |---|---|
//! | iOS | 50 % of `os_proc_available_memory()`, read before each run; always applies |
//! | macOS, Windows, Linux | 80 % of physical memory; the user can run anyway |

/// Physical memory in bytes (0 if unknown).
pub fn total() -> f64 {
    imp::total()
}

/// The budget is a hard cap (iOS closes the app when memory use is too high).
pub const HARD_BUDGET: bool = cfg!(target_os = "ios");

/// The platform's name ("windows", "macos", "linux", "ios").
pub fn name() -> &'static str {
    std::env::consts::OS
}

/// Budget for a run (bytes).
pub fn budget() -> f64 {
    #[cfg(target_os = "ios")]
    {
        // SAFETY: os_proc_available_memory is thread-safe and takes no arguments.
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
        /// libSystem (iOS 13+): memory the process may use before the system terminates it.
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
        // SAFETY: the buffer has the right size and `length` is set as the API requires.
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
        // SAFETY: the name is null-terminated and the buffer is a u64.
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
