//! Laddning av delade bibliotek (.dll/.so/.dylib) med operativsystemets
//! laddare: LoadLibraryW/GetProcAddress (Windows) och dlopen/dlsym (Unix),
//! direkt via FFI utan externa paket.
//!
//! Biblioteket laddas ur aldrig (funktionspekarna används under hela
//! processens livstid).

use crate::model::{fail, Result};
use std::ffi::c_void;
use std::path::Path;

/// Ett laddat bibliotek (handtaget från operativsystemet).
pub struct DynLib {
    handle: *mut c_void,
    pub path: String,
}

// Handtaget är bara en adress som operativsystemet äger; symboluppslag är trådsäkra.
unsafe impl Send for DynLib {}
unsafe impl Sync for DynLib {}

#[cfg(all(windows, feature = "dynlib"))]
mod os {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        fn GetLastError() -> u32;
    }

    pub fn open(path: &std::path::Path) -> Result<*mut c_void, String> {
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        let h = unsafe { LoadLibraryW(wide.as_ptr()) };
        if h.is_null() {
            return Err(format!("LoadLibraryW misslyckades (felkod {})", unsafe { GetLastError() }));
        }
        Ok(h)
    }

    pub fn symbol(h: *mut c_void, name: &str) -> *mut c_void {
        let c: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        unsafe { GetProcAddress(h, c.as_ptr()) }
    }
}

#[cfg(all(unix, feature = "dynlib"))]
mod os {
    use std::ffi::{c_char, c_int, c_void, CStr};

    const RTLD_NOW: c_int = 2;

    #[cfg_attr(target_os = "linux", link(name = "dl"))]
    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlerror() -> *const c_char;
    }

    pub fn open(path: &std::path::Path) -> Result<*mut c_void, String> {
        use std::os::unix::ffi::OsStrExt;
        let c: Vec<u8> = path.as_os_str().as_bytes().iter().copied().chain(std::iter::once(0)).collect();
        let h = unsafe { dlopen(c.as_ptr() as *const c_char, RTLD_NOW) };
        if h.is_null() {
            let e = unsafe { dlerror() };
            let msg = if e.is_null() { "okänt fel".to_string() } else { unsafe { CStr::from_ptr(e) }.to_string_lossy().to_string() };
            return Err(format!("dlopen misslyckades: {msg}"));
        }
        Ok(h)
    }

    pub fn symbol(h: *mut c_void, name: &str) -> *mut c_void {
        let c: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();
        unsafe { dlsym(h, c.as_ptr() as *const c_char) }
    }
}

/// Utan feature `dynlib` (t.ex. iOS, där App Store inte tillåter kod som
/// laddas utanför appen) kan inga bibliotek laddas.
#[cfg(not(feature = "dynlib"))]
mod os {
    use std::ffi::c_void;

    pub fn open(_path: &std::path::Path) -> Result<*mut c_void, String> {
        Err("laddning av delade bibliotek är avstängd i den här byggkonfigurationen (feature dynlib)".into())
    }

    pub fn symbol(_h: *mut c_void, _name: &str) -> *mut c_void {
        std::ptr::null_mut()
    }
}

impl DynLib {
    /// Laddar biblioteket i `path`.
    pub fn open(path: &Path) -> Result<DynLib> {
        if !path.exists() {
            return fail(format!("biblioteket {} finns inte", path.display()));
        }
        match os::open(path) {
            Ok(handle) => Ok(DynLib { handle, path: path.display().to_string() }),
            Err(e) => fail(format!("kunde inte ladda {}: {e}", path.display())),
        }
    }

    /// Adressen till den första av symbolerna `names` som finns (t.ex.
    /// `umat_` från gfortran, `UMAT` från Intel Fortran på Windows).
    pub fn symbol(&self, names: &[&str]) -> Option<(*mut c_void, String)> {
        names.iter().find_map(|n| {
            let p = os::symbol(self.handle, n);
            (!p.is_null()).then(|| (p, n.to_string()))
        })
    }
}
