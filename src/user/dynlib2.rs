//! Användarrutinerna i `routines.rs` från ett delat bibliotek (kompilerad
//! Fortran eller C) med standardargumentlistorna i Fortrans konvention: alla
//! argument som pekare, heltal som `int32`, CHARACTER*80 med dold längd sist.
//!
//! Symbolerna `dload_`, `disp_`, `uhard_`, `sigini_`, `ufield_`, `uexpan_`,
//! `hetval_`, `film_`, `creep_` och `umatht_` (även utan understreck och med
//! versaler) slås upp i biblioteket i `RUSTYFEM_USER` eller laddas med
//! `load_routines`, som registrerar dem för alla namn (tomt prefix). USDFLD
//! (kräver GETVRM) och UAMP finns bara i Rust. Anropen görs ett i taget.

use super::dynlib::DynLib;
use super::routines::*;
use crate::model::Result;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

type P = *const f64;
type M = *mut f64;
type I = *const i32;

type DloadFn = unsafe extern "C" fn(M, I, I, P, I, I, I, I, P, I, *const u8, usize);
type DispFn = unsafe extern "C" fn(M, I, I, P, I, I, I, P);
type UhardFn = unsafe extern "C" fn(M, M, P, P, P, P, P, P, I, I, I, I, I, I, *const u8, I, M, I, P, P, I, P, usize);
type SiginiFn = unsafe extern "C" fn(M, P, I, I, I, I, I, I, I, *const u8, usize);
type UfieldFn = unsafe extern "C" fn(M, I, I, I, I, P, I, P, P, P, I);
type UexpanFn = unsafe extern "C" fn(M, M, P, P, P, P, P, M, *const u8, I, I, usize);
type HetvalFn = unsafe extern "C" fn(*const u8, P, P, P, M, M, P, P, usize);
type FilmFn = unsafe extern "C" fn(M, M, P, I, I, P, I, I, P, I, P, I, *const u8, I, P, usize);
type CreepFn = unsafe extern "C" fn(M, M, M, P, P, P, P, P, P, P, P, P, P, P, *const u8, I, I, P, I, I, I, I, I, I, I, usize);
type UmathtFn = unsafe extern "C" fn(M, M, M, M, M, M, M, P, P, P, P, P, P, P, *const u8, I, I, P, I, P, M, I, I, I, I, I, I, usize);

/// En rutin i ett bibliotek: funktionspekaren och ett lås (Fortran kan ha SAVE-variabler).
pub struct Dll<F> {
    f: F,
    pub lib: Arc<DynLib>,
    pub symbol: String,
    lock: Mutex<()>,
}

fn name80(s: &str) -> [u8; 80] {
    let mut n = [b' '; 80];
    for (i, b) in s.bytes().take(80).enumerate() {
        n[i] = b;
    }
    n
}

fn ptr(v: &[f64], d: &[f64; 1]) -> P {
    if v.is_empty() {
        d.as_ptr()
    } else {
        v.as_ptr()
    }
}

fn mptr(v: &mut [f64], d: &mut [f64; 1]) -> M {
    if v.is_empty() {
        d.as_mut_ptr()
    } else {
        v.as_mut_ptr()
    }
}

fn i(v: impl TryInto<i32>) -> i32 {
    v.try_into().unwrap_or(0)
}

fn finite(sym: &str, v: &[f64]) -> Result<()> {
    if v.iter().any(|x| !x.is_finite()) {
        return crate::model::fail(format!("{sym} gav värden som inte är ändliga tal"));
    }
    Ok(())
}

impl Dload for Dll<DloadFn> {
    fn dload(&self, a: &mut DloadArgs) -> Result<()> {
        let s = name80(a.sname);
        let (ks, ki, ne, np, la, kp, jl) = (a.kstep, a.kinc, i(a.noel), i(a.npt), i(a.layer), i(a.kspt), a.jltyp);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.f)(&mut a.f, &ks, &ki, a.time.as_ptr(), &ne, &np, &la, &kp, a.coords.as_ptr(), &jl, s.as_ptr(), 80) };
        finite(&self.symbol, &[a.f])
    }
}

impl Disp for Dll<DispFn> {
    fn disp(&self, a: &mut DispArgs) -> Result<()> {
        let (ks, ki, no, ne, jd) = (a.kstep, a.kinc, i(a.node), i(a.noel), a.jdof);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.f)(a.u.as_mut_ptr(), &ks, &ki, a.time.as_ptr(), &no, &ne, &jd, a.coords.as_ptr()) };
        finite(&self.symbol, &a.u)
    }
}

impl Uhard for Dll<UhardFn> {
    fn uhard(&self, a: &mut UhardArgs) -> Result<()> {
        let c = name80(a.cmname);
        let (d0, mut d1) = ([0.0], [0.0]);
        let (ne, np, la, kp, ks, ki) = (i(a.noel), i(a.npt), i(a.layer), i(a.kspt), a.kstep, a.kinc);
        let (ns, nf, npr) = (i(a.statev.len()), i(a.predef.len()), i(a.props.len()));
        let st = mptr(a.statev, &mut d1);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(
                &mut a.syield, a.hard.as_mut_ptr(), &a.eqplas, &a.eqplasrt, a.time.as_ptr(), &a.dtime, &a.temp, &a.dtemp, &ne, &np, &la, &kp, &ks, &ki,
                c.as_ptr(), &ns, st, &nf, ptr(a.predef, &d0), ptr(a.dpred, &d0), &npr, ptr(a.props, &d0), 80,
            )
        };
        finite(&self.symbol, &[a.syield, a.hard[0]])
    }
}

impl Sigini for Dll<SiginiFn> {
    fn sigini(&self, a: &mut SiginiArgs) -> Result<()> {
        let mut names = [b' '; 160];
        names[..80].copy_from_slice(&name80(a.names[0]));
        names[80..].copy_from_slice(&name80(a.names[1]));
        let (nt, nc, ne, np, la, kp, lr) = (i(a.ntens), i(a.ncrds), i(a.noel), i(a.npt), i(a.layer), i(a.kspt), a.lrebar);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.f)(a.sigma.as_mut_ptr(), a.coords.as_ptr(), &nt, &nc, &ne, &np, &la, &kp, &lr, names.as_ptr(), 80) };
        finite(&self.symbol, a.sigma)
    }
}

impl Ufield for Dll<UfieldFn> {
    fn ufield(&self, a: &mut UfieldArgs) -> Result<()> {
        let (kf, nsec, ks, ki, no, nf) = (i(a.kfield), 1i32, a.kstep, a.kinc, i(a.node), i(a.nfield));
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.f)(a.field.as_mut_ptr(), &kf, &nsec, &ks, &ki, a.time.as_ptr(), &no, a.coords.as_ptr(), &a.temp, &a.dtemp, &nf) };
        finite(&self.symbol, a.field)
    }
}

impl Uexpan for Dll<UexpanFn> {
    fn uexpan(&self, a: &mut UexpanArgs) -> Result<()> {
        let c = name80(a.cmname);
        let (d0, mut d1) = ([0.0], [0.0]);
        let (ns, ne) = (i(a.statev.len()), i(a.noel));
        let st = mptr(a.statev, &mut d1);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(a.expan.as_mut_ptr(), a.dexpandt.as_mut_ptr(), a.temp.as_ptr(), a.time.as_ptr(), &a.dtime, ptr(a.predef, &d0), ptr(a.dpred, &d0), st, c.as_ptr(), &ns, &ne, 80)
        };
        finite(&self.symbol, a.expan)
    }
}

impl Hetval for Dll<HetvalFn> {
    fn hetval(&self, a: &mut HetvalArgs) -> Result<()> {
        let c = name80(a.cmname);
        let (d0, mut d1) = ([0.0], [0.0]);
        let st = mptr(a.statev, &mut d1);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { (self.f)(c.as_ptr(), a.temp.as_ptr(), a.time.as_ptr(), &a.dtime, st, a.flux.as_mut_ptr(), ptr(a.predef, &d0), ptr(a.dpred, &d0), 80) };
        finite(&self.symbol, &a.flux)
    }
}

impl Film for Dll<FilmFn> {
    fn film(&self, a: &mut FilmArgs) -> Result<()> {
        let s = name80(a.sname);
        let d0 = [0.0];
        let (ks, ki, ne, np, jl, nf, no) = (a.kstep, a.kinc, i(a.noel), i(a.npt), a.jltyp, i(a.field.len()), i(a.node));
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(a.h.as_mut_ptr(), &mut a.sink, &a.temp, &ks, &ki, a.time.as_ptr(), &ne, &np, a.coords.as_ptr(), &jl, ptr(a.field, &d0), &nf, s.as_ptr(), &no, &a.area, 80)
        };
        finite(&self.symbol, &[a.h[0], a.h[1], a.sink])
    }
}

impl Creep for Dll<CreepFn> {
    fn creep(&self, a: &mut CreepArgs) -> Result<()> {
        let c = name80(a.cmname);
        let (d0, mut d1) = ([0.0], [0.0]);
        let (lx, lend, ns, ne, np, la, kp, ks, ki) = (a.leximp, a.lend, i(a.statev.len()), i(a.noel), i(a.npt), i(a.layer), i(a.kspt), a.kstep, a.kinc);
        let st = mptr(a.statev, &mut d1);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(
                a.decra.as_mut_ptr(), a.deswa.as_mut_ptr(), st, &a.serd, a.ec.as_ptr(), a.esw.as_ptr(), &a.p, &a.qtild, &a.temp, &a.dtemp, ptr(a.predef, &d0), ptr(a.dpred, &d0),
                a.time.as_ptr(), &a.dtime, c.as_ptr(), &lx, &lend, a.coords.as_ptr(), &ns, &ne, &np, &la, &kp, &ks, &ki, 80,
            )
        };
        finite(&self.symbol, &a.decra)
    }
}

impl Umatht for Dll<UmathtFn> {
    fn umatht(&self, a: &mut UmathtArgs) -> Result<()> {
        let c = name80(a.cmname);
        let (d0, mut d1) = ([0.0], [0.0]);
        let (ng, ns, npr, ne, np, la, kp, ks, ki) = (i(a.ntgrd), i(a.statev.len()), i(a.props.len()), i(a.noel), i(a.npt), i(a.layer), i(a.kspt), a.kstep, a.kinc);
        let st = mptr(a.statev, &mut d1);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(
                &mut a.u, &mut a.dudt, a.dudg.as_mut_ptr(), a.flux.as_mut_ptr(), a.dfdt.as_mut_ptr(), a.dfdg.as_mut_ptr(), st, &a.temp, &a.dtemp, a.dtemdx.as_ptr(), a.time.as_ptr(),
                &a.dtime, ptr(a.predef, &d0), ptr(a.dpred, &d0), c.as_ptr(), &ng, &ns, ptr(a.props, &d0), &npr, a.coords.as_ptr(), &mut a.pnewdt, &ne, &np, &la, &kp, &ks, &ki, 80,
            )
        };
        finite(&self.symbol, a.flux)
    }
}

/// Rutinerna som finns i ett bibliotek.
#[derive(Default)]
pub struct Routines {
    pub dload: Option<Arc<dyn Dload>>,
    pub disp: Option<Arc<dyn Disp>>,
    pub uhard: Option<Arc<dyn Uhard>>,
    pub sigini: Option<Arc<dyn Sigini>>,
    pub ufield: Option<Arc<dyn Ufield>>,
    pub uexpan: Option<Arc<dyn Uexpan>>,
    pub hetval: Option<Arc<dyn Hetval>>,
    pub film: Option<Arc<dyn Film>>,
    pub creep: Option<Arc<dyn Creep>>,
    pub umatht: Option<Arc<dyn Umatht>>,
}

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty, $tr:ty) => {
        $lib.symbol(&[concat!($name, "_"), $name, $name.to_uppercase().as_str(), format!("{}_", $name.to_uppercase()).as_str()]).map(|(p, s)| {
            // SAFETY: symbolen antas ha rutinens standardargumentlista (dokumenterat krav)
            let f: $ty = unsafe { std::mem::transmute::<*mut std::ffi::c_void, $ty>(p) };
            Arc::new(Dll { f, lib: $lib.clone(), symbol: s, lock: Mutex::new(()) }) as Arc<$tr>
        })
    };
}

/// Slår upp rutinerna i ett laddat bibliotek.
pub fn routines_in(lib: &Arc<DynLib>) -> Routines {
    Routines {
        dload: sym!(lib, "dload", DloadFn, dyn Dload),
        disp: sym!(lib, "disp", DispFn, dyn Disp),
        uhard: sym!(lib, "uhard", UhardFn, dyn Uhard),
        sigini: sym!(lib, "sigini", SiginiFn, dyn Sigini),
        ufield: sym!(lib, "ufield", UfieldFn, dyn Ufield),
        uexpan: sym!(lib, "uexpan", UexpanFn, dyn Uexpan),
        hetval: sym!(lib, "hetval", HetvalFn, dyn Hetval),
        film: sym!(lib, "film", FilmFn, dyn Film),
        creep: sym!(lib, "creep", CreepFn, dyn Creep),
        umatht: sym!(lib, "umatht", UmathtFn, dyn Umatht),
    }
}

/// Laddar ett bibliotek och registrerar alla rutiner som finns i det för
/// alla namn (tomt prefix). Returnerar de registrerade rutinernas namn.
pub fn load_routines(path: &Path) -> Result<Vec<&'static str>> {
    let lib = Arc::new(DynLib::open(path)?);
    let r = routines_in(&lib);
    let mut v = Vec::new();
    macro_rules! reg {
        ($f:ident, $reg:ident, $n:literal) => {
            if let Some(x) = r.$f {
                $reg("", x);
                v.push($n);
            }
        };
    }
    reg!(dload, register_dload, "DLOAD");
    reg!(disp, register_disp, "DISP");
    reg!(uhard, register_uhard, "UHARD");
    reg!(sigini, register_sigini, "SIGINI");
    reg!(ufield, register_ufield, "UFIELD");
    reg!(uexpan, register_uexpan, "UEXPAN");
    reg!(hetval, register_hetval, "HETVAL");
    reg!(film, register_film, "FILM");
    reg!(creep, register_creep, "CREEP");
    reg!(umatht, register_umatht, "UMATHT");
    Ok(v)
}

fn env() -> Result<Option<&'static Routines>> {
    static R: OnceLock<std::result::Result<Option<Routines>, String>> = OnceLock::new();
    let r = R.get_or_init(|| match std::env::var("RUSTYFEM_USER").ok().filter(|_| cfg!(feature = "cli")) {
        Some(p) if !p.trim().is_empty() => DynLib::open(Path::new(p.trim())).map(|l| Some(routines_in(&Arc::new(l)))).map_err(|e| format!("RUSTYFEM_USER: {e}")),
        _ => Ok(None),
    });
    match r {
        Ok(l) => Ok(l.as_ref()),
        Err(e) => crate::model::fail(e.clone()),
    }
}

macro_rules! env_find {
    ($fn_:ident, $f:ident, $tr:ty) => {
        pub fn $fn_() -> Result<Option<Arc<$tr>>> {
            Ok(env()?.and_then(|r| r.$f.clone()))
        }
    };
}
env_find!(find_dload, dload, dyn Dload);
env_find!(find_disp, disp, dyn Disp);
env_find!(find_uhard, uhard, dyn Uhard);
env_find!(find_sigini, sigini, dyn Sigini);
env_find!(find_ufield, ufield, dyn Ufield);
env_find!(find_uexpan, uexpan, dyn Uexpan);
env_find!(find_hetval, hetval, dyn Hetval);
env_find!(find_film, film, dyn Film);
env_find!(find_creep, creep, dyn Creep);
env_find!(find_umatht, umatht, dyn Umatht);

pub fn find_uamp() -> Result<Option<Arc<dyn Uamp>>> {
    Ok(None)
}

pub fn find_usdfld() -> Result<Option<Arc<dyn Usdfld>>> {
    Ok(None)
}
