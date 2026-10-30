//! UMAT: användarmaterial med det etablerade UMAT-gränssnittet.
//!
//! En UMAT skrivs i Rust genom att implementera `Umat` och registrera den med
//! `register(prefix, ...)`; materialet väljs med materialnamnet (CMNAME),
//! där längsta registrerade prefix vinner. En UMAT i ett delat
//! bibliotek (Fortran eller C, symbolen `umat_`/`UMAT`) laddas med
//! `super::load_library` eller miljövariabeln `RUSTYFEM_USER`.
//!
//! Argumenten har de etablerade namnen och betydelserna:
//!
//! | Argument | Innehåll |
//! |---|---|
//! | STRESS(NTENS) | spänning vid inkrementets början → slut (Cauchy, roterad med DROT vid NLGEOM) |
//! | STATEV(NSTATV) | tillståndsvariabler (*DEPVAR) vid början → slut |
//! | DDSDDE(NTENS,NTENS) | Jacobian ∂Δσ/∂Δε, kolumnvis som i Fortran (`ddsdde[i + j·NTENS]`) |
//! | SSE, SPD, SCD | specifik elastisk, plastisk och krypenergi (ackumuleras) |
//! | STRAN, DSTRAN | töjning vid början och töjningsinkrement (ingenjörsskjuvtöjning) |
//! | TIME(2), DTIME | stegtid och total tid vid början, inkrementets längd |
//! | TEMP, DTEMP | temperatur vid början och dess ändring |
//! | NDI, NSHR, NTENS | antal normal- och skjuvkomponenter: 3D 3/3/6, plan töjning och axisymmetri 3/1/4, plan spänning 2/1/3, stänger 1/0/1 |
//! | PROPS(NPROPS) | konstanterna från *USER MATERIAL, CONSTANTS= |
//! | DROT(3,3) | inkrementell rotation (Hughes–Winget från DFGRD0/DFGRD1) |
//! | PNEWDT | sätts < 1 för att begära mindre inkrement (inkrementet görs om) |
//! | DFGRD0, DFGRD1 | deformationsgradient vid början och slut (enhetsmatris utan NLGEOM) |
//!
//! Komponentordningen är 11, 22, 33, 12, 13, 23.
//!
//! COORDS (integrationspunktens läge, aktuellt vid NLGEOM), CELENT, NOEL,
//! NPT, KSTEP, KINC, TIME(1) = stegtid och PREDEF/DPRED (fältvariablerna)
//! kommer från `super::context` (satta av elementen och stegramverket);
//! LAYER = KSPT = 1. PNEWDT < 1 gör om inkrementet med längden PNEWDT · Δt.
//! RPL, DDSDDT, DRPLDE och DRPLDT skickas men används inte.

use super::dynlib::DynLib;
use crate::model::{fail, Result};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

/// Argumenten till en UMAT (Fortran-namnen, 0-baserade index i Rust).
pub struct UmatArgs<'a> {
    pub stress: &'a mut [f64],
    pub statev: &'a mut [f64],
    /// DDSDDE(NTENS, NTENS) kolumnvis: element (i, j) = `ddsdde[i + j * ntens]`.
    pub ddsdde: &'a mut [f64],
    pub sse: &'a mut f64,
    pub spd: &'a mut f64,
    pub scd: &'a mut f64,
    pub rpl: &'a mut f64,
    pub ddsddt: &'a mut [f64],
    pub drplde: &'a mut [f64],
    pub drpldt: &'a mut f64,
    pub stran: &'a [f64],
    pub dstran: &'a [f64],
    pub time: [f64; 2],
    pub dtime: f64,
    pub temp: f64,
    pub dtemp: f64,
    pub predef: &'a [f64],
    pub dpred: &'a [f64],
    pub cmname: &'a str,
    pub ndi: usize,
    pub nshr: usize,
    pub ntens: usize,
    pub nstatv: usize,
    pub props: &'a [f64],
    pub coords: [f64; 3],
    /// DROT(3,3) radvis (`drot[i][j]` = DROT(i+1, j+1)).
    pub drot: [[f64; 3]; 3],
    pub pnewdt: &'a mut f64,
    pub celent: f64,
    /// DFGRD0 och DFGRD1 radvis (`dfgrd1[i][j]` = DFGRD1(i+1, j+1)).
    pub dfgrd0: [[f64; 3]; 3],
    pub dfgrd1: [[f64; 3]; 3],
    pub noel: i64,
    pub npt: usize,
    pub layer: usize,
    pub kspt: usize,
    /// JSTEP(1..4): steg, procedurtyp, NLGEOM (1/0), linjär perturbation (1/0).
    pub jstep: [i32; 4],
    pub kinc: usize,
}

impl UmatArgs<'_> {
    /// DDSDDE(i+1, j+1).
    pub fn d(&self, i: usize, j: usize) -> f64 {
        self.ddsdde[i + j * self.ntens]
    }

    /// Sätter DDSDDE(i+1, j+1).
    pub fn set_d(&mut self, i: usize, j: usize, v: f64) {
        let n = self.ntens;
        self.ddsdde[i + j * n] = v;
    }
}

/// En användarmaterialmodell med UMAT:s semantik.
pub trait Umat: Send + Sync {
    /// Namn för meddelanden.
    fn name(&self) -> String {
        "UMAT".into()
    }

    /// Uppdaterar STRESS, STATEV och energierna och sätter DDSDDE.
    fn umat(&self, a: &mut UmatArgs) -> Result<()>;
}

type Registry = RwLock<Vec<(String, Arc<dyn Umat>)>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| RwLock::new(super::samples::umats()))
}

/// Registrerar en UMAT för material vars namn börjar med `prefix` (versaler
/// spelar ingen roll; tomt prefix = alla material med *USER MATERIAL).
/// En senare registrering med samma prefix ersätter den tidigare.
pub fn register(prefix: &str, umat: Arc<dyn Umat>) {
    let p = prefix.trim().to_uppercase();
    let mut r = registry().write().unwrap_or_else(|e| e.into_inner());
    r.retain(|(q, _)| *q != p);
    r.push((p, umat));
}

/// UMAT för materialet `cmname`: längsta registrerade prefix, annars
/// biblioteket i `RUSTYFEM_USER`.
pub fn find(cmname: &str) -> Result<Option<Arc<dyn Umat>>> {
    let up = cmname.trim().to_uppercase();
    {
        let r = registry().read().unwrap_or_else(|e| e.into_inner());
        if let Some((_, u)) = r.iter().filter(|(p, _)| up.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()) {
            return Ok(Some(u.clone()));
        }
    }
    match super::env_library()? {
        Some(lib) => Ok(lib.umat()),
        None => Ok(None),
    }
}

/// UMAT i Fortrans anropskonvention: alla argument som pekare, CMNAME
/// (CHARACTER*80) med dold längd sist (gfortran och Intel Fortran som standard).
pub type UmatFn = unsafe extern "C" fn(
    stress: *mut f64,
    statev: *mut f64,
    ddsdde: *mut f64,
    sse: *mut f64,
    spd: *mut f64,
    scd: *mut f64,
    rpl: *mut f64,
    ddsddt: *mut f64,
    drplde: *mut f64,
    drpldt: *mut f64,
    stran: *const f64,
    dstran: *const f64,
    time: *const f64,
    dtime: *const f64,
    temp: *const f64,
    dtemp: *const f64,
    predef: *const f64,
    dpred: *const f64,
    cmname: *const u8,
    ndi: *const i32,
    nshr: *const i32,
    ntens: *const i32,
    nstatv: *const i32,
    props: *const f64,
    nprops: *const i32,
    coords: *const f64,
    drot: *const f64,
    pnewdt: *mut f64,
    celent: *const f64,
    dfgrd0: *const f64,
    dfgrd1: *const f64,
    noel: *const i32,
    npt: *const i32,
    layer: *const i32,
    kspt: *const i32,
    jstep: *const i32,
    kinc: *const i32,
    cmname_len: usize,
);

/// UMAT i ett delat bibliotek.
pub struct DllUmat {
    pub f: UmatFn,
    pub lib: Arc<DynLib>,
    pub symbol: String,
    /// Anropen görs ett i taget (Fortrankod kan ha SAVE-variabler).
    lock: Mutex<()>,
}

impl DllUmat {
    pub fn new(f: UmatFn, lib: Arc<DynLib>, symbol: String) -> Self {
        DllUmat { f, lib, symbol, lock: Mutex::new(()) }
    }
}

/// 3×3 radvis → kolumnvis (Fortran).
fn colmajor(m: &[[f64; 3]; 3]) -> [f64; 9] {
    std::array::from_fn(|k| m[k % 3][k / 3])
}

impl Umat for DllUmat {
    fn name(&self) -> String {
        format!("{} i {}", self.symbol, self.lib.path)
    }

    fn umat(&self, a: &mut UmatArgs) -> Result<()> {
        let mut name = [b' '; 80];
        for (i, b) in a.cmname.bytes().take(80).enumerate() {
            name[i] = b;
        }
        let i32_of = |v: usize| v as i32;
        let (ndi, nshr, ntens, nstatv, nprops) = (i32_of(a.ndi), i32_of(a.nshr), i32_of(a.ntens), i32_of(a.nstatv), a.props.len() as i32);
        let (noel, npt, layer, kspt, kinc) = (a.noel as i32, i32_of(a.npt), i32_of(a.layer), i32_of(a.kspt), i32_of(a.kinc));
        let drot = colmajor(&a.drot);
        let f0 = colmajor(&a.dfgrd0);
        let f1 = colmajor(&a.dfgrd1);
        // tomma fält får ändå en giltig adress
        let mut statev_dummy = [0.0f64; 1];
        let statev: *mut f64 = if a.statev.is_empty() { statev_dummy.as_mut_ptr() } else { a.statev.as_mut_ptr() };
        let props_dummy = [0.0f64; 1];
        let props: *const f64 = if a.props.is_empty() { props_dummy.as_ptr() } else { a.props.as_ptr() };
        let pre = [0.0f64; 1];
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        unsafe {
            (self.f)(
                a.stress.as_mut_ptr(),
                statev,
                a.ddsdde.as_mut_ptr(),
                &mut *a.sse,
                &mut *a.spd,
                &mut *a.scd,
                &mut *a.rpl,
                a.ddsddt.as_mut_ptr(),
                a.drplde.as_mut_ptr(),
                &mut *a.drpldt,
                a.stran.as_ptr(),
                a.dstran.as_ptr(),
                a.time.as_ptr(),
                &a.dtime,
                &a.temp,
                &a.dtemp,
                if a.predef.is_empty() { pre.as_ptr() } else { a.predef.as_ptr() },
                if a.dpred.is_empty() { pre.as_ptr() } else { a.dpred.as_ptr() },
                name.as_ptr(),
                &ndi,
                &nshr,
                &ntens,
                &nstatv,
                props,
                &nprops,
                a.coords.as_ptr(),
                drot.as_ptr(),
                &mut *a.pnewdt,
                &a.celent,
                f0.as_ptr(),
                f1.as_ptr(),
                &noel,
                &npt,
                &layer,
                &kspt,
                a.jstep.as_ptr(),
                &kinc,
                80,
            );
        }
        if a.stress.iter().chain(a.ddsdde.iter()).any(|v| !v.is_finite()) {
            return fail(format!("UMAT {} gav spänning eller DDSDDE som inte är ändliga tal", self.symbol));
        }
        Ok(())
    }
}
