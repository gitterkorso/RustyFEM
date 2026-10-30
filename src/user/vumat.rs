//! VUMAT: user materials in explicit dynamics (*USER MATERIAL in
//! *DYNAMIC, EXPLICIT) with the established VUMAT interface.
//!
//! The interface is designed for a block of material points; RustyFEM calls it
//! with one point at a time (NBLOCK = 1), so the fields in `VumatArgs` are the values for
//! that point. The component order is VUMAT's: 3D (NDIR = 3, NSHR = 3)
//! 11, 22, 33, 12, 23, 31 and plane strain (3, 1) 11, 22, 33, 12. The strain
//! is given with tensor shear components (not engineering shear).
//! The deformation gradient and the stretch U (F = RU) have the 3D order
//! 11, 22, 33, 12, 23, 31, 21, 32, 13 (DEFGRADNEW/STRETCHNEW).
//!
//! Stress and strain increments are given in the global
//! basis with RustyFEM's Jaumann rotation (`kinematics/continuum.rs`), not in
//! a Green–Naghdi basis. Without large rotations this is the same thing.
//! RELSPININC is the rotation increment from ΔR (Hughes–Winget) between
//! the deformation gradients.
//!
//! A VUMAT has no tangent; `Vumat::elastic_moduli` gives the elastic
//! moduli needed by the stability limit and the bulk viscosity (instead
//! of estimating them from the stress and strain increments).
//!
//! ```rust,ignore
//! use rustyfem::user::vumat::{self, Vumat, VumatArgs};
//! struct MyVumat;
//! impl Vumat for MyVumat {
//!     fn vumat(&self, a: &mut VumatArgs) -> rustyfem::Result<()> { /* as in Fortran */ Ok(()) }
//!     fn elastic_moduli(&self, props: &[f64]) -> Option<(f64, f64)> { Some((props[0], props[1])) }
//! }
//! vumat::register("MY_", std::sync::Arc::new(MyVumat)); // material MY_*
//! ```

use crate::model::{fail, Result};
use std::sync::{Arc, OnceLock, RwLock};

/// The arguments to a VUMAT for one material point (the Fortran names in the comments).
pub struct VumatArgs<'a> {
    /// NDIR, NSHR, NSTATEV.
    pub ndir: usize,
    pub nshr: usize,
    pub nstatev: usize,
    /// LANNEAL (always 0).
    pub lanneal: i32,
    /// STEPTIME, TOTALTIME (at the start of the increment), DT.
    pub step_time: f64,
    pub total_time: f64,
    pub dt: f64,
    /// CMNAME.
    pub cmname: &'a str,
    /// COORDMP (0) and CHARLENGTH.
    pub coord_mp: [f64; 3],
    pub char_length: f64,
    /// PROPS.
    pub props: &'a [f64],
    /// DENSITY (current).
    pub density: f64,
    /// STRAININC (tensor shear), RELSPININC.
    pub strain_inc: &'a [f64],
    pub rel_spin_inc: [f64; 3],
    /// TEMPOLD, TEMPNEW.
    pub temp_old: f64,
    pub temp_new: f64,
    /// STRETCHOLD, STRETCHNEW, DEFGRADOLD, DEFGRADNEW (3D: 6, 6, 9, 9 values).
    pub stretch_old: [f64; 6],
    pub stretch_new: [f64; 6],
    pub defgrad_old: [f64; 9],
    pub defgrad_new: [f64; 9],
    /// STRESSOLD, STATEOLD, ENERINTERNOLD, ENERINELASOLD (per unit mass).
    pub stress_old: &'a [f64],
    pub state_old: &'a [f64],
    pub ener_intern_old: f64,
    pub ener_inelas_old: f64,
    /// Output: STRESSNEW, STATENEW, ENERINTERNNEW, ENERINELASNEW.
    pub stress_new: &'a mut [f64],
    pub state_new: &'a mut [f64],
    pub ener_intern_new: &'a mut f64,
    pub ener_inelas_new: &'a mut f64,
}

/// A user material model with VUMAT semantics.
pub trait Vumat: Send + Sync {
    /// Name for messages.
    fn name(&self) -> String {
        "VUMAT".into()
    }

    /// Sets STRESSNEW, STATENEW and the energies.
    fn vumat(&self, a: &mut VumatArgs) -> Result<()>;

    /// Elastic moduli (E, ν) for the stability limit from PROPS.
    fn elastic_moduli(&self, _props: &[f64]) -> Option<(f64, f64)> {
        None
    }
}

type Registry = RwLock<Vec<(String, Arc<dyn Vumat>)>>;

fn registry() -> &'static Registry {
    static R: OnceLock<Registry> = OnceLock::new();
    R.get_or_init(|| RwLock::new(samples()))
}

/// Registers a VUMAT for materials whose name starts with `prefix` (case
/// does not matter). A VUMAT takes precedence over a UMAT with the same material name.
pub fn register(prefix: &str, v: Arc<dyn Vumat>) {
    let p = prefix.trim().to_uppercase();
    let mut r = registry().write().unwrap_or_else(|e| e.into_inner());
    r.retain(|(q, _)| *q != p);
    r.push((p, v));
}

/// The VUMAT for material `cmname` (longest registered prefix).
pub fn find(cmname: &str) -> Option<Arc<dyn Vumat>> {
    let up = cmname.trim().to_uppercase();
    let r = registry().read().unwrap_or_else(|e| e.into_inner());
    r.iter().filter(|(p, _)| !p.is_empty() && up.starts_with(p.as_str())).max_by_key(|(p, _)| p.len()).map(|(_, v)| v.clone())
}

// ---------------------------------------------------------------- examples

/// Linear elastic VUMAT, PROPS = (E, ν). Material `EXAMPLE_VELASTIC*`.
pub struct ElasticVumat;

/// von Mises plasticity with isotropic hardening like *PLASTIC (piecewise
/// linear yield stress): PROPS = (E, ν, σ₁, εᵖ₁, σ₂, εᵖ₂, ...), STATEV(1) =
/// PEEQ. Radial return. Material `EXAMPLE_VMISES*`.
pub struct MisesVumat;

fn lame(e: f64, nu: f64) -> (f64, f64) {
    (e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))
}

/// Elastic predictor σ + λ tr(Δε) I + 2μ Δε (tensor shear).
fn trial(a: &VumatArgs, lam: f64, mu: f64) -> Vec<f64> {
    let nd = a.ndir;
    let tr: f64 = a.strain_inc[..nd].iter().sum();
    (0..nd + a.nshr).map(|i| a.stress_old[i] + 2.0 * mu * a.strain_inc[i] + if i < nd { lam * tr } else { 0.0 }).collect()
}

/// Specific internal energy: e += ½(σₙ + σₙ₊₁) : Δε / ρ.
fn energy(a: &VumatArgs, s_new: &[f64]) -> f64 {
    let n = a.ndir + a.nshr;
    let w: f64 = (0..n).map(|i| 0.5 * (a.stress_old[i] + s_new[i]) * a.strain_inc[i] * if i < a.ndir { 1.0 } else { 2.0 }).sum();
    a.ener_intern_old + w / a.density.max(1e-300)
}

impl Vumat for ElasticVumat {
    fn name(&self) -> String {
        "example: elastic VUMAT".into()
    }

    fn vumat(&self, a: &mut VumatArgs) -> Result<()> {
        if a.props.len() < 2 {
            return fail("EXAMPLE_VELASTIC requires PROPS = (E, ν)");
        }
        let (lam, mu) = lame(a.props[0], a.props[1]);
        let s = trial(a, lam, mu);
        *a.ener_intern_new = energy(a, &s);
        *a.ener_inelas_new = a.ener_inelas_old;
        a.stress_new.copy_from_slice(&s);
        a.state_new.copy_from_slice(a.state_old);
        Ok(())
    }

    fn elastic_moduli(&self, props: &[f64]) -> Option<(f64, f64)> {
        (props.len() >= 2).then(|| (props[0], props[1]))
    }
}

fn yield_stress(tab: &[(f64, f64)], p: f64) -> (f64, f64) {
    if tab.len() == 1 || p <= tab[0].1 {
        return (tab[0].0, 0.0);
    }
    for w in tab.windows(2) {
        let ((s0, p0), (s1, p1)) = (w[0], w[1]);
        if p <= p1 && p1 > p0 {
            let h = (s1 - s0) / (p1 - p0);
            return (s0 + h * (p - p0), h);
        }
    }
    (tab[tab.len() - 1].0, 0.0)
}

impl Vumat for MisesVumat {
    fn name(&self) -> String {
        "example: von Mises VUMAT".into()
    }

    fn vumat(&self, a: &mut VumatArgs) -> Result<()> {
        if a.props.len() < 4 || a.props.len() % 2 != 0 {
            return fail("EXAMPLE_VMISES requires PROPS = (E, ν, σ₁, εᵖ₁, σ₂, εᵖ₂, ...)");
        }
        if a.nstatev < 1 {
            return fail("EXAMPLE_VMISES requires *DEPVAR 1 (PEEQ)");
        }
        let (lam, mu) = lame(a.props[0], a.props[1]);
        let tab: Vec<(f64, f64)> = a.props[2..].chunks(2).map(|c| (c[0], c[1])).collect();
        let mut s = trial(a, lam, mu);
        let nd = a.ndir;
        let n = nd + a.nshr;
        let p_old = a.state_old[0];
        let m = s[..nd].iter().sum::<f64>() / 3.0;
        let dev: Vec<f64> = (0..n).map(|i| s[i] - if i < nd { m } else { 0.0 }).collect();
        // ndir = 3 in 3D and plane strain: s₃₃ is part of the direct components
        let j2: f64 = (0..n).map(|i| dev[i] * dev[i] * if i < nd { 1.0 } else { 2.0 }).sum();
        let q = (1.5 * j2).sqrt();
        let (sy0, _) = yield_stress(&tab, p_old);
        let mut dp = 0.0;
        if q > sy0 {
            for _ in 0..100 {
                let (sy, h) = yield_stress(&tab, p_old + dp);
                let f = q - 3.0 * mu * dp - sy;
                dp += f / (3.0 * mu + h);
                if f.abs() <= 1e-12 * sy.max(1.0) {
                    break;
                }
            }
            let k = 1.0 - 3.0 * mu * dp / q;
            for i in 0..n {
                s[i] = dev[i] * k + if i < nd { m } else { 0.0 };
            }
        }
        let e = energy(a, &s);
        let (sy, _) = yield_stress(&tab, p_old + dp);
        *a.ener_inelas_new = a.ener_inelas_old + sy * dp / a.density.max(1e-300);
        *a.ener_intern_new = e;
        a.stress_new.copy_from_slice(&s);
        a.state_new.copy_from_slice(a.state_old);
        a.state_new[0] = p_old + dp;
        Ok(())
    }

    fn elastic_moduli(&self, props: &[f64]) -> Option<(f64, f64)> {
        (props.len() >= 2).then(|| (props[0], props[1]))
    }
}

fn samples() -> Vec<(String, Arc<dyn Vumat>)> {
    vec![("EXAMPLE_VELASTIC".into(), Arc::new(ElasticVumat)), ("EXAMPLE_VMISES".into(), Arc::new(MisesVumat))]
}
