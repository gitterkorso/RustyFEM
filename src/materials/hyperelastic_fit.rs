//! Area materials4: *HYPERELASTIC, ARRUDA-BOYCE, TEST DATA INPUT – μ and λm
//! fitted to test data (least squares for the
//! relative error E_k = 1 − T_k^th/T_k^test, Marquardt–Levenberg from
//! λm(0) = max(7, 3 λmax)).
//!
//! With the nominal stresses of the incompressible material (λ = 1 + ε),
//!
//!   T = 2 (λ − λ^c) ∂U/∂Ī1,  ∂U/∂Ī1 = μ Σ_{i=1}^{5} i C_i λm^(2−2i) Ī1^(i−1),
//!
//! with c = −2 (uniaxial), −5 (biaxial) and −3 (planar). The stress is
//! linear in μ, so for each λm, μ*(λm) is given by a linear least-squares problem
//! and the scalar error sum S(λm) is minimized locally starting from λm(0)
//! (the same local minimum that the Marquardt–Levenberg iteration reaches).
//!
//! The volumetric part: *VOLUMETRIC TEST DATA (pressure p, volume ratio J) gives D from
//! p = −(J − 1/J)/D (relative error), POISSON=ν gives D = 3(1 − 2ν)/(μ0 (1 + ν)),
//! otherwise the material is incompressible (D = 0). OGDEN and VAN DER WAALS with
//! TEST DATA INPUT are not supported yet (starting values for αi are missing).

use super::hyperelastic_ogden::{Form, HyperOgden};
use super::{MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "hyperelasticity from test data (Arruda–Boyce, TEST DATA INPUT)",
    consumes: &["HYPERELASTIC", "UNIAXIAL TEST DATA", "BIAXIAL TEST DATA", "PLANAR TEST DATA", "VOLUMETRIC TEST DATA"],
    priority: 10,
    build: Some(build),
}];

const AB_C: [f64; 5] = [0.5, 1.0 / 20.0, 11.0 / 1050.0, 19.0 / 7000.0, 519.0 / 673750.0];

/// The model takes *HYPERELASTIC with TEST DATA INPUT for a form that
/// `hyperelastic_ogden.rs`/`hyperelastic_vdw.rs` would otherwise build.
pub fn claims(kw: &RawKeyword) -> bool {
    kw.has("TEST DATA INPUT") && (kw.has("OGDEN") || kw.has("ARRUDA-BOYCE") || kw.has("VAN DER WAALS"))
}

/// (ε, T) from a test data keyword (nominal stress first).
fn pairs(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut v = Vec::new();
    for d in kw.records() {
        let f = kw.floats(&d)?;
        if f.len() < 2 {
            return kw.err("test data: each line must have stress and strain");
        }
        v.push((f[1], f[0]));
    }
    Ok(v)
}

/// A deviatoric test point: Ī1, the gain 2(λ − λ^c) and T.
struct Point {
    i1: f64,
    g: f64,
    t: f64,
}

/// ∂U/∂Ī1 per μ.
fn du(lm: f64, i1: f64) -> f64 {
    (0..5).map(|k| (k + 1) as f64 * AB_C[k] * lm.powi(-2 * k as i32) * i1.powi(k as i32)).sum()
}

/// (S, μ*) at λm.
fn objective(pts: &[Point], lm: f64) -> (f64, f64) {
    // E_k = 1 − μ a_k with a_k = g du / T
    let a: Vec<f64> = pts.iter().map(|p| p.g * du(lm, p.i1) / p.t).collect();
    let mu = a.iter().sum::<f64>() / a.iter().map(|x| x * x).sum::<f64>();
    (a.iter().map(|x| (1.0 - mu * x).powi(2)).sum(), mu)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    if !claims(kw) || def.behavior("MULLINS EFFECT").is_some() {
        return Ok(None);
    }
    if !kw.has("ARRUDA-BOYCE") {
        return kw.err("TEST DATA INPUT is only supported for ARRUDA-BOYCE among the nonlinear forms (not OGDEN or VAN DER WAALS yet)");
    }
    for p in ["USER", "TYPE", "PROPERTIES", "MODULI"] {
        if kw.has(p) {
            return kw.err(format!("the parameter {p} is not supported yet with TEST DATA INPUT"));
        }
    }
    let mut pts = Vec::new();
    let mut lmax: f64 = 1.0;
    for (name, c) in [("UNIAXIAL TEST DATA", -2), ("BIAXIAL TEST DATA", -5), ("PLANAR TEST DATA", -3)] {
        let Some(t) = def.behavior(name) else { continue };
        if t.has("DEPENDENCIES") || t.has("SMOOTH") {
            return t.err("DEPENDENCIES and SMOOTH are not supported yet for test data");
        }
        for (e, s) in pairs(t)? {
            if s == 0.0 {
                continue;
            }
            let l = 1.0 + e;
            if !(l > 0.0) {
                return t.err("test data: the strain must be greater than −1");
            }
            lmax = lmax.max(l).max(1.0 / l);
            let i1 = match c {
                -2 => l * l + 2.0 / l,
                -5 => 2.0 * l * l + l.powi(-4),
                _ => l * l + 1.0 + l.powi(-2),
            };
            pts.push(Point { i1, g: 2.0 * (l - l.powi(c)), t: s });
        }
    }
    if pts.is_empty() {
        return kw.err("TEST DATA INPUT requires *UNIAXIAL, *BIAXIAL or *PLANAR TEST DATA");
    }
    // local minimum of S(λm) from λm(0), in ln λm
    let f = |x: f64| objective(&pts, 1.0 + x.exp()).0;
    let x0 = (7.0f64.max(3.0 * lmax) - 1.0).ln();
    let mut h = 0.05;
    let (mut a, mut b) = (x0, x0 + h);
    let (mut fa, mut fb) = (f(a), f(b));
    if fb > fa {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
        h = -h;
    }
    // step downhill until the function increases again (or λm becomes very large)
    let mut c = b + h;
    let mut fc = f(c);
    let mut n = 0;
    while fc < fb {
        h *= 1.6;
        a = b;
        b = c;
        fb = fc;
        c = b + h;
        fc = f(c);
        n += 1;
        if n > 200 || c.abs() > 30.0 {
            return kw.err("TEST DATA INPUT: the fit of λm does not converge (test data without locking?)");
        }
    }
    // golden-section search in [a, c] with the minimum near b
    let (mut lo, mut hi) = if a < c { (a, c) } else { (c, a) };
    let gr = 0.5 * (5f64.sqrt() - 1.0);
    let mut x1 = hi - gr * (hi - lo);
    let mut x2 = lo + gr * (hi - lo);
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..200 {
        if (hi - lo).abs() < 1e-13 {
            break;
        }
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - gr * (hi - lo);
            f1 = f(x1);
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + gr * (hi - lo);
            f2 = f(x2);
        }
    }
    let lm = 1.0 + (0.5 * (lo + hi)).exp();
    let (_, mu) = objective(&pts, lm);
    if !(mu > 0.0) {
        return kw.err(format!("TEST DATA INPUT: the fitted shear modulus {mu:e} is not positive"));
    }
    let mut model = HyperOgden { form: Form::ArrudaBoyce(mu, lm), d: vec![0.0] };
    let mu0 = super::hyperelastic_split::HyperModel::mu0(&model);
    // volumetric part
    if let Some(t) = def.behavior("VOLUMETRIC TEST DATA") {
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        for (j, p) in pairs(t)? {
            if p == 0.0 {
                continue;
            }
            // p = −x (J − 1/J), x = 1/D; E = 1 − x a, a = −(J − 1/J)/p
            let a = -(j - 1.0 / j) / p;
            s1 += a;
            s2 += a * a;
        }
        let x = if s2 > 0.0 { s1 / s2 } else { 0.0 };
        if !(x > 0.0) {
            return t.err("VOLUMETRIC TEST DATA: the fitted bulk modulus is not positive");
        }
        model.d[0] = 1.0 / x;
    } else if let Some(s) = kw.param("POISSON") {
        let nu: f64 = s.trim().parse().map_err(|_| FemError(format!("{}: POISSON={s} is not a number", kw.location())))?;
        if !(nu < 0.5 && nu > -1.0) {
            return kw.err("POISSON must lie in (−1, 0.5)");
        }
        model.d[0] = 3.0 * (1.0 - 2.0 * nu) / (mu0 * (1.0 + nu));
    }
    if std::env::var("RUSTYFEM_VERBOSE_FIT").is_ok() {
        eprintln!("Arruda–Boyce from test data: μ = {mu:.9e}, λm = {lm:.9e}, D = {:.9e}", model.d[0]);
    }
    Ok(Some(Arc::new(model)))
}
