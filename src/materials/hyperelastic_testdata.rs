//! Hyperelasticitet ur provdata (*HYPERELASTIC, TEST DATA INPUT) för
//! polynomformerna NEO HOOKE, MOONEY-RIVLIN, YEOH, REDUCED POLYNOMIAL och
//! POLYNOMIAL, med *UNIAXIAL TEST DATA, *BIAXIAL TEST DATA, *PLANAR TEST DATA
//! (nominell spänning, nominell töjning) och *VOLUMETRIC TEST DATA (tryck,
//! volymkvot J).
//!
//! Koefficienterna anpassas med minsta kvadratmetoden för det
//! relativa felet E = Σ (1 − T_i^th / T_i^test)² med det inkompressibla
//! materialets nominella spänningar (λ = 1 + ε):
//!
//! - enaxligt: T = 2(1 − λ⁻³)(λ U1 + U2), I1 = λ² + 2/λ, I2 = 2λ + 1/λ²
//! - tvåaxligt (lika): T = 2(λ − λ⁻⁵)(U1 + λ² U2), I1 = 2λ² + λ⁻⁴, I2 = λ⁴ + 2λ⁻²
//! - plant (ren skjuvning): T = 2(λ − λ⁻³)(U1 + U2), I1 = I2 = λ² + 1 + λ⁻²
//!
//! Spänningarna är linjära i C_ij, så anpassningen är ett linjärt
//! minsta kvadratproblem. Volymdelen: med *VOLUMETRIC TEST DATA anpassas 1/D_k
//! till p = −Σ 2k/D_k (J − 1)^(2k−1) på samma sätt; annars ger `POISSON=ν`
//! D1 = 3(1 − 2ν)/(μ0(1 + ν)), och utan någon av dem är materialet helt
//! inkompressibelt (D = 0, hybridelement). Punkter med spänningen 0 hoppas över.

use super::hyperelastic::Hyperelastic;
use super::{MaterialFactory, MaterialModel};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "hyperelasticitet ur provdata (TEST DATA INPUT)",
    consumes: &["HYPERELASTIC", "UNIAXIAL TEST DATA", "BIAXIAL TEST DATA", "PLANAR TEST DATA", "VOLUMETRIC TEST DATA"],
    priority: 10,
    build: Some(build),
}];

/// Nyckelorden för provdata (förbrukas också av *MULLINS EFFECT).
pub const TEST_KEYWORDS: [&str; 4] = ["UNIAXIAL TEST DATA", "BIAXIAL TEST DATA", "PLANAR TEST DATA", "VOLUMETRIC TEST DATA"];

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    if !kw.has("TEST DATA INPUT") || def.behavior("MULLINS EFFECT").is_some() {
        return Ok(None);
    }
    let Some(m) = fit(def)? else { return Ok(None) };
    Ok(Some(Arc::new(m)))
}

/// (ε, T) ur ett provdatanyckelord (nominell spänning först).
fn pairs(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut v = Vec::new();
    for d in kw.records() {
        let f = kw.floats(&d)?;
        if f.len() < 2 {
            return kw.err("provdata: varje rad ska ha spänning och töjning");
        }
        v.push((f[1], f[0]));
    }
    Ok(v)
}

/// Löser minsta kvadratproblemet min |A x − b|² (normalekvationer, Gauss med pivotering).
fn lsq(rows: &[(Vec<f64>, f64)], n: usize) -> Result<Vec<f64>> {
    let mut m = vec![0.0; n * (n + 1)];
    for (a, b) in rows {
        for i in 0..n {
            for j in 0..n {
                m[i * (n + 1) + j] += a[i] * a[j];
            }
            m[i * (n + 1) + n] += a[i] * b;
        }
    }
    for c in 0..n {
        let p = (c..n).max_by(|&x, &y| m[x * (n + 1) + c].abs().total_cmp(&m[y * (n + 1) + c].abs())).unwrap();
        if !(m[p * (n + 1) + c].abs() > 0.0) {
            return fail("TEST DATA INPUT: för få eller olämpliga provdata för att bestämma koefficienterna");
        }
        for k in 0..=n {
            m.swap(c * (n + 1) + k, p * (n + 1) + k);
        }
        for r in 0..n {
            if r != c {
                let f = m[r * (n + 1) + c] / m[c * (n + 1) + c];
                for k in c..=n {
                    m[r * (n + 1) + k] -= f * m[c * (n + 1) + k];
                }
            }
        }
    }
    Ok((0..n).map(|i| m[i * (n + 1) + n] / m[i * (n + 1) + i]).collect())
}

/// Anpassar polynommodellen till provdatat; `None` för former som inte är polynom.
pub fn fit(def: &MaterialDef) -> Result<Option<Hyperelastic>> {
    let Some(kw) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    let n = super::hyperfoam::n_param(kw)?;
    let (kind, cij): (&str, Vec<(usize, usize)>) = if kw.has("NEO HOOKE") {
        ("NEO HOOKE", vec![(1, 0)])
    } else if kw.has("MOONEY-RIVLIN") {
        ("MOONEY-RIVLIN", vec![(1, 0), (0, 1)])
    } else if kw.has("YEOH") {
        ("YEOH", vec![(1, 0), (2, 0), (3, 0)])
    } else if kw.has("REDUCED POLYNOMIAL") {
        ("REDUCED POLYNOMIAL", (1..=n).map(|i| (i, 0)).collect())
    } else if kw.has("POLYNOMIAL") {
        ("POLYNOMIAL", (1..=n).flat_map(|k| (0..=k).map(move |j| (k - j, j))).collect())
    } else {
        return Ok(None);
    };
    let nd = match kind {
        "NEO HOOKE" | "MOONEY-RIVLIN" => 1,
        "YEOH" => 3,
        _ => n,
    };
    for p in ["USER", "TYPE", "PROPERTIES", "BETA"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    if kw.has("MODULI") && kw.param_upper("MODULI").as_deref() != Some("LONG TERM") {
        return kw.err("MODULI stöds bara som LONG TERM");
    }
    // deviatoriska koefficienter
    let mut rows: Vec<(Vec<f64>, f64)> = Vec::new();
    for (name, which) in [("UNIAXIAL TEST DATA", 0), ("BIAXIAL TEST DATA", 1), ("PLANAR TEST DATA", 2)] {
        let Some(t) = def.behavior(name) else { continue };
        for (e, s) in pairs(t)? {
            if s == 0.0 {
                continue;
            }
            let l = 1.0 + e;
            if !(l > 0.0) {
                return t.err("provdata: töjningen ska vara större än −1");
            }
            let (i1, i2, a1, a2) = match which {
                // T = a1 U1 + a2 U2
                0 => (l * l + 2.0 / l, 2.0 * l + 1.0 / (l * l), 2.0 * (1.0 - l.powi(-3)) * l, 2.0 * (1.0 - l.powi(-3))),
                1 => (2.0 * l * l + l.powi(-4), l.powi(4) + 2.0 * l.powi(-2), 2.0 * (l - l.powi(-5)), 2.0 * (l - l.powi(-5)) * l * l),
                _ => (l * l + 1.0 + l.powi(-2), l * l + 1.0 + l.powi(-2), 2.0 * (l - l.powi(-3)), 2.0 * (l - l.powi(-3))),
            };
            let (x, y) = (i1 - 3.0, i2 - 3.0);
            let a: Vec<f64> = cij
                .iter()
                .map(|&(p, q)| {
                    let u1 = if p > 0 { p as f64 * x.powi(p as i32 - 1) * y.powi(q as i32) } else { 0.0 };
                    let u2 = if q > 0 { q as f64 * x.powi(p as i32) * y.powi(q as i32 - 1) } else { 0.0 };
                    (a1 * u1 + a2 * u2) / s
                })
                .collect();
            rows.push((a, 1.0));
        }
    }
    if rows.is_empty() {
        return kw.err("TEST DATA INPUT kräver *UNIAXIAL, *BIAXIAL eller *PLANAR TEST DATA");
    }
    let cv = lsq(&rows, cij.len())?;
    let c: Vec<(usize, usize, f64)> = cij.iter().zip(&cv).map(|(&(p, q), &v)| (p, q, v)).collect();
    let mu0: f64 = c.iter().filter(|&&(p, q, _)| p + q == 1).map(|&(_, _, v)| 2.0 * v).sum();
    if !(mu0 > 0.0) {
        return kw.err(format!("TEST DATA INPUT: den anpassade initiala skjuvmodulen {mu0:e} är inte positiv"));
    }
    // volymdel
    let mut d = vec![0.0; nd];
    if let Some(t) = def.behavior("VOLUMETRIC TEST DATA") {
        let mut vr = Vec::new();
        for (j, p) in pairs(t)? {
            // pairs ger (andra, första) = (J, p)
            if p == 0.0 {
                continue;
            }
            let a: Vec<f64> = (1..=nd).map(|k| -((2 * k) as f64) * (j - 1.0).powi(2 * k as i32 - 1) / p).collect();
            vr.push((a, 1.0));
        }
        let x = lsq(&vr, nd)?;
        if !(x[0] > 0.0) {
            return t.err("VOLUMETRIC TEST DATA: den anpassade kompressionsmodulen är inte positiv");
        }
        d = x.iter().map(|&v| if v != 0.0 { 1.0 / v } else { 0.0 }).collect();
    } else if let Some(s) = kw.param("POISSON") {
        let nu: f64 = s.trim().parse().map_err(|_| crate::model::FemError(format!("{}: POISSON={s} är inget tal", kw.location())))?;
        if !(nu < 0.5 && nu > -1.0) {
            return kw.err("POISSON ska ligga i (−1, 0,5)");
        }
        d[0] = 3.0 * (1.0 - 2.0 * nu) / (mu0 * (1.0 + nu));
    }
    Ok(Some(Hyperelastic { kind: kind.into(), c, d }))
}
