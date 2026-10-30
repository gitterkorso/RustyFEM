//! Exempel på användarrutiner skrivna i Rust (förregistrerade):
//!
//! - UMAT för material vars namn börjar med `EXEMPEL_ELASTISK`: linjär
//!   isotrop elasticitet, PROPS = (E, ν), inga tillståndsvariabler.
//! - UMAT för `EXEMPEL_MISES`: von Mises-plasticitet med isotropt hårdnande,
//!   PROPS = (E, ν, σy₁, ε̄p₁, σy₂, ε̄p₂, ...) som tabellen i *PLASTIC
//!   (linjär interpolation, konstant efter sista punkten), STATEV(1) = ε̄p
//!   (PEEQ), STATEV(2..1+NTENS) = plastisk töjning (ingenjörsskjuvtöjning).
//!   Radiell återföring i inkrementell form σ_tr = σ_n + D Δε med konsistent
//!   tangent (Simo–Taylor), en klassisk UMAT för isotropt hårdnande.
//!   NTENS = 6 och 4 (3D, plan töjning, axisymmetri) samt 1 (stänger).
//! - UEL U9001: linjär stång med 2 noder, MCRD = 2 eller 3 frihetsgrader per
//!   nod (1..MCRD), PROPS = (E, A[, ρ]), SVARS(1) = normalkraft,
//!   ENERGY(2) = töjningsenergi, diagonal massa om ρ ges.
//! - UEL U9002: bilinjär fyrhörning i plan spänning med full 2×2-integrering
//!   (samma som CPS4), PROPS = (E, ν, t), SVARS = (S11, S22, S12) i de fyra
//!   integrationspunkterna (ξ varierar snabbast) om NSVARS ≥ 12.
//!
//! Exemplen är geometriskt linjära (LFLAGS(2) = 1 ger fel för UEL:erna) och
//! är skrivna som en användare skulle skriva dem, oberoende av RustyFEMs
//! inbyggda material och element; testerna (tests/rigid_user.rs) visar att de
//! ger samma resultat som de inbyggda (*ELASTIC, *PLASTIC, T2D2/T3D2, CPS4).

use super::uel::{Uel, UelArgs};
use super::umat::{Umat, UmatArgs};
use crate::model::{fail, Result};
use std::sync::Arc;

/// Förregistrerade UMAT:er (prefix, rutin).
pub fn umats() -> Vec<(String, Arc<dyn Umat>)> {
    vec![("EXEMPEL_ELASTISK".into(), Arc::new(ElasticUmat)), ("EXEMPEL_MISES".into(), Arc::new(MisesUmat))]
}

/// Förregistrerade UEL:er (typnummer, rutin).
pub fn uels() -> Vec<(Option<u32>, Arc<dyn Uel>)> {
    vec![(Some(9001), Arc::new(TrussUel)), (Some(9002), Arc::new(QuadUel))]
}

// ---------------------------------------------------------------------------
// UMAT
// ---------------------------------------------------------------------------

/// Isotrop elasticitetsmatris för NTENS komponenter (kolumnvis = radvis, symmetrisk).
fn elastic_d(e: f64, nu: f64, ndi: usize, ntens: usize) -> Vec<f64> {
    let mut d = vec![0.0; ntens * ntens];
    if ntens == 1 {
        d[0] = e;
        return d;
    }
    if ndi == 2 {
        // plan spänning
        let c = e / (1.0 - nu * nu);
        d[0] = c;
        d[1] = c * nu;
        d[ntens] = c * nu;
        d[ntens + 1] = c;
        d[2 * ntens + 2] = c * (1.0 - nu) / 2.0;
        return d;
    }
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let g = e / (2.0 * (1.0 + nu));
    for i in 0..ndi {
        for j in 0..ndi {
            d[i + j * ntens] = lam;
        }
        d[i + i * ntens] = lam + 2.0 * g;
    }
    for i in ndi..ntens {
        d[i + i * ntens] = g;
    }
    d
}

fn props2(a: &UmatArgs, who: &str) -> Result<(f64, f64)> {
    if a.props.len() < 2 {
        return fail(format!("{who}: PROPS ska vara (E, ν, ...), CONSTANTS={}", a.props.len()));
    }
    Ok((a.props[0], a.props[1]))
}

/// Linjärt elastisk UMAT.
pub struct ElasticUmat;

impl Umat for ElasticUmat {
    fn name(&self) -> String {
        "exempel-UMAT (linjär elasticitet)".into()
    }

    fn umat(&self, a: &mut UmatArgs) -> Result<()> {
        let (e, nu) = props2(a, "EXEMPEL_ELASTISK")?;
        let n = a.ntens;
        let d = elastic_d(e, nu, a.ndi, n);
        a.ddsdde.copy_from_slice(&d);
        let mut de = 0.0;
        for i in 0..n {
            let ds: f64 = (0..n).map(|j| d[i + j * n] * a.dstran[j]).sum();
            de += (a.stress[i] + 0.5 * ds) * a.dstran[i];
            a.stress[i] += ds;
        }
        *a.sse += de;
        Ok(())
    }
}

/// Flytspänning och lutning vid ε̄p ur tabellen (σy, ε̄p)-par i PROPS(3..).
fn hard(table: &[f64], p: f64) -> (f64, f64) {
    let np = table.len() / 2;
    let s = |k: usize| table[2 * k];
    let e = |k: usize| table[2 * k + 1];
    if np == 1 || p <= e(0) {
        let h = if np > 1 { (s(1) - s(0)) / (e(1) - e(0)) } else { 0.0 };
        return (s(0), if p < e(0) { 0.0 } else { h });
    }
    for k in 0..np - 1 {
        if p < e(k + 1) {
            let h = (s(k + 1) - s(k)) / (e(k + 1) - e(k));
            return (s(k) + h * (p - e(k)), h);
        }
    }
    (s(np - 1), 0.0)
}

/// von Mises-plasticitet med isotropt hårdnande.
pub struct MisesUmat;

impl Umat for MisesUmat {
    fn name(&self) -> String {
        "exempel-UMAT (von Mises, isotropt hårdnande)".into()
    }

    fn umat(&self, a: &mut UmatArgs) -> Result<()> {
        let (e, nu) = props2(a, "EXEMPEL_MISES")?;
        let table = &a.props[2..];
        if table.len() < 2 || table.len() % 2 != 0 {
            return fail("EXEMPEL_MISES: PROPS ska vara (E, ν, σy₁, ε̄p₁, σy₂, ε̄p₂, ...)");
        }
        let n = a.ntens;
        if a.nstatv < 1 + n {
            return fail(format!("EXEMPEL_MISES: *DEPVAR ska vara minst {} (PEEQ och plastisk töjning)", 1 + n));
        }
        if n == 3 {
            return fail("EXEMPEL_MISES: plan spänning stöds inte av exempel-UMAT:en");
        }
        let g = e / (2.0 * (1.0 + nu));
        let d = elastic_d(e, nu, a.ndi, n);
        // prövospänning
        let mut st = vec![0.0; n];
        for i in 0..n {
            st[i] = a.stress[i] + (0..n).map(|j| d[i + j * n] * a.dstran[j]).sum::<f64>();
        }
        let p_n = a.statev[0];
        let (sy0, _) = hard(table, p_n);
        if n == 1 {
            // stång: endimensionell återföring
            let f = st[0].abs() - sy0;
            a.ddsdde[0] = e;
            if f <= 1e-12 * sy0 {
                a.stress[0] = st[0];
                return Ok(());
            }
            let mut dp = 0.0;
            for _ in 0..100 {
                let (sy, h) = hard(table, p_n + dp);
                let r = st[0].abs() - e * dp - sy;
                if r.abs() <= 1e-13 * sy0.max(st[0].abs()) {
                    break;
                }
                dp += r / (e + h);
            }
            let (_, h) = hard(table, p_n + dp);
            let sg = st[0].signum();
            a.stress[0] = st[0] - e * dp * sg;
            a.statev[0] = p_n + dp;
            a.statev[1] += dp * sg;
            a.ddsdde[0] = e * h / (e + h);
            return Ok(());
        }
        // 3D och plan töjning/axisymmetri (NDI = 3): deviator med tensorskjuvkomponenter
        let ndi = a.ndi;
        let pm = st[..ndi].iter().sum::<f64>() / 3.0;
        let mut s = st.clone();
        for v in s.iter_mut().take(ndi) {
            *v -= pm;
        }
        // s:s med skjuvkomponenterna två gånger (s33 ingår bland NDI = 3 normalkomponenter)
        let ss: f64 = (0..n).map(|i| if i < ndi { s[i] * s[i] } else { 2.0 * s[i] * s[i] }).sum();
        let q = (1.5 * ss).sqrt();
        a.ddsdde.copy_from_slice(&d);
        if q - sy0 <= 1e-12 * sy0 {
            a.stress.copy_from_slice(&st);
            return Ok(());
        }
        // q − 3G Δp − σy(p_n + Δp) = 0 med Newton
        let mut dp = 0.0;
        for _ in 0..200 {
            let (sy, h) = hard(table, p_n + dp);
            let r = q - 3.0 * g * dp - sy;
            if r.abs() <= 1e-13 * sy0.max(q) {
                break;
            }
            dp += r / (3.0 * g + h);
        }
        let (_, h) = hard(table, p_n + dp);
        let nv: Vec<f64> = s.iter().map(|x| 1.5 * x / q).collect();
        for i in 0..n {
            a.stress[i] = st[i] - 2.0 * g * dp * nv[i];
            let eng = if i < ndi { 1.0 } else { 2.0 };
            a.statev[1 + i] += eng * dp * nv[i];
        }
        a.statev[0] = p_n + dp;
        *a.spd += dp * (q - 1.5 * g * dp);
        // konsistent tangent: K 1⊗1 + 2Gθ P_dev + c n̂⊗n̂, θ = 1 − 3GΔp/q, c = 6G²(Δp/q − 1/(3G + h))
        let k = e / (3.0 * (1.0 - 2.0 * nu));
        let theta = 1.0 - 3.0 * g * dp / q;
        let c = 6.0 * g * g * (dp / q - 1.0 / (3.0 * g + h));
        let snorm = ss.sqrt();
        for i in 0..n {
            for j in 0..n {
                let mm = if i < ndi && j < ndi { 1.0 } else { 0.0 };
                let pdev = if i == j { if i < ndi { 1.0 } else { 0.5 } } else { 0.0 } - mm / 3.0;
                a.ddsdde[i + j * n] = k * mm + 2.0 * g * theta * pdev + c * s[i] / snorm * s[j] / snorm;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UEL
// ---------------------------------------------------------------------------

fn linear_only(a: &UelArgs, who: &str) -> Result<()> {
    if a.lflags[1] != 0 {
        return fail(format!("{who}: exempel-UEL:en är geometriskt linjär (NLGEOM stöds inte)"));
    }
    Ok(())
}

/// Fyller RHS = −K u, AMATRX = K och ENERGY(2) för en linjär UEL.
fn linear_response(a: &mut UelArgs, k: &[f64]) {
    let n = a.ndofel;
    let what = a.lflags[2];
    if what == 1 || what == 2 {
        a.amatrx.copy_from_slice(k);
    }
    if what == 1 || what == 5 {
        let mut w = 0.0;
        for i in 0..n {
            let f: f64 = (0..n).map(|j| k[i + j * n] * a.u[j]).sum();
            a.rhs[i] = -f;
            w += 0.5 * f * a.u[i];
        }
        a.energy[1] = w;
    }
}

/// Linjär stång med två noder.
pub struct TrussUel;

impl Uel for TrussUel {
    fn name(&self) -> String {
        "exempel-UEL U9001 (stång)".into()
    }

    fn uel(&self, a: &mut UelArgs) -> Result<()> {
        linear_only(a, "U9001")?;
        let nd = a.ndofel / 2;
        if a.nnode != 2 || a.ndofel % 2 != 0 || nd < 2 || nd > 3 || a.mcrd < nd {
            return fail("U9001: stången ska ha 2 noder med frihetsgraderna 1..MCRD (MCRD = 2 eller 3)");
        }
        if a.props.len() < 2 {
            return fail("U9001: PROPS ska vara (E, A[, ρ])");
        }
        let (e, area) = (a.props[0], a.props[1]);
        let dx: Vec<f64> = (0..nd).map(|i| a.x(1, i) - a.x(0, i)).collect();
        let l = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
        let t: Vec<f64> = dx.iter().map(|v| v / l).collect();
        let n = a.ndofel;
        if a.lflags[2] == 4 {
            // diagonal massa
            let m = a.props.get(2).copied().unwrap_or(0.0) * area * l / 2.0;
            for i in 0..n {
                a.amatrx[i + i * n] = m;
            }
            return Ok(());
        }
        let ka = e * area / l;
        let mut k = vec![0.0; n * n];
        for (p, sp) in [(0, 1.0), (1, -1.0)] {
            for (q, sq) in [(0, 1.0), (1, -1.0)] {
                for i in 0..nd {
                    for j in 0..nd {
                        k[(p * nd + i) + (q * nd + j) * n] = sp * sq * ka * t[i] * t[j];
                    }
                }
            }
        }
        linear_response(a, &k);
        if !a.svars.is_empty() {
            let du: f64 = (0..nd).map(|i| (a.u[nd + i] - a.u[i]) * t[i]).sum();
            a.svars[0] = ka * du;
        }
        Ok(())
    }
}

/// Bilinjär fyrhörning i plan spänning (som CPS4).
pub struct QuadUel;

impl Uel for QuadUel {
    fn name(&self) -> String {
        "exempel-UEL U9002 (fyrhörning, plan spänning)".into()
    }

    fn uel(&self, a: &mut UelArgs) -> Result<()> {
        linear_only(a, "U9002")?;
        if a.nnode != 4 || a.ndofel != 8 || a.mcrd < 2 {
            return fail("U9002: elementet ska ha 4 noder med frihetsgraderna 1 och 2");
        }
        if a.props.len() < 3 {
            return fail("U9002: PROPS ska vara (E, ν, t)");
        }
        let (e, nu, t) = (a.props[0], a.props[1], a.props[2]);
        let c = e / (1.0 - nu * nu);
        let dm = [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]];
        let g = 1.0 / 3f64.sqrt();
        // ordning: ξ varierar snabbast
        let pts = [(-g, -g), (g, -g), (-g, g), (g, g)];
        let xi = [-1.0, 1.0, 1.0, -1.0];
        let eta = [-1.0, -1.0, 1.0, 1.0];
        let mut k = vec![0.0; 64];
        let mut mass = vec![0.0; 64];
        let rho = a.props.get(3).copied().unwrap_or(0.0);
        let mut bs = Vec::new();
        for &(r, s) in &pts {
            let nn: Vec<f64> = (0..4).map(|i| 0.25 * (1.0 + xi[i] * r) * (1.0 + eta[i] * s)).collect();
            let dr: Vec<f64> = (0..4).map(|i| 0.25 * xi[i] * (1.0 + eta[i] * s)).collect();
            let ds: Vec<f64> = (0..4).map(|i| 0.25 * eta[i] * (1.0 + xi[i] * r)).collect();
            let mut j = [[0.0; 2]; 2];
            for i in 0..4 {
                j[0][0] += dr[i] * a.x(i, 0);
                j[0][1] += dr[i] * a.x(i, 1);
                j[1][0] += ds[i] * a.x(i, 0);
                j[1][1] += ds[i] * a.x(i, 1);
            }
            let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
            if det <= 0.0 {
                return fail(format!("U9002: element {} har negativ eller noll jacobian", a.jelem));
            }
            let mut b = [[0.0; 8]; 3];
            for i in 0..4 {
                let dx = (j[1][1] * dr[i] - j[0][1] * ds[i]) / det;
                let dy = (-j[1][0] * dr[i] + j[0][0] * ds[i]) / det;
                b[0][2 * i] = dx;
                b[1][2 * i + 1] = dy;
                b[2][2 * i] = dy;
                b[2][2 * i + 1] = dx;
            }
            let w = det * t;
            for p in 0..8 {
                for q in 0..8 {
                    let mut v = 0.0;
                    for m in 0..3 {
                        for o in 0..3 {
                            v += b[m][p] * dm[m][o] * b[o][q];
                        }
                    }
                    k[p + q * 8] += v * w;
                }
            }
            for p in 0..4 {
                for q in 0..4 {
                    let v = rho * nn[p] * nn[q] * w;
                    mass[2 * p + 2 * q * 8] += v;
                    mass[(2 * p + 1) + (2 * q + 1) * 8] += v;
                }
            }
            bs.push(b);
        }
        if a.lflags[2] == 4 {
            a.amatrx.copy_from_slice(&mass);
            return Ok(());
        }
        linear_response(a, &k);
        if a.svars.len() >= 12 {
            for (ip, b) in bs.iter().enumerate() {
                let eps: Vec<f64> = (0..3).map(|m| (0..8).map(|q| b[m][q] * a.u[q]).sum()).collect();
                for m in 0..3 {
                    a.svars[3 * ip + m] = (0..3).map(|o| dm[m][o] * eps[o]).sum();
                }
            }
        }
        Ok(())
    }
}
