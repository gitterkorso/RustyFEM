//! Examples of user subroutines written in Rust (pre-registered):
//!
//! - UMAT for materials whose name starts with `EXAMPLE_ELASTIC`: linear
//!   isotropic elasticity, PROPS = (E, ν), no state variables.
//! - UMAT for `EXAMPLE_MISES`: von Mises plasticity with isotropic hardening,
//!   PROPS = (E, ν, σy₁, ε̄p₁, σy₂, ε̄p₂, ...) like the table in *PLASTIC
//!   (linear interpolation, constant after the last point), STATEV(1) = ε̄p
//!   (PEEQ), STATEV(2..1+NTENS) = plastic strain (engineering shear strain).
//!   Radial return in incremental form σ_tr = σ_n + D Δε with consistent
//!   tangent (Simo–Taylor), a classic UMAT for isotropic hardening.
//!   NTENS = 6 and 4 (3D, plane strain, axisymmetry) as well as 1 (trusses).
//! - UEL U9001: linear two-node truss, MCRD = 2 or 3 degrees of freedom per
//!   node (1..MCRD), PROPS = (E, A[, ρ]), SVARS(1) = axial force,
//!   ENERGY(2) = strain energy, lumped mass if ρ is given.
//! - UEL U9002: bilinear plane-stress quadrilateral with full 2×2 integration
//!   (same as CPS4), PROPS = (E, ν, t), SVARS = (S11, S22, S12) at the four
//!   integration points (ξ varies fastest) if NSVARS ≥ 12.
//!
//! The examples are geometrically linear (LFLAGS(2) = 1 gives an error for the UELs) and
//! are written the way a user would write them, independently of RustyFEM's
//! built-in materials and elements; the tests (tests/rigid_user.rs) show that they
//! give the same results as the built-in ones (*ELASTIC, *PLASTIC, T2D2/T3D2, CPS4).

use super::uel::{Uel, UelArgs};
use super::umat::{Umat, UmatArgs};
use crate::model::{fail, Result};
use std::sync::Arc;

/// Pre-registered UMATs (prefix, routine).
pub fn umats() -> Vec<(String, Arc<dyn Umat>)> {
    vec![("EXAMPLE_ELASTIC".into(), Arc::new(ElasticUmat)), ("EXAMPLE_MISES".into(), Arc::new(MisesUmat))]
}

/// Pre-registered UELs (type number, routine).
pub fn uels() -> Vec<(Option<u32>, Arc<dyn Uel>)> {
    vec![(Some(9001), Arc::new(TrussUel)), (Some(9002), Arc::new(QuadUel))]
}

// ---------------------------------------------------------------------------
// UMAT
// ---------------------------------------------------------------------------

/// Isotropic elasticity matrix for NTENS components (column-wise = row-wise, symmetric).
fn elastic_d(e: f64, nu: f64, ndi: usize, ntens: usize) -> Vec<f64> {
    let mut d = vec![0.0; ntens * ntens];
    if ntens == 1 {
        d[0] = e;
        return d;
    }
    if ndi == 2 {
        // plane stress
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
        return fail(format!("{who}: PROPS must be (E, ν, ...), CONSTANTS={}", a.props.len()));
    }
    Ok((a.props[0], a.props[1]))
}

/// Linear elastic UMAT.
pub struct ElasticUmat;

impl Umat for ElasticUmat {
    fn name(&self) -> String {
        "example UMAT (linear elasticity)".into()
    }

    fn umat(&self, a: &mut UmatArgs) -> Result<()> {
        let (e, nu) = props2(a, "EXAMPLE_ELASTIC")?;
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

/// Yield stress and slope at ε̄p from the table of (σy, ε̄p) pairs in PROPS(3..).
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

/// von Mises plasticity with isotropic hardening.
pub struct MisesUmat;

impl Umat for MisesUmat {
    fn name(&self) -> String {
        "example UMAT (von Mises, isotropic hardening)".into()
    }

    fn umat(&self, a: &mut UmatArgs) -> Result<()> {
        let (e, nu) = props2(a, "EXAMPLE_MISES")?;
        let table = &a.props[2..];
        if table.len() < 2 || table.len() % 2 != 0 {
            return fail("EXAMPLE_MISES: PROPS must be (E, ν, σy₁, ε̄p₁, σy₂, ε̄p₂, ...)");
        }
        let n = a.ntens;
        if a.nstatv < 1 + n {
            return fail(format!("EXAMPLE_MISES: *DEPVAR must be at least {} (PEEQ and plastic strain)", 1 + n));
        }
        if n == 3 {
            return fail("EXAMPLE_MISES: plane stress is not supported by the example UMAT");
        }
        let g = e / (2.0 * (1.0 + nu));
        let d = elastic_d(e, nu, a.ndi, n);
        // trial stress
        let mut st = vec![0.0; n];
        for i in 0..n {
            st[i] = a.stress[i] + (0..n).map(|j| d[i + j * n] * a.dstran[j]).sum::<f64>();
        }
        let p_n = a.statev[0];
        let (sy0, _) = hard(table, p_n);
        if n == 1 {
            // truss: one-dimensional return mapping
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
        // 3D and plane strain/axisymmetry (NDI = 3): deviator with tensor shear components
        let ndi = a.ndi;
        let pm = st[..ndi].iter().sum::<f64>() / 3.0;
        let mut s = st.clone();
        for v in s.iter_mut().take(ndi) {
            *v -= pm;
        }
        // s:s with the shear components twice (s33 is among the NDI = 3 normal components)
        let ss: f64 = (0..n).map(|i| if i < ndi { s[i] * s[i] } else { 2.0 * s[i] * s[i] }).sum();
        let q = (1.5 * ss).sqrt();
        a.ddsdde.copy_from_slice(&d);
        if q - sy0 <= 1e-12 * sy0 {
            a.stress.copy_from_slice(&st);
            return Ok(());
        }
        // q − 3G Δp − σy(p_n + Δp) = 0 by Newton
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
        // consistent tangent: K 1⊗1 + 2Gθ P_dev + c n̂⊗n̂, θ = 1 − 3GΔp/q, c = 6G²(Δp/q − 1/(3G + h))
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
        return fail(format!("{who}: the example UEL is geometrically linear (NLGEOM is not supported)"));
    }
    Ok(())
}

/// Fills RHS = −K u, AMATRX = K and ENERGY(2) for a linear UEL.
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

/// Linear two-node truss.
pub struct TrussUel;

impl Uel for TrussUel {
    fn name(&self) -> String {
        "example UEL U9001 (truss)".into()
    }

    fn uel(&self, a: &mut UelArgs) -> Result<()> {
        linear_only(a, "U9001")?;
        let nd = a.ndofel / 2;
        if a.nnode != 2 || a.ndofel % 2 != 0 || nd < 2 || nd > 3 || a.mcrd < nd {
            return fail("U9001: the truss must have 2 nodes with degrees of freedom 1..MCRD (MCRD = 2 or 3)");
        }
        if a.props.len() < 2 {
            return fail("U9001: PROPS must be (E, A[, ρ])");
        }
        let (e, area) = (a.props[0], a.props[1]);
        let dx: Vec<f64> = (0..nd).map(|i| a.x(1, i) - a.x(0, i)).collect();
        let l = dx.iter().map(|v| v * v).sum::<f64>().sqrt();
        let t: Vec<f64> = dx.iter().map(|v| v / l).collect();
        let n = a.ndofel;
        if a.lflags[2] == 4 {
            // lumped mass
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

/// Bilinear plane-stress quadrilateral (like CPS4).
pub struct QuadUel;

impl Uel for QuadUel {
    fn name(&self) -> String {
        "example UEL U9002 (quadrilateral, plane stress)".into()
    }

    fn uel(&self, a: &mut UelArgs) -> Result<()> {
        linear_only(a, "U9002")?;
        if a.nnode != 4 || a.ndofel != 8 || a.mcrd < 2 {
            return fail("U9002: the element must have 4 nodes with degrees of freedom 1 and 2");
        }
        if a.props.len() < 3 {
            return fail("U9002: PROPS must be (E, ν, t)");
        }
        let (e, nu, t) = (a.props[0], a.props[1], a.props[2]);
        let c = e / (1.0 - nu * nu);
        let dm = [[c, c * nu, 0.0], [c * nu, c, 0.0], [0.0, 0.0, c * (1.0 - nu) / 2.0]];
        let g = 1.0 / 3f64.sqrt();
        // order: ξ varies fastest
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
                return fail(format!("U9002: element {} has a negative or zero Jacobian", a.jelem));
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
