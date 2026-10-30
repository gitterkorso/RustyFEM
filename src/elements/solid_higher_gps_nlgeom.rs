//! NLGEOM for the generalized plane strain elements (CPEG3, CPEG4, CPEG6,
//! CPEG8, CPEG8R).
//!
//! Formulation: the fiber length
//! between the bounding planes is
//!
//!   t = t₀ + Δu_z + Δφx (y − Y₀) − Δφy (x − X₀)
//!
//! with the *current* position (x, y) of the point, and the thickness-direction strain is
//! logarithmic, ε_zz = ln(t/t₀), with δε_zz = δt/t and
//! δt = δΔu_z + (y − Y₀) δΔφx − (x − X₀) δΔφy + Δφx δy − Δφy δx.
//!
//! The deformation gradient is block diagonal: F in the plane from the nodal displacements
//! and F₃₃ = t/t₀. The update is the same as for the CPE elements
//! (Hughes–Winget in the midpoint configuration, Cauchy stress, LE = ln V in
//! the output), with ΔF₃₃/F₃₃ in the midpoint configuration as the out-of-plane strain
//! increment. The tangent is numerical (central differences of the internal force,
//! the element has at most 19 degrees of freedom). Only fully integrated elements
//! (CPEG3, CPEG6, CPEG8, CPEG8R); CPEG4 (B-bar)
//! is not included.

use super::common::{self, Ip};
use crate::elements::solid_higher::nlgeom as nl;
use crate::elements::solid_higher::Tech;
use super::Gps;
use crate::elements::api::{ElemCtx, ElemResponse};
use crate::kinematics::tensor::{self as t, M3};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

const MODE: StressMode = StressMode::PlaneStrain;

/// Kinematics at a point for the displacement `u` (in-plane nodal degrees of freedom and
/// the three of the reference node last).
struct Kin {
    /// F in the plane (3×3 with F₃₃ = t/t₀).
    f: M3,
    /// ∇ₓN in the current position.
    g: Vec<[f64; 3]>,
    /// δt per degree of freedom.
    dt: Vec<f64>,
    /// t/t₀.
    f33: f64,
}

fn kin(el: &Gps, ip: &Ip, x: &[[f64; 3]], x0: [f64; 3], t0: f64, u: &[f64]) -> Result<Kin> {
    let nn = el.nn();
    let nin = 2 * nn;
    let mut f = t::I3;
    for a in 0..nn {
        for i in 0..2 {
            for j in 0..2 {
                f[i][j] += u[2 * a + i] * ip.dndx[a][j];
            }
        }
    }
    let (duz, phx, phy) = (u[nin], u[nin + 1], u[nin + 2]);
    let pos = |k: usize| (0..nn).map(|a| ip.n[a] * (x[a][k] + u[2 * a + k])).sum::<f64>();
    let (xc, yc) = (pos(0), pos(1));
    let tt = t0 + duz + phx * (yc - x0[1]) - phy * (xc - x0[0]);
    if !(tt > 0.0) {
        return fail(common::DISTORTED);
    }
    f[2][2] = tt / t0;
    let det2 = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    if !(det2 > 0.0) {
        return fail(common::DISTORTED);
    }
    // ∇ₓN = F⁻ᵀ ∇_X N (in the plane)
    let inv = [[f[1][1] / det2, -f[0][1] / det2], [-f[1][0] / det2, f[0][0] / det2]];
    let g: Vec<[f64; 3]> = ip.dndx.iter().map(|d| [inv[0][0] * d[0] + inv[1][0] * d[1], inv[0][1] * d[0] + inv[1][1] * d[1], 0.0]).collect();
    // the coupling to the in-plane displacements (Δφx δy − Δφy δx) gives
    // forces σ₃₃ (Δφx, −Δφy) Nₐ dv without division by t (the moments at
    // the reference node follow δt/t), so the term is entered here multiplied by t
    let mut dt = vec![0.0; nin + 3];
    for a in 0..nn {
        dt[2 * a] = -phy * ip.n[a] * tt;
        dt[2 * a + 1] = phx * ip.n[a] * tt;
    }
    dt[nin] = 1.0;
    dt[nin + 1] = yc - x0[1];
    dt[nin + 2] = -(xc - x0[0]);
    Ok(Kin { f, g, dt, f33: tt / t0 })
}

struct Eval {
    fint: Vec<f64>,
    std_new: Vec<f64>,
    ext_new: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn eval(el: &Gps, ctx: &ElemCtx, sub: &ElemCtx, ips: &[Ip], mat: &dyn MaterialModel, u: &[f64], un: &[f64], std_old: &[f64], ext_old: Option<&[f64]>) -> Result<Eval> {
    let nn = el.nn();
    let nd = u.len();
    let nips = ips.len();
    let ns = MODE.ncomp();
    let stride = common::ip_stride(MODE, mat);
    let x = &ctx.x[..nn];
    let x0 = ctx.x[nn];
    let t0 = el.base.thickness(sub)?;
    let k1: Vec<Kin> = ips.iter().map(|ip| kin(el, ip, x, x0, t0, u)).collect::<Result<_>>()?;
    let k0: Vec<Kin> = ips.iter().map(|ip| kin(el, ip, x, x0, t0, un)).collect::<Result<_>>()?;
    let det2 = |f: &M3| f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let mut fint = vec![0.0; nd];
    let mut std_new = std_old.to_vec();
    let mut ext_new = vec![0.0; 12 * nips];
    for q in 0..nips {
        let (fa, fb) = (&k0[q].f, &k1[q].f);
        let fm = t::scale(&t::add(fa, fb), 0.5);
        let (fmi, jm) = t::inv(&fm);
        if !(jm > 0.0) {
            return fail(common::DISTORTED);
        }
        // ΔL in the plane (Hughes–Winget) and ΔF₃₃/F₃₃ in the midpoint configuration
        let mut dl = t::mul(&t::sub(fb, fa), &fmi);
        let (h0, h1) = (k0[q].f33, k1[q].f33);
        dl[2][2] = (h1 - h0) / (0.5 * (h0 + h1));
        let fb = *fb;
        let de = t::sym(&dl);
        let dw = t::skew(&dl);
        let (ai, _) = t::inv(&t::sub(&t::I3, &t::scale(&dw, 0.5)));
        let dr = t::mul(&ai, &t::add(&t::I3, &t::scale(&dw, 0.5)));
        let bo = q * stride;
        let sig_n = nl::vec_to_tensor(MODE, &std_old[bo..bo + ns], false);
        let eps_n = match ext_old {
            Some(e) => nl::from6(&e[6 * nips + 6 * q..6 * nips + 6 * q + 6]),
            None => t::zero(),
        };
        let sig_r = t::rotate(&dr, &sig_n);
        let eps1 = t::add(&t::rotate(&dr, &eps_n), &de);
        let strain = nl::tensor_to_vec(MODE, &eps1, true);
        let dstrain = nl::tensor_to_vec(MODE, &de, true);
        let stress_old = nl::tensor_to_vec(MODE, &sig_r, false);
        let mut mold = std_old[bo + ns..bo + stride].to_vec();
        mat.rotate_state(&dr, &mut mold);
        let mut mnew = mold.clone();
        let (temperature, dtemp) = ctx.temperature_at(&ips[q].n[..nn]);
        let inp = MatInput { mode: MODE, strain: &strain, dstrain: &dstrain, stress_old: &stress_old, time: ctx.time, dt: ctx.dt, temperature, dtemp, defgrad: Some(fb) };
        let out = mat.update(&inp, &mold, &mut mnew)?;
        let s = &out.stress;
        // virtual work σ : δD̄ dv
        let w = ips[q].w * det2(&fb) * k1[q].f33;
        let g = &k1[q].g;
        for a in 0..nn {
            let (gx, gy) = (g[a][0], g[a][1]);
            fint[2 * a] += w * (s[0] * gx + s[3] * gy);
            fint[2 * a + 1] += w * (s[1] * gy + s[3] * gx);
        }
        for c in 0..nd {
            fint[c] += w * s[2] * k1[q].dt[c] / (t0 * k1[q].f33);
        }
        std_new[bo..bo + ns].copy_from_slice(&out.stress);
        std_new[bo + ns..bo + stride].copy_from_slice(&mnew);
        let log_e = nl::log_strain_out(MODE, &fb);
        ext_new[6 * q..6 * q + 6].copy_from_slice(&nl::tensor6(&log_e));
        ext_new[6 * nips + 6 * q..6 * nips + 6 * q + 6].copy_from_slice(&nl::tensor6(&eps1));
    }
    Ok(Eval { fint, std_new, ext_new })
}

/// Internal force and (numerical) tangent with NLGEOM.
#[allow(clippy::too_many_arguments)]
pub fn response(el: &Gps, ctx: &ElemCtx, sub: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    // CPEG4 (B-bar), CPEG4R and CPEG4I are not supported with NLGEOM
    if el.base.tech != Tech::Full {
        return crate::elements::api::not_implemented("geometric nonlinearity (NLGEOM)", el.base.name);
    }
    let ips = el.base.ips(sub)?;
    let nips = ips.len();
    let stride = common::ip_stride(MODE, mat);
    let base_len = nips * stride;
    let init;
    let (std_old, ext_old) = match crate::kinematics::ext(so) {
        Some(e) if so.len() == base_len + e.len() + 2 => (&so[..base_len], Some(e)),
        _ if so.len() == base_len => (so, None),
        _ => {
            init = common::init_state(nips, MODE, mat);
            (&init[..], None)
        }
    };
    let un: Vec<f64> = u.iter().zip(du).map(|(a, b)| a - b).collect();
    let ev = eval(el, ctx, sub, &ips, mat, u, &un, std_old, ext_old)?;
    let nd = u.len();
    let k = if tangent {
        let size = sub.x.iter().map(|p| common::norm(common::sub(*p, sub.x[0]))).fold(0.0, f64::max).max(1e-300);
        let nin = 2 * el.nn();
        let mut k = vec![0.0; nd * nd];
        let mut up = u.to_vec();
        for j in 0..nd {
            let h = if j > nin { 1e-6 } else { 1e-6 * size };
            up[j] = u[j] + h;
            let fp = eval(el, ctx, sub, &ips, mat, &up, &un, std_old, ext_old)?.fint;
            up[j] = u[j] - h;
            let fm = eval(el, ctx, sub, &ips, mat, &up, &un, std_old, ext_old)?.fint;
            up[j] = u[j];
            for i in 0..nd {
                k[i * nd + j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        nl::symmetrize(&mut k, nd);
        Some(k)
    } else {
        None
    };
    let mut st = ev.std_new;
    crate::kinematics::push_ext(&mut st, &ev.ext_new, nips);
    *sn = st;
    Ok(ElemResponse { fint: ev.fint, k })
}

/// Output with NLGEOM without NLGEOM state (frame 0): the strain is named LE.
pub fn rename_strain(out: &mut [crate::elements::api::IpField]) {
    for f in out.iter_mut().filter(|f| f.name == "E") {
        f.name = "LE".into();
        f.labels = f.labels.iter().map(|l| l.replacen('E', "LE", 1)).collect();
    }
}
