//! NLGEOM for the incompatible-mode elements (C3D8I, CPS4I, CPE4I, CAX4I)
//! in incremental form (geometrically nonlinear formulation).
//!
//! Instead of adding the modes to the total deformation gradient
//! (Simo–Armero), they are added to the incremental gradient in each increment:
//!
//!   ΔF̄ₚ = ΔF + ΔF̃,   ΔF = ∂x_{t+Δt}/∂x_t,
//!   ΔF̃ = Δαᵢ ⊗ (j(0)/j(ξ))|_t ∇ₓₜφᵢ   (t_t, j from the position at the start of the increment),
//!   ΔF̄ = ΔF̄ₚ · exp(Δβⱼ (j(0)/j(ξ))|_t ϑⱼ) (the volumetric modes multiplicatively),
//!   F̄_{n+1} = ΔF̄ · F̄_n.
//!
//! The internal parameters (Δα, Δβ) restart from zero in each increment and
//! are eliminated locally (f_α = 0). The gradient operators are built with
//! `nlgeom::points` at the position at the start of the increment. The strain increment is
//! Hughes–Winget for ΔF̄ₚ, ΔL = (ΔF̄ₚ − I)·(½(I + ΔF̄ₚ))⁻¹, plus the volumetric modes'
//! logarithmic part b·I (b = Δβⱼ (j(0)/j) ϑⱼ), and δL = δΔF̄ₚ·ΔF̄ₚ⁻¹ + δb·I.
//! The volume in the virtual work is det ΔF̄ times the volume at the start of the increment
//! in 3D but det ΔF̄ₚ in 2D (the volumetric mode does not change the weight). Plane
//! stress (CPS4I) has no volumetric mode; there j(0)/j(ξ) is the reference position ratio
//! since the thickness changes so that the volume is preserved. See
//! `tests/cases/accuracy2/*_nlgeom_inkr*`.
//! F̄_n is stored per point in the state instead of the total
//! mode parameters.

use super::nlgeom::{self as nl, Point};
use super::{Solid, Tech};
use crate::elements::api::{ElemCtx, ElemResponse};
use crate::elements::common;
use crate::kinematics::tensor::{self as t, M3};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

/// Number of values in the block after the standard layout: F̄ (9 per point).
fn fblock_len(nips: usize) -> usize {
    9 * nips
}

/// Degrees of freedom for the volumetric modes (last among the internal ones).
fn n_volume_modes(el: &Solid) -> usize {
    nl::n_internal(el) - el.pdim() * el.ndim()
}

struct Eval {
    fint: Vec<f64>,
    std_new: Vec<f64>,
    ext_new: Vec<f64>,
    fbar_new: Vec<f64>,
    k: Option<Vec<f64>>,
}

/// Internal force and tangent for the increment `dd` (nodal displacement increments
/// followed by Δα, Δβ) with the operators `pts` at the position at the start of the increment.
#[allow(clippy::too_many_arguments)]
fn eval(
    ctx: &ElemCtx,
    mode: StressMode,
    pts: &[Point],
    w_ref: &[f64],
    nvol: usize,
    mat: &dyn MaterialModel,
    dd: &[f64],
    std_old: &[f64],
    fbar_old: &[f64],
    ext_old: Option<&[f64]>,
    tangent: bool,
) -> Result<Eval> {
    let nips = pts.len();
    let ns = mode.ncomp();
    let nd = dd.len();
    let nprin = nd - nvol;
    // the volumetric modes block: 2D (plane strain, axisymmetry) only in the plane
    let nv = if mode == StressMode::ThreeD { 3 } else { 2 };
    let stride = common::ip_stride(mode, mat);
    let mut fint = vec![0.0; nd];
    let mut kmat = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
    let mut std_new = std_old.to_vec();
    let mut ext_new = vec![0.0; 12 * nips];
    let mut fbar_new = vec![0.0; 9 * nips];
    let mut bq = vec![0.0; ns * nd];
    let mut ls: Vec<M3> = vec![t::zero(); nd];
    for (q, pt) in pts.iter().enumerate() {
        // ΔF̄ₚ (nodes and gradient modes) and the exponent b of the volumetric modes
        let fp = nl::grad(&Point { n: Vec::new(), w0: 0.0, m: pt.m[..nprin].to_vec() }, &dd[..nprin]);
        let b: f64 = (0..nvol).map(|k| dd[nprin + k] * pt.m[nprin + k][0][0]).sum();
        let eb = b.exp();
        let mut df = fp;
        for row in df.iter_mut() {
            for v in row.iter_mut().take(nv) {
                *v *= eb;
            }
        }
        let (fpi, jp) = t::inv(&fp);
        let fmp = t::scale(&t::add(&t::I3, &fp), 0.5);
        let (fmpi, jm) = t::inv(&fmp);
        let jd = t::det(&df);
        if !(jm > 0.0) || !(jd > 0.0) || !(jp > 0.0) {
            return fail(common::DISTORTED);
        }
        let fold = nl::from9(&fbar_old[9 * q..9 * q + 9]);
        let f1 = t::mul(&df, &fold);
        // the strain increment: Hughes–Winget for ΔF̄ₚ plus the volumetric modes'
        // logarithmic part b·I
        let mut dl = t::mul(&t::sub(&fp, &t::I3), &fmpi);
        for i in 0..nv {
            dl[i][i] += b;
        }
        let de = t::sym(&dl);
        let dw = t::skew(&dl);
        let (ai, _) = t::inv(&t::sub(&t::I3, &t::scale(&dw, 0.5)));
        let dr = t::mul(&ai, &t::add(&t::I3, &t::scale(&dw, 0.5)));
        let bo = q * stride;
        let sig_n = nl::vec_to_tensor(mode, &std_old[bo..bo + ns], false);
        let eps_n = match ext_old {
            Some(e) => nl::from6(&e[6 * nips + 6 * q..6 * nips + 6 * q + 6]),
            None => t::zero(),
        };
        let sig_r = t::rotate(&dr, &sig_n);
        let eps_r = t::rotate(&dr, &eps_n);
        let eps1 = t::add(&eps_r, &de);
        let strain = nl::tensor_to_vec(mode, &eps1, true);
        let dstrain = nl::tensor_to_vec(mode, &de, true);
        let stress_old = nl::tensor_to_vec(mode, &sig_r, false);
        let mut mold = std_old[bo + ns..bo + stride].to_vec();
        mat.rotate_state(&dr, &mut mold);
        let mut mnew = mold.clone();
        let (temperature, dtemp) = ctx.temperature_at(&pt.n);
        let inp = MatInput {
            mode,
            strain: &strain,
            dstrain: &dstrain,
            stress_old: &stress_old,
            time: ctx.time,
            dt: ctx.dt,
            temperature,
            dtemp,
            defgrad: Some(f1),
        };
        let out = mat.update(&inp, &mold, &mut mnew)?;
        // integration weight: the volume at the start of the increment times det ΔF̄ in
        // 3D, but without the volume change of the volumetric mode in 2D (det ΔF̄ₚ); plane stress:
        // the reference volume (the thickness preserves the volume)
        let w = match mode {
            StressMode::PlaneStress => w_ref[q],
            StressMode::ThreeD => pt.w0 * jd,
            _ => pt.w0 * jp,
        };
        for j in 0..nd {
            ls[j] = if j < nprin {
                t::mul(&pt.m[j], &fpi)
            } else {
                // δΔF_I ΔF_I⁻¹ = δβ (j(0)/j) ϑ I (in the planar block for 2D)
                let mut l = t::zero();
                for i in 0..nv {
                    l[i][i] = pt.m[j][0][0];
                }
                t::mul(&t::mul(&fp, &l), &fpi)
            };
            let mut col = [0.0; 6];
            nl::rate_vec(mode, &ls[j], &mut col);
            for r in 0..ns {
                bq[r * nd + j] = col[r];
            }
        }
        for c in 0..nd {
            let mut s = 0.0;
            for r in 0..ns {
                s += bq[r * nd + c] * out.stress[r];
            }
            fint[c] += s * w;
        }
        if tangent {
            let sig = nl::vec_to_tensor(mode, &out.stress, false);
            let ct = nl::truesdell_tangent(mode, &out.tangent, &sig);
            nl::add_btdb(&mut kmat, &bq, &ct, ns, nd, w);
            let a: Vec<M3> = ls.iter().map(|l| t::mul(l, &sig)).collect();
            for i in 0..nd {
                for j in 0..nd {
                    let mut s = 0.0;
                    for p in 0..3 {
                        for r in 0..3 {
                            s += a[i][p][r] * ls[j][p][r];
                        }
                    }
                    kmat[i * nd + j] += s * w;
                }
            }
        }
        std_new[bo..bo + ns].copy_from_slice(&out.stress);
        std_new[bo + ns..bo + stride].copy_from_slice(&mnew);
        let log_e = nl::log_strain_out(mode, &f1);
        ext_new[6 * q..6 * q + 6].copy_from_slice(&nl::tensor6(&log_e));
        ext_new[6 * nips + 6 * q..6 * nips + 6 * q + 6].copy_from_slice(&nl::tensor6(&eps1));
        fbar_new[9 * q..9 * q + 9].copy_from_slice(&nl::to9(&f1));
    }
    if tangent {
        nl::symmetrize(&mut kmat, nd);
    }
    Ok(Eval { fint, std_new, ext_new, fbar_new, k: tangent.then_some(kmat) })
}

/// Internal force and tangent with NLGEOM for the incompatible-mode elements.
#[allow(clippy::too_many_arguments)]
pub fn response(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    debug_assert!(el.tech == Tech::Incompatible);
    let nd = u.len();
    let ndim = el.ndim();
    // the position at the start of the increment
    let mut c0 = ctx.clone();
    for (a, p) in c0.x.iter_mut().enumerate() {
        for c in 0..ndim {
            p[c] += u[a * ndim + c] - du[a * ndim + c];
        }
    }
    let mut pts = nl::points(el, &c0)?;
    let nips = pts.len();
    let w_ref: Vec<f64> = if el.mode == StressMode::PlaneStress { el.ips(ctx)?.iter().map(|ip| ip.w).collect() } else { Vec::new() };
    if el.mode == StressMode::PlaneStress {
        // j(ξ) = h(ξ) det(∂x/∂ξ): the thickness changes so that the volume is preserved
        // (h·da = h₀·dA), so j(0)/j(ξ) becomes the reference position ratio
        let (_, det0_x) = nl::center_jacobian(el, &ctx.x)?;
        let (_, det0_t) = nl::center_jacobian(el, &c0.x)?;
        let nn = el.shape.n_nodes() * el.ndim();
        for (q, pt) in pts.iter_mut().enumerate() {
            let f = (det0_x / det0_t) * (pt.w0 / w_ref[q]);
            for m in pt.m[nn..].iter_mut() {
                *m = t::scale(m, f);
            }
        }
    }
    let stride = common::ip_stride(el.mode, mat);
    let base_len = nips * stride;
    let na = nl::n_internal(el);
    let nvol = n_volume_modes(el);
    let fl = fblock_len(nips);
    let init;
    let (std_old, fbar_old, ext_old): (&[f64], &[f64], Option<&[f64]>) = match crate::kinematics::ext(so) {
        Some(e) if so.len() == base_len + fl + e.len() + 2 => (&so[..base_len], &so[base_len..base_len + fl], Some(e)),
        _ => {
            // first increment (or after a step without NLGEOM): stress and
            // material state are carried over if present, F̄ from the nodal displacements
            let std0 = if so.len() >= base_len { so[..base_len].to_vec() } else { common::init_state(nips, el.mode, mat) };
            let pref = nl::points(el, ctx)?;
            let un: Vec<f64> = u.iter().zip(du).map(|(a, b)| a - b).collect();
            let mut fb = Vec::with_capacity(fl);
            for p in &pref {
                let f = nl::grad(&Point { n: Vec::new(), w0: 0.0, m: p.m[..nd].to_vec() }, &un);
                fb.extend_from_slice(&nl::to9(&f));
            }
            init = [std0, fb].concat();
            (&init[..base_len], &init[base_len..], None)
        }
    };
    let mut d = du.to_vec();
    d.extend(std::iter::repeat(0.0).take(na));
    let mut ev = eval(ctx, el.mode, &pts, &w_ref, nvol, mat, &d, std_old, fbar_old, ext_old, true)?;
    // local Newton for the internal degrees of freedom: f_α = 0
    let n = nd + na;
    let mut ok = false;
    for _ in 0..100 {
        let scale = ev.fint[..nd].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rnorm = ev.fint[nd..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if rnorm <= 1e-13 * scale.max(1e-300) || rnorm == 0.0 {
            ok = true;
            break;
        }
        let k = ev.k.as_ref().unwrap();
        let kaa: Vec<f64> = (0..na).flat_map(|i| (0..na).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + nd + j]).collect();
        let step = super::tech::solve_dense(&kaa, &ev.fint[nd..], na, 1)?;
        for a in 0..na {
            d[nd + a] -= step[a];
        }
        ev = eval(ctx, el.mode, &pts, &w_ref, nvol, mat, &d, std_old, fbar_old, ext_old, true)?;
    }
    if !ok {
        let scale = ev.fint[..nd].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rnorm = ev.fint[nd..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if rnorm > 1e-8 * scale.max(1e-300) {
            return fail("the incompatible modes did not converge (NLGEOM)");
        }
    }
    let fint = ev.fint[..nd].to_vec();
    let k = if tangent { Some(nl::condense(ev.k.as_ref().unwrap(), nd, na)?) } else { None };
    let mut st = ev.std_new;
    st.extend_from_slice(&ev.fbar_new);
    crate::kinematics::push_ext(&mut st, &ev.ext_new, nips);
    *sn = st;
    Ok(ElemResponse { fint, k })
}
