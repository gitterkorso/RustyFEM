//! Continuum elements with large deformations (NLGEOM=YES), updated Lagrangian.
//!
//! The formulation follows the standard hypoelastic formulation for
//! implicit solvers (Hughes & Winget 1980; Belytschko et al., ch. 5–6):
//!
//! - The strain increment is computed in the midpoint configuration:
//!   ΔL = ΔF · F_{n+½}⁻¹, Δε = sym ΔL, ΔW = skew ΔL.
//! - The incremental rotation ΔR = (I − ½ΔW)⁻¹ (I + ½ΔW) rotates the stress
//!   and the integrated strain from the start of the increment (Jaumann rate),
//!   and the material is updated with the rotated quantities and Δε.
//! - The stress is Cauchy stress in global axes. Internal force
//!   f = ∫ B̄ᵀ σ dv in the current configuration.
//! - Tangent: material part ∫ B̄ᵀ D B̄ dv + stress stiffness ∫ ∇Nᵀ σ ∇N dv.
//!
//! For linear elasticity this gives σ = D : ln V under proportional loading,
//! but with the midpoint rule's approximation of the log strain increment – the same
//! discrete scheme that implicit analysis uses for continuum elements.
//!
//! Plane stress: the thickness is assumed to change so that the volume is preserved (the same assumption
//! for CPS elements; verified via the reaction force), so
//! t·da = t₀·dA and the weight is the reference weight.
//!
//! Volumetric locking (CPE4, C3D8): the nonlinear counterpart of B-bar
//! (Hughes 1980; F-bar according to de Souza Neto et al. 1996):
//! - the volumetric part of the strain increment is replaced by the element average
//!   (weight = the volume in the midpoint configuration),
//! - LE is computed from F̄ = (J̄/J)^(1/nv) F with J̄ = v/V,
//! - internal force f = ∫ B̄ᵀ σ J̄ dV, where the volumetric part of B̄ is the average over the current
//!   volume (virtual work τ̄ : δd̄ with τ̄ = J̄ σ).
//! All three details are verified with one element and inhomogeneous
//! prescribed deformation (S, LE and RF).

use super::tensor::{self as t, M3};
use crate::elements::common::{self, Bbar, Ip};
use crate::elements::{ElemCtx, ElemResponse, IpField};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

thread_local! {
    static END_CONFIG: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The strain increment Δε = sym(ΔF F_{n+1}⁻¹) instead of the midpoint rule
/// in the current thread, for explicit analysis (uniaxial strain gives
/// S33 = (λ+2μ)·Σ Δl/l_{n+1}, the error relative to the log strain grows linearly with Δt). Set by `steps/explicit.rs` around the element calls.
pub fn set_explicit_end_config(on: bool) {
    END_CONFIG.with(|c| c.set(on));
}

fn explicit_end_config() -> bool {
    END_CONFIG.with(|c| c.get())
}

/// Stress vector (standard order 11, 22, 33, 12, 13, 23) → symmetric tensor.
pub fn vec_to_tensor(mode: StressMode, v: &[f64], engineering: bool) -> M3 {
    let f = if engineering { 0.5 } else { 1.0 };
    let mut m = t::zero();
    match mode {
        StressMode::ThreeD => {
            m[0][0] = v[0];
            m[1][1] = v[1];
            m[2][2] = v[2];
            m[0][1] = v[3] * f;
            m[0][2] = v[4] * f;
            m[1][2] = v[5] * f;
        }
        StressMode::PlaneStrain | StressMode::Axisymmetric => {
            m[0][0] = v[0];
            m[1][1] = v[1];
            m[2][2] = v[2];
            m[0][1] = v[3] * f;
        }
        StressMode::PlaneStress => {
            m[0][0] = v[0];
            m[1][1] = v[1];
            m[0][1] = v[2] * f;
        }
        StressMode::Uniaxial => m[0][0] = v[0],
    }
    m[1][0] = m[0][1];
    m[2][0] = m[0][2];
    m[2][1] = m[1][2];
    m
}

/// Symmetric tensor → vector in the component order of `mode`.
pub fn tensor_to_vec(mode: StressMode, m: &M3, engineering: bool) -> Vec<f64> {
    let f = if engineering { 2.0 } else { 1.0 };
    match mode {
        StressMode::ThreeD => vec![m[0][0], m[1][1], m[2][2], m[0][1] * f, m[0][2] * f, m[1][2] * f],
        StressMode::PlaneStrain | StressMode::Axisymmetric => vec![m[0][0], m[1][1], m[2][2], m[0][1] * f],
        StressMode::PlaneStress => vec![m[0][0], m[1][1], m[0][1] * f],
        StressMode::Uniaxial => vec![m[0][0]],
    }
}

/// Deformation gradient F = I + ∂u/∂X at an integration point (2D: F33 = 1).
pub fn defgrad(ndim: usize, dndx: &[[f64; 3]], u: &[f64]) -> M3 {
    let mut f = t::I3;
    for (a, g) in dndx.iter().enumerate() {
        for i in 0..ndim {
            let ui = u[a * ndim + i];
            for j in 0..ndim {
                f[i][j] += ui * g[j];
            }
        }
    }
    f
}

fn tensor6(m: &M3) -> [f64; 6] {
    [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]
}

fn from6(v: &[f64]) -> M3 {
    [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
}

/// Logarithmic strain for output (LE) in a 2D/3D state.
pub fn log_strain_out(mode: StressMode, f: &M3) -> M3 {
    let mut le = t::log_strain(f);
    if mode == StressMode::PlaneStress {
        // the thickness strain is not reported in LE for CPS elements (LE33 = 0)
        le[2][2] = 0.0;
    }
    le
}

/// Output with NLGEOM before the element has an NLGEOM state (frame 0, or the first
/// frame after a step without NLGEOM): E is replaced by LE from the nodal displacements'
/// F̄ (the same F-bar as in `response`). When the state exists,
/// `kinematics::fix_frame_output` does the replacement.
pub fn initial_log_strain(mode: StressMode, ips: &[Ip], bbar: Bbar, u: &[f64], state: &[f64], fields: &mut Vec<IpField>) {
    if super::ext(state).is_some() {
        return;
    }
    let ndim = common::ndim(mode);
    let fs: Vec<M3> = ips.iter().map(|ip| defgrad(ndim, &ip.dndx, u)).collect();
    let nv = if bbar == Bbar::Planar { 2 } else { 3 };
    let jdet = |f: &M3| if nv == 2 { f[0][0] * f[1][1] - f[0][1] * f[1][0] } else { t::det(f) };
    if !fs.iter().all(|f| jdet(f) > 0.0) {
        return;
    }
    let jbar = {
        let (s, w) = ips.iter().zip(&fs).fold((0.0, 0.0), |(s, w), (ip, f)| (s + jdet(f) * ip.w, w + ip.w));
        s / w
    };
    let labels: &[&str] = if mode == StressMode::ThreeD { &["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"] } else { &["LE11", "LE22", "LE33", "LE12"] };
    let mut le = IpField::new("LE", labels);
    for (q, f) in fs.iter().enumerate() {
        let mut fb = *f;
        if bbar != Bbar::None {
            let c = (jbar / jdet(f)).powf(1.0 / nv as f64);
            for row in fb.iter_mut().take(nv) {
                for v in row.iter_mut().take(nv) {
                    *v *= c;
                }
            }
        }
        let m = log_strain_out(mode, &fb);
        // shear components as engineering strain
        let vals = if mode == StressMode::ThreeD {
            vec![m[0][0], m[1][1], m[2][2], 2.0 * m[0][1], 2.0 * m[0][2], 2.0 * m[1][2]]
        } else {
            vec![m[0][0], m[1][1], m[2][2], 2.0 * m[0][1]]
        };
        le.points.push((q + 1, 0, vals));
    }
    fields.retain(|f| f.name != "E" && f.name != "LE");
    fields.push(le);
}

/// Internal force and tangent for a continuum element with large deformations.
///
/// `ips` are the integration points in the reference configuration (derivatives with respect to X,
/// weight = det J · Gauss weight · thickness). The tangent is the material part (with
/// Truesdell correction) plus the stress stiffness, symmetrized. With
/// the environment variable `RUSTYFEM_NL_FDTANGENT` the exact tangent is used instead
/// (central differences of the internal force, symmetrized) – for debugging.
#[allow(clippy::too_many_arguments)]
pub fn response(
    ctx: &ElemCtx,
    mode: StressMode,
    ips: &[Ip],
    bbar: Bbar,
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
) -> Result<ElemResponse> {
    // feature area rubber: hyperelastic materials in F-bar elements and plane stress (fbar_hyper.rs)
    if let Some(r) = super::fbar_hyper::try_response(ctx, mode, ips, bbar, mat, u, du, state_old, state_new, tangent)? {
        return Ok(r);
    }
    // explicit analysis: materials such as *EOS are fully integrated (without B-bar)
    let bbar = if explicit_end_config() && mat.explicit_full_integration() { Bbar::None } else { bbar };
    let nips = ips.len();
    let stride = common::ip_stride(mode, mat);
    let base_len = nips * stride;
    let init;
    let (std_old, ext_old) = if state_old.len() >= base_len && (state_old.len() == base_len || super::ext(state_old).is_some()) {
        (&state_old[..base_len], super::ext(state_old))
    } else {
        init = common::init_state(nips, mode, mat);
        (&init[..], None)
    };
    let un: Vec<f64> = u.iter().zip(du).map(|(a, b)| a - b).collect();
    let (fint, st, k) = eval(ctx, mode, ips, bbar, mat, u, &un, std_old, ext_old, true, tangent)?;
    if let Some(st) = st {
        *state_new = st;
    }
    let k = if tangent && crate::settings::with(|s| s.nl_fd_tangent) {
        // check: the exact (unsymmetric) tangent with central differences
        let nd = u.len();
        let size = ctx.x.iter().map(|p| common::norm(common::sub(*p, ctx.x[0]))).fold(0.0, f64::max).max(1e-300);
        let h = 1e-6 * size;
        let mut k = vec![0.0; nd * nd];
        let mut up = u.to_vec();
        for j in 0..nd {
            up[j] = u[j] + h;
            let (fp, _, _) = eval(ctx, mode, ips, bbar, mat, &up, &un, std_old, ext_old, false, false)?;
            up[j] = u[j] - h;
            let (fm, _, _) = eval(ctx, mode, ips, bbar, mat, &up, &un, std_old, ext_old, false, false)?;
            up[j] = u[j];
            for i in 0..nd {
                k[i * nd + j] = (fp[i] - fm[i]) / (2.0 * h);
            }
        }
        symmetrize(&mut k, nd);
        Some(k)
    } else {
        k
    };
    Ok(ElemResponse { fint, k })
}

/// Internal force (and new state if `with_state`) for the displacement `u`
/// with `un` at the start of the increment.
#[allow(clippy::too_many_arguments)]
fn eval(
    ctx: &ElemCtx,
    mode: StressMode,
    ips: &[Ip],
    bbar: Bbar,
    mat: &dyn MaterialModel,
    u: &[f64],
    un: &[f64],
    std_old: &[f64],
    ext_old: Option<&[f64]>,
    with_state: bool,
    tangent: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>, Option<Vec<f64>>)> {
    let nips = ips.len();
    let ndim = common::ndim(mode);
    let ns = mode.ncomp();
    let nd = u.len();
    let stride = common::ip_stride(mode, mat);
    // kinematics at all points
    struct Kin {
        dl: M3,
        f1: M3,
        g: Vec<[f64; 3]>,
    }
    let mut kin = Vec::with_capacity(nips);
    let length = common::char_length(&ctx.x, mode, ips);
    let fs0: Vec<M3> = ips.iter().map(|ip| defgrad(ndim, &ip.dndx, un)).collect();
    let fs1: Vec<M3> = ips.iter().map(|ip| defgrad(ndim, &ip.dndx, u)).collect();
    // The volumetric part (B-bar, nv = 2 for the in-plane part in plane strain)
    let nv = if bbar == Bbar::Planar { 2 } else { 3 };
    let jdet = |f: &M3| if nv == 2 { f[0][0] * f[1][1] - f[0][1] * f[1][0] } else { t::det(f) };
    // J̄ = v/V and F̄ = (J̄/J)^(1/nv) F (for LE and the integration weight)
    let jbar = if bbar == Bbar::None {
        1.0
    } else {
        let (s, w) = ips.iter().zip(&fs1).fold((0.0, 0.0), |(s, w), (ip, f)| (s + jdet(f) * ip.w, w + ip.w));
        s / w
    };
    let mut j1s = Vec::with_capacity(nips);
    let (mut tr_sum, mut wm_sum) = (0.0, 0.0);
    for (q, ip) in ips.iter().enumerate() {
        let f1 = fs1[q];
        // explicit analysis computes the strain increment in the configuration at
        // the end of the increment (`explicit_end_config`), implicit in the midpoint configuration
        let fm = if explicit_end_config() { f1 } else { t::scale(&t::add(&fs0[q], &f1), 0.5) };
        let (fmi, jm) = t::inv(&fm);
        let (f1i, j1) = t::inv(&f1);
        if !(jm > 0.0) || !(j1 > 0.0) || !(jdet(&fs0[q]) > 0.0) || !(jdet(&f1) > 0.0) {
            return fail(common::DISTORTED);
        }
        let dl = t::mul(&t::sub(&f1, &fs0[q]), &fmi);
        tr_sum += (0..nv).map(|i| dl[i][i]).sum::<f64>() * ip.w * jm;
        wm_sum += ip.w * jm;
        let mut fb = f1;
        if bbar != Bbar::None {
            let c = (jbar / jdet(&f1)).powf(1.0 / nv as f64);
            for row in fb.iter_mut().take(nv) {
                for v in row.iter_mut().take(nv) {
                    *v *= c;
                }
            }
        }
        // ∇ₓN = F⁻ᵀ ∇_X N
        let g = ip.dndx.iter().map(|gx| t::mv(&t::transpose(&f1i), *gx)).collect();
        kin.push(Kin { dl, f1: fb, g });
        j1s.push(j1);
    }
    if bbar != Bbar::None {
        // the volumetric part of ΔL is replaced by the element average (weight = volume in the midpoint configuration)
        let avg = tr_sum / wm_sum;
        for k in kin.iter_mut() {
            let tr: f64 = (0..nv).map(|i| k.dl[i][i]).sum();
            for i in 0..nv {
                k.dl[i][i] += (avg - tr) / nv as f64;
            }
        }
    }
    // B matrices in the current configuration; the volumetric part of B̄ is averaged over
    // the current volume (weight det J · w)
    let cur: Vec<Ip> = ips
        .iter()
        .zip(&kin)
        .zip(&j1s)
        .map(|((ip, k), &j1)| Ip { n: ip.n.clone(), dndx: k.g.clone(), w: ip.w * j1 })
        .collect();
    // Integration weight: plane stress t·da = t₀·dA; with B-bar J̄·dV (virtual
    // work τ̄ : δd̄ dV with τ̄ = J̄ σ, the weight of the F-bar formulation), otherwise dv.
    let wint: Vec<f64> = ips
        .iter()
        .zip(&cur)
        .map(|(ip, c)| match (mode, bbar) {
            (StressMode::PlaneStress, _) => ip.w,
            (_, Bbar::None) => c.w,
            _ => ip.w * jbar,
        })
        .collect();
    let bs = common::bmats(mode, &cur, bbar);

    let mut fint = vec![0.0; nd];
    let mut kmat = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
    let mut sn_std = std_old.to_vec();
    let mut ext_new = vec![0.0; 12 * nips];
    // (1/nv) Σ σ_ii per point for the volumetric term in the tangent (bbar_tangent.rs)
    let mut pvol = vec![0.0; nips];
    for (q, k) in kin.iter().enumerate() {
        let mut de = t::sym(&k.dl);
        let dw = t::skew(&k.dl);
        if mode == StressMode::PlaneStrain {
            de[2][2] = 0.0;
        }
        // ΔR = (I − ½ΔW)⁻¹ (I + ½ΔW)
        let (ai, _) = t::inv(&t::sub(&t::I3, &t::scale(&dw, 0.5)));
        let dr = t::mul(&ai, &t::add(&t::I3, &t::scale(&dw, 0.5)));
        let b = q * stride;
        let sig_n = vec_to_tensor(mode, &std_old[b..b + ns], false);
        let eps_n = match ext_old {
            Some(e) => from6(&e[6 * nips + 6 * q..6 * nips + 6 * q + 6]),
            None => t::zero(),
        };
        // explicit analysis: the material is updated with σₙ and Δε, then
        // the Jaumann term ΔWσₙ − σₙΔW is added (additively, with the stress at
        // the start of the increment). Checked: simple shear γ = 0.8 with 1000 increments
        // gives the same S12 and S11 to 7 digits elastically and with von Mises plasticity.
        let spin = |a: &M3| t::sub(&t::mul(&dw, a), &t::mul(a, &dw));
        let explicit = explicit_end_config();
        let sig_r = if explicit { sig_n } else { t::rotate(&dr, &sig_n) };
        let eps_r = if explicit { eps_n } else { t::rotate(&dr, &eps_n) };
        let eps1 = t::add(&eps_r, &de);
        let strain = tensor_to_vec(mode, &eps1, true);
        let dstrain = tensor_to_vec(mode, &de, true);
        let stress_old = tensor_to_vec(mode, &sig_r, false);
        // the material's tensor state (plastic strain, back stresses) follows ΔR like the stress
        let mut mold = std_old[b + ns..b + stride].to_vec();
        if !explicit {
            mat.rotate_state(&dr, &mut mold);
        }
        let mut mnew = mold.clone();
        // user routines: the integration point in the current configuration (user/context.rs)
        {
            let mut xc = [0.0; 3];
            for (i, (ni, x)) in ips[q].n.iter().zip(&ctx.x).enumerate() {
                for d in 0..3 {
                    xc[d] += ni * (x[d] + if d < ndim { u.get(i * ndim + d).copied().unwrap_or(0.0) } else { 0.0 });
                }
            }
            crate::user::context::set_ip(q + 1, xc, common::char_length(&ctx.x, mode, ips), None, ips[q].w);
        }
        let inp = MatInput {
            mode,
            strain: &strain,
            dstrain: &dstrain,
            stress_old: &stress_old,
            time: ctx.time,
            dt: ctx.dt,
            temperature: 0.0,
            dtemp: 0.0,
            defgrad: Some(k.f1),
        };
        // with the element's characteristic length (damage evolution, Hillerborg),
        // as without NLGEOM (`common::response`)
        let mut out = mat.update_with_length(&inp, length, &mold, &mut mnew)?;
        let eps1 = if explicit {
            // the tensor state (plastic strain etc.) gets the same additive
            // rotation increment as the stress, so that ε − εᵖ remains C⁻¹ : σ
            let mut mrot = mold.clone();
            mat.rotate_state(&dr, &mut mrot);
            for ((n, r), o) in mnew.iter_mut().zip(&mrot).zip(&mold) {
                *n += r - o;
            }
            let sig = t::add(&vec_to_tensor(mode, &out.stress, false), &spin(&sig_n));
            out.stress = tensor_to_vec(mode, &sig, false);
            t::add(&eps1, &spin(&eps_n))
        } else {
            eps1
        };
        let bq = &bs[q];
        let w = wint[q];
        for c in 0..nd {
            let mut s = 0.0;
            for r in 0..ns {
                s += bq[r * nd + c] * out.stress[r];
            }
            fint[c] += s * w;
        }
        if tangent {
            let ct = truesdell_tangent(mode, &out.tangent, &vec_to_tensor(mode, &out.stress, false));
            add_btdb(&mut kmat, bq, &ct, ns, nd, w);
            // stress stiffness ∫ ∇Nₐ·σ∇N_b dv I
            let sig = vec_to_tensor(mode, &out.stress, false);
            let g = &cur[q].dndx;
            let nn = g.len();
            for a in 0..nn {
                for bb in 0..nn {
                    let mut sg = 0.0;
                    for i in 0..ndim {
                        for j in 0..ndim {
                            sg += g[a][i] * sig[i][j] * g[bb][j];
                        }
                    }
                    for i in 0..ndim {
                        kmat[(a * ndim + i) * nd + bb * ndim + i] += sg * w;
                    }
                }
            }
        }
        pvol[q] = (0..nv).map(|i| out.stress[i]).sum::<f64>() / nv as f64;
        sn_std[b..b + ns].copy_from_slice(&out.stress);
        sn_std[b + ns..b + stride].copy_from_slice(&mnew);
        if with_state {
            let le = log_strain_out(mode, &k.f1);
            ext_new[6 * q..6 * q + 6].copy_from_slice(&tensor6(&le));
            ext_new[6 * nips + 6 * q..6 * nips + 6 * q + 6].copy_from_slice(&tensor6(&eps1));
        }
    }
    if tangent && bbar != Bbar::None && mode != StressMode::PlaneStress {
        let g: Vec<Vec<[f64; 3]>> = cur.iter().map(|c| c.dndx.clone()).collect();
        let v: Vec<f64> = cur.iter().map(|c| c.w).collect();
        super::bbar_tangent::add_volume_term(&mut kmat, ndim, nv, &g, &v, &wint, &pvol);
    }
    let k = tangent.then_some(kmat);
    if !with_state {
        return Ok((fint, None, k));
    }
    let mut st = sn_std;
    super::push_ext(&mut st, &ext_new, nips);
    Ok((fint, Some(st), k))
}

pub fn symmetrize(k: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (k[i * n + j] + k[j * n + i]);
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
    }
}

fn add_btdb(ke: &mut [f64], b: &[f64], d: &[f64], ns: usize, nd: usize, w: f64) {
    let mut db = vec![0.0; ns * nd];
    for r in 0..ns {
        for m in 0..ns {
            let drm = d[r * ns + m];
            if drm != 0.0 {
                for c in 0..nd {
                    db[r * nd + c] += drm * b[m * nd + c];
                }
            }
        }
    }
    for r in 0..ns {
        for i in 0..nd {
            let bri = b[r * nd + i] * w;
            if bri != 0.0 {
                for j in 0..nd {
                    ke[i * nd + j] += bri * db[r * nd + j];
                }
            }
        }
    }
}

/// Tensor index (i, j) for each Voigt component in `mode`.
fn voigt_pairs(mode: StressMode) -> &'static [(usize, usize)] {
    match mode {
        StressMode::ThreeD => &[(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
        StressMode::PlaneStrain | StressMode::Axisymmetric => &[(0, 0), (1, 1), (2, 2), (0, 1)],
        StressMode::PlaneStress => &[(0, 0), (1, 1), (0, 1)],
        StressMode::Uniaxial => &[(0, 0)],
    }
}

/// Material tangent for the internal force ∫ B̄ᵀσ dv: the Jaumann tangent D minus
/// C′ (σ·d + d·σ) plus ½(σ⊗I + I⊗σ) (the contribution of the volume change, symmetrized).
/// (Belytschko et al., sections 5.4 and 6.4). The unsymmetric rotation part of
/// the Hughes–Winget update is not included, since the symmetrized exact
/// tangent converges worse at large rotations.
fn truesdell_tangent(mode: StressMode, d: &[f64], sig: &M3) -> Vec<f64> {
    if mode == StressMode::PlaneStress {
        // plane stress: the pure material tangent gives the same iteration history
        // (and hence the same automatic increments) in the cases tried
        return d.to_vec();
    }
    let pairs = voigt_pairs(mode);
    let ns = pairs.len();
    let del = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let mut c = d.to_vec();
    for (r, &(i, j)) in pairs.iter().enumerate() {
        for (q, &(k, l)) in pairs.iter().enumerate() {
            let cp = 0.5 * (del(i, k) * sig[l][j] + del(i, l) * sig[k][j] + sig[i][k] * del(j, l) + sig[i][l] * del(j, k));
            let vol = if mode == StressMode::PlaneStress { 0.0 } else { 0.5 * (sig[i][j] * del(k, l) + del(i, j) * sig[k][l]) };
            // shear columns: engineering strain γ = 2ε, so C_v = C_ijkl without a factor
            let f = if k != l { 0.5 } else { 1.0 };
            c[r * ns + q] += (vol - cp) * f * if k != l { 2.0 } else { 1.0 };
        }
    }
    c
}
