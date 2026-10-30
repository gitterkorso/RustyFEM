//! Hyperelastic continuum elements with NLGEOM: F-bar (CPE4, C3D8) and plane
//! stress (CPS4, CPS3), total formulation without incremental rotation.
//!
//! Hyperelastic stress depends only on F, so the element is computed directly from
//! the displacement (no Jaumann integration as for hypoelastic materials in
//! `continuum.rs`):
//!
//! **F-bar** (Nagtegaal et al. 1974; Simo, Taylor & Pister 1985; de Souza Neto
//! et al. 1996, with the volume average instead of the midpoint value):
//! F̄ = (J̄/J)^(1/n) F in the n volumetric directions (n = 3 in 3D, n = 2 in the plane for
//! plane strain) with J̄ = v/V. The internal force is the variation of
//! Π = Σ_q W(F̄_q) w_q:
//!
//!   f·δu = Σ_q w_q τ̄_q : [∇ₓδu + (1/n)(δv/v − div δu) I_n],   τ̄ = τ(F̄)
//!
//! **Volumetric hourglass stiffness**: for hyperelastic materials in CPE4 and C3D8 a
//! stiffness is added against the local volume change deviating from
//! the element average, which is not visible in the stress S. It is verified
//! with some thirty single-element tests with prescribed inhomogeneous
//! deformation (Neo-Hooke, Mooney–Rivlin, Yeoh, distorted elements):
//! the energy per reference volume is
//!
//!   W_h = c μ0 (x²/2 − ln x − 1/2),   x = J/J̄,
//!
//! with μ0 = the initial shear modulus of the material, c = 5/6 in 3D and c = 1 for plane
//! strain. The contribution to the force is Σ_q w_q c μ0 (x_q² − 1)(div_q δu − δv/v),
//! independent of the bulk modulus; at small strains it corresponds to
//! (5/3) μ0 ∫ (ε_v − ε̄_v)² dV in 3D and 2 μ0 ∫ (...)² dA in the plane.
//!
//! **Plane stress**: λ3 = F33 is solved so that τ33 = 0, and the weight is the actual
//! thickness, t dA = J t0 dA0 / j_plane, i.e. f = Σ ∇N·τ w0 with τ = J σ
//! (verified for compressible materials; the hypoelastic
//! branch assumes volume preservation).
//!
//! The tangent is the exact linearization (symmetric, since the force is
//! an energy variation): the material part with C^(τJ) of F̄ and spin terms,
//! geometric stiffness, the second derivatives of the projection, and the hourglass term.

use super::continuum::{defgrad, log_strain_out, symmetrize, tensor_to_vec};
use super::tensor::{self as t, M3};
use crate::elements::common::{self, Bbar, Ip};
use crate::elements::{ElemCtx, ElemResponse};
use crate::materials::hyperelastic_split::{self as hs, HyperModel};
use crate::materials::{MaterialModel, StressMode};
use crate::model::{fail, Result};

/// The coefficient c for the volumetric hourglass stiffness (n = number of volumetric directions).
pub fn hourglass_coef(nv: usize) -> f64 {
    if nv == 3 {
        5.0 / 6.0
    } else {
        1.0
    }
}

/// Called first in `continuum::response`: handles hyperelastic materials
/// in F-bar elements and in plane stress, otherwise `None`.
#[allow(clippy::too_many_arguments)]
pub fn try_response(
    _ctx: &ElemCtx,
    mode: StressMode,
    ips: &[Ip],
    bbar: Bbar,
    mat: &dyn MaterialModel,
    u: &[f64],
    _du: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
) -> Result<Option<ElemResponse>> {
    let Some(h) = mat.hyperelastic() else { return Ok(None) };
    let applies = match mode {
        StressMode::PlaneStress => true,
        StressMode::PlaneStrain | StressMode::ThreeD => bbar != Bbar::None,
        _ => false,
    };
    if !applies || mat.n_state() != h.n_hist() {
        return Ok(None);
    }
    if h.incompressible() {
        return fail("*HYPERELASTIC with D1 = 0 (fully incompressible) requires hybrid elements (C3D8H, CPE4H, ...)");
    }
    let r = eval_hist(mode, ips, bbar, h, u, tangent, true, Some(state_old))?;
    *state_new = r.state;
    Ok(Some(ElemResponse { fint: r.fint, k: r.k }))
}

pub struct Eval {
    pub fint: Vec<f64>,
    pub k: Option<Vec<f64>>,
    pub state: Vec<f64>,
}

/// Kinematics per integration point.
struct Pt {
    w0: f64,
    /// local volume ratio (the n block of F) and the full F̄
    j: f64,
    fb: M3,
    /// ∇ₓN for all nodes
    g: Vec<[f64; 3]>,
}

/// Plane stress: condense Δτ33 = 0 out of the 6×6 tangent (components 11, 22, 12).
fn condense_ps(c: &[[f64; 6]; 6]) -> [[f64; 3]; 3] {
    let idx = [0usize, 1, 3];
    let mut r = [[0.0; 3]; 3];
    for (a, &i) in idx.iter().enumerate() {
        for (b, &j) in idx.iter().enumerate() {
            r[a][b] = c[i][j] - c[i][2] * c[2][j] / c[2][2];
        }
    }
    r
}

/// Internal force, tangent and state (stress + the NLGEOM extension with LE).
pub fn eval(mode: StressMode, ips: &[Ip], bbar: Bbar, h: &dyn HyperModel, u: &[f64], tangent: bool, hourglass: bool) -> Result<Eval> {
    eval_hist(mode, ips, bbar, h, u, tangent, hourglass, None)
}

/// Like `eval`, with the material's history variables (`HyperModel::n_hist`, e.g.
/// *MULLINS EFFECT) from the element state at the start of the increment; they are stored
/// after the stress at each point (stride = ncomp + n_hist).
#[allow(clippy::too_many_arguments)]
pub fn eval_hist(mode: StressMode, ips: &[Ip], bbar: Bbar, h: &dyn HyperModel, u: &[f64], tangent: bool, hourglass: bool, hist_old: Option<&[f64]>) -> Result<Eval> {
    let nips = ips.len();
    let ndim = common::ndim(mode);
    let nn = ips[0].dndx.len();
    let nd = nn * ndim;
    let ns = mode.ncomp();
    let nv = match (mode, bbar) {
        (StressMode::PlaneStress, _) | (_, Bbar::None) => 0,
        (_, Bbar::Planar) => 2,
        (_, Bbar::Volumetric) => 3,
    };
    let jn = |f: &M3| if nv == 3 { t::det(f) } else { f[0][0] * f[1][1] - f[0][1] * f[1][0] };
    let mut pts = Vec::with_capacity(nips);
    let (mut vol0, mut vol) = (0.0, 0.0);
    for ip in ips {
        let f = defgrad(ndim, &ip.dndx, u);
        let (fi, jf) = t::inv(&f);
        let j = jn(&f);
        if !(jf > 0.0) || !(j > 0.0) {
            return fail(common::DISTORTED);
        }
        let g = ip.dndx.iter().map(|gx| t::mv(&t::transpose(&fi), *gx)).collect();
        vol0 += ip.w;
        vol += ip.w * j;
        pts.push(Pt { w0: ip.w, j, fb: f, g });
    }
    let jbar = vol / vol0;
    // Ḡ = (1/v) Σ j w0 G_q  (G_q[a·ndim + i] = ∂N_a/∂x_i)
    let mut gbar = vec![0.0; nd];
    if nv > 0 {
        for p in &pts {
            for a in 0..nn {
                for i in 0..ndim {
                    gbar[a * ndim + i] += p.j * p.w0 * p.g[a][i] / vol;
                }
            }
        }
    }
    let mu0 = h.mu0();
    let hc = if nv > 0 && hourglass { hourglass_coef(nv) * mu0 } else { 0.0 };
    let mut fint = vec![0.0; nd];
    let mut kmat = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
    let nh = h.n_hist();
    let stride = ns + nh;
    let zeros = vec![0.0; nh];
    let old_of = |q: usize| -> &[f64] {
        match hist_old {
            Some(so) if nh > 0 && so.len() >= nips * stride => &so[q * stride + ns..(q + 1) * stride],
            _ => &zeros[..],
        }
    };
    let mut state = vec![0.0; nips * stride];
    let mut ext = vec![0.0; 12 * nips];
    // sum of the coefficients in front of Δ(δv/v) (projection and hourglass)
    let mut mg_coef = 0.0;
    for (q, p) in pts.iter_mut().enumerate() {
        if nv > 0 {
            let c = (jbar / p.j).powf(1.0 / nv as f64);
            for r in 0..nv {
                for s in 0..nv {
                    p.fb[r][s] *= c;
                }
            }
        } else if mode == StressMode::PlaneStress {
            let j2 = p.j;
            p.fb[2][2] = if h.k0() > 10.0 * mu0 { 1.0 / j2 } else { 1.0 };
            p.fb[2][2] = hs::plane_stress_stretch_hist(h, &p.fb, old_of(q))?;
        }
        let mut hn = vec![0.0; nh];
        let tau = hs::kirchhoff_hist(h, &p.fb, old_of(q), &mut hn)?;
        let jfull = t::det(&p.fb);
        let sig = t::scale(&tau, 1.0 / jfull);
        state[q * stride..q * stride + ns].copy_from_slice(&tensor_to_vec(mode, &sig, false));
        state[q * stride + ns..(q + 1) * stride].copy_from_slice(&hn);
        let le = log_strain_out(mode, &p.fb);
        let le6 = [le[0][0], le[1][1], le[2][2], le[0][1], le[0][2], le[1][2]];
        ext[6 * q..6 * q + 6].copy_from_slice(&le6);
        ext[6 * nips + 6 * q..6 * nips + 6 * q + 6].copy_from_slice(&le6);
        let pn: f64 = (0..nv).map(|i| tau[i][i]).sum();
        let x = p.j / jbar;
        let psi = x * x - 1.0;
        let w0 = p.w0;
        for a in 0..nn {
            for i in 0..ndim {
                let r = a * ndim + i;
                let mut s: f64 = (0..ndim).map(|jj| tau[i][jj] * p.g[a][jj]).sum();
                if nv > 0 {
                    s += pn / nv as f64 * (gbar[r] - p.g[a][i]);
                    s += hc * psi * (p.g[a][i] - gbar[r]);
                }
                fint[r] += w0 * s;
            }
        }
        if !tangent {
            continue;
        }
        let k = &mut kmat;
        // ΔL̄ per degree of freedom (9 components, row-wise ij)
        let mut lb = vec![[0.0f64; 9]; nd];
        for b in 0..nn {
            for kk in 0..ndim {
                let c = b * ndim + kk;
                for jj in 0..ndim {
                    lb[c][kk * 3 + jj] = p.g[b][jj];
                }
                if nv > 0 {
                    let corr = (gbar[c] - p.g[b][kk]) / nv as f64;
                    for i in 0..nv {
                        lb[c][i * 3 + i] += corr;
                    }
                }
            }
        }
        // material part: Dvᵀ C Dv (engineering shear)
        let ct = hs::jaumann_tangent(&|f: &M3| hs::kirchhoff_hist(h, f, old_of(q), &mut vec![0.0; nh]), &p.fb)?;
        let dv: Vec<[f64; 6]> = lb
            .iter()
            .map(|l| [l[0], l[4], l[8], l[1] + l[3], l[2] + l[6], l[5] + l[7]])
            .collect();
        if mode == StressMode::PlaneStress {
            let cp = condense_ps(&ct);
            let id = [0usize, 1, 3];
            for r in 0..nd {
                let dr = [dv[r][0], dv[r][1], dv[r][3]];
                let cd: [f64; 3] = std::array::from_fn(|m| (0..3).map(|n| cp[n][m] * dr[n]).sum());
                for c in 0..nd {
                    let s: f64 = (0..3).map(|m| cd[m] * dv[c][id[m]]).sum();
                    k[r * nd + c] += w0 * s;
                }
            }
        } else {
            for r in 0..nd {
                let cd: [f64; 6] = std::array::from_fn(|m| (0..6).map(|n| ct[n][m] * dv[r][n]).sum());
                for c in 0..nd {
                    let s: f64 = (0..6).map(|m| cd[m] * dv[c][m]).sum();
                    k[r * nd + c] += w0 * s;
                }
            }
        }
        // spin part: δL̄ : (ΔW τ − τ ΔW)
        for c in 0..nd {
            let l = &lb[c];
            let lm: M3 = [[l[0], l[1], l[2]], [l[3], l[4], l[5]], [l[6], l[7], l[8]]];
            let w = t::skew(&lm);
            let xm = t::sub(&t::mul(&w, &tau), &t::mul(&tau, &w));
            for r in 0..nd {
                let lr = &lb[r];
                let mut s = 0.0;
                for i in 0..3 {
                    for jj in 0..3 {
                        s += lr[i * 3 + jj] * xm[i][jj];
                    }
                }
                k[r * nd + c] += w0 * s;
            }
        }
        // geometric part −τ : (∇δu ∇Δu) and the T_q of the projection/hourglass
        let tg: Vec<[f64; 3]> = p.g.iter().map(|gb| t::mv(&tau, *gb)).collect();
        let pq = if nv > 0 { pn / nv as f64 } else { 0.0 };
        for a in 0..nn {
            for i in 0..ndim {
                let r = a * ndim + i;
                for b in 0..nn {
                    for kk in 0..ndim {
                        let c = b * ndim + kk;
                        let tq = p.g[a][kk] * p.g[b][i];
                        k[r * nd + c] += w0 * (-p.g[a][kk] * tg[b][i] + (pq - hc * psi) * tq);
                    }
                }
            }
        }
        if nv > 0 {
            // hourglass: 2 x² (G_q − Ḡ)(G_q − Ḡ)ᵀ
            if hc != 0.0 {
                let d: Vec<f64> = (0..nd).map(|r| p.g[r / ndim][r % ndim] - gbar[r]).collect();
                for r in 0..nd {
                    for c in 0..nd {
                        k[r * nd + c] += w0 * hc * 2.0 * x * x * d[r] * d[c];
                    }
                }
            }
            mg_coef += w0 * (pq - hc * psi);
        }
    }
    if tangent && nv > 0 {
        // M_G = Δ(δv/v) = (1/v) Σ_r j_r w_r [G_r G_rᵀ − T_r] − Ḡ Ḡᵀ
        let k = &mut kmat;
        for p in &pts {
            let s = mg_coef * p.j * p.w0 / vol;
            for a in 0..nn {
                for i in 0..ndim {
                    let r = a * ndim + i;
                    for b in 0..nn {
                        for kk in 0..ndim {
                            let c = b * ndim + kk;
                            k[r * nd + c] += s * (p.g[a][i] * p.g[b][kk] - p.g[a][kk] * p.g[b][i]);
                        }
                    }
                }
            }
        }
        for r in 0..nd {
            for c in 0..nd {
                k[r * nd + c] -= mg_coef * gbar[r] * gbar[c];
            }
        }
    }
    if tangent {
        symmetrize(&mut kmat, nd);
    }
    super::push_ext(&mut state, &ext, nips);
    Ok(Eval { fint, k: tangent.then_some(kmat), state })
}
