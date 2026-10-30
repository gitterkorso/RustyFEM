//! Geometric stiffness (stress stiffness, "initial stress matrix") for
//! linear buckling (*BUCKLE) and prestressed natural frequencies.
//!
//! Continuum: the linearization of the internal forces about a stressed state in
//! an updated Lagrangian formulation where the material is described with
//! the Jaumann rate of the Kirchhoff stress. The
//! stress-dependent part of the tangent becomes (Belytschko–Liu–Moran, box 5.1 and
//! 6.x; Simo–Hughes ch. 7)
//!
//! ```text
//!   K_σ = ∫ (∇N)ᵀ σ (∇N) dV · I  −  ∫ Bᵀ C*(σ) B dV,
//!   C*_ijkl = ½(δ_ik σ_jl + δ_il σ_jk + δ_jk σ_il + δ_jl σ_ik)
//! ```
//!
//! i.e. the usual geometric matrix (second variation of the Green–Lagrange
//! strain) plus the correction from the Jaumann rate
//! (see tests/cases/eigen).
//!
//! Trusses: K_σ = N/L [I −I; −I I].
//!
//! Beams (Euler–Bernoulli, cubic Hermite interpolation): the consistent
//! matrix N/(30L)·[36 3L −36 3L; …] for each bending plane plus N/L for
//! the axial displacement. The axial force is taken from the element displacement
//! (the beams store no stress state). In the eigenvalue steps the contribution of the internal
//! axial degrees of freedom is added (`beam_mass::internal_dofs`), for B23 with
//! a term from the bending moment. No torsion term
//! (Wagner term, lateral-torsional buckling) is included for B33: a B33 cantilever
//! with only end moments gives a zero matrix and a compressed bar with small
//! torsional stiffness gets no torsional buckling.
//!
//! The functions are called from the `geometric_stiffness` stubs in the existing
//! element files.

use super::api::{ElemCtx, ElementType};
use super::beam_eb::EulerBeam;
use super::common::{self, Bbar, Ip};
use crate::materials::{MaterialModel, StressMode};
use crate::model::Result;

/// The module registers no element types of its own.
pub static TYPES: &[&dyn ElementType] = &[];

/// The stress tensor (3×3) from the state vector at an integration point.
fn stress_tensor(mode: StressMode, s: &[f64]) -> [[f64; 3]; 3] {
    let g = |i: usize| s.get(i).copied().unwrap_or(0.0);
    match mode {
        StressMode::ThreeD => [[g(0), g(3), g(4)], [g(3), g(1), g(5)], [g(4), g(5), g(2)]],
        StressMode::PlaneStrain | StressMode::Axisymmetric => [[g(0), g(3), 0.0], [g(3), g(1), 0.0], [0.0, 0.0, g(2)]],
        StressMode::PlaneStress => [[g(0), g(2), 0.0], [g(2), g(1), 0.0], [0.0, 0.0, 0.0]],
        StressMode::Uniaxial => [[g(0), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    }
}

/// Geometric stiffness for continuum elements with the integration points `ips` and
/// the stresses in `state` (layout according to `common::ip_stride`). Plane elements
/// use only the in-plane gradients (∂/∂z = 0).
pub fn continuum(mode: StressMode, ips: &[Ip], mat: &dyn MaterialModel, state: &[f64]) -> Result<Vec<f64>> {
    continuum_bbar(mode, ips, mat, state, Bbar::None)
}

/// Like `continuum`, for elements with selectively reduced volumetric strain (B-bar:
/// C3D8, CPE4). They are formulated with a modified deformation gradient
/// F̄ = (J̄/J)^{1/d} F where J̄ is the element mean of J (d = 3, planar 2).
/// The second-order term in ln J̄ − mean(ln J) = ½ Var(div u) gives the addition
///
/// ```text
///   K += ∫ (tr σ / d) (∂_i N_a − ⟨∂_i N_a⟩)(∂_j N_b − ⟨∂_j N_b⟩) dV
/// ```
///
/// (⟨·⟩ = volume mean, tr in the plane for d = 2). Otherwise K_σ is the same
/// as without B-bar. See tests/cases/eigen/buckle_c3d8_pelare,
/// buckle_cpe4_pelare.
pub fn continuum_bbar(mode: StressMode, ips: &[Ip], mat: &dyn MaterialModel, state: &[f64], bbar: Bbar) -> Result<Vec<f64>> {
    let mut k = continuum_plain(mode, ips, mat, state)?;
    if bbar == Bbar::None || ips.is_empty() {
        return Ok(k);
    }
    let nn = ips[0].n.len();
    let nd = common::ndim(mode);
    let ndof = nn * nd;
    let stride = common::ip_stride(mode, mat);
    if state.len() < ips.len() * stride {
        return Ok(k);
    }
    let nvol = if bbar == Bbar::Planar { 2 } else { 3 };
    let wsum: f64 = ips.iter().map(|ip| ip.w).sum();
    let avg: Vec<f64> = (0..ndof).map(|c| ips.iter().map(|ip| ip.w * ip.dndx[c / nd][c % nd]).sum::<f64>() / wsum).collect();
    for (q, ip) in ips.iter().enumerate() {
        let sig = stress_tensor(mode, &state[q * stride..q * stride + mode.ncomp()]);
        let p = (0..nvol).map(|i| sig[i][i]).sum::<f64>() / nvol as f64;
        let g: Vec<f64> = (0..ndof).map(|c| ip.dndx[c / nd][c % nd] - avg[c]).collect();
        for a in 0..ndof {
            for b in 0..ndof {
                k[a * ndof + b] += p * g[a] * g[b] * ip.w;
            }
        }
    }
    Ok(k)
}

fn continuum_plain(mode: StressMode, ips: &[Ip], mat: &dyn MaterialModel, state: &[f64]) -> Result<Vec<f64>> {
    let nn = ips[0].n.len();
    let nd = common::ndim(mode);
    let ndof = nn * nd;
    let stride = common::ip_stride(mode, mat);
    let mut k = vec![0.0; ndof * ndof];
    if state.len() < ips.len() * stride {
        return Ok(k);
    }
    for (q, ip) in ips.iter().enumerate() {
        let sig = stress_tensor(mode, &state[q * stride..q * stride + mode.ncomp()]);
        let sdot = |g: [f64; 3]| -> [f64; 3] {
            let mut r = [0.0; 3];
            for i in 0..nd {
                r[i] = (0..nd).map(|j| sig[i][j] * g[j]).sum();
            }
            r
        };
        for a in 0..nn {
            let ga = ip.dndx[a];
            let sa = sdot(ga);
            for b in 0..nn {
                let gb = ip.dndx[b];
                let sb = sdot(gb);
                let asb: f64 = (0..nd).map(|i| ga[i] * sb[i]).sum();
                let ab: f64 = (0..nd).map(|i| ga[i] * gb[i]).sum();
                for i in 0..nd {
                    for j in 0..nd {
                        // ∇N_a·σ·∇N_b δ_ij − ½(δ_ij ∇N_a·σ·∇N_b + ∂_i N_b (σ∇N_a)_j + ∂_j N_a (σ∇N_b)_i + ∇N_a·∇N_b σ_ij)
                        let d = if i == j { asb } else { 0.0 };
                        let v = d - 0.5 * (d + gb[i] * sa[j] + ga[j] * sb[i] + ab * sig[i][j]);
                        k[(a * nd + i) * ndof + b * nd + j] += v * ip.w;
                    }
                }
            }
        }
    }
    Ok(k)
}

/// Geometric stiffness for a truss (T2D2/T3D2) with the axial force N = S11 · A.
pub fn truss(x: &[[f64; 3]], dim: usize, area: f64, state: &[f64]) -> Vec<f64> {
    let l = common::norm(common::sub(x[1], x[0]));
    let n = state.first().copied().unwrap_or(0.0) * area;
    let nd = 2 * dim;
    let mut k = vec![0.0; nd * nd];
    let t = common::scale(common::sub(x[1], x[0]), 1.0 / l);
    for a in 0..2 {
        for b in 0..2 {
            let s = if a == b { n / l } else { -n / l };
            for d in 0..dim {
                for e in 0..dim {
                    let m = if d == e { 1.0 } else { 0.0 } - 2.0 * t[d] * t[e];
                    k[(a * dim + d) * nd + b * dim + e] = s * m;
                }
            }
        }
    }
    k
}

/// Consistent geometric stiffness for bending in one plane, ordering [v1, θ1, v2, θ2].
fn bending_kg(n: f64, l: f64) -> [[f64; 4]; 4] {
    let c = n / (30.0 * l);
    [
        [36.0 * c, 3.0 * l * c, -36.0 * c, 3.0 * l * c],
        [3.0 * l * c, 4.0 * l * l * c, -3.0 * l * c, -l * l * c],
        [-36.0 * c, -3.0 * l * c, 36.0 * c, -3.0 * l * c],
        [3.0 * l * c, -l * l * c, -3.0 * l * c, 4.0 * l * l * c],
    ]
}

/// Geometric stiffness for B23/B33 in global coordinates given the element's
/// displacement `u` (axial force N = EA/L · Δu along the beam).
pub fn beam(el: &EulerBeam, ctx: &ElemCtx, u: &[f64]) -> Result<Vec<f64>> {
    let d = el.data(ctx)?;
    let f = el.end_forces(ctx, u)?;
    let nd = d.n;
    let l = d.l;
    let mut kl = vec![0.0; nd * nd];
    let mut put = |idx: &[usize], m: &[[f64; 4]; 4], sign: [f64; 4]| {
        for i in 0..4 {
            for j in 0..4 {
                kl[idx[i] * nd + idx[j]] += sign[i] * sign[j] * m[i][j];
            }
        }
    };
    if nd == 6 {
        // [u1, v1, θ1, u2, v2, θ2]; axial force (tension positive) = f[3]
        let n = f[3];
        put(&[1, 2, 4, 5], &bending_kg(n, l), [1.0; 4]);
        kl[0] += n / l;
        kl[3 * nd + 3] += n / l;
        kl[3] -= n / l;
        kl[3 * nd] -= n / l;
    } else {
        // [u, v, w, θx, θy, θz] per node; axial force = f[6]
        let n = f[6];
        // x-y plane: v, θz (like B23)
        put(&[1, 5, 7, 11], &bending_kg(n, l), [1.0; 4]);
        // x-z plane: w, −θy
        put(&[2, 4, 8, 10], &bending_kg(n, l), [1.0, -1.0, 1.0, -1.0]);
        kl[0] += n / l;
        kl[6 * nd + 6] += n / l;
        kl[6] -= n / l;
        kl[6 * nd] -= n / l;
    }
    Ok(common::congruence(&kl, &d.t, nd))
}
