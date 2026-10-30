//! Geometric nonlinearity (NLGEOM=YES) for the engine in `solid_higher.rs`:
//! second-order elements (C3D10, C3D15, C3D20, CPS6, CPE8, CAX8 ...),
//! first-order tetrahedra and wedges (C3D4, C3D6), reduced-integration
//! elements with hourglass stabilization (C3D8R, CPS4R, CPE4R, CAX4R),
//! incompatible modes (C3D8I, CPS4I, CPE4I, CAX4I) and axisymmetry (CAX*).
//!
//! The formulation is the same updated Lagrangian as for CPS4/CPE4/C3D8
//! (`kinematics/continuum.rs`): Hughes–Winget increments in the midpoint configuration
//! (ΔL = ΔF·F_{n+½}⁻¹, Δε = sym ΔL, ΔR = (I − ½ΔW)⁻¹(I + ½ΔW)), Cauchy stress
//! in global axes and logarithmic strain LE = ln V in the output.
//!
//! So that the same core can handle all techniques, the kinematics at each
//! integration point is described as a displacement gradient that is linear in the element's
//! degrees of freedom d (nodal displacements and any internal degrees of freedom α):
//!
//!   H = ∂u/∂X = Σⱼ dⱼ Mⱼ,   F = I + H.
//!
//! - Nodal displacement (a, c): M = e_c ⊗ ∇_X N_a; axisymmetry additionally adds
//!   the hoop component M₃₃ = N_a / R for u_r (F₃₃ = r/R).
//! - Incompatible modes (Wilson/Taylor, in finite form according to Simo–Armero 1992):
//!   the gradient modes are added to the displacement gradient in the reference position,
//!   M = e_c ⊗ (det J₀/det J) J₀⁻ᵀ ∇_ξ φ_k, and the volumetric modes as
//!   (det J₀/det J) ψ · diag(1, 1, 1) (2D: the in-plane components). This is
//!   the same operator as the small-strain version in `solid_higher_tech.rs`.
//!   With NLGEOM the operators are used in incremental form, at the position
//!   at the start of the increment (`solid_higher_incompat.rs`).
//!
//! Virtual work in the current configuration: with lⱼ = Mⱼ F⁻¹ the
//! rate-of-deformation column is Bⱼ = sym lⱼ (Voigt) and
//!
//!   fⱼ = ∫ σ : sym lⱼ dv,
//!   Kᵢⱼ = ∫ Bᵢᵀ C Bⱼ dv + ∫ (lᵢ σ) : lⱼ dv,
//!
//! where C is the Jaumann tangent with the Truesdell correction (Belytschko et al. 2000,
//! sections 5.4 and 6.4) and the second term the stress stiffness. For nodal degrees of
//! freedom the stress stiffness becomes the usual ∫ ∇Nₐ·σ∇N_b dv I. Plane stress
//! uses the reference weight (the thickness changes so that the volume is preserved, t·da =
//! t₀·dA), as for CPS4.
//!
//! Tensorial material states (plastic strain, back stresses) are rotated with
//! ΔR together with the stress (`MaterialModel::rotate_state`).

use super::shapes;
use super::{Solid, Tech};
use crate::elements::api::{ElemCtx, ElemResponse, IpField};
use crate::elements::common::{self, Ip};
use crate::kinematics::tensor::{self as t, M3};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

/// An integration point in the reference position with the gradient operators Mⱼ.
pub struct Point {
    /// Shape functions (for temperature).
    pub n: Vec<f64>,
    /// Reference weight (det J · Gauss weight · thickness or 2πR).
    pub w0: f64,
    /// Mⱼ for all degrees of freedom (nodes first, then internal).
    pub m: Vec<M3>,
}

/// Gradient operators for the nodal degrees of freedom at a reference point.
pub fn nodal_ops(mode: StressMode, ip: &Ip, x: &[[f64; 3]]) -> Vec<M3> {
    let nd = common::ndim(mode);
    let axi = mode == StressMode::Axisymmetric;
    let r = if axi { super::radius(ip, x) } else { 1.0 };
    let mut m = Vec::with_capacity(ip.n.len() * nd);
    for (a, g) in ip.dndx.iter().enumerate() {
        for c in 0..nd {
            let mut mm = t::zero();
            for j in 0..nd {
                mm[c][j] = g[j];
            }
            if axi && c == 0 {
                mm[2][2] = ip.n[a] / r;
            }
            m.push(mm);
        }
    }
    m
}

/// The inverse and determinant of the Jacobian at the element center (as in `solid_higher_tech.rs`).
pub(crate) fn center_jacobian(el: &Solid, x: &[[f64; 3]]) -> Result<([[f64; 3]; 3], f64)> {
    let (_, dn) = shapes::shape(el.shape, [0.0; 3]);
    let dim = el.pdim();
    let mut j = [[0.0; 3]; 3];
    for r in 0..dim {
        for c in 0..dim {
            j[r][c] = (0..dn.len()).map(|i| dn[i][r] * x[i][c]).sum();
        }
    }
    if dim == 2 {
        j[2][2] = 1.0;
    }
    let (ji, det) = common::inv3(j);
    if !(det > 0.0) {
        return fail(common::DISTORTED);
    }
    Ok((ji, det))
}

/// Does the element have a volumetric mode ξη in the planar volumetric strain? CAX4I and
/// CPEG4I always have it; CPE4I gets it with NLGEOM.
fn planar_volume_mode(el: &Solid) -> bool {
    matches!(el.mode, StressMode::Axisymmetric | StressMode::PlaneStrain) || el.name.starts_with("CPEG")
}

/// The gradient operators of the incompatible modes at the points (ordering as
/// `solid_higher_tech.rs`: gradient modes k = 0.. with component c fastest,
/// then the volumetric modes).
fn incompatible_ops(el: &Solid, ctx: &ElemCtx, ips: &[Ip]) -> Result<Vec<Vec<M3>>> {
    let (ji0, det0) = center_jacobian(el, &ctx.x)?;
    let dim = el.pdim();
    let nd = el.ndim();
    let rule = el.rule.points(dim);
    let t_ = el.thickness(ctx)?;
    let axi = el.mode == StressMode::Axisymmetric;
    let r0 = if axi {
        let (n0, _) = shapes::shape(el.shape, [0.0; 3]);
        n0.iter().zip(&ctx.x).map(|(ni, xi)| ni * xi[0]).sum()
    } else {
        1.0
    };
    let mut out = Vec::with_capacity(ips.len());
    for (ip, &(p, gw)) in ips.iter().zip(&rule) {
        let (factor, sc) = match el.mode {
            StressMode::ThreeD => (1.0, 1.0),
            StressMode::Axisymmetric => {
                let r = super::radius(ip, &ctx.x);
                (2.0 * std::f64::consts::PI * r, r0 / r)
            }
            _ => (t_, 1.0),
        };
        let det = ip.w / (gw * factor);
        let s = sc * det0 / det;
        let mut ms = Vec::new();
        for k in 0..dim {
            let mut dphi = [0.0; 3];
            dphi[k] = -2.0 * p[k];
            let mut d = [0.0; 3];
            for c in 0..dim {
                d[c] = (0..dim).map(|r| ji0[c][r] * dphi[r]).sum::<f64>() * s;
            }
            for c in 0..nd {
                let mut mm = t::zero();
                for j in 0..nd {
                    mm[c][j] = d[j];
                }
                ms.push(mm);
            }
        }
        let (psis, nv): (Vec<f64>, usize) = match dim {
            3 => (vec![p[0] * p[1], p[1] * p[2], p[2] * p[0], p[0] * p[1] * p[2]], 3),
            _ if planar_volume_mode(el) => (vec![p[0] * p[1]], 2),
            _ => (Vec::new(), 0),
        };
        for psi in psis {
            let mut mm = t::zero();
            for i in 0..nv {
                mm[i][i] = psi * s;
            }
            ms.push(mm);
        }
        out.push(ms);
    }
    Ok(out)
}

/// Number of internal degrees of freedom (incompatible modes).
pub fn n_internal(el: &Solid) -> usize {
    match el.tech {
        Tech::Incompatible if el.shape.pdim() == 2 && planar_volume_mode(el) => 5,
        Tech::Incompatible => super::tech::n_incompatible(el),
        _ => 0,
    }
}

/// Integration points with gradient operators for NLGEOM.
pub fn points(el: &Solid, ctx: &ElemCtx) -> Result<Vec<Point>> {
    match el.tech {
        Tech::Hourglass => {
            // the mean gradient B̄ = (1/V)∫B dV; the hoop component (1/V)∫Nₐ/R dV
            let (ips, bs) = super::tech::uniform_strain(el, ctx)?;
            let ip = &ips[0];
            let mut m = nodal_ops(el.mode, ip, &ctx.x);
            if el.mode == StressMode::Axisymmetric {
                let ndof = bs[0].len() / 4;
                for a in 0..ip.n.len() {
                    m[2 * a][2][2] = bs[0][2 * ndof + 2 * a];
                }
            }
            Ok(vec![Point { n: ip.n.clone(), w0: ip.w, m }])
        }
        _ => {
            let ips = el.ips(ctx)?;
            let extra = if el.tech == Tech::Incompatible { Some(incompatible_ops(el, ctx, &ips)?) } else { None };
            Ok(ips
                .iter()
                .enumerate()
                .map(|(q, ip)| {
                    let mut m = nodal_ops(el.mode, ip, &ctx.x);
                    if let Some(e) = &extra {
                        m.extend_from_slice(&e[q]);
                    }
                    Point { n: ip.n.clone(), w0: ip.w, m }
                })
                .collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

/// Stress vector → symmetric tensor.
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

/// The Voigt vector of the rate of deformation (engineering shear) from l: sym l.
pub(crate) fn rate_vec(mode: StressMode, l: &M3, out: &mut [f64]) {
    match mode {
        StressMode::ThreeD => {
            out[0] = l[0][0];
            out[1] = l[1][1];
            out[2] = l[2][2];
            out[3] = l[0][1] + l[1][0];
            out[4] = l[0][2] + l[2][0];
            out[5] = l[1][2] + l[2][1];
        }
        StressMode::PlaneStrain | StressMode::Axisymmetric => {
            out[0] = l[0][0];
            out[1] = l[1][1];
            out[2] = l[2][2];
            out[3] = l[0][1] + l[1][0];
        }
        StressMode::PlaneStress => {
            out[0] = l[0][0];
            out[1] = l[1][1];
            out[2] = l[0][1] + l[1][0];
        }
        StressMode::Uniaxial => out[0] = l[0][0],
    }
}

/// 3×3 tensor row-wise as 9 values and back (stored deformation gradient).
pub(crate) fn to9(m: &M3) -> [f64; 9] {
    [m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]]
}

pub(crate) fn from9(v: &[f64]) -> M3 {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

pub(crate) fn tensor6(m: &M3) -> [f64; 6] {
    [m[0][0], m[1][1], m[2][2], m[0][1], m[0][2], m[1][2]]
}

pub(crate) fn from6(v: &[f64]) -> M3 {
    [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
}

fn voigt_pairs(mode: StressMode) -> &'static [(usize, usize)] {
    match mode {
        StressMode::ThreeD => &[(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
        StressMode::PlaneStrain | StressMode::Axisymmetric => &[(0, 0), (1, 1), (2, 2), (0, 1)],
        StressMode::PlaneStress => &[(0, 0), (1, 1), (0, 1)],
        StressMode::Uniaxial => &[(0, 0)],
    }
}

/// Material tangent for ∫ B̄ᵀσ dv: the Jaumann tangent D − C′(σ·d + d·σ) +
/// ½(σ⊗I + I⊗σ), as for CPS4/CPE4/C3D8 (plane stress: only D).
pub(crate) fn truesdell_tangent(mode: StressMode, d: &[f64], sig: &M3) -> Vec<f64> {
    if mode == StressMode::PlaneStress {
        return d.to_vec();
    }
    let pairs = voigt_pairs(mode);
    let ns = pairs.len();
    let del = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    let mut c = d.to_vec();
    for (r, &(i, j)) in pairs.iter().enumerate() {
        for (q, &(k, l)) in pairs.iter().enumerate() {
            let cp = 0.5 * (del(i, k) * sig[l][j] + del(i, l) * sig[k][j] + sig[i][k] * del(j, l) + sig[i][l] * del(j, k));
            let vol = 0.5 * (sig[i][j] * del(k, l) + del(i, j) * sig[k][l]);
            let f = if k != l { 0.5 } else { 1.0 };
            c[r * ns + q] += (vol - cp) * f * if k != l { 2.0 } else { 1.0 };
        }
    }
    c
}

/// Logarithmic strain for output (plane stress: LE33 = 0).
pub(crate) fn log_strain_out(mode: StressMode, f: &M3) -> M3 {
    let mut log_e = t::log_strain(f);
    if mode == StressMode::PlaneStress {
        log_e[2][2] = 0.0;
    }
    log_e
}

/// F̄ = (J̄/J)^(1/nv) F in the planar (nv = 2) or full (nv = 3) block; nv = 0: F.
fn fbar(f: &M3, jbar: f64, nv: usize) -> M3 {
    if nv == 0 {
        return *f;
    }
    let j = if nv == 2 { f[0][0] * f[1][1] - f[0][1] * f[1][0] } else { t::det(f) };
    let c = (jbar / j).powf(1.0 / nv as f64);
    let mut g = *f;
    for row in g.iter_mut().take(nv) {
        for v in row.iter_mut().take(nv) {
            *v *= c;
        }
    }
    g
}

/// Volumetric correction (F-bar, de Souza Neto et al. 1996): the element's
/// volume ratio J̄ = v/V and ∂ln J̄/∂d for a displacement d, and the number of
/// components nv that the volumetric part applies to (2: planar, 3: 3D and axisymmetry).
pub struct VolCorr<'a> {
    pub nv: usize,
    pub jbar: &'a dyn Fn(&[f64]) -> Result<(f64, Vec<f64>)>,
    /// The element mean of tr ΔL (weight = volume in the midpoint configuration)
    /// for (d, dn), or `None` if the volumetric part of ΔL is to be taken from F̄ directly.
    pub trace: Option<&'a dyn Fn(&[f64], &[f64]) -> Result<(f64, f64)>>,
    /// CAX4 (axisymmetry with B-bar): the hoop strain is replaced by the element's
    /// mean H̄ = (1/V)∫ r/R dV and the planar volumetric part is given by J̄/H̄.
    /// Gives (H̄, ∂ln H̄/∂d) for a displacement d.
    pub hoop: Option<&'a dyn Fn(&[f64]) -> Result<(f64, Vec<f64>)>>,
}

/// H̄ = (1/V)∫ r/R dV and ∂ln H̄/∂u for axisymmetric elements (CAX4).
pub fn hoop_mean(el: &Solid, ctx: &ElemCtx, u: &[f64]) -> Result<(f64, Vec<f64>)> {
    let ips = super::ips_for(el.shape, el.mode, &ctx.x, &super::RuleKind::Gauss(2).points(el.pdim()), 1.0)?;
    let nd = el.ndim();
    let (mut h, mut v) = (0.0, 0.0);
    let mut g = vec![0.0; u.len()];
    for ip in &ips {
        let rr = super::radius(ip, &ctx.x);
        let ur: f64 = ip.n.iter().enumerate().map(|(a, n)| n * u[a * nd]).sum();
        h += ip.w * (rr + ur) / rr;
        v += ip.w;
        for (a, n) in ip.n.iter().enumerate() {
            g[a * nd] += ip.w * n / rr;
        }
    }
    let hb = h / v;
    g.iter_mut().for_each(|x| *x /= v * hb);
    Ok((hb, g))
}

/// The mean of tr ΔL over the element with 2×2(×2) points (weight = the volume in
/// the midpoint configuration), as the volumetric part in F-bar for CPE4/C3D8.
///
/// The second value is the mean of the hoop component ΔL₃₃ (axisymmetry).
pub fn mean_trace(el: &Solid, ctx: &ElemCtx, nv: usize, d: &[f64], dn: &[f64]) -> Result<(f64, f64)> {
    let t_ = el.thickness(ctx)?;
    let ips = super::ips_for(el.shape, el.mode, &ctx.x, &super::RuleKind::Gauss(2).points(el.pdim()), t_)?;
    let (mut s, mut w, mut s33) = (0.0, 0.0, 0.0);
    for ip in &ips {
        let pt = Point { n: ip.n.clone(), w0: ip.w, m: nodal_ops(el.mode, ip, &ctx.x) };
        let f0 = grad(&pt, dn);
        let f1 = grad(&pt, d);
        let fm = t::scale(&t::add(&f0, &f1), 0.5);
        let (fmi, jm) = t::inv(&fm);
        let dl = t::mul(&t::sub(&f1, &f0), &fmi);
        s += (0..nv).map(|i| dl[i][i]).sum::<f64>() * ip.w * jm;
        s33 += dl[2][2] * ip.w * jm;
        w += ip.w * jm;
    }
    Ok((s / w, s33 / w))
}

/// J̄ = v/V and ∂ln J̄/∂u = (1/v)∫ tr B dv for the element's exact volume
/// (2×2(×2) points in the current configuration).
pub fn volume_ratio(el: &Solid, ctx: &ElemCtx, v0: f64, u: &[f64]) -> Result<(f64, Vec<f64>)> {
    let nd = el.ndim();
    let x: Vec<[f64; 3]> = ctx.x.iter().enumerate().map(|(a, p)| std::array::from_fn(|c| p[c] + if c < nd { u[a * nd + c] } else { 0.0 })).collect();
    let t_ = el.thickness(ctx)?;
    let ips = super::ips_for(el.shape, el.mode, &x, &super::RuleKind::Gauss(2).points(el.pdim()), t_)?;
    let nv = if el.mode == StressMode::PlaneStrain || el.mode == StressMode::PlaneStress { 2 } else { 3 };
    let v: f64 = ips.iter().map(|ip| ip.w).sum();
    let ndof = x.len() * nd;
    let mut g = vec![0.0; ndof];
    for ip in &ips {
        let b = super::bmat_any(el.mode, ip, &x);
        for (c, gc) in g.iter_mut().enumerate() {
            *gc += ip.w * (0..nv).map(|r| b[r * ndof + c]).sum::<f64>();
        }
    }
    g.iter_mut().for_each(|x| *x /= v);
    Ok((v / v0, g))
}

pub(crate) fn grad(pt: &Point, d: &[f64]) -> M3 {
    let mut f = t::I3;
    for (m, &dj) in pt.m.iter().zip(d) {
        if dj != 0.0 {
            for i in 0..3 {
                for k in 0..3 {
                    f[i][k] += dj * m[i][k];
                }
            }
        }
    }
    f
}

// ---------------------------------------------------------------------------
// The core
// ---------------------------------------------------------------------------

/// Result of an evaluation: internal force over all degrees of freedom (nodes and
/// internal), new state (standard layout + LE/integrated strain per point)
/// and tangent.
struct Eval {
    fint: Vec<f64>,
    std_new: Vec<f64>,
    ext_new: Vec<f64>,
    k: Option<Vec<f64>>,
}

/// Internal force and tangent for the degrees of freedom `d` (total) with `dn` at
/// the start of the increment.
#[allow(clippy::too_many_arguments)]
fn eval(
    ctx: &ElemCtx,
    temperatures: &[(f64, f64)],
    mode: StressMode,
    pts: &[Point],
    vol: Option<&VolCorr>,
    mat: &dyn MaterialModel,
    d: &[f64],
    dn: &[f64],
    std_old: &[f64],
    ext_old: Option<&[f64]>,
    tangent: bool,
) -> Result<Eval> {
    let nips = pts.len();
    // F-bar: J̄ at the start and end of the increment, ∂ln J̄/∂d at the end
    let (jb1, gvol, nv) = match vol {
        Some(v) => {
            let (b, g) = (v.jbar)(d)?;
            if !(b > 0.0) {
                return fail(common::DISTORTED);
            }
            (b, g, v.nv)
        }
        None => (1.0, Vec::new(), 0),
    };
    let vtrace = match vol.and_then(|v| v.trace) {
        Some(f) => Some(f(d, dn)?),
        None => None,
    };
    let vhoop = match vol.and_then(|v| v.hoop) {
        Some(f) => Some(f(d)?),
        None => None,
    };
    // H̄ at the start of the increment: the hoop strain increment from H̄ (see below)
    let vhoop0 = match vol.and_then(|v| v.hoop) {
        Some(f) => Some(f(dn)?),
        None => None,
    };

    let ns = mode.ncomp();
    let nd = d.len();
    let stride = common::ip_stride(mode, mat);
    let mut fint = vec![0.0; nd];
    let mut kmat = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
    let mut std_new = std_old.to_vec();
    let mut ext_new = vec![0.0; 12 * nips];
    let mut bq = vec![0.0; ns * nd];
    let mut ls: Vec<M3> = vec![t::zero(); nd];
    for (q, pt) in pts.iter().enumerate() {
        // the Hughes–Winget increment is computed with F (without volumetric correction); with
        // F-bar the volumetric part of ΔL is replaced by the element mean, and F̄ =
        // (J̄/J)^(1/nv) F is used for LE, the material's F and the integration weight
        let f0 = grad(pt, dn);
        let f1raw = grad(pt, d);
        let f1 = match &vhoop {
            // CAX4: F₃₃ = H̄, det F = J̄ (the planar part is scaled)
            Some((hb, _)) => {
                let mut g = fbar(&f1raw, jb1 / hb, 2);
                g[2][2] = *hb;
                g
            }
            None => fbar(&f1raw, jb1, nv),
        };
        let (f1ri, _) = t::inv(&f1raw);
        let fm = t::scale(&t::add(&f0, &f1raw), 0.5);
        let (fmi, jm) = t::inv(&fm);
        let j1 = t::det(&f1);
        if !(jm > 0.0) || !(j1 > 0.0) || !(t::det(&f0) > 0.0) || !(t::det(&f1raw) > 0.0) {
            return fail(common::DISTORTED);
        }
        let mut dl = t::mul(&t::sub(&f1raw, &f0), &fmi);
        // CAX4/CAX4R: the hoop strain increment ΔH̄/H̄_{n+½} from the element's
        // mean H̄ (instead of the mean of ΔL₃₃); see
        // `tests/cases/accuracy2/cax4r_*` and tests/cases/rubber/cax4h_tatning_*
        let hoop_inc = match (&vhoop, &vhoop0) {
            (Some((h1, _)), Some((h0, _))) => Some((h1 - h0) / (0.5 * (h1 + h0))),
            _ => None,
        };
        if let Some((tr, tr33)) = vtrace {
            let tr33 = hoop_inc.unwrap_or(tr33);
            if vhoop.is_some() {
                // CAX4: the hoop part and the planar volumetric part separately
                let cur = dl[0][0] + dl[1][1];
                let corr = (tr - tr33 - cur) / 2.0;
                dl[0][0] += corr;
                dl[1][1] += corr;
                dl[2][2] = tr33;
            } else {
                let cur: f64 = (0..nv).map(|i| dl[i][i]).sum();
                for i in 0..nv {
                    dl[i][i] += (tr - cur) / nv as f64;
                }
            }
        }
        let de = t::sym(&dl);
        let dw = t::skew(&dl);
        // ΔR = (I − ½ΔW)⁻¹ (I + ½ΔW)
        let (ai, _) = t::inv(&t::sub(&t::I3, &t::scale(&dw, 0.5)));
        let dr = t::mul(&ai, &t::add(&t::I3, &t::scale(&dw, 0.5)));
        let b = q * stride;
        let sig_n = vec_to_tensor(mode, &std_old[b..b + ns], false);
        let eps_n = match ext_old {
            Some(e) => from6(&e[6 * nips + 6 * q..6 * nips + 6 * q + 6]),
            None => t::zero(),
        };
        let sig_r = t::rotate(&dr, &sig_n);
        let eps_r = t::rotate(&dr, &eps_n);
        let eps1 = t::add(&eps_r, &de);
        let strain = tensor_to_vec(mode, &eps1, true);
        let dstrain = tensor_to_vec(mode, &de, true);
        let stress_old = tensor_to_vec(mode, &sig_r, false);
        let mut mold = std_old[b + ns..b + stride].to_vec();
        mat.rotate_state(&dr, &mut mold);
        let mut mnew = mold.clone();
        let (temperature, dtemp) = temperatures[q];
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
        let w = if mode == StressMode::PlaneStress { pt.w0 } else { pt.w0 * j1 };
        // lⱼ = Mⱼ F⁻¹ and the B columns
        for (j, m) in pt.m.iter().enumerate() {
            ls[j] = t::mul(m, &f1ri);
            if let Some((_, gh)) = &vhoop {
                // CAX4: δF₃₃/F₃₃ = δH̄/H̄, planar volumetric part δJ̄/J̄ − δH̄/H̄
                ls[j][2][2] = gh[j];
                let cur = ls[j][0][0] + ls[j][1][1];
                let corr = (gvol.get(j).copied().unwrap_or(0.0) - gh[j] - cur) / 2.0;
                ls[j][0][0] += corr;
                ls[j][1][1] += corr;
            } else if nv > 0 {
                // the volumetric part from J̄: δF̄·F̄⁻¹ = δF·F⁻¹ + ⅟nv (δJ̄/J̄ − tr δF·F⁻¹) I
                let tr: f64 = (0..nv).map(|i| ls[j][i][i]).sum();
                let corr = (gvol.get(j).copied().unwrap_or(0.0) - tr) / nv as f64;
                for i in 0..nv {
                    ls[j][i][i] += corr;
                }
            }
            let mut col = [0.0; 6];
            rate_vec(mode, &ls[j], &mut col);
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
            let sig = vec_to_tensor(mode, &out.stress, false);
            let ct = truesdell_tangent(mode, &out.tangent, &sig);
            add_btdb(&mut kmat, &bq, &ct, ns, nd, w);
            // stress stiffness (lᵢ σ) : lⱼ
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
        std_new[b..b + ns].copy_from_slice(&out.stress);
        std_new[b + ns..b + stride].copy_from_slice(&mnew);
        let log_e = log_strain_out(mode, &f1);
        ext_new[6 * q..6 * q + 6].copy_from_slice(&tensor6(&log_e));
        ext_new[6 * nips + 6 * q..6 * nips + 6 * q + 6].copy_from_slice(&tensor6(&eps1));
    }
    if tangent {
        symmetrize(&mut kmat, nd);
    }
    Ok(Eval { fint, std_new, ext_new, k: tangent.then_some(kmat) })
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

pub(crate) fn add_btdb(ke: &mut [f64], b: &[f64], d: &[f64], ns: usize, nd: usize, w: f64) {
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

/// Splitting of a stored NLGEOM state: (standard layout, internal
/// degrees of freedom, LE/integrated strain). Without a state (the first
/// increment after a step without NLGEOM) initial values are used.
fn split_state<'a>(so: &'a [f64], base_len: usize, na: usize) -> Option<(&'a [f64], &'a [f64], Option<&'a [f64]>)> {
    if so.len() == base_len + na {
        return Some((&so[..base_len], &so[base_len..], None));
    }
    let e = crate::kinematics::ext(so)?;
    if so.len() < base_len + na + e.len() + 2 {
        return None;
    }
    Some((&so[..base_len], &so[base_len..base_len + na], Some(e)))
}

/// Internal force and tangent with NLGEOM for the engine.
#[allow(clippy::too_many_arguments)]
pub fn response(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    if el.tech == Tech::Incompatible {
        // incremental form of the incompatible modes
        return super::incompat::response(el, ctx, mat, u, du, so, sn, tangent);
    }
    let pts = points(el, ctx)?;
    let nips = pts.len();
    let stride = common::ip_stride(el.mode, mat);
    let base_len = nips * stride;
    let na = n_internal(el);
    let nd = u.len();
    let init;
    let small = super::tech::n_incompatible(el);
    let (std_old, a_old, ext_old) = match split_state(so, base_len, na) {
        Some(s) => s,
        // state from a step without NLGEOM: stress and material state
        // are carried over, the incompatible parameters restart from zero
        None if na > 0 && so.len() == base_len + small && small != na => {
            init = [&so[..base_len], &vec![0.0; na][..]].concat();
            (&init[..base_len], &init[base_len..], None)
        }
        None => {
            init = [common::init_state(nips, el.mode, mat), vec![0.0; na]].concat();
            (&init[..base_len], &init[base_len..], None)
        }
    };
    let un: Vec<f64> = u.iter().zip(du).map(|(a, b)| a - b).collect();
    let mut d = u.to_vec();
    d.extend_from_slice(a_old);
    let mut dn = un;
    dn.extend_from_slice(a_old);
    // reduced-integration elements: the volumetric part from the element's exact volume
    let v0: f64 = pts.iter().map(|p| p.w0).sum();
    let jfun = |dd: &[f64]| volume_ratio(el, ctx, v0, &dd[..nd]);
    let nvc = if matches!(el.mode, StressMode::PlaneStrain | StressMode::PlaneStress) { 2 } else { 3 };
    let tfun = |dd: &[f64], ddn: &[f64]| mean_trace(el, ctx, nvc, &dd[..nd], &ddn[..nd]);
    let hfun = |dd: &[f64]| hoop_mean(el, ctx, &dd[..nd]);
    // CAX4 and CAX4R: the hoop component from H̄ and the planar volumetric part from J̄/H̄
    // separately (see tests/cases/accuracy2)
    let axi_bbar = matches!(el.tech, Tech::Bbar | Tech::Hourglass) && el.mode == StressMode::Axisymmetric;
    let vc = VolCorr { nv: nvc, jbar: &jfun, trace: Some(&tfun), hoop: if axi_bbar { Some(&hfun) } else { None } };
    let vol = matches!(el.tech, Tech::Hourglass | Tech::Bbar).then_some(&vc);
    let temperatures = el.temperatures(ctx);
    let mut ev = eval(ctx, &temperatures, el.mode, &pts, vol, mat, &d, &dn, std_old, ext_old, tangent || na > 0)?;
    if na > 0 {
        // local Newton for the internal degrees of freedom: f_α = 0
        let n = nd + na;
        let mut ok = false;
        for _ in 0..50 {
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
            ev = eval(ctx, &temperatures, el.mode, &pts, vol, mat, &d, &dn, std_old, ext_old, true)?;
        }
        if !ok {
            let scale = ev.fint[..nd].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let rnorm = ev.fint[nd..].iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if rnorm > 1e-8 * scale.max(1e-300) {
                return fail("the incompatible modes did not converge (NLGEOM)");
            }
        }
    }
    let mut fint = ev.fint[..nd].to_vec();
    let mut k = match (&ev.k, tangent) {
        (Some(k), true) if na > 0 => Some(condense(k, nd, na)?),
        (Some(k), true) => Some(k.clone()),
        _ => None,
    };
    // area explicit_adv: in explicit dynamics the hourglass part is replaced by steps/explicit_solid_nlgeom.rs
    if el.tech == Tech::Hourglass && !crate::settings::with(|s| s.hourglass_off) && !crate::steps::explicit::solid_nl::skip_standard() {
        // (RUSTYFEM_HG_OFF: only the stress part, for debugging)
        hourglass(el, ctx, mat, u, &mut fint, k.as_mut())?;
    }
    let mut st = ev.std_new;
    st.extend_from_slice(&d[nd..]);
    crate::kinematics::push_ext(&mut st, &ev.ext_new, nips);
    *sn = st;
    Ok(ElemResponse { fint, k })
}

/// Static condensation K = Kuu − Kuα Kαα⁻¹ Kαu.
pub(crate) fn condense(k: &[f64], nd: usize, na: usize) -> Result<Vec<f64>> {
    let n = nd + na;
    let kaa: Vec<f64> = (0..na).flat_map(|i| (0..na).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + nd + j]).collect();
    let kau: Vec<f64> = (0..na).flat_map(|i| (0..nd).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + j]).collect();
    let x = super::tech::solve_dense(&kaa, &kau, na, nd)?;
    let mut kc = vec![0.0; nd * nd];
    for i in 0..nd {
        for j in 0..nd {
            let mut s = k[i * n + j];
            for a in 0..na {
                s -= k[i * n + nd + a] * x[a * nd + j];
            }
            kc[i * nd + j] = s;
        }
    }
    Ok(kc)
}

/// Hourglass forces with NLGEOM.
///
/// The hourglass vectors γ_α and the stiffnesses s_d are the same as at small
/// strains (`tech::hourglass_matrix`, the reference geometry). The hourglass
/// displacement q_α = Σₐ γ_α,a uₐ is pulled back with the element's
/// mean deformation gradient, Q_α = F̄ᵀ q_α (F̄ = I + Σₐ uₐ ⊗ b̄ₐ, b̄ =
/// the mean gradient of the reference volume), and the energy is
///
///   E = ½ Σ_α Σ_d s_d Q_α,d².
///
/// The forces and the tangent are exact derivatives of E and
/// independent of the increment subdivision (total form).
fn hourglass(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], fint: &mut [f64], k: Option<&mut Vec<f64>>) -> Result<()> {
    let (gam, sd, bbar) = hourglass_parts(el, ctx, mat)?;
    let nd = el.ndim();
    let nn = el.shape.n_nodes();
    let n = nn * nd;
    // F̄ (nd × nd)
    let mut f = [[0.0; 3]; 3];
    for i in 0..nd {
        f[i][i] = 1.0;
        for a in 0..nn {
            for kk in 0..nd {
                f[i][kk] += u[a * nd + i] * bbar[a][kk];
            }
        }
    }
    let mut kmat = k;
    for g in &gam {
        let q: Vec<f64> = (0..nd).map(|i| (0..nn).map(|a| g[a] * u[a * nd + i]).sum()).collect();
        let qq: Vec<f64> = (0..nd).map(|kk| (0..nd).map(|i| f[i][kk] * q[i]).sum()).collect();
        // dQ_k/du_(a,j) = b̄_ak q_j + F̄_jk γ_a
        let mut dq = vec![0.0; nd * n];
        for kk in 0..nd {
            for a in 0..nn {
                for j in 0..nd {
                    dq[kk * n + a * nd + j] = bbar[a][kk] * q[j] + f[j][kk] * g[a];
                }
            }
        }
        for kk in 0..nd {
            let sq = sd[kk] * qq[kk];
            for c in 0..n {
                fint[c] += sq * dq[kk * n + c];
            }
        }
        if let Some(km) = kmat.as_deref_mut() {
            for kk in 0..nd {
                let s = sd[kk];
                let sq = s * qq[kk];
                for r in 0..n {
                    let dr = s * dq[kk * n + r];
                    if dr != 0.0 {
                        for c in 0..n {
                            km[r * n + c] += dr * dq[kk * n + c];
                        }
                    }
                }
                // the second derivative: δ_jl (b̄_ak γ_b + b̄_bk γ_a)
                for a in 0..nn {
                    for b in 0..nn {
                        let v = sq * (bbar[a][kk] * g[b] + bbar[b][kk] * g[a]);
                        for j in 0..nd {
                            km[(a * nd + j) * n + b * nd + j] += v;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Hourglass vectors γ_α (per node), stiffnesses s_d and the mean gradient b̄ₐ
/// in the reference position, as in `tech::hourglass_matrix`.
fn hourglass_parts(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<[f64; 3]>)> {
    let (ips, _) = super::tech::uniform_strain(el, ctx)?;
    let ip = &ips[0];
    let vol = ip.w;
    let nd = el.ndim();
    let nn = el.shape.n_nodes();
    let nat = shapes::node_coords(el.shape);
    let hs: Vec<Vec<f64>> = if nd == 3 {
        vec![
            nat.iter().map(|p| p[0] * p[1]).collect(),
            nat.iter().map(|p| p[1] * p[2]).collect(),
            nat.iter().map(|p| p[2] * p[0]).collect(),
            nat.iter().map(|p| p[0] * p[1] * p[2]).collect(),
        ]
    } else {
        vec![nat.iter().map(|p| p[0] * p[1]).collect()]
    };
    let x = &ctx.x;
    let gam: Vec<Vec<f64>> = hs
        .iter()
        .map(|h| {
            let hx: Vec<f64> = (0..3).map(|c| (0..nn).map(|i| h[i] * x[i][c]).sum()).collect();
            (0..nn).map(|i| h[i] - (0..nd).map(|c| hx[c] * ip.dndx[i][c]).sum::<f64>()).collect()
        })
        .collect();
    let g = super::tech::shear_modulus_of(el, mat)?;
    let sbb: Vec<f64> = (0..nd).map(|c| ip.dndx.iter().map(|b| b[c] * b[c]).sum()).collect();
    let s: Vec<f64> = if nd == 3 {
        sbb.iter().map(|v| 1.1785e-3 * g * vol * v).collect()
    } else {
        let tot: f64 = sbb.iter().sum();
        vec![0.005 / 16.0 * g * vol * tot; nd]
    };
    Ok((gam, s, ip.dndx.clone()))
}

/// Output with NLGEOM: Cauchy stress S and the material's fields from the state.
/// The strain (E) is replaced by LE in `kinematics::fix_frame_output` when
/// the state has the NLGEOM extension; otherwise (frame 0, or the first frame
/// after a step without NLGEOM) LE is written directly from the F of the nodal displacements.
pub fn output(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
    let mode = el.mode;
    let nips = match el.tech {
        Tech::Hourglass => 1,
        _ => el.rule.count(el.pdim()),
    };
    let has_ext = crate::kinematics::ext(state).is_some();
    let ns = mode.ncomp();
    let stride = common::ip_stride(mode, mat);
    let (sl, el_) = common::stress_labels(mode);
    let mut s = IpField::new("S", sl);
    let mut e = if has_ext {
        IpField::new("E", el_)
    } else {
        let lab: &[&str] = if mode == StressMode::ThreeD { &["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"] } else { &["LE11", "LE22", "LE33", "LE12"] };
        IpField::new("LE", lab)
    };
    let pts = if has_ext { Vec::new() } else { points(el, ctx)? };
    let groups = mat.output_fields(mode);
    let mut extra: Vec<IpField> = groups.iter().map(|(n, l)| IpField::new(n, l)).collect();
    for q in 0..nips {
        let st = if state.len() >= (q + 1) * stride { &state[q * stride..(q + 1) * stride] } else { &[][..] };
        let sig: Vec<f64> = if st.len() >= ns { st[..ns].to_vec() } else { vec![0.0; ns] };
        let sig = if mode == StressMode::PlaneStress { vec![sig[0], sig[1], 0.0, sig[2]] } else { sig };
        let nz = sig.len();
        s.points.push((q + 1, 0, sig));
        if has_ext {
            e.points.push((q + 1, 0, vec![0.0; nz]));
        } else {
            let pt = &pts[q];
            let f = grad(&Point { n: pt.n.clone(), w0: pt.w0, m: pt.m[..u.len()].to_vec() }, u);
            let log_e = log_strain_out(mode, &f);
            let v = tensor6(&log_e);
            e.points.push((q + 1, 0, if nz == 6 { vec![v[0], v[1], v[2], 2.0 * v[3], 2.0 * v[4], 2.0 * v[5]] } else { vec![v[0], v[1], v[2], 2.0 * v[3]] }));
        }
        if !groups.is_empty() && st.len() == stride {
            let vals = mat.output_field_values(mode, &st[ns..]);
            let mut k = 0;
            for (f, (_, l)) in extra.iter_mut().zip(&groups) {
                let v: Vec<f64> = (0..l.len()).map(|i| vals.get(k + i).copied().unwrap_or(0.0)).collect();
                k += l.len();
                f.points.push((q + 1, 0, v));
            }
        }
    }
    let mut out = vec![s, e];
    out.extend(extra);
    Ok(out)
}
