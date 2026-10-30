//! Mass matrix and internal degrees of freedom for the Euler–Bernoulli beams B23 and B33.
//!
//! B23/B33 interpolate not only the deflection but also
//! the axial displacement cubically (Hermite) and therefore have two internal
//! degrees of freedom per element (the axial strain at the ends, scaled by L). In a
//! static analysis they are condensed out exactly (the stiffness becomes EA/L), but in
//! eigenvalue steps they are part of the eigenvalue problem. This shows in axial modes and in
//! modes where axial and bending motion couple (frames), see
//! tests/cases/eigen.
//!
//! - Bending and axial (internal degrees of freedom): ∫ρA HᵀH dx with the cubic
//!   Hermite functions H, integrated with 3-point Gauss.
//!   No rotary inertia for bending.
//! - Torsion (B33): consistent linear mass with ρ(I11+I22)·L.
//! - The nodal mass (`mass`, without internal degrees of freedom) uses for axial
//!   ρAL·[5 1; 1 5]/12, which gives almost the same axial frequencies as the cubic
//!   interpolation. Eigenvalue steps use `internal_dofs` instead.
//! - Diagonal mass (MassKind::Lumped): ρAL/2 per node in translation and
//!   ρ(I11+I22)L/2 in torsion, no bending rotary mass.

use super::api::{ElemCtx, ElementType, MassKind};
use super::beam_eb::EulerBeam;
use super::common;
use crate::model::Result;

/// The module registers no element types of its own.
pub static TYPES: &[&dyn ElementType] = &[];

/// 3-point Gauss on [0, 1].
const GAUSS3: [(f64, f64); 3] = [
    (0.112_701_665_379_258_3, 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.887_298_334_620_741_7, 5.0 / 18.0),
];

/// Number of internal degrees of freedom per B23/B33 element.
pub const N_INTERNAL: usize = 2;

/// Hermite mass for total mass `m`, ordering [v1, θ1, v2, θ2] (θ scaled by L).
fn bending(m: f64, l: f64) -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for &(x, w) in &GAUSS3 {
        let h = [1.0 - 3.0 * x * x + 2.0 * x.powi(3), l * (x - 2.0 * x * x + x.powi(3)), 3.0 * x * x - 2.0 * x.powi(3), l * (x.powi(3) - x * x)];
        for i in 0..4 {
            for j in 0..4 {
                out[i][j] += m * w * h[i] * h[j];
            }
        }
    }
    out
}

/// c · ∫ H'ᵢ H'ⱼ dx for the Hermite functions, ordering [u1, s1, u2, s2].
fn hermite_grad(c: f64, l: f64) -> [[f64; 4]; 4] {
    let k = c / (30.0 * l);
    [
        [36.0 * k, 3.0 * l * k, -36.0 * k, 3.0 * l * k],
        [3.0 * l * k, 4.0 * l * l * k, -3.0 * l * k, -l * l * k],
        [-36.0 * k, -3.0 * l * k, 36.0 * k, -3.0 * l * k],
        [3.0 * l * k, -l * l * k, -3.0 * l * k, 4.0 * l * l * k],
    ]
}

/// Axial mass in the nodal mass (without internal degrees of freedom).
const AXIAL_DIAG: f64 = 5.0 / 12.0;

/// Axial (or torsional) mass for total mass `m`; `a` is the diagonal fraction
/// in the consistent variant.
fn axial(m: f64, kind: MassKind, a: f64) -> [[f64; 2]; 2] {
    match kind {
        MassKind::Lumped => [[m / 2.0, 0.0], [0.0, m / 2.0]],
        MassKind::Consistent => [[a * m, (0.5 - a) * m], [(0.5 - a) * m, a * m]],
    }
}

pub fn mass(el: &EulerBeam, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
    let d = el.data(ctx)?;
    let nd = d.n;
    let l = d.l;
    let m = d.density * d.area * l;
    let mut ml = vec![0.0; nd * nd];
    let per = nd / 2;
    let mut put2 = |a: usize, mm: [[f64; 2]; 2]| {
        for i in 0..2 {
            for j in 0..2 {
                ml[(a + i * per) * nd + a + j * per] += mm[i][j];
            }
        }
    };
    put2(0, axial(m, kind, AXIAL_DIAG));
    if nd == 12 {
        let b = ctx.section(el.name())?.beam().map(|b| b.profile.i11 + b.profile.i22).unwrap_or(0.0);
        put2(3, axial(d.density * b * l, kind, 1.0 / 3.0));
    }
    let planes: &[([usize; 4], [f64; 4])] =
        if nd == 6 { &[([1, 2, 4, 5], [1.0; 4])] } else { &[([1, 5, 7, 11], [1.0; 4]), ([2, 4, 8, 10], [1.0, -1.0, 1.0, -1.0])] };
    for (idx, sg) in planes {
        match kind {
            MassKind::Consistent => {
                let mb = bending(m, l);
                for i in 0..4 {
                    for j in 0..4 {
                        ml[idx[i] * nd + idx[j]] += sg[i] * sg[j] * mb[i][j];
                    }
                }
            }
            MassKind::Lumped => {
                ml[idx[0] * nd + idx[0]] += m / 2.0;
                ml[idx[2] * nd + idx[2]] += m / 2.0;
            }
        }
    }
    Ok(common::congruence(&ml, &d.t, nd))
}

/// Additions for the internal degrees of freedom in a B23/B33 element: matrices of
/// size (n + 2)², with the element's nodal degrees of freedom first (global
/// coordinates) and the two internal ones last. They are *added* to the element's usual
/// matrices (`stiffness`, `mass` with MassKind::Consistent and
/// `geometric_stiffness`), which they correct for the cubic axial interpolation.
pub struct BeamInternal {
    pub n: usize,
    pub k: Vec<f64>,
    pub m: Vec<f64>,
    /// Geometric stiffness (only if a displacement was given).
    pub kg: Option<Vec<f64>>,
}

/// Internal degrees of freedom for B23/B33 (`None` for other elements). `u` =
/// the element displacement for the geometric stiffness (axial force).
pub fn internal_dofs(name: &str, ctx: &ElemCtx, u: Option<&[f64]>) -> Option<Result<BeamInternal>> {
    let el = match name {
        "B23" => &super::beam_eb::B23,
        "B33" => &super::beam_eb::B33,
        _ => return None,
    };
    Some(build_internal(el, ctx, u))
}

fn build_internal(el: &EulerBeam, ctx: &ElemCtx, u: Option<&[f64]>) -> Result<BeamInternal> {
    // area beams3: curved element (beam_eb_curved.rs)
    if let Some(c) = super::beam_eb_curved::Curved::of(el, ctx)? {
        return super::beam_eb_curved::internal(&c, el, ctx);
    }
    let d = el.data(ctx)?;
    let nd = d.n;
    let ne = nd + N_INTERNAL;
    let per = nd / 2;
    let l = d.l;
    // local ordering [u1, s1, u2, s2]
    let idx = [0, nd, per, nd + 1];
    let delta = |herm: [[f64; 4]; 4], lin: [[f64; 2]; 2]| -> Vec<f64> {
        let mut a = vec![0.0; ne * ne];
        for i in 0..4 {
            for j in 0..4 {
                a[idx[i] * ne + idx[j]] += herm[i][j];
            }
        }
        for (i, &r) in [0, per].iter().enumerate() {
            for (j, &c) in [0, per].iter().enumerate() {
                a[r * ne + c] -= lin[i][j];
            }
        }
        a
    };
    let ea = d.k_local[0] * l;
    let lin = |c: f64| [[c / l, -c / l], [-c / l, c / l]];
    let m = d.density * d.area * l;
    // transformation: the element's T for the nodal degrees of freedom, identity for the internal ones
    let mut t = vec![0.0; ne * ne];
    for i in 0..nd {
        for j in 0..nd {
            t[i * ne + j] = d.t[i * nd + j];
        }
    }
    for i in nd..ne {
        t[i * ne + i] = 1.0;
    }
    let k = common::congruence(&delta(hermite_grad(ea, l), lin(ea)), &t, ne);
    let mm = common::congruence(&delta(bending(m, l), axial(m, MassKind::Consistent, AXIAL_DIAG)), &t, ne);
    let kg = match u {
        Some(u) => {
            let f = el.end_forces(ctx, u)?;
            // axial force (tension positive) = the axial force at node 2
            let n = f[per];
            let mut kgl = delta(hermite_grad(n, l), lin(n));
            if nd == 6 {
                add_moment_terms(&mut kgl, ne, &idx, &d, &common::matvec(&d.t, u));
            }
            Some(common::congruence(&kgl, &t, ne))
        }
        None => None,
    };
    Ok(BeamInternal { n: ne, k, m: mm, kg })
}

/// First and second derivatives of the Hermite functions (w.r.t. the physical coordinate) at
/// the point x ∈ [0, 1], ordering [w1, s1, w2, s2] with s = slope (scaled like θ).
fn hermite_d(x: f64, l: f64) -> ([f64; 4], [f64; 4]) {
    (
        [(-6.0 * x + 6.0 * x * x) / l, 1.0 - 4.0 * x + 3.0 * x * x, (6.0 * x - 6.0 * x * x) / l, 3.0 * x * x - 2.0 * x],
        [(-6.0 + 12.0 * x) / (l * l), (-4.0 + 6.0 * x) / l, (6.0 - 12.0 * x) / (l * l), (6.0 * x - 2.0) / l],
    )
}

/// The bending moment contribution to the geometric stiffness for B23 (local
/// coordinates): second-order curvature κ₂ = −u''·v', with M = EI·v'' from
/// the element displacement. The term couples the internal axial degrees of freedom to
/// the deflection, so it exists only with internal degrees of freedom. The form is verified with
/// buckling loads for a cantilever with end moment and transverse force, respectively
/// (tests/cases/eigen/buckle_b23_moment, buckle_b23_tvarkraft). The term is not used
/// for B33.
fn add_moment_terms(kg: &mut [f64], ne: usize, idx_u: &[usize; 4], d: &super::beam_eb::BeamData, ul: &[f64]) {
    let nd = d.n;
    let l = d.l;
    // (index for [v1, θ1, v2, θ2], sign, EI)
    let planes = [([1usize, 2, 4, 5], [1.0f64; 4], d.k_local[nd + 1] * l.powi(3) / 12.0)];
    for (iv, sg, ei) in planes {
        let v: [f64; 4] = std::array::from_fn(|i| sg[i] * ul[iv[i]]);
        for &(x, w) in &GAUSS3 {
            let (dv, ddv) = hermite_d(x, l);
            let m = ei * (0..4).map(|i| ddv[i] * v[i]).sum::<f64>();
            for a in 0..4 {
                for b in 0..4 {
                    let val = -m * ddv[a] * dv[b] * sg[b] * w * l;
                    kg[idx_u[a] * ne + iv[b]] += val;
                    kg[iv[b] * ne + idx_u[a]] += val;
                }
            }
        }
    }
}
