//! Nodal normals for B31 (area beams3): averaged or given
//! normals (`elements/normals.rs`) that rotate about the element axis between
//! the nodes give an initially twisted beam:
//!
//! - The section orientation at the integration point is the mean angle of
//!   the nodal normals (projected perpendicular to the element) about the axis.
//!   A normal inclined towards the element axis (curvature) does not affect the stiffness.
//! - The twist gives a coupling between axial strain and twist: the strain
//!   becomes ε + τ·(I₁₁ + I₂₂)/A·θₜ' with τ = sin(α₂ − α₁)/L (αₙ = angle of the nodal normal
//!   about the axis), i.e. the section stiffness [[EA, EIₚτ], [EIₚτ, GJ + (EIₚτ)²/EA]].
//! - The rotary inertia in the diagonal mass is computed in the nodes' own
//!   directions (the tangent perpendicular to the normal), which rotate linearly along
//!   the element: Jₙ = ρ∫Nₙ R(x) diag(Iₚ, I₁₁, I₂₂) R(x)ᵀ dx.
//!
//! Applies to linear elastic B31 (not section points, NLGEOM or B32).

use super::api::ElemCtx;
use super::common::{cross, dot, norm, scale, sub};

pub static TYPES: &[&dyn super::api::ElementType] = &[];

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// The nodal normals (n2) for the element if they differ from the element's own.
fn normals(ctx: &ElemCtx, nn: usize, dim: usize) -> Option<Vec<[f64; 3]>> {
    if nn != 2 || dim != 3 {
        return None;
    }
    super::normals::of(ctx).filter(|v| v.len() >= 2)
}

/// The angle αₙ of the nodal normal about the axis t relative to the axes [t, n1, n2].
fn angle(axes: &[[f64; 3]; 3], n2: [f64; 3]) -> f64 {
    let t = axes[0];
    let p = sub(n2, scale(t, dot(n2, t)));
    (-dot(p, axes[1])).atan2(dot(p, axes[2]))
}

fn rotate(axes: &[[f64; 3]; 3], a: f64) -> [[f64; 3]; 3] {
    let (s, c) = a.sin_cos();
    let n1 = [0, 1, 2].map(|k| axes[1][k] * c + axes[2][k] * s);
    let n2 = [0, 1, 2].map(|k| axes[2][k] * c - axes[1][k] * s);
    [axes[0], n1, n2]
}

/// The axes at the integration point (mean angle) and the twist τ.
pub fn mean_axes(ctx: &ElemCtx, nn: usize, dim: usize, axes: [[f64; 3]; 3], l: f64) -> ([[f64; 3]; 3], f64) {
    let Some(nv) = normals(ctx, nn, dim) else { return (axes, 0.0) };
    let (a1, a2) = (angle(&axes, nv[0]), angle(&axes, nv[1]));
    (rotate(&axes, 0.5 * (a1 + a2)), (a2 - a1).sin() / l)
}

/// The coupling stiffness (global, 12 × 12) for the twist τ: L·(c(bε bθᵀ + bθ bεᵀ) + c²/EA bθ bθᵀ).
pub fn coupling(axes: &[[f64; 3]; 3], l: f64, tau: f64, ea: f64, eip: f64) -> Vec<f64> {
    let n = 12;
    let mut k = vec![0.0; n * n];
    if tau == 0.0 || ea == 0.0 {
        return k;
    }
    let t = axes[0];
    let mut be = vec![0.0; n];
    let mut bt = vec![0.0; n];
    for c in 0..3 {
        be[c] = -t[c] / l;
        be[6 + c] = t[c] / l;
        bt[3 + c] = -t[c] / l;
        bt[9 + c] = t[c] / l;
    }
    let c = eip * tau;
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = l * (c * (be[i] * bt[j] + bt[i] * be[j]) + c * c / ea * bt[i] * bt[j]);
        }
    }
    k
}

/// The nodes' rotary inertia (global 3 × 3 per node) for a diagonal mass, or
/// `None` if the element has no nodal normals.
pub fn rotary(ctx: &ElemCtx, nn: usize, dim: usize, rho: f64, ip: [f64; 3]) -> Option<[[[f64; 3]; 3]; 2]> {
    let nv = normals(ctx, nn, dim)?;
    let c = sub(ctx.x[1], ctx.x[0]);
    let l = norm(c);
    // the nodes' directions [t, n1, n2] with the tangent perpendicular to the normal
    let fr: Vec<[[f64; 3]; 3]> = nv
        .iter()
        .map(|&n2| {
            let n2 = unit(n2);
            let t = unit(sub(c, scale(n2, dot(c, n2))));
            [t, cross(n2, t), n2]
        })
        .collect();
    // relative rotation R₂R₁ᵀ as a rotation vector
    let r = |a: [f64; 3], b: [f64; 3]| [a[0] * b[0], a[0] * b[1], a[0] * b[2], a[1] * b[0], a[1] * b[1], a[1] * b[2], a[2] * b[0], a[2] * b[1], a[2] * b[2]];
    let mut q = [0.0; 9];
    for k in 0..3 {
        let o = r(fr[1][k], fr[0][k]);
        for i in 0..9 {
            q[i] += o[i];
        }
    }
    let tr = q[0] + q[4] + q[8];
    let ang = ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos();
    let axis = if ang > 1e-14 { unit([q[7] - q[5], q[2] - q[6], q[3] - q[1]]) } else { [1.0, 0.0, 0.0] };
    let rot = |v: [f64; 3], a: f64| -> [f64; 3] {
        let (s, co) = a.sin_cos();
        let kxv = cross(axis, v);
        let kdv = dot(axis, v);
        [0, 1, 2].map(|i| v[i] * co + kxv[i] * s + axis[i] * kdv * (1.0 - co))
    };
    let g = [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];
    let mut out = [[[0.0; 3]; 3]; 2];
    for (xi, w) in g {
        let s = (1.0 + xi) / 2.0;
        let n = [1.0 - s, s];
        let f: [[f64; 3]; 3] = std::array::from_fn(|k| rot(fr[0][k], s * ang));
        for (a, o) in out.iter_mut().enumerate() {
            let wl = rho * n[a] * w * l / 2.0;
            for k in 0..3 {
                for i in 0..3 {
                    for j in 0..3 {
                        o[i][j] += wl * ip[k] * f[k][i] * f[k][j];
                    }
                }
            }
        }
    }
    Some(out)
}
