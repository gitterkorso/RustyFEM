//! Element-by-element stability limit for explicit dynamics
//! (`ELEMENT BY ELEMENT`, `IMPROVED DT METHOD=NO`).
//!
//! Δtₑ = Lₑ/c_d · (√(1+ξ²) − ξ), where c_d = √(D₁₁/ρ) (D = the elastic tangent of the
//! material: λ+2μ for solids and plane strain, E/(1−ν²) for plane stress,
//! shells and membranes) and ξ the critical damping ratio in the highest mode from
//! the bulk viscosity (ξ = b₁ − b₂² Lₑ/c_d · min(0, ε̇_vol)) and Rayleigh β/α.
//! The characteristic length Lₑ with bᵢ = ∫∇Nᵢ dV (checked with the field EDT
//! for free elements of different shapes):
//!
//! | Element | Lₑ |
//! |---|---|
//! | 3D solids (C3D8, C3D8R, C3D4, ...) | V / √(2 Σᵢ\|bᵢ\|²) |
//! | quadrilaterals (CPE4, CPS4R, S4R, M3D4R, ...) | A / √(Σᵢ\|bᵢ\|²) = A / √((d₁² + d₂²)/2) (d = diagonals) |
//! | triangles (CPE3, S3R, M3D3, ...) | 4A / √(3 Σ lₖ²) (l = edges) |
//! | trusses | 0.99 L |
//!
//! The default method (`IMPROVED DT METHOD=YES`) is used for shells and
//! membranes (`improved_plane_stress`: √0.92 and √0.97 times the exact
//! highest eigenfrequency for constant strain, respectively). For 3D solids the
//! formulas above are used even with the method, which is on the safe side. Plane
//! strain (CPE3, CPE4R) is not affected by the method.
//! Beams and elements missing from the table get the element's highest
//! eigenfrequency with diagonal mass (power iteration); for beams this is
//! a conservative limit.

use crate::elements::common::{cross, dot, norm, sub};
use crate::elements::{ElemCtx, Family};
use crate::fem::assembly::Prepared;
use crate::materials::StressMode;
use crate::model::Result;

/// Element geometry for stability and bulk viscosity: bᵢ = ∫∇Nᵢ dV
/// (without thickness for planar elements and shells) and the volume (area).
pub struct ElemGeom {
    pub b: Vec<[f64; 3]>,
    pub vol: f64,
    /// Characteristic length.
    pub lchar: f64,
}

/// Characteristic length of the truss relative to L (see the module documentation).
pub const TRUSS_LE: f64 = 0.99;

fn hex8_b(x: &[[f64; 3]]) -> Option<(Vec<[f64; 3]>, f64)> {
    let nat: [[f64; 3]; 8] = [[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]];
    let g = 1.0 / 3f64.sqrt();
    let mut b = vec![[0.0; 3]; 8];
    let mut vol = 0.0;
    for &p in &[[-g, -g, -g], [g, -g, -g], [g, g, -g], [-g, g, -g], [-g, -g, g], [g, -g, g], [g, g, g], [-g, g, g]] {
        let dn: Vec<[f64; 3]> = nat
            .iter()
            .map(|c| {
                let f = |k: usize| 1.0 + c[k] * p[k];
                [0.125 * c[0] * f(1) * f(2), 0.125 * c[1] * f(0) * f(2), 0.125 * c[2] * f(0) * f(1)]
            })
            .collect();
        let mut j = [[0.0; 3]; 3];
        for i in 0..8 {
            for r in 0..3 {
                for c in 0..3 {
                    j[r][c] += dn[i][r] * x[i][c];
                }
            }
        }
        let (ji, det) = crate::elements::common::inv3(j);
        if det <= 0.0 {
            return None;
        }
        vol += det;
        for i in 0..8 {
            for c in 0..3 {
                b[i][c] += (0..3).map(|r| ji[c][r] * dn[i][r]).sum::<f64>() * det;
            }
        }
    }
    Some((b, vol))
}

fn tet4_b(x: &[[f64; 3]]) -> Option<(Vec<[f64; 3]>, f64)> {
    let (a, bb, c) = (sub(x[1], x[0]), sub(x[2], x[0]), sub(x[3], x[0]));
    let v6 = dot(a, cross(bb, c));
    if v6 <= 0.0 {
        return None;
    }
    // ∫∇Nᵢ dV = (face opposite node i) · (outward normal) / 3 with sign
    let faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    let mut b = vec![[0.0; 3]; 4];
    for (i, f) in faces.iter().enumerate() {
        let n = cross(sub(x[f[1]], x[f[0]]), sub(x[f[2]], x[f[0]]));
        // n points out of the tetrahedron for the face opposite node i; ∇Nᵢ points inwards
        for c in 0..3 {
            b[i][c] = -n[c] / 6.0;
        }
    }
    Some((b, v6 / 6.0))
}

/// Planar coordinates (2D) for a planar element in 3D (shell, membrane) and
/// the basis vectors e₁, e₂ of the plane.
fn planar(x: &[[f64; 3]]) -> (Vec<[f64; 3]>, [f64; 3], [f64; 3]) {
    let n = if x.len() == 4 { cross(sub(x[2], x[0]), sub(x[3], x[1])) } else { cross(sub(x[1], x[0]), sub(x[2], x[0])) };
    let nl = norm(n);
    let e3 = [n[0] / nl, n[1] / nl, n[2] / nl];
    let d = sub(x[1], x[0]);
    let dl = norm(d);
    let e1 = [d[0] / dl, d[1] / dl, d[2] / dl];
    let e2 = cross(e3, e1);
    let xp = x
        .iter()
        .map(|p| {
            let r = sub(*p, x[0]);
            [dot(r, e1), dot(r, e2), 0.0]
        })
        .collect();
    (xp, e1, e2)
}

/// Planar coordinates for `family` (unchanged for planar continuum elements)
/// and a function that maps bᵢ back to global axes.
fn plane_of(family: Family, x: &[[f64; 3]]) -> (Vec<[f64; 3]>, [f64; 3], [f64; 3]) {
    if family == Family::Continuum2D {
        (x.to_vec(), [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    } else {
        planar(x)
    }
}

fn to_global(b: Vec<[f64; 3]>, e1: [f64; 3], e2: [f64; 3]) -> Vec<[f64; 3]> {
    b.into_iter().map(|v| std::array::from_fn(|c| v[0] * e1[c] + v[1] * e2[c])).collect()
}

fn quad4_b(x: &[[f64; 3]]) -> (Vec<[f64; 3]>, f64) {
    let mut b = vec![[0.0; 3]; 4];
    for i in 0..4 {
        let (n1, p1) = (x[(i + 1) % 4], x[(i + 3) % 4]);
        b[i] = [0.5 * (n1[1] - p1[1]), 0.5 * (p1[0] - n1[0]), 0.0];
    }
    let a = 0.5 * ((x[2][0] - x[0][0]) * (x[3][1] - x[1][1]) - (x[3][0] - x[1][0]) * (x[2][1] - x[0][1]));
    (b, a)
}

fn tri3_b(x: &[[f64; 3]]) -> (Vec<[f64; 3]>, f64) {
    let mut b = vec![[0.0; 3]; 3];
    for i in 0..3 {
        let (j, k) = (x[(i + 1) % 3], x[(i + 2) % 3]);
        b[i] = [0.5 * (j[1] - k[1]), 0.5 * (k[0] - j[0]), 0.0];
    }
    let a = 0.5 * ((x[1][0] - x[0][0]) * (x[2][1] - x[0][1]) - (x[2][0] - x[0][0]) * (x[1][1] - x[0][1]));
    (b, a)
}

/// Geometry for elements with a known characteristic length (see the table), in
/// the configuration `x`. `None` for other elements.
pub fn geometry(family: Family, name: &str, x: &[[f64; 3]]) -> Option<ElemGeom> {
    let nn = x.len();
    let len2 = |a: [f64; 3], b: [f64; 3]| {
        let d = sub(a, b);
        dot(d, d)
    };
    match family {
        Family::Continuum3D if nn == 8 => {
            let (b, vol) = hex8_b(x)?;
            let s: f64 = b.iter().map(|v| dot(*v, *v)).sum();
            Some(ElemGeom { lchar: vol / (2.0 * s).sqrt(), b, vol })
        }
        Family::Continuum3D if nn == 4 && name.starts_with("C3D4") => {
            let (b, vol) = tet4_b(x)?;
            let s: f64 = b.iter().map(|v| dot(*v, *v)).sum();
            Some(ElemGeom { lchar: vol / (2.0 * s).sqrt(), b, vol })
        }
        Family::Continuum2D | Family::Shell | Family::Membrane if nn == 4 => {
            let (xp, e1, e2) = plane_of(family, x);
            let (b, a) = quad4_b(&xp);
            if a <= 0.0 {
                return None;
            }
            let s: f64 = b.iter().map(|v| dot(*v, *v)).sum();
            Some(ElemGeom { lchar: a / s.sqrt(), b: to_global(b, e1, e2), vol: a })
        }
        Family::Continuum2D | Family::Shell | Family::Membrane if nn == 3 => {
            let (xp, e1, e2) = plane_of(family, x);
            let (b, a) = tri3_b(&xp);
            if a <= 0.0 {
                return None;
            }
            let sl = len2(xp[0], xp[1]) + len2(xp[1], xp[2]) + len2(xp[2], xp[0]);
            Some(ElemGeom { lchar: 4.0 * a / (3.0 * sl).sqrt(), b: to_global(b, e1, e2), vol: a })
        }
        Family::Truss => {
            let l = len2(x[0], x[1]).sqrt();
            Some(ElemGeom { lchar: TRUSS_LE * l, b: Vec::new(), vol: l })
        }
        _ => None,
    }
}

/// Dilatational wave speed of the material c_d = √(D₁₁/ρ) for the element's stress state.
pub fn wave_speed(ctx: &ElemCtx, family: Family, name: &str) -> Result<Option<f64>> {
    let Some(mat) = ctx.material else { return Ok(None) };
    if ctx.density <= 0.0 {
        return Ok(None);
    }
    let mode = match family {
        Family::Continuum3D => StressMode::ThreeD,
        Family::Continuum2D if name.starts_with("CPS") => StressMode::PlaneStress,
        Family::Continuum2D => StressMode::PlaneStrain,
        Family::Shell | Family::Membrane => StressMode::PlaneStress,
        Family::Truss => StressMode::Uniaxial,
        _ => return Ok(None),
    };
    if let (Family::Truss, Some((e, _))) = (family, mat.elastic_moduli()) {
        return Ok(Some((e / ctx.density).sqrt()));
    }
    let d = mat.tangent_linear(mode)?;
    let n = mode.ncomp();
    let d11 = (0..n.min(3)).map(|i| d[i * n + i]).fold(0.0f64, f64::max);
    Ok(Some((d11 / ctx.density).sqrt()))
}

/// Highest eigenvalue ω² of M⁻¹K for an element (diagonal mass), with power iteration.
pub fn element_omega2(k: &[f64], m: &[f64]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 0.0;
    }
    // symmetric form D^{-1/2} K D^{-1/2}
    let d: Vec<f64> = m.iter().map(|&x| if x > 0.0 { 1.0 / x.sqrt() } else { 0.0 }).collect();
    let a: Vec<f64> = (0..n * n).map(|ij| d[ij / n] * k[ij] * d[ij % n]).collect();
    let mut x: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let mut lam = 0.0;
    for _ in 0..500 {
        let y: Vec<f64> = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum()).collect();
        let ny = y.iter().map(|v| v * v).sum::<f64>().sqrt();
        if ny == 0.0 {
            return 0.0;
        }
        let l = x.iter().zip(&y).map(|(p, q)| p * q).sum::<f64>() / x.iter().map(|v| v * v).sum::<f64>();
        x = y.iter().map(|v| v / ny).collect();
        if (l - lam).abs() <= 1e-10 * l.abs() {
            return l;
        }
        lam = l;
    }
    lam
}

/// Undamped stability limit for element `e` in the configuration `x`:
/// (2/ωₑ, Lₑ, c_d). `mdiag` is the element's diagonal mass (for
/// the power iteration), `None` for elements without mass or stiffness.
pub fn element_limit(p: &Prepared, e: usize, ctx: &ElemCtx, x: &[[f64; 3]], mdiag: &[f64]) -> Result<Option<(f64, f64, f64)>> {
    let el = &p.model.elements[e];
    let fam = el.etype.family();
    let name = el.etype.name();
    if let (Some(g), Some(c)) = (geometry(fam, name, x), wave_speed(ctx, fam, name)?) {
        if g.lchar > 0.0 && c > 0.0 {
            return Ok(Some((g.lchar / c, g.lchar, c)));
        }
    }
    if mdiag.iter().all(|&v| v <= 0.0) {
        return Ok(None);
    }
    let k = el.etype.stiffness(ctx)?;
    if k.len() != mdiag.len() * mdiag.len() {
        return Ok(None);
    }
    let w = element_omega2(&k, mdiag).max(0.0).sqrt();
    if w <= 0.0 {
        return Ok(None);
    }
    Ok(Some((2.0 / w, 0.0, 0.0)))
}

/// Improved stability limit (`IMPROVED DT METHOD=YES`, default)
/// for shells and membranes: Lₑ = max(Lₑ,conv, k·L_exact), where L_exact = 2c/ω is
/// the exact highest eigenfrequency for the element's constant strain field with
/// diagonal mass and plane stress,
///
/// ```text
/// ω² = (n/ρ) · max(E′/2 [(s₁+s₂) + √((s₁−s₂)² + 4ν² s₁s₂)],  G (s₁+s₂))
/// ```
///
/// (n = number of nodes, s₁, s₂ = the eigenvalues of Σᵢ bᵢ⊗bᵢ / A², E′ = E/(1−ν²)),
/// and k² = 0.92 for quadrilaterals, 0.97 for triangles (empirical factors,
/// checked for squares, 1×2 and 1×3 rectangles and parallelograms with
/// ν = 0.3 and 0.45 as well as right and equilateral triangles).
pub fn improved_plane_stress(family: Family, x: &[[f64; 3]], nu: f64) -> Option<f64> {
    if !matches!(family, Family::Shell | Family::Membrane) || !(x.len() == 3 || x.len() == 4) {
        return None;
    }
    let (xp, _, _) = planar(x);
    let (b, a) = if x.len() == 4 { quad4_b(&xp) } else { tri3_b(&xp) };
    if a <= 0.0 {
        return None;
    }
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for v in &b {
        sxx += v[0] * v[0];
        syy += v[1] * v[1];
        sxy += v[0] * v[1];
    }
    let (sxx, syy, sxy) = (sxx / (a * a), syy / (a * a), sxy / (a * a));
    let m = 0.5 * (sxx + syy);
    let r = (0.25 * (sxx - syy).powi(2) + sxy * sxy).sqrt();
    let (s1, s2) = (m + r, m - r);
    // in units of E′: the normal part and the shear part G = E′(1 − ν)/2
    let normal = 0.5 * ((s1 + s2) + ((s1 - s2).powi(2) + 4.0 * nu * nu * s1 * s2).sqrt());
    let shear = 0.5 * (1.0 - nu) * (s1 + s2);
    let lam = normal.max(shear);
    if !(lam > 0.0) {
        return None;
    }
    let n = x.len() as f64;
    let k2 = if x.len() == 4 { 0.92 } else { 0.97 };
    Some((k2 * 4.0 / (n * lam)).sqrt())
}

/// The damping factor √(1+ξ²) − ξ.
pub fn damping_factor(xi: f64) -> f64 {
    (1.0 + xi * xi).sqrt() - xi
}
