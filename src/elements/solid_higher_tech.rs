//! Element techniques for `Solid`: hourglass stabilization (C3D8R, CPS4R, CPE4R,
//! CAX4R), incompatible modes (C3D8I, CPS4I, CPE4I), mass matrices and
//! integration points for loads.

use super::shapes::Shape;
use super::{ips_for, RuleKind, Solid};
use crate::elements::api::{ElemCtx, ElemResponse, IpField, MassKind};
use crate::elements::common::{self, Ip};
use crate::materials::{MaterialModel, StressMode};
use crate::model::{fail, Result};

// ---------------------------------------------------------------------------
// Small dense matrix operations
// ---------------------------------------------------------------------------

/// Solves A X = B (A n×n row-wise, B n×m row-wise) with Gaussian elimination and
/// partial pivoting. Returns X.
pub fn solve_dense(a: &[f64], b: &[f64], n: usize, m: usize) -> Result<Vec<f64>> {
    let mut a = a.to_vec();
    let mut x = b.to_vec();
    for k in 0..n {
        let mut p = k;
        for i in k + 1..n {
            if a[i * n + k].abs() > a[p * n + k].abs() {
                p = i;
            }
        }
        if a[p * n + k].abs() < 1e-300 {
            return fail("singular element matrix (incompatible modes)");
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            for j in 0..m {
                x.swap(k * m + j, p * m + j);
            }
        }
        let piv = a[k * n + k];
        for i in k + 1..n {
            let f = a[i * n + k] / piv;
            if f != 0.0 {
                for j in k..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
                for j in 0..m {
                    x[i * m + j] -= f * x[k * m + j];
                }
            }
        }
    }
    for k in (0..n).rev() {
        for j in 0..m {
            let mut s = x[k * m + j];
            for i in k + 1..n {
                s -= a[k * n + i] * x[i * m + j];
            }
            x[k * m + j] = s / a[k * n + k];
        }
    }
    Ok(x)
}

// ---------------------------------------------------------------------------
// Mass and loads
// ---------------------------------------------------------------------------

/// Integration points for body forces and volume weights.
pub fn load_ips(el: &Solid, ctx: &ElemCtx) -> Result<Vec<Ip>> {
    el.ips_exact(ctx)
}

/// Rule for consistent mass. Second-order elements are integrated with
/// 3×3(×3) Gauss for C3D20/CPS8,
/// 6 points for 6-node triangles, 6×3 for C3D15 and 15 points (Keast) for
/// C3D10. First-order elements are integrated exactly.
pub fn mass_rule(el: &Solid) -> RuleKind {
    match el.shape {
        Shape::Hex20 | Shape::Quad8 => RuleKind::Gauss(3),
        Shape::Tri6 => RuleKind::Tri(6),
        Shape::Wedge15 => RuleKind::Wedge(6, 3),
        Shape::Tet10 => RuleKind::Tet(15),
        Shape::Hex8 | Shape::Quad4 => RuleKind::Gauss(2),
        Shape::Tri3 => RuleKind::Tri(3),
        Shape::Tet4 => RuleKind::Tet(4),
        Shape::Wedge6 => RuleKind::Wedge(3, 2),
        Shape::Line2 | Shape::Line3 => RuleKind::Gauss(3),
    }
}

/// Mass matrix. Consistent: ∫ρNᵀN dV. Diagonal: see `lumped_mass`.
///
/// The default is diagonal mass for first-order
/// elements and consistent mass for second-order elements.
pub fn mass(el: &Solid, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
    let t = el.thickness(ctx)?;
    let ips = ips_for(el.shape, el.mode, &ctx.x, &mass_rule(el).points(el.pdim()), t)?;
    let nd = el.ndim();
    match kind {
        MassKind::Consistent => Ok(common::mass(&ips, ctx.density, nd, false)),
        MassKind::Lumped => Ok(lumped_mass(el, &ips, ctx.density)),
    }
}

/// Diagonal mass. First-order elements: row sum. Second-order
/// elements: diagonal scaling (HRZ) so that the total mass is preserved and no node
/// gets negative mass.
pub fn lumped_mass(el: &Solid, ips: &[Ip], rho: f64) -> Vec<f64> {
    let nd = el.ndim();
    let nn = ips[0].n.len();
    let n = nn * nd;
    let mut diag = vec![0.0; nn];
    let mut total = 0.0;
    let mut rowsum = vec![0.0; nn];
    for ip in ips {
        let s: f64 = ip.n.iter().sum();
        for a in 0..nn {
            diag[a] += rho * ip.n[a] * ip.n[a] * ip.w;
            rowsum[a] += rho * ip.n[a] * s * ip.w;
        }
        total += rho * ip.w;
    }
    let quadratic = matches!(el.shape, Shape::Tri6 | Shape::Quad8 | Shape::Tet10 | Shape::Wedge15 | Shape::Hex20);
    let vals: Vec<f64> = if quadratic {
        let ds: f64 = diag.iter().sum();
        diag.iter().map(|d| d * total / ds).collect()
    } else {
        rowsum
    };
    let mut m = vec![0.0; n * n];
    for a in 0..nn {
        for d in 0..nd {
            let i = a * nd + d;
            m[i * n + i] = vals[a];
        }
    }
    m
}

// ---------------------------------------------------------------------------
// Incompatible modes
// ---------------------------------------------------------------------------

/// Number of incompatible degrees of freedom: C3D8I 13 (9 gradient modes + 4
/// volumetric modes), CPS4I/CPE4I 4, CAX4I and CPEG4I 5 (4 gradient modes + 1 volumetric mode).
pub fn n_incompatible(el: &Solid) -> usize {
    match (el.shape, el.mode) {
        (Shape::Hex8, _) => 13,
        (_, StressMode::Axisymmetric) => 5,
        _ if planar_volume_mode(el) => 5,
        _ => 4,
    }
}

/// Does the element have the volumetric mode ξη in ε11+ε22 (CAX4I and CPEG4I; CPS4I/CPE4I do
/// not)?
fn planar_volume_mode(el: &Solid) -> bool {
    el.mode == StressMode::Axisymmetric || el.name.starts_with("CPEG")
}

/// The inverse and determinant of the Jacobian at the element center.
fn center_jacobian(el: &Solid, x: &[[f64; 3]]) -> Result<([[f64; 3]; 3], f64)> {
    let (_, dn) = super::shapes::shape(el.shape, [0.0; 3]);
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

/// Enhanced B matrices G (ncomp × nα) per integration point (enhanced
/// strain, Simo–Rifai):
///
/// - gradient modes: φₖ = 1 − ξₖ² for each displacement component, with
///   the gradient taken with the Jacobian J₀ at the element center,
/// - volumetric modes (3D: ξη, ηζ, ζξ, ξηζ in ε11+ε22+ε33; axisymmetry: ξη in
///   εrr+εzz; generalized plane strain: ξη in ε11+ε22),
///
/// all scaled by det J₀ / det J (axisymmetry: r₀ det J₀ / (r det J)) so that
/// ∫G dV = 0 and the element passes the patch test. The mode set:
/// C3D8I 13 modes, CPS4I/CPE4I 4, CAX4I 5.
fn incompatible_bmats(el: &Solid, ctx: &ElemCtx, ips: &[Ip]) -> Result<Vec<Vec<f64>>> {
    let (ji0, det0) = center_jacobian(el, &ctx.x)?;
    let dim = el.pdim();
    let rule = el.rule.points(dim);
    let t = el.thickness(ctx)?;
    let axi = el.mode == StressMode::Axisymmetric;
    let r0 = if axi {
        let (n0, _) = super::shapes::shape(el.shape, [0.0; 3]);
        n0.iter().zip(&ctx.x).map(|(ni, xi)| ni * xi[0]).sum()
    } else {
        1.0
    };
    let ns = el.mode.ncomp();
    let na = n_incompatible(el);
    let mut out = Vec::with_capacity(ips.len());
    for (ip, &(p, gw)) in ips.iter().zip(&rule) {
        let (factor, sc) = match el.mode {
            StressMode::ThreeD => (1.0, 1.0),
            StressMode::Axisymmetric => {
                let r = super::radius(ip, &ctx.x);
                (2.0 * std::f64::consts::PI * r, r0 / r)
            }
            _ => (t, 1.0),
        };
        let det = ip.w / (gw * factor);
        let s = sc * det0 / det;
        let mut g = Vec::with_capacity(dim);
        for k in 0..dim {
            let mut dphi = [0.0; 3];
            dphi[k] = -2.0 * p[k];
            let mut d = [0.0; 3];
            for c in 0..dim {
                d[c] = (0..dim).map(|r| ji0[c][r] * dphi[r]).sum::<f64>() * s;
            }
            g.push(d);
        }
        let bg = match el.mode {
            // the hoop strain gets no contribution from the gradient modes
            StressMode::Axisymmetric => crate::elements::axisym::bmat_axi(&vec![0.0; g.len()], &g, 1.0),
            _ => common::bmat(el.mode, &g),
        };
        let ng = bg.len() / ns;
        let (psis, nv): (Vec<f64>, usize) = match (dim, el.mode) {
            (3, _) => (vec![p[0] * p[1], p[1] * p[2], p[2] * p[0], p[0] * p[1] * p[2]], 3),
            (_, StressMode::Axisymmetric) => (vec![p[0] * p[1]], 2),
            _ if planar_volume_mode(el) => (vec![p[0] * p[1]], 2),
            _ => (Vec::new(), 0),
        };
        let mut b = vec![0.0; ns * na];
        for r in 0..ns {
            for c in 0..ng {
                b[r * na + c] = bg[r * ng + c];
            }
            if r < nv {
                for (e, psi) in psis.iter().enumerate() {
                    b[r * na + ng + e] = psi * s;
                }
            }
        }
        out.push(b);
    }
    Ok(out)
}

/// [B | G] row-wise (ncomp × (nd + nα)).
fn extend(b: &[f64], g: &[f64], ns: usize) -> Vec<f64> {
    let nd = b.len() / ns;
    let na = g.len() / ns;
    let mut e = Vec::with_capacity(ns * (nd + na));
    for r in 0..ns {
        e.extend_from_slice(&b[r * nd..(r + 1) * nd]);
        e.extend_from_slice(&g[r * na..(r + 1) * na]);
    }
    e
}

/// Appends columns to each B matrix with the values `vals[q]` in
/// the ε33 row (row 2) and zero in the other rows (point q). Used for
/// the reference node degrees of freedom in generalized plane strain
/// (`solid_higher_gps.rs`).
pub fn append_eps33(bs: &mut [Vec<f64>], ns: usize, vals: &[[f64; 3]]) {
    for (b, v) in bs.iter_mut().zip(vals) {
        let nd = b.len() / ns;
        let mut e = Vec::with_capacity(ns * (nd + 3));
        for r in 0..ns {
            e.extend_from_slice(&b[r * nd..(r + 1) * nd]);
            e.extend((0..3).map(|k| if r == 2 { v[k] } else { 0.0 }));
        }
        *b = e;
    }
}

/// The ε33 rows for `append_eps33` at the element's integration points.
pub type Eps33Rows<'a> = &'a dyn Fn(&[Ip]) -> Vec<[f64; 3]>;

fn extended_bmats(el: &Solid, ctx: &ElemCtx, eps33: Option<Eps33Rows>) -> Result<(Vec<Ip>, Vec<Vec<f64>>, usize, usize)> {
    let ips = el.ips(ctx)?;
    let mut bs = el.bmats(&ctx.x, &ips);
    if let Some(f) = eps33 {
        append_eps33(&mut bs, el.mode.ncomp(), &f(&ips));
    }
    let gs = incompatible_bmats(el, ctx, &ips)?;
    let ns = el.mode.ncomp();
    let nd = bs[0].len() / ns;
    let na = gs[0].len() / ns;
    let be = bs.iter().zip(&gs).map(|(b, g)| extend(b, g, ns)).collect();
    Ok((ips, be, nd, na))
}

/// Static condensation: K = Kuu − Kuα Kαα⁻¹ Kαu and Kαα⁻¹ Kαu.
fn condense(k: &[f64], nd: usize, na: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let n = nd + na;
    let kaa: Vec<f64> = (0..na).flat_map(|i| (0..na).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + nd + j]).collect();
    let kau: Vec<f64> = (0..na).flat_map(|i| (0..nd).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + j]).collect();
    let x = solve_dense(&kaa, &kau, na, nd)?; // Kαα⁻¹ Kαu
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
    Ok((kc, x))
}

pub fn incompatible_stiffness(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel) -> Result<Vec<f64>> {
    incompatible_stiffness_x(el, ctx, mat, None)
}

/// Like `incompatible_stiffness` with extra ε33 columns (see `append_eps33`).
pub fn incompatible_stiffness_x(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, eps33: Option<Eps33Rows>) -> Result<Vec<f64>> {
    let (ips, be, nd, na) = extended_bmats(el, ctx, eps33)?;
    let k = common::stiffness(el.mode, &ips, &be, mat)?;
    Ok(condense(&k, nd, na)?.0)
}

/// Internal force with local Newton iteration for the incompatible degrees of freedom
/// (exact after one iteration for linear materials). α is stored last in the state.
#[allow(clippy::too_many_arguments)]
pub fn incompatible_response(
    el: &Solid,
    ctx: &ElemCtx,
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    so: &[f64],
    sn: &mut Vec<f64>,
    tangent: bool,
) -> Result<ElemResponse> {
    incompatible_response_x(el, ctx, mat, u, du, so, sn, tangent, None)
}

/// Like `incompatible_response` with extra ε33 columns (see `append_eps33`).
#[allow(clippy::too_many_arguments)]
pub fn incompatible_response_x(
    el: &Solid,
    ctx: &ElemCtx,
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    so: &[f64],
    sn: &mut Vec<f64>,
    tangent: bool,
    eps33: Option<Eps33Rows>,
) -> Result<ElemResponse> {
    let (ips, be, nd, na) = extended_bmats(el, ctx, eps33)?;
    let nip = ips.len() * common::ip_stride(el.mode, mat);
    let (so_ip, a_old) = if so.len() == nip + na { (&so[..nip], so[nip..].to_vec()) } else { (&[][..], vec![0.0; na]) };
    let mut alpha = a_old.clone();
    let mut ext_u = u.to_vec();
    ext_u.extend_from_slice(&alpha);
    let mut ext_du = du.to_vec();
    ext_du.extend(alpha.iter().zip(&a_old).map(|(a, b)| a - b));
    let n = nd + na;
    let mut last = None;
    for it in 0..30 {
        for a in 0..na {
            ext_u[nd + a] = alpha[a];
            ext_du[nd + a] = alpha[a] - a_old[a];
        }
        let r = common::response_with_temperatures(ctx, el.mode, &ips, &be, mat, &ext_u, &ext_du, so_ip, sn, true, Some(&el.temperatures(ctx)))?;
        let k = r.k.as_ref().unwrap();
        let ra = &r.fint[nd..];
        let scale = r.fint[..nd].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rnorm = ra.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let done = rnorm <= 1e-12 * scale.max(1e-300) || (mat.is_linear() && it == 1);
        if done || it == 29 {
            last = Some(r);
            break;
        }
        let kaa: Vec<f64> = (0..na).flat_map(|i| (0..na).map(move |j| (i, j))).map(|(i, j)| k[(nd + i) * n + nd + j]).collect();
        let d = solve_dense(&kaa, ra, na, 1)?;
        for a in 0..na {
            alpha[a] -= d[a];
        }
    }
    let r = last.unwrap();
    sn.extend_from_slice(&alpha);
    let fint = r.fint[..nd].to_vec();
    let k = if tangent { Some(condense(r.k.as_ref().unwrap(), nd, na)?.0) } else { None };
    Ok(ElemResponse { fint, k })
}

/// Output with the strain contribution of the incompatible modes. For linear materials
/// α is recomputed from the displacement (independent of the stored state).
pub fn incompatible_output(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
    incompatible_output_x(el, ctx, mat, u, state, None)
}

/// Like `incompatible_output` with extra ε33 columns (see `append_eps33`).
pub fn incompatible_output_x(
    el: &Solid,
    ctx: &ElemCtx,
    mat: &dyn MaterialModel,
    u: &[f64],
    state: &[f64],
    eps33: Option<Eps33Rows>,
) -> Result<Vec<IpField>> {
    let (ips, be, nd, na) = extended_bmats(el, ctx, eps33)?;
    let nip = ips.len() * common::ip_stride(el.mode, mat);
    let alpha: Vec<f64> = if mat.is_linear() {
        let k = common::stiffness(el.mode, &ips, &be, mat)?;
        let (_, x) = condense(&k, nd, na)?;
        (0..na).map(|a| -(0..nd).map(|j| x[a * nd + j] * u[j]).sum::<f64>()).collect()
    } else if state.len() == nip + na {
        state[nip..].to_vec()
    } else {
        vec![0.0; na]
    };
    let mut ext_u = u.to_vec();
    ext_u.extend_from_slice(&alpha);
    let st = if state.len() >= nip { &state[..nip] } else { state };
    if mat.is_linear() {
        // the stress directly from the strain (the state may be from the start of the step)
        let d = mat.tangent_linear(el.mode)?;
        let ns = el.mode.ncomp();
        let mut s = Vec::with_capacity(nip);
        for b in &be {
            let e = common::strain(b, &ext_u);
            s.extend((0..ns).map(|r| (0..ns).map(|c| d[r * ns + c] * e[c]).sum::<f64>()));
        }
        return Ok(common::output(el.mode, &be, mat, &ext_u, &s));
    }
    Ok(common::output(el.mode, &be, mat, &ext_u, st))
}

// ---------------------------------------------------------------------------
// Hourglass stabilization
// ---------------------------------------------------------------------------

/// Kinematics for one-point reduced-integration elements: the B matrix is
/// the volume average B̄ = (1/V)∫B dV ("uniform strain", Flanagan–Belytschko),
/// integrated with 2×2(×2) points. The point gets the weight V and the position at
/// the volume centroid (COORD, also for CAX4R).
pub fn uniform_strain(el: &Solid, ctx: &ElemCtx) -> Result<(Vec<Ip>, Vec<Vec<f64>>)> {
    let t = el.thickness(ctx)?;
    let full = ips_for(el.shape, el.mode, &ctx.x, &RuleKind::Gauss(2).points(el.pdim()), t)?;
    let vol: f64 = full.iter().map(|ip| ip.w).sum();
    let nn = full[0].n.len();
    let mut n = vec![0.0; nn];
    let mut dndx = vec![[0.0; 3]; nn];
    let mut b: Vec<f64> = Vec::new();
    for ip in &full {
        let bq = super::bmat_any(el.mode, ip, &ctx.x);
        if b.is_empty() {
            b = vec![0.0; bq.len()];
        }
        for (x, y) in b.iter_mut().zip(&bq) {
            *x += y * ip.w / vol;
        }
        for i in 0..nn {
            n[i] += ip.n[i] * ip.w / vol;
            for c in 0..3 {
                dndx[i][c] += ip.dndx[i][c] * ip.w / vol;
            }
        }
    }
    Ok((vec![Ip { n, dndx, w: vol }], vec![b]))
}

/// Shear modulus for the hourglass stiffness (for the NLGEOM variant in `solid_higher_nlgeom.rs`).
pub fn shear_modulus_of(el: &Solid, mat: &dyn MaterialModel) -> Result<f64> {
    shear_modulus(el, mat)
}

/// Shear modulus from the material's elastic tangent (for the hourglass stiffness).
fn shear_modulus(el: &Solid, mat: &dyn MaterialModel) -> Result<f64> {
    if let Some((e, nu)) = mat.elastic_moduli() {
        return Ok(e / (2.0 * (1.0 + nu)));
    }
    let d = mat.tangent_linear(el.mode)?;
    let ns = el.mode.ncomp();
    Ok(d[(ns - 1) * ns + ns - 1])
}

/// Hourglass stiffness according to the default method (the stiffness method).
///
/// The hourglass vectors γₐ = hₐ − (hₐ·x) b̄ (Flanagan–Belytschko) with
/// hₐ = ξη (planar) and ξη, ηζ, ζξ, ξηζ (3D), respectively, at the corner nodes, and
/// K = Σₐ γₐγₐᵀ ⊗ S where S depends on the shear modulus G, the volume V and b̄:
/// - planar/axisymmetric: S = 0.005/16 · G V Σᵢ|b̄ᵢ|² · I
/// - 3D: S = diag(s₁, s₂, s₃), s_d = 0.0011785 · G V Σᵢ b̄ᵢd²
///
/// The constants correspond to the default value of the hourglass stiffness. The stiffness is linear
/// and is based on the elastic shear modulus also for nonlinear materials.
pub fn hourglass_matrix(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel) -> Result<Vec<f64>> {
    let (ips, _) = uniform_strain(el, ctx)?;
    let ip = &ips[0];
    let vol = ip.w;
    let nd = el.ndim();
    let nn = el.shape.n_nodes();
    let n = nn * nd;
    let nat = super::shapes::node_coords(el.shape);
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
    let g = shear_modulus(el, mat)?;
    let sbb: Vec<f64> = (0..nd).map(|c| ip.dndx.iter().map(|b| b[c] * b[c]).sum()).collect();
    let s: Vec<f64> = if nd == 3 {
        sbb.iter().map(|v| 1.1785e-3 * g * vol * v).collect()
    } else {
        let tot: f64 = sbb.iter().sum();
        vec![0.005 / 16.0 * g * vol * tot; nd]
    };
    let mut k = vec![0.0; n * n];
    for gm in &gam {
        for i in 0..nn {
            for j in 0..nn {
                let v = gm[i] * gm[j];
                for d in 0..nd {
                    k[(i * nd + d) * n + j * nd + d] += s[d] * v;
                }
            }
        }
    }
    Ok(k)
}

pub fn hourglass_stiffness(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel) -> Result<Vec<f64>> {
    let (ips, bs) = uniform_strain(el, ctx)?;
    let mut k = common::stiffness(el.mode, &ips, &bs, mat)?;
    let kh = hourglass_matrix(el, ctx, mat)?;
    for (a, b) in k.iter_mut().zip(&kh) {
        *a += b;
    }
    Ok(k)
}

#[allow(clippy::too_many_arguments)]
pub fn hourglass_response(
    el: &Solid,
    ctx: &ElemCtx,
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    so: &[f64],
    sn: &mut Vec<f64>,
    tangent: bool,
) -> Result<ElemResponse> {
    let (ips, bs) = uniform_strain(el, ctx)?;
    let mut r = common::response(ctx, el.mode, &ips, &bs, mat, u, du, so, sn, tangent)?;
    let kh = hourglass_matrix(el, ctx, mat)?;
    let fh = common::matvec(&kh, u);
    for (a, b) in r.fint.iter_mut().zip(&fh) {
        *a += b;
    }
    if let Some(k) = r.k.as_mut() {
        for (a, b) in k.iter_mut().zip(&kh) {
            *a += b;
        }
    }
    Ok(r)
}

pub fn hourglass_output(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
    let (_, bs) = uniform_strain(el, ctx)?;
    Ok(common::output(el.mode, &bs, mat, u, state))
}

impl Solid {
    /// Number of degrees of freedom in the element.
    pub fn n_nodes_dofs(&self) -> usize {
        self.shape.n_nodes() * self.ndim()
    }
}

// ---------------------------------------------------------------------------
// Geometric stiffness
// ---------------------------------------------------------------------------

/// Geometric stiffness (stress stiffness) Kσ = ∫ ∇Nᵢ·σ·∇Nⱼ dV · I from
/// the stresses in the element state (small strains, reference geometry).
/// Axisymmetry also gets the hoop term σθθ NᵢNⱼ / r² for u_r. Reduced-
/// integration elements use B̄ and the stress at the point (no
/// hourglass part); for incompatible modes only the nodal displacements are used.
pub fn geometric_stiffness(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, state: &[f64]) -> Result<Vec<f64>> {
    let ips = match el.tech {
        super::Tech::Hourglass => uniform_strain(el, ctx)?.0,
        _ => el.ips(ctx)?,
    };
    let stride = common::ip_stride(el.mode, mat);
    let nd = el.ndim();
    let nn = el.shape.n_nodes();
    let n = nn * nd;
    let mut k = vec![0.0; n * n];
    for (q, ip) in ips.iter().enumerate() {
        let s = if state.len() >= (q + 1) * stride { &state[q * stride..q * stride + el.mode.ncomp()] } else { continue };
        // σ as 3×3 (the in-plane part for plane elements) and hoop stress
        let (sig, hoop): ([[f64; 3]; 3], f64) = match el.mode {
            StressMode::ThreeD => ([[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]], 0.0),
            StressMode::PlaneStress => ([[s[0], s[2], 0.0], [s[2], s[1], 0.0], [0.0; 3]], 0.0),
            StressMode::Axisymmetric => ([[s[0], s[3], 0.0], [s[3], s[1], 0.0], [0.0; 3]], s[2]),
            _ => ([[s[0], s[3], 0.0], [s[3], s[1], 0.0], [0.0; 3]], 0.0),
        };
        let r = if el.mode == StressMode::Axisymmetric { super::radius(ip, &ctx.x) } else { 1.0 };
        for i in 0..nn {
            let gi = ip.dndx[i];
            let sg: [f64; 3] = [0, 1, 2].map(|a| (0..3).map(|b| sig[a][b] * gi[b]).sum());
            for j in 0..nn {
                let v = common::dot(sg, ip.dndx[j]) * ip.w;
                for d in 0..nd {
                    k[(i * nd + d) * n + j * nd + d] += v;
                }
                if hoop != 0.0 {
                    k[(i * nd) * n + j * nd] += hoop * ip.n[i] * ip.n[j] / (r * r) * ip.w;
                }
            }
        }
    }
    Ok(k)
}
