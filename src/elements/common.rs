//! Shared building blocks for elements: small matrix operations and a
//! continuum core (B matrices with B-bar, internal force, stiffness, mass, loads
//! and output) shared by all continuum families.
//!
//! New continuum elements (C3D20, C3D10, CAX4, ...) only need to compute their
//! integration points (`Ip`) and can then use the functions here.

use super::api::{ElemCtx, ElemResponse, IpField};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

/// 1/√3, Gauss point for the 2-point rule.
pub const G2: f64 = 0.577_350_269_189_625_8;

pub const DISTORTED: &str = "an element is distorted or inverted (negative Jacobian) – check the node ordering";

/// Integration point: shape functions, derivatives in global coordinates and
/// weight (det(J) · Gauss weight · thickness).
#[derive(Debug, Clone)]
pub struct Ip {
    pub n: Vec<f64>,
    pub dndx: Vec<[f64; 3]>,
    pub w: f64,
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
pub fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn centroid(x: &[[f64; 3]]) -> [f64; 3] {
    let n = x.len() as f64;
    [0, 1, 2].map(|a| x.iter().map(|p| p[a]).sum::<f64>() / n)
}

/// Inverse and determinant of a 3×3 matrix.
pub fn inv3(m: [[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let id = 1.0 / det;
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id,
        ],
    ];
    (inv, det)
}

/// y = A x for a row-wise n×n matrix.
pub fn matvec(a: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| a[i * n..(i + 1) * n].iter().zip(x).map(|(p, q)| p * q).sum()).collect()
}

/// Tᵀ k T for row-wise n×n matrices.
pub fn congruence(k: &[f64], t: &[f64], n: usize) -> Vec<f64> {
    let mut kt = vec![0.0; n * n];
    for i in 0..n {
        for m in 0..n {
            let kim = k[i * n + m];
            if kim != 0.0 {
                for j in 0..n {
                    kt[i * n + j] += kim * t[m * n + j];
                }
            }
        }
    }
    let mut out = vec![0.0; n * n];
    for m in 0..n {
        for i in 0..n {
            let tmi = t[m * n + i];
            if tmi != 0.0 {
                for j in 0..n {
                    out[i * n + j] += tmi * kt[m * n + j];
                }
            }
        }
    }
    out
}

/// Block diagonal transformation matrix with `blocks` copies of the 3×3 matrix `r`.
pub fn block_diag(r: &[[f64; 3]; 3], blocks: usize) -> Vec<f64> {
    let n = 3 * blocks;
    let mut t = vec![0.0; n * n];
    for b in 0..blocks {
        for i in 0..3 {
            for j in 0..3 {
                t[(3 * b + i) * n + 3 * b + j] = r[i][j];
            }
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Continuum core
// ---------------------------------------------------------------------------

/// How the volumetric strain is treated (selectively reduced integration for CPE4/C3D8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bbar {
    None,
    /// Plane strain: ε11+ε22 is averaged, ε33 = 0 is kept.
    Planar,
    /// 3D: ε11+ε22+ε33 is averaged over the element.
    Volumetric,
}

/// Number of displacement components per node for a stress state.
pub fn ndim(mode: StressMode) -> usize {
    match mode {
        StressMode::ThreeD => 3,
        _ => 2,
    }
}

/// B matrix (row-wise ncomp × ndof) for small strains.
pub fn bmat(mode: StressMode, dndx: &[[f64; 3]]) -> Vec<f64> {
    let nn = dndx.len();
    let ns = mode.ncomp();
    let nd = ndim(mode) * nn;
    let mut b = vec![0.0; ns * nd];
    match mode {
        StressMode::PlaneStress | StressMode::PlaneStrain => {
            let shear = ns - 1;
            for (i, d) in dndx.iter().enumerate() {
                b[2 * i] = d[0];
                b[nd + 2 * i + 1] = d[1];
                b[shear * nd + 2 * i] = d[1];
                b[shear * nd + 2 * i + 1] = d[0];
            }
        }
        StressMode::ThreeD => {
            for (i, d) in dndx.iter().enumerate() {
                let c = 3 * i;
                b[c] = d[0];
                b[nd + c + 1] = d[1];
                b[2 * nd + c + 2] = d[2];
                b[3 * nd + c] = d[1];
                b[3 * nd + c + 1] = d[0];
                b[4 * nd + c] = d[2];
                b[4 * nd + c + 2] = d[0];
                b[5 * nd + c + 1] = d[2];
                b[5 * nd + c + 2] = d[1];
            }
        }
        _ => panic!("internal: bmat for {mode:?} is missing (axisymmetry is added in elements/axisym.rs)"),
    }
    b
}

/// B matrices per integration point, with B-bar if `bbar` specifies it.
pub fn bmats(mode: StressMode, ips: &[Ip], bbar: Bbar) -> Vec<Vec<f64>> {
    let mut bs: Vec<Vec<f64>> = ips.iter().map(|ip| bmat(mode, &ip.dndx)).collect();
    if bbar == Bbar::None || bs.is_empty() {
        return bs;
    }
    let nd = bs[0].len() / mode.ncomp();
    let vol = |b: &[f64], c: usize| b[c] + b[nd + c] + b[2 * nd + c];
    let wsum: f64 = ips.iter().map(|ip| ip.w).sum();
    let avg: Vec<f64> = (0..nd).map(|c| ips.iter().zip(&bs).map(|(ip, b)| ip.w * vol(b, c)).sum::<f64>() / wsum).collect();
    for b in bs.iter_mut() {
        for c in 0..nd {
            if bbar == Bbar::Planar {
                let corr = (avg[c] - vol(b, c)) / 2.0;
                b[c] += corr;
                b[nd + c] += corr;
            } else {
                let corr = (avg[c] - vol(b, c)) / 3.0;
                for r in 0..3 {
                    b[r * nd + c] += corr;
                }
            }
        }
    }
    bs
}

/// ε = B u
pub fn strain(b: &[f64], u: &[f64]) -> Vec<f64> {
    let nd = u.len();
    let ns = b.len() / nd;
    (0..ns).map(|r| b[r * nd..(r + 1) * nd].iter().zip(u).map(|(p, q)| p * q).sum()).collect()
}

/// State layout per integration point: [stress (ncomp), material state].
pub fn ip_stride(mode: StressMode, mat: &dyn MaterialModel) -> usize {
    mode.ncomp() + mat.n_state()
}

pub fn init_state(n_ips: usize, mode: StressMode, mat: &dyn MaterialModel) -> Vec<f64> {
    let stride = ip_stride(mode, mat);
    let mut s = vec![0.0; n_ips * stride];
    for ip in 0..n_ips {
        mat.init_state(&mut s[ip * stride + mode.ncomp()..(ip + 1) * stride]);
    }
    s
}

/// Linear stiffness like `stiffness`, but with the element's material orientation
/// (*ORIENTATION) if the section has one. Elements should call this one.
pub fn stiffness_ctx(ctx: &ElemCtx, mode: StressMode, ips: &[Ip], bs: &[Vec<f64>], mat: &dyn MaterialModel) -> Result<Vec<f64>> {
    let Some(ori) = ctx.orientation else { return stiffness(mode, ips, bs, mat) };
    let d = mat.tangent_linear(mode)?;
    let ns = mode.ncomp();
    let nd = bs[0].len() / ns;
    let mut ke = vec![0.0; nd * nd];
    for (ip, b) in ips.iter().zip(bs) {
        let t = ori.strain_transform_mode(ip_point(ctx, ip), mode);
        let dg = congruence(&d, &t, ns);
        add_btdb(&mut ke, b, &dg, ns, nd, ip.w);
    }
    Ok(ke)
}

/// The position of the integration point in the reference configuration.
pub fn ip_point(ctx: &ElemCtx, ip: &Ip) -> [f64; 3] {
    let mut p = [0.0; 3];
    for (ni, x) in ip.n.iter().zip(&ctx.x) {
        for d in 0..3 {
            p[d] += ni * x[d];
        }
    }
    p
}

/// Linear stiffness ∫Bᵀ D B dV with the material's elastic tangent.
pub fn stiffness(mode: StressMode, ips: &[Ip], bs: &[Vec<f64>], mat: &dyn MaterialModel) -> Result<Vec<f64>> {
    let d = mat.tangent_linear(mode)?;
    let ns = mode.ncomp();
    let nd = bs[0].len() / ns;
    let mut ke = vec![0.0; nd * nd];
    for (ip, b) in ips.iter().zip(bs) {
        add_btdb(&mut ke, b, &d, ns, nd, ip.w);
    }
    Ok(ke)
}

fn add_btdb(ke: &mut [f64], b: &[f64], d: &[f64], ns: usize, nd: usize, w: f64) {
    // db = D B (ns × nd)
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

/// Internal force ∫Bᵀσ dV and tangent ∫Bᵀ C B dV with stress update at each
/// integration point (small strains).
#[allow(clippy::too_many_arguments)]
pub fn response(
    ctx: &ElemCtx,
    mode: StressMode,
    ips: &[Ip],
    bs: &[Vec<f64>],
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
) -> Result<ElemResponse> {
    response_with_temperatures(ctx, mode, ips, bs, mat, u, du, state_old, state_new, tangent, None)
}

/// As `response`, with element-specific temperatures at the integration points.
#[allow(clippy::too_many_arguments)]
pub fn response_with_temperatures(
    ctx: &ElemCtx,
    mode: StressMode,
    ips: &[Ip],
    bs: &[Vec<f64>],
    mat: &dyn MaterialModel,
    u: &[f64],
    du: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
    temperatures: Option<&[(f64, f64)]>,
) -> Result<ElemResponse> {
    let ns = mode.ncomp();
    let nd = u.len();
    let stride = ip_stride(mode, mat);
    let old_owned;
    let state_old = if state_old.len() == ips.len() * stride {
        state_old
    } else {
        old_owned = init_state(ips.len(), mode, mat);
        &old_owned[..]
    };
    state_new.clear();
    state_new.extend_from_slice(state_old);
    let mut fint = vec![0.0; nd];
    let mut k = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
    let length = char_length(&ctx.x, mode, ips);
    for (q, (ip, b)) in ips.iter().zip(bs).enumerate() {
        let mut eps = strain(b, u);
        let mut deps = strain(b, du);
        // local material orientation: strain to local axes, stress and tangent back
        let tr = ctx.orientation.map(|o| o.strain_transform_mode(ip_point(ctx, ip), mode));
        if let Some(t) = &tr {
            eps = matvec(t, &eps);
            deps = matvec(t, &deps);
        }
        let (temperature, dtemp) = temperatures.map_or_else(|| ctx.temperature_at(&ip.n), |ts| ts[q]);
        crate::user::context::set_ip(q + 1, ip_point(ctx, ip), length, None, ip.w);
        let base = q * stride;
        let (sold, mold) = state_old[base..base + stride].split_at(ns);
        let mut mnew = mold.to_vec();
        let inp = MatInput {
            mode,
            strain: &eps,
            dstrain: &deps,
            stress_old: sold,
            time: ctx.time,
            dt: ctx.dt,
            temperature,
            dtemp,
            defgrad: None,
        };
        let mut out = mat.update_with_length(&inp, length, mold, &mut mnew)?;
        // the state stores the stress in local axes (as it is reported)
        state_new[base..base + ns].copy_from_slice(&out.stress);
        if let Some(t) = &tr {
            out.stress = (0..ns).map(|j| (0..ns).map(|i| t[i * ns + j] * out.stress[i]).sum()).collect();
            if tangent {
                out.tangent = congruence(&out.tangent, t, ns);
            }
        }
        for c in 0..nd {
            let mut s = 0.0;
            for r in 0..ns {
                s += b[r * nd + c] * out.stress[r];
            }
            fint[c] += s * ip.w;
        }
        if tangent {
            add_btdb(&mut k, b, &out.tangent, ns, nd, ip.w);
        }
        state_new[base + ns..base + stride].copy_from_slice(&mnew);
    }
    Ok(ElemResponse { fint, k: tangent.then_some(k) })
}

/// The element's characteristic length (for softening materials, Hillerborg):
/// the cube root of the volume for solids, the square root of the area in
/// the plane (the r–z plane for axisymmetry) for plane elements, half that length
/// for second-order elements.
pub fn char_length(x: &[[f64; 3]], mode: StressMode, ips: &[Ip]) -> f64 {
    let n = x.len();
    match mode {
        StressMode::ThreeD => {
            let v: f64 = ips.iter().map(|ip| ip.w).sum();
            let l = v.abs().cbrt();
            if matches!(n, 10 | 15 | 20 | 27) {
                0.5 * l
            } else {
                l
            }
        }
        StressMode::Uniaxial => {
            let d = sub(x[n - 1], x[0]);
            norm(d)
        }
        _ => {
            // the corner nodes come first in the node order
            let nc = if matches!(n, 3 | 6) { 3 } else { 4.min(n) };
            let mut a = 0.0;
            for i in 0..nc {
                let (p, q) = (x[i], x[(i + 1) % nc]);
                a += p[0] * q[1] - q[0] * p[1];
            }
            let l = (0.5 * a).abs().sqrt();
            if n > nc {
                0.5 * l
            } else {
                l
            }
        }
    }
}

/// Consistent or row-summed mass matrix ∫ρ NᵀN dV.
pub fn mass(ips: &[Ip], rho: f64, ndim: usize, lumped: bool) -> Vec<f64> {
    let nn = ips[0].n.len();
    let nd = nn * ndim;
    let mut m = vec![0.0; nd * nd];
    for ip in ips {
        for a in 0..nn {
            for b in 0..nn {
                let v = rho * ip.n[a] * ip.n[b] * ip.w;
                for d in 0..ndim {
                    m[(a * ndim + d) * nd + b * ndim + d] += v;
                }
            }
        }
    }
    if lumped {
        for i in 0..nd {
            let s: f64 = m[i * nd..(i + 1) * nd].iter().sum();
            m[i * nd..(i + 1) * nd].iter_mut().for_each(|v| *v = 0.0);
            m[i * nd + i] = s;
        }
    }
    m
}

/// ∫Nᵢ dV per node.
pub fn volume_weights(ips: &[Ip]) -> Vec<f64> {
    let nn = ips[0].n.len();
    let mut w = vec![0.0; nn];
    for ip in ips {
        for i in 0..nn {
            w[i] += ip.n[i] * ip.w;
        }
    }
    w
}

/// Equivalent nodal forces for the body force `b` (force per volume): ∫Nᵢ b dV.
pub fn body_force(ips: &[Ip], b: [f64; 3], ndim: usize) -> Vec<f64> {
    let w = volume_weights(ips);
    let mut f = vec![0.0; w.len() * ndim];
    for (i, wi) in w.iter().enumerate() {
        for d in 0..ndim {
            f[i * ndim + d] = b[d] * wi;
        }
    }
    f
}

/// Body force for the *DLOAD types BX, BY, BZ and GRAV (force per volume).
pub fn body_load_vector(kind: &str, magnitude: f64, extra: &[f64], density: f64, elem: &str) -> Result<Option<[f64; 3]>> {
    Ok(Some(match kind {
        "BX" => [magnitude, 0.0, 0.0],
        "BY" => [0.0, magnitude, 0.0],
        "BZ" => [0.0, 0.0, magnitude],
        "GRAV" => {
            let d = [extra.first().copied().unwrap_or(0.0), extra.get(1).copied().unwrap_or(0.0), extra.get(2).copied().unwrap_or(0.0)];
            let l = norm(d);
            if l == 0.0 {
                return fail(format!("the GRAV load on {elem} has no direction"));
            }
            if density == 0.0 {
                return fail(format!("the GRAV load on {elem} requires *DENSITY in the material"));
            }
            scale(d, magnitude * density / l)
        }
        _ => return Ok(None),
    }))
}

/// Equivalent nodal forces for a normal pressure `p` (positive into the material)
/// on an edge (2 nodes, thickness `t`), a triangle (3) or a bilinear
/// quadrilateral (4). The sign of the normal is chosen so that it points away from `inside`.
pub fn face_pressure(xf: &[[f64; 3]], inside: [f64; 3], p: f64, t: f64) -> Result<Vec<[f64; 3]>> {
    let fc = centroid(xf);
    let out = |n: [f64; 3]| if dot(sub(fc, inside), n) < 0.0 { -1.0 } else { 1.0 };
    match xf.len() {
        2 => {
            let d = sub(xf[1], xf[0]);
            let l = norm(d);
            let mut n = [d[1] / l, -d[0] / l, 0.0];
            n = scale(n, out(n));
            let w = l * t / 2.0;
            Ok(vec![scale(n, -p * w); 2])
        }
        3 => {
            let av = cross(sub(xf[1], xf[0]), sub(xf[2], xf[0]));
            let s = out(av);
            Ok(vec![scale(av, -p * s / 6.0); 3])
        }
        4 => {
            let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let mut f = vec![[0.0; 3]; 4];
            let mut total = [0.0; 3];
            let mut contrib = vec![[0.0; 3]; 4];
            for &(xi, eta) in &[(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)] {
                let mut dxi = [0.0; 3];
                let mut deta = [0.0; 3];
                let mut nv = [0.0; 4];
                for i in 0..4 {
                    nv[i] = 0.25 * (1.0 + xi * nat[i][0]) * (1.0 + eta * nat[i][1]);
                    let a = 0.25 * nat[i][0] * (1.0 + eta * nat[i][1]);
                    let b = 0.25 * nat[i][1] * (1.0 + xi * nat[i][0]);
                    for d in 0..3 {
                        dxi[d] += a * xf[i][d];
                        deta[d] += b * xf[i][d];
                    }
                }
                let av = cross(dxi, deta);
                for d in 0..3 {
                    total[d] += av[d];
                }
                for i in 0..4 {
                    for d in 0..3 {
                        contrib[i][d] += nv[i] * av[d];
                    }
                }
            }
            let s = out(total);
            for i in 0..4 {
                f[i] = scale(contrib[i], -p * s);
            }
            Ok(f)
        }
        n => fail(format!("pressure on faces with {n} nodes is not supported yet")),
    }
}

/// Pressure on face `face` (*DLOAD P1, P2, ...) as an element vector.
pub fn pressure_load(
    x: &[[f64; 3]],
    faces: &[&[usize]],
    kind: &str,
    p: f64,
    t: f64,
    ndim: usize,
    elem: &str,
) -> Result<Option<Vec<f64>>> {
    let Some(num) = kind.strip_prefix('P').and_then(|s| s.parse::<usize>().ok()) else {
        return Ok(None);
    };
    if num == 0 || num > faces.len() {
        return fail(format!("{elem} has only the faces P1–P{}, not {kind}", faces.len()));
    }
    let face = faces[num - 1];
    let xf: Vec<[f64; 3]> = face.iter().map(|&i| x[i]).collect();
    let fn_ = face_pressure(&xf, centroid(x), p, t)?;
    let mut f = vec![0.0; x.len() * ndim];
    for (k, &i) in face.iter().enumerate() {
        for d in 0..ndim {
            f[i * ndim + d] += fn_[k][d];
        }
    }
    Ok(Some(f))
}

/// Field names for stress and strain in component order.
pub fn stress_labels(mode: StressMode) -> (&'static [&'static str], &'static [&'static str]) {
    match mode {
        StressMode::ThreeD => (&["S11", "S22", "S33", "S12", "S13", "S23"], &["E11", "E22", "E33", "E12", "E13", "E23"]),
        StressMode::PlaneStrain | StressMode::Axisymmetric | StressMode::PlaneStress => {
            (&["S11", "S22", "S33", "S12"], &["E11", "E22", "E33", "E12"])
        }
        StressMode::Uniaxial => (&["S11"], &["E11"]),
    }
}

/// Output S, E and the material's own fields (e.g. PEEQ) per integration point.
///
/// Plane stress is written with S33 = 0 and E33 = 0 (the thickness strain
/// is not reported in E for CPS elements).
pub fn output(mode: StressMode, bs: &[Vec<f64>], mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Vec<IpField> {
    output_impl(None, mode, bs, mat, u, state)
}

/// Like `output`, but the strain E is reported in the element's local
/// material orientation (*ORIENTATION). Elements should call this one.
pub fn output_ctx(ctx: &ElemCtx, mode: StressMode, ips: &[Ip], bs: &[Vec<f64>], mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Vec<IpField> {
    let tr: Option<Vec<Vec<f64>>> = ctx.orientation.map(|o| ips.iter().map(|ip| o.strain_transform_mode(ip_point(ctx, ip), mode)).collect());
    output_impl(tr.as_deref(), mode, bs, mat, u, state)
}

fn output_impl(tr: Option<&[Vec<f64>]>, mode: StressMode, bs: &[Vec<f64>], mat: &dyn MaterialModel, u: &[f64], state: &[f64]) -> Vec<IpField> {
    let ns = mode.ncomp();
    let stride = ip_stride(mode, mat);
    let (sl, el) = stress_labels(mode);
    let mut s = IpField::new("S", sl);
    let mut e = IpField::new("E", el);
    let groups = mat.output_fields(mode);
    let mut extra: Vec<IpField> = groups.iter().map(|(n, l)| IpField::new(n, l)).collect();
    for (q, b) in bs.iter().enumerate() {
        let mut eps = strain(b, u);
        if let Some(t) = tr.and_then(|t| t.get(q)) {
            eps = matvec(t, &eps);
        }
        let st = if state.len() >= (q + 1) * stride { &state[q * stride..(q + 1) * stride] } else { &[][..] };
        let sig: Vec<f64> = if st.len() >= ns { st[..ns].to_vec() } else { vec![0.0; ns] };
        match mode {
            StressMode::PlaneStress => {
                s.points.push((q + 1, 0, vec![sig[0], sig[1], 0.0, sig[2]]));
                e.points.push((q + 1, 0, vec![eps[0], eps[1], 0.0, eps[2]]));
            }
            _ => {
                s.points.push((q + 1, 0, sig));
                e.points.push((q + 1, 0, eps));
            }
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
    out
}
