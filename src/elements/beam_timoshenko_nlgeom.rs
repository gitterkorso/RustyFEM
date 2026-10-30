//! Timoshenko beams (B21, B22, B31, B32) with NLGEOM=YES. The formulation
//! follows geometrically exact beam theory (Reissner 1972, Simo 1985,
//! Simo–Vu-Quoc 1986) in updated, incremental form.
//!
//! Kinematics at an integration point (Λ = [t_s, n₁, n₂] the section axes,
//! Λ₀ in the reference position, x′ = dx/dS w.r.t. the reference arc length):
//!
//! - stretch λ = |x′|, axis direction a = x′/λ;
//! - axial strain ε = ln λ + ½ τ² (I11 + I22)/A (logarithmic strain plus
//!   the mean elongation of the fibers under twist, the "helix effect");
//! - shear γ₁ = n₁·a, γ₂ = n₂·a (sine of the angle between section and axis);
//! - curvature and twist [τ, κ₁, κ₂] = K̃/λ, where K̃ is the section's relative
//!   rotation per reference length in its own axes.
//!
//! The rotations are incremental: with the nodes' rotation increments
//! Δφₐ = log(Rₐ Rₐ,ₙᵀ) (spatial) and Δφ(ξ) = Σ Nₐ Δφₐ,
//! Λ = exp([Δφ]) Λₙ and K̃ = K̃ₙ + Λₙᵀ T(Δφ)ᵀ Δφ′. Within an increment this corresponds to
//! linear interpolation of (incremental) rotation vectors (the formulation
//! is incremental, not total). Λ and K̃ per integration point are the element's
//! state.
//!
//! The section forces are s = C e with the linear element's section matrix C,
//! where the slenderness compensation of the shear stiffness is computed with the element's current
//! length. Internal forces are obtained with virtual strains in the current
//! configuration, integrated over the current length (w·J·λ):
//!
//! - δε = a·δx′ₛ (+ τ (I11 + I22)/A · t_s·δθ′ₛ/λ, the twist term per reference length),
//! - δγ₁ = n₁·δx′ₛ − n₂·δθ, δγ₂ = n₂·δx′ₛ + n₁·δθ,
//! - δ[τ, κ₁, κ₂] = Λᵀ δθ′ₛ,
//!
//! where ′ₛ is the derivative w.r.t. the current arc length and δθ spatial virtual rotations.
//! Linearized, this gives exactly the strains of the linear beam in
//! `beam_timoshenko.rs`. The force is not the gradient of a potential, so
//! the tangent ∂f/∂q (exact, with hyper-dual numbers w.r.t. nodal displacements and spins,
//! R ← exp([δw]) R) is not symmetric; it is symmetrized for the symmetric
//! solvers. This gives linear instead of quadratic convergence when the moments are
//! large.
//!
//! The geometric stiffness for *BUCKLE and *FREQUENCY with preload is the part
//! of the tangent in the reference position that is proportional to the section forces (all
//! section forces, not only the axial force).

use super::{gauss, shape, BeamGeom, TimoBeam};
use crate::elements::{ElemCtx, ElemResponse};
use crate::kinematics::hyperdual::{self as hd, Hd, T3, V3};
use crate::kinematics::tensor as t;
use crate::model::{fail, Result};

// ---------------------------------------------------------------------------
// Rotation helpers for hyper-dual numbers (also used by the shell NLGEOM)
// ---------------------------------------------------------------------------

/// Constant hyper-dual matrix.
pub fn cst_m<const N: usize>(m: &t::M3) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| Hd::cst(m[i][j])))
}

/// A B.
pub fn mm<const N: usize>(a: &T3<N>, b: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}

/// Aᵀ B.
pub fn mtm<const N: usize>(a: &T3<N>, b: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[0][i] * b[0][j] + a[1][i] * b[1][j] + a[2][i] * b[2][j]))
}

/// Aᵀ v.
pub fn mtv<const N: usize>(a: &T3<N>, v: &V3<N>) -> V3<N> {
    std::array::from_fn(|i| a[0][i] * v[0] + a[1][i] * v[1] + a[2][i] * v[2])
}

/// f(x) with given f, f′, f″ at x.
pub fn chain<const N: usize>(x: &Hd<N>, f: f64, d1: f64, d2: f64) -> Hd<N> {
    let mut r = Hd::cst(f);
    for i in 0..N {
        r.g[i] = d1 * x.g[i];
        for j in 0..N {
            r.h[i][j] = d1 * x.h[i][j] + d2 * x.g[i] * x.g[j];
        }
    }
    r
}

/// h(ν) = atan(√ν)/√ν with derivatives w.r.t. ν (series for small ν).
fn atan_coef(nu: f64) -> (f64, f64, f64) {
    if nu < 1e-2 {
        // h = Σ (−1)^k ν^k / (2k + 1)
        let (mut f, mut d1, mut d2) = (0.0, 0.0, 0.0);
        let mut p = 1.0;
        for k in 0..12usize {
            let c = if k % 2 == 0 { 1.0 } else { -1.0 } / (2 * k + 1) as f64;
            f += c * p;
            if k >= 1 {
                d1 += c * k as f64 * if k >= 2 { nu.powi(k as i32 - 1) } else { 1.0 };
            }
            if k >= 2 {
                d2 += c * (k * (k - 1)) as f64 * if k >= 3 { nu.powi(k as i32 - 2) } else { 1.0 };
            }
            p *= nu;
        }
        return (f, d1, d2);
    }
    let r = nu.sqrt();
    let at = r.atan();
    let h = at / r;
    let q = 1.0 + r * r;
    let hr = 1.0 / (r * q) - at / (r * r);
    let hrr = -(1.0 + 3.0 * r * r) / (r * r * q * q) - (1.0 / (r * r * q) - 2.0 * at / (r * r * r));
    (h, hr / (2.0 * r), hrr / (4.0 * nu) - hr / (4.0 * r * r * r))
}

/// The rotation vector of a rotation matrix (angle < π), via the quaternion:
/// s = ½ axial(R − Rᵀ) = sin θ·a, w² = cos²(θ/2) = (1 + tr R)/4 and
/// φ = h(ν)·s/w² with ν = tan²(θ/2) = |s|²/(4w⁴), h(ν) = atan(√ν)/√ν.
pub fn log_so3<const N: usize>(r: &T3<N>) -> Option<V3<N>> {
    let w2 = (r[0][0] + r[1][1] + r[2][2] + 1.0) * 0.25;
    if !(w2.v > 1e-8) {
        return None;
    }
    let s: V3<N> = [(r[2][1] - r[1][2]) * 0.5, (r[0][2] - r[2][0]) * 0.5, (r[1][0] - r[0][1]) * 0.5];
    let iw2 = w2.recip();
    let nu = hd::dot3(&s, &s) * iw2 * iw2 * 0.25;
    let (f, d1, d2) = atan_coef(nu.v);
    let h = chain(&nu, f, d1, d2) * iw2;
    Some(s.map(|v| v * h))
}

/// Spin variable: the rotation matrix exp([w]) R₀.
pub fn spun<const N: usize>(w: &V3<N>, r0: &t::M3) -> T3<N> {
    mm(&hd::exp_so3(w), &cst_m(r0))
}

// ---------------------------------------------------------------------------
// The beam
// ---------------------------------------------------------------------------

/// State per integration point: the section axes Λ (9, row-wise) and
/// the accumulated relative rotation per reference length K̃ (3, material).
const IP_STATE: usize = 12;

/// The current position and the position at the start of the increment.
struct Current {
    /// Nodal positions relative to node 1 (current).
    dx: Vec<[f64; 3]>,
    /// The nodes' rotations now and at the start of the increment.
    rot: Vec<t::M3>,
    rot_n: Vec<t::M3>,
    /// Λₙ and K̃ₙ per integration point at the start of the increment.
    lam_n: Vec<t::M3>,
    k_n: Vec<[f64; 3]>,
}

/// Strains and virtual strains at an integration point.
struct Point<const N: usize> {
    /// [ε, κ₁, κ₂, τ, γ₁, γ₂]
    e: [Hd<N>; 6],
    /// Virtual strains per nodal variable: de[r][6a + c] (c: 0–2 δx, 3–5 δθ).
    de: [Vec<Hd<N>>; 6],
    /// Weight w·J·λ (integration over the current length).
    wl: Hd<N>,
    /// New state (Λ, K̃).
    lam: t::M3,
    k: [f64; 3],
}

/// ln for hyper-dual numbers.
fn ln_hd<const N: usize>(x: Hd<N>) -> Hd<N> {
    let v = x.v;
    chain(&x, v.ln(), 1.0 / v, -1.0 / (v * v))
}

/// Strains at the integration points. `var(a, c)` gives the perturbation for node a,
/// component c (0–2 displacement, 3–5 spin).
fn points<const N: usize>(b: &TimoBeam, g: &BeamGeom, cur: &Current, var: &dyn Fn(usize, usize) -> Hd<N>) -> Option<Vec<Point<N>>> {
    let nn = b.nn;
    let xs: Vec<V3<N>> = (0..nn).map(|a| std::array::from_fn(|i| var(a, i) + cur.dx[a][i])).collect();
    // the nodes' rotation increments (spatial): Δφₐ = log(exp([w]) Rₐ Rₐₙᵀ)
    let mut dphi: Vec<V3<N>> = Vec::with_capacity(nn);
    for a in 0..nn {
        let r = spun(&[var(a, 3), var(a, 4), var(a, 5)], &cur.rot[a]);
        dphi.push(log_so3(&mm(&r, &cst_m(&t::transpose(&cur.rot_n[a]))))?);
    }
    let ipa = if g.sec.a > 0.0 { (g.sec.i11 + g.sec.i22) / g.sec.a } else { 0.0 };
    let z = Hd::<N>::cst(0.0);
    let mut out = Vec::new();
    for (q, &(xi, w)) in gauss(b.n_gauss()).iter().enumerate() {
        let (n, dn) = shape(nn, xi);
        let j = g.jac(nn, xi);
        let mut xd: V3<N> = [z; 3];
        let mut psi: V3<N> = [z; 3];
        let mut dpsi: V3<N> = [z; 3];
        for a in 0..nn {
            for k in 0..3 {
                xd[k] = xd[k] + xs[a][k] * (dn[a] / j);
                psi[k] = psi[k] + dphi[a][k] * n[a];
                dpsi[k] = dpsi[k] + dphi[a][k] * (dn[a] / j);
            }
        }
        // stretch λ = |x′| and the axis direction a
        let lam = hd::dot3(&xd, &xd).sqrt();
        let il = lam.recip();
        let av: V3<N> = xd.map(|v| v * il);
        // the section axes Λ = exp([Δφ]) Λₙ (columns t_s, n₁, n₂)
        let lnh = cst_m::<N>(&cur.lam_n[q]);
        let lm = mm(&hd::exp_so3(&psi), &lnh);
        let col = |k: usize| -> V3<N> { [lm[0][k], lm[1][k], lm[2][k]] };
        let (ts, n1, n2) = (col(0), col(1), col(2));
        // relative rotation per reference length: K̃ = K̃ₙ + Λₙᵀ T(Δφ)ᵀ Δφ′, curvature K̃/λ
        let kr = mtv(&hd::tangent_so3(&psi), &dpsi);
        let kt: V3<N> = std::array::from_fn(|i| mtv(&lnh, &kr)[i] + cur.k_n[q][i]);
        let kk = kt.map(|v| v * il);
        // axial strain: ln λ plus the mean elongation of the fibers under twist,
        // ½ τ² (I11 + I22)/A (the helix effect)
        // the shear strain is computed from the reference position (for midside nodes that do not lie
        // exactly on the line between the end nodes, e.g. rounded coordinates)
        // (the same computation as for the current position, so that the strains are exactly zero in the reference position)
        let x0d: [f64; 3] = std::array::from_fn(|k| (0..nn).fold(0.0, |acc, a| acc + (g.x[a][k] - g.x[0][k]) * (dn[a] / j)));
        let l0d = (x0d[0] * x0d[0] + x0d[1] * x0d[1] + x0d[2] * x0d[2]).sqrt();
        let g0 = [1, 2].map(|c| (0..3).map(|k| g.axes[c][k] * (x0d[k] * (1.0 / l0d))).sum::<f64>());
        let e = [ln_hd(lam) - l0d.ln() + kk[0] * kk[0] * (0.5 * ipa), kk[1], kk[2], kk[0], hd::dot3(&n1, &av) - g0[0], hd::dot3(&n2, &av) - g0[1]];
        // virtual strains (updated configuration):
        // δε = a·δx′ₛ, δγ₁ = n₁·δx′ₛ − n₂·δθ, δγ₂ = n₂·δx′ₛ + n₁·δθ, δκ = Λᵀ δθ′ₛ
        let nd = 6 * nn;
        let mut de: [Vec<Hd<N>>; 6] = std::array::from_fn(|_| vec![z; nd]);
        for a in 0..nn {
            let dxs = il * (dn[a] / j);
            for k in 0..3 {
                de[0][6 * a + k] = av[k] * dxs;
                de[4][6 * a + k] = n1[k] * dxs;
                de[5][6 * a + k] = n2[k] * dxs;
                de[4][6 * a + 3 + k] = -n2[k] * n[a];
                de[5][6 * a + 3 + k] = n1[k] * n[a];
                de[3][6 * a + 3 + k] = ts[k] * dxs;
                de[1][6 * a + 3 + k] = n1[k] * dxs;
                de[2][6 * a + 3 + k] = n2[k] * dxs;
                // the virtual work of the twist term is computed per reference length (the factor 1/λ)
                de[0][6 * a + 3 + k] = kk[0] * ts[k] * dxs * il * ipa;
            }
        }
        out.push(Point {
            e,
            de,
            wl: lam * (w * j),
            lam: std::array::from_fn(|i| std::array::from_fn(|k| lm[i][k].v)),
            k: kt.map(|v| v.v),
        });
    }
    Some(out)
}

/// T(φ) in f64 (dR = [T dφ]× R).
fn so3_tangent(p: [f64; 3]) -> t::M3 {
    let q: V3<1> = p.map(Hd::<1>::cst);
    let m = hd::tangent_so3(&q);
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j].v))
}

/// The position now (displacement `u`) and at the start of the increment (`u − du`,
/// the state `state`). Without a state (the first NLGEOM increment)
/// Λₙ and K̃ₙ are computed from the rotation vectors at the start of the increment as if they had been reached
/// in a single increment from the reference position.
fn current(b: &TimoBeam, g: &BeamGeom, u: &[f64], du: &[f64], state: &[f64]) -> Current {
    let nn = b.nn;
    let u6 = b.expand_vector(u);
    let du6 = if du.len() == u.len() { b.expand_vector(du) } else { vec![0.0; u6.len()] };
    let p0: [f64; 3] = std::array::from_fn(|i| g.x[0][i] + u6[i]);
    let phi = |a: usize| -> [f64; 3] { [u6[6 * a + 3], u6[6 * a + 4], u6[6 * a + 5]] };
    let phi_n: Vec<[f64; 3]> = (0..nn).map(|a| std::array::from_fn(|i| u6[6 * a + 3 + i] - du6[6 * a + 3 + i])).collect();
    let ng = b.n_gauss();
    let (lam_n, k_n) = if state.len() == IP_STATE * ng {
        (
            (0..ng).map(|q| std::array::from_fn(|i| std::array::from_fn(|k| state[IP_STATE * q + 3 * i + k]))).collect(),
            (0..ng).map(|q| std::array::from_fn(|i| state[IP_STATE * q + 9 + i])).collect(),
        )
    } else {
        let lam0: t::M3 = std::array::from_fn(|i| std::array::from_fn(|k| g.axes[k][i]));
        let mut ls = Vec::new();
        let mut ks = Vec::new();
        for &(xi, _) in gauss(ng) {
            let (n, dn) = shape(nn, xi);
            let j = g.jac(nn, xi);
            let pm: [f64; 3] = std::array::from_fn(|i| (0..nn).map(|a| n[a] * phi_n[a][i]).sum());
            let dp: [f64; 3] = std::array::from_fn(|i| (0..nn).map(|a| dn[a] / j * phi_n[a][i]).sum());
            ls.push(t::mul(&t::exp_so3(pm), &lam0));
            ks.push(t::mv(&t::transpose(&lam0), t::mv(&t::transpose(&so3_tangent(pm)), dp)));
        }
        (ls, ks)
    };
    Current {
        dx: (0..nn).map(|a| std::array::from_fn(|i| g.x[a][i] + u6[6 * a + i] - p0[i])).collect(),
        rot: (0..nn).map(|a| t::exp_so3(phi(a))).collect(),
        rot_n: phi_n.iter().map(|p| t::exp_so3(*p)).collect(),
        lam_n,
        k_n,
    }
}

/// The section matrix with the slenderness compensation of the shear stiffness computed with
/// the element's current length (distance between the end nodes).
/// The section matrix with the current length for the displacement `u` (output).
pub fn current_c_for(b: &TimoBeam, g: &BeamGeom, u: &[f64]) -> [[f64; 6]; 6] {
    let cur = current(b, g, u, &[], &[]);
    current_c(b, g, &cur)
}

fn current_c(b: &TimoBeam, g: &BeamGeom, cur: &Current) -> [[f64; 6]; 6] {
    let d = cur.dx[b.nn - 1];
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let gc = BeamGeom { axes: g.axes, l: if l > 0.0 { l } else { g.l }, x: g.x.clone(), sec: g.sec.clone() };
    b.section_matrix(&gc)
}

fn too_large(ctx: &ElemCtx, b: &TimoBeam) -> crate::model::FemError {
    crate::model::FemError(format!("element {} ({}): too large rotation increment (≥ 180°)", ctx.label, b.name))
}

#[allow(clippy::too_many_arguments)]
fn response_n<const N: usize>(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    let g = b.geom(ctx)?;
    // area struct_nonlin: with section points (nonlinear material) the
    // section point state comes first, then Λ and K̃ per integration point
    let fib = if super::plastic::is_plastic(ctx) { Some(super::plastic::Fibers::new(ctx, b.name)?) } else { None };
    let ng = b.n_gauss();
    let plen = fib.as_ref().map_or(0, |f| f.ip_len() * ng);
    let pinit;
    let (pso, so) = match &fib {
        Some(_) if so.len() >= plen => (&so[..plen], &so[plen..]),
        Some(fb) => {
            pinit = fb.init_state(ng);
            (&pinit[..], &so[..0])
        }
        None => (&so[..0], so),
    };
    let mut psn = pso.to_vec();
    let cur = current(b, &g, u, du, so);
    let c = current_c(b, &g, &cur);
    let planar = b.dim == 2;
    // variable index for node a, component k (None = no degree of freedom)
    let idx = |a: usize, k: usize| -> Option<usize> {
        if planar {
            match k {
                0 | 1 => Some(3 * a + k),
                5 => Some(3 * a + 2),
                _ => None,
            }
        } else {
            Some(6 * a + k)
        }
    };
    let var = |a: usize, k: usize| -> Hd<N> { idx(a, k).map_or(Hd::cst(0.0), |i| Hd::var(0.0, i)) };
    let pts = points::<N>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    // f = Σ w J λ δeᵀ C e
    let mut f = vec![Hd::<N>::cst(0.0); N];
    for (ip, p) in pts.iter().enumerate() {
        let s: [Hd<N>; 6] = match &fib {
            // section points: s(e) with the tangent Cₜ, s = s₀ + Cₜ (e − e₀) for the derivatives
            Some(fb) => {
                let ev: [f64; 6] = std::array::from_fn(|r| p.e[r].v);
                let len = fb.ip_len();
                let (nsh, _) = shape(b.nn, gauss(b.n_gauss())[ip].0);
                let (s0, ct) = fb.response(ctx, &c, &ev, &nsh, &pso[ip * len..(ip + 1) * len], &mut psn[ip * len..(ip + 1) * len])?;
                std::array::from_fn(|r| (0..6).fold(Hd::cst(s0[r]), |acc, q| if ct[r][q] != 0.0 { acc + (p.e[q] - ev[q]) * ct[r][q] } else { acc }))
            }
            None => std::array::from_fn(|r| (0..6).fold(Hd::cst(0.0), |acc, q| if c[r][q] != 0.0 { acc + p.e[q] * c[r][q] } else { acc })),
        };
        for a in 0..b.nn {
            for k in 0..6 {
                let Some(i) = idx(a, k) else { continue };
                let mut v = Hd::cst(0.0);
                for r in 0..6 {
                    let d = p.de[r][6 * a + k];
                    if d.v != 0.0 || d.g.iter().any(|x| *x != 0.0) {
                        v = v + d * s[r];
                    }
                }
                f[i] = f[i] + v * p.wl;
            }
        }
    }
    let fint: Vec<f64> = f.iter().map(|x| x.v).collect();
    if fint.iter().any(|v| !v.is_finite()) {
        return fail(format!("element {} ({}): invalid deformed geometry", ctx.label, b.name));
    }
    sn.clear();
    sn.extend_from_slice(&psn);
    for p in &pts {
        for row in &p.lam {
            sn.extend_from_slice(row);
        }
        sn.extend_from_slice(&p.k);
    }
    // tangent: ∂f/∂q (not symmetric; symmetrized for the symmetric solver)
    let k = tangent.then(|| {
        let mut k = vec![0.0; N * N];
        for i in 0..N {
            for j in 0..N {
                k[i * N + j] = 0.5 * (f[i].g[j] + f[j].g[i]);
            }
        }
        k
    });
    Ok(ElemResponse { fint, k })
}

/// Internal force and tangent with NLGEOM.
pub fn response(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    match (b.dim, b.nn) {
        (2, 2) => response_n::<6>(b, ctx, u, du, so, sn, tangent),
        (2, _) => response_n::<9>(b, ctx, u, du, so, sn, tangent),
        (_, 2) => response_n::<12>(b, ctx, u, du, so, sn, tangent),
        _ => response_n::<18>(b, ctx, u, du, so, sn, tangent),
    }
}

fn geometric_n<const N: usize>(b: &TimoBeam, ctx: &ElemCtx, forces: &[[f64; 6]]) -> Result<Vec<f64>> {
    let g = b.geom(ctx)?;
    let zero = vec![0.0; N];
    let cur = current(b, &g, &zero, &[], &[]);
    let planar = b.dim == 2;
    let idx = |a: usize, k: usize| -> Option<usize> {
        if planar {
            match k {
                0 | 1 => Some(3 * a + k),
                5 => Some(3 * a + 2),
                _ => None,
            }
        } else {
            Some(6 * a + k)
        }
    };
    let var = |a: usize, k: usize| -> Hd<N> { idx(a, k).map_or(Hd::cst(0.0), |i| Hd::var(0.0, i)) };
    let pts = points::<N>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    // f(q) = Σ w J λ δe(q)ᵀ s with fixed section forces s: ∂f/∂q in the reference position is
    // the part of the NLGEOM tangent that is proportional to the stresses
    let mut f = vec![Hd::<N>::cst(0.0); N];
    for (p, s) in pts.iter().zip(forces) {
        for a in 0..b.nn {
            for k in 0..6 {
                let Some(i) = idx(a, k) else { continue };
                let mut v = Hd::cst(0.0);
                for r in 0..6 {
                    if s[r] != 0.0 {
                        v = v + p.de[r][6 * a + k] * s[r];
                    }
                }
                f[i] = f[i] + v * p.wl;
            }
        }
    }
    let mut k = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            k[i * N + j] = 0.5 * (f[i].g[j] + f[j].g[i]);
        }
    }
    Ok(k)
}

/// Geometric stiffness (stress stiffness) for *BUCKLE and *FREQUENCY with
/// preload: the stress-dependent part of the NLGEOM tangent in the reference position,
/// with the section forces `forces` (per integration point, ordering as
/// `section_forces`) from a linear solution.
pub fn geometric(b: &TimoBeam, ctx: &ElemCtx, forces: &[[f64; 6]]) -> Result<Vec<f64>> {
    match (b.dim, b.nn) {
        (2, 2) => geometric_n::<6>(b, ctx, forces),
        (2, _) => geometric_n::<9>(b, ctx, forces),
        (_, 2) => geometric_n::<12>(b, ctx, forces),
        _ => geometric_n::<18>(b, ctx, forces),
    }
}

/// Section forces [N, M1, M2, T, V1, V2] per integration point with NLGEOM
/// (the current section axes) for the accepted state `state`.
pub fn section_forces(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<[f64; 6]>> {
    let g = b.geom(ctx)?;
    let ng = b.n_gauss();
    let state = if state.len() > IP_STATE * ng { &state[state.len() - IP_STATE * ng..] } else { state };
    let cur = current(b, &g, u, &[], state);
    let c = current_c(b, &g, &cur);
    let var = |_: usize, _: usize| Hd::<1>::cst(0.0);
    let pts = points::<1>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    Ok(pts.iter().map(|p| std::array::from_fn(|r| (0..6).map(|s| c[r][s] * p.e[s].v).sum())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_inverts_exp() {
        for p in [[0.0, 0.0, 0.0], [1e-5, -2e-6, 3e-6], [0.3, -0.2, 0.5], [1.2, 0.9, -1.4], [0.0, 0.0, 2.9]] {
            let r = t::exp_so3(p);
            let l = log_so3::<1>(&cst_m(&r)).unwrap();
            for i in 0..3 {
                assert!((l[i].v - p[i]).abs() < 1e-12, "{p:?}: {}", l[i].v);
            }
        }
    }

    #[test]
    fn log_derivatives() {
        // d/dw log(exp(w) R) at w = 0 against differences, and the second derivative
        let p = [0.4, -0.7, 0.9];
        let r0 = t::exp_so3(p);
        let w: V3<3> = std::array::from_fn(|i| Hd::var(0.0, i));
        let l = log_so3(&spun(&w, &r0)).unwrap();
        let f = |w: [f64; 3]| t::log_so3(&t::mul(&t::exp_so3(w), &r0));
        let h = 1e-5;
        for k in 0..3 {
            let mut wp = [0.0; 3];
            wp[k] = h;
            let mut wm = [0.0; 3];
            wm[k] = -h;
            let (fp, fm) = (f(wp), f(wm));
            for i in 0..3 {
                assert!(((fp[i] - fm[i]) / (2.0 * h) - l[i].g[k]).abs() < 1e-8);
            }
            for m in 0..3 {
                let d = |a: f64, b: f64| {
                    let mut ww = [0.0; 3];
                    ww[k] += a;
                    ww[m] += b;
                    f(ww)
                };
                let (pp, pm, mp, mq) = (d(h, h), d(h, -h), d(-h, h), d(-h, -h));
                for i in 0..3 {
                    let fd = (pp[i] - pm[i] - mp[i] + mq[i]) / (4.0 * h * h);
                    assert!((fd - l[i].h[k][m]).abs() < 1e-4, "{i} {k} {m}: {fd} {}", l[i].h[k][m]);
                }
            }
        }
    }

    #[test]
    fn atan_coef_continuous() {
        let (a, b) = (atan_coef(1e-2 - 1e-13), atan_coef(1e-2 + 1e-13));
        assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-10 && (a.2 - b.2).abs() < 1e-8);
    }
}
