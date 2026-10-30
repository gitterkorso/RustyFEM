//! Euler–Bernoulli-balkar (B23, B33) vid NLGEOM=YES, formulerade för implicit
//! analys:
//!
//! - Balkaxelns läge interpoleras med kubiska Hermitepolynom i nodlägena och
//!   nodtangenterna dx/dS = (dl/dS) · C(φ) · T, där T är elementets riktning i
//!   referensläget och C(φ) nodens rotation. Sträckningen dl/dS i varje nod är
//!   en inre variabel i elementet (inte delad med grannelementen, så att
//!   axiell töjning inte låses i knutpunkter) och elimineras statiskt.
//! - Generaliserade töjningar (rak balk, X″ = 0): Greens töjning
//!   ε = ½(x′·x′ − 1), krökningsändringarna K₂ = n₁·x″, K₁ = −n₂·x″ och
//!   vridningen e₁ = n₂ · dñ₁/dS, där ñ₁ = C(ψ) N₁ med rotationsvektorn ψ
//!   linjärt interpolerad mellan noderna, n₁ = ñ₁:s del vinkelrät mot
//!   tangenten (normerad) och n₂ = t × n₁.
//! - Linjärelastiskt tvärsnitt: Φ = ∫ ½(EA ε² + EI₁₁ K₁² + EI₂₂ K₂² + GJ e₁²) dS
//!   med tre gausspunkter längs balken.
//!
//! Inre kraft och tangent är derivator av Φ med hyperduala tal
//! (`hyperdual.rs`) med avseende på nodförskjutningar och rumsliga spinn
//! (C ← exp([δw]) C, samma uppdatering som NLGEOM-steget gör). Kraften är
//! exakt; i B33 är tangenten approximativ i vridningstermerna (rotations-
//! vektorns andra derivata mot spinnet tas inte med), vilket bara påverkar
//! konvergenshastigheten.
//!
//! Rotationerna i `u` är totala rotationsvektorer (som UR i utdata).

use super::hyperdual::{self as hd, Hd};
use super::tensor as t;
use crate::elements::beam_eb::EulerBeam;
use crate::elements::{ElemCtx, ElemResponse, ElementType};
use crate::model::{fail, Result};

pub fn response(b: &EulerBeam, ctx: &ElemCtx, u: &[f64], state_old: &[f64], state_new: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    if b.dim() == 2 {
        plane(b, ctx, u, state_old, state_new, tangent)
    } else {
        spatial(b, ctx, u, state_old, state_new, tangent)
    }
}

/// Gausspunkter och vikter på [0, 1] (3 punkter längs balken).
const GP3: [(f64, f64); 3] = [(0.112_701_665_379_258_31, 5.0 / 18.0), (0.5, 8.0 / 18.0), (0.887_298_334_620_741_7, 5.0 / 18.0)];

/// Hermitefunktionernas första och andra derivata mot ξ i [x₁, T₁, x₂, T₂]-ordning.
fn hermite(xi: f64) -> ([f64; 4], [f64; 4]) {
    (
        [6.0 * xi * xi - 6.0 * xi, 3.0 * xi * xi - 4.0 * xi + 1.0, -6.0 * xi * xi + 6.0 * xi, 3.0 * xi * xi - 2.0 * xi],
        [12.0 * xi - 6.0, 6.0 * xi - 4.0, -12.0 * xi + 6.0, 6.0 * xi - 2.0],
    )
}

/// Inre kraft, tangent och nya inre variabler.
type Condensed = (Vec<f64>, Option<Vec<f64>>, [f64; 2]);

/// Löser ∂Φ/∂m = 0 för de två inre variablerna (de sista två i q) med Newton
/// och returnerar den statiskt kondenserade kraften och tangenten för de
/// övriga `np` variablerna samt m.
fn condense<const N: usize>(ctx: &ElemCtx, name: &str, np: usize, m0: [f64; 2], tangent: bool, eval: &dyn Fn(&[f64; 2]) -> Hd<N>) -> Result<Condensed> {
    let (i1, i2) = (np, np + 1);
    let mut m = m0;
    let mut e = eval(&m);
    for _ in 0..30 {
        let (a, bb, c) = (e.h[i1][i1], e.h[i1][i2], e.h[i2][i2]);
        let det = a * c - bb * bb;
        if !(det > 0.0) {
            return fail(format!("element {} ({name}): sträckningen kunde inte bestämmas", ctx.label));
        }
        let dm = [(c * e.g[i1] - bb * e.g[i2]) / det, (a * e.g[i2] - bb * e.g[i1]) / det];
        m[0] -= dm[0];
        m[1] -= dm[1];
        e = eval(&m);
        if dm[0].abs().max(dm[1].abs()) < 1e-15 {
            break;
        }
    }
    let fint = e.g[..np].to_vec();
    let k = tangent.then(|| {
        // K = Φ_pp − Φ_pm Φ_mm⁻¹ Φ_mp, symmetriserad
        let (a, bb, c) = (e.h[i1][i1], e.h[i1][i2], e.h[i2][i2]);
        let det = a * c - bb * bb;
        let inv = [[c / det, -bb / det], [-bb / det, a / det]];
        let mut k = vec![0.0; np * np];
        for i in 0..np {
            for j in 0..np {
                let mut s = 0.5 * (e.h[i][j] + e.h[j][i]);
                for r in 0..2 {
                    for q in 0..2 {
                        s -= e.h[i][np + r] * inv[r][q] * e.h[np + q][j];
                    }
                }
                k[i * np + j] = s;
            }
        }
        k
    });
    Ok((fint, k, m))
}

type Hd8 = Hd<8>;
type Hd14 = Hd<14>;

/// Töjningsenergin för B23 med variablerna q = [u1, v1, φ1, u2, v2, φ2, m1, m2]
/// (m = nodens sträckning dl/dS − 1).
fn plane_energy(x0: [[f64; 3]; 2], l0: f64, beta0: f64, ea: f64, ei: f64, q: &[Hd8; 8]) -> Hd8 {
    // lägen relativt nod 1 (q är störningar kring u, x0 = [0, x₂ − x₁])
    let x1 = [q[0] + x0[0][0], q[1] + x0[0][1]];
    let x2 = [q[3] + x0[1][0], q[4] + x0[1][1]];
    let tan = |phi: Hd8, m: Hd8| {
        let a = phi + beta0;
        let s = (m + 1.0) * l0;
        [s * a.cos(), s * a.sin()]
    };
    let t1 = tan(q[2], q[6]);
    let t2 = tan(q[5], q[7]);
    let mut w = Hd8::cst(0.0);
    for &(xi, wt) in &GP3 {
        let (d1, d2) = hermite(xi);
        let comb = |d: &[f64; 4], k: usize, s: f64| (x1[k] * d[0] + t1[k] * d[1] + x2[k] * d[2] + t2[k] * d[3]) * s;
        let xd = [comb(&d1, 0, 1.0 / l0), comb(&d1, 1, 1.0 / l0)];
        let xdd = [comb(&d2, 0, 1.0 / (l0 * l0)), comb(&d2, 1, 1.0 / (l0 * l0))];
        let g = xd[0] * xd[0] + xd[1] * xd[1];
        let eps = (g - 1.0) * 0.5;
        // K = n · x″ med n = t roterad +90°
        let k = (xd[0] * xdd[1] - xd[1] * xdd[0]) / g.sqrt();
        w = w + (eps * eps * ea + k * k * ei) * (0.5 * wt * l0);
    }
    w
}

/// B23: [u1, v1, φ1, u2, v2, φ2]; tillståndet är de två nodsträckningarna.
fn plane(b: &EulerBeam, ctx: &ElemCtx, u: &[f64], state_old: &[f64], state_new: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    let d = b.data(ctx)?;
    let l0 = d.l;
    let ea = d.k_local[0] * l0;
    let ei = d.k_local[2 * 6 + 2] * l0 / 4.0;
    let beta0 = (ctx.x[1][1] - ctx.x[0][1]).atan2(ctx.x[1][0] - ctx.x[0][0]);
    let x0 = [[0.0; 3], [(ctx.x[1][0] - ctx.x[0][0]) + (u[3] - u[0]), (ctx.x[1][1] - ctx.x[0][1]) + (u[4] - u[1]), 0.0]];
    let m0 = if state_old.len() == 2 { [state_old[0], state_old[1]] } else { [0.0; 2] };
    let eval = |m: &[f64; 2]| {
        let q: [Hd8; 8] = std::array::from_fn(|i| {
            let v = match i {
                2 | 5 => u[i],
                6 | 7 => m[i - 6],
                _ => 0.0,
            };
            Hd8::var(v, i)
        });
        plane_energy(x0, l0, beta0, ea, ei, &q)
    };
    let (fint, k, m) = condense(ctx, b.name(), 6, m0, tangent, &eval)?;
    *state_new = m.to_vec();
    Ok(ElemResponse { fint, k })
}

/// Styvheter och referensgeometri för B33.
struct Spatial {
    ea: f64,
    ei11: f64,
    ei22: f64,
    gj: f64,
    l0: f64,
    /// Elementets riktning T och tvärsnittets n1-riktning N₁ i referensläget.
    tv: [f64; 3],
    n1: [f64; 3],
}

/// Aktuellt läge kring vilket B33-energin deriveras.
struct Current {
    /// Nodlägen relativt nod 1: [0, x₂ − x₁].
    dx: [[f64; 3]; 2],
    c0: [t::M3; 2],
    phi0: [[f64; 3]; 2],
    /// T(φ₀)⁻¹: ändringen av rotationsvektorn per spinn.
    tinv: [t::M3; 2],
}

/// Töjningsenergin för B33. Variablerna: [δx₁, δw₁, δx₂, δw₂, m₁, m₂], där
/// δx och δw (spinn) är störningar kring det aktuella läget.
fn spatial_energy(s: &Spatial, cur: &Current, q: &[Hd14; 14]) -> Hd14 {
    let cst = Hd14::cst;
    let l0 = s.l0;
    let mut xs = [[cst(0.0); 3]; 2];
    let mut tans = [[cst(0.0); 3]; 2];
    let mut phis = [[cst(0.0); 3]; 2];
    for a in 0..2 {
        let o = 6 * a;
        // lägen relativt nod 1 (bara skillnader ingår; minskar avrundningsfelet i ε)
        xs[a] = std::array::from_fn(|i| q[o + i] + cur.dx[a][i]);
        let w: hd::V3<14> = [q[o + 3], q[o + 4], q[o + 5]];
        let e = hd::exp_so3(&w);
        // nodens rotation C = exp([w]) C₀ och tangent (1 + m) L C T
        let ct: hd::V3<14> = std::array::from_fn(|i| cst((0..3).map(|j| cur.c0[a][i][j] * s.tv[j]).sum()));
        let rt = hd::matvec3(&e, &ct);
        let sc = (q[12 + a] + 1.0) * l0;
        tans[a] = std::array::from_fn(|i| rt[i] * sc);
        // rotationsvektorn till första ordningen i spinnet: φ = φ₀ + T(φ₀)⁻¹ w
        phis[a] = std::array::from_fn(|i| (0..3).fold(cst(cur.phi0[a][i]), |acc, j| acc + w[j] * cur.tinv[a][i][j]));
    }
    let dphi: hd::V3<14> = std::array::from_fn(|i| phis[1][i] - phis[0][i]);
    let n1c: hd::V3<14> = s.n1.map(cst);
    let mut wsum = cst(0.0);
    for &(xi, wt) in &GP3 {
        let (d1, d2) = hermite(xi);
        let comb = |d: &[f64; 4], sc: f64| -> hd::V3<14> {
            std::array::from_fn(|k| (xs[0][k] * d[0] + tans[0][k] * d[1] + xs[1][k] * d[2] + tans[1][k] * d[3]) * sc)
        };
        let xd = comb(&d1, 1.0 / l0);
        let xdd = comb(&d2, 1.0 / (l0 * l0));
        let g = hd::dot3(&xd, &xd);
        let eps = (g - 1.0) * 0.5;
        let gl = g.sqrt();
        let tt: hd::V3<14> = std::array::from_fn(|i| xd[i] / gl);
        let psi: hd::V3<14> = std::array::from_fn(|i| phis[0][i] * (1.0 - xi) + phis[1][i] * xi);
        let nt = hd::matvec3(&hd::exp_so3(&psi), &n1c);
        // dñ₁/dS = [T(ψ) dψ/dS] × ñ₁
        let om = hd::matvec3(&hd::tangent_so3(&psi), &dphi);
        let dnt: hd::V3<14> = hd::cross3(&om, &nt).map(|v| v * (1.0 / l0));
        let proj = hd::dot3(&nt, &tt);
        let nb: hd::V3<14> = std::array::from_fn(|i| nt[i] - proj * tt[i]);
        let nl = hd::dot3(&nb, &nb).sqrt();
        let nn1: hd::V3<14> = nb.map(|v| v / nl);
        let nn2 = hd::cross3(&tt, &nn1);
        let k2 = hd::dot3(&nn1, &xdd);
        let k1 = -hd::dot3(&nn2, &xdd);
        let e1 = hd::dot3(&nn2, &dnt);
        let dens = eps * eps * s.ea + k1 * k1 * s.ei11 + k2 * k2 * s.ei22 + e1 * e1 * s.gj;
        wsum = wsum + dens * (0.5 * wt * l0);
    }
    wsum
}

/// B33: [u1, u2, u3, φ1, φ2, φ3] per nod; tillståndet är nodsträckningarna.
fn spatial(b: &EulerBeam, ctx: &ElemCtx, u: &[f64], state_old: &[f64], state_new: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    let d = b.data(ctx)?;
    let l0 = d.l;
    let kl = &d.k_local;
    let l3 = l0 * l0 * l0;
    let [tv, n1, _] = d.axes;
    let sp = Spatial { ea: kl[0] * l0, gj: kl[3 * 12 + 3] * l0, ei22: kl[12 + 1] * l3 / 12.0, ei11: kl[2 * 12 + 2] * l3 / 12.0, l0, tv, n1 };
    let phi0: [[f64; 3]; 2] = std::array::from_fn(|a| [u[6 * a + 3], u[6 * a + 4], u[6 * a + 5]]);
    let cur = Current {
        dx: [[0.0; 3], std::array::from_fn(|i| (ctx.x[1][i] - ctx.x[0][i]) + (u[6 + i] - u[i]))],
        c0: phi0.map(t::exp_so3),
        phi0,
        tinv: phi0.map(|p| t::inv(&so3_tangent(p)).0),
    };
    let m0 = if state_old.len() == 2 { [state_old[0], state_old[1]] } else { [0.0; 2] };
    let eval = |m: &[f64; 2]| {
        let q: [Hd14; 14] = std::array::from_fn(|i| Hd14::var(if i < 12 { 0.0 } else { m[i - 12] }, i));
        spatial_energy(&sp, &cur, &q)
    };
    let (fint, k, m) = condense(ctx, b.name(), 12, m0, tangent, &eval)?;
    if fint.iter().any(|v| !v.is_finite()) {
        return fail(format!("element {} ({}): ogiltig deformerad geometri", ctx.label, b.name()));
    }
    *state_new = m.to_vec();
    Ok(ElemResponse { fint, k })
}

/// T(φ) i f64 (dC = [T dφ]× C).
fn so3_tangent(p: [f64; 3]) -> t::M3 {
    let q: hd::V3<1> = p.map(Hd::<1>::cst);
    let m = hd::tangent_so3(&q);
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j].v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T(φ) stämmer med differenser av exp: dC/dφ_k Cᵀ = [T e_k]×.
    #[test]
    fn so3_tangent_matches_exp() {
        let p = [0.4, -1.1, 0.7];
        let tm = so3_tangent(p);
        let c = t::exp_so3(p);
        let h = 1e-6;
        for k in 0..3 {
            let mut pp = p;
            pp[k] += h;
            let mut pm = p;
            pm[k] -= h;
            let dc = t::scale(&t::sub(&t::exp_so3(pp), &t::exp_so3(pm)), 0.5 / h);
            let w = t::mul(&dc, &t::transpose(&c));
            let ax = [w[2][1], w[0][2], w[1][0]];
            for i in 0..3 {
                assert!((ax[i] - tm[i][k]).abs() < 1e-8, "{k} {i}: {} {}", ax[i], tm[i][k]);
            }
        }
    }
}
