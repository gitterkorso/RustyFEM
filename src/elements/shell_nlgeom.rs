//! Skal och membran (S4, S4R, S3, S3R, M3D4, M3D3) vid NLGEOM=YES:
//! elementoberoende korotationell formulering (Rankin–Nour-Omid 1988,
//! Felippa–Haugen 2005, Battini–Pacoste 2002).
//!
//! Elementets stelkroppsrörelse avskiljs med ett medföljande system E(x)
//! (normalen från diagonalerna för fyrhörningar respektive från två kanter för
//! trianglar, 1-axeln längs elementets ξ-riktning respektive första kanten).
//! Stelkroppsrotationen är R_r = E E₀ᵀ, och de deformationella förskjutningarna
//! uttrycks i referenslägets orientering:
//!
//! - u_dₐ = R_rᵀ (xₐ − x_c) − (Xₐ − X_c)  (x_c = nodernas medelpunkt),
//! - θ_dₐ = log(R_rᵀ Rₐ)  (nodens rotation relativt elementets).
//!
//! Töjningsenergin är Φ = ½ dᵀ (K₀ − Kₘ) d + Φₘ(x), där K₀ är elementets
//! linjära styvhet i referensläget (samma formulering som i linjär analys) och
//! membrandelen räknas med logaritmisk töjning direkt ur nodlägena:
//! Φₘ = Σ ½ t A εᵀ D ε med ε = ½ ln C, C = gᵀg (ytans metrik i de lokala
//! referensaxlarna, gα = ∂x/∂Xα), i membranets integrationspunkter (en punkt i
//! mitten för S4/S4R, konstant töjning för S3/S3R/M3D3, 2×2 för M3D4). Kₘ = ∂²Φₘ
//! i referensläget är exakt den linjära membrandelen, så den dras av från K₀
//! (M3D4/M3D3 blir rena Hencky-membran). Det motsvarar finita
//! membrantöjningar med tvärsnittets standardvärde POISSON = 0,5 (konstant
//! volym, t = t₀·exp(−(e11 + e22)), Kirchhoff- = Cauchyspänning); se
//! tests/cases/nlgeom_struct.
//!
//! Kraft och tangent är derivator av Φ med avseende på nodförskjutningar och
//! rumsliga spinn (R ← exp([δw]) R, som NLGEOM-steget uppdaterar noder med tre
//! rotationer): f = Jᵀ K d + ∇Φₘ och K = Jᵀ K J + Σᵢ (K d)ᵢ ∂²dᵢ + ∂²Φₘ, där
//! J = ∂d/∂q och ∂²dᵢ räknas exakt med hyperduala tal. Tangenten är därmed
//! konsistent och symmetrisk.
//!
//! Böjning, tvärskjuvning, timglas och drill förutsätter små deformationella
//! töjningar men tillåter godtyckligt stora rotationer. En formulering med
//! skjuvtöjningen som sinus av vinkeln mellan direktör och yta, tvärkraften
//! längs den roterade normalen och krökningen per aktuell längd skiljer sig
//! med storleksordningen töjningen. Utdata (S, SF, SM) räknas med de
//! deformationella förskjutningarna i elementets lokala axlar, som följer med
//! elementets rotation, med den logaritmiska membrantöjningen och membrankrafter
//! med aktuell tjocklek; membranen skriver S och LE. Tryck (*DLOAD P) följer den
//! deformerade ytan (`kinematics::pressure`).

use crate::elements::beam_timoshenko::nlgeom::{chain, log_so3, mtm, mtv, spun};
use crate::elements::{ElemCtx, ElemResponse};
use crate::kinematics::hyperdual::{self as hd, Hd, T3, V3};
use crate::kinematics::tensor as t;
use crate::model::{fail, Result};

/// Medföljande system E (kolumner e₁, e₂, e₃) för nodlägena `x`.
fn frame<const N: usize>(x: &[V3<N>]) -> T3<N> {
    let sub = |a: &V3<N>, b: &V3<N>| -> V3<N> { std::array::from_fn(|i| a[i] - b[i]) };
    let unit = |v: V3<N>| -> V3<N> {
        let l = hd::dot3(&v, &v).sqrt().recip();
        v.map(|c| c * l)
    };
    let (n, g) = if x.len() == 4 {
        let n = hd::cross3(&sub(&x[2], &x[0]), &sub(&x[3], &x[1]));
        let g: V3<N> = std::array::from_fn(|i| x[1][i] + x[2][i] - x[0][i] - x[3][i]);
        (n, g)
    } else {
        let a = sub(&x[1], &x[0]);
        (hd::cross3(&a, &sub(&x[2], &x[0])), a)
    };
    let e3 = unit(n);
    let p = hd::dot3(&g, &e3);
    let e1 = unit(std::array::from_fn(|i| g[i] - e3[i] * p));
    let e2 = hd::cross3(&e3, &e1);
    std::array::from_fn(|i| [e1[i], e2[i], e3[i]])
}

/// Deformationella förskjutningar d (nodvis, `nd` = 3 eller 6 per nod).
/// `var(a, c)` ger störningen för nod a, komponent c (0–2 förskjutning, 3–5 spinn).
fn deformational<const N: usize>(x0: &[[f64; 3]], u: &[f64], nd: usize, var: &dyn Fn(usize, usize) -> Hd<N>) -> Option<Vec<Hd<N>>> {
    let nn = x0.len();
    // nodlägen relativt nod 1 (minskar avrundningsfelet)
    let rel0: Vec<[f64; 3]> = x0.iter().map(|p| std::array::from_fn(|i| p[i] - x0[0][i])).collect();
    let xs: Vec<V3<N>> = (0..nn)
        .map(|a| std::array::from_fn(|i| var(a, i) + (rel0[a][i] + u[nd * a + i] - u[i])))
        .collect();
    let xr: Vec<V3<N>> = rel0.iter().map(|p| p.map(Hd::cst)).collect();
    let e = frame(&xs);
    let e0 = frame(&xr);
    // R_r = E E₀ᵀ; R_rᵀ = E₀ Eᵀ
    let e0t: T3<N> = std::array::from_fn(|i| std::array::from_fn(|j| e0[j][i]));
    let rr: T3<N> = std::array::from_fn(|i| std::array::from_fn(|j| e[i][0] * e0t[0][j] + e[i][1] * e0t[1][j] + e[i][2] * e0t[2][j]));
    let mean = |v: &[V3<N>]| -> V3<N> { std::array::from_fn(|i| v.iter().fold(Hd::cst(0.0), |acc, p| acc + p[i]) * (1.0 / nn as f64)) };
    let (xc, xc0) = (mean(&xs), mean(&xr));
    let mut d = Vec::with_capacity(nd * nn);
    for a in 0..nn {
        let dx: V3<N> = std::array::from_fn(|i| xs[a][i] - xc[i]);
        let ud = mtv(&rr, &dx);
        for i in 0..3 {
            d.push(ud[i] - (xr[a][i] - xc0[i]).v);
        }
        if nd == 6 {
            let r0 = t::exp_so3([u[nd * a + 3], u[nd * a + 4], u[nd * a + 5]]);
            let ra = spun(&[var(a, 3), var(a, 4), var(a, 5)], &r0);
            let th = log_so3(&mtm(&rr, &ra))?;
            d.extend_from_slice(&th);
        }
    }
    Some(d)
}

fn too_large(ctx: &ElemCtx, name: &str) -> crate::model::FemError {
    crate::model::FemError(format!("element {} ({name}): för stor rotation av en nod relativt elementet (≥ 180°)", ctx.label))
}

/// Integrationsregel för membrandelen.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MemRule {
    /// En punkt i mitten (S4R, S4).
    Center,
    /// 2×2 gausspunkter (M3D4).
    Gauss4,
    /// Konstant töjning (S3, S3R, M3D3).
    Tri,
}

/// Membrandelens data: formfunktionernas derivator mot de lokala
/// referenskoordinaterna och vikt (area) per integrationspunkt, tjocklek och
/// plan spänningsmatris.
pub struct Membrane {
    ips: Vec<(Vec<[f64; 2]>, f64)>,
    t: f64,
    d: [f64; 9],
}

impl Membrane {
    pub fn new(ctx: &ElemCtx, name: &str, rule: MemRule) -> Result<Membrane> {
        let loc = super::local_frame(&ctx.x, ctx.label, name)?;
        let sec = super::ShellSec::get(ctx, name)?;
        let ips = match rule {
            MemRule::Center => {
                let p = super::q4pt(&loc.xl, 0.0, 0.0);
                vec![((0..4).map(|i| [p.bx[i], p.by[i]]).collect(), 4.0 * p.det)]
            }
            MemRule::Gauss4 => super::GP4
                .iter()
                .map(|&(s, t)| {
                    let p = super::q4pt(&loc.xl, s, t);
                    ((0..4).map(|i| [p.bx[i], p.by[i]]).collect(), p.det)
                })
                .collect(),
            MemRule::Tri => {
                let (bx, by, a) = super::tri::cst(&loc.xl);
                vec![((0..3).map(|i| [bx[i], by[i]]).collect(), a)]
            }
        };
        Ok(Membrane { ips, t: sec.t, d: sec.d })
    }

    /// Logaritmiska membrantöjningar [e11, e22, 2·e12] (lokala referensaxlar)
    /// i integrationspunkterna för nodlägena `x`.
    pub fn strains<const N: usize>(&self, x: &[V3<N>]) -> Vec<[Hd<N>; 3]> {
        self.ips
            .iter()
            .map(|(gr, _)| {
                let g: [V3<N>; 2] = std::array::from_fn(|al| {
                    std::array::from_fn(|i| x.iter().zip(gr).fold(Hd::cst(0.0), |acc, (p, d)| acc + p[i] * d[al]))
                });
                let (a, b, c) = (hd::dot3(&g[0], &g[0]), hd::dot3(&g[0], &g[1]), hd::dot3(&g[1], &g[1]));
                let l = log2(a, b, c);
                [l[0] * 0.5, l[1] * 0.5, l[2] * 0.5]
            })
            .collect()
    }

    /// Φₘ = Σ ½ t A εᵀ D ε med logaritmisk töjning.
    fn energy<const N: usize>(&self, x: &[V3<N>]) -> Hd<N> {
        let mut w = Hd::cst(0.0);
        for (e, (_, a)) in self.strains(x).iter().zip(&self.ips) {
            for i in 0..3 {
                let mut se = Hd::cst(0.0);
                for j in 0..3 {
                    let dij = self.d[i * 3 + j];
                    if dij != 0.0 {
                        se = se + e[j] * dij;
                    }
                }
                w = w + e[i] * se * (0.5 * self.t * a);
            }
        }
        w
    }
}

/// g(z) = atanh(√z)/√z med derivator mot z (serie för små z).
fn atanh_coef(z: f64) -> (f64, f64, f64) {
    if z < 1e-2 {
        let (mut f, mut d1, mut d2) = (0.0, 0.0, 0.0);
        for k in 0..14usize {
            let c = 1.0 / (2 * k + 1) as f64;
            f += c * z.powi(k as i32);
            if k >= 1 {
                d1 += c * k as f64 * z.powi(k as i32 - 1);
            }
            if k >= 2 {
                d2 += c * (k * (k - 1)) as f64 * z.powi(k as i32 - 2);
            }
        }
        return (f, d1, d2);
    }
    let r = z.sqrt();
    let at = r.atanh();
    let q = 1.0 - r * r;
    let hr = 1.0 / (r * q) - at / (r * r);
    let hrr = -(1.0 - 3.0 * r * r) / (r * r * q * q) - 1.0 / (r * r * q) + 2.0 * at / (r * r * r);
    (at / r, hr / (2.0 * r), hrr / (4.0 * z) - hr / (4.0 * r * r * r))
}

/// ln C för en symmetrisk positivt definit 2×2-matris C = [[a, b], [b, c]] som
/// Voigt-vektor [l11, l22, 2·l12]: ln C = α I + β C med s = (a + c)/2,
/// q = ((a − c)/2)² + b², β = g(q/s²)/s och α = ½ ln(s² − q) − g(q/s²). Uttrycket
/// är glatt även när egenvärdena sammanfaller (som i referensläget).
fn log2<const N: usize>(a: Hd<N>, b: Hd<N>, c: Hd<N>) -> [Hd<N>; 3] {
    let s = (a + c) * 0.5;
    let h = (a - c) * 0.5;
    let q = h * h + b * b;
    let is = s.recip();
    let z = q * is * is;
    let (g, g1, g2) = atanh_coef(z.v);
    let gz = chain(&z, g, g1, g2);
    let beta = gz * is;
    let det = s * s - q;
    let ld = chain(&det, det.v.ln(), 1.0 / det.v, -1.0 / (det.v * det.v));
    let alpha = ld * 0.5 - gz;
    [alpha + beta * a, alpha + beta * c, beta * b * 2.0]
}

/// Nodlägen relativt nod 1 som hyperduala tal (förskjutning `u` plus störning).
pub fn positions_hd<const N: usize>(x0: &[[f64; 3]], u: &[f64], nd: usize, var: &dyn Fn(usize, usize) -> Hd<N>) -> Vec<V3<N>> {
    positions(x0, u, nd, var)
}

/// Deformationella förskjutningar och rotationer (6 per nod) som hyperduala
/// tal (område struct_nonlin: skal med integrering genom tjockleken).
pub fn deformational_hd<const N: usize>(ctx: &ElemCtx, name: &str, u: &[f64], var: &dyn Fn(usize, usize) -> Hd<N>) -> Result<Vec<Hd<N>>> {
    deformational::<N>(&ctx.x, u, 6, var).ok_or_else(|| too_large(ctx, name))
}

fn positions<const N: usize>(x0: &[[f64; 3]], u: &[f64], nd: usize, var: &dyn Fn(usize, usize) -> Hd<N>) -> Vec<V3<N>> {
    (0..x0.len()).map(|a| std::array::from_fn(|i| var(a, i) + (x0[a][i] - x0[0][i] + u[nd * a + i] - u[i]))).collect()
}

/// Logaritmiska membrantöjningar [e11, e22, 2·e12] per integrationspunkt
/// (regeln `rule`), för utdata.
pub fn membrane_strains(ctx: &ElemCtx, name: &str, u: &[f64], rule: MemRule) -> Result<Vec<[f64; 3]>> {
    let nn = ctx.x.len();
    let nd = u.len() / nn;
    let mem = Membrane::new(ctx, name, rule)?;
    let var = |_: usize, _: usize| Hd::<1>::cst(0.0);
    Ok(mem.strains::<1>(&positions::<1>(&ctx.x, u, nd, &var)).iter().map(|e| e.map(|v| v.v)).collect())
}

fn response_n<const N: usize>(ctx: &ElemCtx, name: &str, k0: &[f64], u: &[f64], nd: usize, mem: Option<&Membrane>, tangent: bool) -> Result<ElemResponse> {
    let var = |a: usize, c: usize| -> Hd<N> { if c < nd { Hd::var(0.0, nd * a + c) } else { Hd::cst(0.0) } };
    let d = deformational::<N>(&ctx.x, u, nd, &var).ok_or_else(|| too_large(ctx, name))?;
    // membrandelen med logaritmisk töjning: Φ = ½ dᵀ (K₀ − Kₘ) d + Φₘ(x), där
    // Kₘ = ∂²Φₘ i referensläget är membrandelen av den linjära styvheten
    let (krest, phim) = match mem {
        Some(m) => {
            let zero = vec![0.0; u.len()];
            let km = m.energy(&positions::<N>(&ctx.x, &zero, nd, &var));
            let mut kr = k0.to_vec();
            for i in 0..N {
                for j in 0..N {
                    kr[i * N + j] -= km.h[i][j];
                }
            }
            (kr, Some(m.energy(&positions::<N>(&ctx.x, u, nd, &var))))
        }
        None => (k0.to_vec(), None),
    };
    let k0 = &krest[..];
    // s = K₀ d
    let s: Vec<f64> = (0..N).map(|i| (0..N).map(|j| k0[i * N + j] * d[j].v).sum()).collect();
    let mut fint: Vec<f64> = (0..N).map(|q| (0..N).map(|i| d[i].g[q] * s[i]).sum()).collect();
    if let Some(pm) = &phim {
        for q in 0..N {
            fint[q] += pm.g[q];
        }
    }
    if fint.iter().any(|v| !v.is_finite()) {
        return fail(format!("element {} ({name}): ogiltig deformerad geometri", ctx.label));
    }
    let k = tangent.then(|| {
        // K = Jᵀ K₀ J + Σ sᵢ Hᵢ
        let mut kj = vec![0.0; N * N]; // K₀ J
        for i in 0..N {
            for j in 0..N {
                let kij = k0[i * N + j];
                if kij != 0.0 {
                    for q in 0..N {
                        kj[i * N + q] += kij * d[j].g[q];
                    }
                }
            }
        }
        let mut k = vec![0.0; N * N];
        for i in 0..N {
            for p in 0..N {
                let jip = d[i].g[p];
                if jip != 0.0 {
                    for q in 0..N {
                        k[p * N + q] += jip * kj[i * N + q];
                    }
                }
            }
            if s[i] != 0.0 {
                for p in 0..N {
                    for q in 0..N {
                        k[p * N + q] += s[i] * d[i].h[p][q];
                    }
                }
            }
        }
        if let Some(pm) = &phim {
            for p in 0..N {
                for q in 0..N {
                    k[p * N + q] += pm.h[p][q];
                }
            }
        }
        // symmetrisera avrundningen
        for p in 0..N {
            for q in 0..p {
                let v = 0.5 * (k[p * N + q] + k[q * N + p]);
                k[p * N + q] = v;
                k[q * N + p] = v;
            }
        }
        k
    });
    Ok(ElemResponse { fint, k })
}

/// Inre kraft och tangent vid NLGEOM för ett skal (6 frihetsgrader per nod)
/// eller membran (3 per nod) med linjär styvhet `k0` i referensläget.
pub fn response(ctx: &ElemCtx, name: &str, k0: &[f64], u: &[f64], rule: MemRule, tangent: bool) -> Result<ElemResponse> {
    let nn = ctx.x.len();
    let nd = u.len() / nn;
    let mem = Membrane::new(ctx, name, rule)?;
    let m = Some(&mem);
    match (nn, nd) {
        (4, 6) => response_n::<24>(ctx, name, k0, u, nd, m, tangent),
        (3, 6) => response_n::<18>(ctx, name, k0, u, nd, m, tangent),
        (4, 3) => response_n::<12>(ctx, name, k0, u, nd, m, tangent),
        (3, 3) => response_n::<9>(ctx, name, k0, u, nd, m, tangent),
        _ => fail(format!("element {} ({name}): NLGEOM stöds inte för {nn} noder", ctx.label)),
    }
}

/// Membrankrafterna SF1–SF3 med aktuell tjocklek t = t₀·exp(−(e11 + e22)):
/// Standardvärdet för skal vid NLGEOM är tvärsnittets Poissons tal 0,5
/// (konstant volym).
pub fn scale_membrane_forces(out: &mut [crate::elements::IpField], e: [f64; 3]) {
    let f = (-(e[0] + e[1])).exp();
    for fld in out.iter_mut().filter(|f| f.name == "SF") {
        for (_, _, v) in fld.points.iter_mut() {
            for x in v.iter_mut().take(3) {
                *x *= f;
            }
        }
    }
}

/// Deformationella förskjutningar (i referenslägets orientering) för utdata.
pub fn deformational_u(ctx: &ElemCtx, name: &str, u: &[f64]) -> Result<Vec<f64>> {
    let nn = ctx.x.len();
    let nd = u.len() / nn;
    let var = |_: usize, _: usize| Hd::<1>::cst(0.0);
    let d = deformational::<1>(&ctx.x, u, nd, &var).ok_or_else(|| too_large(ctx, name))?;
    Ok(d.iter().map(|v| v.v).collect())
}

/// Geometrisk styvhet för S4R: membrankrafterna `nm` med enpunktsintegrering
/// i mitten (som elementets övriga membran- och böjtermer), samma för u, v och w.
/// `x` är de plana lokala nodkoordinaterna; resultatet i lokala frihetsgrader.
pub fn geometric_s4r(x: &[[f64; 2]], nm: [f64; 3]) -> Vec<f64> {
    let n = 24;
    let mut k = vec![0.0; n * n];
    let p = super::q4pt(x, 0.0, 0.0);
    let a = 4.0 * p.det;
    for i in 0..4 {
        for j in 0..4 {
            let v = (p.bx[i] * (nm[0] * p.bx[j] + nm[2] * p.by[j]) + p.by[i] * (nm[2] * p.bx[j] + nm[1] * p.by[j])) * a;
            for d in 0..3 {
                k[(i * 6 + d) * n + j * 6 + d] += v;
            }
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stelkroppsrörelse ger inga deformationella förskjutningar.
    #[test]
    fn rigid_motion_is_strain_free() {
        let x0 = [[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [1.2, 1.0, 0.1], [0.1, 0.9, 0.0]];
        let rot = [0.7, -0.4, 1.1];
        let r = t::exp_so3(rot);
        let mut u = vec![0.0; 24];
        for a in 0..4 {
            let p = t::mv(&r, x0[a]);
            for i in 0..3 {
                u[6 * a + i] = p[i] - x0[a][i] + [0.3, -0.2, 0.5][i];
                u[6 * a + 3 + i] = rot[i];
            }
        }
        let var = |_: usize, _: usize| Hd::<1>::cst(0.0);
        let d = deformational::<1>(&x0, &u, 6, &var).unwrap();
        for v in d {
            assert!(v.v.abs() < 1e-12, "{}", v.v);
        }
    }

    /// Deformationella förskjutningar och deras derivator mot differenser.
    #[test]
    fn derivatives_match_differences() {
        let x0 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 0.8, 0.0]];
        let u: Vec<f64> = (0..18).map(|i| 0.05 * ((i * 7 % 11) as f64 - 5.0) / 5.0).collect();
        let var = |a: usize, c: usize| Hd::<18>::var(0.0, 6 * a + c);
        let d = deformational::<18>(&x0, &u, 6, &var).unwrap();
        let h = 1e-6;
        for q in 0..18 {
            let (a, c) = (q / 6, q % 6);
            let f = |s: f64| {
                let vv = |aa: usize, cc: usize| Hd::<1>::cst(if aa == a && cc == c { s } else { 0.0 });
                deformational::<1>(&x0, &u, 6, &vv).unwrap().iter().map(|v| v.v).collect::<Vec<f64>>()
            };
            let (fp, fm) = (f(h), f(-h));
            for i in 0..18 {
                let fd = (fp[i] - fm[i]) / (2.0 * h);
                assert!((fd - d[i].g[q]).abs() < 1e-7, "d{i}/dq{q}: {fd} {}", d[i].g[q]);
            }
        }
    }
}
