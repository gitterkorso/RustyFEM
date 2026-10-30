//! Elementvis stabilitetsgräns för explicit dynamik
//! (`ELEMENT BY ELEMENT`, `IMPROVED DT METHOD=NO`).
//!
//! Δtₑ = Lₑ/c_d · (√(1+ξ²) − ξ), där c_d = √(D₁₁/ρ) (D = materialets elastiska
//! tangent: λ+2μ för solider och plan töjning, E/(1−ν²) för plan spänning,
//! skal och membran) och ξ den kritiska dämpningskvoten i högsta moden från
//! volymviskositeten (ξ = b₁ − b₂² Lₑ/c_d · min(0, ε̇_vol)) och Rayleigh-β/α.
//! Den karakteristiska längden Lₑ med bᵢ = ∫∇Nᵢ dV (kontrollerad med fältet EDT
//! för fria element med olika former):
//!
//! | Element | Lₑ |
//! |---|---|
//! | 3D-solider (C3D8, C3D8R, C3D4, ...) | V / √(2 Σᵢ\|bᵢ\|²) |
//! | fyrhörningar (CPE4, CPS4R, S4R, M3D4R, ...) | A / √(Σᵢ\|bᵢ\|²) = A / √((d₁² + d₂²)/2) (d = diagonaler) |
//! | trianglar (CPE3, S3R, M3D3, ...) | 4A / √(3 Σ lₖ²) (l = kanter) |
//! | stänger | 0,99 L |
//!
//! Standardmetoden (`IMPROVED DT METHOD=YES`) används för skal och
//! membran (`improved_plane_stress`: √0,92 respektive √0,97 gånger den exakta
//! högsta egenfrekvensen för konstant töjning). För 3D-solider används
//! formlerna ovan även med metoden, vilket är på den säkra sidan. Plan
//! töjning (CPE3, CPE4R) påverkas inte av metoden.
//! Balkar och element som saknas i tabellen får elementets högsta
//! egenfrekvens med diagonal massa (potensiteration); för balkar är det
//! en konservativ gräns.

use crate::elements::common::{cross, dot, norm, sub};
use crate::elements::{ElemCtx, Family};
use crate::fem::assembly::Prepared;
use crate::materials::StressMode;
use crate::model::Result;

/// Elementets geometri för stabilitet och volymviskositet: bᵢ = ∫∇Nᵢ dV
/// (utan tjocklek för plana element och skal) och volymen (arean).
pub struct ElemGeom {
    pub b: Vec<[f64; 3]>,
    pub vol: f64,
    /// Karakteristisk längd.
    pub lchar: f64,
}

/// Stångens karakteristiska längd relativt L (se modulens dokumentation).
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
    // ∫∇Nᵢ dV = (yta motsatt nod i) · (utåtriktad normal) / 3 med tecken
    let faces = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];
    let mut b = vec![[0.0; 3]; 4];
    for (i, f) in faces.iter().enumerate() {
        let n = cross(sub(x[f[1]], x[f[0]]), sub(x[f[2]], x[f[0]]));
        // n pekar ut från tetraedern för ytan motsatt nod i; ∇Nᵢ pekar inåt
        for c in 0..3 {
            b[i][c] = -n[c] / 6.0;
        }
    }
    Some((b, v6 / 6.0))
}

/// Plana koordinater (2D) för ett plant element i 3D (skal, membran) och
/// planets basvektorer e₁, e₂.
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

/// Plana koordinater för `family` (oförändrade för plana kontinuumelement)
/// och en funktion som för bᵢ tillbaka till globala axlar.
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

/// Geometri för element med känd karakteristisk längd (se tabellen), i
/// konfigurationen `x`. `None` för övriga element.
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

/// Materialets dilatationsvåghastighet c_d = √(D₁₁/ρ) för elementets spänningstillstånd.
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

/// Högsta egenvärdet ω² av M⁻¹K för ett element (diagonal massa), med potensiteration.
pub fn element_omega2(k: &[f64], m: &[f64]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 0.0;
    }
    // symmetrisk form D^{-1/2} K D^{-1/2}
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

/// Odämpad stabilitetsgräns för element `e` i konfigurationen `x`:
/// (2/ωₑ, Lₑ, c_d). `mdiag` är elementets diagonala massa (för
/// potensiterationen), `None` för element utan massa eller styvhet.
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

/// Förbättrad stabilitetsgräns (`IMPROVED DT METHOD=YES`, standard)
/// för skal och membran: Lₑ = max(Lₑ,konv, k·L_exakt), där L_exakt = 2c/ω är
/// den exakta högsta egenfrekvensen för elementets konstanta töjningsfält med
/// diagonal massa och plan spänning,
///
/// ```text
/// ω² = (n/ρ) · max(E′/2 [(s₁+s₂) + √((s₁−s₂)² + 4ν² s₁s₂)],  G (s₁+s₂))
/// ```
///
/// (n = antal noder, s₁, s₂ = egenvärdena av Σᵢ bᵢ⊗bᵢ / A², E′ = E/(1−ν²)),
/// och k² = 0,92 för fyrhörningar, 0,97 för trianglar (empiriska faktorer,
/// kontrollerade för kvadrater, rektanglar 1×2 och 1×3 och parallellogram med
/// ν = 0,3 och 0,45 samt räta och liksidiga trianglar).
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
    // i enheter av E′: normaldelen och skjuvdelen G = E′(1 − ν)/2
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

/// Dämpningsfaktorn √(1+ξ²) − ξ.
pub fn damping_factor(xi: f64) -> f64 {
    (1.0 + xi * xi).sqrt() - xi
}
