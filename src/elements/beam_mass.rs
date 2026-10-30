//! Massmatris och inre frihetsgrader för Euler–Bernoulli-balkarna B23 och B33.
//!
//! B23/B33 interpolerar inte bara utböjningen utan också
//! axialförskjutningen kubiskt (Hermite) och har därför två inre
//! frihetsgrader per element (axialtöjningen i ändarna, skalad med L). I en
//! statisk analys kondenseras de bort exakt (styvheten blir EA/L), men i
//! egenvärdessteg ingår de i egenvärdesproblemet. Det syns i axialmoder och i
//! moder där axial- och böjrörelse kopplas (ramar), se
//! tests/cases/eigen.
//!
//! - Böjning och axialt (inre frihetsgrader): ∫ρA HᵀH dx med de kubiska
//!   Hermite-funktionerna H, integrerat med 3-punkts Gauss.
//!   Ingen rotationströghet för böjning.
//! - Vridning (B33): konsistent linjär massa med ρ(I11+I22)·L.
//! - Nodmassan (`mass`, utan inre frihetsgrader) använder för axialt
//!   ρAL·[5 1; 1 5]/12, som ger nästan samma axialfrekvenser som den kubiska
//!   interpolationen. Egenvärdessteg använder `internal_dofs` i stället.
//! - Diagonal massa (MassKind::Lumped): ρAL/2 per nod i translation och
//!   ρ(I11+I22)L/2 i vridning, ingen böjningsrotationsmassa.

use super::api::{ElemCtx, ElementType, MassKind};
use super::beam_eb::EulerBeam;
use super::common;
use crate::model::Result;

/// Modulen registrerar inga egna elementtyper.
pub static TYPES: &[&dyn ElementType] = &[];

/// 3-punkts Gauss på [0, 1].
const GAUSS3: [(f64, f64); 3] = [
    (0.112_701_665_379_258_3, 5.0 / 18.0),
    (0.5, 8.0 / 18.0),
    (0.887_298_334_620_741_7, 5.0 / 18.0),
];

/// Antal inre frihetsgrader per B23/B33-element.
pub const N_INTERNAL: usize = 2;

/// Hermite-massa för total massa `m`, ordning [v1, θ1, v2, θ2] (θ skalad med L).
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

/// c · ∫ H'ᵢ H'ⱼ dx för Hermite-funktionerna, ordning [u1, s1, u2, s2].
fn hermite_grad(c: f64, l: f64) -> [[f64; 4]; 4] {
    let k = c / (30.0 * l);
    [
        [36.0 * k, 3.0 * l * k, -36.0 * k, 3.0 * l * k],
        [3.0 * l * k, 4.0 * l * l * k, -3.0 * l * k, -l * l * k],
        [-36.0 * k, -3.0 * l * k, 36.0 * k, -3.0 * l * k],
        [3.0 * l * k, -l * l * k, -3.0 * l * k, 4.0 * l * l * k],
    ]
}

/// Axialmassa i nodmassan (utan inre frihetsgrader).
const AXIAL_DIAG: f64 = 5.0 / 12.0;

/// Axial (eller vridnings-) massa för total massa `m`; `a` är diagonalens andel
/// i den konsistenta varianten.
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

/// Tillägg för de inre frihetsgraderna i ett B23/B33-element: matriser av
/// storlek (n + 2)², med elementets nodfrihetsgrader först (globala
/// koordinater) och de två inre sist. De *adderas* till elementets vanliga
/// matriser (`stiffness`, `mass` med MassKind::Consistent och
/// `geometric_stiffness`), som de korrigerar för den kubiska axialinterpolationen.
pub struct BeamInternal {
    pub n: usize,
    pub k: Vec<f64>,
    pub m: Vec<f64>,
    /// Geometrisk styvhet (bara om förskjutning gavs).
    pub kg: Option<Vec<f64>>,
}

/// Inre frihetsgrader för B23/B33 (`None` för andra element). `u` =
/// elementets förskjutning för den geometriska styvheten (normalkraften).
pub fn internal_dofs(name: &str, ctx: &ElemCtx, u: Option<&[f64]>) -> Option<Result<BeamInternal>> {
    let el = match name {
        "B23" => &super::beam_eb::B23,
        "B33" => &super::beam_eb::B33,
        _ => return None,
    };
    Some(build_internal(el, ctx, u))
}

fn build_internal(el: &EulerBeam, ctx: &ElemCtx, u: Option<&[f64]>) -> Result<BeamInternal> {
    // område beams3: krökt element (beam_eb_curved.rs)
    if let Some(c) = super::beam_eb_curved::Curved::of(el, ctx)? {
        return super::beam_eb_curved::internal(&c, el, ctx);
    }
    let d = el.data(ctx)?;
    let nd = d.n;
    let ne = nd + N_INTERNAL;
    let per = nd / 2;
    let l = d.l;
    // lokal ordning [u1, s1, u2, s2]
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
    // transformation: elementets T för nodfrihetsgraderna, identitet för de inre
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
            // normalkraft (drag positiv) = axialkraften i nod 2
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

/// Hermite-funktionernas första och andra derivata (map. fysisk koordinat) i
/// punkten x ∈ [0, 1], ordning [w1, s1, w2, s2] med s = lutning (skalad som θ).
fn hermite_d(x: f64, l: f64) -> ([f64; 4], [f64; 4]) {
    (
        [(-6.0 * x + 6.0 * x * x) / l, 1.0 - 4.0 * x + 3.0 * x * x, (6.0 * x - 6.0 * x * x) / l, 3.0 * x * x - 2.0 * x],
        [(-6.0 + 12.0 * x) / (l * l), (-4.0 + 6.0 * x) / l, (6.0 - 12.0 * x) / (l * l), (6.0 * x - 2.0) / l],
    )
}

/// Böjmomentets bidrag till den geometriska styvheten för B23 (lokala
/// koordinater): andra ordningens krökning κ₂ = −u''·v', med M = EI·v'' ur
/// elementets förskjutning. Termen kopplar de inre axialfrihetsgraderna till
/// utböjningen, så den finns bara med inre frihetsgrader. Formen är kontrollerad med
/// knäcklaster för en konsol med ändmoment respektive tvärkraft
/// (tests/cases/eigen/buckle_b23_moment, buckle_b23_tvarkraft). Termen används
/// inte för B33.
fn add_moment_terms(kg: &mut [f64], ne: usize, idx_u: &[usize; 4], d: &super::beam_eb::BeamData, ul: &[f64]) {
    let nd = d.n;
    let l = d.l;
    // (index för [v1, θ1, v2, θ2], tecken, EI)
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
