//! Reducerat integrerade kontinuumelement (C3D8R, CPE4R, CPS4R) i explicit
//! dynamik (NLGEOM=NO) med explicit hourglass-kontroll.
//!
//! Elementet räknas som i implicit analys (`elements/solid_higher_tech.rs`:
//! enpunkts "uniform strain" med materialuppdatering), men Standard-elementets
//! hourglass-styvhet byts mot den explicita:
//!
//! - styvheten per mod a och riktning d är c = 0,0125·(2/n)·G·V·Σᵢ|∇Nᵢ|²
//!   (n = 2 eller 3 rumsdimensioner; V med tjocklek), isotrop i riktningarna
//!   och med Flanagan–Belytschkos γₐ (h = ξη för fyrhörningar, ξη, ηζ, ζξ, ξηζ
//!   för hexaedrar). Kontrollerat med elementstyvheten för
//!   `HOURGLASS=STIFFNESS` (kub och förvrängd hexaeder, kvadrat och
//!   parallellogram).
//! - standard (RELAX STIFFNESS) är integralviskoelastisk som för skalen
//!   (`explicit_shell::relax`) men med tidskonstanten 1000 inkrement
//!   (avläst över 200–20 000 inkrement, samma för C3D8R och CPE4R).
//!
//! Explicit-tillståndet per element är [q, Q_v, Q] för alla (mod, riktning).

use super::dt::geometry;
use super::shell::{relax, Hourglass};
use crate::elements::{ElemCtx, Family};
use crate::model::{fail, Result};

/// Relaxationens tidskonstant (inkrement) för kontinuumelement.
pub const RELAX_TAU_SOLID: f64 = 1000.0;

/// Hourglass-vektorerna h (naturliga koordinater) för hexaeder respektive fyrhörning.
fn base_vectors(nn: usize) -> Vec<Vec<f64>> {
    if nn == 8 {
        let nat: [[f64; 3]; 8] = [[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]];
        vec![
            nat.iter().map(|p| p[0] * p[1]).collect(),
            nat.iter().map(|p| p[1] * p[2]).collect(),
            nat.iter().map(|p| p[2] * p[0]).collect(),
            nat.iter().map(|p| p[0] * p[1] * p[2]).collect(),
        ]
    } else {
        vec![vec![1.0, -1.0, 1.0, -1.0]]
    }
}

/// Elementet `name` hanteras här (Standard-statiken för hourglass-delen).
pub fn standard(name: &str) -> Option<&'static crate::elements::solid_higher::Solid> {
    use crate::elements::solid_higher as sh;
    match name {
        "C3D8R" => Some(&sh::C3D8R),
        "CPE4R" => Some(&sh::CPE4R),
        "CPS4R" => Some(&sh::CPS4R),
        _ => None,
    }
}

/// Inre kraft med explicit hourglass: elementets egen kraft (med
/// Standard-hourglass) minus Standard-hourglass plus explicit-hourglass.
/// Returnerar (kraft, nytt explicit-tillstånd, ökning av ALLAE).
pub fn hourglass_force(ctx: &ElemCtx, el: &crate::elements::solid_higher::Solid, family: Family, fint: &mut [f64], u: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, f64)> {
    let mat = ctx.material(el.name)?;
    let kstd = crate::elements::solid_higher::tech::hourglass_matrix(el, ctx, mat)?;
    let n = u.len();
    for i in 0..n {
        fint[i] -= (0..n).map(|j| kstd[i * n + j] * u[j]).sum::<f64>();
    }
    let Some(g) = geometry(family, el.name, &ctx.x) else {
        return fail(format!("element {} ({}): kan inte beräkna hourglass-vektorerna", ctx.label, el.name));
    };
    let nn = ctx.x.len();
    let nd = if nn == 8 { 3 } else { 2 };
    let thick = if nd == 2 { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 };
    let grad: Vec<[f64; 3]> = g.b.iter().map(|b| [b[0] / g.vol, b[1] / g.vol, b[2] / g.vol]).collect();
    let sbb: f64 = grad.iter().map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sum();
    let gmod = crate::elements::solid_higher::tech::shear_modulus_of(el, mat)?;
    let c = 0.0125 * (2.0 / nd as f64) * gmod * g.vol * thick * sbb;
    let hs = base_vectors(nn);
    let nm = hs.len() * nd;
    let mut st = vec![0.0; 3 * nm];
    let mut dae = 0.0;
    for (a, h) in hs.iter().enumerate() {
        let hx: [f64; 3] = std::array::from_fn(|k| (0..nn).map(|i| h[i] * ctx.x[i][k]).sum());
        let gam: Vec<f64> = (0..nn).map(|i| h[i] - (0..nd).map(|k| hx[k] * grad[i][k]).sum::<f64>()).collect();
        for d in 0..nd {
            let m = a * nd + d;
            let q: f64 = (0..nn).map(|i| gam[i] * u[i * nd + d]).sum();
            let q0 = old.get(m).copied().unwrap_or(0.0);
            let (qn, qv) = match mode {
                Hourglass::Stiffness => relax(mode, c, q, q0, 0.0),
                Hourglass::Relax => {
                    let qv0 = old.get(nm + m).copied().unwrap_or(0.0);
                    let qv = qv0 * (-1.0 / RELAX_TAU_SOLID).exp() + (1.0 - super::shell::RELAX_ALPHA) * (q - q0) * (-0.5 / RELAX_TAU_SOLID).exp();
                    (c * (super::shell::RELAX_ALPHA * q + qv), qv)
                }
            };
            st[m] = q;
            st[nm + m] = qv;
            st[2 * nm + m] = qn;
            dae += 0.5 * (qn + old.get(2 * nm + m).copied().unwrap_or(0.0)) * (q - q0);
            for i in 0..nn {
                fint[i * nd + d] += gam[i] * qn;
            }
        }
    }
    Ok((st, dae))
}
