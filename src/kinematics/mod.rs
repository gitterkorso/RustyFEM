//! Kinematik för geometriskt olinjär analys (NLGEOM=YES).
//!
//! - `continuum`: uppdaterad Lagrange med Hughes–Winget-inkrement (CPS/CPE/C3D8
//!   och andra kontinuumelement som har integrationspunkter i referensläget).
//! - `truss`: stänger med logaritmisk töjning.
//! - `beam`: Euler–Bernoulli-balkar (B23, B33) med Hermiteinterpolation av läget.
//! - `hyperdual`: hyperduala tal för exakta derivator av töjningsenergin.
//! - `pressure`: följande tryck på deformerade ytor.
//! - `tensor`: 3×3-tensorer, polär uppdelning, rotationer.
//!
//! Elementen anropar hit från sin NLGEOM-rad i `internal_force`. Tillståndet
//! per element är kontinuumkärnans vanliga layout (spänning + materialtillstånd
//! per integrationspunkt) följt av ett tillägg som bara den här modulen läser:
//! logaritmisk töjning (LE) och integrerad töjning per integrationspunkt, sist
//! antalet punkter och en markör. Spänningen är Cauchy-spänning i globala axlar,
//! så S kan skrivas ut som vanligt.

pub mod bbar_tangent;
pub mod beam;
pub mod continuum;
pub mod fbar_hyper;
pub mod hyperdual;
pub mod pressure;
pub mod tensor;
pub mod truss;

use crate::elements::{Family, IpField};
use crate::fem::assembly::Prepared;
use crate::fem::output::FrameData;

/// Markör sist i elementtillståndet för NLGEOM-tillägget.
pub const MAGIC: f64 = -7.357_913_e77;

/// Lägger tillägget (12 värden per punkt: LE, integrerad töjning) sist i tillståndet.
pub fn push_ext(state: &mut Vec<f64>, ext: &[f64], nips: usize) {
    state.extend_from_slice(ext);
    state.push(nips as f64);
    state.push(MAGIC);
}

/// Tillägget (LE för alla punkter, sedan integrerad töjning) om det finns.
pub fn ext(state: &[f64]) -> Option<&[f64]> {
    let n = state.len();
    if n < 2 || state[n - 1] != MAGIC {
        return None;
    }
    let nips = state[n - 2] as usize;
    let len = 12 * nips;
    if n < len + 2 {
        return None;
    }
    Some(&state[n - 2 - len..n - 2])
}

/// Ersätter småtöjningsfältet E med logaritmisk töjning LE (utdata vid NLGEOM)
/// för element som har NLGEOM-tillstånd.
pub fn fix_frame_output(p: &Prepared, frame: &mut FrameData, state: &[Vec<f64>]) {
    for (e, fields) in frame.elem.iter_mut().enumerate() {
        let Some(x) = state.get(e).and_then(|s| ext(s)) else { continue };
        let nips = x.len() / 12;
        let fam = p.model.elements[e].etype.family();
        let labels: &[&str] = match fam {
            Family::Truss => &["LE11"],
            Family::Continuum2D | Family::Axisymmetric => &["LE11", "LE22", "LE33", "LE12"],
            Family::Continuum3D => &["LE11", "LE22", "LE33", "LE12", "LE13", "LE23"],
            _ => continue,
        };
        let mut le = IpField::new("LE", labels);
        for q in 0..nips {
            let v = &x[6 * q..6 * q + 6];
            // skjuvkomponenter som ingenjörstöjning
            let vals = match labels.len() {
                1 => vec![v[0]],
                4 => vec![v[0], v[1], v[2], 2.0 * v[3]],
                _ => vec![v[0], v[1], v[2], 2.0 * v[3], 2.0 * v[4], 2.0 * v[5]],
            };
            le.points.push((q + 1, 0, vals));
        }
        fields.retain(|f| f.name != "E" && f.name != "LE");
        fields.push(le);
    }
}
