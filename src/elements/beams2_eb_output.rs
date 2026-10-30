//! Section forces for the Euler–Bernoulli beams B23 and B33 (area beams2).
//!
//! Output is SF1 (axial force) and SM (bending moment, for B33 also
//! torque) at three integration points (the Gauss points ξ = 0, ∓√(3/5) along
//! the element); the transverse shear forces are zero in the output (not available for B23/B33).
//! The components are named as for B21/B31 (SF11, SF22, SF12 for planar beams). The moments are computed from
//! the curvature of the cubic (Hermite) deflection at the point, with the same
//! sign as the Timoshenko beams: κ1 = θ1', κ2 = θ2' with θ1, θ2 the rotations
//! about n1, n2 (for B23, n1 = (0, 0, −1), so SM1 = −EI v'').
//! The stiffnesses (EA, EI, GJ) are read from the element's local stiffness matrix.

use crate::elements::api::{ElemCtx, IpField};
use crate::elements::beam_eb::EulerBeam;
use crate::elements::common;
use crate::elements::ElementType;
use crate::model::Result;

/// Second derivative of the Hermite polynomials (v1, v1', v2, v2') at x ∈ [0, L].
fn hermite_dd(x: f64, l: f64) -> [f64; 4] {
    [-6.0 / (l * l) + 12.0 * x / l.powi(3), -4.0 / l + 6.0 * x / (l * l), 6.0 / (l * l) - 12.0 * x / l.powi(3), -2.0 / l + 6.0 * x / (l * l)]
}

/// Positions of the integration points (fraction of the length).
pub const POINTS: [f64; 3] = [0.112_701_665_379_258_31, 0.5, 0.887_298_334_620_741_7];

pub fn output(el: &EulerBeam, ctx: &ElemCtx, u: &[f64]) -> Result<Vec<IpField>> {
    let d = el.data(ctx)?;
    let ul = common::matvec(&d.t, u);
    let (l, n, k) = (d.l, d.n, &d.k_local);
    let kk = |i: usize, j: usize| k[i * n + j];
    if el.dim() == 2 {
        let mut sf = IpField::new("SF", &["SF11", "SF22", "SF12"]);
        let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
        let (ea, ei) = (kk(0, 0) * l, kk(1, 1) * l.powi(3) / 12.0);
        let nf = ea * (ul[3] - ul[0]) / l;
        for (q, &f) in POINTS.iter().enumerate() {
            let h = hermite_dd(f * l, l);
            let v2 = h[0] * ul[1] + h[1] * ul[2] + h[2] * ul[4] + h[3] * ul[5];
            sf.points.push((q + 1, 0, vec![nf, 0.0, 0.0]));
            sm.points.push((q + 1, 0, vec![-ei * v2, 0.0, 0.0]));
        }
        return Ok(vec![sf, sm]);
    }
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let (ea, gj) = (kk(0, 0) * l, kk(3, 3) * l);
    let (ei22, ei11) = (kk(1, 1) * l.powi(3) / 12.0, kk(2, 2) * l.powi(3) / 12.0);
    let nf = ea * (ul[6] - ul[0]) / l;
    let tq = gj * (ul[9] - ul[3]) / l;
    for (q, &f) in POINTS.iter().enumerate() {
        let h = hermite_dd(f * l, l);
        // x–y plane: v (local y = n1), θz = v'; x–z plane: w (local z = n2), θy = −w'
        let vdd = h[0] * ul[1] + h[1] * ul[5] + h[2] * ul[7] + h[3] * ul[11];
        let wdd = h[0] * ul[2] - h[1] * ul[4] + h[2] * ul[8] - h[3] * ul[10];
        sf.points.push((q + 1, 0, vec![nf, 0.0, 0.0]));
        sm.points.push((q + 1, 0, vec![-ei11 * wdd, ei22 * vdd, tq]));
    }
    Ok(vec![sf, sm])
}
