//! Snittkrafter för Euler–Bernoullibalkarna B23 och B33 (område beams2).
//!
//! Utdata är SF1 (normalkraft) och SM (böjmoment, för B33 även
//! vridmoment) i tre integrationspunkter (gausspunkterna ξ = 0, ∓√(3/5) längs
//! elementet); tvärkrafterna är noll i utdata (finns inte för B23/B33).
//! Komponenterna heter som för B21/B31 (SF11, SF22, SF12 för plana balkar). Momenten räknas ur
//! krökningen av den kubiska (Hermite) utböjningen i punkten, med samma
//! tecken som Timoshenkobalkarna: κ1 = θ1', κ2 = θ2' med θ1, θ2 rotationerna
//! kring n1, n2 (för B23 är n1 = (0, 0, −1), alltså SM1 = −EI v'').
//! Styvheterna (EA, EI, GJ) läses ur elementets lokala styvhetsmatris.

use crate::elements::api::{ElemCtx, IpField};
use crate::elements::beam_eb::EulerBeam;
use crate::elements::common;
use crate::elements::ElementType;
use crate::model::Result;

/// Andraderivatan av Hermitepolynomen (v1, v1', v2, v2') i x ∈ [0, L].
fn hermite_dd(x: f64, l: f64) -> [f64; 4] {
    [-6.0 / (l * l) + 12.0 * x / l.powi(3), -4.0 / l + 6.0 * x / (l * l), 6.0 / (l * l) - 12.0 * x / l.powi(3), -2.0 / l + 6.0 * x / (l * l)]
}

/// Integrationspunkternas lägen (andel av längden).
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
        // x–y-planet: v (lokal y = n1), θz = v'; x–z-planet: w (lokal z = n2), θy = −w'
        let vdd = h[0] * ul[1] + h[1] * ul[5] + h[2] * ul[7] + h[3] * ul[11];
        let wdd = h[0] * ul[2] - h[1] * ul[4] + h[2] * ul[8] - h[3] * ul[10];
        sf.points.push((q + 1, 0, vec![nf, 0.0, 0.0]));
        sm.points.push((q + 1, 0, vec![-ei11 * wdd, ei22 * vdd, tq]));
    }
    Ok(vec![sf, sm])
}
