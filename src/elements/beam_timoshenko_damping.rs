//! Tvärskjuvningens del av Timoshenkobalkarnas styvhet (B21, B22, B31, B32)
//! för materialdämpningen i de modala stegen.
//!
//! Tvärskjuvstyvheten ingår inte i den styvhetsproportionella
//! dämpningen (*DAMPING, BETA) och den strukturella dämpningen
//! (*DAMPING, STRUCTURAL) för balkar i *COMPLEX FREQUENCY, *STEADY STATE
//! DYNAMICS, SUBSPACE PROJECTION och *DYNAMIC, SUBSPACE
//! (B21, `tests/cases/procedures2/cplx_b21_material`).
//! `shear_stiffness_matrix` ger skjuvdelen K − K(γ-styvhet = 0) i elementets
//! globala frihetsgrader, så att anroparen kan dra av den.
//! Inkluderas med `#[path]` från `beam_timoshenko.rs`.

use super::super::api::ElemCtx;
use super::{gauss, TimoBeam, B21, B22, B31, B32};
use crate::elements::common;
use crate::model::Result;

/// Balken med elementnamnet `name` (bara grundtyperna, inte hybrid- eller
/// rörvarianterna).
fn beam(name: &str) -> Option<&'static TimoBeam> {
    [&B21, &B22, &B31, &B32].into_iter().find(|b| b.name == name)
}

/// Skjuvdelen av elementstyvheten (globala frihetsgrader, n×n radvis) för en
/// Timoshenkobalk, `None` för andra element och för SECTION=NONLINEAR GENERAL.
pub fn shear_stiffness_matrix(name: &str, ctx: &ElemCtx) -> Option<Result<Vec<f64>>> {
    let b = beam(name)?;
    if super::nlg::is_nlg(ctx) {
        return None;
    }
    Some(shear_part(b, ctx))
}

fn shear_part(b: &TimoBeam, ctx: &ElemCtx) -> Result<Vec<f64>> {
    let g = b.geom(ctx)?;
    // tvärsnittsmatrisens skjuvdel: γ1, γ2 och deras koppling till
    // vridningen kring skjuvcentrum (section_matrix)
    let (k1, k2) = b.shear_stiffness(&g);
    let s = &g.sec;
    let mut d = [[0.0; 6]; 6];
    d[3][3] = k1 * s.x2s * s.x2s + k2 * s.x1s * s.x1s;
    d[3][4] = -k1 * s.x2s;
    d[4][3] = d[3][4];
    d[3][5] = k2 * s.x1s;
    d[5][3] = d[3][5];
    d[4][4] = k1;
    d[5][5] = k2;
    let nd = 6 * b.nn;
    let mut k = vec![0.0; nd * nd];
    for &(xi, w) in gauss(b.n_gauss()) {
        let (bm, j) = b.bmat(&g, xi);
        let wj = w * j;
        for r in 0..6 {
            for c in 0..6 {
                let drc = d[r][c];
                if drc == 0.0 {
                    continue;
                }
                for p in 0..nd {
                    let brp = bm[r * nd + p];
                    if brp == 0.0 {
                        continue;
                    }
                    for q in 0..nd {
                        k[p * nd + q] += wj * brp * drc * bm[c * nd + q];
                    }
                }
            }
        }
    }
    let t = b.transform(&g);
    Ok(b.reduce_matrix(&common::congruence(&k, &t, nd)))
}
