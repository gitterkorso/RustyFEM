//! Trusses (T2D2, T3D2) with large displacements and strains.
//!
//! The strain is logarithmic, ε = ln(l/L), and the stress is Cauchy stress.
//! The cross-sectional area is updated so that the volume is preserved, A = A₀ L / l, which is
//! standard for trusses (verified: 20 % stretch gives RF = S·A₀/1.2).
//! The axial force is N = σ A and the internal force N·[−e, e] with current direction e.

use crate::elements::common;
use crate::elements::{ElemCtx, ElemResponse};
use crate::materials::{MatInput, StressMode};
use crate::model::{fail, Result};

fn geom(x: &[[f64; 3]], u: &[f64], dim: usize) -> ([f64; 3], f64) {
    let mut d = common::sub(x[1], x[0]);
    for i in 0..dim {
        d[i] += u[dim + i] - u[i];
    }
    let l = common::norm(d);
    (common::scale(d, 1.0 / l.max(1e-300)), l)
}

#[allow(clippy::too_many_arguments)]
pub fn response(
    ctx: &ElemCtx,
    name: &str,
    dim: usize,
    area: f64,
    u: &[f64],
    du: &[f64],
    state_old: &[f64],
    state_new: &mut Vec<f64>,
    tangent: bool,
) -> Result<ElemResponse> {
    let mat = ctx.material(name)?;
    let mode = StressMode::Uniaxial;
    let stride = common::ip_stride(mode, mat);
    let init;
    let old = if state_old.len() == stride || (state_old.len() > stride && super::ext(state_old).is_some()) {
        &state_old[..stride]
    } else {
        init = common::init_state(1, mode, mat);
        &init[..]
    };
    let ll = common::norm(common::sub(ctx.x[1], ctx.x[0]));
    let un: Vec<f64> = u.iter().zip(du).map(|(a, b)| a - b).collect();
    let (e, l) = geom(&ctx.x, u, dim);
    let (_, l0) = geom(&ctx.x, &un, dim);
    if !(l > 0.0) || !(l0 > 0.0) {
        return fail(format!("{name}: the truss has reached zero length"));
    }
    let eps = [(l / ll).ln()];
    let deps = [(l / l0).ln()];
    let mut mnew = old[1..].to_vec();
    let inp = MatInput {
        mode,
        strain: &eps,
        dstrain: &deps,
        stress_old: &old[..1],
        time: ctx.time,
        dt: ctx.dt,
        temperature: 0.0,
        dtemp: 0.0,
        defgrad: None,
    };
    let out = mat.update(&inp, &old[1..], &mut mnew)?;
    let sig = out.stress[0];
    let a0l = area * ll;
    let n = sig * a0l / l;
    let nd = 2 * dim;
    let mut fint = vec![0.0; nd];
    for i in 0..dim {
        fint[i] = -n * e[i];
        fint[dim + i] = n * e[i];
    }
    let k = tangent.then(|| {
        let dn = a0l / (l * l) * (out.tangent[0] - sig);
        let mut k = vec![0.0; nd * nd];
        for i in 0..dim {
            for j in 0..dim {
                let v = dn * e[i] * e[j] + n / l * ((i == j) as u8 as f64 - e[i] * e[j]);
                for (a, b, s) in [(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)] {
                    k[(a * dim + i) * nd + b * dim + j] += s * v;
                }
            }
        }
        k
    });
    state_new.clear();
    state_new.push(sig);
    state_new.extend(mnew);
    let mut ext = vec![0.0; 12];
    ext[0] = eps[0];
    ext[6] = eps[0];
    super::push_ext(state_new, &ext, 1);
    Ok(ElemResponse { fint, k })
}
