//! External loads with NLGEOM: follower pressure (*DLOAD Pn) on the deformed
//! surface, and the load's contribution to the tangent (load stiffness).
//!
//! The pressure acts normal to the current surface and on its current area,
//! so the equivalent nodal forces are computed with the element's usual
//! `distributed_load` but with deformed nodal coordinates. Body forces (BX,
//! GRAV, ...) and beam loads (PX, PY, PZ) are "dead" loads and are computed in
//! the reference configuration (mass is conserved). The load stiffness −∂f/∂u is computed with central
//! differences per loaded element and symmetrized (for a symmetric solver).

use crate::elements::ElemCtx;
use crate::fem::assembly::Prepared;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::DistLoad;
use crate::model::{FemError, Result};

/// Does the load follow the surface? (Pn for continuum elements.)
pub fn is_follower(kind: &str) -> bool {
    kind.len() > 1 && kind.starts_with('P') && kind[1..].chars().all(|c| c.is_ascii_digit())
}

/// Does the load follow the surface? Pn on continuum elements and P on shells and membranes
/// (area nlgeom_struct).
fn follows(p: &Prepared, l: &DistLoad) -> bool {
    use crate::elements::Family;
    is_follower(&l.kind) || (l.kind == "P" && matches!(p.model.elements[l.elem].etype.family(), Family::Shell | Family::Membrane))
}

fn deformed_ctx<'a>(p: &'a Prepared, e: usize, u: &[NodeVec], time: f64) -> ElemCtx<'a> {
    let mut ctx = p.ctx(e, true, time, 0.0);
    for (k, &n) in p.model.elements[e].nodes.iter().enumerate() {
        for d in 0..3 {
            ctx.x[k][d] += u[n][d];
        }
    }
    ctx
}

fn elem_load(p: &Prepared, l: &DistLoad, u: &[NodeVec], time: f64) -> Result<Vec<f64>> {
    let e = l.elem;
    let m = p.model;
    let ue = p.gather(e, u);
    // user load PnNU (DLOAD, user/loads.rs) in the reference configuration
    if let Some(fe) = crate::user::loads::dload(p, l, time).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))? {
        return Ok(fe);
    }
    let r = if follows(p, l) {
        let ctx = deformed_ctx(p, e, u, time);
        m.elements[e].etype.distributed_load(&ctx, l, &ue)
    } else {
        let ctx = p.ctx(e, true, time, 0.0);
        m.elements[e].etype.distributed_load(&ctx, l, &ue)
    };
    r.map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))
}

/// Load stiffness for a loaded element: (element, −∂f/∂u symmetrized).
pub type LoadStiffness = Vec<(usize, Vec<f64>)>;

/// Nodal loads at displacement `u` and (if `stiffness`) the load stiffness.
pub fn external(
    p: &Prepared,
    cloads: &[(usize, usize, f64)],
    dloads: &[DistLoad],
    u: &[NodeVec],
    time: f64,
    stiffness: bool,
) -> Result<(Vec<NodeVec>, LoadStiffness)> {
    let m = p.model;
    let mut f = vec![[0.0; SLOTS]; m.nodes.len()];
    for &(n, s, v) in cloads {
        f[n][s] += v;
    }
    let mut ks = Vec::new();
    for l in dloads {
        let fe = elem_load(p, l, u, time)?;
        for (a, &(n, s)) in p.edofs[l.elem].iter().enumerate() {
            f[n][s] += fe[a];
        }
        if stiffness && follows(p, l) && l.magnitude != 0.0 {
            let dofs = &p.edofs[l.elem];
            let nd = dofs.len();
            let x = &m.elements[l.elem].nodes;
            let size = x.iter().map(|&n| m.nodes[n]).fold(0.0f64, |acc, q| acc.max(crate::elements::common::norm(crate::elements::common::sub(q, m.nodes[x[0]]))));
            let h = 1e-7 * size.max(1e-30);
            let mut k = vec![0.0; nd * nd];
            let mut up = u.to_vec();
            for (b, &(n, s)) in dofs.iter().enumerate() {
                if s >= 3 {
                    continue;
                }
                let u0 = up[n][s];
                up[n][s] = u0 + h;
                let fp = elem_load(p, l, &up, time)?;
                up[n][s] = u0 - h;
                let fm = elem_load(p, l, &up, time)?;
                up[n][s] = u0;
                for a in 0..nd {
                    k[a * nd + b] = -(fp[a] - fm[a]) / (2.0 * h);
                }
            }
            super::continuum::symmetrize(&mut k, nd);
            ks.push((l.elem, k));
        }
    }
    Ok((f, ks))
}
