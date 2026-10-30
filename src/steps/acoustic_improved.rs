//! Area gaps1: *SIMPEDANCE/*IMPEDANCE, NONREFLECTING=IMPROVED in acoustic
//! *DYNAMIC (included from `acoustic.rs`).
//!
//! The plane wave condition with arbitrary angle of incidence (Sandler 1998):
//!
//! ```text
//!   n⁻·∇p / ρ_f = cos θ/√(ρ_f K_f) (ṗ + γ/(2ρ_f) p),   cos θ = |∂p/∂n| / |∇p|
//! ```
//!
//! cos θ is computed at each face integration point from the element pressure gradient with
//! the predicted nodal pressures p + Δt ṗ + ½Δt² p̈ (the pressures are updated with
//! explicit central differences) and is 1 where the gradient is zero. The boundary
//! damping ∫ cos θ N N/√(ρK) dS and stiffness ∫ cos θ N N γ/(2ρ√(ρK)) dS are therefore
//! rebuilt in each increment (`matrices`).

use super::*;

/// An IMPROVED boundary: (element, face index).
pub(super) struct Improved {
    pub faces: Vec<(usize, usize)>,
}

/// Natural coordinates for the point `xp` in the element (Newton with
/// difference quotients for the Jacobian of the mapping).
fn natural(a: &AcousticElem, x: &[[f64; 3]], xp: [f64; 3]) -> Result<[f64; 3]> {
    let dim = a.geom.dim();
    let shape = a.geom.shape;
    let pos = |xi: [f64; 3]| -> Result<[f64; 3]> {
        let ip = crate::elements::heat::ip_at(shape, x, xi, 1.0, 1.0, false)?;
        let mut p = [0.0; 3];
        for (i, xn) in x.iter().enumerate() {
            for d in 0..3 {
                p[d] += ip.n[i] * xn[d];
            }
        }
        Ok(p)
    };
    let tri = matches!(shape, crate::elements::heat::Shape::Tri3 | crate::elements::heat::Shape::Tet4 | crate::elements::heat::Shape::Wedge6);
    let mut xi = if tri { [0.25, 0.25, 0.0] } else { [0.0; 3] };
    for _ in 0..30 {
        let p0 = pos(xi)?;
        let r: Vec<f64> = (0..dim).map(|d| xp[d] - p0[d]).collect();
        if r.iter().map(|v| v.abs()).fold(0.0, f64::max) < 1e-13 * (1.0 + xp.iter().map(|v| v.abs()).fold(0.0, f64::max)) {
            break;
        }
        let h = 1e-6;
        let mut jm = vec![0.0; dim * dim];
        for c in 0..dim {
            let mut xh = xi;
            xh[c] += h;
            let ph = pos(xh)?;
            for rr in 0..dim {
                jm[rr * dim + c] = (ph[rr] - p0[rr]) / h;
            }
        }
        let Some(dx) = solve::dense_solve(&jm, &r, dim, 1) else { break };
        for c in 0..dim {
            xi[c] += dx[c];
        }
    }
    Ok(xi)
}

/// The face normal (unit vector) from the corner nodes.
fn face_normal(a: &AcousticElem, x: &[[f64; 3]], fi: usize) -> [f64; 3] {
    let fl = a.geom.faces()[fi];
    let n = if a.geom.dim() == 2 {
        let t = [x[fl[1]][0] - x[fl[0]][0], x[fl[1]][1] - x[fl[0]][1]];
        [t[1], -t[0], 0.0]
    } else {
        let (d1, d2) = if fl.len() >= 4 && !matches!(a.geom.shape, crate::elements::heat::Shape::Tet4 | crate::elements::heat::Shape::Tet10) {
            (sub3(x[fl[2]], x[fl[0]]), sub3(x[fl[3]], x[fl[1]]))
        } else {
            (sub3(x[fl[1]], x[fl[0]]), sub3(x[fl[2]], x[fl[0]]))
        };
        [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]]
    };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 { [n[0] / l, n[1] / l, n[2] / l] } else { n }
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

impl Improved {
    /// The boundary damping and stiffness matrices (all degrees of freedom) for
    /// the nodal pressures `p` (per node).
    pub(super) fn matrices(&self, sys: &Sys, p: &[f64]) -> Result<(Gen, Gen)> {
        let m = sys.p.model;
        let na = sys.n_all();
        let mut c = Gen::new(na);
        let mut k = Gen::new(na);
        for &(e, fi) in &self.faces {
            let el = &m.elements[e];
            let Some(a) = ac::lookup(el.etype.name()) else { continue };
            let props = material_props(m, e)?;
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&nd| m.nodes[nd]).collect();
            let pe: Vec<f64> = el.nodes.iter().map(|&nd| p[nd]).collect();
            let nrm = face_normal(a, &x, fi);
            let cos = |xp: [f64; 3]| -> f64 {
                let Ok(xi) = natural(a, &x, xp) else { return 1.0 };
                let Ok(ip) = crate::elements::heat::ip_at(a.geom.shape, &x, xi, 1.0, 1.0, false) else { return 1.0 };
                let mut g = [0.0; 3];
                for (i, d) in ip.dndx.iter().enumerate() {
                    for q in 0..3 {
                        g[q] += d[q] * pe[i];
                    }
                }
                let gl = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                if gl <= 1e-300 {
                    return 1.0;
                }
                ((g[0] * nrm[0] + g[1] * nrm[1] + g[2] * nrm[2]).abs() / gl).min(1.0)
            };
            let he = face_matrix_w(a, &x, fi, thickness(m, e), &cos)?;
            let z = (props.density * props.bulk).sqrt();
            let gamma = props.drag_at(None);
            let ce: Vec<f64> = he.iter().map(|v| v / z).collect();
            let ke: Vec<f64> = he.iter().map(|v| v * gamma / (2.0 * props.density * z)).collect();
            sys.scatter(e, &ce, &mut c);
            if gamma != 0.0 {
                sys.scatter(e, &ke, &mut k);
            }
        }
        c.finish();
        k.finish();
        Ok((c, k))
    }
}
