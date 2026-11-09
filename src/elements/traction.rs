//! Surface traction on element faces: *DSLOAD with the load types TRVEC and TRSHR.
//!
//! `inp/keywords/dsload.rs` translates a traction on face Sn (SPOS/SNEG for shells = face 1)
//! to the element load `TRVECn` or `TRSHRn` with the direction in `extra`. The
//! equivalent nodal forces are fᵢ = ∫ Nᵢ t dA over the face in the reference
//! configuration, for all element types with 3D faces (3, 4, 6 or 8 nodes):
//!
//! - TRVEC: t = magnitude · d/|d| (fixed direction, global axes).
//! - TRSHR: t = magnitude · the unit projection of d onto the face's tangent plane
//!   at each integration point (pure shear traction).
//!
//! The forces go to U1–U3 of the face nodes. With NLGEOM the load is dead (reference
//! area and fixed direction).

use super::api::{ElemCtx, ElementType};
use super::common::{cross, dot, norm, scale, sub};
use crate::fem::dof::{U1, U3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub static TYPES: &[&dyn ElementType] = &[];

/// (shear, face number) for `TRVECn` / `TRSHRn`.
pub fn parse(kind: &str) -> Option<(bool, usize)> {
    let (shear, rest) = if let Some(r) = kind.strip_prefix("TRVEC") {
        (false, r)
    } else {
        (true, kind.strip_prefix("TRSHR")?)
    };
    Some((shear, rest.parse().ok()?))
}

/// Gauss points and weights on [0, 1] (3 points).
const G3: [(f64, f64); 3] = [(0.112_701_665_379_258_3, 5.0 / 18.0), (0.5, 8.0 / 18.0), (0.887_298_334_620_741_7, 5.0 / 18.0)];

/// Shape functions and derivatives (∂/∂r, ∂/∂s) of a face with n nodes in the
/// Abaqus node order (corners, then midside nodes).
fn shape(n: usize, r: f64, s: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    Some(match n {
        3 => (vec![1.0 - r - s, r, s], vec![-1.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0]),
        6 => {
            let l = 1.0 - r - s;
            (
                vec![l * (2.0 * l - 1.0), r * (2.0 * r - 1.0), s * (2.0 * s - 1.0), 4.0 * r * l, 4.0 * r * s, 4.0 * s * l],
                vec![1.0 - 4.0 * l, 4.0 * r - 1.0, 0.0, 4.0 * (l - r), 4.0 * s, -4.0 * s],
                vec![1.0 - 4.0 * l, 0.0, 4.0 * s - 1.0, -4.0 * r, 4.0 * r, 4.0 * (l - s)],
            )
        }
        4 | 8 => {
            // r, s ∈ [0, 1] → ξ, η ∈ [−1, 1]
            let (x, y) = (2.0 * r - 1.0, 2.0 * s - 1.0);
            let c = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let mut nv = vec![0.0; n];
            let mut dx = vec![0.0; n];
            let mut dy = vec![0.0; n];
            for i in 0..4 {
                let (a, b) = (c[i][0], c[i][1]);
                if n == 4 {
                    nv[i] = 0.25 * (1.0 + a * x) * (1.0 + b * y);
                    dx[i] = 0.25 * a * (1.0 + b * y);
                    dy[i] = 0.25 * b * (1.0 + a * x);
                } else {
                    nv[i] = 0.25 * (1.0 + a * x) * (1.0 + b * y) * (a * x + b * y - 1.0);
                    dx[i] = 0.25 * a * (1.0 + b * y) * (2.0 * a * x + b * y);
                    dy[i] = 0.25 * b * (1.0 + a * x) * (a * x + 2.0 * b * y);
                }
            }
            if n == 8 {
                // midside nodes 5–8 on the edges 1-2, 2-3, 3-4, 4-1
                nv[4] = 0.5 * (1.0 - x * x) * (1.0 - y);
                dx[4] = -x * (1.0 - y);
                dy[4] = -0.5 * (1.0 - x * x);
                nv[5] = 0.5 * (1.0 + x) * (1.0 - y * y);
                dx[5] = 0.5 * (1.0 - y * y);
                dy[5] = -y * (1.0 + x);
                nv[6] = 0.5 * (1.0 - x * x) * (1.0 + y);
                dx[6] = -x * (1.0 + y);
                dy[6] = 0.5 * (1.0 - x * x);
                nv[7] = 0.5 * (1.0 - x) * (1.0 - y * y);
                dx[7] = -0.5 * (1.0 - y * y);
                dy[7] = -y * (1.0 - x);
            }
            // chain rule to r, s
            (nv, dx.iter().map(|v| 2.0 * v).collect(), dy.iter().map(|v| 2.0 * v).collect())
        }
        _ => return None,
    })
}

/// Integration points (r, s, weight) on the face: 3×3 Gauss on the square, collapsed
/// (Duffy) on triangles.
fn points(n: usize) -> Vec<(f64, f64, f64)> {
    let tri = n == 3 || n == 6;
    let mut v = Vec::with_capacity(9);
    for &(a, wa) in &G3 {
        for &(b, wb) in &G3 {
            if tri {
                v.push((a, b * (1.0 - a), wa * wb * (1.0 - a)));
            } else {
                v.push((a, b, wa * wb));
            }
        }
    }
    v
}

/// Nodal forces (3 per face node) for a traction on the face with the coordinates `xf`.
pub fn face_forces(xf: &[[f64; 3]], shear: bool, mag: f64, dir: [f64; 3]) -> Result<Vec<[f64; 3]>> {
    let n = xf.len();
    let ld = norm(dir);
    if !(ld > 0.0) {
        return fail("the traction direction (TRVEC/TRSHR) is zero");
    }
    let d = scale(dir, 1.0 / ld);
    let mut f = vec![[0.0; 3]; n];
    for (r, s, w) in points(n) {
        let Some((nv, dr, ds)) = shape(n, r, s) else {
            return fail(format!("surface traction on faces with {n} nodes is not supported yet"));
        };
        let mut xr = [0.0; 3];
        let mut xs = [0.0; 3];
        for i in 0..n {
            for k in 0..3 {
                xr[k] += dr[i] * xf[i][k];
                xs[k] += ds[i] * xf[i][k];
            }
        }
        let av = cross(xr, xs);
        let da = norm(av);
        if !(da > 0.0) {
            return fail("surface traction on a face with zero area");
        }
        let t = if shear {
            let nn = scale(av, 1.0 / da);
            let p = sub(d, scale(nn, dot(d, nn)));
            let lp = norm(p);
            if lp < 1e-8 {
                return fail("TRSHR: the direction is normal to the surface (no shear component)");
            }
            scale(p, mag / lp)
        } else {
            scale(d, mag)
        };
        for i in 0..n {
            for k in 0..3 {
                f[i][k] += nv[i] * t[k] * da * w;
            }
        }
    }
    Ok(f)
}

/// The element vector for a TRVECn/TRSHRn load, or `None` for other load types.
pub fn element_load(ctx: &ElemCtx, et: &dyn ElementType, load: &DistLoad) -> Result<Option<Vec<f64>>> {
    let Some((shear, num)) = parse(&load.kind) else { return Ok(None) };
    let faces = et.faces();
    if num == 0 || num > faces.len() {
        return fail(format!("{} has no face {num} for surface traction", et.name()));
    }
    if et.dim() != 3 {
        return fail(format!("surface traction (TRVEC/TRSHR) is only supported for 3D elements and shells, not {}", et.name()));
    }
    let face = faces[num - 1];
    let xf: Vec<[f64; 3]> = face.iter().map(|&i| ctx.x[i]).collect();
    let dir = [0, 1, 2].map(|k| load.extra.get(k).copied().unwrap_or(0.0));
    let fnod = face_forces(&xf, shear, load.magnitude, dir)?;
    let mut fe = Vec::new();
    for i in 0..et.n_nodes() {
        let k = face.iter().position(|&j| j == i);
        for &d in et.node_dofs(i) {
            fe.push(match k {
                Some(k) if (U1..=U3).contains(&d) => fnod[k][(d - U1) as usize],
                _ => 0.0,
            });
        }
    }
    Ok(Some(fe))
}
