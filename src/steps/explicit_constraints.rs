//! Kinematic constraints in explicit dynamics (included with `#[path]` from
//! `explicit.rs`): the locked components of connector elements
//! (`Interaction::kinematic_constraints`).
//!
//! The central difference method eliminates no constraints, so the constraint forces λ
//! are computed in each increment so that the predicted position satisfies the constraints:
//! with the rows C and the values g at the position u and the free acceleration a_f,
//! g(u') ≈ g(u) + h C v + κ C a = 0, a = a_f − M⁻¹ Cᵀ λ gives the system
//! (C M⁻¹ Cᵀ) λ = (g(u) + h C v)/κ + C a_f (Lagrange multipliers at
//! position level, so that the constraints are enforced exactly). The force
//! Cᵀ λ is added to the internal force (and is thereby included in the reactions), λ
//! is written out as the connector elements' CTF/CTM. Prescribed degrees of freedom have
//! no mobility (M⁻¹ = 0). The connector element state is committed after
//! each increment (`commit`).

use super::mass::LumpedMass;
use crate::fem::assembly::{DofMap, Prepared, NONE};
use crate::fem::dof::{self, NodeVec, SLOTS};
use crate::model::Result;

/// Computes the constraint forces and adds them to `fint` (the acceleration is
/// M⁻¹(Σ `add` − `fint` − Σ `sub`)). `u` is the position where
/// the acceleration applies, `v` the velocity that carries the position forward, `h` the next
/// increment and `kappa` the weight of the acceleration in the next position. Returns true if
/// `fint` changed (the acceleration must then be recomputed).
#[allow(clippy::too_many_arguments)]
pub fn project(p: &Prepared, dm: &DofMap, mass: &LumpedMass, nlgeom: bool, u: &[NodeVec], v: &[NodeVec], h: f64, kappa: f64, add: &[&[NodeVec]], fint: &mut [NodeVec], sub: &[&[NodeVec]]) -> Result<bool> {
    let mut rows: Vec<(usize, Vec<(usize, usize, f64)>, f64)> = Vec::new();
    for (ii, it) in p.interactions.iter().enumerate() {
        for (r, g) in it.kinematic_constraints(u, nlgeom) {
            let r: Vec<(usize, usize, f64)> = r.into_iter().map(|(n, d, c)| (n, dof::slot_of(d), c)).filter(|t| t.2 != 0.0).collect();
            rows.push((ii, r, g));
        }
    }
    if rows.is_empty() || kappa <= 0.0 {
        return Ok(false);
    }
    let nc = rows.len();
    if nc > 3000 {
        return crate::model::fail(format!("too many kinematic constraints in explicit dynamics ({nc})"));
    }
    let free = |n: usize, s: usize| dm.eq[n][s] != NONE;
    // free acceleration a_f = M⁻¹ (Σ add − fint − Σ sub)
    let accel = |n: usize| -> NodeVec {
        let r: NodeVec = std::array::from_fn(|s| add.iter().map(|o| o[n][s]).sum::<f64>() - fint[n][s] - sub.iter().map(|o| o[n][s]).sum::<f64>());
        mass.solve(n, &r, |s| free(n, s))
    };
    // M⁻¹ cₖ per constraint (nodal)
    let minv: Vec<Vec<(usize, NodeVec)>> = rows
        .iter()
        .map(|(_, r, _)| {
            let mut nodes: Vec<usize> = r.iter().map(|t| t.0).collect();
            nodes.sort_unstable();
            nodes.dedup();
            nodes
                .into_iter()
                .map(|n| {
                    let mut c = [0.0; SLOTS];
                    for &(m, s, x) in r {
                        if m == n {
                            c[s] += x;
                        }
                    }
                    (n, mass.solve(n, &c, |s| free(n, s)))
                })
                .collect()
        })
        .collect();
    let mut a = vec![0.0; nc * nc];
    let mut b = vec![0.0; nc];
    for i in 0..nc {
        let (_, ri, gi) = &rows[i];
        let mut cv = 0.0;
        let mut ca = 0.0;
        for &(n, s, c) in ri {
            cv += c * v[n][s];
            ca += c * accel(n)[s];
        }
        b[i] = (gi + h * cv) / kappa + ca;
        for j in 0..nc {
            let mut x = 0.0;
            for &(n, s, c) in ri {
                if let Some((_, mv)) = minv[j].iter().find(|q| q.0 == n) {
                    x += c * mv[s];
                }
            }
            a[i * nc + j] = x;
        }
    }
    let lam = solve_dense(&mut a, &mut b, nc);
    for (i, (_, r, _)) in rows.iter().enumerate() {
        for &(n, s, c) in r {
            fint[n][s] += c * lam[i];
        }
    }
    // the forces to the interactions (output)
    let mut k = 0;
    while k < nc {
        let ii = rows[k].0;
        let start = k;
        while k < nc && rows[k].0 == ii {
            k += 1;
        }
        p.interactions[ii].set_constraint_forces(&lam[start..k]);
    }
    Ok(true)
}

/// Commits the connector element state after an increment.
pub fn commit(p: &Prepared) {
    for it in p.interactions.iter().filter(|it| it.name().starts_with("*CONNECTOR")) {
        it.commit();
    }
}

/// Gaussian elimination with pivoting; singular (redundant) constraints give λ = 0.
fn solve_dense(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max).max(1e-300);
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs())).unwrap();
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            b.swap(k, p);
        }
        let d = a[k * n + k];
        if d.abs() <= 1e-12 * scale {
            continue;
        }
        for i in k + 1..n {
            let f = a[i * n + k] / d;
            if f != 0.0 {
                for j in k..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
                b[i] -= f * b[k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let d = a[k * n + k];
        if d.abs() <= 1e-12 * scale {
            continue;
        }
        let mut s = b[k];
        for j in k + 1..n {
            s -= a[k * n + j] * x[j];
        }
        x[k] = s / d;
    }
    x
}
