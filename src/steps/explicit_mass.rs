//! Diagonal (lumped) mass for explicit dynamics.
//!
//! - Continuum, trusses and membranes: the row-summed element mass
//!   (`MassKind::Lumped`, ρ∫Nᵢ dV per node).
//! - Shells (S4R, S4, S3R, S3): the translational mass mᵢ = ρt∫Nᵢ dA and an isotropic
//!   rotary inertia Iᵢ = mᵢ (l²/16 + A/12), independent of the thickness and
//!   ν. For quadrilaterals l is A divided by the largest distance between
//!   the midpoints of opposite edges, and for triangles the shortest edge.
//!   The formula is checked with the acceleration in the first frame of
//!   unit loads on free elements of different shapes (square, rectangle,
//!   parallelogram, irregular quadrilateral, right, equilateral and obtuse
//!   triangle) for t ≤ 0.1·l.
//! - Beams (B31, B21, ...): the element's diagonal mass with the physical
//!   rotary inertia ρ∫Nᵢ dx · (I₁₁ + I₂₂, I₁₁, I₂₂) in the local beam axes
//!   The rotary inertia is stored as a full
//!   3×3 block per node in global axes.
//! - *MASS (MASS elements) and *ROTARY INERTIA (ROTARYI elements) from
//!   the interactions.
//!
//! The mass can be scaled element by element (mass scaling): the factor multiplies
//! the element density.

use crate::elements::{Family, MassKind};
use crate::fem::assembly::Prepared;
use crate::fem::dof::{self, NodeVec, SLOTS};
use crate::model::{FemError, Result};

/// Diagonal mass per node and slot as well as (for beams and ROTARYI) full
/// rotation blocks in global axes.
#[derive(Debug, Clone)]
pub struct LumpedMass {
    pub diag: Vec<NodeVec>,
    /// Rotary inertia (slots 3–5) as a full block; `None` = diagonal in `diag`.
    pub rot: Vec<Option<[[f64; 3]; 3]>>,
    /// The inverse of `rot`.
    rot_inv: Vec<Option<[[f64; 3]; 3]>>,
}

impl LumpedMass {
    /// M⁻¹ f for node `n`; degrees of freedom without mass give 0.
    pub fn solve(&self, n: usize, f: &NodeVec, free: impl Fn(usize) -> bool) -> NodeVec {
        let mut a = [0.0; SLOTS];
        for s in 0..SLOTS {
            if free(s) && self.diag[n][s] > 0.0 {
                a[s] = f[s] / self.diag[n][s];
            }
        }
        if let Some(ri) = &self.rot_inv[n] {
            for i in 0..3 {
                a[3 + i] = if free(3 + i) { (0..3).map(|j| ri[i][j] * f[3 + j]).sum() } else { 0.0 };
            }
        }
        a
    }

    /// M v for node `n`.
    pub fn mul(&self, n: usize, v: &NodeVec) -> NodeVec {
        let mut r = [0.0; SLOTS];
        for s in 0..SLOTS {
            r[s] = self.diag[n][s] * v[s];
        }
        if let Some(b) = &self.rot[n] {
            for i in 0..3 {
                r[3 + i] = (0..3).map(|j| b[i][j] * v[3 + j]).sum();
            }
        }
        r
    }

    /// Kinetic energy ½ vᵀ M v.
    pub fn kinetic(&self, v: &[NodeVec]) -> f64 {
        let mut e = 0.0;
        for (n, vn) in v.iter().enumerate() {
            let mv = self.mul(n, vn);
            e += (0..7).map(|s| mv[s] * vn[s]).sum::<f64>();
        }
        0.5 * e
    }

    /// Total translational mass (sum over nodes of slot 0).
    pub fn total(&self) -> f64 {
        self.diag.iter().map(|d| d[0]).sum()
    }
}

/// Rotary inertia of the shell per unit mass, l²/16 + A/12 (see the module documentation).
pub fn shell_rotary_factor(x: &[[f64; 3]]) -> f64 {
    use crate::elements::common::{cross, norm, sub};
    let dist = |a: [f64; 3], b: [f64; 3]| norm(sub(a, b));
    match x.len() {
        3 => {
            let area = 0.5 * norm(cross(sub(x[1], x[0]), sub(x[2], x[0])));
            let l = dist(x[0], x[1]).min(dist(x[1], x[2])).min(dist(x[2], x[0]));
            l * l / 16.0 + area / 12.0
        }
        4 => {
            let area = 0.5 * norm(cross(sub(x[2], x[0]), sub(x[3], x[1])));
            let mid = |a: usize, b: usize| -> [f64; 3] { std::array::from_fn(|c| 0.5 * (x[a][c] + x[b][c])) };
            let d1 = dist(mid(0, 1), mid(2, 3));
            let d2 = dist(mid(1, 2), mid(3, 0));
            let l = area / d1.max(d2);
            l * l / 16.0 + area / 12.0
        }
        _ => 0.0,
    }
}

/// Builds the diagonal mass. `scale[e]` multiplies the element density
/// (mass scaling, empty = 1).
pub fn lumped(p: &Prepared, scale: &[f64]) -> Result<LumpedMass> {
    let m = p.model;
    let nn = m.nodes.len();
    let mut diag = vec![[0.0; SLOTS]; nn];
    let mut rot: Vec<Option<[[f64; 3]; 3]>> = vec![None; nn];
    let dim = m.dim();
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        if ctx.density <= 0.0 {
            continue;
        }
        let f = scale.get(e).copied().unwrap_or(1.0);
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        let me = el.etype.mass(&ctx, MassKind::Lumped).map_err(wrap)?;
        let dl = &p.edofs[e];
        let nd = dl.len();
        if me.len() != nd * nd {
            continue;
        }
        let shell = el.etype.family() == Family::Shell;
        let r_shell = if shell { shell_rotary_factor(&ctx.x) } else { 0.0 };
        let beam = el.etype.family() == Family::Beam && dim == 3;
        // translational mass per local node (for the rotary inertia of the shell)
        for (a, &(n, s)) in dl.iter().enumerate() {
            if (3..6).contains(&s) && (shell || beam) {
                continue;
            }
            diag[n][s] += f * me[a * nd + a];
        }
        if shell {
            for (a, &(n, s)) in dl.iter().enumerate() {
                if s == 0 {
                    let mt = f * me[a * nd + a];
                    for r in 3..6 {
                        if p.active[n] & (1 << r) != 0 {
                            diag[n][r] += mt * r_shell;
                        }
                    }
                }
            }
        }
        if beam {
            // rotation block per node
            for (a, &(n, s)) in dl.iter().enumerate() {
                if !(3..6).contains(&s) {
                    continue;
                }
                for (b, &(n2, s2)) in dl.iter().enumerate() {
                    if n2 == n && (3..6).contains(&s2) {
                        let blk = rot[n].get_or_insert([[0.0; 3]; 3]);
                        blk[s - 3][s2 - 3] += f * me[a * nd + b];
                    }
                }
            }
        }
    }
    // point masses and rotary inertia from interactions (*MASS, *ROTARY INERTIA)
    for it in &p.interactions {
        for (na, da, nb, db, v) in it.mass() {
            let (sa, sb) = (dof::slot_of(da), dof::slot_of(db));
            if na != nb {
                return Err(FemError(format!("{}: coupling between nodes in the mass is not supported in explicit dynamics", it.name())));
            }
            if sa == sb && !(3..6).contains(&sa) {
                diag[na][sa] += v;
            } else if (3..6).contains(&sa) && (3..6).contains(&sb) {
                if dim == 3 {
                    rot[na].get_or_insert([[0.0; 3]; 3])[sa - 3][sb - 3] += v;
                } else if sa == sb {
                    diag[na][sa] += v;
                }
            }
        }
    }
    // nodes with blocks: put the diagonal part (shells) in the block
    let mut rot_inv = vec![None; nn];
    for n in 0..nn {
        if let Some(b) = rot[n].as_mut() {
            for i in 0..3 {
                b[i][i] += diag[n][3 + i];
                diag[n][3 + i] = b[i][i];
            }
            let off = b[0][1].abs() + b[0][2].abs() + b[1][2].abs() + b[1][0].abs() + b[2][0].abs() + b[2][1].abs();
            if off <= 1e-14 * (b[0][0] + b[1][1] + b[2][2]) {
                rot[n] = None;
                continue;
            }
            let (inv, det) = crate::elements::common::inv3(*b);
            if det.abs() <= 0.0 {
                return Err(FemError(format!("node {}: singular rotary inertia", m.node_name(n))));
            }
            rot_inv[n] = Some(inv);
        }
    }
    Ok(LumpedMass { diag, rot, rot_inv })
}
