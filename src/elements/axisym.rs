//! Axisymmetriska kontinuumelement: CAX3, CAX4, CAX4R, CAX4I, CAX6, CAX8, CAX8R.
//!
//! Koordinat 1 är radien r och koordinat 2 axeln z. Töjningarnas
//! ordning är [εrr, εzz, εθθ, γrz] med εθθ = u_r / r. Integrationen tas över
//! hela varvet (vikten innehåller 2πr), så krafter och reaktioner är totala
//! värden runt axeln.
//!
//! Elementen använder motorn i `solid_higher.rs`. CAX4 har selektivt
//! reducerad integrering (B-bar av εrr+εzz+εθθ) som CPE4/C3D8.

use super::solid_higher::shapes::Shape;
use super::solid_higher::{RuleKind, Solid, Tech};
use super::ElementType;
use crate::materials::StressMode;

const fn ax(name: &'static str, shape: Shape, rule: RuleKind, tech: Tech) -> Solid {
    Solid { name, shape, mode: StressMode::Axisymmetric, rule, tech }
}

pub static CAX3: Solid = ax("CAX3", Shape::Tri3, RuleKind::Tri(1), Tech::Full);
pub static CAX4: Solid = ax("CAX4", Shape::Quad4, RuleKind::Gauss(2), Tech::Bbar);
pub static CAX4R: Solid = ax("CAX4R", Shape::Quad4, RuleKind::Gauss(1), Tech::Hourglass);
pub static CAX4I: Solid = ax("CAX4I", Shape::Quad4, RuleKind::Gauss(2), Tech::Incompatible);
pub static CAX6: Solid = ax("CAX6", Shape::Tri6, RuleKind::Tri(3), Tech::Full);
pub static CAX8: Solid = ax("CAX8", Shape::Quad8, RuleKind::Gauss(3), Tech::Full);
pub static CAX8R: Solid = ax("CAX8R", Shape::Quad8, RuleKind::Gauss(2), Tech::Full);

pub static TYPES: &[&dyn ElementType] = &[&CAX3, &CAX4, &CAX4I, &CAX4R, &CAX6, &CAX8, &CAX8R];

/// B-matris (4 × 2n) för axisymmetri: rader [εrr, εzz, εθθ, γrz], kolumner (u_r, u_z) per nod.
pub fn bmat_axi(n: &[f64], dndx: &[[f64; 3]], r: f64) -> Vec<f64> {
    let nn = n.len();
    let nd = 2 * nn;
    let mut b = vec![0.0; 4 * nd];
    for i in 0..nn {
        let d = dndx[i];
        b[2 * i] = d[0];
        b[nd + 2 * i + 1] = d[1];
        b[2 * nd + 2 * i] = n[i] / r;
        b[3 * nd + 2 * i] = d[1];
        b[3 * nd + 2 * i + 1] = d[0];
    }
    b
}
