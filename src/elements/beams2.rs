//! Beams, part 2 (area beams2): hybrid beams, open thin-walled beams
//! with warping, the section library and section forces for B23/B33.
//!
//! - The hybrid beams B21H, B22H, B31H, B32H (Timoshenko) have
//!   axial force and transverse shear forces as separate variables. Condensed, the
//!   element matrix is the same as for B21 ... B32 except that the shear stiffness
//!   is (1 + 10⁻⁶) times larger, which
//!   `TimoBeam::shear_stiffness` does for names ending in H. B23H and B33H
//!   (Euler–Bernoulli with the axial force as a variable) are B23/B33.
//! - Open beams B31OS and B32OS (degree of freedom 7 = warping amplitude) in
//!   `beams2_open.rs`.
//! - The sections HEX, TRAPEZOID, THICK PIPE, ARBITRARY, J for RECT
//!   and warping quantities in `beams2_sections.rs`.
//! - Section forces SF/SM for B23/B33 in `beams2_eb_output.rs`.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::beam_timoshenko::TimoBeam;
use crate::fem::dof::Dof;
use crate::fem::model::{DistLoad, SectionDef};
use crate::model::Result;

#[path = "beams2_sections.rs"]
pub mod sections;
#[path = "beams2_open.rs"]
pub mod open;
#[path = "beams2_eb_output.rs"]
pub mod eb_output;

pub static B21H: TimoBeam = TimoBeam::named("B21H", 2, 2);
pub static B22H: TimoBeam = TimoBeam::named("B22H", 2, 3);
pub static B31H: TimoBeam = TimoBeam::named("B31H", 3, 2);
pub static B32H: TimoBeam = TimoBeam::named("B32H", 3, 3);
pub static B23H: Alias = Alias { name: "B23H", base: &super::beam_eb::B23 };
pub static B33H: Alias = Alias { name: "B33H", base: &super::beam_eb::B33 };

pub static TYPES: &[&dyn ElementType] = &[&B21H, &B22H, &B31H, &B32H, &B23H, &B33H, &open::B31OS, &open::B32OS];

/// Element type with its own name that is otherwise an existing type.
pub struct Alias {
    name: &'static str,
    base: &'static dyn ElementType,
}

impl ElementType for Alias {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        self.base.family()
    }
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn n_nodes(&self) -> usize {
        self.base.n_nodes()
    }
    fn n_connectivity(&self) -> usize {
        self.base.n_connectivity()
    }
    fn section_nodes(&self, section: &SectionDef) -> Result<Vec<i64>> {
        self.base.section_nodes(section)
    }
    fn internal_nodes(&self) -> usize {
        self.base.internal_nodes()
    }
    fn node_dofs(&self, i: usize) -> &'static [Dof] {
        self.base.node_dofs(i)
    }
    fn n_ips(&self) -> usize {
        self.base.n_ips()
    }
    fn vtk_type(&self) -> u8 {
        self.base.vtk_type()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.base.faces()
    }
    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.base.init_state(ctx)
    }
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.base.stiffness(ctx)
    }
    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        self.base.internal_force(ctx, u, du, so, sn, tangent)
    }
    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        self.base.mass(ctx, kind)
    }
    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        self.base.geometric_stiffness(ctx, u, state)
    }
    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        self.base.distributed_load(ctx, load, u)
    }
    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.base.volume_weights(ctx)
    }
    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        self.base.output(ctx, u, state)
    }
}
