//! Balkar, del 2 (område beams2): hybridbalkar, öppna tunnväggiga balkar
//! med välvning, tvärsnittsbiblioteket och snittkrafter för B23/B33.
//!
//! - Hybridbalkarna B21H, B22H, B31H, B32H (Timoshenko) har
//!   normalkraft och tvärkrafter som egna variabler. Kondenserad är
//!   elementmatrisen densamma som för B21 ... B32 utom att skjuvstyvheten
//!   är (1 + 10⁻⁶) gånger större, vilket
//!   `TimoBeam::shear_stiffness` gör för namn som slutar på H. B23H och B33H
//!   (Euler–Bernoulli med normalkraften som variabel) är B23/B33.
//! - Öppna balkar B31OS och B32OS (frihetsgrad 7 = välvningsamplitud) i
//!   `beams2_open.rs`.
//! - Tvärsnitten HEX, TRAPEZOID, THICK PIPE, ARBITRARY, J för RECT
//!   och välvningsstorheter i `beams2_sections.rs`.
//! - Snittkrafter SF/SM för B23/B33 i `beams2_eb_output.rs`.

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

/// Elementtyp med eget namn som i övrigt är en befintlig typ.
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
