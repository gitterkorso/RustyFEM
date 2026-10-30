//! More hybrid elements (area hybrid2): C3D8RH, CPE4RH, CAX4RH, CPEG3H,
//! CPEG4H, CPEG4RH, CPEG6H and C3D10MH, plus geometric stiffness (*BUCKLE)
//! for the hybrid elements in `hybrid.rs`.
//!
//! The pressure variables are kept uncondensed on an internal node (constant pressure: one
//! variable; C3D10MH: linear pressure with four variables after the modified
//! tetrahedron's internal displacement node). If they are condensed out, the
//! stiffness is **exactly** the same as for the corresponding element without the hybrid
//! formulation (C3D8R, CPE4R, CAX4R with the same hourglass stiffness, CPEG3, CPEG4,
//! CPEG4R, CPEG6 and C3D10M; relative difference ≤ 1e-11) – the projection of
//! the volumetric strain onto the pressure basis is the identity when the element has as many
//! integration points as pressure variables (or B-bar/uniform strain
//! already has constant volumetric strain), and no β term is used in these
//! elements. Also ν = 0.5 agrees exactly with ν = 0.4999999
//! (`materials/incompressible_elastic.rs`).
//!
//! The elements are therefore computed with the base element (`Alias`): the same nodes,
//! integration points, output, loads and mass. Hyperelastic materials
//! are not supported: they require enhanced hourglass stiffness (HOURGLASS=
//! ENHANCED) in the reduced-integration elements, and fully incompressible
//! materials (D1 = 0) require the pressure variables.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::fem::dof::Dof;
use crate::fem::model::{DistLoad, SectionDef};
use crate::model::{fail, Result};

/// Hybrid element computed as its base element.
pub struct Alias {
    pub name: &'static str,
    pub base: &'static dyn ElementType,
}

pub static C3D8RH: Alias = Alias { name: "C3D8RH", base: &super::solid_higher::C3D8R };
pub static CPE4RH: Alias = Alias { name: "CPE4RH", base: &super::solid_higher::CPE4R };
pub static CAX4RH: Alias = Alias { name: "CAX4RH", base: &super::axisym::CAX4R };
pub static CPEG3H: Alias = Alias { name: "CPEG3H", base: &super::solid_higher_gps::CPEG3 };
pub static CPEG4H: Alias = Alias { name: "CPEG4H", base: &super::solid_higher_gps::CPEG4 };
pub static CPEG4RH: Alias = Alias { name: "CPEG4RH", base: &super::solid_higher_gps::CPEG4R };
pub static CPEG6H: Alias = Alias { name: "CPEG6H", base: &super::solid_higher_gps::CPEG6 };
pub static C3D10MH: Alias = Alias { name: "C3D10MH", base: &super::solid_modified::C3D10M };

pub static TYPES: &[&dyn ElementType] = &[&C3D8RH, &CPE4RH, &CAX4RH, &CPEG3H, &CPEG4H, &CPEG4RH, &CPEG6H, &C3D10MH];

impl Alias {
    /// Hyperelastic materials are not supported (see the module documentation).
    fn check(&self, ctx: &ElemCtx) -> Result<()> {
        if ctx.material.is_some_and(|m| m.hyperelastic().is_some()) {
            return fail(format!(
                "{}: hyperelastic materials are not supported in {} (enhanced hourglass stiffness and pressure variables are missing; use C3D8H, CPE4H, CAX4H, ...)",
                ctx.label, self.name
            ));
        }
        Ok(())
    }
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
    fn n_dofs(&self) -> usize {
        self.base.n_dofs()
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
        self.check(ctx)?;
        self.base.init_state(ctx)
    }
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.check(ctx)?;
        self.base.stiffness(ctx)
    }
    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        self.check(ctx)?;
        self.base.internal_force(ctx, u, du, so, sn, tangent)
    }
    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        self.base.mass(ctx, kind)
    }
    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        match self.name {
            "C3D8RH" => reduced_geometric(&super::solid_higher::C3D8R, ctx, state),
            "CPE4RH" => reduced_geometric(&super::solid_higher::CPE4R, ctx, state),
            _ => self.base.geometric_stiffness(ctx, u, state),
        }
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

// ---------------------------------------------------------------------------
// Geometric stiffness for the hybrid elements in hybrid.rs
// ---------------------------------------------------------------------------

/// The stress stiffness of a hybrid element from the stresses in the state
/// (the Cauchy stress with the hybrid pressure per integration point), in the element's
/// geometric degrees of freedom: the same expression as for C3D8/CPE4/CPS4
/// (`geometric::continuum`, with the Jaumann correction).
/// Axisymmetric elements are not supported (the hoop term is not verified).
pub fn hybrid_geometric(h: &super::hybrid::Hybrid, ctx: &ElemCtx, mat: &dyn crate::materials::MaterialModel, state: &[f64]) -> Result<Vec<f64>> {
    let el = &h.base;
    if el.mode == crate::materials::StressMode::Axisymmetric {
        return not_implemented("geometric stiffness (*BUCKLE)", el.name);
    }
    super::geometric::continuum(el.mode, &el.ips(ctx)?, mat, state)
}

/// Stress stiffness for the reduced-integration base elements (C3D8R,
/// CPE4R, CAX4R except the hoop term): the uniform-strain point and the same
/// expression as `geometric::continuum`.
fn reduced_geometric(base: &super::solid_higher::Solid, ctx: &ElemCtx, state: &[f64]) -> Result<Vec<f64>> {
    let (ips, _) = super::solid_higher::tech::uniform_strain(base, ctx)?;
    let mat = ctx.material(base.name)?;
    super::geometric::continuum(base.mode, &ips, mat, state)
}
