//! Fler hybridelement (område hybrid2): C3D8RH, CPE4RH, CAX4RH, CPEG3H,
//! CPEG4H, CPEG4RH, CPEG6H och C3D10MH, samt geometrisk styvhet (*BUCKLE)
//! för hybridelementen i `hybrid.rs`.
//!
//! Tryckvariablerna ligger okondenserade på en intern nod (konstant tryck: en
//! variabel; C3D10MH: linjärt tryck med fyra variabler efter den modifierade
//! tetraederns interna förskjutningsnod). Kondenserar man bort dem blir
//! styvheten **exakt** densamma som för motsvarande element utan hybrid-
//! formulering (C3D8R, CPE4R, CAX4R med samma timglasstyvhet, CPEG3, CPEG4,
//! CPEG4R, CPEG6 och C3D10M; relativ skillnad ≤ 1e-11) – projektionen av
//! volymtöjningen på tryckbasen är identiteten när elementet har lika många
//! integrationspunkter som tryckvariabler (eller B-bar/likformig töjning
//! redan har konstant volymtöjning), och ingen β-term används i dessa
//! element. Även ν = 0,5 stämmer exakt med ν = 0,4999999
//! (`materials/incompressible_elastic.rs`).
//!
//! Elementen räknas därför med grundelementet (`Alias`): samma noder,
//! integrationspunkter, utdata, laster och massa. Hyperelastiska material
//! stöds inte: de kräver förstärkt timglasstyvhet (HOURGLASS=
//! ENHANCED) i de reducerat integrerade elementen och helt inkompressibla
//! material (D1 = 0) kräver tryckvariablerna.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::fem::dof::Dof;
use crate::fem::model::{DistLoad, SectionDef};
use crate::model::{fail, Result};

/// Hybridelement som räknas som sitt grundelement.
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
    /// Hyperelastiska material stöds inte (se modulens dokumentation).
    fn check(&self, ctx: &ElemCtx) -> Result<()> {
        if ctx.material.is_some_and(|m| m.hyperelastic().is_some()) {
            return fail(format!(
                "{}: hyperelastiska material stöds inte i {} (förstärkt timglasstyvhet och trycksvariabler saknas; använd C3D8H, CPE4H, CAX4H, ...)",
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
// Geometrisk styvhet för hybridelementen i hybrid.rs
// ---------------------------------------------------------------------------

/// Spänningsstyvheten för ett hybridelement ur spänningarna i tillståndet
/// (Cauchyspänningen med hybridtrycket per integrationspunkt), i elementets
/// geometriska frihetsgrader: samma uttryck som för C3D8/CPE4/CPS4
/// (`geometric::continuum`, med Jaumann-korrektionen).
/// Axisymmetriska element stöds inte (ringtermen är inte kontrollerad).
pub fn hybrid_geometric(h: &super::hybrid::Hybrid, ctx: &ElemCtx, mat: &dyn crate::materials::MaterialModel, state: &[f64]) -> Result<Vec<f64>> {
    let el = &h.base;
    if el.mode == crate::materials::StressMode::Axisymmetric {
        return not_implemented("geometrisk styvhet (*BUCKLE)", el.name);
    }
    super::geometric::continuum(el.mode, &el.ips(ctx)?, mat, state)
}

/// Spänningsstyvhet för de reducerat integrerade grundelementen (C3D8R,
/// CPE4R, CAX4R utom ringtermen): punkten med likformig töjning och samma
/// uttryck som `geometric::continuum`.
fn reduced_geometric(base: &super::solid_higher::Solid, ctx: &ElemCtx, state: &[f64]) -> Result<Vec<f64>> {
    let (ips, _) = super::solid_higher::tech::uniform_strain(base, ctx)?;
    let mat = ctx.material(base.name)?;
    super::geometric::continuum(base.mode, &ips, mat, state)
}
