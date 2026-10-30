//! Stela element: R3D4, R3D3 (3D), R2D2 (plant) och RAX2 (axisymmetriskt).
//!
//! Stela element har ingen styvhet, massa eller spänning. De ger bara noder
//! (med translationerna U1–U3 respektive U1–U2) och ytor som
//! kan användas som stela masterytor i kontakt (*SURFACE med SPOS/SNEG, se
//! `interactions/geom.rs`). Elementen måste ingå i en *RIGID BODY
//! (`interactions/rigid_body.rs`) som knyter deras noder till referensnoden.
//!
//! Ytornas positiva normal (SPOS) följer nodordningen:
//! högerhandsregeln för R3D4/R3D3 och riktningen från nod 1 till nod 2
//! vriden 90° moturs för R2D2/RAX2.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, MassKind};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::model::Result;

pub struct Rigid {
    name: &'static str,
    dim: usize,
    nn: usize,
    faces: &'static [&'static [usize]],
    vtk: u8,
}

pub static R3D4: Rigid = Rigid { name: "R3D4", dim: 3, nn: 4, faces: &[&[0, 1, 2, 3]], vtk: 9 };
pub static R3D3: Rigid = Rigid { name: "R3D3", dim: 3, nn: 3, faces: &[&[0, 1, 2]], vtk: 5 };
pub static R2D2: Rigid = Rigid { name: "R2D2", dim: 2, nn: 2, faces: &[&[0, 1]], vtk: 3 };
pub static RAX2: Rigid = Rigid { name: "RAX2", dim: 2, nn: 2, faces: &[&[0, 1]], vtk: 3 };

pub static TYPES: &[&dyn ElementType] = &[&R2D2, &R3D3, &R3D4, &RAX2];

/// Är elementtypen ett stelt element (R3D4, R3D3, R2D2, RAX2)?
pub fn is_rigid(t: &dyn ElementType) -> bool {
    TYPES.iter().any(|r| r.name() == t.name())
}

impl ElementType for Rigid {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.dim == 2 { &[U1, U2] } else { &[U1, U2, U3] }
    }
    fn vtk_type(&self) -> u8 {
        self.vtk
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.faces
    }

    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        Ok(vec![0.0; n * n])
    }

    /// Ingen inre kraft: rörelsen bestäms helt av den stela kroppen (även vid NLGEOM).
    fn internal_force(&self, _ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let n = u.len();
        sn.clear();
        Ok(ElemResponse { fint: vec![0.0; n], k: tangent.then(|| vec![0.0; n * n]) })
    }

    /// Stela element har ingen egen massa (massa och tröghet anges på
    /// referensnoden med *MASS/*ROTARY INERTIA).
    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        Ok(vec![0.0; n * n])
    }

    fn geometric_stiffness(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        Ok(vec![0.0; n * n])
    }

    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(vec![0.0; self.nn])
    }

    fn distributed_load(&self, _ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        super::api::not_implemented(&format!("lasttypen {} (laster på stela kroppar anges på referensnoden)", load.kind), self.name)
    }
}
