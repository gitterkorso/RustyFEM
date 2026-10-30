//! Coupled thermal-electrical elements (DC2D3E, DC2D4E, DC2D8E, DCAX4E,
//! DC3D4E, DC3D8E, DC3D20E) for *COUPLED THERMAL-ELECTRICAL.
//!
//! Each node has the electric potential (degree of freedom 9, EPOT) and
//! the temperature (11). Geometry, integration rules and surfaces are the same as for
//! the heat conduction elements (`elements/heat.rs`); the step type
//! (`steps/thermal_electric.rs`) uses the `HeatElem` part both for
//! heat conduction and for electrical conduction.

use super::api::{ElemCtx, ElementType, Family};
use super::heat::{gauss_rule, shape_dim, shape_nodes, HeatElem, Shape};
use crate::fem::dof::{Dof, EPOT, NT11};
use crate::model::Result;

pub struct TeElem {
    pub heat: HeatElem,
}

macro_rules! te {
    ($id:ident, $name:expr, $shape:expr, $ax:expr) => {
        pub static $id: TeElem = TeElem { heat: HeatElem { name: $name, shape: $shape, axisym: $ax } };
    };
}

te!(DC2D3E, "DC2D3E", Shape::Tri3, false);
te!(DC2D4E, "DC2D4E", Shape::Quad4, false);
te!(DC2D8E, "DC2D8E", Shape::Quad8, false);
te!(DCAX4E, "DCAX4E", Shape::Quad4, true);
te!(DC3D4E, "DC3D4E", Shape::Tet4, false);
te!(DC3D8E, "DC3D8E", Shape::Hex8, false);
te!(DC3D20E, "DC3D20E", Shape::Hex20, false);

pub static TYPES: &[&dyn ElementType] = &[&DC2D3E, &DC2D4E, &DC2D8E, &DCAX4E, &DC3D4E, &DC3D8E, &DC3D20E];

static ALL: &[&TeElem] = &[&DC2D3E, &DC2D4E, &DC2D8E, &DCAX4E, &DC3D4E, &DC3D8E, &DC3D20E];

/// The geometry part of the thermal-electrical element `name`.
pub fn lookup(name: &str) -> Option<&'static HeatElem> {
    ALL.iter().find(|e| e.heat.name == name).map(|e| &e.heat)
        // area coupled2: Q3D4, Q3D8, Q3D20
        .or_else(|| super::thermal_electric_structural::thermal(name))
}

impl ElementType for TeElem {
    fn name(&self) -> &'static str {
        self.heat.name
    }
    fn family(&self) -> Family {
        Family::Heat
    }
    fn dim(&self) -> usize {
        shape_dim(self.heat.shape)
    }
    fn n_nodes(&self) -> usize {
        shape_nodes(self.heat.shape)
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[EPOT, NT11]
    }
    fn n_ips(&self) -> usize {
        gauss_rule(self.heat.shape).len()
    }
    fn vtk_type(&self) -> u8 {
        ElementType::vtk_type(&self.heat)
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.heat.faces()
    }
    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        ElementType::volume_weights(&self.heat, ctx)
    }
}
