//! Punkt- och kopplingselement: fjädrar (SPRING1, SPRING2, SPRINGA),
//! dämpare (DASHPOT1, DASHPOT2, DASHPOTA), punktmassor (MASS, ROTARYI) och
//! kopplingselement (CONN2D2, CONN3D2).
//!
//! Elementen bär bara noder och nummer. Deras frihetsgrader beror på
//! tvärsnittsdata (t.ex. vilken frihetsgrad en SPRING1 verkar i), så själva
//! bidragen (styvhet, massa, dämpning, villkor) byggs av interaktionerna i
//! `src/interactions/springs.rs` och `src/interactions/connector.rs` från
//! *SPRING, *DASHPOT, *MASS, *ROTARY INERTIA och *CONNECTOR SECTION.
//! Elementen själva aktiverar inga frihetsgrader.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, MassKind};
use crate::fem::dof::Dof;
use crate::model::Result;

pub struct Point {
    name: &'static str,
    nodes: usize,
}

pub static SPRING1: Point = Point { name: "SPRING1", nodes: 1 };
pub static SPRING2: Point = Point { name: "SPRING2", nodes: 2 };
pub static SPRINGA: Point = Point { name: "SPRINGA", nodes: 2 };
pub static DASHPOT1: Point = Point { name: "DASHPOT1", nodes: 1 };
pub static DASHPOT2: Point = Point { name: "DASHPOT2", nodes: 2 };
pub static DASHPOTA: Point = Point { name: "DASHPOTA", nodes: 2 };
pub static MASS: Point = Point { name: "MASS", nodes: 1 };
pub static ROTARYI: Point = Point { name: "ROTARYI", nodes: 1 };
pub static CONN2D2: Point = Point { name: "CONN2D2", nodes: 2 };
pub static CONN3D2: Point = Point { name: "CONN3D2", nodes: 2 };

pub static TYPES: &[&dyn ElementType] = &[&SPRING1, &SPRING2, &SPRINGA, &DASHPOT1, &DASHPOT2, &DASHPOTA, &MASS, &ROTARYI, &CONN2D2, &CONN3D2];

impl ElementType for Point {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Connector
    }
    /// 2 så att elementen inte gör en plan modell tredimensionell.
    fn dim(&self) -> usize {
        if self.name == "CONN3D2" { 3 } else { 2 }
    }
    fn n_nodes(&self) -> usize {
        self.nodes
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[]
    }
    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }
    fn internal_force(&self, _ctx: &ElemCtx, _u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        Ok(ElemResponse { fint: Vec::new(), k: tangent.then(Vec::new) })
    }
    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }
    fn geometric_stiffness(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }
    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }
}
