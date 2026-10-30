//! Thermal-electrical-structural elements Q3D4, Q3D8 and Q3D20 (area
//! coupled2) for *COUPLED THERMAL-ELECTRICAL-STRUCTURAL.
//!
//! Each node has the displacements (1–3), the electric potential (9) and
//! the temperature (11). Like the coupled elements in `heat_coupled.rs`: the
//! mechanical part is C3D4, C3D8 and C3D20 respectively in the element registry
//! (vectors and matrices are mapped between the degree-of-freedom orderings, potential
//! and temperature get zeros), the thermal and electrical part a
//! heat conduction element of the same shape, which `steps/heat.rs` and
//! `steps/thermal_electric.rs` use (`heat::lookup`,
//! `thermal_electric::lookup`). The coupling is handled by the step type
//! (`steps/thermal_electric_structural.rs`).

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::heat::{HeatElem, Shape};
use crate::fem::dof::{Dof, EPOT, NT11, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct QElem {
    pub name: &'static str,
    /// The mechanical element (name in the registry).
    pub mech: &'static str,
    pub heat: HeatElem,
}

pub static Q3D4: QElem = QElem { name: "Q3D4", mech: "C3D4", heat: HeatElem { name: "Q3D4", shape: Shape::Tet4, axisym: false } };
pub static Q3D8: QElem = QElem { name: "Q3D8", mech: "C3D8", heat: HeatElem { name: "Q3D8", shape: Shape::Hex8, axisym: false } };
pub static Q3D20: QElem = QElem { name: "Q3D20", mech: "C3D20", heat: HeatElem { name: "Q3D20", shape: Shape::Hex20, axisym: false } };

pub static TYPES: &[&dyn ElementType] = &[&Q3D4, &Q3D8, &Q3D20];

static ALL: &[&QElem] = &[&Q3D4, &Q3D8, &Q3D20];

/// The element named `name`.
pub fn lookup(name: &str) -> Option<&'static QElem> {
    ALL.iter().copied().find(|e| e.name == name)
}

/// The heat and electricity part.
pub fn thermal(name: &str) -> Option<&'static HeatElem> {
    lookup(name).map(|e| &e.heat)
}

/// Name of the mechanical part (thermal strain).
pub fn mech_name(name: &str) -> Option<&'static str> {
    lookup(name).map(|e| e.mech)
}

/// Degrees of freedom per node and the number of mechanical ones among them.
const ND: usize = 5;
const NM: usize = 3;

impl QElem {
    fn mech_elem(&self) -> Result<&'static dyn ElementType> {
        match super::lookup(self.mech) {
            Some(m) => Ok(m),
            None => fail(format!("{}: the mechanical element {} is not in the registry", self.name, self.mech)),
        }
    }

    fn to_mech(&self, v: &[f64]) -> Vec<f64> {
        v.chunks(ND).flat_map(|c| c[..NM].to_vec()).collect()
    }

    fn from_mech(&self, v: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(v.len() / NM * ND);
        for c in v.chunks(NM) {
            out.extend_from_slice(c);
            out.extend_from_slice(&[0.0; ND - NM]);
        }
        out
    }

    fn from_mech_matrix(&self, k: &[f64]) -> Vec<f64> {
        let n = (k.len() as f64).sqrt().round() as usize;
        let nc = n / NM * ND;
        let idx = |i: usize| i / NM * ND + i % NM;
        let mut out = vec![0.0; nc * nc];
        for i in 0..n {
            for j in 0..n {
                out[idx(i) * nc + idx(j)] = k[i * n + j];
            }
        }
        out
    }
}

impl ElementType for QElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Continuum3D
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        self.heat.n_nodes()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, EPOT, NT11]
    }
    fn n_ips(&self) -> usize {
        self.mech_elem().map(|m| m.n_ips()).unwrap_or(0)
    }
    fn vtk_type(&self) -> u8 {
        self.mech_elem().map(|m| m.vtk_type()).unwrap_or(0)
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.heat.faces()
    }
    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.mech_elem()?.init_state(ctx)
    }
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(self.from_mech_matrix(&self.mech_elem()?.stiffness(ctx)?))
    }
    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], state_old: &[f64], state_new: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let r = self.mech_elem()?.internal_force(ctx, &self.to_mech(u), &self.to_mech(du), state_old, state_new, tangent)?;
        Ok(ElemResponse { fint: self.from_mech(&r.fint), k: r.k.map(|k| self.from_mech_matrix(&k)) })
    }
    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(self.from_mech_matrix(&self.mech_elem()?.mass(ctx, kind)?))
    }
    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        Ok(self.from_mech(&self.mech_elem()?.distributed_load(ctx, load, &self.to_mech(u))?))
    }
    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.mech_elem()?.volume_weights(ctx)
    }
    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        self.mech_elem()?.output(ctx, &self.to_mech(u), state)
    }
}
