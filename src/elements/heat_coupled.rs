//! Coupled temperature–displacement elements (C3D8T, CPE4T, CPS4T, CAX4T, CPE3T, CPS3T, C3D4T) for
//! *COUPLED TEMPERATURE-DISPLACEMENT.
//!
//! Each node has the displacements (1–2 or 1–3) and the temperature (11).
//! The element is composed of two parts:
//! - the mechanical part is the corresponding mechanical element (C3D8, CPE4, CPS4,
//!   CAX4) in the element registry; vectors and matrices are mapped between
//!   the elements' degree-of-freedom orderings (the temperature slots get zeros);
//! - the thermal part is a heat conduction element of the same shape
//!   (`thermal`), which the step type in `steps/heat_coupled.rs` uses for
//!   conduction, capacity and surface fluxes just as for the DC elements.
//!
//! The coupling (thermal strain from the nodal temperatures) is handled by the step type.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::heat::{HeatElem, Shape};
use crate::fem::dof::{Dof, NT11, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

pub struct CoupledElem {
    pub name: &'static str,
    /// The mechanical element (name in the registry).
    pub mech: &'static str,
    pub thermal: &'static HeatElem,
    pub family: Family,
}

static TH_HEX8: HeatElem = HeatElem { name: "C3D8T", shape: Shape::Hex8, axisym: false };
static TH_CPE4: HeatElem = HeatElem { name: "CPE4T", shape: Shape::Quad4, axisym: false };
static TH_CPS4: HeatElem = HeatElem { name: "CPS4T", shape: Shape::Quad4, axisym: false };
static TH_CAX4: HeatElem = HeatElem { name: "CAX4T", shape: Shape::Quad4, axisym: true };
// triangles and tetrahedra (also for *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT, steps/explicit_coupled.rs)
static TH_CPE3: HeatElem = HeatElem { name: "CPE3T", shape: Shape::Tri3, axisym: false };
static TH_CPS3: HeatElem = HeatElem { name: "CPS3T", shape: Shape::Tri3, axisym: false };
static TH_TET4: HeatElem = HeatElem { name: "C3D4T", shape: Shape::Tet4, axisym: false };

pub static C3D8T: CoupledElem = CoupledElem { name: "C3D8T", mech: "C3D8", thermal: &TH_HEX8, family: Family::Continuum3D };
pub static CPE4T: CoupledElem = CoupledElem { name: "CPE4T", mech: "CPE4", thermal: &TH_CPE4, family: Family::Continuum2D };
pub static CPS4T: CoupledElem = CoupledElem { name: "CPS4T", mech: "CPS4", thermal: &TH_CPS4, family: Family::Continuum2D };
pub static CAX4T: CoupledElem = CoupledElem { name: "CAX4T", mech: "CAX4", thermal: &TH_CAX4, family: Family::Axisymmetric };
pub static CPE3T: CoupledElem = CoupledElem { name: "CPE3T", mech: "CPE3", thermal: &TH_CPE3, family: Family::Continuum2D };
pub static CPS3T: CoupledElem = CoupledElem { name: "CPS3T", mech: "CPS3", thermal: &TH_CPS3, family: Family::Continuum2D };
pub static C3D4T: CoupledElem = CoupledElem { name: "C3D4T", mech: "C3D4", thermal: &TH_TET4, family: Family::Continuum3D };

pub static TYPES: &[&dyn ElementType] = &[&C3D8T, &CPE4T, &CPS4T, &CAX4T, &CPE3T, &CPS3T, &C3D4T];

static ALL: &[&CoupledElem] = &[&C3D8T, &CPE4T, &CPS4T, &CAX4T, &CPE3T, &CPS3T, &C3D4T];

/// The coupled element named `name`.
pub fn lookup(name: &str) -> Option<&'static CoupledElem> {
    // area coupled2: CPE4RT, CPS4RT, CAX4RT, C3D8RT in heat_coupled_reduced.rs
    ALL.iter().copied().find(|e| e.name == name).or_else(|| super::heat_coupled_reduced::lookup(name))
}

/// The heat conduction part of a coupled element.
pub fn thermal(name: &str) -> Option<&'static HeatElem> {
    lookup(name).map(|e| e.thermal)
}

/// Name of the mechanical part of a coupled element (for thermal strain).
pub fn mech_name(name: &str) -> Option<&'static str> {
    lookup(name).map(|e| e.mech)
}

impl CoupledElem {
    fn mech_elem(&self) -> Result<&'static dyn ElementType> {
        match super::lookup(self.mech) {
            Some(m) => Ok(m),
            None => fail(format!("{}: the mechanical element {} is not in the registry", self.name, self.mech)),
        }
    }

    /// Number of mechanical degrees of freedom per node.
    fn nm(&self) -> usize {
        if self.family == Family::Continuum3D {
            3
        } else {
            2
        }
    }

    /// Extracts the mechanical degrees of freedom from a coupled vector.
    fn to_mech(&self, v: &[f64]) -> Vec<f64> {
        let nm = self.nm();
        v.chunks(nm + 1).flat_map(|c| c[..nm].to_vec()).collect()
    }

    /// Mechanical vector → coupled (zeros in the temperature slots).
    fn from_mech(&self, v: &[f64]) -> Vec<f64> {
        let nm = self.nm();
        let mut out = Vec::with_capacity(v.len() / nm * (nm + 1));
        for c in v.chunks(nm) {
            out.extend_from_slice(c);
            out.push(0.0);
        }
        out
    }

    /// Mechanical matrix (n×n) → coupled.
    fn from_mech_matrix(&self, k: &[f64]) -> Vec<f64> {
        let nm = self.nm();
        let n = (k.len() as f64).sqrt().round() as usize;
        let nc = n / nm * (nm + 1);
        let idx = |i: usize| i / nm * (nm + 1) + i % nm;
        let mut out = vec![0.0; nc * nc];
        for i in 0..n {
            for j in 0..n {
                out[idx(i) * nc + idx(j)] = k[i * n + j];
            }
        }
        out
    }
}

impl ElementType for CoupledElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        self.family
    }
    fn dim(&self) -> usize {
        self.thermal.dim()
    }
    fn n_nodes(&self) -> usize {
        self.thermal.n_nodes()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.family == Family::Continuum3D {
            &[U1, U2, U3, NT11]
        } else {
            &[U1, U2, NT11]
        }
    }
    fn n_ips(&self) -> usize {
        self.mech_elem().map(|m| m.n_ips()).unwrap_or(0)
    }
    fn vtk_type(&self) -> u8 {
        self.mech_elem().map(|m| m.vtk_type()).unwrap_or(0)
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.thermal.faces()
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
