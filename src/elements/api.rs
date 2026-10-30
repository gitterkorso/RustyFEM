//! The element interface: everything an element can contribute to an analysis.
//!
//! An element implements `ElementType` for one or more element names and
//! is registered in its family's `TYPES` list (see `elements/mod.rs`). All methods
//! except the basic ones have default implementations that return the error
//! "not implemented for <element>", so a new element type can start with just
//! linear stiffness and grow method by method.
//!
//! Conventions:
//! - The element's degree-of-freedom vector is node-wise: for node 0 all `node_dofs(0)`,
//!   then node 1 etc. Matrices are dense, row-wise n×n (`Vec<f64>`).
//! - Coordinates (`ElemCtx::x`) are the reference configuration. `u` is the total
//!   displacement from the reference position, `du` the change since the start of the increment.
//! - The element state (`state`) is a free-form `Vec<f64>` that the element itself
//!   partitions, typically per integration point: stress + material state.
//!   The core keeps one accepted (`state_old`) and one trial (`state_new`) copy.

use crate::fem::dof::Dof;
use crate::fem::model::{DistLoad, SectionDef};
use crate::materials::MaterialModel;
use crate::model::{fail, Result};

/// Element family (for output, surfaces and checks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Plane continuum elements (plane stress/strain).
    Continuum2D,
    /// Axisymmetric continuum elements.
    Axisymmetric,
    Continuum3D,
    Truss,
    Beam,
    Shell,
    Membrane,
    /// Springs, dashpots, masses, connector elements.
    Connector,
    /// Heat conduction (NT11 only).
    Heat,
    Other,
}

/// Mass matrix: consistent or diagonal (lumped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassKind {
    Consistent,
    Lumped,
}

/// Everything the element needs to know about its surroundings.
#[derive(Clone)]
pub struct ElemCtx<'a> {
    /// The element's index in the model and element number.
    pub index: usize,
    pub label: i64,
    /// Nodal coordinates in the reference position.
    pub x: Vec<[f64; 3]>,
    pub section: Option<&'a SectionDef>,
    pub material: Option<&'a dyn MaterialModel>,
    /// Density from *DENSITY (0 if missing).
    pub density: f64,
    /// Geometrically nonlinear analysis (NLGEOM=YES).
    pub nlgeom: bool,
    /// Total time at the end of the increment and the increment length.
    pub time: f64,
    pub dt: f64,
    /// Nodal temperatures (predefined temperature field, *TEMPERATURE and
    /// *INITIAL CONDITIONS, TYPE=TEMPERATURE) at the end and start of the increment,
    /// respectively, in the element's node order. Empty if the model has no temperature field.
    /// If the model has temperature gradients through the thickness (third field on
    /// *TEMPERATURE), shells and beams get 2·nn values: the nodal temperatures followed
    /// by the gradients (`fem::assembly::Prepared::ctx`).
    pub temp: Vec<f64>,
    pub temp_old: Vec<f64>,
    /// Local material orientation from the section's ORIENTATION= (*ORIENTATION).
    pub orientation: Option<&'a crate::materials::anisotropic::Orientation>,
}

impl<'a> ElemCtx<'a> {
    /// Temperature and temperature change at an integration point with
    /// the shape functions `n` ((0, 0) if the model has no temperature field).
    ///
    /// First-order elements use the mean of the element's
    /// nodal temperatures for thermal strain (tests/cases/materials: C3D8
    /// with a skewed mesh, CPS4 and T2D2).
    /// Second-order and incompatible-mode continuum elements instead use
    /// the element's temperature interpolation (`Solid::temperatures`).
    pub fn temperature_at(&self, n: &[f64]) -> (f64, f64) {
        if self.temp.len() != n.len() || self.temp_old.len() != n.len() {
            return (0.0, 0.0);
        }
        let m = n.len() as f64;
        let t: f64 = self.temp.iter().sum::<f64>() / m;
        let t0: f64 = self.temp_old.iter().sum::<f64>() / m;
        (t, t - t0)
    }

    /// Temperature and temperature change interpolated with the shape functions `n`.
    pub fn temperature_interpolated(&self, n: &[f64]) -> (f64, f64) {
        if self.temp.len() != n.len() || self.temp_old.len() != n.len() {
            return (0.0, 0.0);
        }
        let t: f64 = n.iter().zip(&self.temp).map(|(a, b)| a * b).sum();
        let t0: f64 = n.iter().zip(&self.temp_old).map(|(a, b)| a * b).sum();
        (t, t - t0)
    }

    pub fn section(&self, elem: &str) -> Result<&'a SectionDef> {
        match self.section {
            Some(s) => Ok(s),
            None => fail(format!("element {} ({elem}) has no section (*SOLID SECTION, *BEAM SECTION, ...)", self.label)),
        }
    }

    pub fn material(&self, elem: &str) -> Result<&'a dyn MaterialModel> {
        match self.material {
            Some(m) => Ok(m),
            None => fail(format!("element {} ({elem}) has no material", self.label)),
        }
    }
}

/// Internal force and (optionally) tangent stiffness.
#[derive(Debug, Clone, Default)]
pub struct ElemResponse {
    pub fint: Vec<f64>,
    pub k: Option<Vec<f64>>,
}

/// One output field per integration point, e.g. S with the components S11, S22, S33, S12.
#[derive(Debug, Clone, Default)]
pub struct IpField {
    /// Field name: "S", "E", "PEEQ", "SF", "SM", ...
    pub name: String,
    /// Component names (for scalars the same as the field name).
    pub labels: Vec<String>,
    /// (integration point 1.., section point 0 = none, values)
    pub points: Vec<(usize, usize, Vec<f64>)>,
}

impl IpField {
    pub fn new(name: &str, labels: &[&str]) -> Self {
        IpField { name: name.into(), labels: labels.iter().map(|s| s.to_string()).collect(), points: Vec::new() }
    }
}

/// Error for methods that an element type does not have yet.
pub fn not_implemented<T>(what: &str, elem: &str) -> Result<T> {
    fail(format!("{what} is not implemented for {elem}"))
}

/// An element type, named according to the .inp format.
pub trait ElementType: Send + Sync {
    /// The element name, e.g. "CPS4".
    fn name(&self) -> &'static str;

    fn family(&self) -> Family;

    /// Spatial dimension (2 for plane elements, 3 otherwise).
    fn dim(&self) -> usize;

    fn n_nodes(&self) -> usize;

    /// Number of nodes on the *ELEMENT line. Default: all nodes. Elements whose last
    /// nodes are given in the section (the reference node for generalized plane strain,
    /// *SOLID SECTION, REF NODE=) have fewer; they are added by `FeModel::finalize`
    /// from `section_nodes`.
    fn n_connectivity(&self) -> usize {
        self.n_nodes()
    }

    /// Nodes (node numbers in the element's instance) that are appended to the element's
    /// node list based on the section. Default: none.
    fn section_nodes(&self, _section: &SectionDef) -> Result<Vec<i64>> {
        Ok(Vec::new())
    }

    /// Number of internal nodes that `FeModel::finalize` creates for the element (at
    /// the element centroid, last in the node list),
    /// e.g. for the pressure variables of hybrid elements. Default: none. The element must then
    /// have `n_connectivity()` = the number of nodes on the *ELEMENT line.
    fn internal_nodes(&self) -> usize {
        0
    }

    /// Degrees of freedom for a node (degree-of-freedom numbers, see `fem::dof`). 1–7 and 11 are supported.
    fn node_dofs(&self, local_node: usize) -> &'static [Dof];

    /// Number of degrees of freedom in the element.
    fn n_dofs(&self) -> usize {
        (0..self.n_nodes()).map(|i| self.node_dofs(i).len()).sum()
    }

    /// Number of integration points (for output and state).
    fn n_ips(&self) -> usize {
        0
    }

    /// VTK cell type for ParaView (0 = none).
    fn vtk_type(&self) -> u8 {
        0
    }

    /// The element faces in the .inp format numbering (face 1 = index 0). Used for
    /// *DLOAD P1..P6, boundary surfaces and contact.
    fn faces(&self) -> &'static [&'static [usize]] {
        &[]
    }

    /// Initial value of the element state (e.g. zero stress and the material's
    /// initial state at each integration point).
    fn init_state(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(Vec::new())
    }

    /// Linear stiffness matrix in global coordinates.
    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        not_implemented("linear stiffness", self.name())
    }

    /// Internal force and tangent given the displacement and material state.
    ///
    /// Default: `K u` with the linear stiffness, but only if the material is
    /// linear and the analysis is geometrically linear. Elements to be used
    /// with nonlinear materials or NLGEOM must implement the method.
    fn internal_force(
        &self,
        ctx: &ElemCtx,
        u: &[f64],
        _du: &[f64],
        _state_old: &[f64],
        _state_new: &mut Vec<f64>,
        tangent: bool,
    ) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometric nonlinearity (NLGEOM)", self.name());
        }
        if let Some(m) = ctx.material {
            if !m.is_linear() {
                return not_implemented(&format!("nonlinear material ({})", m.name()), self.name());
            }
        }
        let k = self.stiffness(ctx)?;
        let n = u.len();
        let fint = (0..n).map(|i| (0..n).map(|j| k[i * n + j] * u[j]).sum()).collect();
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    /// Mass matrix.
    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        not_implemented("mass matrix", self.name())
    }

    /// Geometric stiffness (stress stiffness) for buckling, given the current state.
    fn geometric_stiffness(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        not_implemented("geometric stiffness", self.name())
    }

    /// Equivalent nodal forces for a distributed load (*DLOAD). `u` is the current
    /// displacement (for follower loads with NLGEOM).
    fn distributed_load(&self, _ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        not_implemented(&format!("load type {}", load.kind), self.name())
    }

    /// ∫Nᵢ dV per node (including thickness or cross-sectional area). Used for
    /// self-weight in .rusty models and diagonal mass.
    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        not_implemented("volume weights", self.name())
    }

    /// Output per integration point (S, E, PEEQ, SF, SM, ...).
    fn output(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        Ok(Vec::new())
    }
}
