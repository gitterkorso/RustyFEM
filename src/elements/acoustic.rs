//! Acoustic elements with one degree of freedom per node, acoustic pressure p
//! (degree of freedom 8, output POR): AC2D3, AC2D4, AC2D8, ACAX3, ACAX4, ACAX8,
//! AC3D4, AC3D6, AC3D8, AC3D10 and AC3D20.
//!
//! Formulation (Zienkiewicz–Newton): with the shape functions N
//!
//! - "stiffness" K_f = ∫ (1/ρ_f) ∇N·∇N dV,
//! - "mass"      M_f = ∫ (1/K_f) N N dV,
//! - volumetric drag in the time domain C_f = ∫ γ/(ρ_f K_f) N N dV,
//!
//! so that M_f p̈ + C_f ṗ + K_f p = Q, where Q = ∫ N n⁻·ü dS is the inward
//! volumetric acceleration (*CLOAD on degree of freedom 8). In the frequency domain
//! 1/ρ_f is replaced by 1/ρ̃ with the complex density ρ̃ = ρ_f + γ/(iΩ) (see
//! `steps/acoustic.rs`).
//!
//! The geometry (shape functions, Gauss rules, surfaces) is shared with
//! the heat conduction elements (`elements/heat.rs`). Plane elements use
//! the thickness from *SOLID SECTION; axisymmetric ones (x = r, y = z) integrate over
//! the full revolution (2π r).

use super::api::{ElemCtx, ElementType, Family, MassKind};
use super::common::Ip;
use super::heat::{HeatElem, Shape};
use crate::fem::dof::{Dof, POR};
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub struct AcousticElem {
    pub name: &'static str,
    /// The geometry (same as the corresponding heat conduction element).
    pub geom: HeatElem,
}

macro_rules! ac {
    ($id:ident, $name:literal, $shape:expr, $ax:expr) => {
        pub static $id: AcousticElem = AcousticElem { name: $name, geom: HeatElem { name: $name, shape: $shape, axisym: $ax } };
    };
}

ac!(AC2D3, "AC2D3", Shape::Tri3, false);
ac!(AC2D4, "AC2D4", Shape::Quad4, false);
ac!(AC2D8, "AC2D8", Shape::Quad8, false);
ac!(ACAX3, "ACAX3", Shape::Tri3, true);
ac!(ACAX4, "ACAX4", Shape::Quad4, true);
ac!(ACAX8, "ACAX8", Shape::Quad8, true);
ac!(AC3D4, "AC3D4", Shape::Tet4, false);
ac!(AC3D6, "AC3D6", Shape::Wedge6, false);
ac!(AC3D8, "AC3D8", Shape::Hex8, false);
ac!(AC3D10, "AC3D10", Shape::Tet10, false);
ac!(AC3D20, "AC3D20", Shape::Hex20, false);

pub static TYPES: &[&dyn ElementType] = &[&AC2D3, &AC2D4, &AC2D8, &ACAX3, &ACAX4, &ACAX8, &AC3D4, &AC3D6, &AC3D8, &AC3D10, &AC3D20];

static ALL: &[&AcousticElem] = &[&AC2D3, &AC2D4, &AC2D8, &ACAX3, &ACAX4, &ACAX8, &AC3D4, &AC3D6, &AC3D8, &AC3D10, &AC3D20];

/// The acoustic element named `name`.
pub fn lookup(name: &str) -> Option<&'static AcousticElem> {
    // area acoustics2: AC2D4R, AC3D8R, ACAX4R (acoustic_explicit.rs)
    ALL.iter().copied().find(|e| e.name == name).or_else(|| super::acoustic_explicit::lookup(name))
}

/// The element type is acoustic.
pub fn is_acoustic(et: &dyn ElementType) -> bool {
    lookup(et.name()).is_some()
}

/// Geometric element matrices (without material parameters).
pub struct Geometric {
    /// G = ∫ ∇N·∇N dV.
    pub g: Vec<f64>,
    /// H = ∫ N N dV (consistent).
    pub h: Vec<f64>,
}

impl AcousticElem {
    pub fn n_nodes(&self) -> usize {
        self.geom.n_nodes()
    }

    /// First-order element.
    pub fn first_order(&self) -> bool {
        self.geom.first_order()
    }

    /// Integration points. AC3D6 uses 3 × 2 points (triangle rule of
    /// degree 2 times two-point Gauss in ζ), the others
    /// the same rule as the heat conduction elements.
    pub fn ips(&self, x: &[[f64; 3]], thickness: f64) -> Result<Vec<Ip>> {
        if self.geom.shape == Shape::Wedge6 {
            let g = 0.577_350_269_189_625_8;
            let tri = [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]];
            let mut v = Vec::with_capacity(6);
            for z in [-g, g] {
                for t in tri {
                    v.push(super::heat::ip_at(Shape::Wedge6, x, [t[0], t[1], z], 1.0 / 6.0, thickness, false)?);
                }
            }
            return Ok(v);
        }
        self.geom.ips(x, thickness)
    }

    /// G and H for the nodal coordinates `x`.
    pub fn geometric(&self, x: &[[f64; 3]], thickness: f64) -> Result<Geometric> {
        let nn = self.n_nodes();
        let mut g = vec![0.0; nn * nn];
        let mut h = vec![0.0; nn * nn];
        for ip in self.ips(x, thickness)? {
            for i in 0..nn {
                let di = ip.dndx[i];
                for j in 0..nn {
                    let dj = ip.dndx[j];
                    g[i * nn + j] += ip.w * (di[0] * dj[0] + di[1] * dj[1] + di[2] * dj[2]);
                    h[i * nn + j] += ip.w * ip.n[i] * ip.n[j];
                }
            }
        }
        Ok(Geometric { g, h })
    }

    /// H as a diagonal (row-summed) matrix.
    pub fn lump(h: &[f64], nn: usize) -> Vec<f64> {
        let mut d = vec![0.0; nn * nn];
        for i in 0..nn {
            d[i * nn + i] = (0..nn).map(|j| h[i * nn + j]).sum();
        }
        d
    }

    /// The mass matrix M_f = H/K_f (or diagonal) for the bulk modulus `bulk`.
    pub fn fluid_mass(&self, x: &[[f64; 3]], thickness: f64, bulk: f64, kind: MassKind) -> Result<Vec<f64>> {
        let geo = self.geometric(x, thickness)?;
        let nn = self.n_nodes();
        let h = match kind {
            MassKind::Consistent => geo.h,
            MassKind::Lumped => Self::lump(&geo.h, nn),
        };
        Ok(h.iter().map(|v| v / bulk).collect())
    }
}

fn thickness(ctx: &ElemCtx) -> f64 {
    ctx.section.map_or(1.0, |s| s.solid_thickness())
}

/// (ρ_f, K_f) for the element.
fn props(ctx: &ElemCtx, elem: &str) -> Result<(f64, f64)> {
    let m = ctx.material(elem)?;
    let t = m.tangent_linear(StressMode::ThreeD)?;
    let bulk = t.first().copied().unwrap_or(0.0);
    if m.name() != "acoustic medium" || bulk <= 0.0 {
        return fail(format!("element {} ({elem}) requires an acoustic medium (*ACOUSTIC MEDIUM), not {}", ctx.label, m.name()));
    }
    // area gaps1: COMPLEX DENSITY without *DENSITY gives the reference density
    let density = if ctx.density > 0.0 { ctx.density } else { t.get(1).copied().unwrap_or(0.0) };
    if density <= 0.0 {
        return fail(format!("element {} ({elem}): an acoustic medium requires *DENSITY", ctx.label));
    }
    Ok((density, bulk))
}

impl ElementType for AcousticElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        self.geom.dim()
    }
    fn n_nodes(&self) -> usize {
        self.geom.n_nodes()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[POR]
    }
    fn n_ips(&self) -> usize {
        super::heat::gauss_rule(self.geom.shape).len()
    }
    fn vtk_type(&self) -> u8 {
        ElementType::vtk_type(&self.geom)
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.geom.faces()
    }

    /// K_f = ∫ (1/ρ_f) ∇N·∇N dV.
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (rho, _) = props(ctx, self.name)?;
        let geo = self.geometric(&ctx.x, thickness(ctx))?;
        Ok(geo.g.iter().map(|v| v / rho).collect())
    }

    /// M_f = ∫ (1/K_f) N N dV.
    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (_, bulk) = props(ctx, self.name)?;
        self.fluid_mass(&ctx.x, thickness(ctx), bulk, kind)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(super::common::volume_weights(&self.ips(&ctx.x, thickness(ctx))?))
    }
}
