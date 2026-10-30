//! Area acoustics2: acoustic elements for explicit dynamics with reduced
//! integration, AC2D4R, AC3D8R and ACAX4R (degree of freedom 8, POR).
//!
//! The elements have the same geometry as AC2D4, AC3D8 and ACAX4
//! (`elements/acoustic.rs`) and are used by the explicit acoustic
//! step type (`steps/acoustic_explicit.rs`). The stiffness K_f = ∫ ∇N·∇N/ρ dV
//! is fully integrated (2 × 2 and 2 × 2 × 2 points, respectively); for the linear
//! acoustic problem this corresponds to one point with full
//! hourglass stiffness for rectangular plane elements and hexahedra
//! (tests/cases/acoustics2/). For ACAX4R the hourglass stiffness under
//! axisymmetry is not verified.
//! The mass is diagonal (row-summed).

use super::acoustic::AcousticElem;
use super::api::ElementType;
use super::heat::{HeatElem, Shape};

pub static AC2D4R: AcousticElem = AcousticElem { name: "AC2D4R", geom: HeatElem { name: "AC2D4R", shape: Shape::Quad4, axisym: false } };
pub static AC3D8R: AcousticElem = AcousticElem { name: "AC3D8R", geom: HeatElem { name: "AC3D8R", shape: Shape::Hex8, axisym: false } };
pub static ACAX4R: AcousticElem = AcousticElem { name: "ACAX4R", geom: HeatElem { name: "ACAX4R", shape: Shape::Quad4, axisym: true } };

pub static TYPES: &[&dyn ElementType] = &[&AC2D4R, &AC3D8R, &ACAX4R];

static ALL: &[&AcousticElem] = &[&AC2D4R, &AC3D8R, &ACAX4R];

/// The element named `name` (called from `acoustic::lookup`).
pub fn lookup(name: &str) -> Option<&'static AcousticElem> {
    ALL.iter().copied().find(|e| e.name == name)
}
