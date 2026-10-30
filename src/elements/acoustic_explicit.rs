//! Område acoustics2: akustiska element för explicit dynamik med reducerad
//! integration, AC2D4R, AC3D8R och ACAX4R (frihetsgrad 8, POR).
//!
//! Elementen har samma geometri som AC2D4, AC3D8 och ACAX4
//! (`elements/acoustic.rs`) och används av den explicita akustiska
//! stegtypen (`steps/acoustic_explicit.rs`). Styvheten K_f = ∫ ∇N·∇N/ρ dV
//! integreras fullt (2 × 2 respektive 2 × 2 × 2 punkter); för det linjära
//! akustiska problemet motsvarar det en punkt med fullständig
//! hourglass-styvhet för rätvinkliga plana element och hexaedrar
//! (tests/cases/acoustics2/). För ACAX4R är hourglass-styvheten vid
//! axisymmetri inte verifierad.
//! Massan är diagonal (radsummerad).

use super::acoustic::AcousticElem;
use super::api::ElementType;
use super::heat::{HeatElem, Shape};

pub static AC2D4R: AcousticElem = AcousticElem { name: "AC2D4R", geom: HeatElem { name: "AC2D4R", shape: Shape::Quad4, axisym: false } };
pub static AC3D8R: AcousticElem = AcousticElem { name: "AC3D8R", geom: HeatElem { name: "AC3D8R", shape: Shape::Hex8, axisym: false } };
pub static ACAX4R: AcousticElem = AcousticElem { name: "ACAX4R", geom: HeatElem { name: "ACAX4R", shape: Shape::Quad4, axisym: true } };

pub static TYPES: &[&dyn ElementType] = &[&AC2D4R, &AC3D8R, &ACAX4R];

static ALL: &[&AcousticElem] = &[&AC2D4R, &AC3D8R, &ACAX4R];

/// Elementet med namnet `name` (anropas från `acoustic::lookup`).
pub fn lookup(name: &str) -> Option<&'static AcousticElem> {
    ALL.iter().copied().find(|e| e.name == name)
}
