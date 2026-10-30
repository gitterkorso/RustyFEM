//! Reducerat integrerade kopplade temperatur–förskjutningselement CPE4RT,
//! CPS4RT, CAX4RT och C3D8RT (område coupled2).
//!
//! Byggda som de övriga kopplade elementen (`heat_coupled.rs`): den
//! mekaniska delen är CPE4R, CPS4R, CAX4R respektive C3D8R (enpunktsregel
//! med hourglass-kontroll; i *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT med
//! explicit hourglass-kontroll, `steps/explicit_coupled_full.rs`), den termiska
//! delen ett värmeledningselement med samma form. `heat_coupled::lookup`
//! hittar elementen här.

use super::api::{ElementType, Family};
use super::heat::{HeatElem, Shape};
use super::heat_coupled::CoupledElem;

static TH_CPE4R: HeatElem = HeatElem { name: "CPE4RT", shape: Shape::Quad4, axisym: false };
static TH_CPS4R: HeatElem = HeatElem { name: "CPS4RT", shape: Shape::Quad4, axisym: false };
static TH_CAX4R: HeatElem = HeatElem { name: "CAX4RT", shape: Shape::Quad4, axisym: true };
static TH_HEX8R: HeatElem = HeatElem { name: "C3D8RT", shape: Shape::Hex8, axisym: false };

pub static CPE4RT: CoupledElem = CoupledElem { name: "CPE4RT", mech: "CPE4R", thermal: &TH_CPE4R, family: Family::Continuum2D };
pub static CPS4RT: CoupledElem = CoupledElem { name: "CPS4RT", mech: "CPS4R", thermal: &TH_CPS4R, family: Family::Continuum2D };
pub static CAX4RT: CoupledElem = CoupledElem { name: "CAX4RT", mech: "CAX4R", thermal: &TH_CAX4R, family: Family::Axisymmetric };
pub static C3D8RT: CoupledElem = CoupledElem { name: "C3D8RT", mech: "C3D8R", thermal: &TH_HEX8R, family: Family::Continuum3D };

pub static TYPES: &[&dyn ElementType] = &[&CPE4RT, &CPS4RT, &CAX4RT, &C3D8RT];

static ALL: &[&CoupledElem] = &[&CPE4RT, &CPS4RT, &CAX4RT, &C3D8RT];

/// Det reducerade kopplade elementet med namnet `name`.
pub fn lookup(name: &str) -> Option<&'static CoupledElem> {
    ALL.iter().copied().find(|e| e.name == name)
}
