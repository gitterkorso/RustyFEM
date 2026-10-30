//! Area explicit_adv: advanced methods for explicit analysis in the general
//! explicit solver (`steps/explicit.rs`).
//!
//! - SPH (`explicit_adv_sph.rs`): PC3D particles, `*SOLID SECTION` with the
//!   characteristic length, `*SECTION CONTROLS` (KERNEL=, SPH FORMULATION=,
//!   SPH BULK VISCOSITY=, smoothing length), also C3D8R converted to
//!   particles at the start of the analysis (`*SECTION CONTROLS, ELEMENT CONVERSION=YES`,
//!   the `inp` step `conversion::apply`).
//! - DEM (`explicit_adv_dem.rs`): PD3D particles, `*DISCRETE SECTION`,
//!   particle contact in general contact (LINEAR, HERTZ, friction, damping).
//! - Euler (`explicit_adv_euler.rs`): EC3D8R, `*EULERIAN SECTION`,
//!   volume fractions, Lagrangian step followed by advection (first or second
//!   order) back to the fixed mesh.
//! - ALE (`explicit_adv_ale.rs`): `*ADAPTIVE MESH` for Lagrangian elements:
//!   volume-weighted smoothing of the mesh and advection of the element state.
//!
//! The explicit solver calls the module in five places (marked
//! "area explicit_adv" in `explicit.rs`): when the step is built (`build`), when
//! the diagonal mass is built (`patch_mass`), in each force computation
//! (`forces`: forces, state and, for Eulerian elements, the mass), after
//! the force computation (`after_forces`: the Eulerian nodes' velocities after
//! the advection) and when a frame is written (`output`). The elements themselves
//! (`elements/explicit_adv.rs`) give no forces or masses.

#[path = "explicit_adv_ale.rs"]
pub mod ale;
#[path = "explicit_adv_cel.rs"]
pub mod cel;
#[path = "explicit_adv_conversion.rs"]
pub mod conversion;
#[path = "explicit_adv_dem.rs"]
pub mod dem;
#[path = "explicit_adv_euler.rs"]
pub mod euler;
#[path = "explicit_adv_sph.rs"]
pub mod sph;

use super::mass::LumpedMass;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::{RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::model::Result;

/// A force computation: time, increment, displacement, displacement increment,
/// velocity (half step) and the state at the start of the increment.
pub struct Eval<'a> {
    /// Total time at the end of the increment.
    pub time: f64,
    /// Length of the increment (0 at the start of the step).
    pub dt: f64,
    pub u: &'a [NodeVec],
    pub du: &'a [NodeVec],
    pub v: &'a [NodeVec],
    pub state_old: &'a [Vec<f64>],
    /// Locked DOFs per node (bitmask over slots).
    pub fixed: &'a [u16],
}

/// Output that the module contributes to the forces.
pub struct Sink<'a> {
    pub state_new: &'a mut [Vec<f64>],
    pub fint: &'a mut [NodeVec],
    pub fext: &'a mut [NodeVec],
    pub fvis: &'a mut [NodeVec],
    pub fcon: &'a mut [NodeVec],
    pub mass: &'a mut LumpedMass,
}

/// The area's parts for an explicit step.
pub struct Advanced {
    pub sph: Option<sph::Sph>,
    pub dem: Option<dem::Dem>,
    pub euler: Option<euler::Euler>,
    pub ale: Option<ale::Ale>,
    /// Coupled Eulerian–Lagrangian contact.
    pub cel: Option<cel::Cel>,
    /// Work of the particle contact and friction (ALLFD) in the latest increment.
    pub fric: f64,
}

/// Bulk viscosity parameters for the step (default 0.06 and 1.2).
pub fn bulk_parameters(step: &StepDef) -> Result<(f64, f64)> {
    let mut bulk = (0.06, 1.2);
    for b in step.extra.iter().filter(|k| k.name == "BULK VISCOSITY") {
        let v = b.first_floats()?;
        bulk = (v.first().copied().unwrap_or(0.06), v.get(1).copied().unwrap_or(1.2));
    }
    Ok(bulk)
}

/// Has an earlier step set NLGEOM=NO (which is then inherited)? Otherwise
/// explicit analysis defaults to NLGEOM=YES for steps without the parameter.
/// Models with *ANNEAL keep NLGEOM=NO (the annealing in `steps/anneal.rs`
/// relies on the state without NLGEOM).
pub fn nlgeom_off_inherited(m: &crate::fem::model::FeModel, index: usize) -> bool {
    m.steps.iter().any(|s| s.procedure_name() == "ANNEAL") || m.steps[..index].iter().rev().find_map(|s| s.nlgeom).is_some_and(|n| !n)
}

/// *SECTION CONTROLS, NAME= for a section (CONTROLS=).
pub fn section_controls<'a>(p: &'a Prepared, sec: &crate::fem::model::SectionDef) -> Result<Option<&'a RawKeyword>> {
    let Some(name) = sec.raw.param_upper("CONTROLS") else { return Ok(None) };
    match p.model.extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())) {
        Some(k) => Ok(Some(k)),
        None => sec.raw.err(format!("*SECTION CONTROLS, NAME={name} is missing")),
    }
}

impl Advanced {
    /// Builds the area's parts; `None` if the model has none of them.
    pub fn build(p: &Prepared, step: &StepDef, nlgeom: bool) -> Result<Option<Advanced>> {
        let (b1, b2) = bulk_parameters(step)?;
        let sph = sph::Sph::build(p, b1, b2)?;
        let dem = dem::Dem::build(p, step)?;
        let euler = euler::Euler::build(p, step, b1, b2)?;
        let ale = ale::Ale::build(p, step, nlgeom)?;
        if sph.is_none() && dem.is_none() && euler.is_none() && ale.is_none() {
            return Ok(None);
        }
        let cel = match &euler {
            Some(eu) => cel::Cel::build(p, eu)?,
            None => None,
        };
        Ok(Some(Advanced { sph, dem, euler, ale, cel, fric: 0.0 }))
    }

    /// Adds the mass of the particles and Eulerian nodes to the diagonal mass.
    pub fn patch_mass(&self, mass: &mut LumpedMass) {
        if let Some(s) = &self.sph {
            s.patch_mass(mass);
        }
        if let Some(d) = &self.dem {
            d.patch_mass(mass);
        }
        if let Some(eu) = &self.euler {
            eu.patch_mass(mass);
        }
        if let Some(a) = &self.ale {
            a.patch_mass(mass);
        }
    }

    /// Forces and state for an increment.
    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink) -> Result<()> {
        self.fric = 0.0;
        if let Some(s) = self.sph.as_mut() {
            s.forces(p, ev, out)?;
        }
        if let Some(d) = self.dem.as_mut() {
            self.fric += d.forces(p, ev, out)?;
        }
        if let Some(eu) = self.euler.as_mut() {
            eu.forces(p, ev, out)?;
        }
        if let (Some(c), Some(eu)) = (self.cel.as_mut(), self.euler.as_ref()) {
            c.forces(p, ev, out, eu)?;
        }
        if let Some(a) = self.ale.as_mut() {
            a.forces(p, ev, out)?;
        }
        Ok(())
    }

    /// After the force computation: the Eulerian nodes back to their positions and
    /// their velocities after the advection; the ALE nodes' new positions.
    pub fn after_forces(&mut self, u: &mut [NodeVec], vh: &mut [NodeVec], fixed: &[u16]) {
        if let Some(eu) = self.euler.as_mut() {
            eu.after_forces(u, vh, fixed);
        }
        if let Some(a) = self.ale.as_mut() {
            a.after_forces(u, vh, fixed);
        }
    }

    /// Stability limit (Δt, element) for the particles and Eulerian elements
    /// with the state from the latest force computation.
    pub fn stable(&self) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for x in [self.sph.as_ref().and_then(|s| s.dt_est), self.euler.as_ref().and_then(|e| e.dt_est)].into_iter().flatten() {
            if best.is_none_or(|b| x.0 < b.0) {
                best = Some(x);
            }
        }
        best
    }

    /// The element's displacements without the ALE mesh relocations (for the elements
    /// without NLGEOM: strain and hourglass stiffness), `None` outside the ALE domain.
    pub fn hourglass_u(&self, nodes: &[usize], ue: &[f64]) -> Option<Vec<f64>> {
        self.ale.as_ref().and_then(|a| a.hourglass_u(nodes, ue))
    }

    /// Initial energy (*EOS, specific energy) for ALLIE.
    pub fn initial_energy(&self) -> f64 {
        self.sph.as_ref().map_or(0.0, |s| s.initial_energy) + self.euler.as_ref().map_or(0.0, |e| e.initial_energy)
    }

    /// Output for the area's elements and nodes.
    pub fn output(&self, p: &Prepared, fr: &mut FrameData, u: &[NodeVec], state: &[Vec<f64>]) -> Result<()> {
        if let Some(s) = &self.sph {
            s.output(p, fr, u, state)?;
        }
        if let Some(d) = &self.dem {
            d.output(p, fr);
        }
        if let Some(eu) = &self.euler {
            eu.output(p, fr, state)?;
        }
        Ok(())
    }
}
