//! Område explicit_adv: avancerade metoder för explicit analys i den allmänna
//! explicita lösaren (`steps/explicit.rs`).
//!
//! - SPH (`explicit_adv_sph.rs`): PC3D-partiklar, `*SOLID SECTION` med den
//!   karakteristiska längden, `*SECTION CONTROLS` (KERNEL=, SPH FORMULATION=,
//!   SPH BULK VISCOSITY=, utjämningslängd), även C3D8R som omvandlas till
//!   partiklar i analysens början (`*SECTION CONTROLS, ELEMENT CONVERSION=YES`,
//!   `inp`-steget `conversion::apply`).
//! - DEM (`explicit_adv_dem.rs`): PD3D-partiklar, `*DISCRETE SECTION`,
//!   partikelkontakt i general contact (LINEAR, HERTZ, friktion, dämpning).
//! - Euler (`explicit_adv_euler.rs`): EC3D8R, `*EULERIAN SECTION`,
//!   volymandelar, Lagrangesteg följt av advektion (första eller andra
//!   ordningen) tillbaka till det fasta nätet.
//! - ALE (`explicit_adv_ale.rs`): `*ADAPTIVE MESH` för Lagrange-element:
//!   volymviktad utjämning av nätet och advektion av elementtillståndet.
//!
//! Den explicita lösaren anropar modulen på fem ställen (markerade
//! "område explicit_adv" i `explicit.rs`): när steget byggs (`build`), när
//! den diagonala massan byggs (`patch_mass`), i varje kraftberäkning
//! (`forces`: krafter, tillstånd och för Euler-element massan), efter
//! kraftberäkningen (`after_forces`: Euler-nodernas hastigheter efter
//! advektionen) och när en frame skrivs (`output`). Elementen själva
//! (`elements/explicit_adv.rs`) ger inga krafter eller massor.

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

/// En kraftberäkning: tid, inkrement, förskjutning, förskjutningsinkrement,
/// hastighet (halvsteg) och tillståndet vid inkrementets början.
pub struct Eval<'a> {
    /// Total tid vid inkrementets slut.
    pub time: f64,
    /// Inkrementets längd (0 i stegets början).
    pub dt: f64,
    pub u: &'a [NodeVec],
    pub du: &'a [NodeVec],
    pub v: &'a [NodeVec],
    pub state_old: &'a [Vec<f64>],
    /// Låsta frihetsgrader per nod (bitmask över slots).
    pub fixed: &'a [u16],
}

/// Utdata som modulen bidrar med till krafterna.
pub struct Sink<'a> {
    pub state_new: &'a mut [Vec<f64>],
    pub fint: &'a mut [NodeVec],
    pub fext: &'a mut [NodeVec],
    pub fvis: &'a mut [NodeVec],
    pub fcon: &'a mut [NodeVec],
    pub mass: &'a mut LumpedMass,
}

/// Områdets delar för ett explicit steg.
pub struct Advanced {
    pub sph: Option<sph::Sph>,
    pub dem: Option<dem::Dem>,
    pub euler: Option<euler::Euler>,
    pub ale: Option<ale::Ale>,
    /// Kopplad Euler–Lagrange-kontakt.
    pub cel: Option<cel::Cel>,
    /// Partikelkontaktens och friktionens arbete (ALLFD) i senaste inkrementet.
    pub fric: f64,
}

/// Volymviskositetens parametrar för steget (standard 0,06 och 1,2).
pub fn bulk_parameters(step: &StepDef) -> Result<(f64, f64)> {
    let mut bulk = (0.06, 1.2);
    for b in step.extra.iter().filter(|k| k.name == "BULK VISCOSITY") {
        let v = b.first_floats()?;
        bulk = (v.first().copied().unwrap_or(0.06), v.get(1).copied().unwrap_or(1.2));
    }
    Ok(bulk)
}

/// Har ett tidigare steg angett NLGEOM=NO (som då ärvs)? Annars gäller
/// explicit analys standard NLGEOM=YES för steg utan parametern.
/// Modeller med *ANNEAL behåller NLGEOM=NO (glödgningen i `steps/anneal.rs`
/// bygger på tillståndet utan NLGEOM).
pub fn nlgeom_off_inherited(m: &crate::fem::model::FeModel, index: usize) -> bool {
    m.steps.iter().any(|s| s.procedure_name() == "ANNEAL") || m.steps[..index].iter().rev().find_map(|s| s.nlgeom).is_some_and(|n| !n)
}

/// *SECTION CONTROLS, NAME= för ett tvärsnitt (CONTROLS=).
pub fn section_controls<'a>(p: &'a Prepared, sec: &crate::fem::model::SectionDef) -> Result<Option<&'a RawKeyword>> {
    let Some(name) = sec.raw.param_upper("CONTROLS") else { return Ok(None) };
    match p.model.extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())) {
        Some(k) => Ok(Some(k)),
        None => sec.raw.err(format!("*SECTION CONTROLS, NAME={name} saknas")),
    }
}

impl Advanced {
    /// Bygger områdets delar; `None` om modellen inte har något av dem.
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

    /// Lägger partiklarnas och Euler-nodernas massa i den diagonala massan.
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

    /// Krafter och tillstånd för ett inkrement.
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

    /// Efter kraftberäkningen: Euler-noderna tillbaka till sina lägen och
    /// deras hastigheter efter advektionen; ALE-nodernas nya lägen.
    pub fn after_forces(&mut self, u: &mut [NodeVec], vh: &mut [NodeVec], fixed: &[u16]) {
        if let Some(eu) = self.euler.as_mut() {
            eu.after_forces(u, vh, fixed);
        }
        if let Some(a) = self.ale.as_mut() {
            a.after_forces(u, vh, fixed);
        }
    }

    /// Stabilitetsgräns (Δt, element) för partiklarna och Euler-elementen
    /// med tillståndet från senaste kraftberäkningen.
    pub fn stable(&self) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for x in [self.sph.as_ref().and_then(|s| s.dt_est), self.euler.as_ref().and_then(|e| e.dt_est)].into_iter().flatten() {
            if best.is_none_or(|b| x.0 < b.0) {
                best = Some(x);
            }
        }
        best
    }

    /// Elementets förskjutningar utan ALE-nätets omflyttningar (för elementen
    /// utan NLGEOM: töjning och hourglass-styvhet), `None` utanför ALE-domänen.
    pub fn hourglass_u(&self, nodes: &[usize], ue: &[f64]) -> Option<Vec<f64>> {
        self.ale.as_ref().and_then(|a| a.hourglass_u(nodes, ue))
    }

    /// Begynnelseenergi (*EOS, specifik energi) för ALLIE.
    pub fn initial_energy(&self) -> f64 {
        self.sph.as_ref().map_or(0.0, |s| s.initial_energy) + self.euler.as_ref().map_or(0.0, |e| e.initial_energy)
    }

    /// Utdata för områdets element och noder.
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
