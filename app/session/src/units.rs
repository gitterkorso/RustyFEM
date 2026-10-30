//! Enhetssystem för visning. Motorn är enhetslös (konsekventa
//! enheter); valet här ändrar bara etiketterna i UI:t och räknar aldrig om
//! några värden.

use rustyfem::fem::model::FeModel;
use serde::{Deserialize, Serialize};
use specta::Type;

/// Etiketten som projektfiler före enhetsvalet alltid sparade.
pub const LEGACY_LABEL: &str = "SI (m, N, Pa)";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Type)]
#[serde(rename_all = "snake_case")]
pub enum UnitSystem {
    /// Inget angivet: värdena visas utan enheter.
    None,
    /// m, N, kg, s, Pa.
    Si,
    /// mm, N, t, s, MPa.
    SiMm,
    /// in, lbf, lbf·s²/in, s, psi.
    Us,
}

/// Enhetssymboler för de storheter som UI:t visar. Tom sträng = ingen enhet.
#[derive(Debug, Clone, Serialize, Type)]
pub struct UnitSymbols {
    pub length: String,
    pub area: String,
    pub force: String,
    pub moment: String,
    pub stress: String,
    pub density: String,
    pub mass: String,
    pub time: String,
    pub acceleration: String,
    pub angle: String,
}

impl UnitSystem {
    pub const ALL: [UnitSystem; 4] = [UnitSystem::None, UnitSystem::Si, UnitSystem::SiMm, UnitSystem::Us];

    /// Nyckel i projektfilen.
    pub fn key(self) -> &'static str {
        match self {
            UnitSystem::None => "none",
            UnitSystem::Si => "si",
            UnitSystem::SiMm => "si_mm",
            UnitSystem::Us => "us",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            UnitSystem::None => "Ej angivet",
            UnitSystem::Si => "SI (m, N, kg, s, Pa)",
            UnitSystem::SiMm => "SI-mm (mm, N, t, s, MPa)",
            UnitSystem::Us => "US (in, lbf, s, psi)",
        }
    }

    /// Läser nyckeln i projektfilen (okänd nyckel ger `None`).
    pub fn parse(s: &str) -> UnitSystem {
        UnitSystem::ALL.into_iter().find(|u| u.key() == s.trim()).unwrap_or(UnitSystem::None)
    }

    pub fn symbols(self) -> UnitSymbols {
        let [length, area, force, moment, stress, density, mass, time, acceleration] = match self {
            UnitSystem::None => [""; 9],
            UnitSystem::Si => ["m", "m²", "N", "N·m", "Pa", "kg/m³", "kg", "s", "m/s²"],
            UnitSystem::SiMm => ["mm", "mm²", "N", "N·mm", "MPa", "t/mm³", "t", "s", "mm/s²"],
            UnitSystem::Us => ["in", "in²", "lbf", "lbf·in", "psi", "lbf·s²/in⁴", "lbf·s²/in", "s", "in/s²"],
        };
        let angle = if self == UnitSystem::None { "" } else { "rad" };
        UnitSymbols {
            length: length.into(),
            area: area.into(),
            force: force.into(),
            moment: moment.into(),
            stress: stress.into(),
            density: density.into(),
            mass: mass.into(),
            time: time.into(),
            acceleration: acceleration.into(),
            angle: angle.into(),
        }
    }

    /// Enheten för ett resultatfält (komponenten behövs inte: alla komponenter
    /// i ett fält har samma storhet). Okända och dimensionslösa fält ger "".
    pub fn field_unit(self, field: &str) -> String {
        let s = self.symbols();
        match field {
            "U" => s.length,
            "UR" => s.angle,
            "RF" | "CF" => s.force,
            "RM" | "CM" => s.moment,
            "S" | "MISES" => s.stress,
            _ => String::new(),
        }
    }

    /// Gissar enhetssystemet för en `.inp` ur materialens elasticitetsmoduler
    /// (stål: 2,1e11 Pa, 2,1e5 MPa, 3e7 psi). Inget material ger `None`.
    pub fn guess(m: &FeModel) -> UnitSystem {
        let e_max = m
            .materials
            .iter()
            .flat_map(|mt| mt.behaviors.iter())
            .filter(|k| k.name == "ELASTIC")
            .filter_map(|k| k.data.first().and_then(|l| l.fields.first()).and_then(|f| f.trim().replace(['d', 'D'], "e").parse::<f64>().ok()))
            .fold(0.0f64, f64::max);
        match e_max {
            e if e >= 1e9 => UnitSystem::Si,
            e if e >= 1e7 => UnitSystem::Us,
            e if e >= 1e2 => UnitSystem::SiMm,
            _ => UnitSystem::None,
        }
    }
}
