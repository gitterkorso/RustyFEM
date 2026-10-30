//! Frihetsgrader med .inp-formatets numrering.
//!
//! Internt har varje nod åtta platser ("slots") så att alla elementfamiljer
//! kan dela samma nodvektorer:
//!
//! | slot | nummer | storhet |
//! |---|---|---|
//! | 0–2 | 1–3 | förskjutning U1–U3 |
//! | 3–5 | 4–6 | rotation UR1–UR3 |
//! | 6 | 7 | välvning (warping) i öppna balktvärsnitt |
//! | 7 | 11 | temperatur NT11 |
//! | 7 | 8 | porvattentryck POR (*SOILS; delar slot med NT11) |
//! | 6 | 9 | elektrisk potential EPOT (termisk-elektriska och piezoelektriska element; delar slot med WARP) |
//!
//! Ett element anger sina frihetsgrader per nod med frihetsgradsnummer (`Dof`),
//! t.ex. `&[U1, U2]` för CPS4 och `&[U1, U2, U3, UR1, UR2, UR3]` för B33.

/// Nummer för en frihetsgrad (1–7 eller 11).
pub type Dof = u8;

pub const U1: Dof = 1;
pub const U2: Dof = 2;
pub const U3: Dof = 3;
pub const UR1: Dof = 4;
pub const UR2: Dof = 5;
pub const UR3: Dof = 6;
pub const WARP: Dof = 7;
pub const NT11: Dof = 11;
/// Porvattentryck (frihetsgrad 8), lagras i samma slot som NT11.
pub const POR: Dof = 8;
/// Elektrisk potential (frihetsgrad 9), lagras i samma slot som välvningen
/// (element med potential har aldrig välvning).
pub const EPOT: Dof = 9;

/// Antal platser per nod i nodvektorer.
pub const SLOTS: usize = 8;

/// Frihetsgradsnummer för varje slot.
pub const SLOT_DOF: [Dof; SLOTS] = [U1, U2, U3, UR1, UR2, UR3, WARP, NT11];

/// En nodvektor: ett värde per slot.
pub type NodeVec = [f64; SLOTS];

/// Slot för en frihetsgrad, eller `None` om den inte stöds.
pub fn slot(d: Dof) -> Option<usize> {
    match d {
        1..=7 => Some(d as usize - 1),
        11 => Some(7),
        8 => Some(7),
        9 => Some(6),
        _ => None,
    }
}

/// Slot för en frihetsgrad som redan är kontrollerad (panik annars).
pub fn slot_of(d: Dof) -> usize {
    slot(d).unwrap_or_else(|| panic!("intern: frihetsgrad {d} saknar slot"))
}

/// Namn för utskrift och felmeddelanden.
pub fn name(d: Dof) -> &'static str {
    match d {
        1 => "U1",
        2 => "U2",
        3 => "U3",
        4 => "UR1",
        5 => "UR2",
        6 => "UR3",
        7 => "WARP",
        11 => "NT11",
        8 => "POR",
        9 => "EPOT",
        _ => "?",
    }
}

/// Konvergensfält för Newton-kontrollen (varje fält kontrolleras för sig).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    /// Förskjutning / kraft.
    Displacement,
    /// Rotation / moment.
    Rotation,
    /// Välvning / bimoment.
    Warping,
    /// Temperatur / värmeflöde.
    Temperature,
}

pub fn field(d: Dof) -> Field {
    match d {
        1..=3 => Field::Displacement,
        4..=6 => Field::Rotation,
        7 => Field::Warping,
        9 => Field::Warping,
        _ => Field::Temperature,
    }
}

/// Bitmask över slots (bit i = slot i).
pub fn mask(dofs: &[Dof]) -> u16 {
    dofs.iter().fold(0u16, |m, &d| m | (1 << slot_of(d)))
}
