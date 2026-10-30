//! Degrees of freedom with the .inp-format numbering.
//!
//! Internally each node has eight slots so that all element families
//! can share the same nodal vectors:
//!
//! | slot | number | quantity |
//! |---|---|---|
//! | 0–2 | 1–3 | displacement U1–U3 |
//! | 3–5 | 4–6 | rotation UR1–UR3 |
//! | 6 | 7 | warping in open beam cross-sections |
//! | 7 | 11 | temperature NT11 |
//! | 7 | 8 | pore water pressure POR (*SOILS; shares slot with NT11) |
//! | 6 | 9 | electric potential EPOT (thermal-electric and piezoelectric elements; shares slot with WARP) |
//!
//! An element declares its degrees of freedom per node with DOF numbers (`Dof`),
//! e.g. `&[U1, U2]` for CPS4 and `&[U1, U2, U3, UR1, UR2, UR3]` for B33.

/// Number of a degree of freedom (1–7 or 11).
pub type Dof = u8;

pub const U1: Dof = 1;
pub const U2: Dof = 2;
pub const U3: Dof = 3;
pub const UR1: Dof = 4;
pub const UR2: Dof = 5;
pub const UR3: Dof = 6;
pub const WARP: Dof = 7;
pub const NT11: Dof = 11;
/// Pore water pressure (DOF 8), stored in the same slot as NT11.
pub const POR: Dof = 8;
/// Electric potential (DOF 9), stored in the same slot as warping
/// (elements with potential never have warping).
pub const EPOT: Dof = 9;

/// Number of slots per node in nodal vectors.
pub const SLOTS: usize = 8;

/// DOF number for each slot.
pub const SLOT_DOF: [Dof; SLOTS] = [U1, U2, U3, UR1, UR2, UR3, WARP, NT11];

/// A nodal vector: one value per slot.
pub type NodeVec = [f64; SLOTS];

/// Slot for a degree of freedom, or `None` if it is not supported.
pub fn slot(d: Dof) -> Option<usize> {
    match d {
        1..=7 => Some(d as usize - 1),
        11 => Some(7),
        8 => Some(7),
        9 => Some(6),
        _ => None,
    }
}

/// Slot for a degree of freedom that has already been checked (panics otherwise).
pub fn slot_of(d: Dof) -> usize {
    slot(d).unwrap_or_else(|| panic!("internal: degree of freedom {d} has no slot"))
}

/// Name for printing and error messages.
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

/// Convergence fields for the Newton check (each field is checked separately).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    /// Displacement / force.
    Displacement,
    /// Rotation / moment.
    Rotation,
    /// Warping / bimoment.
    Warping,
    /// Temperature / heat flux.
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

/// Bit mask over slots (bit i = slot i).
pub fn mask(dofs: &[Dof]) -> u16 {
    dofs.iter().fold(0u16, |m, &d| m | (1 << slot_of(d)))
}
