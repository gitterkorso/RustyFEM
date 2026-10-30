//! Kärnan: modell, frihetsgrader, assemblering, tillstånd och utdata.
//!
//! Filerna här är delade av alla områden. Ändra dem bara när ett gränssnitt
//! verkligen behöver utökas (se ARCHITECTURE.md).

pub mod assembly;
pub mod dat;
pub mod dof;
pub mod model;
pub mod mpc;
pub mod output;
pub mod state;
pub mod system;
