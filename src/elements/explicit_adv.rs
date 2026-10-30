//! Område explicit_adv: element för avancerade metoder i explicit dynamik.
//!
//! - PC3D: SPH-partikel (en nod, U1–U3). Massan ρ (2L)³ med L från
//!   `*SOLID SECTION`.
//! - PD3D: diskret partikel (DEM, en nod, U1–U3 och UR1–UR3), stel sfär med
//!   radien och densiteten från `*DISCRETE SECTION`.
//! - EC3D8R: Euler-element (åtta noder, U1–U3) med `*EULERIAN SECTION`.
//!
//! Elementen har familjen `Other`: den allmänna explicita lösaren
//! (`steps/explicit.rs`) räknar inte deras krafter, massor, volymviskositet
//! eller stabilitetsgräns. Allt detta görs av `steps/explicit_adv.rs`
//! (SPH, DEM, Euler), som också skriver deras utdata. I implicita steg ger
//! elementen ett fel.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::model::{fail, Result};

/// Ett element vars mekanik sköts av `steps/explicit_adv.rs`.
pub struct AdvElem {
    pub name: &'static str,
    pub nodes: usize,
    pub rot: bool,
}

pub static PC3D: AdvElem = AdvElem { name: "PC3D", nodes: 1, rot: false };
pub static PD3D: AdvElem = AdvElem { name: "PD3D", nodes: 1, rot: true };
pub static EC3D8R: AdvElem = AdvElem { name: "EC3D8R", nodes: 8, rot: false };

pub static TYPES: &[&dyn ElementType] = &[&PC3D, &PD3D, &EC3D8R];

/// Är elementtypen en partikel (PC3D, PD3D)?
pub fn is_particle(name: &str) -> bool {
    matches!(name, "PC3D" | "PD3D")
}

/// Är elementtypen en av områdets (PC3D, PD3D, EC3D8R)?
pub fn is_adv(name: &str) -> bool {
    matches!(name, "PC3D" | "PD3D" | "EC3D8R")
}

impl ElementType for AdvElem {
    fn name(&self) -> &'static str {
        self.name
    }

    fn family(&self) -> Family {
        Family::Other
    }

    fn dim(&self) -> usize {
        3
    }

    fn n_nodes(&self) -> usize {
        self.nodes
    }

    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.rot {
            &[U1, U2, U3, UR1, UR2, UR3]
        } else {
            &[U1, U2, U3]
        }
    }

    fn n_ips(&self) -> usize {
        1
    }

    fn vtk_type(&self) -> u8 {
        if self.nodes == 8 {
            12
        } else {
            1
        }
    }

    fn faces(&self) -> &'static [&'static [usize]] {
        if self.nodes == 8 {
            super::continuum3d::HEX8_FACES
        } else {
            &[]
        }
    }

    fn init_state(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        // tillståndet skapas av steps/explicit_adv.rs i första explicita steget
        Ok(Vec::new())
    }

    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        fail(format!("{} kan bara användas i *DYNAMIC, EXPLICIT", self.name))
    }

    fn internal_force(&self, _ctx: &ElemCtx, u: &[f64], _du: &[f64], state_old: &[f64], state_new: &mut Vec<f64>, _tangent: bool) -> Result<ElemResponse> {
        // explicit: krafterna och tillståndet sköts av steps/explicit_adv.rs
        *state_new = state_old.to_vec();
        Ok(ElemResponse { fint: vec![0.0; u.len()], k: None })
    }

    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        // massan läggs till av steps/explicit_adv.rs
        Ok(Vec::new())
    }

    /// Tyngdkraft (GRAV, BX, BY, BZ) på partiklar: m g med partikelns massa
    /// (PC3D: ρ (2L)³, PD3D: ρ 4πr³/3). Euler-elementens massa ändras med
    /// materialet; deras volymkrafter läggs till av steps/explicit_adv_euler.rs.
    fn distributed_load(&self, ctx: &ElemCtx, load: &crate::fem::model::DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let n = self.n_dofs();
        if self.nodes == 8 {
            if matches!(load.kind.as_str(), "GRAV" | "BX" | "BY" | "BZ") {
                return Ok(vec![0.0; n]);
            }
            return super::api::not_implemented(&format!("lasttypen {}", load.kind), self.name);
        }
        let sec = ctx.section(self.name)?;
        let v = sec.raw.first_floats()?;
        let size = v.first().copied().unwrap_or(0.0);
        let (rho, vol) = if self.rot {
            let rho = sec.raw.param("DENSITY").and_then(crate::fem::model::parse_f64).unwrap_or(0.0);
            (rho, 4.0 / 3.0 * std::f64::consts::PI * size.powi(3))
        } else {
            (ctx.density, (2.0 * size).powi(3))
        };
        let Some(b) = super::common::body_load_vector(&load.kind, load.magnitude, &load.extra, rho, self.name)? else {
            return super::api::not_implemented(&format!("lasttypen {}", load.kind), self.name);
        };
        // BX/BY/BZ är kraft per volym, GRAV per volym genom densiteten
        let mut f = vec![0.0; n];
        for c in 0..3 {
            f[c] = b[c] * vol;
        }
        Ok(f)
    }

    fn volume_weights(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(vec![0.0; self.nodes])
    }

    fn output(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        Ok(Vec::new())
    }
}
