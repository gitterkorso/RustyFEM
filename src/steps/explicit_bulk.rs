//! Bulk viscosity (*BULK VISCOSITY b₁, b₂; default 0.06 and 1.2) in explicit
//! dynamics (Belytschko, Liu & Moran ch. 6.2):
//!
//! ```text
//! p_bv = b₁ ρ c_d Lₑ ε̇_vol − ρ (b₂ Lₑ)² ε̇_vol² [ε̇_vol < 0]      (σ_bv = p_bv I)
//! ```
//!
//! where ε̇_vol = Σᵢ bᵢ·u̇ᵢ / V is computed with the half-step velocity u̇(i+½) and
//! bᵢ = ∫∇Nᵢ dV, Lₑ and c_d as in the stability limit (`explicit_dt`).
//! The nodal force is fᵢ = −p_bv t bᵢ (t = thickness for planar elements and shells).
//! The viscous stress is included in the reactions and the energy ALLVD but not in S.
//!
//! - Solids and planar elements: both terms; the quadratic one with the factor
//!   1/8 for fully integrated hexahedra (`quadratic_factor`). The damping ratio
//!   ξ in the stability limit keeps the factor 1 (with 1/8 the
//!   validated cases c3d8_explicit_volymviskositet and
//!   stang_smooth_step_plast deviated).
//! - Removed elements (*MODEL CHANGE, damage) have no bulk viscosity.
//! - Shells and membranes: the linear term with the in-plane membrane strain rate.
//! - Trusses: only the linear term (b₂ does not affect
//!   the result), σ_bv = b₁ ρ c Lₑ ε̇ with Lₑ = 0.99 L.
//! - Beams, springs etc.: no bulk viscosity (warning if b₁ ≠ 0).

use super::dt::{geometry, wave_speed};
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::model::Result;

/// An element with bulk viscosity.
pub struct BulkElem {
    pub e: usize,
    rho: f64,
    c: f64,
    /// Unscaled density and wave speed (mass scaling).
    rho0: f64,
    c0: f64,
    /// Modulus factor (c_d,3D/c)² for shells and membranes, otherwise 1.
    fmod: f64,
    /// Thickness (planar elements, shells), area (trusses) or 1.
    t: f64,
    family: Family,
    /// Volume (area) in the reference configuration (current density with NLGEOM).
    vol0: f64,
    /// Factor on the quadratic term (`quadratic_factor`).
    qfac: f64,
}

/// Factor on the quadratic term: 1/8 for fully integrated hexahedra
/// (C3D8, C3D8I, C3D8H), otherwise 1 (verified with
/// uniaxial SMOOTH STEP compression of 10 % with b₁ = 0, independent of b₂,
/// element shape and mesh size).
pub fn quadratic_factor(family: Family, n_ips: usize, n_nodes: usize) -> f64 {
    if family == Family::Continuum3D && n_nodes == 8 && n_ips == 8 {
        0.125
    } else {
        1.0
    }
}

pub struct BulkViscosity {
    pub b1: f64,
    pub b2: f64,
    pub elems: Vec<BulkElem>,
    /// Elements without bulk viscosity in RustyFEM (beams etc.).
    pub unsupported: Vec<usize>,
    /// The ratio c_d,eff/c_d per element (index = element) from the material's
    /// effective moduli (`explicit_moduli`); 1 for linear materials.
    pub cratio: Vec<f64>,
    /// Current density ρ₀V₀/V and current geometry (NLGEOM).
    pub nlgeom: bool,
    /// Only the linear term (start of the step, t = 0: the quadratic
    /// term is not included there).
    pub linear_only: bool,
    /// Area acoustics2: the bulk viscosity power per unit mass σ_bv ε̇_vol/ρ
    /// per element at the latest `forces` (for the EOS energy, `materials/eos.rs`).
    pub work: std::sync::Mutex<Vec<(usize, f64)>>,
    /// Factor s on Lₑ per element (index = element; empty = 1). Area coupled2:
    /// when the thermal limit governs the element's time increment,
    /// c_d·Δtₑ = s·Lₑ is used in the linear term and s²·Lₑ in the
    /// quadratic one (`explicit_coupled_full.rs`).
    pub le_scale: Vec<f64>,
}

impl BulkViscosity {
    pub fn new(p: &Prepared, b1: f64, b2: f64) -> Result<Self> {
        let m = p.model;
        let mut elems = Vec::new();
        let mut unsupported = Vec::new();
        let cratio = vec![1.0; m.elements.len()];
        if b1 == 0.0 && b2 == 0.0 {
            return Ok(BulkViscosity { b1, b2, elems, unsupported, cratio, nlgeom: false, linear_only: false, work: Default::default(), le_scale: Vec::new() });
        }
        for (e, el) in m.elements.iter().enumerate() {
            let ctx = p.ctx(e, false, 0.0, 0.0);
            if ctx.density <= 0.0 {
                continue;
            }
            let fam = el.etype.family();
            let name = el.etype.name();
            let c = wave_speed(&ctx, fam, name)?;
            match (c, geometry(fam, name, &ctx.x)) {
                (Some(c), Some(g0)) => {
                    let t = match fam {
                        Family::Continuum2D | Family::Shell | Family::Membrane | Family::Truss => ctx.section.map_or(1.0, |s| s.solid_thickness()),
                        _ => 1.0,
                    };
                    // shells and membranes: the bulk viscosity uses the modulus λ + 2μ
                    // (3D) instead of E/(1 − ν²)
                    let fmod = match (fam, ctx.material.and_then(|mm| mm.elastic_moduli())) {
                        (Family::Shell | Family::Membrane, Some((_, nu))) => (1.0 - nu) * (1.0 - nu * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu)),
                        _ => 1.0,
                    };
                    let qfac = quadratic_factor(fam, el.etype.n_ips(), el.etype.n_nodes());
                    elems.push(BulkElem { e, rho: ctx.density, c, rho0: ctx.density, c0: c, fmod, t, family: fam, vol0: g0.vol, qfac });
                }
                _ => {
                    if !matches!(fam, Family::Connector | Family::Other) {
                        unsupported.push(e);
                    }
                }
            }
        }
        Ok(BulkViscosity { b1, b2, elems, unsupported, cratio, nlgeom: false, linear_only: false, work: Default::default(), le_scale: Vec::new() })
    }

    /// Mass scaling: the density ρ₀fₑ and the wave speed c₀/√fₑ.
    pub fn scale_density(&mut self, f: &[f64]) {
        for be in &mut self.elems {
            let s = f.get(be.e).copied().unwrap_or(1.0);
            be.rho = be.rho0 * s;
            be.c = be.c0 / s.max(1e-300).sqrt();
        }
    }

    /// Adds the bulk viscosity nodal forces to `fv` for the configuration `x(e)`
    /// and the velocity `v`. Returns ε̇_vol per element (same order as `elems`).
    pub fn forces(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, v: &[NodeVec], fv: &mut [NodeVec]) -> Vec<f64> {
        let m = p.model;
        let dim = m.dim();
        let mut rates = Vec::with_capacity(self.elems.len());
        let mut work = Vec::new();
        for be in &self.elems {
            let el = &m.elements[be.e];
            // removed elements (*MODEL CHANGE, damage) have no bulk viscosity
            if p.activity.is_removed(be.e) {
                rates.push(0.0);
                continue;
            }
            let xe = x(be.e);
            let Some(g) = geometry(be.family, el.etype.name(), &xe) else {
                rates.push(0.0);
                continue;
            };
            if be.family == Family::Truss {
                let d = crate::elements::common::sub(xe[1], xe[0]);
                let l = crate::elements::common::norm(d);
                let (a, b) = (el.nodes[0], el.nodes[1]);
                let rate: f64 = (0..dim).map(|c| (v[b][c] - v[a][c]) * d[c] / l).sum::<f64>() / l;
                let sig = self.b1 * be.rho * be.c * g.lchar * rate;
                for c in 0..dim {
                    let f = sig * be.t * d[c] / l;
                    fv[a][c] -= f;
                    fv[b][c] += f;
                }
                rates.push(rate);
                continue;
            }
            let rate: f64 = el.nodes.iter().zip(&g.b).map(|(&n, b)| (0..3).map(|c| b[c] * v[n][c]).sum::<f64>()).sum::<f64>() / g.vol;
            let continuum = matches!(be.family, Family::Continuum2D | Family::Continuum3D);
            // current density and the material's effective
            // dilatational wave speed (plasticity lowers the shear modulus)
            let rho = if self.nlgeom && continuum && g.vol > 0.0 { be.rho * be.vol0 / g.vol } else { be.rho };
            let mut c = be.c * self.cratio.get(be.e).copied().unwrap_or(1.0);
            if rho != be.rho {
                c *= (be.rho / rho).sqrt();
            }
            let lchar = g.lchar * self.le_scale.get(be.e).copied().unwrap_or(1.0);
            let mut pbv = self.b1 * rho * c * lchar * rate * be.fmod;
            if rate < 0.0 && continuum && !self.linear_only {
                // thermally governed (s < 1): the quadratic term is almost gone;
                // the length s²·Lₑ
                let sc = self.le_scale.get(be.e).copied().unwrap_or(1.0);
                let lq = g.lchar * sc * sc;
                pbv -= be.qfac * rho * (self.b2 * lq * rate).powi(2);
            }
            for (&n, b) in el.nodes.iter().zip(&g.b) {
                for c in 0..dim {
                    fv[n][c] += pbv * be.t * b[c];
                }
            }
            if continuum && rho > 0.0 {
                work.push((be.e, pbv * rate / rho));
            }
            rates.push(rate);
        }
        *self.work.lock().unwrap() = work;
        rates
    }

    /// The element's effective dilatational wave speed (element no. `i` in `elems`).
    pub fn wave_speed_of(&self, i: usize) -> f64 {
        let be = &self.elems[i];
        be.c * self.cratio.get(be.e).copied().unwrap_or(1.0)
    }

    /// Critical damping ratio in the highest mode for element no. `i` in `elems`:
    /// ξ = b₁ − b₂² Lₑ/c_d · min(0, ε̇_vol) (trusses: b₁).
    pub fn xi(&self, i: usize, lchar: f64, rate: f64) -> f64 {
        let be = &self.elems[i];
        if be.family == Family::Truss {
            return self.b1;
        }
        if !matches!(be.family, Family::Continuum2D | Family::Continuum3D) {
            return self.b1;
        }
        let c = be.c * self.cratio.get(be.e).copied().unwrap_or(1.0);
        self.b1 - self.b2 * self.b2 * lchar / c * rate.min(0.0)
    }
}
