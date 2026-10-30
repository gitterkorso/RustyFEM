//! Volymviskositet (*BULK VISCOSITY b₁, b₂; standard 0,06 och 1,2) i explicit
//! dynamik (Belytschko, Liu & Moran kap. 6.2):
//!
//! ```text
//! p_bv = b₁ ρ c_d Lₑ ε̇_vol − ρ (b₂ Lₑ)² ε̇_vol² [ε̇_vol < 0]      (σ_bv = p_bv I)
//! ```
//!
//! där ε̇_vol = Σᵢ bᵢ·u̇ᵢ / V beräknas med halvstegshastigheten u̇(i+½) och
//! bᵢ = ∫∇Nᵢ dV, Lₑ och c_d som i stabilitetsgränsen (`explicit_dt`).
//! Nodkraften är fᵢ = −p_bv t bᵢ (t = tjocklek för plana element och skal).
//! Den viskösa spänningen ingår i reaktionerna och energin ALLVD men inte i S.
//!
//! - Solider och plana element: båda termerna; den kvadratiska med faktorn
//!   1/8 för fullt integrerade hexaedrar (`quadratic_factor`). Dämpningskvoten
//!   ξ i stabilitetsgränsen behåller faktorn 1 (med 1/8 där avvek de
//!   validerade fallen c3d8_explicit_volymviskositet och
//!   stang_smooth_step_plast).
//! - Borttagna element (*MODEL CHANGE, skada) har ingen volymviskositet.
//! - Skal och membran: den linjära termen med membrantöjningshastigheten i planet.
//! - Stänger: bara den linjära termen (b₂ påverkar inte
//!   resultatet), σ_bv = b₁ ρ c Lₑ ε̇ med Lₑ = 0,99 L.
//! - Balkar, fjädrar m.m.: ingen volymviskositet (varning om b₁ ≠ 0).

use super::dt::{geometry, wave_speed};
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::model::Result;

/// Ett element med volymviskositet.
pub struct BulkElem {
    pub e: usize,
    rho: f64,
    c: f64,
    /// Oskalad densitet och våghastighet (massskalning).
    rho0: f64,
    c0: f64,
    /// Modulfaktor (c_d,3D/c)² för skal och membran, annars 1.
    fmod: f64,
    /// Tjocklek (plana element, skal), area (stänger) eller 1.
    t: f64,
    family: Family,
    /// Volym (area) i referenskonfigurationen (aktuell densitet vid NLGEOM).
    vol0: f64,
    /// Faktor på den kvadratiska termen (`quadratic_factor`).
    qfac: f64,
}

/// Faktor på den kvadratiska termen: 1/8 för fullt integrerade hexaedrar
/// (C3D8, C3D8I, C3D8H), annars 1 (kontrollerat med
/// enaxlig SMOOTH STEP-hoptryckning 10 % med b₁ = 0, oberoende av b₂,
/// elementform och nätstorlek).
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
    /// Element som saknar volymviskositet i RustyFEM (balkar m.m.).
    pub unsupported: Vec<usize>,
    /// Kvoten c_d,eff/c_d per element (index = element) från materialets
    /// effektiva moduler (`explicit_moduli`); 1 för linjära material.
    pub cratio: Vec<f64>,
    /// Aktuell densitet ρ₀V₀/V och aktuell geometri (NLGEOM).
    pub nlgeom: bool,
    /// Bara den linjära termen (stegets början, t = 0: den kvadratiska
    /// termen ingår inte där).
    pub linear_only: bool,
    /// Område acoustics2: volymviskositetens effekt per massenhet σ_bv ε̇_vol/ρ
    /// per element vid senaste `forces` (för EOS-energin, `materials/eos.rs`).
    pub work: std::sync::Mutex<Vec<(usize, f64)>>,
    /// Faktor s på Lₑ per element (index = element; tom = 1). Område coupled2:
    /// när den termiska gränsen styr elementets tidsinkrement används
    /// c_d·Δtₑ = s·Lₑ i den linjära termen och s²·Lₑ i den
    /// kvadratiska (`explicit_coupled_full.rs`).
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
                    // skal och membran: volymviskositeten använder modulen λ + 2μ
                    // (3D) i stället för E/(1 − ν²)
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

    /// Massskalning: densiteten ρ₀fₑ och våghastigheten c₀/√fₑ.
    pub fn scale_density(&mut self, f: &[f64]) {
        for be in &mut self.elems {
            let s = f.get(be.e).copied().unwrap_or(1.0);
            be.rho = be.rho0 * s;
            be.c = be.c0 / s.max(1e-300).sqrt();
        }
    }

    /// Lägger volymviskositetens nodkrafter i `fv` för konfigurationen `x(e)`
    /// och hastigheten `v`. Returnerar ε̇_vol per element (samma ordning som `elems`).
    pub fn forces(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, v: &[NodeVec], fv: &mut [NodeVec]) -> Vec<f64> {
        let m = p.model;
        let dim = m.dim();
        let mut rates = Vec::with_capacity(self.elems.len());
        let mut work = Vec::new();
        for be in &self.elems {
            let el = &m.elements[be.e];
            // borttagna element (*MODEL CHANGE, skada) har ingen volymviskositet
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
            // aktuell densitet och materialets effektiva
            // dilatationsvåghastighet (plasticitet sänker skjuvmodulen)
            let rho = if self.nlgeom && continuum && g.vol > 0.0 { be.rho * be.vol0 / g.vol } else { be.rho };
            let mut c = be.c * self.cratio.get(be.e).copied().unwrap_or(1.0);
            if rho != be.rho {
                c *= (be.rho / rho).sqrt();
            }
            let lchar = g.lchar * self.le_scale.get(be.e).copied().unwrap_or(1.0);
            let mut pbv = self.b1 * rho * c * lchar * rate * be.fmod;
            if rate < 0.0 && continuum && !self.linear_only {
                // termiskt styrt (s < 1): den kvadratiska termen är nästan borta;
                // längden s²·Lₑ
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

    /// Elementets effektiva dilatationsvåghastighet (element nr `i` i `elems`).
    pub fn wave_speed_of(&self, i: usize) -> f64 {
        let be = &self.elems[i];
        be.c * self.cratio.get(be.e).copied().unwrap_or(1.0)
    }

    /// Kritisk dämpningskvot i högsta moden för element nr `i` i `elems`:
    /// ξ = b₁ − b₂² Lₑ/c_d · min(0, ε̇_vol) (stänger: b₁).
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
