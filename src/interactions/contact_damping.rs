//! Område acoustics2: *CONTACT DAMPING i implicit analys (inkluderas med
//! `#[path]` från `contact.rs`) och gemensam tolkning för explicit analys
//! (`steps/explicit_contact.rs`).
//!
//! Implicit analys:
//! `*CONTACT DAMPING, DEFINITION=DAMPING COEFFICIENT, TANGENT FRACTION=s`
//! med dataraden μ₀, c₀, η. Dämpningskoefficienten (tryck per hastighet)
//! beror på spelet c: μ = μ₀ för c ≤ η c₀ (och i kontakt), avtar sedan linjärt
//! till noll vid c = c₀ (η = 0: linjärt från c = 0; c₀ = 0: bara i kontakt).
//! Dämpningsspänningen motverkar den relativa hastigheten v = Δu_rel/Δt:
//! normalt σ = μ v_n, tangentiellt τ = s μ v_t (s = TANGENT FRACTION,
//! standard 0 i implicit analys). Krafterna är μ A v med nodarean A och ingår
//! i tangenten med μ A/Δt. Dämpningen verkar inte i perturbationssteg.
//! DEFINITION=CRITICAL DAMPING FRACTION finns bara i explicit analys.

use crate::fem::model::RawKeyword;
use crate::model::Result;

/// Kontaktdämpning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Damping {
    /// Kritisk dämpningskvot (bara Explicit) i stället för en koefficient.
    pub critical: bool,
    /// μ₀ (koefficient) eller kritisk dämpningskvot.
    pub mu0: f64,
    /// Spelet c₀ där dämpningen upphör (Standard).
    pub c0: f64,
    /// η (andel av c₀ med konstant μ₀, Standard).
    pub eta: f64,
    /// TANGENT FRACTION (`None` = procedurens standard: 0 i Standard, 1 i Explicit).
    pub tangent: Option<f64>,
}

/// Läser *CONTACT DAMPING. `explicit` anger proceduren (CRITICAL DAMPING
/// FRACTION är standard i Explicit och inte tillåtet i Standard).
pub fn parse(kw: &RawKeyword, explicit: bool) -> Result<Damping> {
    let def = kw.param_upper("DEFINITION").unwrap_or_else(|| if explicit { "CRITICAL DAMPING FRACTION".into() } else { "DAMPING COEFFICIENT".into() });
    let critical = match def.trim() {
        "CRITICAL DAMPING FRACTION" => {
            if !explicit {
                return kw.err("*CONTACT DAMPING, DEFINITION=CRITICAL DAMPING FRACTION finns bara i explicit analys");
            }
            true
        }
        "DAMPING COEFFICIENT" => false,
        other => return kw.err(format!("DEFINITION={other} stöds inte (DAMPING COEFFICIENT, CRITICAL DAMPING FRACTION)")),
    };
    let v = kw.first_floats()?;
    let mu0 = v.first().copied().unwrap_or(0.0);
    if mu0 < 0.0 {
        return kw.err("*CONTACT DAMPING: dämpningen ska vara ≥ 0");
    }
    let c0 = v.get(1).copied().unwrap_or(0.0).max(0.0);
    let eta = v.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let tangent = match kw.param("TANGENT FRACTION") {
        Some(s) => match crate::fem::model::parse_f64(s.trim()) {
            Some(t) if t >= 0.0 => Some(t),
            _ => return kw.err("TANGENT FRACTION= ska vara ett tal ≥ 0"),
        },
        None => None,
    };
    Ok(Damping { critical, mu0, c0, eta, tangent })
}

impl Damping {
    /// Dämpningskoefficienten vid spelet `c` (c ≤ 0 = kontakt).
    pub fn coefficient(&self, c: f64) -> f64 {
        if c <= 0.0 {
            return self.mu0;
        }
        if self.c0 <= 0.0 || c >= self.c0 {
            return 0.0;
        }
        let ce = self.eta * self.c0;
        if c <= ce {
            self.mu0
        } else {
            self.mu0 * (self.c0 - c) / (self.c0 - ce)
        }
    }
}
