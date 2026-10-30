//! Area acoustics2: *CONTACT DAMPING in implicit analysis (included with
//! `#[path]` from `contact.rs`) and shared parsing for explicit analysis
//! (`steps/explicit_contact.rs`).
//!
//! Implicit analysis:
//! `*CONTACT DAMPING, DEFINITION=DAMPING COEFFICIENT, TANGENT FRACTION=s`
//! with the data line μ₀, c₀, η. The damping coefficient (pressure per velocity)
//! depends on the clearance c: μ = μ₀ for c ≤ η c₀ (and in contact), then decreases linearly
//! to zero at c = c₀ (η = 0: linearly from c = 0; c₀ = 0: only in contact).
//! The damping stress opposes the relative velocity v = Δu_rel/Δt:
//! normal σ = μ v_n, tangential τ = s μ v_t (s = TANGENT FRACTION,
//! default 0 in implicit analysis). The forces are μ A v with the nodal area A and enter
//! the tangent as μ A/Δt. The damping does not act in perturbation steps.
//! DEFINITION=CRITICAL DAMPING FRACTION is only available in explicit analysis.

use crate::fem::model::RawKeyword;
use crate::model::Result;

/// Contact damping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Damping {
    /// Critical damping fraction (explicit only) instead of a coefficient.
    pub critical: bool,
    /// μ₀ (coefficient) or critical damping fraction.
    pub mu0: f64,
    /// Clearance c₀ at which the damping vanishes (implicit).
    pub c0: f64,
    /// η (fraction of c₀ with constant μ₀, implicit).
    pub eta: f64,
    /// TANGENT FRACTION (`None` = the procedure's default: 0 in implicit, 1 in explicit analysis).
    pub tangent: Option<f64>,
}

/// Reads *CONTACT DAMPING. `explicit` selects the procedure (CRITICAL DAMPING
/// FRACTION is the default in explicit analysis and not allowed in implicit analysis).
pub fn parse(kw: &RawKeyword, explicit: bool) -> Result<Damping> {
    let def = kw.param_upper("DEFINITION").unwrap_or_else(|| if explicit { "CRITICAL DAMPING FRACTION".into() } else { "DAMPING COEFFICIENT".into() });
    let critical = match def.trim() {
        "CRITICAL DAMPING FRACTION" => {
            if !explicit {
                return kw.err("*CONTACT DAMPING, DEFINITION=CRITICAL DAMPING FRACTION is only available in explicit analysis");
            }
            true
        }
        "DAMPING COEFFICIENT" => false,
        other => return kw.err(format!("DEFINITION={other} is not supported (DAMPING COEFFICIENT, CRITICAL DAMPING FRACTION)")),
    };
    let v = kw.first_floats()?;
    let mu0 = v.first().copied().unwrap_or(0.0);
    if mu0 < 0.0 {
        return kw.err("*CONTACT DAMPING: the damping must be ≥ 0");
    }
    let c0 = v.get(1).copied().unwrap_or(0.0).max(0.0);
    let eta = v.get(2).copied().unwrap_or(0.0).clamp(0.0, 1.0);
    let tangent = match kw.param("TANGENT FRACTION") {
        Some(s) => match crate::fem::model::parse_f64(s.trim()) {
            Some(t) if t >= 0.0 => Some(t),
            _ => return kw.err("TANGENT FRACTION= must be a number ≥ 0"),
        },
        None => None,
    };
    Ok(Damping { critical, mu0, c0, eta, tangent })
}

impl Damping {
    /// Damping coefficient at clearance `c` (c ≤ 0 = contact).
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
