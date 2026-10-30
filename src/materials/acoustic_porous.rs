//! Område gaps1: frekvensberoende akustiska medier och kavitation
//! (`*ACOUSTIC MEDIUM`, inkluderas från `acoustic.rs`).
//!
//! - COMPLEX BULK MODULUS: rader (Re K̃, Im K̃, frekvens), Im K̃ ≥ 0.
//! - COMPLEX DENSITY: rader (Re ρ̃, Im ρ̃, frekvens), Im ρ̃ ≤ 0.
//!   Båda interpoleras linjärt i frekvensen och är konstanta utanför
//!   tabellen. De gäller bara i *STEADY STATE DYNAMICS.
//! - POROUS MODEL=DELANY BAZLEY (standard), DELANY-BAZLEY eller MIKI: data
//!   flödesresistiviteten R. Med x = ρ_f f / R (f i Hz) och c₀ = √(K_f/ρ_f):
//!
//!   ```text
//!   k̃ = (ω/c₀)(1 + C₁ x^C₂ − i C₃ x^C₄),   Z̃ = ρ_f c₀ (1 + C₅ x^C₆ − i C₇ x^C₈)
//!   ρ̃ = k̃ Z̃ / ω,   K̃ = K_f (Z̃/k̃)/|Z̃/k̃|   (område acoustics2, se `complex_at`)
//!   ```
//!
//!   (konstanterna för Delany–Bazley och Miki nedan). Craggs modell ges som
//!   vanligt med
//!   densitet ρ_f K_s Ω, volymetriskt motstånd R Ω och K_f = ρ_f c_f².
//! - CAVITATION LIMIT p_c (bara explicit analys): p = max(p_v, p_c − p₀)
//!   med pseudotrycket p_v = −K_f ε_V (`steps/acoustic_explicit.rs`).
//!   Implicit analys ignorerar gränsen.
//!
//! Volymetriskt motstånd γ läggs till den komplexa densiteten som
//! ρ̃ + γ/(iΩ) (`AcousticProps::complex_at`).

use super::AcousticProps;
use crate::fem::model::RawKeyword;
use crate::model::Result;

/// Porös modell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PorousModel {
    DelanyBazley,
    Miki,
}

impl PorousModel {
    /// C₁ … C₈.
    fn constants(self) -> [f64; 8] {
        match self {
            PorousModel::DelanyBazley => [0.0978, -0.7, 0.189, -0.595, 0.0571, -0.754, 0.087, -0.732],
            PorousModel::Miki => [0.1227, -0.618, 0.1792, -0.618, 0.0786, -0.632, 0.1205, -0.632],
        }
    }
}

/// Frekvensberoende delar av ett akustiskt medium.
#[derive(Debug, Clone, Default)]
pub struct ComplexProps {
    /// (Re, Im, frekvens), sorterad efter frekvens.
    pub bulk: Vec<(f64, f64, f64)>,
    pub density: Vec<(f64, f64, f64)>,
    /// Porös modell och flödesresistivitet.
    pub porous: Option<(PorousModel, f64)>,
    /// Kavitationsgräns p_c (absolut tryck), `None` = ingen kavitation.
    pub cavitation: Option<f64>,
}

impl ComplexProps {
    /// Någon egenskap beror på frekvensen.
    pub fn any(&self) -> bool {
        !self.bulk.is_empty() || !self.density.is_empty() || self.porous.is_some()
    }
}

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64, f64)>> {
    let mut t = Vec::new();
    for d in kw.records() {
        let v = kw.floats(&d)?;
        if v.is_empty() {
            continue;
        }
        t.push((v[0], v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)));
    }
    if t.is_empty() {
        return kw.err("datarader saknas");
    }
    t.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    Ok(t)
}

/// Läser en *ACOUSTIC MEDIUM-post med någon av parametrarna ovan; `false`
/// om posten är BULK MODULUS eller VOLUMETRIC DRAG (läses av `acoustic.rs`).
pub fn parse(kw: &RawKeyword, c: &mut ComplexProps) -> Result<bool> {
    if kw.has("COMPLEX BULK MODULUS") {
        c.bulk = table(kw)?;
        if c.bulk.iter().any(|r| !(r.0 > 0.0)) {
            return kw.err("COMPLEX BULK MODULUS: realdelen ska vara positiv");
        }
        return Ok(true);
    }
    if kw.has("COMPLEX DENSITY") {
        c.density = table(kw)?;
        if c.density.iter().any(|r| !(r.0 > 0.0)) {
            return kw.err("COMPLEX DENSITY: realdelen ska vara positiv");
        }
        return Ok(true);
    }
    if kw.has("POROUS MODEL") {
        let m = match kw.param_upper("POROUS MODEL").as_deref().map(str::trim) {
            None | Some("") | Some("DELANY BAZLEY") | Some("DELANY-BAZLEY") => PorousModel::DelanyBazley,
            Some("MIKI") => PorousModel::Miki,
            Some(o) => return kw.err(format!("POROUS MODEL={o} stöds inte (DELANY BAZLEY, DELANY-BAZLEY, MIKI)")),
        };
        let r = kw.first_floats()?.first().copied().unwrap_or(0.0);
        if !(r > 0.0) {
            return kw.err("POROUS MODEL: ange flödesresistiviteten (> 0)");
        }
        c.porous = Some((m, r));
        return Ok(true);
    }
    if kw.has("CAVITATION LIMIT") {
        if kw.data.len() > 1 {
            return kw.err("CAVITATION LIMIT: bara en konstant gräns stöds");
        }
        c.cavitation = Some(kw.first_floats()?.first().copied().unwrap_or(0.0));
        return Ok(true);
    }
    Ok(false)
}

/// Referensvärden (realdelen på första raden) när BULK MODULUS eller
/// *DENSITY saknas.
pub fn reference_values(p: &mut AcousticProps, have_bulk: &mut bool) -> Result<()> {
    // COMPLEX DENSITY och *DENSITY får inte anges samtidigt
    if !p.complex.density.is_empty() && p.density > 0.0 {
        return crate::model::fail(format!("material {}: *ACOUSTIC MEDIUM, COMPLEX DENSITY kan inte kombineras med *DENSITY", p.name));
    }
    if !p.complex.bulk.is_empty() && *have_bulk {
        return crate::model::fail(format!("material {}: *ACOUSTIC MEDIUM, COMPLEX BULK MODULUS kan inte kombineras med BULK MODULUS", p.name));
    }
    if p.complex.porous.is_some() && (!p.complex.bulk.is_empty() || !p.complex.density.is_empty()) {
        return crate::model::fail(format!("material {}: POROUS MODEL kan inte kombineras med COMPLEX BULK MODULUS/COMPLEX DENSITY", p.name));
    }
    if !*have_bulk {
        if let Some(r) = p.complex.bulk.first() {
            p.bulk = r.0;
            *have_bulk = true;
        }
    }
    if !(p.density > 0.0) {
        if let Some(r) = p.complex.density.first() {
            p.density = r.0;
        }
    }
    Ok(())
}

fn interp(t: &[(f64, f64, f64)], f: f64) -> (f64, f64) {
    if t.len() == 1 || f <= t[0].2 {
        return (t[0].0, t[0].1);
    }
    let l = t.len() - 1;
    if f >= t[l].2 {
        return (t[l].0, t[l].1);
    }
    let i = t.windows(2).position(|w| f >= w[0].2 && f <= w[1].2).unwrap_or(0);
    let s = (f - t[i].2) / (t[i + 1].2 - t[i].2);
    (t[i].0 + s * (t[i + 1].0 - t[i].0), t[i].1 + s * (t[i + 1].1 - t[i].1))
}

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cdiv(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

impl AcousticProps {
    /// Materialet har frekvensberoende egenskaper i *STEADY STATE DYNAMICS.
    pub fn freq_dependent(&self) -> bool {
        !self.drag.is_empty() || self.complex.any()
    }

    /// Kontroll av den porösa modellen vid frekvensen `f` (Re ρ̃ > 0
    /// och Im K̃ > 0, annars fel).
    pub fn porous_check(&self, f: f64) -> Result<()> {
        if self.complex.porous.is_none() || f <= 0.0 {
            return Ok(());
        }
        let (rho, k) = self.complex_at(f);
        if !(rho.0 > 0.0) {
            return crate::model::fail(format!("material {}: realdelen av den komplexa densiteten från POROUS MODEL är inte positiv vid {f} Hz (utanför modellens giltighetsområde)", self.name));
        }
        if !(k.1 > 0.0) {
            return crate::model::fail(format!("material {}: imaginärdelen av den komplexa kompressionsmodulen från POROUS MODEL är inte positiv vid {f} Hz (utanför modellens giltighetsområde)", self.name));
        }
        Ok(())
    }

    /// (ρ̃, K̃) vid frekvensen `f` (Hz) som (Re, Im): komplex densitet inklusive
    /// volymetriskt motstånd γ/(iΩ) och komplex kompressionsmodul.
    pub fn complex_at(&self, f: f64) -> ((f64, f64), (f64, f64)) {
        let w = 2.0 * std::f64::consts::PI * f;
        let (mut rho, k) = if let Some((m, r)) = self.complex.porous {
            let c = m.constants();
            let c0 = (self.bulk / self.density).sqrt();
            let x = (self.density * f / r).max(1e-300);
            let kk = (w / c0 * (1.0 + c[0] * x.powf(c[1])), -w / c0 * c[2] * x.powf(c[3]));
            let z = (self.density * c0 * (1.0 + c[4] * x.powf(c[5])), -self.density * c0 * c[6] * x.powf(c[7]));
            if w > 0.0 {
                let rho = cmul(kk, z);
                // område acoustics2: kompressionsmodulen har fasen av ω Z̃/k̃ men
                // beloppet K_f (|ω Z̃/k̃| = K_f |Ẑ|/|k̂| med Ẑ = Z̃/(ρc₀), k̂ = k̃ c₀/ω),
                // se tests/cases/gaps1/ac_poros_*_ssd
                let kb = cdiv((w * z.0, w * z.1), kk);
                let f = (kk.0.hypot(kk.1) / (w / c0)) / (z.0.hypot(z.1) / (self.density * c0));
                let kb = (kb.0 * f, kb.1 * f);
                ((rho.0 / w, rho.1 / w), kb)
            } else {
                ((self.density, 0.0), (self.bulk, 0.0))
            }
        } else {
            let rho = if self.complex.density.is_empty() { (self.density, 0.0) } else { interp(&self.complex.density, f) };
            let k = if self.complex.bulk.is_empty() { (self.bulk, 0.0) } else { interp(&self.complex.bulk, f) };
            (rho, k)
        };
        let g = self.drag_at(Some(f));
        if g != 0.0 && w > 0.0 {
            rho.1 -= g / w;
        }
        (rho, k)
    }
}
