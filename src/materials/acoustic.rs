//! Acoustic medium (*ACOUSTIC MEDIUM): bulk modulus K_f and volumetric
//! drag γ (VOLUMETRIC DRAG), together with *DENSITY ρ_f.
//!
//! The material is used only by the acoustic elements (`elements/acoustic.rs`)
//! and the acoustic steps (`steps/acoustic.rs`), which read the properties with
//! `AcousticProps::from_def`. The material model exists so that `Prepared`
//! can build the material; it gives an error if used in a mechanical element.
//! `tangent_linear` returns [K_f] so that the element's `stiffness`/`mass` reaches
//! the bulk modulus via `ElemCtx::material`.
//!
//! *ACOUSTIC MEDIUM may appear several times in the same material (BULK MODULUS
//! and VOLUMETRIC DRAG). The bulk modulus must not depend on
//! temperature; the volumetric drag may depend on frequency (second
//! field, linear interpolation, constant outside the table; used in
//! the frequency domain, the time domain uses the first line).
//!
//! Area gaps1: COMPLEX BULK MODULUS, COMPLEX DENSITY (frequency tables),
//! POROUS MODEL=DELANY BAZLEY/MIKI and CAVITATION LIMIT (explicit analysis) –
//! see `acoustic_porous.rs`. Without BULK MODULUS or *DENSITY respectively,
//! the real part of the first line is used as the reference value (frequency and time domain
//! outside *STEADY STATE DYNAMICS).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

#[path = "acoustic_porous.rs"]
pub mod porous;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "acoustic medium",
    consumes: &["ACOUSTIC MEDIUM"],
    priority: -50,
    build: Some(build),
}];

/// The properties of the acoustic medium.
#[derive(Debug, Clone, Default)]
pub struct AcousticProps {
    pub name: String,
    /// Bulk modulus K_f.
    pub bulk: f64,
    /// Density ρ_f.
    pub density: f64,
    /// Volumetric drag: (γ, frequency in Hz), sorted by frequency.
    pub drag: Vec<(f64, f64)>,
    /// Area gaps1: frequency-dependent material (`acoustic_porous.rs`).
    pub complex: porous::ComplexProps,
}

impl AcousticProps {
    /// The properties from a material definition, `None` if *ACOUSTIC MEDIUM is absent.
    pub fn from_def(def: &MaterialDef) -> Result<Option<AcousticProps>> {
        let kws: Vec<&RawKeyword> = def.behaviors.iter().filter(|b| b.name == "ACOUSTIC MEDIUM").collect();
        if kws.is_empty() {
            return Ok(None);
        }
        let mut p = AcousticProps { name: def.name.clone(), density: def.density, ..Default::default() };
        let mut have_bulk = false;
        for kw in kws {
            // area gaps1: complex and porous materials, cavitation
            if porous::parse(kw, &mut p.complex)? {
                if kw.has("DEPENDENCIES") && kw.param("DEPENDENCIES").is_some_and(|d| !d.trim().is_empty() && d.trim() != "0") {
                    return kw.err("field-variable-dependent acoustic medium is not supported");
                }
                continue;
            }
            if kw.has("DEPENDENCIES") && kw.param("DEPENDENCIES").is_some_and(|d| !d.trim().is_empty() && d.trim() != "0") {
                return kw.err("field-variable-dependent acoustic medium is not supported");
            }
            if kw.has("VOLUMETRIC DRAG") {
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    p.drag.push((v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0)));
                }
            } else {
                if kw.data.len() > 1 {
                    return kw.err("temperature-dependent bulk modulus is not supported yet (one data line)");
                }
                let v = kw.first_floats()?;
                p.bulk = v.first().copied().unwrap_or(0.0);
                have_bulk = true;
            }
        }
        porous::reference_values(&mut p, &mut have_bulk)?;
        if !have_bulk || p.bulk <= 0.0 {
            return fail(format!("material {}: *ACOUSTIC MEDIUM, BULK MODULUS is missing or not positive", def.name));
        }
        if p.density <= 0.0 {
            return fail(format!("material {}: acoustic medium requires *DENSITY > 0", def.name));
        }
        p.drag.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Some(p))
    }

    /// Volumetric drag at the frequency `f` (Hz); the time domain calls with
    /// `None` and gets the first line.
    pub fn drag_at(&self, f: Option<f64>) -> f64 {
        let d = &self.drag;
        if d.is_empty() {
            return 0.0;
        }
        let Some(f) = f else { return d[0].0 };
        if d.len() == 1 || f <= d[0].1 {
            return d[0].0;
        }
        let last = d.len() - 1;
        if f >= d[last].1 {
            return d[last].0;
        }
        let i = d.windows(2).position(|w| f >= w[0].1 && f <= w[1].1).unwrap_or(0);
        let s = (f - d[i].1) / (d[i + 1].1 - d[i].1);
        d[i].0 + s * (d[i + 1].0 - d[i].0)
    }

    /// The speed of sound c = √(K/ρ).
    pub fn speed(&self) -> f64 {
        (self.bulk / self.density).sqrt()
    }
}

/// The material model for an acoustic medium.
#[derive(Debug)]
pub struct AcousticMedium {
    name: String,
    bulk: f64,
    density: f64,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(p) = AcousticProps::from_def(def)? else { return Ok(None) };
    Ok(Some(Arc::new(AcousticMedium { name: def.name.clone(), bulk: p.bulk, density: p.density })))
}

impl MaterialModel for AcousticMedium {
    fn name(&self) -> &str {
        "acoustic medium"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: acoustic medium (*ACOUSTIC MEDIUM) is used in a mechanical element", self.name))
    }

    /// [K_f] (the bulk modulus) for the acoustic elements.
    fn tangent_linear(&self, _mode: StressMode) -> Result<Vec<f64>> {
        // area gaps1: the reference density as the second value (COMPLEX DENSITY without *DENSITY)
        Ok(vec![self.bulk, self.density])
    }
}
