//! Akustiskt medium (*ACOUSTIC MEDIUM): kompressionsmodul K_f och volymetriskt
//! motstånd γ (VOLUMETRIC DRAG), tillsammans med *DENSITY ρ_f.
//!
//! Materialet används bara av de akustiska elementen (`elements/acoustic.rs`)
//! och de akustiska stegen (`steps/acoustic.rs`), som läser egenskaperna med
//! `AcousticProps::from_def`. Materialmodellen finns för att `Prepared` ska
//! kunna bygga materialet; den ger fel om den används i ett mekaniskt element.
//! `tangent_linear` returnerar [K_f] så att elementets `stiffness`/`mass` når
//! kompressionsmodulen via `ElemCtx::material`.
//!
//! *ACOUSTIC MEDIUM får förekomma flera gånger i samma material (BULK MODULUS
//! och VOLUMETRIC DRAG). Kompressionsmodulen får inte bero på
//! temperaturen; det volymetriska motståndet får bero på frekvensen (andra
//! fältet, linjär interpolation, konstant utanför tabellen; används i
//! frekvensdomänen, tidsdomänen använder första raden).
//!
//! Område gaps1: COMPLEX BULK MODULUS, COMPLEX DENSITY (frekvenstabeller),
//! POROUS MODEL=DELANY BAZLEY/MIKI och CAVITATION LIMIT (explicit analys) –
//! se `acoustic_porous.rs`. Utan BULK MODULUS respektive *DENSITY används
//! realdelen på första raden som referensvärde (frekvens- och tidsdomänen
//! utanför *STEADY STATE DYNAMICS).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

#[path = "acoustic_porous.rs"]
pub mod porous;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "akustiskt medium",
    consumes: &["ACOUSTIC MEDIUM"],
    priority: -50,
    build: Some(build),
}];

/// Det akustiska mediets egenskaper.
#[derive(Debug, Clone, Default)]
pub struct AcousticProps {
    pub name: String,
    /// Kompressionsmodul K_f.
    pub bulk: f64,
    /// Densitet ρ_f.
    pub density: f64,
    /// Volymetriskt motstånd: (γ, frekvens i Hz), sorterat efter frekvens.
    pub drag: Vec<(f64, f64)>,
    /// Område gaps1: frekvensberoende material (`acoustic_porous.rs`).
    pub complex: porous::ComplexProps,
}

impl AcousticProps {
    /// Egenskaperna ur en materialdefinition, `None` om *ACOUSTIC MEDIUM saknas.
    pub fn from_def(def: &MaterialDef) -> Result<Option<AcousticProps>> {
        let kws: Vec<&RawKeyword> = def.behaviors.iter().filter(|b| b.name == "ACOUSTIC MEDIUM").collect();
        if kws.is_empty() {
            return Ok(None);
        }
        let mut p = AcousticProps { name: def.name.clone(), density: def.density, ..Default::default() };
        let mut have_bulk = false;
        for kw in kws {
            // område gaps1: komplexa och porösa material, kavitation
            if porous::parse(kw, &mut p.complex)? {
                if kw.has("DEPENDENCIES") && kw.param("DEPENDENCIES").is_some_and(|d| !d.trim().is_empty() && d.trim() != "0") {
                    return kw.err("fältvariabelberoende akustiskt medium stöds inte");
                }
                continue;
            }
            if kw.has("DEPENDENCIES") && kw.param("DEPENDENCIES").is_some_and(|d| !d.trim().is_empty() && d.trim() != "0") {
                return kw.err("fältvariabelberoende akustiskt medium stöds inte");
            }
            if kw.has("VOLUMETRIC DRAG") {
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    p.drag.push((v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0)));
                }
            } else {
                if kw.data.len() > 1 {
                    return kw.err("temperaturberoende kompressionsmodul stöds inte ännu (en datarad)");
                }
                let v = kw.first_floats()?;
                p.bulk = v.first().copied().unwrap_or(0.0);
                have_bulk = true;
            }
        }
        porous::reference_values(&mut p, &mut have_bulk)?;
        if !have_bulk || p.bulk <= 0.0 {
            return fail(format!("material {}: *ACOUSTIC MEDIUM, BULK MODULUS saknas eller är inte positiv", def.name));
        }
        if p.density <= 0.0 {
            return fail(format!("material {}: akustiskt medium kräver *DENSITY > 0", def.name));
        }
        p.drag.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Some(p))
    }

    /// Volymetriskt motstånd vid frekvensen `f` (Hz); tidsdomänen anropar med
    /// `None` och får första raden.
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

    /// Ljudhastigheten c = √(K/ρ).
    pub fn speed(&self) -> f64 {
        (self.bulk / self.density).sqrt()
    }
}

/// Materialmodellen för ett akustiskt medium.
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
        "akustiskt medium"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: akustiskt medium (*ACOUSTIC MEDIUM) används i ett mekaniskt element", self.name))
    }

    /// [K_f] (kompressionsmodulen) för de akustiska elementen.
    fn tangent_linear(&self, _mode: StressMode) -> Result<Vec<f64>> {
        // område gaps1: referensdensiteten som andra värde (COMPLEX DENSITY utan *DENSITY)
        Ok(vec![self.bulk, self.density])
    }
}
