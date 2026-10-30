//! Tidsberoende materialbeteende (krypning och viskoelasticitet): vilket
//! beteende som är aktivt i ett steg och övervakning av tidsintegrationen.
//!
//! Regler:
//! - *CREEP är aktivt i *VISCO. I *STATIC och *DYNAMIC ignoreras krypningen.
//! - *VISCOELASTIC är aktivt i *VISCO och *DYNAMIC. I *STATIC ignoreras det och
//!   svaret är det momentana (instantana moduler, viskösa töjningar oförändrade);
//!   *STATIC, LONG TERM ger den fullt relaxerade lösningen.
//! - Krypning integreras i geometriskt linjära *VISCO-steg explicit (framåt
//!   Euler, "integration scheme 1") och byter till implicit (bakåt Euler) när
//!   plasticitet är aktiv eller stabilitetsgränsen begränsar inkrementet nio
//!   gånger i följd. Med NLGEOM används alltid implicit integration.
//!   CREEP=EXPLICIT på *VISCO tvingar explicit integration.
//!
//! Materialen är delade (`Arc`) och oföränderliga under analysen, så läget och
//! övervakningen ligger i atomära fält (`TimeDependence`) som materialet äger.
//! Elementen anropar materialen parallellt; felmåttet samlas med `fetch_max`
//! (icke-negativa f64 har samma ordning som sina bitmönster).

use super::MaterialModel;
use crate::fem::model::{FeModel, StepDef};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Inga fabriker här: modulen är gemensam för `creep.rs` och `viscoelastic.rs`.
pub static FACTORIES: &[super::MaterialFactory] = &[];

/// Krypning av.
pub const CREEP_OFF: u8 = 0;
/// Explicit (framåt Euler) krypintegration.
pub const CREEP_EXPLICIT: u8 = 1;
/// Implicit (bakåt Euler) krypintegration.
pub const CREEP_IMPLICIT: u8 = 2;

/// Viskoelasticitet ignoreras: momentant svar med instantana moduler.
pub const VISCO_INSTANT: u8 = 0;
/// Viskoelasticitet integreras i tiden.
pub const VISCO_ACTIVE: u8 = 1;
/// Fullt relaxerat svar (*STATIC, LONG TERM).
pub const VISCO_LONG_TERM: u8 = 2;

/// Läge och integrationsövervakning för ett tidsberoende material.
#[derive(Debug)]
pub struct TimeDependence {
    creep: AtomicU8,
    visco: AtomicU8,
    /// Största felmått i senaste utvärderingen (f64-bitar).
    err: AtomicU64,
    /// Minsta stabila tidsinkrement för explicit krypning (f64-bitar).
    dts: AtomicU64,
    /// Absoluta nollpunkten θZ från *PHYSICAL CONSTANTS, ABSOLUTE ZERO= (f64-bitar, NaN om den saknas).
    zero: AtomicU64,
    /// Explicit krypning med temperaturen vid inkrementets början (kopplade
    /// temperatur–förskjutningssteg, där temperaturen är en obekant).
    temp_start: AtomicU8,
    /// Allmänna gaskonstanten R från *PHYSICAL CONSTANTS (f64-bitar, NaN om den saknas).
    gas: AtomicU64,
}

impl Default for TimeDependence {
    fn default() -> Self {
        TimeDependence { creep: AtomicU8::new(CREEP_OFF), visco: AtomicU8::new(VISCO_INSTANT), err: AtomicU64::new(0), dts: AtomicU64::new(f64::INFINITY.to_bits()), zero: AtomicU64::new(f64::NAN.to_bits()), temp_start: AtomicU8::new(0), gas: AtomicU64::new(f64::NAN.to_bits()) }
    }
}

impl TimeDependence {
    pub fn creep(&self) -> u8 {
        self.creep.load(Ordering::Relaxed)
    }

    pub fn visco(&self) -> u8 {
        self.visco.load(Ordering::Relaxed)
    }

    pub fn set(&self, creep: u8, visco: u8) {
        self.creep.store(creep, Ordering::Relaxed);
        self.visco.store(visco, Ordering::Relaxed);
    }

    pub fn set_creep(&self, creep: u8) {
        self.creep.store(creep, Ordering::Relaxed);
    }

    /// Nollställer felmåttet och stabilitetsgränsen (före en utvärdering).
    pub fn reset(&self) {
        self.err.store(0, Ordering::Relaxed);
        self.dts.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }

    /// Felmått i en integrationspunkt (CETOL-jämförelse).
    pub fn record_error(&self, e: f64) {
        if e.is_finite() && e > 0.0 {
            self.err.fetch_max(e.to_bits(), Ordering::Relaxed);
        }
    }

    /// Stabilt tidsinkrement i en integrationspunkt (explicit krypning).
    pub fn record_stable_dt(&self, dt: f64) {
        if dt.is_finite() && dt > 0.0 {
            self.dts.fetch_min(dt.to_bits(), Ordering::Relaxed);
        }
    }

    pub fn error(&self) -> f64 {
        f64::from_bits(self.err.load(Ordering::Relaxed))
    }

    pub fn stable_dt(&self) -> f64 {
        f64::from_bits(self.dts.load(Ordering::Relaxed))
    }

    /// Allmänna gaskonstanten (NaN om *PHYSICAL CONSTANTS, UNIVERSAL GAS CONSTANT= saknas).
    pub fn gas_constant(&self) -> f64 {
        f64::from_bits(self.gas.load(Ordering::Relaxed))
    }

    /// Explicit krypning ska använda temperaturen vid inkrementets början
    /// (*COUPLED TEMPERATURE-DISPLACEMENT).
    pub fn explicit_temp_at_start(&self) -> bool {
        self.temp_start.load(Ordering::Relaxed) != 0
    }

    /// Absoluta nollpunkten (NaN om *PHYSICAL CONSTANTS, ABSOLUTE ZERO= saknas).
    pub fn absolute_zero(&self) -> f64 {
        f64::from_bits(self.zero.load(Ordering::Relaxed))
    }
}

/// UNIVERSAL GAS CONSTANT= från *PHYSICAL CONSTANTS (NaN om den saknas).
pub fn gas_constant(model: &FeModel) -> f64 {
    model
        .extra
        .iter()
        .filter(|k| k.name == "PHYSICAL CONSTANTS")
        .find_map(|k| k.param("UNIVERSAL GAS CONSTANT").and_then(crate::fem::model::parse_f64))
        .unwrap_or(f64::NAN)
}

/// ABSOLUTE ZERO= från *PHYSICAL CONSTANTS (NaN om den saknas).
pub fn absolute_zero(model: &FeModel) -> f64 {
    model
        .extra
        .iter()
        .filter(|k| k.name == "PHYSICAL CONSTANTS")
        .find_map(|k| k.param("ABSOLUTE ZERO").and_then(crate::fem::model::parse_f64))
        .unwrap_or(f64::NAN)
}

/// Lägen (krypning, viskoelasticitet) för ett steg enligt reglerna ovan.
pub fn modes_for(step: &StepDef, nlgeom: bool) -> (u8, u8) {
    let p = step.procedure.as_ref();
    match step.procedure_name() {
        "VISCO" => {
            let explicit = p.and_then(|k| k.param_upper("CREEP")).as_deref() == Some("EXPLICIT");
            let creep = if explicit {
                CREEP_EXPLICIT
            } else if nlgeom {
                CREEP_IMPLICIT
            } else {
                CREEP_EXPLICIT
            };
            (creep, VISCO_ACTIVE)
        }
        "DYNAMIC" => (CREEP_OFF, VISCO_ACTIVE),
        // krypning och viskoelasticitet i transienta kopplade steg
        // (integrationsschema 1/3 som i *VISCO; CREEP=NONE stänger av båda)
        "COUPLED TEMPERATURE-DISPLACEMENT" => match p.and_then(|k| k.param_upper("CREEP")).as_deref() {
            _ if p.is_some_and(|k| k.has("STEADY STATE")) => (CREEP_OFF, VISCO_INSTANT),
            Some("NONE") => (CREEP_OFF, VISCO_INSTANT),
            Some("EXPLICIT") => (CREEP_EXPLICIT, VISCO_ACTIVE),
            _ => (if nlgeom { CREEP_IMPLICIT } else { CREEP_EXPLICIT }, VISCO_ACTIVE),
        },
        "STATIC" if p.is_some_and(|k| k.has("LONG TERM")) => (CREEP_OFF, VISCO_LONG_TERM),
        _ => (CREEP_OFF, VISCO_INSTANT),
    }
}

/// Anropas före varje steg (`steps::run`): sätter läget i alla tidsberoende
/// material. Perturbationssteg ändrar inget (tillståndet återställs ändå).
pub fn begin_step(model: &FeModel, materials: &[Option<Arc<dyn MaterialModel>>], step: &StepDef, nlgeom: bool) {
    // fysikaliska konstanter (även för perturbationssteg, t.ex. *TRS i
    // *STEADY STATE DYNAMICS)
    let (z0, r0) = (absolute_zero(model), gas_constant(model));
    for m in materials.iter().flatten() {
        if let Some(td) = m.time_dependence() {
            td.zero.store(z0.to_bits(), Ordering::Relaxed);
            td.gas.store(r0.to_bits(), Ordering::Relaxed);
        }
    }
    if step.perturbation {
        return;
    }
    let (c, v) = modes_for(step, nlgeom);
    let z = absolute_zero(model);
    for m in materials.iter().flatten() {
        if let Some(td) = m.time_dependence() {
            td.set(c, v);
            td.zero.store(z.to_bits(), Ordering::Relaxed);
            td.temp_start.store((step.procedure_name() == "COUPLED TEMPERATURE-DISPLACEMENT") as u8, Ordering::Relaxed);
        }
    }
}

/// Alla tidsberoende material i modellen.
pub fn all(materials: &[Option<Arc<dyn MaterialModel>>]) -> Vec<&TimeDependence> {
    materials.iter().flatten().filter_map(|m| m.time_dependence()).collect()
}

/// Nollställer övervakningen i alla material.
pub fn reset_all(tds: &[&TimeDependence]) {
    tds.iter().for_each(|t| t.reset());
}

/// (största felmått, minsta stabila tidsinkrement) över alla material.
pub fn monitor(tds: &[&TimeDependence]) -> (f64, f64) {
    tds.iter().fold((0.0f64, f64::INFINITY), |(e, d), t| (e.max(t.error()), d.min(t.stable_dt())))
}

/// *VISCO utan AMPLITUDE= på *STEP: laster och fördefinierade fält verkar
/// direkt (STEP), föreskrivna förskjutningar rampas (kvasistatisk analys).
pub fn default_step_amplitude(step: &StepDef) -> bool {
    step.procedure_name() == "VISCO" && step.header.param("AMPLITUDE").is_none()
}
