//! Time-dependent material behavior (creep and viscoelasticity): which
//! behavior is active in a step and monitoring of the time integration.
//!
//! Rules:
//! - *CREEP is active in *VISCO. In *STATIC and *DYNAMIC the creep is ignored.
//! - *VISCOELASTIC is active in *VISCO and *DYNAMIC. In *STATIC it is ignored and
//!   the response is the instantaneous one (instantaneous moduli, viscous strains unchanged);
//!   *STATIC, LONG TERM gives the fully relaxed solution.
//! - Creep is integrated explicitly in geometrically linear *VISCO steps (forward
//!   Euler, "integration scheme 1") and switches to implicit (backward Euler) when
//!   plasticity is active or the stability limit restricts the increment nine
//!   times in a row. With NLGEOM implicit integration is always used.
//!   CREEP=EXPLICIT on *VISCO forces explicit integration.
//!
//! The materials are shared (`Arc`) and immutable during the analysis, so the mode and
//! the monitoring live in atomic fields (`TimeDependence`) owned by the material.
//! The elements call the materials in parallel; the error measure is collected with `fetch_max`
//! (non-negative f64 have the same ordering as their bit patterns).

use super::MaterialModel;
use crate::fem::model::{FeModel, StepDef};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// No factories here: the module is shared by `creep.rs` and `viscoelastic.rs`.
pub static FACTORIES: &[super::MaterialFactory] = &[];

/// Creep off.
pub const CREEP_OFF: u8 = 0;
/// Explicit (forward Euler) creep integration.
pub const CREEP_EXPLICIT: u8 = 1;
/// Implicit (backward Euler) creep integration.
pub const CREEP_IMPLICIT: u8 = 2;

/// Viscoelasticity ignored: instantaneous response with instantaneous moduli.
pub const VISCO_INSTANT: u8 = 0;
/// Viscoelasticity integrated in time.
pub const VISCO_ACTIVE: u8 = 1;
/// Fully relaxed response (*STATIC, LONG TERM).
pub const VISCO_LONG_TERM: u8 = 2;

/// Mode and integration monitoring for a time-dependent material.
#[derive(Debug)]
pub struct TimeDependence {
    creep: AtomicU8,
    visco: AtomicU8,
    /// Largest error measure in the latest evaluation (f64 bits).
    err: AtomicU64,
    /// Smallest stable time increment for explicit creep (f64 bits).
    dts: AtomicU64,
    /// Absolute zero θZ from *PHYSICAL CONSTANTS, ABSOLUTE ZERO= (f64 bits, NaN if missing).
    zero: AtomicU64,
    /// Explicit creep with the temperature at the start of the increment (coupled
    /// temperature–displacement steps, where the temperature is an unknown).
    temp_start: AtomicU8,
    /// Universal gas constant R from *PHYSICAL CONSTANTS (f64 bits, NaN if missing).
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

    /// Resets the error measure and the stability limit (before an evaluation).
    pub fn reset(&self) {
        self.err.store(0, Ordering::Relaxed);
        self.dts.store(f64::INFINITY.to_bits(), Ordering::Relaxed);
    }

    /// Error measure at an integration point (CETOL comparison).
    pub fn record_error(&self, e: f64) {
        if e.is_finite() && e > 0.0 {
            self.err.fetch_max(e.to_bits(), Ordering::Relaxed);
        }
    }

    /// Stable time increment at an integration point (explicit creep).
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

    /// Universal gas constant (NaN if *PHYSICAL CONSTANTS, UNIVERSAL GAS CONSTANT= is missing).
    pub fn gas_constant(&self) -> f64 {
        f64::from_bits(self.gas.load(Ordering::Relaxed))
    }

    /// Explicit creep should use the temperature at the start of the increment
    /// (*COUPLED TEMPERATURE-DISPLACEMENT).
    pub fn explicit_temp_at_start(&self) -> bool {
        self.temp_start.load(Ordering::Relaxed) != 0
    }

    /// Absolute zero (NaN if *PHYSICAL CONSTANTS, ABSOLUTE ZERO= is missing).
    pub fn absolute_zero(&self) -> f64 {
        f64::from_bits(self.zero.load(Ordering::Relaxed))
    }
}

/// UNIVERSAL GAS CONSTANT= from *PHYSICAL CONSTANTS (NaN if missing).
pub fn gas_constant(model: &FeModel) -> f64 {
    model
        .extra
        .iter()
        .filter(|k| k.name == "PHYSICAL CONSTANTS")
        .find_map(|k| k.param("UNIVERSAL GAS CONSTANT").and_then(crate::fem::model::parse_f64))
        .unwrap_or(f64::NAN)
}

/// ABSOLUTE ZERO= from *PHYSICAL CONSTANTS (NaN if missing).
pub fn absolute_zero(model: &FeModel) -> f64 {
    model
        .extra
        .iter()
        .filter(|k| k.name == "PHYSICAL CONSTANTS")
        .find_map(|k| k.param("ABSOLUTE ZERO").and_then(crate::fem::model::parse_f64))
        .unwrap_or(f64::NAN)
}

/// Modes (creep, viscoelasticity) for a step according to the rules above.
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
        // creep and viscoelasticity in transient coupled steps
        // (integration scheme 1/3 as in *VISCO; CREEP=NONE turns off both)
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

/// Called before each step (`steps::run`): sets the mode in all time-dependent
/// materials. Perturbation steps change nothing (the state is restored anyway).
pub fn begin_step(model: &FeModel, materials: &[Option<Arc<dyn MaterialModel>>], step: &StepDef, nlgeom: bool) {
    // physical constants (also for perturbation steps, e.g. *TRS in
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

/// All time-dependent materials in the model.
pub fn all(materials: &[Option<Arc<dyn MaterialModel>>]) -> Vec<&TimeDependence> {
    materials.iter().flatten().filter_map(|m| m.time_dependence()).collect()
}

/// Resets the monitoring in all materials.
pub fn reset_all(tds: &[&TimeDependence]) {
    tds.iter().for_each(|t| t.reset());
}

/// (largest error measure, smallest stable time increment) over all materials.
pub fn monitor(tds: &[&TimeDependence]) -> (f64, f64) {
    tds.iter().fold((0.0f64, f64::INFINITY), |(e, d), t| (e.max(t.error()), d.min(t.stable_dt())))
}

/// *VISCO without AMPLITUDE= on *STEP: loads and predefined fields act
/// immediately (STEP), prescribed displacements are ramped (quasi-static analysis).
pub fn default_step_amplitude(step: &StepDef) -> bool {
    step.procedure_name() == "VISCO" && step.header.param("AMPLITUDE").is_none()
}
