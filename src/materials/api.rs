//! The material interface: stress update with state per integration point.
//!
//! A material model is built from a `MaterialDef` (the behaviors *ELASTIC,
//! *PLASTIC, ... as written in the .inp file) by a registered `MaterialFactory`.
//! The element calls `update` at every integration point with the strain in
//! the integration point's stress state (`StressMode`).
//!
//! Component order, with shear strains as engineering strains (γ):
//! - `ThreeD`:      [11, 22, 33, 12, 13, 23]
//! - `PlaneStrain`: [11, 22, 33, 12]   (ε33 = 0)
//! - `Axisymmetric`:[11, 22, 33, 12]   (33 = hoop)
//! - `PlaneStress`: [11, 22, 12]       (σ33 = 0)
//! - `Uniaxial`:    [11]               (trusses)

use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressMode {
    ThreeD,
    PlaneStrain,
    Axisymmetric,
    PlaneStress,
    Uniaxial,
}

impl StressMode {
    pub fn ncomp(self) -> usize {
        match self {
            StressMode::ThreeD => 6,
            StressMode::PlaneStrain | StressMode::Axisymmetric => 4,
            StressMode::PlaneStress => 3,
            StressMode::Uniaxial => 1,
        }
    }
}

/// Input to a stress update.
pub struct MatInput<'a> {
    pub mode: StressMode,
    /// Total (small) strain at the end of the increment.
    pub strain: &'a [f64],
    /// Strain increment since the start of the increment.
    pub dstrain: &'a [f64],
    /// Stress at the start of the increment.
    pub stress_old: &'a [f64],
    pub time: f64,
    pub dt: f64,
    /// Temperature at the end of the increment and its change (0 if no temperature).
    pub temperature: f64,
    pub dtemp: f64,
    /// Deformation gradient with NLGEOM (row-wise), otherwise `None`.
    pub defgrad: Option<[[f64; 3]; 3]>,
}

/// Result of a stress update.
#[derive(Debug, Clone, Default)]
pub struct MatOutput {
    pub stress: Vec<f64>,
    /// Consistent tangent dσ/dε, ncomp × ncomp row-wise.
    pub tangent: Vec<f64>,
}

pub trait MaterialModel: Send + Sync {
    /// Model name for messages, e.g. "ELASTIC".
    fn name(&self) -> &str;

    /// Linear material: the stress is `D ε` with constant `D` and no state.
    fn is_linear(&self) -> bool {
        false
    }

    /// Linear response when the temperature is given (e.g. thermal strain on top of
    /// a linear material): the stress is affine in the strain within an
    /// increment. Used by the coupled analysis (`steps/heat_coupled.rs`)
    /// to solve the mechanical part with a single solve.
    fn is_linear_at_given_temperature(&self) -> bool {
        self.is_linear()
    }

    /// Number of state variables per integration point.
    fn n_state(&self) -> usize {
        0
    }

    fn init_state(&self, _state: &mut [f64]) {}

    /// Initial state with initial conditions at the integration point
    /// (*INITIAL CONDITIONS, TYPE=STRESS and TYPE=RATIO, see
    /// `initial_state.rs`): the stress `stress` (the components for `mode`)
    /// and the void ratio e0 if given. Called after `init_state`; the element's
    /// stress is already set. Default: error if the stress is nonzero, since
    /// the material then does not account for an initial stress.
    fn init_state_at(&self, _mode: StressMode, stress: &[f64], _void_ratio: Option<f64>, _state: &mut [f64]) -> Result<()> {
        if stress.iter().any(|&s| s != 0.0) {
            return fail(format!("material model {} does not support initial stress (*INITIAL CONDITIONS, TYPE=STRESS)", self.name()));
        }
        Ok(())
    }

    /// (E, ν) for elements that need the elastic moduli directly (beams, trusses).
    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        None
    }

    /// Stress update: new stress, tangent and new state.
    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput>;

    /// Stress update with the element's characteristic length `length`
    /// (Hillerborg: softening materials whose fracture energy or failure displacement
    /// is regularized with the element size). The continuum elements
    /// (`elements::common::response`) call this. Default: `update`.
    fn update_with_length(&self, input: &MatInput, _length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update(input, state_old, state_new)
    }

    /// The material has an unsymmetric tangent that requires the unsymmetric
    /// solver (selected automatically, e.g. for *CONCRETE DAMAGED PLASTICITY
    /// even without *STEP, UNSYMM=YES). Default: no.
    fn requires_unsymmetric(&self) -> bool {
        false
    }

    /// Rotates tensorial state variables (plastic strain, back stresses,
    /// ...) with the incremental rotation `r` under NLGEOM (x' = R x Rᵀ), so that
    /// they follow the material like the stress does (Hughes–Winget). Default: no
    /// state to rotate.
    fn rotate_state(&self, _r: &[[f64; 3]; 3], _state: &mut [f64]) {}

    /// *ANNEAL (`steps/anneal.rs`): makes the material point stress-free at the
    /// current total strain `strain` (11, 22, 33, 12, 13, 23 with
    /// engineering shear) and resets the hardening. Returns `false` if
    /// the material does not support annealing. Default: no.
    fn anneal(&self, _strain: &[f64; 6], _state: &mut [f64]) -> bool {
        false
    }

    /// Current (isentropic) bulk modulus ρc² from the material point's
    /// state for the stability limit in explicit dynamics (e.g. *EOS).
    /// Default: `None` (the elastic bulk modulus is used).
    fn current_bulk_modulus(&self, _state: &[f64]) -> Option<f64> {
        None
    }

    /// Explicit analysis integrates the material at every integration point without
    /// B-bar/F-bar (full integration of the volumetric part too) for C3D8 with NLGEOM
    /// (*EOS: the pressure varies between the points). Default: `false`.
    fn explicit_full_integration(&self) -> bool {
        false
    }

    /// Initial state for equations of state (*EOS): the specific
    /// energy `em` (*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY) or
    /// the initial temperature with absolute zero. Returns
    /// (the initial pressure p₀ (stress −p₀ I), specific internal energy for
    /// ALLIE), `None` = nothing to do. Default: `None`.
    fn init_energy(&self, _state: &mut [f64], _em: Option<f64>, _temperature: Option<f64>, _abs_zero: f64) -> Result<Option<(f64, f64)>> {
        Ok(None)
    }

    /// Feature area gaps1: the integration point's position `x` and the element's
    /// characteristic length at the start of the analysis (*EOS, TYPE=JWL: the detonation wave's
    /// arrival time). Called by `eos::apply_initial` after `init_energy`.
    /// Default: nothing.
    fn init_at_point(&self, _state: &mut [f64], _x: [f64; 3], _length: f64) {}

    /// Feature area materials5: the initial value r of the relative density
    /// (*INITIAL CONDITIONS, TYPE=RELATIVE DENSITY) for porous materials
    /// (*POROUS METAL PLASTICITY). Called by
    /// `porous_metal::apply_relative_density` after `init_state`. Returns
    /// `false` if the material has no relative density. Default: no.
    fn init_relative_density(&self, _r: f64, _state: &mut [f64]) -> bool {
        false
    }

    /// Elastic (linear) tangent for linear stiffness and perturbation steps.
    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let n = mode.ncomp();
        let z = vec![0.0; n];
        let mut st = vec![0.0; self.n_state()];
        self.init_state(&mut st);
        let mut st2 = st.clone();
        let inp = MatInput { mode, strain: &z, dstrain: &z, stress_old: &z, time: 0.0, dt: 0.0, temperature: 0.0, dtemp: 0.0, defgrad: None };
        Ok(self.update(&inp, &st, &mut st2)?.tangent)
    }

    /// Names of extra output per integration point, e.g. ["PEEQ"].
    fn output_names(&self) -> &[&'static str] {
        &[]
    }

    /// Values for `output_names` from the state.
    fn output_values(&self, _state: &[f64]) -> Vec<f64> {
        Vec::new()
    }

    /// Output fields with component names for the stress state `mode`, e.g.
    /// `[("PEEQ", ["PEEQ"]), ("PE", ["PE11", "PE22", ...])]`.
    /// Default: `output_names` as scalar fields.
    fn output_fields(&self, _mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        self.output_names().iter().map(|n| (*n, vec![*n])).collect()
    }

    /// Values for `output_fields`: all components in sequence.
    fn output_field_values(&self, _mode: StressMode, state: &[f64]) -> Vec<f64> {
        self.output_values(state)
    }

    /// Hyperelastic materials: the split into deviatoric and volumetric parts
    /// for elements that handle the volumetric part themselves (F-bar with NLGEOM, plane
    /// stress, hybrid elements). See `hyperelastic_split.rs`.
    fn hyperelastic(&self) -> Option<&dyn super::hyperelastic_split::HyperModel> {
        None
    }

    /// Time-dependent materials (creep, viscoelasticity): mode per step and
    /// monitoring of the time integration for *VISCO (CETOL). Materials that
    /// wrap others (e.g. thermal expansion) must forward the call.
    /// See `time_dependence.rs`.
    fn time_dependence(&self) -> Option<&super::time_dependence::TimeDependence> {
        None
    }

    /// Viscoelasticity in the frequency domain (*VISCOELASTIC, FREQUENCY=...) for
    /// *STEADY STATE DYNAMICS, DIRECT: (ΔGs, ΔKs, Gℓ, Kℓ) at the angular frequency
    /// `omega` and the temperature `temperature`, where ΔGs and ΔKs are
    /// the storage moduli's contributions to the moduli in `tangent_linear` and Gℓ,
    /// Kℓ the loss moduli (see `viscoelastic_freq.rs`). Default: nothing.
    fn frequency_moduli(&self, _omega: f64, _temperature: f64) -> Option<[f64; 4]> {
        None
    }

    /// Traction–separation law for cohesive elements (*ELASTIC, TYPE=TRACTION,
    /// see `cohesive.rs`). Default: nothing.
    fn cohesive(&self) -> Option<&super::cohesive::TractionSeparation> {
        None
    }
}

/// Builds a material model from a material definition.
///
/// `build` returns `Ok(None)` if the factory does not recognize the material (e.g.
/// wrong TYPE= on *ELASTIC). Among the factories that respond, the one with the highest
/// `priority` is chosen. All behaviors in the material must be consumed by the chosen
/// factory (`consumes`) or by a passive entry (`build: None`), otherwise it is
/// an error – no behavior is silently ignored.
pub struct MaterialFactory {
    pub name: &'static str,
    /// Behaviors (keywords without '*') that the factory uses.
    pub consumes: &'static [&'static str],
    pub priority: i32,
    pub build: Option<fn(&MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>>>,
}

/// Helper for material models: row-wise matrix times vector.
pub fn matvec(d: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n).map(|i| (0..n).map(|j| d[i * n + j] * x[j]).sum()).collect()
}

pub fn unsupported<T>(def: &MaterialDef, what: &str) -> Result<T> {
    fail(format!("material {}: {what}", def.name))
}
