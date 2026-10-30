//! Thermal expansion (*EXPANSION) and the predefined temperature field.
//!
//! Temperature fields in stress analysis:
//! - *INITIAL CONDITIONS, TYPE=TEMPERATURE: nodal temperatures at the start
//!   (nodes not specified have temperature 0),
//! - *TEMPERATURE in a step: new nodal temperatures at the end of the step
//!   (OP=MOD default, OP=NEW first resets to the initial temperatures).
//!   Within the step the temperature is ramped linearly from the previous value (or
//!   applied immediately with AMPLITUDE=STEP on *STEP). The values are inherited by subsequent steps.
//!
//! In first-order elements the mean of the element's
//! nodal temperatures is used at all integration points
//! (`ElemCtx::temperature_at`).
//!
//! Thermal strain with the secant coefficient α(T) and the reference temperature
//! θ0 (ZERO=, default 0):
//!
//!   ε_th = α(T)(T − θ0) − α(T_I)(T_I − θ0)
//!
//! where T_I is the initial temperature. It is updated incrementally in
//! the integration point state: Δε_th = α(T)(T − θ0) − α(T−ΔT)(T−ΔT − θ0).
//! TYPE=ISO (default), ORTHO (α11, α22, α33 in the material's local axes)
//! and ANISO (α11, α22, α33, α12, α13, α23; the shear part as a tensor component,
//! i.e. the engineering shear strain becomes 2α12 ΔT).
//! The mechanical strain ε − ε_th is passed to the underlying material
//! (elasticity, plasticity, ...).

use super::anisotropic::mode_components;
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{FeModel, MaterialDef, RawKeyword};
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "thermal expansion",
    // all behaviors consumed by underlying models in RustyFEM
    consumes: &[
        "CREEP",
        "CYCLIC HARDENING",
        "DAMAGE EVOLUTION",
        "DAMAGE INITIATION",
        "DRUCKER PRAGER",
        "DRUCKER PRAGER HARDENING",
        "ELASTIC",
        "EXPANSION",
        "HYPERELASTIC",
        "PLASTIC",
        "POTENTIAL",
        "RATE DEPENDENT",
        "VISCOELASTIC",
    ],
    priority: 100,
    build: Some(build),
}];

/// Expansion coefficient: (α components (6, tensor), temperature) in increasing temperature.
#[derive(Debug, Clone)]
pub struct Expansion {
    pub table: Vec<([f64; 6], f64)>,
    pub zero: f64,
}

impl Expansion {
    pub fn parse(kw: &RawKeyword) -> Result<Expansion> {
        let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
        let n = match ty.as_str() {
            "ISO" => 1,
            "ORTHO" => 3,
            "ANISO" => 6,
            o => return kw.err(format!("TYPE={o} is not supported (ISO, ORTHO, ANISO)")),
        };
        for p in ["DEPENDENCIES", "USER", "PORE FLUID"] {
            if kw.has(p) {
                return kw.err(format!("the {p} parameter is not supported yet"));
            }
        }
        let zero = match kw.param("ZERO") {
            Some(z) => crate::fem::model::parse_f64(z).ok_or_else(|| crate::model::FemError(format!("{}: ZERO={z} is not a number", kw.location())))?,
            None => 0.0,
        };
        let mut table = Vec::new();
        for d in kw.records() {
            let v = kw.floats(&d)?;
            if v.len() < n {
                return kw.err_at(d.line, format!("TYPE={ty} requires {n} coefficients"));
            }
            let mut a = [0.0; 6];
            match n {
                1 => {
                    a[0] = v[0];
                    a[1] = v[0];
                    a[2] = v[0];
                }
                _ => a[..n].copy_from_slice(&v[..n]),
            }
            let t = v.get(n).copied().unwrap_or(0.0);
            if v.len() > n + 1 && v[n + 1..].iter().any(|&x| x != 0.0) {
                return kw.err_at(d.line, "field-dependent expansion is not supported yet");
            }
            table.push((a, t));
        }
        if table.is_empty() {
            return kw.err("expansion coefficient missing");
        }
        if table.len() > 1 {
            for w in table.windows(2) {
                if !(w[1].1 > w[0].1) {
                    return kw.err("the temperatures in the table must increase");
                }
            }
        }
        Ok(Expansion { table, zero })
    }

    /// α(T) (linear interpolation, constant outside the table).
    pub fn alpha(&self, t: f64) -> [f64; 6] {
        let tb = &self.table;
        if tb.len() == 1 || t <= tb[0].1 {
            return tb[0].0;
        }
        for w in tb.windows(2) {
            if t < w[1].1 {
                let f = (t - w[0].1) / (w[1].1 - w[0].1);
                return std::array::from_fn(|i| w[0].0[i] + f * (w[1].0[i] - w[0].0[i]));
            }
        }
        tb[tb.len() - 1].0
    }

    /// α(T)(T − θ0) as strain (engineering shear).
    pub fn secant_strain(&self, t: f64) -> [f64; 6] {
        let a = self.alpha(t);
        std::array::from_fn(|i| a[i] * (t - self.zero) * if i < 3 { 1.0 } else { 2.0 })
    }
}

/// Thermal strain on top of another material.
pub struct Thermal {
    pub exp: Expansion,
    pub inner: Arc<dyn MaterialModel>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("EXPANSION") else { return Ok(None) };
    let exp = Expansion::parse(kw)?;
    let mut rest = def.clone();
    rest.behaviors.retain(|b| b.name != "EXPANSION");
    let inner = super::build(&rest)?;
    Ok(Some(Arc::new(Thermal { exp, inner })))
}

const NTH: usize = 6;

impl MaterialModel for Thermal {
    fn name(&self) -> &str {
        "EXPANSION"
    }

    fn n_state(&self) -> usize {
        NTH + self.inner.n_state()
    }

    fn init_state(&self, state: &mut [f64]) {
        state[..NTH].iter_mut().for_each(|x| *x = 0.0);
        self.inner.init_state(&mut state[NTH..]);
    }

    fn is_linear_at_given_temperature(&self) -> bool {
        self.inner.is_linear()
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        super::plasticity::rotate_v6(r, &mut state[..NTH], true);
        self.inner.rotate_state(r, &mut state[NTH..]);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let t = input.temperature;
        let t_old = t - input.dtemp;
        let (a, b) = (self.exp.secant_strain(t), self.exp.secant_strain(t_old));
        let idx = mode_components(input.mode);
        let mut th = [0.0; NTH];
        for i in 0..NTH {
            th[i] = state_old[i] + a[i] - b[i];
        }
        let mut eps = input.strain.to_vec();
        let mut deps = input.dstrain.to_vec();
        for (k, &i) in idx.iter().enumerate() {
            eps[k] -= th[i];
            deps[k] -= th[i] - state_old[i];
        }
        let inp = MatInput { strain: &eps, dstrain: &deps, ..*input };
        let (old_th, old_in) = state_old.split_at(NTH);
        let _ = old_th;
        let (new_th, new_in) = state_new.split_at_mut(NTH);
        new_th.copy_from_slice(&th);
        self.inner.update(&inp, old_in, new_in)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn time_dependence(&self) -> Option<&super::time_dependence::TimeDependence> {
        self.inner.time_dependence()
    }

    fn output_names(&self) -> &[&'static str] {
        self.inner.output_names()
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        self.inner.output_values(&state[NTH.min(state.len())..])
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        v.push(("THE", the_labels(mode)));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = self.inner.output_field_values(mode, &state[NTH.min(state.len())..]);
        let idx: &[usize] = match mode {
            StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
            StressMode::Uniaxial => &[0],
            _ => &[0, 1, 2, 3],
        };
        v.extend(idx.iter().map(|&i| state.get(i).copied().unwrap_or(0.0)));
        v
    }
}

fn the_labels(mode: StressMode) -> Vec<&'static str> {
    match mode {
        StressMode::ThreeD => vec!["THE11", "THE22", "THE33", "THE12", "THE13", "THE23"],
        StressMode::Uniaxial => vec!["THE11"],
        _ => vec!["THE11", "THE22", "THE33", "THE12"],
    }
}

// ---------------------------------------------------------------------------
// The temperature field
// ---------------------------------------------------------------------------

/// A step in the temperature history: [start, start + length], temperatures at
/// the start and end, linear ramp or step.
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    period: f64,
    t0: Vec<f64>,
    t1: Vec<f64>,
    ramp: bool,
}

/// Nodal temperatures as a function of total time. The vectors have 2·nn slots:
/// the nodal temperatures followed by the temperature gradients through the thickness
/// (shells, third field on the *TEMPERATURE line; feature area struct_nonlin).
#[derive(Debug, Clone)]
pub struct TemperatureField {
    initial: Vec<f64>,
    segments: Vec<Segment>,
    nn: usize,
    gradients: bool,
}

/// Data lines with resolved nodes: fields [node index, value] (see `inp/keywords/temperature.rs`).
pub const RESOLVED: &str = "_NODINDEX";

fn apply(kw: &RawKeyword, t: &mut [f64]) -> Result<()> {
    let nn = t.len() / 2;
    for d in &kw.data {
        let v = kw.floats(d)?;
        let n = v[0] as usize;
        if n < nn {
            t[n] = v.get(1).copied().unwrap_or(0.0);
            t[nn + n] = v.get(2).copied().unwrap_or(0.0);
        }
    }
    Ok(())
}

/// Procedures without time duration (linear perturbations).
fn timeless(proc_name: &str) -> bool {
    matches!(proc_name, "FREQUENCY" | "BUCKLE" | "COMPLEX FREQUENCY" | "MODAL DYNAMIC" | "STEADY STATE DYNAMICS" | "RESPONSE SPECTRUM")
}

impl TemperatureField {
    /// Builds the temperature history, or `None` if the model has no temperatures.
    pub fn build(model: &FeModel) -> Result<Option<TemperatureField>> {
        let nn = model.nodes.len();
        let ic: Vec<&RawKeyword> = model
            .extra
            .iter()
            .filter(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED) && k.param_upper("TYPE").as_deref() == Some("TEMPERATURE"))
            .collect();
        let any_step = model.steps.iter().any(|s| s.extra.iter().any(|k| k.name == "TEMPERATURE"));
        if ic.is_empty() && !any_step {
            return Ok(None);
        }
        let gradients = ic.iter().copied().chain(model.steps.iter().flat_map(|s| s.extra.iter().filter(|k| k.name == "TEMPERATURE"))).any(|k| k.data.iter().any(|d| d.fields.len() > 2));
        let mut initial = vec![0.0; 2 * nn];
        for k in ic {
            apply(k, &mut initial)?;
        }
        let mut cur = initial.clone();
        let mut time = 0.0;
        let mut segments = Vec::new();
        for s in &model.steps {
            if s.perturbation || timeless(s.procedure_name()) {
                continue;
            }
            let period = s
                .procedure
                .as_ref()
                .and_then(|p| p.first_floats().ok())
                .and_then(|v| v.get(1).copied())
                .filter(|&x| x > 0.0)
                .unwrap_or(1.0);
            let mut next = cur.clone();
            for k in s.extra.iter().filter(|k| k.name == "TEMPERATURE") {
                if k.param_upper("OP").as_deref() == Some("NEW") {
                    next.clone_from(&initial);
                }
                apply(k, &mut next)?;
            }
            segments.push(Segment { start: time, period, t0: cur.clone(), t1: next.clone(), ramp: !s.amplitude_step && !super::time_dependence::default_step_amplitude(s) });
            time += period;
            cur = next;
        }
        Ok(Some(TemperatureField { initial, segments, nn, gradients }))
    }

    /// The model has temperature gradients through the thickness.
    pub fn has_gradients(&self) -> bool {
        self.gradients
    }

    /// The temperature gradients through the thickness at the nodes `nodes` at total time `time`.
    pub fn gradient_at(&self, nodes: &[usize], time: f64) -> Vec<f64> {
        let idx: Vec<usize> = nodes.iter().map(|&n| self.nn + n).collect();
        self.at(&idx, time)
    }

    /// Temperatures at the nodes `nodes` at total time `time`.
    pub fn at(&self, nodes: &[usize], time: f64) -> Vec<f64> {
        let seg = self.segments.iter().find(|s| time <= s.start + s.period * (1.0 + 1e-12));
        match seg {
            _ if time <= 0.0 || self.segments.is_empty() => nodes.iter().map(|&n| self.initial[n]).collect(),
            Some(s) => {
                let f = ((time - s.start) / s.period).clamp(0.0, 1.0);
                let a = if s.ramp { f } else if f > 0.0 { 1.0 } else { 0.0 };
                nodes.iter().map(|&n| s.t0[n] + a * (s.t1[n] - s.t0[n])).collect()
            }
            None => {
                let s = self.segments.last().unwrap();
                nodes.iter().map(|&n| s.t1[n]).collect()
            }
        }
    }
}
