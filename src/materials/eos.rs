//! Equations of state (*EOS) for explicit dynamics.
//!
//! Both forms are linear in the specific internal energy Eₘ, p = f(J) + γ(J) Eₘ:
//!
//! ```text
//! TYPE=USUP (Mie–Grüneisen, linear Us–Up Hugoniot; data c₀, s, Γ₀):
//!   p = ρ₀c₀² η (1 − Γ₀η/2) / (1 − sη)² + Γ₀ ρ₀ Eₘ,   η = 1 − ρ₀/ρ = 1 − J
//! TYPE=IDEAL GAS (data R, p_A; *SPECIFIC HEAT c_v):
//!   p + p_A = ρ R (θ − θ_Z) = (R/c_v) ρ Eₘ,   Eₘ = c_v (θ − θ_Z)
//! ```
//!
//! The energy equation (adiabatic) is integrated with the trapezoidal rule over
//! the increment, Eₘ,ₙ₊₁ = Eₘ,ₙ − ½(pₙ + pₙ₊₁)ΔJ/ρ₀ + Δw_dev/ρ₀, and is solved from
//! the equation for p (linear in pₙ₊₁). The work of the bulk viscosity is added to
//! the energy after the increment (`add_bulk_work`). The deviatoric part is hypoelastic with
//! `*ELASTIC, TYPE=SHEAR` (G), otherwise there is no shear stiffness (fluid). J = det F
//! with NLGEOM, otherwise 1 + tr ε. The stability limit uses the isentropic
//! bulk modulus ρc² = J (f_H′(η) + Γ₀p) or (1 + R/c_v)(p + p_A), respectively
//! (`current_bulk_modulus`).
//!
//! Explicit analysis computes the pressure at each integration point (C3D8 fully
//! integrated, without B-bar; `explicit_full_integration`).
//!
//! Initial state (`apply_initial`, called by `steps/explicit.rs` at the
//! start of the analysis): `*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY` (element,
//! Eₘ₀) and for ideal gas otherwise Eₘ₀ = c_v (θ₀ − θ_Z) from
//! `*INITIAL CONDITIONS, TYPE=TEMPERATURE` (the nodal average) and
//! `*PHYSICAL CONSTANTS, ABSOLUTE ZERO=`. The initial pressure p(J = 1, Eₘ₀)
//! is written as the initial stress −p₀ I.
//!
//! `*TENSILE FAILURE` (not with ideal gas): when the pressure becomes more tensile than
//! −σ_c the material point fails. ELEMENT DELETION=YES (default): the stress
//! becomes zero and the element is removed (SDEG = 1 at the point, `steps/model_change.rs`);
//! otherwise SHEAR=BRITTLE (deviator zero for the rest of the analysis) or DUCTILE
//! and PRESSURE=BRITTLE (p ≥ 0) or DUCTILE (p ≥ −σ_c).
//!
//! Area gaps1: TYPE=TABULAR and TYPE=JWL with *DETONATION POINT
//! (`eos_forms.rs`).
//!
//! Area acoustics2: P-α (*EOS COMPACTION, `eos_compaction.rs`) and
//! the work of the bulk viscosity in the energy equation (`add_bulk_work`).
//!
//! Limitations: not IGNITION AND GROWTH or
//! user-defined, no temperature output, only 3D and plane strain.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::assembly::Prepared;
use crate::fem::model::{parse_f64, FeModel, MaterialDef, RawKeyword};
use crate::kinematics::tensor;
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

#[path = "eos_forms.rs"]
pub mod forms;
// area acoustics2: *EOS COMPACTION (P-α)
#[path = "eos_compaction.rs"]
pub mod compaction;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "equation of state (*EOS, TYPE=USUP, IDEAL GAS, TABULAR, JWL)",
    consumes: &["EOS", "ELASTIC", "TENSILE FAILURE", "SPECIFIC HEAT", "DETONATION POINT", "EOS COMPACTION"],
    priority: 20,
    build: Some(build),
}];

/// Parameter that marks a resolved *INITIAL CONDITIONS, TYPE=SPECIFIC
/// ENERGY (lines: element index, Eₘ₀).
pub const RESOLVED: &str = "RUSTYFEM_EOS";

#[derive(Debug, Clone)]
enum Form {
    UsUp { c0: f64, s: f64, gamma0: f64 },
    IdealGas { r: f64, pa: f64, cv: f64 },
    /// Area gaps1.
    Tabular(forms::Tabular),
    Jwl(forms::Jwl),
}

#[derive(Debug, Clone, Copy)]
struct TensileFailure {
    /// Hydrostatic cutoff stress (positive in tension).
    cutoff: f64,
    delete: bool,
    shear_brittle: bool,
    pressure_brittle: bool,
}

#[derive(Debug, Clone)]
pub struct Eos {
    rho0: f64,
    form: Form,
    g: f64,
    tf: Option<TensileFailure>,
}

/// State slots.
const EM: usize = 0;
const JAC: usize = 1;
const PRES: usize = 2;
const FAILED: usize = 3;
/// JWL: arrival time of the detonation wave and the characteristic element length.
const TD: usize = 4;
const LE: usize = 5;

fn first(kw: &RawKeyword) -> Result<Vec<f64>> {
    kw.first_floats()
}

fn tensile_failure(kw: &RawKeyword) -> Result<TensileFailure> {
    if kw.has("DEPENDENCIES") || kw.data.len() > 1 {
        return kw.err("*TENSILE FAILURE: only one constant cutoff stress is supported");
    }
    let cutoff = first(kw)?.first().copied().unwrap_or(f64::NAN);
    if !cutoff.is_finite() {
        return kw.err("*TENSILE FAILURE: give the hydrostatic cutoff stress");
    }
    let delete = kw.flag("ELEMENT DELETION", true)?;
    let brittle = |key: &str| -> Result<bool> {
        match kw.param_upper(key).as_deref() {
            Some("BRITTLE") => Ok(true),
            Some("DUCTILE") => Ok(false),
            None if delete => Ok(false),
            None => kw.err(format!("*TENSILE FAILURE, ELEMENT DELETION=NO requires {key}=BRITTLE or DUCTILE")),
            Some(o) => kw.err(format!("{key}={o} is not supported (BRITTLE, DUCTILE)")),
        }
    };
    Ok(TensileFailure { cutoff, delete, shear_brittle: brittle("SHEAR")?, pressure_brittle: brittle("PRESSURE")? })
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if def.behavior("DETONATION POINT").is_some() && def.behavior("EOS").is_some_and(|k| k.param_upper("TYPE").as_deref() != Some("JWL")) {
        return def.behavior("DETONATION POINT").unwrap().err("*DETONATION POINT requires *EOS, TYPE=JWL");
    }
    let Some(kw) = def.behavior("EOS") else {
        if let Some(t) = def.behavior("TENSILE FAILURE") {
            if def.behavior("PLASTIC").is_some() {
                return t.err("*TENSILE FAILURE with *PLASTIC is not yet supported (only with *EOS)");
            }
        }
        return Ok(None);
    };
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "MIE-GRUNEISEN".into());
    let v = first(kw)?;
    let form = match ty.as_str() {
        "USUP" => {
            if v.is_empty() || !(v[0] > 0.0) {
                return kw.err("*EOS, TYPE=USUP: give c₀ (> 0), s and Γ₀");
            }
            Form::UsUp { c0: v[0], s: v.get(1).copied().unwrap_or(0.0), gamma0: v.get(2).copied().unwrap_or(0.0) }
        }
        "IDEAL GAS" => {
            let r = v.first().copied().unwrap_or(0.0);
            if !(r > 0.0) {
                return kw.err("*EOS, TYPE=IDEAL GAS: give the gas constant R (> 0) and the ambient pressure p_A");
            }
            let Some(sh) = def.behavior("SPECIFIC HEAT") else {
                return kw.err("*EOS, TYPE=IDEAL GAS requires *SPECIFIC HEAT (c_v)");
            };
            if sh.data.len() > 1 || sh.has("DEPENDENCIES") {
                return sh.err("*SPECIFIC HEAT with *EOS, TYPE=IDEAL GAS: only constant c_v is supported");
            }
            let cv = first(sh)?.first().copied().unwrap_or(0.0);
            if !(cv > 0.0) {
                return sh.err("*SPECIFIC HEAT: c_v must be > 0");
            }
            if let Some(t) = def.behavior("TENSILE FAILURE") {
                return t.err("*TENSILE FAILURE cannot be used with *EOS, TYPE=IDEAL GAS");
            }
            Form::IdealGas { r, pa: v.get(1).copied().unwrap_or(0.0), cv }
        }
        "TABULAR" => Form::Tabular(forms::Tabular::parse(kw)?),
        "JWL" => {
            if let Some(t) = def.behavior("TENSILE FAILURE") {
                return t.err("*TENSILE FAILURE cannot be used with *EOS, TYPE=JWL");
            }
            Form::Jwl(forms::Jwl::parse(kw, def)?)
        }
        o => return kw.err(format!("*EOS, TYPE={o} is not yet supported (USUP, IDEAL GAS, TABULAR, JWL)")),
    };
    let g = match def.behavior("ELASTIC") {
        Some(e) => {
            if e.param_upper("TYPE").as_deref() != Some("SHEAR") {
                return e.err("with *EOS, *ELASTIC must have TYPE=SHEAR (the shear modulus G)");
            }
            first(e)?.first().copied().unwrap_or(0.0)
        }
        None => 0.0,
    };
    let rho0 = match def.behavior("DENSITY") {
        Some(d) => first(d)?.first().copied().unwrap_or(0.0),
        None => 0.0,
    };
    if !(rho0 > 0.0) {
        return kw.err("*EOS requires *DENSITY");
    }
    let tf = def.behavior("TENSILE FAILURE").map(tensile_failure).transpose()?;
    let eos = Eos { rho0, form, g, tf };
    // area acoustics2: P-α
    if let Some(c) = def.behavior("EOS COMPACTION") {
        if eos.tf.is_some() {
            return c.err("*EOS COMPACTION with *TENSILE FAILURE is not yet supported");
        }
        return Ok(Some(Arc::new(compaction::build(eos, c)?)));
    }
    Ok(Some(Arc::new(eos)))
}

impl Eos {
    /// p = f(J) + γ(J) Eₘ: (f, γ).
    fn linear_in_energy(&self, j: f64) -> (f64, f64) {
        match self.form {
            Form::UsUp { c0, s, gamma0 } => {
                let eta = 1.0 - j;
                let d = 1.0 - s * eta;
                (self.rho0 * c0 * c0 * eta * (1.0 - 0.5 * gamma0 * eta) / (d * d), gamma0 * self.rho0)
            }
            Form::IdealGas { r, pa, cv } => (-pa, r / cv * self.rho0 / j),
            // area gaps1: p = f₁(ln J) + ρ₀ f₂(ln J) Eₘ
            Form::Tabular(ref t) => {
                let (f1, f2, _, _) = t.eval(j.max(1e-12).ln());
                (f1, self.rho0 * f2)
            }
            Form::Jwl(ref jw) => jw.linear_in_energy(j, self.rho0),
        }
    }

    /// The ambient pressure p_A (ideal gas), otherwise 0.
    fn ambient(&self) -> f64 {
        match self.form {
            Form::IdealGas { pa, .. } => pa,
            _ => 0.0,
        }
    }

    /// Reference modulus for the linear tangent (the wave speed before the first
    /// increment; `current_bulk_modulus` gives the current one).
    fn bulk(&self) -> f64 {
        match self.form {
            Form::UsUp { c0, .. } => self.rho0 * c0 * c0,
            Form::IdealGas { r, pa, cv } => (1.0 + r / cv) * pa.max(1e-6 * self.rho0 * r * 300.0),
            Form::Tabular(ref t) => (-t.eval(0.0).2).max(0.0),
            Form::Jwl(ref jw) => jw.kpd.max(jw.bulk(1.0, jw.e0, self.rho0)),
        }
    }

    fn tangent(&self, mode: StressMode) -> Vec<f64> {
        let n = mode.ncomp();
        let k = self.bulk();
        let mut d = vec![0.0; n * n];
        for i in 0..3.min(n) {
            for j in 0..3.min(n) {
                d[i * n + j] = k - 2.0 * self.g / 3.0;
            }
            d[i * n + i] += 2.0 * self.g;
        }
        for i in 3..n {
            d[i * n + i] = self.g;
        }
        d
    }
}

impl MaterialModel for Eos {
    fn name(&self) -> &str {
        match self.form {
            Form::UsUp { .. } => "*EOS, TYPE=USUP",
            Form::IdealGas { .. } => "*EOS, TYPE=IDEAL GAS",
            Form::Tabular(_) => "*EOS, TYPE=TABULAR",
            Form::Jwl(_) => "*EOS, TYPE=JWL",
        }
    }

    fn is_linear(&self) -> bool {
        false
    }

    fn n_state(&self) -> usize {
        if matches!(self.form, Form::Jwl(_)) {
            6
        } else {
            4
        }
    }

    fn current_bulk_modulus(&self, state: &[f64]) -> Option<f64> {
        let (j, p) = (state[JAC], state[PRES]);
        match self.form {
            // K_s = ρc² = J (f_H′(η) + Γ₀ p) from p(ρ, Eₘ) at constant entropy
            Form::UsUp { c0, s, gamma0: g } => {
                let eta = 1.0 - j;
                let d = 1.0 - s * eta;
                let fp = self.rho0 * c0 * c0 * ((1.0 - g * eta) * d + 2.0 * s * eta * (1.0 - 0.5 * g * eta)) / (d * d * d);
                Some((j * (fp + g * p)).max(0.0))
            }
            // ρc² = γ (p + p_A), γ = c_p/c_v = 1 + R/c_v
            Form::IdealGas { r, pa, cv } => Some(((1.0 + r / cv) * (p + pa)).max(0.0)),
            // area gaps1: ρc² = −(f₁′ + ρ₀ f₂′ Eₘ) + f₂ p J
            Form::Tabular(ref t) => {
                let (_, f2, d1, d2) = t.eval(j.max(1e-12).ln());
                Some((-(d1 + self.rho0 * d2 * state[EM]) + f2 * p * j).max(0.0))
            }
            Form::Jwl(ref jw) => Some(jw.kpd.max(jw.bulk(j, state[EM], self.rho0))),
        }
    }

    fn explicit_full_integration(&self) -> bool {
        true
    }

    fn init_state(&self, state: &mut [f64]) {
        state[EM] = 0.0;
        state[JAC] = 1.0;
        state[PRES] = 0.0;
        state[FAILED] = 0.0;
        if matches!(self.form, Form::Jwl(_)) {
            state[TD] = 0.0;
            state[LE] = 0.0;
        }
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(self.tangent(mode))
    }

    fn output_names(&self) -> &[&'static str] {
        // failed points with element deletion (`steps/model_change.rs` reads SDEG)
        match self.tf {
            Some(t) if t.delete => &["SDEG"],
            _ => &[],
        }
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        match self.tf {
            Some(t) if t.delete => vec![state[FAILED]],
            _ => Vec::new(),
        }
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mode = input.mode;
        if !matches!(mode, StressMode::ThreeD | StressMode::PlaneStrain) {
            return fail("*EOS is only supported in 3D and plane strain");
        }
        let n = mode.ncomp();
        let j1 = match input.defgrad {
            Some(f) => tensor::det(&f),
            None => 1.0 + input.strain[..3].iter().sum::<f64>(),
        };
        let (em0, j0) = (state_old[EM], state_old[JAC]);
        let failed0 = state_old.get(FAILED).copied().unwrap_or(0.0) > 0.5;
        let p0 = -input.stress_old[..3].iter().sum::<f64>() / 3.0;
        // deviatoric part (hypoelastic)
        let tr = input.dstrain[..3].iter().sum::<f64>() / 3.0;
        let mut s_new = vec![0.0; n];
        let dev = |i: usize| if i < 3 { (input.stress_old[i] + p0, input.dstrain[i] - tr) } else { (input.stress_old[i], 0.5 * input.dstrain[i]) };
        for (i, s) in s_new.iter_mut().enumerate() {
            let (so, de) = dev(i);
            *s = so + 2.0 * self.g * de;
        }
        let dj = j1 - j0;
        // ideal gas: the heat from the absolute pressure p + p_A (the same
        // adiabatic process for p + p_A with and without ambient pressure)
        let pa = self.ambient();
        let solve = |s_new: &[f64]| -> (f64, f64) {
            // Δw_dev per unit mass: (s : Δe) · J/ρ₀ (current volume per mass)
            let mut wdev = 0.0;
            for (i, sn) in s_new.iter().enumerate() {
                let (so, de) = dev(i);
                wdev += 0.5 * (so + sn) * de * if i < 3 { 1.0 } else { 2.0 };
            }
            let wdev = wdev * j1 / self.rho0;
            let (f, gam) = self.linear_in_energy(j1);
            let a = em0 + wdev - 0.5 * (p0 + 2.0 * pa) * dj / self.rho0;
            ((f + gam * a) / (1.0 + 0.5 * gam * dj / self.rho0), wdev)
        };
        let (mut p1, mut wdev) = solve(&s_new);
        // area gaps1: JWL – the pressure from K_pd before detonation, then F_b p_JWL
        if let Form::Jwl(ref jw) = self.form {
            let (td, le) = (state_old[TD], state_old[LE]);
            let fb = jw.burn(input.time, td, le);
            let mut wd = 0.0;
            for (i, sn) in s_new.iter().enumerate() {
                let (so, de) = dev(i);
                wd += 0.5 * (so + sn) * de * if i < 3 { 1.0 } else { 2.0 };
            }
            wdev = wd * j1 / self.rho0;
            let a = em0 + wdev - 0.5 * p0 * dj / self.rho0;
            p1 = if fb <= 0.0 {
                -jw.kpd * j1.max(1e-12).ln()
            } else {
                let (f, gam) = jw.linear_in_energy(j1, self.rho0);
                fb * (f + gam * a) / (1.0 + 0.5 * fb * gam * dj / self.rho0)
            };
            state_new[TD] = td;
            state_new[LE] = le;
        }
        let mut failed = failed0;
        if let Some(t) = self.tf {
            if !failed && p1 < -t.cutoff {
                failed = true;
            }
            if failed {
                if t.delete || t.shear_brittle {
                    s_new.iter_mut().for_each(|s| *s = 0.0);
                    wdev = solve(&s_new).1;
                }
                p1 = if t.delete {
                    0.0
                } else if t.pressure_brittle {
                    p1.max(0.0)
                } else {
                    p1.max(-t.cutoff)
                };
            }
        }
        let em1 = em0 + wdev - 0.5 * (p0 + p1 + 2.0 * pa) * dj / self.rho0;
        let mut stress = s_new;
        for v in stress.iter_mut().take(3) {
            *v -= p1;
        }
        state_new[EM] = em1;
        state_new[JAC] = j1;
        state_new[PRES] = p1;
        state_new[FAILED] = if failed { 1.0 } else { 0.0 };
        Ok(MatOutput { stress, tangent: self.tangent(mode) })
    }

    fn init_at_point(&self, state: &mut [f64], x: [f64; 3], length: f64) {
        if let Form::Jwl(ref jw) = self.form {
            state[TD] = jw.arrival(x);
            state[LE] = length;
        }
    }

    fn init_energy(&self, state: &mut [f64], em: Option<f64>, temperature: Option<f64>, abs_zero: f64) -> Result<Option<(f64, f64)>> {
        // the pressure is determined by the initial stress (Us–Up: 0) or
        // the initial temperature (ideal gas); *INITIAL CONDITIONS, TYPE=SPECIFIC
        // ENERGY only changes the internal energy ALLIE (the same motion with
        // and without, ALLIE(0) = Σ m Eₘ₀)
        match self.form {
            Form::UsUp { .. } | Form::Tabular(_) => Ok(em.map(|e| (0.0, e))),
            // area gaps1: Eₘ₀ = E₀ unless otherwise specified, the pressure starts at 0
            Form::Jwl(ref jw) => {
                let e = em.unwrap_or(jw.e0);
                state[EM] = e;
                Ok(Some((0.0, e)))
            }
            Form::IdealGas { cv, .. } => {
                let Some(t) = temperature else {
                    return fail("*EOS, TYPE=IDEAL GAS requires *INITIAL CONDITIONS, TYPE=TEMPERATURE (the initial temperature)");
                };
                let eth = cv * (t - abs_zero);
                if !(eth > 0.0) {
                    return fail("*EOS, TYPE=IDEAL GAS: the initial temperature must be above absolute zero");
                }
                state[EM] = eth;
                let (f, gam) = self.linear_in_energy(1.0);
                let p = f + gam * eth;
                state[PRES] = p;
                Ok(Some((p, em.unwrap_or(eth))))
            }
        }
    }
}

/// Absolute zero θ_Z (*PHYSICAL CONSTANTS, ABSOLUTE ZERO=), otherwise 0.
fn absolute_zero(m: &FeModel) -> Result<f64> {
    match m.extra.iter().rev().find(|k| k.name == "PHYSICAL CONSTANTS").and_then(|k| k.param("ABSOLUTE ZERO").map(|v| (k, v))) {
        Some((k, v)) => parse_f64(v).ok_or_else(|| FemError(format!("{}: invalid ABSOLUTE ZERO={v}", k.location()))),
        None => Ok(0.0),
    }
}

/// The initial state for *EOS materials (see the module documentation):
/// specific energy and initial pressure in the element states `states`.
/// Returns the internal energy Σ m Eₘ₀ at the start of the analysis (ALLIE).
pub fn apply_initial(p: &Prepared, states: &mut [Vec<f64>]) -> Result<f64> {
    let mut energy = 0.0;
    let m = p.model;
    let ne = m.elements.len();
    let mut em: Vec<Option<f64>> = vec![None; ne];
    for kw in m.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED)) {
        for d in &kw.data {
            let v = kw.floats(d)?;
            if let (Some(&e), Some(&x)) = (v.first(), v.get(1)) {
                em[e as usize] = Some(x);
            }
        }
    }
    let temps = super::thermal::TemperatureField::build(m)?;
    let tz = absolute_zero(m)?;
    for e in 0..ne {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let Some(mat) = ctx.material else { continue };
        if !mat.name().starts_with("*EOS") {
            if em[e].is_some() {
                return fail(format!("{}: *INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY requires a material with *EOS", m.elem_name(e)));
            }
            continue;
        }
        let el = &m.elements[e];
        let temp = temps.as_ref().map(|t| {
            let v = t.at(&el.nodes, 0.0);
            v.iter().take(el.nodes.len()).sum::<f64>() / el.nodes.len().max(1) as f64
        });
        let nip = el.etype.n_ips();
        let ns = mat.n_state();
        let len = states[e].len();
        let nc = if el.etype.family() == crate::elements::Family::Continuum3D { 6 } else { 4 };
        if nip == 0 || len < nip * (nc + ns) {
            continue;
        }
        let stride = nc + ns;
        // element mass (equal at the integration points)
        let mass = crate::steps::explicit::dt::geometry(el.etype.family(), el.etype.name(), &ctx.x).map_or(0.0, |g| g.vol) * ctx.density;
        // area gaps1: integration point positions and characteristic length (JWL)
        let ipx = forms_ip_positions(el.etype.name(), &ctx.x, nip);
        let le = crate::steps::explicit::dt::geometry(el.etype.family(), el.etype.name(), &ctx.x).map_or(0.0, |g| g.vol).abs().cbrt();
        for ip in 0..nip {
            let st = &mut states[e][ip * stride..(ip + 1) * stride];
            let pr = mat.init_energy(&mut st[nc..], em[e], temp, tz).map_err(|err| FemError(format!("{}: {}", m.elem_name(e), err.0)))?;
            mat.init_at_point(&mut st[nc..], ipx[ip], le);
            if let Some((pr, e0)) = pr {
                for v in st.iter_mut().take(3) {
                    *v = -pr;
                }
                energy += mass * e0 / nip as f64;
            }
        }
    }
    Ok(energy)
}

/// Area explicit_adv: sets J = ρ₀/ρ in an *EOS state after
/// the Euler advection (the ΔJ of the energy equation is computed from this value).
pub fn set_jacobian(mat: &dyn MaterialModel, state: &mut [f64], j: f64) {
    if mat.name().starts_with("*EOS") && state.len() > JAC {
        state[JAC] = j;
    }
}

/// Area gaps1: integration point positions (standard order) for C3D8
/// (2 × 2 × 2 Gauss), otherwise the centroid for all points.
fn forms_ip_positions(etype: &str, x: &[[f64; 3]], nip: usize) -> Vec<[f64; 3]> {
    let n = x.len().max(1);
    let mut c = [0.0; 3];
    for p in x {
        for d in 0..3 {
            c[d] += p[d] / n as f64;
        }
    }
    if nip == 8 && x.len() == 8 && etype.starts_with("C3D8") {
        let g = 1.0 / 3f64.sqrt();
        let nat: [[f64; 3]; 8] = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]];
        let mut out = Vec::with_capacity(8);
        for k in 0..8 {
            // order: ξ fastest, then η, then ζ
            let (a, b, cz) = ([-g, g][k % 2], [-g, g][(k / 2) % 2], [-g, g][k / 4]);
            let mut p = [0.0; 3];
            for (i, xi) in x.iter().enumerate() {
                let w = 0.125 * (1.0 + a * nat[i][0]) * (1.0 + b * nat[i][1]) * (1.0 + cz * nat[i][2]);
                for d in 0..3 {
                    p[d] += w * xi[d];
                }
            }
            out.push(p);
        }
        return out;
    }
    vec![c; nip.max(1)]
}

/// Area acoustics2: the work of the bulk viscosity in the energy equation (is
/// included): Eₘ increases by σ_bv ε̇_vol Δt/ρ at all the element's
/// integration points. `work` holds (element, σ_bv ε̇_vol/ρ) from
/// `steps/explicit_bulk.rs`.
pub fn add_bulk_work(p: &Prepared, states: &mut [Vec<f64>], work: &[(usize, f64)], dt: f64) {
    let m = p.model;
    for &(e, w) in work {
        if w == 0.0 {
            continue;
        }
        let Some(mi) = m.elements[e].section.and_then(|s| m.sections[s].material_index) else { continue };
        let Some(mat) = p.materials.get(mi).and_then(|x| x.as_deref()) else { continue };
        if !mat.name().starts_with("*EOS") {
            continue;
        }
        let el = &m.elements[e];
        let nip = el.etype.n_ips();
        let ns = mat.n_state();
        let nc = if el.etype.family() == crate::elements::Family::Continuum3D { 6 } else { 4 };
        let stride = nc + ns;
        let Some(st) = states.get_mut(e) else { continue };
        if nip == 0 || st.len() < nip * stride {
            continue;
        }
        for ip in 0..nip {
            st[ip * stride + nc + EM] += w * dt;
        }
    }
}
