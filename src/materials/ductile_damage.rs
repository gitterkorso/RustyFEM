//! Progressive damage for ductile metals: *DAMAGE INITIATION
//! (CRITERION=DUCTILE or SHEAR) with *DAMAGE EVOLUTION (TYPE=DISPLACEMENT
//! or ENERGY) on top of plasticity (*PLASTIC, von Mises).
//!
//! The underlying plasticity model gives the effective (undamaged)
//! stress σ̄ and the equivalent plastic strain ε̄p; the stress is
//! σ = (1 − D) σ̄.
//!
//! Initiation (ω ≥ 1), integrated with the average of the initiation strain at
//! the start and end of the increment, Δω = Δε̄p / ½(ε̄D(ηₙ) + ε̄D(ηₙ₊₁)):
//! - DUCTILE: ω_D = ∫ dε̄p / ε̄D(η), triaxiality η = −p/q, table (ε̄D, η),
//! - SHEAR: ω_S = ∫ dε̄p / ε̄S(θs), θs = (q + ks p)/τmax with τmax = (σ1 − σ3)/2
//!   and KS = ks (default 0.3), table (ε̄S, θs),
//! - JOHNSON COOK: ω = ∫ dε̄p / ε̄D with ε̄D = [d1 + d2 e^(−d3 η)][1 + d4 ln(ε̄̇p/ε̇0)]
//!   (1 + d5 θ̂) (`plasticity_rate::JcFailure`), ε̄̇p = Δε̄p/Δt; output JCCRT.
//!
//! The tables are interpolated linearly and are constant outside. From
//! the increment where the criterion is reached, the equivalent plastic
//! displacement ū̇p = L ε̄̇p grows with the element's characteristic length L
//! (Hillerborg, `update_with_length`), and so does the damage d. Linear
//! softening counts the whole initiation increment's ūp, exponential softening only the part
//! after the initiation point (linearly interpolated in ω):
//! - TYPE=DISPLACEMENT, SOFTENING=LINEAR: d = ūp/ūf,
//! - TYPE=DISPLACEMENT, SOFTENING=EXPONENTIAL: d = (1 − e^(−α ūp/ūf))/(1 − e^(−α)),
//! - TYPE=DISPLACEMENT, SOFTENING=TABULAR: d(ūp) from a table (d, ūp),
//! - TYPE=ENERGY, SOFTENING=LINEAR: d = ūp/ūf with ūf = 2Gf/σy0 (σy0 =
//!   the yield stress at the initiation point, interpolated within the increment),
//! - TYPE=ENERGY, SOFTENING=EXPONENTIAL: d = 1 − exp(−∫σy dūp / Gf).
//!
//! With several criteria D = max dᵢ (DEGRADATION=MAXIMUM, default) or
//! D = 1 − Π(1 − dᵢ) (MULTIPLICATIVE). D is limited to Dmax = 0.99 so that
//! the stiffness never becomes zero (element deletion is not supported).
//!
//! The tangent is (1 − D) C̄ when the damage does not grow and otherwise numerical
//! (forward difference of the whole update); it is unsymmetric and negative
//! after the onset of softening, so the unsymmetric (LU) solver is selected.
//!
//! Output: the underlying model's fields (PEEQ, PE) plus DUCTCRT, SHRCRT
//! (ω) and SDEG (D).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "ductile damage",
    consumes: &["ELASTIC", "PLASTIC", "CYCLIC HARDENING", "DAMAGE INITIATION", "DAMAGE EVOLUTION", "POTENTIAL", "RATE DEPENDENT"],
    priority: 20,
    build: Some(build),
}];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Criterion {
    Ductile,
    Shear { ks: f64 },
    /// CRITERION=JOHNSON COOK: ε̄D(η, ε̄̇p, θ) in closed form (`plasticity_rate.rs`);
    /// the state stores ε̄D instead of η.
    JohnsonCook(super::plasticity_rate::JcFailure),
}

#[derive(Debug, Clone)]
enum Evolution {
    /// No evolution: only the initiation criterion is reported.
    None,
    DispLinear { uf: f64 },
    DispExp { uf: f64, alpha: f64 },
    DispTable(Vec<(f64, f64)>),
    EnergyLinear { gf: f64 },
    EnergyExp { gf: f64 },
}

#[derive(Debug, Clone)]
struct Damage {
    crit: Criterion,
    /// (x, ε̄ at initiation), x = η or θs.
    table: Vec<(f64, f64)>,
    evol: Evolution,
    /// Feature area materials5: rate-/temperature-dependent table
    /// (`damage_tables.rs`), evaluated with the increment's ε̄̇pl = Δε̄pl/Δt and
    /// temperature at both ends (η at the start and end).
    dep: Option<super::damage_tables::DepTable>,
}

pub struct DuctileDamage {
    inner: Arc<dyn MaterialModel>,
    peeq: usize,
    crits: Vec<Damage>,
    multiplicative: bool,
}

/// Per criterion: [ω, ūp, d, σy0 (ENERGY LINEAR), ∫σy dūp (ENERGY EXPONENTIAL),
/// η or θs at the end of the increment].
const PER: usize = 6;
/// Shared, after the criteria: [D, q̄ at the end of the increment]. D is limited to:
const DMAX: f64 = 0.99;

/// Dmax: 0.99 or according to *SECTION CONTROLS/element deletion (`steps/model_change.rs`).
fn dmax() -> f64 {
    crate::steps::model_change::max_degradation().unwrap_or(DMAX)
}

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "strain-rate- or temperature-dependent initiation is not supported yet");
        }
        let (e, x) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(e > 0.0) {
            return kw.err_at(d.line, "the initiation strain must be positive");
        }
        if t.last().is_some_and(|&(xl, _)| !(x > xl)) {
            return kw.err_at(d.line, "the triaxiality (or θs) must increase in the table");
        }
        t.push((x, e));
    }
    if t.is_empty() {
        return kw.err("data is missing");
    }
    Ok(t)
}

fn evolution(kw: &RawKeyword) -> Result<Evolution> {
    if kw.has("DEPENDENCIES") || kw.data.len() > 1 && kw.param_upper("SOFTENING").as_deref() != Some("TABULAR") {
        return kw.err("temperature- or field-dependent damage evolution is not supported yet");
    }
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "DISPLACEMENT".into());
    let soft = kw.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
    if kw.has("MIXED MODE BEHAVIOR") || kw.has("MODE MIX RATIO") || kw.has("POWER") {
        return kw.err("mixed modes (MIXED MODE BEHAVIOR) are not supported (only for cohesive elements)");
    }
    let v = kw.first_floats()?;
    let a = v.first().copied().unwrap_or(0.0);
    match (ty.as_str(), soft.as_str()) {
        ("DISPLACEMENT", "TABULAR") => {
            let mut t: Vec<(f64, f64)> = Vec::new();
            for d in &kw.data {
                let v = kw.floats(d)?;
                let (dd, u) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                if !(0.0..1.0).contains(&dd) || t.last().is_some_and(|&(ul, _)| !(u > ul)) {
                    return kw.err_at(d.line, "the table must have 0 ≤ d < 1 and increasing ūp");
                }
                t.push((u, dd));
            }
            Ok(Evolution::DispTable(t))
        }
        _ if !(a > 0.0) => kw.err("the failure displacement or fracture energy must be positive"),
        ("DISPLACEMENT", "LINEAR") => Ok(Evolution::DispLinear { uf: a }),
        ("DISPLACEMENT", "EXPONENTIAL") => {
            let alpha = v.get(1).copied().unwrap_or(0.0);
            if !(alpha > 0.0) {
                return kw.err("SOFTENING=EXPONENTIAL requires the exponent α > 0");
            }
            Ok(Evolution::DispExp { uf: a, alpha })
        }
        ("ENERGY", "LINEAR") => Ok(Evolution::EnergyLinear { gf: a }),
        ("ENERGY", "EXPONENTIAL") => Ok(Evolution::EnergyExp { gf: a }),
        (t, s) => kw.err(format!("TYPE={t}, SOFTENING={s} is not supported yet")),
    }
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if super::cohesive::is_cohesive(def) {
        return Ok(None); // traction-separation for cohesive elements: materials/cohesive.rs
    }
    if super::xfem_damage::claims(def) {
        return Ok(None); // XFEM crack: materials/xfem_damage.rs
    }
    if super::hashin::claims(def) || super::forming_limit::claims(def) {
        return Ok(None); // feature area materials5: materials/hashin.rs, materials/forming_limit.rs
    }
    if def.behavior("DAMAGE INITIATION").is_none() {
        if let Some(e) = def.behavior("DAMAGE EVOLUTION") {
            return e.err("*DAMAGE EVOLUTION requires a preceding *DAMAGE INITIATION");
        }
        return Ok(None);
    }
    // criteria in order; *DAMAGE EVOLUTION belongs to the nearest preceding *DAMAGE INITIATION
    let mut crits: Vec<Damage> = Vec::new();
    let mut degradation: Option<String> = None;
    for b in &def.behaviors {
        match b.name.as_str() {
            "DAMAGE INITIATION" => {
                let crit = match b.param_upper("CRITERION").as_deref() {
                    Some("DUCTILE") => Criterion::Ductile,
                    Some("SHEAR") => {
                        let ks = match b.param("KS") {
                            Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: invalid KS", b.location())))?,
                            None => 0.3,
                        };
                        Criterion::Shear { ks }
                    }
                    Some("JOHNSON COOK") => Criterion::JohnsonCook(super::plasticity_rate::JcFailure::parse(b)?),
                    Some(o) => return b.err(format!("CRITERION={o} is not supported yet (DUCTILE, SHEAR, JOHNSON COOK)")),
                    None => return b.err("CRITERION= is missing"),
                };
                if b.has("DEPENDENCIES") {
                    return b.err("field-dependent initiation is not supported yet");
                }
                // Johnson–Cook: no table, the state holds ε̄D directly (unbounded range)
                let dep = if matches!(crit, Criterion::JohnsonCook(_)) { None } else { super::damage_tables::DepTable::parse(b)? };
                let table = if matches!(crit, Criterion::JohnsonCook(_)) || dep.is_some() { vec![(-1e300, 1.0), (1e300, 1.0)] } else { table(b)? };
                crits.push(Damage { crit, table, evol: Evolution::None, dep });
            }
            "DAMAGE EVOLUTION" => {
                let Some(c) = crits.last_mut() else {
                    return b.err("*DAMAGE EVOLUTION requires a preceding *DAMAGE INITIATION");
                };
                if !matches!(c.evol, Evolution::None) {
                    return b.err("the criterion already has a *DAMAGE EVOLUTION");
                }
                let deg = b.param_upper("DEGRADATION").unwrap_or_else(|| "MAXIMUM".into());
                if deg != "MAXIMUM" && deg != "MULTIPLICATIVE" {
                    return b.err(format!("DEGRADATION={deg} is not supported (MAXIMUM, MULTIPLICATIVE)"));
                }
                if degradation.as_ref().is_some_and(|d| *d != deg) {
                    return b.err("all criteria must have the same DEGRADATION");
                }
                degradation = Some(deg);
                c.evol = evolution(b)?;
            }
            _ => {}
        }
    }
    let Some(pk) = def.behavior("PLASTIC") else {
        return fail(format!("material {}: *DAMAGE INITIATION requires *PLASTIC", def.name));
    };
    let base = MaterialDef {
        name: def.name.clone(),
        behaviors: def.behaviors.iter().filter(|b| !b.name.starts_with("DAMAGE ")).cloned().collect(),
        density: def.density,
    };
    let inner = super::build(&base)?;
    let Some(peeq) = inner.output_names().iter().position(|n| *n == "PEEQ") else {
        return pk.err("the damage model requires a plasticity model with PEEQ");
    };
    Ok(Some(Arc::new(DuctileDamage { inner, peeq, crits, multiplicative: degradation.as_deref() == Some("MULTIPLICATIVE") })))
}

fn interp(t: &[(f64, f64)], x: f64) -> f64 {
    super::principal::interp(t, x).0
}

/// 3D stress (Voigt, tensor shear) from the stress in a stress state.
fn stress3(mode: StressMode, s: &[f64]) -> [f64; 6] {
    match mode {
        StressMode::ThreeD => [s[0], s[1], s[2], s[3], s[4], s[5]],
        StressMode::PlaneStrain | StressMode::Axisymmetric => [s[0], s[1], s[2], s[3], 0.0, 0.0],
        StressMode::PlaneStress => [s[0], s[1], 0.0, s[2], 0.0, 0.0],
        StressMode::Uniaxial => [s[0], 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

impl DuctileDamage {
    fn n_inner(&self) -> usize {
        self.inner.n_state()
    }

    fn base(&self) -> usize {
        self.n_inner()
    }

    fn d_index(&self) -> usize {
        self.base() + PER * self.crits.len()
    }

    /// The whole update: effective update, criteria, damage and stress.
    fn eval(&self, input: &MatInput, len: f64, st: &[f64], stn: &mut [f64]) -> Result<(MatOutput, bool)> {
        let ni = self.n_inner();
        let out = self.inner.update(input, &st[..ni], &mut stn[..ni])?;
        stn[ni..].copy_from_slice(&st[ni..]);
        let p_old = self.inner.output_values(&st[..ni])[self.peeq];
        let p_new = self.inner.output_values(&stn[..ni])[self.peeq];
        let dp = (p_new - p_old).max(0.0);
        let s3 = stress3(input.mode, &out.stress);
        let (l, _) = super::principal::principal(&s3);
        let (p, q, _) = super::principal::pq_theta(&l);
        let di = self.d_index();
        let q_old = st[di + 1];
        stn[di + 1] = q;
        let mut growing = false;
        for (k, c) in self.crits.iter().enumerate() {
            let o = self.base() + PER * k;
            let x = match c.crit {
                Criterion::Ductile => {
                    if q > 0.0 {
                        -p / q
                    } else {
                        0.0
                    }
                }
                Criterion::Shear { ks } => {
                    let tau = 0.5 * (l[0] - l[2]);
                    if tau > 0.0 {
                        (q + ks * p) / tau
                    } else {
                        0.0
                    }
                }
                Criterion::JohnsonCook(jc) => {
                    let eta = if q > 0.0 { -p / q } else { 0.0 };
                    let rate = if input.dt > 0.0 { dp / input.dt } else { 0.0 };
                    jc.strain(eta, rate, input.temperature)
                }
            };
            // η (θs) is limited to the table's range (the table is constant outside)
            let x = x.clamp(c.table[0].0, c.table[c.table.len() - 1].0);
            stn[o + 5] = x;
            if dp <= 0.0 {
                continue;
            }
            // the initiation criterion with the average of ε̄D at the start
            // and end of the increment
            let w_old = st[o];
            // (Johnson–Cook: ε̄D is stored directly; 0 = not set yet, use the new value)
            let eps_d = |v: f64| match c.crit {
                Criterion::JohnsonCook(_) if v > 0.0 => v,
                Criterion::JohnsonCook(_) => x,
                // feature area materials5: both ends with the increment's rate and
                // temperature
                _ if c.dep.is_some() => c.dep.as_ref().map_or(1.0, |t| t.eval(v, if input.dt > 0.0 { dp / input.dt } else { 0.0 }, input.temperature)),
                _ => interp(&c.table, v),
            };
            let w = w_old + dp / (0.5 * (eps_d(st[o + 5]) + eps_d(x)));
            stn[o] = w;
            if w < 1.0 || matches!(c.evol, Evolution::None) {
                continue;
            }
            // fraction of the increment after initiation
            let after = if w_old < 1.0 { (w - 1.0) / (w - w_old) } else { 1.0 };
            if w_old < 1.0 {
                // the yield stress at the initiation point
                stn[o + 3] = q - after * (q - q_old);
                stn[o + 4] = 0.0;
            }
            // linear softening counts the whole initiation increment, exponential
            // only the part after the initiation point
            let du_full = len * dp;
            let du_after = after * du_full;
            let d = match &c.evol {
                Evolution::None => 0.0,
                Evolution::DispLinear { uf } => {
                    stn[o + 1] = st[o + 1] + du_full;
                    stn[o + 1] / uf
                }
                Evolution::DispExp { uf, alpha } => {
                    stn[o + 1] = st[o + 1] + du_after;
                    (1.0 - (-alpha * stn[o + 1] / uf).exp()) / (1.0 - (-alpha).exp())
                }
                Evolution::DispTable(t) => {
                    stn[o + 1] = st[o + 1] + du_full;
                    interp(t, stn[o + 1])
                }
                Evolution::EnergyLinear { gf } => {
                    stn[o + 1] = st[o + 1] + du_full;
                    stn[o + 1] * stn[o + 3] / (2.0 * gf)
                }
                Evolution::EnergyExp { gf } => {
                    stn[o + 1] = st[o + 1] + du_after;
                    // ∫σy dūp with the yield stress at the end of the increment
                    let g = stn[o + 4] + q * du_after;
                    stn[o + 4] = g;
                    1.0 - (-g / gf).exp()
                }
            };
            stn[o + 2] = d.clamp(st[o + 2], dmax().max(st[o + 2]));
            growing = true;
        }
        let dtot = if self.multiplicative {
            1.0 - self.crits.iter().enumerate().map(|(k, _)| 1.0 - stn[self.base() + PER * k + 2]).product::<f64>()
        } else {
            self.crits.iter().enumerate().map(|(k, _)| stn[self.base() + PER * k + 2]).fold(0.0, f64::max)
        };
        stn[di] = dtot.min(dmax()).max(st[di]);
        let f = 1.0 - stn[di];
        let stress = out.stress.iter().map(|s| f * s).collect();
        let tangent = out.tangent.iter().map(|c| f * c).collect();
        Ok((MatOutput { stress, tangent }, growing))
    }
}

impl MaterialModel for DuctileDamage {
    fn name(&self) -> &str {
        "DAMAGE INITIATION"
    }

    fn n_state(&self) -> usize {
        self.n_inner() + PER * self.crits.len() + 2
    }

    fn init_state(&self, state: &mut [f64]) {
        let ni = self.n_inner();
        self.inner.init_state(&mut state[..ni]);
        for v in &mut state[ni..] {
            *v = 0.0;
        }
        for (k, c) in self.crits.iter().enumerate() {
            state[ni + PER * k + 5] = 0.0f64.clamp(c.table[0].0, c.table[c.table.len() - 1].0);
        }
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn requires_unsymmetric(&self) -> bool {
        // the softening gives an indefinite, unsymmetric tangent
        self.crits.iter().any(|c| !matches!(c.evol, Evolution::None))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        let ni = self.n_inner();
        self.inner.rotate_state(r, &mut state[..ni]);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        // without the element's characteristic length (trusses etc.): L = 1
        self.update_with_length(input, 1.0, state_old, state_new)
    }

    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let (mut out, growing) = self.eval(input, length, state_old, state_new)?;
        if growing {
            // numerical tangent: the damage depends on the strain via ε̄p
            let n = input.strain.len();
            let scale = input.strain.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-6);
            let h = 1e-7 * scale;
            let mut tmp = state_new.to_vec();
            let mut c = vec![0.0; n * n];
            for j in 0..n {
                let mut e = input.strain.to_vec();
                e[j] += h;
                let mut de = input.dstrain.to_vec();
                de[j] += h;
                let inp = MatInput { strain: &e, dstrain: &de, ..*input };
                let (o2, _) = self.eval(&inp, length, state_old, &mut tmp)?;
                for i in 0..n {
                    c[i * n + j] = (o2.stress[i] - out.stress[i]) / h;
                }
            }
            out.tangent = c;
        }
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        if self.crits.iter().any(|c| c.crit == Criterion::Ductile) {
            v.push(("DUCTCRT", vec!["DUCTCRT"]));
        }
        if self.crits.iter().any(|c| matches!(c.crit, Criterion::Shear { .. })) {
            v.push(("SHRCRT", vec!["SHRCRT"]));
        }
        if self.crits.iter().any(|c| matches!(c.crit, Criterion::JohnsonCook(_))) {
            v.push(("JCCRT", vec!["JCCRT"]));
        }
        v.push(("SDEG", vec!["SDEG"]));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let ni = self.n_inner();
        let mut v = self.inner.output_field_values(mode, &state[..ni]);
        // the criterion is reported capped at 1 when it has a damage evolution
        let crit = |want: fn(&Criterion) -> bool| -> Option<f64> {
            self.crits
                .iter()
                .enumerate()
                .filter(|(_, c)| want(&c.crit))
                .map(|(k, c)| {
                    let w = state[self.base() + PER * k];
                    if matches!(c.evol, Evolution::None) {
                        w
                    } else {
                        w.min(1.0)
                    }
                })
                .reduce(f64::max)
        };
        if let Some(w) = crit(|c| *c == Criterion::Ductile) {
            v.push(w);
        }
        if let Some(w) = crit(|c| matches!(c, Criterion::Shear { .. })) {
            v.push(w);
        }
        if let Some(w) = crit(|c| matches!(c, Criterion::JohnsonCook(_))) {
            v.push(w);
        }
        v.push(state[self.d_index()]);
        v
    }
}
