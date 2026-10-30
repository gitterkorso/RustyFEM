//! Amplitudes (*AMPLITUDE) and evaluation of loads/boundary conditions with AMPLITUDE=.
//!
//! *AMPLITUDE is stored by the .inp reader in `FeModel::extra`. Here the
//! supported definitions are interpreted:
//!
//! - `DEFINITION=TABULAR` (default): pairs (t, A), linear interpolation, constant
//!   before the first and after the last point. `SMOOTH` is not supported (without
//!   the parameter no smoothing is done).
//! - `DEFINITION=EQUALLY SPACED`: `FIXED INTERVAL=Δt`, `BEGIN=t0`, values.
//! - `DEFINITION=PERIODIC`: line 1 `N, ω, t0, A0`, then pairs (Aₙ, Bₙ):
//!   A = A0 + Σ [Aₙ cos nω(t−t0) + Bₙ sin nω(t−t0)] for t ≥ t0, otherwise A0.
//! - `DEFINITION=SMOOTH STEP`: pairs (t, A); between two points
//!   A = Aᵢ + (Aᵢ₊₁ − Aᵢ) ξ³(10 − 15ξ + 6ξ²), ξ = (t − tᵢ)/(tᵢ₊₁ − tᵢ).
//! - `DEFINITION=DECAY`: `A0, A, t0, td`: A = A0 + A exp(−(t−t0)/td) for t ≥ t0.
//!
//! `TIME=STEP TIME` (default) or `TIME=TOTAL TIME`. `VALUE=RELATIVE`
//! (default) multiplies the load magnitude; `VALUE=ABSOLUTE` replaces it.
//! `SCALEX`, `SCALEY`, `SHIFTX`, `SHIFTY` are supported for tabular definitions.
//! All definitions are verified in tests/cases/dynamics/.

use crate::fem::model::{DistLoad, FeModel, RawKeyword};
use crate::fem::state::StepLoads;
use crate::model::{fail, Result};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
enum Def {
    Tabular(Vec<(f64, f64)>),
    Smooth(Vec<(f64, f64)>),
    Periodic { omega: f64, t0: f64, a0: f64, terms: Vec<(f64, f64)> },
    Decay { a0: f64, a: f64, t0: f64, td: f64 },
    /// DEFINITION=USER: UAMP (`user::routines`).
    User(crate::user::routines::UampRef),
}

/// An amplitude: A(t) with step time or total time.
#[derive(Debug, Clone)]
pub struct Amplitude {
    pub name: String,
    def: Def,
    pub total_time: bool,
    pub absolute: bool,
}

fn interp(p: &[(f64, f64)], t: f64, smooth: bool) -> f64 {
    if p.is_empty() {
        return 0.0;
    }
    if t <= p[0].0 {
        return p[0].1;
    }
    for w in p.windows(2) {
        let ((t0, a0), (t1, a1)) = (w[0], w[1]);
        if t <= t1 {
            if t1 <= t0 {
                return a1;
            }
            let xi = (t - t0) / (t1 - t0);
            let f = if smooth { xi * xi * xi * (10.0 - 15.0 * xi + 6.0 * xi * xi) } else { xi };
            return a0 + (a1 - a0) * f;
        }
    }
    p[p.len() - 1].1
}

impl Amplitude {
    /// A at step time `step_time` and total time `total_time`.
    pub fn value(&self, step_time: f64, total_time: f64) -> f64 {
        let t = if self.total_time { total_time } else { step_time };
        match &self.def {
            Def::User(u) => u.eval(step_time, total_time).0,
            Def::Tabular(p) => interp(p, t, false),
            Def::Smooth(p) => interp(p, t, true),
            Def::Periodic { omega, t0, a0, terms } => {
                if t < *t0 {
                    return *a0;
                }
                let x = t - t0;
                a0 + terms.iter().enumerate().map(|(k, (a, b))| {
                    let w = (k + 1) as f64 * omega * x;
                    a * w.cos() + b * w.sin()
                }).sum::<f64>()
            }
            Def::Decay { a0, a, t0, td } => {
                if t < *t0 {
                    *a0
                } else {
                    a0 + a * (-(t - t0) / td).exp()
                }
            }
        }
    }

    /// First and second time derivatives (A′, A″) at the same time as `value`.
    /// Used for velocity and acceleration in prescribed degrees of freedom
    /// (the analytical derivatives of the amplitude are reported there, verified for
    /// SMOOTH STEP). TABULAR gives the slope of the segment and A″ = 0.
    pub fn rates(&self, step_time: f64, total_time: f64) -> (f64, f64) {
        let t = if self.total_time { total_time } else { step_time };
        match &self.def {
            Def::User(u) => {
                let (_, d1, d2) = u.eval(step_time, total_time);
                (d1, d2)
            }
            Def::Tabular(p) | Def::Smooth(p) => {
                let smooth = matches!(self.def, Def::Smooth(_));
                // area explicit_adv: right derivative at the first point (explicit analysis: V at t = 0)
                if p.len() < 2 || t < p[0].0 || t > p[p.len() - 1].0 {
                    return (0.0, 0.0);
                }
                for w in p.windows(2) {
                    let ((t0, a0), (t1, a1)) = (w[0], w[1]);
                    if t <= t1 && t1 > t0 {
                        let d = t1 - t0;
                        let xi = (t - t0) / d;
                        return if smooth {
                            let f1 = 30.0 * xi * xi * (1.0 - xi) * (1.0 - xi);
                            let f2 = 60.0 * xi * (1.0 - 3.0 * xi + 2.0 * xi * xi);
                            ((a1 - a0) * f1 / d, (a1 - a0) * f2 / (d * d))
                        } else {
                            ((a1 - a0) / d, 0.0)
                        };
                    }
                }
                (0.0, 0.0)
            }
            Def::Periodic { omega, t0, terms, .. } => {
                if t < *t0 {
                    return (0.0, 0.0);
                }
                let x = t - t0;
                let mut d1 = 0.0;
                let mut d2 = 0.0;
                for (k, (a, b)) in terms.iter().enumerate() {
                    let w = (k + 1) as f64 * omega;
                    let (sn, cs) = (w * x).sin_cos();
                    d1 += w * (-a * sn + b * cs);
                    d2 += -w * w * (a * cs + b * sn);
                }
                (d1, d2)
            }
            Def::Decay { a, t0, td, .. } => {
                if t < *t0 {
                    (0.0, 0.0)
                } else {
                    let e = a * (-(t - t0) / td).exp();
                    (-e / td, e / (td * td))
                }
            }
        }
    }

    /// Parses an *AMPLITUDE block.
    pub fn parse(kw: &RawKeyword) -> Result<Amplitude> {
        let name = match kw.param("NAME") {
            Some(n) if !n.trim().is_empty() => n.trim().to_uppercase(),
            _ => return kw.err("the parameter NAME= is missing"),
        };
        let def_name = kw.param_upper("DEFINITION").unwrap_or_else(|| "TABULAR".into());
        let total_time = match kw.param_upper("TIME") {
            None => false,
            Some(t) if t == "STEP TIME" => false,
            Some(t) if t == "TOTAL TIME" => true,
            Some(t) => return kw.err(format!("TIME={t} must be STEP TIME or TOTAL TIME")),
        };
        let absolute = match kw.param_upper("VALUE") {
            None => false,
            Some(v) if v == "RELATIVE" => false,
            Some(v) if v == "ABSOLUTE" => true,
            Some(v) => return kw.err(format!("VALUE={v} must be RELATIVE or ABSOLUTE")),
        };
        if let Some(s) = kw.param("SMOOTH") {
            if crate::fem::model::parse_f64(s).unwrap_or(1.0) != 0.0 {
                return kw.err("SMOOTH= is not supported yet");
            }
        }
        if kw.has("INPUT") {
            return kw.err("INPUT= is not supported yet");
        }
        let pf = |k: &str, d: f64| -> Result<f64> {
            match kw.param(k) {
                Some(v) => crate::fem::model::parse_f64(v).ok_or_else(|| crate::model::FemError(format!("{}: {k}={v} is not a number", kw.location()))),
                None => Ok(d),
            }
        };
        let (sx, sy, hx, hy) = (pf("SCALEX", 1.0)?, pf("SCALEY", 1.0)?, pf("SHIFTX", 0.0)?, pf("SHIFTY", 0.0)?);
        let mut vals = Vec::new();
        for d in &kw.data {
            vals.extend(kw.floats(d)?.into_iter().zip(d.fields.iter()).filter(|(_, f)| !f.trim().is_empty()).map(|(v, _)| v));
        }
        let pairs = |vals: &[f64]| -> Result<Vec<(f64, f64)>> {
            if vals.len() % 2 != 0 || vals.is_empty() {
                return kw.err("the table must consist of pairs (time, amplitude)");
            }
            let p: Vec<(f64, f64)> = vals.chunks(2).map(|c| (c[0] * sx + hx, c[1] * sy + hy)).collect();
            if p.windows(2).any(|w| w[1].0 < w[0].0) {
                return kw.err("the times must be increasing");
            }
            Ok(p)
        };
        let def = match def_name.as_str() {
            "TABULAR" => Def::Tabular(pairs(&vals)?),
            "SMOOTH STEP" => Def::Smooth(pairs(&vals)?),
            "EQUALLY SPACED" => {
                let dt = pf("FIXED INTERVAL", 0.0)?;
                if dt <= 0.0 {
                    return kw.err("FIXED INTERVAL= is missing");
                }
                let t0 = pf("BEGIN", 0.0)?;
                Def::Tabular(vals.iter().enumerate().map(|(i, &a)| ((t0 + i as f64 * dt) * sx + hx, a * sy + hy)).collect())
            }
            "PERIODIC" => {
                // first data line: N, ω, t0, A0; then Aₙ, Bₙ
                let first = kw.data.first().map(|d| kw.floats(d)).transpose()?.unwrap_or_default();
                if first.len() < 4 {
                    return kw.err("the first data line must be 'N, ω, t0, A0'");
                }
                let n = first[0].round() as usize;
                let mut rest = Vec::new();
                for d in &kw.data[1..] {
                    rest.extend(kw.floats(d)?.into_iter().zip(d.fields.iter()).filter(|(_, f)| !f.trim().is_empty()).map(|(v, _)| v));
                }
                if rest.len() < 2 * n {
                    return kw.err(format!("PERIODIC with N={n} requires {} coefficients", 2 * n));
                }
                Def::Periodic { omega: first[1], t0: first[2], a0: first[3], terms: rest[..2 * n].chunks(2).map(|c| (c[0], c[1])).collect() }
            }
            "DECAY" => {
                if vals.len() < 4 {
                    return kw.err("DECAY requires 'A0, A, t0, td'");
                }
                Def::Decay { a0: vals[0], a: vals[1], t0: vals[2], td: vals[3] }
            }
            "USER" => Def::User(crate::user::routines::UampRef::new(kw, &name)?),
            other => return kw.err(format!("DEFINITION={other} is not supported yet")),
        };
        Ok(Amplitude { name, def, total_time, absolute })
    }
}

/// All amplitudes in the model.
pub fn amplitudes(model: &FeModel) -> Result<BTreeMap<String, Amplitude>> {
    let mut m = BTreeMap::new();
    for kw in model.extra.iter().filter(|k| k.name == "AMPLITUDE") {
        let a = Amplitude::parse(kw)?;
        m.insert(a.name.clone(), a);
    }
    Ok(m)
}

/// Evaluates a step's loads and boundary conditions with amplitudes.
///
/// Entries without AMPLITUDE= follow the step's default amplitude: linear ramp from
/// the previous value (`ramp`) or immediately (step). Entries with AMPLITUDE= get
/// the value magnitude · A(t) (VALUE=RELATIVE).
pub struct LoadEval<'a> {
    pub loads: &'a StepLoads,
    pub amps: BTreeMap<String, Amplitude>,
    /// Default amplitude: linear ramp over the step (otherwise step).
    pub ramp: bool,
    /// Length of the step and total time at the start of the step.
    pub period: f64,
    pub time0: f64,
    /// The step default amplitude acts fully already at t = 0 (explicit analysis).
    pub step_full_at_start: bool,
}

impl<'a> LoadEval<'a> {
    pub fn new(model: &FeModel, loads: &'a StepLoads, ramp: bool, period: f64, time0: f64) -> Result<Self> {
        let amps = amplitudes(model)?;
        for name in loads.amp_bc.values().chain(loads.amp_cload.values()).chain(loads.amp_dload.iter().flatten()) {
            if !amps.contains_key(name) {
                return fail(format!("the amplitude {name} is not defined (*AMPLITUDE, NAME={name})"));
            }
        }
        Ok(LoadEval { loads, amps, ramp, period, time0, step_full_at_start: false })
    }

    /// The default amplitude factor at step time fraction `f`.
    pub fn default_factor(&self, f: f64) -> f64 {
        if self.ramp {
            f.clamp(0.0, 1.0)
        } else if f > 0.0 || self.period == 0.0 || self.step_full_at_start {
            1.0
        } else {
            0.0
        }
    }

    fn eval(&self, name: Option<&String>, (a, b): (f64, f64), step_time: f64) -> f64 {
        match name {
            Some(n) => {
                let amp = &self.amps[n];
                let v = amp.value(step_time, self.time0 + step_time);
                if amp.absolute {
                    v
                } else {
                    b * v
                }
            }
            None => {
                let f = if self.period > 0.0 { step_time / self.period } else { 1.0 };
                a + (b - a) * self.default_factor(f)
            }
        }
    }

    /// Prescribed values at the step time `step_time`.
    pub fn bc_at(&self, step_time: f64) -> Vec<(usize, usize, f64)> {
        self.loads.bc.iter().map(|(&k, &v)| (k.0, k.1, self.eval(self.loads.amp_bc.get(&k), v, step_time))).collect()
    }

    /// Velocity and acceleration (v, a) in prescribed degrees of freedom with AMPLITUDE=.
    pub fn bc_rates_at(&self, step_time: f64) -> Vec<(usize, usize, f64, f64)> {
        self.loads
            .amp_bc
            .iter()
            .filter_map(|(k, name)| {
                let amp = &self.amps[name];
                let (d1, d2) = amp.rates(step_time, self.time0 + step_time);
                let scale = if amp.absolute { 1.0 } else { self.loads.bc.get(k)?.1 };
                Some((k.0, k.1, scale * d1, scale * d2))
            })
            .collect()
    }

    /// Point loads at the step time (including decaying reactions from released boundary conditions).
    pub fn cloads_at(&self, step_time: f64) -> Vec<(usize, usize, f64)> {
        let mut v: Vec<(usize, usize, f64)> =
            self.loads.cload.iter().map(|(&k, &c)| (k.0, k.1, self.eval(self.loads.amp_cload.get(&k), c, step_time))).collect();
        let f = if self.period > 0.0 { step_time / self.period } else { 1.0 };
        for &(n, s, r) in &self.loads.released {
            v.push((n, s, -r * (1.0 - self.default_factor(f))));
        }
        v
    }

    pub fn dloads_at(&self, step_time: f64) -> Vec<DistLoad> {
        self.loads
            .dload
            .iter()
            .enumerate()
            .map(|(i, (l, start))| {
                let mut l = l.clone();
                l.magnitude = self.eval(self.loads.amp_dload.get(i).and_then(|a| a.as_ref()), (*start, l.magnitude), step_time);
                l
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::model::DataLine;

    fn kw(params: &[(&str, &str)], rows: &[&str]) -> RawKeyword {
        RawKeyword {
            name: "AMPLITUDE".into(),
            params: params.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
            data: rows.iter().enumerate().map(|(i, r)| DataLine { fields: r.split(',').map(|s| s.trim().to_string()).collect(), line: i + 1, continued: false }).collect(),
            file: "test".into(),
            line: 0,
        }
    }

    /// Reference values for the amplitude definitions (see tests/cases/dynamics).
    #[test]
    fn definitions_match_reference_values() {
        let tab = Amplitude::parse(&kw(&[("NAME", "T")], &["0.0, 0.0, 0.3, 1.0, 0.6, 0.5, 1.0, 2.0"])).unwrap();
        assert!((tab.value(0.65, 0.0) - 0.6875).abs() < 1e-12);
        let per = Amplitude::parse(&kw(&[("NAME", "P"), ("DEFINITION", "PERIODIC")], &["1, 6.0, 0.2, 0.5", "0.3, 0.7"])).unwrap();
        assert!((per.value(0.05, 0.0) - 0.5).abs() < 1e-12);
        assert!((per.value(0.25, 0.0) - 0.993465).abs() < 1e-6);
        assert!((per.value(0.9, 0.0) + 0.257181).abs() < 1e-6);
        let ss = Amplitude::parse(&kw(&[("NAME", "S"), ("DEFINITION", "SMOOTH STEP")], &["0.1, 0.0, 0.8, 1.5"])).unwrap();
        assert!((ss.value(0.2, 0.0) - 0.034896).abs() < 1e-6);
        assert!((ss.value(0.45, 0.0) - 0.75).abs() < 1e-12);
        let eq = Amplitude::parse(&kw(&[("NAME", "E"), ("DEFINITION", "EQUALLY SPACED"), ("FIXED INTERVAL", "0.25"), ("BEGIN", "0.1")], &["0.0, 1.0, 3.0, 2.0, 1.5"])).unwrap();
        assert!((eq.value(0.15, 0.0) - 0.2).abs() < 1e-12);
        assert!((eq.value(0.9, 0.0) - 1.9).abs() < 1e-12);
    }
}
