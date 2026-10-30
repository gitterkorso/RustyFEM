//! Automatic time incrementation in implicit dynamics (*DYNAMIC).
//! The rules are checked against test cases (bar, beam, continuum,
//! contact; see tests/cases/dyn_incr/), driven by the half-step residual and
//! *CONTROLS, PARAMETERS=TIME INCREMENTATION:
//!
//! - **Application**: TRANSIENT FIDELITY gives INCREMENTATION=CONSERVATIVE
//!   (the half-step residual controls), MODERATE DISSIPATION and QUASI-STATIC give
//!   AGGRESSIVE (only convergence controls, as in statics; NOHAF). INCREMENTATION=
//!   and NOHAF on *DYNAMIC override.
//! - **Largest increment** (0 or empty on the data line): TF max(T/100, Δt₀),
//!   MD max(T/10, Δt₀), QS T. **Smallest**: min(Δt₀, 10⁻⁵ T).
//! - **Half-step tolerance** T_J: HAFTOL (force; moment HAFTOL · characteristic
//!   element length), otherwise HALFINC SCALE FACTOR (default 1000, 10000 with
//!   contact) times the time average of the force (moment) including the current
//!   attempt.
//! - **Cutback**: |R½| = S_J > T_J gives a new attempt with D_A (T_J/S_J) Δt
//!   (D_A = 0.85, no lower bound except the minimum increment, which is tried
//!   once); lack of convergence gives D_f Δt (0.25). At most I_A = 5 attempts.
//! - **Increase (CONSERVATIVE)**: sliding window over the I_T = 3 latest
//!   increments since the latest cutback. If Δt S_i/Δt_i < W_G T_J for all i
//!   (Δt = nominal increment) the proposal is min(D_G Δt_p, D_M Δt) with
//!   Δt_p = I_T T_J / Σ(S_i/Δt_i), D_G = 0.8, D_M = 1.25. In linear problems
//!   the proposal is used only if it is at least D_L D_M Δt (D_L = 0.95), otherwise
//!   Δt is kept (avoids a new factorization). The window is not reset after an
//!   increase, so Δt can increase in every increment.
//! - **Increase (AGGRESSIVE)**: D_D Δt (1.5) when two consecutive increments (since
//!   the latest cutback) have converged in at most I_G = 4 iterations. More than
//!   I_L = 10 iterations gives D_B Δt (0.75) in the next increment.
//! - **Time points**: the increment is shortened to hit the next output time point
//!   (*OUTPUT TIME INTERVAL/NUMBER INTERVAL/TIME POINTS, field and history,
//!   except TIME MARKS=NO) and the end of the step, without stretching (D_T = 1). After an
//!   increment shortened to a time point the incrementation continues with
//!   max(nominal Δt, Δt₀), i.e. the suggested initial increment if the
//!   nominal one is smaller (checked with Δt₀ = 1.5e-5 and 2e-5, different
//!   time points and amplitudes).

use crate::fem::dof::Field;
use crate::fem::model::{parse_f64, FeModel, RawKeyword, StepDef};
use crate::model::{FemError, Result};
use std::collections::BTreeMap;

/// Parameters of the time incrementation (*CONTROLS, PARAMETERS=TIME INCREMENTATION).
#[derive(Debug, Clone)]
pub struct IncParams {
    pub i_l: usize,
    pub i_g: usize,
    pub i_a: usize,
    pub i_t: usize,
    pub d_f: f64,
    pub d_b: f64,
    pub d_a: f64,
    pub d_d: f64,
    pub w_g: f64,
    pub d_g: f64,
    pub d_m: f64,
    pub d_m_dyn: f64,
    pub d_l: f64,
}

impl Default for IncParams {
    fn default() -> Self {
        IncParams { i_l: 10, i_g: 4, i_a: 5, i_t: 3, d_f: 0.25, d_b: 0.75, d_a: 0.85, d_d: 1.5, w_g: 0.75, d_g: 0.8, d_m: 1.5, d_m_dyn: 1.25, d_l: 0.95 }
    }
}

/// The numbers on a data line; empty fields become `None`.
fn values(kw: &RawKeyword, d: &crate::fem::model::DataLine) -> Result<Vec<Option<f64>>> {
    d.fields
        .iter()
        .map(|f| {
            let f = f.trim();
            if f.is_empty() {
                Ok(None)
            } else {
                parse_f64(f).map(Some).ok_or_else(|| FemError(format!("{} line {} (*CONTROLS): '{f}' is not a number", kw.file, d.line)))
            }
        })
        .collect()
}

impl IncParams {
    /// Reads *CONTROLS, PARAMETERS=TIME INCREMENTATION in the step (RESET gives default values).
    pub fn from_step(step: &StepDef) -> Result<IncParams> {
        let mut p = IncParams::default();
        for kw in step.extra.iter().filter(|k| k.name == "CONTROLS") {
            if kw.has("RESET") {
                p = IncParams::default();
            }
            if kw.param_upper("PARAMETERS").as_deref() != Some("TIME INCREMENTATION") {
                continue;
            }
            let recs = kw.records();
            let line = |i: usize| -> Result<Vec<Option<f64>>> { recs.get(i).map_or(Ok(Vec::new()), |d| values(kw, d)) };
            let (l1, l2, l3) = (line(0)?, line(1)?, line(2)?);
            let get = |v: &Vec<Option<f64>>, i: usize| v.get(i).copied().flatten().filter(|x| *x > 0.0);
            let setu = |x: &mut usize, v: Option<f64>| {
                if let Some(v) = v {
                    *x = v.round() as usize;
                }
            };
            let setf = |x: &mut f64, v: Option<f64>| {
                if let Some(v) = v {
                    *x = v;
                }
            };
            // first line: I0, IR, IP, IC, IL, IG, IS, IA, IJ, IT
            setu(&mut p.i_l, get(&l1, 4));
            setu(&mut p.i_g, get(&l1, 5));
            setu(&mut p.i_a, get(&l1, 7));
            setu(&mut p.i_t, get(&l1, 9));
            // second line: Df, DC, DB, DA, DS, DH, DD, WG
            setf(&mut p.d_f, get(&l2, 0));
            setf(&mut p.d_b, get(&l2, 2));
            setf(&mut p.d_a, get(&l2, 3));
            setf(&mut p.d_d, get(&l2, 6));
            setf(&mut p.w_g, get(&l2, 7));
            // third line: DG, DM, DM (dynamics), DM (diffusion), DL, ...
            setf(&mut p.d_g, get(&l3, 0));
            setf(&mut p.d_m, get(&l3, 1));
            setf(&mut p.d_m_dyn, get(&l3, 2));
            setf(&mut p.d_l, get(&l3, 4));
        }
        p.i_t = p.i_t.clamp(1, 10);
        Ok(p)
    }
}

/// Output request for field frames.
#[derive(Debug, Clone)]
pub enum FieldRequest {
    /// Every n-th increment.
    Every(usize),
    /// At these step times.
    Points(Vec<f64>),
}

/// Output time points and field request for the step.
#[derive(Debug, Clone, Default)]
pub struct OutputPlan {
    /// Time points (step time, ascending, ≤ the step time) that the increments must hit.
    /// The end of the step counts as a time point only if output is requested there (then
    /// the last increment can be stretched to it).
    pub points: Vec<f64>,
    /// Field request (empty: every increment).
    pub field: Vec<FieldRequest>,
}

/// *TIME POINTS, NAME=name (with or without GENERATE).
fn named_time_points(model: &FeModel, kw: &RawKeyword, name: &str) -> Result<Vec<f64>> {
    let Some(tp) = model.extra.iter().find(|k| k.name == "TIME POINTS" && k.param_upper("NAME").as_deref() == Some(name)) else {
        return kw.err(format!("*TIME POINTS, NAME={name} does not exist"));
    };
    let mut out = Vec::new();
    for d in tp.records() {
        let v = tp.floats(&d)?;
        if tp.has("GENERATE") {
            let (a, b, s) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0));
            if s <= 0.0 {
                return tp.err_at(d.line, "GENERATE requires a positive time increment");
            }
            let n = ((b - a) / s + 1e-9).floor() as usize;
            out.extend((0..=n).map(|i| a + i as f64 * s));
        } else {
            out.extend(v);
        }
    }
    Ok(out)
}

impl OutputPlan {
    /// Reads the step's *OUTPUT requests (FIELD and HISTORY).
    pub fn from_step(model: &FeModel, step: &StepDef, period: f64) -> Result<OutputPlan> {
        let mut plan = OutputPlan::default();
        let eps = 1e-9 * period;
        for kw in step.output.iter().filter(|k| k.name == "OUTPUT") {
            let field = kw.has("FIELD") || !kw.has("HISTORY");
            let num = |k: &str| kw.param(k).and_then(parse_f64);
            let marks = kw.param_upper("TIME MARKS").map_or(true, |v| v == "YES");
            let mut pts: Option<Vec<f64>> = None;
            if let Some(n) = num("FREQUENCY") {
                if field {
                    plan.field.push(FieldRequest::Every((n.round() as usize).max(1)));
                }
                continue;
            } else if let Some(n) = num("NUMBER INTERVAL") {
                let n = (n.round() as usize).max(1);
                pts = Some((1..=n).map(|i| period * i as f64 / n as f64).collect());
            } else if let Some(d) = num("TIME INTERVAL") {
                if d > 0.0 {
                    let n = (period / d + 1e-9).floor() as usize;
                    pts = Some((1..=n).map(|i| d * i as f64).collect());
                }
            } else if let Some(name) = kw.param_upper("TIME POINTS") {
                pts = Some(named_time_points(model, kw, &name)?);
            }
            match pts {
                Some(p) => {
                    if marks {
                        plan.points.extend(p.iter().copied().filter(|&t| t > eps && t < period + eps).map(|t| t.min(period)));
                    }
                    if field {
                        plan.field.push(FieldRequest::Points(p));
                    }
                }
                None if field => plan.field.push(FieldRequest::Every(1)),
                None => {}
            }
        }
        plan.points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        plan.points.dedup_by(|a, b| (*a - *b).abs() <= eps);
        Ok(plan)
    }

    /// Should a frame be written after increment `inc` ending at step time `t`?
    pub fn write_frame(&self, inc: usize, t: f64, eps: f64) -> bool {
        if self.field.is_empty() {
            return true;
        }
        self.field.iter().any(|r| match r {
            FieldRequest::Every(k) => inc % k == 0,
            FieldRequest::Points(p) => p.iter().any(|&x| (x - t).abs() <= eps),
        })
    }
}

/// Incrementation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Fixed increments (DIRECT).
    Fixed,
    /// The half-step residual controls (TRANSIENT FIDELITY).
    Conservative,
    /// Only convergence controls (MODERATE DISSIPATION, QUASI-STATIC).
    Aggressive,
}

/// Half-step tolerance: absolute (HAFTOL) or scale factor on the time average.
#[derive(Debug, Clone, Copy)]
pub enum HalfTol {
    Absolute(f64),
    Scale(f64),
}

/// An accepted increment in the increase window.
#[derive(Debug, Clone)]
struct Done {
    /// Half-step residual per field.
    s: BTreeMap<Field, f64>,
    dt: f64,
    iters: usize,
}

/// The state of the automatic incrementation.
#[derive(Debug, Clone)]
pub struct Incrementer {
    pub mode: Mode,
    pub p: IncParams,
    pub dt0: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    pub period: f64,
    /// Nominal increment (before shortening to time points).
    pub nominal: f64,
    pub tol: Option<HalfTol>,
    /// Characteristic element length (moment tolerance = the force tolerance · the length).
    pub length: f64,
    pub linear: bool,
    points: Vec<f64>,
    window: Vec<Done>,
    /// Attempts in the current increment.
    pub attempt: usize,
    /// Length of the current attempt and whether it was shortened to a time point.
    pub h: f64,
    pub to_point: bool,
}

/// Outcome of an attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    Accept,
    /// Redo the increment with this Δt.
    Retry(f64),
}

impl Incrementer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(mode: Mode, p: IncParams, dt0: f64, dt_min: f64, dt_max: f64, period: f64, tol: Option<HalfTol>, length: f64, linear: bool, points: Vec<f64>) -> Self {
        Incrementer { mode, p, dt0, dt_min, dt_max, period, nominal: dt0, tol, length, linear, points, window: Vec::new(), attempt: 0, h: dt0, to_point: false }
    }

    fn eps(&self) -> f64 {
        1e-9 * self.period
    }

    /// Length of the next attempt from step time `t`.
    pub fn next(&mut self, t: f64) -> f64 {
        let eps = self.eps();
        let mut h = if self.mode == Mode::Fixed { self.dt0 } else { self.nominal.min(self.dt_max) };
        self.to_point = false;
        if self.mode != Mode::Fixed {
            if let Some(&tp) = self.points.iter().find(|&&x| x > t + eps) {
                if t + h > tp - eps {
                    h = tp - t;
                    self.to_point = true;
                }
            }
        }
        if t + h > self.period - eps {
            h = self.period - t;
        }
        self.h = h;
        h
    }

    /// Tolerance per field at the time average `tavg` (including the current attempt).
    pub fn tolerance(&self, tavg: &BTreeMap<Field, f64>) -> BTreeMap<Field, f64> {
        let mut out = BTreeMap::new();
        match self.tol {
            Some(HalfTol::Absolute(f)) => {
                out.insert(Field::Displacement, f);
                out.insert(Field::Rotation, f * self.length);
                out.insert(Field::Warping, f * self.length * self.length);
            }
            Some(HalfTol::Scale(s)) => {
                for (f, q) in tavg {
                    out.insert(*f, s * q);
                }
            }
            None => {}
        }
        out
    }

    /// Check of a converged attempt: the half-step residual `s` against `tol`.
    pub fn check(&mut self, s: &BTreeMap<Field, f64>, tol: &BTreeMap<Field, f64>) -> Result<Verdict> {
        if self.mode != Mode::Conservative {
            return Ok(Verdict::Accept);
        }
        let mut f = f64::INFINITY;
        for (fld, &sv) in s {
            if let Some(&t) = tol.get(fld) {
                if sv > t {
                    f = f.min(self.p.d_a * t / sv);
                }
            }
        }
        if f.is_finite() {
            return self.cut(f).map(Verdict::Retry);
        }
        Ok(Verdict::Accept)
    }

    /// Lack of convergence: new attempt with D_f Δt.
    pub fn diverged(&mut self) -> Result<f64> {
        if self.mode == Mode::Fixed {
            return crate::model::fail("the increment did not converge (DIRECT, fixed increments)");
        }
        self.cut(self.p.d_f)
    }

    fn cut(&mut self, f: f64) -> Result<f64> {
        self.attempt += 1;
        self.window.clear();
        if self.attempt >= self.p.i_a {
            return crate::model::fail(format!("too many attempts ({}) in the same increment – the time increment {:.3e} is not sufficient", self.p.i_a, self.h));
        }
        let mut h = self.h * f;
        if h < self.dt_min {
            if self.h <= self.dt_min * (1.0 + 1e-9) {
                return crate::model::fail(format!("the time increment became smaller than the minimum ({:.3e})", self.dt_min));
            }
            h = self.dt_min;
        }
        self.nominal = h;
        self.h = h;
        self.to_point = false;
        Ok(h)
    }

    /// An increase from `base` to `prop` is used: in linear problems only if it
    /// is at least D_L D_M (avoids a new factorization for small changes).
    fn allowed_increase(&self, prop: f64, base: f64) -> bool {
        prop > base * (1.0 + 1e-9) && (!self.linear || prop >= self.p.d_l * self.p.d_m_dyn * base * (1.0 - 1e-6))
    }

    /// Accepted increment with the half-step residual `s`, the tolerance `tol` and
    /// `iters` equilibrium iterations: determines the next nominal increment.
    /// Returns `Some(Δt)` if the increment was increased.
    /// `t` is the step time at the end of the increment.
    pub fn accepted(&mut self, t: f64, s: BTreeMap<Field, f64>, tol: &BTreeMap<Field, f64>, iters: usize) -> Option<f64> {
        self.attempt = 0;
        if self.mode == Mode::Fixed {
            return None;
        }
        let h = self.h;
        // after an increment shortened to a time point: max(nominal, Δt₀)
        let base = if self.to_point && h < self.nominal { self.nominal.max(self.dt0) } else { self.nominal };
        self.window.push(Done { s, dt: h, iters });
        let it = self.p.i_t.max(2);
        if self.window.len() > it {
            self.window.remove(0);
        }
        let mut next = base;
        let mut grown = None;
        // the accuracy proposal D_G Δt_p (without the cap D_M), if the condition is satisfied
        let mut accuracy: Option<f64> = None;
        match self.mode {
            Mode::Conservative => {
                if self.window.len() >= self.p.i_t && self.window.iter().all(|d| d.iters <= self.p.i_l) {
                    let win = &self.window[self.window.len() - self.p.i_t..];
                    let mut ok = true;
                    let mut dtp = f64::INFINITY;
                    for (fld, &t) in tol {
                        let sum: f64 = win.iter().map(|d| d.s.get(fld).copied().unwrap_or(0.0) / d.dt).sum();
                        if win.iter().any(|d| base * d.s.get(fld).copied().unwrap_or(0.0) / d.dt >= self.p.w_g * t) {
                            ok = false;
                        }
                        if sum > 0.0 {
                            dtp = dtp.min(self.p.i_t as f64 * t / sum);
                        }
                    }
                    if ok {
                        accuracy = Some(self.p.d_g * dtp);
                        let prop = (self.p.d_g * dtp).min(self.p.d_m_dyn * base);
                        if self.allowed_increase(prop, base) {
                            next = prop;
                            grown = Some(prop.min(self.dt_max));
                        }
                    }
                }
            }
            Mode::Aggressive => {
                let n = self.window.len();
                if n >= 2 && self.window[n - 1].iters <= self.p.i_g && self.window[n - 2].iters <= self.p.i_g {
                    next = base * self.p.d_d.min(self.p.d_m);
                    grown = Some(next.min(self.dt_max));
                }
            }
            Mode::Fixed => {}
        }
        // Next output time point r beyond the increment (probed in
        // tests/cases/dyn_incr/): if r lies within D_M times the new increment
        // the next increment becomes min(D_G Δt_p, r) without the cap D_M (only r if
        // the accuracy condition is not satisfied), provided that it is an
        // allowed increase (at least D_L D_M Δt in linear problems); otherwise
        // Δt is kept. The increment is thus stretched to the time point instead of leaving
        // a small last increment, and an increase that would pass a
        // time point that the current Δt does not reach is not made.
        if self.mode == Mode::Conservative {
            let eps = self.eps();
            if let Some(&tp) = self.points.iter().find(|&&x| x > t + eps) {
                let r = tp - t;
                if r > base * (1.0 + 1e-6) && r <= self.p.d_m_dyn * next * (1.0 + 1e-6) {
                    let cand = accuracy.map_or(r, |a| a.min(r));
                    if self.allowed_increase(cand, base) {
                        next = cand;
                        grown = Some(cand.min(self.dt_max));
                    } else {
                        next = base;
                        grown = None;
                    }
                }
            }
        }
        if iters > self.p.i_l {
            next = self.p.d_b * h;
            grown = None;
        }
        self.nominal = next.min(self.dt_max);
        grown
    }
}
