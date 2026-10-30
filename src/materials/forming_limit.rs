//! Forming limit criteria for sheet metal (*DAMAGE INITIATION, CRITERION=FLD, FLSD
//! and MSFLD) on top of a plasticity model (*PLASTIC), in plane stress
//! (CPS elements and shells). The criteria are reported without
//! damage evolution (FLDCRT, FLSDCRT, MSFLDCRT); the value is the maximum so far.
//!
//! The criteria:
//! - FLD: ω = ε1/ε1^FLD(ε2) with the in-plane major and minor principal values of
//!   the plastic strain PE in CPS elements and of the total strain in
//!   shells (S4R); the table (ε1^FLD, ε2) is interpolated linearly.
//! - FLSD: ω = σ1/σ1^FLSD(σ2) with the in-plane principal stresses; table (σ1^FLSD, σ2).
//! - MSFLD (Müschenborn–Sonne): ω = ε̄pl/ε̄^MSFLD(α) with α = Δε2/Δε1 of
//!   the increment's plastic strain (in-plane principal values) and the table
//!   interpreted as (ε̄^MSFLD, α).
//!
//! Outside the tables the extrapolation is linear (also for MSFLD). An increment
//! without plastic strain does not change MSFLD. With a single table row
//! the limit is constant.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "forming limit criteria (FLD, FLSD, MSFLD)",
    consumes: &["ELASTIC", "PLASTIC", "CYCLIC HARDENING", "DAMAGE INITIATION", "POTENTIAL", "RATE DEPENDENT"],
    priority: 25,
    build: Some(build),
}];

const CRITERIA: &[&str] = &["FLD", "FLSD", "MSFLD"];

/// True if the material has a forming limit criterion.
pub fn claims(def: &MaterialDef) -> bool {
    def.behaviors.iter().any(|b| b.name == "DAMAGE INITIATION" && b.param_upper("CRITERION").is_some_and(|c| CRITERIA.contains(&c.as_str())))
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Fld,
    Flsd,
    Msfld,
}

struct Crit {
    kind: Kind,
    /// (x, limit) with increasing x (ε2, σ2 or α).
    table: Vec<(f64, f64)>,
}

pub struct FormingLimit {
    inner: Arc<dyn MaterialModel>,
    crits: Vec<Crit>,
}

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "rate- or temperature-dependent forming limits are not supported yet");
        }
        let (lim, x) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(lim > 0.0) {
            return kw.err_at(d.line, "the limit value must be positive");
        }
        if t.last().is_some_and(|&(xl, _)| !(x > xl)) {
            return kw.err_at(d.line, "the second column must increase in the table");
        }
        t.push((x, lim));
    }
    if t.is_empty() {
        return kw.err("data missing");
    }
    Ok(t)
}

/// Linear interpolation with linear extrapolation outside the table.
fn interp_lin(t: &[(f64, f64)], x: f64) -> f64 {
    if t.len() == 1 {
        return t[0].1;
    }
    let n = t.len();
    let k = if x <= t[0].0 {
        0
    } else if x >= t[n - 1].0 {
        n - 2
    } else {
        t.windows(2).position(|w| x < w[1].0).unwrap_or(n - 2)
    };
    let ((x0, y0), (x1, y1)) = (t[k], t[k + 1]);
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    let mut crits = Vec::new();
    for b in &def.behaviors {
        match b.name.as_str() {
            "DAMAGE INITIATION" => {
                let kind = match b.param_upper("CRITERION").as_deref() {
                    Some("FLD") => Kind::Fld,
                    Some("FLSD") => Kind::Flsd,
                    Some("MSFLD") => Kind::Msfld,
                    Some(o) => return b.err(format!("CRITERION={o} cannot be combined with FLD/FLSD/MSFLD yet")),
                    None => return b.err("CRITERION= missing"),
                };
                if b.has("DEPENDENCIES") || b.has("OMEGA") && kind != Kind::Msfld {
                    return b.err("field-dependent forming limits are not supported yet");
                }
                crits.push(Crit { kind, table: table(b)? });
            }
            "DAMAGE EVOLUTION" => return b.err("damage evolution after FLD/FLSD/MSFLD is not supported yet (only the initiation criterion)"),
            _ => {}
        }
    }
    if def.behavior("PLASTIC").is_none() {
        return fail(format!("material {}: CRITERION=FLD/FLSD/MSFLD requires *PLASTIC", def.name));
    }
    let base = MaterialDef {
        name: def.name.clone(),
        behaviors: def.behaviors.iter().filter(|b| !b.name.starts_with("DAMAGE ")).cloned().collect(),
        density: def.density,
    };
    let inner = super::build(&base)?;
    Ok(Some(Arc::new(FormingLimit { inner, crits })))
}

/// In-plane principal values (major, minor) of (a11, a22, a12 as a tensor).
fn principal2(a11: f64, a22: f64, a12: f64) -> (f64, f64) {
    let m = 0.5 * (a11 + a22);
    let r = (0.25 * (a11 - a22).powi(2) + a12 * a12).sqrt();
    (m + r, m - r)
}

impl FormingLimit {
    fn n_inner(&self) -> usize {
        self.inner.n_state()
    }

    /// (PEEQ, PE11, PE22, PE12 (engineering shear)) from the inner model's output.
    fn plastic(&self, state: &[f64]) -> [f64; 4] {
        let fields = self.inner.output_fields(StressMode::PlaneStress);
        let vals = self.inner.output_field_values(StressMode::PlaneStress, state);
        let (mut peeq, mut pe) = (0.0, [0.0; 3]);
        let mut k = 0;
        for (n, comps) in &fields {
            match *n {
                "PEEQ" => peeq = vals[k],
                "PE" => {
                    for (i, c) in comps.iter().enumerate() {
                        match *c {
                            "PE11" => pe[0] = vals[k + i],
                            "PE22" => pe[1] = vals[k + i],
                            "PE12" => pe[2] = vals[k + i],
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            k += comps.len();
        }
        [peeq, pe[0], pe[1], pe[2]]
    }

    /// Update; `total` = FLD with total strain (shells), otherwise plastic.
    fn update_impl(&self, input: &MatInput, total: bool, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        if input.mode != StressMode::PlaneStress {
            return fail("CRITERION=FLD/FLSD/MSFLD is only supported in plane stress (CPS elements and shells)");
        }
        let ni = self.n_inner();
        let out = self.inner.update(input, &state_old[..ni], &mut state_new[..ni])?;
        let p_old = self.plastic(&state_old[..ni]);
        let p_new = self.plastic(&state_new[..ni]);
        let s = &out.stress;
        for (k, c) in self.crits.iter().enumerate() {
            let w_old = state_old[ni + k];
            let w = match c.kind {
                Kind::Fld => {
                    let e = input.strain;
                    let (e1, e2) = if total { principal2(e[0], e[1], 0.5 * e[2]) } else { principal2(p_new[1], p_new[2], 0.5 * p_new[3]) };
                    e1 / interp_lin(&c.table, e2)
                }
                Kind::Flsd => {
                    let (s1, s2) = principal2(s[0], s[1], s[2]);
                    s1 / interp_lin(&c.table, s2)
                }
                Kind::Msfld => {
                    let d = [p_new[1] - p_old[1], p_new[2] - p_old[2], p_new[3] - p_old[3]];
                    let (d1, d2) = principal2(d[0], d[1], 0.5 * d[2]);
                    if p_new[0] > p_old[0] && d1.abs() > 1e-300 {
                        p_new[0] / interp_lin(&c.table, d2 / d1)
                    } else {
                        w_old
                    }
                }
            };
            state_new[ni + k] = w_old.max(w);
        }
        Ok(out)
    }
}

impl MaterialModel for FormingLimit {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn n_state(&self) -> usize {
        self.n_inner() + self.crits.len()
    }

    fn init_state(&self, state: &mut [f64]) {
        self.inner.init_state(&mut state[..self.n_inner()]);
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    /// Shells (the section integration calls `update`): FLD with the total
    /// in-plane strain (verified for S4R).
    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update_impl(input, true, state_old, state_new)
    }

    /// Continuum elements (CPS): FLD with the plastic strain.
    fn update_with_length(&self, input: &MatInput, _length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update_impl(input, false, state_old, state_new)
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        let ni = self.n_inner();
        self.inner.rotate_state(r, &mut state[..ni]);
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        for c in &self.crits {
            let n = match c.kind {
                Kind::Fld => "FLDCRT",
                Kind::Flsd => "FLSDCRT",
                Kind::Msfld => "MSFLDCRT",
            };
            v.push((n, vec![n]));
        }
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let ni = self.n_inner();
        let mut v = self.inner.output_field_values(mode, &state[..ni]);
        v.extend((0..self.crits.len()).map(|k| state[ni + k]));
        v
    }

    fn output_names(&self) -> &[&'static str] {
        self.inner.output_names()
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        self.inner.output_values(&state[..self.n_inner()])
    }
}
