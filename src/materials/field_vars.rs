//! Predefined field variables:
//!
//! - `*INITIAL CONDITIONS, TYPE=FIELD, VARIABLE=n`: nodal values at the start
//!   (nodes that are not given have the value 0),
//! - `*FIELD, VARIABLE=n` in a step: nodal values at the end of the step
//!   (OP=MOD default, OP=NEW resets the variable to the initial values).
//!   The values are ramped linearly over the step from the previous value (or set
//!   directly with AMPLITUDE=STEP on *STEP) and are inherited by following steps,
//! - `*FIELD, USER, VARIABLE=n`: the values at the nodes are given by UFIELD
//!   (`user::routines::Ufield`) at every point in time.
//!
//! The node references are resolved when the file is read (`inp/keywords/user2.rs`):
//! the data lines become [node index, value] with the parameter `_NODINDEX` and
//! the variable number in `VARIABLE`. The materials read the fields at the point via
//! `user::context` (the element's nodal values, interpolated at the
//! integration point) and interpolate material data with DEPENDENCIES=
//! (`field_dependent.rs`).

use crate::fem::model::{FeModel, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

/// Data lines with resolved nodes (same marker as the temperature field).
pub const RESOLVED: &str = crate::materials::thermal::RESOLVED;

pub static FACTORIES: &[super::MaterialFactory] = &[];

#[derive(Clone)]
struct UserPart {
    /// Variable (0-based), nodes and the name the routine is selected by.
    var: usize,
    nodes: Vec<usize>,
    routine: Arc<dyn crate::user::routines::Ufield>,
}

#[derive(Clone)]
struct Segment {
    start: f64,
    period: f64,
    f0: Vec<f64>,
    f1: Vec<f64>,
    ramp: bool,
    kstep: i32,
    user: Vec<UserPart>,
}

/// The field variables at the nodes as a function of total time (nn × nfield, node by node).
#[derive(Clone)]
pub struct FieldHistory {
    nf: usize,
    initial: Vec<f64>,
    segments: Vec<Segment>,
    labels: Vec<i64>,
    coords: Vec<[f64; 3]>,
}

fn variable(kw: &RawKeyword) -> Result<usize> {
    match kw.param("VARIABLE") {
        None => Ok(1),
        Some(v) => match crate::inp::parse_label(v) {
            Some(n) if n >= 1 => Ok(n as usize),
            _ => kw.err(format!("VARIABLE={v} must be an integer ≥ 1")),
        },
    }
}

fn apply(kw: &RawKeyword, nf: usize, v: &mut [f64]) -> Result<()> {
    let k = variable(kw)? - 1;
    for d in &kw.data {
        let x = kw.floats(d)?;
        let n = x[0] as usize;
        if (n + 1) * nf <= v.len() {
            v[n * nf + k] = x.get(1).copied().unwrap_or(0.0);
        }
    }
    Ok(())
}

fn timeless(p: &str) -> bool {
    matches!(p, "FREQUENCY" | "BUCKLE" | "COMPLEX FREQUENCY" | "MODAL DYNAMIC" | "STEADY STATE DYNAMICS" | "RESPONSE SPECTRUM")
}

impl FieldHistory {
    /// The field history, or `None` if the model has no field variables.
    pub fn build(model: &FeModel) -> Result<Option<FieldHistory>> {
        let is_ic = |k: &&RawKeyword| k.name == "INITIAL CONDITIONS" && k.has(RESOLVED) && k.param_upper("TYPE").as_deref() == Some("FIELD");
        let ic: Vec<&RawKeyword> = model.extra.iter().filter(is_ic).collect();
        let fields: Vec<&RawKeyword> = model.steps.iter().flat_map(|s| s.extra.iter().filter(|k| k.name == "FIELD")).collect();
        if ic.is_empty() && fields.is_empty() {
            return Ok(None);
        }
        let mut nf = 0;
        for k in ic.iter().chain(fields.iter()) {
            nf = nf.max(variable(k)?);
        }
        let nn = model.nodes.len();
        let mut initial = vec![0.0; nn * nf];
        for k in &ic {
            apply(k, nf, &mut initial)?;
        }
        let mut cur = initial.clone();
        let mut time = 0.0;
        let mut segments = Vec::new();
        for (i, s) in model.steps.iter().enumerate() {
            if s.perturbation || timeless(s.procedure_name()) {
                continue;
            }
            let period = crate::user::context::step_period(s);
            let mut next = cur.clone();
            let mut user = Vec::new();
            for k in s.extra.iter().filter(|k| k.name == "FIELD") {
                let var = variable(k)? - 1;
                if k.param_upper("OP").as_deref() == Some("NEW") {
                    for n in 0..nn {
                        next[n * nf + var] = initial[n * nf + var];
                    }
                }
                if k.has("USER") {
                    let name = k.param("_NAME").unwrap_or("").to_string();
                    let Some(routine) = crate::user::routines::find_ufield(&name)? else {
                        return crate::user::routines::missing("UFIELD", "node set", &name);
                    };
                    let nodes = k.data.iter().filter_map(|d| k.floats(d).ok().and_then(|x| x.first().map(|&n| n as usize))).collect();
                    user.push(UserPart { var, nodes, routine });
                    continue;
                }
                apply(k, nf, &mut next)?;
            }
            let ramp = !s.amplitude_step && !super::time_dependence::default_step_amplitude(s);
            segments.push(Segment { start: time, period, f0: cur.clone(), f1: next.clone(), ramp, kstep: i as i32 + 1, user });
            time += period;
            cur = next;
        }
        Ok(Some(FieldHistory { nf, initial, segments, labels: model.node_label.clone(), coords: model.nodes.clone() }))
    }

    /// Number of field variables (largest VARIABLE=).
    pub fn nfield(&self) -> usize {
        self.nf
    }

    /// The field variables at the nodes `nodes` at total time `time` (node by node, `nfield` per node).
    pub fn at(&self, nodes: &[usize], time: f64) -> Vec<f64> {
        let nf = self.nf;
        let mut out = Vec::with_capacity(nodes.len() * nf);
        let seg = self.segments.iter().find(|s| time <= s.start + s.period * (1.0 + 1e-12));
        let seg = match seg {
            _ if time <= 0.0 || self.segments.is_empty() => {
                for &n in nodes {
                    out.extend_from_slice(&self.initial[n * nf..(n + 1) * nf]);
                }
                // UFIELD also applies at the start of the step
                if let Some(s) = self.segments.first() {
                    self.apply_user(s, nodes, time.max(0.0), &mut out);
                }
                return out;
            }
            Some(s) => s,
            None => {
                let s = self.segments.last().unwrap();
                for &n in nodes {
                    out.extend_from_slice(&s.f1[n * nf..(n + 1) * nf]);
                }
                self.apply_user(s, nodes, time, &mut out);
                return out;
            }
        };
        let f = ((time - seg.start) / seg.period).clamp(0.0, 1.0);
        let a = if seg.ramp { f } else if f > 0.0 { 1.0 } else { 0.0 };
        for &n in nodes {
            for k in 0..nf {
                let (x0, x1) = (seg.f0[n * nf + k], seg.f1[n * nf + k]);
                out.push(x0 + a * (x1 - x0));
            }
        }
        self.apply_user(seg, nodes, time, &mut out);
        out
    }

    /// UFIELD for the nodes in the step's *FIELD, USER.
    fn apply_user(&self, seg: &Segment, nodes: &[usize], time: f64, out: &mut [f64]) {
        if seg.user.is_empty() {
            return;
        }
        let nf = self.nf;
        let kinc = crate::user::context::step_inc().1;
        for (i, &n) in nodes.iter().enumerate() {
            for u in seg.user.iter().filter(|u| u.nodes.contains(&n)) {
                let field = &mut out[i * nf..(i + 1) * nf];
                let mut a = crate::user::routines::UfieldArgs {
                    field,
                    kfield: u.var + 1,
                    kstep: seg.kstep,
                    kinc,
                    time: [time - seg.start, time],
                    node: self.labels.get(n).copied().unwrap_or(0),
                    coords: self.coords.get(n).copied().unwrap_or([0.0; 3]),
                    temp: 0.0,
                    dtemp: 0.0,
                    nfield: nf,
                };
                // errors in the routine cannot be returned from here: the value is kept
                if let Err(e) = u.routine.ufield(&mut a) {
                    eprintln!("warning: UFIELD at node {}: {e}", a.node);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Interpolation in temperature and field variables (DEPENDENCIES=)
// ---------------------------------------------------------------------------

/// Table with dependencies: each entry has property values followed by the
/// independent variables (inner first: e.g. [T, f1, ..., fn], for *PLASTIC
/// [ε̄p, T, f1, ..., fn]). The entries are ordered: the inner variable
/// varies fastest, the outer (fn) slowest.
#[derive(Debug, Clone)]
pub struct DepTable {
    pub nprops: usize,
    pub nvars: usize,
    pub rows: Vec<Vec<f64>>,
}

impl DepTable {
    /// Reads the entries from the data lines: `nprops` properties, temperature and
    /// `ndep` field variables per entry; at most 8 values per line (input rule,
    /// longer entries continue on the next line). `inner` extra independent
    /// variables lie between the properties and the temperature (*PLASTIC: ε̄p).
    pub fn parse(kw: &RawKeyword, nprops: usize, inner: usize, ndep: usize) -> Result<DepTable> {
        let per = nprops + inner + 1 + ndep;
        let lines_per = per.div_ceil(8);
        let mut rows = Vec::new();
        let data = &kw.data;
        let mut i = 0;
        while i < data.len() {
            let mut v = Vec::new();
            for k in 0..lines_per {
                let Some(d) = data.get(i + k) else {
                    return kw.err_at(data[i].line, format!("the entry must have {per} values (DEPENDENCIES={ndep})"));
                };
                let mut x = kw.floats(d)?;
                x.truncate(8);
                if k + 1 < lines_per {
                    x.resize(8, 0.0);
                }
                v.extend(x);
            }
            v.resize(per, 0.0);
            rows.push(v);
            i += lines_per;
        }
        if rows.is_empty() {
            return kw.err("data line is missing");
        }
        Ok(DepTable { nprops: nprops + inner, nvars: 1 + ndep, rows })
    }

    /// The values (the first `nprops` columns) at the independent variables `x`
    /// (same order as in the entry). Hierarchical linear interpolation: first in
    /// the outermost variable between the groups with the same value, then inwards;
    /// constant outside the table.
    pub fn eval(&self, x: &[f64]) -> Vec<f64> {
        let idx: Vec<usize> = (0..self.rows.len()).collect();
        eval_rows(&self.rows, &idx, self.nprops, self.nvars, x)
    }
}

/// Interpolates the entries `idx` in variable `level` (0 = inner) and inwards.
fn eval_rows(rows: &[Vec<f64>], idx: &[usize], np: usize, level_count: usize, x: &[f64]) -> Vec<f64> {
    if level_count == 0 || idx.len() == 1 {
        return rows[idx[0]][..np].to_vec();
    }
    let lv = level_count - 1;
    let col = np + lv;
    // groups with the same value in the variable (in order)
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for &r in idx {
        let v = rows[r][col];
        match groups.last_mut() {
            Some((g, members)) if *g == v => members.push(r),
            _ => groups.push((v, vec![r])),
        }
    }
    let xv = x.get(lv).copied().unwrap_or(0.0);
    if groups.len() == 1 || xv <= groups[0].0 {
        return eval_rows(rows, &groups[0].1, np, lv, x);
    }
    let last = groups.len() - 1;
    if xv >= groups[last].0 {
        return eval_rows(rows, &groups[last].1, np, lv, x);
    }
    let k = groups.windows(2).position(|w| xv >= w[0].0 && xv <= w[1].0).unwrap_or(0);
    let (a, b) = (&groups[k], &groups[k + 1]);
    let s = if b.0 > a.0 { (xv - a.0) / (b.0 - a.0) } else { 0.0 };
    let va = eval_rows(rows, &a.1, np, lv, x);
    let vb = eval_rows(rows, &b.1, np, lv, x);
    va.iter().zip(&vb).map(|(p, q)| p + s * (q - p)).collect()
}

/// Error message when the field history is used without fields.
pub fn require_fields() -> Result<()> {
    fail("the material has DEPENDENCIES= but the model has no field variables (*FIELD, *INITIAL CONDITIONS, TYPE=FIELD)")
}
