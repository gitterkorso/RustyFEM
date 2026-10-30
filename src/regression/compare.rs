//! Comparison of RustyFEM results against a baseline.
//!
//! The error for a field in a frame is max |RustyFEM − baseline| over all nodes/
//! integration points and components, divided by the largest |reference value| in
//! the field (same normalization as `rustyfem validate`).
//!
//! - Nodal fields (U, UR, RF, RM, NT11, ...) are driven by the baseline's nodes: a node or
//!   component missing in RustyFEM counts as 0 (inactive degree of freedom).
//! - Element fields (S, E, PEEQ, ...) are driven by RustyFEM's points: every point must
//!   exist in the baseline. Reference points without a counterpart (e.g. section points in
//!   beams) are counted and reported but do not cause a failure.
//! - FRAMEVALUE compares the frame value (step time, natural frequency, eigenvalue) relatively.
//! - "H:NAME" compares the last value of history output NAME.
//! - "HA:NAME" compares all values of history output NAME (paired by time or
//!   mode number, normalized by the largest |reference value|), e.g. EIGVAL, PF1, EM2.
//! - "MODEVALUE" compares the eigenvalue in the frame description ("EigenValue = ..." in
//!   *BUCKLE, "Value = ..." in *FREQUENCY) relatively. The baseline has only 5
//!   significant digits there, so the tolerance should be ≥ 5e-5. (Buckling steps have no
//!   other eigenvalue in the reference results.)
//! - `floor` in the side file: nodal fields are normalized by at least floor · largest |reference value|
//!   in the field over all frames of the step (for numerical noise in components that are
//!   zero in a mode, e.g. rotations in a pure axial mode).
//! - `floor_elem`: the same for element fields but with the largest |reference value| in the field
//!   over all steps and frames of the case, e.g. for damage variables and plastic strains that
//!   are ≈ 0 (numerical noise) in frames before yielding or damage initiation.

use super::case::CaseSpec;
use crate::fem::output::{Frame, ResultSet};
use std::fmt::Write as _;

#[derive(Debug, Clone)]
pub struct FieldReport {
    pub step: usize,
    pub frame: usize,
    pub field: String,
    pub err: f64,
    pub max_ref: f64,
    pub n: usize,
    /// Reference points without a counterpart in RustyFEM (element fields).
    pub unmatched: usize,
    pub tol: f64,
    /// Where the largest error is.
    pub worst_at: String,
}

#[derive(Debug, Clone, Default)]
pub struct CaseReport {
    pub fields: Vec<FieldReport>,
    /// Errors that make the case fail (missing fields, different number of steps, ...).
    pub problems: Vec<String>,
    pub notes: Vec<String>,
}

impl CaseReport {
    pub fn passed(&self) -> bool {
        self.problems.is_empty() && self.fields.iter().all(|f| f.err <= f.tol)
    }

    pub fn worst(&self) -> f64 {
        self.fields.iter().map(|f| f.err).fold(0.0, f64::max)
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        for f in &self.fields {
            let mark = if f.err <= f.tol { "OK  " } else { "FAIL" };
            let _ = write!(
                s,
                "  {mark} step {} frame {:>3} {:<10} max rel. error {:.2e} (tol {:.0e}, |max| {:.4e}, {} values",
                f.step, f.frame, f.field, f.err, f.tol, f.max_ref, f.n
            );
            if f.unmatched > 0 {
                let _ = write!(s, ", {} reference points not compared", f.unmatched);
            }
            s.push(')');
            if f.err > f.tol && !f.worst_at.is_empty() {
                let _ = write!(s, " – largest error at {}", f.worst_at);
            }
            s.push('\n');
        }
        for p in &self.problems {
            let _ = writeln!(s, "  FAIL {p}");
        }
        for n in &self.notes {
            let _ = writeln!(s, "  note: {n}");
        }
        s
    }
}

fn compare_node(ours: &Frame, base: &Frame, field: &str, sign: f64, norm_min: f64) -> Option<(f64, f64, usize, String)> {
    let a = base.node.get(field)?;
    let o = ours.node.get(field);
    let mut maxa = 0.0f64;
    let mut maxe = 0.0f64;
    let mut n = 0;
    let mut at = String::new();
    for (key, comps) in &a.items {
        for &(id, va) in comps {
            let label = &a.labels[id as usize];
            let vo = o.and_then(|f| f.get(key, label)).unwrap_or(0.0) * sign;
            maxa = maxa.max(va.abs());
            let e = (vo - va).abs();
            if e > maxe {
                maxe = e;
                at = format!("{}.{} {label} (RustyFEM {vo:.6e}, reference {va:.6e})", key.0, key.1);
            }
            n += 1;
        }
    }
    let norm = maxa.max(norm_min);
    Some((if norm > 0.0 { maxe / norm } else { maxe }, maxa, n, at))
}

/// Largest |value| in the nodal field `field` in a frame.
fn node_max(f: &Frame, field: &str) -> f64 {
    f.node.get(field).map_or(0.0, |a| a.items.values().flat_map(|c| c.iter().map(|p| p.1.abs())).fold(0.0, f64::max))
}

fn elem_max(f: &Frame, field: &str) -> f64 {
    f.elem.get(field).map_or(0.0, |a| a.items.values().flat_map(|c| c.iter().map(|p| p.1.abs())).fold(0.0, f64::max))
}

/// The eigenvalue in a frame description ("EigenValue = x" or "Value = x").
pub fn description_value(desc: &str) -> Option<f64> {
    let i = desc.find("Value =")?;
    desc[i + 7..].split_whitespace().next()?.parse().ok()
}

/// (error, max, count, unmatched reference points, missing in the baseline, location)
fn compare_elem(ours: &Frame, base: &Frame, field: &str, sign: f64, norm_min: f64) -> Option<(f64, f64, usize, usize, usize, String)> {
    let a = base.elem.get(field)?;
    let o = ours.elem.get(field)?;
    let mut maxa = 0.0f64;
    let mut maxe = 0.0f64;
    let mut n = 0;
    let mut missing = 0;
    let mut at = String::new();
    for (key, comps) in &o.items {
        if !a.items.contains_key(key) {
            missing += 1;
            continue;
        }
        for &(id, vo) in comps {
            let label = &o.labels[id as usize];
            let Some(va) = a.get(key, label) else { continue };
            let vo = vo * sign;
            maxa = maxa.max(va.abs());
            let e = (vo - va).abs();
            if e > maxe {
                maxe = e;
                at = format!("{}.{} ip {} {label} (RustyFEM {vo:.6e}, reference {va:.6e})", key.0, key.1, key.2);
            }
            n += 1;
        }
    }
    let unmatched = a.items.keys().filter(|k| !o.items.contains_key(k)).count();
    let norm = maxa.max(norm_min);
    Some((if norm > 0.0 { maxe / norm } else { maxe }, maxa, n, unmatched, missing, at))
}

/// Compares RustyFEM (`ours`) against the baseline (`base`) according to the side file.
pub fn compare(ours: &ResultSet, base: &ResultSet, spec: &CaseSpec) -> CaseReport {
    let mut rep = CaseReport::default();
    if ours.steps.len() != base.steps.len() {
        rep.problems.push(format!("RustyFEM has {} steps, reference {}", ours.steps.len(), base.steps.len()));
    }
    for (so, sa) in ours.steps.iter().zip(&base.steps) {
        if !spec.steps.is_empty() && !spec.steps.contains(&sa.index) {
            continue;
        }
        let pairs: Vec<(&Frame, &Frame)> = if spec.all_frames || sa.procedure.contains("FREQUENCY") || sa.procedure.contains("BUCKLE") {
            // compare frames with the same frame number (modes in eigenvalue steps)
            if so.frames.len() != sa.frames.len() {
                rep.problems.push(format!("step {}: RustyFEM has {} frames, reference {}", sa.index, so.frames.len(), sa.frames.len()));
            }
            let mut v = Vec::new();
            for fo in &so.frames {
                match sa.frames.iter().find(|f| f.index == fo.index) {
                    Some(fa) => v.push((fo, fa)),
                    None => rep.problems.push(format!("step {}: the baseline is missing frame {}", sa.index, fo.index)),
                }
            }
            v
        } else {
            match (so.frames.last(), sa.frames.last()) {
                (Some(a), Some(b)) => vec![(a, b)],
                _ => {
                    rep.problems.push(format!("step {}: frames are missing", sa.index));
                    continue;
                }
            }
        };
        for (fo, fa) in pairs {
            for field in &spec.fields {
                let tol = spec.tol_for(field);
                let mut push = |err: f64, max_ref: f64, n: usize, unmatched: usize, worst_at: String| {
                    rep.fields.push(FieldReport { step: sa.index, frame: fa.index, field: field.clone(), err, max_ref, n, unmatched, tol, worst_at })
                };
                if field == "FRAMEVALUE" {
                    let e = (fo.value - fa.value).abs() / fa.value.abs().max(1e-300);
                    push(if fa.value == 0.0 { (fo.value - fa.value).abs() } else { e }, fa.value.abs(), 1, 0, format!("RustyFEM {:.6e}, reference {:.6e}", fo.value, fa.value));
                    continue;
                }
                if field == "MODEVALUE" {
                    // frames without an eigenvalue (base state, static steps) are skipped
                    if description_value(&fa.description).is_none() {
                        continue;
                    }
                    match (description_value(&fo.description), description_value(&fa.description)) {
                        (Some(a), Some(b)) => {
                            let e = if b == 0.0 { (a - b).abs() } else { (a - b).abs() / b.abs() };
                            push(e, b.abs(), 1, 0, format!("RustyFEM {a:.6e}, reference {b:.6e}"));
                        }
                        _ => rep.problems.push(format!("step {} frame {}: eigenvalue missing in the description", sa.index, fa.index)),
                    }
                    continue;
                }
                if let Some(h) = field.strip_prefix("HA:") {
                    // all values in history output, paired by time (mode number in eigenvalue steps),
                    // normalized by the largest |reference value|; compared once per step
                    if fa.index != sa.frames.last().map_or(fa.index, |f| f.index) {
                        continue;
                    }
                    let find = |rs: &crate::fem::output::StepResult| rs.history.iter().find(|x| x.name.eq_ignore_ascii_case(h)).map(|x| x.data.clone());
                    match (find(so), find(sa)) {
                        (Some(a), Some(b)) => {
                            // the sign may flip per point if the field is listed in sign_free (e.g. PF follows the mode's sign)
                            let free = spec.sign_free.contains(field);
                            let maxb = b.iter().fold(0.0f64, |m, p| m.max(p.1.abs()));
                            let mut maxe = 0.0f64;
                            let mut at = String::new();
                            for &(t, vb) in &b {
                                let va = a.iter().find(|p| (p.0 - t).abs() <= 1e-9 * t.abs().max(1.0)).map(|p| p.1);
                                let Some(va) = va else {
                                    rep.problems.push(format!("step {}: RustyFEM is missing {h} at time/mode {t}", sa.index));
                                    continue;
                                };
                                let e = if free { (va.abs() - vb.abs()).abs() } else { (va - vb).abs() };
                                if e > maxe {
                                    maxe = e;
                                    at = format!("time/mode {t}: RustyFEM {va:.6e}, reference {vb:.6e}");
                                }
                            }
                            push(if maxb > 0.0 { maxe / maxb } else { maxe }, maxb, b.len(), 0, at);
                        }
                        // missing in both (e.g. EIGFREQ in a static step): nothing to compare
                        (None, None) => {}
                        (None, _) => rep.problems.push(format!("step {}: RustyFEM is missing history output {h}", sa.index)),
                        (_, None) => rep.problems.push(format!("step {}: the baseline is missing history output {h}", sa.index)),
                    }
                    continue;
                }
                if let Some(h) = field.strip_prefix("H:") {
                    let last = |rs: &crate::fem::output::StepResult| rs.history.iter().find(|x| x.name.eq_ignore_ascii_case(h)).and_then(|x| x.data.last().map(|p| p.1));
                    match (last(so), last(sa)) {
                        (Some(a), Some(b)) => push((a - b).abs() / b.abs().max(1e-300), b.abs(), 1, 0, format!("RustyFEM {a:.6e}, reference {b:.6e}")),
                        (None, _) => rep.problems.push(format!("step {}: RustyFEM is missing history output {h}", sa.index)),
                        (_, None) => rep.problems.push(format!("step {}: the baseline is missing history output {h}", sa.index)),
                    }
                    continue;
                }
                let free = spec.sign_free.contains(field);
                let best = |f: &dyn Fn(f64) -> Option<(f64, f64, usize, usize, usize, String)>| -> Option<(f64, f64, usize, usize, usize, String)> {
                    let p = f(1.0)?;
                    if free {
                        let m = f(-1.0)?;
                        if m.0 < p.0 {
                            return Some(m);
                        }
                    }
                    Some(p)
                };
                if fa.node.contains_key(field) {
                    let norm_min = if spec.floor > 0.0 { spec.floor * sa.frames.iter().map(|f| node_max(f, field)).fold(0.0, f64::max) } else { 0.0 };
                    // Explicit absolute tolerance handles a physically zero field
                    // without weakening the relative check on nonzero results.
                    let norm_min = norm_min.max(spec.absolute_norm_floor(field));
                    if let Some((e, m, n, _, _, at)) = best(&|s| compare_node(fo, fa, field, s, norm_min).map(|(e, m, n, at)| (e, m, n, 0, 0, at))) {
                        push(e, m, n, 0, at);
                    }
                } else if fa.elem.contains_key(field) {
                    if !fo.elem.contains_key(field) {
                        rep.problems.push(format!("step {} frame {}: RustyFEM is missing the field {field}", sa.index, fa.index));
                        continue;
                    }
                    let norm_min = if spec.floor_elem > 0.0 { spec.floor_elem * base.steps.iter().flat_map(|st| st.frames.iter()).map(|f| elem_max(f, field)).fold(0.0, f64::max) } else { 0.0 };
                    let norm_min = norm_min.max(spec.absolute_norm_floor(field));
                    if let Some((e, m, n, un, miss, at)) = best(&|s| compare_elem(fo, fa, field, s, norm_min)) {
                        if miss > 0 {
                            rep.problems.push(format!("step {} frame {}: {miss} points in {field} are missing in the baseline", sa.index, fa.index));
                        }
                        push(e, m, n, un, at);
                    }
                } else if fo.node.contains_key(field) || fo.elem.contains_key(field) {
                    rep.problems.push(format!("step {} frame {}: the baseline is missing the field {field} (is it requested in *OUTPUT?)", sa.index, fa.index));
                } else {
                    let note = format!("the field {field} exists neither in RustyFEM nor in the baseline");
                    if !rep.notes.contains(&note) {
                        rep.notes.push(note);
                    }
                }
            }
        }
    }
    rep
}
