//! Printout to the `.dat` file: `*NODE PRINT`, `*EL PRINT`
//! and `*ENERGY PRINT` (tables per increment).
//!
//! - `*NODE PRINT, NSET=, FREQUENCY=n, TOTALS=YES|NO, SUMMARY=YES|NO`:
//!   one row per node with the requested variables (U, RF, UR, RM, NT, CF, ...,
//!   whole fields or components such as U1, RF2), followed by MAX/MIN with
//!   the node (SUMMARY, default YES) and TOTAL (TOTALS, default NO).
//! - `*EL PRINT, ELSET=, POSITION=INTEGRATION POINTS|CENTROIDAL, FREQUENCY=n,
//!   TOTALS=, SUMMARY=`: one row per integration point (or the element
//!   centroid: the mean of the points, for quadrilaterals and
//!   fully integrated hexahedra) and element type; variables such as S, E, PE,
//!   PEEQ, SDV, components such as S11 and MISES. Several data lines give several tables.
//! - `*ENERGY PRINT, FREQUENCY=n`: energy totals of the model: elastic
//!   strain energy (ALLSE = ∫ ½ σ:(ε − εp − εcr − εth) dV with
//!   the integration point weights), plastic and creep dissipation (ALLPD,
//!   ALLCD: trapezoidal rule ∫σ:dεp over the increments), internal energy
//!   (ALLIE = ALLSE + ALLPD + ALLCD) and external work (= ALLIE in statics).
//!
//! The tables are written at increments evenly divisible by FREQUENCY and
//! always at the last increment of the step (FREQUENCY=0 turns them off). Number
//! format: four significant digits in the element tables, E14.7 for
//! the nodes.

use crate::fem::model::{FeModel, RawKeyword};
use crate::fem::output::{AnalysisOutput, Frame, ResultSet};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Does the model have any print request?
pub fn has_print(model: &FeModel) -> bool {
    model.steps.iter().any(|s| s.output.iter().any(|k| matches!(k.name.as_str(), "NODE PRINT" | "EL PRINT" | "ENERGY PRINT")))
}

/// E format as in Fortran (mantissa d.ddd…, two-digit exponent).
pub fn fmt_e(v: f64, dec: usize) -> String {
    let s = format!("{v:.dec$E}");
    let (m, e) = s.split_once('E').unwrap_or((&s, "0"));
    let ex: i32 = e.parse().unwrap_or(0);
    format!("{m}E{}{:02}", if ex < 0 { '-' } else { '+' }, ex.abs())
}

/// Four significant digits as in the element tables: fixed form for 0.1 ≤ |v| < 10⁴,
/// otherwise E form with four decimals.
pub fn fmt_g4(v: f64) -> String {
    if v == 0.0 {
        return "0.000".into();
    }
    let a = v.abs();
    if (0.1..9999.5).contains(&a) {
        let digits = (a.log10().floor() as i32 + 1).max(0);
        let dec = (4 - digits).max(0) as usize;
        let s = format!("{v:.dec$}");
        if dec == 0 {
            format!("{s}.")
        } else {
            s
        }
    } else {
        fmt_e(v, 4)
    }
}

/// Energy totals (Fortran G12.6): six significant digits, fixed form for
/// 0.1 ≤ |v| < 10⁶, otherwise E form.
fn fmt_energy(v: f64) -> String {
    if v == 0.0 {
        return " 0.00000    ".into();
    }
    let a = v.abs();
    if (0.1..999_999.5).contains(&a) {
        let digits = (a.log10().floor() as i32 + 1).max(0);
        let dec = (6 - digits).max(0) as usize;
        format!("{:>12}", format!("{v:.dec$}    "))
    } else {
        fmt_e(v, 6)
    }
}

fn flag(kw: &RawKeyword, key: &str, default: bool) -> bool {
    kw.flag(key, default).unwrap_or(default)
}

fn frequency(kw: &RawKeyword) -> usize {
    kw.param("FREQUENCY").and_then(crate::inp::parse_label).map_or(1, |f| f.max(0) as usize)
}

/// The variable names on the data lines, one list per line.
fn variable_lines(kw: &RawKeyword) -> Vec<Vec<String>> {
    kw.data.iter().map(|d| d.fields.iter().map(|f| f.trim().to_uppercase()).filter(|f| !f.is_empty()).collect::<Vec<_>>()).filter(|v| !v.is_empty()).collect()
}

/// (field name, component) for a requested variable, e.g. "U" → all, "U2" → ("U", "U2").
fn node_columns(frame: &Frame, vars: &[String]) -> Vec<(String, String)> {
    let mut cols = Vec::new();
    for v in vars {
        let v = if v == "NT" { "NT11".to_string() } else { v.clone() };
        if let Some(f) = frame.node.get(&v) {
            for l in &f.labels {
                cols.push((v.clone(), l.clone()));
            }
            continue;
        }
        // component of a field: strip the trailing digits
        let base: String = v.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
        if let Some(f) = frame.node.get(&base) {
            if f.labels.contains(&v) {
                cols.push((base, v));
            }
        }
    }
    cols
}

fn summary(out: &mut String, cols: usize, vals: &[Vec<f64>], keys: &[String], label: &str, totals: bool, summary: bool, width: usize) {
    if vals.is_empty() {
        return;
    }
    let mut line = |title: &str, which: &dyn Fn(usize) -> (f64, usize)| {
        let _ = write!(out, "\n {title:<14}");
        let mut at = String::new();
        for c in 0..cols {
            let (v, i) = which(c);
            let _ = write!(out, "{:>width$}", fmt_g4(v));
            let _ = write!(at, "{:>width$}", keys[i]);
        }
        if !label.is_empty() {
            let _ = write!(out, "\n {label:<14}{at}");
        }
        out.push('\n');
    };
    if summary {
        line("MAX", &|c| {
            let mut b = (f64::NEG_INFINITY, 0);
            for (i, r) in vals.iter().enumerate() {
                if r[c] > b.0 {
                    b = (r[c], i);
                }
            }
            b
        });
        line("MIN", &|c| {
            let mut b = (f64::INFINITY, 0);
            for (i, r) in vals.iter().enumerate() {
                if r[c] < b.0 {
                    b = (r[c], i);
                }
            }
            b
        });
    }
    if totals {
        let _ = write!(out, "\n {:<14}", "TOTAL");
        for c in 0..cols {
            let s: f64 = vals.iter().map(|r| r[c]).sum();
            let _ = write!(out, "{:>width$}", fmt_g4(s));
        }
        out.push('\n');
    }
}

fn node_print(out: &mut String, model: &FeModel, frame: &Frame, kw: &RawKeyword) {
    let set = kw.param("NSET").map(|s| s.trim().to_uppercase());
    let nodes: Vec<usize> = match &set {
        Some(s) => match model.nsets.get(s) {
            Some(v) => v.clone(),
            None => {
                let _ = writeln!(out, "\n ***WARNING: node set {s} does not exist (*NODE PRINT)");
                return;
            }
        },
        None => (0..model.nodes.len()).collect(),
    };
    for vars in variable_lines(kw) {
        let cols = node_columns(frame, &vars);
        if cols.is_empty() {
            let _ = writeln!(out, "\n ***WARNING: the variables {} are not in the results (*NODE PRINT)", vars.join(", "));
            continue;
        }
        let _ = write!(out, "\n\n\n   Table for {}\n\n", match &set {
            Some(s) => format!("the nodes in node set {s}"),
            None => "all nodes".to_string(),
        });
        let _ = write!(out, "       NODE        ");
        for (_, l) in &cols {
            let _ = write!(out, "{l:<15}");
        }
        let _ = write!(out, "\n\n");
        let mut vals = Vec::new();
        let mut keys = Vec::new();
        for &n in &nodes {
            let key = (model.instances[model.node_inst[n] as usize].clone(), model.node_label[n]);
            let row: Vec<f64> = cols.iter().map(|(f, l)| frame.node.get(f).and_then(|fd| fd.get(&key, l)).unwrap_or(0.0)).collect();
            let _ = write!(out, "{:>10}     ", model.node_label[n]);
            for v in &row {
                let _ = write!(out, "{:<15}", fmt_e(*v, 7));
            }
            out.push('\n');
            vals.push(row);
            keys.push(model.node_label[n].to_string());
        }
        summary(out, cols.len(), &vals, &keys, "AT NODE", flag(kw, "TOTALS", false), flag(kw, "SUMMARY", true), 12);
    }
}

/// Components (field, component) for element variables; "MISES" is computed from S.
fn elem_columns(frame: &Frame, vars: &[String]) -> Vec<(String, String)> {
    let mut cols = Vec::new();
    for v in vars {
        if v == "MISES" && frame.elem.contains_key("S") {
            cols.push(("S".into(), "MISES".into()));
            continue;
        }
        if let Some(f) = frame.elem.get(v) {
            for l in &f.labels {
                cols.push((v.clone(), l.clone()));
            }
            continue;
        }
        let base: String = v.trim_end_matches(|c: char| c.is_ascii_digit()).to_string();
        if let Some(f) = frame.elem.get(&base) {
            if f.labels.contains(v) {
                cols.push((base, v.clone()));
            }
        }
    }
    cols
}

fn mises(frame: &Frame, key: &(String, i64, usize, usize)) -> f64 {
    let s = |l: &str| frame.elem.get("S").and_then(|f| f.get(key, l)).unwrap_or(0.0);
    let (a, b, c, d, e, f) = (s("S11"), s("S22"), s("S33"), s("S12"), s("S13"), s("S23"));
    (0.5 * ((a - b).powi(2) + (b - c).powi(2) + (c - a).powi(2)) + 3.0 * (d * d + e * e + f * f)).sqrt()
}

fn el_print(out: &mut String, model: &FeModel, frame: &Frame, kw: &RawKeyword) {
    let set = kw.param("ELSET").map(|s| s.trim().to_uppercase());
    let elems: Vec<usize> = match &set {
        Some(s) => match model.elsets.get(s) {
            Some(v) => v.clone(),
            None => {
                let _ = writeln!(out, "\n ***WARNING: element set {s} does not exist (*EL PRINT)");
                return;
            }
        },
        None => (0..model.elements.len()).collect(),
    };
    let pos = kw.param_upper("POSITION").unwrap_or_else(|| "INTEGRATION POINTS".into());
    let centroid = pos == "CENTROIDAL";
    if !centroid && pos != "INTEGRATION POINTS" {
        let _ = writeln!(out, "\n ***WARNING: POSITION={pos} is not supported (*EL PRINT writes INTEGRATION POINTS or CENTROIDAL)");
        return;
    }
    // elements per type, in order
    let mut by_type: Vec<(&'static str, Vec<usize>)> = Vec::new();
    for &e in &elems {
        let t = model.elements[e].etype.name();
        match by_type.iter_mut().find(|(n, _)| *n == t) {
            Some((_, v)) => v.push(e),
            None => by_type.push((t, vec![e])),
        }
    }
    for vars in variable_lines(kw) {
        let cols = elem_columns(frame, &vars);
        if cols.is_empty() {
            let _ = writeln!(out, "\n ***WARNING: the variables {} are not in the results (*EL PRINT)", vars.join(", "));
            continue;
        }
        for (ty, list) in &by_type {
            let where_ = if centroid { "at the element centroids" } else { "at the integration points" };
            let set_txt = set.as_deref().map(|s| format!(", element set {s}")).unwrap_or_default();
            let _ = write!(out, "\n\n\n   Table {where_}, element type {ty}{set_txt}\n\n");
            let _ = write!(out, "{}", if centroid { "    ELEMENT          " } else { "    ELEMENT  IP             " });
            for (_, l) in &cols {
                let _ = write!(out, "{l:<12}");
            }
            let _ = write!(out, "{}", "\n\n");
            let mut vals = Vec::new();
            let mut keys = Vec::new();
            for &e in list {
                let el = &model.elements[e];
                let inst = model.instances[el.inst as usize].clone();
                // the integration points (section point 0 or the first) present in S or the first field
                let probe = frame.elem.get(&cols[0].0);
                let mut pts: Vec<(usize, usize)> = probe
                    .map(|f| f.items.keys().filter(|k| k.0 == inst && k.1 == el.label).map(|k| (k.2, k.3)).collect())
                    .unwrap_or_default();
                pts.sort();
                pts.dedup_by_key(|p| p.0);
                if pts.is_empty() {
                    continue;
                }
                let value = |key: &(String, i64, usize, usize), f: &str, l: &str| -> f64 {
                    if l == "MISES" {
                        mises(frame, key)
                    } else {
                        frame.elem.get(f).and_then(|fd| fd.get(key, l)).unwrap_or(0.0)
                    }
                };
                if centroid {
                    let row: Vec<f64> = cols
                        .iter()
                        .map(|(f, l)| pts.iter().map(|&(ip, sp)| value(&(inst.clone(), el.label, ip, sp), f, l)).sum::<f64>() / pts.len() as f64)
                        .collect();
                    let _ = write!(out, "{:>12}     ", el.label);
                    for v in &row {
                        let _ = write!(out, "{:>12}", fmt_g4(*v));
                    }
                    out.push('\n');
                    vals.push(row);
                    keys.push(el.label.to_string());
                } else {
                    for &(ip, sp) in &pts {
                        let key = (inst.clone(), el.label, ip, sp);
                        let row: Vec<f64> = cols.iter().map(|(f, l)| value(&key, f, l)).collect();
                        let _ = write!(out, "{:>12}{:>4}     ", el.label, ip);
                        for v in &row {
                            let _ = write!(out, "{:>12}", fmt_g4(*v));
                        }
                        out.push('\n');
                        vals.push(row);
                        keys.push(el.label.to_string());
                    }
                }
            }
            summary(out, cols.len(), &vals, &keys, "ELEMENT", flag(kw, "TOTALS", false), flag(kw, "SUMMARY", true), 12);
        }
    }
}

/// Energies in a frame and the plastic/creep strain at the integration points.
struct Energy {
    se: f64,
    /// (key → (σ, εp, εcr)) for the dissipation between frames.
    pts: BTreeMap<(String, i64, usize, usize), ([f64; 6], [f64; 6], [f64; 6])>,
}

const SL: [&str; 6] = ["11", "22", "33", "12", "13", "23"];

fn energy(model: &FeModel, frame: &Frame, weights: &BTreeMap<(String, i64), Vec<f64>>) -> Energy {
    let mut se = 0.0;
    let mut pts = BTreeMap::new();
    let Some(s) = frame.elem.get("S") else { return Energy { se, pts } };
    let get = |f: &str, key: &(String, i64, usize, usize), c: &str| frame.elem.get(f).and_then(|fd| fd.get(key, &format!("{f}{c}"))).unwrap_or(0.0);
    for key in s.items.keys() {
        let Some(w) = weights.get(&(key.0.clone(), key.1)).and_then(|w| w.get(key.2.wrapping_sub(1))) else { continue };
        let mut sig = [0.0; 6];
        let mut eel = [0.0; 6];
        let mut ep = [0.0; 6];
        let mut ec = [0.0; 6];
        for (i, c) in SL.iter().enumerate() {
            sig[i] = get("S", key, c);
            ep[i] = get("PE", key, c);
            ec[i] = get("CE", key, c);
            eel[i] = get("E", key, c) - ep[i] - ec[i] - get("THE", key, c);
        }
        // σ:ε with engineering shear strain
        let dot: f64 = (0..6).map(|i| sig[i] * eel[i]).sum();
        se += 0.5 * dot * w;
        pts.insert(key.clone(), (sig, ep, ec));
    }
    let _ = model;
    Energy { se, pts }
}

/// Integration point weights (volume) per element: the element reports
/// them to `user::context` when the internal force is computed (continuum elements).
fn ip_weights(model: &FeModel) -> BTreeMap<(String, i64), Vec<f64>> {
    let mut m = BTreeMap::new();
    let Ok(prep) = crate::fem::assembly::Prepared::new(model) else { return m };
    let Ok(states) = prep.init_states() else { return m };
    for (e, el) in model.elements.iter().enumerate() {
        let ctx = prep.ctx(e, false, 0.0, 0.0);
        if ctx.material.is_none() {
            continue;
        }
        let z = vec![0.0; el.etype.n_dofs()];
        let mut rec = crate::user::context::record_ips(|| {
            let mut sn = Vec::new();
            let _ = el.etype.internal_force(&ctx, &z, &z, &states[e], &mut sn, false);
        });
        if rec.is_empty() {
            continue;
        }
        rec.sort_by_key(|r| r.0);
        m.insert((model.instances[el.inst as usize].clone(), el.label), rec.iter().map(|r| r.2).collect());
    }
    m
}

fn dissipation(a: &Energy, b: &Energy, weights: &BTreeMap<(String, i64), Vec<f64>>) -> (f64, f64) {
    let (mut pd, mut cd) = (0.0, 0.0);
    for (k, (s1, p1, c1)) in &b.pts {
        let Some((s0, p0, c0)) = a.pts.get(k) else { continue };
        let Some(w) = weights.get(&(k.0.clone(), k.1)).and_then(|w| w.get(k.2.wrapping_sub(1))) else { continue };
        for i in 0..6 {
            let s = 0.5 * (s0[i] + s1[i]);
            pd += s * (p1[i] - p0[i]) * w;
            cd += s * (c1[i] - c0[i]) * w;
        }
    }
    (pd, cd)
}

/// The .dat text for the analysis (empty if the model has no print request).
pub fn write(model: &FeModel, out: &AnalysisOutput) -> String {
    if !has_print(model) {
        return String::new();
    }
    let rs: ResultSet = out.to_result_set(model);
    let mut t = String::new();
    let _ = writeln!(t, "\n   RustyFEM: printout (*NODE PRINT, *EL PRINT, *ENERGY PRINT)\n");
    let needs_energy = model.steps.iter().any(|s| s.output.iter().any(|k| k.name == "ENERGY PRINT"));
    let weights = if needs_energy { ip_weights(model) } else { BTreeMap::new() };
    let (mut pd, mut cd) = (0.0, 0.0);
    let mut prev: Option<Energy> = None;
    let mut total_time = 0.0;
    for (si, (step, so)) in model.steps.iter().zip(&rs.steps).enumerate() {
        let prints: Vec<&RawKeyword> = step.output.iter().filter(|k| matches!(k.name.as_str(), "NODE PRINT" | "EL PRINT" | "ENERGY PRINT")).collect();
        let period = crate::user::context::step_period(step);
        let _ = writeln!(t, "\n\n                              S T E P       {}     {}\n", si + 1, step.name.to_uppercase());
        let nf = so.frames.len();
        // frame 0 is the state at the start of the step when the step has several frames
        let initial = nf > 1 && so.frames[0].value == 0.0;
        for (fi, frame) in so.frames.iter().enumerate() {
            if fi == 0 && initial {
                if needs_energy && prev.is_none() {
                    prev = Some(energy(model, frame, &weights));
                }
                continue;
            }
            let inc = if initial { fi } else { fi + 1 };
            let last = fi + 1 == nf;
            let en = if needs_energy { Some(energy(model, frame, &weights)) } else { None };
            if let (Some(a), Some(b)) = (&prev, &en) {
                let (p, c) = dissipation(a, b, &weights);
                pd += p;
                cd += c;
            }
            // order: energy, elements, nodes
            let rank = |k: &RawKeyword| match k.name.as_str() {
                "ENERGY PRINT" => 0,
                "EL PRINT" => 1,
                _ => 2,
            };
            let mut due: Vec<&&RawKeyword> = prints.iter().filter(|k| {
                let f = frequency(k);
                f > 0 && (last || inc % f == 0)
            }).collect();
            due.sort_by_key(|k| rank(k));
            if !due.is_empty() {
                let st = frame.value;
                let _ = write!(
                    t,
                    "\n   Increment {inc}: step time {}, total time {}\n",
                    fmt_g4(st),
                    fmt_g4(total_time + st)
                );
                for k in due {
                    match k.name.as_str() {
                        "NODE PRINT" => {
                            let _ = write!(t, "\n\n   NODE RESULTS\n");
                            node_print(&mut t, model, frame, k);
                        }
                        "EL PRINT" => {
                            let _ = write!(t, "\n\n   ELEMENT RESULTS\n");
                            el_print(&mut t, model, frame, k);
                        }
                        _ => {
                            if let Some(e) = &en {
                                let ie = e.se + pd + cd;
                                let _ = write!(t, "\n\n   ENERGY TOTALS for the whole model at t = {}\n\n", fmt_e(total_time + st, 5));
                                for (name, v) in [
                                    ("Elastic strain energy", e.se),
                                    ("Kinetic energy", 0.0),
                                    ("External work", ie),
                                    ("Plastic dissipation", pd),
                                    ("Creep dissipation", cd),
                                    ("Internal energy (stress power)", ie),
                                    ("Energy balance", 0.0),
                                ] {
                                    let _ = writeln!(t, "   {name:<34}{}", fmt_energy(v));
                                }
                            }
                        }
                    }
                }
            }
            if en.is_some() {
                prev = en;
            }
        }
        total_time += period;
    }
    let _ = writeln!(t, "\n\n   Analysis completed\n");
    t
}
