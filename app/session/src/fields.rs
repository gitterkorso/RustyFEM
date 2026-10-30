//! Resultatfält för visning: katalog över steg, frames och fält, och värden
//! per nod (elementfält utjämnade till noderna) som `f32`.

use crate::mesh::DisplayMesh;
use rustyfem::fem::output::{AnalysisOutput, FrameData};
use serde::Serialize;
use specta::Type;

#[derive(Debug, Clone, Serialize, Type, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Position {
    Node,
    Element,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct FieldInfo {
    pub name: String,
    pub position: Position,
    pub components: Vec<String>,
    /// Enheten i modellens enhetssystem ("" om okänd, dimensionslös eller inte satt).
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct FrameInfo {
    pub index: u32,
    pub value: f64,
    pub mode: i32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct HistoryInfo {
    pub index: u32,
    pub region: String,
    pub name: String,
    pub points: u32,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct StepResultInfo {
    pub index: u32,
    pub name: String,
    pub procedure: String,
    pub increments: u32,
    pub iterations: u32,
    pub frames: Vec<FrameInfo>,
    pub fields: Vec<FieldInfo>,
    pub history: Vec<HistoryInfo>,
}

const STRESS6: [&str; 6] = ["S11", "S22", "S33", "S12", "S13", "S23"];

fn magnitude(c: &[String]) -> Vec<String> {
    let mut v = c.to_vec();
    if c.len() > 1 {
        v.push("Magnitude".into());
    }
    v
}

fn slot_names(names: &[&str]) -> Vec<String> {
    magnitude(&names.iter().map(|s| s.to_string()).collect::<Vec<_>>())
}

/// Fält som finns i en frame.
pub fn frame_fields(fr: &FrameData) -> Vec<FieldInfo> {
    let active = fr.active.iter().fold(0u16, |a, &b| a | b);
    let mut out = Vec::new();
    let node = |name: &str, comps: Vec<String>| FieldInfo { name: name.into(), position: Position::Node, components: comps, unit: String::new() };
    let dim3 = active & 0b100 != 0;
    if active & 0b111 != 0 {
        out.push(node("U", slot_names(if dim3 { &["U1", "U2", "U3"] } else { &["U1", "U2"] })));
    }
    if active & 0b111000 != 0 {
        out.push(node("UR", slot_names(&["UR1", "UR2", "UR3"])));
    }
    if active & 0b1000_0000 != 0 {
        out.push(node("NT11", vec!["NT11".into()]));
    }
    if active & 0b111 != 0 && !fr.rf.is_empty() {
        out.push(node("RF", slot_names(if dim3 { &["RF1", "RF2", "RF3"] } else { &["RF1", "RF2"] })));
    }
    for (name, comps, _) in &fr.node_extra {
        out.push(node(name, magnitude(comps)));
    }
    let mut elem: Vec<FieldInfo> = Vec::new();
    for fields in &fr.elem {
        for f in fields {
            match elem.iter_mut().find(|x| x.name == f.name) {
                Some(x) => {
                    for l in &f.labels {
                        if !x.components.contains(l) {
                            x.components.push(l.clone());
                        }
                    }
                }
                None => elem.push(FieldInfo { name: f.name.clone(), position: Position::Element, components: f.labels.clone(), unit: String::new() }),
            }
        }
    }
    for f in &mut elem {
        if f.name == "S" && f.components.iter().any(|c| STRESS6.contains(&c.as_str())) {
            f.components.insert(0, "Mises".into());
        }
    }
    out.extend(elem);
    out
}

/// Katalog för alla steg.
pub fn catalog(out: &AnalysisOutput) -> Vec<StepResultInfo> {
    out.steps
        .iter()
        .enumerate()
        .map(|(i, s)| StepResultInfo {
            index: i as u32,
            name: s.name.clone(),
            procedure: s.procedure.clone(),
            increments: s.increments as u32,
            iterations: s.iterations as u32,
            frames: s
                .frames
                .iter()
                .enumerate()
                .map(|(k, f)| FrameInfo { index: k as u32, value: f.value, mode: f.mode as i32, description: f.description.clone() })
                .collect(),
            fields: s.frames.last().map(frame_fields).unwrap_or_default(),
            history: s
                .history
                .iter()
                .enumerate()
                .map(|(k, h)| HistoryInfo { index: k as u32, region: h.0.clone(), name: h.1.clone(), points: h.2.len() as u32 })
                .collect(),
        })
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Ett elementfälts komponent i en punkt (Mises beräknas ur spänningarna).
fn point_value(labels: &[String], vals: &[f64], comp: &str) -> Option<f64> {
    if comp == "Mises" {
        let mut s = [0.0; 6];
        for (k, l) in STRESS6.iter().enumerate() {
            if let Some(i) = labels.iter().position(|x| x == l) {
                s[k] = vals[i];
            }
        }
        return Some(rustyfem::results::mises(&s));
    }
    labels.iter().position(|x| x == comp).map(|i| vals[i])
}

/// Elementets medelvärde av en komponent över alla integrations- och sektionspunkter.
pub fn element_mean(fr: &FrameData, e: usize, field: &str, comp: &str) -> Option<f64> {
    let f = fr.elem.get(e)?.iter().find(|f| f.name == field)?;
    let (mut s, mut n) = (0.0, 0usize);
    for (_, _, vals) in &f.points {
        if let Some(v) = point_value(&f.labels, vals, comp) {
            s += v;
            n += 1;
        }
    }
    (n > 0).then(|| s / n as f64)
}

/// Värden per nod (NaN där fältet saknas), samt min och max.
pub fn node_values(fr: &FrameData, mesh: &DisplayMesh, n_nodes: usize, field: &FieldInfo, comp: &str) -> (Vec<f32>, f64, f64) {
    let mut out = vec![f32::NAN; n_nodes];
    let ci = field.components.iter().position(|c| c == comp);
    let is_mag = comp == "Magnitude";
    let pick = |v: &[f64]| -> f64 {
        if is_mag { norm(v) } else { ci.and_then(|i| v.get(i).copied()).unwrap_or(f64::NAN) }
    };
    match field.position {
        Position::Node => {
            let slots: Option<(usize, usize)> = match field.name.as_str() {
                "U" => Some((0, 3)),
                "UR" => Some((3, 6)),
                "NT11" => Some((7, 8)),
                _ => None,
            };
            if let Some((a, b)) = slots {
                for (n, u) in fr.u.iter().enumerate().take(n_nodes) {
                    if fr.active.get(n).is_some_and(|m| (m >> a) & ((1 << (b - a)) - 1) != 0) {
                        out[n] = pick(&u[a..b]) as f32;
                    }
                }
            } else if field.name == "RF" {
                for (n, r) in fr.rf.iter().enumerate().take(n_nodes) {
                    if fr.constrained.get(n).is_some_and(|m| m & 0b111 != 0) {
                        out[n] = pick(&r[0..3]) as f32;
                    }
                }
            } else if let Some((_, _, vals)) = fr.node_extra.iter().find(|x| x.0 == field.name) {
                for (n, v) in vals.iter().enumerate().take(n_nodes) {
                    if !v.is_empty() {
                        out[n] = pick(v) as f32;
                    }
                }
            }
        }
        Position::Element => {
            let mut sum = vec![0.0f64; n_nodes];
            let mut cnt = vec![0u32; n_nodes];
            for (e, nodes) in mesh.elem_nodes.iter().enumerate() {
                let v = if is_mag {
                    fr.elem.get(e).and_then(|fs| fs.iter().find(|f| f.name == field.name)).map(|f| {
                        let n = f.points.len().max(1) as f64;
                        f.points.iter().map(|p| norm(&p.2)).sum::<f64>() / n
                    })
                } else {
                    element_mean(fr, e, &field.name, comp)
                };
                if let Some(v) = v {
                    for &n in nodes {
                        sum[n as usize] += v;
                        cnt[n as usize] += 1;
                    }
                }
            }
            for n in 0..n_nodes {
                if cnt[n] > 0 {
                    out[n] = (sum[n] / cnt[n] as f64) as f32;
                }
            }
        }
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for (n, &v) in out.iter().enumerate() {
        if v.is_finite() && mesh.used.get(n).copied().unwrap_or(true) {
            lo = lo.min(v as f64);
            hi = hi.max(v as f64);
        }
    }
    if !lo.is_finite() {
        lo = 0.0;
        hi = 0.0;
    }
    (out, lo, hi)
}

/// Förskjutningar U1–U3 per nod som `f32`.
pub fn displacement(fr: &FrameData, n_nodes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_nodes * 3];
    for (n, u) in fr.u.iter().enumerate().take(n_nodes) {
        for k in 0..3 {
            out[3 * n + k] = u[k] as f32;
        }
    }
    out
}
