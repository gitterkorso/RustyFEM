//! Resultat per step och frame.
//!
//! `FrameData` är det steg-typerna skriver (vektorer per nod/element). För
//! jämförelse med referensresultat och för fil-utdata görs den om till `Frame` med
//! fält nycklade på (instans, nod) eller (instans, element, ip, sektionspunkt).
//!
//! Textformatet "rustyfem-res 1" skrivs av RustyFEM (`rustyfem run fil.inp
//! --out res.txt`) och används för referensresultaten i regressionstesterna:
//!
//! ```text
//! STEP;stegnr;namn;procedur
//! FRAME;stegnr;framenr;frameValue;mod;beskrivning
//! N;FÄLT;instans;nod;KOMP=värde;...
//! E;FÄLT;instans;element;ip;sektionspunkt;KOMP=värde;...
//! H;stegnr;region;utdata;tid:värde;...
//! ```

use crate::elements::IpField;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::FeModel;
use crate::model::{fail, Result};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Rå utdata för en frame, skriven av en steg-typ.
#[derive(Debug, Clone, Default)]
pub struct FrameData {
    /// Stegtid (eller egenvärde/frekvens i egenvärdessteg).
    pub value: f64,
    /// Modnummer i egenvärdessteg, annars −1.
    pub mode: i64,
    pub description: String,
    /// Nodvärden (förskjutning, rotation, temperatur) per slot.
    pub u: Vec<NodeVec>,
    /// Reaktioner i låsta frihetsgrader.
    pub rf: Vec<NodeVec>,
    /// Aktiva respektive låsta slots per nod (bitmask).
    pub active: Vec<u16>,
    pub constrained: Vec<u16>,
    /// Utdata per element (tom om elementet saknar utdata).
    pub elem: Vec<Vec<IpField>>,
    /// Ytterligare nodfält (t.ex. "V", "A"): (fältnamn, komponentnamn, värden per nod).
    pub node_extra: Vec<(String, Vec<String>, Vec<Vec<f64>>)>,
}

#[derive(Debug, Clone, Default)]
pub struct StepOutput {
    pub name: String,
    pub procedure: String,
    pub frames: Vec<FrameData>,
    /// History-utdata: (region, namn, [(tid, värde)]).
    pub history: Vec<(String, String, Vec<(f64, f64)>)>,
    /// Antal inkrement och iterationer (för utskrift).
    pub increments: usize,
    pub iterations: usize,
}

#[derive(Debug, Clone, Default)]
pub struct AnalysisOutput {
    pub steps: Vec<StepOutput>,
    /// Antal obekanta i senaste lösningen.
    pub n_eq: usize,
}

/// Ett fält: komponentnamn och värden per nyckel.
#[derive(Debug, Clone)]
pub struct Field<K: Ord> {
    pub labels: Vec<String>,
    pub items: BTreeMap<K, Vec<(u16, f64)>>,
}

impl<K: Ord> Default for Field<K> {
    fn default() -> Self {
        Field { labels: Vec::new(), items: BTreeMap::new() }
    }
}

impl<K: Ord> Field<K> {
    pub fn label_id(&mut self, l: &str) -> u16 {
        match self.labels.iter().position(|x| x == l) {
            Some(i) => i as u16,
            None => {
                self.labels.push(l.to_string());
                (self.labels.len() - 1) as u16
            }
        }
    }

    pub fn insert(&mut self, key: K, comps: &[(&str, f64)]) {
        let ids: Vec<(u16, f64)> = comps.iter().map(|(l, v)| (self.label_id(l), *v)).collect();
        let e = self.items.entry(key).or_default();
        for (id, v) in ids {
            match e.iter_mut().find(|p| p.0 == id) {
                Some(p) => p.1 = v,
                None => e.push((id, v)),
            }
        }
    }

    pub fn get(&self, key: &K, label: &str) -> Option<f64> {
        let id = self.labels.iter().position(|x| x == label)? as u16;
        self.items.get(key)?.iter().find(|p| p.0 == id).map(|p| p.1)
    }
}

pub type NodeKey = (String, i64);
pub type IpKey = (String, i64, usize, usize);

#[derive(Debug, Clone, Default)]
pub struct Frame {
    pub index: usize,
    pub value: f64,
    pub mode: i64,
    pub description: String,
    pub node: BTreeMap<String, Field<NodeKey>>,
    pub elem: BTreeMap<String, Field<IpKey>>,
}

#[derive(Debug, Clone, Default)]
pub struct HistoryOutput {
    pub region: String,
    pub name: String,
    pub data: Vec<(f64, f64)>,
}

#[derive(Debug, Clone, Default)]
pub struct StepResult {
    pub index: usize,
    pub name: String,
    pub procedure: String,
    pub frames: Vec<Frame>,
    pub history: Vec<HistoryOutput>,
}

/// Resultat för en hel analys (alla steps och frames).
#[derive(Debug, Clone, Default)]
pub struct ResultSet {
    pub steps: Vec<StepResult>,
}

/// Normaliserar fältnamn: plana rotationer kan heta UR3/RM3.
fn field_name(f: &str) -> &str {
    match f {
        "UR3" => "UR",
        "RM3" => "RM",
        other => other,
    }
}

impl Frame {
    /// Bygger en frame med standardfältnamn från rå utdata.
    pub fn from_data(model: &FeModel, d: &FrameData, index: usize) -> Frame {
        let mut fr = Frame { index, value: d.value, mode: d.mode, description: d.description.clone(), ..Default::default() };
        let dim = model.dim();
        let (ul, rl, url, rml): (&[&str], &[&str], &[&str], &[&str]) = if dim == 2 {
            (&["U1", "U2"], &["RF1", "RF2"], &["UR3"], &["RM3"])
        } else {
            (&["U1", "U2", "U3"], &["RF1", "RF2", "RF3"], &["UR1", "UR2", "UR3"], &["RM1", "RM2", "RM3"])
        };
        let rot_slots: &[usize] = if dim == 2 { &[5] } else { &[3, 4, 5] };
        let mut add = |name: &str, n: usize, labels: &[&str], slots: &[usize], src: &[NodeVec]| {
            let key = (model.instances[model.node_inst[n] as usize].clone(), model.node_label[n]);
            let comps: Vec<(&str, f64)> = labels.iter().zip(slots).map(|(l, &s)| (*l, src.get(n).map_or(0.0, |v| v[s]))).collect();
            fr.node.entry(name.to_string()).or_default().insert(key, &comps);
        };
        for n in 0..model.nodes.len() {
            let act = d.active.get(n).copied().unwrap_or(0);
            add("U", n, ul, &[0, 1, 2], &d.u);
            if !d.rf.is_empty() {
                add("RF", n, rl, &[0, 1, 2], &d.rf);
            }
            if rot_slots.iter().any(|&s| act & (1 << s) != 0) {
                add("UR", n, url, rot_slots, &d.u);
                if !d.rf.is_empty() {
                    add("RM", n, rml, rot_slots, &d.rf);
                }
            }
            if act & (1 << 7) != 0 {
                add("NT11", n, &["NT11"], &[7], &d.u);
                if !d.rf.is_empty() {
                    add("RFL11", n, &["RFL11"], &[7], &d.rf);
                }
            }
        }
        for (name, labels, vals) in &d.node_extra {
            let f = fr.node.entry(name.clone()).or_default();
            for (n, v) in vals.iter().enumerate() {
                let key = (model.instances[model.node_inst[n] as usize].clone(), model.node_label[n]);
                let comps: Vec<(&str, f64)> = labels.iter().map(|s| s.as_str()).zip(v.iter().copied()).collect();
                f.insert(key, &comps);
            }
        }
        for (e, fields) in d.elem.iter().enumerate() {
            let el = &model.elements[e];
            let inst = model.instances[el.inst as usize].clone();
            for f in fields {
                let field = fr.elem.entry(f.name.clone()).or_default();
                for (ip, sp, v) in &f.points {
                    let comps: Vec<(&str, f64)> = f.labels.iter().map(|s| s.as_str()).zip(v.iter().copied()).collect();
                    field.insert((inst.clone(), el.label, *ip, *sp), &comps);
                }
            }
        }
        fr
    }
}

impl AnalysisOutput {
    pub fn to_result_set(&self, model: &FeModel) -> ResultSet {
        ResultSet {
            steps: self
                .steps
                .iter()
                .enumerate()
                .map(|(i, s)| StepResult {
                    index: i + 1,
                    name: s.name.clone(),
                    procedure: s.procedure.clone(),
                    frames: s.frames.iter().enumerate().map(|(k, f)| Frame::from_data(model, f, k)).collect(),
                    history: s.history.iter().map(|(r, n, d)| HistoryOutput { region: r.clone(), name: n.clone(), data: d.clone() }).collect(),
                })
                .collect(),
        }
    }
}

fn clean(s: &str) -> String {
    s.replace([';', '\n', '\r'], " ")
}

impl ResultSet {
    /// Skriver resultaten i textformatet "rustyfem-res 1".
    pub fn to_text(&self) -> String {
        let mut s = String::from("# rustyfem-res 1\n");
        for st in &self.steps {
            let _ = writeln!(s, "STEP;{};{};{}", st.index, clean(&st.name), clean(&st.procedure));
            for fr in &st.frames {
                let _ = writeln!(s, "FRAME;{};{};{:.9e};{};{}", st.index, fr.index, fr.value, fr.mode, clean(&fr.description));
                for (name, f) in &fr.node {
                    for ((inst, label), comps) in &f.items {
                        let _ = write!(s, "N;{name};{inst};{label}");
                        for (id, v) in comps {
                            let _ = write!(s, ";{}={:.9e}", f.labels[*id as usize], v);
                        }
                        s.push('\n');
                    }
                }
                for (name, f) in &fr.elem {
                    for ((inst, e, ip, sp), comps) in &f.items {
                        let _ = write!(s, "E;{name};{inst};{e};{ip};{sp}");
                        for (id, v) in comps {
                            let _ = write!(s, ";{}={:.9e}", f.labels[*id as usize], v);
                        }
                        s.push('\n');
                    }
                }
            }
            for h in &st.history {
                let _ = write!(s, "H;{};{};{}", st.index, clean(&h.region), clean(&h.name));
                for (t, v) in &h.data {
                    let _ = write!(s, ";{t:.9e}:{v:.9e}");
                }
                s.push('\n');
            }
        }
        s
    }

    /// Läser textformatet "rustyfem-res 1".
    pub fn parse(text: &str) -> Result<ResultSet> {
        let mut rs = ResultSet::default();
        for (no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let p: Vec<&str> = line.split(';').collect();
            let bad = || crate::model::FemError(format!("resultatfil rad {}: oläsbar rad '{line}'", no + 1));
            let num = |s: &str| s.trim().parse::<f64>().map_err(|_| bad());
            let int = |s: &str| s.trim().parse::<i64>().map_err(|_| bad());
            let comps = |fields: &[&str]| -> Result<Vec<(String, f64)>> {
                fields
                    .iter()
                    .filter(|f| !f.is_empty())
                    .map(|f| {
                        let (l, v) = f.split_once('=').ok_or_else(bad)?;
                        Ok((l.to_string(), num(v)?))
                    })
                    .collect()
            };
            match p[0] {
                "STEP" if p.len() >= 4 => rs.steps.push(StepResult {
                    index: int(p[1])? as usize,
                    name: p[2].to_string(),
                    procedure: p[3].to_string(),
                    ..Default::default()
                }),
                "FRAME" if p.len() >= 5 => {
                    let st = rs.steps.last_mut().ok_or_else(bad)?;
                    st.frames.push(Frame {
                        index: int(p[2])? as usize,
                        value: num(p[3])?,
                        mode: int(p[4])?,
                        description: p.get(5).unwrap_or(&"").to_string(),
                        ..Default::default()
                    });
                }
                "N" if p.len() >= 4 => {
                    let fr = rs.steps.last_mut().and_then(|s| s.frames.last_mut()).ok_or_else(bad)?;
                    let c = comps(&p[4..])?;
                    let cr: Vec<(&str, f64)> = c.iter().map(|(l, v)| (l.as_str(), *v)).collect();
                    fr.node.entry(field_name(p[1]).to_string()).or_default().insert((p[2].to_uppercase(), int(p[3])?), &cr);
                }
                "E" if p.len() >= 6 => {
                    let fr = rs.steps.last_mut().and_then(|s| s.frames.last_mut()).ok_or_else(bad)?;
                    let c = comps(&p[6..])?;
                    let cr: Vec<(&str, f64)> = c.iter().map(|(l, v)| (l.as_str(), *v)).collect();
                    let key = (p[2].to_uppercase(), int(p[3])?, int(p[4])? as usize, int(p[5])? as usize);
                    fr.elem.entry(p[1].to_string()).or_default().insert(key, &cr);
                }
                "H" if p.len() >= 4 => {
                    let st = rs.steps.last_mut().ok_or_else(bad)?;
                    let mut data = Vec::new();
                    for f in &p[4..] {
                        if f.is_empty() {
                            continue;
                        }
                        let (t, v) = f.split_once(':').ok_or_else(bad)?;
                        data.push((num(t)?, num(v)?));
                    }
                    st.history.push(HistoryOutput { region: p[2].to_string(), name: p[3].to_string(), data });
                }
                _ => return fail(format!("resultatfil rad {}: okänd rad '{line}'", no + 1)),
            }
        }
        Ok(rs)
    }
}

/// Summa av en nodvektor över alla noder.
pub fn sum_nodes(v: &[NodeVec]) -> NodeVec {
    let mut s = [0.0; SLOTS];
    for n in v {
        for i in 0..SLOTS {
            s[i] += n[i];
        }
    }
    s
}
