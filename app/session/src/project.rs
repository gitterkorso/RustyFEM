//! The project file `.rustyproj`: binary format with magic number and version.
//!
//! ```text
//! "RUSTYPRJ"  u32 version
//! source:     u8 kind (0 = .rusty, 1 = .inp), string path, string text
//! included files: u32 count, (string name, bytes contents) ...
//! edits:      u32 count, (u8 kind, fields) ...   (the commands in order)
//! unit system: string
//! results:    u8 present, then the analysis output (steps, frames, fields, history)
//! ```
//! Integers and floats are little-endian; strings and bytes have their length (u64) first.
//! The model is rebuilt from the source and the edits when the file is opened.

use crate::edit::Edit;
use rustyfem::elements::IpField;
use rustyfem::fem::dof::SLOTS;
use rustyfem::fem::output::{AnalysisOutput, FrameData, StepOutput};

const MAGIC: &[u8; 8] = b"RUSTYPRJ";
pub const VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    Rusty,
    Inp,
}

#[derive(Debug, Clone)]
pub struct Project {
    pub kind: SourceKind,
    pub path: String,
    pub text: String,
    pub included: Vec<(String, Vec<u8>)>,
    pub edits: Vec<Edit>,
    pub units: String,
    pub results: Option<AnalysisOutput>,
}

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend(v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend(v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend(v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend(v.to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
    fn opt_f64(&mut self, v: Option<f64>) {
        match v {
            Some(x) => {
                self.u8(1);
                self.f64(x);
            }
            None => self.u8(0),
        }
    }
    fn strs(&mut self, v: &[String]) {
        self.u32(v.len() as u32);
        for s in v {
            self.str(s);
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

type Res<T> = Result<T, String>;

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        if self.p + n > self.b.len() {
            return Err("the project file is corrupt (ends too early)".into());
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Res<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Res<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Res<Vec<u8>> {
        let n = self.u64()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    fn str(&mut self) -> Res<String> {
        String::from_utf8(self.bytes()?).map_err(|_| "the project file is corrupt (invalid text)".to_string())
    }
    fn opt_f64(&mut self) -> Res<Option<f64>> {
        Ok(if self.u8()? == 1 { Some(self.f64()?) } else { None })
    }
    fn strs(&mut self) -> Res<Vec<String>> {
        let n = self.u32()?;
        (0..n).map(|_| self.str()).collect()
    }
}

fn write_edit(w: &mut W, e: &Edit) {
    match e {
        Edit::FemText { text } => {
            w.u8(0);
            w.str(text);
        }
        Edit::MaterialData { material, behavior, line, fields } => {
            w.u8(1);
            w.u32(*material);
            w.u32(*behavior);
            w.u32(*line);
            w.strs(fields);
        }
        Edit::Density { material, value } => {
            w.u8(2);
            w.u32(*material);
            w.f64(*value);
        }
        Edit::SectionThickness { section, value } => {
            w.u8(3);
            w.u32(*section);
            w.f64(*value);
        }
        Edit::ProcedureData { step, line, fields } => {
            w.u8(4);
            w.u32(*step);
            w.u32(*line);
            w.strs(fields);
        }
        Edit::Nlgeom { step, value } => {
            w.u8(5);
            w.u32(*step);
            w.u8(match value {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            });
        }
        Edit::MaxIncrements { step, value } => {
            w.u8(6);
            w.u32(*step);
            w.u32(*value);
        }
        Edit::BcValue { step, block, item, value } => {
            w.u8(7);
            w.u32(step.map(|s| s + 1).unwrap_or(0));
            w.u32(*block);
            w.u32(*item);
            w.f64(*value);
        }
        Edit::CloadValue { step, block, item, value } => {
            w.u8(8);
            w.u32(*step);
            w.u32(*block);
            w.u32(*item);
            w.f64(*value);
        }
        Edit::DloadValue { step, block, item, value } => {
            w.u8(9);
            w.u32(*step);
            w.u32(*block);
            w.u32(*item);
            w.f64(*value);
        }
        Edit::StepName { step, name } => {
            w.u8(10);
            w.u32(*step);
            w.str(name);
        }
    }
}

fn read_edit(r: &mut R) -> Res<Edit> {
    Ok(match r.u8()? {
        0 => Edit::FemText { text: r.str()? },
        1 => Edit::MaterialData { material: r.u32()?, behavior: r.u32()?, line: r.u32()?, fields: r.strs()? },
        2 => Edit::Density { material: r.u32()?, value: r.f64()? },
        3 => Edit::SectionThickness { section: r.u32()?, value: r.f64()? },
        4 => Edit::ProcedureData { step: r.u32()?, line: r.u32()?, fields: r.strs()? },
        5 => Edit::Nlgeom {
            step: r.u32()?,
            value: match r.u8()? {
                0 => None,
                1 => Some(false),
                _ => Some(true),
            },
        },
        6 => Edit::MaxIncrements { step: r.u32()?, value: r.u32()? },
        7 => {
            let s = r.u32()?;
            Edit::BcValue { step: if s == 0 { None } else { Some(s - 1) }, block: r.u32()?, item: r.u32()?, value: r.f64()? }
        }
        8 => Edit::CloadValue { step: r.u32()?, block: r.u32()?, item: r.u32()?, value: r.f64()? },
        9 => Edit::DloadValue { step: r.u32()?, block: r.u32()?, item: r.u32()?, value: r.f64()? },
        10 => Edit::StepName { step: r.u32()?, name: r.str()? },
        k => return Err(format!("unknown edit {k} in the project file (newer version?)")),
    })
}

fn write_nodevecs(w: &mut W, v: &[[f64; SLOTS]]) {
    w.u64(v.len() as u64);
    for x in v {
        for &y in x {
            w.f64(y);
        }
    }
}

fn read_nodevecs(r: &mut R) -> Res<Vec<[f64; SLOTS]>> {
    let n = r.u64()? as usize;
    let mut out = Vec::with_capacity(n.min(1 << 24));
    for _ in 0..n {
        let mut x = [0.0; SLOTS];
        for y in &mut x {
            *y = r.f64()?;
        }
        out.push(x);
    }
    Ok(out)
}

fn write_frame(w: &mut W, f: &FrameData) {
    w.f64(f.value);
    w.i64(f.mode);
    w.str(&f.description);
    write_nodevecs(w, &f.u);
    write_nodevecs(w, &f.rf);
    for v in [&f.active, &f.constrained] {
        w.u64(v.len() as u64);
        for &m in v.iter() {
            w.0.extend(m.to_le_bytes());
        }
    }
    w.u64(f.elem.len() as u64);
    for fields in &f.elem {
        w.u32(fields.len() as u32);
        for fl in fields {
            w.str(&fl.name);
            w.strs(&fl.labels);
            w.u32(fl.points.len() as u32);
            for (ip, sp, vals) in &fl.points {
                w.u32(*ip as u32);
                w.u32(*sp as u32);
                w.u32(vals.len() as u32);
                for &v in vals {
                    w.f64(v);
                }
            }
        }
    }
    w.u32(f.node_extra.len() as u32);
    for (name, comps, vals) in &f.node_extra {
        w.str(name);
        w.strs(comps);
        w.u64(vals.len() as u64);
        for v in vals {
            w.u32(v.len() as u32);
            for &x in v {
                w.f64(x);
            }
        }
    }
}

fn read_frame(r: &mut R) -> Res<FrameData> {
    let value = r.f64()?;
    let mode = r.i64()?;
    let description = r.str()?;
    let u = read_nodevecs(r)?;
    let rf = read_nodevecs(r)?;
    let mut masks = Vec::new();
    for _ in 0..2 {
        let n = r.u64()? as usize;
        let mut v = Vec::with_capacity(n.min(1 << 24));
        for _ in 0..n {
            v.push(u16::from_le_bytes(r.take(2)?.try_into().unwrap()));
        }
        masks.push(v);
    }
    let constrained = masks.pop().unwrap();
    let active = masks.pop().unwrap();
    let ne = r.u64()? as usize;
    let mut elem = Vec::with_capacity(ne.min(1 << 24));
    for _ in 0..ne {
        let nf = r.u32()?;
        let mut fields = Vec::new();
        for _ in 0..nf {
            let name = r.str()?;
            let labels = r.strs()?;
            let np = r.u32()?;
            let mut points = Vec::new();
            for _ in 0..np {
                let ip = r.u32()? as usize;
                let sp = r.u32()? as usize;
                let nv = r.u32()?;
                let vals = (0..nv).map(|_| r.f64()).collect::<Res<Vec<f64>>>()?;
                points.push((ip, sp, vals));
            }
            fields.push(IpField { name, labels, points });
        }
        elem.push(fields);
    }
    let nx = r.u32()?;
    let mut node_extra = Vec::new();
    for _ in 0..nx {
        let name = r.str()?;
        let comps = r.strs()?;
        let n = r.u64()? as usize;
        let mut vals = Vec::with_capacity(n.min(1 << 24));
        for _ in 0..n {
            let k = r.u32()?;
            vals.push((0..k).map(|_| r.f64()).collect::<Res<Vec<f64>>>()?);
        }
        node_extra.push((name, comps, vals));
    }
    Ok(FrameData { value, mode, description, u, rf, active, constrained, elem, node_extra })
}

fn write_results(w: &mut W, o: &AnalysisOutput) {
    w.u64(o.n_eq as u64);
    w.u32(o.steps.len() as u32);
    for s in &o.steps {
        w.str(&s.name);
        w.str(&s.procedure);
        w.u64(s.increments as u64);
        w.u64(s.iterations as u64);
        w.u32(s.frames.len() as u32);
        for f in &s.frames {
            write_frame(w, f);
        }
        w.u32(s.history.len() as u32);
        for (region, name, data) in &s.history {
            w.str(region);
            w.str(name);
            w.u64(data.len() as u64);
            for &(t, v) in data {
                w.f64(t);
                w.f64(v);
            }
        }
    }
}

fn read_results(r: &mut R) -> Res<AnalysisOutput> {
    let n_eq = r.u64()? as usize;
    let ns = r.u32()?;
    let mut steps = Vec::new();
    for _ in 0..ns {
        let name = r.str()?;
        let procedure = r.str()?;
        let increments = r.u64()? as usize;
        let iterations = r.u64()? as usize;
        let nf = r.u32()?;
        let frames = (0..nf).map(|_| read_frame(r)).collect::<Res<Vec<_>>>()?;
        let nh = r.u32()?;
        let mut history = Vec::new();
        for _ in 0..nh {
            let region = r.str()?;
            let hname = r.str()?;
            let n = r.u64()? as usize;
            let mut data = Vec::with_capacity(n.min(1 << 24));
            for _ in 0..n {
                data.push((r.f64()?, r.f64()?));
            }
            history.push((region, hname, data));
        }
        steps.push(StepOutput { name, procedure, frames, history, increments, iterations });
    }
    Ok(AnalysisOutput { steps, n_eq })
}

impl Project {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(MAGIC);
        w.u32(VERSION);
        w.u8(match self.kind {
            SourceKind::Rusty => 0,
            SourceKind::Inp => 1,
        });
        w.str(&self.path);
        w.str(&self.text);
        w.u32(self.included.len() as u32);
        for (n, b) in &self.included {
            w.str(n);
            w.bytes(b);
        }
        w.u32(self.edits.len() as u32);
        for e in &self.edits {
            write_edit(&mut w, e);
        }
        w.str(&self.units);
        match &self.results {
            Some(o) => {
                w.u8(1);
                write_results(&mut w, o);
            }
            None => w.u8(0),
        }
        w.opt_f64(None);
        w.0
    }

    pub fn from_bytes(b: &[u8]) -> Res<Project> {
        if b.len() < 12 || &b[..8] != MAGIC {
            return Err("the file is not a RustyFEM project file".into());
        }
        let mut r = R { b, p: 8 };
        let version = r.u32()?;
        if version > VERSION {
            return Err(format!("the project file has version {version}; this version of the app reads up to {VERSION}"));
        }
        let kind = if r.u8()? == 0 { SourceKind::Rusty } else { SourceKind::Inp };
        let path = r.str()?;
        let text = r.str()?;
        let ni = r.u32()?;
        let mut included = Vec::new();
        for _ in 0..ni {
            included.push((r.str()?, r.bytes()?));
        }
        let ne = r.u32()?;
        let edits = (0..ne).map(|_| read_edit(&mut r)).collect::<Res<Vec<_>>>()?;
        let units = r.str()?;
        let results = if r.u8()? == 1 { Some(read_results(&mut r)?) } else { None };
        let _reserved = r.opt_f64()?;
        Ok(Project { kind, path, text, included, edits, units, results })
    }
}
