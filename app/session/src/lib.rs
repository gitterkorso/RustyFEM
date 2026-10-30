//! App-API för RustyFEM (se PLAN.md, Fas 1).
//!
//! En `Session` är en öppen modell: källan (`.rusty`-skript eller `.inp`), den
//! byggda `FeModel`, ångra/gör om, visningsgeometri och senaste resultat.
//! Körningen (`run_analysis`) görs av anroparen på en egen tråd, med framsteg
//! och avbrott via `rustyfem::progress`. Inget här ändrar motorns resultat.

pub mod edit;
pub mod fields;
pub mod info;
pub mod memory;
pub mod mesh;
pub mod platform;
pub mod project;
pub mod units;

use edit::Edit;
use fields::{FieldInfo, Position, StepResultInfo};
use mesh::DisplayMesh;
use project::{Project, SourceKind};
use units::{UnitSymbols, UnitSystem};
use rustyfem::fem::model::FeModel;
use rustyfem::fem::output::AnalysisOutput;
use rustyfem::progress::{self, Event};
use rustyfem::steps::{self, RunOptions};
use serde::Serialize;
use specta::Type;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub type Res<T> = Result<T, String>;

/// Modellens källa.
#[derive(Debug, Clone, Serialize, Type)]
pub struct SessionInfo {
    pub kind: String,
    pub path: Option<String>,
    pub name: String,
    pub units: UnitSystem,
    /// Enhetssymbolerna för `units` (tomma när inget enhetssystem är angivet).
    pub unit_symbols: UnitSymbols,
    /// Aktuellt `.rusty`-skript (bara för `.rusty`-sessioner).
    pub script: Option<String>,
    /// Skriptets dimension (2 eller 3; bara för `.rusty`-sessioner). Gäller
    /// även en tom modell, där `FeModel` saknar element att avgöra den av.
    pub script_dim: Option<u32>,
    /// Varför modellen inte går att lösa än (t.ex. inga element, eller ett
    /// stöd som inte träffar några noder). `None` när modellen är komplett.
    pub problem: Option<String>,
    pub undo: Option<String>,
    pub redo: Option<String>,
    pub dirty: bool,
    pub has_results: bool,
    pub results_stale: bool,
}

pub struct Session {
    pub kind: SourceKind,
    pub path: Option<PathBuf>,
    /// Ursprunglig källtext (för `.rusty` den aktuella texten efter redigeringar).
    original: String,
    text: String,
    included: Vec<(String, Vec<u8>)>,
    model: Arc<FeModel>,
    mesh: Arc<DisplayMesh>,
    script_dim: Option<u32>,
    problem: Option<String>,
    undo: Vec<(Edit, Edit)>,
    redo: Vec<(Edit, Edit)>,
    pub results: Option<Arc<AnalysisOutput>>,
    pub results_stale: bool,
    pub units: UnitSystem,
    pub dirty: bool,
}

/// Mall för en ny modell.
pub const NEW_MODEL: &str = "# Ny modell – verktygen i appen lägger till rader här.\n\
model 2d\n\
material steel E=210e9 nu=0.3\n\
plate x=0..2 y=0..0.2 t=0.01 size=0.05\n\
fix x=0\n\
load x=2 fy=-1000\n";


/// Tom modell att bygga från grunden.
pub const EMPTY_2D: &str = "# Ny modell – verktygen i appen lägger till rader här.\nmodel 2d\n";
pub const EMPTY_3D: &str = "# Ny modell – verktygen i appen lägger till rader här.\nmodel 3d\n";

/// En `FeModel` byggd från ett `.rusty`-skript.
struct Built {
    model: FeModel,
    dim: u32,
    /// Varför modellen inte går att lösa än.
    problem: Option<String>,
}

/// Bygger en `FeModel` från ett `.rusty`-skript. Fel i skriptet (syntax,
/// okända kommandon, material utan E) avvisas. En modell som är riktig men
/// ofullständig – inga element än, eller stöd och laster som inte träffar
/// några noder – byggs utan steg, så att man kan bygga den rad för rad.
fn build_script(text: &str) -> Res<Built> {
    let m = rustyfem::script::parse(text).map_err(|e| e.0)?;
    let dim = if m.dim == rustyfem::model::Dim::Two { 2 } else { 3 };
    match rustyfem::legacy::to_fe_static(&m) {
        Ok(model) => Ok(Built { model, dim, problem: None }),
        Err(e) => {
            let model = rustyfem::legacy::to_fe(&m).map_err(|_| e.0.clone())?;
            Ok(Built { model, dim, problem: Some(e.0) })
        }
    }
}

impl Session {
    fn from_script(path: Option<PathBuf>, text: String, b: Built) -> Session {
        let mut s = Session::from_model(SourceKind::Rusty, path, text, Vec::new(), b.model);
        s.script_dim = Some(b.dim);
        s.problem = b.problem;
        s
    }

    fn from_model(kind: SourceKind, path: Option<PathBuf>, text: String, included: Vec<(String, Vec<u8>)>, model: FeModel) -> Session {
        // .rusty-skripten använder SI (standardmaterialen är i Pa); en .inp gissas ur materialen
        let units = if kind == SourceKind::Rusty { UnitSystem::Si } else { UnitSystem::guess(&model) };
        let mesh = Arc::new(mesh::extract(&model));
        Session {
            kind,
            path,
            original: text.clone(),
            text,
            included,
            model: Arc::new(model),
            mesh,
            script_dim: None,
            problem: None,
            undo: Vec::new(),
            redo: Vec::new(),
            results: None,
            results_stale: false,
            units,
            dirty: false,
        }
    }

    /// Ny `.rusty`-modell från text.
    pub fn new_fem(text: &str) -> Res<Session> {
        Ok(Session::from_script(None, text.to_string(), build_script(text)?))
    }

    /// `.inp`-text; relativa `*INCLUDE` tolkas från `dir`.
    pub fn from_inp_text(text: &str, dir: &Path) -> Res<Session> {
        let r = rustyfem::inp::read_str(text, dir).map_err(|e| e.0)?;
        Ok(Session::from_model(SourceKind::Inp, None, text.to_string(), Vec::new(), r.model))
    }

    /// Öppnar `.inp`, `.rusty` eller `.rustyproj`.
    pub fn open(path: &Path) -> Res<Session> {
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let read = |p: &Path| std::fs::read(p).map_err(|e| format!("kunde inte läsa {}: {e}", p.display()));
        match ext.as_str() {
            "rustyproj" => Session::from_project(Project::from_bytes(&read(path)?)?, Some(path.to_path_buf())),
            "rusty" => {
                let text = String::from_utf8_lossy(&read(path)?).into_owned();
                let b = build_script(&text)?;
                Ok(Session::from_script(Some(path.to_path_buf()), text, b))
            }
            _ => {
                let r = rustyfem::inp::read_file(path).map_err(|e| e.0)?;
                let text = String::from_utf8_lossy(&read(path)?).into_owned();
                let dir = path.parent().unwrap_or(Path::new("."));
                let mut included = Vec::new();
                for p in &r.included {
                    let name = p.strip_prefix(dir).unwrap_or(p).to_string_lossy().replace('\\', "/");
                    if let Ok(b) = std::fs::read(p) {
                        included.push((name, b));
                    }
                }
                Ok(Session::from_model(SourceKind::Inp, Some(path.to_path_buf()), text, included, r.model))
            }
        }
    }

    fn from_project(p: Project, path: Option<PathBuf>) -> Res<Session> {
        let mut s = match p.kind {
            SourceKind::Rusty => Session::from_script(None, p.text.clone(), build_script(&p.text)?),
            SourceKind::Inp => {
                // *INCLUDE läses ur projektfilen via en filresolver (fungerar även i iOS-sandlådan)
                let root = Path::new("projekt");
                let files = &p.included;
                let resolve = |path: &Path| -> rustyfem::Result<Vec<u8>> {
                    let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/");
                    files
                        .iter()
                        .find(|(n, _)| *n == rel)
                        .map(|(_, b)| b.clone())
                        .ok_or_else(|| rustyfem::FemError(format!("projektfilen saknar den inkluderade filen {rel}")))
                };
                let r = rustyfem::inp::read_str_with(&p.text, root, &resolve).map_err(|e| e.0)?;
                Session::from_model(SourceKind::Inp, None, p.text.clone(), p.included.clone(), r.model)
            }
        };
        for e in &p.edits {
            s.apply(e.clone())?;
        }
        s.path = path;
        // äldre projektfiler sparade alltid "SI (m, N, Pa)"; då gäller standardvalet ovan
        if p.units != units::LEGACY_LABEL {
            s.units = UnitSystem::parse(&p.units);
        }
        s.results = p.results.map(Arc::new);
        s.results_stale = false;
        s.dirty = false;
        Ok(s)
    }

    /// Projektfilens innehåll.
    pub fn to_project(&self) -> Vec<u8> {
        Project {
            kind: self.kind,
            path: self.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_default(),
            text: self.original.clone(),
            included: self.included.clone(),
            edits: self.undo.iter().map(|(f, _)| f.clone()).collect(),
            units: self.units.key().into(),
            results: self.results.as_ref().map(|r| (**r).clone()),
        }
        .to_bytes()
    }

    pub fn model(&self) -> &Arc<FeModel> {
        &self.model
    }

    pub fn mesh(&self) -> &Arc<DisplayMesh> {
        &self.mesh
    }

    /// Varför modellen inte går att lösa än (`None` när den är komplett).
    pub fn problem(&self) -> Option<&str> {
        self.problem.as_deref()
    }

    pub fn info(&self) -> SessionInfo {
        let name = self.path.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Namnlös".into());
        SessionInfo {
            kind: match self.kind {
                SourceKind::Rusty => "rusty".into(),
                SourceKind::Inp => "inp".into(),
            },
            path: self.path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            name,
            units: self.units,
            unit_symbols: self.units.symbols(),
            script: (self.kind == SourceKind::Rusty).then(|| self.text.clone()),
            script_dim: self.script_dim,
            problem: self.problem.clone(),
            undo: self.undo.last().map(|e| e.0.label()),
            redo: self.redo.last().map(|e| e.0.label()),
            dirty: self.dirty,
            has_results: self.results.is_some(),
            results_stale: self.results_stale,
        }
    }

    /// Utför en ändring. Returnerar `true` om geometrin ändrades.
    fn exec(&mut self, e: &Edit) -> Res<(Edit, bool)> {
        match e {
            Edit::FemText { text } => {
                if self.kind != SourceKind::Rusty {
                    return Err("skripttext kan bara ändras i en .rusty-modell".into());
                }
                let b = build_script(text)?;
                let old = std::mem::replace(&mut self.text, text.clone());
                self.mesh = Arc::new(mesh::extract(&b.model));
                self.model = Arc::new(b.model);
                self.script_dim = Some(b.dim);
                self.problem = b.problem;
                Ok((Edit::FemText { text: old }, true))
            }
            _ => {
                let inv = edit::apply(Arc::make_mut(&mut self.model), e)?;
                Ok((inv, false))
            }
        }
    }

    /// Utför en ändring och lägger den i ångra-historiken.
    pub fn apply(&mut self, e: Edit) -> Res<bool> {
        let (inv, geom) = self.exec(&e)?;
        self.undo.push((e, inv));
        self.redo.clear();
        self.touched();
        Ok(geom)
    }

    pub fn undo(&mut self) -> Res<bool> {
        let Some((fwd, inv)) = self.undo.pop() else { return Ok(false) };
        match self.exec(&inv) {
            Ok((_, geom)) => {
                self.redo.push((fwd, inv));
                self.touched();
                Ok(geom)
            }
            Err(e) => {
                self.undo.push((fwd, inv));
                Err(e)
            }
        }
    }

    pub fn redo(&mut self) -> Res<bool> {
        let Some((fwd, inv)) = self.redo.pop() else { return Ok(false) };
        match self.exec(&fwd) {
            Ok((_, geom)) => {
                self.undo.push((fwd, inv));
                self.touched();
                Ok(geom)
            }
            Err(e) => {
                self.redo.push((fwd, inv));
                Err(e)
            }
        }
    }

    /// Byter enhetssystem. Bara etiketterna ändras; inga värden räknas om.
    pub fn set_units(&mut self, u: UnitSystem) {
        if u != self.units {
            self.units = u;
            self.dirty = true;
        }
    }

    fn touched(&mut self) {
        self.dirty = true;
        if self.results.is_some() {
            self.results_stale = true;
        }
    }

    pub fn set_results(&mut self, out: AnalysisOutput) {
        self.results = Some(Arc::new(out));
        self.results_stale = false;
        self.dirty = true;
    }

    /// Resultatkatalogen med fältens enheter (`None` utan resultat).
    pub fn catalog(&self) -> Option<Vec<StepResultInfo>> {
        let mut cat = fields::catalog(self.results.as_ref()?);
        for f in cat.iter_mut().flat_map(|s| s.fields.iter_mut()) {
            f.unit = self.units.field_unit(&f.name);
        }
        Some(cat)
    }

    fn frame(&self, step: usize, frame: usize) -> Res<&rustyfem::fem::output::FrameData> {
        let r = self.results.as_ref().ok_or("inga resultat – lös modellen först")?;
        r.steps.get(step).and_then(|s| s.frames.get(frame)).ok_or_else(|| format!("steg {} frame {} finns inte", step + 1, frame))
    }

    /// Nodvärden för ett fält: `f32` per nod, sist min och max (f32).
    pub fn field_values(&self, step: usize, frame: usize, field: &str, comp: &str) -> Res<Vec<u8>> {
        let fr = self.frame(step, frame)?;
        let fi = fields::frame_fields(fr).into_iter().find(|f| f.name == field).ok_or_else(|| format!("fältet {field} finns inte i framen"))?;
        let (v, lo, hi) = fields::node_values(fr, &self.mesh, self.model.nodes.len(), &fi, comp);
        let mut b = Vec::with_capacity(v.len() * 4 + 8);
        for x in v {
            b.extend(x.to_le_bytes());
        }
        b.extend((lo as f32).to_le_bytes());
        b.extend((hi as f32).to_le_bytes());
        Ok(b)
    }

    /// Förskjutningar (f32 ×3 per nod).
    pub fn displacement(&self, step: usize, frame: usize) -> Res<Vec<u8>> {
        let fr = self.frame(step, frame)?;
        Ok(fields::displacement(fr, self.model.nodes.len()).into_iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    /// Min och max av ett fält (nodvärden, elementfält utjämnade) med noderna där de finns.
    pub fn field_stats(&self, step: usize, frame: usize, field: &str, comp: &str) -> Res<FieldStats> {
        let fr = self.frame(step, frame)?;
        let fi = fields::frame_fields(fr).into_iter().find(|f| f.name == field).ok_or_else(|| format!("fältet {field} finns inte i framen"))?;
        let comp = if comp.is_empty() { fi.components[0].clone() } else { comp.to_string() };
        if !fi.components.contains(&comp) {
            return Err(format!("{field} har komponenterna {}", fi.components.join(", ")));
        }
        let (v, lo, hi) = fields::node_values(fr, &self.mesh, self.model.nodes.len(), &fi, &comp);
        let at = |x: f64| v.iter().position(|&y| y.is_finite() && (y as f64 - x).abs() <= 1e-6 * x.abs().max(1e-30)).map(|n| self.model.node_name(n));
        Ok(FieldStats { field: field.into(), unit: self.units.field_unit(field), component: comp, min: lo, max: hi, min_at: at(lo), max_at: at(hi) })
    }

    /// Nod- eller elementindex från "INSTANS.nummer" eller bara "nummer".
    pub fn lookup(&self, what: &str, element: bool) -> Res<usize> {
        let m = &self.model;
        let (inst, label) = match what.rsplit_once('.') {
            Some((i, l)) => (m.instance_index(&i.to_uppercase()).ok_or_else(|| format!("instansen {i} finns inte"))?, l),
            None => (m.instance_index(rustyfem::fem::model::FLAT_INSTANCE).or_else(|| m.instance_index(rustyfem::fem::model::ASSEMBLY_INSTANCE)).unwrap_or(0), what),
        };
        let label: i64 = label.trim().parse().map_err(|_| format!("'{what}' är inget nod-/elementnummer"))?;
        let found = if element { m.element(inst, label) } else { m.node(inst, label) };
        found.ok_or_else(|| format!("{} {what} finns inte", if element { "elementet" } else { "noden" }))
    }

    pub fn history(&self, step: usize, index: usize) -> Res<Vec<(f64, f64)>> {
        let r = self.results.as_ref().ok_or("inga resultat")?;
        r.steps.get(step).and_then(|s| s.history.get(index)).map(|h| h.2.clone()).ok_or_else(|| "history-utdata finns inte".to_string())
    }

    /// Uppgifter om ett element (och dess resultat i en frame).
    pub fn element_info(&self, e: usize, at: Option<(usize, usize)>) -> Res<PickInfo> {
        let m = &self.model;
        let el = m.elements.get(e).ok_or("elementet finns inte")?;
        let section = el.section.and_then(|s| m.sections.get(s));
        let mut values = Vec::new();
        if let Some((s, f)) = at {
            if let Ok(fr) = self.frame(s, f) {
                for fl in fr.elem.get(e).map(|v| v.as_slice()).unwrap_or(&[]) {
                    let fi = FieldInfo { name: fl.name.clone(), position: Position::Element, components: fl.labels.clone(), unit: String::new() };
                    let mut comps: Vec<String> = fi.components.clone();
                    if fl.name == "S" {
                        comps.insert(0, "Mises".into());
                    }
                    for c in comps {
                        if let Some(v) = fields::element_mean(fr, e, &fl.name, &c) {
                            values.push((format!("{} {}", if c == fl.name { String::new() } else { fl.name.clone() }, c).trim().to_string(), v, self.units.field_unit(&fl.name)));
                        }
                    }
                }
            }
        }
        Ok(PickInfo {
            kind: "element".into(),
            name: m.elem_name(e),
            index: e as u32,
            details: vec![
                ("Typ".into(), el.etype.name().into()),
                ("Noder".into(), el.nodes.iter().take(el.etype.n_connectivity()).map(|&n| m.node_label[n].to_string()).collect::<Vec<_>>().join(", ")),
                ("Tvärsnitt".into(), section.map(|s| s.elset.clone()).unwrap_or_else(|| "–".into())),
                ("Material".into(), section.map(|s| s.material.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| "–".into())),
            ],
            values,
        })
    }

    /// Uppgifter om en nod.
    pub fn node_info(&self, n: usize, at: Option<(usize, usize)>) -> Res<PickInfo> {
        let m = &self.model;
        let p = m.nodes.get(n).ok_or("noden finns inte")?;
        let mut values = Vec::new();
        if let Some((s, f)) = at {
            if let Ok(fr) = self.frame(s, f) {
                for fi in fields::frame_fields(fr).iter().filter(|f| f.position == Position::Node) {
                    for c in &fi.components {
                        let (v, _, _) = fields::node_values(fr, &self.mesh, m.nodes.len(), fi, c);
                        if v[n].is_finite() {
                            values.push((if c == &fi.name { c.clone() } else { format!("{} {c}", fi.name).replace("U U", "U").replace("RF RF", "RF").replace("UR UR", "UR") }, v[n] as f64, self.units.field_unit(&fi.name)));
                        }
                    }
                }
            }
        }
        Ok(PickInfo {
            kind: "node".into(),
            name: m.node_name(n),
            index: n as u32,
            details: vec![(
                "Koordinater".into(),
                format!("{:.6e}, {:.6e}, {:.6e}{}", p[0], p[1], p[2], Some(self.units.symbols().length).filter(|l| !l.is_empty()).map(|l| format!(" {l}")).unwrap_or_default()),
            )],
            values,
        })
    }
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct FieldStats {
    pub field: String,
    pub component: String,
    /// Fältets enhet i modellens enhetssystem ("" om okänd eller dimensionslös).
    pub unit: String,
    pub min: f64,
    pub max: f64,
    pub min_at: Option<String>,
    pub max_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Type)]
pub struct PickInfo {
    pub kind: String,
    pub name: String,
    pub index: u32,
    pub details: Vec<(String, String)>,
    /// (namn, värde, enhet)
    pub values: Vec<(String, f64, String)>,
}

/// Händelser under en körning.
#[derive(Debug, Clone, Serialize, Type)]
#[serde(tag = "type")]
pub enum RunEvent {
    Started { solver: String, note: String, steps: u32 },
    Step { index: u32, name: String, procedure: String },
    Increment { fraction: f64, iterations: u32 },
    Iteration { iteration: u32, residual: f64 },
    Finished { ok: bool, cancelled: bool, seconds: f64, message: String },
}

/// Kör analysen på den aktuella tråden. `on` får framstegshändelser
/// (iterationer högst ungefär 20 per sekund); `cancel` avbryter mellan
/// iterationerna.
pub fn run_analysis(model: &FeModel, solver: Option<String>, all_frames: bool, cancel: Arc<AtomicBool>, mut on: impl FnMut(RunEvent) + 'static) -> Res<AnalysisOutput> {
    let opts = RunOptions { all_frames, verbose: false, solver, settings: None };
    let mut last = Instant::now();
    let hook = move |e: &Event| {
        match *e {
            Event::Step { index, name, procedure } => on(RunEvent::Step { index: index as u32, name: name.into(), procedure: procedure.into() }),
            Event::Increment { fraction, iterations } => on(RunEvent::Increment { fraction, iterations: iterations as u32 }),
            Event::Iteration { iteration, residual } => {
                if last.elapsed().as_millis() >= 50 {
                    last = Instant::now();
                    on(RunEvent::Iteration { iteration: iteration as u32, residual });
                }
            }
        }
        !cancel.load(Ordering::Relaxed)
    };
    progress::with_hook(hook, || steps::run(model, &opts)).map(|(o, _)| o).map_err(|e| e.0)
}

#[cfg(test)]
mod tests;
