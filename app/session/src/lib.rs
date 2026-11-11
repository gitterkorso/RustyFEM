//! App API for RustyFEM (see PLAN.md, Phase 1).
//!
//! A `Session` is an open model: the source (`.rusty` script or `.inp`), the
//! built `FeModel`, undo/redo, display geometry and the latest results.
//! The run (`run_analysis`) is done by the caller on its own thread, with progress
//! and cancellation via `rustyfem::progress`. Nothing here changes the engine's results.

pub mod edit;
pub mod fields;
pub mod info;
pub mod memory;
pub mod mesh;
pub mod platform;
pub mod profiles;
pub mod project;
pub mod units;
pub mod visual;

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

/// The model's source.
#[derive(Debug, Clone, Serialize, Type)]
pub struct SessionInfo {
    pub kind: String,
    pub path: Option<String>,
    pub name: String,
    pub units: UnitSystem,
    /// Unit symbols for `units` (empty when no unit system is specified).
    pub unit_symbols: UnitSymbols,
    /// Current `.rusty` script (only for `.rusty` sessions).
    pub script: Option<String>,
    /// The script's dimension (2 or 3; only for `.rusty` sessions). Also applies
    /// to an empty model, where `FeModel` has no elements to determine it from.
    pub script_dim: Option<u32>,
    /// Why the model cannot be solved yet (e.g. no elements, or a
    /// support that hits no nodes). `None` when the model is complete.
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
    /// Original source text (for `.rusty`, the current text after edits).
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
    /// Visualization data: the model split into parts/members/bodies (per model),
    /// and recovered nodal stresses (per results, step and frame).
    visual: Option<(usize, Arc<visual::VisualModel>)>,
    recovered: Option<((usize, usize, usize, usize), Arc<visual::Recovered>)>,
}

/// Template for a new model.
pub const NEW_MODEL: &str = "# New model – the tools in the app add lines here.\n\
model 2d\n\
material steel E=210e9 nu=0.3\n\
plate x=0..2 y=0..0.2 t=0.01 size=0.05\n\
fix x=0\n\
load x=2 fy=-1000\n";


/// Empty model to build from scratch.
pub const EMPTY_2D: &str = "# New model – the tools in the app add lines here.\nmodel 2d\n";
pub const EMPTY_3D: &str = "# New model – the tools in the app add lines here.\nmodel 3d\n";

/// A `FeModel` built from a `.rusty` script.
struct Built {
    model: FeModel,
    dim: u32,
    /// Why the model cannot be solved yet.
    problem: Option<String>,
}

/// Builds a `FeModel` from a `.rusty` script. Errors in the script (syntax,
/// unknown commands, material without E) are rejected. A model that is valid but
/// incomplete – no elements yet, or supports and loads that hit no
/// nodes – is built without steps, so it can be built line by line.
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
        // .rusty scripts use SI (the default materials are in Pa); an .inp is guessed from its materials
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
            visual: None,
            recovered: None,
        }
    }

    /// New `.rusty` model from text.
    pub fn new_fem(text: &str) -> Res<Session> {
        Ok(Session::from_script(None, text.to_string(), build_script(text)?))
    }

    /// `.inp` text; relative `*INCLUDE` paths are resolved from `dir`.
    pub fn from_inp_text(text: &str, dir: &Path) -> Res<Session> {
        let r = rustyfem::inp::read_str(text, dir).map_err(|e| e.0)?;
        Ok(Session::from_model(SourceKind::Inp, None, text.to_string(), Vec::new(), r.model))
    }

    /// Opens `.inp`, `.rusty` or `.rustyproj`.
    pub fn open(path: &Path) -> Res<Session> {
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let read = |p: &Path| std::fs::read(p).map_err(|e| format!("could not read {}: {e}", p.display()));
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
                // *INCLUDE is read from the project file via a file resolver (also works in the iOS sandbox)
                let root = Path::new("project");
                let files = &p.included;
                let resolve = |path: &Path| -> rustyfem::Result<Vec<u8>> {
                    let rel = path.strip_prefix(root).unwrap_or(path).to_string_lossy().replace('\\', "/");
                    files
                        .iter()
                        .find(|(n, _)| *n == rel)
                        .map(|(_, b)| b.clone())
                        .ok_or_else(|| rustyfem::FemError(format!("the project file does not contain the included file {rel}")))
                };
                let r = rustyfem::inp::read_str_with(&p.text, root, &resolve).map_err(|e| e.0)?;
                Session::from_model(SourceKind::Inp, None, p.text.clone(), p.included.clone(), r.model)
            }
        };
        for e in &p.edits {
            s.apply(e.clone())?;
        }
        s.path = path;
        // older project files always saved "SI (m, N, Pa)"; then the default choice above applies
        if p.units != units::LEGACY_LABEL {
            s.units = UnitSystem::parse(&p.units);
        }
        s.results = p.results.map(Arc::new);
        s.results_stale = false;
        s.dirty = false;
        Ok(s)
    }

    /// Contents of the project file.
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

    /// Why the model cannot be solved yet (`None` when it is complete).
    pub fn problem(&self) -> Option<&str> {
        self.problem.as_deref()
    }

    pub fn info(&self) -> SessionInfo {
        let name = self.path.as_ref().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "Untitled".into());
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

    /// Performs an edit. Returns `true` if the geometry changed.
    fn exec(&mut self, e: &Edit) -> Res<(Edit, bool)> {
        match e {
            Edit::FemText { text } => {
                if self.kind != SourceKind::Rusty {
                    return Err("script text can only be changed in a .rusty model".into());
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

    /// Performs an edit and adds it to the undo history.
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

    /// Changes the unit system. Only the labels change; no values are converted.
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

    /// The results catalog with the fields' units (`None` without results).
    pub fn catalog(&self) -> Option<Vec<StepResultInfo>> {
        let mut cat = fields::catalog(self.results.as_ref()?);
        for f in cat.iter_mut().flat_map(|s| s.fields.iter_mut()) {
            f.unit = self.units.field_unit(&f.name);
        }
        Some(cat)
    }

    fn frame(&self, step: usize, frame: usize) -> Res<&rustyfem::fem::output::FrameData> {
        let r = self.results.as_ref().ok_or("no results – solve the model first")?;
        r.steps.get(step).and_then(|s| s.frames.get(frame)).ok_or_else(|| format!("step {} frame {} does not exist", step + 1, frame))
    }

    /// Nodal values for a field: `f32` per node, followed by min and max (f32).
    pub fn field_values(&self, step: usize, frame: usize, field: &str, comp: &str) -> Res<Vec<u8>> {
        let fr = self.frame(step, frame)?;
        let fi = fields::frame_fields(fr).into_iter().find(|f| f.name == field).ok_or_else(|| format!("field {field} does not exist in the frame"))?;
        let (v, lo, hi) = fields::node_values(fr, &self.mesh, self.model.nodes.len(), &fi, comp);
        let mut b = Vec::with_capacity(v.len() * 4 + 8);
        for x in v {
            b.extend(x.to_le_bytes());
        }
        b.extend((lo as f32).to_le_bytes());
        b.extend((hi as f32).to_le_bytes());
        Ok(b)
    }

    /// The model for the Visualization tab (cached per model).
    pub fn visual_model(&mut self) -> Arc<visual::VisualModel> {
        let key = Arc::as_ptr(&self.model) as usize;
        if let Some((k, v)) = &self.visual {
            if *k == key {
                return v.clone();
            }
        }
        let v = Arc::new(visual::model(&self.model));
        self.visual = Some((key, v.clone()));
        v
    }

    /// Member section forces, supports with reactions and rotations in a frame.
    pub fn visual_frame(&mut self, step: usize, frame: usize) -> Res<visual::VisualFrame> {
        let vm = self.visual_model();
        let fr = self.frame(step, frame)?;
        Ok(visual::frame(&self.model, &vm, fr))
    }

    /// Shell section forces (element means) and the local directions.
    pub fn visual_shell(&self, step: usize, frame: usize, element: usize) -> Res<visual::ShellForces> {
        if element >= self.model.elements.len() {
            return Err("the element does not exist".into());
        }
        Ok(visual::shell(&self.model, self.frame(step, frame)?, element))
    }

    fn recovered(&mut self, step: usize, frame: usize) -> Res<(Arc<visual::VisualModel>, Arc<visual::Recovered>)> {
        let vm = self.visual_model();
        let r = self.results.as_ref().ok_or("no results – solve the model first")?;
        let key = (Arc::as_ptr(r) as usize, Arc::as_ptr(&self.model) as usize, step, frame);
        if let Some((k, v)) = &self.recovered {
            if *k == key {
                return Ok((vm, v.clone()));
            }
        }
        // NLGEOM is inherited from the previous steps
        let nlgeom = self.model.steps.iter().take(step + 1).filter_map(|s| s.nlgeom).last().unwrap_or(false);
        let rec = Arc::new(visual::recover(&self.model, &vm, self.frame(step, frame)?, nlgeom));
        self.recovered = Some((key, rec.clone()));
        Ok((vm, rec))
    }

    /// Section cut of a solid or planar body at `s` along `axis` (from the body's centre).
    pub fn visual_cut(&mut self, step: usize, frame: usize, body: usize, axis: [f64; 3], s: f64) -> Res<visual::CutResult> {
        let (vm, rec) = self.recovered(step, frame)?;
        let b = vm.bodies.get(body).ok_or("the body does not exist")?;
        let l = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let a = if l > 0.0 { [axis[0] / l, axis[1] / l, axis[2] / l] } else { b.axis };
        let p = [b.center[0] + s * a[0], b.center[1] + s * a[1], b.center[2] + s * a[2]];
        let mut r = visual::cut(&self.model, &vm, &rec, body, p, a, true);
        r.s = s;
        Ok(r)
    }

    /// Section force diagrams of a solid or planar body along `axis`.
    pub fn visual_diagram(&mut self, step: usize, frame: usize, body: usize, axis: [f64; 3], n: usize) -> Res<visual::BodyDiagram> {
        let (vm, rec) = self.recovered(step, frame)?;
        if body >= vm.bodies.len() {
            return Err("the body does not exist".into());
        }
        Ok(visual::diagram(&self.model, &vm, &rec, body, axis, n))
    }

    /// Displacements (f32 ×3 per node).
    pub fn displacement(&self, step: usize, frame: usize) -> Res<Vec<u8>> {
        let fr = self.frame(step, frame)?;
        Ok(fields::displacement(fr, self.model.nodes.len()).into_iter().flat_map(|x| x.to_le_bytes()).collect())
    }

    /// Min and max of a field (nodal values, element fields averaged) with the nodes where they occur.
    pub fn field_stats(&self, step: usize, frame: usize, field: &str, comp: &str) -> Res<FieldStats> {
        let fr = self.frame(step, frame)?;
        let fi = fields::frame_fields(fr).into_iter().find(|f| f.name == field).ok_or_else(|| format!("field {field} does not exist in the frame"))?;
        let comp = if comp.is_empty() { fi.components[0].clone() } else { comp.to_string() };
        if !fi.components.contains(&comp) {
            return Err(format!("{field} has the components {}", fi.components.join(", ")));
        }
        let (v, lo, hi) = fields::node_values(fr, &self.mesh, self.model.nodes.len(), &fi, &comp);
        let at = |x: f64| v.iter().position(|&y| y.is_finite() && (y as f64 - x).abs() <= 1e-6 * x.abs().max(1e-30)).map(|n| self.model.node_name(n));
        Ok(FieldStats { field: field.into(), unit: self.units.field_unit(field), component: comp, min: lo, max: hi, min_at: at(lo), max_at: at(hi) })
    }

    /// Node or element index from "INSTANCE.number" or just "number".
    pub fn lookup(&self, what: &str, element: bool) -> Res<usize> {
        let m = &self.model;
        let (inst, label) = match what.rsplit_once('.') {
            Some((i, l)) => (m.instance_index(&i.to_uppercase()).ok_or_else(|| format!("instance {i} does not exist"))?, l),
            None => (m.instance_index(rustyfem::fem::model::FLAT_INSTANCE).or_else(|| m.instance_index(rustyfem::fem::model::ASSEMBLY_INSTANCE)).unwrap_or(0), what),
        };
        let label: i64 = label.trim().parse().map_err(|_| format!("'{what}' is not a node/element number"))?;
        let found = if element { m.element(inst, label) } else { m.node(inst, label) };
        found.ok_or_else(|| format!("{} {what} does not exist", if element { "element" } else { "node" }))
    }

    pub fn history(&self, step: usize, index: usize) -> Res<Vec<(f64, f64)>> {
        let r = self.results.as_ref().ok_or("no results")?;
        r.steps.get(step).and_then(|s| s.history.get(index)).map(|h| h.2.clone()).ok_or_else(|| "history output does not exist".to_string())
    }

    /// Information about an element (and its results in a frame).
    pub fn element_info(&self, e: usize, at: Option<(usize, usize)>) -> Res<PickInfo> {
        let m = &self.model;
        let el = m.elements.get(e).ok_or("the element does not exist")?;
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
                ("Type".into(), el.etype.name().into()),
                ("Nodes".into(), el.nodes.iter().take(el.etype.n_connectivity()).map(|&n| m.node_label[n].to_string()).collect::<Vec<_>>().join(", ")),
                ("Section".into(), section.map(|s| s.elset.clone()).unwrap_or_else(|| "–".into())),
                ("Material".into(), section.map(|s| s.material.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| "–".into())),
            ],
            values,
        })
    }

    /// Information about a node.
    pub fn node_info(&self, n: usize, at: Option<(usize, usize)>) -> Res<PickInfo> {
        let m = &self.model;
        let p = m.nodes.get(n).ok_or("the node does not exist")?;
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
                "Coordinates".into(),
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
    /// The field's unit in the model's unit system ("" if unknown or dimensionless).
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
    /// (name, value, unit)
    pub values: Vec<(String, f64, String)>,
}

/// Events during a run.
#[derive(Debug, Clone, Serialize, Type)]
#[serde(tag = "type")]
pub enum RunEvent {
    Started { solver: String, note: String, steps: u32 },
    Step { index: u32, name: String, procedure: String },
    Increment { fraction: f64, iterations: u32 },
    Iteration { iteration: u32, residual: f64 },
    Finished { ok: bool, cancelled: bool, seconds: f64, message: String },
}

/// Runs the analysis on the current thread. `on` receives progress events
/// (iterations at most about 20 per second); `cancel` aborts between
/// iterations.
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
