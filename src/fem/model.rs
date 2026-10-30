//! Model: nodes, elements, sets, materials, sections and steps.
//!
//! `FeModel` is the common format that the .inp reader builds and that
//! the step framework analyzes. `.rusty` models (`crate::model::Model`) are also
//! translated to a `FeModel` before they are solved (see `crate::legacy`).
//!
//! Names (sets, materials, instances) are stored in upper case (case-insensitive
//! as in the .inp format). Nodes and elements are identified externally by (instance, number);
//! internally by their index in `nodes` and `elements`, respectively.

use crate::elements::ElementType;
use crate::fem::dof::Dof;
use crate::model::{fail, FemError, Result};
use crate::section::BeamSection;
use std::collections::{BTreeMap, HashMap};

/// Instance name for a model without *PART/*ASSEMBLY.
pub const FLAT_INSTANCE: &str = "MODEL";
/// Instance name for nodes and elements defined directly in *ASSEMBLY.
pub const ASSEMBLY_INSTANCE: &str = "ASSEMBLY";

/// A data line in a keyword block.
#[derive(Debug, Clone, Default)]
pub struct DataLine {
    /// The fields between the commas, trimmed (empty fields are kept as "").
    pub fields: Vec<String>,
    /// Line number in the file (1-based).
    pub line: usize,
    /// The line ended with a comma (continues on the next line for e.g. *ELEMENT).
    pub continued: bool,
}

/// A keyword with parameters and data lines, as written in the .inp file.
#[derive(Debug, Clone, Default)]
pub struct RawKeyword {
    /// The keyword in upper case with single spaces, e.g. "SOLID SECTION".
    pub name: String,
    /// (PARAMETER in upper case, value as written or "" if there is no value).
    pub params: Vec<(String, String)>,
    pub data: Vec<DataLine>,
    pub file: String,
    pub line: usize,
}

/// Parses a number in the .inp format ("1.", "2.5e3", "1.0D+03"; empty field = 0).
pub fn parse_f64(s: &str) -> Option<f64> {
    let t = s.trim();
    if t.is_empty() {
        return Some(0.0);
    }
    t.parse::<f64>().ok().or_else(|| t.replace(['D', 'd'], "e").parse::<f64>().ok())
}

impl RawKeyword {
    pub fn param(&self, key: &str) -> Option<&str> {
        self.params.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// Parameter value in upper case (names, YES/NO etc.).
    pub fn param_upper(&self, key: &str) -> Option<String> {
        self.param(key).map(|v| v.trim().to_uppercase())
    }

    pub fn has(&self, key: &str) -> bool {
        self.params.iter().any(|(k, _)| k == key)
    }

    /// YES/NO parameter; `default` if it is missing, YES if it is given without a value.
    pub fn flag(&self, key: &str, default: bool) -> Result<bool> {
        match self.param_upper(key) {
            None => Ok(default),
            Some(v) if v.is_empty() || v == "YES" || v == "ON" => Ok(true),
            Some(v) if v == "NO" || v == "OFF" => Ok(false),
            Some(v) => self.err(format!("{key}={v} must be YES or NO")),
        }
    }

    /// "file.inp line 12 (*SOLID SECTION)" for error messages.
    pub fn location(&self) -> String {
        format!("{} line {} (*{})", self.file, self.line, self.name)
    }

    pub fn err<T>(&self, msg: impl AsRef<str>) -> Result<T> {
        fail(format!("{}: {}", self.location(), msg.as_ref()))
    }

    pub fn err_at<T>(&self, line: usize, msg: impl AsRef<str>) -> Result<T> {
        fail(format!("{} line {} (*{}): {}", self.file, line, self.name, msg.as_ref()))
    }

    /// Data lines where lines ending with a comma are joined with the next one.
    pub fn records(&self) -> Vec<DataLine> {
        let mut out: Vec<DataLine> = Vec::new();
        let mut open = false;
        for d in &self.data {
            if open {
                let last = out.last_mut().unwrap();
                last.fields.extend(d.fields.iter().cloned());
                last.continued = d.continued;
            } else {
                out.push(d.clone());
            }
            open = d.continued;
        }
        out
    }

    /// All numbers on a data line (empty fields become 0).
    pub fn floats(&self, d: &DataLine) -> Result<Vec<f64>> {
        d.fields
            .iter()
            .map(|f| parse_f64(f).ok_or_else(|| FemError(format!("{} line {} (*{}): '{f}' is not a number", self.file, d.line, self.name))))
            .collect()
    }

    /// The numbers on the first data line (empty list if there is no data line).
    pub fn first_floats(&self) -> Result<Vec<f64>> {
        match self.data.first() {
            Some(d) => self.floats(d),
            None => Ok(Vec::new()),
        }
    }
}

/// Material: name, behaviors (*ELASTIC, *PLASTIC, ...) and density.
///
/// The behaviors are stored as written in the file; the material registry (`crate::materials`)
/// selects a material model that consumes them.
#[derive(Debug, Clone, Default)]
pub struct MaterialDef {
    pub name: String,
    pub behaviors: Vec<RawKeyword>,
    pub density: f64,
}

impl MaterialDef {
    pub fn behavior(&self, keyword: &str) -> Option<&RawKeyword> {
        self.behaviors.iter().find(|b| b.name == keyword)
    }

    /// Linear isotropic elastic material (used by .rusty models and tests).
    pub fn elastic(name: &str, e: f64, nu: f64, density: f64) -> Self {
        let kw = RawKeyword {
            name: "ELASTIC".into(),
            data: vec![DataLine { fields: vec![format!("{e:e}"), format!("{nu:e}")], line: 0, continued: false }],
            file: "(internal)".into(),
            ..Default::default()
        };
        MaterialDef { name: name.to_uppercase(), behaviors: vec![kw], density }
    }
}

/// Beam section from *BEAM SECTION or *BEAM GENERAL SECTION.
#[derive(Debug, Clone)]
pub struct BeamSectionDef {
    /// Section shape: RECT, CIRC, PIPE, I, GENERAL, ...
    pub shape: String,
    pub profile: BeamSection,
    pub i12: f64,
    /// Approximate direction of the section's 1-axis (n1). `None` = the default direction.
    pub n1: Option<[f64; 3]>,
    /// (E, G) from *BEAM GENERAL SECTION; otherwise from the material.
    pub moduli: Option<(f64, f64)>,
    pub density: f64,
}

#[derive(Debug, Clone)]
pub enum SectionKind {
    /// *SOLID SECTION: thickness in 2D, area for trusses, no data in 3D.
    Solid { thickness: Option<f64> },
    Beam(BeamSectionDef),
    /// *SHELL SECTION: already stored here, shell elements are added in `elements/shell.rs`.
    Shell { thickness: f64, n_int: usize },
    /// Other sections (e.g. *MEMBRANE SECTION); the data is in `raw`.
    Other,
}

#[derive(Debug, Clone)]
pub struct SectionDef {
    /// Element set the section applies to (upper case, with instance prefix "INST.SET").
    pub elset: String,
    /// Material name (upper case) or empty.
    pub material: String,
    /// Material index, set by `FeModel::finalize`.
    pub material_index: Option<usize>,
    pub kind: SectionKind,
    /// The original keyword (for parameters that only some elements need).
    pub raw: RawKeyword,
}

impl SectionDef {
    /// Thickness (2D), area (trusses) or 1.
    pub fn solid_thickness(&self) -> f64 {
        match &self.kind {
            SectionKind::Solid { thickness: Some(t) } if *t > 0.0 => *t,
            SectionKind::Shell { thickness, .. } => *thickness,
            _ => 1.0,
        }
    }

    pub fn beam(&self) -> Option<&BeamSectionDef> {
        match &self.kind {
            SectionKind::Beam(b) => Some(b),
            _ => None,
        }
    }
}

/// A *BOUNDARY block (OP=NEW clears earlier boundary conditions).
#[derive(Debug, Clone, Default)]
pub struct BcBlock {
    pub op_new: bool,
    /// DISPLACEMENT (default), VELOCITY, ACCELERATION.
    pub kind: String,
    pub amplitude: Option<String>,
    /// (node, degree of freedom, value). Degrees of freedom the node does not have are ignored.
    pub items: Vec<(usize, Dof, f64)>,
    pub source: String,
}

/// A *CLOAD block. Several lines for the same node and degree of freedom are summed
/// (checked).
#[derive(Debug, Clone, Default)]
pub struct CloadBlock {
    pub op_new: bool,
    pub amplitude: Option<String>,
    pub follower: bool,
    pub items: Vec<(usize, Dof, f64)>,
    pub source: String,
}

/// A distributed load on an element (*DLOAD): type (P1, P, GRAV, BX, PY, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct DistLoad {
    pub elem: usize,
    pub kind: String,
    pub magnitude: f64,
    /// Other values on the data line (e.g. the direction for GRAV).
    pub extra: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct DloadBlock {
    pub op_new: bool,
    pub amplitude: Option<String>,
    pub items: Vec<DistLoad>,
    pub source: String,
}

/// A *STEP ... *END STEP.
#[derive(Debug, Clone, Default)]
pub struct StepDef {
    pub name: String,
    /// The *STEP line with all parameters.
    pub header: RawKeyword,
    /// The procedure keyword (*STATIC, *FREQUENCY, ...).
    pub procedure: Option<RawKeyword>,
    /// NLGEOM=YES/NO if given (otherwise the previous step's value is inherited).
    pub nlgeom: Option<bool>,
    pub perturbation: bool,
    /// INC= (maximum number of increments), default 100.
    pub max_increments: usize,
    /// AMPLITUDE=STEP on *STEP (default RAMP for statics).
    pub amplitude_step: bool,
    pub bcs: Vec<BcBlock>,
    pub cloads: Vec<CloadBlock>,
    pub dloads: Vec<DloadBlock>,
    /// Other keywords in the step that a registered module has stored.
    pub extra: Vec<RawKeyword>,
    /// Output requests (*OUTPUT, *NODE OUTPUT, ...). RustyFEM always writes its fields.
    pub output: Vec<RawKeyword>,
}

impl StepDef {
    /// The procedure name in upper case ("STATIC"), or "" if there is none.
    pub fn procedure_name(&self) -> &str {
        self.procedure.as_ref().map(|p| p.name.as_str()).unwrap_or("")
    }
}

/// A surface (*SURFACE): element faces or nodes.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    /// ELEMENT (default) or NODE.
    pub kind: String,
    /// (element, face label) with the .inp-format labels: "S1".."S6" for
    /// continuum, "SPOS"/"SNEG" for shells, "S1" etc. for 2D edges.
    pub faces: Vec<(usize, String)>,
    /// (node, area or weight) for node-based surfaces (weight 1 if missing).
    pub nodes: Vec<(usize, f64)>,
    pub source: String,
}

/// An element in the model.
#[derive(Clone)]
pub struct Element {
    pub etype: &'static dyn ElementType,
    pub label: i64,
    pub inst: u32,
    pub nodes: Vec<usize>,
    /// Section, set by `finalize` from the element sets of the sections.
    pub section: Option<usize>,
}

impl std::fmt::Debug for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {:?}", self.etype.name(), self.label, self.nodes)
    }
}

#[derive(Debug, Clone, Default)]
pub struct FeModel {
    pub heading: String,
    pub nodes: Vec<[f64; 3]>,
    pub node_inst: Vec<u32>,
    pub node_label: Vec<i64>,
    /// Instance names (upper case). Index in `node_inst`/`Element::inst`.
    pub instances: Vec<String>,
    node_lookup: HashMap<(u32, i64), usize>,
    pub elements: Vec<Element>,
    elem_lookup: HashMap<(u32, i64), usize>,
    pub nsets: BTreeMap<String, Vec<usize>>,
    pub elsets: BTreeMap<String, Vec<usize>>,
    /// Surfaces (*SURFACE), names in upper case (with instance prefix "INST.SURF").
    pub surfaces: BTreeMap<String, Surface>,
    pub materials: Vec<MaterialDef>,
    pub sections: Vec<SectionDef>,
    /// Boundary conditions in model data (before the first step).
    pub initial_bcs: Vec<BcBlock>,
    pub steps: Vec<StepDef>,
    /// Other model data that a registered module has stored (e.g. *AMPLITUDE,
    /// *SURFACE, *TIE). Read by the module that owns the keyword.
    pub extra: Vec<RawKeyword>,
    pub warnings: Vec<String>,
}

/// Rotation about the axis through `a` and `b` by the angle `deg` (degrees), Rodrigues' formula.
pub fn rotate_about(p: [f64; 3], a: [f64; 3], b: [f64; 3], deg: f64) -> [f64; 3] {
    let k = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let l = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
    if l == 0.0 || deg == 0.0 {
        return p;
    }
    let k = [k[0] / l, k[1] / l, k[2] / l];
    let v = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
    let r = rotate_vec(v, k, deg);
    [r[0] + a[0], r[1] + a[1], r[2] + a[2]]
}

/// Rotates the vector `v` about the unit axis `k` by `deg` degrees.
pub fn rotate_vec(v: [f64; 3], k: [f64; 3], deg: f64) -> [f64; 3] {
    let th = deg.to_radians();
    let (c, s) = (th.cos(), th.sin());
    let kxv = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
    let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    [0, 1, 2].map(|i| v[i] * c + kxv[i] * s + k[i] * kv * (1.0 - c))
}

/// Placement of an instance: first translation, then rotation.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub translation: [f64; 3],
    /// (point a, point b, angle in degrees)
    pub rotation: Option<([f64; 3], [f64; 3], f64)>,
}

impl Placement {
    pub fn point(&self, p: [f64; 3]) -> [f64; 3] {
        let q = [p[0] + self.translation[0], p[1] + self.translation[1], p[2] + self.translation[2]];
        match self.rotation {
            Some((a, b, deg)) => rotate_about(q, a, b, deg),
            None => q,
        }
    }

    pub fn direction(&self, v: [f64; 3]) -> [f64; 3] {
        match self.rotation {
            Some((a, b, deg)) => rotate_about(v, [0.0; 3], [b[0] - a[0], b[1] - a[1], b[2] - a[2]], deg),
            None => v,
        }
    }
}

impl FeModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index of an instance; created if it does not exist.
    pub fn instance(&mut self, name: &str) -> u32 {
        let up = name.to_uppercase();
        if let Some(i) = self.instances.iter().position(|n| *n == up) {
            return i as u32;
        }
        self.instances.push(up);
        (self.instances.len() - 1) as u32
    }

    pub fn instance_index(&self, name: &str) -> Option<u32> {
        let up = name.to_uppercase();
        self.instances.iter().position(|n| *n == up).map(|i| i as u32)
    }

    /// Adds (or moves, for the same number) a node.
    pub fn add_node(&mut self, inst: u32, label: i64, x: [f64; 3]) -> usize {
        if let Some(&i) = self.node_lookup.get(&(inst, label)) {
            self.nodes[i] = x;
            return i;
        }
        let i = self.nodes.len();
        self.nodes.push(x);
        self.node_inst.push(inst);
        self.node_label.push(label);
        self.node_lookup.insert((inst, label), i);
        i
    }

    pub fn node(&self, inst: u32, label: i64) -> Option<usize> {
        self.node_lookup.get(&(inst, label)).copied()
    }

    pub fn add_element(&mut self, etype: &'static dyn ElementType, inst: u32, label: i64, nodes: Vec<usize>) -> Result<usize> {
        if self.elem_lookup.contains_key(&(inst, label)) {
            return fail(format!("element {} is defined twice", label));
        }
        let i = self.elements.len();
        self.elements.push(Element { etype, label, inst, nodes, section: None });
        self.elem_lookup.insert((inst, label), i);
        Ok(i)
    }

    pub fn element(&self, inst: u32, label: i64) -> Option<usize> {
        self.elem_lookup.get(&(inst, label)).copied()
    }

    /// "MODEL.5" for error messages.
    pub fn node_name(&self, n: usize) -> String {
        format!("{}.{}", self.instances[self.node_inst[n] as usize], self.node_label[n])
    }

    pub fn elem_name(&self, e: usize) -> String {
        let el = &self.elements[e];
        format!("{}.{} ({})", self.instances[el.inst as usize], el.label, el.etype.name())
    }

    pub fn material_index(&self, name: &str) -> Option<usize> {
        let up = name.to_uppercase();
        self.materials.iter().position(|m| m.name == up)
    }

    /// Spatial dimension: 2 if all elements are planar, otherwise 3.
    pub fn dim(&self) -> usize {
        self.elements.iter().map(|e| e.etype.dim()).max().unwrap_or(3)
    }

    /// Adds nodes to a node set (created if it does not exist).
    pub fn add_to_nset(&mut self, name: &str, nodes: &[usize]) {
        let set = self.nsets.entry(name.to_uppercase()).or_default();
        set.extend_from_slice(nodes);
        set.sort_unstable();
        set.dedup();
    }

    pub fn add_to_elset(&mut self, name: &str, elems: &[usize]) {
        let set = self.elsets.entry(name.to_uppercase()).or_default();
        set.extend_from_slice(elems);
        set.sort_unstable();
        set.dedup();
    }

    /// Copies a part (a `FeModel` with only geometry, sets and sections) in
    /// as the instance `name`. Sets and sections get the prefix "NAME.".
    pub fn instantiate(&mut self, part: &FeModel, name: &str, place: Placement) -> Result<()> {
        let inst = self.instance(name);
        let prefix = self.instances[inst as usize].clone();
        let mut nmap = vec![0usize; part.nodes.len()];
        for (i, &x) in part.nodes.iter().enumerate() {
            nmap[i] = self.add_node(inst, part.node_label[i], place.point(x));
        }
        let mut emap = vec![0usize; part.elements.len()];
        for (i, e) in part.elements.iter().enumerate() {
            let nodes = e.nodes.iter().map(|&n| nmap[n]).collect();
            emap[i] = self
                .add_element(e.etype, inst, e.label, nodes)
                .map_err(|err| FemError(format!("instance {prefix}: {err}")))?;
        }
        for (s, v) in &part.nsets {
            let mapped: Vec<usize> = v.iter().map(|&n| nmap[n]).collect();
            self.add_to_nset(&format!("{prefix}.{s}"), &mapped);
        }
        for (s, v) in &part.elsets {
            let mapped: Vec<usize> = v.iter().map(|&e| emap[e]).collect();
            self.add_to_elset(&format!("{prefix}.{s}"), &mapped);
        }
        for (name, sf) in &part.surfaces {
            let mut s2 = sf.clone();
            s2.faces = sf.faces.iter().map(|(e, f)| (emap[*e], f.clone())).collect();
            s2.nodes = sf.nodes.iter().map(|(n, w)| (nmap[*n], *w)).collect();
            self.surfaces.insert(format!("{prefix}.{name}"), s2);
        }
        for sec in &part.sections {
            let mut s = sec.clone();
            s.elset = format!("{prefix}.{}", sec.elset);
            if let SectionKind::Beam(b) = &mut s.kind {
                if let Some(n1) = b.n1 {
                    b.n1 = Some(place.direction(n1));
                }
            }
            self.sections.push(s);
        }
        self.extra.extend(part.extra.iter().cloned());
        self.warnings.extend(part.warnings.iter().cloned());
        Ok(())
    }

    /// Connects sections to elements and materials. Called when the whole file has been read.
    pub fn finalize(&mut self) -> Result<()> {
        for s in self.sections.iter_mut() {
            if s.material.is_empty() {
                s.material_index = None;
                continue;
            }
            let up = s.material.to_uppercase();
            match self.materials.iter().position(|m| m.name == up) {
                Some(i) => s.material_index = Some(i),
                None => return s.raw.err(format!("material '{}' is not defined", s.material)),
            }
        }
        for e in self.elements.iter_mut() {
            e.section = None;
        }
        for (si, s) in self.sections.iter().enumerate() {
            let Some(set) = self.elsets.get(&s.elset) else {
                return s.raw.err(format!("element set '{}' does not exist", s.elset));
            };
            for &e in set {
                if let Some(old) = self.elements[e].section {
                    if old != si {
                        let el = &self.elements[e];
                        return s.raw.err(format!(
                            "element {}.{} already has a section ({})",
                            self.instances[el.inst as usize], el.label, self.sections[old].raw.location()
                        ));
                    }
                }
                self.elements[e].section = Some(si);
            }
        }
        // internal nodes (e.g. the pressure variables of hybrid elements): negative numbers
        // −(16·element number + k + 1) in the element's instance, at the centroid
        for e in 0..self.elements.len() {
            let ni = self.elements[e].etype.internal_nodes();
            if ni == 0 || self.elements[e].nodes.len() != self.elements[e].etype.n_connectivity() {
                continue;
            }
            let (inst, label) = (self.elements[e].inst, self.elements[e].label);
            let nc = self.elements[e].nodes.len() as f64;
            let mut c = [0.0; 3];
            for &n in &self.elements[e].nodes {
                for k in 0..3 {
                    c[k] += self.nodes[n][k] / nc;
                }
            }
            for k in 0..ni {
                let id = self.add_node(inst, -(16 * label + k as i64 + 1), c);
                self.elements[e].nodes.push(id);
            }
        }
        // nodes given in the section (e.g. reference node for generalized plane strain)
        for e in 0..self.elements.len() {
            let el = &self.elements[e];
            let (nc, nn) = (el.etype.n_connectivity(), el.etype.n_nodes());
            if nc == nn || el.nodes.len() != nc {
                continue;
            }
            let Some(si) = el.section else { continue };
            let mut extra = Vec::new();
            for l in el.etype.section_nodes(&self.sections[si]).or_else(|err| self.sections[si].raw.err(err.0))? {
                match self.node(el.inst, l) {
                    Some(n) => extra.push(n),
                    None => return self.sections[si].raw.err(format!("node {l} does not exist")),
                }
            }
            self.elements[e].nodes.extend(extra);
        }
        // acoustics2 area: shared internal nodes for acoustic infinite elements
        crate::elements::acoustic_infinite::add_nodes(self)?;
        // acoustics2 area: *EOS COMPACTION – the density of the porous material
        crate::materials::eos::compaction::adjust_density(self)?;
        Ok(())
    }
}
