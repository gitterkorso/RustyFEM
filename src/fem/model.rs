//! Modell: noder, element, set, material, tvärsnitt och steps.
//!
//! `FeModel` är det gemensamma formatet som .inp-läsaren bygger och som
//! step-ramverket analyserar. Även `.rusty`-modeller (`crate::model::Model`)
//! översätts till en `FeModel` innan de löses (se `crate::legacy`).
//!
//! Namn (set, material, instanser) lagras med versaler (skiftlägesokänsliga
//! som i .inp-formatet). Noder och element identifieras utåt med (instans, nummer);
//! internt med index i `nodes` respektive `elements`.

use crate::elements::ElementType;
use crate::fem::dof::Dof;
use crate::model::{fail, FemError, Result};
use crate::section::BeamSection;
use std::collections::{BTreeMap, HashMap};

/// Instansnamn för en modell utan *PART/*ASSEMBLY.
pub const FLAT_INSTANCE: &str = "MODELL";
/// Instansnamn för noder och element definierade direkt i *ASSEMBLY.
pub const ASSEMBLY_INSTANCE: &str = "ASSEMBLY";

/// En datarad i ett nyckelordsblock.
#[derive(Debug, Clone, Default)]
pub struct DataLine {
    /// Fälten mellan kommatecknen, trimmade (tomma fält behålls som "").
    pub fields: Vec<String>,
    /// Radnummer i filen (1-baserat).
    pub line: usize,
    /// Raden slutade med kommatecken (fortsätter på nästa rad för t.ex. *ELEMENT).
    pub continued: bool,
}

/// Ett nyckelord med parametrar och datarader, som det står i .inp-filen.
#[derive(Debug, Clone, Default)]
pub struct RawKeyword {
    /// Nyckelordet med versaler och enkla mellanslag, t.ex. "SOLID SECTION".
    pub name: String,
    /// (PARAMETER med versaler, värde som skrivet eller "" om värde saknas).
    pub params: Vec<(String, String)>,
    pub data: Vec<DataLine>,
    pub file: String,
    pub line: usize,
}

/// Tolkar ett tal i .inp-formatet ("1.", "2.5e3", "1.0D+03"; tomt fält = 0).
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

    /// Parametervärde med versaler (namn, YES/NO osv.).
    pub fn param_upper(&self, key: &str) -> Option<String> {
        self.param(key).map(|v| v.trim().to_uppercase())
    }

    pub fn has(&self, key: &str) -> bool {
        self.params.iter().any(|(k, _)| k == key)
    }

    /// YES/NO-parameter; `default` om den saknas, YES om den står utan värde.
    pub fn flag(&self, key: &str, default: bool) -> Result<bool> {
        match self.param_upper(key) {
            None => Ok(default),
            Some(v) if v.is_empty() || v == "YES" || v == "ON" => Ok(true),
            Some(v) if v == "NO" || v == "OFF" => Ok(false),
            Some(v) => self.err(format!("{key}={v} ska vara YES eller NO")),
        }
    }

    /// "fil.inp rad 12 (*SOLID SECTION)" för felmeddelanden.
    pub fn location(&self) -> String {
        format!("{} rad {} (*{})", self.file, self.line, self.name)
    }

    pub fn err<T>(&self, msg: impl AsRef<str>) -> Result<T> {
        fail(format!("{}: {}", self.location(), msg.as_ref()))
    }

    pub fn err_at<T>(&self, line: usize, msg: impl AsRef<str>) -> Result<T> {
        fail(format!("{} rad {} (*{}): {}", self.file, line, self.name, msg.as_ref()))
    }

    /// Datarader där rader som slutar med komma slås ihop med nästa.
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

    /// Alla tal på en datarad (tomma fält blir 0).
    pub fn floats(&self, d: &DataLine) -> Result<Vec<f64>> {
        d.fields
            .iter()
            .map(|f| parse_f64(f).ok_or_else(|| FemError(format!("{} rad {} (*{}): '{f}' är inget tal", self.file, d.line, self.name))))
            .collect()
    }

    /// Talen på första dataraden (tom lista om dataraden saknas).
    pub fn first_floats(&self) -> Result<Vec<f64>> {
        match self.data.first() {
            Some(d) => self.floats(d),
            None => Ok(Vec::new()),
        }
    }
}

/// Material: namn, beteenden (*ELASTIC, *PLASTIC, ...) och densitet.
///
/// Beteendena sparas som de står i filen; materialregistret (`crate::materials`)
/// väljer en materialmodell som förbrukar dem.
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

    /// Linjärt isotropt elastiskt material (används av .rusty-modeller och tester).
    pub fn elastic(name: &str, e: f64, nu: f64, density: f64) -> Self {
        let kw = RawKeyword {
            name: "ELASTIC".into(),
            data: vec![DataLine { fields: vec![format!("{e:e}"), format!("{nu:e}")], line: 0, continued: false }],
            file: "(intern)".into(),
            ..Default::default()
        };
        MaterialDef { name: name.to_uppercase(), behaviors: vec![kw], density }
    }
}

/// Balktvärsnitt från *BEAM SECTION eller *BEAM GENERAL SECTION.
#[derive(Debug, Clone)]
pub struct BeamSectionDef {
    /// Tvärsnittsform: RECT, CIRC, PIPE, I, GENERAL, ...
    pub shape: String,
    pub profile: BeamSection,
    pub i12: f64,
    /// Ungefärlig riktning för tvärsnittets 1-axel (n1). `None` = standardriktningen.
    pub n1: Option<[f64; 3]>,
    /// (E, G) från *BEAM GENERAL SECTION; annars från materialet.
    pub moduli: Option<(f64, f64)>,
    pub density: f64,
}

#[derive(Debug, Clone)]
pub enum SectionKind {
    /// *SOLID SECTION: tjocklek i 2D, area för stänger, ingen data i 3D.
    Solid { thickness: Option<f64> },
    Beam(BeamSectionDef),
    /// *SHELL SECTION: lagras redan nu, skalelement läggs till i `elements/shell.rs`.
    Shell { thickness: f64, n_int: usize },
    /// Andra tvärsnitt (t.ex. *MEMBRANE SECTION); data finns i `raw`.
    Other,
}

#[derive(Debug, Clone)]
pub struct SectionDef {
    /// Elementset som tvärsnittet gäller (versaler, med instansprefix "INST.SET").
    pub elset: String,
    /// Materialnamn (versaler) eller tomt.
    pub material: String,
    /// Materialindex, satt av `FeModel::finalize`.
    pub material_index: Option<usize>,
    pub kind: SectionKind,
    /// Originalnyckelordet (för parametrar som bara vissa element behöver).
    pub raw: RawKeyword,
}

impl SectionDef {
    /// Tjocklek (2D), area (stänger) eller 1.
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

/// Ett *BOUNDARY-block (OP=NEW nollställer tidigare randvillkor).
#[derive(Debug, Clone, Default)]
pub struct BcBlock {
    pub op_new: bool,
    /// DISPLACEMENT (standard), VELOCITY, ACCELERATION.
    pub kind: String,
    pub amplitude: Option<String>,
    /// (nod, frihetsgrad, värde). Frihetsgrader som noden saknar ignoreras.
    pub items: Vec<(usize, Dof, f64)>,
    pub source: String,
}

/// Ett *CLOAD-block. Flera rader för samma nod och frihetsgrad summeras
/// (kontrollerat).
#[derive(Debug, Clone, Default)]
pub struct CloadBlock {
    pub op_new: bool,
    pub amplitude: Option<String>,
    pub follower: bool,
    pub items: Vec<(usize, Dof, f64)>,
    pub source: String,
}

/// En fördelad last på ett element (*DLOAD): typ (P1, P, GRAV, BX, PY, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct DistLoad {
    pub elem: usize,
    pub kind: String,
    pub magnitude: f64,
    /// Övriga värden på dataraden (t.ex. riktningen för GRAV).
    pub extra: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct DloadBlock {
    pub op_new: bool,
    pub amplitude: Option<String>,
    pub items: Vec<DistLoad>,
    pub source: String,
}

/// Ett *STEP ... *END STEP.
#[derive(Debug, Clone, Default)]
pub struct StepDef {
    pub name: String,
    /// *STEP-raden med alla parametrar.
    pub header: RawKeyword,
    /// Procedurnyckelordet (*STATIC, *FREQUENCY, ...).
    pub procedure: Option<RawKeyword>,
    /// NLGEOM=YES/NO om angivet (annars ärvs föregående steps värde).
    pub nlgeom: Option<bool>,
    pub perturbation: bool,
    /// INC= (största antal inkrement), standard 100.
    pub max_increments: usize,
    /// AMPLITUDE=STEP på *STEP (standard RAMP för statik).
    pub amplitude_step: bool,
    pub bcs: Vec<BcBlock>,
    pub cloads: Vec<CloadBlock>,
    pub dloads: Vec<DloadBlock>,
    /// Övriga nyckelord i steget som en registrerad modul har sparat.
    pub extra: Vec<RawKeyword>,
    /// Utdatabegäran (*OUTPUT, *NODE OUTPUT, ...). RustyFEM skriver alltid sina fält.
    pub output: Vec<RawKeyword>,
}

impl StepDef {
    /// Procedurens namn med versaler ("STATIC"), eller "" om den saknas.
    pub fn procedure_name(&self) -> &str {
        self.procedure.as_ref().map(|p| p.name.as_str()).unwrap_or("")
    }
}

/// En yta (*SURFACE): elementytor eller noder.
#[derive(Debug, Clone, Default)]
pub struct Surface {
    /// ELEMENT (standard) eller NODE.
    pub kind: String,
    /// (element, ytbeteckning) med .inp-formatets beteckningar: "S1".."S6" för
    /// kontinuum, "SPOS"/"SNEG" för skal, "S1" osv. för 2D-kanter.
    pub faces: Vec<(usize, String)>,
    /// (nod, area eller vikt) för nodbaserade ytor (vikt 1 om den saknas).
    pub nodes: Vec<(usize, f64)>,
    pub source: String,
}

/// Ett element i modellen.
#[derive(Clone)]
pub struct Element {
    pub etype: &'static dyn ElementType,
    pub label: i64,
    pub inst: u32,
    pub nodes: Vec<usize>,
    /// Tvärsnitt, satt av `finalize` från tvärsnittens elementset.
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
    /// Instansnamn (versaler). Index i `node_inst`/`Element::inst`.
    pub instances: Vec<String>,
    node_lookup: HashMap<(u32, i64), usize>,
    pub elements: Vec<Element>,
    elem_lookup: HashMap<(u32, i64), usize>,
    pub nsets: BTreeMap<String, Vec<usize>>,
    pub elsets: BTreeMap<String, Vec<usize>>,
    /// Ytor (*SURFACE), namn med versaler (med instansprefix "INST.YTA").
    pub surfaces: BTreeMap<String, Surface>,
    pub materials: Vec<MaterialDef>,
    pub sections: Vec<SectionDef>,
    /// Randvillkor i modelldata (före första steget).
    pub initial_bcs: Vec<BcBlock>,
    pub steps: Vec<StepDef>,
    /// Övriga modelldata som en registrerad modul har sparat (t.ex. *AMPLITUDE,
    /// *SURFACE, *TIE). Läses av den modul som äger nyckelordet.
    pub extra: Vec<RawKeyword>,
    pub warnings: Vec<String>,
}

/// Rotation kring axeln genom `a` och `b` med vinkeln `deg` (grader), Rodrigues formel.
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

/// Roterar vektorn `v` kring enhetsaxeln `k` med `deg` grader.
pub fn rotate_vec(v: [f64; 3], k: [f64; 3], deg: f64) -> [f64; 3] {
    let th = deg.to_radians();
    let (c, s) = (th.cos(), th.sin());
    let kxv = [k[1] * v[2] - k[2] * v[1], k[2] * v[0] - k[0] * v[2], k[0] * v[1] - k[1] * v[0]];
    let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
    [0, 1, 2].map(|i| v[i] * c + kxv[i] * s + k[i] * kv * (1.0 - c))
}

/// Placering av en instans: först förflyttning, sedan rotation.
#[derive(Debug, Clone, Copy)]
pub struct Placement {
    pub translation: [f64; 3],
    /// (punkt a, punkt b, vinkel i grader)
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

    /// Index för en instans; skapas om den saknas.
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

    /// Lägger till (eller flyttar, vid samma nummer) en nod.
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
            return fail(format!("element {} är definierat två gånger", label));
        }
        let i = self.elements.len();
        self.elements.push(Element { etype, label, inst, nodes, section: None });
        self.elem_lookup.insert((inst, label), i);
        Ok(i)
    }

    pub fn element(&self, inst: u32, label: i64) -> Option<usize> {
        self.elem_lookup.get(&(inst, label)).copied()
    }

    /// "MODELL.5" för felmeddelanden.
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

    /// Rumsdimension: 2 om alla element är plana, annars 3.
    pub fn dim(&self) -> usize {
        self.elements.iter().map(|e| e.etype.dim()).max().unwrap_or(3)
    }

    /// Lägger till noder i ett nodset (skapas om det saknas).
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

    /// Kopierar en part (en `FeModel` med bara geometri, set och tvärsnitt) in
    /// som instansen `name`. Set och tvärsnitt får prefixet "NAME.".
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
                .map_err(|err| FemError(format!("instans {prefix}: {err}")))?;
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

    /// Kopplar tvärsnitt till element och material. Anropas när hela filen är läst.
    pub fn finalize(&mut self) -> Result<()> {
        for s in self.sections.iter_mut() {
            if s.material.is_empty() {
                s.material_index = None;
                continue;
            }
            let up = s.material.to_uppercase();
            match self.materials.iter().position(|m| m.name == up) {
                Some(i) => s.material_index = Some(i),
                None => return s.raw.err(format!("materialet '{}' är inte definierat", s.material)),
            }
        }
        for e in self.elements.iter_mut() {
            e.section = None;
        }
        for (si, s) in self.sections.iter().enumerate() {
            let Some(set) = self.elsets.get(&s.elset) else {
                return s.raw.err(format!("elementsetet '{}' finns inte", s.elset));
            };
            for &e in set {
                if let Some(old) = self.elements[e].section {
                    if old != si {
                        let el = &self.elements[e];
                        return s.raw.err(format!(
                            "element {}.{} har redan ett tvärsnitt ({})",
                            self.instances[el.inst as usize], el.label, self.sections[old].raw.location()
                        ));
                    }
                }
                self.elements[e].section = Some(si);
            }
        }
        // interna noder (t.ex. hybridelementens tryckvariabler): negativa nummer
        // −(16·elementnummer + k + 1) i elementets instans, i tyngdpunkten
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
        // noder som anges i tvärsnittet (t.ex. referensnod för generaliserad plan töjning)
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
                    None => return self.sections[si].raw.err(format!("noden {l} finns inte")),
                }
            }
            self.elements[e].nodes.extend(extra);
        }
        // område acoustics2: delade interna noder för akustiska infinita element
        crate::elements::acoustic_infinite::add_nodes(self)?;
        // område acoustics2: *EOS COMPACTION – det porösa materialets densitet
        crate::materials::eos::compaction::adjust_density(self)?;
        Ok(())
    }
}
