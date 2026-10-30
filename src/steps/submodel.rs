//! Nodbaserad submodellering: *SUBMODEL och *BOUNDARY, SUBMODEL, STEP=n.
//!
//! De drivna nodernas värden interpoleras från en global körning
//! (`--global`). Globalmodellens resultat läses från
//! - RustyFEMs resultatfil (`rustyfem run global.inp --out global.res`, RustyFEMs
//!   textformat för resultat); nätet står i kommentarsrader `#XN`/`#XE`/`#XS` (se
//!   `mesh_text`) som andra läsare hoppar över, eller
//! - en omstartfil (`global.rfres`), vars indatatext ger nätet och vars
//!   sparade tillstånd ger nodvärdena vid stegens slut.
//!
//! Filen anges i jobbkontexten (`restart::Job::global`, CLI:t
//! `rustyfem run sub.inp --global global.res`).
//!
//! För varje driven nod söks det globala element (bland GLOBAL ELSET= om det
//! anges) som innehåller noden; de naturliga koordinaterna bestäms med Newton
//! (Gauss–Newton för skal i 3D, dvs. projektion på mittytan) och värdet
//! interpoleras med elementets formfunktioner. En nod utanför alla element
//! drivs av närmaste punkt på det närmaste elementet (ingen extrapolation;
//! kontrollerat i tests/cases/infrastructure) om avståndet är högst
//! EXTERIOR TOLERANCE (standard 0,05) gånger globalmodellens medelelementstorlek
//! eller ABSOLUTE EXTERIOR TOLERANCE; annars blir det fel.
//!
//! Förskjutningar (1–3), rotationer (4–6) och temperatur (11) kan drivas.
//! Värdet vid globalstegets slut (sista framen) används som randvillkorets
//! slutvärde och rampas över submodellens steg som ett vanligt *BOUNDARY;
//! för en linjär globalmodell är det detsamma som tidsinterpolation.
//!
//! Elementformer i globalmodellen: 3-, 4-, 6- och 8-nodiga plana, axisymmetriska
//! element och skal/membran, samt tetraedrar (4, 10), kilar (6) och hexaedrar
//! (8, 20). TYPE=SURFACE (spänningsbaserad submodellering) stöds inte.

use super::api::StepType;
use crate::elements::Family;
use crate::fem::dof::{self, NodeVec, SLOTS};
use crate::fem::model::{FeModel, RawKeyword};
use crate::fem::output::ResultSet;
use crate::model::{fail, FemError, Result};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::path::Path;

/// Inga egna steg-typer.
pub static STEP_TYPES: &[&dyn StepType] = &[];

/// `BcBlock::kind` för *BOUNDARY, SUBMODEL (följt av ":" och globalsteget).
pub const SUBMODEL_KIND: &str = "SUBMODEL:";

/// Nodset med submodellens drivna noder (från *SUBMODEL-dataraderna).
pub const DRIVEN_NSET: &str = "__SUBMODEL_DRIVNA";

// ---------------------------------------------------------------------------
// Elementformer

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    T3,
    T6,
    Q4,
    Q8,
    Tet4,
    Tet10,
    W6,
    H8,
    H20,
}

impl Shape {
    fn code(self) -> &'static str {
        match self {
            Shape::T3 => "T3",
            Shape::T6 => "T6",
            Shape::Q4 => "Q4",
            Shape::Q8 => "Q8",
            Shape::Tet4 => "TET4",
            Shape::Tet10 => "TET10",
            Shape::W6 => "W6",
            Shape::H8 => "H8",
            Shape::H20 => "H20",
        }
    }

    fn from_code(s: &str) -> Option<Shape> {
        Some(match s {
            "T3" => Shape::T3,
            "T6" => Shape::T6,
            "Q4" => Shape::Q4,
            "Q8" => Shape::Q8,
            "TET4" => Shape::Tet4,
            "TET10" => Shape::Tet10,
            "W6" => Shape::W6,
            "H8" => Shape::H8,
            "H20" => Shape::H20,
            _ => return None,
        })
    }

    fn n_nodes(self) -> usize {
        match self {
            Shape::T3 => 3,
            Shape::T6 => 6,
            Shape::Q4 => 4,
            Shape::Q8 => 8,
            Shape::Tet4 => 4,
            Shape::Tet10 => 10,
            Shape::W6 => 6,
            Shape::H8 => 8,
            Shape::H20 => 20,
        }
    }

    /// Antal naturliga koordinater.
    fn pdim(self) -> usize {
        match self {
            Shape::T3 | Shape::T6 | Shape::Q4 | Shape::Q8 => 2,
            _ => 3,
        }
    }

    fn center(self) -> [f64; 3] {
        match self {
            Shape::T3 | Shape::T6 => [1.0 / 3.0, 1.0 / 3.0, 0.0],
            Shape::Tet4 | Shape::Tet10 => [0.25, 0.25, 0.25],
            Shape::W6 => [1.0 / 3.0, 1.0 / 3.0, 0.0],
            _ => [0.0; 3],
        }
    }

    /// Närmaste punkt i referenselementet (ungefärlig för simplex).
    fn clamp(self, x: [f64; 3]) -> [f64; 3] {
        let tri = |r: f64, s: f64| -> (f64, f64) {
            let (r, s) = (r.max(0.0), s.max(0.0));
            let sum = r + s;
            if sum > 1.0 { (r / sum, s / sum) } else { (r, s) }
        };
        match self {
            Shape::T3 | Shape::T6 => {
                let (r, s) = tri(x[0], x[1]);
                [r, s, 0.0]
            }
            Shape::Q4 | Shape::Q8 => [x[0].clamp(-1.0, 1.0), x[1].clamp(-1.0, 1.0), 0.0],
            Shape::Tet4 | Shape::Tet10 => {
                let (r, s, t) = (x[0].max(0.0), x[1].max(0.0), x[2].max(0.0));
                let sum = r + s + t;
                if sum > 1.0 { [r / sum, s / sum, t / sum] } else { [r, s, t] }
            }
            Shape::W6 => {
                let (r, s) = tri(x[0], x[1]);
                [r, s, x[2].clamp(-1.0, 1.0)]
            }
            Shape::H8 | Shape::H20 => [x[0].clamp(-1.0, 1.0), x[1].clamp(-1.0, 1.0), x[2].clamp(-1.0, 1.0)],
        }
    }

    /// Formfunktionerna i den naturliga punkten `x` (.inp-formatets nodordning).
    pub fn n(self, x: [f64; 3]) -> Vec<f64> {
        let (r, s, t) = (x[0], x[1], x[2]);
        match self {
            Shape::T3 => vec![1.0 - r - s, r, s],
            Shape::T6 => {
                let l = [1.0 - r - s, r, s];
                vec![l[0] * (2.0 * l[0] - 1.0), l[1] * (2.0 * l[1] - 1.0), l[2] * (2.0 * l[2] - 1.0), 4.0 * l[0] * l[1], 4.0 * l[1] * l[2], 4.0 * l[2] * l[0]]
            }
            Shape::Q4 => Q4P.iter().map(|p| 0.25 * (1.0 + p[0] * r) * (1.0 + p[1] * s)).collect(),
            Shape::Q8 => Q8P
                .iter()
                .map(|p| {
                    if p[0] == 0.0 {
                        0.5 * (1.0 - r * r) * (1.0 + p[1] * s)
                    } else if p[1] == 0.0 {
                        0.5 * (1.0 + p[0] * r) * (1.0 - s * s)
                    } else {
                        0.25 * (1.0 + p[0] * r) * (1.0 + p[1] * s) * (p[0] * r + p[1] * s - 1.0)
                    }
                })
                .collect(),
            Shape::Tet4 => vec![1.0 - r - s - t, r, s, t],
            Shape::Tet10 => {
                let l = [1.0 - r - s - t, r, s, t];
                let mut v: Vec<f64> = l.iter().map(|&a| a * (2.0 * a - 1.0)).collect();
                for (a, b) in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)] {
                    v.push(4.0 * l[a] * l[b]);
                }
                v
            }
            Shape::W6 => {
                let l = [1.0 - r - s, r, s];
                let mut v: Vec<f64> = l.iter().map(|a| a * 0.5 * (1.0 - t)).collect();
                v.extend(l.iter().map(|a| a * 0.5 * (1.0 + t)));
                v
            }
            Shape::H8 => H8P.iter().map(|p| 0.125 * (1.0 + p[0] * r) * (1.0 + p[1] * s) * (1.0 + p[2] * t)).collect(),
            Shape::H20 => H20P
                .iter()
                .map(|p| {
                    let f = |a: f64, x: f64| if a == 0.0 { 1.0 - x * x } else { 1.0 + a * x };
                    let base = f(p[0], r) * f(p[1], s) * f(p[2], t);
                    if p[0] != 0.0 && p[1] != 0.0 && p[2] != 0.0 {
                        0.125 * base * (p[0] * r + p[1] * s + p[2] * t - 2.0)
                    } else {
                        0.25 * base
                    }
                })
                .collect(),
        }
    }
}

const Q4P: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
const Q8P: [[f64; 2]; 8] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
const H8P: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];
const H20P: [[f64; 3]; 20] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [0.0, -1.0, -1.0],
    [1.0, 0.0, -1.0],
    [0.0, 1.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, -1.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [-1.0, 0.0, 1.0],
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
];

/// Formen för en elementtyp (`None` = används inte för interpolation).
pub fn shape_of(et: &dyn crate::elements::ElementType) -> Option<Shape> {
    let n = et.n_connectivity();
    let planar = matches!(et.family(), Family::Continuum2D | Family::Axisymmetric | Family::Shell | Family::Membrane) || (et.family() == Family::Heat && et.dim() == 2);
    let solid = et.family() == Family::Continuum3D || (et.family() == Family::Heat && et.dim() == 3);
    if planar {
        return match n {
            3 => Some(Shape::T3),
            4 => Some(Shape::Q4),
            6 => Some(Shape::T6),
            8 | 9 => Some(Shape::Q8),
            _ => None,
        };
    }
    if solid {
        return match n {
            4 => Some(Shape::Tet4),
            10 => Some(Shape::Tet10),
            6 => Some(Shape::W6),
            8 => Some(Shape::H8),
            20 | 27 => Some(Shape::H20),
            _ => None,
        };
    }
    None
}

// ---------------------------------------------------------------------------
// Globalmodellens nät

#[derive(Debug, Clone)]
struct GElem {
    shape: Shape,
    nodes: Vec<usize>,
}

/// Globalmodellens nät: noder, element med former och elementset.
#[derive(Debug, Clone, Default)]
pub struct GlobalMesh {
    pub dim: usize,
    nodes: Vec<[f64; 3]>,
    node_key: Vec<(String, i64)>,
    elems: Vec<GElem>,
    elem_key: HashMap<(String, i64), usize>,
    elsets: BTreeMap<String, Vec<usize>>,
}

/// Nätet som kommentarsrader för resultatfilen (läses av `GlobalMesh::parse`).
pub fn mesh_text(m: &FeModel) -> String {
    let mut s = String::from("# rustyfem-mesh 1\n");
    let _ = writeln!(s, "#XD;{}", m.dim());
    for n in 0..m.nodes.len() {
        let x = m.nodes[n];
        let _ = writeln!(s, "#XN;{};{};{:.17e};{:.17e};{:.17e}", m.instances[m.node_inst[n] as usize], m.node_label[n], x[0], x[1], x[2]);
    }
    for e in &m.elements {
        let Some(sh) = shape_of(e.etype) else { continue };
        let inst = &m.instances[e.inst as usize];
        let _ = write!(s, "#XE;{inst};{};{}", e.label, sh.code());
        for &n in e.nodes.iter().take(sh.n_nodes()) {
            let _ = write!(s, ";{}.{}", m.instances[m.node_inst[n] as usize], m.node_label[n]);
        }
        s.push('\n');
    }
    for (name, set) in &m.elsets {
        let _ = write!(s, "#XS;{name}");
        for &e in set {
            let el = &m.elements[e];
            if shape_of(el.etype).is_some() {
                let _ = write!(s, ";{}.{}", m.instances[el.inst as usize], el.label);
            }
        }
        s.push('\n');
    }
    s
}

fn split_ref(r: &str) -> Option<(String, i64)> {
    let (i, l) = r.rsplit_once('.')?;
    Some((i.to_uppercase(), l.trim().parse().ok()?))
}

impl GlobalMesh {
    /// Nätet ur en modell.
    pub fn from_model(m: &FeModel) -> GlobalMesh {
        let text = mesh_text(m);
        GlobalMesh::parse(&text).unwrap_or_default()
    }

    /// Läser nätraderna i en resultatfil (`None` om de saknas).
    pub fn parse(text: &str) -> Option<GlobalMesh> {
        let mut g = GlobalMesh { dim: 3, ..Default::default() };
        let mut nmap: HashMap<(String, i64), usize> = HashMap::new();
        let mut any = false;
        for line in text.lines() {
            let Some(rest) = line.strip_prefix("#X") else { continue };
            let p: Vec<&str> = rest.split(';').collect();
            match p[0] {
                "D" => g.dim = p.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(3),
                "N" if p.len() >= 6 => {
                    let key = (p[1].to_uppercase(), p[2].trim().parse().ok()?);
                    let x = [p[3].trim().parse().ok()?, p[4].trim().parse().ok()?, p[5].trim().parse().ok()?];
                    nmap.insert(key.clone(), g.nodes.len());
                    g.nodes.push(x);
                    g.node_key.push(key);
                    any = true;
                }
                "E" if p.len() >= 4 => {
                    let shape = Shape::from_code(p[3])?;
                    let mut nodes = Vec::new();
                    for r in &p[4..] {
                        nodes.push(*nmap.get(&split_ref(r)?)?);
                    }
                    if nodes.len() != shape.n_nodes() {
                        return None;
                    }
                    g.elem_key.insert((p[1].to_uppercase(), p[2].trim().parse().ok()?), g.elems.len());
                    g.elems.push(GElem { shape, nodes });
                }
                "S" if p.len() >= 2 => {
                    let set: Vec<usize> = p[2..].iter().filter_map(|r| split_ref(r).and_then(|k| g.elem_key.get(&k).copied())).collect();
                    g.elsets.insert(p[1].to_uppercase(), set);
                }
                _ => {}
            }
        }
        any.then_some(g)
    }

    /// Medelstorleken (diagonalen i elementets omslutande låda).
    fn mean_size(&self) -> f64 {
        if self.elems.is_empty() {
            return 1.0;
        }
        let s: f64 = self.elems.iter().map(|e| bbox_diag(&self.bbox(e))).sum();
        s / self.elems.len() as f64
    }

    fn bbox(&self, e: &GElem) -> [[f64; 3]; 2] {
        let mut b = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        for &n in &e.nodes {
            for k in 0..3 {
                b[0][k] = b[0][k].min(self.nodes[n][k]);
                b[1][k] = b[1][k].max(self.nodes[n][k]);
            }
        }
        b
    }

    fn map(&self, e: &GElem, xi: [f64; 3]) -> [f64; 3] {
        let n = e.shape.n(xi);
        let mut x = [0.0; 3];
        for (i, &nd) in e.nodes.iter().enumerate() {
            for k in 0..3 {
                x[k] += n[i] * self.nodes[nd][k];
            }
        }
        x
    }

    /// Naturliga koordinater för punkten `p` i elementet (Gauss–Newton) och
    /// avståndet till elementet (0 inuti).
    fn locate(&self, e: &GElem, p: [f64; 3]) -> ([f64; 3], f64) {
        let pd = e.shape.pdim();
        let sd = if self.dim == 2 { 2 } else { 3 };
        let mut xi = e.shape.center();
        let h = 1e-7;
        for _ in 0..40 {
            let x = self.map(e, xi);
            let r: Vec<f64> = (0..sd).map(|k| x[k] - p[k]).collect();
            // jakobian dx/dξ med centrala differenser
            let mut jac = vec![[0.0; 3]; sd];
            for a in 0..pd {
                let (mut xp, mut xm) = (xi, xi);
                xp[a] += h;
                xm[a] -= h;
                let (fp, fm) = (self.map(e, xp), self.map(e, xm));
                for k in 0..sd {
                    jac[k][a] = (fp[k] - fm[k]) / (2.0 * h);
                }
            }
            // (JᵀJ) Δ = −Jᵀ r
            let mut a = [[0.0; 3]; 3];
            let mut b = [0.0; 3];
            for i in 0..pd {
                for j in 0..pd {
                    a[i][j] = (0..sd).map(|k| jac[k][i] * jac[k][j]).sum();
                }
                b[i] = -(0..sd).map(|k| jac[k][i] * r[k]).sum::<f64>();
            }
            let Some(d) = solve_small(&a, &b, pd) else { break };
            let mut step = 0.0f64;
            for i in 0..pd {
                // begränsa steget så att Newton inte rusar iväg från elementet
                let di = d[i].clamp(-2.0, 2.0);
                xi[i] += di;
                step = step.max(di.abs());
            }
            if step < 1e-13 {
                break;
            }
        }
        let xc = e.shape.clamp(xi);
        let (a, b) = (self.map(e, xc), self.map(e, xi));
        let ins: f64 = (0..sd).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt();
        // projektionens avstånd: för fullt dimensionerade element är punkten själv
        let off: f64 = if pd == sd { (0..sd).map(|k| (b[k] - p[k]).powi(2)).sum::<f64>().sqrt() } else { 0.0 };
        (xi, ins + off)
    }
}

fn bbox_diag(b: &[[f64; 3]; 2]) -> f64 {
    (0..3).map(|k| (b[1][k] - b[0][k]).powi(2)).sum::<f64>().sqrt()
}

fn solve_small(a: &[[f64; 3]; 3], b: &[f64; 3], n: usize) -> Option<[f64; 3]> {
    let mut m = *a;
    let mut x = *b;
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i][c].abs().partial_cmp(&m[j][c].abs()).unwrap())?;
        if m[p][c].abs() < 1e-300 {
            return None;
        }
        m.swap(c, p);
        x.swap(c, p);
        for r in 0..n {
            if r != c {
                let f = m[r][c] / m[c][c];
                for k in c..n {
                    m[r][k] -= f * m[c][k];
                }
                x[r] -= f * x[c];
            }
        }
    }
    let mut out = [0.0; 3];
    for i in 0..n {
        out[i] = x[i] / m[i][i];
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Globalmodellens resultat

/// Globalmodellens nät och nodvärden (per slot) vid varje stegs slut.
pub struct GlobalSource {
    pub mesh: GlobalMesh,
    /// (globalsteg 1-baserat, nodvärden i nätets nodordning).
    pub steps: Vec<(usize, Vec<NodeVec>)>,
}

impl GlobalSource {
    /// Läser en resultatfil (RustyFEMs textformat med nät) eller en omstartfil (.rfres).
    pub fn load(path: &Path) -> Result<GlobalSource> {
        let bytes = std::fs::read(path).map_err(|e| FemError(format!("kunde inte läsa globalmodellen {}: {e}", path.display())))?;
        if bytes.starts_with(b"RUSTYFEM-RESTART") {
            let rf = super::restart::RestartFile::from_bytes(&bytes)?;
            let m = rf.model(path.parent().unwrap_or(Path::new(".")))?;
            let mesh = GlobalMesh::from_model(&m);
            let steps = rf.steps.iter().map(|s| (s.index, s.state.u.clone())).collect();
            return Ok(GlobalSource { mesh, steps });
        }
        let text = String::from_utf8_lossy(&bytes);
        let Some(mesh) = GlobalMesh::parse(&text) else {
            return fail(format!(
                "{} saknar globalmodellens nät – skriv den med 'rustyfem run global.inp --out {}' eller använd omstartfilen",
                path.display(),
                path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()
            ));
        };
        let rs = ResultSet::parse(&text)?;
        Ok(Self::from_results(mesh, &rs))
    }

    /// Nodvärden ur resultaten (sista framen i varje steg).
    pub fn from_results(mesh: GlobalMesh, rs: &ResultSet) -> GlobalSource {
        let mut steps = Vec::new();
        for st in &rs.steps {
            let Some(fr) = st.frames.last() else { continue };
            let mut v = vec![[0.0; SLOTS]; mesh.nodes.len()];
            for (n, key) in mesh.node_key.iter().enumerate() {
                let get = |f: &str, l: &str| fr.node.get(f).and_then(|fd| fd.get(key, l));
                for (k, l) in ["U1", "U2", "U3"].iter().enumerate() {
                    v[n][k] = get("U", l).unwrap_or(0.0);
                }
                for (k, l) in ["UR1", "UR2", "UR3"].iter().enumerate() {
                    v[n][3 + k] = get("UR", l).unwrap_or(0.0);
                }
                v[n][7] = get("NT11", "NT11").unwrap_or(0.0);
            }
            steps.push((st.index, v));
        }
        GlobalSource { mesh, steps }
    }

    fn values(&self, step: usize) -> Option<&Vec<NodeVec>> {
        self.steps.iter().rev().find(|(i, _)| *i == step).map(|(_, v)| v)
    }
}

/// Interpolationen för en driven nod: globala noder och vikter.
#[derive(Debug, Clone)]
pub struct Driver {
    pub nodes: Vec<usize>,
    pub weights: Vec<f64>,
    /// Avstånd till närmaste element (0 inuti).
    pub distance: f64,
}

/// Inställningar från *SUBMODEL.
struct Options {
    elset: Option<Vec<usize>>,
    tol: f64,
}

fn options(kw: &RawKeyword, g: &GlobalMesh) -> Result<Options> {
    if let Some(t) = kw.param_upper("TYPE") {
        if t != "NODE" {
            return kw.err(format!("TYPE={t} stöds inte (bara nodbaserad submodellering, TYPE=NODE)"));
        }
    }
    let num = |k: &str| -> Result<Option<f64>> {
        match kw.param(k) {
            None => Ok(None),
            Some(v) => crate::fem::model::parse_f64(v).map(Some).ok_or_else(|| FemError(format!("{}: {k}={v} är inget tal", kw.location()))),
        }
    };
    let rel = num("EXTERIOR TOLERANCE")?.unwrap_or(0.05);
    let tol = match num("ABSOLUTE EXTERIOR TOLERANCE")? {
        Some(a) => a,
        None => rel * g.mean_size(),
    };
    let elset = match kw.param("GLOBAL ELSET") {
        None => None,
        Some(name) => {
            let mut all = Vec::new();
            for n in name.split(',') {
                let up = n.trim().to_uppercase();
                let set = g
                    .elsets
                    .get(&up)
                    .or_else(|| g.elsets.iter().find(|(k, _)| k.rsplit_once('.').is_some_and(|(_, s)| s == up)).map(|(_, v)| v));
                match set {
                    Some(s) => all.extend(s.iter().copied()),
                    None => return kw.err(format!("GLOBAL ELSET={up} finns inte i globalmodellen")),
                }
            }
            Some(all)
        }
    };
    Ok(Options { elset, tol })
}

impl GlobalMesh {
    /// Interpolationen för punkten `p` (se modulens dokumentation).
    pub fn driver(&self, p: [f64; 3], elset: Option<&[usize]>, tol: f64) -> std::result::Result<Driver, f64> {
        let all: Vec<usize>;
        let cand: &[usize] = match elset {
            Some(s) => s,
            None => {
                all = (0..self.elems.len()).collect();
                &all
            }
        };
        let mut best: Option<(f64, usize, [f64; 3])> = None;
        for &ei in cand {
            let e = &self.elems[ei];
            let b = self.bbox(e);
            let margin = tol + 1e-9 * bbox_diag(&b);
            let sd = if self.dim == 2 { 2 } else { 3 };
            let pd = e.shape.pdim();
            // skal i 3D: lådan förstoras i normalens riktning (projektion)
            let extra = if pd < sd { 0.5 * bbox_diag(&b) } else { 0.0 };
            if (0..sd).any(|k| p[k] < b[0][k] - margin - extra || p[k] > b[1][k] + margin + extra) {
                continue;
            }
            let (xi, d) = self.locate(e, p);
            if best.as_ref().is_none_or(|(bd, _, _)| d < *bd) {
                best = Some((d, ei, xi));
            }
            if d <= 1e-10 * bbox_diag(&b) {
                break;
            }
        }
        match best {
            Some((d, ei, xi)) if d <= tol + 1e-12 => {
                // utanför elementet: närmaste punkt på elementet
                // (ingen extrapolation)
                let e = &self.elems[ei];
                Ok(Driver { nodes: e.nodes.clone(), weights: e.shape.n(e.shape.clamp(xi)), distance: d })
            }
            Some((d, _, _)) => Err(d),
            None => Err(f64::INFINITY),
        }
    }
}

/// Byter *BOUNDARY, SUBMODEL mot vanliga randvillkor med värden från
/// globalmodellen. `None` om modellen inte är en submodell.
pub fn prepare(model: &FeModel) -> Result<Option<FeModel>> {
    let uses = model.steps.iter().any(|s| s.bcs.iter().any(|b| b.kind.starts_with(SUBMODEL_KIND)));
    if !uses {
        return Ok(None);
    }
    let Some(kw) = model.extra.iter().find(|k| k.name == "SUBMODEL") else {
        return fail("*BOUNDARY, SUBMODEL kräver *SUBMODEL i modelldata");
    };
    let Some(path) = super::restart::job().and_then(|j| j.global) else {
        return kw.err("submodellen kräver globalmodellens resultat (rustyfem run sub.inp --global global.res)");
    };
    let g = GlobalSource::load(&path)?;
    let opts = options(kw, &g.mesh)?;
    let driven: Vec<usize> = model.nsets.get(DRIVEN_NSET).cloned().unwrap_or_default();
    let mut cache: HashMap<usize, Driver> = HashMap::new();
    let mut out = model.clone();
    for step in out.steps.iter_mut() {
        for b in step.bcs.iter_mut() {
            let Some(gs) = b.kind.strip_prefix(SUBMODEL_KIND) else { continue };
            let gstep: usize = gs.parse().unwrap_or(1);
            let Some(vals) = g.values(gstep) else {
                return fail(format!("{}: globalmodellen ({}) saknar steg {gstep}", b.source, path.display()));
            };
            for item in b.items.iter_mut() {
                let (n, d, _) = *item;
                if driven.binary_search(&n).is_err() {
                    return fail(format!("{}: nod {} är inte en driven nod i *SUBMODEL", b.source, model.node_name(n)));
                }
                if !cache.contains_key(&n) {
                    let dr = g.mesh.driver(model.nodes[n], opts.elset.as_deref(), opts.tol).map_err(|dist| {
                        let how = if dist.is_finite() { format!("avståndet till närmaste element är {dist:.4e}") } else { "inget element i närheten".into() };
                        FemError(format!(
                            "{}: driven nod {} ligger utanför globalmodellen ({how}, tolerans {:.4e}; öka EXTERIOR TOLERANCE)",
                            kw.location(),
                            model.node_name(n),
                            opts.tol
                        ))
                    })?;
                    cache.insert(n, dr);
                }
                let dr = &cache[&n];
                let s = dof::slot_of(d);
                let v: f64 = dr.nodes.iter().zip(&dr.weights).map(|(&gn, w)| w * vals[gn][s]).sum();
                item.2 = v;
            }
            b.kind = "DISPLACEMENT".into();
        }
    }
    Ok(Some(out))
}
