//! Modellen: geometri, material, stöd och laster på ett ställe.
//!
//! Skillnader mot arbetsflödet med indatafiler (.inp):
//! - ingen part/instance/assembly: geometri läggs direkt i modellen och noder
//!   på samma plats slås ihop automatiskt (balkar fäster i plattor osv.)
//! - ingen separat mesh-modul: varje geometrikommando meshar sig självt
//! - stöd och laster väljs med geometri (`Sel::x(0.0)`) i stället för set
//! - laster anges som totalkraft och fördelas konsistent över valda ytor/kanter
//! - ett enda linjärt statiskt steg
//!
//! Modellen löses genom att översättas till kärnans `FeModel` (se `crate::legacy`).

use crate::results::Results;
use crate::section::BeamSection;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

pub const UX: usize = 0;
pub const UY: usize = 1;
pub const UZ: usize = 2;
pub const RX: usize = 3;
pub const RY: usize = 4;
pub const RZ: usize = 5;
pub const DOF_NAMES: [&str; 6] = ["ux", "uy", "uz", "rx", "ry", "rz"];

#[derive(Debug, Clone)]
pub struct FemError(pub String);

impl fmt::Display for FemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FemError {}

pub type Result<T> = std::result::Result<T, FemError>;

pub(crate) fn fail<T>(msg: impl Into<String>) -> Result<T> {
    Err(FemError(msg.into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    Two,
    Three,
}

/// Elementtyper, namngivna som i .inp-formatet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElemKind {
    T2D2,
    T3D2,
    B23,
    B33,
    CPS3,
    CPS4,
    CPE3,
    CPE4,
    C3D8,
}

impl ElemKind {
    pub fn type_name(self) -> String {
        format!("{:?}", self)
    }

    /// Frihetsgrader per nod (index i UX..RZ).
    pub fn node_dofs(self) -> &'static [usize] {
        use ElemKind::*;
        match self {
            T2D2 | CPS3 | CPS4 | CPE3 | CPE4 => &[UX, UY],
            T3D2 | C3D8 => &[UX, UY, UZ],
            B23 => &[UX, UY, RZ],
            B33 => &[UX, UY, UZ, RX, RY, RZ],
        }
    }

    pub fn is_beam(self) -> bool {
        matches!(self, ElemKind::B23 | ElemKind::B33)
    }

    pub fn is_truss(self) -> bool {
        matches!(self, ElemKind::T2D2 | ElemKind::T3D2)
    }

    pub fn is_solid(self) -> bool {
        !self.is_beam() && !self.is_truss()
    }

    pub fn vtk_type(self) -> u8 {
        use ElemKind::*;
        match self {
            T2D2 | T3D2 | B23 | B33 => 3,
            CPS3 | CPE3 => 5,
            CPS4 | CPE4 => 9,
            C3D8 => 12,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Material {
    pub name: String,
    pub e: f64,
    pub nu: f64,
    pub density: f64,
}

impl Material {
    pub fn new(name: &str, e: f64, nu: f64) -> Self {
        Material { name: name.to_string(), e, nu, density: 0.0 }
    }

    pub fn with_density(mut self, density: f64) -> Self {
        self.density = density;
        self
    }

    /// Stål i SI-enheter (Pa, kg/m³).
    pub fn steel() -> Self {
        Material::new("steel", 210e9, 0.3).with_density(7850.0)
    }

    /// Aluminium i SI-enheter (Pa, kg/m³).
    pub fn aluminium() -> Self {
        Material::new("aluminium", 70e9, 0.33).with_density(2700.0)
    }

    pub fn shear_modulus(&self) -> f64 {
        self.e / (2.0 * (1.0 + self.nu))
    }
}

#[derive(Debug, Clone)]
pub enum Section {
    /// Kontinuumelement; `thickness` används bara i 2D.
    Solid { material: usize, thickness: f64 },
    Truss { material: usize, area: f64 },
    /// `n1` är balkens första tvärsnittsaxel.
    Beam { material: usize, profile: BeamSection, n1: [f64; 3] },
}

impl Section {
    pub fn material(&self) -> usize {
        match self {
            Section::Solid { material, .. } | Section::Truss { material, .. } | Section::Beam { material, .. } => *material,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Element {
    pub kind: ElemKind,
    pub nodes: Vec<usize>,
    pub section: usize,
}

/// Nodurval med geometri. Exempel: `Sel::x(0.0)`, `Sel::x(2.0).and_y_in(0.0, 0.1)`.
#[derive(Debug, Clone)]
pub enum Sel {
    All,
    /// Lista av (axel, min, max).
    Box(Vec<(usize, f64, f64)>),
    Point([f64; 3]),
    Nodes(Vec<usize>),
}

impl Sel {
    pub fn all() -> Self {
        Sel::All
    }
    pub fn x(v: f64) -> Self {
        Sel::Box(vec![(0, v, v)])
    }
    pub fn y(v: f64) -> Self {
        Sel::Box(vec![(1, v, v)])
    }
    pub fn z(v: f64) -> Self {
        Sel::Box(vec![(2, v, v)])
    }
    pub fn x_in(lo: f64, hi: f64) -> Self {
        Sel::Box(vec![(0, lo, hi)])
    }
    pub fn y_in(lo: f64, hi: f64) -> Self {
        Sel::Box(vec![(1, lo, hi)])
    }
    pub fn z_in(lo: f64, hi: f64) -> Self {
        Sel::Box(vec![(2, lo, hi)])
    }
    pub fn point(p: [f64; 3]) -> Self {
        Sel::Point(p)
    }
    pub fn nodes(ids: Vec<usize>) -> Self {
        Sel::Nodes(ids)
    }

    /// Lägg till ett villkor `lo <= koordinat[axis] <= hi`.
    pub fn and(self, axis: usize, lo: f64, hi: f64) -> Self {
        match self {
            Sel::Box(mut c) => {
                c.push((axis, lo, hi));
                Sel::Box(c)
            }
            Sel::All => Sel::Box(vec![(axis, lo, hi)]),
            other => other,
        }
    }
    pub fn and_x(self, v: f64) -> Self {
        self.and(0, v, v)
    }
    pub fn and_y(self, v: f64) -> Self {
        self.and(1, v, v)
    }
    pub fn and_z(self, v: f64) -> Self {
        self.and(2, v, v)
    }
    pub fn and_x_in(self, lo: f64, hi: f64) -> Self {
        self.and(0, lo, hi)
    }
    pub fn and_y_in(self, lo: f64, hi: f64) -> Self {
        self.and(1, lo, hi)
    }
    pub fn and_z_in(self, lo: f64, hi: f64) -> Self {
        self.and(2, lo, hi)
    }

    pub fn describe(&self) -> String {
        match self {
            Sel::All => "all".into(),
            Sel::Box(c) => c
                .iter()
                .map(|&(a, lo, hi)| {
                    let ax = ["x", "y", "z"][a];
                    if lo == hi { format!("{ax}={lo}") } else { format!("{ax}={lo}..{hi}") }
                })
                .collect::<Vec<_>>()
                .join(" "),
            Sel::Point(p) => format!("at=({},{},{})", p[0], p[1], p[2]),
            Sel::Nodes(n) => format!("{} noder", n.len()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Fixity {
    /// Alla frihetsgrader låsta (inspänning).
    Fixed,
    /// Translationer låsta, rotationer fria (led).
    Pinned,
    /// Valda frihetsgrader med föreskrivet värde.
    Dofs(Vec<(usize, f64)>),
}

#[derive(Debug, Clone)]
pub struct Support {
    pub sel: Sel,
    pub fixity: Fixity,
}

#[derive(Debug, Clone)]
pub enum Load {
    /// Total kraft [fx,fy,fz] och moment [mx,my,mz] på urvalet.
    Force { sel: Sel, f: [f64; 6] },
    /// Normaltryck (positivt = trycker in i materialet) på urvalets randytor.
    Pressure { sel: Sel, p: f64 },
}

/// En randkant (2D) eller randyta (3D) på ett kontinuumelement.
pub(crate) struct Facet {
    pub nodes: Vec<usize>,
    pub elem: usize,
}

#[derive(Debug, Clone)]
pub struct Model {
    pub dim: Dim,
    /// Plant töjningstillstånd i stället för plan spänning (2D).
    pub plane_strain: bool,
    /// Meshar 2D-ytor med trianglar i stället för fyrhörningar.
    pub triangles: bool,
    /// Avstånd under vilket noder slås ihop. Sätt före geometrin.
    pub merge_tol: f64,
    pub nodes: Vec<[f64; 3]>,
    pub elements: Vec<Element>,
    pub materials: Vec<Material>,
    pub sections: Vec<Section>,
    pub supports: Vec<Support>,
    pub loads: Vec<Load>,
    pub gravity: Option<[f64; 3]>,
    grid: HashMap<[i64; 3], Vec<usize>>,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

impl Model {
    fn with_dim(dim: Dim) -> Self {
        Model {
            dim,
            plane_strain: false,
            triangles: false,
            merge_tol: 1e-6,
            nodes: Vec::new(),
            elements: Vec::new(),
            materials: Vec::new(),
            sections: Vec::new(),
            supports: Vec::new(),
            loads: Vec::new(),
            gravity: None,
            grid: HashMap::new(),
        }
    }

    /// Plan modell (x-y). Plan spänning om inte `plane_strain` sätts.
    pub fn new_2d() -> Self {
        Self::with_dim(Dim::Two)
    }

    pub fn new_3d() -> Self {
        Self::with_dim(Dim::Three)
    }

    /// Lägger till (eller ersätter, vid samma namn) ett material.
    pub fn material(&mut self, mat: Material) -> usize {
        if let Some(i) = self.materials.iter().position(|m| m.name == mat.name) {
            self.materials[i] = mat;
            i
        } else {
            self.materials.push(mat);
            self.materials.len() - 1
        }
    }

    fn cell(&self, p: &[f64; 3]) -> [i64; 3] {
        let h = self.merge_tol * 4.0;
        [(p[0] / h).floor() as i64, (p[1] / h).floor() as i64, (p[2] / h).floor() as i64]
    }

    /// Lägger till en nod, eller returnerar en befintlig nod på samma plats.
    pub fn add_node(&mut self, mut p: [f64; 3]) -> usize {
        if self.dim == Dim::Two {
            p[2] = 0.0;
        }
        let c = self.cell(&p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = self.grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                        for &i in list {
                            if norm(sub(self.nodes[i], p)) <= self.merge_tol {
                                return i;
                            }
                        }
                    }
                }
            }
        }
        let id = self.nodes.len();
        self.nodes.push(p);
        self.grid.entry(c).or_default().push(id);
        id
    }

    pub fn add_section(&mut self, s: Section) -> usize {
        self.sections.push(s);
        self.sections.len() - 1
    }

    pub fn add_element(&mut self, kind: ElemKind, nodes: Vec<usize>, section: usize) -> usize {
        self.elements.push(Element { kind, nodes, section });
        self.elements.len() - 1
    }

    fn check_material(&self, material: usize) -> Result<()> {
        if material >= self.materials.len() {
            return fail(format!("material nr {material} finns inte – lägg till det med Model::material först"));
        }
        Ok(())
    }

    /// Rektangulär platta mellan hörnen `p0` och `p1`, meshad med elementstorlek `size`.
    pub fn plate(&mut self, p0: [f64; 2], p1: [f64; 2], thickness: f64, size: f64, material: usize) -> Result<Range<usize>> {
        let (x0, x1) = (p0[0].min(p1[0]), p0[0].max(p1[0]));
        let (y0, y1) = (p0[1].min(p1[1]), p0[1].max(p1[1]));
        self.quad_region([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], thickness, size, material)
    }

    /// Godtycklig fyrhörning (fyra hörn i följd), strukturerat meshad.
    pub fn quad_region(&mut self, corners: [[f64; 2]; 4], thickness: f64, size: f64, material: usize) -> Result<Range<usize>> {
        if self.dim != Dim::Two {
            return fail("plattor (plate/quad) kräver en 2D-modell – använd box för 3D");
        }
        if !(thickness > 0.0) || !(size > 0.0) {
            return fail("tjocklek och elementstorlek måste vara större än noll");
        }
        self.check_material(material)?;
        let mut c = corners;
        let area2: f64 = (0..4).map(|i| c[i][0] * c[(i + 1) % 4][1] - c[(i + 1) % 4][0] * c[i][1]).sum();
        if area2.abs() < 1e-300 {
            return fail("fyrhörningen har ingen area");
        }
        if area2 < 0.0 {
            c.swap(1, 3);
        }
        let len = |a: [f64; 2], b: [f64; 2]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt();
        let nx = ((len(c[0], c[1]).max(len(c[3], c[2])) / size).ceil() as usize).max(1);
        let ny = ((len(c[0], c[3]).max(len(c[1], c[2])) / size).ceil() as usize).max(1);
        let sec = self.add_section(Section::Solid { material, thickness });
        let mut ids = vec![0; (nx + 1) * (ny + 1)];
        for j in 0..=ny {
            for i in 0..=nx {
                let s = i as f64 / nx as f64;
                let t = j as f64 / ny as f64;
                let w = [(1.0 - s) * (1.0 - t), s * (1.0 - t), s * t, (1.0 - s) * t];
                let x: f64 = (0..4).map(|k| w[k] * c[k][0]).sum();
                let y: f64 = (0..4).map(|k| w[k] * c[k][1]).sum();
                ids[j * (nx + 1) + i] = self.add_node([x, y, 0.0]);
            }
        }
        let start = self.elements.len();
        for j in 0..ny {
            for i in 0..nx {
                let n0 = ids[j * (nx + 1) + i];
                let n1 = ids[j * (nx + 1) + i + 1];
                let n2 = ids[(j + 1) * (nx + 1) + i + 1];
                let n3 = ids[(j + 1) * (nx + 1) + i];
                if self.triangles {
                    self.add_element(ElemKind::CPS3, vec![n0, n1, n2], sec);
                    self.add_element(ElemKind::CPS3, vec![n0, n2, n3], sec);
                } else {
                    self.add_element(ElemKind::CPS4, vec![n0, n1, n2, n3], sec);
                }
            }
        }
        Ok(start..self.elements.len())
    }

    /// Rätblock mellan hörnen `p0` och `p1`, meshat med hexaedrar (C3D8).
    pub fn box_solid(&mut self, p0: [f64; 3], p1: [f64; 3], size: f64, material: usize) -> Result<Range<usize>> {
        if self.dim != Dim::Three {
            return fail("box kräver en 3D-modell");
        }
        if !(size > 0.0) {
            return fail("elementstorlek måste vara större än noll");
        }
        self.check_material(material)?;
        let lo = [p0[0].min(p1[0]), p0[1].min(p1[1]), p0[2].min(p1[2])];
        let hi = [p0[0].max(p1[0]), p0[1].max(p1[1]), p0[2].max(p1[2])];
        let n: Vec<usize> = (0..3).map(|a| (((hi[a] - lo[a]) / size).ceil() as usize).max(1)).collect();
        if (0..3).any(|a| hi[a] - lo[a] <= 0.0) {
            return fail("box måste ha utsträckning i alla tre riktningar");
        }
        let sec = self.add_section(Section::Solid { material, thickness: 1.0 });
        let idx = |i: usize, j: usize, k: usize| (k * (n[1] + 1) + j) * (n[0] + 1) + i;
        let mut ids = vec![0; (n[0] + 1) * (n[1] + 1) * (n[2] + 1)];
        for k in 0..=n[2] {
            for j in 0..=n[1] {
                for i in 0..=n[0] {
                    let p = [
                        lo[0] + (hi[0] - lo[0]) * i as f64 / n[0] as f64,
                        lo[1] + (hi[1] - lo[1]) * j as f64 / n[1] as f64,
                        lo[2] + (hi[2] - lo[2]) * k as f64 / n[2] as f64,
                    ];
                    ids[idx(i, j, k)] = self.add_node(p);
                }
            }
        }
        let start = self.elements.len();
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let nodes = vec![
                        ids[idx(i, j, k)],
                        ids[idx(i + 1, j, k)],
                        ids[idx(i + 1, j + 1, k)],
                        ids[idx(i, j + 1, k)],
                        ids[idx(i, j, k + 1)],
                        ids[idx(i + 1, j, k + 1)],
                        ids[idx(i + 1, j + 1, k + 1)],
                        ids[idx(i, j + 1, k + 1)],
                    ];
                    self.add_element(ElemKind::C3D8, nodes, sec);
                }
            }
        }
        Ok(start..self.elements.len())
    }

    /// Balk från `a` till `b` uppdelad i `divisions` element (B23 i 2D, B33 i 3D).
    ///
    /// Tvärsnittets höjd (i11) ligger i riktningen `up`: standard är y i 2D och
    /// z i 3D (x för vertikala balkar i 3D).
    pub fn beam(
        &mut self,
        a: [f64; 3],
        b: [f64; 3],
        profile: BeamSection,
        material: usize,
        divisions: usize,
        up: Option<[f64; 3]>,
    ) -> Result<Range<usize>> {
        self.check_material(material)?;
        let (a, b) = if self.dim == Dim::Two { ([a[0], a[1], 0.0], [b[0], b[1], 0.0]) } else { (a, b) };
        let d = sub(b, a);
        let l = norm(d);
        if l <= self.merge_tol {
            return fail("balken har ingen längd");
        }
        let t = scale(d, 1.0 / l);
        let n1 = if self.dim == Dim::Two {
            [0.0, 0.0, -1.0]
        } else {
            let mut upv = up.unwrap_or([0.0, 0.0, 1.0]);
            if norm(upv) == 0.0 {
                return fail("up-riktningen får inte vara nollvektorn");
            }
            upv = scale(upv, 1.0 / norm(upv));
            if dot(upv, t).abs() > 0.999 {
                if up.is_some() {
                    return fail("up-riktningen är parallell med balken");
                }
                upv = [1.0, 0.0, 0.0];
            }
            let n2 = sub(upv, scale(t, dot(upv, t)));
            let n2 = scale(n2, 1.0 / norm(n2));
            cross(n2, t)
        };
        let kind = if self.dim == Dim::Two { ElemKind::B23 } else { ElemKind::B33 };
        let sec = self.add_section(Section::Beam { material, profile, n1 });
        let n = divisions.max(1);
        let ids: Vec<usize> = (0..=n).map(|k| self.add_node([0usize, 1, 2].map(|i| a[i] + d[i] * k as f64 / n as f64))).collect();
        let start = self.elements.len();
        for k in 0..n {
            self.add_element(kind, vec![ids[k], ids[k + 1]], sec);
        }
        Ok(start..self.elements.len())
    }

    /// Stång (endast axialkraft) från `a` till `b`.
    pub fn truss(&mut self, a: [f64; 3], b: [f64; 3], area: f64, material: usize) -> Result<usize> {
        self.check_material(material)?;
        if !(area > 0.0) {
            return fail("stångens area måste vara större än noll");
        }
        let na = self.add_node(a);
        let nb = self.add_node(b);
        if na == nb {
            return fail("stången har ingen längd");
        }
        let kind = if self.dim == Dim::Two { ElemKind::T2D2 } else { ElemKind::T3D2 };
        let sec = self.add_section(Section::Truss { material, area });
        Ok(self.add_element(kind, vec![na, nb], sec))
    }

    /// Inspänning: alla frihetsgrader låsta.
    pub fn fix(&mut self, sel: Sel) {
        self.supports.push(Support { sel, fixity: Fixity::Fixed });
    }

    /// Led: translationer låsta, rotationer fria.
    pub fn pin(&mut self, sel: Sel) {
        self.supports.push(Support { sel, fixity: Fixity::Pinned });
    }

    /// Valda frihetsgrader med föreskrivna värden, t.ex. `&[(UY, 0.0)]`.
    pub fn support(&mut self, sel: Sel, dofs: &[(usize, f64)]) {
        self.supports.push(Support { sel, fixity: Fixity::Dofs(dofs.to_vec()) });
    }

    /// Total kraft på urvalet. Över ytor/kanter fördelas den som jämnt tryck.
    pub fn force(&mut self, sel: Sel, f: [f64; 3]) {
        self.loads.push(Load::Force { sel, f: [f[0], f[1], f[2], 0.0, 0.0, 0.0] });
    }

    /// Totalt moment på urvalet (kräver balknoder).
    pub fn moment(&mut self, sel: Sel, m: [f64; 3]) {
        self.loads.push(Load::Force { sel, f: [0.0, 0.0, 0.0, m[0], m[1], m[2]] });
    }

    pub fn pressure(&mut self, sel: Sel, p: f64) {
        self.loads.push(Load::Pressure { sel, p });
    }

    /// Tyngdacceleration, t.ex. `[0.0, -9.81, 0.0]`. Kräver densitet på materialen.
    pub fn set_gravity(&mut self, g: [f64; 3]) {
        self.gravity = Some(g);
    }

    /// Faktisk elementtyp (plan spänning/töjning avgörs av modellen).
    pub fn elem_kind(&self, e: &Element) -> ElemKind {
        match (e.kind, self.plane_strain) {
            (ElemKind::CPS4, true) => ElemKind::CPE4,
            (ElemKind::CPS3, true) => ElemKind::CPE3,
            (k, _) => k,
        }
    }

    pub fn bbox(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &self.nodes {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        (lo, hi)
    }

    fn sel_tol(&self) -> f64 {
        let (lo, hi) = self.bbox();
        (norm(sub(hi, lo)) * 1e-6).max(self.merge_tol)
    }

    /// Noderna som ett urval träffar.
    pub fn select(&self, sel: &Sel) -> Result<Vec<usize>> {
        let tol = self.sel_tol();
        let n = self.nodes.len();
        let v: Vec<usize> = match sel {
            Sel::All => (0..n).collect(),
            Sel::Box(conds) => (0..n)
                .filter(|&i| conds.iter().all(|&(ax, lo, hi)| self.nodes[i][ax] >= lo - tol && self.nodes[i][ax] <= hi + tol))
                .collect(),
            Sel::Point(p) => {
                let v: Vec<usize> = (0..n).filter(|&i| norm(sub(self.nodes[i], *p)) <= tol).collect();
                if v.is_empty() {
                    if let Some(near) = (0..n).min_by(|&a, &b| {
                        norm(sub(self.nodes[a], *p)).partial_cmp(&norm(sub(self.nodes[b], *p))).unwrap()
                    }) {
                        let q = self.nodes[near];
                        return fail(format!(
                            "ingen nod vid ({}, {}, {}); närmaste nod ligger vid ({}, {}, {})",
                            p[0], p[1], p[2], q[0], q[1], q[2]
                        ));
                    }
                }
                v
            }
            Sel::Nodes(ids) => {
                if let Some(&bad) = ids.iter().find(|&&i| i >= n) {
                    return fail(format!("nod {bad} finns inte"));
                }
                ids.clone()
            }
        };
        if v.is_empty() {
            return fail(format!("urvalet '{}' träffade inga noder", sel.describe()));
        }
        Ok(v)
    }

    pub fn check(&self) -> Result<()> {
        if self.elements.is_empty() {
            return fail("modellen har inga element – lägg till geometri (plate, quad, box, beam, truss)");
        }
        for s in &self.sections {
            self.check_material(s.material())?;
        }
        Ok(())
    }

    /// Bitmask per nod över vilka frihetsgrader som finns (från anslutna element).
    pub fn active_dofs(&self) -> Vec<u8> {
        let mut act = vec![0u8; self.nodes.len()];
        for e in &self.elements {
            let mut mask = 0u8;
            for &d in self.elem_kind(e).node_dofs() {
                mask |= 1 << d;
            }
            for &n in &e.nodes {
                act[n] |= mask;
            }
        }
        act
    }

    /// Alla låsta frihetsgrader som (nod, frihetsgrad, värde).
    pub fn constraints(&self, active: &[u8]) -> Result<Vec<(usize, usize, f64)>> {
        let mut map: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for s in &self.supports {
            let nodes = self.select(&s.sel)?;
            for &n in &nodes {
                let on = |d: usize| active[n] & (1 << d) != 0;
                match &s.fixity {
                    Fixity::Fixed => (0..6).filter(|&d| on(d)).for_each(|d| {
                        map.insert((n, d), 0.0);
                    }),
                    Fixity::Pinned => (0..3).filter(|&d| on(d)).for_each(|d| {
                        map.insert((n, d), 0.0);
                    }),
                    Fixity::Dofs(list) => {
                        for &(d, v) in list {
                            if self.dim == Dim::Two && matches!(d, UZ | RX | RY) {
                                return fail(format!("frihetsgraden {} finns inte i en 2D-modell", DOF_NAMES[d]));
                            }
                            if on(d) {
                                map.insert((n, d), v);
                            }
                        }
                    }
                }
            }
        }
        if map.is_empty() {
            return fail("modellen saknar stöd – lägg till t.ex. fix/pin/support");
        }
        Ok(map.into_iter().map(|((n, d), v)| (n, d, v)).collect())
    }

    fn facet_table(kind: ElemKind) -> &'static [&'static [usize]] {
        use ElemKind::*;
        match kind {
            CPS3 | CPE3 => &[&[0, 1], &[1, 2], &[2, 0]],
            CPS4 | CPE4 => &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
            C3D8 => &[&[0, 3, 2, 1], &[4, 5, 6, 7], &[0, 1, 5, 4], &[1, 2, 6, 5], &[2, 3, 7, 6], &[3, 0, 4, 7]],
            _ => &[],
        }
    }

    pub(crate) fn boundary_facets(&self) -> Vec<Facet> {
        let mut seen: HashMap<Vec<usize>, (usize, Facet)> = HashMap::new();
        for (ei, e) in self.elements.iter().enumerate() {
            for loc in Self::facet_table(e.kind) {
                let nodes: Vec<usize> = loc.iter().map(|&k| e.nodes[k]).collect();
                let mut key = nodes.clone();
                key.sort_unstable();
                seen.entry(key).and_modify(|v| v.0 += 1).or_insert((1, Facet { nodes, elem: ei }));
            }
        }
        let mut v: Vec<Facet> = seen.into_values().filter(|(c, _)| *c == 1).map(|(_, f)| f).collect();
        v.sort_by_key(|f| (f.elem, f.nodes[0]));
        v
    }

    /// För varje nod på randytan: (nod, ∫N dA, ∫N n dA) med utåtriktad normal n.
    fn facet_integrals(&self, f: &Facet) -> Vec<(usize, f64, [f64; 3])> {
        let e = &self.elements[f.elem];
        let ne = e.nodes.len() as f64;
        let mut c = [0.0; 3];
        for &n in &e.nodes {
            for a in 0..3 {
                c[a] += self.nodes[n][a] / ne;
            }
        }
        let t = match (&self.sections[e.section], self.dim) {
            (Section::Solid { thickness, .. }, Dim::Two) => *thickness,
            _ => 1.0,
        };
        let x: Vec<[f64; 3]> = f.nodes.iter().map(|&n| self.nodes[n]).collect();
        if x.len() == 2 {
            let d = sub(x[1], x[0]);
            let l = norm(d);
            let mut nrm = [d[1] / l, -d[0] / l, 0.0];
            let mid = scale([x[0][0] + x[1][0], x[0][1] + x[1][1], 0.0], 0.5);
            if dot(sub(mid, c), nrm) < 0.0 {
                nrm = scale(nrm, -1.0);
            }
            let w = l * t / 2.0;
            return f.nodes.iter().map(|&n| (n, w, scale(nrm, w))).collect();
        }
        const G: f64 = 0.577_350_269_189_625_8;
        let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
        let mut w = [0.0; 4];
        let mut nw = [[0.0; 3]; 4];
        let mut total = [0.0; 3];
        for &(xi, eta) in &[(-G, -G), (G, -G), (-G, G), (G, G)] {
            let mut dxi = [0.0; 3];
            let mut deta = [0.0; 3];
            let mut nv = [0.0; 4];
            for i in 0..4 {
                nv[i] = 0.25 * (1.0 + xi * nat[i][0]) * (1.0 + eta * nat[i][1]);
                let dn_dxi = 0.25 * nat[i][0] * (1.0 + eta * nat[i][1]);
                let dn_deta = 0.25 * nat[i][1] * (1.0 + xi * nat[i][0]);
                for a in 0..3 {
                    dxi[a] += dn_dxi * x[i][a];
                    deta[a] += dn_deta * x[i][a];
                }
            }
            let av = cross(dxi, deta);
            for i in 0..4 {
                w[i] += nv[i] * norm(av);
                for a in 0..3 {
                    nw[i][a] += nv[i] * av[a];
                }
            }
            for a in 0..3 {
                total[a] += av[a];
            }
        }
        let fc = scale([0usize, 1, 2].map(|a| x.iter().map(|p| p[a]).sum::<f64>()), 0.25);
        let sign = if dot(sub(fc, c), total) < 0.0 { -1.0 } else { 1.0 };
        (0..4).map(|i| (f.nodes[i], w[i], scale(nw[i], sign))).collect()
    }

    /// Ekvivalenta nodlaster från alla laster och tyngdkraft.
    pub fn nodal_loads(&self, active: &[u8]) -> Result<Vec<[f64; 6]>> {
        let mut f = vec![[0.0; 6]; self.nodes.len()];
        let mut facets: Option<Vec<Facet>> = None;
        for load in &self.loads {
            match load {
                Load::Force { sel, f: ff } => {
                    if self.dim == Dim::Two && (ff[2] != 0.0 || ff[3] != 0.0 || ff[4] != 0.0) {
                        return fail("fz, mx och my finns inte i en 2D-modell");
                    }
                    let nodes = self.select(sel)?;
                    let set: HashSet<usize> = nodes.iter().copied().collect();
                    if ff[..3].iter().any(|&v| v != 0.0) {
                        let all = facets.get_or_insert_with(|| self.boundary_facets());
                        let mut weights: BTreeMap<usize, f64> = BTreeMap::new();
                        for fac in all.iter().filter(|fc| fc.nodes.iter().all(|n| set.contains(n))) {
                            for (n, w, _) in self.facet_integrals(fac) {
                                *weights.entry(n).or_default() += w;
                            }
                        }
                        if weights.is_empty() {
                            for &n in &nodes {
                                weights.insert(n, 1.0);
                            }
                        }
                        let total: f64 = weights.values().sum();
                        for (&n, &w) in &weights {
                            for d in 0..3 {
                                f[n][d] += ff[d] * w / total;
                            }
                        }
                    }
                    for d in 3..6 {
                        if ff[d] == 0.0 {
                            continue;
                        }
                        let rot: Vec<usize> = nodes.iter().copied().filter(|&n| active[n] & (1 << d) != 0).collect();
                        if rot.is_empty() {
                            return fail(format!(
                                "momentet {} på '{}' kräver noder med rotationsfrihetsgrad (balkar)",
                                ["mx", "my", "mz"][d - 3],
                                sel.describe()
                            ));
                        }
                        for &n in &rot {
                            f[n][d] += ff[d] / rot.len() as f64;
                        }
                    }
                }
                Load::Pressure { sel, p } => {
                    let set: HashSet<usize> = self.select(sel)?.into_iter().collect();
                    let all = facets.get_or_insert_with(|| self.boundary_facets());
                    let mut any = false;
                    for fac in all.iter().filter(|fc| fc.nodes.iter().all(|n| set.contains(n))) {
                        any = true;
                        for (n, _, nw) in self.facet_integrals(fac) {
                            for d in 0..3 {
                                f[n][d] -= p * nw[d];
                            }
                        }
                    }
                    if !any {
                        return fail(format!("tryck på '{}' kräver en randkant/randyta på plattor eller block", sel.describe()));
                    }
                }
            }
        }
        if let Some(g) = self.gravity {
            let mut any = false;
            for (ei, e) in self.elements.iter().enumerate() {
                let sec = &self.sections[e.section];
                let rho = self.materials[sec.material()].density;
                if rho == 0.0 {
                    continue;
                }
                any = true;
                let w = crate::legacy::volume_weights(self, ei)?;
                for (k, &n) in e.nodes.iter().enumerate() {
                    for d in 0..3 {
                        f[n][d] += rho * g[d] * w[k];
                    }
                }
            }
            if !any {
                return fail("tyngdkraft kräver att minst ett material har densitet (rho)");
            }
        }
        Ok(f)
    }

    pub fn solve(&self) -> Result<Results> {
        crate::legacy::solve(self)
    }
}
