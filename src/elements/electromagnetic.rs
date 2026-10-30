//! Elektromagnetiska element (EMC3D4, EMC2D3): Nédélecs kantelement av
//! lägsta ordningen (första sorten) för den magnetiska vektorpotentialen A.
//! EMC2D3 är en plan triangel med A i planet (∇×A vinkelrätt mot planet) och
//! tjockleken från *SOLID SECTION.
//!
//! Frihetsgraderna är A:s tangentiella komponent längs elementets kanter
//! (en per kant, inga nodfrihetsgrader). Med barycentriska koordinater λᵢ är
//! kantfunktionen för kanten a → b (global riktning från lägre till högre
//! nodindex)
//!
//!   W = λ_a ∇λ_b − λ_b ∇λ_a,   ∇×W = 2 ∇λ_a × ∇λ_b (konstant),
//!
//! och elementmatriserna för ∇×(ν∇×A) + iωσA = J:
//!
//!   K_ef = ν V (∇×W_e)·(∇×W_f),  M_ef = σ ∫ W_e·W_f dV (exakt, ∫λᵢλⱼ = V(1+δᵢⱼ)/20),
//!   f_e  = J·∫W_e dV = J·V/4 (∇λ_b − ∇λ_a).
//!
//! Steg-typen i `steps/electromagnetic.rs` assemblerar och löser.

use super::api::{ElementType, Family};
use crate::fem::dof::Dof;
use crate::model::{fail, Result};

pub struct EmElem {
    pub name: &'static str,
    /// Plan triangel (EMC2D3) i stället för tetraeder.
    pub planar: bool,
}

pub static EMC3D4: EmElem = EmElem { name: "EMC3D4", planar: false };
pub static EMC2D3: EmElem = EmElem { name: "EMC2D3", planar: true };
pub static TYPES: &[&dyn ElementType] = &[&EMC2D3, &EMC3D4];

/// Det elektromagnetiska elementet `name`.
pub fn lookup(name: &str) -> Option<&'static EmElem> {
    [&EMC2D3, &EMC3D4].into_iter().find(|e| e.name == name)
}

/// Lokala kanter (nodpar).
pub const TET_EDGES: [(usize, usize); 6] = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)];
pub const TRI_EDGES: [(usize, usize); 3] = [(0, 1), (1, 2), (2, 0)];
const TET4_FACES: &[&[usize]] = &[&[0, 1, 2], &[0, 3, 1], &[1, 3, 2], &[2, 3, 0]];
const TRI3_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 0]];

impl EmElem {
    pub fn edges(&self) -> &'static [(usize, usize)] {
        if self.planar {
            &TRI_EDGES
        } else {
            &TET_EDGES
        }
    }

    /// Geometrin (2D: arean gånger tjockleken som volym, ∇λ i planet).
    pub fn geom(&self, x: &[[f64; 3]], thickness: f64) -> Result<TetGeom> {
        if !self.planar {
            return tet_geom(x);
        }
        let (x1, y1) = (x[1][0] - x[0][0], x[1][1] - x[0][1]);
        let (x2, y2) = (x[2][0] - x[0][0], x[2][1] - x[0][1]);
        let det = x1 * y2 - x2 * y1;
        if det <= 0.0 {
            return fail("triangeln har negativ eller noll area (nodordningen)");
        }
        let mut grad = [[0.0; 3]; 4];
        grad[1] = [y2 / det, -x2 / det, 0.0];
        grad[2] = [-y1 / det, x1 / det, 0.0];
        grad[0] = [-(grad[1][0] + grad[2][0]), -(grad[1][1] + grad[2][1]), 0.0];
        Ok(TetGeom { vol: 0.5 * det * thickness, grad, dim: 2 })
    }
}

/// Tetraederns geometri: volym och ∇λᵢ.
pub struct TetGeom {
    pub vol: f64,
    pub grad: [[f64; 3]; 4],
    /// 3 för tetraeder, 2 för triangel (∫λᵢλⱼ = A(1+δᵢⱼ)/12).
    pub dim: usize,
}

pub fn tet_geom(x: &[[f64; 3]]) -> Result<TetGeom> {
    let d = |i: usize| [x[i][0] - x[0][0], x[i][1] - x[0][1], x[i][2] - x[0][2]];
    let j = [d(1), d(2), d(3)];
    let (inv, det) = super::common::inv3([[j[0][0], j[1][0], j[2][0]], [j[0][1], j[1][1], j[2][1]], [j[0][2], j[1][2], j[2][2]]]);
    if det <= 0.0 {
        return fail("tetraedern har negativ eller noll volym (nodordningen)");
    }
    // λ₁..λ₃ = rader i J⁻¹ (J kolumnvis = kanter), λ₀ = 1 − Σ
    let mut grad = [[0.0; 3]; 4];
    for k in 0..3 {
        for c in 0..3 {
            grad[k + 1][c] = inv[k][c];
        }
    }
    for c in 0..3 {
        grad[0][c] = -(grad[1][c] + grad[2][c] + grad[3][c]);
    }
    Ok(TetGeom { vol: det / 6.0, grad, dim: 3 })
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Kantens (a, b) i elementets riktning: rotorn och ∫W dV / V.
pub struct EdgeFns {
    pub curl: Vec<[f64; 3]>,
    pub mean: Vec<[f64; 3]>,
    /// ∫ W_e·W_f dV.
    pub mass: Vec<Vec<f64>>,
}

pub fn edge_fns(g: &TetGeom, edges: &[(usize, usize)]) -> EdgeFns {
    let ne = edges.len();
    let mut curl = vec![[0.0; 3]; ne];
    let mut mean = vec![[0.0; 3]; ne];
    let cm = if g.dim == 2 { 1.0 / 3.0 } else { 0.25 };
    for (e, &(a, b)) in edges.iter().enumerate() {
        let c = cross(g.grad[a], g.grad[b]);
        curl[e] = [2.0 * c[0], 2.0 * c[1], 2.0 * c[2]];
        mean[e] = [0, 1, 2].map(|k| cm * (g.grad[b][k] - g.grad[a][k]));
    }
    let den = if g.dim == 2 { 12.0 } else { 20.0 };
    let l = |i: usize, j: usize| g.vol * if i == j { 2.0 } else { 1.0 } / den;
    let mut mass = vec![vec![0.0; ne]; ne];
    for (e, &(a, b)) in edges.iter().enumerate() {
        for (f, &(c, d)) in edges.iter().enumerate() {
            // (λa∇λb − λb∇λa)·(λc∇λd − λd∇λc)
            mass[e][f] = l(a, c) * dot(g.grad[b], g.grad[d]) - l(a, d) * dot(g.grad[b], g.grad[c]) - l(b, c) * dot(g.grad[a], g.grad[d]) + l(b, d) * dot(g.grad[a], g.grad[c]);
        }
    }
    EdgeFns { curl, mean, mass }
}

impl ElementType for EmElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        if self.planar {
            2
        } else {
            3
        }
    }
    fn n_nodes(&self) -> usize {
        if self.planar {
            3
        } else {
            4
        }
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[]
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        if self.planar {
            5
        } else {
            10
        }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.planar {
            TRI3_FACES
        } else {
            TET4_FACES
        }
    }
}
