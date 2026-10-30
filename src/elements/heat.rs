//! Heat conduction elements (diffusion elements) with one degree of freedom, NT11, per node:
//! DC2D3, DC2D4, DC2D8, DCAX3, DCAX4, DCAX8, DC3D4, DC3D6, DC3D8, DC3D10 and DC3D20.
//!
//! Formulation (Galerkin, see e.g. Zienkiewicz–Taylor, the chapter on heat conduction,
//! or Bathe 7.2):
//!
//! - conduction: f_i = ∫ ∇N_i · k(T) ∇T dV, tangent ∫ ∇N_i·k ∇N_j + ∇N_i·(dk/dT ∇T) N_j dV;
//! - heat capacity (transient, backward Euler): f_i = ∫ N_i ρ c(T) (T − T_n)/Δt dV
//!   with c at the end of the increment. First-order elements have
//!   diagonal capacity (see `cap_ips`), second-order ones consistent;
//! - fluxes: *DFLUX BF (per volume) and Sn (per area, positive into the body),
//!   *FILM q = h (T∞ − T) and *RADIATE q = σε((T∞−Tz)⁴ − (T−Tz)⁴) on the surfaces.
//!
//! The integration points are numbered with ξ fastest. Plane elements
//! use the thickness from *SOLID SECTION; axisymmetric ones (x = r, y = z)
//! integrate over the full revolution (2π r), so nodal fluxes and reactions apply to the whole
//! circumference.
//!
//! The step type in `steps/heat.rs` calls the methods on `HeatElem` directly (via
//! `lookup`), since the thermal properties of the material are read from
//! the material definition and not via `MaterialModel`.

use super::api::{ElemCtx, ElementType, Family, IpField};
use super::common::{inv3, Ip, DISTORTED};
use crate::fem::dof::{Dof, NT11};
use crate::materials::heat::HeatProps;
use crate::model::{fail, Result};

/// The element geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Tri3,
    Quad4,
    Quad8,
    Tet4,
    Tet10,
    Wedge6,
    Hex8,
    Hex20,
}

pub struct HeatElem {
    pub name: &'static str,
    pub shape: Shape,
    pub axisym: bool,
}

pub static DC2D3: HeatElem = HeatElem { name: "DC2D3", shape: Shape::Tri3, axisym: false };
pub static DC2D4: HeatElem = HeatElem { name: "DC2D4", shape: Shape::Quad4, axisym: false };
pub static DC2D8: HeatElem = HeatElem { name: "DC2D8", shape: Shape::Quad8, axisym: false };
pub static DCAX3: HeatElem = HeatElem { name: "DCAX3", shape: Shape::Tri3, axisym: true };
pub static DCAX4: HeatElem = HeatElem { name: "DCAX4", shape: Shape::Quad4, axisym: true };
pub static DCAX8: HeatElem = HeatElem { name: "DCAX8", shape: Shape::Quad8, axisym: true };
pub static DC3D4: HeatElem = HeatElem { name: "DC3D4", shape: Shape::Tet4, axisym: false };
pub static DC3D6: HeatElem = HeatElem { name: "DC3D6", shape: Shape::Wedge6, axisym: false };
pub static DC3D8: HeatElem = HeatElem { name: "DC3D8", shape: Shape::Hex8, axisym: false };
pub static DC3D10: HeatElem = HeatElem { name: "DC3D10", shape: Shape::Tet10, axisym: false };
pub static DC3D20: HeatElem = HeatElem { name: "DC3D20", shape: Shape::Hex20, axisym: false };

pub static TYPES: &[&dyn ElementType] = &[&DC2D3, &DC2D4, &DC2D8, &DCAX3, &DCAX4, &DCAX8, &DC3D4, &DC3D6, &DC3D8, &DC3D10, &DC3D20];

static ALL: &[&HeatElem] = &[&DC2D3, &DC2D4, &DC2D8, &DCAX3, &DCAX4, &DCAX8, &DC3D4, &DC3D6, &DC3D8, &DC3D10, &DC3D20];

/// The heat conduction element named `name`, or the heat conduction part
/// of a coupled temperature–displacement element (C3D8T, CPE4T, ...).
pub fn lookup(name: &str) -> Option<&'static HeatElem> {
    ALL.iter().copied().find(|e| e.name == name).or_else(|| super::heat_coupled::thermal(name))
        // area coupled2: Q3D4, Q3D8, Q3D20
        .or_else(|| super::thermal_electric_structural::thermal(name))
}

// ---------------------------------------------------------------------------
// Faces (the .inp format numbering). Second-order edges/faces: corners first, then midside nodes.
// ---------------------------------------------------------------------------

const TRI3_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 0]];
const QUAD4_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
const QUAD8_FACES: &[&[usize]] = &[&[0, 1, 4], &[1, 2, 5], &[2, 3, 6], &[3, 0, 7]];
const TET4_FACES: &[&[usize]] = &[&[0, 1, 2], &[0, 3, 1], &[1, 3, 2], &[2, 3, 0]];
const TET10_FACES: &[&[usize]] = &[&[0, 1, 2, 4, 5, 6], &[0, 3, 1, 7, 8, 4], &[1, 3, 2, 8, 9, 5], &[2, 3, 0, 9, 7, 6]];
const WEDGE6_FACES: &[&[usize]] = &[&[0, 1, 2], &[3, 5, 4], &[0, 3, 4, 1], &[1, 4, 5, 2], &[2, 5, 3, 0]];
const HEX8_FACES: &[&[usize]] = &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]];
const HEX20_FACES: &[&[usize]] = &[
    &[0, 1, 2, 3, 8, 9, 10, 11],
    &[4, 7, 6, 5, 15, 14, 13, 12],
    &[0, 4, 5, 1, 16, 12, 17, 8],
    &[1, 5, 6, 2, 17, 13, 18, 9],
    &[2, 6, 7, 3, 18, 14, 19, 10],
    &[3, 7, 4, 0, 19, 15, 16, 11],
];

// ---------------------------------------------------------------------------
// Shape functions in natural coordinates
// ---------------------------------------------------------------------------

const QUAD_NAT: [[f64; 2]; 8] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];

const HEX_NAT: [[f64; 3]; 20] = [
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

/// Shape functions and derivatives d/dξ for a geometry at the point `p`.
pub fn shape_fn(shape: Shape, p: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (xi, eta, zeta) = (p[0], p[1], p[2]);
    match shape {
        Shape::Tri3 => (vec![1.0 - xi - eta, xi, eta], vec![[-1.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        Shape::Quad4 => {
            let mut n = vec![0.0; 4];
            let mut d = vec![[0.0; 3]; 4];
            for i in 0..4 {
                let [a, b] = QUAD_NAT[i];
                n[i] = 0.25 * (1.0 + a * xi) * (1.0 + b * eta);
                d[i] = [0.25 * a * (1.0 + b * eta), 0.25 * b * (1.0 + a * xi), 0.0];
            }
            (n, d)
        }
        Shape::Quad8 => {
            let mut n = vec![0.0; 8];
            let mut d = vec![[0.0; 3]; 8];
            for i in 0..8 {
                let [a, b] = QUAD_NAT[i];
                if i < 4 {
                    n[i] = 0.25 * (1.0 + a * xi) * (1.0 + b * eta) * (a * xi + b * eta - 1.0);
                    d[i] = [
                        0.25 * a * (1.0 + b * eta) * (2.0 * a * xi + b * eta),
                        0.25 * b * (1.0 + a * xi) * (a * xi + 2.0 * b * eta),
                        0.0,
                    ];
                } else if a == 0.0 {
                    n[i] = 0.5 * (1.0 - xi * xi) * (1.0 + b * eta);
                    d[i] = [-xi * (1.0 + b * eta), 0.5 * b * (1.0 - xi * xi), 0.0];
                } else {
                    n[i] = 0.5 * (1.0 + a * xi) * (1.0 - eta * eta);
                    d[i] = [0.5 * a * (1.0 - eta * eta), -eta * (1.0 + a * xi), 0.0];
                }
            }
            (n, d)
        }
        Shape::Tet4 => (
            vec![1.0 - xi - eta - zeta, xi, eta, zeta],
            vec![[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ),
        Shape::Tet10 => {
            let l = [1.0 - xi - eta - zeta, xi, eta, zeta];
            let dl: [[f64; 3]; 4] = [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            let mut n = vec![0.0; 10];
            let mut d = vec![[0.0; 3]; 10];
            for i in 0..4 {
                n[i] = l[i] * (2.0 * l[i] - 1.0);
                for c in 0..3 {
                    d[i][c] = (4.0 * l[i] - 1.0) * dl[i][c];
                }
            }
            for (k, &(a, b)) in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)].iter().enumerate() {
                n[4 + k] = 4.0 * l[a] * l[b];
                for c in 0..3 {
                    d[4 + k][c] = 4.0 * (dl[a][c] * l[b] + l[a] * dl[b][c]);
                }
            }
            (n, d)
        }
        Shape::Wedge6 => {
            // triangle (ξ, η) × line ζ ∈ [−1, 1]
            let l = [1.0 - xi - eta, xi, eta];
            let dl: [[f64; 2]; 3] = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut n = vec![0.0; 6];
            let mut d = vec![[0.0; 3]; 6];
            for (k, s) in [(0usize, -1.0f64), (1, 1.0)] {
                let h = 0.5 * (1.0 + s * zeta);
                for i in 0..3 {
                    n[3 * k + i] = l[i] * h;
                    d[3 * k + i] = [dl[i][0] * h, dl[i][1] * h, l[i] * 0.5 * s];
                }
            }
            (n, d)
        }
        Shape::Hex8 => {
            let mut n = vec![0.0; 8];
            let mut d = vec![[0.0; 3]; 8];
            for i in 0..8 {
                let [a, b, c] = HEX_NAT[i];
                let f = [1.0 + a * xi, 1.0 + b * eta, 1.0 + c * zeta];
                n[i] = 0.125 * f[0] * f[1] * f[2];
                d[i] = [0.125 * a * f[1] * f[2], 0.125 * b * f[0] * f[2], 0.125 * c * f[0] * f[1]];
            }
            (n, d)
        }
        Shape::Hex20 => {
            let mut n = vec![0.0; 20];
            let mut d = vec![[0.0; 3]; 20];
            for i in 0..20 {
                let [a, b, c] = HEX_NAT[i];
                if i < 8 {
                    let f = [1.0 + a * xi, 1.0 + b * eta, 1.0 + c * zeta];
                    let g = a * xi + b * eta + c * zeta - 2.0;
                    n[i] = 0.125 * f[0] * f[1] * f[2] * g;
                    d[i] = [
                        0.125 * a * f[1] * f[2] * (g + f[0]),
                        0.125 * b * f[0] * f[2] * (g + f[1]),
                        0.125 * c * f[0] * f[1] * (g + f[2]),
                    ];
                } else if a == 0.0 {
                    let (fb, fc) = (1.0 + b * eta, 1.0 + c * zeta);
                    n[i] = 0.25 * (1.0 - xi * xi) * fb * fc;
                    d[i] = [-0.5 * xi * fb * fc, 0.25 * b * (1.0 - xi * xi) * fc, 0.25 * c * (1.0 - xi * xi) * fb];
                } else if b == 0.0 {
                    let (fa, fc) = (1.0 + a * xi, 1.0 + c * zeta);
                    n[i] = 0.25 * (1.0 - eta * eta) * fa * fc;
                    d[i] = [0.25 * a * (1.0 - eta * eta) * fc, -0.5 * eta * fa * fc, 0.25 * c * (1.0 - eta * eta) * fa];
                } else {
                    let (fa, fb) = (1.0 + a * xi, 1.0 + b * eta);
                    n[i] = 0.25 * (1.0 - zeta * zeta) * fa * fb;
                    d[i] = [0.25 * a * (1.0 - zeta * zeta) * fb, 0.25 * b * (1.0 - zeta * zeta) * fa, -0.5 * zeta * fa * fb];
                }
            }
            (n, d)
        }
    }
}

pub fn shape_dim(shape: Shape) -> usize {
    match shape {
        Shape::Tri3 | Shape::Quad4 | Shape::Quad8 => 2,
        _ => 3,
    }
}

pub fn shape_nodes(shape: Shape) -> usize {
    match shape {
        Shape::Tri3 => 3,
        Shape::Quad4 | Shape::Tet4 => 4,
        Shape::Wedge6 => 6,
        Shape::Quad8 | Shape::Hex8 => 8,
        Shape::Tet10 => 10,
        Shape::Hex20 => 20,
    }
}

const G3: [f64; 3] = [-0.774_596_669_241_483_4, 0.0, 0.774_596_669_241_483_4];
const W3: [f64; 3] = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0];
const G2: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];

/// Gauss rule (points and weights in natural coordinates).
pub fn gauss_rule(shape: Shape) -> Vec<([f64; 3], f64)> {
    match shape {
        // DC2D3 has three points (the gradient is constant)
        Shape::Tri3 => vec![([1.0 / 6.0, 1.0 / 6.0, 0.0], 1.0 / 6.0), ([2.0 / 3.0, 1.0 / 6.0, 0.0], 1.0 / 6.0), ([1.0 / 6.0, 2.0 / 3.0, 0.0], 1.0 / 6.0)],
        Shape::Quad4 => {
            let mut v = Vec::new();
            for &e in &G2 {
                for &x in &G2 {
                    v.push(([x, e, 0.0], 1.0));
                }
            }
            v
        }
        Shape::Quad8 => {
            let mut v = Vec::new();
            for j in 0..3 {
                for i in 0..3 {
                    v.push(([G3[i], G3[j], 0.0], W3[i] * W3[j]));
                }
            }
            v
        }
        // DC3D4 has four points (the gradient is constant)
        Shape::Tet4 => tet4_rule(),
        // DC3D10: 15-point rule of degree 5 (Keast)
        Shape::Tet10 => {
            let (a1, a2, c) = (0.091_971_078_052_723_03, 0.319_793_627_829_629_9, 0.056_350_832_689_629_16);
            let (b1, b2, d) = (1.0 - 3.0 * a1, 1.0 - 3.0 * a2, 0.5 - c);
            let (w0, w1, w2, w3) = (8.0 / 405.0, 0.011_989_513_963_169_777, 0.011_511_367_871_045_378, 5.0 / 567.0);
            vec![
                ([0.25, 0.25, 0.25], w0),
                ([a1, a1, a1], w1),
                ([b1, a1, a1], w1),
                ([a1, b1, a1], w1),
                ([a1, a1, b1], w1),
                ([a2, a2, a2], w2),
                ([b2, a2, a2], w2),
                ([a2, b2, a2], w2),
                ([a2, a2, b2], w2),
                ([c, c, d], w3),
                ([d, c, c], w3),
                ([d, d, c], w3),
                ([c, d, d], w3),
                ([c, d, c], w3),
                ([d, c, d], w3),
            ]
        }
        Shape::Wedge6 => {
            let t = 1.0 / 3.0;
            vec![([t, t, G2[0]], 0.5), ([t, t, G2[1]], 0.5)]
        }
        Shape::Hex8 => {
            let mut v = Vec::new();
            for &z in &G2 {
                for &e in &G2 {
                    for &x in &G2 {
                        v.push(([x, e, z], 1.0));
                    }
                }
            }
            v
        }
        Shape::Hex20 => {
            let mut v = Vec::new();
            for k in 0..3 {
                for j in 0..3 {
                    for i in 0..3 {
                        v.push(([G3[i], G3[j], G3[k]], W3[i] * W3[j] * W3[k]));
                    }
                }
            }
            v
        }
    }
}

/// Four-point rule for a tetrahedron (degree 2), point i closest to node i.
fn tet4_rule() -> Vec<([f64; 3], f64)> {
    let a = 0.138_196_601_125_010_5;
    let b = 0.585_410_196_624_968_5;
    let w = 1.0 / 24.0;
    vec![([a, a, a], w), ([b, a, a], w), ([a, b, a], w), ([a, a, b], w)]
}

/// Nodal integration (points at the corners) for the capacity term in first-order elements.
fn nodal_rule(shape: Shape) -> Option<Vec<([f64; 3], f64)>> {
    Some(match shape {
        Shape::Tri3 => vec![([0.0, 0.0, 0.0], 1.0 / 6.0), ([1.0, 0.0, 0.0], 1.0 / 6.0), ([0.0, 1.0, 0.0], 1.0 / 6.0)],
        Shape::Quad4 => (0..4).map(|i| ([QUAD_NAT[i][0], QUAD_NAT[i][1], 0.0], 1.0)).collect(),
        Shape::Tet4 => vec![
            ([0.0, 0.0, 0.0], 1.0 / 24.0),
            ([1.0, 0.0, 0.0], 1.0 / 24.0),
            ([0.0, 1.0, 0.0], 1.0 / 24.0),
            ([0.0, 0.0, 1.0], 1.0 / 24.0),
        ],
        Shape::Wedge6 => {
            let c = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
            let mut v = Vec::new();
            for z in [-1.0, 1.0] {
                for p in c {
                    v.push(([p[0], p[1], z], 1.0 / 6.0));
                }
            }
            v
        }
        Shape::Hex8 => (0..8).map(|i| (HEX_NAT[i], 1.0)).collect(),
        _ => return None,
    })
}

/// Integration point at `p` with the Gauss weight `wg`. `x` is the nodal coordinates; for
/// axisymmetry the weight is multiplied by 2π r.
pub fn ip_at(shape: Shape, x: &[[f64; 3]], p: [f64; 3], wg: f64, thickness: f64, axisym: bool) -> Result<Ip> {
    let (n, dn) = shape_fn(shape, p);
    let nn = n.len();
    let dim = shape_dim(shape);
    let mut j = [[0.0; 3]; 3];
    for r in 0..dim {
        for c in 0..dim {
            j[r][c] = (0..nn).map(|i| dn[i][r] * x[i][c]).sum();
        }
    }
    if dim == 2 {
        j[2][2] = 1.0;
    }
    let (ji, det) = inv3(j);
    if det <= 0.0 {
        return fail(DISTORTED);
    }
    let dndx = (0..nn)
        .map(|i| {
            let mut d = [0.0; 3];
            for c in 0..dim {
                d[c] = (0..dim).map(|r| ji[c][r] * dn[i][r]).sum();
            }
            d
        })
        .collect();
    let mut w = det * wg;
    if dim == 2 {
        if axisym {
            let r: f64 = (0..nn).map(|i| n[i] * x[i][0]).sum();
            w *= 2.0 * std::f64::consts::PI * r;
        } else {
            w *= thickness;
        }
    }
    Ok(Ip { n, dndx, w })
}

/// Surface integration point: shape functions for all nodes of the element and weight (dA).
#[derive(Debug, Clone)]
pub struct FacePoint {
    pub n: Vec<f64>,
    pub w: f64,
}

/// Shape functions and tangents for an edge/face with `k` nodes in local coordinates.
fn face_shape(k: usize, dim: usize, s: [f64; 2]) -> (Vec<f64>, Vec<[f64; 2]>) {
    let (a, b) = (s[0], s[1]);
    match (dim, k) {
        (2, 2) => (vec![0.5 * (1.0 - a), 0.5 * (1.0 + a)], vec![[-0.5, 0.0], [0.5, 0.0]]),
        (2, 3) => (
            vec![0.5 * a * (a - 1.0), 0.5 * a * (a + 1.0), 1.0 - a * a],
            vec![[a - 0.5, 0.0], [a + 0.5, 0.0], [-2.0 * a, 0.0]],
        ),
        (3, 3) => (vec![1.0 - a - b, a, b], vec![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
        (3, 4) => {
            let (n, d) = shape_fn(Shape::Quad4, [a, b, 0.0]);
            (n, d.iter().map(|v| [v[0], v[1]]).collect())
        }
        (3, 6) => {
            let l = [1.0 - a - b, a, b];
            let dl = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]];
            let mut n = vec![0.0; 6];
            let mut d = vec![[0.0; 2]; 6];
            for i in 0..3 {
                n[i] = l[i] * (2.0 * l[i] - 1.0);
                d[i] = [(4.0 * l[i] - 1.0) * dl[i][0], (4.0 * l[i] - 1.0) * dl[i][1]];
            }
            for (m, &(p, q)) in [(0, 1), (1, 2), (2, 0)].iter().enumerate() {
                n[3 + m] = 4.0 * l[p] * l[q];
                d[3 + m] = [4.0 * (dl[p][0] * l[q] + l[p] * dl[q][0]), 4.0 * (dl[p][1] * l[q] + l[p] * dl[q][1])];
            }
            (n, d)
        }
        (3, 8) => {
            let (n, d) = shape_fn(Shape::Quad8, [a, b, 0.0]);
            (n, d.iter().map(|v| [v[0], v[1]]).collect())
        }
        _ => (Vec::new(), Vec::new()),
    }
}

/// Gauss rule for an edge/face with `k` nodes.
fn face_rule(k: usize, dim: usize) -> Vec<([f64; 2], f64)> {
    match (dim, k) {
        (2, 2) => G2.iter().map(|&a| ([a, 0.0], 1.0)).collect(),
        (2, 3) => (0..3).map(|i| ([G3[i], 0.0], W3[i])).collect(),
        (3, 3) => vec![([1.0 / 6.0, 1.0 / 6.0], 1.0 / 6.0), ([2.0 / 3.0, 1.0 / 6.0], 1.0 / 6.0), ([1.0 / 6.0, 2.0 / 3.0], 1.0 / 6.0)],
        (3, 6) => {
            let (a1, b1, w1) = (0.059_715_871_789_770, 0.470_142_064_105_115, 0.132_394_152_788_506);
            let (a2, b2, w2) = (0.797_426_985_353_087, 0.101_286_507_323_456, 0.125_939_180_544_827);
            let pts = [
                ([1.0 / 3.0, 1.0 / 3.0], 0.225),
                ([a1, b1], w1),
                ([b1, a1], w1),
                ([b1, b1], w1),
                ([a2, b2], w2),
                ([b2, a2], w2),
                ([b2, b2], w2),
            ];
            pts.iter().map(|&(p, w)| (p, 0.5 * w)).collect()
        }
        (3, 4) => {
            let mut v = Vec::new();
            for &e in &G2 {
                for &x in &G2 {
                    v.push(([x, e], 1.0));
                }
            }
            v
        }
        (3, 8) => {
            let mut v = Vec::new();
            for j in 0..3 {
                for i in 0..3 {
                    v.push(([G3[i], G3[j]], W3[i] * W3[j]));
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Diagonal rule for a first-order edge/face: each corner gets the weight
/// of the Gauss point closest to the corner (triangle: one third of the area).
/// Returns (point, weight, corner).
fn face_lumped_rule(k: usize, dim: usize) -> Option<Vec<([f64; 2], f64, usize)>> {
    let g = G2[1];
    Some(match (dim, k) {
        (2, 2) => vec![([-g, 0.0], 1.0, 0), ([g, 0.0], 1.0, 1)],
        (3, 3) => (0..3).map(|a| ([1.0 / 3.0, 1.0 / 3.0], 1.0 / 6.0, a)).collect(),
        (3, 4) => vec![([-g, -g], 1.0, 0), ([g, -g], 1.0, 1), ([g, g], 1.0, 2), ([-g, g], 1.0, 3)],
        _ => return None,
    })
}

impl HeatElem {
    pub fn dim(&self) -> usize {
        shape_dim(self.shape)
    }

    pub fn n_nodes(&self) -> usize {
        shape_nodes(self.shape)
    }

    /// First-order element (diagonal capacity).
    pub fn first_order(&self) -> bool {
        nodal_rule(self.shape).is_some()
    }

    pub fn faces(&self) -> &'static [&'static [usize]] {
        match self.shape {
            Shape::Tri3 => TRI3_FACES,
            Shape::Quad4 => QUAD4_FACES,
            Shape::Quad8 => QUAD8_FACES,
            Shape::Tet4 => TET4_FACES,
            Shape::Tet10 => TET10_FACES,
            Shape::Wedge6 => WEDGE6_FACES,
            Shape::Hex8 => HEX8_FACES,
            Shape::Hex20 => HEX20_FACES,
        }
    }

    /// Integration points for conduction and body fluxes.
    pub fn ips(&self, x: &[[f64; 3]], thickness: f64) -> Result<Vec<Ip>> {
        gauss_rule(self.shape).into_iter().map(|(p, w)| ip_at(self.shape, x, p, w, thickness, self.axisym)).collect()
    }

    /// Diagonal integration: each node gets the weight of the Gauss point closest to the node
    /// (the point at ξ_i/√3); the shape functions are replaced by δ_ij.
    pub fn gauss_lumped_ips(&self, x: &[[f64; 3]], thickness: f64) -> Result<Vec<Ip>> {
        let Some(rule) = nodal_rule(self.shape) else { return self.ips(x, thickness) };
        let nn = self.n_nodes();
        let mut out = Vec::with_capacity(nn);
        match self.shape {
            Shape::Quad4 | Shape::Hex8 => {
                for (i, (p, _)) in rule.into_iter().enumerate() {
                    let q = [p[0] * G2[1], p[1] * G2[1], p[2] * G2[1]];
                    let mut ip = ip_at(self.shape, x, q, 1.0, thickness, self.axisym)?;
                    ip.n = (0..nn).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                    out.push(ip);
                }
            }
            _ => {
                // one Gauss point: equal shares of the volume
                let ips = self.ips(x, thickness)?;
                let vol: f64 = ips.iter().map(|ip| ip.w).sum();
                for i in 0..nn {
                    let mut ip = ips[0].clone();
                    ip.n = (0..nn).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                    ip.w = vol / nn as f64;
                    out.push(ip);
                }
            }
        }
        Ok(out)
    }

    /// Integration points for the capacity term and the body flux (BF). First-order
    /// elements are integrated diagonally: plane and three-dimensional
    /// elements give each node the weight of the Gauss point closest to the node
    /// (`gauss_lumped_ips`), axisymmetric elements the row-summed weight
    /// ∫N_i 2πr dA. Second-order elements use the Gauss rule (consistent).
    pub fn cap_ips(&self, x: &[[f64; 3]], thickness: f64) -> Result<Vec<Ip>> {
        if self.axisym && self.first_order() {
            // axisymmetry: row-summed diagonal, the weight ∫N_i 2πr dA
            let ips = self.ips(x, thickness)?;
            let nn = self.n_nodes();
            return Ok((0..nn)
                .map(|i| {
                    let mut ip = ips[0].clone();
                    ip.n = (0..nn).map(|j| if j == i { 1.0 } else { 0.0 }).collect();
                    ip.w = ips.iter().map(|g| g.n[i] * g.w).sum();
                    ip
                })
                .collect());
        }
        if self.first_order() {
            self.gauss_lumped_ips(x, thickness)
        } else {
            self.ips(x, thickness)
        }
    }

    /// Points on the face `face` (0-based). `lumped` = diagonal integration
    /// (each corner gets the weight of the nearest Gauss point) for first-order faces.
    pub fn face_points(&self, x: &[[f64; 3]], face: usize, thickness: f64, lumped: bool) -> Result<Vec<FacePoint>> {
        let faces = self.faces();
        let Some(&fnodes) = faces.get(face) else {
            return fail(format!("{} has only the faces 1–{}", self.name, faces.len()));
        };
        let dim = self.dim();
        let k = fnodes.len();
        let rule: Vec<([f64; 2], f64, Option<usize>)> = match face_lumped_rule(k, dim).filter(|_| lumped) {
            Some(r) => r.into_iter().map(|(p, w, a)| (p, w, Some(a))).collect(),
            None => face_rule(k, dim).into_iter().map(|(p, w)| (p, w, None)).collect(),
        };
        let mut out = Vec::with_capacity(rule.len());
        for (s, wg, owner) in rule {
            let (nf, df) = face_shape(k, dim, s);
            let mut t1 = [0.0; 3];
            let mut t2 = [0.0; 3];
            let mut xp = [0.0; 3];
            for (a, &i) in fnodes.iter().enumerate() {
                for c in 0..3 {
                    t1[c] += df[a][0] * x[i][c];
                    t2[c] += df[a][1] * x[i][c];
                    xp[c] += nf[a] * x[i][c];
                }
            }
            let da = if dim == 2 {
                let l = (t1[0] * t1[0] + t1[1] * t1[1]).sqrt();
                if self.axisym { l * 2.0 * std::f64::consts::PI * xp[0] } else { l * thickness }
            } else {
                let c = super::common::cross(t1, t2);
                super::common::norm(c)
            };
            let mut n = vec![0.0; x.len()];
            match owner {
                Some(a) => n[fnodes[a]] = 1.0,
                None => {
                    for (a, &i) in fnodes.iter().enumerate() {
                        n[i] = nf[a];
                    }
                }
            }
            out.push(FacePoint { n, w: da * wg });
        }
        Ok(out)
    }

    /// Conduction: internal heat flux f = ∫∇Nᵀ k ∇T dV and tangent.
    pub fn conduction(&self, ips: &[Ip], props: &HeatProps, temp: &[f64], tangent: bool) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
        let nn = temp.len();
        let mut f = vec![0.0; nn];
        let mut k = if tangent { vec![0.0; nn * nn] } else { Vec::new() };
        for ip in ips {
            let t: f64 = ip.n.iter().zip(temp).map(|(a, b)| a * b).sum();
            let mut g = [0.0; 3];
            for (d, &ti) in ip.dndx.iter().zip(temp) {
                for c in 0..3 {
                    g[c] += d[c] * ti;
                }
            }
            let (kc, dk) = props.conductivity(t)?;
            let kg = [0, 1, 2].map(|r| (0..3).map(|c| kc[r][c] * g[c]).sum::<f64>());
            for i in 0..nn {
                let di = ip.dndx[i];
                f[i] += ip.w * (di[0] * kg[0] + di[1] * kg[1] + di[2] * kg[2]);
            }
            if tangent {
                let dkg = [0, 1, 2].map(|r| (0..3).map(|c| dk[r][c] * g[c]).sum::<f64>());
                let varies = dkg.iter().any(|v| *v != 0.0);
                for i in 0..nn {
                    let di = ip.dndx[i];
                    let kdi = [0, 1, 2].map(|r| (0..3).map(|c| kc[r][c] * di[c]).sum::<f64>());
                    let ddk = di[0] * dkg[0] + di[1] * dkg[1] + di[2] * dkg[2];
                    for j in 0..nn {
                        let dj = ip.dndx[j];
                        let mut v = kdi[0] * dj[0] + kdi[1] * dj[1] + kdi[2] * dj[2];
                        if varies {
                            v += ddk * ip.n[j];
                        }
                        k[i * nn + j] += ip.w * v;
                    }
                }
            }
        }
        Ok((f, tangent.then_some(k)))
    }

    /// Heat capacity: f = ∫ N ρ (U(T) − U(Tₙ))/Δt dV and tangent ∫ N ρ c(T) N /Δt dV.
    pub fn capacity(&self, cips: &[Ip], props: &HeatProps, temp: &[f64], temp_old: &[f64], dt: f64, tangent: bool) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
        let nn = temp.len();
        let mut f = vec![0.0; nn];
        let mut k = if tangent { vec![0.0; nn * nn] } else { Vec::new() };
        let Some(c) = props.capacity() else {
            return Ok((f, tangent.then_some(k)));
        };
        let latent = !props.latent.is_empty();
        let rho = props.density;
        if rho == 0.0 {
            return fail(format!("material {}: *DENSITY is missing (required for transient heat conduction)", props.name));
        }
        for ip in cips {
            let t: f64 = ip.n.iter().zip(temp).map(|(a, b)| a * b).sum();
            let t0: f64 = ip.n.iter().zip(temp_old).map(|(a, b)| a * b).sum();
            // ρ c(T) (T − Tₙ)/Δt with c at the end of the increment
            let (cv, dc) = c.eval(t);
            let mut du = rho * cv[0] * (t - t0) / dt;
            let mut dudt = rho * (cv[0] + dc[0] * (t - t0)) / dt;
            if latent {
                // latent heat: the enthalpy difference ρ L (f(T) − f(Tₙ))/Δt
                let (l1, dl1) = props.latent_energy(t);
                let (l0, _) = props.latent_energy(t0);
                du += rho * (l1 - l0) / dt;
                dudt += rho * dl1 / dt;
            }
            for i in 0..nn {
                f[i] += ip.w * ip.n[i] * du;
                if tangent {
                    for j in 0..nn {
                        k[i * nn + j] += ip.w * ip.n[i] * dudt * ip.n[j];
                    }
                }
            }
        }
        Ok((f, tangent.then_some(k)))
    }

    /// The heat flux HFL = −k ∇T at the integration points.
    pub fn heat_flux(&self, ips: &[Ip], props: &HeatProps, temp: &[f64]) -> Result<IpField> {
        let labels: &[&str] = if self.dim() == 2 { &["HFL1", "HFL2"] } else { &["HFL1", "HFL2", "HFL3"] };
        let mut fld = IpField::new("HFL", labels);
        for (q, ip) in ips.iter().enumerate() {
            let t: f64 = ip.n.iter().zip(temp).map(|(a, b)| a * b).sum();
            let mut g = [0.0; 3];
            for (d, &ti) in ip.dndx.iter().zip(temp) {
                for c in 0..3 {
                    g[c] += d[c] * ti;
                }
            }
            let (kc, _) = props.conductivity(t)?;
            let h: Vec<f64> = (0..labels.len()).map(|r| -(0..3).map(|c| kc[r][c] * g[c]).sum::<f64>()).collect();
            fld.points.push((q + 1, 0, h));
        }
        Ok(fld)
    }

    /// The face number (0-based) for a load label such as "S3", "F2" or "R1".
    pub fn face_index(&self, label: &str) -> Result<usize> {
        let num = label.get(1..).and_then(|s| s.parse::<usize>().ok());
        match num {
            Some(n) if n >= 1 && n <= self.faces().len() => Ok(n - 1),
            _ => fail(format!("{} has the faces 1–{}, not {label}", self.name, self.faces().len())),
        }
    }
}

impl ElementType for HeatElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Heat
    }
    fn dim(&self) -> usize {
        shape_dim(self.shape)
    }
    fn n_nodes(&self) -> usize {
        shape_nodes(self.shape)
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[NT11]
    }
    fn n_ips(&self) -> usize {
        gauss_rule(self.shape).len()
    }
    fn vtk_type(&self) -> u8 {
        match self.shape {
            Shape::Tri3 => 5,
            Shape::Quad4 => 9,
            Shape::Quad8 => 23,
            Shape::Tet4 => 10,
            Shape::Tet10 => 24,
            Shape::Wedge6 => 13,
            Shape::Hex8 => 12,
            Shape::Hex20 => 25,
        }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        HeatElem::faces(self)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let t = match ctx.section {
            Some(s) => s.solid_thickness(),
            None => 1.0,
        };
        Ok(super::common::volume_weights(&self.ips(&ctx.x, t)?))
    }
}

/// Weights for interpolating nodal temperatures to the integration points of a
/// mechanical element (thermal strain), one row per integration point.
/// First-order elements get the element's mean temperature at all points
/// (e.g. CPS4, CPE4 and C3D8); second-order
/// elements the values of the shape functions. `None` if the element type is unknown.
pub fn mechanical_ip_shapes(etype: &str, x: &[[f64; 3]]) -> Option<Vec<Vec<f64>>> {
    let etype = super::heat_coupled::mech_name(etype).or_else(|| super::thermal_electric_structural::mech_name(etype)).unwrap_or(etype);
    let rows = mechanical_ip_interp(etype, x)?;
    let n = x.len();
    let first = matches!(n, 2 | 3 | 4 | 6 | 8) && !matches!(etype, "CPS8" | "CPE8" | "CAX8" | "CPS8R" | "CPE8R" | "CAX8R" | "CPS6" | "CPE6" | "CAX6");
    if first {
        Some(vec![vec![1.0 / n as f64; n]; rows.len()])
    } else {
        Some(rows)
    }
}

fn mechanical_ip_interp(etype: &str, x: &[[f64; 3]]) -> Option<Vec<Vec<f64>>> {
    let (shape, rule): (Shape, Vec<([f64; 3], f64)>) = match etype {
        "CPS3" | "CPE3" | "CAX3" => (Shape::Tri3, vec![([1.0 / 3.0, 1.0 / 3.0, 0.0], 0.5)]),
        "CPS4" | "CPE4" | "CAX4" | "CPE4H" | "CAX4H" | "CPS4I" | "CPE4I" | "CAX4I" => (Shape::Quad4, gauss_rule(Shape::Quad4)),
        "CPS4R" | "CPE4R" | "CAX4R" => (Shape::Quad4, vec![([0.0, 0.0, 0.0], 4.0)]),
        "CPS8" | "CPE8" | "CAX8" => (Shape::Quad8, gauss_rule(Shape::Quad8)),
        "CPS8R" | "CPE8R" | "CAX8R" => (Shape::Quad8, gauss_rule(Shape::Quad4)),
        "C3D4" => (Shape::Tet4, vec![([0.25, 0.25, 0.25], 1.0 / 6.0)]),
        "C3D10" => (Shape::Tet10, tet4_rule()),
        "C3D6" => (Shape::Wedge6, gauss_rule(Shape::Wedge6)),
        "C3D8" | "C3D8H" | "C3D8I" => (Shape::Hex8, gauss_rule(Shape::Hex8)),
        "C3D8R" => (Shape::Hex8, vec![([0.0, 0.0, 0.0], 8.0)]),
        "C3D20" => (Shape::Hex20, gauss_rule(Shape::Hex20)),
        "C3D20R" => (Shape::Hex20, gauss_rule(Shape::Hex8)),
        "T2D2" | "T3D2" => return Some(vec![vec![0.5, 0.5]]),
        _ => return None,
    };
    if x.len() != shape_nodes(shape) {
        return None;
    }
    Some(rule.into_iter().map(|(p, _)| shape_fn(shape, p).0).collect())
}
