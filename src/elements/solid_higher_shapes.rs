//! Shape functions, integration rules and faces for the continuum elements in
//! `solid_higher.rs` and `axisym.rs`.
//!
//! The shape functions are written once with dual numbers (value + three partial derivatives)
//! so that the derivatives are exact without being written by hand.
//!
//! Natural coordinates:
//! - hexahedra and quadrilaterals: ξ, η, ζ ∈ [−1, 1]
//! - tetrahedra and triangles: (ξ, η, ζ) with L₁ = 1 − ξ − η − ζ, L₂ = ξ, L₃ = η, L₄ = ζ
//! - wedges: triangle (ξ, η) times ζ ∈ [−1, 1]
//!
//! The node numbering follows the .inp format (corners first, then midside nodes in edge order).

use std::ops::{Add, Mul, Neg, Sub};

/// Dual number: value and derivatives with respect to (ξ, η, ζ).
#[derive(Debug, Clone, Copy)]
pub struct D {
    pub v: f64,
    pub d: [f64; 3],
}

impl D {
    pub fn c(v: f64) -> D {
        D { v, d: [0.0; 3] }
    }
    pub fn var(v: f64, i: usize) -> D {
        let mut d = [0.0; 3];
        d[i] = 1.0;
        D { v, d }
    }
}

impl Add for D {
    type Output = D;
    fn add(self, o: D) -> D {
        D { v: self.v + o.v, d: [self.d[0] + o.d[0], self.d[1] + o.d[1], self.d[2] + o.d[2]] }
    }
}
impl Sub for D {
    type Output = D;
    fn sub(self, o: D) -> D {
        D { v: self.v - o.v, d: [self.d[0] - o.d[0], self.d[1] - o.d[1], self.d[2] - o.d[2]] }
    }
}
impl Mul for D {
    type Output = D;
    fn mul(self, o: D) -> D {
        D { v: self.v * o.v, d: [0, 1, 2].map(|i| self.d[i] * o.v + self.v * o.d[i]) }
    }
}
impl Mul<f64> for D {
    type Output = D;
    fn mul(self, s: f64) -> D {
        D { v: self.v * s, d: self.d.map(|x| x * s) }
    }
}
impl Add<f64> for D {
    type Output = D;
    fn add(self, s: f64) -> D {
        D { v: self.v + s, d: self.d }
    }
}
impl Neg for D {
    type Output = D;
    fn neg(self) -> D {
        self * -1.0
    }
}

/// Element geometry (interpolation type).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Line2,
    Line3,
    Tri3,
    Tri6,
    Quad4,
    Quad8,
    Tet4,
    Tet10,
    Wedge6,
    Wedge15,
    Hex8,
    Hex20,
}

impl Shape {
    pub fn n_nodes(self) -> usize {
        match self {
            Shape::Line2 => 2,
            Shape::Line3 => 3,
            Shape::Tri3 => 3,
            Shape::Tri6 => 6,
            Shape::Quad4 => 4,
            Shape::Quad8 => 8,
            Shape::Tet4 => 4,
            Shape::Tet10 => 10,
            Shape::Wedge6 => 6,
            Shape::Wedge15 => 15,
            Shape::Hex8 => 8,
            Shape::Hex20 => 20,
        }
    }

    /// Number of natural coordinates (1 for lines, 2 for surfaces, 3 for volumes).
    pub fn pdim(self) -> usize {
        match self {
            Shape::Line2 | Shape::Line3 => 1,
            Shape::Tri3 | Shape::Tri6 | Shape::Quad4 | Shape::Quad8 => 2,
            _ => 3,
        }
    }

    /// VTK cell type.
    pub fn vtk(self) -> u8 {
        match self {
            Shape::Line2 => 3,
            Shape::Line3 => 21,
            Shape::Tri3 => 5,
            Shape::Tri6 => 22,
            Shape::Quad4 => 9,
            Shape::Quad8 => 23,
            Shape::Tet4 => 10,
            Shape::Tet10 => 24,
            Shape::Wedge6 => 13,
            Shape::Wedge15 => 26,
            Shape::Hex8 => 12,
            Shape::Hex20 => 25,
        }
    }

    /// Faces (edges in 2D) in the .inp format numbering S1, S2, ... Corner nodes first,
    /// then midside nodes in edge order, so that the face can be interpolated with
    /// `face_shape`.
    pub fn faces(self) -> &'static [&'static [usize]] {
        match self {
            Shape::Tri3 => &[&[0, 1], &[1, 2], &[2, 0]],
            Shape::Tri6 => &[&[0, 1, 3], &[1, 2, 4], &[2, 0, 5]],
            Shape::Quad4 => &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]],
            Shape::Quad8 => &[&[0, 1, 4], &[1, 2, 5], &[2, 3, 6], &[3, 0, 7]],
            Shape::Tet4 => &[&[0, 1, 2], &[0, 3, 1], &[1, 3, 2], &[2, 3, 0]],
            Shape::Tet10 => &[&[0, 1, 2, 4, 5, 6], &[0, 3, 1, 7, 8, 4], &[1, 3, 2, 8, 9, 5], &[2, 3, 0, 9, 7, 6]],
            Shape::Wedge6 => &[&[0, 1, 2], &[3, 5, 4], &[0, 3, 4, 1], &[1, 4, 5, 2], &[2, 5, 3, 0]],
            Shape::Wedge15 => &[
                &[0, 1, 2, 6, 7, 8],
                &[3, 5, 4, 11, 10, 9],
                &[0, 3, 4, 1, 12, 9, 13, 6],
                &[1, 4, 5, 2, 13, 10, 14, 7],
                &[2, 5, 3, 0, 14, 11, 12, 8],
            ],
            Shape::Hex8 => &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]],
            Shape::Hex20 => &[
                &[0, 1, 2, 3, 8, 9, 10, 11],
                &[4, 7, 6, 5, 15, 14, 13, 12],
                &[0, 4, 5, 1, 16, 12, 17, 8],
                &[1, 5, 6, 2, 17, 13, 18, 9],
                &[2, 6, 7, 3, 18, 14, 19, 10],
                &[3, 7, 4, 0, 19, 15, 16, 11],
            ],
            Shape::Line2 | Shape::Line3 => &[],
        }
    }

    /// Geometry for a face with `n` nodes of this element shape.
    pub fn face_shape(self, n: usize) -> Shape {
        match (self.pdim(), n) {
            (2, 2) => Shape::Line2,
            (2, 3) => Shape::Line3,
            (3, 3) => Shape::Tri3,
            (3, 6) => Shape::Tri6,
            (3, 4) => Shape::Quad4,
            _ => Shape::Quad8,
        }
    }
}

const HEX_C: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Natural coordinates of the midside nodes for C3D20 (nodes 9–20).
const HEX20_MID: [[f64; 3]; 12] = [
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

const QUAD_C: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
const QUAD8_MID: [[f64; 2]; 4] = [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];

/// Shape functions as dual numbers at the point `p` (natural coordinates).
pub fn shape_dual(s: Shape, p: [f64; 3]) -> Vec<D> {
    let x = D::var(p[0], 0);
    let y = D::var(p[1], 1);
    let z = D::var(p[2], 2);
    let one = D::c(1.0);
    match s {
        Shape::Line2 => vec![(one - x) * 0.5, (one + x) * 0.5],
        Shape::Line3 => vec![x * (x - one) * 0.5, x * (x + one) * 0.5, one - x * x],
        Shape::Tri3 => vec![one - x - y, x, y],
        Shape::Tri6 => {
            let l = [one - x - y, x, y];
            let c = |a: D| a * (a * 2.0 + -1.0);
            vec![c(l[0]), c(l[1]), c(l[2]), l[0] * l[1] * 4.0, l[1] * l[2] * 4.0, l[2] * l[0] * 4.0]
        }
        Shape::Quad4 => QUAD_C.iter().map(|c| (one + x * c[0]) * (one + y * c[1]) * 0.25).collect(),
        Shape::Quad8 => {
            let mut n: Vec<D> =
                QUAD_C.iter().map(|c| (one + x * c[0]) * (one + y * c[1]) * (x * c[0] + y * c[1] + -1.0) * 0.25).collect();
            for m in QUAD8_MID {
                n.push(if m[0] == 0.0 { (one - x * x) * (one + y * m[1]) * 0.5 } else { (one + x * m[0]) * (one - y * y) * 0.5 });
            }
            n
        }
        Shape::Tet4 => vec![one - x - y - z, x, y, z],
        Shape::Tet10 => {
            let l = [one - x - y - z, x, y, z];
            let c = |a: D| a * (a * 2.0 + -1.0);
            vec![
                c(l[0]),
                c(l[1]),
                c(l[2]),
                c(l[3]),
                l[0] * l[1] * 4.0,
                l[1] * l[2] * 4.0,
                l[2] * l[0] * 4.0,
                l[0] * l[3] * 4.0,
                l[1] * l[3] * 4.0,
                l[2] * l[3] * 4.0,
            ]
        }
        Shape::Wedge6 => {
            let l = [one - x - y, x, y];
            let lo = (one - z) * 0.5;
            let hi = (one + z) * 0.5;
            vec![l[0] * lo, l[1] * lo, l[2] * lo, l[0] * hi, l[1] * hi, l[2] * hi]
        }
        Shape::Wedge15 => {
            let l = [one - x - y, x, y];
            let lo = one - z;
            let hi = one + z;
            let b = one - z * z;
            let mut n = Vec::with_capacity(15);
            for side in [lo, hi] {
                for li in l {
                    n.push(li * (li * 2.0 + -1.0) * side * 0.5 - li * b * 0.5);
                }
            }
            for side in [lo, hi] {
                for (a, c) in [(0, 1), (1, 2), (2, 0)] {
                    n.push(l[a] * l[c] * side * 2.0);
                }
            }
            for li in l {
                n.push(li * b);
            }
            n
        }
        Shape::Hex8 => HEX_C.iter().map(|c| (one + x * c[0]) * (one + y * c[1]) * (one + z * c[2]) * 0.125).collect(),
        Shape::Hex20 => {
            let mut n: Vec<D> = HEX_C
                .iter()
                .map(|c| (one + x * c[0]) * (one + y * c[1]) * (one + z * c[2]) * (x * c[0] + y * c[1] + z * c[2] + -2.0) * 0.125)
                .collect();
            for m in HEX20_MID {
                let f = |q: D, c: f64| if c == 0.0 { one - q * q } else { one + q * c };
                n.push(f(x, m[0]) * f(y, m[1]) * f(z, m[2]) * 0.25);
            }
            n
        }
    }
}

/// Shape functions and derivatives with respect to the natural coordinates.
pub fn shape(s: Shape, p: [f64; 3]) -> (Vec<f64>, Vec<[f64; 3]>) {
    let n = shape_dual(s, p);
    (n.iter().map(|d| d.v).collect(), n.iter().map(|d| d.d).collect())
}

/// Natural coordinates of the element's nodes.
pub fn node_coords(s: Shape) -> Vec<[f64; 3]> {
    match s {
        Shape::Line2 => vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        Shape::Line3 => vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        Shape::Tri3 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        Shape::Tri6 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
        Shape::Quad4 => QUAD_C.iter().map(|c| [c[0], c[1], 0.0]).collect(),
        Shape::Quad8 => QUAD_C.iter().chain(QUAD8_MID.iter()).map(|c| [c[0], c[1], 0.0]).collect(),
        Shape::Tet4 => vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        Shape::Tet10 => {
            let c = node_coords(Shape::Tet4);
            let mut v = c.clone();
            for (a, b) in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)] {
                v.push([0, 1, 2].map(|i| 0.5 * (c[a][i] + c[b][i])));
            }
            v
        }
        Shape::Wedge6 | Shape::Wedge15 => {
            let t = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
            let mut v: Vec<[f64; 3]> = Vec::new();
            for z in [-1.0, 1.0] {
                for p in t {
                    v.push([p[0], p[1], z]);
                }
            }
            if s == Shape::Wedge15 {
                for z in [-1.0, 1.0] {
                    for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                        v.push([0.5 * (t[a][0] + t[b][0]), 0.5 * (t[a][1] + t[b][1]), z]);
                    }
                }
                for p in t {
                    v.push([p[0], p[1], 0.0]);
                }
            }
            v
        }
        Shape::Hex8 => HEX_C.to_vec(),
        Shape::Hex20 => HEX_C.iter().chain(HEX20_MID.iter()).copied().collect(),
    }
}

// ---------------------------------------------------------------------------
// Integration rules
// ---------------------------------------------------------------------------

/// Integration rule: points in natural coordinates and weights.
pub type Rule = Vec<([f64; 3], f64)>;

/// Gauss points and weights on [−1, 1] in ascending order.
pub fn gauss1d(n: usize) -> Vec<(f64, f64)> {
    match n {
        1 => vec![(0.0, 2.0)],
        2 => {
            let g = 1.0 / 3f64.sqrt();
            vec![(-g, 1.0), (g, 1.0)]
        }
        3 => {
            let g = 0.6f64.sqrt();
            vec![(-g, 5.0 / 9.0), (0.0, 8.0 / 9.0), (g, 5.0 / 9.0)]
        }
        _ => {
            let (a, b) = (0.339_981_043_584_856_3, 0.861_136_311_594_052_6);
            let (wa, wb) = (0.652_145_154_862_546_1, 0.347_854_845_137_453_9);
            vec![(-b, wb), (-a, wa), (a, wa), (b, wb)]
        }
    }
}

/// n×n (×n) Gauss rule with the first coordinate fastest.
pub fn gauss_tensor(n: usize, dim: usize) -> Rule {
    let g = gauss1d(n);
    let mut r = Vec::new();
    match dim {
        1 => {
            for &(a, wa) in &g {
                r.push(([a, 0.0, 0.0], wa));
            }
        }
        2 => {
            for &(b, wb) in &g {
                for &(a, wa) in &g {
                    r.push(([a, b, 0.0], wa * wb));
                }
            }
        }
        _ => {
            for &(c, wc) in &g {
                for &(b, wb) in &g {
                    for &(a, wa) in &g {
                        r.push(([a, b, c], wa * wb * wc));
                    }
                }
            }
        }
    }
    r
}

/// Triangle rules (area 1/2 in natural coordinates).
pub fn tri_rule(n: usize) -> Vec<([f64; 2], f64)> {
    match n {
        1 => vec![([1.0 / 3.0, 1.0 / 3.0], 0.5)],
        3 => vec![([1.0 / 6.0, 1.0 / 6.0], 1.0 / 6.0), ([2.0 / 3.0, 1.0 / 6.0], 1.0 / 6.0), ([1.0 / 6.0, 2.0 / 3.0], 1.0 / 6.0)],
        4 => vec![([1.0 / 3.0, 1.0 / 3.0], -27.0 / 96.0), ([0.2, 0.2], 25.0 / 96.0), ([0.6, 0.2], 25.0 / 96.0), ([0.2, 0.6], 25.0 / 96.0)],
        6 => {
            // degree 4 (Dunavant)
            let (a, wa) = (0.445_948_490_915_965, 0.223_381_589_678_011 / 2.0);
            let (b, wb) = (0.091_576_213_509_771, 0.109_951_743_655_322 / 2.0);
            vec![
                ([a, a], wa),
                ([1.0 - 2.0 * a, a], wa),
                ([a, 1.0 - 2.0 * a], wa),
                ([b, b], wb),
                ([1.0 - 2.0 * b, b], wb),
                ([b, 1.0 - 2.0 * b], wb),
            ]
        }
        7 => {
            // degree 5 (Radon)
            let (a, wa) = (0.470_142_064_105_115, 0.132_394_152_788_506 / 2.0);
            let (b, wb) = (0.101_286_507_323_456, 0.125_939_180_544_827 / 2.0);
            vec![
                ([1.0 / 3.0, 1.0 / 3.0], 0.1125),
                ([a, a], wa),
                ([1.0 - 2.0 * a, a], wa),
                ([a, 1.0 - 2.0 * a], wa),
                ([b, b], wb),
                ([1.0 - 2.0 * b, b], wb),
                ([b, 1.0 - 2.0 * b], wb),
            ]
        }
        _ => {
            // Collapsed Gauss rule (Duffy), 4×4 points, exact for degree 5.
            let g: Vec<(f64, f64)> = gauss1d(4).into_iter().map(|(t, w)| (0.5 * (t + 1.0), 0.5 * w)).collect();
            let mut r = Vec::new();
            for &(u, wu) in &g {
                for &(v, wv) in &g {
                    r.push(([u, v * (1.0 - u)], wu * wv * (1.0 - u)));
                }
            }
            r
        }
    }
}

/// Tetrahedron rules (volume 1/6 in natural coordinates).
pub fn tet_rule(n: usize) -> Rule {
    match n {
        1 => vec![([0.25; 3], 1.0 / 6.0)],
        4 => {
            let a = 0.585_410_196_624_968_5;
            let b = 0.138_196_601_125_010_5;
            let w = 1.0 / 24.0;
            // point order: near node 1, 2, 3, 4
            vec![([b, b, b], w), ([a, b, b], w), ([b, a, b], w), ([b, b, a], w)]
        }
        5 => {
            let w = 9.0 / 120.0;
            vec![([0.25; 3], -4.0 / 30.0), ([1.0 / 6.0; 3], w), ([0.5, 1.0 / 6.0, 1.0 / 6.0], w), ([1.0 / 6.0, 0.5, 1.0 / 6.0], w), ([1.0 / 6.0, 1.0 / 6.0, 0.5], w)]
        }
        11 => {
            // Keast, degree 4
            let mut r = vec![([0.25; 3], -0.013_155_555_555_555_6)];
            let (a, b) = (0.071_428_571_428_571_4, 0.785_714_285_714_286);
            for p in [[a, a, a], [b, a, a], [a, b, a], [a, a, b]] {
                r.push((p, 0.007_622_222_222_222_22));
            }
            let (c, d) = (0.399_403_576_166_799, 0.100_596_423_833_201);
            for p in [[c, c, d], [c, d, c], [d, c, c], [c, d, d], [d, c, d], [d, d, c]] {
                r.push((p, 0.024_888_888_888_888_9));
            }
            r
        }
        14 => {
            // Walkington, degree 5
            let mut r = Vec::new();
            for (a, w) in [(0.092_735_250_310_891_23, 0.018_781_320_953_002_64), (0.310_885_919_263_300_6, 0.012_248_840_519_393_66)] {
                let b = 1.0 - 3.0 * a;
                for p in [[a, a, a], [b, a, a], [a, b, a], [a, a, b]] {
                    r.push((p, w));
                }
            }
            let (c, d) = (0.454_496_295_874_350_4, 0.045_503_704_125_649_6);
            for p in [[c, c, d], [c, d, c], [d, c, c], [c, d, d], [d, c, d], [d, d, c]] {
                r.push((p, 0.007_091_003_462_846_911));
            }
            r
        }
        15 => {
            // Keast, degree 5
            let mut r = vec![([0.25; 3], 0.030_283_678_097_089)];
            let t = 1.0 / 3.0;
            for p in [[0.0, t, t], [t, 0.0, t], [t, t, 0.0], [t, t, t]] {
                r.push((p, 0.006_026_785_714_286));
            }
            let (a, b) = (0.090_909_090_909_091, 0.727_272_727_272_727);
            for p in [[a, a, a], [b, a, a], [a, b, a], [a, a, b]] {
                r.push((p, 0.011_645_249_086_029));
            }
            let (c, d) = (0.433_449_846_426_336, 0.066_550_153_573_664);
            for p in [[c, c, d], [c, d, c], [d, c, c], [c, d, d], [d, c, d], [d, d, c]] {
                r.push((p, 0.010_949_141_561_386));
            }
            r
        }
        _ => {
            // Collapsed Gauss rule (Duffy): ξ = u, η = v(1−u), ζ = w(1−u)(1−v),
            // 4 points per direction, exact for polynomials of degree 5.
            let g: Vec<(f64, f64)> = gauss1d(4).into_iter().map(|(t, w)| (0.5 * (t + 1.0), 0.5 * w)).collect();
            let mut r = Vec::new();
            for &(u, wu) in &g {
                for &(v, wv) in &g {
                    for &(w, ww) in &g {
                        let p = [u, v * (1.0 - u), w * (1.0 - u) * (1.0 - v)];
                        r.push((p, wu * wv * ww * (1.0 - u) * (1.0 - u) * (1.0 - v)));
                    }
                }
            }
            r
        }
    }
}

/// Wedge rule: triangle rule (n_tri) times Gauss rule (n_z) in ζ, the triangle fastest.
pub fn wedge_rule(n_tri: usize, n_z: usize) -> Rule {
    let mut r = Vec::new();
    for &(z, wz) in &gauss1d(n_z) {
        for &(p, w) in &tri_rule(n_tri) {
            r.push(([p[0], p[1], z], w * wz));
        }
    }
    r
}

/// Rule for a face (or edge) that integrates loads accurately enough.
pub fn face_rule(s: Shape) -> Rule {
    match s {
        Shape::Line2 | Shape::Line3 => gauss_tensor(4, 1),
        Shape::Tri3 | Shape::Tri6 => tri_rule(7).into_iter().map(|(p, w)| ([p[0], p[1], 0.0], w)).collect(),
        _ => gauss_tensor(4, 2),
    }
}
