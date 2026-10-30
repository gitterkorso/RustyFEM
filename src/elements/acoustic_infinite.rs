//! Acoustic infinite elements (area acoustics2): ACIN2D2, ACIN2D3, ACIN3D3,
//! ACIN3D4, ACIN3D6, ACIN3D8, ACINAX2 and ACINAX3.
//!
//! Formulation (Astley 1994, conjugated infinite elements): the element is defined by the base face nodes
//! (the face towards the finite mesh) and a reference point x₀ (*SOLID SECTION,
//! REF NODE= or the coordinates after the thickness on the data line). Each base node α
//! has the radius r_α = |x_α − x₀| and the nodal ray n_α = (x_α − x₀)/r_α. The mapping
//!
//! ```text
//!   x(g, h, ξ) = x₀ + 2/(1 − ξ) Σ b_α(g, h) r_α n_α,    ξ ∈ [−1, 1)
//! ```
//!
//! gives r₀/r = (1 − ξ)/2. The solution p = Σ φ_j p_j e^{−ik(r−r₀)} and the weight
//! δp = F φ_i e^{+ik(r−r₀)} with F = (r₀/r)² and φ = f b_α B_β, f = r₀/r
//! (3D, axisymmetry) or √(r₀/r) (planar), B₁ = 1 and B_β = (1 + ξ) B̂_β where
//! B̂_β are orthogonal with respect to the weight (1 + ξ)² f² F (Jacobi polynomials; ten functions in
//! the radial direction). The phase r − r₀ = Σ r_α q_α b_α (1 + ξ)/(1 − ξ)
//! with q_α = the minimum over the elements at the node of n_α · n_E (the surface normal).
//!
//! The element matrices (constant for frequency-independent materials):
//!
//! ```text
//!   K_g = ∫ ∇(Fφ_i)·∇φ_j dV
//!   C_g = ∫ [∇(Fφ_i)·∇μ φ_j − Fφ_i ∇μ·∇φ_j] dV        (μ = r − r₀)
//!   M_g = ∫ F (1 − ∇μ·∇μ) φ_i φ_j dV
//!   H_g = ∫ F φ_i φ_j dV
//! ```
//!
//! and in the frequency domain A(Ω) = K_g/ρ̃ − i (k̃/ρ̃) C_g − (Ω²/K_f) M_g with
//! k̃ = Ω √(ρ̃/K_f). In the time domain: stiffness K_g/ρ, damping −C_g/(ρ c) +
//! γ/(ρ K_f) H_g and mass M_g/K_f. The matrices are built by the acoustic steps
//! (`steps/acoustic_infinite.rs`); the element's own stiffness and mass are zero.
//!
//! Degrees of freedom: the pressure of the base nodes (POR, β = 1) and nine generalized
//! degrees of freedom per base node (β = 2..10) on shared internal nodes with the number
//! −(3·10⁹ + 16·node + k) that `add_nodes` creates when the file has been read. The first
//! internal node lies at the reference point and the second has q_α as its
//! x coordinate (the element reads them from there).

use super::api::{ElemCtx, ElementType, Family, MassKind};
use crate::fem::dof::{Dof, POR};
use crate::fem::model::FeModel;
use crate::materials::StressMode;
use crate::model::{fail, Result};

/// Number of functions in the radial direction (always ten).
pub const NRAD: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Base {
    Line2,
    Line3,
    Tri3,
    Quad4,
    Tri6,
    Quad8,
}

impl Base {
    pub fn n(self) -> usize {
        match self {
            Base::Line2 => 2,
            Base::Line3 => 3,
            Base::Tri3 => 3,
            Base::Quad4 => 4,
            Base::Tri6 => 6,
            Base::Quad8 => 8,
        }
    }
    fn surface(self) -> bool {
        !matches!(self, Base::Line2 | Base::Line3)
    }
    /// Shape functions and derivatives (∂/∂g, ∂/∂h) at (g, h).
    fn shape(self, g: f64, h: f64) -> (Vec<f64>, Vec<[f64; 2]>) {
        match self {
            Base::Line2 => (vec![0.5 * (1.0 - g), 0.5 * (1.0 + g)], vec![[-0.5, 0.0], [0.5, 0.0]]),
            // node order: the end nodes 1, 2 and the midside node 3
            Base::Line3 => (
                vec![0.5 * g * (g - 1.0), 0.5 * g * (g + 1.0), 1.0 - g * g],
                vec![[g - 0.5, 0.0], [g + 0.5, 0.0], [-2.0 * g, 0.0]],
            ),
            Base::Tri3 => (vec![1.0 - g - h, g, h], vec![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]),
            Base::Tri6 => {
                let l = 1.0 - g - h;
                (
                    vec![l * (2.0 * l - 1.0), g * (2.0 * g - 1.0), h * (2.0 * h - 1.0), 4.0 * l * g, 4.0 * g * h, 4.0 * h * l],
                    vec![
                        [-(4.0 * l - 1.0), -(4.0 * l - 1.0)],
                        [4.0 * g - 1.0, 0.0],
                        [0.0, 4.0 * h - 1.0],
                        [4.0 * (l - g), -4.0 * g],
                        [4.0 * h, 4.0 * g],
                        [-4.0 * h, 4.0 * (l - h)],
                    ],
                )
            }
            Base::Quad4 => {
                let c = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                let mut n = Vec::new();
                let mut d = Vec::new();
                for p in c {
                    n.push(0.25 * (1.0 + p[0] * g) * (1.0 + p[1] * h));
                    d.push([0.25 * p[0] * (1.0 + p[1] * h), 0.25 * p[1] * (1.0 + p[0] * g)]);
                }
                (n, d)
            }
            Base::Quad8 => {
                let c = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                let mut n = Vec::new();
                let mut d = Vec::new();
                for p in c {
                    let (a, b) = (p[0], p[1]);
                    n.push(0.25 * (1.0 + a * g) * (1.0 + b * h) * (a * g + b * h - 1.0));
                    d.push([
                        0.25 * a * (1.0 + b * h) * (2.0 * a * g + b * h),
                        0.25 * b * (1.0 + a * g) * (a * g + 2.0 * b * h),
                    ]);
                }
                // midside nodes 5 (h = −1), 6 (g = 1), 7 (h = 1), 8 (g = −1)
                n.push(0.5 * (1.0 - g * g) * (1.0 - h));
                d.push([-g * (1.0 - h), -0.5 * (1.0 - g * g)]);
                n.push(0.5 * (1.0 + g) * (1.0 - h * h));
                d.push([0.5 * (1.0 - h * h), -h * (1.0 + g)]);
                n.push(0.5 * (1.0 - g * g) * (1.0 + h));
                d.push([-g * (1.0 + h), 0.5 * (1.0 - g * g)]);
                n.push(0.5 * (1.0 - g) * (1.0 - h * h));
                d.push([-0.5 * (1.0 - h * h), -h * (1.0 - g)]);
                (n, d)
            }
        }
    }
    /// The parameter coordinates of the nodes.
    fn node_coords(self) -> Vec<[f64; 2]> {
        match self {
            Base::Line2 => vec![[-1.0, 0.0], [1.0, 0.0]],
            Base::Line3 => vec![[-1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            Base::Tri3 => vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            Base::Tri6 => vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]],
            Base::Quad4 => vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            Base::Quad8 => vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
        }
    }
    /// Integration rule on the base face ((g, h), weight).
    fn rule(self) -> Vec<([f64; 2], f64)> {
        let gl = |n: usize| gauss_legendre(n);
        match self {
            Base::Line2 | Base::Line3 => {
                let n = rule_order(if self == Base::Line2 { 2 } else { 3 });
                gl(n).into_iter().map(|(x, w)| ([x, 0.0], w)).collect()
            }
            Base::Quad4 | Base::Quad8 => {
                let n = rule_order(if self == Base::Quad4 { 2 } else { 3 });
                let g = gl(n);
                let mut v = Vec::new();
                for &(y, wy) in &g {
                    for &(x, wx) in &g {
                        v.push(([x, y], wx * wy));
                    }
                }
                v
            }
            Base::Tri3 | Base::Tri6 => {
                if self == Base::Tri3 && rule_order(1) == 1 {
                    vec![([1.0 / 6.0, 1.0 / 6.0], 1.0 / 6.0), ([2.0 / 3.0, 1.0 / 6.0], 1.0 / 6.0), ([1.0 / 6.0, 2.0 / 3.0], 1.0 / 6.0)]
                } else {
                    // collapsed square (Duffy) with n × n points
                    let n = rule_order(if self == Base::Tri3 { 2 } else { 3 }).max(2);
                    let g = gl(n);
                    let mut v = Vec::new();
                    for &(a, wa) in &g {
                        for &(b, wb) in &g {
                            let u = 0.5 * (1.0 + a);
                            let s = 0.5 * (1.0 + b);
                            v.push(([u * (1.0 - s), u * s], wa * wb * 0.25 * u));
                        }
                    }
                    v
                }
            }
        }
    }
}

/// Number of points per direction on the base face (RUSTYFEM_ACIN_RULE=n changes it, for
/// debugging).
fn rule_order(default: usize) -> usize {
    std::env::var("RUSTYFEM_ACIN_RULE").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

pub struct AcInf {
    pub name: &'static str,
    pub base: Base,
    pub axisym: bool,
}

impl AcInf {
    pub fn nb(&self) -> usize {
        self.base.n()
    }
    /// 3D decay (f = r₀/r) for spatial and axisymmetric elements.
    fn decay3(&self) -> bool {
        self.base.surface() || self.axisym
    }
}

pub static ACIN2D2: AcInf = AcInf { name: "ACIN2D2", base: Base::Line2, axisym: false };
pub static ACIN2D3: AcInf = AcInf { name: "ACIN2D3", base: Base::Line3, axisym: false };
pub static ACIN3D3: AcInf = AcInf { name: "ACIN3D3", base: Base::Tri3, axisym: false };
pub static ACIN3D4: AcInf = AcInf { name: "ACIN3D4", base: Base::Quad4, axisym: false };
pub static ACIN3D6: AcInf = AcInf { name: "ACIN3D6", base: Base::Tri6, axisym: false };
pub static ACIN3D8: AcInf = AcInf { name: "ACIN3D8", base: Base::Quad8, axisym: false };
pub static ACINAX2: AcInf = AcInf { name: "ACINAX2", base: Base::Line2, axisym: true };
pub static ACINAX3: AcInf = AcInf { name: "ACINAX3", base: Base::Line3, axisym: true };

pub static TYPES: &[&dyn ElementType] = &[&ACIN2D2, &ACIN2D3, &ACIN3D3, &ACIN3D4, &ACIN3D6, &ACIN3D8, &ACINAX2, &ACINAX3];

static ALL: &[&AcInf] = &[&ACIN2D2, &ACIN2D3, &ACIN3D3, &ACIN3D4, &ACIN3D6, &ACIN3D8, &ACINAX2, &ACINAX3];

/// The acoustic infinite element named `name`.
pub fn lookup(name: &str) -> Option<&'static AcInf> {
    ALL.iter().copied().find(|e| e.name == name)
}

/// Gauss–Legendre with n points on [−1, 1].
pub fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 0.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        v.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    v.reverse();
    v
}

/// Radial functions B_β(ξ) and derivatives (β = 1..NRAD): B₁ = 1,
/// B_β = (1 + ξ) B̂_β with B̂ orthogonal with respect to (1 + ξ)² (1 − ξ)^a, a = 4 (3D) or 3.
/// Coefficients in the monomial basis (ξ^k), orthogonalized with Gram–Schmidt.
fn radial_coeffs(a: i32) -> Vec<Vec<f64>> {
    let q = gauss_legendre(24);
    let w: Vec<f64> = q.iter().map(|&(x, w)| w * (1.0 + x).powi(2) * (1.0 - x).powi(a)).collect();
    let eval = |c: &[f64], x: f64| c.iter().rev().fold(0.0, |s, v| s * x + v);
    let dot = |a: &[f64], b: &[f64]| q.iter().zip(&w).map(|(&(x, _), wi)| wi * eval(a, x) * eval(b, x)).sum::<f64>();
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for k in 0..NRAD - 1 {
        let mut c = vec![0.0; NRAD - 1];
        c[k] = 1.0;
        for _ in 0..2 {
            for b in &basis {
                let s = dot(&c, b);
                for (ci, bi) in c.iter_mut().zip(b) {
                    *ci -= s * bi;
                }
            }
        }
        let nrm = dot(&c, &c).sqrt();
        c.iter_mut().for_each(|v| *v /= nrm);
        basis.push(c);
    }
    // B_β = (1 + ξ) B̂: coefficients of degree k + 1
    let mut out = vec![{
        let mut c = vec![0.0; NRAD];
        c[0] = 1.0;
        c
    }];
    for b in basis {
        let mut c = vec![0.0; NRAD];
        for (k, v) in b.iter().enumerate() {
            c[k] += v;
            c[k + 1] += v;
        }
        out.push(c);
    }
    out
}

fn radial(coeffs: &[Vec<f64>], x: f64) -> (Vec<f64>, Vec<f64>) {
    let mut b = Vec::with_capacity(NRAD);
    let mut d = Vec::with_capacity(NRAD);
    for c in coeffs {
        let mut v = 0.0;
        let mut dv = 0.0;
        for k in (0..c.len()).rev() {
            dv = dv * x + v;
            v = v * x + c[k];
        }
        b.push(v);
        d.push(dv);
    }
    (b, d)
}

/// The element matrices (row-wise, nn × nn with nn = NRAD · nb, the node order in
/// the element): K_g, C_g, M_g and H_g (see the module documentation).
pub struct InfMatrices {
    pub kg: Vec<f64>,
    pub cg: Vec<f64>,
    pub mg: Vec<f64>,
    pub hg: Vec<f64>,
}

/// The element's local degree of freedom for (base node α, radial function β).
fn local(nb: usize, a: usize, b: usize) -> usize {
    if b == 0 { a } else { nb + a * (NRAD - 1) + (b - 1) }
}

impl AcInf {
    /// The reference point and q_α from the internal nodes.
    fn ref_data(&self, x: &[[f64; 3]]) -> Result<([f64; 3], Vec<f64>)> {
        let nb = self.nb();
        if x.len() != nb * NRAD {
            return fail(format!("{}: the internal nodes are missing", self.name));
        }
        let x0 = x[local(nb, 0, 1)];
        let q = (0..nb).map(|a| x[local(nb, a, 2)][0]).collect();
        Ok((x0, q))
    }

    /// The element matrices for the nodal coordinates `x` (including internal nodes) and the thickness.
    pub fn matrices(&self, x: &[[f64; 3]], thickness: f64) -> Result<InfMatrices> {
        let nb = self.nb();
        let (x0, q) = self.ref_data(x)?;
        let sdim = if self.base.surface() { 3 } else { 2 };
        // r_α n_α and r_α q_α
        let mut y = vec![[0.0; 3]; nb];
        let mut rq = vec![0.0; nb];
        for a in 0..nb {
            let d = [x[a][0] - x0[0], x[a][1] - x0[1], x[a][2] - x0[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if r <= 0.0 {
                return fail(format!("{}: a base node lies at the reference point", self.name));
            }
            y[a] = d;
            rq[a] = r * q[a];
        }
        let a_exp = if self.decay3() { 4 } else { 3 };
        let coeffs = radial_coeffs(a_exp);
        let nn = nb * NRAD;
        let mut kg = vec![0.0; nn * nn];
        let mut cg = vec![0.0; nn * nn];
        let mut mg = vec![0.0; nn * nn];
        let mut hg = vec![0.0; nn * nn];
        let rad = gauss_legendre(20);
        for (gh, wb) in self.base.rule() {
            let (bn, bd) = self.base.shape(gh[0], gh[1]);
            // y(g, h), dy/dg, dy/dh, Q, dQ
            let mut yy = [0.0; 3];
            let mut yg = [0.0; 3];
            let mut yh = [0.0; 3];
            let (mut qv, mut qg, mut qh) = (0.0, 0.0, 0.0);
            for a in 0..nb {
                for d in 0..3 {
                    yy[d] += bn[a] * y[a][d];
                    yg[d] += bd[a][0] * y[a][d];
                    yh[d] += bd[a][1] * y[a][d];
                }
                qv += bn[a] * rq[a];
                qg += bd[a][0] * rq[a];
                qh += bd[a][1] * rq[a];
            }
            for &(xi, wr) in &rad {
                let s = 2.0 / (1.0 - xi);
                let sp = 2.0 / ((1.0 - xi) * (1.0 - xi));
                // Jacobian: columns ∂x/∂g, (∂x/∂h), ∂x/∂ξ
                let cols: Vec<[f64; 3]> = if sdim == 3 {
                    vec![[s * yg[0], s * yg[1], s * yg[2]], [s * yh[0], s * yh[1], s * yh[2]], [sp * yy[0], sp * yy[1], sp * yy[2]]]
                } else {
                    vec![[s * yg[0], s * yg[1], 0.0], [sp * yy[0], sp * yy[1], 0.0]]
                };
                let (det, inv) = invert(&cols, sdim)?;
                let mut wv = wb * wr * det.abs();
                if self.axisym {
                    wv *= 2.0 * std::f64::consts::PI * (x0[0] + s * yy[0]);
                } else if sdim == 2 {
                    wv *= thickness;
                }
                // physical gradients of the parameters: ∇a = row a of J⁻¹
                let grad = |dpar: &[f64]| -> [f64; 3] {
                    let mut g = [0.0; 3];
                    for (k, dp) in dpar.iter().enumerate() {
                        for d in 0..sdim {
                            g[d] += dp * inv[k][d];
                        }
                    }
                    g
                };
                let om = 0.5 * (1.0 - xi); // r₀/r
                let (f, df) = if self.decay3() { (om, -0.5) } else { (om.sqrt(), -0.25 / om.sqrt()) };
                let big_f = om * om;
                let d_big_f = -om;
                let (bb, bdx) = radial(&coeffs, xi);
                // μ = Q (1 + ξ)/(1 − ξ)
                let t = (1.0 + xi) / (1.0 - xi);
                let dt = 2.0 / ((1.0 - xi) * (1.0 - xi));
                let dmu = if sdim == 3 { grad(&[qg * t, qh * t, qv * dt]) } else { grad(&[qg * t, qv * dt]) };
                let mu2 = dmu[0] * dmu[0] + dmu[1] * dmu[1] + dmu[2] * dmu[2];
                // φ and ∇φ, Fφ and ∇(Fφ) per local degree of freedom
                let mut phi = vec![0.0; nn];
                let mut gphi = vec![[0.0; 3]; nn];
                let mut fphi = vec![0.0; nn];
                let mut gfphi = vec![[0.0; 3]; nn];
                for a in 0..nb {
                    for b in 0..NRAD {
                        let i = local(nb, a, b);
                        let v = f * bn[a] * bb[b];
                        let dg = f * bd[a][0] * bb[b];
                        let dh = f * bd[a][1] * bb[b];
                        let dx = bn[a] * (df * bb[b] + f * bdx[b]);
                        phi[i] = v;
                        gphi[i] = if sdim == 3 { grad(&[dg, dh, dx]) } else { grad(&[dg, dx]) };
                        fphi[i] = big_f * v;
                        let fdx = d_big_f * v + big_f * dx;
                        gfphi[i] = if sdim == 3 { grad(&[big_f * dg, big_f * dh, fdx]) } else { grad(&[big_f * dg, fdx]) };
                    }
                }
                let dot = |a: &[f64; 3], b: &[f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
                for i in 0..nn {
                    let gfm = dot(&gfphi[i], &dmu);
                    for j in 0..nn {
                        let k = i * nn + j;
                        kg[k] += wv * dot(&gfphi[i], &gphi[j]);
                        cg[k] += wv * (gfm * phi[j] - fphi[i] * dot(&dmu, &gphi[j]));
                        mg[k] += wv * (1.0 - mu2) * fphi[i] * phi[j];
                        hg[k] += wv * fphi[i] * phi[j];
                    }
                }
            }
        }
        Ok(InfMatrices { kg, cg, mg, hg })
    }
}

/// Determinant and inverse (rows = parameters, columns = x, y, z) for a
/// Jacobian with the columns `cols` (∂x/∂parameter).
fn invert(cols: &[[f64; 3]], n: usize) -> Result<(f64, [[f64; 3]; 3])> {
    let mut inv = [[0.0; 3]; 3];
    if n == 2 {
        // J = [[x_g, x_ξ], [y_g, y_ξ]]
        let (a, b, c, d) = (cols[0][0], cols[1][0], cols[0][1], cols[1][1]);
        let det = a * d - b * c;
        if det.abs() < 1e-300 {
            return fail("acoustic infinite element: singular mapping");
        }
        inv[0] = [d / det, -b / det, 0.0];
        inv[1] = [-c / det, a / det, 0.0];
        return Ok((det, inv));
    }
    let j = |r: usize, c: usize| cols[c][r];
    let det = j(0, 0) * (j(1, 1) * j(2, 2) - j(1, 2) * j(2, 1)) - j(0, 1) * (j(1, 0) * j(2, 2) - j(1, 2) * j(2, 0)) + j(0, 2) * (j(1, 0) * j(2, 1) - j(1, 1) * j(2, 0));
    if det.abs() < 1e-300 {
        return fail("acoustic infinite element: singular mapping");
    }
    // (J⁻¹)[p][x]: parameter p, coordinate x
    for p in 0..3 {
        for xx in 0..3 {
            let (r1, r2) = ((xx + 1) % 3, (xx + 2) % 3);
            let (c1, c2) = ((p + 1) % 3, (p + 2) % 3);
            inv[p][xx] = (j(r1, c1) * j(r2, c2) - j(r1, c2) * j(r2, c1)) / det;
        }
    }
    Ok((det, inv))
}

/// (ρ_f, K_f) for the element.
fn props(ctx: &ElemCtx, name: &str) -> Result<(f64, f64)> {
    let m = ctx.material(name)?;
    let t = m.tangent_linear(StressMode::ThreeD)?;
    let bulk = t.first().copied().unwrap_or(0.0);
    if m.name() != "acoustic medium" || bulk <= 0.0 {
        return fail(format!("element {} ({name}) requires an acoustic medium (*ACOUSTIC MEDIUM)", ctx.label));
    }
    Ok((ctx.density, bulk))
}

impl ElementType for AcInf {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Other
    }
    fn dim(&self) -> usize {
        if self.base.surface() { 3 } else { 2 }
    }
    fn n_nodes(&self) -> usize {
        self.nb() * NRAD
    }
    fn n_connectivity(&self) -> usize {
        self.nb()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[POR]
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        match self.base {
            Base::Line2 => &[&[0, 1]],
            Base::Line3 => &[&[0, 1, 2]],
            Base::Tri3 => &[&[0, 1, 2]],
            Base::Quad4 => &[&[0, 1, 2, 3]],
            Base::Tri6 => &[&[0, 1, 2, 3, 4, 5]],
            Base::Quad8 => &[&[0, 1, 2, 3, 4, 5, 6, 7]],
        }
    }
    /// Zero: the matrices are built by the acoustic steps (unsymmetric, frequency-dependent).
    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        props(ctx, self.name)?;
        let n = self.n_nodes();
        Ok(vec![0.0; n * n])
    }
    fn mass(&self, _ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        let n = self.n_nodes();
        Ok(vec![0.0; n * n])
    }
}

/// The element's surface normal at base node `a` (unit vector, directed away from `x0`).
fn facet_normal(e: &AcInf, x: &[[f64; 3]], a: usize, x0: [f64; 3]) -> [f64; 3] {
    let p = e.base.node_coords()[a];
    let (_, d) = e.base.shape(p[0], p[1]);
    let nb = e.nb();
    let mut tg = [0.0; 3];
    let mut th = [0.0; 3];
    for k in 0..nb {
        for c in 0..3 {
            tg[c] += d[k][0] * x[k][c];
            th[c] += d[k][1] * x[k][c];
        }
    }
    let mut n = if e.base.surface() {
        [tg[1] * th[2] - tg[2] * th[1], tg[2] * th[0] - tg[0] * th[2], tg[0] * th[1] - tg[1] * th[0]]
    } else {
        [tg[1], -tg[0], 0.0]
    };
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 {
        n.iter_mut().for_each(|v| *v /= l);
    }
    // direction: away from the reference point (the face centroid)
    let mut c = [0.0; 3];
    for xk in x.iter().take(nb) {
        for k in 0..3 {
            c[k] += xk[k] / nb as f64;
        }
    }
    let s = n[0] * (c[0] - x0[0]) + n[1] * (c[1] - x0[1]) + n[2] * (c[2] - x0[2]);
    if s < 0.0 {
        n.iter_mut().for_each(|v| *v = -*v);
    }
    n
}

/// Creates the internal nodes for all acoustic infinite elements (called from
/// `FeModel::finalize`): reference point, q_α and nine shared nodes per base node.
pub fn add_nodes(m: &mut FeModel) -> Result<()> {
    let elems: Vec<usize> = (0..m.elements.len())
        .filter(|&e| lookup(m.elements[e].etype.name()).is_some_and(|a| m.elements[e].nodes.len() == a.nb()))
        .collect();
    if elems.is_empty() {
        return Ok(());
    }
    // reference point per section
    let mut x0s: Vec<Option<[f64; 3]>> = vec![None; m.sections.len()];
    for &e in &elems {
        let Some(si) = m.elements[e].section else {
            return fail(format!("{} has no section (*SOLID SECTION with a reference point)", m.elem_name(e)));
        };
        if x0s[si].is_some() {
            continue;
        }
        let s = &m.sections[si];
        let x0 = if let Some(r) = s.raw.param("REF NODE") {
            let r = r.trim();
            let inst = m.elements[e].inst;
            let nd = match r.parse::<i64>() {
                Ok(l) => m.node(inst, l),
                Err(_) => m.nsets.get(&r.to_uppercase()).and_then(|v| if v.len() == 1 { Some(v[0]) } else { None }),
            };
            match nd {
                Some(n) => m.nodes[n],
                None => return s.raw.err(format!("REF NODE={r}: the reference node does not exist (a node number or a set with one node)")),
            }
        } else {
            let v = s.raw.first_floats().unwrap_or_default();
            if v.len() < 3 {
                return s.raw.err("acoustic infinite elements require a reference point: REF NODE= or the coordinates after the thickness on the data line");
            }
            [v[1], v[2], v.get(3).copied().unwrap_or(0.0)]
        };
        x0s[si] = Some(x0);
    }
    // q_α = min n_α · n_E
    let mut q: std::collections::HashMap<usize, (f64, [f64; 3])> = std::collections::HashMap::new();
    for &e in &elems {
        let a = lookup(m.elements[e].etype.name()).unwrap();
        let x0 = x0s[m.elements[e].section.unwrap()].unwrap();
        let x: Vec<[f64; 3]> = m.elements[e].nodes.iter().map(|&n| m.nodes[n]).collect();
        for k in 0..a.nb() {
            let nd = m.elements[e].nodes[k];
            let d = [x[k][0] - x0[0], x[k][1] - x0[1], x[k][2] - x0[2]];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let ray = [d[0] / r, d[1] / r, d[2] / r];
            let ne = facet_normal(a, &x, k, x0);
            let c = ray[0] * ne[0] + ray[1] * ne[1] + ray[2] * ne[2];
            let ent = q.entry(nd).or_insert((f64::INFINITY, x0));
            if (ent.1[0] - x0[0]).abs() + (ent.1[1] - x0[1]).abs() + (ent.1[2] - x0[2]).abs() > 1e-12 * (1.0 + r) {
                return fail(format!("{}: the node is shared by acoustic infinite elements with different reference points", m.elem_name(e)));
            }
            ent.0 = ent.0.min(c);
        }
    }
    for (&nd, &(c, _)) in &q {
        if c <= 0.0 {
            return fail(format!("acoustic infinite element: negative cosine ({c:.4}) at node {} – the reference point is misplaced", m.node_label[nd]));
        }
    }
    for &e in &elems {
        let a = lookup(m.elements[e].etype.name()).unwrap();
        let inst = m.elements[e].inst;
        let mut extra = Vec::new();
        for k in 0..a.nb() {
            let nd = m.elements[e].nodes[k];
            let (c, x0) = q[&nd];
            let label = m.node_label[nd];
            for j in 0..NRAD - 1 {
                let xj = match j {
                    0 => x0,
                    1 => [c, 0.0, 0.0],
                    _ => m.nodes[nd],
                };
                let lab = -(3_000_000_000 + 16 * label + j as i64);
                let id = match m.node(inst, lab) {
                    Some(i) => i,
                    None => m.add_node(inst, lab, xj),
                };
                extra.push(id);
            }
        }
        m.elements[e].nodes.extend(extra);
    }
    Ok(())
}
