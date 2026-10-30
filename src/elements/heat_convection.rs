//! Elements for forced convection/diffusion:
//! DCC1D2, DCCAX2 (links), DCC2D4, DCCAX4 (quadrilaterals) and
//! DCC3D8 (hexahedron), plus the variants with dispersion control (…D).
//!
//! Formulation (Yu–Heinrich): Petrov–
//! Galerkin with the weight functions Wᴺ = Nᴺ + α h/2 v̂·∇Nᴺ, where v̄ is the element's
//! mean velocity (mass flow per area / ρ, mean of the nodal values), h = Σ_α |h_α·v̂|
//! (isoparametric lines through the center), γ = |v̄| h ρc/k (local
//! Péclet number) and α = coth(γ/2) − 2/γ ("upwind weighting"):
//!
//! - convection  K_c = ∫ c Wᴺ (G·∇Nᴹ) dV, G = ρv interpolated from the nodes;
//! - conduction  K_k = ∫ ∇Nᴺ·k ∇Nᴹ dV (the second-derivative term α h/2 v̂·∇Nᴺ k:∇∇Nᴹ
//!   vanishes for parallelogram-shaped elements and is not included);
//! - capacity    C   = ∫ ρc Wᴺ Nᴹ dV, consistent and exactly integrated.
//!
//! Time integration (Crank–Nicolson): C(θ₁ − θ₀)/Δt + ½(K_c + K_k)(θ₁ + θ₀) = F;
//! steady state (K_c + K_k) θ = F. The dispersion control (the β terms in …D) gives
//! an error until it is supported. The links: DCC1D2 has the cross-sectional area from *SOLID
//! SECTION and *MASS FLOW RATE gives the total mass flow (positive from node 1 towards
//! node 2); DCCAX2 has the channel thickness t in the r–z plane (cross-section 2πr t).
//! The matrices are computed by `steps/heat_convection.rs`.

use super::api::{ElemCtx, ElementType, Family};
use crate::fem::dof::{Dof, NT11};
use crate::model::{fail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Link in space (DCC1D2).
    Link,
    /// Axisymmetric link in the r–z plane (DCCAX2).
    LinkAx,
    /// Plane quadrilateral (DCC2D4) with thickness.
    Quad,
    /// Axisymmetric quadrilateral (DCCAX4).
    QuadAx,
    /// Hexahedron (DCC3D8).
    Hex,
}

pub struct ConvElem {
    pub name: &'static str,
    pub kind: Kind,
    /// Dispersion control (…D).
    pub dispersion: bool,
}

pub static DCC1D2: ConvElem = ConvElem { name: "DCC1D2", kind: Kind::Link, dispersion: false };
pub static DCC1D2D: ConvElem = ConvElem { name: "DCC1D2D", kind: Kind::Link, dispersion: true };
pub static DCCAX2: ConvElem = ConvElem { name: "DCCAX2", kind: Kind::LinkAx, dispersion: false };
pub static DCCAX2D: ConvElem = ConvElem { name: "DCCAX2D", kind: Kind::LinkAx, dispersion: true };
pub static DCC2D4: ConvElem = ConvElem { name: "DCC2D4", kind: Kind::Quad, dispersion: false };
pub static DCC2D4D: ConvElem = ConvElem { name: "DCC2D4D", kind: Kind::Quad, dispersion: true };
pub static DCCAX4: ConvElem = ConvElem { name: "DCCAX4", kind: Kind::QuadAx, dispersion: false };
pub static DCCAX4D: ConvElem = ConvElem { name: "DCCAX4D", kind: Kind::QuadAx, dispersion: true };
pub static DCC3D8: ConvElem = ConvElem { name: "DCC3D8", kind: Kind::Hex, dispersion: false };
pub static DCC3D8D: ConvElem = ConvElem { name: "DCC3D8D", kind: Kind::Hex, dispersion: true };

pub static TYPES: &[&dyn ElementType] = &[&DCC1D2, &DCC1D2D, &DCCAX2, &DCCAX2D, &DCC2D4, &DCC2D4D, &DCCAX4, &DCCAX4D, &DCC3D8, &DCC3D8D];
static ALL: &[&ConvElem] = &[&DCC1D2, &DCC1D2D, &DCCAX2, &DCCAX2D, &DCC2D4, &DCC2D4D, &DCCAX4, &DCCAX4D, &DCC3D8, &DCC3D8D];

/// The convection element named `name`.
pub fn lookup(name: &str) -> Option<&'static ConvElem> {
    ALL.iter().copied().find(|e| e.name == name)
}

const LINK_FACES: &[&[usize]] = &[&[0], &[1]];
const QUAD4_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]];
const HEX8_FACES: &[&[usize]] = &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]];

/// An integration point: shape functions, gradients, weight (dV) and radius.
pub struct ConvIp {
    pub n: Vec<f64>,
    pub dndx: Vec<[f64; 3]>,
    pub w: f64,
}

/// The element matrices (row-wise nn × nn).
pub struct ConvMatrices {
    /// K_c + K_k.
    pub k: Vec<f64>,
    /// Capacity (empty in steady-state analysis).
    pub c: Vec<f64>,
    /// The heat flux −k∇Nᴹ per integration point (for HFL): [point][component][node].
    pub flux: Vec<[Vec<f64>; 3]>,
}

const G2: f64 = 0.577_350_269_189_625_8;

impl ConvElem {
    pub fn n_nodes(&self) -> usize {
        match self.kind {
            Kind::Link | Kind::LinkAx => 2,
            Kind::Quad | Kind::QuadAx => 4,
            Kind::Hex => 8,
        }
    }

    pub fn is_link(&self) -> bool {
        matches!(self.kind, Kind::Link | Kind::LinkAx)
    }

    /// Integration points (Gauss, ξ fastest). Links: one point at the center.
    pub fn ips(&self, x: &[[f64; 3]], section: f64) -> Result<Vec<ConvIp>> {
        let two_pi = 2.0 * std::f64::consts::PI;
        match self.kind {
            Kind::Link | Kind::LinkAx => {
                let t = [x[1][0] - x[0][0], x[1][1] - x[0][1], x[1][2] - x[0][2]];
                let l = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
                if l <= 0.0 {
                    return fail("the link has zero length");
                }
                let e = [t[0] / l, t[1] / l, t[2] / l];
                let area = if self.kind == Kind::LinkAx { section * two_pi * 0.5 * (x[0][0] + x[1][0]) } else { section };
                Ok(vec![ConvIp { n: vec![0.5, 0.5], dndx: vec![[-e[0] / l, -e[1] / l, -e[2] / l], [e[0] / l, e[1] / l, e[2] / l]], w: area * l }])
            }
            Kind::Quad | Kind::QuadAx => {
                let mut out = Vec::new();
                for (b, a) in [(-G2, -G2), (-G2, G2), (G2, -G2), (G2, G2)] {
                    let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
                    let n: Vec<f64> = nat.iter().map(|c| 0.25 * (1.0 + c[0] * a) * (1.0 + c[1] * b)).collect();
                    let dn: Vec<[f64; 2]> = nat.iter().map(|c| [0.25 * c[0] * (1.0 + c[1] * b), 0.25 * c[1] * (1.0 + c[0] * a)]).collect();
                    let mut j = [[0.0; 2]; 2];
                    for i in 0..4 {
                        for r in 0..2 {
                            for s in 0..2 {
                                j[r][s] += dn[i][r] * x[i][s];
                            }
                        }
                    }
                    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                    if det <= 0.0 {
                        return fail("the element is distorted (negative Jacobian)");
                    }
                    let inv = [[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]];
                    let dndx: Vec<[f64; 3]> = dn.iter().map(|d| [inv[0][0] * d[0] + inv[0][1] * d[1], inv[1][0] * d[0] + inv[1][1] * d[1], 0.0]).collect();
                    let mut w = det;
                    if self.kind == Kind::QuadAx {
                        let r: f64 = (0..4).map(|i| n[i] * x[i][0]).sum();
                        w *= two_pi * r;
                    } else {
                        w *= section;
                    }
                    out.push(ConvIp { n, dndx, w });
                }
                // the order ξ fastest: (−,−), (+,−), (−,+), (+,+)
                Ok(out)
            }
            Kind::Hex => {
                let nat = [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]];
                let mut out = Vec::new();
                for &c3 in &[-G2, G2] {
                    for &c2 in &[-G2, G2] {
                        for &c1 in &[-G2, G2] {
                            let p = [c1, c2, c3];
                            let n: Vec<f64> = nat.iter().map(|c| 0.125 * (1.0 + c[0] * p[0]) * (1.0 + c[1] * p[1]) * (1.0 + c[2] * p[2])).collect();
                            let dn: Vec<[f64; 3]> = nat
                                .iter()
                                .map(|c| {
                                    [
                                        0.125 * c[0] * (1.0 + c[1] * p[1]) * (1.0 + c[2] * p[2]),
                                        0.125 * c[1] * (1.0 + c[0] * p[0]) * (1.0 + c[2] * p[2]),
                                        0.125 * c[2] * (1.0 + c[0] * p[0]) * (1.0 + c[1] * p[1]),
                                    ]
                                })
                                .collect();
                            let mut j = [[0.0; 3]; 3];
                            for i in 0..8 {
                                for r in 0..3 {
                                    for s in 0..3 {
                                        j[r][s] += dn[i][r] * x[i][s];
                                    }
                                }
                            }
                            let (inv, det) = super::common::inv3(j);
                            if det <= 0.0 {
                                return fail("the element is distorted (negative Jacobian)");
                            }
                            let dndx: Vec<[f64; 3]> = dn.iter().map(|d| [0, 1, 2].map(|r| (0..3).map(|s| inv[r][s] * d[s]).sum::<f64>())).collect();
                            out.push(ConvIp { n, dndx, w: det });
                        }
                    }
                }
                Ok(out)
            }
        }
    }

    /// Isoparametric lines through the center (h_α), for h = Σ|h_α·v̂|.
    fn iso_lines(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let mid = |ids: &[usize]| {
            let mut c = [0.0; 3];
            for &i in ids {
                for k in 0..3 {
                    c[k] += x[i][k] / ids.len() as f64;
                }
            }
            c
        };
        let d = |a: [f64; 3], b: [f64; 3]| [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        match self.kind {
            Kind::Link | Kind::LinkAx => vec![d(x[0], x[1])],
            Kind::Quad | Kind::QuadAx => vec![d(mid(&[0, 3]), mid(&[1, 2])), d(mid(&[0, 1]), mid(&[3, 2]))],
            Kind::Hex => vec![d(mid(&[0, 3, 4, 7]), mid(&[1, 2, 5, 6])), d(mid(&[0, 1, 4, 5]), mid(&[3, 2, 7, 6])), d(mid(&[0, 1, 2, 3]), mid(&[4, 5, 6, 7]))],
        }
    }

    /// Element and capacity matrices. `g` = mass flow per area (ρv) per node
    /// (links: the vector along the link), `k` = the conductivity, `rho`, `c`.
    #[allow(clippy::too_many_arguments)]
    pub fn matrices(&self, x: &[[f64; 3]], section: f64, k: [[f64; 3]; 3], rho: f64, c: f64, g: &[[f64; 3]], transient: bool, lumped: bool) -> Result<ConvMatrices> {
        let nn = self.n_nodes();
        let ips = self.ips(x, section)?;
        // the element's mean velocity and direction
        let mut gm = [0.0; 3];
        for gn in g {
            for d in 0..3 {
                gm[d] += gn[d] / nn as f64;
            }
        }
        let gl = (gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]).sqrt();
        let (vhat, alpha_h) = if gl > 0.0 && rho > 0.0 && c > 0.0 {
            let vh = [gm[0] / gl, gm[1] / gl, gm[2] / gl];
            let h: f64 = self.iso_lines(x).iter().map(|l| (l[0] * vh[0] + l[1] * vh[1] + l[2] * vh[2]).abs()).sum();
            let kv: f64 = (0..3).map(|r| (0..3).map(|s| vh[r] * k[r][s] * vh[s]).sum::<f64>()).sum();
            // γ = |v| h ρ c / k with |v| = |G|/ρ
            let gamma = if kv > 0.0 { gl * h * c / kv } else { f64::INFINITY };
            let alpha = if gamma.is_infinite() {
                1.0
            } else if gamma < 1e-6 {
                gamma / 6.0
            } else {
                1.0 / (0.5 * gamma).tanh() - 2.0 / gamma
            };
            (vh, alpha * h / 2.0)
        } else {
            ([0.0; 3], 0.0)
        };
        let mut km = vec![0.0; nn * nn];
        let mut cm = if transient { vec![0.0; nn * nn] } else { Vec::new() };
        let mut flux = Vec::with_capacity(ips.len());
        for ip in &ips {
            let mut gp = [0.0; 3];
            for (i, gn) in g.iter().enumerate() {
                for d in 0..3 {
                    gp[d] += ip.n[i] * gn[d];
                }
            }
            let wt: Vec<f64> = (0..nn).map(|i| ip.n[i] + alpha_h * (vhat[0] * ip.dndx[i][0] + vhat[1] * ip.dndx[i][1] + vhat[2] * ip.dndx[i][2])).collect();
            for a in 0..nn {
                let kda = [0, 1, 2].map(|r| (0..3).map(|s| k[r][s] * ip.dndx[a][s]).sum::<f64>());
                for b in 0..nn {
                    let db = ip.dndx[b];
                    let cond = kda[0] * db[0] + kda[1] * db[1] + kda[2] * db[2];
                    let conv = c * wt[a] * (gp[0] * db[0] + gp[1] * db[1] + gp[2] * db[2]);
                    km[a * nn + b] += ip.w * (cond + conv);
                    if transient && !lumped {
                        cm[a * nn + b] += ip.w * rho * c * wt[a] * ip.n[b];
                    }
                }
            }
            if transient && lumped {
                // diagonal Galerkin part, the upwind part consistent
                for a in 0..nn {
                    cm[a * nn + a] += ip.w * rho * c * ip.n[a];
                    for b in 0..nn {
                        cm[a * nn + b] += ip.w * rho * c * (wt[a] - ip.n[a]) * ip.n[b];
                    }
                }
            }
            let mut f: [Vec<f64>; 3] = [vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]];
            for r in 0..3 {
                for b in 0..nn {
                    f[r][b] = -(0..3).map(|s| k[r][s] * ip.dndx[b][s]).sum::<f64>();
                }
            }
            flux.push(f);
        }
        if transient && !lumped && self.is_link() {
            // links have one integration point: the consistent capacity
            // is integrated exactly (Nᵃ Nᵇ: L/3, L/6; the upwind part constant · L/2)
            let ip = &ips[0];
            for a in 0..nn {
                let up = alpha_h * (vhat[0] * ip.dndx[a][0] + vhat[1] * ip.dndx[a][1] + vhat[2] * ip.dndx[a][2]);
                for b in 0..nn {
                    let nn_ab = if a == b { 1.0 / 3.0 } else { 1.0 / 6.0 };
                    cm[a * nn + b] = ip.w * rho * c * (nn_ab + 0.5 * up);
                }
            }
        }
        Ok(ConvMatrices { k: km, c: cm, flux })
    }
}

impl ElementType for ConvElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Heat
    }
    fn dim(&self) -> usize {
        match self.kind {
            Kind::Link | Kind::Hex => 3,
            _ => 2,
        }
    }
    fn n_nodes(&self) -> usize {
        ConvElem::n_nodes(self)
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[NT11]
    }
    fn n_ips(&self) -> usize {
        match self.kind {
            Kind::Link | Kind::LinkAx => 1,
            Kind::Quad | Kind::QuadAx => 4,
            Kind::Hex => 8,
        }
    }
    fn vtk_type(&self) -> u8 {
        match self.kind {
            Kind::Link | Kind::LinkAx => 3,
            Kind::Quad | Kind::QuadAx => 9,
            Kind::Hex => 12,
        }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        match self.kind {
            Kind::Link | Kind::LinkAx => LINK_FACES,
            Kind::Quad | Kind::QuadAx => QUAD4_FACES,
            Kind::Hex => HEX8_FACES,
        }
    }
    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let t = ctx.section.map(|s| s.solid_thickness()).unwrap_or(1.0);
        Ok(self.ips(&ctx.x, t)?.iter().map(|ip| ip.w).collect())
    }
}
