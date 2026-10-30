//! Three-dimensional first-order continuum elements: C3D8.
//!
//! C3D8: trilinear hexahedron, full 2×2×2 integration with B-bar where
//! the volumetric strain is averaged over the element (selectively
//! reduced integration). Higher-order and reduced-integration
//! elements (C3D20, C3D10, C3D8R, C3D8I, ...) belong in `solid_higher.rs`.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, inv3, Bbar, Ip, DISTORTED, G2};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::materials::StressMode;
use crate::model::Result;

pub struct Hex8 {
    name: &'static str,
}

pub static C3D8: Hex8 = Hex8 { name: "C3D8" };

pub static TYPES: &[&dyn ElementType] = &[&C3D8];

/// Faces in the .inp format numbering (S1–S6).
pub const HEX8_FACES: &[&[usize]] = &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]];

/// Natural coordinates of the hexahedron corners.
pub const HEX8_NAT: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Integration point for a trilinear hexahedron at natural coordinates `p` with weight `wg`.
pub fn hex8_ip(x: &[[f64; 3]], p: [f64; 3], wg: f64) -> Result<Ip> {
    let mut n = vec![0.0; 8];
    let mut dn = [[0.0; 8]; 3];
    for i in 0..8 {
        let nat = HEX8_NAT[i];
        let f = [1.0 + p[0] * nat[0], 1.0 + p[1] * nat[1], 1.0 + p[2] * nat[2]];
        n[i] = 0.125 * f[0] * f[1] * f[2];
        dn[0][i] = 0.125 * nat[0] * f[1] * f[2];
        dn[1][i] = 0.125 * nat[1] * f[0] * f[2];
        dn[2][i] = 0.125 * nat[2] * f[0] * f[1];
    }
    let mut j = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            j[r][c] = (0..8).map(|i| dn[r][i] * x[i][c]).sum();
        }
    }
    let (ji, det) = inv3(j);
    if det <= 0.0 {
        return crate::model::fail(DISTORTED);
    }
    let dndx = (0..8)
        .map(|i| {
            let mut d = [0.0; 3];
            for c in 0..3 {
                d[c] = (0..3).map(|r| ji[c][r] * dn[r][i]).sum();
            }
            d
        })
        .collect();
    Ok(Ip { n, dndx, w: det * wg })
}

/// 2×2×2 Gauss points (ξ fastest, then η, then ζ).
pub fn hex8_ips(x: &[[f64; 3]]) -> Result<Vec<Ip>> {
    let mut ips = Vec::with_capacity(8);
    for &zeta in &[-G2, G2] {
        for &eta in &[-G2, G2] {
            for &xi in &[-G2, G2] {
                ips.push(hex8_ip(x, [xi, eta, zeta], 1.0)?);
            }
        }
    }
    Ok(ips)
}

const MODE: StressMode = StressMode::ThreeD;

impl ElementType for Hex8 {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Continuum3D
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        8
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3]
    }
    fn n_ips(&self) -> usize {
        8
    }
    fn vtk_type(&self) -> u8 {
        12
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        HEX8_FACES
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(8, MODE, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let ips = hex8_ips(&ctx.x)?;
        let bs = common::bmats(MODE, &ips, Bbar::Volumetric);
        common::stiffness_ctx(ctx, MODE, &ips, &bs, ctx.material(self.name)?)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // area nlgeom: replace the line below with a call to the nonlinear kinematics
            return crate::kinematics::continuum::response(ctx, MODE, &hex8_ips(&ctx.x)?, Bbar::Volumetric, ctx.material(self.name)?, u, du, so, sn, tangent);
        }
        let ips = hex8_ips(&ctx.x)?;
        let bs = common::bmats(MODE, &ips, Bbar::Volumetric);
        common::response(ctx, MODE, &ips, &bs, ctx.material(self.name)?, u, du, so, sn, tangent)
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(common::mass(&hex8_ips(&ctx.x)?, ctx.density, 3, kind == MassKind::Lumped))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        if let Some(f) = common::pressure_load(&ctx.x, HEX8_FACES, &load.kind, load.magnitude, 1.0, 3, self.name)? {
            return Ok(f);
        }
        if let Some(b) = common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)? {
            return Ok(common::body_force(&hex8_ips(&ctx.x)?, b, 3));
        }
        super::api::not_implemented(&format!("load type {}", load.kind), self.name)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        super::geometric::continuum_bbar(MODE, &hex8_ips(&ctx.x)?, ctx.material(self.name)?, state, Bbar::Volumetric)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::volume_weights(&hex8_ips(&ctx.x)?))
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let ips = hex8_ips(&ctx.x)?;
        let bs = common::bmats(MODE, &ips, Bbar::Volumetric);
        let mut out = common::output_ctx(ctx, MODE, &ips, &bs, ctx.material(self.name)?, u, state);
        if ctx.nlgeom {
            crate::kinematics::continuum::initial_log_strain(MODE, &ips, Bbar::Volumetric, u, state, &mut out);
        }
        Ok(out)
    }
}
