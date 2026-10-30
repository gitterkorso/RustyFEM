//! Plana kontinuumelement: CPS3, CPS4 (plan spänning) och CPE3, CPE4 (plan töjning).
//!
//! - CPS4: bilinjär fyrhörning, full 2×2-integrering.
//! - CPE4: full integrering med B-bar där den plana volymtöjningen ε11+ε22
//!   medelvärdesbildas och ε33 = 0 behålls (selektivt reducerad integrering).
//! - CPS3/CPE3: konstant töjning.
//!
//! Integrationspunkterna numreras med ξ snabbast.

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Bbar, Ip, DISTORTED, G2};
use crate::fem::dof::{Dof, U1, U2};
use crate::fem::model::DistLoad;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub struct Continuum2D {
    name: &'static str,
    nn: usize,
    mode: StressMode,
    bbar: Bbar,
}

pub static CPS3: Continuum2D = Continuum2D { name: "CPS3", nn: 3, mode: StressMode::PlaneStress, bbar: Bbar::None };
pub static CPS4: Continuum2D = Continuum2D { name: "CPS4", nn: 4, mode: StressMode::PlaneStress, bbar: Bbar::None };
pub static CPE3: Continuum2D = Continuum2D { name: "CPE3", nn: 3, mode: StressMode::PlaneStrain, bbar: Bbar::None };
pub static CPE4: Continuum2D = Continuum2D { name: "CPE4", nn: 4, mode: StressMode::PlaneStrain, bbar: Bbar::Planar };

pub static TYPES: &[&dyn ElementType] = &[&CPE3, &CPE4, &CPS3, &CPS4];

const TRI_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 0]];
const QUAD_FACES: &[&[usize]] = &[&[0, 1], &[1, 2], &[2, 3], &[3, 0]];

/// Integrationspunkter för linjär triangel eller bilinjär fyrhörning.
pub fn ips_2d(x: &[[f64; 3]], thickness: f64) -> Result<Vec<Ip>> {
    match x.len() {
        3 => {
            let (x1, y1, x2, y2, x3, y3) = (x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1]);
            let det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
            if det <= 0.0 {
                return fail(DISTORTED);
            }
            let dndx = vec![
                [(y2 - y3) / det, (x3 - x2) / det, 0.0],
                [(y3 - y1) / det, (x1 - x3) / det, 0.0],
                [(y1 - y2) / det, (x2 - x1) / det, 0.0],
            ];
            Ok(vec![Ip { n: vec![1.0 / 3.0; 3], dndx, w: 0.5 * det * thickness }])
        }
        4 => {
            let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let mut ips = Vec::with_capacity(4);
            for &(xi, eta) in &[(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)] {
                let mut n = vec![0.0; 4];
                let mut dxi = [0.0; 4];
                let mut deta = [0.0; 4];
                for i in 0..4 {
                    n[i] = 0.25 * (1.0 + xi * nat[i][0]) * (1.0 + eta * nat[i][1]);
                    dxi[i] = 0.25 * nat[i][0] * (1.0 + eta * nat[i][1]);
                    deta[i] = 0.25 * nat[i][1] * (1.0 + xi * nat[i][0]);
                }
                let mut j = [[0.0; 2]; 2];
                for i in 0..4 {
                    j[0][0] += dxi[i] * x[i][0];
                    j[0][1] += dxi[i] * x[i][1];
                    j[1][0] += deta[i] * x[i][0];
                    j[1][1] += deta[i] * x[i][1];
                }
                let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                if det <= 0.0 {
                    return fail(DISTORTED);
                }
                let dndx = (0..4)
                    .map(|i| [(j[1][1] * dxi[i] - j[0][1] * deta[i]) / det, (-j[1][0] * dxi[i] + j[0][0] * deta[i]) / det, 0.0])
                    .collect();
                ips.push(Ip { n, dndx, w: det * thickness });
            }
            Ok(ips)
        }
        n => fail(format!("intern: ips_2d med {n} noder")),
    }
}

impl Continuum2D {
    fn ips(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let t = ctx.section(self.name)?.solid_thickness();
        ips_2d(&ctx.x, t)
    }
}

impl ElementType for Continuum2D {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Continuum2D
    }
    fn dim(&self) -> usize {
        2
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2]
    }
    fn n_ips(&self) -> usize {
        if self.nn == 3 { 1 } else { 4 }
    }
    fn vtk_type(&self) -> u8 {
        if self.nn == 3 { 5 } else { 9 }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.nn == 3 { TRI_FACES } else { QUAD_FACES }
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(self.n_ips(), self.mode, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let ips = self.ips(ctx)?;
        let bs = common::bmats(self.mode, &ips, self.bbar);
        common::stiffness_ctx(ctx, self.mode, &ips, &bs, ctx.material(self.name)?)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // område nlgeom: ersätt raden nedan med ett anrop till den olinjära kinematiken
            return crate::kinematics::continuum::response(ctx, self.mode, &self.ips(ctx)?, self.bbar, ctx.material(self.name)?, u, du, so, sn, tangent);
        }
        let ips = self.ips(ctx)?;
        let bs = common::bmats(self.mode, &ips, self.bbar);
        common::response(ctx, self.mode, &ips, &bs, ctx.material(self.name)?, u, du, so, sn, tangent)
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(common::mass(&self.ips(ctx)?, ctx.density, 2, kind == MassKind::Lumped))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let t = ctx.section(self.name)?.solid_thickness();
        if let Some(f) = common::pressure_load(&ctx.x, self.faces(), &load.kind, load.magnitude, t, 2, self.name)? {
            return Ok(f);
        }
        if let Some(b) = common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)? {
            return Ok(common::body_force(&self.ips(ctx)?, b, 2));
        }
        super::api::not_implemented(&format!("lasttypen {}", load.kind), self.name)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        super::geometric::continuum_bbar(self.mode, &self.ips(ctx)?, ctx.material(self.name)?, state, self.bbar)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::volume_weights(&self.ips(ctx)?))
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let ips = self.ips(ctx)?;
        let bs = common::bmats(self.mode, &ips, self.bbar);
        let mut out = common::output_ctx(ctx, self.mode, &ips, &bs, ctx.material(self.name)?, u, state);
        if ctx.nlgeom {
            crate::kinematics::continuum::initial_log_strain(self.mode, &ips, self.bbar, u, state, &mut out);
        }
        Ok(out)
    }
}
