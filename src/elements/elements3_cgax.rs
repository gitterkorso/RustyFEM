//! CGAX3, CGAX4, CGAX6, CGAX8, CGAX8R: generaliserade axisymmetriska
//! element med vridning (område elements3).
//!
//! Frihetsgrader per nod: u_r (1), u_z (2) och vridningsvinkeln φ kring
//! symmetriaxeln (5, i radianer). θ-axeln är e_r × e_z (högersystemet
//! r, z, θ), så u_θ = −r φ. Töjningarnas ordning är
//! [εrr, εzz, εθθ, γrz, γrθ, γzθ] med εθθ = u_r/r, γrθ = −r ∂φ/∂r och
//! γzθ = −r ∂φ/∂z; materialet används i sin
//! tredimensionella form (S11 = σrr, S22 = σzz, S33 = σθθ, S12 = σrz,
//! S13 = σrθ, S23 = σzθ).
//!
//! I planet är elementen som CAX-elementen i `axisym.rs` (samma formfunktioner,
//! integrationsregler och B-bar för CGAX4); allt integreras över hela varvet
//! (vikten 2πr). CGAX3 (en punkt) tar radien i vridningstermerna i volymens
//! tyngdpunkt ∫r² dA/∫r dA. Massan har rotationströgheten ρ r² för φ
//! och är radsummerad för CGAX3/CGAX4 (se `tests/elements3.rs`).

use super::super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::super::common::{self, Ip};
use super::super::solid_higher::shapes::Shape;
use super::super::solid_higher::{apply_bbar, face_pressure, ips_for, radius, RuleKind};
use crate::fem::dof::{Dof, U1, U2, UR2};
use crate::fem::model::DistLoad;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub struct Cgax {
    pub name: &'static str,
    pub shape: Shape,
    pub rule: RuleKind,
    pub bbar: bool,
}

pub static CGAX3: Cgax = Cgax { name: "CGAX3", shape: Shape::Tri3, rule: RuleKind::Tri(1), bbar: false };
pub static CGAX4: Cgax = Cgax { name: "CGAX4", shape: Shape::Quad4, rule: RuleKind::Gauss(2), bbar: true };
pub static CGAX6: Cgax = Cgax { name: "CGAX6", shape: Shape::Tri6, rule: RuleKind::Tri(3), bbar: false };
pub static CGAX8: Cgax = Cgax { name: "CGAX8", shape: Shape::Quad8, rule: RuleKind::Gauss(3), bbar: false };
pub static CGAX8R: Cgax = Cgax { name: "CGAX8R", shape: Shape::Quad8, rule: RuleKind::Gauss(2), bbar: false };

const MODE: StressMode = StressMode::ThreeD;

impl Cgax {
    fn ips(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        ips_for(self.shape, StressMode::Axisymmetric, &ctx.x, &self.rule.points(2), 1.0).map_err(|e| crate::model::FemError(format!("element {} ({}): {e}", ctx.label, self.name)))
    }

    fn ips_exact(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let rule = match self.shape {
            Shape::Tri3 | Shape::Tri6 => RuleKind::Tri(7),
            _ => RuleKind::Gauss(4),
        };
        ips_for(self.shape, StressMode::Axisymmetric, &ctx.x, &rule.points(2), 1.0)
    }

    /// Massans regel (som CAX-elementen, `solid_higher_tech.rs::mass_rule`).
    fn ips_mass(&self, ctx: &ElemCtx) -> Result<Vec<Ip>> {
        let rule = match self.shape {
            Shape::Quad8 => RuleKind::Gauss(3),
            Shape::Tri6 => RuleKind::Tri(6),
            Shape::Quad4 => RuleKind::Gauss(2),
            _ => RuleKind::Tri(3),
        };
        ips_for(self.shape, StressMode::Axisymmetric, &ctx.x, &rule.points(2), 1.0)
    }

    /// B-matriser (6 × 3n) per integrationspunkt.
    pub fn bmats(&self, x: &[[f64; 3]], ips: &[Ip]) -> Vec<Vec<f64>> {
        let mut b4: Vec<Vec<f64>> = ips.iter().map(|ip| super::super::axisym::bmat_axi(&ip.n, &ip.dndx, radius(ip, x))).collect();
        if self.bbar {
            apply_bbar(StressMode::Axisymmetric, ips, &mut b4);
        }
        let nn = x.len();
        let (n2, n3) = (2 * nn, 3 * nn);
        // CGAX3 (en punkt): vridningens radie är volymens tyngdpunkt
        // ∫r² dA / ∫r dA, inte areans
        let r_twist = (self.shape == Shape::Tri3 && ips.len() == 1).then(|| {
            let r = [x[0][0], x[1][0], x[2][0]];
            let m2 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[0] * r[1] + r[1] * r[2] + r[2] * r[0]) / 6.0;
            m2 / ((r[0] + r[1] + r[2]) / 3.0)
        });
        ips.iter()
            .zip(&b4)
            .map(|(ip, b)| {
                let r = r_twist.unwrap_or_else(|| radius(ip, x));
                let mut out = vec![0.0; 6 * n3];
                for row in 0..4 {
                    for i in 0..nn {
                        for d in 0..2 {
                            out[row * n3 + 3 * i + d] = b[row * n2 + 2 * i + d];
                        }
                    }
                }
                // θ-riktningen är e_r × e_z (högersystemet r, z, θ), dvs. u_θ = −r φ
                for i in 0..nn {
                    out[4 * n3 + 3 * i + 2] = -r * ip.dndx[i][0];
                    out[5 * n3 + 3 * i + 2] = -r * ip.dndx[i][1];
                }
                out
            })
            .collect()
    }
}

impl ElementType for Cgax {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Axisymmetric
    }
    fn dim(&self) -> usize {
        2
    }
    fn n_nodes(&self) -> usize {
        self.shape.n_nodes()
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, UR2]
    }
    fn n_ips(&self) -> usize {
        self.rule.count(2)
    }
    fn vtk_type(&self) -> u8 {
        self.shape.vtk()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.shape.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(self.n_ips(), MODE, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mat = ctx.material(self.name)?;
        let ips = self.ips(ctx)?;
        let bs = self.bmats(&ctx.x, &ips);
        common::stiffness(MODE, &ips, &bs, mat)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        let mat = ctx.material(self.name)?;
        let ips = self.ips(ctx)?;
        let bs = self.bmats(&ctx.x, &ips);
        common::response(ctx, MODE, &ips, &bs, mat, u, du, so, sn, tangent)
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let ips = self.ips_mass(ctx)?;
        let nn = self.n_nodes();
        let nd = 3 * nn;
        let rho = ctx.density;
        let mut m = vec![0.0; nd * nd];
        for ip in &ips {
            let r = radius(ip, &ctx.x);
            for a in 0..nn {
                for b in 0..nn {
                    let v = rho * ip.n[a] * ip.n[b] * ip.w;
                    m[(3 * a) * nd + 3 * b] += v;
                    m[(3 * a + 1) * nd + 3 * b + 1] += v;
                    m[(3 * a + 2) * nd + 3 * b + 2] += v * r * r;
                }
            }
        }
        if kind == MassKind::Lumped {
            let quadratic = matches!(self.shape, Shape::Tri6 | Shape::Quad8);
            for d in 0..3 {
                // första ordningen: radsumma; andra ordningen: diagonalskalning (HRZ)
                let rows: Vec<usize> = (0..nn).map(|a| 3 * a + d).collect();
                let total: f64 = rows.iter().map(|&i| rows.iter().map(|&j| m[i * nd + j]).sum::<f64>()).sum();
                let ds: f64 = rows.iter().map(|&i| m[i * nd + i]).sum();
                let vals: Vec<f64> = rows
                    .iter()
                    .map(|&i| if quadratic { m[i * nd + i] * total / ds } else { rows.iter().map(|&j| m[i * nd + j]).sum() })
                    .collect();
                for (k, &i) in rows.iter().enumerate() {
                    for &j in &rows {
                        m[i * nd + j] = 0.0;
                    }
                    m[i * nd + i] = vals[k];
                }
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let nn = self.n_nodes();
        let spread = |f2: Vec<f64>| -> Vec<f64> {
            let mut f = vec![0.0; 3 * nn];
            for i in 0..nn {
                f[3 * i] = f2[2 * i];
                f[3 * i + 1] = f2[2 * i + 1];
            }
            f
        };
        if let Some(num) = load.kind.strip_prefix('P').and_then(|s| s.parse::<usize>().ok()) {
            let nf = self.shape.faces().len();
            if num == 0 || num > nf {
                return fail(format!("{} har bara ytorna P1–P{nf}, inte {}", self.name, load.kind));
            }
            return Ok(spread(face_pressure(self.shape, StressMode::Axisymmetric, &ctx.x, num - 1, load.magnitude, 1.0)));
        }
        let b = match load.kind.as_str() {
            "BR" => Some([load.magnitude, 0.0, 0.0]),
            "BZ" => Some([0.0, load.magnitude, 0.0]),
            _ => common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)?,
        };
        match b {
            Some(b) => Ok(spread(common::body_force(&self.ips_exact(ctx)?, b, 2))),
            None => not_implemented(&format!("lasttypen {}", load.kind), self.name),
        }
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::volume_weights(&self.ips_exact(ctx)?))
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let mat = ctx.material(self.name)?;
        let ips = self.ips(ctx)?;
        let bs = self.bmats(&ctx.x, &ips);
        Ok(common::output(MODE, &bs, mat, u, state))
    }
}
