//! Generaliserad plan töjning: CPEG3, CPEG4, CPEG4R, CPEG4I, CPEG6, CPEG8, CPEG8R.
//!
//! Modellen ligger mellan två begränsningsplan (avstånd t = tvärsnittets
//! tjocklek) som kan röra sig som stela kroppar relativt varandra:
//! förlängningen Δl och rotationerna Δφx, Δφy är frihetsgraderna 3, 4 och 5
//! i en gemensam referensnod (*SOLID SECTION, REF NODE=n). Referensnodens läge
//! (x₀, y₀) är punkten som planen roterar kring.
//!
//! Töjningen ut ur planet är ε33 = (Δl + Δφx (y − y₀) − Δφy (x − x₀)) / t i
//! varje integrationspunkt. CPEG4 (B-bar) använder volymmedelvärdet, dvs.
//! värdet i areatyngdpunkten, så att ε33 är konstant i elementet
//! (ett ensamt CPEG4 ger singulära rotationsfrihetsgrader och samma E33 i alla
//! punkter, och reaktionsmomenten blir RM1 = F (ȳ − y₀), RM2 = −F (x̄ − x₀)).
//! I planet är elementen som motsvarande CPE-element i `solid_higher.rs`
//! (B-bar av ε11+ε22 för CPEG4, hourglass-styvhet för CPEG4R). CPEG4I har,
//! till skillnad från CPE4I, även volymmoden ξη i ε11+ε22 (som CAX4I).
//! Referensnodens massa är planets stela rörelse ut ur
//! planet (se `mass`).
//!
//! Formuleringen är linjär (se även tests/cases/solids/cpeg6_skiva.inp).
//!
//! Elementraden i .inp har bara elementets egna noder; referensnoden läggs
//! sist i nodlistan av `FeModel::finalize` (via `section_nodes`).

#[path = "solid_higher_gps_nlgeom.rs"]
mod nlgeom;

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip};
use super::solid_higher::shapes::Shape;
use super::solid_higher::{tech, RuleKind, Solid, Tech};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2};
use crate::fem::model::{DistLoad, SectionDef};
use crate::materials::StressMode;
use crate::model::{fail, Result};

/// Element för generaliserad plan töjning: ett plant töjningselement i
/// motorn plus referensnodens tre frihetsgrader.
pub struct Gps {
    pub base: Solid,
}

const fn g(name: &'static str, shape: Shape, rule: RuleKind, tech: Tech) -> Gps {
    Gps { base: Solid { name, shape, mode: StressMode::PlaneStrain, rule, tech } }
}

pub static CPEG3: Gps = g("CPEG3", Shape::Tri3, RuleKind::Tri(1), Tech::Full);
pub static CPEG4: Gps = g("CPEG4", Shape::Quad4, RuleKind::Gauss(2), Tech::Bbar);
pub static CPEG4R: Gps = g("CPEG4R", Shape::Quad4, RuleKind::Gauss(1), Tech::Hourglass);
pub static CPEG4I: Gps = g("CPEG4I", Shape::Quad4, RuleKind::Gauss(2), Tech::Incompatible);
pub static CPEG6: Gps = g("CPEG6", Shape::Tri6, RuleKind::Tri(3), Tech::Full);
pub static CPEG8: Gps = g("CPEG8", Shape::Quad8, RuleKind::Gauss(3), Tech::Full);
pub static CPEG8R: Gps = g("CPEG8R", Shape::Quad8, RuleKind::Gauss(2), Tech::Full);

pub static TYPES: &[&dyn ElementType] = &[&CPEG3, &CPEG4, &CPEG4I, &CPEG4R, &CPEG6, &CPEG8, &CPEG8R];

const REF_DOFS: &[Dof] = &[U3, UR1, UR2];

impl Gps {
    /// Antal noder i planet (utan referensnoden).
    fn nn(&self) -> usize {
        self.base.shape.n_nodes()
    }

    /// Antal frihetsgrader i planet.
    fn nin(&self) -> usize {
        2 * self.nn()
    }

    /// Kontext med bara elementets egna noder (för motorns funktioner).
    fn sub<'a>(&self, ctx: &ElemCtx<'a>) -> Result<ElemCtx<'a>> {
        if ctx.x.len() != self.nn() + 1 {
            return fail(format!(
                "element {} ({}) saknar referensnod (*SOLID SECTION, REF NODE=)",
                ctx.label, self.base.name
            ));
        }
        let mut c = ctx.clone();
        c.x.truncate(self.nn());
        Ok(c)
    }

    /// ∂ε33/∂(Δl, Δφx, Δφy) i punkterna `ips`: [1, y − y₀, −(x − x₀)] / t med
    /// punktens läge. Med B-bar (CPEG4) ersätts raden med sitt volymmedelvärde,
    /// dvs. värdet i areatyngdpunkten (ε33 blir konstant i elementet).
    fn eps33_rows(&self, ctx: &ElemCtx, sub: &ElemCtx, ips: &[Ip]) -> Result<Vec<[f64; 3]>> {
        let t = self.base.thickness(sub)?;
        let x0 = ctx.x[self.nn()];
        let row = |ip: &Ip| -> [f64; 3] {
            let p = |k: usize| ip.n.iter().zip(&sub.x).map(|(ni, xi)| ni * xi[k]).sum::<f64>();
            [1.0 / t, (p(1) - x0[1]) / t, -(p(0) - x0[0]) / t]
        };
        let mut rows: Vec<[f64; 3]> = ips.iter().map(row).collect();
        if self.base.tech == Tech::Bbar {
            let a: f64 = ips.iter().map(|ip| ip.w).sum();
            let m = [0, 1, 2].map(|k| ips.iter().zip(&rows).map(|(ip, r)| ip.w * r[k]).sum::<f64>() / a);
            rows.iter_mut().for_each(|r| *r = m);
        }
        Ok(rows)
    }

    /// B-matriser med referensnodens kolumner (inte inkompatibla moder).
    fn bmats(&self, ctx: &ElemCtx, sub: &ElemCtx) -> Result<(Vec<Ip>, Vec<Vec<f64>>)> {
        let (ips, mut bs) = match self.base.tech {
            Tech::Hourglass => tech::uniform_strain(&self.base, sub)?,
            _ => {
                let ips = self.base.ips(sub)?;
                let bs = self.base.bmats(&sub.x, &ips);
                (ips, bs)
            }
        };
        let rows = self.eps33_rows(ctx, sub, &ips)?;
        tech::append_eps33(&mut bs, 4, &rows);
        Ok((ips, bs))
    }

    /// Lägger en matris för frihetsgraderna i planet i elementets matris
    /// (referensnodens frihetsgrader sist).
    fn embed(&self, k: &[f64]) -> Vec<f64> {
        let m = self.nin();
        let n = m + 3;
        let mut out = vec![0.0; n * n];
        for i in 0..m {
            out[i * n..i * n + m].copy_from_slice(&k[i * m..(i + 1) * m]);
        }
        out
    }

    fn pad(&self, mut f: Vec<f64>) -> Vec<f64> {
        f.extend([0.0; 3]);
        f
    }
}

impl ElementType for Gps {
    fn name(&self) -> &'static str {
        self.base.name
    }
    fn family(&self) -> Family {
        Family::Continuum2D
    }
    fn dim(&self) -> usize {
        2
    }
    fn n_nodes(&self) -> usize {
        self.nn() + 1
    }
    fn n_connectivity(&self) -> usize {
        self.nn()
    }
    fn section_nodes(&self, section: &SectionDef) -> Result<Vec<i64>> {
        match section.raw.param("REF NODE") {
            Some(v) => match v.trim().parse::<i64>() {
                Ok(l) => Ok(vec![l]),
                Err(_) => fail(format!("REF NODE={v}: ange referensnoden med nodnummer (nodset stöds inte ännu)")),
            },
            None => fail(format!("{} kräver REF NODE= (referensnod för generaliserad plan töjning)", self.base.name)),
        }
    }
    fn node_dofs(&self, k: usize) -> &'static [Dof] {
        if k < self.nn() { &[U1, U2] } else { REF_DOFS }
    }
    fn n_ips(&self) -> usize {
        self.base.n_ips()
    }
    fn vtk_type(&self) -> u8 {
        self.base.vtk_type()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.base.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.base.init_state(&self.sub(ctx)?)
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let sub = self.sub(ctx)?;
        let mat = ctx.material(self.base.name)?;
        if self.base.tech == Tech::Incompatible {
            let rows = |ips: &[Ip]| self.eps33_rows(ctx, &sub, ips).unwrap_or_default();
            return tech::incompatible_stiffness_x(&self.base, &sub, mat, Some(&rows));
        }
        let (ips, bs) = self.bmats(ctx, &sub)?;
        let mut k = common::stiffness(StressMode::PlaneStrain, &ips, &bs, mat)?;
        if self.base.tech == Tech::Hourglass {
            let kh = self.embed(&tech::hourglass_matrix(&self.base, &sub, mat)?);
            k.iter_mut().zip(&kh).for_each(|(a, b)| *a += b);
        }
        Ok(k)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let sub = self.sub(ctx)?;
        let mat = ctx.material(self.base.name)?;
        if ctx.nlgeom {
            // område accuracy2: NLGEOM i solid_higher_gps_nlgeom.rs
            return nlgeom::response(self, ctx, &sub, mat, u, du, so, sn, tangent);
        }
        if self.base.tech == Tech::Incompatible {
            let rows = |ips: &[Ip]| self.eps33_rows(ctx, &sub, ips).unwrap_or_default();
            return tech::incompatible_response_x(&self.base, &sub, mat, u, du, so, sn, tangent, Some(&rows));
        }
        let (ips, bs) = self.bmats(ctx, &sub)?;
        let mut r = common::response_with_temperatures(&sub, StressMode::PlaneStrain, &ips, &bs, mat, u, du, so, sn, tangent, Some(&self.base.temperatures(&sub)))?;
        if self.base.tech == Tech::Hourglass {
            let kh = self.embed(&tech::hourglass_matrix(&self.base, &sub, mat)?);
            let fh = common::matvec(&kh, u);
            r.fint.iter_mut().zip(&fh).for_each(|(a, b)| *a += b);
            if let Some(k) = r.k.as_mut() {
                k.iter_mut().zip(&kh).for_each(|(a, b)| *a += b);
            }
        }
        Ok(r)
    }

    /// Massa: elementets massa i planet plus referensnodens block
    /// ∫ρ g gᵀ dV med g = [1, y − y₀, −(x − x₀)] (begränsningsplanets stela
    /// rörelse ut ur planet), samma för konsistent och diagonal massa och utan
    /// koppling till frihetsgraderna i planet.
    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let sub = self.sub(ctx)?;
        let mut m = self.embed(&self.base.mass(&sub, kind)?);
        let t = self.base.thickness(&sub)?;
        let rule = tech::mass_rule(&self.base).points(2);
        let ips = super::solid_higher::ips_for(self.base.shape, StressMode::PlaneStrain, &sub.x, &rule, t)?;
        let x0 = ctx.x[self.nn()];
        let n = self.nin() + 3;
        let r = self.nin();
        for ip in &ips {
            let p = |k: usize| ip.n.iter().zip(&sub.x).map(|(ni, xi)| ni * xi[k]).sum::<f64>();
            let g = [1.0, p(1) - x0[1], -(p(0) - x0[0])];
            for a in 0..3 {
                for b in 0..3 {
                    m[(r + a) * n + r + b] += ctx.density * g[a] * g[b] * ip.w;
                }
            }
        }
        Ok(m)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let m = self.nin().min(u.len());
        Ok(self.pad(self.base.distributed_load(&self.sub(ctx)?, load, &u[..m])?))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        let m = self.nin().min(u.len());
        Ok(self.embed(&self.base.geometric_stiffness(&self.sub(ctx)?, &u[..m], state)?))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut w = self.base.volume_weights(&self.sub(ctx)?)?;
        w.push(0.0);
        Ok(w)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let sub = self.sub(ctx)?;
        let mat = ctx.material(self.base.name)?;
        if self.base.tech == Tech::Incompatible {
            let rows = |ips: &[Ip]| self.eps33_rows(ctx, &sub, ips).unwrap_or_default();
            return tech::incompatible_output_x(&self.base, &sub, mat, u, state, Some(&rows));
        }
        let (_, bs) = self.bmats(ctx, &sub)?;
        let mut out = common::output(StressMode::PlaneStrain, &bs, mat, u, state);
        if ctx.nlgeom && crate::kinematics::ext(state).is_none() {
            nlgeom::rename_strain(&mut out);
        }
        Ok(out)
    }
}

/// Alla element för generaliserad plan töjning (för tester).
pub fn all() -> Vec<&'static Gps> {
    vec![&CPEG3, &CPEG4, &CPEG4R, &CPEG4I, &CPEG6, &CPEG8, &CPEG8R]
}
