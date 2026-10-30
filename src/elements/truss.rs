//! Trusses: T2D2 and T3D2 (axial force only, linear interpolation).
//!
//! The area is given on the data line of *SOLID SECTION. The stress S11 = N/A and
//! the strain E11 are constant along the element (one integration point).

use super::api::{ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common;
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::materials::{MatInput, StressMode};
use crate::model::Result;

pub struct Truss {
    name: &'static str,
    dim: usize,
}

pub static T2D2: Truss = Truss { name: "T2D2", dim: 2 };
pub static T3D2: Truss = Truss { name: "T3D2", dim: 3 };

pub static TYPES: &[&dyn ElementType] = &[&T2D2, &T3D2];

/// Unit vector along the truss and the length.
pub fn axis(x: &[[f64; 3]]) -> ([f64; 3], f64) {
    let d = common::sub(x[1], x[0]);
    let l = common::norm(d);
    (common::scale(d, 1.0 / l), l)
}

impl Truss {
    fn area(&self, ctx: &ElemCtx) -> Result<f64> {
        Ok(ctx.section(self.name)?.solid_thickness())
    }

    /// Strain ε = Δl / l (small displacements).
    fn strain(&self, x: &[[f64; 3]], u: &[f64]) -> f64 {
        let (t, l) = axis(x);
        let nd = self.dim;
        (0..nd).map(|i| (u[nd + i] - u[i]) * t[i]).sum::<f64>() / l
    }

    /// The B vector (dε/du).
    fn bvec(&self, x: &[[f64; 3]]) -> Vec<f64> {
        let (t, l) = axis(x);
        let nd = self.dim;
        let mut b = vec![0.0; 2 * nd];
        for i in 0..nd {
            b[i] = -t[i] / l;
            b[nd + i] = t[i] / l;
        }
        b
    }
}

impl ElementType for Truss {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Truss
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        2
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.dim == 2 { &[U1, U2] } else { &[U1, U2, U3] }
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        3
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(1, StressMode::Uniaxial, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let e = ctx.material(self.name)?.tangent_linear(StressMode::Uniaxial)?[0];
        let (_, l) = axis(&ctx.x);
        let b = self.bvec(&ctx.x);
        let ea_l = e * self.area(ctx)? * l;
        let n = b.len();
        Ok((0..n * n).map(|k| ea_l * b[k / n] * b[k % n]).collect())
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // area nlgeom: replace the line below with a call to the nonlinear kinematics
            return crate::kinematics::truss::response(ctx, self.name, self.dim, self.area(ctx)?, u, du, so, sn, tangent);
        }
        let mat = ctx.material(self.name)?;
        let stride = common::ip_stride(StressMode::Uniaxial, mat);
        let old = if so.len() == stride { so.to_vec() } else { common::init_state(1, StressMode::Uniaxial, mat) };
        let eps = [self.strain(&ctx.x, u)];
        let deps = [self.strain(&ctx.x, du)];
        let mut mnew = old[1..].to_vec();
        let (temperature, dtemp) = ctx.temperature_at(&[0.5, 0.5]);
        let inp = MatInput {
            mode: StressMode::Uniaxial,
            strain: &eps,
            dstrain: &deps,
            stress_old: &old[..1],
            time: ctx.time,
            dt: ctx.dt,
            temperature,
            dtemp,
            defgrad: None,
        };
        let out = mat.update(&inp, &old[1..], &mut mnew)?;
        let (_, l) = axis(&ctx.x);
        let vol = self.area(ctx)? * l;
        let b = self.bvec(&ctx.x);
        let n = b.len();
        let fint = b.iter().map(|bi| bi * out.stress[0] * vol).collect();
        let k = tangent.then(|| (0..n * n).map(|k| out.tangent[0] * vol * b[k / n] * b[k % n]).collect());
        sn.clear();
        sn.push(out.stress[0]);
        sn.extend(mnew);
        Ok(ElemResponse { fint, k })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (_, l) = axis(&ctx.x);
        let m = ctx.density * self.area(ctx)? * l;
        let nd = self.dim;
        let n = 2 * nd;
        let mut out = vec![0.0; n * n];
        for a in 0..2 {
            for b in 0..2 {
                let v = match kind {
                    MassKind::Lumped => if a == b { m / 2.0 } else { 0.0 },
                    MassKind::Consistent => if a == b { m / 3.0 } else { m / 6.0 },
                };
                for d in 0..nd {
                    out[(a * nd + d) * n + b * nd + d] = v;
                }
            }
        }
        Ok(out)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let w = self.volume_weights(ctx)?;
        if let Some(b) = common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)? {
            let nd = self.dim;
            return Ok((0..2 * nd).map(|k| b[k % nd] * w[k / nd]).collect());
        }
        super::api::not_implemented(&format!("load type {}", load.kind), self.name)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        Ok(super::geometric::truss(&ctx.x, self.dim, self.area(ctx)?, state))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (_, l) = axis(&ctx.x);
        Ok(vec![self.area(ctx)? * l / 2.0; 2])
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let mut s = IpField::new("S", &["S11"]);
        let mut e = IpField::new("E", &["E11"]);
        s.points.push((1, 0, vec![state.first().copied().unwrap_or(0.0)]));
        e.points.push((1, 0, vec![self.strain(&ctx.x, u)]));
        let mut out = vec![s, e];
        // the material's own fields (PEEQ, PE, ...)
        if let Some(mat) = ctx.material {
            let stride = common::ip_stride(StressMode::Uniaxial, mat);
            if state.len() == stride {
                let vals = mat.output_field_values(StressMode::Uniaxial, &state[1..]);
                let mut k = 0;
                for (name, labels) in mat.output_fields(StressMode::Uniaxial) {
                    let mut f = IpField::new(name, &labels);
                    f.points.push((1, 0, (0..labels.len()).map(|i| vals.get(k + i).copied().unwrap_or(0.0)).collect()));
                    k += labels.len();
                    out.push(f);
                }
            }
        }
        Ok(out)
    }
}
