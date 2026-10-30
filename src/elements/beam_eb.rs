//! Euler–Bernoulli beams with cubic interpolation: B23 (planar) and B33 (spatial).
//!
//! Local axes: t along the beam, n1 the section 1-axis and
//! n2 = t × n1. Bending about the 1-axis uses I11, about the 2-axis I22.
//! B23 has n1 = (0, 0, −1). For B33 an approximate n1 is given on the second data line
//! of the section (default (0, 0, −1)); it is made perpendicular to the beam as
//! n2 = t × n1 / |t × n1|, n1 = n2 × t.
//!
//! Distributed loads (PX, PY, PZ, GRAV) give consistent nodal forces and moments.
//! Section forces as output (SF/SM) and the mass matrix are not implemented yet.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, block_diag, congruence, cross, norm, scale, sub};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};
use crate::section::BeamSection;

pub struct EulerBeam {
    name: &'static str,
    dim: usize,
}

pub static B23: EulerBeam = EulerBeam { name: "B23", dim: 2 };
pub static B33: EulerBeam = EulerBeam { name: "B33", dim: 3 };

pub static TYPES: &[&dyn ElementType] = &[&B23, &B33];

/// Local stiffness for a planar Euler–Bernoulli beam, ordering [u1,v1,θ1,u2,v2,θ2].
fn local_2d(e: f64, a: f64, i: f64, l: f64) -> Vec<f64> {
    let ea = e * a / l;
    let k = e * i / l.powi(3);
    let mut m = vec![0.0; 36];
    let mut set = |r: usize, c: usize, v: f64| {
        m[r * 6 + c] = v;
        m[c * 6 + r] = v;
    };
    set(0, 0, ea);
    set(0, 3, -ea);
    set(3, 3, ea);
    set(1, 1, 12.0 * k);
    set(1, 2, 6.0 * k * l);
    set(1, 4, -12.0 * k);
    set(1, 5, 6.0 * k * l);
    set(2, 2, 4.0 * k * l * l);
    set(2, 4, -6.0 * k * l);
    set(2, 5, 2.0 * k * l * l);
    set(4, 4, 12.0 * k);
    set(4, 5, -6.0 * k * l);
    set(5, 5, 4.0 * k * l * l);
    m
}

/// Local stiffness for a 3D beam, ordering per node [u,v,w,θx,θy,θz].
/// Local y = n1 (bending about y uses I11), local z = n2.
fn local_3d(e: f64, g: f64, s: &BeamSection, l: f64) -> Vec<f64> {
    let mut m = vec![0.0; 144];
    let mut set = |r: usize, c: usize, v: f64| {
        m[r * 12 + c] = v;
        m[c * 12 + r] = v;
    };
    let ea = e * s.a / l;
    let gj = g * s.j / l;
    set(0, 0, ea);
    set(0, 6, -ea);
    set(6, 6, ea);
    set(3, 3, gj);
    set(3, 9, -gj);
    set(9, 9, gj);
    // bending in the x-y plane (v, θz), about local z: I22
    let a = e * s.i22 / l.powi(3);
    set(1, 1, 12.0 * a);
    set(1, 5, 6.0 * a * l);
    set(1, 7, -12.0 * a);
    set(1, 11, 6.0 * a * l);
    set(5, 5, 4.0 * a * l * l);
    set(5, 7, -6.0 * a * l);
    set(5, 11, 2.0 * a * l * l);
    set(7, 7, 12.0 * a);
    set(7, 11, -6.0 * a * l);
    set(11, 11, 4.0 * a * l * l);
    // bending in the x-z plane (w, θy), about local y: I11
    let b = e * s.i11 / l.powi(3);
    set(2, 2, 12.0 * b);
    set(2, 4, -6.0 * b * l);
    set(2, 8, -12.0 * b);
    set(2, 10, -6.0 * b * l);
    set(4, 4, 4.0 * b * l * l);
    set(4, 8, 6.0 * b * l);
    set(4, 10, 2.0 * b * l * l);
    set(8, 8, 12.0 * b);
    set(8, 10, 6.0 * b * l);
    set(10, 10, 4.0 * b * l * l);
    m
}

/// Computed data for a beam element.
pub struct BeamData {
    /// Local stiffness and transformation global → local (n × n).
    pub k_local: Vec<f64>,
    pub t: Vec<f64>,
    pub n: usize,
    pub l: f64,
    /// Local axes [t, n1, n2].
    pub axes: [[f64; 3]; 3],
    pub area: f64,
    pub density: f64,
}

impl EulerBeam {
    pub fn data(&self, ctx: &ElemCtx) -> Result<BeamData> {
        let sec = ctx.section(self.name)?;
        let Some(b) = sec.beam() else {
            return fail(format!("element {} ({}) requires *BEAM SECTION or *BEAM GENERAL SECTION", ctx.label, self.name));
        };
        if b.i12 != 0.0 {
            return not_implemented("I12 ≠ 0 (unsymmetric section)", self.name);
        }
        let (e, g) = match b.moduli {
            Some(m) => m,
            None => {
                let mat = ctx.material(self.name)?;
                if !mat.is_linear() {
                    return not_implemented(&format!("nonlinear material ({})", mat.name()), self.name);
                }
                let Some((e, nu)) = mat.elastic_moduli() else {
                    return fail(format!("{} requires an isotropic linear elastic material", self.name));
                };
                (e, e / (2.0 * (1.0 + nu)))
            }
        };
        let d = sub(ctx.x[1], ctx.x[0]);
        let l = norm(d);
        if l == 0.0 {
            return fail(format!("element {} ({}) has zero length", ctx.label, self.name));
        }
        let t = scale(d, 1.0 / l);
        let density = if b.density > 0.0 { b.density } else { ctx.density };
        if self.dim == 2 {
            let (c, s) = (t[0], t[1]);
            let r = [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]];
            Ok(BeamData {
                k_local: local_2d(e, b.profile.a, b.profile.i11, l),
                t: block_diag(&r, 2),
                n: 6,
                l,
                axes: [t, [0.0, 0.0, -1.0], [-s, c, 0.0]],
                area: b.profile.a,
                density,
            })
        } else {
            let n1a = b.n1.unwrap_or([0.0, 0.0, -1.0]);
            let n2 = cross(t, n1a);
            let ln2 = norm(n2);
            if ln2 < 1e-8 * norm(n1a).max(1e-300) {
                return fail(format!(
                    "element {} (B33) is parallel to the section n1 direction – give n1 on the second data line of the section",
                    ctx.label
                ));
            }
            let n2 = scale(n2, 1.0 / ln2);
            let n1 = cross(n2, t);
            Ok(BeamData {
                k_local: local_3d(e, g, &b.profile, l),
                t: block_diag(&[t, n1, n2], 4),
                n: 12,
                l,
                axes: [t, n1, n2],
                area: b.profile.a,
                density,
            })
        }
    }

    /// Section forces in local coordinates: k_local · T · u.
    /// B23: [N1,V1,M1,N2,V2,M2], B33: [N,Vy,Vz,T,My,Mz] per node.
    pub fn end_forces(&self, ctx: &ElemCtx, u: &[f64]) -> Result<Vec<f64>> {
        let d = self.data(ctx)?;
        let ul = common::matvec(&d.t, u);
        Ok(common::matvec(&d.k_local, &ul))
    }
}

impl ElementType for EulerBeam {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn n_nodes(&self) -> usize {
        2
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        if self.dim == 2 { &[U1, U2, UR3] } else { &[U1, U2, U3, UR1, UR2, UR3] }
    }
    fn vtk_type(&self) -> u8 {
        3
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        // area beams3: initially curved when the nodal normals are inclined (beam_eb_curved.rs)
        if let Some(c) = super::beam_eb_curved::Curved::of(self, ctx)? {
            return Ok(c.condensed());
        }
        let d = self.data(ctx)?;
        Ok(congruence(&d.k_local, &d.t, d.n))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // area nlgeom: corotational formulation
            return crate::kinematics::beam::response(self, ctx, u, _so, _sn, tangent);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        super::beam_mass::mass(self, ctx, kind)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        super::geometric::beam(self, ctx, u)
    }

    fn output(&self, _ctx: &ElemCtx, _u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        // area beams2: section forces SF/SM per integration point
        // (area beams3: curved elements in beam_eb_curved.rs)
        if let Some(c) = super::beam_eb_curved::Curved::of(self, _ctx)? {
            return Ok(super::beam_eb_curved::output(&c, self.dim, _u));
        }
        super::beams2::eb_output::output(self, _ctx, _u)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let d = self.data(ctx)?;
        // load per unit length in global coordinates
        let q = match load.kind.as_str() {
            "PX" => [load.magnitude, 0.0, 0.0],
            "PY" => [0.0, load.magnitude, 0.0],
            "PZ" => [0.0, 0.0, load.magnitude],
            "GRAV" => {
                let b = common::body_load_vector("GRAV", load.magnitude, &load.extra, d.density, self.name)?.unwrap();
                scale(b, d.area)
            }
            _ => return not_implemented(&format!("load type {}", load.kind), self.name),
        };
        if self.dim == 2 && q[2] != 0.0 {
            return fail(format!("{}: load in the z direction on a planar beam", self.name));
        }
        // area beams3: curved element
        if let Some(c) = super::beam_eb_curved::Curved::of(self, ctx)? {
            return Ok(c.load(q));
        }
        let ql = [0, 1, 2].map(|i| common::dot(d.axes[i], q));
        let l = d.l;
        let fl: Vec<f64> = if self.dim == 2 {
            // local y is [-s, c]
            let qy = common::dot(d.axes[2], q);
            vec![ql[0] * l / 2.0, qy * l / 2.0, qy * l * l / 12.0, ql[0] * l / 2.0, qy * l / 2.0, -qy * l * l / 12.0]
        } else {
            let (qx, qy, qz) = (ql[0], ql[1], ql[2]);
            let m = l * l / 12.0;
            vec![
                qx * l / 2.0,
                qy * l / 2.0,
                qz * l / 2.0,
                0.0,
                -qz * m,
                qy * m,
                qx * l / 2.0,
                qy * l / 2.0,
                qz * l / 2.0,
                0.0,
                qz * m,
                -qy * m,
            ]
        };
        // global = Tᵀ local
        let n = d.n;
        Ok((0..n).map(|j| (0..n).map(|i| d.t[i * n + j] * fl[i]).sum()).collect())
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let d = self.data(ctx)?;
        Ok(vec![d.area * d.l / 2.0; 2])
    }
}
