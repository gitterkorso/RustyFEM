//! Euler–Bernoulli-balkar med kubisk interpolation: B23 (plan) och B33 (rumslig).
//!
//! Lokala axlar: t längs balken, n1 tvärsnittets 1-axel och
//! n2 = t × n1. Böjning kring 1-axeln använder I11, kring 2-axeln I22.
//! B23 har n1 = (0, 0, −1). För B33 anges en ungefärlig n1 på tvärsnittets
//! andra datarad (standard (0, 0, −1)); den görs vinkelrät mot balken som
//! n2 = t × n1 / |t × n1|, n1 = n2 × t.
//!
//! Fördelade laster (PX, PY, PZ, GRAV) ger konsistenta nodkrafter och -moment.
//! Snittkrafter som utdata (SF/SM) och massmatris är inte implementerade ännu.

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

/// Lokal styvhet för plan Euler–Bernoulli-balk, ordning [u1,v1,θ1,u2,v2,θ2].
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

/// Lokal styvhet för 3D-balk, ordning per nod [u,v,w,θx,θy,θz].
/// Lokal y = n1 (böjning kring y använder I11), lokal z = n2.
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
    // böjning i x-y-planet (v, θz), kring lokal z: I22
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
    // böjning i x-z-planet (w, θy), kring lokal y: I11
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

/// Beräknade data för ett balkelement.
pub struct BeamData {
    /// Lokal styvhet och transformation global → lokal (n × n).
    pub k_local: Vec<f64>,
    pub t: Vec<f64>,
    pub n: usize,
    pub l: f64,
    /// Lokala axlar [t, n1, n2].
    pub axes: [[f64; 3]; 3],
    pub area: f64,
    pub density: f64,
}

impl EulerBeam {
    pub fn data(&self, ctx: &ElemCtx) -> Result<BeamData> {
        let sec = ctx.section(self.name)?;
        let Some(b) = sec.beam() else {
            return fail(format!("element {} ({}) kräver *BEAM SECTION eller *BEAM GENERAL SECTION", ctx.label, self.name));
        };
        if b.i12 != 0.0 {
            return not_implemented("I12 ≠ 0 (osymmetriskt tvärsnitt)", self.name);
        }
        let (e, g) = match b.moduli {
            Some(m) => m,
            None => {
                let mat = ctx.material(self.name)?;
                if !mat.is_linear() {
                    return not_implemented(&format!("olinjärt material ({})", mat.name()), self.name);
                }
                let Some((e, nu)) = mat.elastic_moduli() else {
                    return fail(format!("{} kräver isotrop linjärelastisk material", self.name));
                };
                (e, e / (2.0 * (1.0 + nu)))
            }
        };
        let d = sub(ctx.x[1], ctx.x[0]);
        let l = norm(d);
        if l == 0.0 {
            return fail(format!("element {} ({}) har ingen längd", ctx.label, self.name));
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
                    "element {} (B33) är parallellt med tvärsnittets n1-riktning – ange n1 på tvärsnittets andra datarad",
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

    /// Snittkrafter i lokala koordinater: k_lokal · T · u.
    /// B23: [N1,V1,M1,N2,V2,M2], B33: [N,Vy,Vz,T,My,Mz] per nod.
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
        // område beams3: initialt krökt när nodnormalerna lutar (beam_eb_curved.rs)
        if let Some(c) = super::beam_eb_curved::Curved::of(self, ctx)? {
            return Ok(c.condensed());
        }
        let d = self.data(ctx)?;
        Ok(congruence(&d.k_local, &d.t, d.n))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            // område nlgeom: korotationell formulering
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
        // område beams2: snittkrafter SF/SM per integrationspunkt
        // (område beams3: krökta element i beam_eb_curved.rs)
        if let Some(c) = super::beam_eb_curved::Curved::of(self, _ctx)? {
            return Ok(super::beam_eb_curved::output(&c, self.dim, _u));
        }
        super::beams2::eb_output::output(self, _ctx, _u)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let d = self.data(ctx)?;
        // last per längdenhet i globala koordinater
        let q = match load.kind.as_str() {
            "PX" => [load.magnitude, 0.0, 0.0],
            "PY" => [0.0, load.magnitude, 0.0],
            "PZ" => [0.0, 0.0, load.magnitude],
            "GRAV" => {
                let b = common::body_load_vector("GRAV", load.magnitude, &load.extra, d.density, self.name)?.unwrap();
                scale(b, d.area)
            }
            _ => return not_implemented(&format!("lasttypen {}", load.kind), self.name),
        };
        if self.dim == 2 && q[2] != 0.0 {
            return fail(format!("{}: last i z-led på en plan balk", self.name));
        }
        // område beams3: krökt element
        if let Some(c) = super::beam_eb_curved::Curved::of(self, ctx)? {
            return Ok(c.load(q));
        }
        let ql = [0, 1, 2].map(|i| common::dot(d.axes[i], q));
        let l = d.l;
        let fl: Vec<f64> = if self.dim == 2 {
            // lokal y är [-s, c]
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
        // global = Tᵀ lokal
        let n = d.n;
        Ok((0..n).map(|j| (0..n).map(|i| d.t[i * n + j] * fl[i]).sum()).collect())
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let d = self.data(ctx)?;
        Ok(vec![d.area * d.l / 2.0; 2])
    }
}
