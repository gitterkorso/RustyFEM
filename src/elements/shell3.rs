//! Thin shells and axisymmetric shells (area shells3):
//!
//! - `shell3_axi.rs`: axisymmetric shells SAX1, SAX2 and membranes MAX1, MAX2,
//! - `shell3_tri.rs`: STRI3 (flat facet, discrete Kirchhoff),
//! - `shell3_q4.rs`: S4R5 (thin four-node shell, five degrees of freedom per node),
//! - `shell3_q9.rs`: S9R5 and S8R5 (thin doubly curved shells, five
//!   degrees of freedom per node; S8R5 has an internal midside node).
//!
//! The formulations are described in each file; see also `tests/shells3.rs`.
//!
//! The five-degree-of-freedom shells (S4R5, S8R5, S9R5) have two rotations per
//! node about the axes t1, t2 perpendicular to the nodal normal (t1 = the projection of
//! the global axis that lies farthest from the normal, t2 = n × t1). Here
//! they are assembled with six global degrees of freedom per node: the element sees only
//! the components of the rotation in the node's tangent plane, and the rotation about
//! the nodal normal gets a very small stiffness (`DRILL_EPS`) so that the system does not
//! become singular. Displacements and rotations in the tangent plane are therefore
//! affected only at machine precision level when the nodal normals of adjacent elements
//! coincide (flat meshes).

#[path = "shell3_axi.rs"]
pub mod axi;
#[path = "shell3_q4.rs"]
pub mod q4;
#[path = "shell3_q9.rs"]
pub mod q9;
#[path = "shell3_tri.rs"]
pub mod tri;

use super::api::{ElemCtx, ElementType, IpField};
use super::common::{cross, dot, norm, scale, sub};
use super::shell2::Sec;
use crate::fem::model::SectionKind;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub static TYPES: &[&dyn ElementType] = &[&axi::SAX1, &axi::SAX2, &axi::MAX1, &axi::MAX2, &tri::STRI3, &q4::S4R5, &q9::S9R5, &q9::S8R5];

/// Relative stiffness for the rotation about the nodal normal in the five-degree-of-freedom shells
/// (fraction of the element's largest rotational diagonal entry).
pub const DRILL_EPS: f64 = 1e-9;

pub fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Shell section for the elements of this area: *SHELL SECTION (via `shell2::Sec`)
/// or *MEMBRANE SECTION (membrane stiffness only).
pub fn section(ctx: &ElemCtx, name: &str, membrane: bool) -> Result<Sec> {
    if !membrane {
        return Sec::get(ctx, name);
    }
    let sec = ctx.section(name)?;
    let t = match &sec.kind {
        SectionKind::Shell { thickness, .. } => *thickness,
        _ if sec.raw.name == "MEMBRANE SECTION" => {
            let v = sec.raw.first_floats()?;
            v.first().copied().filter(|&t| t != 0.0).unwrap_or(1.0)
        }
        _ => return fail(format!("element {} ({name}) requires *MEMBRANE SECTION", ctx.label)),
    };
    let mat = ctx.material(name)?;
    if !mat.is_linear() {
        return super::api::not_implemented(&format!("nonlinear material ({})", mat.name()), name);
    }
    let dv = mat.tangent_linear(StressMode::PlaneStress)?;
    if dv.len() != 9 {
        return fail(format!("{name}: the material gives no plane stress matrix"));
    }
    let mut d = [0.0; 9];
    d.copy_from_slice(&dv);
    Ok(Sec { t, d, ks: [0.0; 3], g: d[8], rho: ctx.density, n_int: 1 })
}

/// The shear modulus from the section's plane stress matrix (isotropic material: G).
pub fn shear_modulus(sec: &Sec) -> f64 {
    sec.d[8]
}

/// The transverse shear stiffness (K11) before slenderness compensation.
pub fn k_shear(sec: &Sec) -> f64 {
    sec.ks[0]
}

/// The slenderness compensation 1/(1 + 0.25·10⁻⁴·a/t²).
pub fn fp(a: f64, t: f64) -> f64 {
    super::shell2::fp(a, t)
}

/// Nodal basis for the five-degree-of-freedom shells: t1 = the projection of the
/// global axis with the smallest component along the normal (on ties the one with the lowest
/// number), t2 = n × t1.
pub fn axes5(n: [f64; 3]) -> [[f64; 3]; 3] {
    let n = unit(n);
    let mut k = 0;
    for c in 1..3 {
        if n[c].abs() < n[k].abs() {
            k = c;
        }
    }
    let mut g = [0.0; 3];
    g[k] = 1.0;
    let t1 = unit(sub(g, scale(n, dot(g, n))));
    [t1, cross(n, t1), n]
}

/// Default shell axes (1 = projection of global X, or Z if
/// the normal lies within 0.1° of X).
pub fn default_axes(n: [f64; 3]) -> [[f64; 3]; 3] {
    super::shell2::default_axes(n)
}

/// Adds the small rotational stiffness about the nodal normals `nrm` to a
/// matrix with six degrees of freedom per node (rotations at positions 3..6).
pub fn add_drill_eps(k: &mut [f64], nrm: &[[f64; 3]]) {
    let n = 6 * nrm.len();
    let mut mx = 0.0f64;
    for i in 0..nrm.len() {
        for d in 3..6 {
            mx = mx.max(k[(6 * i + d) * n + 6 * i + d].abs());
        }
    }
    let kd = DRILL_EPS * mx.max(f64::MIN_POSITIVE);
    for (i, ni) in nrm.iter().enumerate() {
        for a in 0..3 {
            for b in 0..3 {
                k[(6 * i + 3 + a) * n + 6 * i + 3 + b] += kd * ni[a] * ni[b];
            }
        }
    }
}

/// Projects a matrix with six degrees of freedom per node so that the rotations
/// act only through their components in the nodes' tangent planes: K ← Pᵀ K P with
/// P = I − n nᵀ for the rotation blocks.
pub fn project_tangent(k: &[f64], nrm: &[[f64; 3]]) -> Vec<f64> {
    let nn = nrm.len();
    let n = 6 * nn;
    let mut p = vec![0.0; n * n];
    for i in 0..nn {
        for d in 0..3 {
            p[(6 * i + d) * n + 6 * i + d] = 1.0;
        }
        for a in 0..3 {
            for b in 0..3 {
                let v = if a == b { 1.0 } else { 0.0 } - nrm[i][a] * nrm[i][b];
                p[(6 * i + 3 + a) * n + 6 * i + 3 + b] = v;
            }
        }
    }
    super::common::congruence(k, &p, n)
}

/// Generalized strains at an integration point.
#[derive(Debug, Clone, Default)]
pub struct Gen {
    /// Membrane (ε11, ε22, γ12).
    pub eps: [f64; 3],
    /// Curvature (κ11, κ22, 2κ12).
    pub kap: [f64; 3],
    /// Transverse shear forces (SF4, SF5).
    pub q: [f64; 2],
}

/// S at section points 1 and n, SF (SF1–SF6) and SM (SM1–SM3).
pub fn output(sec: &Sec, pts: &[Gen]) -> Vec<IpField> {
    let mut s = IpField::new("S", &["S11", "S22", "S12"]);
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let tb = sec.t.powi(3) / 12.0;
    for (q, g) in pts.iter().enumerate() {
        let ip = q + 1;
        let sn = sec.dv(g.eps);
        let mb = sec.dv(g.kap);
        sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, sn[2] * sec.t, g.q[0], g.q[1], 0.0]));
        sm.points.push((ip, 0, vec![mb[0] * tb, mb[1] * tb, mb[2] * tb]));
        let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
        for sp in sps {
            let z = sec.section_z(sp);
            let st = sec.dv([g.eps[0] + z * g.kap[0], g.eps[1] + z * g.kap[1], g.eps[2] + z * g.kap[2]]);
            s.points.push((ip, sp, st.to_vec()));
        }
    }
    vec![s, sf, sm]
}

/// y = B u for a row b.
pub fn row_dot(b: &[f64], u: &[f64]) -> f64 {
    b.iter().zip(u).map(|(a, c)| a * c).sum()
}
