//! Second-order shells and continuum shells (area shells2): S8R
//! (`shell2_s8r.rs`), SC8R and SC6R (`shell2_sc.rs`), NLGEOM for them
//! (`shell2_nlgeom.rs`).
//!
//! This file contains what the elements share: the section (homogeneous
//! *SHELL SECTION with isotropic or anisotropic linear elastic material),
//! local axes (default axes or the section's ORIENTATION=), small
//! matrix helpers and output (S, SF, SM).
//!
//! Local axes: the 1-axis is the projection of global X onto the shell's tangent plane
//! (global Z if the normal lies within 0.1° of X), 2 = n × 1. With
//! *SHELL SECTION, ORIENTATION= the 1-axis is the projection of the orientation's
//! 1-axis onto the tangent plane (the orientation's 3-axis if the 1-axis lies within 0.1°
//! of the normal), where the orientation axes are first rotated by the additional
//! rotation on the second data line of *ORIENTATION (see
//! tests/cases/shells2/s8r_platta_orientering).

#[path = "shell2_nlgeom.rs"]
pub mod nlgeom;
#[path = "shell2_s8r.rs"]
pub mod s8r;
#[path = "shell2_sc.rs"]
pub mod sc;

use super::api::{ElemCtx, ElementType, IpField};
use super::common::{cross, dot, norm, scale, sub};
use crate::fem::model::SectionKind;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub static TYPES: &[&dyn ElementType] = &[&s8r::S8R, &sc::SC8R, &sc::SC6R];

/// Element types that support *SHELL SECTION, ORIENTATION= (read by
/// `inp/keywords/section.rs`).
pub const ORIENTATION_TYPES: &[&str] = &["S8R", "SC6R", "SC8R"];

/// The slenderness compensation for the transverse shear stiffness, f = 1/(1 + 0.25·10⁻⁴·a/t²)
/// (a = the area belonging to the integration point or node).
pub fn fp(a: f64, t: f64) -> f64 {
    1.0 / (1.0 + 0.25e-4 * a / (t * t))
}

pub fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Homogeneous shell section for S8R/SC8R/SC6R.
#[derive(Debug, Clone)]
pub struct Sec {
    pub t: f64,
    /// Plane stress D (3×3, row-wise) in the local axes (the orientation).
    pub d: [f64; 9],
    /// Transverse shear stiffness [K11, K22, K12] (force per length, before slenderness compensation).
    pub ks: [f64; 3],
    /// Shear modulus for the thickness modulus (continuum shells: E_t = 2G).
    pub g: f64,
    pub rho: f64,
    /// Number of section points for output (Simpson), 3 for *SHELL GENERAL SECTION.
    pub n_int: usize,
}

impl Sec {
    pub fn get(ctx: &ElemCtx, name: &str) -> Result<Sec> {
        let sec = ctx.section(name)?;
        let (t, n_int) = match &sec.kind {
            SectionKind::Shell { thickness, n_int } => (*thickness, *n_int),
            _ => return fail(format!("element {} ({name}) requires *SHELL SECTION (homogeneous)", ctx.label)),
        };
        if !(t > 0.0) {
            return fail(format!("element {} ({name}): the shell thickness must be positive", ctx.label));
        }
        if let Some(o) = sec.raw.param_upper("OFFSET") {
            if !matches!(o.as_str(), "MIDDLE" | "0" | "0." | "0.0") {
                return fail(format!("element {} ({name}): *SHELL SECTION, OFFSET={o} is not supported (midsurface only)", ctx.label));
            }
        }
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
        // shear moduli: isotropic G, otherwise G13 and G23 from the three-dimensional matrix
        let (g, g13, g23) = match mat.elastic_moduli() {
            Some((e, nu)) => {
                let g = e / (2.0 * (1.0 + nu));
                (g, g, g)
            }
            None => match mat.tangent_linear(StressMode::ThreeD) {
                Ok(c) if c.len() == 36 => (d[8], c[4 * 6 + 4], c[5 * 6 + 5]),
                _ => (d[8], 0.0, 0.0),
            },
        };
        let tss = sec.raw.param(super::beam_timoshenko::section::TSS_PARAM).map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| crate::fem::model::parse_f64(x).unwrap_or(0.0)).collect();
            [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)]
        });
        let ks = match tss {
            Some(k) => k,
            None if g13 > 0.0 && g23 > 0.0 => [5.0 / 6.0 * g13 * t, 5.0 / 6.0 * g23 * t, 0.0],
            None => {
                return fail(format!(
                    "element {} ({name}): the material gives no transverse shear moduli (G13, G23) – specify *TRANSVERSE SHEAR STIFFNESS",
                    ctx.label
                ))
            }
        };
        let n_int = if sec.raw.name == "SHELL GENERAL SECTION" { 3 } else { n_int };
        Ok(Sec { t, d, ks, g, rho: ctx.density, n_int })
    }

    /// The transverse shear matrix [[K11, K12], [K12, K22]] with the compensation for the area `a`.
    pub fn shear_c(&self, a: f64) -> [f64; 4] {
        let f = fp(a, self.t);
        [self.ks[0] * f, self.ks[2] * f, self.ks[2] * f, self.ks[1] * f]
    }

    /// z of the section point (Simpson, n points through the thickness).
    pub fn section_z(&self, sp: usize) -> f64 {
        let n = self.n_int.max(1);
        if n == 1 {
            return 0.0;
        }
        -self.t / 2.0 + self.t * (sp as f64 - 1.0) / (n as f64 - 1.0)
    }

    pub fn dv(&self, e: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i| (0..3).map(|j| self.d[i * 3 + j] * e[j]).sum())
    }
}

/// Default axes for a shell with the normal `n` (rows e1, e2, n).
pub fn default_axes(n: [f64; 3]) -> [[f64; 3]; 3] {
    let n = unit(n);
    let g = if n[0].abs() > (0.1f64.to_radians()).cos() { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let e1 = unit(sub(g, scale(n, dot(g, n))));
    [e1, cross(n, e1), n]
}

/// Local axes (rows e1, e2, n) at the point `p` with the normal `n`, with
/// the section orientation if there is one.
pub fn axes(ctx: &ElemCtx, n: [f64; 3], p: [f64; 3]) -> [[f64; 3]; 3] {
    let Some(o) = ctx.orientation else { return default_axes(n) };
    let n = unit(n);
    // the orientation axes including the additional rotation (in space), then
    // projection onto the tangent plane
    let ax = o.axes_at(p);
    let g = if dot(ax[0], n).abs() > (0.1f64.to_radians()).cos() { ax[2] } else { ax[0] };
    let e1 = unit(sub(g, scale(n, dot(g, n))));
    [e1, cross(n, e1), n]
}

// ---------------------------------------------------------------------------
// Small dense matrix helpers (row-wise)
// ---------------------------------------------------------------------------

/// K += w · aᵀ C b for a, b with `r` rows and `n` columns, C r×r.
pub fn add_btcb(k: &mut [f64], n: usize, a: &[f64], c: &[f64], b: &[f64], r: usize, w: f64) {
    let mut cb = vec![0.0; r * n];
    for i in 0..r {
        for j in 0..r {
            let cij = c[i * r + j];
            if cij != 0.0 {
                for q in 0..n {
                    cb[i * n + q] += cij * b[j * n + q];
                }
            }
        }
    }
    for i in 0..r {
        for p in 0..n {
            let aip = a[i * n + p];
            if aip != 0.0 {
                let f = w * aip;
                for q in 0..n {
                    k[p * n + q] += f * cb[i * n + q];
                }
            }
        }
    }
}

/// K += c · g gᵀ.
pub fn add_outer(k: &mut [f64], n: usize, g: &[f64], c: f64) {
    for p in 0..n {
        if g[p] != 0.0 {
            for q in 0..n {
                k[p * n + q] += c * g[p] * g[q];
            }
        }
    }
}

/// y = B u for B with `r` rows.
pub fn rows_dot(b: &[f64], r: usize, u: &[f64]) -> Vec<f64> {
    let n = u.len();
    (0..r).map(|i| (0..n).map(|q| b[i * n + q] * u[q]).sum()).collect()
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// Generalized strains at an integration point: membrane ε (ε11, ε22, γ12),
/// curvature κ, transverse shear γ (γ13, γ23), thickness stress·t (SF6) and
/// the compensated shear matrix.
pub struct GenStrain {
    pub eps: [f64; 3],
    pub kap: [f64; 3],
    pub gam: [f64; 2],
    pub shear_c: [f64; 4],
    pub sf6: f64,
}

/// S at section points 1 and n, SF (SF1–SF6) and SM (SM1–SM3).
/// `s33` = true gives S with the components S11, S22, S33, S12 (continuum shells).
pub fn output(sec: &Sec, pts: &[GenStrain], s33: bool) -> Vec<IpField> {
    let mut s = if s33 { IpField::new("S", &["S11", "S22", "S33", "S12"]) } else { IpField::new("S", &["S11", "S22", "S12"]) };
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let tb = sec.t.powi(3) / 12.0;
    for (q, g) in pts.iter().enumerate() {
        let ip = q + 1;
        let sn = sec.dv(g.eps);
        let mb = sec.dv(g.kap);
        let c = &g.shear_c;
        let q1 = c[0] * g.gam[0] + c[1] * g.gam[1];
        let q2 = c[2] * g.gam[0] + c[3] * g.gam[1];
        sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, sn[2] * sec.t, q1, q2, g.sf6]));
        sm.points.push((ip, 0, vec![mb[0] * tb, mb[1] * tb, mb[2] * tb]));
        if sec.n_int > 0 {
            let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
            for sp in sps {
                let z = sec.section_z(sp);
                let st = sec.dv([g.eps[0] + z * g.kap[0], g.eps[1] + z * g.kap[1], g.eps[2] + z * g.kap[2]]);
                let v = if s33 { vec![st[0], st[1], 0.0, st[2]] } else { st.to_vec() };
                s.points.push((ip, sp, v));
            }
        }
    }
    let mut out = vec![sf, sm];
    if !s.points.is_empty() {
        out.insert(0, s);
    }
    out
}
