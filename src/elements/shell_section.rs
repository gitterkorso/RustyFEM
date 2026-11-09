//! Shells with integration through the thickness (area struct_nonlin): nonlinear
//! materials (plasticity), temperature, *ORIENTATION and layered sections
//! (*SHELL SECTION, COMPOSITE) for S4R, S4, S3 and S3R.
//!
//! The section is integrated at section points through the thickness with Simpson's
//! rule: a homogeneous section has n points (default 5) over
//! the full thickness, a layered one n points (default 3) in each layer.
//! The section points are numbered from the bottom (SNEG) upwards, layer by layer.
//! At each point the material is updated in plane stress with the strain
//! ε(z) = ε₀ + z·κ (rotated to the material axes) and the temperature
//! T(z) = T + z·∂T/∂n. The section forces and the section tangent become
//!
//!   N = Σ wₖ σₖ,  M = Σ wₖ zₖ σₖ,  C = Σ wₖ [Cₖ, zₖCₖ; zₖCₖ, zₖ²Cₖ].
//!
//! Transverse shear, hourglass stabilization and drill are elastic with the
//! original section stiffness.
//!
//! Material axes: the element's local axes (see `shell.rs`), rotated to
//! the 1-axis of *ORIENTATION projected onto the shell plane (the section's
//! ORIENTATION=), plus the layer angle for layered sections. S and
//! the material's scalar output (PEEQ) are reported at section point 1 and the
//! last one, S in the material axes; SF and SM in the orientation axes.
//!
//! With NLGEOM the corotational shell formulation is used (`shell_nlgeom.rs`):
//! the membrane strain is logarithmic from the nodal positions, the curvature linear in the
//! deformational rotations, and the elastic parts (shear, hourglass,
//! drill) as energy in the deformational displacements.

use super::nlgeom::{self, MemRule};
use super::{local_frame, Local, ShellSec};
use crate::elements::api::{ElemCtx, ElemResponse, IpField};
use crate::elements::common;
use crate::fem::model::parse_f64;
use crate::kinematics::hyperdual::Hd;
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

/// Section point: position z from the midsurface, weight (Simpson, with thickness) and
/// the angle from the element's local 1-axis to the material 1-axis (cos, sin).
#[derive(Clone, Copy, Debug)]
pub struct SecPoint {
    pub z: f64,
    pub w: f64,
    pub c: f64,
    pub s: f64,
}

/// Generalized strains per integration point (stored for the strain increment).
pub const NGEN: usize = 6;

/// Section with section points.
pub struct Layup<'a> {
    pub t: f64,
    pub pts: Vec<SecPoint>,
    /// The layers (z0, z1, cos, sin) for layered sections.
    plies: Vec<(f64, f64, f64, f64)>,
    pub mat: &'a dyn MaterialModel,
    nst: usize,
    /// Angle from the element's 1-axis to the orientation 1-axis (output for SF, SM).
    pub out: (f64, f64),
}

/// Simpson points and weights for a layer [z0, z0 + h] with n points.
fn simpson(z0: f64, h: f64, n: usize) -> Result<Vec<(f64, f64)>> {
    if n == 1 {
        return Ok(vec![(z0 + 0.5 * h, h)]);
    }
    if n % 2 == 0 {
        return fail(format!("the number of section points ({n}) must be odd (Simpson's rule)"));
    }
    let dz = h / (n - 1) as f64;
    Ok((0..n)
        .map(|k| {
            let f = if k == 0 || k == n - 1 { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
            (z0 + dz * k as f64, f * dz / 3.0)
        })
        .collect())
}

/// The angle (cos, sin) from the element's local 1-axis to the material 1-axis
/// for the section's *ORIENTATION. The axis k about which the additional rotation
/// is made (second data line, default 1) is the shell normal direction, and
/// the in-plane material axes are the next two (k = 1: 2 and 3, k = 2: 3 and 1,
/// k = 3: 1 and 2). The material 1-axis is the first of them projected onto
/// the shell plane.
fn orientation_angle(ctx: &ElemCtx, loc: &Local) -> Result<(f64, f64)> {
    let Some(o) = ctx.orientation else { return Ok((1.0, 0.0)) };
    let ax = o.axes_at(common::centroid(&ctx.x));
    let k = o.extra.map_or(1, |e| e.0);
    let (a1, a2) = (ax[k % 3], ax[(k + 1) % 3]);
    let n = loc.r[2];
    let proj = |v: [f64; 3]| -> [f64; 3] { common::sub(v, common::scale(n, common::dot(v, n))) };
    let mut p = proj(a1);
    if common::norm(p) < 1e-3 {
        // the 1-axis is (almost) perpendicular to the shell: the 1-axis becomes the 2-axis × n
        let q = proj(a2);
        p = common::cross(q, n);
    }
    let (a, b) = (common::dot(p, loc.r[0]), common::dot(p, loc.r[1]));
    let l = (a * a + b * b).sqrt();
    if !(l > 0.0) {
        return fail(format!("element {}: the orientation gives no direction in the shell plane", ctx.label));
    }
    Ok((a / l, b / l))
}

/// Thickness and layers (thickness, points, angle in degrees) for *SHELL SECTION, COMPOSITE.
pub fn composite_plies(sec: &crate::fem::model::SectionDef) -> Result<Vec<(f64, usize, f64)>> {
    let mut plies = Vec::new();
    for d in sec.raw.records() {
        if d.fields.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        let num = |i: usize, def: f64| -> Result<f64> {
            match d.fields.get(i).map(|f| f.trim()).filter(|f| !f.is_empty()) {
                Some(f) => parse_f64(f).ok_or_else(|| crate::model::FemError(format!("{} line {}: '{f}' is not a number", sec.raw.file, d.line))),
                None => Ok(def),
            }
        };
        let t = num(0, 0.0)?;
        let n = num(1, 3.0)?.round().max(1.0) as usize;
        let ang = match d.fields.get(3).map(|f| f.trim()).filter(|f| !f.is_empty()) {
            None => 0.0,
            Some(f) => match parse_f64(f) {
                Some(a) => a,
                None => return sec.raw.err_at(d.line, format!("the layer orientation '{f}' is not supported yet (give the angle in degrees)")),
            },
        };
        if !(t > 0.0) {
            return sec.raw.err_at(d.line, "the layer thickness must be positive");
        }
        plies.push((t, n, ang));
    }
    if plies.is_empty() {
        return sec.raw.err("*SHELL SECTION, COMPOSITE has no layers");
    }
    Ok(plies)
}

/// Elements that support *SHELL SECTION, OFFSET=.
pub const OFFSET_TYPES: &[&str] = &["S4", "S4R", "S3", "S3R"];

/// *SHELL SECTION, OFFSET= as a fraction of the thickness: the distance from the
/// midsurface to the reference surface (the nodes) along the positive normal.
/// SPOS = 0.5 (the nodes on the top face), SNEG = −0.5, a number, or 0 without OFFSET.
pub fn offset(sec: &crate::fem::model::SectionDef) -> Result<f64> {
    if sec.raw.name != "SHELL SECTION" {
        return Ok(0.0);
    }
    let Some(v) = sec.raw.param("OFFSET") else { return Ok(0.0) };
    match v.trim().to_uppercase().as_str() {
        "SPOS" => Ok(0.5),
        "SNEG" => Ok(-0.5),
        "" => sec.raw.err("OFFSET= needs a value (SPOS, SNEG or a fraction of the thickness)"),
        o => match parse_f64(o) {
            Some(x) => Ok(x),
            None => sec.raw.err(format!("OFFSET={o} is not supported (give SPOS, SNEG or a fraction of the thickness)")),
        },
    }
}

/// The section is integrated through the thickness (nonlinear material, orientation,
/// layered section or offset). Otherwise the homogeneous elastic formulation is used.
pub fn is_general(ctx: &ElemCtx) -> bool {
    let Some(sec) = ctx.section else { return false };
    if sec.raw.name != "SHELL SECTION" {
        return false;
    }
    if sec.raw.has("COMPOSITE") || ctx.orientation.is_some() || offset(sec).map_or(true, |o| o != 0.0) {
        return true;
    }
    match ctx.material {
        Some(m) => !m.is_linear() || m.elastic_moduli().is_none(),
        None => false,
    }
}

impl<'a> Layup<'a> {
    pub fn new(ctx: &ElemCtx<'a>, name: &str, loc: &Local) -> Result<Layup<'a>> {
        let sec = ctx.section(name)?;
        let mat = ctx.material(name)?;
        if sec.raw.has("TEMPERATURE") {
            return sec.raw.err("TEMPERATURE= on *SHELL SECTION is not supported yet");
        }
        let out = orientation_angle(ctx, loc)?;
        let mut pts = Vec::new();
        let mut layers = Vec::new();
        let t;
        if sec.raw.has("COMPOSITE") {
            let plies = composite_plies(sec)?;
            t = plies.iter().map(|p| p.0).sum::<f64>();
            let mut z0 = -0.5 * t;
            for (h, n, ang) in plies {
                let (ca, sa) = (ang.to_radians().cos(), ang.to_radians().sin());
                // angle = the orientation's + the layer's
                let (c, s) = (out.0 * ca - out.1 * sa, out.1 * ca + out.0 * sa);
                for (z, w) in simpson(z0, h, n)? {
                    pts.push(SecPoint { z, w, c, s });
                }
                layers.push((z0, z0 + h, c, s));
                z0 += h;
            }
        } else {
            let crate::fem::model::SectionKind::Shell { thickness, n_int } = sec.kind else {
                return fail(format!("element {} ({name}) requires *SHELL SECTION", ctx.label));
            };
            t = thickness;
            for (z, w) in simpson(-0.5 * t, t, n_int.max(1))? {
                pts.push(SecPoint { z, w, c: out.0, s: out.1 });
            }
        }
        // OFFSET=: the nodes lie on the reference surface at OFFSET·t from the midsurface,
        // so the section points (z from the reference surface) move by −OFFSET·t
        let dz = -offset(sec)? * t;
        if dz != 0.0 {
            for p in pts.iter_mut() {
                p.z += dz;
            }
            for l in layers.iter_mut() {
                l.0 += dz;
                l.1 += dz;
            }
        }
        Ok(Layup { t, pts, plies: layers, mat, nst: mat.n_state(), out })
    }

    fn stride(&self) -> usize {
        3 + self.nst
    }

    /// The state length per integration point.
    pub fn ip_len(&self) -> usize {
        NGEN + self.pts.len() * self.stride()
    }

    pub fn init_state(&self, nip: usize) -> Vec<f64> {
        let mut st = vec![0.0; nip * self.ip_len()];
        for q in 0..nip {
            for k in 0..self.pts.len() {
                let o = q * self.ip_len() + NGEN + k * self.stride() + 3;
                self.mat.init_state(&mut st[o..o + self.nst]);
            }
        }
        st
    }

    /// Elastic section matrix 6×6 for [ε; κ] (local axes).
    pub fn elastic_c(&self) -> Result<[f64; 36]> {
        let d = self.mat.tangent_linear(StressMode::PlaneStress)?;
        let mut c = [0.0; 36];
        for p in &self.pts {
            let cl = rotate_tangent(&d, p.c, p.s);
            add_point(&mut c, &cl, p.z, p.w);
        }
        Ok(c)
    }

    /// Section forces [N; M] and tangent at an integration point. `e` is the
    /// generalized strains at the end of the increment (local axes), `temp`
    /// = (T, ΔT, ∂T/∂n, Δ∂T/∂n) at the midsurface; `so`/`sn` the point state.
    pub fn response(&self, ctx: &ElemCtx, e: &[f64; 6], temp: [f64; 4], so: &[f64], sn: &mut [f64]) -> Result<([f64; 6], [f64; 36])> {
        let st = self.stride();
        let e_old: [f64; 6] = if so.len() >= NGEN { std::array::from_fn(|i| so[i]) } else { [0.0; 6] };
        sn[..NGEN].copy_from_slice(e);
        let mut s6 = [0.0; 6];
        let mut c6 = [0.0; 36];
        for (k, p) in self.pts.iter().enumerate() {
            let el = [e[0] + p.z * e[3], e[1] + p.z * e[4], e[2] + p.z * e[5]];
            let eo = [e_old[0] + p.z * e_old[3], e_old[1] + p.z * e_old[4], e_old[2] + p.z * e_old[5]];
            let em = strain_to(&el, p.c, p.s);
            let emo = strain_to(&eo, p.c, p.s);
            let de = [em[0] - emo[0], em[1] - emo[1], em[2] - emo[2]];
            let o = NGEN + k * st;
            let (pso, psn) = (&so[o..o + st], &mut sn[o..o + st]);
            let inp = MatInput {
                mode: StressMode::PlaneStress,
                strain: &em,
                dstrain: &de,
                stress_old: &pso[..3],
                time: ctx.time,
                dt: ctx.dt,
                temperature: temp[0] + p.z * temp[2],
                dtemp: temp[1] + p.z * temp[3],
                defgrad: None,
            };
            let r = self.mat.update(&inp, &pso[3..], &mut psn[3..]).map_err(|err| crate::model::FemError(format!("element {} section point {}: {}", ctx.label, k + 1, err.0)))?;
            psn[..3].copy_from_slice(&r.stress[..3]);
            let sl = stress_from(&r.stress, p.c, p.s);
            for i in 0..3 {
                s6[i] += p.w * sl[i];
                s6[3 + i] += p.w * p.z * sl[i];
            }
            let cl = rotate_tangent(&r.tangent, p.c, p.s);
            add_point(&mut c6, &cl, p.z, p.w);
        }
        Ok((s6, c6))
    }

    /// Stress (material axes) at section point k from the state.
    fn stress_at(&self, ipst: &[f64], k: usize) -> [f64; 3] {
        let o = NGEN + k * self.stride();
        [ipst[o], ipst[o + 1], ipst[o + 2]]
    }

    /// Section forces [N; M] from stored stresses (local axes).
    pub fn forces_from_state(&self, ipst: &[f64]) -> [f64; 6] {
        let mut s6 = [0.0; 6];
        for (k, p) in self.pts.iter().enumerate() {
            let sl = stress_from(&self.stress_at(ipst, k), p.c, p.s);
            for i in 0..3 {
                s6[i] += p.w * sl[i];
                s6[3 + i] += p.w * p.z * sl[i];
            }
        }
        s6
    }

    /// Output S (section point 1 and the last), scalar material fields, SF and SM
    /// per integration point. `states` = state per integration point,
    /// `shear` = transverse shear forces (local axes), `mscale` = factor on SF1–SF3.
    pub fn output(&self, states: &[&[f64]], shear: &[[f64; 2]], mscale: f64) -> Vec<IpField> {
        let mut s = IpField::new("S", &["S11", "S22", "S12"]);
        let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
        let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
        let names: Vec<&'static str> = self
            .mat
            .output_fields(StressMode::PlaneStress)
            .into_iter()
            .filter(|(_, l)| l.len() == 1)
            .map(|(n, _)| n)
            .collect();
        let all_names: Vec<(&'static str, usize)> = self.mat.output_fields(StressMode::PlaneStress).into_iter().map(|(n, l)| (n, l.len())).collect();
        let mut extra: Vec<IpField> = names.iter().map(|n| IpField::new(n, &[n])).collect();
        let last = self.pts.len();
        let sps = if last > 1 { vec![1, last] } else { vec![1] };
        let (c, sn) = self.out;
        for (q, st) in states.iter().enumerate() {
            let ip = q + 1;
            let f = self.forces_from_state(st);
            let nn = stress_to(&[f[0], f[1], f[2]], c, sn);
            let mm = stress_to(&[f[3], f[4], f[5]], c, sn);
            let qv = shear[q];
            let qo = [c * qv[0] + sn * qv[1], -sn * qv[0] + c * qv[1]];
            sf.points.push((ip, 0, vec![nn[0] * mscale, nn[1] * mscale, nn[2] * mscale, qo[0], qo[1], 0.0]));
            sm.points.push((ip, 0, mm.to_vec()));
            for &sp in &sps {
                s.points.push((ip, sp, self.stress_at(st, sp - 1).to_vec()));
                if !names.is_empty() {
                    let o = NGEN + (sp - 1) * self.stride() + 3;
                    let vals = self.mat.output_field_values(StressMode::PlaneStress, &st[o..o + self.nst]);
                    let mut pos = 0;
                    let mut fi = 0;
                    for (n, len) in &all_names {
                        if *len == 1 && names.contains(n) {
                            extra[fi].points.push((ip, sp, vec![vals.get(pos).copied().unwrap_or(0.0)]));
                            fi += 1;
                        }
                        pos += len;
                    }
                }
            }
        }
        let mut out = vec![s, sf, sm];
        out.extend(extra);
        out
    }
}

/// Transverse shear stiffness [K11, K22, K12] in the element's local axes (scaled by
/// the factor 1/(1 + 0.25·10⁻⁴·A/t²)) and the section drill stiffness.
///
/// Homogeneous sections: like the linear element (5/6·G·t or
/// *TRANSVERSE SHEAR STIFFNESS). Layered sections without *TRANSVERSE SHEAR
/// STIFFNESS: for bending in the direction α (cylindrical bending)
/// the transverse shear stress is τ(z) = V·S(z)/D with S(z) = ∫ Q̄αα (z' − z_n) dz'
/// from the bottom, and the energy gives K_αα = D² / ∫ S²/G_α3 dz (5/6·G·t for a
/// homogeneous section). G_α3 is the layer's transverse shear modulus in the element axes
/// (G13 and G23 from the material's 3D stiffness; for LAMINA, which does not store them,
/// G12 is used).
pub fn shear_stiffness(ctx: &ElemCtx, name: &str, loc: &Local, sec: &mut ShellSec, a: f64) -> Result<[f64; 3]> {
    let fp = 1.0 / (1.0 + 0.25e-4 * a / (sec.t * sec.t));
    let composite = ctx.section.is_some_and(|s| s.raw.has("COMPOSITE"));
    if let Some(k) = sec.tss {
        if composite {
            sec.tss = Some([0.5 * (k[0] + k[1]), 0.5 * (k[0] + k[1]), 0.0]);
            return Ok([k[0] * fp, k[1] * fp, k[2] * fp]);
        }
    }
    if !composite {
        let ks = sec.ks(a);
        return Ok([ks, ks, 0.0]);
    }
    let lay = Layup::new(ctx, name, loc)?;
    let dps = lay.mat.tangent_linear(StressMode::PlaneStress)?;
    let (g13, g23) = match lay.mat.tangent_linear(StressMode::ThreeD) {
        Ok(d) if d.len() == 36 => (d[4 * 6 + 4], d[5 * 6 + 5]),
        _ => (dps[8], dps[8]),
    };
    // unit moment Mα (N = 0, other moments zero): strains from the section's
    // full flexibility, the stresses σ(z) = Q̄(z)(ε₀ + z κ) per unit
    // transverse force Vα = dMα/dxα, and the transverse shear stress from equilibrium
    // τ(z) = −∫ ∂σ/∂xα dz' (Vα gives both τ13 and τ23 in layers with an angle)
    let abd = lay.elastic_c()?;
    let Some(fl) = super::inv_n(&abd, 6) else {
        return fail(format!("element {} ({name}): the section stiffness matrix is singular", ctx.label));
    };
    let gp = [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];
    // τ per layer: τ(z) = τ(z0) − ∫_{z0}^{z} (q0 + q1 z') dz' (σ linear in z within the layer)
    let mut f = [[0.0; 2]; 2];
    let mut tau0 = [[0.0; 2]; 2]; // [load case][component 13, 23]
    for &(z0, z1, c, sn) in &lay.plies {
        let q = rotate_tangent(&dps, c, sn);
        let g = [g13 * c * c + g23 * sn * sn, g13 * sn * sn + g23 * c * c, c * sn * (g13 - g23)];
        let det = g[0] * g[1] - g[2] * g[2];
        let gi = [g[1] / det, g[0] / det, -g[2] / det];
        // load case α: generalized strains for the unit moment Mα
        let lin: Vec<[[f64; 2]; 2]> = (0..2)
            .map(|al| {
                let e: [f64; 6] = std::array::from_fn(|r| fl[r * 6 + 3 + al]);
                // σ(z) = Q̄ (ε₀ + z κ) = a + b z for the components 11, 22, 12
                let a: [f64; 3] = std::array::from_fn(|r| (0..3).map(|cc| q[r * 3 + cc] * e[cc]).sum());
                let b: [f64; 3] = std::array::from_fn(|r| (0..3).map(|cc| q[r * 3 + cc] * e[3 + cc]).sum());
                // ∂σ/∂xα: α = 1 gives (σ11, σ12) for (τ13, τ23), α = 2 gives (σ12, σ22)
                let (i13, i23) = if al == 0 { (0, 2) } else { (2, 1) };
                [[a[i13], b[i13]], [a[i23], b[i23]]]
            })
            .collect();
        let tau = |al: usize, comp: usize, z: f64| -> f64 {
            let [a, b] = lin[al][comp];
            tau0[al][comp] - (a * (z - z0) + 0.5 * b * (z * z - z0 * z0))
        };
        let h = z1 - z0;
        for &(xi, w) in &gp {
            let z = z0 + 0.5 * h * (1.0 + xi);
            let t: [[f64; 2]; 2] = std::array::from_fn(|al| [tau(al, 0, z), tau(al, 1, z)]);
            for al in 0..2 {
                for be in 0..2 {
                    let v = t[al][0] * (gi[0] * t[be][0] + gi[2] * t[be][1]) + t[al][1] * (gi[2] * t[be][0] + gi[1] * t[be][1]);
                    f[al][be] += 0.5 * h * w * v;
                }
            }
        }
        let next: [[f64; 2]; 2] = std::array::from_fn(|al| [tau(al, 0, z1), tau(al, 1, z1)]);
        tau0 = next;
    }
    // K = F⁻¹
    let det = f[0][0] * f[1][1] - f[0][1] * f[1][0];
    let k = [f[1][1] / det, f[0][0] / det, -f[0][1] / det];
    // drill and other elastic terms follow the transverse shear stiffness
    sec.tss = Some([0.5 * (k[0] + k[1]), 0.5 * (k[0] + k[1]), 0.0]);
    Ok([k[0] * fp, k[1] * fp, k[2] * fp])
}

/// Strain [ε11, ε22, γ12] in axes rotated by the angle (c, s) from the local ones.
pub fn strain_to(e: &[f64; 3], c: f64, s: f64) -> [f64; 3] {
    [
        c * c * e[0] + s * s * e[1] + c * s * e[2],
        s * s * e[0] + c * c * e[1] - c * s * e[2],
        -2.0 * c * s * e[0] + 2.0 * c * s * e[1] + (c * c - s * s) * e[2],
    ]
}

/// Stress [σ11, σ22, σ12] in axes rotated by the angle (c, s) from the local ones.
pub fn stress_to(sg: &[f64; 3], c: f64, s: f64) -> [f64; 3] {
    [
        c * c * sg[0] + s * s * sg[1] + 2.0 * c * s * sg[2],
        s * s * sg[0] + c * c * sg[1] - 2.0 * c * s * sg[2],
        -c * s * sg[0] + c * s * sg[1] + (c * c - s * s) * sg[2],
    ]
}

/// Stress in local axes from the material's: σ_l = Tεᵀ σ_m.
fn stress_from(sm: &[f64], c: f64, s: f64) -> [f64; 3] {
    [
        c * c * sm[0] + s * s * sm[1] - 2.0 * c * s * sm[2],
        s * s * sm[0] + c * c * sm[1] + 2.0 * c * s * sm[2],
        c * s * sm[0] - c * s * sm[1] + (c * c - s * s) * sm[2],
    ]
}

/// Tangent in local axes: Tεᵀ C Tε.
fn rotate_tangent(d: &[f64], c: f64, s: f64) -> [f64; 9] {
    let t = [c * c, s * s, c * s, s * s, c * c, -c * s, -2.0 * c * s, 2.0 * c * s, c * c - s * s];
    let mut dt = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            dt[i * 3 + j] = (0..3).map(|k| d[i * 3 + k] * t[k * 3 + j]).sum();
        }
    }
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = (0..3).map(|k| t[k * 3 + i] * dt[k * 3 + j]).sum();
        }
    }
    out
}

fn add_point(c6: &mut [f64; 36], cl: &[f64; 9], z: f64, w: f64) {
    for i in 0..3 {
        for j in 0..3 {
            let v = w * cl[i * 3 + j];
            c6[i * 6 + j] += v;
            c6[i * 6 + j + 3] += z * v;
            c6[(i + 3) * 6 + j] += z * v;
            c6[(i + 3) * 6 + j + 3] += z * z * v;
        }
    }
}

/// Temperature (T, ΔT, ∂T/∂n, Δ∂T/∂n) at the midsurface. The nodal temperatures in
/// `ctx.temp` are followed by the gradients through the thickness if present
/// (2·nn values, see `fem::assembly`).
pub fn temperature(ctx: &ElemCtx, nn: usize) -> [f64; 4] {
    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    if ctx.temp.len() < nn || ctx.temp_old.len() < nn {
        return [0.0; 4];
    }
    let t = avg(&ctx.temp[..nn]);
    let t0 = avg(&ctx.temp_old[..nn]);
    let (g, g0) = if ctx.temp.len() >= 2 * nn && ctx.temp_old.len() >= 2 * nn { (avg(&ctx.temp[nn..2 * nn]), avg(&ctx.temp_old[nn..2 * nn])) } else { (0.0, 0.0) };
    [t, t - t0, g, g - g0]
}

/// The element kinematics for the integrated formulation: B matrices
/// (6 × 6nn, local degrees of freedom) and weights per integration point, the
/// elastic remainder (shear, hourglass, drill; local) and the transverse shear
/// per integration point (2 × 6nn).
pub struct Kin {
    pub loc: Local,
    pub sec: ShellSec,
    pub ips: Vec<(Vec<f64>, f64)>,
    pub rest: Vec<f64>,
    pub shear: Vec<Vec<f64>>,
    /// The transverse shear stiffness [K11, K22, K12] (scaled, local axes).
    pub kshear: [f64; 3],
    pub rule: MemRule,
}

impl Kin {
    fn nd(&self) -> usize {
        self.rest.len().isqrt()
    }

    fn t(&self) -> Vec<f64> {
        common::block_diag(&self.loc.r, 2 * self.nd() / 6)
    }

    /// Elastic stiffness (global) with the elastic section matrix.
    pub fn stiffness(&self, lay: &Layup) -> Result<Vec<f64>> {
        let c = lay.elastic_c()?;
        let nd = self.nd();
        let mut k = self.rest.clone();
        for (b, w) in &self.ips {
            super::add_btcb(&mut k, nd, b, &c, b, 6, *w);
        }
        Ok(super::to_global(&k, &self.loc.r, nd / 6))
    }

    /// Internal force and tangent (geometrically linear).
    pub fn response(&self, ctx: &ElemCtx, lay: &Layup, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let nd = self.nd();
        let ul = common::matvec(&self.t(), u);
        let len = lay.ip_len();
        let so_ok = so.len() == self.ips.len() * len;
        let init;
        let so = if so_ok {
            so
        } else {
            init = lay.init_state(self.ips.len());
            &init[..]
        };
        sn.clear();
        sn.extend_from_slice(so);
        let temp = temperature(ctx, nd / 6);
        let mut fl = common::matvec(&self.rest, &ul);
        let mut kl = if tangent { self.rest.clone() } else { Vec::new() };
        for (q, (b, w)) in self.ips.iter().enumerate() {
            let e: [f64; 6] = std::array::from_fn(|r| (0..nd).map(|j| b[r * nd + j] * ul[j]).sum());
            let (s, c) = lay.response(ctx, &e, temp, &so[q * len..(q + 1) * len], &mut sn[q * len..(q + 1) * len])?;
            for j in 0..nd {
                fl[j] += w * (0..6).map(|r| b[r * nd + j] * s[r]).sum::<f64>();
            }
            if tangent {
                super::add_btcb(&mut kl, nd, b, &c, b, 6, *w);
            }
        }
        let fint = super::vec_to_global(&fl, &self.loc.r, 2 * nd / 6);
        let k = tangent.then(|| super::to_global(&kl, &self.loc.r, nd / 6));
        Ok(ElemResponse { fint, k })
    }

    /// Transverse shear forces per integration point for local displacements `ul`.
    pub fn shear_forces(&self, ul: &[f64]) -> Vec<[f64; 2]> {
        let nd = self.nd();
        let k = self.kshear;
        self.shear
            .iter()
            .map(|g| {
                let g1: f64 = (0..nd).map(|j| g[j] * ul[j]).sum();
                let g2: f64 = (0..nd).map(|j| g[nd + j] * ul[j]).sum();
                [k[0] * g1 + k[2] * g2, k[2] * g1 + k[1] * g2]
            })
            .collect()
    }

    /// Output. With NLGEOM `u` is the total displacements; the transverse shear forces
    /// are then computed with the deformational ones.
    pub fn output(&self, ctx: &ElemCtx, name: &str, lay: &Layup, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let len = lay.ip_len();
        let nip = self.ips.len();
        let ud = if ctx.nlgeom { nlgeom::deformational_u(ctx, name, u)? } else { u.to_vec() };
        let ul = common::matvec(&self.t(), &ud);
        let mut st = state.to_vec();
        if st.len() != nip * len && ctx.nlgeom {
            st = lay.init_state(nip);
        } else if !ctx.nlgeom && (st.len() != nip * len || ctx.material.is_some_and(|m| m.is_linear())) {
            // linear material (without state in linear steps): recompute from u
            let so = lay.init_state(nip);
            let mut sn = Vec::new();
            self.response(ctx, lay, u, &so, &mut sn, false)?;
            st = sn;
        }
        let states: Vec<&[f64]> = (0..nip).map(|q| &st[q * len..(q + 1) * len]).collect();
        let shear = self.shear_forces(&ul);
        let ms = if ctx.nlgeom {
            let el = nlgeom::membrane_strains(ctx, name, u, self.rule)?[0];
            (-(el[0] + el[1])).exp()
        } else {
            1.0
        };
        Ok(lay.output(&states, &shear, ms))
    }

    /// Internal force and tangent with NLGEOM (corotational, see the module documentation).
    #[allow(clippy::too_many_arguments)]
    pub fn response_nlgeom(&self, ctx: &ElemCtx, name: &str, lay: &Layup, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        match self.nd() {
            24 => self.nl_n::<24>(ctx, name, lay, u, so, sn, tangent),
            18 => self.nl_n::<18>(ctx, name, lay, u, so, sn, tangent),
            n => fail(format!("element {} ({name}): NLGEOM with {n} degrees of freedom is not supported", ctx.label)),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn nl_n<const N: usize>(&self, ctx: &ElemCtx, name: &str, lay: &Layup, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let var = |a: usize, c: usize| -> Hd<N> { Hd::var(0.0, 6 * a + c) };
        let d = nlgeom::deformational_hd::<N>(ctx, name, u, &var)?;
        let mem = nlgeom::Membrane::new(ctx, name, self.rule)?;
        let eps = mem.strains::<N>(&nlgeom::positions_hd::<N>(&ctx.x, u, 6, &var));
        let t = self.t();
        // local deformational displacements ul = T d (hyper-dual)
        let ul: Vec<Hd<N>> = (0..N).map(|i| (0..N).fold(Hd::cst(0.0), |acc, j| if t[i * N + j] != 0.0 { acc + d[j] * t[i * N + j] } else { acc })).collect();
        let lin = |b: &[f64], r: usize| -> Hd<N> { (0..N).fold(Hd::cst(0.0), |acc, j| if b[r * N + j] != 0.0 { acc + ul[j] * b[r * N + j] } else { acc }) };
        // the linear membrane part at the center (for the assumed strain of S4)
        let (b0, _) = &self.ips[0];
        let bm_c: [Hd<N>; 3] = if self.rule == MemRule::Center && self.ips.len() > 1 { mem_center::<N>(self, &ul) } else { std::array::from_fn(|r| lin(b0, r)) };
        let len = lay.ip_len();
        let nip = self.ips.len();
        let init;
        let so = if so.len() == nip * len {
            so
        } else {
            init = lay.init_state(nip);
            &init[..]
        };
        sn.clear();
        sn.extend_from_slice(so);
        let temp = temperature(ctx, N / 6);
        // elastic remainder: Φ = ½ dᵀ K_r d in global reference axes
        let kr = super::to_global(&self.rest, &self.loc.r, N / 6);
        let s: Vec<f64> = (0..N).map(|i| (0..N).map(|j| kr[i * N + j] * d[j].v).sum()).collect();
        let mut fint: Vec<f64> = (0..N).map(|q| (0..N).map(|i| d[i].g[q] * s[i]).sum()).collect();
        let mut k = vec![0.0; N * N];
        if tangent {
            let mut kj = vec![0.0; N * N];
            for i in 0..N {
                for j in 0..N {
                    let kij = kr[i * N + j];
                    if kij != 0.0 {
                        for q in 0..N {
                            kj[i * N + q] += kij * d[j].g[q];
                        }
                    }
                }
            }
            for i in 0..N {
                for p in 0..N {
                    let jip = d[i].g[p];
                    if jip != 0.0 {
                        for q in 0..N {
                            k[p * N + q] += jip * kj[i * N + q];
                        }
                    }
                }
                if s[i] != 0.0 {
                    for p in 0..N {
                        for q in 0..N {
                            k[p * N + q] += s[i] * d[i].h[p][q];
                        }
                    }
                }
            }
        }
        let el = eps[0];
        for (q, (b, w)) in self.ips.iter().enumerate() {
            // generalized strains: logarithmic membrane strain (plus the assumed
            // strain of S4 at the integration point) and linear curvature
            let e: [Hd<N>; 6] = std::array::from_fn(|r| if r < 3 { el[r] + (lin(b, r) - bm_c[r]) } else { lin(b, r) });
            let ev: [f64; 6] = std::array::from_fn(|r| e[r].v);
            let (sg, c) = lay.response(ctx, &ev, temp, &so[q * len..(q + 1) * len], &mut sn[q * len..(q + 1) * len])?;
            for p in 0..N {
                fint[p] += w * (0..6).map(|r| e[r].g[p] * sg[r]).sum::<f64>();
            }
            if tangent {
                for p in 0..N {
                    for r in 0..6 {
                        let erp = e[r].g[p];
                        if erp == 0.0 {
                            continue;
                        }
                        for s2 in 0..6 {
                            let v = w * erp * c[r * 6 + s2];
                            if v != 0.0 {
                                for qq in 0..N {
                                    k[p * N + qq] += v * e[s2].g[qq];
                                }
                            }
                        }
                    }
                }
                for r in 0..6 {
                    if sg[r] != 0.0 {
                        for p in 0..N {
                            for qq in 0..N {
                                k[p * N + qq] += w * sg[r] * e[r].h[p][qq];
                            }
                        }
                    }
                }
            }
        }
        if fint.iter().any(|v| !v.is_finite()) {
            return fail(format!("element {} ({name}): invalid deformed geometry", ctx.label));
        }
        if tangent {
            for p in 0..N {
                for q in 0..p {
                    let v = 0.5 * (k[p * N + q] + k[q * N + p]);
                    k[p * N + q] = v;
                    k[q * N + p] = v;
                }
            }
        }
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }
}

/// The linear membrane strain at the center (S4: the one-point part without the assumed strain).
fn mem_center<const N: usize>(kin: &Kin, ul: &[Hd<N>]) -> [Hd<N>; 3] {
    let p0 = super::q4pt(&kin.loc.xl, 0.0, 0.0);
    let b = super::bmem(&p0.bx, &p0.by);
    std::array::from_fn(|r| (0..N).fold(Hd::cst(0.0), |acc, j| if b[r * N + j] != 0.0 { acc + ul[j] * b[r * N + j] } else { acc }))
}

/// The element's local system (for `Layup::new` before the kinematics).
pub fn frame(ctx: &ElemCtx, name: &str) -> Result<Local> {
    local_frame(&ctx.x, ctx.label, name)
}
