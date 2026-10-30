//! Contact against analytical rigid surfaces: *SURFACE, TYPE=SEGMENTS (planar and
//! axisymmetric models), TYPE=CYLINDER and TYPE=REVOLUTION (3D) with
//! *RIGID BODY, ANALYTICAL SURFACE=, REF NODE= as the master surface in *CONTACT PAIR.
//!
//! Geometry:
//! - the profile is connected segments in a plane: START, LINE (end point),
//!   CIRCL (end point, center; the shorter arc, < 179.74°) and PARAB
//!   (midpoint, end point; quadratic curve with the midpoint at parameter ½),
//! - SEGMENTS: global (X, Y) or (r, z); CYLINDER: local (x, y) with origin a,
//!   the x-axis towards b and the generator vector a → c along negative local z
//!   (c is adjusted to be perpendicular to ab), infinite along the generator vector;
//!   REVOLUTION: (r, z) with origin a and the symmetry axis a → b (local z),
//! - the outward normal is n = e₃ × s (s = profile direction, e₃ = plane normal),
//!   i.e. the direction rotated 90° counterclockwise in the profile plane,
//! - inside a *PART: the profile in the part's system (CYLINDER: a = origin, b on
//!   the x-axis, c on the negative z-axis; REVOLUTION: a = origin, r along x, the axis
//!   along y), placed with the instance's translation and rotation.
//!
//! The surface follows the reference node R exactly (even without NLGEOM, as is usual for
//! analytical surfaces): x = x_R + R(θ_R)(X − X_R). The slave node is projected onto the surface in
//! the body's reference configuration, p_b = X_R + Rᵀ(x_s − x_R); the closest point q and
//! the normal n are mapped back with R. The gap is g = n · (x_s − q).
//!
//! Discretization: node to surface (default for *CONTACT PAIR in implicit analysis).
//! Finite sliding: closest point in every evaluation; SMALL SLIDING:
//! the tangent plane at the slave node's projection in the reference configuration, which follows the body.
//! Normal contact and friction as in `contact.rs` (HARD with a stiff
//! penalty coefficient 1e8 · element stiffness/area against a rigid surface, PENALTY,
//! LINEAR/EXPONENTIAL, Coulomb with penalty-regularized sticking and critical elastic
//! slip SLIP TOLERANCE · mean size of the slave faces). The relative slip
//! is computed in the body's system (slave node motion relative to the surface).
//!
//! The variation δg = n · (δu_s − δu_R − δw × a) with a = q − x_R gives the forces
//! f_s = A(−p n + τ), f_R = −f_s and M_R = −a × f_s on the reference node and
//! the tangent A Gᵀ (p' n nᵀ + K_t) G (without the surface curvature term, which only
//! affects convergence).
//!
//! Limitations: no SURFACE TO SURFACE, no FILLET RADIUS, no
//! user surfaces (TYPE=USER/RSURFU), no *CLEARANCE.

use super::contact::{self, PairProps};
use super::geom::{self, Facet};
use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::elements::common::{cross, dot, norm, scale, sub};
use crate::fem::dof::{Dof, NodeVec, U1, UR1, UR3};
use crate::fem::model::{parse_f64, FeModel, Placement, RawKeyword};
use crate::kinematics::tensor;
use crate::model::{fail, Result};
use std::sync::Mutex;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CONTACT PAIR"], build }];

/// Penalty coefficient for hard contact (direct method) against an analytical surface,
/// relative to the slave side's element stiffness. Lower than for rigid elements (1e8): with
/// curved surfaces and coordinates of order 1, round-off in the closest
/// point (~1e-16) otherwise gives force noise above the convergence criteria, and the gap's
/// linearization error during sliding becomes too large. The penetration becomes
/// ~1e-7 of the deformation.
const HARD_FACTOR_ANALYTICAL: f64 = 1e7;

// ---------------------------------------------------------------- names and placement

/// The placement as a parameter (`_PLACE`) for keywords from a part.
pub fn encode_placement(p: &Placement) -> String {
    let t = p.translation;
    match p.rotation {
        Some((a, b, d)) => format!("{};{};{};{};{};{};{};{};{};{}", t[0], t[1], t[2], a[0], a[1], a[2], b[0], b[1], b[2], d),
        None => format!("{};{};{}", t[0], t[1], t[2]),
    }
}

fn decode_placement(s: &str) -> Placement {
    let v: Vec<f64> = s.split(';').filter_map(|x| x.trim().parse::<f64>().ok()).collect();
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let rotation = (v.len() >= 10).then(|| ([g(3), g(4), g(5)], [g(6), g(7), g(8)], g(9)));
    Placement { translation: [g(0), g(1), g(2)], rotation }
}

/// The instance placement for a keyword from a part (none if it is at model level).
pub fn placement_of(kw: &RawKeyword) -> Option<Placement> {
    kw.param("_PLACE").map(decode_placement)
}

/// A name in the keyword: with instance prefix if the keyword comes from a part.
pub fn scoped(kw: &RawKeyword, token: &str) -> String {
    let t = token.trim().to_uppercase();
    match kw.param("_INSTANCE") {
        Some(i) => format!("{}.{t}", i.to_uppercase()),
        None => t,
    }
}

/// Is the surface an analytical rigid surface?
pub fn is_analytical(m: &FeModel, name: &str) -> bool {
    m.surfaces.get(&name.trim().to_uppercase()).is_some_and(|s| crate::inp::keywords::analytical::is_analytical_type(&s.kind))
}

// ---------------------------------------------------------------- profile

#[derive(Debug, Clone)]
enum Seg {
    Line { a: [f64; 2], b: [f64; 2] },
    /// Center, radius, start angle and (signed) angle change.
    Arc { c: [f64; 2], r: f64, t0: f64, dt: f64 },
    /// Quadratic curve through p0 (t = 0), pm (t = ½) and p1 (t = 1).
    Parab { p0: [f64; 2], pm: [f64; 2], p1: [f64; 2] },
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}
fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}
fn len2(a: [f64; 2]) -> f64 {
    dot2(a, a).sqrt()
}

impl Seg {
    /// (point, unit tangent in the profile direction) at parameter t ∈ [0, 1].
    fn eval(&self, t: f64) -> ([f64; 2], [f64; 2]) {
        match *self {
            Seg::Line { a, b } => {
                let d = sub2(b, a);
                let l = len2(d);
                ([a[0] + t * d[0], a[1] + t * d[1]], [d[0] / l, d[1] / l])
            }
            Seg::Arc { c, r, t0, dt } => {
                let th = t0 + t * dt;
                let sg = dt.signum();
                ([c[0] + r * th.cos(), c[1] + r * th.sin()], [-sg * th.sin(), sg * th.cos()])
            }
            Seg::Parab { p0, pm, p1 } => {
                let (l0, lm, l1) = (2.0 * (t - 0.5) * (t - 1.0), -4.0 * t * (t - 1.0), 2.0 * t * (t - 0.5));
                let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
                let p = [l0 * p0[0] + lm * pm[0] + l1 * p1[0], l0 * p0[1] + lm * pm[1] + l1 * p1[1]];
                let d = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
                let l = len2(d);
                (p, [d[0] / l, d[1] / l])
            }
        }
    }

    /// Parameter of the closest point (unclamped for line and arc).
    fn closest_t(&self, p: [f64; 2]) -> f64 {
        match *self {
            Seg::Line { a, b } => {
                let d = sub2(b, a);
                dot2(sub2(p, a), d) / dot2(d, d)
            }
            Seg::Arc { c, t0, dt, .. } => {
                let phi = (p[1] - c[1]).atan2(p[0] - c[0]);
                let mut rel = phi - t0;
                // relative angle in (−π, π] around the arc's midpoint
                let mid = 0.5 * dt;
                while rel - mid > std::f64::consts::PI {
                    rel -= 2.0 * std::f64::consts::PI;
                }
                while rel - mid <= -std::f64::consts::PI {
                    rel += 2.0 * std::f64::consts::PI;
                }
                rel / dt
            }
            Seg::Parab { .. } => {
                // initial value from a coarse search, then Newton on (x(t) − p) · x'(t) = 0
                let mut best = (f64::INFINITY, 0.0);
                for k in 0..=20 {
                    let t = k as f64 / 20.0;
                    let (x, _) = self.eval(t);
                    let d = len2(sub2(x, p));
                    if d < best.0 {
                        best = (d, t);
                    }
                }
                let mut t = best.1;
                let Seg::Parab { p0, pm, p1 } = *self else { unreachable!() };
                let dd = [4.0 * p0[0] - 8.0 * pm[0] + 4.0 * p1[0], 4.0 * p0[1] - 8.0 * pm[1] + 4.0 * p1[1]];
                for _ in 0..50 {
                    let (x, _) = self.eval(t);
                    let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
                    let d = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
                    let r = sub2(x, p);
                    let g = dot2(r, d);
                    let h = dot2(d, d) + dot2(r, dd);
                    if h.abs() < 1e-300 {
                        break;
                    }
                    let st = g / h;
                    t -= st;
                    if st.abs() < 1e-15 {
                        break;
                    }
                }
                t
            }
        }
    }
}

/// Closest point on the profile: (point, outward normal, signed distance).
/// `None` if the point lies beyond the ends of the profile.
fn profile_closest(prof: &[Seg], p: [f64; 2]) -> Option<([f64; 2], [f64; 2], f64)> {
    let mut best: Option<(f64, usize, f64)> = None;
    for (i, s) in prof.iter().enumerate() {
        let t = s.closest_t(p).clamp(0.0, 1.0);
        let (q, _) = s.eval(t);
        let d = len2(sub2(p, q));
        if best.is_none_or(|b| d < b.0 - 1e-14 * (1.0 + d)) {
            best = Some((d, i, t));
        }
    }
    let (_, i, t) = best?;
    let tol = 1e-12;
    let (q, s) = prof[i].eval(t);
    let n = [-s[1], s[0]];
    let at_start = t <= tol;
    let at_end = t >= 1.0 - tol;
    if (at_start && i == 0) || (at_end && i + 1 == prof.len()) {
        // beyond the end of the profile: only if the point lies straight above/below the end
        let raw = prof[i].closest_t(p);
        if (at_start && raw < -1e-6) || (at_end && raw > 1.0 + 1e-6) {
            return None;
        }
    }
    if at_start || at_end {
        // corner between two segments: the normal towards the point (on the side of the mean normal)
        let j = if at_start { i.wrapping_sub(1) } else { i + 1 };
        if let Some(o) = prof.get(j) {
            let (_, so) = o.eval(if at_start { 1.0 } else { 0.0 });
            // smooth transitions (same tangent) are treated as an ordinary point on the surface
            let smooth = dot2(so, s) > 1.0 - 1e-10;
            let navg = [n[0] - so[1], n[1] + so[0]];
            let d = sub2(p, q);
            let l = len2(d);
            if !smooth && l > 1e-6 * (1.0 + len2(q)) {
                let sg = if dot2(d, navg) >= 0.0 { 1.0 } else { -1.0 };
                let nn = [sg * d[0] / l, sg * d[1] / l];
                return Some((q, nn, sg * l));
            }
        }
    }
    let g = dot2(sub2(p, q), n);
    Some((q, n, g))
}

/// Change of the normal along the profile at the point `p`: (κ', s) with
/// δn = κ' s (s · δp), where s is the tangent and κ' = κ/(1 + κ g) with κ > 0 for
/// a convex surface (normal away from the center of curvature). Corner: κ' = 1/distance.
fn profile_curvature(prof: &[Seg], p: [f64; 2]) -> Option<(f64, [f64; 2])> {
    let (q, n, g) = profile_closest(prof, p)?;
    let s = [n[1], -n[0]];
    let mut best: Option<(f64, usize, f64)> = None;
    for (i, sg) in prof.iter().enumerate() {
        let t = sg.closest_t(p).clamp(0.0, 1.0);
        let (qq, _) = sg.eval(t);
        let d = len2(sub2(qq, q));
        if best.is_none_or(|b| d < b.0) {
            best = Some((d, i, t));
        }
    }
    let (_, i, t) = best?;
    let (_, si) = prof[i].eval(t);
    if dot2(si, s).abs() < 1.0 - 1e-9 {
        // corner: the normal is the direction towards the point
        return Some((if g != 0.0 { 1.0 / g } else { 0.0 }, s));
    }
    let kappa = match prof[i] {
        Seg::Line { .. } => 0.0,
        Seg::Arc { r, dt, .. } => -dt.signum() / r,
        Seg::Parab { p0, pm, p1 } => {
            let (d0, dm, d1) = (4.0 * t - 3.0, -8.0 * t + 4.0, 4.0 * t - 1.0);
            let x1 = [d0 * p0[0] + dm * pm[0] + d1 * p1[0], d0 * p0[1] + dm * pm[1] + d1 * p1[1]];
            let x2 = [4.0 * p0[0] - 8.0 * pm[0] + 4.0 * p1[0], 4.0 * p0[1] - 8.0 * pm[1] + 4.0 * p1[1]];
            -(x1[0] * x2[1] - x1[1] * x2[0]) / len2(x1).powi(3)
        }
    };
    let den = 1.0 + kappa * g;
    Some((if den.abs() > 1e-12 { kappa / den } else { 0.0 }, s))
}

fn parse_profile(kw: &RawKeyword, recs: &[crate::fem::model::DataLine]) -> Result<Vec<Seg>> {
    let mut prof = Vec::new();
    let mut cur: Option<[f64; 2]> = None;
    for d in recs {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).collect();
        if f.iter().all(|s| s.is_empty()) {
            continue;
        }
        let word = f[0].to_uppercase();
        let v: Vec<f64> = f[1..].iter().filter(|s| !s.is_empty()).map(|s| parse_f64(s).ok_or(())).collect::<std::result::Result<_, _>>().or_else(|_| kw.err_at(d.line, "invalid number"))?;
        let need = |k: usize| -> Result<()> {
            if v.len() < k {
                return kw.err_at(d.line, format!("{word} requires {k} coordinates"));
            }
            Ok(())
        };
        match word.as_str() {
            "START" => {
                need(2)?;
                if cur.is_some() {
                    return kw.err_at(d.line, "START may only appear first");
                }
                cur = Some([v[0], v[1]]);
            }
            "LINE" | "CIRCL" | "PARAB" | "PARABOLA" => {
                let Some(a) = cur else { return kw.err_at(d.line, "the profile must start with START") };
                match word.as_str() {
                    "LINE" => {
                        need(2)?;
                        let b = [v[0], v[1]];
                        if len2(sub2(b, a)) == 0.0 {
                            return kw.err_at(d.line, "the line has zero length");
                        }
                        prof.push(Seg::Line { a, b });
                        cur = Some(b);
                    }
                    "CIRCL" => {
                        need(4)?;
                        let (b, c) = ([v[0], v[1]], [v[2], v[3]]);
                        let (r0, r1) = (len2(sub2(a, c)), len2(sub2(b, c)));
                        if r0 == 0.0 || (r0 - r1).abs() > 1e-6 * r0 {
                            return kw.err_at(d.line, "the arc's start and end points are not at the same distance from the center");
                        }
                        let t0 = (a[1] - c[1]).atan2(a[0] - c[0]);
                        let t1 = (b[1] - c[1]).atan2(b[0] - c[0]);
                        let mut dt = t1 - t0;
                        while dt > std::f64::consts::PI {
                            dt -= 2.0 * std::f64::consts::PI;
                        }
                        while dt <= -std::f64::consts::PI {
                            dt += 2.0 * std::f64::consts::PI;
                        }
                        if dt.abs() >= 179.74f64.to_radians() || dt == 0.0 {
                            return kw.err_at(d.line, "the arc must be shorter than 179.74° (split it in two)");
                        }
                        prof.push(Seg::Arc { c, r: r0, t0, dt });
                        cur = Some(b);
                    }
                    _ => {
                        need(4)?;
                        let (pm, p1) = ([v[0], v[1]], [v[2], v[3]]);
                        prof.push(Seg::Parab { p0: a, pm, p1 });
                        cur = Some(p1);
                    }
                }
            }
            other => return kw.err_at(d.line, format!("unknown segment '{other}' (START, LINE, CIRCL, PARAB)")),
        }
    }
    if prof.is_empty() {
        return kw.err("the analytical surface has no segments (LINE, CIRCL, PARAB)");
    }
    Ok(prof)
}

// ---------------------------------------------------------------- the surface

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    /// SEGMENTS or CYLINDER: the profile in (x, y), z = ez.
    Swept,
    /// REVOLUTION: the profile in (r, z) around ez.
    Revolution,
}

/// An analytical surface in the reference configuration (global coordinates).
#[derive(Debug, Clone)]
pub struct AnalyticalSurface {
    kind: Kind,
    a: [f64; 3],
    ex: [f64; 3],
    ey: [f64; 3],
    ez: [f64; 3],
    prof: Vec<Seg>,
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v).max(1e-300))
}

impl AnalyticalSurface {
    /// Reads the surface `name` (upper case, with instance prefix) from the model's stored *SURFACE.
    pub fn from_model(m: &FeModel, name: &str) -> Result<AnalyticalSurface> {
        let up = name.trim().to_uppercase();
        let Some(kw) = m.extra.iter().find(|k| k.name == "SURFACE" && k.param("NAME").is_some_and(|n| scoped(k, n) == up)) else {
            return fail(format!("the analytical surface {up} does not exist"));
        };
        if kw.param("FILLET RADIUS").and_then(parse_f64).is_some_and(|r| r != 0.0) {
            return kw.err("FILLET RADIUS is not supported yet");
        }
        let ty = kw.param_upper("TYPE").unwrap_or_default();
        let recs: Vec<crate::fem::model::DataLine> = kw.records().into_iter().filter(|d| d.fields.iter().any(|s| !s.trim().is_empty())).collect();
        let is_word = |d: &crate::fem::model::DataLine| d.fields.first().is_some_and(|s| s.trim().chars().next().is_some_and(|c| c.is_ascii_alphabetic()));
        let in_part = kw.has("_PART");
        let pts = |d: &crate::fem::model::DataLine, k: usize| -> Result<Vec<f64>> {
            let v = kw.floats(d)?;
            if v.len() < k {
                return kw.err_at(d.line, format!("the line must have {k} coordinates"));
            }
            Ok(v)
        };
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let (kind, a, ex, ey, ez, first) = match ty.as_str() {
            "SEGMENTS" => {
                if m.dim() != 2 {
                    return kw.err("TYPE=SEGMENTS applies to planar and axisymmetric models (use CYLINDER or REVOLUTION in 3D)");
                }
                (Kind::Swept, [0.0; 3], x, y, z, 0)
            }
            "CYLINDER" => {
                if !recs.first().is_some_and(|d| !is_word(d)) {
                    if !in_part {
                        return kw.err("TYPE=CYLINDER requires the points a, b and c on the first two data lines");
                    }
                    (Kind::Swept, [0.0; 3], x, y, z, 0)
                } else {
                    let v1 = pts(&recs[0], 6)?;
                    let Some(d2) = recs.get(1).filter(|d| !is_word(d)) else { return kw.err("point c (the generator vector) is missing on the second data line") };
                    let v2 = pts(d2, 3)?;
                    let a = [v1[0], v1[1], v1[2]];
                    let ex = unit(sub([v1[3], v1[4], v1[5]], a));
                    let g = sub([v2[0], v2[1], v2[2]], a);
                    let g = sub(g, scale(ex, dot(g, ex)));
                    if norm(g) == 0.0 {
                        return kw.err("point c lies on the line ab");
                    }
                    // the generator vector a → c is the negative local z-axis
                    let ez = scale(unit(g), -1.0);
                    let ey = cross(ez, ex);
                    (Kind::Swept, a, ex, ey, ez, 2)
                }
            }
            "REVOLUTION" => {
                if !recs.first().is_some_and(|d| !is_word(d)) {
                    if !in_part {
                        return kw.err("TYPE=REVOLUTION requires the points a and b on the first data line");
                    }
                    // in a part: r along x, the symmetry axis along y
                    (Kind::Revolution, [0.0; 3], x, [0.0, 0.0, -1.0], y, 0)
                } else {
                    let v1 = pts(&recs[0], 6)?;
                    let a = [v1[0], v1[1], v1[2]];
                    let ez = unit(sub([v1[3], v1[4], v1[5]], a));
                    let t = if ez[0].abs() < 0.9 { x } else { y };
                    let ex = unit(sub(t, scale(ez, dot(t, ez))));
                    let ey = cross(ez, ex);
                    (Kind::Revolution, a, ex, ey, ez, 1)
                }
            }
            other => return kw.err(format!("TYPE={other} is not supported for analytical surfaces")),
        };
        let prof = parse_profile(kw, &recs[first..])?;
        let mut s = AnalyticalSurface { kind, a, ex, ey, ez, prof };
        if let Some(p) = placement_of(kw) {
            s.a = p.point(s.a);
            s.ex = p.direction(s.ex);
            s.ey = p.direction(s.ey);
            s.ez = p.direction(s.ez);
        }
        Ok(s)
    }

    /// Variation of the normal for the point `p` (reference configuration): δn = Σ κ'ᵢ tᵢ (tᵢ · δp)
    /// with the tangent directions tᵢ (the profile direction and, for REVOLUTION,
    /// the circumferential direction with κ' = n_r / r).
    pub fn curvatures(&self, p: [f64; 3]) -> Vec<(f64, [f64; 3])> {
        let d = sub(p, self.a);
        let zc = dot(d, self.ez);
        match self.kind {
            Kind::Swept => {
                let Some((k, s)) = profile_curvature(&self.prof, [dot(d, self.ex), dot(d, self.ey)]) else { return Vec::new() };
                vec![(k, [0, 1, 2].map(|c| s[0] * self.ex[c] + s[1] * self.ey[c]))]
            }
            Kind::Revolution => {
                let rad = sub(d, scale(self.ez, zc));
                let r = norm(rad);
                if r <= 1e-14 * (1.0 + norm(d)) {
                    return Vec::new();
                }
                let er = scale(rad, 1.0 / r);
                let Some((k, s)) = profile_curvature(&self.prof, [r, zc]) else { return Vec::new() };
                let Some((_, n, _)) = profile_closest(&self.prof, [r, zc]) else { return Vec::new() };
                let et = cross(self.ez, er);
                vec![(k, [0, 1, 2].map(|c| s[0] * er[c] + s[1] * self.ez[c])), (n[0] / r, et)]
            }
        }
    }

    /// Closest point and outward normal for the point `p` (reference configuration):
    /// (point, normal, signed distance).
    pub fn closest(&self, p: [f64; 3]) -> Option<([f64; 3], [f64; 3], f64)> {
        let d = sub(p, self.a);
        let zc = dot(d, self.ez);
        match self.kind {
            Kind::Swept => {
                let (xl, yl) = (dot(d, self.ex), dot(d, self.ey));
                let (q, n, g) = profile_closest(&self.prof, [xl, yl])?;
                let q3 = [0, 1, 2].map(|k| self.a[k] + q[0] * self.ex[k] + q[1] * self.ey[k] + zc * self.ez[k]);
                let n3 = [0, 1, 2].map(|k| n[0] * self.ex[k] + n[1] * self.ey[k]);
                Some((q3, n3, g))
            }
            Kind::Revolution => {
                let rad = sub(d, scale(self.ez, zc));
                let r = norm(rad);
                let er = if r > 1e-14 * (1.0 + norm(d)) { scale(rad, 1.0 / r) } else { self.ex };
                let (q, n, g) = profile_closest(&self.prof, [r, zc])?;
                let q3 = [0, 1, 2].map(|k| self.a[k] + q[0] * er[k] + q[1] * self.ez[k]);
                let n3 = [0, 1, 2].map(|k| n[0] * er[k] + n[1] * self.ez[k]);
                Some((q3, n3, g))
            }
        }
    }
}

// ---------------------------------------------------------------- the contact

#[derive(Debug, Clone, Copy, Default)]
struct NodeState {
    p: f64,
    gap: f64,
    /// Elastic slip in the body's system.
    slip: [f64; 3],
    /// Total slip (spatial, for CSLIP).
    total_slip: [f64; 3],
    shear: [f64; 3],
    normal: [f64; 3],
    contact: bool,
    /// The slave node's position in the body's system.
    pb: [f64; 3],
    /// CPRESS (contact force / current area).
    cpress: f64,
    /// Number of consecutive evaluations the node has been closed (contact status).
    age: u32,
}

#[derive(Debug, Default)]
struct State {
    committed: Vec<NodeState>,
    trial: Vec<NodeState>,
    /// Largest contact pressure in the latest evaluation (for the opening tolerance).
    pmax: f64,
}

/// Area per slave node (axisymmetric: the full revolution, 2π r, like the element
/// forces) in the reference configuration or, with `u`, in the current configuration.
fn slave_areas(m: &FeModel, facets: &[Facet], node_weights: &[(usize, f64)], nodes: &[usize], u: Option<&[NodeVec]>) -> Vec<f64> {
    let idx = |n: usize| nodes.binary_search(&n).unwrap();
    let mut area = vec![0.0; nodes.len()];
    for f in facets {
        let x = geom::coords(m, f, u);
        let axi = m.elements[f.elem].etype.family() == crate::elements::Family::Axisymmetric;
        for (xi, w) in geom::gauss(f.dim, x.len(), 3) {
            let fp = geom::eval(f, &x, xi);
            let wr = if axi { 2.0 * std::f64::consts::PI * fp.x[0] } else { 1.0 };
            for k in 0..x.len() {
                area[idx(f.nodes[k])] += fp.n[k] * w * fp.jac * wr;
            }
        }
    }
    for &(n, w) in node_weights {
        area[idx(n)] += if w > 0.0 { w } else { 1.0 };
    }
    area
}

pub struct AnalyticalContact {
    name: String,
    nodes: Vec<usize>,
    /// Slave node areas in the reference configuration.
    area: Vec<f64>,
    facets: Vec<Facet>,
    node_weights: Vec<(usize, f64)>,
    /// Hard contact with the direct method (force from the penalty, CPRESS = force/area).
    direct: bool,
    k_hard: Vec<f64>,
    /// The slave side's element stiffness per slave node (bound for the curvature term in the tangent).
    k_elem: Vec<f64>,
    surf: AnalyticalSurface,
    refn: usize,
    xr0: [f64; 3],
    props: PairProps,
    gamma: f64,
    dim: usize,
    /// SMALL SLIDING: the tangent plane (point, normal) in the body's system per slave node.
    anchors: Option<Vec<Option<([f64; 3], [f64; 3])>>>,
    state: Mutex<State>,
}

/// The reference node of the rigid body that owns the analytical surface `surf`.
fn rigid_ref(m: &FeModel, kw: &RawKeyword, surf: &str) -> Result<usize> {
    let up = surf.trim().to_uppercase();
    for rb in m.extra.iter().filter(|k| k.name == "RIGID BODY") {
        let Some(s) = rb.param("ANALYTICAL SURFACE") else { continue };
        if scoped(rb, s) != up {
            continue;
        }
        let r = crate::inp::required(rb, "REF NODE")?;
        let v = geom::node_ref(m, rb, &scoped(rb, r))?;
        if v.len() != 1 {
            return rb.err(format!("REF NODE={r} must be a single node"));
        }
        return Ok(v[0]);
    }
    kw.err(format!("the analytical surface {up} is not part of any *RIGID BODY, ANALYTICAL SURFACE="))
}

fn build(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let mut out: Vec<Box<dyn Interaction>> = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.len() < 2 || !is_analytical(m, f[1]) {
            continue;
        }
        if is_analytical(m, f[0]) {
            return kw.err_at(d.line, "an analytical surface can only be a master surface");
        }
        if kw.param_upper("TYPE").is_some_and(|t| t == "SURFACE TO SURFACE") {
            return kw.err_at(d.line, "TYPE=SURFACE TO SURFACE against an analytical surface is not supported yet (use NODE TO SURFACE)");
        }
        if m.extra.iter().any(|c| c.name == "CLEARANCE" && c.param_upper("MASTER").is_some_and(|x| x == f[1].to_uppercase())) {
            return kw.err_at(d.line, "*CLEARANCE against an analytical surface is not supported yet");
        }
        let ip = crate::inp::required(kw, "INTERACTION")?;
        let props = PairProps::read(m, kw, ip)?;
        let name = format!("*CONTACT PAIR {}/{} ({})", f[0], f[1], kw.location());
        out.push(Box::new(AnalyticalContact::new(m, kw, name, f[0], f[1], props, kw.has("SMALL SLIDING"))?));
    }
    Ok(out)
}

impl AnalyticalContact {
    fn new(m: &FeModel, kw: &RawKeyword, name: String, slave: &str, master: &str, props: PairProps, small: bool) -> Result<AnalyticalContact> {
        let surf = AnalyticalSurface::from_model(m, master)?;
        let refn = rigid_ref(m, kw, master)?;
        let ss = geom::surface(m, kw, slave)?;
        let dim = m.dim();
        let mut nodes: Vec<usize>;
        let mut facets: Vec<Facet> = Vec::new();
        if ss.kind == "NODE" {
            nodes = ss.nodes.iter().map(|p| p.0).collect();
        } else {
            facets = geom::facets(m, ss, &name)?;
            nodes = facets.iter().flat_map(|f| f.nodes.iter().copied()).collect();
        }
        nodes.sort_unstable();
        nodes.dedup();
        let node_weights: Vec<(usize, f64)> = if ss.kind == "NODE" { ss.nodes.clone() } else { Vec::new() };
        let area = slave_areas(m, &facets, &node_weights, &nodes, None);
        let sizes: Vec<f64> = facets.iter().map(|f| contact::facet_length(f, &geom::coords(m, f, None))).collect();
        // penalty coefficient per slave node
        let mut k_hard = Vec::with_capacity(nodes.len());
        let mut k_elem = Vec::with_capacity(nodes.len());
        for (i, &n) in nodes.iter().enumerate() {
            // the slave side's element stiffness at the node
            let mut s = 0.0;
            let mut c = 0usize;
            for (e, el) in m.elements.iter().enumerate() {
                if el.nodes.contains(&n) && !crate::elements::rigid::is_rigid(el.etype) {
                    s += contact::representative_stiffness(m, e)?;
                    c += 1;
                }
            }
            let ks = if c > 0 { s / c as f64 } else { 1.0 };
            k_elem.push(ks);
            let k = match props.penalty() {
                Some((Some(k), sc)) => k * sc,
                Some((None, sc)) => {
                    let v: Vec<f64> = facets.iter().filter(|f| f.nodes.contains(&n)).map(|f| contact::default_penalty(m, f)).collect::<Result<_>>()?;
                    sc * v.iter().sum::<f64>() / v.len().max(1) as f64
                }
                // direct: very stiff penalty coefficient from the slave side's elements
                None => HARD_FACTOR_ANALYTICAL * ks / area[i].max(1e-300),
            };
            k_hard.push(k);
        }
        let lc = if sizes.is_empty() { 1.0 } else { sizes.iter().sum::<f64>() / sizes.len() as f64 };
        let xr0 = m.nodes[refn];
        let anchors = small.then(|| nodes.iter().map(|&n| surf.closest(m.nodes[n]).map(|(q, nn, _)| (q, nn))).collect());
        let committed: Vec<NodeState> = nodes.iter().map(|&n| NodeState { pb: m.nodes[n], ..Default::default() }).collect();
        Ok(AnalyticalContact {
            name,
            area,
            facets,
            node_weights,
            direct: props.penalty().is_none() && props.is_hard(),
            k_hard,
            k_elem,
            surf,
            refn,
            xr0,
            props,
            gamma: props.slip_tol() * lc,
            dim,
            anchors,
            state: Mutex::new(State { trial: committed.clone(), committed, pmax: 0.0 }),
            nodes,
        })
    }

    fn rotation(&self, u: &[NodeVec]) -> [[f64; 3]; 3] {
        let ur = &u[self.refn];
        let th = if self.dim == 2 { [0.0, 0.0, ur[5]] } else { [ur[3], ur[4], ur[5]] };
        tensor::exp_so3(th)
    }

    fn ref_dofs(&self) -> Vec<Dof> {
        let mut v = geom::trans_dofs(self.dim).to_vec();
        v.extend(geom::rot_dofs(self.dim));
        v
    }
}

impl Interaction for AnalyticalContact {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        self.ref_dofs().into_iter().map(|d| (self.refn, d)).collect()
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.nodes.iter().map(|&n| vec![n, self.refn]).collect()
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], mut k: Option<&mut dyn TangentSink>) -> Result<()> {
        let m = ctx.model;
        let dim = self.dim;
        let rot = self.rotation(u);
        let rt = tensor::transpose(&rot);
        let ur = u[self.refn];
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        let mut pmax = 0.0f64;
        let area_cur = if self.anchors.is_none() { slave_areas(m, &self.facets, &self.node_weights, &self.nodes, Some(u)) } else { self.area.clone() };
        for (i, &sn) in self.nodes.iter().enumerate() {
            let old = st.committed[i];
            let prev_eval = st.trial[i];
            // the slave node in the body's reference configuration: p_b = X_s + d with
            // d = (Rᵀ − I)(X_s − X_R) + Rᵀ(u_s − u_R); the gap is computed as
            // n · (X_s − q) + n · d so that round-off in the large coordinates
            // does not cause noise in the penalty force between iterations
            let xs0 = m.nodes[sn];
            let r0 = sub(xs0, self.xr0);
            let w = sub([u[sn][0], u[sn][1], u[sn][2]], [ur[0], ur[1], ur[2]]);
            let rr0 = tensor::mv(&rt, r0);
            let rw = tensor::mv(&rt, w);
            let dpb = [rr0[0] - r0[0] + rw[0], rr0[1] - r0[1] + rw[1], rr0[2] - r0[2] + rw[2]];
            let pb = [xs0[0] + dpb[0], xs0[1] + dpb[1], xs0[2] + dpb[2]];
            let mut ns = NodeState { total_slip: old.total_slip, pb, ..Default::default() };
            let proj = match &self.anchors {
                Some(a) => a[i].map(|(q, n)| (q, n)),
                None => self.surf.closest(pb).map(|(q, n, _)| (q, n)),
            };
            let Some((qb, nb)) = proj else {
                st.trial[i] = ns;
                continue;
            };
            let g = dot(sub(xs0, qb), nb) + dot(dpb, nb);
            let n = tensor::mv(&rot, nb);
            let a = tensor::mv(&rot, sub(qb, self.xr0));
            ns.gap = g;
            ns.normal = n;
            if crate::settings::with(|s| s.contact_debug) {
                eprintln!("analytical contact {} node {}: g={g:.6e} n={n:?} a={a:?}", self.name, m.node_name(sn));
            }
            let h = -g;
            let kh = self.k_hard[i];
            // hard contact with contact status: a closed node is opened
            // only under tensile pressure (beyond a small tolerance), an open one closes on
            // overclosure; otherwise nodes at the edge of the contact zone alternate between
            // open (overclosure) and closed (tension) from iteration to iteration
            let (p, dp, closed) = if self.props.is_hard() {
                let was = st.trial[i].contact;
                let age = st.trial[i].age;
                // a node that has just closed is kept closed for one more iteration:
                // the gap linearization (curved surface, tangential motion) otherwise gives
                // a small error that, with the stiff penalty coefficient, looks like tension
                let closed = if was { age < 2 || kh * h >= -1e-8 * st.pmax } else { h > 0.0 };
                ns.age = if closed && was { age.saturating_add(1) } else { 0 };
                if closed != was && crate::settings::with(|s| s.contact_debug) {
                    eprintln!("  status {}: {} h={h:.4e} x={pb:?}", m.node_name(sn), if closed { "closed" } else { "open" });
                }
                if closed { (kh * h, kh, true) } else { (0.0, 0.0, false) }
            } else {
                let (p, dp) = self.props.pressure(h, kh);
                (p, dp, self.props.active(h))
            };
            pmax = pmax.max(p);
            ns.p = p;
            ns.contact = closed;
            // friction in the body's system
            let mut tau_b = [0.0; 3];
            let mut kt_b: Option<[[f64; 3]; 3]> = None;
            let dpb = sub(pb, old.pb);
            let dn = dot(dpb, nb);
            let ds = [dpb[0] - dn * nb[0], dpb[1] - dn * nb[1], dpb[2] - dn * nb[2]];
            let mu = self.props.mu();
            // friction only for nodes that were closed in the previous evaluation: a node
            // that has just closed (often with large overclosure from the extrapolation) would
            // otherwise get an unreasonable friction force μp
            let fric = mu > 0.0 && ns.contact && p > 0.0 && prev_eval.contact && prev_eval.age >= 1;
            if ns.contact && !fric {
                let d = tensor::mv(&rot, ds);
                for c in 0..3 {
                    ns.total_slip[c] += d[c];
                }
            }
            if fric {
                let eo = old.slip;
                let eon = dot(eo, nb);
                let e_tr = [eo[0] - eon * nb[0] + ds[0], eo[1] - eon * nb[1] + ds[1], eo[2] - eon * nb[2] + ds[2]];
                let e_len = norm(e_tr);
                let mu_p = mu * p;
                let gam = self.gamma.max(1e-300);
                let mut pt = [[0.0; 3]; 3];
                for r in 0..3 {
                    for c in 0..3 {
                        pt[r][c] = (if r == c { 1.0 } else { 0.0 }) - nb[r] * nb[c];
                    }
                }
                if dim == 2 {
                    pt[2] = [0.0; 3];
                    pt[0][2] = 0.0;
                    pt[1][2] = 0.0;
                }
                if e_len <= gam {
                    tau_b = scale(e_tr, mu_p / gam);
                    ns.slip = e_tr;
                    kt_b = Some(pt.map(|row| row.map(|v| v * mu_p / gam)));
                } else {
                    let eh = scale(e_tr, 1.0 / e_len);
                    tau_b = scale(eh, mu_p);
                    ns.slip = scale(eh, gam);
                    let f = mu_p / e_len;
                    let mut kk = [[0.0; 3]; 3];
                    for r in 0..3 {
                        for c in 0..3 {
                            kk[r][c] = f * (pt[r][c] - eh[r] * eh[c]);
                        }
                    }
                    // stuck in the previous evaluation (or slipping in the other
                    // direction): the sticking tangent. The slip tangent has no stiffness
                    // in the slip direction, so Newton otherwise alternates between slipping in
                    // both directions when the node actually sticks.
                    let prev = prev_eval;
                    let stuck_before = !prev.contact || norm(prev.slip) < gam * (1.0 - 1e-9) || dot(prev.slip, eh) < 0.0;
                    kt_b = Some(if stuck_before { pt.map(|row| row.map(|v| v * mu_p / gam)) } else { kk });
                }
                let dslip = sub(ds, sub(ns.slip, [eo[0] - eon * nb[0], eo[1] - eon * nb[1], eo[2] - eon * nb[2]]));
                let d = tensor::mv(&rot, dslip);
                for c in 0..3 {
                    ns.total_slip[c] = old.total_slip[c] + d[c];
                }
            }
            let tau = tensor::mv(&rot, tau_b);
            if crate::settings::with(|s| s.contact_debug) {
                eprintln!("  friction {}: p={p:.4e} ds={ds:?} tau={tau:?} uR={:?} us={:?}", m.node_name(sn), &ur[..3], &u[sn][..3]);
            }
            ns.shear = tau;
            st.trial[i] = ns;
            if !ns.contact && p == 0.0 {
                continue;
            }
            // the force: direct method from the penalty (k = f · k_e / A₀), otherwise p · A in
            // the current configuration; CPRESS = force / current area (finite sliding)
            let area = if self.direct { self.area[i] } else { area_cur[i] };
            st.trial[i].cpress = p * area / area_cur[i].max(1e-300);
            let tr = [-p * n[0] + tau[0], -p * n[1] + tau[1], -p * n[2] + tau[2]];
            let r = self.refn;
            for d in 0..dim {
                fint[sn][d] += area * tr[d];
                fint[r][d] -= area * tr[d];
            }
            let mom = cross(a, tr);
            if dim == 2 {
                fint[r][5] -= area * mom[2];
            } else {
                for d in 0..3 {
                    fint[r][3 + d] -= area * mom[d];
                }
            }
            let Some(k) = k.as_deref_mut() else { continue };
            // D = p' n nᵀ + R K_t Rᵀ (spatial)
            let mut dmat = [[0.0; 3]; 3];
            for r1 in 0..3 {
                for c in 0..3 {
                    dmat[r1][c] = dp * n[r1] * n[c];
                }
            }
            // surface curvature (finite sliding): δ(−p n) = −p κ' t tᵀ δx. Only for nodes that
            // were already closed in the previous evaluation (a node that has just closed can have
            // a large temporary penalty pressure, and the term is destabilizing for
            // convex surfaces)
            // the term is limited to a quarter of the element stiffness at the node (a
            // large temporary penalty pressure in one iteration otherwise makes the tangent indefinite)
            if self.anchors.is_none() && p > 0.0 && prev_eval.contact {
                for (kap, tb) in self.surf.curvatures(pb) {
                    let t = tensor::mv(&rot, tb);
                    let pg = p.min(0.25 * self.k_elem[i] / (area * kap.abs()).max(1e-300));
                    for r1 in 0..3 {
                        for c in 0..3 {
                            dmat[r1][c] -= pg * kap * t[r1] * t[c];
                        }
                    }
                }
            }
            if let Some(kb) = kt_b {
                let ks = tensor::mul(&tensor::mul(&rot, &kb), &rt);
                for r1 in 0..3 {
                    for c in 0..3 {
                        dmat[r1][c] += ks[r1][c];
                    }
                }
            }
            // G: relative displacement = δu_s − δu_R + a × δw
            let mut cols: Vec<(usize, Dof, [f64; 3])> = Vec::new();
            for d in 0..dim {
                let mut e = [0.0; 3];
                e[d] = 1.0;
                cols.push((sn, U1 + d as Dof, e));
                cols.push((r, U1 + d as Dof, scale(e, -1.0)));
            }
            if dim == 2 {
                cols.push((r, UR3, cross(a, [0.0, 0.0, 1.0])));
            } else {
                for j in 0..3 {
                    let mut e = [0.0; 3];
                    e[j] = 1.0;
                    cols.push((r, UR1 + j as Dof, cross(a, e)));
                }
            }
            for (na, da, ga) in &cols {
                for (nb2, db, gb) in &cols {
                    let mut v = 0.0;
                    for r1 in 0..3 {
                        for c in 0..3 {
                            v += ga[r1] * dmat[r1][c] * gb[c];
                        }
                    }
                    let v = area * v;
                    if v != 0.0 {
                        k.add(*na, *da, *nb2, *db, v);
                    }
                }
            }
        }
        st.pmax = pmax;
        Ok(())
    }

    fn commit(&self) {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        st.committed.clone_from(&st.trial);
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let st = self.state.lock().unwrap();
        let mut cp = NodeOutput { name: "CPRESS".into(), labels: vec!["CPRESS".into()], values: Vec::new() };
        let mut co = NodeOutput { name: "COPEN".into(), labels: vec!["COPEN".into()], values: Vec::new() };
        let ncomp = self.dim - 1;
        let mut sh: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSHEAR{k}"), labels: vec![format!("CSHEAR{k}")], values: Vec::new() }).collect();
        let mut sl: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSLIP{k}"), labels: vec![format!("CSLIP{k}")], values: Vec::new() }).collect();
        for (i, &n) in self.nodes.iter().enumerate() {
            let s = st.committed[i];
            cp.values.push((n, vec![s.cpress.max(0.0)]));
            co.values.push((n, vec![s.gap]));
            let t = contact::output_tangents(s.normal, self.dim);
            for k in 0..ncomp {
                sh[k].values.push((n, vec![dot(s.shear, t[k])]));
                sl[k].values.push((n, vec![dot(s.total_slip, t[k])]));
            }
        }
        let mut out = vec![cp, co];
        out.extend(sh);
        out.extend(sl);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_normal_and_arc() {
        // line from (1, 0) to (−1, 0): the normal = the direction rotated 90° counterclockwise = (0, −1)
        let p = vec![Seg::Line { a: [1.0, 0.0], b: [-1.0, 0.0] }];
        let (_, n, g) = profile_closest(&p, [0.2, 0.5]).unwrap();
        assert!((n[1] + 1.0).abs() < 1e-14 && (g + 0.5).abs() < 1e-14);
        // quarter circle counterclockwise around the origin with radius 2: the normal towards the center
        let a = Seg::Arc { c: [0.0, 0.0], r: 2.0, t0: 0.0, dt: std::f64::consts::FRAC_PI_2 };
        let (q, n, g) = profile_closest(&[a], [3.0, 3.0]).unwrap();
        let s = 2.0 / 2f64.sqrt();
        assert!((q[0] - s).abs() < 1e-12 && (q[1] - s).abs() < 1e-12);
        assert!((n[0] + 1.0 / 2f64.sqrt()).abs() < 1e-12 && (g + (18f64.sqrt() - 2.0)).abs() < 1e-12);
    }
}
