//! Extensions of the contact in implicit analysis (included with `#[path]` from
//! `contact.rs`):
//!
//! - *FRICTION, ANISOTROPIC (3D): the friction coefficients μ₁, μ₂ in the local
//!   tangent directions t₁, t₂ (the same as for CSHEAR1/2). Elastic slip
//!   with the stiffness μᵢ p/γ in direction i; the slip limit is the ellipse
//!   (τ₁/μ₁)² + (τ₂/μ₂)² = p², and when slipping τᵢ = μᵢ p êᵢ with ê the
//!   direction of the elastic slip predictor (radial return in the normalized
//!   stresses τᵢ/μᵢ, so that isotropic friction is the special case
//!   μ₁ = μ₂).

use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

/// Self-contact for the surface `name` (*CONTACT INCLUSIONS with one surface or
/// ALL EXTERIOR for a body that can reach itself): the surface is both slave and
/// master, with weight ½ since each contact location is counted from both
/// sides (balanced master–slave as in general contact). Master facets whose
/// element contains the slave point's nodes are skipped, so that a node does
/// not come into contact with its own elements.
pub(super) fn self_pair(m: &FeModel, kw: &RawKeyword, name: &str, per_pair: &[(String, String, String)], p: super::Props) -> Result<super::ContactPair> {
    let up = name.to_uppercase();
    let props = match per_pair.iter().find(|(a, b, _)| *a == up && *b == up) {
        Some((_, _, n)) => super::props(m, kw, n)?,
        None => p,
    };
    let label = format!("*CONTACT {name} (self-contact, {})", kw.location());
    let mut cp = super::ContactPair::new(m, kw, label, name, name, props, true, false, 0.5)?;
    cp.make_self(m);
    Ok(cp)
}

/// The body (its exterior surfaces) is not convex in the reference
/// configuration, so it can come into contact with itself: some node lies in
/// front of some facet's plane.
pub(super) fn can_self_contact(m: &FeModel, faces: &[(usize, String)]) -> bool {
    let fs: Vec<super::Facet> = faces
        .iter()
        .filter_map(|(e, f)| {
            let el = &m.elements[*e];
            let k: usize = f.trim_start_matches('S').parse().ok()?;
            let face = el.etype.faces().get(k.checked_sub(1)?)?;
            let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            Some(super::Facet { elem: *e, nodes: face.iter().map(|&q| el.nodes[q]).collect(), inside: crate::elements::common::centroid(&xs), dim: el.etype.dim(), thick: 1.0 })
        })
        .collect();
    let mut nodes: Vec<usize> = fs.iter().flat_map(|f| f.nodes.iter().copied()).collect();
    nodes.sort_unstable();
    nodes.dedup();
    for f in &fs {
        let x = super::geom::coords(m, f, None);
        let fp = super::geom::eval(f, &x, [0.0, 0.0]);
        let size = super::facet_size(f, &x);
        for &n in &nodes {
            let d = sub(m.nodes[n], fp.x);
            if dot(d, fp.normal) > 1e-6 * size {
                return true;
            }
        }
    }
    false
}

/// Adjustment of slave nodes at the start of the analysis (strain-free):
///
/// - *CONTACT PAIR, ADJUST=a: slave nodes with gap g ≤ a (also overlapping)
///   are moved onto the master surface along its normal; ADJUST=node set: the
///   node set's slave nodes are moved onto the master surface.
/// - *CONTACT INITIALIZATION DATA (via *CONTACT INITIALIZATION ASSIGNMENT in
///   *CONTACT): slave nodes with −SEARCH BELOW ≤ g ≤ SEARCH ABOVE are moved so
///   that the gap becomes INITIAL CLEARANCE (default 0; without SEARCH ABOVE only
///   overlapping nodes). The surface pair is interpreted as for general contact
///   (first surface is slave). INITIAL CLEARANCE usually does not move the
///   nodes; here they are moved to the given gap (same contact gap, slightly
///   changed geometry).
///
/// The master facet is the closest master facet in the reference configuration
/// (projection within the facet); the new node positions become reference
/// positions (U is measured from there).
pub(super) fn adjust(m: &mut FeModel) -> Result<()> {
    let mut moves: Vec<(usize, [f64; 3])> = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT PAIR") {
        let Some(a) = kw.param("ADJUST") else { continue };
        let a = a.trim();
        let (limit, set): (Option<f64>, Option<Vec<usize>>) = match crate::fem::model::parse_f64(a) {
            Some(v) => (Some(v), None),
            None if a.eq_ignore_ascii_case("NO") => continue,
            None => (None, Some(super::geom::node_ref(m, kw, a)?)),
        };
        if limit == Some(0.0) {
            continue;
        }
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if f.len() < 2 || super::super::analytical::is_analytical(m, f[1]) {
                continue;
            }
            for (n, g, nm) in gaps(m, kw, f[0], f[1])? {
                let take = match (&set, limit) {
                    (Some(s), _) => s.contains(&n),
                    (None, Some(l)) => g <= l,
                    _ => false,
                };
                if take {
                    moves.push((n, [-g * nm[0], -g * nm[1], -g * nm[2]]));
                }
            }
        }
    }
    // general contact: *CONTACT INITIALIZATION ASSIGNMENT → *CONTACT INITIALIZATION DATA
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT INITIALIZATION ASSIGNMENT") {
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).collect();
            if f.len() < 3 || f[0].is_empty() || f[1].is_empty() || f[2].is_empty() {
                return kw.err_at(d.line, "write 'surface, surface, initialization name' (two surfaces are required)");
            }
            let up = f[2].to_uppercase();
            let Some(init) = m.extra.iter().find(|k| k.name == "CONTACT INITIALIZATION DATA" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
                return kw.err_at(d.line, format!("*CONTACT INITIALIZATION DATA, NAME={} does not exist", f[2]));
            };
            if init.has("INTERFERENCE FIT") {
                return init.err("INTERFERENCE FIT is not supported yet");
            }
            let num = |k: &str| -> Result<Option<f64>> {
                match init.param(k) {
                    Some(v) => crate::fem::model::parse_f64(v).map(Some).ok_or_else(|| crate::model::FemError(format!("{}: {k}= must be a number", init.location()))),
                    None => Ok(None),
                }
            };
            let c = num("INITIAL CLEARANCE")?.unwrap_or(0.0);
            let above = num("SEARCH ABOVE")?.unwrap_or(0.0).max(c);
            let below = num("SEARCH BELOW")?;
            for (n, g, nm) in gaps(m, kw, f[0], f[1])? {
                let lo = below.map_or(f64::NEG_INFINITY, |b| -b);
                if g >= lo && (g <= above || g < c) && g != c {
                    let s = c - g;
                    moves.push((n, [s * nm[0], s * nm[1], s * nm[2]]));
                }
            }
        }
    }
    for (n, d) in moves {
        for k in 0..3 {
            m.nodes[n][k] += d[k];
        }
    }
    Ok(())
}

/// (slave node, gap, master normal) for the slave surface nodes against the
/// closest master facet (projection within the facet) in the reference configuration.
fn gaps(m: &FeModel, kw: &RawKeyword, slave: &str, master: &str) -> Result<Vec<(usize, f64, [f64; 3])>> {
    let ss = super::geom::surface(m, kw, slave)?;
    let ms = super::geom::surface(m, kw, master)?;
    let nodes = super::geom::surface_nodes(m, ss)?;
    let mf = super::geom::facets(m, ms, "master surface")?;
    let mut out = Vec::new();
    for n in nodes {
        let p = m.nodes[n];
        let mut best: Option<(f64, f64, [f64; 3])> = None;
        for f in &mf {
            if f.nodes.contains(&n) {
                continue;
            }
            let x = super::geom::coords(m, f, None);
            let (xi, dist, inside) = super::geom::project(f, &x, p);
            if !inside {
                continue;
            }
            let fp = super::geom::eval(f, &x, xi);
            if best.is_none_or(|b| dist < b.0) {
                best = Some((dist, dot(sub(p, fp.x), fp.normal), fp.normal));
            }
        }
        if let Some((_, g, nm)) = best {
            out.push((n, g, nm));
        }
    }
    Ok(out)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// μ₁, μ₂ from the data line of *FRICTION, ANISOTROPIC.
pub(super) fn anisotropic(sub: &RawKeyword) -> Result<(f64, f64)> {
    if sub.data.len() > 1 {
        return sub.err("tabular anisotropic friction is not supported yet (one line with μ₁, μ₂)");
    }
    let v = sub.first_floats()?;
    match v.as_slice() {
        [m1, m2, ..] if *m1 >= 0.0 && *m2 >= 0.0 => Ok((*m1, *m2)),
        [m1] => Ok((*m1, *m1)),
        _ => sub.err("*FRICTION, ANISOTROPIC requires μ₁, μ₂"),
    }
}

/// The friction response for a slave node.
pub(super) struct Aniso {
    pub tau: [f64; 3],
    pub dtau_dp: [f64; 3],
    pub slip: [f64; 3],
    /// ∂τ/∂e (unsymmetric when μ₁ ≠ μ₂; the symmetric part is used in symmetric solvers).
    pub kt: [[f64; 3]; 3],
}

/// Anisotropic friction for the elastic slip predictor `e_tr` (in the tangent
/// plane), the normal `n`, the pressure `p` and the critical elastic slip `gam`.
pub(super) fn aniso_friction(e_tr: [f64; 3], n: [f64; 3], p: f64, mu1: f64, mu2: f64, gam: f64) -> Aniso {
    let t = super::local_tangents(n, 3);
    let mu = [mu1, mu2];
    let e = [dot(e_tr, t[0]), dot(e_tr, t[1])];
    let e_len = (e[0] * e[0] + e[1] * e[1]).sqrt();
    let mut out = Aniso { tau: [0.0; 3], dtau_dp: [0.0; 3], slip: e_tr, kt: [[0.0; 3]; 3] };
    // dτ_i/de_j in local directions
    let mut d = [[0.0; 2]; 2];
    let mut tl = [0.0; 2];
    let mut dpl = [0.0; 2];
    if e_len <= gam {
        for i in 0..2 {
            tl[i] = mu[i] * p * e[i] / gam;
            dpl[i] = mu[i] * e[i] / gam;
            d[i][i] = mu[i] * p / gam;
        }
    } else {
        let eh = [e[0] / e_len, e[1] / e_len];
        for i in 0..2 {
            tl[i] = mu[i] * p * eh[i];
            dpl[i] = mu[i] * eh[i];
            for j in 0..2 {
                d[i][j] = mu[i] * p * ((if i == j { 1.0 } else { 0.0 }) - eh[i] * eh[j]) / e_len;
            }
        }
        out.slip = std::array::from_fn(|c| gam * (eh[0] * t[0][c] + eh[1] * t[1][c]));
    }
    for c in 0..3 {
        out.tau[c] = tl[0] * t[0][c] + tl[1] * t[1][c];
        out.dtau_dp[c] = dpl[0] * t[0][c] + dpl[1] * t[1][c];
    }
    for a in 0..3 {
        for b in 0..3 {
            let mut v = 0.0;
            for i in 0..2 {
                for j in 0..2 {
                    v += t[i][a] * d[i][j] * t[j][b];
                }
            }
            out.kt[a][b] = v;
        }
    }
    out
}

/// Symmetric and skew-symmetric parts of a 3×3 matrix.
pub(super) fn split(k: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let s = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (k[a][b] + k[b][a])));
    let w = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (k[a][b] - k[b][a])));
    (s, w)
}

/// The skew-symmetric part of the friction tangent (unsymmetric solver):
/// K[B₁, B₂] += A c₁ c₂ W.
pub(super) fn add_skew(k: &mut dyn super::TangentSink, coef: &[(usize, f64)], area: f64, w: &[[f64; 3]; 3], dim: usize) {
    for &(b1, c1) in coef {
        for &(b2, c2) in coef {
            let f = area * c1 * c2;
            for p1 in 0..dim {
                for p2 in 0..dim {
                    let v = f * w[p1][p2];
                    if v != 0.0 {
                        k.add_unsym(b1, crate::fem::dof::U1 + p1 as crate::fem::dof::Dof, b2, crate::fem::dof::U1 + p2 as crate::fem::dof::Dof, v);
                    }
                }
            }
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
