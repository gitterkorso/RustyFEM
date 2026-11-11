//! Data for the Visualization tab: the model split into parts (one per section),
//! beam members (chains of beam and truss elements), solid bodies (connected
//! solid or planar continuum elements) and supports with their reactions.
//!
//! Section forces:
//! - members: from the elements' own output (SF/SM, trusses S11·A), as global force and
//!   moment vectors acting on the positive section (outward normal along the member);
//! - solid and planar bodies: the stresses integrated over a plane cut. The nodal
//!   stresses are recovered from the element means by a linear least-squares fit over
//!   the two element rings around each node (exact for a linear stress field), then
//!   interpolated linearly along the element edges onto the cut polygon, which is
//!   integrated exactly for that piecewise linear field.

use crate::mesh::{shape, Shape};
use crate::profiles;
use rustyfem::elements::Family;
use rustyfem::fem::model::{FeModel, SectionKind};
use rustyfem::fem::output::FrameData;
use serde::Serialize;
use std::collections::HashMap;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> Option<V3> {
    let l = norm(a);
    (l > 0.0 && l.is_finite()).then(|| scale(a, 1.0 / l))
}
fn f32v(a: V3) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}

// ---------------------------------------------------------------- the model

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// Element set of the section (or the element type for elements without a section).
    pub name: String,
    /// "beam", "truss", "shell", "membrane", "solid", "plane", "other".
    pub kind: String,
    /// Beam section shape and dimensions (*BEAM SECTION order), if known.
    pub shape: Option<String>,
    pub dims: Vec<f64>,
    /// Beam/truss cross-section area, shell thickness or plane thickness.
    pub area: Option<f64>,
    pub thickness: Option<f64>,
    pub material: String,
    /// Mass (density × volume, in the model's units; 0 if the density is missing).
    pub mass: f64,
    pub elements: u32,
    /// Index into `VisualModel::profiles` (beams and trusses), −1 = none.
    pub profile: i32,
}

/// A beam profile: an outline (with an optional hole), or convex pieces when the
/// shape has no single outline. Coordinates (x1, x2) along n1 and n2.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Profile {
    pub outer: Vec<[f64; 2]>,
    pub hole: Option<Vec<[f64; 2]>>,
    pub pieces: Vec<Vec<[f64; 2]>>,
    /// Round: the sides are shaded smoothly.
    pub smooth: bool,
}

/// A chain of line elements of one part without junctions or kinks.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Member {
    pub part: u32,
    /// Path nodes in order (end and midside nodes) and their distance along the member.
    pub nodes: Vec<u32>,
    pub s: Vec<f64>,
    /// Section axes at the path nodes (n1, n2 of the elements, perpendicular to the path).
    pub n1: Vec<[f32; 3]>,
    pub n2: Vec<[f32; 3]>,
    pub elements: Vec<u32>,
    /// +1 if the element runs along the member, −1 if against it.
    pub dirs: Vec<i8>,
    pub length: f64,
}

/// Connected solid (3D) or planar (2D) continuum elements.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Body {
    pub dim: u32,
    pub elements: u32,
    pub center: V3,
    /// Main axis (longest direction, snapped to a global axis when close) and the range
    /// of the nodes along it, measured from `center`.
    pub axis: V3,
    pub s0: f64,
    pub s1: f64,
    /// Smallest and largest coordinates of the nodes.
    pub lo: V3,
    pub hi: V3,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualModel {
    pub parts: Vec<Part>,
    pub profiles: Vec<Profile>,
    pub members: Vec<Member>,
    pub bodies: Vec<Body>,
    /// Per element: part (−1 = not shown), member and body (−1 = none).
    pub elem_part: Vec<i32>,
    pub elem_member: Vec<i32>,
    pub elem_body: Vec<i32>,
}

fn kind_of(f: Family) -> &'static str {
    match f {
        Family::Beam => "beam",
        Family::Truss => "truss",
        Family::Shell => "shell",
        Family::Membrane => "membrane",
        Family::Continuum3D => "solid",
        Family::Continuum2D => "plane",
        Family::Axisymmetric => "axisymmetric",
        _ => "other",
    }
}

/// Corner nodes of a continuum, shell or membrane element (no midside nodes).
fn corners(model: &FeModel, e: usize) -> (Shape, Vec<usize>) {
    let el = &model.elements[e];
    let (sh, _) = shape(el.etype.name(), el.nodes.len());
    let n = match sh {
        Shape::Tet => 4,
        Shape::Wedge => 6,
        Shape::Hex => 8,
        Shape::Face => match el.nodes.len() {
            3 | 6 => 3,
            _ => 4,
        },
        _ => el.nodes.len(),
    };
    (sh, el.nodes.iter().take(n).copied().collect())
}

fn tet_volume(a: V3, b: V3, c: V3, d: V3) -> f64 {
    dot(sub(b, a), cross(sub(c, a), sub(d, a))).abs() / 6.0
}

/// Volume of a solid or area of a face from its corners.
fn measure(model: &FeModel, e: usize) -> f64 {
    let (sh, c) = corners(model, e);
    let x: Vec<V3> = c.iter().map(|&n| model.nodes[n]).collect();
    let tets: &[[usize; 4]] = match sh {
        Shape::Tet => &[[0, 1, 2, 3]],
        Shape::Wedge => &[[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
        Shape::Hex => &[[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]],
        Shape::Face => {
            let mut a = 0.0;
            for k in 1..x.len().saturating_sub(1) {
                a += norm(cross(sub(x[k], x[0]), sub(x[k + 1], x[0]))) / 2.0;
            }
            return a;
        }
        _ => return 0.0,
    };
    tets.iter().map(|t| tet_volume(x[t[0]], x[t[1]], x[t[2]], x[t[3]])).sum()
}

/// Outline of a section shape (same geometry as `profiles::shape_polys`).
fn outline(shape: &str, v: &[f64]) -> Option<(Vec<[f64; 2]>, Option<Vec<[f64; 2]>>, bool)> {
    let ngon = |r: f64, n: usize, a0: f64| -> Vec<[f64; 2]> {
        (0..n).map(|k| {
            let a = a0 + 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            [r * a.cos(), r * a.sin()]
        }).collect()
    };
    let rect = |x0: f64, x1: f64, y0: f64, y1: f64| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
    let need = |n: usize| (v.len() >= n).then_some(());
    Some(match shape {
        "RECT" => {
            need(2)?;
            (rect(-v[0] / 2.0, v[0] / 2.0, -v[1] / 2.0, v[1] / 2.0), None, false)
        }
        "CIRC" => {
            need(1)?;
            (ngon(v[0], 48, 0.0), None, true)
        }
        "PIPE" | "THICK PIPE" => {
            need(2)?;
            let ri = v[0] - v[1];
            (ngon(v[0], 48, 0.0), (ri > 0.0).then(|| ngon(ri, 48, 0.0)), true)
        }
        "BOX" => {
            need(6)?;
            let (a, b, t1, t2, t3, t4) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let (xl, xr, yb, yt) = (-a / 2.0, a / 2.0, -b / 2.0, b / 2.0);
            let hole = (xl + t3 < xr - t1 && yb + t4 < yt - t2).then(|| rect(xl + t3, xr - t1, yb + t4, yt - t2));
            (rect(xl, xr, yb, yt), hole, false)
        }
        "I" => {
            need(7)?;
            let (l, h, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            let (yb, yt, w) = (-l, -l + h, t3 / 2.0);
            (
                vec![
                    [-b1 / 2.0, yb],
                    [b1 / 2.0, yb],
                    [b1 / 2.0, yb + t1],
                    [w, yb + t1],
                    [w, yt - t2],
                    [b2 / 2.0, yt - t2],
                    [b2 / 2.0, yt],
                    [-b2 / 2.0, yt],
                    [-b2 / 2.0, yt - t2],
                    [-w, yt - t2],
                    [-w, yb + t1],
                    [-b1 / 2.0, yb + t1],
                ],
                None,
                false,
            )
        }
        "L" => {
            need(4)?;
            let (a, b, t1, t2) = (v[0], v[1], v[2], v[3]);
            (
                vec![[-t2 / 2.0, -t1 / 2.0], [a - t2 / 2.0, -t1 / 2.0], [a - t2 / 2.0, t1 / 2.0], [t2 / 2.0, t1 / 2.0], [t2 / 2.0, b - t1 / 2.0], [-t2 / 2.0, b - t1 / 2.0]],
                None,
                false,
            )
        }
        "HEX" => {
            need(2)?;
            let a0 = std::f64::consts::PI / 6.0;
            let inner = v[0] - 2.0 * v[1] / 3f64.sqrt();
            (ngon(v[0], 6, a0), (inner > 0.0).then(|| ngon(inner, 6, a0)), false)
        }
        "TRAPEZOID" => {
            need(4)?;
            let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
            (vec![[-a / 2.0, -d], [a / 2.0, -d], [c / 2.0, b - d], [-c / 2.0, b - d]], None, false)
        }
        "CHANNEL" => {
            need(8)?;
            let (l, h, b1, b2, t1, t2, t3, o) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            (
                vec![[-o, -l], [-o + b1, -l], [-o + b1, -l + t1], [-o + t3, -l + t1], [-o + t3, -l + h - t2], [-o + b2, -l + h - t2], [-o + b2, -l + h], [-o, -l + h]],
                None,
                false,
            )
        }
        _ => return None,
    })
}

fn profile_of(model: &FeModel, si: usize, fam: Family) -> Option<Profile> {
    let sec = &model.sections[si];
    if let Some((sh, dims)) = profiles::section_shape(sec) {
        if let Some((outer, hole, smooth)) = outline(&sh, &dims) {
            return Some(Profile { outer, hole, pieces: Vec::new(), smooth });
        }
    }
    let pieces = profiles::section_polys(sec, fam)?;
    // a truss is drawn as the circle with its area
    if matches!(fam, Family::Truss) && pieces.len() == 1 {
        if let SectionKind::Solid { thickness: Some(a) } = sec.kind {
            let r = (a / std::f64::consts::PI).sqrt();
            let c = (0..32).map(|k| {
                let t = 2.0 * std::f64::consts::PI * k as f64 / 32.0;
                [r * t.cos(), r * t.sin()]
            });
            return Some(Profile { outer: c.collect(), hole: None, pieces: Vec::new(), smooth: true });
        }
    }
    if pieces.len() == 1 {
        return Some(Profile { outer: pieces[0].clone(), hole: None, pieces: Vec::new(), smooth: false });
    }
    Some(Profile { outer: Vec::new(), hole: None, pieces, smooth: false })
}

fn density_of(model: &FeModel, si: Option<usize>) -> f64 {
    let Some(sec) = si.map(|s| &model.sections[s]) else { return 0.0 };
    if let Some(b) = sec.beam() {
        if b.density > 0.0 {
            return b.density;
        }
    }
    sec.material_index.map_or(0.0, |i| model.materials[i].density)
}

struct UnionFind(Vec<usize>);
impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind((0..n).collect())
    }
    fn find(&mut self, a: usize) -> usize {
        let mut r = a;
        while self.0[r] != r {
            r = self.0[r];
        }
        let mut a = a;
        while self.0[a] != r {
            let n = self.0[a];
            self.0[a] = r;
            a = n;
        }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.0[a] = b;
        }
    }
}

/// Snaps a direction to the nearest global axis when it is within about 15° of it,
/// and points it along the positive direction of its largest component.
fn tidy_axis(a: V3) -> V3 {
    let k = (0..3).max_by(|&i, &j| a[i].abs().total_cmp(&a[j].abs())).unwrap_or(0);
    let mut a = if a[k] < 0.0 { scale(a, -1.0) } else { a };
    if a[k].abs() > 0.966 {
        a = [0.0; 3];
        a[k] = 1.0;
    }
    a
}

/// Eigen decomposition of a symmetric 3×3 matrix (Jacobi): values and vectors (columns).
fn eig3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off <= 1e-300 || off <= 1e-15 * (a[0][0].abs() + a[1][1].abs() + a[2][2].abs()) {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let th = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = th.signum() / (th.abs() + (th * th + 1.0).sqrt());
            let t = if th == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Chains of line elements within each part, split at junctions, free ends and kinks.
fn members(model: &FeModel, elem_part: &[i32], parts: &[Part]) -> Vec<Member> {
    let mut out = Vec::new();
    let mut by_part: HashMap<i32, Vec<usize>> = HashMap::new();
    for (e, &p) in elem_part.iter().enumerate() {
        if p >= 0 && matches!(parts[p as usize].kind.as_str(), "beam" | "truss") && model.elements[e].nodes.len() >= 2 {
            by_part.entry(p).or_default().push(e);
        }
    }
    let mut keys: Vec<i32> = by_part.keys().copied().collect();
    keys.sort();
    let ends = |e: usize| {
        let n = &model.elements[e].nodes;
        (n[0], n[n.len() - 1])
    };
    let dir_of = |e: usize| {
        let (a, b) = ends(e);
        unit(sub(model.nodes[b], model.nodes[a])).unwrap_or([1.0, 0.0, 0.0])
    };
    for p in keys {
        let els = &by_part[&p];
        let mut at: HashMap<usize, Vec<usize>> = HashMap::new();
        for &e in els {
            let (a, b) = ends(e);
            at.entry(a).or_default().push(e);
            at.entry(b).or_default().push(e);
        }
        // a node breaks the chain unless exactly two elements meet there in nearly a straight line
        let breaks = |n: usize| -> bool {
            let l = &at[&n];
            if l.len() != 2 {
                return true;
            }
            let (d0, d1) = (dir_of(l[0]), dir_of(l[1]));
            dot(d0, d1).abs() < 0.94 // about 20°
        };
        let mut used = std::collections::HashSet::new();
        // start at break nodes first, then any remaining (closed loops)
        let mut starts: Vec<(usize, usize)> = Vec::new();
        for &e in els {
            let (a, b) = ends(e);
            if breaks(a) {
                starts.push((e, a));
            }
            if breaks(b) {
                starts.push((e, b));
            }
        }
        for &e in els {
            starts.push((e, ends(e).0));
        }
        for (e0, n0) in starts {
            if used.contains(&e0) {
                continue;
            }
            let mut chain: Vec<(usize, bool)> = Vec::new(); // (element, runs forward a→b)
            let (mut e, mut n) = (e0, n0);
            loop {
                used.insert(e);
                let (a, b) = ends(e);
                let fwd = a == n;
                chain.push((e, fwd));
                n = if fwd { b } else { a };
                if breaks(n) {
                    break;
                }
                match at[&n].iter().find(|&&x| !used.contains(&x)) {
                    Some(&x) => e = x,
                    None => break,
                }
            }
            out.push(build_member(model, p as u32, &chain));
        }
    }
    out
}

fn build_member(model: &FeModel, part: u32, chain: &[(usize, bool)]) -> Member {
    let mut chain = chain.to_vec();
    // path nodes in order
    let path = |chain: &[(usize, bool)]| -> Vec<usize> {
        let mut nodes: Vec<usize> = Vec::new();
        for &(e, fwd) in chain {
            let mut n = model.elements[e].nodes.clone();
            if !fwd {
                n.reverse();
            }
            if nodes.is_empty() {
                nodes.extend(n);
            } else {
                nodes.extend(n.into_iter().skip(1));
            }
        }
        nodes
    };
    let mut nodes = path(&chain);
    // the member runs along the positive direction of its largest extent
    let d = sub(model.nodes[*nodes.last().unwrap()], model.nodes[nodes[0]]);
    let k = (0..3).max_by(|&i, &j| d[i].abs().total_cmp(&d[j].abs())).unwrap();
    if d[k] < 0.0 {
        chain.reverse();
        for c in chain.iter_mut() {
            c.1 = !c.1;
        }
        nodes = path(&chain);
    }
    let mut s = vec![0.0];
    for w in nodes.windows(2) {
        s.push(s.last().unwrap() + norm(sub(model.nodes[w[1]], model.nodes[w[0]])));
    }
    // element axes (n1 as the elements compute it, n2 = t × n1)
    let axes: Vec<(V3, V3)> = chain
        .iter()
        .map(|&(e, _)| {
            let el = &model.elements[e];
            let (a, b) = (el.nodes[0], el.nodes[el.nodes.len() - 1]);
            let t = unit(sub(model.nodes[b], model.nodes[a])).unwrap_or([1.0, 0.0, 0.0]);
            let n1a = match el.section.and_then(|si| model.sections[si].beam()) {
                Some(bs) if el.etype.dim() == 3 => bs.n1.unwrap_or([0.0, 0.0, -1.0]),
                _ => [0.0, 0.0, -1.0],
            };
            let n1 = profiles::segment_n1(t, n1a);
            (n1, cross(t, n1))
        })
        .collect();
    // per path node: the axes of the elements it belongs to, made perpendicular to the path
    let mut n1s = Vec::with_capacity(nodes.len());
    let mut n2s = Vec::with_capacity(nodes.len());
    let mut k = 0usize; // path index
    let mut owner = vec![Vec::new(); nodes.len()];
    for (i, &(e, _)) in chain.iter().enumerate() {
        let m = model.elements[e].nodes.len();
        for j in 0..m {
            owner[k + j].push(i);
        }
        k += m - 1;
    }
    for (p, o) in owner.iter().enumerate() {
        let a = nodes[p.saturating_sub(1)];
        let b = nodes[(p + 1).min(nodes.len() - 1)];
        let t = unit(sub(model.nodes[b], model.nodes[a])).unwrap_or([1.0, 0.0, 0.0]);
        let (mut n1, mut n2) = ([0.0; 3], [0.0; 3]);
        for &i in o {
            n1 = add(n1, axes[i].0);
            n2 = add(n2, axes[i].1);
        }
        let n1p = unit(sub(n1, scale(t, dot(n1, t)))).unwrap_or_else(|| profiles::segment_n1(t, [0.0, 0.0, -1.0]));
        let mut n2p = cross(t, n1p);
        if dot(n2p, n2) < 0.0 {
            n2p = scale(n2p, -1.0);
        }
        n1s.push(f32v(n1p));
        n2s.push(f32v(n2p));
    }
    Member {
        part,
        length: *s.last().unwrap(),
        nodes: nodes.iter().map(|&n| n as u32).collect(),
        s,
        n1: n1s,
        n2: n2s,
        elements: chain.iter().map(|c| c.0 as u32).collect(),
        dirs: chain.iter().map(|c| if c.1 { 1 } else { -1 }).collect(),
    }
}

fn pretty(name: &str) -> String {
    let n = name.rsplit('.').next().unwrap_or(name).replace('_', " ").trim().to_string();
    let lower = n.to_lowercase();
    let mut c = lower.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => name.to_string(),
    }
}

pub fn model(model: &FeModel) -> VisualModel {
    let ne = model.elements.len();
    let mut parts: Vec<Part> = Vec::new();
    let mut profiles_out: Vec<Profile> = Vec::new();
    let mut key_part: HashMap<(Option<usize>, &'static str), usize> = HashMap::new();
    let mut elem_part = vec![-1i32; ne];
    for (e, el) in model.elements.iter().enumerate() {
        let fam = el.etype.family();
        let kind = kind_of(fam);
        if matches!(fam, Family::Heat | Family::Connector) {
            continue;
        }
        let key = (el.section, kind);
        let p = *key_part.entry(key).or_insert_with(|| {
            let (name, material) = match el.section {
                Some(si) => {
                    let s = &model.sections[si];
                    (pretty(&s.elset), s.material.clone())
                }
                None => (el.etype.name().to_string(), String::new()),
            };
            let mut part = Part { name, kind: kind.into(), shape: None, dims: Vec::new(), area: None, thickness: None, material, mass: 0.0, elements: 0, profile: -1 };
            if let Some(si) = el.section {
                let sec = &model.sections[si];
                match fam {
                    Family::Beam | Family::Truss => {
                        if let Some((sh, d)) = profiles::section_shape(sec) {
                            part.shape = Some(sh);
                            part.dims = d;
                        }
                        part.area = match &sec.kind {
                            SectionKind::Beam(b) => Some(b.profile.a),
                            SectionKind::Solid { thickness: Some(a) } => Some(*a),
                            _ => None,
                        };
                        if let Some(pr) = profile_of(model, si, fam) {
                            profiles_out.push(pr);
                            part.profile = profiles_out.len() as i32 - 1;
                        }
                    }
                    Family::Shell | Family::Membrane => part.thickness = profiles::shell_thickness(sec).map(|x| x.0),
                    Family::Continuum2D => part.thickness = Some(sec.solid_thickness()),
                    _ => {}
                }
            }
            parts.push(part);
            parts.len() - 1
        });
        elem_part[e] = p as i32;
        let part = &mut parts[p];
        part.elements += 1;
        let rho = density_of(model, el.section);
        let vol = match fam {
            Family::Beam | Family::Truss => {
                let (a, b) = (el.nodes[0], el.nodes[el.nodes.len() - 1]);
                norm(sub(model.nodes[b], model.nodes[a])) * part.area.unwrap_or(0.0)
            }
            Family::Shell | Family::Membrane | Family::Continuum2D => measure(model, e) * part.thickness.unwrap_or(0.0),
            Family::Continuum3D => measure(model, e),
            _ => 0.0,
        };
        part.mass += rho * vol;
    }
    let members = members(model, &elem_part, &parts);
    let mut elem_member = vec![-1i32; ne];
    for (m, mb) in members.iter().enumerate() {
        for &e in &mb.elements {
            elem_member[e as usize] = m as i32;
        }
    }
    // bodies: connected continuum elements
    let cont: Vec<usize> = (0..ne).filter(|&e| matches!(model.elements[e].etype.family(), Family::Continuum3D | Family::Continuum2D) && elem_part[e] >= 0).collect();
    let mut uf = UnionFind::new(model.nodes.len());
    for &e in &cont {
        let n = &model.elements[e].nodes;
        for w in n.windows(2) {
            uf.union(w[0], w[1]);
        }
    }
    let mut root_body: HashMap<usize, usize> = HashMap::new();
    let mut elem_body = vec![-1i32; ne];
    let mut body_elems: Vec<Vec<usize>> = Vec::new();
    for &e in &cont {
        let r = uf.find(model.elements[e].nodes[0]);
        let b = *root_body.entry(r).or_insert_with(|| {
            body_elems.push(Vec::new());
            body_elems.len() - 1
        });
        body_elems[b].push(e);
        elem_body[e] = b as i32;
    }
    let bodies = body_elems
        .iter()
        .map(|els| {
            let dim = if model.elements[els[0]].etype.family() == Family::Continuum2D { 2 } else { 3 };
            let mut seen = std::collections::HashSet::new();
            let mut pts = Vec::new();
            for &e in els {
                for n in corners(model, e).1 {
                    if seen.insert(n) {
                        pts.push(model.nodes[n]);
                    }
                }
            }
            let c = scale(pts.iter().fold([0.0; 3], |a, &p| add(a, p)), 1.0 / pts.len().max(1) as f64);
            let mut cov = [[0.0; 3]; 3];
            for p in &pts {
                let d = sub(*p, c);
                for i in 0..3 {
                    for j in 0..3 {
                        cov[i][j] += d[i] * d[j];
                    }
                }
            }
            let (l, v) = eig3(cov);
            let k = (0..3).max_by(|&i, &j| l[i].total_cmp(&l[j])).unwrap();
            let axis = tidy_axis([v[0][k], v[1][k], v[2][k]]);
            let (mut s0, mut s1) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
            for p in &pts {
                let s = dot(sub(*p, c), axis);
                s0 = s0.min(s);
                s1 = s1.max(s);
                for i in 0..3 {
                    lo[i] = lo[i].min(p[i]);
                    hi[i] = hi[i].max(p[i]);
                }
            }
            Body { dim, elements: els.len() as u32, center: c, axis, s0, s1, lo, hi }
        })
        .collect();
    VisualModel { parts, profiles: profiles_out, members, bodies, elem_part, elem_member, elem_body }
}

// ---------------------------------------------------------------- results of a frame

/// Section forces of a member: global force and moment vectors on the positive section
/// (outward normal along the member) at points along it, sorted by `s`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MemberForces {
    pub s: Vec<f64>,
    pub f: Vec<[f64; 6]>,
    /// The shear forces are derived from the slope of the moments (the elements give none).
    pub derived_shear: bool,
}

/// Constrained nodes that belong together (connected through elements, same constraints).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Support {
    pub nodes: Vec<u32>,
    /// Constrained slots: bits 0–2 translations, 3–5 rotations.
    pub mask: u16,
    pub center: V3,
    /// Reaction force and moment about `center` (sum over the nodes).
    pub f: V3,
    pub m: V3,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VisualFrame {
    pub members: Vec<MemberForces>,
    pub supports: Vec<Support>,
    /// Nodal rotations UR1–UR3 (3 per node; empty when the model has no members).
    pub rot: Vec<f32>,
}

/// Fractions along the element of n integration points (Gauss–Legendre, ascending).
fn fractions(n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![0.5],
        2 => vec![0.5 - 0.5 / 3f64.sqrt(), 0.5 + 0.5 / 3f64.sqrt()],
        3 => vec![0.112_701_665_379_258_31, 0.5, 0.887_298_334_620_741_7],
        n => (0..n).map(|k| (k as f64 + 0.5) / n as f64).collect(),
    }
}

fn comp(labels: &[String], vals: &[f64], names: &[&str]) -> f64 {
    names.iter().find_map(|n| labels.iter().position(|l| l == n)).and_then(|i| vals.get(i)).copied().unwrap_or(0.0)
}

fn member_forces(model: &FeModel, fr: &FrameData, m: &Member) -> MemberForces {
    let pos: HashMap<u32, f64> = m.nodes.iter().copied().zip(m.s.iter().copied()).collect();
    let mut pts: Vec<(f64, usize, V3, V3)> = Vec::new(); // (s, element in chain, F, M)
    let mut any_shear = false;
    for (i, &e) in m.elements.iter().enumerate() {
        let e = e as usize;
        let el = &model.elements[e];
        let (a, b) = (el.nodes[0], el.nodes[el.nodes.len() - 1]);
        let t = unit(sub(model.nodes[b], model.nodes[a])).unwrap_or([1.0, 0.0, 0.0]);
        let n1a = match el.section.and_then(|si| model.sections[si].beam()) {
            Some(bs) if el.etype.dim() == 3 => bs.n1.unwrap_or([0.0, 0.0, -1.0]),
            _ => [0.0, 0.0, -1.0],
        };
        let n1 = profiles::segment_n1(t, n1a);
        let n2 = cross(t, n1);
        let (sa, sb) = (pos[&(a as u32)], pos[&(b as u32)]);
        let dir = m.dirs[i] as f64;
        let fields = fr.elem.get(e).map(|v| v.as_slice()).unwrap_or(&[]);
        let sf = fields.iter().find(|f| f.name == "SF");
        let sm = fields.iter().find(|f| f.name == "SM");
        let st = fields.iter().find(|f| f.name == "S");
        let area = match el.section.map(|s| &model.sections[s].kind) {
            Some(SectionKind::Solid { thickness: Some(a) }) => *a,
            Some(SectionKind::Beam(b)) => b.profile.a,
            _ => 0.0,
        };
        let n_pts = sf.map(|f| f.points.len()).or(sm.map(|f| f.points.len())).or(st.map(|f| f.points.len())).unwrap_or(0);
        let fr_ = fractions(n_pts);
        for q in 0..n_pts {
            let (mut nn, mut v2, mut v1, mut m1, mut m2, mut tq) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            if let Some(f) = sf {
                if let Some(p) = f.points.get(q) {
                    nn = comp(&f.labels, &p.2, &["SF1", "SF11"]);
                    v2 = comp(&f.labels, &p.2, &["SF2", "SF22"]);
                    v1 = comp(&f.labels, &p.2, &["SF3"]);
                }
            } else if let Some(f) = st {
                // trusses: axial stress × area (mean over the section points of this point)
                let ps: Vec<_> = f.points.iter().filter(|p| p.0 == f.points[q].0).collect();
                if !ps.is_empty() && f.points[q].1 == ps[0].1 {
                    nn = ps.iter().map(|p| comp(&f.labels, &p.2, &["S11"])).sum::<f64>() / ps.len() as f64 * area;
                } else {
                    continue;
                }
            }
            if let Some(f) = sm {
                if let Some(p) = f.points.get(q) {
                    m1 = comp(&f.labels, &p.2, &["SM1"]);
                    m2 = comp(&f.labels, &p.2, &["SM2"]);
                    tq = comp(&f.labels, &p.2, &["SM3"]);
                }
            }
            any_shear |= v1 != 0.0 || v2 != 0.0;
            let force = add(add(scale(t, nn), scale(n2, v2)), scale(n1, v1));
            let moment = add(add(scale(n1, m1), scale(n2, m2)), scale(t, tq));
            let s = sa + fr_[q] * (sb - sa);
            pts.push((s, i, scale(force, dir), scale(moment, dir)));
        }
    }
    pts.sort_by(|x, y| x.0.total_cmp(&y.0));
    // Euler–Bernoulli beams give no shear forces: V⊥ = t × dM/ds (equilibrium of a slice)
    let has_moment = pts.iter().any(|p| norm(p.3) > 0.0);
    let derived = !any_shear && has_moment && m.elements.iter().all(|&e| model.elements[e as usize].etype.family() == Family::Beam);
    if derived {
        let n = pts.len();
        let mut out = pts.clone();
        for k in 0..n {
            let ei = pts[k].1;
            let own: Vec<usize> = (0..n).filter(|&j| pts[j].1 == ei).collect();
            let (ia, ib) = if own.len() >= 2 { (own[0], own[own.len() - 1]) } else { (k.saturating_sub(1), (k + 1).min(n - 1)) };
            let ds = pts[ib].0 - pts[ia].0;
            if ds.abs() < 1e-300 {
                continue;
            }
            // least-squares slope over the element's own points (central difference otherwise)
            let slope = if own.len() >= 2 {
                let sm = own.iter().map(|&j| pts[j].0).sum::<f64>() / own.len() as f64;
                let mm = scale(own.iter().fold([0.0; 3], |a, &j| add(a, pts[j].3)), 1.0 / own.len() as f64);
                let den: f64 = own.iter().map(|&j| (pts[j].0 - sm).powi(2)).sum();
                scale(own.iter().fold([0.0; 3], |a, &j| add(a, scale(sub(pts[j].3, mm), pts[j].0 - sm))), 1.0 / den)
            } else {
                scale(sub(pts[ib].3, pts[ia].3), 1.0 / ds)
            };
            // the member tangent at the point
            let el = &model.elements[m.elements[ei] as usize];
            let (a, b) = (el.nodes[0], el.nodes[el.nodes.len() - 1]);
            let t = scale(unit(sub(model.nodes[b], model.nodes[a])).unwrap_or([1.0, 0.0, 0.0]), m.dirs[ei] as f64);
            out[k].2 = add(scale(t, dot(pts[k].2, t)), cross(t, slope));
        }
        pts = out;
    }
    MemberForces { s: pts.iter().map(|p| p.0).collect(), f: pts.iter().map(|p| [p.2[0], p.2[1], p.2[2], p.3[0], p.3[1], p.3[2]]).collect(), derived_shear: derived }
}

fn supports(model: &FeModel, fr: &FrameData) -> Vec<Support> {
    let nn = model.nodes.len();
    let mask = |n: usize| fr.constrained.get(n).copied().unwrap_or(0) & 0x3f;
    let mut uf = UnionFind::new(nn);
    for el in &model.elements {
        let c: Vec<usize> = el.nodes.iter().copied().filter(|&n| mask(n) != 0).collect();
        for w in c.windows(2) {
            if mask(w[0]) == mask(w[1]) {
                uf.union(w[0], w[1]);
            }
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut order = Vec::new();
    for n in 0..nn {
        if mask(n) != 0 {
            let r = uf.find(n);
            let g = groups.entry(r).or_default();
            if g.is_empty() {
                order.push(r);
            }
            g.push(n);
        }
    }
    order
        .into_iter()
        .map(|r| {
            let nodes = &groups[&r];
            let c = scale(nodes.iter().fold([0.0; 3], |a, &n| add(a, model.nodes[n])), 1.0 / nodes.len() as f64);
            let (mut f, mut m) = ([0.0; 3], [0.0; 3]);
            for &n in nodes {
                if let Some(rf) = fr.rf.get(n) {
                    let fv = [rf[0], rf[1], rf[2]];
                    f = add(f, fv);
                    m = add(add(m, cross(sub(model.nodes[n], c), fv)), [rf[3], rf[4], rf[5]]);
                }
            }
            Support { nodes: nodes.iter().map(|&n| n as u32).collect(), mask: mask(nodes[0]), center: c, f, m }
        })
        .collect()
}

pub fn frame(model: &FeModel, vm: &VisualModel, fr: &FrameData) -> VisualFrame {
    let members = vm.members.iter().map(|m| member_forces(model, fr, m)).collect();
    let rot = if vm.members.is_empty() {
        Vec::new()
    } else {
        let mut r = vec![0.0f32; 3 * model.nodes.len()];
        for (n, u) in fr.u.iter().enumerate().take(model.nodes.len()) {
            for k in 0..3 {
                r[3 * n + k] = u[3 + k] as f32;
            }
        }
        r
    };
    VisualFrame { members, supports: supports(model, fr), rot }
}

// ---------------------------------------------------------------- shells

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ShellForces {
    /// N11, N22, N12, V13, V23 (per unit width) and M11, M22, M12 (per unit width), element means.
    pub sf: Option<Vec<f64>>,
    pub sm: Option<Vec<f64>>,
    /// The local directions 1 and 2 (Abaqus default: 1 = global x projected onto the shell).
    pub dir1: V3,
    pub dir2: V3,
}

pub fn shell(model: &FeModel, fr: &FrameData, e: usize) -> ShellForces {
    let mean = |name: &str, comps: &[&str]| -> Option<Vec<f64>> {
        let f = fr.elem.get(e)?.iter().find(|f| f.name == name)?;
        Some(comps.iter().map(|c| crate::fields::element_mean(fr, e, name, c).unwrap_or(0.0)).collect::<Vec<_>>()).filter(|_| !f.points.is_empty())
    };
    let (_, c) = corners(model, e);
    let x: Vec<V3> = c.iter().map(|&n| model.nodes[n]).collect();
    let nrm = if x.len() >= 4 { unit(cross(sub(x[2], x[0]), sub(x[3], x[1]))) } else if x.len() == 3 { unit(cross(sub(x[1], x[0]), sub(x[2], x[0]))) } else { None }.unwrap_or([0.0, 0.0, 1.0]);
    let g = if nrm[0].abs() > (0.1f64).to_radians().cos() { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let d1 = unit(sub(g, scale(nrm, dot(g, nrm)))).unwrap_or([1.0, 0.0, 0.0]);
    ShellForces { sf: mean("SF", &["SF1", "SF2", "SF3", "SF4", "SF5"]), sm: mean("SM", &["SM1", "SM2", "SM3"]), dir1: d1, dir2: cross(nrm, d1) }
}

// ---------------------------------------------------------------- solid and planar bodies: stress integration

/// Recovered nodal stresses (S11, S22, S33, S12, S13, S23) of all continuum corner nodes,
/// and the elements' internal nodal forces when they are known.
pub struct Recovered {
    pub nodal: Vec<Option<[f64; 6]>>,
    /// Per element: (node, force on the element from the node, translations); `None` for an
    /// element whose nodal forces are not known.
    pub fint: Option<Vec<Option<Vec<(u32, V3)>>>>,
}

/// Sums an element's internal force vector (in the order of its DOFs) to translational forces per node.
fn per_node(model: &FeModel, e: usize, fint: &[f64]) -> Option<Vec<(u32, V3)>> {
    let el = &model.elements[e];
    let mut acc: Vec<(u32, V3)> = Vec::new();
    let mut k = 0usize;
    for (a, &n) in el.nodes.iter().enumerate() {
        for &d in el.etype.node_dofs(a) {
            let slot = rustyfem::fem::dof::slot(d)?;
            let v = *fint.get(k)?;
            k += 1;
            if slot < 3 {
                match acc.iter_mut().find(|x| x.0 == n as u32) {
                    Some(x) => x.1[slot] += v,
                    None => {
                        let mut f = [0.0; 3];
                        f[slot] = v;
                        acc.push((n as u32, f));
                    }
                }
            }
        }
    }
    (k == fint.len()).then_some(acc)
}

/// Internal nodal forces of the body elements. The solver stores them with the results
/// (`FrameData::nforc`, evaluated with the converged material state, so they are right for
/// plasticity, NLGEOM and temperature loads too). For results without them (older project
/// files) they are recomputed as K u, which is only valid for a linear model without a
/// temperature field; otherwise `None` (the stresses are integrated instead).
fn internal_forces(model: &FeModel, vm: &VisualModel, fr: &FrameData, nlgeom: bool) -> Option<Vec<Option<Vec<(u32, V3)>>>> {
    let ne = model.elements.len();
    if fr.nforc.len() == ne {
        return Some((0..ne).map(|e| if vm.elem_body[e] < 0 || fr.nforc[e].is_empty() { None } else { per_node(model, e, &fr.nforc[e]) }).collect());
    }
    if nlgeom {
        return None;
    }
    let p = rustyfem::fem::assembly::Prepared::new(model).ok()?;
    if !p.is_linear() || p.temperature.is_some() || fr.u.len() < model.nodes.len() {
        return None;
    }
    let mut out = vec![None; ne];
    for e in 0..ne {
        if vm.elem_body[e] < 0 {
            continue;
        }
        let ctx = p.ctx(e, false, fr.value, 0.0);
        let u = p.gather(e, &fr.u);
        let et = model.elements[e].etype;
        let old = et.init_state(&ctx).ok()?;
        let mut new = old.clone();
        let r = et.internal_force(&ctx, &u, &vec![0.0; u.len()], &old, &mut new, false).ok()?;
        out[e] = per_node(model, e, &r.fint);
    }
    Some(out)
}

const S6: [&str; 6] = ["S11", "S22", "S33", "S12", "S13", "S23"];

fn element_stress(fr: &FrameData, e: usize) -> Option<[f64; 6]> {
    let f = fr.elem.get(e)?.iter().find(|f| f.name == "S")?;
    let idx: Vec<Option<usize>> = S6.iter().map(|c| f.labels.iter().position(|l| l == c)).collect();
    if f.points.is_empty() {
        return None;
    }
    let mut s = [0.0; 6];
    for p in &f.points {
        for k in 0..6 {
            if let Some(i) = idx[k] {
                s[k] += p.2.get(i).copied().unwrap_or(0.0);
            }
        }
    }
    Some(s.map(|x| x / f.points.len() as f64))
}

pub fn recover(model: &FeModel, vm: &VisualModel, fr: &FrameData, nlgeom: bool) -> Recovered {
    let nn = model.nodes.len();
    let ne = model.elements.len();
    let mut mean: Vec<Option<[f64; 6]>> = vec![None; ne];
    let mut cen: Vec<V3> = vec![[0.0; 3]; ne];
    let mut node_elems: Vec<Vec<u32>> = vec![Vec::new(); nn];
    for e in 0..ne {
        if vm.elem_body[e] < 0 {
            continue;
        }
        let (_, c) = corners(model, e);
        cen[e] = scale(c.iter().fold([0.0; 3], |a, &n| add(a, model.nodes[n])), 1.0 / c.len() as f64);
        mean[e] = element_stress(fr, e);
        for &n in &model.elements[e].nodes {
            node_elems[n].push(e as u32);
        }
    }
    let mut nodal: Vec<Option<[f64; 6]>> = vec![None; nn];
    let mut stamp = vec![usize::MAX; ne];
    let mut patch: Vec<usize> = Vec::new();
    for n in 0..nn {
        if node_elems[n].is_empty() {
            continue;
        }
        patch.clear();
        for &e1 in &node_elems[n] {
            for &m in &model.elements[e1 as usize].nodes {
                for &e2 in &node_elems[m] {
                    let e2 = e2 as usize;
                    if stamp[e2] != n && mean[e2].is_some() && vm.elem_body[e2] == vm.elem_body[e1 as usize] {
                        stamp[e2] = n;
                        patch.push(e2);
                    }
                }
            }
        }
        if patch.is_empty() {
            continue;
        }
        let k = patch.len() as f64;
        let cb = scale(patch.iter().fold([0.0; 3], |a, &e| add(a, cen[e])), 1.0 / k);
        let mut sb = [0.0; 6];
        for &e in &patch {
            let s = mean[e].unwrap();
            for c in 0..6 {
                sb[c] += s[c] / k;
            }
        }
        let mut cov = [[0.0; 3]; 3];
        let mut rhs = [[0.0; 3]; 6];
        for &e in &patch {
            let d = sub(cen[e], cb);
            let s = mean[e].unwrap();
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
                for c in 0..6 {
                    rhs[c][i] += d[i] * (s[c] - sb[c]);
                }
            }
        }
        // gradient = C⁺ b (directions without spread in the patch get no gradient)
        let (l, v) = eig3(cov);
        let lmax = l.iter().fold(0.0f64, |a, &x| a.max(x));
        let dx = sub(model.nodes[n], cb);
        let mut out = sb;
        for q in 0..3 {
            if lmax <= 0.0 || l[q] <= 1e-8 * lmax {
                continue;
            }
            let vq = [v[0][q], v[1][q], v[2][q]];
            let w = dot(vq, dx) / l[q];
            for c in 0..6 {
                out[c] += w * dot(vq, rhs[c]);
            }
        }
        nodal[n] = Some(out);
    }
    Recovered { nodal, fint: internal_forces(model, vm, fr, nlgeom) }
}

/// Edges (corner pairs) of the continuum shapes.
fn shape_edges(sh: Shape, n: usize) -> Vec<(usize, usize)> {
    match sh {
        Shape::Tet => vec![(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
        Shape::Wedge => vec![(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)],
        Shape::Hex => vec![(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)],
        Shape::Face => (0..n).map(|k| (k, (k + 1) % n)).collect(),
        _ => vec![],
    }
}

fn traction(s: &[f64; 6], n: V3) -> V3 {
    // S11, S22, S33, S12, S13, S23
    [s[0] * n[0] + s[3] * n[1] + s[4] * n[2], s[3] * n[0] + s[1] * n[1] + s[5] * n[2], s[4] * n[0] + s[5] * n[1] + s[2] * n[2]]
}

/// Resultant of the stresses on a cut: force and moment about the area centroid.
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CutResult {
    pub s: f64,
    pub area: f64,
    pub centroid: V3,
    pub f: V3,
    pub m: V3,
    /// The cut face: triangles (3D bodies) or segments (planar bodies), flat xyz.
    pub tris: Vec<f32>,
    pub segs: Vec<f32>,
    /// "nodal forces" (free-body cut, exact for the FE solution) or "stresses" (integrated).
    pub method: String,
}

/// Cuts body `body` with the plane through `p` with normal `n`. The result is the
/// force and moment that the part on the +n side exerts on the part on the −n side,
/// i.e. on the positive section (outward normal +n).
pub fn cut(model: &FeModel, vm: &VisualModel, rec: &Recovered, body: usize, p: V3, n: V3, geometry: bool) -> CutResult {
    let mut out = CutResult::default();
    let (mut fi, mut ri, mut rxt, mut a_sum) = ([0.0; 3], [0.0; 3], [0.0; 3], 0.0);
    let pl_u = unit(cross(n, if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] })).unwrap();
    let pl_v = cross(n, pl_u);
    // free-body cut: the force on the −n part from the elements that cross the plane
    let (mut fn_, mut mn) = ([0.0; 3], [0.0; 3]);
    let mut nodal_ok = rec.fint.is_some();
    for e in 0..model.elements.len() {
        if vm.elem_body[e] != body as i32 {
            continue;
        }
        let (sh, c) = corners(model, e);
        let d: Vec<f64> = c.iter().map(|&k| dot(sub(model.nodes[k], p), n)).collect();
        let neg = d.iter().any(|&x| x < 0.0);
        let pos = d.iter().any(|&x| x >= 0.0);
        if !(neg && pos) {
            continue;
        }
        if let Some(fint) = &rec.fint {
            match &fint[e] {
                Some(fe) => {
                    for &(k, f) in fe {
                        let x = model.nodes[k as usize];
                        if dot(sub(x, p), n) < 0.0 {
                            fn_ = sub(fn_, f);
                            mn = sub(mn, cross(x, f));
                        }
                    }
                }
                None => nodal_ok = false,
            }
        }
        let svals: Vec<[f64; 6]> = c.iter().map(|&k| rec.nodal[k].unwrap_or([0.0; 6])).collect();
        let mut pts: Vec<(V3, [f64; 6])> = Vec::new();
        for (a, b) in shape_edges(sh, c.len()) {
            let (da, db) = (d[a], d[b]);
            if (da < 0.0) == (db < 0.0) {
                continue;
            }
            let w = da / (da - db);
            let x = add(model.nodes[c[a]], scale(sub(model.nodes[c[b]], model.nodes[c[a]]), w));
            let mut s = [0.0; 6];
            for k in 0..6 {
                s[k] = svals[a][k] + w * (svals[b][k] - svals[a][k]);
            }
            let size = norm(sub(model.nodes[c[b]], model.nodes[c[a]]));
            if !pts.iter().any(|q| norm(sub(q.0, x)) <= 1e-9 * size) {
                pts.push((x, s));
            }
        }
        if sh == Shape::Face {
            // planar element: a segment through the element times its thickness
            if pts.len() < 2 {
                continue;
            }
            let (mut i0, mut i1, mut dm) = (0, 1, -1.0);
            for i in 0..pts.len() {
                for j in i + 1..pts.len() {
                    let dd = norm(sub(pts[i].0, pts[j].0));
                    if dd > dm {
                        (i0, i1, dm) = (i, j, dd);
                    }
                }
            }
            let th = model.elements[e].section.map_or(1.0, |s| model.sections[s].solid_thickness());
            let ((xa, sa), (xb, sb)) = (pts[i0], pts[i1]);
            let (ta, tb) = (traction(&sa, n), traction(&sb, n));
            let xm = scale(add(xa, xb), 0.5);
            let tm = scale(add(ta, tb), 0.5);
            let l = dm * th;
            fi = add(fi, scale(add(ta, tb), l / 2.0));
            rxt = add(rxt, scale(add(add(cross(xa, ta), scale(cross(xm, tm), 4.0)), cross(xb, tb)), l / 6.0));
            ri = add(ri, scale(xm, l));
            a_sum += l;
            if geometry {
                out.segs.extend(f32v(xa));
                out.segs.extend(f32v(xb));
            }
            continue;
        }
        if pts.len() < 3 {
            continue;
        }
        let cm = scale(pts.iter().fold([0.0; 3], |a, q| add(a, q.0)), 1.0 / pts.len() as f64);
        pts.sort_by(|a, b| {
            let (da, db) = (sub(a.0, cm), sub(b.0, cm));
            dot(da, pl_v).atan2(dot(da, pl_u)).total_cmp(&dot(db, pl_v).atan2(dot(db, pl_u)))
        });
        for k in 1..pts.len() - 1 {
            let tri = [pts[0], pts[k], pts[k + 1]];
            let ar = norm(cross(sub(tri[1].0, tri[0].0), sub(tri[2].0, tri[0].0))) / 2.0;
            if ar <= 0.0 {
                continue;
            }
            let t: Vec<V3> = tri.iter().map(|q| traction(&q.1, n)).collect();
            fi = add(fi, scale(add(add(t[0], t[1]), t[2]), ar / 3.0));
            // r × t is quadratic over the triangle: the edge midpoint rule is exact
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let xm = scale(add(tri[i].0, tri[j].0), 0.5);
                let tm = scale(add(t[i], t[j]), 0.5);
                rxt = add(rxt, scale(cross(xm, tm), ar / 3.0));
            }
            ri = add(ri, scale(add(add(tri[0].0, tri[1].0), tri[2].0), ar / 3.0));
            a_sum += ar;
            if geometry {
                for q in tri {
                    out.tris.extend(f32v(q.0));
                }
            }
        }
    }
    if a_sum > 0.0 {
        let c = scale(ri, 1.0 / a_sum);
        out.area = a_sum;
        out.centroid = c;
        if nodal_ok {
            out.f = fn_;
            out.m = sub(mn, cross(c, fn_));
            out.method = "nodal forces".into();
        } else {
            out.f = fi;
            out.m = sub(rxt, cross(c, fi));
            out.method = "stresses".into();
        }
    }
    out
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BodyDiagram {
    pub axis: V3,
    pub s0: f64,
    pub s1: f64,
    pub s: Vec<f64>,
    /// F (3) and M (3) about the cut centroid per station, on the positive section.
    pub f: Vec<[f64; 6]>,
    pub area: Vec<f64>,
}

/// Range of a body's nodes along `axis` measured from its centre.
pub fn body_range(model: &FeModel, vm: &VisualModel, body: usize, axis: V3) -> (f64, f64) {
    let c = vm.bodies[body].center;
    let (mut s0, mut s1) = (f64::INFINITY, f64::NEG_INFINITY);
    for e in 0..model.elements.len() {
        if vm.elem_body[e] == body as i32 {
            for k in corners(model, e).1 {
                let s = dot(sub(model.nodes[k], c), axis);
                s0 = s0.min(s);
                s1 = s1.max(s);
            }
        }
    }
    (s0, s1)
}

pub fn diagram(model: &FeModel, vm: &VisualModel, rec: &Recovered, body: usize, axis: V3, n: usize) -> BodyDiagram {
    let axis = unit(axis).unwrap_or(vm.bodies[body].axis);
    let (s0, s1) = body_range(model, vm, body, axis);
    let c = vm.bodies[body].center;
    let n = n.clamp(2, 1000);
    let len = s1 - s0;
    let mut out = BodyDiagram { axis, s0, s1, s: Vec::new(), f: Vec::new(), area: Vec::new() };
    for i in 0..n {
        let s = s0 + len * (0.002 + 0.996 * i as f64 / (n - 1) as f64);
        let r = cut(model, vm, rec, body, add(c, scale(axis, s)), axis, false);
        if r.area <= 0.0 {
            continue;
        }
        out.s.push(s);
        out.f.push([r.f[0], r.f[1], r.f[2], r.m[0], r.m[1], r.m[2]]);
        out.area.push(r.area);
    }
    out
}

#[cfg(test)]
mod tests {
    use crate::{run_analysis, Session};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    fn solved(script: &str) -> Session {
        let mut s = Session::new_fem(script).unwrap();
        let out = run_analysis(s.model(), None, false, Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
        s.set_results(out);
        s
    }

    /// Total applied load = −(sum of the reactions).
    fn load(s: &mut Session) -> [f64; 3] {
        let fr = s.visual_frame(0, s.results.as_ref().unwrap().steps[0].frames.len() - 1).unwrap();
        let mut r = [0.0; 3];
        for sp in &fr.supports {
            for k in 0..3 {
                r[k] -= sp.f[k];
            }
        }
        r
    }

    fn last(s: &Session) -> usize {
        s.results.as_ref().unwrap().steps[0].frames.len() - 1
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-12)
    }

    #[test]
    fn solid_cantilever_cut_matches_statics() {
        let mut s = solved("model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.05\nfix x=0\nload x=1 fy=-5000\n");
        let p = load(&mut s);
        assert!(close(p[1], -5000.0, 1e-6), "{p:?}");
        let vm = s.visual_model();
        assert_eq!(vm.bodies.len(), 1);
        let b = &vm.bodies[0];
        assert_eq!(b.axis, [1.0, 0.0, 0.0]);
        let f = last(&s);
        for x in [0.3, 0.5, 0.7] {
            let r = s.visual_cut(0, f, 0, [1.0, 0.0, 0.0], x - b.center[0]).unwrap();
            assert!(close(r.area, 0.04, 1e-9), "{}", r.area);
            // the part beyond the cut carries the tip load: F = P, M = (x_P − x) e_x × P
            assert_eq!(r.method, "nodal forces");
            assert!(close(r.f[1], p[1], 1e-6), "x={x}: {:?}", r.f);
            assert!(r.f[0].abs() < 1e-6 * p[1].abs(), "{:?}", r.f);
            assert!(close(r.m[2], (1.0 - x) * p[1], 1e-6), "x={x}: {:?} vs {}", r.m, (1.0 - x) * p[1]);
        }
        // the stress integration (used for nonlinear models) is close
        let fr = &s.results.as_ref().unwrap().steps[0].frames[f];
        let mut rec = super::recover(s.model(), &vm, fr, false);
        rec.fint = None;
        let r = super::cut(s.model(), &vm, &rec, 0, [0.5, 0.1, 0.1], [1.0, 0.0, 0.0], false);
        assert_eq!(r.method, "stresses");
        assert!(close(r.f[1], p[1], 0.2), "{:?}", r.f);
        assert!(close(r.m[2], 0.5 * p[1], 0.05), "{:?}", r.m);
        let d = s.visual_diagram(0, f, 0, [1.0, 0.0, 0.0], 41).unwrap();
        assert!(d.s.len() > 30);
    }

    #[test]
    fn plane_cantilever_cut_matches_statics() {
        let mut s = solved("model 2d\nmaterial steel E=210e9 nu=0.3\nplate x=0..2 y=0..0.2 t=0.01 size=0.025\nfix x=0\nload x=2 fy=-1000\n");
        let p = load(&mut s);
        let vm = s.visual_model();
        assert_eq!(vm.bodies.len(), 1);
        assert_eq!(vm.bodies[0].dim, 2);
        let c = vm.bodies[0].center;
        let f = last(&s);
        for x in [0.5, 1.0, 1.5] {
            let r = s.visual_cut(0, f, 0, [1.0, 0.0, 0.0], x - c[0]).unwrap();
            assert!(close(r.area, 0.2 * 0.01, 1e-9), "{}", r.area);
            assert!(close(r.f[1], p[1], 1e-6), "x={x}: {:?} {p:?}", r.f);
            assert!(close(r.m[2], (2.0 - x) * p[1], 1e-6), "x={x}: {:?}", r.m);
        }
    }

    #[test]
    fn beam_member_forces_match_statics() {
        let mut s = solved("model 3d\nmaterial steel\nbeam (0,0,0) (2,0,0) section=rect(0.1,0.1) n=10\nfix x=0\nload at=(2,0,0) fy=-1000\n");
        let p = load(&mut s);
        let vm = s.visual_model();
        assert_eq!(vm.members.len(), 1);
        assert!(close(vm.members[0].length, 2.0, 1e-12));
        let f = last(&s);
        let fr = s.visual_frame(0, f).unwrap();
        let m = &fr.members[0];
        assert!(!m.s.is_empty());
        for (k, &x) in m.s.iter().enumerate() {
            let v = m.f[k];
            assert!(close(v[1], p[1], 1e-6), "s={x}: {v:?}");
            assert!(close(v[5], (2.0 - x) * p[1], 1e-6), "s={x}: {v:?}");
        }
    }
}

#[cfg(test)]
mod nonlinear_tests {
    use crate::{run_analysis, Session};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// C3D8 cantilever 1 × 0.2 × 0.2 (nx × 4 × 4 elements), fixed at x = 0, total load `p`
    /// in −y spread over the nodes of the end face. `extra` goes into the material.
    fn cantilever(nx: usize, p: f64, material: &str, step: &str) -> String {
        let (ny, nz) = (4, 4);
        let id = |i: usize, j: usize, k: usize| 1 + i + (nx + 1) * (j + (ny + 1) * k);
        let mut s = String::from("*HEADING\ncantilever\n*NODE\n");
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    s += &format!("{}, {}, {}, {}\n", id(i, j, k), i as f64 / nx as f64, 0.2 * j as f64 / ny as f64, 0.2 * k as f64 / nz as f64);
                }
            }
        }
        s += "*ELEMENT, TYPE=C3D8, ELSET=BEAM\n";
        let mut e = 1;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    s += &format!("{e}, {}, {}, {}, {}, {}, {}, {}, {}\n", id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k), id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1));
                    e += 1;
                }
            }
        }
        let face = |i: usize| (0..=nz).flat_map(move |k| (0..=ny).map(move |j| id(i, j, k))).map(|n| n.to_string()).collect::<Vec<_>>().join(", ");
        let nn = (ny + 1) * (nz + 1);
        s += &format!("*NSET, NSET=ROOT\n{}\n*NSET, NSET=TIP\n{}\n", face(0), face(nx));
        s += &format!("*MATERIAL, NAME=STEEL\n*ELASTIC\n210e9, 0.3\n{material}*SOLID SECTION, ELSET=BEAM, MATERIAL=STEEL\n*BOUNDARY\nROOT, 1, 3\n");
        s += &format!("{step}\n*CLOAD\nTIP, 2, {}\n*END STEP\n", -p / nn as f64);
        s
    }

    fn solve(inp: &str) -> Session {
        let mut s = Session::from_inp_text(inp, std::path::Path::new(".")).unwrap();
        let out = run_analysis(s.model(), None, false, Arc::new(AtomicBool::new(false)), |_| {}).unwrap();
        s.set_results(out);
        s
    }

    fn check(s: &mut Session, p: f64, tol: f64) {
        let f = s.results.as_ref().unwrap().steps[0].frames.len() - 1;
        let fr = &s.results.as_ref().unwrap().steps[0].frames[f];
        assert_eq!(fr.nforc.len(), s.model().elements.len(), "the solver stored the nodal forces");
        let c = s.visual_model().bodies[0].center;
        for x in [0.15, 0.5, 0.85] {
            let r = s.visual_cut(0, f, 0, [1.0, 0.0, 0.0], x - c[0]).unwrap();
            assert_eq!(r.method, "nodal forces");
            assert!((r.f[1] + p).abs() <= tol * p, "x={x}: {:?}", r.f);
            assert!(r.f[0].abs() <= tol * p, "x={x}: {:?}", r.f);
            let m = -(1.0 - x) * p;
            assert!((r.m[2] - m).abs() <= tol * m.abs(), "x={x}: {:?} vs {m}", r.m);
        }
    }

    #[test]
    fn plastic_cantilever_cut_matches_statics() {
        // the root fibres yield (elastic limit about 267 kN), plastic collapse at 400 kN
        let p = 370e3;
        let mut s = solve(&cantilever(10, p, "*PLASTIC\n200e6, 0\n", "*STEP\n*STATIC\n0.1, 1.0, 1e-5, 0.1"));
        let f = s.results.as_ref().unwrap().steps[0].frames.len() - 1;
        // the root has yielded
        let fr = &s.results.as_ref().unwrap().steps[0].frames[f];
        let peeq = fr.elem.iter().filter_map(|fs| fs.iter().find(|x| x.name == "PEEQ")).flat_map(|x| x.points.iter().map(|q| q.2[0])).fold(0.0, f64::max);
        assert!(peeq > 1e-5, "no plastic strain: {peeq}");
        check(&mut s, p, 1e-6);
    }

    #[test]
    fn nlgeom_cantilever_nodal_forces_are_stored() {
        let p = 50e3;
        let mut s = solve(&cantilever(10, p, "", "*STEP, NLGEOM=YES\n*STATIC\n0.25, 1.0"));
        // small deflection: the statics of the undeformed beam hold within a few per mille
        check(&mut s, p, 5e-3);
    }
}
