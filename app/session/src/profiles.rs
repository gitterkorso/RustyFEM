//! Data for drawing beams with their real cross-section and shells with their
//! thickness (the viewer extrudes them in 3D).
//!
//! Beams: the profile as convex polygons in the section plane (x1 along n1,
//! x2 along n2, like *BEAM SECTION) and n1 per display line segment. The shape comes from
//! *BEAM SECTION, SECTION=... or from the `.rusty` section (`BeamSection::outline`);
//! sections given only by their properties (*BEAM GENERAL SECTION) are drawn as the
//! rectangle with the same A, I11 and I22. Trusses are drawn as a circle with their area.
//!
//! Shells and membranes: thickness, the midsurface position relative to the nodes
//! (OFFSET) and the element normal per element.

use crate::mesh::DisplayMesh;
use rustyfem::elements::Family;
use rustyfem::fem::model::{FeModel, SectionDef, SectionKind};
use std::collections::HashMap;
use std::f64::consts::PI;

type Poly = Vec<[f64; 2]>;

fn rect(x0: f64, x1: f64, y0: f64, y1: f64) -> Poly {
    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
}

/// Regular polygon with n corners (radius r, first corner at angle a0).
fn ngon(r: f64, n: usize, a0: f64) -> Vec<[f64; 2]> {
    (0..n).map(|k| {
        let a = a0 + 2.0 * PI * k as f64 / n as f64;
        [r * a.cos(), r * a.sin()]
    }).collect()
}

/// Ring between two polygons with the same number of corners, as quadrilaterals.
fn ring(outer: &[[f64; 2]], inner: &[[f64; 2]]) -> Vec<Poly> {
    let n = outer.len();
    (0..n).map(|k| {
        let j = (k + 1) % n;
        vec![inner[k], outer[k], outer[j], inner[j]]
    }).collect()
}

/// A straight wall from a to b with thickness t, extended by t/2 at both ends (closes corners).
fn wall(a: [f64; 2], b: [f64; 2], t: f64) -> Option<Poly> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let l = (dx * dx + dy * dy).sqrt();
    if !(t > 0.0 && l > 0.0) {
        return None;
    }
    let (ux, uy) = (dx / l * t / 2.0, dy / l * t / 2.0);
    let (nx, ny) = (-uy, ux);
    Some(vec![
        [a[0] - ux - nx, a[1] - uy - ny],
        [b[0] + ux - nx, b[1] + uy - ny],
        [b[0] + ux + nx, b[1] + uy + ny],
        [a[0] - ux + nx, a[1] - uy + ny],
    ])
}

fn chain(pts: &[[f64; 2]], t: &[f64]) -> Vec<Poly> {
    pts.windows(2).zip(t).filter_map(|(w, &t)| wall(w[0], w[1], t)).collect()
}

const ROUND: usize = 32;

/// Polygons for a section shape with its dimensions (Abaqus *BEAM SECTION order).
pub fn shape_polys(shape: &str, v: &[f64]) -> Option<Vec<Poly>> {
    let need = |n: usize| (v.len() >= n).then_some(());
    let polys = match shape {
        "RECT" => {
            need(2)?;
            vec![rect(-v[0] / 2.0, v[0] / 2.0, -v[1] / 2.0, v[1] / 2.0)]
        }
        "CIRC" => {
            need(1)?;
            vec![ngon(v[0], ROUND, 0.0)]
        }
        "PIPE" | "THICK PIPE" => {
            need(2)?;
            let ri = (v[0] - v[1]).max(0.0);
            if ri == 0.0 {
                vec![ngon(v[0], ROUND, 0.0)]
            } else {
                ring(&ngon(v[0], ROUND, 0.0), &ngon(ri, ROUND, 0.0))
            }
        }
        "BOX" => {
            need(6)?;
            let (a, b, t1, t2, t3, t4) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let (xl, xr, yb, yt) = (-a / 2.0, a / 2.0, -b / 2.0, b / 2.0);
            vec![
                rect(xr - t1, xr, yb, yt),
                rect(xl, xl + t3, yb, yt),
                rect(xl + t3, xr - t1, yt - t2, yt),
                rect(xl + t3, xr - t1, yb, yb + t4),
            ]
        }
        "I" => {
            need(7)?;
            let (l, h, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            vec![
                rect(-b1 / 2.0, b1 / 2.0, -l, -l + t1),
                rect(-b2 / 2.0, b2 / 2.0, -l + h - t2, -l + h),
                rect(-t3 / 2.0, t3 / 2.0, -l + t1, -l + h - t2),
            ]
        }
        "L" => {
            need(4)?;
            let (a, b, t1, t2) = (v[0], v[1], v[2], v[3]);
            vec![rect(-t2 / 2.0, a - t2 / 2.0, -t1 / 2.0, t1 / 2.0), rect(-t2 / 2.0, t2 / 2.0, t1 / 2.0, b - t1 / 2.0)]
        }
        "HEX" => {
            need(2)?;
            let a0 = PI / 6.0;
            let inner = v[0] - 2.0 * v[1] / 3f64.sqrt();
            if inner <= 0.0 {
                vec![ngon(v[0], 6, a0)]
            } else {
                ring(&ngon(v[0], 6, a0), &ngon(inner, 6, a0))
            }
        }
        "TRAPEZOID" => {
            need(4)?;
            let (a, b, c, d) = (v[0], v[1], v[2], v[3]);
            vec![vec![[-a / 2.0, -d], [a / 2.0, -d], [c / 2.0, b - d], [-c / 2.0, b - d]]]
        }
        "CHANNEL" => {
            need(8)?;
            let (l, h, b1, b2, t1, t2, t3, o) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            vec![
                rect(-o, -o + b1, -l, -l + t1),
                rect(-o, -o + b2, -l + h - t2, -l + h),
                rect(-o, -o + t3, -l + t1, -l + h - t2),
            ]
        }
        "HAT" => {
            need(8)?;
            let (l, h, b, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            let (yb, yt) = (-l + t2 / 2.0, -l + h - t1 / 2.0);
            let (xo, xi, xt) = (b / 2.0, b / 2.0 - b2, b1 / 2.0);
            let pts = [[-xo, yb], [-xi, yb], [-xt, yt], [xt, yt], [xi, yb], [xo, yb]];
            chain(&pts, &[t2, t3, t1, t3, t2])
        }
        "ARBITRARY" => {
            // count, x1, x2, then (x1, x2, t) per segment
            let n = *v.first()? as usize;
            need(3 + 3 * n)?;
            let mut a = [v[1], v[2]];
            let mut out = Vec::new();
            for k in 0..n {
                let o = 3 + 3 * k;
                let b = [v[o], v[o + 1]];
                out.extend(wall(a, b, v[o + 2]));
                a = b;
            }
            out
        }
        _ => return None,
    };
    let ok = !polys.is_empty() && polys.iter().all(|p| p.iter().all(|q| q[0].is_finite() && q[1].is_finite()));
    ok.then_some(polys)
}

/// The rectangle with the same A, I11 (about the 1-axis: height along 2) and I22.
fn equivalent_rect(a: f64, i11: f64, i22: f64) -> Option<Vec<Poly>> {
    if !(a > 0.0 && i11 > 0.0 && i22 > 0.0) {
        return None;
    }
    let h = (12.0 * i11 / a).sqrt();
    let b = (12.0 * i22 / a).sqrt();
    Some(vec![rect(-b / 2.0, b / 2.0, -h / 2.0, h / 2.0)])
}

/// Profile polygons for a section used by line elements.
fn section_polys(sec: &SectionDef, family: Family) -> Option<Vec<Poly>> {
    match &sec.kind {
        SectionKind::Beam(b) => {
            if let Some((shape, dims)) = &b.profile.outline {
                if let Some(p) = shape_polys(shape, dims) {
                    return Some(p);
                }
            }
            let dims = rustyfem::elements::beams2::sections::data_values(&sec.raw, &b.shape).ok();
            if let Some(p) = dims.and_then(|d| shape_polys(&b.shape, &d)) {
                return Some(p);
            }
            equivalent_rect(b.profile.a, b.profile.i11, b.profile.i22)
        }
        // trusses: a circle with the cross-section area
        SectionKind::Solid { thickness: Some(a) } if matches!(family, Family::Truss) && *a > 0.0 => Some(vec![ngon((a / PI).sqrt(), 16, 0.0)]),
        _ => None,
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn unit(a: [f64; 3]) -> Option<[f64; 3]> {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    (l > 0.0).then(|| [a[0] / l, a[1] / l, a[2] / l])
}

/// n1 for a segment with the direction t: the section's n1 (default (0, 0, −1);
/// planar beams always (0, 0, −1)) made perpendicular to t, like `beam_axes`.
fn segment_n1(t: [f64; 3], n1a: [f64; 3]) -> [f64; 3] {
    let n2 = cross(t, n1a);
    let n2 = unit(n2).or_else(|| unit(cross(t, [1.0, 0.0, 0.0]))).or_else(|| unit(cross(t, [0.0, 1.0, 0.0]))).unwrap_or([0.0, 1.0, 0.0]);
    cross(n2, t)
}

/// Shell thickness and the midsurface position relative to the nodes (−OFFSET·t).
fn shell_thickness(sec: &SectionDef) -> Option<(f64, f64)> {
    let t = match &sec.kind {
        SectionKind::Shell { thickness, .. } => *thickness,
        SectionKind::Other if sec.raw.name == "SHELL SECTION" && sec.raw.has("COMPOSITE") => {
            rustyfem::elements::shell::section::composite_plies(sec).ok()?.iter().map(|p| p.0).sum()
        }
        SectionKind::Other if sec.raw.name == "SHELL GENERAL SECTION" || sec.raw.name == "MEMBRANE SECTION" => {
            sec.raw.first_floats().ok()?.first().copied()?
        }
        _ => return None,
    };
    if !(t > 0.0) {
        return None;
    }
    let off = rustyfem::elements::shell::section::offset(sec).unwrap_or(0.0);
    Some((t, -off * t))
}

/// Packs the data:
/// `u32` nseg, then per line segment `i32` profile (−1 = none) and n1 (f32 ×3);
/// `u32` nprof, then per profile `u32` npoly and per polygon `u32` npts and the points (f32 ×2);
/// `u32` nel, then per element thickness, midsurface position and normal (f32 ×5; thickness 0 = not a shell);
/// finally `u32` 1 if the model has 3D elements (beams, shells, solids in space), otherwise 0.
pub fn pack(model: &FeModel, mesh: &DisplayMesh) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::new();
    let u32b = |b: &mut Vec<u8>, x: u32| b.extend(x.to_le_bytes());
    let f32b = |b: &mut Vec<u8>, x: f64| b.extend((x as f32).to_le_bytes());
    let mut profiles: Vec<Vec<Poly>> = Vec::new();
    let mut by_section: HashMap<usize, i32> = HashMap::new();
    u32b(&mut b, mesh.line_elem.len() as u32);
    for (s, &e) in mesh.line_elem.iter().enumerate() {
        let el = &model.elements[e as usize];
        let fam = el.etype.family();
        let prof = match el.section {
            Some(si) if matches!(fam, Family::Beam | Family::Truss) => *by_section.entry(si).or_insert_with(|| match section_polys(&model.sections[si], fam) {
                Some(p) => {
                    profiles.push(p);
                    profiles.len() as i32 - 1
                }
                None => -1,
            }),
            _ => -1,
        };
        let (pa, pb) = (model.nodes[mesh.lines[2 * s] as usize], model.nodes[mesh.lines[2 * s + 1] as usize]);
        let t = unit(sub(pb, pa)).unwrap_or([1.0, 0.0, 0.0]);
        let n1a = match el.section.and_then(|si| model.sections[si].beam()) {
            Some(bs) if el.etype.dim() == 3 => bs.n1.unwrap_or([0.0, 0.0, -1.0]),
            _ => [0.0, 0.0, -1.0],
        };
        b.extend(prof.to_le_bytes());
        for x in segment_n1(t, n1a) {
            f32b(&mut b, x);
        }
    }
    u32b(&mut b, profiles.len() as u32);
    for p in &profiles {
        u32b(&mut b, p.len() as u32);
        for poly in p {
            u32b(&mut b, poly.len() as u32);
            for q in poly {
                f32b(&mut b, q[0]);
                f32b(&mut b, q[1]);
            }
        }
    }
    u32b(&mut b, model.elements.len() as u32);
    for el in &model.elements {
        let fam = el.etype.family();
        let shell = if matches!(fam, Family::Shell | Family::Membrane) && el.etype.dim() == 3 {
            el.section.and_then(|si| shell_thickness(&model.sections[si]))
        } else {
            None
        };
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
        let normal = match x.len() {
            n if n >= 4 && n != 6 => unit(cross(sub(x[2], x[0]), sub(x[3], x[1]))),
            n if n >= 3 => unit(cross(sub(x[1], x[0]), sub(x[2], x[0]))),
            _ => None,
        };
        match (shell, normal) {
            (Some((t, z)), Some(nv)) => {
                for v in [t, z, nv[0], nv[1], nv[2]] {
                    f32b(&mut b, v);
                }
            }
            _ => {
                for _ in 0..5 {
                    f32b(&mut b, 0.0);
                }
            }
        }
    }
    u32b(&mut b, model.elements.iter().any(|e| e.etype.dim() == 3) as u32);
    b
}
