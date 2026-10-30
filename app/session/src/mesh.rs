//! Visningsgeometri: yttre ytor av solider, skalens och de plana elementens
//! ytor, balkarnas och stängernas linjer. Andra ordningens ytor delas upp i
//! trianglar via mittnoderna.
//!
//! Alla index pekar på modellens noder (`FeModel::nodes`), så förskjutningar
//! och nodvärden kan läggas direkt på samma buffertar.

use rustyfem::fem::model::FeModel;
use std::collections::HashMap;

/// Hur ett element visas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Tet,
    Wedge,
    Hex,
    /// Plan yta (plana element, skal, membran, styva ytor).
    Face,
    Line,
    Point,
    None,
}

fn starts(name: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| name.starts_with(p))
}

/// Visningsform och antal noder som används (hörn + mittnoder).
pub fn shape(name: &str, n_conn: usize) -> (Shape, usize) {
    let solid = |n: usize| match n {
        4 | 10 => (Shape::Tet, n),
        6 | 15 => (Shape::Wedge, n),
        8 | 20 => (Shape::Hex, n),
        27 => (Shape::Hex, 20),
        _ => (Shape::None, 0),
    };
    let face = |n: usize| match n {
        3 | 4 | 6 | 8 => (Shape::Face, n),
        9 => (Shape::Face, 8),
        _ => (Shape::None, 0),
    };
    let line = |n: usize| match n {
        2 | 3 => (Shape::Line, n),
        1 => (Shape::Point, 1),
        _ => (Shape::None, 0),
    };
    if starts(name, &["SPRING", "DASHPOT", "CONN", "JOINT"]) {
        return line(n_conn.min(2));
    }
    if starts(name, &["MASS", "ROTARYI", "HEATCAP"]) {
        return (Shape::Point, 1);
    }
    if starts(name, &["SAX", "DSAX", "RAX", "R2D", "B2", "B3", "PIPE", "T2D", "T3D", "FRAME", "DC1D", "GAPUNI"]) {
        return line(n_conn);
    }
    if starts(name, &["SC"]) {
        return solid(n_conn);
    }
    if starts(name, &["C3D", "DC3D", "Q3D", "GK3D", "COH3D"]) {
        return solid(n_conn);
    }
    if starts(name, &["S", "M3D", "R3D", "DS", "CP", "CAX", "CGAX", "DC2D", "DCAX", "CPEG", "COH2D", "WARP"]) {
        return face(n_conn);
    }
    match n_conn {
        1 | 2 => line(n_conn),
        _ => (Shape::None, 0),
    }
}

/// Ytor (hörn i ordning) för solider.
fn solid_faces(s: Shape) -> &'static [&'static [usize]] {
    match s {
        Shape::Tet => &[&[0, 1, 2], &[0, 3, 1], &[1, 3, 2], &[2, 3, 0]],
        Shape::Wedge => &[&[0, 1, 2], &[3, 5, 4], &[0, 3, 4, 1], &[1, 4, 5, 2], &[2, 5, 3, 0]],
        Shape::Hex => &[&[0, 1, 2, 3], &[4, 7, 6, 5], &[0, 4, 5, 1], &[1, 5, 6, 2], &[2, 6, 7, 3], &[3, 7, 4, 0]],
        _ => &[],
    }
}

/// Kanter (hörn a, hörn b, mittnod) för andra ordningens element.
fn mid_edges(s: Shape, n: usize) -> &'static [(usize, usize, usize)] {
    match (s, n) {
        (Shape::Tet, 10) => &[(0, 1, 4), (1, 2, 5), (2, 0, 6), (0, 3, 7), (1, 3, 8), (2, 3, 9)],
        (Shape::Wedge, 15) => &[(0, 1, 6), (1, 2, 7), (2, 0, 8), (3, 4, 9), (4, 5, 10), (5, 3, 11), (0, 3, 12), (1, 4, 13), (2, 5, 14)],
        (Shape::Hex, 20) => &[
            (0, 1, 8),
            (1, 2, 9),
            (2, 3, 10),
            (3, 0, 11),
            (4, 5, 12),
            (5, 6, 13),
            (6, 7, 14),
            (7, 4, 15),
            (0, 4, 16),
            (1, 5, 17),
            (2, 6, 18),
            (3, 7, 19),
        ],
        (Shape::Face, 6) => &[(0, 1, 3), (1, 2, 4), (2, 0, 5)],
        (Shape::Face, 8) => &[(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)],
        _ => &[],
    }
}

fn mid_of(edges: &[(usize, usize, usize)], a: usize, b: usize) -> Option<usize> {
    edges.iter().find(|&&(x, y, _)| (x == a && y == b) || (x == b && y == a)).map(|e| e.2)
}

/// Ytgeometri för visning.
#[derive(Debug, Clone, Default)]
pub struct DisplayMesh {
    /// Trianglar (nodindex).
    pub tris: Vec<u32>,
    /// Element för varje triangel.
    pub tri_elem: Vec<u32>,
    /// Elementkanter på ytan (nodpar), utan triangelns diagonaler.
    pub edges: Vec<u32>,
    /// Balkar, stänger och fjädrar (nodpar).
    pub lines: Vec<u32>,
    pub line_elem: Vec<u32>,
    /// Punktelement (massor).
    pub points: Vec<u32>,
    pub point_elem: Vec<u32>,
    /// Noder som används av något visat element.
    pub used: Vec<bool>,
    /// Hörn- och mittnoder per element (för utjämning av elementfält).
    pub elem_nodes: Vec<Vec<u32>>,
}

/// En polygon (hörn och eventuella mittnoder) delad i trianglar och kanter.
fn emit_polygon(out: &mut DisplayMesh, corners: &[usize], mids: &[Option<usize>], elem: u32, edge_set: &mut HashMap<(u32, u32), ()>) {
    let n = corners.len();
    // kanter längs polygonen (via mittnoden om den finns)
    for i in 0..n {
        let (a, b) = (corners[i], corners[(i + 1) % n]);
        let mut add = |x: usize, y: usize| {
            let k = if x < y { (x as u32, y as u32) } else { (y as u32, x as u32) };
            if edge_set.insert(k, ()).is_none() {
                out.edges.extend([k.0, k.1]);
            }
        };
        match mids[i] {
            Some(m) => {
                add(a, m);
                add(m, b);
            }
            None => add(a, b),
        }
    }
    let mut tri = |a: usize, b: usize, c: usize| {
        out.tris.extend([a as u32, b as u32, c as u32]);
        out.tri_elem.push(elem);
    };
    if mids.iter().all(|m| m.is_some()) {
        // hörntrianglar + inre polygon av mittnoderna
        let m: Vec<usize> = mids.iter().map(|m| m.unwrap()).collect();
        for i in 0..n {
            tri(m[(i + n - 1) % n], corners[i], m[i]);
        }
        for i in 1..n - 1 {
            tri(m[0], m[i], m[i + 1]);
        }
    } else {
        for i in 1..n - 1 {
            tri(corners[0], corners[i], corners[i + 1]);
        }
    }
}

/// Extraherar visningsgeometrin för hela modellen.
pub fn extract(model: &FeModel) -> DisplayMesh {
    let mut out = DisplayMesh { used: vec![false; model.nodes.len()], ..Default::default() };
    let mut edge_set: HashMap<(u32, u32), ()> = HashMap::new();
    // solidytor räknas först så att inre ytor kan tas bort
    let mut solid_faces_map: HashMap<Vec<usize>, (usize, usize, u8)> = HashMap::new();
    let mut shapes = Vec::with_capacity(model.elements.len());
    for (ei, el) in model.elements.iter().enumerate() {
        let nc = el.etype.n_connectivity().min(el.nodes.len());
        let (s, n) = shape(el.etype.name(), nc);
        shapes.push((s, n));
        out.elem_nodes.push(el.nodes[..n.min(el.nodes.len())].iter().map(|&x| x as u32).collect());
        for &v in &el.nodes[..n.min(el.nodes.len())] {
            out.used[v] = true;
        }
        if matches!(s, Shape::Tet | Shape::Wedge | Shape::Hex) {
            for (fi, f) in solid_faces(s).iter().enumerate() {
                let mut key: Vec<usize> = f.iter().map(|&i| el.nodes[i]).collect();
                key.sort_unstable();
                solid_faces_map.entry(key).and_modify(|e| e.2 += 1).or_insert((ei, fi, 1));
            }
        }
    }
    let mut exterior: Vec<(usize, usize)> = solid_faces_map.values().filter(|v| v.2 == 1).map(|v| (v.0, v.1)).collect();
    exterior.sort_unstable();
    for (ei, fi) in exterior {
        let el = &model.elements[ei];
        let (s, n) = shapes[ei];
        let f = solid_faces(s)[fi];
        let me = mid_edges(s, n);
        let corners: Vec<usize> = f.iter().map(|&i| el.nodes[i]).collect();
        let mids: Vec<Option<usize>> = (0..f.len()).map(|i| mid_of(me, f[i], f[(i + 1) % f.len()]).map(|m| el.nodes[m])).collect();
        emit_polygon(&mut out, &corners, &mids, ei as u32, &mut edge_set);
    }
    for (ei, el) in model.elements.iter().enumerate() {
        let (s, n) = shapes[ei];
        match s {
            Shape::Face => {
                let nc = if n == 6 { 3 } else if n == 8 { 4 } else { n };
                let me = mid_edges(s, n);
                let corners: Vec<usize> = (0..nc).map(|i| el.nodes[i]).collect();
                let mids: Vec<Option<usize>> = (0..nc).map(|i| mid_of(me, i, (i + 1) % nc).map(|m| el.nodes[m])).collect();
                emit_polygon(&mut out, &corners, &mids, ei as u32, &mut edge_set);
            }
            Shape::Line => {
                for w in el.nodes[..n].windows(2) {
                    out.lines.extend([w[0] as u32, w[1] as u32]);
                    out.line_elem.push(ei as u32);
                }
            }
            Shape::Point => {
                out.points.push(el.nodes[0] as u32);
                out.point_elem.push(ei as u32);
            }
            _ => {}
        }
    }
    out
}

/// Packar nodkoordinater och visningsgeometri i en buffert:
/// `u32` antal (noder, trianglar, kanter, linjer, punkter), därefter
/// koordinater (f32 ×3), trianglar, triangelns element, kanter, linjer,
/// linjens element, punkter och punktens element (alla u32).
pub fn pack(model: &FeModel, m: &DisplayMesh) -> Vec<u8> {
    let mut b: Vec<u8> = Vec::with_capacity(20 + model.nodes.len() * 12 + (m.tris.len() + m.tri_elem.len() + m.edges.len() + m.lines.len() * 2) * 4);
    for c in [model.nodes.len(), m.tri_elem.len(), m.edges.len() / 2, m.line_elem.len(), m.point_elem.len()] {
        b.extend((c as u32).to_le_bytes());
    }
    for p in &model.nodes {
        for &x in p {
            b.extend((x as f32).to_le_bytes());
        }
    }
    for arr in [&m.tris, &m.tri_elem, &m.edges, &m.lines, &m.line_elem, &m.points, &m.point_elem] {
        for &x in arr.iter() {
            b.extend(x.to_le_bytes());
        }
    }
    b
}
