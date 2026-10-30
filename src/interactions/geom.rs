//! Shared geometry for constraints and contact: surfaces (element faces and
//! nodes), shape functions on surfaces, projection of a point onto a surface
//! and integration points on surfaces.
//!
//! A facet (`Facet`) is an element edge (2D) or element face (3D) with nodes in
//! .inp format order (`ElementType::faces`). The normal always points out of
//! the element (away from the element centroid), regardless of node order.

use crate::elements::common::{cross, dot, norm, scale, sub};
use crate::fem::dof::{self, Dof, NodeVec, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::{FeModel, RawKeyword, Surface};
use crate::model::{fail, Result};

/// Active slots per node from the elements (before the interactions).
pub fn element_active(m: &FeModel) -> Vec<u16> {
    let mut a = vec![0u16; m.nodes.len()];
    for e in &m.elements {
        for (k, &n) in e.nodes.iter().enumerate() {
            for &d in e.etype.node_dofs(k) {
                if let Some(s) = dof::slot(d) {
                    a[n] |= 1 << s;
                }
            }
        }
    }
    a
}

pub fn has(active: &[u16], n: usize, d: Dof) -> bool {
    active[n] & (1 << dof::slot_of(d)) != 0
}

/// Translations and rotations in the model dimension.
pub fn trans_dofs(dim: usize) -> &'static [Dof] {
    if dim == 2 { &[U1, U2] } else { &[U1, U2, U3] }
}

pub fn rot_dofs(dim: usize) -> &'static [Dof] {
    if dim == 2 { &[UR3] } else { &[UR1, UR2, UR3] }
}

/// An element face.
#[derive(Debug, Clone)]
pub struct Facet {
    pub elem: usize,
    pub nodes: Vec<usize>,
    /// Element centroid (for the normal direction).
    pub inside: [f64; 3],
    pub dim: usize,
    /// Thickness for planar elements (area per unit length), otherwise 1.
    pub thick: f64,
}

/// Face label "S3" → index 2.
fn face_index(name: &str) -> Option<usize> {
    name.strip_prefix('S').and_then(|n| n.parse::<usize>().ok()).filter(|&i| i >= 1).map(|i| i - 1)
}

/// The element faces of a *SURFACE (TYPE=ELEMENT).
pub fn facets(m: &FeModel, s: &Surface, what: &str) -> Result<Vec<Facet>> {
    let mut out = Vec::new();
    for (e, f) in &s.faces {
        let el = &m.elements[*e];
        let faces = el.etype.faces();
        // SPOS/SNEG: the single face of a surface element (rigid element, shell)
        // with the normal from the node order (+1) or opposite (−1)
        let side = match f.as_str() {
            "SPOS" if faces.len() == 1 => 1.0,
            "SNEG" if faces.len() == 1 => -1.0,
            _ => 0.0,
        };
        let Some(i) = face_index(f).or((side != 0.0).then_some(0)).filter(|&i| i < faces.len()) else {
            return fail(format!("{what}: face {f} does not exist for {} ({})", m.elem_name(*e), s.source));
        };
        let nodes: Vec<usize> = faces[i].iter().map(|&k| el.nodes[k]).collect();
        let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
        let mut c = crate::elements::common::centroid(&xs);
        let dim = el.etype.dim();
        if side != 0.0 {
            // the "inside" point is placed on the negative side so that the normal
            // becomes ±n: right-hand rule in 3D, (node 1 → node 2) rotated 90° CCW in 2D
            let xf: Vec<[f64; 3]> = nodes.iter().map(|&n| m.nodes[n]).collect();
            let fc = crate::elements::common::centroid(&xf);
            let (n, size) = if dim == 2 {
                let t = sub(xf[1], xf[0]);
                ([-t[1], t[0], 0.0], norm(t))
            } else {
                let nn = cross(sub(xf[1], xf[0]), sub(xf[xf.len() - 1], xf[0]));
                (nn, norm(nn).sqrt())
            };
            let ln = norm(n).max(1e-300);
            c = [0, 1, 2].map(|k| fc[k] - side * n[k] / ln * size);
        }
        let thick = match (dim, el.section) {
            (2, Some(si)) => m.sections[si].solid_thickness(),
            _ => 1.0,
        };
        out.push(Facet { elem: *e, nodes, inside: c, dim, thick });
    }
    Ok(out)
}

/// Surface by name (upper case, possibly with instance prefix).
pub fn surface<'a>(m: &'a FeModel, kw: &RawKeyword, name: &str) -> Result<&'a Surface> {
    let up = name.trim().to_uppercase();
    match m.surfaces.get(&up) {
        Some(s) => Ok(s),
        None => kw.err(format!("surface '{}' does not exist", name.trim())),
    }
}

/// Nodes of a surface (the element faces' nodes or the node surface's nodes), sorted.
pub fn surface_nodes(m: &FeModel, s: &Surface) -> Result<Vec<usize>> {
    let mut v: Vec<usize> = if s.kind == "NODE" {
        s.nodes.iter().map(|p| p.0).collect()
    } else {
        facets(m, s, "surface")?.into_iter().flat_map(|f| f.nodes).collect()
    };
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

/// Node set or surface (for *COUPLING SURFACE= and *TIE with node surfaces).
pub fn nodes_of(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<Vec<usize>> {
    let up = name.trim().to_uppercase();
    if let Some(s) = m.surfaces.get(&up) {
        return surface_nodes(m, s);
    }
    if let Some(s) = m.nsets.get(&up) {
        return Ok(s.clone());
    }
    kw.err(format!("'{}' is neither a surface nor a node set", name.trim()))
}

/// A node reference: number (in the model's only instance or ASSEMBLY), "INST.number" or node set.
pub fn node_ref(m: &FeModel, kw: &RawKeyword, token: &str) -> Result<Vec<usize>> {
    let t = token.trim();
    let up = t.to_uppercase();
    if let Some(l) = crate::inp::parse_label(t) {
        for inst in [crate::fem::model::FLAT_INSTANCE, crate::fem::model::ASSEMBLY_INSTANCE] {
            if let Some(i) = m.instance_index(inst) {
                if let Some(n) = m.node(i, l) {
                    return Ok(vec![n]);
                }
            }
        }
        if m.instances.len() == 1 {
            if let Some(n) = m.node(0, l) {
                return Ok(vec![n]);
            }
        }
        return kw.err(format!("node {t} does not exist"));
    }
    if let Some((a, b)) = up.rsplit_once('.') {
        if let (Some(l), Some(i)) = (crate::inp::parse_label(b), m.instance_index(a)) {
            if let Some(n) = m.node(i, l) {
                return Ok(vec![n]);
            }
        }
    }
    if let Some(s) = m.nsets.get(&up) {
        return Ok(s.clone());
    }
    kw.err(format!("node set '{t}' does not exist"))
}

/// Element reference: number, "INST.number" or element set.
pub fn elem_ref(m: &FeModel, kw: &RawKeyword, token: &str) -> Result<Vec<usize>> {
    let t = token.trim();
    let up = t.to_uppercase();
    if let Some(s) = m.elsets.get(&up) {
        return Ok(s.clone());
    }
    if let Some(l) = crate::inp::parse_label(t) {
        for i in 0..m.instances.len() as u32 {
            if let Some(e) = m.element(i, l) {
                return Ok(vec![e]);
            }
        }
    }
    if let Some((a, b)) = up.rsplit_once('.') {
        if let (Some(l), Some(i)) = (crate::inp::parse_label(b), m.instance_index(a)) {
            if let Some(e) = m.element(i, l) {
                return Ok(vec![e]);
            }
        }
    }
    kw.err(format!("element set '{t}' does not exist"))
}

/// Shape functions and derivatives on a facet with `nn` nodes in dimension `dim`
/// (2D: edges with 2 or 3 nodes; 3D: triangles 3/6, quadrilaterals 4/8).
/// Returns (N, dN/dξ, dN/dη).
pub fn shape(dim: usize, nn: usize, xi: [f64; 2]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (r, s) = (xi[0], xi[1]);
    match (dim, nn) {
        (2, 2) => (vec![0.5 * (1.0 - r), 0.5 * (1.0 + r)], vec![-0.5, 0.5], vec![0.0; 2]),
        (2, 3) => (
            vec![0.5 * r * (r - 1.0), 0.5 * r * (r + 1.0), 1.0 - r * r],
            vec![r - 0.5, r + 0.5, -2.0 * r],
            vec![0.0; 3],
        ),
        (_, 3) => (vec![1.0 - r - s, r, s], vec![-1.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0]),
        (_, 4) => {
            let n = vec![
                0.25 * (1.0 - r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 - s),
                0.25 * (1.0 + r) * (1.0 + s),
                0.25 * (1.0 - r) * (1.0 + s),
            ];
            let dr = vec![-0.25 * (1.0 - s), 0.25 * (1.0 - s), 0.25 * (1.0 + s), -0.25 * (1.0 + s)];
            let ds = vec![-0.25 * (1.0 - r), -0.25 * (1.0 + r), 0.25 * (1.0 + r), 0.25 * (1.0 - r)];
            (n, dr, ds)
        }
        (_, 6) => {
            let t = 1.0 - r - s;
            (
                vec![t * (2.0 * t - 1.0), r * (2.0 * r - 1.0), s * (2.0 * s - 1.0), 4.0 * r * t, 4.0 * r * s, 4.0 * s * t],
                vec![1.0 - 4.0 * t, 4.0 * r - 1.0, 0.0, 4.0 * (t - r), 4.0 * s, -4.0 * s],
                vec![1.0 - 4.0 * t, 0.0, 4.0 * s - 1.0, -4.0 * r, 4.0 * r, 4.0 * (t - s)],
            )
        }
        (_, 8) => {
            let c = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let mut n = vec![0.0; 8];
            let mut dr = vec![0.0; 8];
            let mut ds = vec![0.0; 8];
            for (i, &(a, b)) in c.iter().enumerate() {
                n[i] = 0.25 * (1.0 + a * r) * (1.0 + b * s) * (a * r + b * s - 1.0);
                dr[i] = 0.25 * a * (1.0 + b * s) * (2.0 * a * r + b * s);
                ds[i] = 0.25 * b * (1.0 + a * r) * (a * r + 2.0 * b * s);
            }
            // midside nodes 5–8 on the edges s=-1, r=1, s=1, r=-1
            n[4] = 0.5 * (1.0 - r * r) * (1.0 - s);
            dr[4] = -r * (1.0 - s);
            ds[4] = -0.5 * (1.0 - r * r);
            n[5] = 0.5 * (1.0 + r) * (1.0 - s * s);
            dr[5] = 0.5 * (1.0 - s * s);
            ds[5] = -(1.0 + r) * s;
            n[6] = 0.5 * (1.0 - r * r) * (1.0 + s);
            dr[6] = -r * (1.0 + s);
            ds[6] = 0.5 * (1.0 - r * r);
            n[7] = 0.5 * (1.0 - r) * (1.0 - s * s);
            dr[7] = -0.5 * (1.0 - s * s);
            ds[7] = -(1.0 - r) * s;
            (n, dr, ds)
        }
        _ => (vec![1.0 / nn as f64; nn], vec![0.0; nn], vec![0.0; nn]),
    }
}

/// Is the facet a triangle (parameter domain 0 ≤ ξ, η, ξ+η ≤ 1)?
pub fn is_tri(dim: usize, nn: usize) -> bool {
    dim == 3 && (nn == 3 || nn == 6)
}

/// Point, tangents, normal (outward, unit vector) and surface element
/// (length/area per parameter unit) at the point ξ on the facet with nodal
/// coordinates `x`.
pub struct FacetPoint {
    pub n: Vec<f64>,
    pub x: [f64; 3],
    pub t1: [f64; 3],
    pub t2: [f64; 3],
    pub normal: [f64; 3],
    pub jac: f64,
}

pub fn eval(f: &Facet, x: &[[f64; 3]], xi: [f64; 2]) -> FacetPoint {
    let (n, dr, ds) = shape(f.dim, x.len(), xi);
    let mut p = [0.0; 3];
    let mut t1 = [0.0; 3];
    let mut t2 = [0.0; 3];
    for (k, xk) in x.iter().enumerate() {
        for i in 0..3 {
            p[i] += n[k] * xk[i];
            t1[i] += dr[k] * xk[i];
            t2[i] += ds[k] * xk[i];
        }
    }
    let (mut normal, jac) = if f.dim == 2 {
        let l = norm(t1);
        ([t1[1] / l, -t1[0] / l, 0.0], l * f.thick)
    } else {
        let c = cross(t1, t2);
        let a = norm(c);
        (scale(c, 1.0 / a), a)
    };
    // outward: away from the element interior
    if dot(normal, sub(p, f.inside)) < 0.0 {
        normal = scale(normal, -1.0);
    }
    FacetPoint { n, x: p, t1, t2, normal, jac }
}

/// Nodal coordinates of the facet: reference + displacement (if `u` is given).
pub fn coords(m: &FeModel, f: &Facet, u: Option<&[NodeVec]>) -> Vec<[f64; 3]> {
    f.nodes
        .iter()
        .map(|&n| {
            let x = m.nodes[n];
            match u {
                Some(u) => [x[0] + u[n][0], x[1] + u[n][1], x[2] + u[n][2]],
                None => x,
            }
        })
        .collect()
}

/// Closest point on the facet to `p`: (ξ, distance, ξ inside the facet within the tolerance `tol`).
pub fn project(f: &Facet, x: &[[f64; 3]], p: [f64; 3]) -> ([f64; 2], f64, bool) {
    let tri = is_tri(f.dim, x.len());
    let mut xi = if tri { [1.0 / 3.0, 1.0 / 3.0] } else { [0.0, 0.0] };
    for _ in 0..30 {
        let fp = eval(f, x, xi);
        let d = sub(p, fp.x);
        if f.dim == 2 {
            let g = dot(d, fp.t1);
            let h = dot(fp.t1, fp.t1);
            let step = g / h;
            xi[0] += step;
            if step.abs() < 1e-13 {
                break;
            }
        } else {
            // Gauss–Newton on the 2×2 system
            let a11 = dot(fp.t1, fp.t1);
            let a12 = dot(fp.t1, fp.t2);
            let a22 = dot(fp.t2, fp.t2);
            let b1 = dot(d, fp.t1);
            let b2 = dot(d, fp.t2);
            let det = a11 * a22 - a12 * a12;
            if det.abs() < 1e-300 {
                break;
            }
            let s1 = (a22 * b1 - a12 * b2) / det;
            let s2 = (a11 * b2 - a12 * b1) / det;
            xi[0] += s1;
            xi[1] += s2;
            if s1.abs() + s2.abs() < 1e-13 {
                break;
            }
        }
        // keep the iteration close to the facet
        for v in xi.iter_mut() {
            *v = v.clamp(-2.0, 2.0);
        }
    }
    let inside = |xi: [f64; 2], tol: f64| -> bool {
        if f.dim == 2 {
            xi[0].abs() <= 1.0 + tol
        } else if tri {
            xi[0] >= -tol && xi[1] >= -tol && xi[0] + xi[1] <= 1.0 + tol
        } else {
            xi[0].abs() <= 1.0 + tol && xi[1].abs() <= 1.0 + tol
        }
    };
    let ok = inside(xi, 1e-6);
    // outside: clip to the facet boundary for the distance
    let mut xc = xi;
    if !ok {
        if f.dim == 2 || !tri {
            xc[0] = xc[0].clamp(-1.0, 1.0);
            xc[1] = xc[1].clamp(-1.0, 1.0);
        } else {
            xc[0] = xc[0].max(0.0);
            xc[1] = xc[1].max(0.0);
            let s = xc[0] + xc[1];
            if s > 1.0 {
                xc[0] /= s;
                xc[1] /= s;
            }
        }
    }
    let fp = eval(f, x, xc);
    (if ok { xi } else { xc }, norm(sub(p, fp.x)), ok)
}

/// Integration points (ξ, weight) on a facet with `nn` nodes.
pub fn gauss(dim: usize, nn: usize, order: usize) -> Vec<([f64; 2], f64)> {
    let g1: Vec<(f64, f64)> = match order {
        1 => vec![(0.0, 2.0)],
        2 => {
            let a = 1.0 / 3f64.sqrt();
            vec![(-a, 1.0), (a, 1.0)]
        }
        3 => {
            let a = (0.6f64).sqrt();
            vec![(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
        }
        _ => {
            let a = (3.0 / 7.0 - 2.0 / 7.0 * (6.0f64 / 5.0).sqrt()).sqrt();
            let b = (3.0 / 7.0 + 2.0 / 7.0 * (6.0f64 / 5.0).sqrt()).sqrt();
            let wa = (18.0 + 30f64.sqrt()) / 36.0;
            let wb = (18.0 - 30f64.sqrt()) / 36.0;
            vec![(-b, wb), (-a, wa), (a, wa), (b, wb)]
        }
    };
    if dim == 2 {
        return g1.iter().map(|&(r, w)| ([r, 0.0], w)).collect();
    }
    if is_tri(dim, nn) {
        // collapsed square (Duffy) with the same order
        let mut v = Vec::new();
        for &(a, wa) in &g1 {
            for &(b, wb) in &g1 {
                let u = 0.5 * (1.0 + a);
                let s = 0.5 * (1.0 + b);
                v.push(([u * (1.0 - s), u * s], wa * wb * 0.25 * u));
            }
        }
        return v;
    }
    let mut v = Vec::new();
    for &(a, wa) in &g1 {
        for &(b, wb) in &g1 {
            v.push(([a, b], wa * wb));
        }
    }
    v
}

/// ∫ Nₖ dA for the facet nodes (reference geometry).
pub fn nodal_areas(f: &Facet, x: &[[f64; 3]]) -> Vec<f64> {
    let mut a = vec![0.0; x.len()];
    for (xi, w) in gauss(f.dim, x.len(), 3) {
        let fp = eval(f, x, xi);
        for k in 0..x.len() {
            a[k] += fp.n[k] * w * fp.jac;
        }
    }
    a
}

/// 3×3 matrix times vector.
pub fn mv(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * v[0] + a[i][1] * v[1] + a[i][2] * v[2])
}

/// Integration points on the slave facet for mortar integrals against another
/// surface. In 2D the edge is split at the projections of the points `breaks`
/// (the master surface nodes), so that piecewise polynomial integrands are
/// integrated exactly (segmentation). In 3D the facet is split into
/// `n_sub` × `n_sub` subcells.
pub fn mortar_points(f: &Facet, x: &[[f64; 3]], breaks: &[[f64; 3]], n_sub: usize) -> Vec<([f64; 2], f64)> {
    let g = gauss(2, 2, 3);
    if f.dim == 2 {
        let mut cuts = vec![-1.0, 1.0];
        for &p in breaks {
            let (xi, _, _) = project(f, x, p);
            if xi[0] > -1.0 + 1e-10 && xi[0] < 1.0 - 1e-10 {
                cuts.push(xi[0]);
            }
        }
        cuts.sort_by(|a, b| a.total_cmp(b));
        cuts.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let mut v = Vec::new();
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            for &(xi, wg) in &g {
                v.push(([0.5 * (a + b) + 0.5 * (b - a) * xi[0], 0.0], wg * 0.5 * (b - a)));
            }
        }
        return v;
    }
    let g2 = gauss(3, 4, 2);
    let n = n_sub.max(1);
    let h = 2.0 / n as f64;
    let mut v = Vec::new();
    let tri = is_tri(f.dim, x.len());
    for i in 0..n {
        for j in 0..n {
            for &(xi, wg) in &g2 {
                let r = -1.0 + h * (i as f64 + 0.5 * (1.0 + xi[0]));
                let s = -1.0 + h * (j as f64 + 0.5 * (1.0 + xi[1]));
                let w = wg * 0.25 * h * h;
                if tri {
                    let u = 0.5 * (1.0 + r);
                    let t = 0.5 * (1.0 + s);
                    v.push(([u * (1.0 - t), u * t], w * 0.25 * u));
                } else {
                    v.push(([r, s], w));
                }
            }
        }
    }
    v
}

/// Coefficients of the element-wise dual (biorthogonal) basis on a facet:
/// Φ_A = Σ_B a_AB N_B with ∫ Φ_A N_B dA = δ_AB ∫ N_A dA (Wohlmuth 2000).
/// Returns (a row-major, ∫N_A dA).
pub fn dual_basis(f: &Facet, x: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let nn = x.len();
    let mut d = vec![0.0; nn * nn];
    let mut m = vec![0.0; nn];
    for (xi, w) in gauss(f.dim, nn, 4) {
        let fp = eval(f, x, xi);
        for a in 0..nn {
            m[a] += fp.n[a] * w * fp.jac;
            for b in 0..nn {
                d[a * nn + b] += fp.n[a] * fp.n[b] * w * fp.jac;
            }
        }
    }
    // A = diag(m) D⁻¹ (D symmetric)
    let dinv = invert(&d, nn);
    let mut a = vec![0.0; nn * nn];
    for i in 0..nn {
        for j in 0..nn {
            a[i * nn + j] = m[i] * dinv[i * nn + j];
        }
    }
    (a, m)
}

/// Inverse of a small dense matrix (Gauss–Jordan with pivoting).
pub fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().total_cmp(&m[j * n + c].abs())).unwrap();
        if p != c {
            for k in 0..n {
                m.swap(c * n + k, p * n + k);
                inv.swap(c * n + k, p * n + k);
            }
        }
        let d = m[c * n + c];
        if d == 0.0 {
            continue;
        }
        for k in 0..n {
            m[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * n + c];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[c * n + k];
                        inv[r * n + k] -= f * inv[c * n + k];
                    }
                }
            }
        }
    }
    inv
}
