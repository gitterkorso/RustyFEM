//! Nodal normals (area normals): averaged shell normals
//! and beam normals (n2) and `*NORMAL`.
//!
//! A normal is computed per element and node from the element's own geometry
//! (shells: the surface normal at the node, beams: n2 = t × n1 at the node). Where several elements
//! meet, the mean of their normals is used if all differ pairwise by less than
//! 20°; otherwise each element keeps its own normal (crease). Example
//! with S4R5: a kink of 18° and 15° gives a common normal
//! (five degrees of freedom at the node), 22° and a fan with 12° between
//! neighbors (24° between the outer ones) gives the elements' own normals (six
//! degrees of freedom). A normal from `*NORMAL` (element, node) applies only to that
//! element at that node and takes precedence over the mean.
//!
//! The normals are computed when the file has been read (`apply`, called from `inp::dispatch`)
//! and stored in a registry per model; the sections of the affected elements get
//! the parameter `_NORMALS` with the registry number, so that the model can be cloned.
//! Elements that get no normal other than their own are not stored, so their
//! matrices are unchanged. The elements fetch the normals with `of(ctx)`.

use super::api::{ElemCtx, ElementType};
use super::common::{cross, dot, norm, scale, sub};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::{FemError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub static TYPES: &[&dyn ElementType] = &[];

/// The parameter on the section that points to the registry table.
pub const PARAM: &str = "_NORMALS";

/// The limit angle for averaging (default 20°).
pub const AVERAGE_ANGLE_DEG: f64 = 20.0;

type Table = HashMap<usize, Vec<[f64; 3]>>;

fn registry() -> &'static Mutex<Vec<Arc<Table>>> {
    static R: OnceLock<Mutex<Vec<Arc<Table>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// Group of elements whose normals are averaged with each other.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// Shells whose formulation uses nodal normals.
    Shell,
    /// Beams in space (the normal = n2).
    Beam,
}

/// Element types that use nodal normals: S8R (with
/// averaged normals), and
/// (area beams3) B23/B33 that become curved (`beam_eb_curved.rs`) and B31 that
/// become twisted (`beam_twist.rs`).
/// For S9R5/S8R5 and S4R5 the nodal normals also affect the membrane
/// hourglass stiffness and the transverse shear, and for the beams B31/B32 an
/// averaged n2 gives a twist of the section along the element and for B23/B33
/// a curved beam; for now they keep their own normals (`own` can already
/// compute them).
pub fn group(name: &str) -> Option<Group> {
    match name {
        "S8R" => Some(Group::Shell),
        // area beams3: cubic beams become curved (elements/beam_eb_curved.rs)
        "B23" | "B33" => Some(Group::Beam),
        // area beams3: twisted B31 (elements/beam_twist.rs)
        "B31" => Some(Group::Beam),
        _ => None,
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// The normals of an element from the registry (`None` = the element's own).
pub fn of(ctx: &ElemCtx) -> Option<Vec<[f64; 3]>> {
    let id: usize = ctx.section?.raw.param(PARAM)?.parse().ok()?;
    let t = registry().lock().ok()?.get(id)?.clone();
    t.get(&ctx.index).cloned()
}

/// The shell normals at nodes 0..n from 8-node serendipity geometry (S8R).
fn shell8(x: &[[f64; 3]]) -> Vec<[f64; 3]> {
    super::shell2::s8r::nodal_normals(x).to_vec()
}

/// The normals at corner and midside nodes for 9-node geometry (S9R5, S8R5 with
/// the center point from the serendipity surface).
fn shell9(x: &[[f64; 3]], internal: bool) -> Vec<[f64; 3]> {
    let mut x: Vec<[f64; 3]> = x.to_vec();
    if internal {
        x.truncate(8);
        x.push([0, 1, 2].map(|c| -0.25 * (0..4).map(|i| x[i][c]).sum::<f64>() + 0.5 * (4..8).map(|i| x[i][c]).sum::<f64>()));
    }
    let nat = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]];
    let n = if internal { 8 } else { 9 };
    nat[..n]
        .iter()
        .map(|&[s, t]| {
            let (_, d) = super::shell3::q9::shape9(s, t);
            let a = [0, 1].map(|al| [0, 1, 2].map(|c| (0..9).map(|i| d[al][i] * x[i][c]).sum::<f64>()));
            unit(cross(a[0], a[1]))
        })
        .collect()
}

/// Six-node triangle (STRI65): the normal from the quadratic geometry at the nodes.
fn shell6(x: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let nat = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]];
    nat.iter()
        .map(|&[r, s]| {
            let l = 1.0 - r - s;
            // dN/dr, dN/ds for nodes 1..6 (corner 1 = (0,0), 2 = (1,0), 3 = (0,1))
            let dr = [-(4.0 * l - 1.0), 4.0 * r - 1.0, 0.0, 4.0 * (l - r), 4.0 * s, -4.0 * s];
            let ds = [-(4.0 * l - 1.0), 0.0, 4.0 * s - 1.0, -4.0 * r, 4.0 * r, 4.0 * (l - s)];
            let a = [0, 1, 2].map(|c| (0..6).map(|i| dr[i] * x[i][c]).sum::<f64>());
            let b = [0, 1, 2].map(|c| (0..6).map(|i| ds[i] * x[i][c]).sum::<f64>());
            unit(cross(a, b))
        })
        .collect()
}

/// The beam tangent at the nodes (2 or 3 nodes along the beam).
pub fn beam_tangents(x: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let nn = x.len();
    let xis: &[f64] = if nn == 2 { &[-1.0, 1.0] } else { &[-1.0, 0.0, 1.0] };
    xis.iter()
        .map(|&xi| {
            let (_, dn) = super::beam_timoshenko::shape(nn, xi);
            unit([0, 1, 2].map(|c| (0..nn).map(|i| dn[i] * x[i][c]).sum::<f64>()))
        })
        .collect()
}

/// The element's own normals at its (outer) nodes.
fn own(m: &FeModel, e: usize) -> Option<(Group, Vec<[f64; 3]>)> {
    let el = &m.elements[e];
    let name = el.etype.name();
    let g = group(name)?;
    let nc = el.etype.n_connectivity().min(el.nodes.len());
    let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
    let v = match (g, name) {
        (Group::Shell, "S8R") if nc == 8 => shell8(&x[..8]),
        (Group::Shell, "S9R5") if nc == 9 => shell9(&x[..9], false),
        (Group::Shell, "S8R5") if nc == 8 => shell9(&x[..8], true),
        (Group::Shell, "STRI65") if nc == 6 => shell6(&x[..6]),
        (Group::Beam, _) if nc == 2 || nc == 3 => {
            let n1 = el.section.and_then(|s| m.sections[s].beam()).and_then(|b| b.n1);
            beam_tangents(&x[..nc])
                .into_iter()
                .map(|t| super::beam_timoshenko::beam_axes(t, n1, 3, el.label, name).map(|a| a[2]).unwrap_or([0.0; 3]))
                .collect()
        }
        _ => return None,
    };
    if v.iter().any(|n| !(norm(*n) > 0.5)) {
        return None;
    }
    Some((g, v))
}

/// Normals from `*NORMAL` (stored by `inp/keywords/normals.rs`): (element, node) → direction.
fn explicit(m: &FeModel) -> Result<HashMap<(usize, usize), [f64; 3]>> {
    let mut out = HashMap::new();
    for kw in m.extra.iter().filter(|k| k.name == "NORMAL") {
        let place = crate::interactions::analytical::placement_of(kw);
        for d in &kw.data {
            let f = &d.fields;
            let inst_name = match kw.param("_INSTANCE") {
                Some(i) => i.to_uppercase(),
                None => f[0].clone(),
            };
            let Some(inst) = m.instance_index(&inst_name) else {
                return kw.err_at(d.line, format!("the instance '{inst_name}' does not exist"));
            };
            let el: i64 = f[1].parse().map_err(|_| FemError("*NORMAL: invalid element number".into()))?;
            let nd: i64 = f[2].parse().map_err(|_| FemError("*NORMAL: invalid node number".into()))?;
            let (Some(e), Some(n)) = (m.element(inst, el), m.node(inst, nd)) else {
                return kw.err_at(d.line, format!("*NORMAL: the element {el} or the node {nd} does not exist"));
            };
            if !m.elements[e].nodes.contains(&n) {
                return kw.err_at(d.line, format!("*NORMAL: the node {nd} does not belong to element {el}"));
            }
            let v: Vec<f64> = f[3..].iter().map(|s| s.parse().unwrap_or(0.0)).collect();
            let mut nv = [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)];
            if let Some(p) = &place {
                nv = p.direction(nv);
            }
            if !(norm(nv) > 0.0) {
                return kw.err_at(d.line, "*NORMAL: the normal has zero length");
            }
            out.insert((e, n), unit(nv));
        }
    }
    Ok(out)
}

/// Averaged normals per element and node: element → normals at
/// the element's outer nodes, only for elements where some normal differs from
/// the element's own.
pub fn compute(m: &FeModel) -> Result<Table> {
    let exp = explicit(m)?;
    let mut owns: HashMap<usize, (Group, Vec<[f64; 3]>)> = HashMap::new();
    let mut at_node: HashMap<usize, Vec<(usize, usize)>> = HashMap::new(); // node → (element, local node)
    for e in 0..m.elements.len() {
        if let Some(o) = own(m, e) {
            for (k, &n) in m.elements[e].nodes.iter().take(o.1.len()).enumerate() {
                at_node.entry(n).or_default().push((e, k));
            }
            owns.insert(e, o);
        }
    }
    let cmin = AVERAGE_ANGLE_DEG.to_radians().cos();
    let mut fin: Table = HashMap::new();
    for (&n, list) in &at_node {
        for &(e, k) in list {
            let (g, ref v) = owns[&e];
            let mine = v[k];
            let result = if let Some(x) = exp.get(&(e, n)) {
                *x
            } else {
                // the group: elements (same group, without their own *NORMAL at the node) within
                // 20° (elements with *NORMAL are not counted, see
                // tests/cases/normals/s8r_vikt_10_normal)
                let peers: Vec<(usize, [f64; 3])> = list
                    .iter()
                    .filter(|&&(f, _)| owns[&f].0 == g && !exp.contains_key(&(f, n)))
                    .map(|&(f, kf)| (f, owns[&f].1[kf]))
                    .collect();
                let within = |a: [f64; 3]| -> Vec<usize> { peers.iter().filter(|(_, b)| dot(a, *b) >= cmin).map(|(f, _)| *f).collect() };
                let mut grp = within(mine);
                grp.sort_unstable();
                grp.dedup();
                // mean only if the group is closed (all pairwise within 20°)
                let closed = peers.iter().filter(|(f, _)| grp.contains(f)).all(|(_, b)| {
                    let mut g2 = within(*b);
                    g2.sort_unstable();
                    g2.dedup();
                    g2 == grp
                });
                if closed && grp.len() > 1 {
                    let mut s = [0.0; 3];
                    for (f, b) in &peers {
                        if grp.contains(f) {
                            s = [s[0] + b[0], s[1] + b[1], s[2] + b[2]];
                        }
                    }
                    if norm(s) > 0.0 { unit(s) } else { mine }
                } else {
                    mine
                }
            };
            if norm(sub(result, mine)) > 1e-13 {
                fin.entry(e).or_insert_with(|| v.clone())[k] = result;
            }
        }
    }
    Ok(fin)
}

/// Computes the normals and marks the sections (called when the file has been read).
pub fn apply(m: &mut FeModel) -> Result<()> {
    if !m.elements.iter().any(|e| group(e.etype.name()).is_some()) {
        if m.extra.iter().any(|k| k.name == "NORMAL") {
            m.warnings.push("*NORMAL ignored: the model has no elements that use nodal normals".into());
        }
        return Ok(());
    }
    let ignored = m
        .extra
        .iter()
        .filter(|k| k.name == "NORMAL")
        .flat_map(|k| k.data.iter().map(move |d| (k, d)))
        .filter(|(k, d)| {
            let inst = k.param("_INSTANCE").map(|i| i.to_uppercase()).unwrap_or_else(|| d.fields[0].clone());
            let el = m.instance_index(&inst).and_then(|i| d.fields[1].parse().ok().and_then(|l| m.element(i, l)));
            el.is_none_or(|e| group(m.elements[e].etype.name()).is_none())
        })
        .count();
    if ignored > 0 {
        m.warnings.push(format!("*NORMAL: {ignored} normals apply to elements that do not use nodal normals (only S8R) and are ignored"));
    }
    let t = compute(m)?;
    if t.is_empty() {
        return Ok(());
    }
    let secs: Vec<usize> = {
        let mut s: Vec<usize> = t.keys().filter_map(|&e| m.elements[e].section).collect();
        s.sort_unstable();
        s.dedup();
        s
    };
    let id = {
        let mut r = registry().lock().map_err(|_| FemError("the normal registry is locked".into()))?;
        r.push(Arc::new(t));
        r.len() - 1
    };
    for s in secs {
        let raw: &mut RawKeyword = &mut m.sections[s].raw;
        raw.params.retain(|(k, _)| k != PARAM);
        raw.params.push((PARAM.into(), id.to_string()));
    }
    Ok(())
}
