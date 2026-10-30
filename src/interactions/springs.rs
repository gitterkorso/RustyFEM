//! Springs (*SPRING for SPRING1/SPRING2/SPRINGA), dashpots (*DASHPOT),
//! point masses (*MASS) and rotary inertia (*ROTARY INERTIA).
//!
//! Data lines for *SPRING and *DASHPOT: first a line with the degrees of freedom
//! (SPRING1: one, SPRING2: two, SPRINGA: empty or omitted), then a line
//! with the stiffness/damping coefficient.
//!
//! - SPRING1: force k·u_i to ground.
//! - SPRING2: force k·(u_i(a) − u_j(b)) between selected degrees of freedom of two nodes.
//! - SPRINGA: axial spring along the line between the nodes. Small displacements:
//!   K = k n nᵀ. With NLGEOM the force k (l − L) acts along the current line and
//!   the tangent gets the geometric contribution k (l − L)/l (I − n nᵀ).

use super::geom;
use super::{Interaction, InteractionCtx, InteractionFactory, TangentSink};
use crate::elements::common::{norm, sub};
use crate::fem::dof::{self, Dof, NodeVec, U1};
use crate::fem::model::{FeModel, RawKeyword};
use crate::model::Result;

pub static FACTORIES: &[InteractionFactory] = &[
    InteractionFactory { keywords: &["DASHPOT"], build: dashpot },
    InteractionFactory { keywords: &["MASS"], build: mass },
    InteractionFactory { keywords: &["ROTARY INERTIA"], build: rotary },
    InteractionFactory { keywords: &["SPRING"], build: spring },
];

#[derive(Debug, Clone)]
enum Kind {
    /// (node, degree of freedom) to ground.
    One(usize, Dof),
    /// (node a, degree of freedom i, node b, degree of freedom j).
    Two(usize, Dof, usize, Dof),
    /// Axial between the nodes, with the unit vector a → b in the reference configuration.
    Axial(usize, usize, [f64; 3]),
}

pub struct Springs {
    name: String,
    items: Vec<(Kind, f64)>,
    dim: usize,
    /// Dashpots instead of springs (contribute only damping).
    damper: bool,
}

/// The elements in ELSET= and (degrees of freedom, coefficient) from the data lines.
fn parse(m: &FeModel, kw: &RawKeyword) -> Result<(Vec<usize>, Vec<Dof>, f64)> {
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= is not supported yet");
    }
    if kw.has("NONLINEAR") {
        return kw.err("NONLINEAR is not supported yet");
    }
    let elset = crate::inp::required(kw, "ELSET")?;
    let elems = geom::elem_ref(m, kw, elset)?;
    let mut lines: Vec<Vec<String>> = kw.data.iter().map(|d| d.fields.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).collect();
    // the last line is the coefficient, the line before it (if any) the degrees of freedom
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let Some(last) = lines.pop() else { return kw.err("data line with stiffness/coefficient is missing") };
    let c = crate::fem::model::parse_f64(&last[0]).ok_or_else(|| crate::model::FemError(format!("{}: '{}' is not a number", kw.location(), last[0])))?;
    let mut dofs = Vec::new();
    if let Some(l) = lines.last() {
        for s in l {
            match crate::inp::parse_label(s) {
                Some(d @ 1..=6) => dofs.push(d as Dof),
                _ => return kw.err(format!("'{s}' is not a degree of freedom")),
            }
        }
    }
    Ok((elems, dofs, c))
}

fn build_items(m: &FeModel, kw: &RawKeyword, damper: bool) -> Result<Springs> {
    let (elems, dofs, c) = parse(m, kw)?;
    let mut items = Vec::new();
    for e in elems {
        let el = &m.elements[e];
        let ty = el.etype.name();
        let k = match (ty, damper) {
            ("SPRING1", false) | ("DASHPOT1", true) => {
                let Some(&d) = dofs.first() else { return kw.err("degree of freedom missing for SPRING1/DASHPOT1") };
                Kind::One(el.nodes[0], d)
            }
            ("SPRING2", false) | ("DASHPOT2", true) => {
                if dofs.len() < 2 {
                    return kw.err("two degrees of freedom are required for SPRING2/DASHPOT2");
                }
                Kind::Two(el.nodes[0], dofs[0], el.nodes[1], dofs[1])
            }
            ("SPRINGA", false) | ("DASHPOTA", true) => {
                let (a, b) = (el.nodes[0], el.nodes[1]);
                let d = sub(m.nodes[b], m.nodes[a]);
                let l = norm(d);
                if l == 0.0 {
                    return kw.err(format!("element {} has zero length", el.label));
                }
                Kind::Axial(a, b, [d[0] / l, d[1] / l, d[2] / l])
            }
            _ => return kw.err(format!("element {} ({ty}) does not match *{}", el.label, kw.name)),
        };
        items.push((k, c));
    }
    Ok(Springs { name: format!("*{} ({})", kw.name, kw.location()), items, dim: m.dim(), damper })
}

fn spring(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    Ok(vec![Box::new(build_items(m, kw, false)?)])
}

fn dashpot(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    Ok(vec![Box::new(build_items(m, kw, true)?)])
}

impl Springs {
    /// Matrix contributions (node, dof, node, dof, value) for small displacements.
    fn matrix(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        let mut v = Vec::new();
        for (k, c) in &self.items {
            match *k {
                Kind::One(a, d) => v.push((a, d, a, d, *c)),
                Kind::Two(a, i, b, j) => {
                    v.push((a, i, a, i, *c));
                    v.push((b, j, b, j, *c));
                    v.push((a, i, b, j, -*c));
                    v.push((b, j, a, i, -*c));
                }
                Kind::Axial(a, b, n) => {
                    for p in 0..self.dim {
                        for q in 0..self.dim {
                            let kv = c * n[p] * n[q];
                            let (dp, dq) = (U1 + p as Dof, U1 + q as Dof);
                            v.push((a, dp, a, dq, kv));
                            v.push((b, dp, b, dq, kv));
                            v.push((a, dp, b, dq, -kv));
                            v.push((b, dp, a, dq, -kv));
                        }
                    }
                }
            }
        }
        v
    }
}

impl Interaction for Springs {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        if self.damper {
            return Vec::new();
        }
        let mut v = Vec::new();
        for (k, _) in &self.items {
            match *k {
                Kind::One(a, d) => v.push((a, d)),
                Kind::Two(a, i, b, j) => {
                    v.push((a, i));
                    v.push((b, j));
                }
                Kind::Axial(a, b, _) => {
                    for &d in geom::trans_dofs(self.dim) {
                        v.push((a, d));
                        v.push((b, d));
                    }
                }
            }
        }
        v
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.items
            .iter()
            .filter_map(|(k, _)| match *k {
                Kind::Two(a, _, b, _) | Kind::Axial(a, b, _) => Some(vec![a, b]),
                _ => None,
            })
            .collect()
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], mut k: Option<&mut dyn TangentSink>) -> Result<()> {
        if self.damper {
            return Ok(());
        }
        let m = ctx.model;
        for (kind, c) in &self.items {
            match *kind {
                Kind::Axial(a, b, _) if ctx.nlgeom => {
                    // large displacement: force along the current line
                    let x0 = sub(m.nodes[b], m.nodes[a]);
                    let l0 = norm(x0);
                    let mut d = x0;
                    for i in 0..3 {
                        d[i] += u[b][i] - u[a][i];
                    }
                    let l = norm(d);
                    let n = [d[0] / l, d[1] / l, d[2] / l];
                    let f = c * (l - l0);
                    for i in 0..self.dim {
                        fint[b][i] += f * n[i];
                        fint[a][i] -= f * n[i];
                    }
                    if let Some(k) = k.as_deref_mut() {
                        for p in 0..self.dim {
                            for q in 0..self.dim {
                                let kv = c * n[p] * n[q] + f / l * ((if p == q { 1.0 } else { 0.0 }) - n[p] * n[q]);
                                let (dp, dq) = (U1 + p as Dof, U1 + q as Dof);
                                k.add(a, dp, a, dq, kv);
                                k.add(b, dp, b, dq, kv);
                                k.add(a, dp, b, dq, -kv);
                                k.add(b, dp, a, dq, -kv);
                            }
                        }
                    }
                }
                _ => {
                    let one = Springs { name: String::new(), items: vec![(kind.clone(), *c)], dim: self.dim, damper: false };
                    for (na, da, nb, db, v) in one.matrix() {
                        fint[na][dof::slot_of(da)] += v * u[nb][dof::slot_of(db)];
                        if let Some(k) = k.as_deref_mut() {
                            k.add(na, da, nb, db, v);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn damping(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        if self.damper {
            // the dashpots have the same structure as the springs, with c instead of k
            return self.matrix();
        }
        Vec::new()
    }
}

// ---------------------------------------------------------------- masses

pub struct Masses {
    name: String,
    /// (node, dof, node, dof, value)
    entries: Vec<(usize, Dof, usize, Dof, f64)>,
}

impl Interaction for Masses {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn is_linear(&self) -> bool {
        true
    }
    fn mass(&self) -> Vec<(usize, Dof, usize, Dof, f64)> {
        self.entries.clone()
    }
}

fn mass(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let elset = crate::inp::required(kw, "ELSET")?;
    let elems = geom::elem_ref(m, kw, elset)?;
    let v = kw.first_floats()?;
    let mv = v.first().copied().unwrap_or(0.0);
    let dim = m.dim();
    let mut entries = Vec::new();
    for e in elems {
        let n = m.elements[e].nodes[0];
        for &d in geom::trans_dofs(dim) {
            entries.push((n, d, n, d, mv));
        }
    }
    Ok(vec![Box::new(Masses { name: format!("*MASS ({})", kw.location()), entries })])
}

fn rotary(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= is not supported yet");
    }
    let elset = crate::inp::required(kw, "ELSET")?;
    let elems = geom::elem_ref(m, kw, elset)?;
    let v = kw.first_floats()?;
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    // I11, I22, I33, I12, I13, I23
    let ii = [[g(0), g(3), g(4)], [g(3), g(1), g(5)], [g(4), g(5), g(2)]];
    let mut entries = Vec::new();
    for e in elems {
        let n = m.elements[e].nodes[0];
        for a in 0..3 {
            for b in 0..3 {
                if ii[a][b] != 0.0 {
                    entries.push((n, 4 + a as Dof, n, 4 + b as Dof, ii[a][b]));
                }
            }
        }
    }
    Ok(vec![Box::new(Masses { name: format!("*ROTARY INERTIA ({})", kw.location()), entries })])
}
