//! Fjädrar (*SPRING för SPRING1/SPRING2/SPRINGA), dämpare (*DASHPOT),
//! punktmassor (*MASS) och rotationströghet (*ROTARY INERTIA).
//!
//! Dataraderna för *SPRING och *DASHPOT: först en rad med frihetsgraderna
//! (SPRING1: en, SPRING2: två, SPRINGA: tom eller utelämnad), sedan en rad
//! med styvheten/dämpningskoefficienten.
//!
//! - SPRING1: kraft k·u_i mot marken.
//! - SPRING2: kraft k·(u_i(a) − u_j(b)) mellan två noders valda frihetsgrader.
//! - SPRINGA: axiell fjäder längs linjen mellan noderna. Små förskjutningar:
//!   K = k n nᵀ. Med NLGEOM verkar kraften k (l − L) längs aktuell linje och
//!   tangenten får det geometriska tillskottet k (l − L)/l (I − n nᵀ).

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
    /// (nod, frihetsgrad) mot marken.
    One(usize, Dof),
    /// (nod a, frihetsgrad i, nod b, frihetsgrad j).
    Two(usize, Dof, usize, Dof),
    /// Axiell mellan noderna, med enhetsvektorn a → b i referensläget.
    Axial(usize, usize, [f64; 3]),
}

pub struct Springs {
    name: String,
    items: Vec<(Kind, f64)>,
    dim: usize,
    /// Dämpare i stället för fjädrar (bidrar bara med dämpning).
    damper: bool,
}

/// Elementen i ELSET= och (frihetsgrader, koefficient) från dataraderna.
fn parse(m: &FeModel, kw: &RawKeyword) -> Result<(Vec<usize>, Vec<Dof>, f64)> {
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= stöds inte ännu");
    }
    if kw.has("NONLINEAR") {
        return kw.err("NONLINEAR stöds inte ännu");
    }
    let elset = crate::inp::required(kw, "ELSET")?;
    let elems = geom::elem_ref(m, kw, elset)?;
    let mut lines: Vec<Vec<String>> = kw.data.iter().map(|d| d.fields.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).collect();
    // sista raden är koefficienten, raden före (om den finns) frihetsgraderna
    while lines.last().is_some_and(|l| l.is_empty()) {
        lines.pop();
    }
    let Some(last) = lines.pop() else { return kw.err("datarad med styvhet/koefficient saknas") };
    let c = crate::fem::model::parse_f64(&last[0]).ok_or_else(|| crate::model::FemError(format!("{}: '{}' är inget tal", kw.location(), last[0])))?;
    let mut dofs = Vec::new();
    if let Some(l) = lines.last() {
        for s in l {
            match crate::inp::parse_label(s) {
                Some(d @ 1..=6) => dofs.push(d as Dof),
                _ => return kw.err(format!("'{s}' är ingen frihetsgrad")),
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
                let Some(&d) = dofs.first() else { return kw.err("frihetsgrad saknas för SPRING1/DASHPOT1") };
                Kind::One(el.nodes[0], d)
            }
            ("SPRING2", false) | ("DASHPOT2", true) => {
                if dofs.len() < 2 {
                    return kw.err("två frihetsgrader krävs för SPRING2/DASHPOT2");
                }
                Kind::Two(el.nodes[0], dofs[0], el.nodes[1], dofs[1])
            }
            ("SPRINGA", false) | ("DASHPOTA", true) => {
                let (a, b) = (el.nodes[0], el.nodes[1]);
                let d = sub(m.nodes[b], m.nodes[a]);
                let l = norm(d);
                if l == 0.0 {
                    return kw.err(format!("elementet {} har noll längd", el.label));
                }
                Kind::Axial(a, b, [d[0] / l, d[1] / l, d[2] / l])
            }
            _ => return kw.err(format!("elementet {} ({ty}) passar inte *{}", el.label, kw.name)),
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
    /// Matrisbidrag (nod, frihetsgrad, nod, frihetsgrad, värde) för små förskjutningar.
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
                    // stor förskjutning: kraft längs aktuell linje
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
            // dämparna har samma struktur som fjädrarna, med c i stället för k
            return self.matrix();
        }
        Vec::new()
    }
}

// ---------------------------------------------------------------- massor

pub struct Masses {
    name: String,
    /// (nod, frihetsgrad, nod, frihetsgrad, värde)
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
        return kw.err("ORIENTATION= stöds inte ännu");
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
