//! Kohesiv kontakt (ytbaserad): *CONTACT PAIR med en *SURFACE INTERACTION
//! som har *COHESIVE BEHAVIOR, med *DAMAGE INITIATION, *DAMAGE EVOLUTION och
//! *DAMAGE STABILIZATION.
//!
//! Formulering (nod mot yta, små glidningar som i implicit analys för
//! kontaktpar): varje slavnod A projiceras i referensläget på närmaste
//! masteryta (punkten ξ, formfunktionerna M_C och masterns normal n samt
//! tangenterna s, t). Separationen är δ = Rᵀ (u_A − Σ_C M_C(ξ) u_C) med
//! R = [n s t], och dragspänningen t(δ) ges av traction–separationslagen i
//! `materials/cohesive.rs` med δ som "töjning" (konstitutiv tjocklek 1):
//! okopplad elasticitet t = K δ, initiering MAXS/QUADS/MAXU/QUADU (MAXU och
//! QUADU = MAXE/QUADE med separationerna), utveckling med förskjutning eller
//! energi och viskös regularisering (*DAMAGE STABILIZATION). Kraften är
//! A_A R t med slavnodens andel A_A av slavytan.
//!
//! ELIGIBILITY=ORIGINAL CONTACTS (standard): bara slavnoder som är i kontakt
//! i början (gap ≤ 1e-6 · ytstorleken) binds. I tryck gäller den vanliga
//! hårda kontakten i normalriktningen, här med straffstyvheten
//! 1e3 · Knn; obundna noder har bara den.
//!
//! Utdata (nodvis i slavnoderna): CSDMG (skadan), CSMAXSCRT, CSQUADSCRT,
//! CSMAXUCRT eller CSQUADUCRT (initieringskriteriet).

use std::sync::{Arc, Mutex};

use super::constraints::sub_options;
use super::geom::{self, Facet};
use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::fem::dof::{NodeVec, U1, U2, U3};
use crate::fem::model::{DataLine, FeModel, MaterialDef, RawKeyword};
use crate::materials::cohesive::TractionSeparation;
use crate::materials::MaterialModel;
use crate::model::{fail, Result};

/// Straffstyvhet för hård kontakt i tryck relativt Knn.
const HARD_FACTOR: f64 = 1e3;

pub static FACTORIES: &[InteractionFactory] = &[
    InteractionFactory { keywords: &["CONTACT PAIR"], build: pair },
    InteractionFactory { keywords: &["CONTACT"], build: general },
];

/// Underalternativ till *SURFACE INTERACTION som den kohesiva kontakten läser.
pub const SI_OPTIONS: &[&str] = &["SURFACE BEHAVIOR", "FRICTION", "CONTACT DAMPING", "COHESIVE BEHAVIOR", "DAMAGE INITIATION", "DAMAGE EVOLUTION", "DAMAGE STABILIZATION"];

fn interaction<'a>(m: &'a FeModel, name: &str) -> Option<&'a RawKeyword> {
    let up = name.trim().to_uppercase();
    m.extra.iter().find(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(up.as_str()))
}

/// Ytinteraktionen `name` har *COHESIVE BEHAVIOR.
pub fn is_cohesive(m: &FeModel, name: &str) -> bool {
    interaction(m, name).is_some_and(|si| sub_options(m, si, SI_OPTIONS).iter().any(|k| k.name == "COHESIVE BEHAVIOR"))
}

/// General contact med kohesiva egenskaper stöds inte ännu.
fn general(m: &FeModel, _kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    for a in m.extra.iter().filter(|k| k.name == "CONTACT PROPERTY ASSIGNMENT") {
        for d in a.records() {
            if let Some(ip) = d.fields.get(2).filter(|s| !s.trim().is_empty()) {
                if is_cohesive(m, ip) {
                    return a.err_at(d.line, "kohesiv kontakt (*COHESIVE BEHAVIOR) i general contact stöds inte ännu (använd *CONTACT PAIR)");
                }
            }
        }
    }
    Ok(Vec::new())
}

fn line(fields: &[String], at: usize) -> DataLine {
    DataLine { fields: fields.to_vec(), line: at, continued: false }
}

/// Traction–separationslagen och viskositeten ur ytinteraktionens underalternativ.
fn law(m: &FeModel, si: &RawKeyword) -> Result<(Arc<dyn MaterialModel>, f64)> {
    let subs = sub_options(m, si, SI_OPTIONS);
    let cb = subs.iter().find(|k| k.name == "COHESIVE BEHAVIOR").copied().ok_or_else(|| crate::model::FemError(format!("{}: *COHESIVE BEHAVIOR saknas", si.location())))?;
    if cb.param_upper("TYPE").is_some_and(|t| t != "UNCOUPLED") {
        return cb.err("TYPE=COUPLED stöds inte ännu (bara UNCOUPLED)");
    }
    for (p, ok) in [("ELIGIBILITY", "ORIGINAL CONTACTS"), ("COHERE", "ORIGINAL CONTACTS")] {
        if let Some(v) = cb.param_upper(p).filter(|v| v != ok) {
            return cb.err(format!("{p}={v} stöds inte ännu (bara ORIGINAL CONTACTS)"));
        }
    }
    for p in ["DEPENDENCIES", "REPEATED CONTACTS"] {
        if cb.has(p) {
            return cb.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    if cb.data.len() > 1 {
        return cb.err("temperaturberoende kohesiv styvhet stöds inte ännu");
    }
    let v = cb.first_floats()?;
    let kn = v.first().copied().unwrap_or(0.0);
    if !(kn > 0.0) {
        return cb.err("ange den kohesiva styvheten Knn, Kss, Ktt (standardstyvheten stöds inte ännu)");
    }
    let ks = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(kn);
    let kt = v.get(2).copied().filter(|&x| x > 0.0).unwrap_or(ks);
    for f in subs.iter().filter(|k| k.name == "FRICTION") {
        if f.has("ROUGH") || f.first_floats()?.first().is_some_and(|&mu| mu != 0.0) {
            return f.err("friktion tillsammans med *COHESIVE BEHAVIOR stöds inte ännu");
        }
    }
    // materialdefinition för materials/cohesive.rs (MAXU/QUADU = MAXE/QUADE med δ som töjning)
    let mut el = cb.clone();
    el.name = "ELASTIC".into();
    el.params = vec![("TYPE".into(), "TRACTION".into())];
    el.data = vec![line(&[kn.to_string(), ks.to_string(), kt.to_string()], cb.line)];
    let mut behaviors = vec![el];
    let mut visc = 0.0;
    for k in &subs {
        match k.name.as_str() {
            "DAMAGE INITIATION" => {
                let mut d = (*k).clone();
                let c = k.param_upper("CRITERION").unwrap_or_default();
                let mapped = match c.as_str() {
                    "MAXS" | "QUADS" => c.clone(),
                    "MAXU" => "MAXE".into(),
                    "QUADU" => "QUADE".into(),
                    o => return k.err(format!("CRITERION={o} stöds inte för kohesiv kontakt (MAXS, QUADS, MAXU, QUADU)")),
                };
                d.params.retain(|(p, _)| p != "CRITERION");
                d.params.push(("CRITERION".into(), mapped));
                behaviors.push(d);
            }
            "DAMAGE EVOLUTION" => behaviors.push((*k).clone()),
            "DAMAGE STABILIZATION" => visc = k.first_floats()?.first().copied().unwrap_or(0.0),
            _ => {}
        }
    }
    let def = MaterialDef { name: format!("kohesiv kontakt {}", si.param("NAME").unwrap_or("")), behaviors, density: 0.0 };
    let build = crate::materials::cohesive::FACTORIES[0].build.expect("kohesiv lag");
    let Some(model) = build(&def)? else {
        return si.err("den kohesiva lagen kunde inte byggas");
    };
    if model.cohesive().is_none() {
        return si.err("den kohesiva lagen kunde inte byggas");
    }
    Ok((model, visc))
}

/// En slavnods villkor.
#[derive(Debug, Clone)]
struct Term {
    node: usize,
    /// (nod, koefficient): 1 för slavnoden, −M_C för masternoderna.
    coef: Vec<(usize, f64)>,
    /// Lokala riktningar: normal, s, t.
    r: [[f64; 3]; 3],
    area: f64,
    bonded: bool,
}

pub struct CohesiveContact {
    name: String,
    terms: Vec<Term>,
    dim: usize,
    model: Arc<dyn MaterialModel>,
    visc: f64,
    kn: f64,
    st: Mutex<(Vec<Vec<f64>>, Vec<Vec<f64>>)>,
}

impl CohesiveContact {
    fn ts(&self) -> &TractionSeparation {
        self.model.cohesive().expect("kohesiv lag")
    }
}

/// Tangentriktningar i ytan med normalen n (globala 1-axelns projektion i 3D).
fn tangents(n: [f64; 3], dim: usize) -> ([f64; 3], [f64; 3]) {
    if dim == 2 {
        return ([-n[1], n[0], 0.0], [0.0, 0.0, 1.0]);
    }
    let e = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
    let d = e[0] * n[0] + e[1] * n[1] + e[2] * n[2];
    let mut t1 = [e[0] - d * n[0], e[1] - d * n[1], e[2] - d * n[2]];
    let l = (t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]).sqrt();
    t1 = [t1[0] / l, t1[1] / l, t1[2] / l];
    let t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]];
    (t1, t2)
}

fn pair(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let Some(ip) = kw.param("INTERACTION") else { return Ok(Vec::new()) };
    if !is_cohesive(m, ip) {
        return Ok(Vec::new());
    }
    if crate::steps::explicit::contact::only_explicit(m) {
        return kw.err("kohesiv kontakt i explicit analys stöds inte ännu");
    }
    if kw.param_upper("TYPE").is_some_and(|t| t != "NODE TO SURFACE") {
        return kw.err("kohesiv kontakt stöds bara med TYPE=NODE TO SURFACE");
    }
    let si = interaction(m, ip).unwrap();
    let (model, visc) = law(m, si)?;
    let kn = model.cohesive().unwrap().elastic()[0][0];
    let mut out: Vec<Box<dyn Interaction>> = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 2 {
            return kw.err_at(d.line, "skriv slavyta, masteryta");
        }
        let slave = geom::surface(m, kw, f[0])?;
        let master = geom::surface(m, kw, f[1])?;
        if master.kind == "NODE" {
            return kw.err_at(d.line, "masterytan måste vara elementbaserad");
        }
        let mfac = geom::facets(m, master, "*CONTACT PAIR")?;
        let dim = mfac.first().map_or(3, |f| f.dim);
        // slavnodernas areor
        let mut areas: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
        if slave.kind == "NODE" {
            for &(n, a) in &slave.nodes {
                *areas.entry(n).or_default() += a;
            }
        } else {
            for sf in geom::facets(m, slave, "*CONTACT PAIR")? {
                let x = geom::coords(m, &sf, None);
                for (k, a) in geom::nodal_areas(&sf, &x).into_iter().enumerate() {
                    *areas.entry(sf.nodes[k]).or_default() += a;
                }
            }
        }
        let size = mfac.iter().map(|f| {
            let x = geom::coords(m, f, None);
            x.iter().map(|p| crate::elements::common::norm(crate::elements::common::sub(*p, x[0]))).fold(0.0, f64::max)
        });
        let size = size.fold(0.0, f64::max).max(1e-30);
        let mut terms = Vec::new();
        for (&n, &area) in &areas {
            let p = m.nodes[n];
            // närmaste masteryta (helst en där projektionen ligger inne i ytan)
            let mut best: Option<(bool, f64, &Facet, [f64; 2])> = None;
            for mf in &mfac {
                if mf.nodes.contains(&n) {
                    continue;
                }
                let x = geom::coords(m, mf, None);
                let (xi, dist, inside) = geom::project(mf, &x, p);
                let better = match best {
                    None => true,
                    Some((bi, bd, _, _)) => (inside && !bi) || (inside == bi && dist < bd),
                };
                if better {
                    best = Some((inside, dist, mf, xi));
                }
            }
            let Some((inside, _, mf, xi)) = best else { continue };
            let x = geom::coords(m, mf, None);
            let fp = geom::eval(mf, &x, xi);
            let nrm = fp.normal;
            let gap = (0..3).map(|q| (p[q] - fp.x[q]) * nrm[q]).sum::<f64>();
            let (s, t) = tangents(nrm, dim);
            let mut coef = vec![(n, 1.0)];
            for (k, &mn) in mf.nodes.iter().enumerate() {
                coef.push((mn, -fp.n[k]));
            }
            let bonded = inside && gap.abs() <= 1e-6 * size;
            terms.push(Term { node: n, coef, r: [nrm, s, t], area, bonded });
        }
        let ns = model.cohesive().unwrap().n_state();
        let mut st0 = vec![vec![0.0; ns]; terms.len()];
        for s in st0.iter_mut() {
            model.cohesive().unwrap().init_state(s);
        }
        out.push(Box::new(CohesiveContact {
            name: format!("*CONTACT PAIR (kohesiv) {}/{} ({})", f[0], f[1], kw.location()),
            terms,
            dim,
            model: model.clone(),
            visc,
            kn,
            st: Mutex::new((st0.clone(), st0)),
        }));
    }
    if out.is_empty() {
        return fail(format!("{}: inga kontaktytor", kw.location()));
    }
    Ok(out)
}

impl Interaction for CohesiveContact {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.terms.iter().map(|t| t.coef.iter().map(|c| c.0).collect()).collect()
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], k: Option<&mut dyn TangentSink>) -> Result<()> {
        let nd = self.dim;
        let ts = self.ts();
        let mut st = self.st.lock().unwrap();
        let (committed, trial) = &mut *st;
        let dofs = [U1, U2, U3];
        let mut k = k;
        for (i, t) in self.terms.iter().enumerate() {
            // relativ förskjutning i globala och lokala riktningar
            let mut du = [0.0; 3];
            for &(n, c) in &t.coef {
                for q in 0..3 {
                    du[q] += c * u[n][q];
                }
            }
            let mut e = [0.0; 3];
            for a in 0..nd {
                e[a] = (0..3).map(|q| t.r[a][q] * du[q]).sum();
            }
            // i tryck gäller den vanliga hårda kontakten (straff) i normalriktningen
            let kh = HARD_FACTOR * self.kn;
            let (tr, d) = if t.bonded {
                let o = ts.update(&e, nd, 1.0, ctx.dt, self.visc, &committed[i], &mut trial[i]);
                let (mut tr, mut d) = (o.t, o.dtde);
                if e[0] < 0.0 {
                    tr[0] = kh * e[0];
                    d[0] = [kh, 0.0, 0.0];
                    for row in d.iter_mut().skip(1) {
                        row[0] = 0.0;
                    }
                }
                (tr, d)
            } else if e[0] < 0.0 {
                let mut d = [[0.0; 3]; 3];
                d[0][0] = kh;
                ([kh * e[0], 0.0, 0.0], d)
            } else {
                ([0.0; 3], [[0.0; 3]; 3])
            };
            // global kraft F = A R t och tangent A R D Rᵀ
            let mut f = [0.0; 3];
            for a in 0..nd {
                for q in 0..3 {
                    f[q] += t.area * t.r[a][q] * tr[a];
                }
            }
            for &(n, c) in &t.coef {
                for q in 0..nd.max(2) {
                    fint[n][q] += c * f[q];
                }
            }
            let Some(k) = k.as_deref_mut() else { continue };
            let mut kg = [[0.0; 3]; 3];
            for p in 0..3 {
                for q in 0..3 {
                    let mut s = 0.0;
                    for a in 0..nd {
                        for b in 0..nd {
                            s += t.r[a][p] * d[a][b] * t.r[b][q];
                        }
                    }
                    kg[p][q] = t.area * s;
                }
            }
            let gd = if nd == 2 { 2 } else { 3 };
            let unsym = k.is_unsymmetric();
            for &(na, ca) in &t.coef {
                for &(nb, cb) in &t.coef {
                    for p in 0..gd {
                        for q in 0..gd {
                            if unsym {
                                let v = ca * cb * kg[p][q];
                                if v != 0.0 {
                                    k.add_unsym(na, dofs[p], nb, dofs[q], v);
                                }
                            } else {
                                let v = ca * cb * 0.5 * (kg[p][q] + kg[q][p]);
                                if v != 0.0 {
                                    k.add(na, dofs[p], nb, dofs[q], v);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn commit(&self) {
        let mut st = self.st.lock().unwrap();
        let (c, t) = &mut *st;
        c.clone_from(t);
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let ts = self.ts();
        let st = self.st.lock().unwrap();
        let mut dmg = NodeOutput { name: "CSDMG".into(), labels: vec!["CSDMG".into()], values: Vec::new() };
        let crit_name = ts.criterion_name().map(|c| match c {
            "MAXECRT" => "CSMAXUCRT".to_string(),
            "QUADECRT" => "CSQUADUCRT".to_string(),
            o => format!("CS{o}"),
        });
        let mut crit = crit_name.map(|n| NodeOutput { name: n.clone(), labels: vec![n], values: Vec::new() });
        for (i, t) in self.terms.iter().enumerate() {
            let (dv, cr, _) = ts.outputs(&st.0[i]);
            dmg.values.push((t.node, vec![dv]));
            if let Some(c) = crit.as_mut() {
                c.values.push((t.node, vec![cr]));
            }
        }
        let mut out = vec![dmg];
        out.extend(crit);
        out
    }
}
