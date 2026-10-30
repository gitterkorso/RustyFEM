//! *ELECTROMAGNETIC, LOW FREQUENCY, TIME HARMONIC: virvelströmsanalys med
//! kantelementen i `elements/electromagnetic.rs` (EMC3D4).
//!
//! Ekvationen för den komplexa vektorpotentialen A (tidsberoende e^{iωt}):
//!
//!   ∇×(μ⁻¹ ∇×A) + iωσA = J_s,
//!
//! med den naturliga randvillkoret n × H = 0 på modellens yta. Det reella
//! systemet [K, −ωM; ωM, K] (a_r, a_i) = (f, 0) löses med den osymmetriska
//! lösaren, en block (två ekvationer) per kant. Indata: *MAGNETIC
//! PERMEABILITY (μ, isotrop), *ELECTRICAL CONDUCTIVITY (σ, isotrop) och
//! *DECURRENT med typen CJ (strömtäthet: storlek och riktning). En frekvens per
//! dataraad (frekvensområden ger fel).
//!
//! Utdata per element (hela elementet; EMC2D3: EMB3, EMH3 och
//! EMCD1–2): EMB = ∇×A, EMH = B/μ,
//! EMCD = −iωσ Ā (virvelströmmen, elementmedel) och EMJH (tidsmedelvärdet av
//! Joulevärmen per volym ½|J̄|²/σ ur elementmedlet); realdelen som fältet självt och
//! imaginärdelen som FÄLT_IM.

use super::api::{StepCtx, StepType};
use crate::elements::electromagnetic as em;
use crate::elements::IpField;
use crate::fem::model::{FeModel, StepDef};
use crate::fem::output::FrameData;
use crate::fem::dof::SLOTS;
use crate::materials::heat::Table;
use crate::model::{fail, FemError, Result};
use crate::solver::Graph;
use std::collections::BTreeMap;

pub struct Electromagnetic;
pub static STEP_TYPES: &[&dyn StepType] = &[&Electromagnetic];

/// μ och σ för ett material.
fn em_props(model: &FeModel, mi: usize) -> Result<(f64, f64)> {
    let def = &model.materials[mi];
    let first = |name: &str| -> Result<Option<f64>> {
        let Some(kw) = def.behavior(name) else { return Ok(None) };
        let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISO".into());
        if ty != "ISO" && ty != "ISOTROPIC" {
            return kw.err(format!("TYPE={ty} stöds inte ännu för elektromagnetism (ISO)"));
        }
        if kw.has("FREQUENCY") || kw.has("DEPENDENCIES") {
            return kw.err("frekvens- och fältberoende stöds inte ännu");
        }
        let t: Table = crate::materials::heat::read_table(kw, 1)?;
        if t.varies() {
            return kw.err("temperaturberoende värden stöds inte ännu");
        }
        Ok(Some(t.value(0.0)))
    };
    let Some(mu) = first("MAGNETIC PERMEABILITY")? else {
        return fail(format!("material {}: *MAGNETIC PERMEABILITY saknas", def.name));
    };
    if mu <= 0.0 {
        return fail(format!("material {}: permeabiliteten ska vara positiv", def.name));
    }
    Ok((mu, first("ELECTRICAL CONDUCTIVITY")?.unwrap_or(0.0)))
}

/// Frekvenserna från procedurens datarader.
fn frequencies(step: &StepDef) -> Result<Vec<f64>> {
    let Some(kw) = step.procedure.as_ref() else { return fail("proceduren saknas") };
    if !kw.has("LOW FREQUENCY") {
        return kw.err("bara LOW FREQUENCY stöds");
    }
    if !kw.has("TIME HARMONIC") {
        return kw.err("bara TIME HARMONIC stöds ännu (TRANSIENT saknas)");
    }
    let mut out = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        let f0 = v.first().copied().unwrap_or(0.0);
        if v.get(1).copied().unwrap_or(0.0) != 0.0 {
            return kw.err_at(d.line, "frekvensområden stöds inte ännu (ange en frekvens per rad)");
        }
        out.push(f0);
    }
    if out.is_empty() {
        return kw.err("frekvens saknas");
    }
    Ok(out)
}

struct EmEl {
    e: usize,
    edges: Vec<usize>,
    f: em::EdgeFns,
    vol: f64,
    mu: f64,
    sigma: f64,
    planar: bool,
}

impl StepType for Electromagnetic {
    fn name(&self) -> &'static str {
        "ELECTROMAGNETIC"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "ELECTROMAGNETIC").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let model = ctx.prep.model;
        if ctx.step.procedure.as_ref().is_some_and(|k| k.has("STABILIZATION")) {
            return fail("STABILIZATION= stöds inte ännu");
        }
        let freqs = frequencies(ctx.step)?;
        // kanter: (min nod, max nod) → index
        let mut edge_idx: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        let mut els = Vec::new();
        let mut props: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
        for (e, el) in model.elements.iter().enumerate() {
            let Some(eme) = em::lookup(el.etype.name()) else {
                return fail(format!("{}: *ELECTROMAGNETIC kräver elektromagnetiska element (EMC2D3, EMC3D4)", model.elem_name(e)));
            };
            let Some(si) = el.section else { return fail(format!("{}: saknar *SOLID SECTION", model.elem_name(e))) };
            let Some(mi) = model.sections[si].material_index else { return fail(format!("{}: tvärsnittet saknar material", model.elem_name(e))) };
            let (mu, sigma) = match props.get(&mi) {
                Some(p) => *p,
                None => {
                    let p = em_props(model, mi)?;
                    props.insert(mi, p);
                    p
                }
            };
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let g = eme.geom(&x, model.sections[si].solid_thickness()).map_err(|err| FemError(format!("{}: {}", model.elem_name(e), err.0)))?;
            let mut edges = Vec::new();
            let mut loc = Vec::new();
            for &(a, b) in eme.edges() {
                // kantfunktionerna i den globala riktningen (lägre → högre nodindex)
                let (na, nb) = (el.nodes[a], el.nodes[b]);
                loc.push(if na > nb { (b, a) } else { (a, b) });
                let key = (na.min(nb), na.max(nb));
                let n = edge_idx.len();
                edges.push(*edge_idx.entry(key).or_insert(n));
            }
            let f = em::edge_fns(&g, &loc);
            els.push(EmEl { e, edges, f, vol: g.vol, mu, sigma, planar: eme.planar });
        }
        let ne = edge_idx.len();
        // källor: *DECURRENT CJ
        let mut src = vec![[0.0f64; 3]; model.elements.len()];
        for l in ctx.loads.dloads_at(1.0) {
            match l.kind.as_str() {
                "ECUR:CJ" => {
                    let d = [0, 1, 2].map(|c| l.extra.get(c).copied().unwrap_or(0.0));
                    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if n == 0.0 {
                        return fail(format!("{}: *DECURRENT CJ saknar riktning", model.elem_name(l.elem)));
                    }
                    for c in 0..3 {
                        src[l.elem][c] += l.magnitude * d[c] / n;
                    }
                }
                k => return fail(format!("{}: lasten {k} stöds inte i *ELECTROMAGNETIC", model.elem_name(l.elem))),
            }
        }
        if !ctx.loads.bc.is_empty() || !ctx.loads.cload.is_empty() {
            return fail("*BOUNDARY och *CLOAD stöds inte i *ELECTROMAGNETIC (randvillkoret är n × H = 0)");
        }
        // matrisstrukturen: block = kant (två ekvationer), koordinat = kantens mitt
        let mut coords = vec![[0.0; 3]; ne];
        for (&(a, b), &i) in &edge_idx {
            coords[i] = [0, 1, 2].map(|c| 0.5 * (model.nodes[a][c] + model.nodes[b][c]));
        }
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ne];
        for el in &els {
            for &a in &el.edges {
                adj[a].extend(el.edges.iter().copied().filter(|&b| b != a));
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        let solver = crate::solver::select(Some("unsymmetric"))?;
        let mut frames = Vec::new();
        for &freq in &freqs {
            let w = 2.0 * std::f64::consts::PI * freq;
            let mut mat = solver.prepare(&Graph { coords: coords.clone(), adj: adj.clone(), ndof: vec![2; ne] })?;
            let first: Vec<usize> = mat.first_dof().to_vec();
            let mut rhs = vec![0.0; 2 * ne];
            for el in &els {
                let ned = el.edges.len();
                for p in 0..ned {
                    let rp = first[el.edges[p]];
                    let fp = el.vol * (0..3).map(|c| src[el.e][c] * el.f.mean[p][c]).sum::<f64>();
                    rhs[rp] += fp;
                    for q in 0..ned {
                        let cq = first[el.edges[q]];
                        let k = el.vol / el.mu * (0..3).map(|c| el.f.curl[p][c] * el.f.curl[q][c]).sum::<f64>();
                        let m = w * el.sigma * el.f.mass[p][q];
                        mat.add_unsym(rp, cq, k);
                        mat.add_unsym(rp + 1, cq + 1, k);
                        if m != 0.0 {
                            mat.add_unsym(rp, cq + 1, -m);
                            mat.add_unsym(rp + 1, cq, m);
                        }
                    }
                }
            }
            if let Err(q) = mat.factor() {
                return fail(format!("ekvationssystemet är singulärt (ekvation {q}); icke-ledande områden kräver stabilisering, som inte stöds ännu"));
            }
            mat.solve(&mut rhs);
            let ar: Vec<f64> = (0..ne).map(|i| rhs[first[i]]).collect();
            let ai: Vec<f64> = (0..ne).map(|i| rhs[first[i] + 1]).collect();
            let mut elem = vec![Vec::new(); model.elements.len()];
            for el in &els {
                let mut b = [[0.0; 3]; 2];
                let mut am = [[0.0; 3]; 2];
                let mut aa = 0.0;
                for (part, a) in [&ar, &ai].into_iter().enumerate() {
                    for p in 0..el.edges.len() {
                        let v = a[el.edges[p]];
                        for c in 0..3 {
                            b[part][c] += v * el.f.curl[p][c];
                            am[part][c] += v * el.f.mean[p][c];
                        }
                        for q in 0..el.edges.len() {
                            aa += v * el.f.mass[p][q] * a[el.edges[q]];
                        }
                    }
                }
                let lab3 = |s: &str| [format!("{s}1"), format!("{s}2"), format!("{s}3")];
                let mk = |name: &str, base: &str, v: [f64; 3]| {
                    let l = lab3(base);
                    let mut f = IpField::new(name, &[l[0].as_str(), l[1].as_str(), l[2].as_str()]);
                    f.points.push((0, 0, v.to_vec()));
                    f
                };
                let h = |v: [f64; 3]| v.map(|x| x / el.mu);
                // J = −iωσA: Re = ωσ A_i, Im = −ωσ A_r
                let jr = am[1].map(|x| w * el.sigma * x);
                let ji = am[0].map(|x| -w * el.sigma * x);
                let mut jh = IpField::new("EMJH", &["EMJH"]);
                // Joulevärmen ur elementets medelström: ½ |J̄|²/σ
                let _ = aa;
                let j2: f64 = (0..3).map(|c| jr[c] * jr[c] + ji[c] * ji[c]).sum();
                jh.points.push((0, 0, vec![if el.sigma > 0.0 { 0.5 * j2 / el.sigma } else { 0.0 }]));
                elem[el.e] = if el.planar {
                    // plant: B och H vinkelräta mot planet (EMB3, EMH3), strömmen i planet
                    let sc = |name: &str, v: f64| {
                        let base = name.trim_end_matches("_IM");
                        let mut f = IpField::new(name, &[base]);
                        f.points.push((0, 0, vec![v]));
                        f
                    };
                    let v2 = |name: &str, v: [f64; 3]| {
                        let mut f = IpField::new(name, &["EMCD1", "EMCD2"]);
                        f.points.push((0, 0, vec![v[0], v[1]]));
                        f
                    };
                    vec![
                        sc("EMB3", b[0][2]),
                        sc("EMB3_IM", b[1][2]),
                        sc("EMH3", b[0][2] / el.mu),
                        sc("EMH3_IM", b[1][2] / el.mu),
                        v2("EMCD", jr),
                        v2("EMCD_IM", ji),
                        jh,
                    ]
                } else {
                    vec![
                        mk("EMB", "EMB", b[0]),
                        mk("EMB_IM", "EMB", b[1]),
                        mk("EMH", "EMH", h(b[0])),
                        mk("EMH_IM", "EMH", h(b[1])),
                        mk("EMCD", "EMCD", jr),
                        mk("EMCD_IM", "EMCD", ji),
                        jh,
                    ]
                };
            }
            let nn = model.nodes.len();
            frames.push(FrameData {
                value: freq,
                mode: -1,
                description: format!("Increment {}: Frequency = {freq}", frames.len() + 1),
                u: vec![[0.0; SLOTS]; nn],
                rf: vec![[0.0; SLOTS]; nn],
                active: vec![0; nn],
                constrained: vec![0; nn],
                elem,
                node_extra: Vec::new(),
            });
        }
        ctx.out.frames.extend(frames);
        Ok(())
    }
}
