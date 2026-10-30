//! Termisk kontakt mellan ytor: *GAP CONDUCTANCE och *GAP RADIATION i den
//! *SURFACE INTERACTION som ett *CONTACT PAIR refererar till.
//!
//! Diskretisering nod mot yta (standard för *CONTACT PAIR): varje
//! slavnod A med slavytans nodarea A_A = ∫N_A dA projiceras på närmaste
//! masteryta (formfunktionerna M_C i projektionen). Spaltens avstånd d är
//! avståndet längs masterytans normal (positivt när ytorna är åtskilda, 0 vid
//! genomträngning). Värmeflödet från master till slav:
//!
//! - ledning: q = k(d) (T_M − T_A), T_M = Σ M_C T_C;
//! - strålning: q = F(d) σ ((T_M − T_Z)⁴ − (T_A − T_Z)⁴) med
//!   F = f(d) / (1/ε_A + 1/ε_B − 1) (f = siktfaktorn från tabellen).
//!
//! Slavnoden får +A_A q och masternoderna −M_C A_A q. Tabellerna k(d) och
//! f(d) interpoleras linjärt och hålls konstanta utanför (en rad = konstant).
//! Steg-typen för värmeledning (`steps/heat.rs`) anropar `build` och
//! `contribute`; i mekaniken bidrar paret som vanligt via `contact.rs`.

use super::geom::{self, Facet};
use super::{Interaction, InteractionFactory};
use crate::elements::common::{dot, sub};
use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::{fail, FemError, Result};

/// Registret ger bara kopplingarna (slavnod–masternoder) till
/// matrisstrukturen; flödena räknas av värmeledningen via `build`.
pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CONTACT PAIR"], build: links }];

/// Matrisstrukturen för termisk kontakt.
struct ThermalLink {
    groups: Vec<Vec<usize>>,
}

impl Interaction for ThermalLink {
    fn name(&self) -> String {
        "termisk kontakt".into()
    }
    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.groups.clone()
    }
}

fn links(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let pairs = build_pair(m, kw)?;
    if pairs.is_empty() {
        return Ok(Vec::new());
    }
    let groups = pairs.iter().flat_map(|p| p.nodes.iter().map(|s| std::iter::once(s.node).chain(s.master.iter().map(|c| c.0)).collect())).collect();
    Ok(vec![Box::new(ThermalLink { groups })])
}

/// En slavnod med sin projektion.
#[derive(Debug, Clone)]
struct SlaveNode {
    node: usize,
    area: f64,
    /// Masternoder och formfunktioner i projektionen.
    master: Vec<(usize, f64)>,
    /// Avstånd i referensläget längs masternormalen.
    gap0: f64,
    /// Masterytans utåtriktade normal i projektionen.
    normal: [f64; 3],
    /// Projektionspunkt i referensläget.
    xm: [f64; 3],
}

/// Termisk kontakt för ett ytpar.
#[derive(Debug, Clone)]
pub struct ThermalPair {
    nodes: Vec<SlaveNode>,
    /// (d, k) sorterat efter d.
    conductance: Vec<(f64, f64)>,
    /// (d, f) och 1/(1/ε_A + 1/ε_B − 1).
    radiation: Option<(Vec<(f64, f64)>, f64)>,
}

fn table(v: &[(f64, f64)], d: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    if d <= v[0].0 {
        return v[0].1;
    }
    for w in v.windows(2) {
        if d <= w[1].0 {
            let s = (d - w[0].0) / (w[1].0 - w[0].0);
            return w[0].1 + s * (w[1].1 - w[0].1);
        }
    }
    v[v.len() - 1].1
}

fn floats(kw: &RawKeyword, d: &crate::fem::model::DataLine) -> Result<Vec<f64>> {
    d.fields
        .iter()
        .filter(|f| !f.trim().is_empty())
        .map(|f| parse_f64(f).ok_or_else(|| FemError(format!("{} rad {}: '{f}' är inget tal", kw.file, d.line))))
        .collect()
}

/// Underalternativ till en *SURFACE INTERACTION (nyckelorden direkt efter den).
fn interaction_options<'a>(m: &'a FeModel, name: &str) -> Vec<&'a RawKeyword> {
    const SUB: &[&str] = &["FRICTION", "SURFACE BEHAVIOR", "GAP CONDUCTANCE", "GAP RADIATION", "GAP HEAT GENERATION", "CONTACT DAMPING"];
    let Some(i) = m.extra.iter().position(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(name)) else {
        return Vec::new();
    };
    m.extra.iter().skip(i + 1).take_while(|k| SUB.contains(&k.name.as_str())).collect()
}

/// Ett *CONTACT PAIR vars alla ytor ligger på värmeledningselement (ingen
/// mekanisk kontakt; bara termisk kontakt via `build`).
pub fn only_heat(m: &FeModel, kw: &RawKeyword) -> bool {
    let mut any = false;
    for d in kw.records() {
        for f in d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            let Some(s) = m.surfaces.get(&f.to_uppercase()) else { return false };
            for (e, _) in &s.faces {
                if m.elements[*e].etype.family() != crate::elements::Family::Heat {
                    return false;
                }
                any = true;
            }
        }
    }
    any
}

/// Alla par med termiska egenskaper i modellen.
pub fn build(m: &FeModel) -> Result<Vec<ThermalPair>> {
    let mut out = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "CONTACT PAIR") {
        out.extend(build_pair(m, kw)?);
    }
    Ok(out)
}

/// Termiska par för ett *CONTACT PAIR (tomt om interaktionen saknar
/// *GAP CONDUCTANCE och *GAP RADIATION).
fn build_pair(m: &FeModel, kw: &RawKeyword) -> Result<Vec<ThermalPair>> {
    let mut out = Vec::new();
    {
        let Some(iname) = kw.param_upper("INTERACTION") else { return Ok(out) };
        let opts = interaction_options(m, &iname);
        let mut conductance = Vec::new();
        let mut radiation = None;
        for o in &opts {
            match o.name.as_str() {
                "GAP CONDUCTANCE" => {
                    for p in ["PRESSURE", "USER", "DEPENDENCIES"] {
                        if o.has(p) {
                            return o.err(format!("{p} stöds inte ännu (bara k mot spaltavstånd)"));
                        }
                    }
                    for d in o.records() {
                        let v = floats(o, &d)?;
                        if v.is_empty() {
                            continue;
                        }
                        conductance.push((v.get(1).copied().unwrap_or(0.0), v[0]));
                    }
                }
                "GAP RADIATION" => {
                    let rec = o.records();
                    let Some(first) = rec.first() else { return o.err("skriv 'ε slav, ε master'") };
                    let e = floats(o, first)?;
                    if e.len() < 2 || e[0] <= 0.0 || e[1] <= 0.0 {
                        return o.err("skriv 'ε slav, ε master' (båda > 0)");
                    }
                    let mut view = Vec::new();
                    for d in &rec[1..] {
                        let v = floats(o, d)?;
                        if !v.is_empty() {
                            view.push((v.get(1).copied().unwrap_or(0.0), v[0]));
                        }
                    }
                    if view.is_empty() {
                        view.push((0.0, 1.0));
                    }
                    view.sort_by(|a, b| a.0.total_cmp(&b.0));
                    radiation = Some((view, 1.0 / (1.0 / e[0] + 1.0 / e[1] - 1.0)));
                }
                "GAP HEAT GENERATION" => return o.err("*GAP HEAT GENERATION stöds inte ännu"),
                _ => {}
            }
        }
        if conductance.is_empty() && radiation.is_none() {
            return Ok(out);
        }
        conductance.sort_by(|a, b| a.0.total_cmp(&b.0));
        for d in kw.records() {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if f.len() < 2 {
                return kw.err_at(d.line, "skriv 'slavyta, masteryta'");
            }
            let ss = geom::surface(m, kw, f[0])?;
            let ms = geom::surface(m, kw, f[1])?;
            let sf = geom::facets(m, ss, "slavyta")?;
            let mf = geom::facets(m, ms, "masteryta")?;
            out.push(ThermalPair { nodes: slave_nodes(m, &sf, &mf)?, conductance: conductance.clone(), radiation: radiation.clone() });
        }
    }
    Ok(out)
}

fn slave_nodes(m: &FeModel, sf: &[Facet], mf: &[Facet]) -> Result<Vec<SlaveNode>> {
    let mut area: std::collections::BTreeMap<usize, f64> = std::collections::BTreeMap::new();
    for f in sf {
        let x = geom::coords(m, f, None);
        for (k, a) in geom::nodal_areas(f, &x).into_iter().enumerate() {
            *area.entry(f.nodes[k]).or_default() += a;
        }
    }
    let mut out = Vec::new();
    for (&n, &a) in &area {
        let p = m.nodes[n];
        // närmaste masteryta (helst med projektionen inne i ytan)
        let mut best: Option<(bool, f64, usize, [f64; 2])> = None;
        for (i, f) in mf.iter().enumerate() {
            let x = geom::coords(m, f, None);
            let (xi, dist, inside) = geom::project(f, &x, p);
            let better = match best {
                None => true,
                Some((bi, bd, _, _)) => (inside && !bi) || (inside == bi && dist < bd - 1e-14),
            };
            if better {
                best = Some((inside, dist, i, xi));
            }
        }
        let Some((_, _, i, xi)) = best else { return fail("termisk kontakt: masterytan är tom") };
        let f = &mf[i];
        let x = geom::coords(m, f, None);
        let fp = geom::eval(f, &x, xi);
        let gap0 = dot(sub(p, fp.x), fp.normal);
        out.push(SlaveNode {
            node: n,
            area: a,
            master: f.nodes.iter().copied().zip(fp.n.iter().copied()).filter(|(_, w)| w.abs() > 1e-14).collect(),
            gap0,
            normal: fp.normal,
            xm: fp.x,
        });
    }
    Ok(out)
}

impl ThermalPair {
    /// Flödet per area in i varje slavnod, (nod, q).
    pub fn surface_flux(&self, u: &[NodeVec], ts: usize, disp: bool, abs_zero: f64, sigma: f64) -> Vec<(usize, f64)> {
        let mut f = vec![[0.0; crate::fem::dof::SLOTS]; u.len()];
        let _ = self.contribute(u, ts, disp, abs_zero, sigma, &mut f, false);
        self.nodes.iter().map(|s| (s.node, -f[s.node][ts] / s.area)).collect()
    }

    /// Bidrag till inre värmeflöde (slot `ts`) och tangent (nodlista, matris)
    /// vid temperaturerna i `u`. Med `disp` räknas spaltavståndet i aktuellt läge.
    pub fn contribute(
        &self,
        u: &[NodeVec],
        ts: usize,
        disp: bool,
        abs_zero: f64,
        sigma: f64,
        fint: &mut [NodeVec],
        tangent: bool,
    ) -> Result<Vec<(Vec<usize>, Vec<f64>)>> {
        if self.radiation.is_some() && sigma == 0.0 {
            return fail("*GAP RADIATION kräver *PHYSICAL CONSTANTS, ABSOLUTE ZERO=..., STEFAN BOLTZMANN=...");
        }
        let mut mats = Vec::new();
        for s in &self.nodes {
            let mut d = s.gap0;
            if disp {
                let mut um = [0.0; 3];
                for &(c, w) in &s.master {
                    for k in 0..3 {
                        um[k] += w * u[c][k];
                    }
                }
                let us = u[s.node];
                let xs = [0, 1, 2].map(|k| s.xm[k] + s.gap0 * s.normal[k] + us[k] - um[k] - s.xm[k]);
                d = dot(xs, s.normal);
            }
            let d = d.max(0.0);
            let ta = u[s.node][ts];
            let tm: f64 = s.master.iter().map(|&(c, w)| w * u[c][ts]).sum();
            // q = flöde in i slaven per area och derivatorna d/dT_A, d/dT_M
            let k = table(&self.conductance, d);
            let mut q = k * (tm - ta);
            let mut dqa = -k;
            let mut dqm = k;
            if let Some((view, fe)) = &self.radiation {
                let c = sigma * fe * table(view, d);
                let a = ta - abs_zero;
                let b = tm - abs_zero;
                q += c * (b.powi(4) - a.powi(4));
                dqa -= 4.0 * c * a.powi(3);
                dqm += 4.0 * c * b.powi(3);
            }
            // inre flöde = −tillfört flöde
            fint[s.node][ts] -= s.area * q;
            for &(c, w) in &s.master {
                fint[c][ts] += w * s.area * q;
            }
            if tangent {
                let mut nodes = vec![s.node];
                nodes.extend(s.master.iter().map(|p| p.0));
                let coef: Vec<f64> = std::iter::once(-1.0).chain(s.master.iter().map(|p| p.1)).collect();
                let dq: Vec<f64> = std::iter::once(dqa).chain(s.master.iter().map(|p| dqm * p.1)).collect();
                let nd = nodes.len();
                let mut ke = vec![0.0; nd * nd];
                for i in 0..nd {
                    for j in 0..nd {
                        ke[i * nd + j] = coef[i] * s.area * dq[j];
                    }
                }
                mats.push((nodes, ke));
            }
        }
        Ok(mats)
    }
}
