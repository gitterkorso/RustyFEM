//! Område explicit_adv: ALE-nät (`*ADAPTIVE MESH`) för Lagrange-element
//! (C3D8R, C3D8) i explicit analys.
//!
//! Var FREQUENCY:e inkrement (standard 10) flyttas domänens inre noder med
//! `MESH SWEEPS` (standard 5) utjämningssvep: volymviktad utjämning
//! x ← Σ Vₑ cₑ / Σ Vₑ (cₑ = elementets tyngdpunkt, standard) och
//! Laplace-utjämning x ← medelvärdet av kantgrannarna, blandade med vikterna
//! på *ADAPTIVE MESH CONTROLS dataraden (volymetrisk, Laplace,
//! ekvipotential; ekvipotential räknas som Laplace). Första gången görs i
//! stället `INITIAL MESH SWEEPS` svep (standard 5). Noder på domänens rand
//! och noder med randvillkor följer materialet (Lagrange).
//!
//! Elementtillståndet (spänning och materialtillstånd, en punkt) advekteras
//! från det Lagrangeska till det utjämnade nätet med ytornas svepta volymer
//! (givarelementets värden, volymviktat) och massan följer flödet;
//! rörelsemängden advekteras med elementmittprojektion (MOMENTUM
//! ADVECTION=ELEMENT CENTER PROJECTION, standard; HALF INDEX SHIFT räknas på
//! samma sätt). Nodmassorna räknas om från elementmassorna med elementens
//! ursprungliga fördelning.
//!
//! Begränsningar: materialet uppdateras inkrementellt (hypoelastiskt), så
//! material som läser den totala deformationsgradienten (*EOS, hyperelastiska)
//! stöds inte i ALE-domänen; randnoderna utjämnas inte längs ytan;
//! hourglass-tillståndet advekteras inte; *ADAPTIVE MESH CONSTRAINT och
//! Euler-ränder stöds inte.

use super::{Eval, Sink};
use crate::elements::MassKind;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::{parse_f64, StepDef};
use crate::model::{fail, Result};
use std::collections::{BTreeSet, HashMap};

struct AElem {
    e: usize,
    nodes: [usize; 8],
    /// Massa (ändras med advektionen).
    mass: f64,
    /// Nodvis andel av elementets massa (diagonal massa i början).
    frac: [f64; 8],
    nbr: [Option<(usize, usize)>; 6],
}

pub struct Ale {
    elems: Vec<AElem>,
    /// Inre noder som utjämnas.
    free: Vec<usize>,
    /// Alla domänens noder.
    nodes: Vec<usize>,
    /// Element runt varje nod och kantgrannar.
    node_elems: HashMap<usize, Vec<usize>>,
    node_nbrs: HashMap<usize, Vec<usize>>,
    freq: usize,
    sweeps: usize,
    initial_sweeps: usize,
    weights: [f64; 2],
    inc: usize,
    first: bool,
    pending: Vec<(usize, [f64; 3], [f64; 3])>,
    /// Nätets sammanlagda omflyttning per nod (för hourglass-styvheten).
    reloc: HashMap<usize, [f64; 3]>,
}

fn centroid(x: &[[f64; 3]; 8]) -> [f64; 3] {
    std::array::from_fn(|c| x.iter().map(|p| p[c]).sum::<f64>() / 8.0)
}

fn volume(x: &[[f64; 3]; 8]) -> f64 {
    crate::steps::explicit::dt::geometry(crate::elements::Family::Continuum3D, "C3D8R", &x.to_vec()).map_or(0.0, |g| g.vol)
}

/// Svept volym (positiv utåt) när ytan (utåtriktad nodordning) går från `a` till `b`.
fn swept(a: &[[f64; 3]; 4], b: &[[f64; 3]; 4]) -> f64 {
    let x = [a[0], a[3], a[2], a[1], b[0], b[3], b[2], b[1]];
    // tetraederuppdelning som i explicit_adv_euler.rs
    let c: [f64; 3] = std::array::from_fn(|k| x.iter().map(|p| p[k]).sum::<f64>() / 8.0);
    let mut v = 0.0;
    for f in crate::elements::continuum3d::HEX8_FACES {
        let fc: [f64; 3] = std::array::from_fn(|k| f.iter().map(|&i| x[i][k]).sum::<f64>() / 4.0);
        for t in 0..4 {
            let (p, q) = (x[f[t]], x[f[(t + 1) % 4]]);
            let u = [fc[0] - c[0], fc[1] - c[1], fc[2] - c[2]];
            let vv = [q[0] - c[0], q[1] - c[1], q[2] - c[2]];
            let w = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            v += (u[0] * (vv[1] * w[2] - vv[2] * w[1]) - u[1] * (vv[0] * w[2] - vv[2] * w[0]) + u[2] * (vv[0] * w[1] - vv[1] * w[0])) / 6.0;
        }
    }
    v
}

impl Ale {
    pub fn build(p: &Prepared, step: &StepDef, _nlgeom: bool) -> Result<Option<Ale>> {
        let m = p.model;
        let Some(am) = step.extra.iter().rev().find(|k| k.name == "ADAPTIVE MESH") else { return Ok(None) };
        let Some(set) = am.param("ELSET").map(|s| s.trim().to_uppercase()) else { return am.err("ELSET= saknas") };
        let Some(els) = m.elsets.get(&set) else { return am.err(format!("elementsetet '{set}' finns inte")) };
        let freq = am.param("FREQUENCY").and_then(parse_f64).map_or(10, |f| f.max(1.0) as usize);
        let sweeps = am.param("MESH SWEEPS").and_then(parse_f64).map_or(5, |f| f.max(0.0) as usize);
        let mut initial_sweeps = am.param("INITIAL MESH SWEEPS").and_then(parse_f64).map(|f| f.max(0.0) as usize);
        let mut weights = [1.0, 0.0];
        if let Some(cn) = am.param_upper("CONTROLS") {
            let Some(c) = m.extra.iter().find(|k| k.name == "ADAPTIVE MESH CONTROLS" && k.param_upper("NAME").as_deref() == Some(cn.as_str())) else {
                return am.err(format!("*ADAPTIVE MESH CONTROLS, NAME={cn} saknas"));
            };
            let v = c.first_floats().unwrap_or_default();
            if !v.is_empty() {
                let (wv, wl, we) = (v.first().copied().unwrap_or(1.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0));
                let s = (wv + wl + we).max(1.0);
                weights = [wv / s, (wl + we) / s];
            }
            if c.param_upper("SMOOTHING OBJECTIVE").as_deref() == Some("GRADED") && initial_sweeps.is_none() {
                initial_sweeps = Some(2);
            }
            for p in ["CURVATURE REFINEMENT", "MESHING PREDICTOR"] {
                if c.has(p) && p == "MESHING PREDICTOR" && c.param_upper(p).as_deref() == Some("PREVIOUS") {
                    return c.err("MESHING PREDICTOR=PREVIOUS stöds inte ännu");
                }
            }
        }
        let mut elems = Vec::new();
        for &e in els {
            let el = &m.elements[e];
            if !matches!(el.etype.name(), "C3D8R" | "C3D8") {
                return am.err(format!("*ADAPTIVE MESH: {} stöds inte ännu (C3D8R, C3D8)", el.etype.name()));
            }
            let ctx = p.ctx(e, false, 0.0, 0.0);
            if let Some(mat) = ctx.material {
                if mat.name().starts_with("*EOS") || mat.hyperelastic().is_some() {
                    return am.err(format!("*ADAPTIVE MESH: materialet {} stöds inte i ALE-domänen (inkrementell spänningsuppdatering krävs)", mat.name()));
                }
            }
            let x: [[f64; 3]; 8] = std::array::from_fn(|k| m.nodes[el.nodes[k]]);
            let mass = ctx.density * volume(&x);
            let me = el.etype.mass(&ctx, MassKind::Lumped)?;
            let nd = p.edofs[e].len();
            let mut frac = [0.125; 8];
            if me.len() == nd * nd && mass > 0.0 {
                for (a, &(_, s)) in p.edofs[e].iter().enumerate() {
                    if s == 0 {
                        frac[a / 3] = me[a * nd + a] / mass;
                    }
                }
            }
            elems.push(AElem { e, nodes: std::array::from_fn(|k| el.nodes[k]), mass, frac, nbr: [None; 6] });
        }
        if elems.is_empty() {
            return Ok(None);
        }
        let mut faces: HashMap<Vec<usize>, (usize, usize)> = HashMap::new();
        for k in 0..elems.len() {
            for (f, fnodes) in crate::elements::continuum3d::HEX8_FACES.iter().enumerate() {
                let mut key: Vec<usize> = fnodes.iter().map(|&i| elems[k].nodes[i]).collect();
                key.sort_unstable();
                if let Some(&(k2, f2)) = faces.get(&key) {
                    elems[k].nbr[f] = Some((k2, f2));
                    elems[k2].nbr[f2] = Some((k, f));
                    faces.remove(&key);
                } else {
                    faces.insert(key, (k, f));
                }
            }
        }
        // randnoder (yttre ytor) följer materialet
        let mut boundary: BTreeSet<usize> = BTreeSet::new();
        for key in faces.keys() {
            boundary.extend(key.iter().copied());
        }
        let mut node_elems: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut node_nbrs: HashMap<usize, Vec<usize>> = HashMap::new();
        const EDGES: [(usize, usize); 12] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)];
        for (k, el) in elems.iter().enumerate() {
            for &n in &el.nodes {
                node_elems.entry(n).or_default().push(k);
            }
            for &(a, b) in &EDGES {
                let (na, nb) = (el.nodes[a], el.nodes[b]);
                let v = node_nbrs.entry(na).or_default();
                if !v.contains(&nb) {
                    v.push(nb);
                }
                let v = node_nbrs.entry(nb).or_default();
                if !v.contains(&na) {
                    v.push(na);
                }
            }
        }
        let mut nodes: Vec<usize> = node_elems.keys().copied().collect();
        nodes.sort_unstable();
        // noder som också hör till element utanför domänen följer materialet
        let mut outside: BTreeSet<usize> = BTreeSet::new();
        let inset: BTreeSet<usize> = elems.iter().map(|x| x.e).collect();
        for (e, el) in m.elements.iter().enumerate() {
            if !inset.contains(&e) {
                outside.extend(el.nodes.iter().copied());
            }
        }
        let free: Vec<usize> = nodes.iter().copied().filter(|n| !boundary.contains(n) && !outside.contains(n)).collect();
        if free.is_empty() {
            return fail(format!("*ADAPTIVE MESH, ELSET={set}: domänen har inga inre noder att utjämna"));
        }
        Ok(Some(Ale { elems, free, nodes, node_elems, node_nbrs, freq, sweeps, initial_sweeps: initial_sweeps.unwrap_or(5), weights, inc: 0, first: true, pending: Vec::new(), reloc: HashMap::new() }))
    }

    /// Nodmassorna för de inre noderna från elementmassorna.
    pub fn patch_mass(&self, mass: &mut super::LumpedMass) {
        if self.inc == 0 && self.reloc.is_empty() {
            return;
        }
        let mut mn: HashMap<usize, f64> = HashMap::new();
        for el in &self.elems {
            for (a, &n) in el.nodes.iter().enumerate() {
                *mn.entry(n).or_insert(0.0) += el.frac[a] * el.mass;
            }
        }
        for &n in &self.free {
            for c in 0..3 {
                mass.diag[n][c] = mn[&n];
            }
        }
    }

    /// Elementets förskjutningar utan nätets omflyttningar (`None` om inget
    /// av elementets noder har flyttats).
    pub fn hourglass_u(&self, nodes: &[usize], ue: &[f64]) -> Option<Vec<f64>> {
        if self.reloc.is_empty() || nodes.len() * 3 != ue.len() || !nodes.iter().any(|n| self.reloc.contains_key(n)) {
            return None;
        }
        let mut v = ue.to_vec();
        for (a, n) in nodes.iter().enumerate() {
            if let Some(r) = self.reloc.get(n) {
                for c in 0..3 {
                    v[3 * a + c] -= r[c];
                }
            }
        }
        Some(v)
    }

    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink) -> Result<()> {
        self.pending.clear();
        // stegets början: bara den första utjämningen (materialet i vila, massan
        // följer elementens nya volymer)
        let initial = ev.dt <= 0.0;
        if initial {
            if !self.first || ev.time > 0.0 {
                return Ok(());
            }
        } else {
            self.inc += 1;
            if self.inc % self.freq != 0 {
                return Ok(());
            }
        }
        let m = p.model;
        let pos = |n: usize, u: &[NodeVec]| -> [f64; 3] { std::array::from_fn(|c| m.nodes[n][c] + u[n][c]) };
        // utjämning: nya lägen för de inre noderna
        let mut x: HashMap<usize, [f64; 3]> = self.nodes.iter().map(|&n| (n, pos(n, ev.u))).collect();
        let xlag = x.clone();
        let fixed = |n: usize| ev.fixed.get(n).is_some_and(|&f| f & 0b111 != 0);
        let sweeps = if self.first { self.initial_sweeps } else { self.sweeps };
        self.first = false;
        for _ in 0..sweeps {
            let geo: Vec<([f64; 3], f64)> = self
                .elems
                .iter()
                .map(|el| {
                    let xe: [[f64; 3]; 8] = std::array::from_fn(|k| x[&el.nodes[k]]);
                    (centroid(&xe), volume(&xe).max(0.0))
                })
                .collect();
            let mut nx = x.clone();
            for &n in &self.free {
                if fixed(n) {
                    continue;
                }
                let mut vsum = 0.0;
                let mut vc = [0.0; 3];
                for &k in &self.node_elems[&n] {
                    let (c, v) = geo[k];
                    vsum += v;
                    for d in 0..3 {
                        vc[d] += v * c[d];
                    }
                }
                let nb = &self.node_nbrs[&n];
                let lap: [f64; 3] = std::array::from_fn(|d| nb.iter().map(|q| x[q][d]).sum::<f64>() / nb.len().max(1) as f64);
                let old = x[&n];
                let wsum = self.weights[0] + self.weights[1];
                let new: [f64; 3] = std::array::from_fn(|d| {
                    let vol = if vsum > 0.0 { vc[d] / vsum } else { old[d] };
                    (1.0 - wsum) * old[d] + self.weights[0] * vol + self.weights[1] * lap[d]
                });
                nx.insert(n, new);
            }
            x = nx;
        }
        if initial {
            // massan per element med den ursprungliga densiteten och den nya volymen
            for el in self.elems.iter_mut() {
                let x0: [[f64; 3]; 8] = std::array::from_fn(|k| xlag[&el.nodes[k]]);
                let x1: [[f64; 3]; 8] = std::array::from_fn(|k| x[&el.nodes[k]]);
                let (v0, v1) = (volume(&x0), volume(&x1));
                if v0 > 0.0 {
                    el.mass *= v1 / v0;
                }
            }
            let mut mn: HashMap<usize, f64> = HashMap::new();
            for el in &self.elems {
                for (a, &n) in el.nodes.iter().enumerate() {
                    *mn.entry(n).or_insert(0.0) += el.frac[a] * el.mass;
                }
            }
            for &n in &self.nodes {
                let xn = x[&n];
                let un: [f64; 3] = std::array::from_fn(|c| xn[c] - m.nodes[n][c]);
                let d: [f64; 3] = std::array::from_fn(|c| xn[c] - xlag[&n][c]);
                if d.iter().any(|v| *v != 0.0) {
                    let r = self.reloc.entry(n).or_insert([0.0; 3]);
                    for c in 0..3 {
                        r[c] += d[c];
                    }
                }
                let v: [f64; 3] = std::array::from_fn(|c| ev.v[n][c]);
                self.pending.push((n, un, v));
                if self.free.contains(&n) {
                    for c in 0..3 {
                        out.mass.diag[n][c] = mn[&n];
                    }
                }
            }
            self.inc = self.inc.max(1);
            return Ok(());
        }
        // advektion av elementtillståndet från det Lagrangeska nätet till det nya
        let ne = self.elems.len();
        let vlag: Vec<f64> = self.elems.iter().map(|el| volume(&std::array::from_fn(|k| xlag[&el.nodes[k]])).max(1e-300)).collect();
        let vbar: Vec<[f64; 3]> = self.elems.iter().map(|el| std::array::from_fn(|c| el.nodes.iter().map(|&n| ev.v[n][c]).sum::<f64>() / 8.0)).collect();
        let st0: Vec<Vec<f64>> = self.elems.iter().map(|el| out.state_new[el.e].clone()).collect();
        let mut sv: Vec<Vec<f64>> = (0..ne).map(|k| st0[k].iter().map(|s| s * vlag[k]).collect()).collect();
        let mut vol = vlag.clone();
        let mut mas: Vec<f64> = self.elems.iter().map(|el| el.mass).collect();
        let mut mom: Vec<[f64; 3]> = (0..ne).map(|k| std::array::from_fn(|c| mas[k] * vbar[k][c])).collect();
        let mut touched = vec![false; ne];
        for k in 0..ne {
            let el = &self.elems[k];
            for (f, fnodes) in crate::elements::continuum3d::HEX8_FACES.iter().enumerate() {
                let Some((k2, _)) = el.nbr[f] else { continue };
                if k2 < k {
                    continue;
                }
                let a: [[f64; 3]; 4] = std::array::from_fn(|i| xlag[&el.nodes[fnodes[i]]]);
                let b: [[f64; 3]; 4] = std::array::from_fn(|i| x[&el.nodes[fnodes[i]]]);
                let s = swept(&a, &b);
                // ytan går utåt från k till det nya nätet: k växer och tar emot material
                let (d, r) = if s < 0.0 { (k, k2) } else { (k2, k) };
                let frac = (s.abs() / vlag[d]).min(0.5);
                if frac <= 0.0 {
                    continue;
                }
                touched[d] = true;
                touched[r] = true;
                let dm = self.elems[d].mass * frac;
                let dv = vlag[d] * frac;
                vol[d] -= dv;
                vol[r] += dv;
                mas[d] -= dm;
                mas[r] += dm;
                for i in 0..sv[d].len().min(sv[r].len()) {
                    let q = st0[d][i] * dv;
                    sv[d][i] -= q;
                    sv[r][i] += q;
                }
                for c in 0..3 {
                    mom[d][c] -= dm * vbar[d][c];
                    mom[r][c] += dm * vbar[d][c];
                }
            }
        }
        let mut dvel = vec![[0.0f64; 3]; ne];
        for k in 0..ne {
            let el = &mut self.elems[k];
            el.mass = mas[k].max(0.0);
            // element utan flöde behåller tillståndet exakt; NLGEOM-tillståndets
            // markörer (`kinematics::push_ext`, två sista värdena) advekteras inte
            if vol[k] > 0.0 && touched[k] {
                let n = sv[k].len();
                let keep = if crate::kinematics::ext(&st0[k]).is_some() { 2 } else { 0 };
                let mut new: Vec<f64> = sv[k].iter().map(|q| q / vol[k]).collect();
                new[n - keep..].copy_from_slice(&st0[k][n - keep..]);
                out.state_new[el.e] = new;
            }
            if el.mass > 0.0 {
                for c in 0..3 {
                    dvel[k][c] = mom[k][c] / el.mass - vbar[k][c];
                }
            }
        }
        // nodmassor och hastigheter (elementmittprojektion)
        let mut num: HashMap<usize, [f64; 4]> = HashMap::new();
        let mut mn: HashMap<usize, f64> = HashMap::new();
        for (k, el) in self.elems.iter().enumerate() {
            for (a, &n) in el.nodes.iter().enumerate() {
                let e = num.entry(n).or_insert([0.0; 4]);
                for c in 0..3 {
                    e[c] += el.mass * dvel[k][c];
                }
                e[3] += el.mass;
                *mn.entry(n).or_insert(0.0) += el.frac[a] * el.mass;
            }
        }
        for &n in &self.nodes {
            let e = num[&n];
            let v: [f64; 3] = std::array::from_fn(|c| ev.v[n][c] + if e[3] > 0.0 { e[c] / e[3] } else { 0.0 });
            let xn = x[&n];
            let un: [f64; 3] = std::array::from_fn(|c| xn[c] - m.nodes[n][c]);
            let d: [f64; 3] = std::array::from_fn(|c| xn[c] - xlag[&n][c]);
            if d.iter().any(|v| *v != 0.0) {
                let r = self.reloc.entry(n).or_insert([0.0; 3]);
                for c in 0..3 {
                    r[c] += d[c];
                }
            }
            self.pending.push((n, un, v));
            if !self.free.contains(&n) {
                continue;
            }
            for c in 0..3 {
                out.mass.diag[n][c] = mn[&n];
            }
        }
        Ok(())
    }

    /// Nya lägen och hastigheter efter utjämningen (utom låsta frihetsgrader).
    pub fn after_forces(&mut self, u: &mut [NodeVec], vh: &mut [NodeVec], fixed: &[u16]) {
        for &(n, un, v) in &self.pending {
            for c in 0..3 {
                if fixed[n] & (1 << c) == 0 {
                    u[n][c] = un[c];
                    vh[n][c] = v[c];
                }
            }
        }
        self.pending.clear();
    }
}
