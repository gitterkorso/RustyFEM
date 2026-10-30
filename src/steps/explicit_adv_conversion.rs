//! Område explicit_adv: omvandling av kontinuumelement till SPH-partiklar i
//! analysens början (`*SECTION CONTROLS, ELEMENT CONVERSION=YES`) i
//! explicit analys.
//!
//! Anropas från `inp::read` när filen är läst (`apply`). Föräldraelementen
//! (C3D8R, C3D8) i ett *SOLID SECTION vars CONTROLS= har ELEMENT
//! CONVERSION=YES ersätts av vilande element (`SPH_PARENT`, ingen kraft eller
//! massa) och partiklar (PC3D) skapas:
//!
//! - `SPH CONVERSION=PER ELEMENT` med `CONVERSION CRITERION=TIME` och
//!   tröskeln 0 (fjärde dataraden: antal partiklar per isoparametrisk
//!   riktning n ≤ 7, tröskel): n³ partiklar i ξᵢ = −1 + (2i + 1)/n, var och en
//!   med volymen Vₑ/n³ (massan delas lika) och L = ½(Vₑ/n³)^⅓.
//! - `SPH CONVERSION=BACKGROUND GRID` (standard): partiklar i ett likformigt
//!   rutnät med sidan a (fjärde dataraden, standard: elementens medelstorlek
//!   Vₑ^⅓) med cellernas mittpunkter i c₀ + (k + ½)a, c₀ = elementmängdens
//!   masscentrum (avläst: 6³ partiklar i en kub med sidan 6a); bara punkter inuti
//!   ett föräldraelement ger partiklar, alla med volymen a³ (L = a/2).
//!
//! Partiklarna får nya nod- och elementnummer efter de största i instansen
//! (testfallen jämför därför energier och föräldranätets
//! oberoende storheter). Dessutom skapas
//! mängderna `<ELSET>_SECT_SPH` (partiklarna), `<ELSET>_SECT_E_SPH` (deras
//! noder), `ALL_GENERATED_ELEMENTS_SPH` och nodytan `<ELSET>_SECT_NE_SPH`.
//! Begynnelsehastigheten: partiklarna i ett föräldraelement som har en nod
//! med *INITIAL CONDITIONS, TYPE=VELOCITY får nodens hastighet (villkoret
//! tillämpas på partiklarna i alla element som rör nodmängden), och *DLOAD (GRAV) på föräldraelementen gäller partiklarna. Konvertering vid
//! en senare tid eller efter töjnings-/spänningskriterier stöds inte ännu.

use crate::elements::explicit_adv::AdvElem;
use crate::fem::model::{parse_f64, DataLine, FeModel, RawKeyword, SectionDef, SectionKind, Surface};
use crate::model::{fail, Result};
use std::collections::{BTreeMap, HashMap};

/// Vilande föräldraelement efter omvandlingen (inga krafter eller massor).
pub static SPH_PARENT8: AdvElem = AdvElem { name: "SPH_PARENT", nodes: 8, rot: false };

/// Nyckelordet med partiklarnas karakteristiska längd (elementindex, L).
pub const LENGTHS: &str = "RUSTYFEM SPH LENGTH";

/// Formfunktionerna för en åttanodig hexaeder i ξ.
pub fn hex_shape(xi: [f64; 3]) -> [f64; 8] {
    let nat: [[f64; 3]; 8] = [[-1., -1., -1.], [1., -1., -1.], [1., 1., -1.], [-1., 1., -1.], [-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]];
    std::array::from_fn(|a| 0.125 * (1.0 + nat[a][0] * xi[0]) * (1.0 + nat[a][1] * xi[1]) * (1.0 + nat[a][2] * xi[2]))
}

fn map(x: &[[f64; 3]], xi: [f64; 3]) -> [f64; 3] {
    let n = hex_shape(xi);
    std::array::from_fn(|c| (0..8).map(|a| n[a] * x[a][c]).sum())
}

/// Isoparametriska koordinater för punkten `p` i hexaedern `x` (Newton med
/// differensjacobian), `None` om iterationen inte konvergerar.
pub fn inverse(x: &[[f64; 3]], p: [f64; 3]) -> Option<[f64; 3]> {
    let mut xi = [0.0; 3];
    for _ in 0..30 {
        let f = map(x, xi);
        let r = [p[0] - f[0], p[1] - f[1], p[2] - f[2]];
        let h = 1e-6;
        let mut j = [[0.0; 3]; 3];
        for k in 0..3 {
            let mut xh = xi;
            xh[k] += h;
            let fh = map(x, xh);
            for c in 0..3 {
                j[c][k] = (fh[c] - f[c]) / h;
            }
        }
        let (ji, det) = crate::elements::common::inv3(j);
        if det.abs() < 1e-300 {
            return None;
        }
        let d: [f64; 3] = std::array::from_fn(|c| (0..3).map(|k| ji[c][k] * r[k]).sum());
        for c in 0..3 {
            xi[c] += d[c];
        }
        if d.iter().map(|v| v.abs()).sum::<f64>() < 1e-12 {
            return Some(xi);
        }
    }
    None
}

fn hex_volume(x: &[[f64; 3]]) -> f64 {
    let xs: Vec<[f64; 3]> = x.to_vec();
    crate::steps::explicit::dt::geometry(crate::elements::Family::Continuum3D, "C3D8R", &xs).map_or(0.0, |g| g.vol)
}

/// Omvandlar föräldraelementen till partiklar (se modulens dokumentation).
pub fn apply(m: &mut FeModel) -> Result<()> {
    let mut jobs: Vec<(usize, RawKeyword)> = Vec::new();
    for (si, sec) in m.sections.iter().enumerate() {
        if sec.raw.name != "SOLID SECTION" {
            continue;
        }
        let Some(name) = sec.raw.param_upper("CONTROLS") else { continue };
        let Some(sc) = m.extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())) else { continue };
        if sc.param_upper("ELEMENT CONVERSION").as_deref() != Some("YES") {
            continue;
        }
        jobs.push((si, sc.clone()));
    }
    if jobs.is_empty() {
        return Ok(());
    }
    let mut all_generated: Vec<usize> = Vec::new();
    let mut lengths: Vec<(usize, f64)> = Vec::new();
    let mut parent_of: HashMap<usize, Vec<(usize, [f64; 8])>> = HashMap::new();
    for (si, sc) in jobs {
        let sec = m.sections[si].clone();
        let field = |row: usize, k: usize| -> Option<f64> { sc.data.get(row).and_then(|d| d.fields.get(k)).filter(|f| !f.trim().is_empty()).and_then(|f| parse_f64(f)) };
        let per_element = match sc.param_upper("SPH CONVERSION").as_deref() {
            Some("PER ELEMENT") => true,
            None | Some("BACKGROUND GRID") => false,
            Some(o) => return sc.err(format!("SPH CONVERSION={o} stöds inte (PER ELEMENT, BACKGROUND GRID)")),
        };
        if per_element {
            match sc.param_upper("CONVERSION CRITERION").as_deref() {
                None | Some("TIME") => {}
                Some(o) => return sc.err(format!("CONVERSION CRITERION={o} stöds inte ännu (bara TIME med tröskeln 0)")),
            }
            if field(3, 1).unwrap_or(0.0) != 0.0 {
                return sc.err("CONVERSION CRITERION=TIME: bara omvandling i analysens början (tröskeln 0) stöds ännu");
            }
        }
        if sc.data.get(3).and_then(|d| d.fields.get(1)).is_some_and(|f| !f.trim().is_empty() && parse_f64(f).is_none()) {
            return sc.err("BACKGROUND GRID: orienteringar för rutnätet stöds inte ännu");
        }
        let Some(set) = m.elsets.get(&sec.elset).cloned() else { return sec.raw.err(format!("elementsetet '{}' finns inte", sec.elset)) };
        let inst = set.first().map(|&e| m.elements[e].inst).unwrap_or(0);
        for &e in &set {
            let name = m.elements[e].etype.name();
            if !matches!(name, "C3D8R" | "C3D8") {
                return sec.raw.err(format!("ELEMENT CONVERSION: {name} stöds inte ännu (C3D8R, C3D8)"));
            }
        }
        // positioner, volym per partikel och föräldraelementets formfunktioner
        let mut gen: Vec<(usize, [f64; 3], f64, [f64; 8])> = Vec::new();
        if per_element {
            let n = field(3, 0).unwrap_or(1.0).round().clamp(1.0, 7.0) as usize;
            for &e in &set {
                let x: Vec<[f64; 3]> = m.elements[e].nodes.iter().map(|&k| m.nodes[k]).collect();
                let v = hex_volume(&x) / (n * n * n) as f64;
                for k in 0..n {
                    for j in 0..n {
                        for i in 0..n {
                            let xi = [-1.0 + (2 * i + 1) as f64 / n as f64, -1.0 + (2 * j + 1) as f64 / n as f64, -1.0 + (2 * k + 1) as f64 / n as f64];
                            gen.push((e, map(&x, xi), v, hex_shape(xi)));
                        }
                    }
                }
            }
        } else {
            let vols: Vec<f64> = set.iter().map(|&e| hex_volume(&m.elements[e].nodes.iter().map(|&k| m.nodes[k]).collect::<Vec<_>>())).collect();
            let vtot: f64 = vols.iter().sum();
            let a = field(3, 0).filter(|&a| a > 0.0).unwrap_or_else(|| (vtot / set.len() as f64).cbrt());
            // origo i masscentrum (elementens tyngdpunkter viktade med volymen)
            let mut c0 = [0.0; 3];
            for (q, &e) in set.iter().enumerate() {
                let x: Vec<[f64; 3]> = m.elements[e].nodes.iter().map(|&k| m.nodes[k]).collect();
                let c = map(&x, [0.0; 3]);
                for d in 0..3 {
                    c0[d] += c[d] * vols[q] / vtot;
                }
            }
            let mut taken: HashMap<(i64, i64, i64), ()> = HashMap::new();
            for &e in &set {
                let x: Vec<[f64; 3]> = m.elements[e].nodes.iter().map(|&k| m.nodes[k]).collect();
                let lo: [f64; 3] = std::array::from_fn(|d| x.iter().map(|p| p[d]).fold(f64::INFINITY, f64::min));
                let hi: [f64; 3] = std::array::from_fn(|d| x.iter().map(|p| p[d]).fold(f64::NEG_INFINITY, f64::max));
                let r = |d: usize| (((lo[d] - c0[d]) / a - 0.5 - 1e-9).ceil() as i64, ((hi[d] - c0[d]) / a - 0.5 + 1e-9).floor() as i64);
                let (rx, ry, rz) = (r(0), r(1), r(2));
                for kz in rz.0..=rz.1 {
                    for ky in ry.0..=ry.1 {
                        for kx in rx.0..=rx.1 {
                            if taken.contains_key(&(kx, ky, kz)) {
                                continue;
                            }
                            let p = [c0[0] + (kx as f64 + 0.5) * a, c0[1] + (ky as f64 + 0.5) * a, c0[2] + (kz as f64 + 0.5) * a];
                            let Some(xi) = inverse(&x, p) else { continue };
                            if xi.iter().all(|v| v.abs() <= 1.0 + 1e-9) {
                                taken.insert((kx, ky, kz), ());
                                gen.push((e, p, a * a * a, hex_shape(xi)));
                            }
                        }
                    }
                }
            }
        }
        if gen.is_empty() {
            return sec.raw.err("ELEMENT CONVERSION: inga partiklar skapades (för stort rutnät?)");
        }
        // nya nod- och elementnummer efter de största i instansen
        let mut nlab = (0..m.nodes.len()).filter(|&i| m.node_inst[i] == inst).map(|i| m.node_label[i]).max().unwrap_or(0);
        let mut elab = m.elements.iter().filter(|el| el.inst == inst).map(|el| el.label).max().unwrap_or(0);
        let pc3d = crate::elements::lookup("PC3D").expect("PC3D är registrerat");
        let new_sec = m.sections.len();
        let lmean = gen.iter().map(|g| 0.5 * g.2.cbrt()).sum::<f64>() / gen.len() as f64;
        let mut raw = sec.raw.clone();
        raw.data = vec![DataLine { fields: vec![format!("{lmean:e}")], line: sec.raw.line, continued: false }];
        let set_name = format!("{}_SECT_SPH", sec.elset);
        m.sections.push(SectionDef { elset: set_name.clone(), material: sec.material.clone(), material_index: sec.material_index, kind: SectionKind::Solid { thickness: Some(lmean) }, raw });
        let mut elems = Vec::new();
        let mut nodes = Vec::new();
        for (pe, p, v, sh) in gen {
            nlab += 1;
            elab += 1;
            let n = m.add_node(inst, nlab, p);
            let e = m.add_element(pc3d, inst, elab, vec![n])?;
            m.elements[e].section = Some(new_sec);
            lengths.push((e, 0.5 * v.cbrt()));
            parent_of.entry(pe).or_default().push((n, sh));
            elems.push(e);
            nodes.push(n);
            let _ = e;
        }
        all_generated.extend(&elems);
        let nset = format!("{}_SECT_E_SPH", sec.elset);
        m.nsets.insert(nset, nodes.clone());
        m.surfaces.insert(format!("{}_SECT_NE_SPH", sec.elset), Surface { kind: "NODE".into(), faces: Vec::new(), nodes: nodes.iter().map(|&n| (n, 1.0)).collect(), source: sec.raw.location() });
        m.elsets.insert(set_name, elems);
        // föräldrarna blir vilande
        for &e in &set {
            m.elements[e].etype = &SPH_PARENT8;
        }
    }
    let mut ag = m.elsets.get("ALL_GENERATED_ELEMENTS_SPH").cloned().unwrap_or_default();
    ag.extend(all_generated);
    m.elsets.insert("ALL_GENERATED_ELEMENTS_SPH".into(), ag);
    // partiklarnas längder (massan per partikel)
    let mut kw = RawKeyword { name: LENGTHS.into(), params: Vec::new(), data: Vec::new(), file: String::new(), line: 0 };
    for (e, l) in &lengths {
        kw.data.push(DataLine { fields: vec![e.to_string(), format!("{l:e}")], line: 0, continued: false });
    }
    m.extra.push(kw);
    // begynnelsehastigheten från föräldraelementets noder
    let mut vel: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
    let mut add = Vec::new();
    for kw in m.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.param_upper("TYPE").as_deref() == Some("VELOCITY")) {
        for d in &kw.data {
            let Ok(n) = d.fields[0].trim_start_matches('#').parse::<usize>() else { continue };
            let vals: Vec<f64> = d.fields[1..].iter().filter_map(|f| parse_f64(f)).collect();
            if vals.len() >= 2 && (1.0..=3.0).contains(&vals[0]) {
                vel.entry(n).or_insert([0.0; 3])[vals[0] as usize - 1] = vals[1];
            }
        }
        let _ = kw;
    }
    if !vel.is_empty() {
        let mut out = RawKeyword { name: "INITIAL CONDITIONS".into(), params: vec![("TYPE".into(), "VELOCITY".into())], data: Vec::new(), file: String::new(), line: 0 };
        for (&pe, list) in &parent_of {
            let pn = m.elements[pe].nodes.clone();
            for (n, _sh) in list {
                for c in 0..3 {
                    // partiklarna i ett föräldraelement som rör en nod med
                    // begynnelsehastighet får hela nodens hastighet (största beloppet)
                    let v = pn.iter().filter_map(|k| vel.get(k).map(|x| x[c])).fold(0.0f64, |a, b| if b.abs() > a.abs() { b } else { a });
                    if v != 0.0 {
                        out.data.push(DataLine { fields: vec![format!("#{n}"), (c + 1).to_string(), format!("{v:e}")], line: 0, continued: false });
                    }
                }
            }
        }
        add.push(out);
    }
    m.extra.extend(add);
    // *DLOAD (GRAV, BX, BY, BZ) på föräldrarna gäller partiklarna
    let elem_of_node: HashMap<usize, usize> = m.elements.iter().enumerate().filter(|(_, el)| el.etype.name() == "PC3D").map(|(e, el)| (el.nodes[0], e)).collect();
    for step in m.steps.iter_mut() {
        for blk in step.dloads.iter_mut() {
            let mut extra = Vec::new();
            for dl in &blk.items {
                if let Some(list) = parent_of.get(&dl.elem) {
                    if !matches!(dl.kind.as_str(), "GRAV" | "BX" | "BY" | "BZ") {
                        return fail(format!("ELEMENT CONVERSION: *DLOAD {} på föräldraelement stöds inte", dl.kind));
                    }
                    for (n, _) in list {
                        let mut d = dl.clone();
                        d.elem = elem_of_node[n];
                        extra.push(d);
                    }
                }
            }
            blk.items.retain(|dl| !parent_of.contains_key(&dl.elem));
            blk.items.extend(extra);
        }
    }
    Ok(())
}

/// Partikelns karakteristiska längd efter omvandlingen (elementindex → L).
pub fn lengths(m: &FeModel) -> HashMap<usize, f64> {
    let mut h = HashMap::new();
    for kw in m.extra.iter().filter(|k| k.name == LENGTHS) {
        for d in &kw.data {
            if let (Ok(e), Some(l)) = (d.fields[0].parse::<usize>(), parse_f64(&d.fields[1])) {
                h.insert(e, l);
            }
        }
    }
    h
}
