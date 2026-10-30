//! Jämförelse av RustyFEM-resultat mot en baslinje.
//!
//! Felet för ett fält i en frame är max |RustyFEM − baslinje| över alla noder/
//! integrationspunkter och komponenter, delat med största |referensvärde| i
//! fältet (samma normering som `rustyfem validate`).
//!
//! - Nodfält (U, UR, RF, RM, NT11, ...) styrs av baslinjens noder: en nod eller
//!   komponent som saknas i RustyFEM räknas som 0 (inaktiv frihetsgrad).
//! - Elementfält (S, E, PEEQ, ...) styrs av RustyFEMs punkter: varje punkt måste
//!   finnas i baslinjen. referenspunkter utan motsvarighet (t.ex. sektionspunkter i
//!   balkar) räknas och redovisas men ger inte fel.
//! - FRAMEVALUE jämför framevärdet (stegtid, egenfrekvens, egenvärde) relativt.
//! - "H:NAMN" jämför sista värdet av history-utdata NAMN.
//! - "HA:NAMN" jämför alla värden av history-utdata NAMN (parade på tid eller
//!   modnummer, normerat med största |referensvärde|), t.ex. EIGVAL, PF1, EM2.
//! - "MODEVALUE" jämför egenvärdet i framebeskrivningen ("EigenValue = ..." i
//!   *BUCKLE, "Value = ..." i *FREQUENCY) relativt. baslinjen har bara 5
//!   värdesiffror där, så toleransen bör vara ≥ 5e-5. (Knäckningssteg har inget
//!   annat egenvärde i referensresultaten.)
//! - `floor` i sidofilen: nodfält normeras minst med floor · största |referensvärde|
//!   i fältet över stegets alla frames (för numeriskt brus i komponenter som är
//!   noll i en mod, t.ex. rotationer i en ren axialmod).
//! - `floor_elem`: samma för elementfält men med största |referensvärde| i fältet
//!   över fallets alla steg och frames, t.ex. för skadevariabler och plastiska töjningar som
//!   är ≈ 0 (numeriskt brus) i frames före flytning eller skadeinitiering.

use super::case::CaseSpec;
use crate::fem::output::{Frame, ResultSet};
use std::fmt::Write as _;

#[derive(Debug, Clone)]
pub struct FieldReport {
    pub step: usize,
    pub frame: usize,
    pub field: String,
    pub err: f64,
    pub max_ref: f64,
    pub n: usize,
    /// referenspunkter utan motsvarighet i RustyFEM (elementfält).
    pub unmatched: usize,
    pub tol: f64,
    /// Var största felet finns.
    pub worst_at: String,
}

#[derive(Debug, Clone, Default)]
pub struct CaseReport {
    pub fields: Vec<FieldReport>,
    /// Fel som gör att fallet inte godkänns (saknade fält, olika antal steg, ...).
    pub problems: Vec<String>,
    pub notes: Vec<String>,
}

impl CaseReport {
    pub fn passed(&self) -> bool {
        self.problems.is_empty() && self.fields.iter().all(|f| f.err <= f.tol)
    }

    pub fn worst(&self) -> f64 {
        self.fields.iter().map(|f| f.err).fold(0.0, f64::max)
    }

    pub fn text(&self) -> String {
        let mut s = String::new();
        for f in &self.fields {
            let mark = if f.err <= f.tol { "OK " } else { "FEL" };
            let _ = write!(
                s,
                "  {mark} steg {} frame {:>3} {:<10} max rel. fel {:.2e} (tol {:.0e}, |max| {:.4e}, {} värden",
                f.step, f.frame, f.field, f.err, f.tol, f.max_ref, f.n
            );
            if f.unmatched > 0 {
                let _ = write!(s, ", {} referenspunkter ej jämförda", f.unmatched);
            }
            s.push(')');
            if f.err > f.tol && !f.worst_at.is_empty() {
                let _ = write!(s, " – största fel vid {}", f.worst_at);
            }
            s.push('\n');
        }
        for p in &self.problems {
            let _ = writeln!(s, "  FEL {p}");
        }
        for n in &self.notes {
            let _ = writeln!(s, "  not: {n}");
        }
        s
    }
}

fn compare_node(ours: &Frame, base: &Frame, field: &str, sign: f64, norm_min: f64) -> Option<(f64, f64, usize, String)> {
    let a = base.node.get(field)?;
    let o = ours.node.get(field);
    let mut maxa = 0.0f64;
    let mut maxe = 0.0f64;
    let mut n = 0;
    let mut at = String::new();
    for (key, comps) in &a.items {
        for &(id, va) in comps {
            let label = &a.labels[id as usize];
            let vo = o.and_then(|f| f.get(key, label)).unwrap_or(0.0) * sign;
            maxa = maxa.max(va.abs());
            let e = (vo - va).abs();
            if e > maxe {
                maxe = e;
                at = format!("{}.{} {label} (RustyFEM {vo:.6e}, referens {va:.6e})", key.0, key.1);
            }
            n += 1;
        }
    }
    let norm = maxa.max(norm_min);
    Some((if norm > 0.0 { maxe / norm } else { maxe }, maxa, n, at))
}

/// Största |värde| i nodfältet `field` i en frame.
fn node_max(f: &Frame, field: &str) -> f64 {
    f.node.get(field).map_or(0.0, |a| a.items.values().flat_map(|c| c.iter().map(|p| p.1.abs())).fold(0.0, f64::max))
}

fn elem_max(f: &Frame, field: &str) -> f64 {
    f.elem.get(field).map_or(0.0, |a| a.items.values().flat_map(|c| c.iter().map(|p| p.1.abs())).fold(0.0, f64::max))
}

/// Egenvärdet i en framebeskrivning ("EigenValue = x" eller "Value = x").
pub fn description_value(desc: &str) -> Option<f64> {
    let i = desc.find("Value =")?;
    desc[i + 7..].split_whitespace().next()?.parse().ok()
}

/// (fel, max, antal, ej matchade referenspunkter, saknade i baslinjen, plats)
fn compare_elem(ours: &Frame, base: &Frame, field: &str, sign: f64, norm_min: f64) -> Option<(f64, f64, usize, usize, usize, String)> {
    let a = base.elem.get(field)?;
    let o = ours.elem.get(field)?;
    let mut maxa = 0.0f64;
    let mut maxe = 0.0f64;
    let mut n = 0;
    let mut missing = 0;
    let mut at = String::new();
    for (key, comps) in &o.items {
        if !a.items.contains_key(key) {
            missing += 1;
            continue;
        }
        for &(id, vo) in comps {
            let label = &o.labels[id as usize];
            let Some(va) = a.get(key, label) else { continue };
            let vo = vo * sign;
            maxa = maxa.max(va.abs());
            let e = (vo - va).abs();
            if e > maxe {
                maxe = e;
                at = format!("{}.{} ip {} {label} (RustyFEM {vo:.6e}, referens {va:.6e})", key.0, key.1, key.2);
            }
            n += 1;
        }
    }
    let unmatched = a.items.keys().filter(|k| !o.items.contains_key(k)).count();
    let norm = maxa.max(norm_min);
    Some((if norm > 0.0 { maxe / norm } else { maxe }, maxa, n, unmatched, missing, at))
}

/// Jämför RustyFEM (`ours`) mot baslinjen (`base`) enligt sidofilen.
pub fn compare(ours: &ResultSet, base: &ResultSet, spec: &CaseSpec) -> CaseReport {
    let mut rep = CaseReport::default();
    if ours.steps.len() != base.steps.len() {
        rep.problems.push(format!("RustyFEM har {} steg, referens {}", ours.steps.len(), base.steps.len()));
    }
    for (so, sa) in ours.steps.iter().zip(&base.steps) {
        if !spec.steps.is_empty() && !spec.steps.contains(&sa.index) {
            continue;
        }
        let pairs: Vec<(&Frame, &Frame)> = if spec.all_frames || sa.procedure.contains("FREQUENCY") || sa.procedure.contains("BUCKLE") {
            // jämför frames med samma framenummer (moder i egenvärdessteg)
            if so.frames.len() != sa.frames.len() {
                rep.problems.push(format!("steg {}: RustyFEM har {} frames, referens {}", sa.index, so.frames.len(), sa.frames.len()));
            }
            let mut v = Vec::new();
            for fo in &so.frames {
                match sa.frames.iter().find(|f| f.index == fo.index) {
                    Some(fa) => v.push((fo, fa)),
                    None => rep.problems.push(format!("steg {}: baslinjen saknar frame {}", sa.index, fo.index)),
                }
            }
            v
        } else {
            match (so.frames.last(), sa.frames.last()) {
                (Some(a), Some(b)) => vec![(a, b)],
                _ => {
                    rep.problems.push(format!("steg {}: frames saknas", sa.index));
                    continue;
                }
            }
        };
        for (fo, fa) in pairs {
            for field in &spec.fields {
                let tol = spec.tol_for(field);
                let mut push = |err: f64, max_ref: f64, n: usize, unmatched: usize, worst_at: String| {
                    rep.fields.push(FieldReport { step: sa.index, frame: fa.index, field: field.clone(), err, max_ref, n, unmatched, tol, worst_at })
                };
                if field == "FRAMEVALUE" {
                    let e = (fo.value - fa.value).abs() / fa.value.abs().max(1e-300);
                    push(if fa.value == 0.0 { (fo.value - fa.value).abs() } else { e }, fa.value.abs(), 1, 0, format!("RustyFEM {:.6e}, referens {:.6e}", fo.value, fa.value));
                    continue;
                }
                if field == "MODEVALUE" {
                    // frames utan egenvärde (bastillstånd, statiska steg) hoppas över
                    if description_value(&fa.description).is_none() {
                        continue;
                    }
                    match (description_value(&fo.description), description_value(&fa.description)) {
                        (Some(a), Some(b)) => {
                            let e = if b == 0.0 { (a - b).abs() } else { (a - b).abs() / b.abs() };
                            push(e, b.abs(), 1, 0, format!("RustyFEM {a:.6e}, referens {b:.6e}"));
                        }
                        _ => rep.problems.push(format!("steg {} frame {}: egenvärde saknas i beskrivningen", sa.index, fa.index)),
                    }
                    continue;
                }
                if let Some(h) = field.strip_prefix("HA:") {
                    // alla värden i history-utdata, parade på tid (modnummer i egenvärdessteg),
                    // normerat med största |referensvärde|; jämförs en gång per steg
                    if fa.index != sa.frames.last().map_or(fa.index, |f| f.index) {
                        continue;
                    }
                    let find = |rs: &crate::fem::output::StepResult| rs.history.iter().find(|x| x.name.eq_ignore_ascii_case(h)).map(|x| x.data.clone());
                    match (find(so), find(sa)) {
                        (Some(a), Some(b)) => {
                            // tecknet får skifta per punkt om fältet står i sign_free (t.ex. PF följer modens tecken)
                            let free = spec.sign_free.contains(field);
                            let maxb = b.iter().fold(0.0f64, |m, p| m.max(p.1.abs()));
                            let mut maxe = 0.0f64;
                            let mut at = String::new();
                            for &(t, vb) in &b {
                                let va = a.iter().find(|p| (p.0 - t).abs() <= 1e-9 * t.abs().max(1.0)).map(|p| p.1);
                                let Some(va) = va else {
                                    rep.problems.push(format!("steg {}: RustyFEM saknar {h} vid tid/mod {t}", sa.index));
                                    continue;
                                };
                                let e = if free { (va.abs() - vb.abs()).abs() } else { (va - vb).abs() };
                                if e > maxe {
                                    maxe = e;
                                    at = format!("tid/mod {t}: RustyFEM {va:.6e}, referens {vb:.6e}");
                                }
                            }
                            push(if maxb > 0.0 { maxe / maxb } else { maxe }, maxb, b.len(), 0, at);
                        }
                        // saknas i båda (t.ex. EIGFREQ i ett statiskt steg): inget att jämföra
                        (None, None) => {}
                        (None, _) => rep.problems.push(format!("steg {}: RustyFEM saknar history-utdata {h}", sa.index)),
                        (_, None) => rep.problems.push(format!("steg {}: baslinjen saknar history-utdata {h}", sa.index)),
                    }
                    continue;
                }
                if let Some(h) = field.strip_prefix("H:") {
                    let last = |rs: &crate::fem::output::StepResult| rs.history.iter().find(|x| x.name.eq_ignore_ascii_case(h)).and_then(|x| x.data.last().map(|p| p.1));
                    match (last(so), last(sa)) {
                        (Some(a), Some(b)) => push((a - b).abs() / b.abs().max(1e-300), b.abs(), 1, 0, format!("RustyFEM {a:.6e}, referens {b:.6e}")),
                        (None, _) => rep.problems.push(format!("steg {}: RustyFEM saknar history-utdata {h}", sa.index)),
                        (_, None) => rep.problems.push(format!("steg {}: baslinjen saknar history-utdata {h}", sa.index)),
                    }
                    continue;
                }
                let free = spec.sign_free.contains(field);
                let best = |f: &dyn Fn(f64) -> Option<(f64, f64, usize, usize, usize, String)>| -> Option<(f64, f64, usize, usize, usize, String)> {
                    let p = f(1.0)?;
                    if free {
                        let m = f(-1.0)?;
                        if m.0 < p.0 {
                            return Some(m);
                        }
                    }
                    Some(p)
                };
                if fa.node.contains_key(field) {
                    let norm_min = if spec.floor > 0.0 { spec.floor * sa.frames.iter().map(|f| node_max(f, field)).fold(0.0, f64::max) } else { 0.0 };
                    // Explicit absolute tolerance handles a physically zero field
                    // without weakening the relative check on nonzero results.
                    let norm_min = norm_min.max(spec.absolute_norm_floor(field));
                    if let Some((e, m, n, _, _, at)) = best(&|s| compare_node(fo, fa, field, s, norm_min).map(|(e, m, n, at)| (e, m, n, 0, 0, at))) {
                        push(e, m, n, 0, at);
                    }
                } else if fa.elem.contains_key(field) {
                    if !fo.elem.contains_key(field) {
                        rep.problems.push(format!("steg {} frame {}: RustyFEM saknar fältet {field}", sa.index, fa.index));
                        continue;
                    }
                    let norm_min = if spec.floor_elem > 0.0 { spec.floor_elem * base.steps.iter().flat_map(|st| st.frames.iter()).map(|f| elem_max(f, field)).fold(0.0, f64::max) } else { 0.0 };
                    let norm_min = norm_min.max(spec.absolute_norm_floor(field));
                    if let Some((e, m, n, un, miss, at)) = best(&|s| compare_elem(fo, fa, field, s, norm_min)) {
                        if miss > 0 {
                            rep.problems.push(format!("steg {} frame {}: {miss} punkter i {field} saknas i baslinjen", sa.index, fa.index));
                        }
                        push(e, m, n, un, at);
                    }
                } else if fo.node.contains_key(field) || fo.elem.contains_key(field) {
                    rep.problems.push(format!("steg {} frame {}: baslinjen saknar fältet {field} (begärs det i *OUTPUT?)", sa.index, fa.index));
                } else {
                    let note = format!("fältet {field} finns varken i RustyFEM eller baslinjen");
                    if !rep.notes.contains(&note) {
                        rep.notes.push(note);
                    }
                }
            }
        }
    }
    rep
}
