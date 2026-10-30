//! Kontakt i explicit dynamik: *CONTACT (general contact) och *CONTACT PAIR
//! med straffmetod, nod mot yta, och Coulombfriktion.
//!
//! Normalkraften för en slavnod med genomträngningen p (mot masterytans
//! utåtriktade normal) är, i general contact,
//!
//! ```text
//! F(i) = m/Δt(i)² · (κ p(i) + γ (p(i) − p(i−1))),   κ = 0,5436, γ = 0,0422
//! ```
//!
//! där m är den minsta av slavnodens och masterytans nodmassor och Δt aktuellt
//! inkrement: en massbaserad straffstyvhet (oberoende av materialets styvhet)
//! med en liten viskös del. Formen är kontrollerad för en kub som
//! faller på ett block: kraften i varje inkrement (reaktion i det fasta
//! blocket och rörelsemängden i ett fritt block) är linjär i p(i) och
//! p(i) − p(i−1), skalar med densiteten och 1/Δt² (Δt = 5·10⁻⁸ och 10⁻⁷) och
//! är densamma när masterkroppen är fast, fri eller tre gånger tyngre.
//! Kraften verkar på slavnoden längs normalen och på masterytans noder med
//! formfunktionerna i projektionspunkten.
//!
//! Den viskösa delen motsvarar standarddämpningen för straffkontakt
//! (*CONTACT DAMPING, DEFINITION=CRITICAL DAMPING FRACTION, B = 0,03:
//! f = B √(4 m k) ṗ med k = κ m/Δt² ger γ = 2B√κ = 0,044 ≈ 0,0422). Med
//! *CONTACT DAMPING (CRITICAL DAMPING FRACTION) under *SURFACE INTERACTION
//! skalas γ med B/0,03.
//!
//! - *CONTACT, *CONTACT INCLUSIONS (ALL EXTERIOR eller ytpar),
//!   *CONTACT PROPERTY ASSIGNMENT: balanserad master–slav (varje yta är slav
//!   mot den andra med vikten ½). ALL EXTERIOR
//!   ger de yttre elementytorna för varje sammanhängande kropp: solider,
//!   plana element, skal och membran. Varje kropp har också självkontakt
//!   (noder mot kroppens egna ytor som inte delar noden; ytor djupare än
//!   halva masterelementet och ytor som inte vetter mot slavnodens yta räknas
//!   inte). En yta i *CONTACT INCLUSIONS utan motyta ger självkontakt för ytan.
//! - Skal och membran i general contact är dubbelsidiga med kontakttjockleken
//!   (halva skaltjockleken på var sida om mittytan, som standard
//!   `*SURFACE PROPERTY ASSIGNMENT, PROPERTY=THICKNESS` ORIGINAL); ett
//!   tal på dataraden ger en nominell tjocklek och tredje fältet en
//!   skalfaktor. Tjockleken begränsas till 45 % av den kortaste kanten (`THICKNESS_FRACTION`,
//!   avläst: S4R 1 cm med t = 6 mm får kontakt först efter 2,2–2,3 mm i
//!   stället för 3 mm, t = 2 mm hela tjockleken). Sidan väljs efter var noden befann sig vid kontaktens
//!   början. Kontakttjockleken gäller både när skalet är master och när
//!   skalets noder är slavar.
//! - *CONTACT PAIR: ren master–slav (första ytan slav) med samma straffkraft.
//!   OBS: kinematisk kontakt (standard för kontaktpar) och den elementbaserade
//!   straffstyvheten med MECHANICAL CONSTRAINT=PENALTY ersätts båda med den
//!   massbaserade straffkraften.
//! - Friktion (*FRICTION μ): tangentiell straffjäder med samma styvhet
//!   m κ/Δt² som normalkraften och glidning när |F_t| > μ F (radiell återföring).
//!
//! Kanter mot kanter (kant-mot-kant-kontakt i general contact, t.ex.
//! korsande skalkanter) stöds inte: bara noder mot ytor.
//!
//! Kontaktkrafterna ingår i reaktionerna; friktionsarbetet ger ALLFD och
//! normalkrafternas arbete ALLPW.

use super::mass::LumpedMass;
use crate::elements::common::{cross, dot, norm, sub};
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::{FeModel, RawKeyword, SectionKind, StepDef};
use crate::fem::output::FrameData;
use crate::interactions::geom::{self, Facet};
use crate::model::{fail, Result};
use std::collections::BTreeMap;

/// Straffstyvhetens och dämpningens koefficienter (se modulens dokumentation).
pub const KAPPA: f64 = 0.5436;
pub const GAMMA: f64 = 0.0422;
/// Största kontakttjocklek för skal relativt facettens kortaste kant.
pub const THICKNESS_FRACTION: f64 = 0.45;
/// Standarddämpning B för straffkontakt (motsvarar `GAMMA`).
pub const DEFAULT_DAMPING: f64 = 0.03;

/// Egenskaper ur *SURFACE INTERACTION.
#[derive(Clone, Copy)]
struct Props {
    mu: f64,
    /// Den viskösa koefficienten γ.
    gamma: f64,
    /// Område acoustics2: *CONTACT DAMPING, DEFINITION=DAMPING COEFFICIENT μ₀
    /// (tryck per hastighet; kraft per hastighet för nodytor) i stället för γ.
    coeff: Option<f64>,
    /// Område acoustics2: TANGENT FRACTION (dämpning av den elastiska glidhastigheten).
    tangent: Option<f64>,
}

impl Default for Props {
    fn default() -> Self {
        Props { mu: 0.0, gamma: GAMMA, coeff: None, tangent: None }
    }
}

struct Pair {
    slaves: Vec<usize>,
    /// Slavytans facetter (för slavnodernas normaler; tom för nodytor).
    slave_faces: Vec<Facet>,
    master: Vec<Facet>,
    /// Kontakttjocklekens halva per masterfacett (0 för solider) och om
    /// facetten är dubbelsidig (skal, membran).
    master_off: Vec<f64>,
    master_shell: Vec<bool>,
    /// Halva kontakttjockleken i slavnoden och om noden hör till ett skal.
    slave_off: Vec<f64>,
    slave_shell: Vec<bool>,
    weight: f64,
    props: Props,
    /// Föregående genomträngning, tangentialkraft och sida (skal) per slavnod.
    prev: Vec<f64>,
    ft: Vec<[f64; 3]>,
    side: Vec<f64>,
    /// Område acoustics2: slavnodernas area (1 för nodytor) för DAMPING COEFFICIENT.
    slave_area: Vec<f64>,
}

pub struct ExplicitContact {
    pairs: Vec<Pair>,
    /// Kontaktkraft per nod efter senaste utvärdering (för utdata).
    last: Vec<NodeVec>,
}

/// Alla steg är *DYNAMIC, EXPLICIT: den implicita kontakten
/// (`interactions/contact.rs`) byggs då inte (självkontakt, *CONTACT DAMPING
/// och skal i general contact finns bara här).
pub fn only_explicit(m: &FeModel) -> bool {
    !m.steps.is_empty() && m.steps.iter().all(|s| s.procedure.as_ref().is_some_and(|p| p.name == "DYNAMIC" && p.has("EXPLICIT")))
}

fn is_shell(f: Family) -> bool {
    matches!(f, Family::Shell | Family::Membrane)
}

/// Yttre elementytor per sammanhängande kropp (solider, plana element, skal och membran).
fn exterior(m: &FeModel) -> Vec<Vec<(usize, usize)>> {
    let mut count: BTreeMap<Vec<usize>, Vec<(usize, usize)>> = BTreeMap::new();
    for (e, el) in m.elements.iter().enumerate() {
        let fam = el.etype.family();
        if !matches!(fam, Family::Continuum2D | Family::Continuum3D) && !is_shell(fam) {
            continue;
        }
        for (fi, f) in el.etype.faces().iter().enumerate() {
            let mut key: Vec<usize> = f.iter().map(|&k| el.nodes[k]).collect();
            key.sort_unstable();
            count.entry(key).or_default().push((e, fi));
        }
    }
    let mut parent: Vec<usize> = (0..m.nodes.len()).collect();
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut j = i;
        while p[j] != r {
            let nx = p[j];
            p[j] = r;
            j = nx;
        }
        r
    }
    for el in &m.elements {
        for w in el.nodes.windows(2) {
            let (a, b) = (find(&mut parent, w[0]), find(&mut parent, w[1]));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let mut bodies: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (_, v) in count {
        if v.len() == 1 {
            let (e, fi) = v[0];
            let b = find(&mut parent, m.elements[e].nodes[0]);
            bodies.entry(b).or_default().push((e, fi));
        }
    }
    bodies.into_values().collect()
}

/// Facetter för elementytor (element, ytindex).
fn facets_of(m: &FeModel, faces: &[(usize, usize)]) -> Vec<Facet> {
    faces
        .iter()
        .map(|&(e, fi)| {
            let el = &m.elements[e];
            let nodes: Vec<usize> = el.etype.faces()[fi].iter().map(|&k| el.nodes[k]).collect();
            let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            let thick = match (el.etype.dim(), el.section) {
                (2, Some(si)) => m.sections[si].solid_thickness(),
                _ => 1.0,
            };
            let mut inside = crate::elements::common::centroid(&xs);
            if is_shell(el.etype.family()) && nodes.len() >= 3 {
                // skal: punkten "inuti" på mittytans negativa sida (normalen = SPOS)
                let xf: Vec<[f64; 3]> = nodes.iter().map(|&n| m.nodes[n]).collect();
                let nn = cross(sub(xf[1], xf[0]), sub(xf[xf.len() - 1], xf[0]));
                let l = norm(nn).max(1e-300);
                let size = l.sqrt();
                inside = [0, 1, 2].map(|k| inside[k] - nn[k] / l * size);
            }
            Facet { elem: e, nodes, inside, dim: el.etype.dim(), thick }
        })
        .collect()
}

fn nodes_of_facets(fs: &[Facet]) -> Vec<usize> {
    let mut v: Vec<usize> = fs.iter().flat_map(|f| f.nodes.iter().copied()).collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Egenskaperna för *SURFACE INTERACTION `name`: friktion och kontaktdämpning.
fn properties(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<Props> {
    let up = name.trim().to_uppercase();
    let Some(pos) = m.extra.iter().position(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*SURFACE INTERACTION, NAME={name} finns inte"));
    };
    let mut pr = Props::default();
    for sub in m.extra.iter().skip(pos + 1).take_while(|k| matches!(k.name.as_str(), "SURFACE BEHAVIOR" | "FRICTION" | "CONTACT DAMPING")) {
        match sub.name.as_str() {
            "FRICTION" => {
                if sub.has("ROUGH") || sub.data.len() > 1 || sub.has("ANISOTROPIC") || sub.has("DEPENDENCIES") {
                    return sub.err("bara isotrop Coulombfriktion med en konstant μ stöds i explicit dynamik");
                }
                pr.mu = sub.first_floats()?.first().copied().unwrap_or(0.0);
            }
            "SURFACE BEHAVIOR" => {
                let po = sub.param_upper("PRESSURE-OVERCLOSURE").unwrap_or_else(|| "HARD".into());
                if po != "HARD" || sub.has("NO SEPARATION") {
                    return sub.err("bara hård kontakt (PRESSURE-OVERCLOSURE=HARD) stöds i explicit dynamik");
                }
            }
            _ => {
                // område acoustics2: DAMPING COEFFICIENT och TANGENT FRACTION
                let d = crate::interactions::contact::damping::parse(sub, true)?;
                if d.critical {
                    pr.gamma = GAMMA * d.mu0 / DEFAULT_DAMPING;
                } else {
                    pr.gamma = 0.0;
                    pr.coeff = Some(d.mu0);
                }
                pr.tangent = d.tangent;
            }
        }
    }
    Ok(pr)
}

/// Underordnade nyckelord direkt efter `kw` i modellens lista.
fn sub_options<'a>(m: &'a FeModel, kw: &RawKeyword, names: &[&str]) -> Vec<&'a RawKeyword> {
    let Some(i) = m.extra.iter().position(|k| std::ptr::eq(k, kw)) else { return Vec::new() };
    m.extra.iter().skip(i + 1).take_while(|k| names.contains(&k.name.as_str())).collect()
}

/// Skalets (membranets) tjocklek ur tvärsnittet.
fn shell_thickness(m: &FeModel, e: usize) -> f64 {
    let Some(si) = m.elements[e].section else { return 0.0 };
    let s = &m.sections[si];
    match &s.kind {
        SectionKind::Shell { thickness, .. } => *thickness,
        _ => s.raw.first_floats().ok().and_then(|v| v.first().copied()).filter(|t| *t > 0.0).unwrap_or(0.0),
    }
}

/// Kontakttjocklek per element ur *SURFACE PROPERTY ASSIGNMENT, PROPERTY=THICKNESS
/// (`None` = skalets tjocklek): (nominell tjocklek eller `None` = ORIGINAL, skalfaktor).
type Thickness = Vec<(Option<f64>, f64)>;

fn thickness_rules(m: &FeModel, subs: &[&RawKeyword]) -> Result<Thickness> {
    let mut t: Thickness = vec![(None, 1.0); m.elements.len()];
    for s in subs.iter().filter(|s| s.name == "SURFACE PROPERTY ASSIGNMENT") {
        if s.param_upper("PROPERTY").as_deref() != Some("THICKNESS") {
            return s.err("*SURFACE PROPERTY ASSIGNMENT: bara PROPERTY=THICKNESS stöds i explicit dynamik");
        }
        for d in &s.data {
            let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
            let what = f.get(1).copied().unwrap_or("ORIGINAL").to_uppercase();
            let nominal = match what.as_str() {
                "" | "ORIGINAL" => None,
                "CURRENT" | "THINNING" => return s.err_at(d.line, format!("{what} stöds inte (ORIGINAL eller ett tal)")),
                v => Some(crate::fem::model::parse_f64(v).ok_or_else(|| crate::model::FemError(format!("{}: ogiltig tjocklek '{v}'", s.location())))?),
            };
            let scale = match f.get(2).filter(|x| !x.is_empty()) {
                Some(v) => crate::fem::model::parse_f64(v).ok_or_else(|| crate::model::FemError(format!("{}: ogiltig skalfaktor '{v}'", s.location())))?,
                None => 1.0,
            };
            let elems: Vec<usize> = match f.first().filter(|x| !x.is_empty()) {
                None => (0..m.elements.len()).collect(),
                Some(name) => {
                    let mut v: Vec<usize> = geom::surface(m, s, name)?.faces.iter().map(|p| p.0).collect();
                    v.dedup();
                    v
                }
            };
            for e in elems {
                t[e] = (nominal, scale);
            }
        }
    }
    Ok(t)
}

impl Pair {
    fn new(m: &FeModel, slave_faces: Vec<Facet>, master: Vec<Facet>, weight: f64, props: Props, thick: &Thickness) -> Pair {
        let off_of = |e: usize| -> f64 {
            if !is_shell(m.elements[e].etype.family()) {
                return 0.0;
            }
            let (nom, sc) = thick[e];
            // kontakttjockleken begränsas till 45 % av den kortaste kanten
            let el = &m.elements[e];
            let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
            let edge = (0..xs.len()).map(|i| norm(sub(xs[(i + 1) % xs.len()], xs[i]))).fold(f64::INFINITY, f64::min);
            0.5 * (nom.unwrap_or_else(|| shell_thickness(m, e)) * sc).min(THICKNESS_FRACTION * edge)
        };
        let master_off = master.iter().map(|f| off_of(f.elem)).collect();
        let master_shell = master.iter().map(|f| is_shell(m.elements[f.elem].etype.family())).collect();
        let slaves = nodes_of_facets(&slave_faces);
        let mut so: BTreeMap<usize, (f64, bool)> = BTreeMap::new();
        for f in &slave_faces {
            let o = off_of(f.elem);
            let sh = is_shell(m.elements[f.elem].etype.family());
            for &n in &f.nodes {
                let e = so.entry(n).or_insert((0.0, false));
                e.0 = e.0.max(o);
                e.1 |= sh;
            }
        }
        let slave_off = slaves.iter().map(|n| so.get(n).map_or(0.0, |v| v.0)).collect();
        let slave_shell = slaves.iter().map(|n| so.get(n).is_some_and(|v| v.1)).collect();
        let n = slaves.len();
        let mut area: BTreeMap<usize, f64> = BTreeMap::new();
        for f in &slave_faces {
            let a = facet_area(m, f) / f.nodes.len().max(1) as f64;
            for &nd in &f.nodes {
                *area.entry(nd).or_insert(0.0) += a;
            }
        }
        let slave_area = slaves.iter().map(|nd| area.get(nd).copied().unwrap_or(1.0)).collect();
        Pair { slaves, slave_faces, master, master_off, master_shell, slave_off, slave_shell, weight, props, prev: vec![0.0; n], ft: vec![[0.0; 3]; n], side: vec![0.0; n], slave_area }
    }

    /// Nodyta som slav (ingen tjocklek, inga normaler).
    fn nodes(m: &FeModel, slaves: Vec<usize>, master: Vec<Facet>, props: Props, thick: &Thickness) -> Pair {
        let mut p = Pair::new(m, Vec::new(), master, 1.0, props, thick);
        let n = slaves.len();
        p.slaves = slaves;
        p.slave_off = vec![0.0; n];
        p.slave_shell = vec![false; n];
        p.prev = vec![0.0; n];
        p.ft = vec![[0.0; 3]; n];
        p.side = vec![0.0; n];
        p.slave_area = vec![1.0; n];
        p
    }
}

impl ExplicitContact {
    /// Bygger kontakten från modellens *CONTACT och *CONTACT PAIR. `None` om kontakt saknas.
    pub fn build(p: &Prepared, _step: &StepDef) -> Result<Option<Self>> {
        let m = p.model;
        let mut pairs = Vec::new();
        let none: Thickness = vec![(None, 1.0); m.elements.len()];
        for kw in &m.extra {
            match kw.name.as_str() {
                "CONTACT" => {
                    let subs = sub_options(m, kw, &["CONTACT INCLUSIONS", "CONTACT PROPERTY ASSIGNMENT", "CONTACT EXCLUSIONS", "CONTACT FORMULATION", "CONTACT CONTROLS ASSIGNMENT", "CONTACT INITIALIZATION ASSIGNMENT", "SURFACE PROPERTY ASSIGNMENT"]);
                    let mut surf_pairs: Vec<(String, String)> = Vec::new();
                    let mut selfs: Vec<String> = Vec::new();
                    let mut all = false;
                    let mut props = Props::default();
                    // område explicit_adv: fel i egenskaperna gäller bara par utan partiklar
                    let mut prop_err: Option<crate::model::FemError> = None;
                    let mut adv_pairs = 0usize;
                    for s in &subs {
                        match s.name.as_str() {
                            "CONTACT INCLUSIONS" => {
                                all |= s.has("ALL EXTERIOR") || s.data.iter().all(|d| d.fields.iter().all(|f| f.trim().is_empty()));
                                for d in &s.data {
                                    let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                                    if f.len() >= 2 && !f[0].is_empty() && !f[1].is_empty() {
                                        surf_pairs.push((f[0].to_string(), f[1].to_string()));
                                    } else if let Some(x) = f.first().filter(|x| !x.is_empty()) {
                                        selfs.push(x.to_string());
                                    }
                                }
                            }
                            "CONTACT PROPERTY ASSIGNMENT" => {
                                for d in &s.data {
                                    let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                                    if f.len() >= 3 && !f[2].is_empty() {
                                        // område explicit_adv: partikelpar läses av explicit_adv_dem.rs
                                        if !f[0].is_empty() && super::adv::dem::is_particle_pair(m, f[0], f[1]) {
                                            continue;
                                        }
                                        if !f[0].is_empty() || !f[1].is_empty() {
                                            return s.err_at(d.line, "egenskaper per ytpar stöds inte (skriv ' , , INTERAKTION')");
                                        }
                                        // område explicit_adv: egenskaper som bara partikelkontakten stöder
                                        props = match properties(m, s, f[2]) {
                                            Ok(pr) => pr,
                                            Err(e) => {
                                                prop_err = Some(e);
                                                Props::default()
                                            }
                                        };
                                    }
                                }
                            }
                            "CONTACT CONTROLS ASSIGNMENT" | "CONTACT FORMULATION" | "SURFACE PROPERTY ASSIGNMENT" => {}
                            _ => return s.err("stöds inte i explicit dynamik"),
                        }
                    }
                    let thick = thickness_rules(m, &subs)?;
                    let n0 = pairs.len();
                    if all {
                        let bodies: Vec<Vec<Facet>> = exterior(m).iter().map(|b| facets_of(m, b)).collect();
                        for a in 0..bodies.len() {
                            for b in a + 1..bodies.len() {
                                pairs.push(Pair::new(m, bodies[a].clone(), bodies[b].clone(), 0.5, props, &thick));
                                pairs.push(Pair::new(m, bodies[b].clone(), bodies[a].clone(), 0.5, props, &thick));
                            }
                            // självkontakt (en kropps kontakt med sig själv)
                            pairs.push(Pair::new(m, bodies[a].clone(), bodies[a].clone(), 0.5, props, &thick));
                        }
                    }
                    // område explicit_adv: par med partikelytor (PD3D, PC3D) sköts av explicit_adv_dem.rs
                    let n_all = surf_pairs.len() + selfs.len();
                    surf_pairs.retain(|(a, b)| !super::adv::dem::is_particle_pair(m, a, b));
                    selfs.retain(|a| !super::adv::dem::is_particle_pair(m, a, a));
                    adv_pairs += n_all - surf_pairs.len() - selfs.len();
                    if let Some(e) = prop_err.take() {
                        if all || !surf_pairs.is_empty() || !selfs.is_empty() {
                            return Err(e);
                        }
                    }
                    for (a, b) in surf_pairs {
                        let fa = geom::facets(m, geom::surface(m, kw, &a)?, &a)?;
                        let fb = geom::facets(m, geom::surface(m, kw, &b)?, &b)?;
                        if a.trim().eq_ignore_ascii_case(b.trim()) {
                            pairs.push(Pair::new(m, fa.clone(), fb, 0.5, props, &thick));
                            continue;
                        }
                        pairs.push(Pair::new(m, fa.clone(), fb.clone(), 0.5, props, &thick));
                        pairs.push(Pair::new(m, fb, fa, 0.5, props, &thick));
                    }
                    for a in selfs {
                        let fa = geom::facets(m, geom::surface(m, kw, &a)?, &a)?;
                        pairs.push(Pair::new(m, fa.clone(), fa, 0.5, props, &thick));
                    }
                    if pairs.len() == n0 && adv_pairs == 0 {
                        return fail(format!("{}: *CONTACT utan kontaktytor (*CONTACT INCLUSIONS)", kw.location()));
                    }
                }
                "CONTACT PAIR" => {
                    let ip = crate::inp::required(kw, "INTERACTION")?;
                    let props = properties(m, kw, ip)?;
                    if let Some(mc) = kw.param_upper("MECHANICAL CONSTRAINT") {
                        if mc != "PENALTY" && mc != "KINEMATIC" {
                            return kw.err(format!("MECHANICAL CONSTRAINT={mc} stöds inte (PENALTY, KINEMATIC)"));
                        }
                    }
                    for d in kw.records() {
                        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                        if f.is_empty() {
                            continue;
                        }
                        if f.len() < 2 {
                            return kw.err_at(d.line, "skriv slavyta, masteryta");
                        }
                        let sl = geom::surface(m, kw, f[0])?;
                        let master = geom::facets(m, geom::surface(m, kw, f[1])?, f[1])?;
                        // kontaktpar: ingen kontakttjocklek (som utan *SURFACE PROPERTY)
                        if sl.kind == "NODE" {
                            let slaves = geom::surface_nodes(m, sl)?;
                            pairs.push(Pair::nodes(m, slaves, master, props, &none));
                        } else {
                            let mut pr = Pair::new(m, geom::facets(m, sl, f[0])?, master, 1.0, props, &none);
                            pr.master_off.iter_mut().for_each(|o| *o = 0.0);
                            pr.master_shell.iter_mut().for_each(|s| *s = false);
                            pr.slave_off.iter_mut().for_each(|o| *o = 0.0);
                            pr.slave_shell.iter_mut().for_each(|s| *s = false);
                            pairs.push(pr);
                        }
                    }
                }
                _ => {}
            }
        }
        if pairs.is_empty() {
            return Ok(None);
        }
        Ok(Some(ExplicitContact { pairs, last: vec![[0.0; crate::fem::dof::SLOTS]; m.nodes.len()] }))
    }

    /// Kontaktkrafter på noderna (läggs i `f`) för förskjutningen `u` efter ett
    /// inkrement `dt` med förskjutningsändringen `du`. Returnerar
    /// friktionsdissipationen i inkrementet.
    #[allow(clippy::too_many_arguments)]
    pub fn forces(&mut self, m: &FeModel, u: &[NodeVec], du: &[NodeVec], dt: f64, mass: &LumpedMass, f: &mut [NodeVec], commit: bool) -> Result<f64> {
        let _ = commit;
        if dt <= 0.0 {
            return Ok(0.0);
        }
        let x = |n: usize| -> [f64; 3] { std::array::from_fn(|c| m.nodes[n][c] + u[n][c]) };
        let mut fd = 0.0;
        for pr in &mut self.pairs {
            // facetternas aktuella koordinater och omslutande lådor
            let fx: Vec<Vec<[f64; 3]>> = pr.master.iter().map(|fc| fc.nodes.iter().map(|&n| x(n)).collect()).collect();
            let boxes: Vec<([f64; 3], [f64; 3], f64)> = fx
                .iter()
                .zip(&pr.master_off)
                .map(|(xs, &off)| {
                    let mut lo = xs[0];
                    let mut hi = xs[0];
                    for p in xs {
                        for c in 0..3 {
                            lo[c] = lo[c].min(p[c] - off);
                            hi[c] = hi[c].max(p[c] + off);
                        }
                    }
                    let size = (0..3).map(|c| hi[c] - lo[c]).fold(0.0, f64::max);
                    (lo, hi, size)
                })
                .collect();
            // halva masterelementets djup längs normalen (avstånd yta–elementets
            // tyngdpunkt); för skal hela kontakttjockleken
            let depth: Vec<f64> = pr
                .master
                .iter()
                .zip(&fx)
                .enumerate()
                .map(|(i, (fc, xs))| {
                    if pr.master_shell[i] {
                        return 2.0 * pr.master_off[i];
                    }
                    let el = &m.elements[fc.elem];
                    let xe: Vec<[f64; 3]> = el.nodes.iter().map(|&n| x(n)).collect();
                    let ce = crate::elements::common::centroid(&xe);
                    let cf = crate::elements::common::centroid(xs);
                    norm(sub(cf, ce))
                })
                .collect();
            // slavnodernas utåtriktade normaler (medel av slavytans facetter)
            let mut sn: BTreeMap<usize, [f64; 3]> = BTreeMap::new();
            for fc in &pr.slave_faces {
                if is_shell(m.elements[fc.elem].etype.family()) {
                    continue;
                }
                let xs: Vec<[f64; 3]> = fc.nodes.iter().map(|&n| x(n)).collect();
                let c = if fc.dim == 2 || fc.nodes.len() == 3 { [1.0 / 3.0, 1.0 / 3.0] } else { [0.0, 0.0] };
                let c = if fc.dim == 2 { [0.0, 0.0] } else { c };
                let nf = geom::eval(fc, &xs, c).normal;
                for &n in &fc.nodes {
                    let e = sn.entry(n).or_insert([0.0; 3]);
                    for q in 0..3 {
                        e[q] += nf[q];
                    }
                }
            }
            for (k, &s) in pr.slaves.iter().enumerate() {
                let xs = x(s);
                // skalnoder (dubbelsidiga) har ingen bestämd normal
                let ns = if pr.slave_shell[k] {
                    None
                } else {
                    sn.get(&s).map(|v| {
                        let l = norm(*v).max(1e-300);
                        [v[0] / l, v[1] / l, v[2] / l]
                    })
                };
                let soff = pr.slave_off[k];
                let mut best: Option<(usize, [f64; 2], f64, [f64; 3], f64)> = None;
                for (fi, fc) in pr.master.iter().enumerate() {
                    if fc.nodes.contains(&s) {
                        continue;
                    }
                    let (lo, hi, size) = boxes[fi];
                    let tol = 0.5 * size + soff;
                    if (0..3).any(|c| xs[c] < lo[c] - tol || xs[c] > hi[c] + tol) {
                        continue;
                    }
                    let (xi, _, inside) = geom::project(fc, &fx[fi], xs);
                    let fp = geom::eval(fc, &fx[fi], xi);
                    if !inside {
                        // noder precis på facettens kant eller hörn (t.ex. symmetriska
                        // kroppar som möts nod mot nod) räknas som innanför
                        let r = sub(xs, fp.x);
                        let dn = dot(r, fp.normal);
                        let t = norm([r[0] - dn * fp.normal[0], r[1] - dn * fp.normal[1], r[2] - dn * fp.normal[2]]);
                        if t > 1e-3 * size {
                            continue;
                        }
                    }
                    let off = pr.master_off[fi] + soff;
                    // avstånd till mittytan längs normalen; skal: sidan från kontaktens början
                    let d = dot(sub(xs, fp.x), fp.normal);
                    let (nrm, side, pen) = if pr.master_shell[fi] {
                        let sd = if pr.side[k] != 0.0 { pr.side[k] } else if d >= 0.0 { 1.0 } else { -1.0 };
                        (fp.normal.map(|v| sd * v), sd, off - sd * d)
                    } else {
                        (fp.normal, 0.0, off - d)
                    };
                    // bara ytor som vetter mot slavnodens egen yta (motriktade normaler)
                    if ns.is_some_and(|v| dot(v, nrm) > -0.3) {
                        continue;
                    }
                    // noder i ytans plan (t.ex. hörn på en sidoyta) räknas inte, och
                    // inte heller noder djupare än halva masterelementet (bakifrån)
                    if pen <= 1e-9 * size || pen > depth[fi] + soff {
                        continue;
                    }
                    if best.as_ref().is_none_or(|b| pen < b.2) {
                        best = Some((fi, xi, pen, nrm, side));
                    }
                }
                let Some((fi, xi, pen, nrm, side)) = best else {
                    pr.prev[k] = 0.0;
                    pr.ft[k] = [0.0; 3];
                    pr.side[k] = 0.0;
                    continue;
                };
                pr.side[k] = side;
                let fc = &pr.master[fi];
                let fp = geom::eval(fc, &fx[fi], xi);
                let mm = fc.nodes.iter().map(|&n| mass.diag[n][0]).fold(f64::INFINITY, f64::min);
                let meff = mass.diag[s][0].min(mm);
                let kpen = meff / (dt * dt);
                // område acoustics2: DAMPING COEFFICIENT ger den viskösa kraften μ₀ A ṗ
                let visc = match pr.props.coeff {
                    Some(c) => c * pr.slave_area[k] * (pen - pr.prev[k]) / dt,
                    None => kpen * pr.props.gamma * (pen - pr.prev[k]),
                };
                // viskös koefficient (kraft per hastighet) för TANGENT FRACTION
                let cvisc = match pr.props.coeff {
                    Some(c) => c * pr.slave_area[k],
                    None => kpen * pr.props.gamma * dt,
                };
                let fnor = (kpen * KAPPA * pen + visc).max(0.0) * pr.weight;
                pr.prev[k] = pen;
                if crate::settings::with(|s| s.contact_debug) {
                    eprintln!("kontakt: slav {s} yta {fi} pen {pen:.4e} F {fnor:.4e} meff {meff:.4e} dt {dt:.3e}");
                }
                let mut fv = [nrm[0] * fnor, nrm[1] * fnor, nrm[2] * fnor];
                // friktion: tangentiell straffjäder på den relativa glidningen
                if pr.props.mu > 0.0 {
                    let mut ds = [0.0; 3];
                    for c in 0..3 {
                        ds[c] = du[s][c] - fc.nodes.iter().zip(&fp.n).map(|(&n, w)| w * du[n][c]).sum::<f64>();
                    }
                    let dn = dot(ds, nrm);
                    for c in 0..3 {
                        ds[c] -= dn * nrm[c];
                    }
                    let old = pr.ft[k];
                    let on = dot(old, nrm);
                    let kt = KAPPA * kpen * pr.weight;
                    let mut ft: [f64; 3] = std::array::from_fn(|c| old[c] - on * nrm[c] - kt * ds[c]);
                    let lim = pr.props.mu * fnor;
                    let mag = norm(ft);
                    if mag > lim {
                        let sc = if mag > 0.0 { lim / mag } else { 0.0 };
                        for v in ft.iter_mut() {
                            *v *= sc;
                        }
                    }
                    // område acoustics2: TANGENT FRACTION – dämpning av den elastiska
                    // glidhastigheten (ändringen av den elastiska glidningen −Δf_t/k_t)
                    let mut fdamp = [0.0; 3];
                    if let Some(tf) = pr.props.tangent {
                        for c in 0..3 {
                            fdamp[c] = tf * cvisc * pr.weight * (ft[c] - (old[c] - on * nrm[c])) / (kt * dt);
                        }
                    }
                    // dissipation = arbete på glidningen minus ändrad elastisk energi
                    let w = -dot(ft, ds) - dot(fdamp, ds);
                    let de = (dot(ft, ft) - dot(old, old)) / (2.0 * kt);
                    fd += w - de;
                    pr.ft[k] = ft;
                    for c in 0..3 {
                        fv[c] += ft[c] + fdamp[c];
                    }
                }
                for c in 0..3 {
                    f[s][c] += fv[c];
                    for (&n, w) in fc.nodes.iter().zip(&fp.n) {
                        f[n][c] -= w * fv[c];
                    }
                }
            }
        }
        self.last = f.to_vec();
        Ok(fd)
    }

    /// Kontaktkraften som nodfält CFORCE (summerat över alla
    /// kontaktpar; används för egen kontroll).
    pub fn output(&self, _m: &FeModel, _fr: &mut FrameData) {}

    /// (ALLFD, ALLPW) i inkrementet: friktionsdissipationen och normalkrafternas
    /// arbete (kontaktkrafternas arbete plus friktionsdissipationen).
    pub fn energy_increment(&self, f0: &[NodeVec], f1: &[NodeVec], du: &[NodeVec], fric: f64) -> (f64, f64) {
        let w = super::energy::work(f0, f1, du);
        (fric, w + fric)
    }
}

/// Område acoustics2: facettens area (plana element: längd gånger tjocklek).
fn facet_area(m: &FeModel, f: &Facet) -> f64 {
    let x: Vec<[f64; 3]> = f.nodes.iter().map(|&n| m.nodes[n]).collect();
    if f.dim == 2 || x.len() == 2 || (x.len() == 3 && f.dim == 2) {
        return norm(sub(x[1], x[0])) * f.thick;
    }
    // hörnen: 3 för trianglar (3 eller 6 noder), annars 4
    let nc = if x.len() == 3 || x.len() == 6 { 3 } else { 4 };
    let mut a = [0.0; 3];
    for i in 1..nc - 1 {
        let u = sub(x[i], x[0]);
        let v = sub(x[i + 1], x[0]);
        a[0] += u[1] * v[2] - u[2] * v[1];
        a[1] += u[2] * v[0] - u[0] * v[2];
        a[2] += u[0] * v[1] - u[1] * v[0];
    }
    0.5 * norm(a)
}
