//! Kontakt: *CONTACT PAIR och *CONTACT (general contact) med
//! *SURFACE INTERACTION, *SURFACE BEHAVIOR och *FRICTION.
//!
//! Diskretisering (per slavnod A):
//! - SURFACE TO SURFACE (fyrhörningar och tvånodskanter): segmentviktning.
//!   Slavnoden får den del av varje ytelement som ligger närmast den i
//!   parameterrummet (halva kanten, en fjärdedel av fyrhörningen) med konstant
//!   vikt χ_A; delen integreras med fasta punkter (`tributary_points`). Gapet
//!   h̄_A = ∫ χ_A g dA / ∫ χ_A dA, trycket p̄_A = p(h̄_A) och kraften
//!   A_A p̄_A ∂h̄_A/∂u. Normalen är slavnormalen i varje punkt och masterpunkten
//!   ligger längs den (strålprojektion). SMALL SLIDING: masterytans
//!   formfunktioner i punktens projektion i referensläget. Finite sliding:
//!   punktens cell (intervall/cirkelskiva) fördelas över masterelementen
//!   (`cell_weights`), vilket också ger viktningen vid masterytans fria kanter.
//!   CPRESS skrivs ut utjämnad, CPRESS_A = ∫ χ_A Σ_B N_B p̄_B dA / A_A.
//!   General contact: ren master–slav per ytpar, finast indelade ytan som slav.
//! - SURFACE TO SURFACE (trianglar, kvadratiska ytor): mortarvikter med
//!   elementvis dual bas Φ på slavytan (samma som *TIE): gapet
//!   g_A = n_A · (x_A − Σ_C T_AC x_C) med T_AC = ∫ Φ_A M_C dA / ∫ Φ_A dA.
//! - NODE TO SURFACE (standard för *CONTACT PAIR i implicit analys):
//!   slavnodens projektion på närmaste masteryta, T_AC = M_C(ξ), ytan =
//!   slavnodens andel av slavytan. Med finite sliding i 2D utjämnas masterytan
//!   nära noderna (SMOOTH=, standard 0,2).
//!
//! Normalkontakt (h = −g, överlappning):
//! - HARD: straff p = k h för h ≥ 0 med en styv straffkoefficient
//!   k = `HARD_FACTOR` · (representativ elementstyvhet)/A, så att
//!   genomträngningen blir försumblig (i praktiken samma lösning som med
//!   Lagrange-multiplikatorer). Yta mot yta med finite sliding och
//!   general contact: straffmetod med standardstyvheten 10 · M/h för
//!   den mjukaste sidan, med de ytelement som möts vid slavnoden.
//! - LINEAR: p = k h med lutningen k från *SURFACE BEHAVIOR.
//! - EXPONENTIAL: p = p0/(e − 1) · (h/c0 + 1) (exp(h/c0 + 1) − 1) för h ≥ −c0.
//!   OBS: inte validerad.
//!
//! Friktion (Coulomb, *FRICTION μ) med straffreglerad fastläge ("elastisk
//! glidning"): kritisk elastisk glidning γ = F_f · l_c med
//! F_f = SLIP TOLERANCE (0,005) och l_c = slavytornas medelstorlek. Fastläge:
//! τ = μ p e/γ; glidning när |e| > γ: τ = μ p ê (radiell återföring, Simo &
//! Hughes kap. 2; Wriggers, Computational Contact Mechanics kap. 5).
//! Tangenten är symmetrisk (kopplingen dτ/dp utelämnas).
//!
//! Potentialen Π = Σ_A A_A (∫ p dh + ½ k_t |e|²) ger inre kraft och tangent:
//! f = Σ_A A_A (−p n_A + τ_A) ⊗ c_A och K = Σ_A A_A (p' n nᵀ + K_t) ⊗ c_A c_Aᵀ,
//! där c_A är koefficienterna (1 för slavnoden, −T_AC för masternoderna).
//!
//! SMALL SLIDING: koefficienter och normaler beräknas en gång i
//! referensläget. Annars (finite sliding) projiceras slavytan på nytt i
//! aktuell konfiguration vid varje utvärdering.

#[path = "contact_ext.rs"]
mod ext;
// område acoustics2: *CONTACT DAMPING
#[path = "contact_damping.rs"]
pub mod damping;

use super::constraints::sub_options;
use super::geom::{self, Facet};
use super::{Interaction, InteractionCtx, InteractionFactory, NodeOutput, TangentSink};
use crate::elements::common::{dot, norm, scale, sub};
use crate::elements::ElemCtx;
use crate::fem::dof::{Dof, NodeVec, U1};
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::{fail, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;

pub static FACTORIES: &[InteractionFactory] = &[
    InteractionFactory { keywords: &["CONTACT"], build: general },
    InteractionFactory { keywords: &["CONTACT PAIR"], build: pair },
];

/// Kanter med större vinkel än arccos(CORNER_COS) räknas som hörn (S2S finite sliding).
const CORNER_COS: f64 = 0.5;
/// Sökzon utanför masterytans fria kanter (andel av ytelementets storlek) för
/// slavpunkter vars cell delvis ligger på masterytan (`edge_fraction`).
const EXTENSION_ZONE: f64 = 0.5;

/// Klistringsstyvhet för ROUGH relativt normalens straffkoefficient (i
/// stället för Lagrange-multiplikatorer; 1e-4 · k_hård ger fortfarande försumbar
/// elastisk glidning men konvergerar).
const LAGRANGE_STICK: f64 = 1e-4;

/// Straffkoefficient för hård kontakt relativt underliggande elements styvhet.
pub const HARD_FACTOR: f64 = 1e4;
/// Straffkoefficient för hård kontakt (direkt metod) med SMALL SLIDING.
pub const HARD_FACTOR_SMALL: f64 = 1e8;
/// Straffkoefficient för hård kontakt (direkt metod) mot en stel masteryta.
pub const HARD_FACTOR_RIGID: f64 = 1e8;

#[derive(Debug, Clone, Copy)]
enum Law {
    Hard,
    Linear(f64),
    Exponential { c0: f64, p0: f64 },
    /// PRESSURE-OVERCLOSURE=TABULAR: (överlappning, tryck), högst 32 punkter;
    /// 0 före första punkten, sista lutningen efter sista.
    Tabular([(f64, f64); 32], usize),
}

/// Hur hård kontakt upprätthålls (*SURFACE BEHAVIOR, DIRECT/PENALTY/AUGMENTED LAGRANGE).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Enforce {
    /// Exakt (Lagrange-multiplikatorer); här en mycket styv straffkoefficient.
    Direct,
    /// Linjär straffmetod med given styvhet (tryck per överlappning) eller
    /// standardstyvheten (None), multiplicerad med skalfaktorn.
    Penalty(Option<f64>, f64),
}

#[derive(Debug, Clone, Copy)]
struct Props {
    law: Law,
    /// None: formuleringens standard (DIRECT för nod mot yta, PENALTY för
    /// SURFACE TO SURFACE och general contact).
    enforce: Option<Enforce>,
    mu: f64,
    slip_tol: f64,
    /// *SURFACE BEHAVIOR, NO SEPARATION: kontakt som har slutits öppnar inte igen.
    no_sep: bool,
    /// *FRICTION, ROUGH: ingen glidning.
    rough: bool,
    /// *FRICTION, LAGRANGE: fastläget upprätthålls (styv straffkoefficient).
    lagrange: bool,
    /// *FRICTION, ELASTIC SLIP=γ (absolut kritisk elastisk glidning).
    elastic_slip: Option<f64>,
    /// *FRICTION, EXPONENTIAL DECAY: (μ_k, d_c), μ_s = `mu`; μ = μ_k + (μ_s − μ_k) e^(−d_c γ̇).
    decay: Option<(f64, f64)>,
    /// *FRICTION, ANISOTROPIC: μ₂ i lokal riktning 2 (μ₁ = `mu`), `contact_ext.rs`.
    mu2: Option<f64>,
    /// Område acoustics2: *CONTACT DAMPING (`contact_damping.rs`).
    damp: Option<damping::Damping>,
}

impl Props {
    fn base() -> Props {
        Props { law: Law::Hard, enforce: None, mu: 0.0, slip_tol: 0.005, no_sep: false, rough: false, lagrange: false, elastic_slip: None, decay: None, mu2: None, damp: None }
    }
}

impl Law {
    /// (p, dp/dh) för överlappningen h med straffkoefficienten `k_hard`.
    fn pressure(&self, h: f64, k_hard: f64) -> (f64, f64) {
        match *self {
            Law::Hard => {
                if h >= 0.0 {
                    (k_hard * h, k_hard)
                } else {
                    (0.0, 0.0)
                }
            }
            Law::Linear(k) => {
                if h >= 0.0 {
                    (k * h, k)
                } else {
                    (0.0, 0.0)
                }
            }
            Law::Tabular(t, n) => {
                let t = &t[..n];
                if h < t[0].0 {
                    return (0.0, 0.0);
                }
                for w in t.windows(2) {
                    if h <= w[1].0 {
                        let s = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
                        return (w[0].1 + s * (h - w[0].0), s);
                    }
                }
                let s = (t[n - 1].1 - t[n - 2].1) / (t[n - 1].0 - t[n - 2].0);
                (t[n - 1].1 + s * (h - t[n - 1].0), s)
            }
            Law::Exponential { c0, p0 } => {
                if h <= -c0 {
                    return (0.0, 0.0);
                }
                let e1 = std::f64::consts::E - 1.0;
                let z = h / c0 + 1.0;
                let ez = z.exp();
                (p0 / e1 * z * (ez - 1.0), p0 / e1 / c0 * ((ez - 1.0) + z * ez))
            }
        }
    }

    /// Kontakten är aktiv (bidrar med styvhet) vid överlappningen h.
    fn active(&self, h: f64) -> bool {
        match *self {
            Law::Exponential { c0, .. } => h > -c0,
            Law::Tabular(t, _) => h >= t[0].0,
            _ => h >= 0.0,
        }
    }
}

fn props(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<Props> {
    let up = name.trim().to_uppercase();
    let Some(si) = m.extra.iter().find(|k| k.name == "SURFACE INTERACTION" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*SURFACE INTERACTION, NAME={name} finns inte"));
    };
    let mut p = Props::base();
    for sub in sub_options(m, si, &["SURFACE BEHAVIOR", "FRICTION", "CONTACT DAMPING"]) {
        match sub.name.as_str() {
            "SURFACE BEHAVIOR" => {
                p.no_sep = sub.has("NO SEPARATION");
                let v = sub.first_floats()?;
                if sub.has("PENALTY") {
                    let ty = sub.param_upper("PENALTY").unwrap_or_default();
                    if !(ty.is_empty() || ty == "LINEAR") {
                        return sub.err(format!("PENALTY={ty} stöds inte ännu (LINEAR)"));
                    }
                    // dataraden: styvhet (tom = standard), (oanvänd), skalfaktor
                    let f: Vec<&str> = sub.data.first().map(|d| d.fields.iter().map(|s| s.trim()).collect()).unwrap_or_default();
                    let k = f.first().filter(|s| !s.is_empty()).map(|s| parse_f64(s).ok_or(())).transpose();
                    let sc = f.get(2).filter(|s| !s.is_empty()).map(|s| parse_f64(s).ok_or(())).transpose();
                    let (Ok(k), Ok(sc)) = (k, sc) else {
                        return sub.err("ogiltigt tal på dataraden");
                    };
                    if f.iter().enumerate().any(|(i, s)| i != 0 && i != 2 && !s.is_empty()) {
                        return sub.err("bara straffstyvhet (fält 1) och skalfaktor (fält 3) stöds på dataraden");
                    }
                    p.enforce = Some(Enforce::Penalty(k, sc.unwrap_or(1.0)));
                } else if sub.has("AUGMENTED LAGRANGE") || sub.has("DIRECT") {
                    // båda ger (inom toleranser) ingen genomträngning
                    p.enforce = Some(Enforce::Direct);
                }
                let v = if p.enforce.is_some_and(|e| matches!(e, Enforce::Penalty(..))) { Vec::new() } else { v };
                p.law = match sub.param_upper("PRESSURE-OVERCLOSURE").as_deref().unwrap_or("HARD") {
                    "HARD" => Law::Hard,
                    "LINEAR" => match v.first() {
                        Some(&k) if k > 0.0 => Law::Linear(k),
                        _ => return sub.err("PRESSURE-OVERCLOSURE=LINEAR kräver lutningen på dataraden"),
                    },
                    "EXPONENTIAL" => {
                        if v.len() < 2 || v[0] <= 0.0 {
                            return sub.err("PRESSURE-OVERCLOSURE=EXPONENTIAL kräver c0, p0");
                        }
                        Law::Exponential { c0: v[0], p0: v[1] }
                    }
                    "TABULAR" => {
                        // dataraderna: tryck, överlappning
                        let mut t = [(0.0, 0.0); 32];
                        let mut n = 0;
                        for d in &sub.data {
                            let f = sub.floats(d)?;
                            if f.len() < 2 {
                                return sub.err_at(d.line, "skriv tryck, överlappning");
                            }
                            if n == 32 {
                                return sub.err("högst 32 punkter i PRESSURE-OVERCLOSURE=TABULAR");
                            }
                            t[n] = (f[1], f[0]);
                            n += 1;
                        }
                        if n < 2 {
                            return sub.err("PRESSURE-OVERCLOSURE=TABULAR kräver minst två punkter");
                        }
                        t[..n].sort_by(|a, b| a.0.total_cmp(&b.0));
                        if t[0].1 != 0.0 {
                            return sub.err("första punkten i PRESSURE-OVERCLOSURE=TABULAR måste ha trycket 0");
                        }
                        Law::Tabular(t, n)
                    }
                    "SCALE FACTOR" => return sub.err("PRESSURE-OVERCLOSURE=SCALE FACTOR finns bara för general contact i explicit analys"),
                    other => return sub.err(format!("PRESSURE-OVERCLOSURE={other} stöds inte ännu (HARD, LINEAR, EXPONENTIAL, TABULAR)")),
                };
                if p.enforce.is_some() && !matches!(p.law, Law::Hard) {
                    return sub.err("PENALTY/AUGMENTED LAGRANGE/DIRECT gäller bara hård kontakt");
                }
            }
            "FRICTION" => {
                if sub.has("DEPENDENCIES") || sub.has("TAUMAX") || sub.has("TEST DATA") {
                    return sub.err("beroende eller begränsad (TAUMAX) friktion stöds inte ännu");
                }
                // område contact3: anisotrop friktion (μ₁, μ₂ i de lokala riktningarna)
                if sub.has("ANISOTROPIC") {
                    let (m1, m2) = ext::anisotropic(sub)?;
                    p.mu = m1;
                    p.mu2 = Some(m2);
                    if let Some(s) = sub.param("SLIP TOLERANCE").and_then(parse_f64) {
                        p.slip_tol = s;
                    }
                    if let Some(s) = sub.param("ELASTIC SLIP").and_then(parse_f64) {
                        p.elastic_slip = Some(s);
                    }
                    continue;
                }
                p.rough = sub.has("ROUGH");
                p.lagrange = sub.has("LAGRANGE");
                if sub.has("EXPONENTIAL DECAY") {
                    // datarad: μ_s, μ_k, d_c
                    let v = sub.first_floats()?;
                    if v.len() < 3 {
                        return sub.err("EXPONENTIAL DECAY kräver μ_s, μ_k, d_c");
                    }
                    p.mu = v[0];
                    p.decay = Some((v[1], v[2]));
                } else if !p.rough {
                    if sub.data.len() > 1 {
                        return sub.err("tabulerad friktion stöds inte ännu (en rad med μ)");
                    }
                    p.mu = sub.first_floats()?.first().copied().unwrap_or(0.0);
                }
                if let Some(s) = sub.param("SLIP TOLERANCE").and_then(parse_f64) {
                    p.slip_tol = s;
                }
                if let Some(s) = sub.param("ELASTIC SLIP") {
                    match parse_f64(s) {
                        Some(g) if g > 0.0 => p.elastic_slip = Some(g),
                        _ => return sub.err("ELASTIC SLIP= kräver ett positivt tal"),
                    }
                }
            }
            _ => p.damp = Some(damping::parse(sub, false)?),
        }
    }
    Ok(p)
}

/// Villkoret för en slavnod vid en utvärdering.
#[derive(Debug, Clone, Default)]
struct NodeTerm {
    /// (nod, koefficient): 1 för slavnoden, −T_AC för masternoderna.
    coef: Vec<(usize, f64)>,
    normal: [f64; 3],
    area: f64,
    /// Utjämning av nodtrycket för utdata (CPRESS): (slavnodens index, vikt).
    /// Tom = trycket skrivs ut som det är.
    smooth: Vec<(usize, f64)>,
    /// Vektorkoefficienter för gapet (normal per integrationspunkt):
    /// g = Σ_B v_B · x_B. Tom = g = n · Σ_B c_B x_B.
    vcoef: Vec<(usize, [f64; 3])>,
}

/// Tillstånd per slavnod.
#[derive(Debug, Clone, Copy, Default)]
struct NodeState {
    p: f64,
    gap: f64,
    /// Elastisk glidning (vektor i tangentplanet).
    slip: [f64; 3],
    /// Total glidning (för CSLIP).
    total_slip: [f64; 3],
    shear: [f64; 3],
    normal: [f64; 3],
    contact: bool,
    /// Glidhastighet i senaste inkrementet (EXPONENTIAL DECAY).
    rate: f64,
}

#[derive(Debug, Default)]
struct State {
    committed: Vec<NodeState>,
    trial: Vec<NodeState>,
    /// Förskjutningar vid senast godkända inkrement och vid senaste utvärderingen.
    u_committed: Vec<NodeVec>,
    u_last: Vec<NodeVec>,
    /// Utjämningsvikter för CPRESS (se `NodeTerm::smooth`).
    smooth_trial: Vec<Vec<(usize, f64)>>,
    smooth_committed: Vec<Vec<(usize, f64)>>,
}

/// Slavsidan: elementytor eller noder (nodbaserad yta med areor).
enum SlaveSide {
    Facets(Vec<Facet>),
    Nodes(Vec<(usize, f64)>),
}

pub struct ContactPair {
    name: String,
    slave: SlaveSide,
    /// Slavnoderna (sorterade) och index per nod.
    nodes: Vec<usize>,
    master: Vec<Facet>,
    master_nodes: Vec<usize>,
    /// Masterytornas storlek (för sökavståndet bakom ytan).
    master_size: Vec<f64>,
    /// Förlängning av masterytan vid fria kanter (S2S finite sliding), i parameterenheter.
    master_ext: Vec<[f64; 4]>,
    /// Granne över varje kant (ξ = −1, ξ = 1, η = −1, η = 1) i masterytan, utom över hörn.
    master_adj: Vec<[Option<usize>; 4]>,
    s2s: bool,
    props: Props,
    /// Straffkoefficient (tryck per överlappning) per slavnod för HARD.
    k_hard: Vec<f64>,
    /// Kritisk elastisk glidning.
    gamma: f64,
    dim: usize,
    /// Vikt för kraftbidraget (0,5 för symmetriska par i general contact).
    weight: f64,
    /// Utjämning av masterytan (SMOOTH=, nod mot yta med finite sliding i 2D).
    smooth: f64,
    /// SMALL SLIDING: villkoren beräknas en gång i referensläget.
    small: bool,
    /// Förberäknade villkor (SMALL SLIDING).
    fixed: Option<Vec<Option<NodeTerm>>>,
    /// Tillägg till gapet per slavnod (*CLEARANCE): g = g_geom − g_geom,0 + c.
    gap_offset: Vec<f64>,
    /// Initialt gap per slavnod med *CLEARANCE (för tangenten vid öppet spel).
    initial_gap: Vec<f64>,
    state: Mutex<State>,
    /// Självkontakt (slav- och masterytan är samma yta, `contact_ext.rs`):
    /// masterytor vars element innehåller slavpunktens noder hoppas över.
    self_contact: bool,
    /// Nodlistan för varje masterytas element (självkontakt).
    master_elem_nodes: Vec<Vec<usize>>,
}

/// Representativ translationsstyvhet (medelvärde av diagonalen) för ett element.
fn element_stiffness(m: &FeModel, e: usize) -> Result<f64> {
    let el = &m.elements[e];
    let section = el.section.map(|s| &m.sections[s]);
    let mi = section.and_then(|s| s.material_index);
    let mat = match mi {
        Some(i) => Some(crate::materials::build(&m.materials[i])?),
        None => None,
    };
    let ctx = ElemCtx {
        index: e,
        label: el.label,
        x: el.nodes.iter().map(|&n| m.nodes[n]).collect(),
        section,
        material: mat.as_deref(),
        density: 0.0,
        nlgeom: false,
        time: 0.0,
        dt: 0.0,
        temp: Vec::new(),
        temp_old: Vec::new(),
        orientation: None,
    };
    let k = el.etype.stiffness(&ctx)?;
    let n = el.etype.n_dofs();
    let mut s = 0.0;
    let mut c = 0;
    let mut q = 0;
    for (a, _) in el.nodes.iter().enumerate() {
        for &d in el.etype.node_dofs(a) {
            if d <= 3 {
                s += k[q * n + q].abs();
                c += 1;
            }
            q += 1;
        }
    }
    Ok(if c > 0 { s / c as f64 } else { 0.0 })
}

/// Standardstyvhet för straffmetoden vid en elementyta (tryck per
/// överlappning): 10 · M / h, där h = elementvolym / ytans area och
/// M = (3λ + 8μ)/9 för isotrop elasticitet. Formeln är kontrollerad
/// för CPE4 (olika E, ν och elementhöjd); för 3D-element är den inte
/// kontrollerad. Saknas isotropt *ELASTIC används elementets medelstyvhet (diagonalen) / ytans area.
fn penalty_stiffness(m: &FeModel, f: &Facet) -> Result<f64> {
    let el = &m.elements[f.elem];
    // stel yta (R3D4, R2D2, ...): oändligt styv, den deformerbara sidan bestämmer
    if crate::elements::rigid::is_rigid(el.etype) {
        return Ok(f64::INFINITY);
    }
    let xs: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
    let xf = geom::coords(m, f, None);
    let af: f64 = geom::nodal_areas(f, &xf).iter().sum::<f64>() / f.thick;
    let vol = element_volume(m, f.elem, &xs);
    let iso = el.section.and_then(|s| m.sections[s].material_index).and_then(|i| m.materials[i].behavior("ELASTIC")).filter(|b| b.param_upper("TYPE").is_none_or(|t| t == "ISOTROPIC"));
    if let (Some(b), true) = (iso, vol > 0.0 && af > 0.0) {
        let v = b.first_floats()?;
        if v.len() >= 2 && b.data.len() == 1 {
            let (e, nu) = (v[0], v[1]);
            let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let mu = e / (2.0 * (1.0 + nu));
            return Ok(10.0 * (3.0 * lam + 8.0 * mu) / 9.0 / (vol / af));
        }
    }
    Ok(10.0 * element_stiffness(m, f.elem)? / (af * f.thick).max(1e-300))
}

/// Elementets volym (area i 2D) med divergenssatsen över elementytorna.
fn element_volume(m: &FeModel, e: usize, xs: &[[f64; 3]]) -> f64 {
    let el = &m.elements[e];
    let c = crate::elements::common::centroid(xs);
    let dim = el.etype.dim();
    let mut v = 0.0;
    for face in el.etype.faces() {
        let f = Facet { elem: e, nodes: face.iter().map(|&k| el.nodes[k]).collect(), inside: c, dim, thick: 1.0 };
        let x: Vec<[f64; 3]> = f.nodes.iter().map(|&n| m.nodes[n]).collect();
        for (xi, w) in geom::gauss(dim, x.len(), 3) {
            let fp = geom::eval(&f, &x, xi);
            v += dot(sub(fp.x, c), fp.normal) * w * fp.jac;
        }
    }
    v / dim as f64
}

/// Ytans storlek: längden i 2D, √area i 3D (antagande).
fn facet_size(f: &Facet, x: &[[f64; 3]]) -> f64 {
    if f.dim == 3 {
        return geom::nodal_areas(f, x).iter().sum::<f64>().sqrt();
    }
    let mut d = 0.0f64;
    for a in x {
        for b in x {
            d = d.max(norm(sub(*a, *b)));
        }
    }
    d
}

/// Integrationspunkter (ξ, vikt i parameterytan) per hörnnod för
/// segmentviktningen i SURFACE TO SURFACE: varje slavnod får den del av ytelementet
/// som ligger närmast noden i parameterrummet (halva kanten i 2D, en
/// fjärdedel av fyrhörningen i 3D) med konstant vikt 1. Punkterna i varje
/// del är bestämda ur kontaktkrafter mot masterytor med en knyck på känt
/// läge: i 2D fyra punkter med lika vikt i
/// t = ½ ± √0,15 och ½ ± √0,15/3 längs halvkanten, i 3D 3 × 3 punkter med
/// lika vikt i t = ½, ½ ± 1/(2√2) (Chebyshevs trepunktsregel) per riktning.
/// Andra ytor (trianglar, kvadratiska) → None.
fn tributary_points(f: &Facet) -> Option<Vec<Vec<([f64; 2], f64)>>> {
    match (f.dim, f.nodes.len()) {
        (2, 2) => {
            let a = 0.15f64.sqrt();
            let t = [0.5 - a, 0.5 - a / 3.0, 0.5 + a / 3.0, 0.5 + a];
            // nod 0: ξ ∈ [−1, 0], nod 1: ξ ∈ [0, 1]
            Some(vec![t.iter().map(|&t| ([-1.0 + t, 0.0], 0.25)).collect(), t.iter().map(|&t| ([t, 0.0], 0.25)).collect()])
        }
        (3, 4) => {
            let a = 0.5 / 2f64.sqrt();
            let t = [0.5 - a, 0.5, 0.5 + a];
            let corner = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            Some(
                corner
                    .iter()
                    .map(|&(cr, cs)| {
                        let mut v = Vec::new();
                        for &tr in &t {
                            for &ts in &t {
                                // från hörnet (t = 0) mot mitten (t = 1)
                                v.push(([cr * (1.0 - tr), cs * (1.0 - ts)], 1.0 / 9.0));
                            }
                        }
                        v
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Skärningen mellan strålen p + t n och ytan: (ξ, t) om skärningen ligger
/// inom ytelementet (tolerans 1e-6 i parameterrummet).
/// `ext` = tillåten förlängning utanför kanterna ξ = −1, ξ = 1, η = −1, η = 1
/// (i parameterenheter).
fn ray_project(f: &Facet, x: &[[f64; 3]], p: [f64; 3], n: [f64; 3], ext: [f64; 4]) -> Option<([f64; 2], f64)> {
    let tri = geom::is_tri(f.dim, x.len());
    // två riktningar vinkelräta mot n
    let e1 = if f.dim == 2 { [n[1], -n[0], 0.0] } else {
        let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let d = dot(a, n);
        let v = [a[0] - d * n[0], a[1] - d * n[1], a[2] - d * n[2]];
        scale(v, 1.0 / norm(v))
    };
    let e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
    let mut xi = if tri { [1.0 / 3.0, 1.0 / 3.0] } else { [0.0, 0.0] };
    let mut conv = false;
    for _ in 0..30 {
        let fp = geom::eval(f, x, xi);
        let d = sub(fp.x, p);
        if f.dim == 2 {
            let h = dot(e1, fp.t1);
            if h.abs() < 1e-300 {
                return None;
            }
            let st = -dot(e1, d) / h;
            xi[0] += st;
            if st.abs() < 1e-14 {
                conv = true;
                break;
            }
        } else {
            let (a11, a12, a21, a22) = (dot(e1, fp.t1), dot(e1, fp.t2), dot(e2, fp.t1), dot(e2, fp.t2));
            let det = a11 * a22 - a12 * a21;
            if det.abs() < 1e-300 {
                return None;
            }
            let (b1, b2) = (-dot(e1, d), -dot(e2, d));
            let s1 = (a22 * b1 - a12 * b2) / det;
            let s2 = (a11 * b2 - a21 * b1) / det;
            xi[0] += s1;
            xi[1] += s2;
            if s1.abs() + s2.abs() < 1e-14 {
                conv = true;
                break;
            }
        }
        for v in xi.iter_mut() {
            *v = v.clamp(-3.0, 3.0);
        }
    }
    if !conv {
        // stora koordinater: stegkriteriet nås inte p.g.a. avrundning; godta
        // om strålens avstånd till punkten är försumbart relativt ytans storlek
        let fp = geom::eval(f, x, xi);
        let d = sub(fp.x, p);
        let size = norm(fp.t1) + norm(fp.t2);
        let res = if f.dim == 2 { dot(e1, d).abs() } else { dot(e1, d).abs() + dot(e2, d).abs() };
        conv = res <= 1e-10 * size.max(1e-300);
    }
    let tol = 1e-6;
    let inside = if f.dim == 2 {
        xi[0] >= -1.0 - ext[0] - tol && xi[0] <= 1.0 + ext[1] + tol
    } else if tri {
        xi[0] >= -tol && xi[1] >= -tol && xi[0] + xi[1] <= 1.0 + tol
    } else {
        xi[0] >= -1.0 - ext[0] - tol && xi[0] <= 1.0 + ext[1] + tol && xi[1] >= -1.0 - ext[2] - tol && xi[1] <= 1.0 + ext[3] + tol
    };
    if !conv || !inside {
        return None;
    }
    let fp = geom::eval(f, x, xi);
    Some((xi, dot(sub(fp.x, p), n)))
}

/// Masterytans förlängning vid fria kanter (kanter utan granne i samma yta
/// eller med en granne över ett hörn): `frac` av ytelementets storlek, i
/// parameterenheter per kant (ξ = −1, ξ = 1, η = −1, η = 1).
fn master_extension(master: &[Facet], x: &[[f64; 3]], frac: f64) -> Vec<[f64; 4]> {
    let normal = |f: &Facet| {
        let xs: Vec<[f64; 3]> = f.nodes.iter().map(|&n| x[n]).collect();
        geom::eval(f, &xs, [0.0, 0.0]).normal
    };
    let normals: Vec<[f64; 3]> = master.iter().map(normal).collect();
    master
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let edges: Vec<Vec<usize>> = match (f.dim, f.nodes.len()) {
                (2, 2) => vec![vec![f.nodes[0]], vec![f.nodes[1]]],
                (3, 4) => vec![vec![f.nodes[0], f.nodes[3]], vec![f.nodes[1], f.nodes[2]], vec![f.nodes[0], f.nodes[1]], vec![f.nodes[2], f.nodes[3]]],
                _ => return [0.0; 4],
            };
            let mut e = [0.0; 4];
            for (k, ed) in edges.iter().enumerate() {
                let neighbour = master.iter().enumerate().any(|(j, g)| j != i && ed.iter().all(|n| g.nodes.contains(n)) && dot(normals[i], normals[j]) > CORNER_COS);
                if !neighbour {
                    e[k] = 2.0 * frac;
                }
            }
            e
        })
        .collect()
}

/// Andel av en cirkelskiva (radie 1, centrum i origo) med x > a och y > b.
fn quadrant_fraction(a: f64, b: f64) -> f64 {
    if a >= 1.0 || b >= 1.0 {
        return 0.0;
    }
    // x = sin θ, dx = cos θ dθ; längden av {y > b} ∩ skivan är s − max(b, −s), s = cos θ
    let th0 = a.max(-1.0).asin();
    let th1 = std::f64::consts::FRAC_PI_2;
    let mut cuts = vec![th0, th1];
    if b.abs() < 1.0 {
        let tb = b.abs().acos();
        for t in [tb, -tb] {
            if t > th0 && t < th1 {
                cuts.push(t);
            }
        }
    }
    cuts.sort_by(|x, y| x.total_cmp(y));
    let g = geom::gauss(2, 2, 4);
    let mut area = 0.0;
    for w in cuts.windows(2) {
        let (t0, t1) = (w[0], w[1]);
        let nsub = 8;
        for k in 0..nsub {
            let a0 = t0 + (t1 - t0) * k as f64 / nsub as f64;
            let a1 = t0 + (t1 - t0) * (k + 1) as f64 / nsub as f64;
            for &(xi, wg) in &g {
                let t = 0.5 * (a0 + a1) + 0.5 * (a1 - a0) * xi[0];
                let s = t.cos();
                let len = (s - b.max(-s)).max(0.0);
                area += wg * 0.5 * (a1 - a0) * len * s;
            }
        }
    }
    area / std::f64::consts::PI
}

/// Grannelementen över masterytornas kanter (ξ = −1, ξ = 1, η = −1, η = 1;
/// i 2D nod 0 och nod 1), utom över hörn (normalerna skiljer mer än
/// arccos(CORNER_COS)).
fn master_adjacency(master: &[Facet], x: &[[f64; 3]]) -> Vec<[Option<usize>; 4]> {
    let normals: Vec<[f64; 3]> = master.iter().map(|f| geom::eval(f, &f.nodes.iter().map(|&n| x[n]).collect::<Vec<_>>(), [0.0, 0.0]).normal).collect();
    master
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let edges: Vec<Vec<usize>> = match (f.dim, f.nodes.len()) {
                (2, 2) => vec![vec![f.nodes[0]], vec![f.nodes[1]]],
                (3, 4) => vec![vec![f.nodes[0], f.nodes[3]], vec![f.nodes[1], f.nodes[2]], vec![f.nodes[0], f.nodes[1]], vec![f.nodes[2], f.nodes[3]]],
                _ => return [None; 4],
            };
            let mut a = [None; 4];
            for (k, ed) in edges.iter().enumerate() {
                a[k] = master.iter().enumerate().position(|(j, g)| j != i && ed.iter().all(|n| g.nodes.contains(n)) && dot(normals[i], normals[j]) > CORNER_COS);
            }
            a
        })
        .collect()
}

impl ContactPair {
    #[allow(clippy::too_many_arguments)]
    fn new(m: &FeModel, kw: &RawKeyword, name: String, slave_name: &str, master_name: &str, props: Props, s2s: bool, small: bool, weight: f64) -> Result<ContactPair> {
        let ss = geom::surface(m, kw, slave_name)?;
        let ms = geom::surface(m, kw, master_name)?;
        if ms.kind == "NODE" {
            return kw.err("masterytan ska vara elementbaserad");
        }
        let master = geom::facets(m, ms, &name)?;
        let mut master_nodes: Vec<usize> = master.iter().flat_map(|f| f.nodes.iter().copied()).collect();
        master_nodes.sort_unstable();
        master_nodes.dedup();
        let (slave, s2s) = if ss.kind == "NODE" { (SlaveSide::Nodes(ss.nodes.clone()), false) } else { (SlaveSide::Facets(geom::facets(m, ss, &name)?), s2s) };
        let nodes: Vec<usize> = match &slave {
            SlaveSide::Facets(f) => {
                let mut v: Vec<usize> = f.iter().flat_map(|f| f.nodes.iter().copied()).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
            SlaveSide::Nodes(n) => {
                let mut v: Vec<usize> = n.iter().map(|p| p.0).collect();
                v.sort_unstable();
                v.dedup();
                v
            }
        };
        let dim = m.dim();
        // representativ styvhet per slavnod: medel av underliggande element på båda sidor
        let mut kn = vec![0.0; nodes.len()];
        let mut cnt = vec![0usize; nodes.len()];
        let mut sizes = Vec::new();
        let mut mk = 0.0;
        for f in &master {
            mk += element_stiffness(m, f.elem)?;
        }
        mk /= master.len().max(1) as f64;
        match &slave {
            SlaveSide::Facets(fs) => {
                for f in fs {
                    let k = element_stiffness(m, f.elem)?;
                    let x = geom::coords(m, f, None);
                    sizes.push(facet_size(f, &x));
                    for &n in &f.nodes {
                        let i = nodes.binary_search(&n).unwrap();
                        kn[i] += k;
                        cnt[i] += 1;
                    }
                }
            }
            SlaveSide::Nodes(_) => {}
        }
        let ns_facets = sizes.len();
        let master_size: Vec<f64> = master.iter().map(|f| facet_size(f, &geom::coords(m, f, None))).collect();
        sizes.extend(&master_size);
        // karakteristisk längd = slavytornas medelstorlek (kontrollerat i 2D;
        // utan elementbaserad slavyta används masterytan)
        let lc = if ns_facets > 0 { sizes[..ns_facets].iter().sum::<f64>() / ns_facets as f64 } else { sizes.iter().sum::<f64>() / sizes.len().max(1) as f64 };
        // tributär area per slavnod (för straffkoefficientens skalning)
        let mut area = vec![0.0; nodes.len()];
        match &slave {
            SlaveSide::Facets(fs) => {
                for f in fs {
                    let x = geom::coords(m, f, None);
                    for (k, a) in geom::nodal_areas(f, &x).into_iter().enumerate() {
                        area[nodes.binary_search(&f.nodes[k]).unwrap()] += a;
                    }
                }
            }
            SlaveSide::Nodes(ns) => {
                for &(n, w) in ns {
                    area[nodes.binary_search(&n).unwrap()] += if w > 0.0 { w } else { 1.0 };
                }
            }
        }
        // stel masteryta (R3D4, R2D2, ...): bara slavsidan deformeras
        let rigid_master = !master.is_empty() && master.iter().all(|f| crate::elements::rigid::is_rigid(m.elements[f.elem].etype));
        // straffmetod för SURFACE TO SURFACE med finite sliding, annars direkt
        let enforce = props.enforce.unwrap_or(if s2s && !small { Enforce::Penalty(None, 1.0) } else { Enforce::Direct });
        let k_hard: Vec<f64> = match enforce {
            Enforce::Penalty(Some(k), sc) => vec![k * sc; nodes.len()],
            Enforce::Penalty(None, sc) => {
                // standardstyvhet: 10 · M/h för den mjukaste sidan, med de
                // ytelement som möts vid slavnoden (närmaste vända masteryta i
                // referensläget och slavnodens ytelement vända mot den); general
                // contact ger samma resultat som motsvarande *CONTACT PAIR
                let pmf: Vec<f64> = master.iter().map(|f| penalty_stiffness(m, f)).collect::<Result<_>>()?;
                let mnorm: Vec<[f64; 3]> = master.iter().map(|f| geom::eval(f, &geom::coords(m, f, None), [0.0, 0.0]).normal).collect();
                let pm_all = pmf.iter().sum::<f64>() / pmf.len().max(1) as f64;
                let mut out = Vec::with_capacity(nodes.len());
                for (i, &n) in nodes.iter().enumerate() {
                    let sfs: Vec<&Facet> = match &slave {
                        SlaveSide::Facets(fs) => fs.iter().filter(|f| f.nodes.contains(&n)).collect(),
                        SlaveSide::Nodes(_) => Vec::new(),
                    };
                    let snorm: Vec<[f64; 3]> = sfs.iter().map(|f| geom::eval(f, &geom::coords(m, f, None), [0.0, 0.0]).normal).collect();
                    // närmaste masteryta som är vänd mot någon av slavnodens ytor
                    let mut best: Option<(f64, usize)> = None;
                    for (j, f) in master.iter().enumerate() {
                        if !snorm.is_empty() && !snorm.iter().any(|ns| dot(*ns, mnorm[j]) < -0.5) {
                            continue;
                        }
                        let (_, d, _) = geom::project(f, &geom::coords(m, f, None), m.nodes[n]);
                        if best.is_none_or(|b| d < b.0) {
                            best = Some((d, j));
                        }
                    }
                    let (pm, mn) = match best {
                        Some((_, j)) => (pmf[j], Some(mnorm[j])),
                        None => (pm_all, None),
                    };
                    let mut ps = 0.0;
                    let mut pc = 0usize;
                    for (f, ns) in sfs.iter().zip(&snorm) {
                        if mn.is_none_or(|mn| dot(*ns, mn) < -0.5) {
                            ps += penalty_stiffness(m, f)?;
                            pc += 1;
                        }
                    }
                    let mut kp = if pc > 0 { (ps / pc as f64).min(pm) } else { pm };
                    if !kp.is_finite() {
                        // stel masteryta utan vänd slavyta: slavnodens egna ytor
                        let v: Vec<f64> = sfs.iter().map(|f| penalty_stiffness(m, f)).collect::<Result<_>>()?;
                        kp = v.iter().sum::<f64>() / v.len().max(1) as f64;
                    }
                    out.push(sc * kp);
                    let _ = i;
                }
                out
            }
            Enforce::Direct => (0..nodes.len())
                .map(|i| {
                    let ks = if cnt[i] > 0 { kn[i] / cnt[i] as f64 } else { mk };
                    let k = if ks > 0.0 && mk > 0.0 { 2.0 * ks * mk / (ks + mk) } else { ks.max(mk) };
                    // SMALL SLIDING är (bortsett från kontaktstatus) linjärt och tål
                    // en styvare straffkoefficient utan konvergensproblem
                    let f = if small { HARD_FACTOR_SMALL } else if rigid_master { HARD_FACTOR_RIGID } else { HARD_FACTOR };
                    f * k / area[i].max(1e-300)
                })
                .collect(),
        };
        if crate::settings::with(|s| s.contact_debug) {
            eprintln!("kontakt {name}: master k={mk:.6e}, slav k={:?}, A={:?}", kn.iter().zip(&cnt).map(|(k, c)| k / (*c).max(1) as f64).collect::<Vec<_>>(), area);
        }
        let ns = nodes.len();
        let master_ext = if s2s && !small { master_extension(&master, &m.nodes, EXTENSION_ZONE) } else { vec![[0.0; 4]; master.len()] };
        let master_adj = master_adjacency(&master, &m.nodes);
        let mut cp = ContactPair {
            master_ext,
            master_adj,
            name,
            slave,
            nodes,
            master,
            master_nodes,
            master_size,
            s2s,
            props,
            k_hard,
            gamma: props.slip_tol * lc,
            dim,
            weight,
            smooth: 0.2,
            small,
            fixed: None,
            gap_offset: vec![0.0; ns],
            initial_gap: vec![0.0; ns],
            state: Mutex::new(State {
                committed: vec![NodeState::default(); ns],
                trial: vec![NodeState::default(); ns],
                u_committed: vec![[0.0; 8]; m.nodes.len()],
                u_last: vec![[0.0; 8]; m.nodes.len()],
                smooth_trial: vec![Vec::new(); ns],
                smooth_committed: vec![Vec::new(); ns],
            }),
            self_contact: false,
            master_elem_nodes: Vec::new(),
        };
        if small {
            let x: Vec<[f64; 3]> = m.nodes.clone();
            cp.fixed = Some(cp.terms(m, &x));
        }
        Ok(cp)
    }

    /// *CLEARANCE, SLAVE=, MASTER=, VALUE= eller TABULAR (datarader "nod eller
    /// nodset, värde"; senare rader går före tidigare): det
    /// initiala gapet sätts till det givna värdet utan att geometrin ändras,
    /// g = g_geom − g_geom,0 + c (COPEN = c i frame 0). Bara för
    /// SMALL SLIDING.
    fn apply_clearance(&mut self, m: &FeModel, slave: &str, master: &str) -> Result<()> {
        let (su, mu) = (slave.trim().to_uppercase(), master.trim().to_uppercase());
        for kw in m.extra.iter().filter(|k| k.name == "CLEARANCE") {
            let ks = kw.param_upper("SLAVE").unwrap_or_default();
            let km = kw.param_upper("MASTER").unwrap_or_default();
            if ks != su || km != mu {
                continue;
            }
            if !self.small {
                return kw.err("*CLEARANCE gäller bara kontaktpar med SMALL SLIDING");
            }
            let mut c: Vec<Option<f64>> = vec![None; self.nodes.len()];
            if let Some(v) = kw.param("VALUE") {
                // VALUE gäller alla slavnoder; dataraderna används inte
                let Some(v) = parse_f64(v) else { return kw.err("VALUE= ska vara ett tal") };
                c.iter_mut().for_each(|x| *x = Some(v));
            } else if !kw.has("TABULAR") {
                return kw.err("*CLEARANCE kräver VALUE= eller TABULAR");
            }
            for d in kw.records().into_iter().filter(|_| kw.param("VALUE").is_none()) {
                let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
                if f.is_empty() {
                    continue;
                }
                if f.len() < 2 {
                    return kw.err_at(d.line, "skriv nod (eller nodset), spel");
                }
                let Some(v) = parse_f64(f[1]) else { return kw.err_at(d.line, "spelet ska vara ett tal") };
                for n in geom::node_ref(m, kw, f[0])? {
                    match self.nodes.binary_search(&n) {
                        Ok(i) => c[i] = Some(v),
                        Err(_) => return kw.err_at(d.line, format!("nod {} är inte slavnod i paret", m.node_name(n))),
                    }
                }
            }
            let Some(terms) = &self.fixed else { return kw.err("*CLEARANCE kräver SMALL SLIDING") };
            for (i, t) in terms.iter().enumerate() {
                let (Some(t), Some(ci)) = (t, c[i]) else { continue };
                let g0: f64 = if t.vcoef.is_empty() {
                    let mut r = [0.0; 3];
                    for &(b, cb) in &t.coef {
                        for d in 0..3 {
                            r[d] += cb * m.nodes[b][d];
                        }
                    }
                    dot(t.normal, r)
                } else {
                    t.vcoef.iter().map(|(b, v)| dot(*v, m.nodes[*b])).sum()
                };
                self.gap_offset[i] = ci - g0;
                self.initial_gap[i] = ci;
            }
        }
        Ok(())
    }

    /// Villkor per slavnod i konfigurationen `x` (alla noders koordinater).
    fn terms(&self, _m: &FeModel, x: &[[f64; 3]]) -> Vec<Option<NodeTerm>> {
        let mx: Vec<Vec<[f64; 3]>> = self.master.iter().map(|f| f.nodes.iter().map(|&n| x[n]).collect()).collect();
        // närmaste masteryta som är vänd mot slavytan (om slavnormalen ges)
        // självkontakt: slavpunktens noder (masterytor i samma element hoppas över)
        let skip: std::cell::RefCell<Vec<usize>> = std::cell::RefCell::new(Vec::new());
        let skipped = |i: usize| self.self_contact && self.master_elem_nodes[i].iter().any(|n| skip.borrow().contains(n));
        let closest = |p: [f64; 3], ns: Option<[f64; 3]>| -> Option<(usize, [f64; 2], [f64; 3], [f64; 3])> {
            let mut best: Option<(usize, [f64; 2], f64, [f64; 3], [f64; 3])> = None;
            for (i, f) in self.master.iter().enumerate() {
                if skipped(i) {
                    continue;
                }
                let (xi, d, inside) = geom::project(f, &mx[i], p);
                if !inside {
                    continue;
                }
                let fp = geom::eval(f, &mx[i], xi);
                // masterytans normal ska vara ungefär motriktad slavnormalen
                // (inom 60°), annars är ytorna inte vända mot varandra
                if let Some(n) = ns {
                    if dot(n, fp.normal) > -0.5 * norm(n) {
                        continue;
                    }
                }
                // punkter långt bakom masterytan (mer än en halv ytstorlek) är
                // inte i kontakt med den (t.ex. kroppens baksida i general contact)
                if dot(sub(p, fp.x), fp.normal) < -0.5 * self.master_size[i] {
                    continue;
                }
                if best.as_ref().is_none_or(|b| d < b.2) {
                    best = Some((i, xi, d, fp.normal, fp.x));
                }
            }
            best.map(|b| (b.0, b.1, b.3, b.4))
        };
        // masterpunkt längs slavnormalen (S2S): ytan som strålen x_s − t n_s
        // träffar inom ytelementet närmast slavpunkten (minst |t|)
        let closest_ray = |p: [f64; 3], ns: [f64; 3]| -> Option<(usize, [f64; 2], [f64; 3], [f64; 3])> {
            let mut best: Option<(usize, [f64; 2], f64, [f64; 3], [f64; 3])> = None;
            for (i, f) in self.master.iter().enumerate() {
                if skipped(i) {
                    continue;
                }
                let Some((xi, t)) = ray_project(f, &mx[i], p, ns, self.master_ext[i]) else { continue };
                let fp = geom::eval(f, &mx[i], xi);
                if dot(ns, fp.normal) > -0.5 {
                    continue;
                }
                // t < 0: genomträngning; för långt bakom masterytan räknas inte
                if t < -0.5 * self.master_size[i] {
                    continue;
                }
                // träffar i förlängningen räknas efter träffar inom ytan
                let outside = xi[0].abs() > 1.0 + 1e-9 || (f.dim == 3 && xi[1].abs() > 1.0 + 1e-9);
                let key = t.abs() + if outside { 1e30 } else { 0.0 };
                if best.as_ref().is_none_or(|b| key < b.2) {
                    best = Some((i, xi, key, fp.normal, fp.x));
                }
            }
            best.map(|b| (b.0, b.1, b.3, b.4))
        };
        let mut out: Vec<Option<NodeTerm>> = vec![None; self.nodes.len()];
        let idx = |n: usize| self.nodes.binary_search(&n).unwrap();
        match &self.slave {
            SlaveSide::Facets(fs) if self.s2s && fs.iter().all(|f| tributary_points(f).is_some()) => {
                // segmentviktning: konstant vikt över slavnodens del av ytelementet
                let mut acc: Vec<(f64, BTreeMap<usize, f64>, BTreeMap<usize, f64>, [f64; 3])> = vec![(0.0, BTreeMap::new(), BTreeMap::new(), [0.0; 3]); self.nodes.len()];
                // vektorkoefficienter: Σ_p w_p (−n_p) N_B (slav) och −Σ_p w_p (−n_p) M_C (master)
                let mut vacc: Vec<BTreeMap<usize, [f64; 3]>> = vec![BTreeMap::new(); self.nodes.len()];
                for sf in fs {
                    let xs: Vec<[f64; 3]> = sf.nodes.iter().map(|&n| x[n]).collect();
                    let pts = tributary_points(sf).unwrap();
                    for (k, &sn) in sf.nodes.iter().enumerate() {
                        let i = idx(sn);
                        for &(xi, w) in &pts[k] {
                            let fp = geom::eval(sf, &xs, xi);
                            if self.self_contact {
                                skip.borrow_mut().clone_from(&sf.nodes);
                            }
                            let Some((mi, mxi, _, xproj)) = closest_ray(fp.x, fp.normal) else {
                                if crate::settings::with(|s| s.contact_debug) {
                                    eprintln!("cell: ingen masterpunkt för {:?}", fp.x);
                                }
                                continue;
                            };
                            let mf = &self.master[mi];
                            let (shm, _, _) = geom::shape(mf.dim, mf.nodes.len(), mxi);
                            // masterytans vikter: SMALL SLIDING formfunktionerna i
                            // projektionen, annars cellfördelningen (`cell_weights`)
                            let (mw, phi) = if self.small {
                                (mf.nodes.iter().copied().zip(shm.iter().copied()).collect::<Vec<_>>(), 1.0)
                            } else {
                                self.cell_weights(&mx, sf, &xs, mi, mxi, fp.x, fp.normal)
                            };
                            let _ = xproj;
                            if phi <= 0.0 {
                                continue;
                            }
                            let wg = w * fp.jac;
                            let e = &mut acc[i];
                            let va = &mut vacc[i];
                            let nv = scale(fp.normal, -1.0);
                            let mut add_v = |n: usize, c: f64| {
                                let v = va.entry(n).or_insert([0.0; 3]);
                                for d in 0..3 {
                                    v[d] += c * nv[d];
                                }
                            };
                            e.0 += wg * phi;
                            for (b, &nb) in sf.nodes.iter().enumerate() {
                                *e.1.entry(nb).or_default() += wg * phi * fp.n[b];
                                add_v(nb, wg * phi * fp.n[b]);
                            }
                            for &(mn, c) in &mw {
                                *e.2.entry(mn).or_default() += wg * c;
                                add_v(mn, -wg * c);
                            }
                            for c in 0..3 {
                                e.3[c] -= wg * fp.normal[c];
                            }
                        }
                    }
                }
                for (i, ((area, ds, ms, nsum), va)) in acc.into_iter().zip(vacc).enumerate() {
                    let l = norm(nsum);
                    if area <= 0.0 || l == 0.0 {
                        continue;
                    }
                    let mut coef: Vec<(usize, f64)> = Vec::new();
                    let mut smooth = Vec::new();
                    for (n, d) in ds {
                        if d != 0.0 {
                            coef.push((n, d / area));
                            smooth.push((idx(n), d / area));
                        }
                    }
                    coef.extend(ms.into_iter().filter(|p| p.1 != 0.0).map(|(n, c)| (n, -c / area)));
                    let vcoef = va.into_iter().map(|(n, v)| (n, scale(v, 1.0 / area))).collect();
                    out[i] = Some(NodeTerm { coef, normal: scale(nsum, 1.0 / l), area, smooth, vcoef });
                }
            }
            SlaveSide::Facets(fs) if self.s2s => {
                let mpts: Vec<[f64; 3]> = self.master_nodes.iter().map(|&n| x[n]).collect();
                let mut acc: Vec<(f64, f64, BTreeMap<usize, f64>, [f64; 3])> = vec![(0.0, 0.0, BTreeMap::new(), [0.0; 3]); self.nodes.len()];
                for sf in fs {
                    let xs: Vec<[f64; 3]> = sf.nodes.iter().map(|&n| x[n]).collect();
                    let nn = xs.len();
                    let (a, _) = geom::dual_basis(sf, &xs);
                    for (xi, wg) in geom::mortar_points(sf, &xs, &mpts, 8) {
                        let fp = geom::eval(sf, &xs, xi);
                        if self.self_contact {
                            skip.borrow_mut().clone_from(&sf.nodes);
                        }
                        let Some((mi, mxi, nm, _)) = closest(fp.x, Some(fp.normal)) else { continue };
                        let mf = &self.master[mi];
                        let (shm, _, _) = geom::shape(mf.dim, mf.nodes.len(), mxi);
                        for (k, &sn) in sf.nodes.iter().enumerate() {
                            let i = idx(sn);
                            let wn = fp.n[k] * wg * fp.jac;
                            let phi: f64 = (0..nn).map(|b| a[k * nn + b] * fp.n[b]).sum();
                            let wp = phi * wg * fp.jac;
                            let e = &mut acc[i];
                            e.0 += wn;
                            e.1 += wp;
                            for (j, &mn) in mf.nodes.iter().enumerate() {
                                *e.2.entry(mn).or_default() += wp * shm[j];
                            }
                            for c in 0..3 {
                                e.3[c] += wn * nm[c];
                            }
                        }
                    }
                }
                for (i, (area, phisum, map, nsum)) in acc.into_iter().enumerate() {
                    if area <= 0.0 || phisum.abs() <= 1e-12 * area {
                        continue;
                    }
                    let l = norm(nsum);
                    if l == 0.0 {
                        continue;
                    }
                    let mut coef = vec![(self.nodes[i], 1.0)];
                    coef.extend(map.into_iter().filter(|p| p.1 != 0.0).map(|(n, c)| (n, -c / phisum)));
                    out[i] = Some(NodeTerm { coef, normal: scale(nsum, 1.0 / l), area, smooth: Vec::new(), vcoef: Vec::new() });
                }
            }
            _ => {
                // nod mot yta
                let mut area = vec![0.0; self.nodes.len()];
                let mut nsl: Vec<Option<[f64; 3]>> = vec![None; self.nodes.len()];
                match &self.slave {
                    SlaveSide::Facets(fs) => {
                        for f in fs {
                            let xs: Vec<[f64; 3]> = f.nodes.iter().map(|&n| x[n]).collect();
                            let fp = geom::eval(f, &xs, if geom::is_tri(f.dim, xs.len()) { [1.0 / 3.0, 1.0 / 3.0] } else { [0.0, 0.0] });
                            for (k, a) in geom::nodal_areas(f, &xs).into_iter().enumerate() {
                                let i = idx(f.nodes[k]);
                                area[i] += a;
                                let v = nsl[i].unwrap_or([0.0; 3]);
                                nsl[i] = Some([v[0] + a * fp.normal[0], v[1] + a * fp.normal[1], v[2] + a * fp.normal[2]]);
                            }
                        }
                    }
                    SlaveSide::Nodes(ns) => {
                        for &(n, w) in ns {
                            area[idx(n)] += if w > 0.0 { w } else { 1.0 };
                        }
                    }
                }
                for (i, &sn) in self.nodes.iter().enumerate() {
                    if self.self_contact {
                        *skip.borrow_mut() = vec![sn];
                    }
                    let Some((mi, mxi, nm, _)) = closest(x[sn], nsl[i]) else { continue };
                    let mf = &self.master[mi];
                    let (shm, _, _) = geom::shape(mf.dim, mf.nodes.len(), mxi);
                    let mut coef = vec![(sn, 1.0)];
                    if let Some((w, n)) = self.smoothed(x, mi, mxi[0], x[sn], nm) {
                        coef.extend(w.into_iter().filter(|p| p.1 != 0.0).map(|(n, c)| (n, -c)));
                        out[i] = Some(NodeTerm { coef, normal: n, area: area[i], smooth: Vec::new(), vcoef: Vec::new() });
                        continue;
                    }
                    coef.extend(mf.nodes.iter().copied().zip(shm).filter(|p| p.1 != 0.0).map(|(n, c)| (n, -c)));
                    out[i] = Some(NodeTerm { coef, normal: nm, area: area[i], smooth: Vec::new(), vcoef: Vec::new() });
                }
            }
        }
        out
    }

    /// Mastervikter för en slavpunkt med finite sliding (SURFACE TO SURFACE).
    /// Varje integrationspunkt representerar en cell med radien r kring
    /// punkten: ett intervall [p − r, p + r] i 2D, en cirkelskiva i 3D, med
    /// r = avståndet från den yttersta integrationspunkten till delområdets
    /// kant ((½ − √0,15) · halvkantens längd i 2D, (½ − 1/(2√2)) · √(fjärdedelens
    /// area) i 3D). Cellen fördelas på masterytans ytelement efter hur stor
    /// andel av den som ligger på respektive element (grannar över kanter och
    /// hörn i samma yta), och varje elements formfunktioner tas i punkten
    /// (förlängda utanför elementet). Andelar utanför fria kanter faller bort.
    /// Returnerar (vikter per masternod, summan av andelarna). Kontrollerat med
    /// krafter på masternoder vid olika lägen av
    /// en masternod/-kant relativt slavpunkterna (2D och 3D) och föreskrivna
    /// förskjutningar nära masterytans ände.
    #[allow(clippy::too_many_arguments)]
    fn cell_weights(&self, mx: &[Vec<[f64; 3]>], sf: &Facet, xs: &[[f64; 3]], mi: usize, mxi: [f64; 2], p: [f64; 3], ns: [f64; 3]) -> (Vec<(usize, f64)>, f64) {
        let mf = &self.master[mi];
        let shape_at = |j: usize, xi: [f64; 2]| -> Vec<(usize, f64)> {
            let f = &self.master[j];
            let (sh, _, _) = geom::shape(f.dim, f.nodes.len(), xi);
            f.nodes.iter().copied().zip(sh).collect()
        };
        let lin2 = self.dim == 2 && mf.nodes.len() == 2 && sf.nodes.len() == 2;
        let quad = self.dim == 3 && mf.nodes.len() == 4 && sf.nodes.len() == 4;
        if !(lin2 || quad) {
            let inside = mxi[0].abs() <= 1.0 + 1e-6 && (self.dim == 2 || mxi[1].abs() <= 1.0 + 1e-6);
            return (shape_at(mi, mxi), if inside { 1.0 } else { 0.0 });
        }
        let fp = geom::eval(mf, &mx[mi], mxi);
        // punkten i grannens parameterrum (förlängt utanför grannen)
        let at = |j: usize| -> Option<Vec<(usize, f64)>> { ray_project(&self.master[j], &mx[j], p, ns, [3.0; 4]).map(|(xi, _)| shape_at(j, xi)) };
        let mut parts: Vec<(Option<usize>, f64)> = Vec::new();
        if lin2 {
            let r = (0.5 - 0.15f64.sqrt()) * 0.5 * norm(sub(xs[1], xs[0]));
            let l = norm(fp.t1);
            // avstånd till kanterna ξ = −1 och ξ = 1 (negativt utanför)
            let d = [(1.0 + mxi[0]) * l, (1.0 - mxi[0]) * l];
            let mut own = 1.0;
            for k in 0..2 {
                let f = ((r - d[k]) / (2.0 * r)).clamp(0.0, 1.0);
                if f > 0.0 {
                    own -= f;
                    parts.push((self.master_adj[mi][k], f));
                }
            }
            parts.push((Some(mi), own.max(0.0)));
        } else {
            let area: f64 = geom::nodal_areas(sf, xs).iter().sum();
            let r = (0.5 - 0.5 / 2f64.sqrt()) * (0.25 * area).sqrt();
            let (l1, l2) = (norm(fp.t1), norm(fp.t2));
            // kanter: ξ = −1, ξ = 1, η = −1, η = 1
            let d = [(1.0 + mxi[0]) * l1, (1.0 - mxi[0]) * l1, (1.0 + mxi[1]) * l2, (1.0 - mxi[1]) * l2];
            let seg = |d: f64| -> f64 {
                let q = (d / r).clamp(-1.0, 1.0);
                (q.acos() - q * (1.0 - q * q).sqrt()) / std::f64::consts::PI
            };
            let fr: Vec<f64> = d.iter().map(|&d| seg(d)).collect();
            let mut own = 1.0;
            for k in 0..4 {
                if fr[k] > 0.0 {
                    own -= fr[k];
                    parts.push((self.master_adj[mi][k], fr[k]));
                }
            }
            // hörn: andelen bortom två kanter har räknats för båda kanterna
            for (k1, k2) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
                if fr[k1] > 0.0 && fr[k2] > 0.0 {
                    let c = quadrant_fraction(d[k1] / r, d[k2] / r);
                    if c > 0.0 {
                        own += c;
                        parts.push((self.master_adj[mi][k1], -c));
                        parts.push((self.master_adj[mi][k2], -c));
                        let diag = match (self.master_adj[mi][k1], self.master_adj[mi][k2]) {
                            (Some(j1), Some(j2)) => {
                                let corner: Vec<usize> = mf.nodes.iter().copied().filter(|n| self.master[j1].nodes.contains(n) && self.master[j2].nodes.contains(n)).collect();
                                self.master
                                    .iter()
                                    .enumerate()
                                    .position(|(j, f)| j != mi && j != j1 && j != j2 && !corner.is_empty() && corner.iter().all(|n| f.nodes.contains(n)) && self.master_adj[j].contains(&Some(j1)) && self.master_adj[j].contains(&Some(j2)))
                            }
                            _ => None,
                        };
                        parts.push((diag, c));
                    }
                }
            }
            parts.push((Some(mi), own.max(0.0)));
        }
        let mut w: BTreeMap<usize, f64> = BTreeMap::new();
        let mut total = 0.0;
        for (j, f) in parts {
            let Some(j) = j else { continue };
            let sh = if j == mi { Some(shape_at(mi, mxi)) } else { at(j) };
            let Some(sh) = sh else {
                if crate::settings::with(|s| s.contact_debug) {
                    eprintln!("cell: grannen {j} saknar projektion (p={p:?})");
                }
                continue;
            };
            total += f;
            for (n, c) in sh {
                *w.entry(n).or_default() += f * c;
            }
        }
        if crate::settings::with(|s| s.contact_debug) && total < 1.0 - 1e-12 {
            eprintln!("cell: andel {total} vid {p:?}, master {mi} ξ={mxi:?}, grannar {:?}", self.master_adj[mi]);
        }
        (w.into_iter().filter(|p| p.1 != 0.0).collect(), total)
    }

    /// Utjämnad masteryta (2D, finite sliding, nod mot yta): nära en masternod B
    /// mellan grannarna A och C ersätts kanterna av en kvadratisk Bézierkurva
    /// från P_a = B + s (A − B) till P_b = B + s (C − B) med B som styrpunkt
    /// (s = SMOOTH, standard 0,2). Ger vikterna för A, B, C och normalen i
    /// slavpunktens projektion på kurvan.
    fn smoothed(&self, x: &[[f64; 3]], mi: usize, xi: f64, p: [f64; 3], n_facet: [f64; 3]) -> Option<(Vec<(usize, f64)>, [f64; 3])> {
        let s = self.smooth;
        let mf = &self.master[mi];
        if s <= 0.0 || self.small || self.dim != 2 || mf.nodes.len() != 2 {
            return None;
        }
        let z = 0.5 * (xi + 1.0);
        // nod B och grannkanten
        let (b, c) = if z < s {
            (mf.nodes[0], mf.nodes[1])
        } else if z > 1.0 - s {
            (mf.nodes[1], mf.nodes[0])
        } else {
            return None;
        };
        let other = self.master.iter().enumerate().find(|(j, f)| *j != mi && f.nodes.len() == 2 && f.nodes.contains(&b))?;
        let a = if other.1.nodes[0] == b { other.1.nodes[1] } else { other.1.nodes[0] };
        let (xa, xb, xc) = (x[a], x[b], x[c]);
        let pa = [xb[0] + s * (xa[0] - xb[0]), xb[1] + s * (xa[1] - xb[1]), 0.0];
        let pb = [xb[0] + s * (xc[0] - xb[0]), xb[1] + s * (xc[1] - xb[1]), 0.0];
        let curve = |t: f64| -> ([f64; 3], [f64; 3], [f64; 3]) {
            let c0 = (1.0 - t) * (1.0 - t);
            let c1 = 2.0 * t * (1.0 - t);
            let c2 = t * t;
            let pt = [0, 1, 2].map(|k| c0 * pa[k] + c1 * xb[k] + c2 * pb[k]);
            let d1 = [0, 1, 2].map(|k| 2.0 * (1.0 - t) * (xb[k] - pa[k]) + 2.0 * t * (pb[k] - xb[k]));
            let d2 = [0, 1, 2].map(|k| 2.0 * (pa[k] - 2.0 * xb[k] + pb[k]));
            (pt, d1, d2)
        };
        let mut t = 0.5;
        for _ in 0..50 {
            let (pt, d1, d2) = curve(t);
            let r = sub(p, pt);
            let g = dot(r, d1);
            let h = dot(d1, d1) - dot(r, d2);
            if h.abs() < 1e-300 {
                break;
            }
            let dt = g / h;
            t = (t + dt).clamp(0.0, 1.0);
            if dt.abs() < 1e-14 {
                break;
            }
        }
        let (_, d1, _) = curve(t);
        let l = norm(d1);
        let mut n = [d1[1] / l, -d1[0] / l, 0.0];
        if dot(n, n_facet) < 0.0 {
            n = scale(n, -1.0);
        }
        let wa = (1.0 - t) * (1.0 - t) * s;
        let wc = t * t * s;
        Some((vec![(a, wa), (b, 1.0 - wa - wc), (c, wc)], n))
    }
}

impl Interaction for ContactPair {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn connectivity(&self) -> Vec<Vec<usize>> {
        match &self.fixed {
            Some(f) => f.iter().flatten().map(|t| t.coef.iter().map(|c| c.0).collect()).collect(),
            None => {
                // slavnoden, slavnoderna i dess ytelement (utjämnad viktning) och alla masternoder
                let mut patch: Vec<Vec<usize>> = self.nodes.iter().map(|&n| vec![n]).collect();
                if let SlaveSide::Facets(fs) = &self.slave {
                    for f in fs {
                        for &a in &f.nodes {
                            let i = self.nodes.binary_search(&a).unwrap();
                            patch[i].extend(f.nodes.iter().copied().filter(|&b| b != a));
                        }
                    }
                }
                patch
                    .into_iter()
                    .map(|mut v| {
                        v.sort_unstable();
                        v.dedup();
                        // självkontakt: masternoderna är också slavnoder
                        v.extend(self.master_nodes.iter().copied().filter(|n| self.self_contact || self.nodes.binary_search(n).is_err()));
                        v
                    })
                    .collect()
            }
        }
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], mut k: Option<&mut dyn TangentSink>) -> Result<()> {
        let m = ctx.model;
        let x: Vec<[f64; 3]> = m.nodes.iter().zip(u).map(|(p, d)| [p[0] + d[0], p[1] + d[1], p[2] + d[2]]).collect();
        let computed;
        let terms: &[Option<NodeTerm>] = match &self.fixed {
            Some(f) => f,
            None => {
                computed = self.terms(m, &x);
                &computed
            }
        };
        if crate::settings::with(|s| s.contact_debug) {
            for (i, t) in terms.iter().enumerate() {
                if let Some(t) = t {
                    eprintln!("kontakt {} nod {}: n={:?} A={:.4e} c={:?}", self.name, m.node_name(self.nodes[i]), t.normal, t.area, t.coef.iter().map(|(n, c)| (m.node_label[*n], (c * 1e4).round() / 1e4)).collect::<Vec<_>>());
                }
            }
        }
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        let dim = self.dim;
        for (i, t) in terms.iter().enumerate() {
            let old = st.committed[i];
            // glidhastigheten från föregående iteration (EXPONENTIAL DECAY, implicit)
            let rate_it = st.trial[i].rate;
            let mut ns = NodeState { total_slip: old.total_slip, ..Default::default() };
            let Some(t) = t else {
                st.trial[i] = ns;
                st.smooth_trial[i].clear();
                continue;
            };
            st.smooth_trial[i].clone_from(&t.smooth);
            let n = t.normal;
            // gap och relativ förskjutning sedan inkrementets början
            let mut r = [0.0; 3];
            let mut dr = [0.0; 3];
            for &(b, c) in &t.coef {
                for d in 0..dim {
                    r[d] += c * x[b][d];
                    dr[d] += c * (u[b][d] - st.u_committed[b][d]);
                }
            }
            let g = self.gap_offset[i] + if t.vcoef.is_empty() { dot(n, r) } else { t.vcoef.iter().map(|(b, v)| dot(*v, x[*b])).sum::<f64>() };
            if crate::settings::with(|s| s.contact_debug) {
                eprintln!("  gap {}: g={g:.6e}", m.node_name(self.nodes[i]));
            }
            ns.gap = g;
            let h = -g;
            let kh = self.k_hard[i];
            let (mut p, mut dp) = self.props.law.pressure(h, kh);
            // NO SEPARATION: slutna kontakter öppnar inte (dragkraft tillåts)
            let bonded = self.props.no_sep && old.contact && h < 0.0;
            if bonded {
                let k = match self.props.law {
                    Law::Linear(k) => k,
                    _ => kh,
                };
                p = k * h;
                dp = k;
            }
            // initialt öppet spel (*CLEARANCE): normalstyvheten tas med i tangenten
            // tills spelet öppnar mer än det givna, så att en kropp som bara hålls
            // av kontakten inte blir en mekanism i första iterationen (påverkar
            // bara iterationsvägen, inte jämvikten)
            let pre_open = self.initial_gap[i] > 0.0 && !old.contact && g > 0.0 && g <= 2.0 * self.initial_gap[i];
            if pre_open {
                dp = match self.props.law {
                    Law::Linear(k) => k,
                    _ => kh,
                };
            }
            let area = t.area * self.weight;
            ns.p = p;
            ns.contact = self.props.law.active(h) || bonded;
            // friktion
            let mut tau = [0.0; 3];
            let mut kt: Option<[[f64; 3]; 3]> = None;
            // ∂τ/∂p (för den osymmetriska friktionstangenten dτ/dp · dp/dh)
            let mut dtau_dp = [0.0; 3];
            // osymmetrisk del av friktionstangenten (anisotrop friktion, contact_ext.rs)
            let mut kt_skew: Option<[[f64; 3]; 3]> = None;
            ns.normal = n;
            let dn = dot(dr, n);
            let ds = [dr[0] - dn * n[0], dr[1] - dn * n[1], dr[2] - dn * n[2]];
            let fric = (self.props.mu > 0.0 || self.props.rough) && ns.contact && (p > 0.0 || (self.props.rough && bonded));
            if ns.contact && !fric {
                // friktionsfritt: all tangentiell relativ rörelse är glidning
                for c in 0..3 {
                    ns.total_slip[c] += ds[c];
                }
            }
            if fric {
                let eo = old.slip;
                let eon = dot(eo, n);
                let e_tr = [eo[0] - eon * n[0] + ds[0], eo[1] - eon * n[1] + ds[1], eo[2] - eon * n[2] + ds[2]];
                let e_len = norm(e_tr);
                // friktionskoefficient (EXPONENTIAL DECAY: från glidhastigheten i inkrementet)
                let mu = match self.props.decay {
                    Some((mk, dc)) => mk + (self.props.mu - mk) * (-dc * rate_it).exp(),
                    _ => self.props.mu,
                };
                // ROUGH: ingen glidgräns; LAGRANGE: fastläget med den styva straffkoefficienten
                let mu_p = if self.props.rough { f64::INFINITY } else { mu * p };
                let gam = if self.props.rough {
                    f64::INFINITY
                } else if self.props.lagrange {
                    // fastläget nästan exakt: 1 % av standardens elastiska glidning
                    (0.01 * self.gamma).max(1e-300)
                } else {
                    self.props.elastic_slip.unwrap_or(self.gamma).max(1e-300)
                };
                let ks = if self.props.rough { kh * LAGRANGE_STICK } else { mu_p / gam };
                let mut pt = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        pt[a][b] = (if a == b { 1.0 } else { 0.0 }) - n[a] * n[b];
                    }
                }
                if dim == 2 {
                    pt[2] = [0.0; 3];
                    pt[0][2] = 0.0;
                    pt[1][2] = 0.0;
                }
                if let (Some(mu2), 3, false) = (self.props.mu2, dim, self.props.rough) {
                    // område contact3: anisotrop friktion (contact_ext.rs)
                    let a = ext::aniso_friction(e_tr, n, p, mu, mu2, gam);
                    tau = a.tau;
                    dtau_dp = a.dtau_dp;
                    ns.slip = a.slip;
                    let (ks, kw) = ext::split(&a.kt);
                    kt = Some(ks);
                    kt_skew = Some(kw);
                } else if e_len <= gam {
                    tau = scale(e_tr, ks);
                    if !self.props.rough {
                        dtau_dp = scale(e_tr, mu / gam);
                    }
                    ns.slip = e_tr;
                    kt = Some(pt.map(|row| row.map(|v| v * ks)));
                } else {
                    let eh = scale(e_tr, 1.0 / e_len);
                    tau = scale(eh, mu_p);
                    dtau_dp = scale(eh, mu);
                    ns.slip = scale(eh, gam);
                    let f = mu_p / e_len;
                    let mut kk = [[0.0; 3]; 3];
                    for a in 0..3 {
                        for b in 0..3 {
                            kk[a][b] = f * (pt[a][b] - eh[a] * eh[b]);
                        }
                    }
                    kt = Some(kk);
                }
                // total glidning: tangentiell relativ rörelse minus ändringen av den elastiska delen
                for c in 0..3 {
                    ns.total_slip[c] = old.total_slip[c] + ds[c] - (ns.slip[c] - (eo[c] - eon * n[c]));
                }
                if ctx.dt > 0.0 {
                    ns.rate = norm(sub(ns.total_slip, old.total_slip)) / ctx.dt;
                }
                ns.shear = tau;
            }
            st.trial[i] = ns;
            // område acoustics2: viskös kontaktdämpning μ A v (normal och s·tangentiell)
            if let (Some(dm), true) = (self.props.damp, ctx.dt > 0.0) {
                // koefficienten från det aktuella spelet (kontrollerat;
                // spelet vid inkrementets början ger 6 % fel)
                let mu = dm.coefficient(g);
                if mu > 0.0 {
                    let tf = dm.tangent.unwrap_or(0.0);
                    let cv = mu / ctx.dt;
                    let trd: [f64; 3] = std::array::from_fn(|c| cv * (dn * n[c] + tf * ds[c]));
                    for &(b, c) in &t.coef {
                        for d in 0..dim {
                            fint[b][d] += area * c * trd[d];
                        }
                    }
                    if let Some(k) = k.as_deref_mut() {
                        for &(b1, c1) in &t.coef {
                            for &(b2, c2) in &t.coef {
                                let f = area * c1 * c2 * cv;
                                for p1 in 0..dim {
                                    for p2 in 0..dim {
                                        let pt = (if p1 == p2 { 1.0 } else { 0.0 }) - n[p1] * n[p2];
                                        let v = f * (n[p1] * n[p2] + tf * pt);
                                        if v != 0.0 {
                                            k.add(b1, U1 + p1 as Dof, b2, U1 + p2 as Dof, v);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if !ns.contact && p == 0.0 && !pre_open {
                continue;
            }
            // krafter: f_B = A c_B (−p n + τ), med vektorkoefficienter f_B = A (−p v_B + c_B τ)
            let vec_n = !t.vcoef.is_empty();
            let tr = if vec_n { tau } else { [-p * n[0] + tau[0], -p * n[1] + tau[1], -p * n[2] + tau[2]] };
            for &(b, c) in &t.coef {
                for d in 0..dim {
                    fint[b][d] += area * c * tr[d];
                }
            }
            for (b, v) in &t.vcoef {
                for d in 0..dim {
                    fint[*b][d] -= area * p * v[d];
                }
            }
            if let Some(k) = k.as_deref_mut() {
                if vec_n && dp != 0.0 {
                    for (b1, v1) in &t.vcoef {
                        for (b2, v2) in &t.vcoef {
                            for p1 in 0..dim {
                                for p2 in 0..dim {
                                    let v = area * dp * v1[p1] * v2[p2];
                                    if v != 0.0 {
                                        k.add(*b1, U1 + p1 as Dof, *b2, U1 + p2 as Dof, v);
                                    }
                                }
                            }
                        }
                    }
                }
                let mut kk = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        kk[a][b] = if vec_n { 0.0 } else { dp * n[a] * n[b] } + kt.map_or(0.0, |m| m[a][b]);
                    }
                }
                for &(b1, c1) in &t.coef {
                    for &(b2, c2) in &t.coef {
                        let f = area * c1 * c2;
                        for p1 in 0..dim {
                            for p2 in 0..dim {
                                let v = f * kk[p1][p2];
                                if v != 0.0 {
                                    k.add(b1, U1 + p1 as Dof, b2, U1 + p2 as Dof, v);
                                }
                            }
                        }
                    }
                }
                // osymmetrisk lösare: friktionskraftens beroende av trycket,
                // ∂f_B1/∂x_B2 = A c₁ c₂ (∂τ/∂p) p' (−n)ᵀ (utelämnas i den symmetriska tangenten)
                if let (Some(sk), true) = (kt_skew, k.is_unsymmetric()) {
                    ext::add_skew(k, &t.coef, area, &sk, dim);
                }
                if k.is_unsymmetric() && !vec_n && dp != 0.0 && dtau_dp.iter().any(|v| *v != 0.0) {
                    for &(b1, c1) in &t.coef {
                        for &(b2, c2) in &t.coef {
                            let f = -area * c1 * c2 * dp;
                            for p1 in 0..dim {
                                for p2 in 0..dim {
                                    let v = f * dtau_dp[p1] * n[p2];
                                    if v != 0.0 {
                                        k.add_unsym(b1, U1 + p1 as Dof, b2, U1 + p2 as Dof, v);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        st.u_last = u.to_vec();
        Ok(())
    }

    fn commit(&self) {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        st.committed.clone_from(&st.trial);
        st.u_committed.clone_from(&st.u_last);
        st.smooth_committed.clone_from(&st.smooth_trial);
    }

    fn node_output(&self) -> Vec<NodeOutput> {
        let st = self.state.lock().unwrap();
        let mut cp = NodeOutput { name: "CPRESS".into(), labels: vec!["CPRESS".into()], values: Vec::new() };
        let mut co = NodeOutput { name: "COPEN".into(), labels: vec!["COPEN".into()], values: Vec::new() };
        let mut out = Vec::new();
        let ncomp = self.dim - 1;
        let mut sh: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSHEAR{k}"), labels: vec![format!("CSHEAR{k}")], values: Vec::new() }).collect();
        let mut sl: Vec<NodeOutput> = (1..=ncomp).map(|k| NodeOutput { name: format!("CSLIP{k}"), labels: vec![format!("CSLIP{k}")], values: Vec::new() }).collect();
        for (i, &n) in self.nodes.iter().enumerate() {
            let s = st.committed[i];
            let sm = &st.smooth_committed[i];
            let p = if sm.is_empty() { s.p } else { sm.iter().map(|&(j, w)| w * st.committed[j].p).sum() };
            cp.values.push((n, vec![p]));
            co.values.push((n, vec![s.gap]));
            let t = local_tangents(s.normal, self.dim);
            for k in 0..ncomp {
                // CSHEAR har samma tecken som den elastiska glidningen
                sh[k].values.push((n, vec![dot(s.shear, t[k])]));
                sl[k].values.push((n, vec![dot(s.total_slip, t[k])]));
            }
        }
        out.push(cp);
        out.push(co);
        out.extend(sh);
        out.extend(sl);
        out
    }
}

/// Lokala tangentriktningar för kontaktutdata (CSHEAR/CSLIP):
/// i 2D t₁ = (n_y, −n_x); i 3D t₁ = globala x projicerad på tangentplanet
/// (globala z om normalen är nästan parallell med x) och t₂ = n × t₁.
fn local_tangents(n: [f64; 3], dim: usize) -> [[f64; 3]; 2] {
    if norm(n) == 0.0 {
        return [[0.0; 3]; 2];
    }
    if dim == 2 {
        return [[n[1], -n[0], 0.0], [0.0; 3]];
    }
    let e = if n[0].abs() > (0.1f64.to_radians()).cos() { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let d = dot(e, n);
    let t1 = [e[0] - d * n[0], e[1] - d * n[1], e[2] - d * n[2]];
    let t1 = scale(t1, 1.0 / norm(t1));
    let t2 = [n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]];
    [t1, t2]
}

fn pair(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    // bara explicita steg: kontakten sköts av steps/explicit_contact.rs
    if crate::steps::explicit::contact::only_explicit(m) {
        return Ok(Vec::new());
    }
    // område multiphysics: ytor på värmeledningselement har bara termisk kontakt (thermal_contact.rs)
    if super::thermal_contact::only_heat(m, kw) {
        return Ok(Vec::new());
    }
    let ip = crate::inp::required(kw, "INTERACTION")?;
    // område cavity: kohesiv kontakt (*COHESIVE BEHAVIOR) i interactions/cohesive_contact.rs
    if super::cohesive_contact::is_cohesive(m, ip) {
        return Ok(Vec::new());
    }
    let p = props(m, kw, ip)?;
    // nod mot yta är standard för *CONTACT PAIR
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "NODE TO SURFACE".into());
    let s2s = match ty.as_str() {
        "SURFACE TO SURFACE" => true,
        "NODE TO SURFACE" => false,
        other => return kw.err(format!("TYPE={other} stöds inte (NODE TO SURFACE, SURFACE TO SURFACE)")),
    };
    // ADJUST= justerar slavnoderna när filen läses (`adjust_slaves`)
    let small = kw.has("SMALL SLIDING");
    let mut out: Vec<Box<dyn Interaction>> = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 2 {
            return kw.err_at(d.line, "skriv slavyta, masteryta");
        }
        // analytisk stel masteryta: interactions/analytical.rs
        if super::analytical::is_analytical(m, f[1]) {
            continue;
        }
        let name = format!("*CONTACT PAIR {}/{} ({})", f[0], f[1], kw.location());
        let mut cp =ContactPair::new(m, kw, name, f[0], f[1], p, s2s, small, 1.0)?;
        if let Some(v) = kw.param("SMOOTH").and_then(parse_f64) {
            cp.smooth = v.clamp(0.0, 0.5);
        }
        cp.apply_clearance(m, f[0], f[1])?;
        out.push(Box::new(cp));
    }
    Ok(out)
}

// ---------------------------------------------------------------- general contact

/// Yttre elementytor (ytor som bara ett element har) per kropp (sammanhängande nät).
fn exterior(m: &FeModel) -> Vec<(usize, Vec<(usize, String)>)> {
    let mut count: BTreeMap<Vec<usize>, Vec<(usize, usize)>> = BTreeMap::new();
    for (e, el) in m.elements.iter().enumerate() {
        use crate::elements::Family;
        if !matches!(el.etype.family(), Family::Continuum2D | Family::Continuum3D) {
            continue;
        }
        for (fi, f) in el.etype.faces().iter().enumerate() {
            let mut key: Vec<usize> = f.iter().map(|&k| el.nodes[k]).collect();
            key.sort_unstable();
            count.entry(key).or_default().push((e, fi));
        }
    }
    // kroppar: union-find över noder via element
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
    let mut bodies: BTreeMap<usize, Vec<(usize, String)>> = BTreeMap::new();
    for (_, v) in count {
        if v.len() == 1 {
            let (e, fi) = v[0];
            let b = find(&mut parent, m.elements[e].nodes[0]);
            bodies.entry(b).or_default().push((e, format!("S{}", fi + 1)));
        }
    }
    bodies.into_iter().collect()
}

fn general(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    // bara explicita steg: kontakten sköts av steps/explicit_contact.rs
    if crate::steps::explicit::contact::only_explicit(m) {
        return Ok(Vec::new());
    }
    let subs = sub_options(m, kw, &["CONTACT INCLUSIONS", "CONTACT PROPERTY ASSIGNMENT", "CONTACT EXCLUSIONS", "CONTACT FORMULATION", "SURFACE PROPERTY ASSIGNMENT", "CONTACT INITIALIZATION ASSIGNMENT", "CONTACT CONTROLS ASSIGNMENT"]);
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut all = false;
    let mut ip: Option<String> = None;
    // egenskaper per ytpar: (yta, yta, interaktion)
    let mut per_pair: Vec<(String, String, String)> = Vec::new();
    for s in &subs {
        match s.name.as_str() {
            "CONTACT INCLUSIONS" => {
                if s.has("ALL EXTERIOR") {
                    all = true;
                }
                for d in &s.data {
                    let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                    if f.len() >= 2 && !f[0].is_empty() && !f[1].is_empty() {
                        pairs.push((f[0].to_string(), f[1].to_string()));
                    } else if f.first().is_some_and(|x| !x.is_empty()) {
                        // område contact3: självkontakt (en yta)
                        pairs.push((f[0].to_string(), f[0].to_string()));
                    }
                }
            }
            "CONTACT PROPERTY ASSIGNMENT" => {
                for d in &s.data {
                    let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                    if f.len() >= 3 {
                        if f[0].is_empty() && f[1].is_empty() {
                            ip = Some(f[2].to_string());
                        } else if f[0].is_empty() || f[1].is_empty() {
                            return s.err_at(d.line, "egenskaper för en enda yta stöds inte ännu (skriv två ytor eller ' , , INTERAKTION')");
                        } else {
                            per_pair.push((f[0].to_uppercase(), f[1].to_uppercase(), f[2].to_string()));
                        }
                    }
                }
            }
            // *CONTACT INITIALIZATION ASSIGNMENT justerar noderna när filen läses (`adjust_slaves`)
            "CONTACT EXCLUSIONS" | "SURFACE PROPERTY ASSIGNMENT" => {
                return s.err("stöds inte ännu");
            }
            _ => {}
        }
    }
    let p = match &ip {
        Some(n) => props(m, kw, n)?,
        None => Props::base(),
    };
    let mut out: Vec<Box<dyn Interaction>> = Vec::new();
    let mut tmp = m.clone();
    if all {
        let bodies = exterior(m);
        for (bi, (_, faces)) in bodies.iter().enumerate() {
            let name = format!("_GC_KROPP{bi}");
            tmp.surfaces.insert(name.clone(), crate::fem::model::Surface { kind: "ELEMENT".into(), faces: faces.clone(), nodes: Vec::new(), source: kw.location() });
        }
        for a in 0..bodies.len() {
            for b in a + 1..bodies.len() {
                pairs.push((format!("_GC_KROPP{a}"), format!("_GC_KROPP{b}")));
            }
        }
        // område contact3: ALL EXTERIOR omfattar självkontakt för varje kropp
        // (bara om kroppen kan komma i kontakt med sig själv, `ext::can_self_contact`)
        for (a, (_, faces)) in bodies.iter().enumerate() {
            if ext::can_self_contact(m, faces) {
                pairs.push((format!("_GC_KROPP{a}"), format!("_GC_KROPP{a}")));
            }
        }
    }
    if pairs.is_empty() {
        return fail(format!("{}: *CONTACT utan kontaktytor (*CONTACT INCLUSIONS)", kw.location()));
    }
    for (a, b) in pairs {
        if a.eq_ignore_ascii_case(&b) {
            out.push(Box::new(ext::self_pair(&tmp, kw, &a, &per_pair, p)?));
            continue;
        }
        // ren master–slav per ytpar: ytan med finast nät
        // (minst medelstorlek på ytelementen) blir slav; vid lika storlek den
        // yta vars element kommer först.
        let (sa, ea) = surface_size(&tmp, kw, &a)?;
        let (sb, eb) = surface_size(&tmp, kw, &b)?;
        let a_slave = if (sa - sb).abs() <= 1e-6 * sa.max(sb) { ea <= eb } else { sa < sb };
        let (s, mm) = if a_slave { (&a, &b) } else { (&b, &a) };
        let name = format!("*CONTACT {s}/{mm} ({})", kw.location());
        // *CONTACT PROPERTY ASSIGNMENT för just det här ytparet (i valfri ordning)
        let (ua, ub) = (a.to_uppercase(), b.to_uppercase());
        let pp = match per_pair.iter().find(|(x, y, _)| (*x == ua && *y == ub) || (*x == ub && *y == ua)) {
            Some((_, _, n)) => props(m, kw, n)?,
            None => p,
        };
        out.push(Box::new(ContactPair::new(&tmp, kw, name, s, mm, pp, true, false, 1.0)?));
    }
    Ok(out)
}

/// Medelstorlek för ytans ytelement och dess första element (för valet av slavyta).
fn surface_size(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<(f64, usize)> {
    let fs = geom::facets(m, geom::surface(m, kw, name)?, name)?;
    let mut s = 0.0;
    for f in &fs {
        s += facet_size(f, &geom::coords(m, f, None));
    }
    let first = fs.iter().map(|f| f.elem).min().unwrap_or(usize::MAX);
    Ok((s / fs.len().max(1) as f64, first))
}

// ------------------------------------------------ för analytiska stela ytor

/// Kontaktegenskaperna från *SURFACE INTERACTION för `interactions/analytical.rs`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PairProps(Props);

impl PairProps {
    pub(crate) fn read(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<PairProps> {
        props(m, kw, name).map(PairProps)
    }
    /// (p, dp/dh) för överlappningen h med straffkoefficienten `k_hard` (HARD).
    pub(crate) fn pressure(&self, h: f64, k_hard: f64) -> (f64, f64) {
        self.0.law.pressure(h, k_hard)
    }
    pub(crate) fn active(&self, h: f64) -> bool {
        self.0.law.active(h)
    }
    pub(crate) fn mu(&self) -> f64 {
        self.0.mu
    }
    pub(crate) fn slip_tol(&self) -> f64 {
        self.0.slip_tol
    }
    /// Hård kontakt (PRESSURE-OVERCLOSURE=HARD, direkt eller med straff).
    pub(crate) fn is_hard(&self) -> bool {
        matches!(self.0.law, Law::Hard)
    }
    /// Hård kontakt med straffmetod: Some((given styvhet eller None = standardstyvhet, skalfaktor)).
    pub(crate) fn penalty(&self) -> Option<(Option<f64>, f64)> {
        match self.0.enforce {
            Some(Enforce::Penalty(k, s)) => Some((k, s)),
            _ => None,
        }
    }
}

/// Representativ translationsstyvhet för ett element (som för kontaktpar).
pub(crate) fn representative_stiffness(m: &FeModel, e: usize) -> Result<f64> {
    element_stiffness(m, e)
}

/// Standardstyvhet för straffmetoden vid en elementyta.
pub(crate) fn default_penalty(m: &FeModel, f: &Facet) -> Result<f64> {
    penalty_stiffness(m, f)
}

/// Ytelementets storlek (för den kritiska elastiska glidningen).
pub(crate) fn facet_length(f: &Facet, x: &[[f64; 3]]) -> f64 {
    facet_size(f, x)
}

/// Lokala tangentriktningar för CSHEAR/CSLIP.
pub(crate) fn output_tangents(n: [f64; 3], dim: usize) -> [[f64; 3]; 2] {
    local_tangents(n, dim)
}

impl ContactPair {
    /// Gör paret till självkontakt (slav = master, `contact_ext.rs`).
    fn make_self(&mut self, m: &FeModel) {
        self.self_contact = true;
        self.master_elem_nodes = self.master.iter().map(|f| m.elements[f.elem].nodes.clone()).collect();
    }
}

/// Töjningsfri justering av slavnoder (*CONTACT PAIR, ADJUST= och
/// *CONTACT INITIALIZATION DATA), anropas när filen är läst (`contact_ext.rs`).
pub fn adjust_slaves(m: &mut FeModel) -> Result<()> {
    ext::adjust(m)
}
