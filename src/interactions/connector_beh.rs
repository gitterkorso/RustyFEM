//! Kopplingselementens beteenden (inkluderas med `#[path]` från `connector.rs`):
//! *CONNECTOR ELASTICITY (linjär, matris, NONLINEAR, RIGID), *CONNECTOR DAMPING,
//! *CONNECTOR PLASTICITY + *CONNECTOR HARDENING, *CONNECTOR DAMAGE INITIATION/
//! EVOLUTION, *CONNECTOR STOP, *CONNECTOR LOCK och *CONNECTOR FRICTION.
//!
//! Alla beteenden är okopplade per komponent (utom elasticitetsmatrisen) och
//! endimensionella, med följande semantik:
//! - NONLINEAR: tabell (kraft, förskjutning), konstant utanför tabellen
//!   (EXTRAPOLATION=CONSTANT), elastisk (samma kurva vid avlastning).
//! - Plasticitet: F = k (u − u_p), flytvillkor |F − α| ≤ R(ū_p). Isotrop
//!   hårdnande R tabulerat (kraft, plastisk rörelse; konstant efter sista
//!   punkten) eller EXPONENTIAL LAW R = F₀ + Q∞ (1 − e^(−b ū_p)); kinematiskt
//!   (TYPE=KINEMATIC, DEFINITION=PARAMETERS: F₀, C, γ) dα = C du_p − γ α dū_p.
//!   Implicit returalgoritm (Simo & Hughes kap. 1) med konsistent tangent.
//! - Skada: initiering när kraften (CRITERION=FORCE, standard), rörelsen
//!   (MOTION) eller den plastiska rörelsen (PLASTIC MOTION) når gränserna;
//!   TYPE=MOTION, SOFTENING=LINEAR: d = u_f (u − u₀)/(u (u_f − u₀)) med u
//!   räknad i den riktning där skadan initierades (rörelse åt andra hållet
//!   ger ingen ny skada), u_f = u₀ + Δu_f, oåterkalleligt; kraften blir
//!   (1 − d)·F.
//! - Stopp och lås gäller lägen (CP = läget i referensläget + CU), straff.
//! - Friktion: F_f = μ N med N = den inre kontaktkraften på dataraden,
//!   klistring med STICK STIFFNESS (standard: elastisk glidning 1e-4 · längden).
//! - RIGID: elastisk styvhet 1e6.
//! - Dämpning: viskös F = c v i *DYNAMIC, v = Δu/Δt över inkrementet.

use crate::interactions::constraints::sub_options;
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::Result;

#[derive(Debug, Clone, Default)]
pub enum Elastic {
    #[default]
    None,
    Linear(f64),
    /// (förskjutning, kraft), sorterad efter förskjutning.
    Table(Vec<(f64, f64)>),
}

impl Elastic {
    /// (kraft, iterationsstyvhet) vid förskjutningen u; `u_prev` är
    /// föregående utvärdering. För den styckvis linjära kurvan pendlar Newton
    /// med tangenten mellan segmenten; sekanten från föregående iteration
    /// konvergerar och blir exakt när två iterationer ligger i samma segment.
    pub fn eval(&self, u: f64, u_prev: f64) -> (f64, f64) {
        match self {
            Elastic::None => (0.0, 0.0),
            Elastic::Linear(k) => (k * u, *k),
            Elastic::Table(t) => {
                let (f, s) = interp(t, u);
                let du = u - u_prev;
                let scale = t.iter().map(|p| p.0.abs()).fold(0.0, f64::max).max(1e-300);
                let k = if du.abs() > 1e-12 * scale {
                    let (fp, _) = interp(t, u_prev);
                    (f - fp) / du
                } else {
                    let h = 1e-9 * scale;
                    if s == 0.0 {
                        interp(t, u + h).1.max(interp(t, u - h).1)
                    } else {
                        s
                    }
                };
                (f, k)
            }
        }
    }

    /// Representativ styvhet (för straffkoefficienten).
    pub fn scale(&self) -> f64 {
        match self {
            Elastic::None => 0.0,
            Elastic::Linear(k) => k.abs(),
            Elastic::Table(t) => t.windows(2).map(|w| ((w[1].1 - w[0].1) / (w[1].0 - w[0].0)).abs()).fold(0.0, f64::max),
        }
    }
}

/// Styckvis linjär interpolation, konstant utanför: (värde, lutning).
fn interp(t: &[(f64, f64)], x: f64) -> (f64, f64) {
    if t.is_empty() {
        return (0.0, 0.0);
    }
    if x <= t[0].0 {
        return (t[0].1, 0.0);
    }
    for w in t.windows(2) {
        if x <= w[1].0 {
            let s = (w[1].1 - w[0].1) / (w[1].0 - w[0].0);
            return (w[0].1 + s * (x - w[0].0), s);
        }
    }
    (t[t.len() - 1].1, 0.0)
}

#[derive(Debug, Clone)]
pub enum Iso {
    /// (plastisk rörelse, flytkraft)
    Table(Vec<(f64, f64)>),
    Exp { f0: f64, q: f64, b: f64 },
}

impl Iso {
    fn eval(&self, ub: f64) -> (f64, f64) {
        match self {
            Iso::Table(t) => interp(t, ub),
            Iso::Exp { f0, q, b } => (f0 + q * (1.0 - (-b * ub).exp()), q * b * (-b * ub).exp()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Plastic {
    pub iso: Iso,
    /// Kinematiskt hårdnande (C, γ).
    pub kin: Option<(f64, f64)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Criterion {
    Force,
    Motion,
    PlasticMotion,
}

#[derive(Debug, Clone)]
pub struct Damage {
    pub criterion: Criterion,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    /// Δu_f (TYPE=MOTION, SOFTENING=LINEAR); None = bara initiering.
    pub du_f: Option<f64>,
    pub d_max: f64,
}

#[derive(Debug, Clone)]
pub struct Friction {
    pub f0: f64,
    pub mu: f64,
    pub stick: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct Lock {
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    /// Komponenterna (0..6) som låses.
    pub comps: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct CompBeh {
    pub elastic: Elastic,
    pub plastic: Option<Plastic>,
    pub damage: Option<Damage>,
    pub friction: Option<Friction>,
    pub stop: Option<(Option<f64>, Option<f64>)>,
    pub lock: Option<Lock>,
    pub damping: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Behavior {
    pub comps: [CompBeh; 6],
    /// Elasticitetsmatris över de tillgängliga komponenterna (6×6, radvis).
    pub matrix: Option<[[f64; 6]; 6]>,
    /// Komponenter som är stela (*CONNECTOR ELASTICITY, RIGID).
    pub rigid: [bool; 6],
}

impl Behavior {
    pub fn is_linear(&self) -> bool {
        self.comps.iter().all(|c| matches!(c.elastic, Elastic::None | Elastic::Linear(_)) && c.plastic.is_none() && c.damage.as_ref().is_none_or(|d| d.du_f.is_none()) && c.friction.is_none() && c.stop.is_none() && c.lock.is_none())
    }
}

/// Styvheten för *CONNECTOR ELASTICITY, RIGID.
pub const RIGID_STIFFNESS: f64 = 1e6;

const NAMES: &[&str] = &[
    "CONNECTOR ELASTICITY",
    "CONNECTOR DAMPING",
    "CONNECTOR PLASTICITY",
    "CONNECTOR HARDENING",
    "CONNECTOR DAMAGE INITIATION",
    "CONNECTOR DAMAGE EVOLUTION",
    "CONNECTOR FRICTION",
    "CONNECTOR POTENTIAL",
    "CONNECTOR DERIVED COMPONENT",
    "FRICTION",
    "CONNECTOR STOP",
    "CONNECTOR LOCK",
    "CONNECTOR CONSTITUTIVE REFERENCE",
];

/// Fält på första dataraden (tomma fält = None).
fn opt_fields(kw: &RawKeyword) -> Result<Vec<Option<f64>>> {
    let Some(d) = kw.data.first() else { return Ok(Vec::new()) };
    d.fields
        .iter()
        .map(|s| {
            let s = s.trim();
            if s.is_empty() {
                Ok(None)
            } else {
                parse_f64(s).map(Some).ok_or(()).or_else(|_| kw.err(format!("'{s}' är inget tal")))
            }
        })
        .collect()
}

fn table(kw: &RawKeyword, swap: bool) -> Result<Vec<(f64, f64)>> {
    let mut t = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() < 2 {
            return kw.err_at(d.line, "skriv två värden per rad");
        }
        t.push(if swap { (v[1], v[0]) } else { (v[0], v[1]) });
    }
    t.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(t)
}

fn component(kw: &RawKeyword, dim: usize) -> Result<usize> {
    match kw.param("COMPONENT").and_then(crate::inp::parse_label) {
        Some(c @ 1..=6) => {
            let c = c as usize - 1;
            if dim == 2 && !matches!(c, 0 | 1 | 5) {
                return kw.err("i 2D finns komponenterna 1, 2 och 6");
            }
            Ok(c)
        }
        _ => kw.err("COMPONENT=1..6 krävs"),
    }
}

/// Läser beteendet *CONNECTOR BEHAVIOR, NAME=`name`. `avail` = tillgängliga komponenter.
pub fn parse(m: &FeModel, kw: &RawKeyword, name: &str, avail: &[usize]) -> Result<Behavior> {
    let up = name.trim().to_uppercase();
    let Some(b) = m.extra.iter().find(|k| k.name == "CONNECTOR BEHAVIOR" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*CONNECTOR BEHAVIOR, NAME={name} finns inte"));
    };
    let dim = m.dim();
    let mut beh = Behavior::default();
    let subs = sub_options(m, b, NAMES);
    let mut last_plastic: Option<usize> = None;
    let mut last_damage: Option<usize> = None;
    let mut last_friction: Option<usize> = None;
    let mut pending: Option<Friction> = None;
    for s in subs {
        if s.has("DEPENDENCIES") || s.has("FREQUENCY DEPENDENCE") {
            return s.err("DEPENDENCIES stöds inte ännu för kopplingsbeteenden");
        }
        match s.name.as_str() {
            "CONNECTOR ELASTICITY" => {
                if s.has("RIGID") {
                    let comps: Vec<usize> = s.data.iter().flat_map(|d| d.fields.iter()).filter(|f| !f.trim().is_empty()).map(|f| crate::inp::parse_label(f.trim()).map(|c| c as usize)).collect::<Option<Vec<_>>>().ok_or(()).or_else(|_| s.err("komponentnummer förväntades"))?;
                    let list: Vec<usize> = if let Ok(c) = component(s, dim) { vec![c] } else if comps.is_empty() { avail.to_vec() } else { comps.into_iter().map(|c| c - 1).collect() };
                    for c in list {
                        if c >= 6 || !avail.contains(&c) {
                            return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                        }
                        beh.rigid[c] = true;
                    }
                    continue;
                }
                if !s.has("COMPONENT") {
                    if s.has("NONLINEAR") {
                        return s.err("NONLINEAR kräver COMPONENT=");
                    }
                    // matrisform: D11, D12, D22, D13, D23, D33, ... över de tillgängliga komponenterna
                    let v: Vec<f64> = s.records().iter().map(|d| s.floats(d)).collect::<Result<Vec<_>>>()?.concat();
                    // 21 värden: hela 6×6-matrisen över komponenterna 1–6 (som
                    // formatet kräver), annars bara de tillgängliga komponenterna
                    let full: Vec<usize> = (0..6).collect();
                    let avail0 = avail;
                    let avail: &[usize] = if v.len() >= 21 { &full } else { avail };
                    let n = avail.len();
                    if v.len() < n * (n + 1) / 2 {
                        return s.err(format!("elasticitetsmatrisen kräver {} värden för {n} tillgängliga komponenter", n * (n + 1) / 2));
                    }
                    let mut mat = [[0.0; 6]; 6];
                    let mut q = 0;
                    for j in 0..n {
                        for i in 0..=j {
                            mat[avail[i]][avail[j]] = v[q];
                            mat[avail[j]][avail[i]] = v[q];
                            q += 1;
                        }
                    }
                    // komponenter som inte är tillgängliga ignoreras
                    for c in 0..6 {
                        if !avail0.contains(&c) {
                            for d in 0..6 {
                                mat[c][d] = 0.0;
                                mat[d][c] = 0.0;
                            }
                        }
                    }
                    beh.matrix = Some(mat);
                    continue;
                }
                let c = component(s, dim)?;
                if !avail.contains(&c) {
                    return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                }
                if s.has("NONLINEAR") {
                    // dataraderna: kraft, förskjutning
                    beh.comps[c].elastic = Elastic::Table(table(s, true)?);
                } else {
                    if s.data.len() > 1 {
                        return s.err("temperaturberoende elasticitet stöds inte ännu (en rad med styvheten)");
                    }
                    let v = s.first_floats()?;
                    let Some(&k) = v.first() else { return s.err("styvhet saknas") };
                    beh.comps[c].elastic = Elastic::Linear(k);
                }
            }
            "CONNECTOR DAMPING" => {
                if s.has("NONLINEAR") || s.param_upper("TYPE").is_some_and(|t| t != "VISCOUS") {
                    return s.err("bara linjär viskös dämpning stöds (COMPONENT=, en rad med c)");
                }
                let c = component(s, dim)?;
                let v = s.first_floats()?;
                beh.comps[c].damping = v.first().copied().unwrap_or(0.0);
            }
            "CONNECTOR PLASTICITY" => {
                let c = component(s, dim)?;
                if !avail.contains(&c) {
                    return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                }
                beh.comps[c].plastic = Some(Plastic { iso: Iso::Table(Vec::new()), kin: None });
                last_plastic = Some(c);
            }
            "CONNECTOR HARDENING" => {
                let Some(c) = last_plastic else { return s.err("*CONNECTOR HARDENING måste följa *CONNECTOR PLASTICITY") };
                let p = beh.comps[c].plastic.as_mut().unwrap();
                let ty = s.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
                let def = s.param_upper("DEFINITION");
                match ty.as_str() {
                    "ISOTROPIC" => match def.as_deref().unwrap_or("TABULAR") {
                        "TABULAR" => p.iso = Iso::Table(table(s, true)?),
                        "EXPONENTIAL LAW" => {
                            let v = s.first_floats()?;
                            if v.len() < 3 {
                                return s.err("EXPONENTIAL LAW kräver F0, Q∞, b");
                            }
                            p.iso = Iso::Exp { f0: v[0], q: v[1], b: v[2] };
                        }
                        other => return s.err(format!("DEFINITION={other} stöds inte (TABULAR, EXPONENTIAL LAW)")),
                    },
                    "KINEMATIC" => {
                        if def.as_deref() != Some("PARAMETERS") {
                            return s.err("kinematiskt hårdnande kräver DEFINITION=PARAMETERS (F0, C, γ)");
                        }
                        let v = s.first_floats()?;
                        if v.len() < 2 {
                            return s.err("skriv F0, C[, γ]");
                        }
                        p.kin = Some((v[1], v.get(2).copied().unwrap_or(0.0)));
                        if matches!(&p.iso, Iso::Table(t) if t.is_empty()) {
                            p.iso = Iso::Table(vec![(0.0, v[0])]);
                        }
                    }
                    other => return s.err(format!("TYPE={other} stöds inte (ISOTROPIC, KINEMATIC)")),
                }
            }
            "CONNECTOR DAMAGE INITIATION" => {
                let c = component(s, dim)?;
                let criterion = match s.param_upper("CRITERION").as_deref().unwrap_or("FORCE") {
                    "FORCE" => Criterion::Force,
                    "MOTION" => Criterion::Motion,
                    "PLASTIC MOTION" => Criterion::PlasticMotion,
                    other => return s.err(format!("CRITERION={other} stöds inte (FORCE, MOTION, PLASTIC MOTION)")),
                };
                let f = opt_fields(s)?;
                let d_max = s.param("MAXIMUM DEGRADATION").and_then(parse_f64).unwrap_or(1.0);
                beh.comps[c].damage = Some(Damage { criterion, lo: f.first().copied().flatten(), hi: f.get(1).copied().flatten(), du_f: None, d_max });
                last_damage = Some(c);
            }
            "CONNECTOR DAMAGE EVOLUTION" => {
                let Some(c) = last_damage else { return s.err("*CONNECTOR DAMAGE EVOLUTION måste följa *CONNECTOR DAMAGE INITIATION") };
                let ty = s.param_upper("TYPE").unwrap_or_else(|| "MOTION".into());
                let soft = s.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
                if ty != "MOTION" || soft != "LINEAR" {
                    return s.err(format!("TYPE={ty}, SOFTENING={soft} stöds inte ännu (TYPE=MOTION, SOFTENING=LINEAR)"));
                }
                if s.has("AFFECTED COMPONENTS") {
                    return s.err("AFFECTED COMPONENTS stöds inte ännu");
                }
                let v = s.first_floats()?;
                let Some(&du) = v.first().filter(|x| **x > 0.0) else { return s.err("Δu_f > 0 krävs") };
                beh.comps[c].damage.as_mut().unwrap().du_f = Some(du);
            }
            "CONNECTOR FRICTION" => {
                if s.has("PREDEFINED") {
                    return s.err("PREDEFINED friktion stöds inte ännu (använd COMPONENT= och den inre kontaktkraften)");
                }
                if s.has("CONTACT FORCE") {
                    return s.err("CONTACT FORCE= (härledda komponenter) stöds inte ännu; ange den inre kontaktkraften på dataraden");
                }
                let f0 = s.first_floats()?.first().copied().unwrap_or(0.0);
                let stick = s.param("STICK STIFFNESS").and_then(parse_f64);
                let fr = Friction { f0, mu: 0.0, stick };
                if s.has("COMPONENT") {
                    let c = component(s, dim)?;
                    if !avail.contains(&c) {
                        return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                    }
                    beh.comps[c].friction = Some(fr);
                    last_friction = Some(c);
                } else {
                    // utan COMPONENT= anges riktningen med *CONNECTOR POTENTIAL (en komponent)
                    pending = Some(fr);
                    last_friction = None;
                }
            }
            "CONNECTOR POTENTIAL" => {
                let Some(fr) = pending.take() else { return s.err("*CONNECTOR POTENTIAL stöds bara efter *CONNECTOR FRICTION utan COMPONENT=") };
                if s.data.len() != 1 {
                    return s.err("bara en komponent i *CONNECTOR POTENTIAL stöds (riktningen för friktionen)");
                }
                let f = &s.data[0].fields;
                let Some(c) = f.first().and_then(|x| crate::inp::parse_label(x.trim())).filter(|c| (1..=6).contains(c)) else {
                    return s.err("komponentnummer saknas");
                };
                let c = c as usize - 1;
                if let Some(w) = f.get(1).and_then(|x| parse_f64(x.trim())) {
                    if (w.abs() - 1.0).abs() > 1e-12 {
                        return s.err("bara skalfaktorn 1 stöds i *CONNECTOR POTENTIAL");
                    }
                }
                if !avail.contains(&c) {
                    return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                }
                beh.comps[c].friction = Some(fr);
                last_friction = Some(c);
            }
            "FRICTION" => {
                if s.data.len() > 1 {
                    return s.err("tabulerad friktion stöds inte ännu för kopplingselement");
                }
                let mu = s.first_floats()?.first().copied().unwrap_or(0.0);
                if let Some(p) = pending.as_mut() {
                    p.mu = mu;
                } else if let Some(c) = last_friction {
                    beh.comps[c].friction.as_mut().unwrap().mu = mu;
                } else {
                    return s.err("*FRICTION i *CONNECTOR BEHAVIOR måste följa *CONNECTOR FRICTION");
                }
            }
            "CONNECTOR DERIVED COMPONENT" => return s.err("*CONNECTOR DERIVED COMPONENT stöds inte ännu"),
            "CONNECTOR STOP" => {
                let c = component(s, dim)?;
                if !avail.contains(&c) {
                    return s.err(format!("komponent {} är inte tillgänglig för kopplingstypen", c + 1));
                }
                let f = opt_fields(s)?;
                beh.comps[c].stop = Some((f.first().copied().flatten(), f.get(1).copied().flatten()));
            }
            "CONNECTOR LOCK" => {
                let c = component(s, dim)?;
                let f = opt_fields(s)?;
                if f.len() > 2 && f[2..].iter().any(|x| x.is_some()) {
                    return s.err("bara lägesgränser stöds för *CONNECTOR LOCK (inte krafter)");
                }
                let comps = match s.param_upper("LOCK").as_deref().unwrap_or("ALL") {
                    "ALL" => avail.to_vec(),
                    x => match crate::inp::parse_label(x) {
                        Some(k @ 1..=6) => vec![k as usize - 1],
                        _ => return s.err(format!("LOCK={x}: ALL eller ett komponentnummer")),
                    },
                };
                beh.comps[c].lock = Some(Lock { lo: f.first().copied().flatten(), hi: f.get(1).copied().flatten(), comps });
            }
            "CONNECTOR CONSTITUTIVE REFERENCE" => {
                let f = opt_fields(s)?;
                if f.iter().any(|x| x.is_some_and(|v| v != 0.0)) {
                    return s.err("referenslängder skilda från referensläget stöds inte ännu");
                }
            }
            _ => {}
        }
    }
    // RIGID: en elastisk styvhet 1e6 (oberoende av
    // modellens övriga styvheter)
    for c in 0..6 {
        if beh.rigid[c] {
            beh.comps[c].elastic = Elastic::Linear(RIGID_STIFFNESS);
        }
    }
    if pending.is_some() {
        return b.err("*CONNECTOR FRICTION utan COMPONENT= kräver *CONNECTOR POTENTIAL");
    }
    for (c, cb) in beh.comps.iter().enumerate() {
        if let Some(p) = &cb.plastic {
            if matches!(&p.iso, Iso::Table(t) if t.is_empty()) {
                return b.err(format!("*CONNECTOR PLASTICITY, COMPONENT={} saknar *CONNECTOR HARDENING", c + 1));
            }
            if !matches!(cb.elastic, Elastic::Linear(_)) {
                return b.err(format!("plasticitet i komponent {} kräver linjär *CONNECTOR ELASTICITY", c + 1));
            }
        }
        if let Some(d) = &cb.damage {
            if d.du_f.is_some() && cb.plastic.is_some() {
                return b.err(format!("skadeutveckling med plasticitet (komponent {}) stöds inte ännu", c + 1));
            }
            if d.du_f.is_some() && !matches!(cb.elastic, Elastic::Linear(_)) {
                return b.err(format!("skadeutveckling i komponent {} kräver linjär *CONNECTOR ELASTICITY", c + 1));
            }
        }
        if let Some(f) = &cb.friction {
            if f.mu < 0.0 {
                return b.err("negativ friktionskoefficient");
            }
        }
    }
    Ok(beh)
}

/// Tillstånd per komponent.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompState {
    /// Plastisk rörelse, ackumulerad plastisk rörelse, mothållskraft.
    pub up: f64,
    pub ub: f64,
    pub alpha: f64,
    /// Skada och initieringsrörelse (0 = ej initierad).
    pub d: f64,
    pub u0: f64,
    /// Riktningen (±1) där skadan initierades; utvecklingen följer rörelsen i den riktningen.
    pub dsign: f64,
    /// Friktionskraft (klistring/glidning) och total glidning.
    pub ff: f64,
    pub slip: f64,
    /// Låst och låst värde (CU).
    pub locked: bool,
    pub lock_value: f64,
    /// Kraften från föreskriven rörelse (för avvecklingen när den tas bort).
    pub fm: f64,
}

/// Elastoplastisk kraft: (kraft, tangent, nytt tillstånd).
pub fn plastic(k: f64, p: &Plastic, u: f64, old: &CompState) -> (f64, f64, CompState) {
    let mut st = *old;
    let f_tr = k * (u - old.up);
    let xi = f_tr - old.alpha;
    let (r0, _) = p.iso.eval(old.ub);
    if xi.abs() <= r0 * (1.0 + 1e-12) {
        return (f_tr, k, st);
    }
    let s = xi.signum();
    let (c, g) = p.kin.unwrap_or((0.0, 0.0));
    // r(Δλ) = s (F − α) − R(ū) = 0
    let resid = |dl: f64| -> (f64, f64, f64) {
        let f = f_tr - k * s * dl;
        let a = (old.alpha + c * s * dl) / (1.0 + g * dl);
        let (r, h) = p.iso.eval(old.ub + dl);
        let da = (c * s - g * a) / (1.0 + g * dl);
        (s * (f - a) - r, -k - s * da - h, a)
    };
    let mut dl = 0.0;
    let mut lo = 0.0;
    let mut hi = xi.abs() / k.max(1e-300) * 2.0 + 1e-300;
    // säkerställ att hi ger r < 0
    for _ in 0..200 {
        if resid(hi).0 < 0.0 {
            break;
        }
        hi *= 2.0;
    }
    for _ in 0..200 {
        let (r, dr, _) = resid(dl);
        if r.abs() <= 1e-14 * r0.abs().max(1e-300) {
            break;
        }
        if r > 0.0 {
            lo = dl;
        } else {
            hi = dl;
        }
        let mut n = dl - r / dr;
        if !(n > lo && n < hi) || dr >= 0.0 {
            n = 0.5 * (lo + hi);
        }
        dl = n;
    }
    let (_, dr, a) = resid(dl);
    st.up = old.up + s * dl;
    st.ub = old.ub + dl;
    st.alpha = a;
    let f = f_tr - k * s * dl;
    // dF/du = k − k² /(k + s α' + R') = k + k²/dr
    let kt = k + k * k / dr;
    (f, kt, st)
}

/// Skada: (faktor 1 − d, d(1 − d)/du · F_eff-delen) uppdaterar tillståndet.
/// `f_eff` = oskadad kraft, `u` = rörelsen, `k` = elastisk styvhet.
pub fn damage(dm: &Damage, k: f64, u: f64, f_eff: f64, st: &mut CompState) -> (f64, f64) {
    let v = match dm.criterion {
        Criterion::Force => f_eff,
        Criterion::Motion => u,
        Criterion::PlasticMotion => st.ub,
    };
    if st.u0 == 0.0 {
        let hit = dm.hi.is_some_and(|h| v >= h) || dm.lo.is_some_and(|l| v <= l);
        if !hit {
            return (1.0, 0.0);
        }
        // initieringsrörelsen på den sida som nås
        let lim = if dm.hi.is_some_and(|h| v >= h) { dm.hi.unwrap() } else { dm.lo.unwrap() };
        st.dsign = if dm.hi.is_some_and(|h| v >= h) { 1.0 } else { -1.0 };
        st.u0 = match dm.criterion {
            Criterion::Force => (lim / k).abs(),
            _ => lim.abs(),
        }
        .max(1e-300);
    }
    let Some(du_f) = dm.du_f else { return (1.0, 0.0) };
    // rörelsen i initieringsriktningen (skadan växer inte vid rörelse åt andra hållet)
    let au = u * st.dsign;
    let u0 = st.u0;
    let uf = u0 + du_f;
    if au <= u0 {
        return (1.0 - st.d, 0.0);
    }
    let dn = (uf * (au - u0) / (au * (uf - u0))).min(dm.d_max);
    if dn > st.d {
        st.d = dn;
        // dd/du (med tecken)
        let dd = if dn < dm.d_max { uf * u0 / (au * au * (uf - u0)) * st.dsign } else { 0.0 };
        return (1.0 - dn, -dd);
    }
    (1.0 - st.d, 0.0)
}
