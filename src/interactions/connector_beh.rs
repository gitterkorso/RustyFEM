//! Connector element behaviors (included with `#[path]` from `connector.rs`):
//! *CONNECTOR ELASTICITY (linear, matrix, NONLINEAR, RIGID), *CONNECTOR DAMPING,
//! *CONNECTOR PLASTICITY + *CONNECTOR HARDENING, *CONNECTOR DAMAGE INITIATION/
//! EVOLUTION, *CONNECTOR STOP, *CONNECTOR LOCK and *CONNECTOR FRICTION.
//!
//! All behaviors are uncoupled per component (except the elasticity matrix) and
//! one-dimensional, with the following semantics:
//! - NONLINEAR: table (force, displacement), constant outside the table
//!   (EXTRAPOLATION=CONSTANT), elastic (same curve on unloading).
//! - Plasticity: F = k (u − u_p), yield condition |F − α| ≤ R(ū_p). Isotropic
//!   hardening R tabulated (force, plastic motion; constant after the last
//!   point) or EXPONENTIAL LAW R = F₀ + Q∞ (1 − e^(−b ū_p)); kinematic
//!   (TYPE=KINEMATIC, DEFINITION=PARAMETERS: F₀, C, γ) dα = C du_p − γ α dū_p.
//!   Implicit return mapping (Simo & Hughes ch. 1) with consistent tangent.
//! - Damage: initiation when the force (CRITERION=FORCE, default), the motion
//!   (MOTION) or the plastic motion (PLASTIC MOTION) reaches the limits;
//!   TYPE=MOTION, SOFTENING=LINEAR: d = u_f (u − u₀)/(u (u_f − u₀)) with u
//!   measured in the direction in which the damage was initiated (motion in the other direction
//!   gives no new damage), u_f = u₀ + Δu_f, irreversible; the force becomes
//!   (1 − d)·F.
//! - Stops and locks apply to positions (CP = position in the reference configuration + CU), penalty.
//! - Friction: F_f = μ N with N = the internal contact force on the data line,
//!   sticking with STICK STIFFNESS (default: elastic slip 1e-4 · the length).
//! - RIGID: elastic stiffness 1e6.
//! - Damping: viscous F = c v in *DYNAMIC, v = Δu/Δt over the increment.

use crate::interactions::constraints::sub_options;
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::Result;

#[derive(Debug, Clone, Default)]
pub enum Elastic {
    #[default]
    None,
    Linear(f64),
    /// (displacement, force), sorted by displacement.
    Table(Vec<(f64, f64)>),
}

impl Elastic {
    /// (force, iteration stiffness) at the displacement u; `u_prev` is
    /// the previous evaluation. For the piecewise linear curve, Newton with the tangent
    /// oscillates between segments; the secant from the previous iteration
    /// converges and becomes exact when two iterations lie in the same segment.
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

    /// Representative stiffness (for the penalty coefficient).
    pub fn scale(&self) -> f64 {
        match self {
            Elastic::None => 0.0,
            Elastic::Linear(k) => k.abs(),
            Elastic::Table(t) => t.windows(2).map(|w| ((w[1].1 - w[0].1) / (w[1].0 - w[0].0)).abs()).fold(0.0, f64::max),
        }
    }
}

/// Piecewise linear interpolation, constant outside: (value, slope).
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
    /// (plastic motion, yield force)
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
    /// Kinematic hardening (C, γ).
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
    /// Δu_f (TYPE=MOTION, SOFTENING=LINEAR); None = initiation only.
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
    /// The components (0..6) that are locked.
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
    /// Elasticity matrix over the available components (6×6, row-major).
    pub matrix: Option<[[f64; 6]; 6]>,
    /// Components that are rigid (*CONNECTOR ELASTICITY, RIGID).
    pub rigid: [bool; 6],
}

impl Behavior {
    pub fn is_linear(&self) -> bool {
        self.comps.iter().all(|c| matches!(c.elastic, Elastic::None | Elastic::Linear(_)) && c.plastic.is_none() && c.damage.as_ref().is_none_or(|d| d.du_f.is_none()) && c.friction.is_none() && c.stop.is_none() && c.lock.is_none())
    }
}

/// The stiffness for *CONNECTOR ELASTICITY, RIGID.
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

/// Fields on the first data line (empty fields = None).
fn opt_fields(kw: &RawKeyword) -> Result<Vec<Option<f64>>> {
    let Some(d) = kw.data.first() else { return Ok(Vec::new()) };
    d.fields
        .iter()
        .map(|s| {
            let s = s.trim();
            if s.is_empty() {
                Ok(None)
            } else {
                parse_f64(s).map(Some).ok_or(()).or_else(|_| kw.err(format!("'{s}' is not a number")))
            }
        })
        .collect()
}

fn table(kw: &RawKeyword, swap: bool) -> Result<Vec<(f64, f64)>> {
    let mut t = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() < 2 {
            return kw.err_at(d.line, "enter two values per line");
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
                return kw.err("in 2D the components are 1, 2 and 6");
            }
            Ok(c)
        }
        _ => kw.err("COMPONENT=1..6 is required"),
    }
}

/// Reads the behavior *CONNECTOR BEHAVIOR, NAME=`name`. `avail` = available components.
pub fn parse(m: &FeModel, kw: &RawKeyword, name: &str, avail: &[usize]) -> Result<Behavior> {
    let up = name.trim().to_uppercase();
    let Some(b) = m.extra.iter().find(|k| k.name == "CONNECTOR BEHAVIOR" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*CONNECTOR BEHAVIOR, NAME={name} does not exist"));
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
            return s.err("DEPENDENCIES is not supported yet for connector behaviors");
        }
        match s.name.as_str() {
            "CONNECTOR ELASTICITY" => {
                if s.has("RIGID") {
                    let comps: Vec<usize> = s.data.iter().flat_map(|d| d.fields.iter()).filter(|f| !f.trim().is_empty()).map(|f| crate::inp::parse_label(f.trim()).map(|c| c as usize)).collect::<Option<Vec<_>>>().ok_or(()).or_else(|_| s.err("component number expected"))?;
                    let list: Vec<usize> = if let Ok(c) = component(s, dim) { vec![c] } else if comps.is_empty() { avail.to_vec() } else { comps.into_iter().map(|c| c - 1).collect() };
                    for c in list {
                        if c >= 6 || !avail.contains(&c) {
                            return s.err(format!("component {} is not available for the connection type", c + 1));
                        }
                        beh.rigid[c] = true;
                    }
                    continue;
                }
                if !s.has("COMPONENT") {
                    if s.has("NONLINEAR") {
                        return s.err("NONLINEAR requires COMPONENT=");
                    }
                    // matrix form: D11, D12, D22, D13, D23, D33, ... over the available components
                    let v: Vec<f64> = s.records().iter().map(|d| s.floats(d)).collect::<Result<Vec<_>>>()?.concat();
                    // 21 values: the full 6×6 matrix over components 1–6 (as
                    // the format requires), otherwise only the available components
                    let full: Vec<usize> = (0..6).collect();
                    let avail0 = avail;
                    let avail: &[usize] = if v.len() >= 21 { &full } else { avail };
                    let n = avail.len();
                    if v.len() < n * (n + 1) / 2 {
                        return s.err(format!("the elasticity matrix requires {} values for {n} available components", n * (n + 1) / 2));
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
                    // components that are not available are ignored
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
                    return s.err(format!("component {} is not available for the connection type", c + 1));
                }
                if s.has("NONLINEAR") {
                    // the data lines: force, displacement
                    beh.comps[c].elastic = Elastic::Table(table(s, true)?);
                } else {
                    if s.data.len() > 1 {
                        return s.err("temperature-dependent elasticity is not supported yet (one line with the stiffness)");
                    }
                    let v = s.first_floats()?;
                    let Some(&k) = v.first() else { return s.err("stiffness missing") };
                    beh.comps[c].elastic = Elastic::Linear(k);
                }
            }
            "CONNECTOR DAMPING" => {
                if s.has("NONLINEAR") || s.param_upper("TYPE").is_some_and(|t| t != "VISCOUS") {
                    return s.err("only linear viscous damping is supported (COMPONENT=, one line with c)");
                }
                let c = component(s, dim)?;
                let v = s.first_floats()?;
                beh.comps[c].damping = v.first().copied().unwrap_or(0.0);
            }
            "CONNECTOR PLASTICITY" => {
                let c = component(s, dim)?;
                if !avail.contains(&c) {
                    return s.err(format!("component {} is not available for the connection type", c + 1));
                }
                beh.comps[c].plastic = Some(Plastic { iso: Iso::Table(Vec::new()), kin: None });
                last_plastic = Some(c);
            }
            "CONNECTOR HARDENING" => {
                let Some(c) = last_plastic else { return s.err("*CONNECTOR HARDENING must follow *CONNECTOR PLASTICITY") };
                let p = beh.comps[c].plastic.as_mut().unwrap();
                let ty = s.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
                let def = s.param_upper("DEFINITION");
                match ty.as_str() {
                    "ISOTROPIC" => match def.as_deref().unwrap_or("TABULAR") {
                        "TABULAR" => p.iso = Iso::Table(table(s, true)?),
                        "EXPONENTIAL LAW" => {
                            let v = s.first_floats()?;
                            if v.len() < 3 {
                                return s.err("EXPONENTIAL LAW requires F0, Q∞, b");
                            }
                            p.iso = Iso::Exp { f0: v[0], q: v[1], b: v[2] };
                        }
                        other => return s.err(format!("DEFINITION={other} is not supported (TABULAR, EXPONENTIAL LAW)")),
                    },
                    "KINEMATIC" => {
                        if def.as_deref() != Some("PARAMETERS") {
                            return s.err("kinematic hardening requires DEFINITION=PARAMETERS (F0, C, γ)");
                        }
                        let v = s.first_floats()?;
                        if v.len() < 2 {
                            return s.err("enter F0, C[, γ]");
                        }
                        p.kin = Some((v[1], v.get(2).copied().unwrap_or(0.0)));
                        if matches!(&p.iso, Iso::Table(t) if t.is_empty()) {
                            p.iso = Iso::Table(vec![(0.0, v[0])]);
                        }
                    }
                    other => return s.err(format!("TYPE={other} is not supported (ISOTROPIC, KINEMATIC)")),
                }
            }
            "CONNECTOR DAMAGE INITIATION" => {
                let c = component(s, dim)?;
                let criterion = match s.param_upper("CRITERION").as_deref().unwrap_or("FORCE") {
                    "FORCE" => Criterion::Force,
                    "MOTION" => Criterion::Motion,
                    "PLASTIC MOTION" => Criterion::PlasticMotion,
                    other => return s.err(format!("CRITERION={other} is not supported (FORCE, MOTION, PLASTIC MOTION)")),
                };
                let f = opt_fields(s)?;
                let d_max = s.param("MAXIMUM DEGRADATION").and_then(parse_f64).unwrap_or(1.0);
                beh.comps[c].damage = Some(Damage { criterion, lo: f.first().copied().flatten(), hi: f.get(1).copied().flatten(), du_f: None, d_max });
                last_damage = Some(c);
            }
            "CONNECTOR DAMAGE EVOLUTION" => {
                let Some(c) = last_damage else { return s.err("*CONNECTOR DAMAGE EVOLUTION must follow *CONNECTOR DAMAGE INITIATION") };
                let ty = s.param_upper("TYPE").unwrap_or_else(|| "MOTION".into());
                let soft = s.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
                if ty != "MOTION" || soft != "LINEAR" {
                    return s.err(format!("TYPE={ty}, SOFTENING={soft} is not supported yet (TYPE=MOTION, SOFTENING=LINEAR)"));
                }
                if s.has("AFFECTED COMPONENTS") {
                    return s.err("AFFECTED COMPONENTS is not supported yet");
                }
                let v = s.first_floats()?;
                let Some(&du) = v.first().filter(|x| **x > 0.0) else { return s.err("Δu_f > 0 is required") };
                beh.comps[c].damage.as_mut().unwrap().du_f = Some(du);
            }
            "CONNECTOR FRICTION" => {
                if s.has("PREDEFINED") {
                    return s.err("PREDEFINED friction is not supported yet (use COMPONENT= and the internal contact force)");
                }
                if s.has("CONTACT FORCE") {
                    return s.err("CONTACT FORCE= (derived components) is not supported yet; give the internal contact force on the data line");
                }
                let f0 = s.first_floats()?.first().copied().unwrap_or(0.0);
                let stick = s.param("STICK STIFFNESS").and_then(parse_f64);
                let fr = Friction { f0, mu: 0.0, stick };
                if s.has("COMPONENT") {
                    let c = component(s, dim)?;
                    if !avail.contains(&c) {
                        return s.err(format!("component {} is not available for the connection type", c + 1));
                    }
                    beh.comps[c].friction = Some(fr);
                    last_friction = Some(c);
                } else {
                    // without COMPONENT= the direction is given with *CONNECTOR POTENTIAL (one component)
                    pending = Some(fr);
                    last_friction = None;
                }
            }
            "CONNECTOR POTENTIAL" => {
                let Some(fr) = pending.take() else { return s.err("*CONNECTOR POTENTIAL is only supported after *CONNECTOR FRICTION without COMPONENT=") };
                if s.data.len() != 1 {
                    return s.err("only one component in *CONNECTOR POTENTIAL is supported (the friction direction)");
                }
                let f = &s.data[0].fields;
                let Some(c) = f.first().and_then(|x| crate::inp::parse_label(x.trim())).filter(|c| (1..=6).contains(c)) else {
                    return s.err("component number missing");
                };
                let c = c as usize - 1;
                if let Some(w) = f.get(1).and_then(|x| parse_f64(x.trim())) {
                    if (w.abs() - 1.0).abs() > 1e-12 {
                        return s.err("only scale factor 1 is supported in *CONNECTOR POTENTIAL");
                    }
                }
                if !avail.contains(&c) {
                    return s.err(format!("component {} is not available for the connection type", c + 1));
                }
                beh.comps[c].friction = Some(fr);
                last_friction = Some(c);
            }
            "FRICTION" => {
                if s.data.len() > 1 {
                    return s.err("tabular friction is not supported yet for connector elements");
                }
                let mu = s.first_floats()?.first().copied().unwrap_or(0.0);
                if let Some(p) = pending.as_mut() {
                    p.mu = mu;
                } else if let Some(c) = last_friction {
                    beh.comps[c].friction.as_mut().unwrap().mu = mu;
                } else {
                    return s.err("*FRICTION in *CONNECTOR BEHAVIOR must follow *CONNECTOR FRICTION");
                }
            }
            "CONNECTOR DERIVED COMPONENT" => return s.err("*CONNECTOR DERIVED COMPONENT is not supported yet"),
            "CONNECTOR STOP" => {
                let c = component(s, dim)?;
                if !avail.contains(&c) {
                    return s.err(format!("component {} is not available for the connection type", c + 1));
                }
                let f = opt_fields(s)?;
                beh.comps[c].stop = Some((f.first().copied().flatten(), f.get(1).copied().flatten()));
            }
            "CONNECTOR LOCK" => {
                let c = component(s, dim)?;
                let f = opt_fields(s)?;
                if f.len() > 2 && f[2..].iter().any(|x| x.is_some()) {
                    return s.err("only position limits are supported for *CONNECTOR LOCK (not forces)");
                }
                let comps = match s.param_upper("LOCK").as_deref().unwrap_or("ALL") {
                    "ALL" => avail.to_vec(),
                    x => match crate::inp::parse_label(x) {
                        Some(k @ 1..=6) => vec![k as usize - 1],
                        _ => return s.err(format!("LOCK={x}: ALL or a component number")),
                    },
                };
                beh.comps[c].lock = Some(Lock { lo: f.first().copied().flatten(), hi: f.get(1).copied().flatten(), comps });
            }
            "CONNECTOR CONSTITUTIVE REFERENCE" => {
                let f = opt_fields(s)?;
                if f.iter().any(|x| x.is_some_and(|v| v != 0.0)) {
                    return s.err("reference lengths different from the reference configuration are not supported yet");
                }
            }
            _ => {}
        }
    }
    // RIGID: an elastic stiffness 1e6 (independent of
    // the model's other stiffnesses)
    for c in 0..6 {
        if beh.rigid[c] {
            beh.comps[c].elastic = Elastic::Linear(RIGID_STIFFNESS);
        }
    }
    if pending.is_some() {
        return b.err("*CONNECTOR FRICTION without COMPONENT= requires *CONNECTOR POTENTIAL");
    }
    for (c, cb) in beh.comps.iter().enumerate() {
        if let Some(p) = &cb.plastic {
            if matches!(&p.iso, Iso::Table(t) if t.is_empty()) {
                return b.err(format!("*CONNECTOR PLASTICITY, COMPONENT={} has no *CONNECTOR HARDENING", c + 1));
            }
            if !matches!(cb.elastic, Elastic::Linear(_)) {
                return b.err(format!("plasticity in component {} requires linear *CONNECTOR ELASTICITY", c + 1));
            }
        }
        if let Some(d) = &cb.damage {
            if d.du_f.is_some() && cb.plastic.is_some() {
                return b.err(format!("damage evolution with plasticity (component {}) is not supported yet", c + 1));
            }
            if d.du_f.is_some() && !matches!(cb.elastic, Elastic::Linear(_)) {
                return b.err(format!("damage evolution in component {} requires linear *CONNECTOR ELASTICITY", c + 1));
            }
        }
        if let Some(f) = &cb.friction {
            if f.mu < 0.0 {
                return b.err("negative friction coefficient");
            }
        }
    }
    Ok(beh)
}

/// State per component.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompState {
    /// Plastic motion, accumulated plastic motion, back force.
    pub up: f64,
    pub ub: f64,
    pub alpha: f64,
    /// Damage and initiation motion (0 = not initiated).
    pub d: f64,
    pub u0: f64,
    /// The direction (±1) in which the damage was initiated; the evolution follows the motion in that direction.
    pub dsign: f64,
    /// Friction force (sticking/slipping) and total slip.
    pub ff: f64,
    pub slip: f64,
    /// Locked and locked value (CU).
    pub locked: bool,
    pub lock_value: f64,
    /// The force from prescribed motion (for ramping it down when it is removed).
    pub fm: f64,
}

/// Elastoplastic force: (force, tangent, new state).
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
    // make sure hi gives r < 0
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

/// Damage: (factor 1 − d, d(1 − d)/du · F_eff part); updates the state.
/// `f_eff` = undamaged force, `u` = the motion, `k` = elastic stiffness.
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
        // the initiation motion on the side that is reached
        let lim = if dm.hi.is_some_and(|h| v >= h) { dm.hi.unwrap() } else { dm.lo.unwrap() };
        st.dsign = if dm.hi.is_some_and(|h| v >= h) { 1.0 } else { -1.0 };
        st.u0 = match dm.criterion {
            Criterion::Force => (lim / k).abs(),
            _ => lim.abs(),
        }
        .max(1e-300);
    }
    let Some(du_f) = dm.du_f else { return (1.0, 0.0) };
    // the motion in the initiation direction (the damage does not grow for motion in the other direction)
    let au = u * st.dsign;
    let u0 = st.u0;
    let uf = u0 + du_f;
    if au <= u0 {
        return (1.0 - st.d, 0.0);
    }
    let dn = (uf * (au - u0) / (au * (uf - u0))).min(dm.d_max);
    if dn > st.d {
        st.d = dn;
        // dd/du (signed)
        let dd = if dn < dm.d_max { uf * u0 / (au * au * (uf - u0)) * st.dsign } else { 0.0 };
        return (1.0 - dn, -dd);
    }
    (1.0 - st.d, 0.0)
}
