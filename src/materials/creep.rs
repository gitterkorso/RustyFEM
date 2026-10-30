//! Krypning (*CREEP) med isotrop elasticitet och (valfritt) von Mises-
//! plasticitet med isotropt hårdnande (*PLASTIC).
//!
//! Kryplagar (ε̄̇cr ekvivalent kryptöjningshastighet, q̃ von Mises-spänning):
//! - LAW=TIME (tidshårdnande): ε̄̇cr = A q̃ⁿ tᵐ, t = total tid (TIME=TOTAL,
//!   standard) eller kryptid (TIME=CREEP, summan av tiden i *VISCO-steg),
//! - LAW=STRAIN (töjningshårdnande, standard): ε̄̇cr = (A q̃ⁿ [(m+1) ε̄cr]ᵐ)^(1/(m+1)),
//! - LAW=TIME POWER LAW och LAW=POWER LAW: samma lagar med q0, n, m, ε̇0,
//!   omräknade till A = ε̇0^(m+1)/q0ⁿ (POWER LAW har ett tomt fält före ε̇0),
//! - LAW=HYPERB (Garofalo): ε̄̇cr = A (sinh B q̃)ⁿ exp(−ΔH/(R(θ − θZ))) med
//!   θZ från *PHYSICAL CONSTANTS, ABSOLUTE ZERO=.
//! A, n, m (TIME, STRAIN, POWER LAW-formerna) kan bero på temperaturen
//! (sista fältet); de interpoleras linjärt, konstant utanför tabellen. *ELASTIC
//! kan bero på temperaturen som i `thermal_elastic.rs`.
//!
//! Kryptöjningsinkrementet under ett inkrement (kontrollerat med
//! enelementsprov, fasta inkrement):
//! - tidshårdnande: Δε̄ = A q̃ⁿ [(t+Δt)^(m+1) − t^(m+1)]/(m+1), dvs. exakt
//!   tidsintegral med spänningen hållen fast,
//! - töjningshårdnande: samma uttryck med den ekvivalenta tiden
//!   t̃ = ((m+1) ε̄cr/(A q̃ⁿ))^(1/(m+1)) i stället för t (t̃ = total tid när ε̄cr = 0,
//!   som i första krypinkrementet),
//! - HYPERB: Δε̄ = ε̄̇cr Δt.
//! Explicit integration (framåt Euler, standard i geometriskt linjära *VISCO)
//! tar q̃, ε̄cr, flödesriktning n = 3/2 s/q̃ och de temperaturberoende
//! tabellkonstanterna (A, n, m) från inkrementets början, men temperaturen i
//! HYPERB-lagens Arrhenius-faktor vid slutet (båda kontrollerade med
//! temperatursteg och rampad temperatur); tangenten är elastisk. Implicit integration (bakåt Euler; NLGEOM,
//! eller när plasticitet är aktiv i punkten) använder q̃ och temperatur vid
//! slutet med radiell återföring: q̃ + 3G Δε̄(q̃) = q̃_tr, och med plasticitet
//! dessutom q̃ = σ0(ε̄pl + Δε̄pl). Den konsistenta tangenten är sluten:
//!
//!   D = K I⊗I + 2G θ P_dev + 2G (β − θ) N̂⊗N̂,  θ = q̃/q̃_tr, β = dq̃/dq̃_tr,
//!
//! med β = 1/(1 + 3G Δε̄′) för krypning, β = H/(3G + H + 3G Δε̄′ H) för
//! krypning och plasticitet (H = dσ0/dε̄pl).
//!
//! Felmåttet för CETOL (*VISCO) är skillnaden mellan kryptöjningsinkrementet
//! från villkoren vid inkrementets början och vid slutet,
//! |Δε̄(q̃_slut) − Δε̄(q̃_början)| med samma hårdnandetillstånd – med det måttet
//! och reglerna för tidsinkrementering (`steps/visco.rs`) bestäms
//! inkrementföljden. Stabilitetsgränsen för
//! explicit integration är Cormeaus (1975) Δt_s = 4(1+ν)/(3E ∂ε̄̇cr/∂q̃), för
//! potenslagen 4(1+ν) q̃/(3nE ε̄̇cr) (kontrollerat med ett tjockväggigt rör).
//!
//! Krypningen är aktiv bara i *VISCO (`time_dependence.rs`); i *STATIC beter
//! sig materialet elastiskt(-plastiskt).
//!
//! Utdata: CEEQ, CE (kryptöjning, ingenjörsskjuvtöjning som E) och med
//! *PLASTIC även PEEQ och PE.

use super::anisotropic::reduce_6x6;
use super::plasticity::{ddot, dev, iso_d6, mises, mv6, pe_labels, rotate_v6, update_reduced, Hardening, M6, V6};
use super::time_dependence::{TimeDependence, CREEP_EXPLICIT, CREEP_IMPLICIT, CREEP_OFF};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::cell::Cell;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "krypning (*CREEP) med elasticitet och von Mises-plasticitet",
    consumes: &["CREEP", "ELASTIC", "PLASTIC"],
    priority: 20,
    build: Some(build),
}];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Law {
    Time,
    Strain,
    Hyperb,
    /// LAW=USER: CREEP (`user::routines::Creep`, väljs med materialnamnet).
    User,
}

/// Kryplagen med sina konstanter.
#[derive(Debug, Clone)]
pub(crate) struct CreepLaw {
    pub(crate) law: Law,
    /// (A, n, m, −, T) eller för POWER LAW-formerna (q0, n, m, ε̇0, T), i stigande
    /// temperatur (TIME/STRAIN).
    table: Vec<[f64; 5]>,
    /// Tabellen har POWER LAW-formens konstanter (interpoleras före omräkningen till A).
    power: bool,
    /// HYPERB: A, B, n, ΔH, R.
    pub(crate) hyp: [f64; 5],
    /// TIME=CREEP: kryptid i stället för total tid.
    pub(crate) creep_time: bool,
    /// LAW=USER: rutinen och materialnamnet.
    pub(crate) user: Option<crate::user::routines::CreepRef>,
}

/// Tillståndslayout: [ε̄cr, εcr (6), ε̄pl, εpl (6), ε 3D (6), kryptid].
const CBAR: usize = 0;
const ECR: usize = 1;
const P: usize = 7;
const EPL: usize = 8;
const EPS3: usize = 14;
const TCR: usize = 20;
const NST: usize = 21;

pub(crate) fn parse_law(kw: &RawKeyword) -> Result<CreepLaw> {
    for p in ["DEPENDENCIES", "POTENTIAL"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let name = kw.param_upper("LAW").unwrap_or_else(|| "STRAIN".into());
    let creep_time = match kw.param_upper("TIME").as_deref() {
        None | Some("TOTAL") => false,
        Some("CREEP") => true,
        Some(o) => return kw.err(format!("TIME={o} ska vara TOTAL eller CREEP")),
    };
    if name == "USER" {
        // CREEP väljs med materialnamnet (sätts av `build`)
        return Ok(CreepLaw { law: Law::User, table: Vec::new(), power: false, hyp: [0.0; 5], creep_time, user: None });
    }
    let rows: Vec<(usize, Vec<f64>)> = kw.records().iter().map(|d| Ok((d.line, kw.floats(d)?))).collect::<Result<_>>()?;
    if rows.is_empty() {
        return kw.err("kryplagens konstanter saknas");
    }
    let (law, cols): (Law, usize) = match name.as_str() {
        "TIME" | "TIME POWER LAW" => (Law::Time, if name == "TIME" { 3 } else { 4 }),
        "STRAIN" | "POWER LAW" => (Law::Strain, if name == "STRAIN" { 3 } else { 5 }),
        "HYPERB" => (Law::Hyperb, 5),
        o => return kw.err(format!("LAW={o} stöds inte ännu (TIME, STRAIN, TIME POWER LAW, POWER LAW, HYPERB)")),
    };
    if law == Law::Hyperb {
        if rows.len() > 1 {
            return kw.err("LAW=HYPERB har en datarad (konstanterna kan inte bero på temperaturen)");
        }
        let v = &rows[0].1;
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let hyp = [g(0), g(1), g(2), g(3), g(4)];
        if v.len() > 5 && v[5..].iter().any(|&x| x != 0.0) {
            return kw.err_at(rows[0].0, "fältberoende kryplag stöds inte ännu");
        }
        if !(hyp[0] > 0.0 && hyp[1] > 0.0 && hyp[2] > 0.0) {
            return kw.err_at(rows[0].0, "A, B och n ska vara positiva");
        }
        if hyp[3] != 0.0 && !(hyp[4] > 0.0) {
            return kw.err_at(rows[0].0, "R (allmänna gaskonstanten) ska vara positiv när ΔH ≠ 0");
        }
        return Ok(CreepLaw { law, table: Vec::new(), power: false, hyp, creep_time, user: None });
    }
    let power = !matches!(name.as_str(), "TIME" | "STRAIN");
    let mut table: Vec<[f64; 5]> = Vec::new();
    for (line, v) in &rows {
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let (row, a, n, m, temp) = match name.as_str() {
            "TIME" | "STRAIN" => ([g(0), g(1), g(2), 0.0], g(0), g(1), g(2), g(3)),
            _ => {
                // q0, n, m, [a,] ε̇0 (standard 1), T
                let (q0, n, m) = (g(0), g(1), g(2));
                let (e0, temp) = if name == "POWER LAW" {
                    if g(3) != 0.0 {
                        return kw.err_at(*line, "POWER LAW med a ≠ 0 stöds inte ännu");
                    }
                    (v.get(4).copied().filter(|&x| x != 0.0).unwrap_or(1.0), g(5))
                } else {
                    (v.get(3).copied().filter(|&x| x != 0.0).unwrap_or(1.0), g(4))
                };
                if !(q0 > 0.0) {
                    return kw.err_at(*line, "q0 ska vara positivt");
                }
                ([q0, n, m, e0], e0.powf(m + 1.0) / q0.powf(n), n, m, temp)
            }
        };
        let extra = cols + 1;
        if v.len() > extra && v[extra..].iter().any(|&x| x != 0.0) {
            return kw.err_at(*line, "fältberoende kryplag stöds inte ännu");
        }
        if !(a > 0.0 && n > 0.0) || !(m > -1.0 && m <= 0.0) {
            return kw.err_at(*line, "kryplagen kräver A > 0, n > 0 och −1 < m ≤ 0");
        }
        if let Some(l) = table.last() {
            if !(temp > l[4]) {
                return kw.err_at(*line, "temperaturerna i *CREEP ska öka");
            }
        }
        table.push([row[0], row[1], row[2], row[3], temp]);
    }
    Ok(CreepLaw { law, table, power, hyp: [0.0; 5], creep_time, user: None })
}

impl CreepLaw {
    /// (A, n, m) vid temperaturen `t`. Användarens konstanter interpoleras
    /// linjärt (för POWER LAW-formerna q0, n, m och ε̇0) och räknas
    /// sedan om till A.
    fn params(&self, t: f64) -> [f64; 3] {
        let tab = &self.table;
        let r: [f64; 4] = if tab.len() == 1 || t <= tab[0][4] {
            std::array::from_fn(|i| tab[0][i])
        } else if let Some(w) = tab.windows(2).find(|w| t < w[1][4]) {
            let s = (t - w[0][4]) / (w[1][4] - w[0][4]);
            std::array::from_fn(|i| w[0][i] + s * (w[1][i] - w[0][i]))
        } else {
            std::array::from_fn(|i| tab[tab.len() - 1][i])
        };
        if self.power {
            [r[3].powf(r[2] + 1.0) / r[0].powf(r[1]), r[1], r[2]]
        } else {
            [r[0], r[1], r[2]]
        }
    }

    fn arrhenius(&self, temp: f64, tz: f64) -> f64 {
        let [_, _, _, dh, r] = self.hyp;
        if dh == 0.0 {
            1.0
        } else {
            (-dh / (r * (temp - tz))).exp()
        }
    }

    /// Ekvivalent kryptöjningsinkrement och dess derivata med avseende på q̃
    /// för spänningen `q`, kryptöjningen `ecr` och tiden `t` vid inkrementets början.
    /// Tabellkonstanterna (A, n, m) tas vid temperaturen `temp`, Arrhenius-
    /// faktorn i HYPERB vid `theta`.
    pub(crate) fn increment(&self, q: f64, ecr: f64, t: f64, dt: f64, temp: f64, theta: f64, tz: f64) -> (f64, f64) {
        if !(q > 0.0) || !(dt > 0.0) {
            return (0.0, 0.0);
        }
        match self.law {
            Law::Time | Law::Strain => {
                let [a, n, m] = self.params(temp);
                let k = a * q.powf(n);
                let mp = m + 1.0;
                let strain = self.law == Law::Strain && ecr > 0.0;
                let te = if strain { (mp * ecr / k).powf(1.0 / mp) } else { t.max(0.0) };
                let d = k * ((te + dt).powf(mp) - te.powf(mp)) / mp;
                let dd = if strain { n * k * dt * (te + dt).powf(m) / (q * mp) } else { n * d / q };
                (d, dd)
            }
            Law::User => match &self.user {
                Some(u) => u.increment(q, ecr, t, dt, temp),
                None => (0.0, 0.0),
            },
            Law::Hyperb => {
                let [a, b, n, _, _] = self.hyp;
                let arr = self.arrhenius(theta, tz);
                let (s, c) = ((b * q).sinh(), (b * q).cosh());
                (a * s.powf(n) * arr * dt, a * n * s.powf(n - 1.0) * c * b * arr * dt)
            }
        }
    }
}

/// Krypning med elasticitet och valfri plasticitet.
#[derive(Debug)]
pub struct Creep {
    e: f64,
    nu: f64,
    /// Temperaturberoende elasticitet (E, ν, T); tom om konstant.
    etab: Vec<(f64, f64, f64)>,
    law: CreepLaw,
    /// Isotropt hårdnande (*PLASTIC), ev. per temperatur.
    plastic: Option<(Hardening, Vec<(f64, Vec<(f64, f64)>)>)>,
    td: TimeDependence,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("CREEP") else { return Ok(None) };
    if super::creep_ext::claims(def) {
        return Ok(None);
    }
    let Some(ekw) = def.behavior("ELASTIC") else {
        return kw.err("*CREEP kräver *ELASTIC");
    };
    let ty = ekw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" {
        return ekw.err("*CREEP stöds bara med isotrop elasticitet (TYPE=ISOTROPIC)");
    }
    let etab = super::thermal_elastic::iso_table(ekw)?;
    let (e, nu) = (etab[0].0, etab[0].1);
    let etab = if etab.len() > 1 { etab } else { Vec::new() };
    let mut law = parse_law(kw)?;
    if law.law == Law::User {
        law.user = Some(crate::user::routines::CreepRef::new(&def.name)?);
    }
    let plastic = match def.behavior("PLASTIC") {
        None => None,
        Some(pk) => {
            let hard = pk.param_upper("HARDENING").unwrap_or_else(|| "ISOTROPIC".into());
            if hard != "ISOTROPIC" {
                return pk.err("*CREEP med *PLASTIC stöds bara med HARDENING=ISOTROPIC");
            }
            for p in ["RATE", "DEPENDENCIES"] {
                if pk.has(p) {
                    return pk.err(format!("parametern {p} stöds inte ännu"));
                }
            }
            let tabs = super::thermal_elastic::yield_tables(pk)?;
            let first = Hardening::Table(tabs[0].1.clone());
            Some((first, if tabs.len() > 1 { tabs } else { Vec::new() }))
        }
    };
    Ok(Some(Arc::new(Creep { e, nu, etab, law, plastic, td: TimeDependence::default() })))
}

/// Villkoren för en uppdatering.
struct Ctx {
    mode: u8,
    t: f64,
    dt: f64,
    temp: f64,
    temp_old: f64,
    tz: f64,
    /// Temperaturen i Arrhenius-faktorn vid explicit integration (slutet,
    /// eller början i *COUPLED TEMPERATURE-DISPLACEMENT).
    theta: f64,
}

/// Felmått och stabilitetsgräns från senaste 3D-uppdateringen.
#[derive(Default, Clone, Copy)]
struct Monitor {
    err: f64,
    dts: f64,
}

impl Creep {
    fn moduli(&self, temp: f64) -> (f64, f64) {
        if self.etab.is_empty() {
            (self.e, self.nu)
        } else {
            super::thermal_elastic::interp_iso(&self.etab, temp)
        }
    }

    fn hardening(&self, temp: f64) -> Option<Hardening> {
        self.plastic.as_ref().map(|(h, tabs)| if tabs.is_empty() { h.clone() } else { Hardening::Table(super::thermal_elastic::yield_at(tabs, temp)) })
    }

    /// Spänning och konsistent tangent i 3D för total töjning `eps`.
    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64], c: &Ctx, mon: &Cell<Monitor>) -> Result<(V6, M6)> {
        stn[..NST].copy_from_slice(&st[..NST]);
        let (e1, nu1) = self.moduli(c.temp);
        let d1 = iso_d6(e1, nu1);
        let g = e1 / (2.0 * (1.0 + nu1));
        let kb = e1 / (3.0 * (1.0 - 2.0 * nu1));
        let hard = self.hardening(c.temp);
        let p_n = st[P];
        let ecbar = st[CBAR];
        let ecr: V6 = std::array::from_fn(|i| st[ECR + i]);
        let epl: V6 = std::array::from_fn(|i| st[EPL + i]);
        let law = &self.law;
        let creep_on = c.mode != CREEP_OFF && c.dt > 0.0;
        if creep_on {
            stn[TCR] = st[TCR] + c.dt;
        }
        let tlaw = if law.creep_time { st[TCR] } else { c.t };
        // spänningen vid inkrementets början (för explicit integration och felmåttet)
        let start = || -> (V6, f64) {
            let (e0, nu0) = self.moduli(c.temp_old);
            let eps0: V6 = std::array::from_fn(|i| st[EPS3 + i] - ecr[i] - epl[i]);
            let s0 = dev(&mv6(&iso_d6(e0, nu0), &eps0));
            let q0 = mises(&s0);
            (s0, q0)
        };
        if creep_on && c.mode == CREEP_EXPLICIT {
            let (s0, q0) = start();
            let mut ecr1 = ecr;
            let (dc, dcq) = law.increment(q0, ecbar, tlaw, c.dt, c.temp_old, c.theta, c.tz);
            if q0 > 0.0 {
                for i in 0..6 {
                    let eng = if i < 3 { 1.0 } else { 2.0 };
                    ecr1[i] = ecr[i] + eng * dc * 1.5 * s0[i] / q0;
                }
            }
            let ee: V6 = std::array::from_fn(|i| eps[i] - ecr1[i] - epl[i]);
            let sig = mv6(&d1, &ee);
            let q1 = mises(&dev(&sig));
            let yields = hard.as_ref().is_some_and(|h| q1 - h.eval(p_n).0 > 1e-12 * h.eval(p_n).0);
            if !yields {
                let (de, _) = law.increment(q1, ecbar, tlaw, c.dt, c.temp_old, c.theta, c.tz);
                // stabilitetsgränsen (Cormeau 1975): Δt_s = 4(1+ν)/(3E ∂ε̄̇/∂q̃) med
                // inkrementets medelderivata ∂Δε̄/∂q̃ / Δt
                let (e0, nu0) = self.moduli(c.temp_old);
                let dts = if dcq > 0.0 && dcq.is_finite() { 4.0 * (1.0 + nu0) * c.dt / (3.0 * e0 * dcq) } else { f64::INFINITY };
                mon.set(Monitor { err: (de - dc).abs(), dts });
                stn[CBAR] = ecbar + dc;
                stn[ECR..ECR + 6].copy_from_slice(&ecr1);
                return Ok((sig, d1));
            }
            // plasticitet aktiv: implicit integration i punkten
        }
        // implicit (bakåt Euler) eller bara plasticitet
        let ee: V6 = std::array::from_fn(|i| eps[i] - ecr[i] - epl[i]);
        let sig_tr = mv6(&d1, &ee);
        let s_tr = dev(&sig_tr);
        let q_tr = mises(&s_tr);
        let cr = |q: f64| -> (f64, f64) { if creep_on { law.increment(q, ecbar, tlaw, c.dt, c.temp, c.temp, c.tz) } else { (0.0, 0.0) } };
        if !(q_tr > 0.0) {
            mon.set(Monitor { err: 0.0, dts: f64::INFINITY });
            return Ok((sig_tr, d1));
        }
        // krypning: q + 3G Δε̄(q) = q_tr
        let mut q = q_tr;
        let (mut dc, mut dcp) = cr(q);
        if dc > 0.0 {
            let (mut lo, mut hi) = (0.0, q_tr);
            q = q_tr - 3.0 * g * dc.min(q_tr / (3.0 * g));
            for _ in 0..200 {
                let (f, fp) = cr(q);
                let h = q + 3.0 * g * f - q_tr;
                if h.abs() <= 1e-13 * q_tr {
                    break;
                }
                if h > 0.0 {
                    hi = q;
                } else {
                    lo = q;
                }
                let mut next = q - h / (1.0 + 3.0 * g * fp);
                if !(next > lo && next < hi) {
                    next = 0.5 * (lo + hi);
                }
                if (hi - lo) <= 1e-15 * q_tr {
                    q = next;
                    break;
                }
                q = next;
            }
            (dc, dcp) = cr(q);
        }
        let mut dp = 0.0;
        let mut beta = 1.0 / (1.0 + 3.0 * g * dcp);
        if let Some(h) = &hard {
            let (sy, _) = h.eval(p_n);
            if q - sy > 1e-12 * sy {
                // plasticitet: q = σ0(ε̄pl + Δε̄pl), q_tr − 3G(Δε̄cr(q) + Δε̄pl) = q
                let r = |x: f64| -> (f64, f64, f64, f64) {
                    let (s, hh) = h.eval(p_n + x);
                    let (f, fp) = cr(s);
                    (q_tr - 3.0 * g * (x + f) - s, -3.0 * g - 3.0 * g * fp * hh - hh, s, hh)
                };
                let (mut lo, mut hi) = (0.0, (q_tr - sy) / (3.0 * g));
                let mut x = 0.0;
                let mut conv = false;
                for _ in 0..200 {
                    let (rv, dr, _, _) = r(x);
                    if rv.abs() <= 1e-13 * q_tr {
                        conv = true;
                        break;
                    }
                    if rv > 0.0 {
                        lo = x;
                    } else {
                        hi = x;
                    }
                    let mut next = if dr < 0.0 { x - rv / dr } else { f64::NAN };
                    if !(next > lo && next < hi) {
                        next = 0.5 * (lo + hi);
                    }
                    if (hi - lo) <= 1e-15 * hi.max(1e-300) {
                        conv = true;
                        x = next;
                        break;
                    }
                    x = next;
                }
                if !conv {
                    return fail("returalgoritmen för krypning med plasticitet konvergerade inte");
                }
                let (_, _, s, hh) = r(x);
                dp = x;
                q = s;
                (dc, dcp) = cr(q);
                beta = hh / (3.0 * g + hh + 3.0 * g * dcp * hh);
            }
        }
        let n: V6 = std::array::from_fn(|i| 1.5 * s_tr[i] / q_tr);
        let mut sig = sig_tr;
        for i in 0..6 {
            sig[i] -= 2.0 * g * (dc + dp) * n[i];
        }
        for i in 0..6 {
            let eng = if i < 3 { 1.0 } else { 2.0 };
            stn[ECR + i] = ecr[i] + eng * dc * n[i];
            stn[EPL + i] = epl[i] + eng * dp * n[i];
        }
        stn[CBAR] = ecbar + dc;
        stn[P] = p_n + dp;
        // felmått: inkrementet från villkoren vid början jämfört med slutet
        let mut m = Monitor { err: 0.0, dts: f64::INFINITY };
        if creep_on {
            let (_, q0) = start();
            let (d0, _) = law.increment(q0, ecbar, tlaw, c.dt, c.temp_old, c.theta, c.tz);
            m.err = (dc - d0).abs();
        }
        mon.set(m);
        // konsistent tangent
        let theta = q / q_tr;
        let nrm = ddot(&s_tr, &s_tr).sqrt();
        let nn: V6 = std::array::from_fn(|i| s_tr[i] / nrm);
        let cc = 2.0 * g * (beta - theta);
        let mut d = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let mm = if i < 3 && j < 3 { 1.0 } else { 0.0 };
                let pdev = if i == j { if i < 3 { 1.0 } else { 0.5 } } else { 0.0 } - mm / 3.0;
                d[i][j] = kb * mm + 2.0 * g * theta * pdev + cc * nn[i] * nn[j];
            }
        }
        Ok((sig, d))
    }
}

impl MaterialModel for Creep {
    fn name(&self) -> &str {
        "CREEP"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[ECR..ECR + 6], true);
        rotate_v6(r, &mut state[EPL..EPL + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mut mode = self.td.creep();
        if mode == CREEP_EXPLICIT && input.defgrad.is_some() {
            mode = CREEP_IMPLICIT;
        }
        let tz = self.td.absolute_zero();
        if mode != CREEP_OFF && self.law.law == Law::Hyperb && self.law.hyp[3] != 0.0 && !tz.is_finite() {
            return fail("*CREEP, LAW=HYPERB med ΔH ≠ 0 kräver *PHYSICAL CONSTANTS, ABSOLUTE ZERO=");
        }
        let c = Ctx { mode, t: input.time - input.dt, dt: input.dt, temp: input.temperature, temp_old: input.temperature - input.dtemp, tz, theta: if mode == CREEP_EXPLICIT && self.td.explicit_temp_at_start() { input.temperature - input.dtemp } else { input.temperature } };
        let (e1, nu1) = self.moduli(c.temp);
        let ce = iso_d6(e1, nu1);
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mon = Cell::new(Monitor { err: 0.0, dts: f64::INFINITY });
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, state_old, state_new, &c, &mon) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        let m = mon.get();
        self.td.record_error(m.err);
        if mode == CREEP_EXPLICIT {
            self.td.record_stable_dt(m.dts);
        }
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(reduce_6x6(&iso_d6(self.e, self.nu), mode))
    }

    fn output_names(&self) -> &[&'static str] {
        if self.plastic.is_some() {
            &["CEEQ", "PEEQ"]
        } else {
            &["CEEQ"]
        }
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[CBAR]];
        if self.plastic.is_some() {
            v.push(state[P]);
        }
        v
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let ce: Vec<&'static str> = match mode {
            StressMode::ThreeD => vec!["CE11", "CE22", "CE33", "CE12", "CE13", "CE23"],
            StressMode::Uniaxial => vec!["CE11"],
            _ => vec!["CE11", "CE22", "CE33", "CE12"],
        };
        let mut v = vec![("CEEQ", vec!["CEEQ"]), ("CE", ce)];
        if self.plastic.is_some() {
            v.push(("PEEQ", vec!["PEEQ"]));
            v.push(("PE", pe_labels(mode).0));
        }
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let idx = pe_labels(mode).1;
        let mut v = vec![state[CBAR]];
        v.extend(idx.iter().map(|&i| state[ECR + i]));
        if self.plastic.is_some() {
            v.push(state[P]);
            v.extend(idx.iter().map(|&i| state[EPL + i]));
        }
        if mode == StressMode::PlaneStress {
            // CE33 = 0 och PE33 = 0 redovisas för plan spänning (som E33)
            v[3] = 0.0;
            if self.plastic.is_some() {
                v[8] = 0.0;
            }
        }
        v
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        Some(&self.td)
    }
}
