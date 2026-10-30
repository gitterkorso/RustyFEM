//! Kryplagar och krypningsformer utöver `creep.rs`: *CREEP, LAW=ANAND,
//! LAW=DARVEAUX och LAW=DOUBLE POWER, volymetrisk svällning (*SWELLING) och
//! anisotrop krypning med Hills potential (*POTENTIAL under *CREEP), med
//! isotrop linjär elasticitet. Materialet byggs här (i stället för i
//! `creep.rs`) när någon av dessa förekommer (`claims`).
//!
//! Lagarna för krypning och svällning, q̃ = von Mises- eller Hills
//! ekvivalenta spänning, θ temperaturen
//! och θZ absoluta nollpunkten (*PHYSICAL CONSTANTS, ABSOLUTE ZERO=):
//!
//! - DOUBLE POWER: ε̄̇ = A1 e^(−B1/(θ−θZ)) (q̃/σ0)^C1 + A2 e^(−B2/(θ−θZ)) (q̃/σ0)^C2,
//! - DARVEAUX: ε̄̇ = ε̄̇s [1 + εT B e^(−B ε̄̇s t)], ε̄̇s = Css [sinh(α q̃)]ⁿ e^(−Q/(R(θ−θZ))),
//!   t = total tid,
//! - ANAND: ε̄̇ = A [sinh(ξ q̃/s)]^(1/m) e^(−Q/(R(θ−θZ))) med deformationsmotståndet
//!   s: ṡ = h0 |1 − s/s*|ᵃ sign(1 − s/s*) ε̄̇, s* = ŝ [ε̄̇/A e^(Q/(R(θ−θZ)))]ⁿ,
//!   h0 = A0 + A1 (θ−θZ) + A2 (θ−θZ)² + A3 ε̄̇ + A4 ε̄̇², s0 = S1 + S2 (θ−θZ) + S3 (θ−θZ)²,
//! - övriga lagar (TIME, STRAIN, POWER LAW, TIME POWER LAW, HYPERB) som i `creep.rs`.
//!
//! Svällning: ε̇ˢʷ = ε̄̇ˢʷ(θ)/3 · I med ε̄̇ˢʷ från tabellen (volymetrisk
//! töjningshastighet mot temperatur, linjär interpolation, temperaturen vid
//! inkrementets början). Svällningen ingår
//! i CE (men inte i CEEQ).
//!
//! Hills potential (*POTENTIAL: R11, R22, R33, R12, R13, R23, i materialets
//! lokala axlar) ger q̃ = √(σᵀPσ) och flödesriktningen ∂q̃/∂σ = Pσ/q̃
//! (`plasticity_hill::hill_matrix`).
//!
//! Integration som för `creep.rs` (läget sätts per steg av
//! `time_dependence.rs`): explicit framåt Euler med spänningen och
//! tillståndet vid inkrementets början; Arrhenius-faktorn i ANAND, DARVEAUX
//! och DOUBLE POWER samt svällningen tas vid inkrementets början, HYPERB:s vid
//! slutet (utom i *COUPLED TEMPERATURE-DISPLACEMENT, där allt tas vid början).
//! DARVEAUX använder tiden vid inkrementets slut, ANAND uppdaterar s implicit
//! med spänningen vid slutet. Tidshårdnande
//! lagar i `creep.rs` har exakt tidsintegral för fast spänning. Annars
//! implicit bakåt Euler (NLGEOM) med Newton på (σ, Δε̄, s) och numerisk
//! tangent. Felmåttet för CETOL och Cormeaus stabilitetsgräns som i `creep.rs`.
//! Utdata CEEQ och CE.

use super::anisotropic::reduce_6x6;
use super::creep::{parse_law, CreepLaw};
use super::plasticity::{dev, iso_d6, mises, mv6, pe_labels, rotate_v6, update_reduced, M6, V6};
use super::time_dependence::{TimeDependence, CREEP_EXPLICIT, CREEP_IMPLICIT, CREEP_OFF};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::cell::Cell;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "krypning (ANAND, DARVEAUX, DOUBLE POWER, *POTENTIAL) och svällning (*SWELLING)",
    consumes: &["CREEP", "ELASTIC", "SWELLING", "POTENTIAL"],
    priority: 21,
    build: Some(build),
}];

/// Om materialet ska byggas här i stället för i `creep.rs`.
pub fn claims(def: &MaterialDef) -> bool {
    if def.behavior("PLASTIC").is_some() {
        return def.behavior("CREEP").is_some_and(new_law) || def.behavior("SWELLING").is_some();
    }
    def.behavior("SWELLING").is_some() || def.behavior("CREEP").is_some_and(|k| new_law(k) || def.behavior("POTENTIAL").is_some())
}

fn new_law(kw: &RawKeyword) -> bool {
    matches!(kw.param_upper("LAW").as_deref(), Some("ANAND") | Some("DARVEAUX") | Some("DOUBLE POWER"))
}

#[derive(Debug, Clone)]
enum Law {
    Std(CreepLaw),
    DoublePower { a1: f64, b1: f64, c1: f64, a2: f64, b2: f64, c2: f64, s0: f64 },
    Darveaux { css: f64, qr: f64, alpha: f64, n: f64, et: f64, b: f64 },
    /// [S1, Q/R, A, ξ, m, A0, ŝ, n, a, S2, S3, A1, A2, A3, A4]
    Anand([f64; 15]),
}

#[derive(Debug)]
pub struct CreepExt {
    e: f64,
    nu: f64,
    etab: Vec<(f64, f64, f64)>,
    law: Option<Law>,
    /// Svällningshastighet mot temperatur.
    swell: Vec<(f64, f64)>,
    /// Hills matris (Voigt med tensorskjuvkomponenter), `None` = von Mises.
    hill: Option<M6>,
    td: TimeDependence,
}

/// Tillstånd: [ε̄cr, εcr+εsw (6), s (Anand), ε 3D (6), kryptid].
const CBAR: usize = 0;
const ECR: usize = 1;
const SANAND: usize = 7;
const EPS3: usize = 8;
const TCR: usize = 14;
const NST: usize = 15;

fn parse_new(kw: &RawKeyword) -> Result<Law> {
    if kw.has("DEPENDENCIES") {
        return kw.err("parametern DEPENDENCIES stöds inte ännu");
    }
    let rows: Vec<Vec<f64>> = kw.data.iter().map(|d| kw.floats(d)).collect::<Result<_>>()?;
    let name = kw.param_upper("LAW").unwrap_or_default();
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let g = |i: usize| flat.get(i).copied().unwrap_or(0.0);
    match name.as_str() {
        "DOUBLE POWER" => {
            if rows.len() != 1 {
                return kw.err("LAW=DOUBLE POWER har en datarad");
            }
            let s0 = g(6);
            if !(s0 > 0.0) || !(g(0) >= 0.0 && g(3) >= 0.0) {
                return kw.err("LAW=DOUBLE POWER kräver σ0 > 0 och A1, A2 ≥ 0");
            }
            Ok(Law::DoublePower { a1: g(0), b1: g(1), c1: g(2), a2: g(3), b2: g(4), c2: g(5), s0 })
        }
        "DARVEAUX" => {
            if rows.len() != 1 {
                return kw.err("LAW=DARVEAUX har en datarad");
            }
            if !(g(0) > 0.0 && g(2) > 0.0 && g(3) > 0.0) {
                return kw.err("LAW=DARVEAUX kräver Css > 0, α > 0 och n > 0");
            }
            Ok(Law::Darveaux { css: g(0), qr: g(1), alpha: g(2), n: g(3), et: g(4), b: g(5) })
        }
        _ => {
            if rows.len() != 2 {
                return kw.err("LAW=ANAND har två datarader (S1, Q/R, A, ξ, m, A0, ŝ, n och a, S2, S3, A1, A2, A3, A4)");
            }
            let mut p = [0.0; 15];
            for (i, x) in p.iter_mut().enumerate() {
                *x = rows[i / 8].get(i % 8).copied().unwrap_or(0.0);
            }
            if !(p[0] > 0.0 && p[2] > 0.0 && p[3] > 0.0 && p[4] > 0.0 && p[6] > 0.0) {
                return kw.err("LAW=ANAND kräver S1, A, ξ, m och ŝ > 0");
            }
            Ok(Law::Anand(p))
        }
    }
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    if let Some(pk) = def.behavior("PLASTIC") {
        return pk.err("*PLASTIC tillsammans med LAW=ANAND/DARVEAUX/DOUBLE POWER eller *SWELLING stöds inte ännu");
    }
    let Some(ekw) = def.behavior("ELASTIC") else {
        return fail(format!("material {}: krypning och svällning kräver *ELASTIC", def.name));
    };
    let ty = ekw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" {
        return ekw.err("krypning och svällning stöds bara med isotrop elasticitet (TYPE=ISOTROPIC)");
    }
    let etab = super::thermal_elastic::iso_table(ekw)?;
    let (e, nu) = (etab[0].0, etab[0].1);
    let etab = if etab.len() > 1 { etab } else { Vec::new() };
    let law = match def.behavior("CREEP") {
        None => None,
        Some(kw) if new_law(kw) => Some(parse_new(kw)?),
        Some(kw) => Some(Law::Std(parse_law(kw)?)),
    };
    let mut swell = Vec::new();
    if let Some(sk) = def.behavior("SWELLING") {
        for p in ["LAW", "DEPENDENCIES"] {
            if sk.has(p) {
                return sk.err(format!("parametern {p} stöds inte ännu för *SWELLING"));
            }
        }
        for d in &sk.data {
            let v = sk.floats(d)?;
            if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
                return sk.err_at(d.line, "fältberoende svällning stöds inte ännu");
            }
            let (r, t) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
            if let Some(&(_, tl)) = swell.last() {
                if !(t > tl) {
                    return sk.err_at(d.line, "temperaturerna i *SWELLING ska öka");
                }
            }
            swell.push((r, t));
        }
        if swell.is_empty() {
            return sk.err("svällningshastigheten saknas");
        }
    }
    let hill = match def.behavior("POTENTIAL") {
        None => None,
        Some(pk) => {
            if pk.data.len() > 1 || pk.has("DEPENDENCIES") {
                return pk.err("temperatur- eller fältberoende *POTENTIAL stöds inte ännu");
            }
            let v = pk.first_floats()?;
            if v.len() < 6 || v[..6].iter().any(|&x| !(x > 0.0)) {
                return pk.err("*POTENTIAL kräver positiva R11, R22, R33, R12, R13 och R23");
            }
            if def.behavior("CREEP").is_none() {
                return pk.err("*POTENTIAL utan *PLASTIC kräver *CREEP");
            }
            Some(super::plasticity_hill::hill_matrix(&[v[0], v[1], v[2], v[3], v[4], v[5]]))
        }
    };
    Ok(Some(Arc::new(CreepExt { e, nu, etab, law, swell, hill, td: TimeDependence::default() })))
}

/// Villkoren för en uppdatering.
struct Ctx {
    mode: u8,
    /// Total tid vid inkrementets början.
    t: f64,
    dt: f64,
    temp: f64,
    temp_old: f64,
    tz: f64,
    /// Temperaturen i Arrhenius-faktorn vid explicit integration (slutet,
    /// eller början i *COUPLED TEMPERATURE-DISPLACEMENT).
    theta: f64,
}

#[derive(Default, Clone, Copy)]
struct Monitor {
    err: f64,
    dts: f64,
}

/// Voigtvektor med tensorskjuvkomponenter → q̃ och ∂q̃/∂σ (ingenjörsform, dvs.
/// skjuvkomponenterna är derivatan med avseende på σ12 osv.).
fn equivalent(hill: &Option<M6>, sig: &V6) -> (f64, V6) {
    match hill {
        None => {
            let s = dev(sig);
            let q = mises(&s);
            if !(q > 0.0) {
                return (0.0, [0.0; 6]);
            }
            (q, std::array::from_fn(|i| 1.5 * s[i] / q * if i < 3 { 1.0 } else { 2.0 }))
        }
        Some(p) => {
            let ps = mv6(p, sig);
            let q2: f64 = (0..6).map(|i| sig[i] * ps[i]).sum();
            if !(q2 > 0.0) {
                return (0.0, [0.0; 6]);
            }
            let q = q2.sqrt();
            (q, std::array::from_fn(|i| ps[i] / q))
        }
    }
}

/// Löser f(s) = 0 med Newton (numerisk derivata) från startvärdet `s0`.
fn solve_scalar(s0: f64, f: impl Fn(f64) -> f64) -> f64 {
    let mut s = s0;
    for _ in 0..100 {
        let r = f(s);
        let h = 1e-7 * s.abs().max(1e-12);
        let d = (f(s + h) - f(s - h)) / (2.0 * h);
        if !(d.abs() > 0.0) {
            break;
        }
        let step = r / d;
        s -= step;
        if step.abs() <= 1e-15 * s.abs() {
            break;
        }
    }
    s
}

impl CreepExt {
    fn moduli(&self, temp: f64) -> (f64, f64) {
        if self.etab.is_empty() {
            (self.e, self.nu)
        } else {
            super::thermal_elastic::interp_iso(&self.etab, temp)
        }
    }

    fn swell_rate(&self, temp: f64) -> f64 {
        let t = &self.swell;
        if t.is_empty() {
            return 0.0;
        }
        if t.len() == 1 || temp <= t[0].1 {
            return t[0].0;
        }
        for w in t.windows(2) {
            if temp < w[1].1 {
                let s = (temp - w[0].1) / (w[1].1 - w[0].1);
                return w[0].0 + s * (w[1].0 - w[0].0);
            }
        }
        t[t.len() - 1].0
    }

    fn needs_zero(&self) -> bool {
        match &self.law {
            Some(Law::Std(l)) => l.law == super::creep::Law::Hyperb && l.hyp[3] != 0.0,
            Some(_) => true,
            None => false,
        }
    }

    /// Anands initiala deformationsmotstånd vid temperaturen θ.
    fn anand_s0(p: &[f64; 15], th: f64) -> f64 {
        p[0] + p[9] * th + p[10] * th * th
    }

    /// Ekvivalent kryptöjningsinkrement Δε̄ och ändringen av s för spänningen
    /// `q`, tillståndet (ε̄cr, s), tiden `t` (kryptid för TIME=CREEP, annars
    /// total), temperaturerna (tabellkonstanter vid `temp`, Arrhenius vid `theta`).
    #[allow(clippy::too_many_arguments)]
    fn increment(&self, q: f64, ecr: f64, s: f64, t: f64, dt: f64, temp: f64, theta: f64, tz: f64) -> (f64, f64) {
        let Some(law) = &self.law else { return (0.0, 0.0) };
        if !(q > 0.0) || !(dt > 0.0) {
            return (0.0, 0.0);
        }
        // ANAND, DARVEAUX och DOUBLE POWER: Arrhenius-faktorn vid `temp`
        // (inkrementets början vid explicit integration, kontrollerat med
        // temperaturramp; HYPERB i `creep.rs` använder slutet)
        let _ = theta;
        let th = temp - tz;
        match law {
            Law::Std(l) => (l.increment(q, ecr, t, dt, temp, theta, tz).0, 0.0),
            Law::DoublePower { a1, b1, c1, a2, b2, c2, s0 } => {
                let r = a1 * (-b1 / th).exp() * (q / s0).powf(*c1) + a2 * (-b2 / th).exp() * (q / s0).powf(*c2);
                (r * dt, 0.0)
            }
            Law::Darveaux { css, qr, alpha, n, et, b } => {
                let rs = css * (alpha * q).sinh().powf(*n) * (-qr / th).exp();
                // hastigheten med tiden vid inkrementets slut
                let d = rs * (1.0 + et * b * (-b * rs * (t + dt)).exp()) * dt;
                (d, 0.0)
            }
            Law::Anand(p) => {
                let arr = (-p[1] / th).exp();
                let rate = p[2] * (p[3] * q / s).sinh().powf(1.0 / p[4]) * arr;
                let sstar = p[6] * (rate / (p[2] * arr)).powf(p[7]);
                let h0 = p[5] + p[11] * th + p[12] * th * th + p[13] * rate + p[14] * rate * rate;
                let x = 1.0 - s / sstar;
                let sdot = h0 * x.abs().powf(p[8]) * x.signum() * rate;
                (rate * dt, sdot * dt)
            }
        }
    }

    /// Tiden som kryplagen använder vid inkrementets början.
    fn law_time(&self, st: &[f64], c: &Ctx) -> f64 {
        match &self.law {
            Some(Law::Std(l)) if l.creep_time => st[TCR],
            _ => c.t,
        }
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64], c: &Ctx, mon: &Cell<Monitor>) -> Result<(V6, M6)> {
        stn[..NST].copy_from_slice(&st[..NST]);
        let (e1, nu1) = self.moduli(c.temp);
        let d1 = iso_d6(e1, nu1);
        let on = c.mode != CREEP_OFF && c.dt > 0.0;
        if !on {
            let ee: V6 = std::array::from_fn(|i| eps[i] - st[ECR + i]);
            mon.set(Monitor { err: 0.0, dts: f64::INFINITY });
            return Ok((mv6(&d1, &ee), d1));
        }
        stn[TCR] = st[TCR] + c.dt;
        let tl = self.law_time(st, c);
        let ecr: V6 = std::array::from_fn(|i| st[ECR + i]);
        let ecbar = st[CBAR];
        let s_n = if let Some(Law::Anand(p)) = &self.law {
            if st[SANAND] > 0.0 {
                st[SANAND]
            } else {
                Self::anand_s0(p, c.temp_old - c.tz)
            }
        } else {
            0.0
        };
        // svällning: hastigheten vid inkrementets början (kontrollerat i
        // *VISCO med temperaturramp och i *COUPLED TEMPERATURE-DISPLACEMENT)
        let dsw = self.swell_rate(c.temp_old) * c.dt / 3.0;
        // spänningen vid inkrementets början
        let (e0, nu0) = self.moduli(c.temp_old);
        let eps0: V6 = std::array::from_fn(|i| st[EPS3 + i] - ecr[i]);
        let sig0 = mv6(&iso_d6(e0, nu0), &eps0);
        let (q0, n0) = equivalent(&self.hill, &sig0);
        let inc = |q: f64, s: f64| self.increment(q, ecbar, s, tl, c.dt, c.temp_old, c.theta, c.tz);
        let (dc0, ds0) = inc(q0, s_n);
        if c.mode == CREEP_EXPLICIT {
            let mut ecr1 = ecr;
            for i in 0..6 {
                ecr1[i] += dc0 * n0[i] + if i < 3 { dsw } else { 0.0 };
            }
            let ee: V6 = std::array::from_fn(|i| eps[i] - ecr1[i]);
            let sig = mv6(&d1, &ee);
            let (q1, _) = equivalent(&self.hill, &sig);
            let (dc1, _) = inc(q1, s_n);
            // stabilitetsgräns (Cormeau) med numerisk derivata ∂Δε̄/∂q̃
            let hq = 1e-6 * q0.max(1e-12);
            let dq = (inc(q0 + hq, s_n).0 - inc((q0 - hq).max(0.0), s_n).0) / (q0 + hq - (q0 - hq).max(0.0));
            let dts = if dq > 0.0 && dq.is_finite() { 4.0 * (1.0 + nu0) * c.dt / (3.0 * e0 * dq) } else { f64::INFINITY };
            mon.set(Monitor { err: (dc1 - dc0).abs(), dts });
            stn[CBAR] = ecbar + dc0;
            stn[ECR..ECR + 6].copy_from_slice(&ecr1);
            if let Some(Law::Anand(_)) = &self.law {
                // s uppdateras implicit med spänningen vid inkrementets
                // slut och temperaturen vid slutet, s = sₙ + ṡ(q̃₁, s, θ₁) Δt
                // (kontrollerat med temperaturramp)
                let _ = ds0;
                let inc1 = |q: f64, s: f64| self.increment(q, ecbar, s, tl, c.dt, c.temp, c.temp, c.tz);
                stn[SANAND] = solve_scalar(s_n, |s| s - s_n - inc1(q1, s).1);
            }
            return Ok((sig, d1));
        }
        // implicit: Newton på (σ, Δε̄, s)
        let sig = self.implicit(eps, &ecr, ecbar, s_n, dsw, &d1, &|q, s| self.increment(q, ecbar, s, tl, c.dt, c.temp, c.temp, c.tz), stn)?;
        let (q1, _) = equivalent(&self.hill, &sig);
        mon.set(Monitor { err: (inc(q1, s_n).0 - dc0).abs(), dts: f64::INFINITY });
        // numerisk tangent (centraldifferens)
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-8);
        let h = 1e-6 * scale;
        let mut tmp = vec![0.0; NST];
        let mut cmat = [[0.0; 6]; 6];
        for j in 0..6 {
            let mut ep = *eps;
            let mut em = *eps;
            ep[j] += h;
            em[j] -= h;
            let f = |q: f64, s: f64| self.increment(q, ecbar, s, tl, c.dt, c.temp, c.temp, c.tz);
            let sp = self.implicit(&ep, &ecr, ecbar, s_n, dsw, &d1, &f, &mut tmp)?;
            let sm = self.implicit(&em, &ecr, ecbar, s_n, dsw, &d1, &f, &mut tmp)?;
            for i in 0..6 {
                cmat[i][j] = (sp[i] - sm[i]) / (2.0 * h);
            }
        }
        Ok((sig, cmat))
    }

    /// Bakåt Euler: σ = D(ε − εcr,n − Δε̄ n(σ) − Δεsw), Δε̄ = Δ(q̃(σ), s),
    /// s = sₙ + Δs(q̃(σ), s). Skriver tillståndet i `stn`.
    #[allow(clippy::too_many_arguments)]
    fn implicit(&self, eps: &V6, ecr: &V6, ecbar: f64, s_n: f64, dsw: f64, d1: &M6, inc: &dyn Fn(f64, f64) -> (f64, f64), stn: &mut [f64]) -> Result<V6> {
        let anand = matches!(self.law, Some(Law::Anand(_)));
        let ee: V6 = std::array::from_fn(|i| eps[i] - ecr[i] - if i < 3 { dsw } else { 0.0 });
        let sig_tr = mv6(d1, &ee);
        let sc = sig_tr.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-30);
        let ssc = s_n.abs().max(1.0);
        // x = [σ/sc (6), s/ssc]
        let resid = |x: &[f64; 7]| -> [f64; 7] {
            let sig: V6 = std::array::from_fn(|i| x[i] * sc);
            let s = x[6] * ssc;
            let (q, n) = equivalent(&self.hill, &sig);
            let (dc, ds) = inc(q, s);
            let de: V6 = std::array::from_fn(|i| dc * n[i]);
            let dsig = mv6(d1, &de);
            let mut r = [0.0; 7];
            for i in 0..6 {
                r[i] = (sig[i] - sig_tr[i] + dsig[i]) / sc;
            }
            r[6] = if anand { (s - s_n - ds) / ssc } else { 0.0 };
            r
        };
        let mut x = [0.0; 7];
        for i in 0..6 {
            x[i] = sig_tr[i] / sc;
        }
        x[6] = s_n / ssc;
        let nrm = |r: &[f64; 7]| r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut r = resid(&x);
        let nu = if anand { 7 } else { 6 };
        for _ in 0..100 {
            if nrm(&r) <= 1e-14 {
                break;
            }
            let mut jac = vec![0.0; nu * nu];
            for j in 0..nu {
                let hs = 1e-7 * x[j].abs().max(1e-3);
                let mut xp = x;
                let mut xm = x;
                xp[j] += hs;
                xm[j] -= hs;
                let (rp, rm) = (resid(&xp), resid(&xm));
                for i in 0..nu {
                    jac[i * nu + j] = (rp[i] - rm[i]) / (2.0 * hs);
                }
            }
            let rhs: Vec<f64> = (0..nu).map(|i| -r[i]).collect();
            let Some(d) = super::anisotropic::solve_dense(&jac, &rhs) else { break };
            let r0 = nrm(&r);
            let mut lam = 1.0;
            loop {
                let mut xn = x;
                for i in 0..nu {
                    xn[i] += lam * d[i];
                }
                let rn = resid(&xn);
                if nrm(&rn) < r0 || lam < 1e-6 {
                    x = xn;
                    r = rn;
                    break;
                }
                lam *= 0.5;
            }
        }
        if nrm(&r) > 1e-9 {
            return fail(format!("implicit krypintegration konvergerade inte (residual {:e})", nrm(&r)));
        }
        let sig: V6 = std::array::from_fn(|i| x[i] * sc);
        let s = x[6] * ssc;
        let (q, n) = equivalent(&self.hill, &sig);
        let (dc, _) = inc(q, s);
        for i in 0..6 {
            stn[ECR + i] = ecr[i] + dc * n[i] + if i < 3 { dsw } else { 0.0 };
        }
        stn[CBAR] = ecbar + dc;
        if anand {
            stn[SANAND] = s;
        }
        Ok(sig)
    }
}

impl MaterialModel for CreepExt {
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
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mut mode = self.td.creep();
        if mode == CREEP_EXPLICIT && input.defgrad.is_some() {
            mode = CREEP_IMPLICIT;
        }
        let tz = self.td.absolute_zero();
        if mode != CREEP_OFF && self.needs_zero() && !tz.is_finite() {
            return fail("kryplagen kräver *PHYSICAL CONSTANTS, ABSOLUTE ZERO=");
        }
        let c = Ctx { mode, t: input.time - input.dt, dt: input.dt, temp: input.temperature, temp_old: input.temperature - input.dtemp, tz: if tz.is_finite() { tz } else { 0.0 }, theta: if mode == CREEP_EXPLICIT && self.td.explicit_temp_at_start() { input.temperature - input.dtemp } else { input.temperature } };
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
        &["CEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[CBAR]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let ce: Vec<&'static str> = match mode {
            StressMode::ThreeD => vec!["CE11", "CE22", "CE33", "CE12", "CE13", "CE23"],
            StressMode::Uniaxial => vec!["CE11"],
            _ => vec!["CE11", "CE22", "CE33", "CE12"],
        };
        vec![("CEEQ", vec!["CEEQ"]), ("CE", ce)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let idx = pe_labels(mode).1;
        let mut v = vec![state[CBAR]];
        v.extend(idx.iter().map(|&i| state[ECR + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        Some(&self.td)
    }
}
