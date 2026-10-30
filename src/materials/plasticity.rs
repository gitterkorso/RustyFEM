//! Plasticitet med von Mises flytvillkor (*PLASTIC) och isotrop linjär elasticitet.
//!
//! Hårdnande:
//! - HARDENING=ISOTROPIC (standard): flytspänningen som tabell över plastisk
//!   ekvivalent töjning (linjär interpolation, konstant efter sista punkten),
//! - HARDENING=KINEMATIC: linjärt kinematiskt hårdnande (Prager/Ziegler),
//!   C = (σ1 − σ0)/εp1 från två datapunkter, konstant flytytestorlek σ0,
//! - HARDENING=COMBINED, DATATYPE=PARAMETERS: olinjärt kinematiskt hårdnande
//!   (Armstrong–Frederick/Chaboche) med en eller flera mottensorer
//!   (NUMBER BACKSTRESSES=n, data σ0, C1, γ1, C2, γ2, ...) och isotrop del från
//!   *CYCLIC HARDENING (tabell σ0(ε̄p) eller PARAMETERS σ|0, Q∞, b enligt Voce).
//!
//! Integrationen är implicit Euler (radial return / closest point projection):
//! med prövospänningen σ_tr = D(ε − εp_n). Mottensorernas evolution
//! α̇_k = 2/3 C_k n ṗ − γ_k α_k ṗ integreras exakt för fix flödesriktning n
//! under inkrementet: α_k = θ_k α_k,n + 2/3 ψ_k n med θ_k = e^(−γ_k Δp) och
//! ψ_k = C_k (1 − θ_k)/γ_k (se tests/cases/materials). Med den relativa deviatorn
//! ξ̃(Δp) = s_tr − Σ θ_k α_k,n löses den skalära ekvationen
//!
//!   q(ξ̃) − 3G Δp − Σ ψ_k − σ0(p_n + Δp) = 0
//!
//! med Newton (inneslutet mellan säkra gränser). Flödesriktningen är
//! n = 3/2 ξ̃/q(ξ̃) och Δεp = Δp n. Den konsistenta tangenten är
//! sluten (Simo–Taylor) när alla γ_k = 0, annars numeriskt deriverad av
//! returalgoritmen (framåtdifferens).
//!
//! Plan spänning och stänger löses med 3D-algoritmen och lokal Newton på de
//! spänningsfria töjningskomponenterna (σ33 = σ13 = σ23 = 0 respektive bara σ11 ≠ 0);
//! tangenten kondenseras statiskt. Plan töjning och axisymmetri använder
//! 3D-algoritmen direkt med ε13 = ε23 = 0.
//!
//! Temperaturberoende (HARDENING=ISOTROPIC): *ELASTIC med rader (E, ν, T) och
//! *PLASTIC med rader (σ0, ε̄p, T). Modulerna och flytkurvan interpoleras
//! linjärt i temperaturen vid inkrementets slut (konstant utanför tabellen) och
//! spänningen räknas i totalform σ = D(T)(ε − εp). Se
//! `thermal_elastic.rs` för tabellerna.
//!
//! Utdata: PEEQ (ackumulerad plastisk ekvivalent töjning ∫√(2/3 ε̇p:ε̇p) dt)
//! och PE (plastisk töjning, ingenjörsskjuvtöjning som E).

use super::anisotropic::{free_components, mode_components, solve_dense};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "von Mises-plasticitet",
    consumes: &["ELASTIC", "PLASTIC", "CYCLIC HARDENING"],
    priority: 10,
    build: Some(build),
}];

pub type M6 = [[f64; 6]; 6];
pub type V6 = [f64; 6];

/// Isotrop elasticitetsmatris (3D, ingenjörsskjuvtöjning).
pub fn iso_d6(e: f64, nu: f64) -> M6 {
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lam;
        }
        d[i][i] = lam + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    d
}

pub fn mv6(d: &M6, x: &V6) -> V6 {
    std::array::from_fn(|i| (0..6).map(|j| d[i][j] * x[j]).sum())
}

/// Deviator av en spänningslik vektor (tensorskjuvkomponenter).
pub fn dev(s: &V6) -> V6 {
    let p = (s[0] + s[1] + s[2]) / 3.0;
    [s[0] - p, s[1] - p, s[2] - p, s[3], s[4], s[5]]
}

/// a:b för spänningslika vektorer (skjuvkomponenter räknas två gånger).
pub fn ddot(a: &V6, b: &V6) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + 2.0 * (a[3] * b[3] + a[4] * b[4] + a[5] * b[5])
}

/// von Mises ekvivalent √(3/2 s:s) för en deviator.
/// Roterar en symmetrisk tensor i Voigt-form (ordning 11, 22, 33, 12, 13, 23)
/// till R A Rᵀ. `engineering`: skjuvkomponenterna är ingenjörstöjningar (2·A12).
pub fn rotate_v6(r: &[[f64; 3]; 3], v: &mut [f64], engineering: bool) {
    let f = if engineering { 0.5 } else { 1.0 };
    let a = [[v[0], v[3] * f, v[4] * f], [v[3] * f, v[1], v[5] * f], [v[4] * f, v[5] * f, v[2]]];
    let mut ra = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ra[i][j] = (0..3).map(|k| r[i][k] * a[k][j]).sum();
        }
    }
    let mut b = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = (0..3).map(|k| ra[i][k] * r[j][k]).sum();
        }
    }
    let g = 1.0 / f;
    v[..6].copy_from_slice(&[b[0][0], b[1][1], b[2][2], b[0][1] * g, b[0][2] * g, b[1][2] * g]);
}

pub fn mises(s: &V6) -> f64 {
    (1.5 * ddot(s, s)).max(0.0).sqrt()
}

// ---------------------------------------------------------------------------
// Reduktion från 3D till plan spänning, plan töjning och stänger
// ---------------------------------------------------------------------------

/// Uppdatering i 3D: total töjning (6, ingenjörsskjuvning) → (spänning, tangent).
pub type Update3D<'a> = dyn FnMut(&V6) -> Result<(V6, M6)> + 'a;

/// Kör en 3D-spänningsuppdatering i spänningstillståndet `mode`.
///
/// `eps3_old` är den fullständiga 3D-töjningen vid inkrementets början (för
/// startgissningen av de spänningsfria komponenterna), `ce` den elastiska
/// 3D-matrisen. Returnerar 3D-töjningen och resultatet i `mode`.
pub fn update_reduced(mode: StressMode, strain: &[f64], eps3_old: &V6, ce: &M6, f: &mut Update3D) -> Result<(V6, MatOutput)> {
    let a = mode_components(mode);
    let b = free_components(mode);
    let mut eps = [0.0; 6];
    for (k, &i) in a.iter().enumerate() {
        eps[i] = strain[k];
    }
    if !b.is_empty() {
        // startgissning: gammal töjning + elastisk kondensation av ändringen
        let nb = b.len();
        let dbb: Vec<f64> = b.iter().flat_map(|&i| b.iter().map(move |&j| ce[i][j])).collect();
        let rhs: Vec<f64> = b.iter().map(|&i| -a.iter().map(|&j| ce[i][j] * (eps[j] - eps3_old[j])).sum::<f64>()).collect();
        let d = solve_dense(&dbb, &rhs).unwrap_or_else(|| vec![0.0; nb]);
        for (k, &i) in b.iter().enumerate() {
            eps[i] = eps3_old[i] + d[k];
        }
    }
    let mut res = f(&eps)?;
    if !b.is_empty() {
        let nb = b.len();
        let scale = res.0.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(ce[0][0] * 1e-12);
        let mut ok = false;
        for _ in 0..60 {
            let r: Vec<f64> = b.iter().map(|&i| res.0[i]).collect();
            if r.iter().all(|x| x.abs() <= 1e-12 * scale) {
                ok = true;
                break;
            }
            let cbb: Vec<f64> = b.iter().flat_map(|&i| b.iter().map(move |&j| res.1[i][j])).collect();
            let Some(d) = solve_dense(&cbb, &r) else { break };
            for (k, &i) in b.iter().enumerate() {
                eps[i] -= d[k];
            }
            res = f(&eps)?;
            let _ = nb;
        }
        if !ok {
            let r = b.iter().map(|&i| res.0[i].abs()).fold(0.0, f64::max);
            if r > 1e-8 * scale {
                return fail(format!("materialpunkten konvergerade inte mot spänningsfrihet ({mode:?}), |σ| = {r:e}"));
            }
        }
    }
    let (sig, c) = res;
    let stress: Vec<f64> = a.iter().map(|&i| sig[i]).collect();
    let na = a.len();
    let mut tangent: Vec<f64> = a.iter().flat_map(|&i| a.iter().map(move |&j| c[i][j])).collect();
    if !b.is_empty() {
        let nb = b.len();
        let cbb: Vec<f64> = b.iter().flat_map(|&i| b.iter().map(move |&j| c[i][j])).collect();
        // C_aa − C_ab C_bb⁻¹ C_ba, kolumn för kolumn
        for (cj, &j) in a.iter().enumerate() {
            let rhs: Vec<f64> = b.iter().map(|&i| c[i][j]).collect();
            if let Some(x) = solve_dense(&cbb, &rhs) {
                for (ri, &i) in a.iter().enumerate() {
                    let s: f64 = (0..nb).map(|k| c[i][b[k]] * x[k]).sum();
                    tangent[ri * na + cj] -= s;
                }
            }
        }
    }
    Ok((eps, MatOutput { stress, tangent }))
}

/// Komponentnamn för plastisk töjning i ett spänningstillstånd.
pub fn pe_labels(mode: StressMode) -> (Vec<&'static str>, &'static [usize]) {
    match mode {
        StressMode::ThreeD => (vec!["PE11", "PE22", "PE33", "PE12", "PE13", "PE23"], &[0, 1, 2, 3, 4, 5]),
        StressMode::Uniaxial => (vec!["PE11"], &[0]),
        _ => (vec!["PE11", "PE22", "PE33", "PE12"], &[0, 1, 2, 3]),
    }
}

// ---------------------------------------------------------------------------
// Hårdnande
// ---------------------------------------------------------------------------

/// Flytytans storlek σ0 som funktion av plastisk ekvivalent töjning.
#[derive(Debug, Clone)]
pub enum Hardening {
    /// (ε̄p, σ0)-punkter, stigande ε̄p, första ε̄p = 0.
    Table(Vec<(f64, f64)>),
    /// σ0 = s0 + q (1 − e^(−b ε̄p)).
    Voce { s0: f64, q: f64, b: f64 },
}

impl Hardening {
    /// (σ0, dσ0/dε̄p) vid `p`.
    pub fn eval(&self, p: f64) -> (f64, f64) {
        match self {
            Hardening::Table(t) => {
                if t.len() == 1 || p <= t[0].0 {
                    let h = if t.len() > 1 { (t[1].1 - t[0].1) / (t[1].0 - t[0].0) } else { 0.0 };
                    return (t[0].1, if p < t[0].0 { 0.0 } else { h });
                }
                for w in t.windows(2) {
                    let ((p0, s0), (p1, s1)) = (w[0], w[1]);
                    if p < p1 {
                        let h = (s1 - s0) / (p1 - p0);
                        return (s0 + h * (p - p0), h);
                    }
                }
                (t[t.len() - 1].1, 0.0)
            }
            Hardening::Voce { s0, q, b } => {
                let e = (-b * p).exp();
                (s0 + q * (1.0 - e), q * b * e)
            }
        }
    }

    pub fn initial(&self) -> f64 {
        self.eval(0.0).0
    }
}

/// Tolkar en tabell (σ0, ε̄p) från datarader.
fn yield_table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "temperatur- eller fältberoende hårdnande stöds inte ännu");
        }
        let s = v.first().copied().unwrap_or(0.0);
        let p = v.get(1).copied().unwrap_or(0.0);
        if !(s > 0.0) {
            return kw.err_at(d.line, "flytspänningen ska vara positiv");
        }
        if let Some(&(pl, _)) = t.last() {
            if !(p > pl) {
                return kw.err_at(d.line, "den plastiska töjningen ska öka i tabellen");
            }
        } else if p != 0.0 {
            return kw.err_at(d.line, "första punkten ska ha plastisk töjning 0");
        }
        t.push((p, s));
    }
    if t.is_empty() {
        return kw.err("flytspänning saknas");
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// Modellen
// ---------------------------------------------------------------------------

/// von Mises-plasticitet med isotropt och (linjärt eller olinjärt) kinematiskt hårdnande.
#[derive(Debug, Clone)]
pub struct Mises {
    pub e: f64,
    pub nu: f64,
    pub g: f64,
    pub ce: M6,
    pub iso: Hardening,
    /// Mottensorer (C_k, γ_k).
    pub back: Vec<(f64, f64)>,
    /// Temperaturberoende elasticitet (E, ν, T); tom om konstant.
    pub etab: Vec<(f64, f64, f64)>,
    /// Temperaturberoende flytkurvor (T, tabell); tom om oberoende av temperaturen.
    pub ytab: Vec<(f64, Vec<(f64, f64)>)>,
}

/// Tillståndslayout: [ε̄p, εp (6), ε 3D (6), α_k (6 per mottensor)].
const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const ALPHA: usize = 13;

fn elastic_iso(def: &MaterialDef) -> Result<Vec<(f64, f64, f64)>> {
    let Some(kw) = def.behavior("ELASTIC") else {
        return fail(format!("material {}: *PLASTIC kräver *ELASTIC", def.name));
    };
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" {
        return kw.err("von Mises-plasticitet kräver isotrop elasticitet (TYPE=ISOTROPIC)");
    }
    if kw.has("TEMPERATURE") {
        return kw.err("parametern TEMPERATURE stöds inte ännu");
    }
    super::thermal_elastic::iso_table(kw)
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("PLASTIC") else { return Ok(None) };
    if super::plasticity_hill::claims(def) {
        return Ok(None); // Hill, *RATE DEPENDENT, Johnson–Cook, anisotrop elasticitet
    }
    let etab = elastic_iso(def)?;
    let (e, nu) = (etab[0].0, etab[0].1);
    let etab = if etab.len() > 1 { etab } else { Vec::new() };
    let mut ytab = Vec::new();
    for p in ["RATE", "DEPENDENCIES"] {
        if kw.has(p) {
            return kw.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let hard = kw.param_upper("HARDENING").unwrap_or_else(|| "ISOTROPIC".into());
    let cyclic = def.behavior("CYCLIC HARDENING");
    let (iso, back) = match hard.as_str() {
        "ISOTROPIC" => {
            if cyclic.is_some() {
                return kw.err("*CYCLIC HARDENING kräver HARDENING=COMBINED");
            }
            let tabs = super::thermal_elastic::yield_tables(kw)?;
            let first = tabs[0].1.clone();
            if tabs.len() > 1 {
                ytab = tabs;
            }
            (Hardening::Table(first), Vec::new())
        }
        "KINEMATIC" => {
            if cyclic.is_some() {
                return kw.err("*CYCLIC HARDENING kräver HARDENING=COMBINED");
            }
            let t = yield_table(kw)?;
            if t.len() > 2 {
                return kw.err("linjärt kinematiskt hårdnande tar högst två datapunkter (σ0, 0) och (σ1, εp1)");
            }
            let c = if t.len() == 2 { (t[1].1 - t[0].1) / t[1].0 } else { 0.0 };
            (Hardening::Table(vec![t[0]]), vec![(c, 0.0)])
        }
        "COMBINED" => {
            let dt = kw.param_upper("DATATYPE").unwrap_or_else(|| "HALF CYCLE".into());
            if dt != "PARAMETERS" {
                return kw.err(format!("HARDENING=COMBINED stöds bara med DATATYPE=PARAMETERS (inte {dt})"));
            }
            let nb = match kw.param("NUMBER BACKSTRESSES") {
                Some(s) => s.trim().parse::<usize>().ok().filter(|&n| (1..=10).contains(&n)).ok_or_else(|| {
                    crate::model::FemError(format!("{}: NUMBER BACKSTRESSES={s} ska vara 1–10", kw.location()))
                })?,
                None => 1,
            };
            let mut v = Vec::new();
            for d in kw.records() {
                v.extend(kw.floats(&d)?);
            }
            if kw.records().len() > 1 {
                return kw.err("temperaturberoende parametrar (flera datarader) stöds inte ännu");
            }
            if v.len() < 1 + 2 * nb {
                return kw.err(format!("dataraden ska ha σ0 och {nb} par (C, γ)"));
            }
            if v.len() > 1 + 2 * nb && v[1 + 2 * nb..].iter().any(|&x| x != 0.0) {
                return kw.err("temperaturberoende parametrar stöds inte ännu");
            }
            let s0 = v[0];
            let back: Vec<(f64, f64)> = (0..nb).map(|k| (v[1 + 2 * k], v[2 + 2 * k])).collect();
            let iso = match cyclic {
                None => Hardening::Table(vec![(0.0, s0)]),
                Some(ck) => {
                    if ck.param_upper("PARAMETERS").is_some() || ck.has("PARAMETERS") {
                        let w = ck.first_floats()?;
                        if w.len() < 3 {
                            return ck.err("PARAMETERS kräver σ|0, Q∞ och b");
                        }
                        Hardening::Voce { s0: w[0], q: w[1], b: w[2] }
                    } else {
                        Hardening::Table(yield_table(ck)?)
                    }
                }
            };
            (iso, back)
        }
        other => return kw.err(format!("HARDENING={other} stöds inte ännu (ISOTROPIC, KINEMATIC, COMBINED)")),
    };
    if !(iso.initial() > 0.0) {
        return kw.err("den initiala flytspänningen ska vara positiv");
    }
    let g = e / (2.0 * (1.0 + nu));
    if !etab.is_empty() && hard != "ISOTROPIC" {
        return kw.err("temperaturberoende elasticitet med kinematiskt hårdnande stöds inte ännu");
    }
    Ok(Some(Arc::new(Mises { e, nu, g, ce: iso_d6(e, nu), iso, back, etab, ytab })))
}

/// C (1 − e^(−γ Δp))/γ (→ C Δp när γ → 0): mottensorns tillväxt under
/// inkrementet med fix flödesriktning.
fn psi(c: f64, gam: f64, dp: f64) -> f64 {
    let x = gam * dp;
    if x.abs() < 1e-8 {
        c * dp * (1.0 - 0.5 * x)
    } else {
        c * (1.0 - (-x).exp()) / gam
    }
}

/// Resultat av returalgoritmen i 3D.
struct Ret {
    sig: V6,
    dp: f64,
    /// Flödesriktning n = 3/2 ξ/q (tensorkomponenter).
    n: V6,
    q_tr: f64,
    xi_tr: V6,
}

impl Mises {
    /// Modellen vid temperaturen `t` (bara när något är temperaturberoende).
    fn at_temperature(&self, t: f64) -> Option<Mises> {
        if self.etab.is_empty() && self.ytab.is_empty() {
            return None;
        }
        let (e, nu) = if self.etab.is_empty() { (self.e, self.nu) } else { super::thermal_elastic::interp_iso(&self.etab, t) };
        let iso = if self.ytab.is_empty() { self.iso.clone() } else { Hardening::Table(super::thermal_elastic::yield_at(&self.ytab, t)) };
        Some(Mises { e, nu, g: e / (2.0 * (1.0 + nu)), ce: iso_d6(e, nu), iso, back: self.back.clone(), etab: Vec::new(), ytab: Vec::new() })
    }

    fn n_back(&self) -> usize {
        self.back.len()
    }

    /// Index för den plastiska töjningen vid senaste *ANNEAL (område coupled2).
    fn anneal_offset(&self) -> usize {
        ALPHA + 6 * self.n_back()
    }

    fn alpha(st: &[f64], k: usize) -> V6 {
        std::array::from_fn(|i| st[ALPHA + 6 * k + i])
    }

    /// Returalgoritmen för total 3D-töjning `eps` från tillståndet `st` (skriver `stn`).
    fn ret(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<Ret> {
        let g = self.g;
        let p_n = st[P];
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        let s_tr = dev(&sig_tr);
        let nb = self.n_back();
        let alphas: Vec<V6> = (0..nb).map(|k| Self::alpha(st, k)).collect();
        let mut xi_tr = s_tr;
        for a in &alphas {
            for i in 0..6 {
                xi_tr[i] -= a[i];
            }
        }
        let q_tr = mises(&xi_tr);
        let (sy, _) = self.iso.eval(p_n);
        stn[..ALPHA + 6 * nb].copy_from_slice(&st[..ALPHA + 6 * nb]);
        if q_tr - sy <= 1e-12 * sy {
            return Ok(Ret { sig: sig_tr, dp: 0.0, n: [0.0; 6], q_tr, xi_tr });
        }
        // ξ̃(Δp) = s_tr − Σ θ_k α_k,n och dess derivata
        let xi_of = |dp: f64| -> (V6, V6) {
            let mut x = s_tr;
            let mut dx = [0.0; 6];
            for (k, a) in alphas.iter().enumerate() {
                let gam = self.back[k].1;
                let th = (-gam * dp).exp();
                for i in 0..6 {
                    x[i] -= a[i] * th;
                    dx[i] += a[i] * gam * th;
                }
            }
            (x, dx)
        };
        let gfun = |dp: f64| -> (f64, f64) {
            let (x, dx) = xi_of(dp);
            let q = mises(&x);
            let dq = if q > 0.0 { 1.5 * ddot(&x, &dx) / q } else { 0.0 };
            let mut c = 3.0 * g * dp;
            let mut dc = 3.0 * g;
            for &(ck, gk) in &self.back {
                c += psi(ck, gk, dp);
                dc += ck * (-gk * dp).exp();
            }
            let (s, h) = self.iso.eval(p_n + dp);
            (q - c - s, dq - dc - h)
        };
        let qa: f64 = alphas.iter().map(mises).sum();
        let mut lo = 0.0;
        let mut hi = (mises(&s_tr) + qa) / (3.0 * g) + 1e-30;
        let mut dp = 0.0;
        let mut conv = false;
        for _ in 0..200 {
            let (gv, dg) = gfun(dp);
            if gv.abs() <= 1e-13 * sy.max(q_tr) {
                conv = true;
                break;
            }
            if gv > 0.0 {
                lo = dp;
            } else {
                hi = dp;
            }
            let mut next = if dg < 0.0 { dp - gv / dg } else { f64::NAN };
            if !(next > lo && next < hi) {
                next = 0.5 * (lo + hi);
            }
            if (hi - lo) <= 1e-14 * hi.max(1e-300) {
                conv = true;
                dp = next;
                break;
            }
            dp = next;
        }
        if !conv {
            return fail("returalgoritmen för von Mises-plasticitet konvergerade inte");
        }
        let (x, _) = xi_of(dp);
        let q = mises(&x);
        let n: V6 = std::array::from_fn(|i| 1.5 * x[i] / q);
        let mut sig = sig_tr;
        for i in 0..6 {
            sig[i] -= 2.0 * g * dp * n[i];
        }
        stn[P] = p_n + dp;
        for i in 0..6 {
            let eng = if i < 3 { 1.0 } else { 2.0 };
            stn[EP + i] = st[EP + i] + eng * dp * n[i];
        }
        for (k, a) in alphas.iter().enumerate() {
            let (ck, gk) = self.back[k];
            let th = (-gk * dp).exp();
            let ps = psi(ck, gk, dp);
            for i in 0..6 {
                stn[ALPHA + 6 * k + i] = a[i] * th + 2.0 / 3.0 * ps * n[i];
            }
        }
        Ok(Ret { sig, dp, n, q_tr, xi_tr })
    }

    /// Spänning och konsistent tangent i 3D.
    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let r = self.ret(eps, st, stn)?;
        if r.dp == 0.0 {
            return Ok((r.sig, self.ce));
        }
        let g = self.g;
        let k = self.e / (3.0 * (1.0 - 2.0 * self.nu));
        if self.back.iter().all(|&(_, gk)| gk == 0.0) {
            // sluten konsistent tangent (Simo–Taylor), riktningen är fix
            let h: f64 = self.iso.eval(st[P] + r.dp).1 + self.back.iter().map(|b| b.0).sum::<f64>();
            let nrm = ddot(&r.xi_tr, &r.xi_tr).sqrt();
            let nn: V6 = std::array::from_fn(|i| r.xi_tr[i] / nrm);
            let theta = 1.0 - 3.0 * g * r.dp / r.q_tr;
            let c = 6.0 * g * g * (r.dp / r.q_tr - 1.0 / (3.0 * g + h));
            let mut d = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let mm = if i < 3 && j < 3 { 1.0 } else { 0.0 };
                    let pdev = if i == j { if i < 3 { 1.0 } else { 0.5 } } else { 0.0 } - mm / 3.0;
                    d[i][j] = k * mm + 2.0 * g * theta * pdev + c * nn[i] * nn[j];
                }
            }
            let _ = r.n;
            return Ok((r.sig, d));
        }
        // numerisk konsistent tangent (framåtdifferens av returalgoritmen)
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.iso.initial() / self.e);
        let hstep = 1e-7 * scale;
        let mut d = [[0.0; 6]; 6];
        let mut tmp = stn.to_vec();
        for j in 0..6 {
            let mut e2 = *eps;
            e2[j] += hstep;
            let r2 = self.ret(&e2, st, &mut tmp)?;
            for i in 0..6 {
                d[i][j] = (r2.sig[i] - r.sig[i]) / hstep;
            }
        }
        // symmetrisera (exakt tangent är symmetrisk för associativ plasticitet)
        for i in 0..6 {
            for j in 0..i {
                let m = 0.5 * (d[i][j] + d[j][i]);
                d[i][j] = m;
                d[j][i] = m;
            }
        }
        Ok((r.sig, d))
    }
}

impl MaterialModel for Mises {
    fn name(&self) -> &str {
        "PLASTIC (von Mises)"
    }

    fn n_state(&self) -> usize {
        // område coupled2: sist den plastiska töjningen vid senaste *ANNEAL (6)
        ALPHA + 6 * self.n_back() + 6
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        // plastisk töjning, total 3D-töjning och mottensorer följer materialet
        rotate_v6(r, &mut state[EP..EP + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
        for k in 0..self.n_back() {
            rotate_v6(r, &mut state[ALPHA + 6 * k..ALPHA + 6 * k + 6], false);
        }
        let a0 = self.anneal_offset();
        rotate_v6(r, &mut state[a0..a0 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let a0 = self.anneal_offset();
        state_new[a0..a0 + 6].copy_from_slice(&state_old[a0..a0 + 6]);
        if let Some(m) = self.at_temperature(input.temperature) {
            return m.update(input, state_old, state_new);
        }
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        // sista anropet i update_reduced gjordes med eps3: state_new är aktuellt
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[P]]
    }

    /// *ANNEAL: den plastiska töjningen blir den totala (spänningsfritt),
    /// PEEQ och mottensorerna nollställs (`steps/anneal.rs`). PE redovisas
    /// därefter från glödgningen (PE = 0 direkt efter): den
    /// plastiska töjningen vid glödgningen sparas sist i tillståndet.
    fn anneal(&self, strain: &[f64; 6], state: &mut [f64]) -> bool {
        state[P] = 0.0;
        state[EP..EP + 6].copy_from_slice(strain);
        state[EPS3..EPS3 + 6].copy_from_slice(strain);
        let a0 = self.anneal_offset();
        state[ALPHA..a0].iter_mut().for_each(|x| *x = 0.0);
        state[a0..a0 + 6].copy_from_slice(strain);
        true
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[P]];
        let a0 = self.anneal_offset();
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i] - state.get(a0 + i).copied().unwrap_or(0.0)));
        if mode == StressMode::PlaneStress {
            // PE33 = 0 redovisas för plan spänning (som E33), kontrollerat
            v[3] = 0.0;
        }
        v
    }
}
