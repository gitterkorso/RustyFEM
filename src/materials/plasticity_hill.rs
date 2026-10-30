//! Utökad metallplasticitet: Hills anisotropa flytvillkor
//! (*POTENTIAL under *PLASTIC), hastighetsberoende flytspänning
//! (*RATE DEPENDENT), Johnson–Cooks hårdnande (*PLASTIC, HARDENING=JOHNSON
//! COOK) och von Mises med anisotrop elasticitet. Rena von Mises-material
//! med isotrop elasticitet hanteras av `plasticity.rs` som förut; den här
//! modellen väljs bara när något av ovanstående finns (`claims`).
//!
//! Flytvillkoret (Hill 1948, tensorkomponenter, ξ = σ − α):
//!
//!   f(ξ)² = F(ξ22−ξ33)² + G(ξ33−ξ11)² + H(ξ11−ξ22)² + 2Lξ23² + 2Mξ13² + 2Nξ12² = ξᵀPξ
//!
//! med förhållandena Rij (*POTENTIAL: R11, R22, R33, R12, R13, R23):
//! F = ½(1/R22² + 1/R33² − 1/R11²), G = ½(1/R33² + 1/R11² − 1/R22²),
//! H = ½(1/R11² + 1/R22² − 1/R33²), L = 3/(2R23²), M = 3/(2R13²),
//! N = 3/(2R12²). Utan *POTENTIAL är alla Rij = 1 (von Mises).
//!
//! Associativt flöde: Δεp = Δλ Pξ/f (ingenjörsskjuvning, eftersom ∂f/∂σ i
//! Voigt-form direkt ger ingenjörskomponenterna), och plastiskt arbete ger
//! ε̄̇p = λ̇ (σy ε̄̇p = ξ:ε̇p). Mottensorerna följer Ziegler/Chaboche,
//! α̇ₖ = Cₖ/σy dev(ξ) ε̄̇p − γₖ αₖ ε̄̇p, integrerat exakt för fix
//! riktning under inkrementet: αₖ = θₖ αₖ,ₙ + ψₖ dev(ξ)/σy med θₖ = e^(−γₖΔλ)
//! och ψₖ = Cₖ(1 − θₖ)/γₖ (för von Mises samma schema som `plasticity.rs`).
//!
//! Implicit Euler (närmaste-punkt-projektion): med prövospänningen
//! σ_tr = D(ε − εp,ₙ) och σ = σ_tr − (Δλ/σy) D P ξ blir ξ lösningen till det
//! linjära systemet A ξ = b, A = I + (Δλ D P + Ψ Q)/σy, b = σ_tr − Σθₖαₖ,ₙ
//! (Q = deviatorprojektionen, Ψ = Σψₖ). Kvar är den skalära ekvationen
//!
//!   φ(Δλ) = f(ξ(Δλ)) − σy(ε̄p,ₙ + Δλ, Δλ/Δt, T) = 0,
//!
//! som löses med Newton inom säkra gränser. Den konsistenta tangenten
//! dσ/dε är sluten (implicit derivering av A ξ = b och φ = 0), även med
//! hastighetsberoende, temperaturberoende och olinjärt kinematiskt hårdnande.
//!
//! Elasticiteten är isotrop (även temperaturberoende E, ν, T) eller
//! anisotrop (*ELASTIC, TYPE=ENGINEERING CONSTANTS/ORTHOTROPIC/ANISOTROPIC,
//! via `anisotropic.rs`). Hills förhållanden och den anisotropa elasticiteten
//! gäller i materialets lokala axlar (*ORIENTATION, som kontinuumelementen
//! och skalen redan roterar till). Plan spänning och stänger går via
//! `plasticity::update_reduced` (lokal Newton på de spänningsfria
//! komponenterna).
//!
//! Utdata: PEEQ och PE som von Mises-modellen. Tillståndslayouten är
//! densamma som i `plasticity.rs`: [ε̄p, εp (6), ε 3D (6), αₖ (6 per mottensor)].

use super::anisotropic::{invert, reduce_6x6};
use super::plasticity::{iso_d6, pe_labels, rotate_v6, update_reduced, Hardening, M6, V6};
use super::plasticity_rate::{JohnsonCook, RateLaw};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "metallplasticitet (Hill, hastighetsberoende, Johnson–Cook)",
    consumes: &["ELASTIC", "PLASTIC", "CYCLIC HARDENING", "POTENTIAL", "RATE DEPENDENT"],
    priority: 15,
    build: Some(build),
}];

const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const ALPHA: usize = 13;

/// Ska materialet hanteras av den här modellen (och inte av `plasticity.rs`)?
pub fn claims(def: &MaterialDef) -> bool {
    let Some(pk) = def.behavior("PLASTIC") else { return false };
    if def.behavior("POTENTIAL").is_some() || def.behavior("RATE DEPENDENT").is_some() {
        return true;
    }
    // område materials4: flytkurvor per töjningshastighet (*PLASTIC, RATE=)
    if def.behaviors.iter().any(|b| b.name == "PLASTIC" && b.has("RATE")) {
        return true;
    }
    if pk.param_upper("HARDENING").as_deref() == Some("JOHNSON COOK") {
        return true;
    }
    def.behavior("ELASTIC").is_some_and(|e| e.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC"))
}

/// Hills matris P (Voigt med tensorskjuvkomponenter, f² = ξᵀPξ) ur Rij.
pub fn hill_matrix(r: &[f64; 6]) -> M6 {
    let inv2 = |x: f64| 1.0 / (x * x);
    let (a, b, c) = (inv2(r[0]), inv2(r[1]), inv2(r[2]));
    let f = 0.5 * (b + c - a);
    let g = 0.5 * (c + a - b);
    let h = 0.5 * (a + b - c);
    let (l, m, n) = (1.5 * inv2(r[5]), 1.5 * inv2(r[4]), 1.5 * inv2(r[3]));
    let mut p = [[0.0; 6]; 6];
    p[0][0] = g + h;
    p[1][1] = f + h;
    p[2][2] = f + g;
    p[0][1] = -h;
    p[1][0] = -h;
    p[0][2] = -g;
    p[2][0] = -g;
    p[1][2] = -f;
    p[2][1] = -f;
    p[3][3] = 2.0 * n;
    p[4][4] = 2.0 * m;
    p[5][5] = 2.0 * l;
    p
}

#[derive(Debug, Clone)]
enum Elastic {
    /// (E, ν, T) i stigande temperatur (en post om konstant).
    Iso(Vec<(f64, f64, f64)>),
    Aniso(M6),
}

impl Elastic {
    fn d_at(&self, t: f64) -> M6 {
        match self {
            Elastic::Iso(tab) => {
                let (e, nu) = super::thermal_elastic::interp_iso(tab, t);
                iso_d6(e, nu)
            }
            Elastic::Aniso(d) => *d,
        }
    }
}

#[derive(Debug, Clone)]
enum Iso {
    Fixed(Hardening),
    /// Temperaturberoende tabeller (T, [(ε̄p, σ0)]).
    Temp(Vec<(f64, Vec<(f64, f64)>)>),
    JohnsonCook(JohnsonCook),
}

/// Isotropt hårdnande vid en given temperatur.
enum IsoAt<'a> {
    H(&'a Hardening),
    Owned(Hardening),
    Jc(JohnsonCook, f64),
}

impl IsoAt<'_> {
    fn eval(&self, p: f64) -> (f64, f64) {
        match self {
            IsoAt::H(h) => h.eval(p),
            IsoAt::Owned(h) => h.eval(p),
            IsoAt::Jc(j, t) => j.eval(p, *t),
        }
    }
}

/// Hill-/Mises-plasticitet med isotropt, kinematiskt och hastighetsberoende hårdnande.
#[derive(Debug, Clone)]
pub struct MetalPlastic {
    elastic: Elastic,
    /// Elasticitetsmatrisen vid referenstemperaturen (linjär styvhet).
    ce: M6,
    hill: M6,
    iso: Iso,
    /// Mottensorer (C, γ).
    back: Vec<(f64, f64)>,
    rate: RateLaw,
    /// Område materials4: flytkurvor per töjningshastighet (*PLASTIC, RATE=),
    /// (ε̄̇p, tabell) i stigande hastighet; tom = ingen.
    rate_tabs: Vec<(f64, Vec<(f64, f64)>)>,
    /// Område materials4: temperaturberoende *POTENTIAL, (T, Rij) i stigande T; tom = konstant.
    hill_t: Vec<(f64, [f64; 6])>,
}

/// Mellanresultat för ett värde på Δλ.
struct Ev {
    xi: V6,
    ainv: [f64; 36],
    f: f64,
    sy: f64,
    sy_d: f64,
    phi: f64,
    dphi: f64,
    /// ∂ξ/∂Δλ.
    xi_d: V6,
}

fn mm(a: &M6, b: &M6) -> M6 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..6).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mv(a: &M6, x: &V6) -> V6 {
    std::array::from_fn(|i| (0..6).map(|j| a[i][j] * x[j]).sum())
}

fn mvf(a: &[f64; 36], x: &V6) -> V6 {
    std::array::from_fn(|i| (0..6).map(|j| a[i * 6 + j] * x[j]).sum())
}

fn dot(a: &V6, b: &V6) -> f64 {
    (0..6).map(|i| a[i] * b[i]).sum()
}

/// Deviatorprojektionen (Voigt, tensorkomponenter).
fn qdev() -> M6 {
    let mut q = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            q[i][j] = if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 };
        }
        q[i + 3][i + 3] = 1.0;
    }
    q
}

/// C (1 − e^(−γ Δλ))/γ (→ C Δλ när γ → 0).
fn psi(c: f64, gam: f64, dl: f64) -> f64 {
    let x = gam * dl;
    if x.abs() < 1e-8 {
        c * dl * (1.0 - 0.5 * x)
    } else {
        c * (1.0 - (-x).exp()) / gam
    }
}

fn floats_all(kw: &RawKeyword) -> Result<Vec<f64>> {
    let mut v = Vec::new();
    for d in kw.records() {
        v.extend(kw.floats(&d)?);
    }
    Ok(v)
}

fn single_table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let t = super::thermal_elastic::yield_tables(kw)?;
    if t.len() > 1 {
        return kw.err("temperaturberoende data stöds inte här ännu");
    }
    Ok(t[0].1.clone())
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    let pk = def.behavior("PLASTIC").unwrap();
    let Some(ek) = def.behavior("ELASTIC") else {
        return fail(format!("material {}: *PLASTIC kräver *ELASTIC", def.name));
    };
    if ek.has("TEMPERATURE") {
        return ek.err("parametern TEMPERATURE stöds inte ännu");
    }
    let ety = ek.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    let elastic = if ety == "ISOTROPIC" {
        Elastic::Iso(super::thermal_elastic::iso_table(ek)?)
    } else {
        if ety == "LAMINA" {
            return ek.err("plasticitet med *ELASTIC, TYPE=LAMINA stöds inte (använd ENGINEERING CONSTANTS)");
        }
        let only = MaterialDef { name: def.name.clone(), behaviors: vec![ek.clone()], density: def.density };
        let Some(m) = (super::anisotropic::FACTORIES[0].build.unwrap())(&only)? else {
            return ek.err(format!("TYPE={ety} stöds inte"));
        };
        let d = m.tangent_linear(StressMode::ThreeD)?;
        Elastic::Aniso(std::array::from_fn(|i| std::array::from_fn(|j| d[i * 6 + j])))
    };
    let ce = elastic.d_at(match &elastic {
        Elastic::Iso(t) => t[0].2,
        _ => 0.0,
    });
    if pk.has("DEPENDENCIES") {
        return pk.err("parametern DEPENDENCIES stöds inte ännu");
    }
    let rate_tabs = super::plasticity_rate_tables::rate_tables(def)?;
    let hard = pk.param_upper("HARDENING").unwrap_or_else(|| "ISOTROPIC".into());
    let cyclic = def.behavior("CYCLIC HARDENING");
    if cyclic.is_some() && hard != "COMBINED" {
        return pk.err("*CYCLIC HARDENING kräver HARDENING=COMBINED");
    }
    let (iso, back) = match hard.as_str() {
        "ISOTROPIC" => {
            let tabs = super::thermal_elastic::yield_tables(pk)?;
            if tabs.len() > 1 {
                (Iso::Temp(tabs), Vec::new())
            } else {
                (Iso::Fixed(Hardening::Table(tabs[0].1.clone())), Vec::new())
            }
        }
        "JOHNSON COOK" => (Iso::JohnsonCook(JohnsonCook::parse(pk)?), Vec::new()),
        "KINEMATIC" => {
            let t = single_table(pk)?;
            if t.len() > 2 {
                return pk.err("linjärt kinematiskt hårdnande tar högst två datapunkter (σ0, 0) och (σ1, εp1)");
            }
            let c = if t.len() == 2 { (t[1].1 - t[0].1) / t[1].0 } else { 0.0 };
            (Iso::Fixed(Hardening::Table(vec![t[0]])), vec![(c, 0.0)])
        }
        "COMBINED" => {
            let dt = pk.param_upper("DATATYPE").unwrap_or_else(|| "HALF CYCLE".into());
            if dt != "PARAMETERS" {
                return pk.err(format!("HARDENING=COMBINED stöds bara med DATATYPE=PARAMETERS (inte {dt})"));
            }
            let nb = match pk.param("NUMBER BACKSTRESSES") {
                Some(s) => s
                    .trim()
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| (1..=10).contains(&n))
                    .ok_or_else(|| FemError(format!("{}: NUMBER BACKSTRESSES={s} ska vara 1–10", pk.location())))?,
                None => 1,
            };
            if pk.records().len() > 1 {
                return pk.err("temperaturberoende parametrar (flera datarader) stöds inte ännu");
            }
            let v = floats_all(pk)?;
            if v.len() < 1 + 2 * nb {
                return pk.err(format!("dataraden ska ha σ0 och {nb} par (C, γ)"));
            }
            if v.len() > 1 + 2 * nb && v[1 + 2 * nb..].iter().any(|&x| x != 0.0) {
                return pk.err("temperaturberoende parametrar stöds inte ännu");
            }
            let back: Vec<(f64, f64)> = (0..nb).map(|k| (v[1 + 2 * k], v[2 + 2 * k])).collect();
            let iso = match cyclic {
                None => Hardening::Table(vec![(0.0, v[0])]),
                Some(ck) => {
                    if ck.has("PARAMETERS") {
                        let w = ck.first_floats()?;
                        if w.len() < 3 {
                            return ck.err("PARAMETERS kräver σ|0, Q∞ och b");
                        }
                        Hardening::Voce { s0: w[0], q: w[1], b: w[2] }
                    } else {
                        Hardening::Table(single_table(ck)?)
                    }
                }
            };
            (Iso::Fixed(iso), back)
        }
        other => return pk.err(format!("HARDENING={other} stöds inte ännu (ISOTROPIC, KINEMATIC, COMBINED, JOHNSON COOK)")),
    };
    if !back.is_empty() && matches!(&elastic, Elastic::Iso(t) if t.len() > 1) {
        return pk.err("temperaturberoende elasticitet med kinematiskt hårdnande stöds inte ännu");
    }
    let hill = match def.behavior("POTENTIAL") {
        None => hill_matrix(&[1.0; 6]),
        Some(pot) => {
            if pot.has("DEPENDENCIES") {
                return pot.err("fältberoende *POTENTIAL stöds inte ännu");
            }
            let v = floats_all(pot)?;
            if v.len() < 6 {
                return pot.err("*POTENTIAL kräver R11, R22, R33, R12, R13 och R23");
            }
            let r: [f64; 6] = std::array::from_fn(|i| v[i]);
            if r.iter().any(|&x| !(x > 0.0)) {
                return pot.err("förhållandena Rij ska vara positiva");
            }
            let p = hill_matrix(&r);
            // flytytan ska vara sluten i deviatorrummet: F+G, F+H, G+H > 0 och FG+GH+HF > 0
            let (f, g, h) = (-p[1][2], -p[0][2], -p[0][1]);
            if !(f + g > 0.0 && f + h > 0.0 && g + h > 0.0 && f * g + g * h + h * f > 0.0) {
                return pot.err("förhållandena Rij ger ingen sluten flytyta (Hill-konstanterna F, G, H)");
            }
            p
        }
    };
    let rate = match def.behavior("RATE DEPENDENT") {
        None => RateLaw::None,
        Some(rk) => RateLaw::parse(rk)?,
    };
    let hill_t = match def.behavior("POTENTIAL") {
        Some(pot) if pot.records().len() > 1 => super::plasticity_rate_tables::potential_table(pot)?,
        _ => Vec::new(),
    };
    if !rate_tabs.is_empty() && !matches!(rate, RateLaw::None) {
        return pk.err("*PLASTIC, RATE= och *RATE DEPENDENT kan inte kombineras");
    }
    let m = MetalPlastic { elastic, ce, hill, iso, back, rate, rate_tabs, hill_t };
    let s0 = m.iso_at(0.0).eval(0.0).0;
    let s0 = match &m.iso {
        Iso::JohnsonCook(j) => j.a,
        _ => s0,
    };
    if !(s0 > 0.0) {
        return pk.err("den initiala flytspänningen ska vara positiv");
    }
    Ok(Some(Arc::new(m)))
}

impl MetalPlastic {
    fn iso_at(&self, t: f64) -> IsoAt<'_> {
        match &self.iso {
            Iso::Fixed(h) => IsoAt::H(h),
            Iso::Temp(tabs) => IsoAt::Owned(Hardening::Table(super::thermal_elastic::yield_at(tabs, t))),
            Iso::JohnsonCook(j) => IsoAt::Jc(*j, t),
        }
    }

    fn nb(&self) -> usize {
        self.back.len()
    }

    /// Flytspänning (σy, dσy/dΔλ) vid Δλ.
    fn yield_stress(&self, iso: &IsoAt, p_n: f64, dl: f64, dt: f64) -> (f64, f64) {
        if !self.rate_tabs.is_empty() {
            return super::plasticity_rate_tables::yield_stress(&self.rate_tabs, p_n + dl, dl, dt);
        }
        let (s, h) = iso.eval(p_n + dl);
        if self.rate.is_none() {
            return (s, h);
        }
        if dt <= 0.0 {
            // obegränsad töjningshastighet
            let r = self.rate.ratio_infinite().unwrap_or(f64::INFINITY);
            return (s * r, h * r);
        }
        let (r, dr) = self.rate.ratio(dl / dt);
        (s * r, h * r + s * dr / dt)
    }

    /// Returalgoritmen och den konsistenta tangenten i 3D.
    fn update3d(&self, d: &M6, iso: &IsoAt, pmat: &M6, dt: f64, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let nb = self.nb();
        let p_n = st[P];
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv(d, &ee);
        let alphas: Vec<V6> = (0..nb).map(|k| std::array::from_fn(|i| st[ALPHA + 6 * k + i])).collect();
        let mut xi_tr = sig_tr;
        for a in &alphas {
            for i in 0..6 {
                xi_tr[i] -= a[i];
            }
        }
        let f_tr = dot(&xi_tr, &mv(pmat, &xi_tr)).max(0.0).sqrt();
        let (sy0, _) = self.yield_stress(iso, p_n, 0.0, dt);
        stn[..ALPHA + 6 * nb].copy_from_slice(&st[..ALPHA + 6 * nb]);
        if f_tr - sy0 <= 1e-12 * sy0 || sy0 <= 0.0 && f_tr == 0.0 || sy0.is_infinite() {
            return Ok((sig_tr, *d));
        }
        let dp = mm(d, pmat);
        let q = qdev();
        let eval = |dl: f64| -> Result<Ev> {
            let (sy, sy_d) = self.yield_stress(iso, p_n, dl, dt);
            let sy = sy.max(1e-300);
            let mut ps = 0.0;
            let mut ps_d = 0.0;
            let mut b = sig_tr;
            let mut b_d = [0.0; 6];
            for (k, a) in alphas.iter().enumerate() {
                let (ck, gk) = self.back[k];
                let th = (-gk * dl).exp();
                ps += psi(ck, gk, dl);
                ps_d += ck * th;
                for i in 0..6 {
                    b[i] -= th * a[i];
                    b_d[i] += gk * th * a[i];
                }
            }
            let mut amat = [0.0; 36];
            let mut a_d = [[0.0; 6]; 6];
            for i in 0..6 {
                for j in 0..6 {
                    let num = dl * dp[i][j] + ps * q[i][j];
                    amat[i * 6 + j] = if i == j { 1.0 } else { 0.0 } + num / sy;
                    a_d[i][j] = (dp[i][j] + ps_d * q[i][j]) / sy - num * sy_d / (sy * sy);
                }
            }
            let ainv: [f64; 36] = invert(&amat, 6)
                .ok_or_else(|| FemError("returalgoritmen för Hill-plasticitet: singulärt system".into()))?
                .try_into()
                .unwrap();
            let xi = mvf(&ainv, &b);
            let pxi = mv(pmat, &xi);
            let f = dot(&xi, &pxi).max(0.0).sqrt();
            let rhs: V6 = std::array::from_fn(|i| b_d[i] - (0..6).map(|j| a_d[i][j] * xi[j]).sum::<f64>());
            let xi_d = mvf(&ainv, &rhs);
            let n: V6 = std::array::from_fn(|i| if f > 0.0 { pxi[i] / f } else { 0.0 });
            let dphi = dot(&n, &xi_d) - sy_d;
            Ok(Ev { xi, ainv, f, sy, sy_d, phi: f - sy, dphi, xi_d })
        };
        // gränser: φ(0) > 0; sök en övre gräns där φ < 0
        let gref = d[3][3].min(d[4][4]).min(d[5][5]).max(1e-300);
        let mut lo = 0.0;
        let mut hi = (f_tr - sy0) / (3.0 * gref) + 1e-300;
        let mut ev_hi = eval(hi)?;
        let mut grow = 0;
        while ev_hi.phi > 0.0 {
            lo = hi;
            hi *= 4.0;
            ev_hi = eval(hi)?;
            grow += 1;
            if grow > 200 {
                return fail("returalgoritmen för Hill-plasticitet hittade ingen övre gräns för Δλ");
            }
        }
        let tol = 1e-13 * sy0.max(f_tr);
        let mut dl = if grow == 0 { 0.5 * hi } else { 0.5 * (lo + hi) };
        let mut ev = eval(dl)?;
        let mut conv = false;
        for _ in 0..300 {
            if ev.phi.abs() <= tol {
                conv = true;
                break;
            }
            if ev.phi > 0.0 {
                lo = dl;
            } else {
                hi = dl;
            }
            if hi - lo <= 1e-15 * hi {
                conv = true;
                break;
            }
            let mut next = if ev.dphi < 0.0 { dl - ev.phi / ev.dphi } else { f64::NAN };
            if !(next > lo && next < hi) {
                next = 0.5 * (lo + hi);
            }
            dl = next;
            ev = eval(dl)?;
        }
        if !conv {
            return fail("returalgoritmen för Hill-plasticitet konvergerade inte");
        }
        let Ev { xi, ainv, f, sy, sy_d, xi_d, .. } = ev;
        let pxi = mv(pmat, &xi);
        let c = dl / sy;
        let dpxi = mv(&dp, &xi);
        let sig: V6 = std::array::from_fn(|i| sig_tr[i] - c * dpxi[i]);
        stn[P] = p_n + dl;
        // Δεp = Δλ Pξ/f (ingenjörskomponenter)
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dl * pxi[i] / f;
        }
        let xdev = mv(&q, &xi);
        for (k, a) in alphas.iter().enumerate() {
            let (ck, gk) = self.back[k];
            let th = (-gk * dl).exp();
            let ps = psi(ck, gk, dl);
            for i in 0..6 {
                stn[ALPHA + 6 * k + i] = th * a[i] + ps * xdev[i] / sy;
            }
        }
        // konsistent tangent
        let n: V6 = std::array::from_fn(|i| pxi[i] / f);
        let kappa = sy_d - dot(&n, &xi_d);
        // r = (nᵀ A⁻¹ D)ᵀ
        let at_n: V6 = std::array::from_fn(|j| (0..6).map(|i| n[i] * ainv[i * 6 + j]).sum());
        let r: V6 = std::array::from_fn(|j| (0..6).map(|i| at_n[i] * d[i][j]).sum());
        let c_l = 1.0 / sy - dl * sy_d / (sy * sy);
        // u = D P (ξ c_λ + c ξ')
        let w: V6 = std::array::from_fn(|i| xi[i] * c_l + c * xi_d[i]);
        let u = mv(&dp, &w);
        // D − c D P A⁻¹ D − u rᵀ/κ
        let mut ad = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                ad[i][j] = (0..6).map(|k| ainv[i * 6 + k] * d[k][j]).sum();
            }
        }
        let dpad = mm(&dp, &ad);
        let mut tan = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                tan[i][j] = d[i][j] - c * dpad[i][j] - if kappa != 0.0 { u[i] * r[j] / kappa } else { 0.0 };
            }
        }
        Ok((sig, tan))
    }
}

impl MaterialModel for MetalPlastic {
    fn name(&self) -> &str {
        "PLASTIC (Hill/hastighetsberoende)"
    }

    fn n_state(&self) -> usize {
        ALPHA + 6 * self.nb()
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        match &self.elastic {
            Elastic::Iso(t) => Some((t[0].0, t[0].1)),
            Elastic::Aniso(_) => None,
        }
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[EP..EP + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
        for k in 0..self.nb() {
            rotate_v6(r, &mut state[ALPHA + 6 * k..ALPHA + 6 * k + 6], false);
        }
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let t = input.temperature;
        let d = self.elastic.d_at(t);
        let iso = self.iso_at(t);
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let pmat = if self.hill_t.is_empty() { self.hill } else { hill_matrix(&super::plasticity_rate_tables::potential_at(&self.hill_t, t)) };
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(&d, &iso, &pmat, input.dt, e, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &d, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[P]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[P]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v
    }
}
