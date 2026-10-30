//! Viskoelasticitet i tidsplanet (*VISCOELASTIC, TIME=PRONY) med *ELASTIC
//! (små töjningar) eller *HYPERELASTIC (stora töjningar).
//!
//! Relaxationsfunktionerna är Prony-serier med gemensamma relaxationstider:
//!
//!   G(t) = G0 (1 − Σ gᵢ (1 − e^(−t/τᵢ))),  K(t) = K0 (1 − Σ kᵢ (1 − e^(−t/τᵢ))),
//!
//! dataraderna är (gᵢ, kᵢ, τᵢ). Elasticitetskonstanterna i *ELASTIC och
//! *HYPERELASTIC är långtidsvärden (MODULI=LONG TERM, standard för
//! båda) eller momentana (MODULI=INSTANTANEOUS): G0 = G∞/(1 − Σgᵢ) och
//! K0 = K∞/(1 − Σkᵢ).
//!
//! **Små töjningar**: deviatorn är
//! s = 2G0 (e − Σ gᵢ eᵢ) och trycket −p = K0 (φ − Σ kᵢ φᵢ) med de viskösa
//! töjningarna eᵢ, φᵢ. Med töjningen linjär i tiden under inkrementet blir
//!
//!   Δeᵢ = aᵢ Δe + (1 − e^(−Δt/τᵢ)) (eₙ − eᵢ,ₙ),  aᵢ = 1 − τᵢ/Δt (1 − e^(−Δt/τᵢ))
//!
//! (aᵢ ≈ Δt/(2τᵢ) för Δt/τᵢ < 10⁻⁷) och tangenten G_T = G0 (1 − Σ gᵢ aᵢ),
//! K_T = K0 (1 − Σ kᵢ aᵢ). Plan spänning och stänger löses med 3D-uppdateringen
//! och lokal Newton på de spänningsfria komponenterna (`plasticity::update_reduced`).
//!
//! **Stora töjningar**:
//! den momentana Kirchhoffspänningen τ0 = τ0ᴰ(F̄) + τ0ᴴ(J) I kommer från den
//! hyperelastiska modellen (med momentana konstanter) och de inre
//! spänningarna integreras med framåtförd deviatordel:
//!
//!   τᵢᴰ(t+Δt) = αᵢ gᵢ τ0ᴰ(t+Δt) + βᵢ gᵢ ΔF̄ τ0ᴰ(t) ΔF̄ᵀ + γᵢ ΔF̄ τᵢᴰ(t) ΔF̄ᵀ,
//!   τᵢᴴ(t+Δt) = αᵢ kᵢ τ0ᴴ(t+Δt) + βᵢ kᵢ τ0ᴴ(t) + γᵢ τᵢᴴ(t),
//!
//! γᵢ = e^(−Δt/τᵢ), αᵢ = 1 − τᵢ/Δt (1 − γᵢ), βᵢ = τᵢ/Δt (1 − γᵢ) − γᵢ, ΔF̄ =
//! F̄(t+Δt) F̄(t)⁻¹, och τ = τ0 − Σ dev τᵢᴰ − Σ τᵢᴴ I. Tangenten är
//! Jaumann-tangenten C^(τJ)/J av hela uppdateringen (numerisk perturbation av
//! F som i `hyperelastic_split.rs`), samma konvention som `hyperelastic.rs`.
//! Plan spänning stöds inte med stora töjningar.
//!
//! Läget per steg (`time_dependence.rs`): *VISCO och *DYNAMIC integrerar i
//! tiden; *STATIC ger det momentana svaret (Δt = 0: de inre töjningarna/
//! spänningarna ändras inte, utom framföringen); *STATIC, LONG TERM ger den
//! fullt relaxerade lösningen i varje inkrement (bara stegets slut är
//! meningsfullt).
//!
//! Felmåttet för CETOL är skillnaden i viskös töjningsökning
//! beräknad med töjningshastigheterna vid inkrementets början och slut:
//! Δē = Δt Σ ((gᵢ s0 − sᵢ)ₙ₊₁ − (gᵢ s0 − sᵢ)ₙ)/(2G0 τᵢ) och motsvarande Δφ̄ för
//! volymdelen, Δε = √(2/3 Δē:Δē + 1/3 Δφ̄²).

use super::anisotropic::reduce_6x6;
use super::hyperelastic_split::{jaumann_tangent, voigt, HyperModel};
use super::plasticity::{update_reduced, M6, V6};
use super::time_dependence::{TimeDependence, VISCO_ACTIVE, VISCO_LONG_TERM};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::kinematics::tensor::{self as t, M3};
use crate::model::{fail, Result};
use std::cell::Cell;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[
    MaterialFactory {
        name: "viskoelasticitet (*VISCOELASTIC, TIME=PRONY) med *ELASTIC",
        consumes: &["ELASTIC", "VISCOELASTIC"],
        priority: 20,
        build: Some(build_small),
    },
    MaterialFactory {
        name: "viskoelasticitet (*VISCOELASTIC, TIME=PRONY) med *HYPERELASTIC",
        consumes: &["HYPERELASTIC", "VISCOELASTIC"],
        priority: 20,
        build: Some(build_finite),
    },
];

/// Prony-serien: (gᵢ, kᵢ, τᵢ).
#[derive(Debug, Clone)]
pub struct Prony {
    pub terms: Vec<(f64, f64, f64)>,
}

impl Prony {
    pub fn parse(kw: &RawKeyword) -> Result<Prony> {
        if kw.has("FREQUENCY") {
            return kw.err("*VISCOELASTIC, FREQUENCY= (frekvensplanet) stöds inte ännu");
        }
        match kw.param_upper("TIME").as_deref() {
            Some("PRONY") => {}
            Some(o) => return kw.err(format!("TIME={o} stöds inte ännu (bara TIME=PRONY)")),
            None => return kw.err("*VISCOELASTIC kräver TIME=PRONY (andra former stöds inte ännu)"),
        }
        for p in ["NONLINEAR", "TYPE", "PRELOAD"] {
            if kw.has(p) && !(p == "TYPE" && kw.param_upper("TYPE").as_deref() == Some("ISOTROPIC")) {
                return kw.err(format!("parametern {p} stöds inte ännu"));
            }
        }
        let mut terms = Vec::new();
        for d in kw.records() {
            let v = kw.floats(&d)?;
            let g = v.first().copied().unwrap_or(0.0);
            let k = v.get(1).copied().unwrap_or(0.0);
            let tau = v.get(2).copied().unwrap_or(0.0);
            if !(g >= 0.0 && k >= 0.0 && tau > 0.0) {
                return kw.err_at(d.line, "Prony-termen kräver gᵢ ≥ 0, kᵢ ≥ 0 och τᵢ > 0");
            }
            terms.push((g, k, tau));
        }
        if terms.is_empty() {
            return kw.err("Prony-serien saknas");
        }
        let p = Prony { terms };
        if !(p.sum_g() < 1.0 && p.sum_k() < 1.0) {
            return kw.err("summan av gᵢ och summan av kᵢ ska vara mindre än 1");
        }
        Ok(p)
    }

    pub fn sum_g(&self) -> f64 {
        self.terms.iter().map(|x| x.0).sum()
    }

    pub fn sum_k(&self) -> f64 {
        self.terms.iter().map(|x| x.1).sum()
    }
}

/// Integrationskoefficienter (α, β, γ) för en term med relaxationstiden `tau`
/// och tidsinkrementet `dt` (∞ = fullt relaxerat).
fn coeffs(dt: f64, tau: f64) -> (f64, f64, f64) {
    if dt.is_infinite() {
        return (1.0, 0.0, 0.0);
    }
    let x = dt / tau;
    if x < 1e-7 {
        return (0.5 * x, 0.5 * x, 1.0 - x);
    }
    let g = (-x).exp();
    let a = 1.0 - (1.0 - g) / x;
    (a, (1.0 - g) / x - g, g)
}

/// Inkrementets tid för materialet enligt stegets läge.
fn effective_dt(td: &TimeDependence, dt: f64) -> f64 {
    match td.visco() {
        VISCO_ACTIVE => dt.max(0.0),
        VISCO_LONG_TERM => f64::INFINITY,
        _ => 0.0,
    }
}

/// Om MODULI=INSTANTANEOUS anges (annars LONG TERM, standard).
fn instantaneous(kw: &RawKeyword) -> Result<bool> {
    match kw.param_upper("MODULI").as_deref() {
        None | Some("LONG TERM") => Ok(false),
        Some("INSTANTANEOUS") => Ok(true),
        Some(o) => kw.err(format!("MODULI={o} ska vara LONG TERM eller INSTANTANEOUS")),
    }
}

fn without_moduli(kw: &RawKeyword) -> RawKeyword {
    let mut k = kw.clone();
    k.params.retain(|(p, _)| p != "MODULI");
    k
}

// ---------------------------------------------------------------------------
// Små töjningar
// ---------------------------------------------------------------------------

/// Linjär viskoelasticitet med isotrop elasticitet.
#[derive(Debug)]
pub struct ViscoSmall {
    /// Momentana moduler.
    g0: f64,
    k0: f64,
    prony: Prony,
    td: TimeDependence,
}

/// Tillståndslayout: [ε 3D (6), per term: eᵢ (6, tensorkomponenter), φᵢ].
const EPS: usize = 0;
const TERM: usize = 6;
const TSTRIDE: usize = 7;

fn build_small(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(vk) = def.behavior("VISCOELASTIC") else { return Ok(None) };
    if super::viscoelastic_freq::claims(def) {
        return Ok(None);
    }
    let Some(ek) = def.behavior("ELASTIC") else { return Ok(None) };
    if def.behavior("HYPERELASTIC").is_some() {
        return Ok(None);
    }
    let ty = ek.param_upper("TYPE").unwrap_or_else(|| "ISOTROPIC".into());
    if ty != "ISOTROPIC" {
        return ek.err("*VISCOELASTIC stöds bara med isotrop elasticitet (TYPE=ISOTROPIC)");
    }
    let inst = instantaneous(ek)?;
    let tab = super::thermal_elastic::iso_table(&without_moduli(ek))?;
    if tab.len() > 1 {
        return ek.err("temperaturberoende elasticitet med *VISCOELASTIC stöds inte ännu");
    }
    let prony = Prony::parse(vk)?;
    let (e, nu) = (tab[0].0, tab[0].1);
    let (mut g0, mut k0) = (e / (2.0 * (1.0 + nu)), e / (3.0 * (1.0 - 2.0 * nu)));
    if !inst {
        g0 /= 1.0 - prony.sum_g();
        k0 /= 1.0 - prony.sum_k();
    }
    Ok(Some(Arc::new(ViscoSmall { g0, k0, prony, td: TimeDependence::default() })))
}

fn iso_from_gk(g: f64, k: f64) -> M6 {
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = k - 2.0 * g / 3.0;
        }
        d[i][i] = k + 4.0 * g / 3.0;
        d[i + 3][i + 3] = g;
    }
    d
}

/// Deviator (tensorkomponenter) och volymtöjning för en töjning med ingenjörsskjuvning.
fn split(eps: &V6) -> (V6, f64) {
    let phi = eps[0] + eps[1] + eps[2];
    let e = [eps[0] - phi / 3.0, eps[1] - phi / 3.0, eps[2] - phi / 3.0, 0.5 * eps[3], 0.5 * eps[4], 0.5 * eps[5]];
    (e, phi)
}

fn norm_dev(a: &V6) -> f64 {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + 2.0 * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5])
}

impl ViscoSmall {
    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64], dt: f64, err: &Cell<f64>) -> (V6, M6) {
        let n = self.prony.terms.len();
        stn[..TERM + TSTRIDE * n].copy_from_slice(&st[..TERM + TSTRIDE * n]);
        let eps_n: V6 = std::array::from_fn(|i| st[EPS + i]);
        let (e1, phi1) = split(eps);
        let (e0, phi0) = split(&eps_n);
        let mut sum_ge = [0.0; 6];
        let mut sum_kp = 0.0;
        let (mut ga, mut ka) = (0.0, 0.0);
        let mut dev_err = [0.0; 6];
        let mut vol_err = 0.0;
        for (i, &(g, k, tau)) in self.prony.terms.iter().enumerate() {
            let b = TERM + TSTRIDE * i;
            let (a, _, gam) = coeffs(dt, tau);
            let ei: V6 = std::array::from_fn(|j| st[b + j]);
            let pi = st[b + 6];
            let mut ei1 = [0.0; 6];
            for j in 0..6 {
                ei1[j] = ei[j] + a * (e1[j] - e0[j]) + (1.0 - gam) * (e0[j] - ei[j]);
                stn[b + j] = ei1[j];
                sum_ge[j] += g * ei1[j];
                if dt.is_finite() && dt > 0.0 {
                    dev_err[j] += dt * g / tau * ((e1[j] - ei1[j]) - (e0[j] - ei[j]));
                }
            }
            let pi1 = pi + a * (phi1 - phi0) + (1.0 - gam) * (phi0 - pi);
            stn[b + 6] = pi1;
            sum_kp += k * pi1;
            if dt.is_finite() && dt > 0.0 {
                vol_err += dt * k / tau * ((phi1 - pi1) - (phi0 - pi));
            }
            ga += g * a;
            ka += k * a;
        }
        err.set((2.0 / 3.0 * norm_dev(&dev_err) + vol_err * vol_err / 3.0).sqrt());
        let p = self.k0 * (phi1 - sum_kp);
        let mut sig = [0.0; 6];
        for j in 0..6 {
            let s = 2.0 * self.g0 * (e1[j] - sum_ge[j]);
            sig[j] = s + if j < 3 { p } else { 0.0 };
        }
        stn[EPS..EPS + 6].copy_from_slice(eps);
        (sig, iso_from_gk(self.g0 * (1.0 - ga), self.k0 * (1.0 - ka)))
    }
}

impl MaterialModel for ViscoSmall {
    fn name(&self) -> &str {
        "VISCOELASTIC"
    }

    fn n_state(&self) -> usize {
        TERM + TSTRIDE * self.prony.terms.len()
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        let (g, k) = (self.g0, self.k0);
        Some((9.0 * k * g / (3.0 * k + g), (3.0 * k - 2.0 * g) / (2.0 * (3.0 * k + g))))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        super::plasticity::rotate_v6(r, &mut state[EPS..EPS + 6], true);
        for i in 0..self.prony.terms.len() {
            let b = TERM + TSTRIDE * i;
            super::plasticity::rotate_v6(r, &mut state[b..b + 6], false);
        }
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let dt = effective_dt(&self.td, input.dt);
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS + i]);
        let ce = iso_from_gk(self.g0, self.k0);
        let err = Cell::new(0.0);
        let mut f = |e: &V6| -> Result<(V6, M6)> { Ok(self.update3d(e, state_old, state_new, dt, &err)) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &ce, &mut f)?;
        state_new[EPS..EPS + 6].copy_from_slice(&eps3);
        self.td.record_error(err.get());
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(reduce_6x6(&iso_from_gk(self.g0, self.k0), mode))
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        Some(&self.td)
    }
}

// ---------------------------------------------------------------------------
// Stora töjningar
// ---------------------------------------------------------------------------

/// Viskoelasticitet ovanpå en hyperelastisk modell.
pub struct ViscoHyper {
    inner: Arc<dyn MaterialModel>,
    /// Skalfaktorer från de angivna konstanterna till de momentana (deviator, volym).
    sd: f64,
    sv: f64,
    prony: Prony,
    td: TimeDependence,
}

/// Tillståndslayout: [Fₙ (9, radvis), per term: τᵢᴰ (6, Voigt), τᵢᴴ].
const FN: usize = 0;
const HTERM: usize = 9;

fn build_finite(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(vk) = def.behavior("VISCOELASTIC") else { return Ok(None) };
    if super::viscoelastic_freq::claims(def) {
        return Ok(None);
    }
    let Some(hk) = def.behavior("HYPERELASTIC") else { return Ok(None) };
    if def.behavior("ELASTIC").is_some() {
        return hk.err("*HYPERELASTIC och *ELASTIC i samma material stöds inte");
    }
    let inst = instantaneous(hk)?;
    let prony = Prony::parse(vk)?;
    let mut rest = def.clone();
    rest.behaviors.retain(|b| b.name == "HYPERELASTIC");
    rest.behaviors = rest.behaviors.iter().map(without_moduli).collect();
    let inner = super::build(&rest)?;
    let Some(h) = inner.hyperelastic() else {
        return hk.err("den hyperelastiska modellen saknar uppdelning i deviator och volym");
    };
    if h.incompressible() {
        return hk.err("*VISCOELASTIC med helt inkompressibel *HYPERELASTIC (D1 = 0) stöds inte ännu");
    }
    let (sd, sv) = if inst { (1.0, 1.0) } else { (1.0 / (1.0 - prony.sum_g()), 1.0 / (1.0 - prony.sum_k())) };
    Ok(Some(Arc::new(ViscoHyper { inner, sd, sv, prony, td: TimeDependence::default() })))
}

fn dev3(a: &M3) -> M3 {
    let p = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let mut d = *a;
    for i in 0..3 {
        d[i][i] -= p;
    }
    d
}

fn from_voigt(v: &[f64]) -> M3 {
    [[v[0], v[3], v[4]], [v[3], v[1], v[5]], [v[4], v[5], v[2]]]
}

fn push(df: &M3, a: &M3) -> M3 {
    t::mul(&t::mul(df, a), &t::transpose(df))
}

impl ViscoHyper {
    fn hyper(&self) -> &dyn HyperModel {
        self.inner.hyperelastic().expect("kontrollerat i build")
    }

    /// Momentan (deviatorisk Kirchhoff, τᴴ) för F.
    fn instant(&self, f: &M3) -> Result<(M3, f64)> {
        let h = self.hyper();
        let j = t::det(f);
        if !(j > 0.0) {
            return fail("hyperelastiskt material: J ≤ 0 (elementet har vänts ut och in)");
        }
        let td = t::scale(&h.tau_dev(f)?, self.sd);
        let (up, _) = h.vol(j);
        Ok((td, self.sv * j * up))
    }

    /// Kirchhoffspänningen vid F (inkrementets slut) från tillståndet `st`;
    /// skriver nytt tillstånd i `stn` om det ges. Returnerar (τ, felmått).
    fn tau(&self, f1: &M3, st: &[f64], dt: f64, stn: Option<&mut [f64]>) -> Result<(M3, f64)> {
        let fnn: M3 = [[st[FN], st[FN + 1], st[FN + 2]], [st[FN + 3], st[FN + 4], st[FN + 5]], [st[FN + 6], st[FN + 7], st[FN + 8]]];
        let (j0, j1) = (t::det(&fnn), t::det(f1));
        let fb0 = t::scale(&fnn, j0.powf(-1.0 / 3.0));
        let fb1 = t::scale(f1, j1.powf(-1.0 / 3.0));
        let (fb0i, _) = t::inv(&fb0);
        let df = t::mul(&fb1, &fb0i);
        let (d1, h1) = self.instant(f1)?;
        let (d0, h0) = self.instant(&fnn)?;
        let d0h = push(&df, &d0);
        let mut tau = d1;
        for i in 0..3 {
            tau[i][i] += h1;
        }
        let mut out = stn;
        let mut errd = t::zero();
        let mut errv = 0.0;
        for (i, &(g, k, ti)) in self.prony.terms.iter().enumerate() {
            let b = HTERM + 7 * i;
            let (a, be, ga) = coeffs(dt, ti);
            let tid0 = from_voigt(&st[b..b + 6]);
            let tih0 = st[b + 6];
            let tidh = push(&df, &tid0);
            let mut tid1 = t::zero();
            for r in 0..3 {
                for c in 0..3 {
                    tid1[r][c] = a * g * d1[r][c] + be * g * d0h[r][c] + ga * tidh[r][c];
                }
            }
            let tih1 = a * k * h1 + be * k * h0 + ga * tih0;
            let dd = dev3(&tid1);
            for r in 0..3 {
                for c in 0..3 {
                    tau[r][c] -= dd[r][c];
                }
                tau[r][r] -= tih1;
            }
            if dt.is_finite() && dt > 0.0 {
                let ddh = dev3(&tidh);
                for r in 0..3 {
                    for c in 0..3 {
                        errd[r][c] += dt / ti * ((g * d1[r][c] - dd[r][c]) - (g * d0h[r][c] - ddh[r][c]));
                    }
                }
                errv += dt / ti * ((k * h1 - tih1) - (k * h0 - tih0));
            }
            if let Some(s) = out.as_deref_mut() {
                s[b..b + 6].copy_from_slice(&voigt(&tid1));
                s[b + 6] = tih1;
            }
        }
        if let Some(s) = out.as_deref_mut() {
            for r in 0..3 {
                for c in 0..3 {
                    s[FN + 3 * r + c] = f1[r][c];
                }
            }
        }
        let h = self.hyper();
        let (mu0, k0) = (h.mu0() * self.sd, h.k0() * self.sv);
        let ed: f64 = errd.iter().flatten().map(|x| x * x).sum::<f64>() / (2.0 * mu0).powi(2);
        let ev = errv / k0;
        Ok((tau, (2.0 / 3.0 * ed + ev * ev / 3.0).sqrt()))
    }
}

impl MaterialModel for ViscoHyper {
    fn name(&self) -> &str {
        "HYPERELASTIC + VISCOELASTIC"
    }

    fn n_state(&self) -> usize {
        HTERM + 7 * self.prony.terms.len()
    }

    fn init_state(&self, state: &mut [f64]) {
        state.iter_mut().for_each(|x| *x = 0.0);
        state[FN] = 1.0;
        state[FN + 4] = 1.0;
        state[FN + 8] = 1.0;
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        let h = self.hyper();
        let (mu, k) = (h.mu0() * self.sd, h.k0() * self.sv);
        Some((9.0 * k * mu / (3.0 * k + mu), (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let Some(f) = input.defgrad else {
            return fail("*HYPERELASTIC kräver NLGEOM=YES på *STEP");
        };
        let mode = input.mode;
        if matches!(mode, StressMode::PlaneStress | StressMode::Uniaxial) {
            return fail("*HYPERELASTIC med *VISCOELASTIC stöds inte i plan spänning eller stänger ännu");
        }
        let dt = effective_dt(&self.td, input.dt);
        let (tau, err) = self.tau(&f, state_old, dt, Some(state_new))?;
        self.td.record_error(err);
        let j = t::det(&f);
        let s = voigt(&tau).map(|x| x / j);
        let mut c = jaumann_tangent(&|g: &M3| Ok(self.tau(g, state_old, dt, None)?.0), &f)?;
        for r in c.iter_mut() {
            for v in r.iter_mut() {
                *v /= j;
            }
        }
        let idx: &[usize] = match mode {
            StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
            _ => &[0, 1, 2, 3],
        };
        let stress: Vec<f64> = idx.iter().map(|&i| s[i]).collect();
        let tangent = idx.iter().flat_map(|&i| idx.iter().map(move |&k| c[i][k])).collect();
        Ok(MatOutput { stress, tangent })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let h = self.hyper();
        Ok(reduce_6x6(&iso_from_gk(h.mu0() * self.sd, h.k0() * self.sv), mode))
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        Some(&self.td)
    }
}
