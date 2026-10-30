//! Progressiv skada för duktila metaller: *DAMAGE INITIATION
//! (CRITERION=DUCTILE eller SHEAR) med *DAMAGE EVOLUTION (TYPE=DISPLACEMENT
//! eller ENERGY) ovanpå plasticitet (*PLASTIC, von Mises).
//!
//! Den underliggande plasticitetsmodellen ger den effektiva (oskadade)
//! spänningen σ̄ och plastisk ekvivalent töjning ε̄p; spänningen är
//! σ = (1 − D) σ̄.
//!
//! Initiering (ω ≥ 1), integrerat med medelvärdet av initieringstöjningen vid
//! inkrementets början och slut, Δω = Δε̄p / ½(ε̄D(ηₙ) + ε̄D(ηₙ₊₁)):
//! - DUCTILE: ω_D = ∫ dε̄p / ε̄D(η), triaxialitet η = −p/q, tabell (ε̄D, η),
//! - SHEAR: ω_S = ∫ dε̄p / ε̄S(θs), θs = (q + ks p)/τmax med τmax = (σ1 − σ3)/2
//!   och KS = ks (standard 0,3), tabell (ε̄S, θs),
//! - JOHNSON COOK: ω = ∫ dε̄p / ε̄D med ε̄D = [d1 + d2 e^(−d3 η)][1 + d4 ln(ε̄̇p/ε̇0)]
//!   (1 + d5 θ̂) (`plasticity_rate::JcFailure`), ε̄̇p = Δε̄p/Δt; utdata JCCRT.
//!
//! Tabellerna interpoleras linjärt och är konstanta utanför. Från och med
//! inkrementet där kriteriet uppnås växer den plastiska ekvivalenta
//! förskjutningen ū̇p = L ε̄̇p med elementets karakteristiska längd L
//! (Hillerborg, `update_with_length`) och skadan d. Linjär
//! mjukning räknar hela initieringsinkrementets ūp, exponentiell mjukning bara delen
//! efter initieringspunkten (linjärt interpolerad i ω):
//! - TYPE=DISPLACEMENT, SOFTENING=LINEAR: d = ūp/ūf,
//! - TYPE=DISPLACEMENT, SOFTENING=EXPONENTIAL: d = (1 − e^(−α ūp/ūf))/(1 − e^(−α)),
//! - TYPE=DISPLACEMENT, SOFTENING=TABULAR: d(ūp) ur tabell (d, ūp),
//! - TYPE=ENERGY, SOFTENING=LINEAR: d = ūp/ūf med ūf = 2Gf/σy0 (σy0 =
//!   flytspänningen i initieringspunkten, interpolerad inom inkrementet),
//! - TYPE=ENERGY, SOFTENING=EXPONENTIAL: d = 1 − exp(−∫σy dūp / Gf).
//!
//! Med flera kriterier är D = max dᵢ (DEGRADATION=MAXIMUM, standard) eller
//! D = 1 − Π(1 − dᵢ) (MULTIPLICATIVE). D begränsas till Dmax = 0,99 så att
//! styvheten aldrig blir noll (elementborttagning stöds inte).
//!
//! Tangenten är (1 − D) C̄ när skadan inte växer och annars numerisk
//! (framåtdifferens av hela uppdateringen); den är osymmetrisk och negativ
//! efter mjukningens början, så den osymmetriska (LU-)lösaren väljs.
//!
//! Utdata: den underliggande modellens fält (PEEQ, PE) samt DUCTCRT, SHRCRT
//! (ω) och SDEG (D).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "duktil skada",
    consumes: &["ELASTIC", "PLASTIC", "CYCLIC HARDENING", "DAMAGE INITIATION", "DAMAGE EVOLUTION", "POTENTIAL", "RATE DEPENDENT"],
    priority: 20,
    build: Some(build),
}];

#[derive(Debug, Clone, Copy, PartialEq)]
enum Criterion {
    Ductile,
    Shear { ks: f64 },
    /// CRITERION=JOHNSON COOK: ε̄D(η, ε̄̇p, θ) i sluten form (`plasticity_rate.rs`);
    /// tillståndet sparar ε̄D i stället för η.
    JohnsonCook(super::plasticity_rate::JcFailure),
}

#[derive(Debug, Clone)]
enum Evolution {
    /// Ingen utveckling: bara initieringskriteriet redovisas.
    None,
    DispLinear { uf: f64 },
    DispExp { uf: f64, alpha: f64 },
    DispTable(Vec<(f64, f64)>),
    EnergyLinear { gf: f64 },
    EnergyExp { gf: f64 },
}

#[derive(Debug, Clone)]
struct Damage {
    crit: Criterion,
    /// (x, ε̄ vid initiering), x = η eller θs.
    table: Vec<(f64, f64)>,
    evol: Evolution,
    /// Område materials5: hastighets-/temperaturberoende tabell
    /// (`damage_tables.rs`), utvärderad med inkrementets ε̄̇pl = Δε̄pl/Δt och
    /// temperatur i båda ändarna (η vid början och slut).
    dep: Option<super::damage_tables::DepTable>,
}

pub struct DuctileDamage {
    inner: Arc<dyn MaterialModel>,
    peeq: usize,
    crits: Vec<Damage>,
    multiplicative: bool,
}

/// Per kriterium: [ω, ūp, d, σy0 (ENERGY LINEAR), ∫σy dūp (ENERGY EXPONENTIAL),
/// η eller θs vid inkrementets slut].
const PER: usize = 6;
/// Gemensamt efter kriterierna: [D, q̄ vid inkrementets slut]. D begränsas till:
const DMAX: f64 = 0.99;

/// Dmax: 0,99 eller enligt *SECTION CONTROLS/elementborttagning (`steps/model_change.rs`).
fn dmax() -> f64 {
    crate::steps::model_change::max_degradation().unwrap_or(DMAX)
}

fn table(kw: &RawKeyword) -> Result<Vec<(f64, f64)>> {
    let mut t: Vec<(f64, f64)> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "töjningshastighets- eller temperaturberoende initiering stöds inte ännu");
        }
        let (e, x) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(e > 0.0) {
            return kw.err_at(d.line, "initieringstöjningen ska vara positiv");
        }
        if t.last().is_some_and(|&(xl, _)| !(x > xl)) {
            return kw.err_at(d.line, "triaxialiteten (eller θs) ska öka i tabellen");
        }
        t.push((x, e));
    }
    if t.is_empty() {
        return kw.err("data saknas");
    }
    Ok(t)
}

fn evolution(kw: &RawKeyword) -> Result<Evolution> {
    if kw.has("DEPENDENCIES") || kw.data.len() > 1 && kw.param_upper("SOFTENING").as_deref() != Some("TABULAR") {
        return kw.err("temperatur- eller fältberoende skadeutveckling stöds inte ännu");
    }
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "DISPLACEMENT".into());
    let soft = kw.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
    if kw.has("MIXED MODE BEHAVIOR") || kw.has("MODE MIX RATIO") || kw.has("POWER") {
        return kw.err("blandade moder (MIXED MODE BEHAVIOR) stöds inte (bara för kohesiva element)");
    }
    let v = kw.first_floats()?;
    let a = v.first().copied().unwrap_or(0.0);
    match (ty.as_str(), soft.as_str()) {
        ("DISPLACEMENT", "TABULAR") => {
            let mut t: Vec<(f64, f64)> = Vec::new();
            for d in &kw.data {
                let v = kw.floats(d)?;
                let (dd, u) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                if !(0.0..1.0).contains(&dd) || t.last().is_some_and(|&(ul, _)| !(u > ul)) {
                    return kw.err_at(d.line, "tabellen ska ha 0 ≤ d < 1 och växande ūp");
                }
                t.push((u, dd));
            }
            Ok(Evolution::DispTable(t))
        }
        _ if !(a > 0.0) => kw.err("brottförskjutningen eller brottenergin ska vara positiv"),
        ("DISPLACEMENT", "LINEAR") => Ok(Evolution::DispLinear { uf: a }),
        ("DISPLACEMENT", "EXPONENTIAL") => {
            let alpha = v.get(1).copied().unwrap_or(0.0);
            if !(alpha > 0.0) {
                return kw.err("SOFTENING=EXPONENTIAL kräver exponenten α > 0");
            }
            Ok(Evolution::DispExp { uf: a, alpha })
        }
        ("ENERGY", "LINEAR") => Ok(Evolution::EnergyLinear { gf: a }),
        ("ENERGY", "EXPONENTIAL") => Ok(Evolution::EnergyExp { gf: a }),
        (t, s) => kw.err(format!("TYPE={t}, SOFTENING={s} stöds inte ännu")),
    }
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if super::cohesive::is_cohesive(def) {
        return Ok(None); // traction-separation för kohesiva element: materials/cohesive.rs
    }
    if super::xfem_damage::claims(def) {
        return Ok(None); // XFEM-spricka: materials/xfem_damage.rs
    }
    if super::hashin::claims(def) || super::forming_limit::claims(def) {
        return Ok(None); // område materials5: materials/hashin.rs, materials/forming_limit.rs
    }
    if def.behavior("DAMAGE INITIATION").is_none() {
        if let Some(e) = def.behavior("DAMAGE EVOLUTION") {
            return e.err("*DAMAGE EVOLUTION kräver en föregående *DAMAGE INITIATION");
        }
        return Ok(None);
    }
    // kriterier i ordning; *DAMAGE EVOLUTION hör till närmast föregående *DAMAGE INITIATION
    let mut crits: Vec<Damage> = Vec::new();
    let mut degradation: Option<String> = None;
    for b in &def.behaviors {
        match b.name.as_str() {
            "DAMAGE INITIATION" => {
                let crit = match b.param_upper("CRITERION").as_deref() {
                    Some("DUCTILE") => Criterion::Ductile,
                    Some("SHEAR") => {
                        let ks = match b.param("KS") {
                            Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: ogiltigt KS", b.location())))?,
                            None => 0.3,
                        };
                        Criterion::Shear { ks }
                    }
                    Some("JOHNSON COOK") => Criterion::JohnsonCook(super::plasticity_rate::JcFailure::parse(b)?),
                    Some(o) => return b.err(format!("CRITERION={o} stöds inte ännu (DUCTILE, SHEAR, JOHNSON COOK)")),
                    None => return b.err("CRITERION= saknas"),
                };
                if b.has("DEPENDENCIES") {
                    return b.err("fältberoende initiering stöds inte ännu");
                }
                // Johnson–Cook: ingen tabell, tillståndet har ε̄D direkt (obegränsat intervall)
                let dep = if matches!(crit, Criterion::JohnsonCook(_)) { None } else { super::damage_tables::DepTable::parse(b)? };
                let table = if matches!(crit, Criterion::JohnsonCook(_)) || dep.is_some() { vec![(-1e300, 1.0), (1e300, 1.0)] } else { table(b)? };
                crits.push(Damage { crit, table, evol: Evolution::None, dep });
            }
            "DAMAGE EVOLUTION" => {
                let Some(c) = crits.last_mut() else {
                    return b.err("*DAMAGE EVOLUTION kräver en föregående *DAMAGE INITIATION");
                };
                if !matches!(c.evol, Evolution::None) {
                    return b.err("kriteriet har redan en *DAMAGE EVOLUTION");
                }
                let deg = b.param_upper("DEGRADATION").unwrap_or_else(|| "MAXIMUM".into());
                if deg != "MAXIMUM" && deg != "MULTIPLICATIVE" {
                    return b.err(format!("DEGRADATION={deg} stöds inte (MAXIMUM, MULTIPLICATIVE)"));
                }
                if degradation.as_ref().is_some_and(|d| *d != deg) {
                    return b.err("alla kriterier ska ha samma DEGRADATION");
                }
                degradation = Some(deg);
                c.evol = evolution(b)?;
            }
            _ => {}
        }
    }
    let Some(pk) = def.behavior("PLASTIC") else {
        return fail(format!("material {}: *DAMAGE INITIATION kräver *PLASTIC", def.name));
    };
    let base = MaterialDef {
        name: def.name.clone(),
        behaviors: def.behaviors.iter().filter(|b| !b.name.starts_with("DAMAGE ")).cloned().collect(),
        density: def.density,
    };
    let inner = super::build(&base)?;
    let Some(peeq) = inner.output_names().iter().position(|n| *n == "PEEQ") else {
        return pk.err("skademodellen kräver en plasticitetsmodell med PEEQ");
    };
    Ok(Some(Arc::new(DuctileDamage { inner, peeq, crits, multiplicative: degradation.as_deref() == Some("MULTIPLICATIVE") })))
}

fn interp(t: &[(f64, f64)], x: f64) -> f64 {
    super::principal::interp(t, x).0
}

/// 3D-spänning (Voigt, tensorskjuvning) ur spänningen i ett spänningstillstånd.
fn stress3(mode: StressMode, s: &[f64]) -> [f64; 6] {
    match mode {
        StressMode::ThreeD => [s[0], s[1], s[2], s[3], s[4], s[5]],
        StressMode::PlaneStrain | StressMode::Axisymmetric => [s[0], s[1], s[2], s[3], 0.0, 0.0],
        StressMode::PlaneStress => [s[0], s[1], 0.0, s[2], 0.0, 0.0],
        StressMode::Uniaxial => [s[0], 0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

impl DuctileDamage {
    fn n_inner(&self) -> usize {
        self.inner.n_state()
    }

    fn base(&self) -> usize {
        self.n_inner()
    }

    fn d_index(&self) -> usize {
        self.base() + PER * self.crits.len()
    }

    /// Hela uppdateringen: effektiv uppdatering, kriterier, skada och spänning.
    fn eval(&self, input: &MatInput, len: f64, st: &[f64], stn: &mut [f64]) -> Result<(MatOutput, bool)> {
        let ni = self.n_inner();
        let out = self.inner.update(input, &st[..ni], &mut stn[..ni])?;
        stn[ni..].copy_from_slice(&st[ni..]);
        let p_old = self.inner.output_values(&st[..ni])[self.peeq];
        let p_new = self.inner.output_values(&stn[..ni])[self.peeq];
        let dp = (p_new - p_old).max(0.0);
        let s3 = stress3(input.mode, &out.stress);
        let (l, _) = super::principal::principal(&s3);
        let (p, q, _) = super::principal::pq_theta(&l);
        let di = self.d_index();
        let q_old = st[di + 1];
        stn[di + 1] = q;
        let mut growing = false;
        for (k, c) in self.crits.iter().enumerate() {
            let o = self.base() + PER * k;
            let x = match c.crit {
                Criterion::Ductile => {
                    if q > 0.0 {
                        -p / q
                    } else {
                        0.0
                    }
                }
                Criterion::Shear { ks } => {
                    let tau = 0.5 * (l[0] - l[2]);
                    if tau > 0.0 {
                        (q + ks * p) / tau
                    } else {
                        0.0
                    }
                }
                Criterion::JohnsonCook(jc) => {
                    let eta = if q > 0.0 { -p / q } else { 0.0 };
                    let rate = if input.dt > 0.0 { dp / input.dt } else { 0.0 };
                    jc.strain(eta, rate, input.temperature)
                }
            };
            // η (θs) begränsas till tabellens intervall (tabellen är konstant utanför)
            let x = x.clamp(c.table[0].0, c.table[c.table.len() - 1].0);
            stn[o + 5] = x;
            if dp <= 0.0 {
                continue;
            }
            // initieringskriteriet med medelvärdet av ε̄D vid inkrementets början
            // och slut
            let w_old = st[o];
            // (Johnson–Cook: ε̄D sparas direkt; 0 = inte satt än, använd det nya värdet)
            let eps_d = |v: f64| match c.crit {
                Criterion::JohnsonCook(_) if v > 0.0 => v,
                Criterion::JohnsonCook(_) => x,
                // område materials5: båda ändarna med inkrementets hastighet och
                // temperatur
                _ if c.dep.is_some() => c.dep.as_ref().map_or(1.0, |t| t.eval(v, if input.dt > 0.0 { dp / input.dt } else { 0.0 }, input.temperature)),
                _ => interp(&c.table, v),
            };
            let w = w_old + dp / (0.5 * (eps_d(st[o + 5]) + eps_d(x)));
            stn[o] = w;
            if w < 1.0 || matches!(c.evol, Evolution::None) {
                continue;
            }
            // andel av inkrementet efter initieringen
            let after = if w_old < 1.0 { (w - 1.0) / (w - w_old) } else { 1.0 };
            if w_old < 1.0 {
                // flytspänningen vid initieringspunkten
                stn[o + 3] = q - after * (q - q_old);
                stn[o + 4] = 0.0;
            }
            // linjär mjukning räknar hela initieringsinkrementet, exponentiell
            // bara delen efter initieringspunkten
            let du_full = len * dp;
            let du_after = after * du_full;
            let d = match &c.evol {
                Evolution::None => 0.0,
                Evolution::DispLinear { uf } => {
                    stn[o + 1] = st[o + 1] + du_full;
                    stn[o + 1] / uf
                }
                Evolution::DispExp { uf, alpha } => {
                    stn[o + 1] = st[o + 1] + du_after;
                    (1.0 - (-alpha * stn[o + 1] / uf).exp()) / (1.0 - (-alpha).exp())
                }
                Evolution::DispTable(t) => {
                    stn[o + 1] = st[o + 1] + du_full;
                    interp(t, stn[o + 1])
                }
                Evolution::EnergyLinear { gf } => {
                    stn[o + 1] = st[o + 1] + du_full;
                    stn[o + 1] * stn[o + 3] / (2.0 * gf)
                }
                Evolution::EnergyExp { gf } => {
                    stn[o + 1] = st[o + 1] + du_after;
                    // ∫σy dūp med flytspänningen vid inkrementets slut
                    let g = stn[o + 4] + q * du_after;
                    stn[o + 4] = g;
                    1.0 - (-g / gf).exp()
                }
            };
            stn[o + 2] = d.clamp(st[o + 2], dmax().max(st[o + 2]));
            growing = true;
        }
        let dtot = if self.multiplicative {
            1.0 - self.crits.iter().enumerate().map(|(k, _)| 1.0 - stn[self.base() + PER * k + 2]).product::<f64>()
        } else {
            self.crits.iter().enumerate().map(|(k, _)| stn[self.base() + PER * k + 2]).fold(0.0, f64::max)
        };
        stn[di] = dtot.min(dmax()).max(st[di]);
        let f = 1.0 - stn[di];
        let stress = out.stress.iter().map(|s| f * s).collect();
        let tangent = out.tangent.iter().map(|c| f * c).collect();
        Ok((MatOutput { stress, tangent }, growing))
    }
}

impl MaterialModel for DuctileDamage {
    fn name(&self) -> &str {
        "DAMAGE INITIATION"
    }

    fn n_state(&self) -> usize {
        self.n_inner() + PER * self.crits.len() + 2
    }

    fn init_state(&self, state: &mut [f64]) {
        let ni = self.n_inner();
        self.inner.init_state(&mut state[..ni]);
        for v in &mut state[ni..] {
            *v = 0.0;
        }
        for (k, c) in self.crits.iter().enumerate() {
            state[ni + PER * k + 5] = 0.0f64.clamp(c.table[0].0, c.table[c.table.len() - 1].0);
        }
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn requires_unsymmetric(&self) -> bool {
        // mjukningen ger en indefinit, osymmetrisk tangent
        self.crits.iter().any(|c| !matches!(c.evol, Evolution::None))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        let ni = self.n_inner();
        self.inner.rotate_state(r, &mut state[..ni]);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        // utan elementets karakteristiska längd (stänger m.fl.): L = 1
        self.update_with_length(input, 1.0, state_old, state_new)
    }

    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let (mut out, growing) = self.eval(input, length, state_old, state_new)?;
        if growing {
            // numerisk tangent: skadan beror på töjningen via ε̄p
            let n = input.strain.len();
            let scale = input.strain.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-6);
            let h = 1e-7 * scale;
            let mut tmp = state_new.to_vec();
            let mut c = vec![0.0; n * n];
            for j in 0..n {
                let mut e = input.strain.to_vec();
                e[j] += h;
                let mut de = input.dstrain.to_vec();
                de[j] += h;
                let inp = MatInput { strain: &e, dstrain: &de, ..*input };
                let (o2, _) = self.eval(&inp, length, state_old, &mut tmp)?;
                for i in 0..n {
                    c[i * n + j] = (o2.stress[i] - out.stress[i]) / h;
                }
            }
            out.tangent = c;
        }
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        if self.crits.iter().any(|c| c.crit == Criterion::Ductile) {
            v.push(("DUCTCRT", vec!["DUCTCRT"]));
        }
        if self.crits.iter().any(|c| matches!(c.crit, Criterion::Shear { .. })) {
            v.push(("SHRCRT", vec!["SHRCRT"]));
        }
        if self.crits.iter().any(|c| matches!(c.crit, Criterion::JohnsonCook(_))) {
            v.push(("JCCRT", vec!["JCCRT"]));
        }
        v.push(("SDEG", vec!["SDEG"]));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let ni = self.n_inner();
        let mut v = self.inner.output_field_values(mode, &state[..ni]);
        // kriteriet redovisas begränsat till 1 när det har en skadeutveckling
        let crit = |want: fn(&Criterion) -> bool| -> Option<f64> {
            self.crits
                .iter()
                .enumerate()
                .filter(|(_, c)| want(&c.crit))
                .map(|(k, c)| {
                    let w = state[self.base() + PER * k];
                    if matches!(c.evol, Evolution::None) {
                        w
                    } else {
                        w.min(1.0)
                    }
                })
                .reduce(f64::max)
        };
        if let Some(w) = crit(|c| *c == Criterion::Ductile) {
            v.push(w);
        }
        if let Some(w) = crit(|c| matches!(c, Criterion::Shear { .. })) {
            v.push(w);
        }
        if let Some(w) = crit(|c| matches!(c, Criterion::JohnsonCook(_))) {
            v.push(w);
        }
        v.push(state[self.d_index()]);
        v
    }
}
