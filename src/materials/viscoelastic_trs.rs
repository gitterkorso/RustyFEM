//! Termoreologiskt enkel temperaturinverkan (*TRS) för viskoelasticitet.
//!
//! Den reducerade tiden ξ(t) = ∫ ds/A(θ(s)) ersätter tiden i relaxations-
//! funktionerna:
//!
//! - DEFINITION=WLF (standard): log₁₀ A = −C1 (θ − θ0)/(C2 + θ − θ0), data θ0, C1, C2;
//!   för θ ≤ θ0 − C2 är svaret elastiskt med momentana moduler (A = ∞),
//! - DEFINITION=ARRHENIUS: ln A = E0/R (1/(θ − θZ) − 1/(θ0 − θZ)), data θ0, E0
//!   (R och θZ från *PHYSICAL CONSTANTS) eller θ0, tomt, E0/R,
//! - DEFINITION=TABULAR: log₁₀ A mot temperaturen (linjär interpolation,
//!   konstant utanför tabellen).
//!
//! I tidsplanet (*VISCOELASTIC, TIME=PRONY) omsluter `ViscoTrs` den vanliga
//! viskoelastiska modellen (`viscoelastic.rs`) och ger den inkrementet i
//! reducerad tid Δξ = Δt (1/A₁ − 1/A₀)/ln(A₀/A₁) med A vid inkrementets början
//! och slut (exakt integral när ln A ändras linjärt i tiden; kontrollerat
//! med temperaturramp och temperatursteg). I frekvensplanet
//! används den reducerade vinkelfrekvensen ωr = A(θ) ω (`viscoelastic_freq.rs`).

use super::time_dependence::TimeDependence;
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "viskoelasticitet i tidsplanet med *TRS (reducerad tid)",
    consumes: &["ELASTIC", "HYPERELASTIC", "VISCOELASTIC", "TRS"],
    priority: 30,
    build: Some(build),
}];

/// Förskjutningsfunktionen A(θ).
#[derive(Debug, Clone)]
pub enum Trs {
    Wlf { t0: f64, c1: f64, c2: f64 },
    /// θ0 och antingen E0 (kräver R) eller E0/R.
    Arrhenius { t0: f64, e0: f64, e0_r: Option<f64> },
    /// (log₁₀ A, θ) i stigande temperatur.
    Tabular(Vec<(f64, f64)>),
}

impl Trs {
    pub fn parse(kw: &RawKeyword) -> Result<Trs> {
        if kw.has("DEPENDENCIES") {
            return kw.err("fältberoende *TRS stöds inte ännu");
        }
        let def = kw.param_upper("DEFINITION").unwrap_or_else(|| "WLF".into());
        if def == "TABULAR" {
            let mut t: Vec<(f64, f64)> = Vec::new();
            for d in &kw.data {
                let v = kw.floats(d)?;
                let (la, th) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                if let Some(&(_, tl)) = t.last() {
                    if !(th > tl) {
                        return kw.err_at(d.line, "temperaturerna i *TRS, DEFINITION=TABULAR ska öka");
                    }
                }
                t.push((la, th));
            }
            if t.is_empty() {
                return kw.err("tabellen för förskjutningsfunktionen saknas");
            }
            return Ok(Trs::Tabular(t));
        }
        if kw.data.len() != 1 {
            return kw.err("*TRS har en datarad");
        }
        let v = kw.first_floats()?;
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        match def.as_str() {
            "WLF" => {
                if g(2) == 0.0 {
                    return kw.err("WLF kräver C2 ≠ 0");
                }
                Ok(Trs::Wlf { t0: g(0), c1: g(1), c2: g(2) })
            }
            "ARRHENIUS" => {
                let e0_r = v.get(2).copied().filter(|&x| x != 0.0);
                if e0_r.is_none() && g(1) == 0.0 {
                    return kw.err("ARRHENIUS kräver aktiveringsenergin E0 eller E0/R");
                }
                Ok(Trs::Arrhenius { t0: g(0), e0: g(1), e0_r })
            }
            o => kw.err(format!("DEFINITION={o} stöds inte ännu (WLF, ARRHENIUS, TABULAR)")),
        }
    }

    /// A(θ) med absoluta nollpunkten `tz` och gaskonstanten `r` (för ARRHENIUS).
    pub fn shift(&self, theta: f64, tz: f64, r: f64) -> Result<f64> {
        match self {
            Trs::Wlf { t0, c1, c2 } => {
                let d = theta - t0;
                if c2 + d <= 0.0 {
                    return Ok(f64::INFINITY);
                }
                Ok(10f64.powf(-c1 * d / (c2 + d)))
            }
            Trs::Arrhenius { t0, e0, e0_r } => {
                if !tz.is_finite() {
                    return fail("*TRS, DEFINITION=ARRHENIUS kräver *PHYSICAL CONSTANTS, ABSOLUTE ZERO=");
                }
                let k = match e0_r {
                    Some(x) => *x,
                    None => {
                        if !(r > 0.0) {
                            return fail("*TRS, DEFINITION=ARRHENIUS med E0 kräver *PHYSICAL CONSTANTS, UNIVERSAL GAS CONSTANT=");
                        }
                        e0 / r
                    }
                };
                Ok((k * (1.0 / (theta - tz) - 1.0 / (t0 - tz))).exp())
            }
            Trs::Tabular(t) => {
                let la = if t.len() == 1 || theta <= t[0].1 {
                    t[0].0
                } else if let Some(w) = t.windows(2).find(|w| theta < w[1].1) {
                    w[0].0 + (theta - w[0].1) / (w[1].1 - w[0].1) * (w[1].0 - w[0].0)
                } else {
                    t[t.len() - 1].0
                };
                Ok(10f64.powf(la))
            }
        }
    }
}

/// Viskoelasticitet i tidsplanet med reducerad tid.
pub struct ViscoTrs {
    inner: Arc<dyn MaterialModel>,
    trs: Trs,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(tk) = def.behavior("TRS") else { return Ok(None) };
    let Some(vk) = def.behavior("VISCOELASTIC") else {
        return tk.err("*TRS kräver *VISCOELASTIC");
    };
    if vk.has("FREQUENCY") {
        // frekvensplanet: `viscoelastic_freq.rs`
        return Ok(None);
    }
    let trs = Trs::parse(tk)?;
    let mut d2 = def.clone();
    d2.behaviors.retain(|b| b.name != "TRS");
    let inner = super::build(&d2)?;
    if inner.time_dependence().is_none() {
        return tk.err("*TRS kräver ett tidsberoende viskoelastiskt material");
    }
    Ok(Some(Arc::new(ViscoTrs { inner, trs })))
}

impl ViscoTrs {
    fn td(&self) -> &TimeDependence {
        self.inner.time_dependence().unwrap()
    }
}

impl MaterialModel for ViscoTrs {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn n_state(&self) -> usize {
        self.inner.n_state()
    }

    fn init_state(&self, state: &mut [f64]) {
        self.inner.init_state(state)
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        self.inner.rotate_state(r, state)
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let td = self.td();
        let (tz, r) = (td.absolute_zero(), td.gas_constant());
        // 1/A vid inkrementets början och slut; Δξ = Δt (b1 − b0)/ln(b1/b0), exakt
        // när ln A är linjär i tiden under inkrementet
        let b1 = 1.0 / self.trs.shift(input.temperature, tz, r)?;
        let b0 = 1.0 / self.trs.shift(input.temperature - input.dtemp, tz, r)?;
        let dxi = if !(b0 > 0.0 && b1 > 0.0) {
            0.0
        } else if ((b1 - b0) / b1).abs() < 1e-10 {
            input.dt * 0.5 * (b0 + b1)
        } else {
            input.dt * (b1 - b0) / (b1 / b0).ln()
        };
        let inp = MatInput { mode: input.mode, strain: input.strain, dstrain: input.dstrain, stress_old: input.stress_old, time: input.time, dt: dxi, temperature: input.temperature, dtemp: input.dtemp, defgrad: input.defgrad };
        self.inner.update(&inp, state_old, state_new)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn output_names(&self) -> &[&'static str] {
        self.inner.output_names()
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        self.inner.output_values(state)
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        self.inner.output_fields(mode)
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        self.inner.output_field_values(mode, state)
    }

    fn hyperelastic(&self) -> Option<&dyn super::hyperelastic_split::HyperModel> {
        self.inner.hyperelastic()
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        self.inner.time_dependence()
    }
}
