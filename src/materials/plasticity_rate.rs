//! Hastighetsberoende flytspänning (*RATE DEPENDENT) och Johnson–Cooks
//! hårdnande (*PLASTIC, HARDENING=JOHNSON COOK; Johnson & Cook 1983).
//!
//! Den dynamiska flytspänningen är σ̄ = σ0(ε̄p, T) · R(ε̄̇p) där R är
//! - TYPE=POWER LAW (Cowper–Symonds, data D, n): ε̄̇p = D (R − 1)ⁿ, dvs.
//!   R = 1 + (ε̄̇p/D)^(1/n),
//! - TYPE=JOHNSON COOK (data C, ε̇0): R = 1 + C ln(ε̄̇p/ε̇0) för ε̄̇p ≥ ε̇0,
//!   annars R = 1,
//! - TYPE=YIELD RATIO (tabell R, ε̄̇p): linjär interpolation i ε̄̇p, konstant
//!   utanför tabellen.
//!
//! Johnson–Cooks hårdnande (data A, B, n, m, θmelt, θtransition):
//!
//!   σ0 = [A + B (ε̄p)ⁿ] (1 − θ̂ᵐ),
//!   θ̂ = 0 för θ < θtransition, (θ − θtransition)/(θmelt − θtransition)
//!   emellan och 1 över θmelt.
//!
//! Töjningshastigheten i returalgoritmen är ε̄̇p = Δε̄p/Δt (implicit), där Δt
//! är inkrementets tid i steget (*STATIC: stegets tidsperiod ger hastigheten).
//! Utan tid (Δt = 0, t.ex. när steget börjar och den inre kraften räknas om)
//! är hastigheten obegränsad: svaret är elastiskt för POWER LAW och
//! JOHNSON COOK (C > 0) och använder tabellens sista R för YIELD RATIO.

use super::MaterialFactory;
use crate::fem::model::RawKeyword;
use crate::model::Result;

/// Inga egna fabriker: lagarna används av `plasticity_hill.rs`.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Hastighetslag för flytspänningens förhållande R(ε̄̇p).
#[derive(Debug, Clone)]
pub enum RateLaw {
    None,
    Power { d: f64, n: f64 },
    JohnsonCook { c: f64, e0: f64 },
    /// (ε̄̇p, R) med stigande ε̄̇p.
    Ratio(Vec<(f64, f64)>),
}

impl RateLaw {
    /// Tolkar *RATE DEPENDENT.
    pub fn parse(kw: &RawKeyword) -> Result<RateLaw> {
        let ty = kw.param_upper("TYPE").unwrap_or_else(|| "POWER LAW".into());
        if kw.has("DEPENDENCIES") {
            return kw.err("fältberoende hastighetsberoende stöds inte ännu");
        }
        match ty.as_str() {
            "POWER LAW" | "JOHNSON COOK" => {
                if kw.data.len() > 1 {
                    return kw.err("temperaturberoende hastighetsparametrar (flera datarader) stöds inte ännu");
                }
                let v = kw.first_floats()?;
                if v.len() < 2 {
                    return kw.err(format!("TYPE={ty} kräver två konstanter"));
                }
                if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) && kw.data.len() > 1 {
                    return kw.err("temperaturberoende hastighetsparametrar stöds inte ännu");
                }
                if ty == "POWER LAW" {
                    if !(v[0] > 0.0 && v[1] > 0.0) {
                        return kw.err("POWER LAW: D och n ska vara positiva");
                    }
                    Ok(RateLaw::Power { d: v[0], n: v[1] })
                } else {
                    if !(v[0] >= 0.0 && v[1] > 0.0) {
                        return kw.err("JOHNSON COOK: C ska vara icke-negativ och ε̇0 positiv");
                    }
                    Ok(RateLaw::JohnsonCook { c: v[0], e0: v[1] })
                }
            }
            "YIELD RATIO" => {
                let mut t: Vec<(f64, f64)> = Vec::new();
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
                        return kw.err_at(d.line, "temperaturberoende flytspänningsförhållande stöds inte ännu");
                    }
                    let (r, rate) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                    if !(r > 0.0) || rate < 0.0 {
                        return kw.err_at(d.line, "R ska vara positivt och töjningshastigheten icke-negativ");
                    }
                    if t.last().is_some_and(|&(xl, _)| !(rate > xl)) {
                        return kw.err_at(d.line, "töjningshastigheten ska öka i tabellen");
                    }
                    t.push((rate, r));
                }
                if t.is_empty() {
                    return kw.err("tabell saknas");
                }
                Ok(RateLaw::Ratio(t))
            }
            o => kw.err(format!("TYPE={o} stöds inte (POWER LAW, JOHNSON COOK, YIELD RATIO)")),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, RateLaw::None)
    }

    /// R vid obegränsad töjningshastighet (inkrement utan tid, Δt = 0):
    /// `None` om R växer obegränsat (då kan inget plastiskt flöde ske på noll tid).
    pub fn ratio_infinite(&self) -> Option<f64> {
        match self {
            RateLaw::None => Some(1.0),
            RateLaw::Power { .. } => None,
            RateLaw::JohnsonCook { c, .. } => (*c == 0.0).then_some(1.0),
            RateLaw::Ratio(t) => Some(t[t.len() - 1].1),
        }
    }

    /// (R, dR/dε̄̇p) vid töjningshastigheten `rate` ≥ 0.
    pub fn ratio(&self, rate: f64) -> (f64, f64) {
        let rate = rate.max(0.0);
        match self {
            RateLaw::None => (1.0, 0.0),
            RateLaw::Power { d, n } => {
                if rate <= 0.0 {
                    return (1.0, 0.0);
                }
                let x = rate / d;
                let y = x.powf(1.0 / n);
                (1.0 + y, y / (n * rate))
            }
            RateLaw::JohnsonCook { c, e0 } => {
                if rate <= *e0 {
                    (1.0, 0.0)
                } else {
                    (1.0 + c * (rate / e0).ln(), c / rate)
                }
            }
            RateLaw::Ratio(t) => super::principal::interp(t, rate),
        }
    }
}

/// Johnson–Cooks hårdnande.
#[derive(Debug, Clone, Copy)]
pub struct JohnsonCook {
    pub a: f64,
    pub b: f64,
    pub n: f64,
    pub m: f64,
    pub tmelt: f64,
    pub ttr: f64,
}

impl JohnsonCook {
    /// Tolkar dataraden till *PLASTIC, HARDENING=JOHNSON COOK.
    pub fn parse(kw: &RawKeyword) -> Result<JohnsonCook> {
        if kw.data.len() > 1 {
            return kw.err("HARDENING=JOHNSON COOK tar en datarad (A, B, n, m, θmelt, θtransition)");
        }
        let v = kw.first_floats()?;
        if v.len() < 4 {
            return kw.err("HARDENING=JOHNSON COOK kräver A, B, n, m, θmelt och θtransition");
        }
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let jc = JohnsonCook { a: g(0), b: g(1), n: g(2), m: g(3), tmelt: g(4), ttr: g(5) };
        if !(jc.a > 0.0) || jc.b < 0.0 || jc.n < 0.0 {
            return kw.err("Johnson–Cook: A ska vara positivt, B och n icke-negativa");
        }
        if jc.tmelt <= jc.ttr && !(jc.tmelt == 0.0 && jc.ttr == 0.0) {
            return kw.err("Johnson–Cook: smälttemperaturen ska vara högre än övergångstemperaturen");
        }
        Ok(jc)
    }

    /// Den homologa temperaturen θ̂.
    pub fn theta_hat(&self, t: f64) -> f64 {
        if self.tmelt <= self.ttr || t < self.ttr {
            0.0
        } else if t > self.tmelt {
            1.0
        } else {
            (t - self.ttr) / (self.tmelt - self.ttr)
        }
    }

    /// Temperaturfaktorn 1 − θ̂ᵐ.
    pub fn temp_factor(&self, t: f64) -> f64 {
        let th = self.theta_hat(t);
        if th <= 0.0 {
            1.0
        } else {
            1.0 - th.powf(self.m)
        }
    }

    /// (σ0, dσ0/dε̄p) vid ε̄p = `p` och temperaturen `t`.
    pub fn eval(&self, p: f64, t: f64) -> (f64, f64) {
        let f = self.temp_factor(t);
        let p = p.max(0.0);
        let s = self.a + self.b * p.powf(self.n);
        let h = if self.b == 0.0 || self.n == 0.0 {
            0.0
        } else if self.n == 1.0 {
            self.b
        } else {
            self.b * self.n * p.max(1e-300).powf(self.n - 1.0)
        };
        (s * f, h.min(1e300) * f)
    }
}

/// Johnson–Cooks brottkriterium för *DAMAGE INITIATION, CRITERION=JOHNSON
/// COOK (data d1, d2, d3, d4, d5, θmelt, θtransition, ε̇0):
///
///   ε̄D = [d1 + d2 e^(−d3 η)] [1 + d4 ln(ε̄̇p/ε̇0)] (1 + d5 θ̂)
///
/// med triaxialiteten η = −p/q. Under referenshastigheten (ε̄̇p < ε̇0) används
/// ε̄̇p = ε̇0, dvs. hastighetsfaktorn är minst 1 (som i hastighetsberoendet).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JcFailure {
    pub d: [f64; 5],
    pub tmelt: f64,
    pub ttr: f64,
    pub e0: f64,
}

impl JcFailure {
    pub fn parse(kw: &RawKeyword) -> Result<JcFailure> {
        let recs = kw.records();
        if recs.len() > 1 {
            return kw.err("CRITERION=JOHNSON COOK tar en datarad (d1, d2, d3, d4, d5, θmelt, θtransition, ε̇0)");
        }
        let v = match recs.first() {
            Some(d) => kw.floats(d)?,
            None => Vec::new(),
        };
        if v.len() < 5 {
            return kw.err("CRITERION=JOHNSON COOK kräver d1–d5 (och θmelt, θtransition, ε̇0)");
        }
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let e0 = if g(7) > 0.0 { g(7) } else { 1.0 };
        let f = JcFailure { d: [g(0), g(1), g(2), g(3), g(4)], tmelt: g(5), ttr: g(6), e0 };
        if f.tmelt < f.ttr {
            return kw.err("Johnson–Cook: smälttemperaturen ska vara högre än övergångstemperaturen");
        }
        Ok(f)
    }

    /// Initieringstöjningen ε̄D vid triaxialiteten `eta`, töjningshastigheten
    /// `rate` och temperaturen `t`.
    pub fn strain(&self, eta: f64, rate: f64, t: f64) -> f64 {
        let [d1, d2, d3, d4, d5] = self.d;
        let th = if self.tmelt <= self.ttr || t < self.ttr {
            0.0
        } else if t > self.tmelt {
            1.0
        } else {
            (t - self.ttr) / (self.tmelt - self.ttr)
        };
        let r = (rate / self.e0).max(1.0);
        let e = (d1 + d2 * (-d3 * eta).exp()) * (1.0 + d4 * r.ln()) * (1.0 + d5 * th);
        e.max(1e-12)
    }
}
