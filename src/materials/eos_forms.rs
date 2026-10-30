//! Område gaps1: fler tillståndsekvationer för `eos.rs` (inkluderas därifrån).
//!
//! ```text
//! TYPE=TABULAR (data f₁, f₂, ε_vol i fallande ε_vol):
//!   p = f₁(ε_vol) + ρ₀ f₂(ε_vol) Eₘ,   ε_vol = ln(ρ₀/ρ) = ln J
//! TYPE=JWL (data C_d, A, B, ω, R₁, R₂, E₀, K_pd; *DETONATION POINT):
//!   p_JWL = A (1 − ω/(R₁ J)) e^(−R₁ J) + B (1 − ω/(R₂ J)) e^(−R₂ J) + (ω ρ₀/J) Eₘ
//! ```
//!
//! TABULAR: styckvis linjär mellan punkterna, utanför tabellen extrapolerad
//! med den sista lutningen.
//!
//! JWL (programmerad förbränning): detonationsvågen når materialpunkten vid
//! t_d = min_N [t_dN + |x − x_dN| / C_d] (total tid, `init_at_point` med
//! integrationspunktens läge). Före t_d räknas trycket ur volymtöjningen med
//! K_pd, p = −K_pd ln J; därefter p = F_b p_JWL med förbränningsandelen
//! F_b = min(1, (t − t_d) C_d / (B_s l_e)), B_s = 2,5 och l_e elementets
//! karakteristiska längd. Den specifika energin börjar på E₀
//! (*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY skriver över).

use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::Result;

/// Standardtolerans för regularisering av tabelldata i explicit analys.
const RTOL: f64 = 0.03;

/// Tabellerad tillståndsekvation: (ε_vol, f₁, f₂) i stigande ε_vol.
#[derive(Debug, Clone)]
pub struct Tabular {
    pts: Vec<(f64, f64, f64)>,
}

impl Tabular {
    pub fn parse(kw: &RawKeyword) -> Result<Tabular> {
        let mut pts = Vec::new();
        for d in kw.records() {
            let v = kw.floats(&d)?;
            if v.is_empty() {
                continue;
            }
            pts.push((v.get(2).copied().unwrap_or(0.0), v[0], v.get(1).copied().unwrap_or(0.0)));
        }
        if pts.len() < 2 {
            return kw.err("*EOS, TYPE=TABULAR: ange minst två rader (f₁, f₂, ε_vol)");
        }
        if pts.windows(2).any(|w| !(w[1].0 < w[0].0)) {
            return kw.err("*EOS, TYPE=TABULAR: volymtöjningarna ska stå i fallande ordning");
        }
        pts.reverse();
        if !pts.iter().any(|p| p.0 == 0.0) {
            return kw.err("*EOS, TYPE=TABULAR: tabellen ska ha en rad för ε_vol = 0");
        }
        let user = Tabular { pts };
        Ok(user.regularized(RTOL))
    }

    /// Explicit analys regulariserar tabellen till lika intervall
    /// (tests/cases/gaps1/eos_tabular_*): intervallet är h = (ε_max − ε_min)/n
    /// med n = antalet användarintervall, nätet går genom ε_vol = 0, och n ökas
    /// tills felet i alla användarpunkter är högst RTOL gånger f₁:s (och f₂:s)
    /// variationsbredd (RTOL = 0,03, *MATERIAL, RTOL= läses inte).
    fn regularized(&self, rtol: f64) -> Tabular {
        let p = &self.pts;
        let (x0, x1) = (p[0].0, p[p.len() - 1].0);
        let range = |k: usize| {
            let (lo, hi) = p.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), q| {
                let v = if k == 1 { q.1 } else { q.2 };
                (a.min(v), b.max(v))
            });
            // f₂ ≈ konstant ger liten variationsbredd: samma nät som för f₁
            // godtas ändå, så måttet är minst största |värdet|
            (hi - lo).max(hi.abs().max(lo.abs()))
        };
        let (r1, r2) = (range(1), range(2));
        let nu = p.len() - 1;
        for n in nu..=100 * nu {
            let h = (x1 - x0) / n as f64;
            let k0 = (x0 / h - 1e-9).floor() as i64;
            let k1 = (x1 / h + 1e-9).ceil() as i64;
            let grid: Vec<(f64, f64, f64)> = (k0..=k1)
                .map(|k| {
                    let x = k as f64 * h;
                    let (a, b, _, _) = self.eval(x);
                    (x, a, b)
                })
                .collect();
            let reg = Tabular { pts: grid };
            let ok = p.iter().all(|q| {
                let (a, b, _, _) = reg.eval(q.0);
                (a - q.1).abs() <= rtol * r1 * (1.0 + 1e-9) + 1e-300 && (b - q.2).abs() <= rtol * r2 * (1.0 + 1e-9) + 1e-300
            });
            if ok {
                return reg;
            }
        }
        self.clone()
    }

    /// (f₁, f₂, f₁′, f₂′) vid ε_vol.
    pub fn eval(&self, ev: f64) -> (f64, f64, f64, f64) {
        let p = &self.pts;
        let n = p.len();
        let i = if ev <= p[0].0 {
            0
        } else if ev >= p[n - 1].0 {
            n - 2
        } else {
            p.windows(2).position(|w| ev >= w[0].0 && ev <= w[1].0).unwrap_or(0)
        };
        let (a, b) = (p[i], p[i + 1]);
        let d = b.0 - a.0;
        let s1 = (b.1 - a.1) / d;
        let s2 = (b.2 - a.2) / d;
        (a.1 + s1 * (ev - a.0), a.2 + s2 * (ev - a.0), s1, s2)
    }
}

/// JWL med detonationspunkter (x, y, z, fördröjning).
#[derive(Debug, Clone)]
pub struct Jwl {
    pub cd: f64,
    a: f64,
    b: f64,
    w: f64,
    r1: f64,
    r2: f64,
    pub e0: f64,
    pub kpd: f64,
    points: Vec<[f64; 4]>,
}

/// Förbränningsvågens bredd B_s.
const BS: f64 = 2.5;

impl Jwl {
    pub fn parse(kw: &RawKeyword, def: &MaterialDef) -> Result<Jwl> {
        let v = kw.first_floats()?;
        let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
        let j = Jwl { cd: g(0), a: g(1), b: g(2), w: g(3), r1: g(4), r2: g(5), e0: g(6), kpd: g(7), points: Vec::new() };
        if !(j.cd > 0.0 && j.r1 > 0.0 && j.r2 > 0.0) {
            return kw.err("*EOS, TYPE=JWL: ange C_d, A, B, ω, R₁, R₂ (C_d, R₁, R₂ > 0), E₀ och K_pd");
        }
        let Some(dp) = def.behavior("DETONATION POINT") else {
            return kw.err("*EOS, TYPE=JWL kräver *DETONATION POINT");
        };
        let mut j = j;
        for d in dp.records() {
            let p = dp.floats(&d)?;
            if p.is_empty() {
                continue;
            }
            let g = |i: usize| p.get(i).copied().unwrap_or(0.0);
            j.points.push([g(0), g(1), g(2), g(3)]);
        }
        if j.points.is_empty() {
            return dp.err("*DETONATION POINT: ange minst en punkt");
        }
        Ok(j)
    }

    /// Detonationsvågens ankomsttid till punkten `x`.
    pub fn arrival(&self, x: [f64; 3]) -> f64 {
        self.points
            .iter()
            .map(|p| p[3] + ((x[0] - p[0]).powi(2) + (x[1] - p[1]).powi(2) + (x[2] - p[2]).powi(2)).sqrt() / self.cd)
            .fold(f64::INFINITY, f64::min)
    }

    /// Förbränningsandelen vid tiden `t` (0 före ankomsten).
    pub fn burn(&self, t: f64, td: f64, le: f64) -> f64 {
        if t < td {
            return 0.0;
        }
        if le <= 0.0 {
            return 1.0;
        }
        ((t - td) * self.cd / (BS * le)).min(1.0)
    }

    /// p_JWL = f(J) + γ(J) Eₘ: (f, γ).
    pub fn linear_in_energy(&self, j: f64, rho0: f64) -> (f64, f64) {
        let j = j.max(1e-6);
        let f = self.a * (1.0 - self.w / (self.r1 * j)) * (-self.r1 * j).exp() + self.b * (1.0 - self.w / (self.r2 * j)) * (-self.r2 * j).exp();
        (f, self.w * rho0 / j)
    }

    /// Isentropisk kompressionsmodul ρc² = −J dp/dJ|_s för p_JWL (dEₘ = −p dJ/ρ₀).
    pub fn bulk(&self, j: f64, em: f64, rho0: f64) -> f64 {
        let h = 1e-6 * j.max(1e-3);
        let (f1, g1) = self.linear_in_energy(j + h, rho0);
        let (f0, g0) = self.linear_in_energy(j - h, rho0);
        let (f, g) = self.linear_in_energy(j, rho0);
        let p = f + g * em;
        let dpdj = (f1 + g1 * em - f0 - g0 * em) / (2.0 * h) - g * p / rho0;
        (-j * dpdj).max(0.0)
    }
}
