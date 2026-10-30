//! Hjälpfält för interaktionsintegralen (`fracture.rs`).
//!
//! Fälten skrivs i sprickspetsens lokala koordinater (x₁ i sprickans
//! tillväxtriktning, x₂ normal mot sprickplanet, x₃ längs sprickfronten) med
//! dualtal i (x₁, x₂), så att förskjutningsgradienten blir exakt. Spänningen
//! räknas ur töjningen med Hookes lag, så hjälpfältet är ett exakt
//! elasticitetsfält (jämvikt och kompatibilitet) även för plan spänning.
//!
//! - Mod I, II: Williams asymptotiska fält med K = 1.
//! - Mod III: antiplan skjuvning med K = 1.
//! - T-spänning: punktkraft f = 1 i x₁ i sprickspetsen (Michell/Flamant),
//!   σ₁₁ = −cos³θ/(πr), σ₂₂ = −cosθ sin²θ/(πr), σ₁₂ = −cos²θ sinθ/(πr).

use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

/// Dualtal: värde och derivator med avseende på (x₁, x₂).
#[derive(Debug, Clone, Copy)]
pub struct D2 {
    pub v: f64,
    pub d: [f64; 2],
}

impl D2 {
    pub fn c(v: f64) -> D2 {
        D2 { v, d: [0.0; 2] }
    }
    fn chain(self, f: f64, df: f64) -> D2 {
        D2 { v: f, d: [self.d[0] * df, self.d[1] * df] }
    }
    pub fn sqrt(self) -> D2 {
        let s = self.v.sqrt();
        self.chain(s, 0.5 / s)
    }
    pub fn sin(self) -> D2 {
        self.chain(self.v.sin(), self.v.cos())
    }
    pub fn cos(self) -> D2 {
        self.chain(self.v.cos(), -self.v.sin())
    }
    pub fn ln(self) -> D2 {
        self.chain(self.v.ln(), 1.0 / self.v)
    }
    /// atan2(y, x) med y = self.
    pub fn atan2(self, x: D2) -> D2 {
        let r2 = x.v * x.v + self.v * self.v;
        D2 { v: self.v.atan2(x.v), d: [0, 1].map(|i| (x.v * self.d[i] - self.v * x.d[i]) / r2) }
    }
}

impl Add for D2 {
    type Output = D2;
    fn add(self, o: D2) -> D2 {
        D2 { v: self.v + o.v, d: [self.d[0] + o.d[0], self.d[1] + o.d[1]] }
    }
}
impl Sub for D2 {
    type Output = D2;
    fn sub(self, o: D2) -> D2 {
        D2 { v: self.v - o.v, d: [self.d[0] - o.d[0], self.d[1] - o.d[1]] }
    }
}
impl Mul for D2 {
    type Output = D2;
    fn mul(self, o: D2) -> D2 {
        D2 { v: self.v * o.v, d: [0, 1].map(|i| self.d[i] * o.v + self.v * o.d[i]) }
    }
}
impl Mul<f64> for D2 {
    type Output = D2;
    fn mul(self, s: f64) -> D2 {
        D2 { v: self.v * s, d: self.d.map(|x| x * s) }
    }
}
impl Add<f64> for D2 {
    type Output = D2;
    fn add(self, s: f64) -> D2 {
        D2 { v: self.v + s, d: self.d }
    }
}
impl Neg for D2 {
    type Output = D2;
    fn neg(self) -> D2 {
        self * -1.0
    }
}

/// Hjälpfältets typ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxMode {
    I,
    II,
    III,
    /// Punktkraft f = 1 i x₁ (T-spänning).
    T,
}

/// Elasticitet för hjälpfältet.
#[derive(Debug, Clone, Copy)]
pub struct Elastic {
    pub e: f64,
    pub nu: f64,
    /// Plan spänning (annars plan töjning, även för axisymmetri och 3D).
    pub plane_stress: bool,
}

impl Elastic {
    pub fn mu(&self) -> f64 {
        self.e / (2.0 * (1.0 + self.nu))
    }
    pub fn kappa(&self) -> f64 {
        if self.plane_stress {
            (3.0 - self.nu) / (1.0 + self.nu)
        } else {
            3.0 - 4.0 * self.nu
        }
    }
    /// E* = E (plan spänning) eller E/(1 − ν²).
    pub fn e_star(&self) -> f64 {
        if self.plane_stress {
            self.e
        } else {
            self.e / (1.0 - self.nu * self.nu)
        }
    }
}

/// Hjälpfältet i en punkt (lokala koordinater relativt spetsen).
#[derive(Debug, Clone, Copy)]
pub struct AuxField {
    /// Förskjutning u_j.
    pub u: [f64; 3],
    /// Förskjutningsgradient g[j][i] = ∂u_j/∂x_i.
    pub g: [[f64; 3]; 3],
    /// Spänning (tensor, symmetrisk).
    pub s: [[f64; 3]; 3],
}

/// Hjälpfältet för `mode` i punkten (x₁, x₂).
pub fn aux(mode: AuxMode, el: &Elastic, x1: f64, x2: f64) -> AuxField {
    let x = D2 { v: x1, d: [1.0, 0.0] };
    let y = D2 { v: x2, d: [0.0, 1.0] };
    let r = (x * x + y * y).sqrt();
    let th = y.atan2(x);
    let mu = el.mu();
    let k = el.kappa();
    let h = th * 0.5;
    let (sh, ch) = (h.sin(), h.cos());
    let sq = (r * (1.0 / (2.0 * PI))).sqrt();
    let zero = D2::c(0.0);
    let u: [D2; 3] = match mode {
        AuxMode::I => [
            sq * ch * (sh * sh * 2.0 + (k - 1.0)) * (1.0 / (2.0 * mu)),
            sq * sh * (ch * ch * -2.0 + (k + 1.0)) * (1.0 / (2.0 * mu)),
            zero,
        ],
        AuxMode::II => [
            sq * sh * (ch * ch * 2.0 + (k + 1.0)) * (1.0 / (2.0 * mu)),
            -(sq * ch * (sh * sh * -2.0 + (k - 1.0)) * (1.0 / (2.0 * mu))),
            zero,
        ],
        AuxMode::III => [zero, zero, sq * sh * (2.0 / mu)],
        AuxMode::T => {
            let (s, c) = (th.sin(), th.cos());
            [
                r.ln() * (-(1.0 + k) / (8.0 * PI * mu)) - s * s * (1.0 / (4.0 * PI * mu)),
                th * (-(k - 1.0) / (8.0 * PI * mu)) + s * c * (1.0 / (4.0 * PI * mu)),
                zero,
            ]
        }
    };
    let mut g = [[0.0; 3]; 3];
    for j in 0..3 {
        g[j][0] = u[j].d[0];
        g[j][1] = u[j].d[1];
    }
    let mut eps = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            eps[i][j] = 0.5 * (g[i][j] + g[j][i]);
        }
    }
    let nu = el.nu;
    let mut s = [[0.0; 3]; 3];
    if el.plane_stress {
        let c = el.e / (1.0 - nu * nu);
        s[0][0] = c * (eps[0][0] + nu * eps[1][1]);
        s[1][1] = c * (eps[1][1] + nu * eps[0][0]);
    } else {
        let lam = el.e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let tr = eps[0][0] + eps[1][1];
        s[0][0] = lam * tr + 2.0 * mu * eps[0][0];
        s[1][1] = lam * tr + 2.0 * mu * eps[1][1];
        s[2][2] = lam * tr;
    }
    for (i, j) in [(0, 1), (0, 2), (1, 2)] {
        s[i][j] = 2.0 * mu * eps[i][j];
        s[j][i] = s[i][j];
    }
    AuxField { u: [u[0].v, u[1].v, u[2].v], g, s }
}
