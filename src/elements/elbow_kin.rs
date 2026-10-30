//! Kinematik för rörkrökarna (ELBOW31, ELBOW32, ELBOW31B, ELBOW31C):
//! referensgeometri och linjäriserade töjningar i en punkt (S, φ) på
//! rörväggens mittyta. Inkluderas med `#[path]` från `elbow.rs`.
//!
//! Formuleringen är linjäriserad kring referensläget (raka och krökta
//! element med 0–6 moder):
//!
//! - Axeln är en cirkelbåge med torusradien R genom elementets ändnoder
//!   (R = 0: rak), båglängden S ∈ [0, L]. Riktarna A1 = a2 × a3 ur
//!   orienteringspunkten (normalen till krökens plan), A3 = den normerade
//!   tangenten Σ dHₙ/dξ Xₙ (kordan för tvånodselement) och A2 = A3 × A1 mot
//!   extradosen. Krökningen κ = 1/R ger A2' = κA3 och A3' = −κA2.
//! - Väggens läge X = X̄ + r₀R(φ), R = A1 cos φ + A2 sin φ, T = −A1 sin φ +
//!   A2 cos φ, med mittytans radie r₀ = r − t/2. Metriken G_SS = λ²,
//!   G_φφ = r₀², λ = 1 + r₀κ sin φ.
//! - Förskjutningen u = ū + ω × r₀R + u_r R + u_t T + y₃A3 (ū, ω interpoleras
//!   med Hₙ, ovaliseringen med Fouriertermer gånger Hₙ – eller 1 för
//!   ELBOW31B/C) och normalen n̄ = x,φ × a3 + r₀γ₃ a3.
//! - Koiter–Sanders töjningar ε_αβ = ½(g − G), κ_αβ = B − b + sym(B G⁻¹ ε)
//!   i fysikaliska komponenter (ε_SS/λ², ε_φφ/r₀², 2ε_Sφ/(λr₀)) och
//!   tvärskjuvningen γ_t = n · x,S (utan normering).

use super::super::common::{cross, dot, norm, scale, sub};
use crate::model::{fail, Result};

/// Ovaliseringsfältets komponent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comp {
    /// Radiell förskjutning u_r.
    Ur,
    /// Tangentiell förskjutning u_t.
    Ut,
    /// Välvning y₃ (längs axeln).
    Y3,
    /// Normalens lutning γ₃ längs axeln.
    G3,
}

/// Fourierfunktion: 1, sin pφ eller cos pφ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trig {
    One,
    Sin,
    Cos,
}

/// En ovaliseringsfrihetsgrad: komponent, Fourierordning p och funktion.
#[derive(Clone, Copy, Debug)]
pub struct Fld {
    pub comp: Comp,
    pub p: usize,
    pub trig: Trig,
}

/// Elementvariant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// ELBOW31, ELBOW32: ovalisering och välvning per nod.
    Full,
    /// ELBOW31B: bara ovalisering, konstant i elementet.
    B,
    /// ELBOW31C: som ELBOW31B utan udda moder p ≥ 3.
    C,
}

/// Ovaliseringsfrihetsgradernas ordning (per nod för ELBOW31/32,
/// per element för ELBOW31B/C). Moden p = 0: likformig u_r (och γ₃); p = 1:
/// u_r (och γ₃) med sin φ, sedan cos φ; p ≥ 2: först i krökens plan (u_r, y₃,
/// γ₃ med Q_sym(pφ) och u_t med Q_asym(pφ)), sedan ur planet med funktionerna
/// ombytta, där Q_sym = cos pφ för jämna p och sin pφ för udda.
pub fn fields(var: Variant, nmod: usize) -> Vec<Fld> {
    let f = |comp, p, trig| Fld { comp, p, trig };
    let qs = |p: usize| if p % 2 == 0 { Trig::Cos } else { Trig::Sin };
    let qa = |p: usize| if p % 2 == 0 { Trig::Sin } else { Trig::Cos };
    let mut v = Vec::new();
    match var {
        Variant::Full => {
            v.push(f(Comp::Ur, 0, Trig::One));
            v.push(f(Comp::G3, 0, Trig::One));
            for p in 1..=nmod {
                if p == 1 {
                    v.extend([f(Comp::Ur, 1, Trig::Sin), f(Comp::G3, 1, Trig::Sin), f(Comp::Ur, 1, Trig::Cos), f(Comp::G3, 1, Trig::Cos)]);
                } else {
                    for (a, b) in [(qs(p), qa(p)), (qa(p), qs(p))] {
                        v.extend([f(Comp::Ur, p, a), f(Comp::Ut, p, b), f(Comp::Y3, p, a), f(Comp::G3, p, a)]);
                    }
                }
            }
        }
        Variant::B | Variant::C => {
            v.push(f(Comp::Ur, 0, Trig::One));
            for p in 1..=nmod {
                if p == 1 {
                    v.extend([f(Comp::Ur, 1, Trig::Sin), f(Comp::Ur, 1, Trig::Cos)]);
                } else if var == Variant::C && p % 2 == 1 {
                    continue;
                } else {
                    for (a, b) in [(qs(p), qa(p)), (qa(p), qs(p))] {
                        v.extend([f(Comp::Ur, p, a), f(Comp::Ut, p, b)]);
                    }
                }
            }
        }
    }
    v
}

/// (ψ, ψ_φ, ψ_φφ) för Fourierfunktionen.
pub fn trig(t: Trig, p: usize, phi: f64) -> (f64, f64, f64) {
    let pf = p as f64;
    match t {
        Trig::One => (1.0, 0.0, 0.0),
        Trig::Sin => ((pf * phi).sin(), pf * (pf * phi).cos(), -pf * pf * (pf * phi).sin()),
        Trig::Cos => ((pf * phi).cos(), -pf * (pf * phi).sin(), -pf * pf * (pf * phi).cos()),
    }
}

/// Tvärsnittsdata från *BEAM SECTION, SECTION=ELBOW.
#[derive(Clone, Debug)]
pub struct Sec {
    /// Ytterradie, väggtjocklek och torusradie (0 = rakt rör).
    pub r: f64,
    pub t: f64,
    pub big_r: f64,
    /// Orienteringspunkten (a2 pekar mot den).
    pub opt: [f64; 3],
    /// Integrationspunkter genom väggen (udda) och runt röret (minst 8), antal moder.
    pub nt: usize,
    pub nc: usize,
    pub nmod: usize,
}

/// Elementets geometri.
pub struct Geo {
    pub nn: usize,
    pub x: Vec<[f64; 3]>,
    pub r0: f64,
    pub t: f64,
    pub kap: f64,
    pub l: f64,
    pub a1: [f64; 3],
    pub fields: Vec<Fld>,
    pub per_elem: bool,
    /// Antal ovaliseringsfrihetsgrader per nod (eller per element).
    pub nf: usize,
    /// Balkfrihetsgrader (6 per nod) och totalt.
    pub nb: usize,
    pub nd: usize,
}

/// Linjäriserade töjningar i en punkt: membrantöjning och krökning
/// (11, 22, 12 med ingenjörsskjuvning) som rader över elementets frihetsgrader,
/// tvärskjuvningen γ_t och areafaktorn J = λr₀.
pub struct Pt {
    pub be: [Vec<f64>; 3],
    pub bk: [Vec<f64>; 3],
    pub gt: Vec<f64>,
    pub j: f64,
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Linjärkombination av rader.
fn comb(nd: usize, terms: &[(f64, &[f64])]) -> Vec<f64> {
    let mut out = vec![0.0; nd];
    for (c, v) in terms {
        if *c != 0.0 {
            for (o, x) in out.iter_mut().zip(v.iter()) {
                *o += c * x;
            }
        }
    }
    out
}

impl Geo {
    /// `x`: elementets användarnoder (2 eller 3) i ordning.
    pub fn new(x: &[[f64; 3]], sec: &Sec, var: Variant, label: i64) -> Result<Geo> {
        let nn = x.len();
        let (x0, x1) = (x[0], x[nn - 1]);
        let d = sub(x1, x0);
        let lc = norm(d);
        if !(lc > 0.0) {
            return fail(format!("element {label} (ELBOW) har ingen längd"));
        }
        let a3 = scale(d, 1.0 / lc);
        let v = sub(sec.opt, x0);
        let a2 = sub(v, scale(a3, dot(v, a3)));
        if !(norm(a2) > 1e-10 * norm(v).max(lc)) {
            return fail(format!("element {label} (ELBOW): orienteringspunkten ligger på rörets axel"));
        }
        let a2 = unit(a2);
        let a1 = cross(a2, a3);
        let (kap, l) = if sec.big_r > 0.0 {
            if sec.big_r < lc / 2.0 {
                return fail(format!("element {label} (ELBOW): torusradien {} är mindre än halva elementlängden", sec.big_r));
            }
            (1.0 / sec.big_r, 2.0 * sec.big_r * (lc / (2.0 * sec.big_r)).asin())
        } else {
            (0.0, lc)
        };
        let fields = fields(var, sec.nmod);
        let nf = fields.len();
        let per_elem = var != Variant::Full;
        let nb = 6 * nn;
        let nd = nb + if per_elem { nf } else { nn * nf };
        Ok(Geo { nn, x: x.to_vec(), r0: sec.r - sec.t / 2.0, t: sec.t, kap, l, a1, fields, per_elem, nf, nb, nd })
    }

    /// Formfunktioner och deras derivator d/dξ i ξ ∈ [−1, 1].
    pub fn shape(&self, xi: f64) -> ([f64; 3], [f64; 3]) {
        if self.nn == 2 {
            ([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0, 0.0], [-0.5, 0.5, 0.0])
        } else {
            ([xi * (xi - 1.0) / 2.0, 1.0 - xi * xi, xi * (xi + 1.0) / 2.0], [xi - 0.5, -2.0 * xi, xi + 0.5])
        }
    }

    /// Stationer längs axeln: (S, vikt). En punkt för tvånodselementen,
    /// tvåpunkts Gauss för ELBOW32.
    pub fn stations(&self) -> Vec<(f64, f64)> {
        if self.nn == 2 {
            vec![(self.l / 2.0, self.l)]
        } else {
            let g = 1.0 / 3f64.sqrt();
            vec![(self.l / 2.0 * (1.0 - g), self.l / 2.0), (self.l / 2.0 * (1.0 + g), self.l / 2.0)]
        }
    }

    /// Riktarna [A1, A2, A3] i stationen S.
    pub fn frame(&self, s: f64) -> [[f64; 3]; 3] {
        let xi = 2.0 * s / self.l - 1.0;
        let (_, dh) = self.shape(xi);
        let mut tg = [0.0; 3];
        for n in 0..self.nn {
            tg = [0, 1, 2].map(|c| tg[c] + dh[n] * self.x[n][c]);
        }
        let a3 = unit(tg);
        let a1 = unit(sub(self.a1, scale(a3, dot(self.a1, a3))));
        [a1, cross(a3, a1), a3]
    }

    /// Vinkeln för integrationspunkt k (0-baserad) runt röret: punkt 1 vid
    /// intradosen (φ = −π/2), sedan moturs (mot A1 vid φ = 0).
    pub fn phi(nc: usize, k: usize) -> f64 {
        -std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * k as f64 / nc as f64
    }

    /// Index för ovaliseringsfrihetsgraden `k` hos användarnod `n` i elementvektorn.
    pub fn oval_index(&self, n: usize, k: usize) -> usize {
        if self.per_elem {
            self.nb + k
        } else {
            self.nb + n * self.nf + k
        }
    }

    /// Linjäriserade töjningar i (S, φ).
    pub fn point(&self, s: f64, phi: f64) -> Pt {
        let nd = self.nd;
        let (r0, kap) = (self.r0, self.kap);
        let [a1, a2, a3] = self.frame(s);
        let (sn, cs) = phi.sin_cos();
        let rv = [0, 1, 2].map(|c| a1[c] * cs + a2[c] * sn);
        let tv = [0, 1, 2].map(|c| -a1[c] * sn + a2[c] * cs);
        let lam = 1.0 + r0 * kap * sn;
        let xi = 2.0 * s / self.l - 1.0;
        let (h, dhx) = self.shape(xi);
        let dh = dhx.map(|v| v * 2.0 / self.l);
        // ū', ω och ω' projicerade på en riktning
        let beam = |v: [f64; 3], off: usize, w: &[f64; 3]| -> Vec<f64> {
            let mut z = vec![0.0; nd];
            for n in 0..self.nn {
                for c in 0..3 {
                    z[6 * n + off + c] = w[n] * v[c];
                }
            }
            z
        };
        let du = |v: [f64; 3]| beam(v, 0, &dh);
        let om = |v: [f64; 3]| beam(v, 3, &h);
        let om1 = |v: [f64; 3]| beam(v, 3, &dh);
        // ovaliseringsfälten: [komponent][värde, d/dS, d/dφ, d²/dφdS, d²/dφ²]
        let mut f = vec![vec![vec![0.0; nd]; 5]; 4];
        let nodes = if self.per_elem { 1 } else { self.nn };
        for n in 0..nodes {
            let (hn, dhn) = if self.per_elem { (1.0, 0.0) } else { (h[n], dh[n]) };
            for (k, fl) in self.fields.iter().enumerate() {
                let j = self.oval_index(n, k);
                let (ps, pf, pff) = trig(fl.trig, fl.p, phi);
                let c = fl.comp as usize;
                f[c][0][j] += hn * ps;
                f[c][1][j] += dhn * ps;
                f[c][2][j] += hn * pf;
                f[c][3][j] += dhn * pf;
                f[c][4][j] += hn * pff;
            }
        }
        let (ur, ut, y3, g3) = (&f[0], &f[1], &f[2], &f[3]);
        let (du_a3, du_t, du_r) = (du(a3), du(tv), du(rv));
        let (om_r, om_t, om_a3) = (om(rv), om(tv), om(a3));
        let (om1_t, om1_a3) = (om1(tv), om1(a3));
        // membrantöjningar (kovarianta)
        let e_ss = comb(nd, &[(lam, &du_a3), (-lam * r0, &om1_t), (lam * kap * sn, &ur[0]), (lam * kap * cs, &ut[0]), (lam, &y3[1])]);
        let e_pp = comb(nd, &[(r0, &ur[0]), (r0, &ut[2])]);
        let e2_sp = comb(nd, &[(r0, &om_r), (lam, &y3[2]), (r0, &du_t), (r0 * r0, &om1_a3), (r0, &ut[1]), (-r0 * kap * cs, &y3[0])]);
        // tvärskjuvning
        let gt = comb(nd, &[(1.0, &du_r), (-1.0, &om_t), (1.0, &ur[1]), (-kap * sn, &y3[0]), (lam, &g3[0])]);
        // normalens ändring δn = P T + Q A3 och dess derivator
        let p = comb(nd, &[(1.0, &om_a3), (-1.0 / r0, &ur[2]), (1.0 / r0, &ut[0])]);
        let q = comb(nd, &[(1.0, &g3[0]), (-1.0, &om_t)]);
        let p_f = comb(nd, &[(-1.0 / r0, &ur[4]), (1.0 / r0, &ut[2])]);
        let q_f = comb(nd, &[(1.0, &g3[2]), (1.0, &om_r)]);
        let p_s = comb(nd, &[(1.0, &om1_a3), (-kap * sn, &om_r), (-kap * cs, &om_t), (-1.0 / r0, &ur[3]), (1.0 / r0, &ut[1])]);
        let q_s = comb(nd, &[(1.0, &g3[1]), (-1.0, &om1_t), (-kap * cs, &om_a3)]);
        let tu_s = comb(nd, &[(1.0, &du_t), (r0, &om1_a3), (-r0 * kap * sn, &om_r), (1.0, &ut[1]), (-kap * cs, &y3[0])]);
        // −δb och Sanders term
        let k_ss = comb(nd, &[(kap * sn / lam, &e_ss), (lam * kap * cs, &p), (lam, &q_s), (-kap * sn / lam, &e_ss)]);
        let k_pp = comb(nd, &[(1.0, &ur[0]), (1.0, &ut[2]), (r0, &p_f), (-1.0 / r0, &e_pp)]);
        let k_sp = comb(
            nd,
            &[
                (0.5, &tu_s),
                (0.5 * lam, &q_f),
                (0.5 * kap * sn * r0, &om_r),
                (0.5 * kap * sn, &y3[2]),
                (0.5 * r0, &p_s),
                (-0.5 * r0 * kap * cs, &q),
                (0.25 * (-kap * sn / lam - 1.0 / r0), &e2_sp),
            ],
        );
        let sc = [1.0 / (lam * lam), 1.0 / (r0 * r0), 1.0 / (lam * r0)];
        let be = [scale_v(&e_ss, sc[0]), scale_v(&e_pp, sc[1]), scale_v(&e2_sp, sc[2])];
        let bk = [scale_v(&k_ss, sc[0]), scale_v(&k_pp, sc[1]), scale_v(&k_sp, 2.0 * sc[2])];
        Pt { be, bk, gt, j: lam * r0 }
    }
}

fn scale_v(v: &[f64], c: f64) -> Vec<f64> {
    v.iter().map(|x| x * c).collect()
}
