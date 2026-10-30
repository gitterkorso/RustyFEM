//! Kinematics for the pipe elbows (ELBOW31, ELBOW32, ELBOW31B, ELBOW31C):
//! reference geometry and linearized strains at a point (S, φ) on
//! the midsurface of the pipe wall. Included with `#[path]` from `elbow.rs`.
//!
//! The formulation is linearized about the reference position (straight and curved
//! elements with 0–6 modes):
//!
//! - The axis is a circular arc with the torus radius R through the element's end nodes
//!   (R = 0: straight), the arc length S ∈ [0, L]. The directors A1 = a2 × a3 from
//!   the orientation point (the normal to the plane of the elbow), A3 = the normalized
//!   tangent Σ dHₙ/dξ Xₙ (the chord for two-node elements) and A2 = A3 × A1 towards
//!   the extrados. The curvature κ = 1/R gives A2' = κA3 and A3' = −κA2.
//! - The wall position X = X̄ + r₀R(φ), R = A1 cos φ + A2 sin φ, T = −A1 sin φ +
//!   A2 cos φ, with the midsurface radius r₀ = r − t/2. The metric G_SS = λ²,
//!   G_φφ = r₀², λ = 1 + r₀κ sin φ.
//! - The displacement u = ū + ω × r₀R + u_r R + u_t T + y₃A3 (ū, ω are interpolated
//!   with Hₙ, the ovalization with Fourier terms times Hₙ – or 1 for
//!   ELBOW31B/C) and the normal n̄ = x,φ × a3 + r₀γ₃ a3.
//! - Koiter–Sanders strains ε_αβ = ½(g − G), κ_αβ = B − b + sym(B G⁻¹ ε)
//!   in physical components (ε_SS/λ², ε_φφ/r₀², 2ε_Sφ/(λr₀)) and
//!   the transverse shear γ_t = n · x,S (without normalization).

use super::super::common::{cross, dot, norm, scale, sub};
use crate::model::{fail, Result};

/// Component of the ovalization field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comp {
    /// Radial displacement u_r.
    Ur,
    /// Tangential displacement u_t.
    Ut,
    /// Warping y₃ (along the axis).
    Y3,
    /// Inclination γ₃ of the normal along the axis.
    G3,
}

/// Fourier function: 1, sin pφ or cos pφ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trig {
    One,
    Sin,
    Cos,
}

/// An ovalization degree of freedom: component, Fourier order p and function.
#[derive(Clone, Copy, Debug)]
pub struct Fld {
    pub comp: Comp,
    pub p: usize,
    pub trig: Trig,
}

/// Element variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// ELBOW31, ELBOW32: ovalization and warping per node.
    Full,
    /// ELBOW31B: ovalization only, constant in the element.
    B,
    /// ELBOW31C: like ELBOW31B without odd modes p ≥ 3.
    C,
}

/// Order of the ovalization degrees of freedom (per node for ELBOW31/32,
/// per element for ELBOW31B/C). The mode p = 0: uniform u_r (and γ₃); p = 1:
/// u_r (and γ₃) with sin φ, then cos φ; p ≥ 2: first in the plane of the elbow (u_r, y₃,
/// γ₃ with Q_sym(pφ) and u_t with Q_asym(pφ)), then out of plane with the functions
/// swapped, where Q_sym = cos pφ for even p and sin pφ for odd.
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

/// (ψ, ψ_φ, ψ_φφ) for the Fourier function.
pub fn trig(t: Trig, p: usize, phi: f64) -> (f64, f64, f64) {
    let pf = p as f64;
    match t {
        Trig::One => (1.0, 0.0, 0.0),
        Trig::Sin => ((pf * phi).sin(), pf * (pf * phi).cos(), -pf * pf * (pf * phi).sin()),
        Trig::Cos => ((pf * phi).cos(), -pf * (pf * phi).sin(), -pf * pf * (pf * phi).cos()),
    }
}

/// Section data from *BEAM SECTION, SECTION=ELBOW.
#[derive(Clone, Debug)]
pub struct Sec {
    /// Outer radius, wall thickness and torus radius (0 = straight pipe).
    pub r: f64,
    pub t: f64,
    pub big_r: f64,
    /// The orientation point (a2 points towards it).
    pub opt: [f64; 3],
    /// Integration points through the wall (odd) and around the pipe (at least 8), number of modes.
    pub nt: usize,
    pub nc: usize,
    pub nmod: usize,
}

/// The element geometry.
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
    /// Number of ovalization degrees of freedom per node (or per element).
    pub nf: usize,
    /// Beam degrees of freedom (6 per node) and total.
    pub nb: usize,
    pub nd: usize,
}

/// Linearized strains at a point: membrane strain and curvature
/// (11, 22, 12 with engineering shear) as rows over the element's degrees of freedom,
/// the transverse shear γ_t and the area factor J = λr₀.
pub struct Pt {
    pub be: [Vec<f64>; 3],
    pub bk: [Vec<f64>; 3],
    pub gt: Vec<f64>,
    pub j: f64,
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Linear combination of rows.
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
    /// `x`: the element's user nodes (2 or 3) in order.
    pub fn new(x: &[[f64; 3]], sec: &Sec, var: Variant, label: i64) -> Result<Geo> {
        let nn = x.len();
        let (x0, x1) = (x[0], x[nn - 1]);
        let d = sub(x1, x0);
        let lc = norm(d);
        if !(lc > 0.0) {
            return fail(format!("element {label} (ELBOW) has zero length"));
        }
        let a3 = scale(d, 1.0 / lc);
        let v = sub(sec.opt, x0);
        let a2 = sub(v, scale(a3, dot(v, a3)));
        if !(norm(a2) > 1e-10 * norm(v).max(lc)) {
            return fail(format!("element {label} (ELBOW): the orientation point lies on the pipe axis"));
        }
        let a2 = unit(a2);
        let a1 = cross(a2, a3);
        let (kap, l) = if sec.big_r > 0.0 {
            if sec.big_r < lc / 2.0 {
                return fail(format!("element {label} (ELBOW): the torus radius {} is less than half the element length", sec.big_r));
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

    /// Shape functions and their derivatives d/dξ at ξ ∈ [−1, 1].
    pub fn shape(&self, xi: f64) -> ([f64; 3], [f64; 3]) {
        if self.nn == 2 {
            ([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0, 0.0], [-0.5, 0.5, 0.0])
        } else {
            ([xi * (xi - 1.0) / 2.0, 1.0 - xi * xi, xi * (xi + 1.0) / 2.0], [xi - 0.5, -2.0 * xi, xi + 0.5])
        }
    }

    /// Stations along the axis: (S, weight). One point for the two-node elements,
    /// two-point Gauss for ELBOW32.
    pub fn stations(&self) -> Vec<(f64, f64)> {
        if self.nn == 2 {
            vec![(self.l / 2.0, self.l)]
        } else {
            let g = 1.0 / 3f64.sqrt();
            vec![(self.l / 2.0 * (1.0 - g), self.l / 2.0), (self.l / 2.0 * (1.0 + g), self.l / 2.0)]
        }
    }

    /// The directors [A1, A2, A3] at the station S.
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

    /// The angle for integration point k (0-based) around the pipe: point 1 at
    /// the intrados (φ = −π/2), then counterclockwise (towards A1 at φ = 0).
    pub fn phi(nc: usize, k: usize) -> f64 {
        -std::f64::consts::FRAC_PI_2 + 2.0 * std::f64::consts::PI * k as f64 / nc as f64
    }

    /// Index of the ovalization degree of freedom `k` of user node `n` in the element vector.
    pub fn oval_index(&self, n: usize, k: usize) -> usize {
        if self.per_elem {
            self.nb + k
        } else {
            self.nb + n * self.nf + k
        }
    }

    /// Linearized strains at (S, φ).
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
        // ū', ω and ω' projected onto a direction
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
        // the ovalization fields: [component][value, d/dS, d/dφ, d²/dφdS, d²/dφ²]
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
        // membrane strains (covariant)
        let e_ss = comb(nd, &[(lam, &du_a3), (-lam * r0, &om1_t), (lam * kap * sn, &ur[0]), (lam * kap * cs, &ut[0]), (lam, &y3[1])]);
        let e_pp = comb(nd, &[(r0, &ur[0]), (r0, &ut[2])]);
        let e2_sp = comb(nd, &[(r0, &om_r), (lam, &y3[2]), (r0, &du_t), (r0 * r0, &om1_a3), (r0, &ut[1]), (-r0 * kap * cs, &y3[0])]);
        // transverse shear
        let gt = comb(nd, &[(1.0, &du_r), (-1.0, &om_t), (1.0, &ur[1]), (-kap * sn, &y3[0]), (lam, &g3[0])]);
        // the change of the normal δn = P T + Q A3 and its derivatives
        let p = comb(nd, &[(1.0, &om_a3), (-1.0 / r0, &ur[2]), (1.0 / r0, &ut[0])]);
        let q = comb(nd, &[(1.0, &g3[0]), (-1.0, &om_t)]);
        let p_f = comb(nd, &[(-1.0 / r0, &ur[4]), (1.0 / r0, &ut[2])]);
        let q_f = comb(nd, &[(1.0, &g3[2]), (1.0, &om_r)]);
        let p_s = comb(nd, &[(1.0, &om1_a3), (-kap * sn, &om_r), (-kap * cs, &om_t), (-1.0 / r0, &ur[3]), (1.0 / r0, &ut[1])]);
        let q_s = comb(nd, &[(1.0, &g3[1]), (-1.0, &om1_t), (-kap * cs, &om_a3)]);
        let tu_s = comb(nd, &[(1.0, &du_t), (r0, &om1_a3), (-r0 * kap * sn, &om_r), (1.0, &ut[1]), (-kap * cs, &y3[0])]);
        // −δb and the Sanders term
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
