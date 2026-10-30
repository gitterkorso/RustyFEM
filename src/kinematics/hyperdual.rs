//! Hyperdual numbers: value, gradient and Hessian with respect to N variables.
//!
//! Used for elements whose internal force and tangent are derived from a
//! strain energy Φ(q): f = ∂Φ/∂q and K = ∂²Φ/∂q² become exact (no
//! numerical differentiation). The arithmetic rules are second-order forward differentiation
//! (Griewank & Walther, "Evaluating derivatives", ch. 13).

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug)]
pub struct Hd<const N: usize> {
    pub v: f64,
    pub g: [f64; N],
    pub h: [[f64; N]; N],
}

impl<const N: usize> Hd<N> {
    pub fn cst(v: f64) -> Self {
        Hd { v, g: [0.0; N], h: [[0.0; N]; N] }
    }

    /// Variable number `i` with value `v`.
    pub fn var(v: f64, i: usize) -> Self {
        let mut x = Self::cst(v);
        x.g[i] = 1.0;
        x
    }

    /// f(x) with derivatives f′ and f″ at x.
    fn chain(self, f: f64, d1: f64, d2: f64) -> Self {
        let mut r = Self::cst(f);
        for i in 0..N {
            r.g[i] = d1 * self.g[i];
            for j in 0..N {
                r.h[i][j] = d1 * self.h[i][j] + d2 * self.g[i] * self.g[j];
            }
        }
        r
    }

    pub fn sin(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(s, c, -s)
    }

    pub fn cos(self) -> Self {
        let (s, c) = self.v.sin_cos();
        self.chain(c, -s, -c)
    }

    pub fn sqrt(self) -> Self {
        let r = self.v.sqrt();
        self.chain(r, 0.5 / r, -0.25 / (r * self.v))
    }

    pub fn recip(self) -> Self {
        let r = 1.0 / self.v;
        self.chain(r, -r * r, 2.0 * r * r * r)
    }

    pub fn scale(self, s: f64) -> Self {
        let mut r = self;
        r.v *= s;
        for i in 0..N {
            r.g[i] *= s;
            for j in 0..N {
                r.h[i][j] *= s;
            }
        }
        r
    }
}

impl<const N: usize> Add for Hd<N> {
    type Output = Self;
    fn add(mut self, o: Self) -> Self {
        self.v += o.v;
        for i in 0..N {
            self.g[i] += o.g[i];
            for j in 0..N {
                self.h[i][j] += o.h[i][j];
            }
        }
        self
    }
}

impl<const N: usize> Sub for Hd<N> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + (-o)
    }
}

impl<const N: usize> Neg for Hd<N> {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl<const N: usize> Mul for Hd<N> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut r = Self::cst(self.v * o.v);
        for i in 0..N {
            r.g[i] = self.g[i] * o.v + self.v * o.g[i];
            for j in 0..N {
                r.h[i][j] = self.h[i][j] * o.v + self.v * o.h[i][j] + self.g[i] * o.g[j] + self.g[j] * o.g[i];
            }
        }
        r
    }
}

impl<const N: usize> Div for Hd<N> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        self * o.recip()
    }
}

impl<const N: usize> Add<f64> for Hd<N> {
    type Output = Self;
    fn add(mut self, o: f64) -> Self {
        self.v += o;
        self
    }
}

impl<const N: usize> Sub<f64> for Hd<N> {
    type Output = Self;
    fn sub(mut self, o: f64) -> Self {
        self.v -= o;
        self
    }
}

impl<const N: usize> Mul<f64> for Hd<N> {
    type Output = Self;
    fn mul(self, o: f64) -> Self {
        self.scale(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivatives() {
        // f = sin(x) y² / sqrt(x + y) at (x, y) = (0.7, 1.3), compared with differences
        let f = |x: Hd<2>, y: Hd<2>| x.sin() * y * y / (x + y).sqrt();
        let (x0, y0) = (0.7, 1.3);
        let r = f(Hd::var(x0, 0), Hd::var(y0, 1));
        let fv = |x: f64, y: f64| x.sin() * y * y / (x + y).sqrt();
        assert!((r.v - fv(x0, y0)).abs() < 1e-14);
        let h = 1e-5;
        let gx = (fv(x0 + h, y0) - fv(x0 - h, y0)) / (2.0 * h);
        let gy = (fv(x0, y0 + h) - fv(x0, y0 - h)) / (2.0 * h);
        assert!((r.g[0] - gx).abs() < 1e-9 && (r.g[1] - gy).abs() < 1e-9);
        let hxy = (fv(x0 + h, y0 + h) - fv(x0 + h, y0 - h) - fv(x0 - h, y0 + h) + fv(x0 - h, y0 - h)) / (4.0 * h * h);
        let hxx = (fv(x0 + h, y0) - 2.0 * fv(x0, y0) + fv(x0 - h, y0)) / (h * h);
        assert!((r.h[0][1] - hxy).abs() < 1e-5 && (r.h[1][0] - hxy).abs() < 1e-5);
        assert!((r.h[0][0] - hxx).abs() < 1e-4);
    }
}

/// Hyperdual 3-vector and 3×3 matrix.
pub type V3<const N: usize> = [Hd<N>; 3];
pub type T3<const N: usize> = [[Hd<N>; 3]; 3];

pub fn dot3<const N: usize>(a: &V3<N>, b: &V3<N>) -> Hd<N> {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3<const N: usize>(a: &V3<N>, b: &V3<N>) -> V3<N> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn matvec3<const N: usize>(m: &T3<N>, v: &V3<N>) -> V3<N> {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// The coefficients in Rodrigues' formula as functions of s = θ²:
/// k = 0: sin θ/θ, k = 1: (1 − cos θ)/θ², k = 2: (θ − sin θ)/θ³.
/// Returns (f, df/ds, d²f/ds²); series expansion for small θ.
fn rodrigues_coef(k: usize, s: f64) -> (f64, f64, f64) {
    if s < 0.25 {
        // f = Σ (−1)^j s^j / (2j + 1 + k)!
        let (mut f, mut d1, mut d2) = (0.0, 0.0, 0.0);
        let mut fact = (1..=(1 + k)).map(|x| x as f64).product::<f64>();
        let mut sj = 1.0;
        for j in 0..20usize {
            if j > 0 {
                let n = 2 * j + 1 + k;
                fact *= ((n - 1) * n) as f64;
            }
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let c = sign / fact;
            f += c * sj;
            if j >= 1 {
                d1 += c * j as f64 * sj / s.max(1e-300) * if s == 0.0 { 0.0 } else { 1.0 };
            }
            if j >= 2 {
                d2 += c * (j * (j - 1)) as f64 * sj / (s * s).max(1e-300) * if s == 0.0 { 0.0 } else { 1.0 };
            }
            sj *= s;
        }
        // s = 0: the derivatives are the coefficients of s¹ and 2·s²
        if s == 0.0 {
            let f3 = |n: usize| (1..=n).map(|x| x as f64).product::<f64>();
            d1 = -1.0 / f3(3 + k);
            d2 = 2.0 / f3(5 + k);
        }
        return (f, d1, d2);
    }
    let t = s.sqrt();
    let (sn, cs) = t.sin_cos();
    // f(θ), f′(θ), f″(θ)
    let (f, fp, fpp) = match k {
        0 => (sn / t, (t * cs - sn) / (t * t), (-t * t * sn - 2.0 * t * cs + 2.0 * sn) / (t * t * t)),
        1 => {
            let f = (1.0 - cs) / (t * t);
            let fp = sn / (t * t) - 2.0 * (1.0 - cs) / (t * t * t);
            let fpp = cs / (t * t) - 4.0 * sn / (t * t * t) + 6.0 * (1.0 - cs) / (t * t * t * t);
            (f, fp, fpp)
        }
        _ => {
            let f = (t - sn) / (t * t * t);
            let fp = (1.0 - cs) / (t * t * t) - 3.0 * (t - sn) / (t * t * t * t);
            let fpp = sn / (t * t * t) - 6.0 * (1.0 - cs) / (t * t * t * t) + 12.0 * (t - sn) / (t * t * t * t * t);
            (f, fp, fpp)
        }
    };
    (f, fp / (2.0 * t), (fpp - fp / t) / (4.0 * t * t))
}

fn coef<const N: usize>(k: usize, s: Hd<N>) -> Hd<N> {
    let (f, d1, d2) = rodrigues_coef(k, s.v);
    let mut r = Hd::cst(f);
    for i in 0..N {
        r.g[i] = d1 * s.g[i];
        for j in 0..N {
            r.h[i][j] = d1 * s.h[i][j] + d2 * s.g[i] * s.g[j];
        }
    }
    r
}

fn skew3<const N: usize>(p: &V3<N>) -> T3<N> {
    let z = Hd::cst(0.0);
    [[z, -p[2], p[1]], [p[2], z, -p[0]], [-p[1], p[0], z]]
}

fn mat_mul3<const N: usize>(a: &T3<N>, b: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}

/// The rotation matrix C(ψ) = I + a Ψ + b Ψ² for the rotation vector ψ.
pub fn exp_so3<const N: usize>(p: &V3<N>) -> T3<N> {
    let s = dot3(p, p);
    let (a, b) = (coef(0, s), coef(1, s));
    let w = skew3(p);
    let w2 = mat_mul3(&w, &w);
    std::array::from_fn(|i| std::array::from_fn(|j| w[i][j] * a + w2[i][j] * b + if i == j { 1.0 } else { 0.0 }))
}

/// The tangent operator T(ψ) = I + b Ψ + c Ψ², dC = [T dψ]× C.
pub fn tangent_so3<const N: usize>(p: &V3<N>) -> T3<N> {
    let s = dot3(p, p);
    let (b, c) = (coef(1, s), coef(2, s));
    let w = skew3(p);
    let w2 = mat_mul3(&w, &w);
    std::array::from_fn(|i| std::array::from_fn(|j| w[i][j] * b + w2[i][j] * c + if i == j { 1.0 } else { 0.0 }))
}

#[cfg(test)]
mod so3_tests {
    use super::*;

    #[test]
    fn rodrigues_coefficients() {
        for &s in &[0.0, 1e-6, 0.2, 0.3, 2.0, 30.0] {
            for k in 0..3 {
                let f = |s: f64| rodrigues_coef(k, s).0;
                let (v, d1, d2) = rodrigues_coef(k, s);
                let h = 1e-4 * (1.0 + s);
                let (fp, fm) = (f(s + h), f((s - h).max(0.0)));
                if s > h {
                    assert!(((fp - fm) / (2.0 * h) - d1).abs() < 1e-7, "k {k} s {s}");
                    assert!(((fp - 2.0 * v + fm) / (h * h) - d2).abs() < 1e-4, "k {k} s {s}: {d2}");
                }
            }
        }
        // continuity at s = 0.25
        for k in 0..3 {
            let (a, b) = (rodrigues_coef(k, 0.25 - 1e-12), rodrigues_coef(k, 0.25 + 1e-12));
            assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-10 && (a.2 - b.2).abs() < 1e-8, "k {k}");
        }
    }
}
