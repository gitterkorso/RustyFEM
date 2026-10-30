//! Dense complex matrices for the projected (modal) problems in
//! *COMPLEX FREQUENCY, *RANDOM RESPONSE and the SUBSPACE projections.
//!
//! - `eigenvalues`: all eigenvalues of a general complex matrix
//!   (Hessenberg reduction with stabilized elimination as in EISPACK `comhes`,
//!   followed by QR iteration with a single Wilkinson shift and Givens rotations).
//! - `eigenvector`: eigenvector for a known eigenvalue by inverse iteration.
//! - `solve`: LU with partial pivoting.
//!
//! The matrices are row-major `n × n`.

/// Complex number.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Cx {
    pub re: f64,
    pub im: f64,
}

impl Cx {
    pub const ZERO: Cx = Cx { re: 0.0, im: 0.0 };
    pub const ONE: Cx = Cx { re: 1.0, im: 0.0 };
    pub fn new(re: f64, im: f64) -> Cx {
        Cx { re, im }
    }
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    pub fn norm2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn conj(self) -> Cx {
        Cx::new(self.re, -self.im)
    }
    pub fn scale(self, s: f64) -> Cx {
        Cx::new(self.re * s, self.im * s)
    }
    pub fn sqrt(self) -> Cx {
        let r = self.abs();
        if r == 0.0 {
            return Cx::ZERO;
        }
        let a = ((r + self.re.abs()) / 2.0).sqrt();
        if self.re >= 0.0 {
            Cx::new(a, self.im / (2.0 * a))
        } else {
            Cx::new(self.im.abs() / (2.0 * a), if self.im >= 0.0 { a } else { -a })
        }
    }
}

impl std::ops::Add for Cx {
    type Output = Cx;
    fn add(self, o: Cx) -> Cx {
        Cx::new(self.re + o.re, self.im + o.im)
    }
}
impl std::ops::Sub for Cx {
    type Output = Cx;
    fn sub(self, o: Cx) -> Cx {
        Cx::new(self.re - o.re, self.im - o.im)
    }
}
impl std::ops::Mul for Cx {
    type Output = Cx;
    fn mul(self, o: Cx) -> Cx {
        Cx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}
impl std::ops::Div for Cx {
    type Output = Cx;
    fn div(self, o: Cx) -> Cx {
        // Smith's algorithm
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            Cx::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            Cx::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
}
impl std::ops::Neg for Cx {
    type Output = Cx;
    fn neg(self) -> Cx {
        Cx::new(-self.re, -self.im)
    }
}
impl std::ops::AddAssign for Cx {
    fn add_assign(&mut self, o: Cx) {
        self.re += o.re;
        self.im += o.im;
    }
}
impl std::ops::SubAssign for Cx {
    fn sub_assign(&mut self, o: Cx) {
        self.re -= o.re;
        self.im -= o.im;
    }
}

/// Solves A x = b (A row-major n × n) with LU and partial pivoting. Pivots
/// that are exactly zero are replaced by a small number (for inverse iteration).
pub fn solve(a: &[Cx], n: usize, b: &[Cx]) -> Vec<Cx> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    let scale = m.iter().map(|v| v.abs()).fold(0.0, f64::max).max(f64::MIN_POSITIVE);
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs())).unwrap();
        if p != k {
            for c in 0..n {
                m.swap(k * n + c, p * n + c);
            }
            x.swap(k, p);
        }
        if m[k * n + k].abs() <= 1e-300 {
            m[k * n + k] = Cx::new(scale * 1e-15, 0.0);
        }
        let piv = m[k * n + k];
        for i in k + 1..n {
            let f = m[i * n + k] / piv;
            if f == Cx::ZERO {
                continue;
            }
            for c in k + 1..n {
                let v = m[k * n + c];
                m[i * n + c] -= f * v;
            }
            m[i * n + k] = Cx::ZERO;
            let xk = x[k];
            x[i] -= f * xk;
        }
    }
    for k in (0..n).rev() {
        let mut s = x[k];
        for c in k + 1..n {
            s -= m[k * n + c] * x[c];
        }
        x[k] = s / m[k * n + k];
    }
    x
}

/// All eigenvalues of the general complex matrix `a` (row-major n × n).
pub fn eigenvalues(a: &[Cx], n: usize) -> Vec<Cx> {
    let mut h = a.to_vec();
    let at = |i: usize, j: usize| i * n + j;
    // Hessenberg form with stabilized elimination (similarity transformations)
    for m in 1..n.saturating_sub(1) {
        let p = (m..n).max_by(|&i, &j| h[at(i, m - 1)].abs().total_cmp(&h[at(j, m - 1)].abs())).unwrap();
        if h[at(p, m - 1)].abs() == 0.0 {
            continue;
        }
        if p != m {
            for c in 0..n {
                h.swap(at(p, c), at(m, c));
            }
            for r in 0..n {
                h.swap(at(r, p), at(r, m));
            }
        }
        let piv = h[at(m, m - 1)];
        for i in m + 1..n {
            let t = h[at(i, m - 1)] / piv;
            if t == Cx::ZERO {
                continue;
            }
            for c in 0..n {
                let v = h[at(m, c)];
                h[at(i, c)] -= t * v;
            }
            for r in 0..n {
                let v = h[at(r, i)];
                h[at(r, m)] += t * v;
            }
        }
    }
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            h[at(i, j)] = Cx::ZERO;
        }
    }
    // QR iteration with Wilkinson shift
    let mut ev = vec![Cx::ZERO; n];
    if n == 0 {
        return ev;
    }
    let mut hi = n - 1;
    let mut its = 0usize;
    loop {
        if hi == 0 {
            ev[0] = h[at(0, 0)];
            break;
        }
        // search for a negligible subdiagonal
        let mut l = hi;
        while l > 0 {
            let s = h[at(l - 1, l - 1)].abs() + h[at(l, l)].abs();
            if h[at(l, l - 1)].abs() <= f64::EPSILON * s.max(f64::MIN_POSITIVE) {
                h[at(l, l - 1)] = Cx::ZERO;
                break;
            }
            l -= 1;
        }
        if l == hi {
            ev[hi] = h[at(hi, hi)];
            hi -= 1;
            its = 0;
            continue;
        }
        its += 1;
        if its > 300 {
            // give up: the diagonal as an approximation
            for i in 0..=hi {
                ev[i] = h[at(i, i)];
            }
            break;
        }
        let mu = if its % 11 == 10 {
            h[at(hi, hi)] + Cx::new(h[at(hi, hi - 1)].abs(), 0.0)
        } else {
            let (a, b, c, d) = (h[at(hi - 1, hi - 1)], h[at(hi - 1, hi)], h[at(hi, hi - 1)], h[at(hi, hi)]);
            let half = (a - d).scale(0.5);
            let disc = (half * half + b * c).sqrt();
            let m = (a + d).scale(0.5);
            let (m1, m2) = (m + disc, m - disc);
            if (m1 - d).abs() <= (m2 - d).abs() { m1 } else { m2 }
        };
        for k in l..=hi {
            h[at(k, k)] -= mu;
        }
        let mut rots: Vec<(Cx, Cx)> = Vec::with_capacity(hi - l);
        for k in l..hi {
            let (x, y) = (h[at(k, k)], h[at(k + 1, k)]);
            let r = (x.norm2() + y.norm2()).sqrt();
            let (c, s) = if r == 0.0 { (Cx::ONE, Cx::ZERO) } else { (x.scale(1.0 / r), y.scale(1.0 / r)) };
            for col in k..=hi {
                let (p, q) = (h[at(k, col)], h[at(k + 1, col)]);
                h[at(k, col)] = c.conj() * p + s.conj() * q;
                h[at(k + 1, col)] = c * q - s * p;
            }
            rots.push((c, s));
        }
        for (j, &(c, s)) in rots.iter().enumerate() {
            let k = l + j;
            for row in l..=(k + 2).min(hi) {
                let (p, q) = (h[at(row, k)], h[at(row, k + 1)]);
                h[at(row, k)] = p * c + q * s;
                h[at(row, k + 1)] = q * c.conj() - p * s.conj();
            }
        }
        for k in l..=hi {
            h[at(k, k)] += mu;
        }
    }
    ev
}

/// Eigenvector for the eigenvalue `mu` by inverse iteration (normalized to |x| = 1).
pub fn eigenvector(a: &[Cx], n: usize, mu: Cx) -> Vec<Cx> {
    let scale = a.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1.0);
    let shift = mu + Cx::new(scale * 1e-13, scale * 1e-13);
    let mut m = a.to_vec();
    for i in 0..n {
        m[i * n + i] -= shift;
    }
    let mut x: Vec<Cx> = (0..n).map(|i| Cx::new(1.0 + 0.1 * i as f64, 0.3 - 0.05 * i as f64)).collect();
    for _ in 0..4 {
        x = solve(&m, n, &x);
        let nrm = x.iter().map(|v| v.norm2()).sum::<f64>().sqrt();
        if nrm == 0.0 || !nrm.is_finite() {
            break;
        }
        x.iter_mut().for_each(|v| *v = v.scale(1.0 / nrm));
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalues_of_damped_oscillators() {
        // two decoupled damped oscillators in first-order form
        let (w, z) = ([3.0f64, 7.0], [0.05f64, 0.2]);
        let n = 4;
        let mut a = vec![Cx::ZERO; n * n];
        for i in 0..2 {
            a[i * n + 2 + i] = Cx::ONE;
            a[(2 + i) * n + i] = Cx::new(-w[i] * w[i], 0.0);
            a[(2 + i) * n + 2 + i] = Cx::new(-2.0 * z[i] * w[i], 0.0);
        }
        let mut ev = eigenvalues(&a, n);
        ev.retain(|e| e.im > 0.0);
        ev.sort_by(|a, b| a.im.total_cmp(&b.im));
        for i in 0..2 {
            assert!((ev[i].re + z[i] * w[i]).abs() < 1e-12, "{:?}", ev);
            assert!((ev[i].im - w[i] * (1.0 - z[i] * z[i]).sqrt()).abs() < 1e-12);
            let v = eigenvector(&a, n, ev[i]);
            // A v = μ v
            for r in 0..n {
                let mut s = Cx::ZERO;
                for c in 0..n {
                    s += a[r * n + c] * v[c];
                }
                assert!((s - ev[i] * v[r]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn solve_complex_system() {
        let a = vec![Cx::new(2.0, 1.0), Cx::new(0.0, -1.0), Cx::new(1.0, 0.0), Cx::new(3.0, 0.5)];
        let x = vec![Cx::new(1.0, 2.0), Cx::new(-0.5, 0.25)];
        let b = vec![a[0] * x[0] + a[1] * x[1], a[2] * x[0] + a[3] * x[1]];
        let y = solve(&a, 2, &b);
        assert!((y[0] - x[0]).abs() < 1e-14 && (y[1] - x[1]).abs() < 1e-14);
    }
}
