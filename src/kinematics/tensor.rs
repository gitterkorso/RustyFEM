//! Små 3×3-tensoroperationer för stora deformationer: produkter, invers,
//! egenvärden för symmetriska matriser (Jacobi), matrislogaritm, polär
//! uppdelning och rotationer (Rodrigues/exponentialavbildning och dess invers).

pub type M3 = [[f64; 3]; 3];

pub const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn zero() -> M3 {
    [[0.0; 3]; 3]
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut c = zero();
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

pub fn transpose(a: &M3) -> M3 {
    let mut c = zero();
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[j][i];
        }
    }
    c
}

/// A Bᵀ
pub fn mul_bt(a: &M3, b: &M3) -> M3 {
    mul(a, &transpose(b))
}

/// Aᵀ B
pub fn mul_at(a: &M3, b: &M3) -> M3 {
    mul(&transpose(a), b)
}

pub fn add(a: &M3, b: &M3) -> M3 {
    let mut c = *a;
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] += b[i][j];
        }
    }
    c
}

pub fn sub(a: &M3, b: &M3) -> M3 {
    let mut c = *a;
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] -= b[i][j];
        }
    }
    c
}

pub fn scale(a: &M3, s: f64) -> M3 {
    let mut c = *a;
    for r in c.iter_mut() {
        for v in r.iter_mut() {
            *v *= s;
        }
    }
    c
}

pub fn mv(a: &M3, x: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| a[i][0] * x[0] + a[i][1] * x[1] + a[i][2] * x[2])
}

pub fn det(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

pub fn inv(m: &M3) -> (M3, f64) {
    crate::elements::common::inv3(*m)
}

/// R A Rᵀ
pub fn rotate(r: &M3, a: &M3) -> M3 {
    mul(&mul(r, a), &transpose(r))
}

/// Rᵀ A R
pub fn rotate_back(r: &M3, a: &M3) -> M3 {
    mul(&mul(&transpose(r), a), r)
}

pub fn sym(a: &M3) -> M3 {
    let mut c = zero();
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = 0.5 * (a[i][j] + a[j][i]);
        }
    }
    c
}

pub fn skew(a: &M3) -> M3 {
    let mut c = zero();
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = 0.5 * (a[i][j] - a[j][i]);
        }
    }
    c
}

/// Egenvärden och egenvektorer (kolumner i den returnerade matrisen) för en
/// symmetrisk matris, med Jacobis rotationsmetod.
pub fn eig_sym(a: &M3) -> ([f64; 3], M3) {
    let mut a = *a;
    let mut v = I3;
    for _ in 0..60 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let diag = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-300 || off <= 1e-18 * diag {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() <= 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = if theta == 0.0 { 1.0 } else { theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()) };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p][k];
                let aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k][p];
                let vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// f(A) = Σ f(λᵢ) nᵢ⊗nᵢ för en symmetrisk matris.
pub fn sym_fn(a: &M3, f: impl Fn(f64) -> f64) -> M3 {
    let (l, v) = eig_sym(a);
    let mut c = zero();
    for k in 0..3 {
        let fk = f(l[k]);
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] += fk * v[i][k] * v[j][k];
            }
        }
    }
    c
}

/// Logaritmisk töjning ln V = ½ ln(F Fᵀ) (rumsliga axlar).
pub fn log_strain(f: &M3) -> M3 {
    sym_fn(&mul_bt(f, f), |x| 0.5 * x.max(1e-300).ln())
}

/// Polär uppdelning F = R U; returnerar (R, U).
pub fn polar(f: &M3) -> (M3, M3) {
    let c = mul_at(f, f);
    let u = sym_fn(&c, |x| x.max(0.0).sqrt());
    let ui = sym_fn(&c, |x| 1.0 / x.max(1e-300).sqrt());
    (mul(f, &ui), u)
}

/// Snedsymmetrisk matris [w]× så att [w]× x = w × x.
pub fn spin(w: [f64; 3]) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// Axialvektorn till den snedsymmetriska delen av A.
pub fn axial(a: &M3) -> [f64; 3] {
    [0.5 * (a[2][1] - a[1][2]), 0.5 * (a[0][2] - a[2][0]), 0.5 * (a[1][0] - a[0][1])]
}

/// Rotationsmatris exp([θ]×) (Rodrigues formel).
pub fn exp_so3(t: [f64; 3]) -> M3 {
    let a2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
    let a = a2.sqrt();
    let (s1, s2) = if a < 1e-4 {
        (1.0 - a2 / 6.0 + a2 * a2 / 120.0, 0.5 - a2 / 24.0 + a2 * a2 / 720.0)
    } else {
        (a.sin() / a, (1.0 - a.cos()) / a2)
    };
    let w = spin(t);
    let w2 = mul(&w, &w);
    add(&add(&I3, &scale(&w, s1)), &scale(&w2, s2))
}

/// Rotationsvektor θ med exp([θ]×) = R och |θ| ≤ π (via kvaternion, stabil nära π).
pub fn log_so3(r: &M3) -> [f64; 3] {
    let tr = r[0][0] + r[1][1] + r[2][2];
    let m = [tr, r[0][0], r[1][1], r[2][2]];
    let mut k = 0;
    for i in 1..4 {
        if m[i] > m[k] {
            k = i;
        }
    }
    let mut q = [0.0; 4];
    if k == 0 {
        q[0] = 0.5 * (1.0 + tr).sqrt();
        let f = 0.25 / q[0];
        q[1] = (r[2][1] - r[1][2]) * f;
        q[2] = (r[0][2] - r[2][0]) * f;
        q[3] = (r[1][0] - r[0][1]) * f;
    } else {
        let i = k - 1;
        let j = (i + 1) % 3;
        let l = (i + 2) % 3;
        let qi = (0.5 * r[i][i] + 0.25 * (1.0 - tr)).max(0.0).sqrt();
        let f = 0.25 / qi;
        q[i + 1] = qi;
        q[0] = (r[l][j] - r[j][l]) * f;
        q[j + 1] = (r[j][i] + r[i][j]) * f;
        q[l + 1] = (r[l][i] + r[i][l]) * f;
    }
    if q[0] < 0.0 {
        q = q.map(|x| -x);
    }
    let sn = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if sn < 1e-300 {
        return [0.0; 3];
    }
    let ang = 2.0 * sn.atan2(q[0]);
    let f = ang / sn;
    [q[1] * f, q[2] * f, q[3] * f]
}

/// Rotationsvektor för R som ligger närmast `near` (lägger till 2πk längs
/// axeln), så att totala rotationer kan växa förbi π utan hopp.
pub fn log_so3_near(r: &M3, near: [f64; 3]) -> [f64; 3] {
    let two_pi = 2.0 * std::f64::consts::PI;
    let t = log_so3(r);
    let a = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    let n = (near[0] * near[0] + near[1] * near[1] + near[2] * near[2]).sqrt();
    if n < 3.0 {
        return t;
    }
    let e = if a > 1e-12 { [t[0] / a, t[1] / a, t[2] / a] } else { [near[0] / n, near[1] / n, near[2] / n] };
    let mut best = t;
    let mut bd = f64::MAX;
    for k in -4i32..=4 {
        for sgn in [1.0, -1.0] {
            let ang = sgn * a + two_pi * k as f64;
            let c = [e[0] * ang, e[1] * ang, e[2] * ang];
            let d = (0..3).map(|i| (c[i] - near[i]).powi(2)).sum::<f64>();
            if d < bd {
                bd = d;
                best = c;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        for t in [[0.1, -0.2, 0.3], [2.0, 1.0, -0.5], [0.0, 0.0, 3.1], [1e-9, 0.0, 0.0]] {
            let r = exp_so3(t);
            let b = log_so3(&r);
            for i in 0..3 {
                assert!((b[i] - t[i]).abs() < 1e-10, "{t:?} {b:?}");
            }
        }
        let t = [0.0, 0.0, 4.0];
        let b = log_so3_near(&exp_so3(t), [0.0, 0.0, 3.9]);
        assert!((b[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn polar_and_log() {
        let f = [[1.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (r, u) = polar(&f);
        let fr = mul(&r, &u);
        for i in 0..3 {
            for j in 0..3 {
                assert!((fr[i][j] - f[i][j]).abs() < 1e-12);
            }
        }
        // ln V = ½ ln(F Fᵀ) för F ovan (egenvärdesuppdelning med numpy)
        let le = log_strain(&f);
        assert!((le[0][0] - 0.439708007951929).abs() < 1e-12, "{le:?}");
        assert!((le[1][1] + 0.03424289984376449).abs() < 1e-12, "{le:?}");
        assert!((le[0][1] - 0.1579836359318978).abs() < 1e-12, "{le:?}");
    }
}
