//! Curved Euler–Bernoulli beams B23/B33 (area beams3): when the nodal normals
//! (averaged, `elements/normals.rs`, or from *NORMAL) are not
//! perpendicular to the chord, the element is initially curved.
//!
//! Formulation (linearized):
//!
//! - The axis is a Hermite curve X(ξ) = N₁X₁ + N₂X₂ + J(N₃T₁ + N₄T₂) with
//!   the nodal tangents Tₙ perpendicular to the nodal normals (the chord projected) and
//!   J = |dX/dξ| at the element center (fixed point, a quadratic equation), S = Jξ.
//! - The displacement is interpolated the same way with the nodal values u, the rotations
//!   φ and the stretch λ at the ends (an internal node with two degrees of freedom):
//!   du/dξ = J(λₙTₙ + φₙ × Tₙ); the rotation about the axis linearly.
//! - Strains: ε = G⁻¹X'·u', the curvatures
//!   K₂ = G⁻¹(δn₁·X'' + N₁·u'' − ε N₁·X''),
//!   K₁ = −G⁻¹(δn₂·X'' + N₂·u'' − ε N₂·X'') and the twist
//!   e₁ = ω'·t + ω·(N₁' × N₂) − g^{-1/2}(N₂·u')(t·N₁'), with three Gauss points
//!   and dL = dS = J dξ.
//! - The mass: ρA∫uᵀu dL (three Gauss points) and ρ(I₁₁ + I₂₂) for the torsion,
//!   no rotary inertia for bending (like the straight elements).
//!
//! λ is condensed out in
//! `stiffness` and added as internal degrees of freedom in eigenvalue steps
//! (`beam_mass::internal_dofs`). NLGEOM still uses straight elements.

use super::api::{ElemCtx, ElementType};
use super::beam_eb::EulerBeam;
use super::common::{cross, dot, norm, scale, sub};
use crate::model::{fail, Result};

/// The limit for when a nodal normal counts as inclined to the chord.
const TILT: f64 = 1e-10;

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The Hermite functions and their derivatives at ξ ∈ [−1, 1].
fn herm(xi: f64) -> ([f64; 4], [f64; 4], [f64; 4]) {
    (
        [(1.0 - xi).powi(2) * (2.0 + xi) / 4.0, (1.0 + xi).powi(2) * (2.0 - xi) / 4.0, (1.0 - xi).powi(2) * (1.0 + xi) / 4.0, -(1.0 + xi).powi(2) * (1.0 - xi) / 4.0],
        [-3.0 * (1.0 - xi * xi) / 4.0, 3.0 * (1.0 - xi * xi) / 4.0, (1.0 - xi) * (-1.0 - 3.0 * xi) / 4.0, (1.0 + xi) * (3.0 * xi - 1.0) / 4.0],
        [1.5 * xi, -1.5 * xi, (-1.0 + 3.0 * xi) / 2.0, (1.0 + 3.0 * xi) / 2.0],
    )
}

const GAUSS3: [(f64, f64); 3] = [(-0.774_596_669_241_483_4, 5.0 / 9.0), (0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)];

/// A curved element: geometry and stiffnesses.
pub struct Curved {
    /// The element's degrees of freedom (6 for B23, 12 for B33) and the total with λ₁, λ₂.
    pub n: usize,
    pub ne: usize,
    x: [[f64; 3]; 2],
    t: [[f64; 3]; 2],
    n1: [[f64; 3]; 2],
    j: f64,
    ea: f64,
    ei11: f64,
    ei22: f64,
    gj: f64,
    rho_a: f64,
    rho_ip: f64,
}

/// Rows at a point: strains (ε, K₁, K₂, e₁) and displacement u (3 rows).
struct Rows {
    e: [Vec<f64>; 4],
    u: [Vec<f64>; 3],
    tw: Vec<f64>,
    dl: f64,
}

impl Curved {
    /// The element's curved data, or `None` if the nodal normals are perpendicular to
    /// the chord and equal (straight element).
    pub fn of(el: &EulerBeam, ctx: &ElemCtx) -> Result<Option<Curved>> {
        let Some(nrm) = super::normals::of(ctx) else { return Ok(None) };
        if nrm.len() < 2 {
            return Ok(None);
        }
        let (x0, x1) = (ctx.x[0], ctx.x[1]);
        let c = sub(x1, x0);
        let l = norm(c);
        let cu = scale(c, 1.0 / l);
        let n2 = [unit(nrm[0]), unit(nrm[1])];
        let tilted = n2.iter().any(|n| dot(*n, cu).abs() > TILT) || norm(sub(n2[0], n2[1])) > TILT;
        if !tilted {
            return Ok(None);
        }
        let d = el.data(ctx)?;
        let (k, nd) = (&d.k_local, d.n);
        let (ea, ei11, ei22, gj) = if nd == 6 {
            (k[0] * l, k[nd + 1] * l.powi(3) / 12.0, 0.0, 0.0)
        } else {
            (k[0] * l, k[2 * nd + 2] * l.powi(3) / 12.0, k[nd + 1] * l.powi(3) / 12.0, k[3 * nd + 3] * l)
        };
        let t: [[f64; 3]; 2] = std::array::from_fn(|i| unit(sub(c, scale(n2[i], dot(c, n2[i])))));
        if t.iter().any(|v| !(norm(*v) > 0.5)) {
            return fail(format!("element {} ({}): the nodal normal is parallel to the element", ctx.label, el.name()));
        }
        let n1: [[f64; 3]; 2] = std::array::from_fn(|i| cross(n2[i], t[i]));
        // J = |dX/dξ| at the center (fixed point): J = |¾c − ¼J(T₁ + T₂)|, a quadratic equation
        let ts = add3(t[0], t[1]);
        let (qa, qb, qc) = (1.0 - dot(ts, ts) / 16.0, 0.375 * dot(c, ts), -0.5625 * l * l);
        let j = (-qb + (qb * qb - 4.0 * qa * qc).sqrt()) / (2.0 * qa);
        let ip = ctx.section(el.name())?.beam().map(|b| b.profile.i11 + b.profile.i22).unwrap_or(0.0);
        Ok(Some(Curved { n: nd, ne: nd + 2, x: [x0, x1], t, n1, j, ea, ei11, ei22, gj, rho_a: d.density * d.area, rho_ip: d.density * ip }))
    }

    /// The column in the element vector for node `a`, component `c` (0–2 displacement,
    /// 3–5 rotation) or `None` if B23 lacks it.
    fn col(&self, a: usize, c: usize) -> Option<usize> {
        if self.n == 12 {
            Some(6 * a + c)
        } else {
            match c {
                0 | 1 => Some(3 * a + c),
                5 => Some(3 * a + 2),
                _ => None,
            }
        }
    }

    /// The axis at ξ: (X', X'', t, N1, N2) with derivatives w.r.t. S = Jξ.
    fn frame(&self, xi: f64) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
        let (_, dn, ddn) = herm(xi);
        let j = self.j;
        let xs = [0, 1, 2].map(|c| (dn[0] * self.x[0][c] + dn[1] * self.x[1][c] + j * (dn[2] * self.t[0][c] + dn[3] * self.t[1][c])) / j);
        let xss = [0, 1, 2].map(|c| (ddn[0] * self.x[0][c] + ddn[1] * self.x[1][c] + j * (ddn[2] * self.t[0][c] + ddn[3] * self.t[1][c])) / (j * j));
        let t = unit(xs);
        let p = [(1.0 - xi) / 2.0, (1.0 + xi) / 2.0];
        let n1 = add3(scale(self.n1[0], p[0]), scale(self.n1[1], p[1]));
        let n1 = unit(sub(n1, scale(t, dot(n1, t))));
        (xs, xss, t, n1, cross(t, n1))
    }

    fn rows(&self, xi: f64) -> Rows {
        let ne = self.ne;
        let j = self.j;
        let (n, dn, ddn) = herm(xi);
        let (xs, xss, t, n1, n2) = self.frame(xi);
        let h = 1e-6;
        let n1d = scale(sub(self.frame(xi + h).3, self.frame(xi - h).3), 1.0 / (2.0 * h * j));
        let g = dot(xs, xs);
        let sg = g.sqrt();
        // u, u' and u'' (3 components) and ω and ω' as rows
        let mut u = [vec![0.0; ne], vec![0.0; ne], vec![0.0; ne]];
        let mut us = u.clone();
        let mut uss = u.clone();
        let mut om = u.clone();
        let mut om1 = u.clone();
        let p = [(1.0 - xi) / 2.0, (1.0 + xi) / 2.0];
        for a in 0..2 {
            for c in 0..3 {
                if let Some(k) = self.col(a, c) {
                    u[c][k] += n[a];
                    us[c][k] += dn[a] / j;
                    uss[c][k] += ddn[a] / (j * j);
                }
                if let Some(k) = self.col(a, 3 + c) {
                    om[c][k] += p[a];
                    om1[c][k] += if a == 0 { -0.5 / j } else { 0.5 / j };
                }
            }
            let tn = self.t[a];
            let lam = self.n + a;
            // du/dξ at the node = J(λ T + φ × T)
            for c in 0..3 {
                u[c][lam] += j * n[2 + a] * tn[c];
                us[c][lam] += dn[2 + a] * tn[c];
                uss[c][lam] += ddn[2 + a] * tn[c] / j;
            }
            for k in 0..3 {
                let Some(col) = self.col(a, 3 + k) else { continue };
                let mut e = [0.0; 3];
                e[k] = 1.0;
                let v = cross(e, tn);
                for c in 0..3 {
                    u[c][col] += j * n[2 + a] * v[c];
                    us[c][col] += dn[2 + a] * v[c];
                    uss[c][col] += ddn[2 + a] * v[c] / j;
                }
            }
        }
        let dotv = |v: [f64; 3], r: &[Vec<f64>; 3]| -> Vec<f64> { (0..ne).map(|k| v[0] * r[0][k] + v[1] * r[1][k] + v[2] * r[2][k]).collect() };
        let de: Vec<f64> = dotv(xs, &us).iter().map(|x| x / g).collect();
        let tw = dotv(t, &om);
        let n1us = dotv(n1, &us);
        let n2us = dotv(n2, &us);
        let (n1xss, n2xss, txss) = (dot(n1, xss), dot(n2, xss), dot(t, xss));
        let n1uss = dotv(n1, &uss);
        let n2uss = dotv(n2, &uss);
        let mut k1 = vec![0.0; ne];
        let mut k2 = vec![0.0; ne];
        let mut e1 = vec![0.0; ne];
        let t_om1 = dotv(t, &om1);
        let w_om = dotv(cross(n1d, n2), &om);
        let tn1d = dot(t, n1d);
        for k in 0..ne {
            // δn₁·X'' = (δω·t) N₂·X'' − g^{-1/2}(N₁·u') t·X''
            let dn1 = tw[k] * n2xss - n1us[k] / sg * txss;
            let dn2 = -tw[k] * n1xss - n2us[k] / sg * txss;
            k2[k] = (dn1 + n1uss[k] - de[k] * n1xss) / g;
            k1[k] = -(dn2 + n2uss[k] - de[k] * n2xss) / g;
            e1[k] = t_om1[k] + w_om[k] - n2us[k] / sg * tn1d;
        }
        // the torsion in the mass: the nodes' twist about their tangents, linearly
        let mut twn = vec![0.0; ne];
        for a in 0..2 {
            for c in 0..3 {
                if let Some(k) = self.col(a, 3 + c) {
                    twn[k] += p[a] * self.t[a][c];
                }
            }
        }
        let _ = tw;
        Rows { e: [de, k1, k2, e1], u, tw: twn, dl: j }
    }

    /// Stiffness and mass with λ₁, λ₂ last (ne × ne, global coordinates).
    pub fn full(&self) -> (Vec<f64>, Vec<f64>) {
        let ne = self.ne;
        let mut k = vec![0.0; ne * ne];
        let mut m = vec![0.0; ne * ne];
        let d = [self.ea, self.ei11, self.ei22, self.gj];
        for &(xi, w) in &GAUSS3 {
            let r = self.rows(xi);
            let wl = w * r.dl;
            for (s, row) in d.iter().zip(&r.e) {
                if *s != 0.0 {
                    outer_add(&mut k, ne, wl * s, row, row);
                }
            }
            for c in 0..3 {
                outer_add(&mut m, ne, wl * self.rho_a, &r.u[c], &r.u[c]);
            }
            if self.n == 12 {
                outer_add(&mut m, ne, wl * self.rho_ip, &r.tw, &r.tw);
            }
        }
        (k, m)
    }

    /// Static condensation of λ: (K_uu − K_uλ K_λλ⁻¹ K_λu, K_λλ⁻¹ K_λu).
    pub fn condensed(&self) -> Vec<f64> {
        let (k, _) = self.full();
        condense(&k, self.n, self.ne).0
    }

    /// Distributed load (force per length q in global coordinates), condensed.
    pub fn load(&self, q: [f64; 3]) -> Vec<f64> {
        let ne = self.ne;
        let mut f = vec![0.0; ne];
        for &(xi, w) in &GAUSS3 {
            let r = self.rows(xi);
            for c in 0..3 {
                for kk in 0..ne {
                    f[kk] += w * r.dl * q[c] * r.u[c][kk];
                }
            }
        }
        let (k, _) = self.full();
        let (_, kinv_kl, kinv) = condense(&k, self.n, ne);
        // f_c = f_u − K_uλ K_λλ⁻¹ f_λ
        let n = self.n;
        let fl = [f[n], f[n + 1]];
        let mut out = f[..n].to_vec();
        for i in 0..n {
            let kul = [k[i * ne + n], k[i * ne + n + 1]];
            let y = [kinv[0][0] * fl[0] + kinv[0][1] * fl[1], kinv[1][0] * fl[0] + kinv[1][1] * fl[1]];
            out[i] -= kul[0] * y[0] + kul[1] * y[1];
        }
        let _ = kinv_kl;
        out
    }

    /// Section forces at the Gauss points (N, M₁, M₂, T) with λ from the condensation.
    pub fn section_forces(&self, u: &[f64]) -> Vec<[f64; 4]> {
        let ne = self.ne;
        let (k, _) = self.full();
        let (_, kinv_kl, _) = condense(&k, self.n, ne);
        let mut full = u.to_vec();
        for a in 0..2 {
            full.push(-(0..self.n).map(|i| kinv_kl[a][i] * u[i]).sum::<f64>());
        }
        let d = [self.ea, self.ei11, self.ei22, self.gj];
        GAUSS3
            .iter()
            .map(|&(xi, _)| {
                let r = self.rows(xi);
                std::array::from_fn(|q| d[q] * r.e[q].iter().zip(&full).map(|(a, b)| a * b).sum::<f64>())
            })
            .collect()
    }
}

fn outer_add(k: &mut [f64], n: usize, c: f64, a: &[f64], b: &[f64]) {
    for i in 0..n {
        let ai = c * a[i];
        if ai != 0.0 {
            for (x, y) in k[i * n..(i + 1) * n].iter_mut().zip(b) {
                *x += ai * y;
            }
        }
    }
}

/// Condenses out the last two degrees of freedom: (K_c, K_λλ⁻¹K_λu, K_λλ⁻¹).
#[allow(clippy::type_complexity)]
fn condense(k: &[f64], n: usize, ne: usize) -> (Vec<f64>, [Vec<f64>; 2], [[f64; 2]; 2]) {
    let a = k[n * ne + n];
    let b = k[n * ne + n + 1];
    let d = k[(n + 1) * ne + n + 1];
    let det = a * d - b * b;
    let inv = [[d / det, -b / det], [-b / det, a / det]];
    let kl: [Vec<f64>; 2] = std::array::from_fn(|r| (0..n).map(|i| k[(n + r) * ne + i]).collect());
    let x: [Vec<f64>; 2] = std::array::from_fn(|r| (0..n).map(|i| inv[r][0] * kl[0][i] + inv[r][1] * kl[1][i]).collect());
    let mut kc = vec![0.0; n * n];
    for i in 0..n {
        for jj in 0..n {
            kc[i * n + jj] = k[i * ne + jj] - (k[i * ne + n] * x[0][jj] + k[i * ne + n + 1] * x[1][jj]);
        }
    }
    (kc, x, inv)
}

/// Section forces SF/SM at the three Gauss points (like `beams2_eb_output.rs`).
pub fn output(c: &Curved, dim: usize, u: &[f64]) -> Vec<super::api::IpField> {
    use super::api::IpField;
    let f = c.section_forces(u);
    let mut sf = if dim == 2 { IpField::new("SF", &["SF11", "SF22", "SF12"]) } else { IpField::new("SF", &["SF1", "SF2", "SF3"]) };
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    for (q, v) in f.iter().enumerate() {
        sf.points.push((q + 1, 0, vec![v[0], 0.0, 0.0]));
        if dim == 2 {
            sm.points.push((q + 1, 0, vec![v[1], 0.0, 0.0]));
        } else {
            sm.points.push((q + 1, 0, vec![v[1], v[2], v[3]]));
        }
    }
    vec![sf, sm]
}

/// Internal degrees of freedom for eigenvalue steps: the difference between the full
/// curved matrix (with λ₁, λ₂) and the element's nodal matrices (`stiffness`,
/// `mass` with consistent mass), which `beam_mass::internal_dofs` adds.
pub fn internal(c: &Curved, el: &EulerBeam, ctx: &ElemCtx) -> Result<super::beam_mass::BeamInternal> {
    use super::api::MassKind;
    let (mut k, mut m) = c.full();
    let n = c.n;
    let ne = c.ne;
    let kc = c.condensed();
    let mn = el.mass(ctx, MassKind::Consistent)?;
    for i in 0..n {
        for j in 0..n {
            k[i * ne + j] -= kc[i * n + j];
            m[i * ne + j] -= mn[i * n + j];
        }
    }
    Ok(super::beam_mass::BeamInternal { n: ne, k, m, kg: None })
}
pub static TYPES: &[&dyn super::api::ElementType] = &[];
