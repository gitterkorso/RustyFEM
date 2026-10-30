//! NLGEOM för S8R, SC8R och SC6R: korotationell formulering (Rankin–
//! Nour-Omid 1988, Felippa–Haugen 2005), som för S4R i `shell_nlgeom.rs`.
//!
//! Elementets stelkroppsrörelse avskiljs med ett medföljande system E(x):
//! normalen från diagonalerna mellan hörnen (S8R: hörnnoderna, SC8R:
//! mittytans hörn ½(x_undersida + x_översida)), 1-axeln längs ξ-riktningen
//! (trianglar: normalen från två kanter och 1-axeln längs första kanten), och
//! referenspunkten x_c = hörnens medelpunkt. Med R_r = E E₀ᵀ är de
//! deformationella förskjutningarna
//!
//! - u_dₐ = R_rᵀ (xₐ − x_c) − (Xₐ − X_c),
//! - θ_dₐ = log(R_rᵀ Rₐ) (S8R: nodens rotation relativt elementets).
//!
//! Töjningsenergin är Φ(d) = ½ dᵀ K₀ d + Φₙ(d) med elementets linjära
//! styvhet K₀ i referensläget och membrandelens kvadratiska (Green–Lagrange)
//! termer i det medföljande systemet: εαβ = εαβ,lin + ½ u,α·u,β med
//! u = de deformationella förskjutningarna, Φₙ = Σ ½ t w (εᵀDε − εₗᵢₙᵀDεₗᵢₙ)
//! i membranets integrationspunkter (S8R 2×2, SC8R/SC6R mitten). För
//! SC8R/SC6R räknas även tjocklekstöjningen som Green-töjning,
//! ε33 = (|d + Δ|² − |d|²)/(2|d|²); annars ger en rotation φ av direktören
//! relativt elementet en falsk tjocklekstöjning −φ²/2. Utan Φₙ ger
//! ett krökt element (S8R kan böjas inom elementet) en falsk membrantöjning av
//! storleksordningen (Δθ)²/24 från kordans förkortning. Med s = ∂Φ/∂d och
//! K_d = ∂²Φ/∂d² blir f = Jᵀ s och K = Jᵀ K_d J + Σᵢ sᵢ ∂²dᵢ (J = ∂d/∂q,
//! andraderivatorna exakt med hyperduala tal; rotationerna varieras med
//! rumsliga spinn som i NLGEOM-steget). Formuleringen tillåter godtyckligt
//! stora rotationer men små töjningar, vilket är antagandet för S8R.
//! För SC8R/SC6R (finita membrantöjningar) ger det avvikelser av
//! storleksordningen töjningen.
//!
//! För S8R beror u_dₐ och θ_dₐ bara på hörnens förskjutningar och nod a:s egna
//! frihetsgrader (högst 18 variabler), så derivatorna räknas nod för nod med
//! `Hd<18>` och läggs in i elementets 48 frihetsgrader.

use crate::elements::beam_timoshenko::nlgeom::{log_so3, mtm, mtv, spun};
use crate::elements::{ElemCtx, ElemResponse};
use crate::kinematics::hyperdual::{self as hd, Hd, T3, V3};
use crate::kinematics::tensor as t;
use crate::model::{fail, FemError, Result};

/// Medföljande system E (kolumner e₁, e₂, e₃) för hörnen `c` (3 eller 4 punkter).
fn frame<const N: usize>(c: &[V3<N>]) -> T3<N> {
    let sub = |a: &V3<N>, b: &V3<N>| -> V3<N> { std::array::from_fn(|i| a[i] - b[i]) };
    let unit = |v: V3<N>| -> V3<N> {
        let l = hd::dot3(&v, &v).sqrt().recip();
        v.map(|q| q * l)
    };
    let (n, g) = if c.len() == 4 {
        let n = hd::cross3(&sub(&c[2], &c[0]), &sub(&c[3], &c[1]));
        let g: V3<N> = std::array::from_fn(|i| c[1][i] + c[2][i] - c[0][i] - c[3][i]);
        (n, g)
    } else {
        let a = sub(&c[1], &c[0]);
        (hd::cross3(&a, &sub(&c[2], &c[0])), a)
    };
    let e3 = unit(n);
    let p = hd::dot3(&g, &e3);
    let e1 = unit(std::array::from_fn(|i| g[i] - e3[i] * p));
    let e2 = hd::cross3(&e3, &e1);
    std::array::from_fn(|i| [e1[i], e2[i], e3[i]])
}

/// R_r = E E₀ᵀ.
fn rot_r<const N: usize>(e: &T3<N>, e0: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| e[i][0] * e0[j][0] + e[i][1] * e0[j][1] + e[i][2] * e0[j][2]))
}

fn mean<const N: usize>(v: &[V3<N>]) -> V3<N> {
    let k = 1.0 / v.len() as f64;
    std::array::from_fn(|i| v.iter().fold(Hd::cst(0.0), |acc, p| acc + p[i]) * k)
}

fn too_large(ctx: &ElemCtx, name: &str) -> FemError {
    FemError(format!("element {} ({name}): för stor rotation av en nod relativt elementet (≥ 180°)", ctx.label))
}

// ---------------------------------------------------------------------------
// S8R: nodvis med Hd<18>
// ---------------------------------------------------------------------------

/// Lokala variabler för nod a: hörnens förskjutningar 0–11, nodens egen
/// förskjutning 12–14 (mittnoder) och rotation 15–17. Returnerar globalt index
/// (frihetsgrad i elementet) per lokal variabel (usize::MAX = oanvänd).
fn s8r_vars(a: usize) -> [usize; 18] {
    let mut m = [usize::MAX; 18];
    for i in 0..4 {
        for c in 0..3 {
            m[3 * i + c] = 6 * i + c;
        }
    }
    if a >= 4 {
        for c in 0..3 {
            m[12 + c] = 6 * a + c;
        }
    }
    for c in 0..3 {
        m[15 + c] = 6 * a + 3 + c;
    }
    m
}

/// Deformationella förskjutningar och rotation för nod a (6 hyperduala tal).
fn s8r_node(x0: &[[f64; 3]], u: &[f64], a: usize, with_vars: bool) -> Option<[Hd<18>; 6]> {
    let rel = |b: usize, i: usize| x0[b][i] - x0[0][i];
    let var = |idx: usize| -> Hd<18> { if with_vars { Hd::var(0.0, idx) } else { Hd::cst(0.0) } };
    let pos = |b: usize| -> V3<18> {
        std::array::from_fn(|i| {
            let base = rel(b, i) + u[6 * b + i] - u[i];
            let v = if b < 4 { var(3 * b + i) } else if b == a { var(12 + i) } else { Hd::cst(0.0) };
            v + base
        })
    };
    let refp = |b: usize| -> V3<18> { std::array::from_fn(|i| Hd::cst(rel(b, i))) };
    let c: Vec<V3<18>> = (0..4).map(pos).collect();
    let c0: Vec<V3<18>> = (0..4).map(refp).collect();
    let e = frame(&c);
    let e0 = frame(&c0);
    let rr = rot_r(&e, &e0);
    let (xc, xc0) = (mean(&c), mean(&c0));
    let xa = pos(a);
    let dx: V3<18> = std::array::from_fn(|i| xa[i] - xc[i]);
    let ud = mtv(&rr, &dx);
    let r0 = t::exp_so3([u[6 * a + 3], u[6 * a + 4], u[6 * a + 5]]);
    let ra = spun(&[var(15), var(16), var(17)], &r0);
    let th = log_so3(&mtm(&rr, &ra))?;
    let xr = refp(a);
    Some(std::array::from_fn(|k| if k < 3 { ud[k] - (xr[k].v - xc0[k].v) } else { th[k - 3] }))
}

/// Deformationella förskjutningar (48) för utdata.
pub fn deformational_s8r(ctx: &ElemCtx, name: &str, u: &[f64]) -> Result<Vec<f64>> {
    let mut d = Vec::with_capacity(48);
    for a in 0..8 {
        let v = s8r_node(&ctx.x, u, a, false).ok_or_else(|| too_large(ctx, name))?;
        d.extend(v.iter().map(|q| q.v));
    }
    Ok(d)
}

/// Inre kraft och tangent för S8R vid NLGEOM med linjär styvhet `k0`.
pub fn response_s8r(ctx: &ElemCtx, name: &str, k0: &[f64], ops: &[QuadGroup], u: &[f64], tangent: bool) -> Result<ElemResponse> {
    const N: usize = 48;
    let mut dv = vec![0.0; N];
    let mut jac = vec![0.0; N * N];
    let mut parts: Vec<([usize; 18], [Hd<18>; 6])> = Vec::with_capacity(8);
    for a in 0..8 {
        let m = s8r_vars(a);
        let v = s8r_node(&ctx.x, u, a, true).ok_or_else(|| too_large(ctx, name))?;
        for k in 0..6 {
            let row = 6 * a + k;
            dv[row] = v[k].v;
            for (l, &g) in m.iter().enumerate() {
                if g != usize::MAX {
                    jac[row * N + g] += v[k].g[l];
                }
            }
        }
        parts.push((m, v));
    }
    assemble(ctx, name, k0, ops, &dv, &jac, tangent, |k, s| {
        for (a, (m, v)) in parts.iter().enumerate() {
            for kk in 0..6 {
                let si = s[6 * a + kk];
                if si == 0.0 {
                    continue;
                }
                for (l1, &g1) in m.iter().enumerate() {
                    if g1 == usize::MAX {
                        continue;
                    }
                    for (l2, &g2) in m.iter().enumerate() {
                        if g2 != usize::MAX {
                            k[g1 * N + g2] += si * v[kk].h[l1][l2];
                        }
                    }
                }
            }
        }
    })
}

/// Töjningsgrupp med kvadratiska termer: e_k = l_k·d + ½ dᵀQ_k d (k = 1..m)
/// och energin ½ w eᵀCe; den linjära delen ½ w (Ld)ᵀC(Ld) finns redan i K₀.
pub struct QuadGroup {
    /// L (m × n, radvis).
    pub l: Vec<f64>,
    /// Q_k (n × n, radvis; tom = 0).
    pub q: Vec<Vec<f64>>,
    /// C (m × m, radvis).
    pub c: Vec<f64>,
    pub w: f64,
}

impl QuadGroup {
    /// Membranet i en integrationspunkt: e = [ε11, ε22, γ12] med de kvadratiska
    /// termerna [½u,1·u,1, ½u,2·u,2, u,1·u,2], där u,α = Aα d (3 × n, globala
    /// komponenter), och C = D·t.
    pub fn membrane(bm: Vec<f64>, a: [Vec<f64>; 2], c: [f64; 9], w: f64) -> QuadGroup {
        let n = bm.len() / 3;
        let ata = |x: &[f64], y: &[f64]| -> Vec<f64> {
            let mut q = vec![0.0; n * n];
            for r in 0..3 {
                for p in 0..n {
                    let xp = x[r * n + p];
                    if xp != 0.0 {
                        for k in 0..n {
                            q[p * n + k] += xp * y[r * n + k];
                        }
                    }
                }
            }
            q
        };
        let q0 = ata(&a[0], &a[0]);
        let q1 = ata(&a[1], &a[1]);
        let q01 = ata(&a[0], &a[1]);
        let q2: Vec<f64> = (0..n * n).map(|i| q01[i] + q01[(i % n) * n + i / n]).collect();
        QuadGroup { l: bm, q: vec![q0, q1, q2], c: c.to_vec(), w }
    }

    fn m(&self) -> usize {
        self.q.len()
    }
}

/// Kvadratiska töjningar ½ dᵀQ_k d per grupp (för utdata).
pub fn quad_strains(groups: &[QuadGroup], d: &[f64]) -> Vec<Vec<f64>> {
    let n = d.len();
    groups
        .iter()
        .map(|g| {
            g.q.iter()
                .map(|q| if q.is_empty() { 0.0 } else { 0.5 * (0..n).map(|p| d[p] * (0..n).map(|k| q[p * n + k] * d[k]).sum::<f64>()).sum::<f64>() })
                .collect()
        })
        .collect()
}

/// Geometrisk styvhet Σ w Σ_k σ_k Q_k med σ = C·(L u) (linjära töjningar ur `u`):
/// för membrangrupper ∫ N_αβ u,α·u,β dA, alla tre förskjutningskomponenterna.
pub fn geometric(groups: &[QuadGroup], u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut k = vec![0.0; n * n];
    for g in groups {
        let m = g.m();
        let el: Vec<f64> = (0..m).map(|r| (0..n).map(|q| g.l[r * n + q] * u[q]).sum()).collect();
        for (r, q) in g.q.iter().enumerate() {
            let sig: f64 = (0..m).map(|j| g.c[r * m + j] * el[j]).sum();
            if !q.is_empty() && sig != 0.0 {
                for i in 0..n * n {
                    k[i] += g.w * sig * q[i];
                }
            }
        }
    }
    k
}

/// s = ∂Φ/∂d och K_d = ∂²Φ/∂d² för Φ = ½ dᵀK₀d + Σ grupper (½ w eᵀCe − ½ w (Ld)ᵀC(Ld)).
fn energy_derivs(k0: &[f64], groups: &[QuadGroup], d: &[f64], tangent: bool) -> (Vec<f64>, Option<Vec<f64>>) {
    let n = d.len();
    let mut s: Vec<f64> = (0..n).map(|i| (0..n).map(|j| k0[i * n + j] * d[j]).sum()).collect();
    let mut kd = tangent.then(|| k0.to_vec());
    for g in groups {
        let m = g.m();
        // Q_k d och de kvadratiska töjningarna
        let qd: Vec<Vec<f64>> = g
            .q
            .iter()
            .map(|q| if q.is_empty() { vec![0.0; n] } else { (0..n).map(|p| (0..n).map(|k| q[p * n + k] * d[k]).sum()).collect() })
            .collect();
        let eq: Vec<f64> = qd.iter().map(|v| 0.5 * (0..n).map(|p| v[p] * d[p]).sum::<f64>()).collect();
        let el: Vec<f64> = (0..m).map(|r| (0..n).map(|q| g.l[r * n + q] * d[q]).sum()).collect();
        let e: Vec<f64> = (0..m).map(|r| el[r] + eq[r]).collect();
        let sig: Vec<f64> = (0..m).map(|i| (0..m).map(|j| g.c[i * m + j] * e[j]).sum()).collect();
        let seq: Vec<f64> = (0..m).map(|i| (0..m).map(|j| g.c[i * m + j] * eq[j]).sum()).collect();
        // s += w (Lᵀ C eq + Σ σ_k Q_k d)
        for q in 0..n {
            let v: f64 = (0..m).map(|r| g.l[r * n + q] * seq[r] + qd[r][q] * sig[r]).sum();
            s[q] += g.w * v;
        }
        if let Some(k) = kd.as_mut() {
            // K += w [(L + G)ᵀC(L + G) − LᵀCL + Σ σ_k Q_k], G = raderna Q_k d
            let full: Vec<f64> = (0..m * n).map(|i| g.l[i] + qd[i / n][i % n]).collect();
            super::add_btcb(k, n, &full, &g.c, &full, m, g.w);
            super::add_btcb(k, n, &g.l, &g.c, &g.l, m, -g.w);
            for (r, q) in g.q.iter().enumerate() {
                if !q.is_empty() && sig[r] != 0.0 {
                    let f = g.w * sig[r];
                    for i in 0..n * n {
                        k[i] += f * q[i];
                    }
                }
            }
        }
    }
    (s, kd)
}

/// f = Jᵀ s och K = Jᵀ K_d J + Σ sᵢ Hᵢ (`geo` lägger till den sista termen).
fn assemble(ctx: &ElemCtx, name: &str, k0: &[f64], ops: &[QuadGroup], d: &[f64], jac: &[f64], tangent: bool, geo: impl Fn(&mut [f64], &[f64])) -> Result<ElemResponse> {
    let n = d.len();
    let (s, kd) = energy_derivs(k0, ops, d, tangent);
    let fint: Vec<f64> = (0..n).map(|q| (0..n).map(|i| jac[i * n + q] * s[i]).sum()).collect();
    if fint.iter().any(|v| !v.is_finite()) {
        return fail(format!("element {} ({name}): ogiltig deformerad geometri", ctx.label));
    }
    let k = kd.map(|kd| {
        let mut kj = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let kij = kd[i * n + j];
                if kij != 0.0 {
                    for q in 0..n {
                        kj[i * n + q] += kij * jac[j * n + q];
                    }
                }
            }
        }
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for p in 0..n {
                let jip = jac[i * n + p];
                if jip != 0.0 {
                    for q in 0..n {
                        k[p * n + q] += jip * kj[i * n + q];
                    }
                }
            }
        }
        geo(&mut k, &s);
        for p in 0..n {
            for q in 0..p {
                let v = 0.5 * (k[p * n + q] + k[q * n + p]);
                k[p * n + q] = v;
                k[q * n + p] = v;
            }
        }
        k
    });
    Ok(ElemResponse { fint, k })
}

// ---------------------------------------------------------------------------
// SC8R och SC6R: alla frihetsgrader på en gång
// ---------------------------------------------------------------------------

/// Deformationella förskjutningar för ett kontinuumskal med `nc` hörn per yta
/// (noderna 0..nc undersidan, nc..2nc översidan).
fn sc_def<const N: usize>(x0: &[[f64; 3]], u: &[f64], with_vars: bool) -> Vec<Hd<N>> {
    let nn = x0.len();
    let nc = nn / 2;
    let var = |q: usize| -> Hd<N> { if with_vars { Hd::var(0.0, q) } else { Hd::cst(0.0) } };
    let xs: Vec<V3<N>> = (0..nn).map(|a| std::array::from_fn(|i| var(3 * a + i) + (x0[a][i] - x0[0][i] + u[3 * a + i] - u[i]))).collect();
    let xr: Vec<V3<N>> = (0..nn).map(|a| std::array::from_fn(|i| Hd::cst(x0[a][i] - x0[0][i]))).collect();
    let mid = |v: &[V3<N>]| -> Vec<V3<N>> { (0..nc).map(|a| std::array::from_fn(|i| (v[a][i] + v[a + nc][i]) * 0.5)).collect() };
    let (c, c0) = (mid(&xs), mid(&xr));
    let e = frame(&c);
    let e0 = frame(&c0);
    let rr = rot_r(&e, &e0);
    let (xc, xc0) = (mean(&c), mean(&c0));
    let mut d = Vec::with_capacity(3 * nn);
    for a in 0..nn {
        let dx: V3<N> = std::array::from_fn(|i| xs[a][i] - xc[i]);
        let ud = mtv(&rr, &dx);
        for i in 0..3 {
            d.push(ud[i] - (xr[a][i].v - xc0[i].v));
        }
    }
    d
}

fn sc_response_n<const N: usize>(ctx: &ElemCtx, name: &str, k0: &[f64], ops: &[QuadGroup], u: &[f64], tangent: bool) -> Result<ElemResponse> {
    let d = sc_def::<N>(&ctx.x, u, true);
    let dv: Vec<f64> = d.iter().map(|q| q.v).collect();
    let mut jac = vec![0.0; N * N];
    for i in 0..N {
        jac[i * N..(i + 1) * N].copy_from_slice(&d[i].g);
    }
    assemble(ctx, name, k0, ops, &dv, &jac, tangent, |k, s| {
        for i in 0..N {
            if s[i] != 0.0 {
                for p in 0..N {
                    for q in 0..N {
                        k[p * N + q] += s[i] * d[i].h[p][q];
                    }
                }
            }
        }
    })
}

/// Inre kraft och tangent för SC8R/SC6R vid NLGEOM.
pub fn response_sc(ctx: &ElemCtx, name: &str, k0: &[f64], ops: &[QuadGroup], u: &[f64], tangent: bool) -> Result<ElemResponse> {
    match ctx.x.len() {
        8 => sc_response_n::<24>(ctx, name, k0, ops, u, tangent),
        6 => sc_response_n::<18>(ctx, name, k0, ops, u, tangent),
        n => fail(format!("element {} ({name}): NLGEOM stöds inte för {n} noder", ctx.label)),
    }
}

/// Deformationella förskjutningar för utdata (SC8R/SC6R).
pub fn deformational_sc(ctx: &ElemCtx, u: &[f64]) -> Vec<f64> {
    sc_def::<1>(&ctx.x, u, false).iter().map(|q| q.v).collect()
}
