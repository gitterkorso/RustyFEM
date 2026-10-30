//! Timoshenkobalkar (B21, B22, B31, B32) vid NLGEOM=YES. Formuleringen
//! följer geometriskt exakt balkteori (Reissner 1972, Simo 1985,
//! Simo–Vu-Quoc 1986) i uppdaterad, inkrementell form.
//!
//! Kinematik i en integrationspunkt (Λ = [t_s, n₁, n₂] tvärsnittets axlar,
//! Λ₀ i referensläget, x′ = dx/dS mot referensens båglängd):
//!
//! - sträckning λ = |x′|, axelns riktning a = x′/λ;
//! - axiell töjning ε = ln λ + ½ τ² (I11 + I22)/A (logaritmisk töjning plus
//!   fibrernas medelförlängning vid vridning, "helixeffekten");
//! - skjuvning γ₁ = n₁·a, γ₂ = n₂·a (sinus av vinkeln mellan tvärsnitt och axel);
//! - krökning och vridning [τ, κ₁, κ₂] = K̃/λ, där K̃ är tvärsnittets relativa
//!   rotation per referenslängd i dess egna axlar.
//!
//! Rotationerna är inkrementella: med nodernas rotationsinkrement
//! Δφₐ = log(Rₐ Rₐ,ₙᵀ) (rumsliga) och Δφ(ξ) = Σ Nₐ Δφₐ är
//! Λ = exp([Δφ]) Λₙ och K̃ = K̃ₙ + Λₙᵀ T(Δφ)ᵀ Δφ′. Inom ett inkrement motsvarar
//! det linjär interpolation av (inkrementella) rotationsvektorer (formuleringen
//! är inkrementell, inte total). Λ och K̃ per integrationspunkt är elementets
//! tillstånd.
//!
//! Snittkrafterna är s = C e med det linjära elementets tvärsnittsmatris C,
//! där skjuvstyvhetens slankhetskompensation räknas med elementets aktuella
//! längd. Inre krafter fås med virtuella töjningar i den aktuella
//! konfigurationen, integrerade över aktuell längd (w·J·λ):
//!
//! - δε = a·δx′ₛ (+ τ (I11 + I22)/A · t_s·δθ′ₛ/λ, vridningstermen per referenslängd),
//! - δγ₁ = n₁·δx′ₛ − n₂·δθ, δγ₂ = n₂·δx′ₛ + n₁·δθ,
//! - δ[τ, κ₁, κ₂] = Λᵀ δθ′ₛ,
//!
//! där ′ₛ är derivata mot aktuell båglängd och δθ rumsliga virtuella rotationer.
//! Linjäriserat ger det exakt den linjära balkens töjningar i
//! `beam_timoshenko.rs`. Kraften är inte gradienten av en potential, så
//! tangenten ∂f/∂q (exakt, med hyperduala tal mot nodförskjutningar och spinn,
//! R ← exp([δw]) R) är inte symmetrisk; den symmetriseras för de symmetriska
//! lösarna. Det ger linjär i stället för kvadratisk konvergens när momenten är
//! stora.
//!
//! Den geometriska styvheten för *BUCKLE och *FREQUENCY med förlast är den del
//! av tangenten i referensläget som är proportionell mot snittkrafterna (alla
//! snittkrafter, inte bara normalkraften).

use super::{gauss, shape, BeamGeom, TimoBeam};
use crate::elements::{ElemCtx, ElemResponse};
use crate::kinematics::hyperdual::{self as hd, Hd, T3, V3};
use crate::kinematics::tensor as t;
use crate::model::{fail, Result};

// ---------------------------------------------------------------------------
// Rotationshjälpare för hyperduala tal (används också av skalens NLGEOM)
// ---------------------------------------------------------------------------

/// Konstant hyperdual matris.
pub fn cst_m<const N: usize>(m: &t::M3) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| Hd::cst(m[i][j])))
}

/// A B.
pub fn mm<const N: usize>(a: &T3<N>, b: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
}

/// Aᵀ B.
pub fn mtm<const N: usize>(a: &T3<N>, b: &T3<N>) -> T3<N> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[0][i] * b[0][j] + a[1][i] * b[1][j] + a[2][i] * b[2][j]))
}

/// Aᵀ v.
pub fn mtv<const N: usize>(a: &T3<N>, v: &V3<N>) -> V3<N> {
    std::array::from_fn(|i| a[0][i] * v[0] + a[1][i] * v[1] + a[2][i] * v[2])
}

/// f(x) med givna f, f′, f″ i x.
pub fn chain<const N: usize>(x: &Hd<N>, f: f64, d1: f64, d2: f64) -> Hd<N> {
    let mut r = Hd::cst(f);
    for i in 0..N {
        r.g[i] = d1 * x.g[i];
        for j in 0..N {
            r.h[i][j] = d1 * x.h[i][j] + d2 * x.g[i] * x.g[j];
        }
    }
    r
}

/// h(ν) = atan(√ν)/√ν med derivator mot ν (serie för små ν).
fn atan_coef(nu: f64) -> (f64, f64, f64) {
    if nu < 1e-2 {
        // h = Σ (−1)^k ν^k / (2k + 1)
        let (mut f, mut d1, mut d2) = (0.0, 0.0, 0.0);
        let mut p = 1.0;
        for k in 0..12usize {
            let c = if k % 2 == 0 { 1.0 } else { -1.0 } / (2 * k + 1) as f64;
            f += c * p;
            if k >= 1 {
                d1 += c * k as f64 * if k >= 2 { nu.powi(k as i32 - 1) } else { 1.0 };
            }
            if k >= 2 {
                d2 += c * (k * (k - 1)) as f64 * if k >= 3 { nu.powi(k as i32 - 2) } else { 1.0 };
            }
            p *= nu;
        }
        return (f, d1, d2);
    }
    let r = nu.sqrt();
    let at = r.atan();
    let h = at / r;
    let q = 1.0 + r * r;
    let hr = 1.0 / (r * q) - at / (r * r);
    let hrr = -(1.0 + 3.0 * r * r) / (r * r * q * q) - (1.0 / (r * r * q) - 2.0 * at / (r * r * r));
    (h, hr / (2.0 * r), hrr / (4.0 * nu) - hr / (4.0 * r * r * r))
}

/// Rotationsvektorn för en rotationsmatris (vinkel < π), via kvaternionen:
/// s = ½ axial(R − Rᵀ) = sin θ·a, w² = cos²(θ/2) = (1 + tr R)/4 och
/// φ = h(ν)·s/w² med ν = tan²(θ/2) = |s|²/(4w⁴), h(ν) = atan(√ν)/√ν.
pub fn log_so3<const N: usize>(r: &T3<N>) -> Option<V3<N>> {
    let w2 = (r[0][0] + r[1][1] + r[2][2] + 1.0) * 0.25;
    if !(w2.v > 1e-8) {
        return None;
    }
    let s: V3<N> = [(r[2][1] - r[1][2]) * 0.5, (r[0][2] - r[2][0]) * 0.5, (r[1][0] - r[0][1]) * 0.5];
    let iw2 = w2.recip();
    let nu = hd::dot3(&s, &s) * iw2 * iw2 * 0.25;
    let (f, d1, d2) = atan_coef(nu.v);
    let h = chain(&nu, f, d1, d2) * iw2;
    Some(s.map(|v| v * h))
}

/// Spinnvariabel: rotationsmatrisen exp([w]) R₀.
pub fn spun<const N: usize>(w: &V3<N>, r0: &t::M3) -> T3<N> {
    mm(&hd::exp_so3(w), &cst_m(r0))
}

// ---------------------------------------------------------------------------
// Balken
// ---------------------------------------------------------------------------

/// Tillstånd per integrationspunkt: tvärsnittets axlar Λ (9, radvis) och
/// den ackumulerade relativa rotationen per referenslängd K̃ (3, materiell).
const IP_STATE: usize = 12;

/// Aktuellt läge och läget vid inkrementets början.
struct Current {
    /// Nodlägen relativt nod 1 (aktuella).
    dx: Vec<[f64; 3]>,
    /// Nodernas rotationer nu och vid inkrementets början.
    rot: Vec<t::M3>,
    rot_n: Vec<t::M3>,
    /// Λₙ och K̃ₙ per integrationspunkt vid inkrementets början.
    lam_n: Vec<t::M3>,
    k_n: Vec<[f64; 3]>,
}

/// Töjningar och virtuella töjningar i en integrationspunkt.
struct Point<const N: usize> {
    /// [ε, κ₁, κ₂, τ, γ₁, γ₂]
    e: [Hd<N>; 6],
    /// Virtuella töjningar per nodvariabel: de[r][6a + c] (c: 0–2 δx, 3–5 δθ).
    de: [Vec<Hd<N>>; 6],
    /// Vikt w·J·λ (integration över aktuell längd).
    wl: Hd<N>,
    /// Nytt tillstånd (Λ, K̃).
    lam: t::M3,
    k: [f64; 3],
}

/// ln för hyperduala tal.
fn ln_hd<const N: usize>(x: Hd<N>) -> Hd<N> {
    let v = x.v;
    chain(&x, v.ln(), 1.0 / v, -1.0 / (v * v))
}

/// Töjningar i integrationspunkterna. `var(a, c)` ger störningen för nod a,
/// komponent c (0–2 förskjutning, 3–5 spinn).
fn points<const N: usize>(b: &TimoBeam, g: &BeamGeom, cur: &Current, var: &dyn Fn(usize, usize) -> Hd<N>) -> Option<Vec<Point<N>>> {
    let nn = b.nn;
    let xs: Vec<V3<N>> = (0..nn).map(|a| std::array::from_fn(|i| var(a, i) + cur.dx[a][i])).collect();
    // nodernas rotationsinkrement (rumsliga): Δφₐ = log(exp([w]) Rₐ Rₐₙᵀ)
    let mut dphi: Vec<V3<N>> = Vec::with_capacity(nn);
    for a in 0..nn {
        let r = spun(&[var(a, 3), var(a, 4), var(a, 5)], &cur.rot[a]);
        dphi.push(log_so3(&mm(&r, &cst_m(&t::transpose(&cur.rot_n[a]))))?);
    }
    let ipa = if g.sec.a > 0.0 { (g.sec.i11 + g.sec.i22) / g.sec.a } else { 0.0 };
    let z = Hd::<N>::cst(0.0);
    let mut out = Vec::new();
    for (q, &(xi, w)) in gauss(b.n_gauss()).iter().enumerate() {
        let (n, dn) = shape(nn, xi);
        let j = g.jac(nn, xi);
        let mut xd: V3<N> = [z; 3];
        let mut psi: V3<N> = [z; 3];
        let mut dpsi: V3<N> = [z; 3];
        for a in 0..nn {
            for k in 0..3 {
                xd[k] = xd[k] + xs[a][k] * (dn[a] / j);
                psi[k] = psi[k] + dphi[a][k] * n[a];
                dpsi[k] = dpsi[k] + dphi[a][k] * (dn[a] / j);
            }
        }
        // sträckning λ = |x′| och axelns riktning a
        let lam = hd::dot3(&xd, &xd).sqrt();
        let il = lam.recip();
        let av: V3<N> = xd.map(|v| v * il);
        // tvärsnittets axlar Λ = exp([Δφ]) Λₙ (kolumner t_s, n₁, n₂)
        let lnh = cst_m::<N>(&cur.lam_n[q]);
        let lm = mm(&hd::exp_so3(&psi), &lnh);
        let col = |k: usize| -> V3<N> { [lm[0][k], lm[1][k], lm[2][k]] };
        let (ts, n1, n2) = (col(0), col(1), col(2));
        // relativ rotation per referenslängd: K̃ = K̃ₙ + Λₙᵀ T(Δφ)ᵀ Δφ′, krökning K̃/λ
        let kr = mtv(&hd::tangent_so3(&psi), &dpsi);
        let kt: V3<N> = std::array::from_fn(|i| mtv(&lnh, &kr)[i] + cur.k_n[q][i]);
        let kk = kt.map(|v| v * il);
        // axiell töjning: ln λ plus fibrernas medelförlängning vid vridning,
        // ½ τ² (I11 + I22)/A (helixeffekten)
        // skjuvtöjningen räknas från referensläget (för mittnoder som inte ligger
        // exakt på linjen mellan ändnoderna, t.ex. avrundade koordinater)
        // (samma räkning som för det aktuella läget, så att töjningarna är exakt noll i referensläget)
        let x0d: [f64; 3] = std::array::from_fn(|k| (0..nn).fold(0.0, |acc, a| acc + (g.x[a][k] - g.x[0][k]) * (dn[a] / j)));
        let l0d = (x0d[0] * x0d[0] + x0d[1] * x0d[1] + x0d[2] * x0d[2]).sqrt();
        let g0 = [1, 2].map(|c| (0..3).map(|k| g.axes[c][k] * (x0d[k] * (1.0 / l0d))).sum::<f64>());
        let e = [ln_hd(lam) - l0d.ln() + kk[0] * kk[0] * (0.5 * ipa), kk[1], kk[2], kk[0], hd::dot3(&n1, &av) - g0[0], hd::dot3(&n2, &av) - g0[1]];
        // virtuella töjningar (uppdaterad konfiguration):
        // δε = a·δx′ₛ, δγ₁ = n₁·δx′ₛ − n₂·δθ, δγ₂ = n₂·δx′ₛ + n₁·δθ, δκ = Λᵀ δθ′ₛ
        let nd = 6 * nn;
        let mut de: [Vec<Hd<N>>; 6] = std::array::from_fn(|_| vec![z; nd]);
        for a in 0..nn {
            let dxs = il * (dn[a] / j);
            for k in 0..3 {
                de[0][6 * a + k] = av[k] * dxs;
                de[4][6 * a + k] = n1[k] * dxs;
                de[5][6 * a + k] = n2[k] * dxs;
                de[4][6 * a + 3 + k] = -n2[k] * n[a];
                de[5][6 * a + 3 + k] = n1[k] * n[a];
                de[3][6 * a + 3 + k] = ts[k] * dxs;
                de[1][6 * a + 3 + k] = n1[k] * dxs;
                de[2][6 * a + 3 + k] = n2[k] * dxs;
                // vridningstermens virtuella arbete räknas per referenslängd (faktorn 1/λ)
                de[0][6 * a + 3 + k] = kk[0] * ts[k] * dxs * il * ipa;
            }
        }
        out.push(Point {
            e,
            de,
            wl: lam * (w * j),
            lam: std::array::from_fn(|i| std::array::from_fn(|k| lm[i][k].v)),
            k: kt.map(|v| v.v),
        });
    }
    Some(out)
}

/// T(φ) i f64 (dR = [T dφ]× R).
fn so3_tangent(p: [f64; 3]) -> t::M3 {
    let q: V3<1> = p.map(Hd::<1>::cst);
    let m = hd::tangent_so3(&q);
    std::array::from_fn(|i| std::array::from_fn(|j| m[i][j].v))
}

/// Läget nu (förskjutning `u`) och vid inkrementets början (`u − du`,
/// tillståndet `state`). Saknas tillstånd (första NLGEOM-inkrementet) räknas
/// Λₙ och K̃ₙ ur rotationsvektorerna vid inkrementets början som om de nåtts
/// i ett enda inkrement från referensläget.
fn current(b: &TimoBeam, g: &BeamGeom, u: &[f64], du: &[f64], state: &[f64]) -> Current {
    let nn = b.nn;
    let u6 = b.expand_vector(u);
    let du6 = if du.len() == u.len() { b.expand_vector(du) } else { vec![0.0; u6.len()] };
    let p0: [f64; 3] = std::array::from_fn(|i| g.x[0][i] + u6[i]);
    let phi = |a: usize| -> [f64; 3] { [u6[6 * a + 3], u6[6 * a + 4], u6[6 * a + 5]] };
    let phi_n: Vec<[f64; 3]> = (0..nn).map(|a| std::array::from_fn(|i| u6[6 * a + 3 + i] - du6[6 * a + 3 + i])).collect();
    let ng = b.n_gauss();
    let (lam_n, k_n) = if state.len() == IP_STATE * ng {
        (
            (0..ng).map(|q| std::array::from_fn(|i| std::array::from_fn(|k| state[IP_STATE * q + 3 * i + k]))).collect(),
            (0..ng).map(|q| std::array::from_fn(|i| state[IP_STATE * q + 9 + i])).collect(),
        )
    } else {
        let lam0: t::M3 = std::array::from_fn(|i| std::array::from_fn(|k| g.axes[k][i]));
        let mut ls = Vec::new();
        let mut ks = Vec::new();
        for &(xi, _) in gauss(ng) {
            let (n, dn) = shape(nn, xi);
            let j = g.jac(nn, xi);
            let pm: [f64; 3] = std::array::from_fn(|i| (0..nn).map(|a| n[a] * phi_n[a][i]).sum());
            let dp: [f64; 3] = std::array::from_fn(|i| (0..nn).map(|a| dn[a] / j * phi_n[a][i]).sum());
            ls.push(t::mul(&t::exp_so3(pm), &lam0));
            ks.push(t::mv(&t::transpose(&lam0), t::mv(&t::transpose(&so3_tangent(pm)), dp)));
        }
        (ls, ks)
    };
    Current {
        dx: (0..nn).map(|a| std::array::from_fn(|i| g.x[a][i] + u6[6 * a + i] - p0[i])).collect(),
        rot: (0..nn).map(|a| t::exp_so3(phi(a))).collect(),
        rot_n: phi_n.iter().map(|p| t::exp_so3(*p)).collect(),
        lam_n,
        k_n,
    }
}

/// Tvärsnittsmatrisen med skjuvstyvhetens slankhetskompensation räknad med
/// elementets aktuella längd (avstånd mellan ändnoderna).
/// Tvärsnittsmatrisen med aktuell längd för förskjutningen `u` (utdata).
pub fn current_c_for(b: &TimoBeam, g: &BeamGeom, u: &[f64]) -> [[f64; 6]; 6] {
    let cur = current(b, g, u, &[], &[]);
    current_c(b, g, &cur)
}

fn current_c(b: &TimoBeam, g: &BeamGeom, cur: &Current) -> [[f64; 6]; 6] {
    let d = cur.dx[b.nn - 1];
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let gc = BeamGeom { axes: g.axes, l: if l > 0.0 { l } else { g.l }, x: g.x.clone(), sec: g.sec.clone() };
    b.section_matrix(&gc)
}

fn too_large(ctx: &ElemCtx, b: &TimoBeam) -> crate::model::FemError {
    crate::model::FemError(format!("element {} ({}): för stort rotationsinkrement (≥ 180°)", ctx.label, b.name))
}

#[allow(clippy::too_many_arguments)]
fn response_n<const N: usize>(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    let g = b.geom(ctx)?;
    // område struct_nonlin: med sektionspunkter (olinjärt material) ligger
    // sektionspunkternas tillstånd först, sedan Λ och K̃ per integrationspunkt
    let fib = if super::plastic::is_plastic(ctx) { Some(super::plastic::Fibers::new(ctx, b.name)?) } else { None };
    let ng = b.n_gauss();
    let plen = fib.as_ref().map_or(0, |f| f.ip_len() * ng);
    let pinit;
    let (pso, so) = match &fib {
        Some(_) if so.len() >= plen => (&so[..plen], &so[plen..]),
        Some(fb) => {
            pinit = fb.init_state(ng);
            (&pinit[..], &so[..0])
        }
        None => (&so[..0], so),
    };
    let mut psn = pso.to_vec();
    let cur = current(b, &g, u, du, so);
    let c = current_c(b, &g, &cur);
    let planar = b.dim == 2;
    // variabelindex för nod a, komponent k (None = ingen frihetsgrad)
    let idx = |a: usize, k: usize| -> Option<usize> {
        if planar {
            match k {
                0 | 1 => Some(3 * a + k),
                5 => Some(3 * a + 2),
                _ => None,
            }
        } else {
            Some(6 * a + k)
        }
    };
    let var = |a: usize, k: usize| -> Hd<N> { idx(a, k).map_or(Hd::cst(0.0), |i| Hd::var(0.0, i)) };
    let pts = points::<N>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    // f = Σ w J λ δeᵀ C e
    let mut f = vec![Hd::<N>::cst(0.0); N];
    for (ip, p) in pts.iter().enumerate() {
        let s: [Hd<N>; 6] = match &fib {
            // sektionspunkter: s(e) med tangenten Cₜ, s = s₀ + Cₜ (e − e₀) för derivatorna
            Some(fb) => {
                let ev: [f64; 6] = std::array::from_fn(|r| p.e[r].v);
                let len = fb.ip_len();
                let (nsh, _) = shape(b.nn, gauss(b.n_gauss())[ip].0);
                let (s0, ct) = fb.response(ctx, &c, &ev, &nsh, &pso[ip * len..(ip + 1) * len], &mut psn[ip * len..(ip + 1) * len])?;
                std::array::from_fn(|r| (0..6).fold(Hd::cst(s0[r]), |acc, q| if ct[r][q] != 0.0 { acc + (p.e[q] - ev[q]) * ct[r][q] } else { acc }))
            }
            None => std::array::from_fn(|r| (0..6).fold(Hd::cst(0.0), |acc, q| if c[r][q] != 0.0 { acc + p.e[q] * c[r][q] } else { acc })),
        };
        for a in 0..b.nn {
            for k in 0..6 {
                let Some(i) = idx(a, k) else { continue };
                let mut v = Hd::cst(0.0);
                for r in 0..6 {
                    let d = p.de[r][6 * a + k];
                    if d.v != 0.0 || d.g.iter().any(|x| *x != 0.0) {
                        v = v + d * s[r];
                    }
                }
                f[i] = f[i] + v * p.wl;
            }
        }
    }
    let fint: Vec<f64> = f.iter().map(|x| x.v).collect();
    if fint.iter().any(|v| !v.is_finite()) {
        return fail(format!("element {} ({}): ogiltig deformerad geometri", ctx.label, b.name));
    }
    sn.clear();
    sn.extend_from_slice(&psn);
    for p in &pts {
        for row in &p.lam {
            sn.extend_from_slice(row);
        }
        sn.extend_from_slice(&p.k);
    }
    // tangent: ∂f/∂q (inte symmetrisk; symmetriseras för den symmetriska lösaren)
    let k = tangent.then(|| {
        let mut k = vec![0.0; N * N];
        for i in 0..N {
            for j in 0..N {
                k[i * N + j] = 0.5 * (f[i].g[j] + f[j].g[i]);
            }
        }
        k
    });
    Ok(ElemResponse { fint, k })
}

/// Inre kraft och tangent vid NLGEOM.
pub fn response(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
    match (b.dim, b.nn) {
        (2, 2) => response_n::<6>(b, ctx, u, du, so, sn, tangent),
        (2, _) => response_n::<9>(b, ctx, u, du, so, sn, tangent),
        (_, 2) => response_n::<12>(b, ctx, u, du, so, sn, tangent),
        _ => response_n::<18>(b, ctx, u, du, so, sn, tangent),
    }
}

fn geometric_n<const N: usize>(b: &TimoBeam, ctx: &ElemCtx, forces: &[[f64; 6]]) -> Result<Vec<f64>> {
    let g = b.geom(ctx)?;
    let zero = vec![0.0; N];
    let cur = current(b, &g, &zero, &[], &[]);
    let planar = b.dim == 2;
    let idx = |a: usize, k: usize| -> Option<usize> {
        if planar {
            match k {
                0 | 1 => Some(3 * a + k),
                5 => Some(3 * a + 2),
                _ => None,
            }
        } else {
            Some(6 * a + k)
        }
    };
    let var = |a: usize, k: usize| -> Hd<N> { idx(a, k).map_or(Hd::cst(0.0), |i| Hd::var(0.0, i)) };
    let pts = points::<N>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    // f(q) = Σ w J λ δe(q)ᵀ s med fasta snittkrafter s: ∂f/∂q i referensläget är
    // den del av NLGEOM-tangenten som är proportionell mot spänningarna
    let mut f = vec![Hd::<N>::cst(0.0); N];
    for (p, s) in pts.iter().zip(forces) {
        for a in 0..b.nn {
            for k in 0..6 {
                let Some(i) = idx(a, k) else { continue };
                let mut v = Hd::cst(0.0);
                for r in 0..6 {
                    if s[r] != 0.0 {
                        v = v + p.de[r][6 * a + k] * s[r];
                    }
                }
                f[i] = f[i] + v * p.wl;
            }
        }
    }
    let mut k = vec![0.0; N * N];
    for i in 0..N {
        for j in 0..N {
            k[i * N + j] = 0.5 * (f[i].g[j] + f[j].g[i]);
        }
    }
    Ok(k)
}

/// Geometrisk styvhet (spänningsstyvhet) för *BUCKLE och *FREQUENCY med
/// förlast: den spänningsberoende delen av NLGEOM-tangenten i referensläget,
/// med snittkrafterna `forces` (per integrationspunkt, ordning som
/// `section_forces`) från en linjär lösning.
pub fn geometric(b: &TimoBeam, ctx: &ElemCtx, forces: &[[f64; 6]]) -> Result<Vec<f64>> {
    match (b.dim, b.nn) {
        (2, 2) => geometric_n::<6>(b, ctx, forces),
        (2, _) => geometric_n::<9>(b, ctx, forces),
        (_, 2) => geometric_n::<12>(b, ctx, forces),
        _ => geometric_n::<18>(b, ctx, forces),
    }
}

/// Snittkrafter [N, M1, M2, T, V1, V2] per integrationspunkt vid NLGEOM
/// (tvärsnittets aktuella axlar) för det godkända tillståndet `state`.
pub fn section_forces(b: &TimoBeam, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<[f64; 6]>> {
    let g = b.geom(ctx)?;
    let ng = b.n_gauss();
    let state = if state.len() > IP_STATE * ng { &state[state.len() - IP_STATE * ng..] } else { state };
    let cur = current(b, &g, u, &[], state);
    let c = current_c(b, &g, &cur);
    let var = |_: usize, _: usize| Hd::<1>::cst(0.0);
    let pts = points::<1>(b, &g, &cur, &var).ok_or_else(|| too_large(ctx, b))?;
    Ok(pts.iter().map(|p| std::array::from_fn(|r| (0..6).map(|s| c[r][s] * p.e[s].v).sum())).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_inverts_exp() {
        for p in [[0.0, 0.0, 0.0], [1e-5, -2e-6, 3e-6], [0.3, -0.2, 0.5], [1.2, 0.9, -1.4], [0.0, 0.0, 2.9]] {
            let r = t::exp_so3(p);
            let l = log_so3::<1>(&cst_m(&r)).unwrap();
            for i in 0..3 {
                assert!((l[i].v - p[i]).abs() < 1e-12, "{p:?}: {}", l[i].v);
            }
        }
    }

    #[test]
    fn log_derivatives() {
        // d/dw log(exp(w) R) i w = 0 mot differenser, och andraderivatan
        let p = [0.4, -0.7, 0.9];
        let r0 = t::exp_so3(p);
        let w: V3<3> = std::array::from_fn(|i| Hd::var(0.0, i));
        let l = log_so3(&spun(&w, &r0)).unwrap();
        let f = |w: [f64; 3]| t::log_so3(&t::mul(&t::exp_so3(w), &r0));
        let h = 1e-5;
        for k in 0..3 {
            let mut wp = [0.0; 3];
            wp[k] = h;
            let mut wm = [0.0; 3];
            wm[k] = -h;
            let (fp, fm) = (f(wp), f(wm));
            for i in 0..3 {
                assert!(((fp[i] - fm[i]) / (2.0 * h) - l[i].g[k]).abs() < 1e-8);
            }
            for m in 0..3 {
                let d = |a: f64, b: f64| {
                    let mut ww = [0.0; 3];
                    ww[k] += a;
                    ww[m] += b;
                    f(ww)
                };
                let (pp, pm, mp, mq) = (d(h, h), d(h, -h), d(-h, h), d(-h, -h));
                for i in 0..3 {
                    let fd = (pp[i] - pm[i] - mp[i] + mq[i]) / (4.0 * h * h);
                    assert!((fd - l[i].h[k][m]).abs() < 1e-4, "{i} {k} {m}: {fd} {}", l[i].h[k][m]);
                }
            }
        }
    }

    #[test]
    fn atan_coef_continuous() {
        let (a, b) = (atan_coef(1e-2 - 1e-13), atan_coef(1e-2 + 1e-13));
        assert!((a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-10 && (a.2 - b.2).abs() < 1e-8);
    }
}
