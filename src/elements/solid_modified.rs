//! Modifierad tiotodstetraeder C3D10M.
//!
//! Formulering:
//!
//! - Elementet har en intern nod (nod 11, tre förskjutningar) och delas i fyra
//!   hexaedrar, en per hörn: hörnet, de tre kantmittnoderna vid hörnet, de tre
//!   ytcentrumen på hörnets ytor och den interna noden. Hexaedrarna har
//!   trilinjär interpolation i det fysiska rummet.
//! - Ytcentrumet är ingen nod: dess läge och förskjutning är −1/15 av ytans
//!   hörn plus 2/5 av ytans kantmittnoder. Den interna nodens läge är
//!   w_h·Σhörn + w_m·Σkantmitt med w_h = 83/12200, w_m = 989/6100 (konstanter
//!   oberoende av geometrin).
//! - Varje hexaeder har en integrationspunkt med likformig töjning (medelvärdet
//!   av B över hexaedern, Flanagan–Belytschko), dvs. fyra integrationspunkter
//!   (punkt k hör till hörn k).
//! - Diagonal massa ρ·∫Nᵢ dV med hexaedrarnas funktioner (ytcentrumens andel
//!   fördelad som ovan); volymkrafter fördelas på samma sätt.
//!   Tryck på en yta fördelas som på fyra plana deltrianglar (hörn A/12,
//!   kantmitt A/4).
//! - Stabilisering: elementmatrisen ovan saknar styvhet för tre moder (den
//!   interna noden mot de övriga). Standardstabiliseringen (samma som
//!   HOURGLASS=ENHANCED) lägger på varje hexaeders ξηζ-mod (Flanagan–
//!   Belytschkos formvektor γ₄ i alla tre riktningar) samma styvhet som
//!   C3D8R med HOURGLASS=ENHANCED ger den moden: en 3×3-matris som är
//!   diagonal i hexaederns naturliga bas g_r = ∂x/∂ξ_r (se
//!   `enhanced_mode4`).
//!
//! Material: alla material som kontinuumelementen stöder (spänningsuppdatering
//! i de fyra punkterna, stabiliseringen elastisk med E och ν ur den linjära
//! tangenten). NLGEOM stöds inte.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip, DISTORTED};
use super::solid_higher::shapes::Shape;
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub struct ModifiedTet {
    name: &'static str,
}

pub static C3D10M: ModifiedTet = ModifiedTet { name: "C3D10M" };

pub static TYPES: &[&dyn ElementType] = &[&C3D10M];

/// Den interna nodens läge (och den enda dolda kopplingen i hourglass-moden).
pub const W_CORNER: f64 = 83.0 / 12200.0;
pub const W_MID: f64 = 989.0 / 6100.0;
/// Ytcentrum: andel per hörn respektive kantmittnod.
const F_CORNER: f64 = -1.0 / 15.0;
const F_MID: f64 = 0.4;

/// Kantmittnod mellan hörnen a och b (.inp-formatets numrering).
fn edge(a: usize, b: usize) -> usize {
    match (a.min(b), a.max(b)) {
        (0, 1) => 4,
        (1, 2) => 5,
        (0, 2) => 6,
        (0, 3) => 7,
        (1, 3) => 8,
        _ => 9,
    }
}

/// Ytor (hörn) i den interna ordningen för ytcentrumen 10..13.
const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]];

fn face_center(f: [usize; 3]) -> usize {
    let mut s = f;
    s.sort_unstable();
    10 + FACES.iter().position(|g| *g == s).unwrap()
}

/// Hexaederns 15 punkter (10 noder, 4 ytcentrum, intern nod) som
/// linjärkombinationer av elementets 11 noder.
fn tmat() -> [[f64; 11]; 15] {
    let mut t = [[0.0; 11]; 15];
    for (i, row) in t.iter_mut().enumerate().take(10) {
        row[i] = 1.0;
    }
    t[14][10] = 1.0;
    for (k, f) in FACES.iter().enumerate() {
        for &c in f {
            t[10 + k][c] = F_CORNER;
        }
        for a in 0..3 {
            for b in a + 1..3 {
                t[10 + k][edge(f[a], f[b])] = F_MID;
            }
        }
    }
    t
}

/// De fyra hexaedrarna (punktindex 0..14), hexaeder k vid hörn k.
fn hexes() -> [[usize; 8]; 4] {
    let mut h = [[0; 8]; 4];
    for (a, hx) in h.iter_mut().enumerate() {
        let o: Vec<usize> = (0..4).filter(|&b| b != a).collect();
        let (b, c, d) = (o[0], o[1], o[2]);
        *hx = [a, edge(a, b), face_center([a, b, c]), edge(a, c), edge(a, d), face_center([a, b, d]), 14, face_center([a, c, d])];
    }
    h
}

const NAT: [[f64; 3]; 8] =
    [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]];

fn hex_shape(p: [f64; 3]) -> ([f64; 8], [[f64; 3]; 8]) {
    let mut n = [0.0; 8];
    let mut d = [[0.0; 3]; 8];
    for i in 0..8 {
        let s = NAT[i];
        let f = [1.0 + s[0] * p[0], 1.0 + s[1] * p[1], 1.0 + s[2] * p[2]];
        n[i] = f[0] * f[1] * f[2] / 8.0;
        d[i] = [s[0] * f[1] * f[2] / 8.0, s[1] * f[0] * f[2] / 8.0, s[2] * f[0] * f[1] / 8.0];
    }
    (n, d)
}

/// En hexaeder: volym, medelgradienter per elementnod (11), formvektorn γ₄
/// per elementnod och ∫N per punkt (15).
struct Sub {
    v: f64,
    dndx: Vec<[f64; 3]>,
    n_center: Vec<f64>,
    gamma: Vec<f64>,
    wn: [f64; 8],
    pts: [usize; 8],
    /// Jacobianen i hexaederns mitt: rad r = ∂x/∂ξ_r.
    j0: [[f64; 3]; 3],
}

fn subs(x10: &[[f64; 3]]) -> Result<Vec<Sub>> {
    let t = tmat();
    let mut p = [[0.0; 3]; 15];
    for (i, row) in t.iter().enumerate() {
        for c in 0..3 {
            p[i][c] = (0..10).map(|j| row[j] * x10[j][c]).sum();
        }
    }
    for c in 0..3 {
        p[14][c] = (0..4).map(|j| W_CORNER * x10[j][c]).sum::<f64>() + (4..10).map(|j| W_MID * x10[j][c]).sum::<f64>();
    }
    let g = common::G2;
    let mut out = Vec::new();
    for hx in hexes() {
        let x: Vec<[f64; 3]> = hx.iter().map(|&i| p[i]).collect();
        let mut bsum = [[0.0; 3]; 8];
        let mut vol = 0.0;
        let mut wn = [0.0; 8];
        for &a in &[-g, g] {
            for &b in &[-g, g] {
                for &c in &[-g, g] {
                    let (n, dn) = hex_shape([a, b, c]);
                    let mut j = [[0.0; 3]; 3];
                    for r in 0..3 {
                        for cc in 0..3 {
                            j[r][cc] = (0..8).map(|i| dn[i][r] * x[i][cc]).sum();
                        }
                    }
                    let (ji, det) = common::inv3(j);
                    if !(det.abs() > 0.0) {
                        return fail(DISTORTED);
                    }
                    let w = det.abs();
                    for i in 0..8 {
                        for cc in 0..3 {
                            bsum[i][cc] += (0..3).map(|r| ji[cc][r] * dn[i][r]).sum::<f64>() * w;
                        }
                        wn[i] += n[i] * w;
                    }
                    vol += w;
                }
            }
        }
        let b: Vec<[f64; 3]> = bsum.iter().map(|r| [r[0] / vol, r[1] / vol, r[2] / vol]).collect();
        // γ₄ = Γ₄ − b·(x·Γ₄)
        let h4: Vec<f64> = NAT.iter().map(|s| s[0] * s[1] * s[2]).collect();
        let hx4: [f64; 3] = [0, 1, 2].map(|c| (0..8).map(|i| h4[i] * x[i][c]).sum());
        let gam: Vec<f64> = (0..8).map(|i| h4[i] - (0..3).map(|c| b[i][c] * hx4[c]).sum::<f64>()).collect();
        let (n0, d0) = hex_shape([0.0; 3]);
        let mut j0 = [[0.0; 3]; 3];
        for r in 0..3 {
            for cc in 0..3 {
                j0[r][cc] = (0..8).map(|i| d0[i][r] * x[i][cc]).sum();
            }
        }
        let mut dndx = vec![[0.0; 3]; 11];
        let mut nc = vec![0.0; 11];
        let mut g11 = vec![0.0; 11];
        for (k, &pi) in hx.iter().enumerate() {
            for j in 0..11 {
                let tj = t[pi][j];
                if tj != 0.0 {
                    for c in 0..3 {
                        dndx[j][c] += tj * b[k][c];
                    }
                    nc[j] += tj * n0[k];
                    g11[j] += tj * gam[k];
                }
            }
        }
        out.push(Sub { v: vol, dndx, n_center: nc, gamma: g11, wn, pts: hx, j0 });
    }
    Ok(out)
}

/// E och ν ur en (isotrop) linjär tangent.
fn young_poisson(d: &[f64]) -> (f64, f64) {
    let g = d[3 * 6 + 3];
    let lam = d[1];
    if !(lam + g > 0.0) {
        return (2.0 * g, 0.0);
    }
    (g * (3.0 * lam + 2.0 * g) / (lam + g), lam / (2.0 * (lam + g)))
}

/// Förstärkt hourglass-styvhet (HOURGLASS=ENHANCED, Puso) för en
/// hexaeders ξηζ-mod, som 3×3-matris Q på formvektorn γ₄ (K = Σ Q_cd (γ₄⊗e_c)(γ₄⊗e_d)ᵀ).
///
/// För C3D8R och C3D10M: Q = Σ_r Λ_r g_r g_rᵀ med g_r = ∂x/∂ξ_r i mitten
/// (raderna i J₀), Λ_r = E·V·φ_r/36, V = 8|det J₀| och
/// φ_r = 1ᵀH⁻¹1, där H (2×2) bildas av kantlängderna a = 2|g_r| och
/// b, c = 2|g_s|, 2|g_t| (s, t ≠ r):
/// H = [[a⁴ + b⁴ + 2ν a²b², a⁴ + ν(a²(b² + c²) − b²c²)], [·, a⁴ + c⁴ + 2ν a²c²]].
pub fn enhanced_mode4(j0: &[[f64; 3]; 3], e: f64, nu: f64) -> [[f64; 3]; 3] {
    let det = j0[0][0] * (j0[1][1] * j0[2][2] - j0[1][2] * j0[2][1]) - j0[0][1] * (j0[1][0] * j0[2][2] - j0[1][2] * j0[2][0])
        + j0[0][2] * (j0[1][0] * j0[2][1] - j0[1][1] * j0[2][0]);
    let v = 8.0 * det.abs();
    let l: [f64; 3] = [0, 1, 2].map(|r| 4.0 * (j0[r][0] * j0[r][0] + j0[r][1] * j0[r][1] + j0[r][2] * j0[r][2]));
    let mut q = [[0.0; 3]; 3];
    for r in 0..3 {
        let a2 = l[r];
        let (b2, c2) = (l[(r + 1) % 3], l[(r + 2) % 3]);
        let h11 = a2 * a2 + b2 * b2 + 2.0 * nu * a2 * b2;
        let h22 = a2 * a2 + c2 * c2 + 2.0 * nu * a2 * c2;
        let h12 = a2 * a2 + nu * (a2 * (b2 + c2) - b2 * c2);
        let dh = h11 * h22 - h12 * h12;
        if !(dh.abs() > 0.0) {
            continue;
        }
        let phi = (h11 + h22 - 2.0 * h12) / dh;
        let lam = e * v * phi / 36.0;
        for a in 0..3 {
            for b in 0..3 {
                q[a][b] += lam * j0[r][a] * j0[r][b];
            }
        }
    }
    q
}

impl ModifiedTet {
    fn x10<'a>(&self, ctx: &'a ElemCtx) -> &'a [[f64; 3]] {
        &ctx.x[..10]
    }

    fn ips(&self, ctx: &ElemCtx) -> Result<(Vec<Ip>, Vec<Vec<f64>>, Vec<Sub>)> {
        let ss = subs(self.x10(ctx))?;
        let mut ips = Vec::new();
        let mut bs = Vec::new();
        for s in &ss {
            bs.push(common::bmat(StressMode::ThreeD, &s.dndx));
            ips.push(Ip { n: s.n_center.clone(), dndx: s.dndx.clone(), w: s.v });
        }
        Ok((ips, bs, ss))
    }

    /// Stabiliseringens matris (33×33).
    fn stabilization(&self, ctx: &ElemCtx, ss: &[Sub]) -> Result<Vec<f64>> {
        let d = ctx.material(self.name)?.tangent_linear(StressMode::ThreeD)?;
        let (e, nu) = young_poisson(&d);
        let n = 33;
        let mut k = vec![0.0; n * n];
        for s in ss {
            let q = enhanced_mode4(&s.j0, e, nu);
            for i in 0..11 {
                for j in 0..11 {
                    let v = s.gamma[i] * s.gamma[j];
                    if v == 0.0 {
                        continue;
                    }
                    for a in 0..3 {
                        for b in 0..3 {
                            k[(3 * i + a) * n + 3 * j + b] += v * q[a][b];
                        }
                    }
                }
            }
        }
        Ok(k)
    }

    /// ∫Nᵢ dV för de 11 noderna.
    fn weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let ss = subs(self.x10(ctx))?;
        let t = tmat();
        let mut w = vec![0.0; 11];
        for s in &ss {
            for (k, &pi) in s.pts.iter().enumerate() {
                for j in 0..11 {
                    w[j] += t[pi][j] * s.wn[k];
                }
            }
        }
        Ok(w)
    }
}

/// Elementmatrisen utan stabilisering (33×33) och hexaedrarnas formvektorer
/// γ₄ (11 värden var). Hela elementmatrisen skiljer sig från huvuddelen bara inom
/// rummet som spänns av γ₄ ⊗ eᵢ (används i tests/elements2.rs).
pub fn main_part(ctx: &ElemCtx) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let (ips, bs, ss) = C3D10M.ips(ctx)?;
    let k = common::stiffness_ctx(ctx, StressMode::ThreeD, &ips, &bs, ctx.material(C3D10M.name)?)?;
    Ok((k, ss.into_iter().map(|s| s.gamma).collect()))
}

impl ElementType for ModifiedTet {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Continuum3D
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        11
    }
    fn n_connectivity(&self) -> usize {
        10
    }
    fn internal_nodes(&self) -> usize {
        1
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3]
    }
    fn n_ips(&self) -> usize {
        4
    }
    fn vtk_type(&self) -> u8 {
        24
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        Shape::Tet10.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(common::init_state(4, StressMode::ThreeD, ctx.material(self.name)?))
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (ips, bs, ss) = self.ips(ctx)?;
        let mut k = common::stiffness_ctx(ctx, StressMode::ThreeD, &ips, &bs, ctx.material(self.name)?)?;
        for (a, b) in k.iter_mut().zip(self.stabilization(ctx, &ss)?) {
            *a += b;
        }
        Ok(k)
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("geometrisk olinjäritet (NLGEOM)", self.name);
        }
        let (ips, bs, ss) = self.ips(ctx)?;
        let mut r = common::response(ctx, StressMode::ThreeD, &ips, &bs, ctx.material(self.name)?, u, du, so, sn, tangent)?;
        // stabiliseringen är elastisk (se docs/REFERENS.md)
        let kh = self.stabilization(ctx, &ss)?;
        let fh = common::matvec(&kh, u);
        for (a, b) in r.fint.iter_mut().zip(&fh) {
            *a += b;
        }
        if let Some(k) = r.k.as_mut() {
            for (a, b) in k.iter_mut().zip(&kh) {
                *a += b;
            }
        }
        Ok(r)
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        // de modifierade elementen har alltid diagonal massa
        let w = self.weights(ctx)?;
        let n = 33;
        let mut m = vec![0.0; n * n];
        for (i, wi) in w.iter().enumerate() {
            for c in 0..3 {
                m[(3 * i + c) * n + 3 * i + c] = ctx.density * wi;
            }
        }
        Ok(m)
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        let (ips, _, _) = self.ips(ctx)?;
        let stride = common::ip_stride(StressMode::ThreeD, ctx.material(self.name)?);
        let n = 33;
        let mut k = vec![0.0; n * n];
        for (q, ip) in ips.iter().enumerate() {
            if state.len() < (q + 1) * stride {
                continue;
            }
            let s = &state[q * stride..q * stride + 6];
            let sig = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
            for i in 0..11 {
                let gi = ip.dndx[i];
                let sg: [f64; 3] = [0, 1, 2].map(|a| (0..3).map(|b| sig[a][b] * gi[b]).sum());
                for j in 0..11 {
                    let v = common::dot(sg, ip.dndx[j]) * ip.w;
                    for c in 0..3 {
                        k[(3 * i + c) * n + 3 * j + c] += v;
                    }
                }
            }
        }
        Ok(k)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let x = self.x10(ctx);
        if let Some(num) = load.kind.strip_prefix('P').and_then(|s| s.parse::<usize>().ok()) {
            let faces = Shape::Tet10.faces();
            if num == 0 || num > faces.len() {
                return fail(format!("{} har bara ytorna P1–P4, inte {}", self.name, load.kind));
            }
            let fl = faces[num - 1];
            let (c, m) = ([fl[0], fl[1], fl[2]], [fl[3], fl[4], fl[5]]);
            // fyra deltrianglar: tre hörntrianglar och mittriangeln
            let tris = [[c[0], m[0], m[2]], [c[1], m[1], m[0]], [c[2], m[2], m[1]], [m[0], m[1], m[2]]];
            let inside = common::centroid(&x[..4]);
            let mut f = vec![0.0; 33];
            for tr in tris {
                let xf: Vec<[f64; 3]> = tr.iter().map(|&i| x[i]).collect();
                let fe = common::face_pressure(&xf, inside, load.magnitude, 1.0)?;
                for (k, &i) in tr.iter().enumerate() {
                    for d in 0..3 {
                        f[3 * i + d] += fe[k][d];
                    }
                }
            }
            return Ok(f);
        }
        if let Some(b) = common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)? {
            let w = self.weights(ctx)?;
            let mut f = vec![0.0; 33];
            for (i, wi) in w.iter().enumerate() {
                for d in 0..3 {
                    f[3 * i + d] = b[d] * wi;
                }
            }
            return Ok(f);
        }
        not_implemented(&format!("lasttypen {}", load.kind), self.name)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        self.weights(ctx)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let (ips, bs, _) = self.ips(ctx)?;
        Ok(common::output_ctx(ctx, StressMode::ThreeD, &ips, &bs, ctx.material(self.name)?, u, state))
    }
}
