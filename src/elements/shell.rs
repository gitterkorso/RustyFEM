//! Skal- och membranelement: S4R, S4, M3D4, M3D3 och (i `shell_tri.rs`) S3, S3R.
//!
//! Plana fasettelement med sex frihetsgrader per nod (U1–U3, UR1–UR3) i ett
//! lokalt system per element: n = elementets normal (x,ξ × x,η i mitten), lokal
//! 1-axel = projektionen av global X på planet (global Z om normalen ligger
//! inom 0,1° från X), 2-axel = n × 1-axeln.
//!
//! Formulering (Mindlin–Reissner, se Bathe kap. 5.4, Hughes kap. 5,
//! Belytschko–Liu–Moran kap. 9):
//!
//! - Membran: bilinjär fyrhörning. S4R: enpunktsintegrering och
//!   Flanagan–Belytschko-stabilisering av de två membrana nollenergimoderna
//!   (stabiliseringsstyvhet 0,005·G·t·A·Σbᵢ·bᵢ/16 per mod). S4: enpunkts-
//!   integrering plus en antagen töjning för de böjande moderna (ren böjning i
//!   planet med tvärkontraktion ν̄ = 0,29, utan skjuvning – ger ren böjning utan
//!   skjuvlåsning).
//! - Böjning: S4R enpunktsintegrering plus stabilisering av de två
//!   rotationsmoderna (0,01125·G·t³/12·A·Σbᵢ·bᵢ/16 per mod); S4 full 2×2.
//! - Tvärskjuvning: antagen skjuvtöjning från kanternas mittpunkter (Bathe–
//!   Dvorkin/MITC4). Det konstanta skjuvfältet integreras i mitten; de två
//!   "timglas"-kombinationerna av kanttöjningarna får egna styvheter
//!   (α = ks·8/3·(|x,ξ|² + |x,η|²)/A för (Δξ + Δη) och 0,004·α för (Δξ − Δη)).
//!   ks = 5/6·G·t skalad med 1/(1 + 0,25·10⁻⁴·A/t²).
//! - Rotation kring normalen (drill): straff på skillnaden mellan nodens
//!   rotation och membranets rotation ½(v,x − u,y) i noden, styvhet
//!   kᵢ = G·t·det(Jᵢ)/(6 + 0,6·A/t²).
//!
//! Begränsningar för allmänt förvrängda fyrhörningar beskrivs i docs/REFERENS.md.
//!
//! Tvärsnitt: *SHELL SECTION och *SHELL GENERAL SECTION (homogena, isotropt
//! linjärelastiskt material). Utdata: S (S11, S22, S12) i sektionspunkterna 1
//! (SNEG) och n (SPOS), SF (SF1–SF6) och SM (SM1–SM3) per integrationspunkt.

#[path = "shell_tri.rs"]
pub mod tri;
#[path = "shell_nlgeom.rs"]
pub mod nlgeom;
#[path = "shell_section.rs"]
pub mod section;

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, cross, dot, norm, scale, sub, G2};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::{DistLoad, SectionKind};
use crate::model::{fail, Result};

/// Tvärkontraktion i S4:s antagna böjtöjning i planet.
const NU_BAR: f64 = 0.29;

// ---------------------------------------------------------------------------
// Små täta matrishjälpare (radvisa)
// ---------------------------------------------------------------------------

/// K += w · aᵀ C b för a, b med `r` rader och `n` kolumner, C r×r.
pub fn add_btcb(k: &mut [f64], n: usize, a: &[f64], c: &[f64], b: &[f64], r: usize, w: f64) {
    let mut cb = vec![0.0; r * n];
    for i in 0..r {
        for j in 0..r {
            let cij = c[i * r + j];
            if cij != 0.0 {
                for q in 0..n {
                    cb[i * n + q] += cij * b[j * n + q];
                }
            }
        }
    }
    for i in 0..r {
        for p in 0..n {
            let aip = a[i * n + p];
            if aip != 0.0 {
                let f = w * aip;
                for q in 0..n {
                    k[p * n + q] += f * cb[i * n + q];
                }
            }
        }
    }
}

/// K += c · g gᵀ.
pub fn add_outer(k: &mut [f64], n: usize, g: &[f64], c: f64) {
    for p in 0..n {
        if g[p] != 0.0 {
            for q in 0..n {
                k[p * n + q] += c * g[p] * g[q];
            }
        }
    }
}

fn inv2(j: [[f64; 2]; 2]) -> ([[f64; 2]; 2], f64) {
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    ([[j[1][1] / det, -j[0][1] / det], [-j[1][0] / det, j[0][0] / det]], det)
}

/// Invers av en liten n×n-matris (Gauss–Jordan med pivotering).
pub fn inv_n(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| m[i * n + c].abs().partial_cmp(&m[j * n + c].abs()).unwrap())?;
        if m[p * n + c].abs() < 1e-300 {
            return None;
        }
        for k in 0..n {
            m.swap(c * n + k, p * n + k);
            inv.swap(c * n + k, p * n + k);
        }
        let d = m[c * n + c];
        for k in 0..n {
            m[c * n + k] /= d;
            inv[c * n + k] /= d;
        }
        for r in 0..n {
            if r != c {
                let f = m[r * n + c];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[c * n + k];
                        inv[r * n + k] -= f * inv[c * n + k];
                    }
                }
            }
        }
    }
    Some(inv)
}

// ---------------------------------------------------------------------------
// Lokalt system och tvärsnitt
// ---------------------------------------------------------------------------

/// Elementets lokala system: R (rader e1, e2, n) och plana nodkoordinater.
pub struct Local {
    pub r: [[f64; 3]; 3],
    pub xl: Vec<[f64; 2]>,
}

/// Lokala standardaxlar för ett skal med normalen `n`.
pub fn shell_axes(n: [f64; 3]) -> [[f64; 3]; 3] {
    let n = scale(n, 1.0 / norm(n));
    // 1-axel: projektion av global X, eller av global Z om normalen ligger inom 0,1° från X
    let g = if n[0].abs() > (0.1f64.to_radians()).cos() { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let p = sub(g, scale(n, dot(g, n)));
    let e1 = scale(p, 1.0 / norm(p));
    let e2 = cross(n, e1);
    [e1, e2, n]
}

pub fn local_frame(x: &[[f64; 3]], label: i64, name: &str) -> Result<Local> {
    let nc = if x.len() == 3 || x.len() == 6 { 3 } else { 4 };
    let n = if nc == 3 { cross(sub(x[1], x[0]), sub(x[2], x[0])) } else { cross(sub(x[2], x[0]), sub(x[3], x[1])) };
    if norm(n) == 0.0 {
        return fail(format!("element {label} ({name}) har ingen area"));
    }
    let r = shell_axes(n);
    let c = common::centroid(&x[..nc]);
    let xl = x.iter().map(|p| {
        let d = sub(*p, c);
        [dot(r[0], d), dot(r[1], d)]
    }).collect();
    Ok(Local { r, xl })
}

/// Homogent skaltvärsnitt.
#[derive(Debug, Clone)]
pub struct ShellSec {
    pub t: f64,
    pub e: f64,
    pub nu: f64,
    pub g: f64,
    /// Plan spänning D (3×3, radvis).
    pub d: [f64; 9],
    pub rho: f64,
    /// Antal sektionspunkter (Simpson) för utdata, 0 = *SHELL GENERAL SECTION.
    pub n_int: usize,
    /// Tvärskjuvstyvhet (K11, K22, K12) från *TRANSVERSE SHEAR STIFFNESS.
    pub tss: Option<[f64; 3]>,
}

impl ShellSec {
    pub fn get(ctx: &ElemCtx, name: &str) -> Result<ShellSec> {
        let sec = ctx.section(name)?;
        // område struct_nonlin: *SHELL SECTION integreras genom tjockleken
        // (shell_section.rs) för olinjära material och skiktade tvärsnitt
        let shell_section = sec.raw.name == "SHELL SECTION";
        let (t, n_int) = match &sec.kind {
            SectionKind::Shell { thickness, n_int } => (*thickness, *n_int),
            SectionKind::Other if shell_section && sec.raw.has("COMPOSITE") => (section::composite_plies(sec)?.iter().map(|p| p.0).sum(), 3),
            SectionKind::Other if sec.raw.name == "MEMBRANE SECTION" => {
                let v = sec.raw.first_floats()?;
                (v.first().copied().filter(|&t| t != 0.0).unwrap_or(1.0), 1)
            }
            _ => return fail(format!("element {} ({name}) kräver *SHELL SECTION (homogent) eller *MEMBRANE SECTION", ctx.label)),
        };
        if !(t > 0.0) {
            return fail(format!("element {} ({name}): skaltjockleken måste vara positiv", ctx.label));
        }
        let mat = ctx.material(name)?;
        if !mat.is_linear() && !shell_section {
            return not_implemented(&format!("olinjärt material ({})", mat.name()), name);
        }
        let (e, nu, d) = match mat.elastic_moduli() {
            Some((e, nu)) => {
                let f = e / (1.0 - nu * nu);
                (e, nu, [f, f * nu, 0.0, f * nu, f, 0.0, 0.0, 0.0, f * (1.0 - nu) / 2.0])
            }
            // anisotropt material (t.ex. LAMINA) i *SHELL SECTION: materialets
            // plana spänningsmatris för stabilisering och drill
            None if shell_section => {
                let dm = mat.tangent_linear(crate::materials::StressMode::PlaneStress)?;
                let nu = dm[1] / dm[0];
                (dm[0] * (1.0 - nu * nu), nu, std::array::from_fn(|i| dm[i]))
            }
            None => return fail(format!("{name} kräver isotropt linjärelastiskt material")),
        };
        let tss = sec.raw.param(super::beam_timoshenko::section::TSS_PARAM).map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| crate::fem::model::parse_f64(x).unwrap_or(0.0)).collect();
            [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)]
        });
        // *SHELL GENERAL SECTION: spänningar redovisas i sektionspunkterna 1 och 3 (under- och översida)
        let n_int = if sec.raw.name == "SHELL GENERAL SECTION" { 3 } else { n_int };
        let g = if mat.elastic_moduli().is_some() { e / (2.0 * (1.0 + nu)) } else { d[8] };
        Ok(ShellSec { t, e, nu, g, d, rho: ctx.density, n_int, tss })
    }

    /// Skalad tvärskjuvstyvhet (isotrop del) för ett element med arean `a`.
    pub fn ks(&self, a: f64) -> f64 {
        let fp = 1.0 / (1.0 + 0.25e-4 * a / (self.t * self.t));
        match self.tss {
            Some(k) => k[0] * fp,
            None => 5.0 / 6.0 * self.g * self.t * fp,
        }
    }

    /// Sektionspunkternas z (Simpson, n punkter genom tjockleken).
    pub fn section_z(&self, sp: usize) -> f64 {
        let n = self.n_int.max(1);
        if n == 1 {
            return 0.0;
        }
        -self.t / 2.0 + self.t * (sp as f64 - 1.0) / (n as f64 - 1.0)
    }
}

// ---------------------------------------------------------------------------
// Bilinjär fyrhörning
// ---------------------------------------------------------------------------

const NAT4: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
const GP4: [(f64, f64); 4] = [(-G2, -G2), (G2, -G2), (-G2, G2), (G2, G2)];

/// Formfunktioner, naturliga derivator, J = [[x,ξ y,ξ],[x,η y,η]].
pub struct Q4Pt {
    pub n: [f64; 4],
    pub ds: [f64; 4],
    pub dt: [f64; 4],
    pub j: [[f64; 2]; 2],
    pub det: f64,
    /// dN/dx, dN/dy
    pub bx: [f64; 4],
    pub by: [f64; 4],
}

pub fn q4pt(x: &[[f64; 2]], s: f64, t: f64) -> Q4Pt {
    let mut n = [0.0; 4];
    let mut ds = [0.0; 4];
    let mut dt = [0.0; 4];
    for i in 0..4 {
        let (a, b) = (NAT4[i][0], NAT4[i][1]);
        n[i] = 0.25 * (1.0 + a * s) * (1.0 + b * t);
        ds[i] = 0.25 * a * (1.0 + b * t);
        dt[i] = 0.25 * b * (1.0 + a * s);
    }
    let mut j = [[0.0; 2]; 2];
    for i in 0..4 {
        j[0][0] += ds[i] * x[i][0];
        j[0][1] += ds[i] * x[i][1];
        j[1][0] += dt[i] * x[i][0];
        j[1][1] += dt[i] * x[i][1];
    }
    let (ji, det) = inv2(j);
    let mut bx = [0.0; 4];
    let mut by = [0.0; 4];
    for i in 0..4 {
        bx[i] = ji[0][0] * ds[i] + ji[0][1] * dt[i];
        by[i] = ji[1][0] * ds[i] + ji[1][1] * dt[i];
    }
    Q4Pt { n, ds, dt, j, det, bx, by }
}

/// Lokala frihetsgrader per nod i skalen: [u, v, w, θx, θy, θz].
const ND: usize = 6;

/// Membran-B (3 × 6nn) för dN/dx, dN/dy.
fn bmem(bx: &[f64], by: &[f64]) -> Vec<f64> {
    let nn = bx.len();
    let n = ND * nn;
    let mut b = vec![0.0; 3 * n];
    for i in 0..nn {
        b[i * ND] = bx[i];
        b[n + i * ND + 1] = by[i];
        b[2 * n + i * ND] = by[i];
        b[2 * n + i * ND + 1] = bx[i];
    }
    b
}

/// Böj-B (3 × 6nn): κ = [θy,x, −θx,y, θy,y − θx,x].
fn bbend(bx: &[f64], by: &[f64]) -> Vec<f64> {
    let nn = bx.len();
    let n = ND * nn;
    let mut b = vec![0.0; 3 * n];
    for i in 0..nn {
        b[i * ND + 4] = bx[i];
        b[n + i * ND + 3] = -by[i];
        b[2 * n + i * ND + 4] = by[i];
        b[2 * n + i * ND + 3] = -bx[i];
    }
    b
}

/// Kovariant tvärskjuvtöjning e_ξ (dir = 0) eller e_η (dir = 1) i (s, t).
fn cov_shear(x: &[[f64; 2]], s: f64, t: f64, dir: usize) -> Vec<f64> {
    let p = q4pt(x, s, t);
    let d = if dir == 0 { p.ds } else { p.dt };
    let xv = p.j[dir];
    let mut g = vec![0.0; ND * 4];
    for i in 0..4 {
        g[i * ND + 2] = d[i];
        g[i * ND + 4] = p.n[i] * xv[0];
        g[i * ND + 3] = -p.n[i] * xv[1];
    }
    g
}

/// Timglasvektorn γ = h − (h·x) bx − (h·y) by (Flanagan–Belytschko, onormerad).
fn hourglass(x: &[[f64; 2]], p0: &Q4Pt) -> [f64; 4] {
    let h = [1.0, -1.0, 1.0, -1.0];
    let hx: f64 = (0..4).map(|i| h[i] * x[i][0]).sum();
    let hy: f64 = (0..4).map(|i| h[i] * x[i][1]).sum();
    [0, 1, 2, 3].map(|i| h[i] - hx * p0.bx[i] - hy * p0.by[i])
}

/// Tvärskjuvningens kanttöjningar (4 × 24) och koefficientmatris (4 × 4),
/// samt matrisen M (2 × 4) som ger det konstanta skjuvfältet γ0 = M·s.
struct ShearModel {
    s: Vec<f64>,
    c: [f64; 16],
    m: [[f64; 4]; 2],
}

fn shear_model(x: &[[f64; 2]], ks: f64) -> ShearModel {
    shear_model_k(x, [ks, ks, 0.0])
}

/// Som `shear_model` med en riktningsberoende tvärskjuvstyvhet [K11, K22, K12]
/// i elementets lokala axlar (område struct_nonlin: skiktade tvärsnitt);
/// timglasdelen använder medelvärdet (K11 + K22)/2.
fn shear_model_k(x: &[[f64; 2]], kk: [f64; 3]) -> ShearModel {
    let ks = 0.5 * (kk[0] + kk[1]);
    let rows = [cov_shear(x, 0.0, -1.0, 0), cov_shear(x, -1.0, 0.0, 1), cov_shear(x, 0.0, 1.0, 0), cov_shear(x, 1.0, 0.0, 1)];
    let mut s = vec![0.0; 4 * 24];
    for (r, v) in rows.iter().enumerate() {
        s[r * 24..(r + 1) * 24].copy_from_slice(v);
    }
    let p0 = q4pt(x, 0.0, 0.0);
    let a = 4.0 * p0.det;
    let (ji, _) = inv2(p0.j);
    // M = J0⁻¹ P, P = [[½,0,½,0],[0,½,0,½]]
    let m = [
        [0.5 * ji[0][0], 0.5 * ji[0][1], 0.5 * ji[0][0], 0.5 * ji[0][1]],
        [0.5 * ji[1][0], 0.5 * ji[1][1], 0.5 * ji[1][0], 0.5 * ji[1][1]],
    ];
    // område normals: timglasdelen delas upp med kanternas egna tangenter
    // x,ξ(0, ∓1), x,η(∓1, 0) (för parallellogram = mittens); för
    // trapetser blir felet i böj-/skjuvblocket av andra ordningen
    // i förvrängningen
    let et = [q4pt(x, 0.0, -1.0).j[0], q4pt(x, -1.0, 0.0).j[1], q4pt(x, 0.0, 1.0).j[0], q4pt(x, 1.0, 0.0).j[1]];
    let mut c = [0.0; 16];
    for i in 0..4 {
        for j in 0..4 {
            c[i * 4 + j] = a * (kk[0] * m[0][i] * m[0][j] + kk[1] * m[1][i] * m[1][j] + kk[2] * (m[0][i] * m[1][j] + m[1][i] * m[0][j]));
        }
    }
    // timglasdelen: [T Q2]⁻¹, T = kanternas tangenter, Q2 = [(-1,-1,1,1), (-1,1,1,-1)]
    let xs = p0.j[0];
    let xt = p0.j[1];
    let tq = [
        et[0][0], et[0][1], -1.0, -1.0, //
        et[1][0], et[1][1], -1.0, 1.0, //
        et[2][0], et[2][1], 1.0, 1.0, //
        et[3][0], et[3][1], 1.0, -1.0,
    ];
    if let Some(mi) = inv_n(&tq, 4) {
        let alpha = ks * 8.0 / 3.0 * (xs[0] * xs[0] + xs[1] * xs[1] + xt[0] * xt[0] + xt[1] * xt[1]) / a;
        let coef = [alpha, 0.004 * alpha];
        for (h, &cf) in coef.iter().enumerate() {
            let row = &mi[(2 + h) * 4..(3 + h) * 4];
            for i in 0..4 {
                for j in 0..4 {
                    c[i * 4 + j] += cf * row[i] * row[j];
                }
            }
        }
    }
    ShearModel { s, c, m }
}

/// Drillstyvhet: Σ kᵢ (θzᵢ − ωᵢ)², ωᵢ = ½(v,x − u,y) i nod i.
fn add_drill(k: &mut [f64], x: &[[f64; 2]], sec: &ShellSec, a: f64) {
    let n = ND * 4;
    // = G·t/(6 + 0,6·A/t²) för homogena tvärsnitt; följer tvärskjuvstyvheten
    let kts = sec.tss.map(|k| k[0]).unwrap_or(5.0 / 6.0 * sec.g * sec.t);
    let kfac = 0.2 * kts / (1.0 + 0.1 * a / (sec.t * sec.t));
    for i in 0..4 {
        let p = q4pt(x, NAT4[i][0], NAT4[i][1]);
        let mut g = vec![0.0; n];
        g[i * ND + 5] = 1.0;
        for j in 0..4 {
            g[j * ND] += 0.5 * p.by[j];
            g[j * ND + 1] -= 0.5 * p.bx[j];
        }
        add_outer(k, n, &g, kfac * p.det);
    }
}

/// Plan spänning D·t (membran) och D·t³/12 (böjning) som 6×6 för [ε; κ].
fn section_c(sec: &ShellSec) -> [f64; 36] {
    let mut c = [0.0; 36];
    let tb = sec.t.powi(3) / 12.0;
    for i in 0..3 {
        for j in 0..3 {
            c[i * 6 + j] = sec.d[i * 3 + j] * sec.t;
            c[(i + 3) * 6 + j + 3] = sec.d[i * 3 + j] * tb;
        }
    }
    c
}

fn stack(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut v = a.to_vec();
    v.extend_from_slice(b);
    v
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum QuadKind {
    S4R,
    S4,
}

pub struct ShellQuad {
    name: &'static str,
    kind: QuadKind,
}

pub static S4R: ShellQuad = ShellQuad { name: "S4R", kind: QuadKind::S4R };
pub static S4: ShellQuad = ShellQuad { name: "S4", kind: QuadKind::S4 };

impl ShellQuad {
    /// Lokal styvhet (24 × 24) i [u, v, w, θx, θy, θz] per nod.
    fn local_k(&self, x: &[[f64; 2]], sec: &ShellSec) -> Vec<f64> {
        let n = 24;
        let mut k = vec![0.0; n * n];
        let p0 = q4pt(x, 0.0, 0.0);
        let a = 4.0 * p0.det;
        let c = section_c(sec);
        let bm0 = bmem(&p0.bx, &p0.by);
        let bb0 = bbend(&p0.bx, &p0.by);
        let gam = hourglass(x, &p0);
        let bb: f64 = (0..4).map(|i| p0.bx[i] * p0.bx[i] + p0.by[i] * p0.by[i]).sum();
        match self.kind {
            QuadKind::S4R => {
                let b = stack(&bm0, &bb0);
                add_btcb(&mut k, n, &b, &c, &b, 6, a);
                let cm = 0.0003125 * sec.g * sec.t * a * bb;
                let cb = 0.01125 * sec.g * sec.t.powi(3) / 12.0 * a * bb;
                for (d, cf) in [(0usize, cm), (1, cm), (3, cb), (4, cb)] {
                    let mut g = vec![0.0; n];
                    for i in 0..4 {
                        g[i * ND + d] = gam[i];
                    }
                    add_outer(&mut k, n, &g, cf);
                }
            }
            QuadKind::S4 => {
                // membran: enpunkt + antagen böjtöjning i planet
                let mut cm3 = [0.0; 9];
                for i in 0..9 {
                    cm3[i] = sec.d[i] * sec.t;
                }
                add_btcb(&mut k, n, &bm0, &cm3, &bm0, 3, a);
                let (ji, _) = inv2(p0.j);
                let gx = [ji[0][0], ji[1][0]];
                let ge = [ji[0][1], ji[1][1]];
                let ef = sec.e * (1.0 + NU_BAR * NU_BAR - 2.0 * sec.nu * NU_BAR) / (1.0 - sec.nu * sec.nu);
                let f = sec.t * ef * a / 3.0 / 16.0;
                let ch = [
                    [f * (gx[0] * gx[0] + ge[0] * ge[0]), f * (gx[0] * gx[1] + ge[0] * ge[1])],
                    [f * (gx[1] * gx[0] + ge[1] * ge[0]), f * (gx[1] * gx[1] + ge[1] * ge[1])],
                ];
                for i in 0..4 {
                    for j in 0..4 {
                        for (di, dj) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                            k[(i * ND + di) * n + j * ND + dj] += gam[i] * gam[j] * ch[di][dj];
                        }
                    }
                }
                // böjning: full 2×2
                let mut cb3 = [0.0; 9];
                let tb = sec.t.powi(3) / 12.0;
                for i in 0..9 {
                    cb3[i] = sec.d[i] * tb;
                }
                for &(s, t) in &GP4 {
                    let p = q4pt(x, s, t);
                    let b = bbend(&p.bx, &p.by);
                    add_btcb(&mut k, n, &b, &cb3, &b, 3, p.det);
                }
            }
        }
        let sm = shear_model(x, sec.ks(a));
        add_btcb(&mut k, n, &sm.s, &sm.c, &sm.s, 4, 1.0);
        add_drill(&mut k, x, sec, a);
        k
    }

    fn setup(&self, ctx: &ElemCtx) -> Result<(Local, ShellSec)> {
        let loc = local_frame(&ctx.x, ctx.label, self.name)?;
        let p0 = q4pt(&loc.xl, 0.0, 0.0);
        for &(s, t) in &[(0.0, 0.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            if q4pt(&loc.xl, s, t).det <= 0.0 || p0.det <= 0.0 {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
        }
        Ok((loc, ShellSec::get(ctx, self.name)?))
    }

    /// Kinematik för integrering genom tjockleken (område struct_nonlin,
    /// shell_section.rs): B för [ε; κ] per integrationspunkt och den elastiska
    /// resten (tvärskjuvning, drill och för S4R timglasstabilisering).
    fn kin(&self, ctx: &ElemCtx) -> Result<section::Kin> {
        let (loc, mut sec) = self.setup(ctx)?;
        let x = &loc.xl;
        let n = 24;
        let p0 = q4pt(x, 0.0, 0.0);
        let a = 4.0 * p0.det;
        let kk = section::shear_stiffness(ctx, self.name, &loc, &mut sec, a)?;
        let sm = shear_model_k(x, kk);
        let mut gs = vec![0.0; 2 * n];
        for r in 0..2 {
            for q in 0..n {
                gs[r * n + q] = (0..4).map(|k| sm.m[r][k] * sm.s[k * n + q]).sum();
            }
        }
        let mut rest = vec![0.0; n * n];
        add_btcb(&mut rest, n, &sm.s, &sm.c, &sm.s, 4, 1.0);
        add_drill(&mut rest, x, &sec, a);
        let bm0 = bmem(&p0.bx, &p0.by);
        let ips = match self.kind {
            QuadKind::S4R => {
                let gam = hourglass(x, &p0);
                let bb: f64 = (0..4).map(|i| p0.bx[i] * p0.bx[i] + p0.by[i] * p0.by[i]).sum();
                let cm = 0.0003125 * sec.g * sec.t * a * bb;
                let cb = 0.01125 * sec.g * sec.t.powi(3) / 12.0 * a * bb;
                for (d, cf) in [(0usize, cm), (1, cm), (3, cb), (4, cb)] {
                    let mut g = vec![0.0; n];
                    for i in 0..4 {
                        g[i * ND + d] = gam[i];
                    }
                    add_outer(&mut rest, n, &g, cf);
                }
                vec![(stack(&bm0, &bbend(&p0.bx, &p0.by)), a)]
            }
            QuadKind::S4 => {
                // membranet: enpunktsdelen plus den antagna böjtöjningen i
                // integrationspunkten (samma töjning som `strains`)
                let gam = hourglass(x, &p0);
                let (ji, _) = inv2(p0.j);
                let grads = [[ji[0][0], ji[1][0]], [ji[0][1], ji[1][1]]];
                GP4.iter()
                    .map(|&(s, t)| {
                        let p = q4pt(x, s, t);
                        let mut bm = bm0.clone();
                        for (k, gr) in grads.iter().enumerate() {
                            let coord = if k == 0 { t } else { s };
                            let l = (gr[0] * gr[0] + gr[1] * gr[1]).sqrt();
                            let nv = [gr[0] / l, gr[1] / l];
                            let tv = [-nv[1], nv[0]];
                            let v = [
                                nv[0] * nv[0] - NU_BAR * tv[0] * tv[0],
                                nv[1] * nv[1] - NU_BAR * tv[1] * tv[1],
                                2.0 * (nv[0] * nv[1] - NU_BAR * tv[0] * tv[1]),
                            ];
                            for i in 0..4 {
                                for (dd, comp) in [(0usize, nv[0]), (1, nv[1])] {
                                    let cf = l * coord * comp * gam[i] / 4.0;
                                    for r in 0..3 {
                                        bm[r * n + i * ND + dd] += cf * v[r];
                                    }
                                }
                            }
                        }
                        (stack(&bm, &bbend(&p.bx, &p.by)), p.det)
                    })
                    .collect()
            }
        };
        let nip = ips.len();
        Ok(section::Kin { loc, sec, ips, rest, shear: vec![gs; nip], kshear: kk, rule: nlgeom::MemRule::Center })
    }

    /// Lokala förskjutningar ur globala.
    fn local_u(loc: &Local, u: &[f64], nn: usize) -> Vec<f64> {
        let t = common::block_diag(&loc.r, 2 * nn);
        common::matvec(&t, u)
    }

    /// Generaliserade töjningar [ε(3), κ(3), γ(2)] i integrationspunkterna.
    fn strains(&self, x: &[[f64; 2]], sec: &ShellSec, ul: &[f64]) -> Vec<[f64; 8]> {
        let p0 = q4pt(x, 0.0, 0.0);
        let a = 4.0 * p0.det;
        let sm = shear_model(x, sec.ks(a));
        let sv: Vec<f64> = (0..4).map(|r| (0..24).map(|q| sm.s[r * 24 + q] * ul[q]).sum()).collect();
        let g0 = [0, 1].map(|r| (0..4).map(|q| sm.m[r][q] * sv[q]).sum::<f64>());
        let mv = |b: &[f64]| -> [f64; 3] { [0, 1, 2].map(|r| (0..24).map(|q| b[r * 24 + q] * ul[q]).sum()) };
        match self.kind {
            QuadKind::S4R => {
                let e = mv(&bmem(&p0.bx, &p0.by));
                let kk = mv(&bbend(&p0.bx, &p0.by));
                vec![[e[0], e[1], e[2], kk[0], kk[1], kk[2], g0[0], g0[1]]]
            }
            QuadKind::S4 => {
                let e0 = mv(&bmem(&p0.bx, &p0.by));
                // timglasamplituder q = γ·u/4 och den antagna böjtöjningen
                let gam = hourglass(x, &p0);
                let qx: f64 = (0..4).map(|i| gam[i] * ul[i * ND]).sum::<f64>() / 4.0;
                let qy: f64 = (0..4).map(|i| gam[i] * ul[i * ND + 1]).sum::<f64>() / 4.0;
                let (ji, _) = inv2(p0.j);
                let grads = [[ji[0][0], ji[1][0]], [ji[0][1], ji[1][1]]];
                GP4.iter()
                    .map(|&(s, t)| {
                        let p = q4pt(x, s, t);
                        let mut e = e0;
                        for (k, gr) in grads.iter().enumerate() {
                            let coord = if k == 0 { t } else { s };
                            let l = (gr[0] * gr[0] + gr[1] * gr[1]).sqrt();
                            let nv = [gr[0] / l, gr[1] / l];
                            let tv = [-nv[1], nv[0]];
                            let qn = qx * nv[0] + qy * nv[1];
                            let amp = l * coord * qn;
                            // ε = amp (n⊗n − ν̄ t⊗t)
                            e[0] += amp * (nv[0] * nv[0] - NU_BAR * tv[0] * tv[0]);
                            e[1] += amp * (nv[1] * nv[1] - NU_BAR * tv[1] * tv[1]);
                            e[2] += amp * 2.0 * (nv[0] * nv[1] - NU_BAR * tv[0] * tv[1]);
                        }
                        let kk = mv(&bbend(&p.bx, &p.by));
                        [e[0], e[1], e[2], kk[0], kk[1], kk[2], g0[0], g0[1]]
                    })
                    .collect()
            }
        }
    }
}

/// Global matris ur lokal (6 frihetsgrader per nod).
pub fn to_global(kl: &[f64], r: &[[f64; 3]; 3], nn: usize) -> Vec<f64> {
    let t = common::block_diag(r, 2 * nn);
    common::congruence(kl, &t, ND * nn)
}

/// Global vektor ur lokal: f = Tᵀ fl.
pub fn vec_to_global(fl: &[f64], r: &[[f64; 3]; 3], blocks: usize) -> Vec<f64> {
    let t = common::block_diag(r, blocks);
    let n = 3 * blocks;
    (0..n).map(|j| (0..n).map(|i| t[i * n + j] * fl[i]).sum()).collect()
}

/// Utdata för skal från generaliserade töjningar per integrationspunkt.
pub fn shell_output(sec: &ShellSec, strains: &[[f64; 8]], ks: f64) -> Vec<IpField> {
    let mut s = IpField::new("S", &["S11", "S22", "S12"]);
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let d = &sec.d;
    let dv = |e: [f64; 3]| -> [f64; 3] { [0, 1, 2].map(|i| (0..3).map(|j| d[i * 3 + j] * e[j]).sum()) };
    for (q, e) in strains.iter().enumerate() {
        let ip = q + 1;
        let em = [e[0], e[1], e[2]];
        let kk = [e[3], e[4], e[5]];
        let sn = dv(em);
        let mb = dv(kk);
        let tb = sec.t.powi(3) / 12.0;
        sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, sn[2] * sec.t, ks * e[6], ks * e[7], 0.0]));
        sm.points.push((ip, 0, vec![mb[0] * tb, mb[1] * tb, mb[2] * tb]));
        if sec.n_int > 0 {
            let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
            for sp in sps {
                let z = sec.section_z(sp);
                let st = dv([em[0] + z * kk[0], em[1] + z * kk[1], em[2] + z * kk[2]]);
                s.points.push((ip, sp, st.to_vec()));
            }
        }
    }
    let mut out = vec![sf, sm];
    if !s.points.is_empty() {
        out.insert(0, s);
    }
    out
}

/// Konsistent eller diagonal massa (24 × 24) för en fyrhörning i lokala dof.
fn quad_mass(x: &[[f64; 2]], sec: &ShellSec, lumped: bool) -> Vec<f64> {
    let n = 24;
    let mut m = vec![0.0; n * n];
    let mt = sec.rho * sec.t;
    let mr = sec.rho * sec.t.powi(3) / 12.0;
    for &(s, t) in &GP4 {
        let p = q4pt(x, s, t);
        for i in 0..4 {
            for j in 0..4 {
                let f = p.n[i] * p.n[j] * p.det;
                let (ri, cj) = if lumped { (i, i) } else { (i, j) };
                for d in 0..3 {
                    m[(ri * ND + d) * n + cj * ND + d] += f * mt;
                }
                for d in 3..5 {
                    m[(ri * ND + d) * n + cj * ND + d] += f * mr;
                }
                if lumped {
                    // område struct_nonlin: diagonal massa (*DYNAMIC) har samma
                    // rotationströghet kring normalen som kring planets axlar
                    m[(ri * ND + 5) * n + cj * ND + 5] += f * mr;
                }
            }
        }
    }
    m
}

/// Ekvivalenta nodkrafter (lokala, 3 per nod) för last per area `q` (lokal riktning).
fn quad_area_load(x: &[[f64; 2]], q: [f64; 3]) -> Vec<f64> {
    let mut f = vec![0.0; 12];
    for &(s, t) in &GP4 {
        let p = q4pt(x, s, t);
        for i in 0..4 {
            for d in 0..3 {
                f[3 * i + d] += p.n[i] * p.det * q[d];
            }
        }
    }
    f
}

/// Last per ytenhet i globala koordinater för P, GRAV, BX/BY/BZ.
pub fn area_load_vector(load: &DistLoad, normal: [f64; 3], sec: &ShellSec, name: &str) -> Result<Option<[f64; 3]>> {
    match load.kind.as_str() {
        // positivt tryck P verkar i normalens riktning
        "P" => Ok(Some(scale(normal, load.magnitude))),
        _ => match common::body_load_vector(&load.kind, load.magnitude, &load.extra, sec.rho, name)? {
            Some(b) => Ok(Some(scale(b, sec.t))),
            None => Ok(None),
        },
    }
}

impl ElementType for ShellQuad {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Shell
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        4
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3, UR1, UR2, UR3]
    }
    fn n_ips(&self) -> usize {
        if self.kind == QuadKind::S4R { 1 } else { 4 }
    }
    fn vtk_type(&self) -> u8 {
        9
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        &[&[0, 1, 2, 3]]
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            let loc = section::frame(ctx, self.name)?;
            return Ok(section::Layup::new(ctx, self.name, &loc)?.init_state(self.n_ips()));
        }
        Ok(Vec::new())
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            // område struct_nonlin: tvärsnittets elastiska styvhet ur sektionspunkterna
            let kin = self.kin(ctx)?;
            return kin.stiffness(&section::Layup::new(ctx, self.name, &kin.loc)?);
        }
        let (loc, sec) = self.setup(ctx)?;
        Ok(to_global(&self.local_k(&loc.xl, &sec), &loc.r, 4))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if section::is_general(ctx) {
            // område struct_nonlin: integrering genom tjockleken (shell_section.rs)
            let kin = self.kin(ctx)?;
            let lay = section::Layup::new(ctx, self.name, &kin.loc)?;
            if ctx.nlgeom {
                return kin.response_nlgeom(ctx, self.name, &lay, u, _so, _sn, tangent);
            }
            return kin.response(ctx, &lay, u, _so, _sn, tangent);
        }
        if ctx.nlgeom {
            // område nlgeom_struct: korotationell formulering (shell_nlgeom.rs)
            let k0 = self.stiffness(ctx)?;
            return nlgeom::response(ctx, self.name, &k0, u, nlgeom::MemRule::Center, tangent);
        }
        let k = self.stiffness(ctx)?;
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        Ok(to_global(&quad_mass(&loc.xl, &sec, kind == MassKind::Lumped), &loc.r, 4))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        if section::is_general(ctx) {
            return not_implemented("geometrisk styvhet med integrering genom tjockleken", self.name);
        }
        let (loc, sec) = self.setup(ctx)?;
        let ul = Self::local_u(&loc, u, 4);
        let eps = self.strains(&loc.xl, &sec, &ul);
        // membrankrafter i mitten (medel av integrationspunkterna)
        let mut nm = [0.0; 3];
        for e in &eps {
            for i in 0..3 {
                nm[i] += (0..3).map(|j| sec.d[i * 3 + j] * e[j]).sum::<f64>() * sec.t / eps.len() as f64;
            }
        }
        if self.kind == QuadKind::S4R {
            // område nlgeom_struct: S4R integrerar spänningsstyvheten i en punkt
            return Ok(to_global(&nlgeom::geometric_s4r(&loc.xl, nm), &loc.r, 4));
        }
        Ok(to_global(&geometric_q4(&loc.xl, nm), &loc.r, 4))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let Some(q) = area_load_vector(load, loc.r[2], &sec, self.name)? else {
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        };
        let ql = [dot(loc.r[0], q), dot(loc.r[1], q), dot(loc.r[2], q)];
        let fl = quad_area_load(&loc.xl, ql);
        let mut f6 = vec![0.0; 24];
        for i in 0..4 {
            for d in 0..3 {
                f6[i * ND + d] = fl[3 * i + d];
            }
        }
        Ok(vec_to_global(&f6, &loc.r, 8))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let f = quad_area_load(&loc.xl, [sec.t, 0.0, 0.0]);
        Ok((0..4).map(|i| f[3 * i]).collect())
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        if section::is_general(ctx) {
            let kin = self.kin(ctx)?;
            let lay = section::Layup::new(ctx, self.name, &kin.loc)?;
            return kin.output(ctx, self.name, &lay, u, _state);
        }
        let (loc, sec) = self.setup(ctx)?;
        // NLGEOM: deformationella förskjutningar i det medföljande systemet
        let u_tot = u;
        let d;
        let u = if ctx.nlgeom {
            d = nlgeom::deformational_u(ctx, self.name, u)?;
            &d[..]
        } else {
            u
        };
        let ul = Self::local_u(&loc, u, 4);
        let mut eps = self.strains(&loc.xl, &sec, &ul);
        let a = 4.0 * q4pt(&loc.xl, 0.0, 0.0).det;
        if !ctx.nlgeom {
            return Ok(shell_output(&sec, &eps, sec.ks(a)));
        }
        // område nlgeom_struct: logaritmisk membrantöjning (mitten) i stället för
        // den linjära, och membrankrafter med aktuell tjocklek
        let p0 = q4pt(&loc.xl, 0.0, 0.0);
        let bm0 = bmem(&p0.bx, &p0.by);
        let lin: Vec<f64> = (0..3).map(|r| (0..24).map(|q| bm0[r * 24 + q] * ul[q]).sum()).collect();
        let el = nlgeom::membrane_strains(ctx, self.name, u_tot, nlgeom::MemRule::Center)?[0];
        for e in eps.iter_mut() {
            for i in 0..3 {
                e[i] += el[i] - lin[i];
            }
        }
        let mut out = shell_output(&sec, &eps, sec.ks(a));
        nlgeom::scale_membrane_forces(&mut out, el);
        Ok(out)
    }
}

/// Geometrisk styvhet för plattböjning: ∫ ∇wᵀ N ∇w dA (membrankrafter N),
/// samma för u och v (fullständig plan spänningsstyvhet).
pub fn geometric_q4(x: &[[f64; 2]], nm: [f64; 3]) -> Vec<f64> {
    let n = 24;
    let mut k = vec![0.0; n * n];
    for &(s, t) in &GP4 {
        let p = q4pt(x, s, t);
        for i in 0..4 {
            for j in 0..4 {
                let v = (p.bx[i] * (nm[0] * p.bx[j] + nm[2] * p.by[j]) + p.by[i] * (nm[2] * p.bx[j] + nm[1] * p.by[j])) * p.det;
                for d in 0..3 {
                    k[(i * ND + d) * n + j * ND + d] += v;
                }
            }
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Membran M3D4 och M3D3
// ---------------------------------------------------------------------------

pub struct Membrane {
    name: &'static str,
    nn: usize,
}

pub static M3D4: Membrane = Membrane { name: "M3D4", nn: 4 };
pub static M3D3: Membrane = Membrane { name: "M3D3", nn: 3 };

impl Membrane {
    /// Integrationspunkter (dN/dx, dN/dy, N, vikt) i lokala koordinater.
    fn ips(&self, x: &[[f64; 2]]) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
        if self.nn == 4 {
            GP4.iter().map(|&(s, t)| {
                let p = q4pt(x, s, t);
                (p.bx.to_vec(), p.by.to_vec(), p.n.to_vec(), p.det)
            }).collect()
        } else {
            let (bx, by, a) = tri::cst(x);
            vec![(bx.to_vec(), by.to_vec(), vec![1.0 / 3.0; 3], a)]
        }
    }

    fn setup(&self, ctx: &ElemCtx) -> Result<(Local, ShellSec)> {
        let loc = local_frame(&ctx.x, ctx.label, self.name)?;
        for (_, _, _, w) in self.ips(&loc.xl) {
            if w <= 0.0 {
                return fail(format!("element {} ({}): {}", ctx.label, self.name, common::DISTORTED));
            }
        }
        Ok((loc, ShellSec::get(ctx, self.name)?))
    }

    fn bm(bx: &[f64], by: &[f64]) -> Vec<f64> {
        let nn = bx.len();
        let n = 3 * nn;
        let mut b = vec![0.0; 3 * n];
        for i in 0..nn {
            b[3 * i] = bx[i];
            b[n + 3 * i + 1] = by[i];
            b[2 * n + 3 * i] = by[i];
            b[2 * n + 3 * i + 1] = bx[i];
        }
        b
    }
}

impl ElementType for Membrane {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Membrane
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &[U1, U2, U3]
    }
    fn n_ips(&self) -> usize {
        if self.nn == 4 { 4 } else { 1 }
    }
    fn vtk_type(&self) -> u8 {
        if self.nn == 4 { 9 } else { 5 }
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        if self.nn == 4 { &[&[0, 1, 2, 3]] } else { &[&[0, 1, 2]] }
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let n = 3 * self.nn;
        let mut k = vec![0.0; n * n];
        let mut c = [0.0; 9];
        for i in 0..9 {
            c[i] = sec.d[i] * sec.t;
        }
        for (bx, by, _, w) in self.ips(&loc.xl) {
            let b = Self::bm(&bx, &by);
            add_btcb(&mut k, n, &b, &c, &b, 3, w);
        }
        let t = common::block_diag(&loc.r, self.nn);
        Ok(common::congruence(&k, &t, n))
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let k = self.stiffness(ctx)?;
        if ctx.nlgeom {
            // område nlgeom_struct: korotationell formulering (shell_nlgeom.rs)
            let rule = if self.nn == 4 { nlgeom::MemRule::Gauss4 } else { nlgeom::MemRule::Tri };
            return nlgeom::response(ctx, self.name, &k, u, rule, tangent);
        }
        Ok(ElemResponse { fint: common::matvec(&k, u), k: tangent.then_some(k) })
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let n = 3 * self.nn;
        let mut m = vec![0.0; n * n];
        let mt = sec.rho * sec.t;
        if self.nn == 4 {
            for &(s, t) in &GP4 {
                let p = q4pt(&loc.xl, s, t);
                for i in 0..4 {
                    for j in 0..4 {
                        let f = p.n[i] * p.n[j] * p.det * mt;
                        let cj = if kind == MassKind::Lumped { i } else { j };
                        for d in 0..3 {
                            m[(3 * i + d) * n + 3 * cj + d] += f;
                        }
                    }
                }
            }
        } else {
            let (_, _, a) = tri::cst(&loc.xl);
            for i in 0..3 {
                for j in 0..3 {
                    let f = if kind == MassKind::Lumped { if i == j { a / 3.0 } else { 0.0 } } else if i == j { a / 6.0 } else { a / 12.0 };
                    for d in 0..3 {
                        m[(3 * i + d) * n + 3 * j + d] += f * mt;
                    }
                }
            }
        }
        let t = common::block_diag(&loc.r, self.nn);
        Ok(common::congruence(&m, &t, n))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let n = 3 * self.nn;
        let t = common::block_diag(&loc.r, self.nn);
        let ul = common::matvec(&t, u);
        let mut k = vec![0.0; n * n];
        for (bx, by, _, w) in self.ips(&loc.xl) {
            let b = Self::bm(&bx, &by);
            let e: Vec<f64> = (0..3).map(|r| (0..n).map(|q| b[r * n + q] * ul[q]).sum()).collect();
            let nm: Vec<f64> = (0..3).map(|i| (0..3).map(|j| sec.d[i * 3 + j] * e[j]).sum::<f64>() * sec.t).collect();
            for i in 0..self.nn {
                for j in 0..self.nn {
                    let v = (bx[i] * (nm[0] * bx[j] + nm[2] * by[j]) + by[i] * (nm[2] * bx[j] + nm[1] * by[j])) * w;
                    for d in 0..3 {
                        k[(3 * i + d) * n + 3 * j + d] += v;
                    }
                }
            }
        }
        Ok(common::congruence(&k, &t, n))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let Some(q) = area_load_vector(load, loc.r[2], &sec, self.name)? else {
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        };
        let ql = [dot(loc.r[0], q), dot(loc.r[1], q), dot(loc.r[2], q)];
        let mut f = vec![0.0; 3 * self.nn];
        if self.nn == 4 {
            f = quad_area_load(&loc.xl, ql);
        } else {
            let (_, _, a) = tri::cst(&loc.xl);
            for i in 0..3 {
                for d in 0..3 {
                    f[3 * i + d] = ql[d] * a / 3.0;
                }
            }
        }
        Ok(vec_to_global(&f, &loc.r, self.nn))
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (loc, sec) = self.setup(ctx)?;
        let mut v = vec![0.0; self.nn];
        for (_, _, n, w) in self.ips(&loc.xl) {
            for i in 0..self.nn {
                v[i] += n[i] * w * sec.t;
            }
        }
        Ok(v)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], _state: &[f64]) -> Result<Vec<IpField>> {
        let (loc, sec) = self.setup(ctx)?;
        if ctx.nlgeom {
            // område nlgeom_struct: logaritmisk töjning LE och spänning S = D·LE
            let rule = if self.nn == 4 { nlgeom::MemRule::Gauss4 } else { nlgeom::MemRule::Tri };
            let el = nlgeom::membrane_strains(ctx, self.name, u, rule)?;
            let mut s = IpField::new("S", &["S11", "S22", "S12"]);
            let mut log_e = IpField::new("LE", &["LE11", "LE22", "LE12"]);
            for (q, ev) in el.iter().enumerate() {
                let sv: Vec<f64> = (0..3).map(|i| (0..3).map(|j| sec.d[i * 3 + j] * ev[j]).sum()).collect();
                s.points.push((q + 1, 0, sv));
                log_e.points.push((q + 1, 0, ev.to_vec()));
            }
            return Ok(vec![s, log_e]);
        }
        let n = 3 * self.nn;
        let t = common::block_diag(&loc.r, self.nn);
        let ul = common::matvec(&t, u);
        let mut s = IpField::new("S", &["S11", "S22", "S12"]);
        let mut e = IpField::new("E", &["E11", "E22", "E12"]);
        for (q, (bx, by, _, _)) in self.ips(&loc.xl).into_iter().enumerate() {
            let b = Self::bm(&bx, &by);
            let ev: Vec<f64> = (0..3).map(|r| (0..n).map(|k| b[r * n + k] * ul[k]).sum()).collect();
            let sv: Vec<f64> = (0..3).map(|i| (0..3).map(|j| sec.d[i * 3 + j] * ev[j]).sum()).collect();
            s.points.push((q + 1, 0, sv));
            e.points.push((q + 1, 0, ev));
        }
        Ok(vec![s, e])
    }
}

pub static TYPES: &[&dyn ElementType] = &[&S4R, &S4, &M3D4, &M3D3, &tri::S3, &tri::S3R];
