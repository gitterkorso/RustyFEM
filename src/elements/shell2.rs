//! Andra ordningens skal och kontinuumskal (område shells2): S8R
//! (`shell2_s8r.rs`), SC8R och SC6R (`shell2_sc.rs`), NLGEOM för dem
//! (`shell2_nlgeom.rs`).
//!
//! Den här filen innehåller det som elementen delar: tvärsnittet (homogent
//! *SHELL SECTION med isotropt eller anisotropt linjärelastiskt material),
//! lokala axlar (standardaxlar eller tvärsnittets ORIENTATION=), små
//! matrishjälpare och utdata (S, SF, SM).
//!
//! Lokala axlar: 1-axeln är projektionen av global X på skalets tangentplan
//! (global Z om normalen ligger inom 0,1° från X), 2 = n × 1. Med
//! *SHELL SECTION, ORIENTATION= är 1-axeln projektionen av orienteringens
//! 1-axel på tangentplanet (orienteringens 3-axel om 1-axeln ligger inom 0,1°
//! från normalen), där orienteringens axlar först vrids med den extra
//! rotationen på *ORIENTATION:s andra datarad (se
//! tests/cases/shells2/s8r_platta_orientering).

#[path = "shell2_nlgeom.rs"]
pub mod nlgeom;
#[path = "shell2_s8r.rs"]
pub mod s8r;
#[path = "shell2_sc.rs"]
pub mod sc;

use super::api::{ElemCtx, ElementType, IpField};
use super::common::{cross, dot, norm, scale, sub};
use crate::fem::model::SectionKind;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub static TYPES: &[&dyn ElementType] = &[&s8r::S8R, &sc::SC8R, &sc::SC6R];

/// Elementtyper som stöder *SHELL SECTION, ORIENTATION= (läses av
/// `inp/keywords/section.rs`).
pub const ORIENTATION_TYPES: &[&str] = &["S8R", "SC6R", "SC8R"];

/// Slankhetskompensationen för tvärskjuvstyvheten, f = 1/(1 + 0,25·10⁻⁴·a/t²)
/// (a = arean som hör till integrationspunkten eller noden).
pub fn fp(a: f64, t: f64) -> f64 {
    1.0 / (1.0 + 0.25e-4 * a / (t * t))
}

pub fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Homogent skaltvärsnitt för S8R/SC8R/SC6R.
#[derive(Debug, Clone)]
pub struct Sec {
    pub t: f64,
    /// Plan spänning D (3×3, radvis) i de lokala axlarna (orienteringen).
    pub d: [f64; 9],
    /// Tvärskjuvstyvhet [K11, K22, K12] (kraft per längd, före slankhetskompensation).
    pub ks: [f64; 3],
    /// Skjuvmodul för tjockleksmodulen (kontinuumskal: E_t = 2G).
    pub g: f64,
    pub rho: f64,
    /// Antal sektionspunkter för utdata (Simpson), 3 för *SHELL GENERAL SECTION.
    pub n_int: usize,
}

impl Sec {
    pub fn get(ctx: &ElemCtx, name: &str) -> Result<Sec> {
        let sec = ctx.section(name)?;
        let (t, n_int) = match &sec.kind {
            SectionKind::Shell { thickness, n_int } => (*thickness, *n_int),
            _ => return fail(format!("element {} ({name}) kräver *SHELL SECTION (homogent)", ctx.label)),
        };
        if !(t > 0.0) {
            return fail(format!("element {} ({name}): skaltjockleken måste vara positiv", ctx.label));
        }
        if let Some(o) = sec.raw.param_upper("OFFSET") {
            if !matches!(o.as_str(), "MIDDLE" | "0" | "0." | "0.0") {
                return fail(format!("element {} ({name}): *SHELL SECTION, OFFSET={o} stöds inte (bara mittytan)", ctx.label));
            }
        }
        let mat = ctx.material(name)?;
        if !mat.is_linear() {
            return super::api::not_implemented(&format!("olinjärt material ({})", mat.name()), name);
        }
        let dv = mat.tangent_linear(StressMode::PlaneStress)?;
        if dv.len() != 9 {
            return fail(format!("{name}: materialet ger ingen plan spänningsmatris"));
        }
        let mut d = [0.0; 9];
        d.copy_from_slice(&dv);
        // skjuvmoduler: isotropt G, annars G13 och G23 ur den tredimensionella matrisen
        let (g, g13, g23) = match mat.elastic_moduli() {
            Some((e, nu)) => {
                let g = e / (2.0 * (1.0 + nu));
                (g, g, g)
            }
            None => match mat.tangent_linear(StressMode::ThreeD) {
                Ok(c) if c.len() == 36 => (d[8], c[4 * 6 + 4], c[5 * 6 + 5]),
                _ => (d[8], 0.0, 0.0),
            },
        };
        let tss = sec.raw.param(super::beam_timoshenko::section::TSS_PARAM).map(|s| {
            let v: Vec<f64> = s.split(',').map(|x| crate::fem::model::parse_f64(x).unwrap_or(0.0)).collect();
            [v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)]
        });
        let ks = match tss {
            Some(k) => k,
            None if g13 > 0.0 && g23 > 0.0 => [5.0 / 6.0 * g13 * t, 5.0 / 6.0 * g23 * t, 0.0],
            None => {
                return fail(format!(
                    "element {} ({name}): materialet ger inga tvärskjuvmoduler (G13, G23) – ange *TRANSVERSE SHEAR STIFFNESS",
                    ctx.label
                ))
            }
        };
        let n_int = if sec.raw.name == "SHELL GENERAL SECTION" { 3 } else { n_int };
        Ok(Sec { t, d, ks, g, rho: ctx.density, n_int })
    }

    /// Tvärskjuvningens matris [[K11, K12], [K12, K22]] med kompensationen för arean `a`.
    pub fn shear_c(&self, a: f64) -> [f64; 4] {
        let f = fp(a, self.t);
        [self.ks[0] * f, self.ks[2] * f, self.ks[2] * f, self.ks[1] * f]
    }

    /// Sektionspunktens z (Simpson, n punkter genom tjockleken).
    pub fn section_z(&self, sp: usize) -> f64 {
        let n = self.n_int.max(1);
        if n == 1 {
            return 0.0;
        }
        -self.t / 2.0 + self.t * (sp as f64 - 1.0) / (n as f64 - 1.0)
    }

    pub fn dv(&self, e: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i| (0..3).map(|j| self.d[i * 3 + j] * e[j]).sum())
    }
}

/// Standardaxlar för ett skal med normalen `n` (rader e1, e2, n).
pub fn default_axes(n: [f64; 3]) -> [[f64; 3]; 3] {
    let n = unit(n);
    let g = if n[0].abs() > (0.1f64.to_radians()).cos() { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] };
    let e1 = unit(sub(g, scale(n, dot(g, n))));
    [e1, cross(n, e1), n]
}

/// Lokala axlar (rader e1, e2, n) i punkten `p` med normalen `n`, med
/// tvärsnittets orientering om den finns.
pub fn axes(ctx: &ElemCtx, n: [f64; 3], p: [f64; 3]) -> [[f64; 3]; 3] {
    let Some(o) = ctx.orientation else { return default_axes(n) };
    let n = unit(n);
    // orienteringens axlar inklusive den extra rotationen (i rummet), sedan
    // projektion på tangentplanet
    let ax = o.axes_at(p);
    let g = if dot(ax[0], n).abs() > (0.1f64.to_radians()).cos() { ax[2] } else { ax[0] };
    let e1 = unit(sub(g, scale(n, dot(g, n))));
    [e1, cross(n, e1), n]
}

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

/// y = B u för B med `r` rader.
pub fn rows_dot(b: &[f64], r: usize, u: &[f64]) -> Vec<f64> {
    let n = u.len();
    (0..r).map(|i| (0..n).map(|q| b[i * n + q] * u[q]).sum()).collect()
}

// ---------------------------------------------------------------------------
// Utdata
// ---------------------------------------------------------------------------

/// Generaliserade töjningar i en integrationspunkt: membran ε (ε11, ε22, γ12),
/// krökning κ, tvärskjuvning γ (γ13, γ23), tjockleksspänning·t (SF6) och
/// kompenserad skjuvmatris.
pub struct GenStrain {
    pub eps: [f64; 3],
    pub kap: [f64; 3],
    pub gam: [f64; 2],
    pub shear_c: [f64; 4],
    pub sf6: f64,
}

/// S i sektionspunkterna 1 och n, SF (SF1–SF6) och SM (SM1–SM3).
/// `s33` = true ger S med komponenterna S11, S22, S33, S12 (kontinuumskal).
pub fn output(sec: &Sec, pts: &[GenStrain], s33: bool) -> Vec<IpField> {
    let mut s = if s33 { IpField::new("S", &["S11", "S22", "S33", "S12"]) } else { IpField::new("S", &["S11", "S22", "S12"]) };
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let tb = sec.t.powi(3) / 12.0;
    for (q, g) in pts.iter().enumerate() {
        let ip = q + 1;
        let sn = sec.dv(g.eps);
        let mb = sec.dv(g.kap);
        let c = &g.shear_c;
        let q1 = c[0] * g.gam[0] + c[1] * g.gam[1];
        let q2 = c[2] * g.gam[0] + c[3] * g.gam[1];
        sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, sn[2] * sec.t, q1, q2, g.sf6]));
        sm.points.push((ip, 0, vec![mb[0] * tb, mb[1] * tb, mb[2] * tb]));
        if sec.n_int > 0 {
            let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
            for sp in sps {
                let z = sec.section_z(sp);
                let st = sec.dv([g.eps[0] + z * g.kap[0], g.eps[1] + z * g.kap[1], g.eps[2] + z * g.kap[2]]);
                let v = if s33 { vec![st[0], st[1], 0.0, st[2]] } else { st.to_vec() };
                s.points.push((ip, sp, v));
            }
        }
    }
    let mut out = vec![sf, sm];
    if !s.points.is_empty() {
        out.insert(0, s);
    }
    out
}
