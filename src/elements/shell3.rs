//! Tunna skal och axisymmetriska skal (område shells3):
//!
//! - `shell3_axi.rs`: axisymmetriska skal SAX1, SAX2 och membran MAX1, MAX2,
//! - `shell3_tri.rs`: STRI3 (plan fasett, diskret Kirchhoff),
//! - `shell3_q4.rs`: S4R5 (tunt fyrnodsskal, fem frihetsgrader per nod),
//! - `shell3_q9.rs`: S9R5 och S8R5 (tunna dubbelkrökta skal, fem
//!   frihetsgrader per nod; S8R5 har en intern mittnod).
//!
//! Formuleringarna beskrivs i respektive fil; se även `tests/shells3.rs`.
//!
//! Femfrihetsgradsskalen (S4R5, S8R5, S9R5) har två rotationer per
//! nod kring axlarna t1, t2 vinkelräta mot nodnormalen (t1 = projektionen av
//! den globala axel som ligger längst från normalen, t2 = n × t1). Här
//! assembleras de med sex globala frihetsgrader per nod: elementet ser bara
//! rotationens komponenter i nodens tangentplan, och rotationen kring
//! nodnormalen får en mycket liten styvhet (`DRILL_EPS`) så att systemet inte
//! blir singulärt. Förskjutningar och rotationer i tangentplanet påverkas
//! därmed bara på maskinprecisionsnivå när nodnormalerna i angränsande element
//! sammanfaller (plana nät).

#[path = "shell3_axi.rs"]
pub mod axi;
#[path = "shell3_q4.rs"]
pub mod q4;
#[path = "shell3_q9.rs"]
pub mod q9;
#[path = "shell3_tri.rs"]
pub mod tri;

use super::api::{ElemCtx, ElementType, IpField};
use super::common::{cross, dot, norm, scale, sub};
use super::shell2::Sec;
use crate::fem::model::SectionKind;
use crate::materials::StressMode;
use crate::model::{fail, Result};

pub static TYPES: &[&dyn ElementType] = &[&axi::SAX1, &axi::SAX2, &axi::MAX1, &axi::MAX2, &tri::STRI3, &q4::S4R5, &q9::S9R5, &q9::S8R5];

/// Relativ styvhet för rotationen kring nodnormalen i femfrihetsgradsskalen
/// (andel av elementets största rotationsdiagonal).
pub const DRILL_EPS: f64 = 1e-9;

pub fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// Skaltvärsnitt för elementen i området: *SHELL SECTION (via `shell2::Sec`)
/// eller *MEMBRANE SECTION (bara membranstyvhet).
pub fn section(ctx: &ElemCtx, name: &str, membrane: bool) -> Result<Sec> {
    if !membrane {
        return Sec::get(ctx, name);
    }
    let sec = ctx.section(name)?;
    let t = match &sec.kind {
        SectionKind::Shell { thickness, .. } => *thickness,
        _ if sec.raw.name == "MEMBRANE SECTION" => {
            let v = sec.raw.first_floats()?;
            v.first().copied().filter(|&t| t != 0.0).unwrap_or(1.0)
        }
        _ => return fail(format!("element {} ({name}) kräver *MEMBRANE SECTION", ctx.label)),
    };
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
    Ok(Sec { t, d, ks: [0.0; 3], g: d[8], rho: ctx.density, n_int: 1 })
}

/// Skjuvmodulen ur tvärsnittets plana spänningsmatris (isotropt material: G).
pub fn shear_modulus(sec: &Sec) -> f64 {
    sec.d[8]
}

/// Tvärskjuvstyvheten (K11) före slankhetskompensation.
pub fn k_shear(sec: &Sec) -> f64 {
    sec.ks[0]
}

/// Slankhetskompensationen 1/(1 + 0,25·10⁻⁴·a/t²).
pub fn fp(a: f64, t: f64) -> f64 {
    super::shell2::fp(a, t)
}

/// Nodbasen för femfrihetsgradsskalen: t1 = projektionen av den
/// globala axel som har minst komponent längs normalen (vid lika den med lägst
/// nummer), t2 = n × t1.
pub fn axes5(n: [f64; 3]) -> [[f64; 3]; 3] {
    let n = unit(n);
    let mut k = 0;
    for c in 1..3 {
        if n[c].abs() < n[k].abs() {
            k = c;
        }
    }
    let mut g = [0.0; 3];
    g[k] = 1.0;
    let t1 = unit(sub(g, scale(n, dot(g, n))));
    [t1, cross(n, t1), n]
}

/// Standardaxlar för skal (1 = projektionen av global X, eller Z om
/// normalen ligger inom 0,1° från X).
pub fn default_axes(n: [f64; 3]) -> [[f64; 3]; 3] {
    super::shell2::default_axes(n)
}

/// Lägger till den lilla rotationsstyvheten kring nodnormalerna `nrm` i en
/// matris med sex frihetsgrader per nod (rotationer på plats 3..6).
pub fn add_drill_eps(k: &mut [f64], nrm: &[[f64; 3]]) {
    let n = 6 * nrm.len();
    let mut mx = 0.0f64;
    for i in 0..nrm.len() {
        for d in 3..6 {
            mx = mx.max(k[(6 * i + d) * n + 6 * i + d].abs());
        }
    }
    let kd = DRILL_EPS * mx.max(f64::MIN_POSITIVE);
    for (i, ni) in nrm.iter().enumerate() {
        for a in 0..3 {
            for b in 0..3 {
                k[(6 * i + 3 + a) * n + 6 * i + 3 + b] += kd * ni[a] * ni[b];
            }
        }
    }
}

/// Projicerar en matris med sex frihetsgrader per nod så att rotationerna
/// bara verkar genom sina komponenter i nodernas tangentplan: K ← Pᵀ K P med
/// P = I − n nᵀ för rotationsblocken.
pub fn project_tangent(k: &[f64], nrm: &[[f64; 3]]) -> Vec<f64> {
    let nn = nrm.len();
    let n = 6 * nn;
    let mut p = vec![0.0; n * n];
    for i in 0..nn {
        for d in 0..3 {
            p[(6 * i + d) * n + 6 * i + d] = 1.0;
        }
        for a in 0..3 {
            for b in 0..3 {
                let v = if a == b { 1.0 } else { 0.0 } - nrm[i][a] * nrm[i][b];
                p[(6 * i + 3 + a) * n + 6 * i + 3 + b] = v;
            }
        }
    }
    super::common::congruence(k, &p, n)
}

/// Generaliserade töjningar i en integrationspunkt.
#[derive(Debug, Clone, Default)]
pub struct Gen {
    /// Membran (ε11, ε22, γ12).
    pub eps: [f64; 3],
    /// Krökning (κ11, κ22, 2κ12).
    pub kap: [f64; 3],
    /// Tvärskjuvkrafter (SF4, SF5).
    pub q: [f64; 2],
}

/// S i sektionspunkterna 1 och n, SF (SF1–SF6) och SM (SM1–SM3).
pub fn output(sec: &Sec, pts: &[Gen]) -> Vec<IpField> {
    let mut s = IpField::new("S", &["S11", "S22", "S12"]);
    let mut sf = IpField::new("SF", &["SF1", "SF2", "SF3", "SF4", "SF5", "SF6"]);
    let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
    let tb = sec.t.powi(3) / 12.0;
    for (q, g) in pts.iter().enumerate() {
        let ip = q + 1;
        let sn = sec.dv(g.eps);
        let mb = sec.dv(g.kap);
        sf.points.push((ip, 0, vec![sn[0] * sec.t, sn[1] * sec.t, sn[2] * sec.t, g.q[0], g.q[1], 0.0]));
        sm.points.push((ip, 0, vec![mb[0] * tb, mb[1] * tb, mb[2] * tb]));
        let sps = if sec.n_int > 1 { vec![1, sec.n_int] } else { vec![1] };
        for sp in sps {
            let z = sec.section_z(sp);
            let st = sec.dv([g.eps[0] + z * g.kap[0], g.eps[1] + z * g.kap[1], g.eps[2] + z * g.kap[2]]);
            s.points.push((ip, sp, st.to_vec()));
        }
    }
    vec![s, sf, sm]
}

/// y = B u för en rad b.
pub fn row_dot(b: &[f64], u: &[f64]) -> f64 {
    b.iter().zip(u).map(|(a, c)| a * c).sum()
}
