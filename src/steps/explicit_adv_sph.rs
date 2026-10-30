//! Område explicit_adv: SPH (smoothed particle hydrodynamics) med PC3D i
//! explicit analys.
//!
//! Partikeln i är en nod med massan mᵢ = ρ₀ (2L)³ (L = den karakteristiska
//! längden på `*SOLID SECTION`, halva kubens sida). Med kärnan W
//! och utjämningslängden h (stöd 2h) är
//!
//! ```text
//! Lᵢ   = Σⱼ (mⱼ/ρⱼ) (vⱼ − vᵢ) ⊗ ∇ᵢWᵢⱼ                      hastighetsgradient
//! Fᵢ  ← (I + Lᵢ Δt) Fᵢ,  ρᵢ = ρ₀ / det Fᵢ,  Δε = sym(Lᵢ) Δt
//! σᵢ  ← materialet(σᵢ + ΔW σᵢ − σᵢ ΔW, Δε, F)            (Jaumann, ΔW = skew(Lᵢ) Δt)
//! v̇ᵢ   = Σⱼ mⱼ (σᵢ/ρᵢ² + σⱼ/ρⱼ² − Πᵢⱼ I) ∇ᵢWᵢⱼ
//! Πᵢⱼ  = (−α c̄ μᵢⱼ + β μᵢⱼ²)/ρ̄ᵢⱼ om vᵢⱼ·rᵢⱼ < 0,  μᵢⱼ = h vᵢⱼ·rᵢⱼ/(rᵢⱼ² + 0,01h²)
//! ```
//!
//! (Libersky & Petschek 1993). Spänningen räknas med materialmodellen som för ett
//! reducerat integrerat element (en punkt per partikel). α och β är
//! volymviskositetens koefficienter (`*BULK VISCOSITY`, standard 0,06 och
//! 1,2) gånger skalfaktorerna på *SECTION CONTROLS första datarad (fält 4
//! och 5); `SPH BULK VISCOSITY=VORTICITY WEIGHTED` multiplicerar Πᵢⱼ med
//! Balsaras faktor ½(fᵢ + fⱼ), fᵢ = |∇·v|/(|∇·v| + |∇×v|). Den viskösa kraften
//! räknas som volymviskositet (ALLVD).
//!
//! Kärnor (`KERNEL=`): CUBIC (standard, kubisk spline, σ = 1/π), QUADRATIC
//! (Johnson m.fl. 1996, σ = 5/(4π)) och QUINTIC (Wendland C2
//! (1 − q/2)⁴(2q + 1) med normeringen σ = 7/8), alla med stödet 2h.
//!
//! Utjämningslängden är konstant (`SPH SMOOTHING LENGTH=CONSTANT`, standard)
//! eller följer partikelns volym, h = h₀ (ρ₀/ρ)^⅓ (VARIABLE). Standardvärdet
//! per tvärsnitt är h = 1,35 s (s = medelavståndet till närmaste partikel),
//! ökat med faktorn 10/9 tills partiklarna i medel har minst
//! min(`TARGET_NEIGHBORS`, N) partiklar inom 2h i analysens början. Regeln är
//! kontrollerad med (fast Δt, användarangiven h mot standard-h, kuber
//! med 27–729 partiklar och plattor: h = 1,35, 1,5, 1,667 och 1,852 gånger
//! avståndet); *SECTION CONTROLS andra datarad ger h direkt
//! (fält 1) eller en skalfaktor (fält 2) och det minsta antalet grannar
//! (fält 3, standard 5: färre grannar ger Lᵢ = 0, deformationsgradienten
//! hålls konstant).
//!
//! Partiklar i olika tvärsnitt påverkar inte varandra.
//!
//! Utdata per partikel: S (punkt 1), DENSITY (punkt 1), EVOL = m/ρ (hela
//! elementet) och materialets tillståndsutdata (PEEQ, ...).
//!
//! Stabilitetsgränsen (automatisk inkrementering) är Δt = `DT_FACTOR` · h/(c + |v|)
//! per partikel (faktorn är vald för en kub); testfallen
//! använder fast Δt.

use super::{section_controls, Eval, Sink};
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::parse_f64;
use crate::elements::IpField;
use crate::fem::output::FrameData;
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, FemError, Result};
use std::collections::HashMap;

/// Minsta medelantal partiklar inom stödet 2h för standardlängden (vald för
/// kuber och plattor av partiklar).
pub const TARGET_NEIGHBORS: f64 = 35.0;
/// Stabilitetsgränsens faktor Δt = DT_FACTOR · h/(c + |v|).
pub const DT_FACTOR: f64 = 0.3646;

/// Tillståndets platser per partikel.
const S: usize = 0;
const FG: usize = 6;
const EPS: usize = 15;
const RHO: usize = 21;
const HS: usize = 22;
const MAT: usize = 23;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kernel {
    Cubic,
    Quadratic,
    Quintic,
}

impl Kernel {
    /// (f(q), f′(q)) med W = σ/h³ f(q); noll för q ≥ 2.
    pub fn eval(self, q: f64) -> (f64, f64) {
        if q >= self.support() {
            return (0.0, 0.0);
        }
        match self {
            Kernel::Cubic => {
                let s = 1.0 / std::f64::consts::PI;
                if q <= 1.0 {
                    (s * (1.0 - 1.5 * q * q + 0.75 * q * q * q), s * (-3.0 * q + 2.25 * q * q))
                } else {
                    let a = 2.0 - q;
                    (s * 0.25 * a * a * a, -s * 0.75 * a * a)
                }
            }
            Kernel::Quadratic => {
                let s = 5.0 / (4.0 * std::f64::consts::PI);
                (s * (3.0 / 16.0 * q * q - 0.75 * q + 0.75), s * (0.375 * q - 0.75))
            }
            Kernel::Quintic => {
                // Wendland C2 med normeringen 7/8 (2π/3 gånger den
                // normerade 21/(16π))
                let s = 0.875;
                let a = 1.0 - 0.5 * q;
                (s * a.powi(4) * (2.0 * q + 1.0), -s * 5.0 * q * a.powi(3))
            }
        }
    }

    /// Stödets radie i enheter av h.
    pub fn support(self) -> f64 {
        2.0
    }

    /// W och ∂W/∂r för avståndet r och utjämningslängden h.
    pub fn w(self, r: f64, h: f64) -> (f64, f64) {
        let (f, fp) = self.eval(r / h);
        let h3 = h * h * h;
        (f / h3, fp / (h3 * h))
    }
}

/// Ett tvärsnitt med partiklar.
pub struct Group {
    pub kernel: Kernel,
    /// Utjämningslängd i analysens början.
    pub h0: f64,
    pub variable_h: bool,
    pub alpha: f64,
    pub beta: f64,
    pub vorticity: bool,
    pub min_neighbors: usize,
    /// Karakteristisk längd L (halva kubens sida).
    pub length: f64,
    pub rho0: f64,
    pub ns: usize,
}

pub struct Particle {
    pub e: usize,
    pub node: usize,
    pub group: usize,
    pub mass: f64,
}

pub struct Sph {
    pub groups: Vec<Group>,
    pub parts: Vec<Particle>,
    /// Partikelindex per element (usize::MAX = inte en partikel).
    pub of_elem: Vec<usize>,
    pub initial_energy: f64,
    /// Antal grannar per partikel i senaste kraftberäkningen.
    neighbors: Vec<usize>,
    /// Stabilitetsgränsen (Δt, element) från senaste kraftberäkningen.
    pub dt_est: Option<(f64, usize)>,
}

fn mat_of<'a>(p: &'a Prepared, e: usize) -> Result<&'a dyn MaterialModel> {
    let ctx = p.ctx(e, false, 0.0, 0.0);
    ctx.material.ok_or_else(|| FemError(format!("{}: PC3D saknar material (*SOLID SECTION, MATERIAL=)", p.model.elem_name(e))))
}

/// Grannar inom avståndet `r` för punkterna `x` (rutnät med sidan r).
pub fn neighbor_pairs(x: &[[f64; 3]], r: f64) -> Vec<(usize, usize, f64)> {
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    let key = |p: &[f64; 3]| ((p[0] / r).floor() as i64, (p[1] / r).floor() as i64, (p[2] / r).floor() as i64);
    for (i, p) in x.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let r2 = r * r;
    let mut out = Vec::new();
    for (i, p) in x.iter().enumerate() {
        let (a, b, c) = key(p);
        for da in -1..=1 {
            for db in -1..=1 {
                for dc in -1..=1 {
                    let Some(cell) = grid.get(&(a + da, b + db, c + dc)) else { continue };
                    for &j in cell {
                        if j <= i {
                            continue;
                        }
                        let d = [x[j][0] - p[0], x[j][1] - p[1], x[j][2] - p[2]];
                        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if d2 < r2 {
                            out.push((i, j, d2.sqrt()));
                        }
                    }
                }
            }
        }
    }
    out
}

/// Standardvärde för utjämningslängden: h = 1,35 s (s = punkternas
/// medelavstånd till närmaste granne), ökad med faktorn 10/9 tills punkterna i
/// medel har minst min(`target`, N) punkter (sig själv medräknad) närmare än 2h.
/// Se modulens dokumentation.
pub fn default_smoothing_length(x: &[[f64; 3]], target: f64) -> f64 {
    let n = x.len();
    if n < 2 {
        return 1.0;
    }
    let nf = n as f64;
    // medelavståndet till närmaste granne (rutnät med växande sida)
    let mut nearest = vec![f64::INFINITY; n];
    let ext = (0..3).map(|c| x.iter().map(|p| p[c]).fold(f64::NEG_INFINITY, f64::max) - x.iter().map(|p| p[c]).fold(f64::INFINITY, f64::min)).fold(0.0f64, f64::max);
    let mut r = (ext.max(1e-300) / nf.cbrt()).max(1e-300);
    for _ in 0..60 {
        for (i, j, d) in neighbor_pairs(x, r) {
            nearest[i] = nearest[i].min(d);
            nearest[j] = nearest[j].min(d);
        }
        if nearest.iter().all(|d| d.is_finite()) {
            break;
        }
        r *= 2.0;
    }
    let s = nearest.iter().filter(|d| d.is_finite()).sum::<f64>() / nf;
    let goal = target.min(nf);
    let count = |h: f64| -> f64 { (nf + 2.0 * neighbor_pairs(x, 2.0 * h * (1.0 - 1e-6)).len() as f64) / nf };
    let mut h = 1.35 * s;
    for _ in 0..60 {
        if count(h) >= goal - 1e-9 {
            break;
        }
        h /= 0.9;
    }
    h
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn det3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Spänningen (ordning 11, 22, 33, 12, 13, 23) som tensor.
pub fn tensor(s: &[f64]) -> [[f64; 3]; 3] {
    [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]]
}

/// Ljudhastigheten √((K + 4G/3)/ρ) för materialet i tillståndet `ms`.
pub fn sound_speed(mat: &dyn MaterialModel, ms: &[f64], rho: f64) -> f64 {
    let d = match mat.tangent_linear(StressMode::ThreeD) {
        Ok(d) if d.len() == 36 => d,
        _ => return 0.0,
    };
    let g = 0.5 * (d[0] - d[1]);
    let k = mat.current_bulk_modulus(ms).unwrap_or(d[1] + 2.0 * g / 3.0);
    ((k + 4.0 * g / 3.0).max(0.0) / rho.max(1e-300)).sqrt()
}

impl Sph {
    pub fn build(p: &Prepared, b1: f64, b2: f64) -> Result<Option<Sph>> {
        let m = p.model;
        let mut groups: Vec<Group> = Vec::new();
        let mut group_of_sec: HashMap<usize, usize> = HashMap::new();
        let mut parts = Vec::new();
        let mut of_elem = vec![usize::MAX; m.elements.len()];
        for (e, el) in m.elements.iter().enumerate() {
            if el.etype.name() != "PC3D" {
                continue;
            }
            let Some(si) = el.section else { return fail(format!("{} saknar tvärsnitt (*SOLID SECTION)", m.elem_name(e))) };
            let g = match group_of_sec.get(&si) {
                Some(&g) => g,
                None => {
                    let sec = &m.sections[si];
                    let length = sec.raw.first_floats()?.first().copied().unwrap_or(0.0);
                    if !(length > 0.0) {
                        return sec.raw.err("PC3D: ange den karakteristiska längden (> 0) på dataraden");
                    }
                    let mat = mat_of(p, e)?;
                    let rho0 = p.ctx(e, false, 0.0, 0.0).density;
                    if !(rho0 > 0.0) {
                        return sec.raw.err("PC3D: materialet saknar *DENSITY");
                    }
                    let mut grp = Group {
                        kernel: Kernel::Cubic,
                        h0: 0.0,
                        variable_h: false,
                        alpha: b1,
                        beta: b2,
                        vorticity: false,
                        min_neighbors: 5,
                        length,
                        rho0,
                        ns: mat.n_state(),
                    };
                    let mut user_h = None;
                    let mut hfac = 1.0;
                    if let Some(sc) = section_controls(p, sec)? {
                        grp.kernel = match sc.param_upper("KERNEL").as_deref() {
                            None | Some("CUBIC") => Kernel::Cubic,
                            Some("QUADRATIC") => Kernel::Quadratic,
                            Some("QUINTIC") => Kernel::Quintic,
                            Some(o) => return sc.err(format!("KERNEL={o} stöds inte (CUBIC, QUADRATIC, QUINTIC)")),
                        };
                        match sc.param_upper("SPH FORMULATION").as_deref() {
                            None | Some("CLASSICAL") => {}
                            Some(o) => return sc.err(format!("SPH FORMULATION={o} stöds inte ännu (bara CLASSICAL)")),
                        }
                        grp.variable_h = match sc.param_upper("SPH SMOOTHING LENGTH").as_deref() {
                            None | Some("CONSTANT") => false,
                            Some("VARIABLE") => true,
                            Some(o) => return sc.err(format!("SPH SMOOTHING LENGTH={o} stöds inte (CONSTANT, VARIABLE)")),
                        };
                        grp.vorticity = match sc.param_upper("SPH BULK VISCOSITY").as_deref() {
                            None | Some("UNIFORM") => false,
                            Some("VORTICITY WEIGHTED") => true,
                            Some(o) => return sc.err(format!("SPH BULK VISCOSITY={o} stöds inte (UNIFORM, VORTICITY WEIGHTED)")),
                        };
                        if sc.param_upper("SPH TENSILE INSTABILITY CONTROL").as_deref() == Some("YES") {
                            return sc.err("SPH TENSILE INSTABILITY CONTROL=YES stöds inte ännu");
                        }
                        let field = |row: usize, k: usize| -> Option<f64> {
                            sc.data.get(row).and_then(|d| d.fields.get(k)).filter(|f| !f.trim().is_empty()).and_then(|f| parse_f64(f))
                        };
                        if let Some(s1) = field(0, 3) {
                            grp.alpha = b1 * s1;
                        }
                        if let Some(s2) = field(0, 4) {
                            grp.beta = b2 * s2;
                        }
                        user_h = field(1, 0).filter(|&h| h > 0.0);
                        if let Some(f) = field(1, 1).filter(|&f| f > 0.0) {
                            hfac = f;
                        }
                        if let Some(n) = field(1, 2).filter(|&n| n > 0.0) {
                            grp.min_neighbors = n as usize;
                        }
                    }
                    // standardlängden bestäms när alla partiklar i tvärsnittet är kända
                    grp.h0 = user_h.map_or(-hfac, |h| h * hfac);
                    groups.push(grp);
                    group_of_sec.insert(si, groups.len() - 1);
                    groups.len() - 1
                }
            };
            let grp = &groups[g];
            let mass = grp.rho0 * (2.0 * grp.length).powi(3);
            of_elem[e] = parts.len();
            parts.push(Particle { e, node: el.nodes[0], group: g, mass });
        }
        if parts.is_empty() {
            return Ok(None);
        }
        // omvandlade element: partikelns egen längd (explicit_adv_conversion.rs)
        let conv = super::conversion::lengths(m);
        for q in parts.iter_mut() {
            if let Some(&l) = conv.get(&q.e) {
                q.mass = groups[q.group].rho0 * (2.0 * l).powi(3);
            }
        }
        for (g, grp) in groups.iter_mut().enumerate() {
            if grp.h0 < 0.0 {
                let x: Vec<[f64; 3]> = parts.iter().filter(|q| q.group == g).map(|q| m.nodes[q.node]).collect();
                grp.h0 = -grp.h0 * default_smoothing_length(&x, TARGET_NEIGHBORS);
                if std::env::var("RUSTYFEM_DEBUG_SPH").is_ok() {
                    eprintln!("SPH: tvärsnitt {g}: h = {:.6e} ({} partiklar)", grp.h0, x.len());
                }
            }
        }
        let n = parts.len();
        let mut sph = Sph { groups, parts, of_elem, initial_energy: 0.0, neighbors: vec![0; n], dt_est: None };
        sph.initial_energy = sph.initial_energy_of(p)?;
        Ok(Some(sph))
    }

    /// Begynnelseenergin Σ m Eₘ₀ (*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY).
    fn initial_energy_of(&self, p: &Prepared) -> Result<f64> {
        let em = specific_energies(p)?;
        Ok(self.parts.iter().map(|q| em.get(&q.e).copied().unwrap_or(0.0) * q.mass).sum())
    }

    pub fn patch_mass(&self, mass: &mut super::LumpedMass) {
        for q in &self.parts {
            for s in 0..3 {
                mass.diag[q.node][s] += q.mass;
            }
        }
    }

    /// Partikelns tillstånd i analysens början.
    fn init_state(&self, p: &Prepared, i: usize, em: Option<f64>) -> Result<Vec<f64>> {
        let q = &self.parts[i];
        let g = &self.groups[q.group];
        let mat = mat_of(p, q.e)?;
        let mut st = vec![0.0; MAT + g.ns];
        for k in 0..3 {
            st[FG + 4 * k] = 1.0;
        }
        st[RHO] = g.rho0;
        st[HS] = g.h0;
        mat.init_state(&mut st[MAT..]);
        if let Some((p0, _)) = mat.init_energy(&mut st[MAT..], em, None, 0.0).map_err(|err| FemError(format!("{}: {}", p.model.elem_name(q.e), err.0)))? {
            for k in 0..3 {
                st[S + k] = -p0;
            }
        }
        Ok(st)
    }

    pub fn forces(&mut self, p: &Prepared, ev: &Eval, out: &mut Sink) -> Result<()> {
        let m = p.model;
        let n = self.parts.len();
        self.dt_est = None;
        let em = if self.parts.iter().any(|q| ev.state_old[q.e].is_empty()) { specific_energies(p)? } else { HashMap::new() };
        // tillståndet vid inkrementets början
        let mut old: Vec<Vec<f64>> = Vec::with_capacity(n);
        for (i, q) in self.parts.iter().enumerate() {
            let st = &ev.state_old[q.e];
            old.push(if st.is_empty() { self.init_state(p, i, em.get(&q.e).copied())? } else { st.clone() });
        }
        let x: Vec<[f64; 3]> = self.parts.iter().map(|q| std::array::from_fn(|c| m.nodes[q.node][c] + ev.u[q.node][c])).collect();
        let v: Vec<[f64; 3]> = self.parts.iter().map(|q| std::array::from_fn(|c| ev.v[q.node][c])).collect();
        let hmax = old.iter().map(|s| s[HS]).fold(0.0f64, f64::max);
        let smax = self.groups.iter().map(|g| g.kernel.support()).fold(2.0f64, f64::max);
        let pairs: Vec<(usize, usize, f64)> = neighbor_pairs(&x, smax * hmax)
            .into_iter()
            .filter(|&(i, j, r)| self.parts[i].group == self.parts[j].group && r < 0.5 * self.groups[self.parts[i].group].kernel.support() * (old[i][HS] + old[j][HS]))
            .collect();
        let hij = |i: usize, j: usize, st: &[Vec<f64>]| 0.5 * (st[i][HS] + st[j][HS]);
        // hastighetsgradienten med tätheterna vid inkrementets början
        let mut lg = vec![[[0.0f64; 3]; 3]; n];
        let mut count = vec![0usize; n];
        for &(i, j, r) in &pairs {
            let g = &self.groups[self.parts[i].group];
            let h = hij(i, j, &old);
            if r <= 0.0 || r >= g.kernel.support() * h {
                continue;
            }
            let (_, dw) = g.kernel.w(r, h);
            let rij: [f64; 3] = std::array::from_fn(|c| x[i][c] - x[j][c]);
            let grad: [f64; 3] = std::array::from_fn(|c| dw * rij[c] / r);
            let dv: [f64; 3] = std::array::from_fn(|c| v[j][c] - v[i][c]);
            let (vi, vj) = (self.parts[i].mass / old[i][RHO], self.parts[j].mass / old[j][RHO]);
            for a in 0..3 {
                for b in 0..3 {
                    // ∇ⱼWⱼᵢ = −∇ᵢWᵢⱼ och (vᵢ − vⱼ) = −dv
                    lg[i][a][b] += vj * dv[a] * grad[b];
                    lg[j][a][b] += vi * dv[a] * grad[b];
                }
            }
            count[i] += 1;
            count[j] += 1;
        }
        // spänningsuppdatering per partikel
        let mut new: Vec<Vec<f64>> = old.clone();
        let mut balsara = vec![1.0; n];
        for i in 0..n {
            let q = &self.parts[i];
            let g = &self.groups[q.group];
            if count[i] < g.min_neighbors {
                lg[i] = [[0.0; 3]; 3];
            }
            let l = lg[i];
            let div = l[0][0] + l[1][1] + l[2][2];
            let curl = [l[2][1] - l[1][2], l[0][2] - l[2][0], l[1][0] - l[0][1]];
            let cn = (curl[0] * curl[0] + curl[1] * curl[1] + curl[2] * curl[2]).sqrt();
            balsara[i] = if div.abs() > 1e-12 * (div.abs() + cn + 1e-300) && div.abs() + cn > 0.0 { div.abs() / (div.abs() + cn) } else { 0.0 };
            if ev.dt <= 0.0 {
                continue;
            }
            let dt = ev.dt;
            let st = &old[i];
            let f0: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| st[FG + 3 * a + b]));
            let inc: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| if a == b { 1.0 } else { 0.0 } + l[a][b] * dt));
            let f1 = mat3_mul(&inc, &f0);
            let d: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (l[a][b] + l[b][a]) * dt));
            let w: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (l[a][b] - l[b][a]) * dt));
            let deps = [d[0][0], d[1][1], d[2][2], 2.0 * d[0][1], 2.0 * d[0][2], 2.0 * d[1][2]];
            let s0 = tensor(&st[S..S + 6]);
            // Jaumann: σ + ΔW σ − σ ΔW
            let ws = mat3_mul(&w, &s0);
            let sw = mat3_mul(&s0, &w);
            let sr: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| s0[a][b] + ws[a][b] - sw[a][b]));
            let srot = [sr[0][0], sr[1][1], sr[2][2], sr[0][1], sr[0][2], sr[1][2]];
            let eps1: Vec<f64> = (0..6).map(|k| st[EPS + k] + deps[k]).collect();
            let mat = mat_of(p, q.e)?;
            let ns = g.ns;
            let mut ms_new = vec![0.0; ns];
            let input = MatInput { mode: StressMode::ThreeD, strain: &eps1, dstrain: &deps, stress_old: &srot, time: ev.time, dt, temperature: 0.0, dtemp: 0.0, defgrad: Some(f1) };
            let r = mat.update(&input, &st[MAT..MAT + ns], &mut ms_new).map_err(|err| FemError(format!("{}: {}", m.elem_name(q.e), err.0)))?;
            let ns_ = &mut new[i];
            ns_[S..S + 6].copy_from_slice(&r.stress[..6]);
            for a in 0..3 {
                for b in 0..3 {
                    ns_[FG + 3 * a + b] = f1[a][b];
                }
            }
            ns_[EPS..EPS + 6].copy_from_slice(&eps1);
            let j = det3(&f1);
            ns_[RHO] = g.rho0 / j.max(1e-12);
            if g.variable_h {
                ns_[HS] = g.h0 * j.max(1e-12).cbrt();
            }
            ns_[MAT..MAT + ns].copy_from_slice(&ms_new);
        }
        // krafter med tillståndet vid inkrementets slut
        let mut fi = vec![[0.0f64; 3]; n];
        let mut fv = vec![[0.0f64; 3]; n];
        let cs: Vec<f64> = (0..n)
            .map(|i| {
                let q = &self.parts[i];
                match mat_of(p, q.e) {
                    Ok(mat) => sound_speed(mat, &new[i][MAT..], new[i][RHO]),
                    Err(_) => 0.0,
                }
            })
            .collect();
        for &(i, j, r) in &pairs {
            let g = &self.groups[self.parts[i].group];
            let h = hij(i, j, &new);
            if r <= 0.0 || r >= g.kernel.support() * h {
                continue;
            }
            let (_, dw) = g.kernel.w(r, h);
            let rij: [f64; 3] = std::array::from_fn(|c| x[i][c] - x[j][c]);
            let grad: [f64; 3] = std::array::from_fn(|c| dw * rij[c] / r);
            let (mi, mj) = (self.parts[i].mass, self.parts[j].mass);
            let (ri, rj) = (new[i][RHO], new[j][RHO]);
            let (si, sj) = (tensor(&new[i][S..S + 6]), tensor(&new[j][S..S + 6]));
            // mᵢ v̇ᵢ = mᵢ Σ mⱼ (σᵢ/ρᵢ² + σⱼ/ρⱼ²) ∇ᵢWᵢⱼ; inre kraften är −mᵢ v̇ᵢ
            for a in 0..3 {
                let t: f64 = (0..3).map(|b| (si[a][b] / (ri * ri) + sj[a][b] / (rj * rj)) * grad[b]).sum();
                let f = mi * mj * t;
                fi[i][a] -= f;
                fi[j][a] += f;
            }
            // artificiell viskositet
            let vij: [f64; 3] = std::array::from_fn(|c| v[i][c] - v[j][c]);
            let vr = vij[0] * rij[0] + vij[1] * rij[1] + vij[2] * rij[2];
            if vr < 0.0 && (g.alpha != 0.0 || g.beta != 0.0) {
                let mu = h * vr / (r * r + 0.01 * h * h);
                let cbar = 0.5 * (cs[i] + cs[j]);
                let rbar = 0.5 * (ri + rj);
                // den kvadratiska termen med β² (som volymviskositetens b₂²)
                let mut pi = (-g.alpha * cbar * mu + g.beta * g.beta * mu * mu) / rbar;
                if g.vorticity {
                    pi *= 0.5 * (balsara[i] + balsara[j]);
                }
                for a in 0..3 {
                    let f = mi * mj * pi * grad[a];
                    fv[i][a] += f;
                    fv[j][a] -= f;
                }
            }
        }
        for (i, q) in self.parts.iter().enumerate() {
            for a in 0..3 {
                out.fint[q.node][a] += fi[i][a];
                out.fvis[q.node][a] += fv[i][a];
            }
            let speed = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
            if cs[i] > 0.0 {
                let dt = DT_FACTOR * new[i][HS] / (cs[i] + speed);
                if self.dt_est.is_none_or(|b| dt < b.0) {
                    self.dt_est = Some((dt, q.e));
                }
            }
            out.state_new[q.e] = std::mem::take(&mut new[i]);
        }
        self.neighbors = count;
        Ok(())
    }

    pub fn output(&self, p: &Prepared, fr: &mut FrameData, _u: &[NodeVec], state: &[Vec<f64>]) -> Result<()> {
        let em = if self.parts.iter().any(|q| state[q.e].is_empty()) { specific_energies(p)? } else { HashMap::new() };
        for (i, q) in self.parts.iter().enumerate() {
            // stegets början: begynnelsetillståndet
            let init;
            let st = if state[q.e].is_empty() {
                init = self.init_state(p, i, em.get(&q.e).copied())?;
                &init
            } else {
                &state[q.e]
            };
            let g = &self.groups[q.group];
            let mat = mat_of(p, q.e)?;
            let mut s = IpField::new("S", &["S11", "S22", "S33", "S12", "S13", "S23"]);
            s.points.push((1, 0, st[S..S + 6].to_vec()));
            let mut d = IpField::new("DENSITY", &["DENSITY"]);
            d.points.push((1, 0, vec![st[RHO]]));
            let mut ev = IpField::new("EVOL", &["EVOL"]);
            ev.points.push((0, 0, vec![q.mass / st[RHO]]));
            let mut fields = vec![s, d, ev];
            let names = mat.output_names();
            if !names.is_empty() {
                let vals = mat.output_values(&st[MAT..MAT + g.ns]);
                for (k, nm) in names.iter().enumerate() {
                    if let Some(&val) = vals.get(k) {
                        let mut f = IpField::new(nm, &[nm]);
                        f.points.push((1, 0, vec![val]));
                        fields.push(f);
                    }
                }
            }
            let _ = i;
            if let Some(slot) = fr.elem.get_mut(q.e) {
                *slot = fields;
            }
        }
        Ok(())
    }
}

/// Specifik begynnelseenergi per element (*INITIAL CONDITIONS, TYPE=SPECIFIC ENERGY).
pub fn specific_energies(p: &Prepared) -> Result<HashMap<usize, f64>> {
    let mut em = HashMap::new();
    for kw in p.model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(crate::materials::eos::RESOLVED)) {
        for d in &kw.data {
            let v = kw.floats(d)?;
            if let (Some(&e), Some(&x)) = (v.first(), v.get(1)) {
                em.insert(e as usize, x);
            }
        }
    }
    Ok(em)
}
