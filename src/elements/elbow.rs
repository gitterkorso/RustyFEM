//! Rörkrökar med deformerbart tvärsnitt (område beams3): ELBOW31, ELBOW32,
//! ELBOW31B och ELBOW31C med `*BEAM SECTION, SECTION=ELBOW`.
//!
//! Elementen är skal runt röret: balkaxelns förskjutning och rotation
//! interpoleras linjärt (ELBOW31/31B/31C) eller kvadratiskt (ELBOW32) längs
//! axeln och tvärsnittets ovalisering, välvning och normalens lutning med
//! Fourierserier runt röret (se `elbow_kin.rs` för kinematiken).
//!
//! - Integrering: en station längs axeln (ELBOW31/31B/31C) eller
//!   tvåpunkts Gauss (ELBOW32), `nc` (minst 8) likformigt fördelade punkter
//!   runt röret med punkt 1 vid intradosen och Simpsons regel genom väggen
//!   (plan spänning i varje punkt).
//! - Tvärskjuvningen γ_t = n · x,S bestraffas med (5/6)·G·t·f per punkt,
//!   f = 1/(1 + 0,25·10⁻⁴·A/t²) med punktens area A; med noll moder bara i
//!   punkt 1. ELBOW31B/C saknar γ₃ och straffet.
//! - Ovaliseringsfrihetsgraderna (interna noder) ligger på genererade
//!   noder med nummer −(6·10¹² + 16·nod + k) (ELBOW31/32, delas av elementen i
//!   noden, så att ovaliseringen är kontinuerlig) respektive
//!   −(7·10¹² + 16·element + k) (ELBOW31B/C, per element), sex frihetsgrader
//!   (1–6) per genererad nod. `setup` skapar dem när filen är läst.
//! - Massan: per frihetsgrad en massa per längdenhet (translationer
//!   ρt·2πr₀, rotationer ρtπr₀³, ovaliseringen ρtr₀∫ψ²dφ, γ₃ ingen) gånger
//!   nodkvadratur längs axeln (L/2 per nod) för tvånodselementen och
//!   konsistent ∫HₘHₙ dS för ELBOW32, utan kopplingar mellan komponenterna.
//! - Laster: PI/PE kräver den effektiva diametern D och verkar på radien
//!   D/2 (mantelyta och slutna ändar, i jämvikt för krökta element), GRAV och
//!   BX/BY/BZ bara på axelns translationer.
//! - `*BOUNDARY` NOWARP (välvning och normalens lutning), NOOVAL (u_r och u_t
//!   för p ≥ 1) och NODEFORM (allt) verkar på de genererade noderna.
//! - Utdata: S och E (S11 längs röret, S22 runt, S12) i inner- och
//!   ytterytan i varje punkt runt röret (sektionspunkt (k − 1)·nt + j),
//!   SF1 och SM1 (kring a2), SM2 (kring a1, krökens plan), SM3 (vridning)
//!   integrerade ur spänningarna, samt materialets utdata (PEEQ, ...).
//!
//! Plasticitet och andra materialmodeller med plan spänning fungerar
//! (geometriskt linjärt); NLGEOM, temperaturfält genom väggen, *MPC ELBOW och
//! HPI/HPE är inte implementerade.

#[path = "elbow_kin.rs"]
pub mod kin;

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::{DistLoad, FeModel, RawKeyword};
use crate::materials::{MatInput, StressMode};
use crate::model::{fail, Result};
use kin::{Comp, Geo, Sec, Variant};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const DOF6: [Dof; 6] = [U1, U2, U3, UR1, UR2, UR3];

/// Markörer för *BOUNDARY NOWARP, NOOVAL och NODEFORM (ersätts av `setup`).
pub const NOWARP: Dof = 201;
pub const NOOVAL: Dof = 202;
pub const NODEFORM: Dof = 203;

/// Nummerbaser för de genererade noderna.
const NODE_BASE: i64 = 6_000_000_000_000;
const ELEM_BASE: i64 = 7_000_000_000_000;

/// Elementtypen som den står i *ELEMENT (innan `setup` vet antalet moder).
pub struct ElbowBase {
    name: &'static str,
    nn: usize,
    var: Variant,
}

pub static ELBOW31: ElbowBase = ElbowBase { name: "ELBOW31", nn: 2, var: Variant::Full };
pub static ELBOW32: ElbowBase = ElbowBase { name: "ELBOW32", nn: 3, var: Variant::Full };
pub static ELBOW31B: ElbowBase = ElbowBase { name: "ELBOW31B", nn: 2, var: Variant::B };
pub static ELBOW31C: ElbowBase = ElbowBase { name: "ELBOW31C", nn: 2, var: Variant::C };

pub static TYPES: &[&dyn ElementType] = &[&ELBOW31, &ELBOW32, &ELBOW31B, &ELBOW31C];

fn base_of(name: &str) -> Option<&'static ElbowBase> {
    [&ELBOW31, &ELBOW32, &ELBOW31B, &ELBOW31C].into_iter().find(|b| b.name == name)
}

/// Läser *BEAM SECTION, SECTION=ELBOW: r, t, R; orienteringspunkt;
/// punkter genom väggen, punkter runt röret, antal moder.
pub fn section(raw: &RawKeyword) -> Result<Sec> {
    let recs = raw.records();
    let line = |i: usize| -> Result<Vec<f64>> {
        match recs.get(i) {
            Some(d) => {
                let mut v = Vec::new();
                for s in &d.fields {
                    let s = s.trim();
                    v.push(if s.is_empty() { 0.0 } else { crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{} rad {}: '{s}' är inget tal", raw.file, d.line)))? });
                }
                Ok(v)
            }
            None => Ok(Vec::new()),
        }
    };
    let l0 = line(0)?;
    if l0.len() < 2 || !(l0[0] > 0.0) || !(l0[1] > 0.0) || l0[1] >= 2.0 * l0[0] {
        return raw.err("SECTION=ELBOW: första dataraden ska vara ytterradie r, väggtjocklek t (0 < t < 2r) och torusradie R");
    }
    let l1 = line(1)?;
    if l1.len() < 3 {
        return raw.err("SECTION=ELBOW: andra dataraden ska ange orienteringspunktens koordinater");
    }
    let l2 = line(2)?;
    let nt = l2.first().copied().filter(|&v| v > 0.0).unwrap_or(5.0) as usize;
    let nc = l2.get(1).copied().filter(|&v| v > 0.0).unwrap_or(20.0) as usize;
    let nmod = l2.get(2).copied().unwrap_or(0.0).max(0.0) as usize;
    if nt % 2 == 0 {
        return raw.err("SECTION=ELBOW: antalet integrationspunkter genom väggen ska vara udda");
    }
    if nmod > 6 {
        return raw.err("SECTION=ELBOW: högst 6 ovaliseringsmoder");
    }
    Ok(Sec { r: l0[0], t: l0[1], big_r: l0.get(2).copied().unwrap_or(0.0).max(0.0), opt: [l1[0], l1[1], l1[2]], nt, nc: nc.max(8), nmod })
}

/// En konfigurerad elementtyp (variant, antal moder, punkter) med de
/// genererade noderna sist i nodlistan.
pub struct Elbow {
    name: &'static str,
    nn: usize,
    var: Variant,
    nmod: usize,
    nc: usize,
    nt: usize,
    /// Frihetsgrader per genererad nod (i elementets ordning).
    gen: Vec<usize>,
}

/// Antalet genererade noder (sex frihetsgrader var) för `n` frihetsgrader.
fn chunks(n: usize) -> Vec<usize> {
    (0..n.div_ceil(6)).map(|i| (n - 6 * i).min(6)).collect()
}

fn configured(base: &ElbowBase, sec: &Sec) -> &'static Elbow {
    static CACHE: OnceLock<Mutex<HashMap<(&'static str, usize, usize, usize), &'static Elbow>>> = OnceLock::new();
    let mut c = CACHE.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    c.entry((base.name, sec.nmod, sec.nc, sec.nt))
        .or_insert_with(|| {
            let nf = kin::fields(base.var, sec.nmod).len();
            let gen = if base.var == Variant::Full { (0..base.nn).flat_map(|_| chunks(nf)).collect() } else { chunks(nf) };
            Box::leak(Box::new(Elbow { name: base.name, nn: base.nn, var: base.var, nmod: sec.nmod, nc: sec.nc, nt: sec.nt, gen }))
        })
}

/// Skapar ovaliseringsnoderna och byter elementtyp för ELBOW-elementen samt
/// ersätter NOWARP/NOOVAL/NODEFORM med randvillkor på de genererade noderna
/// (anropas när filen är läst, efter `FeModel::finalize`).
pub fn setup(m: &mut FeModel) -> Result<()> {
    // största antalet frihetsgrader per användarnod (ELBOW31/32)
    let mut need: HashMap<usize, usize> = HashMap::new();
    let mut todo = Vec::new();
    for e in 0..m.elements.len() {
        let Some(base) = base_of(m.elements[e].etype.name()) else { continue };
        if m.elements[e].nodes.len() != base.nn {
            continue;
        }
        let Some(si) = m.elements[e].section else {
            return fail(format!("element {} ({}) saknar *BEAM SECTION, SECTION=ELBOW", m.elements[e].label, base.name));
        };
        let raw = &m.sections[si].raw;
        if !raw.param_upper("SECTION").is_some_and(|s| s == "ELBOW") {
            return raw.err(format!("{} kräver *BEAM SECTION, SECTION=ELBOW", base.name));
        }
        let sec = section(raw)?;
        let nf = kin::fields(base.var, sec.nmod).len();
        if base.var == Variant::Full {
            for &n in &m.elements[e].nodes {
                let v = need.entry(n).or_insert(0);
                *v = (*v).max(nf);
            }
        }
        todo.push((e, base, sec));
    }
    let mut gen_nodes: HashMap<usize, Vec<usize>> = HashMap::new();
    for (e, base, sec) in todo {
        let ty = configured(base, &sec);
        let inst = m.elements[e].inst;
        let mut extra = Vec::new();
        if base.var == Variant::Full {
            let nf = kin::fields(base.var, sec.nmod).len();
            for i in 0..base.nn {
                let n = m.elements[e].nodes[i];
                let list = gen_nodes.entry(n).or_insert_with(Vec::new);
                if list.is_empty() {
                    for k in 0..need[&n].div_ceil(6) {
                        let label = -(NODE_BASE + 16 * m.node_label[n] + k as i64);
                        let x = m.nodes[n];
                        let id = m.node(inst, label).unwrap_or_else(|| m.add_node(inst, label, x));
                        list.push(id);
                    }
                }
                extra.extend(list[..nf.div_ceil(6)].iter().copied());
            }
        } else {
            let x = m.nodes[m.elements[e].nodes[0]];
            for k in 0..ty.gen.len() {
                let label = -(ELEM_BASE + 16 * m.elements[e].label + k as i64);
                let id = m.node(inst, label).unwrap_or_else(|| m.add_node(inst, label, x));
                extra.push(id);
            }
        }
        m.elements[e].nodes.extend(extra);
        m.elements[e].etype = ty;
    }
    // randvillkor på tvärsnittets deformation
    let max_nf = kin::fields(Variant::Full, 6).len();
    let all_fields = kin::fields(Variant::Full, 6);
    let expand = |items: &mut Vec<(usize, Dof, f64)>, warnings: &mut Vec<String>| {
        let mut out = Vec::with_capacity(items.len());
        for &(n, d, v) in items.iter() {
            if !(NOWARP..=NODEFORM).contains(&d) {
                out.push((n, d, v));
                continue;
            }
            let Some(list) = gen_nodes.get(&n) else {
                warnings.push(format!("NOWARP/NOOVAL/NODEFORM på nod {} ignoreras: noden hör inte till ELBOW31/ELBOW32", n));
                continue;
            };
            let nf = need[&n].min(max_nf);
            for (k, f) in all_fields.iter().enumerate().take(nf) {
                // se tests/cases/beams3/elbow31_krok_frekvens*:
                // NOWARP låser välvningen och normalens lutning, NOOVAL u_r och u_t
                // för p ≥ 1, NODEFORM allt
                let hit = match d {
                    NOWARP => matches!(f.comp, Comp::Y3 | Comp::G3),
                    NOOVAL => f.p >= 1 && matches!(f.comp, Comp::Ur | Comp::Ut),
                    _ => true,
                };
                if hit {
                    out.push((list[k / 6], DOF6[k % 6], 0.0));
                }
            }
        }
        *items = out;
    };
    let mut warnings = Vec::new();
    for b in m.initial_bcs.iter_mut() {
        expand(&mut b.items, &mut warnings);
    }
    for s in m.steps.iter_mut() {
        for b in s.bcs.iter_mut() {
            expand(&mut b.items, &mut warnings);
        }
    }
    warnings.sort();
    warnings.dedup();
    m.warnings.extend(warnings);
    Ok(())
}

impl ElementType for ElbowBase {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, _: usize) -> &'static [Dof] {
        &DOF6
    }
    fn vtk_type(&self) -> u8 {
        if self.nn == 2 { 3 } else { 21 }
    }
    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        fail(format!("intern: {} är inte förberett (elbow::setup)", self.name))
    }
}

/// Materialdata i elementet.
struct Mat<'a> {
    model: &'a dyn crate::materials::MaterialModel,
    g: f64,
}

impl Elbow {
    fn geo(&self, ctx: &ElemCtx) -> Result<(Geo, Sec)> {
        let sec = section(&ctx.section(self.name)?.raw)?;
        let g = Geo::new(&ctx.x[..self.nn], &sec, self.var, ctx.label)?;
        Ok((g, sec))
    }

    fn mat<'a>(&self, ctx: &ElemCtx<'a>) -> Result<Mat<'a>> {
        let model = ctx.material(self.name)?;
        let Some((e, nu)) = model.elastic_moduli() else {
            return not_implemented(&format!("materialet {} (kräver elasticitet med E och ν för tvärskjuvningen)", model.name()), self.name);
        };
        Ok(Mat { model, g: e / (2.0 * (1.0 + nu)) })
    }

    fn n_points(&self) -> usize {
        (if self.nn == 2 { 1 } else { 2 }) * self.nc * self.nt
    }

    /// Simpsons regel genom väggen: (h, vikt).
    fn thickness(&self, t: f64) -> Vec<(f64, f64)> {
        let n = self.nt;
        if n == 1 {
            return vec![(0.0, t)];
        }
        let dh = t / (n - 1) as f64;
        (0..n)
            .map(|i| {
                let w = if i == 0 || i == n - 1 { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
                (-t / 2.0 + dh * i as f64, w * dh / 3.0)
            })
            .collect()
    }

    /// Tvärskjuvningen bestraffas i punkt k?
    fn shear_at(&self, k: usize) -> bool {
        self.var == Variant::Full && (self.nmod > 0 || k == 0)
    }

    fn stride(&self, mat: &Mat) -> usize {
        3 + mat.model.n_state()
    }

    /// Nodtemperaturen (medelvärde av användarnoderna) vid inkrementets slut och ändringen.
    fn temperature(&self, ctx: &ElemCtx) -> (f64, f64) {
        if ctx.temp.len() < self.nn || ctx.temp_old.len() < self.nn {
            return (0.0, 0.0);
        }
        let t: f64 = ctx.temp[..self.nn].iter().sum::<f64>() / self.nn as f64;
        let t0: f64 = ctx.temp_old[..self.nn].iter().sum::<f64>() / self.nn as f64;
        (t, t - t0)
    }

    /// Inre kraft, tangent och nytt tillstånd.
    fn respond(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.nlgeom {
            return not_implemented("NLGEOM", self.name);
        }
        let (geo, sec) = self.geo(ctx)?;
        let mat = self.mat(ctx)?;
        let nd = geo.nd;
        if u.len() != nd {
            return fail(format!("intern: {} har {} frihetsgrader, väntade {nd}", self.name, u.len()));
        }
        let stride = self.stride(&mat);
        let np = self.n_points();
        if so.len() != np * stride {
            return fail(format!("intern: {} har fel tillståndslängd", self.name));
        }
        sn.resize(np * stride, 0.0);
        let (temperature, dtemp) = self.temperature(ctx);
        let thick = self.thickness(sec.t);
        let mut fint = vec![0.0; nd];
        let mut k = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
        let wphi = 2.0 * std::f64::consts::PI / self.nc as f64;
        let mut q = 0;
        for (s, ws) in geo.stations() {
            for kc in 0..self.nc {
                let phi = Geo::phi(self.nc, kc);
                let pt = geo.point(s, phi);
                let area = ws * wphi * pt.j;
                for &(h, wh) in &thick {
                    let bh: [Vec<f64>; 3] = [0, 1, 2].map(|c| pt.be[c].iter().zip(&pt.bk[c]).map(|(a, b)| a + h * b).collect());
                    let eps: Vec<f64> = (0..3).map(|c| dot_v(&bh[c], u)).collect();
                    let deps: Vec<f64> = (0..3).map(|c| dot_v(&bh[c], du)).collect();
                    let base = q * stride;
                    let (sold, mold) = so[base..base + stride].split_at(3);
                    let mut mnew = mold.to_vec();
                    let out = mat.model.update(
                        &MatInput { mode: StressMode::PlaneStress, strain: &eps, dstrain: &deps, stress_old: sold, time: ctx.time, dt: ctx.dt, temperature, dtemp, defgrad: None },
                        mold,
                        &mut mnew,
                    )?;
                    sn[base..base + 3].copy_from_slice(&out.stress[..3]);
                    sn[base + 3..base + stride].copy_from_slice(&mnew);
                    let w = area * wh;
                    for c in 0..3 {
                        let sc = w * out.stress[c];
                        if sc != 0.0 {
                            for (f, b) in fint.iter_mut().zip(&bh[c]) {
                                *f += sc * b;
                            }
                        }
                    }
                    if tangent {
                        // K += w Bᵀ D B
                        let mut db = vec![vec![0.0; nd]; 3];
                        for a in 0..3 {
                            for c in 0..3 {
                                let d = out.tangent[a * 3 + c];
                                if d != 0.0 {
                                    for (x, b) in db[a].iter_mut().zip(&bh[c]) {
                                        *x += w * d * b;
                                    }
                                }
                            }
                        }
                        for a in 0..3 {
                            for i in 0..nd {
                                let bi = bh[a][i];
                                if bi != 0.0 {
                                    let row = &mut k[i * nd..(i + 1) * nd];
                                    for (x, y) in row.iter_mut().zip(&db[a]) {
                                        *x += bi * y;
                                    }
                                }
                            }
                        }
                    }
                    q += 1;
                }
                if self.shear_at(kc) {
                    let f = 1.0 / (1.0 + 0.25e-4 * area / (sec.t * sec.t));
                    let c = 5.0 / 6.0 * mat.g * sec.t * f * area;
                    let g = dot_v(&pt.gt, u);
                    for (x, b) in fint.iter_mut().zip(&pt.gt) {
                        *x += c * g * b;
                    }
                    if tangent {
                        for i in 0..nd {
                            let bi = c * pt.gt[i];
                            if bi != 0.0 {
                                for (x, y) in k[i * nd..(i + 1) * nd].iter_mut().zip(&pt.gt) {
                                    *x += bi * y;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    /// Massmatrisen: per frihetsgrad en massa per längdenhet (translationer
    /// ρt·2πr₀, rotationer ρtπr₀³, ovalisering ρtr₀∫ψ²dφ, γ₃ ingen) gånger
    /// axelns massmatris: nodkvadratur (L/2 per nod) för tvånodselementen,
    /// konsistent ∫HₘHₙ dS för ELBOW32 och L för ELBOW31B/C:s elementvisa
    /// ovalisering. Inga kopplingar mellan olika frihetsgrader i en nod.
    fn mass_matrix(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let (geo, sec) = self.geo(ctx)?;
        let rho = ctx.density;
        let r0 = geo.r0;
        let l = geo.l;
        let nd = geo.nd;
        let mut m = vec![0.0; nd * nd];
        let axis: Vec<Vec<f64>> = if self.nn == 2 {
            vec![vec![l / 2.0, 0.0], vec![0.0, l / 2.0]]
        } else {
            let c = l / 30.0;
            vec![vec![4.0 * c, 2.0 * c, -c], vec![2.0 * c, 16.0 * c, 2.0 * c], vec![-c, 2.0 * c, 4.0 * c]]
        };
        let pi = std::f64::consts::PI;
        let mt = rho * sec.t * 2.0 * pi * r0;
        let mr = rho * sec.t * pi * r0.powi(3);
        for a in 0..self.nn {
            for b in 0..self.nn {
                for c in 0..3 {
                    m[(6 * a + c) * nd + 6 * b + c] = mt * axis[a][b];
                    m[(6 * a + 3 + c) * nd + 6 * b + 3 + c] = mr * axis[a][b];
                }
            }
        }
        let oval = |f: &kin::Fld| -> f64 {
            if f.comp == Comp::G3 {
                return 0.0;
            }
            rho * sec.t * r0 * if f.p == 0 { 2.0 * pi } else { pi }
        };
        for (k, f) in geo.fields.iter().enumerate() {
            if geo.per_elem {
                let i = geo.oval_index(0, k);
                m[i * nd + i] = oval(f) * l;
            } else {
                for a in 0..self.nn {
                    for b in 0..self.nn {
                        m[geo.oval_index(a, k) * nd + geo.oval_index(b, k)] = oval(f) * axis[a][b];
                    }
                }
            }
        }
        Ok(m)
    }

    /// Spänningar och töjningar per punkt (station, punkt runt röret, punkt genom väggen).
    fn point_values(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<([f64; 3], [f64; 3])>> {
        let mat = self.mat(ctx)?;
        let stride = self.stride(&mat);
        let (geo, sec) = self.geo(ctx)?;
        let np = self.n_points();
        let mut out = Vec::with_capacity(np);
        let thick = self.thickness(sec.t);
        let mut q = 0;
        for (s, _) in geo.stations() {
            for kc in 0..self.nc {
                let pt = geo.point(s, Geo::phi(self.nc, kc));
                for &(h, _) in &thick {
                    let eps = [0, 1, 2].map(|c| pt.be[c].iter().zip(&pt.bk[c]).zip(u).map(|((a, b), x)| (a + h * b) * x).sum::<f64>());
                    let sig = if state.len() == np * stride { [state[q * stride], state[q * stride + 1], state[q * stride + 2]] } else { [0.0; 3] };
                    out.push((sig, eps));
                    q += 1;
                }
            }
        }
        Ok(out)
    }
}

/// Gauss–Legendres punkter och vikter på [−1, 1] (Newton på Legendrepolynomet).
fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
    let n = n.max(1);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut x = -(std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            // P_n(x) och derivatan med rekursionen
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=n {
                let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                p0 = p1;
                p1 = p2;
            }
            let pn = if n == 1 { x } else { p1 };
            let pm = if n == 1 { 1.0 } else { p0 };
            dp = n as f64 * (x * pn - pm) / (x * x - 1.0);
            let dx = pn / dp;
            x -= dx;
            if dx.abs() < 1e-16 {
                break;
            }
        }
        out.push((x, 2.0 / ((1.0 - x * x) * dp * dp)));
    }
    out
}

fn dot_v(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl ElementType for Elbow {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        Family::Beam
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        self.nn + self.gen.len()
    }
    fn n_connectivity(&self) -> usize {
        self.nn
    }
    fn node_dofs(&self, i: usize) -> &'static [Dof] {
        if i < self.nn {
            &DOF6
        } else {
            &DOF6[..self.gen[i - self.nn]]
        }
    }
    fn n_ips(&self) -> usize {
        self.n_points()
    }
    fn vtk_type(&self) -> u8 {
        0
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mat = self.mat(ctx)?;
        let stride = self.stride(&mat);
        let mut s = vec![0.0; self.n_points() * stride];
        for q in 0..self.n_points() {
            mat.model.init_state(&mut s[q * stride + 3..(q + 1) * stride]);
        }
        Ok(s)
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let so = self.init_state(ctx)?;
        let nd = self.n_dofs();
        let mut sn = Vec::new();
        let z = vec![0.0; nd];
        let r = self.respond(ctx, &z, &z, &so, &mut sn, true)?;
        Ok(r.k.unwrap_or_default())
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        self.respond(ctx, u, du, so, sn, tangent)
    }

    fn mass(&self, ctx: &ElemCtx, _kind: MassKind) -> Result<Vec<f64>> {
        // samma massa för konsistent och diagonal
        self.mass_matrix(ctx)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, _u: &[f64]) -> Result<Vec<f64>> {
        let (geo, sec) = self.geo(ctx)?;
        let nd = geo.nd;
        let mut f = vec![0.0; nd];
        let r0 = geo.r0;
        let wphi = 2.0 * std::f64::consts::PI / self.nc as f64;
        let pressure = match load.kind.as_str() {
            "PI" => Some(load.magnitude),
            "PE" => Some(-load.magnitude),
            _ => None,
        };
        if let Some(p) = pressure {
            // den effektiva diametern D (fjärde fältet) krävs: trycket
            // verkar på en yta med radien ρ = D/2 runt axeln, mantelytan med
            // p ρ (1 + ρκ sin φ)(R·δx̄ + δu_r) och ändytorna (slutet rör) med
            // p πρ² längs bågens tangent i noderna, så att elementets last är i
            // jämvikt
            let Some(diam) = load.extra.first().copied().filter(|d| *d > 0.0) else {
                return fail(format!("{}: tryck {} kräver den effektiva diametern (fjärde fältet på *DLOAD-raden)", self.name, load.kind));
            };
            let rho = diam / 2.0;
            // mantelytan integreras längs den krökta axeln med riktarna som
            // vrids längs bågen (sexpunkts Gauss; med stationerna blir
            // lasten inte i jämvikt i krökta element)
            let [ma1, ma2, ma3] = geo.frame(geo.l / 2.0);
            for (xi, wx) in gauss_legendre(6) {
                let s = geo.l / 2.0 * (1.0 + xi);
                let ws = wx * geo.l / 2.0;
                let (h, _) = geo.shape(xi);
                // bågens riktare i S (vriden κ(S − L/2) kring A1 från mitten)
                let be = geo.kap * (s - geo.l / 2.0);
                let a2 = [0, 1, 2].map(|c| ma2[c] * be.cos() + ma3[c] * be.sin());
                for kc in 0..self.nc {
                    let phi = Geo::phi(self.nc, kc);
                    let rv = [0, 1, 2].map(|c| ma1[c] * phi.cos() + a2[c] * phi.sin());
                    let lamp = 1.0 + rho * geo.kap * phi.sin();
                    let w = p * rho * lamp * ws * wphi;
                    for n in 0..self.nn {
                        for c in 0..3 {
                            f[6 * n + c] += w * h[n] * rv[c];
                        }
                    }
                    let nodes = if geo.per_elem { 1 } else { self.nn };
                    for n in 0..nodes {
                        let hn = if geo.per_elem { 1.0 } else { h[n] };
                        for (k, fl) in geo.fields.iter().enumerate() {
                            if fl.comp == Comp::Ur {
                                f[geo.oval_index(n, k)] += w * hn * kin::trig(fl.trig, fl.p, phi).0;
                            }
                        }
                    }
                }
            }
            let cap = p * std::f64::consts::PI * rho * rho;
            let [_, a2, a3] = geo.frame(geo.l / 2.0);
            let half = geo.kap * geo.l / 2.0;
            let t0 = [0, 1, 2].map(|c| a3[c] * half.cos() + a2[c] * half.sin());
            let t1 = [0, 1, 2].map(|c| a3[c] * half.cos() - a2[c] * half.sin());
            let last = self.nn - 1;
            for c in 0..3 {
                f[c] -= cap * t0[c];
                f[6 * last + c] += cap * t1[c];
            }
            return Ok(f);
        }
        let Some(b) = super::common::body_load_vector(&load.kind, load.magnitude, &load.extra, ctx.density, self.name)? else {
            return not_implemented(&format!("lasttypen {}", load.kind), self.name);
        };
        // volymkraft: bara axelns translationer, t·2πr₀·∫Hₙ dS gånger kraften per volym
        // (väggens moment och ovaliseringen får ingen last)
        for (s, ws) in geo.stations() {
            let xi = 2.0 * s / geo.l - 1.0;
            let (h, _) = geo.shape(xi);
            let w = sec.t * 2.0 * std::f64::consts::PI * r0 * ws;
            for n in 0..self.nn {
                for c in 0..3 {
                    f[6 * n + c] += w * h[n] * b[c];
                }
            }
        }
        Ok(f)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let (geo, sec) = self.geo(ctx)?;
        let vals = self.point_values(ctx, u, state)?;
        let thick = self.thickness(sec.t);
        let mut s = IpField::new("S", &["S11", "S22", "S12"]);
        let mut e = IpField::new("E", &["E11", "E22", "E12"]);
        let mut sf = IpField::new("SF", &["SF1"]);
        let mut sm = IpField::new("SM", &["SM1", "SM2", "SM3"]);
        // materialets utdata (PEEQ, ...) i samma punkter som S
        let mat = self.mat(ctx)?;
        let stride = self.stride(&mat);
        let names = mat.model.output_names();
        let mut extra: Vec<IpField> = names.iter().map(|n| IpField::new(n, &[n])).collect();
        let r0 = geo.r0;
        let wphi = 2.0 * std::f64::consts::PI / self.nc as f64;
        let mut q = 0;
        for (ist, _) in geo.stations().iter().enumerate() {
            let (mut n1, mut m1, mut m2, mut m3) = (0.0, 0.0, 0.0, 0.0);
            for kc in 0..self.nc {
                let phi = Geo::phi(self.nc, kc);
                let (sn, cs) = phi.sin_cos();
                let (mut n11, mut n12, mut mm11) = (0.0, 0.0, 0.0);
                for (j, &(h, wh)) in thick.iter().enumerate() {
                    let (sig, eps) = vals[q];
                    n11 += wh * sig[0];
                    n12 += wh * sig[2];
                    mm11 += wh * h * sig[0];
                    if j == 0 || j == self.nt - 1 {
                        let sp = kc * self.nt + j + 1;
                        s.points.push((ist + 1, sp, sig.to_vec()));
                        e.points.push((ist + 1, sp, eps.to_vec()));
                        if !names.is_empty() && state.len() >= (q + 1) * stride {
                            let v = mat.model.output_values(&state[q * stride + 3..(q + 1) * stride]);
                            for (fld, x) in extra.iter_mut().zip(v) {
                                fld.points.push((ist + 1, sp, vec![x]));
                            }
                        }
                    }
                    q += 1;
                }
                let _ = mm11;
                n1 += wphi * r0 * n11;
                m1 += wphi * r0 * r0 * n11 * sn;
                m2 += -wphi * r0 * r0 * n11 * cs;
                m3 += wphi * r0 * r0 * n12;
            }
            sf.points.push((ist + 1, 0, vec![n1]));
            sm.points.push((ist + 1, 0, vec![m2, m1, m3]));
        }
        let mut out = vec![s, e, sf, sm];
        out.extend(extra);
        Ok(out)
    }
}
