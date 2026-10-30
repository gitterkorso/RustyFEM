//! Akustiska och kopplade akustisk–strukturella steg: *FREQUENCY,
//! *STEADY STATE DYNAMICS, DIRECT och *DYNAMIC (implicit) för modeller med
//! akustiska element (`elements/acoustic.rs`). Stegtypen tar över de vanliga
//! stegen (`claim_model`) när modellen har akustiska element.
//!
//! Ekvationer (kopplad akustik–struktur, Zienkiewicz–Newton), med S kopplingsmatrisen från *TIE
//! (`interactions/acoustic.rs`, fluidens utåtriktade normal):
//!
//! ```text
//!   [M_s 0  ] [ü]   [C_s 0  ] [u̇]   [K_s −Sᵀ ] [u]   [F]
//!   [S   M_f] [p̈] + [0   C_f] [ṗ] + [0    K_f] [p] = [Q]
//! ```
//!
//! - M_f = ∫ N N/K_f dV, K_f = ∫ ∇N·∇N/ρ_f dV, Q = inåtriktad volymacceleration
//!   (*CLOAD på frihetsgrad 8).
//! - Impedans (*IMPEDANCE PROPERTY med *SIMPEDANCE/*IMPEDANCE): randtermen
//!   (iΩ/c₁ − Ω²/k₁) ∫N N dS i frekvensdomänen, dvs. dämpning ∫N N/c₁ och
//!   massa ∫N N/k₁ i tidsdomänen. DATA=IMPEDANCE: varje rad görs om till
//!   admittansen 1/Z vid sin frekvens, och admittansen interpoleras.
//! - Icke-reflekterande rand (NONREFLECTING=PLANAR/CIRCULAR/SPHERICAL):
//!   n⁻·∇p = f (i k̃ + β) p med k̃ = Ω √(ρ̃/K_f), f = 1 och β = 0, 1/(2r₁),
//!   1/r₁; i frekvensdomänen randtermen (1/ρ̃)(i k̃ + β) ∫N N dS, i tidsdomänen
//!   dämpning ∫N N/√(ρK) och styvhet ∫N N (β/ρ + γ/(2ρ√(ρK))) dS.
//! - Volymetriskt motstånd γ: i frekvensdomänen komplex densitet
//!   ρ̃ = ρ_f + γ/(iΩ) i K_f (exakt), i tidsdomänen C_f = ∫ γ/(ρ_f K_f) N N dV.
//!
//! *FREQUENCY löser det osymmetriska egenvärdesproblemet K x = ω² M x
//! (dämpning och impedanser ignoreras) med skift-invert
//! subspace-iteration (`acoustic_solve.rs`). Moderna normeras:
//! uᵀM_s u + pᵀM_f p = 1 utan koppling, annars med vänster- och
//! högervektorerna uᵀM_s u + (pᵀM_f p + pᵀS u)/ω² = 1.
//!
//! *STEADY STATE DYNAMICS, DIRECT löser (K − Ω²M + iΩC + impedanser) x = F
//! direkt som det reella systemet [[A_r, −A_i], [A_i, A_r]] med den
//! osymmetriska LU-lösaren. Föreskrivna värden (*BOUNDARY) och laster är
//! reella. Utdata: U, POR (realdel) och U_IM, POR_IM (imaginärdel).
//!
//! *DYNAMIC: HHT-α (standardvärden för APPLICATION=TRANSIENT
//! FIDELITY, α = −0,05) med fasta inkrement (DIRECT) eller dataradens
//! inkrement; systemet är linjärt så varje inkrement är en lösning.
//! Impedansränderna (massa ∫NN/k₁, dämpning ∫NN (1/c₁ + γ/(ρk₁)), styvhet
//! ∫NN γ/(ρc₁) respektive strålningsvillkorets termer) verkar vid
//! inkrementets slut utan HHT-viktning.
//! Utdata: U, V, A, POR.
//!
//! Utdata för trycket: nodfältet POR för noder med akustiskt
//! tryck. Massmatrisen för akustiska element: se `fluid_mass_kind`.

#[path = "acoustic_solve.rs"]
pub mod solve;

// område gaps1: icke-reflekterande ränder (ELLIPTICAL, PROLATE SPHEROIDAL, IMPROVED)
#[path = "acoustic_nonrefl.rs"]
mod nonrefl;
#[path = "acoustic_improved.rs"]
mod improved;
// område acoustics2: akustiska infinita element
#[path = "acoustic_infinite.rs"]
mod infinite;
// område acoustics2: akustik i explicit analys
#[path = "acoustic_explicit.rs"]
mod explicit;

use super::api::{StepCtx, StepType};
use super::dynamic::amplitude::LoadEval;
use super::modal::plain_frame;
use crate::elements::acoustic::{self as ac, AcousticElem};
use crate::elements::MassKind;
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, FeModel, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::interactions::geom;
use crate::materials::acoustic::AcousticProps;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::Recorder;
use solve::{factor_gen, Gen, GenFactor};

pub struct AcousticStep;

pub static STEP_TYPES: &[&dyn StepType] = &[&AcousticStep];

/// Slot för det akustiska trycket (frihetsgrad 8).
pub const PS: usize = 7;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Modellen har akustiska element.
pub fn model_is_acoustic(m: &FeModel) -> bool {
    m.elements.iter().any(|e| ac::is_acoustic(e.etype))
}

/// Massmatristyp för akustiska element i frekvens- och implicita steg
/// (kontrollerat mot egenvärden i testfallen): diagonal (radsummerad) för AC2D3, AC2D4,
/// ACAX3, ACAX4, AC3D4 och AC3D8, konsistent för AC3D6 och andra ordningens
/// element.
pub fn fluid_mass_kind(e: &AcousticElem) -> MassKind {
    if e.first_order() && e.geom.shape != crate::elements::heat::Shape::Wedge6 { MassKind::Lumped } else { MassKind::Consistent }
}

// ---------------------------------------------------------------------------
// Komplexa tal
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct C(pub f64, pub f64);

impl C {
    fn add(self, o: C) -> C {
        C(self.0 + o.0, self.1 + o.1)
    }
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
    fn sqrt(self) -> C {
        let r = self.0.hypot(self.1);
        let re = (0.5 * (r + self.0)).max(0.0).sqrt();
        let im = (0.5 * (r - self.0)).max(0.0).sqrt();
        C(re, if self.1 < 0.0 { -im } else { im })
    }
    fn scale(self, s: f64) -> C {
        C(self.0 * s, self.1 * s)
    }
}

// ---------------------------------------------------------------------------
// Impedansränder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum ImpKind {
    /// Tabell (1/k₁, 1/c₁, frekvens), sorterad efter frekvens.
    Admittance(Vec<(f64, f64, f64)>),
    /// Strålningsvillkor med spridningsterm β.
    Radiation(f64),
}

struct ImpBoundary {
    kind: ImpKind,
    /// ∫ N N dS i alla frihetsgrader (fria + låsta).
    h: Gen,
    props: AcousticProps,
}

fn interp3(t: &[(f64, f64, f64)], f: f64) -> (f64, f64) {
    if t.len() == 1 || f <= t[0].2 {
        return (t[0].0, t[0].1);
    }
    let l = t.len() - 1;
    if f >= t[l].2 {
        return (t[l].0, t[l].1);
    }
    let i = t.windows(2).position(|w| f >= w[0].2 && f <= w[1].2).unwrap_or(0);
    let s = (f - t[i].2) / (t[i + 1].2 - t[i].2);
    (t[i].0 + s * (t[i + 1].0 - t[i].0), t[i].1 + s * (t[i + 1].1 - t[i].1))
}

impl ImpBoundary {
    /// Randtermens koefficient i frekvensdomänen vid Ω (multipliceras med H).
    fn coeff(&self, w: f64) -> C {
        let f = w / TWO_PI;
        match &self.kind {
            ImpKind::Admittance(t) => {
                let (ik1, ic1) = interp3(t, f);
                C(-w * w * ik1, w * ic1)
            }
            ImpKind::Radiation(beta) => {
                let p = &self.props;
                // ρ̃ = ρ + γ/(iΩ) = ρ − iγ/Ω; område gaps1: komplex densitet och
                // kompressionsmodul (COMPLEX ..., POROUS MODEL)
                let (rt, kt) = p.complex_at(f);
                let rho_t = C(rt.0, rt.1);
                let k = rho_t.div(C(kt.0, kt.1)).sqrt().scale(w);
                let base = C(1.0, 0.0).div(rho_t).mul(C(-k.1, k.0).add(C(*beta, 0.0)));
                // område gaps1: med spridningsterm (β > 0) och volymetriskt
                // motstånd läggs den reella termen till
                // γ/(ρ√(ρK))/√(1 + (γ/(Ωρ))²) (gäller för
                // CIRCULAR, SPHERICAL och ELLIPTICAL/PROLATE SPHEROIDAL,
                // tests/cases/gaps1/ac_*_rand_ssd)
                if *beta > 0.0 {
                    let g = if w > 0.0 { p.drag_at(Some(f)) / (w * p.density) } else { 0.0 };
                    // faktorn 1/√(1 + (γ/(Ωρ))²) ger minst avvikelse (≤ 1e-5)
                    let fac = 1.0 / (1.0 + g * g).sqrt();
                    base.add(C(fac * p.drag_at(Some(f)) / (p.density * (p.density * p.bulk).sqrt()), 0.0))
                } else {
                    base
                }
            }
        }
    }

    /// (massa, dämpning, styvhet)-koefficienter i tidsdomänen.
    fn time_coeffs(&self) -> (f64, f64, f64) {
        match &self.kind {
            ImpKind::Admittance(t) => {
                let g = self.props.drag_at(None) / self.props.density;
                (t[0].0, g * t[0].0 + t[0].1, g * t[0].1)
            }
            ImpKind::Radiation(beta) => {
                let p = &self.props;
                let z = (p.density * p.bulk).sqrt();
                let gamma = p.drag_at(None);
                (0.0, 1.0 / z, beta / p.density + gamma / (2.0 * p.density * z))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Det assemblerade systemet
// ---------------------------------------------------------------------------

/// Numrering, matriser och laster för ett akustiskt steg. Matriserna `ms`,
/// `mf`, `s`, `cf`, `drag` och impedansernas `h` har alla frihetsgrader:
/// de fria (samma nummer som i `DofMap`) och sedan de låsta.
struct Sys<'a, 'm> {
    p: &'a Prepared<'m>,
    dm: DofMap,
    /// Antal fria ekvationer.
    n: usize,
    /// Index för låsta (nod, slot) (≥ n), `NONE` annars.
    cidx: Vec<[usize; SLOTS]>,
    con_owner: Vec<(usize, usize)>,
    /// Styvheten i bastillståndet (fria × fria): struktur och K_f med 1/ρ_f.
    k0: Gen,
    ms: Gen,
    mf: Gen,
    /// S (fluidrader, strukturkolumner).
    s: Gen,
    /// C_f = ∫ γ/(ρK) N N dV (tidsdomänen).
    cf: Gen,
    /// G/ρ per material med volymetriskt motstånd (för den komplexa densiteten).
    drag: Vec<(AcousticProps, Gen)>,
    /// Område gaps1: H/K_f per material i `drag` (för den komplexa kompressionsmodulen).
    cmass: Vec<Gen>,
    imp: Vec<ImpBoundary>,
    /// Område gaps1: NONREFLECTING=IMPROVED (cos θ, byggs om per inkrement).
    improved: Vec<improved::Improved>,
    /// Noder med akustiskt tryck.
    pnode: Vec<bool>,
    /// Trycknoder vars förskjutning interpoleras från strukturen (nod mot yta).
    fluid_u: Vec<(usize, Vec<(usize, f64)>)>,
    /// Område acoustics2: akustiska infinita element per material.
    inf: Vec<infinite::InfGroup>,
}

fn material_props(m: &FeModel, e: usize) -> Result<AcousticProps> {
    let el = &m.elements[e];
    let mi = el.section.and_then(|s| m.sections[s].material_index);
    let Some(mi) = mi else { return fail(format!("{} saknar material", m.elem_name(e))) };
    match AcousticProps::from_def(&m.materials[mi])? {
        Some(p) => Ok(p),
        None => fail(format!("{}: materialet {} saknar *ACOUSTIC MEDIUM", m.elem_name(e), m.materials[mi].name)),
    }
}

fn thickness(m: &FeModel, e: usize) -> f64 {
    m.elements[e].section.map_or(1.0, |s| m.sections[s].solid_thickness())
}

/// Impedansränderna som gäller i steg `index` (modelldata och stegdata till
/// och med steget; OP=NEW tar bort tidigare).
fn impedance_keywords(m: &FeModel, index: usize) -> Vec<&RawKeyword> {
    let mut v: Vec<&RawKeyword> = m.extra.iter().filter(|k| k.name == "IMPEDANCE" || k.name == "SIMPEDANCE").collect();
    for st in m.steps.iter().take(index + 1) {
        for k in st.extra.iter().filter(|k| k.name == "IMPEDANCE" || k.name == "SIMPEDANCE") {
            if k.param_upper("OP").is_some_and(|o| o == "NEW") {
                v.clear();
            }
            v.push(k);
        }
    }
    v
}

fn property_table(m: &FeModel, name: &str, kw: &RawKeyword) -> Result<ImpKind> {
    let Some(pk) = m.extra.iter().find(|k| k.name == "IMPEDANCE PROPERTY" && k.param("NAME").is_some_and(|n| n.trim().eq_ignore_ascii_case(name))) else {
        return kw.err(format!("*IMPEDANCE PROPERTY, NAME={name} finns inte"));
    };
    let mut t = Vec::new();
    for d in pk.records() {
        let v = pk.floats(&d)?;
        t.push((v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0)));
    }
    if t.is_empty() {
        return pk.err("datarader saknas");
    }
    t.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    match pk.param_upper("DATA").as_deref() {
        None | Some("ADMITTANCE") => Ok(ImpKind::Admittance(t)),
        Some("IMPEDANCE") => {
            // varje rad görs om till admittans (1/k₁, 1/c₁) vid sin
            // frekvens, och admittansen interpoleras
            let mut a = Vec::with_capacity(t.len());
            for &(zr, zi, f) in &t {
                let y = C(1.0, 0.0).div(C(zr, zi));
                let w = TWO_PI * f;
                if y.1 != 0.0 && w <= 0.0 {
                    return pk.err("en impedans med imaginärdel kräver en frekvens > 0");
                }
                a.push((if w > 0.0 { y.1 / w } else { 0.0 }, y.0, f));
            }
            Ok(ImpKind::Admittance(a))
        }
        Some(x) => pk.err(format!("DATA={x} ska vara ADMITTANCE eller IMPEDANCE")),
    }
}

/// Randmatrisen H = ∫ N N dS för elementytan `fi` (konsistent). För
/// axisymmetriska element används den plana matrisen ∫ N N dl gånger
/// 2π r_c med r_c = medelradien av ytans hörn (se
/// tests/cases/acoustics/acax4_sfar_ssd).
#[allow(dead_code)]
fn face_matrix(a: &AcousticElem, x: &[[f64; 3]], fi: usize, thickness: f64) -> Result<Vec<f64>> {
    face_matrix_w(a, x, fi, thickness, &|_| 1.0)
}

/// Område gaps1: ∫ f(x) N N dS med en viktfunktion i integrationspunkterna
/// (geometrifaktorn för NONREFLECTING=ELLIPTICAL/PROLATE SPHEROIDAL).
fn face_matrix_w(a: &AcousticElem, x: &[[f64; 3]], fi: usize, thickness: f64, wf: &dyn Fn([f64; 3]) -> f64) -> Result<Vec<f64>> {
    let nne = x.len();
    let mut he = vec![0.0; nne * nne];
    let (geo, scale) = if a.geom.axisym {
        let fl = a.geom.faces()[fi];
        let nc = if fl.len() == 3 { 2 } else { fl.len() };
        let rc: f64 = fl[..nc].iter().map(|&i| x[i][0]).sum::<f64>() / nc as f64;
        (crate::elements::heat::HeatElem { name: a.name, shape: a.geom.shape, axisym: false }, 2.0 * std::f64::consts::PI * rc)
    } else {
        (crate::elements::heat::HeatElem { name: a.name, shape: a.geom.shape, axisym: false }, 1.0)
    };
    let t = if a.geom.axisym { 1.0 } else { thickness };
    for fp in geo.face_points(x, fi, t, false)? {
        let mut xp = [0.0; 3];
        for (i, xi) in x.iter().enumerate() {
            for d in 0..3 {
                xp[d] += fp.n[i] * xi[d];
            }
        }
        let fw = wf(xp);
        for i in 0..nne {
            if fp.n[i] == 0.0 {
                continue;
            }
            for j in 0..nne {
                he[i * nne + j] += fw * scale * fp.w * fp.n[i] * fp.n[j];
            }
        }
    }
    Ok(he)
}

impl<'a, 'm> Sys<'a, 'm> {
    fn new(ctx: &StepCtx<'a, 'm>, with_imp: bool) -> Result<Self> {
        let p = ctx.prep;
        let m = p.model;
        let nn = m.nodes.len();
        let unsym = crate::solver::select(Some("unsymmetric"))?;
        let (dm, _) = assembly::dofmap(p, &ctx.loads.constrained(), unsym)?;
        let n = dm.n_eq;
        let mut cidx = vec![[NONE; SLOTS]; nn];
        let mut con_owner = Vec::new();
        for nd in 0..nn {
            for s in 0..SLOTS {
                if dm.constrained[nd] & (1 << s) != 0 {
                    cidx[nd][s] = n + con_owner.len();
                    con_owner.push((nd, s));
                }
            }
        }
        let pnode: Vec<bool> = (0..nn).map(|nd| p.active[nd] & (1 << PS) != 0).collect();
        for e in &m.elements {
            if e.etype.family() == crate::elements::Family::Heat {
                return fail("akustiska element och värmeledningselement i samma modell stöds inte");
            }
        }
        let mut sys = Sys {
            p,
            dm,
            n,
            cidx,
            con_owner,
            k0: Gen::new(n),
            ms: Gen::new(0),
            mf: Gen::new(0),
            s: Gen::new(0),
            cf: Gen::new(0),
            drag: Vec::new(),
            cmass: Vec::new(),
            imp: Vec::new(),
            improved: Vec::new(),
            pnode,
            fluid_u: Vec::new(),
            inf: Vec::new(),
        };
        let na = sys.n_all();
        sys.ms = Gen::new(na);
        sys.mf = Gen::new(na);
        sys.s = Gen::new(na);
        sys.cf = Gen::new(na);
        // styvheten i bastillståndet (element och interaktioner)
        let zero = vec![[0.0; SLOTS]; nn];
        let mut rec = Recorder::new(n);
        assembly::internal(p, &sys.dm, &ctx.state.u, &zero, &ctx.state.elem_state, ctx.nlgeom, ctx.state.time, 0.0, Some(&mut rec))?;
        let ks = rec.build();
        for r in 0..n {
            for idx in ks.rowptr[r]..ks.rowptr[r + 1] {
                sys.k0.add(r, ks.col[idx] as usize, ks.val[idx]);
            }
        }
        sys.k0.finish();
        // massor och fluidens dämpning
        let mut drag: Vec<(usize, AcousticProps, Gen)> = Vec::new();
        let mut cmass: Vec<Gen> = Vec::new();
        for (e, el) in m.elements.iter().enumerate() {
            let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
            let mi = el.section.and_then(|s| m.sections[s].material_index);
            if let Some(a) = ac::lookup(el.etype.name()) {
                let props = material_props(m, e)?;
                let x: Vec<[f64; 3]> = el.nodes.iter().map(|&nd| m.nodes[nd]).collect();
                let t = thickness(m, e);
                let geo = a.geometric(&x, t).map_err(wrap)?;
                let nne = el.nodes.len();
                let h = match fluid_mass_kind(a) {
                    MassKind::Consistent => geo.h.clone(),
                    MassKind::Lumped => AcousticElem::lump(&geo.h, nne),
                };
                let mfe: Vec<f64> = h.iter().map(|v| v / props.bulk).collect();
                sys.scatter_into(e, &mfe, Target::Mf);
                let gamma = props.drag_at(None);
                if gamma != 0.0 {
                    let ce: Vec<f64> = h.iter().map(|v| v * gamma / (props.density * props.bulk)).collect();
                    sys.scatter_into(e, &ce, Target::Cf);
                }
                if props.freq_dependent() {
                    let ge: Vec<f64> = geo.g.iter().map(|v| v / props.density).collect();
                    let i = match drag.iter().position(|d| Some(d.0) == mi) {
                        Some(i) => i,
                        None => {
                            drag.push((mi.unwrap_or(0), props.clone(), Gen::new(na)));
                            cmass.push(Gen::new(na));
                            drag.len() - 1
                        }
                    };
                    let mut g = std::mem::take(&mut drag[i].2);
                    sys.scatter(e, &ge, &mut g);
                    drag[i].2 = g;
                    let mut g = std::mem::take(&mut cmass[i]);
                    sys.scatter(e, &mfe, &mut g);
                    cmass[i] = g;
                }
            } else {
                // område acoustics2: infinita element byggs i build_infinite
                if crate::elements::acoustic_infinite::lookup(el.etype.name()).is_some() {
                    continue;
                }
                if let Some(mi) = mi {
                    if m.materials[mi].behavior("DAMPING").is_some() {
                        return fail(format!("material {}: *DAMPING stöds inte ännu i akustiska steg", m.materials[mi].name));
                    }
                }
                let pc = p.ctx(e, false, ctx.state.time, 0.0);
                if pc.density == 0.0 && pc.section.and_then(|s| s.beam()).is_none() {
                    continue;
                }
                let me = el.etype.mass(&pc, super::frequency::mass_kind(el.etype)).map_err(wrap)?;
                sys.scatter_into(e, &me, Target::Ms);
            }
        }
        for d in drag.iter_mut() {
            d.2.finish();
        }
        sys.drag = drag.into_iter().map(|(_, p, g)| (p, g)).collect();
        for g in cmass.iter_mut() {
            g.finish();
        }
        sys.cmass = cmass;
        // akustisk–strukturell koppling
        for c in crate::interactions::acoustic::couplings(m)? {
            sys.fluid_u.extend(c.fluid_u.iter().cloned());
            for &(pn, un, dir, v) in &c.entries {
                let rows = sys.eqs(pn, PS);
                let cols = sys.eqs(un, dir);
                for &(r, cr) in &rows {
                    for &(cc, ccf) in &cols {
                        sys.s.add(r, cc, cr * ccf * v);
                    }
                }
            }
        }
        sys.inf = sys.build_infinite()?;
        for g in [&mut sys.ms, &mut sys.mf, &mut sys.s, &mut sys.cf] {
            g.finish();
        }
        if with_imp {
            sys.build_impedances(ctx.index)?;
        }
        Ok(sys)
    }

    fn n_all(&self) -> usize {
        self.n + self.con_owner.len()
    }

    /// Ekvationer och koefficienter för (nod, slot) i numreringen med låsta.
    fn eqs(&self, nd: usize, s: usize) -> Vec<(usize, f64)> {
        if self.cidx[nd][s] != NONE {
            return vec![(self.cidx[nd][s], 1.0)];
        }
        let mut v = Vec::new();
        self.dm.for_each_eq(nd, s, |q, c| v.push((q, c)));
        v
    }

    fn scatter(&self, e: usize, ke: &[f64], g: &mut Gen) {
        let dl = &self.p.edofs[e];
        let maps: Vec<Vec<(usize, f64)>> = dl.iter().map(|&(nd, s)| self.eqs(nd, s)).collect();
        let nd = dl.len();
        for a in 0..nd {
            for b in 0..nd {
                let v = ke[a * nd + b];
                if v == 0.0 {
                    continue;
                }
                for &(r, cr) in &maps[a] {
                    for &(c, cc) in &maps[b] {
                        g.add(r, c, cr * cc * v);
                    }
                }
            }
        }
    }

    fn scatter_into(&mut self, e: usize, ke: &[f64], t: Target) {
        let mut g = std::mem::take(match t {
            Target::Ms => &mut self.ms,
            Target::Mf => &mut self.mf,
            Target::Cf => &mut self.cf,
        });
        self.scatter(e, ke, &mut g);
        *match t {
            Target::Ms => &mut self.ms,
            Target::Mf => &mut self.mf,
            Target::Cf => &mut self.cf,
        } = g;
    }

    fn build_impedances(&mut self, index: usize) -> Result<()> {
        // område gaps1: ränderna byggs i acoustic_nonrefl.rs (även ELLIPTICAL,
        // PROLATE SPHEROIDAL och IMPROVED)
        self.build_impedances_gaps1(index)
    }

    /// Strukturrad-/fluidkolumnblocket −Sᵀ i styvheten (alla frihetsgrader).
    fn minus_st(&self) -> Gen {
        let mut g = Gen::new(self.n_all());
        for (r, row) in self.s.rows.iter().enumerate() {
            for &(c, v) in row {
                g.add(c as usize, r, -v);
            }
        }
        g.finish();
        g
    }

    /// Lösarens block: en nod med fria ekvationer (`mult` ekvationer per fri frihetsgrad).
    fn blocks(&self, mult: usize) -> Vec<([f64; 3], Vec<usize>)> {
        let m = self.p.model;
        let mut per: Vec<Vec<usize>> = vec![Vec::new(); m.nodes.len()];
        for q in 0..self.n {
            let (nd, _) = self.dm.owner[q];
            for k in 0..mult {
                per[nd].push(mult * q + k);
            }
        }
        per.into_iter().enumerate().filter(|(_, v)| !v.is_empty()).map(|(nd, v)| (m.nodes[nd], v)).collect()
    }

    /// Nodvektor ur en vektor över de fria ekvationerna (+ låsta värden).
    fn to_nodes(&self, x: &[f64], fixed: &[f64]) -> Vec<NodeVec> {
        let nn = self.p.model.nodes.len();
        let mut u = vec![[0.0; SLOTS]; nn];
        for (q, &(nd, s)) in self.dm.owner.iter().enumerate() {
            u[nd][s] = x[q];
        }
        for (k, &(nd, s)) in self.con_owner.iter().enumerate() {
            u[nd][s] = fixed.get(k).copied().unwrap_or(0.0);
        }
        crate::steps::dynamic::expand_rate(&self.dm, &mut u);
        u
    }

    /// U i trycknoder som är slavar i en nod-mot-yta-koppling.
    fn with_fluid_u(&self, u: &[NodeVec]) -> Vec<NodeVec> {
        let mut v = u.to_vec();
        for (pn, interp) in &self.fluid_u {
            for d in 0..3 {
                v[*pn][d] = interp.iter().map(|&(un, w)| w * u[un][d]).sum();
            }
        }
        v
    }

    /// POR-fältet (bara noder med akustiskt tryck).
    fn por(&self, u: &[NodeVec]) -> Vec<Vec<f64>> {
        u.iter().zip(&self.pnode).map(|(v, &p)| if p { vec![v[PS]] } else { Vec::new() }).collect()
    }
}

#[derive(Clone, Copy)]
enum Target {
    Ms,
    Mf,
    Cf,
}

// ---------------------------------------------------------------------------
// Stegtypen
// ---------------------------------------------------------------------------

impl StepType for AcousticStep {
    fn name(&self) -> &'static str {
        "ACOUSTIC"
    }

    fn claim(&self, _step: &StepDef, _nlgeom: bool) -> Option<i32> {
        None
    }

    fn claim_model(&self, step: &StepDef, _nlgeom: bool, model: &FeModel) -> Option<i32> {
        if !model_is_acoustic(model) {
            return None;
        }
        match step.procedure_name() {
            "FREQUENCY" => Some(50),
            "STEADY STATE DYNAMICS" if step.procedure.as_ref().is_some_and(|k| k.has("DIRECT")) => Some(50),
            "DYNAMIC" if !step.procedure.as_ref().is_some_and(|k| k.has("EXPLICIT")) => Some(50),
            // område acoustics2: *DYNAMIC, EXPLICIT (acoustic_explicit.rs)
            "DYNAMIC" => Some(50),
            _ => None,
        }
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        match ctx.step.procedure_name() {
            "FREQUENCY" => frequency(ctx),
            "STEADY STATE DYNAMICS" => steady_state(ctx),
            "DYNAMIC" if ctx.step.procedure.as_ref().is_some_and(|k| k.has("EXPLICIT")) => explicit::dynamic_explicit(ctx),
            "DYNAMIC" => dynamic(ctx),
            other => fail(format!("*{other} stöds inte för akustiska modeller")),
        }
    }
}

/// Frame med U (och elementutdata) samt POR.
fn frame(ctx: &StepCtx, sys: &Sys, u: &[NodeVec], value: f64, mode: i64, desc: String) -> Result<FrameData> {
    let mut fr = plain_frame(ctx, &sys.dm, &sys.with_fluid_u(u), value, mode, desc)?;
    fr.node_extra.push(("POR".into(), vec!["POR".into()], sys.por(u)));
    Ok(fr)
}

// ---------------------------------------------------------------------------
// *FREQUENCY
// ---------------------------------------------------------------------------

fn data_field(kw: &RawKeyword, i: usize) -> Result<Option<f64>> {
    super::frequency::data_field(kw, i)
}

fn frequency(ctx: &mut StepCtx) -> Result<()> {
    let kw = ctx.step.procedure.clone().unwrap();
    let solver_name = kw.param_upper("EIGENSOLVER").unwrap_or_else(|| "LANCZOS".into());
    let (nev, min_f, max_f, shift) = match solver_name.as_str() {
        "LANCZOS" | "AMS" => (data_field(&kw, 0)?, data_field(&kw, 1)?, data_field(&kw, 2)?, data_field(&kw, 3)?),
        "SUBSPACE" => (data_field(&kw, 0)?, None, data_field(&kw, 1)?, data_field(&kw, 2)?),
        other => return kw.err(format!("EIGENSOLVER={other} stöds inte")),
    };
    let nev = nev.map(|v| v.round().max(0.0) as usize).filter(|&v| v > 0);
    if nev.is_none() && max_f.is_none() {
        return kw.err("ange antal egenvärden eller högsta frekvens på dataraden");
    }
    let sys = Sys::new(ctx, false)?;
    let n = sys.n;
    if n == 0 {
        return fail("*FREQUENCY: inga fria frihetsgrader");
    }
    let mst = sys.minus_st();
    let ksym = sys.infinite_stiffness();
    let mut k = Gen::combine(&[(&sys.k0, 1.0), (&mst, 1.0), (&ksym, 1.0)], n);
    let mut m = Gen::combine(&[(&sys.ms, 1.0), (&sys.mf, 1.0), (&sys.s, 1.0)], n);
    // skalning D K D, D M D med D = 1 för strukturen och d för trycket så att
    // fluidens och strukturens rader har samma storleksordning (egenvärdena
    // ändras inte; vektorn x = D x̂). Annars styr strukturdelen ortonormeringen
    // och konvergenskontrollen helt och trycket blir onoggrant.
    let dscale = {
        let kd = k.diag();
        let med = |fl: bool| -> f64 {
            let mut v: Vec<f64> = (0..n).filter(|&q| (sys.dm.owner[q].1 == PS) == fl && kd[q] > 0.0).map(|q| kd[q]).collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let (ks, kf) = (med(false), med(true));
        if ks > 0.0 && kf > 0.0 { (ks / kf).sqrt() } else { 1.0 }
    };
    let dvec: Vec<f64> = (0..n).map(|q| if sys.dm.owner[q].1 == PS { dscale } else { 1.0 }).collect();
    if dscale != 1.0 {
        for g in [&mut k, &mut m] {
            for (r, row) in g.rows.iter_mut().enumerate() {
                for e in row.iter_mut() {
                    e.1 *= dvec[r] * dvec[e.0 as usize];
                }
            }
        }
    }
    let unsym = crate::solver::select(Some("unsymmetric"))?;
    let blocks = sys.blocks(1);
    // skift: angivet, 0 eller negativt (singulär K: kavitetens konstanta tryck, stelkroppsmoder)
    let kd = k.diag();
    let md = m.diag();
    // område acoustics2: utan de infinita elementens generaliserade frihetsgrader (nästan ingen massa)
    let mut ratios: Vec<f64> = kd
        .iter()
        .zip(&md)
        .enumerate()
        .filter(|(q, (a, b))| **a > 0.0 && **b > 0.0 && sys.p.model.node_label[sys.dm.owner[*q].0] > -3_000_000_000)
        .map(|(_, (a, b))| a / b)
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let scale = ratios.get(ratios.len() / 2).copied().unwrap_or(1.0);
    let mut cands: Vec<f64> = Vec::new();
    if let Some(s) = shift {
        cands.push((TWO_PI * s).powi(2));
    }
    cands.push(0.0);
    cands.extend((0..5).map(|i| -scale * 10f64.powi(-8 + 2 * i)));
    let try_factor = |sigma: f64| -> Result<Option<GenFactor>> {
        let a = Gen::combine(&[(&k, 1.0), (&m, -sigma)], n);
        let Ok(f) = factor_gen(&a, &blocks, unsym)? else { return Ok(None) };
        // kontroll med slumpat högerled: nästan singulära matriser ger stor residual
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let b: Vec<f64> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let mut x = b.clone();
        f.solve(&mut x);
        let r = a.mul(&x);
        let err = r.iter().zip(&b).fold(0.0f64, |s, (ri, bi)| s.max((ri - bi).abs()));
        Ok((err <= 1e-6 * 0.5).then_some(f))
    };
    let mut found = None;
    for c in cands {
        if let Some(f) = try_factor(c)? {
            found = Some((c, f));
            break;
        }
    }
    let Some((mut sigma, mut fac)) = found else {
        return fail("*FREQUENCY: systemmatrisen kunde inte faktoriseras ens med negativt skift");
    };
    let lam_max = max_f.map(|f| (TWO_PI * f).powi(2));
    let lam_min = min_f.map(|f| (TWO_PI * f).powi(2));
    let mut want = nev.unwrap_or(10).min(n);
    if sigma < 0.0 && shift.is_none() {
        // singulära moder: grov uppskattning, sedan skift ≈ −0,1 λ₁ (elastisk)
        let op = |b: &mut [f64]| fac.solve(b);
        if let Ok(r) = solve::subspace_unsym(&k, &m, sigma, &op, want.min(n), 1e-6, 80) {
            let el = r.lambda.iter().copied().filter(|l| *l > 10.0 * sigma.abs()).fold(f64::INFINITY, f64::min);
            if el.is_finite() {
                if let Some(f) = try_factor(-0.1 * el)? {
                    sigma = -0.1 * el;
                    fac = f;
                }
            }
        }
    }
    let res = loop {
        let op = |b: &mut [f64]| fac.solve(b);
        let r = solve::subspace_unsym(&k, &m, sigma, &op, want, 1e-10, 400)?;
        let below = lam_min.map_or(0, |lm| r.lambda.iter().filter(|l| **l < lm).count());
        let top = r.lambda.last().copied();
        let full = want >= n;
        let enough = match (nev, lam_max, top) {
            (Some(nv), _, _) => r.lambda.len() - below >= nv || full,
            (None, Some(lm), Some(t)) => t > lm || full,
            _ => true,
        };
        if enough {
            break r;
        }
        want = match nev {
            Some(nv) => (nv + below).max(want + 1).min(n),
            None => (want * 2).min(n),
        };
    };
    let mut modes: Vec<(f64, Vec<f64>)> =
        res.lambda.into_iter().zip(res.vectors).map(|(l, v)| (l, v.iter().zip(&dvec).map(|(a, d)| a * d).collect())).collect();
    modes.retain(|(l, _)| lam_max.is_none_or(|mx| *l <= mx) && lam_min.is_none_or(|mn| *l >= mn));
    if let Some(nv) = nev {
        modes.truncate(nv);
    }
    if ctx.opts.verbose {
        eprintln!("  akustisk egenvärdesextraktion: σ = {sigma:.4e}, {n} obekanta, {} iterationer", res.iterations);
    }
    // utdata
    let base = super::frequency::base_frame(ctx.prep, &sys.dm, ctx.state)?;
    ctx.out.frames.push(base);
    let names = ["EIGVAL", "EIGFREQ", "GM"];
    let mut hist: Vec<Vec<(f64, f64)>> = vec![Vec::new(); names.len()];
    let ms_f = sys.ms.leading(n);
    let mf_f = sys.mf.leading(n);
    let s_f = sys.s.leading(n);
    let coupled = sys.s.rows.iter().any(|r| !r.is_empty());
    for (i, (lam, mut x)) in modes.into_iter().enumerate() {
        let mode = i + 1;
        let lam = if lam.abs() < 1e-9 * scale { 0.0 } else { lam };
        // generaliserad massa med vänstervektorn (u, p/λ)
        let ux: Vec<f64> = (0..n).map(|q| if sys.dm.owner[q].1 != PS { x[q] } else { 0.0 }).collect();
        let px: Vec<f64> = (0..n).map(|q| if sys.dm.owner[q].1 == PS { x[q] } else { 0.0 }).collect();
        let dotp = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let mu = dotp(&ux, &ms_f.mul(&ux));
        let mp = dotp(&px, &mf_f.mul(&px));
        // ren akustik (ingen koppling) pᵀM_f p + uᵀM_s u = 1; kopplade
        // moder normeras med vänster- och högervektorerna (u, p/λ) och (u, p):
        // uᵀM_s u + (pᵀM_f p + pᵀS u)/λ = 1, för λ = 0 pᵀM_f p + pᵀS u = 1
        let ps = dotp(&px, &s_f.mul(&ux));
        let gm = if !coupled {
            mu + mp
        } else if lam > 0.0 {
            mu + (mp + ps) / lam
        } else {
            mp + ps + mu
        };
        let gm = if gm > 0.0 { gm } else { mu + mp };
        if gm > 0.0 {
            let f = 1.0 / gm.sqrt();
            x.iter_mut().for_each(|v| *v *= f);
        }
        // tecken: största komponenten positiv (tecknet är annars godtyckligt)
        let big = x.iter().copied().fold(0.0f64, |a, v| if v.abs() > a.abs() { v } else { a });
        if big < 0.0 {
            x.iter_mut().for_each(|v| *v = -*v);
        }
        let u = sys.to_nodes(&x, &[]);
        let freq = lam.max(0.0).sqrt() / TWO_PI;
        let g5 = super::frequency::g5;
        let desc = format!("Mode {mode:>9}: Value = {:>11} Freq = {:>11} (cycles/time)", g5(lam), g5(freq));
        let mut fr = frame(ctx, &sys, &u, mode as f64, mode as i64, desc)?;
        fr.mode = mode as i64;
        ctx.out.frames.push(fr);
        let t = mode as f64;
        hist[0].push((t, lam));
        hist[1].push((t, freq));
        hist[2].push((t, 1.0));
    }
    for (name, data) in names.iter().zip(hist) {
        ctx.out.history.push((super::frequency::HISTORY_REGION.to_string(), name.to_string(), data));
    }
    ctx.out.increments = 1;
    ctx.out.iterations = res.iterations;
    Ok(())
}

// ---------------------------------------------------------------------------
// *STEADY STATE DYNAMICS, DIRECT
// ---------------------------------------------------------------------------

/// Föreskrivna värden i de låsta frihetsgraderna (perturbation: ändringen från bastillståndet).
fn prescribed(ctx: &StepCtx, sys: &Sys, values: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut x = vec![0.0; sys.con_owner.len()];
    for &(nd, s, v) in values {
        let k = sys.cidx[nd][s];
        if k != NONE {
            let base = if ctx.step.perturbation { ctx.state.u[nd][s] } else { 0.0 };
            x[k - sys.n] = v - base;
        }
    }
    x
}

/// K_fc x_c (fria rader) för de låsta värdena `xc` via elementens inre krafter.
fn k_times_fixed(ctx: &StepCtx, sys: &Sys, xc: &[f64]) -> Result<Vec<f64>> {
    let p = ctx.prep;
    let nn = p.model.nodes.len();
    let mut u = vec![[0.0; SLOTS]; nn];
    let mut any = false;
    for (k, &(nd, s)) in sys.con_owner.iter().enumerate() {
        u[nd][s] = xc[k];
        any |= xc[k] != 0.0;
    }
    let mut r = vec![0.0; sys.n];
    if !any {
        return Ok(r);
    }
    let st0 = p.init_states()?;
    let int = assembly::internal(p, &sys.dm, &u, &u, &st0, false, ctx.state.time, 0.0, None)?;
    for (q, &(nd, s)) in sys.dm.owner.iter().enumerate() {
        r[q] = int.fint[nd][s];
    }
    Ok(r)
}

fn steady_state(ctx: &mut StepCtx) -> Result<()> {
    let kw = ctx.step.procedure.clone().unwrap();
    if kw.has("SUBSPACE PROJECTION") || kw.has("REAL ONLY") {
        return kw.err("SUBSPACE PROJECTION/REAL ONLY stöds inte");
    }
    let sys = Sys::new(ctx, true)?;
    let p = ctx.prep;
    let model = p.model;
    let nn = model.nodes.len();
    let n = sys.n;
    let na = sys.n_all();
    let pts = super::modal::steady_state::frequency_points(&kw, &[])?;
    let mut lev = LoadEval::new(model, ctx.loads, false, 1.0, ctx.state.time)?;
    lev.step_full_at_start = true;
    let unsym = crate::solver::select(Some("unsymmetric"))?;
    let blocks = sys.blocks(2);
    let mst = sys.minus_st();
    // frekvensoberoende delar (alla frihetsgrader): massa och koppling
    let mall = Gen::combine(&[(&sys.ms, 1.0), (&sys.mf, 1.0), (&sys.s, 1.0)], na);
    let z = vec![[0.0; SLOTS]; nn];
    let mut frames = Vec::new();
    let mut f0 = frame(ctx, &sys, &z, 0.0, -1, "Increment 0: Frequency = 0.000".into())?;
    add_im(ctx, &sys, &mut f0, &z)?;
    frames.push(f0);
    for (kf, &f) in pts.iter().enumerate() {
        let w = TWO_PI * f;
        // A = A_r + i A_i (alla frihetsgrader)
        let mut terms_r: Vec<(&Gen, f64)> = vec![(&mst, 1.0), (&mall, -w * w)];
        let mut terms_i: Vec<(&Gen, f64)> = Vec::new();
        let mut drag_c: Vec<C> = Vec::new();
        let mut kc: Vec<C> = Vec::new();
        for (pr, _) in &sys.drag {
            // ρ/ρ̃ − 1 med ρ̃ = ρ − iγ/Ω; område gaps1: komplex densitet och
            // kompressionsmodul, −Ω² (K/K̃ − 1) H/K
            pr.porous_check(f)?;
            let (rt, kt) = pr.complex_at(f);
            let r = if w > 0.0 { C(pr.density, 0.0).div(C(rt.0, rt.1)) } else { C(1.0, 0.0) };
            drag_c.push(C(r.0 - 1.0, r.1));
            let q = C(pr.bulk, 0.0).div(C(kt.0, kt.1));
            kc.push(C(-w * w * (q.0 - 1.0), -w * w * q.1));
        }
        for ((_, g), c) in sys.drag.iter().zip(&drag_c) {
            terms_r.push((g, c.0));
            terms_i.push((g, c.1));
        }
        for (g, c) in sys.cmass.iter().zip(&kc) {
            terms_r.push((g, c.0));
            terms_i.push((g, c.1));
        }
        // område acoustics2: infinita element
        let inf_c: Vec<(C, C, C)> = sys.inf.iter().map(|g| g.coeffs(w)).collect();
        for (g, c) in sys.inf.iter().zip(&inf_c) {
            terms_r.extend([(&g.k, c.0 .0), (&g.c, c.1 .0), (&g.m, c.2 .0)]);
            terms_i.extend([(&g.k, c.0 .1), (&g.c, c.1 .1), (&g.m, c.2 .1)]);
        }
        let imp_c: Vec<C> = sys.imp.iter().map(|b| b.coeff(w)).collect();
        for (b, c) in sys.imp.iter().zip(&imp_c) {
            terms_r.push((&b.h, c.0));
            terms_i.push((&b.h, c.1));
        }
        let ar_all = Gen::combine(&terms_r, na);
        let ai_all = Gen::combine(&terms_i, na);
        // reellt system 2n×2n: rad 2q: A_r x − A_i y, rad 2q+1: A_i x + A_r y
        let mut big = Gen::new(2 * n);
        let k0 = &sys.k0;
        for q in 0..n {
            for &(c, v) in &k0.rows[q] {
                let c = c as usize;
                big.add(2 * q, 2 * c, v);
                big.add(2 * q + 1, 2 * c + 1, v);
            }
            for &(c, v) in &ar_all.rows[q] {
                let c = c as usize;
                if c < n {
                    big.add(2 * q, 2 * c, v);
                    big.add(2 * q + 1, 2 * c + 1, v);
                }
            }
            for &(c, v) in &ai_all.rows[q] {
                let c = c as usize;
                if c < n {
                    big.add(2 * q, 2 * c + 1, -v);
                    big.add(2 * q + 1, 2 * c, v);
                }
            }
        }
        big.finish();
        let fac = match factor_gen(&big, &blocks, unsym)? {
            Ok(fc) => fc,
            Err(_) => return fail(format!("*STEADY STATE DYNAMICS, DIRECT: dynamiska styvhetsmatrisen är singulär vid {f} Hz (resonans utan dämpning?)")),
        };
        // högerled: laster och föreskrivna värden
        let fext = assembly::external(p, &lev.cloads_at(f), &lev.dloads_at(f), &z, false, ctx.state.time)?;
        let xc = prescribed(ctx, &sys, &lev.bc_at(f));
        let kx = k_times_fixed(ctx, &sys, &xc)?;
        let mut xall = vec![0.0; na];
        xall[n..].copy_from_slice(&xc);
        let arx = ar_all.mul(&xall);
        let aix = ai_all.mul(&xall);
        let mut rhs = vec![0.0; 2 * n];
        for q in 0..n {
            let (nd, s) = sys.dm.owner[q];
            rhs[2 * q] = fext[nd][s] - kx[q] - arx[q];
            rhs[2 * q + 1] = -aix[q];
        }
        fac.solve(&mut rhs);
        let xr: Vec<f64> = (0..n).map(|q| rhs[2 * q]).collect();
        let xi: Vec<f64> = (0..n).map(|q| rhs[2 * q + 1]).collect();
        let re = sys.to_nodes(&xr, &xc);
        let im = sys.to_nodes(&xi, &vec![0.0; xc.len()]);
        let mut fr = frame(ctx, &sys, &re, f, -1, format!("Increment {}: Frequency = {f:.4}", kf + 1))?;
        add_im(ctx, &sys, &mut fr, &im)?;
        frames.push(fr);
    }
    if !ctx.opts.all_frames {
        let last = frames.pop();
        frames.clear();
        frames.extend(last);
    }
    ctx.out.increments = frames.len();
    ctx.out.frames.extend(frames);
    Ok(())
}

/// Imaginärdelen: U_IM (UR_IM), POR_IM och elementutdata med ändelsen _IM.
fn add_im(ctx: &StepCtx, sys: &Sys, fr: &mut FrameData, im: &[NodeVec]) -> Result<()> {
    let dim = ctx.prep.model.dim();
    let rot = super::modal::has_rotations(ctx);
    let im_u = sys.with_fluid_u(im);
    let mut tmp = FrameData::default();
    super::modal::add_node_field(&mut tmp, dim, rot, "U", "UR", &im_u);
    for (name, labels, vals) in tmp.node_extra {
        fr.node_extra.push((format!("{name}_IM"), labels, vals));
    }
    if rot {
        // realdelens rotationer som eget fält (som i steady_state.rs)
        let mut tr = FrameData::default();
        super::modal::add_node_field(&mut tr, dim, rot, "_", "UR", &fr.u);
        fr.node_extra.extend(tr.node_extra.into_iter().filter(|f| f.0 != "_" && dim == 2));
    }
    fr.node_extra.push(("POR_IM".into(), vec!["POR".into()], sys.por(im)));
    let fi = plain_frame(ctx, &sys.dm, im, fr.value, -1, String::new())?;
    for (e, fields) in fi.elem.into_iter().enumerate() {
        for mut fld in fields {
            fld.name = format!("{}_IM", fld.name);
            if let Some(v) = fr.elem.get_mut(e) {
                v.push(fld);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// *DYNAMIC (implicit, linjärt)
// ---------------------------------------------------------------------------

fn dynamic(ctx: &mut StepCtx) -> Result<()> {
    let kw = ctx.step.procedure.clone().unwrap();
    let hht = super::dynamic::Hht::from_keyword(&kw)?;
    let (alpha, beta, gamma) = (hht.alpha, hht.beta, hht.gamma);
    let d = kw.first_floats()?;
    let dt0 = d.first().copied().unwrap_or(0.0);
    let period = d.get(1).copied().filter(|v| *v > 0.0).unwrap_or(1.0);
    if !kw.has("DIRECT") && ctx.opts.verbose {
        eprintln!("  akustisk *DYNAMIC: automatisk inkrementering från Δt = {dt0}");
    }
    let h = if dt0 > 0.0 { dt0.min(period) } else { period / 100.0 };
    let sys = Sys::new(ctx, true)?;
    let p = ctx.prep;
    let model = p.model;
    let n = sys.n;
    let nn = model.nodes.len();
    // tidsdomänens matriser (fria × fria)
    let mst = sys.minus_st();
    let mut mt: Vec<(&Gen, f64)> = vec![(&sys.ms, 1.0), (&sys.mf, 1.0), (&sys.s, 1.0)];
    let mut ct0: Vec<(&Gen, f64)> = vec![(&sys.cf, 1.0)];
    let mut ct: Vec<(&Gen, f64)> = Vec::new();
    let mut kt: Vec<(&Gen, f64)> = vec![(&sys.k0, 1.0), (&mst, 1.0)];
    // område acoustics2: infinita element (styvhet K_g/ρ, dämpning −C_g/(ρc) + γ/(ρK) H_g)
    for g in &sys.inf {
        kt.push((&g.k, 1.0));
        ct0.extend([(&g.c, -1.0), (&g.cd, 1.0)]);
    }
    let mut kti: Vec<(&Gen, f64)> = Vec::new();
    let coeffs: Vec<(f64, f64, f64)> = sys.imp.iter().map(|b| b.time_coeffs()).collect();
    for (b, c) in sys.imp.iter().zip(&coeffs) {
        mt.push((&b.h, c.0));
        ct.push((&b.h, c.1));
        kti.push((&b.h, c.2));
    }
    let mm = Gen::combine(&mt, n);
    let cm = Gen::combine(&ct0, n);
    let ci = Gen::combine(&ct, n);
    // impedansränderna verkar vid inkrementets slut utan HHT-viktning
    // (kontrollerat i tests/cases/acoustics/ac3d8_ror_dynamic)
    let ki = Gen::combine(&kti, n);
    let km = Gen::combine(&kt, n);
    let unsym = crate::solver::select(Some("unsymmetric"))?;
    let blocks = sys.blocks(1);
    let ramp = super::dynamic::default_ramp(ctx.step);
    let lev = LoadEval::new(model, ctx.loads, ramp, period, ctx.state.time)?;
    let force = |t: f64| -> Result<Vec<f64>> {
        let z = vec![[0.0; SLOTS]; nn];
        let f = assembly::external(p, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, ctx.state.time + t)?;
        Ok((0..n).map(|q| f[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect())
    };
    // starttillstånd (förskjutning och tryck från föregående steg, hastighet noll
    // eller *INITIAL CONDITIONS, TYPE=VELOCITY)
    let mut u: Vec<f64> = (0..n).map(|q| ctx.state.u[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect();
    let v0 = super::dynamic::initial_velocity(p, ctx.state)?;
    let mut v: Vec<f64> = (0..n).map(|q| v0[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect();
    let mut f_old = force(0.0)?;
    // begynnelseacceleration: M a = F − C v − K u
    let ku: Vec<f64> = km.mul(&u).iter().zip(ki.mul(&u)).map(|(a, b)| a + b).collect();
    let cv: Vec<f64> = cm.mul(&v).iter().zip(ci.mul(&v)).map(|(a, b)| a + b).collect();
    let mut a = if !hht.initial_accel {
        vec![0.0; n]
    } else {
        let Ok(mf) = factor_gen(&mm, &blocks, unsym)? else { return fail("*DYNAMIC: massmatrisen är singulär") };
        let mut b: Vec<f64> = (0..n).map(|q| f_old[q] - ku[q] - cv[q]).collect();
        mf.solve(&mut b);
        b
    };
    // effektiv matris
    let c1 = (1.0 + alpha) * gamma * h;
    let c2 = (1.0 + alpha) * beta * h * h;
    let eff = Gen::combine(&[(&mm, 1.0), (&cm, c1), (&ci, gamma * h), (&km, c2), (&ki, beta * h * h)], n);
    let Ok(fac) = factor_gen(&eff, &blocks, unsym)? else { return fail("*DYNAMIC: effektiva systemmatrisen är singulär") };
    let sched = super::dynamic::output_schedule(ctx.step, period, super::dynamic::Schedule::Every(1))?;
    let wants = |inc: usize, t: f64| match sched {
        super::dynamic::Schedule::Every(k) => inc % k == 0,
        super::dynamic::Schedule::Interval(dt) => dt > 0.0 && ((t / dt) - (t / dt).round()).abs() < 1e-6,
    };
    // område gaps1: automatisk inkrementering (utan DIRECT) med samma regler som *DYNAMIC
    // (`dynamic_incr.rs`): halvstegsresidualen, ökning med D_M och kortning
    // mot utdatatidpunkter
    let auto = !kw.has("DIRECT");
    let tc = super::dynamic::TimeControls::from_keyword(&kw)?;
    let plan = super::dynamic::incr::OutputPlan::from_step(model, ctx.step, period)?;
    let mut inc_ctl = super::dynamic::incr::Incrementer::new(
        if auto { if tc.conservative { super::dynamic::incr::Mode::Conservative } else { super::dynamic::incr::Mode::Aggressive } } else { super::dynamic::incr::Mode::Fixed },
        super::dynamic::incr::IncParams::from_step(ctx.step)?,
        h,
        tc.dt_min.min(h),
        if auto { tc.dt_max } else { h },
        period,
        tc.half_tol,
        super::dynamic::characteristic_length(model),
        true,
        if auto { plan.points.clone() } else { Vec::new() },
    );
    let eps = 1e-9 * period;
    let mut t = 0.0;
    let mut frames = Vec::new();
    let emit = |ctx: &StepCtx, t: f64, u: &[f64], v: &[f64], a: &[f64], inc: usize| -> Result<FrameData> {
        let un = sys.to_nodes(u, &[]);
        let vn = sys.to_nodes(v, &[]);
        let an = sys.to_nodes(a, &[]);
        let mut fr = frame(ctx, &sys, &un, t, -1, format!("Increment {inc}: Step Time = {t:.4}"))?;
        super::dynamic::add_va(&mut fr, model.dim(), &vn, &an);
        Ok(fr)
    };
    if ctx.opts.all_frames {
        frames.push(emit(ctx, 0.0, &u, &v, &a, 0)?);
    }
    // fältet för varje ekvation (halvstegsresidualen per fält)
    let fields: Vec<crate::fem::dof::Field> = (0..n).map(|q| crate::fem::dof::field(crate::fem::dof::SLOT_DOF[sys.dm.owner[q].1])).collect();
    // HHT-termen (K u + C v − F) vid inkrementets och föregående inkrements början
    let mut r_start: Vec<f64> = {
        let (k0u, c0v) = (km.mul(&u), cm.mul(&v));
        (0..n).map(|q| k0u[q] + c0v[q] - f_old[q]).collect()
    };
    let mut r_prev = r_start.clone();
    let mut fac_h = h;
    let mut fac_cur = fac;
    let mut inc = 0usize;
    let mut hk = inc_ctl.next(0.0);
    while t < period - eps {
        if inc >= ctx.step.max_increments {
            return fail(format!("steget nådde största antal inkrement ({}) vid stegtid {t:.4e} – öka INC= på *STEP", ctx.step.max_increments));
        }
        // område gaps1: NONREFLECTING=IMPROVED – randen med cos θ ur de förutsagda trycken
        let imp_k = if sys.improved.is_empty() {
            None
        } else {
            let pp: Vec<f64> = (0..n).map(|q| u[q] + hk * v[q] + 0.5 * hk * hk * a[q]).collect();
            let pn = sys.to_nodes(&pp, &[]);
            let pnod: Vec<f64> = pn.iter().map(|x| x[PS]).collect();
            let (mut cs, mut ks) = (Vec::new(), Vec::new());
            for b in &sys.improved {
                let (c, k) = b.matrices(&sys, &pnod)?;
                cs.push(c);
                ks.push(k);
            }
            let mut tc: Vec<(&Gen, f64)> = vec![(&ci, 1.0)];
            tc.extend(cs.iter().map(|g| (g, 1.0)));
            let mut tk: Vec<(&Gen, f64)> = vec![(&ki, 1.0)];
            tk.extend(ks.iter().map(|g| (g, 1.0)));
            Some((Gen::combine(&tc, n), Gen::combine(&tk, n)))
        };
        let (ci, ki) = match &imp_k {
            Some((c, k)) => (c, k),
            None => (&ci, &ki),
        };
        if (hk - fac_h).abs() > 1e-12 * fac_h || imp_k.is_some() {
            let e = Gen::combine(&[(&mm, 1.0), (&cm, (1.0 + alpha) * gamma * hk), (ci, gamma * hk), (&km, (1.0 + alpha) * beta * hk * hk), (ki, beta * hk * hk)], n);
            let Ok(fk) = factor_gen(&e, &blocks, unsym)? else { return fail("*DYNAMIC: effektiva systemmatrisen är singulär") };
            fac_cur = fk;
            fac_h = hk;
        }
        let t1 = t + hk;
        let f_new = force(t1)?;
        // prediktor
        let up: Vec<f64> = (0..n).map(|q| u[q] + hk * v[q] + hk * hk * (0.5 - beta) * a[q]).collect();
        let vp: Vec<f64> = (0..n).map(|q| v[q] + hk * (1.0 - gamma) * a[q]).collect();
        // M a₁ + (1+α)(C v₁ + K u₁) − α(C v₀ + K u₀) + C_i v₁ + K_i u₁ = (1+α) F₁ − α F₀
        // (C_i, K_i: impedansränderna)
        let kup = km.mul(&up);
        let cvp = cm.mul(&vp);
        let ku0 = km.mul(&u);
        let cv0 = cm.mul(&v);
        let civp = ci.mul(&vp);
        let kiup = ki.mul(&up);
        let mut b: Vec<f64> = (0..n)
            .map(|q| (1.0 + alpha) * f_new[q] - alpha * f_old[q] - (1.0 + alpha) * (kup[q] + cvp[q]) + alpha * (ku0[q] + cv0[q]) - civp[q] - kiup[q])
            .collect();
        fac_cur.solve(&mut b);
        let a1 = b;
        let u1: Vec<f64> = (0..n).map(|q| up[q] + beta * hk * hk * a1[q]).collect();
        let v1: Vec<f64> = (0..n).map(|q| vp[q] + gamma * hk * a1[q]).collect();
        if auto {
            // halvstegsresidualen R½ = M a½ + (1+α)(K u½ + C v½ − F₁) − α·½(r₀ + r₋₁) + C_i v½ + K_i u½
            let hh = 0.5 * hk;
            let ah: Vec<f64> = (0..n).map(|q| 0.5 * (a[q] + a1[q])).collect();
            let uh: Vec<f64> = (0..n).map(|q| u[q] + hh * v[q] + hh * hh * ((0.5 - beta) * a[q] + beta * ah[q])).collect();
            let vh: Vec<f64> = (0..n).map(|q| v[q] + hh * ((1.0 - gamma) * a[q] + gamma * ah[q])).collect();
            let (mah, kuh, cvh, civh, kiuh) = (mm.mul(&ah), km.mul(&uh), cm.mul(&vh), ci.mul(&vh), ki.mul(&uh));
            let mut s: std::collections::BTreeMap<crate::fem::dof::Field, f64> = std::collections::BTreeMap::new();
            for q in 0..n {
                let r = mah[q] + (1.0 + alpha) * (kuh[q] + cvh[q] - f_new[q]) - alpha * 0.5 * (r_start[q] + r_prev[q]) + civh[q] + kiuh[q];
                let e = s.entry(fields[q]).or_default();
                *e = e.max(r.abs());
            }
            // tidsmedelvärdet av flödet (för HALFINC SCALE FACTOR)
            let mut tavg: std::collections::BTreeMap<crate::fem::dof::Field, f64> = std::collections::BTreeMap::new();
            for q in 0..n {
                let e = tavg.entry(fields[q]).or_default();
                *e = e.max(f_new[q].abs().max((kup[q] + cvp[q]).abs()));
            }
            let tol = inc_ctl.tolerance(&tavg);
            if let super::dynamic::incr::Verdict::Retry(hn) = inc_ctl.check(&s, &tol)? {
                hk = hn;
                continue;
            }
            inc_ctl.accepted(t1, s, &tol, 1);
        }
        let (k1u, c1v) = (km.mul(&u1), cm.mul(&v1));
        r_prev = std::mem::replace(&mut r_start, (0..n).map(|q| k1u[q] + c1v[q] - f_new[q]).collect());
        u = u1;
        v = v1;
        a = a1;
        f_old = f_new;
        t = t1;
        inc += 1;
        let last = t >= period - eps;
        let want = if auto { plan.write_frame(inc, t, eps) } else { wants(inc, t) };
        if ctx.opts.all_frames && (want || last) {
            frames.push(emit(ctx, t, &u, &v, &a, inc)?);
        }
        if last && !ctx.opts.all_frames {
            frames.push(emit(ctx, t, &u, &v, &a, inc)?);
        }
        if !last {
            hk = inc_ctl.next(t);
        }
    }
    let nsteps = inc;
    // tillståndet efter steget
    let un = sys.to_nodes(&u, &[]);
    for nd in 0..nn {
        for s in 0..SLOTS {
            if ctx.prep.active[nd] & (1 << s) != 0 && sys.cidx[nd][s] == NONE {
                ctx.state.u[nd][s] = un[nd][s];
            }
        }
    }
    ctx.state.time += period;
    ctx.out.increments = nsteps;
    ctx.out.frames.extend(frames);
    Ok(())
}
