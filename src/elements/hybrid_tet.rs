//! C3D4H: linjär hybridtetraeder med kontinuerligt nodvis tryckfält.
//!
//! Formulering (*ELASTIC och *HYPERELASTIC NEO HOOKE, även med D1 = 0):
//!
//! - Trycket q (positivt vid kompression) interpoleras linjärt från
//!   hörnnoderna och är **kontinuerligt** mellan elementen. Det lagras
//!   på en intern nod per nätnod (frihetsgrad 1): `setup`
//!   (anropas när .inp-filen är läst) skapar en trycknod per hörnnod och
//!   material, med nummer −(3·10⁹ + 64·nod + materialindex), som läggs sist
//!   i elementets nodlista (noderna 5–8, frihetsgrad 1). Trycknoden är en
//!   förskjutningsfrihetsgrad, så att Newtons konvergenskriterier per fält
//!   gäller även tryckekvationerna; variabeln skalas med μ (q = μ · u₅..₈),
//!   så att den blir dimensionslös och ekvationerna får kraftliknande
//!   storlek. Systemet blir en sadelpunkt, och modeller med C3D4H löses med
//!   den indefinita lösaren (`auto_solver`, anropas i `steps::run`).
//! - Elementet har en integrationspunkt; där används elementets medeltryck
//!   q̄ = Σ qᵢ / 4. Med m = ∫N dV = V/4 per nod och tryckmassan
//!   M = ∫NNᵀ dV = V/20 (1 + δᵢⱼ) är blocken (linjär elasticitet, μ, K)
//!
//!   K_uu = V Bᵀ D_dev B + β K V d dᵀ,   K_uq = −(1 − β) d mᵀ,
//!   K_qq = −(1 − β)/K [m mᵀ/V + α (M − m mᵀ/V)] − c α/μ (M − m mᵀ/V)
//!
//!   med d = volymtöjningsraden (tr B), β = 10⁻⁶ (som de andra
//!   hybridelementen, inte för hyperelastiska material), α = 0,005 och c = 1
//!   för helt inkompressibla material (ν = 0,5, D1 = 0), annars 0. Termen
//!   med α är en stabilisering av trycket mot elementets medelvärde
//!   (polynomprojektion); den gör sadelpunktsproblemet entydigt även för
//!   helt inkompressibla material (`tests/hybrid2.rs`).
//! - Hydrostatisk spänning i punkten: (1 − β)(−q̄) + β (−p_mat).
//!
//! Andra material vid små töjningar (plasticitet m.m.): spänningen räknas av
//! materialet och dess hydrostatiska del byts ut mot (1 − β)(−q̄), med den
//! elastiska kompressionsmodulen K i tryckekvationen
//! (1 − β) m (p_mat − q̄)/K (exakt för isotrop elasticitet + tryckoberoende
//! plasticitet, t.ex. Mises).
//!
//! NLGEOM med hyperelastiskt material: energin
//! Π = Σₑ V₀ [W_dev(F) + p (J − 1) − U*(p)] med p = −q̄ och U* volymenergins
//! Legendretransform (U* = 0 för D1 = 0), samma stabilisering som ovan med
//! K = K₀. Den deviatoriska delen räknas med `hybrid::nl_eval_hist`
//! (Lagrangemultiplikator med konstant bas). *ELASTIC/plasticitet vid NLGEOM
//! stöds inte.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip};
use super::solid_higher::{self as sh, Solid};
use crate::fem::dof::{Dof, U1, U2, U3};
use crate::fem::model::{DistLoad, FeModel};
use crate::materials::incompressible_elastic::is_incompressible;
use crate::materials::{MaterialModel, StressMode};
use crate::model::{fail, Result};
use std::collections::HashMap;

pub struct NodalTet;

pub static C3D4H: NodalTet = NodalTet;

pub static TYPES: &[&dyn ElementType] = &[&C3D4H];

/// Stabiliseringskonstanten α (oberoende av ν).
pub const ALPHA: f64 = 0.005;

/// Nummer för trycknoden till nod `label` och material nummer `m`.
pub fn pressure_label(label: i64, m: usize) -> i64 {
    -(3_000_000_000 + 64 * label + m as i64)
}

fn base() -> &'static Solid {
    &sh::C3D4
}

/// Lägger till trycknoderna (en per hörnnod och material) sist i varje
/// C3D4H-elements nodlista. Anropas från `inp` när filen är läst.
pub fn setup(model: &mut FeModel) -> Result<()> {
    let mut mats: Vec<String> = Vec::new();
    let mut made: HashMap<(usize, usize), usize> = HashMap::new();
    for e in 0..model.elements.len() {
        if model.elements[e].etype.name() != "C3D4H" || model.elements[e].nodes.len() != 4 {
            continue;
        }
        let mat = model.elements[e].section.map(|s| model.sections[s].material.clone()).unwrap_or_default();
        let mi = match mats.iter().position(|m| *m == mat) {
            Some(i) => i,
            None => {
                mats.push(mat);
                mats.len() - 1
            }
        };
        if mi >= 64 {
            return fail("C3D4H: högst 64 material med C3D4H-element stöds");
        }
        let mut extra = Vec::with_capacity(4);
        for a in 0..4 {
            let n = model.elements[e].nodes[a];
            let id = match made.get(&(n, mi)) {
                Some(&id) => id,
                None => {
                    let (inst, label, x) = (model.node_inst[n], model.node_label[n], model.nodes[n]);
                    let id = model.add_node(inst, pressure_label(label, mi), x);
                    made.insert((n, mi), id);
                    id
                }
            };
            extra.push(id);
        }
        model.elements[e].nodes.extend(extra);
    }
    Ok(())
}

/// Har modellen element med nodvisa tryckfrihetsgrader (C3D4H)?
pub fn model_has_nodal_pressure(model: &FeModel) -> bool {
    model.elements.iter().any(|e| e.etype.name() == "C3D4H")
}

/// Den indefinita lösaren utan varning om negativa egenvärden (sadelpunkts-
/// systemet har lika många negativa egenvärden som tryckfrihetsgrader).
struct QuietIndefinite;

static QUIET: QuietIndefinite = QuietIndefinite;

impl crate::solver::LinearSolver for QuietIndefinite {
    fn name(&self) -> &'static str {
        "indefinite"
    }
    fn prepare(&self, g: &crate::solver::Graph) -> Result<Box<dyn crate::solver::SystemMatrix>> {
        crate::solver::indefinite::prepare_quiet(g)
    }
}

/// Lösaren för ett steg: den indefinita i stället för standardlösaren
/// (`direct`) när modellen har C3D4H, om användaren inte valt lösare.
pub fn auto_solver(model: &FeModel, solver: &'static dyn crate::solver::LinearSolver, forced: bool) -> &'static dyn crate::solver::LinearSolver {
    if !forced && solver.name() == "direct" && model_has_nodal_pressure(model) {
        return &QUIET;
    }
    solver
}

/// Hur elementet räknas för materialet.
enum Kind {
    /// Linjär elasticitet (även hyperelastiskt utan NLGEOM): μ, 1/K, β, inkompressibelt.
    Linear { mu: f64, kinv: f64, beta: f64, incomp: bool },
    /// Annat material vid små töjningar: elastisk kompressionsmodul K, μ, inkompressibelt.
    General { k: f64, mu: f64, incomp: bool },
    /// Hyperelastiskt vid NLGEOM.
    Hyper,
}

fn kind(ctx: &ElemCtx, mat: &dyn MaterialModel) -> Result<Kind> {
    if let Some(h) = mat.hyperelastic() {
        if ctx.nlgeom {
            if h.kirchhoff_full(&crate::kinematics::tensor::I3).is_some() {
                return fail(format!("{}: *HYPERFOAM stöds inte i C3D4H", ctx.label));
            }
            return Ok(Kind::Hyper);
        }
        let kinv = if h.incompressible() { 0.0 } else { 1.0 / h.k0() };
        return Ok(Kind::Linear { mu: h.mu0(), kinv, beta: 0.0, incomp: h.incompressible() });
    }
    if ctx.nlgeom {
        return not_implemented("NLGEOM med andra material än *HYPERELASTIC", "C3D4H");
    }
    let Some((e, nu)) = mat.elastic_moduli() else {
        return fail(format!("{}: C3D4H kräver ett material med isotrop elasticitet (E, ν)", ctx.label));
    };
    let mu = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    let incomp = is_incompressible(nu);
    if mat.is_linear() {
        return Ok(Kind::Linear { mu, kinv: 1.0 / k, beta: super::hybrid::BETA, incomp });
    }
    Ok(Kind::General { k, mu, incomp })
}

/// Skalan s i q = s · u (trycknodernas frihetsgrad): skjuvmodulen.
fn scale(mat: &dyn MaterialModel) -> f64 {
    if let Some(h) = mat.hyperelastic() {
        return h.mu0();
    }
    mat.elastic_moduli().map_or(1.0, |(e, nu)| e / (2.0 * (1.0 + nu)))
}

/// Delar elementvektorn (12 förskjutningar, 4 skalade tryck) i förskjutningar och tryck.
fn split(v: &[f64], s: f64) -> (Vec<f64>, [f64; 4]) {
    let mut q = [0.0; 4];
    for a in 0..4 {
        q[a] = s * v.get(12 + a).copied().unwrap_or(0.0);
    }
    (v[..12].to_vec(), q)
}

/// Index i elementvektorn för tryck i nod a.
fn iq(a: usize) -> usize {
    12 + a
}

/// Kontext med bara hörnnoderna.
fn geo<'a>(ctx: &ElemCtx<'a>) -> ElemCtx<'a> {
    let mut c = ctx.clone();
    c.x.truncate(4);
    c
}

/// Stabiliseringsmatrisen S = M − m mᵀ/V (S_ii = 3V/80, S_ij = −V/80).
fn smat(v: f64) -> [[f64; 4]; 4] {
    let mut s = [[-v / 80.0; 4]; 4];
    for (i, r) in s.iter_mut().enumerate() {
        r[i] = 3.0 * v / 80.0;
    }
    s
}

/// Punkten, B och volymtöjningsraden.
fn kin(ctx: &ElemCtx) -> Result<(Vec<Ip>, Vec<Vec<f64>>, Vec<f64>, f64)> {
    let el = base();
    let ips = el.ips(ctx)?;
    let bs = el.bmats(&ctx.x, &ips);
    let b = &bs[0];
    let dv: Vec<f64> = (0..12).map(|c| b[c] + b[12 + c] + b[24 + c]).collect();
    let v = ips[0].w;
    Ok((ips, bs, dv, v))
}

/// Residual och tangent för hela elementet (fysiskt tryck q).
struct Asm {
    f: Vec<f64>,
    k: Vec<f64>,
}

impl Asm {
    fn new() -> Self {
        Asm { f: vec![0.0; 16], k: vec![0.0; 256] }
    }
    fn kadd(&mut self, i: usize, j: usize, v: f64) {
        self.k[i * 16 + j] += v;
    }
    /// Byter till den skalade tryckvariabeln u = q/s (rader och kolumner × s).
    fn scaled(mut self, s: f64) -> Self {
        for i in 12..16 {
            self.f[i] *= s;
            for j in 0..16 {
                self.k[i * 16 + j] *= s;
                self.k[j * 16 + i] *= s;
            }
        }
        self
    }
}

/// Tryckblockets bidrag −c_m (m mᵀ/V) − c_s S och dess del av residualen.
///
/// Med `fs_old` (små töjningar) är stabiliseringens kraft ett tillstånd som
/// uppdateras med hela trycket i varje inkrement,
/// F_s = F_s,n − c_s S q: resultatet beror då på antalet inkrement (ett
/// inkrement ger den totala formen). Returnerar det nya F_s. Utan `fs_old`
/// (hyperelastiskt vid NLGEOM) används den totala formen −c_s S q.
fn pressure_block(a: &mut Asm, q: &[f64; 4], v: f64, c_m: f64, c_s: f64, tangent: bool, fs_old: Option<&[f64]>) -> [f64; 4] {
    let s = smat(v);
    let mut fs = [0.0; 4];
    for i in 0..4 {
        fs[i] = fs_old.map_or(0.0, |f| f[i]);
        for j in 0..4 {
            fs[i] -= c_s * s[i][j] * q[j];
            a.f[iq(i)] += -c_m * v / 16.0 * q[j];
            if tangent {
                a.kadd(iq(i), iq(j), -c_m * v / 16.0 - c_s * s[i][j]);
            }
        }
        a.f[iq(i)] += fs[i];
    }
    fs
}

/// Stabiliseringskraften vid inkrementets början (sist i tillståndet).
fn fs_of(so: &[f64], len: usize) -> [f64; 4] {
    if so.len() == len && len >= 4 {
        [so[len - 4], so[len - 3], so[len - 2], so[len - 1]]
    } else {
        [0.0; 4]
    }
}

/// Kopplingen K_uq = K_qu = −(1 − β) d V/4.
fn coupling(a: &mut Asm, dv: &[f64], v: f64, beta: f64) {
    for i in 0..12 {
        for k in 0..4 {
            let c = -(1.0 - beta) * v * dv[i] / 4.0;
            a.kadd(i, iq(k), c);
            a.kadd(iq(k), i, c);
        }
    }
}

impl NodalTet {
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, ctx: &ElemCtx, u: &[f64], q: &[f64; 4], so: &[f64], tangent: bool, mu: f64, kinv: f64, beta: f64, incomp: bool) -> Result<(Asm, Vec<f64>, Vec<f64>)> {
        let (_, bs, dv, v) = kin(ctx)?;
        let b = &bs[0];
        let mut a = Asm::new();
        // deviatoriskt D (ingenjörsskjuvning)
        let mut dd = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                dd[i][j] = 2.0 * mu * (if i == j { 1.0 } else { 0.0 } - 1.0 / 3.0);
            }
            dd[3 + i][3 + i] = mu;
        }
        let kb = if kinv > 0.0 { 1.0 / kinv } else { 0.0 };
        let eps = common::strain(b, u);
        let ev: f64 = eps[0] + eps[1] + eps[2];
        let qbar: f64 = q.iter().sum::<f64>() / 4.0;
        // spänning: deviatorisk + (1 − β)(−q̄) + β K ε_v
        let mut sig: Vec<f64> = (0..6).map(|r| (0..6).map(|m| dd[r][m] * eps[m]).sum()).collect();
        for s in sig.iter_mut().take(3) {
            *s += -(1.0 - beta) * qbar + beta * kb * ev;
        }
        for c in 0..12 {
            a.f[c] += v * (0..6).map(|r| b[r * 12 + c] * sig[r]).sum::<f64>();
        }
        if tangent {
            for i in 0..12 {
                for j in 0..12 {
                    let mut s = 0.0;
                    for r in 0..6 {
                        for m in 0..6 {
                            s += b[r * 12 + i] * dd[r][m] * b[m * 12 + j];
                        }
                    }
                    a.kadd(i, j, v * s + beta * kb * v * dv[i] * dv[j]);
                }
            }
            coupling(&mut a, &dv, v, beta);
        }
        // tryckekvationen: (1 − β)(V/4)(−ε_v) − (1 − β)/K [m mᵀ/V + α S] q − c α/μ S q
        for k in 0..4 {
            a.f[iq(k)] += -(1.0 - beta) * v / 4.0 * ev;
        }
        let c_s = (1.0 - beta) * ALPHA * kinv + if incomp { ALPHA / mu } else { 0.0 };
        let fs = pressure_block(&mut a, q, v, (1.0 - beta) * kinv, c_s, tangent, Some(&fs_of(so, 10)));
        let mut st = sig;
        st.extend_from_slice(&fs);
        Ok((a, st, eps))
    }

    #[allow(clippy::too_many_arguments)]
    fn general(
        &self,
        ctx: &ElemCtx,
        mat: &dyn MaterialModel,
        u: &[f64],
        q: &[f64; 4],
        du: &[f64],
        so: &[f64],
        sn: &mut Vec<f64>,
        tangent: bool,
        kb: f64,
        mu: f64,
        incomp: bool,
    ) -> Result<Asm> {
        let (ips, bs, dv, v) = kin(ctx)?;
        let beta = super::hybrid::BETA;
        let stride = common::ip_stride(StressMode::ThreeD, mat);
        let som = if so.len() == stride + 5 { &so[..stride] } else { &so[..0] };
        let mut snm = Vec::new();
        let r = common::response(ctx, StressMode::ThreeD, &ips, &bs, mat, u, du, som, &mut snm, tangent)?;
        let s = &snm[..6];
        let pmat = -(s[0] + s[1] + s[2]) / 3.0;
        let qbar: f64 = q.iter().sum::<f64>() / 4.0;
        let delta = (1.0 - beta) * (pmat - qbar);
        let mut a = Asm::new();
        for c in 0..12 {
            a.f[c] += r.fint[c] + delta * v * dv[c];
        }
        if let Some(km) = &r.k {
            for i in 0..12 {
                for j in 0..12 {
                    // symmetrisk del av materialtangenten minus den ersatta volymdelen
                    let kij = 0.5 * (km[i * 12 + j] + km[j * 12 + i]) - (1.0 - beta) * kb * v * dv[i] * dv[j];
                    a.kadd(i, j, kij);
                }
            }
            coupling(&mut a, &dv, v, beta);
        }
        for k in 0..4 {
            a.f[iq(k)] += (1.0 - beta) * v / 4.0 * pmat / kb;
        }
        let kinv = 1.0 / kb;
        let c_s = (1.0 - beta) * ALPHA * kinv + if incomp { ALPHA / mu } else { 0.0 };
        let fs = pressure_block(&mut a, q, v, (1.0 - beta) * kinv, c_s, tangent, Some(&fs_of(so, stride + 5)));
        *sn = snm;
        sn.push(delta);
        sn.extend_from_slice(&fs);
        Ok(a)
    }

    #[allow(clippy::too_many_arguments)]
    fn hyper(
        &self,
        ctx: &ElemCtx,
        h: &dyn crate::materials::hyperelastic_split::HyperModel,
        u: &[f64],
        q: &[f64; 4],
        so: &[f64],
        sn: &mut Vec<f64>,
        tangent: bool,
    ) -> Result<Asm> {
        let (ips, _, _, v) = kin(ctx)?;
        let qbar: f64 = q.iter().sum::<f64>() / 4.0;
        let ph = -qbar;
        let r = super::hybrid::nl_eval_hist(
            StressMode::ThreeD,
            &ips,
            &ctx.x,
            super::hybrid::Pressure::Constant,
            0,
            false,
            h,
            u,
            tangent,
            None,
            Some((&[ph], 1.0)),
            Some(so),
        )?;
        // volymtöjningen v(p) ur U'(1 + v) = p och dv/dp = 1/U''
        let (vp, dvdp) = if h.incompressible() { (0.0, 0.0) } else { conjugate(h, ph)? };
        let mut a = Asm::new();
        a.f[..12].copy_from_slice(&r.fint[..12]);
        // ∂Π/∂p = V₀ (J − 1 − v(p)) med fint[12] = V₀ (J − 1); ∂p/∂qₖ = −1/4
        let dpi = r.fint[12] - v * vp;
        for k in 0..4 {
            a.f[iq(k)] += -0.25 * dpi;
        }
        if let Some(km) = &r.k {
            for i in 0..12 {
                for j in 0..12 {
                    a.kadd(i, j, km[i * 13 + j]);
                }
                for k in 0..4 {
                    a.kadd(i, iq(k), -0.25 * km[i * 13 + 12]);
                    a.kadd(iq(k), i, -0.25 * km[12 * 13 + i]);
                }
            }
            for i in 0..4 {
                for j in 0..4 {
                    a.kadd(iq(i), iq(j), -v * dvdp / 16.0);
                }
            }
        }
        // stabiliseringen som vid små töjningar (K = K₀)
        let c_s = if h.incompressible() { ALPHA / h.mu0() } else { ALPHA / h.k0() };
        pressure_block(&mut a, q, v, 0.0, c_s, tangent, None);
        *sn = r.state;
        Ok(a)
    }

    fn response(&self, ctx: &ElemCtx, ue: &[f64], due: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let mat = ctx.material("C3D4H")?;
        let g = geo(ctx);
        let s = scale(mat);
        let (u, q) = split(ue, s);
        let du = &due[..12.min(due.len())];
        let a = match kind(&g, mat)? {
            Kind::Linear { mu, kinv, beta, incomp } => {
                let (a, sig, _) = self.linear(&g, &u, &q, so, tangent, mu, kinv, beta, incomp)?;
                *sn = sig;
                a
            }
            Kind::General { k, mu, incomp } => self.general(&g, mat, &u, &q, du, so, sn, tangent, k, mu, incomp)?,
            Kind::Hyper => self.hyper(&g, mat.hyperelastic().unwrap(), &u, &q, so, sn, tangent)?,
        };
        let a = a.scaled(s);
        let mut k = a.k;
        if tangent {
            crate::kinematics::continuum::symmetrize(&mut k, 16);
        }
        Ok(ElemResponse { fint: a.f, k: tangent.then_some(k) })
    }

    /// Bäddar in en matris för de tolv förskjutningarna i elementets 16 frihetsgrader.
    fn embed(m12: &[f64]) -> Vec<f64> {
        let mut k = vec![0.0; 256];
        for i in 0..12 {
            k[i * 16..i * 16 + 12].copy_from_slice(&m12[i * 12..(i + 1) * 12]);
        }
        k
    }
}

/// Volymtöjningen v och dv/dp = 1/U''(1 + v) för hydrostatisk spänning p
/// (U'(1 + v) = p), med Newton.
fn conjugate(h: &dyn crate::materials::hyperelastic_split::HyperModel, p: f64) -> Result<(f64, f64)> {
    let mut v = p / h.k0();
    for _ in 0..50 {
        let (u1, u2) = h.vol(1.0 + v);
        if !(u2 > 0.0) {
            return fail("C3D4H: volymenergin är inte konvex vid trycket");
        }
        let dv = (p - u1) / u2;
        v += dv;
        if dv.abs() <= 1e-14 * (1.0 + v.abs()) {
            break;
        }
    }
    let (_, u2) = h.vol(1.0 + v);
    Ok((v, 1.0 / u2))
}

impl ElementType for NodalTet {
    fn name(&self) -> &'static str {
        "C3D4H"
    }
    fn family(&self) -> Family {
        Family::Continuum3D
    }
    fn dim(&self) -> usize {
        3
    }
    fn n_nodes(&self) -> usize {
        8
    }
    fn n_connectivity(&self) -> usize {
        4
    }
    fn node_dofs(&self, i: usize) -> &'static [Dof] {
        if i < 4 { &[U1, U2, U3] } else { &[U1] }
    }
    fn n_ips(&self) -> usize {
        1
    }
    fn vtk_type(&self) -> u8 {
        base().vtk_type()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        base().faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mat = ctx.material("C3D4H")?;
        if let Some(h) = mat.hyperelastic() {
            return Ok(vec![0.0; 6 + h.n_hist()]);
        }
        // spänning och stabiliseringskraften F_s (4)
        if mat.is_linear() {
            return Ok(vec![0.0; 10]);
        }
        // materialets tillstånd, den hydrostatiska korrektionen och F_s
        let mut s = common::init_state(1, StressMode::ThreeD, mat);
        s.extend([0.0; 5]);
        Ok(s)
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut sn = Vec::new();
        let z = vec![0.0; 16];
        let so = self.init_state(ctx)?;
        Ok(self.response(ctx, &z, &z, &so, &mut sn, true)?.k.unwrap())
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        if ctx.x.len() != 8 {
            return fail(format!("element {} (C3D4H) saknar trycknoder", ctx.label));
        }
        self.response(ctx, u, du, so, sn, tangent)
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        Ok(Self::embed(&base().mass(&geo(ctx), kind)?))
    }

    fn geometric_stiffness(&self, ctx: &ElemCtx, _u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        let g = geo(ctx);
        let mat = ctx.material("C3D4H")?;
        let ips = base().ips(&g)?;
        let mut s = state.get(0..6).map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; 6]);
        // allmänna material: spänningen plus den hydrostatiska korrektionen
        let general = !mat.is_linear() && mat.hyperelastic().is_none();
        let d = if general && state.len() > 10 { state[state.len() - 5] } else { 0.0 };
        for v in s.iter_mut().take(3) {
            *v += d;
        }
        // samma uttryck som C3D4 (`geometric::continuum`), med spänningen först i tillståndet
        let mut st = vec![0.0; common::ip_stride(StressMode::ThreeD, mat)];
        st[..6].copy_from_slice(&s);
        Ok(Self::embed(&super::geometric::continuum(StressMode::ThreeD, &ips, mat, &st)?))
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let ug: Vec<f64> = if u.len() >= 12 { u[..12].to_vec() } else { vec![0.0; 12] };
        let mut f = base().distributed_load(&geo(ctx), load, &ug)?;
        f.resize(16, 0.0);
        Ok(f)
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut w = base().volume_weights(&geo(ctx))?;
        w.resize(8, 0.0);
        Ok(w)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let g = geo(ctx);
        let (sl, el) = common::stress_labels(StressMode::ThreeD);
        let mut s = IpField::new("S", sl);
        let mut e = IpField::new("E", el);
        let mat = ctx.material("C3D4H")?;
        let ug = &u[..12];
        if !mat.is_linear() && mat.hyperelastic().is_none() {
            // allmänna material: materialets utdata (PEEQ, ...) och den hydrostatiska korrektionen
            let (_, bs, _, _) = kin(&g)?;
            let n = state.len().saturating_sub(5);
            let mut fields = common::output(StressMode::ThreeD, &bs, mat, ug, &state[..n]);
            let d = state.get(n).copied().unwrap_or(0.0);
            for f in fields.iter_mut().filter(|f| f.name == "S") {
                for p in f.points.iter_mut() {
                    for c in 0..3 {
                        p.2[c] += d;
                    }
                }
            }
            return Ok(fields);
        }
        let mut sig = state.get(0..6).map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; 6]);
        sig.resize(6, 0.0);
        let eps = if ctx.nlgeom {
            vec![0.0; 6]
        } else {
            let (_, bs, _, _) = kin(&g)?;
            common::strain(&bs[0], ug)
        };
        s.points.push((1, 0, sig));
        e.points.push((1, 0, eps));
        Ok(vec![s, e])
    }
}
