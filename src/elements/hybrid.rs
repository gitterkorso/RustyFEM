//! Hybridelement (blandad förskjutning–tryck-formulering): C3D8H, C3D10H,
//! C3D20H, C3D20RH, CPE4H, CPE8H, CAX4H, CAX8H m.fl.
//!
//! C3D4H finns inte här: det använder ett kontinuerligt nodvis tryckfält
//! (en tryckvariabel per hörnnod i nätet), som kräver globala tryckfrihetsgrader.
//!
//! Formuleringen är den tre-fältsformulering för nästan och helt
//! inkompressibla material som Simo, Taylor & Pister (1985) och Simo & Taylor
//! (1991) gav (se även Bonet & Wood, kap. 8, och de Souza Neto m.fl., kap. 15):
//! utöver förskjutningen interpoleras trycket p och volymkvoten θ oberoende
//! inom elementet, med samma (diskontinuerliga) bas ψ – konstant (C3D8H,
//! C3D10H, CPE4H, CAX4H, CPE3H) eller linjär i referenskoordinaterna (C3D20H,
//! C3D20RH, CPE8H, CAX8H, CPE6H, ...), med ett bestämt antal tryckvariabler
//! per element. Energin är
//!
//!   Π(u, θ, p) = Σ_q w_q [ W_dev(F̄_q) + U(θ_q) + p_q (J_q − θ_q) ] (+ timglasterm, se nedan)
//!
//! och stationaritet i p och θ ger θ = L2-projektionen av J på tryckbasen och
//! p = projektionen av U′(θ). Båda kondenseras på elementnivå, så elementet
//! har bara förskjutningar i det globala systemet. F̄ = (θ/J)^(1/n) F i de n
//! volymriktningarna (n = 3 i 3D och axisymmetri, n = 2 vid plan töjning)
//! för element med konstant tryck; för element med linjärt tryck används F
//! i den deviatoriska delen.
//!
//! För element med konstant tryck och kompressibelt material blir detta
//! exakt F-bar-elementet (`kinematics/fbar_hyper.rs`), och C3D8H/CPE4H ger
//! samma resultat som C3D8/CPE4, inklusive volymtimglasstyvheten för
//! hyperelastiska material.
//!
//! **Helt inkompressibla material** (*HYPERELASTIC med D1 = 0, *ELASTIC med
//! ν = 0,5) hanteras med en straffkompressionsmodul K = 10⁸ μ0 i U(θ) i
//! stället för en global Lagrangemultiplikator; volymkvoten avviker då från 1
//! med p/K ~ 10⁻⁸ och förskjutningarna från den exakta lösningen med samma
//! storleksordning.
//!
//! **Linjär analys** (utan NLGEOM, *ELASTIC): den linjära motsvarigheten med
//! energin Σ w [μ |dev ε̄|² + K/2 θ²], där θ är projektionen av tr ε och ε̄
//! har volymdelen ersatt med θ i de n riktningarna (B-bar för konstant tryck).
//!
//! **Andra material i C3D8H, CPE4H och CAX4H** (plasticitet, *ELASTIC vid
//! NLGEOM, ...): de ger samma förskjutningar som
//! C3D8, CPE4 och CAX4 (selektivt reducerad integrering), så
//! elementen räknas då med motsvarande första ordningens element
//! (`forsta_ordningen`). Helt inkompressibel *ELASTIC (ν = 0,5) och de övriga
//! hybridelementen stöder bara linjärt elastiska material utan NLGEOM och
//! hyperelastiska material.
//!
//! Tillstånd per integrationspunkt: Cauchyspänning,
//! vid NLGEOM följd av NLGEOM-tillägget med LE (`kinematics::push_ext`).

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField, MassKind};
use super::common::{self, Ip};
use super::solid_higher::shapes::Shape;
use super::solid_higher::{radius, RuleKind, Solid, Tech};
use crate::fem::dof::Dof;
use crate::fem::model::DistLoad;
use crate::kinematics::continuum::{log_strain_out, symmetrize, tensor_to_vec};
use crate::kinematics::fbar_hyper::hourglass_coef;
use crate::kinematics::tensor::{self as t, M3};
use crate::materials::hyperelastic_split::{self as hs, HyperModel};
use crate::materials::{MaterialModel, StressMode};
use crate::model::{fail, Result};

/// Tryckinterpolation inom elementet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    Constant,
    Linear,
}

/// Straffkompressionsmodul för helt inkompressibla material, relativt μ0.
pub const PENALTY: f64 = 1e8;

/// Andel β av volymenergin som integreras med elementets fulla regel i
/// stället för med den projicerade volymkvoten: U = (1 − β) U(θ) + β U(J).
/// Hybridelementen har den här lilla lokala kompressibiliteten i linjär
/// analys med *ELASTIC (hydrostatisk spänning K θ̄ + β K (ε_v − θ̄) i
/// punkterna; C3D6H, C3D10H, C3D15H, C3D20H, C3D20RH,
/// CPE8H och CAX8H, β = 10⁻⁶), men inte C3D8H/CPE4H/CAX4H och inte
/// hyperelastiska material vid NLGEOM.
pub const BETA: f64 = 1e-6;

pub struct Hybrid {
    /// Geometri, integrationsregel och laster (namnet är hybridelementets).
    pub base: Solid,
    pub pressure: Pressure,
    /// Antal volymriktningar i F̄ (0: den deviatoriska delen räknas med F).
    pub nbar: usize,
    /// Volymtimglasstyvhet för hyperelastiska material (som F-bar-elementen).
    pub hourglass: bool,
}

const fn hy(name: &'static str, shape: Shape, mode: StressMode, rule: RuleKind, pressure: Pressure, nbar: usize, hourglass: bool) -> Hybrid {
    Hybrid { base: Solid { name, shape, mode, rule, tech: Tech::Full }, pressure, nbar, hourglass }
}

use Pressure::{Constant as PC, Linear as PL};
use StressMode::{Axisymmetric as AX, PlaneStrain as PE, ThreeD as D3};

pub static C3D8H: Hybrid = hy("C3D8H", Shape::Hex8, D3, RuleKind::Gauss(2), PC, 3, true);
pub static C3D10H: Hybrid = hy("C3D10H", Shape::Tet10, D3, RuleKind::Tet(4), PC, 0, false);
pub static C3D6H: Hybrid = hy("C3D6H", Shape::Wedge6, D3, RuleKind::Wedge(1, 2), PC, 0, false);
pub static C3D15H: Hybrid = hy("C3D15H", Shape::Wedge15, D3, RuleKind::Wedge(3, 3), PL, 0, false);
pub static C3D20H: Hybrid = hy("C3D20H", Shape::Hex20, D3, RuleKind::Gauss(3), PL, 0, false);
pub static C3D20RH: Hybrid = hy("C3D20RH", Shape::Hex20, D3, RuleKind::Gauss(2), PL, 0, false);
pub static CPE3H: Hybrid = hy("CPE3H", Shape::Tri3, PE, RuleKind::Tri(1), PC, 2, false);
pub static CPE4H: Hybrid = hy("CPE4H", Shape::Quad4, PE, RuleKind::Gauss(2), PC, 2, true);
pub static CPE6H: Hybrid = hy("CPE6H", Shape::Tri6, PE, RuleKind::Tri(3), PL, 0, false);
pub static CPE8H: Hybrid = hy("CPE8H", Shape::Quad8, PE, RuleKind::Gauss(3), PL, 0, false);
pub static CPE8RH: Hybrid = hy("CPE8RH", Shape::Quad8, PE, RuleKind::Gauss(2), PL, 0, false);
pub static CAX3H: Hybrid = hy("CAX3H", Shape::Tri3, AX, RuleKind::Tri(1), PC, 3, false);
pub static CAX4H: Hybrid = hy("CAX4H", Shape::Quad4, AX, RuleKind::Gauss(2), PC, 4, true);
pub static CAX6H: Hybrid = hy("CAX6H", Shape::Tri6, AX, RuleKind::Tri(3), PL, 0, false);
pub static CAX8H: Hybrid = hy("CAX8H", Shape::Quad8, AX, RuleKind::Gauss(3), PL, 0, false);
pub static CAX8RH: Hybrid = hy("CAX8RH", Shape::Quad8, AX, RuleKind::Gauss(2), PL, 0, false);

pub static TYPES: &[&dyn ElementType] = &[
    &C3D6H, &C3D8H, &C3D10H, &C3D15H, &C3D20H, &C3D20RH, &CPE3H, &CPE4H, &CPE6H, &CPE8H, &CPE8RH, &CAX3H, &CAX4H, &CAX6H, &CAX8H,
    &CAX8RH,
];

/// Inställningar som kan ändras med miljövariabler vid felsökning
/// (RUSTYFEM_HYB_NBAR, RUSTYFEM_HYB_HG = 0/1).
fn knobs(h: &Hybrid) -> (usize, bool) {
    let nbar = crate::settings::with(|s| s.hybrid_nbar).unwrap_or(h.nbar);
    let hg = crate::settings::with(|s| s.hybrid_hourglass).unwrap_or(h.hourglass);
    (nbar, hg)
}

/// Tryckbasen ψ i varje integrationspunkt och M⁻¹ (M = Σ w ψ ψᵀ).
struct Basis {
    np: usize,
    psi: Vec<Vec<f64>>,
    minv: Vec<f64>,
}

fn basis(kind: Pressure, ndim: usize, ips: &[Ip], x: &[[f64; 3]], nat: Option<&[[f64; 3]]>) -> Result<Basis> {
    let mut pos: Vec<[f64; 3]> = ips.iter().map(|ip| [0, 1, 2].map(|c| ip.n.iter().zip(x).map(|(n, p)| n * p[c]).sum())).collect();
    if let Some(nat) = nat {
        // den linjära tryckbasen är linjär i de naturliga koordinaterna
        // (kontrollerat med förvrängda C3D20H, CPE8H, CAX8H och C3D15H)
        pos = nat.to_vec();
    }
    let w: f64 = ips.iter().map(|ip| ip.w).sum();
    let mut c = [0.0; 3];
    for (ip, p) in ips.iter().zip(&pos) {
        for k in 0..3 {
            c[k] += ip.w * p[k] / w;
        }
    }
    let h = x.iter().map(|p| common::norm(common::sub(*p, x[0]))).fold(0.0, f64::max).max(1e-300);
    let psi: Vec<Vec<f64>> = pos
        .iter()
        .map(|p| match kind {
            Pressure::Constant => vec![1.0],
            Pressure::Linear => std::iter::once(1.0).chain((0..ndim).map(|k| (p[k] - c[k]) / h)).collect(),
        })
        .collect();
    let np = psi[0].len();
    let mut m = vec![0.0; np * np];
    for (ip, s) in ips.iter().zip(&psi) {
        for a in 0..np {
            for b in 0..np {
                m[a * np + b] += ip.w * s[a] * s[b];
            }
        }
    }
    let minv = invert(&m, np)?;
    Ok(Basis { np, psi, minv })
}

/// Invers av en liten tät symmetrisk matris (Gauss–Jordan).
fn invert(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i * n + col].abs().total_cmp(&m[j * n + col].abs())).unwrap();
        if m[piv * n + col].abs() < 1e-300 {
            return fail("hybridelement: tryckbasen är singulär (förvrängt element?)");
        }
        for k in 0..n {
            m.swap(col * n + k, piv * n + k);
            inv.swap(col * n + k, piv * n + k);
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for r in 0..n {
            if r != col {
                let f = m[r * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        m[r * n + k] -= f * m[col * n + k];
                        inv[r * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    Ok(inv)
}

impl Basis {
    /// P_qr = ψ_qᵀ M⁻¹ ψ_r
    fn proj(&self) -> Vec<Vec<f64>> {
        let n = self.psi.len();
        let np = self.np;
        let mp: Vec<Vec<f64>> = self.psi.iter().map(|s| (0..np).map(|a| (0..np).map(|b| self.minv[a * np + b] * s[b]).sum()).collect()).collect();
        (0..n).map(|q| (0..n).map(|r| (0..np).map(|a| self.psi[q][a] * mp[r][a]).sum()).collect()).collect()
    }
}

/// Elastiska moduler (μ, K) för linjär analys; K = straffmodul för ν = 0,5.
fn linear_moduli(mat: &dyn MaterialModel, name: &str) -> Result<(f64, f64)> {
    if let Some(h) = mat.hyperelastic() {
        let k = if h.incompressible() { PENALTY * h.mu0() } else { h.k0() };
        return Ok((h.mu0(), k));
    }
    let Some((e, nu)) = mat.elastic_moduli().filter(|_| mat.is_linear()) else {
        return fail(format!("{name}: hybridelement stöder linjärt isotropt elastiska och hyperelastiska material"));
    };
    let mu = e / (2.0 * (1.0 + nu));
    let k = if nu >= 0.5 - 1e-12 { PENALTY * mu } else { e / (3.0 * (1.0 - 2.0 * nu)) };
    Ok((mu, k))
}

fn linear_incompressible(mat: &dyn MaterialModel) -> bool {
    match mat.hyperelastic() {
        Some(h) => h.incompressible(),
        None => mat.elastic_moduli().map_or(false, |(_, nu)| nu >= 0.5 - 1e-12),
    }
}

impl Hybrid {
    fn ndim(&self) -> usize {
        common::ndim(self.base.mode)
    }

    fn bmat(&self, ip: &Ip, x: &[[f64; 3]]) -> Vec<f64> {
        super::solid_higher::bmat_any(self.base.mode, ip, x)
    }

    /// Linjär styvhet, och spänning/töjning per punkt för förskjutningen `u`.
    fn linear(&self, ctx: &ElemCtx, u: Option<&[f64]>) -> Result<(Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>)> {
        let mode = self.base.mode;
        let (nbar, _) = knobs(self);
        let mat = ctx.material(self.base.name)?;
        let (mu, kb) = linear_moduli(mat, self.base.name)?;
        // β bara för de rena hybridelementen; C3D8H, CPE4H och CAX4H (selektivt
        // reducerad integrering) har den inte
        let beta = if linear_incompressible(mat) || self.hourglass { 0.0 } else { BETA };
        let ips = self.base.ips(ctx)?;
        let bs: Vec<Vec<f64>> = ips.iter().map(|ip| self.bmat(ip, &ctx.x)).collect();
        let ns = mode.ncomp();
        let nd = bs[0].len() / ns;
        let nat: Vec<[f64; 3]> = self.base.rule.points(self.base.shape.pdim()).iter().map(|p| p.0).collect();
        let bas = basis(self.pressure, self.ndim(), &ips, &ctx.x, Some(&nat))?;
        let np = bas.np;
        // b_k = Σ w ψ_k div
        let mut bv = vec![0.0; np * nd];
        for ((ip, b), s) in ips.iter().zip(&bs).zip(&bas.psi) {
            for c in 0..nd {
                let dv = b[c] + b[nd + c] + b[2 * nd + c];
                for k in 0..np {
                    bv[k * nd + c] += ip.w * s[k] * dv;
                }
            }
        }
        // θ-operator per punkt: ψ_qᵀ M⁻¹ b
        let th: Vec<Vec<f64>> = bas
            .psi
            .iter()
            .map(|s| {
                let ms: Vec<f64> = (0..np).map(|a| (0..np).map(|b| bas.minv[a * np + b] * s[b]).sum()).collect();
                (0..nd).map(|c| (0..np).map(|k| ms[k] * bv[k * nd + c]).sum()).collect()
            })
            .collect();
        // deviatorisk D (ingenjörsskjuvning) i 3D-komponenterna (11, 22, 33, skjuv...)
        let shear = ns - 3;
        let mut dd = vec![0.0; ns * ns];
        for i in 0..3 {
            for j in 0..3 {
                dd[i * ns + j] = 2.0 * mu * (if i == j { 1.0 } else { 0.0 } - 1.0 / 3.0);
            }
        }
        for s in 0..shear {
            dd[(3 + s) * ns + 3 + s] = mu;
        }
        let mut ke = vec![0.0; nd * nd];
        let mut bbars = Vec::with_capacity(ips.len());
        for (q, (ip, b)) in ips.iter().zip(&bs).enumerate() {
            let mut bb = b.clone();
            if nbar == 4 {
                // axisymmetri som CAX4 (B-bar): εrr+εzz och εθθ ersätts var för sig med sina medelvärden
                let wsum: f64 = ips.iter().map(|ip| ip.w).sum();
                for c in 0..nd {
                    let pl_avg: f64 = ips.iter().zip(&bs).map(|(ip, b)| ip.w * (b[c] + b[nd + c])).sum::<f64>() / wsum;
                    let ho_avg: f64 = ips.iter().zip(&bs).map(|(ip, b)| ip.w * b[2 * nd + c]).sum::<f64>() / wsum;
                    let corr = (pl_avg - b[c] - b[nd + c]) / 2.0;
                    bb[c] += corr;
                    bb[nd + c] += corr;
                    bb[2 * nd + c] = ho_avg;
                }
            } else if nbar > 0 {
                for c in 0..nd {
                    let tr: f64 = (0..nbar).map(|r| b[r * nd + c]).sum();
                    let corr = (th[q][c] - tr) / nbar as f64;
                    for r in 0..nbar {
                        bb[r * nd + c] += corr;
                    }
                }
            }
            for r in 0..ns {
                for m in 0..ns {
                    let d = dd[r * ns + m];
                    if d == 0.0 {
                        continue;
                    }
                    for i in 0..nd {
                        let a = bb[r * nd + i] * d * ip.w;
                        if a == 0.0 {
                            continue;
                        }
                        for j in 0..nd {
                            ke[i * nd + j] += a * bb[m * nd + j];
                        }
                    }
                }
            }
            bbars.push(bb);
        }
        // K bᵀ M⁻¹ b
        for i in 0..nd {
            for j in 0..nd {
                let mut s = 0.0;
                for a in 0..np {
                    for c in 0..np {
                        s += bv[a * nd + i] * bas.minv[a * np + c] * bv[c * nd + j];
                    }
                }
                ke[i * nd + j] += (1.0 - beta) * kb * s;
            }
        }
        if beta > 0.0 {
            for (ip, b) in ips.iter().zip(&bs) {
                let dv: Vec<f64> = (0..nd).map(|c| b[c] + b[nd + c] + b[2 * nd + c]).collect();
                for i in 0..nd {
                    for j in 0..nd {
                        ke[i * nd + j] += beta * kb * ip.w * dv[i] * dv[j];
                    }
                }
            }
        }
        let mut sig = Vec::new();
        let mut eps = Vec::new();
        if let Some(u) = u {
            for (q, bb) in bbars.iter().enumerate() {
                let e = common::strain(bb, u);
                let thq: f64 = th[q].iter().zip(u).map(|(a, b)| a * b).sum();
                let mut s: Vec<f64> = (0..ns).map(|r| (0..ns).map(|m| dd[r * ns + m] * e[m]).sum()).collect();
                let b = &bs[q];
                let dv: f64 = (0..nd).map(|c| (b[c] + b[nd + c] + b[2 * nd + c]) * u[c]).sum();
                for v in s.iter_mut().take(3) {
                    *v += kb * ((1.0 - beta) * thq + beta * dv);
                }
                sig.push(s);
                eps.push(e);
            }
        }
        Ok((ke, sig, eps))
    }
}

/// Kinematik i en punkt vid NLGEOM.
struct Pt {
    w: f64,
    f: M3,
    j: f64,
    /// δL = δF F⁻¹ för varje frihetsgrad (3×3)
    lg: Vec<M3>,
    /// tr δL per frihetsgrad
    gv: Vec<f64>,
}

fn tr(m: &M3) -> f64 {
    m[0][0] + m[1][1] + m[2][2]
}

fn ddot(a: &M3, b: &M3) -> f64 {
    (0..3).map(|i| (0..3).map(|j| a[i][j] * b[i][j]).sum::<f64>()).sum()
}

/// tr(A B) begränsat till de n första diagonalelementen.
fn trn_prod(a: &M3, b: &M3, n: usize) -> f64 {
    (0..n).map(|i| (0..3).map(|k| a[i][k] * b[k][i]).sum::<f64>()).sum()
}

pub struct NlEval {
    pub fint: Vec<f64>,
    pub k: Option<Vec<f64>>,
    pub state: Vec<f64>,
}

/// Hybridelement vid NLGEOM med hyperelastiskt material.
#[allow(clippy::too_many_arguments)]
pub fn nl_eval(
    mode: StressMode,
    ips: &[Ip],
    x: &[[f64; 3]],
    pressure: Pressure,
    nbar: usize,
    hourglass: bool,
    h: &dyn HyperModel,
    u: &[f64],
    tangent: bool,
    nat: Option<&[[f64; 3]]>,
    mult: Option<(&[f64], f64)>,
) -> Result<NlEval> {
    nl_eval_hist(mode, ips, x, pressure, nbar, hourglass, h, u, tangent, nat, mult, None)
}

/// Som `nl_eval`, med materialets historievariabler (`HyperModel::n_hist`,
/// *MULLINS EFFECT) ur elementtillståndet vid inkrementets början; de ligger
/// efter spänningen i varje punkt.
#[allow(clippy::too_many_arguments)]
pub fn nl_eval_hist(
    mode: StressMode,
    ips: &[Ip],
    x: &[[f64; 3]],
    pressure: Pressure,
    nbar: usize,
    hourglass: bool,
    h: &dyn HyperModel,
    u: &[f64],
    tangent: bool,
    nat: Option<&[[f64; 3]]>,
    mult: Option<(&[f64], f64)>,
    hist_old: Option<&[f64]>,
) -> Result<NlEval> {
    // *HYPERFOAM (ingen uppdelning): hela τ(F̄) i stället för τ_dev, ingen volymterm
    // (används av CAX4, `cax4_hyper`; hybridelementen själva stöder inte skum)
    let full = h.kirchhoff_full(&t::I3).is_some();
    if full && mult.is_some() {
        return fail("*HYPERFOAM stöds inte i hybridelement (använd C3D8, CPE4, CAX4, ...)");
    }
    let tau_of = |f: &M3, old: &[f64], new: &mut [f64]| -> Result<M3> { if full { hs::kirchhoff_hist(h, f, old, new) } else { h.tau_dev_hist(f, old, new) } };
    let ndim = common::ndim(mode);
    let nn = ips[0].n.len();
    let nd = nn * ndim;
    let nq = ips.len();
    let axi = mode == StressMode::Axisymmetric;
    let mu0 = h.mu0();
    let kpen = PENALTY * mu0;
    let vol = |th: f64| -> (f64, f64) {
        if mult.is_some() {
            (0.0, 0.0)
        } else if h.incompressible() {
            (kpen * (th - 1.0), kpen)
        } else {
            h.vol(th)
        }
    };
    let mut pts = Vec::with_capacity(nq);
    for ip in ips {
        let mut f = t::I3;
        for a in 0..nn {
            for i in 0..ndim {
                for jj in 0..ndim {
                    f[i][jj] += u[a * ndim + i] * ip.dndx[a][jj];
                }
            }
        }
        let r0 = if axi { radius(ip, x) } else { 1.0 };
        if axi {
            let ur: f64 = (0..nn).map(|a| ip.n[a] * u[a * ndim]).sum();
            f[2][2] = 1.0 + ur / r0;
        }
        let (fi, j) = t::inv(&f);
        if !(j > 0.0) || (ndim == 2 && !(f[0][0] * f[1][1] - f[0][1] * f[1][0] > 0.0)) {
            return fail(common::DISTORTED);
        }
        let mut lg = vec![t::zero(); nd];
        for a in 0..nn {
            let g = t::mv(&t::transpose(&fi), ip.dndx[a]);
            for i in 0..ndim {
                let m = &mut lg[a * ndim + i];
                for jj in 0..ndim {
                    m[i][jj] = g[jj];
                }
                if axi && i == 0 {
                    m[2][2] = ip.n[a] / (r0 * f[2][2]);
                }
            }
        }
        let gv = lg.iter().map(tr).collect();
        pts.push(Pt { w: ip.w, f, j, lg, gv });
    }
    let bas = basis(pressure, ndim, ips, x, nat)?;
    let pm = bas.proj();
    // θ_q = Σ_r P_qr w_r J_r, dθ_q = Σ_r P_qr w_r J_r G_r
    let theta: Vec<f64> = (0..nq).map(|q| (0..nq).map(|r| pm[q][r] * pts[r].w * pts[r].j).sum()).collect();
    if theta.iter().any(|&v| !(v > 0.0)) {
        return fail(common::DISTORTED);
    }
    let dtheta: Vec<Vec<f64>> = (0..nq)
        .map(|q| (0..nd).map(|c| (0..nq).map(|r| pm[q][r] * pts[r].w * pts[r].j * pts[r].gv[c]).sum()).collect())
        .collect();
    let hc = if hourglass && nbar > 0 { hourglass_coef(if mode == StressMode::PlaneStrain || (axi && nbar == 4) { 2 } else { 3 }) * mu0 } else { 0.0 };
    // tryck i punkterna: projektionen av U′(θ)
    let up: Vec<f64> = theta.iter().map(|&th| vol(th).0).collect();
    let mut ptil: Vec<f64> = (0..nq).map(|q| (0..nq).map(|r| pm[q][r] * pts[r].w * up[r]).sum()).collect();
    // helt inkompressibelt: trycket är Lagrangemultiplikatorer (skalade med s) i tryckbasen
    let np = if mult.is_some() { bas.np } else { 0 };
    if let Some((pv, sc)) = mult {
        for q in 0..nq {
            ptil[q] = sc * (0..np).map(|k| bas.psi[q][k] * pv[k]).sum::<f64>();
        }
    }
    let ns = mode.ncomp();
    let nt = nd + np;
    let mut fint = vec![0.0; nt];
    let mut kmat = if tangent { vec![0.0; nt * nt] } else { Vec::new() };
    let nh = h.n_hist();
    let stride = ns + nh;
    let zeros = vec![0.0; nh];
    let old_of = |q: usize| -> &[f64] {
        match hist_old {
            Some(so) if nh > 0 && so.len() >= nq * stride => &so[q * stride + ns..(q + 1) * stride],
            _ => &zeros[..],
        }
    };
    let mut state = vec![0.0; nq * stride];
    let mut ext = vec![0.0; 12 * nq];
    // koefficient c_q framför Δ(dθ_q) (andra variationen av θ_q)
    let mut c2 = vec![0.0; nq];
    // CAX4H: ringsträckningen ersätts med sitt medelvärde λ̄ och planets del skalas så att J(F̄) = θ
    let axbar = axi && nbar == 4;
    let (lam_bar, lam_v) = if axbar {
        let wl: f64 = pts.iter().map(|p| p.w * p.f[2][2]).sum();
        let ws: f64 = pts.iter().map(|p| p.w).sum();
        let lv: Vec<f64> = (0..nd).map(|c| pts.iter().map(|p| p.w * p.f[2][2] * p.lg[c][2][2]).sum::<f64>() / wl).collect();
        (wl / ws, lv)
    } else {
        (1.0, Vec::new())
    };
    let nbe = if axbar { 2 } else { nbar };
    for q in 0..nq {
        let p = &pts[q];
        let th = theta[q];
        let mut fb = p.f;
        if axbar {
            let jp = p.f[0][0] * p.f[1][1] - p.f[0][1] * p.f[1][0];
            let s = (th / (lam_bar * jp)).sqrt();
            for r in 0..2 {
                for c in 0..2 {
                    fb[r][c] *= s;
                }
            }
            fb[2][2] = lam_bar;
        } else if nbar > 0 {
            // F̄ = (θ/J)^(1/n) F i n-blocket (n = 2 bara vid plan töjning, där J = J_n)
            let s = (th / p.j).powf(1.0 / nbar as f64);
            for r in 0..nbar {
                for c in 0..nbar {
                    fb[r][c] *= s;
                }
            }
        }
        let jb = t::det(&fb);
        let mut hn = vec![0.0; nh];
        let td = tau_of(&fb, old_of(q), &mut hn)?;
        let mut sig = t::scale(&td, 1.0 / jb);
        for i in 0..3 {
            sig[i][i] += ptil[q];
        }
        state[q * stride..q * stride + ns].copy_from_slice(&tensor_to_vec(mode, &sig, false));
        state[q * stride + ns..(q + 1) * stride].copy_from_slice(&hn);
        let log_e = log_strain_out(mode, &fb);
        let le6 = [log_e[0][0], log_e[1][1], log_e[2][2], log_e[0][1], log_e[0][2], log_e[1][2]];
        ext[6 * q..6 * q + 6].copy_from_slice(&le6);
        ext[6 * nq + 6 * q..6 * nq + 6 * q + 6].copy_from_slice(&le6);
        let (u1, u2) = vol(th);
        // timglasets volymkvot: J/θ, för CAX4H planets J_p/J̄_p med J̄_p = θ/λ̄
        let jp = p.f[0][0] * p.f[1][1] - p.f[0][1] * p.f[1][0];
        let x = if axbar { jp * lam_bar / th } else { p.j / th };
        let psi = x * x - 1.0;
        let hvec = |c: usize| if axbar { p.lg[c][0][0] + p.lg[c][1][1] - dtheta[q][c] / th + lam_v[c] } else { p.gv[c] - dtheta[q][c] / th };
        let trd: f64 = (0..nbe).map(|i| td[i][i]).sum();
        let w = p.w;
        // δL̄ per frihetsgrad
        let lb: Vec<M3> = (0..nd)
            .map(|c| {
                let mut m = p.lg[c];
                if axbar {
                    let corr = (dtheta[q][c] / th - lam_v[c] - p.lg[c][0][0] - p.lg[c][1][1]) / 2.0;
                    m[0][0] += corr;
                    m[1][1] += corr;
                    m[2][2] = lam_v[c];
                } else if nbar > 0 {
                    let trn: f64 = (0..nbar).map(|i| p.lg[c][i][i]).sum();
                    let corr = (dtheta[q][c] / th - trn) / nbar as f64;
                    for i in 0..nbar {
                        m[i][i] += corr;
                    }
                }
                m
            })
            .collect();
        for c in 0..nd {
            let hv = hvec(c);
            fint[c] += w * (ddot(&td, &lb[c]) + u1 * dtheta[q][c] + hc * psi * hv);
        }
        // koefficienter framför Δ(dθ_q)/θ (från τ_dev : I_n/n och timglas) och Δ(dθ_q) (U′)
        let a_rel = if nbe > 0 { trd / nbe as f64 } else { 0.0 } - hc * psi;
        c2[q] = w * (u1 + a_rel / th);
        if !tangent {
            continue;
        }
        let k = &mut kmat;
        let ct = hs::jaumann_tangent(&|f: &M3| tau_of(f, old_of(q), &mut vec![0.0; nh]), &fb)?;
        let dv: Vec<[f64; 6]> = lb
            .iter()
            .map(|l| {
                let s = t::sym(l);
                [s[0][0], s[1][1], s[2][2], 2.0 * s[0][1], 2.0 * s[0][2], 2.0 * s[1][2]]
            })
            .collect();
        for r in 0..nd {
            let cd: [f64; 6] = std::array::from_fn(|m| (0..6).map(|n| ct[n][m] * dv[r][n]).sum());
            for c in 0..nd {
                // materialdel och spinndel
                let w_ = t::skew(&lb[c]);
                let xm = t::sub(&t::mul(&w_, &td), &t::mul(&td, &w_));
                let mut s: f64 = (0..6).map(|m| cd[m] * dv[c][m]).sum();
                s += ddot(&lb[r], &xm);
                // geometrisk del −τ_d : (δL ΔL) och T-termer
                let mut prod = t::mul(&p.lg[r], &p.lg[c]);
                let gterm = -ddot(&td, &t::transpose(&prod));
                prod = t::mul(&p.lg[r], &p.lg[c]);
                let tq = tr(&prod);
                let tn = trn_prod(&p.lg[r], &p.lg[c], nbe);
                let mut v = s + gterm;
                if axbar {
                    let ll = lam_v[r] * lam_v[c];
                    v += trd / 2.0 * (tn - dtheta[q][r] * dtheta[q][c] / (th * th) + ll) + td[2][2] * (prod[2][2] - ll);
                } else if nbar > 0 {
                    v += trd / nbar as f64 * (tn - dtheta[q][r] * dtheta[q][c] / (th * th));
                }
                v += u2 * dtheta[q][r] * dtheta[q][c];
                let (hv_r, hv_c) = (hvec(r), hvec(c));
                let dh = if axbar { -tn - lam_v[r] * lam_v[c] } else { -tq };
                v += hc * (2.0 * x * x * hv_r * hv_c + psi * (dh + dtheta[q][r] * dtheta[q][c] / (th * th)));
                k[r * nt + c] += w * v;
            }
        }
    }
    if tangent {
        // Σ_q c2_q Δ(dθ_q) med Δ(dθ_q) = Σ_r P_qr w_r J_r [G_r G_rᵀ − T_r]
        let k = &mut kmat;
        for r in 0..nq {
            let coef: f64 = (0..nq).map(|q| c2[q] * pm[q][r]).sum::<f64>() * pts[r].w * pts[r].j;
            if coef == 0.0 {
                continue;
            }
            let p = &pts[r];
            for a in 0..nd {
                for b in 0..nd {
                    let tq = tr(&t::mul(&p.lg[a], &p.lg[b]));
                    k[a * nt + b] += coef * (p.gv[a] * p.gv[b] - tq);
                }
            }
        }
        if let Some((_, sc)) = mult {
            let k = &mut kmat;
            for (q, p) in pts.iter().enumerate() {
                let w = p.w;
                for a in 0..nd {
                    // Σ p_q J_q (G Gᵀ − T) och kopplingen s ψ_k J_q G_q
                    for b in 0..nd {
                        let tq = tr(&t::mul(&p.lg[a], &p.lg[b]));
                        k[a * nt + b] += w * ptil[q] * p.j * (p.gv[a] * p.gv[b] - tq);
                    }
                    for kk in 0..np {
                        let v = w * sc * bas.psi[q][kk] * p.j * p.gv[a];
                        k[a * nt + nd + kk] += v;
                        k[(nd + kk) * nt + a] += v;
                    }
                }
            }
        }
        symmetrize(&mut kmat, nt);
    }
    if let Some((_, sc)) = mult {
        for (q, p) in pts.iter().enumerate() {
            for c in 0..nd {
                fint[c] += p.w * ptil[q] * p.j * p.gv[c];
            }
            for kk in 0..np {
                fint[nd + kk] += sc * p.w * bas.psi[q][kk] * (p.j - 1.0);
            }
        }
    }
    crate::kinematics::push_ext(&mut state, &ext, nq);
    Ok(NlEval { fint, k: tangent.then_some(kmat), state })
}

/// CAX4 (selektivt reducerad integrering) med kompressibelt hyperelastiskt
/// material (även *HYPERFOAM och *MULLINS EFFECT) vid NLGEOM: räknas med
/// samma F-bar-formulering som CAX4H (konstant volymkvot θ, ringsträckningen
/// ersatt med sitt medelvärde, volymtimglasstyvhet), se
/// `tests/cases/foam/cax4_*`. Anropas från `solid_higher.rs`;
/// `None` för andra element och material.
#[allow(clippy::too_many_arguments)]
pub fn cax4_hyper(el: &Solid, ctx: &ElemCtx, mat: &dyn MaterialModel, u: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<Option<ElemResponse>> {
    if el.name != "CAX4" || el.mode != StressMode::Axisymmetric {
        return Ok(None);
    }
    let Some(h) = mat.hyperelastic() else { return Ok(None) };
    if mat.n_state() != h.n_hist() || h.incompressible() {
        return Ok(None);
    }
    let ips = el.ips(ctx)?;
    let nat: Vec<[f64; 3]> = el.rule.points(el.shape.pdim()).iter().map(|p| p.0).collect();
    let r = nl_eval_hist(el.mode, &ips, &ctx.x, Pressure::Constant, 4, true, h, u, tangent, Some(&nat), None, Some(so))?;
    *sn = r.state;
    Ok(Some(ElemResponse { fint: r.fint, k: r.k }))
}

impl Hybrid {
    /// Antal tryckvariabler per element.
    pub fn n_pressure(&self) -> usize {
        match self.pressure {
            Pressure::Constant => 1,
            Pressure::Linear => self.ndim() + 1,
        }
    }

    /// Motsvarande element utan hybridformulering för andra material än
    /// hyperelastiska (C3D8H → C3D8, CPE4H → CPE4, CAX4H → CAX4), eller `None`
    /// om elementet räknas med hybridformuleringen.
    fn forsta_ordningen(&self, mat: &dyn MaterialModel) -> Option<&'static dyn ElementType> {
        if mat.hyperelastic().is_some() || linear_incompressible(mat) {
            return None;
        }
        match self.base.name {
            "C3D8H" => Some(&super::continuum3d::C3D8),
            "CPE4H" => Some(&super::continuum2d::CPE4),
            "CAX4H" => Some(&super::axisym::CAX4),
            _ => None,
        }
    }

    /// Som `forsta_ordningen`, men bara när hybridvägen inte kan räkna fallet
    /// (NLGEOM eller olinjärt material); linjärt elastiskt utan NLGEOM ger samma
    /// resultat med båda.
    fn delegat(&self, ctx: &ElemCtx) -> Result<Option<&'static dyn ElementType>> {
        let mat = ctx.material(self.base.name)?;
        Ok(self.forsta_ordningen(mat).filter(|_| ctx.nlgeom || !mat.is_linear()))
    }

    /// Antal geometriska noder.
    fn ngeo(&self) -> usize {
        self.base.n_nodes()
    }

    /// Antal frihetsgrader på de geometriska noderna.
    fn ngeo_dofs(&self) -> usize {
        self.ngeo() * self.ndim()
    }

    /// Kontext med bara de geometriska noderna.
    fn geo_ctx<'a>(&self, ctx: &ElemCtx<'a>) -> ElemCtx<'a> {
        let mut c = ctx.clone();
        c.x.truncate(self.ngeo());
        c
    }

    /// Lägger till de interna nodernas frihetsgrader: identitet (oanvända) i
    /// styvheten och u i kraften, så att de förblir noll.
    fn pad(&self, fint: Vec<f64>, k: Option<Vec<f64>>, u: &[f64], used: usize) -> ElemResponse {
        let n = u.len();
        let m = fint.len();
        let mut f = fint;
        f.resize(n, 0.0);
        for i in m.max(used)..n {
            f[i] = u[i];
        }
        let k = k.map(|ke| {
            let mut kk = vec![0.0; n * n];
            for i in 0..m {
                kk[i * n..i * n + m].copy_from_slice(&ke[i * m..(i + 1) * m]);
            }
            for i in m.max(used)..n {
                kk[i * n + i] = 1.0;
            }
            kk
        });
        ElemResponse { fint: f, k }
    }

    /// Skala s för tryckvariablerna: p = s · u_p med s = μ0 / h, så att u_p får
    /// längdenhet och tryckekvationerna kraftenhet (Newton-kriterierna per fält).
    fn pressure_scale(&self, ctx: &ElemCtx, h: &dyn HyperModel) -> f64 {
        let x = &ctx.x[..self.ngeo()];
        let size = x.iter().map(|p| common::norm(common::sub(*p, x[0]))).fold(0.0, f64::max).max(1e-300);
        h.mu0() / size
    }
}

impl ElementType for Hybrid {
    fn name(&self) -> &'static str {
        self.base.name
    }
    fn family(&self) -> Family {
        self.base.family()
    }
    fn dim(&self) -> usize {
        self.base.dim()
    }
    fn n_nodes(&self) -> usize {
        self.ngeo() + self.internal_nodes()
    }
    fn n_connectivity(&self) -> usize {
        self.ngeo()
    }
    /// Tryckvariablerna ligger på interna noder, `ndim` per nod.
    fn internal_nodes(&self) -> usize {
        self.n_pressure().div_ceil(self.ndim())
    }
    fn node_dofs(&self, _i: usize) -> &'static [Dof] {
        self.base.node_dofs(0)
    }
    fn n_ips(&self) -> usize {
        self.base.n_ips()
    }
    fn vtk_type(&self) -> u8 {
        self.base.vtk_type()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.base.faces()
    }

    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        if let Some(m) = ctx.material {
            if let Some(d) = self.forsta_ordningen(m) {
                return d.init_state(&self.geo_ctx(ctx));
            }
        }
        let nh = ctx.material.and_then(|m| m.hyperelastic()).map_or(0, |h| h.n_hist());
        Ok(vec![0.0; self.n_ips() * (self.base.mode.ncomp() + nh)])
    }

    fn stiffness(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let g = self.geo_ctx(ctx);
        let ke = self.linear(&g, None)?.0;
        let n = self.n_dofs();
        Ok(self.pad(vec![0.0; self.ngeo_dofs()], Some(ke), &vec![0.0; n], 0).k.unwrap())
    }

    fn internal_force(&self, ctx: &ElemCtx, u: &[f64], du: &[f64], so: &[f64], sn: &mut Vec<f64>, tangent: bool) -> Result<ElemResponse> {
        let mat = ctx.material(self.base.name)?;
        let g = self.geo_ctx(ctx);
        let ng = self.ngeo_dofs();
        let ug = &u[..ng];
        if let Some(d) = self.delegat(ctx)? {
            let r = d.internal_force(&g, ug, &du[..ng], so, sn, tangent)?;
            return Ok(self.pad(r.fint, r.k, u, ng));
        }
        if ctx.nlgeom {
            let Some(h) = mat.hyperelastic() else {
                return not_implemented("NLGEOM med andra material än *HYPERELASTIC", self.base.name);
            };
            if h.kirchhoff_full(&t::I3).is_some() {
                return fail(format!("{}: *HYPERFOAM stöds inte i hybridelement (använd C3D8, CPE4, CAX4, ...)", self.base.name));
            }
            let (nbar, hg) = knobs(self);
            let ips = self.base.ips(&g)?;
            let nat: Vec<[f64; 3]> = self.base.rule.points(self.base.shape.pdim()).iter().map(|p| p.0).collect();
            let np = self.n_pressure();
            let sc = self.pressure_scale(&g, h);
            let (uin, mult) = if h.incompressible() { (&u[..ng + np], Some((&u[ng..ng + np], sc))) } else { (ug, None) };
            let r = nl_eval_hist(self.base.mode, &ips, &g.x, self.pressure, nbar, hg, h, &uin[..ng], tangent, Some(&nat), mult, Some(so))?;
            *sn = r.state;
            let used = if mult.is_some() { ng + np } else { ng };
            return Ok(self.pad(r.fint, r.k, u, used));
        }
        let (ke, sig, _) = self.linear(&g, Some(ug))?;
        let fint = (0..ng).map(|i| (0..ng).map(|j| ke[i * ng + j] * ug[j]).sum()).collect();
        *sn = sig.concat();
        Ok(self.pad(fint, tangent.then_some(ke), u, 0))
    }

    fn mass(&self, ctx: &ElemCtx, kind: MassKind) -> Result<Vec<f64>> {
        let m = self.base.mass(&self.geo_ctx(ctx), kind)?;
        let (ng, n) = (self.ngeo_dofs(), self.n_dofs());
        let mut out = vec![0.0; n * n];
        for i in 0..ng {
            out[i * n..i * n + ng].copy_from_slice(&m[i * ng..(i + 1) * ng]);
        }
        Ok(out)
    }

    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let ng = self.ngeo_dofs();
        let mut f = self.base.distributed_load(&self.geo_ctx(ctx), load, &u[..ng.min(u.len())])?;
        f.resize(self.n_dofs(), 0.0);
        Ok(f)
    }

    /// Spänningsstyvhet för *BUCKLE (område hybrid2, `hybrid2::hybrid_geometric`).
    fn geometric_stiffness(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<f64>> {
        let g = self.geo_ctx(ctx);
        let ng = self.ngeo_dofs();
        // C3D8H/CPE4H/CAX4H som C3D8/CPE4/CAX4 (även linjärt elastiskt;
        // tillståndet har samma layout)
        let mat = ctx.material(self.base.name)?;
        let kg = match self.forsta_ordningen(mat) {
            Some(d) => d.geometric_stiffness(&g, &u[..ng.min(u.len())], state)?,
            None => super::hybrid2::hybrid_geometric(self, &g, mat, state)?,
        };
        let n = self.n_dofs();
        Ok(self.pad(vec![0.0; ng], Some(kg), &vec![0.0; n], n).k.unwrap())
    }

    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let mut w = self.base.volume_weights(&self.geo_ctx(ctx))?;
        w.resize(self.n_nodes(), 0.0);
        Ok(w)
    }

    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        let mode = self.base.mode;
        let ns = mode.ncomp();
        let (sl, el) = common::stress_labels(mode);
        let mut s = IpField::new("S", sl);
        let mut e = IpField::new("E", el);
        let g = self.geo_ctx(ctx);
        if let Some(d) = self.delegat(ctx)? {
            return d.output(&g, &u[..self.ngeo_dofs()], state);
        }
        let eps = if ctx.nlgeom { Vec::new() } else { self.linear(&g, Some(&u[..self.ngeo_dofs()]))?.2 };
        // historievariabler (*MULLINS EFFECT: SENER, DMENER) efter spänningen vid NLGEOM
        let mat = ctx.material(self.base.name)?;
        let nh = if ctx.nlgeom { mat.hyperelastic().map_or(0, |h| h.n_hist()) } else { 0 };
        let stride = ns + nh;
        let groups = if nh > 0 && nh == mat.n_state() { mat.output_fields(mode) } else { Vec::new() };
        let mut extra: Vec<IpField> = groups.iter().map(|(n, l)| IpField::new(n, l)).collect();
        for q in 0..self.n_ips() {
            let sig = state.get(q * stride..q * stride + ns).map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; ns]);
            s.points.push((q + 1, 0, sig));
            e.points.push((q + 1, 0, eps.get(q).cloned().unwrap_or_else(|| vec![0.0; ns])));
            if !extra.is_empty() {
                let hv = state.get(q * stride + ns..(q + 1) * stride).map(|v| v.to_vec()).unwrap_or_else(|| vec![0.0; nh]);
                let vals = mat.output_field_values(mode, &hv);
                let mut k = 0;
                for (f, (_, l)) in extra.iter_mut().zip(&groups) {
                    f.points.push((q + 1, 0, vals[k..k + l.len()].to_vec()));
                    k += l.len();
                }
            }
        }
        let mut out = vec![s, e];
        out.append(&mut extra);
        Ok(out)
    }
}
