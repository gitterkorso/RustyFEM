//! Kopplade porvattentryck–förskjutningselement (geoteknik, *SOILS):
//! CPE4P, CPE8P, CPE8RP, CAX4P, CAX8P, CAX8RP, C3D8P, C3D20P, C3D20RP.
//!
//! Hörnnoderna har förskjutningarna (1–2 eller 1–3) och porvattentrycket u_w
//! (frihetsgrad 8, slot 7 – samma slot som NT11, en modell har inte båda);
//! mittnoderna hos andra ordningens element har bara förskjutningar. Trycket
//! interpoleras linjärt (bilinjärt/trilinjärt) med hörnnoderna, förskjutningen
//! med elementets alla noder.
//!
//! Formulering (helt mättad jord, små töjningar):
//! - effektiv spänning: σ = σ̄ − u_w I (dragning positiv, u_w positivt vid
//!   tryck). Materialet (`materials/`) ger σ̄, som också redovisas som S. Med
//!   kompressibla korn (*POROUS BULK MODULI, K_g) får materialet töjningen
//!   ε + u_w/(3K_g) I (kornens kompression av porvattentrycket);
//! - jämvikt: ∫ Bᵀ(σ̄ − u_w m) dV = F_ext;
//! - kontinuitet (bakåt-Euler, volymflöde per tidsenhet):
//!   ∫ N_p Δm/Δt dV − ∫ ∇N_p·f dV = inflöde, Darcy f = −(k/γw)(∇u_w − ρw g),
//!   med vätskeinnehållet (linjäriserat) m = n₀ + ε_v + (1 − n₀)u_w/K_g −
//!   tr σ̄/(3K_g) + n₀ u_w/K_w (`fluid_content`; utan *POROUS BULK MODULI bara
//!   ε_v). Stationärt (*SOILS utan CONSOLIDATION, *GEOSTATIC med GRAV)
//!   utelämnas tidsderivatan; ρw g kommer från GRAV (totalt porvattentryck);
//! - portalet: VOIDR = (1 + e₀)/(1 − ε_v) − 1 (med kornens densitetsändring vid
//!   kompressibla korn), permeabiliteten k(e) med e = e₀ + (1 + e₀)ε_v.
//!
//! Den inre "kraften" i tryckfrihetsgraderna är alltså volymflödet ut ur noden
//! per tidsenhet; i låsta tryckfrihetsgrader blir den reaktionen RVF.
//!
//! Integrering: 2×2 (2×2×2) med selektivt reducerad integrering av
//! volymtöjningen för CPE4P/CAX4P/C3D8P, 3×3 (3×3×3) för CPE8P/CAX8P/C3D20P och
//! 2×2 (2×2×2) för de reducerade CPE8RP/CAX8RP/C3D20RP.
//!
//! Begynnelsespänning (*INITIAL CONDITIONS, TYPE=STRESS) läggs in som en
//! elastisk förtöjning ε₀ = C⁻¹σ₀ som adderas till töjningen innan materialet
//! anropas: σ̄ = σ₀ + C(ε − εp) för material med linjär elasticitet, utan att
//! materialmodellerna behöver känna till begynnelsespänningen.

use super::api::{not_implemented, ElemCtx, ElemResponse, ElementType, Family, IpField};
use super::common::{self, Ip, DISTORTED};
use super::solid_higher::shapes::{self, Shape};
use super::solid_higher::{self as sh, RuleKind};
use crate::fem::dof::{Dof, POR, U1, U2, U3};
use crate::fem::model::DistLoad;
use crate::materials::porous::PoroProps;
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};

/// Ett porvattentryckselement.
pub struct SoilElem {
    pub name: &'static str,
    /// Motsvarande mekaniska element i registret (för ytlaster och volymkrafter).
    pub mech: &'static str,
    pub shape: Shape,
    /// Tryckets interpolation (hörnnoderna).
    pub pshape: Shape,
    pub mode: StressMode,
    pub rule: RuleKind,
    /// Selektivt reducerad integrering av volymtöjningen (B-bar).
    pub bbar: bool,
}

const fn el(name: &'static str, mech: &'static str, shape: Shape, pshape: Shape, mode: StressMode, n: usize, bbar: bool) -> SoilElem {
    SoilElem { name, mech, shape, pshape, mode, rule: RuleKind::Gauss(n), bbar }
}

pub static CPE4P: SoilElem = el("CPE4P", "CPE4", Shape::Quad4, Shape::Quad4, StressMode::PlaneStrain, 2, true);
pub static CPE8P: SoilElem = el("CPE8P", "CPE8", Shape::Quad8, Shape::Quad4, StressMode::PlaneStrain, 3, false);
pub static CPE8RP: SoilElem = el("CPE8RP", "CPE8R", Shape::Quad8, Shape::Quad4, StressMode::PlaneStrain, 2, false);
pub static CAX4P: SoilElem = el("CAX4P", "CAX4", Shape::Quad4, Shape::Quad4, StressMode::Axisymmetric, 2, true);
pub static CAX8P: SoilElem = el("CAX8P", "CAX8", Shape::Quad8, Shape::Quad4, StressMode::Axisymmetric, 3, false);
pub static CAX8RP: SoilElem = el("CAX8RP", "CAX8R", Shape::Quad8, Shape::Quad4, StressMode::Axisymmetric, 2, false);
pub static C3D8P: SoilElem = el("C3D8P", "C3D8", Shape::Hex8, Shape::Hex8, StressMode::ThreeD, 2, true);
pub static C3D20P: SoilElem = el("C3D20P", "C3D20", Shape::Hex20, Shape::Hex8, StressMode::ThreeD, 3, false);
pub static C3D20RP: SoilElem = el("C3D20RP", "C3D20R", Shape::Hex20, Shape::Hex8, StressMode::ThreeD, 2, false);

pub static TYPES: &[&dyn ElementType] = &[&C3D20P, &C3D20RP, &C3D8P, &CAX4P, &CAX8P, &CAX8RP, &CPE4P, &CPE8P, &CPE8RP];

static ALL: &[&SoilElem] = &[&C3D20P, &C3D20RP, &C3D8P, &CAX4P, &CAX8P, &CAX8RP, &CPE4P, &CPE8P, &CPE8RP];

/// Porvattentryckselementet med namnet `name`.
pub fn lookup(name: &str) -> Option<&'static SoilElem> {
    ALL.iter().copied().find(|e| e.name == name)
}

/// Kopplingstermen (volymtöjningen i kontinuitetsekvationen) använder samma
/// B-bar-medelvärdesbildade volymtöjning som spänningen (CPE4P, CAX4P och C3D8P).
const COUPLING_BBAR: bool = true;

/// Hur kontinuitetsekvationen behandlas i ett inkrement.
#[derive(Debug, Clone, Copy)]
pub struct FluidMode {
    /// Tidsinkrement (transient).
    pub dt: f64,
    /// Transient (konsolidering) – annars stationär strömning.
    pub transient: bool,
    /// Porvätskans tyngd per volym ρw·g (GRAV, totalt porvattentryck), annars 0.
    pub gravity: [f64; 3],
}

/// Geometri i en integrationspunkt.
pub struct PIp {
    pub ip: Ip,
    /// Tryckets formfunktioner och deras derivator.
    pub np: Vec<f64>,
    pub dnp: Vec<[f64; 3]>,
    /// B-matris (ncomp × förskjutningsfrihetsgrader, med B-bar om elementet har det).
    pub b: Vec<f64>,
    /// Volymtöjningens rad (förskjutningsfrihetsgrader).
    pub bv: Vec<f64>,
    /// Läge.
    pub x: [f64; 3],
}

/// Tillståndslayout per integrationspunkt.
pub struct Layout {
    pub ns: usize,
    pub nst: usize,
}

impl Layout {
    pub fn stride(&self) -> usize {
        2 * self.ns + self.nst + 1
    }
    /// Spänning σ̄.
    pub fn stress(&self) -> usize {
        0
    }
    /// Materialets tillstånd.
    pub fn mat(&self) -> usize {
        self.ns
    }
    /// Elastisk förtöjning från begynnelsespänningen.
    pub fn eps0(&self) -> usize {
        self.ns + self.nst
    }
    /// Begynnelseportal.
    pub fn e0(&self) -> usize {
        2 * self.ns + self.nst
    }
}

fn det_inv_2(j: [[f64; 2]; 2]) -> f64 {
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

impl SoilElem {
    pub fn ndim(&self) -> usize {
        common::ndim(self.mode)
    }

    pub fn n_corner(&self) -> usize {
        self.pshape.n_nodes()
    }

    pub fn pdim(&self) -> usize {
        self.shape.pdim()
    }

    /// Antal förskjutningsfrihetsgrader.
    pub fn nu(&self) -> usize {
        self.ndim() * self.shape.n_nodes()
    }

    /// Index i elementvektorn för förskjutningskomponent `c` i nod `k`.
    pub fn u_index(&self, k: usize, c: usize) -> usize {
        let nm = self.ndim();
        let nc = self.n_corner();
        if k < nc {
            k * (nm + 1) + c
        } else {
            nc * (nm + 1) + (k - nc) * nm + c
        }
    }

    /// Index i elementvektorn för trycket i hörnnod `k`.
    pub fn p_index(&self, k: usize) -> usize {
        let nm = self.ndim();
        k * (nm + 1) + nm
    }

    pub fn n_dof(&self) -> usize {
        self.nu() + self.n_corner()
    }

    /// Delar en elementvektor i förskjutningar (nodvis) och tryck.
    pub fn split(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nm = self.ndim();
        let mut u = Vec::with_capacity(self.nu());
        for k in 0..self.shape.n_nodes() {
            for c in 0..nm {
                u.push(v[self.u_index(k, c)]);
            }
        }
        let p = (0..self.n_corner()).map(|k| v[self.p_index(k)]).collect();
        (u, p)
    }

    /// Index i elementvektorn för förskjutningsfrihetsgrad `a` (nodvis numrering).
    pub fn ua(&self, a: usize) -> usize {
        let nm = self.ndim();
        self.u_index(a / nm, a % nm)
    }

    pub fn layout(&self, mat: &dyn MaterialModel) -> Layout {
        Layout { ns: self.mode.ncomp(), nst: mat.n_state() }
    }

    pub fn thickness(&self, ctx: &ElemCtx) -> Result<f64> {
        match self.mode {
            StressMode::PlaneStrain | StressMode::PlaneStress => Ok(ctx.section(self.name)?.solid_thickness()),
            _ => Ok(1.0),
        }
    }

    /// Integrationspunkter med tryckets formfunktioner och B-matriser.
    pub fn geometry(&self, x: &[[f64; 3]], t: f64) -> Result<Vec<PIp>> {
        let rule = self.rule.points(self.pdim());
        let ips = sh::ips_for(self.shape, self.mode, x, &rule, t)?;
        let mut bs: Vec<Vec<f64>> = ips.iter().map(|ip| sh::bmat_any(self.mode, ip, x)).collect();
        let raw = bs.clone();
        if self.bbar {
            sh::apply_bbar(self.mode, &ips, &mut bs);
        }
        let nu = self.nu();
        let nv = if self.mode == StressMode::PlaneStrain { 2 } else { 3 };
        let mut out = Vec::with_capacity(ips.len());
        for (q, ip) in ips.into_iter().enumerate() {
            let (np, dnp) = self.pressure_shape(x, rule[q].0)?;
            let src = if COUPLING_BBAR { &bs[q] } else { &raw[q] };
            let bv: Vec<f64> = (0..nu).map(|c| (0..nv).map(|r| src[r * nu + c]).sum()).collect();
            let pos = [0, 1, 2].map(|c| ip.n.iter().zip(x).map(|(ni, xi)| ni * xi[c]).sum());
            out.push(PIp { ip, np, dnp, b: std::mem::take(&mut bs[q]), bv, x: pos });
        }
        Ok(out)
    }

    /// Tryckets formfunktioner och globala derivator i den naturliga punkten
    /// `p` (Jacobianen från elementets hela geometri).
    fn pressure_shape(&self, x: &[[f64; 3]], p: [f64; 3]) -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
        let (_, dn) = shapes::shape(self.shape, p);
        let (np, dnn) = shapes::shape(self.pshape, p);
        let mut d = vec![[0.0; 3]; np.len()];
        if self.pdim() == 3 {
            let mut j = [[0.0; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    j[r][c] = (0..dn.len()).map(|i| dn[i][r] * x[i][c]).sum();
                }
            }
            let (ji, det) = common::inv3(j);
            if !(det > 0.0) {
                return fail(DISTORTED);
            }
            for i in 0..np.len() {
                for c in 0..3 {
                    d[i][c] = (0..3).map(|r| ji[c][r] * dnn[i][r]).sum();
                }
            }
        } else {
            let mut j = [[0.0; 2]; 2];
            for r in 0..2 {
                for c in 0..2 {
                    j[r][c] = (0..dn.len()).map(|i| dn[i][r] * x[i][c]).sum();
                }
            }
            let det = det_inv_2(j);
            if !(det > 0.0) {
                return fail(DISTORTED);
            }
            for i in 0..np.len() {
                d[i][0] = (j[1][1] * dnn[i][0] - j[0][1] * dnn[i][1]) / det;
                d[i][1] = (-j[1][0] * dnn[i][0] + j[0][0] * dnn[i][1]) / det;
            }
        }
        Ok((np, d))
    }

    /// Elementets mittpunkt (naturliga koordinater 0).
    pub fn center(&self, x: &[[f64; 3]]) -> [f64; 3] {
        let (n, _) = shapes::shape(self.shape, [0.0; 3]);
        [0, 1, 2].map(|c| n.iter().zip(x).map(|(ni, xi)| ni * xi[c]).sum())
    }

    /// Integrationspunkternas lägen (för begynnelsevillkor som beror på höjden).
    pub fn ip_positions(&self, x: &[[f64; 3]]) -> Vec<[f64; 3]> {
        self.rule
            .points(self.pdim())
            .iter()
            .map(|&(p, _)| {
                let (n, _) = shapes::shape(self.shape, p);
                [0, 1, 2].map(|c| n.iter().zip(x).map(|(ni, xi)| ni * xi[c]).sum())
            })
            .collect()
    }

    /// Kopplad inre "kraft" (förskjutningar: kraft, tryck: volymflöde ut per
    /// tidsenhet) och tangent för elementvektorn `v` (total) med `v_old` vid
    /// inkrementets början.
    #[allow(clippy::too_many_arguments)]
    pub fn coupled(
        &self,
        ctx: &ElemCtx,
        props: &PoroProps,
        v: &[f64],
        v_old: &[f64],
        state_old: &[f64],
        state_new: &mut Vec<f64>,
        fm: FluidMode,
        tangent: bool,
    ) -> Result<ElemResponse> {
        let mat = ctx.material(self.name)?;
        let lay = self.layout(mat);
        let stride = lay.stride();
        let gs = self.geometry(&ctx.x, self.thickness(ctx)?)?;
        let ns = lay.ns;
        let nu = self.nu();
        let nc = self.n_corner();
        let nd = self.n_dof();
        let old_owned;
        let state_old = if state_old.len() == gs.len() * stride {
            state_old
        } else {
            old_owned = self.fresh_state(mat, gs.len());
            &old_owned[..]
        };
        state_new.clear();
        state_new.extend_from_slice(state_old);
        let (uu, pp) = self.split(v);
        let (uu0, pp0) = self.split(v_old);
        let duu: Vec<f64> = uu.iter().zip(&uu0).map(|(a, b)| a - b).collect();
        let ips: Vec<Ip> = gs.iter().map(|g| g.ip.clone()).collect();
        let length = common::char_length(&ctx.x, self.mode, &ips);
        let mut fint = vec![0.0; nd];
        let mut k = if tangent { vec![0.0; nd * nd] } else { Vec::new() };
        let gw = props.perm.gamma_w;
        let ndim = self.ndim();
        let ua: Vec<usize> = (0..nu).map(|a| self.ua(a)).collect();
        let pa: Vec<usize> = (0..nc).map(|i| self.p_index(i)).collect();
        // kornens kompression av porvattentrycket: ε_g = −u_w/(3K_g) I
        let g3 = if props.k_grain > 0.0 { 1.0 / (3.0 * props.k_grain) } else { 0.0 };
        for (q, g) in gs.iter().enumerate() {
            let w = g.ip.w;
            let base = q * stride;
            let st = &state_old[base..base + stride];
            let eps0 = &st[lay.eps0()..lay.eps0() + ns];
            let e0 = st[lay.e0()];
            let p: f64 = g.np.iter().zip(&pp).map(|(a, b)| a * b).sum();
            let p_old: f64 = g.np.iter().zip(&pp0).map(|(a, b)| a * b).sum();
            // töjning för den effektiva spänningen: ε + ε₀ + u_w/(3K_g) I
            let mut eps = common::strain(&g.b, &uu);
            let mut deps = common::strain(&g.b, &duu);
            for (a, b) in eps.iter_mut().zip(eps0) {
                *a += b;
            }
            for i in 0..3 {
                eps[i] += g3 * p;
                deps[i] += g3 * (p - p_old);
            }
            let (temperature, dtemp) = ctx.temperature_at(&g.ip.n);
            let mold = &st[lay.mat()..lay.mat() + lay.nst];
            let mut mnew = mold.to_vec();
            let inp = MatInput {
                mode: self.mode,
                strain: &eps,
                dstrain: &deps,
                stress_old: &st[..ns],
                time: ctx.time,
                dt: ctx.dt,
                temperature,
                dtemp,
                defgrad: None,
            };
            let out = mat.update_with_length(&inp, length, mold, &mut mnew)?;
            state_new[base..base + ns].copy_from_slice(&out.stress);
            state_new[base + lay.mat()..base + lay.mat() + lay.nst].copy_from_slice(&mnew);
            let ev: f64 = g.bv.iter().zip(&uu).map(|(a, b)| a * b).sum();
            let ev_old: f64 = g.bv.iter().zip(&uu0).map(|(a, b)| a * b).sum();
            let mut grad = [0.0; 3];
            for (i, d) in g.dnp.iter().enumerate() {
                for c in 0..ndim {
                    grad[c] += d[c] * pp[i];
                }
            }
            // permeabiliteten vid portalet e = e₀ + (1 + e₀) ε_v
            let (kt, dkt) = props.perm.at(perm_void_ratio(e0, ev));
            let dedev = 1.0 + e0;
            // volymflöde per ytenhet f = −(k/γw)(∇u_w − ρw g)
            let mut f = [0.0; 3];
            let mut dfde = [0.0; 3];
            for r in 0..ndim {
                for c in 0..ndim {
                    f[r] -= kt[r * 3 + c] * (grad[c] - fm.gravity[c]) / gw;
                    dfde[r] -= dkt[r * 3 + c] * (grad[c] - fm.gravity[c]) / gw;
                }
            }
            // vätskeinnehåll per referensvolym (linjäriserat)
            let n0 = e0 / (1.0 + e0);
            let tr = |s: &[f64]| s[0] + s[1] + s[2];
            let content = |evv: f64, pw: f64, trs: f64| fluid_content(props, n0, evv, pw, trs);
            let dm = content(ev, p, tr(&out.stress)) - content(ev_old, p_old, tr(&st[..ns]));
            // jämvikt
            for a in 0..nu {
                let mut s = 0.0;
                for r in 0..ns {
                    s += g.b[r * nu + a] * out.stress[r];
                }
                fint[ua[a]] += (s - g.bv[a] * p) * w;
            }
            // kontinuitet
            for i in 0..nc {
                let mut flow = 0.0;
                for c in 0..ndim {
                    flow -= g.dnp[i][c] * f[c];
                }
                let mut r = flow;
                if fm.transient {
                    r += g.np[i] * dm / fm.dt;
                }
                fint[pa[i]] += r * w;
            }
            if tangent {
                let d = &out.tangent;
                // K_uu = Bᵀ D B
                let mut db = vec![0.0; ns * nu];
                for r in 0..ns {
                    for c in 0..nu {
                        db[r * nu + c] = (0..ns).map(|s| d[r * ns + s] * g.b[s * nu + c]).sum();
                    }
                }
                for a in 0..nu {
                    for b in 0..nu {
                        let mut s = 0.0;
                        for r in 0..ns {
                            s += g.b[r * nu + a] * db[r * nu + b];
                        }
                        k[ua[a] * nd + ua[b]] += s * w;
                    }
                }
                // dσ̄/du_w = D·I/(3K_g)
                let dm_p: Vec<f64> = (0..ns).map(|r| g3 * (0..3).map(|s| d[r * ns + s]).sum::<f64>()).collect();
                for a in 0..nu {
                    let mut s = 0.0;
                    for r in 0..ns {
                        s += g.b[r * nu + a] * dm_p[r];
                    }
                    for j in 0..nc {
                        k[ua[a] * nd + pa[j]] += (s - g.bv[a]) * g.np[j] * w;
                    }
                }
                // ∂(vätskeinnehåll)/∂ε_v, ∂/∂u_w och ∂/∂tr σ̄
                let (c_ev, c_p, c_tr) = content_derivatives(props, n0);
                let dtr_dp: f64 = (0..3).map(|r| dm_p[r]).sum();
                for i in 0..nc {
                    for j in 0..nc {
                        let mut s = 0.0;
                        for r in 0..ndim {
                            for c in 0..ndim {
                                s += g.dnp[i][r] * kt[r * 3 + c] * g.dnp[j][c];
                            }
                        }
                        let mut v = s / gw;
                        if fm.transient {
                            v += g.np[i] * (c_p + c_tr * dtr_dp) * g.np[j] / fm.dt;
                        }
                        k[pa[i] * nd + pa[j]] += v * w;
                    }
                    let mut dflow = 0.0;
                    for c in 0..ndim {
                        dflow -= g.dnp[i][c] * dfde[c];
                    }
                    for b in 0..nu {
                        let mut v = dflow * dedev * g.bv[b];
                        if fm.transient {
                            let dtr: f64 = (0..3).map(|r| db[r * nu + b]).sum();
                            v += g.np[i] * (c_ev * g.bv[b] + c_tr * dtr) / fm.dt;
                        }
                        k[pa[i] * nd + ua[b]] += v * w;
                    }
                }
            }
        }
        Ok(ElemResponse { fint, k: tangent.then_some(k) })
    }

    /// Nytt tillstånd: noll spänning, materialets starttillstånd, ε₀ = 0, e₀ = 0.
    pub fn fresh_state(&self, mat: &dyn MaterialModel, n_ips: usize) -> Vec<f64> {
        let lay = self.layout(mat);
        let stride = lay.stride();
        let mut s = vec![0.0; n_ips * stride];
        for q in 0..n_ips {
            mat.init_state(&mut s[q * stride + lay.mat()..q * stride + lay.mat() + lay.nst]);
        }
        s
    }

    /// Utdata S, E, materialets fält, FLVEL (volymflöde per ytenhet) och VOIDR.
    pub fn soil_output(&self, ctx: &ElemCtx, props: Option<&PoroProps>, v: &[f64], state: &[f64], gravity: [f64; 3]) -> Result<Vec<IpField>> {
        let mat = ctx.material(self.name)?;
        let lay = self.layout(mat);
        let stride = lay.stride();
        let gs = self.geometry(&ctx.x, self.thickness(ctx)?)?;
        let ns = lay.ns;
        let (uu, pp) = self.split(v);
        let (sl, el) = common::stress_labels(self.mode);
        let mut s = IpField::new("S", sl);
        let mut e = IpField::new("E", el);
        let groups = mat.output_fields(self.mode);
        let mut extra: Vec<IpField> = groups.iter().map(|(n, l)| IpField::new(n, l)).collect();
        let ndim = self.ndim();
        let fl: &[&str] = if ndim == 3 { &["FLVEL1", "FLVEL2", "FLVEL3"] } else { &["FLVEL1", "FLVEL2"] };
        let mut flv = IpField::new("FLVEL", fl);
        let mut vr = IpField::new("VOIDR", &["VOIDR"]);
        for (q, g) in gs.iter().enumerate() {
            let st = if state.len() >= (q + 1) * stride { &state[q * stride..(q + 1) * stride] } else { &[][..] };
            let sig: Vec<f64> = if st.len() >= ns { st[..ns].to_vec() } else { vec![0.0; ns] };
            s.points.push((q + 1, 0, sig));
            e.points.push((q + 1, 0, common::strain(&g.b, &uu)));
            if !groups.is_empty() && st.len() == stride {
                let vals = mat.output_field_values(self.mode, &st[lay.mat()..lay.mat() + lay.nst]);
                let mut kk = 0;
                for (f, (_, l)) in extra.iter_mut().zip(&groups) {
                    let v: Vec<f64> = (0..l.len()).map(|i| vals.get(kk + i).copied().unwrap_or(0.0)).collect();
                    kk += l.len();
                    f.points.push((q + 1, 0, v));
                }
            }
            if let Some(props) = props {
                let e0 = st.get(lay.e0()).copied().unwrap_or(0.0);
                let ev: f64 = g.bv.iter().zip(&uu).map(|(a, b)| a * b).sum();
                let (kt, _) = props.perm.at(perm_void_ratio(e0, ev));
                let p: f64 = g.np.iter().zip(&pp).map(|(a, b)| a * b).sum();
                let sig = if st.len() >= ns { &st[..ns] } else { &[0.0; 6][..ns] };
                let ee = void_ratio_grains(props, e0, ev, p, sig[0] + sig[1] + sig[2]);
                let mut grad = [0.0; 3];
                for (i, d) in g.dnp.iter().enumerate() {
                    for c in 0..ndim {
                        grad[c] += d[c] * pp[i];
                    }
                }
                let f: Vec<f64> = (0..ndim).map(|r| -(0..ndim).map(|c| kt[r * 3 + c] * (grad[c] - gravity[c])).sum::<f64>() / props.perm.gamma_w).collect();
                flv.points.push((q + 1, 0, f));
                vr.points.push((q + 1, 0, vec![ee]));
            }
        }
        let mut out = vec![s, e];
        out.extend(extra);
        if props.is_some() {
            out.push(flv);
            out.push(vr);
        }
        Ok(out)
    }

    /// Tyngd (ρ_torr·g + n·ρw·g per referensvolym) som elementvektor:
    /// `dry` = ρ_torr·g och `fluid` = ρw·g med aktuell porositet n = e/(1 + e).
    pub fn gravity(&self, ctx: &ElemCtx, v: &[f64], state: &[f64], dry: [f64; 3], fluid: [f64; 3]) -> Result<Vec<f64>> {
        let mat = ctx.material(self.name)?;
        let lay = self.layout(mat);
        let stride = lay.stride();
        let gs = self.geometry(&ctx.x, self.thickness(ctx)?)?;
        let (uu, _) = self.split(v);
        let nm = self.ndim();
        let mut out = vec![0.0; self.n_dof()];
        for (q, g) in gs.iter().enumerate() {
            let e0 = state.get(q * stride + lay.e0()).copied().unwrap_or(0.0);
            let ev: f64 = g.bv.iter().zip(&uu).map(|(a, b)| a * b).sum();
            // aktuell porositet n = e/(1 + e) per referensvolym
            let e = void_ratio(e0, ev);
            let n = e / (1.0 + e);
            for (k, &nk) in g.ip.n.iter().enumerate() {
                for c in 0..nm {
                    out[self.u_index(k, c)] += nk * (dry[c] + n * fluid[c]) * g.ip.w;
                }
            }
        }
        Ok(out)
    }

    /// Trycket i mittnoderna (lokalt nodnummer, värde) interpolerat från
    /// hörnen (POR redovisas där).
    pub fn midside_pressure(&self, pp: &[f64]) -> Vec<(usize, f64)> {
        let nat = shapes::node_coords(self.shape);
        (self.n_corner()..self.shape.n_nodes())
            .map(|k| {
                let (np, _) = shapes::shape(self.pshape, nat[k]);
                (k, np.iter().zip(pp).map(|(a, b)| a * b).sum())
            })
            .collect()
    }

    fn mech_elem(&self) -> Result<&'static dyn ElementType> {
        match super::lookup(self.mech) {
            Some(m) => Ok(m),
            None => fail(format!("{}: det mekaniska elementet {} finns inte i registret", self.name, self.mech)),
        }
    }

    /// Mekanisk vektor (nodvisa förskjutningar) → elementvektor (nollor i trycken).
    pub fn from_mech(&self, m: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_dof()];
        for (a, &x) in m.iter().enumerate().take(self.nu()) {
            out[self.ua(a)] = x;
        }
        out
    }
}

/// Portalet ur begynnelseportalet och volymtöjningen:
/// VOIDR = (1 + e₀)/(1 − ε_v) − 1 vid små töjningar (skiljer sig från
/// e₀ + (1 + e₀)ε_v i andra ordningen).
pub fn void_ratio(e0: f64, ev: f64) -> f64 {
    (1.0 + e0) / (1.0 - ev) - 1.0
}

/// Portalet med kompressibla korn (VOIDR): 1 + e = (1 + e₀)·J·ρ_g/ρ_g0 med
/// J = 1/(1 − ε_v) och ρ_g/ρ_g0 = 1 + u_w/K_g − tr σ̄/(3(1 − n₀)K_g).
pub fn void_ratio_grains(props: &PoroProps, e0: f64, ev: f64, p: f64, tr_sigma: f64) -> f64 {
    if props.k_grain <= 0.0 {
        return void_ratio(e0, ev);
    }
    let n0 = e0 / (1.0 + e0);
    let rho = 1.0 + p / props.k_grain - tr_sigma / (3.0 * (1.0 - n0) * props.k_grain);
    (1.0 + e0) / (1.0 - ev) * rho - 1.0
}

/// Portalet för permeabiliteten k(e): e₀ + (1 + e₀) ε_v.
pub fn perm_void_ratio(e0: f64, ev: f64) -> f64 {
    e0 + (1.0 + e0) * ev
}

/// Vätskeinnehåll per referensvolym (linjäriserat, med
/// kompressibla korn och vätska):
/// m = n₀ + ε_v + (1 − n₀) u_w/K_g − tr σ̄/(3K_g) + n₀ u_w/K_w.
/// Kornens densitet ändras med porvattentrycket och den effektiva spänningen,
/// vätskans med porvattentrycket.
pub fn fluid_content(props: &PoroProps, n0: f64, ev: f64, p: f64, tr_sigma: f64) -> f64 {
    let (a, b, c) = content_derivatives(props, n0);
    n0 + a * ev + b * p + c * tr_sigma
}

/// (∂m/∂ε_v, ∂m/∂u_w, ∂m/∂tr σ̄) för `fluid_content`.
pub fn content_derivatives(props: &PoroProps, n0: f64) -> (f64, f64, f64) {
    let mut cp = 0.0;
    let mut ct = 0.0;
    if props.k_grain > 0.0 {
        cp += (1.0 - n0) / props.k_grain;
        ct = -1.0 / (3.0 * props.k_grain);
    }
    if props.k_fluid > 0.0 {
        cp += n0 / props.k_fluid;
    }
    (1.0, cp, ct)
}

impl ElementType for SoilElem {
    fn name(&self) -> &'static str {
        self.name
    }
    fn family(&self) -> Family {
        match self.mode {
            StressMode::ThreeD => Family::Continuum3D,
            StressMode::Axisymmetric => Family::Axisymmetric,
            _ => Family::Continuum2D,
        }
    }
    fn dim(&self) -> usize {
        if self.mode == StressMode::ThreeD { 3 } else { 2 }
    }
    fn n_nodes(&self) -> usize {
        self.shape.n_nodes()
    }
    fn node_dofs(&self, k: usize) -> &'static [Dof] {
        let corner = k < self.n_corner();
        match (self.mode == StressMode::ThreeD, corner) {
            (true, true) => &[U1, U2, U3, POR],
            (true, false) => &[U1, U2, U3],
            (false, true) => &[U1, U2, POR],
            (false, false) => &[U1, U2],
        }
    }
    fn n_ips(&self) -> usize {
        self.rule.count(self.pdim())
    }
    fn vtk_type(&self) -> u8 {
        self.shape.vtk()
    }
    fn faces(&self) -> &'static [&'static [usize]] {
        self.shape.faces()
    }
    fn init_state(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        Ok(self.fresh_state(ctx.material(self.name)?, self.n_ips()))
    }
    fn stiffness(&self, _ctx: &ElemCtx) -> Result<Vec<f64>> {
        not_implemented("linjär styvhet utanför *SOILS/*GEOSTATIC", self.name)
    }
    fn internal_force(&self, _ctx: &ElemCtx, _u: &[f64], _du: &[f64], _so: &[f64], _sn: &mut Vec<f64>, _tangent: bool) -> Result<ElemResponse> {
        not_implemented("analys utanför *SOILS/*GEOSTATIC (porvattentryckselement)", self.name)
    }
    fn distributed_load(&self, ctx: &ElemCtx, load: &DistLoad, u: &[f64]) -> Result<Vec<f64>> {
        let (uu, _) = self.split(u);
        Ok(self.from_mech(&self.mech_elem()?.distributed_load(ctx, load, &uu)?))
    }
    fn volume_weights(&self, ctx: &ElemCtx) -> Result<Vec<f64>> {
        let gs = self.geometry(&ctx.x, self.thickness(ctx)?)?;
        let ips: Vec<Ip> = gs.into_iter().map(|g| g.ip).collect();
        Ok(common::volume_weights(&ips))
    }
    fn output(&self, ctx: &ElemCtx, u: &[f64], state: &[f64]) -> Result<Vec<IpField>> {
        self.soil_output(ctx, None, u, state, [0.0; 3])
    }
}
