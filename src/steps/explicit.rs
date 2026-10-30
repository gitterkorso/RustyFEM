//! Explicit dynamik (*DYNAMIC, EXPLICIT) med centraldifferensmetoden och
//! diagonal massa.
//!
//! ```text
//! ü(i)     = M⁻¹ (P(i) − I(i) − C u̇(i−½) + F_k(i))
//! u̇(i+½)  = u̇(i−½) + ½(Δt(i+1) + Δt(i)) ü(i),   u̇(½) = u̇(0) + ½Δt(1) ü(0)
//! u(i+1)   = u(i) + Δt(i+1) u̇(i+½)
//! ```
//!
//! (Belytschko, Liu & Moran, "Nonlinear Finite Elements", kap. 6.) F_k är
//! kontaktkrafterna (`explicit_contact`). Utdata för hastighet vid hela
//! inkrement är u̇(i) = u̇(i−½) + ½Δt(i) ü(i). Rotationsfrihetsgraderna
//! integreras på samma sätt (linjärt, I θ̈ = M) med den explicita
//! rotationströgheten (`explicit_mass`).
//!
//! Delar:
//! - `explicit_mass.rs`: diagonal massa (skal, balkar, punktmassor), massskalning.
//! - `explicit_dt.rs`: elementvis stabilitetsgräns.
//! - `explicit_bulk.rs`: volymviskositet.
//! - `explicit_energy.rs`: energier och history-utdata.
//! - `explicit_scaling.rs`: *FIXED/*VARIABLE MASS SCALING.
//! - `explicit_contact.rs`: kontakt med straffmetod.
//!
//! Tidsinkrement:
//! - `DIRECT USER CONTROL` (eller `FIXED TIME INCREMENTATION` med angivet Δt):
//!   fast Δt från dataraden (steget delas i ⌈T/Δt⌉ lika inkrement).
//! - annars elementvis stabilitetsgräns (`explicit_dt`), multiplicerad med
//!   `SCALE FACTOR` (standard 1), omräknad varje inkrement med aktuell
//!   geometri och densitet ρ₀V₀/V (NLGEOM), materialets effektiva moduler
//!   (`explicit_moduli`) och volymviskositetens kvadratiska term (utom med
//!   *FIXED MASS SCALING, DT=, där Δt = DT hela steget). Skal och
//!   membran får den förbättrade gränsen (IMPROVED DT METHOD=YES, standard),
//!   solider den konventionella. En global skattning (standard utan
//!   ELEMENT BY ELEMENT) används inte: RustyFEM räknar alltid elementvis, vilket
//!   ger kortare eller lika långa inkrement. 3D-solider får den konventionella
//!   gränsen även med IMPROVED DT METHOD=YES (den förbättrade formeln för
//!   solider är inte implementerad).
//!
//! Elementborttagning vid skada (`steps/model_change.rs`):
//! elementet tas bort när skadan nått Dmax i alla integrationspunkter; borttagna
//! element ger ingen kraft, volymviskositet eller stabilitetsgräns. *EOS-material
//! får begynnelseenergi och begynnelsetryck i analysens början
//! (`materials::eos::apply_initial`), och ALLIE börjar med Σ m Eₘ₀.
//!
//! Föreskrivna förskjutningar med amplitud: u(i+1) = A(t_{i+1}) exakt och
//! halvstegshastigheten (u(i+1) − u(i))/Δt; utdata V = u̇(i−½) + ½Δt·A″ och
//! A = A″ (kontrollerat per inkrement för SMOOTH STEP).
//!
//! Kontinuumelement med NLGEOM uppdaterar spänningen
//! (`kinematics::continuum::set_explicit_end_config`): Δε = sym(ΔF F_{n+1}⁻¹),
//! materialet med σₙ, sedan Jaumann-termen ΔWσₙ − σₙΔW additivt.
//!   Stänger får en extra faktor 0,99 efter första inkrementet. Δt minskas
//!   linjärt mot stegets slut (se `run`).
//!
//! Lasternas standardamplitud är steg. NLGEOM följer
//! steget (explicit analys har NLGEOM=YES som standard).
//!
//! Område coupled2: *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT körs också
//! här, med temperaturen integrerad framåt Euler (`explicit_coupled_full.rs`):
//! RT-elementen får explicit hourglass-kontroll (CPE4RT, CPS4RT, C3D8RT),
//! nodtemperaturerna till elementen (reducerade element: volymviktat
//! medelvärde) och den termiska stabilitetsgränsen; volymviskositeten använder
//! c_d·Δtₑ när den termiska gränsen styr elementet.

// område explicit_adv: SPH, DEM, Euler och ALE (steps/explicit_adv.rs)
#[path = "explicit_adv.rs"]
pub mod adv;
#[path = "explicit_bulk.rs"]
mod bulk;
#[path = "explicit_constraints.rs"]
mod constraints;
#[path = "explicit_contact.rs"]
pub mod contact;
// område coupled2: *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT i denna lösare
#[path = "explicit_coupled_full.rs"]
pub mod coupled_full;
#[path = "explicit_dt.rs"]
pub mod dt;
#[path = "explicit_energy.rs"]
mod energy;
#[path = "explicit_mass.rs"]
pub mod mass;
#[path = "explicit_moduli.rs"]
mod moduli;
#[path = "explicit_scaling.rs"]
mod scaling;
#[path = "explicit_shell.rs"]
mod shell;
#[path = "explicit_solid.rs"]
mod solid;
// område explicit_adv: explicit hourglass-kontroll för C3D8R, CPE4R, CPS4R med NLGEOM
#[path = "explicit_solid_nlgeom.rs"]
pub mod solid_nl;

/// Volymviskositetens kvadratiska faktor (för tester).
pub use bulk::quadratic_factor as bulk_quadratic_factor;

/// Hourglass-kontrollens relaxationskärna (för tester).
pub mod hourglass {
    pub use super::shell::{relax, Hourglass};
}

use super::api::{StepCtx, StepType};
use super::dynamic::amplitude::LoadEval;
use super::dynamic::{self, global_matrices, initial_velocity, output_schedule, NodeMatrix, Schedule};
use crate::elements::{Family, MassKind};
use crate::fem::assembly::{self, DofMap, Prepared};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, StepDef};
use crate::fem::output::FrameData;
use crate::interactions::InteractionCtx;
use crate::model::{fail, FemError, Result};
use crate::par;
use mass::LumpedMass;

pub struct Explicit;

pub static STEP_TYPES: &[&dyn StepType] = &[&Explicit];

/// Kontaktinteraktioner hanteras av `explicit_contact` (straffmetod) och inte
/// av implicit-kontakten i `interactions/contact.rs`.
fn is_contact(name: &str) -> bool {
    name.starts_with("*CONTACT")
}

/// Krafter vid ett inkrement.
struct Forces {
    /// Inre kraft (element, fjädrar, kopplingselement).
    fint: Vec<NodeVec>,
    /// Yttre last.
    fext: Vec<NodeVec>,
    /// Viskösa krafter (Rayleigh, dämpare, volymviskositet).
    fvis: Vec<NodeVec>,
    /// Kontaktkrafter (på noderna).
    fcon: Vec<NodeVec>,
    state: Vec<Vec<f64>>,
    /// Volymtöjningshastighet per element i volymviskositeten.
    rates: Vec<f64>,
    /// Kontaktens friktionsdissipation och straffarbete i inkrementet.
    fric: f64,
    /// Explicit-tillstånd per element (S4R:s hourglass och drill).
    xstate: Vec<Vec<f64>>,
    /// Ökning av den artificiella energin (ALLAE) i inkrementet.
    dae: f64,
}

/// Elementets explicit-specifika formulering.
#[derive(Clone)]
enum Kernel {
    /// Elementets egen `internal_force`.
    Element,
    /// S4R med explicit hourglass och drill och den implicita styvheten.
    S4r(shell::Hourglass, Vec<f64>),
    /// S3R med explicit drill.
    S3r(Vec<f64>),
    /// C3D8R, CPE4R, CPS4R med explicit hourglass.
    SolidHg(shell::Hourglass, &'static crate::elements::solid_higher::Solid),
    /// Område explicit_adv: som `SolidHg` men med NLGEOM (inkrementell hourglass).
    SolidHgNl(shell::Hourglass, &'static crate::elements::solid_higher::Solid),
}

struct Explicit1<'p, 'm> {
    p: &'p Prepared<'m>,
    dm: DofMap,
    nlgeom: bool,
    lev: &'p LoadEval<'p>,
    mass: LumpedMass,
    c: Option<NodeMatrix>,
    time0: f64,
    bulk: bulk::BulkViscosity,
    contact: Option<contact::ExplicitContact>,
    dim: usize,
    kernels: Vec<Kernel>,
    /// Förbättrad elementvis stabilitetsgräns (IMPROVED DT METHOD=YES).
    improved_dt: bool,
    /// Kopplad temperatur–förskjutning: elementen får nodtemperaturerna (område coupled2).
    coupled: bool,
    /// Vikter för integrationspunktens temperatur per element (tom = medelvärde).
    tweights: Vec<Vec<f64>>,
    /// Område explicit_adv: partiklar (SPH, DEM), Euler-element och ALE.
    adv: Option<adv::Advanced>,
}

impl Explicit1<'_, '_> {
    fn coords(&self, e: usize, u: &[NodeVec]) -> Vec<[f64; 3]> {
        let m = self.p.model;
        m.elements[e].nodes.iter().map(|&n| std::array::from_fn(|c| m.nodes[n][c] + if self.nlgeom { u[n][c] } else { 0.0 })).collect()
    }

    /// Inre krafter från element och icke-kontaktinteraktioner: (kraft,
    /// elementtillstånd, explicit-tillstånd, ökning av ALLAE).
    #[allow(clippy::type_complexity)]
    fn internal(&self, u: &[NodeVec], du: &[NodeVec], state_old: &[Vec<f64>], xold: &[Vec<f64>], time: f64, dt: f64) -> Result<(Vec<NodeVec>, Vec<Vec<f64>>, Vec<Vec<f64>>, f64)> {
        let p = self.p;
        let m = p.model;
        let ne = m.elements.len();
        let mut fint = vec![[0.0; SLOTS]; m.nodes.len()];
        let res = par::map(ne, par::threads(), |e| -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> {
            #[allow(unused_mut)]
            let mut ctx = p.ctx(e, self.nlgeom, time, dt);
            if self.coupled {
                coupled_full::set_temps(p, &mut ctx, e, u, du, self.tweights.get(e).map(|w| &w[..]).unwrap_or(&[]));
            }
            let ue = p.gather(e, u);
            // område explicit_adv: utan NLGEOM räknar elementen (töjning och hourglass)
            // med materialets förskjutning, utan ALE-nätets omflyttningar
            let ue = match self.adv.as_ref().filter(|_| !self.nlgeom).and_then(|ad| ad.hourglass_u(&m.elements[e].nodes, &ue)) {
                Some(x) => x,
                None => ue,
            };
            let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
            // borttagna element (*MODEL CHANGE, skada) ger ingen kraft (steps/model_change.rs)
            if p.activity.is_removed(e) {
                return Ok((vec![0.0; ue.len()], state_old[e].clone(), xold[e].clone(), 0.0));
            }
            if let Kernel::S4r(mode, k) = &self.kernels[e] {
                let (f, st, dae) = shell::s4r_force(&ctx, k, &ue, &xold[e], *mode).map_err(wrap)?;
                let dw = shell::drill_work(&ctx, &xold[e], &st, &p.gather(e, du)).map_err(wrap)?;
                return Ok((f, state_old[e].clone(), st, dae + dw));
            }
            if let Kernel::S3r(k) = &self.kernels[e] {
                let (f, st) = shell::s3r_force(&ctx, k, &ue).map_err(wrap)?;
                let dw = shell::drill_work(&ctx, &xold[e], &st, &p.gather(e, du)).map_err(wrap)?;
                return Ok((f, state_old[e].clone(), st, dw));
            }
            let due = p.gather(e, du);
            let mut sn = Vec::new();
            crate::kinematics::continuum::set_explicit_end_config(true);
            let hg_nl = matches!(self.kernels[e], Kernel::SolidHgNl(..));
            solid_nl::set_skip_standard(hg_nl);
            let r = match crate::steps::model_change::internal_force(p, e, &ctx, &ue, &due, &state_old[e], &mut sn, false) {
                Some(r) => r,
                None => m.elements[e].etype.internal_force(&ctx, &ue, &due, &state_old[e], &mut sn, false),
            };
            solid_nl::set_skip_standard(false);
            crate::kinematics::continuum::set_explicit_end_config(false);
            let mut r = r.map_err(wrap)?;
            if let Kernel::SolidHgNl(mode, el) = &self.kernels[e] {
                let (st, dae) = solid_nl::hourglass_force(&ctx, el, m.elements[e].etype.family(), &mut r.fint, &ue, &due, &xold[e], *mode).map_err(wrap)?;
                return Ok((r.fint, sn, st, dae));
            }
            if let Kernel::SolidHg(mode, el) = &self.kernels[e] {
                let (st, dae) = coupled_full::hourglass_force(&ctx, el, m.elements[e].etype.family(), &mut r.fint, &ue, &xold[e], *mode).map_err(wrap)?;
                return Ok((r.fint, sn, st, dae));
            }
            Ok((r.fint, sn, Vec::new(), 0.0))
        });
        let mut state_new = Vec::with_capacity(ne);
        let mut xnew = Vec::with_capacity(ne);
        let mut dae = 0.0;
        for (e, r) in res.into_iter().enumerate() {
            let (fe, sn, xs, de) = r?;
            for (a, &(n, s)) in p.edofs[e].iter().enumerate() {
                fint[n][s] += fe[a];
            }
            state_new.push(sn);
            xnew.push(xs);
            dae += de;
        }
        let ictx = InteractionCtx { model: m, time, dt, nlgeom: self.nlgeom };
        for it in p.interactions.iter().filter(|it| !is_contact(&it.name())) {
            it.contribute(&ictx, u, &mut fint, None)?;
        }
        Ok((fint, state_new, xnew, dae))
    }

    /// Alla krafter vid stegtiden t med halvstegshastigheten `v`.
    #[allow(clippy::too_many_arguments)]
    fn forces(&mut self, t: f64, dt: f64, u: &[NodeVec], du: &[NodeVec], v: &[NodeVec], state: &[Vec<f64>], xold: &[Vec<f64>], commit: bool) -> Result<Forces> {
        let (fint, state_new, xstate, dae) = self.internal(u, du, state, xold, self.time0 + t, dt)?;
        if dt > 0.0 {
            let mut ratio = std::mem::take(&mut self.bulk.cratio);
            moduli::update(self.p, &|e| self.coords(e, u), du, state, &state_new, &mut ratio);
            self.bulk.cratio = ratio;
        }
        let fext = assembly::external(self.p, &self.lev.cloads_at(t), &self.lev.dloads_at(t), u, self.nlgeom, self.time0 + t)?;
        let mut fvis = match &self.c {
            Some(c) => c.mul(v),
            None => vec![[0.0; SLOTS]; u.len()],
        };
        let rates = self.bulk.forces(self.p, &|e| self.coords(e, u), v, &mut fvis);
        // område acoustics2: volymviskositetens arbete i EOS-energin
        let mut state_new = state_new;
        if dt > 0.0 {
            let w = self.bulk.work.lock().unwrap();
            crate::materials::eos::add_bulk_work(self.p, &mut state_new, &w, dt);
        }
        let mut fcon = vec![[0.0; SLOTS]; u.len()];
        let mut fric = 0.0;
        if let Some(ct) = self.contact.as_mut() {
            fric = ct.forces(self.p.model, u, du, dt, &self.mass, &mut fcon, commit)?;
        }
        // område explicit_adv: partiklarnas, Euler-elementens och ALE-nätets krafter
        let mut fint = fint;
        let mut fext = fext;
        if let Some(ad) = self.adv.as_mut() {
            let ev = adv::Eval { time: self.time0 + t, dt, u, du, v, state_old: state, fixed: &self.dm.constrained };
            ad.forces(self.p, &ev, &mut adv::Sink { state_new: &mut state_new, fint: &mut fint, fext: &mut fext, fvis: &mut fvis, fcon: &mut fcon, mass: &mut self.mass })?;
            fric += ad.fric;
        }
        Ok(Forces { fint, fext, fvis, fcon, state: state_new, rates, fric, xstate, dae })
    }

    fn accel(&self, f: &Forces) -> Vec<NodeVec> {
        let nn = f.fint.len();
        (0..nn)
            .map(|n| {
                let r: NodeVec = std::array::from_fn(|s| f.fext[n][s] + f.fcon[n][s] - f.fint[n][s] - f.fvis[n][s]);
                self.mass.solve(n, &r, |s| self.dm.eq[n][s] != assembly::NONE)
            })
            .collect()
    }

    /// Reaktion i låsta frihetsgrader: I + C u̇ + M ü − P − F_k.
    fn reactions(&self, f: &Forces, a: &[NodeVec]) -> Vec<NodeVec> {
        let nn = a.len();
        let mut rf = vec![[0.0; SLOTS]; nn];
        for n in 0..nn {
            let ma = self.mass.mul(n, &a[n]);
            for s in 0..SLOTS {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    rf[n][s] = f.fint[n][s] + f.fvis[n][s] + ma[s] - f.fext[n][s] - f.fcon[n][s];
                }
            }
        }
        rf
    }

    #[allow(clippy::too_many_arguments)]
    fn frame(&self, u: &[NodeVec], v: &[NodeVec], a: &[NodeVec], f: &Forces, state: &[Vec<f64>], t: f64, inc: usize) -> Result<FrameData> {
        // RF = I + C u̇ + M ü − P − F_k i låsta frihetsgrader; assembly::frame
        // räknar fint − fext, så den dynamiska delen läggs i fint
        let nn = u.len();
        let fdyn: Vec<NodeVec> = (0..nn)
            .map(|n| {
                let ma = self.mass.mul(n, &a[n]);
                std::array::from_fn(|s| f.fint[n][s] + f.fvis[n][s] + ma[s] - f.fcon[n][s])
            })
            .collect();
        let mut fr = assembly::frame(self.p, &self.dm, u, &fdyn, &f.fext, state, self.nlgeom, self.time0 + t, t, format!("Increment {inc}: Step Time = {t:.4}"))?;
        add_motion(&mut fr, self.dim, &self.p.active, v, a);
        // område explicit_adv: partiklarnas och Euler-elementens utdata
        if let Some(ad) = &self.adv {
            ad.output(self.p, &mut fr, u, state)?;
        }
        if let Some(ct) = &self.contact {
            ct.output(self.p.model, &mut fr);
        }
        Ok(fr)
    }

    /// Odämpad elementvis gräns per element (oändlig utan massa), med
    /// massskalningsfaktorerna `f`.
    fn element_limits(&self, u: &[NodeVec], cache: &mut [Option<(f64, f64, f64)>], f: &[f64]) -> Result<Vec<(f64, f64, f64)>> {
        let p = self.p;
        let m = p.model;
        let mut out = Vec::with_capacity(m.elements.len());
        for (e, el) in m.elements.iter().enumerate() {
            let ctx = p.ctx(e, false, 0.0, 0.0);
            // borttagna element styr inte stabilitetsgränsen
            if ctx.density <= 0.0 || el.etype.family() == Family::Connector || p.activity.is_removed(e) {
                out.push((f64::INFINITY, 0.0, 0.0));
                continue;
            }
            let x = self.coords(e, u);
            let fam = el.etype.family();
            let geometric = dt::geometry(fam, el.etype.name(), &x).is_some();
            let lim = if geometric || cache[e].is_none() {
                let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
                let me = el.etype.mass(&ctx, MassKind::Lumped).map_err(wrap)?;
                let nd = p.edofs[e].len();
                let mut diag: Vec<f64> = (0..nd).map(|i| me.get(i * nd + i).copied().unwrap_or(0.0)).collect();
                // skalens och balkarnas rotationströghet som i explicit_mass
                if fam == Family::Shell {
                    let r = mass::shell_rotary_factor(&ctx.x);
                    let mut mt = 0.0;
                    for (a, &(_, s)) in p.edofs[e].iter().enumerate() {
                        if s == 0 {
                            mt = diag[a];
                        }
                        if (3..6).contains(&s) {
                            diag[a] = mt * r;
                        }
                    }
                }
                let mut l = dt::element_limit(p, e, &ctx, &x, &diag).map_err(wrap)?;
                // standard IMPROVED DT METHOD=YES (skal och membran)
                if self.improved_dt {
                    if let (Some((dte, lchar, c)), Some((_, nu))) = (l, ctx.material.and_then(|mm| mm.elastic_moduli())) {
                        if let Some(li) = dt::improved_plane_stress(fam, &x, nu) {
                            if c > 0.0 && li > lchar {
                                l = Some((dte.max(li / c), lchar, c));
                            }
                        }
                    }
                }
                cache[e] = Some(l.unwrap_or((f64::INFINITY, 0.0, 0.0)));
                cache[e].unwrap()
            } else {
                cache[e].unwrap()
            };
            let s = f.get(e).copied().unwrap_or(1.0).max(0.0).sqrt();
            // materialets effektiva våghastighet (plasticitet, `explicit_moduli`;
            // kontrollerat för en plastisk stång)
            let mut r = self.bulk.cratio.get(e).copied().unwrap_or(1.0).max(1e-6);
            // aktuell densitet ρ₀V₀/V vid NLGEOM: c_d = √(M/ρ) ökar med volymen
            if self.nlgeom && matches!(fam, Family::Continuum2D | Family::Continuum3D) {
                if let (Some(g), Some(g0)) = (dt::geometry(fam, el.etype.name(), &x), dt::geometry(fam, el.etype.name(), &ctx.x)) {
                    if g.vol > 0.0 && g0.vol > 0.0 {
                        r *= (g.vol / g0.vol).sqrt();
                    }
                }
            }
            out.push((lim.0 * s / r, lim.1, lim.2 * r / s.max(1e-300)));
        }
        Ok(out)
    }

    /// Dämpad elementvis gräns per element givet odämpade gränser och
    /// volymviskositetens töjningshastigheter.
    fn damped(&self, lims: &[(f64, f64, f64)], rates: &[f64]) -> Result<Vec<f64>> {
        let m = self.p.model;
        let mut out: Vec<f64> = lims.iter().map(|l| l.0).collect();
        let mut xi = vec![0.0; out.len()];
        for (i, be) in self.bulk.elems.iter().enumerate() {
            let (_, lchar, c) = lims[be.e];
            if c > 0.0 {
                xi[be.e] = self.bulk.xi(i, lchar, rates.get(i).copied().unwrap_or(0.0));
            }
        }
        for (e, el) in m.elements.iter().enumerate() {
            if !out[e].is_finite() {
                continue;
            }
            let mi = el.section.and_then(|s| m.sections[s].material_index);
            let (ar, br) = dynamic::rayleigh(m, mi)?;
            let w = 2.0 / out[e];
            let x = xi[e] + 0.5 * (br * w + ar / w);
            out[e] *= dt::damping_factor(x);
        }
        Ok(out)
    }
}

/// Lägger till V, A, VR, AR (2D: VR3, AR3) i en frame.
fn add_motion(fr: &mut FrameData, dim: usize, active: &[u16], v: &[NodeVec], a: &[NodeVec]) {
    dynamic::add_va(fr, dim, v, a);
    let rot: &[usize] = if dim == 2 { &[5] } else { &[3, 4, 5] };
    if !active.iter().any(|m| rot.iter().any(|&s| m & (1 << s) != 0)) {
        return;
    }
    let (vn, an) = if dim == 2 { ("VR3", "AR3") } else { ("VR", "AR") };
    let labels = |p: &str| -> Vec<String> { if dim == 2 { vec![format!("{p}3")] } else { (1..=3).map(|i| format!("{p}{i}")).collect() } };
    fr.node_extra.push((vn.into(), labels("VR"), v.iter().map(|x| rot.iter().map(|&s| x[s]).collect()).collect()));
    fr.node_extra.push((an.into(), labels("AR"), a.iter().map(|x| rot.iter().map(|&s| x[s]).collect()).collect()));
}

/// Hourglass-kontroll per tvärsnitt från *SECTION CONTROLS (CONTROLS= på
/// tvärsnittet; standard RELAX STIFFNESS).
fn hourglass_mode(p: &Prepared, sec: &crate::fem::model::SectionDef) -> Result<shell::Hourglass> {
    let Some(name) = sec.raw.param_upper("CONTROLS") else { return Ok(shell::Hourglass::Relax) };
    let Some(sc) = p.model.extra.iter().find(|k| k.name == "SECTION CONTROLS" && k.param_upper("NAME").as_deref() == Some(name.as_str())) else {
        return sec.raw.err(format!("*SECTION CONTROLS, NAME={name} saknas"));
    };
    match sc.param_upper("HOURGLASS").as_deref() {
        None | Some("RELAX STIFFNESS") => Ok(shell::Hourglass::Relax),
        Some("STIFFNESS") => Ok(shell::Hourglass::Stiffness),
        Some(o) => sc.err(format!("HOURGLASS={o} stöds inte i explicit dynamik (RELAX STIFFNESS, STIFFNESS)")),
    }
}

/// Explicit-formulering per element.
fn kernels(p: &Prepared, nlgeom: bool) -> Result<Vec<Kernel>> {
    let m = p.model;
    let mut out = Vec::with_capacity(m.elements.len());
    for (e, el) in m.elements.iter().enumerate() {
        let k = match (el.etype.name(), el.section) {
            ("S4R", Some(s)) if !nlgeom => {
                let ctx = p.ctx(e, false, 0.0, 0.0);
                let ks = el.etype.stiffness(&ctx).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
                Kernel::S4r(hourglass_mode(p, &m.sections[s])?, ks)
            }
            (name, Some(s)) if !nlgeom && coupled_full::standard(name).is_some() => Kernel::SolidHg(hourglass_mode(p, &m.sections[s])?, coupled_full::standard(name).unwrap()),
            // område explicit_adv: NLGEOM med inkrementell hourglass
            (name, Some(s)) if nlgeom && solid::standard(name).is_some() => Kernel::SolidHgNl(hourglass_mode(p, &m.sections[s])?, solid::standard(name).unwrap()),
            ("S3R" | "S3", Some(_)) if !nlgeom => {
                let ctx = p.ctx(e, false, 0.0, 0.0);
                Kernel::S3r(el.etype.stiffness(&ctx).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?)
            }
            _ => Kernel::Element,
        };
        out.push(k);
    }
    Ok(out)
}

/// Volymviskositetens parametrar för steget (standard 0,06 och 1,2).
fn bulk_parameters(step: &StepDef) -> Result<(f64, f64)> {
    let mut bulk = (0.06, 1.2);
    for b in step.extra.iter().filter(|k| k.name == "BULK VISCOSITY") {
        let v = b.first_floats()?;
        bulk = (v.first().copied().unwrap_or(0.06), v.get(1).copied().unwrap_or(1.2));
    }
    Ok(bulk)
}

/// Dämpningsmatris: Rayleigh (αM + βK) och dämpare (*DASHPOT). `None` om ingen dämpning finns.
fn damping(p: &Prepared) -> Result<Option<NodeMatrix>> {
    let m = p.model;
    let mut any = false;
    for el in &m.elements {
        let mi = el.section.and_then(|s| m.sections[s].material_index);
        let (a, b) = dynamic::rayleigh(m, mi)?;
        any |= a != 0.0 || b != 0.0;
    }
    let mut c = if any { Some(global_matrices(p, MassKind::Lumped)?.1) } else { None };
    for it in &p.interactions {
        for (na, da, nb, db, v) in it.damping() {
            let cm = c.get_or_insert_with(|| NodeMatrix { rows: vec![Vec::new(); m.nodes.len() * SLOTS] });
            let (ga, gb) = (na * SLOTS + crate::fem::dof::slot_of(da), nb * SLOTS + crate::fem::dof::slot_of(db));
            match cm.rows[ga].iter_mut().find(|e| e.0 == gb) {
                Some(e) => e.1 += v,
                None => cm.rows[ga].push((gb, v)),
            }
        }
    }
    Ok(c)
}

impl StepType for Explicit {
    fn name(&self) -> &'static str {
        "DYNAMIC, EXPLICIT (centraldifferens)"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        // område coupled2: *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT (före explicit_coupled.rs)
        if p.name == coupled_full::PROC && p.has("EXPLICIT") {
            return Some(1);
        }
        (p.name == "DYNAMIC" && p.has("EXPLICIT")).then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        // område explicit_adv: explicit analys har NLGEOM=YES som standard
        // (steg utan NLGEOM= och utan ärvd NLGEOM=NO räknas geometriskt olinjärt)
        if ctx.step.nlgeom.is_none() && !ctx.nlgeom && !crate::steps::explicit::adv::nlgeom_off_inherited(ctx.prep.model, ctx.index) {
            ctx.nlgeom = true;
        }
        let kw = ctx.step.procedure.as_ref().unwrap();
        if ctx.step.perturbation {
            return fail("*DYNAMIC, EXPLICIT kan inte vara ett perturbationssteg");
        }
        let vals = kw.first_floats()?;
        let period = vals.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        let user_dt = vals.first().copied().filter(|&x| x > 0.0);
        let fixed = kw.has("DIRECT USER CONTROL") || (kw.has("FIXED TIME INCREMENTATION") && user_dt.is_some());
        let scale = kw.param("SCALE FACTOR").and_then(parse_f64).unwrap_or(1.0);
        let (b1, b2) = bulk_parameters(ctx.step)?;
        let prep = ctx.prep;
        let model = prep.model;
        let mut bulk = bulk::BulkViscosity::new(prep, b1, b2)?;
        bulk.nlgeom = ctx.nlgeom;
        if b1 != 0.0 && !bulk.unsupported.is_empty() {
            eprintln!(
                "varning: steg {}: volymviskositet (b1 = {b1}) finns inte för {} i RustyFEM och ignoreras där",
                ctx.step.name,
                model.elements[bulk.unsupported[0]].etype.name()
            );
        }
        let ramp = match ctx.step.header.param_upper("AMPLITUDE") {
            Some(a) => a == "RAMP",
            None => false,
        };
        // område coupled2: värmeledningen och mekanikens laster (explicit_coupled_full.rs)
        let mut thermal = None;
        let mech_loads;
        let loads_m = if kw.name == coupled_full::PROC {
            let (th, me) = coupled_full::Thermal::new(ctx, period)?;
            thermal = Some(th);
            mech_loads = me;
            &mech_loads
        } else {
            ctx.loads
        };
        let mut lev = LoadEval::new(model, loads_m, ramp, period, ctx.state.time)?;
        // en steglast verkar fullt vid t = 0 (ü₀ = M⁻¹P), medan
        // amplituder utvärderas exakt vid t = 0
        lev.step_full_at_start = true;
        let lev = lev;
        let ne = model.elements.len();
        let rules = scaling::rules(prep, ctx.step)?;
        let c = damping(prep)?;
        let (dm, _mat) = assembly::dofmap(prep, &ctx.loads.constrained(), ctx.solver)?;
        let contact = contact::ExplicitContact::build(prep, ctx.step)?;
        let kernels = kernels(prep, ctx.nlgeom)?;
        let mut sys = Explicit1 {
            p: prep,
            dm,
            nlgeom: ctx.nlgeom,
            lev: &lev,
            mass: mass::lumped(prep, &[])?,
            c,
            time0: ctx.state.time,
            bulk,
            contact,
            dim: model.dim(),
            kernels,
            improved_dt: !matches!(kw.param_upper("IMPROVED DT METHOD").as_deref(), Some("NO")),
            coupled: thermal.is_some(),
            tweights: match thermal.as_ref() {
                Some(th) => th.temp_weights()?,
                None => Vec::new(),
            },
            adv: adv::Advanced::build(prep, ctx.step, ctx.nlgeom)?,
        };
        // område explicit_adv: partiklarnas och Euler-nodernas massa
        if let Some(ad) = sys.adv.as_ref() {
            ad.patch_mass(&mut sys.mass);
        }
        let nn = model.nodes.len();
        let mut u = ctx.state.u.clone();
        for (n, s, val) in lev.bc_at(0.0) {
            if sys.dm.constrained[n] & (1 << s) != 0 {
                u[n][s] = val;
            }
        }
        // massskalning i stegets början
        let mut fscale = vec![1.0; ne];
        let mut lim_cache: Vec<Option<(f64, f64, f64)>> = vec![None; ne];
        if !rules.is_empty() {
            let lims = sys.element_limits(&u, &mut lim_cache, &fscale)?;
            let d0 = sys.damped(&lims, &[])?;
            scaling::apply(&rules, false, &d0, &mut fscale);
            sys.mass = mass::lumped(prep, &fscale)?;
            if let Some(ad) = sys.adv.as_ref() {
                ad.patch_mass(&mut sys.mass);
            }
            sys.bulk.scale_density(&fscale);
        }
        let mut v = initial_velocity(prep, ctx.state)?;
        // låsta frihetsgrader har hastigheten 0 vid stegets början (V = 0
        // skrivs där även med *INITIAL CONDITIONS, TYPE=VELOCITY), utom de
        // med amplitud nedan
        for n in 0..nn {
            for s in 0..SLOTS {
                if sys.dm.constrained[n] & (1 << s) != 0 {
                    v[n][s] = 0.0;
                }
            }
        }
        for &(n, s, dv, _) in &lev.bc_rates_at(0.0) {
            if sys.dm.constrained[n] & (1 << s) != 0 {
                v[n][s] = dv;
            }
        }
        let mut state = ctx.state.elem_state.clone();
        // *EOS: specifik energi och begynnelsetryck i analysens början (materials/eos.rs)
        // elementborttagning vid skada med den explicita regeln (steps/model_change.rs)
        prep.activity.set_explicit_rule(true);
        let ie0 = if ctx.state.time == 0.0 { crate::materials::eos::apply_initial(prep, &mut state)? } else { 0.0 };
        let ie0 = ie0 + if ctx.state.time == 0.0 { sys.adv.as_ref().map_or(0.0, |a| a.initial_energy()) } else { 0.0 };
        let zero = vec![[0.0; SLOTS]; nn];
        let x0: Vec<Vec<f64>> = vec![Vec::new(); ne];
        sys.bulk.linear_only = true;
        let mut f = sys.forces(0.0, 0.0, &u, &zero, &v, &state, &x0, true)?;
        sys.bulk.linear_only = false;
        // materialets aktuella kompressionsmodul (*EOS) redan i första inkrementet
        {
            let mut ratio = std::mem::take(&mut sys.bulk.cratio);
            moduli::update(prep, &|e| sys.coords(e, &u), &zero, &state, &f.state, &mut ratio);
            sys.bulk.cratio = ratio;
        }
        // område explicit_adv: ALE-nätets första utjämning (INITIAL MESH SWEEPS) i stegets början
        if let Some(ad) = sys.adv.as_mut() {
            ad.after_forces(&mut u, &mut v, &sys.dm.constrained);
        }
        let mut a = sys.accel(&f);
        // område coupled2: temperaturens randvillkor och θ̇ i stegets början
        // (mekaniken ovan med begynnelsetemperaturen i frame 0)
        if let Some(th) = thermal.as_mut() {
            th.start(prep, &|e| sys.coords(e, &u), &u, &f.state)?;
            coupled_full::clear_ts(&mut a);
            coupled_full::clear_ts(&mut v);
        }
        // stabilitetsgräns
        let stable = |sys: &Explicit1, th: Option<&coupled_full::Thermal>, u: &[NodeVec], rates: &[f64], cache: &mut [Option<(f64, f64, f64)>], fs: &[f64]| -> Result<(f64, usize)> {
            let lims = sys.element_limits(u, cache, fs)?;
            let d = sys.damped(&lims, rates)?;
            let mut best = (f64::INFINITY, 0usize);
            for (e, &x) in d.iter().enumerate() {
                if x < best.0 {
                    best = (x, e);
                }
            }
            // termisk gräns Lₑ²/(2α) (explicit_coupled_full.rs)
            if let Some(th) = th {
                let tb = th.stable_in(sys, u, rates);
                if tb.0 < best.0 {
                    best = tb;
                }
            }
            // område explicit_adv: partiklarnas och Euler-elementens gräns
            if let Some(tb) = sys.adv.as_ref().and_then(|a| a.stable()) {
                if tb.0 < best.0 {
                    best = tb;
                }
            }
            if !best.0.is_finite() {
                return fail("kan inte skatta stabilt tidsinkrement: inga element med massa (*DENSITY)");
            }
            Ok((best.0 * scale, best.1))
        };
        let is_truss = |e: usize| model.elements[e].etype.family() == Family::Truss;
        let (dt_first, mut dt) = match (fixed, user_dt) {
            // steget delas i ett helt antal lika inkrement
            // (n = ⌈T/Δt⌉, Δt = T/n)
            (true, Some(d)) => {
                let d = period / (period / d - 1e-9).ceil().max(1.0);
                (d, d)
            }
            (true, None) => return fail("DIRECT USER CONTROL kräver ett tidsinkrement på dataraden"),
            (false, _) => {
                let (d, e) = stable(&sys, thermal.as_ref(), &u, &f.rates, &mut lim_cache, &fscale)?;
                // stänger får 0,99 L/c i första inkrementet och
                // 0,99²·L/c därefter
                (d, if is_truss(e) { d * dt::TRUSS_LE } else { d })
            }
        };
        // kopplingselementens låsta komponenter (explicit_constraints.rs)
        if constraints::project(prep, &sys.dm, &sys.mass, ctx.nlgeom, &u, &v, dt_first, 0.5 * dt_first * dt_first, &[&f.fext, &f.fcon], &mut f.fint, &[&f.fvis])? {
            a = sys.accel(&f);
        }
        constraints::commit(prep);
        // Δt räknas om varje inkrement med aktuell geometri (NLGEOM) och
        // volymviskositetens kvadratiska term (töjningshastigheten). Med
        // *FIXED MASS SCALING, DT= behålls Δt = DT genom hela
        // steget (inkrementen är exakt DT).
        let fixed_dt_scaling = rules.iter().any(|r| !r.variable && r.dt.is_some());
        let recompute = !fixed && !fixed_dt_scaling && (ctx.nlgeom || !prep.is_linear() || b2 != 0.0 && !sys.bulk.elems.is_empty() || thermal.is_some());
        let recompute = recompute || (!fixed && !fixed_dt_scaling && sys.adv.is_some());
        // NUMBER INTERVAL (standard 20) eller TIME INTERVAL; utdata
        // skrivs i första inkrementet vid eller efter varje utdatatid. FREQUENCY=
        // gäller inte i explicit analys (då används 20 intervall).
        let sched = match output_schedule(ctx.step, period, Schedule::Interval(period / 20.0))? {
            Schedule::Every(_) => Schedule::Interval(period / 20.0),
            s => s,
        };
        let se_ok = energy::elastic_energy(prep, &|e| sys.coords(e, &u), &ctx.state.elem_state).is_some();
        let mut history = energy::History::from_step(ctx.step, period, prep.is_linear(), se_ok)?;
        let want_se = !prep.is_linear() && history.as_ref().is_some_and(|h| h.wants_se());
        // ALLIE börjar med den specifika begynnelseenergin (*EOS)
        let mut en = energy::Energies { ke: sys.mass.kinetic(&v), ie: ie0, ..Default::default() };
        let eps = 1e-9 * period;
        if let Some(h) = history.as_mut() {
            h.after_increment(0.0, 0, false, eps, &en);
        }
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            let mut f0 = sys.frame(&u, &v, &a, &f, &state, 0.0, 0)?;
            if let Some(th) = thermal.as_mut() {
                th.frame(&mut f0, &u, 0.0)?;
                f0.rf.iter_mut().for_each(|r| *r = [0.0; SLOTS]);
            }
            frames.push(f0);
        }
        let mut rf_old = sys.reactions(&f, &a);
        let mut t = 0.0f64;
        let mut inc = 0usize;
        let mut next_out = match sched {
            Schedule::Interval(d) => d,
            Schedule::Every(_) => f64::INFINITY,
        };
        let mut vhalf: Option<Vec<NodeVec>> = None;
        let mut dt_prev = 0.0;
        let mut ramp: Option<(f64, usize, f64)> = None;
        let max_inc = 100_000_000usize;
        let vms_rules: Vec<scaling::Rule> = rules.iter().filter(|r| r.variable).cloned().collect();
        while t < period - eps {
            if inc >= max_inc {
                return fail("för många inkrement i explicit steg");
            }
            if inc % 1000 == 0 {
                crate::progress::emit(crate::progress::Event::Increment { fraction: t / period, iterations: 0 })?;
            }
            // variabel massskalning
            if !vms_rules.is_empty() && inc > 0 {
                let due = vms_rules.iter().any(|r| match (r.frequency, r.intervals) {
                    (Some(k), _) => inc % k == 0,
                    (None, Some(nint)) => {
                        let d = period / nint as f64;
                        ((t / d) + 1e-9).floor() > ((t - dt_prev) / d + 1e-9).floor()
                    }
                    _ => false,
                });
                if due {
                    let mut unscaled = vec![1.0; ne];
                    let lims = sys.element_limits(&u, &mut lim_cache, &unscaled)?;
                    let d0 = sys.damped(&lims, &f.rates)?;
                    scaling::apply(&vms_rules, true, &d0, &mut unscaled);
                    // fasta skalningsfaktorer behålls där den variabla inte skalar mer
                    for e in 0..ne {
                        fscale[e] = fscale[e].max(unscaled[e]);
                    }
                    let old = sys.mass.clone();
                    sys.mass = mass::lumped(prep, &fscale)?;
                    if let Some(ad) = sys.adv.as_ref() {
                        ad.patch_mass(&mut sys.mass);
                    }
                    sys.bulk.scale_density(&fscale);
                    let _ = old;
                    a = sys.accel(&f);
                }
            }
            if recompute && inc > 0 {
                let (d, e) = stable(&sys, thermal.as_ref(), &u, &f.rates, &mut lim_cache, &fscale)?;
                dt = if is_truss(e) { d * dt::TRUSS_LE } else { d };
            }
            let mut h = if inc == 0 { dt_first } else { dt };
            if !fixed {
                // stegets slut nås utan ett kort sista inkrement:
                // när högst 10 inkrement återstår (N = ⌈R/Δt⌉) minskas Δt linjärt,
                // Δtⱼ = Δt − j·s, j = 1..N, s = (NΔt − R)/(N(N+1)/2)
                // (kontrollerat för T2D2 och CPE3).
                if ramp.is_none() && inc > 0 {
                    let r = period - t;
                    // utgångsvärdet är högst föregående inkrement (Δt kan växa
                    // språngvis med de effektiva modulerna)
                    let d0 = if dt_prev > 0.0 { dt.min(dt_prev) } else { dt };
                    let n = (r / d0 - 1e-9).ceil().max(1.0);
                    if n <= 10.0 {
                        ramp = Some(((n * d0 - r) / (0.5 * n * (n + 1.0)), 0usize, d0));
                    }
                }
                // Δt i minskningen räknas från Δt när den börjar, även när
                // stabilitetsgränsen ändras under tiden (NLGEOM, plasticitet), så
                // att steget slutar exakt
                if let Some((sl, j, d0)) = &mut ramp {
                    *j += 1;
                    h = *d0 - *j as f64 * *sl;
                }
            }
            h = h.min(period - t);
            if period - t - h < eps {
                h = period - t;
            }
            // område coupled2: volymviskositetens längd när den termiska gränsen styr
            if let Some(th) = thermal.as_ref() {
                sys.bulk.le_scale = th.bulk_scale(&sys, &u);
            }
            // u̇(i+½)
            let vh: Vec<NodeVec> = match &vhalf {
                None => (0..nn).map(|n| std::array::from_fn(|s| v[n][s] + 0.5 * h * a[n][s])).collect(),
                Some(vp) => (0..nn).map(|n| std::array::from_fn(|s| vp[n][s] + 0.5 * (h + dt_prev) * a[n][s])).collect(),
            };
            let t1 = t + h;
            let mut u1: Vec<NodeVec> = (0..nn).map(|n| std::array::from_fn(|s| u[n][s] + h * vh[n][s])).collect();
            let mut vh = vh;
            for (n, s, val) in lev.bc_at(t1) {
                if sys.dm.constrained[n] & (1 << s) != 0 {
                    vh[n][s] = (val - u[n][s]) / h;
                    u1[n][s] = val;
                }
            }
            // område coupled2: θ(i+1) = θ(i) + Δt θ̇(i)
            if let Some(th) = thermal.as_ref() {
                th.advance(&mut u1, h, t1);
                coupled_full::clear_ts(&mut vh);
            }
            let du: Vec<NodeVec> = (0..nn).map(|n| std::array::from_fn(|s| u1[n][s] - u[n][s])).collect();
            let mut f1 = sys.forces(t1, h, &u1, &du, &vh, &state, &f.xstate, true)?;
            // område explicit_adv: Euler-nodernas lägen och hastigheter efter advektionen
            if let Some(ad) = sys.adv.as_mut() {
                ad.after_forces(&mut u1, &mut vh, &sys.dm.constrained);
            }
            let mut a1 = sys.accel(&f1);
            if let Some(th) = thermal.as_mut() {
                th.update_rate(prep, &|e| sys.coords(e, &u1), &u1, t1, h, &state, &f1.state)?;
                coupled_full::clear_ts(&mut a1);
            }
            // kopplingselementens låsta komponenter: nästa inkrement antas lika långt
            if constraints::project(prep, &sys.dm, &sys.mass, ctx.nlgeom, &u1, &vh, h, h * h, &[&f1.fext, &f1.fcon], &mut f1.fint, &[&f1.fvis])? {
                a1 = sys.accel(&f1);
            }
            constraints::commit(prep);
            // hastighet vid hela inkrementet (utdata)
            let mut v1: Vec<NodeVec> = (0..nn).map(|n| std::array::from_fn(|s| vh[n][s] + 0.5 * h * a1[n][s])).collect();
            // föreskrivna frihetsgrader med amplitud: utdata är
            // amplitudens analytiska acceleration A″ och hastigheten
            // u̇(i) = u̇(i−½) + ½Δt A″ (halvstegshastigheten från förskjutningarna,
            // kontrollerat per inkrement för SMOOTH STEP)
            for &(n, s, _dv, da) in &lev.bc_rates_at(t1) {
                if sys.dm.constrained[n] & (1 << s) != 0 {
                    v1[n][s] = vh[n][s] + 0.5 * h * da;
                    a1[n][s] = da;
                }
            }
            // energier
            let rf1 = sys.reactions(&f1, &a1);
            en.ie += energy::work(&f.fint, &f1.fint, &du);
            en.ae += f1.dae;
            en.vd += energy::work(&f.fvis, &f1.fvis, &du);
            en.wk += energy::work(&f.fext, &f1.fext, &du) + energy::work(&rf_old, &rf1, &du);
            if let Some(ct) = &sys.contact {
                let (fd, pw) = ct.energy_increment(&f.fcon, &f1.fcon, &du, f1.fric);
                en.fd += fd;
                en.pw += pw;
            } else if sys.adv.is_some() {
                // område explicit_adv: partikelkontaktens energier (ALLFD, ALLPW)
                en.fd += f1.fric;
                en.pw += energy::work(&f.fcon, &f1.fcon, &du) + f1.fric;
            }
            en.ke = sys.mass.kinetic(&v1);
            u = u1;
            f = f1;
            a = a1;
            v = v1;
            rf_old = rf1;
            state = f.state.clone();
            vhalf = Some(vh);
            dt_prev = h;
            t = t1;
            inc += 1;
            let last = t >= period - eps;
            if want_se {
                en.se = energy::elastic_energy(prep, &|e| sys.coords(e, &u), &state);
            }
            if let Some(hs) = history.as_mut() {
                hs.after_increment(t, inc, last, eps, &en);
            }
            let reached = (t - next_out).abs() <= eps || t > next_out;
            if reached {
                if let Schedule::Interval(d) = sched {
                    while next_out <= t + eps {
                        next_out += d;
                    }
                }
            }
            let write = match sched {
                Schedule::Every(k) => inc % k == 0,
                Schedule::Interval(_) => reached,
            };
            if (all && write) || last {
                let mut fr = sys.frame(&u, &v, &a, &f, &state, t, inc)?;
                if let Some(th) = thermal.as_mut() {
                    coupled_full::clear_ts(&mut v);
                    th.frame(&mut fr, &u, t)?;
                }
                frames.push(fr);
            }
        }
        if ctx.opts.verbose {
            eprintln!("  explicit: {inc} inkrement, Δt = {dt:.4e}");
        }
        prep.activity.set_explicit_rule(false);
        ctx.out.frames.extend(frames);
        if let Some(h) = history {
            for (name, data) in h.data {
                ctx.out.history.push((energy::REGION.to_string(), name, data));
            }
        }
        ctx.out.increments = inc;
        ctx.out.iterations = inc;
        ctx.state.u = u;
        ctx.state.elem_state = state;
        ctx.state.reaction = rf_old;
        ctx.state.vectors.insert("V".into(), v);
        ctx.state.vectors.insert("A".into(), a);
        ctx.state.time += period;
        Ok(())
    }
}
