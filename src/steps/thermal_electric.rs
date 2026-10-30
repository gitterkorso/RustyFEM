//! *COUPLED THERMAL-ELECTRICAL (stationär eller transient med fasta
//! inkrement) med elementen i `elements/thermal_electric.rs`.
//!
//! Ekvationer:
//!
//! - laddningsbalans: ∫ ∇δφ · σ(T) ∇φ dV = ∫ δφ J dS + Σ I (strömmar *CECURRENT,
//!   *DECURRENT CBF/CSn); strömtätheten J = −σ ∇φ;
//! - värmebalans som *HEAT TRANSFER (`steps/heat.rs`) med Joulevärmen
//!   P = η ∇φ · σ ∇φ per volym som källa (η från *JOULE HEAT FRACTION).
//!
//! Varje inkrement löses med fixpunktsiteration mellan fälten (potentialen
//! för givna temperaturer, sedan värmebalansen med Joulevärmen) tills båda
//! fälten ändras mindre än en relativ tolerans 1e-12. Den konvergerade
//! lösningen är densamma som den helt kopplade Newtonlösningen. Följande är
//! kontrollerat i tests/cases/multiphysics/:
//!
//! - Joulevärmen integreras som volymflödet BF (diagonalt för första
//!   ordningens element);
//! - σ(T) i första ordningens element tas vid temperaturen i noden närmast
//!   integrationspunkten, i andra ordningens vid den interpolerade;
//! - transient: Joulevärmen i ett inkrement är medelvärdet med linjärt
//!   varierande potential, η (a·σa + a·σb + b·σb)/3 med a, b = ∇φ vid
//!   inkrementets början och slut; stationärt: η b·σb.
//!
//! Utdata: NT11, RFL11, HFL (värmen), EPOT och RECUR (nodvärden), ECD
//! (strömtäthet −σ∇φ), EPG (∇φ) och JENER (stationärt: σ∇φ·∇φ; transient:
//! ackumulerad elektrisk energi per volym, utan η).

use super::api::{StepCtx, StepType};
use super::heat::{ensure_initial, heat_controls, heat_loads, HeatSystem, INIT_KEY, TS};
use super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::elements::common::Ip;
use crate::elements::heat::HeatElem;
use crate::elements::IpField;
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{Field, NodeVec, SLOTS};
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::fem::state::StepLoads;
use crate::materials::multiphysics::ElecProps;
use crate::model::{fail, FemError, Result};
use crate::solver::{LinearSolver, SystemMatrix};

pub struct ThermalElectric;

// område coupled2: *COUPLED THERMAL-ELECTRICAL-STRUCTURAL (Q3D4, Q3D8, Q3D20)
#[path = "thermal_electric_structural.rs"]
mod structural;

pub static STEP_TYPES: &[&dyn StepType] = &[&ThermalElectric, &structural::TES];

const PROC: &str = "COUPLED THERMAL-ELECTRICAL";

/// Slot för den elektriska potentialen (frihetsgrad 9).
pub const PS: usize = 6;

/// Nyckel i `SolutionState::vectors` för ackumulerad JENER.
const JENER_KEY: &str = "JENER";

/// Joulevärmen integreras diagonalt för första ordningens element (som BF).
const JOULE_LUMPED: bool = true;

struct EElem {
    el: &'static HeatElem,
    props: usize,
    thickness: f64,
    ips: Vec<Ip>,
    bips: Vec<Ip>,
}

/// Den elektriska ledningen (linjär i φ för givna temperaturer).
pub struct ElecSystem<'p, 'm> {
    p: &'p Prepared<'m>,
    loads: StepLoads,
    elems: Vec<EElem>,
    props: Vec<ElecProps>,
    dm: DofMap,
    mat: Box<dyn SystemMatrix>,
    /// Potential per nod.
    pub phi: Vec<f64>,
    /// Potentialen vid inkrementets början.
    phi_old: Vec<f64>,
    /// Transient: Joulevärmen är medelvärdet över inkrementet med linjärt
    /// varierande potential, (a·σa + a·σb + b·σb)/3 (a, b = ∇φ vid början och
    /// slut), och JENER är den ackumulerade energin per volym.
    pub transient: bool,
    /// Ackumulerad JENER per element och integrationspunkt.
    jener: Vec<Vec<f64>>,
    fint: Vec<f64>,
    fext: Vec<f64>,
}

/// Temperaturen för σ(T) i en punkt: första ordningens element använder
/// temperaturen i noden närmast punkten, andra ordningens den
/// interpolerade.
fn temp_at(ip: &Ip, t: &[f64], first_order: bool) -> f64 {
    if first_order {
        let i = (0..ip.n.len()).max_by(|&a, &b| ip.n[a].total_cmp(&ip.n[b])).unwrap_or(0);
        return t[i];
    }
    ip.n.iter().zip(t).map(|(a, b)| a * b).sum()
}

fn grad(ip: &Ip, v: &[f64]) -> [f64; 3] {
    let mut g = [0.0; 3];
    for (d, &x) in ip.dndx.iter().zip(v) {
        for c in 0..3 {
            g[c] += d[c] * x;
        }
    }
    g
}

fn mv(a: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|r| (0..3).map(|c| a[r][c] * v[c]).sum::<f64>())
}

impl<'p, 'm> ElecSystem<'p, 'm> {
    pub fn new(p: &'p Prepared<'m>, loads: StepLoads, state: &crate::fem::state::SolutionState, solver: &dyn LinearSolver) -> Result<Self> {
        let model = p.model;
        let mut props = Vec::new();
        let mut pidx = std::collections::BTreeMap::new();
        let mut elems = Vec::new();
        for (e, el) in model.elements.iter().enumerate() {
            let Some(he) = crate::elements::thermal_electric::lookup(el.etype.name()) else {
                return fail(format!(
                    "{}: *COUPLED THERMAL-ELECTRICAL kräver termisk-elektriska element (DC2D4E, DC3D8E, ...), inte {}",
                    model.elem_name(e),
                    el.etype.name()
                ));
            };
            let Some(si) = el.section else { return fail(format!("{}: saknar *SOLID SECTION", model.elem_name(e))) };
            let sec = &model.sections[si];
            let Some(mi) = sec.material_index else { return fail(format!("{}: tvärsnittet saknar material", model.elem_name(e))) };
            let pi = match pidx.get(&mi) {
                Some(&i) => i,
                None => {
                    let Some(ep) = ElecProps::from_def(&model.materials[mi])? else {
                        return fail(format!("material {}: *ELECTRICAL CONDUCTIVITY saknas", model.materials[mi].name));
                    };
                    props.push(ep);
                    pidx.insert(mi, props.len() - 1);
                    props.len() - 1
                }
            };
            let thickness = sec.solid_thickness();
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            let ips = he.ips(&x, thickness).map_err(wrap)?;
            let bips = if JOULE_LUMPED { he.cap_ips(&x, thickness).map_err(wrap)? } else { ips.clone() };
            elems.push(EElem { el: he, props: pi, thickness, ips, bips });
        }
        let mut constrained = loads.constrained();
        for (n, &a) in p.active.iter().enumerate() {
            for s in (0..SLOTS).filter(|&s| s != PS && a & (1 << s) != 0) {
                constrained.insert((n, s));
            }
        }
        let (dm, mat) = assembly::dofmap(p, &constrained, solver)?;
        let nn = model.nodes.len();
        let phi: Vec<f64> = state.u.iter().map(|u| u[PS]).collect();
        let jener = match state.vectors.get(JENER_KEY) {
            Some(v) => {
                let mut it = v.iter().map(|x| x[0]);
                elems.iter().map(|h| (0..h.ips.len()).map(|_| it.next().unwrap_or(0.0)).collect()).collect()
            }
            None => elems.iter().map(|h| vec![0.0; h.ips.len()]).collect(),
        };
        Ok(ElecSystem { p, loads, elems, props, dm, mat, phi_old: phi.clone(), phi, transient: false, jener, fint: vec![0.0; nn], fext: vec![0.0; nn] })
    }

    /// σ∇-produkten (a·σa + a·σb + b·σb)/3 eller b·σb.
    fn power(&self, s: &[[f64; 3]; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
        let sb = mv(s, b);
        let bb = b[0] * sb[0] + b[1] * sb[1] + b[2] * sb[2];
        if !self.transient {
            return bb;
        }
        let sa = mv(s, a);
        let aa = a[0] * sa[0] + a[1] * sa[1] + a[2] * sa[2];
        let ab = a[0] * sb[0] + a[1] * sb[1] + a[2] * sb[2];
        (aa + ab + bb) / 3.0
    }

    /// Godkänner inkrementet: JENER ackumuleras med inkrementets längd `dt`.
    pub fn commit(&mut self, temp: &[f64], dt: f64) {
        if self.transient {
            for (e, h) in self.elems.iter().enumerate() {
                let nodes = &self.p.model.elements[e].nodes;
                let te: Vec<f64> = nodes.iter().map(|&n| temp[n]).collect();
                let pa: Vec<f64> = nodes.iter().map(|&n| self.phi_old[n]).collect();
                let pb: Vec<f64> = nodes.iter().map(|&n| self.phi[n]).collect();
                for (q, ip) in h.ips.iter().enumerate() {
                    let (s, _) = self.props[h.props].conductivity(temp_at(ip, &te, h.el.first_order()));
                    self.jener[e][q] += dt * self.power(&s, grad(ip, &pa), grad(ip, &pb));
                }
            }
        }
        self.phi_old.clone_from(&self.phi);
    }

    /// JENER som nodvektor för `SolutionState::vectors` (ärvs till nästa steg).
    fn jener_vec(&self) -> Vec<NodeVec> {
        self.jener.iter().flatten().map(|&v| {
            let mut x = [0.0; SLOTS];
            x[0] = v;
            x
        }).collect()
    }

    fn nodes(&self, e: usize) -> &[usize] {
        &self.p.model.elements[e].nodes
    }

    /// Löser potentialen vid stegtidsandelen `t` för nodtemperaturerna `temp`.
    pub fn solve(&mut self, t: f64, temp: &[f64]) -> Result<()> {
        let model = self.p.model;
        for (n, s, v) in self.loads.bc_at(t) {
            if s == PS {
                self.phi[n] = v;
            }
        }
        let nn = model.nodes.len();
        let mut fext = vec![0.0; nn];
        for (n, s, v) in self.loads.cloads_at(t) {
            if s == PS {
                fext[n] += v;
            }
        }
        for l in self.loads.dloads_at(t) {
            let e = l.elem;
            let h = &self.elems[e];
            let nodes = &model.elements[e].nodes;
            let (_, label) = l.kind.split_once(':').unwrap_or(("", ""));
            if label == "CBF" {
                for ip in &h.bips {
                    for (i, &n) in nodes.iter().enumerate() {
                        fext[n] += ip.n[i] * l.magnitude * ip.w;
                    }
                }
            } else {
                let x: Vec<[f64; 3]> = nodes.iter().map(|&n| model.nodes[n]).collect();
                let lab = label.strip_prefix('C').unwrap_or(label);
                let f = h.el.face_index(lab).map_err(|err| FemError(format!("{}: {err}", model.elem_name(e))))?;
                for fp in h.el.face_points(&x, f, h.thickness, false)? {
                    for (i, &n) in nodes.iter().enumerate() {
                        fext[n] += fp.n[i] * l.magnitude * fp.w;
                    }
                }
            }
        }
        self.mat.zero();
        let mut fint = vec![0.0; nn];
        for (e, h) in self.elems.iter().enumerate() {
            let nodes = &model.elements[e].nodes;
            let te: Vec<f64> = nodes.iter().map(|&n| temp[n]).collect();
            let pe: Vec<f64> = nodes.iter().map(|&n| self.phi[n]).collect();
            let k = nodes.len();
            let mut ke = vec![0.0; k * k];
            for ip in &h.ips {
                let (s, _) = self.props[h.props].conductivity(temp_at(ip, &te, h.el.first_order()));
                for i in 0..k {
                    let si = mv(&s, ip.dndx[i]);
                    for j in 0..k {
                        let d = ip.dndx[j];
                        ke[i * k + j] += ip.w * (si[0] * d[0] + si[1] * d[1] + si[2] * d[2]);
                    }
                }
            }
            for i in 0..k {
                fint[nodes[i]] += (0..k).map(|j| ke[i * k + j] * pe[j]).sum::<f64>();
                let a = self.dm.eq[nodes[i]][PS];
                if a == NONE {
                    continue;
                }
                for j in 0..k {
                    let b = self.dm.eq[nodes[j]][PS];
                    if b != NONE && a >= b && ke[i * k + j] != 0.0 {
                        self.mat.add(a, b, 0.5 * (ke[i * k + j] + ke[j * k + i]));
                    }
                }
            }
        }
        if self.dm.n_eq > 0 {
            if let Err(q) = self.mat.factor() {
                return Err(assembly::mechanism_message(self.p, &self.dm, q));
            }
            let mut r: Vec<f64> = self.dm.owner.iter().map(|&(n, _)| fext[n] - fint[n]).collect();
            self.mat.solve(&mut r);
            for (q, &(n, _)) in self.dm.owner.iter().enumerate() {
                self.phi[n] += r[q];
            }
            // inre ström med den nya potentialen
            fint = vec![0.0; nn];
            for (e, h) in self.elems.iter().enumerate() {
                let nodes = self.nodes(e);
                let te: Vec<f64> = nodes.iter().map(|&n| temp[n]).collect();
                let pe: Vec<f64> = nodes.iter().map(|&n| self.phi[n]).collect();
                for ip in &h.ips {
                    let (s, _) = self.props[h.props].conductivity(temp_at(ip, &te, h.el.first_order()));
                    let j = mv(&s, grad(ip, &pe));
                    for (i, &n) in nodes.iter().enumerate() {
                        let d = ip.dndx[i];
                        fint[n] += ip.w * (d[0] * j[0] + d[1] * j[1] + d[2] * j[2]);
                    }
                }
            }
        }
        self.fint = fint;
        self.fext = fext;
        Ok(())
    }

    /// Joulevärme per nod (∫ N η ∇φ·σ∇φ dV) för temperaturerna `temp`.
    pub fn joule(&self, temp: &[f64]) -> Vec<f64> {
        let mut q = vec![0.0; self.phi.len()];
        for (e, h) in self.elems.iter().enumerate() {
            let nodes = self.nodes(e);
            let te: Vec<f64> = nodes.iter().map(|&n| temp[n]).collect();
            let pe: Vec<f64> = nodes.iter().map(|&n| self.phi[n]).collect();
            let pa: Vec<f64> = nodes.iter().map(|&n| self.phi_old[n]).collect();
            let pr = &self.props[h.props];
            for ip in &h.bips {
                let (s, _) = pr.conductivity(temp_at(ip, &te, h.el.first_order()));
                let p = pr.joule_fraction * self.power(&s, grad(ip, &pa), grad(ip, &pe));
                for (i, &n) in nodes.iter().enumerate() {
                    q[n] += ip.n[i] * p * ip.w;
                }
            }
        }
        q
    }

    /// Reaktionsströmmar i låsta potentialer.
    pub fn reactions(&self) -> Vec<f64> {
        (0..self.phi.len()).map(|n| if self.dm.constrained[n] & (1 << PS) != 0 { self.fint[n] - self.fext[n] } else { 0.0 }).collect()
    }

    /// ECD, EPG och JENER per element.
    pub fn output(&self, temp: &[f64]) -> Vec<Vec<IpField>> {
        let mut out = Vec::new();
        for (e, h) in self.elems.iter().enumerate() {
            let nodes = self.nodes(e);
            let te: Vec<f64> = nodes.iter().map(|&n| temp[n]).collect();
            let pe: Vec<f64> = nodes.iter().map(|&n| self.phi[n]).collect();
            let dim = h.el.dim();
            let (l1, l2): (&[&str], &[&str]) = if dim == 2 { (&["ECD1", "ECD2"], &["EPG1", "EPG2"]) } else { (&["ECD1", "ECD2", "ECD3"], &["EPG1", "EPG2", "EPG3"]) };
            let mut ecd = IpField::new("ECD", l1);
            let mut epg = IpField::new("EPG", l2);
            let mut jen = IpField::new("JENER", &["JENER"]);
            for (q, ip) in h.ips.iter().enumerate() {
                let (s, _) = self.props[h.props].conductivity(temp_at(ip, &te, h.el.first_order()));
                let g = grad(ip, &pe);
                let sg = mv(&s, g);
                ecd.points.push((q + 1, 0, (0..dim).map(|c| -sg[c]).collect()));
                epg.points.push((q + 1, 0, g[..dim].to_vec()));
                let j = if self.transient { self.jener[e][q] } else { g[0] * sg[0] + g[1] * sg[1] + g[2] * sg[2] };
                jen.points.push((q + 1, 0, vec![j]));
            }
            out.push(vec![ecd, epg, jen]);
        }
        out
    }
}

/// Ett inkrement av värmebalansen med fast Δt.
struct Sub<'a, 'p, 'm> {
    sys: &'a mut HeatSystem<'p, 'm>,
    t0: f64,
    t1: f64,
}

impl NonlinearSystem for Sub<'_, '_, '_> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        self.sys.is_linear()
    }
    fn begin_increment(&mut self, _a: f64, _b: f64) -> Result<()> {
        self.sys.begin_increment(self.t0, self.t1)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        self.sys.evaluate(tangent)
    }
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        self.sys.solve(r)
    }
    fn apply(&mut self, c: &[f64]) {
        self.sys.apply(c)
    }
    fn increment(&self) -> Vec<f64> {
        self.sys.increment()
    }
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Delar stegets laster i värmens och elektricitetens.
fn split_loads(l: &StepLoads) -> (StepLoads, StepLoads) {
    let mut th = l.clone();
    let mut el = l.clone();
    th.bc.retain(|&(_, s), _| s != PS);
    el.bc.retain(|&(_, s), _| s == PS);
    th.cload.retain(|&(_, s), _| s != PS);
    el.cload.retain(|&(_, s), _| s == PS);
    th.dload.retain(|(d, _)| !d.kind.starts_with("ECUR:"));
    el.dload.retain(|(d, _)| d.kind.starts_with("ECUR:"));
    th.released.retain(|&(_, s, _)| s != PS);
    el.released.retain(|&(_, s, _)| s == PS);
    th.amp_dload.clear();
    el.amp_dload.clear();
    (th, el)
}

fn temps(u: &[NodeVec]) -> Vec<f64> {
    u.iter().map(|v| v[TS]).collect()
}

fn frame(heat: &HeatSystem, elec: &ElecSystem, value: f64, description: String) -> Result<FrameData> {
    let mut f = heat.frame(value, description)?;
    let t = temps(&heat.u_committed);
    let rec = elec.reactions();
    for n in 0..f.u.len() {
        f.u[n][PS] = elec.phi[n];
        f.rf[n][PS] = rec[n];
        let bit = 1u16 << PS;
        f.active[n] = (f.active[n] & !bit) | (elec.dm.active[n] & bit);
        f.constrained[n] = (f.constrained[n] & !bit) | (elec.dm.constrained[n] & bit);
    }
    for (a, b) in f.elem.iter_mut().zip(elec.output(&t)) {
        a.extend(b);
    }
    f.node_extra.push(("EPOT".into(), vec!["EPOT".into()], elec.phi.iter().map(|&v| vec![v]).collect()));
    f.node_extra.push(("RECUR".into(), vec!["RECUR".into()], rec.iter().map(|&v| vec![v]).collect()));
    Ok(f)
}

/// Fixpunktsiteration för ett inkrement [t0, t1].
fn coupled_increment(heat: &mut HeatSystem, elec: &mut ElecSystem, ctrl: &Controls, t0: f64, t1: f64) -> Result<usize> {
    let sub = Controls { initial: 1.0, fixed: true, ..ctrl.clone() };
    let mut iters = 0;
    let mut t_iter = temps(&heat.u_committed);
    // föreskrivna temperaturer gäller redan i första potentiallösningen
    for (n, s, v) in heat.bc_now(t1) {
        if s == TS {
            t_iter[n] = v;
        }
    }
    for k in 0..200 {
        let phi_old = elec.phi.clone();
        elec.solve(t1, &t_iter)?;
        heat.source = elec.joule(&t_iter);
        let mut one = Sub { sys: &mut *heat, t0, t1 };
        iters += newton::run(&mut one, &sub, &mut |_, _| Ok(()))?.iterations;
        let t_new = temps(&heat.u);
        let dt = t_new.iter().zip(&t_iter).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let tmax = t_new.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
        let dp = elec.phi.iter().zip(&phi_old).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let pmax = elec.phi.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
        t_iter = t_new;
        if k > 0 && dt <= 1e-12 * tmax && dp <= 1e-12 * pmax {
            // reaktioner och Joulevärme med de konvergerade fälten
            elec.solve(t1, &t_iter)?;
            heat.commit()?;
            elec.commit(&t_iter, (t1 - t0) * heat.period);
            return Ok(iters);
        }
    }
    fail(format!("den termisk-elektriska kopplingen konvergerade inte vid stegtid {t1:.4} (fixpunktsiteration)"))
}

impl StepType for ThermalElectric {
    fn name(&self) -> &'static str {
        PROC
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == PROC).then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.step.perturbation {
            return fail("*COUPLED THERMAL-ELECTRICAL kan inte vara ett perturbationssteg");
        }
        if let Some(k) = &ctx.step.procedure {
            if k.has("ELECTRICAL") {
                return k.err("ELECTRICAL (bara den elektriska delen) stöds inte ännu");
            }
        }
        let hc = heat_controls(ctx.step)?;
        if hc.deltmx > 0.0 && !hc.steady {
            return fail("DELTMX i *COUPLED THERMAL-ELECTRICAL stöds inte ännu (ange fasta inkrement)");
        }
        let had_init = ctx.state.vectors.contains_key(INIT_KEY);
        ensure_initial(model, ctx.state)?;
        let mut loads = heat_loads(ctx.step, ctx.loads, hc.steady);
        if !had_init {
            for ((n, s), v) in loads.bc.iter_mut() {
                if *s == TS && !ctx.state.bcs.contains_key(&(*n, *s)) {
                    v.0 = ctx.state.u[*n][TS];
                }
            }
        }
        let (th, el) = split_loads(&loads);
        let period = hc.ctrl.period;
        let mut heat = HeatSystem::new(prep, th, ctx.state, !hc.steady, ctx.solver, period)?;
        let mut elec = ElecSystem::new(prep, el, ctx.state, ctx.solver)?;
        elec.transient = !hc.steady;
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            heat.begin_increment(0.0, 0.0)?;
            heat.u.clone_from(&heat.u_committed);
            let tr = heat.transient;
            heat.transient = false;
            heat.evaluate(false)?;
            heat.transient = tr;
            let mut f0 = frame(&heat, &elec, 0.0, "Increment 0: Step Time = 0.0".into())?;
            for r in f0.rf.iter_mut() {
                *r = [0.0; SLOTS];
            }
            frames.push(f0);
        }
        let ctrl = &hc.ctrl;
        let dt0 = ctrl.initial.clamp(1e-12, 1.0);
        let mut t = 0.0f64;
        let mut inc = 0usize;
        let mut iterations = 0usize;
        while t < 1.0 - 1e-12 {
            if inc >= ctrl.max_increments {
                return fail(format!("steget nådde största antal inkrement ({}) – öka INC= på *STEP", ctrl.max_increments));
            }
            let t1 = (t + dt0).min(1.0);
            let t1 = if 1.0 - t1 < 1e-9 { 1.0 } else { t1 };
            iterations += coupled_increment(&mut heat, &mut elec, ctrl, t, t1)?;
            t = t1;
            inc += 1;
            if ctx.opts.verbose {
                eprintln!("  inkrement {inc}: stegtid {:.6}", t * period);
            }
            if all || t >= 1.0 - 1e-12 {
                frames.push(frame(&heat, &elec, t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = inc;
        ctx.out.iterations = iterations;
        let mut u = heat.u_committed.clone();
        let mut rf = heat.reactions();
        let rec = elec.reactions();
        for n in 0..u.len() {
            u[n][PS] = elec.phi[n];
            rf[n][PS] = rec[n];
        }
        ctx.state.u = u;
        ctx.state.reaction = rf;
        ctx.state.vectors.insert(JENER_KEY.into(), elec.jener_vec());
        ctx.state.time += period;
        Ok(())
    }
}
