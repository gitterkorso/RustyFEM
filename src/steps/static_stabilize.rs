//! *STATIC, STABILIZE: automatisk stabilisering av instabila kvasistatiska
//! problem, med eller utan NLGEOM och med kontakt.
//!
//! Viskösa krafter F_v = c M* v läggs till jämvikten, P − I − F_v = 0, där M*
//! är en diagonal (lumpad) massmatris med densiteten 1, v = Δu/Δt
//! nodhastigheten i inkrementet (Δt = inkrementets tid) och c dämpfaktorn.
//! Tangenten får c M*/Δt. De viskösa krafterna verkar i alla noder, även i
//! föreskrivna frihetsgrader (de ingår då i RF), och skrivs som
//! nodfältet VF = −c M* v. Den dissiperade energin ALLSD (history) integreras
//! med trapetsregeln, ½ (F_v,början + F_v,slut)·Δu, där F_v vid inkrementets
//! början räknas med den aktuella dämpfaktorn och förra inkrementets
//! hastighet.
//!
//! Dämpfaktorn:
//! - `STABILIZE, FACTOR=c`: c gäller från första inkrementet.
//! - `STABILIZE=f` (dissiperad energifraktion, standard 2e-4): efter första
//!   inkrementet c = f T (Δu₁ᵀ ΔI₁)/(Δu₁ᵀ M* Δu₁) (T = stegtiden), där Δu₁ och
//!   ΔI₁ är första inkrementets förskjutnings- och inre kraftökning över alla
//!   frihetsgrader – för linjärt svar Rayleighkvoten Δu₁ᵀKΔu₁/Δu₁ᵀM*Δu₁.
//!   Första inkrementet dämpas själv med samma kvot från
//!   ett odämpat provinkrement; konvergerar det inte (stelkroppsrörelse) med
//!   c₁ = f T · medel(Kᵢᵢ)/medel(M*ᵢᵢ) över de fria frihetsgraderna.
//! - Adaptiv stabilisering (standard för `STABILIZE` utan värde, eller med
//!   `ALLSDTOL=` > 0): dämpfaktorn minskas när ALLSD överstiger ALLSDTOL ·
//!   den inre energin och ökas efter inkrement som halverats (en global
//!   faktor, inte elementvisa faktorer).
//!   `ALLSDTOL=0` ger konstant dämpfaktor.
//! - `CONTINUE=YES`: dämpfaktorn från föregående steg med stabilisering.
//!
//! Inkrementeringen och konvergenskontrollen är desamma som för NLGEOM
//! (`nlgeom::iterate`), även utan NLGEOM.

use super::api::{StepCtx, StepType};
use super::newton::{Controls, NonlinearSystem, Residual};
use super::nlgeom::{self, NlSystem};
use crate::elements::MassKind;
use crate::fem::assembly::{self, DofMap, Prepared};
use crate::fem::dof::{Field, NodeVec, SLOTS};
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::SolutionState;
use crate::fem::system::StaticSystem;
use crate::model::{fail, FemError, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct StaticStabilize;

pub static STEP_TYPES: &[&dyn StepType] = &[&StaticStabilize];

/// Standardvärde för den dissiperade energifraktionen.
pub const DEFAULT_FRACTION: f64 = 2e-4;
/// Standardtolerans för adaptiv stabilisering.
pub const DEFAULT_ALLSDTOL: f64 = 0.05;

/// Stabiliseringens indata.
#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    /// Dissiperad energifraktion (om FACTOR saknas).
    pub fraction: f64,
    /// Direkt angiven dämpfaktor.
    pub factor: Option<f64>,
    /// Tolerans för adaptiv stabilisering (0 = konstant dämpfaktor).
    pub allsdtol: f64,
    /// Dämpfaktorn från föregående steg (CONTINUE=YES).
    pub cont: bool,
}

impl Settings {
    /// Läser STABILIZE, FACTOR, ALLSDTOL och CONTINUE på procedurnyckelordet.
    pub fn parse(kw: &RawKeyword) -> Result<Option<Settings>> {
        if !kw.has("STABILIZE") {
            if kw.has("ALLSDTOL") || kw.has("FACTOR") {
                return kw.err("ALLSDTOL= och FACTOR= kräver STABILIZE");
            }
            return Ok(None);
        }
        let num = |k: &str| -> Result<Option<f64>> {
            match kw.param(k).map(str::trim).filter(|s| !s.is_empty()) {
                Some(v) => match parse_f64(v) {
                    Some(x) if x >= 0.0 => Ok(Some(x)),
                    _ => kw.err(format!("ogiltigt {k}={v}")),
                },
                None => Ok(None),
            }
        };
        let given = num("STABILIZE")?;
        let factor = num("FACTOR")?;
        if factor.is_some_and(|f| f <= 0.0) {
            return kw.err("FACTOR= ska vara positiv");
        }
        let fraction = given.filter(|&f| f > 0.0).unwrap_or(DEFAULT_FRACTION);
        let tol_given = kw.has("ALLSDTOL");
        let tol = num("ALLSDTOL")?;
        // adaptiv som standard med standardfraktionen, annars bara med ALLSDTOL
        let allsdtol = match tol {
            Some(t) => t,
            None if tol_given => DEFAULT_ALLSDTOL,
            None if factor.is_none() && (given.is_none() || given == Some(DEFAULT_FRACTION)) => DEFAULT_ALLSDTOL,
            None => 0.0,
        };
        let cont = kw.param_upper("CONTINUE").is_some_and(|v| v == "YES");
        Ok(Some(Settings { fraction, factor, allsdtol, cont }))
    }
}

/// Senaste dämpfaktorn per modell (för CONTINUE=YES).
static LAST_FACTOR: Mutex<Vec<(usize, f64)>> = Mutex::new(Vec::new());

fn model_key(p: &Prepared) -> usize {
    p.model as *const _ as usize
}

/// Det som stabiliseringen behöver av jämviktssystemet (med eller utan NLGEOM).
pub trait Base: NonlinearSystem {
    fn prep(&self) -> &Prepared<'_>;
    fn dm(&self) -> &DofMap;
    fn u(&self) -> &[NodeVec];
    fn u_committed(&self) -> &[NodeVec];
    fn fint_mut(&mut self) -> &mut Vec<NodeVec>;
    /// Adderar `v` till tangentens diagonal för (nod, slot), villkoren inräknade.
    fn add_diag(&mut self, n: usize, s: usize, v: f64);
    /// Inkrementets tid (t1 − t0)·period.
    fn dt(&self) -> f64;
    fn make_frame(&self, value: f64, description: String) -> Result<FrameData>;
    fn store_state(&self, state: &mut SolutionState);
}

impl Base for StaticSystem<'_, '_> {
    fn prep(&self) -> &Prepared<'_> {
        self.p
    }
    fn dm(&self) -> &DofMap {
        &self.dm
    }
    fn u(&self) -> &[NodeVec] {
        &self.u
    }
    fn u_committed(&self) -> &[NodeVec] {
        &self.u_committed
    }
    fn fint_mut(&mut self) -> &mut Vec<NodeVec> {
        &mut self.fint
    }
    fn add_diag(&mut self, n: usize, s: usize, v: f64) {
        if self.mat.is_symmetric() {
            self.dm.add_entry(&mut *self.mat, n, s, n, s, v);
        } else {
            self.dm.add_entry_unsym(&mut *self.mat, n, s, n, s, v);
        }
    }
    fn dt(&self) -> f64 {
        (self.t1 - self.t0) * self.period
    }
    fn make_frame(&self, value: f64, description: String) -> Result<FrameData> {
        assembly::frame(self.p, &self.dm, &self.u_committed, &self.fint, &self.fext, &self.state_old, self.nlgeom, self.time0 + value, value, description)
    }
    fn store_state(&self, state: &mut SolutionState) {
        self.store(state)
    }
}

impl Base for NlSystem<'_, '_> {
    fn prep(&self) -> &Prepared<'_> {
        self.p
    }
    fn dm(&self) -> &DofMap {
        &self.dm
    }
    fn u(&self) -> &[NodeVec] {
        &self.u
    }
    fn u_committed(&self) -> &[NodeVec] {
        &self.u_committed
    }
    fn fint_mut(&mut self) -> &mut Vec<NodeVec> {
        &mut self.fint
    }
    fn add_diag(&mut self, n: usize, s: usize, v: f64) {
        if self.mat.is_symmetric() {
            self.dm.add_entry(&mut *self.mat, n, s, n, s, v);
        } else {
            self.dm.add_entry_unsym(&mut *self.mat, n, s, n, s, v);
        }
    }
    fn dt(&self) -> f64 {
        (self.t1 - self.t0) * self.period
    }
    fn make_frame(&self, value: f64, description: String) -> Result<FrameData> {
        self.frame(value, description)
    }
    fn store_state(&self, state: &mut SolutionState) {
        self.store(state)
    }
}

/// Diagonal massa M* med densiteten 1 per nod och slot (lumpad).
pub fn unit_mass(p: &Prepared) -> Result<Vec<NodeVec>> {
    let m = p.model;
    let mut out = vec![[0.0; SLOTS]; m.nodes.len()];
    for (e, el) in m.elements.iter().enumerate() {
        if p.activity.is_removed(e) || crate::elements::rigid::is_rigid(el.etype) {
            continue;
        }
        let mut ctx = p.ctx(e, false, 0.0, 0.0);
        if ctx.section.is_none() {
            continue;
        }
        ctx.density = 1.0;
        let me = match el.etype.mass(&ctx, MassKind::Lumped) {
            Ok(me) => me,
            // element utan massa (fjädrar, kopplingar): ingen dämpning
            Err(_) => continue,
        };
        let dl = &p.edofs[e];
        let n = dl.len();
        if me.len() != n * n {
            continue;
        }
        for (a, &(nd, s)) in dl.iter().enumerate() {
            let row: f64 = (0..n).map(|b| me[a * n + b]).sum();
            out[nd][s] += row;
        }
    }
    Ok(out)
}

/// Jämviktssystemet med viskösa krafter c M* Δu/Δt.
pub struct Stabilized<S: Base> {
    pub base: S,
    /// Dämpfaktorn som används nu.
    pub c: f64,
    pub mass: Vec<NodeVec>,
    /// Viskösa krafter c M* v vid senaste utvärderingen (före kondensering).
    fv: Vec<NodeVec>,
    /// M* v från förra godkända inkrementet (för trapetsregeln).
    mv_prev: Vec<NodeVec>,
    /// Dissiperad energi och inre arbete (för adaptiv stabilisering).
    pub allsd: f64,
    pub allie: f64,
    fint_prev: Vec<NodeVec>,
    /// Energifraktion f·T för c efter första inkrementet (None = FACTOR).
    energy: Option<(f64, f64)>,
    pub incs: usize,
    adaptive: f64,
    last_t0: Option<f64>,
    c_initial: f64,
}

impl<S: Base> Stabilized<S> {
    fn viscous(&mut self) {
        let dt = self.base.dt();
        let (u, uc) = (self.base.u(), self.base.u_committed());
        let dm = self.base.dm();
        for n in 0..u.len() {
            for s in 0..SLOTS {
                self.fv[n][s] = if dt > 0.0 && dm.active[n] & (1 << s) != 0 && self.mass[n][s] > 0.0 { self.c * self.mass[n][s] * (u[n][s] - uc[n][s]) / dt } else { 0.0 };
            }
        }
    }

    /// (Δuᵀ ΔI, Δuᵀ M* Δu) för inkrementet som just itererats.
    fn energy_quotient(&mut self) -> (f64, f64) {
        let fe = self.fint_elastic();
        let (u, uc) = (self.base.u(), self.base.u_committed());
        let act = &self.base.dm().active;
        let (mut num, mut den) = (0.0, 0.0);
        for n in 0..u.len() {
            for s in 0..SLOTS {
                if act[n] & (1 << s) != 0 {
                    let du = u[n][s] - uc[n][s];
                    num += du * (fe[n][s] - self.fint_prev[n][s]);
                    den += self.mass[n][s] * du * du;
                }
            }
        }
        (num, den)
    }

    /// Inre krafter utan de viskösa (senaste utvärderingen).
    fn fint_elastic(&mut self) -> Vec<NodeVec> {
        let mut fv = self.fv.clone();
        self.base.dm().mpc.condense(&mut fv);
        let f = self.base.fint_mut();
        f.iter().zip(&fv).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect()
    }
}

impl<S: Base> NonlinearSystem for Stabilized<S> {
    fn n_eq(&self) -> usize {
        self.base.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.base.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        false
    }
    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        // adaptiv: ett nytt försök från samma tid = halvering → mer dämpning
        if self.adaptive > 0.0 && self.last_t0 == Some(t0) && self.incs > 0 {
            self.c = (self.c * 2.0).min(self.c_initial * 1e4);
        }
        self.last_t0 = Some(t0);
        self.base.begin_increment(t0, t1)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        let mut r = self.base.evaluate(tangent)?;
        self.viscous();
        let dt = self.base.dt();
        let mut fv = self.fv.clone();
        self.base.dm().mpc.condense(&mut fv);
        {
            let f = self.base.fint_mut();
            for (a, b) in f.iter_mut().zip(&fv) {
                for s in 0..SLOTS {
                    a[s] += b[s];
                }
            }
        }
        for (q, &(n, s)) in self.base.dm().owner.iter().enumerate() {
            r.r[q] -= fv[n][s];
        }
        if tangent && dt > 0.0 && self.c > 0.0 {
            for n in 0..self.mass.len() {
                for s in 0..SLOTS {
                    if self.mass[n][s] > 0.0 {
                        let v = self.c * self.mass[n][s] / dt;
                        self.base.add_diag(n, s, v);
                    }
                }
            }
        }
        Ok(r)
    }
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        self.base.solve(r)
    }
    fn apply(&mut self, c: &[f64]) {
        self.base.apply(c)
    }
    fn increment(&self) -> Vec<f64> {
        self.base.increment()
    }
    fn commit(&mut self) -> Result<()> {
        let dt = self.base.dt();
        let fe = self.fint_elastic();
        let (u, uc) = (self.base.u().to_vec(), self.base.u_committed().to_vec());
        let dm_active = self.base.dm().active.clone();
        // ALLSD med trapetsregeln och inre arbete
        let mut dsd = 0.0;
        let mut dwi = 0.0;
        let mut num = 0.0;
        let mut den = 0.0;
        let mut mv = vec![[0.0; SLOTS]; u.len()];
        for n in 0..u.len() {
            for s in 0..SLOTS {
                if dm_active[n] & (1 << s) == 0 {
                    continue;
                }
                let du = u[n][s] - uc[n][s];
                if dt > 0.0 {
                    mv[n][s] = self.mass[n][s] * du / dt;
                }
                dsd += 0.5 * self.c * (self.mv_prev[n][s] + mv[n][s]) * du;
                dwi += 0.5 * (self.fint_prev[n][s] + fe[n][s]) * du;
                num += du * (fe[n][s] - self.fint_prev[n][s]);
                den += self.mass[n][s] * du * du;
            }
        }
        self.allsd += dsd;
        self.allie += dwi;
        self.base.commit()?;
        self.incs += 1;
        if self.incs == 1 {
            if let Some((ft, _)) = self.energy {
                if den > 0.0 && num > 0.0 {
                    self.c = ft * num / den;
                }
                self.c_initial = self.c;
            }
        }
        if self.adaptive > 0.0 && self.allie > 0.0 && self.allsd > self.adaptive * self.allie {
            // för mycket dissipation: minska dämpningen
            let f = (self.adaptive * self.allie / self.allsd).clamp(0.1, 1.0);
            self.c *= f;
        }
        self.mv_prev = mv;
        self.fint_prev = fe;
        self.last_t0 = None;
        Ok(())
    }
}

/// c₁ = f T · medel(Kᵢᵢ)/medel(M*ᵢᵢ) över de fria frihetsgraderna i läget vid stegets början.
fn initial_factor<S: Base>(sys: &S, mass: &[NodeVec], ft: f64, nlgeom: bool, state: &SolutionState) -> Result<f64> {
    let p = sys.prep();
    let m = p.model;
    let dm = sys.dm();
    let mut kd = vec![[0.0; SLOTS]; m.nodes.len()];
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, nlgeom, state.time, 0.0);
        if ctx.material.is_none() && ctx.section.and_then(|s| s.beam()).is_none() {
            continue;
        }
        let ue = p.gather(e, &state.u);
        let zero = vec![0.0; ue.len()];
        let mut sn = Vec::new();
        let r = match crate::steps::model_change::internal_force(p, e, &ctx, &ue, &zero, &state.elem_state[e], &mut sn, true) {
            Some(r) => r,
            None => el.etype.internal_force(&ctx, &ue, &zero, &state.elem_state[e], &mut sn, true),
        };
        let Ok(r) = r else { continue };
        let Some(k) = r.k else { continue };
        let dl = &p.edofs[e];
        let n = dl.len();
        for (a, &(nd, s)) in dl.iter().enumerate() {
            kd[nd][s] += k[a * n + a];
        }
    }
    let (mut sk, mut sm, mut cnt) = (0.0, 0.0, 0usize);
    for &(n, s) in &dm.owner {
        if mass[n][s] > 0.0 {
            sk += kd[n][s].abs();
            sm += mass[n][s];
            cnt += 1;
        }
    }
    if cnt == 0 || sm <= 0.0 {
        return Ok(0.0);
    }
    Ok(ft * sk / sm)
}

/// Inkrementering som `nlgeom::drive` men för vilket system som helst.
fn drive<S: NonlinearSystem>(sys: &mut S, ctrl: &Controls, on_increment: &mut dyn FnMut(&mut S, f64, usize) -> Result<()>) -> Result<(usize, usize)> {
    let mut t = 0.0f64;
    let mut dt = ctrl.initial.min(1.0);
    let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
    let mut n_flux = 0usize;
    let mut quick_prev = false;
    let mut incs = 0;
    let mut iters = 0;
    while t < 1.0 - 1e-12 {
        if incs >= ctrl.max_increments {
            return fail(format!("steget nådde största antal inkrement ({}) vid stegtid {:.4} – öka INC= på *STEP", ctrl.max_increments, t * ctrl.period));
        }
        if !ctrl.fixed && t + dt > 1.0 - 1e-12 {
            dt = 1.0 - t;
        }
        let mut cutbacks = 0;
        let out = loop {
            sys.begin_increment(t, t + dt)?;
            let mut solve = |s: &mut S, r: &[f64]| s.solve(r);
            match nlgeom::iterate(sys, ctrl, &flux_sum, n_flux, &mut solve)? {
                Some(o) => break o,
                None => {
                    if ctrl.fixed {
                        return fail(format!("inkrementet vid stegtid {:.4} konvergerade inte (DIRECT, fasta inkrement)", (t + dt) * ctrl.period));
                    }
                    cutbacks += 1;
                    dt *= ctrl.cutback;
                    if cutbacks > ctrl.max_cutbacks || dt < ctrl.min {
                        return fail(format!(
                            "ingen konvergens vid stegtid {:.4}: inkrementet blev mindre än minimum ({:.3e}) eller för många halveringar",
                            t * ctrl.period,
                            ctrl.min * ctrl.period
                        ));
                    }
                }
            }
        };
        for (f, q) in &out.flux {
            *flux_sum.entry(*f).or_default() += q;
        }
        n_flux += 1;
        sys.commit()?;
        t += dt;
        incs += 1;
        iters += out.iterations;
        crate::progress::emit(crate::progress::Event::Increment { fraction: t, iterations: out.iterations })?;
        on_increment(sys, t, incs)?;
        if !ctrl.fixed {
            let quick = out.iterations <= ctrl.ig && cutbacks == 0;
            if quick && quick_prev {
                dt *= ctrl.growth;
            }
            quick_prev = quick;
            dt = dt.min(ctrl.max);
        }
    }
    Ok((incs, iters))
}

/// Nodfältet VF = −c M* v.
fn vf_field<S: Base>(sys: &Stabilized<S>) -> (String, Vec<String>, Vec<Vec<f64>>) {
    let p = sys.base.prep();
    let dim = p.model.dim().clamp(1, 3);
    let labels: Vec<String> = (1..=dim).map(|i| format!("VF{i}")).collect();
    let vals = sys.fv.iter().map(|f| (0..dim).map(|s| -f[s]).collect()).collect();
    ("VF".into(), labels, vals)
}

fn run_with<S: Base>(ctx: &mut StepCtx, base: S, settings: &Settings, ctrl: &Controls) -> Result<()> {
    let prep = ctx.prep;
    let mass = unit_mass(prep)?;
    let ft = settings.fraction * ctrl.period;
    let nn = mass.len();
    let mut c = match settings.factor {
        Some(f) => f,
        None => initial_factor(&base, &mass, ft, ctx.nlgeom, ctx.state)?,
    };
    let mut energy = settings.factor.is_none().then_some((ft, 0.0));
    if settings.cont {
        let key = model_key(prep);
        if let Some(&(_, last)) = LAST_FACTOR.lock().unwrap().iter().find(|(k, _)| *k == key) {
            c = last;
            energy = None;
        }
    }
    let mut sys = Stabilized {
        base,
        c,
        mass,
        fv: vec![[0.0; SLOTS]; nn],
        mv_prev: vec![[0.0; SLOTS]; nn],
        allsd: 0.0,
        allie: 0.0,
        fint_prev: vec![[0.0; SLOTS]; nn],
        energy,
        incs: 0,
        adaptive: settings.allsdtol,
        last_t0: None,
        c_initial: c,
    };
    // inre krafter vid stegets början (för ΔI och arbetet)
    sys.base.begin_increment(0.0, 0.0)?;
    sys.base.evaluate(false)?;
    sys.fint_prev = sys.base.fint_mut().clone();
    // energifraktion: dämpfaktorn i första inkrementet ur ett odämpat
    // provinkrement, c₁ = f T Δuᵀ ΔI/Δuᵀ M* Δu (dokumenterade c₁ om det inte
    // konvergerar, t.ex. vid stelkroppsrörelse)
    if let Some((ft, _)) = sys.energy {
        let c_doc = sys.c;
        sys.c = 0.0;
        sys.base.begin_increment(0.0, ctrl.initial.min(1.0))?;
        let mut solve = |s: &mut Stabilized<S>, r: &[f64]| s.solve(r);
        let trial = nlgeom::iterate(&mut sys, ctrl, &BTreeMap::new(), 0, &mut solve);
        sys.c = c_doc;
        if let Ok(Some(_)) = trial {
            let (num, den) = sys.energy_quotient();
            if num > 0.0 && den > 0.0 {
                sys.c = ft * num / den;
            }
        }
        sys.c_initial = sys.c;
        sys.base.begin_increment(0.0, 0.0)?;
    }
    let c = sys.c;
    let all = ctx.opts.all_frames;
    let verbose = ctx.opts.verbose;
    let period = ctrl.period;
    let time0 = ctx.state.time;
    let mut frames = Vec::new();
    if all {
        let mut fr = sys.base.make_frame(0.0, "Increment 0: Step Time = 0.0".into())?;
        let mut z = vf_field(&sys);
        z.2.iter_mut().for_each(|v| v.iter_mut().for_each(|x| *x = 0.0));
        fr.node_extra.push(z);
        frames.push(fr);
    }
    let mut hist: Vec<(f64, f64)> = vec![(0.0, 0.0)];
    if verbose && settings.factor.is_none() {
        eprintln!("  stabilisering: dämpfaktor i första inkrementet {c:.5e}");
    }
    let (incs, iters) = drive(&mut sys, ctrl, &mut |s, t, inc| {
        if verbose {
            eprintln!("  inkrement {inc}: stegtid {:.6}, dämpfaktor {:.5e}, ALLSD {:.5e}", t * period, s.c, s.allsd);
        }
        hist.push((t * period, s.allsd));
        if all || t >= 1.0 - 1e-12 {
            let mut fr = s.base.make_frame(t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?;
            fr.node_extra.push(vf_field(s));
            frames.push(fr);
        }
        Ok(())
    })?;
    if verbose {
        eprintln!("  stabilisering: dämpfaktor {:.5e}, ALLSD {:.5e}", sys.c, sys.allsd);
    }
    {
        let key = model_key(prep);
        let mut lf = LAST_FACTOR.lock().unwrap();
        lf.retain(|(k, _)| *k != key);
        lf.push((key, sys.c));
    }
    let _ = time0;
    ctx.out.frames.extend(frames);
    ctx.out.history.push(("Assembly ASSEMBLY".into(), "ALLSD".into(), hist));
    ctx.out.increments = incs;
    ctx.out.iterations = iters;
    sys.base.store_state(ctx.state);
    ctx.state.time += period;
    Ok(())
}

/// Stegets *CONTROLS (från steget där de anges tills RESET) som i `static_linear.rs`.
fn controls(ctx: &StepCtx) -> Result<Controls> {
    let mut ctrl = Controls::from_procedure(ctx.step.procedure.as_ref(), ctx.step.max_increments)?;
    for s in ctx.prep.model.steps.iter().take(ctx.index + 1) {
        nlgeom::apply_controls(&mut ctrl, &s.extra)?;
    }
    Ok(ctrl)
}

impl StepType for StaticStabilize {
    fn name(&self) -> &'static str {
        "STATIC, STABILIZE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (step.procedure_name() == "STATIC" && p.has("STABILIZE") && !p.has("RIKS") && !step.perturbation).then_some(30)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().ok_or_else(|| FemError("*STATIC saknas".into()))?;
        let Some(settings) = Settings::parse(kw)? else {
            return fail("STABILIZE saknas");
        };
        let ctrl = controls(ctx)?;
        if ctx.nlgeom {
            let sys = NlSystem::new(ctx.prep, ctx.loads, ctx.state, ctx.solver, ctrl.period)?;
            run_with(ctx, sys, &settings, &ctrl)
        } else {
            let sys = StaticSystem::new(ctx.prep, ctx.loads, ctx.state, false, ctx.solver, ctrl.period)?;
            run_with(ctx, sys, &settings, &ctrl)
        }
    }
}
