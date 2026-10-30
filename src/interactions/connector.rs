//! Kopplingselement (CONN2D2, CONN3D2) med *CONNECTOR SECTION,
//! *CONNECTOR BEHAVIOR och stegnyckelorden *CONNECTOR MOTION och
//! *CONNECTOR LOAD.
//!
//! Varje element har sex komponenter (1–3 translation, 4–6 rotation) som
//! kopplingstypen gör tillgängliga, låsta eller saknade
//! (`connector_kin.rs`, alla typer i anslutningsbiblioteket utom
//! ACCELEROMETER, FLOW-CONVERTER, RETRACTOR och SLIPRING). Komponentens
//! värde g = c·u är linjärt i nodfrihetsgraderna (små förskjutningar och
//! rotationer) och kraften F(g) från beteendena (`connector_beh.rs`) ger
//! den inre kraften f = Σ F cᵀ och tangenten K = Σ ∂F/∂g c cᵀ.
//!
//! Låsta komponenter elimineras exakt som linjära villkor (`fem/mpc.rs`);
//! villkorskraften (som ingår i CTF/CTM) räknas ur den
//! okondenserade residualen i slavfrihetsgraderna (`observe_residual`,
//! anropas av de statiska stegen). Stopp, lås och föreskriven rörelse
//! (*CONNECTOR MOTION) upprätthålls med en styv straffkoefficient
//! k = `PENALTY` · (största styvheten vid noderna); felet blir ~1/`PENALTY`
//! relativt. *CONNECTOR ELASTICITY, RIGID är en fjäder med styvheten 1e6
//! (implicit analys).
//!
//! *CONNECTOR LOAD ger lika och motriktade krafter längs komponenten
//! (f = −L cᵀ i den inre kraften; CTF = inre kraft − L). Rörelser och laster
//! rampas linjärt över steget från föregående värde (standard, steg i
//! *DYNAMIC) eller följer AMPLITUDE=; OP=NEW tar bort tidigare – borttagna
//! laster och rörelsens reaktion avvecklas till noll under steget. Stegtiden
//! räknas fram från den totala tiden och stegens längd.
//!
//! *CONNECTOR DAMPING verkar i *DYNAMIC med hastigheten Δg/Δt över
//! inkrementet (HHT-viktad som övriga inre krafter).
//!
//! Utdata (helt element, ip 0): CTF/CTM (total kraft),
//! CU/CUR (relativ rörelse), CEF/CEM (elastisk, efter plasticitet och skada),
//! CUE/CURE, CUP/CURP, CSF/CSM (friktion), CRF/CRM (villkor, stopp, lås,
//! rörelse), CVF/CVM (dämpning), CP/CPR (läge) och CDMG/CDMGR.

#[path = "connector_beh.rs"]
pub mod beh;
#[path = "connector_kin.rs"]
pub mod kin;
#[path = "connector_nl.rs"]
pub mod nl;

use super::geom;
use super::{Interaction, InteractionCtx, InteractionFactory, TangentSink};
use crate::elements::common::{dot, norm, scale, sub};
use crate::elements::{ElemCtx, IpField};
use crate::fem::dof::{self, Dof, NodeVec, U1, UR1};
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::{fail, Result};
use beh::{Behavior, CompState, Elastic};
use kin::Status;
use std::collections::BTreeMap;
use std::sync::Mutex;

pub static FACTORIES: &[InteractionFactory] = &[InteractionFactory { keywords: &["CONNECTOR SECTION"], build: section }];

/// Straffkoefficient för villkor relativt nodernas styvhet.
pub const PENALTY: f64 = 1e8;

/// Föreskrivet värde i ett steg: från, till, amplitud.
#[derive(Debug, Clone)]
struct Ramp {
    from: f64,
    to: f64,
    amp: Option<String>,
    /// Rörelsen togs bort (OP=NEW): reaktionen avvecklas till noll under steget.
    release: bool,
}

struct Elem {
    index: usize,
    /// Noden vars frihetsgrader elimineras i villkoren (b, eller a om b är jordad).
    bnode: usize,
    comps: [kin::CompDef; 6],
    terms: [Vec<(usize, Dof, f64)>; 6],
    k_pen: [f64; 6],
    length: f64,
    /// Per modellsteg: föreskriven rörelse respektive last per komponent.
    motion: Vec<[Option<Ramp>; 6]>,
    load: Vec<[Option<Ramp>; 6]>,
    /// Geometrin för kinematiken med stora rotationer (NLGEOM, `connector_nl.rs`).
    geo: nl::Geo,
}

/// Utdata per element.
#[derive(Debug, Clone, Copy, Default)]
struct Out {
    g: [f64; 6],
    f: [f64; 6],
    fe: [f64; 6],
    fs: [f64; 6],
    fr: [f64; 6],
    fv: [f64; 6],
}

#[derive(Default)]
struct State {
    committed: Vec<[CompState; 6]>,
    trial: Vec<[CompState; 6]>,
    g_committed: Vec<[f64; 6]>,
    g_trial: Vec<[f64; 6]>,
    out_trial: Vec<Out>,
    out_committed: Vec<Out>,
    /// Villkorskrafter i låsta komponenter.
    lambda_trial: Vec<[f64; 6]>,
    lambda_committed: Vec<[f64; 6]>,
}

impl Elem {
    /// Komponenter som är villkor (låsta av kopplingstypen).
    fn is_constraint(&self, beh: &Behavior, c: usize) -> bool {
        let _ = beh;
        self.comps[c].status == Status::Constrained
    }
}

/// Stegens start och längd (för stegtiden) och standardamplitud.
struct Timeline {
    steps: Vec<(f64, f64, bool)>,
}

impl Timeline {
    fn new(m: &FeModel) -> Result<Timeline> {
        let mut t = 0.0;
        let mut steps = Vec::new();
        for s in &m.steps {
            let name = s.procedure_name().to_string();
            let timed = matches!(name.as_str(), "STATIC" | "DYNAMIC" | "VISCO" | "SOILS" | "COUPLED TEMPERATURE-DISPLACEMENT" | "HEAT TRANSFER" | "GEOSTATIC");
            let period = if timed && !s.perturbation { crate::steps::newton::Controls::from_procedure(s.procedure.as_ref(), s.max_increments)?.period } else { 0.0 };
            let quasi = s.procedure.as_ref().and_then(|p| p.param_upper("APPLICATION")).as_deref() == Some("QUASI-STATIC");
            let ramp = match s.header.param_upper("AMPLITUDE") {
                Some(a) => a == "RAMP",
                None => name != "DYNAMIC" || quasi,
            };
            steps.push((t, period, ramp));
            t += period;
        }
        Ok(Timeline { steps })
    }

    /// (stegindex, stegtid, andel) vid totaltiden t.
    fn at(&self, t: f64) -> Option<(usize, f64, f64)> {
        let mut last = None;
        for (i, &(s, p, _)) in self.steps.iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            last = Some(i);
            if t <= s + p * (1.0 + 1e-12) + 1e-300 {
                let st = (t - s).clamp(0.0, p);
                return Some((i, st, st / p));
            }
        }
        last.map(|i| (i, self.steps[i].1, 1.0))
    }
}

pub struct Connectors {
    name: String,
    dim: usize,
    elems: Vec<Elem>,
    beh: Behavior,
    active: Vec<(usize, Dof)>,
    timeline: Timeline,
    amps: BTreeMap<String, crate::steps::dynamic::amplitude::Amplitude>,
    linear: bool,
    /// Modellstegen som är dynamiksteg (dämpningen verkar bara där).
    dynamic: Vec<bool>,
    /// Modellen har NLGEOM- eller explicita steg: låsta komponenter
    /// upprätthålls inte som linjära villkor (som inte kan följa stora
    /// rotationer) utan med straffkoefficienten `PENALTY` i implicita steg och
    /// som kinematiska villkor (`kinematic_constraints`) i explicita steg.
    nl_mode: bool,
    /// Modellstegen som är explicita (*DYNAMIC, EXPLICIT).
    explicit: Vec<bool>,
    state: Mutex<State>,
}

impl Connectors {
    /// Komponenter och rader med stora rotationer (`connector_nl.rs`);
    /// komponenter som saknas ger 0.
    fn eval_nl(&self, e: &Elem, u: &[NodeVec], near: &[f64; 6]) -> nl::Eval {
        let mut ev = e.geo.eval(u, near);
        for c in 0..6 {
            if e.comps[c].status == Status::None {
                ev.g[c] = 0.0;
                ev.rows[c].clear();
            }
        }
        ev
    }

    fn value(&self, r: &Ramp, step: usize, st: f64, frac: f64, total: f64) -> f64 {
        if let Some(a) = r.amp.as_ref().and_then(|n| self.amps.get(n)) {
            return r.to * a.value(st, total);
        }
        if self.timeline.steps[step].2 {
            r.from + (r.to - r.from) * frac
        } else {
            r.to
        }
    }
}

impl Interaction for Connectors {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        self.active.clone()
    }
    fn connectivity(&self) -> Vec<Vec<usize>> {
        self.elems
            .iter()
            .map(|e| {
                let mut v: Vec<usize> = e.terms.iter().flatten().map(|t| t.0).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect()
    }
    fn is_linear(&self) -> bool {
        self.linear
    }

    fn linear_constraints(&self) -> Result<Vec<super::LinearConstraint>> {
        let mut out = Vec::new();
        if self.nl_mode {
            return Ok(out);
        }
        for e in &self.elems {
            for c in 0..6 {
                if !e.is_constraint(&self.beh, c) || e.terms[c].is_empty() {
                    continue;
                }
                // nod b:s frihetsgrad med störst koefficient elimineras i första hand
                let bnode = e.bnode;
                let mut t = e.terms[c].clone();
                t.sort_by(|p, q| (q.0 == bnode).cmp(&(p.0 == bnode)).then((q.1 <= 3).cmp(&(p.1 <= 3))).then(q.2.abs().total_cmp(&p.2.abs())));
                out.push(super::LinearConstraint { terms: t, rhs: 0.0, force_slave: false });
            }
        }
        Ok(out)
    }

    fn observe_residual(&self, fint: &[NodeVec], fext: &[NodeVec], slaves: &[crate::fem::mpc::Slave]) {
        let mut st = self.state.lock().unwrap();
        let slave_set: std::collections::HashSet<(usize, usize)> = slaves.iter().map(|s| (s.node, s.slot)).collect();
        for (ei, e) in self.elems.iter().enumerate() {
            let cons: Vec<usize> = (0..6).filter(|&c| e.is_constraint(&self.beh, c)).collect();
            if cons.is_empty() {
                continue;
            }
            // slavfrihetsgrader i villkorsraderna och A[s][k] = koefficienten
            let mut dofs: Vec<(usize, usize)> = Vec::new();
            for &c in &cons {
                for &(n, d, _) in &e.terms[c] {
                    let s = dof::slot_of(d);
                    if slave_set.contains(&(n, s)) && !dofs.contains(&(n, s)) {
                        dofs.push((n, s));
                    }
                }
            }
            let nk = cons.len();
            // normalekvationer AᵀA λ = Aᵀ b, b = −(fint − fext) i slavarna
            let mut ata = vec![0.0; nk * nk];
            let mut atb = vec![0.0; nk];
            for &(n, s) in &dofs {
                let row: Vec<f64> = cons.iter().map(|&c| e.terms[c].iter().filter(|t| t.0 == n && dof::slot_of(t.1) == s).map(|t| t.2).sum()).collect();
                let b = -(fint[n][s] - fext[n][s]);
                for i in 0..nk {
                    atb[i] += row[i] * b;
                    for j in 0..nk {
                        ata[i * nk + j] += row[i] * row[j];
                    }
                }
            }
            let lam = solve_small(&mut ata, &mut atb, nk);
            let mut l6 = [0.0; 6];
            for (i, &c) in cons.iter().enumerate() {
                l6[c] = lam[i];
            }
            st.lambda_trial[ei] = l6;
        }
    }

    fn contribute(&self, ctx: &InteractionCtx, u: &[NodeVec], fint: &mut [NodeVec], mut k: Option<&mut dyn TangentSink>) -> Result<()> {
        let when = self.timeline.at(ctx.time);
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        let explicit_step = when.is_some_and(|(si, _, _)| self.explicit[si]);
        for (ei, e) in self.elems.iter().enumerate() {
            // NLGEOM: komponenterna och raderna i aktuellt läge (connector_nl.rs)
            let ev = if ctx.nlgeom { Some(self.eval_nl(e, u, &st.g_committed[ei])) } else { None };
            let terms: &[Vec<(usize, Dof, f64)>; 6] = match &ev {
                Some(v) => &v.rows,
                None => &e.terms,
            };
            let mut g = [0.0; 6];
            for c in 0..6 {
                g[c] = match &ev {
                    Some(v) => v.g[c],
                    None => e.terms[c].iter().map(|&(n, d, v)| v * u[n][dof::slot_of(d)]).sum(),
                };
            }
            let old = st.committed[ei];
            let g_old = st.g_committed[ei];
            let mut new = old;
            let mut f = [0.0; 6];
            let mut dmat = [[0.0; 6]; 6];
            let mut out = Out { g, ..Default::default() };
            for c in 0..6 {
                let cd = &e.comps[c];
                if cd.status == Status::None {
                    continue;
                }
                if cd.status == Status::Constrained {
                    // villkor (elimineras exakt i fem/mpc.rs); kraften räknas
                    // ur residualen (`observe_residual`) och läggs till i utdata.
                    // NLGEOM-modeller: styv straffkoefficient i implicita steg,
                    // kinematiska villkor i explicita steg
                    if self.nl_mode && !explicit_step {
                        let fp = e.k_pen[c] * g[c];
                        f[c] += fp;
                        dmat[c][c] += e.k_pen[c];
                        out.fr[c] += fp;
                    }
                    continue;
                }
                let b = &self.beh.comps[c];
                // elasticitet, plasticitet och skada
                let (mut fe, mut ke) = match (&b.plastic, &b.elastic) {
                    (Some(p), Elastic::Linear(kl)) => {
                        let (fp, kp, s) = beh::plastic(*kl, p, g[c], &old[c]);
                        new[c].up = s.up;
                        new[c].ub = s.ub;
                        new[c].alpha = s.alpha;
                        (fp, kp)
                    }
                    (_, el) => el.eval(g[c], st.g_trial[ei][c]),
                };
                if let Some(dm) = &b.damage {
                    let kl = match b.elastic {
                        Elastic::Linear(kl) => kl,
                        _ => ke.max(1e-300),
                    };
                    let mut s = old[c];
                    s.ub = new[c].ub;
                    let (fac, dfac) = beh::damage(dm, kl, g[c], fe, &mut s);
                    new[c].d = s.d;
                    new[c].u0 = s.u0;
                    new[c].dsign = s.dsign;
                    ke = fac * ke + dfac * fe;
                    fe *= fac;
                }
                f[c] += fe;
                dmat[c][c] += ke;
                out.fe[c] += fe;
                // friktion (klistring med elastisk glidning, glidning vid μN)
                if let Some(fr) = &b.friction {
                    let lim = fr.mu * fr.f0.abs();
                    let ks = fr.stick.unwrap_or_else(|| lim / (1e-4 * if e.length > 0.0 { e.length } else { 1.0 }));
                    let tr = old[c].ff + ks * (g[c] - g_old[c]);
                    let (ff, kf) = if tr.abs() <= lim { (tr, ks) } else { (lim * tr.signum(), 0.0) };
                    new[c].ff = ff;
                    new[c].slip = old[c].slip + (g[c] - g_old[c]) - (ff - old[c].ff) / ks.max(1e-300);
                    f[c] += ff;
                    dmat[c][c] += kf;
                    out.fs[c] += ff;
                }
                // stopp (lägen)
                if let Some((lo, hi)) = b.stop {
                    let p = cd.p0 + g[c];
                    let over = if hi.is_some_and(|h| p > h) { p - hi.unwrap() } else if lo.is_some_and(|l| p < l) { p - lo.unwrap() } else { 0.0 };
                    if over != 0.0 {
                        f[c] += e.k_pen[c] * over;
                        dmat[c][c] += e.k_pen[c];
                        out.fr[c] += e.k_pen[c] * over;
                    }
                }
            }
            // lås: när läget i COMPONENT når gränserna låses komponenterna
            for c in 0..6 {
                let Some(l) = &self.beh.comps[c].lock else { continue };
                if e.comps[c].status != Status::Available {
                    continue;
                }
                if !old[c].locked {
                    let p = e.comps[c].p0 + g[c];
                    let lim = if l.hi.is_some_and(|h| p >= h) { l.hi } else if l.lo.is_some_and(|lo| p <= lo) { l.lo } else { None };
                    if let Some(lim) = lim {
                        new[c].locked = true;
                        for &q in &l.comps {
                            new[q].lock_value = if q == c { lim - e.comps[c].p0 } else { 0.0 };
                            new[q].locked = new[q].locked || q != c;
                        }
                        // komponenten själv hålls vid gränsen
                        new[c].lock_value = lim - e.comps[c].p0;
                    }
                }
                if new[c].locked {
                    for &q in &l.comps {
                        // kriteriets komponent hålls vid gränsen redan i inkrementet där
                        // den nås; övriga låses från nästa inkrement (vid sin
                        // referens, CU = 0)
                        if q != c && !old[c].locked {
                            continue;
                        }
                        let v = if q == c { new[c].lock_value } else { new[q].lock_value };
                        f[q] += e.k_pen[q] * (g[q] - v);
                        dmat[q][q] += e.k_pen[q];
                        out.fr[q] += e.k_pen[q] * (g[q] - v);
                    }
                }
            }
            // viskös dämpning i dynamiksteg: hastigheten som Δu/Δt över
            // inkrementet
            if ctx.dt > 0.0 && when.is_some_and(|(si, _, _)| self.dynamic[si]) {
                for c in 0..6 {
                    let cv = self.beh.comps[c].damping;
                    if cv != 0.0 && e.comps[c].status == Status::Available {
                        let fv = cv * (g[c] - g_old[c]) / ctx.dt;
                        f[c] += fv;
                        dmat[c][c] += cv / ctx.dt;
                        out.fv[c] += fv;
                    }
                }
            }
            // matriselasticitet
            if let Some(mt) = &self.beh.matrix {
                for c in 0..6 {
                    for d in 0..6 {
                        if mt[c][d] != 0.0 {
                            f[c] += mt[c][d] * g[d];
                            out.fe[c] += mt[c][d] * g[d];
                            dmat[c][d] += mt[c][d];
                        }
                    }
                }
            }
            // föreskriven rörelse och last
            let mut load = [0.0; 6];
            if let Some((si, stt, frac)) = when {
                for c in 0..6 {
                    if let Some(r) = e.motion.get(si).and_then(|v| v[c].as_ref()) {
                        if r.release {
                            let fr = if self.timeline.steps[si].2 { old[c].fm * (1.0 - frac) } else { 0.0 };
                            f[c] += fr;
                            out.fr[c] += fr;
                            new[c].fm = old[c].fm;
                        } else {
                            let target = self.value(r, si, stt, frac, ctx.time);
                            let fp = e.k_pen[c] * (g[c] - target);
                            f[c] += fp;
                            dmat[c][c] += e.k_pen[c];
                            out.fr[c] += fp;
                            new[c].fm = fp;
                        }
                    }
                    if let Some(r) = e.load.get(si).and_then(|v| v[c].as_ref()) {
                        load[c] = self.value(r, si, stt, frac, ctx.time);
                    }
                }
            }
            // CTF/CTM: den kraft kopplingen överför (inre kraft minus *CONNECTOR LOAD)
            out.f = std::array::from_fn(|c| f[c] - load[c]);
            if crate::settings::with(|s| s.nl_debug) {
                eprintln!("koppling {}: t={:.4} g={:?} F={:?} D={:?}", ctx.model.elements[e.index].label, ctx.time, g, f, (0..6).map(|c| dmat[c][c]).collect::<Vec<_>>());
            }
            for c in 0..6 {
                let fc = f[c] - load[c];
                if fc != 0.0 {
                    for &(n, d, v) in &terms[c] {
                        fint[n][dof::slot_of(d)] += fc * v;
                    }
                }
            }
            if let Some(k) = k.as_deref_mut() {
                for c in 0..6 {
                    for d in 0..6 {
                        let kv = dmat[c][d];
                        if kv == 0.0 {
                            continue;
                        }
                        // symmetrisk del (matrisen är symmetrisk utom vid mjuknande skada)
                        let kv = 0.5 * (kv + dmat[d][c]);
                        for &(n1, d1, v1) in &terms[c] {
                            for &(n2, d2, v2) in &terms[d] {
                                k.add(n1, d1, n2, d2, kv * v1 * v2);
                            }
                        }
                    }
                }
                // NLGEOM: komponentkrafternas geometriska styvhet Σ F ∂²g/∂q²
                if let Some(ev) = &ev {
                    let fc: [f64; 6] = std::array::from_fn(|c| if e.comps[c].status == Status::None { 0.0 } else { f[c] - load[c] });
                    for (n1, d1, n2, d2, v) in e.geo.geometric(ev, &fc) {
                        k.add(n1, d1, n2, d2, v);
                    }
                }
            }
            st.trial[ei] = new;
            st.g_trial[ei] = g;
            st.out_trial[ei] = out;
        }
        Ok(())
    }

    /// Låsta komponenter som kinematiska villkor i explicita steg (NLGEOM-modeller).
    fn kinematic_constraints(&self, u: &[NodeVec], nlgeom: bool) -> Vec<(Vec<(usize, Dof, f64)>, f64)> {
        if !self.nl_mode {
            return Vec::new();
        }
        let st = self.state.lock().unwrap();
        let mut out = Vec::new();
        for (ei, e) in self.elems.iter().enumerate() {
            if !(0..6).any(|c| e.comps[c].status == Status::Constrained) {
                continue;
            }
            let ev = if nlgeom { Some(self.eval_nl(e, u, &st.g_committed[ei])) } else { None };
            for c in 0..6 {
                if e.comps[c].status != Status::Constrained {
                    continue;
                }
                match &ev {
                    Some(v) => out.push((v.rows[c].clone(), v.g[c])),
                    None => out.push((e.terms[c].clone(), e.terms[c].iter().map(|&(n, d, v)| v * u[n][dof::slot_of(d)]).sum())),
                }
            }
        }
        out
    }

    fn set_constraint_forces(&self, lambda: &[f64]) {
        let mut st = self.state.lock().unwrap();
        let mut k = 0;
        for (ei, e) in self.elems.iter().enumerate() {
            for c in 0..6 {
                if e.comps[c].status == Status::Constrained {
                    st.lambda_trial[ei][c] = lambda.get(k).copied().unwrap_or(0.0);
                    k += 1;
                }
            }
        }
    }

    fn commit(&self) {
        let mut st = self.state.lock().unwrap();
        let st = &mut *st;
        st.committed.clone_from(&st.trial);
        st.g_committed.clone_from(&st.g_trial);
        st.out_committed.clone_from(&st.out_trial);
        st.lambda_committed.clone_from(&st.lambda_trial);
    }

    fn elem_output(&self) -> Vec<(usize, IpField)> {
        let st = self.state.lock().unwrap();
        let (tl, rl): (&[usize], &[usize]) = if self.dim == 2 { (&[0, 1], &[5]) } else { (&[0, 1, 2], &[3, 4, 5]) };
        let mut res = Vec::new();
        for (ei, e) in self.elems.iter().enumerate() {
            let mut o = st.out_committed.get(ei).copied().unwrap_or_default();
            let lam = st.lambda_committed.get(ei).copied().unwrap_or_default();
            for c in 0..6 {
                o.f[c] += lam[c];
                o.fr[c] += lam[c];
            }
            let s = st.committed.get(ei).copied().unwrap_or_default();
            let avail = |c: usize| e.comps[c].status == Status::Available;
            let gv = |c: usize| if avail(c) { o.g[c] } else { 0.0 };
            let mut add = |name: &str, base: &str, idx: &[usize], val: &dyn Fn(usize) -> f64| {
                let labels: Vec<String> = idx.iter().map(|&c| format!("{base}{}", if c >= 3 { c - 2 } else { c + 1 })).collect();
                let mut f = IpField { name: name.to_string(), labels, points: Vec::new() };
                f.points.push((0, 0, idx.iter().map(|&c| val(c)).collect()));
                res.push((e.index, f));
            };
            add("CTF", "CTF", tl, &|c| o.f[c]);
            add("CTM", "CTM", rl, &|c| o.f[c]);
            add("CU", "CU", tl, &gv);
            add("CUR", "CUR", rl, &gv);
            add("CEF", "CEF", tl, &|c| o.fe[c]);
            add("CEM", "CEM", rl, &|c| o.fe[c]);
            add("CUE", "CUE", tl, &|c| if avail(c) { o.g[c] - s[c].up } else { 0.0 });
            add("CURE", "CURE", rl, &|c| if avail(c) { o.g[c] - s[c].up } else { 0.0 });
            add("CUP", "CUP", tl, &|c| s[c].up);
            add("CURP", "CURP", rl, &|c| s[c].up);
            add("CSF", "CSF", tl, &|c| o.fs[c]);
            add("CSM", "CSM", rl, &|c| o.fs[c]);
            add("CRF", "CRF", tl, &|c| o.fr[c]);
            add("CRM", "CRM", rl, &|c| o.fr[c]);
            add("CVF", "CVF", tl, &|c| o.fv[c]);
            add("CVM", "CVM", rl, &|c| o.fv[c]);
            add("CP", "CP", tl, &|c| if e.comps[c].status == Status::None { 0.0 } else { e.comps[c].p0 + o.g[c] });
            add("CPR", "CPR", rl, &|c| if e.comps[c].status == Status::None { 0.0 } else { e.comps[c].p0 + o.g[c] });
            add("CDMG", "CDMG", tl, &|c| s[c].d);
            add("CDMGR", "CDMGR", rl, &|c| s[c].d);
        }
        res
    }
}

/// Lokala axlar från *ORIENTATION (SYSTEM=RECTANGULAR, punkterna a och b,
/// valfri extra rotation kring en lokal axel). Kräver att nyckelordet sparats
/// i modellen.
fn orientation(m: &FeModel, kw: &RawKeyword, name: &str) -> Result<[[f64; 3]; 3]> {
    let up = name.trim().to_uppercase();
    let Some(o) = m.extra.iter().find(|k| k.name == "ORIENTATION" && k.param_upper("NAME").as_deref() == Some(up.as_str())) else {
        return kw.err(format!("*ORIENTATION, NAME={name} finns inte (eller sparas inte av .inp-läsaren)"));
    };
    if o.param_upper("SYSTEM").is_some_and(|s| s != "RECTANGULAR") {
        return o.err("bara SYSTEM=RECTANGULAR stöds för kopplingselement");
    }
    if o.has("DEFINITION") && o.param_upper("DEFINITION").as_deref() != Some("COORDINATES") {
        return o.err("bara DEFINITION=COORDINATES stöds för kopplingselement");
    }
    let v = o.first_floats()?;
    if v.len() < 6 {
        return o.err("skriv a1, a2, a3, b1, b2, b3");
    }
    let (a, b) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
    let e1 = scale(a, 1.0 / norm(a));
    let bb = sub(b, scale(e1, dot(b, e1)));
    let e2 = scale(bb, 1.0 / norm(bb));
    let e3 = cross(e1, e2);
    let mut ax = [e1, e2, e3];
    if let Some(d) = o.data.get(1) {
        let f: Vec<f64> = d.fields.iter().filter(|s| !s.trim().is_empty()).filter_map(|s| parse_f64(s)).collect();
        if f.len() >= 2 && f[1] != 0.0 {
            let i = (f[0] as usize).clamp(1, 3) - 1;
            let (c, s) = (f[1].to_radians().cos(), f[1].to_radians().sin());
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            let (ej, ek) = (ax[j], ax[k]);
            ax[j] = [0, 1, 2].map(|q| c * ej[q] + s * ek[q]);
            ax[k] = [0, 1, 2].map(|q| -s * ej[q] + c * ek[q]);
        }
    }
    Ok(ax)
}

/// Löser A x = b (n×n, radvis) med Gausselimination och pivotering; singulära
/// riktningar ger 0.
fn solve_small(a: &mut [f64], b: &mut [f64], n: usize) -> Vec<f64> {
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0, f64::max).max(1e-300);
    for k in 0..n {
        let p = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs())).unwrap();
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            b.swap(k, p);
        }
        let d = a[k * n + k];
        if d.abs() <= 1e-12 * scale {
            continue;
        }
        for i in k + 1..n {
            let f = a[i * n + k] / d;
            if f != 0.0 {
                for j in k..n {
                    a[i * n + j] -= f * a[k * n + j];
                }
                b[i] -= f * b[k];
            }
        }
    }
    let mut x = vec![0.0; n];
    for k in (0..n).rev() {
        let d = a[k * n + k];
        if d.abs() <= 1e-12 * scale {
            continue;
        }
        let mut s = b[k];
        for j in k + 1..n {
            s -= a[k * n + j] * x[j];
        }
        x[k] = s / d;
    }
    x
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// Noden har rotationsstyvhet eller randvillkor i rotation utanför elementen
/// (*SPRING i frihetsgrad 4–6, *BOUNDARY), så att a:s medroterande system kan
/// användas.
fn rot_supported(m: &FeModel, n: usize) -> bool {
    let bc = m.initial_bcs.iter().chain(m.steps.iter().flat_map(|s| s.bcs.iter())).any(|b| b.items.iter().any(|&(q, d, _)| q == n && (4..=6).contains(&d)));
    if bc {
        return true;
    }
    m.extra.iter().filter(|k| k.name == "SPRING").any(|kw| {
        let rot = kw.data.first().is_some_and(|d| d.fields.iter().filter_map(|s| parse_f64(s.trim())).any(|x| x > 3.5));
        rot && kw.param("ELSET").and_then(|es| m.elsets.get(&es.trim().to_uppercase())).is_some_and(|set| set.iter().any(|&e| m.elements[e].nodes.contains(&n)))
    })
}

/// Största diagonalterm (translation, rotation) i styvheten hos elementen
/// och fjädrarna vid noden `n`.
fn node_stiffness(m: &FeModel, n: usize) -> (f64, f64) {
    let (mut kt, mut kr) = (0.0f64, 0.0f64);
    for (e, el) in m.elements.iter().enumerate() {
        if !el.nodes.contains(&n) || el.etype.name().starts_with("CONN") {
            continue;
        }
        let section = el.section.map(|s| &m.sections[s]);
        let mat = match section.and_then(|s| s.material_index) {
            Some(i) => crate::materials::build(&m.materials[i]).ok(),
            None => None,
        };
        let ctx = ElemCtx {
            index: e,
            label: el.label,
            x: el.nodes.iter().map(|&q| m.nodes[q]).collect(),
            section,
            material: mat.as_deref(),
            density: 0.0,
            nlgeom: false,
            time: 0.0,
            dt: 0.0,
            temp: Vec::new(),
            temp_old: Vec::new(),
            orientation: None,
        };
        let Ok(k) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| el.etype.stiffness(&ctx))) else { continue };
        let Ok(k) = k else { continue };
        let nd = el.etype.n_dofs();
        if k.len() != nd * nd {
            continue;
        }
        let mut q = 0;
        for (a, &na) in el.nodes.iter().enumerate() {
            for &d in el.etype.node_dofs(a) {
                if na == n {
                    let v = k[q * nd + q].abs();
                    if d <= 3 {
                        kt = kt.max(v);
                    } else if d <= 6 {
                        kr = kr.max(v);
                    }
                }
                q += 1;
            }
        }
    }
    // *SPRING (SPRING1/SPRING2): första raden frihetsgrader, andra styvheten
    for kw in m.extra.iter().filter(|k| k.name == "SPRING") {
        let Some(es) = kw.param("ELSET") else { continue };
        let Some(set) = m.elsets.get(&es.trim().to_uppercase()) else { continue };
        if !set.iter().any(|&e| m.elements[e].nodes.contains(&n)) {
            continue;
        }
        let dofs: Vec<f64> = kw.data.first().map(|d| d.fields.iter().filter_map(|s| parse_f64(s.trim())).collect()).unwrap_or_default();
        let v = kw.data.get(1).and_then(|d| d.fields.first()).and_then(|s| parse_f64(s.trim())).unwrap_or(0.0).abs();
        if dofs.iter().any(|&d| d > 3.5) {
            kr = kr.max(v);
        } else {
            kt = kt.max(v);
        }
    }
    (kt, kr)
}

/// Föreskrivna värden per element och komponent för varje modellsteg från
/// stegnyckelordet `name` (*CONNECTOR MOTION/LOAD).
fn prescribed(m: &FeModel, name: &str, elem_of: &BTreeMap<usize, usize>, n_elems: usize, avail: &dyn Fn(usize, usize) -> bool) -> Result<Vec<Vec<[Option<Ramp>; 6]>>> {
    // per element: per steg
    let mut out: Vec<Vec<[Option<Ramp>; 6]>> = vec![Vec::new(); n_elems];
    let mut cur: Vec<[Option<f64>; 6]> = vec![[None; 6]; n_elems];
    let mut amp_cur: Vec<[Option<String>; 6]> = vec![Default::default(); n_elems];
    for s in &m.steps {
        let mut set: Vec<[Option<f64>; 6]> = cur.clone();
        let mut amp = amp_cur.clone();
        let from: Vec<[Option<f64>; 6]> = cur.clone();
        for kw in s.extra.iter().filter(|k| k.name == name) {
            if kw.param_upper("OP").as_deref() == Some("NEW") {
                set = vec![[None; 6]; n_elems];
                amp = vec![Default::default(); n_elems];
            }
            if let Some(t) = kw.param_upper("TYPE") {
                if t != "DISPLACEMENT" {
                    return kw.err(format!("TYPE={t} stöds inte ännu (DISPLACEMENT)"));
                }
            }
            let a = kw.param_upper("AMPLITUDE");
            for d in &kw.data {
                let f: Vec<&str> = d.fields.iter().map(|x| x.trim()).collect();
                if f.len() < 2 || f[0].is_empty() {
                    continue;
                }
                let Some(c) = crate::inp::parse_label(f[1]).filter(|c| (1..=6).contains(c)) else {
                    return kw.err_at(d.line, format!("'{}' är ingen komponent (1–6)", f[1]));
                };
                let c = c as usize - 1;
                let v = f.get(2).and_then(|x| if x.is_empty() { Some(0.0) } else { parse_f64(x) }).unwrap_or(0.0);
                for e in geom::elem_ref(m, kw, f[0])? {
                    let Some(&i) = elem_of.get(&e) else { continue };
                    if !avail(i, c) {
                        return kw.err_at(d.line, format!("komponent {} är inte tillgänglig i kopplingselementet {}", c + 1, m.elements[e].label));
                    }
                    set[i][c] = Some(v);
                    amp[i][c] = a.clone();
                }
            }
        }
        for i in 0..n_elems {
            let mut row: [Option<Ramp>; 6] = Default::default();
            for c in 0..6 {
                if let Some(v) = set[i][c] {
                    row[c] = Some(Ramp { from: from[i][c].unwrap_or(0.0), to: v, amp: amp[i][c].clone(), release: false });
                } else if let Some(v) = from[i][c] {
                    // borttagen med OP=NEW: lasten respektive rörelsens reaktion
                    // avvecklas till noll under steget (som för laster och randvillkor)
                    let load = name == "CONNECTOR LOAD";
                    row[c] = Some(Ramp { from: if load { v } else { 0.0 }, to: 0.0, amp: None, release: !load });
                }
            }
            out[i].push(row);
        }
        // laster och rörelser med amplitud fortsätter med sitt slutvärde
        cur = set;
        amp_cur = amp;
    }
    Ok(out)
}

fn section(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let elset = crate::inp::required(kw, "ELSET")?;
    let elems = geom::elem_ref(m, kw, elset)?;
    let recs: Vec<Vec<String>> = kw.data.iter().map(|d| d.fields.iter().map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty()).collect()).collect();
    let Some(types) = recs.first().filter(|r| !r.is_empty()) else {
        return kw.err("kopplingstyp saknas på första dataraden");
    };
    let dim = m.dim();
    let ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let ea = match recs.get(1).and_then(|r| r.first()) {
        Some(o) => orientation(m, kw, o)?,
        None => ident,
    };
    let eb = match recs.get(1).and_then(|r| r.get(1)) {
        Some(o) => orientation(m, kw, o)?,
        None => ea,
    };
    let elem_active = geom::element_active(m);
    // komponenternas status är densamma för alla element i sektionen
    let probe = kin::components(types, [1.0, 0.7, 0.3], &ea, &eb).map_err(|e| crate::model::FemError(format!("{}: {e}", kw.location())))?;
    let comp_ok = |c: usize| dim == 3 || matches!(c, 0 | 1 | 5);
    let avail: Vec<usize> = (0..6).filter(|&c| probe[c].status == Status::Available && comp_ok(c)).collect();
    let beh = match kw.param("BEHAVIOR") {
        Some(b) => beh::parse(m, kw, b, &avail)?,
        None => Behavior::default(),
    };
    let mut out_elems = Vec::new();
    let mut active = Vec::new();
    let mut elem_of = BTreeMap::new();
    let rot_dofs: &[Dof] = geom::rot_dofs(dim);
    for &e in &elems {
        let el = &m.elements[e];
        if !matches!(el.etype.name(), "CONN2D2" | "CONN3D2") {
            return kw.err(format!("elementet {} ({}) är inget kopplingselement", el.label, el.etype.name()));
        }
        // en nod: den andra noden är fast (jordad)
        let (a, b) = match el.nodes.len() {
            2 => (Some(el.nodes[0]), Some(el.nodes[1])),
            1 => (Some(el.nodes[0]), None),
            _ => return fail(format!("{}: kopplingselementet {} har fel antal noder", kw.location(), el.label)),
        };
        let xa = a.map(|n| m.nodes[n]);
        let xb = b.map(|n| m.nodes[n]);
        let (xa, xb) = (xa.unwrap_or_else(|| xb.unwrap()), xb.unwrap_or_else(|| xa.unwrap()));
        let d = sub(xb, xa);
        let comps = kin::components(types, d, &ea, &eb).map_err(|err| crate::model::FemError(format!("{}: element {}: {err}", kw.location(), el.label)))?;
        let has_rot_comps = (3..6).any(|c| comps[c].status != Status::None && comp_ok(c));
        let rot_ok = |n: Option<usize>| n.is_some_and(|n| has_rot_comps || rot_dofs.iter().any(|&q| geom::has(&elem_active, n, q)) || rot_supported(m, n));
        let (ra, rb) = (rot_ok(a), rot_ok(b));
        let mut terms: [Vec<(usize, Dof, f64)>; 6] = Default::default();
        let mut comps2 = comps;
        for c in 0..6 {
            if !comp_ok(c) {
                comps2[c].status = Status::None;
                continue;
            }
            if comps[c].status == Status::None {
                continue;
            }
            let row = comps[c].row;
            for (blk, node, rot, okr) in [(0usize, a, false, true), (3, a, true, ra), (6, b, false, true), (9, b, true, rb)] {
                let Some(n) = node else { continue };
                if rot && !okr {
                    continue;
                }
                for q in 0..3 {
                    let v = row[blk + q];
                    if v.abs() < 1e-14 {
                        continue;
                    }
                    if dim == 2 && ((!rot && q == 2) || (rot && q != 2)) {
                        continue;
                    }
                    let dd = if rot { UR1 + q as Dof } else { U1 + q as Dof };
                    terms[c].push((n, dd, v));
                    active.push((n, dd));
                }
            }
        }
        for n in [a, b].into_iter().flatten() {
            active.extend(geom::trans_dofs(dim).iter().map(|&q| (n, q)));
        }
        // straffkoefficient från nodernas styvhet och kopplingens egen
        let mut kt = 0.0f64;
        let mut kr = 0.0f64;
        for n in [a, b].into_iter().flatten() {
            let (t, r) = node_stiffness(m, n);
            kt = kt.max(t);
            kr = kr.max(r);
        }
        for c in 0..6 {
            let s = beh.comps[c].elastic.scale().max(beh.matrix.map_or(0.0, |mt| mt[c][c].abs()));
            if c < 3 {
                kt = kt.max(s);
            } else {
                kr = kr.max(s);
            }
        }
        let len = norm(d);
        let l2 = if len > 0.0 { len * len } else { 1.0 };
        if kt == 0.0 {
            kt = if kr > 0.0 { kr / l2 } else { 1.0 };
        }
        if kr == 0.0 {
            kr = kt * l2;
        }
        let mut k_pen = [0.0; 6];
        for c in 0..6 {
            k_pen[c] = PENALTY * if c < 3 { kt } else { kr };
        }
        if std::env::var_os("RUSTYFEM_CONN_DEBUG").is_some() {
            eprintln!("koppling {}: kt={kt:.4e} kr={kr:.4e} termer={:?}", el.label, terms);
        }
        elem_of.insert(e, out_elems.len());
        let geo = nl::Geo::new(a, b, xa, xb, ea, eb, ra, rb, dim, types);
        out_elems.push(Elem { index: e, bnode: b.or(a).unwrap(), comps: comps2, terms, k_pen, length: len, motion: Vec::new(), load: Vec::new(), geo });
    }
    let n = out_elems.len();
    let avail_of = |i: usize, c: usize| out_elems[i].comps[c].status == Status::Available;
    let motion = prescribed(m, "CONNECTOR MOTION", &elem_of, n, &avail_of)?;
    let load = prescribed(m, "CONNECTOR LOAD", &elem_of, n, &avail_of)?;
    let any_motion = motion.iter().flatten().any(|r| r.iter().any(|x| x.is_some()));
    let any_load = load.iter().flatten().any(|r| r.iter().any(|x| x.is_some()));
    for (i, (mo, lo)) in motion.into_iter().zip(load).enumerate() {
        out_elems[i].motion = mo;
        out_elems[i].load = lo;
    }
    active.sort_unstable();
    active.dedup();
    let damped = beh.comps.iter().any(|c| c.damping != 0.0);
    let linear = beh.is_linear() && !any_motion && !any_load && !damped;
    let st = State {
        committed: vec![[CompState::default(); 6]; n],
        trial: vec![[CompState::default(); 6]; n],
        g_committed: vec![[0.0; 6]; n],
        g_trial: vec![[0.0; 6]; n],
        out_trial: vec![Out::default(); n],
        out_committed: vec![Out::default(); n],
        lambda_trial: vec![[0.0; 6]; n],
        lambda_committed: vec![[0.0; 6]; n],
    };
    Ok(vec![Box::new(Connectors {
        name: format!("*CONNECTOR SECTION ({})", kw.location()),
        dim,
        elems: out_elems,
        beh,
        active,
        timeline: Timeline::new(m)?,
        amps: crate::steps::dynamic::amplitude::amplitudes(m)?,
        linear,
        dynamic: m.steps.iter().map(|s| s.procedure_name() == "DYNAMIC").collect(),
        nl_mode: m.steps.iter().any(|s| s.nlgeom == Some(true) || is_explicit(s)),
        explicit: m.steps.iter().map(is_explicit).collect(),
        state: Mutex::new(st),
    })])
}

/// Steget är explicit dynamik (*DYNAMIC, EXPLICIT).
fn is_explicit(s: &crate::fem::model::StepDef) -> bool {
    s.procedure_name() == "DYNAMIC" && s.procedure.as_ref().is_some_and(|p| p.has("EXPLICIT"))
}
