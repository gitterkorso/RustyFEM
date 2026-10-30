//! *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL (fullt kopplad
//! termisk-elektrisk-strukturell analys; synonymen *COUPLED
//! THERMAL-ELECTRICAL-STRUCTURAL godtas också, område coupled2) med elementen
//! Q3D4, Q3D8 och Q3D20 (`elements/thermal_electric_structural.rs`).
//!
//! Temperatur, elektrisk potential och förskjutning. Utan kontakt beror den
//! termisk-elektriska delen inte av förskjutningarna, så varje inkrement löses
//! exakt i två led (som `heat_coupled.rs`): först temperatur och potential
//! med Joulevärme (`thermal_electric.rs`, fixpunktsiteration), sedan
//! jämvikten med den termiska töjningen ur temperaturerna vid inkrementets
//! början och slut (`heat_stress.rs`). Den konvergerade lösningen är
//! densamma som den helt kopplade Newtonlösningen.
//!
//! Stationärt eller transient med fasta inkrement. Utdata: U, RF, S, E,
//! NT11, RFL11, HFL, EPOT, RECUR, ECD, EPG, JENER. Inkluderas med `#[path]`
//! från `thermal_electric.rs`.

use super::super::heat::{ensure_initial, heat_controls, heat_loads, HeatSystem, INIT_KEY, TS};
use super::super::heat_stress::ThermoSystem;
use super::super::newton::{self, Controls, NonlinearSystem, Residual};
use super::{coupled_increment, frame, split_loads, temps, ElecSystem, JENER_KEY, PS};
use crate::fem::dof::{Field, SLOTS};
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::fem::state::StepLoads;
use crate::model::{fail, Result};
use crate::steps::api::{StepCtx, StepType};

pub struct ThermalElectricStructural;

pub static TES: ThermalElectricStructural = ThermalElectricStructural;

const PROC: &str = "COUPLED THERMAL-ELECTRICAL-STRUCTURAL";

/// Ett inkrement [t0, t1] av jämvikten med fast längd.
struct Span<'a, 'p, 'm> {
    sys: &'a mut ThermoSystem<'p, 'm>,
    t0: f64,
    t1: f64,
}

impl NonlinearSystem for Span<'_, '_, '_> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        self.sys.is_linear()
    }
    fn begin_increment(&mut self, a: f64, b: f64) -> Result<()> {
        let h = self.t1 - self.t0;
        self.sys.begin_increment(self.t0 + a * h, self.t0 + b * h)
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
        self.sys.commit()
    }
}

/// Mekanikens laster (förskjutningar, krafter, tryck) och resten.
fn split_mech(l: &StepLoads) -> (StepLoads, StepLoads) {
    let other_kind = |k: &str| k.starts_with("DFLUX:") || k.starts_with("FILM") || k.starts_with("RAD:") || k.starts_with("ECUR:");
    let other_slot = |s: usize| s == TS || s == PS;
    let mut me = l.clone();
    let mut te = l.clone();
    me.bc.retain(|&(_, s), _| !other_slot(s));
    te.bc.retain(|&(_, s), _| other_slot(s));
    me.cload.retain(|&(_, s), _| !other_slot(s));
    te.cload.retain(|&(_, s), _| other_slot(s));
    me.dload.retain(|(d, _)| !other_kind(&d.kind));
    te.dload.retain(|(d, _)| other_kind(&d.kind));
    me.released.retain(|&(_, s, _)| !other_slot(s));
    te.released.retain(|&(_, s, _)| other_slot(s));
    me.amp_dload.clear();
    (me, te)
}

/// Mekanikens och den termisk-elektriska delens frame sammanslagna.
fn merged(heat: &HeatSystem, elec: &ElecSystem, mech: &ThermoSystem, value: f64, description: String) -> Result<FrameData> {
    let te = frame(heat, elec, value, description.clone())?;
    let mut m = mech.frame(value, description)?;
    for n in 0..m.u.len() {
        for s in [TS, PS] {
            m.u[n][s] = te.u[n][s];
            m.rf[n][s] = te.rf[n][s];
            let bit = 1u16 << s;
            m.active[n] = (m.active[n] & !bit) | (te.active[n] & bit);
            m.constrained[n] = (m.constrained[n] & !bit) | (te.constrained[n] & bit);
        }
    }
    let model = heat.p.model;
    for (e, (a, b)) in m.elem.iter_mut().zip(te.elem).enumerate() {
        // Q3D4: en integrationspunkt (konstant gradient)
        let one = model.elements[e].etype.n_ips() == 1;
        a.extend(b.into_iter().map(|mut f| {
            if one {
                f.points.truncate(1);
                if let Some(p) = f.points.first_mut() {
                    p.0 = 1;
                }
            }
            f
        }));
    }
    m.node_extra.extend(te.node_extra);
    Ok(m)
}

impl StepType for ThermalElectricStructural {
    fn name(&self) -> &'static str {
        PROC
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        // *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL (före heat_coupled.rs);
        // *COUPLED THERMAL-ELECTRICAL-STRUCTURAL godtas som synonym
        let p = step.procedure.as_ref()?;
        let electrical = p.name == "COUPLED TEMPERATURE-DISPLACEMENT" && p.has("ELECTRICAL") && !matches!(p.param_upper("ELECTRICAL").as_deref(), Some("NO"));
        (p.name == PROC || electrical).then_some(1)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.step.perturbation {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL kan inte vara ett perturbationssteg");
        }
        if ctx.nlgeom {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL med NLGEOM=YES stöds inte ännu");
        }
        if !prep.interactions.is_empty() {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL med kontakt eller villkor stöds inte ännu");
        }
        for (e, el) in model.elements.iter().enumerate() {
            if crate::elements::thermal_electric_structural::lookup(el.etype.name()).is_none() {
                return fail(format!("{}: *COUPLED THERMAL-ELECTRICAL-STRUCTURAL kräver elementen Q3D4, Q3D8 eller Q3D20, inte {}", model.elem_name(e), el.etype.name()));
            }
        }
        let hc = heat_controls(ctx.step)?;
        if hc.deltmx > 0.0 && !hc.steady {
            return fail("DELTMX i *COUPLED THERMAL-ELECTRICAL-STRUCTURAL stöds inte ännu (ange fasta inkrement)");
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
        let (mut me, te) = split_mech(&loads);
        // potentialen är inte en frihetsgrad i jämvikten
        for (n, &a) in prep.active.iter().enumerate() {
            if a & (1 << PS) != 0 {
                me.bc.insert((n, PS), (0.0, 0.0));
            }
        }
        let (th, el) = split_loads(&te);
        let period = hc.ctrl.period;
        let mut heat = HeatSystem::new(prep, th, ctx.state, !hc.steady, ctx.solver, period)?;
        let mut elec = ElecSystem::new(prep, el, ctx.state, ctx.solver)?;
        elec.transient = !hc.steady;
        let me = me;
        let mut mech = ThermoSystem::new(prep, &me, ctx.state, ctx.solver, period, true)?;
        mech.coupled = (temps(&ctx.state.u), temps(&ctx.state.u));
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            heat.begin_increment(0.0, 0.0)?;
            heat.u.clone_from(&heat.u_committed);
            let tr = heat.transient;
            heat.transient = false;
            heat.evaluate(false)?;
            heat.transient = tr;
            mech.begin_increment(0.0, 0.0)?;
            mech.evaluate(false)?;
            let mut f0 = merged(&heat, &elec, &mech, 0.0, "Increment 0: Step Time = 0.0".into())?;
            for r in f0.rf.iter_mut() {
                *r = [0.0; SLOTS];
            }
            frames.push(f0);
        }
        let ctrl = &hc.ctrl;
        let dt0 = ctrl.initial.clamp(1e-12, 1.0);
        let sub = Controls { initial: 1.0, fixed: true, ..ctrl.clone() };
        let mut t = 0.0f64;
        let mut inc = 0usize;
        let mut iterations = 0usize;
        while t < 1.0 - 1e-12 {
            if inc >= ctrl.max_increments {
                return fail(format!("steget nådde största antal inkrement ({}) – öka INC= på *STEP", ctrl.max_increments));
            }
            let t1 = (t + dt0).min(1.0);
            let t1 = if 1.0 - t1 < 1e-9 { 1.0 } else { t1 };
            let before = temps(&heat.u_committed);
            iterations += coupled_increment(&mut heat, &mut elec, ctrl, t, t1)?;
            mech.coupled = (before, temps(&heat.u_committed));
            let mut span = Span { sys: &mut mech, t0: t, t1 };
            iterations += newton::run(&mut span, &sub, &mut |_, _| Ok(()))?.iterations;
            mech.commit()?;
            t = t1;
            inc += 1;
            if ctx.opts.verbose {
                eprintln!("  inkrement {inc}: stegtid {:.6}", t * period);
            }
            if all || t >= 1.0 - 1e-12 {
                frames.push(merged(&heat, &elec, &mech, t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = inc;
        ctx.out.iterations = iterations;
        let mut u = mech.u_committed.clone();
        let mut rf = mech.reactions();
        let hr = heat.reactions();
        let rec = elec.reactions();
        for n in 0..u.len() {
            u[n][TS] = heat.u_committed[n][TS];
            rf[n][TS] = hr[n][TS];
            u[n][PS] = elec.phi[n];
            rf[n][PS] = rec[n];
        }
        ctx.state.u = u;
        ctx.state.reaction = rf;
        ctx.state.elem_state = mech.state_old.clone();
        ctx.state.vectors.insert(JENER_KEY.into(), elec.jener_vec());
        ctx.state.time += period;
        Ok(())
    }
}
