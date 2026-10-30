//! *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL (fully coupled
//! thermal-electrical-structural analysis; the synonym *COUPLED
//! THERMAL-ELECTRICAL-STRUCTURAL is also accepted, area coupled2) with the elements
//! Q3D4, Q3D8 and Q3D20 (`elements/thermal_electric_structural.rs`).
//!
//! Temperature, electric potential and displacement. Without contact the
//! thermal-electrical part does not depend on the displacements, so each increment is solved
//! exactly in two stages (like `heat_coupled.rs`): first temperature and potential
//! with Joule heating (`thermal_electric.rs`, fixed-point iteration), then
//! the equilibrium with the thermal strain from the temperatures at the start and end
//! of the increment (`heat_stress.rs`). The converged solution is
//! the same as the fully coupled Newton solution.
//!
//! Steady-state or transient with fixed increments. Output: U, RF, S, E,
//! NT11, RFL11, HFL, EPOT, RECUR, ECD, EPG, JENER. Included with `#[path]`
//! from `thermal_electric.rs`.

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

/// An increment [t0, t1] of the equilibrium with fixed length.
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

/// The mechanics' loads (displacements, forces, pressures) and the rest.
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

/// The frames of the mechanics and of the thermal-electrical part merged.
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
        // Q3D4: one integration point (constant gradient)
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
        // *COUPLED TEMPERATURE-DISPLACEMENT, ELECTRICAL (before heat_coupled.rs);
        // *COUPLED THERMAL-ELECTRICAL-STRUCTURAL is accepted as a synonym
        let p = step.procedure.as_ref()?;
        let electrical = p.name == "COUPLED TEMPERATURE-DISPLACEMENT" && p.has("ELECTRICAL") && !matches!(p.param_upper("ELECTRICAL").as_deref(), Some("NO"));
        (p.name == PROC || electrical).then_some(1)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.step.perturbation {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL cannot be a perturbation step");
        }
        if ctx.nlgeom {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL with NLGEOM=YES is not supported yet");
        }
        if !prep.interactions.is_empty() {
            return fail("*COUPLED THERMAL-ELECTRICAL-STRUCTURAL with contact or constraints is not supported yet");
        }
        for (e, el) in model.elements.iter().enumerate() {
            if crate::elements::thermal_electric_structural::lookup(el.etype.name()).is_none() {
                return fail(format!("{}: *COUPLED THERMAL-ELECTRICAL-STRUCTURAL requires the elements Q3D4, Q3D8 or Q3D20, not {}", model.elem_name(e), el.etype.name()));
            }
        }
        let hc = heat_controls(ctx.step)?;
        if hc.deltmx > 0.0 && !hc.steady {
            return fail("DELTMX in *COUPLED THERMAL-ELECTRICAL-STRUCTURAL is not supported yet (give fixed increments)");
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
        // the potential is not a degree of freedom in the equilibrium
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
                return fail(format!("the step reached the maximum number of increments ({}) – increase INC= on *STEP", ctrl.max_increments));
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
                eprintln!("  increment {inc}: step time {:.6}", t * period);
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
