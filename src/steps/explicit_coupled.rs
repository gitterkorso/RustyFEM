//! *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT: coupled thermomechanics with
//! explicit time integration for the coupled elements
//! CPE3T, CPS3T and C3D4T (`elements/heat_coupled.rs`).
//!
//! Area coupled2: the procedure is now run by the general explicit solver
//! (`explicit.rs` with `explicit_coupled_full.rs`, priority 1), which also
//! has the RT elements, bulk viscosity, NLGEOM, contact and plastic heating.
//! This step type (priority 0) is no longer used when both exist, and
//! the limitations below apply only to it.
//!
//! In each increment i:
//!
//! ```text
//! ü(i)    = M⁻¹ (P(i) − I(u(i), θ(i)))           diagonal mass
//! u̇(i+½) = u̇(i−½) + ½(Δt(i+1) + Δt(i)) ü(i),   u(i+1) = u(i) + Δt(i+1) u̇(i+½)
//! θ̇(i)    = C⁻¹ (Q(i) − F(θ(i)))                  diagonal (lumped) heat capacity
//! θ(i+1)  = θ(i) + Δt(i+1) θ̇(i)                   forward Euler
//! ```
//!
//! The internal force I is computed with the thermal strain from the nodal temperatures
//! (`ElemCtx::temp`), the heat flux F with conduction, *CFLUX, *DFLUX, *FILM and
//! *RADIATE from `steps/heat.rs` (`HeatSystem`). Prescribed temperatures and
//! displacements are set in each increment.
//!
//! Time increment: `DIRECT USER CONTROL` or `FIXED TIME INCREMENTATION`
//! with Δt on the data line (the step is divided into ⌈T/Δt⌉ equal increments, as
//! *DYNAMIC, EXPLICIT), otherwise the minimum over the elements of the mechanical limit
//! Lₑ/c_d and the thermal Lₑ²/(2α) (α = k/(ρc), Lₑ = cube/square root of
//! the element volume/area) times `SCALE FACTOR`; the test cases give a fixed Δt.
//! Bulk viscosity is not included (corresponds to `*BULK VISCOSITY` 0, 0). NLGEOM, contact and heat generation from plastic dissipation
//! are not supported.
//!
//! Output: U, V, A, NT11, RF, S, E, HFL with frames according to *OUTPUT, FIELD
//! (NUMBER INTERVAL/TIME INTERVAL/FREQUENCY).

use super::api::{StepCtx, StepType};
use super::dynamic::amplitude::LoadEval;
use super::dynamic::{add_va, output_schedule, Schedule};
use super::heat::{ensure_initial, heat_loads, HeatSystem, TS};
use super::newton::NonlinearSystem;
use crate::elements::MassKind;
use crate::fem::assembly::{self, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::StepLoads;
use crate::materials::heat::HeatProps;
use crate::model::{fail, FemError, Result};

pub struct ExplicitCoupled;

pub static STEP_TYPES: &[&dyn StepType] = &[&ExplicitCoupled];

const PROC: &str = "DYNAMIC TEMPERATURE-DISPLACEMENT";

/// The heat conduction loads (temperatures and heat fluxes) and the mechanical ones.
fn split_loads(l: &StepLoads) -> (StepLoads, StepLoads) {
    let thermal_kind = |k: &str| k.starts_with("DFLUX:") || k.starts_with("FILM") || k.starts_with("RAD:");
    let mut th = l.clone();
    let mut me = l.clone();
    th.bc.retain(|&(_, s), _| s == TS);
    me.bc.retain(|&(_, s), _| s != TS);
    th.cload.retain(|&(_, s), _| s == TS);
    me.cload.retain(|&(_, s), _| s != TS);
    th.dload.retain(|(d, _)| thermal_kind(&d.kind));
    me.dload.retain(|(d, _)| !thermal_kind(&d.kind));
    th.released.retain(|&(_, s, _)| s == TS);
    me.released.retain(|&(_, s, _)| s != TS);
    (th, me)
}

impl StepType for ExplicitCoupled {
    fn name(&self) -> &'static str {
        "DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (p.name == PROC).then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if !kw.has("EXPLICIT") {
            return kw.err("only *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT is supported");
        }
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.nlgeom {
            return fail("*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT with NLGEOM=YES is not supported yet (set NLGEOM=NO on *STEP)");
        }
        if !prep.interactions.is_empty() {
            return fail("*DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT with contact, constraints or point elements is not supported yet");
        }
        for (e, el) in model.elements.iter().enumerate() {
            if !["CPE3T", "CPS3T", "C3D4T"].contains(&el.etype.name()) {
                return fail(format!("{}: *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT supports the elements CPE3T, CPS3T and C3D4T, not {}", model.elem_name(e), el.etype.name()));
            }
        }
        let v = kw.first_floats().unwrap_or_default();
        let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        let fixed = kw.has("DIRECT USER CONTROL") || kw.has("FIXED TIME INCREMENTATION");
        let user_dt = v.first().copied().filter(|&x| x > 0.0);
        let scale = match kw.param("SCALE FACTOR") {
            Some(s) => parse_f64(s).ok_or_else(|| FemError(format!("{}: SCALE FACTOR={s} is not a number", kw.location())))?,
            None => 1.0,
        };
        ensure_initial(model, ctx.state)?;
        let loads = heat_loads(ctx.step, ctx.loads, false);
        let (th_loads, me_loads) = split_loads(&loads);
        let mut heat = HeatSystem::new(prep, th_loads, ctx.state, false, ctx.solver, period)?;
        let nn = model.nodes.len();
        let ne = model.elements.len();
        // the mechanical DOFs (the temperatures count as free here)
        let (dm, _) = assembly::dofmap(prep, &me_loads.constrained(), ctx.solver)?;
        let lev = LoadEval::new(model, &me_loads, false, period, ctx.state.time)?;
        let time0 = ctx.state.time;
        // diagonal mass and heat capacity
        let mut mass = vec![[0.0; SLOTS]; nn];
        let mut cap = vec![0.0; nn];
        let mut dt_lim = f64::INFINITY;
        for (e, el) in model.elements.iter().enumerate() {
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            let ectx = prep.ctx(e, false, time0, 0.0);
            let me = el.etype.mass(&ectx, MassKind::Lumped).map_err(wrap)?;
            let nd = prep.edofs[e].len();
            for (a, &(n, s)) in prep.edofs[e].iter().enumerate() {
                let row: f64 = (0..nd).map(|b| me[a * nd + b]).sum();
                mass[n][s] += row;
            }
            let he = crate::elements::heat::lookup(el.etype.name()).ok_or_else(|| FemError(format!("{}: thermal part is missing", model.elem_name(e))))?;
            let sec = &model.sections[el.section.ok_or_else(|| FemError(format!("{}: has no *SOLID SECTION", model.elem_name(e))))?];
            let mi = sec.material_index.ok_or_else(|| FemError(format!("{}: the section has no material", model.elem_name(e))))?;
            let props = HeatProps::from_def(&model.materials[mi])?;
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let thick = sec.solid_thickness();
            let cips = he.cap_ips(&x, thick).map_err(wrap)?;
            let t: Vec<f64> = el.nodes.iter().map(|&n| ctx.state.u[n][TS]).collect();
            let (_, kc) = he.capacity(&cips, &props, &t, &t, 1.0, true).map_err(wrap)?;
            let kc = kc.unwrap_or_default();
            let k = el.nodes.len();
            for a in 0..k {
                cap[el.nodes[a]] += (0..k).map(|b| kc.get(a * k + b).copied().unwrap_or(0.0)).sum::<f64>();
            }
            // stability limits
            let vol: f64 = el.etype.volume_weights(&ectx).map_err(wrap)?.iter().sum();
            let le_len = if el.etype.dim() == 3 { vol.cbrt() } else { (vol / thick.max(1e-300)).sqrt() };
            if let (Some(mat), true) = (ectx.material, ectx.density > 0.0) {
                if let Some((ey, nu)) = mat.elastic_moduli() {
                    let m2 = ey * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
                    dt_lim = dt_lim.min(le_len / (m2 / ectx.density).sqrt());
                }
                let kcond = props.cond.as_ref().map(|c| c.1.value(t.iter().sum::<f64>() / k as f64)).unwrap_or(0.0);
                let ccap = props.specific_heat.as_ref().map(|c| c.value(t.iter().sum::<f64>() / k as f64)).unwrap_or(0.0);
                if kcond > 0.0 && ccap > 0.0 {
                    let alpha = kcond / (ectx.density * ccap);
                    dt_lim = dt_lim.min(le_len * le_len / (2.0 * alpha));
                }
            }
        }
        let (ninc, dt) = match (fixed, user_dt) {
            (true, Some(d)) => {
                let n = (period / d - 1e-9).ceil().max(1.0) as usize;
                (n, period / n as f64)
            }
            (true, None) => return kw.err("DIRECT USER CONTROL requires a time increment on the data line"),
            _ => {
                if !dt_lim.is_finite() {
                    return fail("cannot estimate a stable time increment (missing *DENSITY, *ELASTIC or *SPECIFIC HEAT?)");
                }
                let d = dt_lim * scale;
                let n = (period / d - 1e-9).ceil().max(1.0) as usize;
                (n, period / n as f64)
            }
        };
        if ctx.opts.verbose {
            eprintln!("  {ninc} increments with Δt = {dt:.4e} (stability limit {dt_lim:.4e})");
        }
        // state
        let mut u = ctx.state.u.clone();
        let mut vel = ctx.state.vectors.get("V").cloned().unwrap_or_else(|| vec![[0.0; SLOTS]; nn]);
        let apply_bc = |u: &mut [NodeVec], t: f64| {
            for (n, s, val) in lev.bc_at(t) {
                if dm.constrained[n] & (1 << s) != 0 {
                    u[n][s] = val;
                }
            }
        };
        let apply_tbc = |heat: &HeatSystem, u: &mut [NodeVec], t: f64| {
            for (n, s, val) in heat.bc_now(t / period) {
                if heat.dm.constrained[n] & (1 << s) != 0 {
                    u[n][s] = val;
                }
            }
        };
        // the thermal strain is computed from the temperature before the step's
        // boundary conditions (the initial temperature θ_I)
        let mut temp_prev: Vec<f64> = u.iter().map(|x| x[TS]).collect();
        apply_bc(&mut u, 0.0);
        apply_tbc(&heat, &mut u, 0.0);
        let mut state = ctx.state.elem_state.clone();
        // internal force with the nodal temperatures (the thermal strain) and the heat flux residual
        let mech_force = |u: &[NodeVec], du: &[NodeVec], told: &[f64], st: &[Vec<f64>], t: f64, h: f64| -> Result<(Vec<NodeVec>, Vec<Vec<f64>>)> {
            let mut fint = vec![[0.0; SLOTS]; nn];
            let mut sn_all = Vec::with_capacity(ne);
            for e in 0..ne {
                let el = &model.elements[e];
                let mut ectx = prep.ctx(e, false, time0 + t, h);
                ectx.temp = el.nodes.iter().map(|&n| u[n][TS]).collect();
                ectx.temp_old = el.nodes.iter().map(|&n| told[n]).collect();
                let ue = prep.gather(e, u);
                let due = prep.gather(e, du);
                let mut sn = Vec::new();
                let r = el.etype.internal_force(&ectx, &ue, &due, &st[e], &mut sn, false).map_err(|err| FemError(format!("{}: {err}", model.elem_name(e))))?;
                for (a, &(n, s)) in prep.edofs[e].iter().enumerate() {
                    if s != TS {
                        fint[n][s] += r.fint[a];
                    }
                }
                sn_all.push(sn);
            }
            Ok((fint, sn_all))
        };
        let thermal_rate = |heat: &mut HeatSystem, u: &[NodeVec], t: f64| -> Result<Vec<f64>> {
            heat.u_committed = u.to_vec();
            // fluxes with step amplitude act fully already at t = 0
            let tt = (t / period).max(1e-12);
            heat.begin_increment(tt, tt)?;
            heat.u = u.to_vec();
            heat.evaluate(false)?;
            Ok((0..nn).map(|n| if cap[n] > 0.0 { (heat.fext[n][TS] - heat.fint[n][TS]) / cap[n] } else { 0.0 }).collect())
        };
        let accel = |fint: &[NodeVec], t: f64| -> Result<Vec<NodeVec>> {
            let z = vec![[0.0; SLOTS]; nn];
            let fext = assembly::external(prep, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, time0 + t)?;
            let mut a = vec![[0.0; SLOTS]; nn];
            for n in 0..nn {
                for s in 0..SLOTS {
                    if s != TS && mass[n][s] > 0.0 && dm.eq[n][s] != NONE {
                        a[n][s] = (fext[n][s] - fint[n][s]) / mass[n][s];
                    }
                }
            }
            Ok(a)
        };
        let zero = vec![[0.0; SLOTS]; nn];
        // at the start of the step the initial temperature applies to the mechanics (frame 0
        // is stress-free); the temperature change is applied in the first increment
        let mut u_start = u.clone();
        for n in 0..nn {
            u_start[n][TS] = temp_prev[n];
        }
        let (mut fint, sn0) = mech_force(&u_start, &zero, &temp_prev, &state, 0.0, 0.0)?;
        state = sn0;
        let mut acc = accel(&fint, 0.0)?;
        let mut trate = thermal_rate(&mut heat, &u, 0.0)?;
        let sched = output_schedule(ctx.step, period, Schedule::Every(1))?;
        let dim = model.dim();
        let mk = |heat: &mut HeatSystem, u: &[NodeVec], v: &[NodeVec], a: &[NodeVec], fint: &[NodeVec], st: &[Vec<f64>], t: f64, inc: usize| -> Result<FrameData> {
            let z = vec![[0.0; SLOTS]; nn];
            let fext = assembly::external(prep, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, time0 + t)?;
            // reaction = internal force − external + mass·acceleration in fixed DOFs
            let mut fi = fint.to_vec();
            for n in 0..nn {
                for s in 0..SLOTS {
                    if s != TS {
                        fi[n][s] += mass[n][s] * a[n][s];
                    }
                }
            }
            let desc = format!("Increment {inc}: Step Time = {t:.4}");
            let mut fr = assembly::frame(prep, &dm, u, &fi, &fext, st, false, time0 + t, t, desc.clone())?;
            heat.u = u.to_vec();
            heat.u_committed = u.to_vec();
            heat.evaluate(false)?;
            let h = heat.frame(t, desc)?;
            for n in 0..nn {
                fr.u[n][TS] = h.u[n][TS];
                fr.rf[n][TS] = h.rf[n][TS];
                let bit = 1u16 << TS;
                fr.active[n] = (fr.active[n] & !bit) | (h.active[n] & bit);
                fr.constrained[n] = (fr.constrained[n] & !bit) | (h.constrained[n] & bit);
            }
            // HFL: linear simplex elements have a constant gradient and one
            // integration point (CPE3T, CPS3T, C3D4T)
            for (x, y) in fr.elem.iter_mut().zip(h.elem) {
                x.extend(y.into_iter().filter(|f| f.name == "HFL").map(|mut f| {
                    f.points.truncate(1);
                    if let Some(p) = f.points.first_mut() {
                        p.0 = 1;
                    }
                    f
                }));
            }
            add_va(&mut fr, dim, v, a);
            Ok(fr)
        };
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            let mut f0 = mk(&mut heat, &u, &vel, &acc, &fint, &state, 0.0, 0)?;
            f0.rf.iter_mut().for_each(|r| *r = [0.0; SLOTS]);
            frames.push(f0);
        }
        let mut vh = vel.clone();
        let mut t = 0.0;
        for inc in 1..=ninc {
            let h = dt;
            let hm = if inc == 1 { 0.5 * h } else { h };
            let u_old = u.clone();
            for n in 0..nn {
                for s in 0..SLOTS {
                    if s == TS {
                        u[n][s] += h * trate[n];
                    } else if dm.eq[n][s] != NONE {
                        vh[n][s] += hm * acc[n][s];
                        u[n][s] += h * vh[n][s];
                    }
                }
            }
            t = if inc == ninc { period } else { t + h };
            apply_bc(&mut u, t);
            apply_tbc(&heat, &mut u, t);
            let du: Vec<NodeVec> = u.iter().zip(&u_old).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
            if inc > 1 {
                temp_prev = u_old.iter().map(|x| x[TS]).collect();
            }
            let (f1, sn) = mech_force(&u, &du, &temp_prev, &state, t, h)?;
            fint = f1;
            state = sn;
            acc = accel(&fint, t)?;
            trate = thermal_rate(&mut heat, &u, t)?;
            for n in 0..nn {
                for s in 0..SLOTS {
                    vel[n][s] = if s != TS && dm.eq[n][s] != NONE { vh[n][s] + 0.5 * h * acc[n][s] } else { 0.0 };
                }
            }
            // prescribed displacements: the half-step velocity from the change
            for n in 0..nn {
                for s in 0..SLOTS {
                    if s != TS && dm.constrained[n] & (1 << s) != 0 {
                        vh[n][s] = du[n][s] / h;
                        vel[n][s] = vh[n][s];
                        acc[n][s] = 0.0;
                    }
                }
            }
            let write = match sched {
                Schedule::Every(k) => inc % k == 0,
                Schedule::Interval(d) => ((t / d) - (t / d).round()).abs() < 1e-6 * (period / d).max(1.0),
            };
            if (all && write) || inc == ninc {
                frames.push(mk(&mut heat, &u, &vel, &acc, &fint, &state, t, inc)?);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = ninc;
        ctx.state.u = u;
        ctx.state.elem_state = state;
        ctx.state.vectors.insert("V".into(), vel);
        ctx.state.vectors.insert("A".into(), acc);
        ctx.state.time = time0 + period;
        Ok(())
    }
}
