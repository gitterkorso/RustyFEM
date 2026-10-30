//! Subspace-projektion på egenmoderna från det föregående *FREQUENCY-steget:
//!
//! - `*STEADY STATE DYNAMICS, SUBSPACE PROJECTION=ALL FREQUENCIES |
//!   EIGENFREQUENCY | PROPERTY CHANGE | CONSTANT`: de direkta
//!   ekvationerna (K − Ω²M + iΩC + iS) u = F projiceras på modbasen,
//!   (diag(ωᵢ²) − Ω² I + iΩ Cᵣ + i Sᵣ) q = Φᵀ F, och löses tätt per frekvens.
//!   Till skillnad från den modala SSD ingår all dämpning som kopplade
//!   matriser: dämpare (*DASHPOT), materialets *DAMPING ALPHA/BETA och
//!   STRUCTURAL (samma Cᵣ och Sᵣ som *COMPLEX FREQUENCY), plus *MODAL DAMPING
//!   (diagonalt). Egenskaperna är frekvensoberoende, så alla
//!   projektionsalternativ ger samma resultat.
//! - `*DYNAMIC, SUBSPACE`: explicit centraldifferens i modkoordinaterna
//!   (se `DynamicSubspace`).
//!
//! Frekvenspunkter och utdata som modal *STEADY STATE DYNAMICS (U, U_IM,
//! elementfält och deras _IM-delar).

use super::api::{StepCtx, StepType};
use super::complex_frequency::dense::{self, Cx};
use super::complex_frequency::{complex_frame, projected_damping};
use super::dynamic::amplitude::LoadEval;
use super::dynamic::{default_ramp, output_schedule, Schedule};
use super::modal::basis::{modal_damping, ModalBasis};
use super::modal::steady_state::frequency_points;
use super::modal::{add_velocity_acceleration, has_rotations, plain_frame};
use crate::fem::assembly::{self, NONE};
use crate::fem::dof::{slot_of, SLOTS};
use crate::fem::model::{parse_f64, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};

pub struct SsdSubspace;
pub struct DynamicSubspace;

pub static STEP_TYPES: &[&dyn StepType] = &[&SsdSubspace, &DynamicSubspace];

impl StepType for SsdSubspace {
    fn name(&self) -> &'static str {
        "STEADY STATE DYNAMICS, SUBSPACE PROJECTION"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (p.name == "STEADY STATE DYNAMICS" && p.has("SUBSPACE PROJECTION")).then_some(1)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        match kw.param_upper("SUBSPACE PROJECTION").as_deref() {
            Some("") | Some("ALL FREQUENCIES") | Some("EIGENFREQUENCY") | Some("PROPERTY CHANGE") | Some("CONSTANT") => {}
            Some(o) => return kw.err(format!("SUBSPACE PROJECTION={o} stöds inte")),
            None => {}
        }
        if !ctx.step.bcs.is_empty() {
            return fail("*BOUNDARY i *STEADY STATE DYNAMICS stöds inte – använd *BASE MOTION");
        }
        if ctx.step.extra.iter().any(|k| k.name == "BASE MOTION") {
            return fail("*BASE MOTION i *STEADY STATE DYNAMICS, SUBSPACE PROJECTION stöds inte ännu");
        }
        let prep = ctx.prep;
        let model = prep.model;
        let nn = model.nodes.len();
        let b = ModalBasis::from_frequency_step(ctx)?;
        let n = b.len();
        let md = modal_damping(ctx.step, &b)?;
        let (cr, sr) = projected_damping(ctx, &b)?;
        let tp = 2.0 * std::f64::consts::PI;
        let pts = frequency_points(kw, &b.omega.iter().map(|w| w / tp).collect::<Vec<_>>())?;
        let mut lev = LoadEval::new(model, ctx.loads, false, 1.0, ctx.state.time)?;
        lev.step_full_at_start = true;
        let z = vec![[0.0; SLOTS]; nn];
        let mut frames = vec![complex_frame(ctx, &b.dm, &z, &z, 0.0, -1, "Increment      0: Base State".into())?];
        for (k, &f) in pts.iter().enumerate() {
            let w = tp * f;
            let fext = assembly::external(prep, &lev.cloads_at(f), &lev.dloads_at(f), &z, false, ctx.state.time)?;
            let fm = b.modal_force(&fext);
            let mut a = vec![Cx::ZERO; n * n];
            for i in 0..n {
                for j in 0..n {
                    let mut re = 0.0;
                    let mut im = w * cr[i * n + j] + sr[i * n + j];
                    if i == j {
                        let om = b.omega[i];
                        re += b.lambda[i] - w * w;
                        im += 2.0 * md.zeta[i] * om * w + md.structural[i] * om * om;
                    }
                    a[i * n + j] = Cx::new(re, im);
                }
            }
            let rhs: Vec<Cx> = fm.iter().map(|&x| Cx::new(x, 0.0)).collect();
            let q = dense::solve(&a, n, &rhs);
            let re = b.superpose(&q.iter().map(|c| c.re).collect::<Vec<_>>());
            let im = b.superpose(&q.iter().map(|c| c.im).collect::<Vec<_>>());
            frames.push(complex_frame(ctx, &b.dm, &re, &im, f, -1, format!("Increment {}: Frequency = {f:.4}", k + 1))?);
        }
        if !ctx.opts.all_frames {
            let last = frames.pop();
            frames.clear();
            frames.extend(last);
        }
        ctx.out.increments = frames.len();
        ctx.out.frames.extend(frames);
        Ok(())
    }
}

impl StepType for DynamicSubspace {
    fn name(&self) -> &'static str {
        "DYNAMIC, SUBSPACE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        let p = step.procedure.as_ref()?;
        (p.name == "DYNAMIC" && p.has("SUBSPACE")).then_some(2)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        run_dynamic_subspace(ctx)
    }
}

/// *DYNAMIC, SUBSPACE: explicit centraldifferens i modkoordinaterna,
/// q̈ = Φᵀ F(t) − diag(ω²) q − Cᵣ q̇ med v_{n+½} = v_{n−½} + ½(Δtₙ₊₁ + Δtₙ) aₙ,
/// qₙ₊₁ = qₙ + Δt v_{n+½} och dämpningskraften med v_{n+½}. Cᵣ = Φᵀ C Φ med
/// dämpare (*DASHPOT) och materialets BETA (βK); ALPHA och STRUCTURAL
/// ignoreras. Fast
/// tidsinkrement = dataradens första värde; över
/// stabilitetsgränsen 2/ωmax ger det fel.
fn run_dynamic_subspace(ctx: &mut StepCtx) -> Result<()> {
    let kw = ctx.step.procedure.as_ref().unwrap();
    if !ctx.step.bcs.is_empty() {
        return fail("*BOUNDARY i *DYNAMIC, SUBSPACE stöds inte ännu");
    }
    if ctx.nlgeom {
        return fail("*DYNAMIC, SUBSPACE med NLGEOM=YES stöds inte ännu");
    }
    let v = kw.first_floats()?;
    let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
    let dt0 = v.first().copied().filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: tidsinkrement saknas", kw.location())))?;
    let prep = ctx.prep;
    let model = prep.model;
    let nn = model.nodes.len();
    let b = ModalBasis::from_frequency_step(ctx)?;
    let n = b.len();
    let wmax = b.omega.iter().cloned().fold(0.0, f64::max);
    if wmax * dt0 > 2.0 {
        return kw.err(format!("tidsinkrementet {dt0} överskrider stabilitetsgränsen 2/ωmax = {:.4e}", 2.0 / wmax));
    }
    let cr = subspace_damping(ctx, &b)?;
    let time0 = ctx.state.time;
    let lev = LoadEval::new(model, ctx.loads, default_ramp(ctx.step), period, time0)?;
    let force = |t: f64| -> Result<Vec<f64>> {
        let z = vec![[0.0; SLOTS]; nn];
        let f = assembly::external(prep, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, time0 + t)?;
        Ok(b.modal_force(&f))
    };
    let accel = |f: &[f64], q: &[f64], qd: &[f64]| -> Vec<f64> {
        (0..n).map(|i| f[i] - b.lambda[i] * q[i] - (0..n).map(|j| cr[i * n + j] * qd[j]).sum::<f64>()).collect()
    };
    let mut q = vec![0.0; n];
    let mut qd = vec![0.0; n];
    let mut qdd = accel(&force(0.0)?, &q, &qd);
    let sched = output_schedule(ctx.step, period, Schedule::Every(1))?;
    let all = ctx.opts.all_frames;
    let dim = model.dim();
    let rot = has_rotations(ctx);
    let u0 = ctx.state.u.clone();
    let total = |q: &[f64]| {
        let mut u = b.superpose(q);
        for (x, y) in u.iter_mut().zip(&u0) {
            for s in 0..SLOTS {
                x[s] += y[s];
            }
        }
        u
    };
    let mk = |q: &[f64], qd: &[f64], qdd: &[f64], t: f64, inc: usize| -> Result<FrameData> {
        let mut fr = plain_frame(ctx, &b.dm, &total(q), t, -1, format!("Increment {inc}: Step Time = {t:.4}"))?;
        add_velocity_acceleration(&mut fr, dim, rot, &b.superpose(qd), &b.superpose(qdd));
        Ok(fr)
    };
    let mut frames = Vec::new();
    if all {
        frames.push(mk(&q, &qd, &qdd, 0.0, 0)?);
    }
    let ninc = (period / dt0 - 1e-9).ceil().max(1.0) as usize;
    let mut t = 0.0;
    let mut h_prev = 0.0;
    // halvstegshastigheten v_{n−½}
    let mut vh = qd.clone();
    for inc in 1..=ninc {
        let h = if inc == ninc { period - t } else { dt0 };
        let hm = if inc == 1 { 0.5 * h } else { 0.5 * (h + h_prev) };
        for i in 0..n {
            vh[i] += hm * qdd[i];
            q[i] += h * vh[i];
        }
        t += h;
        qdd = accel(&force(t)?, &q, &vh);
        for i in 0..n {
            qd[i] = vh[i] + 0.5 * h * qdd[i];
        }
        h_prev = h;
        let write = match sched {
            Schedule::Every(k) => inc % k == 0,
            Schedule::Interval(d) => ((t / d) - (t / d).round()).abs() < 1e-6,
        };
        if (all && write) || inc == ninc {
            frames.push(mk(&q, &qd, &qdd, t, inc)?);
        }
    }
    // sluttillståndet ärvs av följande steg
    let u = total(&q);
    let st0 = ctx.state.elem_state.clone();
    let int = assembly::internal(prep, &b.dm, &u, &u, &st0, false, time0 + period, period, None)?;
    ctx.state.elem_state = int.state_new;
    ctx.state.u = u;
    ctx.state.time = time0 + period;
    ctx.out.frames.extend(frames);
    ctx.out.increments = ninc;
    Ok(())
}

/// Projicerad dämpning Φᵀ C Φ för *DYNAMIC, SUBSPACE: dämpare och βK.
fn subspace_damping(ctx: &StepCtx, b: &ModalBasis) -> Result<Vec<f64>> {
    let p = ctx.prep;
    let m = p.model;
    let n = b.len();
    let ns = b.all.n_solve();
    let mut cr = vec![0.0; n * n];
    let idx = |nd: usize, s: usize| -> Option<usize> { Some(b.all.idx[nd][s]).filter(|&g| g != NONE && g < ns) };
    for it in &p.interactions {
        for (na, da, nb, db, v) in it.damping() {
            let (Some(ga), Some(gb)) = (idx(na, slot_of(da)), idx(nb, slot_of(db))) else { continue };
            for i in 0..n {
                for j in 0..n {
                    cr[i * n + j] += b.x[i][ga] * v * b.x[j][gb];
                }
            }
        }
    }
    for (e, el) in m.elements.iter().enumerate() {
        let Some(mi) = el.section.and_then(|s| m.sections[s].material_index) else { continue };
        let Some(kw) = m.materials[mi].behavior("DAMPING") else { continue };
        let beta = match kw.param("BETA") {
            Some(v) => parse_f64(v).ok_or_else(|| FemError(format!("{}: BETA={v} är inget tal", kw.location())))?,
            None => 0.0,
        };
        if beta == 0.0 {
            continue;
        }
        let ctxe = p.ctx(e, false, 0.0, 0.0);
        let ke = el.etype.stiffness(&ctxe).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        let map: Vec<Option<usize>> = p.edofs[e].iter().map(|&(nd, s)| idx(nd, s)).collect();
        let nd = map.len();
        // φᵢ i elementets frihetsgrader
        let xe: Vec<Vec<f64>> = (0..n).map(|i| map.iter().map(|g| g.map_or(0.0, |g| b.x[i][g])).collect()).collect();
        let kx: Vec<Vec<f64>> = xe.iter().map(|x| (0..nd).map(|a| (0..nd).map(|c| ke[a * nd + c] * x[c]).sum()).collect()).collect();
        for i in 0..n {
            for j in 0..n {
                cr[i * n + j] += beta * xe[i].iter().zip(&kx[j]).map(|(a, c)| a * c).sum::<f64>();
            }
        }
    }
    Ok(cr)
}
