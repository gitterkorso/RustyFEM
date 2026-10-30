//! Modal dynamics steps:
//!
//! - `*MODAL DYNAMIC`: modal superposition in time. Each modal equation
//!   q̈ + 2ζω q̇ + ω² q = φᵀP(t) − Γ ü_g(t) is integrated exactly under the assumption
//!   that the load varies linearly over the increment (Clough & Penzien,
//!   "Dynamics of Structures", ch. 7). Fixed
//!   increments from the data line (Δt, step time). Output U, V, A = Σ φ q, Σ φ q̇,
//!   Σ φ q̈ (relative to the base), TU, TV, TA (total) and S.
//! - `*STEADY STATE DYNAMICS` (modal and DIRECT), see `steady_state.rs`.
//! - `*RESPONSE SPECTRUM`, see `modal_spectrum.rs`.
//!
//! The modes are exactly the same as in the preceding *FREQUENCY step (consistent
//! mass and internal DOFs where *FREQUENCY has them), see `modal_basis.rs`
//! for the modal basis, *SELECT EIGENMODES, *MODAL DAMPING and *BASE MOTION.
//!
//! Structural modal damping in the time domain is replaced by an
//! equivalent viscous damping ζ = s/2 (the damping force s ω q̇ gives the same
//! energy loss per cycle at resonance); checked in
//! tests/cases/modal.

#[path = "modal_basis.rs"]
pub mod basis;
#[path = "modal_spectrum.rs"]
pub mod spectrum;
#[path = "steady_state.rs"]
pub mod steady_state;

use super::api::{StepCtx, StepType};
use super::dynamic::amplitude::{amplitudes, Amplitude, LoadEval};
use super::dynamic::{add_va, default_ramp, output_schedule, Schedule};
use crate::fem::assembly::{self, DofMap};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use basis::{base_motions, modal_damping, BaseKind, BaseMotion, ModalBasis};
use std::collections::BTreeMap;

pub struct ModalDynamic;

pub static STEP_TYPES: &[&dyn StepType] = &[&ModalDynamic, &steady_state::SteadyState, &spectrum::ResponseSpectrum];

/// A frame with the displacement `u` (without reactions) and element output.
pub fn plain_frame(ctx: &StepCtx, dm: &DofMap, u: &[NodeVec], value: f64, mode: i64, description: String) -> Result<FrameData> {
    let nn = u.len();
    let z = vec![[0.0; SLOTS]; nn];
    // stress state for the displacement u (linear from the zero state)
    let st0 = ctx.prep.init_states()?;
    let int = assembly::internal(ctx.prep, dm, u, u, &st0, false, ctx.state.time, 0.0, None)?;
    let mut fr = assembly::frame(ctx.prep, dm, u, &z, &z, &int.state_new, false, ctx.state.time, value, description)?;
    fr.mode = mode;
    fr.rf.clear();
    Ok(fr)
}

/// The model has rotational DOFs (beams, shells).
pub fn has_rotations(ctx: &StepCtx) -> bool {
    ctx.prep.active.iter().any(|a| a & 0b111000 != 0)
}

/// Nodal fields with standard names: the translations as `name` (components name1..),
/// the rotations as `rname` (2D: rname3 as a separate field).
pub fn add_node_field(fr: &mut FrameData, dim: usize, rot: bool, name: &str, rname: &str, v: &[NodeVec]) {
    let labels = |p: &str| (1..=dim).map(|i| format!("{p}{i}")).collect::<Vec<_>>();
    fr.node_extra.push((name.into(), labels(name), v.iter().map(|x| x[..dim].to_vec()).collect()));
    if rot {
        if dim == 2 {
            let n = format!("{rname}3");
            fr.node_extra.push((n.clone(), vec![n], v.iter().map(|x| vec![x[5]]).collect()));
        } else {
            fr.node_extra.push((rname.into(), (1..=3).map(|i| format!("{rname}{i}")).collect(), v.iter().map(|x| x[3..6].to_vec()).collect()));
        }
    }
}

/// V and A (with the rotations VR/AR for beams and shells).
pub fn add_velocity_acceleration(fr: &mut FrameData, dim: usize, rot: bool, v: &[NodeVec], a: &[NodeVec]) {
    add_va(fr, dim, v, a);
    if rot {
        let mut tmp = FrameData::default();
        add_node_field(&mut tmp, dim, true, "_", "VR", v);
        add_node_field(&mut tmp, dim, true, "_", "AR", a);
        fr.node_extra.extend(tmp.node_extra.into_iter().filter(|f| f.0 != "_"));
    }
}

/// Amplitude value (or 1 without amplitude) for the base motion.
fn amp_value(amps: &BTreeMap<String, Amplitude>, name: &Option<String>, t: f64, total: f64) -> Result<f64> {
    match name {
        None => Ok(1.0),
        Some(n) => match amps.get(n) {
            Some(a) => {
                let v = a.value(t, total);
                Ok(v)
            }
            None => fail(format!("amplitude {n} is not defined (*AMPLITUDE, NAME={n})")),
        },
    }
}

/// Exact solution of q̈ + 2ζω q̇ + ω² q = f0 + (f1 − f0) τ/h over an increment.
/// Returns (q, q̇, q̈) at τ = h.
pub fn sdof_step(w: f64, zeta: f64, h: f64, q0: f64, v0: f64, f0: f64, f1: f64) -> (f64, f64, f64) {
    let s = (f1 - f0) / h;
    let w2 = w * w;
    // particular solution qp = (f0 + s τ)/ω² − 2ζ s/ω³
    let qp0 = f0 / w2 - 2.0 * zeta * s / (w2 * w);
    let qp1 = (f0 + s * h) / w2 - 2.0 * zeta * s / (w2 * w);
    let vp = s / w2;
    let a = q0 - qp0;
    let zeta = if (zeta - 1.0).abs() < 1e-9 { 1.0 - 1e-9 } else { zeta };
    let e = (-zeta * w * h).exp();
    let (q, v) = if zeta < 1.0 {
        let wd = w * (1.0 - zeta * zeta).sqrt();
        let b = (v0 - vp + zeta * w * a) / wd;
        let (sn, cs) = (wd * h).sin_cos();
        let qh = e * (a * cs + b * sn);
        let vh = e * ((-zeta * w) * (a * cs + b * sn) + wd * (-a * sn + b * cs));
        (qh + qp1, vh + vp)
    } else {
        let wd = w * (zeta * zeta - 1.0).sqrt();
        let b = (v0 - vp + zeta * w * a) / wd;
        let (sn, cs) = ((wd * h).sinh(), (wd * h).cosh());
        let qh = e * (a * cs + b * sn);
        let vh = e * ((-zeta * w) * (a * cs + b * sn) + wd * (a * sn + b * cs));
        (qh + qp1, vh + vp)
    };
    let acc = f1 - 2.0 * zeta * w * v - w2 * q;
    (q, v, acc)
}

/// Base acceleration ü_g(t) per direction (sum of the step's *BASE MOTION).
fn base_accel(bms: &[BaseMotion], amps: &BTreeMap<String, Amplitude>, t: f64, total: f64) -> Result<[f64; 3]> {
    let mut a = [0.0; 3];
    for b in bms {
        a[b.dir] += b.scale * amp_value(amps, &b.amplitude, t, total)?;
    }
    Ok(a)
}

impl StepType for ModalDynamic {
    fn name(&self) -> &'static str {
        "MODAL DYNAMIC"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "MODAL DYNAMIC").then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if !ctx.step.bcs.is_empty() {
            return fail("*BOUNDARY in *MODAL DYNAMIC is not supported – use *BASE MOTION");
        }
        if kw.flag("CONTINUE", false)? {
            return kw.err("CONTINUE=YES is not supported yet");
        }
        let v = kw.first_floats()?;
        let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        let dt0 = v.first().copied().filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: time increment is missing", kw.location())))?;
        let b = ModalBasis::from_frequency_step(ctx)?;
        let damp = modal_damping(ctx.step, &b)?;
        let omega = b.omega.clone();
        // structural damping as equivalent viscous (see the module description)
        let zeta: Vec<f64> = damp.zeta.iter().zip(&damp.structural).map(|(z, s)| z + s / 2.0).collect();
        let prep = ctx.prep;
        let model = prep.model;
        let nn = model.nodes.len();
        let bms = base_motions(ctx.step)?;
        if bms.iter().any(|m| m.kind != BaseKind::Acceleration) {
            return fail("*BASE MOTION in *MODAL DYNAMIC: only TYPE=ACCELERATION is supported");
        }
        let amps = amplitudes(model)?;
        let gamma: Vec<Vec<f64>> = (0..3).map(|d| if bms.iter().any(|m| m.dir == d) { b.participation(&model.nodes, d) } else { Vec::new() }).collect();
        let time0 = ctx.state.time;
        let mut lev = LoadEval::new(model, ctx.loads, default_ramp(ctx.step), period, time0)?;
        // the load at t = 0: amplitudes exactly at t = 0 (ü = 0 in frame 0
        // for an amplitude starting at zero), step loads fully from the start
        lev.step_full_at_start = true;
        let lev = lev;
        let dm = &b.dm;
        let force = |t: f64| -> Result<(Vec<f64>, [f64; 3])> {
            let z = vec![[0.0; SLOTS]; nn];
            let f = assembly::external(prep, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, time0 + t)?;
            let mut fm = b.modal_force(&f);
            let ag = base_accel(&bms, &amps, t, time0 + t)?;
            for d in 0..3 {
                if ag[d] != 0.0 {
                    fm.iter_mut().zip(&gamma[d]).for_each(|(x, g)| *x -= g * ag[d]);
                }
            }
            Ok((fm, ag))
        };
        let nm = b.len();
        let mut q = vec![0.0; nm];
        let mut qd = vec![0.0; nm];
        let (mut f_prev, mut ag_prev) = force(0.0)?;
        let mut qdd: Vec<f64> = (0..nm).map(|i| f_prev[i] - omega[i] * omega[i] * q[i]).collect();
        // displacement and velocity of the base (exact for linear ü_g over the increment)
        let mut ug = [0.0; 3];
        let mut vg = [0.0; 3];
        let sched = output_schedule(ctx.step, period, Schedule::Every(1))?;
        let all = ctx.opts.all_frames;
        let dim = model.dim();
        let rot = has_rotations(ctx);
        let with_base = !bms.is_empty();
        let mk = |q: &[f64], qd: &[f64], qdd: &[f64], ug: &[f64; 3], vg: &[f64; 3], ag: &[f64; 3], t: f64, inc: usize| -> Result<FrameData> {
            let u = b.superpose(q);
            let mut fr = plain_frame(ctx, dm, &u, t, -1, format!("Increment {inc}: Step Time = {t:.4}"))?;
            let (vv, aa) = (b.superpose(qd), b.superpose(qdd));
            add_velocity_acceleration(&mut fr, dim, rot, &vv, &aa);
            if with_base {
                for (name, src, g) in [("TU", &u, ug), ("TV", &vv, vg), ("TA", &aa, ag)] {
                    let mut tot = src.clone();
                    for d in 0..dim {
                        basis::add_rigid(&mut tot, d, g[d]);
                    }
                    add_node_field(&mut fr, dim, false, name, "", &tot);
                }
            }
            Ok(fr)
        };
        let mut frames = Vec::new();
        if all {
            frames.push(mk(&q, &qd, &qdd, &ug, &vg, &ag_prev, 0.0, 0)?);
        }
        let ninc = (period / dt0 - 1e-9).ceil().max(1.0) as usize;
        let mut t = 0.0;
        for inc in 1..=ninc {
            let h = if inc == ninc { period - t } else { dt0 };
            let t1 = t + h;
            let (f1, ag1) = force(t1)?;
            for i in 0..nm {
                if omega[i] <= 0.0 {
                    // rigid-body mode: q̈ = f
                    let a0 = f_prev[i];
                    let s = (f1[i] - f_prev[i]) / h;
                    q[i] += qd[i] * h + a0 * h * h / 2.0 + s * h * h * h / 6.0;
                    qd[i] += a0 * h + s * h * h / 2.0;
                    qdd[i] = f1[i];
                    continue;
                }
                let (x, y, z) = sdof_step(omega[i], zeta[i], h, q[i], qd[i], f_prev[i], f1[i]);
                q[i] = x;
                qd[i] = y;
                qdd[i] = z;
            }
            for d in 0..3 {
                let s = (ag1[d] - ag_prev[d]) / h;
                ug[d] += vg[d] * h + ag_prev[d] * h * h / 2.0 + s * h * h * h / 6.0;
                vg[d] += ag_prev[d] * h + s * h * h / 2.0;
            }
            f_prev = f1;
            ag_prev = ag1;
            t = t1;
            let write = match sched {
                Schedule::Every(k) => inc % k == 0,
                Schedule::Interval(d) => ((t / d) - (t / d).round()).abs() < 1e-6,
            };
            if (all && write) || inc == ninc {
                frames.push(mk(&q, &qd, &qdd, &ug, &vg, &ag_prev, t, inc)?);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = ninc;
        // modal steps are linear perturbations: the state does not change
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Undamped SDOF with constant load: q = f/ω² (1 − cos ωt).
    #[test]
    fn sdof_exact_constant_load() {
        let (w, f) = (10.0, 3.0);
        let (mut q, mut v) = (0.0, 0.0);
        let h = 0.013;
        for _ in 0..50 {
            let r = sdof_step(w, 0.0, h, q, v, f, f);
            q = r.0;
            v = r.1;
        }
        let t = 50.0 * h;
        assert!((q - f / (w * w) * (1.0 - (w * t).cos())).abs() < 1e-12);
        assert!((v - f / w * (w * t).sin()).abs() < 1e-12);
    }

    /// Damped free vibration: q = e^{−ζωt}(cos ωd t + ζω/ωd sin ωd t).
    #[test]
    fn sdof_damped_free_vibration() {
        let (w, z) = (7.0, 0.05);
        let wd = w * (1.0f64 - z * z).sqrt();
        let (mut q, mut v) = (1.0, 0.0);
        for _ in 0..40 {
            let r = sdof_step(w, z, 0.01, q, v, 0.0, 0.0);
            q = r.0;
            v = r.1;
        }
        let t = 0.4;
        let ex = (-z * w * t).exp() * ((wd * t).cos() + z * w / wd * (wd * t).sin());
        assert!((q - ex).abs() < 1e-12);
    }
}
