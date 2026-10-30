//! Modala dynamiksteg:
//!
//! - `*MODAL DYNAMIC`: modal superposition i tiden. Varje modekvation
//!   q̈ + 2ζω q̇ + ω² q = φᵀP(t) − Γ ü_g(t) integreras exakt under antagandet
//!   att lasten varierar linjärt över inkrementet (Clough & Penzien,
//!   "Dynamics of Structures", kap. 7). Fasta
//!   inkrement från dataraden (Δt, stegtid). Utdata U, V, A = Σ φ q, Σ φ q̇,
//!   Σ φ q̈ (relativt basen), TU, TV, TA (totalt) och S.
//! - `*STEADY STATE DYNAMICS` (modal och DIRECT), se `steady_state.rs`.
//! - `*RESPONSE SPECTRUM`, se `modal_spectrum.rs`.
//!
//! Moderna är exakt desamma som i det föregående *FREQUENCY-steget (konsistent
//! massa och inre frihetsgrader där *FREQUENCY har det), se `modal_basis.rs`
//! för modbasen, *SELECT EIGENMODES, *MODAL DAMPING och *BASE MOTION.
//!
//! Strukturell modal dämpning i tidsdomänen ersätts med en
//! ekvivalent viskös dämpning ζ = s/2 (dämpningskraften s ω q̇ ger samma
//! energiförlust per cykel vid resonans); kontrollerat i
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

/// En frame med förskjutningen `u` (utan reaktioner) och elementutdata.
pub fn plain_frame(ctx: &StepCtx, dm: &DofMap, u: &[NodeVec], value: f64, mode: i64, description: String) -> Result<FrameData> {
    let nn = u.len();
    let z = vec![[0.0; SLOTS]; nn];
    // spänningstillstånd för förskjutningen u (linjärt från nollstillståndet)
    let st0 = ctx.prep.init_states()?;
    let int = assembly::internal(ctx.prep, dm, u, u, &st0, false, ctx.state.time, 0.0, None)?;
    let mut fr = assembly::frame(ctx.prep, dm, u, &z, &z, &int.state_new, false, ctx.state.time, value, description)?;
    fr.mode = mode;
    fr.rf.clear();
    Ok(fr)
}

/// Modellen har rotationsfrihetsgrader (balkar, skal).
pub fn has_rotations(ctx: &StepCtx) -> bool {
    ctx.prep.active.iter().any(|a| a & 0b111000 != 0)
}

/// Nodfält med standardnamn: translationerna som `name` (komponenter name1..),
/// rotationerna som `rname` (2D: rname3 som eget fält).
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

/// V och A (med rotationerna VR/AR för balkar och skal).
pub fn add_velocity_acceleration(fr: &mut FrameData, dim: usize, rot: bool, v: &[NodeVec], a: &[NodeVec]) {
    add_va(fr, dim, v, a);
    if rot {
        let mut tmp = FrameData::default();
        add_node_field(&mut tmp, dim, true, "_", "VR", v);
        add_node_field(&mut tmp, dim, true, "_", "AR", a);
        fr.node_extra.extend(tmp.node_extra.into_iter().filter(|f| f.0 != "_"));
    }
}

/// Amplitudvärde (eller 1 utan amplitud) för basrörelsen.
fn amp_value(amps: &BTreeMap<String, Amplitude>, name: &Option<String>, t: f64, total: f64) -> Result<f64> {
    match name {
        None => Ok(1.0),
        Some(n) => match amps.get(n) {
            Some(a) => {
                let v = a.value(t, total);
                Ok(v)
            }
            None => fail(format!("amplituden {n} är inte definierad (*AMPLITUDE, NAME={n})")),
        },
    }
}

/// Exakt lösning av q̈ + 2ζω q̇ + ω² q = f0 + (f1 − f0) τ/h över ett inkrement.
/// Returnerar (q, q̇, q̈) vid τ = h.
pub fn sdof_step(w: f64, zeta: f64, h: f64, q0: f64, v0: f64, f0: f64, f1: f64) -> (f64, f64, f64) {
    let s = (f1 - f0) / h;
    let w2 = w * w;
    // partikulärlösning qp = (f0 + s τ)/ω² − 2ζ s/ω³
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

/// Basacceleration ü_g(t) per riktning (summa av stegets *BASE MOTION).
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
            return fail("*BOUNDARY i *MODAL DYNAMIC stöds inte – använd *BASE MOTION");
        }
        if kw.flag("CONTINUE", false)? {
            return kw.err("CONTINUE=YES stöds inte ännu");
        }
        let v = kw.first_floats()?;
        let period = v.get(1).copied().filter(|&x| x > 0.0).unwrap_or(1.0);
        let dt0 = v.first().copied().filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: tidsinkrement saknas", kw.location())))?;
        let b = ModalBasis::from_frequency_step(ctx)?;
        let damp = modal_damping(ctx.step, &b)?;
        let omega = b.omega.clone();
        // strukturell dämpning som ekvivalent viskös (se modulbeskrivningen)
        let zeta: Vec<f64> = damp.zeta.iter().zip(&damp.structural).map(|(z, s)| z + s / 2.0).collect();
        let prep = ctx.prep;
        let model = prep.model;
        let nn = model.nodes.len();
        let bms = base_motions(ctx.step)?;
        if bms.iter().any(|m| m.kind != BaseKind::Acceleration) {
            return fail("*BASE MOTION i *MODAL DYNAMIC: bara TYPE=ACCELERATION stöds");
        }
        let amps = amplitudes(model)?;
        let gamma: Vec<Vec<f64>> = (0..3).map(|d| if bms.iter().any(|m| m.dir == d) { b.participation(&model.nodes, d) } else { Vec::new() }).collect();
        let time0 = ctx.state.time;
        let mut lev = LoadEval::new(model, ctx.loads, default_ramp(ctx.step), period, time0)?;
        // lasten vid t = 0: amplituder exakt i t = 0 (ü = 0 i frame 0
        // för en amplitud som börjar i noll), steglaster fullt från start
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
        // basens förskjutning och hastighet (exakt för linjär ü_g över inkrementet)
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
                    // stelkroppsmod: q̈ = f
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
        // modala steg är linjära perturbationer: tillståndet ändras inte
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Odämpad SDOF med konstant last: q = f/ω² (1 − cos ωt).
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

    /// Dämpad fri svängning: q = e^{−ζωt}(cos ωd t + ζω/ωd sin ωd t).
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
