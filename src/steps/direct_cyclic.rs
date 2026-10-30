//! *DIRECT CYCLIC: the stabilized cyclic response directly, without
//! stepping through many cycles.
//!
//! The displacement over a cycle with period T is written as a Fourier series
//! u(t) = u₀ + Σₖ (uₖˢ sin kωt + uₖᶜ cos kωt), ω = 2π/T. Each iteration goes
//! through the cycle in N equal increments: at time tᵢ, u(tᵢ) is computed from the series,
//! the material is updated from the state at tᵢ₋₁ (the state is carried over from
//! one iteration to the next, so that the cycle becomes periodic) and the residual
//! R(tᵢ) = P(tᵢ) − I(tᵢ) is formed. The residual's Fourier coefficients
//! R₀ = (1/N) Σ R(tᵢ), Rₖᶜ = (2/N) Σ R(tᵢ) cos kωtᵢ, Rₖˢ = (2/N) Σ R(tᵢ) sin kωtᵢ
//! give the corrections cₖ = K₀⁻¹ Rₖ with the elastic stiffness K₀ (factorized
//! once), which are added to the coefficients.
//!
//! The data line: increment Δt, cycle time T, minimum and maximum
//! increment (fixed incrementation is used), initial number of Fourier terms
//! (11), maximum number (25), increase (5), maximum number of iterations (200).
//! The number of terms is limited so that 2n + 1 ≤ N. Convergence when
//! the largest residual coefficient ≤ 0.005 · the time-average force and the largest
//! correction ≤ 0.005 · the largest displacement coefficient in two consecutive
//! iterations; otherwise the number of terms is increased by the increase every tenth iteration when
//! possible.
//!
//! Output: the last iteration's cycle, frame 0 at the start of the cycle and one
//! frame per increment (U, RF, S, PEEQ, ...).
//! Low-cycle fatigue (FATIGUE) and NLGEOM are not supported.

use super::api::{StepCtx, StepType};
use super::dynamic::amplitude::LoadEval;
use super::frequency::{self, AllDofs};
use crate::fem::assembly::{self, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::StepDef;
use crate::model::{fail, FemError, Result};
use crate::solver::SystemMatrix;

pub struct DirectCyclic;

pub static STEP_TYPES: &[&dyn StepType] = &[&DirectCyclic];

/// Fourier coefficients per DOF in the free numbering:
/// c[0] = constant, c[2k−1] = sin k, c[2k] = cos k.
struct Series {
    c: Vec<Vec<f64>>,
}

impl Series {
    fn eval(&self, n: usize, w: f64, t: f64) -> Vec<f64> {
        let mut u = self.c[0].clone();
        for k in 1..=n {
            let (s, c) = ((k as f64) * w * t).sin_cos();
            for q in 0..u.len() {
                u[q] += self.c[2 * k - 1][q] * s + self.c[2 * k][q] * c;
            }
        }
        u
    }
}

impl StepType for DirectCyclic {
    fn name(&self) -> &'static str {
        "DIRECT CYCLIC"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "DIRECT CYCLIC").then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if kw.has("FATIGUE") && kw.param_upper("FATIGUE").as_deref() != Some("NO") {
            return kw.err("FATIGUE (low-cycle fatigue) is not supported yet");
        }
        if ctx.nlgeom {
            return kw.err("*DIRECT CYCLIC with NLGEOM=YES is not supported");
        }
        let v = kw.first_floats()?;
        let get = |i: usize, d: f64| v.get(i).copied().filter(|&x| x > 0.0).unwrap_or(d);
        let period = get(1, 1.0);
        let dt0 = v.first().copied().filter(|&x| x > 0.0).ok_or_else(|| FemError(format!("{}: time increment is missing", kw.location())))?;
        let (n_init, n_max, n_inc) = (get(4, 11.0).round() as usize, get(5, 25.0).round() as usize, get(6, 5.0).round() as usize);
        let max_iter = get(7, 200.0).round() as usize;
        let ninc = (period / dt0 - 1e-9).ceil().max(1.0) as usize;
        let dt = period / ninc as f64;
        let cap = (ninc.saturating_sub(1)) / 2;
        if cap == 0 {
            return kw.err("too few increments per cycle (at least 3)");
        }
        let mut nterms = n_init.min(cap).max(1);
        let n_max = n_max.min(cap).max(nterms);
        let prep = ctx.prep;
        let model = prep.model;
        let nn = model.nodes.len();
        if prep.interactions.iter().any(|it| !it.linear_constraints().map(|c| c.is_empty()).unwrap_or(true)) {
            return fail("*DIRECT CYCLIC with linear constraints (*EQUATION, *TIE, MPC) is not supported yet");
        }
        for (&(n, s), &(a, b)) in &ctx.loads.bc {
            if a != 0.0 || b != 0.0 || ctx.loads.amp_bc.contains_key(&(n, s)) {
                return fail("*DIRECT CYCLIC: only boundary conditions with value zero are supported");
            }
        }
        let (dm, _) = assembly::dofmap(prep, &ctx.loads.constrained(), ctx.solver)?;
        let all = AllDofs::new(prep, &dm);
        if all.n_int > 0 {
            return fail("*DIRECT CYCLIC with elements with internal DOFs (B23/B33) is not supported");
        }
        let neq = dm.n_eq;
        let kmat = frequency::assemble_stiffness(prep, &dm, &all, ctx.state, false)?.leading(neq);
        let Some(fac) = frequency::factor_checked(&kmat, &all, prep, ctx.solver)? else {
            return fail("*DIRECT CYCLIC: the stiffness matrix is singular (missing boundary conditions?)");
        };
        let time0 = ctx.state.time;
        let lev = LoadEval::new(model, ctx.loads, false, period, time0)?;
        let z = vec![[0.0; SLOTS]; nn];
        let fext_at = |t: f64| assembly::external(prep, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, time0 + t);
        let to_nodes = |x: &[f64], base: &[NodeVec]| -> Vec<NodeVec> {
            let mut u = base.to_vec();
            for n in 0..nn {
                for s in 0..SLOTS {
                    let q = dm.eq[n][s];
                    if q != NONE {
                        u[n][s] = x[q];
                    }
                }
            }
            u
        };
        let free = |f: &[NodeVec]| -> Vec<f64> {
            let mut r = vec![0.0; neq];
            for n in 0..nn {
                for s in 0..SLOTS {
                    let q = dm.eq[n][s];
                    if q != NONE {
                        r[q] = f[n][s];
                    }
                }
            }
            r
        };
        let w = 2.0 * std::f64::consts::PI / period;
        let mut ser = Series { c: vec![vec![0.0; neq]; 2 * n_max + 1] };
        let mut state = ctx.state.elem_state.clone();
        let base_u = ctx.state.u.clone();
        let fexts: Vec<Vec<NodeVec>> = (0..=ninc).map(|i| fext_at(i as f64 * dt)).collect::<Result<_>>()?;
        let mut last: Option<Vec<(Vec<NodeVec>, Vec<NodeVec>, Vec<Vec<f64>>)>> = None;
        let mut converged = false;
        let mut iterations = 0;
        for it in 0..max_iter {
            iterations = it + 1;
            let start_state = state.clone();
            let mut u_prev = to_nodes(&ser.eval(nterms, w, 0.0), &base_u);
            let mut cyc = Vec::with_capacity(ninc + 1);
            // frame 0: the start of the cycle (the state from the preceding iteration)
            let int0 = assembly::internal(prep, &dm, &u_prev, &z, &start_state, false, time0, 0.0, None)?;
            cyc.push((u_prev.clone(), int0.fint.clone(), start_state.clone()));
            let mut rc = vec![vec![0.0; neq]; 2 * nterms + 1];
            let mut favg = 0.0;
            for i in 1..=ninc {
                let t = i as f64 * dt;
                let u = to_nodes(&ser.eval(nterms, w, t), &base_u);
                let du: Vec<NodeVec> = u.iter().zip(&u_prev).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
                let int = assembly::internal(prep, &dm, &u, &du, &state, false, time0 + t, dt, None)?;
                let fe = &fexts[i];
                let (fi, pe) = (free(&int.fint), free(fe));
                let r: Vec<f64> = pe.iter().zip(&fi).map(|(a, b)| a - b).collect();
                favg += fi.iter().chain(pe.iter()).fold(0.0f64, |m, x| m.max(x.abs()));
                for q in 0..neq {
                    rc[0][q] += r[q] / ninc as f64;
                }
                for k in 1..=nterms {
                    let (s, c) = ((k as f64) * w * t).sin_cos();
                    for q in 0..neq {
                        rc[2 * k - 1][q] += 2.0 * r[q] * s / ninc as f64;
                        rc[2 * k][q] += 2.0 * r[q] * c / ninc as f64;
                    }
                }
                state = int.state_new;
                cyc.push((u.clone(), int.fint, state.clone()));
                u_prev = u;
            }
            favg /= ninc as f64;
            last = Some(cyc);
            // corrections
            let rmax = rc.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            let mut cmax = 0.0f64;
            for (k, r) in rc.iter().enumerate() {
                let mut x = r.clone();
                fac.solve(&mut x);
                for q in 0..neq {
                    ser.c[k][q] += x[q];
                }
                cmax = cmax.max(x.iter().fold(0.0f64, |m, y| m.max(y.abs())));
            }
            let umax = ser.c.iter().flatten().fold(0.0f64, |m, x| m.max(x.abs()));
            let tol = 5e-3;
            // convergence in two consecutive iterations (the last
            // iteration's cycle is written out)
            let ok = it > 0 && rmax <= tol * favg.max(1e-300) && cmax <= tol * umax.max(1e-300);
            if ok && converged {
                break;
            }
            converged = ok;
            // more terms if the iteration does not converge
            if it > 0 && it % 10 == 9 && nterms < n_max {
                nterms = (nterms + n_inc).min(n_max);
            }
        }
        if !converged && ctx.opts.verbose {
            eprintln!("warning: *DIRECT CYCLIC did not converge within {max_iter} iterations");
        }
        let cyc = last.unwrap();
        let all_frames = ctx.opts.all_frames;
        let n = cyc.len();
        for (i, (u, fint, st)) in cyc.into_iter().enumerate() {
            if !all_frames && i + 1 < n {
                continue;
            }
            let t = i as f64 * dt;
            let fr = assembly::frame(prep, &dm, &u, &fint, &fexts[i], &st, false, time0 + t, t, format!("Increment {i}: Step Time = {t:.4}"))?;
            if i + 1 == n {
                ctx.state.u = u.clone();
                ctx.state.elem_state = st.clone();
                ctx.state.reaction = fr.rf.clone();
            }
            ctx.out.frames.push(fr);
        }
        ctx.state.time = time0 + period;
        ctx.out.increments = ninc;
        ctx.out.iterations = iterations;
        Ok(())
    }
}
