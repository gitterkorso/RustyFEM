//! *STATIC, RIKS: the arc-length method for unstable equilibrium paths (post-buckling,
//! snap-through).
//!
//! The load factor λ is an unknown: F = F₀ + λ (F_ref − F₀), where F₀ is the load at
//! the start of the step and F_ref the step's loads. The method is the modified
//! Riks method (Riks 1979, Crisfield 1981, Ramm 1981):
//!
//! - The space is scaled: the displacements by ū (the largest magnitude in the first,
//!   linear solution v₀ = K⁻¹P) and the load by |P|, so that the point is (u/ū; λ).
//! - Predictor: the step Δl along the tangent (v; 1), Δλ = ±Δl κ/√(ṽ·ṽ + 1), where
//!   κ = √(ṽ₀·ṽ₀ + 1) at the first point of the step, so that the first increment gets
//!   Δλ = Δl_in/l_period (the arc length is reported in the same unit as Δl_in).
//!   The sign is chosen so that the predictor points in the same direction as the previous
//!   increment.
//! - Corrector: K [v c] = [P R], μ = −c̃·ṽ₀/(ṽ·ṽ₀ + 1), Δu += c + μ v,
//!   Δλ += μ (the correction orthogonal to the latest tangent), and v₀ ← v.
//! - Convergence with the same field-wise criteria as *STATIC (R_n, C_n, *CONTROLS).
//! - The arc length is changed automatically like the time increment in *STATIC (increase 1.5
//!   after two fast increments, quartering on divergence) within [Δl_min, Δl_max];
//!   DIRECT gives a fixed arc length.
//! - The step ends when λ passes λ_end, when the displacement at the given
//!   node/DOF passes the given value, or after INC= increments.
//!
//! Prescribed displacements are not scaled by λ (they are held at their value at the start of the
//! step); loads that do not change in the step are dead loads.

use super::api::{StepCtx, StepType};
use super::newton::NonlinearSystem;
use super::nlgeom::{apply_controls, iterate, NlSystem, RiksLoads};
use crate::fem::dof::Field;
use crate::fem::model::StepDef;
use crate::model::{fail, FemError, Result};
use std::collections::BTreeMap;

pub struct Riks;

pub static STEP_TYPES: &[&dyn StepType] = &[&Riks];

/// Data on the *STATIC, RIKS line.
struct Params {
    dl_in: f64,
    period: f64,
    dl_min: f64,
    dl_max: f64,
    lam_end: Option<f64>,
    /// (node index, slot, limit value)
    disp_end: Option<(usize, usize, f64)>,
}

fn params(ctx: &StepCtx) -> Result<Params> {
    let kw = ctx.step.procedure.as_ref().expect("procedure");
    let recs = kw.records();
    let fields: Vec<String> = recs.first().map(|d| d.fields.iter().map(|f| f.trim().to_string()).collect()).unwrap_or_default();
    let num = |i: usize| -> Result<Option<f64>> {
        match fields.get(i).filter(|f| !f.is_empty()) {
            None => Ok(None),
            Some(f) => crate::fem::model::parse_f64(f).map(Some).ok_or_else(|| FemError(format!("{}: '{f}' is not a number", kw.location()))),
        }
    };
    let period = num(1)?.filter(|&x| x > 0.0).unwrap_or(1.0);
    let dl_in = num(0)?.filter(|&x| x > 0.0).unwrap_or(period);
    let dl_min = num(2)?.filter(|&x| x > 0.0).unwrap_or((dl_in * 1e-5).min(1e-5 * period));
    let dl_max = num(3)?.filter(|&x| x > 0.0).unwrap_or(f64::INFINITY);
    let lam_end = num(4)?.filter(|&x| x != 0.0);
    let m = ctx.prep.model;
    let disp_end = match (fields.get(5).filter(|f| !f.is_empty()), num(6)?, num(7)?) {
        (Some(node), Some(dof), Some(val)) => {
            let n = find_node(m, node).ok_or_else(|| FemError(format!("{}: the node '{node}' for the end condition does not exist", kw.location())))?;
            let d = dof as usize;
            if !(1..=6).contains(&d) {
                return fail(format!("{}: DOF {d} for the end condition is not supported", kw.location()));
            }
            Some((n, d - 1, val))
        }
        _ => None,
    };
    Ok(Params { dl_in: dl_in / period, period, dl_min: dl_min / period, dl_max: dl_max / period, lam_end, disp_end })
}

/// Node as a number (in the single/default instance), INSTANCE.number or a node set with one node.
fn find_node(m: &crate::fem::model::FeModel, s: &str) -> Option<usize> {
    let up = s.to_uppercase();
    if let Some(v) = m.nsets.get(&up).filter(|v| v.len() == 1) {
        return Some(v[0]);
    }
    if let Some((inst, lab)) = up.rsplit_once('.') {
        let i = m.instance_index(inst)?;
        return m.node(i, lab.trim().parse().ok()?);
    }
    let lab: i64 = up.parse().ok()?;
    for name in ["MODEL", "ASSEMBLY"] {
        if let Some(n) = m.instance_index(name).and_then(|i| m.node(i, lab)) {
            return Some(n);
        }
    }
    (0..m.instances.len() as u32).find_map(|i| m.node(i, lab))
}

impl StepType for Riks {
    fn name(&self) -> &'static str {
        "STATIC, RIKS"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "STATIC" && !step.perturbation && step.procedure.as_ref().is_some_and(|p| p.has("RIKS"))).then_some(20)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        if !ctx.nlgeom {
            return fail("*STATIC, RIKS is only supported with NLGEOM=YES");
        }
        let prm = params(ctx)?;
        let fixed = ctx.step.procedure.as_ref().is_some_and(|p| p.has("DIRECT"));
        let mut ctrl = super::newton::Controls { max_increments: ctx.step.max_increments.max(1), ..Default::default() };
        apply_controls(&mut ctrl, &ctx.step.extra)?;
        let mut sys = NlSystem::new(ctx.prep, ctx.loads, ctx.state, ctx.solver, prm.period)?;
        sys.extrapolate = false;
        sys.riks = Some(RiksLoads { lambda: 0.0 });
        let verbose = ctx.opts.verbose;
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            sys.begin_increment(0.0, 0.0)?;
            sys.evaluate(false)?;
            frames.push(sys.frame(0.0, "Increment 0: Arc Length = 0.0".into())?);
        }
        let mut flux_sum: BTreeMap<Field, f64> = BTreeMap::new();
        let mut n_flux = 0usize;
        let mut lam = 0.0f64;
        let mut arc = 0.0f64;
        let mut dl = prm.dl_in;
        let mut ubar = 0.0f64;
        let mut kappa = 1.0f64;
        // the previous increment's change (scaled) and Δλ
        let mut prev: Option<(Vec<f64>, f64)> = None;
        let mut quick_prev = false;
        let mut incs = 0usize;
        let mut iters = 0usize;
        // ū = the largest displacement (translations) in v₀; the dot products include all DOFs
        let wu: Vec<f64> = (0..sys.n_eq()).map(|q| if sys.eq_field(q) == Field::Displacement { 1.0 } else { 0.0 }).collect();
        let scaled_dot = |a: &[f64], b: &[f64], ub: f64| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / (ub * ub);
        loop {
            if incs >= ctrl.max_increments {
                break;
            }
            let mut cutbacks = 0;
            let (out, dlam_inc, du_inc) = loop {
                // tangent and load pattern at the starting point
                sys.begin_increment(arc, arc + dl)?;
                sys.riks = Some(RiksLoads { lambda: lam });
                sys.evaluate(true)?;
                let pat = sys.load_pattern()?;
                if pat.iter().all(|&x| x == 0.0) {
                    return fail("*STATIC, RIKS: the step has no reference load (F_ref = F₀)");
                }
                let v0 = sys.solve(&pat)?;
                if ubar == 0.0 {
                    ubar = v0.iter().zip(&wu).fold(0.0f64, |a, (&x, w)| a.max(w * x.abs()));
                    if !(ubar > 0.0) {
                        return fail("*STATIC, RIKS: the load pattern gives no displacement");
                    }
                    // the arc-length unit: the first increment (Δλ = Δl_in/l_period)
                    // has the length Δl_in in the scaled space
                    kappa = (scaled_dot(&v0, &v0, ubar) + 1.0).sqrt();
                }
                let mag = dl * kappa / (scaled_dot(&v0, &v0, ubar) + 1.0).sqrt();
                let dlam0 = match &prev {
                    None => mag,
                    Some((du_p, dlam_p)) => {
                        let dir = scaled_dot(&v0, du_p, ubar) + dlam_p;
                        if dir >= 0.0 { mag } else { -mag }
                    }
                };
                if crate::settings::with(|s| s.nl_debug) {
                    eprintln!("  riks: |v0|² = {:.10e}, ū = {ubar:.10e}, Δλ0 = {dlam0:.10e}", v0.iter().map(|x| x * x).sum::<f64>());
                }
                let pred: Vec<f64> = v0.iter().map(|x| dlam0 * x).collect();
                sys.apply(&pred);
                let lam_cell = std::cell::Cell::new(lam + dlam0);
                let mut vo = v0.clone();
                sys.riks = Some(RiksLoads { lambda: lam + dlam0 });
                let mut hook = |s: &mut NlSystem, r: &[f64]| -> Result<Vec<f64>> {
                    let pat = s.load_pattern()?;
                    // the same factorization for both right-hand sides
                    let (v, c) = s.solve_pair(&pat, r)?;
                    let mu = -scaled_dot(&c, &vo, ubar) / (scaled_dot(&v, &vo, ubar) + 1.0);
                    let l = lam_cell.get() + mu;
                    lam_cell.set(l);
                    s.riks = Some(RiksLoads { lambda: l });
                    vo = v.clone();
                    Ok((0..c.len()).map(|i| c[i] + mu * v[i]).collect())
                };
                let res = iterate(&mut sys, &ctrl, &flux_sum, n_flux, &mut hook)?;
                match res {
                    Some(o) => {
                        let du = sys.increment();
                        break (o, lam_cell.get() - lam, du);
                    }
                    None => {
                        if fixed {
                            return fail(format!("Riks increment {} did not converge (DIRECT, fixed arc length)", incs + 1));
                        }
                        cutbacks += 1;
                        dl *= ctrl.cutback;
                        if cutbacks > ctrl.max_cutbacks || dl < prm.dl_min {
                            return fail(format!(
                                "*STATIC, RIKS: no convergence at load factor {lam:.6}: the arc length became smaller than the minimum or too many cutbacks"
                            ));
                        }
                        if verbose {
                            eprintln!("  Riks: halving the arc length to {:.4e}", dl * prm.period);
                        }
                    }
                }
            };
            for (f, q) in &out.flux {
                *flux_sum.entry(*f).or_default() += q;
            }
            n_flux += 1;
            sys.commit()?;
            lam += dlam_inc;
            arc += dl;
            prev = Some((du_inc, dlam_inc));
            incs += 1;
            iters += out.iterations;
            crate::progress::emit(crate::progress::Event::Increment { fraction: lam.abs().min(1.0), iterations: out.iterations })?;
            if verbose {
                let info = *sys.info.lock().unwrap();
                eprintln!(
                    "  Riks increment {incs}: load factor {lam:.6}, arc length {:.6}, {} iterations{}",
                    arc * prm.period,
                    out.iterations,
                    if info.negative > 0 { format!(", {} negative eigenvalues", info.negative) } else { String::new() }
                );
            }
            // end condition
            let mut done = false;
            if let Some(lam_e) = prm.lam_end {
                if (lam_e > 0.0 && lam >= lam_e) || (lam_e < 0.0 && lam <= lam_e) {
                    done = true;
                }
            }
            if let Some((node, slot, val)) = prm.disp_end {
                let u = sys.u_committed[node][slot];
                if (val >= 0.0 && u >= val) || (val < 0.0 && u <= val) {
                    done = true;
                }
            }
            if all || done || incs >= ctrl.max_increments {
                frames.push(sys.frame(arc * prm.period, format!("Increment {incs}: Arc Length = {:.4}, LPF = {lam:.6}", arc * prm.period))?);
            }
            if done {
                break;
            }
            if !fixed {
                let quick = out.iterations <= ctrl.ig && cutbacks == 0;
                if quick && quick_prev {
                    dl *= ctrl.growth;
                }
                quick_prev = quick;
                dl = dl.min(prm.dl_max).max(prm.dl_min);
            }
        }
        ctx.out.frames.extend(frames);
        ctx.out.increments = incs;
        ctx.out.iterations = iters;
        sys.store(ctx.state);
        ctx.state.time += arc * prm.period;
        Ok(())
    }
}
