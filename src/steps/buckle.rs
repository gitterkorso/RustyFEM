//! *BUCKLE: linear buckling (eigenvalue buckling, linear perturbation step).
//!
//! Solves (K₀ + λ K_Δ) φ = 0 where
//! - K₀ is the tangent stiffness in the base state including the stress stiffness from
//!   the preload (preceding general steps). It is included even when
//!   the base state is geometrically linear (verified in
//!   tests/cases/eigen/buckle_b23_forlast); after an NLGEOM step the
//!   tangent already contains the stress stiffness,
//! - K_Δ is the stress stiffness from the perturbation load (the step's own loads),
//!   whose stresses are obtained from a linear static solution from zero.
//!
//! Critical load = preload + λ · perturbation load. The problem is written
//! K₀ φ = λ (−K_Δ) φ and solved with subspace iteration (`solver::eigen`) for the
//! eigenvalues closest to zero (including negative ones). The mode shapes are normalized so
//! that the largest displacement component is +1. B23/B33 have internal
//! DOFs in the eigenvalue problem, as in `frequency.rs`.
//!
//! The perturbation load's own stiffness is included for pressure on continuum element
//! faces (*DLOAD Pn on CPS/CPE/C3D8): the pressure follows the surface, K_L = −∂f/∂u, and
//! the symmetric part ½(K_L + K_Lᵀ) is used (verified in
//! tests/cases/eigen/buckle_cps4_tryck). Pressure in the base state
//! (preceding steps) gives no load stiffness here.
//!
//! Data line: SUBSPACE (default): number of eigenvalues, maximum eigenvalue of
//! interest, number of iteration vectors, maximum number of iterations. LANCZOS:
//! number, minimum and maximum eigenvalue. Frames: frame 0 =
//! the base state, frame i = mode i (`value` = mode number) with the description
//! "Mode i: EigenValue = ...".

use super::api::{StepCtx, StepType};
use super::frequency::{assemble_geometric, assemble_stiffness, base_frame, data_field, factor_checked, g5, mode_frame, normalize_displacement, AllDofs};
use super::newton;
use crate::fem::assembly;
use crate::fem::model::StepDef;
use crate::fem::state::SolutionState;
use crate::fem::system::StaticSystem;
use crate::elements::Family;
use crate::fem::assembly::{Prepared, NONE};
use crate::fem::state::StepLoads;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::{self, combine, EigenOptions, SymCsr, Triplets};

pub struct Buckle;

pub static STEP_TYPES: &[&dyn StepType] = &[&Buckle];

impl StepType for Buckle {
    fn name(&self) -> &'static str {
        "BUCKLE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "BUCKLE").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        let model = p.model;
        let nn = model.nodes.len();
        let kw = ctx.step.procedure.as_ref().unwrap();
        match kw.param_upper("EIGENSOLVER").as_deref() {
            None | Some("SUBSPACE") | Some("LANCZOS") => {}
            Some(x) => return kw.err(format!("EIGENSOLVER={x} is not supported (SUBSPACE or LANCZOS)")),
        }
        let lanczos = kw.param_upper("EIGENSOLVER").as_deref() == Some("LANCZOS");
        let nev = data_field(kw, 0)?.map(|v| v.round().max(0.0) as usize).filter(|&n| n > 0);
        let Some(nev) = nev else {
            return kw.err("give the number of eigenvalues on the data line");
        };
        // SUBSPACE: (number, largest eigenvalue, vectors, iterations); LANCZOS: (number, minimum, maximum, ...)
        let (lmin, lmax, vectors) = if lanczos {
            (data_field(kw, 1)?, data_field(kw, 2)?, 0)
        } else {
            (None, data_field(kw, 1)?, data_field(kw, 2)?.map_or(0, |v| v.round().max(0.0) as usize))
        };
        // the perturbation load's stresses (linear, from zero)
        let mut zero = SolutionState::new(model);
        zero.elem_state = p.init_states()?;
        let (du, dstate) = {
            let mut sys = StaticSystem::new(p, ctx.loads, &zero, false, ctx.solver, 1.0)?;
            newton::solve_linear(&mut sys)?;
            (sys.u_committed.clone(), sys.state_old.clone())
        };
        let (dm, _) = assembly::dofmap(p, &ctx.loads.constrained(), ctx.solver)?;
        let all = AllDofs::new(p, &dm);
        let ns = all.n_solve();
        let kg = assemble_geometric(p, &all, &du, &dstate)?;
        let kl = pressure_stiffness(p, &all, ctx.loads)?;
        let kd = combine(&[(&kg, 1.0), (&kl, 1.0)], ns);
        if kd.val.iter().all(|v| *v == 0.0) {
            return fail("*BUCKLE: the perturbation load gives no stresses (missing *CLOAD/*DLOAD in the step?)");
        }
        let base = &*ctx.state;
        let k0 = assemble_stiffness(p, &dm, &all, base, ctx.nlgeom)?;
        let a = if ctx.nlgeom {
            k0
        } else {
            let kb = assemble_geometric(p, &all, &base.u, &base.elem_state)?;
            combine(&[(&k0, 1.0), (&kb, 1.0)], ns)
        };
        let Some(mat) = factor_checked(&a, &all, p, ctx.solver)? else {
            return fail("*BUCKLE: the stiffness in the base state is not positive definite (mechanism, or the preload already exceeds the buckling load)");
        };
        let opts = EigenOptions { nev: nev.min(ns), vectors, ..Default::default() };
        let bmul = |x: &[f64]| kd.mul(x).into_iter().map(|v| -v).collect::<Vec<f64>>();
        let res = eigen::subspace(ns, &mat, &bmul, &opts)?;
        let mut modes: Vec<(f64, Vec<f64>)> = res.theta.iter().zip(res.vectors).map(|(t, v)| (1.0 / t, v)).collect();
        modes.retain(|(l, _)| lmax.is_none_or(|m| l.abs() <= m) && lmin.is_none_or(|m| *l >= m));
        ctx.out.frames.push(base_frame(p, &dm, base)?);
        let time = base.time;
        let mut hist = Vec::new();
        for (i, (lam, v)) in modes.iter().enumerate() {
            let mode = i + 1;
            let mut phi = all.to_nodes(v, nn);
            normalize_displacement(&mut phi, &dm.active);
            // RF = K φ in fixed DOFs (without stress stiffness)
            let fext = vec![[0.0; crate::fem::dof::SLOTS]; nn];
            let desc = format!("Mode {mode:>9}: EigenValue = {:>11}", g5(*lam));
            ctx.out.frames.push(mode_frame(p, &dm, &phi, &fext, time, mode, desc)?);
            hist.push((mode as f64, *lam));
        }
        ctx.out.history.push((super::frequency::HISTORY_REGION.to_string(), "EIGVAL".to_string(), hist));
        ctx.out.increments = 1;
        ctx.out.iterations = res.iterations;
        Ok(())
    }
}

/// Load stiffness for follower pressure (*DLOAD Pn) on continuum elements: K_L = −∂f/∂x
/// with central differences of the element's pressure force (exact for linear and
/// bilinear faces, where the force is at most quadratic in the coordinates),
/// symmetrized.
fn pressure_stiffness(p: &Prepared, all: &AllDofs, loads: &StepLoads) -> Result<SymCsr> {
    let m = p.model;
    let mut t = Triplets::new(all.n());
    for (dl, _) in &loads.dload {
        let e = dl.elem;
        let et = m.elements[e].etype;
        if !matches!(et.family(), Family::Continuum2D | Family::Continuum3D) || !dl.kind.starts_with('P') || dl.kind.len() < 2 {
            continue;
        }
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let ndim = et.dim();
        let th = if ndim == 2 { ctx.section(et.name())?.solid_thickness() } else { 1.0 };
        let force = |x: &[[f64; 3]]| -> Result<Vec<f64>> {
            crate::elements::common::pressure_load(x, et.faces(), &dl.kind, dl.magnitude, th, ndim, et.name())
                .map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?
                .ok_or_else(|| FemError(format!("{}: the load type {} is not a pressure", m.elem_name(e), dl.kind)))
        };
        let nd = ctx.x.len() * ndim;
        let mut k = vec![0.0; nd * nd];
        let size = ctx.x.iter().flat_map(|x| x.iter()).fold(0.0f64, |a, v| a.max(v.abs())).max(1e-300);
        let h = 1e-4 * size;
        for b in 0..nd {
            let mut xp = ctx.x.to_vec();
            let mut xm = ctx.x.to_vec();
            xp[b / ndim][b % ndim] += h;
            xm[b / ndim][b % ndim] -= h;
            let (fp, fm) = (force(&xp)?, force(&xm)?);
            for a in 0..nd {
                k[a * nd + b] = -(fp[a] - fm[a]) / (2.0 * h);
            }
        }
        let map: Vec<usize> = p.edofs[e].iter().map(|&(n, s)| all.idx[n][s]).collect();
        if map.len() != nd {
            continue;
        }
        for a in 0..nd {
            for b in 0..nd {
                let (ia, ib) = (map[a], map[b]);
                if ia != NONE && ib != NONE && ia >= ib {
                    t.add(ia, ib, 0.5 * (k[a * nd + b] + k[b * nd + a]));
                }
            }
        }
    }
    Ok(t.build())
}
