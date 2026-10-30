//! *BUCKLE: linjär knäckning (egenvärdesknäckning, linjärt perturbationssteg).
//!
//! Löser (K₀ + λ K_Δ) φ = 0 där
//! - K₀ är tangentstyvheten i bastillståndet inklusive spänningsstyvheten från
//!   förlasten (föregående allmänna steg). Den tas med även när
//!   bastillståndet är geometriskt linjärt (kontrollerat i
//!   tests/cases/eigen/buckle_b23_forlast); efter ett NLGEOM-steg innehåller
//!   tangenten redan spänningsstyvheten,
//! - K_Δ är spänningsstyvheten från perturbationslasten (stegets egna laster),
//!   vars spänningar fås ur en linjär statisk lösning från noll.
//!
//! Kritisk last = förlast + λ · perturbationslast. Problemet skrivs
//! K₀ φ = λ (−K_Δ) φ och löses med subspace-iteration (`solver::eigen`) för de
//! egenvärden som ligger närmast noll (även negativa). Modformerna normeras så
//! att största förskjutningskomponenten är +1. B23/B33 har inre
//! frihetsgrader i egenvärdesproblemet, som i `frequency.rs`.
//!
//! Perturbationslastens egen styvhet ingår för tryck på kontinuumelementens
//! ytor (*DLOAD Pn på CPS/CPE/C3D8): trycket följer ytan, K_L = −∂f/∂u, och
//! den symmetriska delen ½(K_L + K_Lᵀ) används (kontrollerat i
//! tests/cases/eigen/buckle_cps4_tryck). Tryck i bastillståndet
//! (föregående steg) ger ingen lastyvhet här.
//!
//! Datarad: SUBSPACE (standard): antal egenvärden, högsta egenvärde av
//! intresse, antal iterationsvektorer, högsta antal iterationer. LANCZOS:
//! antal, minsta och högsta egenvärde. Frames: frame 0 =
//! bastillståndet, frame i = mod i (`value` = modnummer) med beskrivningen
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
            Some(x) => return kw.err(format!("EIGENSOLVER={x} stöds inte (SUBSPACE eller LANCZOS)")),
        }
        let lanczos = kw.param_upper("EIGENSOLVER").as_deref() == Some("LANCZOS");
        let nev = data_field(kw, 0)?.map(|v| v.round().max(0.0) as usize).filter(|&n| n > 0);
        let Some(nev) = nev else {
            return kw.err("ange antal egenvärden på dataraden");
        };
        // SUBSPACE: (antal, största egenvärde, vektorer, iterationer); LANCZOS: (antal, minsta, största, ...)
        let (lmin, lmax, vectors) = if lanczos {
            (data_field(kw, 1)?, data_field(kw, 2)?, 0)
        } else {
            (None, data_field(kw, 1)?, data_field(kw, 2)?.map_or(0, |v| v.round().max(0.0) as usize))
        };
        // perturbationslastens spänningar (linjärt, från noll)
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
            return fail("*BUCKLE: perturbationslasten ger inga spänningar (saknas *CLOAD/*DLOAD i steget?)");
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
            return fail("*BUCKLE: styvheten i bastillståndet är inte positivt definit (mekanism, eller förlasten är redan över knäcklasten)");
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
            // RF = K φ i låsta frihetsgrader (utan spänningsstyvhet)
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

/// Lastyvhet för följande tryck (*DLOAD Pn) på kontinuumelement: K_L = −∂f/∂x
/// med centrala differenser av elementets tryckkraft (exakt för linjära och
/// bilinjära ytor, där kraften är högst kvadratisk i koordinaterna),
/// symmetriserad.
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
                .ok_or_else(|| FemError(format!("{}: lasttypen {} är inget tryck", m.elem_name(e), dl.kind)))
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
