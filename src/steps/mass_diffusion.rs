//! *MASS DIFFUSION (stationär eller transient, fasta inkrement eller DCMAX=)
//! med diffusionselementen DC2D3 … DC3D20 (`elements/heat.rs`).
//!
//! Frihetsgraden 11 är den normerade koncentrationen φ = c/s (NNC11).
//! Ficks lag utan Soret- och tryckterm, J = −s D ∇φ, och massbalansen
//!
//! ∫ δφ s φ̇ dV + ∫ ∇δφ · s D ∇φ dV = ∫ δφ q dS
//!
//! är samma diskreta problem som värmeledningen med k = s D och ρc = s, så
//! steget använder `HeatSystem` (`steps/heat.rs`) med de egenskaperna:
//! kapaciteten är diagonal för första ordningens element, *CFLUX/*DFLUX/
//! *BOUNDARY 11 och *INITIAL CONDITIONS, TYPE=CONCENTRATION gäller φ.
//! DCMAX= ger automatisk inkrementering med samma regel som DELTMX.
//!
//! Utdata: NNC11 och RFL11 (nodvärden), CONC = s φ (första ordningens element:
//! φ i noden närmast punkten) och MFL = −s D ∇φ i
//! integrationspunkterna.

use super::api::{StepCtx, StepType};
use super::heat::{heat_controls, heat_loads, run_deltmx, HeatControls, HeatSystem, TS};
use super::newton::{self, NonlinearSystem};
use crate::elements::heat as hel;
use crate::elements::IpField;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{FeModel, MaterialDef, StepDef};
use crate::fem::output::FrameData;
use crate::materials::heat::{CondKind, HeatProps, Table};
use crate::materials::multiphysics::DiffProps;
use crate::model::{fail, FemError, Result};

pub struct MassDiffusion;

pub static STEP_TYPES: &[&dyn StepType] = &[&MassDiffusion];

const PROC: &str = "MASS DIFFUSION";

/// Nyckel i `SolutionState::vectors` för begynnelsekoncentrationerna.
pub const INIT_KEY: &str = "NNC11 INITIAL";

fn props_of(def: &MaterialDef) -> Result<HeatProps> {
    let Some(d) = DiffProps::from_def(def)? else {
        return fail(format!("material {}: *DIFFUSIVITY saknas (krävs för *MASS DIFFUSION)", def.name));
    };
    let k = &d.d;
    let v = vec![d.s * k[0][0], d.s * k[0][1], d.s * k[1][1], d.s * k[0][2], d.s * k[1][2], d.s * k[2][2]];
    Ok(HeatProps {
        name: def.name.clone(),
        cond: Some((CondKind::Aniso, Table::constant(v))),
        specific_heat: Some(Table::constant(vec![d.s])),
        density: 1.0,
        expansion: None,
        latent: Vec::new(),
    })
}

/// Begynnelsevärden från *INITIAL CONDITIONS, TYPE=CONCENTRATION (upplösta).
fn initial(model: &FeModel, state: &mut crate::fem::state::SolutionState) -> Result<()> {
    if state.vectors.contains_key(INIT_KEY) {
        return Ok(());
    }
    let mut v = vec![[0.0; SLOTS]; model.nodes.len()];
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.param_upper("TYPE").as_deref() == Some("CONCENTRATION")) {
        for d in &kw.data {
            let f = kw.floats(d)?;
            if let (Some(&n), Some(&c)) = (f.first(), f.get(1)) {
                let n = n as usize;
                v[n][TS] = c;
                if !state.bcs.contains_key(&(n, TS)) {
                    state.u[n][TS] = c;
                }
            }
        }
    }
    state.vectors.insert(INIT_KEY.into(), v);
    Ok(())
}

/// CONC och MFL per element.
fn output(model: &FeModel, props: &[Option<DiffProps>], u: &[NodeVec]) -> Result<Vec<Vec<IpField>>> {
    let mut out = Vec::new();
    for (e, el) in model.elements.iter().enumerate() {
        let Some(he) = hel::lookup(el.etype.name()) else {
            out.push(Vec::new());
            continue;
        };
        let Some(d) = el.section.and_then(|s| model.sections[s].material_index).and_then(|m| props[m].as_ref()) else {
            out.push(Vec::new());
            continue;
        };
        let t = el.section.map_or(1.0, |s| model.sections[s].solid_thickness());
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
        let ips = he.ips(&x, t).map_err(|err| FemError(format!("{}: {err}", model.elem_name(e))))?;
        let phi: Vec<f64> = el.nodes.iter().map(|&n| u[n][TS]).collect();
        let dim = he.dim();
        let labels: &[&str] = if dim == 2 { &["MFL1", "MFL2"] } else { &["MFL1", "MFL2", "MFL3"] };
        let mut conc = IpField::new("CONC", &["CONC"]);
        let mut mfl = IpField::new("MFL", labels);
        for (q, ip) in ips.iter().enumerate() {
            // första ordningen: värdet i noden närmast punkten (diagonal
            // kapacitet); annars interpolerat
            let c: f64 = if he.first_order() {
                let i = (0..ip.n.len()).max_by(|&a, &b| ip.n[a].total_cmp(&ip.n[b])).unwrap_or(0);
                phi[i]
            } else {
                ip.n.iter().zip(&phi).map(|(a, b)| a * b).sum()
            };
            let mut g = [0.0; 3];
            for (dn, &p) in ip.dndx.iter().zip(&phi) {
                for k in 0..3 {
                    g[k] += dn[k] * p;
                }
            }
            conc.points.push((q + 1, 0, vec![d.s * c]));
            mfl.points.push((q + 1, 0, (0..dim).map(|r| -d.s * (0..3).map(|k| d.d[r][k] * g[k]).sum::<f64>()).collect()));
        }
        out.push(vec![conc, mfl]);
    }
    Ok(out)
}

fn frame(sys: &HeatSystem, props: &[Option<DiffProps>], value: f64, description: String) -> Result<FrameData> {
    let mut f = sys.frame(value, description)?;
    f.elem = output(sys.p.model, props, &sys.u_committed)?;
    f.node_extra.push(("NNC11".into(), vec!["NNC11".into()], sys.u_committed.iter().map(|v| vec![v[TS]]).collect()));
    Ok(f)
}

fn controls(step: &StepDef) -> Result<HeatControls> {
    let mut hc = heat_controls(step)?;
    let kw = step.procedure.as_ref();
    if let Some(d) = kw.and_then(|k| k.param("DCMAX")) {
        let v = crate::fem::model::parse_f64(d).ok_or_else(|| FemError(format!("DCMAX={d} är inget tal")))?.max(0.0);
        hc.deltmx = v;
        if v > 0.0 && !hc.steady {
            hc.ctrl.fixed = false;
        }
    }
    Ok(hc)
}

impl StepType for MassDiffusion {
    fn name(&self) -> &'static str {
        PROC
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == PROC).then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let prep = ctx.prep;
        let model = prep.model;
        if ctx.step.perturbation {
            return fail("*MASS DIFFUSION kan inte vara ett perturbationssteg");
        }
        let hc = controls(ctx.step)?;
        let had_init = ctx.state.vectors.contains_key(INIT_KEY);
        initial(model, ctx.state)?;
        let mut loads = heat_loads(ctx.step, ctx.loads, hc.steady);
        if !had_init {
            for ((n, s), v) in loads.bc.iter_mut() {
                if *s == TS && !ctx.state.bcs.contains_key(&(*n, *s)) {
                    v.0 = ctx.state.u[*n][TS];
                }
            }
        }
        let props: Vec<Option<DiffProps>> = model.materials.iter().map(DiffProps::from_def).collect::<Result<_>>()?;
        let period = hc.ctrl.period;
        let mut sys = HeatSystem::new_with(prep, loads, ctx.state, !hc.steady, ctx.solver, period, &props_of)?;
        let all = ctx.opts.all_frames;
        let mut frames = Vec::new();
        if all {
            sys.begin_increment(0.0, 0.0)?;
            sys.u.clone_from(&sys.u_committed);
            let tr = sys.transient;
            sys.transient = false;
            sys.evaluate(false)?;
            sys.transient = tr;
            frames.push(frame(&sys, &props, 0.0, "Increment 0: Step Time = 0.0".into())?);
        }
        let mut inc = 0;
        let mut on = |s: &mut HeatSystem, t: f64| -> Result<()> {
            inc += 1;
            if all || t >= 1.0 - 1e-12 {
                frames.push(frame(s, &props, t * period, format!("Increment {inc}: Step Time = {:.4}", t * period))?);
            }
            Ok(())
        };
        let rep = if hc.deltmx > 0.0 && !hc.steady { run_deltmx(&mut sys, &hc, &mut on)? } else { newton::run(&mut sys, &hc.ctrl, &mut on)? };
        ctx.out.frames.extend(frames);
        ctx.out.increments = rep.increments;
        ctx.out.iterations = rep.iterations;
        ctx.state.u = sys.u_committed.clone();
        ctx.state.reaction = sys.reactions();
        ctx.state.time += period;
        Ok(())
    }
}
