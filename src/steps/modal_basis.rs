//! The mode basis for the modal steps (*MODAL DYNAMIC, *STEADY STATE DYNAMICS,
//! *RESPONSE SPECTRUM): exactly the same eigenmodes as the preceding
//! *FREQUENCY step, plus modal damping, *SELECT EIGENMODES and *BASE MOTION.
//!
//! The modes are recomputed with `frequency::extract` for the nearest preceding
//! *FREQUENCY step, with that step's boundary conditions and the base state. This gives
//! bit for bit the same modes as the *FREQUENCY step wrote (deterministic
//! extraction), with consistent mass and internal DOFs (B23/B33) where
//! *FREQUENCY has them. The vectors are normalized to generalized mass 1
//! (φᵀMφ = 1 over all DOFs, including the internal ones).
//!
//! *SELECT EIGENMODES: DEFINITION=MODE NUMBERS
//! (default) with a list of mode numbers per data line, or with GENERATE
//! "first, last, increment"; DEFINITION=FREQUENCY RANGE with "lowest, highest
//! frequency" (Hz). The mode numbers in *MODAL DAMPING refer to the original numbers;
//! damping for modes that were not selected is ignored.
//!
//! *MODAL DAMPING: VISCOUS (fraction of critical damping, default), RAYLEIGH
//! (ζ = α/(2ω) + βω/2) and STRUCTURAL (s, imaginary stiffness in ω²(1 + is)).
//! DEFINITION=MODE NUMBERS: "first mode, last mode, value" (RAYLEIGH: α, β).
//! DEFINITION=FREQUENCY RANGE: "frequency (Hz), value" per line, linear
//! interpolation in frequency between the lines and the end value outside. Different damping types are added; the same type in several keywords overrides
//! per mode.
//!
//! *BASE MOTION (the primary base): DOF=1–3, AMPLITUDE=, SCALE=, TYPE=ACCELERATION
//! (default), VELOCITY or DISPLACEMENT. The equation of motion is written relative to
//! the base, u = u_rel + r u_g, so the modal load becomes −Γᵢ ü_g with
//! the participation factor Γᵢ = φᵢᵀ M r (r = rigid-body displacement in the DOF direction
//! over all DOFs, including the fixed ones – the same as PF in *FREQUENCY).
//! U, V, A are written relative to the base and TU, TV, TA as totals.

use super::super::frequency::{self, AllDofs};
use crate::fem::assembly::DofMap;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::fem::state::StepLoads;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::SymCsr;
use crate::steps::api::StepCtx;

/// Eigenmodes from a *FREQUENCY step (after *SELECT EIGENMODES).
pub struct ModalBasis {
    pub dm: DofMap,
    pub all: AllDofs,
    /// The mass matrix over all DOFs (free, internal, fixed).
    pub mfull: SymCsr,
    /// Mode number (1-based) in the *FREQUENCY step.
    pub number: Vec<usize>,
    pub lambda: Vec<f64>,
    pub omega: Vec<f64>,
    /// Mode vectors in the `all` numbering (length `all.n()`), φᵀMφ = 1.
    pub x: Vec<Vec<f64>>,
    /// The nodal values of the mode vectors.
    pub phi: Vec<Vec<NodeVec>>,
}

impl ModalBasis {
    /// The modes from the nearest preceding *FREQUENCY step.
    pub fn from_frequency_step(ctx: &StepCtx) -> Result<ModalBasis> {
        let p = ctx.prep;
        let model = p.model;
        let Some(j) = (0..ctx.index).rev().find(|&i| model.steps[i].procedure_name() == "FREQUENCY") else {
            return fail(format!("*{} requires a preceding *FREQUENCY step", ctx.step.procedure_name()));
        };
        let fs = &model.steps[j];
        let (loads, _) = StepLoads::build_with(model, fs, ctx.state, &p.active, false)?;
        let kw = fs.procedure.as_ref().unwrap();
        let ex = frequency::extract(p, kw, &loads, ctx.state, ctx.state.nlgeom, ctx.solver, false)?;
        let nn = model.nodes.len();
        let selected = select_eigenmodes(ctx.step, &ex.modes.iter().map(|m| m.0).collect::<Vec<_>>())?;
        let mut b = ModalBasis { dm: ex.dm, all: ex.all, mfull: ex.mfull, number: Vec::new(), lambda: Vec::new(), omega: Vec::new(), x: Vec::new(), phi: Vec::new() };
        let ntot = b.all.n();
        for (i, (lam, v)) in ex.modes.into_iter().enumerate() {
            if !selected[i] {
                continue;
            }
            let mut x = v;
            x.resize(ntot, 0.0);
            let gm = b.mfull.bilinear(&x, &x);
            if gm > 0.0 {
                let f = 1.0 / gm.sqrt();
                x.iter_mut().for_each(|y| *y *= f);
            }
            b.phi.push(b.all.to_nodes(&x, nn));
            b.x.push(x);
            b.number.push(i + 1);
            b.lambda.push(lam);
            b.omega.push(lam.max(0.0).sqrt());
        }
        if b.x.is_empty() {
            return fail("no eigenmodes selected (*FREQUENCY/*SELECT EIGENMODES)");
        }
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Modal loads φᵢᵀ F for nodal forces F.
    pub fn modal_force(&self, f: &[NodeVec]) -> Vec<f64> {
        self.phi.iter().map(|phi| phi.iter().zip(f).map(|(a, b)| (0..SLOTS).map(|s| a[s] * b[s]).sum::<f64>()).sum()).collect()
    }

    /// Σ φᵢ qᵢ as a nodal vector.
    pub fn superpose(&self, q: &[f64]) -> Vec<NodeVec> {
        let nn = self.phi.first().map_or(0, |p| p.len());
        let mut u = vec![[0.0; SLOTS]; nn];
        for (phi, &qi) in self.phi.iter().zip(q) {
            if qi == 0.0 {
                continue;
            }
            for (un, pn) in u.iter_mut().zip(phi) {
                for s in 0..SLOTS {
                    un[s] += pn[s] * qi;
                }
            }
        }
        u
    }

    /// Participation factors Γᵢ = φᵢᵀ M r for the rigid-body displacement r in
    /// direction `dir` (0–2 translation, 3–5 rotation about the origin).
    pub fn participation(&self, nodes: &[[f64; 3]], dir: usize) -> Vec<f64> {
        let r = frequency::rigid(&self.all, nodes, dir);
        let mr = self.mfull.mul(&r);
        self.x.iter().map(|x| x.iter().zip(&mr).map(|(a, b)| a * b).sum()).collect()
    }

    /// M r (all DOFs) for the rigid-body displacement in direction `dir`.
    pub fn mass_times_rigid(&self, nodes: &[[f64; 3]], dir: usize) -> Vec<f64> {
        self.mfull.mul(&frequency::rigid(&self.all, nodes, dir))
    }
}

/// Selected modes (indices in the *FREQUENCY step's mode list) according to *SELECT EIGENMODES.
fn select_eigenmodes(step: &StepDef, lambda: &[f64]) -> Result<Vec<bool>> {
    let kws: Vec<&RawKeyword> = step.extra.iter().filter(|k| k.name == "SELECT EIGENMODES").collect();
    if kws.is_empty() {
        return Ok(vec![true; lambda.len()]);
    }
    let mut sel = vec![false; lambda.len()];
    let two_pi = 2.0 * std::f64::consts::PI;
    for kw in kws {
        let def = kw.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
        for d in &kw.data {
            let v = kw.floats(d)?;
            match def.as_str() {
                "MODE NUMBERS" => {
                    let list: Vec<usize> = if kw.has("GENERATE") {
                        if v.len() < 2 {
                            return kw.err_at(d.line, "GENERATE: write 'first mode, last mode[, increment]'");
                        }
                        let inc = v.get(2).copied().filter(|&x| x >= 1.0).unwrap_or(1.0).round() as usize;
                        (v[0].round() as usize..=v[1].round() as usize).step_by(inc).collect()
                    } else {
                        v.iter().map(|x| x.round() as usize).collect()
                    };
                    for m in list {
                        if m >= 1 && m <= sel.len() {
                            sel[m - 1] = true;
                        }
                    }
                }
                "FREQUENCY RANGE" => {
                    if v.len() < 2 {
                        return kw.err_at(d.line, "write 'lowest frequency, highest frequency'");
                    }
                    for (i, l) in lambda.iter().enumerate() {
                        let f = l.max(0.0).sqrt() / two_pi;
                        if f >= v[0] && f <= v[1] {
                            sel[i] = true;
                        }
                    }
                }
                other => return kw.err(format!("DEFINITION={other} must be MODE NUMBERS or FREQUENCY RANGE")),
            }
        }
    }
    Ok(sel)
}

/// Modal damping per selected mode.
#[derive(Debug, Clone)]
pub struct Damping {
    /// Fraction of critical viscous damping (VISCOUS + RAYLEIGH).
    pub zeta: Vec<f64>,
    /// Structural damping s (ω²(1 + i s)).
    pub structural: Vec<f64>,
}

/// Linear interpolation in a table (x, y) sorted in x, end values outside.
fn interp(t: &[(f64, f64)], x: f64) -> f64 {
    if x <= t[0].0 {
        return t[0].1;
    }
    for w in t.windows(2) {
        if x <= w[1].0 {
            let d = w[1].0 - w[0].0;
            return if d > 0.0 { w[0].1 + (w[1].1 - w[0].1) * (x - w[0].0) / d } else { w[1].1 };
        }
    }
    t[t.len() - 1].1
}

/// Damping from the step's *MODAL DAMPING for the mode basis `b`.
pub fn modal_damping(step: &StepDef, b: &ModalBasis) -> Result<Damping> {
    let n = b.len();
    // the same DEFINITION is required in *SELECT EIGENMODES and *MODAL DAMPING in the same step
    let def_of = |k: &RawKeyword| k.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
    if let Some(sel) = step.extra.iter().find(|k| k.name == "SELECT EIGENMODES") {
        if let Some(md) = step.extra.iter().find(|k| k.name == "MODAL DAMPING" && def_of(k) != def_of(sel)) {
            return md.err("DEFINITION= must be the same in *SELECT EIGENMODES and *MODAL DAMPING in the same step");
        }
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    // per type: viscous, Rayleigh (as ζ), structural
    let mut vals = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for kw in step.extra.iter().filter(|k| k.name == "MODAL DAMPING") {
        if kw.has("COMPOSITE") || kw.param_upper("MODAL").is_some_and(|m| m != "DIRECT") {
            return kw.err("MODAL=COMPOSITE is not supported (only MODAL=DIRECT)");
        }
        if let Some(v) = kw.param_upper("VISCOUS") {
            if v != "FRACTION OF CRITICAL DAMPING" {
                return kw.err(format!("VISCOUS={v} is not supported (FRACTION OF CRITICAL DAMPING)"));
            }
        }
        let kind = if kw.has("STRUCTURAL") {
            2
        } else if kw.has("RAYLEIGH") {
            1
        } else {
            0
        };
        let def = kw.param_upper("DEFINITION").unwrap_or_else(|| "MODE NUMBERS".into());
        match def.as_str() {
            "MODE NUMBERS" => {
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    if v.len() < 3 {
                        return kw.err_at(d.line, "write 'first mode, last mode, value'");
                    }
                    let a = v[0].round() as usize;
                    let last = if v[1] > 0.0 { v[1].round() as usize } else { a };
                    for i in 0..n {
                        let m = b.number[i];
                        if m < a || m > last {
                            continue;
                        }
                        vals[kind][i] = if kind == 1 {
                            let w = b.omega[i];
                            if w > 0.0 { v[2] / (2.0 * w) + v.get(3).copied().unwrap_or(0.0) * w / 2.0 } else { 0.0 }
                        } else {
                            v[2]
                        };
                    }
                }
            }
            "FREQUENCY RANGE" => {
                if kind == 1 {
                    return kw.err("RAYLEIGH with DEFINITION=FREQUENCY RANGE is not supported yet");
                }
                let mut t: Vec<(f64, f64)> = Vec::new();
                for d in &kw.data {
                    let v = kw.floats(d)?;
                    if v.len() < 2 {
                        return kw.err_at(d.line, "write 'frequency, value'");
                    }
                    t.push((v[0], v[1]));
                }
                if t.is_empty() {
                    return kw.err("data lines are missing");
                }
                t.sort_by(|a, b| a.0.total_cmp(&b.0));
                for i in 0..n {
                    vals[kind][i] = interp(&t, b.omega[i] / two_pi);
                }
            }
            other => return kw.err(format!("DEFINITION={other} must be MODE NUMBERS or FREQUENCY RANGE")),
        }
    }
    let zeta = (0..n).map(|i| vals[0][i] + vals[1][i]).collect();
    Ok(Damping { zeta, structural: vals[2].clone() })
}

/// Type of base motion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BaseKind {
    Acceleration,
    Velocity,
    Displacement,
}

/// *BASE MOTION (the primary base).
#[derive(Debug, Clone)]
pub struct BaseMotion {
    /// Direction 0–2.
    pub dir: usize,
    pub amplitude: Option<String>,
    pub scale: f64,
    pub kind: BaseKind,
}

/// The step's *BASE MOTION.
pub fn base_motions(step: &StepDef) -> Result<Vec<BaseMotion>> {
    let mut out = Vec::new();
    for kw in step.extra.iter().filter(|k| k.name == "BASE MOTION") {
        if kw.has("BASE NAME") {
            return kw.err("secondary bases (BASE NAME=) are not supported");
        }
        let dof = kw.param("DOF").and_then(parse_f64).ok_or_else(|| FemError(format!("{}: DOF= is missing", kw.location())))?.round() as usize;
        if !(1..=3).contains(&dof) {
            return kw.err("only translations (DOF=1, 2 or 3) are supported");
        }
        let scale = match kw.param("SCALE") {
            Some(s) => parse_f64(s).ok_or_else(|| FemError(format!("{}: SCALE={s} is not a number", kw.location())))?,
            None => 1.0,
        };
        let kind = match kw.param_upper("TYPE").as_deref().unwrap_or("ACCELERATION") {
            "ACCELERATION" => BaseKind::Acceleration,
            "VELOCITY" => BaseKind::Velocity,
            "DISPLACEMENT" => BaseKind::Displacement,
            t => return kw.err(format!("TYPE={t} must be ACCELERATION, VELOCITY or DISPLACEMENT")),
        };
        if kw.has("LOADING") && kw.param_upper("LOADING").as_deref() != Some("UNIFORM") {
            return kw.err("only LOADING=UNIFORM is supported");
        }
        // the base motion requires an amplitude
        let Some(amplitude) = kw.param_upper("AMPLITUDE") else {
            return kw.err("AMPLITUDE= is missing (the base motion requires an amplitude)");
        };
        out.push(BaseMotion { dir: dof - 1, amplitude: Some(amplitude), scale, kind });
    }
    Ok(out)
}

/// Adds r·v (direction `dir`) to the translation in `dir` of all nodes
/// (for TU, TV, TA).
pub fn add_rigid(u: &mut [NodeVec], dir: usize, v: f64) {
    for x in u.iter_mut() {
        x[dir] += v;
    }
}
