//! *RANDOM RESPONSE: response to a stationary random load (power spectral density,
//! PSD) using the modes from the preceding *FREQUENCY step.
//!
//! The loads are divided into load cases: `*CLOAD, LOAD CASE=n` and
//! `*BASE MOTION, DOF=, LOAD CASE=n` (base acceleration). Each load case gives
//! a modal transfer function as in the modal *STEADY STATE DYNAMICS
//! (`steady_state.rs`): qᵢ = Pᵢ / (ωᵢ² − Ω² + i(2ζᵢωᵢΩ + sᵢωᵢ²)), with Pᵢ = φᵢᵀF
//! or −Γᵢ (base motion), and the response H_c = Σ φᵢ qᵢ. Only
//! diagonal damping (*MODAL DAMPING) is included; material and element damping
//! are ignored.
//!
//! The load case weights are given with `*CORRELATION, PSD=name, TYPE=CORRELATED|
//! UNCORRELATED` and the data lines "load case, real part[, imaginary part]": with CORRELATED (default) a load case's loads at
//! different nodes are fully correlated, with UNCORRELATED uncorrelated; different
//! load cases are uncorrelated. The frequency function
//! `*PSD-DEFINITION, NAME=, TYPE=FORCE|BASE` has the data lines "real part,
//! imaginary part, frequency" and is interpolated log–log. The spectral density of a
//! response quantity is S(f) = Σ Re(c Ψ(f)) |H(f)|² over the load cases (CORRELATED)
//! or over the load cases' nodes and degrees of freedom (UNCORRELATED).
//! `*BASE MOTION, SCALE=` does not affect the result.
//!
//! Frequency points as in *STEADY STATE DYNAMICS (`frequency_points`).
//! Output: frame 0 with zeros, then one frame per frequency
//! with `value` = the frequency; U, V, A (and TU, TA for base motion) and S are
//! spectral densities, RU, RV, RA, RTU, RTA and RS are RMS values
//! √(∫ S df) from the lowest frequency to the frame frequency (trapezoidal rule in
//! frequency).

use super::api::{StepCtx, StepType};
use super::complex_frequency::dense::Cx;
use super::modal::basis::{modal_damping, ModalBasis};
use super::modal::steady_state::frequency_points;
use super::modal::{add_node_field, has_rotations, plain_frame};
use crate::fem::dof::{slot_of, NodeVec, SLOTS};
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use std::collections::BTreeMap;

pub struct RandomResponse;

pub static STEP_TYPES: &[&dyn StepType] = &[&RandomResponse];

/// *PSD-DEFINITION: (frequency, value) sorted by frequency.
struct Psd {
    table: Vec<(f64, Cx)>,
}

impl Psd {
    fn parse(kw: &RawKeyword) -> Result<Psd> {
        if let Some(d) = kw.param_upper("DEFINITION") {
            if d != "USER" {
                return kw.err(format!("DEFINITION={d} is not supported (only USER)"));
            }
        }
        let mut table = Vec::new();
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() < 3 {
                return kw.err_at(d.line, "write 'real part, imaginary part, frequency'");
            }
            table.push((v[2], Cx::new(v[0], v[1])));
        }
        if table.is_empty() {
            return kw.err("data lines missing");
        }
        table.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Psd { table })
    }

    /// The value at frequency f: interpolation on a log–log scale between
    /// the table points (linear if any value is ≤ 0), end values outside.
    fn at(&self, f: f64) -> Cx {
        let t = &self.table;
        if f <= t[0].0 {
            return t[0].1;
        }
        for w in t.windows(2) {
            let ((f0, a), (f1, b)) = (w[0], w[1]);
            if f <= f1 {
                if f1 <= f0 {
                    return b;
                }
                let lin = |x: f64, y: f64| x + (y - x) * (f - f0) / (f1 - f0);
                let log = |x: f64, y: f64| {
                    if x > 0.0 && y > 0.0 && f0 > 0.0 {
                        (x.ln() + (y.ln() - x.ln()) * (f.ln() - f0.ln()) / (f1.ln() - f0.ln())).exp()
                    } else if x == y {
                        x
                    } else {
                        lin(x, y)
                    }
                };
                return Cx::new(log(a.re, b.re), if a.im == 0.0 && b.im == 0.0 { 0.0 } else { lin(a.im, b.im) });
            }
        }
        t[t.len() - 1].1
    }
}

/// A term in the cross-correlation: load case, coefficient, PSD function and whether
/// the load case's loads at different nodes are uncorrelated.
struct Term {
    case: i64,
    c: Cx,
    psd: String,
    uncorrelated: bool,
}

/// *CORRELATION: the data lines "load case, real part[, imaginary part]" for both
/// types (the first field is the load case; a later line for
/// the same load case replaces the earlier one). TYPE=CORRELATED
/// (default): the load case's loads at different nodes are fully correlated;
/// UNCORRELATED: they are uncorrelated. Different load cases are uncorrelated.
fn correlations(step: &StepDef) -> Result<Vec<Term>> {
    let mut out: Vec<Term> = Vec::new();
    for kw in step.extra.iter().filter(|k| k.name == "CORRELATION") {
        let Some(psd) = kw.param_upper("PSD") else { return kw.err("PSD= missing") };
        let uncorrelated = match kw.param_upper("TYPE").as_deref() {
            None | Some("CORRELATED") => false,
            Some("UNCORRELATED") => true,
            Some(other) => return kw.err(format!("TYPE={other} is not supported (CORRELATED or UNCORRELATED)")),
        };
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() < 2 {
                return kw.err_at(d.line, "write 'load case, real part[, imaginary part]'");
            }
            let case = v[0].round() as i64;
            let c = Cx::new(v[1], v.get(2).copied().unwrap_or(0.0));
            out.retain(|t| !(t.case == case && t.psd == psd));
            out.push(Term { case, c, psd: psd.clone(), uncorrelated });
        }
    }
    if out.is_empty() {
        return fail("*RANDOM RESPONSE requires *CORRELATION");
    }
    Ok(out)
}

/// A part of a load case: modal load φᵢᵀF (or −Γᵢ for base motion) and
/// the base motion direction.
struct Part {
    p: Vec<f64>,
    dir: Option<usize>,
}

/// The load cases' parts: one per node and degree of freedom (for UNCORRELATED) and
/// their sum (CORRELATED).
fn load_cases(ctx: &StepCtx, b: &ModalBasis) -> Result<BTreeMap<i64, Vec<Part>>> {
    let model = ctx.prep.model;
    let nn = model.nodes.len();
    let mut cases: BTreeMap<i64, Vec<Part>> = BTreeMap::new();
    let num = |kw: &RawKeyword, k: &str| -> Result<Option<f64>> {
        match kw.param(k) {
            Some(s) => parse_f64(s).map(Some).ok_or_else(|| FemError(format!("{}: {k}={s} is not a number", kw.location()))),
            None => Ok(None),
        }
    };
    for mk in ctx.step.extra.iter().filter(|k| k.name == "_LOAD CASE CLOAD") {
        let lc = num(mk, "LOAD CASE")?.unwrap_or(1.0).round() as i64;
        let bi = num(mk, "BLOCK")?.unwrap_or(0.0) as usize;
        let block = &ctx.step.cloads[bi];
        if block.amplitude.is_some() {
            return mk.err("AMPLITUDE= on *CLOAD in *RANDOM RESPONSE is not supported (the frequency dependence is given by *PSD-DEFINITION)");
        }
        for &(n, d, v) in &block.items {
            let mut f = vec![[0.0; SLOTS]; nn];
            f[n][slot_of(d)] += v;
            cases.entry(lc).or_default().push(Part { p: b.modal_force(&f), dir: None });
        }
    }
    if ctx.step.cloads.len() > ctx.step.extra.iter().filter(|k| k.name == "_LOAD CASE CLOAD").count() {
        return fail("*CLOAD in *RANDOM RESPONSE requires LOAD CASE=");
    }
    if !ctx.step.dloads.is_empty() {
        return fail("*DLOAD in *RANDOM RESPONSE is not yet supported (use *CLOAD, LOAD CASE=)");
    }
    for kw in ctx.step.extra.iter().filter(|k| k.name == "BASE MOTION") {
        let dof = num(kw, "DOF")?.ok_or_else(|| FemError(format!("{}: DOF= missing", kw.location())))?.round() as usize;
        if !(1..=3).contains(&dof) {
            return kw.err("only translations (DOF=1, 2 or 3) are supported");
        }
        if kw.param_upper("TYPE").is_some_and(|t| t != "ACCELERATION") {
            return kw.err("only TYPE=ACCELERATION is supported in *RANDOM RESPONSE");
        }
        let lc = num(kw, "LOAD CASE")?.unwrap_or(1.0).round() as i64;
        // SCALE= does not affect the result in *RANDOM RESPONSE:
        // the base acceleration's spectral density is given entirely by *PSD-DEFINITION
        let gamma = b.participation(&model.nodes, dof - 1);
        cases.entry(lc).or_default().push(Part { p: gamma.iter().map(|g| -g).collect(), dir: Some(dof - 1) });
    }
    Ok(cases)
}

/// The spectral density Σ w |H|² for each value in a frame. `resp` is
/// (weight, real part, imaginary part) of the response as frames.
fn psd_frame(resp: &[(f64, FrameData, FrameData)]) -> FrameData {
    let mut out = resp[0].1.clone();
    let comb = |get: &dyn Fn(&FrameData) -> f64| -> f64 { resp.iter().map(|(w, a, b)| w * (get(a).powi(2) + get(b).powi(2))).sum() };
    for n in 0..out.u.len() {
        for s in 0..SLOTS {
            out.u[n][s] = comb(&|f: &FrameData| f.u[n][s]);
        }
    }
    for e in 0..out.elem.len() {
        for k in 0..out.elem[e].len() {
            for p in 0..out.elem[e][k].points.len() {
                for c in 0..out.elem[e][k].points[p].2.len() {
                    out.elem[e][k].points[p].2[c] = comb(&|f: &FrameData| f.elem[e][k].points[p].2[c]);
                }
            }
        }
    }
    out
}

/// Running integral ∫ S df (trapezoidal rule) of all values in frames.
struct Rms {
    prev: Option<(f64, FrameData)>,
    acc: Option<FrameData>,
}

impl Rms {
    fn add(&mut self, f: f64, fr: &FrameData) -> FrameData {
        let mut acc = self.acc.take().unwrap_or_else(|| zero_like(fr));
        if let Some((f0, p)) = &self.prev {
            let h = 0.5 * (f - f0);
            for n in 0..acc.u.len() {
                for s in 0..SLOTS {
                    acc.u[n][s] += h * (p.u[n][s] + fr.u[n][s]);
                }
            }
            for e in 0..acc.elem.len() {
                for k in 0..acc.elem[e].len() {
                    for q in 0..acc.elem[e][k].points.len() {
                        for c in 0..acc.elem[e][k].points[q].2.len() {
                            acc.elem[e][k].points[q].2[c] += h * (p.elem[e][k].points[q].2[c] + fr.elem[e][k].points[q].2[c]);
                        }
                    }
                }
            }
        }
        self.prev = Some((f, fr.clone()));
        let mut rms = acc.clone();
        rms.u.iter_mut().for_each(|x| x.iter_mut().for_each(|v| *v = v.max(0.0).sqrt()));
        for fields in &mut rms.elem {
            for fl in fields {
                for p in &mut fl.points {
                    p.2.iter_mut().for_each(|v| *v = v.max(0.0).sqrt());
                }
            }
        }
        self.acc = Some(acc);
        rms
    }
}

fn zero_like(fr: &FrameData) -> FrameData {
    let mut z = fr.clone();
    z.u.iter_mut().for_each(|x| *x = [0.0; SLOTS]);
    for fields in &mut z.elem {
        for fl in fields {
            for p in &mut fl.points {
                p.2.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }
    z
}

impl StepType for RandomResponse {
    fn name(&self) -> &'static str {
        "RANDOM RESPONSE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "RANDOM RESPONSE").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if !ctx.step.bcs.is_empty() {
            return fail("*BOUNDARY in *RANDOM RESPONSE is not supported");
        }
        let model = ctx.prep.model;
        let b = ModalBasis::from_frequency_step(ctx)?;
        let damp = modal_damping(ctx.step, &b)?;
        let tp = 2.0 * std::f64::consts::PI;
        let pts = frequency_points(kw, &b.omega.iter().map(|w| w / tp).collect::<Vec<_>>())?;
        let cases = load_cases(ctx, &b)?;
        let corr = correlations(ctx.step)?;
        let mut psds: BTreeMap<String, Psd> = BTreeMap::new();
        for t in &corr {
            if psds.contains_key(&t.psd) {
                continue;
            }
            let Some(k) = model.extra.iter().find(|k| k.name == "PSD-DEFINITION" && k.param_upper("NAME").as_deref() == Some(t.psd.as_str())) else {
                return fail(format!("*PSD-DEFINITION, NAME={} missing", t.psd));
            };
            psds.insert(t.psd.clone(), Psd::parse(k)?);
        }
        // response units: (weight term, modal load, base direction)
        let mut units: Vec<(usize, Vec<f64>, Option<usize>)> = Vec::new();
        for (ti, t) in corr.iter().enumerate() {
            let Some(parts) = cases.get(&t.case) else {
                return fail(format!("*CORRELATION: load case {} has no loads (LOAD CASE={})", t.case, t.case));
            };
            if t.uncorrelated {
                units.extend(parts.iter().map(|p| (ti, p.p.clone(), p.dir)));
            } else {
                let mut s = vec![0.0; b.len()];
                for p in parts {
                    s.iter_mut().zip(&p.p).for_each(|(a, x)| *a += x);
                }
                // fully correlated: the base direction applies if the load case is only a base motion
                let dir = if parts.len() == 1 { parts[0].dir } else { None };
                units.push((ti, s, dir));
            }
        }
        for c in cases.keys() {
            if !corr.iter().any(|t| t.case == *c) {
                return fail(format!("load case {c} is not used in any *CORRELATION"));
            }
        }
        let base_dirs: Vec<usize> = units.iter().filter_map(|u| u.2).collect();
        let dim = model.dim();
        let rot = has_rotations(ctx);
        let nn = model.nodes.len();
        let z = vec![[0.0; SLOTS]; nn];
        let zero = plain_frame(ctx, &b.dm, &z, 0.0, -1, "Increment      0: Frequency =    0.000".into())?;
        let mut frames = Vec::new();
        {
            let mut fr = zero.clone();
            for (n, r) in [("RU", "RUR"), ("V", "VR"), ("RV", "RVR"), ("A", "AR"), ("RA", "RAR")] {
                add_node_field(&mut fr, dim, rot, n, r, &z);
            }
            if !base_dirs.is_empty() {
                for (n, r) in [("TU", "TUR"), ("RTU", "RTUR"), ("TA", "TAR"), ("RTA", "RTAR")] {
                    add_node_field(&mut fr, dim, rot, n, r, &z);
                }
            }
            frames.push(fr);
        }
        let mut rms: Vec<Rms> = (0..5).map(|_| Rms { prev: None, acc: None }).collect();
        for (k, &f) in pts.iter().enumerate() {
            let w = tp * f;
            // weight Re(c Ψ(f)) per term
            let wts: Vec<f64> = corr.iter().map(|t| (t.c * psds[&t.psd].at(f)).re).collect();
            // response per unit: relative displacement and (for base motion) total
            let mut resp: Vec<(f64, FrameData, FrameData)> = Vec::new();
            let mut tot: Vec<(f64, FrameData, FrameData)> = Vec::new();
            for (ti, p, dir) in &units {
                let q: Vec<Cx> = (0..b.len())
                    .map(|i| {
                        let om = b.omega[i];
                        Cx::new(p[i], 0.0) / Cx::new(om * om - w * w, 2.0 * damp.zeta[i] * om * w + damp.structural[i] * om * om)
                    })
                    .collect();
                let re = b.superpose(&q.iter().map(|x| x.re).collect::<Vec<_>>());
                let im = b.superpose(&q.iter().map(|x| x.im).collect::<Vec<_>>());
                let desc = format!("Increment {:>6}: Frequency = {:>10}", k + 1, super::frequency::g5(f));
                let fre = plain_frame(ctx, &b.dm, &re, f, -1, desc)?;
                let fim = plain_frame(ctx, &b.dm, &im, f, -1, String::new())?;
                if !base_dirs.is_empty() {
                    // total displacement u + r u_g with u_g = ü_g/(−Ω²) (ü_g = 1 per unit)
                    let (mut tr, ti2) = (re.clone(), im.clone());
                    if let Some(d) = dir {
                        if w > 0.0 {
                            super::modal::basis::add_rigid(&mut tr, *d, -1.0 / (w * w));
                        }
                    }
                    let mut a = zero_like(&fre);
                    a.u = tr;
                    let mut bb = zero_like(&fre);
                    bb.u = ti2;
                    tot.push((wts[*ti], a, bb));
                }
                resp.push((wts[*ti], fre, fim));
            }
            let mut fr = psd_frame(&resp);
            let ru = rms[0].add(f, &fr);
            let su = fr.u.clone();
            let scale = |v: &[NodeVec], c: f64| -> Vec<NodeVec> { v.iter().map(|x| x.map(|y| c * y)).collect() };
            let (sv, sa) = (scale(&su, w * w), scale(&su, w.powi(4)));
            let mut vf = zero_like(&fr);
            vf.u = sv.clone();
            let mut af = zero_like(&fr);
            af.u = sa.clone();
            let rv = rms[1].add(f, &vf);
            let ra = rms[2].add(f, &af);
            add_node_field(&mut fr, dim, rot, "RU", "RUR", &ru.u);
            add_node_field(&mut fr, dim, rot, "V", "VR", &sv);
            add_node_field(&mut fr, dim, rot, "RV", "RVR", &rv.u);
            add_node_field(&mut fr, dim, rot, "A", "AR", &sa);
            add_node_field(&mut fr, dim, rot, "RA", "RAR", &ra.u);
            if !base_dirs.is_empty() {
                let pt = psd_frame(&tot);
                let ta = scale(&pt.u, w.powi(4));
                let rtu = rms[3].add(f, &pt);
                let mut taf = zero_like(&pt);
                taf.u = ta.clone();
                let rta = rms[4].add(f, &taf);
                add_node_field(&mut fr, dim, rot, "TU", "TUR", &pt.u);
                add_node_field(&mut fr, dim, rot, "RTU", "RTUR", &rtu.u);
                add_node_field(&mut fr, dim, rot, "TA", "TAR", &ta);
                add_node_field(&mut fr, dim, rot, "RTA", "RTAR", &rta.u);
            }
            // RMS for the element fields (RS, RE, ...)
            for (e, fields) in ru.elem.into_iter().enumerate() {
                for mut fl in fields {
                    fl.name = format!("R{}", fl.name);
                    fl.labels = fl.labels.iter().map(|l| format!("R{l}")).collect();
                    fr.elem[e].push(fl);
                }
            }
            frames.push(fr);
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
