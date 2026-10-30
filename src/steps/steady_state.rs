//! *STEADY STATE DYNAMICS: harmonic response u e^{iΩt} at a number of frequencies.
//!
//! - Modal (default): qᵢ = (φᵢᵀF − Γᵢ ü_g) / (ωᵢ² − Ω² + i(2ζᵢωᵢΩ + sᵢωᵢ²))
//!   with the modes from the preceding *FREQUENCY step (`modal_basis.rs`),
//!   viscous/Rayleigh damping ζ and structural damping s from *MODAL DAMPING.
//! - `DIRECT`: (K − Ω²M + iΩC + iS) u = F is solved directly in all degrees of freedom
//!   that *FREQUENCY uses (including the internal ones of B23/B33), with the same mass matrix.
//!   C = αR M + βR K and S = s K per material (*DAMPING, ALPHA, BETA,
//!   STRUCTURAL). The complex system is written as the real, symmetric and
//!   indefinite system [[A, B], [B, −A]] [x; −y] = [f_r; f_i] (A = K − Ω²M,
//!   B = ΩC + S, u = x + iy) and solved with the sparse indefinite LDLᵀ solver.
//!
//! Frequency points: `INTERVAL=EIGENFREQUENCY` (default for
//! modal, subintervals between the eigenfrequencies in the range) or `RANGE`
//! (default for DIRECT), `FREQUENCY SCALE=LOGARITHMIC` (default) or
//! `LINEAR`, and bias p: ξ = ½ + ½ sign(y)|y|^{1/p}, y = −1 + 2(k−1)/(n−1)
//! (default p = 3 for EIGENFREQUENCY, 1 for RANGE).
//! Loads with AMPLITUDE= get the amplitude's value at the frequency (Hz).
//!
//! *BASE MOTION (modal only): ü_g = scale · amplitude(f) (ACCELERATION), iΩ · ... (VELOCITY)
//! or −Ω² · ... (DISPLACEMENT); U, V, A relative to the base and TU, TV, TA total.
//!
//! Output: frame 0 with zeros ("Frequency = 0.000"), then
//! one frame per frequency with `value` = the frequency. The real part is written to the
//! regular fields (U, UR, V, A, S, ...), the imaginary part to
//! the fields with the suffix `_IM` (U_IM, UR3_IM/UR_IM, V_IM, A_IM, S_IM, ...).
//! V = iΩU and A = −Ω²U.

use super::basis::{base_motions, modal_damping, BaseKind, BaseMotion, ModalBasis};
use super::{add_node_field, amp_value, has_rotations, plain_frame};
use crate::elements::beam_mass;
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, FeModel, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::{factor_sym, SymCsr, Triplets};
use crate::steps::api::{StepCtx, StepType};
use crate::steps::dynamic::amplitude::{amplitudes, LoadEval};
use crate::steps::frequency::{self, AllDofs};

pub struct SteadyState;

/// Frequency points (Hz).
pub fn frequency_points(kw: &RawKeyword, eig_hz: &[f64]) -> Result<Vec<f64>> {
    // DIRECT has no eigenfrequencies: there RANGE is the default
    let default = if kw.has("DIRECT") { "RANGE" } else { "EIGENFREQUENCY" };
    let interval = kw.param_upper("INTERVAL").unwrap_or_else(|| default.into());
    let log = match kw.param_upper("FREQUENCY SCALE") {
        None => true,
        Some(s) if s == "LOGARITHMIC" => true,
        Some(s) if s == "LINEAR" => false,
        Some(s) => return kw.err(format!("FREQUENCY SCALE={s} must be LOGARITHMIC or LINEAR")),
    };
    let eigen = match interval.as_str() {
        "EIGENFREQUENCY" => true,
        "RANGE" => false,
        other => return kw.err(format!("INTERVAL={other} is not supported (EIGENFREQUENCY or RANGE)")),
    };
    let mut pts: Vec<f64> = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() < 2 {
            return kw.err_at(d.line, "write 'lowest frequency, highest frequency[, number of points[, bias]]'");
        }
        let (f1, f2) = (v[0], v[1]);
        let n = v.get(2).copied().filter(|&x| x >= 1.0).map(|x| x.round() as usize).unwrap_or(20);
        let bias = v.get(3).copied().filter(|&x| x > 0.0).unwrap_or(if eigen { 3.0 } else { 1.0 });
        if log && f1 <= 0.0 {
            return kw.err_at(d.line, "logarithmic scale requires a positive lowest frequency");
        }
        let mut edges = vec![f1];
        if eigen {
            edges.extend(eig_hz.iter().copied().filter(|&f| f > f1 && f < f2));
        }
        edges.push(f2);
        let map = |f: f64| if log { f.log10() } else { f };
        let unmap = |x: f64| if log { 10f64.powf(x) } else { x };
        for w in edges.windows(2) {
            let (a, b) = (map(w[0]), map(w[1]));
            for k in 0..n {
                if n == 1 {
                    pts.push(w[0]);
                    break;
                }
                let y = -1.0 + 2.0 * k as f64 / (n - 1) as f64;
                let xi = 0.5 + 0.5 * y.signum() * y.abs().powf(1.0 / bias);
                let f = if k == 0 { w[0] } else if k == n - 1 { w[1] } else { unmap(a + (b - a) * xi) };
                if pts.last().is_none_or(|&l| (f - l).abs() > 1e-12 * f.abs().max(1.0)) {
                    pts.push(f);
                }
            }
        }
    }
    Ok(pts)
}

/// Complex number (re, im).
#[derive(Debug, Clone, Copy, Default)]
struct C(f64, f64);

impl C {
    #[cfg(test)]
    fn mul(self, o: C) -> C {
        C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0)
    }
    fn div(self, o: C) -> C {
        let d = o.0 * o.0 + o.1 * o.1;
        C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d)
    }
}

/// The base acceleration ü_g (complex) per direction at frequency f (Hz).
fn base_accel(model: &FeModel, bms: &[BaseMotion], f: f64, time0: f64) -> Result<[C; 3]> {
    let amps = amplitudes(model)?;
    let w = 2.0 * std::f64::consts::PI * f;
    let mut a = [C::default(); 3];
    for b in bms {
        let v = b.scale * amp_value(&amps, &b.amplitude, f, time0 + f)?;
        let c = match b.kind {
            BaseKind::Acceleration => C(v, 0.0),
            BaseKind::Velocity => C(0.0, w * v),
            BaseKind::Displacement => C(-w * w * v, 0.0),
        };
        a[b.dir].0 += c.0;
        a[b.dir].1 += c.1;
    }
    Ok(a)
}

/// Frame for a complex response u = re + i·im at frequency f.
#[allow(clippy::too_many_arguments)]
fn complex_frame(ctx: &StepCtx, dm: &DofMap, re: &[NodeVec], im: &[NodeVec], f: f64, k: usize, ag: Option<&[C; 3]>) -> Result<FrameData> {
    let dim = ctx.prep.model.dim();
    let rot = has_rotations(ctx);
    let desc = if k == 0 { "Increment 0: Frequency = 0.000".to_string() } else { format!("Increment {k}: Frequency = {f:.4}") };
    let mut fr = plain_frame(ctx, dm, re, f, -1, desc)?;
    let fi = plain_frame(ctx, dm, im, f, -1, String::new())?;
    // the imaginary part: displacement and element output
    let mut tmp = FrameData::default();
    add_node_field(&mut tmp, dim, rot, "U", "UR", im);
    for (name, labels, vals) in tmp.node_extra {
        fr.node_extra.push((format!("{name}_IM"), labels, vals));
    }
    for (e, fields) in fi.elem.into_iter().enumerate() {
        for mut fld in fields {
            fld.name = format!("{}_IM", fld.name);
            fr.elem[e].push(fld);
        }
    }
    // V = iΩU, A = −Ω²U
    let w = 2.0 * std::f64::consts::PI * f;
    let scale = |src: &[NodeVec], c: f64| -> Vec<NodeVec> { src.iter().map(|x| x.map(|y| c * y)).collect() };
    let (v_re, v_im) = (scale(im, -w), scale(re, w));
    let (a_re, a_im) = (scale(re, -w * w), scale(im, -w * w));
    let mut push = |name: &str, rname: &str, r: &[NodeVec], i: &[NodeVec]| {
        let mut tmp = FrameData::default();
        add_node_field(&mut tmp, dim, rot, name, rname, r);
        let n0 = tmp.node_extra.len();
        add_node_field(&mut tmp, dim, rot, name, rname, i);
        for (j, x) in tmp.node_extra.into_iter().enumerate() {
            if j < n0 {
                fr.node_extra.push(x);
            } else {
                fr.node_extra.push((format!("{}_IM", x.0), x.1, x.2));
            }
        }
    };
    push("V", "VR", &v_re, &v_im);
    push("A", "AR", &a_re, &a_im);
    if let Some(ag) = ag {
        // total values: u_g = ü_g/(−Ω²), u̇_g = ü_g/(iΩ)
        let mut tot = |name: &str, r: &[NodeVec], i: &[NodeVec], g: &dyn Fn(C) -> C| {
            let (mut tr, mut ti) = (r.to_vec(), i.to_vec());
            for d in 0..dim {
                let c = g(ag[d]);
                super::basis::add_rigid(&mut tr, d, c.0);
                super::basis::add_rigid(&mut ti, d, c.1);
            }
            let mut tmp = FrameData::default();
            add_node_field(&mut tmp, dim, false, name, "", &tr);
            add_node_field(&mut tmp, dim, false, name, "", &ti);
            let mut it = tmp.node_extra.into_iter();
            fr.node_extra.extend(it.next());
            if let Some(x) = it.next() {
                fr.node_extra.push((format!("{name}_IM"), x.1, x.2));
            }
        };
        if w > 0.0 {
            tot("TU", re, im, &|c: C| c.div(C(-w * w, 0.0)));
            tot("TV", &v_re, &v_im, &|c: C| c.div(C(0.0, w)));
        }
        tot("TA", &a_re, &a_im, &|c: C| c);
    }
    Ok(fr)
}

/// Rayleigh and structural damping (αR, βR, s) for the material `mi` (*DAMPING).
fn material_damping(model: &FeModel, mi: Option<usize>) -> Result<(f64, f64, f64)> {
    let Some(m) = mi.map(|i| &model.materials[i]) else { return Ok((0.0, 0.0, 0.0)) };
    let Some(kw) = m.behavior("DAMPING") else { return Ok((0.0, 0.0, 0.0)) };
    if kw.has("COMPOSITE") {
        return kw.err("COMPOSITE damping is not supported");
    }
    let num = |k: &str| -> Result<f64> {
        match kw.param(k) {
            Some(v) => parse_f64(v).ok_or_else(|| FemError(format!("{}: {k}={v} is not a number", kw.location()))),
            None => Ok(0.0),
        }
    };
    Ok((num("ALPHA")?, num("BETA")?, num("STRUCTURAL")?))
}

/// The element's degree-of-freedom numbers in `all` (with internal degrees of freedom).
fn elem_map(p: &Prepared, all: &AllDofs, e: usize) -> Vec<usize> {
    let mut v: Vec<usize> = p.edofs[e].iter().map(|&(n, s)| all.idx[n][s]).collect();
    if all.elem_int[e] != NONE {
        v.extend((0..beam_mass::N_INTERNAL).map(|k| all.elem_int[e] + k));
    }
    v
}

/// The damping matrices C = Σ (αR Mₑ + βR Kₑ) and S = Σ s Kₑ in the `all` numbering
/// (only free and internal degrees of freedom).
pub fn damping_matrices(p: &Prepared, all: &AllDofs) -> Result<(SymCsr, SymCsr, bool)> {
    let m = p.model;
    let ns = all.n_solve();
    let mut tc = Triplets::new(ns);
    let mut ts = Triplets::new(ns);
    let mut any = false;
    for (e, el) in m.elements.iter().enumerate() {
        let mi = el.section.and_then(|s| m.sections[s].material_index);
        let (ar, br, s) = material_damping(m, mi)?;
        if ar == 0.0 && br == 0.0 && s == 0.0 {
            continue;
        }
        any = true;
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        let et = el.etype;
        let nd = p.edofs[e].len();
        let mut me = if ar != 0.0 && ctx.density > 0.0 { et.mass(&ctx, frequency::mass_kind(et)).map_err(wrap)? } else { vec![0.0; nd * nd] };
        let mut ke = if br != 0.0 || s != 0.0 { et.stiffness(&ctx).map_err(wrap)? } else { vec![0.0; nd * nd] };
        let map = elem_map(p, all, e);
        let nt = map.len();
        if nt > nd {
            // internal degrees of freedom (B23/B33): extend and add their contribution
            let grow = |a: &[f64]| {
                let mut b = vec![0.0; nt * nt];
                for i in 0..nd {
                    b[i * nt..i * nt + nd].copy_from_slice(&a[i * nd..(i + 1) * nd]);
                }
                b
            };
            me = grow(&me);
            ke = grow(&ke);
            if let Some(bi) = beam_mass::internal_dofs(et.name(), &ctx, None) {
                let bi = bi.map_err(wrap)?;
                me.iter_mut().zip(&bi.m).for_each(|(a, b)| *a += b);
                ke.iter_mut().zip(&bi.k).for_each(|(a, b)| *a += b);
            }
        }
        for a in 0..nt {
            for c in 0..nt {
                let (ia, ic) = (map[a], map[c]);
                if ia == NONE || ic == NONE || ia >= ns || ic >= ns || ia < ic {
                    continue;
                }
                let cv = ar * me[a * nt + c] + br * ke[a * nt + c];
                if cv != 0.0 {
                    tc.add(ia, ic, cv);
                }
                if s != 0.0 && ke[a * nt + c] != 0.0 {
                    ts.add(ia, ic, s * ke[a * nt + c]);
                }
            }
        }
    }
    Ok((tc.build(), ts.build(), any))
}

/// Loads and base motion at frequency f.
struct Excitation<'a> {
    lev: LoadEval<'a>,
    bms: Vec<BaseMotion>,
}

impl Excitation<'_> {
    fn force(&self, ctx: &StepCtx, f: f64) -> Result<Vec<NodeVec>> {
        let nn = ctx.prep.model.nodes.len();
        let z = vec![[0.0; SLOTS]; nn];
        assembly::external(ctx.prep, &self.lev.cloads_at(f), &self.lev.dloads_at(f), &z, false, ctx.state.time)
    }
}

impl StepType for SteadyState {
    fn name(&self) -> &'static str {
        "STEADY STATE DYNAMICS"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "STEADY STATE DYNAMICS").then_some(0)
    }

    fn uses_amplitudes(&self) -> bool {
        true
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if !ctx.step.bcs.is_empty() {
            return fail("*BOUNDARY in *STEADY STATE DYNAMICS is not supported – use *BASE MOTION");
        }
        if kw.has("SUBSPACE PROJECTION") || kw.has("REAL ONLY") {
            return kw.err("SUBSPACE PROJECTION/REAL ONLY is not supported");
        }
        let prep = ctx.prep;
        let model = prep.model;
        let nn = model.nodes.len();
        let mut lev = LoadEval::new(model, ctx.loads, false, 1.0, ctx.state.time)?;
        lev.step_full_at_start = true;
        let ex = Excitation { lev, bms: base_motions(ctx.step)? };
        let with_base = !ex.bms.is_empty();
        let direct = kw.has("DIRECT");
        let tp = 2.0 * std::f64::consts::PI;
        let z = vec![[0.0; SLOTS]; nn];
        let mut frames = Vec::new();
        if !direct {
            let b = ModalBasis::from_frequency_step(ctx)?;
            let damp = modal_damping(ctx.step, &b)?;
            let omega = &b.omega;
            let pts = frequency_points(kw, &omega.iter().map(|w| w / tp).collect::<Vec<_>>())?;
            let gamma: Vec<Vec<f64>> = (0..3).map(|d| if ex.bms.iter().any(|m| m.dir == d) { b.participation(&model.nodes, d) } else { Vec::new() }).collect();
            frames.push(complex_frame(ctx, &b.dm, &z, &z, 0.0, 0, with_base.then_some(&[C::default(); 3]))?);
            for (k, &f) in pts.iter().enumerate() {
                let w = tp * f;
                let fm = b.modal_force(&ex.force(ctx, f)?);
                let ag = base_accel(model, &ex.bms, f, ctx.state.time)?;
                let mut qr = vec![0.0; b.len()];
                let mut qi = vec![0.0; b.len()];
                for i in 0..b.len() {
                    let mut p = C(fm[i], 0.0);
                    for d in 0..3 {
                        if !gamma[d].is_empty() {
                            p.0 -= gamma[d][i] * ag[d].0;
                            p.1 -= gamma[d][i] * ag[d].1;
                        }
                    }
                    let den = C(omega[i] * omega[i] - w * w, 2.0 * damp.zeta[i] * omega[i] * w + damp.structural[i] * omega[i] * omega[i]);
                    let q = p.div(den);
                    qr[i] = q.0;
                    qi[i] = q.1;
                }
                let (re, im) = (b.superpose(&qr), b.superpose(&qi));
                frames.push(complex_frame(ctx, &b.dm, &re, &im, f, k + 1, with_base.then_some(&ag))?);
            }
        } else {
            if with_base {
                // base motion only in modal steps
                return fail("*BASE MOTION cannot be used in *STEADY STATE DYNAMICS, DIRECT");
            }
            let (dm, _) = assembly::dofmap(prep, &ctx.loads.constrained(), ctx.solver)?;
            let all = AllDofs::new(prep, &dm);
            let ns = all.n_solve();
            let kmat = frequency::assemble_stiffness(prep, &dm, &all, ctx.state, ctx.nlgeom)?;
            let mfull = frequency::assemble_mass(prep, &all)?;
            let mmat = mfull.leading(ns);
            let (cmat, smat, _) = damping_matrices(prep, &all)?;
            let pts = frequency_points(kw, &[])?;
            let indef = crate::solver::select(Some("indefinite"))?;
            let blocks: Vec<([f64; 3], Vec<usize>)> =
                all.blocks(prep).into_iter().map(|(c, eqs)| (c, eqs.iter().flat_map(|&q| [2 * q, 2 * q + 1]).collect())).collect();
            frames.push(complex_frame(ctx, &dm, &z, &z, 0.0, 0, None)?);
            for (k, &f) in pts.iter().enumerate() {
                let w = tp * f;
                // [[A, B], [B, −A]], A = K − Ω²M, B = ΩC + S, in the order (x_q, z_q) per degree of freedom
                let mut t = Triplets::new(2 * ns);
                let mut put = |a: &SymCsr, ca: f64, cb: f64| {
                    for r in 0..ns {
                        for idx in a.rowptr[r]..a.rowptr[r + 1] {
                            let c = a.col[idx] as usize;
                            if c > r {
                                continue;
                            }
                            let v = a.val[idx];
                            if ca != 0.0 {
                                t.add(2 * r, 2 * c, ca * v);
                                t.add(2 * r + 1, 2 * c + 1, -ca * v);
                            }
                            if cb != 0.0 {
                                t.add(2 * r + 1, 2 * c, cb * v);
                                if r != c {
                                    t.add(2 * r, 2 * c + 1, cb * v);
                                }
                            }
                        }
                    }
                };
                put(&kmat, 1.0, 0.0);
                put(&mmat, -w * w, 0.0);
                put(&cmat, 0.0, w);
                put(&smat, 0.0, 1.0);
                // viscoelasticity in the frequency domain (*VISCOELASTIC, FREQUENCY=...)
                if let Some((da, db)) = crate::materials::viscoelastic_freq::frequency_matrices(prep, &all, w, ctx.state.time)? {
                    put(&da, 1.0, 0.0);
                    put(&db, 0.0, 1.0);
                }
                let sys = t.build();
                let fac = match factor_sym(&sys, &blocks, indef)? {
                    Ok(f) => f,
                    Err(_) => return fail(format!("*STEADY STATE DYNAMICS, DIRECT: the dynamic stiffness matrix is singular at {f} Hz (resonance without damping?)")),
                };
                let fext = ex.force(ctx, f)?;
                let mut rhs = vec![0.0; 2 * ns];
                for q in 0..ns {
                    let (n, s) = all.owner[q];
                    if n != NONE {
                        rhs[2 * q] = fext[n][s];
                    }
                }
                crate::solver::SystemMatrix::solve(&fac, &mut rhs);
                let xr: Vec<f64> = (0..ns).map(|q| rhs[2 * q]).collect();
                let xi: Vec<f64> = (0..ns).map(|q| -rhs[2 * q + 1]).collect();
                let (re, im) = (all.to_nodes(&xr, nn), all.to_nodes(&xi, nn));
                let mut fr = complex_frame(ctx, &dm, &re, &im, f, k + 1, None)?;
                crate::materials::viscoelastic_freq::correct_stress(prep, &mut fr, &re, &im, w, ctx.state.time)?;
                frames.push(fr);
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_division() {
        let a = C(3.0, -2.0);
        let b = C(1.5, 0.5);
        let q = a.div(b);
        let back = q.mul(b);
        assert!((back.0 - a.0).abs() < 1e-14 && (back.1 - a.1).abs() < 1e-14);
    }
}
