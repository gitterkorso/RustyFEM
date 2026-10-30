//! *COMPLEX FREQUENCY: complex eigenvalues for the damped system
//! (μ² M + μ C + K + i S) φ = 0, projected onto the eigenmodes from the
//! preceding *FREQUENCY step.
//!
//! With the modal basis Φ (φᵀMφ = 1, also internal degrees of freedom, `modal_basis.rs`) the
//! projected problem becomes (μ² I + μ Cᵣ + Kᵣ + i Sᵣ) q = 0 with Kᵣ = diag(ωᵢ²),
//! Cᵣ = Φᵀ C Φ and Sᵣ = Φᵀ S Φ. The damping:
//!
//! - dashpots (*DASHPOT, `Interaction::damping`),
//! - the material's *DAMPING, ALPHA/BETA (C = αM + βK per element) and
//!   STRUCTURAL (S = s K per element), the same matrices as
//!   *STEADY STATE DYNAMICS, DIRECT (`steady_state::damping_matrices`),
//!   except the transverse shear stiffness of the Timoshenko beams, which is not included in
//!   the BETA and STRUCTURAL damping (`complex_beam_shear.rs`).
//!
//! The quadratic problem is written in first-order form
//! A = [[0, I], [−(Kᵣ + i Sᵣ), −Cᵣ]] and solved densely (`complex_dense.rs`).
//! Eigenvalues with positive imaginary part are sorted by increasing imaginary part.
//! The data line: number of eigenvalues, lowest and highest frequency (Hz).
//!
//! Output: frame 0 (the base state, zeros), then one frame
//! per mode with `value` = the mode number. U is the real part of the mode shape and U_IM its
//! imaginary part (conjugate data), normalized so that the largest component
//! (translations and rotations) is 1 + 0i. History EIGREAL (Re μ),
//! EIGIMAG (Im μ, rad/time), EIGFREQ (Im μ / 2π) and DAMPRATIO (−2 Re μ / Im μ)
//! per mode number.
//!
//! Friction-induced unsymmetric stiffness (contact with *FRICTION) is not included:
//! Kᵣ is the diagonal stiffness of the modes.

#[path = "complex_dense.rs"]
pub mod dense;
// area coupled2: the transverse shear of the beams is not included in the BETA/STRUCTURAL damping
#[path = "complex_beam_shear.rs"]
mod beam_shear;

use super::api::{StepCtx, StepType};
use super::frequency::{g5, HISTORY_REGION};
use super::modal::basis::ModalBasis;
use super::modal::{add_node_field, has_rotations, plain_frame};
use crate::fem::assembly::{DofMap, NONE};
use crate::fem::dof::{slot_of, NodeVec, SLOTS};
use crate::fem::model::{parse_f64, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use dense::Cx;

pub struct ComplexFrequency;

pub static STEP_TYPES: &[&dyn StepType] = &[&ComplexFrequency];

/// Frame for a complex nodal field u = re + i·im (U/UR and U_IM/UR_IM plus
/// the element fields and their _IM parts).
pub fn complex_frame(ctx: &StepCtx, dm: &DofMap, re: &[NodeVec], im: &[NodeVec], value: f64, mode: i64, desc: String) -> Result<FrameData> {
    let dim = ctx.prep.model.dim();
    let rot = has_rotations(ctx);
    let mut fr = plain_frame(ctx, dm, re, value, mode, desc)?;
    let fi = plain_frame(ctx, dm, im, value, mode, String::new())?;
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
    Ok(fr)
}

/// Projected damping matrices (Cᵣ, Sᵣ), dense n × n (row-major).
pub fn projected_damping(ctx: &StepCtx, b: &ModalBasis) -> Result<(Vec<f64>, Vec<f64>)> {
    let p = ctx.prep;
    let n = b.len();
    let ns = b.all.n_solve();
    let (cm, sm, _) = super::modal::steady_state::damping_matrices(p, &b.all)?;
    let mut cr = vec![0.0; n * n];
    let mut sr = vec![0.0; n * n];
    let xs: Vec<&[f64]> = b.x.iter().map(|x| &x[..ns]).collect();
    let cx: Vec<Vec<f64>> = xs.iter().map(|x| cm.mul(x)).collect();
    let sx: Vec<Vec<f64>> = xs.iter().map(|x| sm.mul(x)).collect();
    for i in 0..n {
        for j in 0..n {
            cr[i * n + j] = dot(xs[i], &cx[j]);
            sr[i * n + j] = dot(xs[i], &sx[j]);
        }
    }
    beam_shear::subtract(ctx, b, &mut cr, &mut sr)?;
    // dashpots: (node, degree of freedom, node, degree of freedom, value)
    for it in &p.interactions {
        for (na, da, nb, db, v) in it.damping() {
            let (ga, gb) = (b.all.idx[na][slot_of(da)], b.all.idx[nb][slot_of(db)]);
            if ga == NONE || gb == NONE || ga >= ns || gb >= ns {
                continue;
            }
            for i in 0..n {
                for j in 0..n {
                    cr[i * n + j] += b.x[i][ga] * v * b.x[j][gb];
                }
            }
        }
    }
    Ok((cr, sr))
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Complex nodal vector Σ φᵢ qᵢ.
pub fn superpose_complex(b: &ModalBasis, q: &[Cx]) -> (Vec<NodeVec>, Vec<NodeVec>) {
    let re: Vec<f64> = q.iter().map(|c| c.re).collect();
    let im: Vec<f64> = q.iter().map(|c| c.im).collect();
    (b.superpose(&re), b.superpose(&im))
}

/// Normalizes so that the largest component (translations and rotations)
/// becomes 1 + 0i.
fn normalize(re: &mut [NodeVec], im: &mut [NodeVec], active: &[u16]) {
    let mut best = (0.0f64, Cx::ZERO);
    for n in 0..re.len() {
        for s in 0..6 {
            let c = Cx::new(re[n][s], im[n][s]);
            if active[n] & (1 << s) != 0 && c.abs() > best.0 * (1.0 + 1e-6) {
                best = (c.abs(), c);
            }
        }
    }
    if best.0 == 0.0 {
        return;
    }
    for n in 0..re.len() {
        for s in 0..SLOTS {
            let c = Cx::new(re[n][s], im[n][s]) / best.1;
            re[n][s] = c.re;
            im[n][s] = c.im;
        }
    }
}

fn data(step: &StepDef, i: usize) -> Result<Option<f64>> {
    let kw = step.procedure.as_ref().unwrap();
    let Some(d) = kw.data.first() else { return Ok(None) };
    match d.fields.get(i).map(|s| s.trim()) {
        None | Some("") => Ok(None),
        Some(s) => parse_f64(s).map(Some).ok_or_else(|| FemError(format!("{}: '{s}' is not a number", kw.location()))),
    }
}

impl StepType for ComplexFrequency {
    fn name(&self) -> &'static str {
        "COMPLEX FREQUENCY"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "COMPLEX FREQUENCY").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        if kw.has("FRICTION DAMPING") && kw.param_upper("FRICTION DAMPING").as_deref() != Some("NO") {
            return kw.err("FRICTION DAMPING=YES is not supported");
        }
        let b = ModalBasis::from_frequency_step(ctx)?;
        let n = b.len();
        let nev = data(ctx.step, 0)?.map(|v| v.round().max(0.0) as usize).filter(|&k| k > 0).unwrap_or(n);
        let (fmin, fmax) = (data(ctx.step, 1)?, data(ctx.step, 2)?);
        let (cr, sr) = projected_damping(ctx, &b)?;
        // first-order form
        let m = 2 * n;
        let mut a = vec![Cx::ZERO; m * m];
        for i in 0..n {
            a[i * m + n + i] = Cx::ONE;
            for j in 0..n {
                let k = if i == j { b.lambda[i] } else { 0.0 };
                a[(n + i) * m + j] = Cx::new(-k, -sr[i * n + j]);
                a[(n + i) * m + n + j] = Cx::new(-cr[i * n + j], 0.0);
            }
        }
        let mut ev: Vec<Cx> = dense::eigenvalues(&a, m).into_iter().filter(|e| e.im > 0.0).collect();
        ev.sort_by(|x, y| x.im.total_cmp(&y.im));
        let tp = 2.0 * std::f64::consts::PI;
        ev.retain(|e| fmin.is_none_or(|f| e.im / tp >= f) && fmax.is_none_or(|f| e.im / tp <= f));
        ev.truncate(nev);
        if ev.is_empty() {
            return fail("*COMPLEX FREQUENCY: no complex eigenvalues in the requested range");
        }
        let nn = ctx.prep.model.nodes.len();
        let z = vec![[0.0; SLOTS]; nn];
        let mut frames = vec![complex_frame(ctx, &b.dm, &z, &z, 0.0, 0, "Increment      0: Base State".into())?];
        let names = ["EIGREAL", "EIGIMAG", "EIGFREQ", "DAMPRATIO"];
        let mut hist: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 4];
        for (k, mu) in ev.iter().enumerate() {
            let v = dense::eigenvector(&a, m, *mu);
            let (mut re, mut im) = superpose_complex(&b, &v[..n]);
            normalize(&mut re, &mut im, &b.dm.active);
            let mode = k + 1;
            let freq = mu.im / tp;
            let desc = format!("Mode {mode:>9}: Freq = {:>11} (cycles/time) Real part = {}", g5(freq), g5(mu.re));
            frames.push(complex_frame(ctx, &b.dm, &re, &im, mode as f64, mode as i64, desc)?);
            let t = mode as f64;
            hist[0].push((t, mu.re));
            hist[1].push((t, mu.im));
            hist[2].push((t, freq));
            hist[3].push((t, if mu.im != 0.0 { -2.0 * mu.re / mu.im } else { 0.0 }));
        }
        for (name, d) in names.iter().zip(hist) {
            ctx.out.history.push((HISTORY_REGION.to_string(), name.to_string(), d));
        }
        ctx.out.increments = 1;
        ctx.out.frames.extend(frames);
        Ok(())
    }
}
