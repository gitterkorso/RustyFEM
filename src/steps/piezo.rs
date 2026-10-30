//! Piezoelectric analysis in *STATIC and *FREQUENCY with the elements in
//! `elements/piezo.rs` (C3D8E, C3D20E, CPE4E, CPS4E).
//!
//! The types claim *STATIC (without NLGEOM) and *FREQUENCY with priority 1
//! and hand over to `static_linear.rs` and `frequency.rs` respectively when
//! the model has no piezoelectric elements.
//!
//! - *STATIC: linear, K = [[K_uu, K_uφ], [K_φu, −K_κ]] (symmetric, indefinite)
//!   solved with the indefinite LDLᵀ solver. Loads *CLOAD, *CECHARGE
//!   (concentrated charge, DOF 9) and boundary conditions on 1–3 and 9.
//! - *FREQUENCY: the potential is condensed statically (K* = K_uu − K_uφ K_φφ⁻¹ K_φu,
//!   no mass on the potential) and K* x = ω² M x is solved densely (Cholesky of M,
//!   Jacobi). The mass is diagonal for first-order elements and consistent
//!   for C3D20E. The mode shapes are mass-normalized; the potential in the mode is
//!   φ = −K_φφ⁻¹ K_φu x.
//!
//! Output: U, RF, EPOT, RCHG (reaction charge, same sign as the reaction forces), S, E, EFLX,
//! EPG; in *FREQUENCY history EIGVAL, EIGFREQ and GM.

use super::api::{StepCtx, StepType};
use super::newton::Controls;
use crate::elements::piezo::{self, PzElem};
use crate::elements::{ElemCtx, ElementType, IpField, MassKind};
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::StepDef;
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen;

pub struct PiezoStatic;
pub struct PiezoFrequency;

pub static STEP_TYPES: &[&dyn StepType] = &[&PiezoStatic, &PiezoFrequency];

/// Slot for the potential.
const PS: usize = 6;

fn has_piezo(p: &Prepared) -> bool {
    p.model.elements.iter().any(|e| piezo::lookup(e.etype.name()).is_some())
}

struct PzModel {
    elems: Vec<&'static PzElem>,
    a: Vec<[[f64; 9]; 9]>,
}

fn prepare(p: &Prepared, time: f64) -> Result<PzModel> {
    let m = p.model;
    let mut elems = Vec::new();
    let mut a = Vec::new();
    for (e, el) in m.elements.iter().enumerate() {
        let Some(pe) = piezo::lookup(el.etype.name()) else {
            return fail(format!("{}: piezoelectric analysis with elements other than C3D8E, C3D20E, CPE4E and CPS4E is not supported yet ({})", m.elem_name(e), el.etype.name()));
        };
        let ctx = p.ctx(e, false, time, 0.0);
        let Some(mi) = ctx.section.and_then(|s| s.material_index) else {
            return fail(format!("{}: has no *SOLID SECTION with material", m.elem_name(e)));
        };
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        a.push(PzElem::constitutive(&ctx, &m.materials[mi], pe.name).map_err(wrap)?);
        elems.push(pe);
    }
    Ok(PzModel { elems, a })
}

fn elem_matrix(pz: &PzModel, ctx: &ElemCtx, e: usize) -> Result<Vec<f64>> {
    pz.elems[e].matrix(ctx, &pz.a[e])
}

fn elem_output(p: &Prepared, pz: &PzModel, u: &[NodeVec], time: f64) -> Result<Vec<Vec<IpField>>> {
    (0..p.model.elements.len())
        .map(|e| {
            let ctx = p.ctx(e, false, time, 0.0);
            pz.elems[e].fields(&ctx, &pz.a[e], &p.gather(e, u))
        })
        .collect()
}

/// Internal "forces" K u per node (charge rows with the same sign convention as RCHG).
fn internal(p: &Prepared, pz: &PzModel, u: &[NodeVec], time: f64) -> Result<Vec<NodeVec>> {
    let mut f = vec![[0.0; SLOTS]; p.model.nodes.len()];
    for e in 0..p.model.elements.len() {
        let ctx = p.ctx(e, false, time, 0.0);
        let k = elem_matrix(pz, &ctx, e)?;
        let ue = p.gather(e, u);
        let n = ue.len();
        for (i, &(nd, s)) in p.edofs[e].iter().enumerate() {
            let v: f64 = (0..n).map(|j| k[i * n + j] * ue[j]).sum();
            f[nd][s] += if s == PS { -v } else { v };
        }
    }
    Ok(f)
}

fn make_frame(p: &Prepared, dm: &DofMap, pz: &PzModel, u: &[NodeVec], rf: &[NodeVec], value: f64, mode: i64, description: String, time: f64) -> Result<FrameData> {
    let elem = elem_output(p, pz, u, time)?;
    let epot: Vec<Vec<f64>> = u.iter().map(|v| vec![v[PS]]).collect();
    let rchg: Vec<Vec<f64>> = rf.iter().map(|v| vec![v[PS]]).collect();
    Ok(FrameData {
        value,
        mode,
        description,
        u: u.to_vec(),
        rf: rf.to_vec(),
        active: dm.active.clone(),
        constrained: dm.constrained.clone(),
        elem,
        node_extra: vec![("EPOT".into(), vec!["EPOT".into()], epot), ("RCHG".into(), vec!["RCHG".into()], rchg)],
    })
}

impl StepType for PiezoStatic {
    fn name(&self) -> &'static str {
        "STATIC (piezoelectric)"
    }

    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "STATIC" && !nlgeom && !step.procedure.as_ref().is_some_and(|p| p.has("RIKS"))).then_some(1)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        if !has_piezo(p) {
            return super::static_linear::STEP_TYPES[0].run(ctx);
        }
        let model = p.model;
        let ctrl = Controls::from_procedure(ctx.step.procedure.as_ref(), ctx.step.max_increments)?;
        let period = ctrl.period;
        let time = ctx.state.time + period;
        if !ctx.loads.dload.is_empty() {
            return fail("distributed loads in piezoelectric *STATIC are not supported yet");
        }
        let pz = prepare(p, time)?;
        let solver = crate::solver::select(Some("indefinite"))?;
        let (dm, mut mat) = assembly::dofmap(p, &ctx.loads.constrained(), solver)?;
        let mut u = ctx.state.u.clone();
        for (n, s, v) in ctx.loads.bc_at(1.0) {
            u[n][s] = v;
        }
        // free DOFs start from the state before the step (linear: total response)
        for e in 0..model.elements.len() {
            let ectx = p.ctx(e, false, time, 0.0);
            let k = elem_matrix(&pz, &ectx, e)?;
            let n = p.edofs[e].len();
            for (i, &(na, sa)) in p.edofs[e].iter().enumerate() {
                for (j, &(nb, sb)) in p.edofs[e].iter().enumerate() {
                    dm.add_entry(&mut *mat, na, sa, nb, sb, k[i * n + j]);
                }
            }
        }
        let mut fext = vec![[0.0; SLOTS]; model.nodes.len()];
        for (n, s, v) in ctx.loads.cloads_at(1.0) {
            fext[n][s] += v;
        }
        let fint0 = internal(p, &pz, &u, time)?;
        if dm.n_eq > 0 {
            if let Err(q) = mat.factor() {
                return Err(assembly::mechanism_message(p, &dm, q));
            }
            // symmetric form: the charge rows with reversed sign
            let mut r: Vec<f64> = dm.owner.iter().map(|&(n, s)| if s == PS { -(fext[n][s] - fint0[n][s]) } else { fext[n][s] - fint0[n][s] }).collect();
            mat.solve(&mut r);
            for (q, &(n, s)) in dm.owner.iter().enumerate() {
                u[n][s] += r[q];
            }
        }
        let fint = internal(p, &pz, &u, time)?;
        let mut rf = vec![[0.0; SLOTS]; model.nodes.len()];
        for n in 0..rf.len() {
            for s in 0..SLOTS {
                if dm.constrained[n] & (1 << s) != 0 {
                    rf[n][s] = fint[n][s] - fext[n][s];
                }
            }
        }
        if ctx.opts.all_frames {
            let zero = vec![[0.0; SLOTS]; model.nodes.len()];
            ctx.out.frames.push(make_frame(p, &dm, &pz, &ctx.state.u, &zero, 0.0, -1, "Increment 0: Step Time = 0.0".into(), ctx.state.time)?);
        }
        ctx.out.frames.push(make_frame(p, &dm, &pz, &u, &rf, period, -1, format!("Increment 1: Step Time = {period:.4}"), time)?);
        ctx.out.increments = 1;
        ctx.state.u = u;
        ctx.state.reaction = rf;
        ctx.state.time = time;
        Ok(())
    }
}

/// Dense Cholesky (lower triangle, row-wise n×n). `None` if the matrix is not positive definite.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d <= 0.0 {
            return None;
        }
        let d = d.sqrt();
        l[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / d;
        }
    }
    Some(l)
}

/// Solves L Lᵀ x = b.
fn chol_solve(l: &[f64], n: usize, b: &mut [f64]) {
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * b[k]).sum();
        b[i] = (b[i] - s) / l[i * n + i];
    }
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * b[k]).sum();
        b[i] = (b[i] - s) / l[i * n + i];
    }
}

impl StepType for PiezoFrequency {
    fn name(&self) -> &'static str {
        "FREQUENCY (piezoelectric)"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "FREQUENCY").then_some(1)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        if !has_piezo(p) {
            return super::frequency::STEP_TYPES[0].run(ctx);
        }
        let model = p.model;
        let kw = ctx.step.procedure.as_ref().unwrap();
        let nmodes = super::frequency::data_field(kw, 0)?.map_or(1, |v| v.max(1.0) as usize);
        if super::frequency::data_field(kw, 1)?.is_some() || super::frequency::data_field(kw, 2)?.is_some() {
            return kw.err("frequency ranges in piezoelectric *FREQUENCY are not supported yet (give only the number of modes)");
        }
        let time = ctx.state.time;
        let pz = prepare(p, time)?;
        let solver = crate::solver::select(Some("indefinite"))?;
        let (dm, _) = assembly::dofmap(p, &ctx.loads.constrained(), solver)?;
        let ne = dm.n_eq;
        let mut k = vec![0.0; ne * ne];
        let mut mm = vec![0.0; ne * ne];
        for e in 0..model.elements.len() {
            let ectx = p.ctx(e, false, time, 0.0);
            let ke = elem_matrix(&pz, &ectx, e)?;
            let kind = if pz.elems[e].n_nodes() > 8 { MassKind::Consistent } else { MassKind::Lumped };
            let me = pz.elems[e].mass(&ectx, kind)?;
            let n = p.edofs[e].len();
            for (i, &(na, sa)) in p.edofs[e].iter().enumerate() {
                let a = dm.eq[na][sa];
                if a == NONE {
                    continue;
                }
                for (j, &(nb, sb)) in p.edofs[e].iter().enumerate() {
                    let b = dm.eq[nb][sb];
                    if b != NONE {
                        k[a * ne + b] += ke[i * n + j];
                        mm[a * ne + b] += me[i * n + j];
                    }
                }
            }
        }
        let iu: Vec<usize> = (0..ne).filter(|&q| dm.owner[q].1 != PS).collect();
        let ip: Vec<usize> = (0..ne).filter(|&q| dm.owner[q].1 == PS).collect();
        let (nu, np) = (iu.len(), ip.len());
        // −K_φφ (positive definite) and K* = K_uu + K_uφ (−K_φφ)⁻¹ K_φu
        let mut kpp = vec![0.0; np * np];
        for (a, &qa) in ip.iter().enumerate() {
            for (b, &qb) in ip.iter().enumerate() {
                kpp[a * np + b] = -k[qa * ne + qb];
            }
        }
        let lp = if np > 0 { Some(cholesky(&kpp, np).ok_or_else(|| FemError("the dielectric matrix of the potential is not positive definite (missing *DIELECTRIC or boundary conditions on 9?)".into()))?) } else { None };
        // X = (−K_φφ)⁻¹ K_φu (np × nu)
        let mut x = vec![0.0; np * nu];
        if let Some(lp) = &lp {
            for (c, &qc) in iu.iter().enumerate() {
                let mut col: Vec<f64> = ip.iter().map(|&qa| k[qa * ne + qc]).collect();
                chol_solve(lp, np, &mut col);
                for a in 0..np {
                    x[a * nu + c] = col[a];
                }
            }
        }
        let mut ks = vec![0.0; nu * nu];
        for (r, &qr) in iu.iter().enumerate() {
            for (c, &qc) in iu.iter().enumerate() {
                let mut v = k[qr * ne + qc];
                for (a, &qa) in ip.iter().enumerate() {
                    v += k[qr * ne + qa] * x[a * nu + c];
                }
                ks[r * nu + c] = v;
            }
        }
        let mut m = vec![0.0; nu * nu];
        for (r, &qr) in iu.iter().enumerate() {
            for (c, &qc) in iu.iter().enumerate() {
                m[r * nu + c] = mm[qr * ne + qc];
            }
        }
        let lm = cholesky(&m, nu).ok_or_else(|| FemError("the mass matrix is not positive definite (missing *DENSITY?)".into()))?;
        // A = L⁻¹ K* L⁻ᵀ
        let mut a = ks.clone();
        for c in 0..nu {
            // column c: L⁻¹ ks[:, c]
            for i in 0..nu {
                let s: f64 = (0..i).map(|kk| lm[i * nu + kk] * a[kk * nu + c]).sum();
                a[i * nu + c] = (a[i * nu + c] - s) / lm[i * nu + i];
            }
        }
        for r in 0..nu {
            for i in 0..nu {
                let s: f64 = (0..i).map(|kk| lm[i * nu + kk] * a[r * nu + kk]).sum();
                a[r * nu + i] = (a[r * nu + i] - s) / lm[i * nu + i];
            }
        }
        for i in 0..nu {
            for j in 0..i {
                let v = 0.5 * (a[i * nu + j] + a[j * nu + i]);
                a[i * nu + j] = v;
                a[j * nu + i] = v;
            }
        }
        let (lam, vec) = eigen::jacobi(&a, nu);
        let mut order: Vec<usize> = (0..nu).collect();
        order.sort_by(|&i, &j| lam[i].total_cmp(&lam[j]));
        let nn = model.nodes.len();
        let zero = vec![[0.0; SLOTS]; nn];
        ctx.out.frames.push(make_frame(p, &dm, &pz, &ctx.state.u, &zero, 0.0, 0, "Increment      0: Base State".into(), time)?);
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut hist: Vec<Vec<(f64, f64)>> = vec![Vec::new(); 3];
        for (i, &j) in order.iter().take(nmodes).enumerate() {
            let mode = i + 1;
            // x = L⁻ᵀ y
            let mut y: Vec<f64> = (0..nu).map(|r| vec[r * nu + j]).collect();
            for r in (0..nu).rev() {
                let s: f64 = (r + 1..nu).map(|kk| lm[kk * nu + r] * y[kk]).sum();
                y[r] = (y[r] - s) / lm[r * nu + r];
            }
            let mut phi = vec![[0.0; SLOTS]; nn];
            for (r, &q) in iu.iter().enumerate() {
                let (n, s) = dm.owner[q];
                phi[n][s] = y[r];
            }
            for (aa, &q) in ip.iter().enumerate() {
                let (n, s) = dm.owner[q];
                phi[n][s] = (0..nu).map(|c| x[aa * nu + c] * y[c]).sum::<f64>();
            }
            let l = lam[j];
            let freq = l.max(0.0).sqrt() / two_pi;
            let desc = format!("Mode {mode:>9}: Value = {:>11} Freq = {:>11} (cycles/time)", super::frequency::g5(l), super::frequency::g5(freq));
            ctx.out.frames.push(make_frame(p, &dm, &pz, &phi, &zero, mode as f64, mode as i64, desc, time)?);
            hist[0].push((mode as f64, l));
            hist[1].push((mode as f64, freq));
            hist[2].push((mode as f64, 1.0));
        }
        for (name, data) in ["EIGVAL", "EIGFREQ", "GM"].iter().zip(hist) {
            ctx.out.history.push((super::frequency::HISTORY_REGION.to_string(), name.to_string(), data));
        }
        ctx.out.increments = 1;
        Ok(())
    }
}
