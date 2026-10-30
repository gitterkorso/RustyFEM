//! *CONTOUR INTEGRAL, XFEM för stationära XFEM-sprickor (`elements/xfem.rs`).
//!
//! Steg-typen tar *STATIC-steg med en XFEM-konturintegral, kör den vanliga
//! statiska steg-typen och räknar sedan integralerna med domänintegralmetoden
//! som `fracture.rs`, men på XFEM-elementens internt genererade
//! integrationspunkter (med berikningen i ∇u):
//!
//! - Frontpunkterna är frontlinjens (φ = ψ = 0) skärningar med spetselementens
//!   ytor, ordnade längs t = ∇φ × ∇ψ (`<SPRICKA>_XFEM_<k>`), med
//!   history `X/Y/Z at <SPRICKA>_XFEM_<k>`.
//! - Kontur 1: q = 1 i spetselementens noder (de element som innehåller
//!   frontpunkten), kontur c + 1: även i noderna till elementen som delar en
//!   nod med kontur c. Längs fronten fördelas q med en hattfunktion mellan
//!   grannfrontpunkterna och integralen normeras med ∫ q ds.
//! - J, K1, K2, K3 (interaktionsintegral, hjälpfält i plan töjning), JKs och
//!   Cpd (MTS) med namnen `<storhet> at <SPRICKA>_XFEM_<k>_Contour_<c>`.

use super::api::{StepCtx, StepType};
use super::fracture::aux::{aux, AuxMode, Elastic};
use super::fracture::{mts_angle, Request, REGION};
use crate::elements::xfem::{self, geom, physical, Crack};
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::model::{RawKeyword, StepDef};
use crate::model::{fail, Result};
use std::collections::{BTreeSet, HashMap};

pub struct XfemContour;

pub static STEP_TYPES: &[&dyn StepType] = &[&XfemContour];

fn is_xfem(k: &RawKeyword) -> bool {
    k.name == "CONTOUR INTEGRAL" && k.has("XFEM")
}

impl StepType for XfemContour {
    fn name(&self) -> &'static str {
        "STATIC med *CONTOUR INTEGRAL, XFEM"
    }

    fn claim(&self, step: &StepDef, nlgeom: bool) -> Option<i32> {
        let riks = step.procedure.as_ref().is_some_and(|p| p.has("RIKS"));
        (step.procedure_name() == "STATIC" && !nlgeom && !riks && !step.perturbation && step.extra.iter().any(is_xfem)).then_some(6)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let t0 = ctx.state.time;
        super::static_linear::Static.run(ctx)?;
        let period = ctx.state.time - t0;
        let prep = ctx.prep;
        let mut out = Vec::new();
        let regular: Vec<Request> = ctx.step.extra.iter().filter(|k| k.name == "CONTOUR INTEGRAL" && !k.has("XFEM")).map(|k| Request::parse(prep.model, k)).collect::<Result<_>>()?;
        if !regular.is_empty() {
            out.extend(super::fracture::evaluate(prep, &ctx.state.u, &ctx.state.elem_state, ctx.state.time, &regular)?);
        }
        for k in ctx.step.extra.iter().filter(|k| is_xfem(k)) {
            out.extend(evaluate(prep, &ctx.state.u, ctx.state.time, k)?);
        }
        for (name, v) in out {
            ctx.out.history.push((REGION.to_string(), name, vec![(period, v)]));
        }
        Ok(())
    }
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-300);
    [a[0] / n, a[1] / n, a[2] / n]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Punkt på en yta med φ = ψ = 0 (Newton i ytans parametrar), om den finns.
fn face_point(x: &[[f64; 3]], phi: &[f64], psi: &[f64]) -> Option<[f64; 3]> {
    let n = x.len();
    // bilinjär fyrhörning eller linjär triangel i (s, t)
    let shape = |s: f64, t: f64| -> (Vec<f64>, Vec<[f64; 2]>) {
        if n == 4 {
            let a = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
            let v = (0..4).map(|i| 0.25 * (1.0 + a[i][0] * s) * (1.0 + a[i][1] * t)).collect();
            let d = (0..4).map(|i| [0.25 * a[i][0] * (1.0 + a[i][1] * t), 0.25 * a[i][1] * (1.0 + a[i][0] * s)]).collect();
            (v, d)
        } else {
            (vec![1.0 - s - t, s, t], vec![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
        }
    };
    let (mut s, mut t) = if n == 4 { (0.0, 0.0) } else { (1.0 / 3.0, 1.0 / 3.0) };
    for _ in 0..30 {
        let (v, d) = shape(s, t);
        let f1: f64 = v.iter().zip(phi).map(|(a, b)| a * b).sum();
        let f2: f64 = v.iter().zip(psi).map(|(a, b)| a * b).sum();
        let j11: f64 = d.iter().zip(phi).map(|(a, b)| a[0] * b).sum();
        let j12: f64 = d.iter().zip(phi).map(|(a, b)| a[1] * b).sum();
        let j21: f64 = d.iter().zip(psi).map(|(a, b)| a[0] * b).sum();
        let j22: f64 = d.iter().zip(psi).map(|(a, b)| a[1] * b).sum();
        let det = j11 * j22 - j12 * j21;
        if det.abs() < 1e-300 {
            return None;
        }
        let ds = (f1 * j22 - f2 * j12) / det;
        let dt = (j11 * f2 - j21 * f1) / det;
        s -= ds;
        t -= dt;
        if ds.abs() + dt.abs() < 1e-13 {
            break;
        }
    }
    let tol = 1e-8;
    let inside = if n == 4 { s.abs() <= 1.0 + tol && t.abs() <= 1.0 + tol } else { s >= -tol && t >= -tol && s + t <= 1.0 + tol };
    if !inside {
        return None;
    }
    let (v, _) = shape(s, t);
    Some(geom::bary_point(x, &v))
}

struct FrontPoint {
    p: [f64; 3],
    elems: Vec<usize>,
    frame: [[f64; 3]; 3],
    /// Läget längs fronten (ordningsriktningen).
    s: f64,
}

fn front_points(prep: &Prepared, crack: &'static Crack) -> Result<Vec<FrontPoint>> {
    let m = prep.model;
    let tips = xfem::tip_elements(m, crack);
    if tips.is_empty() {
        return fail(format!("*CONTOUR INTEGRAL, XFEM: sprickan {} har ingen front inne i domänen", crack.name));
    }
    let mut pts: Vec<FrontPoint> = Vec::new();
    let mut gphi_sum = [0.0; 3];
    let mut gpsi_sum = [0.0; 3];
    let mut grads: HashMap<usize, ([f64; 3], [f64; 3])> = HashMap::new();
    for (e, phi, psi) in &tips {
        let (r, nodes) = &crack.conn[e];
        let x: Vec<[f64; 3]> = nodes.iter().map(|&n| m.nodes[n]).collect();
        let (_, dndx, _) = physical(*r, &x, r.centroid())?;
        let mut gp = [0.0; 3];
        let mut gs = [0.0; 3];
        for i in 0..r.nn() {
            for k in 0..3 {
                gp[k] += dndx[i][k] * phi[i];
                gs[k] += dndx[i][k] * psi[i];
            }
        }
        grads.insert(*e, (gp, gs));
        for k in 0..3 {
            gphi_sum[k] += gp[k];
            gpsi_sum[k] += gs[k];
        }
        let h = x.iter().map(|p| dot(sub(*p, x[0]), sub(*p, x[0])).sqrt()).fold(0.0, f64::max);
        for f in r.faces() {
            let fx: Vec<[f64; 3]> = f.iter().map(|&i| x[i]).collect();
            let fp: Vec<f64> = f.iter().map(|&i| phi[i]).collect();
            let fs: Vec<f64> = f.iter().map(|&i| psi[i]).collect();
            let Some(p) = face_point(&fx, &fp, &fs) else { continue };
            match pts.iter_mut().find(|q| dot(sub(q.p, p), sub(q.p, p)).sqrt() < 1e-6 * h) {
                Some(q) => {
                    if !q.elems.contains(e) {
                        q.elems.push(*e)
                    }
                }
                None => pts.push(FrontPoint { p, elems: vec![*e], frame: [[0.0; 3]; 3], s: 0.0 }),
            }
        }
    }
    let t = unit(cross(gphi_sum, gpsi_sum));
    pts.sort_by(|a, b| dot(a.p, t).partial_cmp(&dot(b.p, t)).unwrap());
    // tetraedrar: frontpunkterna fördelas jämnt mellan ändpunkterna
    // (lika många som skärningarna); spetselementen är de som spänner över punkten
    if tips.iter().any(|(e, _, _)| crack.conn[e].0 == geom::Ref::Tet4) && pts.len() > 2 {
        let (a, b) = (pts[0].p, pts[pts.len() - 1].p);
        let n = pts.len();
        let span: Vec<(usize, f64, f64)> = tips
            .iter()
            .map(|(e, _, _)| {
                let s: Vec<f64> = crack.conn[e].1.iter().map(|&k| dot(m.nodes[k], t)).collect();
                (*e, s.iter().cloned().fold(f64::MAX, f64::min), s.iter().cloned().fold(f64::MIN, f64::max))
            })
            .collect();
        let tol = 1e-9 * (dot(b, t) - dot(a, t)).abs().max(1e-30);
        for (k, q) in pts.iter_mut().enumerate() {
            let f = k as f64 / (n - 1) as f64;
            q.p = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
            let s = dot(q.p, t);
            q.elems = span.iter().filter(|(_, lo, hi)| s >= lo - tol && s <= hi + tol).map(|x| x.0).collect();
        }
    }
    for q in pts.iter_mut() {
        q.s = dot(q.p, t);
        let mut gp = [0.0; 3];
        let mut gs = [0.0; 3];
        for e in &q.elems {
            let (a, b) = grads[e];
            for k in 0..3 {
                gp[k] += a[k];
                gs[k] += b[k];
            }
        }
        let e2 = unit(gp);
        let e1 = unit(sub(gs, {
            let d = dot(gs, e2);
            [e2[0] * d, e2[1] * d, e2[2] * d]
        }));
        q.frame = [e1, e2, cross(e1, e2)];
    }
    Ok(pts)
}

fn rot(r: &[[f64; 3]; 3], t: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    s += r[i][k] * t[k][l] * r[j][l];
                }
            }
            o[i][j] = s;
        }
    }
    o
}

fn rotv(r: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(r[0], v), dot(r[1], v), dot(r[2], v)]
}

fn evaluate(prep: &Prepared, u: &[NodeVec], time: f64, kw: &RawKeyword) -> Result<Vec<(String, f64)>> {
    let m = prep.model;
    let name = kw.param_upper("CRACK NAME").ok_or_else(|| crate::model::FemError(format!("{}: CRACK NAME= (berikningens namn) saknas", kw.location())))?;
    let Some(crack) = xfem::crack_of(m, &name) else {
        return kw.err(format!("berikningen {name} finns inte (CRACK NAME= ska vara *ENRICHMENT-namnet)"));
    };
    if !crack.stationary {
        return kw.err("konturintegraler med XFEM kräver TYPE=STATIONARY CRACK");
    }
    let kind = kw.param_upper("TYPE").map(|s| s.replace(' ', "")).unwrap_or_else(|| "J".into());
    let kfac = match kind.as_str() {
        "J" => false,
        "KFACTORS" => true,
        t => return kw.err(format!("TYPE={t} stöds inte för XFEM (J, K FACTORS)")),
    };
    let contours: usize = kw.param("CONTOURS").and_then(|c| c.trim().parse().ok()).unwrap_or(1).max(1);
    let pts = front_points(prep, crack)?;
    let el = Elastic { e: crack.e_mod, nu: crack.nu, plane_stress: false };
    let es = el.e_star();
    let mu = el.mu();
    let mut node_elems: HashMap<usize, Vec<usize>> = HashMap::new();
    for &e in &crack.domain {
        for &n in &crack.conn[&e].1 {
            node_elems.entry(n).or_default().push(e);
        }
    }
    let mut out = Vec::new();
    let npts = pts.len();
    for (k, fp) in pts.iter().enumerate() {
        let tag = format!("{}_XFEM_{}", crack.name, k + 1);
        out.push((format!("X at {tag}"), fp.p[0]));
        out.push((format!("Y at {tag}"), fp.p[1]));
        out.push((format!("Z at {tag}"), fp.p[2]));
        let tdir = if npts > 1 { unit(sub(pts[npts - 1].p, pts[0].p)) } else { fp.frame[2] };
        let s0 = dot(fp.p, tdir);
        let lm = if k > 0 { s0 - dot(pts[k - 1].p, tdir) } else { 0.0 };
        let lp = if k + 1 < npts { dot(pts[k + 1].p, tdir) - s0 } else { 0.0 };
        let norm = 0.5 * (lm + lp);
        let hat = |x: [f64; 3]| -> f64 {
            let s = dot(x, tdir) - s0;
            let tol = 1e-9 * (lm + lp).max(1e-300);
            if s.abs() <= tol {
                1.0
            } else if s > 0.0 && lp > 0.0 {
                (1.0 - s / lp).max(0.0)
            } else if s < 0.0 && lm > 0.0 {
                (1.0 + s / lm).max(0.0)
            } else {
                0.0
            }
        };
        let mut ring: BTreeSet<usize> = fp.elems.iter().copied().collect();
        for c in 1..=contours {
            let inside: BTreeSet<usize> = ring.iter().flat_map(|e| crack.conn[e].1.iter().copied()).collect();
            let mut elems: BTreeSet<usize> = BTreeSet::new();
            for n in &inside {
                elems.extend(node_elems[n].iter().copied());
            }
            let mut j = 0.0;
            let mut ii = [0.0; 3];
            for &e in &elems {
                let nodes = &crack.conn[&e].1;
                let qn: Vec<f64> = nodes.iter().map(|n| if inside.contains(n) { hat(m.nodes[*n]) } else { 0.0 }).collect();
                if qn.iter().all(|&v| v == 0.0) {
                    continue;
                }
                let Some(xe) = xfem::as_xfem(m.elements[e].etype) else { continue };
                let ctx = prep.ctx(e, false, time, 0.0);
                let ue = prep.gather(e, u);
                for d in xfem::field_points(xe, &ctx, &ue)? {
                    let mut dq = [0.0; 3];
                    for (i, g) in d.dndx.iter().enumerate() {
                        for l in 0..3 {
                            dq[l] += g[l] * qn[i];
                        }
                    }
                    if dq.iter().all(|&v| v == 0.0) {
                        continue;
                    }
                    let r = &fp.frame;
                    let s = rot(r, &d.s);
                    let eps = rot(r, &d.eps);
                    let g = rot(r, &d.g);
                    let dq = rotv(r, dq);
                    let mut w = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            w += 0.5 * s[a][b] * eps[a][b];
                        }
                    }
                    for i in 0..3 {
                        let mut f = if i == 0 { -w } else { 0.0 };
                        for jj in 0..3 {
                            f += s[i][jj] * g[jj][0];
                        }
                        j += f * dq[i] * d.w;
                    }
                    if kfac {
                        let xl = rotv(r, sub(d.x, fp.p));
                        for (mi, mode) in [AuxMode::I, AuxMode::II, AuxMode::III].into_iter().enumerate() {
                            let a = aux(mode, &el, xl[0], xl[1]);
                            let mut sae = 0.0;
                            for p in 0..3 {
                                for q in 0..3 {
                                    sae += a.s[p][q] * eps[p][q];
                                }
                            }
                            for i in 0..2 {
                                let mut f = if i == 0 { -sae } else { 0.0 };
                                for jj in 0..3 {
                                    f += s[i][jj] * a.g[jj][0] + a.s[i][jj] * g[jj][0];
                                }
                                ii[mi] += f * dq[i] * d.w;
                            }
                        }
                    }
                }
            }
            let nm = if norm > 0.0 { norm } else { 1.0 };
            let ct = format!("{tag}_Contour_{c}");
            if !kfac {
                out.push((format!("J at {ct}"), j / nm));
            }
            if kfac {
                let k1 = 0.5 * es * ii[0] / nm;
                let k2 = 0.5 * es * ii[1] / nm;
                let k3 = mu * ii[2] / nm;
                out.push((format!("K1 at {ct}"), k1));
                out.push((format!("K2 at {ct}"), k2));
                out.push((format!("K3 at {ct}"), k3));
                out.push((format!("JKs at {ct}"), (k1 * k1 + k2 * k2) / es + k3 * k3 / (2.0 * mu)));
                out.push((format!("Cpd at {ct}"), mts_angle(k1, k2)));
            }
            ring = elems;
        }
    }
    Ok(out)
}
