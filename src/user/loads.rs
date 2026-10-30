//! Användarlaster och användarrandvillkor:
//!
//! - DLOAD för `*DLOAD` med lasttypen PnNU (normaltryck på yta n med storlek
//!   från DLOAD i varje integrationspunkt på ytan). Lasttypen sparas av
//!   .inp-läsaren som `PnNU@SETNAMN` (`inp/keywords/loads.rs`) så att rutinen
//!   kan väljas med elementsetets namn. De ekvivalenta nodkrafterna är
//!   Σ Nᵢ p(x_g) n dA över ytans Gausspunkter (2 per kant och 2×2 per
//!   fyrhörning för linjära ytor, 3 respektive 3×3 för kvadratiska, 3 för
//!   trianglar), positivt tryck in i materialet, i referensläget (som en
//!   "död" last). F in till DLOAD är storleken på dataraden, rampad över
//!   steget som andra laster.
//! - DISP för `*BOUNDARY, USER`: värdet vid inkrementets slut ges av DISP
//!   (U(1), totalvärde), anropat för varje nod och frihetsgrad.

use crate::fem::assembly::Prepared;
use crate::fem::model::DistLoad;
use crate::model::{fail, Result};

/// (yta n, setnamn) för en PnNU-last.
pub fn parse_nu(kind: &str) -> Option<(usize, &str)> {
    let (k, name) = kind.split_once('@').unwrap_or((kind, ""));
    let n = k.strip_prefix('P')?.strip_suffix("NU")?.parse::<usize>().ok()?;
    Some((n, name))
}

/// Gausspunkter och vikter på [−1, 1].
fn gauss(n: usize) -> Vec<(f64, f64)> {
    match n {
        2 => {
            let a = 1.0 / 3f64.sqrt();
            vec![(-a, 1.0), (a, 1.0)]
        }
        _ => {
            let a = (0.6f64).sqrt();
            vec![(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
        }
    }
}

/// Formfunktioner och derivator för ytan (lokala koordinater).
fn face_shape(nn: usize, xi: f64, eta: f64) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    Some(match nn {
        2 => (vec![0.5 * (1.0 - xi), 0.5 * (1.0 + xi)], vec![-0.5, 0.5], vec![0.0, 0.0]),
        3 if eta.is_nan() => (vec![0.5 * xi * (xi - 1.0), 0.5 * xi * (xi + 1.0), 1.0 - xi * xi], vec![xi - 0.5, xi + 0.5, -2.0 * xi], vec![0.0; 3]),
        3 => (vec![1.0 - xi - eta, xi, eta], vec![-1.0, 1.0, 0.0], vec![-1.0, 0.0, 1.0]),
        4 => {
            let s = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            (
                s.iter().map(|(a, b)| 0.25 * (1.0 + a * xi) * (1.0 + b * eta)).collect(),
                s.iter().map(|(a, b)| 0.25 * a * (1.0 + b * eta)).collect(),
                s.iter().map(|(a, b)| 0.25 * b * (1.0 + a * xi)).collect(),
            )
        }
        6 => {
            let l = 1.0 - xi - eta;
            (
                vec![l * (2.0 * l - 1.0), xi * (2.0 * xi - 1.0), eta * (2.0 * eta - 1.0), 4.0 * xi * l, 4.0 * xi * eta, 4.0 * eta * l],
                vec![1.0 - 4.0 * l, 4.0 * xi - 1.0, 0.0, 4.0 * (l - xi), 4.0 * eta, -4.0 * eta],
                vec![1.0 - 4.0 * l, 0.0, 4.0 * eta - 1.0, -4.0 * xi, 4.0 * xi, 4.0 * (l - eta)],
            )
        }
        8 => {
            let c = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
            let mut n = vec![0.0; 8];
            let mut dx = vec![0.0; 8];
            let mut de = vec![0.0; 8];
            for (i, &(a, b)) in c.iter().enumerate() {
                n[i] = 0.25 * (1.0 + a * xi) * (1.0 + b * eta) * (a * xi + b * eta - 1.0);
                dx[i] = 0.25 * a * (1.0 + b * eta) * (2.0 * a * xi + b * eta);
                de[i] = 0.25 * b * (1.0 + a * xi) * (a * xi + 2.0 * b * eta);
            }
            let m = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)];
            for (k, &(a, b)) in m.iter().enumerate() {
                let i = 4 + k;
                if a == 0.0 {
                    n[i] = 0.5 * (1.0 - xi * xi) * (1.0 + b * eta);
                    dx[i] = -xi * (1.0 + b * eta);
                    de[i] = 0.5 * b * (1.0 - xi * xi);
                } else {
                    n[i] = 0.5 * (1.0 + a * xi) * (1.0 - eta * eta);
                    dx[i] = 0.5 * a * (1.0 - eta * eta);
                    de[i] = -eta * (1.0 + a * xi);
                }
            }
            (n, dx, de)
        }
        _ => return None,
    })
}

/// Integrationspunkter (ξ, η, vikt) på en yta med `nn` noder; η = NaN för kanter.
fn face_points(nn: usize, dim: usize) -> Vec<(f64, f64, f64)> {
    if dim == 2 {
        return gauss(if nn == 2 { 2 } else { 3 }).into_iter().map(|(a, w)| (a, f64::NAN, w)).collect();
    }
    match nn {
        3 | 6 => vec![(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)],
        _ => {
            let g = gauss(if nn == 4 { 2 } else { 3 });
            let mut v = Vec::new();
            for &(b, wb) in &g {
                for &(a, wa) in &g {
                    v.push((a, b, wa * wb));
                }
            }
            v
        }
    }
}

/// Ekvivalenta nodkrafter (elementvektor) för en PnNU-last, eller `None` om
/// lasten inte är en användarlast.
pub fn dload(p: &Prepared, l: &DistLoad, time: f64) -> Result<Option<Vec<f64>>> {
    let Some((face_no, set)) = parse_nu(&l.kind) else { return Ok(None) };
    let m = p.model;
    let el = &m.elements[l.elem];
    let et = el.etype;
    let faces = et.faces();
    if face_no == 0 || face_no > faces.len() {
        return fail(format!("{} har bara ytorna P1–P{}, inte P{face_no}NU", et.name(), faces.len()));
    }
    use crate::elements::Family;
    let dim = match et.family() {
        Family::Continuum2D => 2,
        Family::Continuum3D => 3,
        f => return fail(format!("{}: lasttypen P{face_no}NU stöds bara för plana element och solider (inte {f:?})", et.name())),
    };
    let Some(routine) = super::routines::find_dload(set)? else {
        return super::routines::missing("DLOAD", "elementsetet", set);
    };
    let ctx = p.ctx(l.elem, false, time, 0.0);
    let thick = if dim == 2 { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 };
    let x = &ctx.x;
    let face = faces[face_no - 1];
    let xf: Vec<[f64; 3]> = face.iter().map(|&i| x[i]).collect();
    let centroid = crate::elements::common::centroid(x);
    let fc = crate::elements::common::centroid(&xf);
    let (kstep, kinc) = super::context::step_inc();
    let tt = [super::context::step_time(time), time];
    let mut fn_ = vec![[0.0; 3]; face.len()];
    for (g, (xi, eta, w)) in face_points(face.len(), dim).into_iter().enumerate() {
        let Some((n, dxi, deta)) = face_shape(face.len(), xi, eta) else {
            return fail(format!("{}: ytor med {} noder stöds inte för PnNU", et.name(), face.len()));
        };
        let mut xg = [0.0; 3];
        let mut a = [0.0; 3];
        let mut b = [0.0; 3];
        for i in 0..face.len() {
            for d in 0..3 {
                xg[d] += n[i] * xf[i][d];
                a[d] += dxi[i] * xf[i][d];
                b[d] += deta[i] * xf[i][d];
            }
        }
        // ytnormal gånger ytelement
        let mut nv = if dim == 2 { [a[1] * thick, -a[0] * thick, 0.0] } else { crate::elements::common::cross(a, b) };
        if crate::elements::common::dot(crate::elements::common::sub(fc, centroid), nv) < 0.0 {
            nv = crate::elements::common::scale(nv, -1.0);
        }
        let mut args = super::routines::DloadArgs {
            f: l.magnitude,
            kstep,
            kinc,
            time: tt,
            noel: el.label,
            npt: g + 1,
            layer: 1,
            kspt: 1,
            coords: xg,
            jltyp: 20 + face_no as i32,
            sname: "",
        };
        routine.dload(&mut args)?;
        for i in 0..face.len() {
            for d in 0..3 {
                fn_[i][d] -= args.f * w * n[i] * nv[d];
            }
        }
    }
    // till elementvektorn (förskjutningsfrihetsgraderna U1..U3 per nod)
    let mut f = vec![0.0; et.n_dofs()];
    let mut off = vec![0usize; el.nodes.len() + 1];
    for k in 0..el.nodes.len() {
        off[k + 1] = off[k] + et.node_dofs(k).len();
    }
    for (k, &i) in face.iter().enumerate() {
        for (j, &dof) in et.node_dofs(i).iter().enumerate() {
            if (1..=dim as i32).contains(&(dof as i32)) {
                f[off[i] + j] += fn_[k][dof as usize - 1];
            }
        }
    }
    Ok(Some(f))
}

// --------------------------------------------------------------------- DISP

/// Nyckelordet i stegets `extra` som listar *BOUNDARY, USER-poster
/// (datarader [nodindex, frihetsgrad, värde], parametern `_NAME` = nodsetet).
pub const USER_BC: &str = "_USER BOUNDARY";

/// Ett randvillkor som DISP ger.
#[derive(Debug, Clone)]
pub struct UserBc {
    pub node: usize,
    pub slot: usize,
    pub jdof: i32,
    /// Värdet på dataraden (U(1) in till DISP).
    pub value: f64,
    pub set: String,
    pub label: i64,
    pub coords: [f64; 3],
}

/// Stegets *BOUNDARY, USER-poster.
pub fn user_bcs(model: &crate::fem::model::FeModel, step: &crate::fem::model::StepDef) -> Result<Vec<UserBc>> {
    let mut v = Vec::new();
    for k in step.extra.iter().filter(|k| k.name == USER_BC) {
        let set = k.param("_NAME").unwrap_or("").to_string();
        if super::routines::find_disp(&set)?.is_none() {
            return super::routines::missing("DISP", "nodsetet", &set);
        }
        for d in &k.data {
            let x = k.floats(d)?;
            let n = x[0] as usize;
            let jdof = x[1] as i32;
            v.push(UserBc { node: n, slot: crate::fem::dof::slot_of(jdof as crate::fem::dof::Dof), jdof, value: x[2], set: set.clone(), label: model.node_label[n], coords: model.nodes[n] });
        }
    }
    Ok(v)
}

/// Värden från DISP vid stegtidsandelen `t`. Fel i rutinen ger en varning och värdet på dataraden.
pub fn disp_values(items: &[UserBc], t: f64) -> Vec<(usize, usize, f64)> {
    let period = super::context::period();
    let st = t * period;
    let tt = [st, super::context::step_start() + st];
    let (kstep, kinc) = super::context::step_inc();
    items
        .iter()
        .map(|b| {
            let mut a = super::routines::DispArgs { u: [b.value, 0.0, 0.0], kstep, kinc, time: tt, node: b.label, noel: 0, jdof: b.jdof, coords: b.coords, set: &b.set };
            match super::routines::find_disp(&b.set).map(|r| r.map(|r| r.disp(&mut a))) {
                Ok(Some(Ok(()))) => (b.node, b.slot, a.u[0]),
                Ok(Some(Err(e))) | Err(e) => {
                    eprintln!("varning: DISP i nod {}: {e}", b.label);
                    (b.node, b.slot, b.value)
                }
                Ok(None) => (b.node, b.slot, b.value),
            }
        })
        .collect()
}

// ------------------------------------------------- värmeledning: HETVAL, UMATHT, FILM

/// (UMATHT, HETVAL) för materialet: *USER MATERIAL, TYPE=THERMAL respektive *HEAT GENERATION.
pub fn heat_user(def: &crate::fem::model::MaterialDef) -> (bool, bool) {
    let umatht = def.behavior("USER MATERIAL").is_some_and(|k| k.param_upper("TYPE").as_deref() == Some("THERMAL"));
    (umatht, def.behavior("HEAT GENERATION").is_some())
}

/// Konstanterna på *USER MATERIAL, TYPE=THERMAL.
fn thermal_props(def: &crate::fem::model::MaterialDef) -> Result<Vec<f64>> {
    let Some(kw) = def.behavior("USER MATERIAL") else { return Ok(Vec::new()) };
    let mut p = Vec::new();
    for d in kw.records() {
        p.extend(kw.floats(&d)?);
    }
    if let Some(n) = kw.param("CONSTANTS").and_then(crate::inp::parse_label) {
        p.resize(n.max(0) as usize, 0.0);
    }
    Ok(p)
}

fn interp(n: &[f64], v: &[f64]) -> f64 {
    n.iter().zip(v).map(|(a, b)| a * b).sum()
}

/// Värmegenerering (HETVAL) som bidrag till det inre flödet (−∫ N r dV) och
/// tangenten. `time` är total tid vid inkrementets början.
pub fn hetval(def: &crate::fem::model::MaterialDef, ips: &[crate::elements::common::Ip], t: &[f64], told: &[f64], time: f64, dt: f64, tangent: bool) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let Some(routine) = super::routines::find_hetval(&def.name)? else {
        return super::routines::missing("HETVAL", "materialet", &def.name);
    };
    let nn = t.len();
    let mut f = vec![0.0; nn];
    let mut k = if tangent { vec![0.0; nn * nn] } else { Vec::new() };
    let (fl, flo) = super::context::point_fields();
    let dpred: Vec<f64> = fl.iter().zip(&flo).map(|(a, b)| a - b).collect();
    for ip in ips {
        let (tp, t0) = (interp(&ip.n, t), interp(&ip.n, told));
        let mut statev: Vec<f64> = Vec::new();
        let mut a = super::routines::HetvalArgs { cmname: &def.name, temp: [tp, tp - t0], time: [super::context::step_time(time), time], dtime: dt, statev: &mut statev, flux: [0.0; 2], predef: &flo, dpred: &dpred };
        routine.hetval(&mut a)?;
        for i in 0..nn {
            f[i] -= ip.w * ip.n[i] * a.flux[0];
            if tangent {
                for j in 0..nn {
                    k[i * nn + j] -= ip.w * ip.n[i] * a.flux[1] * ip.n[j];
                }
            }
        }
    }
    Ok((f, tangent.then_some(k)))
}

/// UMATHT: inre flöde och tangent för ett värmeledningselement: ledning
/// −∫∇Nᵢ·f dV med f = FLUX i `ips` och (transient) lagrad energi
/// ∫ Nᵢ ρ ΔU/Δt dV i `cips` (U in till rutinen är 0, så U ut är ΔU).
#[allow(clippy::too_many_arguments)]
pub fn umatht(
    def: &crate::fem::model::MaterialDef,
    ips: &[crate::elements::common::Ip],
    cips: &[crate::elements::common::Ip],
    x: &[[f64; 3]],
    t: &[f64],
    told: &[f64],
    time: f64,
    dt: f64,
    transient: bool,
    dim: usize,
    noel: i64,
    tangent: bool,
) -> Result<(Vec<f64>, Option<Vec<f64>>)> {
    let Some(routine) = super::routines::find_umatht(&def.name)? else {
        return super::routines::missing("UMATHT", "materialet", &def.name);
    };
    let props = thermal_props(def)?;
    let nn = t.len();
    let mut f = vec![0.0; nn];
    let mut k = if tangent { vec![0.0; nn * nn] } else { Vec::new() };
    let (kstep, kinc) = super::context::step_inc();
    let (fl, flo) = super::context::point_fields();
    let dpred: Vec<f64> = fl.iter().zip(&flo).map(|(a, b)| a - b).collect();
    let tt = [super::context::step_time(time), time];
    let call = |ip: &crate::elements::common::Ip, npt: usize| -> Result<(f64, f64, Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (tp, t0) = (interp(&ip.n, t), interp(&ip.n, told));
        let mut g = vec![0.0; dim];
        for (d, &ti) in ip.dndx.iter().zip(t) {
            for c in 0..dim {
                g[c] += d[c] * ti;
            }
        }
        let mut coords = [0.0; 3];
        for (ni, xi) in ip.n.iter().zip(x) {
            for c in 0..3 {
                coords[c] += ni * xi[c];
            }
        }
        let (mut dudg, mut flux, mut dfdt, mut dfdg) = (vec![0.0; dim], vec![0.0; dim], vec![0.0; dim], vec![0.0; dim * dim]);
        let mut statev: Vec<f64> = Vec::new();
        let mut a = super::routines::UmathtArgs {
            u: 0.0,
            dudt: 0.0,
            dudg: &mut dudg,
            flux: &mut flux,
            dfdt: &mut dfdt,
            dfdg: &mut dfdg,
            statev: &mut statev,
            temp: t0,
            dtemp: tp - t0,
            dtemdx: &g,
            time: tt,
            dtime: dt,
            predef: &flo,
            dpred: &dpred,
            cmname: &def.name,
            ntgrd: dim,
            props: &props,
            coords,
            pnewdt: 1.0,
            noel,
            npt,
            layer: 1,
            kspt: 1,
            kstep,
            kinc,
        };
        routine.umatht(&mut a)?;
        if a.pnewdt < 1.0 {
            super::context::request_pnewdt(a.pnewdt);
        }
        let (u, dudt) = (a.u, a.dudt);
        Ok((u, dudt, flux, dfdt, dfdg))
    };
    for (q, ip) in ips.iter().enumerate() {
        let (_, _, flux, dfdt, dfdg) = call(ip, q + 1)?;
        for i in 0..nn {
            let di = ip.dndx[i];
            f[i] -= ip.w * (0..dim).map(|c| di[c] * flux[c]).sum::<f64>();
            if tangent {
                for j in 0..nn {
                    let dj = ip.dndx[j];
                    let mut v = 0.0;
                    for r in 0..dim {
                        v += di[r] * dfdt[r] * ip.n[j];
                        for c in 0..dim {
                            v += di[r] * dfdg[r + c * dim] * dj[c];
                        }
                    }
                    k[i * nn + j] -= ip.w * v;
                }
            }
        }
    }
    if transient && dt > 0.0 {
        let rho = def.density;
        if rho == 0.0 {
            return fail(format!("material {}: *DENSITY saknas (krävs för transient värmeledning med UMATHT)", def.name));
        }
        for (q, ip) in cips.iter().enumerate() {
            let (du, dudt, ..) = call(ip, q + 1)?;
            for i in 0..nn {
                f[i] += ip.w * ip.n[i] * rho * du / dt;
                if tangent {
                    for j in 0..nn {
                        k[i * nn + j] += ip.w * ip.n[i] * rho * dudt / dt * ip.n[j];
                    }
                }
            }
        }
    }
    Ok((f, tangent.then_some(k)))
}

/// FILM för en ytpunkt: (q, dq/dT) med q = h (T − T∞). `set` väljer rutinen.
pub fn film(set: &str, face: usize, tp: f64, coords: [f64; 3], noel: i64, npt: usize, time: f64) -> Result<(f64, f64)> {
    let Some(routine) = super::routines::find_film(set)? else {
        return super::routines::missing("FILM", "elementsetet", set);
    };
    let (kstep, kinc) = super::context::step_inc();
    let (fl, _) = super::context::point_fields();
    let mut a = super::routines::FilmArgs { h: [0.0; 2], sink: 0.0, temp: tp, kstep, kinc, time: [super::context::step_time(time), time], noel, npt, coords, jltyp: face as i32, field: &fl, sname: "", node: 0, area: 0.0 };
    routine.film(&mut a)?;
    Ok((a.h[0] * (tp - a.sink), a.h[0] + a.h[1] * (tp - a.sink)))
}
