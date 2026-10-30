//! Area acoustics2: acoustics in explicit analysis (`*DYNAMIC, EXPLICIT` in
//! models with acoustic elements; included from `acoustic.rs`).
//!
//! The coupled system (see `acoustic.rs`)
//!
//! ```text
//!   M_s ü + C_s u̇ + K_s u − Sᵀ p = F
//!   M_f p̈ + C_f ṗ + K_f p + S ü = Q
//! ```
//!
//! is integrated with central differences (diagonal fluid mass, the same
//! central difference method as for the structure). The masses are row-summed.
//! The structural accelerations are computed first and the fluid's afterwards with S ü
//! (the mass matrix is block triangular). The damping (volumetric drag,
//! impedance and radiation boundaries) uses the half-step velocity.
//!
//! Cavitation (*ACOUSTIC MEDIUM, CAVITATION LIMIT p_c): the nodal variable is
//! the pseudo-pressure p_v and the pressure p = max(p_v, p_c − p₀) with the static
//! pressure p₀ (*INITIAL CONDITIONS, TYPE=ACOUSTIC STATIC PRESSURE; without it
//! no cavitation) is used in the stiffness term and as output POR
//! (total-wave formulation with the pseudo-pressure as state).
//!
//! The structure must be linear (stiffness from the base state); acoustic
//! infinite elements and prescribed values ≠ 0 are not supported in explicit.
//! Time increment: `DIRECT USER CONTROL` (Δt on the data line), otherwise
//! Δt = min Lₑ/c_f over the acoustic elements (Lₑ = shortest edge),
//! without regard to the structure's stability limit.

use super::*;

/// Diagonal (row sum) of the matrix in the free equations.
fn lumped(g: &Gen, n: usize) -> Vec<f64> {
    let mut d = vec![0.0; n];
    for (r, row) in g.rows.iter().enumerate().take(n) {
        for &(c, v) in row {
            if (c as usize) < n {
                d[r] += v;
            }
        }
    }
    d
}

/// Stable time increment for the acoustic elements (shortest edge / c_f).
fn stable_dt(m: &FeModel) -> Result<f64> {
    let mut dt = f64::INFINITY;
    for (e, el) in m.elements.iter().enumerate() {
        if ac::lookup(el.etype.name()).is_none() {
            continue;
        }
        let p = material_props(m, e)?;
        let c = (p.bulk / p.density).sqrt();
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
        let mut lchar = f64::INFINITY;
        for i in 0..x.len() {
            for j in i + 1..x.len() {
                let d = ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2) + (x[i][2] - x[j][2]).powi(2)).sqrt();
                if d > 0.0 {
                    lchar = lchar.min(d);
                }
            }
        }
        dt = dt.min(lchar / c);
    }
    Ok(dt)
}

/// *BULK VISCOSITY b₁, b₂ (default 0.06 and 1.2).
fn bulk_parameters(step: &StepDef) -> Result<(f64, f64)> {
    let mut b = (0.06, 1.2);
    for k in step.extra.iter().filter(|k| k.name == "BULK VISCOSITY") {
        let v = k.first_floats()?;
        b = (v.first().copied().unwrap_or(0.06), v.get(1).copied().unwrap_or(1.2));
    }
    Ok(b)
}

/// Characteristic length Lₑ as in explicit analysis (`explicit_dt::geometry`),
/// otherwise the shortest edge.
fn char_length(a: &AcousticElem, x: &[[f64; 3]]) -> f64 {
    use crate::elements::heat::Shape;
    use crate::elements::Family;
    let g = match a.geom.shape {
        Shape::Hex8 => crate::steps::explicit::dt::geometry(Family::Continuum3D, "C3D8", x),
        Shape::Tet4 => crate::steps::explicit::dt::geometry(Family::Continuum3D, "C3D4", x),
        Shape::Quad4 | Shape::Tri3 => crate::steps::explicit::dt::geometry(Family::Continuum2D, "CPE4", x),
        _ => None,
    };
    if let Some(g) = g {
        return g.lchar;
    }
    let mut lchar = f64::INFINITY;
    for i in 0..x.len() {
        for j in i + 1..x.len() {
            let d = ((x[i][0] - x[j][0]).powi(2) + (x[i][1] - x[j][1]).powi(2) + (x[i][2] - x[j][2]).powi(2)).sqrt();
            if d > 0.0 {
                lchar = lchar.min(d);
            }
        }
    }
    lchar
}

pub(super) fn dynamic_explicit(ctx: &mut StepCtx) -> Result<()> {
    let kw = ctx.step.procedure.clone().unwrap();
    let d = kw.first_floats()?;
    let direct = kw.has("DIRECT USER CONTROL") || kw.has("FIXED TIME INCREMENTATION");
    let (h, period) = if kw.has("DIRECT USER CONTROL") {
        (d.first().copied().unwrap_or(0.0), d.get(1).copied().unwrap_or(1.0))
    } else {
        (0.0, d.get(1).copied().or(d.first().copied()).filter(|v| *v > 0.0).unwrap_or(1.0))
    };
    let p = ctx.prep;
    let model = p.model;
    if model.elements.iter().any(|e| crate::elements::acoustic_infinite::lookup(e.etype.name()).is_some()) {
        return fail("acoustic infinite elements are not supported yet in *DYNAMIC, EXPLICIT");
    }
    for e in 0..model.elements.len() {
        if ac::lookup(model.elements[e].etype.name()).is_some() {
            continue;
        }
        let c = p.ctx(e, false, 0.0, 0.0);
        if c.material.is_some_and(|mm| !mm.is_linear()) || ctx.nlgeom {
            return fail("*DYNAMIC, EXPLICIT with acoustic elements: the structure must be linear elastic (without NLGEOM)");
        }
    }
    let h = if h > 0.0 { h } else { stable_dt(model)? };
    if !(h > 0.0 && h.is_finite()) {
        return fail("*DYNAMIC, EXPLICIT: could not determine the time increment");
    }
    let _ = direct;
    let mut sys = Sys::new(ctx, true)?;
    // U is written in the tied fluid nodes (node-based *TIE):
    // the structure's displacement with the coupling weights
    {
        let mut w: std::collections::BTreeMap<usize, std::collections::BTreeMap<usize, f64>> = Default::default();
        for c in crate::interactions::acoustic::couplings(model)? {
            for &(pn, un, _, v) in &c.entries {
                *w.entry(pn).or_default().entry(un).or_default() += v.abs();
            }
        }
        for (pn, m) in w {
            if sys.fluid_u.iter().any(|f| f.0 == pn) {
                continue;
            }
            // coincident structural node: its displacement
            let xp = model.nodes[pn];
            if let Some(&un) = m.keys().find(|&&un| {
                let d = model.nodes[un];
                (d[0] - xp[0]).abs() + (d[1] - xp[1]).abs() + (d[2] - xp[2]).abs() < 1e-9 * (1.0 + xp[0].abs() + xp[1].abs() + xp[2].abs())
            }) {
                sys.fluid_u.push((pn, vec![(un, 1.0)]));
                continue;
            }
            let tot: f64 = m.values().sum();
            if tot > 0.0 {
                sys.fluid_u.push((pn, m.into_iter().map(|(un, v)| (un, v / tot)).collect()));
            }
        }
    }
    let sys = sys;
    if !sys.improved.is_empty() {
        return fail("NONREFLECTING=IMPROVED is not supported yet in *DYNAMIC, EXPLICIT");
    }
    let n = sys.n;
    let nn = model.nodes.len();
    // matrices (free equations)
    let mst = sys.minus_st();
    let coeffs: Vec<(f64, f64, f64)> = sys.imp.iter().map(|b| b.time_coeffs()).collect();
    let mut kt: Vec<(&Gen, f64)> = vec![(&sys.k0, 1.0), (&mst, 1.0)];
    let mut ct: Vec<(&Gen, f64)> = vec![(&sys.cf, 1.0)];
    let mut mt: Vec<(&Gen, f64)> = vec![(&sys.ms, 1.0), (&sys.mf, 1.0)];
    for (b, c) in sys.imp.iter().zip(&coeffs) {
        mt.push((&b.h, c.0));
        ct.push((&b.h, c.1));
        kt.push((&b.h, c.2));
    }
    // bulk viscosity in the acoustic elements (switched off with
    // *BULK VISCOSITY 0, 0): the pressure in
    // the stiffness term becomes p + b₁ (Lₑ/c_f) ṗ, i.e. the damping b₁ Lₑ/c_f · K_e
    let (b1, _b2) = bulk_parameters(ctx.step)?;
    let mut cbv = Gen::new(sys.n_all());
    if b1 != 0.0 {
        for (e, el) in model.elements.iter().enumerate() {
            let Some(a) = ac::lookup(el.etype.name()) else { continue };
            let props = material_props(model, e)?;
            let c = (props.bulk / props.density).sqrt();
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&nd| model.nodes[nd]).collect();
            let lchar = char_length(a, &x);
            let geo = a.geometric(&x, thickness(model, e))?;
            let f = b1 * lchar / c / props.density;
            let ke: Vec<f64> = geo.g.iter().map(|v| v * f).collect();
            sys.scatter(e, &ke, &mut cbv);
        }
    }
    cbv.finish();
    ct.push((&cbv, 1.0));
    let km = Gen::combine(&kt, n);
    let cm = Gen::combine(&ct, n);
    let mdiag = lumped(&Gen::combine(&mt, n), n);
    let s = sys.s.leading(n);
    let fluid: Vec<bool> = (0..n).map(|q| sys.dm.owner[q].1 == PS).collect();
    for q in 0..n {
        if !(mdiag[q] > 0.0) {
            let (nd, sl) = sys.dm.owner[q];
            return fail(format!("*DYNAMIC, EXPLICIT: DOF {} at node {} has no mass", crate::fem::dof::SLOT_DOF[sl], model.node_name(nd)));
        }
    }
    // cavitation limit p_c − p₀ per pressure equation; without static pressure
    // (*INITIAL CONDITIONS, TYPE=ACOUSTIC STATIC PRESSURE) a
    // large pressure and no cavitation is assumed
    let mut p0: Vec<Option<f64>> = vec![None; nn];
    for k in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(crate::inp::keywords::acoustics2::RESOLVED)) {
        for d in &k.data {
            let v = k.floats(d)?;
            if let (Some(&nd), Some(&pv)) = (v.first(), v.get(1)) {
                p0[nd as usize] = Some(pv);
            }
        }
    }
    let mut cav = vec![f64::NEG_INFINITY; n];
    for (e, el) in model.elements.iter().enumerate() {
        if ac::lookup(el.etype.name()).is_none() {
            continue;
        }
        let Some(pc) = material_props(model, e)?.complex.cavitation else { continue };
        for &nd in &el.nodes {
            let Some(ps) = p0[nd] else { continue };
            sys.dm.for_each_eq(nd, PS, |q, _| cav[q] = cav[q].max(pc - ps));
        }
    }
    let cap = |x: &[f64]| -> Vec<f64> { x.iter().zip(&cav).map(|(v, c)| if *c > f64::NEG_INFINITY { v.max(*c) } else { *v }).collect() };
    let ramp = crate::steps::dynamic::default_ramp(ctx.step);
    let lev = LoadEval::new(model, ctx.loads, ramp, period, ctx.state.time)?;
    let force = |t: f64| -> Result<Vec<f64>> {
        let z = vec![[0.0; SLOTS]; nn];
        let f = assembly::external(p, &lev.cloads_at(t), &lev.dloads_at(t), &z, false, ctx.state.time + t)?;
        Ok((0..n).map(|q| f[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect())
    };
    // acceleration: the structure first, then the fluid with S ü
    let accel = |f: &[f64], x: &[f64], v: &[f64]| -> Vec<f64> {
        let kx = km.mul(&cap(x));
        let cv = cm.mul(v);
        let mut a = vec![0.0; n];
        for q in 0..n {
            if !fluid[q] {
                a[q] = (f[q] - kx[q] - cv[q]) / mdiag[q];
            }
        }
        let sa = s.mul(&a);
        for q in 0..n {
            if fluid[q] {
                a[q] = (f[q] - kx[q] - cv[q] - sa[q]) / mdiag[q];
            }
        }
        a
    };
    let mut x: Vec<f64> = (0..n).map(|q| ctx.state.u[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect();
    let v0 = crate::steps::dynamic::initial_velocity(p, ctx.state)?;
    let mut v: Vec<f64> = (0..n).map(|q| v0[sys.dm.owner[q].0][sys.dm.owner[q].1]).collect();
    let mut a = accel(&force(0.0)?, &x, &v);
    let sched = crate::steps::dynamic::output_schedule(ctx.step, period, crate::steps::dynamic::Schedule::Interval(period))?;
    let emit = |ctx: &StepCtx, t: f64, x: &[f64], v: &[f64], a: &[f64], inc: usize| -> Result<FrameData> {
        let un = sys.to_nodes(&cap(x), &[]);
        let vn = sys.to_nodes(v, &[]);
        let an = sys.to_nodes(a, &[]);
        let mut fr = frame(ctx, &sys, &un, t, -1, format!("Increment {inc}: Step Time = {t:.4e}"))?;
        crate::steps::dynamic::add_va(&mut fr, model.dim(), &vn, &an);
        Ok(fr)
    };
    let mut frames = vec![emit(ctx, 0.0, &x, &v, &a, 0)?];
    let nsteps = ((period / h) - 1e-9).ceil().max(1.0) as usize;
    if nsteps > ctx.step.max_increments.max(1_000_000) {
        return fail(format!("*DYNAMIC, EXPLICIT: {nsteps} increments needed (Δt = {h:.3e})"));
    }
    let mut t = 0.0;
    let mut next_out = 1usize;
    let mut vh: Vec<f64> = v.clone();
    let mut hprev = 0.0;
    for inc in 1..=nsteps {
        let hk = if inc == nsteps { period - t } else { h };
        // v(n+½) = v(n−½) + ½(Δt(n) + Δt(n+1)) a(n), v(½) = v(0) + ½Δt a(0)
        for q in 0..n {
            vh[q] += 0.5 * (hprev + hk) * a[q];
        }
        hprev = hk;
        for q in 0..n {
            x[q] += hk * vh[q];
        }
        t += hk;
        a = accel(&force(t)?, &x, &vh);
        for q in 0..n {
            v[q] = vh[q] + 0.5 * hk * a[q];
        }
        let last = inc == nsteps;
        let want = match sched {
            crate::steps::dynamic::Schedule::Every(k) => inc % k == 0,
            crate::steps::dynamic::Schedule::Interval(dt) => {
                if dt > 0.0 && t >= next_out as f64 * dt - 1e-6 * h {
                    next_out = (t / dt + 1e-6).floor() as usize + 1;
                    true
                } else {
                    false
                }
            }
        };
        if want || last {
            frames.push(emit(ctx, t, &x, &v, &a, inc)?);
        }
    }
    if !ctx.opts.all_frames {
        let l = frames.pop();
        frames.clear();
        frames.extend(l);
    }
    let un = sys.to_nodes(&cap(&x), &[]);
    for nd in 0..nn {
        for sl in 0..SLOTS {
            if ctx.prep.active[nd] & (1 << sl) != 0 && sys.cidx[nd][sl] == NONE {
                ctx.state.u[nd][sl] = un[nd][sl];
            }
        }
    }
    ctx.state.time += period;
    ctx.out.increments = nsteps;
    ctx.out.frames.extend(frames);
    Ok(())
}
