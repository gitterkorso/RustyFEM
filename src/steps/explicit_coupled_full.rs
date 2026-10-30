//! *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT in the general explicit
//! solver (`explicit.rs`, area coupled2): the mechanics as *DYNAMIC, EXPLICIT
//! (explicit hourglass control for CPE4RT/CPS4RT/C3D8RT, bulk viscosity,
//! NLGEOM, contact, mass scaling, energies) and the temperature with forward
//! Euler and lumped heat capacity:
//!
//! ```text
//! θ̇(i)   = C⁻¹ (Q(i) − F(θ(i)) + Q_pl(i))
//! θ(i+1) = θ(i) + Δt(i+1) θ̇(i)
//! ```
//!
//! The heat flux F (conduction, *CFLUX, *DFLUX, *FILM, *RADIATE and thermal
//! contact *GAP CONDUCTANCE with the gap from the current displacements)
//! is computed with `steps/heat.rs` (`HeatSystem`). Q_pl is the heat from plastic
//! dissipation, η σ:ε̇ᵖˡ with η from *INELASTIC HEAT FRACTION
//! (`materials/inelastic_heat.rs`), distributed equally to the element nodes.
//!
//! Time increment: the mechanical element-by-element limit (`explicit_dt`) and the
//! thermal Lₑ²/(2α) with the same characteristic length Lₑ and α = k/(ρc),
//! recomputed each increment with the current geometry and temperature;
//! without NLGEOM times the bulk viscosity damping factor (`stable_in`), and
//! nodes with thermal contact get a nodal limit (`contact_limit`).
//!
//! Other (verified in the test cases in
//! `tests/cases/coupled2/`): the heat capacity is row-summed (∫Nᵢ ρc dV), the
//! thermal strain of the reduced elements uses the volume-weighted
//! nodal temperature (`temp_weights`), HFL is reported at the element midpoint and
//! the bulk viscosity uses c_d·Δtₑ instead of Lₑ when the thermal
//! limit governs the element (`bulk_scale`).
//!
//! The mechanics at the start of the step is computed with the initial temperature (frame 0
//! is stress-free); the temperature boundary conditions act from the first
//! increment.

use super::super::heat::{ensure_initial, heat_loads, HeatSystem, TS};
use super::super::newton::NonlinearSystem;
use super::dt::geometry;
use super::shell::Hourglass;
use super::StepCtx;
use crate::elements::{ElemCtx, Family};
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::fem::output::FrameData;
use crate::fem::state::StepLoads;
use crate::materials::heat::HeatProps;
use crate::model::{FemError, Result};

/// The procedure name.
pub const PROC: &str = "DYNAMIC TEMPERATURE-DISPLACEMENT";

/// Loads of the heat transfer and of the mechanics.
pub fn split_loads(l: &StepLoads) -> (StepLoads, StepLoads) {
    let thermal_kind = |k: &str| k.starts_with("DFLUX:") || k.starts_with("FILM") || k.starts_with("RAD:");
    let mut th = l.clone();
    let mut me = l.clone();
    th.bc.retain(|&(_, s), _| s == TS);
    me.bc.retain(|&(_, s), _| s != TS);
    th.cload.retain(|&(_, s), _| s == TS);
    me.cload.retain(|&(_, s), _| s != TS);
    th.dload.retain(|(d, _)| thermal_kind(&d.kind));
    me.dload.retain(|(d, _)| !thermal_kind(&d.kind));
    th.released.retain(|&(_, s, _)| s == TS);
    me.released.retain(|&(_, s, _)| s != TS);
    (th, me)
}

/// The default element for explicit hourglass control (also the coupled RT elements).
pub fn standard(name: &str) -> Option<&'static crate::elements::solid_higher::Solid> {
    super::solid::standard(name).or_else(|| crate::elements::heat_coupled::mech_name(name).and_then(super::solid::standard))
}

/// Explicit hourglass control for one element; coupled elements have
/// the temperature last per node, and that part is left untouched.
#[allow(clippy::too_many_arguments)]
pub fn hourglass_force(ctx: &ElemCtx, el: &crate::elements::solid_higher::Solid, family: Family, fint: &mut [f64], u: &[f64], old: &[f64], mode: Hourglass) -> Result<(Vec<f64>, f64)> {
    let nn = ctx.x.len();
    if nn == 0 || u.len() % nn != 0 {
        return super::solid::hourglass_force(ctx, el, family, fint, u, old, mode);
    }
    let per = u.len() / nn;
    let nd = if nn == 8 { 3 } else { 2 };
    if per == nd {
        return super::solid::hourglass_force(ctx, el, family, fint, u, old, mode);
    }
    let pick = |v: &[f64]| -> Vec<f64> { (0..nn).flat_map(|i| v[i * per..i * per + nd].to_vec()).collect() };
    let mut fm = pick(fint);
    let um = pick(u);
    let r = super::solid::hourglass_force(ctx, el, family, &mut fm, &um, old, mode)?;
    for i in 0..nn {
        fint[i * per..i * per + nd].copy_from_slice(&fm[i * nd..(i + 1) * nd]);
    }
    Ok(r)
}

/// Sets the nodal temperatures (the temperature at the end and at the start of the
/// increment) in the element context for coupled elements. With weights `w`
/// (reduced elements) all nodes get the integration point temperature
/// Σ wᵢ θᵢ, so that the element mean becomes that value.
pub fn set_temps(p: &Prepared, ctx: &mut ElemCtx, e: usize, u: &[NodeVec], du: &[NodeVec], w: &[f64]) {
    let el = &p.model.elements[e];
    if !el.etype.node_dofs(0).iter().any(|d| crate::fem::dof::slot_of(*d) == TS) {
        return;
    }
    ctx.temp = el.nodes.iter().map(|&n| u[n][TS]).collect();
    ctx.temp_old = el.nodes.iter().map(|&n| u[n][TS] - du[n][TS]).collect();
    if w.len() == ctx.temp.len() {
        let a: f64 = w.iter().zip(&ctx.temp).map(|(x, y)| x * y).sum();
        let b: f64 = w.iter().zip(&ctx.temp_old).map(|(x, y)| x * y).sum();
        ctx.temp.iter_mut().for_each(|t| *t = a);
        ctx.temp_old.iter_mut().for_each(|t| *t = b);
    }
}

/// The thermal subsystem.
pub struct Thermal<'p, 'm> {
    pub heat: HeatSystem<'p, 'm>,
    /// Lumped heat capacity per node.
    cap: Vec<f64>,
    /// θ̇ per node.
    pub rate: Vec<f64>,
    /// Current nodal temperatures (with the boundary conditions); at the start of the step they can
    /// differ from the temperatures the mechanics was last computed with.
    theta: Vec<f64>,
    /// Fraction of the plastic dissipation that becomes heat, per element.
    eta: Vec<f64>,
    /// Thermal properties per element (`None` without a heat part).
    props: Vec<Option<HeatProps>>,
    period: f64,
}

impl<'p, 'm> Thermal<'p, 'm> {
    /// Builds the subsystem and returns it with the mechanics' loads.
    pub fn new(ctx: &mut StepCtx<'p, 'm>, period: f64) -> Result<(Self, StepLoads)> {
        let prep: &'p Prepared<'m> = ctx.prep;
        let model = prep.model;
        ensure_initial(model, ctx.state)?;
        let loads = heat_loads(ctx.step, ctx.loads, false);
        let (th, me) = split_loads(&loads);
        let mut heat = HeatSystem::new(prep, th, ctx.state, false, ctx.solver, period)?;
        heat.gap_disp = !heat.gaps.is_empty();
        let nn = model.nodes.len();
        let mut cap = vec![0.0; nn];
        let mut props = Vec::with_capacity(model.elements.len());
        let mut eta = vec![0.0; model.elements.len()];
        for (e, el) in model.elements.iter().enumerate() {
            let wrap = |err: FemError| FemError(format!("{}: {err}", model.elem_name(e)));
            let Some(he) = crate::elements::heat::lookup(el.etype.name()) else {
                props.push(None);
                continue;
            };
            let Some(sec) = el.section.map(|s| &model.sections[s]) else {
                props.push(None);
                continue;
            };
            let Some(mi) = sec.material_index else {
                props.push(None);
                continue;
            };
            let pr = HeatProps::from_def(&model.materials[mi]).map_err(wrap)?;
            eta[e] = crate::materials::inelastic_heat::fraction(&model.materials[mi])?.unwrap_or(0.0);
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            // row-summed (consistent) capacity ∫Nᵢ ρc dV
            let cips = he.ips(&x, sec.solid_thickness()).map_err(wrap)?;
            let t: Vec<f64> = el.nodes.iter().map(|&n| ctx.state.u[n][TS]).collect();
            let (_, kc) = he.capacity(&cips, &pr, &t, &t, 1.0, true).map_err(wrap)?;
            let kc = kc.unwrap_or_default();
            let k = el.nodes.len();
            for a in 0..k {
                cap[el.nodes[a]] += (0..k).map(|b| kc.get(a * k + b).copied().unwrap_or(0.0)).sum::<f64>();
            }
            props.push(Some(pr));
        }
        let theta = ctx.state.u.iter().map(|x| x[TS]).collect();
        Ok((Thermal { heat, cap, rate: vec![0.0; nn], theta, eta, props, period }, me))
    }

    /// Prescribed temperatures at the step time `t`.
    fn apply_bc(&self, u: &mut [NodeVec], t: f64) {
        for (n, s, val) in self.heat.bc_now(t / self.period) {
            if self.heat.dm.constrained[n] & (1 << s) != 0 {
                u[n][s] = val;
            }
        }
    }

    /// Start of the step: the boundary conditions at t = 0 and θ̇ with them. The mechanics in
    /// `u` keeps the initial temperature (stress-free frame 0); the jump
    /// to the boundary conditions is computed in the first increment.
    pub fn start(&mut self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], st: &[Vec<f64>]) -> Result<()> {
        let mut ut = u.to_vec();
        self.apply_bc(&mut ut, 0.0);
        self.update_rate(p, x, &ut, 0.0, 0.0, st, st)
    }

    /// θ(i+1) = θ(i) + Δt θ̇(i) and the boundary conditions at t1.
    pub fn advance(&self, u1: &mut [NodeVec], h: f64, t1: f64) {
        for n in 0..u1.len() {
            u1[n][TS] = self.theta[n] + h * self.rate[n];
        }
        self.apply_bc(u1, t1);
    }

    /// Plastic heat per node (power) during an increment of length `h`.
    fn plastic_heat(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, old: &[Vec<f64>], new: &[Vec<f64>], h: f64) -> Vec<f64> {
        let m = p.model;
        let mut q = vec![0.0; m.nodes.len()];
        if h <= 0.0 {
            return q;
        }
        for (e, el) in m.elements.iter().enumerate() {
            if self.eta[e] == 0.0 || p.activity.is_removed(e) {
                continue;
            }
            let ctx = p.ctx(e, false, 0.0, 0.0);
            let Some(mat) = ctx.material else { continue };
            let Some(ip) = mat.output_names().iter().position(|n| *n == "PEEQ") else { continue };
            let nip = el.etype.n_ips();
            let (so, sn) = (&old[e], &new[e]);
            // the stress components according to the element's stress state; extra
            // state (NLGEOM, incompatible modes) lies after the points
            let nc = match el.etype.family() {
                Family::Continuum3D => 6,
                Family::Continuum2D if el.etype.name().starts_with("CPS") => 3,
                _ => 4,
            };
            let stride = nc + mat.n_state();
            if nip == 0 || sn.len() != so.len() || sn.len() < nip * stride {
                continue;
            }
            // the integration point volumes in the current configuration
            let xe = x(e);
            let vol = match geometry(el.etype.family(), el.etype.name(), &xe) {
                Some(g) => g.vol * if el.etype.family() == Family::Continuum2D { ctx.section.map_or(1.0, |s| s.solid_thickness()) } else { 1.0 },
                None => el.etype.volume_weights(&ctx).map(|w| w.iter().sum()).unwrap_or(0.0),
            };
            let mut w = 0.0;
            for k in 0..nip {
                let b = k * stride;
                let p0 = mat.output_values(&so[b + nc..b + stride]).get(ip).copied().unwrap_or(0.0);
                let p1 = mat.output_values(&sn[b + nc..b + stride]).get(ip).copied().unwrap_or(0.0);
                let dp = p1 - p0;
                if dp <= 0.0 {
                    continue;
                }
                w += mises(&sn[b..b + nc]) * dp / nip as f64;
            }
            if w == 0.0 {
                continue;
            }
            let share = self.eta[e] * w * vol / h / el.nodes.len() as f64;
            for &n in &el.nodes {
                q[n] += share;
            }
        }
        q
    }

    /// θ̇ = C⁻¹ (Q − F + Q_pl) in the state `u` at the step time `t`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_rate(&mut self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], t: f64, h: f64, old: &[Vec<f64>], new: &[Vec<f64>]) -> Result<()> {
        self.heat.source = self.plastic_heat(p, x, old, new, h);
        self.theta = u.iter().map(|v| v[TS]).collect();
        self.heat.u_committed = u.to_vec();
        // fluxes with step amplitude act fully already at t = 0
        let tt = (t / self.period).max(1e-12);
        self.heat.begin_increment(tt, tt)?;
        self.heat.u = u.to_vec();
        self.heat.evaluate(false)?;
        let nn = u.len();
        for n in 0..nn {
            self.rate[n] = if self.cap[n] > 0.0 && self.heat.dm.constrained[n] & (1 << TS) == 0 { (self.heat.fext[n][TS] - self.heat.fint[n][TS]) / self.cap[n] } else { 0.0 };
        }
        Ok(())
    }

    /// Thermal stability limit min Lₑ²/(2α) over the elements in the configuration
    /// `u`. Without NLGEOM the limit is multiplied by
    /// the bulk viscosity damping factor √(1 + ξ²) − ξ.
    pub(super) fn stable_in(&self, sys: &super::Explicit1, u: &[NodeVec], rates: &[f64]) -> (f64, usize) {
        let mut xi = vec![0.0; sys.p.model.elements.len()];
        if !sys.nlgeom {
            for (i, be) in sys.bulk.elems.iter().enumerate() {
                let xe = sys.coords(be.e, u);
                if let Some(g) = geometry(sys.p.model.elements[be.e].etype.family(), sys.p.model.elements[be.e].etype.name(), &xe) {
                    // the length of the quadratic term as in the bulk viscosity (s²·Lₑ)
                    let sc = sys.bulk.le_scale.get(be.e).copied().unwrap_or(1.0);
                    xi[be.e] = sys.bulk.xi(i, g.lchar * sc * sc, rates.get(i).copied().unwrap_or(0.0));
                }
            }
        }
        let (d, e) = self.stable(sys.p, &|e| sys.coords(e, u), u, &xi);
        // thermal contact (*GAP CONDUCTANCE): nodal limit
        let dc = self.contact_limit(u).unwrap_or(f64::INFINITY);
        if dc < d {
            return (dc, e);
        }
        (d, e)
    }

    /// Stability limit for nodes with thermal contact: 2Cₙ / Σⱼ|Kₙⱼ| with
    /// the conduction and contact matrices (Gershgorin).
    fn contact_limit(&self, u: &[NodeVec]) -> Result<f64> {
        if self.heat.gaps.is_empty() {
            return Ok(f64::INFINITY);
        }
        let model = self.heat.p.model;
        let nn = model.nodes.len();
        let mut rows = vec![0.0; nn];
        let mut touched = vec![false; nn];
        let mut dummy = vec![[0.0; crate::fem::dof::SLOTS]; nn];
        for g in &self.heat.gaps {
            for (nodes, ke) in g.contribute(u, TS, self.heat.gap_disp, 0.0, 0.0, &mut dummy, true)? {
                let m = nodes.len();
                for a in 0..m {
                    let r: f64 = (0..m).map(|b| ke[a * m + b].abs()).sum();
                    if r > 0.0 {
                        rows[nodes[a]] += r;
                        touched[nodes[a]] = true;
                    }
                }
            }
        }
        for (e, el) in model.elements.iter().enumerate() {
            let (Some(he), Some(pr)) = (crate::elements::heat::lookup(el.etype.name()), &self.props[e]) else { continue };
            if !el.nodes.iter().any(|&n| touched[n]) {
                continue;
            }
            let thick = el.section.map_or(1.0, |s| model.sections[s].solid_thickness());
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let t = el.nodes.iter().map(|&n| u[n][TS]).sum::<f64>() / el.nodes.len() as f64;
            let (kc, _) = pr.conductivity(t)?;
            let k = el.nodes.len();
            let mut km = vec![0.0; k * k];
            for ip in he.ips(&x, thick)? {
                for a in 0..k {
                    for b in 0..k {
                        let mut v = 0.0;
                        for i in 0..3 {
                            for j in 0..3 {
                                v += ip.dndx[a][i] * kc[i][j] * ip.dndx[b][j];
                            }
                        }
                        km[a * k + b] += v * ip.w;
                    }
                }
            }
            for a in 0..k {
                rows[el.nodes[a]] += (0..k).map(|b| km[a * k + b].abs()).sum::<f64>();
            }
        }
        let mut best = f64::INFINITY;
        for n in 0..nn {
            if touched[n] && rows[n] > 0.0 && self.cap[n] > 0.0 {
                best = best.min(2.0 * self.cap[n] / rows[n]);
            }
        }
        Ok(best)
    }

    /// The factor on Lₑ in the bulk viscosity per element: min(1, Δt_th/Δt_m)
    /// with the element's undamped thermal and mechanical limits: the linear
    /// bulk viscosity corresponds to b₁·Δt_th/Δt_m.
    pub(super) fn bulk_scale(&self, sys: &super::Explicit1, u: &[NodeVec]) -> Vec<f64> {
        let m = sys.p.model;
        let mut out = vec![1.0; m.elements.len()];
        let none = vec![0.0; m.elements.len()];
        let dth = self.limits(sys.p, &|e| sys.coords(e, u), u, &none);
        for (i, be) in sys.bulk.elems.iter().enumerate() {
            let el = &m.elements[be.e];
            let Some(g) = geometry(el.etype.family(), el.etype.name(), &sys.coords(be.e, u)) else { continue };
            let c = sys.bulk.wave_speed_of(i);
            if g.lchar > 0.0 && c > 0.0 && dth[be.e].is_finite() {
                out[be.e] = (dth[be.e] * c / g.lchar).min(1.0);
            }
        }
        out
    }

    /// Thermal stability limit min Lₑ²/(2α) · f(ξₑ) over the elements in the configuration `x`.
    fn stable(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], xi: &[f64]) -> (f64, usize) {
        let mut best = (f64::INFINITY, 0usize);
        for (e, d) in self.limits(p, x, u, xi).into_iter().enumerate() {
            if d < best.0 {
                best = (d, e);
            }
        }
        best
    }

    /// Thermal limit per element (infinite without a heat part).
    fn limits(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], xi: &[f64]) -> Vec<f64> {
        let m = p.model;
        let mut out = vec![f64::INFINITY; m.elements.len()];
        for (e, el) in m.elements.iter().enumerate() {
            let Some(pr) = &self.props[e] else { continue };
            if p.activity.is_removed(e) {
                continue;
            }
            let ctx = p.ctx(e, false, 0.0, 0.0);
            let rho = if pr.density > 0.0 { pr.density } else { ctx.density };
            let t = el.nodes.iter().map(|&n| u[n][TS]).sum::<f64>() / el.nodes.len().max(1) as f64;
            let k = pr.cond.as_ref().map(|c| c.1.value(t)).unwrap_or(0.0);
            let c = pr.specific_heat.as_ref().map(|c| c.value(t)).unwrap_or(0.0);
            if k <= 0.0 || c <= 0.0 || rho <= 0.0 {
                continue;
            }
            let xe = x(e);
            let lchar = match geometry(el.etype.family(), el.etype.name(), &xe) {
                Some(g) => g.lchar,
                None => continue,
            };
            out[e] = lchar * lchar * rho * c / (2.0 * k) * super::dt::damping_factor(xi[e]);
        }
        out
    }

    /// Puts temperatures, reaction fluxes and HFL into a frame.
    pub fn frame(&mut self, fr: &mut FrameData, u: &[NodeVec], t: f64) -> Result<()> {
        let mut u = u.to_vec();
        for (x, &th) in u.iter_mut().zip(&self.theta) {
            x[TS] = th;
        }
        let u = &u[..];
        self.heat.u = u.to_vec();
        self.heat.u_committed = u.to_vec();
        let tt = (t / self.period).max(1e-12);
        self.heat.begin_increment(tt, tt)?;
        self.heat.u = u.to_vec();
        self.heat.evaluate(false)?;
        let h = self.heat.frame(t, String::new())?;
        let model = self.heat.p.model;
        for n in 0..fr.u.len() {
            fr.u[n][TS] = h.u[n][TS];
            fr.rf[n][TS] = h.rf[n][TS];
            let bit = 1u16 << TS;
            fr.active[n] = (fr.active[n] & !bit) | (h.active[n] & bit);
            fr.constrained[n] = (fr.constrained[n] & !bit) | (h.constrained[n] & bit);
        }
        for (e, (x, y)) in fr.elem.iter_mut().zip(h.elem).enumerate() {
            let el = &model.elements[e];
            let nip = el.etype.n_ips();
            let centroid = self.centroid_flux(e, u)?;
            x.extend(y.into_iter().filter(|f| f.name == "HFL").map(|mut f| {
                // linear simplex elements (constant gradient) and reduced
                // elements have one integration point: the flux at
                // the element midpoint
                if nip == 1 && f.points.len() > 1 {
                    if let Some(c) = &centroid {
                        f.points = vec![(1, 0, c.clone())];
                    }
                }
                if nip == 1 {
                    f.points.truncate(1);
                    if let Some(p) = f.points.first_mut() {
                        p.0 = 1;
                    }
                }
                f
            }));
        }
        fr.node_extra.extend(h.node_extra);
        Ok(())
    }
}

impl Thermal<'_, '_> {
    /// Weights ∫Nᵢ dV / V for the integration point temperature in elements with one
    /// integration point and quadrilateral or hexahedral shape (CPE4RT, CPS4RT,
    /// CAX4RT, C3D8RT): the thermal strain corresponds to the volume-weighted
    /// nodal temperature. Empty vector = the mean value.
    pub fn temp_weights(&self) -> Result<Vec<Vec<f64>>> {
        let model = self.heat.p.model;
        let mut out = Vec::with_capacity(model.elements.len());
        for el in &model.elements {
            let he = crate::elements::heat::lookup(el.etype.name());
            let ok = el.etype.n_ips() == 1 && he.is_some_and(|h| matches!(h.shape, crate::elements::heat::Shape::Quad4 | crate::elements::heat::Shape::Hex8));
            if !ok {
                out.push(Vec::new());
                continue;
            }
            let he = he.unwrap();
            let thick = el.section.map_or(1.0, |s| model.sections[s].solid_thickness());
            let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
            let ips = he.ips(&x, thick)?;
            let mut w = vec![0.0; el.nodes.len()];
            let mut v = 0.0;
            for ip in &ips {
                for (wi, ni) in w.iter_mut().zip(&ip.n) {
                    *wi += ni * ip.w;
                }
                v += ip.w;
            }
            if v > 0.0 {
                w.iter_mut().for_each(|x| *x /= v);
            }
            out.push(w);
        }
        Ok(out)
    }

    /// HFL at the midpoint (ξ = 0) for quadrilaterals and hexahedra.
    fn centroid_flux(&self, e: usize, u: &[NodeVec]) -> Result<Option<Vec<f64>>> {
        let model = self.heat.p.model;
        let el = &model.elements[e];
        let (Some(he), Some(pr)) = (crate::elements::heat::lookup(el.etype.name()), &self.props[e]) else { return Ok(None) };
        if !matches!(he.shape, crate::elements::heat::Shape::Quad4 | crate::elements::heat::Shape::Hex8) {
            return Ok(None);
        }
        let thick = el.section.map_or(1.0, |s| model.sections[s].solid_thickness());
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| model.nodes[n]).collect();
        let ip = crate::elements::heat::ip_at(he.shape, &x, [0.0; 3], 1.0, thick, he.axisym)?;
        let t: Vec<f64> = el.nodes.iter().map(|&n| u[n][TS]).collect();
        let f = he.heat_flux(&[ip], pr, &t)?;
        Ok(f.points.into_iter().next().map(|p| p.2))
    }
}

/// von Mises stress from the components (3: plane stress, 4: plane strain/axisymmetry, 6: 3D).
fn mises(s: &[f64]) -> f64 {
    let (s11, s22, s33, s12, s13, s23) = match s.len() {
        3 => (s[0], s[1], 0.0, s[2], 0.0, 0.0),
        4 => (s[0], s[1], s[2], s[3], 0.0, 0.0),
        _ => (s[0], s[1], s[2], s[3], s[4], s[5]),
    };
    (0.5 * ((s11 - s22).powi(2) + (s22 - s33).powi(2) + (s33 - s11).powi(2)) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23)).sqrt()
}

/// The capacity mass for the temperature degrees of freedom is zero in the mechanical
/// mass; velocity and acceleration in the temperature slot are zeroed.
pub fn clear_ts(v: &mut [NodeVec]) {
    for x in v.iter_mut() {
        x[TS] = 0.0;
    }
}
