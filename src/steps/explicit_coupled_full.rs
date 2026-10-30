//! *DYNAMIC TEMPERATURE-DISPLACEMENT, EXPLICIT i den allmänna explicita
//! lösaren (`explicit.rs`, område coupled2): mekaniken som *DYNAMIC, EXPLICIT
//! (explicit hourglass-kontroll för CPE4RT/CPS4RT/C3D8RT, volymviskositet,
//! NLGEOM, kontakt, massskalning, energier) och temperaturen med framåt
//! Euler och lumpad värmekapacitet:
//!
//! ```text
//! θ̇(i)   = C⁻¹ (Q(i) − F(θ(i)) + Q_pl(i))
//! θ(i+1) = θ(i) + Δt(i+1) θ̇(i)
//! ```
//!
//! Värmeflödet F (ledning, *CFLUX, *DFLUX, *FILM, *RADIATE och termisk
//! kontakt *GAP CONDUCTANCE med spalten ur de aktuella förskjutningarna)
//! räknas med `steps/heat.rs` (`HeatSystem`). Q_pl är värmen från plastisk
//! dissipation, η σ:ε̇ᵖˡ med η från *INELASTIC HEAT FRACTION
//! (`materials/inelastic_heat.rs`), fördelad lika på elementets noder.
//!
//! Tidsinkrement: den mekaniska elementvisa gränsen (`explicit_dt`) och den
//! termiska Lₑ²/(2α) med samma karakteristiska längd Lₑ och α = k/(ρc),
//! omräknad varje inkrement med aktuell geometri och temperatur;
//! utan NLGEOM gånger volymviskositetens dämpningsfaktor (`stable_in`), och
//! noder med termisk kontakt får en nodvis gräns (`contact_limit`).
//!
//! Övrigt (kontrollerat i testfallen i
//! `tests/cases/coupled2/`): värmekapaciteten är radsummerad (∫Nᵢ ρc dV), de
//! reducerade elementens termiska töjning använder den volymviktade
//! nodtemperaturen (`temp_weights`), HFL redovisas i elementets mittpunkt och
//! volymviskositeten använder c_d·Δtₑ i stället för Lₑ när den termiska
//! gränsen styr elementet (`bulk_scale`).
//!
//! Mekaniken i stegets början räknas med begynnelsetemperaturen (frame 0
//! är spänningsfri); temperaturrandvillkoren verkar från första
//! inkrementet.

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

/// Procedurnamnet.
pub const PROC: &str = "DYNAMIC TEMPERATURE-DISPLACEMENT";

/// Värmeledningens och mekanikens laster.
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

/// Standard-elementet för explicit hourglass-kontroll (även de kopplade RT-elementen).
pub fn standard(name: &str) -> Option<&'static crate::elements::solid_higher::Solid> {
    super::solid::standard(name).or_else(|| crate::elements::heat_coupled::mech_name(name).and_then(super::solid::standard))
}

/// Explicit hourglass-kontroll för ett element; kopplade element har
/// temperaturen sist per nod, och den delen lämnas orörd.
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

/// Sätter nodtemperaturerna (temperaturen i slutet och i början av
/// inkrementet) i elementets kontext för kopplade element. Med vikter `w`
/// (reducerade element) får alla noder integrationspunktens temperatur
/// Σ wᵢ θᵢ, så att elementets medelvärde blir den.
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

/// Det termiska delsystemet.
pub struct Thermal<'p, 'm> {
    pub heat: HeatSystem<'p, 'm>,
    /// Lumpad värmekapacitet per nod.
    cap: Vec<f64>,
    /// θ̇ per nod.
    pub rate: Vec<f64>,
    /// Aktuella nodtemperaturer (med randvillkoren); i stegets början kan de
    /// skilja sig från temperaturerna som mekaniken senast räknades med.
    theta: Vec<f64>,
    /// Andel av den plastiska dissipationen som blir värme, per element.
    eta: Vec<f64>,
    /// Termiska egenskaper per element (`None` utan värmedel).
    props: Vec<Option<HeatProps>>,
    period: f64,
}

impl<'p, 'm> Thermal<'p, 'm> {
    /// Bygger delsystemet och returnerar det med mekanikens laster.
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
            // radsummerad (konsistent) kapacitet ∫Nᵢ ρc dV
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

    /// Föreskrivna temperaturer vid stegtiden `t`.
    fn apply_bc(&self, u: &mut [NodeVec], t: f64) {
        for (n, s, val) in self.heat.bc_now(t / self.period) {
            if self.heat.dm.constrained[n] & (1 << s) != 0 {
                u[n][s] = val;
            }
        }
    }

    /// Stegets början: randvillkoren vid t = 0 och θ̇ med dem. Mekaniken i
    /// `u` behåller begynnelsetemperaturen (spänningsfri frame 0); hoppet
    /// till randvillkoren räknas i första inkrementet.
    pub fn start(&mut self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], st: &[Vec<f64>]) -> Result<()> {
        let mut ut = u.to_vec();
        self.apply_bc(&mut ut, 0.0);
        self.update_rate(p, x, &ut, 0.0, 0.0, st, st)
    }

    /// θ(i+1) = θ(i) + Δt θ̇(i) och randvillkoren vid t1.
    pub fn advance(&self, u1: &mut [NodeVec], h: f64, t1: f64) {
        for n in 0..u1.len() {
            u1[n][TS] = self.theta[n] + h * self.rate[n];
        }
        self.apply_bc(u1, t1);
    }

    /// Plastisk värme per nod (effekt) under ett inkrement av längden `h`.
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
            // spänningens komponenter efter elementets spänningstillstånd; extra
            // tillstånd (NLGEOM, inkompatibla moder) ligger efter punkterna
            let nc = match el.etype.family() {
                Family::Continuum3D => 6,
                Family::Continuum2D if el.etype.name().starts_with("CPS") => 3,
                _ => 4,
            };
            let stride = nc + mat.n_state();
            if nip == 0 || sn.len() != so.len() || sn.len() < nip * stride {
                continue;
            }
            // integrationspunkternas volymer i aktuell konfiguration
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

    /// θ̇ = C⁻¹ (Q − F + Q_pl) i tillståndet `u` vid stegtiden `t`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_rate(&mut self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], t: f64, h: f64, old: &[Vec<f64>], new: &[Vec<f64>]) -> Result<()> {
        self.heat.source = self.plastic_heat(p, x, old, new, h);
        self.theta = u.iter().map(|v| v[TS]).collect();
        self.heat.u_committed = u.to_vec();
        // flöden med stegamplitud verkar fullt redan vid t = 0
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

    /// Termisk stabilitetsgräns min Lₑ²/(2α) över elementen i konfigurationen
    /// `u`. Utan NLGEOM multipliceras gränsen med
    /// volymviskositetens dämpningsfaktor √(1 + ξ²) − ξ.
    pub(super) fn stable_in(&self, sys: &super::Explicit1, u: &[NodeVec], rates: &[f64]) -> (f64, usize) {
        let mut xi = vec![0.0; sys.p.model.elements.len()];
        if !sys.nlgeom {
            for (i, be) in sys.bulk.elems.iter().enumerate() {
                let xe = sys.coords(be.e, u);
                if let Some(g) = geometry(sys.p.model.elements[be.e].etype.family(), sys.p.model.elements[be.e].etype.name(), &xe) {
                    // den kvadratiska termens längd som i volymviskositeten (s²·Lₑ)
                    let sc = sys.bulk.le_scale.get(be.e).copied().unwrap_or(1.0);
                    xi[be.e] = sys.bulk.xi(i, g.lchar * sc * sc, rates.get(i).copied().unwrap_or(0.0));
                }
            }
        }
        let (d, e) = self.stable(sys.p, &|e| sys.coords(e, u), u, &xi);
        // termisk kontakt (*GAP CONDUCTANCE): nodvis gräns
        let dc = self.contact_limit(u).unwrap_or(f64::INFINITY);
        if dc < d {
            return (dc, e);
        }
        (d, e)
    }

    /// Stabilitetsgräns för noder med termisk kontakt: 2Cₙ / Σⱼ|Kₙⱼ| med
    /// ledningens och kontaktens matriser (Gershgorin).
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

    /// Faktorn på Lₑ i volymviskositeten per element: min(1, Δt_th/Δt_m)
    /// med elementets odämpade termiska och mekaniska gränser: den linjära
    /// volymviskositeten motsvarar b₁·Δt_th/Δt_m.
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

    /// Termisk stabilitetsgräns min Lₑ²/(2α) · f(ξₑ) över elementen i konfigurationen `x`.
    fn stable(&self, p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, u: &[NodeVec], xi: &[f64]) -> (f64, usize) {
        let mut best = (f64::INFINITY, 0usize);
        for (e, d) in self.limits(p, x, u, xi).into_iter().enumerate() {
            if d < best.0 {
                best = (d, e);
            }
        }
        best
    }

    /// Termisk gräns per element (oändlig utan värmedel).
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

    /// Lägger temperaturer, reaktionsflöden och HFL i en frame.
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
                // linjära simplexelement (konstant gradient) och reducerade
                // element har en integrationspunkt: flödet i
                // elementets mittpunkt
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
    /// Vikter ∫Nᵢ dV / V för integrationspunktens temperatur i element med en
    /// integrationspunkt och fyrhörnings- eller hexaederform (CPE4RT, CPS4RT,
    /// CAX4RT, C3D8RT): den termiska töjningen svarar mot den volymviktade
    /// nodtemperaturen. Tom vektor = medelvärdet.
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

    /// HFL i mittpunkten (ξ = 0) för fyrhörningar och hexaedrar.
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

/// von Mises-spänning ur komponenterna (3: plan spänning, 4: plan töjning/axisymmetri, 6: 3D).
fn mises(s: &[f64]) -> f64 {
    let (s11, s22, s33, s12, s13, s23) = match s.len() {
        3 => (s[0], s[1], 0.0, s[2], 0.0, 0.0),
        4 => (s[0], s[1], s[2], s[3], 0.0, 0.0),
        _ => (s[0], s[1], s[2], s[3], s[4], s[5]),
    };
    (0.5 * ((s11 - s22).powi(2) + (s22 - s33).powi(2) + (s33 - s11).powi(2)) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23)).sqrt()
}

/// Kapacitetens massa för temperaturfrihetsgraderna är noll i den mekaniska
/// massan; hastighet och acceleration i temperaturens plats nollställs.
pub fn clear_ts(v: &mut [NodeVec]) {
    for x in v.iter_mut() {
        x[TS] = 0.0;
    }
}
