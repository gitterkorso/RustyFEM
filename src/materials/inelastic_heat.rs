//! *INELASTIC HEAT FRACTION (feature area coupled2): the fraction η of the plastic
//! dissipation σ:ε̇ᵖˡ that becomes heat. Read by the coupled steps
//! (`steps/explicit_coupled_full.rs`, `steps/heat_coupled_contact.rs`) and
//! the adiabatic analysis (`steps/adiabatic.rs`). Without a data line η = 0.9
//! (default).
//!
//! The registry has a high-priority factory that wraps the material
//! (built without *INELASTIC HEAT FRACTION) in `Adiabatic`: three extra
//! state variables last per integration point, [T, T_prev, active]. In
//! adiabatic steps (*STATIC, ADIABATIC / *DYNAMIC, ADIABATIC, active = 1)
//! the integration point temperature increases by ΔT = η σ:Δεᵖˡ / (ρ c(T)) after
//! each increment and the material is evaluated at that temperature (TEMP in
//! the output); otherwise the wrapper is transparent.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "fraction of heat from plastic dissipation (adiabatic temperature)",
    // the inner material is built (and checked) without *INELASTIC HEAT
    // FRACTION; listed here are all behaviors that underlying models consume
    consumes: &[
        "CAP HARDENING",
        "CAP PLASTICITY",
        "CLAY HARDENING",
        "CLAY PLASTICITY",
        "CONCRETE",
        "CONCRETE COMPRESSION DAMAGE",
        "CONCRETE COMPRESSION HARDENING",
        "CONCRETE DAMAGED PLASTICITY",
        "CONCRETE TENSION DAMAGE",
        "CONCRETE TENSION STIFFENING",
        "CREEP",
        "CYCLIC HARDENING",
        "DAMAGE EVOLUTION",
        "DAMAGE INITIATION",
        "DAMAGE STABILIZATION",
        "DEPVAR",
        "DRUCKER PRAGER",
        "DRUCKER PRAGER CREEP",
        "DRUCKER PRAGER HARDENING",
        "ELASTIC",
        "EOS",
        "EXPANSION",
        "FAILURE RATIOS",
        "HYPERELASTIC",
        "HYPERFOAM",
        "INELASTIC HEAT FRACTION",
        "MOHR COULOMB",
        "MOHR COULOMB HARDENING",
        "MULLINS EFFECT",
        "PLASTIC",
        "POROUS ELASTIC",
        "POTENTIAL",
        "RATE DEPENDENT",
        "SHEAR RETENTION",
        "SWELLING",
        "TENSILE FAILURE",
        "TENSION CUTOFF",
        "TENSION STIFFENING",
        "TRS",
        "USER MATERIAL",
        "VISCOELASTIC",
    ],
    priority: 200,
    build: Some(build),
}];

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(eta) = fraction(def)? else { return Ok(None) };
    let mut rest = def.clone();
    rest.behaviors.retain(|b| b.name != "INELASTIC HEAT FRACTION");
    let inner = super::build(&rest)?;
    let props = super::heat::HeatProps::from_def(def)?;
    let rho = if props.density > 0.0 {
        props.density
    } else {
        def.behavior("DENSITY").and_then(|k| k.first_floats().ok()).and_then(|v| v.first().copied()).unwrap_or(0.0)
    };
    Ok(Some(Arc::new(Adiabatic { inner, eta, rho, cap: props.specific_heat })))
}

/// Material with adiabatic heating (see the module description).
pub struct Adiabatic {
    pub inner: Arc<dyn MaterialModel>,
    pub eta: f64,
    rho: f64,
    cap: Option<super::heat::Table>,
}

/// Number of extra state variables: [T, T_prev, active].
pub const N_EXTRA: usize = 3;

/// The wrapper's name (`MaterialModel::name`).
pub const NAME: &str = "INELASTIC HEAT FRACTION";

impl Adiabatic {
    fn base(&self) -> usize {
        self.inner.n_state()
    }
}

impl MaterialModel for Adiabatic {
    fn name(&self) -> &str {
        NAME
    }
    fn is_linear_at_given_temperature(&self) -> bool {
        self.inner.is_linear_at_given_temperature()
    }
    fn n_state(&self) -> usize {
        self.base() + N_EXTRA
    }
    fn init_state(&self, state: &mut [f64]) {
        let b = self.base();
        self.inner.init_state(&mut state[..b]);
        state[b..b + N_EXTRA].iter_mut().for_each(|x| *x = 0.0);
    }
    fn init_state_at(&self, mode: StressMode, stress: &[f64], void_ratio: Option<f64>, state: &mut [f64]) -> Result<()> {
        let b = self.base();
        self.inner.init_state_at(mode, stress, void_ratio, &mut state[..b])
    }
    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }
    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update_with_length(input, 0.0, state_old, state_new)
    }
    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let b = self.base();
        let (old_in, old_x) = state_old.split_at(b);
        let (new_in, new_x) = state_new.split_at_mut(b);
        if old_x[2] == 0.0 {
            new_x[..N_EXTRA].copy_from_slice(&old_x[..N_EXTRA]);
            return self.inner.update_with_length(input, length, old_in, new_in);
        }
        // adiabatic: the integration point temperature at the start of the increment
        let t = old_x[0];
        let inp = MatInput { temperature: t, dtemp: t - old_x[1], ..*input };
        let out = self.inner.update_with_length(&inp, length, old_in, new_in)?;
        // ΔT = η σ:Δεᵖˡ / (ρc) with σ at the end of the increment
        let mut dt = 0.0;
        if let Some(ip) = self.inner.output_names().iter().position(|n| *n == "PEEQ") {
            let p0 = self.inner.output_values(old_in).get(ip).copied().unwrap_or(0.0);
            let p1 = self.inner.output_values(new_in).get(ip).copied().unwrap_or(0.0);
            let c = self.cap.as_ref().map(|c| c.value(t)).unwrap_or(0.0);
            if p1 > p0 && self.rho > 0.0 && c > 0.0 {
                let s = &out.stress;
                let sv: Vec<f64> = match input.mode {
                    StressMode::PlaneStress => s.iter().take(3).copied().collect(),
                    StressMode::ThreeD => s.clone(),
                    _ => s.iter().take(4).copied().collect(),
                };
                dt = self.eta * mises(&sv) * (p1 - p0) / (self.rho * c);
            }
        }
        new_x[0] = t + dt;
        new_x[1] = t;
        new_x[2] = old_x[2];
        Ok(out)
    }
    fn requires_unsymmetric(&self) -> bool {
        self.inner.requires_unsymmetric()
    }
    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        let b = self.base();
        self.inner.rotate_state(r, &mut state[..b]);
    }
    fn anneal(&self, strain: &[f64; 6], state: &mut [f64]) -> bool {
        let b = self.base();
        self.inner.anneal(strain, &mut state[..b])
    }
    fn current_bulk_modulus(&self, state: &[f64]) -> Option<f64> {
        self.inner.current_bulk_modulus(&state[..self.base().min(state.len())])
    }
    fn explicit_full_integration(&self) -> bool {
        self.inner.explicit_full_integration()
    }
    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }
    fn output_names(&self) -> &[&'static str] {
        self.inner.output_names()
    }
    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        self.inner.output_values(&state[..self.base().min(state.len())])
    }
    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        v.push(("TEMP", vec!["TEMP"]));
        v
    }
    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let b = self.base().min(state.len());
        let mut v = self.inner.output_field_values(mode, &state[..b]);
        v.push(state.get(b).copied().unwrap_or(0.0));
        v
    }
    fn time_dependence(&self) -> Option<&super::time_dependence::TimeDependence> {
        self.inner.time_dependence()
    }
}

/// Sets the adiabatic mode in the integration point material state `st`.
/// On the first activation the temperature becomes `t0` (the nodal temperature).
pub fn set_active(mat: &dyn MaterialModel, st: &mut [f64], on: bool, t0: f64) {
    if mat.name() != NAME || st.len() < mat.n_state() {
        return;
    }
    let b = mat.n_state() - N_EXTRA;
    if on && st[b + 2] == 0.0 && st[b] == 0.0 && st[b + 1] == 0.0 {
        st[b] = t0;
        st[b + 1] = t0;
    }
    st[b + 2] = if on { 1.0 } else { 0.0 };
}

/// η for the material, `None` if *INELASTIC HEAT FRACTION is missing.
pub fn fraction(def: &MaterialDef) -> Result<Option<f64>> {
    let Some(kw) = def.behavior("INELASTIC HEAT FRACTION") else { return Ok(None) };
    let v = kw.first_floats().unwrap_or_default();
    Ok(Some(v.first().copied().unwrap_or(0.9)))
}

/// von Mises stress from the components (3: plane stress, 4: plane
/// strain/axisymmetry, 6: 3D).
pub fn mises(s: &[f64]) -> f64 {
    let (s11, s22, s33, s12, s13, s23) = match s.len() {
        3 => (s[0], s[1], 0.0, s[2], 0.0, 0.0),
        4 => (s[0], s[1], s[2], s[3], 0.0, 0.0),
        _ => (s[0], s[1], s[2], s[3], s[4], s[5]),
    };
    (0.5 * ((s11 - s22).powi(2) + (s22 - s33).powi(2) + (s33 - s11).powi(2)) + 3.0 * (s12 * s12 + s13 * s13 + s23 * s23)).sqrt()
}

/// η per element (0 without *INELASTIC HEAT FRACTION).
pub fn fractions(model: &crate::fem::model::FeModel) -> Result<Vec<f64>> {
    let mut out = Vec::with_capacity(model.elements.len());
    for el in &model.elements {
        let mi = el.section.and_then(|s| model.sections[s].material_index);
        out.push(match mi {
            Some(i) => fraction(&model.materials[i])?.unwrap_or(0.0),
            None => 0.0,
        });
    }
    Ok(out)
}

/// Number of stress components for the element (the state layout).
pub fn stress_components(el: &crate::fem::model::Element) -> usize {
    use crate::elements::Family;
    match el.etype.family() {
        Family::Continuum3D => 6,
        Family::Continuum2D if el.etype.name().starts_with("CPS") => 3,
        _ => 4,
    }
}

/// Plastic dissipation per integration point and unit volume, σ:Δεᵖˡ = q·Δε̄ᵖˡ
/// (von Mises: the stress at the end of the increment), from the element states
/// before and after an increment. `None` if the element has no PEEQ.
pub fn ip_dissipation(p: &crate::fem::assembly::Prepared, e: usize, old: &[f64], new: &[f64]) -> Option<Vec<f64>> {
    let el = &p.model.elements[e];
    let ctx = p.ctx(e, false, 0.0, 0.0);
    let mat = ctx.material?;
    let ip = mat.output_names().iter().position(|n| *n == "PEEQ")?;
    let nip = el.etype.n_ips();
    // extra state (NLGEOM, incompatible modes) lies after the points
    let nc = stress_components(el);
    let stride = nc + mat.n_state();
    if nip == 0 || new.len() != old.len() || new.len() < nip * stride {
        return None;
    }
    Some(
        (0..nip)
            .map(|k| {
                let b = k * stride;
                let p0 = mat.output_values(&old[b + nc..b + stride]).get(ip).copied().unwrap_or(0.0);
                let p1 = mat.output_values(&new[b + nc..b + stride]).get(ip).copied().unwrap_or(0.0);
                if p1 > p0 {
                    // the mean stress over the increment (trapezoidal rule; the first
                    // plastic increment gives half the heat)
                    let avg: Vec<f64> = (0..nc).map(|i| 0.5 * (old[b + i] + new[b + i])).collect();
                    mises(&avg) * (p1 - p0)
                } else {
                    0.0
                }
            })
            .collect(),
    )
}

/// The heat η σ:Δεᵖˡ per node (energy) over an increment, ∫ Nᵢ r dV with
/// the heat element's shape functions: per integration point when the element's
/// mechanical and thermal points are equally many (CPE4T, C3D8T, ...), otherwise
/// with the element's mean value of r (reduced elements CPE4RT, C3D8RT).
pub fn nodal_heat(p: &crate::fem::assembly::Prepared, old: &[Vec<f64>], new: &[Vec<f64>], eta: &[f64]) -> Result<Vec<f64>> {
    let m = p.model;
    let mut q = vec![0.0; m.nodes.len()];
    for (e, el) in m.elements.iter().enumerate() {
        let eta_e = eta.get(e).copied().unwrap_or(0.0);
        if eta_e == 0.0 || p.activity.is_removed(e) {
            continue;
        }
        let Some(r) = ip_dissipation(p, e, &old[e], &new[e]) else { continue };
        if r.iter().all(|&x| x == 0.0) {
            continue;
        }
        let Some(he) = crate::elements::heat::lookup(el.etype.name()) else { continue };
        let thick = el.section.map_or(1.0, |s| m.sections[s].solid_thickness());
        let x: Vec<[f64; 3]> = el.nodes.iter().map(|&n| m.nodes[n]).collect();
        let ips = he.ips(&x, thick)?;
        let avg = r.iter().sum::<f64>() / r.len() as f64;
        for (k, ip) in ips.iter().enumerate() {
            let rk = if ips.len() == r.len() { r[k] } else { avg };
            for (a, &n) in el.nodes.iter().enumerate() {
                q[n] += eta_e * ip.n[a] * rk * ip.w;
            }
        }
    }
    Ok(q)
}
