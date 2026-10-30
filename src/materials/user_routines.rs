//! Material med användarrutiner (`user::routines`):
//!
//! - UHARD: `*PLASTIC, HARDENING=USER, PROPERTIES=n` (PROPS på dataraderna)
//!   med isotrop elasticitet: von Mises-plasticitet med isotropt hårdnande där
//!   flytspänningen σy(ε̄p) och HARD(1) = dσy/dε̄p ges av UHARD. Radiell
//!   återföring (Newton i Δε̄p) och sluten konsistent tangent; plan spänning
//!   och stänger med 3D-algoritmen och kondensation
//!   (`plasticity::update_reduced`). STATEV från *DEPVAR (utdata SDV1..).
//! - UEXPAN: `*EXPANSION, USER`: termiska töjningsinkrement från UEXPAN
//!   (1 värde isotropt, 3 ortotropt, 6 anisotropt med tensorskjuvning) dras av
//!   från töjningen innan det underliggande materialet uppdateras.
//!
//! Båda väljs före det vanliga registret (`materials::build` anropar
//! `claim`), eftersom de vanliga fabrikerna avvisar HARDENING=USER och USER.

use super::plasticity::{dev, iso_d6, mises, mv6, pe_labels, rotate_v6, update_reduced, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

/// *HEAT GENERATION läses av värmeledningen (HETVAL, `user/loads.rs`).
pub static FACTORIES: &[MaterialFactory] =
    &[MaterialFactory { name: "värmegenerering (HETVAL, läses av värmeledningen)", consumes: &["HEAT GENERATION"], priority: 0, build: None }];

/// Modellen om materialet använder UHARD eller UEXPAN.
pub fn claim(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    // UMATHT (*USER MATERIAL, TYPE=THERMAL): bara värmeledning (steps/heat.rs)
    if crate::user::loads::heat_user(def).0 {
        return Ok(Some(Arc::new(ThermalUser { name: def.name.clone() })));
    }
    if let Some(k) = def.behavior("EXPANSION").filter(|k| k.has("USER")) {
        let Some(routine) = crate::user::routines::find_uexpan(&def.name)? else {
            return crate::user::routines::missing("UEXPAN", "materialet", &def.name);
        };
        let ncomp = match k.param_upper("TYPE").as_deref().unwrap_or("ISO") {
            "ISO" => 1,
            "ORTHO" => 3,
            "ANISO" => 6,
            o => return k.err(format!("TYPE={o} stöds inte (ISO, ORTHO, ANISO)")),
        };
        let mut rest = def.clone();
        rest.behaviors.retain(|b| b.name != "EXPANSION");
        let inner = super::build(&rest)?;
        return Ok(Some(Arc::new(UserExpansion { name: def.name.clone(), ncomp, routine, inner })));
    }
    let Some(pk) = def.behavior("PLASTIC").filter(|k| k.param_upper("HARDENING").as_deref() == Some("USER")) else { return Ok(None) };
    if def.behavior("EXPANSION").is_some() {
        // termisk töjning ovanpå (thermal.rs bygger det inre materialet, som då når hit)
        let build = super::thermal::FACTORIES[0].build.expect("thermal har en fabrik");
        return build(def);
    }
    for b in &def.behaviors {
        if !matches!(b.name.as_str(), "ELASTIC" | "PLASTIC" | "DEPVAR" | "DENSITY") {
            return b.err("*PLASTIC, HARDENING=USER kan bara kombineras med *ELASTIC, *DEPVAR, *DENSITY och *EXPANSION");
        }
    }
    let Some(routine) = crate::user::routines::find_uhard(&def.name)? else {
        return crate::user::routines::missing("UHARD", "materialet", &def.name);
    };
    let Some(ek) = def.behavior("ELASTIC") else {
        return pk.err("*PLASTIC kräver *ELASTIC");
    };
    if ek.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || ek.data.len() > 1 || ek.has("DEPENDENCIES") {
        return ek.err("HARDENING=USER kräver konstant isotrop elasticitet");
    }
    let v = ek.first_floats()?;
    let (e, nu) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) {
        return ek.err("E ska vara positivt och ν mellan −1 och 0,5");
    }
    let mut props = Vec::new();
    for d in pk.records() {
        props.extend(pk.floats(&d)?);
    }
    if let Some(n) = pk.param("PROPERTIES").and_then(crate::inp::parse_label) {
        props.resize(n.max(0) as usize, 0.0);
    }
    let nstatv = match def.behavior("DEPVAR") {
        Some(d) => d.first_floats()?.first().map_or(0, |&x| x.max(0.0) as usize),
        None => 0,
    };
    Ok(Some(Arc::new(UserHardening { name: def.name.clone(), e, nu, g: e / (2.0 * (1.0 + nu)), ce: iso_d6(e, nu), props, nstatv, routine })))
}

// --------------------------------------------------------------------- UHARD

pub struct UserHardening {
    name: String,
    e: f64,
    nu: f64,
    g: f64,
    ce: M6,
    props: Vec<f64>,
    nstatv: usize,
    routine: Arc<dyn crate::user::routines::Uhard>,
}

/// Tillstånd: [ε̄p, εp (6), ε 3D (6), STATEV].
const P: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const SV: usize = 13;

struct Pt<'a> {
    input: &'a MatInput<'a>,
    pt: crate::user::context::Point,
    kstep: i32,
    kinc: i32,
}

impl UserHardening {
    /// σy och dσy/dε̄p vid ε̄p = `p` (UHARD), STATEV uppdateras i `statev`.
    fn hard(&self, c: &Pt, p: f64, dp: f64, statev: &mut [f64]) -> Result<(f64, f64)> {
        let (f, fo) = crate::user::context::point_fields();
        let dpred: Vec<f64> = f.iter().zip(&fo).map(|(a, b)| a - b).collect();
        let t0 = c.input.time - c.input.dt;
        let mut a = crate::user::routines::UhardArgs {
            syield: 0.0,
            hard: [0.0; 3],
            eqplas: p,
            eqplasrt: if c.input.dt > 0.0 { dp / c.input.dt } else { 0.0 },
            time: [crate::user::context::step_time(t0), t0],
            dtime: c.input.dt,
            temp: c.input.temperature,
            dtemp: c.input.dtemp,
            noel: c.pt.noel,
            npt: c.pt.npt,
            layer: 1,
            kspt: 1,
            kstep: c.kstep,
            kinc: c.kinc,
            cmname: &self.name,
            statev,
            predef: &fo,
            dpred: &dpred,
            props: &self.props,
        };
        self.routine.uhard(&mut a)?;
        if !(a.syield > 0.0) {
            return fail(format!("UHARD för {} gav flytspänningen {} (ska vara positiv)", self.name, a.syield));
        }
        Ok((a.syield, a.hard[0]))
    }

    fn update3d(&self, c: &Pt, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let ns = self.nstatv;
        stn[..SV + ns].copy_from_slice(&st[..SV + ns]);
        let p_n = st[P];
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        let s_tr = dev(&sig_tr);
        let q_tr = mises(&s_tr);
        let mut sv = st[SV..SV + ns].to_vec();
        let (sy0, _) = self.hard(c, p_n, 0.0, &mut sv)?;
        if q_tr - sy0 <= 1e-12 * sy0 {
            stn[SV..SV + ns].copy_from_slice(&sv);
            return Ok((sig_tr, self.ce));
        }
        let g = self.g;
        let mut dp = 0.0;
        let mut h = 0.0;
        let mut conv = false;
        for _ in 0..100 {
            sv.copy_from_slice(&st[SV..SV + ns]);
            let (sy, hh) = self.hard(c, p_n + dp, dp, &mut sv)?;
            h = hh;
            let f = q_tr - 3.0 * g * dp - sy;
            let df = -3.0 * g - hh;
            let step = -f / df;
            dp = (dp + step).max(0.0);
            if step.abs() <= 1e-14 * (1.0 + dp) || f.abs() <= 1e-12 * sy {
                conv = true;
                break;
            }
        }
        if !conv {
            return fail(format!("UHARD-plasticitet för {}: returalgoritmen konvergerade inte", self.name));
        }
        sv.copy_from_slice(&st[SV..SV + ns]);
        let (_, hh) = self.hard(c, p_n + dp, dp, &mut sv)?;
        h = if hh.is_finite() { hh } else { h };
        let theta = 1.0 - 3.0 * g * dp / q_tr;
        let nhat: V6 = std::array::from_fn(|i| 1.5 * s_tr[i] / q_tr); // n = 3/2 s/q
        let mut sig = sig_tr;
        for i in 0..6 {
            sig[i] -= 2.0 * g * dp * nhat[i];
        }
        stn[P] = p_n + dp;
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dp * nhat[i] * if i < 3 { 1.0 } else { 2.0 };
        }
        stn[SV..SV + ns].copy_from_slice(&sv);
        // konsistent tangent: K 1⊗1 + 2Gθ P_dev − 2G θ̄ m⊗m, m = s/|s|
        let k = self.e / (3.0 * (1.0 - 2.0 * self.nu));
        let snorm = (s_tr[0] * s_tr[0] + s_tr[1] * s_tr[1] + s_tr[2] * s_tr[2] + 2.0 * (s_tr[3] * s_tr[3] + s_tr[4] * s_tr[4] + s_tr[5] * s_tr[5])).sqrt();
        let m: V6 = std::array::from_fn(|i| s_tr[i] / snorm);
        let thb = 1.0 / (1.0 + h / (3.0 * g)) - (1.0 - theta);
        let mut d = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let pdev = if i < 3 && j < 3 { if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 } } else if i == j { 0.5 } else { 0.0 };
                let kk = if i < 3 && j < 3 { k } else { 0.0 };
                d[i][j] = kk + 2.0 * g * theta * pdev - 2.0 * g * thb * m[i] * m[j];
            }
        }
        Ok((sig, d))
    }
}

impl MaterialModel for UserHardening {
    fn name(&self) -> &str {
        "PLASTIC, HARDENING=USER (UHARD)"
    }

    fn n_state(&self) -> usize {
        SV + self.nstatv
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[EP..EP + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let (kstep, kinc) = crate::user::context::step_inc();
        let c = Pt { input, pt: crate::user::context::point_for_call(), kstep, kinc };
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(&c, e, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)];
        for n in sdv(self.nstatv) {
            v.push((n, vec![n]));
        }
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[P]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v.extend_from_slice(&state[SV..SV + self.nstatv]);
        v
    }
}

fn sdv(n: usize) -> Vec<&'static str> {
    use std::sync::{Mutex, OnceLock};
    static N: OnceLock<Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut v = N.get_or_init(|| Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
    while v.len() < n {
        let s: &'static str = Box::leak(format!("SDV{}", v.len() + 1).into_boxed_str());
        v.push(s);
    }
    v[..n].to_vec()
}

// --------------------------------------------------------------------- UEXPAN

pub struct UserExpansion {
    name: String,
    ncomp: usize,
    routine: Arc<dyn crate::user::routines::Uexpan>,
    inner: Arc<dyn MaterialModel>,
}

const NTH: usize = 6;

impl MaterialModel for UserExpansion {
    fn name(&self) -> &str {
        "EXPANSION, USER (UEXPAN)"
    }

    fn n_state(&self) -> usize {
        NTH + self.inner.n_state()
    }

    fn init_state(&self, state: &mut [f64]) {
        state[..NTH].iter_mut().for_each(|x| *x = 0.0);
        self.inner.init_state(&mut state[NTH..]);
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.inner.elastic_moduli()
    }

    fn is_linear_at_given_temperature(&self) -> bool {
        self.inner.is_linear()
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[..NTH], true);
        self.inner.rotate_state(r, &mut state[NTH..]);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mut expan = vec![0.0; self.ncomp];
        let mut dexp = vec![0.0; self.ncomp];
        let (f, fo) = crate::user::context::point_fields();
        let dpred: Vec<f64> = f.iter().zip(&fo).map(|(a, b)| a - b).collect();
        let t0 = input.time - input.dt;
        let pt = crate::user::context::point();
        let mut statev: Vec<f64> = Vec::new();
        let mut a = crate::user::routines::UexpanArgs {
            expan: &mut expan,
            dexpandt: &mut dexp,
            temp: [input.temperature, input.dtemp],
            time: [crate::user::context::step_time(t0), t0],
            dtime: input.dt,
            predef: &fo,
            dpred: &dpred,
            statev: &mut statev,
            cmname: &self.name,
            noel: pt.noel,
        };
        self.routine.uexpan(&mut a)?;
        // inkrement av termisk töjning (ingenjörsskjuvning i tillståndet)
        let mut d = [0.0; 6];
        match self.ncomp {
            1 => {
                d[0] = expan[0];
                d[1] = expan[0];
                d[2] = expan[0];
            }
            3 => d[..3].copy_from_slice(&expan[..3]),
            _ => {
                for i in 0..6 {
                    d[i] = expan[i] * if i < 3 { 1.0 } else { 2.0 };
                }
            }
        }
        let idx = super::anisotropic::mode_components(input.mode);
        let mut th = [0.0; NTH];
        for i in 0..NTH {
            th[i] = state_old[i] + d[i];
        }
        let mut eps = input.strain.to_vec();
        let mut deps = input.dstrain.to_vec();
        for (k, &i) in idx.iter().enumerate() {
            eps[k] -= th[i];
            deps[k] -= d[i];
        }
        state_new[..NTH].copy_from_slice(&th);
        let inp = MatInput { strain: &eps, dstrain: &deps, ..*input };
        self.inner.update(&inp, &state_old[NTH..], &mut state_new[NTH..])
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        self.inner.tangent_linear(mode)
    }

    fn time_dependence(&self) -> Option<&super::time_dependence::TimeDependence> {
        self.inner.time_dependence()
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.inner.output_fields(mode);
        let l: Vec<&'static str> = match mode {
            StressMode::ThreeD => vec!["THE11", "THE22", "THE33", "THE12", "THE13", "THE23"],
            StressMode::Uniaxial => vec!["THE11"],
            _ => vec!["THE11", "THE22", "THE33", "THE12"],
        };
        v.push(("THE", l));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = self.inner.output_field_values(mode, &state[NTH.min(state.len())..]);
        let idx: &[usize] = match mode {
            StressMode::ThreeD => &[0, 1, 2, 3, 4, 5],
            StressMode::Uniaxial => &[0],
            _ => &[0, 1, 2, 3],
        };
        v.extend(idx.iter().map(|&i| state.get(i).copied().unwrap_or(0.0)));
        v
    }
}

// --------------------------------------------------------------------- UMATHT

/// Material med *USER MATERIAL, TYPE=THERMAL: värmeledningen anropar UMATHT;
/// i mekaniska element ger det fel.
pub struct ThermalUser {
    name: String,
}

impl MaterialModel for ThermalUser {
    fn name(&self) -> &str {
        "USER MATERIAL, TYPE=THERMAL (UMATHT)"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, _input: &MatInput, _so: &[f64], _sn: &mut [f64]) -> Result<MatOutput> {
        fail(format!("material {}: *USER MATERIAL, TYPE=THERMAL (UMATHT) används i ett mekaniskt element", self.name))
    }
}
