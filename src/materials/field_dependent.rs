//! Material data that depends on field variables (DEPENDENCIES=n) and USDFLD
//! (*USER DEFINED FIELD).
//!
//! Behaviors with DEPENDENCIES=n have n field variables last in each entry (after
//! the temperature):
//!
//! - `*ELASTIC` (isotropic): E, ν, T, f1, ..., fn
//! - `*PLASTIC` (isotropic hardening): σ0, ε̄p, T, f1, ..., fn
//! - `*EXPANSION` (ISO, ORTHO, ANISO): α..., T, f1, ..., fn (ZERO= as usual)
//!
//! At most 8 values per data line; longer entries continue on the next line.
//! The entries are ordered with temperature as the fastest varying variable, then f1,
//! f2, ... Interpolation is linear in each variable,
//! hierarchically from the outermost (fn) inwards, with constant values outside
//! the table (`field_vars::DepTable`).
//!
//! Implementation: at each material point the field dependence is "resolved" – for the
//! current field values an ordinary material definition is built with tables in
//! temperature (the union of the table's temperatures and, for *PLASTIC,
//! plastic strains; the interpolation is then exactly the same) and the
//! ordinary model (elasticity, von Mises, temperature dependence) updates
//! the stress. The models are cached per field value. Thermal strain is computed here in
//! total form ε_th = α(T, f)(T − θ0) − α(T_I, f_I)(T_I − θ0).
//!
//! *USER DEFINED FIELD: USDFLD (`user::routines::Usdfld`, selected by
//! the material name) is called first at each point and sets the field variables at
//! the point (also for a UMAT's PREDEF). With *DEPVAR without *USER MATERIAL
//! USDFLD gets its own state variables (output SDV1..). The field values from
//! USDFLD are stored in the state (FV1..) and become the values at the start of the next
//! increment.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{DataLine, MaterialDef, RawKeyword};
use crate::materials::field_vars::DepTable;
use crate::model::{fail, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Selected before the registry (`materials::build` calls `claim` first), since
/// the ordinary factories reject DEPENDENCIES= with an error.
pub static FACTORIES: &[MaterialFactory] = &[];

/// The model if the material has field-dependent data or *USER DEFINED FIELD.
pub fn claim(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    build(def)
}

/// Number of field variables in DEPENDENCIES= (0 if the parameter is missing).
pub fn dependencies(kw: &RawKeyword) -> Result<usize> {
    match kw.param("DEPENDENCIES") {
        None => Ok(0),
        Some(v) if v.trim().is_empty() => Ok(0),
        Some(v) => match crate::inp::parse_label(v) {
            Some(n) if n >= 0 => Ok(n as usize),
            _ => kw.err(format!("DEPENDENCIES={v} must be an integer")),
        },
    }
}

/// Behaviors that support DEPENDENCIES here: (name, number of properties, internal variables).
fn layout(kw: &RawKeyword) -> Option<(usize, usize)> {
    match kw.name.as_str() {
        "ELASTIC" if kw.param_upper("TYPE").map_or(true, |t| t == "ISOTROPIC") => Some((2, 0)),
        "PLASTIC" if kw.param_upper("HARDENING").map_or(true, |t| t == "ISOTROPIC") => Some((1, 1)),
        "EXPANSION" => match kw.param_upper("TYPE").as_deref().unwrap_or("ISO") {
            "ISO" => Some((1, 0)),
            "ORTHO" => Some((3, 0)),
            "ANISO" => Some((6, 0)),
            _ => None,
        },
        _ => None,
    }
}

struct Dep {
    /// Index into the behaviors of the material definition.
    index: usize,
    table: DepTable,
    ndep: usize,
}

struct Usd {
    routine: Arc<dyn crate::user::routines::Usdfld>,
    /// Number of field variables set by USDFLD (largest DEPENDENCIES=, at least 1).
    nfield: usize,
    nstatv: usize,
}

struct Exp {
    table: DepTable,
    ncomp: usize,
    zero: f64,
}

pub struct FieldDependent {
    name: String,
    /// The definition without field-dependent behaviors (EXPANSION with dependencies and
    /// *USER DEFINED FIELD removed; *DEPVAR removed if USDFLD owns it).
    def: MaterialDef,
    deps: Vec<Dep>,
    exp: Option<Exp>,
    usd: Option<Usd>,
    /// The model at field values 0 (layout and output).
    reference: Arc<dyn MaterialModel>,
    cache: Mutex<HashMap<Vec<u64>, Arc<dyn MaterialModel>>>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let udf = def.behavior("USER DEFINED FIELD");
    let mut any = udf.is_some();
    for b in &def.behaviors {
        if dependencies(b)? > 0 && layout(b).is_some() {
            any = true;
        }
    }
    if !any {
        return Ok(None);
    }
    let mut rest = def.clone();
    rest.behaviors.retain(|b| b.name != "USER DEFINED FIELD");
    let mut exp = None;
    if let Some(k) = def.behavior("EXPANSION").filter(|k| dependencies(k).unwrap_or(0) > 0) {
        if k.has("USER") || k.has("PORE FLUID") {
            return k.err("USER and PORE FLUID together with DEPENDENCIES are not supported");
        }
        let (n, _) = layout(k).ok_or_else(|| crate::model::FemError(format!("{}: TYPE is not supported with DEPENDENCIES", k.location())))?;
        let zero = match k.param("ZERO") {
            Some(z) => crate::fem::model::parse_f64(z).ok_or_else(|| crate::model::FemError(format!("{}: ZERO={z} is not a number", k.location())))?,
            None => 0.0,
        };
        exp = Some(Exp { table: DepTable::parse(k, n, 0, dependencies(k)?)?, ncomp: n, zero });
        rest.behaviors.retain(|b| b.name != "EXPANSION");
    }
    let mut deps = Vec::new();
    let mut maxdep = 0;
    for (i, b) in rest.behaviors.iter().enumerate() {
        let n = dependencies(b)?;
        maxdep = maxdep.max(n);
        if n == 0 {
            continue;
        }
        let Some((np, inner)) = layout(b) else {
            return b.err(format!("DEPENDENCIES= is not yet supported for *{} (only *ELASTIC isotropic, *PLASTIC isotropic hardening, *EXPANSION and *CONDUCTIVITY)", b.name));
        };
        deps.push(Dep { index: i, table: DepTable::parse(b, np, inner, n)?, ndep: n });
    }
    if let Some(e) = &exp {
        maxdep = maxdep.max(e.table.nvars - 1);
    }
    let usd = match udf {
        Some(_) => {
            let Some(routine) = crate::user::routines::find_usdfld(&def.name)? else {
                return crate::user::routines::missing("USDFLD", "material", &def.name);
            };
            let mut nstatv = 0;
            if def.behavior("USER MATERIAL").is_none() {
                if let Some(d) = def.behavior("DEPVAR") {
                    nstatv = d.first_floats()?.first().map_or(0, |&v| v.max(0.0) as usize);
                    rest.behaviors.retain(|b| b.name != "DEPVAR");
                }
            }
            Some(Usd { routine, nfield: maxdep.max(1), nstatv })
        }
        None => None,
    };
    let mut m = FieldDependent {
        name: def.name.clone(),
        def: rest,
        deps,
        exp,
        usd,
        reference: Arc::new(super::elastic::IsotropicElastic { e: 1.0, nu: 0.0 }),
        cache: Mutex::new(HashMap::new()),
    };
    m.reference = m.resolved(&[])?;
    Ok(Some(Arc::new(m)))
}

fn fmt(v: f64) -> String {
    format!("{v:e}")
}

fn line(vals: &[f64], line: usize) -> DataLine {
    DataLine { fields: vals.iter().map(|&v| fmt(v)).collect(), line, continued: false }
}

fn sorted_unique(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(f64::total_cmp);
    v.dedup();
    v
}

impl FieldDependent {
    /// The material definition at field values `f` (missing values = 0).
    fn resolve(&self, f: &[f64]) -> MaterialDef {
        let mut def = self.def.clone();
        for d in &self.deps {
            let kw = &mut def.behaviors[d.index];
            let nf = d.ndep;
            let fv: Vec<f64> = (0..nf).map(|k| f.get(k).copied().unwrap_or(0.0)).collect();
            let tab = &d.table;
            let np = tab.nprops;
            let l = kw.line;
            kw.params.retain(|(k, _)| k != "DEPENDENCIES");
            kw.data.clear();
            if kw.name == "PLASTIC" {
                // [σ0 | ε̄p, T, f...]: the union of ε̄p and T
                // nprops = 2 (σ0, ε̄p), the variables [T, f...]
                let temps = sorted_unique(tab.rows.iter().map(|r| r[np]).collect());
                let eps = sorted_unique(tab.rows.iter().map(|r| r[1]).collect());
                for &t in &temps {
                    for &p in &eps {
                        let mut x = vec![p, t];
                        x.extend(&fv);
                        let s = self.plastic_eval(tab, &x);
                        let mut row = vec![s, p];
                        if temps.len() > 1 {
                            row.push(t);
                        }
                        kw.data.push(line(&row, l));
                    }
                }
            } else {
                let temps = sorted_unique(tab.rows.iter().map(|r| r[np]).collect());
                for &t in &temps {
                    let mut x = vec![t];
                    x.extend(&fv);
                    let mut row = tab.eval(&x);
                    if temps.len() > 1 {
                        row.push(t);
                    }
                    kw.data.push(line(&row, l));
                }
            }
        }
        def
    }

    /// σ0 at [ε̄p, T, f...]: the internal variable ε̄p is interpolated per curve.
    fn plastic_eval(&self, tab: &DepTable, x: &[f64]) -> f64 {
        // DepTable with nprops = 2 ([σ0, ε̄p]) and the variables [T, f...]:
        // interpolate the curves in ε̄p (constant after the last point) in each group
        let np = tab.nprops; // σ0 and ε̄p
        let nvars = tab.nvars;
        // groups (same T and f) in order
        let mut groups: Vec<(Vec<f64>, Vec<(f64, f64)>)> = Vec::new();
        for r in &tab.rows {
            let key = r[np..np + nvars].to_vec();
            match groups.last_mut() {
                Some((k, pts)) if *k == key => pts.push((r[1], r[0])),
                _ => groups.push((key, vec![(r[1], r[0])])),
            }
        }
        let p = x[0];
        let rows: Vec<Vec<f64>> = groups
            .iter()
            .map(|(k, pts)| {
                let s = if pts.len() == 1 || p <= pts[0].0 {
                    pts[0].1
                } else {
                    let mut v = pts[pts.len() - 1].1;
                    for w in pts.windows(2) {
                        if p < w[1].0 {
                            v = w[0].1 + (w[1].1 - w[0].1) / (w[1].0 - w[0].0) * (p - w[0].0);
                            break;
                        }
                    }
                    v
                };
                let mut r = vec![s];
                r.extend(k);
                r
            })
            .collect();
        DepTable { nprops: 1, nvars, rows }.eval(&x[1..])[0]
    }

    fn resolved(&self, f: &[f64]) -> Result<Arc<dyn MaterialModel>> {
        let nf = self.deps.iter().map(|d| d.ndep).max().unwrap_or(0);
        let key: Vec<u64> = (0..nf).map(|k| f.get(k).copied().unwrap_or(0.0).to_bits()).collect();
        if let Some(m) = self.cache.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
            return Ok(m.clone());
        }
        let m = super::build(&self.resolve(f))?;
        let mut c = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        if c.len() > 20000 {
            c.clear();
        }
        c.insert(key, m.clone());
        Ok(m)
    }

    fn nth(&self) -> usize {
        if self.exp.is_some() {
            6
        } else {
            0
        }
    }

    /// State layout: [thermal strain (6)?, FV (nfield)?, SDV (nstatv)?, internal].
    fn offsets(&self) -> (usize, usize, usize) {
        let a = self.nth();
        let b = a + self.usd.as_ref().map_or(0, |u| u.nfield);
        let c = b + self.usd.as_ref().map_or(0, |u| u.nstatv);
        (a, b, c)
    }

    /// α(T, f)(T − θ0) as strain (engineering shear).
    fn secant(&self, e: &Exp, t: f64, f: &[f64]) -> [f64; 6] {
        let mut x = vec![t];
        let nf = e.table.nvars - 1;
        x.extend((0..nf).map(|k| f.get(k).copied().unwrap_or(0.0)));
        let v = e.table.eval(&x);
        let mut a = [0.0; 6];
        match e.ncomp {
            1 => {
                a[0] = v[0];
                a[1] = v[0];
                a[2] = v[0];
            }
            n => a[..n].copy_from_slice(&v[..n]),
        }
        std::array::from_fn(|i| a[i] * (t - e.zero) * if i < 3 { 1.0 } else { 2.0 })
    }
}

impl MaterialModel for FieldDependent {
    fn name(&self) -> &str {
        "field-dependent material"
    }

    fn n_state(&self) -> usize {
        self.offsets().2 + self.reference.n_state()
    }

    fn init_state(&self, state: &mut [f64]) {
        let (_, _, c) = self.offsets();
        state[..c].iter_mut().for_each(|x| *x = 0.0);
        self.reference.init_state(&mut state[c..]);
    }

    fn init_state_at(&self, mode: StressMode, stress: &[f64], void_ratio: Option<f64>, state: &mut [f64]) -> Result<()> {
        let c = self.offsets().2;
        self.reference.init_state_at(mode, stress, void_ratio, &mut state[c..])
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        let (f, _) = crate::user::context::point_fields();
        self.resolved(&f).ok().and_then(|m| m.elastic_moduli())
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        let c = self.offsets().2;
        if self.exp.is_some() {
            super::plasticity::rotate_v6(r, &mut state[..6], true);
        }
        self.reference.rotate_state(r, &mut state[c..]);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let (a, b, c) = self.offsets();
        let (mut f, mut f_old) = crate::user::context::point_fields();
        if let Some(u) = &self.usd {
            let pt = crate::user::context::point();
            let (kstep, kinc) = crate::user::context::step_inc();
            let mut field: Vec<f64> = (0..u.nfield).map(|k| f.get(k).copied().unwrap_or(0.0)).collect();
            let mut statev = state_old[b..c].to_vec();
            let mut pnewdt = 1.0;
            let strain: Vec<f64> = input.strain.iter().zip(input.dstrain).map(|(e, d)| e - d).collect();
            let t0 = input.time - input.dt;
            let (ndi, nshr) = match input.mode {
                StressMode::ThreeD => (3, 3),
                StressMode::PlaneStrain | StressMode::Axisymmetric => (3, 1),
                StressMode::PlaneStress => (2, 1),
                StressMode::Uniaxial => (1, 0),
            };
            let mut args = crate::user::routines::UsdfldArgs {
                field: &mut field,
                statev: &mut statev,
                pnewdt: &mut pnewdt,
                celent: pt.celent,
                time: [crate::user::context::step_time(t0), t0],
                dtime: input.dt,
                cmname: &self.name,
                noel: pt.noel,
                npt: pt.npt.max(1),
                layer: 1,
                kspt: 1,
                kstep,
                kinc,
                ndi,
                nshr,
                coords: pt.coords,
                temp: input.temperature,
                stress: input.stress_old,
                strain: &strain,
            };
            u.routine.usdfld(&mut args)?;
            if pnewdt < 1.0 {
                crate::user::context::request_pnewdt(pnewdt);
            }
            // the field values at the start of the increment: those USDFLD gave in the previous increment
            f_old = state_old[a..b].to_vec();
            f = field.clone();
            state_new[a..b].copy_from_slice(&field);
            state_new[b..c].copy_from_slice(&statev);
            crate::user::context::set_user_fields(Some((f.clone(), f_old.clone())));
        }
        let model = self.resolved(&f)?;
        let out = if let Some(e) = &self.exp {
            let t = input.temperature;
            let t_old = t - input.dtemp;
            let (s1, s0) = (self.secant(e, t, &f), self.secant(e, t_old, &f_old));
            let idx = super::anisotropic::mode_components(input.mode);
            let mut th = [0.0; 6];
            for i in 0..6 {
                th[i] = state_old[i] + s1[i] - s0[i];
            }
            let mut eps = input.strain.to_vec();
            let mut deps = input.dstrain.to_vec();
            for (k, &i) in idx.iter().enumerate() {
                eps[k] -= th[i];
                deps[k] -= th[i] - state_old[i];
            }
            state_new[..6].copy_from_slice(&th);
            let inp = MatInput { strain: &eps, dstrain: &deps, ..*input };
            model.update(&inp, &state_old[c..], &mut state_new[c..])
        } else {
            model.update(input, &state_old[c..], &mut state_new[c..])
        };
        if self.usd.is_some() {
            crate::user::context::set_user_fields(None);
        }
        out
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let (f, _) = crate::user::context::point_fields();
        self.resolved(&f)?.tangent_linear(mode)
    }

    fn requires_unsymmetric(&self) -> bool {
        self.reference.requires_unsymmetric()
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v = self.reference.output_fields(mode);
        if let Some(u) = &self.usd {
            for (n, k) in names("SDV", u.nstatv).into_iter().map(|n| (n, vec![n])) {
                v.push((n, k));
            }
            v.push(("FV", names("FV", u.nfield)));
        }
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let (a, b, c) = self.offsets();
        if state.len() < c {
            return Vec::new();
        }
        let mut v = self.reference.output_field_values(mode, &state[c..]);
        if self.usd.is_some() {
            v.extend_from_slice(&state[b..c]);
            v.extend_from_slice(&state[a..b]);
        }
        v
    }
}

/// "SDV1", "FV1", ... with static lifetime.
fn names(prefix: &str, n: usize) -> Vec<&'static str> {
    use std::sync::OnceLock;
    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut m = NAMES.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|e| e.into_inner());
    (1..=n)
        .map(|k| {
            let s = format!("{prefix}{k}");
            *m.entry(s.clone()).or_insert_with(|| Box::leak(s.into_boxed_str()))
        })
        .collect()
}

/// Error if a material behavior with DEPENDENCIES is not supported.
pub fn unsupported(kw: &RawKeyword) -> Result<()> {
    fail(format!("{}: DEPENDENCIES= is not supported for *{}", kw.location(), kw.name))
}
