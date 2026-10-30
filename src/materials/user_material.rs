//! Användarmaterial: *USER MATERIAL, CONSTANTS=n (TYPE=MECHANICAL, UNSYMM)
//! och *DEPVAR. Materialet anropar en UMAT (`crate::user::umat`) som väljs
//! med materialnamnet: en registrerad Rust-UMAT (längsta prefix) eller UMAT:en
//! i biblioteket `RUSTYFEM_USER`.
//!
//! Tillståndet per integrationspunkt är STATEV(1..NSTATV) följt av
//! deformationsgradienten vid inkrementets slut (9 värden, radvis; blir
//! DFGRD0 i nästa inkrement) och SSE, SPD, SCD. Utdata SDV1..SDVn.
//!
//! DDSDDE symmetriseras (½(D + Dᵀ)) om inte *USER MATERIAL, UNSYMM anges;
//! med UNSYMM används DDSDDE som den är och steget väljer den osymmetriska
//! lösaren. COORDS, CELENT, NOEL, NPT, KSTEP, KINC, TIME(1)
//! (stegtid), PREDEF/DPRED (fältvariabler) kommer från `user::context`.
//! PNEWDT < 1 gör att inkrementet görs om med längden PNEWDT · Δt.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor;
use crate::model::{fail, Result};
use crate::user::umat::{self, Umat, UmatArgs};
use std::sync::{Arc, OnceLock};

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "användarmaterial (UMAT)",
    consumes: &["USER MATERIAL", "DEPVAR"],
    priority: 100,
    build: Some(build),
}];

pub struct UserMaterial {
    name: String,
    props: Vec<f64>,
    nstatv: usize,
    umat: Arc<dyn Umat>,
    /// *USER MATERIAL, UNSYMM: DDSDDE används osymmetrisk (annars ½(D + Dᵀ)).
    unsymm: bool,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("USER MATERIAL") else {
        // *DEPVAR för USDFLD (*USER DEFINED FIELD, materials/field_dependent.rs)
        if def.behavior("USER DEFINED FIELD").is_some() {
            return Ok(None);
        }
        if let Some(d) = def.behavior("DEPVAR") {
            return d.err("*DEPVAR utan *USER MATERIAL stöds inte");
        }
        return Ok(None);
    };
    if let Some(t) = kw.param_upper("TYPE") {
        if t != "MECHANICAL" {
            return kw.err(format!("TYPE={t} stöds inte ännu (bara MECHANICAL, dvs. UMAT)"));
        }
    }
    let n = match kw.param("CONSTANTS") {
        Some(v) => crate::inp::parse_label(v).filter(|&n| n >= 0).ok_or_else(|| crate::model::FemError(format!("{}: CONSTANTS={v} ska vara ett heltal", kw.location())))? as usize,
        None => 0,
    };
    let mut props = Vec::new();
    for d in kw.records() {
        props.extend(kw.floats(&d)?);
    }
    // tomma fält i slutet av sista raden räknas inte
    while props.len() > n && props.last() == Some(&0.0) {
        props.pop();
    }
    if props.len() != n {
        return kw.err(format!("CONSTANTS={n} men dataraderna har {} värden", props.len()));
    }
    let nstatv = match def.behavior("DEPVAR") {
        Some(d) => match d.first_floats()?.first() {
            Some(&v) if v >= 0.0 && v.fract() == 0.0 => v as usize,
            _ => return d.err("ange antalet tillståndsvariabler på dataraden"),
        },
        None => 0,
    };
    // en registrerad VUMAT (explicit dynamik) väljs av `user_vumat.rs`
    if crate::user::vumat::find(&def.name).is_some() {
        return Ok(None);
    }
    let Some(u) = umat::find(&def.name)? else {
        return kw.err(format!(
            "ingen UMAT för materialet {}: registrera en i Rust med rustyfem::user::umat::register (exempel: materialnamn som börjar med EXEMPEL_ELASTISK eller EXEMPEL_MISES) eller ange ett bibliotek med umat_ i RUSTYFEM_USER",
            def.name
        ));
    };
    let unsymm = kw.has("UNSYMM");
    Ok(Some(Arc::new(UserMaterial { name: def.name.clone(), props, nstatv, umat: u, unsymm })))
}

/// (NDI, NSHR) för spänningstillståndet.
fn ndi_nshr(mode: StressMode) -> (usize, usize) {
    match mode {
        StressMode::ThreeD => (3, 3),
        StressMode::PlaneStrain | StressMode::Axisymmetric => (3, 1),
        StressMode::PlaneStress => (2, 1),
        StressMode::Uniaxial => (1, 0),
    }
}

const I3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Inkrementell rotation (Hughes–Winget) mellan F0 och F1:
/// ΔR = (I − ½W)⁻¹ (I + ½W), W = skew((F1 − F0) F_½⁻¹), F_½ = ½(F0 + F1).
fn hughes_winget(f0: &[[f64; 3]; 3], f1: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let fm = tensor::scale(&tensor::add(f0, f1), 0.5);
    let (fmi, det) = tensor::inv(&fm);
    if det.abs() < 1e-300 {
        return I3;
    }
    let l = tensor::mul(&tensor::sub(f1, f0), &fmi);
    let w = tensor::skew(&l);
    let a = tensor::sub(&I3, &tensor::scale(&w, 0.5));
    let b = tensor::add(&I3, &tensor::scale(&w, 0.5));
    let (ai, _) = tensor::inv(&a);
    tensor::mul(&ai, &b)
}

/// "SDV1", "SDV2", ... med statisk livstid.
fn sdv_names(n: usize) -> Vec<&'static str> {
    static NAMES: OnceLock<std::sync::Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut v = NAMES.get_or_init(|| std::sync::Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
    while v.len() < n {
        let s: &'static str = Box::leak(format!("SDV{}", v.len() + 1).into_boxed_str());
        v.push(s);
    }
    v[..n].to_vec()
}

impl UserMaterial {
    fn f_index(&self) -> usize {
        self.nstatv
    }
}

impl MaterialModel for UserMaterial {
    fn name(&self) -> &str {
        "USER MATERIAL (UMAT)"
    }

    fn n_state(&self) -> usize {
        self.nstatv + 9 + 3
    }

    fn init_state(&self, state: &mut [f64]) {
        state.iter_mut().for_each(|v| *v = 0.0);
        let k = self.f_index();
        for i in 0..3 {
            state[k + 4 * i] = 1.0;
        }
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let ntens = input.mode.ncomp();
        let (ndi, nshr) = ndi_nshr(input.mode);
        let ns = self.nstatv;
        let k = self.f_index();
        let mut stress = input.stress_old.to_vec();
        let mut statev = state_old[..ns].to_vec();
        let f0: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| state_old[k + 3 * i + j]));
        let f1 = input.defgrad.unwrap_or(I3);
        let drot = if input.defgrad.is_some() { hughes_winget(&f0, &f1) } else { I3 };
        let (mut sse, mut spd, mut scd) = (state_old[k + 9], state_old[k + 10], state_old[k + 11]);
        let (mut rpl, mut drpldt) = (0.0, 0.0);
        let mut ddsdde = vec![0.0; ntens * ntens];
        let mut ddsddt = vec![0.0; ntens];
        let mut drplde = vec![0.0; ntens];
        let stran: Vec<f64> = input.strain.iter().zip(input.dstrain).map(|(e, de)| e - de).collect();
        let mut pnewdt = 1.0;
        let t0 = input.time - input.dt;
        // integrationspunkt, element, steg och fältvariabler (user/context.rs)
        let pt = crate::user::context::point_for_call();
        let (_, kinc) = crate::user::context::step_inc();
        let (fnew, fold) = crate::user::context::point_fields();
        let dpred: Vec<f64> = fnew.iter().zip(&fold).map(|(a, b)| a - b).collect();
        let mut a = UmatArgs {
            stress: &mut stress,
            statev: &mut statev,
            ddsdde: &mut ddsdde,
            sse: &mut sse,
            spd: &mut spd,
            scd: &mut scd,
            rpl: &mut rpl,
            ddsddt: &mut ddsddt,
            drplde: &mut drplde,
            drpldt: &mut drpldt,
            stran: &stran,
            dstran: input.dstrain,
            time: [crate::user::context::step_time(t0), t0],
            dtime: input.dt,
            temp: input.temperature - input.dtemp,
            dtemp: input.dtemp,
            predef: &fold,
            dpred: &dpred,
            cmname: &self.name,
            ndi,
            nshr,
            ntens,
            nstatv: ns,
            props: &self.props,
            coords: pt.coords,
            drot,
            pnewdt: &mut pnewdt,
            celent: pt.celent,
            dfgrd0: f0,
            dfgrd1: f1,
            noel: pt.noel,
            npt: pt.npt,
            layer: 1,
            kspt: 1,
            jstep: crate::user::context::jstep(input.defgrad.is_some()),
            kinc: kinc.max(0) as usize,
        };
        self.umat.umat(&mut a)?;
        if pnewdt < 1.0 {
            // inkrementet görs om med längden PNEWDT · Δt (steps/newton.rs)
            if crate::user::context::current().is_none() {
                return fail(format!("UMAT för {} begär mindre inkrement (PNEWDT = {pnewdt})", self.name));
            }
            crate::user::context::request_pnewdt(pnewdt);
        }
        state_new[..ns].copy_from_slice(&statev);
        for i in 0..3 {
            for j in 0..3 {
                state_new[k + 3 * i + j] = f1[i][j];
            }
        }
        state_new[k + 9] = sse;
        state_new[k + 10] = spd;
        state_new[k + 11] = scd;
        // DDSDDE kolumnvis → radvis, symmetriserad om inte UNSYMM
        let mut tangent = vec![0.0; ntens * ntens];
        for i in 0..ntens {
            for j in 0..ntens {
                tangent[i * ntens + j] = if self.unsymm { ddsdde[i + j * ntens] } else { 0.5 * (ddsdde[i + j * ntens] + ddsdde[j + i * ntens]) };
            }
        }
        Ok(MatOutput { stress, tangent })
    }

    fn requires_unsymmetric(&self) -> bool {
        self.unsymm
    }

    fn output_fields(&self, _mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        sdv_names(self.nstatv).into_iter().map(|n| (n, vec![n])).collect()
    }

    fn output_field_values(&self, _mode: StressMode, state: &[f64]) -> Vec<f64> {
        state[..self.nstatv].to_vec()
    }
}
