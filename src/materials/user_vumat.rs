//! *USER MATERIAL med en registrerad VUMAT (`crate::user::vumat`) för
//! explicit dynamik. Fabriken väljs före UMAT-fabriken (`user_material.rs`)
//! när en VUMAT finns för materialnamnet; annars används UMAT:en.
//!
//! Tillståndet per integrationspunkt är STATEV(1..NSTATEV), ENERINTERN,
//! ENERINELAS och deformationsgradienten vid inkrementets slut (9 värden,
//! radvis; DEFGRADOLD i nästa inkrement). Utdata SDV1..SDVn.
//!
//! Spänningsvektorn i RustyFEM (11, 22, 33, 12, 13, 23 med ingenjörsskjuvtöjning)
//! görs om till VUMAT:s ordning 11, 22, 33, 12, 23, 31 med tensorskjuvning.
//! Bara 3D och plan töjning. Tangenten (för implicita steg) är den elastiska
//! från `Vumat::elastic_moduli`; VUMAT är avsedd för *DYNAMIC, EXPLICIT.

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::kinematics::tensor;
use crate::model::{fail, Result};
use crate::user::vumat::{self, Vumat, VumatArgs};
use std::sync::{Arc, OnceLock};

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "användarmaterial (VUMAT)",
    consumes: &["USER MATERIAL", "DEPVAR"],
    priority: 110,
    build: Some(build),
}];

pub struct VumatMaterial {
    name: String,
    props: Vec<f64>,
    nstatv: usize,
    density: f64,
    vumat: Arc<dyn Vumat>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("USER MATERIAL") else { return Ok(None) };
    let Some(v) = vumat::find(&def.name) else { return Ok(None) };
    let n = match kw.param("CONSTANTS") {
        Some(c) => crate::inp::parse_label(c).filter(|&n| n >= 0).ok_or_else(|| crate::model::FemError(format!("{}: CONSTANTS={c} ska vara ett heltal", kw.location())))? as usize,
        None => 0,
    };
    let mut props = Vec::new();
    for d in kw.records() {
        props.extend(kw.floats(&d)?);
    }
    while props.len() > n && props.last() == Some(&0.0) {
        props.pop();
    }
    if props.len() != n {
        return kw.err(format!("CONSTANTS={n} men dataraderna har {} värden", props.len()));
    }
    let nstatv = match def.behavior("DEPVAR") {
        Some(d) => match d.first_floats()?.first() {
            Some(&x) if x >= 0.0 && x.fract() == 0.0 => x as usize,
            _ => return d.err("ange antalet tillståndsvariabler på dataraden"),
        },
        None => 0,
    };
    let density = match def.behavior("DENSITY") {
        Some(d) => d.first_floats()?.first().copied().unwrap_or(0.0),
        None => 0.0,
    };
    Ok(Some(Arc::new(VumatMaterial { name: def.name.clone(), props, nstatv, density, vumat: v })))
}

fn sdv_names(n: usize) -> Vec<&'static str> {
    static NAMES: OnceLock<std::sync::Mutex<Vec<&'static str>>> = OnceLock::new();
    let mut v = NAMES.get_or_init(|| std::sync::Mutex::new(Vec::new())).lock().unwrap_or_else(|e| e.into_inner());
    while v.len() < n {
        let s: &'static str = Box::leak(format!("SDV{}", v.len() + 1).into_boxed_str());
        v.push(s);
    }
    v[..n].to_vec()
}

/// RustyFEM-index för VUMAT-komponent i (3D: 11, 22, 33, 12, 23, 31).
const MAP3D: [usize; 6] = [0, 1, 2, 3, 5, 4];

fn defgrad9(f: &[[f64; 3]; 3]) -> [f64; 9] {
    [f[0][0], f[1][1], f[2][2], f[0][1], f[1][2], f[2][0], f[1][0], f[2][1], f[0][2]]
}

fn stretch6(f: &[[f64; 3]; 3]) -> [f64; 6] {
    let (_, u) = tensor::polar(f);
    [u[0][0], u[1][1], u[2][2], u[0][1], u[1][2], u[2][0]]
}

impl VumatMaterial {
    fn elastic(&self, mode: StressMode) -> Vec<f64> {
        let n = mode.ncomp();
        let mut d = vec![0.0; n * n];
        if let Some((e, nu)) = self.vumat.elastic_moduli(&self.props) {
            let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            let mu = e / (2.0 * (1.0 + nu));
            let nd = if mode == StressMode::PlaneStress { 2 } else { 3.min(n) };
            for i in 0..nd {
                for j in 0..nd {
                    d[i * n + j] = lam;
                }
                d[i * n + i] += 2.0 * mu;
            }
            for i in nd..n {
                d[i * n + i] = mu;
            }
        }
        d
    }
}

impl MaterialModel for VumatMaterial {
    fn name(&self) -> &str {
        "USER MATERIAL (VUMAT)"
    }

    fn is_linear(&self) -> bool {
        false
    }

    fn n_state(&self) -> usize {
        self.nstatv + 2 + 9
    }

    fn init_state(&self, state: &mut [f64]) {
        state.iter_mut().for_each(|v| *v = 0.0);
        let k = self.nstatv + 2;
        for i in 0..3 {
            state[k + 4 * i] = 1.0;
        }
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        self.vumat.elastic_moduli(&self.props)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        if self.vumat.elastic_moduli(&self.props).is_none() {
            return fail(format!("VUMAT för {} saknar elastiska moduler (Vumat::elastic_moduli) för stabilitetsgränsen", self.name));
        }
        Ok(self.elastic(mode))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let (ndir, nshr, map): (usize, usize, &[usize]) = match input.mode {
            StressMode::ThreeD => (3, 3, &MAP3D),
            StressMode::PlaneStrain => (3, 1, &MAP3D[..4]),
            m => return fail(format!("VUMAT för {}: spänningstillståndet {m:?} stöds inte (bara 3D och plan töjning)", self.name)),
        };
        let n = ndir + nshr;
        let strain_inc: Vec<f64> = (0..n).map(|i| input.dstrain[map[i]] * if i < ndir { 1.0 } else { 0.5 }).collect();
        let stress_old: Vec<f64> = (0..n).map(|i| input.stress_old[map[i]]).collect();
        let ns = self.nstatv;
        let k = ns + 2;
        let f0: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| state_old[k + 3 * i + j]));
        let f1 = input.defgrad.unwrap_or(f0);
        let (_, j1) = tensor::inv(&f1);
        let density = if j1 > 0.0 { self.density / j1 } else { self.density };
        // relativt spinn ur ΔR (Hughes–Winget) mellan F0 och F1
        let fm = tensor::scale(&tensor::add(&f0, &f1), 0.5);
        let (fmi, _) = tensor::inv(&fm);
        let w = tensor::skew(&tensor::mul(&tensor::sub(&f1, &f0), &fmi));
        let mut stress_new = vec![0.0; n];
        let mut statev = vec![0.0; ns];
        let (mut ei, mut en) = (state_old[ns], state_old[ns + 1]);
        let t0 = input.time - input.dt;
        let mut a = VumatArgs {
            ndir,
            nshr,
            nstatev: ns,
            lanneal: 0,
            step_time: t0,
            total_time: t0,
            dt: input.dt,
            cmname: &self.name,
            coord_mp: [0.0; 3],
            char_length: 0.0,
            props: &self.props,
            density,
            strain_inc: &strain_inc,
            rel_spin_inc: [w[2][1], w[0][2], w[1][0]],
            temp_old: input.temperature - input.dtemp,
            temp_new: input.temperature,
            stretch_old: stretch6(&f0),
            stretch_new: stretch6(&f1),
            defgrad_old: defgrad9(&f0),
            defgrad_new: defgrad9(&f1),
            stress_old: &stress_old,
            state_old: &state_old[..ns],
            ener_intern_old: ei,
            ener_inelas_old: en,
            stress_new: &mut stress_new,
            state_new: &mut statev,
            ener_intern_new: &mut ei,
            ener_inelas_new: &mut en,
        };
        self.vumat.vumat(&mut a)?;
        let mut stress = input.stress_old.to_vec();
        for i in 0..n {
            stress[map[i]] = stress_new[i];
        }
        state_new[..ns].copy_from_slice(&statev);
        state_new[ns] = ei;
        state_new[ns + 1] = en;
        for i in 0..3 {
            for j in 0..3 {
                state_new[k + 3 * i + j] = f1[i][j];
            }
        }
        Ok(MatOutput { stress, tangent: self.elastic(input.mode) })
    }

    fn output_fields(&self, _mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        sdv_names(self.nstatv).into_iter().map(|n| (n, vec![n])).collect()
    }

    fn output_field_values(&self, _mode: StressMode, state: &[f64]) -> Vec<f64> {
        state[..self.nstatv].to_vec()
    }
}
