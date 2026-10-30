//! Skada i fiberarmerade kompositer: *ELASTIC, TYPE=LAMINA med *DAMAGE INITIATION,
//! CRITERION=HASHIN (ALPHA=α), *DAMAGE EVOLUTION, TYPE=ENERGY och
//! *DAMAGE STABILIZATION, i plan spänning (CPS-element och skal).
//!
//! Initieringen (Hashin) prövas för varje mod I på den effektiva spänningen
//! σ/(1 − dI) med modens egen skada (före skada är det
//! σ̂ = M σ, M = diag(1/(1 − df), 1/(1 − dm), 1/(1 − ds))), med data XT, XC,
//! YT, YC, SL, ST:
//! - fiberdrag (σ̂11 ≥ 0):    F = (σ̂11/XT)² + α (τ̂12/SL)²,
//! - fibertryck (σ̂11 < 0):   F = (σ̂11/XC)²,
//! - matrisdrag (σ̂22 ≥ 0):   F = (σ̂22/YT)² + (τ̂12/SL)²,
//! - matristryck (σ̂22 < 0):  F = (σ̂22/(2ST))² + [(YC/(2ST))² − 1] σ̂22/YC + (τ̂12/SL)².
//!
//! Utvecklingen per mod I följer den ekvivalenta förskjutningen
//! δeq (Lc = elementets karakteristiska längd, ε12 = ingenjörsskjuvningen γ12):
//! fiberdrag Lc√(⟨ε11⟩² + α ε12²), fibertryck Lc⟨−ε11⟩, matrisdrag
//! Lc√(⟨ε22⟩² + ε12²), matristryck Lc√(⟨−ε22⟩² + ε12²), och motsvarande
//! ekvivalenta effektiva spänning σ̂eq = Lc(⟨σ̂⟩⟨ε⟩ + τ̂12 ε12)/δeq. Vid
//! initieringen sätts δ0 = δeq/√F och δf = 2Gc/(σ̂eq/√F); därefter är
//! d = δf(δeq − δ0)/(δeq(δf − δ0)) (aldrig minskande). Blir 2Gc/σ0 ≤ δ0
//! (tillbakasnäpp) används δf = δ0 + Gc/σ0. Initieringen och
//! skadan bestäms implicit i inkrementet (fixpunktsiteration: σ̂ beror på d
//! genom Poissonkopplingen; δ0 räknas med den
//! slutliga skadan i initieringsinkrementet).
//!
//! df = dft eller dfc och dm = dmt eller dmc efter tecknet på σ̂11 och σ̂22,
//! ds = 1 − (1 − dft)(1 − dfc)(1 − dmt)(1 − dmc), och
//! σ = (1/D)[[(1−df)E1, (1−df)(1−dm)ν21E1, 0], [(1−df)(1−dm)ν12E2, (1−dm)E2, 0],
//! [0, 0, D(1−ds)G12]] ε med D = 1 − (1−df)(1−dm)ν12ν21. Viskös
//! regularisering (*DAMAGE STABILIZATION η per mod): dv = (η dv,n + Δt d)/(η + Δt);
//! de regulariserade värdena används i styvheten, i σ̂ och i utdata.
//!
//! Utdata: DAMAGEFT, DAMAGEFC, DAMAGEMT, DAMAGEMC, DAMAGESHR och
//! HSNFTCRT, HSNFCCRT, HSNMTCRT, HSNMCCRT (största F hittills, 1 efter
//! initiering). Tangenten är numerisk (centraldifferens) när skadan växer,
//! annars den skadade sekantstyvheten. Skal får Lc = 1 (sektionsintegreringen
//! skickar ingen längd).

use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{parse_f64, MaterialDef};
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Hashin-skada (fiberkompositer)",
    consumes: &["ELASTIC", "DAMAGE INITIATION", "DAMAGE EVOLUTION", "DAMAGE STABILIZATION"],
    priority: 40,
    build: Some(build),
}];

/// Sant om materialet har *DAMAGE INITIATION, CRITERION=HASHIN.
pub fn claims(def: &MaterialDef) -> bool {
    def.behaviors.iter().any(|b| b.name == "DAMAGE INITIATION" && b.param_upper("CRITERION").as_deref() == Some("HASHIN"))
}

/// Per mod (ft, fc, mt, mc): [d, dv, Fmax, δ0, δf].
const PER: usize = 5;
const NSTATE: usize = 4 * PER;

pub struct Hashin {
    e1: f64,
    e2: f64,
    nu12: f64,
    nu21: f64,
    g12: f64,
    /// XT, XC, YT, YC, SL, ST
    s: [f64; 6],
    alpha: f64,
    /// Brottenergier Gft, Gfc, Gmt, Gmc (`None` = ingen utveckling).
    gc: Option<[f64; 4]>,
    eta: [f64; 4],
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    let Some(ek) = def.behavior("ELASTIC") else {
        return fail(format!("material {}: CRITERION=HASHIN kräver *ELASTIC, TYPE=LAMINA", def.name));
    };
    if ek.param_upper("TYPE").as_deref() != Some("LAMINA") || ek.data.len() > 1 || ek.has("DEPENDENCIES") {
        return ek.err("CRITERION=HASHIN kräver *ELASTIC, TYPE=LAMINA (temperaturoberoende)");
    }
    let ev = ek.first_floats()?;
    let g = |v: &[f64], i: usize| v.get(i).copied().unwrap_or(0.0);
    let (e1, e2, nu12, g12) = (g(&ev, 0), g(&ev, 1), g(&ev, 2), g(&ev, 3));
    if !(e1 > 0.0 && e2 > 0.0 && g12 > 0.0) {
        return ek.err("E1, E2 och G12 ska vara positiva");
    }
    let nu21 = nu12 * e2 / e1;
    let mut init = None;
    let mut gc = None;
    let mut eta = [0.0; 4];
    for b in &def.behaviors {
        match b.name.as_str() {
            "DAMAGE INITIATION" => {
                if b.param_upper("CRITERION").as_deref() != Some("HASHIN") {
                    return b.err("CRITERION=HASHIN kan inte kombineras med andra kriterier");
                }
                if init.is_some() {
                    return b.err("bara ett *DAMAGE INITIATION, CRITERION=HASHIN per material");
                }
                if b.data.len() > 1 || b.has("DEPENDENCIES") {
                    return b.err("temperatur- eller fältberoende Hashin-data stöds inte ännu");
                }
                let alpha = match b.param("ALPHA") {
                    Some(s) => parse_f64(s).ok_or_else(|| FemError(format!("{}: ogiltigt ALPHA={s}", b.location())))?,
                    None => 0.0,
                };
                let v = b.first_floats()?;
                let s: [f64; 6] = std::array::from_fn(|i| g(&v, i));
                if s.iter().any(|&x| !(x > 0.0)) {
                    return b.err("Hashin kräver positiva XT, XC, YT, YC, SL och ST");
                }
                init = Some((s, alpha));
            }
            "DAMAGE EVOLUTION" => {
                let ty = b.param_upper("TYPE").unwrap_or_else(|| "ENERGY".into());
                let soft = b.param_upper("SOFTENING").unwrap_or_else(|| "LINEAR".into());
                if ty != "ENERGY" || soft != "LINEAR" {
                    return b.err("Hashin-skada stöder bara *DAMAGE EVOLUTION, TYPE=ENERGY, SOFTENING=LINEAR");
                }
                if b.data.len() > 1 || b.has("DEPENDENCIES") {
                    return b.err("temperatur- eller fältberoende skadeutveckling stöds inte ännu");
                }
                let v = b.first_floats()?;
                let gv: [f64; 4] = std::array::from_fn(|i| g(&v, i));
                if gv.iter().any(|&x| !(x > 0.0)) {
                    return b.err("brottenergierna Gft, Gfc, Gmt och Gmc ska vara positiva");
                }
                gc = Some(gv);
            }
            "DAMAGE STABILIZATION" => {
                let v = b.first_floats()?;
                eta = std::array::from_fn(|i| v.get(i).copied().unwrap_or(v.first().copied().unwrap_or(0.0)));
                if eta.iter().any(|&x| x < 0.0) {
                    return b.err("viskositeten ska vara icke-negativ");
                }
            }
            _ => {}
        }
    }
    let (s, alpha) = init.expect("claims");
    Ok(Some(Arc::new(Hashin { e1, e2, nu12, nu21, g12, s, alpha, gc, eta })))
}

/// ⟨x⟩ = max(x, 0)
fn mc(x: f64) -> f64 {
    x.max(0.0)
}

impl Hashin {
    /// Skadad styvhet (plan spänning, 3×3 radvis) för (df, dm, ds).
    fn stiffness(&self, df: f64, dm: f64, ds: f64) -> [f64; 9] {
        let (a, b) = (1.0 - df, 1.0 - dm);
        let d = 1.0 - a * b * self.nu12 * self.nu21;
        [a * self.e1 / d, a * b * self.nu21 * self.e1 / d, 0.0, a * b * self.nu12 * self.e2 / d, b * self.e2 / d, 0.0, 0.0, 0.0, (1.0 - ds) * self.g12]
    }

    /// Kriterievärden (ft, fc, mt, mc) ur σ̂; inaktiva moder får `None`.
    fn criteria(&self, sh: &[f64; 3]) -> [Option<f64>; 4] {
        let [xt, xc, yt, yc, sl, st] = self.s;
        let t = (sh[2] / sl).powi(2);
        let (mut f, mut m) = ([None; 2], [None; 2]);
        if sh[0] >= 0.0 {
            f[0] = Some((sh[0] / xt).powi(2) + self.alpha * t);
        } else {
            f[1] = Some((sh[0] / xc).powi(2));
        }
        if sh[1] >= 0.0 {
            m[0] = Some((sh[1] / yt).powi(2) + t);
        } else {
            m[1] = Some((sh[1] / (2.0 * st)).powi(2) + ((yc / (2.0 * st)).powi(2) - 1.0) * sh[1] / yc + t);
        }
        [f[0], f[1], m[0], m[1]]
    }

    /// Kriterievärdet för mod `m` ur spänningen `s` (utan teckenval).
    fn value(&self, m: usize, s: &[f64; 3]) -> Option<f64> {
        let [xt, xc, yt, yc, sl, st] = self.s;
        let t = (s[2] / sl).powi(2);
        Some(match m {
            0 => (s[0] / xt).powi(2) + self.alpha * t,
            1 => (s[0] / xc).powi(2),
            2 => (s[1] / yt).powi(2) + t,
            _ => (s[1] / (2.0 * st)).powi(2) + ((yc / (2.0 * st)).powi(2) - 1.0) * s[1] / yc + t,
        })
    }

    /// (δeq, σ̂eq) per mod.
    fn equivalent(&self, eps: &[f64], sh: &[f64; 3], lc: f64) -> [(f64, f64); 4] {
        let e12 = eps[2];
        let a = self.alpha;
        let eq = |d2: f64, w: f64| -> (f64, f64) {
            let d = lc * d2.max(0.0).sqrt();
            (d, if d > 0.0 { lc * lc * w / d } else { 0.0 })
        };
        [
            eq(mc(eps[0]).powi(2) + a * e12 * e12, mc(sh[0]) * mc(eps[0]) + a * sh[2] * e12),
            eq(mc(-eps[0]).powi(2), mc(-sh[0]) * mc(-eps[0])),
            eq(mc(eps[1]).powi(2) + e12 * e12, mc(sh[1]) * mc(eps[1]) + sh[2] * e12),
            eq(mc(-eps[1]).powi(2) + e12 * e12, mc(-sh[1]) * mc(-eps[1]) + sh[2] * e12),
        ]
    }

    /// Hela uppdateringen: spänning och nytt tillstånd. Returnerar även om
    /// skadan växte i inkrementet.
    fn stress(&self, eps: &[f64], lc: f64, dt: f64, old: &[f64], new: &mut [f64]) -> ([f64; 3], bool) {
        new.copy_from_slice(&old[..NSTATE]);
        let mut grew = false;
        let mut sig = [0.0; 3];
        // effektiva teckenval börjar från den oskadade spänningen
        let mut sh = {
            let c = self.stiffness(0.0, 0.0, 0.0);
            [c[0] * eps[0] + c[1] * eps[1], c[3] * eps[0] + c[4] * eps[1], c[8] * eps[2]]
        };
        for _ in 0..100 {
            let dv: [f64; 4] = std::array::from_fn(|m| new[m * PER + 1]);
            let df = if sh[0] >= 0.0 { dv[0] } else { dv[1] };
            let dm = if sh[1] >= 0.0 { dv[2] } else { dv[3] };
            let ds = 1.0 - (1.0 - dv[0]) * (1.0 - dv[1]) * (1.0 - dv[2]) * (1.0 - dv[3]);
            let c = self.stiffness(df, dm, ds);
            sig = [c[0] * eps[0] + c[1] * eps[1], c[3] * eps[0] + c[4] * eps[1], c[8] * eps[2]];
            // effektiv spänning σ̂ = M σ i sluten form (definierad även när d = 1)
            let dd = 1.0 - (1.0 - df) * (1.0 - dm) * self.nu12 * self.nu21;
            let shn = [
                (self.e1 * eps[0] + (1.0 - dm) * self.nu21 * self.e1 * eps[1]) / dd,
                ((1.0 - df) * self.nu12 * self.e2 * eps[0] + self.e2 * eps[1]) / dd,
                self.g12 * eps[2],
            ];
            // aktiva moder efter tecknen på σ̂; kriteriet för mod I prövas på
            // σ/(1 − dI) med modens egen skada (en initierad
            // matrismod förstärker inte skjuvtermen i fiberkriteriet)
            let active = self.criteria(&shn);
            let mut crit = [None; 4];
            let mut eqv = [(0.0, 0.0); 4];
            for m in 0..4 {
                if active[m].is_some() {
                    let k = 1.0 / (1.0 - dv[m]).max(1e-12);
                    let sm = [sig[0] * k, sig[1] * k, sig[2] * k];
                    crit[m] = self.value(m, &sm);
                    eqv[m] = self.equivalent(eps, &sm, lc)[m];
                } else {
                    eqv[m] = self.equivalent(eps, &shn, lc)[m];
                }
            }
            let crit_out = crit;
            let mut changed = false;
            for m in 0..4 {
                let o = m * PER;
                let initiated_before = old[o + 3] > 0.0;
                // initiering (δ0, δf) räknas om med aktuell skada tills den är stabil
                let mut fmax = old[o + 2];
                if let Some(f) = crit_out[m] {
                    fmax = fmax.max(f);
                }
                if let Some(f) = crit[m] {
                    if !initiated_before && f >= 1.0 && self.gc.is_some() {
                        let r = f.sqrt();
                        let (d0, s0) = (eqv[m].0 / r, eqv[m].1 / r);
                        new[o + 3] = d0;
                        let gm = self.gc.unwrap()[m];
                        let mut dfin = if s0 > 0.0 { 2.0 * gm / s0 } else { f64::INFINITY };
                        if dfin <= d0 {
                            // tillbakasnäpp: δf = δ0 + Gc/σ0
                            dfin = d0 + gm / s0;
                        }
                        new[o + 4] = dfin;
                    }
                }
                new[o + 2] = fmax;
                let (d0, dfin) = (new[o + 3], new[o + 4]);
                let mut d = old[o];
                if d0 > 0.0 {
                    let de = eqv[m].0;
                    if de > d0 {
                        let dn = if dfin <= d0 || de >= dfin { 1.0 } else { dfin * (de - d0) / (de * (dfin - d0)) };
                        if dn > d {
                            d = dn;
                            grew = true;
                        }
                    }
                }
                new[o] = d;
                let eta = self.eta[m];
                let dvn = if eta > 0.0 && dt > 0.0 { (eta * old[o + 1] + dt * d) / (eta + dt) } else if eta > 0.0 { old[o + 1] } else { d };
                if (dvn - new[o + 1]).abs() > 1e-15 {
                    changed = true;
                }
                new[o + 1] = dvn;
            }
            let sign_changed = (shn[0] >= 0.0) != (sh[0] >= 0.0) || (shn[1] >= 0.0) != (sh[1] >= 0.0);
            sh = shn;
            if !changed && !sign_changed {
                break;
            }
        }
        (sig, grew)
    }
}

impl MaterialModel for Hashin {
    fn name(&self) -> &str {
        "HASHIN"
    }

    fn n_state(&self) -> usize {
        NSTATE
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update_with_length(input, 1.0, state_old, state_new)
    }

    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        if input.mode != StressMode::PlaneStress {
            return fail("CRITERION=HASHIN gäller bara plan spänning (CPS-element och skal)");
        }
        let eps = input.strain;
        let (sig, grew) = self.stress(eps, length, input.dt, state_old, state_new);
        let tangent = if grew || (0..4).any(|m| state_new[m * PER + 1] != state_new[m * PER]) {
            // numerisk tangent av hela uppdateringen
            let h = 1e-7 * eps.iter().fold(1e-3f64, |m, x| m.max(x.abs()));
            let mut t = vec![0.0; 9];
            let mut tmp = vec![0.0; NSTATE];
            for j in 0..3 {
                let (mut ep, mut em) = (eps.to_vec(), eps.to_vec());
                ep[j] += h;
                em[j] -= h;
                let (sp, _) = self.stress(&ep, length, input.dt, state_old, &mut tmp);
                let (sm, _) = self.stress(&em, length, input.dt, state_old, &mut tmp);
                for i in 0..3 {
                    t[i * 3 + j] = (sp[i] - sm[i]) / (2.0 * h);
                }
            }
            t
        } else {
            let dv: [f64; 4] = std::array::from_fn(|m| state_new[m * PER + 1]);
            let df = if sig[0] >= 0.0 { dv[0] } else { dv[1] };
            let dm = if sig[1] >= 0.0 { dv[2] } else { dv[3] };
            let ds = 1.0 - (1.0 - dv[0]) * (1.0 - dv[1]) * (1.0 - dv[2]) * (1.0 - dv[3]);
            self.stiffness(df, dm, ds).to_vec()
        };
        Ok(MatOutput { stress: sig.to_vec(), tangent })
    }

    fn requires_unsymmetric(&self) -> bool {
        self.gc.is_some()
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        match mode {
            StressMode::PlaneStress => Ok(self.stiffness(0.0, 0.0, 0.0).to_vec()),
            _ => fail("CRITERION=HASHIN gäller bara plan spänning (CPS-element och skal)"),
        }
    }

    fn output_names(&self) -> &[&'static str] {
        &["DAMAGEFT", "DAMAGEFC", "DAMAGEMT", "DAMAGEMC", "DAMAGESHR", "HSNFTCRT", "HSNFCCRT", "HSNMTCRT", "HSNMCCRT"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        let dv: [f64; 4] = std::array::from_fn(|m| state[m * PER + 1]);
        let mut v = dv.to_vec();
        v.push(1.0 - (1.0 - dv[0]) * (1.0 - dv[1]) * (1.0 - dv[2]) * (1.0 - dv[3]));
        for m in 0..4 {
            let o = m * PER;
            v.push(if state[o + 3] > 0.0 { 1.0 } else { state[o + 2] });
        }
        v
    }
}
