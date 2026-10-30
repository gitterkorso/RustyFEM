//! Viskoelasticitet i frekvensplanet (*VISCOELASTIC, FREQUENCY=PRONY och
//! FREQUENCY=TABULAR) med isotrop *ELASTIC, för *STEADY STATE DYNAMICS, DIRECT,
//! valfritt med *TRS (reducerad frekvens ωr = A(θ) ω).
//!
//! Viskoelasticitet i frekvensplanet: med de
//! långtida modulerna G∞, K∞ (från *ELASTIC) är lagrings- och förlustmodulerna
//!
//!   Gs(ω) = G∞ (1 − ω ℑ(g*)),  Gℓ(ω) = G∞ ω ℜ(g*)   (och likadant för K),
//!
//! - FREQUENCY=TABULAR: dataraderna är ω ℜ(g*), ω ℑ(g*), ω ℜ(k*), ω ℑ(k*) och
//!   frekvensen f (cykler per tidsenhet), linjär interpolation i f och
//!   konstanta värden utanför tabellen,
//! - FREQUENCY=PRONY: (gᵢ, kᵢ, τᵢ) med Gs = G0 (1 − Σgᵢ) + G0 Σ gᵢ τᵢ²ω²/(1 + τᵢ²ω²),
//!   Gℓ = G0 Σ gᵢ τᵢ ω/(1 + τᵢ²ω²), G0 = G∞/(1 − Σgᵢ) (MODULI=LONG TERM,
//!   standard) eller G0 från *ELASTIC (MODULI=INSTANTANEOUS).
//!
//! I tidsplanets steg (och för grundtillståndet i *STATIC) är materialet
//! linjärt elastiskt med de långtida modulerna. I
//! *STEADY STATE DYNAMICS, DIRECT lägger `frequency_matrices` till
//! elementmatriserna för lagringsmodulernas tillskott (Gs − G∞, Ks − K∞) i den
//! reella delen och för förlustmodulerna (Gℓ, Kℓ) i den imaginära delen av den
//! dynamiska styvheten; elementmatriserna är linjära i (G, K) och räknas med
//! ett hjälpmaterial (`IsoGk`).

use super::anisotropic::reduce_6x6;
use super::time_dependence::TimeDependence;
use super::viscoelastic_trs::Trs;
use super::{matvec, MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::elements::Family;
use crate::fem::assembly::{Prepared, NONE};
use crate::fem::model::MaterialDef;
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::{SymCsr, Triplets};
use crate::steps::frequency::AllDofs;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "viskoelasticitet i frekvensplanet (*VISCOELASTIC, FREQUENCY=PRONY/TABULAR)",
    consumes: &["ELASTIC", "VISCOELASTIC", "TRS"],
    priority: 25,
    build: Some(build),
}];

/// Om materialet har viskoelasticitet i frekvensplanet (byggs här, inte i `viscoelastic.rs`).
pub fn claims(def: &MaterialDef) -> bool {
    def.behavior("VISCOELASTIC").is_some_and(|k| k.has("FREQUENCY"))
}

#[derive(Debug, Clone)]
enum Freq {
    /// G0, K0 och (gᵢ, kᵢ, τᵢ).
    Prony { g0: f64, k0: f64, terms: Vec<(f64, f64, f64)> },
    /// [ω ℜ(g*), ω ℑ(g*), ω ℜ(k*), ω ℑ(k*), f] i stigande f.
    Tabular(Vec<[f64; 5]>),
}

#[derive(Debug)]
pub struct ViscoFreq {
    e: f64,
    nu: f64,
    g_inf: f64,
    k_inf: f64,
    data: Freq,
    trs: Option<Trs>,
    td: TimeDependence,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    if !claims(def) {
        return Ok(None);
    }
    let vk = def.behavior("VISCOELASTIC").unwrap();
    let Some(ek) = def.behavior("ELASTIC") else {
        return vk.err("*VISCOELASTIC, FREQUENCY= stöds bara med *ELASTIC (inte *HYPERELASTIC) ännu");
    };
    if ek.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || ek.data.len() > 1 || ek.has("DEPENDENCIES") {
        return ek.err("*VISCOELASTIC, FREQUENCY= stöds bara med isotrop, temperaturoberoende *ELASTIC");
    }
    for p in ["TYPE", "PRELOAD"] {
        if vk.has(p) && !(p == "TYPE" && vk.param_upper("TYPE").as_deref() == Some("ISOTROPIC")) {
            return vk.err(format!("parametern {p} stöds inte ännu"));
        }
    }
    let inst = match ek.param_upper("MODULI").as_deref() {
        None | Some("LONG TERM") => false,
        Some("INSTANTANEOUS") => true,
        Some(o) => return ek.err(format!("MODULI={o} ska vara LONG TERM eller INSTANTANEOUS")),
    };
    let ev = ek.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0 && nu > -1.0 && nu < 0.5) {
        return ek.err("E ska vara positivt och ν mellan −1 och 0,5");
    }
    let (ge, ke) = (e / (2.0 * (1.0 + nu)), e / (3.0 * (1.0 - 2.0 * nu)));
    let (data, g_inf, k_inf) = match vk.param_upper("FREQUENCY").as_deref() {
        Some("PRONY") => {
            let mut terms = Vec::new();
            for d in vk.records() {
                let v = vk.floats(&d)?;
                let (g, k, tau) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0));
                if !(g >= 0.0 && k >= 0.0 && tau > 0.0) {
                    return vk.err_at(d.line, "Prony-termen kräver gᵢ ≥ 0, kᵢ ≥ 0 och τᵢ > 0");
                }
                terms.push((g, k, tau));
            }
            if terms.is_empty() {
                return vk.err("Prony-serien saknas");
            }
            let (sg, sk): (f64, f64) = (terms.iter().map(|t| t.0).sum(), terms.iter().map(|t| t.1).sum());
            if !(sg < 1.0 && sk < 1.0) {
                return vk.err("summan av gᵢ och summan av kᵢ ska vara mindre än 1");
            }
            let (g0, k0, gi, ki) = if inst { (ge, ke, ge * (1.0 - sg), ke * (1.0 - sk)) } else { (ge / (1.0 - sg), ke / (1.0 - sk), ge, ke) };
            (Freq::Prony { g0, k0, terms }, gi, ki)
        }
        Some("TABULAR") => {
            if inst {
                return ek.err("MODULI=INSTANTANEOUS kan inte användas med FREQUENCY=TABULAR (modulerna är långtida)");
            }
            let mut t: Vec<[f64; 5]> = Vec::new();
            for d in vk.records() {
                let v = vk.floats(&d)?;
                let row: [f64; 5] = std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0));
                if let Some(l) = t.last() {
                    if !(row[4] > l[4]) {
                        return vk.err_at(d.line, "frekvenserna ska öka");
                    }
                }
                t.push(row);
            }
            if t.is_empty() {
                return vk.err("tabellen saknas");
            }
            (Freq::Tabular(t), ge, ke)
        }
        Some(o) => return vk.err(format!("FREQUENCY={o} stöds inte ännu (PRONY, TABULAR)")),
        None => unreachable!(),
    };
    // långtida (E, ν) för det elastiska svaret
    let e_inf = 9.0 * k_inf * g_inf / (3.0 * k_inf + g_inf);
    let nu_inf = (3.0 * k_inf - 2.0 * g_inf) / (2.0 * (3.0 * k_inf + g_inf));
    let trs = def.behavior("TRS").map(Trs::parse).transpose()?;
    Ok(Some(Arc::new(ViscoFreq { e: e_inf, nu: nu_inf, g_inf, k_inf, data, trs, td: TimeDependence::default() })))
}

impl ViscoFreq {
    fn d6(&self) -> super::plasticity::M6 {
        super::plasticity::iso_d6(self.e, self.nu)
    }
}

impl MaterialModel for ViscoFreq {
    fn name(&self) -> &str {
        "VISCOELASTIC"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let tangent = reduce_6x6(&self.d6(), input.mode);
        Ok(MatOutput { stress: matvec(&tangent, input.strain), tangent })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(reduce_6x6(&self.d6(), mode))
    }

    fn time_dependence(&self) -> Option<&TimeDependence> {
        Some(&self.td)
    }

    fn frequency_moduli(&self, omega: f64, temperature: f64) -> Option<[f64; 4]> {
        let a = match &self.trs {
            Some(t) => t.shift(temperature, self.td.absolute_zero(), self.td.gas_constant()).ok()?,
            None => 1.0,
        };
        let w = if a.is_infinite() { 0.0 } else { a * omega };
        Some(match &self.data {
            Freq::Prony { g0, k0, terms } => {
                let (mut gs, mut gl, mut ks, mut kl) = (0.0, 0.0, 0.0, 0.0);
                for &(g, k, tau) in terms {
                    let x = tau * w;
                    let d = 1.0 + x * x;
                    gs += g * x * x / d;
                    gl += g * x / d;
                    ks += k * x * x / d;
                    kl += k * x / d;
                }
                [g0 * gs, k0 * ks, g0 * gl, k0 * kl]
            }
            Freq::Tabular(t) => {
                let f = w / (2.0 * std::f64::consts::PI);
                let r: [f64; 5] = if t.len() == 1 || f <= t[0][4] {
                    t[0]
                } else if let Some(win) = t.windows(2).find(|win| f < win[1][4]) {
                    let s = (f - win[0][4]) / (win[1][4] - win[0][4]);
                    std::array::from_fn(|i| win[0][i] + s * (win[1][i] - win[0][i]))
                } else {
                    t[t.len() - 1]
                };
                [-self.g_inf * r[1], -self.k_inf * r[3], self.g_inf * r[0], self.k_inf * r[2]]
            }
        })
    }
}

/// Isotropt linjärt hjälpmaterial med skjuvmodulen G och kompressionsmodulen
/// K (får vara negativa: tillskott till lagringsmodulerna).
struct IsoGk {
    g: f64,
    k: f64,
}

impl IsoGk {
    fn d6(&self) -> super::plasticity::M6 {
        let mut d = [[0.0; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                d[i][j] = self.k - 2.0 * self.g / 3.0;
            }
            d[i][i] = self.k + 4.0 * self.g / 3.0;
            d[i + 3][i + 3] = self.g;
        }
        d
    }
}

impl MaterialModel for IsoGk {
    fn name(&self) -> &str {
        "VISCOELASTIC (frekvensplanet)"
    }

    fn is_linear(&self) -> bool {
        true
    }

    fn update(&self, input: &MatInput, _state_old: &[f64], _state_new: &mut [f64]) -> Result<MatOutput> {
        let tangent = reduce_6x6(&self.d6(), input.mode);
        Ok(MatOutput { stress: matvec(&tangent, input.strain), tangent })
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(reduce_6x6(&self.d6(), mode))
    }
}

/// Spänningen S för förskjutningen `u` med hjälpmaterialet `mat` i element `e`.
fn stress_with(p: &Prepared, e: usize, mat: &IsoGk, u: &[crate::fem::dof::NodeVec], time: f64) -> Result<Option<IpFieldVals>> {
    let el = &p.model.elements[e];
    let mut c = p.ctx(e, false, time, 0.0);
    c.material = Some(mat);
    let ue = p.gather(e, u);
    let st0 = el.etype.init_state(&c)?;
    let mut st = st0.clone();
    let z = vec![0.0; ue.len()];
    el.etype.internal_force(&c, &ue, &z, &st0, &mut st, false)?;
    let out = el.etype.output(&c, &ue, &st)?;
    Ok(out.into_iter().find(|f| f.name == "S").map(|f| f.points))
}

type IpFieldVals = Vec<(usize, usize, Vec<f64>)>;

/// Lägger till de frekvensberoende delarna av spänningen i en komplex frame
/// från *STEADY STATE DYNAMICS, DIRECT (fälten S och S_IM): med de reella och
/// imaginära förskjutningarna u_r, u_i blir tillskotten
/// S_r += S(ΔGs, ΔKs; u_r) − S(Gℓ, Kℓ; u_i) och S_i += S(ΔGs, ΔKs; u_i) + S(Gℓ, Kℓ; u_r).
pub fn correct_stress(p: &Prepared, fr: &mut crate::fem::output::FrameData, re: &[crate::fem::dof::NodeVec], im: &[crate::fem::dof::NodeVec], omega: f64, time: f64) -> Result<()> {
    let m = p.model;
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, time, 0.0);
        let Some(mat) = ctx.material else { continue };
        let temp = if ctx.temp.is_empty() { 0.0 } else { ctx.temp.iter().take(el.nodes.len()).sum::<f64>() / el.nodes.len() as f64 };
        let Some([dgs, dks, gl, kl]) = mat.frequency_moduli(omega, temp) else { continue };
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        let sa = IsoGk { g: dgs, k: dks };
        let sb = IsoGk { g: gl, k: kl };
        let (Some(a_r), Some(a_i), Some(b_r), Some(b_i)) = (
            stress_with(p, e, &sa, re, time).map_err(wrap)?,
            stress_with(p, e, &sa, im, time).map_err(wrap)?,
            stress_with(p, e, &sb, re, time).map_err(wrap)?,
            stress_with(p, e, &sb, im, time).map_err(wrap)?,
        ) else {
            continue;
        };
        let Some(fields) = fr.elem.get_mut(e) else { continue };
        for f in fields.iter_mut() {
            let (add, sub): (&IpFieldVals, (&IpFieldVals, f64)) = match f.name.as_str() {
                "S" => (&a_r, (&b_i, -1.0)),
                "S_IM" => (&a_i, (&b_r, 1.0)),
                _ => continue,
            };
            for (k, pt) in f.points.iter_mut().enumerate() {
                if let (Some(x), Some(y)) = (add.get(k), sub.0.get(k)) {
                    for (j, v) in pt.2.iter_mut().enumerate() {
                        *v += x.2.get(j).copied().unwrap_or(0.0) + sub.1 * y.2.get(j).copied().unwrap_or(0.0);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Tillskotten till den dynamiska styvhetens reella och imaginära del vid
/// vinkelfrekvensen `omega` från material med viskoelasticitet i
/// frekvensplanet, i `all`-numreringen (fria och inre frihetsgrader, undre
/// triangeln). `None` om modellen saknar sådana material.
pub fn frequency_matrices(p: &Prepared, all: &AllDofs, omega: f64, time: f64) -> Result<Option<(SymCsr, SymCsr)>> {
    let m = p.model;
    let ns = all.n_solve();
    let mut ta = Triplets::new(ns);
    let mut tb = Triplets::new(ns);
    let mut any = false;
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, time, 0.0);
        let Some(mat) = ctx.material else { continue };
        let temp = if ctx.temp.is_empty() { 0.0 } else { ctx.temp.iter().take(el.nodes.len()).sum::<f64>() / el.nodes.len() as f64 };
        let Some([dgs, dks, gl, kl]) = mat.frequency_moduli(omega, temp) else { continue };
        if !matches!(el.etype.family(), Family::Continuum2D | Family::Continuum3D | Family::Axisymmetric) {
            return fail(format!("{}: viskoelasticitet i frekvensplanet stöds bara i kontinuumelement", m.elem_name(e)));
        }
        any = true;
        let wrap = |err: FemError| FemError(format!("{}: {err}", m.elem_name(e)));
        let sa = IsoGk { g: dgs, k: dks };
        let sb = IsoGk { g: gl, k: kl };
        let mut ca = p.ctx(e, false, time, 0.0);
        ca.material = Some(&sa);
        let ka = el.etype.stiffness(&ca).map_err(wrap)?;
        let mut cb = p.ctx(e, false, time, 0.0);
        cb.material = Some(&sb);
        let kb = el.etype.stiffness(&cb).map_err(wrap)?;
        let map: Vec<usize> = p.edofs[e].iter().map(|&(n, s)| all.idx[n][s]).collect();
        let nd = map.len();
        if ka.len() != nd * nd {
            return fail(format!("{}: elementmatrisen har fel storlek för viskoelasticitet i frekvensplanet", m.elem_name(e)));
        }
        for a in 0..nd {
            for c in 0..nd {
                let (ia, ic) = (map[a], map[c]);
                if ia == NONE || ic == NONE || ia >= ns || ic >= ns || ia < ic {
                    continue;
                }
                ta.add(ia, ic, ka[a * nd + c]);
                tb.add(ia, ic, kb[a * nd + c]);
            }
        }
    }
    Ok(any.then(|| (ta.build(), tb.build())))
}
