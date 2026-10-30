//! Porösa metaller (*POROUS METAL PLASTICITY): Gurson–Tvergaard–Needleman
//! ovanpå *ELASTIC och *PLASTIC (matrisens isotropa hårdnande), med
//! *VOID NUCLEATION och *POROUS FAILURE CRITERIA, för implicit analys.
//!
//! Flytfunktion (p = −tr σ/3, q = von Mises, σy = matrisens flytspänning):
//!
//!   Φ = (q/σy)² + 2 q1 f* cosh(−3 q2 p/(2σy)) − (1 + q3 f*²) = 0,
//!
//! associerat flöde Δεpl = Δλ ∂Φ/∂σ. Matrisens plastiska töjning ges av
//! arbetsekvivalensen (1 − f) σy Δε̄m = σ:Δεpl. Porositeten växer genom
//! tillväxt Δf_gr = (1 − f) tr Δεpl och kärnbildning Δf_nucl = A Δε̄m med
//! A = fN/(sN√(2π)) exp(−½((ε̄m − εN)/sN)²), bara i drag (p < 0; ingen
//! kärnbildning i tryck). f* = f, eller med
//! *POROUS FAILURE CRITERIA (fF, fc) Tvergaard–Needlemans
//! f* = fc + (f̄F − fc)(f − fc)/(fF − fc) för f > fc, f̄F = (q1 + √(q1² − q3))/q3.
//!
//! Begynnelseporositeten f0 = 1 − r0 ges av RELATIVE DENSITY= (standard 1)
//! eller, om parametern saknas, av *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY
//! (nodvärden, elementets nodmedelvärde; parametern går före).
//! Returen är bakåt Euler med obekanta (p, q, Δλ, ε̄m, f)
//! (`invariant_return::newton_fd`) och deviatorn i prövospänningens riktning;
//! tangenten är numerisk. Utdata: PEEQ = ε̄m, PE, VVF = f, VVFG = ∫df_gr,
//! VVFN = ∫df_nucl (VVF = f0 + VVFG + VVFN). Tillväxten redovisas per
//! inkrement som (1 − fn) tr Δεpl/(1 + tr Δεpl) och kärnbildningen som resten
//! (den totala porositeten följer ekvationen ovan).

use super::plasticity::{dev, iso_d6, mv6, pe_labels, update_reduced, Hardening, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::assembly::Prepared;
use crate::fem::model::{parse_f64, MaterialDef};
use crate::fem::state::SolutionState;
use crate::model::{fail, FemError, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "porös metallplasticitet (Gurson–Tvergaard–Needleman)",
    consumes: &["ELASTIC", "PLASTIC", "POROUS METAL PLASTICITY", "VOID NUCLEATION", "POROUS FAILURE CRITERIA"],
    priority: 30,
    build: Some(build),
}];

/// Tillstånd: [3D-töjning (6), PE (6), ε̄m, f, VVFG, VVFN].
const EPS3: usize = 0;
const EP: usize = 6;
const PM: usize = 12;
const F: usize = 13;
const FG: usize = 14;
const FN: usize = 15;
const NSTATE: usize = 16;

pub struct Gtn {
    e: f64,
    nu: f64,
    g: f64,
    kb: f64,
    ce: M6,
    hard: Hardening,
    q: [f64; 3],
    /// (εN, sN, fN)
    nucl: Option<[f64; 3]>,
    /// (fF, fc)
    failure: Option<[f64; 2]>,
    f0: f64,
    /// RELATIVE DENSITY= var given (går före *INITIAL CONDITIONS).
    f0_given: bool,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("POROUS METAL PLASTICITY") else {
        for b in ["VOID NUCLEATION", "POROUS FAILURE CRITERIA"] {
            if let Some(k) = def.behavior(b) {
                return k.err(format!("*{b} kräver *POROUS METAL PLASTICITY"));
            }
        }
        return Ok(None);
    };
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperatur- eller fältberoende *POROUS METAL PLASTICITY stöds inte ännu");
    }
    let v = kw.first_floats()?;
    let q1 = v.first().copied().unwrap_or(1.0);
    let q2 = v.get(1).copied().unwrap_or(1.0);
    let q3 = v.get(2).copied().unwrap_or(q1 * q1);
    let (f0, f0_given) = match kw.param("RELATIVE DENSITY") {
        Some(s) => {
            let r = parse_f64(s).ok_or_else(|| FemError(format!("{}: ogiltig RELATIVE DENSITY={s}", kw.location())))?;
            if !(r > 0.0 && r <= 1.0) {
                return kw.err("RELATIVE DENSITY ska ligga i (0, 1]");
            }
            (1.0 - r, true)
        }
        None => (0.0, false),
    };
    let Some(pk) = def.behavior("PLASTIC") else {
        return kw.err("*POROUS METAL PLASTICITY kräver *PLASTIC");
    };
    if pk.param_upper("HARDENING").is_some_and(|h| h != "ISOTROPIC") || pk.data.iter().any(|d| pk.floats(d).map_or(true, |v| v.len() > 2 && v[2..].iter().any(|&x| x != 0.0))) {
        return pk.err("*POROUS METAL PLASTICITY kräver isotropt, temperaturoberoende *PLASTIC");
    }
    let Some(ek) = def.behavior("ELASTIC") else {
        return kw.err("*POROUS METAL PLASTICITY kräver *ELASTIC");
    };
    if ek.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || ek.data.len() > 1 {
        return ek.err("*POROUS METAL PLASTICITY kräver isotrop, temperaturoberoende *ELASTIC");
    }
    let ev = ek.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) {
        return ek.err("ogiltiga elasticitetsdata");
    }
    let mut table = Vec::new();
    for d in &pk.data {
        let v = pk.floats(d)?;
        let (s, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(s > 0.0) || table.last().is_some_and(|&(pl, _)| !(p > pl)) {
            return pk.err_at(d.line, "flytspänningen ska vara positiv och den plastiska töjningen öka");
        }
        table.push((p, s));
    }
    if table.is_empty() {
        return pk.err("flytspänning saknas");
    }
    let nucl = match def.behavior("VOID NUCLEATION") {
        Some(k) => {
            if k.data.len() > 1 {
                return k.err("temperaturberoende kärnbildning stöds inte ännu");
            }
            let v = k.first_floats()?;
            let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
            if !(g(1) > 0.0) || g(2) < 0.0 {
                return k.err("*VOID NUCLEATION kräver εN, sN > 0 och fN ≥ 0");
            }
            Some([g(0), g(1), g(2)])
        }
        None => None,
    };
    let failure = match def.behavior("POROUS FAILURE CRITERIA") {
        Some(k) => {
            let v = k.first_floats()?;
            let (ff, fc) = (v.first().copied().unwrap_or(1.0), v.get(1).copied().unwrap_or(1.0));
            if !(fc > 0.0 && ff > fc) {
                return k.err("*POROUS FAILURE CRITERIA kräver 0 < fc < fF");
            }
            Some([ff, fc])
        }
        None => None,
    };
    let g = e / (2.0 * (1.0 + nu));
    let kb = e / (3.0 * (1.0 - 2.0 * nu));
    Ok(Some(Arc::new(Gtn { e, nu, g, kb, ce: iso_d6(e, nu), hard: Hardening::Table(table), q: [q1, q2, q3], nucl, failure, f0, f0_given })))
}

fn pq(s: &V6) -> (f64, f64) {
    (-(s[0] + s[1] + s[2]) / 3.0, super::plasticity::mises(&dev(s)))
}

impl Gtn {
    /// f* och df*/df.
    fn fstar(&self, f: f64) -> (f64, f64) {
        match self.failure {
            Some([ff, fc]) if f > fc => {
                let [q1, _, q3] = self.q;
                let fbar = (q1 + (q1 * q1 - q3).max(0.0).sqrt()) / q3;
                let k = (fbar - fc) / (ff - fc);
                (fc + k * (f - fc), k)
            }
            _ => (f, 1.0),
        }
    }

    /// Φ, ∂Φ/∂p, ∂Φ/∂q.
    fn phi(&self, p: f64, q: f64, sy: f64, f: f64) -> (f64, f64, f64) {
        let [q1, q2, q3] = self.q;
        let fs = self.fstar(f).0;
        let x = -1.5 * q2 * p / sy;
        let ph = (q / sy).powi(2) + 2.0 * q1 * fs * x.cosh() - (1.0 + q3 * fs * fs);
        (ph, 2.0 * q1 * fs * x.sinh() * (-1.5 * q2 / sy), 2.0 * q / (sy * sy))
    }

    /// Kärnbildningens A(ε̄m) vid trycket p.
    fn nucleation(&self, pm: f64, p: f64) -> f64 {
        match self.nucl {
            Some([en, sn, fnn]) if p < 0.0 => fnn / (sn * (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * ((pm - en) / sn).powi(2)).exp(),
            _ => 0.0,
        }
    }

    /// 3D-uppdatering: (σ, εpl, ε̄m, f, Δf_gr, Δf_nucl, plastisk).
    fn stress3d(&self, eps: &V6, ep_old: &V6, pm_n: f64, f_n: f64) -> Result<(V6, V6, f64, f64, f64, f64, bool)> {
        let ee: V6 = std::array::from_fn(|i| eps[i] - ep_old[i]);
        let str_ = mv6(&self.ce, &ee);
        let (ptr, qtr) = pq(&str_);
        let sy_n = self.hard.eval(pm_n).0;
        if self.phi(ptr, qtr, sy_n, f_n).0 <= 1e-12 {
            return Ok((str_, *ep_old, pm_n, f_n, 0.0, 0.0, false));
        }
        let ss = self.hard.initial();
        let (g, kb) = (self.g, self.kb);
        // obekanta: p/ss, q/ss, Δλ/ss, ε̄m − ε̄m,n, f
        let parts = |x: &[f64]| {
            let (p, q, dl, pm, f) = (x[0] * ss, x[1].max(0.0) * ss, x[2] * ss, pm_n + x[3], x[4]);
            let sy = self.hard.eval(pm).0;
            let (ph, php, phq) = self.phi(p, q, sy, f);
            let dfg = (1.0 - f) * (-dl * php);
            let dfn = self.nucleation(pm, p) * x[3];
            (p, q, dl, pm, f, sy, ph, php, phq, dfg, dfn)
        };
        let resid = |x: &[f64]| -> Vec<f64> {
            let (p, q, dl, _pm, f, sy, ph, php, phq, dfg, dfn) = parts(x);
            vec![
                (p - (ptr - kb * dl * php)) / ss,
                (q - (qtr - 3.0 * g * dl * phq)) / ss,
                ph,
                ((1.0 - f) * sy * x[3] - dl * (p * php + q * phq)) / ss,
                f - f_n - dfg - dfn,
            ]
        };
        let mut x = vec![ptr / ss, qtr / ss, 0.0, 0.0, f_n];
        let typ = [1.0, 1.0, 1e-4, 1e-3, 1e-3];
        let res = super::invariant_return::newton_fd(&mut x, &typ, &resid, 1e-14);
        if !(res <= 1e-10) || x[2] < -1e-14 || x[3] < -1e-12 {
            return fail(format!("*POROUS METAL PLASTICITY: returalgoritmen konvergerade inte (residual {res:e})"));
        }
        let (p, q, dl, pm, f, _sy, _ph, php, phq, _dfg, _dfn) = parts(&x);
        // redovisning: tillväxten med porositeten utan
        // inkrementets kärnbildning, Δf_gr = (1 − fn − Δf_gr) tr Δεpl, och
        // kärnbildningen som resten av Δf
        let tr = -dl * php;
        let dfg = (1.0 - f_n) * tr / (1.0 + tr);
        let dfn = f - f_n - dfg;
        let st = dev(&str_);
        let ratio = if qtr > 1e-300 { q / qtr } else { 0.0 };
        let mut sig = [0.0; 6];
        let mut ep = *ep_old;
        for i in 0..6 {
            sig[i] = st[i] * ratio - if i < 3 { p } else { 0.0 };
            // ∂Φ/∂σ = −Φp/3 I + (3/2) Φq s/q
            let n = if qtr > 1e-300 { 1.5 * phq * st[i] / qtr } else { 0.0 };
            let v = dl * (n - if i < 3 { php / 3.0 } else { 0.0 });
            ep[i] += if i < 3 { v } else { 2.0 * v };
        }
        Ok((sig, ep, pm, f, dfg, dfn, true))
    }
}

impl MaterialModel for Gtn {
    fn name(&self) -> &str {
        "POROUS METAL PLASTICITY"
    }

    fn n_state(&self) -> usize {
        NSTATE
    }

    fn init_state(&self, state: &mut [f64]) {
        state[F] = self.f0;
    }

    fn init_relative_density(&self, r: f64, state: &mut [f64]) -> bool {
        if !self.f0_given {
            state[F] = 1.0 - r;
        }
        true
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let ep_old: V6 = std::array::from_fn(|i| state_old[EP + i]);
        let (pm_n, f_n) = (state_old[PM], state_old[F]);
        let mut last = (ep_old, pm_n, f_n, 0.0, 0.0);
        let mut f = |e: &V6| -> Result<(V6, M6)> {
            let (sig, ep, pm, ff, dfg, dfn, plastic) = self.stress3d(e, &ep_old, pm_n, f_n)?;
            last = (ep, pm, ff, dfg, dfn);
            if !plastic {
                return Ok((sig, self.ce));
            }
            let c = super::principal::fd_tangent(e, &sig, 1e-8, |e2| Ok(self.stress3d(e2, &ep_old, pm_n, f_n)?.0))?;
            Ok((sig, c))
        };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        state_new[EP..EP + 6].copy_from_slice(&last.0);
        state_new[PM] = last.1;
        state_new[F] = last.2;
        state_new[FG] = state_old[FG] + last.3;
        state_new[FN] = state_old[FN] + last.4;
        Ok(out)
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        super::plasticity::rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
        super::plasticity::rotate_v6(r, &mut state[EP..EP + 6], true);
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ", "VVF", "VVFG", "VVFN"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[PM], state[F], state[FG], state[FN]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0), ("VVF", vec!["VVF"]), ("VVFG", vec!["VVFG"]), ("VVFN", vec!["VVFN"])]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[PM]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        v.extend([state[F], state[FG], state[FN]]);
        v
    }
}

/// *INITIAL CONDITIONS, TYPE=RELATIVE DENSITY (`inp/keywords/materials5.rs`):
/// elementets nodmedelvärde av r i alla integrationspunkter
/// (`MaterialModel::init_relative_density`). Anropas av `steps::run` efter
/// `Prepared::init_states`.
pub fn apply_relative_density(prep: &Prepared, state: &mut SolutionState) -> Result<()> {
    let model = prep.model;
    let key = crate::inp::keywords::materials5::RESOLVED_RD;
    if !model.extra.iter().any(|k| k.name == "INITIAL CONDITIONS" && k.has(key)) {
        return Ok(());
    }
    let mut rn = vec![f64::NAN; model.nodes.len()];
    for kw in model.extra.iter().filter(|k| k.name == "INITIAL CONDITIONS" && k.has(key)) {
        for d in &kw.data {
            let v = kw.floats(d)?;
            rn[v[0] as usize] = v[1];
        }
    }
    for e in 0..model.elements.len() {
        let el = &model.elements[e];
        if el.nodes.iter().all(|&n| rn[n].is_nan()) {
            continue;
        }
        let ctx = prep.ctx(e, false, 0.0, 0.0);
        let Some(mat) = ctx.material else { continue };
        if el.nodes.iter().any(|&n| rn[n].is_nan()) {
            return fail(format!("{}: den relativa densiteten saknas i några av elementets noder", model.elem_name(e)));
        }
        let r = el.nodes.iter().map(|&n| rn[n]).sum::<f64>() / el.nodes.len() as f64;
        let nip = el.etype.n_ips();
        let ns = mat.n_state();
        let st = &mut state.elem_state[e];
        let Some(nc) = [6usize, 4, 3, 1].into_iter().find(|&nc| nip > 0 && st.len() == nip * (nc + ns)) else { continue };
        for ip in 0..nip {
            let b = ip * (nc + ns) + nc;
            if !mat.init_relative_density(r, &mut st[b..b + ns]) {
                break;
            }
        }
    }
    Ok(())
}
