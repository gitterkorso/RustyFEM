//! Krossbart skum (*CRUSHABLE FOAM, HARDENING=VOLUMETRIC eller ISOTROPIC)
//! med *CRUSHABLE FOAM HARDENING och linjär *ELASTIC, för implicit analys.
//!
//! Flytytan i (p, q) är en ellips (p = −tr σ/3, q = von Mises):
//! - VOLUMETRIC (data k = σc0/pc0, kt = pt/pc0):
//!   F = √(q² + α²(p − p0)²) − αA, p0 = (pc − pt)/2, A = (pc + pt)/2,
//!   α = 3k/√((3kt + k)(3 − k)). Draghållfastheten pt = kt·pc0 är konstant och
//!   pc följer den kompakterande volymetriska plastiska töjningen
//!   εc = −ε_vol^pl genom tabellen σc(εc) (enaxlig tryckflytspänning):
//!   pc = σc[σc(1/α² + 1/9) + pt/3]/(pt + σc/3) i tabellens punkter och
//!   linjärt mellan dem. Flödet är icke-associerat med
//!   G = √(q² + (9/2)p²) (plastiskt tvärkontraktionstal 0); PEEQ = εc.
//! - ISOTROPIC (data k = σc0/pc0, νp): F = √(q² + α²p²) − σc√(1 + α²/9),
//!   α = 3k/√(9 − k²), associerat i ytans form med
//!   G = √(q² + β²p²), β = (3/√2)√((1 − 2νp)/(1 + νp)); hårdnandet följer
//!   ε̄pl med σc dε̄pl = σ:dεpl (PEEQ = ε̄pl).
//!
//! Tabellen interpoleras linjärt och är konstant utanför. Returen är bakåt
//! Euler i (p, q) med Δλ (och ε̄pl för ISOTROPIC) som obekanta
//! (`invariant_return::newton_fd`); deviatorn behåller prövospänningens
//! riktning. Tangenten är numerisk (framåtdifferens av hela uppdateringen)
//! vid plastisk flytning. Plan spänning och stänger via
//! `plasticity::update_reduced`.

use super::plasticity::{dev, iso_d6, mv6, pe_labels, update_reduced, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "krossbart skum",
    consumes: &["ELASTIC", "CRUSHABLE FOAM", "CRUSHABLE FOAM HARDENING"],
    priority: 10,
    build: Some(build),
}];

/// Tillstånd: [3D-töjning (6), plastisk töjning (6), hårdnandevariabel].
const EPS3: usize = 0;
const EP: usize = 6;
const H: usize = 12;
const NSTATE: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Volumetric,
    Isotropic,
}

pub struct CrushableFoam {
    kind: Kind,
    e: f64,
    nu: f64,
    g: f64,
    kb: f64,
    ce: M6,
    alpha: f64,
    /// β² i flödespotentialen G = √(q² + β²p²).
    beta2: f64,
    /// Draghållfastheten pt (VOLUMETRIC).
    pt: f64,
    /// (εc eller ε̄pl, σc), stigande töjning.
    table: Vec<(f64, f64)>,
    /// VOLUMETRIC: (εc, pc) i tabellens punkter.
    pc_table: Vec<(f64, f64)>,
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("CRUSHABLE FOAM") else {
        if let Some(h) = def.behavior("CRUSHABLE FOAM HARDENING") {
            return h.err("*CRUSHABLE FOAM HARDENING kräver *CRUSHABLE FOAM");
        }
        return Ok(None);
    };
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperatur- eller fältberoende *CRUSHABLE FOAM stöds inte ännu");
    }
    let kind = match kw.param_upper("HARDENING").as_deref() {
        None | Some("VOLUMETRIC") => Kind::Volumetric,
        Some("ISOTROPIC") => Kind::Isotropic,
        Some(o) => return kw.err(format!("HARDENING={o} stöds inte (VOLUMETRIC, ISOTROPIC)")),
    };
    let Some(ek) = def.behavior("ELASTIC") else {
        return kw.err("*CRUSHABLE FOAM kräver *ELASTIC");
    };
    if ek.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || ek.data.len() > 1 || ek.has("DEPENDENCIES") {
        return ek.err("*CRUSHABLE FOAM kräver isotrop, temperaturoberoende *ELASTIC");
    }
    let ev = ek.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0) || !(nu > -1.0 && nu < 0.5) {
        return ek.err("ogiltiga elasticitetsdata");
    }
    let Some(hk) = def.behavior("CRUSHABLE FOAM HARDENING") else {
        return kw.err("*CRUSHABLE FOAM kräver *CRUSHABLE FOAM HARDENING");
    };
    if hk.has("DEPENDENCIES") {
        return hk.err("fältberoende hårdnande stöds inte ännu");
    }
    let mut table: Vec<(f64, f64)> = Vec::new();
    for d in &hk.data {
        let v = hk.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return hk.err_at(d.line, "hastighets- eller temperaturberoende hårdnande stöds inte ännu");
        }
        let (s, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(s > 0.0) {
            return hk.err_at(d.line, "flytspänningen ska vara positiv");
        }
        if table.last().is_some_and(|&(pl, _)| !(p > pl)) {
            return hk.err_at(d.line, "den plastiska töjningen ska öka i tabellen");
        }
        table.push((p, s));
    }
    if table.is_empty() {
        return hk.err("data saknas");
    }
    let v = kw.first_floats()?;
    let k = v.first().copied().unwrap_or(0.0);
    let (alpha, beta2, pt) = match kind {
        Kind::Volumetric => {
            let kt = v.get(1).copied().unwrap_or(0.0);
            if !(k > 0.0 && k < 3.0) || !(kt >= 0.0) {
                return kw.err("HARDENING=VOLUMETRIC kräver 0 < k < 3 och kt ≥ 0");
            }
            let pc0 = sigma_c(&table, 0.0) / k;
            (3.0 * k / ((3.0 * kt + k) * (3.0 - k)).sqrt(), 4.5, kt * pc0)
        }
        Kind::Isotropic => {
            let nup = v.get(1).copied().unwrap_or(0.0);
            if !(k > 0.0 && k < 3.0) || !(nup > -1.0 && nup <= 0.5) {
                return kw.err("HARDENING=ISOTROPIC kräver 0 < k < 3 och −1 < νp ≤ 0,5");
            }
            (3.0 * k / (9.0 - k * k).sqrt(), 4.5 * (1.0 - 2.0 * nup) / (1.0 + nup), 0.0)
        }
    };
    let g = e / (2.0 * (1.0 + nu));
    let kb = e / (3.0 * (1.0 - 2.0 * nu));
    let a2 = alpha * alpha;
    let pc_table = table.iter().map(|&(x, sc)| (x, sc * (sc * (1.0 / a2 + 1.0 / 9.0) + pt / 3.0) / (pt + sc / 3.0))).collect();
    Ok(Some(Arc::new(CrushableFoam { kind, e, nu, g, kb, ce: iso_d6(e, nu), alpha, beta2, pt, table, pc_table })))
}

fn sigma_c(t: &[(f64, f64)], x: f64) -> f64 {
    super::principal::interp(t, x).0
}

fn pq(s: &V6) -> (f64, f64) {
    let p = -(s[0] + s[1] + s[2]) / 3.0;
    (p, super::plasticity::mises(&dev(s)))
}

impl CrushableFoam {
    /// Flytfunktionen F(p, q, h).
    fn yield_fn(&self, p: f64, q: f64, h: f64) -> f64 {
        let sc = sigma_c(&self.table, h);
        match self.kind {
            Kind::Volumetric => {
                let a2 = self.alpha * self.alpha;
                let pt = self.pt;
                // pc i tabellens punkter, linjärt mellan dem (dpc/dεc är
                // konstant mellan punkterna)
                let pc = sigma_c(&self.pc_table, h);
                let (p0, a) = (0.5 * (pc - pt), 0.5 * (pc + pt));
                (q * q + a2 * (p - p0) * (p - p0)).sqrt() - self.alpha * a
            }
            Kind::Isotropic => (q * q + self.alpha * self.alpha * p * p).sqrt() - sc * (1.0 + self.alpha * self.alpha / 9.0).sqrt(),
        }
    }

    /// Skala för spänningarna (flytspänningen i början).
    fn scale(&self) -> f64 {
        self.table[0].1
    }

    /// Spänningsuppdatering i 3D (töjning med ingenjörsskjuvning). Returnerar
    /// spänningen och (plastisk töjning, hårdnande) vid inkrementets slut.
    fn stress3d(&self, eps: &V6, ep_old: &V6, h_old: f64) -> Result<(V6, V6, f64, bool)> {
        let ee: V6 = std::array::from_fn(|i| eps[i] - ep_old[i]);
        let str_ = mv6(&self.ce, &ee);
        let (ptr, qtr) = pq(&str_);
        if self.yield_fn(ptr, qtr, h_old) <= 1e-12 * self.scale() {
            return Ok((str_, *ep_old, h_old, false));
        }
        let ss = self.scale();
        let (g, kb, b2) = (self.g, self.kb, self.beta2);
        let iso = self.kind == Kind::Isotropic;
        // obekanta: p/ss, q/ss, Δλ·E/ss och (ISOTROPIC) ε̄pl
        let lam_s = ss / self.e;
        let resid = |x: &[f64]| -> Vec<f64> {
            let (p, q, dl) = (x[0] * ss, x[1].max(0.0) * ss, x[2] * lam_s);
            let gp = (q * q + b2 * p * p).sqrt().max(1e-300);
            let (gpp, gq) = (b2 * p / gp, q / gp);
            let h = if iso { x[3] } else { h_old + dl * gpp };
            let mut r = vec![(p - (ptr - kb * dl * gpp)) / ss, (q - (qtr - 3.0 * g * dl * gq)) / ss, self.yield_fn(p, q, h) / ss];
            if iso {
                r.push((sigma_c(&self.table, h) * (h - h_old) - dl * gp) / ss);
            }
            r
        };
        let mut x = vec![ptr / ss, qtr / ss, 0.0];
        let mut typ = vec![1.0, 1.0, 1.0];
        if iso {
            x.push(h_old);
            typ.push(1e-2);
        }
        let res = super::invariant_return::newton_fd(&mut x, &typ, &resid, 1e-13);
        if !(res <= 1e-9) || x[2] < -1e-12 {
            return fail(format!("*CRUSHABLE FOAM: returalgoritmen konvergerade inte (residual {res:e}, Δλ = {:e})", x[2] * lam_s));
        }
        let (p, q, dl) = (x[0] * ss, x[1].max(0.0) * ss, x[2] * lam_s);
        let gp = (q * q + b2 * p * p).sqrt().max(1e-300);
        let (gpp, gq) = (b2 * p / gp, q / gp);
        let h = if iso { x[3] } else { h_old + dl * gpp };
        let st = dev(&str_);
        let ratio = if qtr > 1e-300 { q / qtr } else { 0.0 };
        let mut sig = [0.0; 6];
        for i in 0..6 {
            sig[i] = st[i] * ratio - if i < 3 { p } else { 0.0 };
        }
        // Δεpl = Δλ (−Gp/3 I + (3/2) Gq s/q), skjuvning som ingenjörstöjning
        let mut ep = *ep_old;
        for i in 0..6 {
            let n = if qtr > 1e-300 { 1.5 * gq * st[i] / qtr } else { 0.0 };
            let v = dl * (n - if i < 3 { gpp / 3.0 } else { 0.0 });
            ep[i] += if i < 3 { v } else { 2.0 * v };
        }
        Ok((sig, ep, h, true))
    }
}

impl MaterialModel for CrushableFoam {
    fn name(&self) -> &str {
        "CRUSHABLE FOAM"
    }

    fn n_state(&self) -> usize {
        NSTATE
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let ep_old: V6 = std::array::from_fn(|i| state_old[EP + i]);
        let h_old = state_old[H];
        let mut last = (ep_old, h_old);
        let mut f = |e: &V6| -> Result<(V6, M6)> {
            let (sig, ep, h, plastic) = self.stress3d(e, &ep_old, h_old)?;
            last = (ep, h);
            if !plastic {
                return Ok((sig, self.ce));
            }
            let dh = 1e-8;
            let c = super::principal::fd_tangent(e, &sig, dh, |e2| Ok(self.stress3d(e2, &ep_old, h_old)?.0))?;
            Ok((sig, c))
        };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        state_new[EP..EP + 6].copy_from_slice(&last.0);
        state_new[H] = last.1;
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
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[H]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[H]];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        v
    }
}
