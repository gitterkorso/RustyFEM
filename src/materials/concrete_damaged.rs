//! Betong med skadeplasticitet (*CONCRETE DAMAGED PLASTICITY) enligt
//! Lee–Fenves, med *CONCRETE COMPRESSION HARDENING,
//! *CONCRETE TENSION STIFFENING (TYPE=STRAIN, DISPLACEMENT eller GFI: linjär
//! mjukning från σt0 till σt0/100 vid u0 = 2Gf/σt0),
//! *CONCRETE COMPRESSION DAMAGE och *CONCRETE TENSION DAMAGE.
//!
//! Spänning σ = (1 − d) D₀(ε − εp), effektiv spänning σ̄ = D₀(ε − εp).
//! Flytvillkor i effektiv spänning (p̄, q̄, σ̂max = största huvudspänning):
//!
//!   F = [q̄ − 3α p̄ + β ⟨σ̂max⟩ − γ ⟨−σ̂max⟩]/(1 − α) − σ̄c(ε̃c),
//!   α = (σb0/σc0 − 1)/(2σb0/σc0 − 1),  γ = 3(1 − Kc)/(2Kc − 1),
//!   β = σ̄c(ε̃c)/σ̄t(ε̃t) · (1 − α) − (1 + α),
//!
//! med effektiva kohesionsspänningar σ̄c = σc/(1 − dc), σ̄t = σt/(1 − dt).
//! Flödespotentialen är hyperbolisk Drucker–Prager
//! G = √((ε σt0 tan ψ)² + q̄²) − p̄ tan ψ. Hårdnandevariablerna växer med
//! de plastiska huvudtöjningshastigheterna:
//! ε̃̇t = r(σ̄) ε̂̇p_max och ε̃̇c = −(1 − r(σ̄)) ε̂̇p_min med
//! r(σ̄) = Σ⟨σ̂ᵢ⟩/Σ|σ̂ᵢ|. Styvhetsdegraderingen är
//! 1 − d = (1 − s_t dc)(1 − s_c dt), s_t = 1 − w_t r, s_c = 1 − w_c (1 − r)
//! (TENSION RECOVERY = w_t, standard 0; COMPRESSION RECOVERY = w_c, standard 1).
//!
//! Tabellerna anges mot oelastisk töjning ε̃in (tryck),
//! sprickbildningstöjning ε̃ck eller sprickvidd u_ck (drag) och räknas om till
//! plastiska värden ε̃p = ε̃in − d/(1 − d) · σ/E₀ (för sprickvidd med l₀ = 1).
//! Skadan interpoleras i hårdnandetabellens punkter; spänning och skada
//! interpoleras sedan linjärt i den plastiska variabeln. För
//! TYPE=DISPLACEMENT och GFI är dragvariabeln sprickvidden u = ε̃t · L med
//! elementets karakteristiska längd L (Hillerborg, `update_with_length`).
//!
//! Viskös regularisering (μ > 0): ε̇p_v = (εp − εp_v)/μ, ḋ_v = (d − d_v)/μ,
//! diskretiserat implicit, och spänningen σ = (1 − d_v) D₀ (ε − εp_v).
//!
//! Returalgoritmen är implicit Euler i huvudspänningsrummet: för givet Δλ
//! ger flödesregeln q̄ (skalär ekvation), p̄, de plastiska huvudtöjningarna och
//! hårdnandevariablerna explicit, så F(Δλ) = 0 är en skalär ekvation som löses
//! med innesluten sekantmetod. Tangenten är numerisk (centraldifferens av hela
//! uppdateringen inklusive skada och viskositet, se `fd_tangent_central`) och
//! osymmetrisk; den osymmetriska lösaren väljs automatiskt.
//!
//! Utdata: PEEQ (ε̃c), PEEQT (ε̃t), PE (plastisk töjning, regulariserad vid
//! μ > 0), DAMAGEC och DAMAGET (dc, dt utan regularisering) och SDEG (d,
//! regulariserad).

use super::mohr_coulomb::iso_elastic;
use super::plasticity::{iso_d6, mv6, update_reduced, M6, V6};
use super::principal::{fd_tangent_central, from_principal, interp, principal, root_from_zero};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "betong med skadeplasticitet",
    consumes: &[
        "ELASTIC",
        "CONCRETE DAMAGED PLASTICITY",
        "CONCRETE COMPRESSION HARDENING",
        "CONCRETE TENSION STIFFENING",
        "CONCRETE COMPRESSION DAMAGE",
        "CONCRETE TENSION DAMAGE",
    ],
    priority: 10,
    build: Some(build),
}];

/// Uniaxiell kurva i den plastiska variabeln: (x_p, σ, d).
#[derive(Debug, Clone)]
pub struct Curve(pub Vec<(f64, f64, f64)>);

impl Curve {
    /// (σ, d) vid x (linjär interpolation, konstant utanför tabellen).
    pub fn eval(&self, x: f64) -> (f64, f64) {
        let t = &self.0;
        if t.len() == 1 || x <= t[0].0 {
            return (t[0].1, t[0].2);
        }
        for w in t.windows(2) {
            let ((x0, s0, d0), (x1, s1, d1)) = (w[0], w[1]);
            if x < x1 {
                let f = (x - x0) / (x1 - x0);
                return (s0 + f * (s1 - s0), d0 + f * (d1 - d0));
            }
        }
        let l = t[t.len() - 1];
        (l.1, l.2)
    }
}

#[derive(Debug, Clone)]
pub struct Concrete {
    pub e: f64,
    pub nu: f64,
    pub g: f64,
    pub k: f64,
    pub ce: M6,
    pub tan_psi: f64,
    /// ε σt0 tan ψ
    pub a: f64,
    pub alpha: f64,
    pub gamma: f64,
    /// Viskositet μ.
    pub mu: f64,
    pub comp: Curve,
    pub tens: Curve,
    /// Dragkurvan är given mot sprickvidd (TYPE=DISPLACEMENT/GFI).
    pub tens_disp: bool,
    pub wt: f64,
    pub wc: f64,
}

/// Tillståndslayout.
const EC: usize = 0;
const ET: usize = 1;
const EP: usize = 2;
const EVP: usize = 8;
const DCV: usize = 14; // dc
const DTV: usize = 15; // dt
const DV: usize = 16;
const EPS3: usize = 17;
const NST: usize = 23;

/// Läser (värde, x)-rader; tomma fält efter de två första ignoreras om de är 0.
fn pairs(kw: &RawKeyword) -> Result<Vec<(f64, f64, usize)>> {
    let mut t = Vec::new();
    for d in &kw.data {
        let v = kw.floats(d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return kw.err_at(d.line, "hastighets-, temperatur- eller fältberoende data stöds inte ännu");
        }
        t.push((v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), d.line));
    }
    if t.is_empty() {
        return kw.err("data saknas");
    }
    Ok(t)
}

/// Bygger kurvan (x_p, σ, d) från hårdnandetabellen (σ, x_in) och en valfri
/// skadetabell (d, x_in): x_p = x_in − d/(1 − d) σ/E₀.
fn curve(hk: &RawKeyword, hard: &[(f64, f64, usize)], dam: Option<(&RawKeyword, Vec<(f64, f64, usize)>)>, e0: f64, what: &str) -> Result<Curve> {
    let mut dtab: Vec<(f64, f64)> = Vec::new();
    if let Some((dk, rows)) = &dam {
        for &(d, x, line) in rows {
            if !(0.0..1.0).contains(&d) {
                return dk.err_at(line, "skadan ska ligga i [0, 1)");
            }
            if dtab.last().is_some_and(|&(xl, _)| !(x > xl)) {
                return dk.err_at(line, format!("{what} ska öka i skadetabellen"));
            }
            dtab.push((x, d));
        }
    }
    let mut out: Vec<(f64, f64, f64)> = Vec::new();
    for (i, &(s, x, line)) in hard.iter().enumerate() {
        if s < 0.0 {
            return hk.err_at(line, "spänningen ska vara icke-negativ");
        }
        if i == 0 && x != 0.0 {
            return hk.err_at(line, format!("första punkten ska ha {what} 0"));
        }
        if i > 0 && !(x > hard[i - 1].1) {
            return hk.err_at(line, format!("{what} ska öka"));
        }
        let d = if dtab.is_empty() { 0.0 } else { interp(&dtab, x).0 };
        let xp = x - d / (1.0 - d) * s / e0;
        if out.last().is_some_and(|&(xl, _, _)| xp < xl) || xp < 0.0 {
            return hk.err_at(line, format!("negativ eller avtagande plastisk {what} (ε̃p = ε̃in − d/(1−d)·σ/E₀) – kontrollera skadetabellen"));
        }
        out.push((xp, s, d));
    }
    if out[0].1 <= 0.0 {
        return hk.err("första flytspänningen ska vara positiv");
    }
    Ok(Curve(out))
}

fn field(kw: &RawKeyword, i: usize, default: f64) -> Result<f64> {
    let Some(d) = kw.data.first() else { return Ok(default) };
    match d.fields.get(i).map(|s| s.trim()) {
        None | Some("") => Ok(default),
        Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: '{s}' är inget tal", kw.location()))),
    }
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("CONCRETE DAMAGED PLASTICITY") else { return Ok(None) };
    let (e, nu) = iso_elastic(def, kw, "betong med skadeplasticitet")?;
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperaturberoende parametrar stöds inte ännu");
    }
    let psi = field(kw, 0, 0.0)?;
    let ecc = field(kw, 1, 0.1)?;
    let fb = field(kw, 2, 1.16)?;
    let kc = field(kw, 3, 2.0 / 3.0)?;
    let mu = field(kw, 4, 0.0)?;
    if !(0.0..90.0).contains(&psi) || ecc < 0.0 || !(fb > 1.0) || !(kc > 0.5 && kc <= 1.0) || mu < 0.0 {
        return kw.err("ogiltiga parametrar (0 ≤ ψ < 90, ε ≥ 0, σb0/σc0 > 1, 0,5 < Kc ≤ 1, μ ≥ 0)");
    }
    let Some(ck) = def.behavior("CONCRETE COMPRESSION HARDENING") else {
        return kw.err("*CONCRETE DAMAGED PLASTICITY kräver *CONCRETE COMPRESSION HARDENING");
    };
    let Some(tk) = def.behavior("CONCRETE TENSION STIFFENING") else {
        return kw.err("*CONCRETE DAMAGED PLASTICITY kräver *CONCRETE TENSION STIFFENING");
    };
    for k in [ck, tk] {
        if k.has("RATE") || k.has("DEPENDENCIES") {
            return k.err("hastighets- eller temperaturberoende data stöds inte ännu");
        }
    }
    let cdk = def.behavior("CONCRETE COMPRESSION DAMAGE");
    let tdk = def.behavior("CONCRETE TENSION DAMAGE");
    let wt = match cdk {
        Some(k) => match k.param("TENSION RECOVERY") {
            Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: ogiltig TENSION RECOVERY", k.location())))?,
            None => 0.0,
        },
        None => 0.0,
    };
    let wc = match tdk {
        Some(k) => match k.param("COMPRESSION RECOVERY") {
            Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: ogiltig COMPRESSION RECOVERY", k.location())))?,
            None => 1.0,
        },
        None => 1.0,
    };
    if !(0.0..=1.0).contains(&wt) || !(0.0..=1.0).contains(&wc) {
        return kw.err("TENSION RECOVERY och COMPRESSION RECOVERY ska ligga i [0, 1]");
    }
    let cdam = match cdk {
        Some(k) => Some((k, pairs(k)?)),
        None => None,
    };
    let comp = curve(ck, &pairs(ck)?, cdam, e, "oelastisk töjning")?;
    let ttype = tk.param_upper("TYPE").unwrap_or_else(|| "STRAIN".into());
    let tdam = match tdk {
        Some(k) => {
            let dt = k.param_upper("TYPE").unwrap_or_else(|| "STRAIN".into());
            let want = if ttype == "STRAIN" { "STRAIN" } else { "DISPLACEMENT" };
            if dt != want {
                return k.err(format!("TYPE={dt} passar inte *CONCRETE TENSION STIFFENING, TYPE={ttype} (ska vara {want})"));
            }
            Some((k, pairs(k)?))
        }
        None => None,
    };
    let (tens, tens_disp) = match ttype.as_str() {
        "STRAIN" => (curve(tk, &pairs(tk)?, tdam, e, "sprickbildningstöjning")?, false),
        "DISPLACEMENT" => (curve(tk, &pairs(tk)?, tdam, e, "sprickvidd")?, true),
        "GFI" => {
            // linjär mjukning från σt0 vid u = 0 till σt0/100 vid u0 = 2 Gf/σt0
            // (restspänning; lutning (σt0 − σt0/100)/u0)
            let v = pairs(tk)?;
            let (st0, gf, line) = v[0];
            if v.len() > 1 || !(st0 > 0.0 && gf > 0.0) {
                return tk.err_at(line, "TYPE=GFI kräver en rad med σt0 > 0 och Gf > 0");
            }
            let u0 = 2.0 * gf / st0;
            let sres = 0.01 * st0;
            let mut pts = vec![(st0, 0.0, line), (sres, u0, line)];
            // skadetabellens punkter läggs in i den linjära mjukningskurvan
            if let Some((_, rows)) = &tdam {
                for &(_, x, _) in rows {
                    if x > 0.0 && x < u0 && pts.iter().all(|p| (p.1 - x).abs() > 1e-14 * u0) {
                        pts.push((st0 - (st0 - sres) * x / u0, x, line));
                    }
                }
                pts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
            (curve(tk, &pts, tdam, e, "sprickvidd")?, true)
        }
        o => return tk.err(format!("TYPE={o} stöds inte (STRAIN, DISPLACEMENT, GFI)")),
    };
    let st0 = tens.0[0].1;
    let alpha = (fb - 1.0) / (2.0 * fb - 1.0);
    let gamma = 3.0 * (1.0 - kc) / (2.0 * kc - 1.0);
    let tan_psi = psi.to_radians().tan();
    let g = e / (2.0 * (1.0 + nu));
    let k = e / (3.0 * (1.0 - 2.0 * nu));
    Ok(Some(Arc::new(Concrete {
        e,
        nu,
        g,
        k,
        ce: iso_d6(e, nu),
        tan_psi,
        a: ecc * st0 * tan_psi,
        alpha,
        gamma,
        mu,
        comp,
        tens,
        tens_disp,
        wt,
        wc,
    })))
}

/// r(σ̂) = Σ⟨σ̂ᵢ⟩/Σ|σ̂ᵢ| (0 för σ = 0).
fn weight(s: &[f64; 3]) -> f64 {
    let a: f64 = s.iter().map(|x| x.abs()).sum();
    if a > 0.0 {
        s.iter().map(|x| x.max(0.0)).sum::<f64>() / a
    } else {
        0.0
    }
}

/// Resultat av en inviskös retur.
struct Ret {
    sig_bar: [f64; 3],
    dep: [f64; 3],
    et: f64,
    ec: f64,
}

impl Concrete {
    /// (σ̄c, dc, σ̄t, dt, σc, σt) vid hårdnandevariablerna.
    fn hard(&self, et: f64, ec: f64, len: f64) -> (f64, f64, f64, f64) {
        let (sc, dc) = self.comp.eval(ec);
        let xt = if self.tens_disp { et * len } else { et };
        let (st, dt) = self.tens.eval(xt);
        (sc, dc, st, dt)
    }

    fn yield_f(&self, s: &[f64; 3], et: f64, ec: f64, len: f64) -> f64 {
        let (sc, dc, st, dt) = self.hard(et, ec, len);
        let scb = sc / (1.0 - dc);
        let stb = (st / (1.0 - dt)).max(1e-12 * scb);
        let p = -(s[0] + s[1] + s[2]) / 3.0;
        let d = [s[0] + p, s[1] + p, s[2] + p];
        let q = (1.5 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).sqrt();
        let smax = s[0].max(s[1]).max(s[2]);
        let al = self.alpha;
        let beta = scb / stb * (1.0 - al) - (1.0 + al);
        (q - 3.0 * al * p + beta * smax.max(0.0) - self.gamma * (-smax).max(0.0)) / (1.0 - al) - scb
    }

    /// Inviskös retur för givet Δλ.
    fn at(&self, ltr: &[f64; 3], dl: f64, etn: f64, ecn: f64) -> Ret {
        let ptr = -(ltr[0] + ltr[1] + ltr[2]) / 3.0;
        let str_ = [ltr[0] + ptr, ltr[1] + ptr, ltr[2] + ptr];
        let qtr = (1.5 * (str_[0] * str_[0] + str_[1] * str_[1] + str_[2] * str_[2])).sqrt();
        // q + 3G Δλ q/√(a² + q²) = q_tr
        let mut q = if self.a > 0.0 { qtr } else { (qtr - 3.0 * self.g * dl).max(0.0) };
        if self.a > 0.0 {
            q = (qtr - 3.0 * self.g * dl).max(0.0).max(qtr * 1e-3);
            for _ in 0..100 {
                let r = (self.a * self.a + q * q).sqrt();
                let f = q + 3.0 * self.g * dl * q / r - qtr;
                let df = 1.0 + 3.0 * self.g * dl * self.a * self.a / (r * r * r);
                let dq = f / df;
                q = (q - dq).max(0.0);
                if dq.abs() <= 1e-15 * qtr.max(1e-300) {
                    break;
                }
            }
        }
        let gq = if self.a > 0.0 { q / (self.a * self.a + q * q).sqrt() } else if q > 0.0 { 1.0 } else { 0.0 };
        let p = ptr + self.k * dl * self.tan_psi;
        let f = if qtr > 0.0 { q / qtr } else { 0.0 };
        let mut sig_bar = [0.0; 3];
        let mut dep = [0.0; 3];
        for i in 0..3 {
            sig_bar[i] = str_[i] * f - p;
            let n = if qtr > 0.0 { 1.5 * str_[i] / qtr * gq } else { 0.0 };
            dep[i] = dl * (n + self.tan_psi / 3.0);
        }
        let r = weight(&sig_bar);
        let dmax = dep[0].max(dep[1]).max(dep[2]);
        let dmin = dep[0].min(dep[1]).min(dep[2]);
        Ret { sig_bar, dep, et: etn + (r * dmax).max(0.0), ec: ecn + (-(1.0 - r) * dmin).max(0.0) }
    }

    /// Hela uppdateringen i 3D: spänning; nytt tillstånd i `stn`.
    fn ret(&self, eps: &V6, st: &[f64], stn: &mut [f64], dt: f64, len: f64) -> Result<V6> {
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        stn[..NST].copy_from_slice(&st[..NST]);
        let (etn, ecn) = (st[ET], st[EC]);
        let (ltr, v) = principal(&sig_tr);
        let f_tr = self.yield_f(&ltr, etn, ecn, len);
        let scale = ltr.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.comp.0[0].1);
        let res = if f_tr <= 1e-12 * scale {
            Ret { sig_bar: ltr, dep: [0.0; 3], et: etn, ec: ecn }
        } else {
            let f = |dl: f64| -> Result<f64> {
                let r = self.at(&ltr, dl, etn, ecn);
                Ok(self.yield_f(&r.sig_bar, r.et, r.ec, len))
            };
            let x0 = f_tr / (3.0 * self.g + 3.0 * self.k * self.tan_psi);
            let dl = root_from_zero(f, x0, 1e-13 * scale, "betong med skadeplasticitet")?;
            self.at(&ltr, dl, etn, ecn)
        };
        stn[ET] = res.et;
        stn[EC] = res.ec;
        let dep = from_principal(&res.dep, &v, true);
        for i in 0..6 {
            stn[EP + i] = st[EP + i] + dep[i];
        }
        // styvhetsdegradering med återhämtning vid lastomkastning
        let (_, dc, _, dtt) = self.hard(res.et, res.ec, len);
        let r = weight(&res.sig_bar);
        let s_t = 1.0 - self.wt * r;
        let s_c = 1.0 - self.wc * (1.0 - r);
        let d = 1.0 - (1.0 - s_t * dc) * (1.0 - s_c * dtt);
        // viskös regularisering
        let th = if self.mu > 0.0 { if dt > 0.0 { dt / (self.mu + dt) } else { 0.0 } } else { 1.0 };
        for i in 0..6 {
            stn[EVP + i] = th * stn[EP + i] + (1.0 - th) * st[EVP + i];
        }
        // DAMAGEC/DAMAGET redovisas utan regularisering, SDEG regulariserad
        stn[DCV] = dc;
        stn[DTV] = dtt;
        stn[DV] = th * d + (1.0 - th) * st[DV];
        let ev: V6 = std::array::from_fn(|i| eps[i] - stn[EVP + i]);
        let sb = mv6(&self.ce, &ev);
        Ok(std::array::from_fn(|i| (1.0 - stn[DV]) * sb[i]))
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64], dt: f64, len: f64) -> Result<(V6, M6)> {
        let sig = self.ret(eps, st, stn, dt, len)?;
        let plastic = stn[ET] != st[ET] || stn[EC] != st[EC] || (0..6).any(|i| stn[EP + i] != st[EP + i]);
        if !plastic && stn[DV] == 0.0 && (0..6).all(|i| stn[EVP + i] == stn[EP + i]) {
            return Ok((sig, self.ce));
        }
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.tens.0[0].1 / self.e);
        let mut tmp = vec![0.0; NST];
        let c = fd_tangent_central(eps, 1e-7 * scale, |e2| self.ret(e2, st, &mut tmp, dt, len))?;
        Ok((sig, c))
    }
}

impl MaterialModel for Concrete {
    fn name(&self) -> &str {
        "CONCRETE DAMAGED PLASTICITY"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn requires_unsymmetric(&self) -> bool {
        // betongmodellen använder alltid den osymmetriska lösaren
        true
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        for o in [EP, EVP, EPS3] {
            super::plasticity::rotate_v6(r, &mut state[o..o + 6], true);
        }
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        // elementet anger ingen karakteristisk längd (fel för TYPE=DISPLACEMENT/GFI)
        let len = if self.tens_disp { 0.0 } else { 1.0 };
        self.update_with_length(input, len, state_old, state_new)
    }

    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        if self.tens_disp && !(length > 0.0) {
            return fail("betong med skadeplasticitet: elementets karakteristiska längd saknas för TYPE=DISPLACEMENT/GFI");
        }
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let dt = input.dt;
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, state_old, state_new, dt, length) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ", "PEEQT", "DAMAGEC", "DAMAGET", "SDEG"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[EC], state[ET], state[DCV], state[DTV], state[DV]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        let mut v: Vec<(&'static str, Vec<&'static str>)> = self.output_names().iter().map(|n| (*n, vec![*n])).collect();
        v.push(("PE", super::plasticity::pe_labels(mode).0));
        v
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = self.output_values(state);
        let o = if self.mu > 0.0 { EVP } else { EP };
        v.extend(super::plasticity::pe_labels(mode).1.iter().map(|&i| state[o + i]));
        if mode == StressMode::PlaneStress {
            v[7] = 0.0;
        }
        v
    }
}
