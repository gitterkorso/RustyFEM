//! Betong med utsmetad sprickbildning (*CONCRETE med *TENSION STIFFENING,
//! *SHEAR RETENTION och *FAILURE RATIOS) för implicit analys.
//!
//! Tryckytan (isotropt hårdnande, associativ flytning):
//!
//!   f_c = q − √3 a0 p − √3 τ_c(κ) = 0,  τ_c = (1/√3 − a0/3) σ_c(κ),
//!   dε^pl = dλ (1 + c0 (p/σ_c)²) ∂f_c/∂σ,  dκ = dλ (1 + c0/9)(1 − a0/√3),
//!
//! där σ_c(κ) är *CONCRETE-tabellen (κ = plastisk töjning i enaxligt tryck,
//! utdata PEEQ), a0 och c0 ur brottkvoterna r_bc^σ och r_bc^ε.
//!
//! Sprickdetektering (Coulomblinje, associativ "plasticitet" under
//! inkrementet då sprickan bildas):
//!
//!   f_t = q − (3 − b0 s) p − (2 − b0 s/3) σ_t = 0,  s = σ_t/σ_tu,
//!
//! σ_tu = r_t^σ · σ_cu (σ_cu = största spänningen i *CONCRETE), b0 ur
//! kvoterna r_t^σ och f. Under detekteringsinkrementet är
//! σ_t = σ_tu g̃(x) med x = Δε^pl_nn − (σ_tu − σ_t)/E (Δε^pl_nn = största
//! huvudtöjningen av detekteringens plastiska töjning) och
//! g̃ = 0,01 + 0,99 g, där g är tabellen i *TENSION STIFFENING (spänningskvot
//! mot töjning efter sprickbildning; TYPE=DISPLACEMENT: linjärt till noll vid
//! töjningen u0/L − σ_t/E efter sprickbildningen, L = elementets
//! karakteristiska längd, dvs. u0 räknas på den totala töjningen).
//! Materialet mjuknar alltså till
//! 1 % av spänningen, inte till noll.
//!
//! Sprickans normal är detekteringsspänningens största huvudriktning. Efter
//! sprickbildningen: normalkomponenten av detekteringens plastiska
//! töjning blir elastisk; de elastiska töjningarna i de andra riktningarna
//! blir (σ_ββ − ν σ_γγ)/E (normalspänningens tvärkontraktion tas bort) och
//! resten plastisk (utdata PE). Därefter skadad elasticitet i sprickans
//! koordinatsystem med den elastiska normaltöjningen e = ε_nn − ε^pl_nn:
//! - öppen spricka (e > 0): σ_nn = σ_d g̃(e − ε_c) vid pålastning
//!   (e ≥ e_max), annars sekant mot origo σ_nn = e σ_open(e_max)/e_max;
//!   σ_d = normalspänningen vid detekteringen, ε_c = σ_d/E; ingen
//!   tvärkontraktion mellan normalriktningen och de övriga; skjuvmodulen för
//!   sprickans skjuvkomponenter ρ G med ρ = ρ_close (1 − e/ε_max) (
//!   produkten av *SHEAR RETENTION-värdena);
//! - stängd spricka (e ≤ 0): full elasticitet, skjuvmodul ρ_close G.
//!
//! Med stängd spricka prövas tryckytan som i osprucket material (den
//! isotropa elastiska matrisen används i returen).
//!
//! Begränsningar: högst en spricka per integrationspunkt (ingen ytterligare
//! sprickdetektering i ett sprucket läge); temperatur- och fältberoende data
//! stöds inte.
//! Tangenten är numerisk (framåtdifferens av hela uppdateringen).

use super::plasticity::{iso_d6, mv6, update_reduced, M6, V6};
use super::principal::{fd_tangent, interp, principal};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::{MaterialDef, RawKeyword};
use crate::kinematics::tensor::M3;
use crate::model::{fail, Result};
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "betong med utsmetad sprickbildning",
    consumes: &["CONCRETE", "ELASTIC", "FAILURE RATIOS", "SHEAR RETENTION", "TENSION STIFFENING"],
    priority: 10,
    build: Some(build),
}];

const EPL: usize = 0; // oelastisk töjning (6, ingenjörsskjuvning) = PE
const KAPPA: usize = 6; // tryckytans hårdnandevariabel (PEEQ)
const NCR: usize = 7; // antal sprickor (0 eller 1)
const RR: usize = 8; // sprickans koordinatsystem (3×3, kolumner = lokala axlar)
const SD: usize = 17; // normalspänning vid detekteringen
const EC: usize = 18; // ε_c = σ_d/E
const EMAX: usize = 19; // största elastiska normaltöjning (öppen spricka)
const EPS3: usize = 20; // 3D-töjning (för plan spänning)
const NST: usize = 26;

#[derive(Debug, Clone)]
enum Stiffening {
    /// (töjning efter sprickbildning, spänningskvot)
    Strain(Vec<(f64, f64)>),
    /// u0
    Displacement(f64),
}

#[derive(Debug, Clone)]
pub struct ConcreteSmeared {
    e: f64,
    nu: f64,
    g: f64,
    lam: f64,
    k: f64,
    ce: M6,
    /// (κ, σ_c)
    hard: Vec<(f64, f64)>,
    a0: f64,
    c0: f64,
    b0: f64,
    stu: f64,
    ts: Stiffening,
    rho_close: f64,
    eps_max: f64,
}

fn single(kw: &RawKeyword) -> Result<Vec<f64>> {
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperatur- och fältberoende data stöds inte ännu");
    }
    kw.first_floats()
}

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(ck) = def.behavior("CONCRETE") else { return Ok(None) };
    let (e, nu) = super::mohr_coulomb::iso_elastic(def, ck, "*CONCRETE")?;
    if ck.has("DEPENDENCIES") {
        return ck.err("fältberoende tryckhårdnande stöds inte ännu");
    }
    let mut hard: Vec<(f64, f64)> = Vec::new();
    for d in ck.records() {
        let v = ck.floats(&d)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return ck.err_at(d.line, "temperaturberoende tryckhårdnande stöds inte ännu");
        }
        let (s, p) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(s > 0.0) {
            return ck.err_at(d.line, "tryckspänningen ska vara positiv");
        }
        match hard.last() {
            Some(&(pl, _)) if !(p > pl) => return ck.err_at(d.line, "den plastiska töjningen ska öka"),
            None if p != 0.0 => return ck.err_at(d.line, "första punkten ska ha plastisk töjning 0"),
            _ => {}
        }
        hard.push((p, s));
    }
    if hard.is_empty() {
        return ck.err("tryckhårdnandetabell saknas");
    }
    let (mut rbc, mut rt, mut rbe, mut f) = (1.16, 0.09, 1.28, 1.0 / 3.0);
    if let Some(fk) = def.behavior("FAILURE RATIOS") {
        let v = single(fk)?;
        let get = |i: usize, d: f64| v.get(i).copied().filter(|x| *x != 0.0).unwrap_or(d);
        rbc = get(0, rbc);
        rt = get(1, rt);
        rbe = get(2, rbe);
        f = get(3, f);
    }
    let Some(tk) = def.behavior("TENSION STIFFENING") else {
        return ck.err("*CONCRETE kräver *TENSION STIFFENING");
    };
    let ts = match tk.param_upper("TYPE").unwrap_or_else(|| "STRAIN".into()).as_str() {
        "STRAIN" => {
            if tk.has("DEPENDENCIES") {
                return tk.err("fältberoende stöds inte ännu");
            }
            let mut t: Vec<(f64, f64)> = Vec::new();
            for d in tk.records() {
                let v = tk.floats(&d)?;
                if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
                    return tk.err_at(d.line, "temperaturberoende stöds inte ännu");
                }
                let (r, x) = (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
                match t.last() {
                    Some(&(xl, _)) if !(x > xl) => return tk.err_at(d.line, "töjningen ska öka"),
                    None if x != 0.0 || r != 1.0 => return tk.err_at(d.line, "första punkten ska vara spänningskvot 1 vid töjning 0"),
                    _ => {}
                }
                t.push((x, r));
            }
            if t.len() < 2 {
                return tk.err("*TENSION STIFFENING kräver minst två punkter");
            }
            Stiffening::Strain(t)
        }
        "DISPLACEMENT" => {
            let v = single(tk)?;
            let u0 = v.first().copied().unwrap_or(0.0);
            if !(u0 > 0.0) {
                return tk.err("u0 ska vara positivt");
            }
            Stiffening::Displacement(u0)
        }
        o => return tk.err(format!("TYPE={o} stöds inte (STRAIN, DISPLACEMENT)")),
    };
    let (rho_close, eps_max) = match def.behavior("SHEAR RETENTION") {
        Some(sk) => {
            let v = single(sk)?;
            (v.first().copied().filter(|x| *x != 0.0).unwrap_or(1.0), v.get(1).copied().filter(|x| *x != 0.0).unwrap_or(f64::INFINITY))
        }
        None => (1.0, f64::INFINITY),
    };
    let s3 = 3f64.sqrt();
    let a0 = s3 * (1.0 - rbc) / (1.0 - 2.0 * rbc);
    let c0 = 9.0 * (rbe * (s3 - a0) + (a0 - s3 / 2.0)) / (rbe * (a0 - s3) + rbc * rbc * (2.0 * s3 - 4.0 * a0));
    let b0 = 3.0 * (1.0 + (2.0 - f) * rt - (1.0 + (f * rt).powi(2) + f * rt).sqrt()) / (1.0 + rt * (1.0 - f));
    let scu = hard.iter().fold(0.0f64, |m, p| m.max(p.1));
    let g = e / (2.0 * (1.0 + nu));
    Ok(Some(Arc::new(ConcreteSmeared {
        e,
        nu,
        g,
        lam: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
        k: e / (3.0 * (1.0 - 2.0 * nu)),
        ce: iso_d6(e, nu),
        hard,
        a0,
        c0,
        b0,
        stu: rt * scu,
        ts,
        rho_close,
        eps_max,
    })))
}

/// Voigt (ingenjörsskjuvning om `eng`) → tensor.
fn tens(v: &V6, eng: bool) -> M3 {
    let f = if eng { 0.5 } else { 1.0 };
    [[v[0], f * v[3], f * v[4]], [f * v[3], v[1], f * v[5]], [f * v[4], f * v[5], v[2]]]
}

fn voigt(a: &M3, eng: bool) -> V6 {
    let f = if eng { 2.0 } else { 1.0 };
    [a[0][0], a[1][1], a[2][2], f * a[0][1], f * a[0][2], f * a[1][2]]
}

/// Rᵀ A R (till lokala axlar) eller R A Rᵀ (till globala).
fn rot(a: &M3, r: &M3, to_local: bool) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                for l in 0..3 {
                    s += if to_local { r[k][i] * a[k][l] * r[l][j] } else { r[i][k] * a[k][l] * r[j][l] };
                }
            }
            o[i][j] = s;
        }
    }
    o
}

impl ConcreteSmeared {
    fn sig_c(&self, kappa: f64) -> f64 {
        interp(&self.hard, kappa).0
    }

    /// g̃(x) = 0,01 + 0,99 g(x). `sref`: spänningen vid sprickbildningen
    /// (TYPE=DISPLACEMENT: u0 räknas från den totala töjningen, så att
    /// mjuknandet slutar vid töjningen u0/L − σ_ref/E efter sprickbildningen).
    fn gmod(&self, x: f64, len: f64, sref: f64) -> f64 {
        let g = match &self.ts {
            Stiffening::Strain(t) => interp(t, x.max(0.0)).0,
            Stiffening::Displacement(u0) => (1.0 - x.max(0.0) / (u0 / len - sref / self.e)).max(0.0),
        };
        0.01 + 0.99 * g.clamp(0.0, 1.0)
    }

    fn pq(s: &V6) -> (f64, f64, V6) {
        let p = -(s[0] + s[1] + s[2]) / 3.0;
        let d: V6 = [s[0] + p, s[1] + p, s[2] + p, s[3], s[4], s[5]];
        (p, super::plasticity::mises(&d), d)
    }

    fn f_c(&self, s: &V6, kappa: f64) -> f64 {
        let (p, q, _) = Self::pq(s);
        let s3 = 3f64.sqrt();
        q - s3 * self.a0 * p - s3 * (1.0 / s3 - self.a0 / 3.0) * self.sig_c(kappa)
    }

    fn f_t(&self, s: &V6, st: f64) -> f64 {
        let (p, q, _) = Self::pq(s);
        let r = st / self.stu;
        q - (3.0 - self.b0 * r) * p - (2.0 - self.b0 * r / 3.0) * st
    }

    /// Liten Newton med numerisk jacobian för två okända.
    fn newton2(mut x: [f64; 2], scale: [f64; 2], mut res: impl FnMut(&[f64; 2]) -> [f64; 2], what: &str) -> Result<[f64; 2]> {
        let mut r = res(&x);
        for _ in 0..100 {
            let n0 = r[0].abs().max(r[1].abs());
            if n0 <= 1e-13 {
                return Ok(x);
            }
            let mut j = [0.0; 4];
            for c in 0..2 {
                let h = 1e-7 * x[c].abs().max(scale[c]);
                let mut x2 = x;
                x2[c] += h;
                let r2 = res(&x2);
                j[c] = (r2[0] - r[0]) / h;
                j[2 + c] = (r2[1] - r[1]) / h;
            }
            let det = j[0] * j[3] - j[1] * j[2];
            if det == 0.0 || !det.is_finite() {
                break;
            }
            let dx = [(r[0] * j[3] - r[1] * j[1]) / det, (j[0] * r[1] - j[2] * r[0]) / det];
            let mut t = 1.0;
            loop {
                let xn = [x[0] - t * dx[0], x[1] - t * dx[1]];
                let rn = res(&xn);
                if (rn[0].is_finite() && rn[1].is_finite() && rn[0].abs().max(rn[1].abs()) < n0) || t < 1e-4 {
                    x = xn;
                    r = rn;
                    break;
                }
                t *= 0.5;
            }
        }
        if r[0].abs().max(r[1].abs()) <= 1e-9 {
            return Ok(x);
        }
        fail(format!("betong (*CONCRETE): returalgoritmen för {what} konvergerade inte"))
    }

    /// Spänningsuppdatering i 3D (total töjning, ingenjörsskjuvning).
    fn ret(&self, eps: &V6, len: f64, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        stn[..NST].copy_from_slice(&st[..NST]);
        if st[NCR] == 0.0 {
            return self.uncracked(eps, len, st, stn);
        }
        let s = self.cracked(eps, len, st, stn);
        if stn[NCR] < 0.0 {
            // stängd spricka: tryckytan som i osprucket material
            stn[NCR] = 1.0;
            if self.f_c(&s, st[KAPPA]) > 1e-12 * s.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.stu) {
                return self.compression(eps, &s, st, stn);
            }
        }
        Ok(s)
    }

    fn uncracked(&self, eps: &V6, len: f64, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EPL + i]);
        let str_ = mv6(&self.ce, &ee);
        let kn = st[KAPPA];
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.stu);
        if self.f_c(&str_, kn) > 1e-12 * scale {
            return self.compression(eps, &str_, st, stn);
        }
        self.detect(eps, len, &str_, st, stn)
    }

    /// Retur på tryckytan från prövospänningen `str_` (isotrop elasticitet).
    fn compression(&self, _eps: &V6, str_: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let kn = st[KAPPA];
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.stu);
        let (p_tr, q_tr, d_tr) = Self::pq(str_);
        let s3 = 3f64.sqrt();
        {
            // tryckytan: okända Δλ och p
            let kfac = (1.0 + self.c0 / 9.0) * (1.0 - self.a0 / s3);
            let sc0 = self.sig_c(kn);
            let x = Self::newton2(
                [0.0, p_tr],
                [1e-6, scale],
                |x| {
                    let (dl, p) = (x[0], x[1]);
                    let sc = self.sig_c(kn + dl * kfac);
                    let dmu = dl * (1.0 + self.c0 * (p / sc).powi(2));
                    let q = q_tr - 3.0 * self.g * dmu;
                    let pp = p_tr + self.k * s3 * self.a0 * dmu;
                    [(p - pp) / sc0, (q - s3 * self.a0 * p - s3 * (1.0 / s3 - self.a0 / 3.0) * sc) / sc0]
                },
                "tryckytan",
            )?;
            let (dl, p) = (x[0], x[1]);
            let sc = self.sig_c(kn + dl * kfac);
            let dmu = dl * (1.0 + self.c0 * (p / sc).powi(2));
            let q = q_tr - 3.0 * self.g * dmu;
            let f = if q_tr > 0.0 { q / q_tr } else { 0.0 };
            let s: V6 = std::array::from_fn(|i| d_tr[i] * f - if i < 3 { p } else { 0.0 });
            for i in 0..6 {
                let n = if q_tr > 0.0 { 1.5 * d_tr[i] / q_tr } else { 0.0 } + if i < 3 { self.a0 / s3 } else { 0.0 };
                stn[EPL + i] = st[EPL + i] + dmu * n * if i < 3 { 1.0 } else { 2.0 };
            }
            stn[KAPPA] = kn + dl * kfac;
            Ok(s)
        }
    }

    /// Sprickdetektering från prövospänningen `str_`.
    fn detect(&self, eps: &V6, len: f64, str_: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        let str_ = *str_;
        let scale = str_.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(self.stu);
        let (p_tr, q_tr, d_tr) = Self::pq(&str_);
        if self.f_t(&str_, self.stu) <= 1e-12 * scale {
            return Ok(str_);
        }
        // sprickdetektering: okända Δλ och σ_t
        let (lt, v) = principal(&str_);
        let smax = lt[0] + p_tr; // största deviatoriska huvudvärdet
        let nmax = |dl: f64, kf: f64| -> f64 { dl * (if q_tr > 0.0 { 1.5 * smax / q_tr } else { 0.0 } + kf) };
        let x = Self::newton2(
            [0.0, self.stu],
            [1e-9, self.stu],
            |x| {
                let (dl, sgt) = (x[0], x[1]);
                let r = sgt / self.stu;
                let kf = 1.0 - self.b0 * r / 3.0;
                let q = q_tr - 3.0 * self.g * dl;
                let p = p_tr + 3.0 * self.k * kf * dl;
                let xx = nmax(dl, kf) - (self.stu - sgt) / self.e;
                [(q - (3.0 - self.b0 * r) * p - (2.0 - self.b0 * r / 3.0) * sgt) / self.stu, (sgt - self.stu * self.gmod(xx, len, sgt)) / self.stu]
            },
            "sprickdetekteringen",
        )?;
        let (dl, sgt) = (x[0], x[1]);
        let kf = 1.0 - self.b0 * sgt / self.stu / 3.0;
        let q = q_tr - 3.0 * self.g * dl;
        let p = p_tr + 3.0 * self.k * kf * dl;
        let f = if q_tr > 0.0 { q / q_tr } else { 0.0 };
        let s: V6 = std::array::from_fn(|i| d_tr[i] * f - if i < 3 { p } else { 0.0 });
        // huvudspänningar i sprickans system (koaxiella med prövospänningen)
        let sl = rot(&tens(&s, false), &v, true);
        let (s1, s2, s3v) = (sl[0][0], sl[1][1], sl[2][2]);
        let el = rot(&tens(eps, true), &v, true);
        let pl_old = rot(&tens(&std::array::from_fn(|i| st[EPL + i]), true), &v, true);
        let mut pl = el;
        pl[0][0] = pl_old[0][0];
        pl[1][1] = el[1][1] - (s2 - self.nu * s3v) / self.e;
        pl[2][2] = el[2][2] - (s3v - self.nu * s2) / self.e;
        let pg = voigt(&rot(&pl, &v, false), true);
        stn[EPL..EPL + 6].copy_from_slice(&pg);
        for i in 0..3 {
            for j in 0..3 {
                stn[RR + 3 * i + j] = v[i][j];
            }
        }
        stn[NCR] = 1.0;
        stn[SD] = s1;
        stn[EC] = s1 / self.e;
        stn[EMAX] = s1 / self.e;
        Ok(s)
    }

    fn cracked(&self, eps: &V6, len: f64, st: &[f64], stn: &mut [f64]) -> V6 {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = st[RR + 3 * i + j];
            }
        }
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EPL + i]);
        let e = voigt(&rot(&tens(&ee, true), &r, true), true);
        let (lam, g) = (self.lam, self.g);
        let mut s = [0.0; 6];
        if e[0] > 0.0 {
            let (sd, mut ec, emax) = (st[SD], st[EC], st[EMAX]);
            if matches!(self.ts, Stiffening::Displacement(_)) {
                // TYPE=DISPLACEMENT: mjuknandelinjen går genom σ_d vid σ_d/(0,99 E)
                ec /= 0.99;
            }
            s[0] = if e[0] >= emax {
                stn[EMAX] = e[0];
                sd * self.gmod(e[0] - ec, len, sd)
            } else {
                e[0] * sd * self.gmod(emax - ec, len, sd) / emax
            };
            s[1] = lam * (e[1] + e[2]) + 2.0 * g * e[1];
            s[2] = lam * (e[1] + e[2]) + 2.0 * g * e[2];
            let rho = self.rho_close * (1.0 - e[0] / self.eps_max).max(0.0);
            s[3] = rho * g * e[3];
            s[4] = rho * g * e[4];
            s[5] = g * e[5];
        } else {
            // markera stängd spricka (tryckytan prövas i `ret`)
            stn[NCR] = -1.0;
            let tr = e[0] + e[1] + e[2];
            for i in 0..3 {
                s[i] = lam * tr + 2.0 * g * e[i];
            }
            s[3] = self.rho_close * g * e[3];
            s[4] = self.rho_close * g * e[4];
            s[5] = g * e[5];
        }
        voigt(&rot(&tens(&s, false), &r, false), false)
    }

    fn update3d(&self, eps: &V6, len: f64, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let sig = self.ret(eps, len, st, stn)?;
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * self.stu / self.e);
        let mut tmp = vec![0.0; NST];
        let c = fd_tangent(eps, &sig, 1e-7 * scale, |e2| self.ret(e2, len, st, &mut tmp))?;
        Ok((sig, c))
    }
}

impl MaterialModel for ConcreteSmeared {
    fn name(&self) -> &str {
        "CONCRETE"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        self.update_with_length(input, 1.0, state_old, state_new)
    }

    fn update_with_length(&self, input: &MatInput, length: f64, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        if input.mode == StressMode::Uniaxial {
            return fail("*CONCRETE stöds inte ännu i stänger och balkar");
        }
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update3d(e, length, state_old, state_new) };
        let (eps3, out) = update_reduced(input.mode, input.strain, &eps_old, &self.ce, &mut f)?;
        state_new[EPS3..EPS3 + 6].copy_from_slice(&eps3);
        Ok(out)
    }

    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        Ok(super::anisotropic::reduce_6x6(&self.ce, mode))
    }

    fn output_names(&self) -> &[&'static str] {
        &["PEEQ"]
    }

    fn output_values(&self, state: &[f64]) -> Vec<f64> {
        vec![state[KAPPA]]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", super::plasticity::pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![state[KAPPA]];
        v.extend(super::plasticity::pe_labels(mode).1.iter().map(|&i| state[EPL + i]));
        if mode == StressMode::PlaneStress {
            // PE33 = 0 redovisas i plan spänning
            v[3] = 0.0;
        }
        v
    }
}
