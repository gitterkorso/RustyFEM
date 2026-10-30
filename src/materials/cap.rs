//! Modified Drucker–Prager Cap (*CAP PLASTICITY with *CAP HARDENING) and
//! linear elasticity (*ELASTIC).
//!
//! Modified Drucker–Prager/Cap model for geological materials:
//! with p, q, t (K) as in `invariant_return.rs` and c = 1 + α − α/cos β, we have
//!
//! - the shear failure surface Fs = t − p tan β − d = 0 (perfectly plastic),
//! - the cap Fc = √((p − pa)² + (R t/c)²) − R (d + pa tan β) = 0 for p ≥ pa,
//! - the transition surface Ft = √((p − pa)² + (t − (1 − α/cos β)(d + pa tan β))²)
//!   − α (d + pa tan β) = 0 between the shear surface (where it is tangent to it, at
//!   p = pa − α (d + pa tan β) sin β) and pa,
//!
//! pa = (pb − R d)/(1 + R tan β) and pb(ε_vol^in|0 + ε_vol) from
//! *CAP HARDENING (pb versus volumetric inelastic strain, compaction positive).
//! The flow is associated on the cap, Gc = √((p − pa)² + (R t/c)²), and
//! non-associated on the shear and transition surfaces,
//! Gs = √(((pa − p) tan β)² + (t/c)²); ε̇_vol = −tr ε̇ᵖ (compaction hardens
//! the cap, dilatation on the shear surface softens it).
//!
//! The integration is backward Euler with the shared return-mapping algorithm
//! (`invariant_return.rs`), where the active surface is chosen from p relative to pa;
//! the surface is continuously differentiable when α > 0. The tangent is numerical
//! (central difference) and unsymmetric; it is used in full with *STEP,
//! UNSYMM=YES and otherwise as its symmetric part.
//!
//! Output: PEEQ = cap position pb, PE = plastic strain.

use super::invariant_return::{return_map, Inv, Surface};
use super::plasticity::{iso_d6, mv6, pe_labels, rotate_v6, update_reduced, Hardening, M6, V6};
use super::{MatInput, MatOutput, MaterialFactory, MaterialModel, StressMode};
use crate::fem::model::MaterialDef;
use crate::model::Result;
use std::sync::Arc;

pub static FACTORIES: &[MaterialFactory] = &[MaterialFactory {
    name: "Drucker–Prager Cap (*CAP PLASTICITY)",
    consumes: &["ELASTIC", "CAP PLASTICITY", "CAP HARDENING"],
    priority: 10,
    build: Some(build),
}];

#[derive(Debug, Clone)]
pub struct Cap {
    e: f64,
    nu: f64,
    g: f64,
    kb: f64,
    ce: M6,
    d: f64,
    tan_beta: f64,
    cos_beta: f64,
    sin_beta: f64,
    r: f64,
    eps0: f64,
    alpha: f64,
    k: f64,
    /// pb versus volumetric inelastic strain.
    pb: Hardening,
}

/// State: [ε_vol (compaction), εᵖ (6), ε 3D (6)].
const EV: usize = 0;
const EP: usize = 1;
const EPS3: usize = 7;
const NST: usize = 13;

fn build(def: &MaterialDef) -> Result<Option<Arc<dyn MaterialModel>>> {
    let Some(kw) = def.behavior("CAP PLASTICITY") else { return Ok(None) };
    let Some(el) = def.behavior("ELASTIC") else {
        if def.behavior("POROUS ELASTIC").is_some() {
            return kw.err("*CAP PLASTICITY with *POROUS ELASTIC is not supported yet (use *ELASTIC)");
        }
        return kw.err("*CAP PLASTICITY requires *ELASTIC");
    };
    if el.param_upper("TYPE").is_some_and(|t| t != "ISOTROPIC") || el.data.len() > 1 {
        return el.err("Drucker–Prager Cap requires isotropic, temperature-independent elasticity");
    }
    let ev = el.first_floats()?;
    let (e, nu) = (ev.first().copied().unwrap_or(0.0), ev.get(1).copied().unwrap_or(0.0));
    if !(e > 0.0 && nu > -1.0 && nu < 0.5) {
        return el.err("E must be positive and ν between −1 and 0.5");
    }
    if kw.data.len() > 1 || kw.has("DEPENDENCIES") {
        return kw.err("temperature- or field-dependent cap parameters are not supported yet");
    }
    let v = kw.first_floats()?;
    let g = |i: usize| v.get(i).copied().unwrap_or(0.0);
    let (d, beta, r, eps0, alpha) = (g(0), g(1), g(2), g(3), g(4));
    let k = v.get(5).copied().filter(|&x| x != 0.0).unwrap_or(1.0);
    if v.len() > 6 && v[6..].iter().any(|&x| x != 0.0) {
        return kw.err("temperature-dependent cap parameters are not supported yet");
    }
    if !(d >= 0.0) || !(0.0..90.0).contains(&beta) {
        return kw.err("the cohesion d must be non-negative and β between 0 and 90 degrees");
    }
    if !(r > 0.0) {
        return kw.err("the cap eccentricity R must be positive");
    }
    if !(alpha >= 0.0 && alpha < 1.0) {
        return kw.err("the transition surface parameter α must lie between 0 and 1");
    }
    if !(0.778 - 1e-9..=1.0).contains(&k) {
        return kw.err("K must lie between 0.778 and 1");
    }
    let Some(hk) = def.behavior("CAP HARDENING") else {
        return kw.err("*CAP PLASTICITY requires *CAP HARDENING");
    };
    if hk.has("DEPENDENCIES") {
        return hk.err("field-dependent cap hardening is not supported yet");
    }
    let scale = match hk.param("SCALESTRESS") {
        Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| crate::model::FemError(format!("{}: SCALESTRESS must be a number", hk.location())))?,
        None => 1.0,
    };
    let mut t: Vec<(f64, f64)> = Vec::new();
    for dl in &hk.data {
        let v = hk.floats(dl)?;
        if v.len() > 2 && v[2..].iter().any(|&x| x != 0.0) {
            return hk.err_at(dl.line, "temperature-dependent cap hardening is not supported yet");
        }
        let (pb, ev) = (scale * v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0));
        if !(pb > 0.0) {
            return hk.err_at(dl.line, "pb must be positive");
        }
        if let Some(&(e0, p0)) = t.last() {
            if !(ev > e0) || !(pb >= p0) {
                return hk.err_at(dl.line, "the volumetric strain must increase and pb must not decrease");
            }
        }
        t.push((ev, pb));
    }
    if t.is_empty() {
        return hk.err("the hardening curve is missing");
    }
    let br = beta.to_radians();
    let gm = e / (2.0 * (1.0 + nu));
    let kb = e / (3.0 * (1.0 - 2.0 * nu));
    let m = Cap { e, nu, g: gm, kb, ce: iso_d6(e, nu), d, tan_beta: br.tan(), cos_beta: br.cos(), sin_beta: br.sin(), r, eps0, alpha, k, pb: Hardening::Table(t) };
    if !(m.pa(0.0) > -m.d / m.tan_beta.max(1e-300)) {
        return kw.err("the initial cap position pa lies beyond the apex of the shear surface (pb too small at ε_vol^in|0)");
    }
    Ok(Some(Arc::new(m)))
}

impl Cap {
    fn pa(&self, h: f64) -> f64 {
        let pb = self.pb.eval(self.eps0 + h).0;
        (pb - self.r * self.d) / (1.0 + self.r * self.tan_beta)
    }

    fn c(&self) -> f64 {
        1.0 + self.alpha - self.alpha / self.cos_beta
    }

    fn update3d(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<V6> {
        stn[..NST].copy_from_slice(&st[..NST]);
        let ee: V6 = std::array::from_fn(|i| eps[i] - st[EP + i]);
        let sig_tr = mv6(&self.ce, &ee);
        let res = match return_map(self, self.g, self.kb, &sig_tr, st[EV]) {
            // the corner between the shear surface and the cap (α = 0): return with two plastic multipliers
            Err(e) => match self.fixed_return(&sig_tr, st[EV]).or_else(|| self.corner(&sig_tr, st[EV])) {
                Some(r) => Ok(Some(r)),
                None => Err(e),
            },
            r => r,
        };
        match res {
            Ok(None) => Ok(sig_tr),
            Ok(Some(r)) => {
                stn[EV] = r.h;
                for i in 0..6 {
                    stn[EP + i] = st[EP + i] + r.dep[i];
                }
                Ok(r.sig)
            }
            Err(e) => Err(crate::model::FemError(format!("Drucker–Prager Cap: {e}"))),
        }
    }

    /// Return to one surface at a time (cap, transition surface, shear surface); the first
    /// solution that lies in its own region is accepted.
    fn fixed_return(&self, sig_tr: &V6, h_n: f64) -> Option<super::invariant_return::Return> {
        let tol = 1e-9 * self.stress_scale(h_n);
        for region in [CAP, TRANS, SHEAR] {
            if region == TRANS && self.alpha <= 0.0 {
                continue;
            }
            let Ok(Some(r)) = return_map(&Fixed { cap: self, region }, self.g, self.kb, sig_tr, h_n) else { continue };
            let p = super::invariant_return::invariants(&r.sig, self.k).p;
            let pa = self.pa(r.h);
            let da = self.d + pa * self.tan_beta;
            let pt = pa - self.alpha * da * self.sin_beta;
            let ok = match region {
                CAP => p >= pa - tol,
                TRANS => p <= pa + tol && p >= pt - tol,
                _ => p <= pt + tol,
            };
            if ok {
                return Some(r);
            }
        }
        None
    }

    /// Return to the corner where the shear surface and the cap (or the transition surface) meet:
    /// Fs = 0 and Fc = 0 with the flow λs ∂Gs/∂σ + λc ∂Gc/∂σ (Koiter). `None` if
    /// Newton does not converge or a multiplier becomes negative.
    fn corner(&self, sig_tr: &V6, h_n: f64) -> Option<super::invariant_return::Return> {
        use super::invariant_return::{invariants, newton_fd, DP};
        let sc = self.stress_scale(h_n).max(sig_tr.iter().fold(0.0f64, |m, x| m.max(x.abs())));
        let et = sc / (2.0 * self.g);
        let c = self.c();
        let (tb, r) = (self.tan_beta, self.r);
        // flow directions and surfaces at the corner (shear form and cap form, respectively)
        let parts = |x: &[f64]| {
            let sig: V6 = std::array::from_fn(|i| x[i] * sc);
            let h = x[8];
            let inv = invariants(&sig, self.k);
            let pa = self.pa(h);
            let da = self.d + pa * tb;
            let (p, t) = (inv.p, inv.t);
            // Gs
            let xs = (pa - p) * tb;
            let ys = t / c;
            let ss = (xs * xs + ys * ys).sqrt().max(1e-300);
            let ns: V6 = std::array::from_fn(|i| (-xs * tb / ss) * DP[i] + (ys / (c * ss)) * inv.dt[i]);
            // Gc/R
            let yc = r * t / c;
            let sc2 = ((p - pa).powi(2) + yc * yc).sqrt().max(1e-300);
            let nc: V6 = std::array::from_fn(|i| ((p - pa) / (r * sc2)) * DP[i] + (yc / (c * sc2)) * inv.dt[i]);
            let fs = self.cos_beta * (t - p * tb - self.d);
            let fc = if self.alpha > 0.0 {
                let y = t - (1.0 - self.alpha / self.cos_beta) * da;
                ((p - pa).powi(2) + y * y).sqrt() - self.alpha * da
            } else {
                (c / r) * (sc2 - r * da)
            };
            (sig, ns, nc, fs, fc, h)
        };
        let resid = |x: &[f64]| -> Vec<f64> {
            let (sig, ns, nc, fs, fc, h) = parts(x);
            let (ls, lc) = (x[6] * et, x[7] * et);
            let n: V6 = std::array::from_fn(|i| ls * ns[i] + lc * nc[i]);
            let tr = n[0] + n[1] + n[2];
            let dn = super::plasticity::dev(&n);
            let mut out = vec![0.0; 9];
            for i in 0..6 {
                let cn = 2.0 * self.g * dn[i] + if i < 3 { self.kb * tr } else { 0.0 };
                out[i] = (sig[i] - sig_tr[i] + cn) / sc;
            }
            out[6] = fs / sc;
            out[7] = fc / sc;
            out[8] = (h - h_n + tr) / et;
            out
        };
        let mut x = vec![0.0; 9];
        for i in 0..6 {
            x[i] = sig_tr[i] / sc;
        }
        x[6] = 1e-3;
        x[7] = 1e-3;
        x[8] = h_n;
        let mut typ = vec![1.0; 9];
        typ[8] = et;
        let res = newton_fd(&mut x, &typ, &resid, 1e-13);
        if !(res <= 1e-9) || x[6] < 0.0 || x[7] < 0.0 {
            return None;
        }
        let (sig, ns, nc, _, _, h) = parts(&x);
        let (ls, lc) = (x[6] * et, x[7] * et);
        let dep: V6 = std::array::from_fn(|i| (ls * ns[i] + lc * nc[i]) * if i < 3 { 1.0 } else { 2.0 });
        Some(super::invariant_return::Return { sig, dep, h, dl: ls + lc })
    }

    fn update_tangent(&self, eps: &V6, st: &[f64], stn: &mut [f64]) -> Result<(V6, M6)> {
        let sig = self.update3d(eps, st, stn)?;
        if (0..6).all(|i| stn[EP + i] == st[EP + i]) && stn[EV] == st[EV] {
            return Ok((sig, self.ce));
        }
        let s0 = self.pb.eval(self.eps0 + st[EV]).0.max(self.d);
        let scale = eps.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-3 * s0 / self.e);
        let h = 1e-6 * scale;
        let mut c = [[0.0; 6]; 6];
        let mut tmp = vec![0.0; NST];
        for j in 0..6 {
            let mut ep = *eps;
            let mut em = *eps;
            ep[j] += h;
            em[j] -= h;
            let sp = self.update3d(&ep, st, &mut tmp)?;
            let sm = self.update3d(&em, st, &mut tmp)?;
            for i in 0..6 {
                c[i][j] = (sp[i] - sm[i]) / (2.0 * h);
            }
        }
        Ok((sig, c))
    }
}

/// Cap (0), transition surface (1) or shear surface (2).
const CAP: u8 = 0;
const TRANS: u8 = 1;
const SHEAR: u8 = 2;

impl Cap {
    /// Active surface for p at hardening h.
    fn region(&self, p: f64, h: f64) -> u8 {
        let pa = self.pa(h);
        let da = self.d + pa * self.tan_beta;
        if p >= pa {
            CAP
        } else if self.alpha > 0.0 && p >= pa - self.alpha * da * self.sin_beta {
            TRANS
        } else {
            SHEAR
        }
    }

    fn yield_piece(&self, region: u8, p: f64, t: f64, h: f64) -> (f64, f64, f64) {
        let pa = self.pa(h);
        let da = self.d + pa * self.tan_beta;
        let c = self.c();
        match region {
            CAP => {
                let y = self.r * t / c;
                let s = ((p - pa).powi(2) + y * y).sqrt();
                if s <= 0.0 {
                    return (-c * da, 0.0, 0.0);
                }
                // scaled by c/R so that F and ∂F are continuous at p = pa
                let k = c / self.r;
                (k * (s - self.r * da), k * (p - pa) / s, y / s)
            }
            TRANS => {
                // the transition surface is the upper arc of the circle; below the center
                // (t < tc) the point is elastic (F < 0, continuous at t = tc)
                let y = t - (1.0 - self.alpha / self.cos_beta) * da;
                let yp = y.max(0.0);
                let s = ((p - pa).powi(2) + yp * yp).sqrt().max(1e-300);
                (s - self.alpha * da + y.min(0.0), (p - pa) / s, if y > 0.0 { y / s } else { 1.0 })
            }
            _ => {
                // scaled by cos β so that F and ∂F are continuous towards the transition surface
                let cb = self.cos_beta;
                (cb * (t - p * self.tan_beta - self.d), -cb * self.tan_beta, cb)
            }
        }
    }

    fn potential_piece(&self, region: u8, p: f64, t: f64, h: f64) -> (f64, f64) {
        let pa = self.pa(h);
        let c = self.c();
        if region == CAP {
            let y = self.r * t / c;
            let s = ((p - pa).powi(2) + y * y).sqrt();
            if s <= 0.0 {
                return (1.0 / self.r, 0.0);
            }
            // Gc/R: same direction, continuous magnitude towards Gs at p = pa
            ((p - pa) / (self.r * s), y / (c * s))
        } else {
            let x = (pa - p) * self.tan_beta;
            let y = t / c;
            let s = (x * x + y * y).sqrt();
            if s <= 0.0 {
                return (0.0, 1.0);
            }
            (-x * self.tan_beta / s, y / (c * s))
        }
    }
}

/// Surface locked to one region (fallback when the return with automatic region selection
/// oscillates between the surfaces, e.g. at p ≈ pa).
struct Fixed<'a> {
    cap: &'a Cap,
    region: u8,
}

impl Surface for Fixed<'_> {
    fn k(&self) -> f64 {
        self.cap.k
    }
    fn yield_fn(&self, p: f64, t: f64, h: f64) -> (f64, f64, f64) {
        self.cap.yield_piece(self.region, p, t, h)
    }
    fn potential(&self, p: f64, t: f64, h: f64) -> (f64, f64) {
        self.cap.potential_piece(self.region, p, t, h)
    }
    fn hardening_rate(&self, sig: &V6, inv: &Inv, n: &V6, h: f64, h_n: f64) -> f64 {
        self.cap.hardening_rate(sig, inv, n, h, h_n)
    }
    fn stress_scale(&self, h: f64) -> f64 {
        self.cap.stress_scale(h)
    }
}

impl Surface for Cap {
    fn k(&self) -> f64 {
        self.k
    }

    fn yield_fn(&self, p: f64, t: f64, h: f64) -> (f64, f64, f64) {
        self.yield_piece(self.region(p, h), p, t, h)
    }

    fn potential(&self, p: f64, t: f64, h: f64) -> (f64, f64) {
        self.potential_piece(self.region(p, h), p, t, h)
    }

    fn hardening_rate(&self, _sig: &V6, _inv: &Inv, n: &V6, _h: f64, _h_n: f64) -> f64 {
        // compaction: −tr ε̇ᵖ/λ̇
        -(n[0] + n[1] + n[2])
    }

    fn stress_scale(&self, h: f64) -> f64 {
        self.pb.eval(self.eps0 + h).0.max(self.d)
    }
}

impl MaterialModel for Cap {
    fn name(&self) -> &str {
        "CAP PLASTICITY"
    }

    fn n_state(&self) -> usize {
        NST
    }

    fn elastic_moduli(&self) -> Option<(f64, f64)> {
        Some((self.e, self.nu))
    }

    fn rotate_state(&self, r: &[[f64; 3]; 3], state: &mut [f64]) {
        rotate_v6(r, &mut state[EP..EP + 6], true);
        rotate_v6(r, &mut state[EPS3..EPS3 + 6], true);
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let eps_old: V6 = std::array::from_fn(|i| state_old[EPS3 + i]);
        let mut f = |e: &V6| -> Result<(V6, M6)> { self.update_tangent(e, state_old, state_new) };
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
        vec![self.pb.eval(self.eps0 + state[EV]).0]
    }

    fn output_fields(&self, mode: StressMode) -> Vec<(&'static str, Vec<&'static str>)> {
        vec![("PEEQ", vec!["PEEQ"]), ("PE", pe_labels(mode).0)]
    }

    fn output_field_values(&self, mode: StressMode, state: &[f64]) -> Vec<f64> {
        let mut v = vec![self.pb.eval(self.eps0 + state[EV]).0];
        v.extend(pe_labels(mode).1.iter().map(|&i| state[EP + i]));
        if mode == StressMode::PlaneStress {
            v[3] = 0.0;
        }
        v
    }
}
