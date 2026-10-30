//! Timoshenkobalkar (B21, B22, B31, B32) med olinjärt material (område
//! struct_nonlin): integrering över tvärsnittet i sektionspunkter.
//!
//! Med *BEAM SECTION och ett olinjärt material (*PLASTIC, termisk töjning)
//! uppdateras materialet i varje sektionspunkt i enaxligt spänningstillstånd
//! med fibertöjningen ε_f = ε + x2·κ1 − x1·κ2 (x1, x2 i tvärsnittets axlar n1,
//! n2 från referensaxeln). Normalkraft och böjmoment blir
//!
//!   N = Σ w σ,  M1 = Σ w x2 σ,  M2 = −Σ w x1 σ,
//!
//! och tvärsnittets tangent på samma sätt med materialets tangent. Vridning
//! och tvärskjuvning är elastiska (G·J och k·G·A med det linjära elementets
//! formfaktorer och slankhetskompensation); skjuvspänningen från vridning
//! ingår alltså inte i flytvillkoret.
//!
//! Sektionspunkterna (läge och numrering):
//!
//! - RECT: 5 × 5 punkter (Simpson i båda riktningarna), numrerade radvis från
//!   (−a/2, −b/2) med x1 snabbast.
//! - CIRC: mittpunkten och två ringar (r/2 och r) med 8 punkter var, vinkeln
//!   räknad från −x2 mot +x1; Simpson radiellt (för ∫f·ρ dρ, mittpunktens
//!   vikt blir noll) och trapetsregeln runt om.
//! - PIPE (tunnväggigt): 8 punkter på medelradien r − t/2, med samma vinklar.
//! - BOX: 16 punkter längs mittlinjerna (Simpson med 5 punkter per vägg,
//!   hörnen delas): höger vägg nerifrån, överkanten från höger, vänster vägg
//!   uppifrån, underkanten från vänster.
//! - I: 5 punkter i underflänsen (från −b1/2), 3 inre punkter i livet
//!   (nerifrån) och 5 i överflänsen (från −b2/2); livets ändpunkter är
//!   flänsarnas mittpunkter (med samma avdrag för överlappet som i
//!   `beam_timoshenko_section.rs`).
//!
//! Plana balkar (B21, B22) har bara punkterna genom höjden: punkter med samma
//! x2 slås ihop (RECT: 5 punkter nerifrån).
//!
//! Med elastiskt material ger punkterna exakt samma A, S1, S2, I11, I22, I12
//! som tvärsnittsegenskaperna.

use crate::elements::api::{ElemCtx, IpField};
use crate::materials::{MatInput, MaterialModel, StressMode};
use crate::model::{fail, Result};
use std::f64::consts::PI;

/// Antal generaliserade töjningar som lagras per integrationspunkt.
pub const NGEN: usize = 6;

/// Sektionspunkter för en tvärsnittsform: (x1, x2, vikt).
pub fn section_points(shape: &str, v: &[f64]) -> Result<Vec<(f64, f64, f64)>> {
    let simp = [1.0, 4.0, 2.0, 4.0, 1.0];
    let need = |n: usize| if v.len() >= n { Ok(()) } else { fail(format!("SECTION={shape} kräver {n} mått på första dataraden")) };
    let mut p = Vec::new();
    match shape {
        "RECT" => {
            need(2)?;
            let (a, b) = (v[0], v[1]);
            for j in 0..5 {
                for i in 0..5 {
                    let w = (a / 12.0 * simp[i]) * (b / 12.0 * simp[j]);
                    p.push((-a / 2.0 + a * i as f64 / 4.0, -b / 2.0 + b * j as f64 / 4.0, w));
                }
            }
        }
        "CIRC" => {
            need(1)?;
            let r = v[0];
            p.push((0.0, 0.0, 0.0));
            for k in 0..8 {
                let f = (k as f64 * 45.0).to_radians();
                for (rho, wr) in [(0.5 * r, r * r / 3.0), (r, r * r / 6.0)] {
                    p.push((rho * f.sin(), -rho * f.cos(), wr * PI / 4.0));
                }
            }
        }
        "PIPE" => {
            need(2)?;
            let (r, t) = (v[0], v[1]);
            let rm = r - t / 2.0;
            for k in 0..8 {
                let f = (k as f64 * 45.0).to_radians();
                p.push((rm * f.sin(), -rm * f.cos(), 2.0 * PI * rm * t / 8.0));
            }
        }
        "BOX" => {
            need(6)?;
            let (a, b, t1, t2, t3, t4) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let xr = a / 2.0 - t1 / 2.0;
            let xl = -a / 2.0 + t3 / 2.0;
            let yt = b / 2.0 - t2 / 2.0;
            let yb = -b / 2.0 + t4 / 2.0;
            // väggarna: (start, slut, tjocklek)
            let walls = [([xr, yb], [xr, yt], t1), ([xr, yt], [xl, yt], t2), ([xl, yt], [xl, yb], t3), ([xl, yb], [xr, yb], t4)];
            let mut pts: Vec<(f64, f64, f64)> = Vec::new();
            for (wi, (s, e, t)) in walls.iter().enumerate() {
                let l = ((e[0] - s[0]).powi(2) + (e[1] - s[1]).powi(2)).sqrt();
                for k in 0..5 {
                    let f = k as f64 / 4.0;
                    let w = l / 12.0 * simp[k] * t;
                    let x = [s[0] + f * (e[0] - s[0]), s[1] + f * (e[1] - s[1])];
                    if k == 0 && wi > 0 {
                        // hörnet delas med föregående vägg
                        pts.last_mut().unwrap().2 += w;
                    } else if k == 4 && wi == 3 {
                        pts[0].2 += w;
                    } else {
                        pts.push((x[0], x[1], w));
                    }
                }
            }
            p = pts;
        }
        "I" => {
            need(7)?;
            let (l, h, b1, b2, t1, t2, t3) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            let yb = -l + t1 / 2.0;
            let yt = -l + h - t2 / 2.0;
            for i in 0..5 {
                p.push((-b1 / 2.0 + b1 * i as f64 / 4.0, yb, b1 / 12.0 * simp[i] * t1));
            }
            let lw = yt - yb;
            for k in 1..4 {
                p.push((0.0, yb + lw * k as f64 / 4.0, lw / 12.0 * simp[k] * t3));
            }
            for i in 0..5 {
                p.push((-b2 / 2.0 + b2 * i as f64 / 4.0, yt, b2 / 12.0 * simp[i] * t2));
            }
            // livets ändpunkter = flänsarnas mittpunkter, minus överlappet med flänsarna
            p[2].2 += lw / 12.0 * t3 - t3 * t1 / 2.0;
            p[10].2 += lw / 12.0 * t3 - t3 * t2 / 2.0;
        }
        // område beams2: L och ARBITRARY
        _ => match crate::elements::beams2::sections::section_points(shape, v) {
            Some(r) => p = r?,
            None => return fail(format!("SECTION={shape} stöds inte med olinjärt material för Timoshenkobalkar (RECT, CIRC, PIPE, BOX, I, L, ARBITRARY)")),
        },
    }
    Ok(p)
}

/// Tvärsnittet med sektionspunkter och material.
pub struct Fibers<'a> {
    pub pts: Vec<(f64, f64, f64)>,
    pub mat: &'a dyn MaterialModel,
    nst: usize,
}

/// Balken räknas med sektionspunkter: *BEAM SECTION med olinjärt material.
pub fn is_plastic(ctx: &ElemCtx) -> bool {
    match (ctx.section, ctx.material) {
        (Some(s), Some(m)) => s.raw.name == "BEAM SECTION" && !m.is_linear(),
        _ => false,
    }
}

impl<'a> Fibers<'a> {
    pub fn new(ctx: &ElemCtx<'a>, name: &str) -> Result<Fibers<'a>> {
        let sec = ctx.section(name)?;
        let mat = ctx.material(name)?;
        let shape = sec.beam().map(|b| b.shape.clone()).unwrap_or_default();
        let v = crate::elements::beams2::sections::data_values(&sec.raw, &shape)?;
        let mut pts = section_points(&shape, &v).map_err(|e| crate::model::FemError(format!("{}: {}", sec.raw.location(), e.0)))?;
        if name.starts_with("B2") {
            // plana balkar: bara punkterna genom höjden (x2), nerifrån
            let mut merged: Vec<(f64, f64, f64)> = Vec::new();
            for (_, x2, w) in pts {
                match merged.iter_mut().find(|p| (p.1 - x2).abs() <= 1e-12 * (1.0 + x2.abs())) {
                    Some(p) => p.2 += w,
                    None => merged.push((0.0, x2, w)),
                }
            }
            merged.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            pts = merged;
        }
        Ok(Fibers { pts, mat, nst: mat.n_state() })
    }

    fn stride(&self) -> usize {
        1 + self.nst
    }

    /// Tillståndets längd per integrationspunkt: generaliserade töjningar och
    /// (spänning, materialtillstånd) per sektionspunkt.
    pub fn ip_len(&self) -> usize {
        NGEN + self.pts.len() * self.stride()
    }

    pub fn init_state(&self, nip: usize) -> Vec<f64> {
        let mut st = vec![0.0; nip * self.ip_len()];
        for q in 0..nip {
            for k in 0..self.pts.len() {
                let o = q * self.ip_len() + NGEN + k * self.stride() + 1;
                self.mat.init_state(&mut st[o..o + self.nst]);
            }
        }
        st
    }

    /// Snittkrafter [N, M1, M2, T, V1, V2] och tangent (6×6) för de
    /// generaliserade töjningarna `e` = [ε, κ1, κ2, τ, γ1, γ2]. Raderna och
    /// kolumnerna för vridning och skjuvning tas ur den elastiska matrisen `c`.
    /// `n` är formfunktionerna i integrationspunkten (för temperaturen).
    pub fn response(&self, ctx: &ElemCtx, c: &[[f64; 6]; 6], e: &[f64; 6], n: &[f64], so: &[f64], sn: &mut [f64]) -> Result<([f64; 6], [[f64; 6]; 6])> {
        let st = self.stride();
        let e_old: [f64; 6] = std::array::from_fn(|i| so[i]);
        sn[..NGEN].copy_from_slice(e);
        let (t, t0) = temperature(ctx, n);
        if ctx.temp.iter().skip(ctx.x.len()).any(|&g| g != 0.0) {
            return fail(format!("element {}: temperaturgradienter i balkar stöds inte ännu", ctx.label));
        }
        let mut s = [0.0; 6];
        let mut k = [[0.0; 6]; 6];
        for (q, &(x1, x2, w)) in self.pts.iter().enumerate() {
            let g = [1.0, x2, -x1];
            let ef = e[0] + x2 * e[1] - x1 * e[2];
            let ef0 = e_old[0] + x2 * e_old[1] - x1 * e_old[2];
            let o = NGEN + q * st;
            let inp = MatInput {
                mode: StressMode::Uniaxial,
                strain: &[ef],
                dstrain: &[ef - ef0],
                stress_old: &so[o..o + 1],
                time: ctx.time,
                dt: ctx.dt,
                temperature: t,
                dtemp: t - t0,
                defgrad: None,
            };
            let r = self.mat.update(&inp, &so[o + 1..o + st], &mut sn[o + 1..o + st]).map_err(|err| crate::model::FemError(format!("element {} sektionspunkt {}: {}", ctx.label, q + 1, err.0)))?;
            sn[o] = r.stress[0];
            for i in 0..3 {
                s[i] += w * g[i] * r.stress[0];
                for j in 0..3 {
                    k[i][j] += w * g[i] * g[j] * r.tangent[0];
                }
            }
        }
        // vridning och skjuvning elastiskt
        for i in 3..6 {
            s[i] = (0..6).map(|j| c[i][j] * e[j]).sum();
            for j in 0..6 {
                k[i][j] = c[i][j];
                k[j][i] = c[j][i];
            }
        }
        Ok((s, k))
    }

    /// Snittkrafter ur lagrade spänningar och töjningar.
    pub fn forces_from_state(&self, c: &[[f64; 6]; 6], ipst: &[f64]) -> [f64; 6] {
        let mut s = [0.0; 6];
        for (q, &(x1, x2, w)) in self.pts.iter().enumerate() {
            let sg = ipst[NGEN + q * self.stride()];
            s[0] += w * sg;
            s[1] += w * x2 * sg;
            s[2] -= w * x1 * sg;
        }
        for i in 3..6 {
            s[i] = (0..6).map(|j| c[i][j] * ipst[j]).sum();
        }
        s
    }

    /// Utdata i sektionspunkterna: S11 och materialets skalära fält (PEEQ).
    pub fn point_output(&self, states: &[&[f64]]) -> Vec<IpField> {
        let mut s = IpField::new("S", &["S11"]);
        let all: Vec<(&'static str, usize)> = self.mat.output_fields(StressMode::Uniaxial).into_iter().map(|(n, l)| (n, l.len())).collect();
        let mut extra: Vec<(usize, IpField)> = Vec::new();
        let mut pos = 0;
        for (n, len) in &all {
            if *len == 1 {
                extra.push((pos, IpField::new(n, &[n])));
            }
            pos += len;
        }
        for (q, st) in states.iter().enumerate() {
            for k in 0..self.pts.len() {
                let o = NGEN + k * self.stride();
                s.points.push((q + 1, k + 1, vec![st[o]]));
                if !extra.is_empty() {
                    let vals = self.mat.output_field_values(StressMode::Uniaxial, &st[o + 1..o + self.stride()]);
                    for (p, f) in extra.iter_mut() {
                        f.points.push((q + 1, k + 1, vec![vals.get(*p).copied().unwrap_or(0.0)]));
                    }
                }
            }
        }
        let mut out = vec![s];
        out.extend(extra.into_iter().map(|(_, f)| f));
        out
    }
}

/// Temperaturen i integrationspunkten vid inkrementets slut och början
/// (nodtemperaturerna är de första värdena i `ctx.temp`): elementets
/// medelvärde för B21/B31, interpolerad med formfunktionerna `n` för B22/B32
/// (kontrollerat med en linjärt varierande temperatur).
fn temperature(ctx: &ElemCtx, n: &[f64]) -> (f64, f64) {
    let nn = ctx.x.len();
    if ctx.temp.len() < nn || ctx.temp_old.len() < nn {
        return (0.0, 0.0);
    }
    if nn == 3 && n.len() == 3 {
        let f = |v: &[f64]| (0..3).map(|i| n[i] * v[i]).sum::<f64>();
        return (f(&ctx.temp), f(&ctx.temp_old));
    }
    let m = nn as f64;
    (ctx.temp[..nn].iter().sum::<f64>() / m, ctx.temp_old[..nn].iter().sum::<f64>() / m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sektionspunkterna ger samma ytmoment som tvärsnittsegenskaperna.
    #[test]
    fn points_reproduce_section_moments() {
        for (shape, v) in [
            ("RECT", vec![0.1, 0.2]),
            ("CIRC", vec![0.1]),
            ("PIPE", vec![0.1, 0.01]),
            ("BOX", vec![0.1, 0.2, 0.01, 0.02, 0.012, 0.015]),
            ("I", vec![0.1, 0.2, 0.1, 0.12, 0.01, 0.015, 0.008]),
        ] {
            let pts = section_points(shape, &v).unwrap();
            let p = super::super::section::shape_props(shape, &v).unwrap();
            let sum = |f: &dyn Fn(f64, f64) -> f64| pts.iter().map(|&(x1, x2, w)| w * f(x1, x2)).sum::<f64>();
            let rel = |a: f64, b: f64, s: f64| (a - b).abs() / s;
            assert!(rel(sum(&|_, _| 1.0), p.a, p.a) < 1e-12, "{shape} A");
            let ir = p.i11.max(p.i22);
            assert!(rel(sum(&|_, x2| x2 * x2), p.i11, ir) < 1e-12, "{shape} I11 {} {}", sum(&|_, x2| x2 * x2), p.i11);
            assert!(rel(sum(&|x1, _| x1 * x1), p.i22, ir) < 1e-12, "{shape} I22");
            assert!(rel(sum(&|x1, x2| x1 * x2), p.i12, ir) < 1e-12, "{shape} I12");
            let sr = p.a * (p.i11 / p.a).sqrt();
            assert!(rel(sum(&|_, x2| x2), p.s1, sr) < 1e-12, "{shape} S1");
            assert!(rel(sum(&|x1, _| x1), p.s2, sr) < 1e-12, "{shape} S2");
        }
    }
}
