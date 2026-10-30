//! Materialets effektiva moduler i explicit dynamik:
//! dilatationsvåghastigheten för volymviskositeten och stabilitetsgränsen
//! räknas ur spännings- och töjningsinkrementet i varje element,
//!
//! ```text
//! 3K̂ = Δp / Δε_vol,   2μ̂ = Δs : Δe / (Δe : Δe),   c_d = √((K̂ + 4μ̂/3)/ρ)
//! ```
//!
//! (Δs, Δe = deviatoriska delar av elementets medelspänningsökning och
//! töjningsinkrementet Δε = sym Σᵢ bᵢ ⊗ Δuᵢ / V). K̂ räknas bara när
//! |Δε_vol| är minst 1e-3 av töjningsinkrementet (annars den elastiska K) och
//! μ̂ begränsas till [0, μ]. De elastiska K och μ tas ur materialets linjära
//! tangent. Linjära material behåller den elastiska våghastigheten (kvot 1).
//!
//! Kvoten används i volymviskositetens linjära term och i stabilitetsgränsen.
//! Plasticitet sänker då våghastigheten och förlänger Δt; för *EOS (Us–Up)
//! ger K̂ den styvnande Hugoniot-kurvans kortare Δt.

use super::dt::geometry;
use crate::elements::common::ip_stride;
use crate::elements::Family;
use crate::fem::assembly::Prepared;
use crate::fem::dof::NodeVec;
use crate::materials::StressMode;

fn mode_of(family: Family, name: &str) -> Option<StressMode> {
    match family {
        Family::Continuum3D => Some(StressMode::ThreeD),
        Family::Continuum2D if name.starts_with("CPS") => Some(StressMode::PlaneStress),
        Family::Continuum2D if name.starts_with("CPE") => Some(StressMode::PlaneStrain),
        _ => None,
    }
}

/// Uppdaterar `ratio[e]` = c_d,eff/c_d för olinjära kontinuumelement efter ett
/// inkrement med förskjutningsökningen `du`, i konfigurationen `x(e)`.
pub fn update(p: &Prepared, x: &dyn Fn(usize) -> Vec<[f64; 3]>, du: &[NodeVec], old: &[Vec<f64>], new: &[Vec<f64>], ratio: &mut [f64]) {
    let m = p.model;
    let mut cache: Vec<(usize, (f64, f64))> = Vec::new();
    for (e, el) in m.elements.iter().enumerate() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let Some(mat) = ctx.material else { continue };
        // område coupled2: termisk töjning ovanpå ett linjärt material ändrar
        // inte våghastigheten (Δε innehåller den termiska delen)
        if mat.is_linear() || mat.is_linear_at_given_temperature() {
            continue;
        }
        let fam = el.etype.family();
        let Some(mode) = mode_of(fam, el.etype.name()) else { continue };
        let key = mat as *const dyn crate::materials::MaterialModel as *const () as usize;
        let (kb, mu) = match cache.iter().find(|c| c.0 == key) {
            Some(c) => c.1,
            None => {
                let d = match mat.tangent_linear(StressMode::ThreeD) {
                    Ok(d) if d.len() == 36 => d,
                    _ => continue,
                };
                let mu = 0.5 * (d[0] - d[1]);
                let v = (d[1] + 2.0 * mu / 3.0, mu);
                cache.push((key, v));
                v
            }
        };
        if !(kb + 4.0 * mu / 3.0 > 0.0) {
            continue;
        }
        let (so, sn) = (&old[e], &new[e]);
        let nc = mode.ncomp();
        let stride = ip_stride(mode, mat);
        let nips = el.etype.n_ips();
        if stride == 0 || so.len() < nips * stride || sn.len() < nips * stride {
            continue;
        }
        let xe = x(e);
        let Some(g) = geometry(fam, el.etype.name(), &xe) else { continue };
        // töjningsinkrement (tensor) ur bᵢ
        let mut de = [[0.0f64; 3]; 3];
        for (k, &n) in el.nodes.iter().enumerate() {
            for i in 0..3 {
                for j in 0..3 {
                    de[i][j] += 0.5 * (g.b[k][i] * du[n][j] + g.b[k][j] * du[n][i]) / g.vol;
                }
            }
        }
        // medelspänningsökning
        let mut ds = [[0.0f64; 3]; 3];
        for q in 0..nips {
            let d: Vec<f64> = (0..nc).map(|c| (sn[q * stride + c] - so[q * stride + c]) / nips as f64).collect();
            match mode {
                StressMode::ThreeD => {
                    ds[0][0] += d[0];
                    ds[1][1] += d[1];
                    ds[2][2] += d[2];
                    ds[0][1] += d[3];
                    ds[0][2] += d[4];
                    ds[1][2] += d[5];
                }
                StressMode::PlaneStrain => {
                    ds[0][0] += d[0];
                    ds[1][1] += d[1];
                    ds[2][2] += d[2];
                    ds[0][1] += d[3];
                }
                _ => {
                    ds[0][0] += d[0];
                    ds[1][1] += d[1];
                    ds[0][1] += d[2];
                }
            }
        }
        ds[1][0] = ds[0][1];
        ds[2][0] = ds[0][2];
        ds[2][1] = ds[1][2];
        if mode == StressMode::PlaneStress {
            // tjocklekstöjningen ur plan spänning (elastiskt)
            let lam = kb - 2.0 * mu / 3.0;
            de[2][2] = -lam / (lam + 2.0 * mu) * (de[0][0] + de[1][1]);
        }
        let tr_e = (de[0][0] + de[1][1] + de[2][2]) / 3.0;
        let tr_s = (ds[0][0] + ds[1][1] + ds[2][2]) / 3.0;
        let (mut ee, mut es) = (0.0, 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let dei = de[i][j] - if i == j { tr_e } else { 0.0 };
                let dsi = ds[i][j] - if i == j { tr_s } else { 0.0 };
                ee += dei * dei;
                es += dsi * dei;
            }
        }
        let scale = de.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
        // kompressionsmodulen: materialets aktuella (t.ex. *EOS) som medelvärde
        // över integrationspunkterna, annars den elastiska (Δp/Δε_vol gav brus)
        let kcur: Vec<f64> = (0..nips).filter_map(|q| mat.current_bulk_modulus(&sn[q * stride + nc..(q + 1) * stride])).collect();
        let kh = if kcur.len() == nips && nips > 0 { kcur.iter().sum::<f64>() / nips as f64 } else { kb };
        if scale < 1e-300 {
            // utan töjningsinkrement (stegets början): bara den aktuella kompressionsmodulen
            if kcur.len() == nips && nips > 0 {
                let r = ((kh + 4.0 * mu / 3.0) / (kb + 4.0 * mu / 3.0)).sqrt();
                ratio[e] = if r.is_finite() { r } else { 1.0 };
            }
            continue;
        }
        let _ = tr_s;
        let muh = if ee > 1e-24 * scale * scale { (0.5 * es / ee).clamp(0.0, mu) } else { mu };
        let r = ((kh + 4.0 * muh / 3.0) / (kb + 4.0 * mu / 3.0)).sqrt();
        ratio[e] = if r.is_finite() { r } else { 1.0 };
    }
}
