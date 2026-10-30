//! Område acoustics2: *EOS COMPACTION (P-α, Hermann 1968, Carroll–Holt 1972)
//! för ett fast material med *EOS, TYPE=USUP eller TABULAR (inkluderas med
//! `#[path]` från `eos.rs`):
//!
//! - Distensionen α = ρ_s/ρ ≥ 1, α₀ = 1/(1 − n₀) i jungfruligt tillstånd.
//!   *DENSITY anger det fasta materialets densitet ρ_s0; det porösa
//!   materialets densitet (massan) är ρ_s0/α₀ (`adjust_density`, anropas när
//!   filen är läst).
//! - Trycket p = p_s(ρ_s, Eₘ)/α med ρ_s = αρ, dvs. det fasta materialets
//!   J_s = α₀ J/α (J relativt det porösa begynnelsetillståndet).
//! - Plastisk kurva: P_pl(α) = p_S − (p_S − p_e) √((α − 1)/(α_e − 1)).
//! - Elastiska kurvor (linjära, Wardlaw m.fl. 1996):
//!   P_el(α, α_min) = P_pl(α_min) + (α − α_min)/A′(α_min) med Hermanns lutning
//!   A′(α) = α²/K₀ (1 − 1/h²(α)), h(α) = 1 + (c_e − c_s)(α − 1)/(c_s(α₀ − 1)),
//!   K₀ = ρ_s0 c_s² (c_s = c₀ för USUP, √(K₀/ρ_s0) för TABULAR).
//! - α_e (elastiska gränsen) bestäms så att den jungfruliga elastiska kurvan
//!   går genom (p = 0, α₀): α_e − p_e A′(α_e) = α₀.
//! - α_min (minsta α under plastisk kompaktering) är ett tillstånd; vid
//!   p ≥ p_S är materialet helt kompakterat (α = 1 för alltid).
//!
//! I varje inkrement löses α så att EOS-trycket (med energiekvationen över
//! inkrementet, som i `eos.rs`) och P(α, α_min) sammanfaller (bisektion).

use super::*;

/// Platser efter det fasta materialets tillstånd.
const ALPHA: usize = 0;
const ALPHA_MIN: usize = 1;

#[derive(Debug, Clone)]
pub struct Compaction {
    solid: Eos,
    ce: f64,
    alpha0: f64,
    alphae: f64,
    pe: f64,
    ps: f64,
    k0: f64,
    cs: f64,
    /// Det porösa materialets referensdensitet ρ_s0/α₀.
    rho_p: f64,
}

/// α₀ ur *EOS COMPACTION (n₀ på dataraden).
pub fn alpha0_of(kw: &RawKeyword) -> Result<f64> {
    let v = first(kw)?;
    let n0 = v.get(1).copied().unwrap_or(0.0);
    if !(0.0..1.0).contains(&n0) {
        return kw.err("*EOS COMPACTION: porositeten n₀ ska ligga i [0, 1)");
    }
    Ok(1.0 / (1.0 - n0))
}

/// Det porösa materialets densitet ρ_s0/α₀ för material med *EOS COMPACTION
/// (anropas från `FeModel::finalize`; *DENSITY anger den fasta fasen).
pub fn adjust_density(m: &mut FeModel) -> Result<()> {
    for mat in m.materials.iter_mut() {
        let Some(kw) = mat.behavior("EOS COMPACTION") else { continue };
        let a0 = alpha0_of(kw)?;
        if let Some(d) = mat.behavior("DENSITY") {
            let rho = first(d)?.first().copied().unwrap_or(0.0);
            mat.density = rho / a0;
        }
    }
    Ok(())
}

pub fn build(solid: Eos, kw: &RawKeyword) -> Result<Compaction> {
    let v = first(kw)?;
    if v.len() < 4 {
        return kw.err("*EOS COMPACTION: ange c_e, n₀, p_e och p_S");
    }
    let (ce, pe, ps) = (v[0], v[2], v[3]);
    let alpha0 = alpha0_of(kw)?;
    if !(ce > 0.0) || !(pe > 0.0) || !(ps > pe) {
        return kw.err("*EOS COMPACTION: c_e > 0 och 0 < p_e < p_S krävs");
    }
    if !matches!(solid.form, Form::UsUp { .. } | Form::Tabular(_)) {
        return kw.err("*EOS COMPACTION kräver *EOS, TYPE=USUP eller TYPE=TABULAR");
    }
    let k0 = solid.bulk();
    let cs = match solid.form {
        Form::UsUp { c0, .. } => c0,
        _ => (k0 / solid.rho0).sqrt(),
    };
    let mut c = Compaction { solid, ce, alpha0, alphae: alpha0, pe, ps, k0, cs, rho_p: 0.0 };
    c.rho_p = c.solid.rho0 / alpha0;
    // α_e − p_e A′(α_e) = α₀ (fixpunkt)
    if alpha0 > 1.0 {
        let mut a = alpha0;
        for _ in 0..200 {
            let an = alpha0 + pe * c.slope(a);
            if (an - a).abs() < 1e-15 * a {
                a = an;
                break;
            }
            a = an.max(1.0);
        }
        c.alphae = a.max(1.0);
    }
    Ok(c)
}

impl Compaction {
    fn h(&self, a: f64) -> f64 {
        if self.alpha0 <= 1.0 {
            return 1.0;
        }
        1.0 + (self.ce - self.cs) * (a - 1.0) / (self.cs * (self.alpha0 - 1.0))
    }
    /// Hermanns lutning dA/dp (negativ för c_e < c_s).
    fn slope(&self, a: f64) -> f64 {
        let h = self.h(a);
        a * a / self.k0 * (1.0 - 1.0 / (h * h))
    }
    fn p_plastic(&self, a: f64) -> f64 {
        if self.alphae <= 1.0 {
            return self.ps;
        }
        self.ps - (self.ps - self.pe) * ((a - 1.0).max(0.0) / (self.alphae - 1.0)).sqrt()
    }
    /// P(α, α_min).
    fn p_curve(&self, a: f64, amin: f64) -> f64 {
        if a <= amin {
            self.p_plastic(a)
        } else {
            let s = self.slope(amin);
            if s.abs() < 1e-300 {
                self.p_plastic(amin)
            } else {
                self.p_plastic(amin) + (a - amin) / s
            }
        }
    }
}

impl MaterialModel for Compaction {
    fn name(&self) -> &str {
        "*EOS COMPACTION (P-α)"
    }
    fn is_linear(&self) -> bool {
        false
    }
    fn n_state(&self) -> usize {
        self.solid.n_state() + 2
    }
    fn current_bulk_modulus(&self, state: &[f64]) -> Option<f64> {
        let ns = self.solid.n_state();
        let a = state[ns + ALPHA];
        if a <= 1.0 + 1e-12 {
            // helt kompakterat: det fasta materialets modul
            let mut s = state[..ns].to_vec();
            s[JAC] = self.alpha0 * state[JAC];
            return self.solid.current_bulk_modulus(&s);
        }
        // elastisk/plastisk gren: minst den jungfruliga ρ c_e²
        Some(self.rho_p * self.ce * self.ce)
    }
    fn explicit_full_integration(&self) -> bool {
        true
    }
    fn init_state(&self, state: &mut [f64]) {
        let ns = self.solid.n_state();
        self.solid.init_state(&mut state[..ns]);
        state[ns + ALPHA] = self.alpha0;
        state[ns + ALPHA_MIN] = self.alphae;
    }
    fn tangent_linear(&self, mode: StressMode) -> Result<Vec<f64>> {
        let n = mode.ncomp();
        let k = self.rho_p * self.ce * self.ce;
        let g = self.solid.g;
        let mut d = vec![0.0; n * n];
        for i in 0..3.min(n) {
            for j in 0..3.min(n) {
                d[i * n + j] = k - 2.0 * g / 3.0;
            }
            d[i * n + i] += 2.0 * g;
        }
        for i in 3..n {
            d[i * n + i] = g;
        }
        Ok(d)
    }
    fn init_energy(&self, state: &mut [f64], em: Option<f64>, temperature: Option<f64>, abs_zero: f64) -> Result<Option<(f64, f64)>> {
        let ns = self.solid.n_state();
        self.solid.init_energy(&mut state[..ns], em, temperature, abs_zero)
    }
    fn output_names(&self) -> &[&'static str] {
        &[]
    }
    fn output_values(&self, _state: &[f64]) -> Vec<f64> {
        Vec::new()
    }

    fn update(&self, input: &MatInput, state_old: &[f64], state_new: &mut [f64]) -> Result<MatOutput> {
        let mode = input.mode;
        if !matches!(mode, StressMode::ThreeD | StressMode::PlaneStrain) {
            return fail("*EOS COMPACTION stöds bara i 3D och plan töjning");
        }
        let ns = self.solid.n_state();
        let n = mode.ncomp();
        let j1 = match input.defgrad {
            Some(f) => tensor::det(&f),
            None => 1.0 + input.strain[..3].iter().sum::<f64>(),
        };
        let (em0, j0) = (state_old[EM], state_old[JAC]);
        let amin0 = state_old[ns + ALPHA_MIN];
        let p0 = -input.stress_old[..3].iter().sum::<f64>() / 3.0;
        let tr = input.dstrain[..3].iter().sum::<f64>() / 3.0;
        let g = self.solid.g;
        let dev = |i: usize| if i < 3 { (input.stress_old[i] + p0, input.dstrain[i] - tr) } else { (input.stress_old[i], 0.5 * input.dstrain[i]) };
        let mut s_new = vec![0.0; n];
        let mut wdev = 0.0;
        for (i, s) in s_new.iter_mut().enumerate() {
            let (so, de) = dev(i);
            *s = so + 2.0 * g * de;
            wdev += 0.5 * (so + *s) * de * if i < 3 { 1.0 } else { 2.0 };
        }
        let wdev = wdev * j1 / self.rho_p;
        let dj = j1 - j0;
        // Eₘ = A − ½ p ΔJ/ρ_p
        let a_en = em0 + wdev - 0.5 * p0 * dj / self.rho_p;
        let half = 0.5 * dj / self.rho_p;
        // EOS-trycket vid distensionen α
        let p_eos = |a: f64| -> f64 {
            let js = self.alpha0 * j1 / a;
            let (f, gam) = self.solid.linear_in_energy(js);
            (f + gam * a_en) / (a + gam * half)
        };
        let compacted = amin0 <= 1.0 + 1e-12;
        let (alpha, p1) = if compacted {
            (1.0, p_eos(1.0))
        } else {
            // r(α) = p_eos(α) − P(α, α_min): r(1) < 0 (det fasta materialet
            // sträckt, P = p_S) och r > 0 för stora α; bisektion på [1, α_hi]
            let r = |a: f64| p_eos(a) - self.p_curve(a, amin0);
            let mut lo = 1.0;
            let mut hi = (2.0 * self.alpha0).max(amin0 * 1.5);
            if r(lo) >= 0.0 {
                // trycket når p_S: helt kompakterat
                (1.0, p_eos(1.0))
            } else if r(hi) < 0.0 {
                (hi, self.p_curve(hi, amin0))
            } else {
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if r(mid) < 0.0 {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if hi - lo < 1e-14 * hi {
                        break;
                    }
                }
                let a = 0.5 * (lo + hi);
                (a, self.p_curve(a, amin0))
            }
        };
        let amin = if alpha <= 1.0 + 1e-12 { 1.0 } else { amin0.min(alpha) };
        let em1 = a_en - half * p1;
        let mut stress = s_new;
        for v in stress.iter_mut().take(3) {
            *v -= p1;
        }
        state_new[..ns].copy_from_slice(&state_old[..ns]);
        state_new[EM] = em1;
        state_new[JAC] = j1;
        state_new[PRES] = p1;
        state_new[ns + ALPHA] = alpha;
        state_new[ns + ALPHA_MIN] = amin;
        Ok(MatOutput { stress, tangent: self.tangent_linear(mode)? })
    }
}
