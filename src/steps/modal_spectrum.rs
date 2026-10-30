//! *RESPONSE SPECTRUM: peak values from a response spectrum with modal
//! combination (Clough & Penzien, ch. 26; Chopra, "Dynamics of Structures",
//! ch. 13).
//!
//! For each excitation direction d (direction cosines rᵈ and scale factor sᵈ on
//! the data line) and mode i:
//!
//! ```text
//! Γᵢᵈ = φᵢᵀ M rᵈ                 (φᵢ mass-normalized, rᵈ on the translations)
//! Sd(ωᵢ, ζᵢ) = Sa/ωᵢ², Sv/ωᵢ or Sd from *SPECTRUM (TYPE=ACCELERATION,
//!              VELOCITY, DISPLACEMENT; GRAVITY = Sa·G)
//! xᵢᵈ = sᵈ Γᵢᵈ Sd(ωᵢ, ζᵢ) · xᵢ    (xᵢ = the mode's U and S)
//! ```
//!
//! The spectrum is interpolated log–log in frequency (Hz) within each damping value
//! and with lg S linear in damping (linear if any value is ≤ 0); outside
//! the table the end value is used. The direction vector is not normalized.
//! The modes are summed with `SUM` (SRSS, ABS, CQC, NRL, TENP), default ABS, and the directions
//! (which must be orthogonal) with `COMP` (default SRSS): SRSS combines the directions'
//! modal sums, ALGEBRAIC adds the directions' contributions mode by mode before
//! the modal summation. The CQC correlation is Der Kiureghian's:
//! ρᵢⱼ = 8√(ζᵢζⱼ)(ζᵢ + rζⱼ) r^{3/2} / ((1 − r²)² + 4ζᵢζⱼ r (1 + r²) + 4(ζᵢ² + ζⱼ²) r²),
//! r = ωⱼ/ωᵢ. NRL: |largest| + √(Σ others²). TENP: SRSS plus 2|xᵢxⱼ| for modes
//! whose frequencies differ by at most 10 %. SRSS, ABS, CQC and ALGEBRAIC are
//! validated (tests/cases/dynamics); NRL and TENP are implemented following
//! the literature.
//!
//! Output: frame 0 with zeros and frame 1 with the combined
//! (non-negative) values for each component.

use super::basis::{modal_damping, ModalBasis};
use super::plain_frame;
use crate::fem::model::{RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use crate::steps::api::{StepCtx, StepType};

pub struct ResponseSpectrum;

/// A *SPECTRUM: (damping, [(frequency Hz, value)]) sorted, and type.
struct Spectrum {
    tables: Vec<(f64, Vec<(f64, f64)>)>,
    kind: SpecKind,
}

#[derive(Clone, Copy)]
enum SpecKind {
    Displacement,
    Velocity,
    Acceleration(f64),
}

impl Spectrum {
    fn from_keyword(kw: &RawKeyword) -> Result<Spectrum> {
        let kind = match kw.param_upper("TYPE").as_deref().unwrap_or("ACCELERATION") {
            "DISPLACEMENT" => SpecKind::Displacement,
            "VELOCITY" => SpecKind::Velocity,
            "ACCELERATION" => SpecKind::Acceleration(1.0),
            "GRAVITY" => match kw.param("G").and_then(crate::fem::model::parse_f64) {
                Some(g) => SpecKind::Acceleration(g),
                None => return kw.err("TYPE=GRAVITY requires G="),
            },
            t => return kw.err(format!("TYPE={t} is not supported (DISPLACEMENT, VELOCITY, ACCELERATION, GRAVITY)")),
        };
        if kw.has("INPUT") || kw.has("CREATE") || kw.has("EVENT TYPE") {
            return kw.err("only tabular spectra are supported (INPUT/CREATE is not supported yet)");
        }
        let mut tables: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() < 2 {
                return kw.err_at(d.line, "write 'value, frequency[, damping]'");
            }
            let z = v.get(2).copied().unwrap_or(0.0);
            match tables.iter_mut().find(|t| (t.0 - z).abs() <= 1e-12) {
                Some(t) => t.1.push((v[1], v[0])),
                None => tables.push((z, vec![(v[1], v[0])])),
            }
        }
        if tables.is_empty() {
            return kw.err("the spectrum has no data lines");
        }
        for t in &mut tables {
            t.1.sort_by(|a, b| a.0.total_cmp(&b.0));
        }
        tables.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(Spectrum { tables, kind })
    }

    fn table_value(t: &[(f64, f64)], f: f64) -> f64 {
        if f <= t[0].0 {
            return t[0].1;
        }
        for w in t.windows(2) {
            if f <= w[1].0 {
                // log–log (linear in lg f and lg S),
                // tests/cases/dynamics/cps4_responsspektrum
                if w[0].0 > 0.0 && w[0].1 > 0.0 && w[1].1 > 0.0 {
                    let s = (f / w[0].0).ln() / (w[1].0 / w[0].0).ln();
                    return w[0].1 * (w[1].1 / w[0].1).powf(s);
                }
                let s = (f - w[0].0) / (w[1].0 - w[0].0);
                return w[0].1 + s * (w[1].1 - w[0].1);
            }
        }
        t[t.len() - 1].1
    }

    /// Spectral displacement Sd at ω (rad/time) and damping ratio ζ.
    fn sd(&self, w: f64, zeta: f64) -> f64 {
        let f = w / (2.0 * std::f64::consts::PI);
        let t = &self.tables;
        let val = if zeta <= t[0].0 {
            Self::table_value(&t[0].1, f)
        } else if zeta >= t[t.len() - 1].0 {
            Self::table_value(&t[t.len() - 1].1, f)
        } else {
            let k = t.windows(2).position(|w| zeta <= w[1].0).unwrap();
            let (a, b) = (&t[k], &t[k + 1]);
            let s = (zeta - a.0) / (b.0 - a.0);
            let (va, vb) = (Self::table_value(&a.1, f), Self::table_value(&b.1, f));
            // lg S linear in ζ (tests/cases/dynamics/cps4_responsspektrum_cqc)
            if va > 0.0 && vb > 0.0 {
                va * (vb / va).powf(s)
            } else {
                va + s * (vb - va)
            }
        };
        match self.kind {
            SpecKind::Displacement => val,
            SpecKind::Velocity => val / w,
            SpecKind::Acceleration(g) => val * g / (w * w),
        }
    }
}

/// All values in a frame as a vector (U per slot, then element output).
fn flatten(fr: &FrameData) -> Vec<f64> {
    let mut v: Vec<f64> = fr.u.iter().flat_map(|n| n.iter().copied()).collect();
    for e in &fr.elem {
        for f in e {
            for (_, _, x) in &f.points {
                v.extend_from_slice(x);
            }
        }
    }
    v
}

fn unflatten(fr: &mut FrameData, v: &[f64]) {
    let mut k = 0;
    for n in fr.u.iter_mut() {
        for s in n.iter_mut() {
            *s = v[k];
            k += 1;
        }
    }
    for e in fr.elem.iter_mut() {
        for f in e.iter_mut() {
            for (_, _, x) in f.points.iter_mut() {
                for c in x.iter_mut() {
                    *c = v[k];
                    k += 1;
                }
            }
        }
    }
}

/// Der Kiureghian's correlation coefficient.
fn cqc_rho(wi: f64, wj: f64, zi: f64, zj: f64) -> f64 {
    let r = wj / wi;
    let num = 8.0 * (zi * zj).sqrt() * (zi + r * zj) * r.powf(1.5);
    let den = (1.0 - r * r).powi(2) + 4.0 * zi * zj * r * (1.0 + r * r) + 4.0 * (zi * zi + zj * zj) * r * r;
    if den == 0.0 {
        1.0
    } else {
        num / den
    }
}

/// Combines modal contributions `x[i]` (same component) according to the rule.
fn combine(rule: &str, x: &[f64], omega: &[f64], zeta: &[f64]) -> f64 {
    let n = x.len();
    match rule {
        "ABS" => x.iter().map(|v| v.abs()).sum(),
        "NRL" => {
            let (mut mx, mut s2) = (0.0f64, 0.0);
            for &v in x {
                mx = mx.max(v.abs());
                s2 += v * v;
            }
            mx + (s2 - mx * mx).max(0.0).sqrt()
        }
        "CQC" => {
            let mut s = 0.0;
            for i in 0..n {
                for j in 0..n {
                    s += x[i] * x[j] * if i == j { 1.0 } else { cqc_rho(omega[i], omega[j], zeta[i], zeta[j]) };
                }
            }
            s.max(0.0).sqrt()
        }
        "TENP" => {
            let mut s: f64 = x.iter().map(|v| v * v).sum();
            for i in 0..n {
                for j in i + 1..n {
                    if (omega[j] - omega[i]).abs() <= 0.1 * omega[i].abs().min(omega[j].abs()) {
                        s += 2.0 * (x[i] * x[j]).abs();
                    }
                }
            }
            s.sqrt()
        }
        _ => x.iter().map(|v| v * v).sum::<f64>().sqrt(),
    }
}

impl StepType for ResponseSpectrum {
    fn name(&self) -> &'static str {
        "RESPONSE SPECTRUM"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "RESPONSE SPECTRUM").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let kw = ctx.step.procedure.as_ref().unwrap();
        let comp = kw.param_upper("COMP").unwrap_or_else(|| "SRSS".into());
        // default values (tests/cases/dynamics/cps4_responsspektrum_standard)
        let sum = kw.param_upper("SUM").unwrap_or_else(|| "ABS".into());
        if !matches!(sum.as_str(), "SRSS" | "ABS" | "CQC" | "NRL" | "TENP") {
            return kw.err(format!("SUM={sum} is not supported (SRSS, ABS, CQC, NRL, TENP)"));
        }
        if !matches!(comp.as_str(), "SRSS" | "ALGEBRAIC") {
            return kw.err(format!("COMP={comp} is not supported (SRSS, ALGEBRAIC)"));
        }
        let prep = ctx.prep;
        // excitation directions
        let mut dirs: Vec<(Spectrum, [f64; 3], f64)> = Vec::new();
        for d in &kw.data {
            let name = d.fields.first().map(|s| s.trim().to_uppercase()).unwrap_or_default();
            let sk = prep.model.extra.iter().find(|k| k.name == "SPECTRUM" && k.param_upper("NAME").as_deref() == Some(name.as_str()));
            let Some(sk) = sk else {
                return kw.err_at(d.line, format!("the spectrum {name} is missing (*SPECTRUM, NAME={name})"));
            };
            let num = |i: usize, def: f64| -> Result<f64> {
                match d.fields.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: line {}: invalid number '{s}'", kw.location(), d.line))),
                    None => Ok(def),
                }
            };
            let r = [num(1, 0.0)?, num(2, 0.0)?, num(3, 0.0)?];
            let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            if len == 0.0 {
                return kw.err_at(d.line, "the direction cosines are zero");
            }
            // the direction cosines are not normalized: (1, 1, 0) gives √2 times
            // the spectrum at 45° (tests/cases/dynamics/cps4_responsspektrum_cqc)
            dirs.push((Spectrum::from_keyword(sk)?, r, num(4, 1.0)?));
        }
        for i in 0..dirs.len() {
            for j in i + 1..dirs.len() {
                let (a, b) = (dirs[i].1, dirs[j].1);
                if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs() > 1e-6 * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt() {
                    return kw.err(format!("directions {} and {} are not orthogonal", i + 1, j + 1));
                }
            }
        }
        if dirs.is_empty() {
            return kw.err("give at least one direction: 'spectrum, X cosine, Y cosine, Z cosine, scale factor'");
        }
        let b = ModalBasis::from_frequency_step(ctx)?;
        let omega = b.omega.clone();
        let damp = modal_damping(ctx.step, &b)?;
        let zeta = damp.zeta.clone();
        if damp.structural.iter().any(|s| *s != 0.0) {
            return kw.err("STRUCTURAL modal damping is not supported in *RESPONSE SPECTRUM");
        }
        // modal frames (unscaled) as vectors
        let mut template = None;
        let mut xs = Vec::with_capacity(b.len());
        for phi in &b.phi {
            let fr = plain_frame(ctx, &b.dm, phi, 1.0, -1, "Shock Spectrum Response".into())?;
            xs.push(flatten(&fr));
            template.get_or_insert(fr);
        }
        let Some(mut fr) = template else {
            return fail("*RESPONSE SPECTRUM: no eigenmodes");
        };
        let nv = xs[0].len();
        let nm = xs.len();
        // participation factors per global direction (Γ = φᵀ M r over all DOFs)
        let pf: Vec<Vec<f64>> = (0..3).map(|d| if d < prep.model.dim() { b.participation(&prep.model.nodes, d) } else { vec![0.0; nm] }).collect();
        // modal coefficients per direction: sᵈ Γᵢᵈ Sd(ωᵢ, ζᵢ)
        let mut coefs: Vec<Vec<f64>> = Vec::new();
        for (spec, r, scale) in &dirs {
            coefs.push(
                (0..nm)
                    .map(|i| {
                        let g: f64 = (0..3).map(|d| r[d] * pf[d][i]).sum();
                        if omega[i] > 0.0 {
                            scale * g * spec.sd(omega[i], zeta[i])
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            );
        }
        if comp == "ALGEBRAIC" {
            let c: Vec<f64> = (0..nm).map(|i| coefs.iter().map(|c| c[i]).sum()).collect();
            coefs = vec![c];
        }
        let mut total = vec![0.0; nv];
        let mut contrib = vec![0.0; nm];
        for coef in &coefs {
            for k in 0..nv {
                for i in 0..nm {
                    contrib[i] = coef[i] * xs[i][k];
                }
                let v = combine(&sum, &contrib, &omega, &zeta);
                total[k] += v * v;
            }
        }
        total.iter_mut().for_each(|x| *x = x.sqrt());
        unflatten(&mut fr, &total);
        if ctx.opts.all_frames {
            let mut f0 = fr.clone();
            unflatten(&mut f0, &vec![0.0; nv]);
            f0.value = 0.0;
            ctx.out.frames.push(f0);
        }
        ctx.out.frames.push(fr);
        ctx.out.increments = 1;
        Ok(())
    }
}
