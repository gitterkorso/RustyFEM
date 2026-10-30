//! *RESPONSE SPECTRUM: toppvärden ur ett responsspektrum med modal
//! kombination (Clough & Penzien, kap. 26; Chopra, "Dynamics of Structures",
//! kap. 13).
//!
//! För varje excitationsriktning d (riktningscosiner rᵈ och skalfaktor sᵈ på
//! dataraden) och mod i:
//!
//! ```text
//! Γᵢᵈ = φᵢᵀ M rᵈ                 (φᵢ massnormerad, rᵈ på translationerna)
//! Sd(ωᵢ, ζᵢ) = Sa/ωᵢ², Sv/ωᵢ eller Sd ur *SPECTRUM (TYPE=ACCELERATION,
//!              VELOCITY, DISPLACEMENT; GRAVITY = Sa·G)
//! xᵢᵈ = sᵈ Γᵢᵈ Sd(ωᵢ, ζᵢ) · xᵢ    (xᵢ = modens U och S)
//! ```
//!
//! Spektret interpoleras log–log i frekvens (Hz) inom varje dämpningsvärde
//! och med lg S linjärt i dämpning (linjärt om något värde är ≤ 0); utanför
//! tabellen används ändvärdet. Riktningsvektorn normeras inte.
//! Moderna summeras med `SUM` (SRSS, ABS, CQC, NRL, TENP), standard ABS, och riktningarna
//! (som måste vara ortogonala) med `COMP` (standard SRSS): SRSS kombinerar riktningarnas
//! modalsummor, ALGEBRAIC adderar riktningarnas bidrag mod för mod före
//! modalsummeringen. CQC-korrelationen är Der Kiureghians:
//! ρᵢⱼ = 8√(ζᵢζⱼ)(ζᵢ + rζⱼ) r^{3/2} / ((1 − r²)² + 4ζᵢζⱼ r (1 + r²) + 4(ζᵢ² + ζⱼ²) r²),
//! r = ωⱼ/ωᵢ. NRL: |största| + √(Σ övriga²). TENP: SRSS plus 2|xᵢxⱼ| för moder
//! vars frekvenser skiljer sig högst 10 %. SRSS, ABS, CQC och ALGEBRAIC är
//! validerade (tests/cases/dynamics); NRL och TENP är implementerade efter
//! litteraturen.
//!
//! Utdata: frame 0 med nollor och frame 1 med de kombinerade
//! (icke-negativa) värdena för varje komponent.

use super::basis::{modal_damping, ModalBasis};
use super::plain_frame;
use crate::fem::model::{RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::model::{fail, FemError, Result};
use crate::steps::api::{StepCtx, StepType};

pub struct ResponseSpectrum;

/// Ett *SPECTRUM: (dämpning, [(frekvens Hz, värde)]) sorterat, och typ.
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
                None => return kw.err("TYPE=GRAVITY kräver G="),
            },
            t => return kw.err(format!("TYPE={t} stöds inte (DISPLACEMENT, VELOCITY, ACCELERATION, GRAVITY)")),
        };
        if kw.has("INPUT") || kw.has("CREATE") || kw.has("EVENT TYPE") {
            return kw.err("bara tabellerade spektra stöds (INPUT/CREATE stöds inte ännu)");
        }
        let mut tables: Vec<(f64, Vec<(f64, f64)>)> = Vec::new();
        for d in &kw.data {
            let v = kw.floats(d)?;
            if v.len() < 2 {
                return kw.err_at(d.line, "skriv 'värde, frekvens[, dämpning]'");
            }
            let z = v.get(2).copied().unwrap_or(0.0);
            match tables.iter_mut().find(|t| (t.0 - z).abs() <= 1e-12) {
                Some(t) => t.1.push((v[1], v[0])),
                None => tables.push((z, vec![(v[1], v[0])])),
            }
        }
        if tables.is_empty() {
            return kw.err("spektret saknar datarader");
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
                // log–log (linjärt i lg f och lg S),
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

    /// Spektralförskjutning Sd vid ω (rad/tid) och dämpningskvot ζ.
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
            // lg S linjärt i ζ (tests/cases/dynamics/cps4_responsspektrum_cqc)
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

/// Alla värden i en frame som en vektor (U per slot, sedan elementutdata).
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

/// Der Kiureghians korrelationskoefficient.
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

/// Kombinerar modala bidrag `x[i]` (samma komponent) enligt regeln.
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
        // standardvärden (tests/cases/dynamics/cps4_responsspektrum_standard)
        let sum = kw.param_upper("SUM").unwrap_or_else(|| "ABS".into());
        if !matches!(sum.as_str(), "SRSS" | "ABS" | "CQC" | "NRL" | "TENP") {
            return kw.err(format!("SUM={sum} stöds inte (SRSS, ABS, CQC, NRL, TENP)"));
        }
        if !matches!(comp.as_str(), "SRSS" | "ALGEBRAIC") {
            return kw.err(format!("COMP={comp} stöds inte (SRSS, ALGEBRAIC)"));
        }
        let prep = ctx.prep;
        // excitationsriktningar
        let mut dirs: Vec<(Spectrum, [f64; 3], f64)> = Vec::new();
        for d in &kw.data {
            let name = d.fields.first().map(|s| s.trim().to_uppercase()).unwrap_or_default();
            let sk = prep.model.extra.iter().find(|k| k.name == "SPECTRUM" && k.param_upper("NAME").as_deref() == Some(name.as_str()));
            let Some(sk) = sk else {
                return kw.err_at(d.line, format!("spektret {name} saknas (*SPECTRUM, NAME={name})"));
            };
            let num = |i: usize, def: f64| -> Result<f64> {
                match d.fields.get(i).map(|s| s.trim()).filter(|s| !s.is_empty()) {
                    Some(s) => crate::fem::model::parse_f64(s).ok_or_else(|| FemError(format!("{}: rad {}: ogiltigt tal '{s}'", kw.location(), d.line))),
                    None => Ok(def),
                }
            };
            let r = [num(1, 0.0)?, num(2, 0.0)?, num(3, 0.0)?];
            let len = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            if len == 0.0 {
                return kw.err_at(d.line, "riktningscosinerna är noll");
            }
            // riktningscosinerna normeras inte: (1, 1, 0) ger √2 gånger
            // spektret i 45° (tests/cases/dynamics/cps4_responsspektrum_cqc)
            dirs.push((Spectrum::from_keyword(sk)?, r, num(4, 1.0)?));
        }
        for i in 0..dirs.len() {
            for j in i + 1..dirs.len() {
                let (a, b) = (dirs[i].1, dirs[j].1);
                if (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]).abs() > 1e-6 * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt() * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt() {
                    return kw.err(format!("riktning {} och {} är inte ortogonala", i + 1, j + 1));
                }
            }
        }
        if dirs.is_empty() {
            return kw.err("ange minst en riktning: 'spektrum, X-cosinus, Y-cosinus, Z-cosinus, skalfaktor'");
        }
        let b = ModalBasis::from_frequency_step(ctx)?;
        let omega = b.omega.clone();
        let damp = modal_damping(ctx.step, &b)?;
        let zeta = damp.zeta.clone();
        if damp.structural.iter().any(|s| *s != 0.0) {
            return kw.err("STRUCTURAL modal dämpning stöds inte i *RESPONSE SPECTRUM");
        }
        // modala frames (oskalade) som vektorer
        let mut template = None;
        let mut xs = Vec::with_capacity(b.len());
        for phi in &b.phi {
            let fr = plain_frame(ctx, &b.dm, phi, 1.0, -1, "Shock Spectrum Response".into())?;
            xs.push(flatten(&fr));
            template.get_or_insert(fr);
        }
        let Some(mut fr) = template else {
            return fail("*RESPONSE SPECTRUM: inga egenmoder");
        };
        let nv = xs[0].len();
        let nm = xs.len();
        // deltagandefaktorer per global riktning (Γ = φᵀ M r över alla frihetsgrader)
        let pf: Vec<Vec<f64>> = (0..3).map(|d| if d < prep.model.dim() { b.participation(&prep.model.nodes, d) } else { vec![0.0; nm] }).collect();
        // modala koefficienter per riktning: sᵈ Γᵢᵈ Sd(ωᵢ, ζᵢ)
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
