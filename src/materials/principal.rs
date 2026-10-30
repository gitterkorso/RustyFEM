//! Gemensamma byggstenar för tryckberoende plasticitet och skademodeller
//! (`mohr_coulomb.rs`, `concrete_damaged.rs`, `ductile_damage.rs`):
//! huvudspänningar, invarianter, skalär rotsökning och numerisk tangent.
//!
//! Med isotrop elasticitet och isotropa flytytor/flödespotentialer har
//! prövospänningen och den returnerade spänningen samma huvudriktningar, så
//! returalgoritmerna arbetar med de tre huvudspänningarna och bygger sedan
//! tillbaka tensorn med prövospänningens egenvektorer.

use super::plasticity::{M6, V6};
use super::MaterialFactory;
use crate::kinematics::tensor::{eig_sym, M3};
use crate::model::{fail, Result};

/// Modulen har inga egna materialfabriker.
pub static FACTORIES: &[MaterialFactory] = &[];

/// Väljer den osymmetriska lösaren för ett steg om något material kräver det
/// (`MaterialModel::requires_unsymmetric`), t.ex. automatiskt för
/// betong med skadeplasticitet. Gäller inte om lösaren valts uttryckligen
/// (`forced`), om steget anger UNSYMM= eller SOLVER=, eller om en annan än
/// standardlösaren redan valts.
pub fn auto_unsymmetric(
    step: &crate::fem::model::StepDef,
    prep: &crate::fem::assembly::Prepared,
    solver: &'static dyn crate::solver::LinearSolver,
    forced: bool,
) -> &'static dyn crate::solver::LinearSolver {
    if forced || step.header.has("UNSYMM") || step.header.has("SOLVER") || step.procedure.as_ref().is_some_and(|p| p.has("SOLVER")) {
        return solver;
    }
    if solver.name() == "direct" && prep.materials.iter().flatten().any(|m| m.requires_unsymmetric()) {
        return &crate::solver::indefinite::UNSYMMETRIC;
    }
    solver
}

/// Huvudvärden (fallande ordning) och egenvektorer (kolumner) för en
/// spänningslik Voigt-vektor (tensorskjuvkomponenter).
pub fn principal(s: &V6) -> ([f64; 3], M3) {
    let a = [[s[0], s[3], s[4]], [s[3], s[1], s[5]], [s[4], s[5], s[2]]];
    let (l, v) = eig_sym(&a);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| l[j].partial_cmp(&l[i]).unwrap_or(std::cmp::Ordering::Equal));
    let lam = [l[idx[0]], l[idx[1]], l[idx[2]]];
    let mut vs = [[0.0; 3]; 3];
    for (c, &k) in idx.iter().enumerate() {
        for r in 0..3 {
            vs[r][c] = v[r][k];
        }
    }
    (lam, vs)
}

/// Tensor Σ λₖ vₖ⊗vₖ i Voigt-form. `engineering`: skjuvkomponenterna som
/// ingenjörstöjningar (2·A12).
pub fn from_principal(l: &[f64; 3], v: &M3, engineering: bool) -> V6 {
    let mut a = [[0.0; 3]; 3];
    for k in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += l[k] * v[i][k] * v[j][k];
            }
        }
    }
    let f = if engineering { 2.0 } else { 1.0 };
    [a[0][0], a[1][1], a[2][2], f * a[0][1], f * a[0][2], f * a[1][2]]
}

/// Invarianter för huvudspänningar: p = −tr σ/3, q (von Mises) och
/// deviatorvinkeln θ i π-planet, θ = 0 vid enaxligt drag och θ = π/3 vid
/// enaxligt tryck när σ1 ≥ σ2 ≥ σ3 (Θ med cos 3Θ = (r/q)³).
/// Vinkeln räknas globalt (atan2) så att den är kontinuerlig även om
/// ordningen mellan huvudspänningarna byts under en iteration.
pub fn pq_theta(s: &[f64; 3]) -> (f64, f64, f64) {
    let p = -(s[0] + s[1] + s[2]) / 3.0;
    let d = [s[0] + p, s[1] + p, s[2] + p];
    let q = (1.5 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2])).sqrt();
    // π-planets koordinater: x längs (2, −1, −1), y längs (0, 1, −1)
    let x = (2.0 * d[0] - d[1] - d[2]) / 6f64.sqrt();
    let y = (d[1] - d[2]) / 2f64.sqrt();
    let th = if q > 0.0 { y.atan2(x) } else { 0.0 };
    (p, q, th)
}

/// Viker en global π-planvinkel till sextanten [0, π/3]. Returnerar
/// (Θ, dΘ/dθ = ±1).
pub fn fold_theta(th: f64) -> (f64, f64) {
    let t = std::f64::consts::FRAC_PI_3;
    let mut a = th.rem_euclid(2.0 * t);
    let mut sgn = 1.0;
    if a > t {
        a = 2.0 * t - a;
        sgn = -1.0;
    }
    (a, sgn)
}

/// Gradient av en funktion g(p, q, θ) med avseende på huvudspänningarna,
/// givet ∂g/∂p, ∂g/∂q och (1/q)·∂g/∂θ (den senare ändlig även när q → 0).
pub fn grad_pq_theta(th: f64, gp: f64, gq: f64, gth_over_q: f64) -> [f64; 3] {
    // ∂q/∂s = √(3/2) û_ρ, ∂θ/∂s = √(3/2)/q û_θ (deviatoriska enhetsvektorer)
    let ux = [2.0 / 6f64.sqrt(), -1.0 / 6f64.sqrt(), -1.0 / 6f64.sqrt()];
    let uy = [0.0, 1.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt()];
    let (c, s) = (th.cos(), th.sin());
    let k = 1.5f64.sqrt();
    let mut g = [0.0; 3];
    for i in 0..3 {
        let ur = c * ux[i] + s * uy[i];
        let ut = -s * ux[i] + c * uy[i];
        g[i] = -gp / 3.0 + k * (gq * ur + gth_over_q * ut);
    }
    g
}

/// Löser f(x) = 0 för x ≥ 0 där f(0) > 0: inneslutning genom fördubbling från
/// `x0`, sedan Illinois-varianten av regula falsi med Newtonlika sekantsteg.
/// Returnerar den minsta roten som hittas mellan 0 och första teckenbytet.
pub fn root_from_zero(mut f: impl FnMut(f64) -> Result<f64>, x0: f64, ftol: f64, what: &str) -> Result<f64> {
    let f0 = f(0.0)?;
    if f0 <= ftol {
        return Ok(0.0);
    }
    let (mut a, mut fa) = (0.0, f0);
    let mut b = x0.max(1e-300);
    let mut fb = f(b)?;
    let mut n = 0;
    while fb > 0.0 {
        a = b;
        fa = fb;
        b *= 2.0;
        fb = f(b)?;
        n += 1;
        if n > 200 || !fb.is_finite() {
            return fail(format!("{what}: returalgoritmen hittade ingen lösning (ingen inneslutning)"));
        }
    }
    if fb.abs() <= ftol {
        return Ok(b);
    }
    let mut side = 0;
    for _ in 0..300 {
        let x = (a * fb - b * fa) / (fb - fa);
        let fx = f(x)?;
        if fx.abs() <= ftol || (b - a).abs() <= 1e-15 * b.abs().max(1e-300) {
            return Ok(x);
        }
        if fx > 0.0 {
            a = x;
            fa = fx;
            if side == 1 {
                fb *= 0.5;
            }
            side = 1;
        } else {
            b = x;
            fb = fx;
            if side == -1 {
                fa *= 0.5;
            }
            side = -1;
        }
    }
    fail(format!("{what}: returalgoritmen konvergerade inte"))
}

/// Numerisk tangent dσ/dε med framåtdifferenser av en 3D-uppdatering.
pub fn fd_tangent(eps: &V6, sig: &V6, h: f64, mut ret: impl FnMut(&V6) -> Result<V6>) -> Result<M6> {
    let mut c = [[0.0; 6]; 6];
    for j in 0..6 {
        let mut e2 = *eps;
        e2[j] += h;
        let s2 = ret(&e2)?;
        for i in 0..6 {
            c[i][j] = (s2[i] - sig[i]) / h;
        }
    }
    Ok(c)
}

/// Numerisk tangent med centraldifferenser. Används när flytytan har hörn
/// där huvudspänningar sammanfaller (t.ex. σ̂max vid enaxligt tryck i
/// betongmodellen): en framåtdifferens ger då en ensidig derivata som är fel
/// i den symmetriska riktningen, medan centraldifferensen ger medelvärdet av
/// de ensidiga derivatorna och därmed rätt derivata längs symmetrin.
pub fn fd_tangent_central(eps: &V6, h: f64, mut ret: impl FnMut(&V6) -> Result<V6>) -> Result<M6> {
    let mut c = [[0.0; 6]; 6];
    for j in 0..6 {
        let (mut ep, mut em) = (*eps, *eps);
        ep[j] += h;
        em[j] -= h;
        let (sp, sm) = (ret(&ep)?, ret(&em)?);
        for i in 0..6 {
            c[i][j] = (sp[i] - sm[i]) / (2.0 * h);
        }
    }
    Ok(c)
}

/// Styckvis linjär interpolation i en tabell (x, y) med stigande x och
/// konstant värde utanför. Returnerar (y, dy/dx).
pub fn interp(t: &[(f64, f64)], x: f64) -> (f64, f64) {
    if t.len() == 1 || x <= t[0].0 {
        return (t[0].1, 0.0);
    }
    for w in t.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        if x < x1 {
            let h = (y1 - y0) / (x1 - x0);
            return (y0 + h * (x - x0), h);
        }
    }
    (t[t.len() - 1].1, 0.0)
}
