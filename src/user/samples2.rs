//! Examples of the user subroutines in `routines.rs` (registered from the start).
//! Each example is selected by a name starting with `EXAMPLE_` and has a
//! built-in counterpart, so that the regression cases in `tests/cases/user2/`
//! can be compared with a model without user subroutines.
//!
//! | Routine | Name | Content |
//! |---|---|---|
//! | DLOAD | elset `EXAMPLE_DLOAD*` | pressure F·(1 + x + y + z) (F = the magnitude on the data line) |
//! | DISP | nset `EXAMPLE_DISP*` | U = v·(1 + x)·step time (v = the value on the data line) |
//! | UAMP | amplitude `EXAMPLE_SINE*` | A = A0 + B sin(ω t), PROPS = A0, ω, B |
//! | USDFLD | material `EXAMPLE_USDFLD*` | FIELD(1) = 0.1·NOEL |
//! | UFIELD | nset `EXAMPLE_UFIELD*` | FIELD(KFIELD) = (1 + x)·total time |
//! | UHARD | material `EXAMPLE_UHARD*` | σy = σ0 + H ε̄p (PROPS = σ0, H) or with three PROPS σy = σ0 + Q(1 − e^(−b ε̄p)) |
//! | SIGINI | elset `EXAMPLE_SIGINI*` | σ11 = −100, σ22 = −200, σ33 = −50 |
//! | UEXPAN | material `EXAMPLE_UEXPAN*` | secant coefficient α(T) = 1e-5 + 1e-8 T (θ0 = 0), isotropic |
//! | CREEP | material `EXAMPLE_CREEP*` | time hardening Δε̄cr = A q̃ⁿ tᵐ Δt with A = 1e-12, n = 3, m = 0 |
//! | HETVAL | material `EXAMPLE_HETVAL*` | heat generation 1000 per volume |
//! | FILM | elset `EXAMPLE_FILM*` | h = 25, ambient temperature 20 |
//! | UMATHT | material `EXAMPLE_UMATHT*` | linear heat conduction, PROPS = k, c (ρ from *DENSITY) |

use super::routines::*;
use crate::model::Result;
use std::sync::Arc;

pub fn dloads() -> Vec<(String, Arc<dyn Dload>)> {
    vec![("EXAMPLE_DLOAD".into(), Arc::new(ExDload))]
}
pub fn disps() -> Vec<(String, Arc<dyn Disp>)> {
    vec![("EXAMPLE_DISP".into(), Arc::new(ExDisp))]
}
pub fn uamps() -> Vec<(String, Arc<dyn Uamp>)> {
    vec![("EXAMPLE_SINE".into(), Arc::new(ExUamp))]
}
pub fn usdflds() -> Vec<(String, Arc<dyn Usdfld>)> {
    vec![("EXAMPLE_USDFLD".into(), Arc::new(ExUsdfld))]
}
pub fn ufields() -> Vec<(String, Arc<dyn Ufield>)> {
    vec![("EXAMPLE_UFIELD".into(), Arc::new(ExUfield))]
}
pub fn uhards() -> Vec<(String, Arc<dyn Uhard>)> {
    vec![("EXAMPLE_UHARD".into(), Arc::new(ExUhard))]
}
pub fn siginis() -> Vec<(String, Arc<dyn Sigini>)> {
    vec![("EXAMPLE_SIGINI".into(), Arc::new(ExSigini))]
}
pub fn uexpans() -> Vec<(String, Arc<dyn Uexpan>)> {
    vec![("EXAMPLE_UEXPAN".into(), Arc::new(ExUexpan))]
}
pub fn creeps() -> Vec<(String, Arc<dyn Creep>)> {
    vec![("EXAMPLE_CREEP".into(), Arc::new(ExCreep))]
}
pub fn hetvals() -> Vec<(String, Arc<dyn Hetval>)> {
    vec![("EXAMPLE_HETVAL".into(), Arc::new(ExHetval))]
}
pub fn films() -> Vec<(String, Arc<dyn Film>)> {
    vec![("EXAMPLE_FILM".into(), Arc::new(ExFilm))]
}
pub fn umathts() -> Vec<(String, Arc<dyn Umatht>)> {
    vec![("EXAMPLE_UMATHT".into(), Arc::new(ExUmatht))]
}

struct ExDload;
impl Dload for ExDload {
    fn dload(&self, a: &mut DloadArgs) -> Result<()> {
        a.f *= 1.0 + a.coords[0] + a.coords[1] + a.coords[2];
        Ok(())
    }
}

struct ExDisp;
impl Disp for ExDisp {
    fn disp(&self, a: &mut DispArgs) -> Result<()> {
        a.u[0] *= (1.0 + a.coords[0]) * a.time[0];
        Ok(())
    }
}

struct ExUamp;
impl Uamp for ExUamp {
    fn uamp(&self, a: &mut UampArgs) -> Result<()> {
        let p = |i: usize| a.props.get(i).copied().unwrap_or(0.0);
        let (a0, w, b) = (p(0), p(1), p(2));
        let t = a.time[0];
        a.amp_value_new = a0 + b * (w * t).sin();
        a.amp_derivative = b * w * (w * t).cos();
        a.amp_sec_derivative = -b * w * w * (w * t).sin();
        Ok(())
    }
}

struct ExUsdfld;
impl Usdfld for ExUsdfld {
    fn usdfld(&self, a: &mut UsdfldArgs) -> Result<()> {
        // constant per element (so that an equivalent model can be run without the routine)
        if let Some(f) = a.field.first_mut() {
            *f = 0.1 * a.noel as f64;
        }
        Ok(())
    }
}

struct ExUfield;
impl Ufield for ExUfield {
    fn ufield(&self, a: &mut UfieldArgs) -> Result<()> {
        if a.kfield >= 1 && a.kfield <= a.field.len() {
            a.field[a.kfield - 1] = (1.0 + a.coords[0]) * a.time[1];
        }
        Ok(())
    }
}

struct ExUhard;
impl Uhard for ExUhard {
    fn uhard(&self, a: &mut UhardArgs) -> Result<()> {
        let p = |i: usize| a.props.get(i).copied().unwrap_or(0.0);
        if a.props.len() >= 3 {
            let (s0, q, b) = (p(0), p(1), p(2));
            let e = (-b * a.eqplas).exp();
            a.syield = s0 + q * (1.0 - e);
            a.hard = [q * b * e, 0.0, 0.0];
        } else {
            a.syield = p(0) + p(1) * a.eqplas;
            a.hard = [p(1), 0.0, 0.0];
        }
        Ok(())
    }
}

struct ExSigini;
impl Sigini for ExSigini {
    fn sigini(&self, a: &mut SiginiArgs) -> Result<()> {
        // constant (the same initial stress can be given with TYPE=STRESS)
        a.sigma.iter_mut().for_each(|s| *s = 0.0);
        a.sigma[0] = -100.0;
        if a.sigma.len() > 1 {
            a.sigma[1] = -200.0;
        }
        if a.ntens >= 4 {
            a.sigma[2] = -50.0;
        }
        Ok(())
    }
}

struct ExUexpan;
impl Uexpan for ExUexpan {
    fn uexpan(&self, a: &mut UexpanArgs) -> Result<()> {
        let alpha = |t: f64| 1e-5 + 1e-8 * t;
        let (t1, dt) = (a.temp[0], a.temp[1]);
        let t0 = t1 - dt;
        let d = alpha(t1) * t1 - alpha(t0) * t0;
        a.expan.iter_mut().for_each(|x| *x = 0.0);
        a.expan[0] = d;
        a.dexpandt[0] = 1e-5 + 2e-8 * t1;
        Ok(())
    }
}

struct ExCreep;
impl Creep for ExCreep {
    fn creep(&self, a: &mut CreepArgs) -> Result<()> {
        let (aa, n) = (1e-12, 3.0);
        a.decra[0] = aa * a.qtild.powf(n) * a.dtime;
        if a.leximp == 1 {
            a.decra[4] = aa * n * a.qtild.powf(n - 1.0) * a.dtime;
        }
        Ok(())
    }
}

struct ExHetval;
impl Hetval for ExHetval {
    fn hetval(&self, a: &mut HetvalArgs) -> Result<()> {
        a.flux = [1000.0, 0.0];
        Ok(())
    }
}

struct ExFilm;
impl Film for ExFilm {
    fn film(&self, a: &mut FilmArgs) -> Result<()> {
        a.h = [25.0, 0.0];
        a.sink = 20.0;
        Ok(())
    }
}

struct ExUmatht;
impl Umatht for ExUmatht {
    fn umatht(&self, a: &mut UmathtArgs) -> Result<()> {
        let k = a.props.first().copied().unwrap_or(0.0);
        let c = a.props.get(1).copied().unwrap_or(0.0);
        a.dudt = c;
        a.u += c * a.dtemp;
        let n = a.ntgrd;
        for i in 0..n {
            a.flux[i] = -k * a.dtemdx[i];
            a.dfdt[i] = 0.0;
            for j in 0..n {
                a.dfdg[i + j * n] = if i == j { -k } else { 0.0 };
            }
        }
        Ok(())
    }
}
