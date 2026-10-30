//! Contact and thermal contact (*GAP CONDUCTANCE, *GAP RADIATION) in
//! *COUPLED TEMPERATURE-DISPLACEMENT (included with `#[path]` from
//! `heat_coupled.rs`).
//!
//! With contact the heat conduction depends on the displacements (the gap distance d in
//! the gap conductance k(d)) and the equilibrium on the temperatures (thermal strain),
//! so the tangent is no longer block triangular. Each increment is then solved with
//! block Gauss–Seidel: the heat conduction with the displacements from the previous
//! pass, the equilibrium (with mechanical contact, `heat_stress.rs`) with the new
//! temperatures, repeated from the start of the increment until temperatures and
//! displacements no longer change (relative 10⁻⁹); the displacements passed to
//! the heat conduction are under-relaxed when the passes do not contract. The converged solution is
//! the fully coupled one (the whole system solved simultaneously with Newton).

use super::super::heat::{HeatSystem, TS};
use super::super::heat_stress::ThermoSystem;
use super::super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::fem::dof::Field;
use super::{solve_increment, temps};
use crate::fem::dof::NodeVec;
use crate::model::Result;

/// Maximum number of passes per increment.
const MAX_PASSES: usize = 200;
/// Relative tolerance for the change between two passes.
const TOL: f64 = 1e-9;

/// Solves the increment [t0, t1] for both subsystems; returns the number of
/// iterations. With heat from plastic dissipation (area coupled2):
/// `plastic` = (η per element, increment length in time). The heat source
/// η σ:Δεᵖˡ/Δt is computed in each pass from the equilibrium of the previous pass
/// (`materials::inelastic_heat::nodal_heat`), so that the converged
/// solution is the fully coupled backward Euler solution.
pub(super) fn solve_coupled_with(heat: &mut HeatSystem, mech: &mut ThermoSystem, ctrl: &Controls, t0: f64, t1: f64, verbose: bool, plastic: Option<(&[f64], f64)>) -> Result<usize> {
    let h0 = heat.u_committed.clone();
    let m0 = mech.u_committed.clone();
    let s0 = mech.state_old.clone();
    let before = temps(&h0);
    let mut disp: Vec<NodeVec> = m0.clone();
    let mut t_prev: Option<Vec<f64>> = None;
    let mut iterations = 0;
    let mut omega = 1.0f64;
    let mut du_prev = f64::INFINITY;
    for pass in 0..MAX_PASSES {
        // plastic heat from the previous pass's equilibrium
        if let Some((eta, dt)) = plastic {
            heat.source = if pass == 0 || dt <= 0.0 {
                Vec::new()
            } else {
                crate::materials::inelastic_heat::nodal_heat(heat.p, &s0, &mech.state_old, eta)?.into_iter().map(|q| q / dt).collect()
            };
        }
        // heat conduction with the displacements from the previous pass
        heat.u_committed.clone_from(&h0);
        for (h, d) in heat.u_committed.iter_mut().zip(&disp) {
            for s in 0..TS {
                h[s] = d[s];
            }
        }
        iterations += solve_increment(heat, ctrl, t0, t1)?;
        let tn = temps(&heat.u_committed);
        // equilibrium with the new temperatures
        mech.u_committed.clone_from(&m0);
        mech.state_old.clone_from(&s0);
        iterations += solve_mech(mech, ctrl, t0, t1, &before, &tn)?;
        let un = mech.u_committed.clone();
        let tscale = tn.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let uscale = un.iter().flat_map(|v| v[..TS].iter()).fold(1e-30f64, |m, v| m.max(v.abs()));
        let dt = t_prev.as_ref().map_or(f64::INFINITY, |tp| tp.iter().zip(&tn).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())));
        let du = disp.iter().zip(&un).flat_map(|(a, b)| (0..TS).map(move |s| (a[s] - b[s]).abs())).fold(0.0f64, f64::max);
        // under-relaxation when the passes do not contract (the contact opens and
        // closes, the gap conductance changes rapidly with d)
        if pass > 0 && du > 0.7 * du_prev {
            omega = (0.5 * omega).max(0.02);
        }
        if std::env::var_os("RUSTYFEM_KOPPLAD_DEBUG").is_some() {
            eprintln!("    pass {pass}: dT = {dt:.3e}, du = {du:.3e}, ω = {omega}");
        }
        du_prev = du;
        for (d, u) in disp.iter_mut().zip(&un) {
            for s in 0..TS {
                d[s] += omega * (u[s] - d[s]);
            }
        }
        t_prev = Some(tn);
        // after many passes: accept the solution when the changes are at
        // the accuracy of the equilibrium iterations (contact status near closure)
        let done = (pass > 0 && dt <= TOL * tscale && du <= TOL * uscale) || (pass >= 30 && dt <= 1e-6 * tscale && du <= 1e-4 * uscale);
        let done = done && (plastic.is_none() || pass > 1);
        if done {
            if verbose {
                eprintln!("  coupled contact: {} passes", pass + 1);
            }
            // the heat conduction's displacements = the final ones (for output of the gap flux)
            for (h, d) in heat.u_committed.iter_mut().zip(&mech.u_committed) {
                for s in 0..TS {
                    h[s] = d[s];
                }
            }
            return Ok(iterations);
        }
    }
    eprintln!("warning: coupled temperature–displacement with contact: the subsystems did not converge within {MAX_PASSES} passes");
    Ok(iterations)
}

/// The equilibrium over [t0, t1] with automatic subdivision of the increment (contact
/// that closes may require smaller steps); the temperatures are interpolated linearly
/// between the start `ta` and end `tb` of the increment.
fn solve_mech(mech: &mut ThermoSystem, ctrl: &Controls, t0: f64, t1: f64, ta: &[f64], tb: &[f64]) -> Result<usize> {
    let sub = Controls { initial: 1.0, fixed: false, min: 1e-6, max: 1.0, max_increments: 1000, ..ctrl.clone() };
    let mut span = Span { sys: mech, t0, t1, ta, tb };
    let rep = newton::run(&mut span, &sub, &mut |_, _| Ok(()))?;
    Ok(rep.iterations)
}

struct Span<'a, 'b, 'p, 'm> {
    sys: &'a mut ThermoSystem<'p, 'm>,
    t0: f64,
    t1: f64,
    ta: &'b [f64],
    tb: &'b [f64],
}

impl Span<'_, '_, '_, '_> {
    fn temps_at(&self, a: f64) -> Vec<f64> {
        self.ta.iter().zip(self.tb).map(|(x, y)| x + a * (y - x)).collect()
    }
}

impl NonlinearSystem for Span<'_, '_, '_, '_> {
    fn n_eq(&self) -> usize {
        self.sys.n_eq()
    }
    fn eq_field(&self, eq: usize) -> Field {
        self.sys.eq_field(eq)
    }
    fn is_linear(&self) -> bool {
        self.sys.is_linear()
    }
    fn begin_increment(&mut self, a: f64, b: f64) -> Result<()> {
        self.sys.coupled = (self.temps_at(a), self.temps_at(b));
        let h = self.t1 - self.t0;
        self.sys.begin_increment(self.t0 + a * h, self.t0 + b * h)
    }
    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        self.sys.evaluate(tangent)
    }
    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        self.sys.solve(r)
    }
    fn apply(&mut self, c: &[f64]) {
        self.sys.apply(c)
    }
    fn increment(&self) -> Vec<f64> {
        self.sys.increment()
    }
    fn commit(&mut self) -> Result<()> {
        self.sys.commit()
    }
}
