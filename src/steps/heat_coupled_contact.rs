//! Kontakt och termisk kontakt (*GAP CONDUCTANCE, *GAP RADIATION) i
//! *COUPLED TEMPERATURE-DISPLACEMENT (inkluderas med `#[path]` från
//! `heat_coupled.rs`).
//!
//! Med kontakt beror värmeledningen av förskjutningarna (spaltavståndet d i
//! spaltkonduktansen k(d)) och jämvikten av temperaturerna (termisk töjning),
//! så tangenten är inte längre blocktriangulär. Varje inkrement löses då med
//! blockvis Gauss–Seidel: värmeledningen med förskjutningarna från föregående
//! varv, jämvikten (med mekanisk kontakt, `heat_stress.rs`) med de nya
//! temperaturerna, upprepat från inkrementets början tills temperaturer och
//! förskjutningar inte ändras (relativt 10⁻⁹); förskjutningarna till
//! värmeledningen underrelaxeras när varven inte drar ihop sig. Den konvergerade lösningen är
//! den fullt kopplade (hela systemet löst samtidigt med Newton).

use super::super::heat::{HeatSystem, TS};
use super::super::heat_stress::ThermoSystem;
use super::super::newton::{self, Controls, NonlinearSystem, Residual};
use crate::fem::dof::Field;
use super::{solve_increment, temps};
use crate::fem::dof::NodeVec;
use crate::model::Result;

/// Största antal varv per inkrement.
const MAX_PASSES: usize = 200;
/// Relativ tolerans för ändringen mellan två varv.
const TOL: f64 = 1e-9;

/// Löser inkrementet [t0, t1] för båda delsystemen; returnerar antalet
/// iterationer. Med värme från plastisk dissipation (område coupled2):
/// `plastic` = (η per element, inkrementets längd i tid). Värmekällan
/// η σ:Δεᵖˡ/Δt räknas i varje varv ur jämvikten i föregående varv
/// (`materials::inelastic_heat::nodal_heat`), så att den konvergerade
/// lösningen är den fullt kopplade bakåt-Euler-lösningen.
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
        // plastisk värme ur föregående varvs jämvikt
        if let Some((eta, dt)) = plastic {
            heat.source = if pass == 0 || dt <= 0.0 {
                Vec::new()
            } else {
                crate::materials::inelastic_heat::nodal_heat(heat.p, &s0, &mech.state_old, eta)?.into_iter().map(|q| q / dt).collect()
            };
        }
        // värmeledning med förskjutningarna från föregående varv
        heat.u_committed.clone_from(&h0);
        for (h, d) in heat.u_committed.iter_mut().zip(&disp) {
            for s in 0..TS {
                h[s] = d[s];
            }
        }
        iterations += solve_increment(heat, ctrl, t0, t1)?;
        let tn = temps(&heat.u_committed);
        // jämvikt med de nya temperaturerna
        mech.u_committed.clone_from(&m0);
        mech.state_old.clone_from(&s0);
        iterations += solve_mech(mech, ctrl, t0, t1, &before, &tn)?;
        let un = mech.u_committed.clone();
        let tscale = tn.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let uscale = un.iter().flat_map(|v| v[..TS].iter()).fold(1e-30f64, |m, v| m.max(v.abs()));
        let dt = t_prev.as_ref().map_or(f64::INFINITY, |tp| tp.iter().zip(&tn).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())));
        let du = disp.iter().zip(&un).flat_map(|(a, b)| (0..TS).map(move |s| (a[s] - b[s]).abs())).fold(0.0f64, f64::max);
        // underrelaxering när varven inte drar ihop sig (kontakten öppnar och
        // sluter, spaltkonduktansen ändras snabbt med d)
        if pass > 0 && du > 0.7 * du_prev {
            omega = (0.5 * omega).max(0.02);
        }
        if std::env::var_os("RUSTYFEM_KOPPLAD_DEBUG").is_some() {
            eprintln!("    varv {pass}: dT = {dt:.3e}, du = {du:.3e}, ω = {omega}");
        }
        du_prev = du;
        for (d, u) in disp.iter_mut().zip(&un) {
            for s in 0..TS {
                d[s] += omega * (u[s] - d[s]);
            }
        }
        t_prev = Some(tn);
        // efter många varv: godta lösningen när ändringarna ligger på
        // jämviktsiterationernas noggrannhet (kontaktstatus nära slutning)
        let done = (pass > 0 && dt <= TOL * tscale && du <= TOL * uscale) || (pass >= 30 && dt <= 1e-6 * tscale && du <= 1e-4 * uscale);
        let done = done && (plastic.is_none() || pass > 1);
        if done {
            if verbose {
                eprintln!("  kopplad kontakt: {} varv", pass + 1);
            }
            // värmeledningens förskjutningar = de slutliga (för utdata av spaltflödet)
            for (h, d) in heat.u_committed.iter_mut().zip(&mech.u_committed) {
                for s in 0..TS {
                    h[s] = d[s];
                }
            }
            return Ok(iterations);
        }
    }
    eprintln!("varning: kopplad temperatur–förskjutning med kontakt: delsystemen konvergerade inte inom {MAX_PASSES} varv");
    Ok(iterations)
}

/// Jämvikten över [t0, t1] med automatisk delning av inkrementet (kontakten
/// som sluts kan kräva mindre steg); temperaturerna interpoleras linjärt
/// mellan inkrementets början `ta` och slut `tb`.
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
