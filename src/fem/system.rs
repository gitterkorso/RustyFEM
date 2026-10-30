//! Kvasistatisk jämvikt R(u) = F_ext(t) − F_int(u) = 0 som `NonlinearSystem`.
//!
//! Används av STATIC (linjärt och materiellt olinjärt) och kan återanvändas
//! av NLGEOM- och kontaktsteg: sätt `nlgeom` och låt elementen/interaktionerna
//! stå för kinematiken.

use crate::fem::assembly::{self, DofMap, Prepared};
use crate::fem::dof::{self, Field, NodeVec, SLOTS, SLOT_DOF};
use crate::fem::state::{SolutionState, StepLoads};
use crate::model::Result;
use crate::solver::{LinearSolver, SystemMatrix};
use crate::steps::newton::{NonlinearSystem, Residual};
use std::collections::BTreeMap;

pub struct StaticSystem<'p, 'm> {
    pub p: &'p Prepared<'m>,
    pub loads: &'p StepLoads,
    pub nlgeom: bool,
    pub dm: DofMap,
    pub mat: Box<dyn SystemMatrix>,
    /// Aktuell lösning och senast godkända.
    pub u: Vec<NodeVec>,
    pub u_committed: Vec<NodeVec>,
    pub state_old: Vec<Vec<f64>>,
    pub state_new: Vec<Vec<f64>>,
    pub fint: Vec<NodeVec>,
    pub fext: Vec<NodeVec>,
    /// Total tid vid stegets början och stegets längd.
    pub time0: f64,
    pub period: f64,
    pub t0: f64,
    pub t1: f64,
    linear: bool,
    /// Obalanserade inre krafter från begynnelsespänningen (*INITIAL
    /// CONDITIONS, TYPE=STRESS) i första steget; läggs till de yttre krafterna
    /// med faktorn (1 − t) (`materials::initial_state`).
    pub unbalance: Option<Vec<NodeVec>>,
}

impl<'p, 'm> StaticSystem<'p, 'm> {
    /// Startar från tillståndet `state` (u och elementtillstånd).
    pub fn new(
        p: &'p Prepared<'m>,
        loads: &'p StepLoads,
        state: &SolutionState,
        nlgeom: bool,
        solver: &dyn LinearSolver,
        period: f64,
    ) -> Result<Self> {
        let (dm, mat) = assembly::dofmap(p, &loads.constrained(), solver)?;
        let nn = p.model.nodes.len();
        Ok(StaticSystem {
            p,
            loads,
            nlgeom,
            dm,
            mat,
            u: state.u.clone(),
            u_committed: state.u.clone(),
            state_old: state.elem_state.clone(),
            state_new: state.elem_state.clone(),
            fint: vec![[0.0; SLOTS]; nn],
            fext: vec![[0.0; SLOTS]; nn],
            time0: state.time,
            period,
            t0: 0.0,
            t1: 0.0,
            linear: p.is_linear() && !nlgeom,
            unbalance: state.vectors.get(crate::materials::initial_state::UNBALANCE).cloned(),
        })
    }

    /// Reaktioner (F_int − F_ext i låsta frihetsgrader) vid senaste utvärderingen.
    pub fn reactions(&self) -> Vec<NodeVec> {
        let mut r = vec![[0.0; SLOTS]; self.u.len()];
        for (n, rn) in r.iter_mut().enumerate() {
            for s in 0..SLOTS {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    rn[s] = self.fint[n][s] - self.fext[n][s];
                }
            }
        }
        r
    }

    /// Skriver godkänd lösning till tillståndet.
    pub fn store(&self, state: &mut SolutionState) {
        state.u = self.u_committed.clone();
        state.elem_state = self.state_old.clone();
        state.reaction = self.reactions();
    }
}

impl StaticSystem<'_, '_> {
    /// Linjär prediktor för ett inkrement med föreskrivna förskjutningar
    /// (olinjära problem): de fria frihetsgraderna får det linjära svaret på
    /// randvillkorsändringen med tangenten vid inkrementets början,
    /// K_ff c = R(u0) − K_fp Δu_p, där K_fp Δu_p tas som en riktningsderivata av
    /// de inre krafterna. Annars utvärderas första iterationen i ett "knäckt"
    /// läge där bara de föreskrivna noderna har flyttats, vilket ger orimliga
    /// töjningar i elementen intill (t.ex. plasticitet långt bortom flytgränsen).
    /// Den konvergerade lösningen påverkas inte.
    fn predict(&mut self) -> Result<()> {
        let target = std::mem::replace(&mut self.u, self.u_committed.clone());
        let r0 = match self.evaluate(true) {
            Ok(r) => r.r,
            Err(_) => {
                self.u = target;
                return Ok(());
            }
        };
        let fint0 = self.fint.clone();
        let h = 1e-6;
        for (n, un) in self.u.iter_mut().enumerate() {
            for s in 0..SLOTS {
                if self.dm.constrained[n] & (1 << s) != 0 {
                    un[s] = self.u_committed[n][s] + h * (target[n][s] - self.u_committed[n][s]);
                }
            }
        }
        self.dm.mpc.expand(&mut self.u);
        let ok = self.evaluate(false).is_ok();
        let mut rhs = r0;
        if ok {
            for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
                rhs[q] -= (self.fint[n][s] - fint0[n][s]) / h;
            }
        }
        self.u = target;
        if !ok {
            return Ok(());
        }
        if self.mat.factor().is_err() {
            return Ok(());
        }
        self.mat.solve(&mut rhs);
        if rhs.iter().all(|x| x.is_finite()) {
            for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
                self.u[n][s] += rhs[q];
            }
            self.dm.mpc.expand(&mut self.u);
        }
        Ok(())
    }
}

impl NonlinearSystem for StaticSystem<'_, '_> {
    fn n_eq(&self) -> usize {
        self.dm.n_eq
    }

    fn eq_field(&self, eq: usize) -> Field {
        dof::field(SLOT_DOF[self.dm.owner[eq].1])
    }

    fn is_linear(&self) -> bool {
        self.linear
    }

    fn begin_increment(&mut self, t0: f64, t1: f64) -> Result<()> {
        self.t0 = t0;
        self.t1 = t1;
        self.u.clone_from(&self.u_committed);
        let mut moved = false;
        for (n, s, v) in self.loads.bc_at(t1) {
            if self.dm.constrained[n] & (1 << s) != 0 {
                moved |= self.u[n][s] != v;
                self.u[n][s] = v;
            }
        }
        self.dm.mpc.expand(&mut self.u);
        // Prediktorn används inte med olinjära interaktioner (kontakt): den
        // linjära extrapolationen med tangenten från inkrementets början kan
        // öppna kontakten så att kroppar som bara hålls av kontakt blir fria.
        let contact = self.p.interactions.iter().any(|i| !i.is_linear());
        if moved && !self.linear && !contact && self.dm.n_eq > 0 {
            self.predict()?;
        }
        Ok(())
    }

    fn evaluate(&mut self, tangent: bool) -> Result<Residual> {
        let du: Vec<NodeVec> = self.u.iter().zip(&self.u_committed).map(|(a, b)| std::array::from_fn(|s| a[s] - b[s])).collect();
        let time = self.time0 + self.t1 * self.period;
        let dt = (self.t1 - self.t0) * self.period;
        let k: Option<&mut dyn SystemMatrix> = if tangent { Some(&mut *self.mat as &mut dyn SystemMatrix) } else { None };
        let t0 = std::time::Instant::now();
        let int = assembly::internal(self.p, &self.dm, &self.u, &du, &self.state_old, self.nlgeom, time, dt, k)?;
        if crate::settings::with(|s| s.timing) {
            eprintln!("inre krafter{}: {:.3} s", if tangent { " och tangent" } else { "" }, t0.elapsed().as_secs_f64());
        }
        self.fint = int.fint;
        self.state_new = int.state_new;
        self.fext = assembly::external(self.p, &self.loads.cloads_at(self.t1), &self.loads.dloads_at(self.t1), &self.u, self.nlgeom, time)?;
        // område geomaterials: begynnelsespänningens obalans avvecklas linjärt under första steget
        if let Some(f0) = self.unbalance.as_ref().filter(|_| self.t1 > 0.0) {
            crate::materials::initial_state::add_unbalance(&mut self.fext, f0, self.t1);
        }
        // villkorskrafter (kopplingselement) ur den okondenserade residualen
        if !self.dm.mpc.is_empty() {
            for it in &self.p.interactions {
                it.observe_residual(&self.fint, &self.fext, &self.dm.mpc.slaves);
            }
        }
        // linjära villkor: krafter i slavar förs över till deras masters
        self.dm.mpc.condense(&mut self.fint);
        self.dm.mpc.condense(&mut self.fext);
        let mut r = vec![0.0; self.dm.n_eq];
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            r[q] = self.fext[n][s] - self.fint[n][s];
        }
        let mut sum: BTreeMap<Field, (f64, usize)> = BTreeMap::new();
        for n in 0..self.u.len() {
            for s in 0..SLOTS {
                if self.dm.active[n] & (1 << s) != 0 {
                    let e = sum.entry(dof::field(SLOT_DOF[s])).or_default();
                    let v = self.fint[n][s].abs().max(self.fext[n][s].abs());
                    if v > 0.0 {
                        e.0 += v;
                        e.1 += 1;
                    }
                }
            }
        }
        let flux = sum.into_iter().map(|(f, (s, c))| (f, if c > 0 { s / c as f64 } else { 0.0 })).collect();
        Ok(Residual { r, flux })
    }

    fn solve(&mut self, r: &[f64]) -> Result<Vec<f64>> {
        if let Err(q) = self.mat.factor() {
            return Err(assembly::mechanism_message(self.p, &self.dm, q));
        }
        let mut x = r.to_vec();
        self.mat.solve(&mut x);
        Ok(x)
    }

    fn apply(&mut self, c: &[f64]) {
        for (q, &(n, s)) in self.dm.owner.iter().enumerate() {
            self.u[n][s] += c[q];
        }
        self.dm.mpc.expand(&mut self.u);
    }

    fn increment(&self) -> Vec<f64> {
        self.dm.owner.iter().map(|&(n, s)| self.u[n][s] - self.u_committed[n][s]).collect()
    }

    fn commit(&mut self) -> Result<()> {
        self.u_committed.clone_from(&self.u);
        for it in &self.p.interactions {
            it.commit();
        }
        std::mem::swap(&mut self.state_old, &mut self.state_new);
        self.state_new.clone_from(&self.state_old);
        Ok(())
    }
}
