//! Lösningstillstånd som ärvs mellan steps, och laster/randvillkor för ett step.
//!
//! Regler (för allmänna steg):
//! - Randvillkor och laster ärvs från föregående steg. OP=MOD (standard)
//!   ändrar eller lägger till poster; OP=NEW tar först bort alla tidigare
//!   poster av samma slag.
//! - Inom ett step rampas värdena linjärt från föregående stegs värden till
//!   de nya (AMPLITUDE=RAMP, standard för statik), eller sätts direkt
//!   (AMPLITUDE=STEP).
//! - Ett borttaget randvillkor ersätts av sin reaktionskraft som rampas ned
//!   till noll under steget.
//! - Ett nytt föreskrivet värde rampas från nodens aktuella förskjutning.
//!
//! Linjära perturbationssteg (PERTURBATION, FREQUENCY, BUCKLE) utgår från
//! bastillståndet: bara stegets egna laster gäller, ärvda randvillkor har
//! värdet noll, och tillståndet efter steget återställs till bastillståndet.

use crate::fem::dof::{self, NodeVec};
use crate::fem::model::{DistLoad, FeModel, StepDef};
use crate::model::{fail, Result};
use std::collections::{BTreeMap, BTreeSet};

/// Tillstånd som ärvs från step till step.
#[derive(Debug, Clone, Default)]
pub struct SolutionState {
    /// Totala nodvärden (förskjutning, rotation, temperatur, ...).
    pub u: Vec<NodeVec>,
    /// Godkänt elementtillstånd (spänningar, materialtillstånd).
    pub elem_state: Vec<Vec<f64>>,
    /// Total tid vid senaste stegets slut.
    pub time: f64,
    /// NLGEOM är påslaget (ärvs till följande steg).
    pub nlgeom: bool,
    /// Gällande randvillkor (nod, slot) → värde vid stegets slut.
    pub bcs: BTreeMap<(usize, usize), f64>,
    pub cloads: BTreeMap<(usize, usize), f64>,
    pub dloads: BTreeMap<(usize, String), DistLoad>,
    /// Reaktioner vid senaste stegets slut.
    pub reaction: Vec<NodeVec>,
    /// Övriga nodvektorer som steg-typer behöver ärva (t.ex. "V", "A").
    pub vectors: BTreeMap<String, Vec<NodeVec>>,
}

impl SolutionState {
    pub fn new(model: &FeModel) -> Self {
        SolutionState { u: vec![[0.0; dof::SLOTS]; model.nodes.len()], reaction: vec![[0.0; dof::SLOTS]; model.nodes.len()], ..Default::default() }
    }
}

/// Randvillkor och laster under ett step: (värde vid stegets början, vid slutet).
#[derive(Debug, Clone, Default)]
pub struct StepLoads {
    pub bc: BTreeMap<(usize, usize), (f64, f64)>,
    pub cload: BTreeMap<(usize, usize), (f64, f64)>,
    /// Fördelade laster med storlek vid slutet (`magnitude`) och vid början.
    pub dload: Vec<(DistLoad, f64)>,
    /// Reaktioner från borttagna randvillkor: (nod, slot, kraft vid stegets början).
    pub released: Vec<(usize, usize, f64)>,
    /// Linjär ramp över steget (annars steg-amplitud).
    pub ramp: bool,
    pub perturbation: bool,
    /// Amplitudreferenser (AMPLITUDE=) för poster som definieras i steget.
    /// Fylls bara av `build_with` med `allow_amplitudes`; en steg-typ som
    /// läser dem beräknar värdet som storlek · A(t) (se `steps/dynamic_amplitude.rs`).
    pub amp_bc: BTreeMap<(usize, usize), String>,
    pub amp_cload: BTreeMap<(usize, usize), String>,
    /// Parallell med `dload`.
    pub amp_dload: Vec<Option<String>>,
    /// *BOUNDARY, USER: randvillkor vars värde ges av DISP (`user/loads.rs`).
    pub user_bc: Vec<crate::user::loads::UserBc>,
}

impl StepLoads {
    /// Amplitud vid stegtidsandelen `t` ∈ [0, 1].
    pub fn amp(&self, t: f64) -> f64 {
        if self.ramp {
            t.clamp(0.0, 1.0)
        } else if t > 0.0 {
            1.0
        } else {
            0.0
        }
    }

    fn lerp(&self, (a, b): (f64, f64), t: f64) -> f64 {
        a + (b - a) * self.amp(t)
    }

    pub fn constrained(&self) -> BTreeSet<(usize, usize)> {
        self.bc.keys().copied().collect()
    }

    /// Föreskrivna värden vid `t`.
    pub fn bc_at(&self, t: f64) -> Vec<(usize, usize, f64)> {
        let mut v: Vec<(usize, usize, f64)> = self.bc.iter().map(|(&(n, s), &v)| (n, s, self.lerp(v, t))).collect();
        if !self.user_bc.is_empty() {
            let user: BTreeMap<(usize, usize), f64> = crate::user::loads::disp_values(&self.user_bc, t).into_iter().map(|(n, s, x)| ((n, s), x)).collect();
            for (n, s, x) in v.iter_mut() {
                if let Some(u) = user.get(&(*n, *s)) {
                    *x = *u;
                }
            }
        }
        v
    }

    /// Punktlaster vid `t` (inklusive rampade reaktioner från borttagna randvillkor).
    pub fn cloads_at(&self, t: f64) -> Vec<(usize, usize, f64)> {
        let mut v: Vec<(usize, usize, f64)> = self.cload.iter().map(|(&(n, s), &c)| (n, s, self.lerp(c, t))).collect();
        for &(n, s, r) in &self.released {
            v.push((n, s, -r * (1.0 - self.amp(t))));
        }
        v
    }

    pub fn dloads_at(&self, t: f64) -> Vec<DistLoad> {
        self.dload
            .iter()
            .map(|(l, start)| {
                let mut l = l.clone();
                l.magnitude = self.lerp((*start, l.magnitude), t);
                l
            })
            .collect()
    }

    /// Bygger stegets laster och randvillkor från tillståndet före steget.
    /// Returnerar även de nya (ärvda) värdena som gäller efter steget.
    pub fn build(model: &FeModel, step: &StepDef, state: &SolutionState, active: &[u16]) -> Result<(StepLoads, Propagated)> {
        Self::build_with(model, step, state, active, false)
    }

    /// Som `build`. Med `allow_amplitudes` godtas AMPLITUDE= på randvillkor och
    /// laster; referenserna sparas i `amp_bc`/`amp_cload`/`amp_dload` (steg-typen
    /// måste då själv utvärdera amplituderna).
    pub fn build_with(model: &FeModel, step: &StepDef, state: &SolutionState, active: &[u16], allow_amplitudes: bool) -> Result<(StepLoads, Propagated)> {
        let mut amp_bc: BTreeMap<(usize, usize), String> = BTreeMap::new();
        let mut amp_cl: BTreeMap<(usize, usize), String> = BTreeMap::new();
        let mut amp_dl: BTreeMap<(usize, String), String> = BTreeMap::new();
        let mut warn = Vec::new();
        let on = |n: usize, s: usize| active.get(n).is_some_and(|a| a & (1 << s) != 0);
        let mut bcs = if step.perturbation { state.bcs.keys().map(|&k| (k, 0.0)).collect() } else { state.bcs.clone() };
        let mut cl: BTreeMap<(usize, usize), f64> = if step.perturbation { BTreeMap::new() } else { state.cloads.clone() };
        let mut dl: BTreeMap<(usize, String), DistLoad> = if step.perturbation { BTreeMap::new() } else { state.dloads.clone() };
        for b in &step.bcs {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= stöds inte ännu", b.source));
            }
            if !(b.kind.is_empty() || b.kind == "DISPLACEMENT") {
                return fail(format!("{}: TYPE={} stöds inte i detta steg", b.source, b.kind));
            }
            if b.op_new {
                bcs.clear();
                amp_bc.clear();
            }
            for &(n, d, v) in &b.items {
                let s = dof::slot_of(d);
                if on(n, s) {
                    bcs.insert((n, s), v);
                    match &b.amplitude {
                        Some(a) => amp_bc.insert((n, s), a.clone()),
                        None => amp_bc.remove(&(n, s)),
                    };
                } else if v != 0.0 {
                    warn.push(format!("{}: nod {} saknar frihetsgrad {} – randvillkoret ignoreras", b.source, model.node_name(n), d));
                }
            }
        }
        for b in &step.cloads {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= stöds inte ännu", b.source));
            }
            if b.follower {
                return fail(format!("{}: FOLLOWER=YES stöds inte ännu", b.source));
            }
            if b.op_new {
                cl.clear();
                amp_cl.clear();
            }
            // poster i samma block summeras, ersätter tidigare stegs värde
            let mut block: BTreeMap<(usize, usize), f64> = BTreeMap::new();
            for &(n, d, v) in &b.items {
                let s = dof::slot_of(d);
                if !on(n, s) {
                    return fail(format!("{}: nod {} saknar frihetsgrad {d}", b.source, model.node_name(n)));
                }
                *block.entry((n, s)).or_default() += v;
            }
            for k in block.keys() {
                match &b.amplitude {
                    Some(a) => amp_cl.insert(*k, a.clone()),
                    None => amp_cl.remove(k),
                };
            }
            cl.extend(block);
        }
        for b in &step.dloads {
            if b.amplitude.is_some() && !allow_amplitudes {
                return fail(format!("{}: AMPLITUDE= stöds inte ännu", b.source));
            }
            if b.op_new {
                dl.clear();
                amp_dl.clear();
            }
            let mut block: BTreeMap<(usize, String), DistLoad> = BTreeMap::new();
            for l in &b.items {
                let key = (l.elem, l.kind.clone());
                match block.get_mut(&key) {
                    Some(old) if old.extra == l.extra => old.magnitude += l.magnitude,
                    Some(_) => return fail(format!("{}: två {}-laster med olika riktning på samma element", b.source, l.kind)),
                    None => {
                        block.insert(key, l.clone());
                    }
                }
            }
            for k in block.keys() {
                match &b.amplitude {
                    Some(a) => amp_dl.insert(k.clone(), a.clone()),
                    None => amp_dl.remove(k),
                };
            }
            dl.extend(block);
        }
        let mut loads = StepLoads { ramp: !step.amplitude_step, perturbation: step.perturbation, ..Default::default() };
        for (&k, &v) in &bcs {
            let start = if step.perturbation {
                0.0
            } else {
                state.bcs.get(&k).copied().unwrap_or_else(|| state.u.get(k.0).map_or(0.0, |u| u[k.1]))
            };
            loads.bc.insert(k, (start, v));
        }
        if !step.perturbation {
            for (&k, _) in &state.bcs {
                if !bcs.contains_key(&k) {
                    loads.released.push((k.0, k.1, state.reaction.get(k.0).map_or(0.0, |r| r[k.1])));
                }
            }
        }
        let keys: BTreeSet<(usize, usize)> = cl.keys().chain(if step.perturbation { None } else { Some(state.cloads.keys()) }.into_iter().flatten()).copied().collect();
        for k in keys {
            let start = if step.perturbation { 0.0 } else { state.cloads.get(&k).copied().unwrap_or(0.0) };
            loads.cload.insert(k, (start, cl.get(&k).copied().unwrap_or(0.0)));
        }
        let mut dkeys: BTreeSet<(usize, String)> = dl.keys().cloned().collect();
        if !step.perturbation {
            dkeys.extend(state.dloads.keys().cloned());
        }
        for k in dkeys {
            let start = if step.perturbation { 0.0 } else { state.dloads.get(&k).map_or(0.0, |l| l.magnitude) };
            let end = match (dl.get(&k), state.dloads.get(&k)) {
                (Some(l), _) => l.clone(),
                (None, Some(old)) => DistLoad { magnitude: 0.0, ..old.clone() },
                (None, None) => unreachable!(),
            };
            if let (Some(old), false) = (state.dloads.get(&k), step.perturbation) {
                if old.extra != end.extra && start != 0.0 {
                    return fail(format!("steg {}: ändrad riktning för {}-last stöds inte", step.name, k.1));
                }
            }
            loads.amp_dload.push(amp_dl.get(&k).cloned());
            loads.dload.push((end, start));
        }
        loads.amp_bc = amp_bc.into_iter().filter(|(k, _)| loads.bc.contains_key(k)).collect();
        loads.user_bc = crate::user::loads::user_bcs(model, step)?.into_iter().filter(|b| loads.bc.contains_key(&(b.node, b.slot))).collect();
        loads.amp_cload = amp_cl;
        Ok((loads, Propagated { bcs, cloads: cl, dloads: dl, warnings: warn }))
    }
}

/// Randvillkor och laster som gäller efter ett allmänt step.
#[derive(Debug, Clone, Default)]
pub struct Propagated {
    pub bcs: BTreeMap<(usize, usize), f64>,
    pub cloads: BTreeMap<(usize, usize), f64>,
    pub dloads: BTreeMap<(usize, String), DistLoad>,
    pub warnings: Vec<String>,
}
