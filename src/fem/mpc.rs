//! Eliminering av linjära villkor (tie, MPC, *EQUATION, kinematisk och
//! fördelande coupling, kopplingselement) med master–slav-transformation.
//!
//! Varje villkor Σ cᵢ uᵢ = r gör en frihetsgrad beroende (slav):
//! u_s = r' + Σ c'ⱼ u_j, där alla u_j är oberoende (fria eller föreskrivna).
//! Slaven tas bort ur ekvationssystemet och styvhet/krafter i slaven förs över
//! till dess masters, dvs. K_red = Tᵀ K T och f_red = Tᵀ f (Zienkiewicz &
//! Taylor, kap. 1; Cook m.fl., avsnitt 9.2). Matrisen förblir då positivt
//! definit och villkoren uppfylls exakt.
//!
//! Val av slav: villkorets första term om den är fri (första
//! termen i *EQUATION och första noden i *MPC elimineras), annars den fria term
//! som har störst koefficient. Kedjor (en slav som ingår i ett annat villkor)
//! löses genom substitution, så att slavernas uttryck bara innehåller
//! oberoende frihetsgrader.

use crate::fem::assembly::{Prepared, NONE};
use crate::fem::dof::{self, NodeVec, SLOTS};
use crate::interactions::LinearConstraint;
use crate::model::{fail, Result};
use std::collections::{BTreeMap, HashMap};

/// En beroende frihetsgrad: u[node][slot] = rhs + Σ c · u[n][s].
#[derive(Debug, Clone)]
pub struct Slave {
    pub node: usize,
    pub slot: usize,
    pub terms: Vec<(usize, usize, f64)>,
    pub rhs: f64,
}

/// Alla beroende frihetsgrader i ett steg.
#[derive(Debug, Clone, Default)]
pub struct MpcMap {
    /// Slavindex per nod och slot (`NONE` = oberoende). Tom om villkor saknas.
    index: Vec<[usize; SLOTS]>,
    pub slaves: Vec<Slave>,
    /// Masternoder för noder som har slavar (för matrisstrukturen).
    pub masters: HashMap<usize, Vec<usize>>,
    /// Randvillkor (nod, slot) som ignoreras eftersom frihetsgraden är slav.
    pub released: Vec<(usize, usize)>,
}

impl MpcMap {
    pub fn is_empty(&self) -> bool {
        self.slaves.is_empty()
    }

    /// Slaven i (nod, slot), om frihetsgraden är beroende.
    #[inline]
    pub fn slave(&self, n: usize, s: usize) -> Option<&Slave> {
        if self.index.is_empty() {
            return None;
        }
        let i = self.index[n][s];
        (i != NONE).then(|| &self.slaves[i])
    }

    #[inline]
    pub fn is_slave(&self, n: usize, s: usize) -> bool {
        !self.index.is_empty() && self.index[n][s] != NONE
    }

    /// För över krafter i slavarna till deras masters: f ← Tᵀ f (slavernas värden blir 0).
    pub fn condense(&self, f: &mut [NodeVec]) {
        for sl in &self.slaves {
            let v = f[sl.node][sl.slot];
            if v != 0.0 {
                for &(n, s, c) in &sl.terms {
                    f[n][s] += c * v;
                }
            }
            f[sl.node][sl.slot] = 0.0;
        }
    }

    /// Sätter slavernas värden från deras masters: u_s = r + Σ c u_m.
    pub fn expand(&self, u: &mut [NodeVec]) {
        for sl in &self.slaves {
            let mut v = sl.rhs;
            for &(n, s, c) in &sl.terms {
                v += c * u[n][s];
            }
            u[sl.node][sl.slot] = v;
        }
    }

    /// Bygger slavuttrycken från interaktionernas villkor. `constrained` är
    /// låsta slots per nod (randvillkor); de blir aldrig slavar.
    pub fn build(p: &Prepared, constrained: &[u16]) -> Result<MpcMap> {
        let mut all: Vec<(String, LinearConstraint)> = Vec::new();
        for it in &p.interactions {
            for c in it.linear_constraints()? {
                all.push((it.name(), c));
            }
        }
        if all.is_empty() {
            return Ok(MpcMap::default());
        }
        build_from(p, constrained, &all)
    }

    /// Som `build` men med villkoren linjäriserade i läget `u`
    /// (`Interaction::linear_constraints_at`), för villkor som beror på läget.
    pub fn build_at(p: &Prepared, constrained: &[u16], u: &[NodeVec], nlgeom: bool) -> Result<MpcMap> {
        let mut all: Vec<(String, LinearConstraint)> = Vec::new();
        for it in &p.interactions {
            for c in it.linear_constraints_at(u, nlgeom)? {
                all.push((it.name(), c));
            }
        }
        if all.is_empty() {
            return Ok(MpcMap::default());
        }
        build_from(p, constrained, &all)
    }

    /// Samma slavar (nod, slot) i samma ordning.
    pub fn same_slaves(&self, other: &MpcMap) -> bool {
        self.slaves.len() == other.slaves.len() && self.slaves.iter().zip(&other.slaves).all(|(a, b)| a.node == b.node && a.slot == b.slot)
    }

    /// Linjäriserar om villkor som beror på läget (`Interaction::constraints_depend_on_u`)
    /// i `u` och sätter slavarnas värden. Anropas av NLGEOM-steget före varje
    /// utvärdering; utan sådana villkor sätts bara slavarna (`expand`).
    pub fn update_at(&mut self, p: &Prepared, constrained: &[u16], u: &mut [NodeVec]) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        if p.interactions.iter().any(|i| i.constraints_depend_on_u()) {
            // matrisstrukturen (masters) och släppta randvillkor behålls
            let new = MpcMap::build_at(p, constrained, u, true)?;
            if !self.same_slaves(&new) {
                return fail("intern: villkorens slavar ändrades när villkoren linjäriserades om");
            }
            self.slaves = new.slaves;
            self.index = new.index;
        }
        self.expand(u);
        Ok(())
    }
}

type Expr = BTreeMap<(usize, usize), f64>;

fn build_from(p: &Prepared, constrained: &[u16], all: &[(String, LinearConstraint)]) -> Result<MpcMap> {
    let nn = p.model.nodes.len();
    let mut index = vec![[NONE; SLOTS]; nn];
    // slavuttryck (termer, rhs) och vilka slavar som använder en viss frihetsgrad
    let mut exprs: Vec<(usize, usize, Expr, f64)> = Vec::new();
    let mut users: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut released: Vec<(usize, usize)> = Vec::new();
    for (name, c) in all {
        // randvillkor på en slav som måste elimineras ignoreras
        let forced = c.force_slave.then(|| c.terms.first().and_then(|t| dof::slot(t.1).map(|s| (t.0, s)))).flatten();
        if let Some(f) = forced {
            if index[f.0][f.1] == NONE && constrained.get(f.0).is_some_and(|m| m & (1 << f.1) != 0) && !released.contains(&f) {
                released.push(f);
            }
        }
        let is_bc = |n: usize, s: usize| constrained.get(n).is_some_and(|m| m & (1 << s) != 0) && !released.contains(&(n, s));
        // Σ a x = r med slavar substituerade
        let mut e: Expr = BTreeMap::new();
        let mut rhs = c.rhs;
        let mut first: Option<(usize, usize)> = None;
        for &(n, d, a) in &c.terms {
            let Some(s) = dof::slot(d) else {
                return fail(format!("{name}: frihetsgrad {d} stöds inte i villkor"));
            };
            if p.active[n] & (1 << s) == 0 {
                return fail(format!(
                    "{name}: villkoret använder {} i nod {} som inte har den frihetsgraden",
                    dof::name(d),
                    p.model.node_name(n)
                ));
            }
            if first.is_none() {
                first = Some((n, s));
            }
            let i = index[n][s];
            if i != NONE {
                let (_, _, se, sr) = &exprs[i];
                for (&k, &b) in se {
                    *e.entry(k).or_default() += a * b;
                }
                rhs -= a * sr;
            } else {
                *e.entry((n, s)).or_default() += a;
            }
        }
        let amax = e.values().fold(0.0f64, |m, v| m.max(v.abs()));
        e.retain(|_, v| v.abs() > 1e-12 * amax);
        if e.is_empty() {
            if rhs.abs() > 1e-9 * c.rhs.abs().max(1.0) {
                return fail(format!("{name}: motstridiga villkor (överbestämt system)"));
            }
            continue; // redundant villkor (överbestämda villkor tas bort)
        }
        // välj slav
        let free_terms: Vec<((usize, usize), f64)> = e.iter().filter(|((n, s), _)| !is_bc(*n, *s)).map(|(&k, &v)| (k, v)).collect();
        let chosen = match first.and_then(|f| e.get(&f).map(|&v| (f, v))) {
            Some((f, v)) if !is_bc(f.0, f.1) && v.abs() >= 1e-3 * amax => Some((f, v)),
            _ => free_terms.iter().copied().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())),
        };
        let Some(((sn, ss), a_s)) = chosen else {
            // alla termer föreskrivna med randvillkor: randvillkoren bestämmer
            // värdena (sådana överbestämda villkor tas bort)
            continue;
        };
        let mut se: Expr = BTreeMap::new();
        for (&k, &v) in &e {
            if k != (sn, ss) {
                se.insert(k, -v / a_s);
            }
        }
        let sr = rhs / a_s;
        let id = exprs.len();
        // substituera den nya slaven i tidigare uttryck som använder den
        if let Some(us) = users.remove(&(sn, ss)) {
            for u in us {
                let (_, _, ue, ur) = &mut exprs[u];
                let Some(b) = ue.remove(&(sn, ss)) else { continue };
                for (&k, &v) in &se {
                    *ue.entry(k).or_default() += b * v;
                    users.entry(k).or_default().push(u);
                }
                *ur += b * sr;
            }
        }
        for &k in se.keys() {
            users.entry(k).or_default().push(id);
        }
        index[sn][ss] = id;
        exprs.push((sn, ss, se, sr));
    }
    let mut masters: HashMap<usize, Vec<usize>> = HashMap::new();
    let slaves: Vec<Slave> = exprs
        .into_iter()
        .map(|(node, slot, e, rhs)| {
            let terms: Vec<(usize, usize, f64)> = e.into_iter().filter(|(_, v)| *v != 0.0).map(|((n, s), v)| (n, s, v)).collect();
            let m = masters.entry(node).or_default();
            m.extend(terms.iter().map(|t| t.0));
            Slave { node, slot, terms, rhs }
        })
        .collect();
    for m in masters.values_mut() {
        m.sort_unstable();
        m.dedup();
    }
    Ok(MpcMap { index, slaves, masters, released })
}

/// Kontrollerar att villkoren går att bygga (utan randvillkor).
pub fn check(p: &Prepared) -> Result<()> {
    MpcMap::build(p, &vec![0u16; p.model.nodes.len()]).map(|_| ())
}
