//! Elimination of linear constraints (tie, MPC, *EQUATION, kinematic and
//! distributing coupling, connector elements) with a master–slave transformation.
//!
//! Each constraint Σ cᵢ uᵢ = r makes one degree of freedom dependent (slave):
//! u_s = r' + Σ c'ⱼ u_j, where all u_j are independent (free or prescribed).
//! The slave is removed from the equation system and stiffness/forces at the slave are transferred
//! to its masters, i.e. K_red = Tᵀ K T and f_red = Tᵀ f (Zienkiewicz &
//! Taylor, ch. 1; Cook et al., section 9.2). The matrix then remains positive
//! definite and the constraints are satisfied exactly.
//!
//! Choice of slave: the first term of the constraint if it is free (the first
//! term in *EQUATION and the first node in *MPC are eliminated), otherwise the free term
//! with the largest coefficient. Chains (a slave that appears in another constraint)
//! are resolved by substitution, so that the slave expressions only contain
//! independent degrees of freedom.

use crate::fem::assembly::{Prepared, NONE};
use crate::fem::dof::{self, NodeVec, SLOTS};
use crate::interactions::LinearConstraint;
use crate::model::{fail, Result};
use std::collections::{BTreeMap, HashMap};

/// A dependent degree of freedom: u[node][slot] = rhs + Σ c · u[n][s].
#[derive(Debug, Clone)]
pub struct Slave {
    pub node: usize,
    pub slot: usize,
    pub terms: Vec<(usize, usize, f64)>,
    pub rhs: f64,
}

/// All dependent degrees of freedom in a step.
#[derive(Debug, Clone, Default)]
pub struct MpcMap {
    /// Slave index per node and slot (`NONE` = independent). Empty if there are no constraints.
    index: Vec<[usize; SLOTS]>,
    pub slaves: Vec<Slave>,
    /// Master nodes for nodes that have slaves (for the matrix structure).
    pub masters: HashMap<usize, Vec<usize>>,
    /// Boundary conditions (node, slot) that are ignored because the DOF is a slave.
    pub released: Vec<(usize, usize)>,
}

impl MpcMap {
    pub fn is_empty(&self) -> bool {
        self.slaves.is_empty()
    }

    /// The slave at (node, slot), if the DOF is dependent.
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

    /// Transfers forces at the slaves to their masters: f ← Tᵀ f (the slave values become 0).
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

    /// Sets the slave values from their masters: u_s = r + Σ c u_m.
    pub fn expand(&self, u: &mut [NodeVec]) {
        for sl in &self.slaves {
            let mut v = sl.rhs;
            for &(n, s, c) in &sl.terms {
                v += c * u[n][s];
            }
            u[sl.node][sl.slot] = v;
        }
    }

    /// Builds the slave expressions from the constraints of the interactions. `constrained` are
    /// the constrained slots per node (boundary conditions); they never become slaves.
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

    /// Like `build` but with the constraints linearized at the configuration `u`
    /// (`Interaction::linear_constraints_at`), for constraints that depend on the configuration.
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

    /// Same slaves (node, slot) in the same order.
    pub fn same_slaves(&self, other: &MpcMap) -> bool {
        self.slaves.len() == other.slaves.len() && self.slaves.iter().zip(&other.slaves).all(|(a, b)| a.node == b.node && a.slot == b.slot)
    }

    /// Relinearizes constraints that depend on the configuration (`Interaction::constraints_depend_on_u`)
    /// at `u` and sets the slave values. Called by the NLGEOM step before each
    /// evaluation; without such constraints only the slaves are set (`expand`).
    pub fn update_at(&mut self, p: &Prepared, constrained: &[u16], u: &mut [NodeVec]) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        if p.interactions.iter().any(|i| i.constraints_depend_on_u()) {
            // the matrix structure (masters) and released boundary conditions are kept
            let new = MpcMap::build_at(p, constrained, u, true)?;
            if !self.same_slaves(&new) {
                return fail("internal: the constraint slaves changed when the constraints were relinearized");
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
    // slave expressions (terms, rhs) and which slaves use a given DOF
    let mut exprs: Vec<(usize, usize, Expr, f64)> = Vec::new();
    let mut users: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut released: Vec<(usize, usize)> = Vec::new();
    for (name, c) in all {
        // boundary conditions on a slave that must be eliminated are ignored
        let forced = c.force_slave.then(|| c.terms.first().and_then(|t| dof::slot(t.1).map(|s| (t.0, s)))).flatten();
        if let Some(f) = forced {
            if index[f.0][f.1] == NONE && constrained.get(f.0).is_some_and(|m| m & (1 << f.1) != 0) && !released.contains(&f) {
                released.push(f);
            }
        }
        let is_bc = |n: usize, s: usize| constrained.get(n).is_some_and(|m| m & (1 << s) != 0) && !released.contains(&(n, s));
        // Σ a x = r with slaves substituted
        let mut e: Expr = BTreeMap::new();
        let mut rhs = c.rhs;
        let mut first: Option<(usize, usize)> = None;
        for &(n, d, a) in &c.terms {
            let Some(s) = dof::slot(d) else {
                return fail(format!("{name}: degree of freedom {d} is not supported in constraints"));
            };
            if p.active[n] & (1 << s) == 0 {
                return fail(format!(
                    "{name}: the constraint uses {} at node {} which does not have that degree of freedom",
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
                return fail(format!("{name}: conflicting constraints (overdetermined system)"));
            }
            continue; // redundant constraint (overdetermined constraints are removed)
        }
        // choose slave
        let free_terms: Vec<((usize, usize), f64)> = e.iter().filter(|((n, s), _)| !is_bc(*n, *s)).map(|(&k, &v)| (k, v)).collect();
        let chosen = match first.and_then(|f| e.get(&f).map(|&v| (f, v))) {
            Some((f, v)) if !is_bc(f.0, f.1) && v.abs() >= 1e-3 * amax => Some((f, v)),
            _ => free_terms.iter().copied().max_by(|a, b| a.1.abs().total_cmp(&b.1.abs())),
        };
        let Some(((sn, ss), a_s)) = chosen else {
            // all terms prescribed by boundary conditions: the boundary conditions determine
            // the values (such overdetermined constraints are removed)
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
        // substitute the new slave into earlier expressions that use it
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

/// Checks that the constraints can be built (without boundary conditions).
pub fn check(p: &Prepared) -> Result<()> {
    MpcMap::build(p, &vec![0u16; p.model.nodes.len()]).map(|_| ())
}
