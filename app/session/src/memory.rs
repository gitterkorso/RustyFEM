//! Memory estimate before solving and choice of solver (see PLAN.md, "Memory cap
//! and solver choice").

use rustyfem::fem::dof;
use rustyfem::fem::model::FeModel;
use serde::Serialize;
use specta::Type;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Type)]
pub struct MemoryEstimate {
    /// Number of unknowns (without boundary conditions).
    pub unknowns: f64,
    /// Stored entries in the factor (direct solver).
    pub factor_entries: f64,
    /// Estimated requirement with the direct solver (bytes).
    pub direct_bytes: f64,
    /// Estimated requirement with PCG and AMG (bytes).
    pub iterative_bytes: f64,
    pub budget_bytes: f64,
    /// The budget always applies (iOS); otherwise the user can choose to run anyway.
    pub hard: bool,
    /// The platform ("windows", "macos", "linux", "ios").
    pub platform: String,
    /// Suggested solver ("direct", "iterative" or the one the user has locked).
    pub solver: String,
    /// The run fits in the budget with the suggested solver.
    pub fits: bool,
    /// Explanation for the user.
    pub note: String,
}

/// Procedures where PCG works (symmetric and positive definite).
fn spd_only(m: &FeModel) -> bool {
    m.steps.iter().all(|s| {
        let unsymm = s.header.param_upper("UNSYMM").is_some_and(|v| v == "YES");
        let proc_ok = matches!(s.procedure_name(), "STATIC" | "VISCO" | "HEAT TRANSFER") && s.procedure.as_ref().is_none_or(|p| !p.has("RIKS"));
        proc_ok && !unsymm
    }) && !m.materials.iter().any(|mt| mt.behaviors.iter().any(|b| b.name.starts_with("CONCRETE")))
}

/// Estimates the memory requirement and chooses a solver. `threads` = number of threads in the factorization.
pub fn estimate(m: &FeModel, budget: f64, locked: Option<&str>, threads: usize) -> MemoryEstimate {
    let nn = m.nodes.len();
    let mut slots = vec![0u16; nn];
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); nn];
    for e in &m.elements {
        for (i, &n) in e.nodes.iter().enumerate() {
            if i < e.etype.n_nodes() {
                for &d in e.etype.node_dofs(i) {
                    if let Some(s) = dof::slot(d) {
                        slots[n] |= 1 << s;
                    }
                }
            }
            for &k in &e.nodes {
                if k != n {
                    adj[n].insert(k);
                }
            }
        }
    }
    let ndof: Vec<usize> = slots.iter().map(|s| s.count_ones() as usize).collect();
    let unknowns: usize = ndof.iter().sum();
    let adj: Vec<Vec<usize>> = adj.into_iter().map(|s| s.into_iter().collect()).collect();
    let nnz: usize = adj.iter().zip(&ndof).map(|(a, &d)| d * (d + a.iter().map(|&k| ndof[k]).sum::<usize>())).sum::<usize>() / 2;
    let sym = rustyfem::sparse::analyse(&m.nodes, &adj, &ndof);
    let factor = sym.factor_size() as f64;
    let (k, u) = sym.largest_front();
    let front = ((k + u) * (k + u)) as f64 * 8.0 * threads.max(1) as f64;
    let matrix = nnz as f64 * 12.0;
    let model = (m.elements.len() * 200 + nn * 200) as f64;
    let direct = factor * 8.0 * 1.15 + front + matrix + model;
    let iterative = matrix * 3.0 + unknowns as f64 * 8.0 * 8.0 + model;
    let spd = spd_only(m);
    let (solver, fits, note) = match locked {
        Some(s) => {
            let need = if s.starts_with("iterative") { iterative } else { direct };
            (s.to_string(), need <= budget, format!("the solver is locked to {s}"))
        }
        None if direct <= budget => ("direct".to_string(), true, "the direct solver fits in the memory budget".to_string()),
        None if spd && iterative <= budget => (
            "iterative".to_string(),
            true,
            "the direct solver does not fit in the budget – switching to PCG with AMG (symmetric, positive definite system)".to_string(),
        ),
        None => (
            "direct".to_string(),
            false,
            if spd {
                "the model is too large even for the iterative solver – use a coarser mesh".to_string()
            } else {
                "the direct solver does not fit in the budget and the step requires a direct solver (indefinite/unsymmetric or eigenvalues) – use a coarser mesh".to_string()
            },
        ),
    };
    MemoryEstimate {
        unknowns: unknowns as f64,
        factor_entries: factor,
        direct_bytes: direct,
        iterative_bytes: iterative,
        budget_bytes: budget,
        hard: crate::platform::HARD_BUDGET,
        platform: crate::platform::name().into(),
        solver,
        fits,
        note,
    }
}
