//! Minnesuppskattning före lösning och val av lösare (se PLAN.md, "Minnestak
//! och lösarval").

use rustyfem::fem::dof;
use rustyfem::fem::model::FeModel;
use serde::Serialize;
use specta::Type;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Type)]
pub struct MemoryEstimate {
    /// Antal obekanta (utan randvillkor).
    pub unknowns: f64,
    /// Lagrade tal i faktorn (direktlösaren).
    pub factor_entries: f64,
    /// Uppskattat behov med direktlösaren (byte).
    pub direct_bytes: f64,
    /// Uppskattat behov med PCG och AMG (byte).
    pub iterative_bytes: f64,
    pub budget_bytes: f64,
    /// Budgeten gäller alltid (iOS); annars kan användaren välja att köra ändå.
    pub hard: bool,
    /// Plattformen ("windows", "macos", "linux", "ios").
    pub platform: String,
    /// Lösare som föreslås ("direct", "iterative" eller den som användaren låst).
    pub solver: String,
    /// Körningen ryms i budgeten med den föreslagna lösaren.
    pub fits: bool,
    /// Förklaring till användaren.
    pub note: String,
}

/// Procedurer där PCG fungerar (symmetriskt och positivt definit).
fn spd_only(m: &FeModel) -> bool {
    m.steps.iter().all(|s| {
        let unsymm = s.header.param_upper("UNSYMM").is_some_and(|v| v == "YES");
        let proc_ok = matches!(s.procedure_name(), "STATIC" | "VISCO" | "HEAT TRANSFER") && s.procedure.as_ref().is_none_or(|p| !p.has("RIKS"));
        proc_ok && !unsymm
    }) && !m.materials.iter().any(|mt| mt.behaviors.iter().any(|b| b.name.starts_with("CONCRETE")))
}

/// Uppskattar minnesbehovet och väljer lösare. `threads` = antal trådar i faktoriseringen.
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
            (s.to_string(), need <= budget, format!("lösaren är låst till {s}"))
        }
        None if direct <= budget => ("direct".to_string(), true, "direktlösaren ryms i minnesbudgeten".to_string()),
        None if spd && iterative <= budget => (
            "iterative".to_string(),
            true,
            "direktlösaren ryms inte i budgeten – byter till PCG med AMG (symmetriskt, positivt definit system)".to_string(),
        ),
        None => (
            "direct".to_string(),
            false,
            if spd {
                "modellen är för stor även för den iterativa lösaren – gör nätet grövre".to_string()
            } else {
                "direktlösaren ryms inte i budgeten och steget kräver direktlösare (indefinit/osymmetriskt eller egenvärden) – gör nätet grövre".to_string()
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
