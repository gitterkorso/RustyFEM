//! Linear constraints: *EQUATION, *MPC (BEAM, PIN, TIE), *TIE (node-to-surface
//! and surface-to-surface), *COUPLING with *KINEMATIC or *DISTRIBUTING and
//! *KINEMATIC COUPLING.
//!
//! All constraints are formulated for small displacements and rotations
//! (linearized rigid-body motion u = u_R + θ_R × r) and are eliminated by the
//! core (`fem/mpc.rs`). The first term in each constraint is the degree of
//! freedom that is preferably eliminated.
//!
//! Distributing coupling (*DISTRIBUTING) follows from virtual work: if the
//! reference node's force F and moment M are distributed to the coupling nodes
//! as Fᵢ = w̄ᵢ F + w̄ᵢ (T⁻¹ m̃) × rᵢ with rᵢ = xᵢ − x̄ (x̄ = weighted centroid),
//! T = Σ w̄ᵢ (|rᵢ|² I − rᵢ rᵢᵀ) and m̃ = M + (x_R − x̄) × F, then the dual
//! kinematics are θ_R = T⁻¹ Σ w̄ᵢ rᵢ × uᵢ and u_R = Σ w̄ᵢ uᵢ + θ_R × (x_R − x̄).

use super::geom::{self, Facet};
use super::{Interaction, InteractionFactory, LinearConstraint};
use crate::elements::common::{dot, sub};
use crate::fem::dof::{Dof, U1, U2, U3, UR1, UR2, UR3};
use crate::fem::model::{parse_f64, FeModel, RawKeyword};
use crate::model::{fail, FemError, Result};

pub static FACTORIES: &[InteractionFactory] = &[
    InteractionFactory { keywords: &["COUPLING"], build: coupling },
    InteractionFactory { keywords: &["EQUATION"], build: equation },
    InteractionFactory { keywords: &["KINEMATIC COUPLING"], build: kinematic_coupling },
    InteractionFactory { keywords: &["MPC"], build: mpc },
    InteractionFactory { keywords: &["TIE"], build: tie },
];

/// A collection of linear constraints from one keyword.
pub struct Linear {
    pub name: String,
    pub constraints: Vec<LinearConstraint>,
    pub active: Vec<(usize, Dof)>,
}

impl Interaction for Linear {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn active_dofs(&self) -> Vec<(usize, Dof)> {
        self.active.clone()
    }
    fn linear_constraints(&self) -> Result<Vec<LinearConstraint>> {
        Ok(self.constraints.clone())
    }
    fn is_linear(&self) -> bool {
        true
    }
}

fn boxed(name: String, constraints: Vec<LinearConstraint>, active: Vec<(usize, Dof)>) -> Result<Vec<Box<dyn Interaction>>> {
    Ok(vec![Box::new(Linear { name, constraints, active })])
}

/// Index of the keyword in `model.extra` (for subordinate keywords that follow it).
pub fn position(m: &FeModel, kw: &RawKeyword) -> usize {
    m.extra.iter().position(|k| std::ptr::eq(k, kw)).unwrap_or(m.extra.len())
}

/// Keywords that directly follow `kw` in `model.extra` and whose names are in `names`.
pub fn sub_options<'a>(m: &'a FeModel, kw: &RawKeyword, names: &[&str]) -> Vec<&'a RawKeyword> {
    let i = position(m, kw);
    m.extra.iter().skip(i + 1).take_while(|k| names.contains(&k.name.as_str())).collect()
}

fn dof_of(kw: &RawKeyword, line: usize, s: &str) -> Result<Dof> {
    match crate::inp::parse_label(s) {
        Some(d @ (1..=7 | 11)) => Ok(d as Dof),
        _ => kw.err_at(line, format!("'{s}' is not a degree of freedom")),
    }
}

// ---------------------------------------------------------------- *EQUATION

fn equation(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let mut out = Vec::new();
    // Each equation: number of terms N, then N triplets (node, DOF, coefficient).
    // Data lines have at most 12 fields, so we read line by line: first N, then triplets.
    let mut lines = kw.data.iter().peekable();
    while let Some(d) = lines.next() {
        let first: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if first.is_empty() {
            continue;
        }
        let Some(nt) = crate::inp::parse_label(first[0]).filter(|&n| n >= 1) else {
            return kw.err_at(d.line, format!("expected number of terms, got '{}'", first[0]));
        };
        let mut items: Vec<(String, usize)> = first[1..].iter().map(|s| (s.to_string(), d.line)).collect();
        while items.len() < 3 * nt as usize {
            let Some(d2) = lines.next() else { return kw.err("the equation has too few terms") };
            items.extend(d2.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).map(|s| (s.to_string(), d2.line)));
        }
        // node sets are expanded: the equation applies to each node in the set (in order)
        let mut terms: Vec<(Vec<usize>, Dof, f64)> = Vec::new();
        for t in items.chunks(3) {
            let nodes = geom::node_ref(m, kw, &t[0].0)?;
            let dof = dof_of(kw, t[1].1, &t[1].0)?;
            let c = parse_f64(&t[2].0).ok_or_else(|| FemError(format!("{}: '{}' is not a number", kw.location(), t[2].0)))?;
            terms.push((nodes, dof, c));
        }
        let count = terms.iter().map(|t| t.0.len()).max().unwrap_or(1);
        if terms.iter().any(|t| t.0.len() != 1 && t.0.len() != count) {
            return kw.err_at(d.line, "the node sets in the equation have different numbers of nodes");
        }
        for k in 0..count {
            let tt = terms.iter().map(|(ns, d, c)| (if ns.len() == 1 { ns[0] } else { ns[k] }, *d, *c)).collect();
            out.push(LinearConstraint { terms: tt, rhs: 0.0, force_slave: false });
        }
    }
    boxed(format!("*EQUATION ({})", kw.location()), out, Vec::new())
}

// ---------------------------------------------------------------- rigid body

/// Constraints making `node` follow the reference node `r` as a rigid body
/// (small rotations): u = u_R + θ_R × (x − x_R) for the translations in `dofs`,
/// θ = θ_R for the rotations in `dofs` that the node has.
fn rigid_link(m: &FeModel, active: &[u16], node: usize, r: usize, dofs: &[Dof], out: &mut Vec<LinearConstraint>) {
    let dim = m.dim();
    let x = sub(m.nodes[node], m.nodes[r]);
    for &d in dofs {
        match d {
            1..=3 => {
                if dim == 2 && d == U3 {
                    continue;
                }
                if !geom::has(active, node, d) {
                    continue;
                }
                let i = (d - 1) as usize;
                let mut t = vec![(node, d, 1.0), (r, d, -1.0)];
                // (θ × x)_i = ε_ijk θ_j x_k
                let (j, k) = ((i + 1) % 3, (i + 2) % 3);
                if dim == 3 {
                    t.push((r, UR1 + j as Dof, -x[k]));
                    t.push((r, UR1 + k as Dof, x[j]));
                } else {
                    // only θ3: (θ3 e3 × x) = (−θ3 x2, θ3 x1)
                    if i == 0 {
                        t.push((r, UR3, x[1]));
                    } else {
                        t.push((r, UR3, -x[0]));
                    }
                }
                out.push(LinearConstraint { terms: t.into_iter().filter(|p| p.2 != 0.0).collect(), rhs: 0.0, force_slave: false });
            }
            4..=6 => {
                if dim == 2 && d != UR3 {
                    continue;
                }
                if geom::has(active, node, d) {
                    out.push(LinearConstraint { terms: vec![(node, d, 1.0), (r, d, -1.0)], rhs: 0.0, force_slave: false });
                }
            }
            _ => {}
        }
    }
}

/// Reference node DOFs (translations and rotations) in the model dimension.
fn ref_dofs(m: &FeModel, r: usize, rot: bool) -> Vec<(usize, Dof)> {
    let dim = m.dim();
    let mut v: Vec<(usize, Dof)> = geom::trans_dofs(dim).iter().map(|&d| (r, d)).collect();
    if rot {
        v.extend(geom::rot_dofs(dim).iter().map(|&d| (r, d)));
    }
    v
}

/// DOFs from data lines "first, last" (all if missing).
fn dof_ranges(kws: &[&RawKeyword]) -> Result<Vec<Dof>> {
    let mut v = Vec::new();
    for kw in kws {
        for d in &kw.data {
            let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            if f.is_empty() {
                continue;
            }
            let a = dof_of(kw, d.line, f[0])?;
            let b = match f.get(1) {
                Some(s) => dof_of(kw, d.line, s)?,
                None => a,
            };
            for x in a..=b {
                if (1..=6).contains(&x) && !v.contains(&x) {
                    v.push(x);
                }
            }
        }
    }
    if v.is_empty() {
        v = vec![U1, U2, U3, UR1, UR2, UR3];
    }
    v.sort_unstable();
    Ok(v)
}

// ---------------------------------------------------------------- *MPC

fn mpc(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let active = geom::element_active(m);
    let dim = m.dim();
    let mut out = Vec::new();
    let mut act = Vec::new();
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.len() < 3 {
            if f.is_empty() {
                continue;
            }
            return kw.err_at(d.line, "write MPC type, node a, node b");
        }
        let ty = f[0].to_uppercase();
        let a = geom::node_ref(m, kw, f[1])?;
        let b = geom::node_ref(m, kw, f[2])?;
        if b.len() != 1 && b.len() != a.len() {
            return kw.err_at(d.line, "the node sets have different numbers of nodes");
        }
        for (k, &na) in a.iter().enumerate() {
            let nb = if b.len() == 1 { b[0] } else { b[k] };
            match ty.as_str() {
                "BEAM" => {
                    // a follows b as a rigid beam; both nodes have rotations
                    for &d in geom::trans_dofs(dim).iter().chain(geom::rot_dofs(dim)) {
                        act.push((na, d));
                        act.push((nb, d));
                    }
                    let mut full = active.clone();
                    for &(n, d) in &act {
                        full[n] |= 1 << crate::fem::dof::slot_of(d);
                    }
                    rigid_link(m, &full, na, nb, &[U1, U2, U3, UR1, UR2, UR3], &mut out);
                }
                "PIN" | "TIE" => {
                    let mut dofs: Vec<Dof> = geom::trans_dofs(dim).to_vec();
                    if ty == "TIE" {
                        dofs.extend(geom::rot_dofs(dim).iter().filter(|&&d| geom::has(&active, na, d) && geom::has(&active, nb, d)));
                    }
                    for d in dofs {
                        if geom::has(&active, na, d) && geom::has(&active, nb, d) {
                            out.push(LinearConstraint { terms: vec![(na, d, 1.0), (nb, d, -1.0)], rhs: 0.0, force_slave: false });
                        }
                    }
                }
                other => return kw.err_at(d.line, format!("MPC type {other} is not supported yet (BEAM, PIN, TIE)")),
            }
        }
    }
    boxed(format!("*MPC ({})", kw.location()), out, act)
}

// ---------------------------------------------------------------- *COUPLING

fn ref_node(m: &FeModel, kw: &RawKeyword) -> Result<usize> {
    let r = crate::inp::required(kw, "REF NODE")?;
    let v = geom::node_ref(m, kw, r)?;
    if v.len() != 1 {
        return kw.err(format!("REF NODE={r} must be a single node"));
    }
    Ok(v[0])
}

fn coupling(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let r = ref_node(m, kw)?;
    let sname = crate::inp::required(kw, "SURFACE")?;
    let surf = geom::surface(m, kw, sname)?;
    if kw.has("INFLUENCE RADIUS") {
        return kw.err("INFLUENCE RADIUS= is not supported yet");
    }
    if kw.has("ORIENTATION") {
        return kw.err("ORIENTATION= is not supported yet");
    }
    let subs = sub_options(m, kw, &["KINEMATIC", "DISTRIBUTING"]);
    let Some(kind) = subs.first() else {
        return kw.err("*COUPLING must be followed by *KINEMATIC or *DISTRIBUTING");
    };
    let mut nodes = geom::surface_nodes(m, surf)?;
    nodes.retain(|&n| n != r);
    let dofs = dof_ranges(&subs)?;
    let active = geom::element_active(m);
    let dim = m.dim();
    let name = format!("*COUPLING {} ({})", kw.param("CONSTRAINT NAME").unwrap_or(""), kw.location());
    let mut out = Vec::new();
    if kind.name == "KINEMATIC" {
        // Coupled rotations are activated also at nodes that lack them (where
        // the coupling nodes' UR then follow the reference node).
        let mut act = ref_dofs(m, r, true);
        for &n in &nodes {
            act.extend(geom::rot_dofs(dim).iter().filter(|d| dofs.contains(d)).map(|&d| (n, d)));
        }
        let mut full = active.clone();
        for &(n, d) in &act {
            full[n] |= 1 << crate::fem::dof::slot_of(d);
        }
        for &n in &nodes {
            rigid_link(m, &full, n, r, &dofs, &mut out);
        }
        return boxed(name, out, act);
    }
    // distributing coupling
    let method = kind.param_upper("WEIGHTING METHOD").unwrap_or_else(|| "UNIFORM".into());
    let mut w: Vec<f64> = match surf.kind.as_str() {
        "NODE" => {
            let mut v = vec![0.0; nodes.len()];
            for &(n, wt) in &surf.nodes {
                if let Some(i) = nodes.iter().position(|&x| x == n) {
                    v[i] = wt;
                }
            }
            v
        }
        _ => {
            // element surfaces: the weight is the node's share of the area
            let mut v = vec![0.0; nodes.len()];
            for f in geom::facets(m, surf, &name)? {
                let x = geom::coords(m, &f, None);
                for (k, a) in geom::nodal_areas(&f, &x).into_iter().enumerate() {
                    if let Some(i) = nodes.iter().position(|&n| n == f.nodes[k]) {
                        v[i] += a;
                    }
                }
            }
            v
        }
    };
    if method != "UNIFORM" {
        return kind.err(format!("WEIGHTING METHOD={method} is not supported yet (UNIFORM)"));
    }
    let wsum: f64 = w.iter().sum();
    if wsum <= 0.0 {
        return kw.err("the coupling node weights are zero");
    }
    w.iter_mut().for_each(|x| *x /= wsum);
    let xr = m.nodes[r];
    let mut xc = [0.0; 3];
    for (i, &n) in nodes.iter().enumerate() {
        for k in 0..3 {
            xc[k] += w[i] * m.nodes[n][k];
        }
    }
    let rot = dofs.iter().any(|&d| (4..=6).contains(&d));
    let dvec = sub(xr, xc);
    let rs: Vec<[f64; 3]> = nodes.iter().map(|&n| sub(m.nodes[n], xc)).collect();
    // T⁻¹ (3D) or 1/Σw|r|² (2D)
    let mut tinv = [[0.0; 3]; 3];
    if rot {
        if dim == 3 {
            let mut t = [[0.0; 3]; 3];
            for (i, r) in rs.iter().enumerate() {
                let rr = dot(*r, *r);
                for a in 0..3 {
                    for b in 0..3 {
                        t[a][b] += w[i] * ((if a == b { rr } else { 0.0 }) - r[a] * r[b]);
                    }
                }
            }
            let (inv, det) = crate::elements::common::inv3(t);
            if det.abs() < 1e-14 * (t[0][0] + t[1][1] + t[2][2]).powi(3).max(1e-300) {
                return kw.err("the coupling nodes lie on a line – the rotation about the line is undetermined");
            }
            tinv = inv;
        } else {
            let s: f64 = rs.iter().enumerate().map(|(i, r)| w[i] * (r[0] * r[0] + r[1] * r[1])).sum();
            tinv[2][2] = 1.0 / s;
        }
    }
    // θ_R = T⁻¹ Σ wᵢ rᵢ × uᵢ ; (r × u)_a = ε_abc r_b u_c
    let rot_terms = |a: usize| -> Vec<(usize, Dof, f64)> {
        let mut t = Vec::new();
        for (i, &n) in nodes.iter().enumerate() {
            let r = rs[i];
            for c in 0..3 {
                if dim == 2 && c == 2 {
                    continue;
                }
                let d = U1 + c as Dof;
                if !geom::has(&active, n, d) {
                    continue;
                }
                // coefficient of u_c in (T⁻¹ (r × u))_a = Σ_e T⁻¹_ae ε_ebc r_b
                let mut coef = 0.0;
                for e in 0..3 {
                    let (b1, c1) = ((e + 1) % 3, (e + 2) % 3);
                    // ε_e b1 c1 = +1, ε_e c1 b1 = −1
                    let eps_rc = if c == c1 { r[b1] } else if c == b1 { -r[c1] } else { 0.0 };
                    coef += tinv[a][e] * eps_rc;
                }
                if coef != 0.0 {
                    t.push((n, d, w[i] * coef));
                }
            }
        }
        t
    };
    let mut act = ref_dofs(m, r, rot);
    if rot {
        for &d in geom::rot_dofs(dim) {
            if !dofs.contains(&d) {
                continue;
            }
            let a = (d - UR1) as usize;
            let mut t = vec![(r, d, 1.0)];
            t.extend(rot_terms(a).into_iter().map(|(n, dd, c)| (n, dd, -c)));
            out.push(LinearConstraint { terms: t, rhs: 0.0, force_slave: false });
        }
    }
    for &d in geom::trans_dofs(dim) {
        if !dofs.contains(&d) {
            continue;
        }
        let i = (d - U1) as usize;
        let mut t = vec![(r, d, 1.0)];
        for (k, &n) in nodes.iter().enumerate() {
            if geom::has(&active, n, d) {
                t.push((n, d, -w[k]));
            }
        }
        if rot {
            // − (θ_R × d)_i = −(θ_j d_k − θ_k d_j)
            let (j, k) = ((i + 1) % 3, (i + 2) % 3);
            if dim == 3 {
                t.push((r, UR1 + j as Dof, -dvec[k]));
                t.push((r, UR1 + k as Dof, dvec[j]));
            } else if i == 0 {
                t.push((r, UR3, dvec[1]));
            } else {
                t.push((r, UR3, -dvec[0]));
            }
        }
        out.push(LinearConstraint { terms: t.into_iter().filter(|p| p.2 != 0.0).collect(), rhs: 0.0, force_slave: false });
    }
    act.retain(|&(_, d)| d <= 3 || dofs.contains(&d));
    boxed(name, out, act)
}

fn kinematic_coupling(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let r = ref_node(m, kw)?;
    let active = geom::element_active(m);
    let dim = m.dim();
    let mut act = ref_dofs(m, r, true);
    let mut rows: Vec<(Vec<usize>, Vec<Dof>)> = Vec::new();
    for d in &kw.data {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.is_empty() {
            continue;
        }
        let nodes = geom::node_ref(m, kw, f[0])?;
        let a = match f.get(1) {
            Some(s) => dof_of(kw, d.line, s)?,
            None => 1,
        };
        let b = match f.get(2) {
            Some(s) => dof_of(kw, d.line, s)?,
            None if f.len() > 1 => a,
            None => 6,
        };
        let dofs: Vec<Dof> = (a..=b).filter(|x| (1..=6).contains(x)).collect();
        for &n in &nodes {
            act.extend(geom::rot_dofs(dim).iter().filter(|d| dofs.contains(d)).map(|&d| (n, d)));
        }
        rows.push((nodes, dofs));
    }
    let mut full = active.clone();
    for &(n, d) in &act {
        full[n] |= 1 << crate::fem::dof::slot_of(d);
    }
    let mut out = Vec::new();
    for (nodes, dofs) in rows {
        for n in nodes {
            if n != r {
                rigid_link(m, &full, n, r, &dofs, &mut out);
            }
        }
    }
    boxed(format!("*KINEMATIC COUPLING ({})", kw.location()), out, act)
}

// ---------------------------------------------------------------- *TIE

/// Master surface facets with nodal coordinates in the reference configuration.
struct Master {
    facets: Vec<Facet>,
    x: Vec<Vec<[f64; 3]>>,
    size: f64,
}

impl Master {
    fn new(m: &FeModel, f: Vec<Facet>) -> Master {
        let x: Vec<Vec<[f64; 3]>> = f.iter().map(|f| geom::coords(m, f, None)).collect();
        let mut size = 0.0;
        for xs in &x {
            let mut d = 0.0f64;
            for a in xs {
                for b in xs {
                    d = d.max(crate::elements::common::norm(sub(*a, *b)));
                }
            }
            size += d;
        }
        size /= x.len().max(1) as f64;
        Master { facets: f, x, size }
    }

    /// Closest facet: (facet, ξ, distance).
    fn closest(&self, p: [f64; 3]) -> Option<(usize, [f64; 2], f64, bool)> {
        let mut best: Option<(usize, [f64; 2], f64, bool)> = None;
        for (i, f) in self.facets.iter().enumerate() {
            let (xi, d, inside) = geom::project(f, &self.x[i], p);
            let better = match &best {
                None => true,
                Some(b) => (inside && !b.3 && d <= b.2 + 1e-9 * self.size) || (inside == b.3 && d < b.2 - 1e-12 * self.size) || (!inside && b.3 && d < b.2 - 1e-6 * self.size),
            };
            if better {
                best = Some((i, xi, d, inside));
            }
        }
        best
    }
}

fn tie(m: &FeModel, kw: &RawKeyword) -> Result<Vec<Box<dyn Interaction>>> {
    let active = geom::element_active(m);
    let dim = m.dim();
    let name = format!("*TIE {} ({})", kw.param("NAME").unwrap_or(""), kw.location());
    let ty = kw.param_upper("TYPE").unwrap_or_else(|| "SURFACE TO SURFACE".into());
    let no_rot = kw.has("NO ROTATION");
    let tol_given = kw.param("POSITION TOLERANCE").and_then(parse_f64);
    let mut out = Vec::new();
    let mut asi = 0;
    for d in kw.records() {
        let f: Vec<&str> = d.fields.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        if f.is_empty() {
            continue;
        }
        if f.len() < 2 {
            return kw.err_at(d.line, "write slave surface, master surface");
        }
        // acoustics: acoustic–structural pairs are coupled in interactions/acoustic.rs
        if super::acoustic::is_acoustic_pair(m, kw, f[0], f[1])? {
            asi += 1;
            continue;
        }
        let slave = geom::surface(m, kw, f[0])?;
        let msurf = geom::surface(m, kw, f[1])?;
        if msurf.kind == "NODE" {
            return kw.err("the master surface must be element-based");
        }
        let master = Master::new(m, geom::facets(m, msurf, &name)?);
        let tol = tol_given.unwrap_or(0.05 * master.size);
        let mut mnodes: Vec<usize> = master.facets.iter().flat_map(|f| f.nodes.iter().copied()).collect();
        mnodes.sort_unstable();
        mnodes.dedup();
        let snodes = geom::surface_nodes(m, slave)?;
        // Constraint per slave node A: Σ_B D_AB u_B − Σ_C M_AC u_C = 0 (D for slave
        // nodes, M for master nodes). Node-to-surface: D = I and M = the master
        // surface shape functions at the slave node's projection. Surface-to-surface
        // (mortar): D_AB = ∫ N_A N_B dA and M_AC = ∫ N_A M_C(proj) dA over the slave surface.
        let mut rows: Vec<(usize, Vec<(usize, f64)>)> = Vec::new();
        let s2s = ty == "SURFACE TO SURFACE" && slave.kind != "NODE";
        if s2s {
            // Mortar with an element-wise dual basis Φ on the slave surface: the
            // constraint for slave node A becomes u_A = Σ_C (∫ Φ_A M_C dA / ∫ N_A dA) u_C,
            // i.e. each slave node is coupled only to nearby master nodes.
            use std::collections::BTreeMap;
            let mut acc: BTreeMap<usize, (f64, BTreeMap<usize, f64>)> = BTreeMap::new();
            let mpts: Vec<[f64; 3]> = mnodes.iter().map(|&n| m.nodes[n]).collect();
            for sf in geom::facets(m, slave, &name)? {
                let xs = geom::coords(m, &sf, None);
                let nn = xs.len();
                let (a, area) = geom::dual_basis(&sf, &xs);
                for (k, &sn) in sf.nodes.iter().enumerate() {
                    acc.entry(sn).or_default().0 += area[k];
                }
                for (xi, wg) in geom::mortar_points(&sf, &xs, &mpts, 8) {
                    let fp = geom::eval(&sf, &xs, xi);
                    let Some((mi, mxi, dist, _)) = master.closest(fp.x) else { continue };
                    if dist > tol {
                        continue;
                    }
                    let mf = &master.facets[mi];
                    let (nm, _, _) = geom::shape(mf.dim, mf.nodes.len(), mxi);
                    for (k, &sn) in sf.nodes.iter().enumerate() {
                        let phi: f64 = (0..nn).map(|b| a[k * nn + b] * fp.n[b]).sum();
                        let wa = phi * wg * fp.jac;
                        let e = acc.entry(sn).or_default();
                        for (j, &mn) in mf.nodes.iter().enumerate() {
                            *e.1.entry(mn).or_default() += wa * nm[j];
                        }
                    }
                }
            }
            for (sn, (area, map)) in acc {
                let tot: f64 = map.values().sum();
                // slave nodes not (fully) covered by the master surface are not tied
                if area <= 0.0 || (tot / area - 1.0).abs() > 1e-6 {
                    continue;
                }
                let mut r = vec![(sn, 1.0)];
                r.extend(map.into_iter().filter(|&(_, c)| c.abs() > 1e-12 * area).map(|(n, c)| (n, -c / area)));
                rows.push((sn, r));
            }
        } else {
            for &sn in &snodes {
                let Some((mi, mxi, dist, _)) = master.closest(m.nodes[sn]) else { continue };
                if dist > tol {
                    continue;
                }
                let mf = &master.facets[mi];
                let (nm, _, _) = geom::shape(mf.dim, mf.nodes.len(), mxi);
                let mut r = vec![(sn, 1.0)];
                r.extend(mf.nodes.iter().copied().zip(nm).filter(|p| p.1.abs() > 1e-12).map(|(n, c)| (n, -c)));
                rows.push((sn, r));
            }
        }
        for (sn, row) in rows {
            if mnodes.binary_search(&sn).is_ok() {
                continue; // the slave node is also a master node (shared node)
            }
            let mut dofs: Vec<Dof> = geom::trans_dofs(dim).to_vec();
            if !no_rot {
                dofs.extend(geom::rot_dofs(dim).iter().filter(|&&d| row.iter().all(|&(n, _)| geom::has(&active, n, d))));
            }
            for d in dofs {
                if !geom::has(&active, sn, d) {
                    continue;
                }
                let t: Vec<(usize, Dof, f64)> = row.iter().filter(|&&(n, _)| geom::has(&active, n, d)).map(|&(n, c)| (n, d, c)).collect();
                out.push(LinearConstraint { terms: t, rhs: 0.0, force_slave: true });
            }
        }
    }
    if out.is_empty() && asi > 0 {
        return Ok(Vec::new());
    }
    if out.is_empty() {
        return fail(format!("{name}: no slave nodes lie within the position tolerance"));
    }
    boxed(name, out, Vec::new())
}
