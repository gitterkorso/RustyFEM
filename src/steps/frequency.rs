//! *FREQUENCY: natural frequencies and mode shapes (linear perturbation step).
//!
//! Solves (K₀ − ω²M) φ = 0 for the lowest eigenvalues with block
//! subspace iteration with shift-invert (`solver::eigen`), where K₀ is
//! the tangent stiffness in the base state (for a geometrically nonlinear
//! base state the tangent contains the stress stiffness) and M the mass matrix.
//! The shift σ is chosen so that K₀ − σM is positive definite: σ = 0 if possible,
//! otherwise a negative shift (rigid-body modes, free structures). A specified
//! shift (data line, (cycles/time)²) is used if the matrix is then positive definite.
//!
//! Output:
//! - frame 0 = the base state, frame i = mode i (`value` = mode number),
//!   description with eigenvalue and frequency,
//! - mode shapes normalized with respect to mass. NORMALIZATION=DISPLACEMENT (largest
//!   displacement component = 1) applies only with SIM=NO: otherwise
//!   SIM-based extraction is used, which mass-normalizes anyway (with a warning),
//! - history (region "Assembly ASSEMBLY", time = mode number): EIGVAL, EIGFREQ,
//!   GM (generalized mass), PF1–PF6 (participation factors, rotations about
//!   the origin) and EM1–EM6 (effective mass = PF² · GM),
//! - RF = K φ − λ M φ in constrained degrees of freedom.
//!
//! Mass matrices: diagonal for first-order elements, consistent for
//! higher order, shells and B23/B33 (see `mass_kind`).
//! B23/B33 have two internal degrees of freedom per element (cubic axial interpolation,
//! see `elements::beam_mass`) that are part of the eigenvalue problem but not of the
//! output. Therefore the eigenvalue problem is assembled here in a separate numbering
//! (`AllDofs`): free nodal degrees of freedom, internal degrees of freedom, constrained.
//!
//! The helper functions reused by `buckle.rs` are also here.

use super::api::{StepCtx, StepType};
use crate::elements::{beam_mass, ElementType, Family, MassKind};
use crate::fem::assembly::{self, DofMap, Prepared, NONE};
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{parse_f64, RawKeyword, StepDef};
use crate::fem::output::FrameData;
use crate::fem::state::{SolutionState, StepLoads};
use crate::model::{fail, FemError, Result};
use crate::solver::eigen::{self, combine, factor_sym, EigenOptions, Factored, Recorder, SymCsr, Triplets};
use crate::solver::{LinearSolver, SystemMatrix};

pub struct Frequency;

pub static STEP_TYPES: &[&dyn StepType] = &[&Frequency];

pub const HISTORY_REGION: &str = "Assembly ASSEMBLY";

// ---------------------------------------------------------------------------
// Shared helper functions (also used by buckle.rs)
// ---------------------------------------------------------------------------

/// Numbering of all degrees of freedom in the eigenvalue problem: the free
/// nodal degrees of freedom first (same numbers as in `DofMap`), then the elements'
/// internal degrees of freedom (B23/B33), last the constrained nodal degrees of freedom.
pub struct AllDofs {
    pub idx: Vec<[usize; SLOTS]>,
    /// (node, slot) per degree of freedom; internal degrees of freedom have node = `NONE`.
    pub owner: Vec<(usize, usize)>,
    pub n_free: usize,
    /// Number of internal degrees of freedom (numbered n_free .. n_free + n_int).
    pub n_int: usize,
    /// First internal degree of freedom per element (`NONE` if the element has none).
    pub elem_int: Vec<usize>,
}

impl AllDofs {
    pub fn new(p: &Prepared, dm: &DofMap) -> Self {
        let mut owner = dm.owner.clone();
        let mut elem_int = vec![NONE; p.model.elements.len()];
        for (e, el) in p.model.elements.iter().enumerate() {
            if has_internal(el.etype) {
                elem_int[e] = owner.len();
                for k in 0..beam_mass::N_INTERNAL {
                    owner.push((NONE, k));
                }
            }
        }
        let n_int = owner.len() - dm.n_eq;
        let mut idx = dm.eq.clone();
        for (n, a) in dm.active.iter().enumerate() {
            for s in 0..SLOTS {
                if a & (1 << s) != 0 && idx[n][s] == NONE {
                    idx[n][s] = owner.len();
                    owner.push((n, s));
                }
            }
        }
        AllDofs { idx, owner, n_free: dm.n_eq, n_int, elem_int }
    }

    /// All degrees of freedom.
    pub fn n(&self) -> usize {
        self.owner.len()
    }

    /// The degrees of freedom in the eigenvalue problem (free + internal).
    pub fn n_solve(&self) -> usize {
        self.n_free + self.n_int
    }

    /// Nodal vector from a vector in this numbering (internal degrees of freedom are skipped).
    pub fn to_nodes(&self, x: &[f64], nn: usize) -> Vec<NodeVec> {
        let mut u = vec![[0.0; SLOTS]; nn];
        for (q, &(n, s)) in self.owner.iter().enumerate().take(x.len()) {
            if n != NONE {
                u[n][s] = x[q];
            }
        }
        u
    }

    /// Vector from nodal values (internal degrees of freedom = 0).
    pub fn from_nodes(&self, u: &[NodeVec]) -> Vec<f64> {
        self.owner.iter().map(|&(n, s)| if n == NONE { 0.0 } else { u[n][s] }).collect()
    }

    /// Degree-of-freedom numbers for element `e`: the nodal degrees of freedom and, if
    /// `with_internal`, the internal ones.
    fn map(&self, p: &Prepared, e: usize, with_internal: bool) -> Vec<usize> {
        let mut v: Vec<usize> = p.edofs[e].iter().map(|&(n, s)| self.idx[n][s]).collect();
        if with_internal && self.elem_int[e] != NONE {
            v.extend((0..beam_mass::N_INTERNAL).map(|k| self.elem_int[e] + k));
        }
        v
    }

    /// Solver blocks: a node with free degrees of freedom or an element's
    /// internal degrees of freedom.
    pub fn blocks(&self, p: &Prepared) -> Vec<([f64; 3], Vec<usize>)> {
        let m = p.model;
        let mut per_node: Vec<Vec<usize>> = vec![Vec::new(); m.nodes.len()];
        for q in 0..self.n_free {
            per_node[self.owner[q].0].push(q);
        }
        let mut out: Vec<([f64; 3], Vec<usize>)> =
            per_node.into_iter().enumerate().filter(|(_, v)| !v.is_empty()).map(|(n, v)| (m.nodes[n], v)).collect();
        for (e, &f) in self.elem_int.iter().enumerate() {
            if f != NONE {
                let el = &m.elements[e];
                let mut c = [0.0; 3];
                for &n in &el.nodes {
                    for d in 0..3 {
                        c[d] += m.nodes[n][d] / el.nodes.len() as f64;
                    }
                }
                out.push((c, (f..f + beam_mass::N_INTERNAL).collect()));
            }
        }
        out
    }
}

/// Elements with internal degrees of freedom in eigenvalue steps (B23/B33).
fn has_internal(et: &dyn ElementType) -> bool {
    matches!(et.name(), "B23" | "B33")
}

/// Mass matrix type in eigenvalue steps: diagonal
/// (row-summed) mass for first-order elements (CPS4, CPE4, CPS3,
/// C3D8, T2D2, T3D2, ...), consistent mass for
/// higher-order elements, for the cubic beams B23/B33 and for all
/// shells (S4, S4R, S3, S3R: tests/cases/modal/s4_platta_modal,
/// tests/cases/nlgeom_struct/freq_s4r_platta and others).
pub fn mass_kind(et: &dyn ElementType) -> MassKind {
    let n = et.n_nodes();
    let first_order = match et.family() {
        Family::Truss => n == 2,
        Family::Continuum2D | Family::Axisymmetric | Family::Membrane => n <= 4,
        // area nlgeom_struct: consistent mass for shells
        Family::Shell => false,
        Family::Continuum3D => n <= 8,
        Family::Beam => n == 2 && !matches!(et.name().get(..3), Some("B23") | Some("B33")),
        _ => true,
    };
    if first_order { MassKind::Lumped } else { MassKind::Consistent }
}

fn scatter_map(t: &mut Triplets, map: &[usize], ke: &[f64]) {
    let nd = map.len();
    for a in 0..nd {
        for b in 0..nd {
            let (ia, ib) = (map[a], map[b]);
            if ia != NONE && ib != NONE && ia >= ib {
                t.add(ia, ib, ke[a * nd + b]);
            }
        }
    }
}

fn scatter(t: &mut Triplets, p: &Prepared, all: &AllDofs, e: usize, ke: &[f64]) {
    scatter_map(t, &all.map(p, e, false), ke);
}

/// Internal degrees of freedom for element `e` (B23/B33), with geometric stiffness if `u` is given.
fn internal(p: &Prepared, all: &AllDofs, e: usize, u: Option<&[NodeVec]>) -> Result<Option<beam_mass::BeamInternal>> {
    if all.elem_int[e] == NONE {
        return Ok(None);
    }
    let m = p.model;
    let ctx = p.ctx(e, false, 0.0, 0.0);
    let ue = u.map(|u| p.gather(e, u));
    match beam_mass::internal_dofs(m.elements[e].etype.name(), &ctx, ue.as_deref()) {
        Some(r) => r.map(Some).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e)))),
        None => Ok(None),
    }
}

/// The mass matrix in the numbering `all` (with internal degrees of freedom).
pub fn assemble_mass(p: &Prepared, all: &AllDofs) -> Result<SymCsr> {
    let m = p.model;
    let mut t = Triplets::new(all.n());
    for e in 0..m.elements.len() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let et = m.elements[e].etype;
        let me = et.mass(&ctx, mass_kind(et)).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        scatter(&mut t, p, all, e, &me);
        if let Some(bi) = internal(p, all, e, None)? {
            scatter_map(&mut t, &all.map(p, e, true), &bi.m);
        }
    }
    // Point masses and rotary inertia use the interaction interface. Assemble
    // only one triangle, as for the symmetric element mass matrices above.
    for interaction in &p.interactions {
        for (na, da, nb, db, value) in interaction.mass() {
            let ia = all.idx[na][crate::fem::dof::slot_of(da)];
            let ib = all.idx[nb][crate::fem::dof::slot_of(db)];
            if ia != NONE && ib != NONE && ia >= ib {
                t.add(ia, ib, value);
            }
        }
    }
    Ok(t.build())
}

/// Geometric stiffness K_σ for the displacement `u` and the element state `states`.
pub fn assemble_geometric(p: &Prepared, all: &AllDofs, u: &[NodeVec], states: &[Vec<f64>]) -> Result<SymCsr> {
    let m = p.model;
    let mut t = Triplets::new(all.n());
    for e in 0..m.elements.len() {
        let ctx = p.ctx(e, false, 0.0, 0.0);
        let et = m.elements[e].etype;
        let ue = p.gather(e, u);
        let empty = Vec::new();
        let st = states.get(e).unwrap_or(&empty);
        let kg = et.geometric_stiffness(&ctx, &ue, st).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
        scatter(&mut t, p, all, e, &kg);
        if let Some(bi) = internal(p, all, e, Some(u))? {
            if let Some(kg) = &bi.kg {
                scatter_map(&mut t, &all.map(p, e, true), kg);
            }
        }
    }
    Ok(t.build())
}

/// The additional stiffness from the internal degrees of freedom (all degrees of freedom).
fn assemble_internal_stiffness(p: &Prepared, all: &AllDofs) -> Result<SymCsr> {
    let mut t = Triplets::new(all.n());
    for e in 0..p.model.elements.len() {
        if let Some(bi) = internal(p, all, e, None)? {
            scatter_map(&mut t, &all.map(p, e, true), &bi.k);
        }
    }
    Ok(t.build())
}

/// The tangent stiffness K₀ in the base state for the eigenvalue problem (free + internal
/// degrees of freedom): the core assembly (elements and interactions) plus the
/// contribution of the internal degrees of freedom.
pub fn assemble_stiffness(p: &Prepared, dm: &DofMap, all: &AllDofs, state: &SolutionState, nlgeom: bool) -> Result<SymCsr> {
    let zero = vec![[0.0; SLOTS]; p.model.nodes.len()];
    let mut rec = Recorder::new(dm.n_eq);
    assembly::internal(p, dm, &state.u, &zero, &state.elem_state, nlgeom, state.time, 0.0, Some(&mut rec))?;
    let kff = rec.build();
    let kint = assemble_internal_stiffness(p, all)?;
    let ns = all.n_solve();
    Ok(combine(&[(&kff, 1.0), (&kint, 1.0)], ns))
}

/// Factorizes A (free + internal degrees of freedom). `None` if A is not positive
/// definite or so close to singular that the solution becomes inaccurate (tested with
/// a random right-hand side).
pub fn factor_checked(a: &SymCsr, all: &AllDofs, p: &Prepared, solver: &dyn LinearSolver) -> Result<Option<Factored>> {
    let f = match factor_sym(a, &all.blocks(p), solver)? {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };
    let n = a.n;
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let b: Vec<f64> = (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect();
    let mut x = b.clone();
    f.solve(&mut x);
    let r = a.mul(&x);
    let bmax = b.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(1e-300);
    let err = r.iter().zip(&b).fold(0.0f64, |s, (ri, bi)| s.max((ri - bi).abs()));
    Ok((err <= 1e-6 * bmax).then_some(f))
}

/// Median of k_ii / m_ii over degrees of freedom with mass (scale for eigenvalues).
fn diag_ratio(k: &SymCsr, m: &SymCsr) -> f64 {
    let (kd, md) = (k.diag(), m.diag());
    let mut r: Vec<f64> = kd.iter().zip(&md).filter(|(a, b)| **a > 0.0 && **b > 0.0).map(|(a, b)| a / b).collect();
    if r.is_empty() {
        return 1.0;
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r[r.len() / 2]
}

/// Normalizes so that the largest displacement component (translation) becomes +1.
pub fn normalize_displacement(phi: &mut [NodeVec], active: &[u16]) {
    let mut best = 0.0f64;
    for (n, v) in phi.iter().enumerate() {
        for s in 0..3 {
            if active[n] & (1 << s) != 0 && v[s].abs() > best.abs() * (1.0 + 1e-6) {
                best = v[s];
            }
        }
    }
    if best != 0.0 {
        for v in phi.iter_mut() {
            v.iter_mut().for_each(|x| *x /= best);
        }
    }
}

/// Frame for a mode shape: U = φ, RF = (Kφ − fext) in constrained degrees of freedom,
/// element output (S, E) for the stress produced by the mode shape.
#[allow(clippy::too_many_arguments)]
pub fn mode_frame(p: &Prepared, dm: &DofMap, phi: &[NodeVec], fext: &[NodeVec], time: f64, mode: usize, description: String) -> Result<FrameData> {
    let init = p.init_states()?;
    let int = assembly::internal(p, dm, phi, phi, &init, false, time, 0.0, None)?;
    let mut fr = assembly::frame(p, dm, phi, &int.fint, fext, &int.state_new, false, time, mode as f64, description)?;
    fr.mode = mode as i64;
    Ok(fr)
}

/// Frame 0: the base state.
pub fn base_frame(p: &Prepared, dm: &DofMap, state: &SolutionState) -> Result<FrameData> {
    let zero = vec![[0.0; SLOTS]; p.model.nodes.len()];
    let mut fr = assembly::frame(p, dm, &state.u, &state.reaction, &zero, &state.elem_state, state.nlgeom, state.time, 0.0, "Increment      0: Base State".into())?;
    fr.mode = 0;
    Ok(fr)
}

/// Numbers in frame descriptions: 5 significant digits with a decimal point
/// ("16475.", "630.20") for 0.1 ≤ |v| < 10⁵, otherwise 6 significant digits with
/// exponent ("1.07441E+06").
pub fn g5(v: f64) -> String {
    if v == 0.0 {
        return "0.0000".into();
    }
    let e = v.abs().log10().floor() as i32;
    if (-1..5).contains(&e) {
        let d = (4 - e).max(0) as usize;
        let s = format!("{:.*}", d, v);
        // rounding can add a digit (99999.5 → 100000): exponent form then
        if s.trim_start_matches('-').replace('.', "").trim_start_matches('0').len() > 5 && d == 0 {
            return exp6(v);
        }
        if d == 0 { format!("{s}.") } else { s }
    } else {
        exp6(v)
    }
}

fn exp6(v: f64) -> String {
    let s = format!("{:.5E}", v);
    match s.split_once('E') {
        Some((m, x)) => {
            let x: i32 = x.parse().unwrap_or(0);
            format!("{m}E{}{:02}", if x < 0 { '-' } else { '+' }, x.abs())
        }
        None => s,
    }
}

/// Field `i` (0-based) on the first data line as a number, `None` if empty.
pub fn data_field(kw: &RawKeyword, i: usize) -> Result<Option<f64>> {
    let Some(d) = kw.data.first() else { return Ok(None) };
    match d.fields.get(i).map(|s| s.trim()) {
        None | Some("") => Ok(None),
        Some(s) => parse_f64(s).map(Some).ok_or_else(|| FemError(format!("{} line {} (*{}): '{s}' is not a number", kw.file, d.line, kw.name))),
    }
}

// ---------------------------------------------------------------------------
// *FREQUENCY
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct FreqOptions {
    nev: Option<usize>,
    min_freq: Option<f64>,
    max_freq: Option<f64>,
    /// Shift in (rad/time)².
    shift: Option<f64>,
    vectors: usize,
    mass_norm: bool,
}

fn parse_options(kw: &RawKeyword) -> Result<FreqOptions> {
    let solver = kw.param_upper("EIGENSOLVER").unwrap_or_else(|| "LANCZOS".into());
    let two_pi_sq = (2.0 * std::f64::consts::PI).powi(2);
    let nev = data_field(kw, 0)?.map(|v| v.round().max(0.0) as usize).filter(|&n| n > 0);
    let mut o = FreqOptions { nev, min_freq: None, max_freq: None, shift: None, vectors: 0, mass_norm: true };
    match solver.as_str() {
        "LANCZOS" | "AMS" => {
            o.min_freq = data_field(kw, 1)?;
            o.max_freq = data_field(kw, 2)?;
            o.shift = data_field(kw, 3)?.map(|s| s * two_pi_sq);
        }
        "SUBSPACE" => {
            o.max_freq = data_field(kw, 1)?;
            o.shift = data_field(kw, 2)?.map(|s| s * two_pi_sq);
            o.vectors = data_field(kw, 3)?.map_or(0, |v| v.round().max(0.0) as usize);
        }
        other => return kw.err(format!("EIGENSOLVER={other} is not supported (LANCZOS, SUBSPACE or AMS)")),
    }
    match kw.param_upper("NORMALIZATION").as_deref() {
        None | Some("MASS") => {}
        // SIM-based eigenvalue extraction (default for all
        // solvers) can only mass-normalize; DISPLACEMENT applies only with SIM=NO
        Some("DISPLACEMENT") => o.mass_norm = kw.param_upper("SIM").as_deref() != Some("NO"),
        Some(x) => return kw.err(format!("NORMALIZATION={x} must be MASS or DISPLACEMENT")),
    }
    if o.nev.is_none() && o.max_freq.is_none() {
        return kw.err("specify the number of eigenvalues or the highest frequency on the data line");
    }
    Ok(o)
}

/// Rigid-body vector for direction j (0–2 translation, 3–5 rotation about the origin).
/// Internal degrees of freedom (axial strain) are zero in a rigid-body motion.
pub fn rigid(all: &AllDofs, x: &[[f64; 3]], j: usize) -> Vec<f64> {
    all.owner
        .iter()
        .map(|&(n, s)| {
            if n == NONE {
                return 0.0;
            }
            if j < 3 {
                if s == j { 1.0 } else { 0.0 }
            } else {
                let a = j - 3;
                if s == j {
                    return 1.0;
                }
                if s >= 3 {
                    return 0.0;
                }
                // (e_a × x)_s
                let p = x[n];
                let e = [(a == 0) as i32 as f64, (a == 1) as i32 as f64, (a == 2) as i32 as f64];
                let c = [e[1] * p[2] - e[2] * p[1], e[2] * p[0] - e[0] * p[2], e[0] * p[1] - e[1] * p[0]];
                c[s]
            }
        })
        .collect()
}

/// The result of the eigenvalue extraction in a *FREQUENCY step: the numbering,
/// the mass matrix (all degrees of freedom) and the eigenpairs (λ, vector in the `all` numbering
/// without the constrained degrees of freedom, not yet normalized).
pub struct Extracted {
    pub dm: DofMap,
    pub all: AllDofs,
    pub mfull: SymCsr,
    pub modes: Vec<(f64, Vec<f64>)>,
    pub iterations: usize,
    pub mass_norm: bool,
}

/// The eigenvalue extraction for the *FREQUENCY data line `kw` with the boundary conditions in
/// `loads` and the base state `base`. Used by `Frequency::run` and by the
/// modal steps (`modal.rs`), which thereby get exactly the same modes as the
/// *FREQUENCY step.
pub fn extract(p: &Prepared, kw: &RawKeyword, loads: &StepLoads, base: &SolutionState, nlgeom: bool, solver: &'static dyn LinearSolver, verbose: bool) -> Result<Extracted> {
    let o = parse_options(kw)?;
    let (dm, _) = assembly::dofmap(p, &loads.constrained(), solver)?;
    let all = AllDofs::new(p, &dm);
    let ns = all.n_solve();
    let mfull = assemble_mass(p, &all)?;
    let ms = mfull.leading(ns);
    if ms.val.iter().all(|v| *v == 0.0) {
        return fail("*FREQUENCY requires mass (*DENSITY in the materials)");
    }
    // K₀: the tangent in the base state (with NLGEOM it contains the stress stiffness)
    let k0 = assemble_stiffness(p, &dm, &all, base, nlgeom)?;
    let k0 = crate::elements::infinite::frequency_stiffness(p, &all.idx, k0)?; // area elements2: infinite elements without stiffness
    // shift: specified, otherwise 0; if K − σM is not positive definite or
    // nearly singular (rigid-body modes) negative shifts −s are tried with s
    // relative to stiffness/mass on the diagonal
    let try_shift = |c: f64| -> Result<Option<Factored>> { factor_checked(&combine(&[(&k0, 1.0), (&ms, -c)], ns), &all, p, solver) };
    let scale = diag_ratio(&k0, &ms);
    let mut candidates: Vec<f64> = Vec::new();
    if let Some(s0) = o.shift {
        candidates.push(s0);
    }
    if o.shift.is_none_or(|s0| s0 > 0.0) {
        candidates.push(0.0);
    }
    candidates.extend((0..6).map(|k| -scale * 10f64.powi(-10 + 2 * k)));
    let mut found: Option<(f64, Factored)> = None;
    for c in candidates {
        if let Some(f) = try_shift(c)? {
            found = Some((c, f));
            break;
        }
    }
    let Some((mut sigma, mut mat)) = found else {
        return fail("*FREQUENCY: the stiffness matrix could not be factorized even with a negative shift (unstable base state?)");
    };
    let mmul = |x: &[f64]| ms.mul(x);
    if sigma < 0.0 && o.shift.is_none() {
        // rigid-body modes: a small shift gives rounding noise ~ ε·λ₁/|σ| in the
        // elastic modes. Rough estimate first, then shift ≈ 0.1·λ₁.
        let rough = EigenOptions { nev: o.nev.unwrap_or(10).min(ns), vectors: o.vectors, max_iter: 60, tol: 1e-6, allow_unconverged: true };
        let r = eigen::subspace(ns, &mat, &mmul, &rough)?;
        let elastic = r.theta.iter().map(|t| sigma + 1.0 / t).filter(|l| *l > 10.0 * sigma.abs()).fold(f64::INFINITY, f64::min);
        if elastic.is_finite() {
            let c = -0.1 * elastic;
            if let Some(f) = try_shift(c)? {
                sigma = c;
                mat = f;
            }
        }
    }
    if verbose {
        eprintln!("  shift σ = {sigma:.4e}, {ns} unknowns ({} internal)", all.n_int);
    }
    // eigenvalues: with only the highest frequency the count is increased until the limit is passed
    let two_pi = 2.0 * std::f64::consts::PI;
    let lam_max = o.max_freq.map(|f| (two_pi * f).powi(2));
    let lam_min = o.min_freq.map(|f| (two_pi * f).powi(2));
    let mut want = o.nev.unwrap_or(10).min(ns);
    let res = loop {
        let opts = EigenOptions { nev: want, vectors: o.vectors, ..Default::default() };
        let r = eigen::subspace(ns, &mat, &mmul, &opts)?;
        let top = r.theta.last().map(|t| sigma + 1.0 / t);
        // with a lowest frequency, `nev` modes must lie above the limit
        let below = lam_min.map_or(0, |lm| r.theta.iter().filter(|t| sigma + 1.0 / **t < lm).count());
        let full = r.theta.len() < want || want >= ns;
        let enough = match (o.nev, lam_max, top) {
            (Some(n), _, _) => r.theta.len() - below >= n || full,
            (None, Some(lm), Some(t)) => t > lm || full,
            _ => true,
        };
        if enough {
            break r;
        }
        want = match o.nev {
            Some(n) => (n + below).max(want + 1).min(ns),
            None => (want * 2).min(ns),
        };
    };
    let mut modes: Vec<(f64, Vec<f64>)> = res.theta.iter().zip(res.vectors).map(|(t, v)| (sigma + 1.0 / t, v)).collect();
    modes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    modes.retain(|(l, _)| lam_max.is_none_or(|m| *l <= m) && lam_min.is_none_or(|m| *l >= m));
    if let Some(n) = o.nev {
        modes.truncate(n);
    }
    Ok(Extracted { dm, all, mfull, modes, iterations: res.iterations, mass_norm: o.mass_norm })
}

impl StepType for Frequency {
    fn name(&self) -> &'static str {
        "FREQUENCY"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "FREQUENCY").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        let model = p.model;
        let nn = model.nodes.len();
        let kw = ctx.step.procedure.as_ref().unwrap();
        let base = &*ctx.state;
        let ex = extract(p, kw, ctx.loads, base, ctx.nlgeom, ctx.solver, ctx.opts.verbose)?;
        let Extracted { dm, all, mfull, modes, iterations, mass_norm } = ex;
        let two_pi = 2.0 * std::f64::consts::PI;
        // output
        ctx.out.frames.push(base_frame(p, &dm, base)?);
        let time = base.time;
        let rig: Vec<Vec<f64>> = (0..6).map(|j| rigid(&all, &model.nodes, j)).collect();
        let mrig: Vec<Vec<f64>> = rig.iter().map(|r| mfull.mul(r)).collect();
        let kint = if all.n_int > 0 { Some(assemble_internal_stiffness(p, &all)?) } else { None };
        let names = ["EIGVAL", "EIGFREQ", "GM", "PF1", "PF2", "PF3", "PF4", "PF5", "PF6", "EM1", "EM2", "EM3", "EM4", "EM5", "EM6"];
        let mut hist: Vec<Vec<(f64, f64)>> = vec![Vec::new(); names.len()];
        let dim3 = model.dim() == 3;
        for (i, (lam, v)) in modes.iter().enumerate() {
            let mode = i + 1;
            let mut x = v.clone();
            x.resize(all.n(), 0.0);
            // mass normalization (internal degrees of freedom included)
            let gm0 = mfull.bilinear(&x, &x);
            if gm0 > 0.0 {
                let f = 1.0 / gm0.sqrt();
                x.iter_mut().for_each(|y| *y *= f);
            }
            if !mass_norm {
                // largest displacement component = +1; the same factor on all
                // degrees of freedom, including the internal ones
                let mut phi = all.to_nodes(&x, nn);
                let before = phi.clone();
                normalize_displacement(&mut phi, &dm.active);
                let f = (0..x.len())
                    .find(|&q| all.owner[q].0 != NONE && x[q] != 0.0)
                    .map_or(1.0, |q| {
                        let (n, s) = all.owner[q];
                        phi[n][s] / before[n][s]
                    });
                x.iter_mut().for_each(|y| *y *= f);
            }
            let phi = all.to_nodes(&x, nn);
            let gm = mfull.bilinear(&x, &x);
            let freq = lam.max(0.0).sqrt() / two_pi;
            // RF = K φ − λ M φ: the elements' internal forces for φ give K φ without the
            // internal degrees of freedom's contribution, which is added via fext
            let mx = mfull.mul(&x);
            let mut f: Vec<f64> = mx.iter().map(|v| v * lam).collect();
            if let Some(k) = &kint {
                f.iter_mut().zip(k.mul(&x)).for_each(|(a, b)| *a -= b);
            }
            let fext: Vec<NodeVec> = all.to_nodes(&f, nn);
            let desc = format!("Mode {mode:>9}: Value = {:>11} Freq = {:>11} (cycles/time)", g5(*lam), g5(freq));
            ctx.out.frames.push(mode_frame(p, &dm, &phi, &fext, time, mode, desc)?);
            let t = mode as f64;
            hist[0].push((t, *lam));
            hist[1].push((t, freq));
            hist[2].push((t, gm));
            for j in 0..6 {
                let used = dim3 || j == 0 || j == 1 || j == 5;
                let pf = if used && gm > 0.0 { eigen::dot(&x, &mrig[j]) / gm } else { 0.0 };
                hist[3 + j].push((t, pf));
                hist[9 + j].push((t, pf * pf * gm));
            }
        }
        for (name, data) in names.iter().zip(hist) {
            ctx.out.history.push((HISTORY_REGION.to_string(), name.to_string(), data));
        }
        ctx.out.increments = 1;
        ctx.out.iterations = iterations;
        Ok(())
    }
}
