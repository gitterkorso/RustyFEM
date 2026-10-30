//! *SUBSTRUCTURE GENERATE: static condensation to retained degrees of freedom.
//!
//! The step builds the stiffness K (and with MASS MATRIX=YES the mass M) in
//! the base state, splits the degrees of freedom into retained r (*RETAINED NODAL DOFS)
//! and internal i (locked degrees of freedom from *BOUNDARY are removed) and condenses:
//!
//! ```text
//! X = K_ii⁻¹ K_ir,   K_red = K_rr − K_ri X,   T = [I; −X],   M_red = Tᵀ M T
//! ```
//!
//! (Guyan reduction of the mass, default without retained eigenmodes).
//! The substructure is stored in the in-process registry and, with a job context, as
//! `<job>_<Zn>.rfsub` in the job directory. It is used as an element with
//! *ELEMENT, TYPE=Zn, FILE=<job> (`elements/matrix_element.rs`); if
//! neither the registry entry nor the file exists but `<job>.inp` is in the same directory,
//! the generation runs first (in the process). The retained nodes are ordered by
//! increasing node number, as required for the element's nodes.
//!
//! Limitations: no load cases (*SUBSTRUCTURE LOAD CASE), no retained
//! eigenmodes, no recovery of internal results (stresses in the substructure),
//! no constraints (MPC/*TIE) coupling retained degrees of freedom.

use super::api::{StepCtx, StepType};
use crate::elements::MassKind;
use crate::fem::assembly::{self, Prepared, NONE};
use crate::fem::dof::{self, Dof};
use crate::fem::model::{FeModel, StepDef};
use crate::inp::InpCtx;
use crate::interactions::{InteractionCtx, TangentSink};
use crate::model::{fail, FemError, Result};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

pub struct Generate;

pub static STEP_TYPES: &[&dyn StepType] = &[&Generate];

/// A generated substructure.
#[derive(Debug, Clone)]
pub struct SubLib {
    /// The job that generated the substructure (upper case, empty without a job context).
    pub source: String,
    /// Serial number (distinguishes different generations in the element cache).
    pub serial: u64,
    /// The type name, e.g. "Z1".
    pub name: String,
    /// Retained nodes (instance, number) and coordinates, by increasing node number.
    pub labels: Vec<(String, i64)>,
    pub coords: Vec<[f64; 3]>,
    /// Retained degrees of freedom per node (increasing).
    pub dofs: Vec<Vec<Dof>>,
    /// Condensed stiffness and mass, dense and row-major.
    pub k: Vec<f64>,
    pub m: Option<Vec<f64>>,
}

fn registry() -> &'static Mutex<HashMap<(String, String), Arc<SubLib>>> {
    static R: OnceLock<Mutex<HashMap<(String, String), Arc<SubLib>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_serial() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static S: AtomicU64 = AtomicU64::new(1);
    S.fetch_add(1, Ordering::Relaxed)
}

/// Puts the substructure in the registry.
pub fn register(lib: SubLib) -> Arc<SubLib> {
    let a = Arc::new(lib);
    registry().lock().unwrap_or_else(|e| e.into_inner()).insert((a.source.clone(), a.name.clone()), a.clone());
    a
}

impl SubLib {
    /// The text format `rustyfem-substructure 1`.
    pub fn to_text(&self) -> String {
        let mut s = String::from("# rustyfem-substructure 1\n");
        let _ = writeln!(s, "NAME;{}", self.name);
        for (i, (inst, l)) in self.labels.iter().enumerate() {
            let x = self.coords[i];
            let d: Vec<String> = self.dofs[i].iter().map(|d| d.to_string()).collect();
            let _ = writeln!(s, "NODE;{inst};{l};{:.17e};{:.17e};{:.17e};{}", x[0], x[1], x[2], d.join(","));
        }
        let n: usize = self.dofs.iter().map(|d| d.len()).sum();
        for (tag, a) in [("K", Some(&self.k)), ("M", self.m.as_ref())] {
            let Some(a) = a else { continue };
            for i in 0..n {
                for j in 0..=i {
                    let v = a[i * n + j];
                    if v != 0.0 {
                        let _ = writeln!(s, "{tag};{};{};{:.17e}", i + 1, j + 1, v);
                    }
                }
            }
        }
        s
    }

    pub fn parse(text: &str, source: &str) -> Result<SubLib> {
        let mut lib = SubLib { source: source.to_uppercase(), serial: next_serial(), name: String::new(), labels: vec![], coords: vec![], dofs: vec![], k: vec![], m: None };
        let mut ent: Vec<(bool, usize, usize, f64)> = Vec::new();
        for (no, line) in text.lines().enumerate() {
            let t = line.trim();
            if t.is_empty() || t.starts_with('#') {
                continue;
            }
            let p: Vec<&str> = t.split(';').collect();
            let bad = || FemError(format!("substructure file line {}: unreadable line '{t}'", no + 1));
            let num = |s: &str| s.trim().parse::<f64>().map_err(|_| bad());
            let int = |s: &str| s.trim().parse::<i64>().map_err(|_| bad());
            match p[0] {
                "NAME" if p.len() >= 2 => lib.name = p[1].trim().to_uppercase(),
                "NODE" if p.len() >= 7 => {
                    lib.labels.push((p[1].to_string(), int(p[2])?));
                    lib.coords.push([num(p[3])?, num(p[4])?, num(p[5])?]);
                    lib.dofs.push(p[6].split(',').map(|d| int(d).map(|x| x as Dof)).collect::<Result<_>>()?);
                }
                "K" | "M" if p.len() >= 4 => ent.push((p[0] == "M", int(p[1])? as usize - 1, int(p[2])? as usize - 1, num(p[3])?)),
                _ => return Err(bad()),
            }
        }
        let n: usize = lib.dofs.iter().map(|d| d.len()).sum();
        lib.k = vec![0.0; n * n];
        let mut m = vec![0.0; n * n];
        let mut has_m = false;
        for (is_m, i, j, v) in ent {
            if i >= n || j >= n {
                return fail("the substructure file has matrix indices outside the retained degrees of freedom");
            }
            let a = if is_m {
                has_m = true;
                &mut m
            } else {
                &mut lib.k
            };
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        lib.m = has_m.then_some(m);
        Ok(lib)
    }
}

/// The substructure for *ELEMENT, TYPE=`ty`, FILE=`file` (see the module documentation).
pub fn find(ctx: &InpCtx, file: Option<&str>, ty: &str) -> Result<Arc<SubLib>> {
    let src = file.map(|f| f.trim_end_matches(".sim").trim_end_matches(".SIM").to_uppercase()).unwrap_or_default();
    let get = |src: &str| registry().lock().unwrap_or_else(|e| e.into_inner()).get(&(src.to_string(), ty.to_string())).cloned();
    let Some(f) = file else {
        // without FILE=: the most recently generated substructure with the type name
        if let Some(l) = get("") {
            return Ok(l);
        }
        let any = registry().lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|((_, t), _)| t == ty).map(|(_, v)| v.clone()).max_by_key(|l| l.serial);
        return any.ok_or_else(|| FemError(format!("the substructure {ty} does not exist (give FILE=the job that generated it)")));
    };
    let stem = f.trim_end_matches(".sim").trim_end_matches(".SIM");
    let mut dirs = Vec::new();
    if let Some(j) = super::restart::job() {
        dirs.push(j.dir);
    }
    if let Some(d) = &ctx.dir {
        dirs.push(d.clone());
    }
    dirs.push(std::path::PathBuf::from("."));
    for d in &dirs {
        let p = d.join(format!("{stem}_{ty}.rfsub"));
        if let Ok(text) = std::fs::read_to_string(&p) {
            let lib = SubLib::parse(&text, stem).map_err(|e| FemError(format!("{}: {}", p.display(), e.0)))?;
            return Ok(register(lib));
        }
    }
    if let Some(l) = get(&src) {
        return Ok(l);
    }
    // the generation job is next to it: run it first
    for d in &dirs {
        let p = d.join(format!("{stem}.inp"));
        if p.exists() {
            run_generation(&p)?;
            if let Some(l) = get(&src) {
                return Ok(l);
            }
            return fail(format!("{} did not generate the substructure {ty}", p.display()));
        }
    }
    fail(format!("the substructure {ty} from the job {stem} does not exist ({stem}_{ty}.rfsub or {stem}.inp is missing)"))
}

/// Runs a generation job in the process (the files are written to a temporary directory).
pub fn run_generation(inp: &Path) -> Result<()> {
    let mut job = super::restart::Job::for_file(inp);
    job.dir = std::env::temp_dir().join("rustyfem-job").join(format!("{}", std::process::id()));
    let _ = std::fs::create_dir_all(&job.dir);
    job.deck = super::restart::file_deck(inp).ok();
    let _scope = super::restart::scope(job);
    let m = crate::inp::read_file(inp)?;
    super::run(&m.model, &super::RunOptions::default()).map_err(|e| FemError(format!("{}: {}", inp.display(), e.0)))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Assembly of elements and interactions into (node, slot) pairs

/// The element's tangent stiffness in the state (linear stiffness for linear elements).
pub fn element_stiffness(p: &Prepared, e: usize, state: &crate::fem::state::SolutionState, nlgeom: bool) -> Result<Vec<f64>> {
    let el = &p.model.elements[e];
    let ctx = p.ctx(e, nlgeom, state.time, 0.0);
    let linear = !nlgeom && ctx.material.is_none_or(|m| m.is_linear());
    if linear {
        if let Ok(k) = el.etype.stiffness(&ctx) {
            return Ok(k);
        }
    }
    let u = p.gather(e, &state.u);
    let du = vec![0.0; u.len()];
    let old = state.elem_state.get(e).cloned().unwrap_or_default();
    let mut new = old.clone();
    let r = el.etype.internal_force(&ctx, &u, &du, &old, &mut new, true).map_err(|err| FemError(format!("{}: {err}", p.model.elem_name(e))))?;
    match r.k {
        Some(k) => Ok(k),
        None => el.etype.stiffness(&ctx).map_err(|err| FemError(format!("{}: {err}", p.model.elem_name(e)))),
    }
}

/// Sparse matrices keyed on ((node, slot), (node, slot)), both triangles.
pub type PairMatrix = BTreeMap<((usize, usize), (usize, usize)), f64>;

struct PairSink<'a>(&'a mut PairMatrix);

impl TangentSink for PairSink<'_> {
    fn add(&mut self, na: usize, da: Dof, nb: usize, db: Dof, v: f64) {
        *self.0.entry(((na, dof::slot_of(da)), (nb, dof::slot_of(db)))).or_default() += v;
    }
}

/// Stiffness (and mass) for the whole model in the base state, without boundary conditions.
/// `elems` restricts to certain elements (all if `None`).
pub fn assemble_pairs(p: &Prepared, state: &crate::fem::state::SolutionState, nlgeom: bool, mass: bool, elems: Option<&BTreeSet<usize>>) -> Result<(PairMatrix, Option<PairMatrix>)> {
    let m = p.model;
    let mut k = PairMatrix::new();
    let mut mm = PairMatrix::new();
    for e in 0..m.elements.len() {
        if elems.is_some_and(|s| !s.contains(&e)) || p.activity.is_removed(e) {
            continue;
        }
        let ed = &p.edofs[e];
        let ke = element_stiffness(p, e, state, nlgeom)?;
        let n = ed.len();
        for a in 0..n {
            for b in 0..n {
                let v = ke[a * n + b];
                if v != 0.0 || a == b {
                    *k.entry((ed[a], ed[b])).or_default() += v;
                }
            }
        }
        if mass {
            let ctx = p.ctx(e, nlgeom, state.time, 0.0);
            let et = m.elements[e].etype;
            let kind: MassKind = super::frequency::mass_kind(et);
            let me = et.mass(&ctx, kind).map_err(|err| FemError(format!("{}: {err}", m.elem_name(e))))?;
            for a in 0..n {
                for b in 0..n {
                    let v = me[a * n + b];
                    if v != 0.0 {
                        *mm.entry((ed[a], ed[b])).or_default() += v;
                    }
                }
            }
        }
    }
    if elems.is_none() {
        let ictx = InteractionCtx { model: m, time: state.time, dt: 0.0, nlgeom };
        let mut fint = vec![[0.0; dof::SLOTS]; m.nodes.len()];
        for it in &p.interactions {
            it.contribute(&ictx, &state.u, &mut fint, Some(&mut PairSink(&mut k)))?;
            if mass {
                for (na, da, nb, db, v) in it.mass() {
                    *mm.entry(((na, dof::slot_of(da)), (nb, dof::slot_of(db)))).or_default() += v;
                }
            }
        }
    }
    Ok((k, mass.then_some(mm)))
}

// ---------------------------------------------------------------------------
// The step

/// Retained (node, slot) from *RETAINED NODAL DOFS.
fn retained(step: &StepDef, prep: &Prepared) -> Result<BTreeSet<(usize, usize)>> {
    let mut set = BTreeSet::new();
    for kw in step.extra.iter().filter(|k| k.name == "RETAINED NODAL DOFS") {
        for d in &kw.data {
            let n: usize = d.fields[0].parse().map_err(|_| FemError(format!("{}: internal error in the node list", kw.location())))?;
            let dd: Dof = d.fields[1].parse().map_err(|_| FemError(format!("{}: internal error in the node list", kw.location())))?;
            let s = dof::slot_of(dd);
            if prep.active[n] & (1 << s) == 0 {
                return kw.err(format!("node {} does not have degree of freedom {dd}", prep.model.node_name(n)));
            }
            set.insert((n, s));
        }
    }
    if set.is_empty() {
        return fail("*SUBSTRUCTURE GENERATE requires *RETAINED NODAL DOFS");
    }
    Ok(set)
}

fn label_order(m: &FeModel, a: usize, b: usize) -> std::cmp::Ordering {
    (m.node_inst[a], m.node_label[a]).cmp(&(m.node_inst[b], m.node_label[b]))
}

impl StepType for Generate {
    fn name(&self) -> &'static str {
        "SUBSTRUCTURE GENERATE"
    }

    fn claim(&self, step: &StepDef, _nlgeom: bool) -> Option<i32> {
        (step.procedure_name() == "SUBSTRUCTURE GENERATE").then_some(0)
    }

    fn run(&self, ctx: &mut StepCtx) -> Result<()> {
        let p = ctx.prep;
        let m = p.model;
        let kw = ctx.step.procedure.as_ref().unwrap();
        let ty = match kw.param_upper("TYPE").or_else(|| kw.param_upper("NAME")) {
            Some(t) if t.strip_prefix('Z').and_then(|s| s.parse::<u32>().ok()).is_some() => t,
            Some(t) => return kw.err(format!("TYPE={t} must be Zn (n = integer)")),
            None => return kw.err("TYPE=Zn is missing"),
        };
        if kw.flag("RETAINED EIGENMODES", false).unwrap_or(false) || ctx.step.extra.iter().any(|k| k.name == "SELECT EIGENMODES" || k.name == "SUBSTRUCTURE LOAD CASE") {
            return kw.err("retained eigenmodes and load cases (*SUBSTRUCTURE LOAD CASE) are not supported");
        }
        let with_mass = kw.flag("MASS MATRIX", false)?;
        let ret = retained(ctx.step, p)?;
        let mut cons: BTreeSet<(usize, usize)> = ctx.loads.bc.keys().copied().collect();
        for r in &ret {
            if cons.contains(r) {
                return kw.err(format!("degree of freedom {} at node {} is both retained and locked", dof::SLOT_DOF[r.1], m.node_name(r.0)));
            }
        }
        cons.extend(ret.iter().copied());
        let (dm, mut kii) = assembly::dofmap(p, &cons, ctx.solver)?;
        let (kp, mp) = assemble_pairs(p, ctx.state, ctx.nlgeom, with_mass, None)?;
        // retained degrees of freedom by increasing node number, slot
        let mut rlist: Vec<(usize, usize)> = ret.iter().copied().collect();
        rlist.sort_by(|a, b| label_order(m, a.0, b.0).then(a.1.cmp(&b.1)));
        let ridx: HashMap<(usize, usize), usize> = rlist.iter().enumerate().map(|(i, &k)| (k, i)).collect();
        let nr = rlist.len();
        let ni = dm.n_eq;
        // K_ii in the system matrix, K_ir and K_rr dense
        let mut kir = vec![vec![0.0; ni]; nr];
        let mut krr = vec![0.0; nr * nr];
        for (&(a, b), &v) in &kp {
            match (ridx.get(&a), ridx.get(&b)) {
                (Some(&ra), Some(&rb)) => krr[ra * nr + rb] += v,
                (None, Some(&rb)) => dm.for_each_eq(a.0, a.1, |q, c| kir[rb][q] += c * v),
                (None, None) => dm.add_entry(kii.as_mut(), a.0, a.1, b.0, b.1, v),
                (Some(_), None) => {}
            }
        }
        let mut x = vec![vec![0.0; ni]; nr];
        if ni > 0 {
            kii.factor().map_err(|q| assembly::mechanism_message(p, &dm, q))?;
            for j in 0..nr {
                let mut b = kir[j].clone();
                kii.solve(&mut b);
                x[j] = b;
            }
        }
        let mut kred = vec![0.0; nr * nr];
        for i in 0..nr {
            for j in 0..nr {
                let c: f64 = (0..ni).map(|q| kir[i][q] * x[j][q]).sum();
                kred[i * nr + j] = krr[i * nr + j] - c;
            }
        }
        symmetrize(&mut kred, nr);
        let mred = match mp {
            None => None,
            Some(mp) => {
                // M_red = M_rr − M_ri X − Xᵀ M_ir + Xᵀ M_ii X
                let mut mir = vec![vec![0.0; ni]; nr];
                let mut mrr = vec![0.0; nr * nr];
                let mut mii: Vec<(usize, usize, f64)> = Vec::new();
                for (&(a, b), &v) in &mp {
                    match (ridx.get(&a), ridx.get(&b)) {
                        (Some(&ra), Some(&rb)) => mrr[ra * nr + rb] += v,
                        (None, Some(&rb)) => dm.for_each_eq(a.0, a.1, |q, c| mir[rb][q] += c * v),
                        (None, None) => dm.for_each_eq(a.0, a.1, |qa, ca| dm.for_each_eq(b.0, b.1, |qb, cb| mii.push((qa, qb, ca * cb * v)))),
                        (Some(_), None) => {}
                    }
                }
                let mut out = vec![0.0; nr * nr];
                let mut y = vec![vec![0.0; ni]; nr];
                for j in 0..nr {
                    for &(qa, qb, v) in &mii {
                        y[j][qa] += v * x[j][qb];
                    }
                }
                for i in 0..nr {
                    for j in 0..nr {
                        let a: f64 = (0..ni).map(|q| mir[i][q] * x[j][q]).sum();
                        let b: f64 = (0..ni).map(|q| x[i][q] * mir[j][q]).sum();
                        let c: f64 = (0..ni).map(|q| x[i][q] * y[j][q]).sum();
                        out[i * nr + j] = mrr[i * nr + j] - a - b + c;
                    }
                }
                symmetrize(&mut out, nr);
                Some(out)
            }
        };
        // nodes and degrees of freedom
        let mut labels = Vec::new();
        let mut coords = Vec::new();
        let mut dofs: Vec<Vec<Dof>> = Vec::new();
        let mut last = NONE;
        for &(n, s) in &rlist {
            if n != last {
                labels.push((m.instances[m.node_inst[n] as usize].clone(), m.node_label[n]));
                coords.push(m.nodes[n]);
                dofs.push(Vec::new());
                last = n;
            }
            dofs.last_mut().unwrap().push(dof::SLOT_DOF[s]);
        }
        // the degrees of freedom per node increasing (slot 7 = NT11 comes last)
        let job = super::restart::job();
        let source = job.as_ref().map(|j| j.name.to_uppercase()).unwrap_or_default();
        let lib = SubLib { source, serial: next_serial(), name: ty.clone(), labels, coords, dofs, k: kred, m: mred };
        if let Some(j) = &job {
            let path = j.file(&format!("_{ty}.rfsub"));
            std::fs::write(&path, lib.to_text()).map_err(|e| FemError(format!("could not write {}: {e}", path.display())))?;
        }
        register(lib);
        if ctx.opts.verbose {
            eprintln!("  substructure {ty}: {nr} retained and {ni} internal degrees of freedom");
        }
        ctx.out.increments = 1;
        Ok(())
    }
}

fn symmetrize(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
}
