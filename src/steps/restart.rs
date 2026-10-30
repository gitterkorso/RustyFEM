//! Omstart (*RESTART, WRITE / *RESTART, READ) och jobbkontexten.
//!
//! **Jobbkontexten** (`Job`) talar om var en analys skriver och läser sina
//! filer: jobbkatalog och jobbnamn, den fullständiga
//! indatatexten (för omstartfilen), omstartfilen att läsa (`--oldjob`)
//! och globalmodellens resultat för submodellering
//! (`--global`). Den sätts per tråd med `scope` av CLI:t och
//! regressionstesterna; utan kontext skrivs inga filer.
//!
//! **Omstartfilen** `jobb.rfres` är binär (små-endian) och innehåller
//! indatatexten (med *INCLUDE utvecklade), antal noder och element och för
//! varje sparat steg dess nummer, antal inkrement och hela `SolutionState`
//! (nodvärden, elementtillstånd, tid, NLGEOM, gällande randvillkor och laster,
//! reaktioner och övriga nodvektorer som V och A). Tillståndet sparas vid
//! slutet av varje steg där *RESTART, WRITE gäller (FREQUENCY=0 stänger av;
//! OVERLAY behåller bara det senaste steget).
//!
//! **Omstart**: en indatafil med *RESTART, READ, STEP=n (utan
//! modelldata, bara nya steg) läses med `read_job`: den gamla indatatexten
//! fram till och med steg n slås ihop med den nya filens nyckelord och
//! `steps::run` hoppar över steg 1..n och startar från det sparade
//! tillståndet (`before_run`). Utdata innehåller bara de nya stegen, numrerade
//! från 1.
//!
//! Begränsningar: omstart bara vid slutet av ett steg (INC= måste vara stegets
//! sista inkrement), interaktioners inre tillstånd (kontaktstatus, friktion)
//! och elementborttagning vid skada sparas inte (*MODEL CHANGE spelas upp igen).

use super::api::StepType;
use crate::fem::assembly::Prepared;
use crate::fem::dof::{NodeVec, SLOTS};
use crate::fem::model::{DistLoad, FeModel, RawKeyword};
use crate::fem::output::StepOutput;
use crate::fem::state::SolutionState;
use crate::model::{fail, FemError, Result};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

// område gaps1: *IMPORT från explicit analys (obalanskrafter)
#[path = "import_explicit.rs"]
mod import_explicit;

/// Inga egna steg-typer.
pub static STEP_TYPES: &[&dyn StepType] = &[];

/// Var ett jobb skriver och läser sina filer (se modulens dokumentation).
#[derive(Debug, Clone, Default)]
pub struct Job {
    /// Katalogen där jobbets filer skrivs (omstartfil, .mtx, substrukturer).
    pub dir: PathBuf,
    /// Jobbnamnet (indatafilens stam).
    pub name: String,
    /// Fullständig indatatext (*INCLUDE utvecklade) för omstartfilen.
    pub deck: Option<String>,
    /// Omstartfil att läsa (*RESTART, READ och *IMPORT).
    pub oldjob: Option<PathBuf>,
    /// Globalmodellens resultat (`--out`-fil eller omstartfil) för *SUBMODEL.
    pub global: Option<PathBuf>,
}

impl Job {
    /// Jobb för indatafilen `inp`: katalogen och stammen.
    pub fn for_file(inp: &Path) -> Job {
        Job {
            dir: inp.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(".")),
            name: inp.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "jobb".into()),
            ..Default::default()
        }
    }

    /// Sökväg för en fil som jobbet skriver: `<dir>/<namn><suffix>`.
    pub fn file(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}{suffix}", self.name))
    }
}

thread_local! {
    static JOB: RefCell<Option<Job>> = const { RefCell::new(None) };
}

/// Aktuellt jobb i den här tråden (om något).
pub fn job() -> Option<Job> {
    JOB.with(|j| j.borrow().clone())
}

/// Sätter jobbkontexten tills det returnerade värdet släpps.
pub fn scope(job: Job) -> JobScope {
    let old = JOB.with(|j| j.replace(Some(job)));
    JobScope(old)
}

/// Återställer föregående jobbkontext när den släpps.
pub struct JobScope(Option<Job>);

impl Drop for JobScope {
    fn drop(&mut self) {
        let old = self.0.take();
        JOB.with(|j| *j.borrow_mut() = old);
    }
}

/// Omstartfilens sökväg för ett jobbnamn eller en sökväg med eller utan ändelse.
pub fn restart_path(p: &Path) -> PathBuf {
    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("rfres")) {
        p.to_path_buf()
    } else {
        let mut s = p.as_os_str().to_owned();
        s.push(".rfres");
        PathBuf::from(s)
    }
}

// ---------------------------------------------------------------------------
// Indatatext

fn quote(v: &str) -> String {
    if v.contains(',') || v.contains('=') {
        format!("\"{v}\"")
    } else {
        v.to_string()
    }
}

/// Skriver nyckelordsblock som .inp-text (läses tillbaka till samma block).
pub fn deck_text(blocks: &[RawKeyword]) -> String {
    let mut s = String::new();
    for kw in blocks {
        s.push('*');
        s.push_str(&kw.name);
        for (k, v) in &kw.params {
            s.push_str(", ");
            s.push_str(k);
            if !v.is_empty() {
                s.push('=');
                s.push_str(&quote(v));
            }
        }
        s.push('\n');
        for d in &kw.data {
            if kw.name == "HEADING" {
                s.push_str(&d.fields.join(","));
            } else {
                s.push_str(&d.fields.join(", "));
            }
            if d.continued {
                s.push(',');
            }
            s.push('\n');
        }
    }
    s
}

/// Indatatexten för en fil med *INCLUDE utvecklade.
pub fn file_deck(inp: &Path) -> Result<String> {
    let mut inc = Vec::new();
    Ok(deck_text(&crate::inp::reader::blocks_from_file(inp, &mut inc, 0)?))
}

fn is_read(kw: &RawKeyword) -> bool {
    kw.name == "RESTART" && kw.has("READ")
}

/// Läser indatafilen för ett jobb. Innehåller den *RESTART, READ slås den
/// ihop med indatatexten i `job.oldjob` (se modulens dokumentation).
/// `job.deck` får den fullständiga indatatexten.
pub fn read_job(inp: &Path, job: &mut Job) -> Result<crate::inp::InpModel> {
    let mut inc = Vec::new();
    let blocks = crate::inp::reader::blocks_from_file(inp, &mut inc, 0)?;
    if let Some(imp) = blocks.iter().find(|k| k.name == "IMPORT") {
        return read_import(inp, job, imp.clone(), blocks, inc);
    }
    let Some(read) = blocks.iter().find(|k| is_read(k)) else {
        job.deck = Some(deck_text(&blocks));
        return crate::inp::read_file(inp);
    };
    let Some(old) = job.oldjob.clone() else {
        return read.err("*RESTART, READ kräver en omstartfil (rustyfem run fil.inp --oldjob jobb)");
    };
    let rf = RestartFile::load(&restart_path(&old))?;
    let n = read_step(read)?;
    let mut old_inc = Vec::new();
    let mut old_blocks = crate::inp::reader::blocks_from_str(&rf.deck, "(omstart)", inp.parent().unwrap_or(Path::new(".")), &mut old_inc, 0)?;
    // gamla filen fram till och med *END STEP för steg n
    let mut ends = 0;
    let mut cut = old_blocks.len();
    for (i, k) in old_blocks.iter().enumerate() {
        if k.name == "END STEP" {
            ends += 1;
            if ends == n {
                cut = i + 1;
                break;
            }
        }
    }
    if ends < n {
        return read.err(format!("STEP={n} men den gamla analysen har bara {ends} steg"));
    }
    old_blocks.truncate(cut);
    // tidigare *RESTART, READ i den gamla texten gäller inte längre
    old_blocks.retain(|k| !is_read(k));
    for k in blocks {
        if k.name == "HEADING" {
            continue;
        }
        old_blocks.push(k);
    }
    let text = deck_text(&old_blocks);
    let mut m = crate::inp::read_str(&text, inp.parent().unwrap_or(Path::new(".")))?;
    // bara den nya filens *INCLUDE (för att kopiera den till jobbkatalogen)
    m.included = inc;
    job.deck = Some(text);
    Ok(m)
}

/// *IMPORT: den gamla analysens modelldata (noder, element, set, material,
/// tvärsnitt; inga randvillkor, laster eller begynnelsevillkor) följd av den
/// nya filens nyckelord.
fn read_import(inp: &Path, job: &mut Job, imp: RawKeyword, blocks: Vec<RawKeyword>, inc: Vec<PathBuf>) -> Result<crate::inp::InpModel> {
    let Some(old) = job.oldjob.clone() else {
        return imp.err("*IMPORT kräver den tidigare analysens omstartfil (rustyfem run fil.inp --oldjob jobb)");
    };
    let rf = RestartFile::load(&restart_path(&old))?;
    let mut old_inc = Vec::new();
    let old_blocks = crate::inp::reader::blocks_from_str(&rf.deck, "(import)", inp.parent().unwrap_or(Path::new(".")), &mut old_inc, 0)?;
    let mut all: Vec<RawKeyword> = Vec::new();
    for k in old_blocks {
        if k.name == "STEP" {
            break;
        }
        if matches!(k.name.as_str(), "BOUNDARY" | "RESTART" | "INITIAL CONDITIONS" | "IMPORT") {
            continue;
        }
        all.push(k);
    }
    // område gaps1: UPDATE=YES – det importerade läget blir ny referenskonfiguration
    if imp.flag("UPDATE", false)? {
        let n = match imp.param("STEP").and_then(crate::inp::parse_label) {
            Some(x) if x > 0 => x as usize,
            _ => rf.steps.last().map(|s| s.index).unwrap_or(1),
        };
        let Some(saved) = rf.step(n) else { return imp.err(format!("omstartfilen saknar steg {n}")) };
        import_explicit::move_nodes(&mut all, &saved.state.u).map_err(|e| FemError(format!("{}: {}", imp.location(), e.0)))?;
    }
    for k in blocks {
        if k.name != "HEADING" {
            all.push(k);
        }
    }
    let text = deck_text(&all);
    let mut m = crate::inp::read_str(&text, inp.parent().unwrap_or(Path::new(".")))?;
    m.included = inc;
    job.deck = Some(text);
    Ok(m)
}

fn read_step(kw: &RawKeyword) -> Result<usize> {
    let Some(v) = kw.param("STEP") else { return kw.err("*RESTART, READ saknar STEP=") };
    crate::inp::parse_label(v).filter(|&n| n > 0).map(|n| n as usize).ok_or_else(|| FemError(format!("{}: STEP={v} ska vara ett positivt heltal", kw.location())))
}

// ---------------------------------------------------------------------------
// Binärformat

const MAGIC: &[u8] = b"RUSTYFEM-RESTART 1\n";

#[derive(Default)]
struct W(Vec<u8>);

impl W {
    fn u(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u64).to_le_bytes());
    }
    fn f(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn s(&mut self, v: &str) {
        self.u(v.len());
        self.0.extend_from_slice(v.as_bytes());
    }
    fn fs(&mut self, v: &[f64]) {
        self.u(v.len());
        for &x in v {
            self.f(x);
        }
    }
    fn nv(&mut self, v: &[NodeVec]) {
        self.u(v.len());
        for n in v {
            for &x in n {
                self.f(x);
            }
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl R<'_> {
    fn bytes(&mut self, n: usize) -> Result<&[u8]> {
        if self.p + n > self.b.len() {
            return fail("omstartfilen är trasig eller avkortad");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u(&mut self) -> Result<usize> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()) as usize)
    }
    fn f(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn s(&mut self) -> Result<String> {
        let n = self.u()?;
        Ok(String::from_utf8_lossy(self.bytes(n)?).to_string())
    }
    fn fs(&mut self) -> Result<Vec<f64>> {
        let n = self.u()?;
        (0..n).map(|_| self.f()).collect()
    }
    fn nv(&mut self) -> Result<Vec<NodeVec>> {
        let n = self.u()?;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let mut a = [0.0; SLOTS];
            for x in a.iter_mut() {
                *x = self.f()?;
            }
            v.push(a);
        }
        Ok(v)
    }
}

fn write_state(w: &mut W, s: &SolutionState) {
    w.nv(&s.u);
    w.u(s.elem_state.len());
    for e in &s.elem_state {
        w.fs(e);
    }
    w.f(s.time);
    w.u(s.nlgeom as usize);
    for m in [&s.bcs, &s.cloads] {
        w.u(m.len());
        for (&(n, k), &v) in m {
            w.u(n);
            w.u(k);
            w.f(v);
        }
    }
    w.u(s.dloads.len());
    for ((e, k), l) in &s.dloads {
        w.u(*e);
        w.s(k);
        w.u(l.elem);
        w.s(&l.kind);
        w.f(l.magnitude);
        w.fs(&l.extra);
    }
    w.nv(&s.reaction);
    w.u(s.vectors.len());
    for (k, v) in &s.vectors {
        w.s(k);
        w.nv(v);
    }
}

fn read_state(r: &mut R) -> Result<SolutionState> {
    let mut s = SolutionState { u: r.nv()?, ..Default::default() };
    let ne = r.u()?;
    for _ in 0..ne {
        s.elem_state.push(r.fs()?);
    }
    s.time = r.f()?;
    s.nlgeom = r.u()? != 0;
    for which in 0..2 {
        let n = r.u()?;
        for _ in 0..n {
            let (a, b, v) = (r.u()?, r.u()?, r.f()?);
            if which == 0 { s.bcs.insert((a, b), v) } else { s.cloads.insert((a, b), v) };
        }
    }
    let nd = r.u()?;
    for _ in 0..nd {
        let (e, k) = (r.u()?, r.s()?);
        let l = DistLoad { elem: r.u()?, kind: r.s()?, magnitude: r.f()?, extra: r.fs()? };
        s.dloads.insert((e, k), l);
    }
    s.reaction = r.nv()?;
    let nvec = r.u()?;
    for _ in 0..nvec {
        let k = r.s()?;
        s.vectors.insert(k, r.nv()?);
    }
    Ok(s)
}

/// Ett sparat steg i omstartfilen.
#[derive(Debug, Clone)]
pub struct SavedStep {
    /// Stegets nummer (1-baserat) i analysen.
    pub index: usize,
    pub name: String,
    /// Antal inkrement i steget (för INC=).
    pub increments: usize,
    /// Tillståndet vid stegets slut.
    pub state: SolutionState,
}

/// Innehållet i en omstartfil.
#[derive(Debug, Clone, Default)]
pub struct RestartFile {
    pub deck: String,
    pub n_nodes: usize,
    pub n_elems: usize,
    pub steps: Vec<SavedStep>,
}

impl RestartFile {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut w = W(MAGIC.to_vec());
        w.s(&self.deck);
        w.u(self.n_nodes);
        w.u(self.n_elems);
        w.u(self.steps.len());
        for st in &self.steps {
            w.u(st.index);
            w.s(&st.name);
            w.u(st.increments);
            write_state(&mut w, &st.state);
        }
        w.0
    }

    pub fn from_bytes(b: &[u8]) -> Result<RestartFile> {
        if !b.starts_with(MAGIC) {
            return fail("filen är ingen RustyFEM-omstartfil (rfres)");
        }
        let mut r = R { b, p: MAGIC.len() };
        let mut f = RestartFile { deck: r.s()?, n_nodes: r.u()?, n_elems: r.u()?, ..Default::default() };
        let n = r.u()?;
        for _ in 0..n {
            let (index, name, increments) = (r.u()?, r.s()?, r.u()?);
            f.steps.push(SavedStep { index, name, increments, state: read_state(&mut r)? });
        }
        Ok(f)
    }

    pub fn load(path: &Path) -> Result<RestartFile> {
        let b = std::fs::read(path).map_err(|e| FemError(format!("kunde inte läsa omstartfilen {}: {e}", path.display())))?;
        Self::from_bytes(&b).map_err(|e| FemError(format!("{}: {}", path.display(), e.0)))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, self.to_bytes()).map_err(|e| FemError(format!("kunde inte skriva omstartfilen {}: {e}", path.display())))
    }

    /// Modellen som omstartfilen beskriver (indatatexten läst på nytt).
    pub fn model(&self, dir: &Path) -> Result<FeModel> {
        Ok(crate::inp::read_str(&self.deck, dir)?.model)
    }

    pub fn step(&self, index: usize) -> Option<&SavedStep> {
        self.steps.iter().rev().find(|s| s.index == index)
    }
}

// ---------------------------------------------------------------------------
// Krokar i steps::run

/// Omstartens läge under en analys (skapas av `before_run`).
#[derive(Default)]
pub struct RestartRun {
    /// Antal steg som hoppas över (återställda från omstartfilen).
    pub skip: usize,
    saved: Vec<SavedStep>,
    warned: bool,
}

/// *RESTART, WRITE gäller för steg `i` (0-baserat)? Ger (på, OVERLAY).
fn write_active(model: &FeModel, i: usize) -> (bool, bool) {
    let mut on = (false, false);
    let mut apply = |k: &RawKeyword| {
        if k.name == "RESTART" && k.has("WRITE") {
            let freq = k.param("FREQUENCY").and_then(crate::inp::parse_label).unwrap_or(1);
            let ni = k.param("NUMBER INTERVAL").and_then(crate::inp::parse_label).unwrap_or(1);
            on = (freq != 0 && ni != 0, k.has("OVERLAY"));
        }
    };
    model.extra.iter().for_each(&mut apply);
    for s in model.steps.iter().take(i + 1) {
        s.extra.iter().for_each(&mut apply);
    }
    on
}

/// Före första steget: vid *RESTART, READ läses tillståndet från omstartfilen
/// och de återställda stegen hoppas över.
pub fn before_run(model: &FeModel, prep: &Prepared, state: &mut SolutionState) -> Result<RestartRun> {
    let mut rr = RestartRun::default();
    if let Some(imp) = model.extra.iter().find(|k| k.name == "IMPORT") {
        if import_state(model, imp, state)? {
            import_explicit::set_unbalance(prep, state)?;
        }
        return Ok(rr);
    }
    let Some(read) = model.extra.iter().rev().find(|k| is_read(k)) else { return Ok(rr) };
    let n = read_step(read)?;
    let Some(old) = job().and_then(|j| j.oldjob) else {
        return read.err("*RESTART, READ kräver en omstartfil (rustyfem run fil.inp --oldjob jobb)");
    };
    let path = restart_path(&old);
    let rf = RestartFile::load(&path)?;
    if rf.n_nodes != model.nodes.len() || rf.n_elems != model.elements.len() {
        return read.err(format!(
            "omstartfilen {} har {} noder och {} element, modellen {} och {}",
            path.display(),
            rf.n_nodes,
            rf.n_elems,
            model.nodes.len(),
            model.elements.len()
        ));
    }
    let Some(saved) = rf.step(n) else {
        let have: Vec<String> = rf.steps.iter().map(|s| s.index.to_string()).collect();
        return read.err(format!("omstartfilen {} saknar steg {n} (sparade steg: {})", path.display(), have.join(", ")));
    };
    if let Some(inc) = read.param("INC").and_then(crate::inp::parse_label) {
        if inc as usize != saved.increments && !(read.has("END STEP") && (inc as usize) >= saved.increments) {
            return read.err(format!(
                "omstart inne i ett steg stöds inte: INC={inc}, men steg {n} sparades vid sitt slut (inkrement {})",
                saved.increments
            ));
        }
    }
    if n > model.steps.len() {
        return read.err(format!("STEP={n} men modellen har {} steg", model.steps.len()));
    }
    if saved.state.elem_state.len() != state.elem_state.len() {
        return read.err("omstartfilens elementtillstånd passar inte modellen");
    }
    // *MODEL CHANGE i de återställda stegen spelas upp igen
    let mut scratch = saved.state.clone();
    for s in model.steps.iter().take(n) {
        crate::steps::model_change::begin_step(prep, s, &mut scratch)?;
    }
    *state = saved.state.clone();
    rr.skip = n;
    rr.saved = rf.steps.iter().filter(|s| s.index <= n).cloned().collect();
    Ok(rr)
}

/// *IMPORT, STEP=n, STATE=YES, UPDATE=NO: nodvärden och elementtillstånd vid
/// slutet av steg n i den tidigare analysen. Randvillkor, laster och
/// hastigheter följer inte med; analysens tid börjar om från 0.
/// Returnerar `true` om tillståndet kommer från en explicit analys (STATE=YES):
/// då ska obalanskrafterna avvecklas (`import_explicit.rs`).
fn import_state(model: &FeModel, imp: &RawKeyword, state: &mut SolutionState) -> Result<bool> {
    let Some(old) = job().and_then(|j| j.oldjob) else {
        return imp.err("*IMPORT kräver den tidigare analysens omstartfil (rustyfem run fil.inp --oldjob jobb)");
    };
    // område gaps1: UPDATE=YES – noderna är redan flyttade (`read_import`)
    let update = imp.flag("UPDATE", false)?;
    let with_state = imp.flag("STATE", true)?;
    let path = restart_path(&old);
    let rf = RestartFile::load(&path)?;
    if rf.n_nodes != model.nodes.len() || rf.n_elems != model.elements.len() {
        return imp.err(format!(
            "bara import av hela modellen stöds: omstartfilen {} har {} noder och {} element, modellen {} och {}",
            path.display(),
            rf.n_nodes,
            rf.n_elems,
            model.nodes.len(),
            model.elements.len()
        ));
    }
    // elementseten på dataraderna (omräknade till index av *IMPORT-hanteraren)
    let mut listed = std::collections::BTreeSet::new();
    for d in &imp.data {
        for f in &d.fields {
            if let Ok(e) = f.parse::<usize>() {
                listed.insert(e);
            }
        }
    }
    if !imp.data.is_empty() && listed.len() != model.elements.len() {
        return imp.err(format!("bara import av alla element stöds ({} av {} element i de angivna seten)", listed.len(), model.elements.len()));
    }
    let n = match imp.param("STEP") {
        Some(v) => crate::inp::parse_label(v).filter(|&x| x > 0).ok_or_else(|| FemError(format!("{}: STEP={v} ska vara ett positivt heltal", imp.location())))? as usize,
        None => rf.steps.last().map(|s| s.index).unwrap_or(1),
    };
    let Some(saved) = rf.step(n) else {
        let have: Vec<String> = rf.steps.iter().map(|s| s.index.to_string()).collect();
        return imp.err(format!("omstartfilen {} saknar steg {n} (sparade steg: {})", path.display(), have.join(", ")));
    };
    if imp.param("INTERVAL").is_some() {
        eprintln!("varning: *IMPORT: INTERVAL= ignoreras – tillståndet vid slutet av steg {n} importeras");
    }
    // småtöjningsmaterialen räknar med total töjning: en ny referens kräver
    // den integrerade (NLGEOM) töjningen i tillståndet
    if update && !saved.state.nlgeom {
        return imp.err("UPDATE=YES stöds bara när den tidigare analysen var geometriskt olinjär (NLGEOM=YES) – använd UPDATE=NO");
    }
    state.u = if update { vec![[0.0; SLOTS]; saved.state.u.len()] } else { saved.state.u.clone() };
    if with_state {
        // område gaps1: NLGEOM-tillägget (logaritmisk och integrerad töjning sist i
        // tillståndet, `kinematics::push_ext`) följer med; jämför grunddelen
        let base = |s: &[f64]| s.len() - crate::kinematics::ext(s).map_or(0, |x| x.len() + 2);
        if saved.state.elem_state.len() != state.elem_state.len() || saved.state.elem_state.iter().zip(&state.elem_state).any(|(a, b)| base(a) != base(b)) {
            let first = saved.state.elem_state.iter().zip(&state.elem_state).position(|(a, b)| base(a) != base(b));
            let detail = first.map(|e| format!(" (element {}: {} värden i omstartfilen, {} i modellen)", e + 1, saved.state.elem_state[e].len(), state.elem_state[e].len())).unwrap_or_default();
            return imp.err(format!("elementtillståndet i omstartfilen passar inte modellens element (samma elementtyper och material krävs){detail}"));
        }
        state.elem_state = saved.state.elem_state.clone();
    }
    state.nlgeom = saved.state.nlgeom;
    state.time = 0.0;
    Ok(with_state && import_explicit::deck_is_explicit(&rf.deck))
}

impl RestartRun {
    /// Efter steg `i` (0-baserat): skriv omstartfilen om *RESTART, WRITE gäller.
    pub fn after_step(&mut self, model: &FeModel, i: usize, state: &SolutionState, so: &StepOutput) -> Result<()> {
        let (on, overlay) = write_active(model, i);
        if !on {
            return Ok(());
        }
        let Some(j) = job() else {
            if !self.warned {
                eprintln!("varning: *RESTART, WRITE: ingen jobbkatalog – omstartfilen skrivs inte");
                self.warned = true;
            }
            return Ok(());
        };
        if overlay {
            self.saved.clear();
        }
        self.saved.retain(|s| s.index != i + 1);
        self.saved.push(SavedStep { index: i + 1, name: model.steps[i].name.clone(), increments: so.increments.max(1), state: state.clone() });
        let deck = match &j.deck {
            Some(d) => d.clone(),
            None => {
                if !self.warned {
                    eprintln!("varning: *RESTART, WRITE: indatatexten saknas i jobbkontexten – omstartfilen skrivs inte");
                    self.warned = true;
                }
                return Ok(());
            }
        };
        let rf = RestartFile { deck, n_nodes: model.nodes.len(), n_elems: model.elements.len(), steps: self.saved.clone() };
        rf.save(&j.file(".rfres"))
    }
}
