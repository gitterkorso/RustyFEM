//! Tauri-skalet: kommandon som anropar app-API:t (`rustyfem-session`).
//!
//! JSON-kommandon har typer genererade med specta (`web/src/engine/bindings.ts`).
//! Stora data (nät, fält, förskjutningar, symboler) skickas som råa bytes med
//! `tauri::ipc::Response`; de kommandona registreras utanför specta och
//! deklareras för hand i `TauriEngine`.

pub mod mcp;

use rustyfem_session::edit::Edit;
use rustyfem_session::fields::StepResultInfo;
use rustyfem_session::info::ModelInfo;
use rustyfem_session::memory::MemoryEstimate;
use rustyfem_session::units::UnitSystem;
use rustyfem_session::{PickInfo, RunEvent, Session, SessionInfo};
use serde::Serialize;
use specta::Type;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::ipc::{Channel, Response};
use tauri::State;

#[derive(Default)]
pub struct Sessions {
    next: u32,
    open: HashMap<u32, Session>,
    running: HashMap<u32, Arc<AtomicBool>>,
    /// Sessionen som visas i appen (den senast öppnade); MCP-verktygen arbetar i den.
    active: Option<u32>,
}

pub type AppState = Arc<Mutex<Sessions>>;
pub type Res<T> = Result<T, String>;

fn with<T>(st: &AppState, id: u32, f: impl FnOnce(&mut Session) -> Res<T>) -> Res<T> {
    let mut g = st.lock().map_err(|_| "intern: låset är förgiftat".to_string())?;
    let s = g.open.get_mut(&id).ok_or_else(|| format!("session {id} finns inte"))?;
    f(s)
}

#[derive(Serialize, Type, Clone)]
pub struct Opened {
    id: u32,
    session: SessionInfo,
    model: ModelInfo,
}

#[derive(Serialize, Type)]
struct Edited {
    session: SessionInfo,
    model: ModelInfo,
    /// Nätet ändrades (hämta det igen).
    geometry: bool,
}

fn insert(st: &AppState, s: Session) -> Res<Opened> {
    let mut g = st.lock().map_err(|e| e.to_string())?;
    g.next += 1;
    let id = g.next;
    let out = Opened { id, session: s.info(), model: rustyfem_session::info::model_info(s.model()) };
    g.open.insert(id, s);
    g.active = Some(id);
    Ok(out)
}

#[tauri::command]
#[specta::specta]
fn new_model(st: State<'_, AppState>, script: Option<String>) -> Res<Opened> {
    insert(&st, Session::new_fem(script.as_deref().unwrap_or(rustyfem_session::NEW_MODEL))?)
}

#[tauri::command]
#[specta::specta]
async fn open_file(st: State<'_, AppState>, path: String) -> Res<Opened> {
    let s = tauri::async_runtime::spawn_blocking(move || Session::open(&PathBuf::from(path))).await.map_err(|e| e.to_string())??;
    insert(&st, s)
}

#[tauri::command]
#[specta::specta]
fn close_session(st: State<'_, AppState>, id: u32) -> Res<()> {
    let mut g = st.lock().map_err(|e| e.to_string())?;
    if let Some(c) = g.running.remove(&id) {
        c.store(true, Ordering::Relaxed);
    }
    g.open.remove(&id);
    if g.active == Some(id) {
        g.active = None;
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
fn save_project(st: State<'_, AppState>, id: u32, path: String) -> Res<SessionInfo> {
    with(&st, id, |s| {
        let p = PathBuf::from(&path);
        let bytes = if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("rusty")) {
            s.info().script.ok_or("bara .rusty-modeller kan sparas som skript")?.into_bytes()
        } else {
            s.to_project()
        };
        std::fs::write(&p, bytes).map_err(|e| format!("kunde inte skriva {path}: {e}"))?;
        if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case("rusty")) {
            s.path = Some(p);
            s.dirty = false;
        }
        Ok(s.info())
    })
}

#[tauri::command]
#[specta::specta]
fn session_info(st: State<'_, AppState>, id: u32) -> Res<SessionInfo> {
    with(&st, id, |s| Ok(s.info()))
}

#[tauri::command]
#[specta::specta]
fn set_units(st: State<'_, AppState>, id: u32, units: UnitSystem) -> Res<SessionInfo> {
    with(&st, id, |s| {
        s.set_units(units);
        Ok(s.info())
    })
}

/// Enhetssystemen som går att välja, med etikett.
#[tauri::command]
#[specta::specta]
fn unit_systems() -> Vec<(UnitSystem, String)> {
    UnitSystem::ALL.iter().map(|u| (*u, u.label().to_string())).collect()
}

#[tauri::command]
#[specta::specta]
fn model_info(st: State<'_, AppState>, id: u32) -> Res<ModelInfo> {
    with(&st, id, |s| Ok(rustyfem_session::info::model_info(s.model())))
}

fn edited(s: &Session, geometry: bool) -> Edited {
    Edited { session: s.info(), model: rustyfem_session::info::model_info(s.model()), geometry }
}

#[tauri::command]
#[specta::specta]
fn apply_edit(st: State<'_, AppState>, id: u32, edit: Edit) -> Res<Edited> {
    with(&st, id, |s| {
        let g = s.apply(edit)?;
        Ok(edited(s, g))
    })
}

#[tauri::command]
#[specta::specta]
fn undo(st: State<'_, AppState>, id: u32) -> Res<Edited> {
    with(&st, id, |s| {
        let g = s.undo()?;
        Ok(edited(s, g))
    })
}

#[tauri::command]
#[specta::specta]
fn redo(st: State<'_, AppState>, id: u32) -> Res<Edited> {
    with(&st, id, |s| {
        let g = s.redo()?;
        Ok(edited(s, g))
    })
}

#[tauri::command]
#[specta::specta]
fn solvers() -> Vec<String> {
    rustyfem::solver::all().iter().map(|s| s.name().to_string()).collect()
}

#[tauri::command]
#[specta::specta]
async fn estimate(st: State<'_, AppState>, id: u32, solver: Option<String>) -> Res<MemoryEstimate> {
    let model = with(&st, id, |s| Ok(s.model().clone()))?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    tauri::async_runtime::spawn_blocking(move || rustyfem_session::memory::estimate(&model, rustyfem_session::platform::budget(), solver.as_deref(), threads))
        .await
        .map_err(|e| e.to_string())
}

/// Startar lösningen på en egen tråd. Framsteg och slutbesked kommer via `events`.
#[tauri::command]
#[specta::specta]
fn solve(st: State<'_, AppState>, id: u32, solver: Option<String>, all_frames: bool, events: Channel<RunEvent>) -> Res<()> {
    start_run(&st, id, solver, all_frames, move |e| {
        let _ = events.send(e);
    })
}

/// Startar en körning på en egen tråd (används av gränssnittet och MCP).
pub fn start_run(st: &AppState, id: u32, solver: Option<String>, all_frames: bool, emit: impl Fn(RunEvent) + Send + Sync + Clone + 'static) -> Res<()> {
    let state: AppState = st.clone();
    let (model, cancel) = {
        let mut g = st.lock().map_err(|e| e.to_string())?;
        if g.running.contains_key(&id) {
            return Err("en körning pågår redan".into());
        }
        let s = g.open.get(&id).ok_or("sessionen finns inte")?;
        if let Some(p) = s.problem() {
            return Err(format!("modellen kan inte lösas än: {p}"));
        }
        let model = s.model().clone();
        let cancel = Arc::new(AtomicBool::new(false));
        g.running.insert(id, cancel.clone());
        (model, cancel)
    };
    std::thread::Builder::new()
        .name(format!("rustyfem-run-{id}"))
        .spawn(move || {
            let t0 = std::time::Instant::now();
            emit(RunEvent::Started {
                solver: solver.clone().unwrap_or_else(|| "direct".into()),
                note: String::new(),
                steps: model.steps.len() as u32,
            });
            let ch = emit.clone();
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rustyfem_session::run_analysis(&model, solver, all_frames, cancel.clone(), move |e| ch(e))))
                .unwrap_or_else(|p| Err(format!("intern: lösaren avbröts oväntat ({})", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
            let cancelled = cancel.load(Ordering::Relaxed);
            let seconds = t0.elapsed().as_secs_f64();
            let (ok, message) = match res {
                Ok(out) => {
                    let msg = format!("{} obekanta, {} steg", out.n_eq, out.steps.len());
                    if let Ok(mut g) = state.lock() {
                        if let Some(s) = g.open.get_mut(&id) {
                            s.set_results(out);
                        }
                    }
                    (true, msg)
                }
                Err(e) => (false, e),
            };
            if let Ok(mut g) = state.lock() {
                g.running.remove(&id);
            }
            emit(RunEvent::Finished { ok, cancelled, seconds, message });
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
#[specta::specta]
fn cancel(st: State<'_, AppState>, id: u32) -> Res<()> {
    let g = st.lock().map_err(|e| e.to_string())?;
    if let Some(c) = g.running.get(&id) {
        c.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
fn results(st: State<'_, AppState>, id: u32) -> Res<Vec<StepResultInfo>> {
    with(&st, id, |s| Ok(s.catalog().unwrap_or_default()))
}

#[tauri::command]
#[specta::specta]
fn history(st: State<'_, AppState>, id: u32, step: u32, index: u32) -> Res<Vec<(f64, f64)>> {
    with(&st, id, |s| s.history(step as usize, index as usize))
}

#[tauri::command]
#[specta::specta]
fn pick(st: State<'_, AppState>, id: u32, kind: String, index: u32, step: Option<u32>, frame: Option<u32>) -> Res<PickInfo> {
    let at = step.zip(frame).map(|(a, b)| (a as usize, b as usize));
    with(&st, id, |s| if kind == "node" { s.node_info(index as usize, at) } else { s.element_info(index as usize, at) })
}

#[tauri::command]
#[specta::specta]
fn system_memory() -> f64 {
    rustyfem_session::platform::total()
}

#[tauri::command]
#[specta::specta]
fn mcp_status(m: State<'_, Arc<mcp::Mcp>>) -> mcp::McpStatus {
    m.status()
}

/// Startar den lokala MCP-servern (`port` = null: sparad eller standardport).
#[tauri::command]
#[specta::specta]
fn mcp_start(m: State<'_, Arc<mcp::Mcp>>, port: Option<u16>) -> Res<mcp::McpStatus> {
    m.inner().start(port.unwrap_or(0))
}

#[tauri::command]
#[specta::specta]
fn mcp_stop(m: State<'_, Arc<mcp::Mcp>>) -> mcp::McpStatus {
    m.stop()
}

/// Lägger till RustyFEM i Claude Desktops konfiguration; returnerar filerna som skrevs.
#[tauri::command]
#[specta::specta]
fn mcp_install_claude_desktop() -> Res<Vec<String>> {
    mcp::install_claude_desktop()
}

/// Katalog för självtestet (`rustyfem-app --selftest KATALOG`), annars inget.
static SELFTEST: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Ska gränssnittet köra självtestet?
#[tauri::command]
#[specta::specta]
fn selftest() -> bool {
    if let Some(d) = SELFTEST.get() {
        eprintln!("självtest startar, resultat i {}", d.display());
    }
    SELFTEST.get().is_some()
}

/// Självtestet är klart: skriver skärmbilden (PNG som data-URL) och
/// rapporten (JSON) och avslutar appen.
#[tauri::command]
#[specta::specta]
fn selftest_done(app: tauri::AppHandle, png: String, report: String) -> Res<()> {
    let dir = SELFTEST.get().ok_or("inget självtest")?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let b64 = png.split_once(',').map(|x| x.1).unwrap_or(&png);
    std::fs::write(dir.join("selftest.png"), base64_decode(b64)?).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("selftest.json"), report).map_err(|e| e.to_string())?;
    app.exit(0);
    Ok(())
}

fn base64_decode(s: &str) -> Res<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc = 0u32;
    let mut bits = 0;
    for c in s.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        acc = (acc << 6) | val(c).ok_or("ogiltig base64")?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

// --- binära svar (utanför specta) ---

#[tauri::command]
fn mesh(st: State<'_, AppState>, id: u32) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(rustyfem_session::mesh::pack(s.model(), s.mesh()))))
}

#[tauri::command]
fn symbols(st: State<'_, AppState>, id: u32, step: Option<u32>) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(rustyfem_session::info::symbols(s.model(), step.map(|x| x as usize)))))
}

#[tauri::command]
fn field(st: State<'_, AppState>, id: u32, step: u32, frame: u32, field: String, component: String) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(s.field_values(step as usize, frame as usize, &field, &component)?)))
}

#[tauri::command]
fn displacement(st: State<'_, AppState>, id: u32, step: u32, frame: u32) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(s.displacement(step as usize, frame as usize)?)))
}

const BINARY: [&str; 4] = ["mesh", "symbols", "field", "displacement"];

fn specta_builder() -> tauri_specta::Builder<tauri::Wry> {
    tauri_specta::Builder::<tauri::Wry>::new().commands(tauri_specta::collect_commands![
        new_model,
        open_file,
        close_session,
        save_project,
        session_info,
        set_units,
        unit_systems,
        model_info,
        apply_edit,
        undo,
        redo,
        solvers,
        estimate,
        solve,
        cancel,
        results,
        history,
        pick,
        system_memory,
        selftest,
        selftest_done,
        mcp_status,
        mcp_start,
        mcp_stop,
        mcp_install_claude_desktop,
    ])
}

/// Skriver TS-typerna (i debug-byggen vid start, eller med `rustyfem-app --export-bindings FIL`).
pub fn export_bindings(path: &str) -> Result<(), String> {
    // skriv bara om innehållet ändrats, annars startar `tauri dev` om appen i en slinga
    let tmp = std::env::temp_dir().join(format!("rustyfem-bindings-{}.ts", std::process::id()));
    specta_builder()
        .export(specta_typescript::Typescript::default().header("// Genererad från Rust med tauri-specta – ändra inte för hand.
// @ts-nocheck
"), &tmp)
        .map_err(|e| e.to_string())?;
    let new = std::fs::read(&tmp).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&tmp);
    if std::fs::read(path).ok().as_deref() != Some(new.as_slice()) {
        std::fs::write(path, new).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Startar appen i självtestläge: gränssnittet kör ett fast testförlopp och
/// skriver skärmbild och tider till `dir`.
pub fn run_selftest(dir: PathBuf) {
    let _ = SELFTEST.set(dir);
    run_app()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run_app() {
    let builder = specta_builder();
    #[cfg(debug_assertions)]
    if let Err(e) = export_bindings(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/src/engine/bindings.ts")) {
        eprintln!("kunde inte skriva bindings.ts: {e}");
    }
    let typed = builder.invoke_handler();
    let binary: Box<dyn Fn(tauri::ipc::Invoke<tauri::Wry>) -> bool + Send + Sync> = Box::new(tauri::generate_handler![mesh, symbols, field, displacement]);
    let state: AppState = Arc::new(Mutex::new(Sessions::default()));
    let server = mcp::Mcp::new(state.clone());
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage::<AppState>(state)
        .manage(server.clone())
        .setup(move |app| {
            server.set_app(app.handle().clone());
            // MCP-servern startas om den var på förra gången eller om stdio-bryggan startade appen
            if std::env::var_os("RUSTYFEM_MCP_AUTOSTART").is_some() || mcp::load_config().enabled {
                let _ = server.start(0);
            }
            Ok(())
        })
        .invoke_handler(move |invoke| if BINARY.contains(&invoke.message.command()) { binary(invoke) } else { typed(invoke) })
        .run(tauri::generate_context!())
        .expect("kunde inte starta RustyFEM");
}

pub fn run() {
    run_app()
}

