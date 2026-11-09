//! The Tauri shell: commands that call the app API (`rustyfem-session`).
//!
//! JSON commands have types generated with specta (`web/src/engine/bindings.ts`).
//! Large data (mesh, fields, displacements, symbols) is sent as raw bytes with
//! `tauri::ipc::Response`; those commands are registered outside specta and
//! declared by hand in `TauriEngine`.

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
    /// The session shown in the app (the most recently opened); the MCP tools work in it.
    active: Option<u32>,
}

pub type AppState = Arc<Mutex<Sessions>>;
pub type Res<T> = Result<T, String>;

fn with<T>(st: &AppState, id: u32, f: impl FnOnce(&mut Session) -> Res<T>) -> Res<T> {
    let mut g = st.lock().map_err(|_| "internal: the lock is poisoned".to_string())?;
    let s = g.open.get_mut(&id).ok_or_else(|| format!("session {id} does not exist"))?;
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
    /// The mesh changed (fetch it again).
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
            s.info().script.ok_or("only .rusty models can be saved as scripts")?.into_bytes()
        } else {
            s.to_project()
        };
        std::fs::write(&p, bytes).map_err(|e| format!("could not write {path}: {e}"))?;
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

/// The selectable unit systems, with labels.
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

/// Starts the solution on its own thread. Progress and the final message arrive via `events`.
#[tauri::command]
#[specta::specta]
fn solve(st: State<'_, AppState>, id: u32, solver: Option<String>, all_frames: bool, events: Channel<RunEvent>) -> Res<()> {
    start_run(&st, id, solver, all_frames, move |e| {
        let _ = events.send(e);
    })
}

/// Starts a run on its own thread (used by the UI and MCP).
pub fn start_run(st: &AppState, id: u32, solver: Option<String>, all_frames: bool, emit: impl Fn(RunEvent) + Send + Sync + Clone + 'static) -> Res<()> {
    let state: AppState = st.clone();
    let (model, cancel) = {
        let mut g = st.lock().map_err(|e| e.to_string())?;
        if g.running.contains_key(&id) {
            return Err("a run is already in progress".into());
        }
        let s = g.open.get(&id).ok_or("the session does not exist")?;
        if let Some(p) = s.problem() {
            return Err(format!("the model cannot be solved yet: {p}"));
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
                .unwrap_or_else(|p| Err(format!("internal: the solver stopped unexpectedly ({})", p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default())));
            let cancelled = cancel.load(Ordering::Relaxed);
            let seconds = t0.elapsed().as_secs_f64();
            let (ok, message) = match res {
                Ok(out) => {
                    let msg = format!("{} unknowns, {} steps", out.n_eq, out.steps.len());
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

/// Starts the local MCP server (`port` = null: saved or default port;
/// `network` = null: saved choice, true: reachable from other devices).
#[tauri::command]
#[specta::specta]
fn mcp_start(m: State<'_, Arc<mcp::Mcp>>, port: Option<u16>, network: Option<bool>) -> Res<mcp::McpStatus> {
    m.inner().start(port.unwrap_or(0), network)
}

#[tauri::command]
#[specta::specta]
fn mcp_stop(m: State<'_, Arc<mcp::Mcp>>) -> mcp::McpStatus {
    m.stop()
}

/// Adds RustyFEM to the Claude Desktop configuration; returns the files that were written.
#[tauri::command]
#[specta::specta]
fn mcp_install_claude_desktop() -> Res<Vec<String>> {
    mcp::install_claude_desktop()
}

/// Directory for the self-test (`rustyfem-app --selftest DIR`), otherwise nothing.
static SELFTEST: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();

/// Should the UI run the self-test?
#[tauri::command]
#[specta::specta]
fn selftest() -> bool {
    if let Some(d) = SELFTEST.get() {
        eprintln!("self-test starting, results in {}", d.display());
    }
    SELFTEST.get().is_some()
}

/// The self-test is done: writes the screenshot (PNG as a data URL) and
/// the report (JSON) and exits the app.
#[tauri::command]
#[specta::specta]
fn selftest_done(app: tauri::AppHandle, png: String, report: String) -> Res<()> {
    let dir = SELFTEST.get().ok_or("no self-test")?;
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
        acc = (acc << 6) | val(c).ok_or("invalid base64")?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

// --- binary responses (outside specta) ---

#[tauri::command]
fn mesh(st: State<'_, AppState>, id: u32) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(rustyfem_session::mesh::pack(s.model(), s.mesh()))))
}

#[tauri::command]
fn profiles(st: State<'_, AppState>, id: u32) -> Res<Response> {
    with(&st, id, |s| Ok(Response::new(rustyfem_session::profiles::pack(s.model(), s.mesh()))))
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

const BINARY: [&str; 5] = ["mesh", "profiles", "symbols", "field", "displacement"];

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

/// Writes the TS types (in debug builds at startup, or with `rustyfem-app --export-bindings FILE`).
pub fn export_bindings(path: &str) -> Result<(), String> {
    // only write if the contents changed, otherwise `tauri dev` restarts the app in a loop
    let tmp = std::env::temp_dir().join(format!("rustyfem-bindings-{}.ts", std::process::id()));
    specta_builder()
        .export(specta_typescript::Typescript::default().header("// Generated from Rust with tauri-specta – do not edit by hand.
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

/// Starts the app in self-test mode: the UI runs a fixed test sequence and
/// writes a screenshot and timings to `dir`.
pub fn run_selftest(dir: PathBuf) {
    let _ = SELFTEST.set(dir);
    run_app()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run_app() {
    let builder = specta_builder();
    #[cfg(debug_assertions)]
    if let Err(e) = export_bindings(concat!(env!("CARGO_MANIFEST_DIR"), "/../../web/src/engine/bindings.ts")) {
        eprintln!("could not write bindings.ts: {e}");
    }
    let typed = builder.invoke_handler();
    let binary: Box<dyn Fn(tauri::ipc::Invoke<tauri::Wry>) -> bool + Send + Sync> = Box::new(tauri::generate_handler![mesh, profiles, symbols, field, displacement]);
    let state: AppState = Arc::new(Mutex::new(Sessions::default()));
    let server = mcp::Mcp::new(state.clone());
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage::<AppState>(state)
        .manage(server.clone())
        .setup(move |app| {
            server.set_app(app.handle().clone());
            // the MCP server is started if it was on last time or if the stdio bridge started the app
            if std::env::var_os("RUSTYFEM_MCP_AUTOSTART").is_some() || mcp::load_config().enabled {
                let _ = server.start(0, None);
            }
            Ok(())
        })
        .invoke_handler(move |invoke| if BINARY.contains(&invoke.message.command()) { binary(invoke) } else { typed(invoke) })
        .run(tauri::generate_context!())
        .expect("could not start RustyFEM");
}

pub fn run() {
    run_app()
}

