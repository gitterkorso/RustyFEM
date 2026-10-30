//! Local MCP server (Model Context Protocol) built into the app.
//!
//! - HTTP (Streamable HTTP, JSON responses) on `127.0.0.1`, with a bearer token.
//!   Local connections only; web pages are blocked by the Origin check.
//! - stdio: `rustyfem-app --mcp` is a bridge for clients that start
//!   the server as a command (Claude Desktop, Claude Code, Codex). The bridge
//!   starts the app if it is not already running and forwards each JSON-RPC line.
//!
//! The tools work in the model that is open in the app, so edits and
//! solutions are visible in the window immediately and can be undone there.

use crate::{AppState, Opened, Res};
use rustyfem_session::edit::Edit;
use rustyfem_session::units::UnitSystem;
use rustyfem_session::{RunEvent, Session};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use specta::Type;
use std::io::{BufRead, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};

pub const DEFAULT_PORT: u16 = 7456;
const PROTOCOLS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// Saved settings (`mcp.json` in the app's configuration directory).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    pub enabled: bool,
    pub port: u16,
    pub token: String,
}

/// The app's configuration directory (the same for the app and the stdio bridge).
pub fn config_dir() -> PathBuf {
    let home = |v: &str| std::env::var_os(v).map(PathBuf::from);
    let base = if cfg!(windows) {
        home("APPDATA").unwrap_or_else(std::env::temp_dir)
    } else if cfg!(target_os = "macos") {
        home("HOME").map(|h| h.join("Library/Application Support")).unwrap_or_else(std::env::temp_dir)
    } else {
        home("XDG_CONFIG_HOME").or_else(|| home("HOME").map(|h| h.join(".config"))).unwrap_or_else(std::env::temp_dir)
    };
    base.join("se.rustyfem.app")
}

fn config_path() -> PathBuf {
    config_dir().join("mcp.json")
}

pub fn load_config() -> Config {
    let mut c: Config = std::fs::read(config_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if c.port == 0 {
        c.port = DEFAULT_PORT;
    }
    if c.token.is_empty() {
        c.token = new_token();
    }
    c
}

fn save_config(c: &Config) {
    let _ = std::fs::create_dir_all(config_dir());
    let _ = std::fs::write(config_path(), serde_json::to_vec_pretty(c).unwrap_or_default());
}

/// Random token (32 hex characters) from time, process and addresses.
fn new_token() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        h.write_u32(std::process::id());
        h.write_u64(i);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// Status for the UI.
#[derive(Debug, Clone, Serialize, Type)]
pub struct McpStatus {
    pub running: bool,
    pub port: u16,
    pub token: String,
    pub url: String,
    /// Path to the app's executable (for the stdio configuration).
    pub exe: String,
    pub calls: u32,
    pub last_tool: Option<String>,
    pub error: Option<String>,
    /// Claude Desktop configuration files that already contain RustyFEM.
    pub claude_desktop: Vec<String>,
}

pub struct Mcp {
    state: AppState,
    app: Mutex<Option<AppHandle>>,
    server: Mutex<Option<(u16, Arc<AtomicBool>)>>,
    calls: AtomicU32,
    last_tool: Mutex<Option<String>>,
    error: Mutex<Option<String>>,
}

impl Mcp {
    pub fn new(state: AppState) -> Arc<Mcp> {
        Arc::new(Mcp { state, app: Mutex::new(None), server: Mutex::new(None), calls: AtomicU32::new(0), last_tool: Mutex::new(None), error: Mutex::new(None) })
    }

    pub fn set_app(&self, app: AppHandle) {
        *self.app.lock().unwrap() = Some(app);
    }

    fn emit<S: Serialize + Clone>(&self, name: &str, payload: S) {
        if let Some(a) = self.app.lock().unwrap().as_ref() {
            let _ = a.emit(name, payload);
        }
    }

    pub fn status(&self) -> McpStatus {
        let c = load_config();
        let running = self.server.lock().unwrap().as_ref().map(|s| s.0);
        let port = running.unwrap_or(c.port);
        McpStatus {
            running: running.is_some(),
            port,
            token: c.token,
            url: format!("http://127.0.0.1:{port}/mcp"),
            exe: std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default(),
            calls: self.calls.load(Ordering::Relaxed),
            last_tool: self.last_tool.lock().unwrap().clone(),
            error: self.error.lock().unwrap().clone(),
            claude_desktop: claude_desktop_configs().into_iter().filter(|p| has_rustyfem(p)).map(|p| p.display().to_string()).collect(),
        }
    }

    /// Starts the HTTP server (`port` = 0 gives the saved or default port).
    pub fn start(self: &Arc<Self>, port: u16) -> Res<McpStatus> {
        if self.server.lock().unwrap().is_some() {
            return Ok(self.status());
        }
        let mut c = load_config();
        if port != 0 {
            c.port = port;
        }
        let listener = TcpListener::bind(("127.0.0.1", c.port)).map_err(|e| {
            let msg = format!("could not listen on 127.0.0.1:{} ({e}) – choose another port", c.port);
            *self.error.lock().unwrap() = Some(msg.clone());
            msg
        })?;
        c.enabled = true;
        save_config(&c);
        *self.error.lock().unwrap() = None;
        let stop = Arc::new(AtomicBool::new(false));
        *self.server.lock().unwrap() = Some((c.port, stop.clone()));
        let me = self.clone();
        let token = c.token.clone();
        std::thread::Builder::new()
            .name("rustyfem-mcp".into())
            .spawn(move || {
                for conn in listener.incoming() {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let Ok(conn) = conn else { continue };
                    let me = me.clone();
                    let token = token.clone();
                    std::thread::spawn(move || {
                        let _ = me.serve(conn, &token);
                    });
                }
            })
            .map_err(|e| e.to_string())?;
        let s = self.status();
        self.emit("mcp-status", s.clone());
        Ok(s)
    }

    pub fn stop(&self) -> McpStatus {
        if let Some((port, stop)) = self.server.lock().unwrap().take() {
            stop.store(true, Ordering::Relaxed);
            // wake the accept loop
            let _ = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_millis(200));
        }
        let mut c = load_config();
        c.enabled = false;
        save_config(&c);
        let s = self.status();
        self.emit("mcp-status", s.clone());
        s
    }

    /// One HTTP request per connection.
    fn serve(&self, mut conn: TcpStream, token: &str) -> std::io::Result<()> {
        conn.set_read_timeout(Some(Duration::from_secs(30)))?;
        let req = match read_request(&mut conn) {
            Ok(r) => r,
            Err(e) => return respond(&mut conn, 400, "text/plain", e.as_bytes()),
        };
        // web pages must not reach the server (protection against DNS rebinding)
        if let Some(o) = req.header("origin") {
            let ok = ["http://127.0.0.1", "http://localhost", "tauri://", "http://tauri.localhost"].iter().any(|p| o.starts_with(p));
            if !ok {
                return respond(&mut conn, 403, "text/plain", b"origin not allowed");
            }
        }
        if req.path != "/mcp" {
            return respond(&mut conn, 200, "text/plain; charset=utf-8", "RustyFEM MCP server – connect to /mcp".as_bytes());
        }
        if req.method != "POST" {
            return respond(&mut conn, 405, "text/plain", b"POST only");
        }
        if req.header("authorization").map(|a| a.trim()) != Some(&format!("Bearer {token}")) {
            return respond(&mut conn, 401, "text/plain", b"wrong or missing token");
        }
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => {
                let err = json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":format!("invalid JSON: {e}")}});
                return respond(&mut conn, 400, "application/json", err.to_string().as_bytes());
            }
        };
        let out = match body {
            Value::Array(items) => {
                let r: Vec<Value> = items.into_iter().filter_map(|m| self.rpc(m)).collect();
                if r.is_empty() { None } else { Some(Value::Array(r)) }
            }
            m => self.rpc(m),
        };
        match out {
            Some(v) => respond(&mut conn, 200, "application/json", v.to_string().as_bytes()),
            None => respond(&mut conn, 202, "text/plain", b""),
        }
    }

    /// One JSON-RPC message; `None` for notifications.
    fn rpc(&self, m: Value) -> Option<Value> {
        let id = m.get("id").cloned();
        let method = m.get("method").and_then(Value::as_str).unwrap_or("");
        let params = m.get("params").cloned().unwrap_or(Value::Null);
        let id = id?;
        let result: Result<Value, (i64, String)> = match method {
            "initialize" => {
                let want = params.get("protocolVersion").and_then(Value::as_str).unwrap_or(PROTOCOLS[0]);
                let v = if PROTOCOLS.contains(&want) { want } else { PROTOCOLS[0] };
                Ok(json!({
                    "protocolVersion": v,
                    "capabilities": {"tools": {"listChanged": false}},
                    "serverInfo": {"name": "rustyfem", "title": "RustyFEM", "version": env!("CARGO_PKG_VERSION")},
                    "instructions": INSTRUCTIONS,
                }))
            }
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": tools()})),
            "tools/call" => {
                let name = params.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                let args = params.get("arguments").cloned().unwrap_or(json!({}));
                self.calls.fetch_add(1, Ordering::Relaxed);
                *self.last_tool.lock().unwrap() = Some(name.clone());
                let r = self.call(&name, &args);
                self.emit("mcp-status", self.status());
                Ok(match r {
                    Ok(v) => {
                        let text = if let Value::String(s) = &v { s.clone() } else { serde_json::to_string_pretty(&v).unwrap_or_default() };
                        json!({"content": [{"type": "text", "text": text}], "isError": false})
                    }
                    Err(e) => json!({"content": [{"type": "text", "text": e}], "isError": true}),
                })
            }
            "resources/list" => Ok(json!({"resources": []})),
            "prompts/list" => Ok(json!({"prompts": []})),
            _ => Err((-32601, format!("unknown method {method}"))),
        };
        Some(match result {
            Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
            Err((code, msg)) => json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": msg}}),
        })
    }

    /// The session shown in the app (the most recently opened).
    fn active(&self) -> Res<u32> {
        let g = self.state.lock().map_err(|e| e.to_string())?;
        g.active.filter(|id| g.open.contains_key(id)).ok_or_else(|| "no model is open in the app – use new_model or open_file".to_string())
    }

    fn with<T>(&self, f: impl FnOnce(&mut Session) -> Res<T>) -> Res<T> {
        let id = self.active()?;
        crate::with(&self.state, id, f)
    }

    fn changed(&self, geometry: bool) {
        if let Ok(id) = self.active() {
            self.emit("mcp-changed", json!({"id": id, "geometry": geometry}));
        }
    }

    fn opened(&self, s: Session) -> Res<Value> {
        let o: Opened = crate::insert(&self.state, s)?;
        let summary = json!({"nodes": o.model.nodes, "elements": o.model.elements, "steps": o.model.steps.len(), "warnings": o.model.warnings});
        self.emit("mcp-opened", o);
        Ok(summary)
    }

    fn call(&self, name: &str, a: &Value) -> Res<Value> {
        let s = |k: &str| a.get(k).and_then(Value::as_str).map(str::to_string);
        let n = |k: &str| a.get(k).and_then(Value::as_u64).map(|x| x as usize);
        match name {
            "new_model" => self.opened(Session::new_fem(s("script").as_deref().unwrap_or(rustyfem_session::NEW_MODEL))?),
            "open_file" => {
                let p = s("path").ok_or("path is missing")?;
                self.opened(Session::open(&PathBuf::from(p))?)
            }
            "model_info" => self.with(|x| {
                let info = x.info();
                Ok(json!({"session": info, "model": rustyfem_session::info::model_info(x.model())}))
            }),
            "set_units" => {
                let u: UnitSystem = serde_json::from_value(a.get("units").cloned().ok_or("units is missing")?).map_err(|_| "units must be none, si, si_mm or us".to_string())?;
                self.with(|x| {
                    x.set_units(u);
                    Ok(())
                })?;
                self.changed(false);
                Ok(json!({"ok": true, "units": u, "label": u.label()}))
            }
            "get_script" => self.with(|x| x.info().script.ok_or_else(|| "the model is an .inp – it is edited with edit".to_string()).map(Value::String)),
            "set_script" => {
                let text = s("script").ok_or("script is missing")?;
                self.with(|x| x.apply(Edit::FemText { text }))?;
                self.changed(true);
                self.with(|x| Ok(json!({"ok": true, "nodes": x.model().nodes.len(), "elements": x.model().elements.len()})))
            }
            "edit" => {
                let e: Edit = serde_json::from_value(a.get("edit").cloned().ok_or("edit is missing")?).map_err(|e| format!("invalid edit: {e}"))?;
                let g = self.with(|x| x.apply(e))?;
                self.changed(g);
                Ok(json!({"ok": true}))
            }
            "undo" | "redo" => {
                let g = self.with(|x| if name == "undo" { x.undo() } else { x.redo() })?;
                self.changed(g);
                self.with(|x| Ok(json!({"ok": true, "undo": x.info().undo, "redo": x.info().redo})))
            }
            "solve" => self.solve(s("solver"), a.get("all_frames").and_then(Value::as_bool).unwrap_or(true)),
            "results" => self.with(|x| Ok(json!(x.catalog().unwrap_or_default()))),
            "field_stats" => self.with(|x| {
                let cat = x.catalog().ok_or("no results – run solve first")?;
                let step = n("step").unwrap_or(cat.len().saturating_sub(1));
                let frame = n("frame").unwrap_or_else(|| cat.get(step).map(|st| st.frames.len().saturating_sub(1)).unwrap_or(0));
                let st = x.field_stats(step, frame, &s("field").ok_or("field is missing")?, &s("component").unwrap_or_default())?;
                Ok(json!({"step": step, "frame": frame, "stats": st}))
            }),
            "node_info" | "element_info" => self.with(|x| {
                let el = name == "element_info";
                let what = a.get(if el { "element" } else { "node" }).map(|v| v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string())).ok_or("number is missing")?;
                let idx = x.lookup(&what, el)?;
                let at = x.results.as_ref().map(|r| (r.steps.len() - 1, r.steps.last().map(|s| s.frames.len().saturating_sub(1)).unwrap_or(0)));
                Ok(json!(if el { x.element_info(idx, at)? } else { x.node_info(idx, at)? }))
            }),
            "save_project" => {
                let p = PathBuf::from(s("path").ok_or("path is missing")?);
                self.with(|x| {
                    std::fs::write(&p, x.to_project()).map_err(|e| format!("could not write {}: {e}", p.display()))?;
                    x.path = Some(p.clone());
                    x.dirty = false;
                    Ok(json!({"ok": true, "path": p.display().to_string()}))
                })
            }
            "rusty_help" => Ok(Value::String(FEM_HELP.into())),
            _ => Err(format!("unknown tool {name}")),
        }
    }

    /// Solves the model and waits for the result; the progress is shown in the app.
    fn solve(&self, solver: Option<String>, all_frames: bool) -> Res<Value> {
        let id = self.active()?;
        let app = self.app.lock().unwrap().clone();
        let (tx, rx) = std::sync::mpsc::channel::<RunEvent>();
        let emit = move |e: RunEvent| {
            if let Some(a) = &app {
                let _ = a.emit("run-event", json!({"id": id, "event": e.clone()}));
            }
            let _ = tx.send(e);
        };
        crate::start_run(&self.state, id, solver, all_frames, emit)?;
        let mut steps = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(3600)).map_err(|_| "the solution did not respond".to_string())? {
                RunEvent::Step { name, procedure, .. } => steps.push(format!("{name} (*{procedure})")),
                RunEvent::Finished { ok, cancelled, seconds, message } => {
                    if !ok {
                        return Err(if cancelled { "the run was cancelled".into() } else { message });
                    }
                    let summary = self.with(|x| {
                        let cat = x.catalog().unwrap_or_default();
                        let last = cat.last();
                        let (st, fr) = (cat.len().saturating_sub(1), last.map(|s| s.frames.len().saturating_sub(1)).unwrap_or(0));
                        let u = x.field_stats(st, fr, "U", "Magnitude").ok();
                        let mises = x.field_stats(st, fr, "S", "Mises").ok();
                        Ok(json!({"ok": true, "seconds": seconds, "message": message, "steps": steps, "max_U": u, "mises": mises, "fields": last.map(|s| s.fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>())}))
                    })?;
                    return Ok(summary);
                }
                _ => {}
            }
        }
    }
}

const INSTRUCTIONS: &str = "RustyFEM is an FEM solver that also reads input files in .inp format. The tools work in the model that is open in the RustyFEM app; the user sees all changes immediately and can undo them. Build simple models with new_model/set_script (.rusty scripts, see rusty_help) or open an .inp file with open_file. Solve with solve and read results with results, field_stats, node_info and element_info. The engine is unitless with consistent units; the model's unit system (model_info: session.units and unit_symbols) is only a label shown by the app. Change it with set_units if it does not match the values – no values are converted. field_stats and node_info/element_info give the unit of each value.";

const FEM_HELP: &str = "The .rusty script language (one line per command, # starts a comment):
model 2d|3d                       dimension (otherwise inferred)
plane stress|strain               2D: plane stress (default) or plane strain
elements quad|tri                 2D: element shape
material NAME E=.. nu=.. rho=..   steel and aluminium have default values; the latest material is used
plate x=0..2 y=0..0.2 t=0.01 size=0.05          rectangular plate (2D) with thickness t
quad (0,0) (2,0) (2,1) (0,1) t=0.01 size=0.05   quadrilateral
box x=0..1 y=0..0.2 z=0..0.2 size=0.05          block (3D, C3D8)
beam (0,0,0) (2,0,0) section=rect(b,h)|circle(r)|pipe(r,t)|ibeam(h,b,tf,tw) n=10 [up=(0,0,1)]
truss (0,0) (1,1) area=1e-4
fix SEL | pin SEL | support SEL ux uy uz rx ry rz (or ux=0.001)
load SEL fx=.. fy=.. fz=.. mx=.. my=.. mz=..
pressure SEL p=1e5
gravity g=9.81  (or gx= gy= gz=)
SEL: x=0, y=0..1, z=.., at=(x,y[,z]) or all
Example: model 2d / material steel / plate x=0..2 y=0..0.2 t=0.01 size=0.025 / fix x=0 / load x=2 fy=-1000";

fn tools() -> Value {
    let obj = |props: Value, req: &[&str]| json!({"type": "object", "properties": props, "required": req});
    let none = json!({"type": "object", "properties": {}});
    json!([
        {"name": "model_info", "description": "Overview of the model that is open in the app: nodes, elements, sets, materials, sections, steps, boundary conditions and loads.", "inputSchema": none, "annotations": {"readOnlyHint": true}},
        {"name": "new_model", "description": "Creates a new model from a .rusty script (see rusty_help) and shows it in the app.", "inputSchema": obj(json!({"script": {"type": "string", "description": ".rusty script"}}), &[])},
        {"name": "open_file", "description": "Opens a model (.inp, .rusty or .rustyproj) from disk and shows it in the app.", "inputSchema": obj(json!({"path": {"type": "string"}}), &["path"])},
        {"name": "set_units", "description": "Sets the model's unit system (only labels in the app; no values are converted): none, si (m, N, kg, s, Pa), si_mm (mm, N, t, s, MPa) or us (in, lbf, s, psi).", "inputSchema": obj(json!({"units": {"type": "string", "enum": ["none", "si", "si_mm", "us"]}}), &["units"])},
        {"name": "get_script", "description": "Gets the model's .rusty script (.rusty models only).", "inputSchema": none, "annotations": {"readOnlyHint": true}},
        {"name": "set_script", "description": "Replaces the .rusty script; the model is rebuilt and shown in the app. Can be undone with undo.", "inputSchema": obj(json!({"script": {"type": "string"}}), &["script"])},
        {"name": "edit", "description": "Changes a value in an .inp model (can be undone). edit is an object with kind and fields: {kind:'MaterialData',material,behavior,line,fields:[..]}, {kind:'Density',material,value}, {kind:'SectionThickness',section,value}, {kind:'ProcedureData',step,line,fields}, {kind:'Nlgeom',step,value}, {kind:'MaxIncrements',step,value}, {kind:'BcValue',step|null,block,item,value}, {kind:'CloadValue',step,block,item,value}, {kind:'DloadValue',step,block,item,value}, {kind:'StepName',step,name}. Indices are 0-based and come from model_info.", "inputSchema": obj(json!({"edit": {"type": "object"}}), &["edit"])},
        {"name": "undo", "description": "Undoes the latest change.", "inputSchema": none},
        {"name": "redo", "description": "Redoes the most recently undone change.", "inputSchema": none},
        {"name": "solve", "description": "Solves the model (the progress is shown in the app) and returns a summary: maximum displacement, maximum von Mises, fields.", "inputSchema": obj(json!({"solver": {"type": "string", "enum": ["direct", "indefinite", "unsymmetric", "iterative", "iterative-ic0", "iterative-jacobi"]}, "all_frames": {"type": "boolean", "description": "save one frame per increment (default true)"}}), &[])},
        {"name": "results", "description": "Results catalog: steps, frames (time/mode), fields and components, history output.", "inputSchema": none, "annotations": {"readOnlyHint": true}},
        {"name": "field_stats", "description": "Min and max of a field with the node where it occurs. Element fields are averaged to the nodes. S has the component Mises.", "inputSchema": obj(json!({"field": {"type": "string", "description": "e.g. U, S, E, RF, PEEQ, SF, SM"}, "component": {"type": "string", "description": "e.g. Magnitude, U2, Mises, S11 (default: the first)"}, "step": {"type": "integer"}, "frame": {"type": "integer"}}), &["field"]), "annotations": {"readOnlyHint": true}},
        {"name": "node_info", "description": "Coordinates and results for a node (\"12\" or \"INSTANCE.12\").", "inputSchema": obj(json!({"node": {"type": "string"}}), &["node"]), "annotations": {"readOnlyHint": true}},
        {"name": "element_info", "description": "Type, nodes, section and results (mean over the integration points) for an element.", "inputSchema": obj(json!({"element": {"type": "string"}}), &["element"]), "annotations": {"readOnlyHint": true}},
        {"name": "save_project", "description": "Saves the model and the results as a RustyFEM project (.rustyproj).", "inputSchema": obj(json!({"path": {"type": "string"}}), &["path"])},
        {"name": "rusty_help", "description": "The syntax of .rusty scripts, with examples.", "inputSchema": none, "annotations": {"readOnlyHint": true}},
    ])
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(h, _)| h == k).map(|(_, v)| v.as_str())
    }
}

fn read_request(conn: &mut TcpStream) -> Result<Request, String> {
    let mut r = std::io::BufReader::new(conn.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    r.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("/").split('?').next().unwrap_or("/").to_string();
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_lowercase(), v.trim().to_string()));
        }
        if headers.len() > 100 {
            return Err("too many headers".into());
        }
    }
    let len: usize = headers.iter().find(|(k, _)| k == "content-length").and_then(|(_, v)| v.parse().ok()).unwrap_or(0);
    if len > 16 << 20 {
        return Err("request too large".into());
    }
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, headers, body })
}

fn respond(conn: &mut TcpStream, code: u16, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let reason = match code {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        405 => "Method Not Allowed",
        _ => "Error",
    };
    write!(conn, "HTTP/1.1 {code} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
    conn.write_all(body)?;
    conn.flush()
}

/// Sends a JSON-RPC text to the app's server and returns the response (empty for 202).
fn post(port: u16, token: &str, body: &str) -> Result<String, String> {
    let mut c = TcpStream::connect_timeout(&([127, 0, 0, 1], port).into(), Duration::from_secs(2)).map_err(|e| e.to_string())?;
    c.set_read_timeout(Some(Duration::from_secs(3700))).ok();
    write!(
        c,
        "POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nAuthorization: Bearer {token}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .map_err(|e| e.to_string())?;
    let mut resp = Vec::new();
    c.read_to_end(&mut resp).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&resp);
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0);
    if status >= 400 {
        return Err(format!("the server responded {status}: {body}"));
    }
    Ok(body.to_string())
}

/// `rustyfem-app --mcp`: stdio bridge to the app's server. Starts the app with
/// the server switched on if it does not already respond.
pub fn stdio_bridge() {
    let reachable = |c: &Config| TcpStream::connect_timeout(&([127, 0, 0, 1], c.port).into(), Duration::from_millis(300)).is_ok();
    let mut cfg = load_config();
    if !reachable(&cfg) {
        if let Ok(exe) = std::env::current_exe() {
            let _ = std::process::Command::new(exe).env("RUSTYFEM_MCP_AUTOSTART", "1").stdin(std::process::Stdio::null()).stdout(std::process::Stdio::null()).spawn();
        }
        for _ in 0..100 {
            std::thread::sleep(Duration::from_millis(200));
            cfg = load_config();
            if reachable(&cfg) {
                break;
            }
        }
    }
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let cfg = load_config();
        match post(cfg.port, &cfg.token, &line) {
            Ok(body) if !body.trim().is_empty() => {
                let _ = writeln!(out, "{}", body.trim());
                let _ = out.flush();
            }
            Ok(_) => {}
            Err(e) => {
                // reply with a JSON-RPC error to requests (not to notifications)
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if let Some(id) = v.get("id") {
                        let err = json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32000, "message": format!("The RustyFEM app is not responding: {e}. Start the app and switch on the MCP server.")}});
                        let _ = writeln!(out, "{err}");
                        let _ = out.flush();
                    }
                }
            }
        }
    }
}

/// Claude Desktop configuration files (regular installation and the Microsoft Store variant).
pub fn claude_desktop_configs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if cfg!(windows) {
        if let Some(a) = std::env::var_os("APPDATA") {
            v.push(PathBuf::from(a).join("Claude").join("claude_desktop_config.json"));
        }
        if let Some(l) = std::env::var_os("LOCALAPPDATA") {
            if let Ok(rd) = std::fs::read_dir(PathBuf::from(l).join("Packages")) {
                for e in rd.flatten() {
                    if e.file_name().to_string_lossy().starts_with("Claude_") {
                        let d = e.path().join("LocalCache").join("Roaming").join("Claude");
                        if d.exists() {
                            v.push(d.join("claude_desktop_config.json"));
                        }
                    }
                }
            }
        }
    } else if cfg!(target_os = "macos") {
        if let Some(h) = std::env::var_os("HOME") {
            v.push(PathBuf::from(h).join("Library/Application Support/Claude/claude_desktop_config.json"));
        }
    } else if let Some(h) = std::env::var_os("HOME") {
        v.push(PathBuf::from(h).join(".config/Claude/claude_desktop_config.json"));
    }
    v
}

fn has_rustyfem(p: &PathBuf) -> bool {
    std::fs::read(p).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()).is_some_and(|v| v.pointer("/mcpServers/rustyfem").is_some())
}

/// Adds RustyFEM to the Claude Desktop configuration (with a backup).
pub fn install_claude_desktop() -> Res<Vec<String>> {
    let exe = std::env::current_exe().map_err(|e| e.to_string())?.display().to_string();
    let mut paths: Vec<PathBuf> = claude_desktop_configs();
    // if the Claude directory exists for only one variant, only that one is written; otherwise the regular one
    let existing: Vec<PathBuf> = paths.iter().filter(|p| p.parent().is_some_and(|d| d.exists())).cloned().collect();
    if !existing.is_empty() {
        paths = existing;
    } else {
        paths.truncate(1);
    }
    let mut done = Vec::new();
    for p in paths {
        let mut v: Value = match std::fs::read(&p) {
            Ok(b) => {
                let _ = std::fs::write(p.with_extension("json.bak"), &b);
                serde_json::from_slice(&b).map_err(|e| format!("{} is not valid JSON ({e}) – edit it by hand", p.display()))?
            }
            Err(_) => json!({}),
        };
        let o = v.as_object_mut().ok_or("the configuration is not a JSON object")?;
        let servers = o.entry("mcpServers").or_insert_with(|| json!({}));
        servers.as_object_mut().ok_or("mcpServers is not an object")?.insert("rustyfem".into(), json!({"command": exe, "args": ["--mcp"]}));
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
        }
        std::fs::write(&p, serde_json::to_vec_pretty(&v).unwrap()).map_err(|e| format!("could not write {}: {e}", p.display()))?;
        done.push(p.display().to_string());
    }
    Ok(done)
}
