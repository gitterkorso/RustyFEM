# The RustyFEM app

Graphical interface on top of the engine.

| Part | Where | Contents |
|---|---|---|
| App API | `app/session` (crate `rustyfem-session`) | Session, undo/redo, runs with progress and cancellation, memory estimate and solver choice, surface extraction, result fields, project file `.rustyproj` |
| Shell | `app/src-tauri` (crate `rustyfem-app`) | Tauri 2 commands, binary responses for mesh and fields, file dialogs, memory budget per platform |
| Interface | `web/` | React + TypeScript 7, model viewport in plain Three.js, `Engine`/`TauriEngine` |

The engine reports progress through `rustyfem::progress` (steps, increments,
iterations). Without a hook the reports do nothing, so the results do not
change.

The engine is prepared for the app:

- Features `cli` (the command-line program and subprocesses, `rustyfem check`) and `dynlib`
  (UMAT/UEL from shared libraries) are on by default and off in the app.
- `inp::read_str_with` takes a file resolver for `*INCLUDE`; the project file
  reads its included files from the file itself (works in the iOS sandbox).
- The `RUSTYFEM_*` environment variables are collected in
  `rustyfem::settings::Settings` (`RunOptions::settings`); the environment is
  only read with feature `cli`.

## Running

```bash
cd web && npm install && npm run tauri dev
```

Release build (installer for the current platform):

```bash
cd web && npm run tauri build
```

The TS types for the JSON commands are generated from Rust (tauri-specta) into
`web/src/engine/bindings.ts` when the app starts in debug mode, or with
`target/debug/rustyfem-app --export-bindings web/src/engine/bindings.ts`.

## Tests

All build checks (tests, the engine without cli/dynlib, type checks for iOS,
macOS and Linux, the interface):

```bash
sh scripts/check.sh
```

Only the app API:

```bash
cargo test --release -p rustyfem-session
```

```bash
cd web && npx tsc
```

The session tests open regression cases, run them through the session with
progress and cancellation, compare with `steps::run`, and test undo/redo and
that the project file gives the same model and results after reloading.

Self-test in the real app window: opens two models, solves them with the
engine, switches fields and measures the frame time; writes `selftest.png` and
`selftest.json` to the directory and exits.

```bash
target/release/rustyfem-app --selftest selftest
```

CI (`.github/workflows/app.yml`) builds the app on Windows, macOS and Linux and
the engine with the app API for iOS.

## AI assistants (MCP)

The app has a built-in MCP server; nothing extra needs to be downloaded. The
plug button in the top bar opens the panel with start/stop and instructions
for Claude Desktop (a button adds RustyFEM to the configuration), Claude Code,
Codex and other clients.

- stdio: `rustyfem-app --mcp` is a bridge that the client launches. It starts
  the app with the server enabled if it is not already running.
- HTTP: `http://127.0.0.1:7456/mcp` (Streamable HTTP, JSON responses) with
  `Authorization: Bearer TOKEN`. Local connections only; requests with a
  foreign `Origin` are rejected. Port and token are stored in `mcp.json` in
  the app's configuration directory (`%APPDATA%\se.rustyfem.app` on Windows).
- The tools work on the model that is open in the app: edits, solutions and
  progress are shown immediately and can be undone. Tools: `model_info`,
  `new_model`, `open_file`, `get_script`, `set_script`, `edit`, `undo`,
  `redo`, `solve`, `results`, `field_stats`, `node_info`, `element_info`,
  `save_project`, `rusty_help`.

## The interface

- **Model tree** (left): element types, sets, surfaces, materials,
  cross-sections, boundary conditions and steps, with a filter.
- **Properties** (right): editing of material data, density, thickness, step
  procedure data, NLGEOM, INC, boundary conditions and loads (`.inp`); script
  and tools (plate, block, beam, truss, support, load, pressure, gravity) for
  `.rusty`. Everything can be undone.
- **Results**: step, frame/mode, field and component (Mises is computed),
  automatic or fixed deformation scale, animation of frames and modes, banded
  color scales, history curves.
- **Model viewport**: pan, zoom, rotate, 2D mode, standard views (keys 0–6, F
  fits), surfaces/edges/undeformed shape/symbols, GPU picking of elements or
  nodes (larger hit area for touch input), PNG export.
- **Run panel**: solver choice (automatic based on the memory budget),
  progress, residual curve, cancellation and messages.
- `Ctrl+K` command palette, `F5` solve, `Esc` cancel, `Ctrl+Z/Y`, `Ctrl+S`.

## Not done yet

- The iOS build (the Tauri iOS project, signing, TestFlight) requires a Mac
  and Xcode. The engine and the app API are type-checked for iOS here and
  built for iOS in CI.
- Export of edited `.inp` models (`.inp` is import only; projects are saved
  as `.rustyproj`).
- Highlighting of sets in the model viewport.
