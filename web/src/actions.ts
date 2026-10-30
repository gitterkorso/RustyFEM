// Actions: calls to the engine via `Engine` and updates of the UI state.

import type { Edit, Edited, Engine, Opened, RunEvent, StepResultInfo, UnitSystem } from "./engine/Engine";
import { TauriEngine } from "./engine/TauriEngine";
import { get, log, persist, set, type ResultSel } from "./store";

export const engine: Engine = new TauriEngine();

function errText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

export function toast(kind: "error" | "info", text: string) {
  set({ toast: { kind, text } });
  if (kind === "error") log("error", text);
  window.setTimeout(() => {
    if (get().toast?.text === text) set({ toast: null });
  }, kind === "error" ? 7000 : 3500);
}

async function busy<T>(what: string, f: () => Promise<T>): Promise<T | undefined> {
  set({ busy: what });
  try {
    return await f();
  } catch (e) {
    toast("error", errText(e));
    return undefined;
  } finally {
    set({ busy: null });
  }
}

export async function init() {
  try {
    const [solvers, memory, unitSystems] = await Promise.all([engine.solvers(), engine.systemMemory(), engine.unitSystems()]);
    set({ solvers, memory, unitSystems });
  } catch (e) {
    log("warn", `the engine is not responding: ${errText(e)}`);
  }
}

async function adopt(o: Opened, keepOld = false) {
  const old = get().id;
  if (old !== null && !keepOld) engine.close(old).catch(() => {});
  const mesh = await engine.mesh(o.id);
  set((s) => ({
    id: o.id,
    session: o.session,
    model: o.model,
    mesh,
    meshVersion: s.meshVersion + 1,
    mode: "model",
    sel: o.session.kind === "rusty" ? { type: "script" } : { type: "model" },
    symbolStep: o.model.steps.length ? 0 : null,
    results: [],
    rsel: null,
    pick: null,
    animate: "off",
  }));
  for (const w of o.model.warnings) log("warn", w);
  log("info", `${o.session.name}: ${o.model.nodes} nodes, ${o.model.elements} elements, ${o.model.steps.length} steps`);
  if (o.session.has_results) await loadResults();
}

/** Empty models to build from scratch (like `EMPTY_2D`/`EMPTY_3D` in the session). */
export const blankModel = {
  2: "# New model – the tools in the app add lines here.\nmodel 2d\n",
  3: "# New model – the tools in the app add lines here.\nmodel 3d\n",
};

export async function newModel(script?: string) {
  if (!(await confirmDiscard())) return;
  await busy("Creating model", async () => adopt(await engine.newModel(script)));
}

export async function openPath(path: string) {
  await busy("Opening", async () => {
    await adopt(await engine.open(path));
    const recent = [path, ...get().recent.filter((p) => p !== path)].slice(0, 8);
    set({ recent });
    persist("recent", recent);
  });
}

export async function openDialog() {
  if (!(await confirmDiscard())) return;
  const p = await engine.openDialog();
  if (p) await openPath(p);
}

async function confirmDiscard(): Promise<boolean> {
  const s = get().session;
  if (!s?.dirty) return true;
  return window.confirm(`${s.name} has unsaved changes. Continue anyway?`);
}

export async function save(as = false) {
  const { id, session } = get();
  if (id === null || !session) return;
  let path = session.path && session.path.toLowerCase().endsWith(".rustyproj") && !as ? session.path : null;
  if (!path) {
    const base = session.name.replace(/\.(inp|rusty|rustyproj)$/i, "");
    path = await engine.saveDialog(`${base}.rustyproj`, "project");
    if (!path) return;
  }
  const p = path;
  await busy("Saving", async () => {
    const info = await engine.save(id, p);
    set({ session: info });
    toast("info", `Saved ${info.name}`);
    const recent = [p, ...get().recent.filter((x) => x !== p)].slice(0, 8);
    set({ recent });
    persist("recent", recent);
  });
}

export async function exportScript() {
  const { id, session } = get();
  if (id === null || session?.kind !== "rusty") return;
  const p = await engine.saveDialog(session.name.replace(/\.(rusty|rustyproj)$/i, "") + ".rusty", "script");
  if (p) await busy("Exporting", () => engine.save(id, p));
}

async function afterEdit(r: Edited) {
  set({ session: r.session, model: r.model });
  if (r.geometry) {
    const id = get().id!;
    const mesh = await engine.mesh(id);
    set((s) => ({ mesh, meshVersion: s.meshVersion + 1, pick: null }));
  }
  // the step can disappear and come back when a .rusty model is built line by line
  const ss = get().symbolStep;
  if ((ss !== null && r.model.steps.length <= ss) || (ss === null && r.model.steps.length)) set({ symbolStep: r.model.steps.length ? 0 : null });
}

export async function edit(e: Edit): Promise<boolean> {
  const err = await tryEdit(e);
  if (err !== null) toast("error", err);
  return err === null;
}

/** Like `edit`, but the error is returned (to be shown next to the field) instead of as a toast. */
export async function tryEdit(e: Edit): Promise<string | null> {
  const id = get().id;
  if (id === null) return "no model open";
  try {
    await afterEdit(await engine.edit(id, e));
    return null;
  } catch (err) {
    return errText(err);
  }
}

/** Changes the model's unit system. The results catalog is fetched again for the fields' units. */
export async function setUnits(u: UnitSystem) {
  const id = get().id;
  if (id === null) return;
  try {
    const session = await engine.setUnits(id, u);
    set({ session, results: session.has_results ? await engine.results(id) : get().results });
  } catch (err) {
    toast("error", errText(err));
  }
}

export async function undo() {
  const { id, session } = get();
  if (id === null || !session?.undo) return;
  await busy("Undoing", async () => afterEdit(await engine.undo(id)));
}

export async function redo() {
  const { id, session } = get();
  if (id === null || !session?.redo) return;
  await busy("Redoing", async () => afterEdit(await engine.redo(id)));
}

const MB = 1024 * 1024;
export function bytes(b: number | null) {
  if (b === null || !Number.isFinite(b)) return "–";
  return b > 1024 * MB ? `${(b / 1024 / MB).toFixed(1)} GB` : `${Math.max(b / MB, 0.1).toFixed(b > 100 * MB ? 0 : 1)} MB`;
}

export async function estimate() {
  const { id, run } = get();
  if (id === null) return;
  try {
    const est = await engine.estimate(id, run.solver);
    set((s) => ({ run: { ...s.run, estimate: est } }));
    return est;
  } catch (e) {
    toast("error", errText(e));
  }
}

export async function solve() {
  const { id, run, model } = get();
  if (id === null || run.running || !model) return;
  const problem = get().session?.problem;
  if (problem) {
    toast("error", `The model cannot be solved yet: ${problem}`);
    if (get().session?.kind === "rusty") set({ sel: { type: "script" }, right: true });
    return;
  }
  if (!model.steps.length) {
    toast("error", "The model has no steps to solve.");
    return;
  }
  set((s) => ({ dock: true, run: { ...s.run, running: true, progress: 0, stepIndex: 0, steps: model.steps.length, stepName: "", increment: 0, residuals: [] } }));
  const est = await estimate();
  let solver = run.solver;
  if (est) {
    log("info", `Memory: ${bytes(est.direct_bytes)} (direct) / ${bytes(est.iterative_bytes)} (PCG), budget ${bytes(est.budget_bytes)} – ${est.note}`);
    if (!est.fits && !window.confirm(`The estimated memory requirement exceeds the budget (${bytes(est.budget_bytes)}).\n${est.note}\n\nRun anyway?`)) {
      set((s) => ({ run: { ...s.run, running: false } }));
      log("warn", "The run was cancelled before it started (memory)");
      return;
    }
    if (!solver && est.solver !== "direct") solver = est.solver;
  }
  try {
    await engine.solve(id, solver, run.allFrames, onRunEvent);
  } catch (e) {
    set((s) => ({ run: { ...s.run, running: false } }));
    toast("error", errText(e));
  }
}

/** Progress from the engine (our own run or a run started via MCP). */
function onRunEvent(e: RunEvent) {
  const steps = get().model?.steps.length || 1;
  switch (e.type) {
    case "Started":
      set((s) => ({ dock: true, run: { ...s.run, running: true, progress: 0, stepIndex: 0, steps, stepName: "", increment: 0, residuals: [] } }));
      log("info", `Solving with ${e.solver} …`);
      break;
    case "Step":
      log("info", `Step ${e.index + 1}/${steps}: ${e.name} (*${e.procedure})`);
      set((s) => ({ run: { ...s.run, stepIndex: e.index, stepName: e.name, increment: 0, residuals: [], progress: e.index / steps } }));
      break;
    case "Increment":
      set((s) => ({
        run: { ...s.run, increment: s.run.increment + 1, progress: Math.min((s.run.stepIndex + Math.max(0, Math.min(1, e.fraction ?? 0))) / steps, 1) },
      }));
      break;
    case "Iteration":
      set((s) => ({ run: { ...s.run, residuals: [...s.run.residuals.slice(-119), e.residual ?? NaN] } }));
      break;
    case "Finished":
      set((s) => ({ run: { ...s.run, running: false, progress: e.ok ? 1 : s.run.progress } }));
      if (e.ok) {
        log("ok", `Done in ${(e.seconds ?? 0).toFixed(2)} s – ${e.message}`);
        loadResults(true);
      } else if (e.cancelled) log("warn", "The run was cancelled");
      else {
        log("error", e.message);
        toast("error", e.message);
      }
      break;
  }
}

/** Listens to engine events: edits and runs via MCP, the MCP server's status. */
export async function listenEngine() {
  await engine.listen({
    opened: async (o) => {
      log("info", `MCP: opened ${o.session.name}`);
      await adopt(o, get().session?.dirty ?? false);
    },
    changed: async (id, geometry) => {
      if (id !== get().id) return;
      const units = get().session?.units;
      const [session, model] = await Promise.all([engine.sessionInfo(id), engine.modelInfo(id)]);
      await afterEdit({ session, model, geometry });
      // set_units via MCP: the fields' units in the results catalog
      if (session.units !== units && session.has_results) set({ results: await engine.results(id) });
      log("info", `MCP: the model changed${session.undo ? ` (${session.undo})` : ""}`);
    },
    run: (id, e) => {
      if (id === get().id) onRunEvent(e);
    },
    mcp: (s) => set({ mcp: s }),
  });
  set({ mcp: await engine.mcpStatus() });
}

export async function mcpToggle(port: number | null, network: boolean | null = null) {
  try {
    set({ mcp: get().mcp?.running ? await engine.mcpStop() : await engine.mcpStart(port, network) });
  } catch (e) {
    toast("error", errText(e));
    set({ mcp: await engine.mcpStatus() });
  }
}

export async function mcpInstallClaudeDesktop() {
  try {
    const files = await engine.mcpInstallClaudeDesktop();
    toast("info", `RustyFEM added to Claude Desktop – restart Claude Desktop (${files.length} file${files.length === 1 ? "" : "s"})`);
    set({ mcp: await engine.mcpStatus() });
  } catch (e) {
    toast("error", errText(e));
  }
}

export async function cancel() {
  const id = get().id;
  if (id !== null) await engine.cancel(id).catch(() => {});
}

/** Default: S (else U) in the last frame, otherwise the first field. */
function defaultSel(res: StepResultInfo[]): ResultSel | null {
  const st = res.length - 1;
  if (st < 0) return null;
  const s = res[st];
  const f = s.fields.find((x) => x.name === "S") ?? s.fields.find((x) => x.name === "U") ?? s.fields[0];
  if (!f) return null;
  return { step: st, frame: Math.max(0, s.frames.length - 1), field: f.name, component: f.components[0] };
}

export async function loadResults(switchMode = false) {
  const id = get().id;
  if (id === null) return;
  const results = await engine.results(id);
  const old = get().rsel;
  const keep = old && results[old.step]?.fields.some((f) => f.name === old.field) ? { ...old, frame: Math.min(old.frame, results[old.step].frames.length - 1) } : null;
  const session = await engine.sessionInfo(id);
  set({ results, rsel: keep ?? defaultSel(results), session, ...(switchMode && results.length ? { mode: "results" as const } : {}) });
}

export function selectResult(p: Partial<ResultSel>) {
  set((s) => {
    if (!s.rsel) return {};
    const r = { ...s.rsel, ...p };
    const st = s.results[r.step];
    if (!st) return {};
    if (p.step !== undefined) r.frame = Math.max(0, st.frames.length - 1);
    const f = st.fields.find((x) => x.name === r.field) ?? st.fields[0];
    if (f) {
      r.field = f.name;
      if (!f.components.includes(r.component)) r.component = f.components[0];
    }
    return { rsel: r };
  });
}

export async function pick(kind: "node" | "element", index: number) {
  const { id, mode, rsel } = get();
  if (id === null) return;
  try {
    const at = mode === "results" && rsel ? { step: rsel.step, frame: rsel.frame } : null;
    set({ pick: await engine.pick(id, kind, index, at) });
  } catch (e) {
    toast("error", errText(e));
  }
}
