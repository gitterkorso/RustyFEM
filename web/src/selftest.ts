// Self-test in the real app window (`rustyfem-app --selftest DIR`):
// opens models, solves them with the engine, switches results and measures the frame time.
// The screenshot and timings are written to the directory, then the app exits.

import { engine, newModel, selectResult, solve } from "./actions";
import { get, set } from "./store";
import type { Viewer } from "./viewer/Viewer";

const viewer = () => (window as unknown as { __viewer: Viewer }).__viewer;

async function until(what: string, ok: () => boolean, ms = 90_000) {
  const t0 = performance.now();
  while (!ok()) {
    if (performance.now() - t0 > ms) {
      const st = get();
      throw new Error(`self-test: waited too long for ${what} (busy=${st.busy}, running=${st.run.running}, mode=${st.mode}, results=${st.results.length}, log: ${st.log.slice(-3).map((l) => l.text).join(" / ")})`);
    }
    await new Promise((r) => setTimeout(r, 15));
  }
}

const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

/** Targets for a responsive UI (ms). */
const TARGETS: Record<string, number> = {
  frame_model_ms: 16.7,
  frame_results_ms: 16.7,
  first_results_frame_ms: 100,
  field_switch_ms: 100,
  open_ms: 2000,
  // the UI during the solution: longest and 95th percentile of the frame intervals
  during_solve_max_ms: 50,
  during_solve_p95_ms: 25,
};

function checkTargets(r: Record<string, number | string>) {
  const miss: string[] = [];
  for (const [k, lim] of Object.entries(TARGETS)) {
    const v = r[k];
    if (typeof v === "number" && v > lim) miss.push(`${k} ${v} > ${lim}`);
  }
  r.targets = miss.length ? `missed: ${miss.join(", ")}` : "all targets met";
  return miss.length === 0;
}

async function measureCase(name: string, script: string) {
  const r: Record<string, number | string> = { model: name };
  let t = performance.now();
  await newModel(script);
  await until("the model", () => get().mesh !== null && get().busy === null);
  await frame();
  r.open_ms = Math.round(performance.now() - t);
  const m = get().model!;
  r.nodes = m.nodes;
  r.elements = m.elements;
  Object.assign(r, viewer().stats);
  r.frame_model_ms = Number(viewer().benchmark(60).toFixed(2));
  t = performance.now();
  // frame intervals while the solution runs (the model is redrawn every frame)
  const gaps: number[] = [];
  let last = performance.now();
  let live = true;
  const tick = () => {
    const now = performance.now();
    gaps.push(now - last);
    last = now;
    viewer().requestRender();
    if (live) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
  await solve();
  await until("the solution", () => !get().run.running && get().results.length > 0 && get().mode === "results");
  live = false;
  r.solve_ms = Math.round(performance.now() - t);
  const g = gaps.slice(2).sort((a, b) => a - b);
  if (g.length) {
    r.during_solve_frames = g.length;
    r.during_solve_max_ms = Number(g[g.length - 1].toFixed(1));
    r.during_solve_p95_ms = Number(g[Math.floor(g.length * 0.95)].toFixed(1));
  }
  t = performance.now();
  await until("the results view", () => document.querySelector(".legend") !== null);
  await frame();
  r.first_results_frame_ms = Math.round(performance.now() - t);
  t = performance.now();
  set({ mode: "results" });
  selectResult({ field: "U", component: "Magnitude" });
  await until("the field switch", () => document.querySelector(".legend-title strong")?.textContent === "U");
  await frame();
  r.field_switch_ms = Math.round(performance.now() - t);
  // an element field that exists (S for continuum, SF/SM for beams)
  const st = get().results[get().rsel!.step];
  const ef = st.fields.find((f) => f.name === "S") ?? st.fields.find((f) => f.position === "element") ?? st.fields[0];
  r.element_field = ef.name;
  set({ mode: "results" });
  selectResult({ field: ef.name, component: ef.components[0] });
  await until(ef.name, () => document.querySelector(".legend-title strong")?.textContent === ef.name);
  await frame();
  r.frame_results_ms = Number(viewer().benchmark(60).toFixed(2));
  const log = get().log.filter((l) => l.kind === "error");
  if (log.length) r.error = log.map((l) => l.text).join(" | ");
  if (!checkTargets(r)) r.error = String(r.targets);
  return r;
}

export async function runSelftest() {
  if (!(await engine.selftest().catch(() => false))) return;
  const report: Record<string, unknown> = {
    targets_ms: TARGETS, time: new Date().toISOString(), window: `${window.innerWidth}x${window.innerHeight}`, dpr: window.devicePixelRatio };
  const cases: Record<string, unknown>[] = [];
  try {
    window.confirm = () => true;
    set({ left: true, right: true, dock: true });
    cases.push(await measureCase("beam frame", "model 3d\nmaterial steel\nbeam (0,0,0) (0,3,0) (4,3,0) (4,0,0) section=ibeam(0.2,0.1,0.01,0.006) n=40\nbeam (0,3,0) (0,3,3) (4,3,3) (4,3,0) section=rect(0.1,0.1) n=40\nfix y=0\nload at=(2,3,0) fy=-20000\n"));
    cases.push(await measureCase("block 20,480 C3D8", "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.0125\nfix x=0\nload x=1 fy=-5000\n"));
    cases.push(await measureCase("block 78,125 C3D8", "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.008\nfix x=0\nload x=1 fy=-5000\n"));
    // last frame: the large model with Mises and the panels open
    await frame();
    report.ok = !cases.some((c) => "error" in c);
  } catch (e) {
    report.ok = false;
    report.error = String(e instanceof Error ? e.message : e);
  }
  report.cases = cases;
  await frame();
  const png = viewer().screenshot();
  await engine.selftestDone(png, JSON.stringify(report, null, 2));
}
