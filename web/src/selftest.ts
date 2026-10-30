// Självtest i det riktiga appfönstret (`rustyfem-app --selftest KATALOG`):
// öppnar modeller, löser dem med motorn, växlar resultat och mäter bildtiden.
// Skärmbild och tider skrivs till katalogen, sedan avslutas appen.

import { engine, newModel, selectResult, solve } from "./actions";
import { get, set } from "./store";
import type { Viewer } from "./viewer/Viewer";

const viewer = () => (window as unknown as { __viewer: Viewer }).__viewer;

async function until(what: string, ok: () => boolean, ms = 90_000) {
  const t0 = performance.now();
  while (!ok()) {
    if (performance.now() - t0 > ms) {
      const st = get();
      throw new Error(`självtest: väntade för länge på ${what} (busy=${st.busy}, running=${st.run.running}, mode=${st.mode}, results=${st.results.length}, logg: ${st.log.slice(-3).map((l) => l.text).join(" / ")})`);
    }
    await new Promise((r) => setTimeout(r, 15));
  }
}

const frame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

/** Mål för ett responsivt gränssnitt (ms). */
const TARGETS: Record<string, number> = {
  bild_modell_ms: 16.7,
  bild_resultat_ms: 16.7,
  forsta_resultatbild_ms: 100,
  faltbyte_ms: 100,
  oppna_ms: 2000,
  // gränssnittet under lösningen: längsta och 95:e percentilen av bildintervallen
  under_losning_max_ms: 50,
  under_losning_p95_ms: 25,
};

function checkTargets(r: Record<string, number | string>) {
  const miss: string[] = [];
  for (const [k, lim] of Object.entries(TARGETS)) {
    const v = r[k];
    if (typeof v === "number" && v > lim) miss.push(`${k} ${v} > ${lim}`);
  }
  r.mal = miss.length ? `missade: ${miss.join(", ")}` : "alla mål nådda";
  return miss.length === 0;
}

async function measureCase(name: string, script: string) {
  const r: Record<string, number | string> = { modell: name };
  let t = performance.now();
  await newModel(script);
  await until("modellen", () => get().mesh !== null && get().busy === null);
  await frame();
  r.oppna_ms = Math.round(performance.now() - t);
  const m = get().model!;
  r.noder = m.nodes;
  r.element = m.elements;
  Object.assign(r, viewer().stats);
  r.bild_modell_ms = Number(viewer().benchmark(60).toFixed(2));
  t = performance.now();
  // bildintervall medan lösningen pågår (modellen ritas om i varje bild)
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
  await until("lösningen", () => !get().run.running && get().results.length > 0 && get().mode === "results");
  live = false;
  r.losning_ms = Math.round(performance.now() - t);
  const g = gaps.slice(2).sort((a, b) => a - b);
  if (g.length) {
    r.under_losning_bilder = g.length;
    r.under_losning_max_ms = Number(g[g.length - 1].toFixed(1));
    r.under_losning_p95_ms = Number(g[Math.floor(g.length * 0.95)].toFixed(1));
  }
  t = performance.now();
  await until("resultatbilden", () => document.querySelector(".legend") !== null);
  await frame();
  r.forsta_resultatbild_ms = Math.round(performance.now() - t);
  t = performance.now();
  set({ mode: "results" });
  selectResult({ field: "U", component: "Magnitude" });
  await until("fältbytet", () => document.querySelector(".legend-title strong")?.textContent === "U");
  await frame();
  r.faltbyte_ms = Math.round(performance.now() - t);
  // ett elementfält som finns (S för kontinuum, SF/SM för balkar)
  const st = get().results[get().rsel!.step];
  const ef = st.fields.find((f) => f.name === "S") ?? st.fields.find((f) => f.position === "element") ?? st.fields[0];
  r.elementfalt = ef.name;
  set({ mode: "results" });
  selectResult({ field: ef.name, component: ef.components[0] });
  await until(ef.name, () => document.querySelector(".legend-title strong")?.textContent === ef.name);
  await frame();
  r.bild_resultat_ms = Number(viewer().benchmark(60).toFixed(2));
  const log = get().log.filter((l) => l.kind === "error");
  if (log.length) r.fel = log.map((l) => l.text).join(" | ");
  if (!checkTargets(r)) r.fel = String(r.mal);
  return r;
}

export async function runSelftest() {
  if (!(await engine.selftest().catch(() => false))) return;
  const report: Record<string, unknown> = {
    mal_ms: TARGETS, tid: new Date().toISOString(), fonster: `${window.innerWidth}x${window.innerHeight}`, dpr: window.devicePixelRatio };
  const cases: Record<string, unknown>[] = [];
  try {
    window.confirm = () => true;
    set({ left: true, right: true, dock: true });
    cases.push(await measureCase("ram av balkar", "model 3d\nmaterial steel\nbeam (0,0,0) (0,3,0) (4,3,0) (4,0,0) section=ibeam(0.2,0.1,0.01,0.006) n=40\nbeam (0,3,0) (0,3,3) (4,3,3) (4,3,0) section=rect(0.1,0.1) n=40\nfix y=0\nload at=(2,3,0) fy=-20000\n"));
    cases.push(await measureCase("kloss 20 480 C3D8", "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.0125\nfix x=0\nload x=1 fy=-5000\n"));
    cases.push(await measureCase("kloss 78 125 C3D8", "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.008\nfix x=0\nload x=1 fy=-5000\n"));
    // sista bilden: den stora modellen med Mises och panelerna öppna
    await frame();
    report.ok = !cases.some((c) => "fel" in c);
  } catch (e) {
    report.ok = false;
    report.fel = String(e instanceof Error ? e.message : e);
  }
  report.fall = cases;
  await frame();
  const png = viewer().screenshot();
  await engine.selftestDone(png, JSON.stringify(report, null, 2));
}
