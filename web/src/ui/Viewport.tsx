// The viewport in React: connects Viewer to the UI state and shows
// the toolbar, color legend and information about the picked object.

import { useEffect, useRef, useState } from "react";
import { engine, pick, selectResult, toast } from "../actions";
import { colormapCss } from "../viewer/colormaps";
import { Viewer, type ViewDir } from "../viewer/Viewer";
import { currentField, get, persist, set, useShallow, useStore } from "../store";
import { fmt, Icon, IconButton } from "./ui";
import { convertValue, loadUnits, modelUnit, resultUnit, unitOptions } from "./units";

/** Most recently fetched results (key → data), for fast switching and animation. */
const cache = new Map<string, Promise<{ disp: Float32Array; values: Float32Array | null; min: number; max: number; maxU: number }>>();

function fetchFrame(id: number, step: number, frame: number, field: string, comp: string) {
  const key = `${id}/${step}/${frame}/${field}/${comp}`;
  let p = cache.get(key);
  if (!p) {
    p = Promise.all([engine.displacement(id, step, frame), engine.field(id, step, frame, field, comp)]).then(([disp, f]) => {
      let maxU = 0;
      for (let i = 0; i < disp.length; i += 3) maxU = Math.max(maxU, Math.hypot(disp[i], disp[i + 1], disp[i + 2]));
      return { disp, values: f.values, min: f.min, max: f.max, maxU };
    });
    cache.set(key, p);
    p.catch(() => cache.delete(key));
    if (cache.size > 64) cache.delete(cache.keys().next().value!);
  }
  return p;
}

export function clearResultCache() {
  cache.clear();
}

export function Viewport() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<Viewer | null>(null);
  const lastId = useRef<number | null>(null);
  const lastNodes = useRef(0);
  const [viewMenu, setViewMenu] = useState(false);
  const [unitMenu, setUnitMenu] = useState(false);
  const loadUnit = useStore((s) => s.loadUnit);
  const resultUnits = useStore((s) => s.resultUnits);
  const [is2d, setIs2d] = useState(false);
  const [range, setRange] = useState<{ min: number; max: number } | null>(null);

  const id = useStore((s) => s.id);
  const mesh = useStore((s) => s.mesh);
  const meshVersion = useStore((s) => s.meshVersion);
  const mode = useStore((s) => s.mode);
  const rsel = useStore((s) => s.rsel);
  const results = useStore((s) => s.results);
  const session = useStore((s) => s.session);
  const symbolStep = useStore((s) => s.symbolStep);
  const model = useStore((s) => s.model);
  const deformAuto = useStore((s) => s.deformAuto);
  const deformFactor = useStore((s) => s.deformFactor);
  const animate = useStore((s) => s.animate);
  const pickInfo = useStore((s) => s.pick);
  const opts = useShallow((s) => ({ surfaces: s.surfaces, edges: s.edges, profiles: s.profiles, undeformed: s.undeformed, symbols: s.symbols, levels: s.levels, colormap: s.colormap, pickMode: s.pickMode, dark: s.dark }));
  const field = useStore(currentField);
  const busy = useStore((s) => s.busy);

  // create the viewer once
  useEffect(() => {
    const v = new Viewer(
      canvasRef.current!,
      (h) => {
        const into = get().onNodePick;
        if (into && h?.kind === "node") {
          set({ onNodePick: null });
          into(h.index);
          return;
        }
        if (h) pick(h.kind, h.index);
        else set({ pick: null });
      },
      { ...get(), pickMode: get().pickMode },
    );
    viewerRef.current = v;
    (window as unknown as { __viewer: Viewer }).__viewer = v;
    return () => v.dispose();
  }, []);

  useEffect(() => {
    viewerRef.current?.setOptions(opts);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opts.surfaces, opts.edges, opts.profiles, opts.undeformed, opts.symbols, opts.levels, opts.colormap, opts.pickMode, opts.dark]);

  useEffect(() => {
    const v = viewerRef.current;
    if (!v || !mesh) return;
    // same model: keep the camera, except when the first geometry is added to an empty model
    const keep = lastId.current === id && lastNodes.current > 0;
    lastId.current = id;
    lastNodes.current = mesh.nodes;
    if (!keep) clearResultCache();
    v.setMesh(mesh, keep);
    setIs2d(v.mode2d);
  }, [mesh, meshVersion, id]);

  // symbols in model mode
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || id === null || !mesh) return;
    if (mode !== "model") {
      v.setSymbols(null);
      return;
    }
    let live = true;
    // load values in the chosen unit; without a unit system the numbers are shown as written
    const system = session?.units;
    const fmtLoad = loadUnit
      ? (val: number, moment: boolean) => {
          const q = moment ? "moment" : "force";
          const to = moment ? (loadUnits.find(([f]) => f === loadUnit)?.[1] ?? null) : loadUnit;
          const from = modelUnit(q, system);
          return from && to ? `${fmt(convertValue(val, q, from, to))} ${to}` : fmt(val);
        }
      : null;
    engine
      .symbols(id, symbolStep)
      .then((s) => live && v.setSymbols(s, fmtLoad))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [id, mode, symbolStep, mesh, model, loadUnit, session?.units]);

  // results
  useEffect(() => {
    const v = viewerRef.current;
    if (!v || id === null || !mesh) return;
    if (mode !== "results" || !rsel) {
      v.animateMode(false);
      v.setDisplacement(null);
      v.setField(null, 0, 1);
      v.setScale(0);
      setRange(null);
      return;
    }
    let live = true;
    fetchFrame(id, rsel.step, rsel.frame, rsel.field, rsel.component)
      .then((r) => {
        if (!live) return;
        const auto = r.maxU > 0 ? (0.1 * v.bounds.radius * 2) / r.maxU : 0;
        const fr = results[rsel.step]?.frames[rsel.frame];
        const isMode = (fr?.mode ?? -1) > 0;
        // small displacements are magnified automatically, large ones are shown to scale
        const scale = deformAuto ? (isMode || r.maxU < 0.02 * v.bounds.radius ? auto : 1) : deformFactor;
        v.setDisplacement(r.disp);
        v.setField(r.values, r.min, r.max);
        if (animate !== "mode") v.setScale(scale);
        else {
          v.animateMode(false);
          v.setScale(scale);
          v.animateMode(true);
        }
        if (deformAuto) set({ deformFactor: Number(scale.toPrecision(3)) });
        setRange({ min: r.min, max: r.max });
      })
      .catch((e) => toast("error", String(e?.message ?? e)));
    return () => {
      live = false;
    };
  }, [id, mode, rsel, mesh, deformAuto, deformFactor, animate, results]);

  useEffect(() => {
    if (animate !== "mode") viewerRef.current?.animateMode(false);
  }, [animate]);

  // frame animation: prefetch and step
  useEffect(() => {
    if (animate !== "frames" || mode !== "results" || !rsel || id === null) return;
    const st = results[rsel.step];
    if (!st || st.frames.length < 2) return;
    for (const f of st.frames) fetchFrame(id, rsel.step, f.index, rsel.field, rsel.component);
    const t = window.setInterval(() => {
      const r = get().rsel;
      if (!r) return;
      selectResult({ frame: (r.frame + 1) % st.frames.length });
    }, Math.max(60, 2400 / st.frames.length));
    return () => window.clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animate, mode, rsel?.step, rsel?.field, rsel?.component, id, results]);

  useEffect(() => {
    const v = viewerRef.current;
    if (!v) return;
    v.setHighlight(pickInfo ? { kind: pickInfo.kind === "node" ? "node" : "element", index: pickInfo.index } : null);
  }, [pickInfo]);

  // selection in the tool panel
  const marked = useStore((s) => s.marked);
  const waitPick = useStore((s) => s.onNodePick !== null);
  useEffect(() => {
    viewerRef.current?.setMarked(mode === "model" ? marked : null);
  }, [marked, mesh, meshVersion, mode]);

  // keys for views
  useEffect(() => {
    const keys: Record<string, ViewDir> = { "1": "front", "2": "back", "3": "top", "4": "bottom", "5": "left", "6": "right", "0": "iso" };
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select") || e.ctrlKey || e.metaKey || e.altKey) return;
      const v = viewerRef.current;
      if (!v) return;
      if (e.key === "f" || e.key === "F") v.fit();
      else if (keys[e.key]) v.view(keys[e.key]);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);

  const v = viewerRef.current;
  const frame = rsel ? results[rsel.step]?.frames[rsel.frame] : null;
  const timeUnit = session?.unit_symbols.time;
  const showLegend = mode === "results" && rsel && range && field;
  const levels = opts.levels;
  // at most 13 labels; with many bands every second/third band boundary is shown
  const nb = levels || 8;
  const every = Math.ceil(nb / 12);
  // the field's values in the unit chosen in the legend
  const ru = resultUnit(field?.unit, resultUnits);
  const conv = (x: number, from = field?.unit, r = ru) => (r.q && from ? convertValue(x, r.q, from, r.to) : x);
  const setResultUnit = (q: string, u: string) => {
    const next = { ...get().resultUnits, [q]: u };
    set({ resultUnits: next });
    persist("resultUnits", next);
  };
  const span = showLegend ? Math.abs(range!.max - range!.min) : 0;
  // rounding noise around zero (e.g. -1.7e-18) is shown as 0
  const tick = (x: number) => conv(Math.abs(x) < span * 1e-9 ? 0 : x);
  const ticks = showLegend ? Array.from({ length: Math.floor(nb / every) + 1 }, (_, i) => tick(range!.max - ((range!.max - range!.min) * i * every) / nb)) : [];

  const shot = () => {
    if (!v) return;
    const a = document.createElement("a");
    a.href = v.screenshot();
    a.download = `rustyfem-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.png`;
    a.click();
  };

  return (
    <div className="viewport">
      <canvas ref={canvasRef} className="gl" />
      {!mesh && <div className="vp-hint">No model open</div>}
      {mesh && model && model.elements === 0 && mode === "model" && (
        <div className="vp-hint">Empty model – add materials and geometry with the tools in the panel on the right</div>
      )}
      {waitPick && (
        <div className="vp-banner">
          Click a node in the model
          <button type="button" className="btn small" onClick={() => set({ onNodePick: null })}>
            Cancel
          </button>
        </div>
      )}
      {busy && (
        <div className="busy">
          <span className="spinner" />
          {busy} …
        </div>
      )}
      <div className="vp-tools">
        <div className="tool-group">
          <IconButton icon="fit" title="Fit view" kbd="F" onClick={() => v?.fit()} />
          <div className="menu-wrap">
            <IconButton icon="cube" title="Views" active={viewMenu} onClick={() => setViewMenu(!viewMenu)} />
            {viewMenu && (
              <div className="menu" onMouseLeave={() => setViewMenu(false)}>
                {(
                  [
                    ["iso", "Isometric", "0"],
                    ["front", "Front (+Z)", "1"],
                    ["back", "Back", "2"],
                    ["top", "Top (+Y)", "3"],
                    ["bottom", "Bottom", "4"],
                    ["left", "Left", "5"],
                    ["right", "Right (+X)", "6"],
                  ] as [ViewDir, string, string][]
                ).map(([d, l, k]) => (
                  <button
                    key={d}
                    type="button"
                    disabled={is2d && d !== "front"}
                    onClick={() => {
                      v?.view(d);
                      setViewMenu(false);
                    }}
                  >
                    {l}
                    <kbd>{k}</kbd>
                  </button>
                ))}
              </div>
            )}
          </div>
          <IconButton
            icon={is2d ? "square" : "cube"}
            title={is2d ? "2D mode (click for 3D)" : "3D mode (click for 2D)"}
            label={is2d ? "2D" : "3D"}
            onClick={() => {
              if (!v) return;
              v.set2D(!v.mode2d);
              setIs2d(v.mode2d);
              if (!v.mode2d) v.view("iso");
            }}
          />
        </div>
        <div className="tool-group">
          <IconButton icon="surface" title="Surfaces" active={opts.surfaces} onClick={() => set((s) => ({ surfaces: !s.surfaces }))} />
          <IconButton icon="edges" title="Element edges" active={opts.edges} onClick={() => set((s) => ({ edges: !s.edges }))} />
          <IconButton icon="profile" title="Profiles and shell thickness in 3D" active={opts.profiles} onClick={() => set((s) => ({ profiles: !s.profiles }))} />
          <IconButton icon="ghost" title="Undeformed shape" active={opts.undeformed} onClick={() => set((s) => ({ undeformed: !s.undeformed }))} />
          <IconButton icon="anchor" title="Boundary conditions and loads" active={opts.symbols} onClick={() => set((s) => ({ symbols: !s.symbols }))} />
          <div className="menu-wrap">
            <button type="button" className={`ibtn unitbtn${unitMenu ? " active" : ""}`} title="Load values in the model" onClick={() => setUnitMenu(!unitMenu)}>
              {loadUnit ?? "–"}
            </button>
            {unitMenu && (
              <div className="menu" onMouseLeave={() => setUnitMenu(false)}>
                {loadUnits.map(([f, m]) => (
                  <button
                    key={f}
                    type="button"
                    className={loadUnit === f ? "on" : ""}
                    onClick={() => {
                      set({ loadUnit: f });
                      persist("loadUnit", f);
                      setUnitMenu(false);
                    }}
                  >
                    Load values in {f}
                    <kbd>{m}</kbd>
                  </button>
                ))}
                <button
                  type="button"
                  className={loadUnit === null ? "on" : ""}
                  onClick={() => {
                    set({ loadUnit: null });
                    persist("loadUnit", null);
                    setUnitMenu(false);
                  }}
                >
                  Hide load values
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="tool-group">
          <IconButton icon="cursor" title="Pick elements" active={opts.pickMode === "element"} onClick={() => set({ pickMode: "element" })} />
          <IconButton icon="node" title="Pick nodes" active={opts.pickMode === "node"} onClick={() => set({ pickMode: "node" })} />
        </div>
        <div className="tool-group">
          <IconButton icon="camera" title="Save image (PNG)" onClick={shot} />
        </div>
      </div>

      {showLegend && (
        <div className="legend">
          <div className="legend-title">
            <strong>{field!.name}</strong>
            {rsel!.component !== field!.name && <span>{rsel!.component}</span>}
            {ru.q ? (
              <select className="legend-unit" title="Unit" value={ru.to} onChange={(e) => setResultUnit(ru.q!, e.target.value)}>
                {unitOptions(ru.q).map((u) => (
                  <option key={u} value={u}>
                    {u}
                  </option>
                ))}
              </select>
            ) : (
              field!.unit && <span>[{field!.unit}]</span>
            )}
          </div>
          <div className="legend-body">
            <div className="legend-bar" style={{ background: colormapCss(opts.colormap, levels) }} />
            <div className="legend-ticks">
              {ticks.map((t, i) => (
                <span key={i}>{fmt(t)}</span>
              ))}
            </div>
          </div>
          <div className="legend-foot">
            <div>{results[rsel!.step]?.name}</div>
            <div>{frame ? (frame.mode > 0 ? `Mode ${frame.mode}: ${frame.description || `${fmt(frame.value)}`}` : `Time ${fmt(frame.value)}${timeUnit ? ` ${timeUnit}` : ""}`) : ""}</div>
            <div>Deformation ×{fmt(deformFactor, 3)}</div>
          </div>
        </div>
      )}

      {pickInfo && (
        <div className="pickcard">
          <header>
            <Icon name={pickInfo.kind === "node" ? "node" : "cursor"} size={14} />
            <strong>
              {pickInfo.kind === "node" ? "Node" : "Element"} {pickInfo.name}
            </strong>
            <button type="button" className="x" aria-label="Close" onClick={() => set({ pick: null })}>
              <Icon name="close" size={14} />
            </button>
          </header>
          <dl>
            {pickInfo.details.map(([k, val]) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{val}</dd>
              </div>
            ))}
            {pickInfo.values.map(([k, val, unit]) => {
              const r = resultUnit(unit, resultUnits);
              return (
                <div key={k} className="num">
                  <dt>{k}</dt>
                  <dd>
                    {fmt(val == null ? NaN : conv(val, unit, r), 5)}
                    {r.to && <span className="unit"> {r.to}</span>}
                  </dd>
                </div>
              );
            })}
          </dl>
        </div>
      )}
    </div>
  );
}
