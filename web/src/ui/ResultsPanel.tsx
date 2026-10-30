// Val av steg, frame och fält, deformation, animering, färgskala och
// history-utdata.

import { useEffect, useMemo, useState } from "react";
import { engine, selectResult } from "../actions";
import { colormapCss, colormapNames } from "../viewer/colormaps";
import { currentField, persist, set, useStore } from "../store";
import { Empty, fmt, IconButton, NumField, Section, Segmented, Select, Toggle } from "./ui";

function Plot({ data, label }: { data: [number, number][]; label: string }) {
  const W = 260;
  const H = 150;
  const pad = { l: 46, r: 8, t: 8, b: 22 };
  const pts = data.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pts.length < 2) return <Empty>För få punkter.</Empty>;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  if (x1 === x0) x1 = x0 + 1;
  if (y1 === y0) {
    y0 -= 1;
    y1 += 1;
  }
  const sx = (x: number) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const sy = (y: number) => H - pad.b - ((y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join("");
  return (
    <svg className="plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label}>
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line x1={pad.l} x2={W - pad.r} y1={sy(y0 + f * (y1 - y0))} y2={sy(y0 + f * (y1 - y0))} className="gridline" />
          <text x={pad.l - 4} y={sy(y0 + f * (y1 - y0)) + 3} textAnchor="end">
            {fmt(y0 + f * (y1 - y0), 3)}
          </text>
        </g>
      ))}
      <text x={pad.l} y={H - 6}>
        {fmt(x0, 3)}
      </text>
      <text x={W - pad.r} y={H - 6} textAnchor="end">
        {fmt(x1, 3)}
      </text>
      <path d={d} className="curve" />
    </svg>
  );
}

export function ResultsPanel() {
  const results = useStore((s) => s.results);
  const rsel = useStore((s) => s.rsel);
  const id = useStore((s) => s.id);
  const field = useStore(currentField);
  const stale = useStore((s) => s.session?.results_stale ?? false);
  const tu = useStore((s) => s.session?.unit_symbols.time);
  const deformAuto = useStore((s) => s.deformAuto);
  const deformFactor = useStore((s) => s.deformFactor);
  const animate = useStore((s) => s.animate);
  const levels = useStore((s) => s.levels);
  const cmap = useStore((s) => s.colormap);
  const undeformed = useStore((s) => s.undeformed);
  const [hist, setHist] = useState<number | null>(null);
  const [histData, setHistData] = useState<[number, number][] | null>(null);

  const step = rsel ? results[rsel.step] : null;
  useEffect(() => {
    setHist(null);
    setHistData(null);
  }, [rsel?.step]);
  useEffect(() => {
    if (id === null || !rsel || hist === null) return;
    let live = true;
    engine.history(id, rsel.step, hist).then((d) => live && setHistData(d));
    return () => {
      live = false;
    };
  }, [id, rsel?.step, hist]);

  const isModal = useMemo(() => !!step?.frames.some((f) => f.mode > 0), [step]);
  if (!results.length || !rsel || !step) return <Empty>Inga resultat än. Lös modellen (F5).</Empty>;
  const frame = step.frames[rsel.frame];
  return (
    <>
      {stale && <div className="note warn">Modellen har ändrats sedan lösningen – resultaten är inaktuella.</div>}
      <Section title="Steg och frame">
        <Select label="Steg" value={rsel.step} options={results.map((s) => ({ value: s.index, label: `${s.index + 1}. ${s.name} (${s.procedure})` }))} onChange={(v) => selectResult({ step: v })} />
        {step.frames.length > 1 ? (
          <>
            <label className="field">
              <span className="flabel">
                {isModal ? "Mod" : "Frame"} {rsel.frame + 1} / {step.frames.length}
                <span className="muted"> · {frame?.mode && frame.mode > 0 ? frame.description || `f = ${fmt(frame.value)}` : `t = ${fmt(frame?.value)}${tu ? ` ${tu}` : ""}`}</span>
              </span>
              <input type="range" min={0} max={step.frames.length - 1} value={rsel.frame} onChange={(e) => selectResult({ frame: Number(e.target.value) })} />
            </label>
            <div className="row-btns">
              <IconButton icon="undo" title="Föregående" onClick={() => selectResult({ frame: Math.max(0, rsel.frame - 1) })} />
              <IconButton icon="redo" title="Nästa" onClick={() => selectResult({ frame: Math.min(step.frames.length - 1, rsel.frame + 1) })} />
              <span className="grow" />
              <span className="muted">
                {step.increments} inkr · {step.iterations} it
              </span>
            </div>
          </>
        ) : (
          <div className="kv">
            <span>Frame</span>
            <span>{frame?.mode && frame.mode > 0 ? frame.description || `mod ${frame.mode}` : `t = ${fmt(frame?.value)}${tu ? ` ${tu}` : ""}`}</span>
          </div>
        )}
      </Section>
      <Section title="Fält">
        <div className="chips fields">
          {step.fields.map((f) => (
            <button key={f.name} type="button" className={`chip${f.name === rsel.field ? " on" : ""}`} onClick={() => selectResult({ field: f.name })} title={f.position === "node" ? "Nodfält" : "Elementfält (utjämnat till noderna)"}>
              {f.name}
            </button>
          ))}
        </div>
        {field && field.components.length > 1 && (
          <div className="chips">
            {field.components.map((c) => (
              <button key={c} type="button" className={`chip small${c === rsel.component ? " on" : ""}`} onClick={() => selectResult({ component: c })}>
                {c}
              </button>
            ))}
          </div>
        )}
      </Section>
      <Section title="Deformation">
        <Toggle checked={deformAuto} onChange={(v) => set({ deformAuto: v })} label="Automatisk skala" />
        <NumField label="Skalfaktor" value={deformFactor} min={0} disabled={deformAuto} onCommit={(v) => set({ deformFactor: v, deformAuto: false })} />
        <Toggle checked={undeformed} onChange={(v) => set({ undeformed: v })} label="Visa odeformerad form" />
        <Segmented
          small
          value={animate}
          onChange={(v) => set({ animate: v })}
          options={[
            { value: "off", label: "Stilla" },
            { value: "frames", label: "Frames", icon: "film" },
            { value: "mode", label: "Svängning", icon: "wave" },
          ]}
        />
      </Section>
      <Section title="Färgskala">
        <div className="cmaps">
          {colormapNames.map((n) => (
            <button
              key={n}
              type="button"
              className={`cmap${n === cmap ? " on" : ""}`}
              style={{ background: colormapCss(n, 0, "to right") }}
              title={n}
              aria-label={n}
              onClick={() => {
                set({ colormap: n });
                persist("colormap", n);
              }}
            />
          ))}
        </div>
        <Select
          label="Nivåer"
          value={levels}
          options={[0, 6, 8, 10, 12, 16, 24].map((n) => ({ value: n, label: n === 0 ? "Kontinuerlig" : `${n} band` }))}
          onChange={(v) => {
            set({ levels: v });
            persist("levels", v);
          }}
        />
      </Section>
      {step.history.length > 0 && (
        <Section title="History-utdata">
          <Select label="Serie" value={hist ?? -1} options={[{ value: -1, label: "Välj …" }, ...step.history.map((h) => ({ value: h.index, label: `${h.name} · ${h.region}` }))]} onChange={(v) => setHist(v < 0 ? null : v)} />
          {histData && <Plot data={histData} label="History" />}
        </Section>
      )}
    </>
  );
}
