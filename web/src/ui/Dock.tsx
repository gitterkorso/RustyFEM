// Körpanelen: lösare, minne, framsteg, konvergens och meddelanden.

import { useEffect, useRef } from "react";
import { bytes, cancel, estimate, solve } from "../actions";
import { set, useStore } from "../store";
import { Icon, IconButton, Select, Toggle } from "./ui";

function Sparkline({ values }: { values: number[] }) {
  const v = values.filter((x) => Number.isFinite(x) && x > 0).map((x) => Math.log10(x));
  if (v.length < 2) return <svg className="spark" />;
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const W = 160;
  const H = 28;
  const d = v.map((y, i) => `${i ? "L" : "M"}${((i / (v.length - 1)) * W).toFixed(1)},${(H - 2 - ((y - lo) / Math.max(hi - lo, 1e-9)) * (H - 4)).toFixed(1)}`).join("");
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} aria-label="Residual (log)">
      <path d={d} />
    </svg>
  );
}

export function Dock() {
  const run = useStore((s) => s.run);
  const logLines = useStore((s) => s.log);
  const solvers = useStore((s) => s.solvers);
  const id = useStore((s) => s.id);
  const model = useStore((s) => s.model);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logLines.length]);
  useEffect(() => {
    if (id !== null) estimate();
  }, [id, model?.nodes, model?.elements, run.solver]);
  const est = run.estimate;
  const pct = Math.round(run.progress * 100);
  return (
    <div className="dock">
      <div className="dock-bar">
        <div className="dock-run">
          {run.running ? (
            <button type="button" className="btn danger" onClick={cancel} title="Avbryt (Esc)">
              <Icon name="stop" size={14} /> Avbryt
            </button>
          ) : (
            <button type="button" className="btn primary" onClick={solve} disabled={id === null} title="Lös (F5)">
              <Icon name="play" size={14} /> Lös
            </button>
          )}
          <div className="progress" aria-label="Framsteg">
            <div className={`progress-fill${run.running ? " live" : ""}`} style={{ width: `${pct}%` }} />
          </div>
          <span className="mono small">{run.running || pct > 0 ? `${pct} %` : ""}</span>
          {run.running && (
            <span className="muted small ellipsis">
              steg {run.stepIndex + 1}/{run.steps} {run.stepName} · inkrement {run.increment}
            </span>
          )}
          {run.running && <Sparkline values={run.residuals} />}
        </div>
        <div className="dock-opts">
          <Select
            value={run.solver ?? ""}
            options={[{ value: "", label: "Lösare: automatisk" }, ...solvers.map((s) => ({ value: s, label: s }))]}
            onChange={(v) => set((s) => ({ run: { ...s.run, solver: v || null } }))}
            disabled={run.running}
          />
          <Toggle checked={run.allFrames} onChange={(v) => set((s) => ({ run: { ...s.run, allFrames: v } }))} label="Alla inkrement" hint="Spara en frame per inkrement (för animering och kurvor)" />
          {est && (
            <span className={`mem${est.fits ? "" : " over"}`} title={est.note}>
              {bytes(est.solver.startsWith("iterative") ? est.iterative_bytes : est.direct_bytes)} / {bytes(est.budget_bytes)}
            </span>
          )}
          <IconButton icon="close" title="Rensa meddelanden" onClick={() => set({ log: [] })} />
        </div>
      </div>
      <div className="log" ref={ref} role="log">
        {logLines.length === 0 && <div className="muted">Meddelanden från motorn visas här.</div>}
        {logLines.map((l, i) => (
          <div key={i} className={`ll ${l.kind}`}>
            <span className="ts">{new Date(l.t).toLocaleTimeString("sv")}</span>
            <span>{l.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
