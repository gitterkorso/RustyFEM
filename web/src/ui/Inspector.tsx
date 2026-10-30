// Properties of what is selected in the model tree, with editing
// (commands to the engine that can be undone).

import { useEffect, useState } from "react";
import { bytes, edit, setUnits } from "../actions";
import type { BlockInfo, KeywordInfo, ModelInfo, StepInfo } from "../engine/Engine";
import { set, useStore } from "../store";
import { parseScript } from "../script";
import { ScriptPanel } from "./ScriptPanel";
import { Empty, fmt, NumField, Section, Select, Toggle } from "./ui";
import { columnUnit, dloadUnit, dofUnit, withUnit } from "./units";

/** Column names for known keywords. */
const columns: Record<string, string[]> = {
  ELASTIC: ["E", "ν", "T"],
  PLASTIC: ["σy", "εp", "T"],
  DENSITY: ["ρ"],
  EXPANSION: ["α", "T"],
  CONDUCTIVITY: ["k", "T"],
  "SPECIFIC HEAT": ["c", "T"],
  STATIC: ["Δt start", "Time", "Δt min", "Δt max"],
  "HEAT TRANSFER": ["Δt start", "Time", "Δt min", "Δt max", "ΔT max"],
  DYNAMIC: ["Δt start", "Time", "Δt min", "Δt max"],
  VISCO: ["Δt start", "Time", "Δt min", "Δt max"],
  FREQUENCY: ["Count", "Min", "Max", "Shift"],
  BUCKLE: ["Count", "Max", "Vectors", "Iter."],
  "COUPLED TEMPERATURE-DISPLACEMENT": ["Δt start", "Time", "Δt min", "Δt max", "ΔT max"],
};

function KeywordTable(p: { kw: KeywordInfo; readOnly: boolean; onLine: (line: number, fields: string[]) => void }) {
  const cols = columns[p.kw.name] ?? [];
  const u = useStore((s) => s.session?.unit_symbols);
  const width = Math.max(...p.kw.lines.map((l) => l.length), cols.length ? Math.min(cols.length, 2) : 1);
  return (
    <div className="kw">
      <div className="kw-head">
        <code>*{p.kw.name}</code>
        {p.kw.params
          .filter(([k]) => k !== "NAME")
          .map(([k, v]) => (
            <span key={k} className="param">
              {k}
              {v ? `=${v}` : ""}
            </span>
          ))}
      </div>
      {p.kw.lines.length > 0 && (
        <div className="grid" style={{ gridTemplateColumns: `repeat(${width}, minmax(0, 1fr))` }}>
          {cols.length > 0 && Array.from({ length: width }, (_, i) => <span key={`h${i}`} className="gh">{cols[i] ? withUnit(cols[i], columnUnit(p.kw.name, i, u)) : ""}</span>)}
          {p.kw.lines.slice(0, 60).flatMap((line, li) =>
            Array.from({ length: width }, (_, i) => {
              const raw = line[i];
              if (raw === undefined) return <span key={`${li}-${i}`} />;
              const num = Number(raw.replace(/[dD]/, "e"));
              const isNum = raw.trim() === "" || Number.isFinite(num);
              return p.readOnly || !isNum ? (
                <span key={`${li}-${i}`} className="cell">
                  {isNum && raw.trim() !== "" ? fmt(num, 6) : raw}
                </span>
              ) : (
                <NumField
                  key={`${li}-${i}`}
                  value={raw.trim() === "" ? 0 : num}
                  onCommit={(v) => {
                    const f = [...line];
                    f[i] = String(v);
                    p.onLine(li, f);
                  }}
                />
              );
            }),
          )}
        </div>
      )}
      {p.kw.lines.length > 60 && <div className="muted">… {p.kw.lines.length - 60} more lines</div>}
    </div>
  );
}

function Blocks(p: { title: string; blocks: BlockInfo[]; readOnly: boolean; unit: (what: string) => string; onValue: (block: number, item: number, v: number) => void }) {
  if (!p.blocks.length) return null;
  return (
    <Section title={p.title}>
      {p.blocks.map((b) => (
        <div key={b.index} className="block">
          <div className="block-head">
            <span>{b.kind}</span>
            {b.op_new && <span className="param">OP=NEW</span>}
            {b.amplitude && <span className="param">AMPLITUDE={b.amplitude}</span>}
            <span className="muted">{b.total} lines</span>
          </div>
          <div className="rows">
            {b.items.map((it, i) => (
              <div key={i} className="row3">
                <span className="mono">{it.target}</span>
                <span className="dof">{it.what}</span>
                {p.readOnly ? (
                  <span className="cell">
                    {fmt(it.value, 6)}
                    {p.unit(it.what) && <span className="unit"> {p.unit(it.what)}</span>}
                  </span>
                ) : (
                  <NumField value={it.value} unit={p.unit(it.what)} onCommit={(v) => p.onValue(b.index, i, v)} />
                )}
              </div>
            ))}
            {b.total > b.items.length && <div className="muted">… {b.total - b.items.length} more lines</div>}
          </div>
        </div>
      ))}
    </Section>
  );
}

function StepPanel({ st, readOnly }: { st: StepInfo; readOnly: boolean }) {
  const u = useStore((s) => s.session?.unit_symbols);
  const [name, setName] = useState(st.name);
  useEffect(() => setName(st.name), [st.name]);
  const s = st.index;
  return (
    <>
      <Section title="Step">
        <label className="field">
          <span className="flabel">Name</span>
          <span className="finput">
            <input
              value={name}
              disabled={readOnly}
              onChange={(e) => setName(e.target.value)}
              onBlur={() => name.trim() && name !== st.name && edit({ kind: "StepName", step: s, name })}
              onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
            />
          </span>
        </label>
        <div className="kv">
          <span>Procedure</span>
          <code>*{st.procedure || "–"}</code>
        </div>
        {st.perturbation && (
          <div className="kv">
            <span>Type</span>
            <span>linear perturbation</span>
          </div>
        )}
        <Toggle
          checked={!!st.nlgeom}
          label="Geometric nonlinearity (NLGEOM)"
          onChange={(v) => !readOnly && edit({ kind: "Nlgeom", step: s, value: v })}
        />
        <NumField label="Maximum number of increments (INC)" value={st.max_increments} min={1} disabled={readOnly} onCommit={(v) => edit({ kind: "MaxIncrements", step: s, value: Math.round(v) })} />
      </Section>
      {st.procedure_keyword && (
        <Section title="Procedure">
          <KeywordTable kw={st.procedure_keyword} readOnly={readOnly} onLine={(line, fields) => edit({ kind: "ProcedureData", step: s, line, fields })} />
        </Section>
      )}
      <Blocks title="Boundary conditions" blocks={st.bcs} readOnly={readOnly} unit={(w) => dofUnit(w, false, u)} onValue={(block, item, value) => edit({ kind: "BcValue", step: s, block, item, value })} />
      <Blocks title="Point loads" blocks={st.cloads} readOnly={readOnly} unit={(w) => dofUnit(w, true, u)} onValue={(block, item, value) => edit({ kind: "CloadValue", step: s, block, item, value })} />
      <Blocks title="Distributed loads" blocks={st.dloads} readOnly={readOnly} unit={(w) => dloadUnit(w, u)} onValue={(block, item, value) => edit({ kind: "DloadValue", step: s, block, item, value })} />
      {st.other.length > 0 && (
        <Section title="Other in the step">
          <div className="chips">
            {st.other.map((o, i) => (
              <code key={i}>*{o}</code>
            ))}
          </div>
        </Section>
      )}
    </>
  );
}

function ModelSummary({ m }: { m: ModelInfo }) {
  const session = useStore((s) => s.session);
  const mem = useStore((s) => s.memory);
  const est = useStore((s) => s.run.estimate);
  const unitSystems = useStore((s) => s.unitSystems);
  const size = m.bbox_max.map((v, i) => (v ?? 0) - (m.bbox_min[i] ?? 0));
  return (
    <>
      <Section title="Model">
        {m.heading && <p className="heading">{m.heading}</p>}
        <div className="stats">
          <div>
            <b>{m.nodes.toLocaleString("en-US")}</b>
            <span>nodes</span>
          </div>
          <div>
            <b>{m.elements.toLocaleString("en-US")}</b>
            <span>elements</span>
          </div>
          <div>
            <b>{m.steps.length}</b>
            <span>steps</span>
          </div>
          <div>
            <b>{session?.script_dim ?? m.dim}D</b>
            <span>dimension</span>
          </div>
        </div>
        <div className="kv">
          <span>Size</span>
          <span className="mono">
            {size.map((v) => fmt(v, 3)).join(" × ")}
            {session?.unit_symbols.length && <span className="unit"> {session.unit_symbols.length}</span>}
          </span>
        </div>
        <div className="kv">
          <span>Source</span>
          <span className="mono ellipsis" title={session?.path ?? ""}>
            {session?.path ?? "new model"}
          </span>
        </div>
        {session && (
          <>
            <Select label="Units" value={session.units} options={unitSystems.map(([value, label]) => ({ value, label }))} onChange={setUnits} />
            <div className="muted">Labels only: the engine computes with the numbers as written, and nothing is converted when you switch.</div>
          </>
        )}
        <div className="kv">
          <span>Memory (computer)</span>
          <span>{bytes(mem)}</span>
        </div>
        {est && (
          <div className="kv">
            <span>Estimated requirement</span>
            <span>
              {bytes(est.solver.startsWith("iterative") ? est.iterative_bytes : est.direct_bytes)} · {est.unknowns?.toLocaleString("en-US")} unknowns
            </span>
          </div>
        )}
      </Section>
      {m.warnings.length > 0 && (
        <Section title={`Warnings (${m.warnings.length})`}>
          <ul className="warnings">
            {m.warnings.slice(0, 50).map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </Section>
      )}
    </>
  );
}

export function Inspector() {
  const sel = useStore((s) => s.sel);
  const m = useStore((s) => s.model);
  const session = useStore((s) => s.session);
  const id = useStore((s) => s.id);
  if (!m || !session || !sel) return <Empty>Select something in the model tree.</Empty>;
  const readOnly = session.kind === "rusty";
  // materials in script order; the engine's materials are named M1, M2, … in the same order
  const scriptLine = (): number | null => {
    if (sel.type !== "material") return null;
    const lines = parseScript(session.script ?? "").filter((c) => c.cmd === "material");
    // geometry before the first material line gets default steel first; then the order does not match
    return lines.length === m.materials.length ? (lines[sel.index]?.line ?? null) : null;
  };
  const roNote = readOnly && sel.type !== "model" && sel.type !== "script" && (
    <div className="note">
      .rusty models are built in the Build model panel.{" "}
      <button type="button" className="link" onClick={() => set({ sel: { type: "script" }, scriptEdit: scriptLine() })}>
        {scriptLine() !== null ? "Edit the material" : "Open Build model"}
      </button>
    </div>
  );
  switch (sel.type) {
    case "model":
      return <ModelSummary m={m} />;
    case "script":
      // new panel per model: line numbers and forms belong to the model they were opened in
      return <ScriptPanel key={id ?? -1} />;
    case "material": {
      const mt = m.materials[sel.index];
      if (!mt) return <Empty>The material no longer exists.</Empty>;
      return (
        <>
          {roNote}
          <Section title={`Material ${mt.name}`}>
            <NumField label="Density ρ" unit={session.unit_symbols.density} value={mt.density} min={0} disabled={readOnly} onCommit={(v) => edit({ kind: "Density", material: mt.index, value: v })} />
          </Section>
          <Section title="Behaviors">
            {mt.behaviors.map((b, bi) => (
              <KeywordTable key={bi} kw={b} readOnly={readOnly} onLine={(line, fields) => edit({ kind: "MaterialData", material: mt.index, behavior: bi, line, fields })} />
            ))}
          </Section>
        </>
      );
    }
    case "section": {
      const s = m.sections[sel.index];
      if (!s) return <Empty>The section no longer exists.</Empty>;
      return (
        <>
          {roNote}
          <Section title="Section">
            <div className="kv">
              <span>Element set</span>
              <code>{s.elset}</code>
            </div>
            <div className="kv">
              <span>Type</span>
              <span>{s.kind}</span>
            </div>
            <div className="kv">
              <span>Material</span>
              <button
                type="button"
                className="link"
                onClick={() => {
                  const i = m.materials.findIndex((x) => x.name === s.material);
                  if (i >= 0) set({ sel: { type: "material", index: i } });
                }}
              >
                {s.material || "–"}
              </button>
            </div>
            <div className="kv">
              <span>Elements</span>
              <span>{s.elements}</span>
            </div>
            {s.shape && (
              <div className="kv">
                <span>Profile</span>
                <span>{s.shape}</span>
              </div>
            )}
            {s.thickness !== null && (
              <NumField
                label={s.kind === "SHELL" ? "Thickness" : "Thickness / area"}
                unit={s.kind === "SHELL" ? session.unit_symbols.length : [session.unit_symbols.length, session.unit_symbols.area].filter(Boolean).join(" / ")}
                value={s.thickness}
                min={0}
                disabled={readOnly}
                onCommit={(v) => edit({ kind: "SectionThickness", section: s.index, value: v })}
              />
            )}
          </Section>
        </>
      );
    }
    case "step": {
      const st = m.steps[sel.index];
      if (!st) return <Empty>The step no longer exists.</Empty>;
      return (
        <>
          {roNote}
          <StepPanel st={st} readOnly={readOnly} />
        </>
      );
    }
    case "initial":
      return (
        <>
          {roNote}
          <Blocks title="Boundary conditions in model data" blocks={m.initial_bcs} readOnly={readOnly} unit={(w) => dofUnit(w, false, session.unit_symbols)} onValue={(block, item, value) => edit({ kind: "BcValue", step: null, block, item, value })} />
        </>
      );
    case "set": {
      const list = sel.kind === "nset" ? m.nsets : sel.kind === "elset" ? m.elsets : m.surfaces;
      const it = list.find((x) => x.name === sel.name);
      return (
        <Section title={sel.kind === "nset" ? "Node set" : sel.kind === "elset" ? "Element set" : "Surface"}>
          <div className="kv">
            <span>Name</span>
            <code>{sel.name}</code>
          </div>
          <div className="kv">
            <span>{sel.kind === "nset" ? "Nodes" : sel.kind === "elset" ? "Elements" : "Faces/nodes"}</span>
            <span>{it?.count ?? 0}</span>
          </div>
          {sel.kind === "elset" && m.sections.some((s) => s.elset === sel.name) && (
            <div className="kv">
              <span>Section</span>
              <button type="button" className="link" onClick={() => set({ sel: { type: "section", index: m.sections.findIndex((s) => s.elset === sel.name) } })}>
                show
              </button>
            </div>
          )}
        </Section>
      );
    }
    case "etype": {
      const t = m.element_types.find((x) => x.name === sel.name);
      return (
        <Section title="Element type">
          <div className="kv">
            <span>Type</span>
            <code>{sel.name}</code>
          </div>
          <div className="kv">
            <span>Count</span>
            <span>{t?.count ?? 0}</span>
          </div>
        </Section>
      );
    }
  }
}
