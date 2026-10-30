// Egenskaper för det som är markerat i modellträdet, med redigering
// (kommandon mot motorn som går att ångra).

import { useEffect, useState } from "react";
import { bytes, edit, setUnits } from "../actions";
import type { BlockInfo, KeywordInfo, ModelInfo, StepInfo } from "../engine/Engine";
import { set, useStore } from "../store";
import { parseScript } from "../script";
import { ScriptPanel } from "./ScriptPanel";
import { Empty, fmt, NumField, Section, Select, Toggle } from "./ui";
import { columnUnit, dloadUnit, dofUnit, withUnit } from "./units";

/** Kolumnnamn för kända nyckelord. */
const columns: Record<string, string[]> = {
  ELASTIC: ["E", "ν", "T"],
  PLASTIC: ["σy", "εp", "T"],
  DENSITY: ["ρ"],
  EXPANSION: ["α", "T"],
  CONDUCTIVITY: ["k", "T"],
  "SPECIFIC HEAT": ["c", "T"],
  STATIC: ["Δt start", "Tid", "Δt min", "Δt max"],
  "HEAT TRANSFER": ["Δt start", "Tid", "Δt min", "Δt max", "ΔT max"],
  DYNAMIC: ["Δt start", "Tid", "Δt min", "Δt max"],
  VISCO: ["Δt start", "Tid", "Δt min", "Δt max"],
  FREQUENCY: ["Antal", "Min", "Max", "Skift"],
  BUCKLE: ["Antal", "Max", "Vektorer", "Iter."],
  "COUPLED TEMPERATURE-DISPLACEMENT": ["Δt start", "Tid", "Δt min", "Δt max", "ΔT max"],
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
      {p.kw.lines.length > 60 && <div className="muted">… {p.kw.lines.length - 60} rader till</div>}
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
            <span className="muted">{b.total} rader</span>
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
            {b.total > b.items.length && <div className="muted">… {b.total - b.items.length} rader till</div>}
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
      <Section title="Steg">
        <label className="field">
          <span className="flabel">Namn</span>
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
          <span>Procedur</span>
          <code>*{st.procedure || "–"}</code>
        </div>
        {st.perturbation && (
          <div className="kv">
            <span>Typ</span>
            <span>linjär perturbation</span>
          </div>
        )}
        <Toggle
          checked={!!st.nlgeom}
          label="Geometrisk olinjäritet (NLGEOM)"
          onChange={(v) => !readOnly && edit({ kind: "Nlgeom", step: s, value: v })}
        />
        <NumField label="Största antal inkrement (INC)" value={st.max_increments} min={1} disabled={readOnly} onCommit={(v) => edit({ kind: "MaxIncrements", step: s, value: Math.round(v) })} />
      </Section>
      {st.procedure_keyword && (
        <Section title="Procedur">
          <KeywordTable kw={st.procedure_keyword} readOnly={readOnly} onLine={(line, fields) => edit({ kind: "ProcedureData", step: s, line, fields })} />
        </Section>
      )}
      <Blocks title="Randvillkor" blocks={st.bcs} readOnly={readOnly} unit={(w) => dofUnit(w, false, u)} onValue={(block, item, value) => edit({ kind: "BcValue", step: s, block, item, value })} />
      <Blocks title="Punktlaster" blocks={st.cloads} readOnly={readOnly} unit={(w) => dofUnit(w, true, u)} onValue={(block, item, value) => edit({ kind: "CloadValue", step: s, block, item, value })} />
      <Blocks title="Utbredda laster" blocks={st.dloads} readOnly={readOnly} unit={(w) => dloadUnit(w, u)} onValue={(block, item, value) => edit({ kind: "DloadValue", step: s, block, item, value })} />
      {st.other.length > 0 && (
        <Section title="Övrigt i steget">
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
      <Section title="Modell">
        {m.heading && <p className="heading">{m.heading}</p>}
        <div className="stats">
          <div>
            <b>{m.nodes.toLocaleString("sv")}</b>
            <span>noder</span>
          </div>
          <div>
            <b>{m.elements.toLocaleString("sv")}</b>
            <span>element</span>
          </div>
          <div>
            <b>{m.steps.length}</b>
            <span>steg</span>
          </div>
          <div>
            <b>{session?.script_dim ?? m.dim}D</b>
            <span>dimension</span>
          </div>
        </div>
        <div className="kv">
          <span>Storlek</span>
          <span className="mono">
            {size.map((v) => fmt(v, 3)).join(" × ")}
            {session?.unit_symbols.length && <span className="unit"> {session.unit_symbols.length}</span>}
          </span>
        </div>
        <div className="kv">
          <span>Källa</span>
          <span className="mono ellipsis" title={session?.path ?? ""}>
            {session?.path ?? "ny modell"}
          </span>
        </div>
        {session && (
          <>
            <Select label="Enheter" value={session.units} options={unitSystems.map(([value, label]) => ({ value, label }))} onChange={setUnits} />
            <div className="muted">Bara etiketter: motorn räknar med talen som de står, och inget räknas om när du byter.</div>
          </>
        )}
        <div className="kv">
          <span>Minne (dator)</span>
          <span>{bytes(mem)}</span>
        </div>
        {est && (
          <div className="kv">
            <span>Uppskattat behov</span>
            <span>
              {bytes(est.solver.startsWith("iterative") ? est.iterative_bytes : est.direct_bytes)} · {est.unknowns?.toLocaleString("sv")} obekanta
            </span>
          </div>
        )}
      </Section>
      {m.warnings.length > 0 && (
        <Section title={`Varningar (${m.warnings.length})`}>
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
  if (!m || !session || !sel) return <Empty>Välj något i modellträdet.</Empty>;
  const readOnly = session.kind === "rusty";
  // material i ordning i skriptet; motorns material heter M1, M2, … i samma ordning
  const scriptLine = (): number | null => {
    if (sel.type !== "material") return null;
    const lines = parseScript(session.script ?? "").filter((c) => c.cmd === "material");
    // geometri före första materialraden får standardstål först; då stämmer inte ordningen
    return lines.length === m.materials.length ? (lines[sel.index]?.line ?? null) : null;
  };
  const roNote = readOnly && sel.type !== "model" && sel.type !== "script" && (
    <div className="note">
      .rusty-modeller byggs i panelen Bygg modell.{" "}
      <button type="button" className="link" onClick={() => set({ sel: { type: "script" }, scriptEdit: scriptLine() })}>
        {scriptLine() !== null ? "Ändra materialet" : "Öppna Bygg modell"}
      </button>
    </div>
  );
  switch (sel.type) {
    case "model":
      return <ModelSummary m={m} />;
    case "script":
      // ny panel per modell: radnummer och formulär hör till den modell de öppnades i
      return <ScriptPanel key={id ?? -1} />;
    case "material": {
      const mt = m.materials[sel.index];
      if (!mt) return <Empty>Materialet finns inte längre.</Empty>;
      return (
        <>
          {roNote}
          <Section title={`Material ${mt.name}`}>
            <NumField label="Densitet ρ" unit={session.unit_symbols.density} value={mt.density} min={0} disabled={readOnly} onCommit={(v) => edit({ kind: "Density", material: mt.index, value: v })} />
          </Section>
          <Section title="Beteenden">
            {mt.behaviors.map((b, bi) => (
              <KeywordTable key={bi} kw={b} readOnly={readOnly} onLine={(line, fields) => edit({ kind: "MaterialData", material: mt.index, behavior: bi, line, fields })} />
            ))}
          </Section>
        </>
      );
    }
    case "section": {
      const s = m.sections[sel.index];
      if (!s) return <Empty>Tvärsnittet finns inte längre.</Empty>;
      return (
        <>
          {roNote}
          <Section title="Tvärsnitt">
            <div className="kv">
              <span>Elementset</span>
              <code>{s.elset}</code>
            </div>
            <div className="kv">
              <span>Typ</span>
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
              <span>Element</span>
              <span>{s.elements}</span>
            </div>
            {s.shape && (
              <div className="kv">
                <span>Profil</span>
                <span>{s.shape}</span>
              </div>
            )}
            {s.thickness !== null && (
              <NumField
                label={s.kind === "SHELL" ? "Tjocklek" : "Tjocklek / area"}
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
      if (!st) return <Empty>Steget finns inte längre.</Empty>;
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
          <Blocks title="Randvillkor i modelldata" blocks={m.initial_bcs} readOnly={readOnly} unit={(w) => dofUnit(w, false, session.unit_symbols)} onValue={(block, item, value) => edit({ kind: "BcValue", step: null, block, item, value })} />
        </>
      );
    case "set": {
      const list = sel.kind === "nset" ? m.nsets : sel.kind === "elset" ? m.elsets : m.surfaces;
      const it = list.find((x) => x.name === sel.name);
      return (
        <Section title={sel.kind === "nset" ? "Nodset" : sel.kind === "elset" ? "Elementset" : "Yta"}>
          <div className="kv">
            <span>Namn</span>
            <code>{sel.name}</code>
          </div>
          <div className="kv">
            <span>{sel.kind === "nset" ? "Noder" : sel.kind === "elset" ? "Element" : "Ytor/noder"}</span>
            <span>{it?.count ?? 0}</span>
          </div>
          {sel.kind === "elset" && m.sections.some((s) => s.elset === sel.name) && (
            <div className="kv">
              <span>Tvärsnitt</span>
              <button type="button" className="link" onClick={() => set({ sel: { type: "section", index: m.sections.findIndex((s) => s.elset === sel.name) } })}>
                visa
              </button>
            </div>
          )}
        </Section>
      );
    }
    case "etype": {
      const t = m.element_types.find((x) => x.name === sel.name);
      return (
        <Section title="Elementtyp">
          <div className="kv">
            <span>Typ</span>
            <code>{sel.name}</code>
          </div>
          <div className="kv">
            <span>Antal</span>
            <span>{t?.count ?? 0}</span>
          </div>
        </Section>
      );
    }
  }
}
