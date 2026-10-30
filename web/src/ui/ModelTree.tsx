// The model tree: instances, sets, surfaces, materials, sections, steps and loads.

import { useState, type ReactNode } from "react";
import { persist, set, useStore, type TreeSel } from "../store";
import { Icon } from "./ui";

function same(a: TreeSel | null, b: TreeSel) {
  return JSON.stringify(a) === JSON.stringify(b);
}

const openKey = "tree";
let openState: Record<string, boolean> = (() => {
  try {
    return JSON.parse(localStorage.getItem(`rustyfem.${openKey}`) ?? "{}");
  } catch {
    return {};
  }
})();

function Group(p: { id: string; title: string; count?: number; children?: ReactNode; defaultOpen?: boolean; empty?: boolean }) {
  const [open, setOpen] = useState(openState[p.id] ?? p.defaultOpen ?? false);
  const toggle = () => {
    setOpen(!open);
    openState = { ...openState, [p.id]: !open };
    persist(openKey, openState);
  };
  if (p.empty) return null;
  return (
    <div className="tg">
      <button type="button" className="tg-head" onClick={toggle} aria-expanded={open}>
        <span className={`chev${open ? " open" : ""}`}>
          <Icon name="chev" size={12} />
        </span>
        <span className="tg-title">{p.title}</span>
        {p.count !== undefined && <span className="count">{p.count}</span>}
      </button>
      {open && <div className="tg-body">{p.children}</div>}
    </div>
  );
}

function Item(p: { sel: TreeSel; label: string; sub?: string; badge?: string; onClick?: () => void }) {
  const cur = useStore((s) => s.sel);
  return (
    <button
      type="button"
      className={`ti${same(cur, p.sel) ? " on" : ""}`}
      onClick={() => {
        set({ sel: p.sel, right: true });
        p.onClick?.();
      }}
    >
      <span className="ti-label">{p.label}</span>
      {p.sub && <span className="ti-sub">{p.sub}</span>}
      {p.badge && <span className="badge">{p.badge}</span>}
    </button>
  );
}

export function ModelTree() {
  const model = useStore((s) => s.model);
  const session = useStore((s) => s.session);
  const [q, setQ] = useState("");
  if (!model || !session) return null;
  const f = (name: string) => !q || name.toLowerCase().includes(q.toLowerCase());
  const nsets = model.nsets.filter((x) => f(x.name));
  const elsets = model.elsets.filter((x) => f(x.name));
  const surfaces = model.surfaces.filter((x) => f(x.name));
  return (
    <div className="tree">
      <div className="tree-search">
        <Icon name="search" size={14} />
        <input placeholder="Filter" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter the model tree" />
      </div>
      <div className="tree-scroll">
        <Item sel={{ type: "model" }} label={session.name} sub={`${session.script_dim ?? model.dim}D · ${model.nodes} nodes · ${model.elements} elements`} />
        {session.kind === "rusty" && <Item sel={{ type: "script" }} label="Build model" sub="materials, geometry, boundary conditions, loads" />}
        <Group id="etypes" title="Element types" count={model.element_types.length} empty={!model.element_types.length}>
          {model.element_types.map((t) => (
            <Item key={t.name} sel={{ type: "etype", name: t.name }} label={t.name} badge={String(t.count)} />
          ))}
        </Group>
        {model.instances.length > 1 && (
          <Group id="inst" title="Instances" count={model.instances.length}>
            {model.instances.map((n) => (
              <div key={n} className="ti static">
                <span className="ti-label">{n}</span>
              </div>
            ))}
          </Group>
        )}
        <Group id="nsets" title="Node sets" count={nsets.length} empty={!model.nsets.length}>
          {nsets.map((s) => (
            <Item key={s.name} sel={{ type: "set", kind: "nset", name: s.name }} label={s.name} badge={String(s.count)} />
          ))}
        </Group>
        <Group id="elsets" title="Element sets" count={elsets.length} empty={!model.elsets.length}>
          {elsets.map((s) => (
            <Item key={s.name} sel={{ type: "set", kind: "elset", name: s.name }} label={s.name} badge={String(s.count)} />
          ))}
        </Group>
        <Group id="surfaces" title="Surfaces" count={surfaces.length} empty={!model.surfaces.length}>
          {surfaces.map((s) => (
            <Item key={s.name} sel={{ type: "set", kind: "surface", name: s.name }} label={s.name} badge={String(s.count)} />
          ))}
        </Group>
        <Group id="materials" title="Materials" count={model.materials.length} defaultOpen>
          {model.materials.map((m) => (
            <Item key={m.index} sel={{ type: "material", index: m.index }} label={m.name} sub={m.behaviors.map((b) => b.name.toLowerCase()).join(", ")} />
          ))}
        </Group>
        <Group id="sections" title="Sections" count={model.sections.length} defaultOpen>
          {model.sections.map((s) => (
            <Item key={s.index} sel={{ type: "section", index: s.index }} label={s.elset} sub={`${s.kind.toLowerCase()} · ${s.material || "–"}`} />
          ))}
        </Group>
        {model.initial_bcs.length > 0 && <Item sel={{ type: "initial" }} label="Boundary conditions (model data)" badge={String(model.initial_bcs.reduce((a, b) => a + b.total, 0))} />}
        <Group id="steps" title="Steps" count={model.steps.length} defaultOpen>
          {model.steps.map((s) => (
            <Item
              key={s.index}
              sel={{ type: "step", index: s.index }}
              label={s.name}
              sub={`*${s.procedure}${s.nlgeom ? " · NLGEOM" : ""}${s.perturbation ? " · perturbation" : ""}`}
              onClick={() => set({ symbolStep: s.index })}
            />
          ))}
        </Group>
        {model.other.length > 0 && (
          <Group id="other" title="Other keywords" count={model.other.length}>
            {model.other.map((n, i) => (
              <div key={i} className="ti static">
                <span className="ti-label">*{n}</span>
              </div>
            ))}
          </Group>
        )}
      </div>
    </div>
  );
}
