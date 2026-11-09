// Small building blocks: icons, buttons, fields and number formatting.

import { useEffect, useState, type ReactNode } from "react";

const paths: Record<string, string> = {
  new: "M6 3h8l4 4v14H6zM14 3v4h4M12 11v6M9 14h6",
  open: "M3 7h6l2 2h10v10H3zM3 7V5h6",
  save: "M5 3h11l3 3v15H5zM8 3v5h7V3M8 21v-7h8v7",
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3",
  redo: "M15 14l5-5-5-5M20 9H10a6 6 0 000 12h3",
  play: "M7 4l13 8-13 8z",
  stop: "M6 6h12v12H6z",
  fit: "M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5",
  cube: "M12 2l9 5v10l-9 5-9-5V7zM12 12l9-5M12 12v10M12 12L3 7",
  square: "M4 4h16v16H4z",
  edges: "M4 4h16v16H4zM4 12h16M12 4v16",
  profile: "M5 4h14v3h-5.5v10H19v3H5v-3h5.5V7H5z",
  surface: "M4 18l8-14 8 14z",
  ghost: "M4 4h16v16H4z",
  anchor: "M12 3v12M6 21l6-6 6 6M4 21h16",
  cursor: "M5 3l14 7-6 2-2 6z",
  node: "M12 12m-3 0a3 3 0 106 0 3 3 0 10-6 0M4 12h5M15 12h5",
  camera: "M4 8h3l2-3h6l2 3h3v11H4zM12 17a4 4 0 100-8 4 4 0 000 8z",
  sun: "M12 7a5 5 0 100 10 5 5 0 000-10zM12 1v3M12 20v3M1 12h3M20 12h3M4 4l2 2M18 18l2 2M4 20l2-2M18 6l2-2",
  moon: "M20 14A8 8 0 1110 4a7 7 0 0010 10z",
  sidebar: "M3 4h18v16H3zM9 4v16",
  sidebarR: "M3 4h18v16H3zM15 4v16",
  dock: "M3 4h18v16H3zM3 15h18",
  search: "M11 4a7 7 0 100 14 7 7 0 000-14zM21 21l-5-5",
  close: "M6 6l12 12M18 6L6 18",
  chev: "M9 6l6 6-6 6",
  chevDown: "M6 9l6 6 6-6",
  plus: "M12 5v14M5 12h14",
  film: "M4 4h16v16H4zM8 4v16M16 4v16M4 8h4M4 12h4M4 16h4M16 8h4M16 12h4M16 16h4",
  wave: "M2 12c3-6 5-6 8 0s5 6 8 0 3-4 4-4",
  chart: "M4 20V4M4 20h16M7 15l4-5 3 3 5-7",
  layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
  info: "M12 8h.01M11 12h1v5h1M12 21a9 9 0 100-18 9 9 0 000 18z",
  warn: "M12 3l10 18H2zM12 10v5M12 18h.01",
  check: "M5 12l5 5 9-10",
  code: "M8 7l-5 5 5 5M16 7l5 5-5 5",
  tool: "M14 6a4 4 0 00-5 5L3 17l4 4 6-6a4 4 0 005-5l-3 3-3-1-1-3z",
  export: "M12 3v12M7 8l5-5 5 5M4 15v6h16v-6",
  plug: "M9 2v5M15 2v5M6 7h12v4a6 6 0 01-12 0zM12 17v5",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
};

export function Icon({ name, size = 16 }: { name: string; size?: number }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name] ?? paths.info} />
    </svg>
  );
}

export function IconButton(p: { icon: string; title: string; onClick?: () => void; active?: boolean; disabled?: boolean; size?: number; label?: string; kbd?: string }) {
  return (
    <button
      type="button"
      className={`ibtn${p.active ? " active" : ""}${p.label ? " wide" : ""}`}
      title={p.kbd ? `${p.title} (${p.kbd})` : p.title}
      aria-label={p.title}
      aria-pressed={p.active}
      onClick={p.onClick}
      disabled={p.disabled}
    >
      <Icon name={p.icon} size={p.size} />
      {p.label && <span>{p.label}</span>}
    </button>
  );
}

/** Number with four significant digits, exponent when needed. */
export function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) return Number(v.toPrecision(digits)).toString();
  return v.toExponential(digits - 1).replace("e+", "e");
}

export function parseNum(s: string): number | null {
  const t = s.trim().replace(/[dD]/, "e");
  if (!t) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Number field that commits the value on Enter or when focus leaves the field. */
export function NumField(p: { value: number | null; onCommit: (v: number) => void; label?: string; unit?: string; disabled?: boolean; min?: number }) {
  const [text, setText] = useState(fmt(p.value, 6));
  const [bad, setBad] = useState(false);
  useEffect(() => setText(fmt(p.value, 6)), [p.value]);
  const commit = () => {
    const v = parseNum(text);
    if (v === null || (p.min !== undefined && v < p.min)) {
      setBad(true);
      return;
    }
    setBad(false);
    if (v !== p.value) p.onCommit(v);
  };
  return (
    <label className="field">
      {p.label && <span className="flabel">{p.label}</span>}
      <span className={`finput${bad ? " bad" : ""}`}>
        <input
          inputMode="decimal"
          value={text}
          disabled={p.disabled}
          onChange={(e) => setText(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            if (e.key === "Enter") (e.target as HTMLInputElement).blur();
            if (e.key === "Escape") {
              setText(fmt(p.value, 6));
              setBad(false);
            }
          }}
        />
        {p.unit && <span className="unit">{p.unit}</span>}
      </span>
    </label>
  );
}

export function Select<T extends string | number>(p: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label?: string; disabled?: boolean }) {
  return (
    <label className="field">
      {p.label && <span className="flabel">{p.label}</span>}
      <span className="finput select">
        <select
          value={String(p.value)}
          disabled={p.disabled}
          onChange={(e) => {
            const o = p.options.find((x) => String(x.value) === e.target.value);
            if (o) p.onChange(o.value);
          }}
        >
          {p.options.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>
              {o.label}
            </option>
          ))}
        </select>
        <Icon name="chevDown" size={14} />
      </span>
    </label>
  );
}

export function Toggle(p: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="toggle" title={p.hint}>
      <input type="checkbox" checked={p.checked} onChange={(e) => p.onChange(e.target.checked)} />
      <span className="track">
        <span className="thumb" />
      </span>
      <span>{p.label}</span>
    </label>
  );
}

export function Segmented<T extends string>(p: { value: T; options: { value: T; label: string; icon?: string }[]; onChange: (v: T) => void; small?: boolean }) {
  return (
    <div className={`seg${p.small ? " small" : ""}`} role="tablist">
      {p.options.map((o) => (
        <button key={o.value} type="button" role="tab" aria-selected={p.value === o.value} className={p.value === o.value ? "on" : ""} onClick={() => p.onChange(o.value)}>
          {o.icon && <Icon name={o.icon} size={14} />}
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Section(p: { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="sect">
      <header>
        <h3>{p.title}</h3>
        {p.right}
      </header>
      <div className="sect-body">{p.children}</div>
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
