// UI-läge (markering, öppna paneler, val av resultat). Modellen och
// resultaten finns i Rust; här ligger bara en visningskopia.

import { useRef, useSyncExternalStore } from "react";
import type { FieldInfo, McpStatus, MemoryEstimate, MeshData, ModelInfo, PickInfo, SessionInfo, StepResultInfo, UnitSystem } from "./engine/Engine";
import type { ColormapName } from "./viewer/colormaps";

export type Mode = "model" | "results";

/** Markerat objekt i modellträdet. */
export type TreeSel =
  | { type: "model" }
  | { type: "script" }
  | { type: "material"; index: number }
  | { type: "section"; index: number }
  | { type: "step"; index: number }
  | { type: "initial" }
  | { type: "set"; kind: "nset" | "elset" | "surface"; name: string }
  | { type: "etype"; name: string };

export interface LogLine {
  t: number;
  kind: "info" | "warn" | "error" | "ok";
  text: string;
}

export interface RunState {
  running: boolean;
  /** 0–1 över hela analysen. */
  progress: number;
  stepIndex: number;
  steps: number;
  stepName: string;
  increment: number;
  residuals: number[];
  estimate: MemoryEstimate | null;
  solver: string | null;
  allFrames: boolean;
}

export interface ResultSel {
  step: number;
  frame: number;
  field: string;
  component: string;
}

export interface State {
  id: number | null;
  session: SessionInfo | null;
  model: ModelInfo | null;
  mesh: MeshData | null;
  meshVersion: number;
  mode: Mode;
  sel: TreeSel | null;
  symbolStep: number | null;
  results: StepResultInfo[];
  rsel: ResultSel | null;
  deformAuto: boolean;
  deformFactor: number;
  animate: "off" | "frames" | "mode";
  pick: PickInfo | null;
  pickMode: "element" | "node";
  /** Kraftenhet för lastvärden i modellfönstret (null = visa inga värden). */
  loadUnit: string | null;
  /** Noder som urvalet i verktygspanelen träffar (visas i modellfönstret). */
  marked: number[] | null;
  /** Väntar på att användaren väljer en nod i modellfönstret (t.ex. för at=). */
  onNodePick: ((node: number) => void) | null;
  /** Skriptrad (1-baserad) som verktygspanelen ska öppna för ändring. */
  scriptEdit: number | null;
  run: RunState;
  log: LogLine[];
  dock: boolean;
  left: boolean;
  right: boolean;
  dark: boolean;
  surfaces: boolean;
  edges: boolean;
  undeformed: boolean;
  symbols: boolean;
  levels: number;
  colormap: ColormapName;
  palette: boolean;
  busy: string | null;
  toast: { kind: "error" | "info"; text: string } | null;
  recent: string[];
  memory: number;
  solvers: string[];
  unitSystems: [UnitSystem, string][];
  mcp: McpStatus | null;
  mcpPanel: boolean;
}

function load<T>(k: string, d: T): T {
  try {
    const v = localStorage.getItem(`rustyfem.${k}`);
    return v === null ? d : (JSON.parse(v) as T);
  } catch {
    return d;
  }
}

export function persist(k: string, v: unknown) {
  try {
    localStorage.setItem(`rustyfem.${k}`, JSON.stringify(v));
  } catch {
    /* ingen lagring (privat läge) */
  }
}

const narrow = typeof window !== "undefined" && window.innerWidth < 900;

let state: State = {
  id: null,
  session: null,
  model: null,
  mesh: null,
  meshVersion: 0,
  mode: "model",
  sel: null,
  symbolStep: null,
  results: [],
  rsel: null,
  deformAuto: true,
  deformFactor: 1,
  animate: "off",
  pick: null,
  pickMode: "element",
  loadUnit: load<string | null>("loadUnit", "kN"),
  marked: null,
  onNodePick: null,
  scriptEdit: null,
  run: { running: false, progress: 0, stepIndex: 0, steps: 0, stepName: "", increment: 0, residuals: [], estimate: null, solver: null, allFrames: true },
  log: [],
  dock: load("dock", true),
  left: !narrow,
  right: !narrow,
  dark: load("dark", window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true),
  surfaces: true,
  edges: true,
  undeformed: false,
  symbols: true,
  levels: load("levels", 12),
  colormap: load<ColormapName>("colormap", "regnbåge"),
  palette: false,
  busy: null,
  toast: null,
  recent: load<string[]>("recent", []),
  memory: 0,
  solvers: [],
  unitSystems: [],
  mcp: null,
  mcpPanel: false,
};

const subs = new Set<() => void>();

export function get(): State {
  return state;
}

export function set(p: Partial<State> | ((s: State) => Partial<State>)) {
  const patch = typeof p === "function" ? p(state) : p;
  state = { ...state, ...patch };
  for (const f of subs) f();
}

export function subscribe(f: () => void) {
  subs.add(f);
  return () => subs.delete(f);
}

export function useStore<T>(sel: (s: State) => T): T {
  return useSyncExternalStore(subscribe, () => sel(state));
}

/** Som `useStore`, för väljare som bygger ett nytt objekt: jämför fälten ytligt. */
export function useShallow<T extends object>(sel: (s: State) => T): T {
  const ref = useRef<T | null>(null);
  return useSyncExternalStore(subscribe, () => {
    const n = sel(state);
    const o = ref.current;
    if (o && Object.keys(n).every((k) => (n as Record<string, unknown>)[k] === (o as Record<string, unknown>)[k])) return o;
    ref.current = n;
    return n;
  });
}

export function log(kind: LogLine["kind"], text: string) {
  set((s) => ({ log: [...s.log.slice(-400), { t: Date.now(), kind, text }] }));
}

/** Fältets info i det valda steget. */
export function currentField(s: State): FieldInfo | null {
  if (!s.rsel) return null;
  return s.results[s.rsel.step]?.fields.find((f) => f.name === s.rsel!.field) ?? null;
}
