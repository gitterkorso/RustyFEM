// Gränssnittet mot motorn. React-komponenterna pratar bara med `Engine`;
// implementationen är `TauriEngine` (se PLAN.md, grundprincip 2).

import type {
  McpStatus_Serialize,
  Edit_Deserialize,
  Edited_Serialize,
  MemoryEstimate,
  ModelInfo_Serialize,
  Opened_Serialize,
  PickInfo,
  RunEvent,
  SessionInfo_Serialize,
  StepResultInfo,
  UnitSymbols,
  UnitSystem,
} from "./bindings";

export type Edit = Edit_Deserialize;
export type Edited = Edited_Serialize;
export type ModelInfo = ModelInfo_Serialize;
export type Opened = Opened_Serialize;
export type SessionInfo = SessionInfo_Serialize;
export type McpStatus = McpStatus_Serialize;
export type { MemoryEstimate, PickInfo, RunEvent, StepResultInfo, UnitSymbols, UnitSystem };

/** Händelser från motorn som inte är svar på ett anrop (t.ex. ändringar via MCP). */
export interface EngineEvents {
  opened(o: Opened): void;
  changed(id: number, geometry: boolean): void;
  run(id: number, e: RunEvent): void;
  mcp(s: McpStatus): void;
}
export type {
  BlockInfo_Serialize as BlockInfo,
  FieldInfo,
  FrameInfo,
  KeywordInfo,
  MaterialInfo,
  SectionInfo_Serialize as SectionInfo,
  StepInfo_Serialize as StepInfo,
} from "./bindings";

/** Visningsgeometri från Rust (alla index pekar på modellens noder). */
export interface MeshData {
  nodes: number;
  positions: Float32Array;
  tris: Uint32Array;
  triElem: Uint32Array;
  edges: Uint32Array;
  lines: Uint32Array;
  lineElem: Uint32Array;
  points: Uint32Array;
  pointElem: Uint32Array;
}

/** Randvillkor (nod, slotmask) och laster (nod, F1–F3, M1–M3) i ett steg. */
export interface Symbols {
  bcNodes: Uint32Array;
  bcMasks: Uint32Array;
  loadNodes: Uint32Array;
  loads: Float32Array;
}

/** Nodvärden (NaN där fältet saknas) med min och max. */
export interface FieldData {
  values: Float32Array;
  min: number;
  max: number;
}

export interface Engine {
  newModel(script?: string): Promise<Opened>;
  open(path: string): Promise<Opened>;
  close(id: number): Promise<void>;
  save(id: number, path: string): Promise<SessionInfo>;
  sessionInfo(id: number): Promise<SessionInfo>;
  modelInfo(id: number): Promise<ModelInfo>;
  /** Byter enhetssystem (bara etiketter; inga värden räknas om). */
  setUnits(id: number, units: UnitSystem): Promise<SessionInfo>;
  unitSystems(): Promise<[UnitSystem, string][]>;
  edit(id: number, edit: Edit): Promise<Edited>;
  undo(id: number): Promise<Edited>;
  redo(id: number): Promise<Edited>;
  solvers(): Promise<string[]>;
  estimate(id: number, solver: string | null): Promise<MemoryEstimate>;
  solve(id: number, solver: string | null, allFrames: boolean, onEvent: (e: RunEvent) => void): Promise<void>;
  cancel(id: number): Promise<void>;
  results(id: number): Promise<StepResultInfo[]>;
  mesh(id: number): Promise<MeshData>;
  symbols(id: number, step: number | null): Promise<Symbols>;
  field(id: number, step: number, frame: number, field: string, component: string): Promise<FieldData>;
  displacement(id: number, step: number, frame: number): Promise<Float32Array>;
  history(id: number, step: number, index: number): Promise<[number, number][]>;
  pick(id: number, kind: "node" | "element", index: number, at: { step: number; frame: number } | null): Promise<PickInfo>;
  systemMemory(): Promise<number>;
  /** Fildialoger (plattformens egna). `null` om användaren avbryter. */
  openDialog(): Promise<string | null>;
  saveDialog(defaultName: string, kind: "project" | "script"): Promise<string | null>;
  /** Självtestläge (`rustyfem-app --selftest`): kör testförloppet och rapportera. */
  selftest(): Promise<boolean>;
  selftestDone(png: string, report: string): Promise<void>;
  /** Lokal MCP-server för AI-assistenter (Claude, Codex). */
  mcpStatus(): Promise<McpStatus>;
  mcpStart(port: number | null): Promise<McpStatus>;
  mcpStop(): Promise<McpStatus>;
  mcpInstallClaudeDesktop(): Promise<string[]>;
  listen(h: EngineEvents): Promise<() => void>;
}
