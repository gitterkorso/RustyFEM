// The interface to the engine. The React components only talk to `Engine`;
// the implementation is `TauriEngine` (see PLAN.md, core principle 2).

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

/** Events from the engine that are not replies to a call (e.g. edits via MCP). */
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

/** Display geometry from Rust (all indices point to the model's nodes). */
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
  /** Cross-sections and shell thicknesses for drawing in 3D (null if not available). */
  profiles: ProfileData | null;
}

/** Beam profiles and shell thicknesses (see `app/session/src/profiles.rs`). */
export interface ProfileData {
  /** Profile per line segment (−1 = none). */
  segProfile: Int32Array;
  /** The section's 1-axis per line segment (×3). */
  segN1: Float32Array;
  /** Per profile: convex polygons in the section plane (x1, x2 pairs). */
  profiles: Float32Array[][];
  /** Per element: thickness, midsurface position relative to the nodes, normal (×5). Thickness 0 = not a shell. */
  shell: Float32Array;
  /** The model has 3D elements: show it in 3D even if it lies in a plane. */
  threeD: boolean;
}

/** Boundary conditions (node, slot mask) and loads (node, F1–F3, M1–M3) in a step. */
export interface Symbols {
  bcNodes: Uint32Array;
  bcMasks: Uint32Array;
  loadNodes: Uint32Array;
  loads: Float32Array;
}

/** Nodal values (NaN where the field is missing) with min and max. */
export interface FieldData {
  values: Float32Array;
  min: number;
  max: number;
}

// ---- Visualization tab (see `app/session/src/visual.rs`) ----

export type Vec3 = [number, number, number];

export interface VisPart {
  name: string;
  kind: "beam" | "truss" | "shell" | "membrane" | "solid" | "plane" | "axisymmetric" | "other";
  shape: string | null;
  dims: number[];
  area: number | null;
  thickness: number | null;
  material: string;
  mass: number;
  elements: number;
  profile: number;
}
export interface VisProfile {
  outer: [number, number][];
  hole: [number, number][] | null;
  pieces: [number, number][][];
  smooth: boolean;
}
export interface VisMember {
  part: number;
  nodes: number[];
  s: number[];
  n1: Vec3[];
  n2: Vec3[];
  elements: number[];
  dirs: number[];
  length: number;
}
export interface VisBody {
  dim: number;
  elements: number;
  center: Vec3;
  axis: Vec3;
  s0: number;
  s1: number;
  lo: Vec3;
  hi: Vec3;
}
export interface VisModel {
  parts: VisPart[];
  profiles: VisProfile[];
  members: VisMember[];
  bodies: VisBody[];
  elemPart: number[];
  elemMember: number[];
  elemBody: number[];
}
export interface VisMemberForces {
  s: number[];
  /** Global F (3) and M (3) on the positive section. */
  f: number[][];
  derivedShear: boolean;
}
export interface VisSupport {
  nodes: number[];
  mask: number;
  center: Vec3;
  f: Vec3;
  m: Vec3;
}
export interface VisFrame {
  members: VisMemberForces[];
  supports: VisSupport[];
  rot: number[];
}
export interface VisShell {
  sf: number[] | null;
  sm: number[] | null;
  dir1: Vec3;
  dir2: Vec3;
}
export interface VisCut {
  s: number;
  area: number;
  centroid: Vec3;
  f: Vec3;
  m: Vec3;
  tris: number[];
  segs: number[];
  method: string;
}
export interface VisDiagram {
  axis: Vec3;
  s0: number;
  s1: number;
  s: number[];
  f: number[][];
  area: number[];
}

export interface Engine {
  visualModel(id: number): Promise<VisModel>;
  visualFrame(id: number, step: number, frame: number): Promise<VisFrame>;
  visualShell(id: number, step: number, frame: number, element: number): Promise<VisShell>;
  visualCut(id: number, step: number, frame: number, body: number, axis: Vec3, s: number): Promise<VisCut>;
  visualDiagram(id: number, step: number, frame: number, body: number, axis: Vec3, n: number): Promise<VisDiagram>;
  /** Asks for a file name and writes a PNG (data URL); false if the user cancelled. */
  savePng(defaultName: string, dataUrl: string): Promise<boolean>;
  newModel(script?: string): Promise<Opened>;
  open(path: string): Promise<Opened>;
  close(id: number): Promise<void>;
  save(id: number, path: string): Promise<SessionInfo>;
  sessionInfo(id: number): Promise<SessionInfo>;
  modelInfo(id: number): Promise<ModelInfo>;
  /** Changes the unit system (labels only; no values are converted). */
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
  /** File dialogs (the platform's own). `null` if the user cancels. */
  openDialog(): Promise<string | null>;
  saveDialog(defaultName: string, kind: "project" | "script"): Promise<string | null>;
  /** Self-test mode (`rustyfem-app --selftest`): run the test sequence and report. */
  selftest(): Promise<boolean>;
  selftestDone(png: string, report: string): Promise<void>;
  /** Local MCP server for AI assistants (Claude, Codex). */
  mcpStatus(): Promise<McpStatus>;
  mcpStart(port: number | null, network: boolean | null): Promise<McpStatus>;
  mcpStop(): Promise<McpStatus>;
  mcpInstallClaudeDesktop(): Promise<string[]>;
  listen(h: EngineEvents): Promise<() => void>;
}
