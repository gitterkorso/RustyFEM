// The .rusty script in the UI: the lines as commands (for the command list and
// the tool forms) and node selections in the preview. The engine's parser in
// `src/script.rs` is authoritative; this is only a reading for display.

export interface ScriptCmd {
  /** Line number (1-based). */
  line: number;
  cmd: string;
  /** Words without `=` (lowercase), e.g. `steel`, `ux`, `all`. */
  words: string[];
  /** Points `(x,y[,z])` in the order they appear. */
  points: string[];
  /** `key=value` with the key in lowercase. */
  kv: Record<string, string>;
}

/** Removes comments and whitespace inside parentheses (like the engine). */
function normalize(raw: string) {
  const s = raw.split("#")[0];
  let out = "";
  let depth = 0;
  for (const c of s) {
    if (c === "(") depth++;
    else if (c === ")") depth--;
    if (depth > 0 && /\s/.test(c)) continue;
    out += c;
  }
  return out;
}

export function parseScript(text: string): ScriptCmd[] {
  const out: ScriptCmd[] = [];
  text.split("\n").forEach((raw, i) => {
    const toks = normalize(raw).split(/\s+/).filter(Boolean);
    if (!toks.length) return;
    const c: ScriptCmd = { line: i + 1, cmd: toks[0].toLowerCase(), words: [], points: [], kv: {} };
    for (const t of toks.slice(1)) {
      const k = t.indexOf("=");
      if (k > 0) c.kv[t.slice(0, k).toLowerCase()] = t.slice(k + 1);
      else if (t.startsWith("(")) c.points.push(t);
      else c.words.push(t.toLowerCase());
    }
    out.push(c);
  });
  return out;
}

/** Replaces (`text`), appends (`line` = null) or removes (`text` = null) a line. */
export function setLine(script: string, line: number | null, text: string | null): string {
  const lines = script.replace(/\n$/, "").split("\n");
  if (script === "") lines.length = 0;
  if (line === null) {
    if (text !== null) lines.push(text);
  } else if (text === null) lines.splice(line - 1, 1);
  else lines[line - 1] = text;
  return lines.join("\n") + "\n";
}

export type Category = "model" | "material" | "geometry" | "support" | "load";

export const categories: { id: Category; title: string }[] = [
  { id: "model", title: "Settings" },
  { id: "material", title: "Material" },
  { id: "geometry", title: "Geometry" },
  { id: "support", title: "Boundary conditions" },
  { id: "load", title: "Loads" },
];

export function category(cmd: string): Category | null {
  switch (cmd) {
    case "model":
    case "plane":
    case "elements":
      return "model";
    case "material":
      return "material";
    case "plate":
    case "quad":
    case "box":
    case "beam":
    case "truss":
      return "geometry";
    case "fix":
    case "pin":
    case "support":
      return "support";
    case "load":
    case "pressure":
    case "gravity":
      return "load";
    default:
      return null;
  }
}

/** The selection in plain text, e.g. "x = 0" or "point (2,0)". */
export function describeSel(c: ScriptCmd): string {
  if (c.words.includes("all")) return "all nodes";
  if (c.kv.at) return `point ${c.kv.at}`;
  const parts = ["x", "y", "z"].filter((a) => c.kv[a] !== undefined).map((a) => `${a} = ${c.kv[a].replace("..", " … ")}`);
  return parts.join(", ") || "no selection";
}

const geomNames: Record<string, string> = { plate: "Plate", quad: "Quadrilateral", box: "Block", beam: "Beam", truss: "Truss" };

/** Short title and detail for the command list. */
export function summary(c: ScriptCmd): { title: string; sub: string } {
  const mat = c.kv.material ? ` · ${c.kv.material}` : "";
  switch (c.cmd) {
    case "model":
      return { title: `Model ${c.words[0]?.toUpperCase() ?? ""}`, sub: "" };
    case "plane":
      return { title: c.words[0] === "strain" ? "Plane strain" : "Plane stress", sub: "" };
    case "elements":
      return { title: c.words[0] === "tri" ? "Triangular elements" : "Quadrilateral elements", sub: "" };
    case "material": {
      const v = [c.kv.e && `E = ${c.kv.e}`, c.kv.nu && `ν = ${c.kv.nu}`, c.kv.rho && `ρ = ${c.kv.rho}`].filter(Boolean).join(", ");
      return { title: c.words[0] ?? "material", sub: v || "default values" };
    }
    case "plate":
      return { title: geomNames.plate, sub: `x ${c.kv.x ?? "?"}, y ${c.kv.y ?? "?"}, t ${c.kv.t ?? "?"}, h ${c.kv.size ?? "?"}${mat}` };
    case "box":
      return { title: geomNames.box, sub: `x ${c.kv.x ?? "?"}, y ${c.kv.y ?? "?"}, z ${c.kv.z ?? "?"}, h ${c.kv.size ?? "?"}${mat}` };
    case "quad":
      return { title: geomNames.quad, sub: `${c.points.join(" ")}${mat}` };
    case "beam":
      return { title: geomNames.beam, sub: `${c.points.length} points · ${c.kv.section ?? "?"}${mat}` };
    case "truss":
      return { title: geomNames.truss, sub: `${c.points.length} points · A ${c.kv.area ?? "?"}${mat}` };
    case "fix":
      return { title: "Fixed", sub: describeSel(c) };
    case "pin":
      return { title: "Pinned", sub: describeSel(c) };
    case "support": {
      const d = ["ux", "uy", "uz", "rx", "ry", "rz"].filter((k) => c.words.includes(k) || c.kv[k] !== undefined).map((k) => (c.kv[k] !== undefined ? `${k}=${c.kv[k]}` : k));
      return { title: `Support ${d.join(" ")}`, sub: describeSel(c) };
    }
    case "load": {
      const f = ["fx", "fy", "fz", "mx", "my", "mz"].filter((k) => c.kv[k] !== undefined).map((k) => `${k.toUpperCase()} = ${c.kv[k]}`);
      return { title: `Load ${f.join(", ")}`, sub: describeSel(c) };
    }
    case "pressure":
      return { title: `Pressure p = ${c.kv.p ?? "?"}`, sub: describeSel(c) };
    case "gravity": {
      const g = c.points[0] ?? (["gx", "gy", "gz"].some((k) => c.kv[k]) ? `(${["gx", "gy", "gz"].map((k) => c.kv[k] ?? "0").join(",")})` : `g = ${c.kv.g ?? "9.81"}`);
      return { title: "Gravity", sub: g };
    }
    default:
      return { title: c.cmd, sub: "" };
  }
}

/** The script's materials in order (names as written). */
export function materialNames(cmds: ScriptCmd[]): string[] {
  return cmds.filter((c) => c.cmd === "material" && c.words[0]).map((c) => c.words[0]);
}

export interface SelSpec {
  all?: boolean;
  at?: number[];
  box?: [axis: number, lo: number, hi: number][];
}

function num(s: string): number | null {
  const v = Number(s.trim());
  return s.trim() !== "" && Number.isFinite(v) ? v : null;
}

/** "0", "0..1" or "1..0" → [lo, hi]. */
export function parseRange(s: string): [number, number] | null {
  const [a, b] = s.includes("..") ? s.split("..") : [s, s];
  const x = num(a ?? ""), y = num(b ?? "");
  return x === null || y === null ? null : [Math.min(x, y), Math.max(x, y)];
}

/** "(1,2)" or "1,2" → coordinates. */
export function parsePoint(s: string): number[] | null {
  const v = s.replace(/^\(|\)$/g, "").split(",").map(num);
  return v.length >= 2 && v.length <= 3 && v.every((x) => x !== null) ? (v as number[]) : null;
}

/**
 * The nodes hit by the selection, with the same tolerance as the engine
 * (one millionth of the model's diagonal). `null` if the selection cannot be read.
 */
export function selectNodes(pos: Float32Array, n: number, sel: SelSpec): number[] | null {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
    lo[a] = Math.min(lo[a], pos[3 * i + a]);
    hi[a] = Math.max(hi[a], pos[3 * i + a]);
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  // slightly more generous than the engine: the positions are f32
  const tol = Math.max(diag * 1e-6, 1e-6) * 4;
  const out: number[] = [];
  if (sel.all) {
    for (let i = 0; i < n; i++) out.push(i);
  } else if (sel.at) {
    const p = [sel.at[0], sel.at[1], sel.at[2] ?? 0];
    for (let i = 0; i < n; i++) if (Math.hypot(pos[3 * i] - p[0], pos[3 * i + 1] - p[1], pos[3 * i + 2] - p[2]) <= tol) out.push(i);
  } else if (sel.box?.length) {
    for (let i = 0; i < n; i++) if (sel.box.every(([a, l, h]) => pos[3 * i + a] >= l - tol && pos[3 * i + a] <= h + tol)) out.push(i);
  } else return null;
  return out;
}

/** The number, short but exact enough for the script. */
export function short(v: number): string {
  return String(Number(v.toPrecision(10)));
}
