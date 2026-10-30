// .rusty-skriptet i gränssnittet: raderna som kommandon (för kommandolistan och
// verktygsformulären) och nodurval i förhandsvisningen. Motorns tolk i
// `src/script.rs` är den som gäller; det här är bara en läsning för visning.

export interface ScriptCmd {
  /** Radnummer (1-baserat). */
  line: number;
  cmd: string;
  /** Ord utan `=` (gemener), t.ex. `steel`, `ux`, `all`. */
  words: string[];
  /** Punkter `(x,y[,z])` i den ordning de står. */
  points: string[];
  /** `nyckel=värde` med nyckeln i gemener. */
  kv: Record<string, string>;
}

/** Tar bort kommentar och blanksteg inne i parenteser (som motorn). */
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

/** Ersätter (`text`), lägger till (`line` = null) eller tar bort (`text` = null) en rad. */
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
  { id: "model", title: "Inställningar" },
  { id: "material", title: "Material" },
  { id: "geometry", title: "Geometri" },
  { id: "support", title: "Randvillkor" },
  { id: "load", title: "Laster" },
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

/** Urvalet i klartext, t.ex. "x = 0" eller "punkt (2,0)". */
export function describeSel(c: ScriptCmd): string {
  if (c.words.includes("all")) return "alla noder";
  if (c.kv.at) return `punkt ${c.kv.at}`;
  const parts = ["x", "y", "z"].filter((a) => c.kv[a] !== undefined).map((a) => `${a} = ${c.kv[a].replace("..", " … ")}`);
  return parts.join(", ") || "inget urval";
}

const geomNames: Record<string, string> = { plate: "Platta", quad: "Fyrhörning", box: "Kloss", beam: "Balk", truss: "Stång" };

/** Kort rubrik och detalj för kommandolistan. */
export function summary(c: ScriptCmd): { title: string; sub: string } {
  const mat = c.kv.material ? ` · ${c.kv.material}` : "";
  switch (c.cmd) {
    case "model":
      return { title: `Modell ${c.words[0]?.toUpperCase() ?? ""}`, sub: "" };
    case "plane":
      return { title: c.words[0] === "strain" ? "Plan töjning" : "Plan spänning", sub: "" };
    case "elements":
      return { title: c.words[0] === "tri" ? "Triangelelement" : "Fyrhörningselement", sub: "" };
    case "material": {
      const v = [c.kv.e && `E = ${c.kv.e}`, c.kv.nu && `ν = ${c.kv.nu}`, c.kv.rho && `ρ = ${c.kv.rho}`].filter(Boolean).join(", ");
      return { title: c.words[0] ?? "material", sub: v || "standardvärden" };
    }
    case "plate":
      return { title: geomNames.plate, sub: `x ${c.kv.x ?? "?"}, y ${c.kv.y ?? "?"}, t ${c.kv.t ?? "?"}, h ${c.kv.size ?? "?"}${mat}` };
    case "box":
      return { title: geomNames.box, sub: `x ${c.kv.x ?? "?"}, y ${c.kv.y ?? "?"}, z ${c.kv.z ?? "?"}, h ${c.kv.size ?? "?"}${mat}` };
    case "quad":
      return { title: geomNames.quad, sub: `${c.points.join(" ")}${mat}` };
    case "beam":
      return { title: geomNames.beam, sub: `${c.points.length} punkter · ${c.kv.section ?? "?"}${mat}` };
    case "truss":
      return { title: geomNames.truss, sub: `${c.points.length} punkter · A ${c.kv.area ?? "?"}${mat}` };
    case "fix":
      return { title: "Inspänd", sub: describeSel(c) };
    case "pin":
      return { title: "Led", sub: describeSel(c) };
    case "support": {
      const d = ["ux", "uy", "uz", "rx", "ry", "rz"].filter((k) => c.words.includes(k) || c.kv[k] !== undefined).map((k) => (c.kv[k] !== undefined ? `${k}=${c.kv[k]}` : k));
      return { title: `Stöd ${d.join(" ")}`, sub: describeSel(c) };
    }
    case "load": {
      const f = ["fx", "fy", "fz", "mx", "my", "mz"].filter((k) => c.kv[k] !== undefined).map((k) => `${k.toUpperCase()} = ${c.kv[k]}`);
      return { title: `Last ${f.join(", ")}`, sub: describeSel(c) };
    }
    case "pressure":
      return { title: `Tryck p = ${c.kv.p ?? "?"}`, sub: describeSel(c) };
    case "gravity": {
      const g = c.points[0] ?? (["gx", "gy", "gz"].some((k) => c.kv[k]) ? `(${["gx", "gy", "gz"].map((k) => c.kv[k] ?? "0").join(",")})` : `g = ${c.kv.g ?? "9.81"}`);
      return { title: "Tyngd", sub: g };
    }
    default:
      return { title: c.cmd, sub: "" };
  }
}

/** Skriptets material i ordning (namn som de skrivs). */
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

/** "0", "0..1" eller "1..0" → [lo, hi]. */
export function parseRange(s: string): [number, number] | null {
  const [a, b] = s.includes("..") ? s.split("..") : [s, s];
  const x = num(a ?? ""), y = num(b ?? "");
  return x === null || y === null ? null : [Math.min(x, y), Math.max(x, y)];
}

/** "(1,2)" eller "1,2" → koordinater. */
export function parsePoint(s: string): number[] | null {
  const v = s.replace(/^\(|\)$/g, "").split(",").map(num);
  return v.length >= 2 && v.length <= 3 && v.every((x) => x !== null) ? (v as number[]) : null;
}

/**
 * Noderna som urvalet träffar, med samma tolerans som motorn
 * (en miljondel av modellens diagonal). `null` om urvalet inte går att läsa.
 */
export function selectNodes(pos: Float32Array, n: number, sel: SelSpec): number[] | null {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
    lo[a] = Math.min(lo[a], pos[3 * i + a]);
    hi[a] = Math.max(hi[a], pos[3 * i + a]);
  }
  const diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  // något generösare än motorn: positionerna är f32
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

/** Talet kort men exakt nog för skriptet. */
export function short(v: number): string {
  return String(Number(v.toPrecision(10)));
}
