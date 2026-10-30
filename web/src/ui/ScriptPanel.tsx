// .rusty-skriptet: modellens innehåll som en lista av kommandon, verktyg som
// skriver eller ändrar rader och textredigering. Varje tillämpning är ett
// kommando i motorn och kan ångras.

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { solve, toast, tryEdit } from "../actions";
import { categories, category, materialNames, parsePoint, parseRange, parseScript, selectNodes, setLine, short, summary, type ScriptCmd, type SelSpec } from "../script";
import { get, set, useStore } from "../store";
import { Icon, IconButton, Section, Segmented, Select } from "./ui";
import { convert, modelUnit, preferredUnit, rememberUnit, unitOptions, type ConvQ } from "./units";

type Tool = "material" | "plate" | "quad" | "box" | "beam" | "truss" | "support" | "load" | "pressure" | "gravity";
type Fields = Record<string, string>;

const tools: { id: Tool; label: string; dims: (2 | 3)[] }[] = [
  { id: "material", label: "Material", dims: [2, 3] },
  { id: "plate", label: "Platta", dims: [2] },
  { id: "quad", label: "Fyrhörning", dims: [2] },
  { id: "box", label: "Kloss", dims: [3] },
  { id: "beam", label: "Balk", dims: [2, 3] },
  { id: "truss", label: "Stång", dims: [2, 3] },
  { id: "support", label: "Stöd", dims: [2, 3] },
  { id: "load", label: "Last", dims: [2, 3] },
  { id: "pressure", label: "Tryck", dims: [2, 3] },
  { id: "gravity", label: "Tyngd", dims: [2, 3] },
];

const dofs = ["ux", "uy", "uz", "rx", "ry", "rz"];
const forces = ["fx", "fy", "fz", "mx", "my", "mz"];
const shapes: Record<string, { label: string; dims: string[] }> = {
  rect: { label: "Rektangel", dims: ["b", "h"] },
  circle: { label: "Massiv cirkel", dims: ["r"] },
  pipe: { label: "Rör", dims: ["r", "t"] },
  ibeam: { label: "I-balk", dims: ["h", "b", "tf", "tw"] },
  general: { label: "Allmän (A, I11, I22, J)", dims: ["A", "I11", "I22", "J"] },
};
const shapeDefaults: Record<string, string> = { b: "0.05", h: "0.1", r: "0.02", t: "0.003", tf: "0.01", tw: "0.006", A: "1e-3", I11: "1e-6", I22: "1e-6", J: "2e-6" };

/** Formulärfälten för en befintlig rad. */
function formFromCmd(c: ScriptCmd): { tool: Tool; f: Fields } | null {
  const f: Fields = {};
  const sel = () => {
    if (c.words.includes("all")) f.all = "1";
    if (c.kv.at) f.at = c.kv.at;
    for (const a of ["x", "y", "z"]) if (c.kv[a] !== undefined) f[`s${a}`] = c.kv[a];
  };
  const geom = (keys: string[]) => {
    for (const k of keys) if (c.kv[k] !== undefined) f[k] = c.kv[k];
    if (c.kv.material) f.mat = c.kv.material;
    if (c.points.length) f.pts = c.points.join(" ");
  };
  switch (c.cmd) {
    case "material":
      f.name = c.words[0] ?? "";
      if (c.kv.e) f.E = c.kv.e;
      if (c.kv.nu) f.nu = c.kv.nu;
      if (c.kv.rho) f.rho = c.kv.rho;
      return { tool: "material", f };
    case "plate":
    case "box":
    case "quad":
    case "truss":
      geom(["x", "y", "z", "t", "size", "area"]);
      return { tool: c.cmd, f };
    case "beam": {
      geom(["n", "up"]);
      const m = /^(\w+)\((.*)\)$/.exec(c.kv.section ?? "");
      if (m && shapes[m[1]]) {
        f.shape = m[1];
        m[2].split(",").forEach((v, i) => {
          const k = shapes[m[1]].dims[i];
          if (k) f[`d${k}`] = v;
        });
      }
      return { tool: "beam", f };
    }
    case "fix":
    case "pin":
    case "support":
      f.kind = c.cmd;
      for (const d of dofs) {
        if (c.words.includes(d)) f[`c${d}`] = "1";
        if (c.kv[d] !== undefined) {
          f[`c${d}`] = "1";
          f[`v${d}`] = c.kv[d];
        }
      }
      sel();
      return { tool: "support", f };
    case "load":
      for (const k of forces) if (c.kv[k] !== undefined) f[k] = c.kv[k];
      sel();
      return { tool: "load", f };
    case "pressure":
      if (c.kv.p) f.p = c.kv.p;
      sel();
      return { tool: "pressure", f };
    case "gravity":
      for (const k of ["g", "gx", "gy", "gz"]) if (c.kv[k] !== undefined) f[k] = c.kv[k];
      if (c.points[0]) {
        const p = parsePoint(c.points[0]) ?? [];
        ["gx", "gy", "gz"].forEach((k, i) => p[i] !== undefined && (f[k] = String(p[i])));
      }
      return { tool: "gravity", f };
  }
  return null;
}

function T(p: { label: string; v: string; on: (v: string) => void; ph?: string; wide?: boolean; bad?: boolean; unit?: ReactNode }) {
  return (
    <label className={`field${p.wide ? " wide" : ""}`}>
      <span className="flabel">{p.label}</span>
      <span className={`finput${p.bad ? " bad" : ""}`}>
        <input value={p.v} placeholder={p.ph} onChange={(e) => p.on(e.target.value)} spellCheck={false} />
        {p.unit}
      </span>
    </label>
  );
}

/** Koordinaterna för en nod i modellfönstret, avrundade till f32-noggrannhet. */
function nodeCoords(node: number, dim: 2 | 3): string {
  const pos = get().mesh?.positions;
  if (!pos) return "";
  const c = [0, 1, 2].slice(0, dim).map((a) => short(Number(pos[3 * node + a].toPrecision(7))));
  return `(${c.join(",")})`;
}

/** Formulär för ett verktyg. `submit` får raden och returnerar ett fel eller null. */
function ToolForm(p: { tool: Tool; dim: 2 | 3; init: Fields; editing: number | null; materials: string[]; submit: (line: string) => Promise<string | null>; cancel: () => void; remove: () => void }) {
  const { tool, dim } = p;
  const [f, setF] = useState<Fields>(p.init);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const model = useStore((s) => s.model);
  const mesh = useStore((s) => s.mesh);
  const system = useStore((s) => s.session?.units);
  const v = (k: string, d = "") => f[k] ?? d;
  const on = (k: string) => (x: string) => {
    setF((o) => ({ ...o, [k]: x }));
    setOk(null);
  };

  // Enheter: varje värde skrivs i en vald enhet (grupp = fält som delar enhetsval)
  // och räknas om till modellens enhetssystem när raden skrivs. En befintlig rad
  // visas i modellens enheter; en ny i den senast valda enheten.
  const unit = (g: string, q: ConvQ) => f[`u_${g}`] ?? (p.editing !== null ? modelUnit(q, system) : preferredUnit(q, system));
  /** Värdet som det visas (standardvärdet anges i modellens enheter). */
  const dv = (k: string, dModel: string, g: string, q: ConvQ) => f[k] ?? convert(dModel, q, modelUnit(q, system), unit(g, q));
  /** Värdet i modellens enheter, för skriptet. */
  const mv = (k: string, dModel: string, g: string, q: ConvQ) => convert(dv(k, dModel, g, q), q, unit(g, q), modelUnit(q, system));
  const U = (g: string, q: ConvQ) => {
    const u = unit(g, q);
    if (!u) return null;
    return (
      <select
        className="unitsel"
        value={u}
        aria-label="Enhet"
        onChange={(e) => {
          const x = e.target.value;
          setF((o) => ({ ...o, [`u_${g}`]: x }));
          rememberUnit(q, x);
          setOk(null);
        }}
      >
        {unitOptions(q).map((x) => (
          <option key={x} value={x}>
            {x}
          </option>
        ))}
      </select>
    );
  };
  /** Enhetsval för en hel rad fält (koordinater, kraftkomponenter). */
  const UnitCol = (g: string, q: ConvQ) =>
    unit(g, q) ? (
      <label className="field unitcol">
        <span className="flabel">Enhet</span>
        <span className="finput">{U(g, q)}</span>
      </label>
    ) : null;
  /** Värde i modellens enheter → visad enhet (för knappar och nodval). */
  const toShown = (s: string, g: string, q: ConvQ) => convert(s, q, modelUnit(q, system), unit(g, q));

  const pickNode = (k: string, append: boolean) => {
    const prev = get().pickMode;
    set({
      pickMode: "node",
      onNodePick: (n) => {
        set({ pickMode: prev });
        const c = toShown(nodeCoords(n, dim), k === "at" ? "sel" : "pts", "length");
        setF((o) => (append ? { ...o, [k]: [o[k]?.trim(), c].filter(Boolean).join(" ") } : { ...o, [k]: c, sx: "", sy: "", sz: "", all: "" }));
      },
    });
  };

  // urvalet: alla, en punkt eller intervall i x/y/z (i modellens enheter)
  const usesSel = tool === "support" || tool === "load" || tool === "pressure";
  const sv = (k: string) => convert(v(k).trim(), "length", unit("sel", "length"), modelUnit("length", system));
  const selText = () => {
    if (v("all") === "1") return "all";
    const at = sv("at");
    if (at) return `at=${at.startsWith("(") ? at : `(${at})`}`;
    return (["x", "y", "z"] as const)
      .filter((a) => sv(`s${a}`))
      .map((a) => `${a}=${sv(`s${a}`)}`)
      .join(" ");
  };
  const selSpec = (): SelSpec | null => {
    if (v("all") === "1") return { all: true };
    if (sv("at")) {
      const pt = parsePoint(sv("at"));
      return pt ? { at: pt } : null;
    }
    const box: [number, number, number][] = [];
    for (const [i, a] of ["x", "y", "z"].entries()) {
      const s = sv(`s${a}`);
      if (!s) continue;
      const r = parseRange(s);
      if (!r) return null;
      box.push([i, r[0], r[1]]);
    }
    return box.length ? { box } : null;
  };
  const spec = usesSel ? selSpec() : null;
  const hits = useMemo(() => (spec && mesh && mesh.nodes > 0 ? selectNodes(mesh.positions, mesh.nodes, spec) : null), [JSON.stringify(spec), mesh]);
  useEffect(() => {
    set({ marked: hits });
  }, [hits]);
  useEffect(() => () => set({ marked: null, onNodePick: null }), []);

  const bbox = model && model.nodes > 0 ? { lo: model.bbox_min, hi: model.bbox_max } : null;
  const selFields = (
    <div className="selbox">
      <div className="flabel">Urval</div>
      <label className="check">
        <input type="checkbox" checked={v("all") === "1"} onChange={(e) => on("all")(e.target.checked ? "1" : "")} />
        alla noder
      </label>
      {v("all") !== "1" && (
        <>
          <div className="row-fields">
            <T label="x" v={v("sx")} on={on("sx")} ph="0 eller 0..1" />
            <T label="y" v={v("sy")} on={on("sy")} />
            {dim === 3 && <T label="z" v={v("sz")} on={on("sz")} />}
            {UnitCol("sel", "length")}
          </div>
          {bbox && (
            <div className="chips quick" aria-label="Modellens ytterkanter">
              {(["x", "y", "z"] as const).slice(0, dim).flatMap((a, i) =>
                [
                  ["min", bbox.lo[i]],
                  ["max", bbox.hi[i]],
                ].map(([w, val]) => (
                  <button
                    key={a + w}
                    type="button"
                    className="chip small"
                    title={`${a} = ${toShown(short(val as number), "sel", "length")}`}
                    onClick={() => setF((o) => ({ ...o, [`s${a}`]: toShown(short(val as number), "sel", "length"), at: "" }))}
                  >
                    {a} {w}
                  </button>
                )),
              )}
            </div>
          )}
          <div className="row-fields pickrow">
            <T label="eller punkt" v={v("at")} on={on("at")} ph={dim === 3 ? "(x,y,z)" : "(x,y)"} />
            {bbox && <IconButton icon="node" title="Välj nod i modellfönstret" onClick={() => pickNode("at", false)} />}
          </div>
        </>
      )}
      {hits !== null && (
        <div className={`hits${hits.length ? "" : " none"}`}>
          {hits.length ? `Träffar ${hits.length} ${hits.length === 1 ? "nod" : "noder"} (markerade i modellen)` : "Träffar inga noder än"}
        </div>
      )}
    </div>
  );
  const matField = (
    <Select
      label="Material"
      value={v("mat")}
      options={[{ value: "", label: p.materials.length ? "Senast definierade" : "Stål (standard)" }, ...p.materials.map((m) => ({ value: m, label: m }))]}
      onChange={on("mat")}
    />
  );
  const mat = v("mat") ? ` material=${v("mat")}` : "";
  const ptsField = (ph: string) => (
    <div className="row-fields pickrow3">
      <T label="Punkter" v={dv("pts", ph, "pts", "length")} on={on("pts")} wide />
      {UnitCol("pts", "length")}
      {bbox && <IconButton icon="plus" title="Lägg till en nod från modellfönstret" onClick={() => pickNode("pts", true)} />}
    </div>
  );
  /** ` namn=värde` i modellens enheter, eller inget om fältet är tomt. */
  const opt = (k: string, name = k, g?: string, q?: ConvQ) => {
    const x = v(k).trim();
    if (!x) return "";
    return ` ${name}=${g && q ? convert(x, q, unit(g, q), modelUnit(q, system)) : x}`;
  };
  const filled = (...vals: string[]) => vals.every((x) => x.trim());
  /** Platshållare: ett värde i SI visat i vald enhet. */
  const phU = (si: string, g: string, q: ConvQ) => (unit(g, q) ? convert(si, q, "SI", unit(g, q)) : si);

  let line = "";
  let body = null;
  let missing: string | null = null;
  switch (tool) {
    case "material": {
      const name = v("name", "steel").trim().replace(/\s+/g, "_");
      line = `material ${name}${opt("E", "E", "E", "stress")}${opt("nu")}${opt("rho", "rho", "rho", "density")}`;
      if (!name) missing = "Ange ett namn";
      else if (!["steel", "stål", "aluminium", "aluminum"].includes(name.toLowerCase()) && !v("E").trim()) missing = "Ange E (bara steel och aluminium har standardvärden)";
      body = (
        <>
          <T label="Namn" v={v("name", "steel")} on={on("name")} />
          <div className="row-fields">
            <T label="E" v={v("E")} on={on("E")} ph={phU("210e9", "E", "stress")} unit={U("E", "stress")} />
            <T label="ν" v={v("nu")} on={on("nu")} ph="0.3" />
          </div>
          <T label="ρ" v={v("rho")} on={on("rho")} ph={phU("7850", "rho", "density")} unit={U("rho", "density")} />
          <p className="muted">steel och aluminium har standardvärden (i SI). Geometri utan valt material använder det senast definierade.</p>
        </>
      );
      break;
    }
    case "plate": {
      const x = dv("x", "0..1", "xy", "length"), y = dv("y", "0..0.2", "xy", "length"), t = dv("t", "0.01", "t", "length"), size = dv("size", "0.05", "size", "length");
      line = `plate x=${mv("x", "0..1", "xy", "length")} y=${mv("y", "0..0.2", "xy", "length")} t=${mv("t", "0.01", "t", "length")} size=${mv("size", "0.05", "size", "length")}${mat}`;
      if (!filled(x, y, t, size)) missing = "Fyll i alla fält";
      body = (
        <>
          <div className="row-fields">
            <T label="x" v={x} on={on("x")} />
            <T label="y" v={y} on={on("y")} />
            {UnitCol("xy", "length")}
          </div>
          <div className="row-fields">
            <T label="Tjocklek t" v={t} on={on("t")} unit={U("t", "length")} />
            <T label="Elementstorlek" v={size} on={on("size")} unit={U("size", "length")} />
          </div>
          {matField}
        </>
      );
      break;
    }
    case "quad": {
      const ph = "(0,0) (2,0) (2,1) (0,1)";
      const t = dv("t", "0.01", "t", "length"), size = dv("size", "0.05", "size", "length");
      line = `quad ${mv("pts", ph, "pts", "length")} t=${mv("t", "0.01", "t", "length")} size=${mv("size", "0.05", "size", "length")}${mat}`;
      if ((dv("pts", ph, "pts", "length").match(/\(/g) ?? []).length !== 4) missing = "Ange fyra hörn i ordning runt fyrhörningen";
      else if (!filled(t, size)) missing = "Fyll i alla fält";
      body = (
        <>
          {ptsField(ph)}
          <div className="row-fields">
            <T label="Tjocklek t" v={t} on={on("t")} unit={U("t", "length")} />
            <T label="Elementstorlek" v={size} on={on("size")} unit={U("size", "length")} />
          </div>
          {matField}
        </>
      );
      break;
    }
    case "box": {
      const x = dv("x", "0..1", "xyz", "length"), y = dv("y", "0..0.2", "xyz", "length"), z = dv("z", "0..0.2", "xyz", "length"), size = dv("size", "0.05", "size", "length");
      line = `box x=${mv("x", "0..1", "xyz", "length")} y=${mv("y", "0..0.2", "xyz", "length")} z=${mv("z", "0..0.2", "xyz", "length")} size=${mv("size", "0.05", "size", "length")}${mat}`;
      if (!filled(x, y, z, size)) missing = "Fyll i alla fält";
      body = (
        <>
          <div className="row-fields">
            <T label="x" v={x} on={on("x")} />
            <T label="y" v={y} on={on("y")} />
            <T label="z" v={z} on={on("z")} />
            {UnitCol("xyz", "length")}
          </div>
          <T label="Elementstorlek" v={size} on={on("size")} unit={U("size", "length")} />
          {matField}
        </>
      );
      break;
    }
    case "beam": {
      const shape = v("shape", "rect");
      const ph = dim === 3 ? "(0,0,0) (2,0,0)" : "(0,0) (2,0)";
      const dimQ = (k: string): ConvQ => (k === "A" ? "area" : k === "I11" || k === "I22" || k === "J" ? "length4" : "length");
      const vals = shapes[shape].dims.map((k) => mv(`d${k}`, shapeDefaults[k], `d${k}`, dimQ(k)));
      line = `beam ${mv("pts", ph, "pts", "length")} section=${shape}(${vals.join(",")}) n=${v("n", "10")}${dim === 3 ? opt("up") : ""}${mat}`;
      if ((dv("pts", ph, "pts", "length").match(/\(/g) ?? []).length < 2) missing = "Ange minst två punkter";
      body = (
        <>
          {ptsField(ph)}
          <Select label="Tvärsnitt" value={shape} options={Object.entries(shapes).map(([value, s]) => ({ value, label: s.label }))} onChange={on("shape")} />
          <div className="grid2">
            {shapes[shape].dims.map((k) => (
              <T key={k} label={k} v={dv(`d${k}`, shapeDefaults[k], `d${k}`, dimQ(k))} on={on(`d${k}`)} unit={U(`d${k}`, dimQ(k))} />
            ))}
          </div>
          <div className="row-fields">
            <T label="Element per segment" v={v("n", "10")} on={on("n")} />
            {dim === 3 && <T label="Uppåtriktning" v={v("up")} on={on("up")} ph="(0,0,1)" />}
          </div>
          {matField}
        </>
      );
      break;
    }
    case "truss": {
      const ph = dim === 3 ? "(0,0,0) (1,1,0)" : "(0,0) (1,1)";
      const area = dv("area", "1e-4", "area", "area");
      line = `truss ${mv("pts", ph, "pts", "length")} area=${mv("area", "1e-4", "area", "area")}${mat}`;
      if ((dv("pts", ph, "pts", "length").match(/\(/g) ?? []).length < 2) missing = "Ange minst två punkter";
      else if (!filled(area)) missing = "Ange arean";
      body = (
        <>
          {ptsField(ph)}
          <T label="Area" v={area} on={on("area")} unit={U("area", "area")} />
          {matField}
        </>
      );
      break;
    }
    case "support": {
      const kind = v("kind", "fix");
      const avail = dim === 3 ? dofs : ["ux", "uy", "rz"];
      const chosen = avail.filter((d) => v(`c${d}`) === "1");
      // förskjutningar i vald längdenhet, rotationer i radianer
      const dofVal = (d: string) => (d.startsWith("u") ? convert(v(`v${d}`).trim(), "length", unit("sv", "length"), modelUnit("length", system)) : v(`v${d}`).trim());
      const dofText = chosen.map((d) => (v(`v${d}`).trim() && Number(v(`v${d}`)) !== 0 ? `${d}=${dofVal(d)}` : d)).join(" ");
      line = kind === "support" ? `support ${selText()} ${dofText}` : `${kind} ${selText()}`;
      if (kind === "support" && !chosen.length) missing = "Välj minst en frihetsgrad";
      body = (
        <>
          <Segmented
            small
            value={kind}
            onChange={on("kind")}
            options={[
              { value: "fix", label: "Inspänd" },
              { value: "pin", label: "Led" },
              { value: "support", label: "Valda frihetsgrader" },
            ]}
          />
          <p className="muted">
            {kind === "fix" ? "Alla frihetsgrader låsta." : kind === "pin" ? "Förskjutningar låsta, rotationer fria." : "Låsta frihetsgrader; ange ett värde för en föreskriven förskjutning (rotationer i rad)."}
          </p>
          {kind === "support" && (
            <>
              <div className="dofgrid">
                {avail.map((d) => (
                  <div key={d} className="dofcell">
                    <label className="check">
                      <input type="checkbox" checked={v(`c${d}`) === "1"} onChange={(e) => on(`c${d}`)(e.target.checked ? "1" : "")} />
                      {d}
                    </label>
                    {v(`c${d}`) === "1" && (
                      <span className="finput">
                        <input value={v(`v${d}`)} placeholder="0" onChange={(e) => on(`v${d}`)(e.target.value)} aria-label={`Värde för ${d}`} />
                      </span>
                    )}
                  </div>
                ))}
              </div>
              {chosen.some((d) => d.startsWith("u")) && UnitCol("sv", "length")}
            </>
          )}
          {selFields}
        </>
      );
      break;
    }
    case "load": {
      line = `load ${selText()}${forces.map((k) => opt(k, k, k.startsWith("f") ? "f" : "m", k.startsWith("f") ? "force" : "moment")).join("")}`;
      if (!forces.some((k) => v(k).trim() && Number(v(k)) !== 0)) missing = "Ange minst en lastkomponent";
      body = (
        <>
          <div className="row-fields">
            <T label="Fx" v={v("fx")} on={on("fx")} />
            <T label="Fy" v={v("fy")} on={on("fy")} ph={phU("-1000", "f", "force")} />
            {dim === 3 && <T label="Fz" v={v("fz")} on={on("fz")} />}
            {UnitCol("f", "force")}
          </div>
          <div className="row-fields">
            {dim === 3 && <T label="Mx" v={v("mx")} on={on("mx")} />}
            {dim === 3 && <T label="My" v={v("my")} on={on("my")} />}
            <T label="Mz" v={v("mz")} on={on("mz")} />
            {UnitCol("m", "moment")}
          </div>
          <p className="muted">Total kraft och totalt moment på urvalet (fördelas över det). Moment kräver balkar.</p>
          {selFields}
        </>
      );
      break;
    }
    case "pressure":
      line = `pressure ${selText()} p=${mv("p", "1e5", "p", "stress")}`;
      if (!filled(dv("p", "1e5", "p", "stress"))) missing = "Ange trycket";
      body = (
        <>
          <T label="Tryck p" v={dv("p", "1e5", "p", "stress")} on={on("p")} unit={U("p", "stress")} />
          <p className="muted">Normaltryck på randkanter/randytor i urvalet; positivt trycker in i materialet.</p>
          {selFields}
        </>
      );
      break;
    case "gravity": {
      const comps = ["gx", "gy", "gz"].slice(0, dim);
      const vec = comps.some((k) => v(k).trim());
      line = vec ? `gravity${comps.map((k) => opt(k, k, "gv", "acceleration")).join("")}` : `gravity g=${mv("g", "9.81", "g", "acceleration")}`;
      body = (
        <>
          <T label="g (nedåt)" v={dv("g", "9.81", "g", "acceleration")} on={on("g")} unit={U("g", "acceleration")} />
          <div className="row-fields">
            {comps.map((k) => (
              <T key={k} label={k} v={v(k)} on={on(k)} />
            ))}
            {UnitCol("gv", "acceleration")}
          </div>
          <p className="muted">
            Nedåt är {dim === 2 ? "−y" : "−z"}. Ange gx/gy{dim === 3 ? "/gz" : ""} för en annan riktning. Materialen behöver densitet ρ.
          </p>
        </>
      );
      break;
    }
  }
  if (!missing && usesSel && !selText()) missing = "Ange ett urval";
  const go = async () => {
    setErr(null);
    const e = await p.submit(line);
    if (e) setErr(e);
    else setOk(p.editing ? "Ändrad" : "Tillagd");
  };
  return (
    <div className="toolform">
      {body}
      <div className="preview">
        <code>{line}</code>
      </div>
      {err && <div className="formerr">{err}</div>}
      {missing && <div className="muted small">{missing}</div>}
      <div className="row-btns">
        <button type="button" className="btn primary" disabled={!!missing} onClick={go}>
          <Icon name={p.editing ? "check" : "plus"} size={14} /> {p.editing ? "Spara ändring" : "Lägg till"}
        </button>
        {p.editing && (
          <>
            <button type="button" className="btn" onClick={p.cancel}>
              Avbryt
            </button>
            <button type="button" className="btn ghost danger-text" onClick={p.remove}>
              Ta bort
            </button>
          </>
        )}
        {ok && <span className="okmsg">✓ {ok}</span>}
      </div>
    </div>
  );
}

/** Steg för steg för en ny modell: vad som saknas. */
function Guide(p: { cmds: ScriptCmd[]; pick: (t: Tool) => void; geometryTool: Tool }) {
  const has = (cat: string) => p.cmds.some((c) => category(c.cmd) === cat);
  const steps: { label: string; done: boolean; tool?: Tool }[] = [
    { label: "Material", done: has("material"), tool: "material" },
    { label: "Geometri och nät", done: has("geometry"), tool: p.geometryTool },
    { label: "Randvillkor", done: has("support"), tool: "support" },
    { label: "Laster", done: has("load"), tool: "load" },
  ];
  if (steps.every((s) => s.done)) return null;
  return (
    <ol className="guide">
      {steps.map((s) => (
        <li key={s.label} className={s.done ? "done" : ""}>
          <button type="button" className="link" onClick={() => s.tool && p.pick(s.tool)}>
            {s.done ? <Icon name="check" size={12} /> : <span className="dotnum" />}
            {s.label}
          </button>
        </li>
      ))}
      <li>
        <button type="button" className="link" onClick={solve}>
          <span className="dotnum" />
          Lös (F5)
        </button>
      </li>
    </ol>
  );
}

export function ScriptPanel() {
  const session = useStore((s) => s.session);
  const model = useStore((s) => s.model);
  const scriptEdit = useStore((s) => s.scriptEdit);
  const script = session?.script ?? "";
  const [text, setText] = useState(script);
  const [tool, setTool] = useState<Tool>("material");
  const [editing, setEditing] = useState<number | null>(null);
  const [init, setInit] = useState<Fields>({});
  const [formKey, setFormKey] = useState(0);
  const [textErr, setTextErr] = useState<string | null>(null);
  const ta = useRef<HTMLTextAreaElement>(null);
  const toolsRef = useRef<HTMLDivElement>(null);
  useEffect(() => setText(script), [script]);
  const dirty = text !== script;
  const dim = ((session?.script_dim ?? model?.dim) === 2 ? 2 : 3) as 2 | 3;
  const lines = useMemo(() => text.split("\n").length, [text]);
  const cmds = useMemo(() => parseScript(text), [text]);
  const mats = useMemo(() => materialNames(cmds), [cmds]);
  const errLine = Number(/rad (\d+)/.exec(textErr ?? "")?.[1] ?? 0);

  const apply = async (t: string) => {
    const e = await tryEdit({ kind: "FemText", text: t });
    return e;
  };
  const applyText = async () => setTextErr(await apply(text));

  const openTool = (t: Tool, f: Fields = {}, line: number | null = null) => {
    setTool(t);
    setInit(f);
    setEditing(line);
    setFormKey((k) => k + 1);
    requestAnimationFrame(() => toolsRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" }));
  };
  const startEdit = (c: ScriptCmd) => {
    const r = formFromCmd(c);
    if (r) openTool(r.tool, r.f, c.line);
  };
  const remove = async (line: number) => {
    const e = await apply(setLine(text, line, null));
    if (e) setTextErr(e);
    else if (editing !== null) openTool(tool);
  };

  // öppna en rad som valts i modellträdet
  useEffect(() => {
    if (scriptEdit === null) return;
    set({ scriptEdit: null });
    const c = cmds.find((x) => x.line === scriptEdit);
    if (c) startEdit(c);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scriptEdit]);

  /** Inställningsrader (model/plane/elements): ersätt befintlig eller lägg först. */
  const setSetting = async (cmd: string, word: string) => {
    const c = cmds.find((x) => x.cmd === cmd);
    let t: string;
    if (c) t = setLine(text, c.line, `${cmd} ${word}`);
    else {
      const ls = text.split("\n");
      const modelLine = cmds.find((x) => x.cmd === "model");
      const at = modelLine && cmd !== "model" ? modelLine.line : (cmds[0]?.line ?? ls.length) - 1;
      ls.splice(at, 0, `${cmd} ${word}`);
      t = ls.join("\n");
    }
    // t.ex. 3D med en platta i modellen: felet visas där man klickade
    const e = await apply(t);
    if (e) toast("error", e);
  };
  const setting = (cmd: string, d: string) => cmds.find((x) => x.cmd === cmd)?.words[0] ?? d;

  const submit = async (line: string) => {
    const t = editing !== null ? setLine(text, editing, line) : setLine(text, null, line);
    const e = await apply(t);
    if (!e) setTextErr(null);
    return e;
  };
  const geometryTool: Tool = dim === 2 ? "plate" : "box";
  const listed = cmds.filter((c) => category(c.cmd) && category(c.cmd) !== "model");

  return (
    <>
      {session?.problem && (
        <div className="note warn">
          <b>Kan inte lösas än:</b> {session.problem}
        </div>
      )}
      <Section title="Modell">
        <Segmented
          small
          value={String(dim) as "2" | "3"}
          onChange={(v) => setSetting("model", `${v}d`)}
          options={[
            { value: "2", label: "2D" },
            { value: "3", label: "3D" },
          ]}
        />
        {dim === 2 && (
          <div className="row-fields">
            <Select
              label="Tillstånd"
              value={setting("plane", "stress")}
              options={[
                { value: "stress", label: "Plan spänning" },
                { value: "strain", label: "Plan töjning" },
              ]}
              onChange={(v) => setSetting("plane", v)}
            />
            <Select
              label="Element"
              value={setting("elements", "quad")}
              options={[
                { value: "quad", label: "Fyrhörningar" },
                { value: "tri", label: "Trianglar" },
              ]}
              onChange={(v) => setSetting("elements", v)}
            />
          </div>
        )}
        <Guide cmds={cmds} pick={(t) => openTool(t)} geometryTool={geometryTool} />
      </Section>
      <Section title={`Innehåll (${listed.length})`}>
        {!listed.length && <p className="muted">Inga kommandon än. Börja med ett material och en geometri.</p>}
        {categories
          .filter((g) => g.id !== "model")
          .map((g) => {
            const items = listed.filter((c) => category(c.cmd) === g.id);
            if (!items.length) return null;
            return (
              <div key={g.id} className="cmdgroup">
                <div className="cmdgroup-title">{g.title}</div>
                {items.map((c) => {
                  const s = summary(c);
                  return (
                    <div key={c.line} className={`cmdrow${editing === c.line ? " on" : ""}${errLine === c.line ? " bad" : ""}`}>
                      <button type="button" className="cmdmain" onClick={() => startEdit(c)} title="Ändra">
                        <span className="cmd-title">{s.title}</span>
                        {s.sub && <span className="cmd-sub">{s.sub}</span>}
                      </button>
                      <span className="cmd-line">{c.line}</span>
                      <IconButton icon="close" size={13} title={`Ta bort rad ${c.line}`} onClick={() => remove(c.line)} />
                    </div>
                  );
                })}
              </div>
            );
          })}
      </Section>
      <div ref={toolsRef}>
        <Section
          title={editing !== null ? `Ändra rad ${editing}` : "Lägg till"}
          right={
            editing !== null ? (
              <button type="button" className="btn small" onClick={() => openTool(tool)}>
                Ny i stället
              </button>
            ) : undefined
          }
        >
          {editing === null && (
            <div className="toolbar-chips">
              {tools
                .filter((t) => t.dims.includes(dim))
                .map((t) => (
                  <button key={t.id} type="button" className={`chip${tool === t.id ? " on" : ""}`} onClick={() => openTool(t.id)}>
                    {t.label}
                  </button>
                ))}
            </div>
          )}
          {tools.find((t) => t.id === tool)?.dims.includes(dim) || editing !== null ? (
            <ToolForm
              key={`${formKey}-${dim}`}
              tool={tool}
              dim={dim}
              init={init}
              editing={editing}
              materials={mats}
              submit={submit}
              cancel={() => openTool(tool)}
              remove={() => editing !== null && remove(editing)}
            />
          ) : (
            <p className="muted">Välj ett verktyg ovan.</p>
          )}
        </Section>
      </div>
      <Section
        title="Skript"
        right={
          <button type="button" className={`btn small${dirty ? " primary" : ""}`} disabled={!dirty} onClick={applyText} title="Tillämpa (Ctrl+Enter)">
            Tillämpa
          </button>
        }
      >
        <div className={`code${textErr ? " bad" : ""}`}>
          <div className="gutter" aria-hidden="true">
            {Array.from({ length: lines }, (_, i) => (
              <span key={i} className={errLine === i + 1 ? "bad" : undefined}>
                {i + 1}
              </span>
            ))}
          </div>
          <textarea
            ref={ta}
            value={text}
            spellCheck={false}
            onChange={(e) => setText(e.target.value)}
            onScroll={(e) => {
              const g = (e.target as HTMLElement).previousElementSibling as HTMLElement;
              g.scrollTop = (e.target as HTMLElement).scrollTop;
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                applyText();
              }
              if (e.key === "Tab") {
                e.preventDefault();
                const el = e.currentTarget;
                const s = el.selectionStart;
                setText(text.slice(0, s) + "  " + text.slice(el.selectionEnd));
                requestAnimationFrame(() => (el.selectionStart = el.selectionEnd = s + 2));
              }
            }}
            aria-label=".rusty-skript"
          />
        </div>
        {textErr && <div className="formerr">{textErr}</div>}
        {dirty && !textErr && <div className="muted small">Ändringar som inte är tillämpade. Verktygen ovan utgår från texten här.</div>}
        <p className="muted">
          Kommandon: model 2d|3d, plane stress|strain, elements quad|tri, material, plate, quad, box, beam, truss, fix, pin, support, load, pressure, gravity. Ctrl+Enter tillämpar.
        </p>
      </Section>
    </>
  );
}
