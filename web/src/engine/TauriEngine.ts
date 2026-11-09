// `Engine` on top of Tauri: JSON commands via the generated bindings,
// large data as raw bytes (ArrayBuffer) directly into typed arrays.

import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";
import { commands } from "./bindings";
import type { Engine, EngineEvents, FieldData, McpStatus, MeshData, Opened, ProfileData, RunEvent, Symbols, UnitSystem } from "./Engine";

type Result<T> = { status: "ok"; data: T } | { status: "error"; error: string };

async function ok<T>(p: Promise<Result<T>>): Promise<T> {
  const r = await p;
  if (r.status === "error") throw new Error(r.error);
  return r.data;
}

async function bytes(cmd: string, args: Record<string, unknown>): Promise<ArrayBuffer> {
  return invoke<ArrayBuffer>(cmd, args);
}

export class TauriEngine implements Engine {
  newModel = (script?: string) => ok(commands.newModel(script ?? null));
  open = (path: string) => ok(commands.openFile(path));
  close = async (id: number) => {
    await ok(commands.closeSession(id));
  };
  save = (id: number, path: string) => ok(commands.saveProject(id, path));
  sessionInfo = (id: number) => ok(commands.sessionInfo(id));
  modelInfo = (id: number) => ok(commands.modelInfo(id));
  setUnits = (id: number, units: UnitSystem) => ok(commands.setUnits(id, units));
  unitSystems = () => commands.unitSystems();
  edit = (id: number, edit: Parameters<Engine["edit"]>[1]) => ok(commands.applyEdit(id, edit));
  undo = (id: number) => ok(commands.undo(id));
  redo = (id: number) => ok(commands.redo(id));
  solvers = () => commands.solvers();
  estimate = (id: number, solver: string | null) => ok(commands.estimate(id, solver));
  cancel = async (id: number) => {
    await ok(commands.cancel(id));
  };
  results = (id: number) => ok(commands.results(id));
  history = async (id: number, step: number, index: number) =>
    (await ok(commands.history(id, step, index))).map(([t, v]) => [t ?? NaN, v ?? NaN] as [number, number]);
  pick = (id: number, kind: "node" | "element", index: number, at: { step: number; frame: number } | null) =>
    ok(commands.pick(id, kind, index, at?.step ?? null, at?.frame ?? null));
  systemMemory = async () => (await commands.systemMemory()) ?? 0;
  selftest = () => commands.selftest();
  mcpStatus = () => commands.mcpStatus();
  mcpStart = (port: number | null, network: boolean | null) => ok(commands.mcpStart(port, network));
  mcpStop = () => commands.mcpStop();
  mcpInstallClaudeDesktop = () => ok(commands.mcpInstallClaudeDesktop());

  async listen(h: EngineEvents) {
    const offs = await Promise.all([
      listen<Opened>("mcp-opened", (e) => h.opened(e.payload)),
      listen<{ id: number; geometry: boolean }>("mcp-changed", (e) => h.changed(e.payload.id, e.payload.geometry)),
      listen<{ id: number; event: RunEvent }>("run-event", (e) => h.run(e.payload.id, e.payload.event)),
      listen<McpStatus>("mcp-status", (e) => h.mcp(e.payload)),
    ]);
    return () => offs.forEach((f) => f());
  }
  selftestDone = async (png: string, report: string) => {
    await ok(commands.selftestDone(png, report));
  };

  async solve(id: number, solver: string | null, allFrames: boolean, onEvent: (e: RunEvent) => void) {
    const ch = new Channel<RunEvent>();
    ch.onmessage = onEvent;
    await ok(commands.solve(id, solver, allFrames, ch));
  }

  async mesh(id: number): Promise<MeshData> {
    const b = await bytes("mesh", { id });
    const h = new Uint32Array(b, 0, 5);
    const [nodes, ntri, nedge, nline, npoint] = h;
    let o = 20;
    const take = (n: number) => {
      const a = new Uint32Array(b, o, n);
      o += n * 4;
      return a;
    };
    const positions = new Float32Array(b, o, nodes * 3);
    o += nodes * 12;
    return {
      nodes,
      positions,
      tris: take(ntri * 3),
      triElem: take(ntri),
      edges: take(nedge * 2),
      lines: take(nline * 2),
      lineElem: take(nline),
      points: take(npoint),
      pointElem: take(npoint),
      profiles: await this.profiles(id).catch(() => null),
    };
  }

  private async profiles(id: number): Promise<ProfileData> {
    const b = await bytes("profiles", { id });
    const dv = new DataView(b);
    let o = 0;
    const u32 = () => {
      const v = dv.getUint32(o, true);
      o += 4;
      return v;
    };
    const f32 = () => {
      const v = dv.getFloat32(o, true);
      o += 4;
      return v;
    };
    const nseg = u32();
    const segProfile = new Int32Array(nseg);
    const segN1 = new Float32Array(nseg * 3);
    for (let s = 0; s < nseg; s++) {
      segProfile[s] = dv.getInt32(o, true);
      o += 4;
      for (let c = 0; c < 3; c++) segN1[s * 3 + c] = f32();
    }
    const profiles: Float32Array[][] = [];
    for (let p = u32(); p > 0; p--) {
      const polys: Float32Array[] = [];
      for (let q = u32(); q > 0; q--) {
        const n = u32();
        const pts = new Float32Array(n * 2);
        for (let i = 0; i < n * 2; i++) pts[i] = f32();
        polys.push(pts);
      }
      profiles.push(polys);
    }
    const nel = u32();
    const shell = new Float32Array(nel * 5);
    for (let i = 0; i < nel * 5; i++) shell[i] = f32();
    const threeD = o + 4 <= b.byteLength && u32() === 1;
    return { segProfile, segN1, profiles, shell, threeD };
  }

  async symbols(id: number, step: number | null): Promise<Symbols> {
    const b = await bytes("symbols", { id, step });
    const [nbc, nload] = new Uint32Array(b, 0, 2);
    const bc = new Uint32Array(b, 8, nbc * 2);
    const bcNodes = new Uint32Array(nbc);
    const bcMasks = new Uint32Array(nbc);
    for (let i = 0; i < nbc; i++) {
      bcNodes[i] = bc[2 * i];
      bcMasks[i] = bc[2 * i + 1];
    }
    const loadNodes = new Uint32Array(nload);
    const loads = new Float32Array(nload * 6);
    const dv = new DataView(b, 8 + nbc * 8);
    for (let i = 0; i < nload; i++) {
      loadNodes[i] = dv.getUint32(i * 28, true);
      for (let k = 0; k < 6; k++) loads[i * 6 + k] = dv.getFloat32(i * 28 + 4 + k * 4, true);
    }
    return { bcNodes, bcMasks, loadNodes, loads };
  }

  async field(id: number, step: number, frame: number, field: string, component: string): Promise<FieldData> {
    const b = await bytes("field", { id, step, frame, field, component });
    const n = b.byteLength / 4 - 2;
    const all = new Float32Array(b);
    return { values: all.subarray(0, n), min: all[n], max: all[n + 1] };
  }

  async displacement(id: number, step: number, frame: number): Promise<Float32Array> {
    return new Float32Array(await bytes("displacement", { id, step, frame }));
  }

  async openDialog() {
    const r = await open({
      multiple: false,
      filters: [
        { name: "Models", extensions: ["rustyproj", "inp", "rusty"] },
        { name: "RustyFEM project", extensions: ["rustyproj"] },
        { name: "Input file (.inp)", extensions: ["inp"] },
        { name: "RustyFEM script", extensions: ["rusty"] },
      ],
    });
    return typeof r === "string" ? r : null;
  }

  async saveDialog(defaultName: string, kind: "project" | "script") {
    const r = await save({
      defaultPath: defaultName,
      filters: kind === "project" ? [{ name: "RustyFEM project", extensions: ["rustyproj"] }] : [{ name: "RustyFEM script", extensions: ["rusty"] }],
    });
    return r ?? null;
  }
}
