// The Visualization tab: mounts `VisualScene` (plain Three.js) for the open model.

import { useEffect, useRef, useState } from "react";
import { engine, toast } from "../actions";
import { useStore } from "../store";
import { VisualScene } from "../visual/VisualScene";
import { modelUnit, resultUnit } from "./units";
import type { ConvQ } from "./units";

export function VisualView() {
  const host = useRef<HTMLDivElement>(null);
  const id = useStore((s) => s.id);
  const mesh = useStore((s) => s.mesh);
  const meshVersion = useStore((s) => s.meshVersion);
  const results = useStore((s) => s.results);
  const session = useStore((s) => s.session);
  const dark = useStore((s) => s.dark);
  const resultUnits = useStore((s) => s.resultUnits);
  const [err, setErr] = useState<string | null>(null);
  const system = session?.units;
  const name = session?.name ?? "";
  useEffect(() => {
    if (id === null || !mesh || !host.current) return;
    let scene: VisualScene | null = null;
    let dead = false;
    setErr(null);
    engine
      .visualModel(id)
      .then((vm) => {
        if (dead || !host.current) return;
        scene = new VisualScene(host.current, {
          engine,
          id,
          name,
          mesh,
          vm,
          results,
          dark,
          units: {
            system,
            model: (q: ConvQ) => modelUnit(q, system),
            show: (q: ConvQ) => {
              const m = modelUnit(q, system);
              return m ? resultUnit(m, resultUnits).to || m : null;
            },
          },
          savePng: async (url) => {
            try {
              if (await engine.savePng(`${name.replace(/\.[^.]+$/, "") || "model"}.png`, url)) toast("info", "Image saved");
            } catch (e) {
              toast("error", String(e));
            }
          },
        });
      })
      .catch((e) => setErr(String(e)));
    return () => {
      dead = true;
      scene?.dispose();
    };
  }, [id, mesh, meshVersion, results, dark, system, resultUnits, name]);
  return (
    <div className="visual-view" ref={host}>
      {err && <div className="empty pad">{err}</div>}
    </div>
  );
}
