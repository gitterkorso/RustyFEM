// The frame around the viewport: the top bar, the command palette and the start view.

import { useEffect, useMemo, useRef, useState } from "react";
import { blankModel, exportScript, newModel, openDialog, openPath, redo, save, solve, undo, cancel } from "../actions";
import { get, persist, set, useStore } from "../store";
import { Icon, IconButton, Segmented } from "./ui";

export function TopBar() {
  const session = useStore((s) => s.session);
  const mode = useStore((s) => s.mode);
  const running = useStore((s) => s.run.running);
  const progress = useStore((s) => s.run.progress);
  const hasResults = useStore((s) => s.results.length > 0);
  const dark = useStore((s) => s.dark);
  const left = useStore((s) => s.left);
  const right = useStore((s) => s.right);
  const dock = useStore((s) => s.dock);
  const mcpOn = useStore((s) => s.mcp?.running ?? false);
  return (
    <header className="topbar">
      <div className="tb-left">
        <IconButton icon="sidebar" title="Model tree" active={left} onClick={() => set((s) => ({ left: !s.left }))} />
        <span className="brand" aria-label="RustyFEM">
          <img className="brand-icon" src="/brand/rustyfem-icon.png" alt="" width="28" height="28" draggable={false} />
          <b><span className="brand-rust">rusty</span>FEM</b>
        </span>
        <span className="sep" />
        <IconButton icon="new" title="New model (empty 2D)" kbd="Ctrl+N" onClick={() => newModel(blankModel[2])} />
        <IconButton icon="open" title="Open" kbd="Ctrl+O" onClick={openDialog} />
        <IconButton icon="save" title="Save project" kbd="Ctrl+S" onClick={() => save()} disabled={!session} />
        <span className="sep" />
        <IconButton icon="undo" title={session?.undo ? `Undo ${session.undo}` : "Undo"} kbd="Ctrl+Z" onClick={undo} disabled={!session?.undo} />
        <IconButton icon="redo" title={session?.redo ? `Redo ${session.redo}` : "Redo"} kbd="Ctrl+Y" onClick={redo} disabled={!session?.redo} />
      </div>
      <div className="tb-center">
        {session && (
          <>
            <span className="docname" title={session.path ?? ""}>
              {session.name}
              {session.dirty && <span className="dot" title="Unsaved changes" />}
            </span>
            <Segmented
              value={mode}
              onChange={(v) => set({ mode: v, pick: null })}
              options={[
                { value: "model", label: "Model", icon: "layers" },
                { value: "results", label: "Results", icon: "chart" },
              ]}
            />
          </>
        )}
      </div>
      <div className="tb-right">
        {session &&
          (running ? (
            <button type="button" className="btn danger run" onClick={cancel} title="Cancel (Esc)">
              <span className="ring" style={{ ["--p" as string]: `${progress * 360}deg` }} />
              Cancel
            </button>
          ) : (
            <button type="button" className="btn primary run" onClick={solve} title="Solve (F5)" aria-label="Solve">
              <Icon name="play" size={14} />
              <span className="run-label">Solve</span>
            </button>
          ))}
        {hasResults && mode === "model" && (
          <button type="button" className="btn ghost hide-narrow" onClick={() => set({ mode: "results" })}>
            Show results
          </button>
        )}
        <span className="mcp-btn">
          <IconButton icon="plug" title={mcpOn ? "AI assistant (MCP server running)" : "Connect an AI assistant (MCP)"} active={mcpOn} onClick={() => set({ mcpPanel: true })} />
          {mcpOn && <span className="led on" />}
        </span>
        <IconButton icon="search" title="Commands" kbd="Ctrl+K" onClick={() => set({ palette: true })} />
        <IconButton
          icon={dark ? "sun" : "moon"}
          title={dark ? "Light theme" : "Dark theme"}
          onClick={() => {
            set({ dark: !dark });
            persist("dark", !dark);
          }}
        />
        <IconButton icon="dock" title="Run panel" active={dock} onClick={() => set((s) => (persist("dock", !s.dock), { dock: !s.dock }))} />
        <IconButton icon="sidebarR" title="Properties" active={right} onClick={() => set((s) => ({ right: !s.right }))} />
      </div>
    </header>
  );
}

interface Cmd {
  label: string;
  hint?: string;
  kbd?: string;
  run: () => void;
  when?: () => boolean;
}

function commands(): Cmd[] {
  const has = () => get().session !== null;
  return [
    { label: "New empty 2D model", kbd: "Ctrl+N", run: () => newModel(blankModel[2]) },
    { label: "New empty 3D model", run: () => newModel(blankModel[3]) },
    { label: "New model from template (cantilever plate)", run: () => newModel() },
    { label: "Open …", kbd: "Ctrl+O", run: openDialog },
    { label: "Save project", kbd: "Ctrl+S", run: () => save(), when: has },
    { label: "Save project as …", kbd: "Ctrl+Shift+S", run: () => save(true), when: has },
    { label: "Export .rusty script …", run: exportScript, when: () => get().session?.kind === "rusty" },
    { label: "Solve the model", kbd: "F5", run: solve, when: has },
    { label: "Cancel the run", kbd: "Esc", run: cancel, when: () => get().run.running },
    { label: "Undo", kbd: "Ctrl+Z", run: undo, when: () => !!get().session?.undo },
    { label: "Redo", kbd: "Ctrl+Y", run: redo, when: () => !!get().session?.redo },
    { label: "Show model", run: () => set({ mode: "model" }), when: has },
    { label: "Show results", run: () => set({ mode: "results" }), when: () => get().results.length > 0 },
    { label: "Build model (script and tools)", run: () => set({ sel: { type: "script" }, right: true }), when: () => get().session?.kind === "rusty" },
    { label: "Model overview", run: () => set({ sel: { type: "model" }, right: true }), when: has },
    { label: "Connect an AI assistant (MCP) …", run: () => set({ mcpPanel: true }) },
    { label: "Disclaimer – not for real-world design", run: () => set({ disclaimer: true }) },
    { label: "Toggle theme", run: () => set((s) => (persist("dark", !s.dark), { dark: !s.dark })) },
    { label: "Show/hide model tree", run: () => set((s) => ({ left: !s.left })) },
    { label: "Show/hide properties", run: () => set((s) => ({ right: !s.right })) },
    { label: "Show/hide run panel", run: () => set((s) => ({ dock: !s.dock })) },
    { label: "Surfaces on/off", run: () => set((s) => ({ surfaces: !s.surfaces })) },
    { label: "Element edges on/off", run: () => set((s) => ({ edges: !s.edges })) },
    { label: "Profiles and shell thickness in 3D on/off", run: () => set((s) => ({ profiles: !s.profiles })) },
    { label: "Boundary conditions and loads on/off", run: () => set((s) => ({ symbols: !s.symbols })) },
    { label: "Pick elements", run: () => set({ pickMode: "element" }) },
    { label: "Pick nodes", run: () => set({ pickMode: "node" }) },
    ...get().recent.map((p) => ({ label: `Open ${p.split(/[\\/]/).pop()}`, hint: p, run: () => openPath(p) })),
  ];
}

export function Palette() {
  const open = useStore((s) => s.palette);
  const [q, setQ] = useState("");
  const [i, setI] = useState(0);
  const input = useRef<HTMLInputElement>(null);
  const list = useMemo(() => {
    if (!open) return [];
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return commands().filter((c) => (!c.when || c.when()) && words.every((w) => `${c.label} ${c.hint ?? ""}`.toLowerCase().includes(w)));
  }, [open, q]);
  useEffect(() => {
    if (open) {
      setQ("");
      setI(0);
      requestAnimationFrame(() => input.current?.focus());
    }
  }, [open]);
  if (!open) return null;
  const close = () => set({ palette: false });
  const runAt = (k: number) => {
    const c = list[k];
    close();
    c?.run();
  };
  return (
    <div className="overlay" onMouseDown={close}>
      <div className="palette" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="Commands">
        <div className="pal-input">
          <Icon name="search" />
          <input
            ref={input}
            value={q}
            placeholder="Type a command …"
            onChange={(e) => {
              setQ(e.target.value);
              setI(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") {
                e.preventDefault();
                setI(Math.min(i + 1, list.length - 1));
              } else if (e.key === "ArrowUp") {
                e.preventDefault();
                setI(Math.max(i - 1, 0));
              } else if (e.key === "Enter") runAt(i);
              else if (e.key === "Escape") close();
            }}
          />
        </div>
        <div className="pal-list">
          {list.map((c, k) => (
            <button key={c.label + k} type="button" className={k === i ? "on" : ""} onMouseEnter={() => setI(k)} onClick={() => runAt(k)}>
              <span>{c.label}</span>
              {c.hint && <span className="muted ellipsis">{c.hint}</span>}
              {c.kbd && <kbd>{c.kbd}</kbd>}
            </button>
          ))}
          {!list.length && <div className="muted pad">No matches</div>}
        </div>
      </div>
    </div>
  );
}

const examples: { name: string; script: string }[] = [
  {
    name: "Cantilever plate (2D)",
    script: "model 2d\nmaterial steel E=210e9 nu=0.3\nplate x=0..2 y=0..0.2 t=0.01 size=0.025\nfix x=0\nload x=2 fy=-1000\n",
  },
  {
    name: "Loaded block (3D)",
    script: "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.05\nfix x=0\nload x=1 fy=-5000\n",
  },
  {
    name: "Beam frame (3D)",
    script:
      "model 3d\nmaterial steel\nbeam (0,0,0) (0,3,0) (4,3,0) (4,0,0) section=ibeam(0.2,0.1,0.01,0.006) n=12\nbeam (0,3,0) (0,3,3) (4,3,3) (4,3,0) section=rect(0.1,0.1) n=10\nbeam (0,3,3) (0,0,3) section=rect(0.1,0.1) n=8\nbeam (4,3,3) (4,0,3) section=rect(0.1,0.1) n=8\nfix y=0\nload at=(2,3,0) fy=-20000\n",
  },
  {
    name: "Truss (2D)",
    script:
      "model 2d\nmaterial steel\ntruss (0,0) (1,0) (2,0) (3,0) (4,0) area=4e-4\ntruss (0,0) (0.5,0.8) (1,0) (1.5,0.8) (2,0) (2.5,0.8) (3,0) (3.5,0.8) (4,0) area=4e-4\ntruss (0.5,0.8) (1.5,0.8) (2.5,0.8) (3.5,0.8) area=4e-4\npin at=(0,0)\nsupport at=(4,0) uy\nload at=(2,0) fy=-50000\n",
  },
];

export function Welcome() {
  const recent = useStore((s) => s.recent);
  return (
    <div className="welcome">
      <div className="welcome-card">
        <h1>
          <img className="welcome-logo" src="/brand/rustyfem-logo.png" alt="rustyFEM" width="2048" height="768" draggable={false} />
        </h1>
        <p className="muted">Build a model from scratch, open an input file (.inp) or a project, solve and review the results.</p>
        <div className="welcome-actions">
          <button type="button" className="btn primary big" onClick={() => newModel(blankModel[2])}>
            <Icon name="new" /> New 2D model
          </button>
          <button type="button" className="btn big" onClick={() => newModel(blankModel[3])}>
            <Icon name="cube" /> New 3D model
          </button>
          <button type="button" className="btn big" onClick={openDialog}>
            <Icon name="open" /> Open …
          </button>
        </div>
        <div className="welcome-cols">
          <div>
            <h4>Examples</h4>
            {examples.map((e) => (
              <button key={e.name} type="button" className="listbtn" onClick={() => newModel(e.script)}>
                {e.name}
              </button>
            ))}
          </div>
          <div>
            <h4>Recent</h4>
            {recent.length === 0 && <p className="muted">No files yet.</p>}
            {recent.map((p) => (
              <button key={p} type="button" className="listbtn" title={p} onClick={() => openPath(p)}>
                <span>{p.split(/[\\/]/).pop()}</span>
                <span className="muted ellipsis">{p}</span>
              </button>
            ))}
          </div>
        </div>
        <p className="muted small">
          <kbd>Ctrl</kbd>+<kbd>K</kbd> commands · <kbd>F5</kbd> solve · <kbd>F</kbd> fit view · <kbd>0</kbd>–<kbd>6</kbd> views
        </p>
        <p className="muted small welcome-disclaimer">
          Not for real-world design – results come with no warranty.{" "}
          <button type="button" className="linkbtn" onClick={() => set({ disclaimer: true })}>
            Read the disclaimer
          </button>
        </p>
      </div>
    </div>
  );
}
