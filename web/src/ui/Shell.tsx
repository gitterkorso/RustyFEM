// Ramen runt modellfönstret: toppraden, kommandopaletten och startvyn.

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
        <IconButton icon="sidebar" title="Modellträd" active={left} onClick={() => set((s) => ({ left: !s.left }))} />
        <span className="brand" aria-label="RustyFEM">
          <img className="brand-icon" src="/brand/rustyfem-icon.png" alt="" width="28" height="28" draggable={false} />
          <b><span className="brand-rust">rusty</span>FEM</b>
        </span>
        <span className="sep" />
        <IconButton icon="new" title="Ny modell (tom 2D)" kbd="Ctrl+N" onClick={() => newModel(blankModel[2])} />
        <IconButton icon="open" title="Öppna" kbd="Ctrl+O" onClick={openDialog} />
        <IconButton icon="save" title="Spara projekt" kbd="Ctrl+S" onClick={() => save()} disabled={!session} />
        <span className="sep" />
        <IconButton icon="undo" title={session?.undo ? `Ångra ${session.undo}` : "Ångra"} kbd="Ctrl+Z" onClick={undo} disabled={!session?.undo} />
        <IconButton icon="redo" title={session?.redo ? `Gör om ${session.redo}` : "Gör om"} kbd="Ctrl+Y" onClick={redo} disabled={!session?.redo} />
      </div>
      <div className="tb-center">
        {session && (
          <>
            <span className="docname" title={session.path ?? ""}>
              {session.name}
              {session.dirty && <span className="dot" title="Osparade ändringar" />}
            </span>
            <Segmented
              value={mode}
              onChange={(v) => set({ mode: v, pick: null })}
              options={[
                { value: "model", label: "Modell", icon: "layers" },
                { value: "results", label: "Resultat", icon: "chart" },
              ]}
            />
          </>
        )}
      </div>
      <div className="tb-right">
        {session &&
          (running ? (
            <button type="button" className="btn danger run" onClick={cancel} title="Avbryt (Esc)">
              <span className="ring" style={{ ["--p" as string]: `${progress * 360}deg` }} />
              Avbryt
            </button>
          ) : (
            <button type="button" className="btn primary run" onClick={solve} title="Lös (F5)">
              <Icon name="play" size={14} />
              Lös
            </button>
          ))}
        {hasResults && mode === "model" && (
          <button type="button" className="btn ghost hide-narrow" onClick={() => set({ mode: "results" })}>
            Visa resultat
          </button>
        )}
        <span className="mcp-btn">
          <IconButton icon="plug" title={mcpOn ? "AI-assistent (MCP-servern körs)" : "Anslut AI-assistent (MCP)"} active={mcpOn} onClick={() => set({ mcpPanel: true })} />
          {mcpOn && <span className="led on" />}
        </span>
        <IconButton icon="search" title="Kommandon" kbd="Ctrl+K" onClick={() => set({ palette: true })} />
        <IconButton
          icon={dark ? "sun" : "moon"}
          title={dark ? "Ljust tema" : "Mörkt tema"}
          onClick={() => {
            set({ dark: !dark });
            persist("dark", !dark);
          }}
        />
        <IconButton icon="dock" title="Körpanel" active={dock} onClick={() => set((s) => (persist("dock", !s.dock), { dock: !s.dock }))} />
        <IconButton icon="sidebarR" title="Egenskaper" active={right} onClick={() => set((s) => ({ right: !s.right }))} />
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
    { label: "Ny tom 2D-modell", kbd: "Ctrl+N", run: () => newModel(blankModel[2]) },
    { label: "Ny tom 3D-modell", run: () => newModel(blankModel[3]) },
    { label: "Ny modell från mall (konsolplatta)", run: () => newModel() },
    { label: "Öppna …", kbd: "Ctrl+O", run: openDialog },
    { label: "Spara projekt", kbd: "Ctrl+S", run: () => save(), when: has },
    { label: "Spara projekt som …", kbd: "Ctrl+Shift+S", run: () => save(true), when: has },
    { label: "Exportera .rusty-skript …", run: exportScript, when: () => get().session?.kind === "rusty" },
    { label: "Lös modellen", kbd: "F5", run: solve, when: has },
    { label: "Avbryt körningen", kbd: "Esc", run: cancel, when: () => get().run.running },
    { label: "Ångra", kbd: "Ctrl+Z", run: undo, when: () => !!get().session?.undo },
    { label: "Gör om", kbd: "Ctrl+Y", run: redo, when: () => !!get().session?.redo },
    { label: "Visa modell", run: () => set({ mode: "model" }), when: has },
    { label: "Visa resultat", run: () => set({ mode: "results" }), when: () => get().results.length > 0 },
    { label: "Bygg modell (skript och verktyg)", run: () => set({ sel: { type: "script" }, right: true }), when: () => get().session?.kind === "rusty" },
    { label: "Modellöversikt", run: () => set({ sel: { type: "model" }, right: true }), when: has },
    { label: "Anslut AI-assistent (MCP) …", run: () => set({ mcpPanel: true }) },
    { label: "Växla tema", run: () => set((s) => (persist("dark", !s.dark), { dark: !s.dark })) },
    { label: "Visa/dölj modellträd", run: () => set((s) => ({ left: !s.left })) },
    { label: "Visa/dölj egenskaper", run: () => set((s) => ({ right: !s.right })) },
    { label: "Visa/dölj körpanel", run: () => set((s) => ({ dock: !s.dock })) },
    { label: "Ytor av/på", run: () => set((s) => ({ surfaces: !s.surfaces })) },
    { label: "Elementkanter av/på", run: () => set((s) => ({ edges: !s.edges })) },
    { label: "Randvillkor och laster av/på", run: () => set((s) => ({ symbols: !s.symbols })) },
    { label: "Välj element", run: () => set({ pickMode: "element" }) },
    { label: "Välj nod", run: () => set({ pickMode: "node" }) },
    ...get().recent.map((p) => ({ label: `Öppna ${p.split(/[\\/]/).pop()}`, hint: p, run: () => openPath(p) })),
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
      <div className="palette" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="Kommandon">
        <div className="pal-input">
          <Icon name="search" />
          <input
            ref={input}
            value={q}
            placeholder="Skriv ett kommando …"
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
          {!list.length && <div className="muted pad">Inga träffar</div>}
        </div>
      </div>
    </div>
  );
}

const examples: { name: string; script: string }[] = [
  {
    name: "Konsolplatta (2D)",
    script: "model 2d\nmaterial steel E=210e9 nu=0.3\nplate x=0..2 y=0..0.2 t=0.01 size=0.025\nfix x=0\nload x=2 fy=-1000\n",
  },
  {
    name: "Kloss med tryck (3D)",
    script: "model 3d\nmaterial steel\nbox x=0..1 y=0..0.2 z=0..0.2 size=0.05\nfix x=0\nload x=1 fy=-5000\n",
  },
  {
    name: "Ram av balkar (3D)",
    script:
      "model 3d\nmaterial steel\nbeam (0,0,0) (0,3,0) (4,3,0) (4,0,0) section=ibeam(0.2,0.1,0.01,0.006) n=12\nbeam (0,3,0) (0,3,3) (4,3,3) (4,3,0) section=rect(0.1,0.1) n=10\nbeam (0,3,3) (0,0,3) section=rect(0.1,0.1) n=8\nbeam (4,3,3) (4,0,3) section=rect(0.1,0.1) n=8\nfix y=0\nload at=(2,3,0) fy=-20000\n",
  },
  {
    name: "Fackverk (2D)",
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
        <p className="muted">Bygg en modell från grunden, öppna en indatafil (.inp) eller ett projekt, lös och granska resultaten.</p>
        <div className="welcome-actions">
          <button type="button" className="btn primary big" onClick={() => newModel(blankModel[2])}>
            <Icon name="new" /> Ny 2D-modell
          </button>
          <button type="button" className="btn big" onClick={() => newModel(blankModel[3])}>
            <Icon name="cube" /> Ny 3D-modell
          </button>
          <button type="button" className="btn big" onClick={openDialog}>
            <Icon name="open" /> Öppna …
          </button>
        </div>
        <div className="welcome-cols">
          <div>
            <h4>Exempel</h4>
            {examples.map((e) => (
              <button key={e.name} type="button" className="listbtn" onClick={() => newModel(e.script)}>
                {e.name}
              </button>
            ))}
          </div>
          <div>
            <h4>Senaste</h4>
            {recent.length === 0 && <p className="muted">Inga filer än.</p>}
            {recent.map((p) => (
              <button key={p} type="button" className="listbtn" title={p} onClick={() => openPath(p)}>
                <span>{p.split(/[\\/]/).pop()}</span>
                <span className="muted ellipsis">{p}</span>
              </button>
            ))}
          </div>
        </div>
        <p className="muted small">
          <kbd>Ctrl</kbd>+<kbd>K</kbd> kommandon · <kbd>F5</kbd> lös · <kbd>F</kbd> anpassa vyn · <kbd>0</kbd>–<kbd>6</kbd> vyer
        </p>
      </div>
    </div>
  );
}
