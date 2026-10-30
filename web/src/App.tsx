import { useEffect } from "react";
import { runSelftest } from "./selftest";
import { McpPanel } from "./ui/McpPanel";
import { blankModel, cancel, init, listenEngine, newModel, openDialog, redo, save, solve, undo } from "./actions";
import { get, set, useStore } from "./store";
import { Dock } from "./ui/Dock";
import { Inspector } from "./ui/Inspector";
import { ModelTree } from "./ui/ModelTree";
import { ResultsPanel } from "./ui/ResultsPanel";
import { Palette, TopBar, Welcome } from "./ui/Shell";
import { Icon } from "./ui/ui";
import { Viewport } from "./ui/Viewport";

function useShortcuts() {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey;
      const inField = !!(e.target as HTMLElement).closest("input, textarea, select");
      const k = e.key.toLowerCase();
      if (mod && k === "k") {
        e.preventDefault();
        set((s) => ({ palette: !s.palette }));
      } else if (mod && k === "o") {
        e.preventDefault();
        openDialog();
      } else if (mod && k === "n") {
        e.preventDefault();
        newModel(blankModel[2]);
      } else if (mod && k === "s") {
        e.preventDefault();
        save(e.shiftKey);
      } else if (mod && !inField && k === "z" && !e.shiftKey) {
        e.preventDefault();
        undo();
      } else if (mod && !inField && (k === "y" || (k === "z" && e.shiftKey))) {
        e.preventDefault();
        redo();
      } else if (e.key === "F5") {
        e.preventDefault();
        solve();
      } else if (e.key === "Escape" && !inField) {
        if (get().run.running) cancel();
        else set({ pick: null, palette: false, mcpPanel: false, onNodePick: null });
      }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, []);
}

export function App() {
  const hasModel = useStore((s) => s.session !== null);
  const mode = useStore((s) => s.mode);
  const left = useStore((s) => s.left);
  const right = useStore((s) => s.right);
  const dock = useStore((s) => s.dock);
  const dark = useStore((s) => s.dark);
  const toast = useStore((s) => s.toast);
  useShortcuts();
  useEffect(() => {
    init().then(runSelftest);
    listenEngine();
  }, []);
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
  }, [dark]);
  return (
    <div className={`app${hasModel && left ? " has-left" : ""}${hasModel && right ? " has-right" : ""}${hasModel && dock ? " has-dock" : ""}`}>
      <TopBar />
      <main className="main">
        {hasModel && left && (
          <aside className="side left">
            <ModelTree />
          </aside>
        )}
        <div className="center">
          <Viewport />
          {!hasModel && <Welcome />}
        </div>
        {hasModel && right && (
          <aside className="side right">
            <div className="side-tabs">
              <button type="button" className={mode === "model" ? "on" : ""} onClick={() => set({ mode: "model" })}>
                Egenskaper
              </button>
              <button type="button" className={mode === "results" ? "on" : ""} onClick={() => set({ mode: "results" })}>
                Resultat
              </button>
              <button type="button" className="close-narrow" aria-label="Stäng" onClick={() => set({ right: false })}>
                <Icon name="close" size={14} />
              </button>
            </div>
            <div className="side-scroll">{mode === "model" ? <Inspector /> : <ResultsPanel />}</div>
          </aside>
        )}
        {hasModel && dock && <Dock />}
      </main>
      <Palette />
      <McpPanel />
      {toast && (
        <div className={`toast ${toast.kind}`} role="status" onClick={() => set({ toast: null })}>
          <Icon name={toast.kind === "error" ? "warn" : "check"} size={16} />
          <span>{toast.text}</span>
        </div>
      )}
    </div>
  );
}
