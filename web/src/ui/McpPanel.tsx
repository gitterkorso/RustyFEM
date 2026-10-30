// Lokal MCP-server: start/stopp och hur Claude, Codex och andra ansluter.

import { useState } from "react";
import { mcpInstallClaudeDesktop, mcpToggle } from "../actions";
import { set, useStore } from "../store";
import { Icon, Segmented } from "./ui";

type Client = "desktop" | "code" | "codex" | "http";

function Code({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="snippet">
      <pre>{text}</pre>
      <button
        type="button"
        className="btn small"
        onClick={() => {
          navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        <Icon name={copied ? "check" : "copy"} size={13} />
        {copied ? "Kopierat" : "Kopiera"}
      </button>
    </div>
  );
}

export function McpPanel() {
  const open = useStore((s) => s.mcpPanel);
  const st = useStore((s) => s.mcp);
  const [client, setClient] = useState<Client>("desktop");
  const [port, setPort] = useState("");
  if (!open || !st) return null;
  const exe = st.exe;
  const quoted = exe.includes(" ") ? `"${exe}"` : exe;
  const close = () => set({ mcpPanel: false });
  const desktopJson = JSON.stringify({ mcpServers: { rustyfem: { command: exe, args: ["--mcp"] } } }, null, 2);
  return (
    <div className="overlay" onMouseDown={close}>
      <div className="dialog" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="MCP-server">
        <header className="dialog-head">
          <div>
            <h2>Anslut AI-assistent (MCP)</h2>
            <p className="muted">
              Claude eller Codex kan bygga, ändra och lösa modellen som är öppen här. Allt körs lokalt på den här datorn; inget skickas härifrån till någon tjänst.
            </p>
          </div>
          <button type="button" className="ibtn" aria-label="Stäng" onClick={close}>
            <Icon name="close" />
          </button>
        </header>

        <div className="mcp-status">
          <span className={`led${st.running ? " on" : ""}`} />
          <div className="grow">
            <b>{st.running ? "Servern körs" : "Servern är avstängd"}</b>
            <div className="muted small">
              {st.running ? (
                <>
                  {st.url} · {st.calls} anrop{st.last_tool ? ` · senast ${st.last_tool}` : ""}
                </>
              ) : (
                "Slå på för att ansluta via HTTP. Claude Desktop, Claude Code och Codex startar den själva via kommandot nedan."
              )}
            </div>
            {st.error && <div className="small err">{st.error}</div>}
          </div>
          {!st.running && (
            <label className="field port">
              <span className="flabel">Port</span>
              <span className="finput">
                <input inputMode="numeric" placeholder={String(st.port)} value={port} onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))} />
              </span>
            </label>
          )}
          <button type="button" className={`btn ${st.running ? "" : "primary"}`} onClick={() => mcpToggle(port ? Number(port) : null)}>
            {st.running ? "Stoppa" : "Starta"}
          </button>
        </div>

        <Segmented
          value={client}
          onChange={setClient}
          options={[
            { value: "desktop", label: "Claude Desktop" },
            { value: "code", label: "Claude Code" },
            { value: "codex", label: "Codex / ChatGPT" },
            { value: "http", label: "Annan (HTTP)" },
          ]}
        />

        <div className="mcp-body">
          {client === "desktop" && (
            <>
              <ol>
                <li>
                  Klicka <b>Lägg till i Claude Desktop</b> (skriver i <code>claude_desktop_config.json</code>, med en säkerhetskopia).
                </li>
                <li>Starta om Claude Desktop. RustyFEM syns under verktyg (hammaren/kontakten).</li>
                <li>Be Claude t.ex. ”bygg en konsolbalk 2 m, fast inspänd, 1 kN i änden, och lös den”.</li>
              </ol>
              <div className="row-btns">
                <button type="button" className="btn primary" onClick={mcpInstallClaudeDesktop}>
                  <Icon name="plus" size={14} /> Lägg till i Claude Desktop
                </button>
                {st.claude_desktop.length > 0 && (
                  <span className="ok small">
                    <Icon name="check" size={13} /> redan tillagd
                  </span>
                )}
              </div>
              <p className="muted small">Eller lägg till för hand i konfigurationsfilen (Inställningar → Utvecklare → Redigera konfiguration):</p>
              <Code text={desktopJson} />
            </>
          )}
          {client === "code" && (
            <>
              <p>Kör i en terminal. Claude Code startar RustyFEM själv vid behov:</p>
              <Code text={`claude mcp add rustyfem -- ${quoted} --mcp`} />
              <p className="muted small">Eller via HTTP när servern körs:</p>
              <Code text={`claude mcp add --transport http rustyfem ${st.url} --header "Authorization: Bearer ${st.token}"`} />
            </>
          )}
          {client === "codex" && (
            <>
              <p>Codex (CLI och app) – kör i en terminal:</p>
              <Code text={`codex mcp add rustyfem -- ${quoted} --mcp`} />
              <p className="muted small">
                Eller lägg till i <code>~/.codex/config.toml</code>:
              </p>
              <Code text={`[mcp_servers.rustyfem]\ncommand = '${exe}'\nargs = ["--mcp"]`} />
              <p className="muted small">
                ChatGPT-appen: om din version har MCP-servrar under inställningarna, lägg till kommandot <code>{quoted}</code> med argumentet <code>--mcp</code>. ChatGPT på webben kan inte nå lokala servrar.
              </p>
            </>
          )}
          {client === "http" && (
            <>
              <p>Streamable HTTP (JSON-svar), bara från den här datorn. Servern måste vara startad.</p>
              <Code text={`URL:    ${st.url}\nHeader: Authorization: Bearer ${st.token}`} />
              <p className="muted small">Klienter som bara startar kommandon (stdio) använder:</p>
              <Code text={`${quoted} --mcp`} />
            </>
          )}
        </div>
        <footer className="muted small">
          Verktyg: model_info, new_model, open_file, get_script, set_script, edit, undo, redo, solve, results, field_stats, node_info, element_info, save_project, rusty_help. Alla ändringar kan ångras i appen.
        </footer>
      </div>
    </div>
  );
}
