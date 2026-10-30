// Local MCP server: start/stop and how Claude, Codex and others connect.

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
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

export function McpPanel() {
  const open = useStore((s) => s.mcpPanel);
  const st = useStore((s) => s.mcp);
  const [client, setClient] = useState<Client>("desktop");
  const [port, setPort] = useState("");
  const [network, setNetwork] = useState<boolean | null>(null);
  if (!open || !st) return null;
  const net = network ?? st.network;
  // other devices (e.g. Claude Code on a Mac talking to an iPad) use the network address
  const url = st.lan_url ?? st.url;
  const exe = st.exe;
  const quoted = exe.includes(" ") ? `"${exe}"` : exe;
  const close = () => set({ mcpPanel: false });
  const desktopJson = JSON.stringify({ mcpServers: { rustyfem: { command: exe, args: ["--mcp"] } } }, null, 2);
  return (
    <div className="overlay" onMouseDown={close}>
      <div className="dialog" onMouseDown={(e) => e.stopPropagation()} role="dialog" aria-label="MCP server">
        <header className="dialog-head">
          <div>
            <h2>Connect an AI assistant (MCP)</h2>
            <p className="muted">
              Claude or Codex can build, edit and solve the model that is open here. Everything runs locally on this computer; nothing is sent from here to any service.
            </p>
          </div>
          <button type="button" className="ibtn" aria-label="Close" onClick={close}>
            <Icon name="close" />
          </button>
        </header>

        <div className="mcp-status">
          <span className={`led${st.running ? " on" : ""}`} />
          <div className="grow">
            <b>{st.running ? "The server is running" : "The server is off"}</b>
            <div className="muted small">
              {st.running ? (
                <>
                  {url} · {st.calls} calls{st.last_tool ? ` · latest ${st.last_tool}` : ""}
                </>
              ) : (
                "Switch on to connect via HTTP. Claude Desktop, Claude Code and Codex start it themselves via the command below."
              )}
            </div>
            {st.error && <div className="small err">{st.error}</div>}
          </div>
          {!st.running && (
            <label className="check small" title="Listen on the local network so that other devices can connect (still requires the token)">
              <input type="checkbox" checked={net} onChange={(e) => setNetwork(e.target.checked)} />
              Allow from network
            </label>
          )}
          {!st.running && (
            <label className="field port">
              <span className="flabel">Port</span>
              <span className="finput">
                <input inputMode="numeric" placeholder={String(st.port)} value={port} onChange={(e) => setPort(e.target.value.replace(/\D/g, ""))} />
              </span>
            </label>
          )}
          <button type="button" className={`btn ${st.running ? "" : "primary"}`} onClick={() => mcpToggle(port ? Number(port) : null, network)}>
            {st.running ? "Stop" : "Start"}
          </button>
        </div>

        <Segmented
          value={client}
          onChange={setClient}
          options={[
            { value: "desktop", label: "Claude Desktop" },
            { value: "code", label: "Claude Code" },
            { value: "codex", label: "Codex / ChatGPT" },
            { value: "http", label: "Other (HTTP)" },
          ]}
        />

        <div className="mcp-body">
          {client === "desktop" && (
            <>
              <ol>
                <li>
                  Click <b>Add to Claude Desktop</b> (writes to <code>claude_desktop_config.json</code>, with a backup).
                </li>
                <li>Restart Claude Desktop. RustyFEM appears under tools (the hammer/plug icon).</li>
                <li>Ask Claude e.g. “build a 2 m cantilever beam, fixed at one end, 1 kN at the tip, and solve it”.</li>
              </ol>
              <div className="row-btns">
                <button type="button" className="btn primary" onClick={mcpInstallClaudeDesktop}>
                  <Icon name="plus" size={14} /> Add to Claude Desktop
                </button>
                {st.claude_desktop.length > 0 && (
                  <span className="ok small">
                    <Icon name="check" size={13} /> already added
                  </span>
                )}
              </div>
              <p className="muted small">Or add it by hand to the configuration file (Settings → Developer → Edit Config):</p>
              <Code text={desktopJson} />
            </>
          )}
          {client === "code" && (
            <>
              <p>Run in a terminal. Claude Code starts RustyFEM itself when needed:</p>
              <Code text={`claude mcp add rustyfem -- ${quoted} --mcp`} />
              <p className="muted small">Or via HTTP when the server is running:</p>
              <Code text={`claude mcp add --transport http rustyfem ${url} --header "Authorization: Bearer ${st.token}"`} />
            </>
          )}
          {client === "codex" && (
            <>
              <p>Codex (CLI and app) – run in a terminal:</p>
              <Code text={`codex mcp add rustyfem -- ${quoted} --mcp`} />
              <p className="muted small">
                Or add to <code>~/.codex/config.toml</code>:
              </p>
              <Code text={`[mcp_servers.rustyfem]\ncommand = '${exe}'\nargs = ["--mcp"]`} />
              <p className="muted small">
                ChatGPT app: if your version has MCP servers under its settings, add the command <code>{quoted}</code> with the argument <code>--mcp</code>. ChatGPT on the web cannot reach local servers.
              </p>
            </>
          )}
          {client === "http" && (
            <>
              <p>
                Streamable HTTP (JSON responses). The server must be started.{" "}
                {st.network ? "Reachable from other devices on the same network." : "Only from this device – tick “Allow from network” to connect from another computer."}
              </p>
              <Code text={`URL:    ${url}\nHeader: Authorization: Bearer ${st.token}`} />
              <p className="muted small">Clients that only start commands (stdio) use:</p>
              <Code text={`${quoted} --mcp`} />
            </>
          )}
        </div>
        <footer className="muted small">
          Tools: model_info, new_model, open_file, get_script, set_script, edit, undo, redo, solve, results, field_stats, node_info, element_info, save_project, rusty_help. All changes can be undone in the app.
        </footer>
      </div>
    </div>
  );
}
